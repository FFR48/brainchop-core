#!/usr/bin/env python3
"""Build a verified BrainChop HIP artifact for one model and AMD architecture.

The expensive stages are content-addressed and resumable.  This is maintainer
tooling: its output is consumed by the small production runtime, but none of
PyTorch, MIGraphX, tinygrad, or a compiler is needed by normal users.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from export_hip_onnx import model_spec  # noqa: E402
from hip_memory import check_memory  # noqa: E402
from hip_artifacts import (ABI_VERSION, SCHEMA_VERSION, code_object_metadata,  # noqa: E402
                           digest, verify)

# PR #19's network pipeline cache key. PVE adds a separately keyed support
# compile, so unchanged ONNX/MIGraphX/tinygrad network stages remain reusable.
NETWORK_PIPELINE_CACHE_VERSION = \
    "84d434c81a6f526cde2e396122cf51732be39505544caf05006f1d2641ed3963"


def public_spec(spec):
    return {key: value for key, value in spec.items()
            if key not in ("directory", "dilations", "gelu")}


def file_digest(path):
    return digest(Path(path).read_bytes())


def command_plan(args, spec, work):
    onnx = work / "model.onnx"
    compiled = work / "model.mxr"
    migraphx = work / "migraphx"
    tinygrad = work / "tinygrad"
    support = work / "hip-support"
    classify = work / "hip-classify"
    export = [sys.executable, str(SCRIPTS / "export_hip_onnx.py"),
              "--model", spec["name"], "--output", str(onnx),
              "--trace-dim", str(args.trace_dim), "--opset", str(args.opset)]
    compile_command = [args.migraphx_driver, "compile", str(onnx), "--onnx",
                       "--input-dim", "@volume", "1", "1",
                       str(spec["dimension"]), str(spec["dimension"]),
                       str(spec["dimension"]), "--gpu"]
    if not args.native_target:
        compile_command += ["--gpu-arch", args.gfx,
                            "--gpu-num-cus", str(args.num_compute_units),
                            "--gpu-num-chiplets", str(args.num_chiplets)]
    compile_command.append("--fp16")
    if args.exhaustive_tune:
        compile_command.append("--exhaustive-tune")
    if args.backend_options:
        compile_command += ["--backend-options", args.backend_options]
    compile_command += ["--binary", "--output", str(compiled)]
    extract_migraphx = [
        sys.executable, str(SCRIPTS / "extract_hip_kernels.py"), "migraphx",
        "--compiled", str(compiled), "--output", str(migraphx),
        "--gfx", args.gfx]
    tinygrad_commands = []
    for graph in (("widen", "moments", "norm_gelu", "argmax") if spec["supports_pve"]
                  else ("widen", "moments", "norm_gelu")):
        command = [
            sys.executable, str(SCRIPTS / "extract_hip_kernels.py"), "tinygrad",
            "--graph", graph, "--output", str(tinygrad / graph),
            "--gfx", args.gfx, "--channels", str(spec["channels"]),
            "--classes", str(spec["classes"]), "--dimension",
            str(spec["dimension"]), "--gelu", spec["gelu"]]
        if spec["has_affine"]:
            command.append("--affine")
        tinygrad_commands.append(command)
    support_command = None
    if spec["supports_pve"]:
        support_command = [
            sys.executable, str(SCRIPTS / "compile_hip_support.py"),
            "--hipcc", args.hipcc,
            "--source", str(SCRIPTS / "hip_tissue_priors.hip"),
            "--output", str(support), "--gfx", args.gfx,
            "--dimension", str(spec["dimension"]),
            "--include", str(REPO / "src"),
        ]
    classify_command = None
    if not spec["supports_pve"]:
        classify_command = [
            sys.executable, str(SCRIPTS / "compile_hip_support.py"),
            "--hipcc", args.hipcc, "--source", str(SCRIPTS / "hip_classify.hip"),
            "--output", str(classify), "--gfx", args.gfx,
            "--dimension", str(spec["dimension"]), "--channels", str(spec["channels"]),
            "--classes", str(spec["classes"]), "--include", str(REPO / "src")]
        if spec["classifier_bias"]:
            classify_command.append("--classifier-bias")
    return {
        "classify": classify, "compile_classify": classify_command,
        "onnx": onnx, "compiled": compiled, "migraphx": migraphx,
        "tinygrad": tinygrad, "support": support,
        "export": export, "compile": compile_command,
        "extract_migraphx": extract_migraphx,
        "extract_tinygrad": tinygrad_commands,
        "compile_support": support_command,
    }


def _stage_key(name, inputs):
    payload = json.dumps({"stage": name, **inputs}, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_stage(name, command, outputs, inputs, work, env=None, dry_run=False,
              timeout=None):
    key = _stage_key(name, inputs)
    stamp = work / f".{name}.json"
    if stamp.is_file() and all(Path(path).exists() for path in outputs):
        try:
            if json.loads(stamp.read_text()).get("key") == key:
                print(f"[{name}] cached")
                return
        except (OSError, ValueError, TypeError):
            pass
    print("[{}] {}".format(name, " ".join(command)), flush=True)
    if dry_run:
        return
    for output in outputs:
        path = Path(output)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    started = time.monotonic()
    try:
        subprocess.run(command, check=True, cwd=REPO, env=env,
                       timeout=timeout or None)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"[{name}] exceeded {timeout}s; completed MIGraphX tuning results "
            "remain in the problem cache and the same command can resume"
        ) from error
    print(f"[{name}] completed in {time.monotonic() - started:.1f}s", flush=True)
    if not all(Path(path).exists() for path in outputs):
        raise RuntimeError(f"stage {name} did not create all declared outputs")
    stamp.write_text(json.dumps({"key": key, "command": command}, indent=2) + "\n")


class Arena:
    def __init__(self, alignment=4096):
        self.alignment = alignment
        self.cursor = 0

    def place(self, size):
        self.cursor = -(-self.cursor // self.alignment) * self.alignment
        offset = self.cursor
        self.cursor += size
        return offset


def _read_manifest(path):
    path = Path(path)
    return json.loads((path / "manifest.json").read_text()), path


def _kernel_groups(migraphx_directory, hidden_layers, classifier_bias):
    entries, directory = _read_manifest(migraphx_directory)
    by_symbol = {}
    for entry in entries:
        by_symbol.setdefault(entry["symbol"], []).append(entry)
    layer0 = by_symbol.get("mlir_convert_convolution_reshape_convert", [])
    hidden = by_symbol.get("mlir_convolution_reshape_convert", [])
    head_symbol = ("mlir_convolution_broadcast_add_convert" if classifier_bias
                   else "mlir_convolution_convert")
    head = by_symbol.get(head_symbol, [])
    if len(layer0) != 1 or len(hidden) != hidden_layers - 1 or len(head) != 1:
        counts = {name: len(values) for name, values in by_symbol.items()
                  if "convolution" in name}
        raise ValueError("unexpected MIGraphX convolution extraction: " +
                         json.dumps(counts, sort_keys=True))
    return layer0[0], hidden, head[0], directory


def _argument(buffer, offset, size, access):
    return {"kind": "buffer", "buffer": buffer, "offset": offset,
            "bytes": size, "access": access}


def render_catalog(manifest):
    """Render the C-only runtime view of a verified manifest.

    The binary code objects stay in kernels.bin.  This small derived header is
    only offsets and launch metadata, so an ordinary build needs no JSON parser
    and no Python.
    """
    kernel_index = {item["id"]: index
                    for index, item in enumerate(manifest["kernels"])}
    buffer_index = {item["id"]: index
                    for index, item in enumerate(manifest["buffers"])}
    if buffer_index != {"arena": 0, "input": 1, "output": 2}:
        raise ValueError("runtime catalog requires arena, input, output buffer order")
    lines = [
        "/* Generated by scripts/build_hip_artifacts.py; do not edit. */",
        "#ifndef BRAINCHOPC_HIP_ARTIFACT_CATALOG_H",
        "#define BRAINCHOPC_HIP_ARTIFACT_CATALOG_H",
        "",
        "static const bc_hip_kernel bc_hip_artifact_kernels[] = {",
    ]
    for item in manifest["kernels"]:
        lines.append(f"\t{{{item['offset']}ULL, {item['size']}ULL, "
                     f"{json.dumps(item['entry'])}}},")
    lines += ["};", "", "static const bc_hip_argument bc_hip_artifact_args[] = {"]
    first_arguments = []
    argument_count = 0
    schema = manifest["schema_version"]
    for dispatch in manifest["dispatches"]:
        first_arguments.append(argument_count)
        for item in dispatch["args"]:
            if item["kind"] == "buffer":
                lines.append(f"\t{{BC_HIP_ARG_BUFFER, {buffer_index[item['buffer']]}, "
                             f"{item['offset']}ULL, {item['bytes']}ULL, 0ULL}},")
            else:
                lines.append(f"\t{{BC_HIP_ARG_SCALAR, 0, 0ULL, 0ULL, "
                             f"{item['value']}ULL}},")
            argument_count += 1
    lines += ["};", "", "static const bc_hip_dispatch bc_hip_artifact_dispatches[] = {"]
    for start, item in zip(first_arguments, manifest["dispatches"]):
        grid = ", ".join(str(value) for value in item["grid"])
        block = ", ".join(str(value) for value in item["block"])
        mode = {"always": "BC_HIP_DISPATCH_ALWAYS",
                "labels": "BC_HIP_DISPATCH_LABELS",
                "pve": "BC_HIP_DISPATCH_PVE"}.get(item.get("mode", "always"))
        suffix = f", {mode}" if schema >= 2 else ""
        lines.append(f"\t{{{kernel_index[item['kernel']]}, {start}, "
                     f"{len(item['args'])}, {{{grid}}}, {{{block}}}, "
                     f"{item['dynamic_lds_bytes']}{suffix}}},")
    lines += ["};", "", "static const bc_hip_weight bc_hip_artifact_weights[] = {"]
    dtype = {"f16": "BC_HIP_F16", "f32": "BC_HIP_F32"}
    for item in manifest["weights"]:
        if item["target_dtype"] not in dtype:
            raise ValueError(f"runtime cannot upload {item['target_dtype']} weights")
        lines.append(f"\t{{{item['offset']}ULL, {item['bytes']}ULL, "
                     f"{dtype[item['target_dtype']]} }},")
    model = manifest["model"]
    buffers = {item["id"]: item for item in manifest["buffers"]}
    lines += [
        "};", "",
        "static const bc_hip_profile bc_hip_artifact_profile = {",
        f"\t{json.dumps(model['name'])},",
        f"\t{json.dumps(manifest['target']['architecture'])},",
        f"\t{json.dumps(model['descriptor_sha256'])},",
        f"\t{model['dimension']}, {model['channels']}, {model['hidden_layers']}, "
        f"{model['classes']}, {int(model['has_affine'])}, "
        f"{int(model['classifier_bias'])},",
        f"\t{buffers['arena']['bytes']}ULL, {buffers['input']['bytes']}ULL, "
        f"{buffers['output']['bytes']}ULL,",
        "\tbc_hip_artifact_kernels, sizeof(bc_hip_artifact_kernels) / "
        "sizeof(bc_hip_artifact_kernels[0]),",
        "\tbc_hip_artifact_args, sizeof(bc_hip_artifact_args) / "
        "sizeof(bc_hip_artifact_args[0]),",
        "\tbc_hip_artifact_dispatches, sizeof(bc_hip_artifact_dispatches) / "
        "sizeof(bc_hip_artifact_dispatches[0]),",
        "\tbc_hip_artifact_weights, sizeof(bc_hip_artifact_weights) / "
        "sizeof(bc_hip_artifact_weights[0])" + ("," if schema >= 2 else ""),
    ]
    if schema >= 2:
        lines += [
            f"\t{int(model['supports_pve'])},",
            f"\t{buffers['output']['bytes'] if model['supports_pve'] else 0}ULL",
        ]
    lines += ["};", "", "#endif", ""]
    return "\n".join(lines)


def verify_promoted(directory):
    manifest = verify(directory)
    for mode in (("labels", "pve") if manifest["model"].get("supports_pve") else ("labels",)):
        check_memory(manifest, mode)
    return manifest


def pack_artifact(spec, gfx, num_compute_units, num_chiplets, wavefront_size,
                  migraphx_directory, tinygrad_directory, support_directory,
                  output, toolchain, compile_options, classify_directory=None):
    if not spec["supports_pve"] and classify_directory is None:
        raise ValueError("labels-only packing requires an offline classifier object")
    dimension = spec["dimension"]
    voxels = dimension ** 3
    channels = spec["channels"]
    classes = spec["classes"]
    hidden_layers = spec["hidden_layers"]
    layer0, hidden, head, migraphx_root = _kernel_groups(
        migraphx_directory, hidden_layers, spec["classifier_bias"])
    tg = Path(tinygrad_directory)
    graphs = {}
    for name in (("widen", "moments", "norm_gelu", "argmax") if spec["supports_pve"]
                 else ("widen", "moments", "norm_gelu")):
        entries, directory = _read_manifest(tg / name)
        if not entries:
            raise ValueError(f"tinygrad {name} extraction is empty")
        graphs[name] = (entries, directory)
    if len(graphs["widen"][0]) != 1 or len(graphs["norm_gelu"][0]) != 1:
        raise ValueError("widen and norm_gelu must each compile to one kernel")
    norm_roles = {arg["role"] for arg in graphs["norm_gelu"][0][0]["args"]}
    if ("scale" in norm_roles) != spec["has_affine"]:
        raise ValueError("norm_gelu affine arguments do not match the model")

    arena = Arena()
    offsets = {
        "input_f32": arena.place(voxels * 4),
        "raw": arena.place(channels * voxels * 2),
        "activation": arena.place(channels * voxels * 2),
        "sink": arena.place(channels * voxels * 4),
    }
    scratch = {}
    for graph_name in (("moments", "argmax") if spec["supports_pve"] else ("moments",)):
        sizes = {}
        for entry in graphs[graph_name][0]:
            for arg in entry["args"]:
                if arg["role"] not in ("x", "out", "logits"):
                    prior = sizes.setdefault(arg["role"], arg["bytes"])
                    if prior != arg["bytes"]:
                        raise ValueError(f"{graph_name} scratch size is inconsistent")
        scratch[graph_name] = {name: (arena.place(size), size)
                               for name, size in sizes.items()}
    offsets["moments"] = arena.place(2 * channels * 4)
    if spec["has_affine"]:
        offsets["scale"] = arena.place(hidden_layers * channels * 4)
        offsets["bias"] = arena.place(hidden_layers * channels * 4)
    convolution_weights = [arena.place(27 * (channels if index else 1) *
                                      channels * 2)
                           for index in range(hidden_layers)]
    offsets["head_weight"] = arena.place(channels * classes * 2)
    if spec["classifier_bias"]:
        offsets["head_bias"] = arena.place(classes * 2)
    if spec["supports_pve"]:
        offsets["logits"] = arena.place(classes * voxels * 4)

    bundle = bytearray()
    kernel_items, kernel_ids = [], {}

    def add_kernel(directory, entry, producer):
        blob = (directory / entry["file"]).read_bytes()
        metadata = code_object_metadata(blob)
        key = (digest(blob), metadata["entry"])
        if key in kernel_ids:
            return kernel_ids[key]
        while len(bundle) % 64:
            bundle.append(0)
        offset = len(bundle)
        bundle.extend(blob)
        kernel_id = f"{producer}.{len(kernel_items)}"
        kernel_ids[key] = kernel_id
        kernel_items.append({
            "id": kernel_id, "offset": offset, "size": len(blob),
            "sha256": digest(blob), "entry": metadata["entry"],
            "symbol": metadata["symbol"], "producer": producer,
            "target": metadata["target"],
            "kernarg_segment_size": metadata["kernarg_segment_size"],
            "static_lds_bytes": metadata["static_lds_bytes"],
            "arguments": metadata["arguments"],
        })
        return kernel_id

    dispatches = []

    def emit(directory, entry, producer, arguments, dynamic_lds=0,
             mode="always"):
        dispatches.append({
            "kernel": add_kernel(directory, entry, producer),
            "grid": entry["grid"], "block": entry["block"],
            "dynamic_lds_bytes": dynamic_lds, "mode": mode,
            "args": arguments,
        })

    def tinygrad_arguments(entry, mapping):
        result = []
        for slot in entry["args"]:
            if slot["role"] not in mapping:
                raise ValueError(f"nothing supplies tinygrad role {slot['role']}")
            buffer, offset, access = mapping[slot["role"]]
            result.append(_argument(buffer, offset, slot["bytes"], access))
        return result

    widen, widen_dir = graphs["widen"]
    emit(widen_dir, widen[0], "tinygrad", tinygrad_arguments(widen[0], {
        "x": ("input", 0, "read"),
        "out": ("arena", offsets["input_f32"], "write"),
    }))

    raw_bytes = channels * voxels * 2
    activation_bytes = channels * voxels * 2
    sink_bytes = channels * voxels * 4
    moments, moments_dir = graphs["moments"]
    norm, norm_dir = graphs["norm_gelu"]
    for index in range(hidden_layers):
        convolution = layer0 if index == 0 else hidden[index - 1]
        source_offset = offsets["input_f32"] if index == 0 else offsets["activation"]
        source_bytes = voxels * 4 if index == 0 else activation_bytes
        weight_bytes = 27 * (channels if index else 1) * channels * 2
        emit(migraphx_root, convolution, "rocmlir", [
            _argument("arena", source_offset, source_bytes, "read"),
            _argument("arena", convolution_weights[index], weight_bytes, "read"),
            _argument("arena", offsets["sink"], sink_bytes, "write"),
            _argument("arena", offsets["raw"], raw_bytes, "write"),
        ])
        for entry in moments:
            mapping = {
                "x": ("arena", offsets["raw"], "read"),
                "out": ("arena", offsets["moments"], "write"),
            }
            mapping.update({role: ("arena", value[0], "readwrite")
                            for role, value in scratch["moments"].items()})
            emit(moments_dir, entry, "tinygrad",
                 tinygrad_arguments(entry, mapping), entry["dynamic_lds_bytes"])
        mapping = {
            "x": ("arena", offsets["raw"], "read"),
            "out": ("arena", offsets["activation"], "write"),
            "mean": ("arena", offsets["moments"], "read"),
            "inv": ("arena", offsets["moments"] + channels * 4, "read"),
        }
        if spec["has_affine"]:
            mapping["scale"] = ("arena", offsets["scale"] + index * channels * 4,
                                "read")
            mapping["bias"] = ("arena", offsets["bias"] + index * channels * 4,
                               "read")
        emit(norm_dir, norm[0], "tinygrad", tinygrad_arguments(norm[0], mapping),
             norm[0]["dynamic_lds_bytes"])

    if not spec["supports_pve"]:
        classify, classify_dir = _read_manifest(classify_directory)
        if len(classify) != 1:
            raise ValueError("HIP classifier must compile to one kernel")
        mapping = {
            "activation": ("arena", offsets["activation"], "read"),
            "weight": ("arena", offsets["head_weight"], "read"),
            "out": ("output", 0, "write"),
        }
        if spec["classifier_bias"]:
            mapping["bias"] = ("arena", offsets["head_bias"], "read")
        emit(classify_dir, classify[0], "hipcc",
             tinygrad_arguments(classify[0], mapping), mode="labels")
        compile_options = {**compile_options, "bounded_head": head_contract(spec)}
    else:
        head_arguments = [
            _argument("arena", offsets["activation"], activation_bytes, "read"),
            _argument("arena", offsets["head_weight"], channels * classes * 2, "read"),
        ]
        if spec["classifier_bias"]:
            head_arguments.append(_argument("arena", offsets["head_bias"],
                                           classes * 2, "read"))
        head_arguments.append(_argument("arena", offsets["logits"],
                                        classes * voxels * 4, "write"))
        emit(migraphx_root, head, "rocmlir", head_arguments)
        argmax, argmax_dir = graphs["argmax"]
        for entry in argmax:
            mapping = {
                "logits": ("arena", offsets["logits"], "read"),
                "out": ("output", 0, "write"),
            }
            mapping.update({role: ("arena", value[0], "readwrite")
                            for role, value in scratch["argmax"].items()})
            emit(argmax_dir, entry, "tinygrad", tinygrad_arguments(entry, mapping),
                 entry["dynamic_lds_bytes"], mode="labels")

    if spec["supports_pve"]:
        support, support_dir = _read_manifest(support_directory)
        if len(support) != 1:
            raise ValueError("HIP tissue-prior support must compile to one kernel")
        emit(support_dir, support[0], "hipcc", tinygrad_arguments(support[0], {
            "logits": ("arena", offsets["logits"], "read"),
            "out": ("output", 0, "write"),
        }), support[0]["dynamic_lds_bytes"], mode="pve")

    weights = []
    for index in range(hidden_layers):
        source_name = f"conv3d_{index}_gn.kernel"
        size = 27 * (channels if index else 1) * channels * 2
        weights.append({"tensor": source_name, "buffer": "arena",
                        "offset": convolution_weights[index], "bytes": size,
                        "source_dtype": "f32", "target_dtype": "f16",
                        "layout": "tfjs-kx-ky-kz-in-out_to_hip-out-in-kz-ky-kx"})
        if spec["has_affine"]:
            weights.append({"tensor": f"affine_{index}.scale", "buffer": "arena",
                            "offset": offsets["scale"] + index * channels * 4,
                            "bytes": channels * 4, "source_dtype": "f32",
                            "target_dtype": "f32", "layout": "diagonal"})
            weights.append({"tensor": f"affine_{index}.bias", "buffer": "arena",
                            "offset": offsets["bias"] + index * channels * 4,
                            "bytes": channels * 4, "source_dtype": "f32",
                            "target_dtype": "f32", "layout": "contiguous"})
    head_name = "conv3d_final" if spec["has_affine"] else "output"
    weights.append({"tensor": f"{head_name}.kernel", "buffer": "arena",
                    "offset": offsets["head_weight"],
                    "bytes": channels * classes * 2, "source_dtype": "f32",
                    "target_dtype": "f16", "layout": "tfjs-in-out_to_hip-out-in"})
    if spec["classifier_bias"]:
        weights.append({"tensor": f"{head_name}.bias", "buffer": "arena",
                        "offset": offsets["head_bias"], "bytes": classes * 2,
                        "source_dtype": "f32", "target_dtype": "f16",
                        "layout": "contiguous"})

    sources = artifact_sources(spec)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / "kernels.bin"
    bundle_path.write_bytes(bundle)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "abi_version": ABI_VERSION,
        "model": public_spec(spec),
        "target": {"architecture": gfx,
                   "num_compute_units": num_compute_units,
                   "num_chiplets": num_chiplets,
                   "wavefront_size": wavefront_size},
        "toolchain": {**toolchain, "compile_options": compile_options},
        "sources": sources,
        "bundle": {"file": "kernels.bin", "size": len(bundle),
                   "sha256": digest(bundle)},
        "buffers": [
            {"id": "arena", "bytes": arena.cursor, "alignment": 4096,
             "kind": "arena"},
            {"id": "input", "bytes": voxels * 2, "alignment": 64,
             "kind": "input"},
            {"id": "output", "bytes": (4 * voxels * 4
                                         if spec["supports_pve"] else voxels),
             "alignment": 64,
             "kind": "output"},
        ],
        "kernels": kernel_items,
        "dispatches": dispatches,
        "weights": weights,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_promoted(output)
    (output / "catalog.h").write_text(render_catalog(manifest))
    return manifest


def _version(command, fallback):
    try:
        result = subprocess.run(command, check=True, text=True,
                                capture_output=True, timeout=30)
        text = (result.stdout or result.stderr).strip().splitlines()[0]
        return text or fallback
    except (OSError, subprocess.SubprocessError, IndexError):
        return fallback


def artifact_sources(spec):
    paths = [
        SCRIPTS / "build_hip_artifacts.py",
        SCRIPTS / "export_hip_onnx.py",
        SCRIPTS / "extract_hip_kernels.py",
        SCRIPTS / "hip_artifacts.py",
        SCRIPTS / "hip_memory.py",
        SCRIPTS / "gen_model_capabilities.py",
        REPO / "src" / "model_capabilities.json",
        REPO / "src" / "model_capabilities.h",
        spec["directory"] / "model_def.h",
        spec["directory"] / "model.json",
        spec["directory"] / "model_meta.json",
    ]
    if not spec["supports_pve"]:
        paths += [SCRIPTS / "compile_hip_support.py", SCRIPTS / "hip_classify.hip"]
    if spec["supports_pve"]:
        paths += [SCRIPTS / "compile_hip_support.py",
                  SCRIPTS / "hip_tissue_priors.hip",
                  REPO / "src" / "pve_constants.h"]
    return {str(path.relative_to(REPO)): file_digest(path) for path in paths}


def build_support(args, spec, work, plan):
    if not plan["compile_support"]:
        return
    run_stage("hip-support", plan["compile_support"],
              [plan["support"] / "manifest.json",
               plan["support"] / "tissue_priors.hsaco"],
              {"model": public_spec(spec), "gfx": args.gfx,
               "compiler": file_digest(SCRIPTS / "compile_hip_support.py"),
               "source": file_digest(SCRIPTS / "hip_tissue_priors.hip"),
               "constants": file_digest(REPO / "src" / "pve_constants.h"),
               "hipcc": args.hipcc},
              work, dry_run=args.dry_run, timeout=args.support_timeout)


def head_contract(spec):
    return {"layout": "NCDHW", "weight_layout": "class-channel",
            "accumulation": "f32", "workspace_bytes": 0,
            "channels": spec["channels"], "classes": spec["classes"],
            "classifier_bias": spec["classifier_bias"]}


def build_classify(args, spec, work, plan):
    if not plan["compile_classify"]:
        return
    run_stage("hip-classify", plan["compile_classify"],
              [plan["classify"] / "manifest.json", plan["classify"] / "classify.hsaco"],
              {"model": public_spec(spec), "gfx": args.gfx,
               "command": plan["compile_classify"],
               "compiler": file_digest(SCRIPTS / "compile_hip_support.py"),
               "source": file_digest(SCRIPTS / "hip_classify.hip"),
               "hipcc_version": ("dry-run" if args.dry_run else
                                 _version([args.hipcc, "--version"], "unknown"))},
              work, dry_run=args.dry_run, timeout=args.support_timeout)


def replace_label_tail(spec, source, source_root, classify_directory, output, args):
    """Repack a verified full-head artifact, preserving body code objects verbatim.

    Only generated contiguous NCDHW pipelines with the known weight transform
    qualify. Compact the UNION of surviving arena ranges so overlapping moment
    views stay aliased while removed argmax scratch leaves no dead allocation.
    """
    source = Path(source)
    manifest = verify(source, source_root)
    expected = public_spec(spec)
    actual = dict(manifest["model"])
    actual.setdefault("supports_pve", False)
    if actual != expected or spec["supports_pve"]:
        raise ValueError("reused network must match a labels-only model")
    target = dict(architecture=args.gfx, num_compute_units=args.num_compute_units,
                  num_chiplets=args.num_chiplets, wavefront_size=args.wavefront_size)
    if manifest["target"] != target:
        raise ValueError("reused network target mismatch")
    voxels = spec["dimension"] ** 3
    channels, classes = spec["channels"], spec["classes"]
    head_weight = next(w for w in manifest["weights"]
                       if w["layout"] == "tfjs-in-out_to_hip-out-in")
    if head_weight["target_dtype"] != "f16" or head_weight["bytes"] != channels * classes * 2:
        raise ValueError("unsupported classifier weights")
    kernels = {k["id"]: k for k in manifest["kernels"]}
    heads = [i for i, d in enumerate(manifest["dispatches"])
             if kernels[d["kernel"]]["entry"] in
             ("mlir_convolution_broadcast_add_convert", "mlir_convolution_convert")]
    if len(heads) != 1:
        raise ValueError("cannot identify unique full classifier boundary")
    boundary = heads[0]
    original_head = manifest["dispatches"][boundary]
    roles = ["activation", "weight"] + (["bias"] if spec["classifier_bias"] else [])
    if len(original_head["args"]) != len(roles) + 1:
        raise ValueError("unsupported full classifier arguments")
    mapping = dict(zip(roles, original_head["args"]))
    activation = mapping["activation"]
    allowed_read = ("read", "readwrite") if manifest["schema_version"] == 1 else ("read",)
    allowed_write = ("write", "readwrite") if manifest["schema_version"] == 1 else ("write",)
    if activation["bytes"] != channels * voxels * 2 or activation["access"] not in allowed_read:
        raise ValueError("classifier activation is not contiguous fp16 NCDHW")
    producer = manifest["dispatches"][boundary - 1]
    if kernels[producer["kernel"]]["producer"] != "tinygrad" or not any(
            a.get("buffer") == activation["buffer"] and a.get("offset") == activation["offset"]
            and a.get("bytes") == activation["bytes"] and a.get("access") in allowed_write
            for a in producer["args"]):
        raise ValueError("classifier must consume the final norm/GELU output")
    if mapping["weight"]["offset"] != head_weight["offset"]:
        raise ValueError("classifier weight argument does not match upload")
    for a in producer["args"]:
        if (a.get("buffer") == activation["buffer"] and a.get("offset") == activation["offset"]
                and a.get("bytes") == activation["bytes"]):
            a["access"] = "write"
    for r in roles:
        mapping[r]["access"] = "read"
    manifest["dispatches"] = manifest["dispatches"][:boundary]
    for d in manifest["dispatches"]:
        d["mode"] = "always"
    entries, root = _read_manifest(classify_directory)
    if len(entries) != 1:
        raise ValueError("classifier must compile to one kernel")
    entry = entries[0]
    mapping["out"] = _argument("output", 0, voxels, "write")
    wanted_sizes = [channels * voxels * 2, channels * classes * 2] + (
        [classes * 2] if spec["classifier_bias"] else []) + [voxels]
    if [a["role"] for a in entry["args"]] != roles + ["out"] or \
            [a["bytes"] for a in entry["args"]] != wanted_sizes:
        raise ValueError("compiled classifier shape/bias mismatch")
    old_blob = (source / manifest["bundle"]["file"]).read_bytes()
    used = {d["kernel"] for d in manifest["dispatches"]}
    bundle = bytearray()
    items = []
    for k in manifest["kernels"]:
        if k["id"] not in used:
            continue
        part = old_blob[k["offset"]:k["offset"] + k["size"]]
        while len(bundle) % 64:
            bundle.append(0)
        k["offset"] = len(bundle)
        bundle.extend(part)
        items.append(k)
    part = (root / entry["file"]).read_bytes()
    meta = code_object_metadata(part)
    if meta["entry"] != "bc_hip_classify":
        raise ValueError("unexpected classifier entry")
    while len(bundle) % 64:
        bundle.append(0)
    kernel_id = "hipcc.classify"
    items.append({"id": kernel_id, "offset": len(bundle), "size": len(part),
                  "sha256": digest(part), "producer": "hipcc", **meta})
    bundle.extend(part)
    manifest["kernels"] = items
    manifest["dispatches"].append({
        "kernel": kernel_id, "grid": entry["grid"], "block": entry["block"],
        "dynamic_lds_bytes": entry["dynamic_lds_bytes"], "mode": "labels",
        "args": [mapping[a["role"]] for a in entry["args"]]})
    # Relocate all live ranges, including upload-only weights and overlapping
    # reduction views. Alignment is preserved, not inferred from lifetimes.
    views = [a for d in manifest["dispatches"] for a in d["args"]
             if a["kind"] == "buffer" and a["buffer"] == "arena"] + manifest["weights"]
    intervals = sorted((a["offset"], a["offset"] + a["bytes"]) for a in views)
    groups = []
    for lo, hi in intervals:
        if groups and lo <= groups[-1][1]:
            groups[-1][1] = max(hi, groups[-1][1])
        else:
            groups.append([lo, hi])
    arena = Arena()
    moves = [(lo, hi, arena.place(hi - lo)) for lo, hi in groups]
    # mapping shares dictionaries with the original dispatch; update each once.
    for a in {id(a): a for a in views}.values():
        lo, hi, new = next(g for g in moves if g[0] <= a["offset"] < g[1])
        a["offset"] += new - lo
    manifest["buffers"][0]["bytes"] = arena.cursor
    manifest.update(schema_version=SCHEMA_VERSION, abi_version=ABI_VERSION,
                    model=public_spec(spec), sources=artifact_sources(spec))
    manifest["toolchain"]["compile_options"]["bounded_head"] = head_contract(spec)
    manifest["toolchain"]["compile_options"]["reused_network_bundle_sha256"] = digest(old_blob)
    manifest["toolchain"]["compile_options"]["head_hipcc"] = _version([args.hipcc, "--version"], "unknown")
    manifest["bundle"].update(size=len(bundle), sha256=digest(bundle))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernels.bin").write_bytes(bundle)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_promoted(output)
    (output / "catalog.h").write_text(render_catalog(manifest))
    return manifest


def upgrade_label_artifact(spec, source, support_directory, output, args):
    """Reuse every validated network object and add only the PVE support head."""
    source = Path(source)
    manifest = verify(source)
    if manifest["schema_version"] != 1 or manifest["abi_version"] != 1:
        raise ValueError("--reuse-label-artifact requires schema/ABI 1 input")
    expected = public_spec(spec)
    expected.pop("supports_pve")
    if manifest["model"] != expected or not spec["supports_pve"]:
        raise ValueError("reused label artifact does not match a PVE-capable model")
    target = manifest["target"]
    wanted_target = {"architecture": args.gfx,
                     "num_compute_units": args.num_compute_units,
                     "num_chiplets": args.num_chiplets,
                     "wavefront_size": args.wavefront_size}
    if target != wanted_target:
        raise ValueError("reused label artifact target does not match requested target")

    bundle = bytearray((source / manifest["bundle"]["file"]).read_bytes())
    support, support_root = _read_manifest(support_directory)
    if len(support) != 1:
        raise ValueError("HIP tissue-prior support must compile to one kernel")
    entry = support[0]
    blob = (support_root / entry["file"]).read_bytes()
    metadata = code_object_metadata(blob)
    while len(bundle) % 64:
        bundle.append(0)
    offset = len(bundle)
    bundle.extend(blob)
    kernel_id = f"hipcc.{len(manifest['kernels'])}"
    manifest["kernels"].append({
        "id": kernel_id, "offset": offset, "size": len(blob),
        "sha256": digest(blob), "entry": metadata["entry"],
        "symbol": metadata["symbol"], "producer": "hipcc",
        "target": metadata["target"],
        "kernarg_segment_size": metadata["kernarg_segment_size"],
        "static_lds_bytes": metadata["static_lds_bytes"],
        "arguments": metadata["arguments"],
    })

    voxels = spec["dimension"] ** 3
    label_dispatches = [item for item in manifest["dispatches"]
                        if any(arg.get("kind") == "buffer" and
                               arg.get("buffer") == "output" and
                               arg.get("access") == "write"
                               for arg in item["args"])]
    if len(label_dispatches) != 1:
        raise ValueError("reused artifact must have exactly one label dispatch")
    for item in manifest["dispatches"]:
        item["mode"] = "labels" if item is label_dispatches[0] else "always"
    logits = [arg for arg in label_dispatches[0]["args"]
              if arg.get("kind") == "buffer" and arg.get("buffer") == "arena" and
              arg.get("access") == "read" and
              arg.get("bytes") == spec["classes"] * voxels * 4]
    if len(logits) != 1:
        raise ValueError("cannot identify logits in reused label dispatch")
    manifest["dispatches"].append({
        "kernel": kernel_id, "grid": entry["grid"], "block": entry["block"],
        "dynamic_lds_bytes": entry["dynamic_lds_bytes"], "mode": "pve",
        "args": [dict(logits[0]),
                 _argument("output", 0, 4 * voxels * 4, "write")],
    })
    output_buffer = next(item for item in manifest["buffers"]
                         if item["kind"] == "output")
    output_buffer["bytes"] = 4 * voxels * 4
    manifest.update(schema_version=SCHEMA_VERSION, abi_version=ABI_VERSION,
                    model=public_spec(spec), sources=artifact_sources(spec))
    options = dict(manifest["toolchain"]["compile_options"])
    options["pve_support"] = {
        "compiler": _version([args.hipcc, "--version"], "unknown"),
        "reused_label_bundle_sha256": manifest["bundle"]["sha256"],
        "reused_label_bundle_size": manifest["bundle"]["size"],
    }
    manifest["toolchain"]["compile_options"] = options
    manifest["bundle"] = {"file": "kernels.bin", "size": len(bundle),
                          "sha256": digest(bundle)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernels.bin").write_bytes(bundle)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_promoted(output)
    (output / "catalog.h").write_text(render_catalog(manifest))
    return manifest


def build(args):
    if not re.fullmatch(r"gfx[0-9a-f]+", args.gfx):
        raise ValueError("--gfx must be a base architecture such as gfx1151")
    spec = model_spec(args.model)
    work = Path(args.work_dir or
                f"/tmp/brainchop-hip-build/{args.model}-{args.gfx}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    plan = command_plan(args, spec, work)
    if args.reuse_network_artifact:
        if spec["supports_pve"]:
            raise ValueError("--reuse-network-artifact currently requires a labels-only model")
        build_classify(args, spec, work, plan)
        if args.dry_run:
            print(json.dumps({"model": public_spec(spec),
                              "reuse_network_artifact": str(args.reuse_network_artifact),
                              "output": str(Path(args.output).resolve())}, indent=2))
            return None
        manifest = replace_label_tail(spec, args.reuse_network_artifact, args.reuse_source_root,
                                      plan["classify"], args.output, args)
        print(f"verified {spec['name']} {args.gfx}: reused network body, "
              f"replaced label tail -> {args.output}")
        return manifest
    if args.reuse_label_artifact:
        build_support(args, spec, work, plan)
        if args.dry_run:
            print(json.dumps({"model": public_spec(spec),
                              "reuse_label_artifact": str(args.reuse_label_artifact),
                              "output": str(Path(args.output).resolve())}, indent=2))
            return None
        manifest = upgrade_label_artifact(
            spec, args.reuse_label_artifact, plan["support"], args.output, args)
        print(f"verified {spec['name']} {args.gfx}: reused label pipeline, "
              f"added one PVE kernel -> {args.output}")
        return manifest
    network_spec = public_spec(spec)
    network_spec.pop("supports_pve", None)
    common = {"model": network_spec, "gfx": args.gfx,
              "builder": NETWORK_PIPELINE_CACHE_VERSION}
    run_stage("export", plan["export"], [plan["onnx"]],
              # The ONNX embeds weights as initializers; the manifest does not bind them.
              {**common, "exporter": file_digest(SCRIPTS / "export_hip_onnx.py"),
               "weights": file_digest(spec["directory"] / "model.bin"),
               "trace_dim": args.trace_dim, "opset": args.opset}, work,
              dry_run=args.dry_run)
    onnx_hash = "dry-run" if args.dry_run else file_digest(plan["onnx"])
    compile_env = os.environ.copy()
    compile_env.setdefault("MIGRAPHX_TRACE_BENCHMARKING",
                           str(args.migraphx_trace_benchmarking))
    tuning = "exhaustive" if args.exhaustive_tune else "quick"
    problem_cache = Path(args.migraphx_problem_cache or
                         work / f"migraphx-{args.gfx}-{tuning}.json").resolve()
    compile_env.setdefault("MIGRAPHX_PROBLEM_CACHE", str(problem_cache))
    run_stage("compile", plan["compile"], [plan["compiled"]],
              {**common, "onnx": onnx_hash,
               "command": plan["compile"]}, work, env=compile_env,
              dry_run=args.dry_run, timeout=args.compile_timeout)
    migraphx_env = os.environ.copy()
    migraphx_env["PYTHONPATH"] = args.migraphx_python_path + (
        os.pathsep + migraphx_env["PYTHONPATH"]
        if migraphx_env.get("PYTHONPATH") else "")
    run_stage("extract-migraphx", plan["extract_migraphx"],
              [plan["migraphx"] / "manifest.json"],
              {**common, "compiled": ("dry-run" if args.dry_run else
                                       file_digest(plan["compiled"])),
               "extractor": file_digest(SCRIPTS / "extract_hip_kernels.py")},
              work, env=migraphx_env, dry_run=args.dry_run)
    tinygrad_env = os.environ.copy()
    tinygrad_env["DEV"] = args.tinygrad_device.format(gfx=args.gfx)
    tinygrad_env["BEAM"] = str(args.beam)
    for command in plan["extract_tinygrad"]:
        graph = command[command.index("--graph") + 1]
        run_stage(f"tinygrad-{graph}", command,
                  [plan["tinygrad"] / graph / "manifest.json"],
                  {**common, "graph": graph, "beam": args.beam,
                   "device": tinygrad_env["DEV"], "gelu": spec["gelu"],
                   "extractor": file_digest(SCRIPTS / "extract_hip_kernels.py")},
                  work, env=tinygrad_env, dry_run=args.dry_run)
    build_support(args, spec, work, plan)
    if not spec["supports_pve"]:
        build_classify(args, spec, work, plan)
    if args.dry_run:
        print(json.dumps({"model": public_spec(spec), "work_dir": str(work),
                          "output": str(Path(args.output).resolve())}, indent=2))
        return None
    rocm_version = args.rocm_version
    if rocm_version == "auto":
        rocm_path = Path(args.rocm_root) / ".info" / "version"
        rocm_version = rocm_path.read_text().strip() if rocm_path.is_file() else "unknown"
    migraphx_version = args.migraphx_version
    if migraphx_version == "auto":
        migraphx_version = _version([args.migraphx_driver, "--version"], "unknown")
    tinygrad_version = args.tinygrad_version
    if tinygrad_version == "auto":
        tinygrad_version = _version(
            [sys.executable, "-c",
             "import importlib.metadata as m; print(m.version('tinygrad'))"],
            "unknown")
    manifest = pack_artifact(
        spec, args.gfx, args.num_compute_units, args.num_chiplets,
        args.wavefront_size, plan["migraphx"], plan["tinygrad"],
        plan["support"] if spec["supports_pve"] else None, args.output,
        {"rocm": rocm_version, "migraphx": migraphx_version,
         "tinygrad": tinygrad_version},
        {"fp16": True, "exhaustive_tune": args.exhaustive_tune,
         "backend_options": args.backend_options, "beam": args.beam,
         "cross_compiled": not args.native_target},
        classify_directory=None if spec["supports_pve"] else plan["classify"])
    print(f"verified {spec['name']} {args.gfx}: "
          f"{len(manifest['kernels'])} kernels, "
          f"{len(manifest['dispatches'])} dispatches -> {args.output}")
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True,
                        help="directory name under src/models")
    result.add_argument("--gfx", required=True,
                        help="exact base AMD architecture, for example gfx1151")
    result.add_argument("--num-compute-units", required=True, type=int)
    result.add_argument("--num-chiplets", default=1, type=int)
    result.add_argument("--wavefront-size", default=32, type=int,
                        choices=(32, 64))
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--work-dir", type=Path)
    result.add_argument("--reuse-network-artifact", type=Path,
                        help="replace the full label tail of a verified artifact; reuse tuned body")
    result.add_argument("--reuse-source-root", type=Path, default=REPO,
                        help="original source checkout for verifying reused artifact hashes")
    result.add_argument("--reuse-label-artifact", type=Path,
                        help="reuse a verified schema-1 label artifact and "
                             "compile/package only the PVE support kernel")
    result.add_argument("--migraphx-driver", default="migraphx-driver")
    result.add_argument("--hipcc", default="hipcc")
    result.add_argument("--migraphx-python-path", default="/opt/rocm/lib")
    result.add_argument("--rocm-root", default="/opt/rocm")
    result.add_argument("--native-target", action="store_true",
                        help="compile for the attached GPU (MIGraphX 2.15 compatible)")
    result.add_argument("--tinygrad-device", default="AMD:LLVM:{gfx}")
    result.add_argument("--beam", default=2, type=int)
    result.add_argument("--trace-dim", default=32, type=int)
    result.add_argument("--opset", default=18, type=int)
    result.add_argument("--backend-options", default="")
    result.add_argument("--exhaustive-tune", action="store_true")
    result.add_argument("--compile-timeout", default=3600, type=int,
                        help="seconds allowed per MIGraphX compile attempt; "
                             "0 disables the guard (default: %(default)s)")
    result.add_argument("--support-timeout", default=300, type=int,
                        help="seconds allowed for the HIP support-kernel compile; "
                             "0 disables the guard (default: %(default)s)")
    result.add_argument("--migraphx-trace-benchmarking", default=1, type=int,
                        choices=(0, 1, 2, 3),
                        help="MIGraphX benchmark progress detail (default: %(default)s)")
    result.add_argument("--migraphx-problem-cache", type=Path,
                        help="persistent MIGraphX tuning cache (default: a "
                             "quick/exhaustive-specific file in --work-dir)")
    result.add_argument("--rocm-version", default="auto")
    result.add_argument("--migraphx-version", default="auto")
    result.add_argument("--tinygrad-version", default="auto")
    result.add_argument("--dry-run", action="store_true")
    return result


def main():
    try:
        build(parser().parse_args())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
