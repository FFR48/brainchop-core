#!/usr/bin/env python3
"""Extract standalone HSA code objects from MIGraphX and tinygrad.

This is an explicitly version-pinned maintainer interface.  MIGraphX does not
yet expose code-object extraction as a stable API, so a changed ``op().values``
shape fails here during generation without affecting checked-in artifacts.
"""
import argparse
import functools
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from hip_artifacts import code_object_metadata

_RECORDS = []
ET_DYN = 3


def _elf_type(blob):
    return int.from_bytes(blob[16:18], "little") if len(blob) >= 18 else 0


@functools.lru_cache(maxsize=None)
def _find_lld():
    override = os.environ.get("BC_AMDGPU_LLD")
    if override:
        resolved = shutil.which(override)
        if resolved is None:
            raise RuntimeError(f"BC_AMDGPU_LLD={override!r} is not executable")
        return resolved
    resolved = shutil.which("ld.lld")
    if resolved is not None:
        return resolved
    versioned = [
        path
        for directory in os.get_exec_path()
        for path in glob.glob(os.path.join(directory, "ld.lld-*"))
        if re.fullmatch(r"ld\.lld-\d+", os.path.basename(path))
    ]
    if versioned:
        return max(versioned, key=lambda path: int(path.rsplit("-", 1)[1]))
    rocm_lld = "/opt/rocm/llvm/bin/ld.lld"
    if os.access(rocm_lld, os.X_OK):
        return rocm_lld
    raise RuntimeError(
        "no ld.lld found; install LLVM lld or set BC_AMDGPU_LLD")


def _link_hsaco(blob, function_name):
    """Turn tinygrad's relocatable AMDGPU object into a HIP-loadable object."""
    if _elf_type(blob) == ET_DYN:
        return blob
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "kernel.o"
        linked = Path(directory) / "kernel.hsaco"
        source.write_bytes(blob)
        result = subprocess.run(
            [_find_lld(), "--no-undefined", "-shared", "-o", str(linked),
             str(source)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"linking tinygrad kernel {function_name} failed: "
                f"{result.stderr.strip()}")
        return linked.read_bytes()


def _shape_spec(shape):
    result = {"type": shape["type"], "lens": shape["lens"],
              "strides": shape.get("strides")}
    if shape.get("sub_shapes"):
        result["sub_shapes"] = [_shape_spec(item) for item in shape["sub_shapes"]]
    return result


def extract_migraphx(compiled, output, gfx):
    import migraphx

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    program = migraphx.load(str(compiled))
    objects = [instruction for instruction in program.get_main_module()
               if instruction.name() == "gpu::code_object"]
    entries = []
    for index, instruction in enumerate(objects):
        values = instruction.op().values()
        blob = bytes(values["code_object"])
        metadata = code_object_metadata(blob)
        if metadata["target"] != gfx:
            raise ValueError(f"MIGraphX object {index} targets "
                             f"{metadata['target']}, expected {gfx}")
        filename = f"migraphx_{index:03d}.hsaco"
        (output / filename).write_bytes(blob)
        entries.append({
            "file": filename,
            "symbol": values["symbol_name"],
            "global": values["global"],
            "local": values["local"],
            "grid": [values["global"] // values["local"], 1, 1],
            "block": [values["local"], 1, 1],
            "output_arg": values["output_arg"],
            "inputs": [_shape_spec(item) for item in values["expected_inputs"]],
        })
    (output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"extracted {len(entries)} MIGraphX code objects -> {output}")


def _tinygrad_imports():
    from tinygrad import Device, Tensor, dtypes
    from tinygrad.uop.ops import Ops, PatternMatcher, UPat
    import tinygrad.engine.realize as realize
    return Device, Tensor, dtypes, Ops, PatternMatcher, UPat, realize


def _spy(ctx, call, ast):
    Device, Tensor, dtypes, Ops, PatternMatcher, UPat, realize = _tinygrad_imports()
    resolved = realize.resolve_params(call, ctx.input_uops)
    selected = [resolved[index] for index in ast.arg.globals]
    bound = []
    for buffers, _ in realize.unwrap_multi(call, selected):
        bound = [buffer.ensure_allocated() for buffer in buffers]
        break
    _RECORDS.append({
        "info": ast.arg,
        "binary": next((item.arg for item in ast.src if item.op is Ops.BINARY), None),
        "buffers": bound,
    })
    return realize.exec_kernel(ctx, call, ast)


def _capture(build, named):
    Device, Tensor, dtypes, Ops, PatternMatcher, UPat, realize = _tinygrad_imports()
    build()
    Device.default.synchronize()
    _RECORDS.clear()
    saved = realize.pm_exec
    realize.pm_exec = PatternMatcher([
        (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call",
              allow_any_len=True), realize.exec_copy),
        (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call",
              allow_any_len=True), _spy),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="encdec", name="ast"),),
              name="call", allow_any_len=True), realize.exec_encdec),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="graph", name="ast"),),
              name="call", allow_any_len=True), realize.exec_graph),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="hcq",
              src=(UPat(Ops.PROGRAM, name="ast"),)),), name="call",
              allow_any_len=True), realize.exec_hcq),
        (UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, arg="validate", name="ast"),),
              name="call", allow_any_len=True), realize.exec_validate),
    ])
    try:
        result = build()
        Device.default.synchronize()
    finally:
        realize.pm_exec = saved
    if not _RECORDS:
        raise RuntimeError("captured no tinygrad kernels")
    known = {}
    for name, tensor in named.items():
        buffer = tensor.uop.base.realized
        if buffer is None:
            raise RuntimeError(f"tinygrad input {name} was not realized")
        known[id(buffer)] = name
    known[id(result.uop.base.realized)] = "out"
    return list(_RECORDS), known


def _tinygrad_graph(name, channels, classes, dimension, affine, gelu):
    Device, Tensor, dtypes, Ops, PatternMatcher, UPat, realize = _tinygrad_imports()
    if name == "widen":
        value = Tensor.empty(1, 1, dimension, dimension, dimension,
                             dtype=dtypes.half).contiguous().realize()
        return lambda: value.float().realize(), {"x": value}
    if name == "moments":
        value = Tensor.empty(1, channels, dimension, dimension, dimension,
                             dtype=dtypes.half).contiguous().realize()
        def build():
            wide = value.float().reshape(1, channels, -1)
            mean = wide.mean(axis=2, keepdim=True)
            variance = ((wide - mean) ** 2).mean(axis=2, keepdim=True)
            return Tensor.stack(mean.reshape(channels),
                                (variance.reshape(channels) + 1e-5).rsqrt()).realize()
        return build, {"x": value}
    if name == "argmax":
        value = Tensor.empty(1, classes, dimension, dimension, dimension,
                             dtype=dtypes.float).contiguous().realize()
        return (lambda: value.reshape(classes, -1).argmax(axis=0)
                .cast(dtypes.uint8).realize()), {"logits": value}
    if name != "norm_gelu":
        raise ValueError(f"unknown tinygrad graph {name}")
    value = Tensor.empty(1, channels, dimension, dimension, dimension,
                         dtype=dtypes.half).contiguous().realize()
    mean = Tensor.empty(1, channels, 1, 1, 1,
                        dtype=dtypes.float).contiguous().realize()
    inverse = Tensor.empty(1, channels, 1, 1, 1,
                           dtype=dtypes.float).contiguous().realize()
    named = {"x": value, "mean": mean, "inv": inverse}
    if affine:
        named["scale"] = Tensor.empty(1, channels, 1, 1, 1,
                                      dtype=dtypes.float).contiguous().realize()
        named["bias"] = Tensor.empty(1, channels, 1, 1, 1,
                                     dtype=dtypes.float).contiguous().realize()

    def build():
        result = (value.float() - mean) * inverse
        if affine:
            result = result * named["scale"] + named["bias"]
        # tinygrad's fused GELU is the kernel Usman's measured implementation
        # used.  The exact model flavour remains recorded so promotion can gate
        # the resulting numerical tolerance separately for each model.
        return result.gelu().cast(dtypes.half).realize()
    return build, named


def extract_tinygrad(graph, output, gfx, channels, classes, dimension,
                     affine, gelu):
    selected = os.environ.get("DEV", "")
    if gfx not in selected:
        raise RuntimeError(f"DEV must name the target explicitly, for example "
                           f"DEV=AMD:LLVM:{gfx}; got {selected!r}")
    build, named = _tinygrad_graph(graph, channels, classes, dimension,
                                   affine, gelu)
    records, known = _capture(build, named)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scratch, entries = {}, []
    for index, record in enumerate(records):
        binary = record["binary"]
        if binary is None:
            raise RuntimeError(f"tinygrad kernel {index} has no code object")
        binary = _link_hsaco(binary, record["info"].function_name)
        metadata = code_object_metadata(binary)
        if metadata["target"] != gfx:
            raise ValueError(f"tinygrad object targets {metadata['target']}, expected {gfx}")
        filename = f"{graph}_{index:03d}.hsaco"
        (output / filename).write_bytes(binary)
        arguments = []
        for buffer in record["buffers"]:
            key = id(buffer)
            role = known.get(key)
            if role is None:
                role = scratch.setdefault(key, f"scratch{len(scratch)}")
            arguments.append({"role": role, "bytes": buffer.nbytes})
        info = record["info"]
        entries.append({
            "file": filename,
            "symbol": info.function_name,
            "grid": list(info.global_size or (1, 1, 1)),
            "block": list(info.local_size or (1, 1, 1)),
            "dynamic_lds_bytes": 0,
            "args": arguments,
        })
    (output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"extracted {len(entries)} tinygrad {graph} code objects -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mx = subparsers.add_parser("migraphx")
    mx.add_argument("--compiled", required=True, type=Path)
    mx.add_argument("--output", required=True, type=Path)
    mx.add_argument("--gfx", required=True)
    tg = subparsers.add_parser("tinygrad")
    tg.add_argument("--graph", required=True,
                    choices=("widen", "moments", "norm_gelu", "argmax"))
    tg.add_argument("--output", required=True, type=Path)
    tg.add_argument("--gfx", required=True)
    tg.add_argument("--channels", required=True, type=int)
    tg.add_argument("--classes", required=True, type=int)
    tg.add_argument("--dimension", default=256, type=int)
    tg.add_argument("--affine", action="store_true")
    tg.add_argument("--gelu", required=True,
                    choices=("tanh_rational", "exp2_sigmoid"))
    args = parser.parse_args()
    if args.command == "migraphx":
        extract_migraphx(args.compiled, args.output, args.gfx)
    else:
        extract_tinygrad(args.graph, args.output, args.gfx, args.channels,
                         args.classes, args.dimension, args.affine, args.gelu)


if __name__ == "__main__":
    main()
