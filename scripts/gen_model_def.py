#!/usr/bin/env python3
"""Generate a per-model constants header from a brainchop model directory.

usage: gen_model_def.py MODEL_DIR [-o OUTPUT] [--check]

Reads MODEL_DIR/model.json (a TFJS layers-model topology plus a weights
manifest), MODEL_DIR/model.bin (for size validation) and MODEL_DIR/model_meta.json
(the hand-written residue, see below), and writes MODEL_DIR/model_def.h.

--check regenerates into memory and diffs against the file on disk, exiting
nonzero if they differ. That is what makes drift between model.bin and the
compiled-in constants a build failure rather than a silent wrong answer.

WHAT COMES FROM model.json
    channel count, hidden-layer count, class count, the dilation table, whether
    the model has a post-GroupNorm affine, and the exact weight layout -- the
    manifest lists every tensor in file order with its shape, so the per-layer
    offsets and the total float count are derived, not transcribed.

WHAT CANNOT, AND SO LIVES IN model_meta.json
    * Normalization quantiles. model.json says nothing about them; MindGrab uses
      2/98 with a 1e-3 denominator guard and clamps to [0,1], model16 uses 5/95
      with neither.
    * The GELU approximation. BOTH models declare activation "gelu", but the
      reference implementations differ -- MindGrab computes a rational tanh form
      and model16 an exp2 sigmoid form. They are not interchangeable: each is
      what its upstream reference computes, and swapping them breaks bit-exact
      parity. A string in model.json cannot express that.
    * Deliberate divergences from the checkpoint. MindGrab's model.json declares
      use_bias=true on the classifier, but the published runtime omits that bias
      and applying it changes thousands of voxels.
    * Output semantics (a mask over the input image vs. a label map) and which
      CLI options the model supports.

The generator CROSS-CHECKS the metadata against model.json and fails on any
disagreement it was not explicitly told to expect: a mismatch must be declared
with an "_override_reason" key. Silently trusting the metadata would recreate
exactly the class of bug this script exists to prevent.
"""

import argparse
import array
import json
import math
import os
import sys

GELU_FLAVORS = {"tanh_rational": 0, "exp2_sigmoid": 1}
OUTPUT_KINDS = {"mask": 0, "labels": 1}
QUANTILE_MODES = {"linear": 0, "rank": 1}
CAPABILITIES = ("crop", "mask", "border", "save_conform", "ct", "comply",
                "export_classes")


def die(message):
    sys.exit(f"gen_model_def: {message}")


def load_topology(model_dir):
    with open(os.path.join(model_dir, "model.json"), encoding="utf-8") as handle:
        return json.load(handle)


def parse_layers(topology):
    """Split the TFJS layer list into hidden blocks plus a final classifier."""
    layers = topology["modelTopology"]["model_config"]["config"]["layers"]
    convolutions = [layer for layer in layers if layer["class_name"] == "Conv3D"]
    if len(convolutions) < 2:
        die("model.json has fewer than two Conv3D layers")

    classifier = convolutions[-1]
    body = convolutions[:-1]

    # An affine model interleaves conv3d_N_gn with affine_N; a plain one does not.
    affine = [c for c in body if c["config"]["name"].startswith("affine")]
    kernels = [c for c in body if not c["config"]["name"].startswith("affine")]
    has_affine = bool(affine)
    if has_affine and len(affine) != len(kernels):
        die(f"{len(kernels)} conv layers but {len(affine)} affine layers")

    channels = {c["config"]["filters"] for c in kernels}
    if len(channels) != 1:
        die(f"hidden layers disagree on channel count: {sorted(channels)}")

    dilations = []
    for conv in kernels:
        rate = conv["config"]["dilation_rate"]
        if len(set(rate)) != 1:
            die(f"anisotropic dilation {rate} in {conv['config']['name']}")
        dilations.append(rate[0])

    return {
        "channels": channels.pop(),
        "hidden": len(kernels),
        "dilations": dilations,
        "has_affine": has_affine,
        "classes": classifier["config"]["filters"],
        "classifier_bias": bool(classifier["config"].get("use_bias", True)),
    }


def parse_weights(topology, model_dir):
    """Derive the weight layout from the manifest and check it against model.bin."""
    manifest = topology["weightsManifest"]
    tensors = [tensor for group in manifest for tensor in group["weights"]]
    if any(tensor["dtype"] != "float32" for tensor in tensors):
        die("weights manifest declares a non-float32 tensor")

    offsets, running = [], 0
    for tensor in tensors:
        count = math.prod(tensor["shape"])
        offsets.append((tensor["name"], running, count, tensor["shape"]))
        running += count

    binary = os.path.join(model_dir, "model.bin")
    size = os.path.getsize(binary)
    if size % 4:
        die(f"{binary} is {size} bytes, not a whole number of float32")
    if size // 4 != running:
        die(f"{binary} holds {size // 4} floats but model.json describes {running}")

    # Every backend keeps only the affine diagonal, so off-diagonal weights
    # would be dropped silently at run time.
    values = array.array("f")
    with open(binary, "rb") as handle:
        values.fromfile(handle, running)
    if sys.byteorder != "little":
        values.byteswap()
    for name, offset, count, shape in offsets:
        if name.startswith("affine_") and name.endswith("/kernel"):
            channels = shape[-1]
            if any(values[offset + row * channels + col]
                   for row in range(channels) for col in range(channels) if row != col):
                die(f"{name} is not diagonal")

    return {"total": running, "tensors": offsets}


def load_meta(model_dir, derived):
    path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(path):
        die(f"missing {path}; see the module docstring for what it must declare")
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)

    meta.setdefault("loader_transposes_kernel", False)
    for key in ("cli_name", "description", "output_kind", "gelu", "norm",
                "capabilities"):
        if key not in meta:
            die(f"{path} is missing required key '{key}'")
    if meta["gelu"] not in GELU_FLAVORS:
        die(f"unknown gelu flavor '{meta['gelu']}'; expected one of "
            f"{sorted(GELU_FLAVORS)}")
    if meta["output_kind"] not in OUTPUT_KINDS:
        die(f"unknown output_kind '{meta['output_kind']}'")

    norm = meta["norm"]
    for key in ("q_low", "q_high", "denom_eps", "clamp", "quantile_mode"):
        if key not in norm:
            die(f"{path}: norm is missing '{key}'")
    if norm["quantile_mode"] not in QUANTILE_MODES:
        die(f"{path}: unknown quantile_mode '{norm['quantile_mode']}'")
    if not 0.0 <= norm["q_low"] < norm["q_high"] <= 1.0:
        die(f"{path}: norm quantiles {norm['q_low']}..{norm['q_high']} are not "
            "an increasing pair inside [0, 1]")

    unknown = set(meta["capabilities"]) - set(CAPABILITIES)
    if unknown:
        die(f"{path}: unknown capabilities {sorted(unknown)}")

    # A metadata value that contradicts model.json is either a real quirk or a
    # mistake, and the two are indistinguishable without being told. Require the
    # quirk to be declared, so the mistake stays an error.
    declared = meta.get("classifier_bias")
    if declared is not None and bool(declared) != derived["classifier_bias"]:
        if "classifier_bias_override_reason" not in meta:
            die(f"{path}: classifier_bias={declared} contradicts model.json "
                f"(use_bias={derived['classifier_bias']}). If that is "
                "deliberate, add classifier_bias_override_reason explaining why; "
                "otherwise fix the metadata")
        derived = dict(derived, classifier_bias=bool(declared))
    return meta, derived


def render(name, derived, weights, meta):
    """Emit namespaced constants plus a generic alias block.

    Every constant is emitted twice over. BC_<PREFIX>_* is always defined, so
    several models can be included in one translation unit -- which is what the
    transitional multi-model brainchopc needs. BC_MODEL_* aliases appear only
    when the build selects this model with -DBC_MODEL_IS_<NAME>, which is what
    the shared engine wants once there is one executable per model: it can then
    be written against BC_MODEL_CHAN with no idea which model it compiles for.

    Emitting only the generic names would make two models collide in one TU;
    emitting only the namespaced ones would force the shared engine to know
    every model's prefix. Both, with the alias block guarded, costs a few lines
    and serves both shapes of build.
    """
    guard = f"BC_MODEL_DEF_{name.upper().replace('-', '_')}_H"
    prefix = meta.get("c_prefix", name.upper())
    if not prefix.replace("_", "").isalnum() or prefix[0].isdigit():
        die(f"c_prefix '{prefix}' is not a usable C identifier prefix")
    select = "BC_MODEL_IS_" + "".join(
        c if c.isalnum() else "_" for c in name).upper()
    norm = meta["norm"]
    lines = [
        "/*",
        f" * Generated by scripts/gen_model_def.py from models/{name}/.",
        " * Do not edit: `make check-generated` regenerates this and fails on a diff.",
        " *",
        f" * Model:       {name}",
        f" * Description: {meta['description']}",
        " */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f'#define BC_{prefix}_NAME "{name}"',
        f'#define BC_{prefix}_CLI_NAME "{meta["cli_name"]}"',
        f'#define BC_{prefix}_DESCRIPTION "{meta["description"]}"',
        "",
        "/* Architecture, derived from model.json. */",
        f"#define BC_{prefix}_CHAN {derived['channels']}",
        f"#define BC_{prefix}_NHIDDEN {derived['hidden']}",
        f"#define BC_{prefix}_NCLASS {derived['classes']}",
        f"#define BC_{prefix}_HAS_AFFINE {int(derived['has_affine'])}",
        f"#define BC_{prefix}_CLASSIFIER_BIAS {int(derived['classifier_bias'])}",
        "",
        "/* Weight layout, derived from the model.json manifest and checked",
        "   against the byte size of model.bin. */",
        f"#define BC_{prefix}_WEIGHT_FLOATS {weights['total']}",
        "",
        "/* Numerics that model.json cannot express; see models/"
        f"{name}/model_meta.json. */",
        f"#define BC_{prefix}_GELU {GELU_FLAVORS[meta['gelu']]}",
        f"#define BC_{prefix}_NORM_Q_LOW {norm['q_low']!r}",
        f"#define BC_{prefix}_NORM_Q_HIGH {norm['q_high']!r}",
        f"#define BC_{prefix}_NORM_DENOM_EPS {norm['denom_eps']!r}f",
        f"#define BC_{prefix}_NORM_CLAMP {int(norm['clamp'])}",
        f"#define BC_{prefix}_QUANTILE_MODE {QUANTILE_MODES[norm['quantile_mode']]}",
        "",
        "/* 1 when the loader has already swapped the TFJS and NIfTI spatial",
        "   axes, so the kernel packer must not do it again. An asymmetry of",
        "   how each model's weights reach the engine, not of the model. */",
        f"#define BC_{prefix}_LOADER_TRANSPOSE "
        f"{int(meta['loader_transposes_kernel'])}",
        "",
        "/* Output semantics and the CLI surface this build offers. */",
        f"#define BC_{prefix}_OUTPUT_KIND {OUTPUT_KINDS[meta['output_kind']]}",
        f"#define BC_{prefix}_OUTPUT_IS_MASK "
        f"{int(meta['output_kind'] == 'mask')}",
    ]
    for capability in CAPABILITIES:
        enabled = int(bool(meta["capabilities"].get(capability, False)))
        lines.append(f"#define BC_{prefix}_SUPPORTS_{capability.upper()} {enabled}")

    lines += [
        "",
        "/* Per-layer dilation, in layer order. A macro rather than an array so",
        "   that exactly one translation unit defines storage for it; a",
        "   definition in the header would be duplicated per TU. */",
        f"#define BC_{prefix}_DILATION {{ " +
        ", ".join(str(d) for d in derived["dilations"]) + " }",
        "",
        "/* Weight tensors in model.bin order: offset in floats, then length. */",
    ]
    for tensor_name, offset, count, shape in weights["tensors"]:
        symbol = tensor_name.replace("/", "_").upper()
        lines.append(f"#define BC_{prefix}_W_{symbol}_OFF {offset}  /* {shape} */")
        lines.append(f"#define BC_{prefix}_W_{symbol}_LEN {count}")

    generic = ["NAME", "CLI_NAME", "DESCRIPTION", "CHAN", "NHIDDEN", "NCLASS",
               "HAS_AFFINE", "CLASSIFIER_BIAS", "WEIGHT_FLOATS", "GELU",
               "LOADER_TRANSPOSE",
               "NORM_Q_LOW", "NORM_Q_HIGH", "NORM_DENOM_EPS", "NORM_CLAMP",
               "QUANTILE_MODE", "OUTPUT_KIND", "OUTPUT_IS_MASK", "DILATION"]
    generic += [f"SUPPORTS_{c.upper()}" for c in CAPABILITIES]
    lines += [
        "",
        "/* Generic aliases, defined only when this build selects this model.",
        "   They are what lets the shared engine be written against BC_MODEL_*",
        "   without knowing which model it is compiling for. */",
        f"#if defined({select})",
    ]
    lines += [f"#define BC_MODEL_{key} BC_{prefix}_{key}" for key in generic]
    lines += [
        f"#define BC_MODEL_GELU_{meta['gelu'].upper()} 1",
        "#endif",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("-o", "--output")
    parser.add_argument("--check", action="store_true",
                        help="diff against the existing header instead of writing")
    arguments = parser.parse_args()

    model_dir = arguments.model_dir.rstrip("/")
    name = os.path.basename(model_dir)
    topology = load_topology(model_dir)
    derived = parse_layers(topology)
    weights = parse_weights(topology, model_dir)
    meta, derived = load_meta(model_dir, derived)
    text = render(name, derived, weights, meta)

    output = arguments.output or os.path.join(model_dir, "model_def.h")
    if arguments.check:
        if not os.path.exists(output):
            sys.exit(f"gen_model_def: {output} does not exist; run without --check")
        with open(output, encoding="utf-8") as handle:
            if handle.read() != text:
                sys.exit(f"gen_model_def: {output} is stale; "
                         "regenerate it with scripts/gen_model_def.py")
        print(f"gen_model_def: {output} is up to date")
        return
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"gen_model_def: wrote {output} "
          f"({derived['channels']}ch x {derived['hidden']} layers, "
          f"{derived['classes']} classes, {weights['total']} floats)")


if __name__ == "__main__":
    main()
