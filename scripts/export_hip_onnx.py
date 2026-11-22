#!/usr/bin/env python3
"""Export any checked-in BrainChop MeshNet as the ONNX input to MIGraphX.

Heavy imports are deliberately local to ``export``.  The developer compiler
launches this file in its own process so PyTorch's ROCm libraries never share a
process with the system MIGraphX bindings.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIM = 256
EPS = 1e-5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_spec(name, repo=REPO):
    """Read architecture facts from the model files rather than a name table."""
    model_dir = Path(repo) / "src" / "models" / name
    meta = json.loads((model_dir / "model_meta.json").read_text())
    capabilities = json.loads(
        (Path(repo) / "src" / "model_capabilities.json").read_text())
    topology = json.loads((model_dir / "model.json").read_text())
    layers = topology["modelTopology"]["model_config"]["config"]["layers"]
    convolutions = [layer for layer in layers if layer["class_name"] == "Conv3D"]
    hidden = [layer for layer in convolutions
              if layer["config"]["kernel_size"] == [3, 3, 3]]
    heads = [layer for layer in convolutions
             if layer["config"]["kernel_size"] == [1, 1, 1] and
             not layer["config"].get("name", "").startswith("affine_")]
    if not hidden or len(heads) != 1:
        raise ValueError(f"{name}: expected 3x3x3 hidden layers and one 1x1x1 head")
    channels = hidden[0]["config"]["filters"]
    if any(layer["config"]["filters"] != channels for layer in hidden):
        raise ValueError(f"{name}: hidden layers do not have a constant width")
    dilations = [layer["config"]["dilation_rate"] for layer in hidden]
    if any(len(set(value)) != 1 for value in dilations):
        raise ValueError(f"{name}: anisotropic dilation is not supported")
    affine = any(layer["config"].get("name", "").startswith("affine_")
                 for layer in layers)
    if meta.get("gelu") not in ("tanh_rational", "exp2_sigmoid"):
        raise ValueError(f"{name}: unsupported GELU flavour {meta.get('gelu')!r}")
    return {
        "name": name,
        "directory": model_dir,
        "dimension": DIM,
        "channels": channels,
        "hidden_layers": len(hidden),
        "classes": heads[0]["config"]["filters"],
        "dilations": [value[0] for value in dilations],
        "has_affine": affine,
        "classifier_bias": bool(meta["classifier_bias"]),
        "supports_pve": bool(capabilities[name]["pve"]),
        "gelu": meta["gelu"],
        "descriptor_sha256": sha256(model_dir / "model_def.h"),
    }


def _load_tensors(spec):
    import numpy as np

    model_dir = spec["directory"]
    entries = json.loads((model_dir / "model.json").read_text()) \
        ["weightsManifest"][0]["weights"]
    values = np.fromfile(model_dir / "model.bin", dtype="<f4")
    tensors, offset = {}, 0
    for entry in entries:
        size = int(np.prod(entry["shape"]))
        tensors[entry["name"]] = values[offset:offset + size].reshape(entry["shape"])
        offset += size
    if offset != values.size:
        raise ValueError(f"model manifest covers {offset} of {values.size} floats")
    return tensors


def _gelu_tanh_rational(torch, x):
    u = 0.797884583 * (x + 0.044715 * (x * x * x))
    u = torch.clamp(u, -9.0, 9.0)
    s = u * u
    p = -8.29118133e-14
    for coefficient in (5.19263868e-11, -2.00294448e-08, 1.11017944e-05,
                        0.00309865153, 0.130791619, 0.99999994):
        p = p * s + coefficient
    q = 0.000253859733
    for coefficient in (0.024473751, 0.464124829, 1.0):
        q = q * s + coefficient
    return (0.5 * x) * (1.0 + u * p / q)


def _fast_exp2(torch, x):
    x = torch.where(x > -126.0, x, torch.full_like(x, -126.0))
    x = torch.clamp(x, max=126.0)
    whole = torch.floor(x)
    fraction = x - whole
    polynomial = 0.000216128448
    for coefficient in (0.00124678648, 0.0096754498, 0.0554852814,
                        0.240229305, 0.693147044, 1.0):
        polynomial = polynomial * fraction + coefficient
    return polynomial * (2.0 ** whole)


def _gelu_exp2_sigmoid(torch, x):
    u = x + 0.044715 * x * x * x
    return x / (1.0 + _fast_exp2(torch, -2.302208198144325 * u))


def export(name, output, trace_dim=32, opset=18):
    import numpy as np
    import torch
    import torch.nn.functional as functional

    spec = model_spec(name)
    tensors = _load_tensors(spec)
    layers = []
    for index, dilation in enumerate(spec["dilations"]):
        # Manifest [kx,ky,kz,in,out] -> PyTorch [out,in,kz,ky,kx].  The
        # spatial reversal reproduces the C loader/packer for all three models.
        weight = tensors[f"conv3d_{index}_gn/kernel"].transpose(4, 3, 2, 1, 0)
        layer = {"weight": torch.from_numpy(np.ascontiguousarray(weight)),
                 "dilation": dilation}
        if spec["has_affine"]:
            matrix = tensors[f"affine_{index}/kernel"].reshape(
                spec["channels"], spec["channels"])
            diagonal = np.diag(np.diag(matrix))
            if not np.array_equal(matrix, diagonal):
                raise ValueError(f"affine_{index} is not diagonal")
            layer["scale"] = torch.from_numpy(np.diag(matrix).copy())
            layer["bias"] = torch.from_numpy(tensors[f"affine_{index}/bias"].copy())
        layers.append(layer)
    head_name = "conv3d_final" if spec["has_affine"] else "output"
    head_weight = tensors[f"{head_name}/kernel"].transpose(4, 3, 2, 1, 0)
    head_bias = tensors.get(f"{head_name}/bias") if spec["classifier_bias"] else None

    gelu = (_gelu_tanh_rational if spec["gelu"] == "tanh_rational"
            else _gelu_exp2_sigmoid)

    class MeshNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for index, layer in enumerate(layers):
                self.register_buffer(f"w{index}", layer["weight"])
                if spec["has_affine"]:
                    self.register_buffer(f"s{index}", layer["scale"])
                    self.register_buffer(f"b{index}", layer["bias"])
            self.register_buffer("head_weight",
                                 torch.from_numpy(np.ascontiguousarray(head_weight)))
            self.register_buffer("head_bias", torch.from_numpy(head_bias.copy())
                                 if head_bias is not None else
                                 torch.zeros(spec["classes"]))

        def forward(self, value):
            for index, dilation in enumerate(spec["dilations"]):
                value = functional.conv3d(value, getattr(self, f"w{index}"),
                                          dilation=dilation, padding=dilation)
                scale = getattr(self, f"s{index}") if spec["has_affine"] else None
                bias = getattr(self, f"b{index}") if spec["has_affine"] else None
                value = functional.group_norm(value, spec["channels"], scale,
                                              bias, eps=EPS)
                value = gelu(torch, value)
            return functional.conv3d(value, self.head_weight,
                                     bias=self.head_bias if
                                     spec["classifier_bias"] else None)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, trace_dim, trace_dim, trace_dim)
    with torch.no_grad():
        torch.onnx.export(
            MeshNet().eval(), (dummy,), output,
            input_names=["volume"], output_names=["logits"],
            dynamic_axes={"volume": {2: "d", 3: "h", 4: "w"},
                          "logits": {2: "d", 3: "h", 4: "w"}},
            opset_version=opset, do_constant_folding=True, dynamo=False)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-dim", type=int, default=32)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    spec = export(args.model, args.output, args.trace_dim, args.opset)
    print(f"{spec['name']}: {spec['hidden_layers']} x {spec['channels']}, "
          f"{spec['classes']} classes -> {args.output}")


if __name__ == "__main__":
    main()
