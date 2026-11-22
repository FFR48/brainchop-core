#!/bin/sh
# Python reference for brainchop-mindsnap; see .claude/rules/models.md.
# Note: mindsnap v4.0 Gold uses 5/95 rank normalization (matching mindmap and
# brainchop-test), whereas brainchop-cli's qnormalize() hardcodes 2/98 linear.
# usage: BC_PYTHON=<venv with ../brainchop-cli> scripts/oracle_mindsnap.sh in.nii.gz out.nii.gz
# The CLI's DKatlas entry uses older weights, so the v4.0 Gold weights go
# through --custom: a MeshNet config plus model_f32.safetensors keys minus "m.".
set -eu
# BC_PYTHON, not PYTHON: tinygrad reads PYTHON as its interpreter-backend switch.
PY=${BC_PYTHON:?set BC_PYTHON to the python of a venv with ../brainchop-cli}
[ $# -eq 2 ] || { echo "usage: BC_PYTHON=... $0 in.nii.gz out.nii.gz" >&2; exit 2; }
WEIGHTS=${BC_MINDSNAP_WEIGHTS:-$(cd "$(dirname "$0")/../.." && pwd)/brainchop-test/public/models/model24chan104cls_v3_repbn/model_f32.safetensors}
unset PYTHON
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
"$PY" - "$DIR" "$WEIGHTS" <<'PY'
import json, sys
from tinygrad.nn.state import safe_load, safe_save
d = sys.argv[1]
dil = (1, 3, 5, 7, 13, 19, 31, 19, 13, 7, 5, 3, 1)
layers = [{"in_channels": 1 if i == 0 else 24, "out_channels": 24, "kernel_size": 3,
           "stride": 1, "padding": x, "dilation": x} for i, x in enumerate(dil)]
layers.append({"in_channels": 24, "out_channels": 104, "kernel_size": 1, "stride": 1,
               "padding": 0, "dilation": 1, "bias": True})
json.dump({"dropout_p": 0, "bnorm": True, "gelu": True, "affine": True,
           "bias": False, "layers": layers}, open(f"{d}/model.json", "w"))
w = safe_load(sys.argv[2])
safe_save({k.removeprefix("m."): v.to("CPU").contiguous() for k, v in w.items()
           if not k.startswith("m.seq_conv_argmax.")}, f"{d}/model.pth")
PY
"$(dirname "$PY")/brainchop" --custom "$DIR" --no-optimize "$1" -o "$2"
