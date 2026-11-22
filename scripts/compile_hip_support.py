#!/usr/bin/env python3
"""Compile the architecture-exact CAT-lite HIP support kernel."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from hip_artifacts import code_object_metadata


def unbundle_object(hipcc, gfx, bundle, code):
    bundled = bundle.read_bytes()
    if bundled.startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
        compiler = Path(shutil.which(hipcc) or hipcc).resolve()
        roots = [compiler.parent.parent, Path("/opt/rocm")]
        bundler = next((path for root in roots for path in
                        (root / "llvm/bin/clang-offload-bundler",
                         root / "lib/llvm/bin/clang-offload-bundler")
                        if path.is_file()), None)
        if bundler is None:
            raise RuntimeError("hipcc produced an offload bundle but "
                               "clang-offload-bundler was not found")
        subprocess.run([
            str(bundler), "-type=o", "-unbundle", f"-input={bundle}",
            f"-output={code}",
            f"-targets=hipv4-amdgcn-amd-amdhsa--{gfx}",
        ], check=True)
    else:
        code.write_bytes(bundled)
    bundle.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipcc", default="hipcc")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gfx", required=True)
    parser.add_argument("--dimension", required=True, type=int)
    parser.add_argument("--include", required=True, type=Path)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--classes", type=int)
    parser.add_argument("--classifier-bias", action="store_true")
    args = parser.parse_args()
    classify = args.channels is not None
    if classify and (args.channels < 1 or not args.classes or not 1 <= args.classes <= 256):
        parser.error("classifier requires positive channels and 1..256 classes")
    if args.dimension < 1:
        parser.error("dimension must be positive")
    entry = "bc_hip_classify" if classify else "bc_hip_tissue_priors"
    stem = "classify" if classify else "tissue_priors"
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = args.output / (stem + ".bundle")
    code = args.output / (stem + ".hsaco")
    voxels = args.dimension ** 3
    command = [args.hipcc, "--genco", "-O3", "-std=c++17",
               f"--offload-arch={args.gfx}", f"-DBC_HIP_NVOX={voxels}",
               "-I", str(args.include), str(args.source), "-o", str(bundle)]
    if classify:
        command += [f"-DBC_HIP_CHANNELS={args.channels}",
                    f"-DBC_HIP_CLASSES={args.classes}",
                    f"-DBC_HIP_CLASSIFIER_BIAS={int(args.classifier_bias)}"]
    print("[hip-support-compiler] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    unbundle_object(args.hipcc, args.gfx, bundle, code)
    metadata = code_object_metadata(code.read_bytes())
    if metadata["target"] != args.gfx or metadata["entry"] != entry:
        raise ValueError(f"unexpected HIP support object metadata: {metadata}")
    manifest = [{
        "file": code.name,
        "symbol": entry,
        "grid": [(voxels + 255) // 256, 1, 1],
        "block": [256, 1, 1],
        "dynamic_lds_bytes": 0,
        "args": [
            {"role": "logits", "bytes": 18 * voxels * 4},
            {"role": "out", "bytes": 4 * voxels * 4},
        ],
    }]
    if classify:
        manifest[0]["args"] = [
            {"role": "activation", "bytes": args.channels * voxels * 2},
            {"role": "weight", "bytes": args.channels * args.classes * 2},
        ] + ([{"role": "bias", "bytes": args.classes * 2}]
             if args.classifier_bias else []) + [{"role": "out", "bytes": voxels}]
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    kind = "classifier" if classify else "tissue-prior"
    print(f"compiled HIP {kind} kernel for {args.gfx} -> {args.output}")


if __name__ == "__main__":
    main()
