<p align="center">
  <a href="https://brainchop.org">
    <img src="https://raw.githubusercontent.com/neuroneural/brainchop-core/main/docs/assets/brainchop-icon.png" alt="BrainChop" width="140" />
  </a>
</p>

<h1 align="center">BrainChop Core</h1>

<p align="center">
  <strong>Fast, zero-dependency 3D brain segmentation and skull stripping for desktops, HPC clusters, Python pipelines, and web browsers.</strong>
</p>

<p align="center">
  <a href="https://github.com/neuroneural/brainchop-core/actions/workflows/ci.yml"><img src="https://github.com/neuroneural/brainchop-core/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/brainchop/"><img src="https://img.shields.io/pypi/v/brainchop" alt="PyPI" /></a>
  <a href="https://www.npmjs.com/package/@brainchop/core"><img src="https://img.shields.io/npm/v/@brainchop/core" alt="npm" /></a>
  <a href="https://brainchop.org"><img src="https://img.shields.io/badge/web%20app-brainchop.org-blue" alt="brainchop.org" /></a>
  <a href="https://github.com/neuroneural/brainchop-core/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" /></a>
</p>

Powered by four embedded 3D MeshNet models with **no PyTorch, TensorFlow, tinygrad, CUDA Toolkit, or Python ML stack required at runtime**.

- **Web App:** [brainchop.org](https://brainchop.org)
- **Repository:** [github.com/neuroneural/brainchop-core](https://github.com/neuroneural/brainchop-core)
- **License:** MIT (see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md))

## Embedded Models

All four models are compiled directly into self-contained binaries (no separate model downloads or cache directories):

| Model | CLI / Python name | Executable | Channels / Classes | Purpose |
| --- | --- | --- | --- | --- |
| **MindGrab** | `mindgrab` (`--skull-strip`) | `brainchop-mindgrab` | 15ch / 2-class | Modality-agnostic skull stripping & brain masking (`--mask`, `--border`, `--crop`) |
| **MindMap Fast** | `mindmap-fast` *(default)* | `brainchop-mindmap-fast` | 16ch / 18-class (v3.5) | Fast 18-class cortical & subcortical segmentation (`16chan18cls` alias) |
| **MindMap-18** | `mindmap` | `brainchop-mindmap` | 24ch / 18-class (v4.0 Gold) | High-accuracy 18-class segmentation + CAT-lite GM/WM/CSF tissue fractions (`--pve`) |
| **MindSnap-104** | `mindsnap` | `brainchop-mindsnap` | 24ch / 104-class (v4.0 Gold) | Full 104-class Desikan-Killiany cortical & subcortical parcellation |

## Quick Install

### Python / PyPI (Zero ML Dependencies)

```bash
pip install brainchop

# Or run immediately with uv (no virtualenv required)
uvx brainchop input.nii.gz -o labels.nii.gz
```

Prebuilt wheels (`win_amd64`, `macosx_arm64`, `linux_x86_64`) install in ~2 seconds with **zero runtime dependencies** (no PyTorch, no tinygrad, no CUDA Toolkit, no compiler needed). Source installs automatically detect your CPU/GPU (Apple Metal, NVIDIA CUDA Driver API, AMD ROCm HIP) and compile locally. See [docs/pypi.md](docs/pypi.md) for details.

### Homebrew (macOS & Linux)

```bash
brew install neuroneural/tap/brainchop
```

### Browser / JavaScript (npm)

```bash
npm install @brainchop/core
```

Pure-JS WebGPU V4 engine (with automatic low-memory staging for mobile & integrated GPUs), plus WebGL2 and multi-threaded WebAssembly CPU fallbacks. See [js/README.md](js/README.md) for browser API usage.

### Standalone Native Binaries (Zero Python)

Download signed macOS Apple Silicon `.pkg` installers or self-contained Linux / Windows archives from [GitHub Releases](https://github.com/neuroneural/brainchop-core/releases). Each archive includes the `brainchop` dispatcher plus all four standalone executables.

### Build from Source

```bash
make -C src -j4
```

Requires only a C11 compiler (`clang` or `gcc`) and system `zlib`. Builds all four model executables and the zero-dependency `brainchop` dispatcher.

## Usage

### Command Line

```bash
# Default 18-class fast segmentation (MindMap Fast v3.5)
brainchop input.nii.gz -o labels.nii.gz

# High-accuracy 18-class segmentation (MindMap-18 v4.0 Gold)
brainchop input.nii.gz -m mindmap -o labels.nii.gz

# Continuous GM / WM / CSF / Brain partial-volume maps (CAT-lite)
brainchop input.nii.gz -m mindmap --pve -o tissues.nii.gz

# 104-class Desikan-Killiany parcellation (MindSnap-104 v4.0 Gold)
brainchop input.nii.gz -m mindsnap -o parcels.nii.gz

# Skull stripping + binary brain mask (MindGrab)
brainchop input.nii.gz --skull-strip --mask brain_mask.nii.gz -o stripped.nii.gz

# Reslice label output back to original input voxel grid
brainchop input.nii.gz -m mindmap --inverse-conform -o labels_native.nii.gz

# List embedded models or inspect GPU adapters
brainchop --list
brainchop-mindmap -a -1
```

> **Output space:** The Python `brainchop` wrapper preserves historical `brainchop-cli` behavior (label outputs in conformed 256³ space by default; pass `--inverse-conform` for native input space). Standalone native executables (`brainchop-mindmap`, etc.) default to input space (pass `--save-conform` for conformed 256³ space). MindGrab always writes in input space.

### Python API

The `import brainchop` scripting interface requires **zero external packages** (`numpy` is optional; when installed, `vol.data` returns a `numpy.ndarray`):

```python
import brainchop as bc

print(bc.list_models())

vol = bc.load("input.nii.gz")
labels = bc.segment(vol, "mindmap")          # or pass a path / list of paths directly
bc.save(labels, "labels.nii.gz")

# Native input-space output
labels_native = bc.segment("input.nii.gz", "mindsnap", inverse_conform=True)

# CAT-lite GM/WM/CSF/Brain tissue fractions
tissues = bc.segment("input.nii.gz", "mindmap", pve=True)
bc.save(tissues["gm"], "gm.nii.gz")
```

### Browser / JavaScript (WebGPU + WebGL2 + WASM CPU)

Install `@brainchop/core` via npm (`npm install @brainchop/core`) — see [js/README.md](js/README.md) for the browser ES module package supporting client-side 3D segmentation on pure-JS WebGPU V4 (including automatic low-memory staging for mobile & integrated GPUs), WebGL2, and multi-threaded WebAssembly CPU.

## Compute Backends & Performance

`brainchop` automatically picks the fastest validated backend on your machine and falls back cleanly to multi-threaded CPU (`--no-turbo` switches from the default fast 8-bit activation path to exact fp32/fp16 math):

| Backend | Hardware | Auto-Selected | Notes |
| --- | --- | --- | --- |
| **CUDA** | NVIDIA GPUs (`sm60`–`sm90`+, Driver API >= 12.8) | Yes | Embedded offline fatbins — no CUDA Toolkit or `nvcc` required |
| **HIP** | AMD Radeon (`gfx1151` / ROCm) | Yes | Embedded HSACO artifacts — no `hipcc` required |
| **Metal** | Apple Silicon (macOS 13.2+) | Yes | 8-bit Turbo (~0.8–1.2 s for 24ch models on M-series) & exact fp16 |
| **CPU** | Apple SME / NEON, x86-64 AVX-512 VNNI / AVX-VNNI / AVX2+FMA | Yes | 8-bit Turbo & bit-exact fp32 across compilers/architectures |
| **OpenCL** | Explicit opt-in (`--backend opencl`) | No | Available via `-backend opencl` when explicitly requested |

## Documentation

- [Building, Backends, Testing & Packaging Reference](docs/building-and-testing.md) — full compiler flags, macOS `.pkg` notarization, CUDA/HIP offline artifact generation, WASM builds, and verification suites.
- [PyPI Packaging & Adaptive Hardware Selection](docs/pypi.md) — prebuilt wheels, source-first adaptive rebuilds, and offline installs.
- [Browser Package Documentation](js/README.md) — WebGPU, WebGL2, and threaded WASM CPU API.
- [Changelog](CHANGELOG.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

## Citation

If you use BrainChop in academic work, please cite the BrainChop project (see [CITATION.cff](CITATION.cff)):

```bibtex
@article{masoud2023brainchop,
  title   = {Brainchop: In-browser MRI volumetric segmentation and rendering},
  author  = {Masoud, Mohamed and Hu, Farfalla and Plis, Sergey},
  journal = {Journal of Open Source Software},
  volume  = {8},
  number  = {83},
  pages   = {5098},
  year    = {2023},
  doi     = {10.21105/joss.05098}
}
```
