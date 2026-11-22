# Changelog

All notable changes to BrainChop Core are documented in this file.

## [1.0.0] - 2026-11-22

First public release of **BrainChop Core** (`neuroneural/brainchop-core`), replacing the Python/tinygrad runtime (`brainchop==0.2.5`) with a zero-dependency native C/Metal/CUDA/HIP/CPU engine and WebGPU/WebGL2/WASM browser runtime.

### Highlights

- **v4.0 Gold & v3.5 Models:** Upgraded embedded checkpoints to **MindMap-18 v4.0 Gold** (`mindmap`), **MindSnap-104 v4.0 Gold** (`mindsnap`), and **MindMap Fast v3.5** (`mindmap-fast`), alongside modality-agnostic **MindGrab** (`mindgrab`).
- **8-Bit Activation Turbo Mode & Apple M1 Determinism Fix:** Enabled fast 8-bit activation execution by default on Apple Metal, Apple SME, and x86-64 AVX-512 VNNI / AVX-VNNI CPUs (with exact fp32/fp16 available via `--no-turbo`), and fixed an Apple7/M1 `simdgroup_barrier` threadgroup-memory ordering bug in `metal_turbo.h` (`threadgroup_barrier`), ensuring bit-exact run-to-run Metal Turbo determinism.
- **Coil-Flare-Safe Conform & Parity Cleanup:** Unified `f_high = 0.98` conform quantile scaling, `NaN`/`±Inf` sanitization, singular-affine guard, 8-sample majority native-space reslicing, and class-aware connected-component cleanup (`bwlabel_cleanup18`, `bwlabel_cleanup104`) across native and browser runtimes.
- **Portable Prebuilt Wheels & Zero-Dep Python API:** Added prebuilt `win_amd64`, `macosx_arm64`, and `linux_x86_64` wheels with portable ABI checks and zero ML dependencies, plus a backward-compatible stdlib-only `import brainchop` Python API (`Volume`, `load`, `segment`, `save`, `list_models`).
- **Unified Native CLI Dispatcher:** Renamed `brainchop-16chan18cls` to `brainchop-mindmap-fast` and added a tiny zero-dependency C `brainchop` dispatcher for Homebrew, macOS `.pkg`, and standalone Linux/Windows archives.
- **Pure-JS WebGPU V4 Browser Package (`@brainchop/core`):** Replaced `-gpu.wasm` binaries with a pure-JS + WGSL WebGPU V4 runtime supporting `shader-f16`/subgroups/f32 fallback and automatic `LowMem128` + paced Z-chunking for mobile and integrated GPUs, alongside Emscripten WebGL2 and threaded-CPU WASM fallbacks.
- **Explicit-Only OpenCL:** Removed OpenCL from automatic backend selection to avoid slow/TDR-prone integrated-GPU paths; OpenCL remains available via explicit `--backend opencl`.
