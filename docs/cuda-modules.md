# Offline CUDA modules

The checked-in modules are the handwritten CUDA kernels, compiled with nvcc.
Each model has one fatbin, embedded via `payload.S`; generation does not expand
binary bytes into C source.

Generation, integrity checks and a standalone smoke runner accompany the
Driver API inference backend, enabled by default on Linux and Windows.
`CUDA=1` retains explicit legacy nvcc/cudart validation. macOS keeps Metal.

## Default inference without the toolkit

```sh
make -C src METAL=0 OPENCL=0 models
src/brainchop-mindgrab -backend cuda -o mask.nii.gz input.nii.gz
src/brainchop-mindmap-fast -backend cuda -o labels.nii.gz input.nii.gz
```

This build uses the ordinary C compiler and embeds each model's fatbin. It
does not probe or invoke nvcc, include CUDA SDK headers, or link cudart, NVRTC,
nvJitLink or an ML runtime. Running it requires a compatible NVIDIA driver;
the existing CPU path remains usable without that driver. Other normal build
dependencies, such as zlib, are unchanged. `CUDA=1` retains the legacy backend
for comparison, and `CUDA=0` excludes CUDA. Switching modes invalidates objects.

Both adapters include `backend_cuda_host.h`, which owns the existing padded
fp16 packing, affine upload, normalization, classifier, launch selection and
autotuning. The
Driver API adapter loads every catalog entry before allocating inference
buffers, binds the requested CUDA-visible ordinal, and owns a context/module
for each segmentation. Cleanup restores the caller's previous context. Like
the legacy adapter, backend calls must be serialized.

The same `--adapter`, `--tune`, `BRAINCHOPC_TUNE`, `BRAINCHOPC_CUDA_WMMA` and
`BRAINCHOPC_CUDA_PROFILE` controls apply. Driver tuning cache keys include the
device, driver, model, module ABI, a hash of the actual embedded bytes and
forced-JIT settings, so they cannot reuse legacy or different-module results.
Immediate launch failures remain failed even when subsequent synchronization
succeeds; unusable candidates and failed timing samples are excluded. Profiling
events are checked and totals reset for each segmentation.

An error preparing tuning input is reported there and its pending launch status
is cleared before fallback. Invalid scalar/WMMA dispatch fails explicitly in
both adapters. After checked label readback, module/context cleanup errors warn
without failing an otherwise successful segmentation; earlier inference errors
still fail the run.

Full-model validation, using separately built directories and a new output path:

```sh
make -C src CUDA=driver METAL=0 OPENCL=0 test test-cuda
python3 scripts/cuda_driver_ab.py --legacy /path/to/legacy/src \
  --driver /path/to/driver/src --output /tmp/cuda-ab-new --pairs 7
```

The A/B harness checks exact decompressed output equality on all four models and
both tracked inputs, both parcellation models in native/conformed output, scalar/WMMA execution,
and forced/cached tuning for both families. It records executable/output hashes
and all timings; `--pairs 0` runs correctness checks only. Match host compilers,
flags and native CUDA targets when attributing timing differences. The existing
CPU reference and GPU tolerance checks remain separate and unchanged.

`make test` also runs a GPU-free contract test for module resolution/cleanup,
adapter binding, pointer/integer launch arguments and injected launch/timing
failures, including initial tuning-fill recovery and cleanup-only errors.
With `CUDA=1`, it also compiles and runs invalid legacy dispatch checks for all four
models (nvcc required; no GPU required). `make native-package` builds audited
Linux/Windows archives. The manual CI `cuda_only` input exercises default builds
with a poison nvcc, CPU regression, import auditing, and extracted-archive
execution without a driver. Real Windows NVIDIA execution remains a separate
hardware gate; hosted Windows CI has no NVIDIA GPU.

### Coverage

ABI/schema 2 carries per-model geometry and a tissue-prior head; mindmap and
mindsnap use stride 24, the 16-channel models stride 16. All four models run
through the toolkit-free backend; CUDA PVE is mindmap-only.

Native-code execution has been verified on sm70–sm90 (RTX 2080 Ti, A2000, V100,
A100, A40, L40, H100). sm60 and newer-than-sm90 devices are untested.

## Generation and compatibility

```sh
make -C src cuda-modules-generate NVCC=/path/to/cuda-12.8/bin/nvcc
make -C src cuda-modules-check
```

Generation requires **nvcc V12.8.93** and the accompanying cuobjdump, but no GPU.
All four models must compile and pass export validation before replacing checked-in
artifacts. Ordinary builds never run the generator. `check` uses Python's
standard library and needs neither CUDA headers, nvcc nor a driver. It checks
source/model hashes, binary hashes, the catalog, compiler settings and ABI.
`make test` includes these checks and their corruption/drift regression tests.

The artifact profile is fixed and fails rather than silently dropping targets:

- Native code: sm60, sm70, sm75, sm80, sm86, sm89, sm90.
- Forward fallback: compute90 PTX 8.7. The driver selects native code where
  compatible and can JIT this PTX on later compatible architectures.
- 64-bit host/device pointer ABI; conservative minimum driver API version
  **12080 (CUDA 12.8)** for this artifact set. No claim is made for older drivers
  via CUDA minor-version compatibility. This is a driver requirement, not a
  user-side toolkit installation requirement.
- WMMA is valid on sm70 and later. The sm60 image retains the compiled-code
  probe and empty WMMA definitions, as the runtime backend does. A runner must
  consult the probe before selecting WMMA; device capability alone is not enough
  when testing deliberately restricted modules.

Changing the pinned compiler, architecture profile, fast-math flags or kernel
source requires regenerating and revalidating all four models. nvcc is used only on
maintainer machines; the deployed module needs no NVRTC, nvJitLink, Python,
libcudart or separate device-math library.

## Export contract

`cuda_module.cu` includes `cuda_kernels.cuh` in export mode and explicitly
instantiates **all** variants from `cuda_variants.h`. Production builds retain
their original internal linkage. Kernel bodies and arithmetic are shared;
there are no forwarding device calls or rewritten convolutions.

The eight non-template kernels have C entry names. The templated convolution
symbols use nvcc's C++ ABI; their exact names are recorded in the generated
catalog alongside their stable logical identifiers (`scalar_VPT_THREADS` and
`wmma_WARPS`). A consumer uses that catalog rather than guessing mangled names.
The data-only `cuda_geometry.h` and `cuda_geometry_mindmap.h` contracts supply
model-specific launch/layout constants to
the generated catalog and manifest. The driver consumes that generated geometry;
module compilation asserts that every corresponding device constant agrees.
Changing the contract requires regeneration, and an inconsistent compiler
override fails at compile time, including reduction block/shared-array sizes.
The generator checks the emitted PTX entry names and parameter widths against
the contract and checks every native architecture for the complete kernel set.
It rejects unresolved external PTX functions, so device math is resolved offline.

`manifest.json` records schema/ABI versions, source hashes (weights excluded: they upload at run time), flags,
architecture/PTX versions, geometry, ordered argument types and artifact sizes/hashes.
Arguments are device pointers (64-bit), `size_t` (64-bit), or `int` (32-bit), in
the order declared by `cuda_kernels.cuh`. Increment the ABI version for an
incompatible signature or semantic change. Launch dimensions, shared memory,
weight packing and tuning remain responsibilities of the inference runner;
these modules do not define a new graph or tensor layout.

## Driver-only smoke test

```sh
make -C src CUDA=0 METAL=0 OPENCL=0 cuda-module-smoke
# Choose another CUDA-visible ordinal:
make -C src CUDA=0 METAL=0 OPENCL=0 cuda-module-smoke CUDA_MODULE_ADAPTER=1
```

The test is compiled as C, loads `libcuda.so.1` / `nvcuda.dll` dynamically and
resolves every catalog entry. It generates a nonzero input, checks scalar and
WMMA center-tap identity convolutions exactly, verifies the checksum kernel,
and checks the WMMA compiled-code probe. It allocates about 512 MiB of device
memory because the kernel reads use the actual 256-cubed model strides.
Absent drivers/devices and failed checks are failures, not passing skips.

For build/embedding validation on a machine without NVIDIA hardware:

```sh
make -C src CUDA=0 METAL=0 OPENCL=0 cuda-module-smoke-build
/tmp/cuda-module-smoke-mindgrab --embedded-only
/tmp/cuda-module-smoke-model16chan18cls --embedded-only
```

Paths above use the Unix default `BC_TMP=/tmp`; override `BC_TMP` for an isolated
run. Windows uses `build/tmp` and the `.exe` suffix. Embedding-only validation
checks the linked blob's framing and does not substitute for a GPU smoke test.
The smoke test is not a full-model inference/parity gate; use the inference
tests and A/B harness above for that.

References: [nvcc device-only compilation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html),
[Driver API module loading](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).
