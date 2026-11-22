# Offline MindMap tuning

This is maintainer tooling for MindMap CUDA kernel tuning. Search
results remain proposals until validated with the release artifact and PVE.

`scripts/cuda_offline_tune.py` can compile isolated MindMap candidates with
different nvcc register caps, or inspect a supplied executable. Each run
requires an explicit accuracy budget, compares full conformed output to a CPU
reference on every input, and records output hashes, per-layer convolution
timings, compiler/GPU identity and sampled process VRAM. Within one arithmetic
family, changing launch geometry must preserve the output exactly.

The script searches the existing scalar shapes and the portable MindMap WMMA
warp counts. It then combines measured layer choices, validates the combined
schedule, and measures that schedule again with profiling disabled. Profile
events serialize execution and are used for selecting kernels, not reporting
end-to-end speed. The first run is excluded from timing summaries. The final
report includes per-input median CLI time and the speed/memory frontier. The
final comparison alternates the selected schedule with the compiled default,
with profiling disabled for both. The measured default remains eligible for
recommendation, so a slower combined schedule cannot displace it. A `null`
recommended schedule means to leave `BRAINCHOPC_CUDA_LAYERS` unset.

For a short search over a previously built binary:

```sh
python3 scripts/cuda_offline_tune.py \
  --binary /path/to/src/brainchop-mindmap --adapter 1 \
  --candidates s3x512,w4,w8,w16 --repeats 3 \
  --max-label-differences 1024 --output /tmp/mindmap-search-new
```

For compiler experiments on an sm86 machine with the pinned compiler:

```sh
python3 scripts/cuda_offline_tune.py \
  --nvcc /usr/local/cuda/bin/nvcc --arch sm_86 --adapter 1 \
  --register-caps 0,64,96,128 --repeats 3 \
  --max-label-differences 1024 --output /tmp/mindmap-compile-search-new
```

The compiler-search mode currently targets Linux with GCC and the legacy CUDA
adapter. Its executables require that host's CUDA runtime. This does not change
the ordinary release's Driver API backend or introduce a user-side toolkit
dependency. A register cap is an experiment, not a guarantee of different or
better code; nvcc launch-bound constraints may take precedence.

Use `--input` repeatedly to test additional inputs. `--memory-budget-mib`
constrains the recommendation using sampled process VRAM. Sampling can miss a
short peak; use a profiler for a strict memory bound. Missing memory samples are
reported as unknown and cannot satisfy an explicit memory budget. The script
creates a new evidence directory and never changes production artifacts,
source or the user's tune cache. It does not allocate cluster jobs itself.

`BRAINCHOPC_CUDA_LAYERS` accepts a single `s3x512` or `w8` token for a uniform
choice, or one comma-separated token per layer. A uniform WMMA choice keeps
the scalar first layer. An explicit full schedule must use scalar execution
for layer zero. Invalid configurations, unsupported WMMA hardware and shapes
exceeding the portable shared-memory budget are rejected. The override bypasses
the normal tuning cache and is intended for this maintainer workflow.

Before promoting a result, rebuild it as a pinned, hashed Driver API artifact
and repeat full label and PVE validation, native/conformed outputs, both
fixtures and the relevant architecture matrix. The search currently validates
labels; successful search results do not establish PVE correctness. Use
`test/cuda_pve_check.py --schedule TOKEN` (or `--scalar`) to check fractions,
with `--models mindmap` for a 13-layer schedule. Keep the
existing models' regression contracts and the portable fallback. New tile
designs, streaming layouts and architecture-specific instructions can extend
the search later, with the same numerical and memory gates.

Compilation cannot guarantee a globally fastest or smallest implementation.
Register pressure, occupancy, instruction count and spilling trade off against
one another; `--maxrregcount` does not override kernels with launch bounds.
See NVIDIA's [compiler documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
and [register/launch-bound guidance](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#launch-bounds).
This search changes register caps and launch schedules, not the model's weights,
precision, activation layout or tensor-core tile design. Those require separate
experiments and accuracy validation. The compact stride of 24 is what saves
512 MiB of activation storage compared with padding MindMap to 32 channels.

The initial release-artifact experiments found an all-scalar `s3x512` schedule
about 5% faster end to end on A100/H100 for the two fixtures. Register cap 64
showed no clear extra gain and no lower sampled VRAM. This is an optional
measured schedule, not a replacement default.
