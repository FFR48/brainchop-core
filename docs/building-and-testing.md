# BrainChop Core — Building, Backends & Testing Reference

Portable C reimplementation of the brainchop MeshNet models: MindGrab
skull-stripping, `mindmap-fast` (`model16chan18cls`) 18-class segmentation, `mindmap`,
the same 18 classes from a wider 24-channel network, and `mindsnap`,
a 104-class Desikan-Killiany parcellation. The inference core is lean
C with optional platform GPU backends; no Python or machine-learning runtime is
required.

```sh
cd src
make models

./brainchop-mindgrab -o stripped.nii.gz input.nii.gz
./brainchop-mindmap-fast -o labels.nii.gz input.nii.gz
./brainchop-mindmap -o labels.nii.gz input.nii.gz
./brainchop-mindmap --pve -o brain.nii.gz input.nii.gz   # GM/WM/CSF fractions
./brainchop-mindsnap -o atlas.nii.gz input.nii.gz
```

`output.nii.gz` is the default output, so the shortest invocation is:

```sh
./brainchop-mindgrab input.nii.gz
```

## Python packaging

The `brainchop` Python package provides prebuilt wheels and source-first builds
with the `brainchop` CLI (defaulting to `mindmap-fast`) and zero-dependency
`import brainchop` Python API. See
[the installation design, compiler prerequisites and adaptive runtime cache](pypi.md).

## Command line

There is one executable per model, and each contains its model's weights. There
is no model to select, no models directory to install, and nothing to keep
beside the binary. Default Linux and Windows builds also embed the CUDA kernels;
default Linux builds embed gfx1151 HIP kernels for every model but
mindsnap as well (CUDA covers all four).
NVIDIA execution needs a compatible driver, without the CUDA toolkit or runtime.
AMD execution needs the ROCm HIP runtime, but no compiler, Python, MIGraphX or
tinygrad.
Windows builds statically link the ordinary support libraries. Linux builds use
system glibc and zlib. OpenCL loads its driver dynamically when used. Explicit legacy
`CUDA=1` Windows builds retain a companion CUDA DLL for validation.

```text
brainchop-mindgrab       skull stripping in any modality
brainchop-mindmap-fast    18-class brain segmentation
brainchop-mindmap        18-class brain segmentation (24-channel, gDice-priority)
brainchop-mindsnap       104-class Desikan-Killiany parcellation (24-channel, infant refit-synth)
```

`mindsnap` is mindmap's backbone with a 104-class head and the infant
refit-synth weights from `brainchop-test`. It reproduces `brainchop-cli`
(tinygrad) — including its 2/98 quantile normalization, which differs from the
browser app's — to 0 and 2 voxels on the two test inputs, the two being fp32
ties. Backends: CPU, CUDA, Metal, and in the browser WebGPU, WebGL2 and wasm CPU; there is
no OpenCL or HIP module for it. CPU segment time is about 5% above mindmap's.

`mindmap` is `model16chan18cls`'s architecture at 24 channels rather than 16 —
the same 13 hidden layers, the same dilation ladder, the same 18 classes and the
same colormap — so it is the wider and slower member of that pair, not a
different task. On an Apple M4 Pro over `test/t1_crop.nii.gz`, an idle machine:
3.6 s against 6.5 s on the CPU backend and 1.03 s against 1.69 s on Metal.
mindmap with `--pve` takes 1.87–1.89 s on Metal against 7.0–7.6 s on CPU
(three 1 mm scans, alternated), peaking at ≤ 4.0 GB. Metal's default 8-bit
turbo path (below) runs mindmap in about 0.84 s of GPU time against 1.27 s for fp16,
peaking near 1 GB rather than 3.2–4.1 GB. In the browser,
`segmentTissues` on `t1_crop` takes 2.75 s on WebGPU, 5.8 s on WebGL2 and
12.5 s on wasm CPU (Chrome).

`--version` (or `-v`) prints just the version, e.g.
`brainchop-mindgrab 1.0.0`. Any of the four with no arguments, or with
`--help`, also prints it on the description line — `BrainChop mindgrab: skull stripping in any modality
(v1.0.0)`. Every run also prints it beside the backend it selected, which
is the line to quote in a bug report. `0.1` means the CLI is not yet promised to
be stable; the rest is the build date.

Common to all four:

```text
-h, --help
-o, --output PATH
--ct
--comply
--export-classes
-cpu N
-backend auto|cpu|cuda|hip|opencl|metal
```

`--help` lists only the backends the executable was actually built with, so the
line above is the union rather than what any one binary prints. The two browser
backends, `webgpu` and `webgl2`, are selectable the same way but exist only in
the WebAssembly modules.

Offered only where the model supports them, and **absent from `--help`**
otherwise, so an executable never advertises an option it would refuse:

```text
brainchop-mindgrab       --crop [PCT]   --mask [PATH]   --border MM
brainchop-mindmap-fast    --save-conform
brainchop-mindmap        --save-conform   --pve   --legacy-cleanup
brainchop-mindsnap       --save-conform
```

`--pve` (long form `--partial-volume-estimate`) is an explicit model capability
offered only by MindMap. Sharing MindMap's 18-class output count does not make
model16chan18cls PVE-compatible. CPU, CUDA, HIP, Metal, WebGPU and WebGL2
support this mode, and automatic selection on macOS picks Metal. GPU backends run the network and tissue-prior head; the CAT-lite fit
remains on CPU. Automatic selection uses CPU when the selected GPU backend has
no tissue-prior head.

Offered only when the build contains a GPU backend whose kernel shape can be
measured, which means CUDA and OpenCL:

```text
--tune
```

```text
--no-turbo
```

Turbo is the default on the CPU, Metal and WebGPU; `--no-turbo` selects the
exact CPU engine or the fp16 GPU path. Metal and WebGPU default to an 8-bit **turbo** path (`src/metal_turbo.h`,
`src/webgpu_turbo.h`, ported from brainchop-test's V4 engine): square-law u8
activations, fused moments, fp16 matrix operands with fp32 accumulation on
Metal, and a background row skip (`src/turbo_cone.h`; `make TURBO_CONE=0`
computes every row). It agrees with the CPU on 99.2–99.98% of foreground
voxels on the T1 fixtures, but less on non-T1 scans (CT, 7T, BOLD), where
fp16 reaches 99.7–99.9%. `--no-turbo` selects the fp16 path; `--pve` always
uses it. Other GPU backends are unchanged.

The native arm64/x86 CPU turbo engine uses
compact hidden convolutions (`src/meshnet_turbo.h`). Supported Apple SME CPUs
(tested on M4 Pro) use fp16 operands with fp32 accumulation. Other capable
CPUs use coarse/residual activation bytes and hi/lo s8 weights through Arm
i8mm or x86 VNNI. CPUs without SME or either integer ISA (e.g. M1) and wasm run exact.
Both turbo paths agree with exact output on 99.95–100% of foreground voxels
on the T1 fixtures. `--crop`, `--export-classes` and `--pve` always run
exact. Full-volume exact inference (`--no-turbo`) also uses SME on supported
Apple CPUs, with fp32 operands and bit-exact results; cropped regions retain
NEON. On M4 Pro (`t1_crop`, all threads) SME exact is 1.6–2.2x faster than
NEON exact and turbo only 1.1–1.3x faster than SME exact, using ~22% less
peak RAM; int8 turbo (M2–M3) is 1.1–1.4x faster than NEON exact.
`BRAINCHOPC_CPU_SME=0` disables SME in both paths.

The CUDA backend measures its convolution launch shape the first time it meets a
device -- six warps-per-block candidates on a device with tensor cores, eighteen
(voxels-per-thread x block size) on one without -- and caches the winner under
`$XDG_CACHE_HOME/brainchop/tune.txt` (or `~/.cache/brainchop/tune.txt`), keyed on
device, compute capability, driver, model, the compiled architecture list and
which of the two kernels it measured. That happens automatically and takes a few
seconds, once. `--tune` forces a re-measure -- use it if the first run happened
on a busy machine. `BRAINCHOPC_TUNE=0` skips both the cache and the search and
pins the compiled-in shape, which is what you want when comparing two builds
rather than two devices, and what CI wants where no cache is writable.

Every candidate shape must reproduce the reference output bit-exactly before it
can win, so tuning changes speed and never results -- `make test-cuda` and
`make test-opencl` assert that from outside, comparing a tuned run against one
pinned with `BRAINCHOPC_TUNE=0`. On a device with tensor cores the search covers
the tensor-core kernel's warps-per-block; below that it covers the scalar
kernel's voxels-per-thread and block size. `BRAINCHOPC_CUDA_PROFILE=1`
reports per-stage GPU time if you want to see where it went.

Passing an option a sibling executable has produces a message naming the
reason, rather than a bare "unrecognized argument" -- carrying a command line
between them is the likeliest way to hit it:

```console
$ brainchop-mindmap-fast --crop scan.nii.gz
brainchop: --crop is not available here: this model's receptive field spans the whole volume, so cropping
  would feed it mostly zero padding

$ brainchop-mindgrab --pve scan.nii.gz
brainchop: --pve is not available here: CAT-lite tissue fractions are supported and validated only for MindMap
```

### Output space

**Every model writes its output in the input image's own space by default.**
For MindGrab that is inherent: its output *is* the input image with non-brain
voxels removed, at the input's resolution and datatype. For label models the
output is resliced back to the input grid using eight-sample majority voting,
including background samples and first-sample tie breaking, retaining
`NIFTI_INTENT_LABEL` (1002); `--save-conform` keeps the conformed 256³ uint8
volume instead.

This differs from brainchop-cli, where conformed space is the default and
`--inverse-conform` opts into the input grid. The flag has been inverted rather
than renamed, on the grounds that a label map aligned to the image the caller
supplied is what a caller almost always wants. Scripts written against the old
CLI get a message pointing at the replacement:

```console
$ brainchop-mindmap-fast --inverse-conform scan.nii.gz
brainchop: --inverse-conform is not available here: reslicing to the input grid is now the default; use --save-conform
  to keep the conformed 256^3 volume instead
```

MindMap retains PR #16's default speckle cleanup and its `--legacy-cleanup`
option. The other models retain the largest 26-connected nonzero component.
Categorical outputs are checked before cleanup: invalid labels, empty or more
than 95% filled foreground, and excessive fragmentation fail inference rather
than being saved as successful results. Automatic backend selection can retry
on CPU; an explicitly requested GPU reports failure.

`--pve` is the exception to all of the above: its four fraction maps are always
written in the **input image's own space** by default and in the conformed
256³ 1 mm grid with `--save-conform`, exactly as the label output is. The fit
runs on the conformed volume the network saw and the fractions are resliced
afterwards with linear interpolation,
because a tissue fraction is continuous. No label volume is written at all.

### Option notes

- `--ct` applies the reference Hounsfield-to-Cormack transform to the private
  inference image. It never changes the intensities saved by MindGrab. It is
  available to every model, matching brainchop-cli, which passes it to `load()`
  regardless of model.
- `--comply` inserts the reference 256-cubed, 1 mm RAS compliance transform
  before the ordinary model conform.
- `--crop [PCT]` runs inference over the reference percentile bounding box; the
  default percentile is 2.0. The CPU kernels use the cropped convolution
  boundaries and GroupNorm population rather than approximating a crop by
  zeroing a full volume. It is **not offered for `brainchop-mindmap-fast`**,
  whose receptive field is 255 on a 256³ volume: cropping feeds the
  large-dilation layers mostly zero padding, measured at 1.0% of labels changed
  at the default 2% crop and unbounded on tighter crops. Upstream likewise
  states this model family cannot crop. `brainchop-mindmap` shares that dilation
  ladder and likewise does not offer it.
- `--mask [PATH]` writes MindGrab's binary original-grid uint8 mask. With no
  path, it writes `mask.nii.gz`.
- `--border MM` grows that mask by an exact Euclidean physical distance before
  it is returned to the original grid. MM is a nonnegative integer.
- `--export-classes` writes conformed-space float32 raw logits as
  `<output-base>_cN`, carrying over the extension of `--output` (so a `.nii.gz`
  output yields compressed class maps). Maps are generated and written one at a
  time, so the option adds one 256-cubed float buffer rather than retaining
  every class. Note the volume: 18 classes at 256-cubed float32 is ~1 GB.
- `--pve` replaces the crisp label volume with CAT-lite partial-volume
  estimation: the 18 classes are grouped into GM, WM and CSF priors, blurred,
  gated by the brain support, and fitted against the normalized intensity to
  give continuous tissue fractions. It writes **four** files derived from
  `--output` rather than one, so `-o brain.nii.gz` produces `brain_gm.nii.gz`,
  `brain_wm.nii.gz`, `brain_csf.nii.gz` and `brain_brain.nii.gz` (GM + WM,
  whose 0.5 isosurface is a sub-voxel pial surface). Each is uint8 with `scl_slope`
  1/255, so a reader recovers [0, 1] — a deliberate 4× size cut, since 1/255 is
  finer than the fit's own agreement with the reference implementation. A fit
  that refuses (too few confident voxels, implausible tissue ordering) falls
  back to the support-gated priors rather than failing the run. `--pve` accepts
  CPU, CUDA, HIP, Metal, WebGPU or WebGL2 and cannot be combined with
  `--export-classes`. An explicit
  GPU backend without a tissue-prior head is refused.

MindMap label output now uses the browser's 18-class cleanup with neighbor
refill enabled: keep the six-connected brain, retain the largest component of
classes 1, 2, 5, 6 and 13, and repaint suppressed components with the most common
surviving label on their face boundary. Other classes retain all components
inside the brain. `--legacy-cleanup` restores the previous 26-connected
whole-brain filter. This flag is MindMap-only and cannot accompany `--pve` or
`--export-classes`; tissue fractions and raw logits have separate processing.
The browser and native runtimes both enable this refill by default (`--legacy-cleanup`
/ `legacyCleanup: true` restores whole-brain-only filtering).

The reference Python release currently crashes in `--export-classes` after
inference because it asks a label image for a class axis. brainchop implements
the intended raw-logit export.

## Building on macOS

The default macOS build includes multithreaded CPU support and the native Metal
backend. OpenCL is deliberately not included by default.

```sh
cd src
make
```

There is no dependency to install first — threading is a pthread pool, which
macOS provides in libSystem.

Useful variants:

```sh
make METAL=0             # CPU-only executables
make THREADS=0 METAL=0   # serial CPU-only executables
make OPENCL=1 METAL=0    # optional OpenCL compile check; not a supported default
make ZLIBNG_ROOT=/path/to/zlib-ng   # opt-in faster gzip, see below
```

### Faster gzip, optionally

The default links the system zlib, which every supported platform provides.
Compressed output is a real share of a GPU run — on an Apple M4 Pro, MindGrab
through Metal with a `.nii.gz` in and out, the *write + gzip* stage costs
67.9 ms of a 1.55 s run against 16.0 ms to read and inflate the input. (Those
are whole stages, including the 64 MB of I/O, not the library call alone.)

`ZLIBNG_ROOT` links [zlib-ng](https://github.com/zlib-ng/zlib-ng) statically
instead, taking that stage to **21.6 ms — 3.1× — and a 0.9% smaller file**, for
+71 KB per executable. Those figures, and the option itself, have only been
exercised on macOS/arm64; nothing about it is macOS-specific, but treat it as
untested elsewhere. It must be a `ZLIB_COMPAT` build — an installed
zlib-ng from Homebrew or a distro is *not* drop-in, because those use
`zng_`-prefixed symbols:

```sh
# from anywhere; keep the two checkouts straight
git clone https://github.com/zlib-ng/zlib-ng
cd zlib-ng
```

On macOS, **export** the deployment target before configuring zlib-ng — a prefix
binds to `./configure` alone, which does not record it, so `make` then compiles
at the host's default:

```sh
export MACOSX_DEPLOYMENT_TARGET=13.2
./configure --zlib-compat && make
```

On other platforms, configure and build without that export:

```sh
./configure --zlib-compat && make
```

Then build brainchop-core from its own checkout:

```sh
ZNG=$PWD
cd /path/to/brainchop-core && make -C src ZLIBNG_ROOT=$ZNG
```

Getting this wrong costs warnings, not a failed build: you get a wall of
`ld: warning: object file was built for newer 'macOS' version`, the link
succeeds, and the executable's own `minos` is still correct — so
`package_macos.sh` does *not* catch it. Heed the warnings rather than filtering
them.

Switching in or out of `ZLIBNG_ROOT` rebuilds automatically — the selection is
part of the build stamp, and replacing the archive relinks — so no `make clean`
is needed.

It is opt-in and not the default for two reasons: it reintroduces a
from-source dependency, and `test/baseline.sh` compares output files by
SHA-256, so a different deflate implementation changes every `.nii.gz` case's
hash even though the decompressed bytes are identical. `make test` is
unaffected — it compares decompressed content, and passes against a zlib-ng
build.

`.nii.zst` input and output are built in when `pkg-config` finds libzstd
(`ZSTD=0` disables). That links libzstd dynamically, so release packages are
built with `ZSTD=0`.

Only modern Apple Silicon is targeted by the Metal implementation.

The Metal backend runs fixed-shape MSL programs generated from tinygrad's
reference model graphs. It embeds 15 unique MindGrab kernels, 17 unique
kernels each for model16 and mindmap, and 43 for mindsnap (the
classifier's `--fuse-chunk` chunking materializes more distinct kernels for
the 104-class head), plus compact dispatch tables; Python and tinygrad are not
runtime or build dependencies. The checked-in generated headers use BEAM=3 for
all four models, so a plain `make` builds every optimized model without
invoking Python or tinygrad.
`scripts/export_tinygrad_metal.py` is only needed to retune them after changing
the model graph, tinygrad version, or target machine.

With tinygrad and brainchop available in `python3`, regenerate all four headers
at their default BEAM=3 settings, then rebuild:

```sh
make metal-beam
```

The models can be tuned independently, including different search widths:

```sh
make metal-beam MINDGRAB_BEAM=0 MODEL16_BEAM=3 MINDMAP_BEAM=3 MINDSNAP_BEAM=3
make metal-regenerate-mindgrab MINDGRAB_BEAM=3
make metal-regenerate-model16 MODEL16_BEAM=0
make metal-regenerate-mindmap MINDMAP_BEAM=0
make metal-regenerate-mindsnap MINDSNAP_BEAM=0
```

model16, mindmap and mindsnap share a topology, differing only in
channel width (16 vs. 24) and class count (18 vs. 104), so the exporter covers
all three from one branch. It builds them from
`model.safetensors`, **not** from the `model.bin` the CPU engine compiles in:
the fp16 checkpoint is rescaled per layer relative to the payload, which
GroupNorm absorbs exactly, so each is the same model as its `model.bin` with
activations kept in fp16 range.

Set `TINYGRAD_PYTHON=/path/to/python` when tinygrad lives in another virtual
environment. Regeneration overwrites the tracked generated headers and writes
its inspectable JSON manifests under `/tmp` by default; override that location
with `METAL_MANIFEST_DIR`. BEAM results are hardware- and tinygrad-version-
specific, and uncached searches can take several minutes.

The generated paths use fp16 inputs, weights, convolutions, and stored
activations. GroupNorm performs its reduction and arithmetic in fp32 to avoid
fp16 square overflow. MindGrab's reference fused classifier retains its fp32
weights and trained final bias; the two 18-class models' classifiers are fp16.
Metal shader fast-math is disabled to match tinygrad's generated program.
This describes the `--no-turbo` / `--pve` path; the default turbo kernels are
hand-written (`src/metal_turbo.h`).

## Creating a notarized macOS installer package

The release target builds an Apple Silicon `.pkg` that installs one executable
per model into `/usr/local/bin`, which is on the default `PATH`, so the tools
work immediately after installation with no shell configuration.

Each installed executable is a single self-contained file. Its model weights are
compiled in and its threading is pthreads from libSystem, so nothing outside
macOS itself is required at run time -- no Homebrew, no bundled `lib/`
directory, no `models/` folder to keep beside it.

You need Xcode's command-line tools and **two** certificates with their
private keys in your Keychain:

| certificate | signs |
| --- | --- |
| Developer ID **Application** | the executables |
| Developer ID **Installer** | the `.pkg` |

These are separate certificates from the same Apple Developer account, and they
are not interchangeable: a package signed with the Application certificate is
rejected by the installer. Note that an Installer certificate does not appear
under `-p codesigning`, so list them without that filter:

```sh
security find-identity -v
```

Store notarization credentials once. This command prompts securely for the
Apple ID's app-specific password and saves it directly to Keychain; never put
the password in a Make variable, script, or chat message.

```sh
cd src
make macos-notary-profile \
  APPLE_ID='developer@example.com' \
  TEAM_ID='ABCDE12345'
```

Then build, sign, notarize, staple, and Gatekeeper-check the release in one
command:

```sh
make macos-release \
  VERSION=0.1.0 \
  MACOS_SIGN_IDENTITY='Developer ID Application: Your Name (ABCDE12345)' \
  MACOS_INSTALLER_IDENTITY='Developer ID Installer: Your Name (ABCDE12345)'
```

The finished artifact is `dist/brainchopc-0.1.0-macos-arm64.pkg`.
`NOTARY_PROFILE` defaults to `brainchopc-notary`. For a local packaging test
that cannot be notarized, use:

```sh
make macos-pkg-adhoc VERSION=0.1.0
```

Before packaging, the script rejects any executable that carries a dependency
outside `/usr/lib` and `/System`, or whose minimum macOS differs from the
deployment target -- a shipped binary that needs a library the user does not
have is the failure this whole arrangement exists to prevent.
`scripts/verify_macos_pkg.sh` then expands the finished package **without
installing it** and checks that each executable is correctly signed, carries no
foreign dependency or build-machine path, and actually runs. The notarization
step uses `notarytool --wait`, staples the accepted ticket, then validates with
`stapler` and Gatekeeper's `install` policy. See Apple's
[notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow)
and [Developer ID certificate instructions](https://developer.apple.com/help/account/certificates/create-developer-id-certificates/).

## Building on Linux

On Linux, a plain `make` builds the optimized multithreaded CPU backend, the
CUDA Driver API backend, and HIP where a checked-in model/architecture artifact
exists. OpenCL is enabled by default only when `pkg-config --exists OpenCL`
succeeds; otherwise pass `OPENCL=1`. It needs no headers or link library and
loads `libOpenCL.so.1` at run time. No CUDA toolkit, headers, nvcc, cudart,
NVRTC or nvJitLink is needed to compile or run the default build.

Threading is a pthread pool provided by glibc. For Debian or Ubuntu:

```sh
sudo apt install build-essential zlib1g-dev pkg-config
make -C src
# Optional OpenCL backend, no SDK headers or link library needed:
make -C src OPENCL=1
```

Clang is preferred when it can link with `-flto`; otherwise the build falls
back to `cc`/`gcc`. Set `CC` to override this choice.

NVIDIA execution requires a driver exposing CUDA Driver API **12080 or newer**.
The embedded modules contain native sm60/70/75/80/86/89/90 kernels and compute90
PTX for forward JIT. They preserve main's handwritten CUDA kernels and tuner.
Pascal is the architecture floor. Without an available compatible driver,
automatic selection uses another available backend or CPU; `-backend cuda`
requires CUDA and fails if unavailable. See [module validation and compatibility](cuda-modules.md)
for tested devices and the remaining limits, including newer GPUs.

```sh
make -C src CUDA=driver OPENCL=1   # explicit default CUDA plus OpenCL
make -C src CUDA=0 OPENCL=1        # OpenCL plus CPU
make -C src CUDA=0 HIP=0 OPENCL=0  # CPU only
```

The HIP loader is SDK-free and links no ROCm library at build time. It loads
`libamdhip64.so` at runtime and selects an artifact only for an exact base gfx
architecture match. The checked-in gfx1151 profiles cover MindGrab,
`model16chan18cls` and MindMap; another AMD architecture is a separate offline
artifact, not a forward-compatible fallback. Use `HIP=0` to omit the loader or
`HIP_ARCH=gfx...` to select another checked-in artifact. See
[offline HIP artifacts](hip-artifacts.md).

**`brainchop-mindmap` supports CUDA and gfx1151 HIP**, including `--pve`. Its
embedded module uses a compact 24-channel stride; the other two models retain stride 16.
The tensor-core kernel handles the channel tails explicitly. OpenCL still has
a 16-channel ceiling, so MindMap builds with CPU and CUDA on Linux/Windows.
GPU PVE computes tissue priors on the selected GPU and runs the existing CAT-lite fit on CPU.

```sh
src/brainchop-mindmap -backend cuda --pve -o brain.nii.gz input.nii.gz
src/brainchop-mindmap -backend hip --pve -o brain.nii.gz input.nii.gz
```

On x86-64 the CPU backend targets `x86-64-v3` (AVX2 + FMA), requiring Haswell
or Zen 1 or newer. Use `X86_ISA=` for a baseline CPU binary, or
`X86_ISA=x86-64-v2` for the no-FMA middle ground.

### Maintainer CUDA builds

`CUDA=1` explicitly selects the legacy nvcc/cudart backend for comparisons.
Only this mode probes nvcc and accepts `CUDA_ARCH`/`CUDA_GENCODE` to change the
compiled architectures. Those flags do not change the default embedded modules.

```sh
make -C src CUDA=1 OPENCL=0 CUDA_ARCH=sm_86
make -C src cuda-modules-generate NVCC=/path/to/pinned/nvcc
make -C src cuda-modules-check
```

Regeneration uses pinned nvcc 12.8.93 for all four models and requires revalidation;
see [offline CUDA modules](cuda-modules.md). Ordinary builds never regenerate
artifacts. CUDA 13 legacy builds have a narrower architecture floor; changing
the local legacy compiler does not change the shipped module compatibility.

### Linux and Windows release archives

```sh
make -C src native-package VERSION=1.0.0
# Include older x86 CPUs:
make -C src native-package VERSION=1.0.0 X86_ISA=
```

Packaging includes the `CUDA_MODELS` executables — every model with a fatbin,
currently all four — excludes OpenCL, and forces static support-library linkage
on Windows. It checks the embedded
modules and binary imports, then creates a tar.gz (Linux) or zip (Windows) in
`dist/`, with notices, build flags, executable hashes and an archive checksum.
It rejects toolkit/runtime dependencies, Windows companion DLLs and Linux
build-specific runtime search paths. Packaging needs Python and GNU `objdump`;
running the result needs neither.

Linux archives retain system glibc/zlib dependencies. Build on the oldest
supported distribution; this target does not make a universal Linux binary.
Windows archives are unsigned. The [CI workflow](../.github/workflows/ci.yml)
can run just these distribution checks with `cuda_only=true`; Windows NVIDIA
execution is a separate hardware validation step.

### Choosing a GPU

Builds with CUDA, HIP or OpenCL accept `-a N` (`--adapter N`) to pick a GPU, and
`-a -1` to list what is available and exit:

```console
$ brainchop-mindgrab -a -1
cuda:
  0  NVIDIA GeForce RTX 4070 Ti SUPER (15.99 GB, sm_89)
opencl:
  0  AMD Accelerated Parallel Processing / gfx1036 (35.81 GB, integrated)
  1  NVIDIA CUDA / NVIDIA GeForce RTX 4070 Ti SUPER (15.99 GB)

$ brainchop-mindgrab -backend opencl -a 1 input.nii.gz
```

**Indices are per backend**, which is why the listing prints them under
headings: OpenCL flattens every platform's devices into one list, while a CUDA
index is the device ordinal that `nvidia-smi` and `CUDA_VISIBLE_DEVICES` already
use. An out-of-range index is rejected before any work is done, naming the valid
range.

Without `-a`, OpenCL prefers a **discrete** GPU and uses global memory only to
break ties within a class. That ordering matters: an integrated GPU reports a
slice of system RAM, so the listing above shows a 2-CU integrated part claiming
35.81 GB against a 16 GB discrete card. Selecting purely on memory — which
earlier releases did — picks the integrated one and runs roughly 117x slower.

`BRAINCHOPC_OPENCL_DEVICE` still accepts a case-insensitive substring of the
platform or device name, and is the older spelling of the same idea. `-a` takes
precedence over it.

Both GPU implementations need two activation allocations of approximately
1 GB each, plus weights and runtime workspace. An OpenCL device must permit a
single allocation of at least the activation size.

## Building on Windows

The CPU, OpenCL and CUDA backends all build on Windows and produce native
executables with no dependency on the build environment at run time.

**Configure Git before cloning, not after.** Git for Windows defaults to
`core.autocrlf=true`, which puts CRLF into the working tree. The repository
ships a `.gitattributes` that prevents this, but it can only help a checkout
made after it exists. A CRLF checkout of `src/Makefile` fails in a way that is
genuinely hard to diagnose — `git status` stays clean, because Git normalizes
the carriage returns back out on the way into the index, so the damage is
invisible to every Git command you would think to run, while the build silently
loses AVX2 and switches off every Windows-specific branch. If in doubt:

```sh
git clone https://github.com/neuroneural/brainchop-core.git
cd brainchop-core && git add --renormalize . && git checkout -- .
```

### Toolchain

The build uses [MSYS2](https://www.msys2.org)'s UCRT64 environment, which
provides GCC, GNU make and a POSIX shell while emitting ordinary native Windows
binaries. From the **MSYS2 UCRT64** shell:

```sh
pacman -S --needed make mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-zlib \
    mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-pkgconf
cd src && make
```

That builds the CPU and driver-only CUDA backends. Use `OPENCL=1` to include
the dynamically loaded OpenCL backend without installing SDK headers. MindMap and mindsnap receive CPU, CUDA and HIP support; OpenCL remains limited to the
16-channel models.
GPU execution has been validated on an RTX 4070 Ti SUPER (Windows and Linux): `test-cuda` and
`test-opencl` pass within the same budgets, and the driver-only and
`CUDA=1` builds agree voxel for voxel. On NVIDIA, prefer CUDA (automatic
selection already does): the OpenCL driver alone spends about half a second
creating its context on Windows, so OpenCL inference takes two to three times
as long as CUDA there.

Adding `mingw-w64-ucrt-x86_64-clang` and `mingw-w64-ucrt-x86_64-lld` is
optional but worthwhile: the Makefile prefers clang with lld when it can link
`-flto`, which measured 1.65× faster than GCC here with identical output. Add
`mingw-w64-ucrt-x86_64-nodejs` and a sibling `../brainchop-test` checkout for
the browser-oracle gates in `make test`, which otherwise skip.
Threading uses MSYS2's libwinpthread, which arrives with GCC, so it
needs no extra package. Note that Git Bash also reports itself as `MINGW64` but
ships no compiler — it is a shell, not a toolchain, and `make` there fails at
`cc`.

Executables are linked `-static`, so a build can be copied to another machine as
a single file. Build with `WINDOWS_STATIC=0` for a smaller binary that must stay
beside its DLLs.

### CUDA

The default build embeds both CUDA host code and device kernels using the
same UCRT64 C toolchain as the rest of the executable. It needs no Visual Studio
Build Tools or CUDA toolkit and ships no `brainchop-cuda-<model>.dll` or CUDA
runtime DLL. The NVIDIA driver supplies `nvcuda.dll` at runtime. Driver and GPU
requirements are the same as [Linux](#building-on-linux).

To omit OpenCL support entirely (enabled builds also load its driver dynamically):

```sh
make -C src OPENCL=0
make -C src OPENCL=0 test-cuda  # on an NVIDIA machine
```

Only **legacy validation with `CUDA=1`** requires Visual Studio Build Tools
with the C++ workload and NVIDIA's CUDA Toolkit. In that mode nvcc uses MSVC,
and the helper configures both toolchains:

```powershell
scripts\windows_make.ps1 CUDA=1 OPENCL=0 models
scripts\windows_make.ps1 CUDA=1 OPENCL=0 test-cuda
```

The legacy build retains a `brainchop-cuda-<model>.dll` beside each executable
and needs its CUDA runtime. These DLLs are excluded from default release
archives. `scripts\windows_make.ps1 OPENCL=0 models` can also run an ordinary
build without initializing MSVC or nvcc.

### Smart App Control

Windows 11 may refuse to launch freshly built executables, reporting
`Permission denied` with exit code 126 from `make`, or "An Application Control
policy has blocked this file" from PowerShell. This is Smart App Control, which
blocks unsigned binaries that have no cloud reputation — every rebuild is a new
unknown hash, so it can appear to work once and then stop. It is not antivirus,
and Windows Defender's tools report nothing; the evidence is event 3077 in the
`Microsoft-Windows-CodeIntegrity/Operational` log.

Turning it off is the only remedy for locally built binaries, under Windows
Security → App & browser control → Smart App Control. **Be aware that this is
one-way: it cannot be re-enabled without reinstalling Windows.**

## Compute backends

Backend selection is automatic unless `-cpu` or `-backend` is present. The
automatic priority is CUDA, HIP, Metal, WebGPU, WebGL2, OpenCL, then CPU, considering
only backends compiled into the executable and available at runtime. WebGPU and
WebGL2 exist only in the WebAssembly builds, and only one of the two is compiled
into any given module.

- `-cpu N` always forces CPU. `N > 0` sets the number of participating
  threads, counting the calling one, so `-cpu 1` spawns no workers;
  `N <= 0` lets the pool choose: `BC_NUM_THREADS` if set to a positive integer,
  else the CPUs this process may use (the affinity mask on Linux, the machine's
  online count elsewhere).
- `-backend NAME` selects `auto`, `cpu`, `cuda`, `hip`, `opencl`, `metal`, `webgpu` or
  `webgl2`. Each build offers only the backends it was compiled with, and
  `--help` advertises exactly those — an executable never lists an option it
  would refuse, and never accepts one it did not list.
- `BRAINCHOPC_BACKEND=NAME` provides the same selection through the environment.
- `BC_NUM_THREADS=N` sets the participating-thread count when `-cpu N` is not given. The
  flag wins over the variable. Neither is capped at the core count — asking for
  more workers than cores oversubscribes rather than being clamped, which is
  what makes the documented thread-count sweep up to 64 meaningful on a 14-core
  machine. The only ceiling is 1024, so a typo cannot ask for a million stacks.
  Unset, the pool takes the CPUs this process may use — the affinity mask on
  Linux, the machine's core count elsewhere. This replaces `OMP_NUM_THREADS`,
  which no longer has any effect. One stated limitation: on a Linux host with
  more than 1024 CPUs the affinity mask cannot be read and the machine-wide
  count is used, so a process pinned to a few cores of such a machine will
  over-subscribe unless you pass `-cpu N`.
- `BRAINCHOPC_PROFILE=1` prints how long each host stage took -- read, conform,
  inference, connected components, reslice, write -- to stderr. On a GPU run most
  of the wall clock is these stages rather than the GPU, so this is where to look
  when a run is slower than the GPU time suggests. It does not account for all of
  the wall clock; three windows are deliberately unmarked.
- An automatically chosen GPU that fails during inference falls back to CPU.
- An explicitly selected unavailable or failing GPU reports an error rather
  than silently changing backends.
- Exact `--crop` and streamed `--export-classes` use CPU. An explicit GPU
  request for either is unsupported. MindMap `--pve` supports CPU, CUDA, HIP,
  Metal, WebGPU and WebGL2; its
  CAT-lite fit remains on CPU. CT, compliance, mask export and mask borders
  can all use GPU inference.

Examples:

```sh
./brainchop-mindgrab -o output.nii.gz input.nii.gz
./brainchop-mindgrab -cpu 8 -o output.nii.gz input.nii.gz
./brainchop-mindmap-fast -backend cuda -o labels.nii.gz input.nii.gz
./brainchop-mindgrab -backend opencl -o output.nii.gz input.nii.gz
./brainchop-mindmap -backend metal -o labels.nii.gz input.nii.gz
./brainchop-mindmap --pve -o brain.nii.gz input.nii.gz
```

The backend contract is defined in `src/backend.h`. CPU, CUDA, HIP, OpenCL, Metal,
WebGPU and WebGL2 each live in their own translation unit, so platform headers
and libraries do not leak into other builds; the two browser backends are never
compiled together. CUDA, OpenCL, WebGPU and WebGL2 share host-side model
metadata, normalization and weight preparation in `src/backend_accel.c` and
`src/accel_pack_weights.h`. The default Windows CUDA host adapter links directly into the executable;
only the explicit legacy nvcc path uses a DLL exporting the `bc_backend` table.

The CPU implementations share a fixed-shape MeshNet engine in
`src/meshnet_cpu.c`. It instantiates two convolution shapes, `mn_conv_first`
and `mn_conv_hidden`, preserving compile-time channel specialization while
sharing buffer management, reductions, normalization, and classification.
Every C source is shared between the four executables; what differs is the
`-DBC_MODEL_IS_<NAME>` each is compiled with, which binds the generic
`BC_MODEL_*` constants to that model's generated descriptor in
`src/models/<name>/model_def.h`. Largest-component filtering and save behavior
are shared so every backend follows the same Python path. The packed channel
stride the CPU and browser engines lay activations out with is engine policy
rather than model metadata, and lives in one place, `src/chan_stride.h`: 16 for
mindgrab and model16, 24 for mindmap and mindsnap.

## Tests

```sh
make test           # everything below that runs without a GPU
make test-models    # forced-CPU regressions, bit-exact, plus the CLI surface
make test-guards    # exit-code contracts only
make test-cpu-turbo # default CPU turbo: kernels vs scalar/fp32 oracles, CPU_TURBO_GATE
make test-cpu-sme   # exact SME vs NEON, float for float
make test-cat-lite  # the --pve port against the upstream cat-lite.js
make check-payloads # each model.bin's size matches its generated descriptor
make refresh-fixtures MODEL=<name>  # rewrite CPU fixtures after a deliberate weights change
make test-metal     # Metal/CPU parity on macOS: turbo, fp16 and mindmap --pve
make test-cuda      # CUDA/CPU parity when CUDA is built and a device is present
make test-opencl    # OpenCL/CPU parity when OpenCL is built and a device is present
make benchmark      # timing / memory / agreement matrix, every backend this host can run
```

The model regressions require only Python's standard library; there is no numpy
or other third-party dependency. `make test` additionally runs `test-cat-lite`,
which compares the `--pve` port against the reference implementation and so
needs **node and a `../brainchop-test` checkout** (with `npm ci` run in it; `test-label-resample` needs its `gl-matrix`) — it **skips** loudly,
naming `BC_BRAINCHOP_TEST`, rather than failing. Failing would take `make test`
— the whole safety net, as CI runs only by hand — down for every change
unrelated to PVE. A skip does retire the only gate the CAT-lite fit has, so anyone
relying on that gate must check it actually ran. A build whose model list
carries no PVE model says so and passes.

The fixtures are tracked, so the targets run on a fresh clone. Two inputs:
`test/t1_crop.nii.gz`, already 256³ at 1 mm, and `test/t1_2mm.nii.gz`, a real
96×128×128 2 mm INT16 acquisition. CPU reference outputs: `test/bet.nii.gz`
(MindGrab), `test/labels16.nii.gz` (model16 `--save-conform`),
`test/labels16_native_majority.nii.gz` (model16 default, resliced to the input grid), and
`test/labels24.nii.gz` / `test/labels24_native_majority.nii.gz`, the same two cases for
mindmap, and `test/labels104.nii.gz` / `test/labels104_native_majority.nii.gz`
for mindsnap (`labels104_wasm.nii.gz` on wasm, for one FMA-rounded tie).
Model16 uses `labels16_native_majority_wasm.nii.gz` on wasm and x86
for a pre-existing arithmetic difference; all CPU gates remain exact.
The paths are the `TEST_IN` / `TEST_2MM` / `TEST_MG` / `TEST_M16` /
`TEST_M16_NATIVE` / `TEST_MM` / `TEST_MM_NATIVE` / `TEST_MS` / `TEST_MS_NATIVE` variables at the top of the
test section.

Both inputs are needed: `t1_crop.nii.gz` is already on the grid `conform`
targets, so for it the conformed and native output spaces coincide and a test of
that distinction cannot fail. `test/labels16_native_majority.nii.gz` is therefore derived
from `t1_2mm.nii.gz`.

`test-models` also asserts the CLI surface: that each `--help` lists exactly the
options its model supports, and that an option the sibling executable offers is
refused with a nonzero exit rather than ignored.

The exact CPU path (`--no-turbo`) is asserted **bit-exact** against those references, because it is
what every GPU backend is compared against; a tolerance there would only mask CPU
regressions. That assertion has been verified to hold across compilers and
architectures — the references were produced by Apple Clang on arm64, and the
same three 16-channel cases pass bit-exact under GCC 13.3 and Clang 18.1.3 on
x86-64. MindMap's two fixtures also pass bit-exact under GCC 13.3 on the
NVIDIA cluster's x86-64 hosts (2026-09-20), and all seven cases pass under
Clang 22 on Windows/x86-64 (2026-09-30).
Output is also independent of the thread count.

The GPU targets carry tolerances. Metal and WebGPU turbo is gated by
`test/nii_dice.py` at brainchop-test's thresholds (`TURBO_GATE`: macro Dice
0.988, foreground agreement 98.9%; MindGrab 0.995 / 99.5%). The fp16 paths —
Metal and WebGPU under `--no-turbo`, WebGL2 and CUDA — share the same budgets: 64 voxels for MindGrab and 1024 for an
18-class model, out of 16777216. With MindGrab's trained final bias enabled
([PR #29](https://github.com/neuroneural/brainchop-core/pull/29), 2026-10-19),
the measured MindGrab differences are **Metal 35 on Apple M1 and 38 on M4 Pro**,
**WebGPU 33** and **WebGL2 29** on M1 (Chromium and Firefox for WebGL2), 31 and
31 on M4 Pro (2026-10-19), and
**CUDA 38–40** across eight NVIDIA GPU families, with scalar and tensor-core
paths tested. These use the updated `test/bet.nii.gz` reference.

The unchanged model16 measurements are Metal 654, WebGPU 682 and WebGL2 710
(Apple M4 Pro, 2026-08-05), and CUDA 469 (RTX 4070 Ti SUPER). The two browser
backends additionally check model16 in native 2 mm space against a budget of
256, measuring 95 (WebGPU) and 94. None of those budgets has been widened.

mindmap holds the same budgets at 24 channels, on every backend it has, and
the existing backends are gated: on the v4.0 Gold weights **Metal 517 / 92**,
conformed and native. On the previous weights WebGPU read 564 / 88, WebGL2
610 / 86 and CUDA 490–505 / 65–74 across the tested GPU types.
`mindsnap` is a 104-way argmax, so it carries its own wider budget
(`MS_TOL` 2048, `MS_TOL_NATIVE` 512): **Metal 1248 / 171**, **WebGPU 1280 / 198**
and **WebGL2 1219 / 184**; **CUDA 919 / 169**. `make test-metal`,
`make test-wasm-gpu` and `make test-wasm-gl` each run seven cases now — the three
16-channel ones plus mindmap's two and mindsnap's two. Before MindGrab's bias
fix, the 16-channel cases remained at 33 / 710 / 94 on WebGL2 and 37 / 682 /
88 on WebGPU while the kernels were generalized beyond a hardcoded 16-wide
stride. Those historical counts verify that the stride change preserved the
16-channel paths; current MindGrab counts are listed above.

OpenCL is fp16 too and holds the same budgets (`OPENCL_TOL_MINDGRAB` 64,
`OPENCL_TOL_MODEL16` 1024). Bias-enabled MindGrab measures 37 differing voxels
on RTX 2080 Ti; model16's earlier measurement is 460 on RTX 4070 Ti SUPER.
Override the variables on the make command line if a different device
legitimately needs more room.

These budgets were set on `test/t1_crop.nii.gz` and are not a promise for every
brain. On three real-world T1s that are not 256³ 1 mm (from niivue-demo-images,
0.74–0.88 mm), GPU and CPU differed in 0.003–0.007% of conformed voxels with
mean label Dice ≥ 0.9991; one image read 1063–1225 for the 18-class models,
slightly above 1024. `TEST_IN=` and `TEST_2MM=` point the test targets at
other images — for example `make test-cat-lite-run TEST_IN=scan.nii.gz` runs the
`--pve` comparison against `cat-lite.js` on your own scan.

model16's and (since PR #41) mindmap's references are generated by this
implementation, so they detect change, not correctness. MindGrab's and
mindsnap's are corroborated: MindGrab's are bit-identical on both bundled fixtures to the
Python brainchop CLI at
[`31150c3`](https://github.com/neuroneural/brainchop-cli/commit/31150c39dae90dba3d18b00d05d32eba273056af)
or later (PR #60, which restores its trained final classifier bias). On the
pre-PR #41 mindmap weights both fixtures agreed with a tinygrad run at **zero**
differing voxels (current `labels24*` are regression-only until re-run) — which
required substituting this model's 5/95 rank normalization for the 2/98
MindGrab form that the reference's `qnormalize()` hardcodes for every model —
and both mindsnap fixtures agree with `scripts/oracle_mindsnap.sh` (tinygrad
`--custom`, this model's own 2/98 form) to 0 and 2 voxels, both fp32 ties.

Historical checks on three further T1s (2026-10-01, before MindGrab's final
bias was restored; niivue-demo-images `register/`: MNI152 1 mm and 2 mm,
ICBM152 nlin asym 55 ext), CPU vs tinygrad (fp32, NV device) differed in
mindgrab 0/0/0, mindmap (`--legacy-cleanup`, conformed) 1/1/1 and mindsnap
0/2/2 voxels. CUDA vs tinygrad: mindgrab 25/3/32, mindmap 877/1096/806,
mindsnap 1490/1630/1471, mean Dice ≥ 0.9989. Wall time: tinygrad 5.3–7.2 s,
CUDA 0.8–1.4 s (mindmap `--pve` 1.6–2.0 s), CPU 4.4–5.7 s; peak host RSS
~490 MB, 285–530 MB and 2.2–3.5 GB respectively.

`make test-cat-lite` is the `--pve` gate and is not a tolerance fitted after the
fact. Both implementations are driven from one `BRAINCHOPC_PVE_DUMP` file — the
blurred, support-gated priors and the normalized intensity, at exactly the
boundary `cat-lite.js` consumes — so the network, the normalization and fp16 are
all upstream of the comparison. The threshold is float32 epsilon because nothing
smaller can mean anything; measured agreement is 5.96e-08, the last bit of the
float the two sides store. It also asserts the fit is thread-count independent.
`BRAINCHOPC_PVE_DEBUG=1` prints the fitted tissue means and sigmas, which the
reference prints too, and is how a divergence is localised to the fit.

Beyond the make targets, `test/baseline.sh` replays a wider 41-case invocation
matrix (`test/baseline_cases.txt`) against a capture directory. It is not wired
to a make target; invoke it directly:

```sh
test/baseline.sh compare modular test/baseline \
  src/brainchop-mindgrab src/brainchop-mindmap-fast
```

The capture directory is untracked, so a fresh clone must `capture` before it can
`compare`. 15 of the 41 cases run over real acquisitions and skip unless
`BC_EXTRA_DIR` points at them. `baseline_cases.txt` names the two original
executables only; mindmap and `--pve` are not in that matrix.

`make test-cuda` and `make test-opencl` print a message and succeed when the
backend is not built, which is the case on macOS by default. A green run there
proves nothing about those backends; they need a Linux or Windows host with the
corresponding device.

`make ubsan` creates a serial CPU-only UndefinedBehaviorSanitizer build, and
`make asan` the same with `-fsanitize=address,undefined`, which on Linux also
enables LeakSanitizer. Run `make clean` afterwards: both leave instrumented
objects behind that a later plain `make` would relink without the sanitizer
runtime. Each is serial (`THREADS=0`) and takes on the order of half an hour per
model, so they are not part of `make test`.

`make tsan-threads` is a third sanitizer target and is unlike those two: it
builds the pool THREADED, because it exists to test the thread pool rather than
the engine, and it runs in seconds. (It does not read `THREADS` at all — the
recipe hardcodes `-DBC_THREADS`, so `THREADS=0` does not make it serial.) It builds `test/threads_race_check.c` under
ThreadSanitizer and asserts both directions — index-disjoint writes through
`bc_parallel_for` must come back clean, and a deliberately planted race must be
caught. The second half is what makes the first mean anything: a clean run is
equally consistent with the detector being unarmed, so the target fails if the
planted race goes unreported. Also not part of `make test`, since it needs its
own instrumented build.

### Benchmark

`make benchmark` runs a fixed matrix — both tracked inputs, every backend the
build has — and prints wall time, peak CPU resident memory, peak GPU memory and
differing voxels against the CPU result. `test/benchmark.py` names its models
literally and now covers all four executables:

```sh
make benchmark                  # print the table
make benchmark-capture          # write test/benchmark_reference.json
make benchmark-compare          # replay and diff against that reference
```

`benchmark-compare` fails only if voxel agreement got worse; timing deltas are
reported but never asserted, because they depend on machine load. Each GPU cell
runs one discarded warm-up (which absorbs the CUDA autotune) and reports the
median of `--repeats` timed runs; the two CPU rows are timed once. The reference
file records the CPU, GPU, compiler and commit it was captured on, since neither
the timings nor the GPU voxel counts are comparable across machines. Captures
for other hosts sit beside it and are selected with `BENCH_REFERENCE=`:
`test/benchmark_reference_m4pro.json` (Apple M4 Pro) and
`test/benchmark_reference_windows.json` (Windows 11, on the same Ryzen 9 7950X
and RTX 4070 Ti SUPER as the default Linux capture).

The matrix includes a `cpuN1` row, which is the CPU backend restricted to a
single thread (`-cpu 1`). Single-core throughput is the figure that
characterises the convolution kernel itself, separately from thread scaling and
memory bandwidth.

A backend that is not compiled in, or has no device, is skipped rather than
reported — so the table shows which backends are healthy on this host, and says
nothing at all about the ones that did not appear.

Header dependencies are tracked (`-MMD -MP` for the host compilers, `-MD` for
`nvcc`, which spells the feature differently and rejects an unknown option
fatally rather than passing it through), and the objects depend on a
`.build-config` stamp covering the compiler, CPU flags, backend selections and
library choices (`BUILD_CONFIG` in `src/Makefile`), so
changing a header, a backend selection or the zlib choice rebuilds what it
should instead of silently linking stale objects. Each model's objects live in
their own `src/build/<model>/` directory, since each is compiled with
different `-D` flags and cannot be shared.

The default release build already enables `-O3`, fast-math, link-time
optimization, dead-section elimination and the pthread pool. CUDA uses `-O3` and
`--use_fast_math`. The Objective-C++ Metal host runner uses `-O3` and LTO;
its generated MSL deliberately uses strict floating-point math for reference
parity.

## WebAssembly and the browser

The same C builds for the browser, with two GPU backends and a threaded CPU one:
**WebGPU** where it exists, a **WebGL2** fallback for the browsers it has not
reached — notably Linux Firefox — and a **pthreads CPU** module for a browser
with neither, which needs the page to be cross-origin isolated. All three are
separate modules, and a caller downloads only the one it can use.

```sh
make wasm-gpu          # the shipping WebGPU module, one .js + .wasm per model
make wasm-gl           # the WebGL2 fallback module, likewise
make wasm              # the threaded CPU module, likewise (THREADS=0 for serial)
make test-wasm-gpu     # seven cases: fp16 within the Metal budgets, turbo within its gates
make test-wasm-gl      # the same seven cases, same budgets, on WebGL2
make test-wasm         # asserts the CPU module bit-exact with the native build
make test-wasm-mt      # the CPU module in a browser: page, worker, and a
                       #   REQUIRED failure on a page without COOP/COEP
make test-wasm-guards  # asserts a .nii.gz input is refused, not misread
make check-wgsl        # compiles the WGSL in a real browser
make check-glsl        # compiles the GLSL in Chromium and Firefox
```

WebGPU is about twice as fast — on an Apple M4 Pro, 2026-08-04, MindGrab takes
1.63 s in the module against WebGL2's 3.35 s, and the 18-class model 0.91 s
against 1.84 s, both fp16 — and both meet the same fp16 voxel budgets as the
Metal backend, unwidened. WebGPU's default turbo path cuts mindmap's GPU time to
about 1.1 s from 1.9 s (`noTurbo` / `--no-turbo` restores fp16). End to end, including the host stages the module does not time, the
same machine measures 1.52 s for model16 on WebGPU and 2.50 s for mindmap,
against roughly 12.3 s for the JavaScript brainchop page this replaces on an M1.
The WebGPU convolution runs 64x2 workgroups, or 256x1 on an AMD adapter
(output bit-identical; `BC_WG_CONV_PRESET` forces either).

mindmap pays extra on WebGL2 specifically, and the reason is structural rather
than the extra arithmetic: a 24-wide channel stride puts the two-voxel blocking
out of reach, because two z slices at once would want twelve draw buffers
against the eight browsers report. It therefore runs one voxel per fragment —
5.5 s against model16's 2.0 s, an end-to-end pair measured together. Those
are macOS figures; the WebGL2 fallback is known to run on Linux Mesa stacks in
both Firefox and Chrome, but has not been timed there. WebGL2 also runs
**synchronously**, so it blocks the thread it is called on for the whole
segmentation; run it in a Web Worker (below) if the page must stay responsive.

The threaded CPU module runs MindGrab in about 10 s on 14 cores against ~1 s on
WebGPU, and is **bit-exact** with the native executables rather than held to a
voxel budget. The JS `auto` backend takes it after hardware WebGPU and
WebGL2, and never picks an identified software rasteriser (SwiftShader,
llvmpipe, lavapipe, WARP, a WebGPU fallback adapter), which is ~100× slower
still; a masked WebGL2 renderer runs only when CPU cannot, and with none of
them it throws `no-webgpu`. Every backend refuses an empty MindGrab mask
("the network found no brain"), as label models refuse empty labels. It needs
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` on the page, because a pthreads
module imports shared memory. Playwright's patched Firefox Nightly runs it about
20× slower than Chromium, but installed Firefox 153.0.1 ran the model16 /
`t1_2mm` case bit-exact in 7.6 s — level with Chromium once browser startup is
excluded.

mindmap and mindsnap are the models that strain the address space: 24
channels need about 3.2 GB of heap against wasm32's 4 GB ceiling, so each is
bit-exact under node on a machine that can supply that and fails at the memory
growth on one that cannot.

The module handles **uncompressed NIfTI only** — no zlib is linked, so gzip is
the wrapper's job. The wasm tests need `emcc`; `test-wasm-gpu`, `test-wasm-gl`,
`test-wasm-mt`, `check-wgsl` and `check-glsl` additionally need a checkout with
playwright installed (`BC_PLAYWRIGHT_ROOT`). The GPU ones run **headed**,
because headless Chromium on macOS has no GPU; `test-wasm-mt` needs no GPU and
runs headless. None of them are part of `make test`.

`js/` is published to npm as
[`@brainchop/core`](https://www.npmjs.com/package/@brainchop/core)
(`npm i @brainchop/core`), combining a pure-JS WebGPU V4 engine (`dist/models/*.bin`)
with Emscripten WebGL2 and threaded-CPU WASM fallbacks behind
`segment(buffer, {model}) -> {image, mask?, elapsedMs, backend, ranInWorker}`.
It probes for WebGPU, then WebGL2, then cross-origin isolation before fetching
anything, or `backend: 'webgpu' | 'webgl2' | 'cpu'` names one explicitly —
honoured or refused, never silently downgraded. `worker: true` runs the whole
segmentation in a Web Worker, which is the recommended way to call it from a
page whichever backend runs. It owns the things the WASM fallback modules deliberately do
not: gzip, device and context acquisition, the isolation probe, and the decision
to refuse. The worker, model binaries, and fallback modules are imported by literal specifiers, so Vite,
Rollup and webpack emit them and nothing needs copying.
A host running one model can import
`@brainchop/core/<model>` to deploy only that model's assets.
`MODELS[name].colormap` carries each model's label names and colours. Hosts that
serve `dist/` themselves can still point `assetPath` at it. See
[js/README.md](../js/README.md).

The package carries **all four** models on all three backends, and
`js/test/segment.test.mjs` gates seven fixtures on each of them. The
one thing 24 channels changed on the JavaScript side is the WebGPU device
request: it is now `max(1 GiB, 2 x one activation)` per model rather than a flat
1 GiB, which is unchanged at 16 channels and 1.6 GiB at 24. The flat request did
not report an error at 24 — the renderer died, and playwright said only that the
page had closed.

## Licensing

The project is MIT; see `LICENSE` and `THIRD_PARTY_NOTICES.md`, which also
covers libwinpthread (static in Windows executables) and the model weights.
MIT permits including BSD-2-Clause material as long as the notice travels with
it, which is what the niimath section of `THIRD_PARTY_NOTICES.md` is for: the
EDT in `preprocess.c` is a port of niimath's, and `conform.c` reproduces its
conform, comply and reference-grid reslice.
The current `bwlabel.c` is a small independent queue-based 26-connected flood
fill, not the Andersson/SPM labeling implementation in niimath. Therefore the
Andersson source notice is not required for this file. If that upstream
implementation is copied back in later, its BSD notice must accompany it.
