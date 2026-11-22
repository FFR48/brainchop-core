<p align="center">
  <a href="https://brainchop.org">
    <img src="https://raw.githubusercontent.com/neuroneural/brainchop-core/main/docs/assets/brainchop-icon.png" alt="BrainChop" width="120" />
  </a>
</p>

# @brainchop/core

Browser MeshNet brain segmentation on pure-JS WebGPU V4, with WebGL2 and
threaded-CPU WASM fallbacks: MindGrab skull stripping (`mindgrab`), 18-class
parcellation at two widths (`mindmap-fast` / alias `16chan18cls`, `mindmap`),
and a 104-class Desikan-Killiany parcellation (`mindsnap`), over a fixed 256³
grid. A NIfTI `ArrayBuffer` goes in and a NIfTI `ArrayBuffer` comes out.

On **WebGPU**, the entire pipeline — NIfTI-1 I/O, coil-flare-safe conform
(`fHigh = 0.98`), normalization, the 13 or 25 MeshNet layers (ZYCX split-plane
u8x4 V4 kernels with optional `shader-f16` + automatic `f32` fallback, flexible
subgroups, and automatic `LowMem128` + paced Z-chunking), connected-component
cleanup / CAT-lite tissue fitting, and native-space reslicing — runs in **pure
JavaScript + WGSL** reading `models/<name>.bin` directly with **zero WebGPU WASM
binaries**. WebGL2 and threaded CPU remain available as Emscripten WASM
fallbacks compiled from [brainchop-core](https://github.com/neuroneural/brainchop-core).

```js
import { segment } from '@brainchop/core'

const input = await file.arrayBuffer()          // .nii or .nii.gz
const { image, elapsedMs } = await segment(input, {
  model: 'mindgrab',
  worker: true,
  onProgress: ({ stage, progress, layer, totalLayers }) => {
    console.log(stage, Math.round(progress * 100) + '%', layer ? `${layer}/${totalLayers}` : '')
  },
})
// image is gzipped if the input was
```

MindMap and MindMap-Fast filter selected tissue classes and refill suppressed
speckles from surviving neighboring labels by default. Use
`{ model: 'mindmap', legacyCleanup: true }` for the original whole-brain-only
cleanup.

Categorical native-space output uses adaptive center-outward midpoint majority
voting, including background votes outside the source and centermost tie
breaking. Masks use nearest neighbor; continuous tissue maps use trilinear
resampling. Invalid, empty, more than 95% filled, or excessively fragmented
categorical results fail with `inference-failed` before cleanup can disguise the
failure.

## Continuous tissue maps

```js
import { segmentTissues } from '@brainchop/core'

const { tissues, backend } = await segmentTissues(input, {
  model: 'mindmap', worker: true,
})
// tissues.gm, tissues.wm, tissues.csf and tissues.brain (GM + WM) are NIfTI
// ArrayBuffers. Native geometry and input gzip encoding are preserved by default.
```

`tissues.brain`'s 0.5 isosurface is a sub-voxel pial surface, ready to mesh.

`segmentTissues` supports `mindmap` and `mindmap-fast` (or legacy alias
`16chan18cls`) on WebGPU, WebGL2 and CPU. The GPU paths compute grouped priors,
separable Gaussian blur and brain-support gating on the device, followed by the
CAT-lite mixed-class partial-volume fitter. `saveConform`, `gzipOutput`,
`onProgress`, and `webgpuMode` work as for `segment`; mask, border and legacy
categorical cleanup options are refused.

## Backends

`auto` (the default) tries hardware `webgpu`, hardware `webgl2`, then `cpu`
(cross-origin isolated pages only), reports which one ran in `result.backend`,
and throws `no-webgpu` if none can. It never picks a software rasteriser
(SwiftShader, llvmpipe, lavapipe, WARP, or a WebGPU fallback adapter).

### Pure-JS WebGPU V4 auto-detection

The WebGPU V4 engine requires **no `-gpu.wasm` binaries** and automatically
adapts to device capabilities via a 3-layer compatibility system:

1. **Zero-overhead hardware hints:** Apple Silicon uses the fast single-encoder
   path; known shared-RAM iGPUs (Windows Intel UHD/Iris, Linux Mesa Intel Gen9)
   and `<= 128 MiB` storage-buffer devices (e.g. mobile WebGPU) automatically
   select `LowMem128` (max single buffer `<= 80 MiB` for 15/16ch, `<= 120 MiB`
   for 24ch) + paced Z-chunking.
2. **Self-calibrating micro-probe:** Unrecognized or masked GPUs run a brief
   13.5 MiB calibration probe (cached for 30 days) to choose `std + pipelined`,
   `std + paced`, or `LowMem128 + paced`.
3. **Single safe-mode retry:** If an un-paced run hits `device.lost` or OOM,
   safe mode is persisted and the run is retried once in `LowMem128 + paced`.

Feature tiers degrade gracefully with no hard `shader-f16` requirement:
- **Tier 1:** `shader-f16` + `subgroups` (`sg32`, `sg64`, `sg32_64`, `sg_small`)
- **Tier 2:** `shader-f16` without `subgroups` (shared-memory epilogue)
- **Tier 3:** `f32` fallback when `shader-f16` is absent

| backend | needs | artifacts |
| --- | --- | --- |
| `'webgpu'` | WebGPU with `>= 128 MiB` storage buffers, secure context (`shader-f16` & `subgroups` optional) | pure JS + `models/<name>.bin` (340–790 KiB) |
| `'webgl2'` | WebGL2 with `EXT_color_buffer_float` (blocks calling thread — use `worker: true`) | `brainchop-<name>-gl.{js,wasm}` |
| `'cpu'` | **cross-origin isolated** page (`COOP`/`COEP`) and growable shared memory | `brainchop-<name>.{js,wasm}` |

### The CPU fallback needs two headers

Serve the **top-level document** with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

```js
import { checkCpuSupport } from '@brainchop/core'

const cpu = checkCpuSupport()
if (!cpu.supported) console.warn(cpu.reasons.join('\n'))
```

## Requirements

For the WebGPU path: WebGPU with `maxStorageBufferBindingSize` and
`maxBufferSize` of at least **128 MiB** and a **secure context** (`https` or
`localhost`). `shader-f16` and `subgroups` are automatically used when present
and gracefully skipped otherwise.

```js
import { checkSupport } from '@brainchop/core'

const report = await checkSupport()
if (!report.supported) console.warn(report.reasons.join('\n'))
```

## Models

| `model` | does | output |
| --- | --- | --- |
| `'mindgrab'` | skull stripping in any modality | input image with non-brain voxels floored |
| `'mindmap-fast'` | 18-class brain segmentation (16-channel, MindMap-Fast v3.5; `'16chan18cls'` kept as alias) | label volume |
| `'mindmap'` | 18-class brain segmentation (24-channel, MindMap-18 v4.0 Gold) | label volume |
| `'mindsnap'` | 104-class Desikan-Killiany parcellation (24-channel, MindSnap-104 v4.0 Gold) | label volume |

Each `MODELS` entry (and `listModels()`) carries `colormap`: `{ R, G, B, labels }`,
ready for viewers such as NiiVue (`nv.setColormapLabel(1, MODELS.mindsnap.colormap)`).

## Options

```ts
segment(input: ArrayBuffer | ArrayBufferView, options: SegmentOptions)
```

| option | models | meaning |
| --- | --- | --- |
| `model` | all | **required**: `'mindgrab'`, `'mindmap-fast'` (or alias `'16chan18cls'`), `'mindmap'`, `'mindsnap'` |
| `ct` | all | input is CT; convert Hounsfield to Cormack units |
| `comply` | all | insert the compliance transform before conform |
| `saveConform` | label models | return the conformed 256³ volume instead of reslicing to the input grid |
| `legacyCleanup` | label models | restore whole-brain 26-connected cleanup without per-class refill |
| `mask` | `mindgrab` | also return a binary brain mask in `result.mask` |
| `borderMm` | `mindgrab` | grow the mask border, in mm; implies `mask` |
| `gzipOutput` | all | override the default, which matches the input |
| `backend` | all | `'auto'` (default), `'webgpu'`, `'webgl2'`, `'cpu'` |
| `webgpuMode` | all | WebGPU V4 layout/pacing override: `'auto'` (default), `'standard'`, `'low-memory'`, `'recalibrate'` |
| `worker` | all | run the whole segmentation in a Web Worker (**recommended**) |
| `device` | all | reuse a `GPUDevice` you already own (implies `webgpu`) |
| `glContext` | all | reuse a `WebGL2RenderingContext` you already own (implies `webgl2`) |
| `assetPath` | all | base URL for worker, `models/<name>.bin`, and fallback `.js`/`.wasm` when self-hosting `dist/` |
| `timeoutMs` | all | inference limit (default 120000; 900000 for `cpu` or `auto` in a worker) |
| `onLog` | all | receives diagnostic log lines |
| `onProgress` | all | receives monotonic `{ stage, progress, layer?, totalLayers?, message? }` updates |

Output is in the **input image's own space by default**, for every model.

### Bundlers & per-model subpaths

No copying and no `assetPath` are required with Vite, Rollup, or webpack: the
worker, `models/<name>.bin`, and fallback glue are referenced with literal
`new URL(..., import.meta.url)` and `import(...)` specifiers. Vite needs:

```js
// vite.config.js
export default {
  optimizeDeps: { exclude: ['@brainchop/core'] },
  worker: { format: 'es' },
}
```

To deploy only one model's assets, import its subpath entrypoint:

```js
import { segment } from '@brainchop/core/mindgrab'      // or /mindmap-fast, /16chan18cls, /mindmap, /mindsnap
```

## Errors

Every package error throws a `BrainchopError` with a machine-readable `code`:
`no-webgpu`, `no-adapter`, `device-too-small`, `no-f16`, `unsupported-option`,
`bad-input`, `initialization-failed`, `inference-failed`.

## Building & Testing

```sh
npm install
npm run build:webgpu                 # stage models/*.bin, emit types, bundle (no emcc needed!)
npm run makeWasm && npm run build    # full build including WebGL2 + CPU WASM fallbacks (needs emcc)
npm run test:unit                    # Node unit tests (parseModelBin parity, NIfTI, gpu-compat, conform/bwlabels/reslice)
npm test                             # real-browser end-to-end tests (needs playwright via BC_PLAYWRIGHT_ROOT)
```
