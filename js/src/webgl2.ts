import { BrainchopError } from './error.js'
import { ACTIVATION_BYTES, SOFTWARE_RENDERER, activationBytes } from './device.js'
import type { ModelName } from './types.js'

/*
 * WebGL2 capability negotiation, the fallback for browsers without WebGPU.
 *
 * The counterpart of device.ts, and deliberately shaped the same way: JS
 * acquires the context and hands it to the module through
 * `Module.preinitializedWebGLContext`, so every refusal can be phrased in the
 * caller's terms here rather than as a GL error inside C.
 *
 * Unlike device.ts this is entirely SYNCHRONOUS -- getContext is not a promise
 * -- which is also true of the backend behind it: glReadPixels blocks, so the
 * WebGL2 module needs no ASYNCIFY. See the warning on `segment` in index.ts
 * about what that means for the calling thread.
 */

/** Two activation sets ping-pong, so the device needs twice one activation. */
export function glActivationBytes(model: ModelName): number {
  return activationBytes(model) * 2
}

/**
 * Two activation sets for a 16-channel model.
 *
 * @deprecated a per-model number; use `glActivationBytes(model)`.
 */
export const GL_ACTIVATION_BYTES = glActivationBytes('16chan18cls')


/*
 * The limits backend_webgl2.c checks, restated so a refusal can name them
 * before the module is fetched (476 KB for model16, 734 KB for MindGrab).
 *
 * The 3D texture and the four colour attachments sit exactly on the GLES 3.0
 * floors, which is by design: 16 channels at RGBA16F is four 256^3 3D
 * textures, 512 MiB, and ping-ponging needs two such sets. LABEL_W sits on a
 * floor too -- 2048 is the GLES 3.0 MAX_TEXTURE_SIZE minimum -- and it needs
 * only half the width a 256^3 label volume suggests, because four voxels pack
 * into one RGBA8 texel.
 */
const DIM = 256

/*
 * One RGBA plane of the activation set: 256^3 texels x 4 channels x 2 bytes.
 * The SAME size at any channel stride -- a wider model uses MORE planes, not
 * bigger ones -- which is why this is written out rather than divided out of an
 * activation total, where the 16-channel arithmetic happens to agree and the
 * 24-channel arithmetic would not.
 */
const PLANE_BYTES = DIM * DIM * DIM * 4 * 2
const LABEL_W = 2048
const MiB = (n: number) => `${Math.round(n / (1024 * 1024))} MiB`

export interface Webgl2Report {
  supported: boolean
  /** Why not, in the caller's terms. Empty when supported. */
  reasons: string[]
  renderer?: string
  /**
   * true: a software rasteriser, which `auto` never picks. undefined: the
   * renderer string is masked, so `auto` prefers the CPU module to it.
   */
  software?: boolean
  max3dTextureSize?: number
  maxDrawBuffers?: number
  maxColorAttachments?: number
  maxTextureSize?: number
  /** True when a single 128 MiB activation plane actually allocated. */
  allocates?: boolean
}

function newContext(): WebGL2RenderingContext | null {
  // A detached canvas: this must not add anything to the host's DOM.
  //
  // It reaches the module through emscripten's `specialHTMLTargets` map, which
  // run.ts registers under '#brainchop-webgl2' before callMain. NOT through
  // `Module.canvas` -- emscripten 6 defaults
  // DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR to true, so a target is a
  // plain document.querySelector and Module.canvas is never consulted. That
  // mistaken belief is what made the first package build report "requested
  // backend 'webgl2' is unavailable"; it is written down here because this is
  // the file someone opens to ask how the context gets across.
  //
  // It is also why a Worker works at all: there is no document to query there.
  const canvas = typeof OffscreenCanvas !== 'undefined' && typeof document === 'undefined'
    ? new OffscreenCanvas(1, 1)
    : typeof document !== 'undefined' ? document.createElement('canvas') : null
  if (!canvas) return null
  return (canvas as HTMLCanvasElement).getContext('webgl2', {
    alpha: false, depth: false, stencil: false, antialias: false,
    preserveDrawingBuffer: false, powerPreference: 'high-performance',
  }) as WebGL2RenderingContext | null
}

/**
 * Ask whether this browser can run the models on WebGL2, without running one.
 *
 * Cheap and side-effect free apart from one texture allocation, which is the
 * only honest way to answer the question that matters: WebGL2 has no memory
 * query at all, so "will 1 GiB of activations fit" cannot be asked, only tried.
 * This tries a single 128 MiB plane -- an eighth of the real requirement --
 * which catches a device that is nowhere near without reserving what a real run
 * would need.
 */
export function checkWebgl2Support(model: ModelName = '16chan18cls'): Webgl2Report {
  const gl = newContext()
  if (!gl)
    return {
      supported: false,
      reasons: ['a webgl2 context could not be created; this browser is either ' +
        'too old or has WebGL disabled or blocklisted'],
    }

  const reasons: string[] = []
  const debug = gl.getExtension('WEBGL_debug_renderer_info')
  const renderer = debug
    ? (gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) as string)
    : (gl.getParameter(gl.RENDERER) as string)
  // The string only: failIfMajorPerformanceCaveat is also refused on real
  // hardware (Firefox, a flaky GPU process). Masked means unknown.
  const software = SOFTWARE_RENDERER.test(renderer) || (debug ? false : undefined)

  // The hard dependency. WebGL2 can SAMPLE float textures without this but
  // cannot RENDER to them, and every activation is a render target. Note it is
  // EXT_color_buffer_float specifically: Firefox does not expose
  // EXT_color_buffer_half_float at all, and half_float would not cover the
  // RGBA32F moment targets anyway.
  const float = gl.getExtension('EXT_color_buffer_float')
  if (!float)
    reasons.push('this device lacks EXT_color_buffer_float, which the kernels ' +
      'require in order to render into float textures')

  const max3d = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number
  const drawBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS) as number
  const attachments = gl.getParameter(gl.MAX_COLOR_ATTACHMENTS) as number
  const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
  const viewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array

  if (max3d < DIM)
    reasons.push(`MAX_3D_TEXTURE_SIZE is ${max3d}, but the model needs ${DIM}`)
  /*
   * PLANES attachments, not four. backend_webgl2.c checks `< BC_PLANES`, which
   * is CS/4 -- six for a 24-channel model. A literal four here reported a
   * four-draw-buffer device as SUPPORTED for a model that needs six, and the
   * module then failed at framebuffer setup with GL_INVALID_ENUM. A capability
   * check that is wrong in the permissive direction is worse than none.
   */
  const planes = activationBytes(model) / PLANE_BYTES
  if (drawBuffers < planes || attachments < planes)
    reasons.push(`this device allows ${drawBuffers} draw buffers and ` +
      `${attachments} colour attachments, but ${model} writes ${planes} at once`)
  if (maxTexture < LABEL_W || viewport[0] < LABEL_W || viewport[1] < LABEL_W)
    reasons.push(`MAX_TEXTURE_SIZE is ${maxTexture} and MAX_VIEWPORT_DIMS ` +
      `${viewport[0]}x${viewport[1]}, but the label texture is ${LABEL_W}x${LABEL_W}`)

  let allocates: boolean | undefined
  if (float && max3d >= DIM) {
    while (gl.getError() !== gl.NO_ERROR) { /* drain before measuring */ }
    const tex = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_3D, tex)
    gl.texStorage3D(gl.TEXTURE_3D, 1, gl.RGBA16F, DIM, DIM, DIM)
    allocates = gl.getError() === gl.NO_ERROR && !gl.isContextLost()
    gl.deleteTexture(tex)
    if (!allocates)
      reasons.push(`this device could not allocate even one ${MiB(PLANE_BYTES)} ` +
        `activation plane; a ${model} run needs ${MiB(glActivationBytes(model))} of them`)
  }

  // Release the probe context. Browsers cap live WebGL contexts (Chrome ~16)
  // and force-lose the OLDEST when the cap is passed -- so a host that probes
  // and segments repeatedly would eventually kill a context it is still using.
  // WEBGL_lose_context is the only way to release one deterministically;
  // dropping the reference merely waits for GC.
  gl.getExtension('WEBGL_lose_context')?.loseContext()

  return {
    supported: reasons.length === 0,
    reasons,
    renderer,
    software,
    max3dTextureSize: max3d,
    maxDrawBuffers: drawBuffers,
    maxColorAttachments: attachments,
    maxTextureSize: maxTexture,
    allocates,
  }
}

/**
 * Release a context this package created.
 *
 * Not called for a caller-supplied `glContext`: that one is theirs.
 */
export function releaseGlContext(gl: WebGL2RenderingContext): void {
  gl.getExtension('WEBGL_lose_context')?.loseContext()
}

/**
 * Acquire a context sized for the model, or refuse and say what was short.
 *
 * The returned context has EXT_color_buffer_float already enabled. That matters:
 * a WebGL extension is per-context state, and the module asks for it too, but
 * enabling it here means a device that cannot provide it is refused with a
 * sentence rather than a GL error 200 lines into C.
 */
export function acquireGlContext(): WebGL2RenderingContext {
  const gl = newContext()
  if (!gl)
    throw new BrainchopError('no-webgpu',
      'a webgl2 context could not be created; this browser is either too old ' +
      'or has WebGL disabled or blocklisted')
  if (!gl.getExtension('EXT_color_buffer_float'))
    throw new BrainchopError('no-f16',
      'this device lacks EXT_color_buffer_float, which the segmentation ' +
      'kernels require in order to render into float textures')
  const max3d = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number
  if (max3d < DIM)
    throw new BrainchopError('device-too-small',
      `this device allows 3D textures up to ${max3d}, but the model needs ${DIM}`)
  return gl
}
