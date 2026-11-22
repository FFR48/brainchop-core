import { BrainchopError } from './error.js'
import { canonicalModelName } from './device.js'
import type { ModelName } from './types.js'
import { MODULES, type WasmBackend } from './modules.js'

/*
 * Loading and driving one wasm module.
 *
 * The emscripten glue is NOT bundled into this package's own JavaScript. It is
 * staged into dist/ beside it and imported lazily, so a caller pays for the
 * model and backend they ask for rather than for all of them. The specifiers
 * are literal (modules.js), so a host bundler follows them and emits each glue
 * with the .wasm it locates through its own `import.meta.url`.
 */

/* Generous: a 256^3 volume is seconds, so this only fires on a real stall. */
const DEFAULT_TIMEOUT_MS = 120_000
// Initialisation is loading and instantiating a module, not processing a volume.
// It should never inherit the deliberately generous CPU inference allowance.
const INITIALIZATION_TIMEOUT_MS = 120_000
/*
 * The CPU module is not a stalled GPU module; it is a correct one doing tens of
 * billions of multiply-adds. 10 s for MindGrab on 14 cores, so a dual-core
 * laptop is well inside a minute or two -- but the GPU default would turn that
 * into a spurious failure, which is precisely the machine that has no GPU path
 * to fall back to.
 */
export const CPU_TIMEOUT_MS = 900_000

/*
 * One module per model PER BACKEND, and that separation is the packaging
 * decision, not an accident: a WebGPU caller never downloads the WebGL2 build,
 * and deleting the fallback later removes files rather than editing a shared
 * one. The probe in index.ts runs before any of these is fetched. The CPU
 * modules have no suffix: they are the plain `make wasm` output, and the only
 * ones built with -pthread.
 */
const SUFFIX: Record<WasmBackend, string> = { webgl2: '-gl', cpu: '' }

type EmscriptenFactory = (config: Record<string, unknown>) => Promise<EmscriptenModule>

interface EmscriptenModule {
  callMain: (args: string[]) => void
  /** Emscripten's event-target map; see the canvas note in run(). */
  specialHTMLTargets?: Record<string, unknown>
  FS: {
    writeFile: (path: string, data: Uint8Array) => void
    readFile: (path: string) => Uint8Array
    analyzePath: (path: string) => { exists: boolean }
  }
}

const factories = new Map<string, Promise<EmscriptenFactory>>()

/**
 * Where the emscripten glue and its .wasm are served from, when the host set
 * `assetPath` (it copied dist/ somewhere served). Without it, run() imports the
 * glue by a literal specifier (modules.js), which bundlers follow.
 */
export function assetUrl(file: string, assetPath: string): string {
  const base = assetPath.endsWith('/') ? assetPath : `${assetPath}/`
  const here = typeof location !== 'undefined' ? location.href : 'file:///'
  const url = new URL(`${base}${file.replace('./', '')}`, here)
  // Same-origin only, and one implementation of that check rather than two:
  // this feeds a dynamic import() here and `new Worker()` in index.ts, and two
  // copies of a security check are two that can drift apart.
  //
  // An absolute cross-origin value would execute a third party's script in the
  // host page and hand it the caller's GPUDevice and the patient's volume. An
  // opaque origin (file://, a sandboxed iframe) is refused outright rather than
  // compared: there `location.origin` is the STRING "null", which equals the
  // origin of any data: URL, so the comparison would pass one through.
  if (typeof location !== 'undefined') {
    if (location.origin === 'null')
      throw new BrainchopError('unsupported-option',
        'assetPath cannot be used from an opaque origin (file:// or a sandboxed ' +
        'iframe); serve the page over http(s)')
    if (url.origin !== location.origin)
      throw new BrainchopError('unsupported-option',
        `assetPath must be same-origin; ${url.origin} is not ${location.origin}`)
  }
  return url.href
}

function loadFactory(backend: WasmBackend, model: ModelName, assetPath?: string): [string, Promise<EmscriptenFactory>] {
  const canonical = canonicalModelName(model)
  const key = assetPath ? assetUrl(`brainchop-${canonical}${SUFFIX[backend]}.js`, assetPath) : `${backend}/${canonical}`
  let pending = factories.get(key)
  if (!pending) {
    const loader = MODULES[backend][model] ?? MODULES[backend][canonical]
    pending = (assetPath ? import(/* @vite-ignore */ key) : loader())
      .then((m) => m.default as EmscriptenFactory)
    factories.set(key, pending)
    // A transient missing asset must not poison every later Apply in this page.
    void pending.catch(() => {
      if (factories.get(key) === pending) factories.delete(key)
    })
  }
  return [key, pending]
}

export interface RunRequest {
  model: ModelName
  backend: WasmBackend
  /** Required for the webgl2 backend. */
  glContext?: WebGL2RenderingContext
  input: Uint8Array
  /** CLI arguments, excluding the input path and `-o`. */
  args: string[]
  /** Extra files to collect from MEMFS afterwards, by path. */
  extraOutputs?: string[]
  /** Primary output differs for multi-map tissue inference. */
  primaryOutput?: string
  /** Base URL for the emscripten glue and .wasm; see moduleUrl(). */
  assetPath?: string
  /** Abort if the module has not exited within this many ms. */
  timeoutMs?: number
  onLog?: (line: string) => void
}

export interface RunResult {
  image: Uint8Array
  extras: Map<string, Uint8Array>
  elapsedMs: number
}

/**
 * Run one segmentation to completion.
 *
 * The subtlety is the exit: the module is built with ASYNCIFY so the C can wait
 * synchronously on the GPU readback, and under ASYNCIFY `callMain` RETURNS at
 * the first yield rather than at exit. Awaiting it would resolve before any GPU
 * work had happened and report success on an empty output. The module is linked
 * with `-sEXIT_RUNTIME=1` so `onExit` fires for real, and that is what is
 * awaited here.
 */
export async function run(request: RunRequest): Promise<RunResult> {
  const log: string[] = []
  const record = (line: string) => {
    log.push(line)
    request.onLog?.(line)
  }

  let settle: (code: number) => void
  const exited = new Promise<number>((resolve) => { settle = resolve })
  let timer: ReturnType<typeof setTimeout> | undefined

  /*
   * The handover. WebGPU gets a device; WebGL2 gets a context, plus a canvas
   * registration done after the module exists -- see below.
   */
  const config: Record<string, unknown> = {
    noInitialRun: true,
    // bc_cli_parse takes the program name from argv[0]; without this the module
    // would identify itself as whatever host script loaded it.
    thisProgram: `brainchop-${canonicalModelName(request.model)}`,
    print: record,
    printErr: record,
    onExit: (code: number) => settle(code),
  }
  if (request.backend === 'webgl2') {
    config.preinitializedWebGLContext = request.glContext
  }
  // cpu takes no handle at all: the module sizes its own pool from
  // navigator.hardwareConcurrency.
  const limit = request.timeoutMs
    ?? (request.backend === 'cpu' ? CPU_TIMEOUT_MS : DEFAULT_TIMEOUT_MS)

  /*
   * Glue loading AND the factory are RACED, not awaited. The CPU module is why:
   * it imports SHARED memory, and on a page that is not cross-origin isolated
   * Emscripten's pool spawn fails inside a promise nobody holds --
   *   DataCloneError: SharedArrayBuffer transfer requires self.crossOriginIsolated
   * -- so factory() neither resolves nor rejects and a bare await hangs for
   * good. checkCpuSupport() in index.ts refuses that page before we get here,
   * which makes this the second line rather than the first: a host that has
   * SharedArrayBuffer and passes the probe can still fail to start threads, and
   * a hang is the one outcome a caller cannot recover from.
   *
   * It is NOT gated on the backend, which gives WebGPU and WebGL2 an
   * initialisation timeout they did not have before. That is deliberate:
   * fetching and instantiating a module is asynchronous on all three, so the
   * timer can fire on all three, and a module that never instantiates is the
   * same dead end whichever backend asked for it. It covers both the dynamic
   * import and instantiation, and is capped at two minutes even for CPU so a
   * stuck setup does not look like a slow inference. The inference timeout
   * below still uses the caller's full `timeoutMs` and cannot fire for a
   * synchronous WebGL2 run.
   */
  const initializationLimit = Math.min(limit, INITIALIZATION_TIMEOUT_MS)
  const [key, factoryPromise] = loadFactory(request.backend, request.model, request.assetPath)
  let initializationTimedOut = false
  const module = await Promise.race([
    factoryPromise
      .then((createModule) => createModule(config)),
    new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        initializationTimedOut = true
        reject(new BrainchopError('initialization-failed',
          `the ${request.model} module did not finish initialising within ` +
          `${initializationLimit} ms`, log))
      }, initializationLimit)
    }),
  ]).finally(() => {
    clearTimeout(timer)
    // `factory(config)` is not cached, but a dynamic import that never settled
    // is. Forget it after the watchdog so a later Apply can retry.
    if (initializationTimedOut && factories.get(key) === factoryPromise) factories.delete(key)
  })

  /*
   * Register the canvas under the key backend_webgl2.c asks for, before
   * callMain -- which is when emscripten_webgl_create_context runs.
   *
   * NOT `Module.canvas`, and not the "#canvas" selector. Emscripten 6 defaults
   * DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR to true, which makes the
   * target a plain document.querySelector and never consults Module.canvas. A
   * first attempt passed the canvas that way and the module reported
   * "requested backend 'webgl2' is unavailable" -- it had found no canvas at
   * all. Earlier harnesses only worked because their page happened to contain
   * <canvas id="canvas">.
   *
   * specialHTMLTargets is checked before the DOM, so this works for a detached
   * canvas here and for an OffscreenCanvas in a Worker, and cannot collide with
   * a canvas the host page owns.
   */
  if (request.backend === 'webgl2') {
    if (!module.specialHTMLTargets)
      throw new BrainchopError('inference-failed',
        'this WebGL2 module was built without specialHTMLTargets exported; ' +
        'rebuild it with the flags in src/Makefile', log)
    module.specialHTMLTargets['#brainchop-webgl2'] = request.glContext!.canvas
  }

  // Uncompressed only, and named `.nii` so the C does not try to gzip: the
  // module has no zlib, which is the whole reason gzip.ts exists.
  module.FS.writeFile('/in.nii', request.input)

  const started = performance.now()
  try {
    module.callMain([...request.args, '-backend', request.backend,
      '-o', '/out.nii', '/in.nii'])
  } catch (e) {
    // Emscripten throws ExitStatus for a normal exit under some configurations;
    // anything without a numeric status is a genuine fault.
    if (e && typeof (e as { status?: unknown }).status === 'number') {
      // ExitStatus IS the normal exit under EXIT_RUNTIME, and onExit has
      // almost certainly fired already -- but settling here too costs nothing
      // (resolve is idempotent) and removes the one path that could otherwise
      // hang forever: the WebGL2 branch awaits `exited` with no timeout,
      // because a synchronous module gives a timer nothing to fire on.
      settle!((e as { status: number }).status)
    } else {
      record(`threw: ${e instanceof Error ? e.stack ?? e.message : String(e)}`)
      settle!(-1)
    }
  }
  /*
   * A timeout, because `exited` has two settle paths and a lost device takes
   * neither: under ASYNCIFY callMain returns at the first yield, so if the C is
   * parked waiting on a GPU readback that never completes, onExit never fires
   * and the promise would hang forever with ~1 GiB of device memory held.
   *
   * It applies to every path EXCEPT WebGL2, and that is a statement of fact
   * rather than a policy: the WebGL2 module has no ASYNCIFY, so callMain runs
   * the whole segmentation to completion before returning and `exited` has
   * already settled by the time this line is reached. A timer cannot fire while
   * a synchronous call holds the thread, so promising one would be a lie. The
   * CPU module is timeable for the same reason the WebGPU one is, by a
   * different mechanism: -sPROXY_TO_PTHREAD moves main() onto a worker, so
   * callMain returns immediately and this thread stays free to run the timer.
   *
   * It STOPS WAITING; it does not stop the module. That is true of every
   * backend and matters most for cpu, whose pool keeps every core busy and its
   * ~2.15 GB alive for the life of the page. `worker: true` is the only real
   * cancellation -- terminate() ends the threads -- which is one more reason it
   * is the recommended way to call this.
   */
  const code = request.backend === 'webgl2'
    ? await exited
    : await Promise.race([
      exited,
      new Promise<number>((_, reject) => {
        timer = setTimeout(() => reject(new BrainchopError('inference-failed',
          `the ${request.model} module did not finish within ${limit} ms; ` +
          (request.backend === 'cpu'
            ? 'this machine may simply be slower than the limit allows'
            : 'the GPU device may have been lost'), log)), limit)
      }),
    ]).finally(() => clearTimeout(timer))
  const elapsedMs = performance.now() - started

  /*
   * An independent context-loss detector, which is the whole point of it.
   *
   * The C side's only notice is glGetError, which reports
   * GL_CONTEXT_LOST_WEBGL exactly once -- so any drain anywhere can consume it
   * and leave the module reporting success over a volume it never wrote.
   * isContextLost() queries the object's own state and cannot be swallowed by
   * anything. Checked BEFORE the exit status so a lost context is named as
   * such rather than as a generic failure.
   */
  if (request.backend === 'webgl2' && request.glContext?.isContextLost())
    throw new BrainchopError('inference-failed',
      `the WebGL2 context was lost while running ${request.model}; the GPU or ` +
      'its driver reset. Any output from this run is meaningless.', log)

  if (code !== 0)
    throw new BrainchopError('inference-failed',
      `the ${request.model} module exited with status ${code}`, log)
  const primary = request.primaryOutput ?? '/out.nii'
  if (!module.FS.analyzePath(primary).exists)
    throw new BrainchopError('inference-failed',
      `the ${request.model} module exited cleanly but wrote no output`, log)

  const extras = new Map<string, Uint8Array>()
  for (const path of request.extraOutputs ?? [])
    if (module.FS.analyzePath(path).exists) extras.set(path, module.FS.readFile(path))

  return { image: module.FS.readFile(primary), extras, elapsedMs }
}
