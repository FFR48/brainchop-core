#!/usr/bin/env node
// Run a WebGL2-enabled brainchop wasm module in a real browser.
//
//   node test/wasm_gl_run.mjs <module.js> <input.nii> [output.nii] [args...]
//
// The WebGL2 counterpart of wasm_gpu_run.mjs, and structured identically on
// purpose. Two things differ:
//
//   - JS creates the WebGL2 CONTEXT and hands it over through
//     Module.preinitializedWebGLContext, which is what
//     emscripten_webgl_create_context() returns in C. Same division of labour
//     as the WebGPU harness, for the same reason: capability refusals belong
//     where they can be phrased in the caller's terms.
//   - There is no ASYNCIFY. glReadPixels blocks, so callMain runs to
//     completion and its return is meaningful -- unlike the WebGPU module,
//     where callMain returns at the first yield.
//
// The output volume is optional: a BC_WEBGL2_PHASE1 module verifies layer 0 and produces
// no segmentation, so the exit status and the printed log are the result.
//
// Served over http://127.0.0.1 rather than file:// because a file:// origin
// cannot dynamically import the emscripten ES module. WebGL2, unlike WebGPU,
// does not itself require a secure context.

import { existsSync, readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync } from 'node:fs';
import { basename, dirname, join, normalize, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createServer } from 'node:http';

const argv = process.argv.slice(2);
const [modulePath, inputPath] = argv;
// An output path is anything in third position that is not a flag.
const hasOutput = argv[2] && !argv[2].startsWith('-');
const outputPath = hasOutput ? argv[2] : null;
const extra = argv.slice(hasOutput ? 3 : 2);
if (!modulePath || !inputPath) {
  console.error('usage: wasm_gl_run.mjs <module.js> <input.nii> [output.nii] [args...]');
  process.exit(2);
}

// Relative, not an absolute home path: this must work in any checkout.
const root = [process.env.BC_PLAYWRIGHT_ROOT,
  resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'BrowserQC')]
  .filter(Boolean).find((r) => existsSync(join(r, 'node_modules', 'playwright')));
if (!root) { console.error('playwright not found; set BC_PLAYWRIGHT_ROOT'); process.exit(2); }
const require = createRequire(pathToFileURL(join(root, 'package.json')));
const playwright = require('playwright');
const browserName = process.env.BC_GL_BROWSER || 'chromium';

// Stage the module, its .wasm and the input where one static server can see them.
const stage = mkdtempSync(join(tmpdir(), 'bc-gl-'));
/*
 * Removed in a handler, not at the end of the file: the failure paths below
 * leave through process.exit() or an uncaught rejection, and both would skip a
 * trailing rmSync. rmSync is synchronous, so it is legal in one, and swallowed
 * because a throw from an 'exit' handler is printed as an uncaught exception --
 * and on a natural exit, which is how this file ends, it also turns 0 into 1.
 *
 * The retries are for Windows, where an open .wasm gives EBUSY; here removal
 * takes under a millisecond. They cost ~2 s when they do fire, which is
 * affordable only because this runs after every assertion.
 *
 * Signals are covered by accident, and only while a browser is open: playwright
 * installs its own SIGINT/SIGTERM handlers at launch, and its graceful close
 * rejects the pending evaluate, so the run ends through an ordinary exit that
 * DOES fire this handler -- measured, exit 1 and the directory removed. Before
 * launch and after close, bare node semantics return: no 'exit', and the
 * directory survives. Left that way on purpose; our own signal handler would
 * have to call process.exit() and would fabricate a status on a gate harness.
 */
process.on('exit', () => {
  try { rmSync(stage, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); }
  catch (e) { console.error(`could not remove ${stage}: ${e.message}`); }
});

const jsName = basename(modulePath);
const wasmName = jsName.replace(/\.js$/, '.wasm');
copyFileSync(resolve(modulePath), join(stage, jsName));
copyFileSync(join(dirname(resolve(modulePath)), wasmName), join(stage, wasmName));
copyFileSync(resolve(inputPath), join(stage, 'input.nii'));
writeFileSync(join(stage, 'index.html'), '<!doctype html><title>bc</title>');

// The result comes back by POST, NOT as a page.evaluate return value -- see the
// note in wasm_gpu_run.mjs. Returning a 256^3 volume through CDP cost 20+
// seconds and landed inside the number reported as the model's time.
let received = null;
const types = { '.html': 'text/html', '.js': 'text/javascript',
  '.wasm': 'application/wasm', '.nii': 'application/octet-stream' };
const server = createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/result') {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => { received = Buffer.concat(chunks); res.writeHead(204); res.end(); });
    return;
  }
  // Normalized before joining: the raw request target would otherwise let
  // `/../../etc/passwd` escape the staging directory while the harness runs.
  const raw = req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0]);
  const file = join(stage, normalize(raw).replace(/^(\.\.[/\\])+/, ''));
  if (!file.startsWith(stage) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  const ext = file.slice(file.lastIndexOf('.'));
  res.writeHead(200, { 'Content-Type': types[ext] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

// Headed: headless Chromium on macOS runs the headless shell, which has no GPU
// and would silently fall back to a software rasteriser -- not the thing that
// ships, and slow enough to look like a hang.
const launchArgs = browserName === 'chromium'
  ? { headless: process.env.BC_WGSL_HEADLESS === '1',
      args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'] }
  : { headless: process.env.BC_WGSL_HEADLESS === '1' };
const browser = await playwright[browserName].launch(launchArgs);
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('  pageerror:', e.message));
await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'domcontentloaded' });

const started = Date.now();
const result = await page.evaluate(async ({ jsName, args, wantOutput }) => {
  const log = [];
  let ran = 0;
  try {
    // Detached on purpose: backend_webgl2.c looks its canvas up through
    // emscripten's specialHTMLTargets map, not through the DOM, so nothing has
    // to be appended anywhere.
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', {
      alpha: false, depth: false, stencil: false, antialias: false,
      preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) return { status: -1, log: ['no webgl2 context'] };
    // Without this no float texture is renderable. Enabling it here rather than
    // in C is deliberate: the module asks for it too, but a refusal phrased
    // here can name the browser.
    if (!gl.getExtension('EXT_color_buffer_float'))
      return { status: -1, log: ['EXT_color_buffer_float is not available'] };
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    log.push('renderer: ' + (dbg
      ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER)));

    const bytes = new Uint8Array(await (await fetch('./input.nii')).arrayBuffer());
    const factory = (await import(`./${jsName}`)).default;

    let exitCode = null;
    const Module = await factory({
      noInitialRun: true,
      thisProgram: jsName.replace(/\.js$/, ''),
      preinitializedWebGLContext: gl,
      print: (l) => log.push(l),
      printErr: (l) => log.push(l),
      onExit: (code) => { exitCode = code; },
    });
    // Register the canvas under the key the C asks for. Must happen before
    // callMain, which is when emscripten_webgl_create_context runs.
    Module.specialHTMLTargets['#brainchop-webgl2'] = canvas;
    Module.FS.writeFile('/in.nii', bytes);

    const t = performance.now();
    try { Module.callMain([...args, '-o', '/out.nii', '/in.nii']); }
    catch (e) {
      // EXIT_RUNTIME makes a returning main throw ExitStatus; that is a normal
      // exit, not a failure.
      if (e && typeof e.status === 'number') exitCode = e.status;
      else { log.push('threw: ' + (e && e.stack ? e.stack : String(e))); exitCode = -1; }
    }
    ran = Math.round(performance.now() - t);

    if (exitCode === 0 && wantOutput) {
      try {
        await fetch('/result', { method: 'POST', body: Module.FS.readFile('/out.nii') });
      } catch (e) { log.push('no output volume: ' + String(e)); }
    }
    return { status: exitCode === null ? -1 : exitCode, log, ran };
  } catch (e) {
    return { status: -1, log: [...log, 'outer: ' + (e && e.stack ? e.stack : String(e))], ran };
  }
}, { jsName, args: extra, wantOutput: !!outputPath });

const elapsed = ((Date.now() - started) / 1000).toFixed(1);
for (const l of result.log) console.error('  ' + l);
await browser.close();
server.close();

if (result.status !== 0) {
  console.error(`  wasm_gl_run: ${jsName} did not complete (status ${result.status})`);
  process.exit(1);
}
if (outputPath) {
  if (!received) {
    console.error(`  wasm_gl_run: ${jsName} produced no output volume`);
    process.exit(1);
  }
  writeFileSync(outputPath, received);
}
console.error(`  wasm_gl_run: ${jsName}${outputPath ? ' -> ' + outputPath : ''} ` +
  `(${(result.ran / 1000).toFixed(2)}s in the module, ${elapsed}s including the harness)`);
