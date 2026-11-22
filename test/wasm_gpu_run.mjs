#!/usr/bin/env node
// Run a WebGPU-enabled brainchop wasm module in a real browser.
//
//   node test/wasm_gpu_run.mjs <module.js> <input.nii> <output.nii> [args...]
//
// JS owns the asynchronous half -- requestAdapter/requestDevice -- and hands the
// ready device to the module through Module.preinitializedWebGPUDevice, which is
// what emscripten_webgpu_get_device() returns in C. That division is the whole
// reason backend_webgpu.c can stay synchronous.
//
// Served over http://127.0.0.1 rather than file://, for two independent
// reasons: WebGPU requires a secure context (about: is not one), and a file://
// origin cannot dynamically import the emscripten ES module.

import { existsSync, readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync } from 'node:fs';
import { basename, dirname, join, normalize, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createServer } from 'node:http';

const [modulePath, inputPath, outputPath, ...extra] = process.argv.slice(2);
const pve = extra.includes('--pve');
if (!modulePath || !inputPath || !outputPath) {
  console.error('usage: wasm_gpu_run.mjs <module.js> <input.nii> <output.nii> [args...]');
  process.exit(2);
}

// Relative, not an absolute home path: this must work in any checkout.
const root = [process.env.BC_PLAYWRIGHT_ROOT,
  resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'BrowserQC')]
  .filter(Boolean).find((r) => existsSync(join(r, 'node_modules', 'playwright')));
if (!root) { console.error('playwright not found; set BC_PLAYWRIGHT_ROOT'); process.exit(2); }
const require = createRequire(pathToFileURL(join(root, 'package.json')));
const { chromium } = require('playwright');

// Stage the module, its .wasm and the input where one static server can see them.
const stage = mkdtempSync(join(tmpdir(), 'bc-gpu-'));
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

// The result comes back by POST, NOT as a page.evaluate return value.
//
// Returning it was the obvious thing and it cost 20+ SECONDS per case. A
// page.evaluate return crosses the CDP bridge as JSON, so `Array.from(bytes)`
// on a 256^3 volume ships 16.7 million decimal numbers -- about 50 MB of JSON
// to encode, transfer and parse for 16 MB of payload. That overhead lands in
// the same wall-clock number the harness prints as the model's time, and it
// scales with the OUTPUT size, which is why it read as "mindgrab is 28 s and
// model16-at-2mm is 4 s" when the two actually differ by 1.7 s of GPU work.
// A POST is a plain binary body and takes milliseconds.
const received = new Map();
const types = { '.html': 'text/html', '.js': 'text/javascript',
  '.wasm': 'application/wasm', '.nii': 'application/octet-stream' };
const server = createServer((req, res) => {
  if (req.method === 'POST' && /^\/result(?:_(gm|wm|csf))?$/.test(req.url)) {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => { received.set(req.url, Buffer.concat(chunks)); res.writeHead(204); res.end(); });
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

// Headed: headless Chromium on macOS runs the headless shell, which has no GPU.
const browser = await chromium.launch({ headless: process.env.BC_WGSL_HEADLESS === '1',
  channel: process.env.BC_CHROME_CHANNEL || undefined, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('  pageerror:', e.message));
await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'domcontentloaded' });

const started = Date.now();
const result = await page.evaluate(async ({ jsName, args, pve }) => {
  const log = [];
  let ran = 0;
  try {
    if (!navigator.gpu) return { status: -1, log: ['navigator.gpu is undefined'] };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { status: -1, log: ['requestAdapter returned null'] };
    // Ask for whatever the adapter will give. The elevated limits must be
    // requested explicitly -- the defaults are 128/256 MiB -- and a fixed
    // ceiling here is a per-model number in disguise: 16 channels need 512 MiB
    // per activation, 24 channels need 805 MiB and a 1.5 GiB scheduled graph.
    // A 1 GiB request silently killed the renderer on the wider model.
    const want = Number.MAX_SAFE_INTEGER;
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
      requiredLimits: {
        maxBufferSize: Math.min(want, adapter.limits.maxBufferSize),
        maxStorageBufferBindingSize:
          Math.min(want, adapter.limits.maxStorageBufferBindingSize),
      },
    });
    log.push(`limits: maxBufferSize=${device.limits.maxBufferSize} ` +
      `maxStorageBufferBindingSize=${device.limits.maxStorageBufferBindingSize}`);
    device.addEventListener('uncapturederror', (e) => log.push('gpu error: ' + e.error.message));

    const bytes = new Uint8Array(await (await fetch('./input.nii')).arrayBuffer());
    const factory = (await import(`./${jsName}`)).default;

    let settle;
    const finished = new Promise((r) => { settle = r; });
    const Module = await factory({
      noInitialRun: true,
      thisProgram: jsName.replace(/\.js$/, ''),
      preinitializedWebGPUDevice: device,
      print: (l) => log.push(l),
      printErr: (l) => log.push(l),
      onExit: (code) => settle(code),
    });
    Module.FS.writeFile('/in.nii', bytes);
    // ASYNCIFY: callMain returns at the first yield, not at exit. Waiting on it
    // would report success before any GPU work had happened.
    const t = performance.now();
    try { Module.callMain([...args, '-o', '/out.nii', '/in.nii']); }
    catch (e) {
      if (!(e && typeof e.status === 'number')) {
        log.push('threw: ' + (e && e.stack ? e.stack : String(e)));
        settle(-1);
      }
    }
    const status = await Promise.race([
      finished,
      new Promise((r) => setTimeout(() => r('timeout'), 300000)),
    ]);
    // The number worth reporting: input in memory to exit, so it covers the
    // conform, the GPU dispatches and the readback, and nothing about how the
    // harness moves bytes around.
    ran = Math.round(performance.now() - t);
    if (status === 0) {
      for (const suffix of pve ? ['_gm','_wm','_csf'] : ['']) {
        const data = Module.FS.readFile(`/out${suffix}.nii`);
        await fetch(`/result${suffix}`, { method: 'POST', body: data });
      }
    }
    return { status, log, ran };
  } catch (e) { return { status: -1, log: [...log, 'outer: ' + (e && e.stack ? e.stack : String(e))], ran }; }
}, { jsName, args: extra, pve });

const elapsed = ((Date.now() - started) / 1000).toFixed(1);
for (const l of result.log) console.error('  ' + l);
await browser.close();
server.close();

if (result.status !== 0 || received.size !== (pve ? 3 : 1)) {
  console.error(`  wasm_gpu_run: ${jsName} did not complete (status ${result.status})`);
  process.exit(1);
}
for (const suffix of pve ? ['_gm','_wm','_csf'] : [''])
  writeFileSync(suffix ? outputPath.replace(/\.nii$/,`${suffix}.nii`) : outputPath, received.get(`/result${suffix}`));
console.error(`  wasm_gpu_run: ${jsName} -> ${outputPath} ` +
  `(${(result.ran / 1000).toFixed(2)}s in the module, ${elapsed}s including the harness)`);
