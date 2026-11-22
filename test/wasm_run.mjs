#!/usr/bin/env node
// Run a brainchop wasm module on an UNCOMPRESSED NIfTI and write the result.
//
//   node test/wasm_run.mjs <module.js> <input.nii> <output.nii> [args...]
//
// The module is built without zlib on purpose: gzip is the wrapper's job, via
// CompressionStream/DecompressionStream, so the module only ever sees raw
// NIfTI. This harness therefore takes and produces .nii, and the caller
// decompresses first -- which is exactly what the browser wrapper will do.

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';
import { pathToFileURL } from 'node:url';

const [modulePath, inputPath, outputPath, ...extra] = process.argv.slice(2);
if (!modulePath || !inputPath || !outputPath) {
  console.error('usage: wasm_run.mjs <module.js> <input.nii> <output.nii> [args...]');
  process.exit(2);
}

const factory = (await import(pathToFileURL(resolve(modulePath)).href)).default;

// Emscripten writes to stdout/stderr line by line; keep them separate so a
// nonzero exit can be reported with the message that caused it.
const err = [];
// The module is built with -sPROXY_TO_PTHREAD, so main() runs on a worker and
// callMain RETURNS BEFORE IT FINISHES -- reading /out.nii straight afterwards
// fails with ENOENT. onExit is what actually reports the end, so it is
// installed before the factory runs and awaited below. Same shape as
// js/src/run.ts, which needs it for ASYNCIFY rather than for pthreads.
let settle;
const exited = new Promise((resolve) => { settle = resolve; });
const Module = await factory({
  noInitialRun: true,
  // Emscripten defaults thisProgram to process.argv[1] -- this harness -- and
  // bc_cli_parse takes the program name for its usage text from argv[0], so
  // --help would otherwise identify itself as wasm_run.mjs.
  thisProgram: basename(modulePath, '.js'),
  print: () => {},
  printErr: (line) => err.push(line),
  onExit: (code) => settle(code),
});

const inName = '/in.nii';
const outName = '/out.nii';
Module.FS.writeFile(inName, new Uint8Array(readFileSync(inputPath)));

const argv = [...extra, '-o', outName, inName];
// callMain unshifts thisProgram into the array it is given, so keep a copy for
// reporting rather than printing an argument list we did not pass.
const shown = argv.join(' ');
const started = Date.now();
try {
  // The return value is deliberately IGNORED. Under -sPROXY_TO_PTHREAD callMain
  // returns 0 as soon as main has been handed to a worker, which is
  // indistinguishable from a serial build returning 0 after main finished --
  // so onExit, which -sEXIT_RUNTIME makes fire in both, is the only honest
  // signal. Settling on the return value read /out.nii before it existed.
  Module.callMain(argv);
} catch (e) {
  // callMain throws ExitStatus when the module exits synchronously; anything
  // else is a real failure. Settling here costs nothing -- a promise takes its
  // first resolution -- and covers the case where onExit never fires.
  if (e && typeof e.status === 'number') settle(e.status);
  else { console.error(err.join('\n')); throw e; }
}
/*
 * Bounded, because `exited` has no other floor. Under -sPROXY_TO_PTHREAD
 * callMain returns 0 without throwing, so the ExitStatus branch above is dead
 * in the normal path and onExit is the ONLY settle -- and if main traps or
 * deadlocks on the proxied thread, onExit never fires and node waits forever on
 * a pool that keeps its event loop alive. A gate that hangs is worse than one
 * that fails, so this fails.
 */
// The timer is CLEARED, not left pending: node keeps its event loop alive for
// an outstanding setTimeout, so a successful run would otherwise sit there for
// fifteen minutes after writing the output.
let timer;
const status = await Promise.race([
  exited,
  new Promise((resolve) => { timer = setTimeout(() => resolve('timeout'), 900_000); }),
]).finally(() => clearTimeout(timer));
if (status === 'timeout') {
  console.error(err.join('\n'));
  console.error(`wasm_run: ${basename(modulePath)} never exited`);
  process.exit(1);
}
const elapsed = ((Date.now() - started) / 1000).toFixed(1);

if (status !== 0) {
  console.error(err.join('\n'));
  console.error(`wasm_run: ${basename(modulePath)} exited ${status}`);
  process.exit(status);
}

writeFileSync(outputPath, Buffer.from(Module.FS.readFile(outName)));
console.error(`wasm_run: ${basename(modulePath)} ${shown} -> ${outputPath} (${elapsed}s)`);
