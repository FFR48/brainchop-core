#!/usr/bin/env node
// Compile WGSL in a real browser and report the compiler's diagnostics.
//
//   node test/wgsl_validate.mjs <file.wgsl> [more.wgsl ...]
//
// A browser's own WGSL compiler is the only authority that matters here, so
// this drives Chromium rather than a standalone validator: what it accepts is
// by definition what the shipped module can use. Requires the WebGPU adapter to
// come up; if it does not, that is reported as "cannot validate" rather than
// silently passing, because a validator that always succeeds is worse than none.

import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

// brainchop-core keeps test Playwright dependencies external; playwright lives in a sibling
// project. Resolve it explicitly rather than relying on cwd, since node
// resolves bare imports from the SCRIPT's directory.
// Relative, not an absolute home path: this file is committed, so a hardcoded
// home path makes `make check-wgsl` fail on every machine but one -- including
// the Linux box this backend actually exists for.
const roots = [
  process.env.BC_PLAYWRIGHT_ROOT,
  resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'BrowserQC'),
  resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'brainchop-test'),
].filter(Boolean);
const root = roots.find((r) => existsSync(join(r, 'node_modules', 'playwright')));
if (!root) {
  console.error('wgsl_validate: playwright not found; set BC_PLAYWRIGHT_ROOT');
  process.exit(2);
}
const require = createRequire(pathToFileURL(join(root, 'package.json')));
const { chromium } = require('playwright');

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: wgsl_validate.mjs <file.wgsl> [...]');
  process.exit(2);
}

// Headless Chromium on macOS runs the headless shell, which exposes no GPU and
// therefore no navigator.gpu. Validation needs the real browser, so this runs
// headed and briefly opens a window. BC_WGSL_HEADLESS=1 forces headless for
// environments where that does work.
const browser = await chromium.launch({
  headless: process.env.BC_WGSL_HEADLESS === '1',
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,WebGPU',
    '--use-angle=metal',
    '--enable-gpu',
  ],
});
const page = await browser.newPage();
// WebGPU is gated on a secure context, and about:blank is not one -- there
// navigator.gpu is simply undefined, which looks exactly like a browser without
// WebGPU support. A file:// page is a secure context.
const blank = join(tmpdir(), 'bc-wgsl-validate.html');
writeFileSync(blank, '<!doctype html><title>wgsl</title>');
await page.goto(pathToFileURL(blank).href, { waitUntil: 'domcontentloaded' });

const ready = await page.evaluate(async () => {
  if (!navigator.gpu) return { ok: false, why: 'navigator.gpu is undefined' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ok: false, why: 'requestAdapter() returned null' };
  return { ok: true, limits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  }, f16: adapter.features.has('shader-f16') };
});

if (!ready.ok) {
  console.error(`wgsl_validate: cannot validate - ${ready.why}`);
  await browser.close();
  process.exit(3);
}

console.log(`  adapter: maxBufferSize=${(ready.limits.maxBufferSize / 1048576).toFixed(0)} MiB` +
  `  maxStorageBufferBindingSize=${(ready.limits.maxStorageBufferBindingSize / 1048576).toFixed(0)} MiB` +
  `  shader-f16=${ready.f16}`);

let failed = 0;
for (const file of files) {
  const source = readFileSync(file, 'utf8');
  const result = await page.evaluate(async (src) => {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
    });
    // Swallow the device's own error scope so an invalid module surfaces here
    // as data rather than an unhandled console error.
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: src });
    const info = await module.getCompilationInfo();
    const scoped = await device.popErrorScope();
    return {
      messages: info.messages.map((m) => ({
        type: m.type, line: m.lineNum, col: m.linePos, message: m.message,
      })),
      scoped: scoped ? scoped.message : null,
    };
  }, source);

  const errors = result.messages.filter((m) => m.type === 'error');
  const warnings = result.messages.filter((m) => m.type === 'warning');
  const name = basename(file);
  if (errors.length || result.scoped) {
    failed++;
    console.error(`  FAIL ${name}`);
    for (const e of errors) console.error(`    ${e.line}:${e.col} ${e.message}`);
    if (result.scoped) console.error(`    validation: ${result.scoped}`);
  } else {
    console.log(`  ok   ${name}${warnings.length ? ` (${warnings.length} warning(s))` : ''}`);
    for (const w of warnings) console.log(`    ${w.line}:${w.col} ${w.message}`);
  }
}

await browser.close();
process.exit(failed ? 1 : 0);
