#!/usr/bin/env node
// Compile this build's GLSL in a real browser and report the compiler's
// diagnostics.
//
//   node test/glsl_validate.mjs <file.glsl> [more.glsl ...]
//
// A browser's own GLSL compiler is the only authority that matters: what it
// accepts is what the shipped module can use. This is the WebGL2 counterpart of
// wgsl_validate.mjs, and it is STRICTER than that one in a way worth knowing --
// linkProgram validates the MRT output locations, the uniform set and the
// varying interface, none of which createShaderModule has an analogue for. A
// mismatched `layout(location = 3) out` fails here and would otherwise fail at
// run time as a blank attachment.
//
// Runs on Chromium AND Firefox by default. They do not share a compiler --
// Chromium goes through ANGLE, Firefox uses its own translator -- and Firefox
// is the browser this backend exists for, so validating only one would miss
// exactly the case that matters. BC_GLSL_BROWSERS overrides the list.

import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

// brainchop-core keeps test Playwright dependencies external; playwright lives in a sibling
// project. Resolve it explicitly rather than relying on cwd, since node
// resolves bare imports from the SCRIPT's directory.
// Relative, not an absolute home path: this file is committed, so a hardcoded
// home path makes `make check-glsl` fail on every machine but one -- including
// the Linux box this backend actually exists for.
const roots = [
  process.env.BC_PLAYWRIGHT_ROOT,
  resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'BrowserQC'),
  resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'brainchop-test'),
].filter(Boolean);
const root = roots.find((r) => existsSync(join(r, 'node_modules', 'playwright')));
if (!root) {
  console.error('glsl_validate: playwright not found; set BC_PLAYWRIGHT_ROOT');
  process.exit(2);
}
const require = createRequire(pathToFileURL(join(root, 'package.json')));
const playwright = require('playwright');

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: glsl_validate.mjs <file.glsl> [...]');
  process.exit(2);
}

// Split the dump into { name, vertex, fragment }. dump_glsl.c emits a marker
// line before each stage; anything before the first marker is a hard error
// rather than silently-skipped text, because a dump that produced nothing would
// otherwise pass.
function parse(text, file) {
  const programs = new Map();
  const re = /^\/\/--- program (\S+) (vertex|fragment)$/;
  let current = null;
  for (const line of text.split('\n')) {
    const m = re.exec(line);
    if (m) {
      if (!programs.has(m[1])) programs.set(m[1], { name: m[1] });
      current = { program: programs.get(m[1]), stage: m[2] };
      current.program[m[2]] = '';
      continue;
    }
    if (!current) {
      if (line.trim()) throw new Error(`${file}: text before the first marker`);
      continue;
    }
    current.program[current.stage] += line + '\n';
  }
  const out = [...programs.values()];
  if (!out.length) throw new Error(`${file}: no programs found`);
  for (const p of out)
    if (!p.vertex || !p.fragment)
      throw new Error(`${file}: program ${p.name} is missing a stage`);
  return out;
}

const browsers = (process.env.BC_GLSL_BROWSERS || 'chromium,firefox').split(',');
// Headless Chromium on macOS runs the headless shell, which exposes no GPU.
// WebGL2 is more forgiving than WebGPU here (it can fall back to SwiftShader),
// but a software compiler is not the one that ships, so this runs headed like
// the WGSL validator. BC_WGSL_HEADLESS=1 forces headless.
const headless = process.env.BC_WGSL_HEADLESS === '1';

let failed = 0;
for (const name of browsers) {
  const launchArgs = name === 'chromium'
    ? { headless, args: ['--enable-gpu', '--use-angle=metal', '--ignore-gpu-blocklist'] }
    : { headless };
  let browser;
  try {
    browser = await playwright[name].launch(launchArgs);
  } catch (e) {
    console.error(`glsl_validate: cannot launch ${name} - ${e.message.split('\n')[0]}`);
    failed++;
    continue;
  }
  const page = await browser.newPage();
  const blank = join(tmpdir(), 'bc-glsl-validate.html');
  writeFileSync(blank, '<!doctype html><title>glsl</title><canvas id=c></canvas>');
  await page.goto(pathToFileURL(blank).href, { waitUntil: 'domcontentloaded' });

  const info = await page.evaluate(() => {
    const gl = document.getElementById('c').getContext('webgl2');
    if (!gl) return { ok: false, why: 'no webgl2 context' };
    return {
      ok: true,
      renderer: (() => {
        const d = gl.getExtension('WEBGL_debug_renderer_info');
        return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
      })(),
      colorBufferFloat: gl.getExtension('EXT_color_buffer_float') !== null,
    };
  });
  if (!info.ok) {
    console.error(`glsl_validate: ${name} cannot validate - ${info.why}`);
    failed++;
    await browser.close();
    continue;
  }
  console.log(`  ${name}: ${info.renderer}` +
    `  EXT_color_buffer_float=${info.colorBufferFloat}`);

  for (const file of files) {
    let programs;
    try {
      programs = parse(readFileSync(file, 'utf8'), basename(file));
    } catch (e) {
      console.error(`    FAIL ${e.message}`);
      failed++;
      continue;
    }
    for (const p of programs) {
      const r = await page.evaluate(({ vertex, fragment }) => {
        const gl = document.getElementById('c').getContext('webgl2');
        const build = (type, src) => {
          const s = gl.createShader(type);
          gl.shaderSource(s, src);
          gl.compileShader(s);
          const ok = gl.getShaderParameter(s, gl.COMPILE_STATUS);
          return { s, ok, log: gl.getShaderInfoLog(s) || '' };
        };
        const v = build(gl.VERTEX_SHADER, vertex);
        const f = build(gl.FRAGMENT_SHADER, fragment);
        if (!v.ok || !f.ok)
          return { ok: false, vertexLog: v.log, fragmentLog: f.log, linkLog: '' };
        const prog = gl.createProgram();
        gl.attachShader(prog, v.s);
        gl.attachShader(prog, f.s);
        gl.linkProgram(prog);
        const ok = gl.getProgramParameter(prog, gl.LINK_STATUS);
        const linkLog = gl.getProgramInfoLog(prog) || '';
        gl.deleteProgram(prog);
        gl.deleteShader(v.s);
        gl.deleteShader(f.s);
        return { ok, vertexLog: v.log, fragmentLog: f.log, linkLog };
      }, { vertex: p.vertex, fragment: p.fragment });

      const label = `${basename(file)}:${p.name}`;
      const logs = [
        r.vertexLog && `vertex: ${r.vertexLog.trim()}`,
        r.fragmentLog && `fragment: ${r.fragmentLog.trim()}`,
        r.linkLog && `link: ${r.linkLog.trim()}`,
      ].filter(Boolean);
      if (!r.ok) {
        failed++;
        console.error(`    FAIL ${label}`);
        for (const l of logs) console.error(`      ${l}`);
      } else {
        // A warning is reported but does not fail: drivers emit them for
        // portability advice that is often wrong for a fixed target.
        console.log(`    ok   ${label}${logs.length ? ' (with diagnostics)' : ''}`);
        for (const l of logs) console.log(`      ${l}`);
      }
    }
  }
  await browser.close();
}

process.exit(failed ? 1 : 0);
