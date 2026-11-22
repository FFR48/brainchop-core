// Direct oracle: upstream 18-class & 104-class hierarchy with neighbor refill.
// node label_cleanup_check.mjs C_DRIVER BRAINCHOP_TEST [--104] [RAW_F32 [CLEAN_NIFTI]]
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve, join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { execFileSync } from 'node:child_process'
import { gunzipSync } from 'node:zlib'

const rawArgs = process.argv.slice(2)
const driver = rawArgs[0]
const tree = rawArgs[1]
let rest = rawArgs.slice(2)
let mode104 = false
if (rest[0] === '--104') {
  mode104 = true
  rest = rest.slice(1)
}
const [dump, nifti] = rest

const { BWLabeler } = await import(pathToFileURL(resolve(tree, 'bwlabels.js')))
const work = mkdtempSync(join(tmpdir(), 'bc-cleanup-'))
const targets18 = new Set([1, 2, 5, 6, 13])
const dk104FamilyOf = (cls) => (cls >= 1 && cls <= 34 ? 1 : cls >= 35 && cls <= 68 ? 2 : 0)
const log = console.log
console.log = () => {} // keep hundreds of synthetic cases quiet
let cases = 0

function check18(input, dim, name) {
  const bw = new BWLabeler()
  const data = new Uint32Array(input)
  const [count, components] = bw.bwlabel(data, dim, 6, false, false)
  const [, expected] = bw.filter18ClassFromLabels(data, count, components, dim, targets18, true, null, false)
  const floats = new Float32Array(input)
  writeFileSync(join(work, 'in'), Buffer.from(floats.buffer))
  execFileSync(resolve(driver), [join(work, 'in'), join(work, 'out'), ...dim.map(String)])
  const raw = readFileSync(join(work, 'out'))
  const actual = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4)
  let changed = 0
  for (let i = 0; i < actual.length; i++) {
    assert.equal(actual[i], expected[i], `18cls ${name}: voxel ${i}`)
    changed += actual[i] !== input[i]
  }
  cases++
  return { actual: actual.slice(), changed }
}

function check104(input, dim, name) {
  const bw = new BWLabeler()
  const data = new Uint32Array(input)
  const [count, components] = bw.bwlabel(data, dim, 6, false, false)
  const [, expected] = bw.filterHierarchicalFromLabels(data, count, components, dim, 'all', true, true, dk104FamilyOf, false)
  const floats = new Float32Array(input)
  writeFileSync(join(work, 'in'), Buffer.from(floats.buffer))
  execFileSync(resolve(driver), ['--104', join(work, 'in'), join(work, 'out'), ...dim.map(String)])
  const raw = readFileSync(join(work, 'out'))
  const actual = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4)
  let changed = 0
  for (let i = 0; i < actual.length; i++) {
    assert.equal(actual[i], expected[i], `104cls ${name}: voxel ${i}`)
    changed += actual[i] !== input[i]
  }
  cases++
  return { actual: actual.slice(), changed }
}

try {
  if (dump) {
    const raw = readFileSync(dump)
    assert.equal(raw.length, 256 ** 3 * 4)
    const vals = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4)
    const result = mode104 ? check104(vals, [256, 256, 256], dump) : check18(vals, [256, 256, 256], dump)
    if (nifti) {
      let bytes = readFileSync(nifti)
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) bytes = gunzipSync(bytes)
      assert.equal(bytes.readInt16LE(70), 2, 'uint8 labels')
      assert.deepEqual([42, 44, 46].map(x => bytes.readInt16LE(x)), [256, 256, 256])
      const offset = bytes.readFloatLE(108)
      for (let i = 0; i < result.actual.length; i++)
        assert.equal(bytes[offset + i], result.actual[i], `CLI integration voxel ${i}`)
    }
    log(JSON.stringify({ mode: mode104 ? 104 : 18, cases, changed_voxels: result.changed, browser_mismatches: 0, cli_checked: !!nifti }))
  } else {
    const dim = [7, 7, 3], n = 147
    check18(new Uint32Array(n), dim, 'empty')
    check18(new Uint32Array(n).fill(1), dim, 'single class')
    const tied = new Uint32Array(n); tied[0] = 1; tied[n - 1] = 2
    const t = check18(tied, dim, 'binary tie keeps later component').actual
    assert.equal(t[0], 0); assert.equal(t[n - 1], 2)
    // A one-voxel WM island in retained class 3 touches only class 3.
    const speck = new Uint32Array(n).fill(3)
    speck[0] = speck[1] = 1; speck[73] = 1
    assert.equal(check18(speck, dim, 'white matter speckle').actual[73], 3)

    // 104-class edge cases: empty, single class, binary tie, resection cavity island
    check104(new Uint32Array(n), dim, 'empty')
    check104(new Uint32Array(n).fill(1), dim, 'single class')
    const tied104 = new Uint32Array(n); tied104[0] = 1; tied104[n - 1] = 35
    const t104 = check104(tied104, dim, 'binary tie keeps later component').actual
    assert.equal(t104[0], 0); assert.equal(t104[n - 1], 35)
    // Resection cavity fragment (unique class 77 disconnected from main brain mass)
    const cavity = new Uint32Array(n)
    for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) cavity[y * 7 + x] = (x < 2 ? 85 : 3)
    cavity[6 + 6 * 7 + 2 * 49] = 77
    const cavRes = check104(cavity, dim, 'resection cavity fragment').actual
    assert.equal(cavRes[6 + 6 * 7 + 2 * 49], 0)

    // Deterministic random volumes exercise all target classes, face/corner
    // connectivity, ties, boundaries, family gating, and touching suppressed components.
    let seed = 19092026
    for (let trial = 0; trial < 240; trial++) {
      const shape = [2 + trial % 9, 2 + trial % 7, 1 + trial % 5]
      const a = new Uint32Array(shape.reduce((x, y) => x * y, 1))
      for (let i = 0; i < a.length; i++) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
        a[i] = trial % 2 ? seed % 18 : [0, 1, 2, 3, 5, 6, 13][seed % 7]
      }
      check18(a, shape, `random ${trial}`)
    }
    const palette104 = [0, 0, 1, 2, 3, 34, 35, 36, 68, 69, 77, 85, 86, 95, 103]
    for (let trial = 0; trial < 240; trial++) {
      const shape = [2 + trial % 9, 2 + trial % 7, 1 + trial % 5]
      const a = new Uint32Array(shape.reduce((x, y) => x * y, 1))
      for (let i = 0; i < a.length; i++) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
        a[i] = trial % 2 ? (seed % 3 ? seed % 104 : 0) : palette104[seed % palette104.length]
      }
      check104(a, shape, `random104 ${trial}`)
    }
    log(`label cleanup: ${cases} browser-oracle phantoms (18-class & 104-class) PASS`)
  }
} finally { rmSync(work, { recursive: true, force: true }); console.log = log }
