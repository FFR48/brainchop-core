// Both executables use test/cat_lite_check.c's fit-only timer. Compile them
// with identical flags; the baseline links merged main's fitting sources.
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
const [baseArg, candidateArg, inputArg, reportArg, roundsArg = '6'] = process.argv.slice(2);
if (!reportArg) throw Error('usage: benchmark_cat_lite_native.mjs BASELINE CANDIDATE DUMP REPORT [ROUNDS]');
const binaries = { baseline: resolve(baseArg), candidate: resolve(candidateArg) };
const input = resolve(inputArg), rounds = Number(roundsArg);
if (!Number.isInteger(rounds) || rounds < 1) throw Error('invalid round count');
const work = mkdtempSync(join(tmpdir(), 'cat-fit-pair-'));
const hash = file => createHash('sha256').update(readFileSync(file)).digest('hex');
const report = { scope: 'single-thread native fitting; excludes fixture reading and output writing',
  buildFlags: process.env.BC_CAT_BUILD_FLAGS || null, inputSha256: hash(input),
  binarySha256: Object.fromEntries(Object.entries(binaries).map(([v,p])=>[v,hash(p)])), rows: [] };
try {
  for (let round = 0; round < rounds; round++) for (const variant of round % 2 ? ['candidate','baseline'] : ['baseline','candidate']) {
    const run = spawnSync(binaries[variant], [input,join(work,variant)], { encoding:'utf8' });
    if (run.status) throw Error(run.stderr || `exit ${run.status}`);
    const match = run.stderr.match(/CAT_FIT_MS ([\d.]+)/);
    if (!match) throw Error('missing fit timer');
    report.rows.push({ round, variant, fitMs:Number(match[1]) });
  }
  const bytes = Object.keys(binaries).map(v=>readFileSync(join(work,v)));
  if (bytes[0].length !== bytes[1].length) throw Error('output lengths differ');
  const [a,b] = bytes.map(v=>new Float32Array(v.buffer,v.byteOffset,v.length/4));
  let max = 0, over = 0, different = 0;
  for (let i=0;i<a.length;i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) throw Error('nonfinite output');
    const delta=Math.abs(a[i]-b[i]);max=Math.max(max,delta);over+=delta>2**-24;different+=delta!==0;
  }
  report.comparison={max,over,different,gate:2**-24};
  writeFileSync(reportArg,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  if(over)process.exitCode=1;
} finally { rmSync(work,{recursive:true,force:true}); }
