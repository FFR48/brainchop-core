// Stress the fit on real prepared priors: contrast changes alter only its T1
// input; the flip transforms all four priors and T1 together. These are fit
// tests, not independent scans or network/prior-generation coverage.
import assert from 'node:assert/strict';
import { readFileSync,writeFileSync,mkdtempSync,rmSync } from 'node:fs';
import { resolve,join } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';
const [runnerArg,treeArg,dumpArg,reportArg]=process.argv.slice(2);
if(!reportArg)throw Error('usage: cat_lite_volume_cases.mjs C_RUNNER BROWSER_TREE DUMP REPORT');
const {applyCatLitePartialVolume}=await import(pathToFileURL(resolve(treeArg,'cat-lite.js')));
const bytes=readFileSync(dumpArg),n=256**3;
assert.equal(bytes.length,n*20);
const original=new Float32Array(bytes.buffer,bytes.byteOffset,n*5),report=[];
const work=mkdtempSync(join(tmpdir(),'cat-volume-'));
try {
  for(const variant of ['gamma0.7','gamma1.4','flip-x']) {
    const data=original.slice();
    if(variant==='flip-x')for(let plane=0;plane<5;plane++)for(let i=0;i<n;i++)
      data[plane*n+i]=original[plane*n+i-i%256+255-i%256];
    else for(let i=4*n;i<5*n;i++)data[i]=Math.pow(Math.max(0,data[i]),Number(variant.slice(5)));
    const input=join(work,'input.bin'),output=join(work,'output.bin');
    writeFileSync(input,Buffer.from(data.buffer));
    const run=spawnSync(resolve(runnerArg),[input,output],{encoding:'utf8'});
    assert.equal(run.status,0,run.stderr);
    const expected=applyCatLitePartialVolume([0,1,2].map(t=>data.subarray(t*n,(t+1)*n)),data.subarray(4*n),[256,256,256]);
    assert(expected.stats.applied,variant+': expected a valid fit');
    const raw=readFileSync(output),actual=new Float32Array(raw.buffer,raw.byteOffset,raw.length/4);
    assert.equal(actual.length,n*3);
    let max=0,over=0;
    for(let t=0;t<3;t++)for(let i=0;i<n;i++) {
      const a=actual[t*n+i],b=expected.tissues[t][i];assert(Number.isFinite(a)&&Number.isFinite(b));
      const d=Math.abs(a-b);max=Math.max(max,d);over+=d>2**-24;
    }
    report.push({variant,max,over,gate:2**-24});
    writeFileSync(reportArg,JSON.stringify(report,null,2)+'\n');console.log(report.at(-1));
    assert.equal(over,0,variant+': unchanged numerical gate');
  }
}finally{rmSync(work,{recursive:true,force:true});}
