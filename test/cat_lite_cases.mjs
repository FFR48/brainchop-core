// Actual browser CAT-lite is the independent oracle; exercise fit refusal,
// support ties, intensity transforms and nonfinite intensity without inference.
import assert from 'node:assert/strict'
import {readFileSync,writeFileSync,mkdtempSync,rmSync} from 'node:fs'
import {resolve,join} from 'node:path'
import {tmpdir} from 'node:os'
import {pathToFileURL} from 'node:url'
import {spawnSync} from 'node:child_process'
const [exeArg,treeArg]=process.argv.slice(2)
if(!exeArg||!treeArg)throw Error('usage: cat_lite_cases.mjs RUNNER BROWSER_TREE')
const exe=resolve(exeArg),tree=resolve(treeArg)
const {applyCatLitePartialVolume}=await import(pathToFileURL(join(tree,'cat-lite.js')))
const options={catLitePurePriorPower:3,catLitePriorStrength:.8,catLiteIntensityStrength:.75,
  catLiteMixelPrior:.35,catLiteCsfWmMixelPrior:6,catLiteSpatialWeight:.25,
  catLiteMinSupport:.03,catLiteSigmaFloor:.025,catLiteBiasBlockSize:16,catLiteBiasSmoothPasses:3}
const dir=mkdtempSync(join(tmpdir(),'bc-cat-cases-')),dim=[32,32,32],n=dim.reduce((a,b)=>a*b)
let count=0
function check(name,edit) {
  const data=new Float32Array(n*5),planes=[0,1,2,3,4].map(t=>data.subarray(t*n,(t+1)*n))
  for(let z=0;z<32;z++)for(let y=0;y<32;y++)for(let x=0;x<32;x++) {
    const i=x+y*32+z*1024,t=x<11?2:x<22?0:1
    for(let k=0;k<3;k++)planes[k][i]=k===t?.82:.06
    planes[3][i]=.94
    planes[4][i]=[.5,.8,.2][t]+.03*Math.sin(x*.4+y*.2+z*.1)
    edit?.(planes,i,x,y,z)
  }
  const reference=applyCatLitePartialVolume(planes.slice(0,3).map(x=>x.slice()),planes[4],dim,options)
  const input=join(dir,'input'),output=join(dir,'output')
  writeFileSync(input,Buffer.from(data.buffer))
  const run=spawnSync(exe,[input,output,...dim.map(String)],{encoding:'utf8'})
  assert.equal(run.status,0,`${name}: ${run.stderr}`)
  const bytes=readFileSync(output),actual=new Float32Array(bytes.buffer,bytes.byteOffset,3*n)
  let worst=0
  for(let t=0;t<3;t++)for(let i=0;i<n;i++) {
    assert(Number.isFinite(actual[t*n+i])&&Number.isFinite(reference.tissues[t][i]),`${name}: nonfinite result`)
    worst=Math.max(worst,Math.abs(actual[t*n+i]-reference.tissues[t][i]))
  }
  assert(worst<=2**-24,`${name}: max ${worst} exceeds unchanged gate ${2**-24}`)
  console.log(`${name}: applied=${reference.stats.applied}, max=${worst}`);count++
}
try {
  check('three tissues')
  check('contrast scaled',(p,i)=>{p[4][i]=p[4][i]*.7+.1})
  check('inverted contrast refuses fit',(p,i)=>{p[4][i]=1-p[4][i]})
  check('flat intensity refuses fit',(p,i)=>{p[4][i]=.5})
  check('empty support',(p,i)=>{p[0][i]=p[1][i]=p[2][i]=0})
  check('support threshold',(p,i,x)=>{if(x%3===0)for(let t=0;t<3;t++)p[t][i]=.009999})
  check('nonfinite intensity',(p,i)=>{if(i%101===0)p[4][i]=NaN})
  check('equal-size components keep last',(p,i,x,y)=>{if(y===15||y===16)p[0][i]=p[1][i]=p[2][i]=0})
  check('small disconnected islands',(p,i,x,y,z)=>{if(x===15||y===15||z===15)p[0][i]=p[1][i]=p[2][i]=0})
  console.log(`CAT-lite: ${count} synthetic browser-oracle cases PASS`)
} finally {rmSync(dir,{recursive:true,force:true})}
