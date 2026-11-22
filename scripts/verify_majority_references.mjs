#!/usr/bin/env node
// Verify native golden candidates against the actual upstream export function.
// Inputs are frozen baseline CPU conformed results on t1_2mm, not new inference.
import assert from 'node:assert/strict'
import {readFileSync,writeFileSync} from 'node:fs'
import {resolve,join} from 'node:path'
import {createRequire} from 'node:module'
import {gunzipSync} from 'node:zlib'
import {createHash} from 'node:crypto'
const [treeArg,targetArg,...pairs]=process.argv.slice(2)
const tree=resolve(treeArg),require=createRequire(join(tree,'package.json'))
const {mat4}=require('gl-matrix')
const main=readFileSync(join(tree,'main.js'),'utf8'),start=main.indexOf('function resliceLabelsToNative(seg) {')
const end=main.indexOf('return outNV;\n  }',start)
assert(start>=0&&end>start)
const fn=main.slice(start,end+'return outNV;\n  }'.length)
function read(path){let bytes=readFileSync(path);if(path.endsWith('.gz'))bytes=gunzipSync(bytes);return bytes}
function volume(bytes) {
 const dims=Array.from({length:8},(_,i)=>bytes.readInt16LE(40+i*2))
 const affine=Array.from({length:3},(_,r)=>Array.from({length:4},(_,c)=>bytes.readFloatLE(280+r*16+c*4)))
 affine.push([0,0,0,1])
 const type=bytes.readInt16LE(70),count=dims[1]*dims[2]*dims[3],offset=bytes.readFloatLE(108)
 const img=new Float32Array(count)
 for(let i=0;i<count;i++)img[i]=type===2?bytes[offset+i]:type===16?bytes.readFloatLE(offset+4*i):bytes.readInt16LE(offset+2*i)
 return {hdr:{dims,affine,intent_code:1002},img}
}
const native=volume(read(targetArg))
const oracle=new Function('nativeInputNV','mat4','cloneVolume',fn+'\nreturn resliceLabelsToNative;')(native,mat4,x=>({hdr:structuredClone(x.hdr)}))
const hashes={browserFunctionSha256:createHash('sha256').update(fn).digest('hex'),pairs:[]}
for(let k=0;k<pairs.length;k+=2) {
 const source=read(pairs[k]),candidate=read(pairs[k+1]),expected=oracle(volume(source)).img,got=volume(candidate).img
 assert.equal(got.length,expected.length)
 let different=0;for(let i=0;i<got.length;i++)different+=got[i]!==expected[i]
 assert.equal(different,0,pairs[k+1])
 hashes.pairs.push({source:pairs[k].split('/').pop(),candidate:pairs[k+1].split('/').pop(),voxels:got.length,different,
 sourceSha256:createHash('sha256').update(source).digest('hex'),candidateSha256:createHash('sha256').update(candidate).digest('hex')})
}
console.log(JSON.stringify(hashes,null,2))
