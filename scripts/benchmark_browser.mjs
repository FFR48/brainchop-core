#!/usr/bin/env node
// Matched normalized-input inference benchmark, separate from full-pipeline
// measurements. Both runners use one browser build/device and alternating order.
import {readFileSync, writeFileSync, existsSync, mkdirSync} from 'node:fs'
import {resolve, join, basename} from 'node:path'
import {createServer} from 'node:http'
import {createRequire} from 'node:module'
import {createHash} from 'node:crypto'
import {execFileSync} from 'node:child_process'
const [treeArg, wasmArg, dumpArg, outputArg, roundsArg='3', repeatsArg='3'] = process.argv.slice(2)
if (!outputArg) throw Error('usage: benchmark_browser.mjs BROWSER_TREE WASM_DIRECTORY PVE_DUMP OUTPUT_DIRECTORY [ROUNDS REPEATS]')
const tree=resolve(treeArg), wasm=resolve(wasmArg), output=resolve(outputArg)
const candidate=process.env.BC_CANDIDATE_WASM ? resolve(process.env.BC_CANDIDATE_WASM) : null
const rounds=Number(roundsArg), repeats=Number(repeatsArg)
if (!Number.isInteger(rounds)||rounds<1||!Number.isInteger(repeats)||repeats<1) throw Error('invalid repeat counts')
mkdirSync(output,{recursive:true})
const require=createRequire(join(tree,'package.json'))
const {chromium}=require('playwright')
const n=256**3, dump=readFileSync(resolve(dumpArg))
if(dump.length!==n*20 && dump.length!==n*4) throw Error('expected a PVE dump or normalized f32 input')
const input=dump.length===n*4 ? dump : dump.subarray(n*16), hash=x=>createHash('sha256').update(x).digest('hex')
const weights=join(tree,'public/models/model24chan18cls_gdice_prio/model.safetensors')
const runner=join(tree,'webgpu_runners/model24chan18cls_gdice_prio_runner.js')
const files=new Map([
  ['/runner.js',runner],['/weights',weights],['/module.js',join(wasm,'mindmap-benchmark.js')],
  ['/mindmap-benchmark.wasm',join(wasm,'mindmap-benchmark.wasm')],
])
let currentResult
const server=createServer((req,res)=>{
  if(req.method==='POST'&&req.url==='/result') {
    const chunks=[];req.on('data',c=>chunks.push(c));req.on('end',()=>{
      currentResult=Buffer.concat(chunks);res.writeHead(204);res.end()
    });return
  }
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp')
  if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>MindMap benchmark</title>');return}
  if(req.url==='/input'){res.end(input);return}
  const file=files.get(req.url)
  if(!file||!existsSync(file)){res.writeHead(404);res.end();return}
  res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.wasm')?'application/wasm':'application/octet-stream')
  res.end(readFileSync(file))
})
await new Promise(r=>server.listen(0,'127.0.0.1',r))
const url=`http://127.0.0.1:${server.address().port}`
const report={schema:1,scope:'normalized input to raw labels; viewer, cleanup and native resampling excluded',
  coldDefinition:'fresh browser process; driver disk caches are NOT cleared',
  timestampQueries:'when supported: sum of compute-pass timestamp intervals; excludes upload/readback',memoryDefinition:'tracked GPUBuffer allocations, excludes browser/driver/wasm overhead',
  inputSha256:hash(input),weightsSha256:hash(readFileSync(weights)),runnerSha256:hash(readFileSync(runner)),
  wasmSha256:hash(readFileSync(join(wasm,'mindmap-benchmark.wasm'))),
  brainchopC:process.env.BC_C_REVISION || execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
  browserReference:process.env.BC_BROWSER_REVISION || execFileSync('git',['-C',tree,'rev-parse','HEAD'],{encoding:'utf8'}).trim(),rows:[]}
try {
  for(let round=0;round<rounds;round++) for(const variant of (candidate
    ? (round%2?['candidate','upstream','brainchopC']:['brainchopC','upstream','candidate'])
    : (round%2?['upstream','brainchopC']:['brainchopC','upstream']))) {
    const selected=variant==='candidate'?candidate:wasm
    files.set('/module.js',join(selected,'mindmap-benchmark.js'))
    files.set('/mindmap-benchmark.wasm',join(selected,'mindmap-benchmark.wasm'))
    currentResult=null
    const browser=await chromium.launch({...(process.env.BC_CHROME_EXECUTABLE ? {executablePath:process.env.BC_CHROME_EXECUTABLE} : {channel:process.env.BC_CHROME_CHANNEL||'chrome'}),headless:process.env.BC_WGSL_HEADLESS==='1',args:['--enable-unsafe-webgpu',...JSON.parse(process.env.BC_CHROME_ARGS||'[]')]})
    try {
      const page=await browser.newPage();page.on('pageerror',e=>console.error(e.message));await page.goto(url)
      const result=await page.evaluate(async({variant,repeats,n})=>{
        if(!navigator.gpu) throw Error('WebGPU unavailable')
        const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'})
        if(!adapter||adapter.isFallbackAdapter) throw Error('hardware adapter unavailable')
        const info=adapter.info
        if(/swiftshader|llvmpipe|software/i.test(`${info.vendor} ${info.architecture} ${info.device} ${info.description}`)) throw Error('software GPU is not a hardware test')
        const timestampSupported=adapter.features.has('timestamp-query')
        const device=await adapter.requestDevice({requiredFeatures:timestampSupported?['shader-f16','timestamp-query']:['shader-f16'],requiredLimits:{
          maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,
          maxComputeInvocationsPerWorkgroup:adapter.limits.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX:adapter.limits.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupStorageSize:adapter.limits.maxComputeWorkgroupStorageSize}})
        let queryCursor=0
        const queryMarkers=[0],querySet=timestampSupported?device.createQuerySet({type:'timestamp',count:4096}):null
        if(querySet) {
          const createEncoder=device.createCommandEncoder.bind(device)
          device.createCommandEncoder=(...args)=>{
            const encoder=createEncoder(...args),begin=encoder.beginComputePass.bind(encoder)
            encoder.beginComputePass=(descriptor={})=>{
              if(queryCursor+2>4096)throw Error('timestamp capacity exceeded')
              if(descriptor.timestampWrites)throw Error('runner already owns compute timestamps')
              const first=queryCursor;queryCursor+=2
              return begin({...descriptor,timestampWrites:{querySet,beginningOfPassWriteIndex:first,endOfPassWriteIndex:first+1}})
            }
            return encoder
          }
        }
        let live=0,peak=0,total=0,largest=0
        const buffers=[],errors=[]
        device.addEventListener('uncapturederror',e=>errors.push(e.error.message))
        device.lost.then(x=>{if(x.reason!=='destroyed') errors.push(`device lost: ${x.message}`)})
        const original=device.createBuffer.bind(device)
        device.createBuffer=desc=>{
          const b=original(desc); buffers.push(b); live+=Number(desc.size);total+=Number(desc.size)
          peak=Math.max(peak,live);largest=Math.max(largest,Number(desc.size))
          const destroy=b.destroy.bind(b);let destroyed=false
          b.destroy=()=>{if(!destroyed){live-=b.size;destroyed=true}destroy()};return b
        }
        const normalized=new Float32Array(await(await fetch('/input')).arrayBuffer())
        const log=[],runs=[];let setupMs,labels
        device.pushErrorScope('validation');device.pushErrorScope('out-of-memory')
        if(variant==='upstream') {
          // The upstream entry has enableTranspose=true. Convert only storage
          // order; every normalized scalar is otherwise identical.
          const transposed=new Float32Array(n)
          for(let z=0;z<256;z++)for(let y=0;y<256;y++)for(let x=0;x<256;x++)
            transposed[z+y*256+x*65536]=normalized[x+y*256+z*65536]
          const weights=new Uint8Array(await(await fetch('/weights')).arrayBuffer())
          const runner=(await import('/runner.js')).default
          const start=performance.now(),execute=await runner.setupNet(device,weights)
          await device.queue.onSubmittedWorkDone();setupMs=performance.now()-start
          queryMarkers[0]=queryCursor
          for(let i=0;i<repeats;i++) {
            const t=performance.now(),out=await execute(transposed)
            await device.queue.onSubmittedWorkDone();runs.push(performance.now()-t);queryMarkers.push(queryCursor)
            labels=new Float32Array(n)
            for(let z=0;z<256;z++)for(let y=0;y<256;y++)for(let x=0;x<256;x++)
              labels[x+y*256+z*65536]=out[0][z+y*256+x*65536]
          }
        } else {
          const factory=(await import('/module.js')).default
          let finish;const done=new Promise(r=>finish=r),start=performance.now()
          const module=await factory({noInitialRun:true,preinitializedWebGPUDevice:device,
            print:l=>{log.push(l);if(/^BENCH /.test(l))queryMarkers.push(queryCursor)},printErr:l=>log.push(l),onExit:code=>finish(code),onAbort:why=>finish(String(why))})
          setupMs=performance.now()-start
          module.FS.writeFile('/input.f32',new Uint8Array(normalized.buffer))
          try{module.callMain([String(repeats)])}catch(e){if(typeof e.status!=='number')throw e}
          const status=await Promise.race([done,new Promise((_,reject)=>setTimeout(()=>reject(Error('wasm timeout')),300000))])
          if(status!==0)throw Error(`wasm status ${status}: ${log.join('\n')}`)
          for(const l of log){const m=/^BENCH \d+ ([\d.]+)/.exec(l);if(m)runs.push(Number(m[1]))}
          const bytes=module.FS.readFile('/labels.f32');labels=new Float32Array(bytes.buffer,bytes.byteOffset,n)
        }
        const computeMs=[]
        if(querySet&&queryCursor) {
          const resolved=device.createBuffer({size:queryCursor*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC})
          const read=device.createBuffer({size:queryCursor*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})
          const e=device.createCommandEncoder();e.resolveQuerySet(querySet,0,queryCursor,resolved,0)
          e.copyBufferToBuffer(resolved,0,read,0,queryCursor*8);device.queue.submit([e.finish()])
          await read.mapAsync(GPUMapMode.READ)
          const times=new BigUint64Array(read.getMappedRange())
          for(let r=1;r<queryMarkers.length;r++) {
            let nanos=0n;for(let q=queryMarkers[r-1];q<queryMarkers[r];q+=2)nanos+=times[q+1]-times[q]
            computeMs.push(Number(nanos)/1e6)
          }
          read.unmap();read.destroy();resolved.destroy();querySet.destroy()
        }
        const oom=await device.popErrorScope(),validation=await device.popErrorScope()
        if(oom||validation||errors.length)throw Error(JSON.stringify({oom:oom?.message,validation:validation?.message,errors}))
        let foreground=0
        for(const v of labels){if(!Number.isInteger(v)||v<0||v>=18)throw Error('invalid label');foreground+=v!==0}
        if(!foreground||foreground>n*.95)throw Error('unusable categorical result')
        await fetch('/result',{method:'POST',body:new Uint8Array(labels.buffer,labels.byteOffset,labels.byteLength)})
        for(const b of buffers)b.destroy();device.destroy()
        return {adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description},
          setupMs,runs,computeMs,timestampSupported,peakAllocatedBytes:peak,totalAllocatedBytes:total,largestBufferBytes:largest,foreground,log}
      },{variant,repeats,n})
      if(!currentResult||currentResult.length!==n*4)throw Error('missing raw output')
      const name=`${round}-${variant}.f32`;writeFileSync(join(output,name),currentResult)
      report.rows.push({round,variant,wasmSha256:variant==='upstream'?undefined:hash(readFileSync(join(selected,'mindmap-benchmark.wasm'))),browser:browser.version(),...result,output:name,outputSha256:hash(currentResult)})
      writeFileSync(join(output,'benchmark.json'),JSON.stringify(report,null,2))
      console.log(JSON.stringify({round,variant,setupMs:result.setupMs,runs:result.runs,peakAllocatedMiB:result.peakAllocatedBytes/2**20}))
    } finally {await browser.close()}
  }
} finally {server.close()}
