// Measure the actual JavaScript fit against C/wasm, including mandatory wasm
// copies into/out of linear memory. Both receive identical prepared priors.
import{readFileSync,writeFileSync}from'node:fs'
import{resolve,join}from'node:path'
import{createRequire}from'node:module'
import{createServer}from'node:http'
import{createHash}from'node:crypto'
const[treeArg,wasmArg,dumpArg,outArg]=process.argv.slice(2)
if(!outArg)throw Error('usage: benchmark_cat_lite_browser.mjs BROWSER_TREE WASM_DIRECTORY DUMP REPORT')
const tree=resolve(treeArg),wasm=resolve(wasmArg),require=createRequire(join(tree,'package.json'))
const useAdapter=process.env.BC_CAT_ADAPTER==='1'
const{chromium}=require('playwright'),dump=readFileSync(dumpArg)
const files=new Map([['/reference.js',join(tree,'cat-lite.js')],['/module.js',join(wasm,'cat-lite.js')],['/cat-lite.wasm',join(wasm,'cat-lite.wasm')]])
files.set('/cat-lite-wasm.js',join(tree,'cat-lite-wasm.js'))
files.set('/wasm/cat-lite.js',join(wasm,'cat-lite.js'))
files.set('/wasm/cat-lite.wasm',join(wasm,'cat-lite.wasm'))
const server=createServer((req,res)=>{
  if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>CAT-lite fit benchmark</title>');return}
  if(req.url==='/dump'){res.end(dump);return}
  const file=files.get(req.url);if(!file){res.writeHead(404);res.end();return}
  res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':'application/wasm');res.end(readFileSync(file))
})
await new Promise(r=>server.listen(0,'127.0.0.1',r))
const browser=await chromium.launch({...(process.env.BC_CHROME_EXECUTABLE ? {executablePath:process.env.BC_CHROME_EXECUTABLE} : {channel:process.env.BC_CHROME_CHANNEL||'chrome'}),headless:process.env.BC_WGSL_HEADLESS==='1'})
try{
  const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}`)
  const report=await page.evaluate(async(useAdapter)=>{
    const{applyCatLitePartialVolume:js}=await import('/reference.js'),factory=(await import('/module.js')).default
    const data=new Float32Array(await(await fetch('/dump')).arrayBuffer()),n=256**3
    const options={catLitePurePriorPower:3,catLitePriorStrength:.8,catLiteIntensityStrength:.75,catLiteMixelPrior:.35,
      catLiteCsfWmMixelPrior:6,catLiteSpatialWeight:.25,catLiteMinSupport:.03,catLiteSigmaFloor:.025,catLiteBiasBlockSize:16,catLiteBiasSmoothPasses:3}
    const start=performance.now(),module=useAdapter?null:await factory(),setupMs=performance.now()-start
    const adapter=useAdapter?(await import('/cat-lite-wasm.js')).tryCatLiteWasm:null
    const rows=[]
    for(let round=0;round<6;round++){
      const outputs={}
      for(const variant of round%2?['wasm','javascript']:['javascript','wasm']){
        const priors=[0,1,2].map(t=>data.subarray(t*n,(t+1)*n).slice()),intensity=data.subarray(4*n,5*n)
        const before=performance.now()
        let tissues=priors
        if(variant==='javascript')tissues=js(priors,intensity,[256,256,256],options).tissues
        else if(adapter){
          const result=await adapter(priors,intensity,[256,256,256],options)
          if(!result||!result.stats.applied)throw Error('wasm adapter must fit successfully')
          tissues=result.tissues
        }else{
          const planes=module._malloc(n*16),image=module._malloc(n*4),shape=module._malloc(12)
          if(!planes||!image||!shape)throw Error('wasm allocation failed')
          module.HEAPU32.set([256,256,256],shape/4)
          for(let t=0;t<3;t++)module.HEAPF32.set(priors[t],planes/4+t*n)
          module.HEAPF32.set(intensity,image/4)
          const result=module._bc_cat_lite_fit(planes,image,shape)
          if(result)throw Error('wasm fit failed')
          for(let t=0;t<3;t++)priors[t].set(module.HEAPF32.subarray(planes/4+t*n,planes/4+(t+1)*n))
          module._free(shape);module._free(image);module._free(planes)
        }
        rows.push({round,variant,ms:performance.now()-before});outputs[variant]=tissues
      }
      let worst=0,over=0
      for(let t=0;t<3;t++)for(let i=0;i<n;i++){
        const a=outputs.wasm[t][i],b=outputs.javascript[t][i]
        if(!Number.isFinite(a)||!Number.isFinite(b))throw Error('nonfinite result')
        const d=Math.abs(a-b);worst=Math.max(worst,d);over+=d>2**-24
      }
      rows[rows.length-1].comparison={max:worst,overGate:over,gate:2**-24}
      if(over)throw Error(`unchanged C/JS precision gate failed: ${worst}`)
    }
    return{setupMs,useAdapter,rows,scope:'prepared JS prior/intensity arrays to updated JS tissue arrays; wasm upload/output copies included, initial reference-input cloning excluded for both; adapter mode includes lazy module loading in its first fit'}
  },useAdapter)
  report.browser=browser.version();report.inputSha256=createHash('sha256').update(dump).digest('hex')
  report.wasmSha256=createHash('sha256').update(readFileSync(join(wasm,'cat-lite.wasm'))).digest('hex')
  writeFileSync(outArg,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2))
}finally{await browser.close();server.close()}
