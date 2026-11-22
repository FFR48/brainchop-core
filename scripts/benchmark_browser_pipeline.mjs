// Production pipelines on identical already-conformed NIfTI input. Unlike the
// normalized-input kernel benchmark, includes gzip decode, normalization,
// orientation, model loading, inference, guards and default speckle cleanup.
// Endpoint: uint8 labels in conformed voxel order, ready for display/export.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, join, basename } from 'node:path';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
const [treeArg, inputArg, referenceArg, outputArg, roundsArg = '3', repeatsArg = '2'] = process.argv.slice(2);
if (!outputArg) throw Error('usage: benchmark_browser_pipeline.mjs BROWSER_TREE INPUT_NII_GZ CPU_DEFAULT_NII_GZ OUTPUT_DIR [ROUNDS REPEATS]');
const tree = resolve(treeArg), output = resolve(outputArg), root = new URL('..', import.meta.url).pathname;
const rounds = Number(roundsArg), repeats = Number(repeatsArg);
if (!Number.isInteger(rounds) || rounds < 1 || !Number.isInteger(repeats) || repeats < 1) throw Error('invalid repeat counts');
const require = createRequire(join(tree, 'package.json'));
const { createServer } = await import(require.resolve('vite'));
const { chromium } = require('playwright');
const input = readFileSync(inputArg), reference = gunzipSync(readFileSync(referenceArg));
const offset = reference.readFloatLE(108), expected = reference.subarray(offset);
if (expected.length !== 256 ** 3 || reference.readInt16LE(70) !== 2) throw Error('expected conformed uint8 reference');
mkdirSync(output, { recursive: true });
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const pageModule = `
export async function run(variant, repeats) {
  const compressed = await (await fetch('/__input')).arrayBuffer();
  const rows = [], errors = [];
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.isFallbackAdapter) throw Error('hardware WebGPU required');
  const info = adapter.info;
  if (/swiftshader|llvmpipe|software/i.test(info.vendor+' '+info.architecture+' '+info.device+' '+info.description)) throw Error('software GPU');
  for (let iteration = 0; iteration < repeats; iteration++) {
    let labels, device, stats;
    const start = performance.now();
    try {
      if (variant === 'brainchopC') {
        const moduleUrl = location.origin + '/bc/index.js';
        const { segment } = await import(/* @vite-ignore */ moduleUrl);
        const result = await segment(compressed, { model: 'mindmap', backend: 'webgpu',
          saveConform: true, gzipOutput: false, assetPath: location.origin + '/bc/' });
        const header = new DataView(result.image), offset = header.getFloat32(108, true);
        if (header.getInt16(70, true) !== 2) throw Error('non-uint8 C result');
        labels = new Uint8Array(result.image, offset);
        stats = { moduleMs: result.elapsedMs };
      } else {
        const tf = await import('@tensorflow/tfjs');
        const { runInferenceWebGpu } = await import('/inference-webgpu.js');
        const { inferenceModelsList } = await import('/brainchop-parameters.js');
        await tf.setBackend('webgl'); await tf.ready();
        const plain = await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
        const header = new DataView(plain), offset = header.getFloat32(108, true);
        if (header.getInt16(70, true) !== 2 || [42,44,46].some(p => header.getInt16(p,true) !== 256)) throw Error('expected conformed uint8 scan');
        const runAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!runAdapter || runAdapter.isFallbackAdapter) throw Error('hardware adapter unavailable');
        device = await runAdapter.requestDevice({ requiredFeatures: ['shader-f16'], requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } });
        device.pushErrorScope('validation');
        const model = inferenceModelsList.find(m => m.webgpu_runner === 'model24chan18cls_gdice_prio');
        // Sampling is part of the upstream production normalizer. Pin its seed,
        // retaining the actual algorithm rather than replacing it with C's.
        const random = Math.random; let seed = 123;
        Math.random = () => ((seed = (Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
        try {
          await runInferenceWebGpu(device, { isPostProcessEnable: true, fillSuppressedWithNeighborLabel: true },
            model, {}, new Uint8Array(plain, offset, 256**3),
            image => { labels = Uint8Array.from(image); },
            (message, progress, error, data) => { if (error) errors.push(error); if (data) stats = data; });
        } finally { Math.random = random; }
        const validation = await device.popErrorScope();
        if (validation) throw Error(validation.message);
      }
      if (!labels || labels.length !== 256**3 || errors.length) throw Error('pipeline failed: '+errors.join('; '));
      rows.push({ iteration, totalMs: performance.now()-start, stats });
      await fetch('/__result/'+iteration, { method: 'POST', body: labels });
    } finally { device?.destroy(); }
  }
  return { rows, hardware: { vendor:info.vendor, architecture:info.architecture, device:info.device, description:info.description } };
}`;
const received = new Map();
const server = await createServer({ configFile: false, root: tree, cacheDir: join(output, '.vite'),
  server: { host: '127.0.0.1', port: 0, hmr: false },
  plugins: [{ name: 'matched-pipeline',
    resolveId(id) { if (id === '/__pipeline.js') return join(tree, '__pipeline.js'); },
    load(id) { if (id === join(tree, '__pipeline.js')) return pageModule; },
    configureServer(server) { server.middlewares.use((req, res, next) => {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      if (req.url === '/__bench') { res.setHeader('Content-Type','text/html'); res.end('<!doctype html><title>Matched pipeline benchmark</title>'); return; }
      if (req.url === '/__input') { res.end(input); return; }
      if (req.url.startsWith('/__result/') && req.method === 'POST') {
        const chunks = []; req.on('data', c => chunks.push(c)); req.on('end', () => {
          received.set(Number(req.url.split('/').at(-1)), Buffer.concat(chunks)); res.writeHead(204).end();
        }); return;
      }
      if (req.url.startsWith('/bc/')) {
        const name = basename(req.url.split('?')[0]);
        if (!/^(index|brainchop-mindmap-gpu)\.(js|wasm)$/.test(name)) { res.writeHead(404).end(); return; }
        res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : 'application/wasm');
        res.end(readFileSync(join(root, 'js/dist', name))); return;
      }
      next();
    }); },
  }] });
await server.listen();
const report = {
  scope: 'gzip conformed NIfTI bytes to uint8 conformed labels, including normalization/orientation, loading, inference, guards and default cleanup; viewer and native export excluded',
  coldDefinition: 'fresh browser process per round/variant; driver disk caches NOT cleared; warm is another production API call',
  inputSha256: sha(input), referenceSha256: sha(reference),
  cleanup: 'default targeted18 plus neighbor refill in both; browser fillSuppressedWithNeighborLabel=true',
  normalization: 'production algorithms retained; upstream random sampling seed 123 per call',
  sourceSha256: Object.fromEntries(['inference-webgpu.js','tensor-utils.js','bwlabels.js','brainchop-parameters.js',
    'webgpu_runners/model24chan18cls_gdice_prio_runner.js','public/models/model24chan18cls_gdice_prio/model.safetensors']
    .map(file => [file, sha(readFileSync(join(tree,file)))])),
  artifacts: Object.fromEntries(['index.js','brainchop-mindmap-gpu.js','brainchop-mindmap-gpu.wasm']
    .map(file => [file,sha(readFileSync(join(root,'js/dist',file)))])), rows: [],
};
try {
  for (let round=0; round<rounds; round++) for (const variant of round%2 ? ['upstream','brainchopC'] : ['brainchopC','upstream']) {
    received.clear();
    const browser = await chromium.launch({...(process.env.BC_CHROME_EXECUTABLE ? {executablePath:process.env.BC_CHROME_EXECUTABLE} : {channel:process.env.BC_CHROME_CHANNEL||'chrome'}),headless:process.env.BC_WGSL_HEADLESS==='1',args:['--enable-unsafe-webgpu',...JSON.parse(process.env.BC_CHROME_ARGS||'[]')]});
    try {
      const page = await browser.newPage();
      page.on('console', m => { if(m.type()==='error') console.error(m.text()); });
      await page.goto(server.resolvedUrls.local[0]+'__bench');
      const result = await page.evaluate(async ({variant,repeats}) => (await import('/__pipeline.js')).run(variant,repeats), {variant,repeats});
      for (const row of result.rows) {
        const labels = received.get(row.iteration);
        if (!labels || labels.length !== expected.length) throw Error('missing output');
        let differ = 0; for(let i=0;i<labels.length;i++) differ += labels[i] !== expected[i];
        const file = round+'-'+variant+'-'+row.iteration+'.u8'; writeFileSync(join(output,file),labels);
        report.rows.push({round,variant,browser:browser.version(),hardware:result.hardware,...row,
          output:file,outputSha256:sha(labels),differFromCpu:differ,budget:1024,pass:differ<=1024});
        console.log(JSON.stringify(report.rows.at(-1)));
      }
      writeFileSync(join(output,'pipeline.json'),JSON.stringify(report,null,2)+'\n');
    } finally { await browser.close(); }
  }
  if (report.rows.some(r=>!r.pass)) process.exitCode=1;
} finally { await server.close(); }
