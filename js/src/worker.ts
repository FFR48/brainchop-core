/*
 * The worker entry point: one segmentation, off the calling thread.
 *
 * This is bundled STANDALONE rather than importing ./index.js at run time. A
 * host that bundles the package (Vite, Rollup) rewrites its own copy of
 * index.js and this file would then be unable to find it.
 */

import { BrainchopError } from './error.js'
import { segment, segmentTissues } from './index.js'
import type {
  Backend, ProgressEvent, SegmentOptions, TissueOptions, TissueMaps,
} from './types.js'

/** Everything the main thread sends. `options` is already structured-clonable. */
export interface WorkerRequest {
  input: Uint8Array
  tissueMode?: boolean
  options: Omit<SegmentOptions, 'device' | 'glContext' | 'onLog' | 'onProgress' | 'worker'>
}

export type WorkerResponse =
  | { type: 'log'; line: string }
  | { type: 'progress'; event: ProgressEvent }
  | { type: 'done'; image: ArrayBuffer; mask?: ArrayBuffer; tissues?: TissueMaps; elapsedMs: number
      backend: Backend }
  | { type: 'error'; code: string; message: string; log?: string[] }

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const post = (message: WorkerResponse, transfer: Transferable[] = []) =>
    (self as unknown as Worker).postMessage(message, transfer)
  try {
    const options = {
      ...event.data.options,
      onLog: (line: string) => post({ type: 'log', line }),
      onProgress: (progressEvent: ProgressEvent) => post({ type: 'progress', event: progressEvent }),
    }
    const tissue = event.data.tissueMode ? await segmentTissues(event.data.input, options as TissueOptions) : null
    const result = tissue ? { ...tissue, image: tissue.tissues.gm, mask: undefined } : await segment(event.data.input, options)
    // The buffers are transferred, not copied: this side is done with them, and
    // a 16 MB conformed volume is worth not duplicating. The INPUT is never
    // transferred in the other direction -- see the note in index.ts.
    const transfer: Transferable[] = [result.image]
    if (result.mask) transfer.push(result.mask)
    if (tissue) transfer.push(tissue.tissues.wm, tissue.tissues.csf, tissue.tissues.brain)
    post({
      type: 'done', image: result.image, mask: result.mask, tissues: tissue?.tissues,
      elapsedMs: result.elapsedMs, backend: result.backend,
    }, transfer)
  } catch (error) {
    // BrainchopError does not survive structured clone, so its shape is sent
    // explicitly and rebuilt on the other side. Losing the `code` would turn
    // every refusal into a generic failure.
    const e = error as BrainchopError
    post({
      type: 'error',
      code: (e && e.code) || 'inference-failed',
      message: (e && e.message) || String(error),
      log: e && e.log,
    })
  }
}
