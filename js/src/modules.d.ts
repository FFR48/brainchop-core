import type { ModelName } from './types.js'

export type WasmBackend = 'webgl2' | 'cpu'

export declare const MODEL_BINS: Record<ModelName, () => string>
export declare const MODULES: Record<WasmBackend, Record<ModelName, () => Promise<{ default: unknown }>>>
