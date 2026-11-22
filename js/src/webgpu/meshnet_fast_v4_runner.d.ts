import type { V4ExecutionPlan } from './gpu-compatibility.js'

export interface V4ExecuteFn {
  (
    volume: Uint8Array | Float32Array,
    ioOpts?: {
      gpuNativeInput?: boolean
      gpuNativeOutput?: boolean
      normOffset?: number
      normRange?: number
      normClamp?: boolean
      onProgress?: (frac: number, stage: string) => void
    },
  ): Promise<Array<Uint8Array | Float32Array>>
  dispose: (opts?: { keepSharedWorkspace?: boolean }) => void
  gpuBytes: () => number
  __v4Plan: () => V4ExecutionPlan
  kernels: string
}

export declare const supportsF32Fallback: boolean
export declare const supportsGpuNativeIO: boolean
export declare function releaseV4Workspace(device: GPUDevice | null | undefined): void
export declare function warmupV4CorePipelines(device: GPUDevice | null | undefined): Promise<void>
export declare function setupNet(
  device: GPUDevice,
  weightBytes: Uint8Array | ArrayBuffer,
  callbackUI?: (message: string, progress: number) => void,
  modelEntry?: Record<string, unknown>,
): Promise<V4ExecuteFn>
