export interface V4ExecutionPlan {
  layout: 'std' | 'LowMem128'
  execMode: 'single' | 'pipelined' | 'paced' | 'calibrate'
  chunkableStd: boolean
  reason: string
  tier: 'fast' | 'moderate' | 'very-slow' | null
  probeMs: number | null
  estFullLayerMs: number | null
  msPerStrip: number | null
  fromCache: boolean
  safeMode: boolean
  fingerprint: string
  storageKey: string
}

export declare function formatV4PlanDiagnostic(plan: V4ExecutionPlan, d?: unknown): string
export declare function executeWithWebGpuSafeModeRetry<T>(params: {
  device: GPUDevice
  modelEntry: Record<string, unknown>
  runAttempt: (device: GPUDevice, entry: Record<string, unknown>, meta: { isRetry: boolean }) => Promise<T>
  requestFreshDevice: ((info: { failedDevice: GPUDevice; failureKind: string; error: unknown }) => Promise<GPUDevice | null>) | null
  onRetry?: (info: { failureKind: string; error: unknown; retryEntry: Record<string, unknown>; freshDevice: GPUDevice }) => void
  env?: Record<string, unknown> | null
}): Promise<T>
