export declare function applyCatLitePartialVolume(
  volumes: Float32Array[],
  intensity: Float32Array,
  shape: number[],
  options?: Record<string, unknown>,
): {
  tissues: Float32Array[]
  stats: {
    applied: boolean
    reason?: string
    tissueMeans?: { csf: number; gray: number; white: number }
    tissueSigmas?: { csf: number; gray: number; white: number }
    supportCleanup?: { componentCount: number; removedVoxels: number }
  }
}
