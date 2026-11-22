export declare function computeQuantiles(
  data: ArrayLike<number>,
  lowerQuantile?: number,
  upperQuantile?: number,
): { qmin: number; qmax: number }

export declare function computeLinearQuantiles(
  data: Uint8Array | Uint8ClampedArray,
  lowerQuantile?: number,
  upperQuantile?: number,
): { qmin: number; qmax: number }

export declare function computeMinMax(data: ArrayLike<number>): { min: number; max: number }

export declare function linearScaleVolume(
  data: ArrayLike<number>,
  offset: number,
  range: number,
): Float32Array

export declare function transposeVolume3D(
  data: ArrayLike<number>,
  shape?: number[],
  perm?: number[],
): Float32Array

export declare function transposeVolume3DToU8(
  data: ArrayLike<number>,
  shape?: number[],
  perm?: number[],
): Uint8Array

export declare function validateVolumeStats(
  data: ArrayLike<number>,
  isProbability?: boolean,
  description?: string,
): { sum: number; min: number; max: number }
