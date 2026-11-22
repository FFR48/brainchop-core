export declare function invertAffine4x4(m: number[] | number[][]): number[] | null
export declare function mulAffine4x4(a: number[] | number[][], b: number[] | number[][]): number[]
export declare function resliceVolumeData(params: {
  srcImg: ArrayBufferView
  srcDims: number[]
  srcAffine: number[] | number[][]
  dstDims: number[]
  dstAffine: number[] | number[][]
  isLabel: boolean
  isProbability?: boolean
  dstTypedArrayKind?: string
  labelOutputType?: 'uint8' | 'int16'
  labelLut?: ArrayLike<number> | null
}): {
  img: Uint8Array | Int16Array | Float32Array | ArrayBufferView
  hdrPatch: Record<string, number>
}
