// Pure-JS zero-dependency NIfTI-1 (.nii) reader and writer.
// Supports all standard scalar datatypes, big- and little-endian inputs,
// sform/qform affine extraction & quaternion round-trip, scl_slope/scl_inter,
// NIFTI_INTENT_LABEL (1002) tagging, and full 352-byte header preservation.

export const NIFTI_HEADER_BYTES = 348
export const NIFTI_VOX_OFFSET = 352
export const NIFTI_INTENT_NONE = 0
export const NIFTI_INTENT_LABEL = 1002

export const NIFTI_DT = {
  UINT8: 2,
  INT16: 4,
  INT32: 8,
  FLOAT32: 16,
  FLOAT64: 64,
  INT8: 256,
  UINT16: 512,
  UINT32: 768,
  INT64: 1024,
  UINT64: 1280,
} as const

export type NiftiDatatypeCode = (typeof NIFTI_DT)[keyof typeof NIFTI_DT]

export type NiftiTypedArray =
  | Uint8Array
  | Int8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array
  | BigInt64Array
  | BigUint64Array

export interface NiftiHeader {
  dims: number[]
  pixDims: number[]
  datatypeCode: number
  numBitsPerVoxel: number
  voxOffset: number
  sclSlope: number
  sclInter: number
  calMin: number
  calMax: number
  intentCode: number
  qformCode: number
  sformCode: number
  quaternB: number
  quaternC: number
  quaternD: number
  qoffsetX: number
  qoffsetY: number
  qoffsetZ: number
  /** 16-element row-major qform matrix (storage voxel -> mm). */
  qtoXyz: number[]
  /** 16-element row-major sform matrix (storage voxel -> mm). */
  stoXyz: number[]
  /** Preferred affine for display/conform (qform when qformCode > sformCode and valid, else sform). */
  affine: number[]
  littleEndian: boolean
  /** Exact 352-byte copy of the original NIfTI-1 header (normalized to little-endian on write). */
  rawHeader: Uint8Array
}

export interface NiftiVolume {
  header: NiftiHeader
  data: NiftiTypedArray
}

export interface NiftiWritePatch {
  dims?: [number, number, number]
  pixDims?: [number, number, number]
  datatypeCode?: number
  sclSlope?: number
  sclInter?: number
  calMin?: number
  calMax?: number
  intentCode?: number
  sformCode?: number
  qformCode?: number
  /** 16-element row-major affine written to both sform and qform. */
  affine?: number[]
  /** Override stored pixdim[1..3] after quaternion update (matches conform.c reslice). */
  preservePixDims?: [number, number, number]
  descrip?: string
}

export function bytesPerVoxelForDatatype(datatypeCode: number): number {
  switch (datatypeCode) {
    case NIFTI_DT.UINT8:
    case NIFTI_DT.INT8:
      return 1
    case NIFTI_DT.INT16:
    case NIFTI_DT.UINT16:
      return 2
    case NIFTI_DT.INT32:
    case NIFTI_DT.UINT32:
    case NIFTI_DT.FLOAT32:
      return 4
    case NIFTI_DT.FLOAT64:
    case NIFTI_DT.INT64:
    case NIFTI_DT.UINT64:
      return 8
    default:
      return 0
  }
}

export function isValidAffine4x4(m: ArrayLike<number>): boolean {
  if (!m || m.length < 16) return false
  for (let i = 0; i < 16; i++) {
    if (!Number.isFinite(m[i])) return false
  }
  const det =
    m[0] * (m[5] * m[10] - m[6] * m[9]) -
    m[1] * (m[4] * m[10] - m[6] * m[8]) +
    m[2] * (m[4] * m[9] - m[5] * m[8])
  return Number.isFinite(det) && Math.abs(det) >= 1e-6
}

/**
 * Compute 4x4 row-major qto_xyz matrix from NIfTI-1 quaternion parameters.
 * Matches `nifti_quatern_to_dmat44()` in `src/nifti_io.c`.
 */
export function quaternToMat44(
  qb: number,
  qc: number,
  qd: number,
  qx: number,
  qy: number,
  qz: number,
  dx: number,
  dy: number,
  dz: number,
  qfac: number,
): number[] {
  let b = qb
  let c = qc
  let d = qd
  let a = 1.0 - (b * b + c * c + d * d)
  if (a < 1e-7) {
    const norm = Math.sqrt(b * b + c * c + d * d)
    if (norm > 0) {
      const inv = 1.0 / norm
      b *= inv
      c *= inv
      d *= inv
    }
    a = 0.0
  } else {
    a = Math.sqrt(a)
  }

  const xd = dx > 0 ? dx : 1.0
  const yd = dy > 0 ? dy : 1.0
  let zd = dz > 0 ? dz : 1.0
  if (qfac < 0) zd = -zd

  return [
    (a * a + b * b - c * c - d * d) * xd,
    2.0 * (b * c - a * d) * yd,
    2.0 * (b * d + a * c) * zd,
    qx,
    2.0 * (b * c + a * d) * xd,
    (a * a + c * c - b * b - d * d) * yd,
    2.0 * (c * d - a * b) * zd,
    qy,
    2.0 * (b * d - a * c) * xd,
    2.0 * (c * d + a * b) * yd,
    (a * a + d * d - c * c - b * b) * zd,
    qz,
    0,
    0,
    0,
    1,
  ]
}

function mat33Norm1(m: number[]): number {
  let best = 0
  for (let c = 0; c < 3; c++) {
    const s = Math.abs(m[c]) + Math.abs(m[3 + c]) + Math.abs(m[6 + c])
    if (s > best) best = s
  }
  return best
}

function mat33NormInf(m: number[]): number {
  let best = 0
  for (let r = 0; r < 3; r++) {
    const o = r * 3
    const s = Math.abs(m[o]) + Math.abs(m[o + 1]) + Math.abs(m[o + 2])
    if (s > best) best = s
  }
  return best
}

function mat33Inv(m: number[]): number[] | null {
  const c00 = m[4] * m[8] - m[5] * m[7]
  const c01 = -(m[3] * m[8] - m[5] * m[6])
  const c02 = m[3] * m[7] - m[4] * m[6]
  const det = m[0] * c00 + m[1] * c01 + m[2] * c02
  if (!Number.isFinite(det) || Math.abs(det) < 1e-18) return null
  const inv = 1 / det
  return [
    c00 * inv,
    -(m[1] * m[8] - m[2] * m[7]) * inv,
    (m[1] * m[5] - m[2] * m[4]) * inv,
    c01 * inv,
    (m[0] * m[8] - m[2] * m[6]) * inv,
    -(m[0] * m[5] - m[2] * m[3]) * inv,
    c02 * inv,
    -(m[0] * m[7] - m[1] * m[6]) * inv,
    (m[0] * m[4] - m[1] * m[3]) * inv,
  ]
}

function mat33Polar(A: number[]): number[] {
  let X = A.slice()
  let gam = mat33Norm1(X)
  if (gam === 0) return X
  let gami = mat33NormInf(X)
  for (let iter = 0; iter < 64; iter++) {
    const Y = mat33Inv(X)
    if (!Y) break
    const alp = mat33Norm1(Y)
    const alpi = mat33NormInf(Y)
    const gamma = Math.sqrt(Math.sqrt((alp * alpi) / (gam * gami)))
    const g1 = 0.5 * gamma
    const g2 = 0.5 / gamma
    let dif = 0
    const Z = new Array<number>(9)
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const idx = r * 3 + c
        // Y^T[r,c] = Y[c,r]
        const v = g1 * X[idx] + g2 * Y[c * 3 + r]
        dif += Math.abs(v - X[idx])
        Z[idx] = v
      }
    }
    X = Z
    if (dif < 1e-12) break
    gam = mat33Norm1(X)
    gami = mat33NormInf(X)
  }
  return X
}

/**
 * Decompose a 4x4 row-major affine matrix into NIfTI-1 quaternion fields.
 * Matches `nifti_dmat44_to_quatern()` in `src/nifti_io.c`.
 */
export function mat44ToQuatern(m: ArrayLike<number>): {
  quaternB: number
  quaternC: number
  quaternD: number
  qoffsetX: number
  qoffsetY: number
  qoffsetZ: number
  dx: number
  dy: number
  dz: number
  qfac: number
} {
  const qoffsetX = m[3]
  const qoffsetY = m[7]
  const qoffsetZ = m[11]

  let r11 = m[0], r12 = m[1], r13 = m[2]
  let r21 = m[4], r22 = m[5], r23 = m[6]
  let r31 = m[8], r32 = m[9], r33 = m[10]

  let xd = Math.sqrt(r11 * r11 + r21 * r21 + r31 * r31)
  let yd = Math.sqrt(r12 * r12 + r22 * r22 + r32 * r32)
  let zd = Math.sqrt(r13 * r13 + r23 * r23 + r33 * r33)

  if (xd === 0) { r11 = 1; r21 = 0; r31 = 0; xd = 1 }
  if (yd === 0) { r22 = 1; r12 = 0; r32 = 0; yd = 1 }
  if (zd === 0) { r33 = 1; r13 = 0; r23 = 0; zd = 1 }

  const dx = xd
  const dy = yd
  const dz = zd

  r11 /= xd; r21 /= xd; r31 /= xd
  r12 /= yd; r22 /= yd; r32 /= yd
  r13 /= zd; r23 /= zd; r33 /= zd

  const P = mat33Polar([r11, r12, r13, r21, r22, r23, r31, r32, r33])
  r11 = P[0]; r12 = P[1]; r13 = P[2]
  r21 = P[3]; r22 = P[4]; r23 = P[5]
  r31 = P[6]; r32 = P[7]; r33 = P[8]

  const det =
    r11 * r22 * r33 - r11 * r32 * r23 -
    r21 * r12 * r33 + r21 * r32 * r13 +
    r31 * r12 * r23 - r31 * r22 * r13
  let qfac = 1
  if (det <= 0) {
    qfac = -1
    r13 = -r13
    r23 = -r23
    r33 = -r33
  }

  let a = r11 + r22 + r33 + 1.0
  let b: number, c: number, d: number
  if (a > 0.5) {
    a = 0.5 * Math.sqrt(a)
    b = 0.25 * (r32 - r23) / a
    c = 0.25 * (r13 - r31) / a
    d = 0.25 * (r21 - r12) / a
  } else {
    const sx = 1.0 + r11 - (r22 + r33)
    const sy = 1.0 + r22 - (r11 + r33)
    const sz = 1.0 + r33 - (r11 + r22)
    if (sx > 1.0) {
      b = 0.5 * Math.sqrt(sx)
      c = 0.25 * (r12 + r21) / b
      d = 0.25 * (r13 + r31) / b
      a = 0.25 * (r32 - r23) / b
    } else if (sy > 1.0) {
      c = 0.5 * Math.sqrt(sy)
      b = 0.25 * (r12 + r21) / c
      d = 0.25 * (r23 + r32) / c
      a = 0.25 * (r13 - r31) / c
    } else {
      d = 0.5 * Math.sqrt(sz)
      b = 0.25 * (r13 + r31) / d
      c = 0.25 * (r23 + r32) / d
      a = 0.25 * (r21 - r12) / d
    }
    if (a < 0) {
      b = -b
      c = -c
      d = -d
    }
  }

  return {
    quaternB: b,
    quaternC: c,
    quaternD: d,
    qoffsetX,
    qoffsetY,
    qoffsetZ,
    dx,
    dy,
    dz,
    qfac,
  }
}

function readTypedVoxels(
  bytes: Uint8Array,
  voxOffset: number,
  nvox: number,
  datatypeCode: number,
  littleEndian: boolean,
): NiftiTypedArray {
  const bpp = bytesPerVoxelForDatatype(datatypeCode)
  if (!bpp) {
    throw new Error(`Unsupported NIfTI-1 datatype code: ${datatypeCode}`)
  }
  const byteLen = nvox * bpp
  if (voxOffset + byteLen > bytes.byteLength) {
    throw new Error(
      `NIfTI-1 data truncated: need ${voxOffset + byteLen} bytes, got ${bytes.byteLength}`,
    )
  }

  // Fast path on little-endian hosts (all modern JS engines) when input is little-endian.
  if (littleEndian) {
    const aligned = ((bytes.byteOffset + voxOffset) & (bpp - 1)) === 0
    const sliceBuf = aligned
      ? bytes.buffer
      : bytes.slice(voxOffset, voxOffset + byteLen).buffer
    const off = aligned ? bytes.byteOffset + voxOffset : 0
    switch (datatypeCode) {
      case NIFTI_DT.UINT8: return new Uint8Array(sliceBuf, off, nvox)
      case NIFTI_DT.INT8: return new Int8Array(sliceBuf, off, nvox)
      case NIFTI_DT.INT16: return new Int16Array(sliceBuf, off, nvox)
      case NIFTI_DT.UINT16: return new Uint16Array(sliceBuf, off, nvox)
      case NIFTI_DT.INT32: return new Int32Array(sliceBuf, off, nvox)
      case NIFTI_DT.UINT32: return new Uint32Array(sliceBuf, off, nvox)
      case NIFTI_DT.FLOAT32: return new Float32Array(sliceBuf, off, nvox)
      case NIFTI_DT.FLOAT64: return new Float64Array(sliceBuf, off, nvox)
      case NIFTI_DT.INT64: return new BigInt64Array(sliceBuf, off, nvox)
      case NIFTI_DT.UINT64: return new BigUint64Array(sliceBuf, off, nvox)
    }
  }

  // Big-endian fallback via DataView.
  const dv = new DataView(bytes.buffer, bytes.byteOffset + voxOffset, byteLen)
  switch (datatypeCode) {
    case NIFTI_DT.UINT8:
      return bytes.subarray(voxOffset, voxOffset + nvox)
    case NIFTI_DT.INT8:
      return new Int8Array(bytes.slice(voxOffset, voxOffset + nvox).buffer)
    case NIFTI_DT.INT16: {
      const out = new Int16Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getInt16(i * 2, false)
      return out
    }
    case NIFTI_DT.UINT16: {
      const out = new Uint16Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getUint16(i * 2, false)
      return out
    }
    case NIFTI_DT.INT32: {
      const out = new Int32Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getInt32(i * 4, false)
      return out
    }
    case NIFTI_DT.UINT32: {
      const out = new Uint32Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getUint32(i * 4, false)
      return out
    }
    case NIFTI_DT.FLOAT32: {
      const out = new Float32Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getFloat32(i * 4, false)
      return out
    }
    case NIFTI_DT.FLOAT64: {
      const out = new Float64Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getFloat64(i * 8, false)
      return out
    }
    case NIFTI_DT.INT64: {
      const out = new BigInt64Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getBigInt64(i * 8, false)
      return out
    }
    case NIFTI_DT.UINT64: {
      const out = new BigUint64Array(nvox)
      for (let i = 0; i < nvox; i++) out[i] = dv.getBigUint64(i * 8, false)
      return out
    }
    default:
      throw new Error(`Unsupported NIfTI-1 datatype code: ${datatypeCode}`)
  }
}

/**
 * Parse an uncompressed NIfTI-1 (`.nii`) buffer.
 */
export function readNifti1(input: ArrayBuffer | Uint8Array): NiftiVolume {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input)
  if (bytes.byteLength < NIFTI_VOX_OFFSET) {
    throw new Error(`Invalid NIfTI-1 input: expected at least ${NIFTI_VOX_OFFSET} bytes, got ${bytes.byteLength}`)
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  const leSize = dv.getInt32(0, true)
  const beSize = dv.getInt32(0, false)
  let littleEndian: boolean
  if (leSize === NIFTI_HEADER_BYTES) {
    littleEndian = true
  } else if (beSize === NIFTI_HEADER_BYTES) {
    littleEndian = false
  } else {
    throw new Error(`Invalid NIfTI-1 header: sizeof_hdr is ${leSize} (expected ${NIFTI_HEADER_BYTES})`)
  }

  const magic = String.fromCharCode(bytes[344], bytes[345], bytes[346], bytes[347])
  if (magic !== 'n+1\0') {
    throw new Error(`Unsupported NIfTI-1 magic '${magic.replace(/\0/g, '\\0')}'; expected single-file 'n+1\\0'`)
  }

  const dims = new Array<number>(8)
  for (let i = 0; i < 8; i++) dims[i] = dv.getInt16(40 + i * 2, littleEndian)

  const nx = dims[1]
  const ny = dims[2]
  const nz = dims[3]
  const nt = dims[0] >= 4 && dims[4] > 0 ? dims[4] : 1
  if (nx < 1 || ny < 1 || nz < 1 || nt !== 1) {
    throw new Error(`NIfTI-1 input must be a scalar 3D image; got dims=[${dims.slice(0, 5).join(', ')}]`)
  }

  const intentCode = dv.getInt16(68, littleEndian)
  const datatypeCode = dv.getInt16(70, littleEndian)
  const numBitsPerVoxel = dv.getInt16(72, littleEndian)

  const pixDims = new Array<number>(8)
  for (let i = 0; i < 8; i++) pixDims[i] = dv.getFloat32(76 + i * 4, littleEndian)

  const rawVoxOffset = Math.floor(dv.getFloat32(108, littleEndian))
  const voxOffset = rawVoxOffset >= NIFTI_VOX_OFFSET ? rawVoxOffset : NIFTI_VOX_OFFSET
  const sclSlope = dv.getFloat32(112, littleEndian)
  const sclInter = dv.getFloat32(116, littleEndian)
  const calMax = dv.getFloat32(124, littleEndian)
  const calMin = dv.getFloat32(128, littleEndian)
  const qformCode = dv.getInt16(252, littleEndian)
  const sformCode = dv.getInt16(254, littleEndian)

  const quaternB = dv.getFloat32(256, littleEndian)
  const quaternC = dv.getFloat32(260, littleEndian)
  const quaternD = dv.getFloat32(264, littleEndian)
  const qoffsetX = dv.getFloat32(268, littleEndian)
  const qoffsetY = dv.getFloat32(272, littleEndian)
  const qoffsetZ = dv.getFloat32(276, littleEndian)

  const srowX = [
    dv.getFloat32(280, littleEndian),
    dv.getFloat32(284, littleEndian),
    dv.getFloat32(288, littleEndian),
    dv.getFloat32(292, littleEndian),
  ]
  const srowY = [
    dv.getFloat32(296, littleEndian),
    dv.getFloat32(300, littleEndian),
    dv.getFloat32(304, littleEndian),
    dv.getFloat32(308, littleEndian),
  ]
  const srowZ = [
    dv.getFloat32(312, littleEndian),
    dv.getFloat32(316, littleEndian),
    dv.getFloat32(320, littleEndian),
    dv.getFloat32(324, littleEndian),
  ]

  const dx = Math.abs(pixDims[1]) || 1
  const dy = Math.abs(pixDims[2]) || 1
  const dz = Math.abs(pixDims[3]) || 1
  const fallbackDiag = [
    dx, 0, 0, 0,
    0, dy, 0, 0,
    0, 0, dz, 0,
    0, 0, 0, 1,
  ]

  const qfac = pixDims[0] < 0 ? -1 : 1
  const qtoXyz = qformCode > 0
    ? quaternToMat44(quaternB, quaternC, quaternD, qoffsetX, qoffsetY, qoffsetZ, dx, dy, dz, qfac)
    : fallbackDiag.slice()

  const stoXyz = sformCode > 0
    ? [...srowX, ...srowY, ...srowZ, 0, 0, 0, 1]
    : qtoXyz.slice()

  // Match src/conform.c:364-366: qform when qform_code > sform_code and valid, else sform.
  const primary = qformCode > sformCode ? qtoXyz : stoXyz
  const secondary = qformCode > sformCode ? stoXyz : qtoXyz
  const affine = isValidAffine4x4(primary) ? primary.slice() : secondary.slice()

  const nvox = nx * ny * nz
  const data = readTypedVoxels(bytes, voxOffset, nvox, datatypeCode, littleEndian)

  // Preserve original 352-byte header (bytes 348..351 cleared to 0 extension flag on write).
  const rawHeader = new Uint8Array(NIFTI_VOX_OFFSET)
  rawHeader.set(bytes.subarray(0, NIFTI_HEADER_BYTES), 0)

  const header: NiftiHeader = {
    dims,
    pixDims,
    datatypeCode,
    numBitsPerVoxel,
    voxOffset,
    sclSlope,
    sclInter,
    calMin,
    calMax,
    intentCode,
    qformCode,
    sformCode,
    quaternB,
    quaternC,
    quaternD,
    qoffsetX,
    qoffsetY,
    qoffsetZ,
    qtoXyz,
    stoXyz,
    affine,
    littleEndian,
    rawHeader,
  }

  return { header, data }
}

export function isIntegerNiftiDatatype(datatypeCode: number): boolean {
  switch (datatypeCode) {
    case NIFTI_DT.UINT8:
    case NIFTI_DT.INT8:
    case NIFTI_DT.UINT16:
    case NIFTI_DT.INT16:
    case NIFTI_DT.UINT32:
    case NIFTI_DT.INT32:
    case NIFTI_DT.INT64:
    case NIFTI_DT.UINT64:
      return true
    default:
      return false
  }
}

/**
 * Decode scaled voxels to finite Float32Array (matching brainchopc_convert + sanitize_input).
 */
export function decodeNiftiVoxelsToFloat32(
  data: NiftiTypedArray,
  sclSlope: number,
  sclInter: number,
): Float32Array {
  const n = data.length
  const out = new Float32Array(n)
  const hasScaling = Number.isFinite(sclSlope) && sclSlope !== 0
  const slope = hasScaling ? sclSlope : 1
  const inter = hasScaling && Number.isFinite(sclInter) ? sclInter : 0
  const isBigInt = data instanceof BigInt64Array || data instanceof BigUint64Array
  let finiteCount = 0

  for (let i = 0; i < n; i++) {
    const raw = isBigInt ? Number(data[i]) : (data[i] as number)
    const v = Math.fround(raw * slope + inter)
    if (Number.isFinite(v)) {
      out[i] = v
      finiteCount++
    } else {
      out[i] = 0
    }
  }
  if (finiteCount === 0) {
    throw new Error('Input volume has no finite voxels')
  }
  return out
}

/**
 * Apply a binary MindGrab brain mask to a native NIfTI volume (matching
 * mask_integer_input / float fallback in src/brainchopc.c).
 *
 * Integer inputs keep their native stored values and scaling without passing
 * through float32 (preventing clipping, fractional-slope quantization, and
 * >24-bit integer precision loss). Non-brain voxels receive the stored value
 * corresponding to the minimum decoded intensity (maximum stored value when
 * sclSlope < 0). Input arrays are never mutated in place.
 */
export function maskMindGrabVolume(
  data: NiftiTypedArray,
  datatypeCode: number,
  sclSlope: number,
  sclInter: number,
  mask: Uint8Array,
): { data: NiftiTypedArray; sclSlope: number; sclInter: number } {
  if (isIntegerNiftiDatatype(datatypeCode)) {
    const hasScaling = Number.isFinite(sclSlope) && sclSlope !== 0
    const descending = hasScaling && sclSlope < 0
    const outSlope = hasScaling ? sclSlope : 1
    const outInter = hasScaling && Number.isFinite(sclInter) ? sclInter : 0

    if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
      const out = data instanceof BigInt64Array ? new BigInt64Array(data) : new BigUint64Array(data)
      if (out.length > 0) {
        let bg = out[0]
        if (descending) {
          for (let i = 1; i < out.length; i++) if (out[i] > bg) bg = out[i]
        } else {
          for (let i = 1; i < out.length; i++) if (out[i] < bg) bg = out[i]
        }
        for (let i = 0; i < out.length; i++) {
          if (mask[i] === 0) out[i] = bg
        }
      }
      return { data: out, sclSlope: outSlope, sclInter: outInter }
    }

    const out = data.slice()
    if (out.length > 0) {
      let bg = out[0]
      if (descending) {
        for (let i = 1; i < out.length; i++) if (out[i] > bg) bg = out[i]
      } else {
        for (let i = 1; i < out.length; i++) if (out[i] < bg) bg = out[i]
      }
      for (let i = 0; i < out.length; i++) {
        if (mask[i] === 0) out[i] = bg
      }
    }
    return { data: out, sclSlope: outSlope, sclInter: outInter }
  }

  const origFloat = decodeNiftiVoxelsToFloat32(data, sclSlope, sclInter)
  let mn = Infinity
  for (let i = 0; i < origFloat.length; i++) {
    if (origFloat[i] < mn) mn = origFloat[i]
  }
  for (let i = 0; i < origFloat.length; i++) {
    if (mask[i] === 0) origFloat[i] = mn
  }
  return {
    data: encodeFloat32ToDatatype(origFloat, datatypeCode),
    sclSlope: 1,
    sclInter: 0,
  }
}

function clampRound(x: number, lo: number, hi: number): number {
  if (!(x >= lo)) return lo
  if (!(x <= hi)) return hi
  return Math.round(x)
}

/**
 * Convert Float32Array to target NIfTI datatype (matching brainchopc_convert in src/brainchopc.c).
 */
export function encodeFloat32ToDatatype(
  values: Float32Array,
  datatypeCode: number,
): NiftiTypedArray {
  const n = values.length
  switch (datatypeCode) {
    case NIFTI_DT.UINT8: {
      const out = new Uint8Array(n)
      for (let i = 0; i < n; i++) out[i] = clampRound(values[i], 0, 255)
      return out
    }
    case NIFTI_DT.INT8: {
      const out = new Int8Array(n)
      for (let i = 0; i < n; i++) out[i] = clampRound(values[i], -128, 127)
      return out
    }
    case NIFTI_DT.INT16: {
      const out = new Int16Array(n)
      for (let i = 0; i < n; i++) out[i] = clampRound(values[i], -32768, 32767)
      return out
    }
    case NIFTI_DT.UINT16: {
      const out = new Uint16Array(n)
      for (let i = 0; i < n; i++) out[i] = clampRound(values[i], 0, 65535)
      return out
    }
    case NIFTI_DT.INT32: {
      const out = new Int32Array(n)
      for (let i = 0; i < n; i++) out[i] = clampRound(values[i], -2147483648, 2147483647)
      return out
    }
    case NIFTI_DT.UINT32: {
      const out = new Uint32Array(n)
      for (let i = 0; i < n; i++) out[i] = clampRound(values[i], 0, 4294967295)
      return out
    }
    case NIFTI_DT.FLOAT32:
      return values
    case NIFTI_DT.FLOAT64: {
      const out = new Float64Array(n)
      for (let i = 0; i < n; i++) out[i] = values[i]
      return out
    }
    case NIFTI_DT.INT64: {
      const out = new BigInt64Array(n)
      for (let i = 0; i < n; i++) {
        const v = values[i]
        out[i] = !(v >= -9223372036854775808)
          ? -9223372036854775808n
          : !(v < 9223372036854775808)
            ? 9223372036854775807n
            : BigInt(Math.round(v))
      }
      return out
    }
    case NIFTI_DT.UINT64: {
      const out = new BigUint64Array(n)
      for (let i = 0; i < n; i++) {
        const v = values[i]
        out[i] = !(v > 0)
          ? 0n
          : !(v < 18446744073709551616)
            ? 18446744073709551615n
            : BigInt(Math.round(v))
      }
      return out
    }
    default:
      throw new Error(`Unsupported output NIfTI datatype code: ${datatypeCode}`)
  }
}

/**
 * Write a single-file little-endian NIfTI-1 (`.nii`) Uint8Array, preserving the
 * 352-byte base header (or initializing a fresh one) and applying `patch`.
 */
export function writeNifti1(
  baseHeader: NiftiHeader | null,
  data: NiftiTypedArray,
  patch: NiftiWritePatch = {},
): Uint8Array {
  const datatypeCode = patch.datatypeCode ?? baseHeader?.datatypeCode ?? NIFTI_DT.UINT8
  const bpp = bytesPerVoxelForDatatype(datatypeCode)
  if (!bpp) throw new Error(`Unsupported NIfTI-1 datatype code: ${datatypeCode}`)

  const dataBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  const out = new Uint8Array(NIFTI_VOX_OFFSET + dataBytes.byteLength)

  if (baseHeader && baseHeader.littleEndian && baseHeader.rawHeader.byteLength >= NIFTI_HEADER_BYTES) {
    out.set(baseHeader.rawHeader.subarray(0, NIFTI_HEADER_BYTES), 0)
  }

  const dv = new DataView(out.buffer)
  dv.setInt32(0, NIFTI_HEADER_BYTES, true)

  const dims = patch.dims
    ? [3, patch.dims[0], patch.dims[1], patch.dims[2], 1, 1, 1, 1]
    : baseHeader
      ? [3, baseHeader.dims[1], baseHeader.dims[2], baseHeader.dims[3], 1, 1, 1, 1]
      : [3, 1, 1, 1, 1, 1, 1, 1]
  for (let i = 0; i < 8; i++) dv.setInt16(40 + i * 2, dims[i], true)

  const intentCode = patch.intentCode ?? baseHeader?.intentCode ?? NIFTI_INTENT_NONE
  dv.setFloat32(56, 0, true)
  dv.setFloat32(60, 0, true)
  dv.setFloat32(64, 0, true)
  dv.setInt16(68, intentCode, true)
  dv.setInt16(70, datatypeCode, true)
  dv.setInt16(72, bpp * 8, true)

  const pixDims = baseHeader ? baseHeader.pixDims.slice() : [1, 1, 1, 1, 1, 0, 0, 0]
  if (patch.pixDims) {
    pixDims[1] = patch.pixDims[0]
    pixDims[2] = patch.pixDims[1]
    pixDims[3] = patch.pixDims[2]
  }
  for (let i = 0; i < 8; i++) dv.setFloat32(76 + i * 4, pixDims[i] ?? 0, true)

  dv.setFloat32(108, NIFTI_VOX_OFFSET, true)
  dv.setFloat32(112, patch.sclSlope ?? baseHeader?.sclSlope ?? 1, true)
  dv.setFloat32(116, patch.sclInter ?? baseHeader?.sclInter ?? 0, true)
  dv.setFloat32(124, patch.calMax ?? baseHeader?.calMax ?? 0, true)
  dv.setFloat32(128, patch.calMin ?? baseHeader?.calMin ?? 0, true)

  // Clear intent_name (bytes 328..343) and write descrip (bytes 148..227).
  out.fill(0, 328, 344)
  if (patch.descrip !== undefined) {
    out.fill(0, 148, 228)
    for (let i = 0; i < Math.min(80, patch.descrip.length); i++) {
      out[148 + i] = patch.descrip.charCodeAt(i) & 0x7f
    }
  }

  // Write spatial transforms (qform + sform).
  const writeAffine = patch.affine ?? (baseHeader && !baseHeader.littleEndian ? baseHeader.stoXyz : null)
  if (writeAffine) {
    const code = patch.sformCode ?? patch.qformCode ??
      Math.max(baseHeader?.qformCode ?? 0, baseHeader?.sformCode ?? 0, 1)
    dv.setInt16(252, patch.qformCode ?? code, true)
    dv.setInt16(254, patch.sformCode ?? code, true)

    const q = mat44ToQuatern(writeAffine)
    dv.setFloat32(76, q.qfac, true)
    dv.setFloat32(80, patch.preservePixDims ? patch.preservePixDims[0] : q.dx, true)
    dv.setFloat32(84, patch.preservePixDims ? patch.preservePixDims[1] : q.dy, true)
    dv.setFloat32(88, patch.preservePixDims ? patch.preservePixDims[2] : q.dz, true)

    dv.setFloat32(256, q.quaternB, true)
    dv.setFloat32(260, q.quaternC, true)
    dv.setFloat32(264, q.quaternD, true)
    dv.setFloat32(268, q.qoffsetX, true)
    dv.setFloat32(272, q.qoffsetY, true)
    dv.setFloat32(276, q.qoffsetZ, true)

    for (let i = 0; i < 4; i++) dv.setFloat32(280 + i * 4, writeAffine[i], true)
    for (let i = 0; i < 4; i++) dv.setFloat32(296 + i * 4, writeAffine[4 + i], true)
    for (let i = 0; i < 4; i++) dv.setFloat32(312 + i * 4, writeAffine[8 + i], true)
  } else if (baseHeader && !baseHeader.littleEndian) {
    dv.setInt16(252, baseHeader.qformCode, true)
    dv.setInt16(254, baseHeader.sformCode, true)
    dv.setFloat32(256, baseHeader.quaternB, true)
    dv.setFloat32(260, baseHeader.quaternC, true)
    dv.setFloat32(264, baseHeader.quaternD, true)
    dv.setFloat32(268, baseHeader.qoffsetX, true)
    dv.setFloat32(272, baseHeader.qoffsetY, true)
    dv.setFloat32(276, baseHeader.qoffsetZ, true)
    for (let i = 0; i < 4; i++) dv.setFloat32(280 + i * 4, baseHeader.stoXyz[i], true)
    for (let i = 0; i < 4; i++) dv.setFloat32(296 + i * 4, baseHeader.stoXyz[4 + i], true)
    for (let i = 0; i < 4; i++) dv.setFloat32(312 + i * 4, baseHeader.stoXyz[8 + i], true)
  }

  // NIfTI-1 magic "n+1\0" + 4 zero extension bytes.
  out[344] = 0x6e // 'n'
  out[345] = 0x2b // '+'
  out[346] = 0x31 // '1'
  out[347] = 0x00
  out[348] = 0
  out[349] = 0
  out[350] = 0
  out[351] = 0

  out.set(dataBytes, NIFTI_VOX_OFFSET)
  return out
}
