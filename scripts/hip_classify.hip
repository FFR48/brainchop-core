/* Maintainer-compiled label head. The hidden producer is contiguous NCDHW,
 * not the voxel-major layout used by CUDA/WebGPU. Classifier upload is [K,C].
 * One thread owns a voxel; only C features and one best pair live in registers.
 * No class-volume, intermediate scores, scratch buffer or host computation. */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

extern "C" __global__ void bc_hip_classify(
    const half *__restrict__ features, const half *__restrict__ weights,
#if BC_HIP_CLASSIFIER_BIAS
    const half *__restrict__ bias,
#endif
    unsigned char *__restrict__ labels) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= BC_HIP_NVOX) return;
    float x[BC_HIP_CHANNELS];
#pragma unroll
    for (int c = 0; c < BC_HIP_CHANNELS; ++c)
        x[c] = __half2float(features[(size_t)c * BC_HIP_NVOX + i]);
    float best = -INFINITY;
    unsigned char winner = 0;
#pragma unroll 1
    for (int k = 0; k < BC_HIP_CLASSES; ++k) {
#if BC_HIP_CLASSIFIER_BIAS
        float score = __half2float(bias[k]);
#else
        float score = 0.0f;
#endif
#pragma unroll
        for (int c = 0; c < BC_HIP_CHANNELS; ++c)
            score = fmaf(x[c], __half2float(weights[k * BC_HIP_CHANNELS + c]), score);
        if (score > best) { best = score; winner = (unsigned char)k; }
    }
    labels[i] = winner;
}
