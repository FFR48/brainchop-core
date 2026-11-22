#include "backend.h"
#include "backend_accel.h"
#include "cuda_variants.h"
#include "cuda_kernels.cuh"
#include "model_specs.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cuda_error(cudaError_t error, const char *operation) {
	if (error == cudaSuccess) return 0;
	fprintf(stderr, "brainchopc: CUDA %s failed: %s\n",
		operation, cudaGetErrorString(error));
	/*
	 * This one is worth naming, because the message alone sends people to the
	 * driver. It means the fatbin has no cubin for this device and no PTX old
	 * enough to JIT onto it -- a build-time choice, not a broken install. The
	 * default CUDA_GENCODE covers sm_60 through Hopper-and-later; a device
	 * outside that range needs the architecture added, and anything below
	 * sm_60 cannot work at all because of atomicAdd(double *).
	 */
	if (error == cudaErrorNoKernelImageForDevice) {
		int ordinal = 0;
		cudaDeviceProp properties;
		if (cudaGetDevice(&ordinal) == cudaSuccess &&
				cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess)
			fprintf(stderr, "brainchopc: this build has no kernel for %s "
				"(compute capability %d.%d). Rebuild with "
				"CUDA_ARCH=sm_%d%d, or use -backend cpu.\n",
				properties.name, properties.major, properties.minor,
				properties.major, properties.minor);
	}
	return 1;
}

static int cuda_available(void) {
	int count = 0;
	cudaError_t error = cudaGetDeviceCount(&count);
	if (error != cudaSuccess) {
		/* Availability probing is quiet; automatic selection will use another backend. */
		(void)cudaGetLastError();
		return 0;
	}
	return count > 0;
}

/*
 * --adapter support. The index is a CUDA device ordinal, which is already the
 * numbering nvidia-smi and CUDA_VISIBLE_DEVICES use, so there is nothing to
 * flatten here the way the OpenCL backend has to flatten platforms x devices.
 */
static int cuda_adapter_count(void) {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess) {
		(void)cudaGetLastError();
		return 0;
	}
	return count;
}

static int cuda_adapter_name(int index, char *description, size_t capacity) {
	int count = cuda_adapter_count();
	cudaDeviceProp properties;
	if (index < 0 || index >= count) return 1;
	if (cudaGetDeviceProperties(&properties, index) != cudaSuccess) {
		(void)cudaGetLastError();
		return 1;
	}
	if (description && capacity)
		snprintf(description, capacity, "%s (%.2f GB, sm_%d%d)",
			properties.name,
			(double)properties.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
			properties.major, properties.minor);
	return 0;
}


/* Host primitives used by the common inference policy. */
#define gpu_error_t cudaError_t
#define GPU_SUCCESS cudaSuccess
#define gpu_event cudaEvent_t
#define gpu_alloc cudaMalloc
#define gpu_free cudaFree
#define gpu_zero cudaMemset
#define gpu_copy cudaMemcpy
#define GPU_DTOH cudaMemcpyDeviceToHost
#define GPU_HTOD cudaMemcpyHostToDevice
/* Invalid dispatch must be visible even when no CUDA launch was attempted. */
static cudaError_t dispatch_error = cudaSuccess;
static cudaError_t gpu_sync(void) {
    cudaError_t error = cudaDeviceSynchronize();
    return dispatch_error != cudaSuccess ? dispatch_error : error;
}
static cudaError_t gpu_last_error(void) {
    cudaError_t error = cudaGetLastError();
    if (dispatch_error != cudaSuccess) error = dispatch_error;
    dispatch_error = cudaSuccess;
    return error;
}
#define gpu_event_create cudaEventCreate
#define gpu_event_destroy cudaEventDestroy
#define gpu_event_record cudaEventRecord
#define gpu_event_sync cudaEventSynchronize
#define gpu_event_elapsed cudaEventElapsedTime
#define gpu_device cudaGetDevice
#define gpu_properties_get cudaGetDeviceProperties
#define gpu_properties cudaDeviceProp
#define gpu_driver_version cudaDriverGetVersion
#define BC_CUDA_ASSERT static_assert
#define gpu_half __float2half
static int gpu_begin(int ordinal) {
    return ordinal >= 0 ? cuda_error(cudaSetDevice(ordinal), "selecting device") : 0;
}
static int gpu_end(void) { return 0; }

static void gpu_conv(int vpt, int threads, const bc_act *source, bc_act *destination,
        const bc_act *weights, int input_channels, int output_channels,
        int input_stride, int dilation, size_t shared) {
#define X(V, T) if (vpt == V && threads == T) { \
    unsigned blocks = (unsigned)((CUDA_NVOX + (size_t)T * V - 1) / ((size_t)T * V)); \
    conv3d_kernel<V, T><<<blocks, T, shared>>>(source, destination, weights, \
        input_channels, output_channels, input_stride, dilation); return; }
    BC_CUDA_SCALAR_VARIANTS(X)
#undef X
    dispatch_error = cudaErrorInvalidConfiguration;
}
static void gpu_wmma(int warps, const bc_act *source, bc_act *destination,
        const bc_act *weights, int dilation) {
    if (warps <= 0) { dispatch_error = cudaErrorInvalidConfiguration; return; }
    size_t shared = (size_t)27 * CUDA_CS * CUDA_CS * sizeof(bc_act) +
        (size_t)warps * CUDA_WMMA_SCRATCH;
    size_t tiles = (CUDA_NVOX + CUDA_WMMA_TILE - 1) / CUDA_WMMA_TILE;
    unsigned blocks = (unsigned)((tiles + warps - 1) / warps);
#define X(W) if (warps == W) { \
    conv3d_wmma_kernel<W><<<blocks, W * 32, shared>>>(source, destination, weights, dilation); return; }
    BC_CUDA_WMMA_VARIANTS(X)
#undef X
    dispatch_error = cudaErrorInvalidConfiguration;
}
static void gpu_probe(int *flag) { wmma_probe_kernel<<<1, 1>>>(flag); }
static void gpu_fill(bc_act *buffer, size_t count) { fill_pattern_kernel<<<256, 256>>>(buffer, count); }
static void gpu_checksum(const bc_act *buffer, unsigned long long *sum, size_t count) {
    checksum_kernel<<<256, 256>>>(buffer, sum, count);
}
static void gpu_reduce(unsigned blocks, bc_act *buffer, double *sums, int channels) {
    reduce_channels_kernel<<<blocks, CUDA_REDUCE_THREADS>>>(buffer, sums, channels);
}
static void gpu_moments(double *sums, float *mean, float *invstd, int channels) {
    moments_finish_kernel<<<1, CUDA_MAX_CHANNELS>>>(sums, mean, invstd, channels);
}
static void gpu_norm(bc_act *buffer, const float *mean, const float *invstd,
        const float *scale, const float *bias, int channels) {
    unsigned blocks = (unsigned)((CUDA_NVOX + CUDA_THREADS - 1) / CUDA_THREADS);
    norm_gelu_kernel<<<blocks, CUDA_THREADS>>>(buffer, mean, invstd, scale, bias, channels);
}
static void gpu_classify(unsigned blocks, const bc_act *input, float *labels,
        const bc_act *weights, const bc_act *bias, int channels, int classes) {
    classify_kernel<<<blocks, CUDA_THREADS>>>(input, labels, weights, bias, channels, classes);
}

static void gpu_priors(unsigned blocks, const bc_act *input, float *planes,
        const bc_act *weights, const bc_act *bias, int channels, int classes) {
    tissue_priors_kernel<<<blocks, CUDA_THREADS>>>(input, planes, weights, bias, channels, classes);
}

#include "backend_cuda_host.h"
