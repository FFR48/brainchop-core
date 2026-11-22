#ifndef BRAINCHOP_CUDA_DRIVER_MIN_H
#define BRAINCHOP_CUDA_DRIVER_MIN_H

/* ABI declarations for dynamically loaded Driver API entry points. No CUDA
 * SDK headers or link library. Device pointers and size_t are 64-bit: these
 * modules are built for a 64-bit host. CUDA headers must not share this TU. */
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define BC_CUDA_CALL __stdcall
#else
#define BC_CUDA_CALL
#endif
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;

/* name, exported driver symbol, parameters */
#define BC_CUDA_DRIVER_FUNCTIONS(X) \
 X(init, cuInit, (unsigned int)) \
 X(driver_version, cuDriverGetVersion, (int *)) \
 X(device_count, cuDeviceGetCount, (int *)) \
 X(device_get, cuDeviceGet, (CUdevice *, int)) \
 X(device_name, cuDeviceGetName, (char *, int, CUdevice)) \
 X(device_attribute, cuDeviceGetAttribute, (int *, int, CUdevice)) \
 X(device_memory, cuDeviceTotalMem_v2, (size_t *, CUdevice)) \
 X(ctx_current, cuCtxGetCurrent, (CUcontext *)) \
 X(ctx_set, cuCtxSetCurrent, (CUcontext)) \
 X(ctx_create, cuCtxCreate_v2, (CUcontext *, unsigned int, CUdevice)) \
 X(ctx_destroy, cuCtxDestroy_v2, (CUcontext)) \
 X(module_load, cuModuleLoadData, (CUmodule *, const void *)) \
 X(module_unload, cuModuleUnload, (CUmodule)) \
 X(function_get, cuModuleGetFunction, (CUfunction *, CUmodule, const char *)) \
 X(alloc, cuMemAlloc_v2, (CUdeviceptr *, size_t)) \
 X(free, cuMemFree_v2, (CUdeviceptr)) \
 X(copy_htod, cuMemcpyHtoD_v2, (CUdeviceptr, const void *, size_t)) \
 X(copy_dtoh, cuMemcpyDtoH_v2, (void *, CUdeviceptr, size_t)) \
 X(memset_d8, cuMemsetD8_v2, (CUdeviceptr, unsigned char, size_t)) \
 X(launch, cuLaunchKernel, (CUfunction, unsigned int, unsigned int, unsigned int, \
   unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void **, void **)) \
 X(sync, cuCtxSynchronize, (void)) \
 X(event_create, cuEventCreate, (CUevent *, unsigned int)) \
 X(event_destroy, cuEventDestroy_v2, (CUevent)) \
 X(event_record, cuEventRecord, (CUevent, CUstream)) \
 X(event_sync, cuEventSynchronize, (CUevent)) \
 X(event_elapsed, cuEventElapsedTime, (float *, CUevent, CUevent)) \
 X(error_string, cuGetErrorString, (CUresult, const char **))

typedef struct {
#define X(name, symbol, args) CUresult (BC_CUDA_CALL *name) args;
 BC_CUDA_DRIVER_FUNCTIONS(X)
#undef X
} bc_cuda_driver;
#endif
