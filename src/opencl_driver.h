#ifndef BC_OPENCL_DRIVER_H
#define BC_OPENCL_DRIVER_H
/* Minimal OpenCL 1.2 ABI used by this runtime; no SDK headers or link library.
 * Numeric ABI values/signatures: Khronos OpenCL 1.2 API specification.
 * The kernel source and arithmetic remain in backend_opencl.c unchanged. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#define BC_CL_CALL __stdcall
#else
#include <dlfcn.h>
#define BC_CL_CALL
#endif

typedef int32_t cl_int;
typedef uint32_t cl_uint, cl_bool;
typedef uint64_t cl_ulong;
typedef intptr_t cl_context_properties;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;
#define CL_SUCCESS 0
#define CL_FALSE 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU ((cl_ulong)4)
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_HOST_UNIFIED_MEMORY 0x1035
#define CL_MEM_READ_WRITE ((cl_ulong)1)
#define CL_MEM_READ_ONLY ((cl_ulong)4)
#define CL_MEM_COPY_HOST_PTR ((cl_ulong)32)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_QUEUE_PROFILING_ENABLE ((cl_ulong)2)

#define BC_OPENCL_FUNCTIONS(X) \
 X(cl_int,clGetPlatformIDs,(cl_uint,cl_platform_id *,cl_uint *)) \
 X(cl_int,clGetPlatformInfo,(cl_platform_id,cl_uint,size_t,void *,size_t *)) \
 X(cl_int,clGetDeviceIDs,(cl_platform_id,cl_ulong,cl_uint,cl_device_id *,cl_uint *)) \
 X(cl_int,clGetDeviceInfo,(cl_device_id,cl_uint,size_t,void *,size_t *)) \
 X(cl_context,clCreateContext,(const cl_context_properties *,cl_uint,const cl_device_id *,void (BC_CL_CALL *)(const char *,const void *,size_t,void *),void *,cl_int *)) \
 X(cl_command_queue,clCreateCommandQueue,(cl_context,cl_device_id,cl_ulong,cl_int *)) \
 X(cl_mem,clCreateBuffer,(cl_context,cl_ulong,size_t,void *,cl_int *)) \
 X(cl_program,clCreateProgramWithSource,(cl_context,cl_uint,const char **,const size_t *,cl_int *)) \
 X(cl_int,clBuildProgram,(cl_program,cl_uint,const cl_device_id *,const char *,void (BC_CL_CALL *)(cl_program,void *),void *)) \
 X(cl_int,clGetProgramBuildInfo,(cl_program,cl_device_id,cl_uint,size_t,void *,size_t *)) \
 X(cl_kernel,clCreateKernel,(cl_program,const char *,cl_int *)) \
 X(cl_int,clSetKernelArg,(cl_kernel,cl_uint,size_t,const void *)) \
 X(cl_int,clEnqueueNDRangeKernel,(cl_command_queue,cl_kernel,cl_uint,const size_t *,const size_t *,const size_t *,cl_uint,const cl_event *,cl_event *)) \
 X(cl_int,clEnqueueReadBuffer,(cl_command_queue,cl_mem,cl_bool,size_t,size_t,void *,cl_uint,const cl_event *,cl_event *)) \
 X(cl_int,clEnqueueWriteBuffer,(cl_command_queue,cl_mem,cl_bool,size_t,size_t,const void *,cl_uint,const cl_event *,cl_event *)) \
 X(cl_int,clFinish,(cl_command_queue)) \
 X(cl_int,clReleaseMemObject,(cl_mem)) \
 X(cl_int,clReleaseKernel,(cl_kernel)) \
 X(cl_int,clReleaseProgram,(cl_program)) \
 X(cl_int,clReleaseCommandQueue,(cl_command_queue)) \
 X(cl_int,clReleaseContext,(cl_context))
#define BC_OCL_DECLARE(result,name,args) static result (BC_CL_CALL *name) args;
BC_OPENCL_FUNCTIONS(BC_OCL_DECLARE)
#undef BC_OCL_DECLARE

static int bc_opencl_load(void) {
    static int attempted,ready;
    if (attempted) return ready;
    attempted=1;
#ifdef _WIN32
    HMODULE library=LoadLibraryA("OpenCL.dll");
#define BC_OCL_SYMBOL(name) GetProcAddress(library,name)
#define BC_OCL_CLOSE() FreeLibrary(library)
    typedef FARPROC symbol_type;
#else
#ifdef __APPLE__
    void *library=dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL",RTLD_NOW|RTLD_LOCAL);
#else
    void *library=dlopen("libOpenCL.so.1",RTLD_NOW|RTLD_LOCAL);
#endif
#define BC_OCL_SYMBOL(name) dlsym(library,name)
#define BC_OCL_CLOSE() dlclose(library)
    typedef void *symbol_type;
#endif
    if (!library) return 0;
#define BC_OCL_LOAD(result,name,args) do { \
    symbol_type address=BC_OCL_SYMBOL(#name); \
    if (!address || sizeof(name)!=sizeof(address)) goto fail; \
    memcpy(&name,&address,sizeof(name)); \
} while (0);
    BC_OPENCL_FUNCTIONS(BC_OCL_LOAD)
#undef BC_OCL_LOAD
    ready=1;
    return 1;
fail:
    BC_OCL_CLOSE();
    return 0;
#undef BC_OCL_SYMBOL
#undef BC_OCL_CLOSE
}
#endif
