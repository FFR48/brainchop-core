/* Installation-only, baseline-ISA hardware probe. No GPU SDK headers. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cuda_driver_min.h"
#include "hip_device_properties.h"
#include "opencl_driver.h"
#ifdef _WIN32
#define OPEN(n) ((void *)LoadLibraryA(n))
#define SYM(h,n) ((void *)GetProcAddress((HMODULE)(h),n))
#else
#define OPEN(n) dlopen(n, RTLD_NOW | RTLD_LOCAL)
#define SYM(h,n) dlsym(h,n)
#endif
/* Function pointer from an exported symbol; NULL when absent. */
#define BIND(fn,h,n) do { void *a_=SYM(h,n); memcpy(&(fn),&a_,sizeof(fn)); } while (0)
static void probe_hip(void) {
#ifdef __linux__
    void *h = OPEN("libamdhip64.so");
    if (!h) {
        char path[512]; const char *root=getenv("ROCM_PATH");
        snprintf(path,sizeof(path),"%s/lib/libamdhip64.so",root && *root ? root : "/opt/rocm");
        h=OPEN(path);
    }
    if (h) {
        int (*init)(unsigned); int (*count)(int *); int (*props)(bc_hip_device_prop *, int);
        int (*retain)(void **,int); int (*release)(int);
        BIND(init,h,"hipInit"); BIND(count,h,"hipGetDeviceCount");
        BIND(props,h,"hipGetDevicePropertiesR0600");
        if (!props) BIND(props,h,"hipGetDeviceProperties");
        BIND(retain,h,"hipDevicePrimaryCtxRetain"); BIND(release,h,"hipDevicePrimaryCtxRelease");
        int n=0;
        if (init && count && props && retain && release && !init(0) && !count(&n)) for(int i=0;i<n;i++) {
            bc_hip_device_prop p={0}; void *ctx=NULL;
            if (!props(&p,i) && !retain(&ctx,i)) {
                p.gcn_arch_name[255]=0;
                char *suffix=strchr(p.gcn_arch_name,':'); if(suffix) *suffix=0;
                printf("hip\t%d\t%s\n",i,p.gcn_arch_name); release(i);
            }
        }
    }
#endif
}
static void probe_cuda(void) {
#ifdef _WIN32
    void *c=OPEN("nvcuda.dll");
#else
    void *c=OPEN("libcuda.so.1");
#endif
    if(!c) return;
    bc_cuda_driver d;
    CUresult (BC_CUDA_CALL *retain)(CUcontext *,CUdevice), (BC_CUDA_CALL *release)(CUdevice);
    BIND(d.init,c,"cuInit"); BIND(d.device_count,c,"cuDeviceGetCount"); BIND(d.device_get,c,"cuDeviceGet");
    BIND(d.device_attribute,c,"cuDeviceGetAttribute"); BIND(d.driver_version,c,"cuDriverGetVersion");
    BIND(retain,c,"cuDevicePrimaryCtxRetain"); BIND(release,c,"cuDevicePrimaryCtxRelease");
    int n=0,v=0;
    if(d.init && d.device_count && d.device_get && d.device_attribute && d.driver_version && retain && release &&
       !d.init(0) && !d.device_count(&n) && !d.driver_version(&v)) for(int i=0;i<n;i++) {
        CUdevice dev; int major,minor; CUcontext ctx=NULL;
        /* 75/76: CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR */
        if(!d.device_get(&dev,i) && !d.device_attribute(&major,75,dev) && !d.device_attribute(&minor,76,dev) && !retain(&ctx,dev)) {
            printf("cuda\t%d\t%d\t%d\n",i,major*10+minor,v); release(dev);
        }
    }
}
static void probe_opencl(void) {
#ifdef __APPLE__
    return; /* macOS builds never select OpenCL */
#endif
    cl_uint n=0,index=0;
    if(!bc_opencl_load() || clGetPlatformIDs(0,NULL,&n) || !n || n>1024) return;
    cl_platform_id *ps=calloc(n,sizeof *ps);
    if(ps && !clGetPlatformIDs(n,ps,NULL)) for(cl_uint p=0;p<n;p++) {
        cl_uint count=0;
        if(clGetDeviceIDs(ps[p],CL_DEVICE_TYPE_GPU,0,NULL,&count) || !count || count>1024) continue;
        cl_device_id *ds=calloc(count,sizeof *ds);
        if(ds && !clGetDeviceIDs(ps[p],CL_DEVICE_TYPE_GPU,count,ds,NULL)) for(cl_uint d=0;d<count;d++,index++) {
            cl_int error=0; cl_context ctx=clCreateContext(NULL,1,&ds[d],NULL,NULL,&error);
            if(ctx && !error) { printf("opencl\t%u\tportable-source\n",index); clReleaseContext(ctx); }
        }
        free(ds);
    }
    free(ps);
}
int main(int argc, char **argv) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    printf("cpu\tavx2-fma\t%d\n", !!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")));
#endif
    if (argc > 1 && !strcmp(argv[1],"--cpu-only")) return 0;
    if (argc == 1 || !strcmp(argv[1],"--cuda")) probe_cuda();
    if (argc == 1 || !strcmp(argv[1],"--hip")) probe_hip();
    if (argc == 1 || !strcmp(argv[1],"--opencl")) probe_opencl();
    return 0;
}
