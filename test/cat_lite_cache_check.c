#include "../src/cat_lite.c"
#include <assert.h>
int main(void) {
    const size_t n=(192u*1024u*1024u)/36u+1;
    float *data=malloc(n*4*sizeof(float));assert(data);
    for(size_t i=0;i<n;i++) {
        data[i]=i%3==0?.9f:.01f;data[n+i]=i%3==1?.9f:.01f;
        data[2*n+i]=i%3==2?.9f:.01f;data[3*n+i]=i%3==0?.5f:i%3==1?.8f:.2f;
    }
    cl_fit_data fit;
    assert(!cl_fit_make(&fit,data,data+n,data+2*n,data+3*n,n));
    assert(fit.count==n && !fit.weights && fit.indices[n-1]==n-1);
    cl_fit_free(&fit);
    assert(!cl_fit_make(&fit,data,data+n,data+2*n,data+3*n,1200));
    assert(fit.weights);
    cl_stats cached=cl_estimate(&fit,0);assert(cached.applied);
    free(fit.weights);fit.weights=NULL;
    cl_stats uncached=cl_estimate(&fit,0);
    assert(cached.applied==uncached.applied && cached.refusal==uncached.refusal);
    /* Production fast-math can reassociate double reductions differently when
       weights are loaded versus recomputed (observed mean delta 6e-15). Check
       a tight double bound AND exact float32 diagnostics, not struct bytes.
       End-to-end tissue output retains the separate unchanged 2^-24 gate. */
    for(int t=0;t<3;t++) {
        assert(fabs(cached.mean[t]-uncached.mean[t])<=1e-12);
        assert(fabs(cached.sigma[t]-uncached.sigma[t])<=1e-12);
        assert((float)cached.mean[t]==(float)uncached.mean[t]);
        assert((float)cached.sigma[t]==(float)uncached.sigma[t]);
    }
    cl_fit_free(&fit);free(data);
    puts("CAT-lite bounded cache and recomputation precision PASS");
}
