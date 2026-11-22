// MindGrab's bias-aware final argmax over planar fp32 logits. The tuned
// convolution code objects are reused unchanged; only the old max/argmax tail
// is replaced. Compile with hipcc --genco -O3 --offload-arch=gfx1151.
#include <hip/hip_runtime.h>
extern "C" __global__ void bc_mindgrab_bias_argmax(const float *logits,
        const float *bias, unsigned char *labels) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 16777216u)
        labels[i] = logits[i + 16777216u] + bias[1] > logits[i] + bias[0];
}
