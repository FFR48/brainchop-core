#ifndef BRAINCHOPC_CHAN_STRIDE_H
#define BRAINCHOPC_CHAN_STRIDE_H

#include "model_specs.h"

/*
 * The packed channel stride the CPU, WebGPU and WebGL2 engines lay activations
 * out with. Engine policy, not model metadata: model.json states how many
 * channels a layer has, and how many lanes we stride by is our choice. CUDA
 * deliberately keeps its own in the generated module catalog, so it does not
 * read this.
 *
 * Padding is inert. GroupNorm here is one group per channel, so a zeroed
 * channel has mean 0 and variance 0, normalises to (0-0)/sqrt(0+eps) = 0, and
 * gelu(0) = 0. It stays identically zero for every layer, which is why every
 * backend may rely on the stride rather than the logical count.
 *
 * A ladder of literals rather than an arithmetic round-up, because these tokens
 * are stringified into WGSL and GLSL source. BC_WG_STR(expression) would paste
 * the expression text, not its value, and emit "const CS : u32 = (((24)+7)/8*8)u"
 * -- which does not compile and would be found in a shader log, not here.
 */
#if BC_MODEL_CHAN <= 16
#define BC_CHAN_STRIDE 16
#elif BC_MODEL_CHAN <= 24
#define BC_CHAN_STRIDE 24
#else
/* No 32 arm: webgpu_kernels.h only carries accumulator lists for 16 and 24,
   so a stride this header accepted but the shader did not would fail in a
   generated-source error instead of here. Add both together. */
#error "no channel stride for this count: add an arm here, an accumulator list in webgpu_kernels.h, and a plane list in webgl2_kernels.h"
#endif

#endif
