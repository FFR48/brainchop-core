#include "webgpu_kernels.h"

/*
 * One definition of the assembled WGSL. The kernels are macros so the C
 * preprocessor can substitute this model's descriptor constants into the
 * source text -- WGSL has no preprocessor of its own -- and this is where that
 * assembly is realised into storage the backend can hand to the driver.
 */
const char *const bc_wgsl_source = BC_WGSL_SOURCE;
const char *const bc_wgsl_wide_conv_source = BC_WGSL_WIDE_CONV_SOURCE;
