#include "webgl2_kernels.h"
#include <stddef.h>

/*
 * One definition of each assembled GLSL program. The kernels are macros so the
 * C preprocessor can substitute this model's descriptor constants into the
 * source text, and this is where that assembly is realised into storage the
 * backend can hand to glShaderSource.
 *
 * Separate strings rather than one module, unlike bc_wgsl_source: WGSL compiles
 * one module and exposes each entry point as its own pipeline, whereas GLSL has
 * no entry-point concept and every program is its own vertex+fragment pair.
 */
const char *const bc_glsl_vertex = BC_GLSL_VERTEX;
const char *const bc_glsl_conv_first = BC_GLSL_CONV_FIRST;
const char *const bc_glsl_conv_hidden = BC_GLSL_CONV_HIDDEN;
#if BC_WGL_HAS_VOX2
const char *const bc_glsl_conv_hidden2 = BC_GLSL_CONV_HIDDEN2;
#else
/* No VOX=2 at this stride: it would need 2 * CS/4 attachments, past the eight
   browsers report. The backend's probe declines it, so this is never linked. */
const char *const bc_glsl_conv_hidden2 = NULL;
#endif
const char *const bc_glsl_moments_a = BC_GLSL_MOMENTS_A;
const char *const bc_glsl_moments_b = BC_GLSL_MOMENTS_B;
const char *const bc_glsl_moments_finish = BC_GLSL_MOMENTS_FINISH;
const char *const bc_glsl_norm = BC_GLSL_NORM;
const char *const bc_glsl_classify = BC_GLSL_CLASSIFY;
