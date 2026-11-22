/* Emit this build's assembled GLSL so it can be handed to a real compiler.
   The kernels are C macros, so the only way to see the final text -- with the
   descriptor constants substituted -- is to let the preprocessor run.

   Unlike dump_wgsl.c this emits SEVERAL programs, because GLSL has no
   entry-point concept: each program is its own vertex+fragment pair. They are
   separated by a marker line that glsl_validate.mjs splits on, so one file per
   model stays one invocation. */
#include "webgl2_kernels.h"
#include <stdio.h>

static void emit(const char *name, const char *vertex, const char *fragment) {
	printf("//--- program %s vertex\n%s", name, vertex);
	printf("//--- program %s fragment\n%s", name, fragment);
}

int main(void) {
	emit("conv_first", BC_GLSL_VERTEX, BC_GLSL_CONV_FIRST);
	emit("conv_hidden", BC_GLSL_VERTEX, BC_GLSL_CONV_HIDDEN);
#if BC_WGL_HAS_VOX2
	/* Not defined at stride 24: two z slices would want 2 * CS/4 draw buffers,
	   past the eight browsers report, so the backend can never link it. */
	emit("conv_hidden2", BC_GLSL_VERTEX, BC_GLSL_CONV_HIDDEN2);
#endif
	emit("moments_a", BC_GLSL_VERTEX, BC_GLSL_MOMENTS_A);
	emit("moments_b", BC_GLSL_VERTEX, BC_GLSL_MOMENTS_B);
	emit("moments_finish", BC_GLSL_VERTEX, BC_GLSL_MOMENTS_FINISH);
	emit("norm", BC_GLSL_VERTEX, BC_GLSL_NORM);
	emit("classify", BC_GLSL_VERTEX, BC_GLSL_CLASSIFY);
	return 0;
}
