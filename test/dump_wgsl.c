/* Emit this build's assembled WGSL so it can be handed to a real compiler.
   The kernels are C macros, so the only way to see the final text -- with the
   descriptor constants substituted -- is to let the preprocessor run. */
#include "webgpu_kernels.h"
#include <stdio.h>
int main(int argc, char **argv) {
	(void)argv;
	fputs(argc > 1 ? BC_WGSL_WIDE_CONV_SOURCE : BC_WGSL_SOURCE, stdout);
	return 0;
}
