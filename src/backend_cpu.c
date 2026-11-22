#include "backend.h"
#include "meshnet_cpu.h"
#include "model_specs.h"

static int cpu_available(void) {
	return 1;
}

static int cpu_segment(const unsigned char *conformed, float *result,
                       const float *weights,
                       const bc_inference_options *options) {
	/* Sole owner of this check: bc_backend->segment is the seam a future caller
	   could cross. Everything below it assumes non-NULL. */
	if (!conformed || !result || !weights) return 1;
	return meshnet_cpu_segment(conformed, result, weights, options);
}

const bc_backend bc_cpu_backend = {
	"cpu",
	cpu_available,
	cpu_segment,
	/* No adapters to choose between, so --adapter does not apply and this
	   backend is absent from the listing. Spelled out rather than left to the
	   partial-initializer rule, which -Wextra warns about. */
	NULL,
	NULL
};
