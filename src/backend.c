#include "backend.h"
#include <string.h>

/*
 * Every accelerator backend this build contains, in a single place so that the
 * listing cannot drift from what bc_backend_select can actually return. The CPU
 * backend is deliberately absent: it has no adapters to choose between, and
 * listing it would invite "--adapter 0" to mean something on a CPU run.
 */
static const bc_backend *const accelerators[] = {
#ifdef HAVE_CUDA
	&bc_cuda_backend,
#endif
#ifdef HAVE_OPENCL
	&bc_opencl_backend,
#endif
#ifdef HAVE_HIP
	&bc_hip_backend,
#endif
#ifdef HAVE_METAL
	&bc_metal_backend,
#endif
#ifdef HAVE_WEBGPU
	&bc_webgpu_backend,
#endif
#ifdef HAVE_WEBGL2
	&bc_webgl2_backend,
#endif
	NULL
};

int bc_backend_list_adapters(FILE *stream) {
	if (!stream) return 0;
	int total = 0;
	for (size_t i = 0; accelerators[i]; i++) {
		const bc_backend *backend = accelerators[i];
		if (!backend->adapter_count || !backend->adapter_name) continue;
		int count = backend->adapter_count();
		if (count <= 0) {
			/*
			 * Say so rather than staying silent. A backend compiled in but
			 * finding no device is exactly the case someone runs this to
			 * diagnose, and an empty listing that omits it reads as "this build
			 * has no OpenCL" rather than "this machine has no OpenCL device".
			 */
			fprintf(stream, "%s: no adapters found\n", backend->name);
			continue;
		}
		fprintf(stream, "%s:\n", backend->name);
		for (int index = 0; index < count; index++) {
			char description[512] = "";
			if (backend->adapter_name(index, description, sizeof(description)))
				snprintf(description, sizeof(description), "(unavailable)");
			fprintf(stream, "  %d  %s\n", index, description);
			total++;
		}
	}
	if (!total)
		fprintf(stream, "no selectable GPU adapters found\n");
	return total;
}

int bc_backend_adapter_valid(const bc_backend *backend, int index) {
	if (!backend || index < 0) return 1;   /* -1 means "backend chooses" */
	/*
	 * BOTH hooks, matching bc_backend_list_adapters above. Validating on
	 * adapter_count alone would let a backend that supplies it without
	 * adapter_name accept an index the listing never showed -- a selectable
	 * device the user cannot see. No backend does that today; requiring the
	 * pair in both places is what keeps it that way.
	 */
	if (!backend->adapter_count || !backend->adapter_name) {
		fprintf(stderr, "brainchop: the %s backend does not support --adapter\n",
			backend->name);
		return 0;
	}
	int count = backend->adapter_count();
	if (index >= count) {
		fprintf(stderr,
			"brainchop: --adapter %d is out of range for the %s backend, "
			"which has %d (0..%d)\n",
			index, backend->name, count, count > 0 ? count - 1 : 0);
		return 0;
	}
	return 1;
}

const bc_backend *bc_backend_select(int force_cpu, const char *requested) {
	if (force_cpu)
		return &bc_cpu_backend;
	if (requested && *requested && strcmp(requested, "auto") != 0) {
		if (strcmp(requested, "cpu") == 0) return &bc_cpu_backend;
#ifdef HAVE_CUDA
		if (strcmp(requested, "cuda") == 0 && bc_cuda_backend.available())
			return &bc_cuda_backend;
#endif
#ifdef HAVE_OPENCL
		if (strcmp(requested, "opencl") == 0 && bc_opencl_backend.available())
			return &bc_opencl_backend;
#endif
#ifdef HAVE_HIP
		if (strcmp(requested, "hip") == 0 && bc_hip_backend.available())
			return &bc_hip_backend;
#endif
#ifdef HAVE_METAL
		if (strcmp(requested, "metal") == 0 && bc_metal_backend.available())
			return &bc_metal_backend;
#endif
#ifdef HAVE_WEBGPU
		if (strcmp(requested, "webgpu") == 0 && bc_webgpu_backend.available())
			return &bc_webgpu_backend;
#endif
#ifdef HAVE_WEBGL2
		if (strcmp(requested, "webgl2") == 0 && bc_webgl2_backend.available())
			return &bc_webgl2_backend;
#endif
		return NULL;
	}
#ifdef HAVE_CUDA
	if (bc_cuda_backend.available()) return &bc_cuda_backend;
#endif
#ifdef HAVE_HIP
	if (bc_hip_backend.available()) return &bc_hip_backend;
#endif
#ifdef HAVE_METAL
	if (bc_metal_backend.available()) return &bc_metal_backend;
#endif
#ifdef HAVE_WEBGPU
	if (bc_webgpu_backend.available()) return &bc_webgpu_backend;
#endif
/* After WebGPU deliberately: the two are never compiled together today, but if
   they ever are, WebGPU is the preferred path and WebGL2 is the fallback. */
#ifdef HAVE_WEBGL2
	if (bc_webgl2_backend.available()) return &bc_webgl2_backend;
#endif
	/* OpenCL is intentionally explicit-only (-backend opencl): integrated GPUs
	   (e.g. Intel UHD 620) often tune slowly or hit Windows WDDM TDR and run
	   slower than multicore CPU, while discrete NVIDIA/AMD GPUs use CUDA/HIP. */
	return &bc_cpu_backend;
}
