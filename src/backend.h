#ifndef BRAINCHOPC_BACKEND_H
#define BRAINCHOPC_BACKEND_H

#include <stddef.h>
#include <stdio.h>

/*
 * There is no model enumerator and no model parameter: each executable contains
 * exactly one model, chosen at compile time by -DBC_MODEL_IS_<NAME>, and every
 * backend reads its shape from the generated descriptor. An enum here would
 * have to name every model that exists, which is precisely the shared-code edit
 * that adding a model is supposed not to require.
 */

typedef int (*bc_class_writer)(int class_index, const float *values,
		size_t count, void *context);

typedef struct {
	int roi_enabled;
	int x_low, x_high;
	int y_low, y_high;
	int z_low, z_high;
	bc_class_writer class_writer;
	void *class_writer_context;
	/*
	 * --pve: a caller-owned BC_PVE_PLANES x nvox buffer the engine fills with
	 * the grouped GM/WM/CSF tissue priors and the brain support. NULL when the
	 * mode is off, which is every backend that does not implement it -- see
	 * cat_lite.h for the layout.
	 */
	float *tissue_planes;
	/*
	 * Force a re-measure of the per-device kernel shape (--tune). Backends
	 * without a tuner ignore it; the CUDA backend tunes on its own the first
	 * time it meets a device regardless, so this only overrides the cache.
	 */
	int force_tuning;
	/*
	 * Which of this backend's adapters to run on (--adapter), or -1 to let the
	 * backend choose. The index is PER BACKEND and is a position in that
	 * backend's own enumeration order, which is why bc_backend_list_adapters
	 * prints the backends as separate groups: "1" means different hardware to
	 * OpenCL than it does to CUDA.
	 */
	int adapter;
	/* Optional output flag: non-NULL permits device blur/support preparation.
	   NULL requests raw priors, preserving the existing backend contract.
	   Reset before every attempt, including fallback. Appended to preserve
	   offsets of fields used by separately compiled CUDA backends. */
	int *tissue_priors_prepared;
	/* --no-turbo: the CPU runs its exact engine, GPUs their fp16 path.
	   Appended for the same ABI reason. */
	int no_turbo;
} bc_inference_options;

typedef struct bc_backend {
	const char *name;
	int (*available)(void);
	int (*segment)(const unsigned char *conformed, float *result,
	               const float *weights, const bc_inference_options *options);
	/*
	 * Adapter enumeration, for --adapter and its listing form. Both are
	 * optional: a backend that leaves them NULL simply does not appear in the
	 * listing and ignores a requested index. Every existing initializer is
	 * positional with three members, so those backends get NULL here for free.
	 *
	 * They are FUNCTION POINTERS IN THIS STRUCT rather than separately exported
	 * symbols, and on Windows that is load-bearing rather than stylistic: the
	 * CUDA backend lives in a DLL across an MSVC/MinGW ABI boundary, and this
	 * struct is the single symbol that crosses it. Routing enumeration through
	 * it means cuda_exports.def needs no new entry and the boundary stays one
	 * symbol wide.
	 *
	 * adapter_count returns the number of usable adapters, or 0 when the
	 * backend cannot enumerate or nothing is present -- never negative.
	 * adapter_name writes a human-readable description of `index` and returns 0
	 * on success, nonzero if the index is out of range.
	 */
	int (*adapter_count)(void);
	int (*adapter_name)(int index, char *description, size_t capacity);
} bc_backend;

#ifdef __cplusplus
extern "C" {
#endif

extern const bc_backend bc_cpu_backend;
#ifdef HAVE_CUDA
extern const bc_backend bc_cuda_backend;
#endif
#ifdef HAVE_OPENCL
extern const bc_backend bc_opencl_backend;
#endif
#ifdef HAVE_HIP
extern const bc_backend bc_hip_backend;
#endif
#ifdef HAVE_METAL
extern const bc_backend bc_metal_backend;
#endif
#ifdef HAVE_WEBGPU
extern const bc_backend bc_webgpu_backend;
#endif
#ifdef HAVE_WEBGL2
extern const bc_backend bc_webgl2_backend;
#endif

/* Returns the requested or best available backend, or NULL if unavailable. */
const bc_backend *bc_backend_select(int force_cpu, const char *requested);

/*
 * Print every backend's selectable adapters, grouped by backend, with the index
 * --adapter takes. Returns the total number of adapters found, so a caller can
 * tell "no GPU on this machine" from "listed something".
 */
int bc_backend_list_adapters(FILE *stream);

/*
 * Whether `index` is a valid adapter for `backend`. Called at startup so a
 * mistyped index fails immediately, naming the range, rather than after conform
 * and preprocessing have already run.
 */
int bc_backend_adapter_valid(const bc_backend *backend, int index);

#ifdef __cplusplus
}
#endif

#endif
