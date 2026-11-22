#ifndef BRAINCHOPC_BACKEND_ACCEL_H
#define BRAINCHOPC_BACKEND_ACCEL_H

#include "backend.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
	int channels;
	int layers;
	int classes;
	int classifier_bias;
	size_t weight_count;
	const int *dilation;
} bc_accel_model_spec;

#ifdef __cplusplus
extern "C" {
#endif

const bc_accel_model_spec *bc_accel_spec(void);

/*
 * Persistent per-device tuning cache, shared by the accelerator backends.
 *
 * Lives in $XDG_CACHE_HOME/brainchop/tune.txt (else ~/.cache/brainchop/), one
 * line per key: `<key>|<v0>|<v1>|...`. The key is the backend's business and
 * should identify everything whose change could move the optimum -- device,
 * driver, model, and the codegen the values were measured against.
 *
 * This lived in backend_cuda.cu until OpenCL wanted it too. It touches no
 * accelerator API at all, and copying it was exactly the shape of duplication
 * that let the GELU diverge across four backends.
 *
 * Everything here degrades to "no cached answer": an unreadable, unwritable or
 * malformed cache is not an error, because a segmentation must not fail over a
 * performance hint.
 *
 * bc_accel_tune_load returns 0 and fills `values` on a hit, nonzero on a miss.
 * bc_accel_tune_store returns 0 on success. Storing REPLACES every line sharing
 * this key's prefix up to the last separator -- i.e. every field EXCEPT the
 * last must match. So the last field is the only one whose change collects its
 * predecessor; put the stamp there. A change to any earlier field (driver
 * version, codegen list) leaves the old line behind, which is harmless but is
 * not what "self-cleaning" would mean.
 */
int bc_accel_tune_load(const char *key, int *values, int count);
int bc_accel_tune_store(const char *key, const int *values, int count);

/* Host-side preparation shared by CUDA and OpenCL. Returned buffers are owned. */
float *bc_accel_normalize_input(const unsigned char *source);
float *bc_accel_copy_weights(const float *weights);

/*
 * Narrow a normalized fp32 input volume to fp16, in place of the identical
 * serial loop both accelerator backends carried. Returns an owned buffer of
 * ACCEL_NVOX uint16_t holding IEEE half bit patterns -- CUDA reinterprets them
 * as __half, OpenCL hands them straight to a `half` buffer.
 *
 * Parallel and bit-exact for being a pure elementwise map. It is 16.7M
 * conversions on the critical path of every GPU run.
 *
 * One configuration does NOT get the parallelism: on Windows this file is
 * compiled a second time by MSVC for the CUDA DLL, without -DBC_THREADS (it was
 * without /openmp before the pool stopped being OpenMP), so the copy
 * cuda_segment actually calls runs the whole range on the calling thread. The
 * OUTPUT is identical either way -- that is what makes compiling it twice safe
 * rather than a smell -- but the speed claim is Linux and macOS only.
 */
uint16_t *bc_accel_narrow_input(const float *normalized);

#ifdef __cplusplus
}
#endif

#endif
