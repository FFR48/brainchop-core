#ifndef BRAINCHOPC_MESHNET_CPU_H
#define BRAINCHOPC_MESHNET_CPU_H

#include "backend.h"
#include "chan_stride.h"
#include "model_specs.h"

#define MESHNET_CPU_CHANNEL_STRIDE BC_CHAN_STRIDE
#define MESHNET_CPU_MAX_LAYERS BC_MODEL_NHIDDEN

/*
 * A prepared fixed-size MeshNet for the one model this executable contains.
 *
 * There is no variant field: which model this is was settled at compile time by
 * -DBC_MODEL_IS_<NAME>, and every shape and quirk that used to be selected at
 * run time now comes from the generated descriptor -- channel and class counts,
 * whether there is a post-GroupNorm affine, which GELU approximation, whether
 * the classifier applies its bias, the normalization quantiles. Adding a model
 * that reuses an existing GELU flavour therefore needs no change here at all.
 */
typedef struct {
	int layer_count;
	const int *dilation;
	float *kernel[MESHNET_CPU_MAX_LAYERS];
#if BC_MODEL_HAS_AFFINE
	float scale[MESHNET_CPU_MAX_LAYERS][MESHNET_CPU_CHANNEL_STRIDE];
	const float *bias[MESHNET_CPU_MAX_LAYERS];
#endif
	const float *classifier_weights;
#if BC_MODEL_CLASSIFIER_BIAS
	const float *classifier_bias;
#endif
} meshnet_cpu_plan;

/*
 * Build the plan from this model's weights, which are laid out exactly as
 * model.bin holds them. Returns 0 on success.
 */
int meshnet_cpu_prepare(meshnet_cpu_plan *plan, const float *weights);

void meshnet_cpu_destroy_plan(meshnet_cpu_plan *plan);

int meshnet_cpu_infer(const unsigned char *conformed, float *labels,
		const meshnet_cpu_plan *plan, const bc_inference_options *options);

/*
 * Run the whole thing: prepare, infer, destroy. This is what the CPU backend
 * calls, and it is the same call for every model.
 */
int meshnet_cpu_segment(const unsigned char *conformed, float *result,
		const float *weights, const bc_inference_options *options);

#endif
