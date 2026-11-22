/* Exact fp32 SME hidden convolutions; included only by meshnet_cpu.c. */
#ifndef BRAINCHOPC_MESHNET_SME_H
#define BRAINCHOPC_MESHNET_SME_H

#if defined(__APPLE__) && defined(__aarch64__) && __has_include(<arm_sme.h>)
#define MN_SME 1
#include <arm_sme.h>
#include <sys/sysctl.h>

__attribute__((target("sme"))) __arm_locally_streaming static int mn_sme_width(void) {
	return (int)svcntsw();
}

static int mn_sme_available(const char *feature) {
	const char *env = getenv("BRAINCHOPC_CPU_SME");
	if (env && !strcmp(env, "0"))
		return 0;
	int enabled = 0;
	size_t size = sizeof(enabled);
	return !sysctlbyname(feature, &enabled, &size, NULL, 0) && enabled && mn_sme_width() == 16;
}

/* Each tile follows the reference tap/channel order, using one fp32 FMA per update. */
__attribute__((target("sme"))) __arm_locally_streaming __arm_new("za") static void mn_sme_segment(
	const float *src, int x0, int x1, const ptrdiff_t *tap_voxel, const float *const *tap_w, int ntap,
	float *out) {
	svbool_t all = svptrue_b32();
#if MN_CS == 24
	svbool_t tail = svwhilelt_b32((uint64_t)0, (uint64_t)8);
#endif
	for (int x = x0; x < x1; x += 32) {
		int n = x1 - x < 32 ? x1 - x : 32, second = n > 16 ? 16 : 0;
		svbool_t p0 = svwhilelt_b32((uint64_t)0, (uint64_t)(n < 16 ? n : 16));
		svbool_t p1 = svwhilelt_b32((uint64_t)0, (uint64_t)(n > 16 ? n - 16 : 0));
		svzero_za();
		for (int t = 0; t < ntap; t++) {
			size_t v = (size_t)(tap_voxel[t] + x);
			const float *in = src + (v / MN_DIM) * MN_DIM * MN_CS + v % MN_DIM;
			const float *tw = tap_w[t];
			for (int ic = 0; ic < BC_MODEL_CHAN; ic++) {
				svfloat32_t a = svld1_f32(p0, in + ic * MN_DIM), b = svld1_f32(p1, in + ic * MN_DIM + second);
				svfloat32_t w = svld1_f32(all, tw + ic * MN_CS);
				svmopa_za32_f32_m(0, p0, all, a, w);
				svmopa_za32_f32_m(2, p1, all, b, w);
#if MN_CS == 24
				w = svld1_f32(tail, tw + ic * MN_CS + 16);
				svmopa_za32_f32_m(1, p0, tail, a, w);
				svmopa_za32_f32_m(3, p1, tail, b, w);
#endif
			}
		}
		for (int b = 0; b < (n < 16 ? n : 16); b++) {
			svst1_hor_za32(0, b, all, out + (x + b) * MN_CS);
#if MN_CS == 24
			svst1_hor_za32(1, b, tail, out + (x + b) * MN_CS + 16);
#endif
		}
		for (int b = 16; b < n; b++) {
			svst1_hor_za32(2, b - 16, all, out + (x + b) * MN_CS);
#if MN_CS == 24
			svst1_hor_za32(3, b - 16, tail, out + (x + b) * MN_CS + 16);
#endif
		}
	}
}
/* Preserve the reference tail loop: Clang may split its channel reduction. */
static void mn_sme_tail(const float *src, int x, const ptrdiff_t *tap_voxel, const float *const *tap_w,
						int ntap, float *out) {
	float packed[27 * BC_MODEL_CHAN];
	for (int t = 0; t < ntap; t++) {
		size_t v = (size_t)(tap_voxel[t] + x);
		const float *in = src + (v / MN_DIM) * MN_DIM * MN_CS + v % MN_DIM;
		for (int ic = 0; ic < BC_MODEL_CHAN; ic++)
			packed[t * BC_MODEL_CHAN + ic] = in[ic * MN_DIM];
	}
	float acc[MN_CS] = {0};
	for (int t = 0; t < ntap; t++) {
		const float *in = packed + t * BC_MODEL_CHAN, *w = tap_w[t];
		for (int ic = 0; ic < BC_MODEL_CHAN; ic++) {
			float v = in[ic];
			for (int c = 0; c < MN_CS; c++)
				acc[c] += v * w[(size_t)ic * MN_CS + c];
		}
	}
	memcpy(out + x * MN_CS, acc, sizeof acc);
}
static void mn_sme_conv_rows(void *context, int z_low, int z_high) {
	const mn_conv_hidden_slab *s = context;
	const int d = s->dilation;
	for (int z = z_low; z <= z_high; z++)
		for (int y = 0; y < MN_DIM; y++) {
			ptrdiff_t row_offset[9];
			int row_tap[9], rows = 0;
			for (int tz = 0; tz < 3; tz++) {
				int zz = z + (tz - 1) * d;
				if (zz < 0 || zz >= MN_DIM)
					continue;
				for (int ty = 0; ty < 3; ty++) {
					int yy = y + (ty - 1) * d;
					if (yy < 0 || yy >= MN_DIM)
						continue;
					row_offset[rows] = ((ptrdiff_t)zz * MN_DIM + yy) * MN_DIM;
					row_tap[rows++] = (tz * 3 + ty) * 3;
				}
			}
			size_t row = ((size_t)z * MN_DIM + y) * MN_DIM;
			const int segment_x[4] = {0, d, MN_DIM - d, MN_DIM};
			for (int segment = 0; segment < 3; segment++) {
				ptrdiff_t tap_voxel[27];
				const float *tap_w[27];
				int ntap = 0;
				for (int r = 0; r < rows; r++)
					for (int tx = segment == 0; tx <= 2 - (segment == 2); tx++) {
						int tap = row_tap[r] + tx;
						tap_voxel[ntap] = row_offset[r] + (tx - 1) * d;
						tap_w[ntap++] = s->weights + (size_t)tap * BC_MODEL_CHAN * MN_CS;
					}
				int end = segment_x[segment + 1];
				end = segment_x[segment] + (end - segment_x[segment]) / MN_BLOCK * MN_BLOCK;
				mn_sme_segment(s->source, segment_x[segment], end, tap_voxel, tap_w, ntap,
							   s->destination + row * MN_CS);
				for (int x = end; x < segment_x[segment + 1]; x++)
					mn_sme_tail(s->source, x, tap_voxel, tap_w, ntap, s->destination + row * MN_CS);
			}
		}
}
static void mn_sme_conv(const float *source, float *destination, const float *weights, int dilation,
						const mn_region *region) {
	mn_conv_hidden_slab slab = {source, destination, weights, dilation, region};
	bc_parallel_for(0, MN_DIM - 1, MN_GRAIN, mn_sme_conv_rows, &slab);
}
static void mn_sme_norm_rows(void *context, int z_low, int z_high) {
	const mn_norm_slab *slab = context;
	float *buffer = slab->buffer;
	const float *mean = slab->mean, *invstd = slab->invstd;
#if BC_MODEL_HAS_AFFINE
	const float *scale = slab->scale, *bias = slab->bias;
#endif
	/* Keep channel SIMD and transpose only after the original fp32 arithmetic. */
	float tmp[MN_DIM * MN_CS];
	for (int z = z_low; z <= z_high; z++)
		for (int y = 0; y < MN_DIM; y++) {
			float *row = buffer + ((size_t)z * MN_DIM + y) * MN_DIM * MN_CS;
			for (int x = 0; x < MN_DIM; x++)
				for (int channel = 0; channel < MN_CS; channel++) {
					float normalized = (row[x * MN_CS + channel] - mean[channel]) * invstd[channel];
#if BC_MODEL_HAS_AFFINE
					normalized = normalized * scale[channel] + bias[channel];
#endif
					tmp[x * MN_CS + channel] = mn_gelu(normalized);
				}
			for (int c = 0; c < MN_CS; c++)
				for (int x = 0; x < MN_DIM; x++)
					row[c * MN_DIM + x] = tmp[x * MN_CS + c];
		}
}
static void mn_sme_norm(float *buffer, double *partial,
#if BC_MODEL_HAS_AFFINE
						const float *scale, const float *bias,
#endif
						const mn_region *region) {
	float mean[MN_CS], invstd[MN_CS];
	mn_moments(buffer, partial, mean, invstd, region);
	mn_norm_slab slab = {.buffer = buffer,
						 .mean = mean,
						 .invstd = invstd,
#if BC_MODEL_HAS_AFFINE
						 .scale = scale,
						 .bias = bias,
#endif
						 .region = region};
	bc_parallel_for(0, MN_DIM - 1, MN_GRAIN, mn_sme_norm_rows, &slab);
}

#else
#define MN_SME 0
static int mn_sme_available(const char *feature) {
	(void)feature;
	return 0;
}
#endif
#endif
