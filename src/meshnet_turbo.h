/*
 * CPU turbo (default; --no-turbo runs exact): compact hidden convolutions.
 * Included only by meshnet_cpu.c, after the exact engine, whose normalization,
 * first convolution, GroupNorm and classifier it reuses unchanged. The exact
 * engine stays the fixture-gated reference.
 *
 * Activations and weights use hi + lo/128. Three byte dot products compute
 * hi*hi and hi*lo + lo*hi; omitting lo*lo/16384 keeps two accumulators.
 * GroupNorm/GELU run in float; the first layer and final normalization are
 * exact. Runs on arm64 i8mm USDOT or x86 VNNI, with each kernel bit-identical
 * to the scalar reference (test/cpu_turbo_check.c). Apple SME instead uses
 * fp16 operands and fp32 sums, with the same two-byte activation storage.
 */
#ifndef BRAINCHOPC_MESHNET_TURBO_H
#define BRAINCHOPC_MESHNET_TURBO_H

#include <float.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#ifndef AT_HWCAP2       /* glibc < 2.33 lacks the names, not the bits */
#define AT_HWCAP2 26
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
#endif

/* GELU's minimum is -0.17. The residual byte is centered at 64. */
#define MT_ZP 7
#define MT_G (MN_CS / 4)
#define MT_AS (2 * MN_CS) /* per voxel: coarse channels, then residuals */
#define MT_RES_ZP 64
/* A u8 activation volume is read 16 bytes at a time past its last voxel. */
#define MT_ACT_PAD 16

/*
 * Weights are split per output channel: w = (hi + lo / 128) * s, hi and lo s8,
 * ~15 bits. Plain s8 weights cost 1.5-3 points of agreement (docs/audit-history.md).
 * Per tap the hi block then the lo block, each [ic/4][oc][4]: four input
 * channels of one output channel adjacent, as USDOT and VPDPBUSD consume them.
 * corr includes both activation zero points. The low accumulator contains
 * both cross products; scale is [s oc..., s/128 oc...]. Every kernel combines as
 * fma(lo, s_lo, hi * s_hi), so all are bit-identical to the scalar reference.
 */
#define MT_TAP (2 * MN_CS * MN_CS)
typedef void (*mt_block_fn)(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel,
	const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out);

static inline int mt_widx(int ic, int oc) { return ((ic >> 2) * MN_CS + oc) * 4 + (ic & 3); }

/* The reference every kernel must match bit for bit (test/cpu_turbo_check.c). */
#ifdef BC_CPU_TURBO_CHECK
#define MT_DEFINE_SCALAR(NB) \
static void mt_scalar_block##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	for (int b = 0; b < NB; b++) \
		for (int oc = 0; oc < MN_CS; oc++) { \
			int32_t hi = 0, lo = 0; \
			for (int t = 0; t < ntap; t++) { \
				const uint8_t *in = src + (x + b + tap_voxel[t]) * MT_AS; \
				for (int ic = 0; ic < MN_CS; ic++) { \
					hi += tap_w[t][mt_widx(ic, oc)] * in[ic]; \
					lo += tap_w[t][MN_CS * MN_CS + mt_widx(ic, oc)] * in[ic]; \
					lo += tap_w[t][mt_widx(ic, oc)] * in[MN_CS + ic]; \
				} \
			} \
			out[b * MN_CS + oc] = fmaf((float)(lo - corr[MN_CS + oc]), scale[MN_CS + oc], \
				(float)(hi - corr[oc]) * scale[oc]); \
		} \
}
MT_DEFINE_SCALAR(1)
MT_DEFINE_SCALAR(2)
#endif

#if defined(__aarch64__)
#define MT_NEON_STEP(NB, g, lane) \
	for (int o = 0; o < MT_G; o++) { \
		int8x16_t wh = vld1q_s8(tw + ((g) * MN_CS + o * 4) * 4); \
		int8x16_t wl = vld1q_s8(tw + MN_CS * MN_CS + ((g) * MN_CS + o * 4) * 4); \
		for (int b = 0; b < NB; b++) { \
			hi[b][o] = vsudotq_laneq_s32(hi[b][o], wh, a[b], lane); \
			lo[b][o] = vsudotq_laneq_s32(lo[b][o], wl, a[b], lane); \
			lo[b][o] = vsudotq_laneq_s32(lo[b][o], wh, r[b], lane); \
		} \
	}
#if MN_CS == 16
#define MT_NEON_TAIL(NB)
#elif MN_CS == 24
/* Keep tail loads after the first 16 channels so the accumulators stay in registers. */
#define MT_NEON_TAIL(NB) \
	__asm__ volatile("" ::: "memory"); \
	for (int b = 0; b < NB; b++) { \
		a[b] = vld1q_u8(in + b * MT_AS + 16); \
		r[b] = vld1q_u8(in + b * MT_AS + MN_CS + 16); \
	} \
	MT_NEON_STEP(NB, 4, 0) MT_NEON_STEP(NB, 5, 1)
#endif
#define MT_DEFINE_NEON(NB) \
__attribute__((target("+i8mm"))) /* GCC needs the "+" */ \
static void mt_neon_block##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	int32x4_t hi[NB][MT_G], lo[NB][MT_G]; \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_G; o++) hi[b][o] = lo[b][o] = vdupq_n_s32(0); \
	for (int t = 0; t < ntap; t++) { \
		const uint8_t *in = src + (x + tap_voxel[t]) * MT_AS; \
		const int8_t *tw = tap_w[t]; \
		uint8x16_t a[NB], r[NB]; \
		for (int b = 0; b < NB; b++) { \
			a[b] = vld1q_u8(in + b * MT_AS); \
			r[b] = vld1q_u8(in + b * MT_AS + MN_CS); \
		} \
		MT_NEON_STEP(NB, 0, 0) MT_NEON_STEP(NB, 1, 1) MT_NEON_STEP(NB, 2, 2) MT_NEON_STEP(NB, 3, 3) \
		MT_NEON_TAIL(NB) \
	} \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_G; o++) { \
		float32x4_t h = vcvtq_f32_s32(vsubq_s32(hi[b][o], vld1q_s32(corr + 4 * o))); \
		float32x4_t l = vcvtq_f32_s32(vsubq_s32(lo[b][o], vld1q_s32(corr + MN_CS + 4 * o))); \
		vst1q_f32(out + b * MN_CS + 4 * o, vfmaq_f32(vmulq_f32(h, vld1q_f32(scale + 4 * o)), \
			l, vld1q_f32(scale + MN_CS + 4 * o))); \
	} \
}
MT_DEFINE_NEON(1)
MT_DEFINE_NEON(2)

#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MT_O8 (MN_CS / 8)
/* VPDPBUSD: the 4 input bytes of each voxel broadcast against 8 output
   channels x 4 inputs of weights. EVEX (AVX-512 VNNI + VL) or VEX (AVX-VNNI). */
#define MT_DEFINE_X86(NAME, TARGET, DPBUSD, NB) \
__attribute__((target(TARGET))) \
static void NAME##NB(const uint8_t *src, size_t x, const ptrdiff_t *tap_voxel, \
		const int8_t *const *tap_w, int ntap, const int32_t *corr, const float *scale, float *out) { \
	__m256i hi[NB][MT_O8], lo[NB][MT_O8]; \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_O8; o++) hi[b][o] = lo[b][o] = _mm256_setzero_si256(); \
	for (int t = 0; t < ntap; t++) { \
		const uint8_t *in = src + (x + tap_voxel[t]) * MT_AS; \
		const int8_t *tw = tap_w[t]; \
		for (int g = 0; g < MT_G; g++) { \
			__m256i a[NB], r[NB]; \
			for (int b = 0; b < NB; b++) { \
				int32_t four; memcpy(&four, in + b * MT_AS + 4 * g, 4); \
				a[b] = _mm256_set1_epi32(four); \
				memcpy(&four, in + b * MT_AS + MN_CS + 4 * g, 4); \
				r[b] = _mm256_set1_epi32(four); \
			} \
			for (int o = 0; o < MT_O8; o++) { \
				const int8_t *w = tw + (g * MN_CS + 8 * o) * 4; \
				__m256i wh = _mm256_loadu_si256((const __m256i *)w); \
				__m256i wl = _mm256_loadu_si256((const __m256i *)(w + MN_CS * MN_CS)); \
				for (int b = 0; b < NB; b++) { \
					hi[b][o] = DPBUSD(hi[b][o], a[b], wh); \
					lo[b][o] = DPBUSD(lo[b][o], a[b], wl); \
					lo[b][o] = DPBUSD(lo[b][o], r[b], wh); \
				} \
			} \
		} \
	} \
	for (int b = 0; b < NB; b++) for (int o = 0; o < MT_O8; o++) { \
		__m256 h = _mm256_cvtepi32_ps(_mm256_sub_epi32(hi[b][o], \
			_mm256_loadu_si256((const __m256i *)(corr + 8 * o)))); \
		__m256 l = _mm256_cvtepi32_ps(_mm256_sub_epi32(lo[b][o], \
			_mm256_loadu_si256((const __m256i *)(corr + MN_CS + 8 * o)))); \
		_mm256_storeu_ps(out + b * MN_CS + 8 * o, _mm256_fmadd_ps(l, \
			_mm256_loadu_ps(scale + MN_CS + 8 * o), _mm256_mul_ps(h, _mm256_loadu_ps(scale + 8 * o)))); \
	} \
}
MT_DEFINE_X86(mt_vnni512_block, "avx2,fma,avx512f,avx512vl,avx512vnni", _mm256_dpbusd_epi32, 1)
MT_DEFINE_X86(mt_vnni512_block, "avx2,fma,avx512f,avx512vl,avx512vnni", _mm256_dpbusd_epi32, 2)
MT_DEFINE_X86(mt_avxvnni_block, "avx2,fma,avxvnni", _mm256_dpbusd_avx_epi32, 1)
MT_DEFINE_X86(mt_avxvnni_block, "avx2,fma,avxvnni", _mm256_dpbusd_avx_epi32, 2)
#endif

/* The fast kernel this CPU runs (its name), or NULL: turbo then runs the
   exact engine. */
static const char *mt_select(mt_block_fn *block2, mt_block_fn *block1) {
#if defined(__aarch64__)
	int i8mm = 0;
#if defined(__APPLE__)
	size_t size = sizeof(i8mm);
	if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &i8mm, &size, NULL, 0)) i8mm = 0;
#elif defined(__linux__)
	i8mm = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#endif
	if (i8mm) { *block2 = mt_neon_block2; *block1 = mt_neon_block1; return "i8mm"; }
#elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
	if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) {
		*block2 = mt_vnni512_block2; *block1 = mt_vnni512_block1; return "avx512vnni";
	}
	/* Clang < 19 lacks the "avxvnni" builtin string: CPUID.(7,1):EAX[4].
	   The AVX2 test covers the OS's YMM state. */
	unsigned a, b, c, d;
	if (__builtin_cpu_supports("avx2") && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4))) {
		*block2 = mt_avxvnni_block2; *block1 = mt_avxvnni_block1; return "avxvnni";
	}
#endif
	(void)block2; (void)block1;
	return NULL;
}


static int mt_sme_available(void) {
	return mn_sme_available("hw.optional.arm.SME_F16F32");
}

#if MN_SME
/* 512-bit SME: 32 voxels, fp16 operands, fp32 sums. Channel pairs are
   contiguous across x so each matrix operand loads without a gather. */
__attribute__((target("sme"))) __arm_locally_streaming __arm_new("za")
static void mt_sme_segment(const uint8_t *src, int x0, int x1, const ptrdiff_t *tap_voxel,
		const int8_t *const *tap_w, int ntap, float *out) {
	svbool_t all = svptrue_b16(), floats = svptrue_b32();
#if MN_CS == 24
	svbool_t tail = svwhilelt_b16((uint64_t)0, (uint64_t)16);
	svbool_t ftail = svwhilelt_b32((uint64_t)0, (uint64_t)8);
#endif
	for (int x = x0; x < x1; x += 32) {
		int n = x1 - x < 32 ? x1 - x : 32;
		int second = n > 16 ? 32 : 0; /* Keep even inactive load addresses in bounds. */
		svbool_t p0 = svwhilelt_b16((uint64_t)0, (uint64_t)(n < 16 ? n : 16) * 2);
		svbool_t p1 = svwhilelt_b16((uint64_t)0, (uint64_t)(n > 16 ? n - 16 : 0) * 2);
		svzero_za();
		for (int t = 0; t < ntap; t++) {
			size_t v = (size_t)(tap_voxel[t] + x);
			const float16_t *in = (const float16_t *)src
				+ (v / MN_DIM) * MN_DIM * MN_CS + (v % MN_DIM) * 2;
			const float16_t *tw = (const float16_t *)tap_w[t];
			for (int g = 0; g < MN_CS / 2; g++) {
				svfloat16_t a = svld1_f16(p0, in + g * MN_DIM * 2);
				svfloat16_t b = svld1_f16(p1, in + g * MN_DIM * 2 + second);
				svfloat16_t w = svld1_f16(all, tw + g * MN_CS * 2);
				svmopa_za32_f16_m(0, p0, all, a, w);
				svmopa_za32_f16_m(2, p1, all, b, w);
#if MN_CS == 24
				w = svld1_f16(tail, tw + g * MN_CS * 2 + 32);
				svmopa_za32_f16_m(1, p0, tail, a, w);
				svmopa_za32_f16_m(3, p1, tail, b, w);
#endif
			}
		}
		for (int b = 0; b < (n < 16 ? n : 16); b++) {
			svst1_hor_za32(0, b, floats, out + (x + b) * MN_CS);
#if MN_CS == 24
			svst1_hor_za32(1, b, ftail, out + (x + b) * MN_CS + 16);
#endif
		}
		for (int b = 16; b < n; b++) {
			svst1_hor_za32(2, b - 16, floats, out + (x + b) * MN_CS);
#if MN_CS == 24
			svst1_hor_za32(3, b - 16, ftail, out + (x + b) * MN_CS + 16);
#endif
		}
	}
}

static void mt_prepare_sme(const float *kernel, int8_t *packed) {
	float16_t *w = (float16_t *)packed;
	/* Same byte count as hi/lo s8 weights; padded channels remain calloc zeros. */
	for (int t = 0; t < 27; t++)
		for (int ic = 0; ic < BC_MODEL_CHAN; ic++)
			for (int oc = 0; oc < MN_CS; oc++)
				w[t * MN_CS * MN_CS + ((ic / 2) * MN_CS + oc) * 2 + ic % 2]
					= (float16_t)kernel[((size_t)t * BC_MODEL_CHAN + ic) * MN_CS + oc];
}
#endif

/* ---------------------------------------------------- per-z GroupNorm stats */

/* Accumulate `voxels` voxels into one z slice's sums, squares and range. */
static void mt_stats_add(const float *values, size_t voxels, double *sum, double *square,
		float *low, float *high) {
	for (size_t v = 0; v < voxels; v++)
		for (int c = 0; c < MN_CS; c++) {
			float value = values[v * MN_CS + c];
			sum[c] += value;
			square[c] += (double)value * value;
			low[c] = fminf(low[c], value);
			high[c] = fmaxf(high[c], value);
		}
}

/* partial is [z][sum CS, square CS], range [z][min CS, max CS]. */
#define MT_STATS_BEGIN \
	double sum[MN_CS] = { 0 }, square[MN_CS] = { 0 }; \
	float low[MN_CS], high[MN_CS]; \
	for (int c = 0; c < MN_CS; c++) { low[c] = FLT_MAX; high[c] = -FLT_MAX; }
#define MT_STATS_END(partial, range, z) \
	for (int c = 0; c < MN_CS; c++) { \
		(partial)[(size_t)(z) * 2 * MN_CS + c] = sum[c]; \
		(partial)[(size_t)(z) * 2 * MN_CS + MN_CS + c] = square[c]; \
		(range)[(size_t)(z) * 2 * MN_CS + c] = low[c]; \
		(range)[(size_t)(z) * 2 * MN_CS + MN_CS + c] = high[c]; \
	}

typedef struct {
	const float *buffer;
	double *partial;
	float *range;
} mt_stats_slab;

static void mt_stats_rows(void *context, int z_low, int z_high) {
	const mt_stats_slab *s = (const mt_stats_slab *)context;
	for (int z = z_low; z <= z_high; z++) {
		MT_STATS_BEGIN
		mt_stats_add(s->buffer + (size_t)z * MN_DIM * MN_DIM * MN_CS, (size_t)MN_DIM * MN_DIM,
			sum, square, low, high);
		MT_STATS_END(s->partial, s->range, z)
	}
}

/* -------------------------------------------------------------- convolution */

typedef struct {
	const uint8_t *source;
	float *destination;
	const int8_t *weights;   /* int8 planes, or SME fp16 [tap][ic/2][oc][2] */
	const int32_t *tap_sum;  /* [tap][hi oc, lo oc]: MT_ZP * sum of the tap's weights */
	const float *scale;      /* [hi oc, lo oc] */
	mt_block_fn block2, block1;
	int dilation;
	double *partial;         /* stats of the output, per z, as mt_stats_rows */
	float *range;
	int sme;
} mt_conv_slab;

/* MN_DEFINE_CONV's row and x-segment structure: taps outside the volume are
   dropped (zero padding), and their zero-point correction with them. Each
   finished row feeds the GroupNorm stats while it is still in cache. */
static void mt_conv_rows(void *context, int z_low, int z_high) {
	const mt_conv_slab *s = (const mt_conv_slab *)context;
	const int d = s->dilation;
	for (int z = z_low; z <= z_high; z++) {
		MT_STATS_BEGIN
		for (int y = 0; y < MN_DIM; y++) {
			ptrdiff_t row_offset[9];
			int row_tap[9], rows = 0;
			for (int tz = 0; tz < 3; tz++) {
				int zz = z + (tz - 1) * d;
				if (zz < 0 || zz >= MN_DIM) continue;
				for (int ty = 0; ty < 3; ty++) {
					int yy = y + (ty - 1) * d;
					if (yy < 0 || yy >= MN_DIM) continue;
					row_offset[rows] = ((ptrdiff_t)zz * MN_DIM + yy) * MN_DIM;
					row_tap[rows++] = (tz * 3 + ty) * 3;
				}
			}
			const size_t row = ((size_t)z * MN_DIM + y) * MN_DIM;
			const int segment_x[4] = { 0, d, MN_DIM - d, MN_DIM };
			for (int segment = 0; segment < 3; segment++) {
				ptrdiff_t tap_voxel[27];
				const int8_t *tap_w[27];
				int32_t corr[2 * MN_CS] = { 0 };
				int ntap = 0;
				for (int r = 0; r < rows; r++)
					for (int tx = segment == 0; tx <= 2 - (segment == 2); tx++) {
						int tap = row_tap[r] + tx;
						tap_voxel[ntap] = row_offset[r] + (tx - 1) * d;
						tap_w[ntap++] = s->weights + (size_t)tap * MT_TAP;
						if (!s->sme) /* SME has no zero points */
							for (int k = 0; k < 2 * MN_CS; k++) corr[k] += s->tap_sum[tap * 2 * MN_CS + k];
					}
#if MN_SME
				if (s->sme) {
					mt_sme_segment(s->source, segment_x[segment], segment_x[segment + 1],
						tap_voxel, tap_w, ntap, s->destination + row * MN_CS);
					continue;
				}
#endif
				int x = segment_x[segment];
				for (; x + 2 <= segment_x[segment + 1]; x += 2)
					s->block2(s->source, (size_t)x, tap_voxel, tap_w, ntap, corr, s->scale,
						s->destination + (row + x) * MN_CS);
				for (; x < segment_x[segment + 1]; x++)
					s->block1(s->source, (size_t)x, tap_voxel, tap_w, ntap, corr, s->scale,
						s->destination + (row + x) * MN_CS);
			}
			mt_stats_add(s->destination + row * MN_CS, MN_DIM, sum, square, low, high);
		}
		MT_STATS_END(s->partial, s->range, z)
	}
}
_Static_assert(MN_DIM > 2 * 31, "x segments assume every dilation is under DIM/2");

/* Next layer's weights: float kernel times its input's qs, split per output
   channel into hi (+-127) and lo (+-64, the hi residual in 1/128 steps). */
static void mt_prepare(const float *kernel, const float *qs, int8_t *wq, int32_t *tap_sum,
		float *scale) {
	for (int oc = 0; oc < MN_CS; oc++) {
		float peak = 0.0f;
		for (int tap = 0; tap < 27; tap++)
			for (int ic = 0; ic < BC_MODEL_CHAN; ic++)
				peak = fmaxf(peak, fabsf(kernel[((size_t)tap * BC_MODEL_CHAN + ic) * MN_CS + oc] * qs[ic]));
		scale[oc] = peak > 0.0f ? peak / 127.0f : 1.0f;
		scale[MN_CS + oc] = scale[oc] / 128.0f;
	}
	/* Pad input channels stay zero from the calloc in mt_infer. */
	for (int tap = 0; tap < 27; tap++)
		for (int oc = 0; oc < MN_CS; oc++) {
			int32_t sum_hi = 0, sum_lo = 0;
			for (int ic = 0; ic < BC_MODEL_CHAN; ic++) {
				float w = kernel[((size_t)tap * BC_MODEL_CHAN + ic) * MN_CS + oc] * qs[ic] / scale[oc];
				int hi = (int)lrintf(w), lo = (int)lrintf((w - (float)hi) * 128.0f);
				int8_t *t = wq + (size_t)tap * MT_TAP;
				t[mt_widx(ic, oc)] = (int8_t)hi;
				t[MN_CS * MN_CS + mt_widx(ic, oc)] = (int8_t)lo;
				sum_hi += hi;
				sum_lo += lo;
			}
			tap_sum[tap * 2 * MN_CS + oc] = MT_ZP * sum_hi;
			tap_sum[tap * 2 * MN_CS + MN_CS + oc] = MT_ZP * sum_lo + MT_RES_ZP * sum_hi;
		}
}

/* ------------------------------------------------- GroupNorm, GELU, quantize */

typedef struct {
	const float *buffer;
	uint8_t *codes;
	const float *mul, *add, *inv_qs;
} mt_quant_slab;

static void mt_quant_rows(void *context, int z_low, int z_high) {
	const mt_quant_slab *s = (const mt_quant_slab *)context;
	const float *restrict buffer = s->buffer;
	uint8_t *restrict codes = s->codes;
	const float *restrict mul = s->mul, *restrict add = s->add, *restrict inv_qs = s->inv_qs;
	for (int z = z_low; z <= z_high; z++) {
		size_t base = (size_t)z * MN_DIM * MN_DIM * MN_CS;
		for (size_t v = 0; v < (size_t)MN_DIM * MN_DIM; v++)
			for (int c = 0; c < MN_CS; c++) {
				size_t i = base + v * MN_CS + c;
				size_t dst = 2 * (base + v * MN_CS) + c;
				/* >= 0 by the qs floor, so truncating x + 0.5 rounds. */
				float code = mn_gelu(buffer[i] * mul[c] + add[c]) * inv_qs[c] * 128.0f + MT_ZP * 128.0f + 0.5f;
				/* Leave room for rounding the coarse byte without overflowing 255. */
				unsigned q = (unsigned)(code < 32703.0f ? code : 32703.0f);
				codes[dst] = (uint8_t)((q + MT_RES_ZP) >> 7);
				codes[dst + MN_CS] = (uint8_t)((q + MT_RES_ZP) & 127);
			}
	}
}

#if MN_SME
static void mt_quant_sme_rows(void *context, int z_low, int z_high) {
	const mt_quant_slab *s = (const mt_quant_slab *)context;
	const float *restrict buffer = s->buffer;
	float16_t *restrict codes = (float16_t *)s->codes;
	const float *restrict mul = s->mul, *restrict add = s->add;
	/* Pair-inner traversal keeps GELU vectorized despite the transposed stores. */
	for (int z = z_low; z <= z_high; z++)
		for (int y = 0; y < MN_DIM; y++)
			for (int g = 0; g < MN_CS / 2; g++) {
				size_t row = (size_t)z * MN_DIM + y;
				for (int x = 0; x < MN_DIM; x++)
					for (int k = 0; k < 2; k++) {
						int c = g * 2 + k;
						codes[row * MN_DIM * MN_CS + g * MN_DIM * 2 + x * 2 + k]
							= (float16_t)mn_gelu(buffer[(row * MN_DIM + x) * MN_CS + c] * mul[c] + add[c]);
					}
			}
}
#endif

/* Normalize, activate and quantize `buffer` into `codes` from its per-z
   stats; returns qs. */
static void mt_norm_quant(const float *buffer, uint8_t *codes, const double *partial,
		const float *range, const meshnet_cpu_plan *plan, int layer, float *qs, int sme) {
	float mul[MN_CS], add[MN_CS], inv_qs[MN_CS];
	for (int c = 0; c < MN_CS; c++) {
		double sum = 0.0, square = 0.0;
		float low = FLT_MAX, high = -FLT_MAX;
		for (int z = 0; z < MN_DIM; z++) {
			sum += partial[(size_t)z * 2 * MN_CS + c];
			square += partial[(size_t)z * 2 * MN_CS + MN_CS + c];
			low = fminf(low, range[(size_t)z * 2 * MN_CS + c]);
			high = fmaxf(high, range[(size_t)z * 2 * MN_CS + MN_CS + c]);
		}
		double mean = sum / (double)MN_NVOX, variance = square / (double)MN_NVOX - mean * mean;
		if (!(variance > 0.0)) variance = 0.0;
		float invstd = (float)(1.0 / sqrt(variance + MN_EPS)), gamma = 1.0f, beta = 0.0f;
#if BC_MODEL_HAS_AFFINE
		gamma = plan->scale[layer][c];
		beta = c < BC_MODEL_CHAN ? plan->bias[layer][c] : 0.0f;
#else
		(void)plan; (void)layer;
#endif
		mul[c] = invstd * gamma;
		add[c] = beta - (float)mean * mul[c];
		float vmax = fmaxf(0.25f, fmaxf(mn_gelu(low * mul[c] + add[c]), mn_gelu(high * mul[c] + add[c])));
		qs[c] = c < BC_MODEL_CHAN ? fmaxf(vmax / (255.0f - MT_ZP), 0.1705f / MT_ZP) : 1.0f;
		inv_qs[c] = 1.0f / qs[c];
	}
	mt_quant_slab quant = { buffer, codes, mul, add, inv_qs };
#if MN_SME
	if (sme) {
		bc_parallel_for(0, MN_DIM - 1, MN_GRAIN, mt_quant_sme_rows, &quant);
		return;
	}
#else
	(void)sme;
#endif
	bc_parallel_for(0, MN_DIM - 1, MN_GRAIN, mt_quant_rows, &quant);
}

/* ---------------------------------------------------------------- the run */

static int mt_infer(const unsigned char *conformed, float *labels, const meshnet_cpu_plan *plan,
		mt_block_fn block2, mt_block_fn block1, const char *kernel) {
	const int sme = mt_sme_available();
	printfx("%s: CPU turbo (%s, %s)\n", BC_MODEL_NAME, sme ? "fp16" : "int8", sme ? "SME" : kernel);
	mn_region region;
	mn_make_region(NULL, &region);
	float *y = (float *)malloc(MN_NVOX * MN_CS * sizeof(*y));
	/* The codes buffer first holds the normalized fp32 input (CS >= 4 bytes). */
	uint8_t *codes = (uint8_t *)malloc(MN_NVOX * MT_AS + MT_ACT_PAD);
	double *partial = (double *)malloc((size_t)MN_DIM * 2 * MN_CS * sizeof(*partial));
	float *range = (float *)malloc((size_t)MN_DIM * 2 * MN_CS * sizeof(*range));
	int8_t *wq = (int8_t *)calloc((size_t)27 * MT_TAP, 1);
	int32_t tap_sum[27 * 2 * MN_CS] = { 0 };
	float scale[2 * MN_CS], qs[MN_CS];
	if (!y || !codes || !partial || !range || !wq) {
		printfx("%s: out of memory (turbo buffers need about %.1f GB)\n", BC_MODEL_NAME,
			(double)MN_NVOX * MN_CS * (sizeof(float) + 2) / 1e9);
		free(y); free(codes); free(partial); free(range); free(wq);
		return 1;
	}
	memset(codes + MN_NVOX * MT_AS, 0, MT_ACT_PAD);
	memset(labels, 0, MN_NVOX * sizeof(*labels));
	mn_normalize(conformed, (float *)codes, &region);
	mn_conv_first((const float *)codes, y, plan->kernel[0], plan->dilation[0], &region);
	mt_stats_slab stats = { y, partial, range };
	bc_parallel_for(0, MN_DIM - 1, MN_GRAIN, mt_stats_rows, &stats);
	mt_norm_quant(y, codes, partial, range, plan, 0, qs, sme);
	for (int layer = 1; layer < plan->layer_count; layer++) {
#if MN_SME
		if (sme) mt_prepare_sme(plan->kernel[layer], wq);
		else
#endif
			mt_prepare(plan->kernel[layer], qs, wq, tap_sum, scale);
		mt_conv_slab conv = { codes, y, wq, tap_sum, scale, block2, block1, plan->dilation[layer],
			partial, range, sme };
		bc_parallel_for(0, MN_DIM - 1, MN_GRAIN, mt_conv_rows, &conv);
		if (layer + 1 < plan->layer_count)
			mt_norm_quant(y, codes, partial, range, plan, layer, qs, sme);
	}
	const int last = plan->layer_count - 1;
#if BC_MODEL_HAS_AFFINE
	mn_norm(y, partial, plan->scale[last], plan->bias[last], &region);
#else
	(void)last;
	mn_norm(y, partial, &region);
#endif
	mn_classify(y, labels, plan->classifier_weights,
#if BC_MODEL_CLASSIFIER_BIAS
		plan->classifier_bias,
#endif
		&region);
	free(y); free(codes); free(partial); free(range); free(wq);
	return 0;
}

#endif
