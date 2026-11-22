/* CPU turbo: (1) integer kernels equal the scalar reference bit for bit
   (integer accumulation, same float combine); (2) the row driver, taps,
   zero-point correction and padding match the exact convolution on the
   dequantized codes, at border slices; (3) SME fp16 matches its fp32 oracle. Build: make test-cpu-turbo. */
#define BC_CPU_TURBO_CHECK 1
#include "../src/meshnet_cpu.c"
#include <stdio.h>

int main(void) {
	mt_block_fn block2, block1;
	if (!mt_select(&block2, &block1)) {
		printf("cpu_turbo_check: SKIP, this CPU has no i8mm/VNNI kernel\n");
		return 0;
	}
	enum { VOX = 4096 };
	static uint8_t src[VOX * MT_AS + MT_ACT_PAD];
	static float kernel[27 * BC_MODEL_CHAN * MN_CS];
	static int8_t wq[27 * MT_TAP];
	int32_t tap_sum[27 * 2 * MN_CS];
	float scale[2 * MN_CS], qs[MN_CS], fast[2 * MN_CS], ref[2 * MN_CS];
	unsigned seed = 12345;
#define RAND() (seed = seed * 1103515245u + 12345u, seed >> 8)
	int mismatches = 0, sme_bad = 0;
	for (int trial = 0; trial < 200; trial++) {
		for (size_t i = 0; i < sizeof src; i++)
			src[i] = (uint8_t)(RAND() & (i % MT_AS < MN_CS ? 255 : 127));
		for (size_t i = 0; i < sizeof kernel / sizeof *kernel; i++)
			kernel[i] = ((float)(RAND() % 20001) - 10000.0f) * 1e-5f;
		for (int c = 0; c < MN_CS; c++) qs[c] = 0.01f + (float)(RAND() % 1000) * 1e-4f;
		mt_prepare(kernel, qs, wq, tap_sum, scale);
		ptrdiff_t tap_voxel[27];
		const int8_t *tap_w[27];
		int32_t corr[2 * MN_CS] = { 0 };
		int ntap = 1 + (int)(RAND() % 27);
		for (int t = 0; t < ntap; t++) {
			int tap = (int)(RAND() % 27);
			tap_voxel[t] = (ptrdiff_t)(RAND() % (VOX / 2));
			tap_w[t] = wq + (size_t)tap * MT_TAP;
			for (int k = 0; k < 2 * MN_CS; k++) corr[k] += tap_sum[tap * 2 * MN_CS + k];
		}
		size_t x = RAND() % (VOX / 2 - 2);
		block2(src, x, tap_voxel, tap_w, ntap, corr, scale, fast);
		mt_scalar_block2(src, x, tap_voxel, tap_w, ntap, corr, scale, ref);
		mismatches += memcmp(fast, ref, sizeof fast) != 0;
		block1(src, x, tap_voxel, tap_w, ntap, corr, scale, fast);
		mt_scalar_block1(src, x, tap_voxel, tap_w, ntap, corr, scale, ref);
		mismatches += memcmp(fast, ref, MN_CS * sizeof *fast) != 0;
	}
	printf("cpu_turbo_check: %s (%d mismatching blocks of 400)\n", mismatches ? "FAIL" : "ok", mismatches);

	/* (2) Allocations are virtual; only the slabs below are touched. */
	const size_t slice = (size_t)MN_DIM * MN_DIM;
	uint8_t *codes = calloc(MN_NVOX * MT_AS + MT_ACT_PAD, 1);
	float *exact_in = calloc(MN_NVOX * MN_CS, sizeof(float));
	float *got = malloc(MN_NVOX * MN_CS * sizeof(float));
	float *want = malloc(MN_NVOX * MN_CS * sizeof(float));
	double *partial = malloc(MN_DIM * 2 * MN_CS * sizeof(double));
	float *range = malloc(MN_DIM * 2 * MN_CS * sizeof(float));
	if (!codes || !exact_in || !got || !want || !partial || !range) return 1;
	float mul[MN_CS], add[MN_CS], inv_qs[MN_CS];
	for (int c = 0; c < MN_CS; c++) {
		mul[c] = 1.0f; add[c] = 0.0f;
		qs[c] = 0.025f + 0.01f * c; inv_qs[c] = 1.0f / qs[c];
	}
	for (size_t i = 0; i < slice * MN_CS; i++)
		got[i] = ((int)(i % 12001) - 6000) * 0.001f;
	mt_quant_slab quant = { got, codes, mul, add, inv_qs };
	mt_quant_rows(&quant, 0, 0);
	int quant_bad = 0;
	for (size_t v = 0; v < slice; v++) for (int c = 0; c < MN_CS; c++) {
		unsigned hi = codes[v * MT_AS + c], lo = codes[v * MT_AS + MN_CS + c];
		float decoded = ((float)hi - MT_ZP + ((float)lo - MT_RES_ZP) / 128.0f) * qs[c];
		quant_bad += lo > 127 || fabsf(decoded - mn_gelu(got[v * MN_CS + c])) > qs[c] / 256.0f + 2e-6f;
	}
	printf("cpu_turbo_check: residual quantization %s\n", quant_bad ? "FAIL" : "ok");
#if MN_SME
	if (mt_sme_available()) {
		for (int c = 0; c < MN_CS; c++) { mul[c] = 0.25f + 0.1f * c; add[c] = -0.3f * c; }
		mt_quant_sme_rows(&quant, 0, 0);
		const float16_t *half = (const float16_t *)codes;
		for (int y = 0; y < MN_DIM; y++) for (int x = 0; x < MN_DIM; x++)
			for (int c = 0; c < MN_CS; c++) {
				size_t v = (size_t)y * MN_DIM + x;
				float expected = mn_gelu(got[v * MN_CS + c] * mul[c] + add[c]);
				float actual = half[(size_t)y * MN_DIM * MN_CS + (c / 2) * MN_DIM * 2 + x * 2 + c % 2];
				/* Half rounding plus scalar/SIMD GELU rounding near zero. */
				sme_bad += fabsf(actual - expected) > 0.0005f * fabsf(expected) + 2e-6f;
			}
		printf("cpu_turbo_check: SME quantization %s\n", sme_bad ? "FAIL" : "ok");
	}
#endif
	for (int c = 0; c < MN_CS; c++) qs[c] = c < BC_MODEL_CHAN ? qs[c] : 1.0f;
	mt_prepare(kernel, qs, wq, tap_sum, scale);
	const int slab_low[] = { 0, 123, 224 }, slab_high[] = { 34, 133, 255 };
	for (int k = 0; k < 3; k++)
		for (size_t v = slab_low[k] * slice; v < (slab_high[k] + 1) * slice; v++)
			for (int c = 0; c < MN_CS; c++) {
				uint8_t q = c < BC_MODEL_CHAN ? (uint8_t)RAND() : MT_ZP;
				uint8_t r = c < BC_MODEL_CHAN ? (uint8_t)(RAND() & 127) : MT_RES_ZP;
				codes[v * MT_AS + c] = q;
				codes[v * MT_AS + MN_CS + c] = r;
				exact_in[v * MN_CS + c] = ((float)q - MT_ZP + ((float)r - MT_RES_ZP) / 128.0f) * qs[c];
			}
	mn_region region;
	mn_make_region(NULL, &region);
	const struct { int d, z; } cases[] = { {31, 0}, {31, 255}, {19, 3}, {5, 128}, {1, 0}, {1, 255} };
	int bad = 0;
	for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
		mt_conv_slab conv = { codes, got, wq, tap_sum, scale, block2, block1, cases[k].d, partial, range, 0 };
		mt_conv_rows(&conv, cases[k].z, cases[k].z);
		mn_conv_hidden_slab exact = { exact_in, want, kernel, cases[k].d, &region };
		mn_conv_hidden_rows(&exact, cases[k].z, cases[k].z);
		double err = 0, peak = 0;
		for (size_t i = cases[k].z * slice * MN_CS; i < (cases[k].z + 1) * slice * MN_CS; i++) {
			err = fmax(err, fabs((double)got[i] - want[i]));
			peak = fmax(peak, fabs((double)want[i]));
		}
		/* Weight rounding and the omitted residual product remain below 0.1%. */
		bad += !(err <= 1e-3 * peak);
		printf("cpu_turbo_check: conv d=%2d z=%3d relative error %.2g\n", cases[k].d, cases[k].z, err / peak);
	}
	printf("cpu_turbo_check: conv %s\n", bad ? "FAIL" : "ok");
#if MN_SME
	if (mt_sme_available()) {
		float16_t *half = (float16_t *)codes;
		memset(wq, 0, sizeof wq);
		for (size_t i = 0; i < sizeof kernel / sizeof *kernel; i++) kernel[i] = (float16_t)kernel[i];
		mt_prepare_sme(kernel, wq);
		for (int k = 0; k < 3; k++)
			for (size_t row = slab_low[k] * MN_DIM; row < (slab_high[k] + 1) * MN_DIM; row++)
				for (int x = 0; x < MN_DIM; x++) for (int c = 0; c < MN_CS; c++) {
					float16_t value = c < BC_MODEL_CHAN ? (float16_t)(((int)(RAND() % 8001) - 4000) * 0.001f) : 0;
					half[row * MN_DIM * MN_CS + (c / 2) * MN_DIM * 2 + x * 2 + c % 2] = value;
					exact_in[(row * MN_DIM + x) * MN_CS + c] = value;
				}
		for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
			mt_conv_slab conv = { codes, got, wq, tap_sum, scale, block2, block1, cases[k].d, partial, range, 1 };
			mt_conv_rows(&conv, cases[k].z, cases[k].z);
			mn_conv_hidden_slab exact = { exact_in, want, kernel, cases[k].d, &region };
			mn_conv_hidden_rows(&exact, cases[k].z, cases[k].z);
			double err = 0, peak = 0;
			for (size_t i = cases[k].z * slice * MN_CS; i < (cases[k].z + 1) * slice * MN_CS; i++) {
				err = fmax(err, fabs((double)got[i] - want[i]));
				peak = fmax(peak, fabs((double)want[i]));
			}
			/* SME sums pairs of exact fp16 products before accumulating in fp32. */
			sme_bad += !(err <= 2e-5 * peak);
			printf("cpu_turbo_check: SME d=%2d z=%3d relative error %.2g\n", cases[k].d, cases[k].z, err / peak);
		}
		printf("cpu_turbo_check: SME %s\n", sme_bad ? "FAIL" : "ok");
	} else printf("cpu_turbo_check: SME skipped (requires 512-bit Apple SME fp16)\n");
#endif
	free(codes); free(exact_in); free(got); free(want); free(partial); free(range);
	return mismatches != 0 || quant_bad != 0 || bad != 0 || sme_bad != 0;
}
