/* SME and NEON must match every fp32 value, including normalization and segment tails. */
#include "../src/meshnet_cpu.c"
#include <stdio.h>
#if MN_SME
static unsigned seed = 3;
static float random_value(void) {
	seed = seed * 1664525u + 1013904223u;
	return ((int)(seed >> 9) - 4194304) * 0x1p-21f;
}
int main(void) {
	if (!mn_sme_available("hw.optional.arm.SME_F32F32")) {
		puts("cpu_sme_check: SKIP (requires 512-bit Apple SME fp32)");
		return 0;
	}
	const size_t slice = (size_t)MN_DIM * MN_DIM;
	float *a = calloc(MN_NVOX * MN_CS, sizeof(float)), *b = calloc(MN_NVOX * MN_CS, sizeof(float));
	float *got = malloc(MN_NVOX * MN_CS * sizeof(float)), *want = malloc(MN_NVOX * MN_CS * sizeof(float));
	if (!a || !b || !got || !want) {
		free(a);
		free(b);
		free(got);
		free(want);
		return 1;
	}
	float mean[MN_CS], inv[MN_CS], scale[MN_CS], bias[MN_CS];
	mn_region region;
	mn_make_region(NULL, &region);
	for (int c = 0; c < MN_CS; c++) {
		mean[c] = random_value();
		inv[c] = random_value();
		scale[c] = random_value();
		bias[c] = random_value();
	}
	for (size_t i = 0; i < slice * MN_CS; i++)
		a[i] = b[i] = random_value() * 4;
	mn_norm_slab na = {.buffer = a,
					   .mean = mean,
					   .invstd = inv,
#if BC_MODEL_HAS_AFFINE
					   .scale = scale,
					   .bias = bias,
#endif
					   .region = &region};
	mn_norm_slab nb = na;
	nb.buffer = b;
	mn_norm_rows(&na, 0, 0);
	mn_sme_norm_rows(&nb, 0, 0);
	size_t norm_bad = 0;
	for (int y = 0; y < MN_DIM; y++)
		for (int x = 0; x < MN_DIM; x++)
			for (int c = 0; c < MN_CS; c++)
				norm_bad += memcmp(a + ((size_t)y * MN_DIM + x) * MN_CS + c,
								   b + (size_t)y * MN_DIM * MN_CS + c * MN_DIM + x, 4) != 0;
	printf("cpu_sme_check: norm mismatches %zu\n", norm_bad);
	static float weights[27 * BC_MODEL_CHAN * MN_CS];
	for (size_t i = 0; i < sizeof weights / sizeof *weights; i++)
		weights[i] = random_value() * 0.1f;
	const int low[] = {0, 123, 224}, high[] = {34, 133, 255};
	for (int k = 0; k < 3; k++)
		for (int z = low[k]; z <= high[k]; z++)
			for (int y = 0; y < MN_DIM; y++)
				for (int x = 0; x < MN_DIM; x++)
					for (int c = 0; c < MN_CS; c++) {
						float v = random_value();
						if ((seed & 31) == 0)
							v *= 0x1p-130f;
						size_t row = (size_t)z * MN_DIM + y;
						a[(row * MN_DIM + x) * MN_CS + c] = b[row * MN_DIM * MN_CS + c * MN_DIM + x] = v;
					}
	const struct {
		int d, z;
	} cases[] = {{31, 0}, {31, 255}, {19, 3}, {5, 128}, {1, 0}, {1, 255}};
	size_t all_bad = 0;
	for (unsigned k = 0; k < sizeof cases / sizeof *cases; k++) {
		mn_conv_hidden_slab baseline = {a, want, weights, cases[k].d, &region},
							sme = {b, got, weights, cases[k].d, &region};
		mn_conv_hidden_rows(&baseline, cases[k].z, cases[k].z);
		mn_sme_conv_rows(&sme, cases[k].z, cases[k].z);
		size_t bad = 0;
		double maxerr = 0;
		for (size_t i = cases[k].z * slice * MN_CS; i < (cases[k].z + 1) * slice * MN_CS; i++) {
			bad += memcmp(got + i, want + i, 4) != 0;
			maxerr = fmax(maxerr, fabs((double)got[i] - want[i]));
		}
		printf("cpu_sme_check: conv d%d z%d mismatches %zu maxerr %.9g\n", cases[k].d, cases[k].z, bad, maxerr);
		all_bad += bad;
	}
	free(a);
	free(b);
	free(got);
	free(want);
	return norm_bad || all_bad;
}

#else
int main(void) {
	puts("cpu_sme_check: SKIP (SME not built)");
	return 0;
}
#endif
