/* Benchmark-only normalized-input injection. Production builds continue to use
   the histogram normalizer; both benchmark runners receive identical f32 bytes. */
#define bc_accel_normalize_input bc_benchmark_normalized_input
#include "../src/backend_webgpu.c"

float *bc_benchmark_normalized_input(const unsigned char *unused) {
	(void)unused;
	float *data = malloc(BC_NVOX * sizeof(float));
	FILE *file = fopen("/input.f32", "rb");
	if (!data || !file || fread(data, sizeof(float), BC_NVOX, file) != BC_NVOX) {
		free(data); if (file) fclose(file); return NULL;
	}
	fclose(file); return data;
}
