/*
 * Drive bc_cat_lite_fit on a dump written by BRAINCHOPC_PVE_DUMP, and write the
 * three tissue fractions as raw f32.
 *
 * Its counterpart, test/cat_lite_js_check.mjs, runs the upstream cat-lite.js on
 * the SAME dump. Comparing the two outputs tests the port and nothing else: the
 * network, the normalization and the fp16 path are all upstream of the dump and
 * identical for both. An end-to-end comparison would fold porting error and
 * network drift into one number with no way to tell them apart.
 *
 * usage: cat_lite_check DUMP OUT
 */

#include "cat_lite.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if !BC_HAS_PVE
int main(void) {
	fprintf(stderr, "cat_lite_check: this model has no PVE mode\n");
	return 77;
}
#else

#define CHECK_DIM 256

int main(int argc, char **argv) {
	if (argc != 3 && argc != 6) {
		fprintf(stderr, "usage: %s DUMP OUT [NX NY NZ]\n", argv[0]);
		return 2;
	}
	size_t dim[3] = { CHECK_DIM, CHECK_DIM, CHECK_DIM };
	if (argc == 6) for (int a=0; a<3; a++) {
		char *end; unsigned long v=strtoul(argv[a+3], &end, 10);
		if (*end || !v || v>256) return 2;
		dim[a]=v;
	}
	const size_t count = dim[0] * dim[1] * dim[2];
	float *planes = (float *)malloc(count * BC_PVE_PLANES * sizeof(*planes));
	float *intensity = (float *)malloc(count * sizeof(*intensity));
	if (!planes || !intensity) {
		fprintf(stderr, "cat_lite_check: out of memory\n");
		free(planes); free(intensity);
		return 1;
	}
	FILE *handle = fopen(argv[1], "rb");
	if (!handle || fread(planes, sizeof(float), count * BC_PVE_PLANES, handle)
				!= count * BC_PVE_PLANES ||
			fread(intensity, sizeof(float), count, handle) != count) {
		fprintf(stderr, "cat_lite_check: cannot read %s\n", argv[1]);
		if (handle) fclose(handle);
		free(planes); free(intensity);
		return 1;
	}
	fclose(handle);

	struct timespec start, end;
	timespec_get(&start, TIME_UTC);
	if (bc_cat_lite_fit(planes, intensity, dim)) {
		fprintf(stderr, "cat_lite_check: fit failed\n");
		free(planes); free(intensity);
		return 1;
	}
	timespec_get(&end, TIME_UTC);
	fprintf(stderr, "CAT_FIT_MS %.6f\n", (end.tv_sec-start.tv_sec)*1000.0 +
		(end.tv_nsec-start.tv_nsec)/1e6);

	handle = fopen(argv[2], "wb");
	/* GM, WM, CSF, matching the JS harness's write order. */
	if (!handle || fwrite(planes, sizeof(float), count * 3, handle) != count * 3) {
		fprintf(stderr, "cat_lite_check: cannot write %s\n", argv[2]);
		if (handle) fclose(handle);
		free(planes); free(intensity);
		return 1;
	}
	fclose(handle);
	free(planes); free(intensity);
	return 0;
}
#endif
