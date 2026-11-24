/* Unit tests for MindGrab native integer encoding preservation, NIfTI scaling
 * resolution (including scl_slope == 0), direct float32 decode, and 64-bit
 * integer background selection. */
#define main brainchopc_cli_main
#define bc_cpu_backend bc_test_cpu_backend
#include "../src/brainchopc.c"
#undef bc_cpu_backend
#undef main
#include <assert.h>

static int check_bright_center;

static int mock_mindgrab_segment(const unsigned char *input, float *out,
		const float *weights, const bc_inference_options *options) {
	(void)weights; (void)options;
	/* When enabled, assert that conform received bright center (>128) and dark
	 * corner (<32) -- verifying negative-slope decoding did not invert contrast. */
	if (check_bright_center) {
		unsigned char corner = input[0];
		unsigned char center = input[128 + 128 * 256 + 128 * 65536];
		assert(center > 128 && corner < 32);
	}
	memset(out, 0, 256u * 256u * 256u * sizeof(*out));
	for (size_t z = 112; z < 144; z++)
		for (size_t y = 112; y < 144; y++)
			for (size_t x = 112; x < 144; x++)
				out[x + y * 256 + z * 65536] = 1.0f;
	return 0;
}

const bc_backend bc_test_cpu_backend = {.name = "test-cpu", .segment = mock_mindgrab_segment};

static nifti_image *make_test_image(int nx, int ny, int nz, int datatype,
		double slope, double inter) {
	int nbyper = 0;
	nifti_datatype_sizes(datatype, &nbyper, NULL);
	assert(nbyper > 0);
	nifti_image *nim = calloc(1, sizeof(*nim));
	assert(nim);
	nim->ndim = nim->dim[0] = 3;
	nim->nx = nim->dim[1] = nx;
	nim->ny = nim->dim[2] = ny;
	nim->nz = nim->dim[3] = nz;
	nim->nvox = (int64_t)nx * ny * nz;
	nim->dx = nim->pixdim[1] = 10.0;
	nim->dy = nim->pixdim[2] = 10.0;
	nim->dz = nim->pixdim[3] = 10.0;
	nim->datatype = datatype;
	nim->nbyper = nbyper;
	nim->scl_slope = slope;
	nim->scl_inter = inter;
	nim->qform_code = nim->sform_code = NIFTI_XFORM_SCANNER_ANAT;
	memset(&nim->sto_xyz, 0, sizeof(nim->sto_xyz));
	nim->sto_xyz.m[0][0] = 10.0;
	nim->sto_xyz.m[1][1] = 10.0;
	nim->sto_xyz.m[2][2] = 10.0;
	nim->sto_xyz.m[3][3] = 1.0;
	nim->qto_xyz = nim->sto_xyz;
	nim->data = calloc((size_t)nim->nvox, (size_t)nbyper);
	assert(nim->data);
	return nim;
}

static void test_datatype_classification(void) {
	nifti_image img = {0};
	const int ints[] = {
		DT_UINT8, DT_INT8, DT_UINT16, DT_INT16,
		DT_UINT32, DT_INT32, DT_UINT64, DT_INT64
	};
	for (size_t i = 0; i < sizeof(ints) / sizeof(*ints); i++) {
		img.datatype = ints[i];
		assert(integer_input(&img) == 1);
	}
	const int non_ints[] = { DT_FLOAT32, DT_FLOAT64, DT_COMPLEX64, DT_RGB24 };
	for (size_t i = 0; i < sizeof(non_ints) / sizeof(*non_ints); i++) {
		img.datatype = non_ints[i];
		assert(integer_input(&img) == 0);
	}
}

static void test_zero_slope_ignores_intercept(void) {
	nifti_image *nim = make_test_image(4, 1, 1, DT_INT16, 0.0, 500.0);
	int16_t *raw = nim->data;
	raw[0] = 0; raw[1] = 10; raw[2] = 20; raw[3] = 30;

	/* Direct float copy must ignore scl_inter when scl_slope == 0. */
	nifti_image *fcopy = private_float_copy(nim);
	assert(fcopy && fcopy->datatype == DT_FLOAT32);
	assert(fcopy->scl_slope == 1.0 && fcopy->scl_inter == 0.0);
	const float *fv = fcopy->data;
	assert(fv[0] == 0.0f && fv[1] == 10.0f && fv[2] == 20.0f && fv[3] == 30.0f);
	/* Original must remain completely untouched. */
	assert(nim->datatype == DT_INT16 && nim->scl_slope == 0.0 && nim->scl_inter == 500.0);
	nifti_image_free(fcopy);

	/* mask_integer_input must use ascending min and normalize header to (1, 0). */
	const float mask[4] = {0.0f, 1.0f, 0.0f, 1.0f};
	mask_integer_input(nim, mask);
	assert(nim->scl_slope == 1.0 && nim->scl_inter == 0.0);
	assert(raw[0] == 0 && raw[1] == 10 && raw[2] == 0 && raw[3] == 30);

	/* brainchopc_convert must also ignore scl_inter when scl_slope == 0. */
	nim->scl_slope = 0.0;
	nim->scl_inter = 500.0;
	assert(!brainchopc_convert(nim, DT_FLOAT32));
	const float *cv = nim->data;
	assert(cv[0] == 0.0f && cv[1] == 10.0f && cv[2] == 0.0f && cv[3] == 30.0f);
	assert(nim->scl_slope == 1.0 && nim->scl_inter == 0.0);
	nifti_image_free(nim);

	/* Non-finite slope also disables scaling; non-finite intercept with valid slope zeros intercept. */
	double s = 0.0, b = 0.0;
	nifti_image hdr = {0};
	hdr.scl_slope = NAN; hdr.scl_inter = 123.0;
	resolve_nifti_scaling(&hdr, &s, &b);
	assert(s == 1.0 && b == 0.0);
	hdr.scl_slope = INFINITY; hdr.scl_inter = 123.0;
	resolve_nifti_scaling(&hdr, &s, &b);
	assert(s == 1.0 && b == 0.0);
	hdr.scl_slope = -2.5; hdr.scl_inter = NAN;
	resolve_nifti_scaling(&hdr, &s, &b);
	assert(s == -2.5 && b == 0.0);
}

static void test_mask_all_integer_types(void) {
	const float mask[4] = {0.0f, 0.0f, 1.0f, 1.0f};

	/* uint8 positive & negative slope */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_UINT8, 4.0, -10.0);
		uint8_t *v = nim->data;
		v[0] = 200; v[1] = 10; v[2] = 150; v[3] = 250;
		mask_integer_input(nim, mask);
		assert(v[0] == 10 && v[1] == 10 && v[2] == 150 && v[3] == 250);
		assert(nim->scl_slope == 4.0 && nim->scl_inter == -10.0);
		nifti_image_free(nim);
	}
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_UINT8, -4.0, 1000.0);
		uint8_t *v = nim->data;
		v[0] = 200; v[1] = 10; v[2] = 150; v[3] = 250;
		mask_integer_input(nim, mask);
		assert(v[0] == 250 && v[1] == 250 && v[2] == 150 && v[3] == 250);
		assert(nim->scl_slope == -4.0 && nim->scl_inter == 1000.0);
		nifti_image_free(nim);
	}

	/* int8 negative slope */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_INT8, -2.0, 256.0);
		int8_t *v = nim->data;
		v[0] = -100; v[1] = 127; v[2] = -50; v[3] = 0;
		mask_integer_input(nim, mask);
		assert(v[0] == 127 && v[1] == 127 && v[2] == -50 && v[3] == 0);
		nifti_image_free(nim);
	}

	/* uint16 intercept-only (CT-style) */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_UINT16, 1.0, -1024.0);
		uint16_t *v = nim->data;
		v[0] = 500; v[1] = 0; v[2] = 924; v[3] = 1064;
		mask_integer_input(nim, mask);
		assert(v[0] == 0 && v[1] == 0 && v[2] == 924 && v[3] == 1064);
		assert(nim->scl_slope == 1.0 && nim->scl_inter == -1024.0);
		nifti_image_free(nim);
	}

	/* int16 fractional slope (would quantize to 0/1 through float->int16) */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_INT16, 0.0005, 0.0);
		int16_t *v = nim->data;
		v[0] = 100; v[1] = -32768; v[2] = 500; v[3] = 1200;
		mask_integer_input(nim, mask);
		assert(v[0] == -32768 && v[1] == -32768 && v[2] == 500 && v[3] == 1200);
		assert(nim->scl_slope == 0.0005 && nim->scl_inter == 0.0);
		nifti_image_free(nim);
	}

	/* int32 unscaled > 2^24 (must preserve exact low bits) */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_INT32, 1.0, 0.0);
		int32_t *v = nim->data;
		v[0] = 100000005; v[1] = 100000001; v[2] = 100000003; v[3] = 100000007;
		mask_integer_input(nim, mask);
		assert(v[0] == 100000001 && v[1] == 100000001 && v[2] == 100000003 && v[3] == 100000007);
		nifti_image_free(nim);
	}

	/* uint32 negative slope */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_UINT32, -0.5, 4000000000.0);
		uint32_t *v = nim->data;
		v[0] = 3000000001u; v[1] = 3000000003u; v[2] = 2000000001u; v[3] = 1000000001u;
		mask_integer_input(nim, mask);
		assert(v[0] == 3000000003u && v[1] == 3000000003u &&
		       v[2] == 2000000001u && v[3] == 1000000001u);
		nifti_image_free(nim);
	}

	/* int64 adjacent extremes that tie when converted to double */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_INT64, 1.0, 0.0);
		int64_t *v = nim->data;
		v[0] = INT64_MIN + 1; v[1] = INT64_MIN; v[2] = (1LL << 60) + 3; v[3] = INT64_MAX - 1;
		mask_integer_input(nim, mask);
		assert(v[0] == INT64_MIN && v[1] == INT64_MIN &&
		       v[2] == (1LL << 60) + 3 && v[3] == INT64_MAX - 1);
		nifti_image_free(nim);
	}
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_INT64, -1e-12, 1e7);
		int64_t *v = nim->data;
		v[0] = INT64_MAX - 1; v[1] = INT64_MAX; v[2] = 123456789012345LL; v[3] = -42LL;
		mask_integer_input(nim, mask);
		assert(v[0] == INT64_MAX && v[1] == INT64_MAX &&
		       v[2] == 123456789012345LL && v[3] == -42LL);
		nifti_image_free(nim);
	}

	/* uint64 adjacent extremes with positive and negative slopes */
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_UINT64, 1.0 / (double)(1ULL << 45), -32768.0);
		uint64_t *v = nim->data;
		uint64_t base = 1ULL << 60;
		v[0] = base + 1; v[1] = base; v[2] = base + 7; v[3] = UINT64_MAX - 1;
		mask_integer_input(nim, mask);
		assert(v[0] == base && v[1] == base && v[2] == base + 7 && v[3] == UINT64_MAX - 1);
		nifti_image_free(nim);
	}
	{
		nifti_image *nim = make_test_image(4, 1, 1, DT_UINT64, -1.0 / (double)(1ULL << 45), 32768.0);
		uint64_t *v = nim->data;
		v[0] = UINT64_MAX - 1; v[1] = UINT64_MAX; v[2] = (1ULL << 60) + 9; v[3] = 12345ULL;
		mask_integer_input(nim, mask);
		assert(v[0] == UINT64_MAX && v[1] == UINT64_MAX &&
		       v[2] == (1ULL << 60) + 9 && v[3] == 12345ULL);
		nifti_image_free(nim);
	}
}

static void test_run_model_mindgrab_paths(void) {
	bc_cli_options cli = {0};
	cli.adapter = -1;
	float weights = 0.0f;

	/* 1. Negative-slope int16: decoded center is bright, corner is dark. */
	{
		nifti_image *nim = make_test_image(8, 8, 8, DT_INT16, -512.0, 512.0 * 32767.0);
		int16_t *v = nim->data;
		for (int z = 0; z < 8; z++)
			for (int y = 0; y < 8; y++)
				for (int x = 0; x < 8; x++) {
					int in_center = x >= 2 && x <= 5 && y >= 2 && y <= 5 && z >= 2 && z <= 5;
					v[x + y * 8 + z * 64] = (int16_t)(in_center ? 32767 - 1000 : 32767);
				}
		nifti_image *out = NULL;
		check_bright_center = 1;
		assert(!run_model(nim, DT_INT16, 1, &weights, &bc_test_cpu_backend, 0, &cli, &out));
		check_bright_center = 0;
		assert(out == nim);
		assert(out->datatype == DT_INT16 && out->scl_slope == -512.0 && out->scl_inter == 512.0 * 32767.0);
		/* Corner must be masked to max stored value (32767); center kept intact (31767). */
		assert(v[0] == 32767);
		assert(v[4 + 4 * 8 + 4 * 64] == 31767);
		nifti_image_free(out);
	}

	/* 2. Unscaled int32 > 2^24: exact low bits must survive MindGrab without float32 rounding. */
	{
		nifti_image *nim = make_test_image(8, 8, 8, DT_INT32, 1.0, 0.0);
		int32_t *v = nim->data;
		for (int z = 0; z < 8; z++)
			for (int y = 0; y < 8; y++)
				for (int x = 0; x < 8; x++) {
					int in_center = x >= 2 && x <= 5 && y >= 2 && y <= 5 && z >= 2 && z <= 5;
					v[x + y * 8 + z * 64] = in_center ? 100000001 : 0;
				}
		nifti_image *out = NULL;
		assert(!run_model(nim, DT_INT32, 1, &weights, &bc_test_cpu_backend, 0, &cli, &out));
		assert(out == nim && out->datatype == DT_INT32);
		assert(v[0] == 0);
		assert(v[4 + 4 * 8 + 4 * 64] == 100000001);
		nifti_image_free(out);
	}

	/* 3. Float32 with NaN: still takes the sanitized float path (preserve_encoding == 0). */
	{
		nifti_image *nim = make_test_image(8, 8, 8, DT_FLOAT32, 2.0, 10.0);
		float *v = nim->data;
		for (size_t i = 0; i < (size_t)nim->nvox; i++) v[i] = 1.0f;
		v[0] = NAN;
		v[4 + 4 * 8 + 4 * 64] = 100.0f;
		assert(!brainchopc_convert(nim, DT_FLOAT32));
		assert(!sanitize_input(nim, DT_FLOAT32));
		nifti_image *out = NULL;
		assert(!run_model(nim, DT_FLOAT32, 1, &weights, &bc_test_cpu_backend, 0, &cli, &out));
		assert(out == nim && out->datatype == DT_FLOAT32);
		assert(out->scl_slope == 1.0 && out->scl_inter == 0.0);
		float *ov = out->data;
		assert(isfinite(ov[0]) && ov[0] == 0.0f);
		assert(ov[4 + 4 * 8 + 4 * 64] == 210.0f);
		nifti_image_free(out);
	}
}

int main(void) {
	test_datatype_classification();
	test_zero_slope_ignores_intercept();
	test_mask_all_integer_types();
	test_run_model_mindgrab_paths();
	puts("mindgrab encoding: all integer types, scaling, 64-bit extremes and zero-slope rules PASS");
	return 0;
}
