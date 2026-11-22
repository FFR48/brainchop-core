/* Link the real reslicer and CLI helpers without adding a test-only production
   entry point. The normal CLI entry is renamed within this test translation unit. */
#define main brainchopc_cli_main
#include "../src/brainchopc.c"
#undef main

int main(int argc, char **argv) {
	if (argc == 5 && !strcmp(argv[1], "--nifti")) {
		nifti_image *source = nifti_image_read(argv[2], 1);
		nifti_image *target = nifti_image_read(argv[3], 0);
		if (!source || !target || brainchopc_convert(source, DT_FLOAT32) ||
				brainchopc_reslice(source, target, BC_RESAMPLE_LABELS) ||
				brainchopc_convert(source, DT_UINT8) ||
				nifti_set_filenames(source, argv[4], 0, 1)) return 1;
		int failed = nifti_image_write_status(source);
		nifti_image_free(source); nifti_image_free(target);
		return failed;
	}
	if (argc != 3) return 2;
	FILE *f = fopen(argv[1], "rb");
	uint32_t dims[6]; double matrices[32];
	if (!f || fread(dims, sizeof(*dims), 6, f) != 6 ||
			fread(matrices, sizeof(double), 32, f) != 32) return 1;
	nifti_image *source = calloc(1, sizeof(*source)), *target = calloc(1, sizeof(*target));
	if (!source || !target) return 1;
	source->nx = dims[0]; source->ny = dims[1]; source->nz = dims[2];
	target->nx = dims[3]; target->ny = dims[4]; target->nz = dims[5];
	source->nvox = (size_t)dims[0] * dims[1] * dims[2];
	source->datatype = DT_FLOAT32; source->nbyper = sizeof(float);
	source->data = nii_malloc(source->nvox, sizeof(float));
	if (!source->data || fread(source->data, sizeof(float), source->nvox, f) != (size_t)source->nvox) return 1;
	fclose(f);
	for (int i = 0; i < 16; i++) {
		source->sto_xyz.m[i/4][i%4] = matrices[i];
		target->sto_xyz.m[i/4][i%4] = matrices[16+i];
	}
	int status = brainchopc_reslice(source, target, BC_RESAMPLE_LABELS);
	if (!status) {
		f = fopen(argv[2], "wb");
		if (!f || fwrite(source->data, sizeof(float), source->nvox, f) != (size_t)source->nvox || fclose(f)) status = 1;
	}
	nifti_image_free(source); nifti_image_free(target);
	return status;
}
