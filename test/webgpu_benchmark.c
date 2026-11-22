#define main brainchopc_cli_main
#include "../src/brainchopc.c"
#undef main

int main(int argc, char **argv) {
	int repeats = argc > 1 ? atoi(argv[1]) : 3;
	if (repeats < 1 || repeats > 100) return 2;
	float *weights = load_model_weights();
	size_t n = (size_t)256*256*256;
	float *labels = malloc(n*sizeof(float));
	unsigned char unused = 0;
	bc_inference_options options = {0};
	if (!weights || !labels) return 1;
	for (int run=0; run<repeats; run++) {
		double start=bc_now_ms();
		if (bc_webgpu_backend.segment(&unused, labels, weights, &options)) return 1;
		printf("BENCH %d %.6f\n", run, bc_now_ms()-start);
		FILE *out=fopen("/labels.f32","wb");
		if (!out || fwrite(labels,sizeof(float),n,out)!=n || fclose(out)) return 1;
	}
	free(labels); free(weights); return 0;
}
