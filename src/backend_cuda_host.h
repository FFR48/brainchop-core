#include "pve_constants.h"
/* Shared inference, packing, launch selection and autotuning policy.
 * Included once by the selected CUDA adapter (runtime or Driver API).
 * gpu_* primitives below are private to that translation unit. */
static int cuda_wmma_compiled(void) {
	int *device_flag = NULL, host_flag = 0;
	if (gpu_alloc((void **)&device_flag, sizeof(int)) != GPU_SUCCESS) {
		(void)gpu_last_error();
		return 0;
	}
	gpu_probe(device_flag);
	if (gpu_sync() != GPU_SUCCESS ||
			gpu_copy(&host_flag, device_flag, sizeof(int),
				GPU_DTOH) != GPU_SUCCESS) {
		(void)gpu_last_error();
		host_flag = 0;
	}
	gpu_free(device_flag);
	return host_flag == 1;
}

/*
 * Stage timing, gated on BRAINCHOPC_CUDA_PROFILE, mirroring the Metal backend's
 * BRAINCHOPC_METAL_PROFILE. Wall time is a poor guide here: before this work a
 * MindGrab run was ~3.3 s end to end but only ~2.3 s of that was on the GPU, so
 * a tile change worth 20% of kernel time showed up as 14% of what `time` prints.
 *
 * Stage timing synchronizes events, so the timed build serialises the layer
 * loop that the untimed one pipelines. Use it to compare stages, not to quote
 * an end-to-end figure.
 */
static int cuda_profiling(void) {
	static int state = -1;
	if (state < 0) state = getenv("BRAINCHOPC_CUDA_PROFILE") ? 1 : 0;
	return state;
}

static double g_stage_ms[3];
static int g_timing_error;
static char g_variant_label[64] = "vpt=? threads=?";
static const char *const g_stage_name[3] = { "conv", "moments", "norm+gelu" };

typedef struct { gpu_event start, stop; int active; } cuda_stage;

static void stage_begin(cuda_stage *stage) {
	stage->active = 0;
	if (!cuda_profiling()) return;
	if (gpu_event_create(&stage->start) != GPU_SUCCESS) { g_timing_error = 1; return; }
	if (gpu_event_create(&stage->stop) != GPU_SUCCESS) {
		gpu_event_destroy(stage->start);
		g_timing_error = 1;
		return;
	}
	stage->active = 1;
	if (gpu_event_record(stage->start) != GPU_SUCCESS) g_timing_error = 1;
}

static void stage_end(cuda_stage *stage, int index) {
	if (!stage->active) return;
	float ms = 0.0f;
	if (gpu_event_record(stage->stop) != GPU_SUCCESS ||
		gpu_event_sync(stage->stop) != GPU_SUCCESS ||
		gpu_event_elapsed(&ms, stage->start, stage->stop) != GPU_SUCCESS)
		g_timing_error = 1;
	else
		g_stage_ms[index] += ms;
	if (gpu_event_destroy(stage->start) != GPU_SUCCESS) g_timing_error = 1;
	if (gpu_event_destroy(stage->stop) != GPU_SUCCESS) g_timing_error = 1;
}

static void stage_report(void) {
	if (!cuda_profiling()) return;
	double total = g_stage_ms[0] + g_stage_ms[1] + g_stage_ms[2];
	fprintf(stderr, "brainchopc: CUDA stages (%s, reduce=%dx%d)\n",
		g_variant_label, CUDA_REDUCE_BLOCKS, CUDA_REDUCE_THREADS);
	for (int i = 0; i < 3; i++)
		fprintf(stderr, "brainchopc:   %-10s %8.1f ms  %5.1f%%\n",
			g_stage_name[i], g_stage_ms[i],
			total > 0.0 ? 100.0 * g_stage_ms[i] / total : 0.0);
	fprintf(stderr, "brainchopc:   %-10s %8.1f ms\n", "total", total);
}

/* ------------------------------------------------------------------ autotune
 *
 * The convolution's (VPT, THREADS) optimum is a property of the DEVICE, not of
 * the model: both shipped models want 3/512 on an RTX 4070 Ti SUPER, where the
 * previously hardcoded 8/128 costs 2.1-2.4x on the convolutions. The right value
 * elsewhere is unknown, and getting it wrong is not merely slow -- some pairs
 * exceed the register file and fail to launch outright. So the backend searches
 * once per device and remembers the answer.
 *
 * Template instantiation rather than NVRTC. NVRTC would allow an open search
 * space, but only by carrying the convolution a second time as a string
 * literal; this file already records what a duplicated kernel costs, in the
 * GELU that silently diverged across four backends. A fixed grid of eighteen
 * variants is a worse search and a much better invariant.
 */
typedef void (*cuda_conv_launcher)(const bc_act *, bc_act *, const bc_act *,
	int, int, int, int, size_t);

#define X(V, T) \
static void launch_conv_##V##_##T(const bc_act *source, bc_act *destination, \
        const bc_act *weights, int input_channels, int output_channels, \
        int input_stride, int dilation, size_t shared_bytes) { \
    gpu_conv(V, T, source, destination, weights, input_channels, \
        output_channels, input_stride, dilation, shared_bytes); \
}
BC_CUDA_SCALAR_VARIANTS(X)
#undef X

typedef struct {
	int vpt;
	int threads;
	cuda_conv_launcher launch;
} cuda_conv_variant;

/*
 * A tuning candidate: the two integers that get cached, and a launcher that
 * needs nothing but the layer's dilation.
 *
 * There are two families, and which one is searched depends on the device. The
 * scalar family varies (VPT, THREADS); the tensor-core family varies warps per
 * block and pins the second value at 0. Wrapping both behind one signature is
 * what lets a single search loop serve them -- the alternative was two
 * near-identical 130-line loops, which is how the GELU ended up with five
 * copies.
 *
 * This matters because the tensor-core path made the scalar search almost
 * pointless: on sm_70+ the scalar kernel runs LAYER 0 ONLY, so tuning it was
 * measuring a shape that runs once in 13-25 layers. Meanwhile CUDA_WMMA_WARPS
 * sat hardcoded at 8, and 16 is worth 1.27x on this card (MindGrab conv
 * 161 -> 127 ms). The tuner now searches whichever kernel will actually run.
 */
typedef void (*cuda_tune_run)(const bc_act *, bc_act *, const bc_act *, int);

typedef struct {
	int first;              /* VPT, or warps */
	int second;             /* THREADS, or 0 for the tensor-core family */
	cuda_tune_run run;
} cuda_tune_candidate;

/* Tuning always exercises a hidden-layer shape: CUDA_CS in, CUDA_CS out, and
   the CUDA_CS-wide source stride. */
#define X(V, T) \
static void tune_run_scalar_##V##_##T(const bc_act *source, bc_act *destination, \
        const bc_act *weights, int dilation) { \
    launch_conv_##V##_##T(source, destination, weights, CUDA_CS, CUDA_CS, \
        CUDA_CS, dilation, (size_t)27 * CUDA_CS * CUDA_CS * sizeof(bc_act)); \
}
BC_CUDA_SCALAR_VARIANTS(X)
#undef X

static const cuda_tune_candidate cuda_scalar_candidates[] = {
#define X(V, T) { V, T, tune_run_scalar_##V##_##T },
	BC_CUDA_SCALAR_VARIANTS(X)
#undef X
};
#define CUDA_SCALAR_CANDIDATE_COUNT \
	((int)(sizeof(cuda_scalar_candidates) / sizeof(cuda_scalar_candidates[0])))

/*
 * The tensor-core family. Warps per block is the only free parameter: the tile
 * is fixed at 16 voxels by the fragment shape, so a block simply covers WARPS
 * of them. Shared memory is 27*CS*CS halves for the staged weights plus one
 * 1 KiB scratch slot per warp, which is what caps the list at 32 (46592 B,
 * inside the 48 KiB default). 32 also needs 1024 threads, CUDA's per-block
 * maximum.
 *
 * Measured on an RTX 4070 Ti SUPER, MindGrab conv, warps -> ms:
 * 2 -> 332, 4 -> 209, 8 -> 161, 16 -> 127, 24 -> 127, 32 -> 152.
 */
#define X(W) \
static void tune_run_wmma_##W(const bc_act *source, bc_act *destination, \
        const bc_act *weights, int dilation) { \
    gpu_wmma(W, source, destination, weights, dilation); \
}
BC_CUDA_WMMA_VARIANTS(X)
#undef X

static const cuda_tune_candidate cuda_wmma_candidates[] = {
#define X(W) { W, 0, tune_run_wmma_##W },
	BC_CUDA_WMMA_VARIANTS(X)
#undef X
};
#define CUDA_WMMA_CANDIDATE_COUNT \
	((int)(sizeof(cuda_wmma_candidates) / sizeof(cuda_wmma_candidates[0])))

/* Dispatch for the real layer loop, once the tuner has chosen. */
static void cuda_launch_wmma(int warps, const bc_act *source,
		bc_act *destination, const bc_act *weights, int dilation) {
	for (int i = 0; i < CUDA_WMMA_CANDIDATE_COUNT; i++)
		if (cuda_wmma_candidates[i].first == warps) {
			cuda_wmma_candidates[i].run(source, destination, weights, dilation);
			return;
		}
	gpu_wmma(CUDA_WMMA_WARPS, source, destination, weights, dilation);
}

static const cuda_conv_variant cuda_variants[] = {
#define X(V, T) { V, T, launch_conv_##V##_##T },
	BC_CUDA_SCALAR_VARIANTS(X)
#undef X
};
#define CUDA_VARIANT_COUNT \
	((int)(sizeof(cuda_variants) / sizeof(cuda_variants[0])))

/* The compiled-in default, used until a tuned value is available. */
static int cuda_default_variant(void) {
	for (int i = 0; i < CUDA_VARIANT_COUNT; i++)
		if (cuda_variants[i].vpt == CUDA_CONV_VPT &&
				cuda_variants[i].threads == CUDA_CONV_THREADS)
			return i;
	return 0;
}

/* Maintainer offline search override. A single token applies to all hidden
 * layers (layer zero stays scalar); a comma-separated schedule must name
 * every layer explicitly. No runtime compilation or unvalidated indices. */
typedef struct { int scalar, warps; } cuda_layer_choice;
static int cuda_parse_schedule(const char *text, int layers, int wmma_available,
        cuda_layer_choice *out) {
    if (!text || !*text || layers != BC_MODEL_NHIDDEN) return 1;
    char buffer[2048];
    if (strlen(text) >= sizeof(buffer)) return 1;
    strcpy(buffer, text);
    int count = 0;
    char *token = buffer;
    while (token) {
        char *next = strchr(token, ',');
        if (next) *next++ = 0;
        if (count >= layers) return 1;
        int v = 0, t = 0, w = 0, used = 0;
        cuda_layer_choice choice = {cuda_default_variant(), 0};
        if (sscanf(token, "s%dx%d%n", &v, &t, &used) == 2 && used > 0 && !token[used]) {
            choice.scalar = -1;
            for (int i = 0; i < CUDA_VARIANT_COUNT; i++)
                if (cuda_variants[i].vpt == v && cuda_variants[i].threads == t) choice.scalar = i;
            if (choice.scalar < 0) return 1;
        } else {
            used = 0;
            if (sscanf(token, "w%d%n", &w, &used) != 1 || used <= 0 || token[used] || !wmma_available) return 1;
            int found = 0;
            for (int i = 0; i < CUDA_WMMA_CANDIDATE_COUNT; i++)
                if (cuda_wmma_candidates[i].first == w) found = 1;
            /* Both adapters use the portable default shared-memory ceiling. */
            if (!found || 27 * CUDA_CS * CUDA_CS * sizeof(bc_act) +
                    (size_t)w * CUDA_WMMA_SCRATCH > 48 * 1024) return 1;
            choice.warps = w;
        }
        out[count++] = choice;
        token = next;
    }
    if (count == 1) {
        for (int i = 1; i < layers; i++) out[i] = out[0];
        out[0].warps = 0;
    } else if (count != layers || out[0].warps) return 1;
    return 0;
}

static int checksum_of(const bc_act *buffer, size_t count,
		unsigned long long *out) {
	unsigned long long *device_total = NULL;
	if (gpu_alloc((void **)&device_total, sizeof(*device_total)) != GPU_SUCCESS)
		return 1;
	int failed = gpu_zero(device_total, 0, sizeof(*device_total)) != GPU_SUCCESS;
	if (!failed) {
		gpu_checksum(buffer, device_total, count);
		failed = gpu_sync() != GPU_SUCCESS;
	}
	if (!failed)
		failed = gpu_copy(out, device_total, sizeof(*out),
			GPU_DTOH) != GPU_SUCCESS;
	gpu_free(device_total);
	return failed;
}

/*
 * The tuned value is cached by bc_accel_tune_* (backend_accel.c), keyed on
 * everything whose change could move the optimum.
 *
 * CUDA_TUNE_STAMP is bumped by hand whenever the convolution changes in a way
 * that could move the optimum; without it a cache written by an older build
 * would silently pin a shape tuned for a kernel that no longer exists.
 *
 * The compiled architecture list belongs in the key too. The same device,
 * driver and model can be served by a native cubin or by JIT'd PTX depending on
 * how the build was configured (CUDA_GENCODE / CUDA_ARCH), and those are
 * different machine code with plausibly different optima.
 * __CUDA_ARCH_LIST__ is a comma list, hence the variadic stringify.
 */
#define CUDA_TUNE_STAMP 3

#define BC_STR2(...) #__VA_ARGS__
#define BC_STR(...) BC_STR2(__VA_ARGS__)
#ifndef CUDA_TUNE_CODEGEN
#ifdef __CUDA_ARCH_LIST__
#define CUDA_TUNE_CODEGEN BC_STR(__CUDA_ARCH_LIST__)
#else
#define CUDA_TUNE_CODEGEN "default"
#endif
#endif

/* Timed launches per variant; the best is what ranks. See the loop for why. */
#ifndef CUDA_TUNE_REPEATS
#define CUDA_TUNE_REPEATS 3
#endif

/* Device identity, as it appears in the cache key. Spaces are kept; the field
   separator is '|', which no device name or version string contains. */
static void cuda_device_key(char *device, size_t device_capacity,
		int *major, int *minor, int *driver) {
	gpu_properties properties;
	int ordinal = 0;
	*major = *minor = *driver = 0;
	snprintf(device, device_capacity, "unknown");
	if (gpu_device(&ordinal) != GPU_SUCCESS) return;
	if (gpu_properties_get(&properties, ordinal) == GPU_SUCCESS) {
		snprintf(device, device_capacity, "%s", properties.name);
		*major = properties.major;
		*minor = properties.minor;
	}
	gpu_driver_version(driver);
	for (char *p = device; *p; p++)
		if (*p == '|' || *p == '\n') *p = ' ';
}

/*
 * The key names the FAMILY as well as the device, because the two searches
 * cache different quantities -- (VPT, THREADS) for the scalar kernel, (warps, 0)
 * for the tensor-core one -- and a machine can legitimately produce both, e.g.
 * a default fatbin build and a CUDA_ARCH=sm_60 build on the same card. Sharing
 * one line between them would hand the wrong pair to whichever ran second.
 */
static void cuda_cache_key(char *key, size_t capacity, int use_wmma) {
	char device[256];
	int major, minor, driver;
	cuda_device_key(device, sizeof(device), &major, &minor, &driver);
	snprintf(key, capacity, "cuda|%s|%s|%d.%d|%d|%s|%s|%d",
		use_wmma ? "wmma" : "scalar", device, major, minor, driver,
		BC_MODEL_NAME, CUDA_TUNE_CODEGEN, CUDA_TUNE_STAMP);
}

/* Returns the candidate index, or -1 when this key has no usable entry. */
static int cuda_cache_load(const char *key, const cuda_tune_candidate *table,
		int count) {
	int values[2];
	if (bc_accel_tune_load(key, values, 2)) return -1;
	for (int i = 0; i < count; i++)
		if (table[i].first == values[0] && table[i].second == values[1])
			return i;
	return -1;
}

static int cuda_cache_store(const char *key, const cuda_tune_candidate *chosen) {
	int values[2] = { chosen->first, chosen->second };
	return bc_accel_tune_store(key, values, 2);
}

/*
 * Time every candidate on a representative hidden convolution and keep the
 * fastest that reproduces the reference checksum exactly.
 *
 * `scratch_a` is overwritten with a fixed pattern and `scratch_b` receives the
 * output, so this must run BEFORE the real input is uploaded. Both buffers are
 * the caller's; tuning allocates nothing beyond a single counter.
 *
 * A candidate that fails to launch is skipped rather than fatal -- the expected
 * outcome for the widest blocks on a small device, and precisely what the
 * search exists to discover.
 */
static int cuda_tune(bc_act *scratch_a, bc_act *scratch_b,
		const bc_act *weights, const cuda_tune_candidate *table, int count,
		int use_wmma, int *chosen) {
	unsigned long long reference = 0;
	int have_reference = 0, best = -1;
	double best_ms = 0.0;
	float slowest = 0.0f, best_slowest = 0.0f;
	size_t elements = CUDA_NVOX * (size_t)CUDA_CS;
	/* Dilation 4 is mid-range for both models; throughput is flat across
	   dilation, so the choice does not bias the ranking. */
	const int dilation = 4;
	const char *first_label = use_wmma ? "warps" : "vpt";

	gpu_fill(scratch_a, elements);
	gpu_error_t fill_error = gpu_sync();
	if (fill_error != GPU_SUCCESS) {
		(void)gpu_last_error();
		cuda_error(fill_error, "preparing tuning input");
		return 1;
	}

	fprintf(stderr, "brainchopc: tuning the %s CUDA convolution for this device "
		"(%d variants, once per device -- caching the result)\n",
		use_wmma ? "tensor-core" : "scalar", count);

	/*
	 * Establish what "correct" means before ranking anything, so every
	 * candidate is judged against the same answer regardless of table order.
	 * The compiled-in default is the natural reference; if even that cannot
	 * launch here, the first candidate that can becomes the reference instead.
	 */
	{
		int reference_index = 0;
		for (int i = 0; i < count; i++)
			if (use_wmma ? (table[i].first == CUDA_WMMA_WARPS)
					: (table[i].first == CUDA_CONV_VPT &&
						table[i].second == CUDA_CONV_THREADS))
				reference_index = i;
		table[reference_index].run(scratch_a, scratch_b, weights, dilation);
		if (gpu_sync() == GPU_SUCCESS &&
				!checksum_of(scratch_b, elements, &reference))
			have_reference = 1;
		else
			(void)gpu_last_error();
	}

	for (int i = 0; i < count; i++) {
		float milliseconds = 0.0f;
		unsigned long long digest = 0;

		/* Warm-up pass, which is also the launch-viability test. */
		table[i].run(scratch_a, scratch_b, weights, dilation);
		if (gpu_sync() != GPU_SUCCESS) {
			(void)gpu_last_error();
			if (use_wmma)
				fprintf(stderr, "brainchopc:   warps=%-3d unavailable\n",
					table[i].first);
			else
				/* THREADS is what makes a scalar candidate fail to launch, so
				   omitting it gave three indistinguishable "vpt=8" lines. */
				fprintf(stderr, "brainchopc:   vpt=%-2d threads=%-4d "
					"unavailable\n", table[i].first, table[i].second);
			continue;
		}
		if (checksum_of(scratch_b, elements, &digest)) {
			/* Clear it here, or gpu_sync() above will report this
			   failure against the NEXT candidate and skip a good one. */
			(void)gpu_last_error();
			continue;
		}
		if (!have_reference) {
			reference = digest;
			have_reference = 1;
		} else if (digest != reference) {
			fprintf(stderr, "brainchopc:   %s=%-3d rejected (checksum "
				"mismatch)\n", first_label, table[i].first);
			continue;
		}

		/*
		 * Best of CUDA_TUNE_REPEATS, not a single shot: the result is written
		 * to disk and trusted indefinitely, so a one-off measurement taken
		 * while the GPU was busy would be baked in permanently. An audit
		 * reproduced exactly that, costing 49%.
		 */
		for (int repeat = 0; repeat < CUDA_TUNE_REPEATS; repeat++) {
			float sample = 0.0f;
			gpu_event start, stop;
			int bad = gpu_event_create(&start) != GPU_SUCCESS;
			if (!bad) {
				bad = gpu_event_create(&stop) != GPU_SUCCESS;
				if (!bad) {
					bad = gpu_event_record(start) != GPU_SUCCESS;
					if (!bad) table[i].run(scratch_a, scratch_b, weights, dilation);
					bad |= gpu_event_record(stop) != GPU_SUCCESS;
					bad |= gpu_event_sync(stop) != GPU_SUCCESS;
					if (!bad) bad = gpu_event_elapsed(&sample, start, stop) != GPU_SUCCESS;
					bad |= gpu_event_destroy(stop) != GPU_SUCCESS;
				}
				bad |= gpu_event_destroy(start) != GPU_SUCCESS;
			}
			bad |= gpu_last_error() != GPU_SUCCESS;
			if (bad) { milliseconds = 0.0f; break; }
			if (milliseconds <= 0.0f || sample < milliseconds) milliseconds = sample;
			if (sample > slowest) slowest = sample;
		}

		if (milliseconds <= 0.0f) {
			slowest = 0.0f;
			continue;
		}

		if (use_wmma)
			fprintf(stderr, "brainchopc:   warps=%-3d %7.2f ms\n",
				table[i].first, (double)milliseconds);
		else
			fprintf(stderr, "brainchopc:   vpt=%-2d threads=%-4d %7.2f ms\n",
				table[i].first, table[i].second, (double)milliseconds);
		if (best < 0 || milliseconds < best_ms) {
			best = i;
			best_ms = milliseconds;
			best_slowest = slowest;
		}
		/* Reset on EVERY path, not just this one, or a rejected candidate's
		   spread leaks into the next candidate's busy-device advisory. */
		slowest = 0.0f;
	}
	if (best < 0) return 1;
	if (use_wmma)
		fprintf(stderr, "brainchopc: chose warps=%d (%.2f ms)\n",
			table[best].first, best_ms);
	else
		fprintf(stderr, "brainchopc: chose vpt=%d threads=%d (%.2f ms)\n",
			table[best].first, table[best].second, best_ms);
	/*
	 * A wide spread between repeats of the SAME candidate means the device was
	 * doing something else, which is the condition under which this search
	 * picks badly -- and the pick is then cached and sticky. Say so rather
	 * than silently recording it.
	 */
	if (best_slowest > 1.5f * best_ms)
		fprintf(stderr, "brainchopc: timings varied %.2f-%.2f ms for the same "
			"shape, so this device was busy; re-run with --tune when idle\n",
			best_ms, (double)best_slowest);
	*chosen = best;
	return 0;
}

/*
 * Normalisation runs over the STRIDE, not the logical channel count.
 *
 * Every buffer here is CUDA_CS wide, so a reduction told there are 15 channels
 * would walk 15/16 of the activation and assign each element to the wrong
 * channel. That is not a subtle inaccuracy -- it moved 1,042,645 MindGrab
 * voxels when this file first went fp16, and model16 hid it completely because
 * its channel count already equals the stride. The parameter is taken and
 * immediately overridden rather than removed so the call sites still read as
 * "normalise this many channels"; the override is the invariant.
 *
 * The pad lane's moments come out exactly (0, 0), which normalises and
 * activates back to zero.
 */
static int run_normalize(bc_act *buffer, double *sums, float *mean, float *invstd,
		const float *scale, const float *bias, int channels) {
	channels = CUDA_CS;
	if (cuda_error(gpu_zero(sums, 0, (size_t)channels * 2 * sizeof(*sums)),
			"clearing normalization workspace"))
		return 1;
	cuda_stage stage;
	stage_begin(&stage);
	/*
	 * Each thread keeps one channel for its whole walk only if the grid stride
	 * is a multiple of the channel count, so round the block count up to one.
	 * For 16 channels and 256-thread blocks the stride already divides and this
	 * is the identity; for MindGrab's 15 it lifts 256 blocks to 270.
	 */
	unsigned reduce_blocks = CUDA_REDUCE_BLOCKS;
	{
		unsigned a = (unsigned)channels, b = CUDA_REDUCE_THREADS;
		while (b) { unsigned t = a % b; a = b; b = t; }
		unsigned step = (unsigned)channels / a;   /* channels / gcd(channels, threads) */
		reduce_blocks = (reduce_blocks + step - 1) / step * step;
	}
	gpu_reduce(reduce_blocks, buffer, sums, channels);
	gpu_moments(sums, mean, invstd, channels);
	stage_end(&stage, 1);
	stage_begin(&stage);
	gpu_norm(buffer, mean, invstd, scale, bias, channels);
	stage_end(&stage, 2);
	return cuda_error(gpu_last_error(), "normalization kernel") || g_timing_error;
}

/* The padded fp16 weight layout, shared with backend_opencl.c. */
#define bc_pad_store bc_act
#define BC_PAD_STORE(x) gpu_half(x)
#define BC_PAD_CS CUDA_CS
#include "accel_pack_padded.h"

static int cuda_finish(int failed) {
	if (gpu_end())
		fprintf(stderr, "brainchopc: warning: CUDA cleanup did not complete\n");
	return failed ? 1 : 0;
}

static int cuda_segment(const unsigned char *conformed,
		float *result, const float *weights, const bc_inference_options *options) {
	const bc_accel_model_spec *spec = bc_accel_spec();
	if (!conformed || !result || !weights || !spec) return 1;
	if (options && (options->roi_enabled || options->class_writer)) return 2;
	if (options && options->tissue_planes && (spec->classes != 18 || BC_MODEL_OUTPUT_IS_MASK)) return 2;
	/*
	 * Bind the requested device BEFORE the first allocation. The runtime
	 * adapter selects its per-thread device; the Driver API adapter creates
	 * its owned context and loads the embedded module here.
	 *
	 * The tune cache keys on device name and compute capability, so switching
	 * adapters looks up a different entry rather than reusing a shape measured
	 * on the other card.
	 */
	if (gpu_begin(options ? options->adapter : -1))
		return 1;
	memset(g_stage_ms, 0, sizeof(g_stage_ms));
	g_timing_error = 0;
	float *host_input = bc_accel_normalize_input( conformed);
	float *host_weights = bc_accel_copy_weights( weights);
	bc_act *device_a = NULL, *device_b = NULL, *device_weights = NULL;
	float *device_scale = NULL, *device_bias = NULL;
	bc_act *half_input = NULL;
	bc_padded_weights packed;
	memset(&packed, 0, sizeof(packed));
	double *device_sums = NULL;
	float *device_mean = NULL, *device_invstd = NULL;
	int failed = !host_input || !host_weights;
	if (!failed) failed = bc_pack_padded_weights(spec, host_weights, &packed);
	/* CUDA_CS, not spec->channels: the buffer is padded to the stride. Even so
	   this HALVES what the fp32 layout used -- MindGrab went 15x4 to 16x2 bytes
	   per voxel, model16 16x4 to 16x2. */
	size_t activation_bytes = CUDA_NVOX * (size_t)CUDA_CS * sizeof(bc_act);
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_a, activation_bytes),
			"allocating first activation");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_b, activation_bytes),
			"allocating second activation");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_weights,
			packed.count * sizeof(bc_act)), "allocating weights");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_sums,
			(size_t)CUDA_CS * 2 * sizeof(double)),
			"allocating normalization workspace");
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_mean,
			(size_t)CUDA_CS * sizeof(float)), "allocating channel mean") ||
			cuda_error(gpu_alloc((void **)&device_invstd,
			(size_t)CUDA_CS * sizeof(float)),
			"allocating channel inverse deviation");
#if BC_MODEL_HAS_AFFINE
	if (!failed)
		failed = cuda_error(gpu_alloc((void **)&device_scale,
			BC_MODEL_CHAN * sizeof(float)), "allocating affine scale") ||
			cuda_error(gpu_alloc((void **)&device_bias,
			BC_MODEL_CHAN * sizeof(float)), "allocating affine bias");
#endif
	/*
	 * Narrow on the host and upload once. The alternative -- upload fp32 and
	 * convert on the device -- would need a second full-size weight buffer and
	 * a kernel, to save a conversion of 146k values that happens once per run.
	 * host_weights stays fp32 because the affine extraction below reads it.
	 */
	if (!failed) {
		/* Shared with backend_opencl.c and parallel; see bc_accel_narrow_input.
		   bc_f32_to_f16 and __float2half both round to nearest even, so this is
		   the same bits the private loop produced. */
		half_input = (bc_act *)bc_accel_narrow_input(host_input);
		if (!half_input) failed = 1;
	}
	if (!failed) {
		failed = cuda_error(gpu_copy(device_weights, packed.data,
			packed.count * sizeof(bc_act), GPU_HTOD),
			"copying weights");
	}

	/*
	 * Pick the convolution variant, tuning first if this device has no cached
	 * answer. This runs BEFORE the input upload below because it scribbles on
	 * both activation buffers.
	 *
	 * --tune (options->force_tuning) or BRAINCHOPC_TUNE=1 forces a retune even
	 * when a cached value exists; BRAINCHOPC_TUNE=0 pins the compiled-in default
	 * and skips both the cache and the search, which is what you want when
	 * comparing two trees rather than two devices. The explicit argument beats
	 * the inherited environment, so --tune wins over BRAINCHOPC_TUNE=0.
	 */
	/*
	 * Tensor cores need sm_70. Probed rather than assumed: the fatbin spans
	 * sm_60 upward, so a Pascal card runs the same binary and must fall back to
	 * the scalar kernel rather than launching an mma it cannot execute.
	 * BRAINCHOPC_CUDA_WMMA=0 forces the fallback, which is the A/B control for
	 * measuring what the tensor cores are actually worth.
	 */
	int use_wmma = 0;
	if (!failed) {
		int ordinal = 0;
		gpu_properties properties;
		const char *request = getenv("BRAINCHOPC_CUDA_WMMA");
		if (gpu_device(&ordinal) == GPU_SUCCESS &&
				gpu_properties_get(&properties, ordinal) == GPU_SUCCESS)
			use_wmma = properties.major >= 7;
		else
			(void)gpu_last_error();
		/* Both conditions, not either: the device must have the hardware AND
		   this build must contain the instructions for it. */
		if (use_wmma && !cuda_wmma_compiled()) {
			fprintf(stderr, "brainchopc: this build has no tensor-core kernel "
				"for %s (compute capability %d.%d); using the scalar "
				"convolution.\n",
				properties.name, properties.major, properties.minor);
			use_wmma = 0;
		}
		if (request && !strcmp(request, "0")) use_wmma = 0;
	}

	/*
	 * Choose the launch shape for whichever kernel will actually run, and tune
	 * if this device has no cached answer for that family.
	 *
	 * This is keyed on use_wmma because the two families are not
	 * interchangeable: on a tensor-core device the scalar kernel runs layer 0
	 * only, so tuning it would measure a shape used once in 13-25 layers -- and
	 * that is exactly what this backend did until it was measured.
	 */
	const char *schedule_text = getenv("BRAINCHOPC_CUDA_LAYERS");
	cuda_layer_choice schedule[BC_MODEL_NHIDDEN];
	if (!failed && schedule_text && cuda_parse_schedule(schedule_text, spec->layers, use_wmma, schedule)) {
		fprintf(stderr, "brainchopc: invalid or unsupported BRAINCHOPC_CUDA_LAYERS schedule\n");
		failed = 1;
	}
	const cuda_tune_candidate *table = use_wmma
		? cuda_wmma_candidates : cuda_scalar_candidates;
	int table_count = use_wmma
		? CUDA_WMMA_CANDIDATE_COUNT : CUDA_SCALAR_CANDIDATE_COUNT;
	int chosen_index = -1;
	for (int i = 0; i < table_count; i++)
		if (use_wmma ? (table[i].first == CUDA_WMMA_WARPS)
				: (table[i].first == CUDA_CONV_VPT &&
					table[i].second == CUDA_CONV_THREADS))
			chosen_index = i;
	if (chosen_index < 0) chosen_index = 0;

	if (!failed && !schedule_text) {
		const char *request = getenv("BRAINCHOPC_TUNE");
		int forced = (options && options->force_tuning) ||
			(request && !strcmp(request, "1"));
		int disabled = request && !strcmp(request, "0") && !forced;
		char key[768];
		cuda_cache_key(key, sizeof(key), use_wmma);
		int cached = disabled ? -1
			: cuda_cache_load(key, table, table_count);
		if (cached >= 0 && !forced) {
			chosen_index = cached;
		} else if (!disabled) {
			int picked = chosen_index;
			if (!cuda_tune(device_a, device_b, device_weights, table,
					table_count, use_wmma, &picked)) {
				chosen_index = picked;
				if (cuda_cache_store(key, &table[picked]))
					fprintf(stderr, "brainchopc: could not save the tuning "
						"result; it will be measured again next run\n");
			} else {
				fprintf(stderr, "brainchopc: tuning failed; using the "
					"compiled-in convolution shape\n");
			}
		}
	}
	int chosen_warps = use_wmma ? table[chosen_index].first : 0;
	/*
	 * The tuning candidates carry a launcher that fixes the hidden-layer shape,
	 * which cannot serve layer 0 (one input channel). The real loop therefore
	 * dispatches through cuda_variants, looked up by the pair the tuner chose
	 * rather than by index -- the two tables happen to be in the same order and
	 * relying on that would be a silent trap the first time one is reordered.
	 */
	/* Both tables are expanded from BC_CUDA_SCALAR_VARIANTS, so index i is the same
	   shape in each -- guaranteed by construction rather than by convention. */
	BC_CUDA_ASSERT(CUDA_SCALAR_CANDIDATE_COUNT == CUDA_VARIANT_COUNT,
		"the scalar candidate and launcher tables come from one list");
	int scalar_variant = use_wmma ? cuda_default_variant() : chosen_index;

	/* Layer 0 reads this with a stride of ONE, not CUDA_CS -- the input is a
	   single channel and only becomes 16-wide once the first convolution has
	   written it. That is why conv3d_kernel takes input_stride separately. */
	if (!failed)
		failed = cuda_error(gpu_copy(device_b, half_input,
			CUDA_NVOX * sizeof(bc_act), GPU_HTOD), "copying input");

	if (schedule_text)
		snprintf(g_variant_label, sizeof(g_variant_label), "offline layer schedule");
	else if (use_wmma)
		snprintf(g_variant_label, sizeof(g_variant_label), "wmma warps=%d",
			chosen_warps);
	else
		snprintf(g_variant_label, sizeof(g_variant_label), "vpt=%d threads=%d",
			table[chosen_index].first, table[chosen_index].second);

	/*
	 * A second walk, over the ORIGINAL fp32 weights. The convolutions now index
	 * the padded fp16 copy through packed.layer_offset, but the affine block is
	 * still read from host_weights, so its position has to be tracked in the
	 * source layout. Keeping the two walks separate and explicitly named is
	 * what stops the padded and unpadded strides being confused for each other.
	 */
#if BC_MODEL_HAS_AFFINE
	size_t raw_offset = 0;
#endif
	unsigned blocks = (unsigned)((CUDA_NVOX + CUDA_THREADS - 1) / CUDA_THREADS);
	for (int layer = 0; layer < spec->layers && !failed; layer++) {
		int input_stride = layer == 0 ? 1 : CUDA_CS;
		bc_act *source = layer ? ((layer & 1) ? device_a : device_b) : device_b;
		bc_act *destination = (layer & 1) ? device_b : device_a;
		/* The variant owns its own grid; weights go to shared memory. */
		size_t shared_bytes = (size_t)27 * (layer == 0 ? 1 : CUDA_CS) * CUDA_CS *
			sizeof(bc_act);
		int layer_warps = schedule_text ? schedule[layer].warps : (layer > 0 && use_wmma ? chosen_warps : 0);
		int layer_scalar = schedule_text ? schedule[layer].scalar : scalar_variant;
		double before_ms = g_stage_ms[0];
		cuda_stage stage;
		stage_begin(&stage);
		if (layer_warps) {
			/* Layer 0 is excluded by construction: one input channel and a
			   stride-1 source give K=1, which a 16-wide fragment cannot use.
			   It runs the compiled-in scalar shape, which is why the scalar
			   family is not tuned on a tensor-core device -- one layer in
			   13-25 does not repay an 18-variant search. */
			cuda_launch_wmma(layer_warps, source, destination,
				device_weights + packed.layer_offset[layer],
				spec->dilation[layer]);
		} else {
			cuda_variants[layer_scalar].launch(source, destination,
				device_weights + packed.layer_offset[layer],
				layer == 0 ? 1 : CUDA_CS, spec->channels,
				input_stride, spec->dilation[layer], shared_bytes);
		}
		stage_end(&stage, 0);
		if (cuda_profiling() && !g_timing_error)
			fprintf(stderr, "brainchopc: CUDA layer=%d dilation=%d conv_ms=%.6f\n",
				layer, spec->dilation[layer], g_stage_ms[0] - before_ms);
#if BC_MODEL_HAS_AFFINE
		raw_offset += (size_t)27 * (layer == 0 ? 1 : spec->channels) * spec->channels;
#endif
		failed = cuda_error(gpu_last_error(), "convolution kernel") || g_timing_error;
		if (failed) break;
#if !BC_MODEL_HAS_AFFINE
		{
			failed = run_normalize(destination, device_sums, device_mean,
				device_invstd, NULL, NULL, spec->channels);
		}
#else
		{
			float compact_scale[BC_MODEL_CHAN], compact_bias[BC_MODEL_CHAN];
			for (int channel = 0; channel < BC_MODEL_CHAN; channel++) {
				compact_scale[channel] =
					host_weights[raw_offset + channel * BC_MODEL_CHAN + channel];
				compact_bias[channel] = host_weights[raw_offset +
					BC_MODEL_CHAN * BC_MODEL_CHAN + channel];
			}
			raw_offset += BC_MODEL_CHAN * BC_MODEL_CHAN + BC_MODEL_CHAN;
			failed = cuda_error(gpu_copy(device_scale, compact_scale,
				BC_MODEL_CHAN * sizeof(float), GPU_HTOD),
				"copying affine scale") ||
				cuda_error(gpu_copy(device_bias, compact_bias,
				BC_MODEL_CHAN * sizeof(float), GPU_HTOD),
				"copying affine bias");
			if (!failed)
				failed = run_normalize(destination, device_sums, device_mean,
					device_invstd, device_scale, device_bias, spec->channels);
		}
#endif
	}
	if (!failed) {
		bc_act *last = (spec->layers & 1) ? device_a : device_b;
		/*
		 * Labels go to the buffer that is NOT the classifier's input. Aliasing
		 * them would be a genuine race, not merely untidy: labels[voxel] lands
		 * at a smaller byte offset than voxel's own activation row, so a thread
		 * writing its label can clobber source data that another thread has not
		 * read yet. Both shipped models have an odd layer count, so this used to
		 * come out safe by accident; deriving it from the same parity makes it
		 * safe by construction.
		 */
		bc_act *label_buffer = (spec->layers & 1) ? device_b : device_a;
		const bc_act *classifier_weights = device_weights + packed.classifier_offset;
		const bc_act *classifier_bias = spec->classifier_bias
			? classifier_weights + (size_t)spec->channels * spec->classes : NULL;
		/* Labels are fp32: they are small class indices read straight back to
		   the host, and keeping them fp32 avoids narrowing an integer-valued
		   result through fp16 for no benefit. device_a is reused because the
		   classifier's input is `last`, which may be device_b. */
		int priors = options && options->tissue_planes;
		BC_CUDA_ASSERT(CUDA_CS * sizeof(bc_act) >= BC_PRIOR_PLANES * sizeof(float),
			"inactive activation must hold all tissue planes");
		if (priors)
			gpu_priors(blocks, last, (float *)label_buffer,
				classifier_weights, classifier_bias, spec->channels, spec->classes);
		else
			gpu_classify(blocks, last, (float *)label_buffer,
				classifier_weights, classifier_bias, spec->channels, spec->classes);
		failed = cuda_error(gpu_last_error(), priors ? "tissue prior kernel" : "classifier kernel");
		if (!failed)
			failed = cuda_error(gpu_copy(priors ? options->tissue_planes : result, label_buffer,
				CUDA_NVOX * sizeof(float) * (priors ? BC_PRIOR_PLANES : 1), GPU_DTOH),
				priors ? "copying tissue priors" : "copying labels");
	}
	stage_report();
	gpu_free(device_bias);
	gpu_free(device_scale);
	gpu_free(device_invstd);
	gpu_free(device_mean);
	gpu_free(device_sums);
	gpu_free(device_weights);
	gpu_free(device_b);
	gpu_free(device_a);
	free(half_input);
	bc_free_padded_weights(&packed);
	free(host_weights);
	free(host_input);
	/* Readback already checked execution. Cleanup warnings do not invalidate
	 * completed labels, matching the other backends' release policy. */
	return cuda_finish(failed);
}

const bc_backend bc_cuda_backend = {
	"cuda",
	cuda_available,
	cuda_segment,
	cuda_adapter_count,
	cuda_adapter_name
};
