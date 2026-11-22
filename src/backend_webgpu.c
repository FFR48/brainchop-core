/*
 * WebGPU backend, written against the standard webgpu.h.
 *
 * The same source is intended to build two ways:
 *
 *   wasm    emcc --use-port=emdawnwebgpu, calling the BROWSER's WebGPU
 *           implementation. Dawn is neither shipped nor linked; the port
 *           supplies the standard header and the JS bindings.
 *   native  linked against any webgpu.h implementation -- Dawn or
 *           wgpu-native -- with no source change.
 *
 * Nothing here uses a Dawn- or Firefox-specific extension. The only platform
 * difference is how a device is obtained, which is isolated in acquire_device()
 * below. Because this backend is compute-only there is no surface, swap chain
 * or windowing, so the native path needs no GLFW and is simply
 * instance -> adapter -> device.
 *
 * The kernels are in webgpu_kernels.h. This file is the host: it packs weights,
 * builds pipelines and buffers, walks the layers, and reads the labels back.
 * It mirrors meshnet_cpu_infer(), which remains the specification.
 */

#include "backend.h"
#include "backend_accel.h"
#include "model_specs.h"
#include "webgpu_kernels.h"
#include "f16.h"
#include "print.h"

#include <webgpu/webgpu.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

/*
 * Stage timing, gated on BRAINCHOPC_WEBGPU_PROFILE, mirroring the Metal
 * backend's BRAINCHOPC_METAL_PROFILE. NOT reachable from JS in a browser build:
 * emscripten snapshots the environment into __environ during runtime init, so
 * getenv cannot be flipped after the fact -- which is why the browser profilers
 * are compile-time (BC_WEBGPU_PROFILE below) where the native ones are
 * environment variables.
 *
 * The only stage boundary that is not obvious is the last one: WebGPU submits
 * asynchronously, so the queue drains during the readback map wait rather than
 * at submit. "gpu" therefore means submit-through-map-resolved, which is the
 * closest analogue to Metal's waitUntilCompleted span.
 */
static int bc_profiling(void) {
#ifdef BC_WEBGPU_PROFILE
	/* Under emscripten the environment is snapshotted into __environ during
	   runtime init, before any JS could set Module.ENV, so getenv cannot be
	   flipped from the harness. A profiling module is built separately. */
	return 1;
#else
	static int state = -1;
	if (state < 0) state = getenv("BRAINCHOPC_WEBGPU_PROFILE") ? 1 : 0;
	return state;
#endif
}
static double bc_now(void) {
#ifdef __EMSCRIPTEN__
	return emscripten_get_now() / 1000.0;
#else
	return (double)clock() / CLOCKS_PER_SEC;
#endif
}

#define BC_DIM BRAINCHOP_MODEL_DIM
#define BC_NVOX ((size_t)BC_DIM * BC_DIM * BC_DIM)
#define BC_CS BC_WG_CS
#if BC_WG_F32
typedef float bc_store;
#define BC_STORE(x) (x)
#else
typedef uint16_t bc_store;
#define BC_STORE(x) bc_f32_to_f16(x)
#endif
#define BC_GROUPS_X (BC_DIM / BC_WG_LOCAL_X)
#define BC_PARTIAL_GROUPS ((size_t)BC_GROUPS_X * BC_DIM * BC_DIM)

/* The convolution computes BC_WG_VOX voxels per invocation, so it needs
   proportionally fewer workgroups than the one-voxel-per-invocation kernels. */
#define BC_CONV_GROUPS_X (BC_DIM / (BC_WG_LOCAL_X * BC_WG_VOX))
_Static_assert(BC_DIM % (BC_WG_LOCAL_X * BC_WG_VOX) == 0,
	"conv writes a whole block per invocation and has no x bounds test, so "
	"BC_WG_LOCAL_X * BC_WG_VOX must divide the volume dimension exactly");
_Static_assert(BC_DIM % BC_WG_WIDE_X == 0, "BC_WG_WIDE_X must divide the row");

#if defined(__EMSCRIPTEN__) && !defined(BC_WG_CONV_PRESET)
/* emdawnwebgpu reads GPUDevice.adapterInfo unguarded, and browsers before
   Chrome 132 lack it: the TypeError would abort the module. */
EM_JS(int, bc_has_adapter_info, (WGPUDevice device), {
	return !!WebGPU.getJsObject(device).adapterInfo;
});
#endif

/* The wide conv on AMD adapters; -DBC_WG_CONV_PRESET=0/1 forces it off/on. */
static int use_wide_conv(WGPUDevice device) {
#ifdef BC_WG_CONV_PRESET
	(void)device;
	return BC_WG_CONV_PRESET;
#else
#ifdef __EMSCRIPTEN__
	if (!bc_has_adapter_info(device)) return 0;
#endif
	WGPUAdapterInfo info = {0};
	if (wgpuDeviceGetAdapterInfo(device, &info) != WGPUStatus_Success) return 0;
	int amd = info.vendor.length == 3 && !memcmp(info.vendor.data, "amd", 3);
	wgpuAdapterInfoFreeMembers(info);
	return amd;
#endif
}

/*
 * Mirrors the WGSL `Params` struct. Padded to a multiple of 16 bytes because
 * WGSL rounds a uniform struct's size up to its alignment, and a short C
 * counterpart would make wgpuQueueWriteBuffer copy too few bytes.
 */
typedef struct {
	int32_t dilation;
	uint32_t weight_off;
	uint32_t affine_off;
	uint32_t bias_off;
	uint32_t in_channels;
	uint32_t in_stride;
	uint32_t partial_groups;
	uint32_t pad;
} bc_params;

/*
 * The f16 conversion moved to f16.h when the WebGL2 backend needed the
 * identical rounding. See the note there on why it is shared rather than
 * copied; the implementation and its measurements are unchanged.
 */

/*
 * The one platform-dependent step.
 *
 * In a browser the device is created by JavaScript -- requestAdapter and
 * requestDevice are promises, and this function is not -- and handed over
 * through Module.preinitializedWebGPUDevice. That keeps every other call in
 * this file synchronous, which is why it is preferred over making the whole
 * backend asynchronous.
 */
#ifndef __EMSCRIPTEN__
struct bc_adapter_wait { WGPUAdapter adapter; int done; };
struct bc_device_wait { WGPUDevice device; int done; };

static void bc_on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
		WGPUStringView message, void *a, void *b) {
	(void)message; (void)b;
	struct bc_adapter_wait *w = (struct bc_adapter_wait *)a;
	w->adapter = (status == WGPURequestAdapterStatus_Success) ? adapter : NULL;
	w->done = 1;
}

static void bc_on_device(WGPURequestDeviceStatus status, WGPUDevice device,
		WGPUStringView message, void *a, void *b) {
	(void)message; (void)b;
	struct bc_device_wait *w = (struct bc_device_wait *)a;
	w->device = (status == WGPURequestDeviceStatus_Success) ? device : NULL;
	w->done = 1;
}
#endif

/*
 * The device, created once and reused.
 *
 * Caching matters on the native path for three reasons: webgpu_available() is
 * called during backend auto-selection and used to build a whole second,
 * unrelated device purely for a NULL test; the instance and adapter were never
 * released; and the readback wait needs the instance to pump events.
 */
#ifndef __EMSCRIPTEN__
static WGPUInstance g_instance = NULL;
static WGPUInstance bc_instance(void) { return g_instance; }
#endif

static WGPUDevice acquire_device(void) {
#ifdef __EMSCRIPTEN__
	return emscripten_webgpu_get_device();
#else
	static WGPUDevice cached = NULL;
	static int tried = 0;
	if (tried) return cached;
	tried = 1;

	/* Native: create it here. Compute-only, so no surface is involved. */
	g_instance = wgpuCreateInstance(NULL);
	if (!g_instance) return NULL;
	struct bc_adapter_wait got = { NULL, 0 };
	WGPURequestAdapterCallbackInfo info = {0};
	info.mode = WGPUCallbackMode_WaitAnyOnly;
	info.callback = bc_on_adapter;
	info.userdata1 = &got;
	WGPUFuture future = wgpuInstanceRequestAdapter(g_instance, NULL, info);
	WGPUFutureWaitInfo wait = { future, 0 };
	wgpuInstanceWaitAny(g_instance, 1, &wait, UINT64_MAX);
	if (!got.adapter) return NULL;

	/*
	 * Ask for what the model actually needs. A NULL descriptor takes WebGPU's
	 * DEFAULTS -- 128 MiB per storage binding, 256 MiB per buffer, and no
	 * shader-f16 -- all three of which this model exceeds or requires, so the
	 * native path could never have produced a device that passed the limit
	 * check below or compiled an `enable f16` module. Mirrors what
	 * js/src/device.ts requests for the browser.
	 */
	WGPUFeatureName features[1] = { WGPUFeatureName_ShaderF16 };
	WGPULimits limits = {0};
	limits.maxStorageBufferBindingSize = (uint64_t)BC_NVOX * BC_CS * sizeof(bc_store);
	limits.maxBufferSize = limits.maxStorageBufferBindingSize;
	WGPUDeviceDescriptor ddesc = {0};
	ddesc.requiredFeatureCount = 1;
	ddesc.requiredFeatures = features;
	ddesc.requiredLimits = &limits;

	struct bc_device_wait dev = { NULL, 0 };
	WGPURequestDeviceCallbackInfo dinfo = {0};
	dinfo.mode = WGPUCallbackMode_WaitAnyOnly;
	dinfo.callback = bc_on_device;
	dinfo.userdata1 = &dev;
	WGPUFuture dfuture = wgpuAdapterRequestDevice(got.adapter, &ddesc, dinfo);
	WGPUFutureWaitInfo dwait = { dfuture, 0 };
	wgpuInstanceWaitAny(g_instance, 1, &dwait, UINT64_MAX);
	wgpuAdapterRelease(got.adapter);
	cached = dev.device;
	return cached;
#endif
}

static int webgpu_available(void) {
	return acquire_device() != NULL;
}

static WGPUStringView view(const char *text) {
	WGPUStringView v;
	v.data = text;
	v.length = strlen(text);
	return v;
}

static WGPUBuffer make_buffer(WGPUDevice device, uint64_t size,
		WGPUBufferUsage usage) {
	WGPUBufferDescriptor descriptor = {0};
	descriptor.size = size;
	descriptor.usage = usage;
	return wgpuDeviceCreateBuffer(device, &descriptor);
}

static WGPUComputePipeline make_pipeline(WGPUDevice device,
		WGPUPipelineLayout layout, WGPUShaderModule module, const char *entry) {
	WGPUComputeState state = {0};
	state.module = module;
	state.entryPoint = view(entry);
	WGPUComputePipelineDescriptor descriptor = {0};
	descriptor.layout = layout;
	descriptor.compute = state;
	return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

/*
 * The weight packing lives in accel_pack_weights.h, shared with the WebGL2
 * backend so the tap permutation and the classifier order cannot drift between
 * them -- the two are never compiled together, so a divergence would surface
 * only as one backend's output quietly moving. `pad_to` is 0 here: a storage
 * buffer needs no row padding.
 */
#include "accel_pack_weights.h"

/* Set once the readback map completes; see the wait in webgpu_segment(). */
static volatile int g_map_status = -1;
static volatile int g_error_seen = 0;
/*
 * Set when the error scope RESOLVES, regardless of outcome.
 *
 * Without this the validation check is a race the wrong way round: both
 * callbacks are AllowSpontaneous and nothing orders the pop before the map, so
 * if the map resolved first the code read g_error_seen while it was still 0 and
 * returned a label buffer that a failed dispatch had left zeroed -- an entirely
 * blank segmentation, reported as success. That is the exact outcome the error
 * scope was added to prevent, so the wait below must cover both.
 */
static volatile int g_scope_done = 0;

/* Used only by the cleanup label; deliberately touches no shared state. */
static void on_error_scope_drop(WGPUPopErrorScopeStatus status, WGPUErrorType type,
		WGPUStringView message, void *a, void *b) {
	(void)status; (void)type; (void)message; (void)a; (void)b;
}

static void on_error_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type,
		WGPUStringView message, void *a, void *b) {
	(void)a; (void)b;
	g_scope_done++;
	if (status != WGPUPopErrorScopeStatus_Success || type != WGPUErrorType_NoError) {
		g_error_seen = 1;
		if (message.data && message.length)
			printfx("webgpu: %.*s\n", (int)message.length, message.data);
	}
}
static void on_mapped(WGPUMapAsyncStatus status, WGPUStringView message,
		void *a, void *b) {
	(void)message; (void)a; (void)b;
	g_map_status = (int)status;
}

/*
 * Pop the two error scopes, map `readback`, and wait for BOTH -- see
 * g_scope_done. The only asynchronous step: ASYNCIFY lets this synchronous
 * code yield to the browser's event loop; natively the instance must be
 * pumped, since callbacks never run otherwise and a bare spin is a hang.
 * Returns 0 only when the map succeeded and no error was reported.
 */
static int bc_wg_pop_and_map(WGPUDevice device, WGPUBuffer readback, uint64_t bytes) {
	g_error_seen = 0;
	g_scope_done = 0;
	WGPUPopErrorScopeCallbackInfo scope_info = {0};
	scope_info.mode = WGPUCallbackMode_AllowSpontaneous;
	scope_info.callback = on_error_scope;
	wgpuDevicePopErrorScope(device, scope_info);
	wgpuDevicePopErrorScope(device, scope_info);
	g_map_status = -1;
	WGPUBufferMapCallbackInfo map_info = {0};
	map_info.mode = WGPUCallbackMode_AllowSpontaneous;
	map_info.callback = on_mapped;
	wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, bytes, map_info);
#ifdef __EMSCRIPTEN__
	while (g_map_status < 0 || g_scope_done < 2) emscripten_sleep(1);
#else
	while (g_map_status < 0 || g_scope_done < 2)
		wgpuInstanceProcessEvents(bc_instance());
#endif
	if (g_map_status != WGPUMapAsyncStatus_Success) {
		printfx("webgpu: readback failed (status %d)\n", g_map_status);
		return 1;
	}
	if (g_error_seen) {
		printfx("webgpu: the device reported an error; refusing to return a result\n");
		return 1;
	}
	return 0;
}

/*
 * Pop the scopes a failed call left pushed: they would otherwise accumulate
 * on the device. The callback touches NOTHING shared: the pop is spontaneous,
 * so a drop from a failed call can resolve during the NEXT call, and writing
 * g_scope_done/g_error_seen there would either release that call's wait early
 * -- re-opening the blank-result race -- or refuse a good result.
 */
static void bc_wg_drop_scopes(WGPUDevice device, int count) {
	while (count-- > 0) {
		WGPUPopErrorScopeCallbackInfo drop = {0};
		drop.mode = WGPUCallbackMode_AllowSpontaneous;
		drop.callback = on_error_scope_drop;
		wgpuDevicePopErrorScope(device, drop);
	}
}

#include "webgpu_pve.h"
#if !BC_WG_F32
#include "webgpu_turbo.h"
#endif

static int webgpu_segment(const unsigned char *conformed, float *result,
		const float *weights, const bc_inference_options *options) {
	if (!conformed || !result || !weights) return 1;
	/* An ROI or per-class export would need kernels this backend does not
	   have; report "unsupported" so the caller can fall back rather than
	   silently returning a whole-volume answer to a cropped question. */
	if (options && (options->roi_enabled || options->class_writer)) return 2;
	const int pve = options && options->tissue_planes;
#if !BC_HAS_PVE
	if (pve) return 2;
#endif
	const uint64_t output_bytes = BC_NVOX * sizeof(float) * (pve ? 4 : 1);

	double t_start = bc_now(), t_pipelines = 0, t_pack = 0, t_upload = 0, t_gpu = 0;

	/*
	 * Everything releasable, declared up front and NULL until it exists, so the
	 * single `cleanup` label at the bottom can release exactly what was created.
	 * Before this, all ten failure exits leaked the whole graph -- including two
	 * 512 MiB activation buffers -- and the success path leaked the module, both
	 * layouts and all five pipelines. wgpu*Release is a no-op on NULL, so the
	 * label needs no per-handle guards.
	 */
	int status = 1;
	int scope_pushed = 0;
	WGPUShaderModule module = NULL, wide_module = NULL;
	WGPUBindGroupLayout layout = NULL;
	WGPUPipelineLayout pipeline_layout = NULL;
	WGPUComputePipeline conv = NULL, moments = NULL, finish = NULL,
		norm = NULL, classify = NULL;
	WGPUBuffer input_buf = NULL, act_a = NULL, act_b = NULL, weight_buf = NULL,
		param_buf = NULL, moment_buf = NULL, label_buf = NULL, readback = NULL;
	WGPUBindGroup groups[3] = { NULL, NULL, NULL };
	bc_store *packed = NULL, *input16 = NULL;
	float *normalized = NULL;

	WGPUDevice device = acquire_device();
	if (!device) {
		printfx("webgpu: no device; the host must hand one over\n");
		return 1;
	}
	WGPUQueue queue = wgpuDeviceGetQueue(device);
#if !BC_WG_F32
	/* The 8-bit path has no tissue-prior head: --pve stays fp16. */
	if (!pve && !(options && options->no_turbo)) {
		int turbo = webgpu_turbo(device, queue, conformed, result, weights);
		wgpuQueueRelease(queue);
		return turbo;
	}
#endif

	/*
	 * Refuse a device that cannot hold an activation buffer, rather than
	 * letting createBuffer fail deeper in. This is the real constraint on which
	 * devices can run the model at all: 512 MiB per activation at f16.
	 */
	uint64_t activation_bytes = (uint64_t)BC_NVOX * BC_CS * sizeof(bc_store);
	WGPULimits limits = {0};
	if (wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
			(limits.maxStorageBufferBindingSize < activation_bytes ||
			 limits.maxBufferSize < activation_bytes)) {
		printfx("webgpu: this device allows %llu MiB per storage binding, "
			"but the model needs %llu MiB\n",
			(unsigned long long)(limits.maxStorageBufferBindingSize >> 20),
			(unsigned long long)(activation_bytes >> 20));
		goto cleanup;
	}

	/*
	 * WebGPU reports errors asynchronously. Without an error scope a failed
	 * dispatch simply leaves the label buffer zeroed, and this function returns
	 * a blank segmentation with status 0 -- observed exactly once, which is
	 * once too many.
	 */
	wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
	wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
	scope_pushed = 2;

	WGPUShaderSourceWGSL wgsl = {0};
	wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgsl.code = view(bc_wgsl_source);
	WGPUShaderModuleDescriptor module_descriptor = {0};
	module_descriptor.nextInChain = &wgsl.chain;
	module = wgpuDeviceCreateShaderModule(device,
		&module_descriptor);
	if (!module) { printfx("webgpu: shader module failed\n"); goto cleanup; }
	const int wide = use_wide_conv(device);
	const uint32_t conv_groups_x = wide ? BC_DIM / BC_WG_WIDE_X : BC_CONV_GROUPS_X;
	if (wide) {
		wgsl.code = view(bc_wgsl_wide_conv_source);
		wide_module = wgpuDeviceCreateShaderModule(device, &module_descriptor);
		if (!wide_module) { printfx("webgpu: shader module failed\n"); goto cleanup; }
	}

	/*
	 * An EXPLICIT bind group layout, shared by every pipeline.
	 *
	 * The obvious alternative -- wgpuComputePipelineGetBindGroupLayout -- returns
	 * a layout containing only the bindings that ONE entry point references, so
	 * the layout derived from `conv` omits the moments and label buffers and
	 * rejects the six-entry bind group the other kernels need. Declaring the
	 * full layout once lets a single bind group serve all five entry points.
	 */
	WGPUBindGroupLayoutEntry layout_entries[8] = {0};
	static const WGPUBufferBindingType kinds[8] = {
		WGPUBufferBindingType_ReadOnlyStorage,   /* src     */
		WGPUBufferBindingType_Storage,           /* dst     */
		WGPUBufferBindingType_ReadOnlyStorage,   /* weights */
		WGPUBufferBindingType_Uniform,           /* params  */
		WGPUBufferBindingType_Storage,           /* moments */
		WGPUBufferBindingType_Storage,           /* labels  */
		WGPUBufferBindingType_ReadOnlyStorage,   /* weights again, as vec4 */
		WGPUBufferBindingType_ReadOnlyStorage,   /* src again, as vec4     */
	};
	for (uint32_t i = 0; i < 8; i++) {
		layout_entries[i].binding = i;
		layout_entries[i].visibility = WGPUShaderStage_Compute;
		layout_entries[i].buffer.type = kinds[i];
	}
	WGPUBindGroupLayoutDescriptor layout_descriptor = {0};
	layout_descriptor.entryCount = 8;
	layout_descriptor.entries = layout_entries;
	layout =
		wgpuDeviceCreateBindGroupLayout(device, &layout_descriptor);
	WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = {0};
	pipeline_layout_descriptor.bindGroupLayoutCount = 1;
	pipeline_layout_descriptor.bindGroupLayouts = &layout;
	pipeline_layout =
		wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);
	if (!layout || !pipeline_layout) {
		printfx("webgpu: pipeline layout failed\n");
		goto cleanup;
	}

	conv = make_pipeline(device, pipeline_layout, wide ? wide_module : module, "conv");
	moments = make_pipeline(device, pipeline_layout, module, "moments");
	finish = make_pipeline(device, pipeline_layout, module, "moments_finish");
	norm = make_pipeline(device, pipeline_layout, module, "norm");
	classify = make_pipeline(device, pipeline_layout, module, "classify");
	if (!conv || !moments || !finish || !norm || !classify) {
		printfx("webgpu: pipeline creation failed\n");
		goto cleanup;
	}

	t_pipelines = bc_now();

	bc_layer_offsets offsets[BC_MODEL_NHIDDEN];
	uint32_t classifier_off = 0, classifier_bias_off = 0;
	size_t weight_count = 0;
	packed = bc_pack_weights(weights, offsets, &classifier_off,
		&classifier_bias_off, &weight_count, 0);
	if (!packed) { printfx("webgpu: out of memory packing weights\n"); goto cleanup; }

	/* Normalisation stays on the CPU: it is a histogram over 16.7M bytes,
	   trivial next to the convolutions, and sharing bc_accel_normalize_input
	   keeps one declaration of the quantile behaviour. */
	normalized = bc_accel_normalize_input(conformed);
	if (!normalized) goto cleanup;
	input16 = (bc_store *)malloc(BC_NVOX * sizeof(*input16));
	if (!input16) goto cleanup;
	for (size_t i = 0; i < BC_NVOX; i++) input16[i] = BC_STORE(normalized[i]);
	/* Released HERE rather than at the label, so 64 MiB is not held live across
	   the two 512 MiB activation allocations below -- and NULLed, because the
	   label frees it too and emscripten's allocator does not trap a double
	   free. */
	free(normalized);
	normalized = NULL;

	t_pack = bc_now();

	uint64_t moment_bytes =
		((uint64_t)BC_PARTIAL_GROUPS * BC_CS * 2 + BC_CS * 2) * sizeof(float);
	input_buf = make_buffer(device, BC_NVOX * sizeof(bc_store),
		WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
	act_a = make_buffer(device, activation_bytes,
		WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
	act_b = make_buffer(device, activation_bytes,
		WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
	weight_buf = make_buffer(device, weight_count * sizeof(bc_store),
		WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
	param_buf = make_buffer(device, sizeof(bc_params),
		WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
	moment_buf = make_buffer(device, moment_bytes,
		WGPUBufferUsage_Storage);
	label_buf = make_buffer(device, pve ? 4 : BC_NVOX * sizeof(uint32_t),
		WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
	readback = make_buffer(device, output_bytes,
		WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
	if (!input_buf || !act_a || !act_b || !weight_buf || !param_buf ||
			!moment_buf || !label_buf || !readback) {
		printfx("webgpu: buffer allocation failed (needs about %llu MiB)\n",
			(unsigned long long)((2 * activation_bytes + moment_bytes) >> 20));
		goto cleanup;
	}

	wgpuQueueWriteBuffer(queue, weight_buf, 0, packed,
		weight_count * sizeof(bc_store));
	wgpuQueueWriteBuffer(queue, input_buf, 0, input16,
		BC_NVOX * sizeof(bc_store));
	t_upload = bc_now();

	/*
	 * One bind group per (src, dst) orientation. The activation buffers
	 * ping-pong exactly as `first`/`second` do in meshnet_cpu_infer, so only
	 * two layouts are ever needed rather than one per layer.
	 */
	struct { WGPUBuffer src, dst; } orientation[3] = {
		{ input_buf, act_a },   /* layer 0 reads the 1-channel input */
		{ act_a, act_b },
		{ act_b, act_a },
	};
	for (int g = 0; g < 3; g++) {
		WGPUBindGroupEntry entries[8] = {0};
		entries[0].binding = 0; entries[0].buffer = orientation[g].src;
		entries[0].size = wgpuBufferGetSize(orientation[g].src);
		entries[1].binding = 1; entries[1].buffer = orientation[g].dst;
		entries[1].size = wgpuBufferGetSize(orientation[g].dst);
		entries[2].binding = 2; entries[2].buffer = weight_buf;
		entries[2].size = wgpuBufferGetSize(weight_buf);
		entries[3].binding = 3; entries[3].buffer = param_buf;
		entries[3].size = sizeof(bc_params);
		entries[4].binding = 4; entries[4].buffer = moment_buf;
		entries[4].size = moment_bytes;
		entries[5].binding = 5; entries[5].buffer = label_buf;
		entries[5].size = pve ? 4 : BC_NVOX * sizeof(uint32_t);
		/* The same weight buffer as entry 2. Legal because both are read-only. */
		entries[6].binding = 6; entries[6].buffer = weight_buf;
		entries[6].size = wgpuBufferGetSize(weight_buf);
		/* Likewise the source activations; also read-only in every kernel. */
		entries[7].binding = 7; entries[7].buffer = orientation[g].src;
		entries[7].size = wgpuBufferGetSize(orientation[g].src);
		WGPUBindGroupDescriptor descriptor = {0};
		descriptor.layout = layout;
		descriptor.entryCount = 8;
		descriptor.entries = entries;
		groups[g] = wgpuDeviceCreateBindGroup(device, &descriptor);
		if (!groups[g]) {
			printfx("webgpu: bind group %d failed\n", g);
			goto cleanup;
		}
	}

	/*
	 * Each layer is four dispatches: convolve, reduce the moments, fold them,
	 * then normalise and activate in place. Params is rewritten per layer, so
	 * every layer gets its own submission -- the queue write must be ordered
	 * against the dispatches that read it.
	 */
	for (int layer = 0; layer < BC_MODEL_NHIDDEN; layer++) {
		int group = layer == 0 ? 0 : (((layer - 1) & 1) ? 2 : 1);
		bc_params params = {0};
		params.dilation = brainchop_model_dilation[layer];
		params.weight_off = offsets[layer].kernel_off;
		params.affine_off = offsets[layer].affine_off;
		params.bias_off = offsets[layer].bias_off;
		params.in_channels = layer == 0 ? 1u : (uint32_t)BC_CS;
		params.in_stride = layer == 0 ? 1u : (uint32_t)BC_CS;
		params.partial_groups = (uint32_t)BC_PARTIAL_GROUPS;
		wgpuQueueWriteBuffer(queue, param_buf, 0, &params, sizeof(params));

		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
		WGPUComputePassEncoder pass =
			wgpuCommandEncoderBeginComputePass(encoder, NULL);
		wgpuComputePassEncoderSetBindGroup(pass, 0, groups[group], 0, NULL);

		/* BC_WEBGPU_STAGES is a measurement-only mask (bit 0 conv, 1 moments,
		   2 norm). Omitting a stage produces a WRONG result and exists solely to
		   attribute GPU time per dispatch; the default runs all of them. */
#ifndef BC_WEBGPU_STAGES
#define BC_WEBGPU_STAGES 7
#endif
#if BC_WEBGPU_STAGES & 1
		wgpuComputePassEncoderSetPipeline(pass, conv);
		wgpuComputePassEncoderDispatchWorkgroups(pass, conv_groups_x,
			BC_DIM, BC_DIM);
#endif
		/* moments and norm both read the buffer conv just wrote, so they run
		   against the same bind group with src and dst as set above. */
#if BC_WEBGPU_STAGES & 2
		wgpuComputePassEncoderSetPipeline(pass, moments);
		wgpuComputePassEncoderDispatchWorkgroups(pass, BC_GROUPS_X, BC_DIM, BC_DIM);
		wgpuComputePassEncoderSetPipeline(pass, finish);
		wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
#endif
#if BC_WEBGPU_STAGES & 4
		wgpuComputePassEncoderSetPipeline(pass, norm);
		wgpuComputePassEncoderDispatchWorkgroups(pass, BC_GROUPS_X, BC_DIM, BC_DIM);
#endif

		wgpuComputePassEncoderEnd(pass);
		WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
		wgpuQueueSubmit(queue, 1, &commands);
		wgpuCommandBufferRelease(commands);
		wgpuComputePassEncoderRelease(pass);
		wgpuCommandEncoderRelease(encoder);
	}

	/* The classifier reads whichever buffer the last layer wrote. */
#if BC_HAS_PVE
	if (pve) {
		WGPUBuffer last = ((BC_MODEL_NHIDDEN - 1) & 1) ? act_b : act_a;
		WGPUBuffer scratch = last == act_a ? act_b : act_a;
		if (bc_webgpu_prepare_priors(device, queue, last, scratch, weight_buf,
				classifier_off, classifier_bias_off, options->tissue_priors_prepared != NULL)) goto cleanup;
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
		wgpuCommandEncoderCopyBufferToBuffer(encoder, scratch, 0, readback, 0, output_bytes);
		WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
		wgpuQueueSubmit(queue, 1, &commands);
		wgpuCommandBufferRelease(commands); wgpuCommandEncoderRelease(encoder);
	} else
#endif
	{
		/* Even-numbered layers write act_a and odd ones act_b, so the
		   classifier's SOURCE is act_a when the last layer index is even.
		   orientation[1] has src = act_a, orientation[2] src = act_b. */
		int last = ((BC_MODEL_NHIDDEN - 1) & 1) ? 2 : 1;
		bc_params params = {0};
		params.weight_off = classifier_off;
		params.bias_off = classifier_bias_off;
		params.in_stride = (uint32_t)BC_CS;
		params.partial_groups = (uint32_t)BC_PARTIAL_GROUPS;
		wgpuQueueWriteBuffer(queue, param_buf, 0, &params, sizeof(params));

		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
		WGPUComputePassEncoder pass =
			wgpuCommandEncoderBeginComputePass(encoder, NULL);
		wgpuComputePassEncoderSetPipeline(pass, classify);
		wgpuComputePassEncoderSetBindGroup(pass, 0, groups[last], 0, NULL);
		wgpuComputePassEncoderDispatchWorkgroups(pass, BC_GROUPS_X, BC_DIM, BC_DIM);
		wgpuComputePassEncoderEnd(pass);
		wgpuCommandEncoderCopyBufferToBuffer(encoder, label_buf, 0, readback, 0,
			BC_NVOX * sizeof(uint32_t));
		WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
		wgpuQueueSubmit(queue, 1, &commands);
		wgpuCommandBufferRelease(commands);
		wgpuComputePassEncoderRelease(pass);
		wgpuCommandEncoderRelease(encoder);
	}

	/* Popped for real by the helper; the cleanup label must not pop again. */
	scope_pushed = 0;
	if (bc_wg_pop_and_map(device, readback, output_bytes)) goto cleanup;

	t_gpu = bc_now();
	if (bc_profiling())
		printfx("webgpu: pipelines %.3fs, pack %.3fs, upload %.3fs, "
			"gpu+readback %.3fs, conv %dx%d\n",
			t_pipelines - t_start, t_pack - t_pipelines,
			t_upload - t_pack, t_gpu - t_upload,
			wide ? BC_WG_WIDE_X : BC_WG_LOCAL_X, wide ? 1 : BC_WG_VOX);

	const uint32_t *labels = (const uint32_t *)wgpuBufferGetConstMappedRange(
		readback, 0, output_bytes);
	if (!labels) { printfx("webgpu: could not map the readback buffer\n"); goto cleanup; }
	if (pve) {
		memcpy(options->tissue_planes, labels, (size_t)output_bytes);
		if (options->tissue_priors_prepared) *options->tissue_priors_prepared = 1;
	} else for (size_t i = 0; i < BC_NVOX; i++) result[i] = (float)labels[i];
	wgpuBufferUnmap(readback);
	status = 0;

cleanup:
	bc_wg_drop_scopes(device, scope_pushed);
	free(packed);
	free(normalized);
	free(input16);
	for (int g = 0; g < 3; g++) wgpuBindGroupRelease(groups[g]);
	if (input_buf) wgpuBufferDestroy(input_buf);
	wgpuBufferRelease(input_buf);
	if (act_a) wgpuBufferDestroy(act_a);
	wgpuBufferRelease(act_a);
	if (act_b) wgpuBufferDestroy(act_b);
	wgpuBufferRelease(act_b);
	if (weight_buf) wgpuBufferDestroy(weight_buf);
	wgpuBufferRelease(weight_buf);
	if (param_buf) wgpuBufferDestroy(param_buf);
	wgpuBufferRelease(param_buf);
	if (moment_buf) wgpuBufferDestroy(moment_buf);
	wgpuBufferRelease(moment_buf);
	if (label_buf) wgpuBufferDestroy(label_buf);
	wgpuBufferRelease(label_buf);
	if (readback) wgpuBufferDestroy(readback);
	wgpuBufferRelease(readback);
	wgpuComputePipelineRelease(conv);
	wgpuComputePipelineRelease(moments);
	wgpuComputePipelineRelease(finish);
	wgpuComputePipelineRelease(norm);
	wgpuComputePipelineRelease(classify);
	wgpuPipelineLayoutRelease(pipeline_layout);
	wgpuBindGroupLayoutRelease(layout);
	wgpuShaderModuleRelease(module);
	wgpuShaderModuleRelease(wide_module);
	wgpuQueueRelease(queue);
	return status;
}

const bc_backend bc_webgpu_backend = {
	"webgpu",
	webgpu_available,
	webgpu_segment,
	/* The browser owns adapter choice here -- JS requests the adapter and hands
	   the device over -- so there is nothing for --adapter to select between.
	   A native Dawn/wgpu build could enumerate and would fill these in. */
	NULL,
	NULL
};
