/*
 * WebGL2 backend -- the fallback for browsers without WebGPU.
 *
 * Browser only. Unlike backend_webgpu.c there is no native build and none is
 * intended: WebGL2 is reached through emscripten's GLES3 binding, and a desktop
 * GL target would need a windowing system for an API that has no headless mode.
 *
 * The kernels are in webgl2_kernels.h. This file is the host: it packs weights,
 * builds programs and textures, walks the layers and reads the labels back. It
 * mirrors meshnet_cpu_infer(), which remains the specification, and is
 * deliberately structured like backend_webgpu.c so the two read as siblings.
 *
 * WHAT IS DIFFERENT FROM THE WebGPU BACKEND, and why:
 *
 *   - Everything is a fragment shader rendering into a texture. WebGL2 has no
 *     compute shaders, no storage buffers, no shared memory and no atomics.
 *   - The 16 channels of a voxel are one RGBA texel in each of FOUR RGBA16F 3D
 *     textures, written together through MRT. That preserves the property the
 *     WGSL cares most about -- one invocation computes all 16 outputs, so the
 *     27x16 input window is read once -- without needing the aliased vec4
 *     bindings, because an RGBA fetch IS a vec4 load.
 *   - A pass is one draw per z slice, because glFramebufferTextureLayer binds a
 *     single layer of a 3D texture. The 256 framebuffers are built once at init
 *     with their attachments already bound, so a slice costs three GL calls.
 *     Measured: an empty pass is 4.6 ms, 2.7% of a hidden convolution, so this
 *     structure is not what the time goes on.
 *   - The GroupNorm moments are three passes rather than two, none of which
 *     accumulates more than DIM values. See webgl2_kernels.h.
 *   - `norm` is NOT in place. WebGL2 forbids reading and writing one texture in
 *     a draw, so the layer schedule ping-pongs between two activation sets.
 *   - It is entirely SYNCHRONOUS. glReadPixels blocks, so there is no ASYNCIFY,
 *     no map callback, and none of the error-scope-versus-readback ordering
 *     race that backend_webgpu.c had to fix.
 */

#include "backend.h"
#include "backend_accel.h"
#include "model_specs.h"
#include "webgl2_kernels.h"
#include "f16.h"
#include "print.h"

#include <GLES3/gl3.h>
#include <emscripten.h>
#include <emscripten/html5_webgl.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if BC_WEBGL2_PHASE1
#include "webgl2_ref_conv.h"
#include <math.h>
#endif

#define BC_DIM BRAINCHOP_MODEL_DIM
#define BC_NVOX ((size_t)BC_DIM * BC_DIM * BC_DIM)
#define BC_CS BC_WGL_CS
/* Four RGBA textures of four channels each carry the 16-wide channel stride. */
#define BC_PLANES (BC_CS / 4)

/*
 * Texture units, DERIVED from BC_PLANES rather than the literals 4, 5 and 6.
 *
 * The activation planes occupy units 0..BC_PLANES-1, so at CS 24 they reach
 * unit 5 and collided with the weight texture on 4 and the partial sums on 5 --
 * the hidden layer drew with GL_INVALID_OPERATION while layer 0, which binds a
 * single source, was already exact. At CS 16 these are 4, 5 and 6 exactly as
 * before. The GLES 3.0 floor for MAX_TEXTURE_IMAGE_UNITS is 16, so 9 is safe.
 */
#define BC_UNIT_WTS   BC_PLANES
#define BC_UNIT_SUM   (BC_PLANES + 1)
#define BC_UNIT_SQ    (BC_PLANES + 2)

/*
 * WebGL's own error code, which GLES3/gl3.h does not declare.
 *
 * It is reported by glGetError EXACTLY ONCE and then never again until the
 * context is restored, so whoever consumes it owns the only notice there will
 * be. That is why nothing in this file discards errors blindly.
 */
#ifndef GL_CONTEXT_LOST_WEBGL
#define GL_CONTEXT_LOST_WEBGL 0x9242
#endif

_Static_assert(BC_CS % 4 == 0, "the channel stride must divide into RGBA texels");
_Static_assert(BC_DIM % 2 == 0,
	"the VOX=2 convolution writes z and z+1 together, so the volume depth must "
	"be even");
_Static_assert((size_t)BC_WGL_LAB_W * BC_WGL_LAB_W * 4 == BC_NVOX,
	"the label texture must hold exactly four voxels per texel with no slack");

/*
 * Activation storage. f16 by default, matching Metal and WebGPU.
 *
 * BC_WGL_F32 is the diagnostic that separates error inherent to f16 storage
 * from a defect in the kernels. Note the SHADERS do not change: texelFetch on a
 * sampler3D yields a vec4 whatever the internal format is, so unlike the WGSL
 * -- which had to switch its declared element type -- this is a host-side
 * change of internalformat and upload type only.
 */
#ifndef BC_WGL_F32
#define BC_WGL_F32 0
#endif
#if BC_WGL_F32
typedef float bc_store;
#define BC_STORE(x) (x)
#define BC_ACT_FORMAT GL_RGBA32F
#define BC_ACT_TYPE GL_FLOAT
#define BC_IN_FORMAT GL_R32F
#define BC_IN_TYPE GL_FLOAT
#else
typedef uint16_t bc_store;
#define BC_STORE(x) bc_f32_to_f16(x)
#define BC_ACT_FORMAT GL_RGBA16F
#define BC_ACT_TYPE GL_HALF_FLOAT
#define BC_IN_FORMAT GL_R16F
#define BC_IN_TYPE GL_HALF_FLOAT
#endif

/*
 * Voxels per fragment in the HIDDEN convolution, along z.
 *
 * 2 halves the weight traffic that is 91-93% of GPU time; 1 is the portable
 * path and the A/B control. This is only a CEILING: eight draw buffers are
 * needed and the GLES 3.0 floor is four, so the backend probes and falls back.
 * See BC_GLSL_CONV_HIDDEN2 for why the blocking is along z rather than x.
 */
#ifndef BC_WGL_VOX
#define BC_WGL_VOX 2
#endif
#if BC_WGL_VOX != 1 && BC_WGL_VOX != 2
#error "BC_WGL_VOX must be 1 or 2"
#endif

/*
 * Stage selection, for attribution only. Bit 0 conv, 1 moments, 2 norm,
 * 3 classify. Omitting a stage produces a WRONG result and exists solely to
 * measure where the time goes, exactly as BC_WEBGPU_STAGES does.
 */
#ifndef BC_WEBGL2_STAGES
#define BC_WEBGL2_STAGES 15
#endif
#define BC_STAGE_CONV (BC_WEBGL2_STAGES & 1)
#define BC_STAGE_MOMENTS (BC_WEBGL2_STAGES & 2)
#define BC_STAGE_NORM (BC_WEBGL2_STAGES & 4)
#define BC_STAGE_CLASSIFY (BC_WEBGL2_STAGES & 8)

/*
 * Stage timing. Compile-time, not an environment variable: emscripten
 * snapshots the environment into __environ during runtime init, before any JS
 * could set Module.ENV, so getenv cannot be flipped from the harness. Same trap
 * BC_WEBGPU_PROFILE documents; build a separate profiling module.
 */
#ifndef BC_WEBGL2_PROFILE
#define BC_WEBGL2_PROFILE 0
#endif

/* ------------------------------------------------------------------ context */

/*
 * The one platform-dependent step, and the counterpart of
 * backend_webgpu.c's acquire_device().
 *
 * JavaScript creates the context and hands it over through
 * Module.preinitializedWebGLContext (-sGL_PREINITIALIZED_CONTEXT=1), so every
 * capability refusal can be phrased in the caller's terms in TypeScript rather
 * than here. Verified on Chromium and Firefox.
 *
 * The flag does NOT make the context current by itself: create_context must
 * still be called, and returns the JS context instead of creating one. Assuming
 * otherwise yields handle 0 and a null GL function table, which looks exactly
 * like the flag being unsupported.
 *
 * THE TARGET IS NOT "#canvas", and that is load-bearing. Emscripten 6 defaults
 * DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR to true, which makes the target
 * a plain `document.querySelector` -- Module.canvas is never consulted. So
 * "#canvas" would require the host page to contain an element with that id,
 * which is both a collision risk (NiiVue pages have canvases) and impossible in
 * a Worker, where there is no document at all.
 *
 * Instead the host registers its canvas under a namespaced key in emscripten's
 * `specialHTMLTargets` map, which findEventTarget checks BEFORE the DOM. That
 * works for a detached canvas on the main thread and for an OffscreenCanvas in
 * a Worker, and cannot collide with anything the page owns.
 */
#define BC_WEBGL2_TARGET "#brainchop-webgl2"
static EMSCRIPTEN_WEBGL_CONTEXT_HANDLE acquire_context(void) {
	static EMSCRIPTEN_WEBGL_CONTEXT_HANDLE cached = 0;
	static int tried = 0;
	if (tried) {
		if (cached > 0) emscripten_webgl_make_context_current(cached);
		return cached;
	}
	tried = 1;

	EmscriptenWebGLContextAttributes attrs;
	emscripten_webgl_init_context_attributes(&attrs);
	attrs.majorVersion = 2;
	attrs.minorVersion = 0;
	attrs.alpha = EM_FALSE;
	attrs.depth = EM_FALSE;
	attrs.stencil = EM_FALSE;
	attrs.antialias = EM_FALSE;
	attrs.preserveDrawingBuffer = EM_FALSE;
	attrs.powerPreference = EM_WEBGL_POWER_PREFERENCE_HIGH_PERFORMANCE;
	attrs.failIfMajorPerformanceCaveat = EM_FALSE;

	cached = emscripten_webgl_create_context(BC_WEBGL2_TARGET, &attrs);
	if (cached <= 0) return 0;
	if (emscripten_webgl_make_context_current(cached) != EMSCRIPTEN_RESULT_SUCCESS) {
		cached = 0;
		return 0;
	}
	return cached;
}

/* ------------------------------------------------------------------ helpers */

/*
 * Drain the error queue and report.
 *
 * Called after each init step rather than once at the end, because the queue
 * holds only the errors it holds and a later failure would be attributed to the
 * wrong call. Returns 1 if anything was pending. WebGL2 has nothing like
 * WebGPU's error scopes, so this plus glCheckFramebufferStatus is the whole
 * mechanism -- and per the standing rule, any of it firing must make segment()
 * fail rather than return a blank volume with status 0.
 */
static int gl_failed(const char *where) {
	GLenum first = GL_NO_ERROR;
	int count = 0;
	for (;;) {
		GLenum e = glGetError();
		if (e == GL_NO_ERROR) break;
		if (!count) first = e;
		if (++count > 8) break;
	}
	if (first == GL_CONTEXT_LOST_WEBGL)
		printfx("webgl2: the GL context was lost during %s; the device or its "
			"driver reset. No result can be trusted after this.\n", where);
	else if (count)
		printfx("webgl2: %s failed (GL error 0x%04x%s)\n", where,
			(unsigned)first, count > 1 ? ", and more pending" : "");
	return count != 0;
}

/*
 * A real GPU sync, for per-stage timing only.
 *
 * glFinish alone is NOT one here: measured, 256 draws "completed" in 0.1 ms,
 * which they cannot have. WebGL defers and emscripten's glFinish is just
 * another queued command. A one-pixel readback is a genuine round trip and
 * forces the queue to drain. Without it any timing in this file measures
 * nothing, which is worse than not measuring.
 *
 * The whole-run number needs no help: it is taken after the label readback,
 * which is itself a full sync. Only the per-stage breakdown has to insert
 * these, and each one stalls the pipeline -- which is why they are compiled out
 * unless BC_WEBGL2_PROFILE asks for them.
 */
/*
 * The read type must match the attachment: GL_FLOAT from the RGBA8 label
 * texture is GL_INVALID_OPERATION, not a lenient conversion. The first
 * profiling build failed exactly there, which is a small argument for running
 * the diagnostic builds rather than only shipping the default one.
 */
#if BC_WEBGL2_PROFILE
static void gl_sync(GLuint fbo, GLenum type) {
	unsigned char bytes[4];
	float pixel[4];
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glFinish();
	if (type == GL_UNSIGNED_BYTE)
		glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, bytes);
	else
		glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, pixel);
}
#endif

static GLuint compile_shader(GLenum type, const char *source, const char *name) {
	GLuint shader = glCreateShader(type);
	if (!shader) return 0;
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);
	GLint ok = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024];
		GLsizei len = 0;
		glGetShaderInfoLog(shader, (GLsizei)sizeof(log) - 1, &len, log);
		log[len < (GLsizei)sizeof(log) ? len : (GLsizei)sizeof(log) - 1] = '\0';
		printfx("webgl2: %s %s shader failed to compile: %s\n", name,
			type == GL_VERTEX_SHADER ? "vertex" : "fragment", log);
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}

/*
 * Link a program and drop the shaders.
 *
 * linkProgram is where the MRT output locations and the uniform interface are
 * validated, which is stricter than anything the WGSL path has; make check-glsl
 * runs exactly this in a real browser so a mismatch is caught at build time
 * rather than as a blank attachment at run time.
 */
static GLuint link_program(const char *vertex_source, const char *fragment_source,
		const char *name) {
	GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source, name);
	if (!vertex) return 0;
	GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source, name);
	if (!fragment) { glDeleteShader(vertex); return 0; }

	GLuint program = glCreateProgram();
	glAttachShader(program, vertex);
	glAttachShader(program, fragment);
	glLinkProgram(program);
	glDeleteShader(vertex);
	glDeleteShader(fragment);

	GLint ok = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[1024];
		GLsizei len = 0;
		glGetProgramInfoLog(program, (GLsizei)sizeof(log) - 1, &len, log);
		log[len < (GLsizei)sizeof(log) ? len : (GLsizei)sizeof(log) - 1] = '\0';
		printfx("webgl2: %s program failed to link: %s\n", name, log);
		glDeleteProgram(program);
		return 0;
	}
	return program;
}

/* Bind a sampler uniform to a texture unit once, at setup. */
static void bind_unit(GLuint program, const char *name, int unit) {
	GLint location = glGetUniformLocation(program, name);
	if (location >= 0) glUniform1i(location, unit);
}

static GLuint make_tex3d(int width, int height, int depth, GLenum internal) {
	GLuint tex = 0;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexStorage3D(GL_TEXTURE_3D, 1, internal, width, height, depth);
	return tex;
}

static GLuint make_tex2d(int width, int height, GLenum internal) {
	GLuint tex = 0;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexStorage2D(GL_TEXTURE_2D, 1, internal, width, height);
	return tex;
}

/*
 * BC_PLANES attachments, BUILT rather than listed.
 *
 * These were four and eight literals against an array sized BC_PLANES and
 * 2 * BC_PLANES. At CS 16 those agree; at CS 24 the array is six long with four
 * initializers, so the last two entries are a zero-valued GLenum -- GL_NONE --
 * and framebuffer setup fails with GL_INVALID_ENUM. The enums are consecutive
 * by definition, so derive them.
 */
/* One array, long enough for the VOX=2 set: its first BC_PLANES entries ARE the
   single-slice set, and every consumer passes an explicit count or indexes
   below BC_PLANES. */
static GLenum g_draw_buffers[2 * BC_PLANES];
#define g_draw_buffers8 g_draw_buffers

_Static_assert(GL_COLOR_ATTACHMENT1 == GL_COLOR_ATTACHMENT0 + 1,
	"the attachment enums must be consecutive for this to derive them");

static void init_draw_buffers(void) {
	for (int i = 0; i < 2 * BC_PLANES; i++)
		g_draw_buffers[i] = (GLenum)(GL_COLOR_ATTACHMENT0 + i);
}

/* ------------------------------------------------------------------ weights */

/*
 * The weight packing lives in accel_pack_weights.h, shared with the WebGPU
 * backend. It was a verbatim copy of that file's version whose only real
 * difference was the row padding, which is now the `pad_to` argument; see the
 * header for why sharing it matters more than the duplication saved.
 */
#include "accel_pack_weights.h"

/* ------------------------------------------------------------- capabilities */

/*
 * Refuse, naming the limit that was short, rather than failing at draw time.
 *
 * The floors checked here are the ones the activation layout was designed
 * against: a 256^3 3D texture and four colour attachments are both exactly the
 * GLES 3.0 minimum, so a conformant implementation cannot be short of them --
 * but "cannot" is worth asserting when the alternative is an incomplete
 * framebuffer 200 lines later. EXT_color_buffer_float is the real risk: it is
 * an extension, not core, and without it no float texture is renderable at all.
 */
static int check_capabilities(void) {
	GLint max3d = 0, draw_buffers = 0, attachments = 0, max2d = 0, viewport[2] = {0, 0};
	glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max3d);
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &draw_buffers);
	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &attachments);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max2d);
	glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewport);

	int ok = 1;
	if (!emscripten_webgl_enable_extension(emscripten_webgl_get_current_context(),
			"EXT_color_buffer_float")) {
		printfx("webgl2: this device lacks EXT_color_buffer_float, which the "
			"model requires to render into float textures\n");
		ok = 0;
	}
	if (max3d < BC_DIM) {
		printfx("webgl2: MAX_3D_TEXTURE_SIZE is %d, but the model needs %d\n",
			(int)max3d, BC_DIM);
		ok = 0;
	}
	if (draw_buffers < BC_PLANES || attachments < BC_PLANES) {
		printfx("webgl2: this device allows %d draw buffers and %d colour "
			"attachments, but the model writes %d at once\n",
			(int)draw_buffers, (int)attachments, BC_PLANES);
		ok = 0;
	}
	/* The label texture is the widest 2D surface, and the classifier renders a
	   full-size viewport over it. */
	if (max2d < BC_WGL_LAB_W || viewport[0] < BC_WGL_LAB_W ||
			viewport[1] < BC_WGL_LAB_W) {
		printfx("webgl2: MAX_TEXTURE_SIZE is %d and MAX_VIEWPORT_DIMS %dx%d, "
			"but the label texture is %dx%d\n", (int)max2d,
			(int)viewport[0], (int)viewport[1], BC_WGL_LAB_W, BC_WGL_LAB_W);
		ok = 0;
	}
	return ok;
}

static int webgl2_available(void) {
	return acquire_context() > 0;
}

/* --------------------------------------------- layer-0 verification build */

#if BC_WEBGL2_PHASE1
/*
 * Compare one slice of the GPU's layer-0 output against the reference.
 *
 * The comparison is in f16 ULPs, not absolute error, and that is the point.
 * The GPU STORES its result as f16, so a raw float difference mostly measures
 * output quantisation -- which is expected and uninteresting. Rounding the
 * reference to f16 and differencing the bit patterns instead asks the sharp
 * question: given the same inputs, did the two produce the same f16?
 *
 * A non-zero tail is expected and is not a defect: GLSL ES 3.00 has no fma()
 * and no `precise`, so the driver may contract a*b+c, changing the result by
 * one rounding before it is stored. Anything beyond a couple of ULPs is a
 * plumbing bug -- a transposed tap, a swapped attachment, a wrong dilation --
 * and those show up as gross errors, not as a widened tail.
 */
typedef struct {
	size_t total;
	size_t exact;
	size_t within1;
	int max_ulp;
	double max_abs;          /* against the UNROUNDED reference, for scale */
	double max_value;
	int worst_x, worst_y, worst_c;
} bc_slice_stats;

static void compare_slice(const float *gpu, const float *ref, bc_slice_stats *out) {
	memset(out, 0, sizeof(*out));
	for (int y = 0; y < BC_DIM; y++)
	for (int x = 0; x < BC_DIM; x++)
	for (int c = 0; c < BC_CS; c++) {
		size_t i = (((size_t)y * BC_DIM) + x) * BC_CS + c;
		float g = gpu[i], r = ref[i];
		uint16_t gh = bc_f32_to_f16(g), rh = bc_f32_to_f16(r);
		int ulp = (int)gh - (int)rh;
		if (ulp < 0) ulp = -ulp;
		/* Sign-magnitude means +0 and -0 differ by 0x8000 as integers; they are
		   the same number, so normalise before differencing. */
		if ((gh & 0x7fffu) == 0 && (rh & 0x7fffu) == 0) ulp = 0;
		out->total++;
		if (ulp == 0) out->exact++;
		if (ulp <= 1) out->within1++;
		if (ulp > out->max_ulp) {
			out->max_ulp = ulp;
			out->worst_x = x; out->worst_y = y; out->worst_c = c;
		}
		double abs_diff = fabs((double)g - (double)r);
		if (abs_diff > out->max_abs) out->max_abs = abs_diff;
		if (fabs((double)r) > out->max_value) out->max_value = fabs((double)r);
	}
}

static void report_slice(int z, const bc_slice_stats *s) {
	printfx("webgl2 layer-0: z=%3d  exact %zu/%zu (%.4f%%)  within 1 ulp %.4f%%  "
		"max %d ulp at (x=%d,y=%d,c=%d)  max|dz|=%.3e  peak|ref|=%.3e%s\n",
		z, s->exact, s->total, 100.0 * (double)s->exact / (double)s->total,
		100.0 * (double)s->within1 / (double)s->total,
		s->max_ulp, s->worst_x, s->worst_y, s->worst_c,
		s->max_abs, s->max_value, s->max_value == 0.0 ? "  VACUOUS" : "");
}
#endif

#include "webgl2_pve.h"

/* ------------------------------------------------------------------ segment */

static int webgl2_segment(const unsigned char *conformed, float *result,
		const float *weights, const bc_inference_options *options) {
	if (!conformed || !result || !weights) return 1;
	/* An ROI or per-class export would need kernels this backend does not
	   have; report "unsupported" so the caller can fall back rather than
	   silently returning a whole-volume answer to a cropped question. */
	if (options && (options->roi_enabled || options->class_writer)) return 2;

#if !BC_HAS_PVE
    if (options && options->tissue_planes) return 2;
#endif

	/*
	 * Everything releasable, declared up front and 0 until it exists, so the
	 * single `cleanup` label can release exactly what was created. Same
	 * discipline backend_webgpu.c arrived at after ten failure paths were found
	 * to leak two 512 MiB buffers each.
	 */
	int status = 1;
	init_draw_buffers();
	GLuint input_tex = 0, weight_tex = 0, label_tex = 0;
	GLuint act[2][BC_PLANES];
	GLuint part_sum = 0, part_sq = 0, red_sum = 0, red_sq = 0;
	GLuint mom_mean = 0, mom_inv = 0;
	GLuint act_fbo[2][BC_DIM];
	GLuint vox2_fbo[BC_DIM / 2];
	GLuint part_fbo = 0, red_fbo = 0, mom_fbo = 0, label_fbo = 0;
	GLuint p_conv_first = 0, p_conv_hidden = 0, p_conv_hidden2 = 0,
		p_moments_a = 0, p_moments_b = 0,
		p_moments_finish = 0, p_norm = 0, p_classify = 0, vao = 0;
	int fbo_sets = 0, vox2_fbos = 0, use_vox2 = 0;
	float *normalized = NULL, *ref_input = NULL;
	bc_store *packed = NULL, *input16 = NULL;
	unsigned char *labels = NULL;
	float *plane = NULL, *gpu_slice = NULL, *ref_slice = NULL, *ref_kernel = NULL;
	for (int s = 0; s < 2; s++)
		for (int i = 0; i < BC_PLANES; i++) act[s][i] = 0;

	if (acquire_context() <= 0) {
		printfx("webgl2: no context; the host must hand one over\n");
		return 1;
	}
	if (!check_capabilities()) goto cleanup;
	/*
	 * Drain what the capability probe left -- queries only, so nothing here
	 * should error -- and refuse on ANYTHING, not merely on context loss.
	 *
	 * Narrowed to match the framebuffer drain below, which the 2026-08-04 audit
	 * rewrote after finding that a wide drain is exactly how a single-shot
	 * notice gets consumed. These two predated that discipline. GL_OUT_OF_MEMORY
	 * is the case that matters: it is reachable here on a weak stack, and
	 * swallowing it leaves every later gl_failed() reading clean.
	 */
	if (gl_failed("capability probe")) goto cleanup;

	/* ---------------------------------------------------------- host prep */

	normalized = bc_accel_normalize_input(conformed);
	if (!normalized) { printfx("webgl2: out of memory normalizing input\n"); goto cleanup; }

	input16 = (bc_store *)malloc(BC_NVOX * sizeof(*input16));
	if (!input16) { printfx("webgl2: out of memory\n"); goto cleanup; }
	for (size_t i = 0; i < BC_NVOX; i++) input16[i] = BC_STORE(normalized[i]);

#if BC_WEBGL2_PHASE1
	/*
	 * The reference must see the values the GPU actually saw, not the values
	 * the CPU would have used. Feeding it `normalized` would fold f16 input
	 * rounding into the comparison and make a plumbing bug indistinguishable
	 * from storage precision.
	 */
	ref_input = (float *)malloc(BC_NVOX * sizeof(*ref_input));
	if (!ref_input) { printfx("webgl2: out of memory\n"); goto cleanup; }
	for (size_t i = 0; i < BC_NVOX; i++)
#if BC_WGL_F32
		ref_input[i] = input16[i];
#else
		ref_input[i] = bc_f16_to_f32(input16[i]);
#endif
#endif
	free(normalized);
	normalized = NULL;

	bc_layer_offsets offsets[BC_MODEL_NHIDDEN];
	uint32_t classifier_off = 0, classifier_bias_off = 0;
	size_t weight_count = 0;
	/* One texture row is BC_WGL_WTEX_W RGBA texels, i.e. that many elements
	   times four; padding to it lets the upload be a single texSubImage2D. */
	const size_t per_row = (size_t)BC_WGL_WTEX_W * 4;
	packed = bc_pack_weights(weights, offsets, &classifier_off,
		&classifier_bias_off, &weight_count, per_row);
	int weight_rows = (int)((weight_count + per_row - 1) / per_row);
	if (!packed) { printfx("webgl2: out of memory packing weights\n"); goto cleanup; }

	/*
	 * calloc, not malloc. glReadPixels is the only writer; if it ever no-ops
	 * without raising a detectable error, malloc would make 16.7 MB of
	 * uninitialised heap the segmentation. Zero is not correct either, but it
	 * is diagnosable -- an empty mask is obviously wrong, brain-shaped garbage
	 * is not.
	 */
	labels = (unsigned char *)calloc(BC_NVOX, 1);
	if (!labels) { printfx("webgl2: out of memory\n"); goto cleanup; }

	/*
	 * P is always "the current activation" and Q the convolution's scratch.
	 * Declared here because the framebuffer setup below already needs to know
	 * which set the eight-attachment convolution writes into.
	 */
	enum { P = 0, Q = 1 };

	/* ------------------------------------------------------------ textures */

	input_tex = make_tex3d(BC_DIM, BC_DIM, BC_DIM, BC_IN_FORMAT);
	if (gl_failed("input texture allocation")) goto cleanup;
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, BC_DIM, BC_DIM, BC_DIM,
		GL_RED, BC_IN_TYPE, input16);
	if (gl_failed("input texture upload")) goto cleanup;

	weight_tex = make_tex2d(BC_WGL_WTEX_W, weight_rows, BC_ACT_FORMAT);
	if (gl_failed("weight texture allocation")) goto cleanup;
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, BC_WGL_WTEX_W, weight_rows,
		GL_RGBA, BC_ACT_TYPE, packed);
	if (gl_failed("weight texture upload")) goto cleanup;

	/*
	 * Two activation sets, 512 MiB each. This is the allocation the whole
	 * design is sized around and the one most likely to fail on a small device.
	 * 1.0 GiB was probed as allocatable AND writable -- allocation alone is not
	 * the test, since a driver may defer the failure to first write.
	 */
	for (int s = 0; s < 2; s++)
		for (int i = 0; i < BC_PLANES; i++) {
			act[s][i] = make_tex3d(BC_DIM, BC_DIM, BC_DIM, BC_ACT_FORMAT);
			if (gl_failed("activation texture allocation")) goto cleanup;
		}

	/* Moment scratch. Pass A writes one partial per (x, y) column per group. */
	part_sum = make_tex2d(BC_DIM, BC_DIM * BC_PLANES, GL_RGBA32F);
	part_sq = make_tex2d(BC_DIM, BC_DIM * BC_PLANES, GL_RGBA32F);
	red_sum = make_tex2d(BC_DIM, BC_PLANES, GL_RGBA32F);
	red_sq = make_tex2d(BC_DIM, BC_PLANES, GL_RGBA32F);
	mom_mean = make_tex2d(1, BC_PLANES, GL_RGBA32F);
	mom_inv = make_tex2d(1, BC_PLANES, GL_RGBA32F);
	label_tex = make_tex2d(BC_WGL_LAB_W, BC_WGL_LAB_W, GL_RGBA8);
	if (gl_failed("scratch texture allocation")) goto cleanup;

	/* --------------------------------------------------------- framebuffers */

	/*
	 * One framebuffer per z slice per activation set, attachments bound once.
	 * Rebinding four layers per draw instead would quadruple the per-slice call
	 * count for no benefit; 512 framebuffer objects are cheap and immutable.
	 */
	for (int s = 0; s < 2; s++) {
		glGenFramebuffers(BC_DIM, act_fbo[s]);
		fbo_sets = s + 1;
		for (int z = 0; z < BC_DIM; z++) {
			glBindFramebuffer(GL_FRAMEBUFFER, act_fbo[s][z]);
			for (int i = 0; i < BC_PLANES; i++)
				glFramebufferTextureLayer(GL_FRAMEBUFFER, g_draw_buffers[i],
					act[s][i], 0, z);
			glDrawBuffers(BC_PLANES, g_draw_buffers);
			/* EVERY slice, not a sample. This is the authoritative test that
			   the attachments are usable, and the setup path below downgrades
			   the error queue to a warning on the strength of it -- so it had
			   better cover all 512 framebuffers rather than two of them. */
			GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (st != GL_FRAMEBUFFER_COMPLETE) {
				printfx("webgl2: framebuffer incomplete (0x%04x) at set %d "
					"slice %d with %d RGBA float 3D layers attached\n",
					(unsigned)st, s, z, BC_PLANES);
				goto cleanup;
			}
		}
	}

	/*
	 * The VOX=2 convolution needs eight draw buffers where the spec guarantees
	 * four, so it is a probed upgrade rather than the design. Chromium and
	 * Firefox both report eight; a device with four still runs, one voxel per
	 * fragment.
	 */
	{
		GLint db = 0, ca = 0;
		glGetIntegerv(GL_MAX_DRAW_BUFFERS, &db);
		glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &ca);
		use_vox2 = BC_WGL_HAS_VOX2 && (BC_WGL_VOX == 2) &&
			db >= 2 * BC_PLANES && ca >= 2 * BC_PLANES;
	}
	if (use_vox2) {
		/* Layer z and layer z+1 of one texture are different IMAGES, so both
		   may be attached to the same framebuffer. That is what lets the
		   activation layout stay exactly as the other kernels expect it. */
		glGenFramebuffers(BC_DIM / 2, vox2_fbo);
		vox2_fbos = BC_DIM / 2;
		for (int k = 0; k < BC_DIM / 2; k++) {
			glBindFramebuffer(GL_FRAMEBUFFER, vox2_fbo[k]);
			for (int i = 0; i < BC_PLANES; i++) {
				glFramebufferTextureLayer(GL_FRAMEBUFFER, g_draw_buffers8[i],
					act[Q][i], 0, 2 * k);
				glFramebufferTextureLayer(GL_FRAMEBUFFER,
					g_draw_buffers8[BC_PLANES + i], act[Q][i], 0, 2 * k + 1);
			}
			glDrawBuffers(2 * BC_PLANES, g_draw_buffers8);
			/*
			 * EVERY one, not just k == 0. The note printed after framebuffer
			 * setup downgrades a lingering GL error to a warning ON THE
			 * STRENGTH OF completeness, and quotes the total count while doing
			 * it -- so sampling one of 128 here made that sentence assert
			 * something the code had not established. Cost is 127 extra
			 * queries once per run, against 39204 GL calls.
			 */
			if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
				/* Not fatal: fall back rather than refuse, since the
				   4-attachment path computes the same thing. */
				printfx("webgl2: eight-attachment framebuffer incomplete; "
					"using one voxel per fragment\n");
				glDeleteFramebuffers(BC_DIM / 2, vox2_fbo);
				vox2_fbos = 0;
				use_vox2 = 0;
				/*
				 * Clear the incompleteness this fallback is answering, and
				 * NOTHING else. GL_INVALID_FRAMEBUFFER_OPERATION is what an
				 * incomplete framebuffer raises and is the error being handled;
				 * anything else -- a context loss, an OOM -- means the fallback
				 * is not the right answer and the run must stop.
				 */
				for (GLenum e = glGetError(); e != GL_NO_ERROR; e = glGetError())
					if (e != GL_INVALID_FRAMEBUFFER_OPERATION) {
						printfx("webgl2: %s while building the eight-attachment "
							"framebuffers; refusing to continue\n",
							e == GL_CONTEXT_LOST_WEBGL
								? "the GL context was lost"
								: "an unexpected GL error");
						goto cleanup;
					}
				break;
			}
		}
	}

	{
		const GLenum two[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		const GLenum one[1] = { GL_COLOR_ATTACHMENT0 };
		struct { GLuint *fbo; GLuint a, b; } pairs[3] = {
			{ &part_fbo, part_sum, part_sq },
			{ &red_fbo, red_sum, red_sq },
			{ &mom_fbo, mom_mean, mom_inv },
		};
		for (int i = 0; i < 3; i++) {
			glGenFramebuffers(1, pairs[i].fbo);
			glBindFramebuffer(GL_FRAMEBUFFER, *pairs[i].fbo);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				GL_TEXTURE_2D, pairs[i].a, 0);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
				GL_TEXTURE_2D, pairs[i].b, 0);
			glDrawBuffers(2, two);
			GLenum mst = glCheckFramebufferStatus(GL_FRAMEBUFFER);
			if (mst != GL_FRAMEBUFFER_COMPLETE) {
				printfx("webgl2: moment framebuffer %d incomplete (0x%04x)\n",
					i, (unsigned)mst);
				goto cleanup;
			}
		}
		glGenFramebuffers(1, &label_fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, label_fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
			GL_TEXTURE_2D, label_tex, 0);
		glDrawBuffers(1, one);
		GLenum lst = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (lst != GL_FRAMEBUFFER_COMPLETE) {
			printfx("webgl2: label framebuffer incomplete (0x%04x)\n",
				(unsigned)lst);
			goto cleanup;
		}
	}
	/*
	 * Framebuffer setup reports by COMPLETENESS, not by the error queue.
	 *
	 * ATTRIBUTED, not guessed. On Firefox 151 / macOS,
	 * glCheckFramebufferStatus returns GL_FRAMEBUFFER_COMPLETE and RAISES
	 * GL_INVALID_FRAMEBUFFER_OPERATION (0x0506) at the same time, when four
	 * RGBA16F 3D layers are attached and drawBuffers covers all four. With a
	 * single attachment it does not, and Chromium does not at either width.
	 * Reproduced in plain JavaScript with no emscripten involved --
	 * test/webgl2_spike/probe_page.html, probe P6 -- draining the queue after
	 * every call, so the source is the completeness query itself and not our
	 * attachment or drawBuffers calls, which each leave it clean.
	 *
	 * The run then produces output BIT-IDENTICAL to Chromium's: 33 differing
	 * voxels against the CPU reference on both browsers.
	 *
	 * So this is reported and stepped over rather than refused. That is a
	 * deliberate exception to the rule that a GL error must fail the run, and
	 * it is bounded: completeness is checked for all 512 framebuffers above,
	 * and every draw and readback afterwards still treats the error queue as
	 * fatal. Refusing here would mean the backend does not run on the one
	 * browser it exists for, on the strength of an error no operation of ours
	 * can be shown to have caused.
	 */
	{
		/*
		 * Tolerate GL_INVALID_FRAMEBUFFER_OPERATION and NOTHING ELSE.
		 *
		 * An earlier version drained the queue and continued on whatever it
		 * found, which was far wider than the Firefox bug it was written for
		 * and quietly created the failure this project treats as cardinal.
		 * glGetError reports GL_CONTEXT_LOST_WEBGL exactly ONCE; swallowing it
		 * here disarms every gl_failed() for the rest of the run, so each
		 * subsequent check reads clean while every GL call is a no-op, and
		 * segment() returns a whole-volume result with status 0. The same
		 * applies to GL_OUT_OF_MEMORY, which is a live possibility precisely
		 * here: 1.1 GiB of textures were allocated immediately above, on the
		 * weak stacks this backend exists for.
		 */
		GLenum unexpected = GL_NO_ERROR;
		int tolerated = 0;
		for (GLenum e = glGetError(); e != GL_NO_ERROR; e = glGetError()) {
			if (e == GL_INVALID_FRAMEBUFFER_OPERATION) { tolerated++; continue; }
			if (unexpected == GL_NO_ERROR) unexpected = e;
		}
		if (unexpected != GL_NO_ERROR) {
			if (unexpected == GL_CONTEXT_LOST_WEBGL)
				printfx("webgl2: the GL context was lost during framebuffer "
					"setup; refusing to continue\n");
			else
				printfx("webgl2: framebuffer setup failed (GL error 0x%04x)\n",
					(unsigned)unexpected);
			goto cleanup;
		}
		if (tolerated)
			printfx("webgl2: note - glCheckFramebufferStatus left "
				"GL_INVALID_FRAMEBUFFER_OPERATION pending %d time(s) while "
				"reporting every one of %d framebuffers complete; this is the "
				"known Firefox behaviour, continuing\n",
				tolerated, 2 * BC_DIM + vox2_fbos + 4);
	}

	/* ------------------------------------------------------------ programs */

	p_conv_first = link_program(bc_glsl_vertex, bc_glsl_conv_first, "conv_first");
	p_conv_hidden = link_program(bc_glsl_vertex, bc_glsl_conv_hidden, "conv_hidden");
	if (use_vox2) {
		p_conv_hidden2 = link_program(bc_glsl_vertex, bc_glsl_conv_hidden2,
			"conv_hidden2");
		if (!p_conv_hidden2) {
			/*
			 * Fall back rather than refuse, matching the eight-attachment
			 * framebuffer check 100 lines up: both are the same capability
			 * question, and the 4-attachment path computes the identical
			 * result -- measured 33/710/94 either way. A driver that reports
			 * eight draw buffers but rejects `layout(location = 7)` should
			 * cost speed, not the whole run.
			 */
			printfx("webgl2: the eight-attachment convolution did not link; "
				"using one voxel per fragment\n");
			use_vox2 = 0;
			if (vox2_fbos) {
				glDeleteFramebuffers(vox2_fbos, vox2_fbo);
				vox2_fbos = 0;
			}
		}
	}
	p_moments_a = link_program(bc_glsl_vertex, bc_glsl_moments_a, "moments_a");
	p_moments_b = link_program(bc_glsl_vertex, bc_glsl_moments_b, "moments_b");
	p_moments_finish = link_program(bc_glsl_vertex, bc_glsl_moments_finish,
		"moments_finish");
	p_norm = link_program(bc_glsl_vertex, bc_glsl_norm, "norm");
	p_classify = link_program(bc_glsl_vertex, bc_glsl_classify, "classify");
	if (!p_conv_first || !p_conv_hidden || !p_moments_a || !p_moments_b ||
			!p_moments_finish || !p_norm || !p_classify) goto cleanup;

	/*
	 * Texture units, assigned once. Units 0-3 carry the 3D activations, 4 the
	 * weights and 5-6 the 2D moment textures.
	 *
	 * NOT an invariant that no unit holds two targets: make_tex2d binds on
	 * whatever unit is active, which is unit 0 during setup, so unit 0 also
	 * keeps a stale GL_TEXTURE_2D binding for the run. That is harmless -- no
	 * program samples unit 0 as 2D, and GL allows one unit to have a binding
	 * per target -- but an earlier version of this comment claimed the
	 * separation as a rule, which it never was.
	 */
	glUseProgram(p_conv_first);
	bind_unit(p_conv_first, "src", 0);
	bind_unit(p_conv_first, "wts", BC_UNIT_WTS);
	glUseProgram(p_conv_hidden);
	for (int i = 0; i < BC_PLANES; i++) {
		char name[4] = { 's', (char)('0' + i), '\0', '\0' };
		bind_unit(p_conv_hidden, name, i);
	}
	bind_unit(p_conv_hidden, "wts", BC_UNIT_WTS);
	if (use_vox2) {
		glUseProgram(p_conv_hidden2);
		for (int i = 0; i < BC_PLANES; i++) {
			char name[4] = { 's', (char)('0' + i), '\0', '\0' };
			bind_unit(p_conv_hidden2, name, i);
		}
		bind_unit(p_conv_hidden2, "wts", BC_UNIT_WTS);
	}
	glUseProgram(p_moments_a);
	bind_unit(p_moments_a, "src", 0);
	glUseProgram(p_moments_b);
	bind_unit(p_moments_b, "pSum", BC_UNIT_SUM);
	bind_unit(p_moments_b, "pSq", BC_UNIT_SQ);
	glUseProgram(p_moments_finish);
	bind_unit(p_moments_finish, "pSum", BC_UNIT_SUM);
	bind_unit(p_moments_finish, "pSq", BC_UNIT_SQ);
	glUseProgram(p_norm);
	for (int i = 0; i < BC_PLANES; i++) {
		char name[4] = { 's', (char)('0' + i), '\0', '\0' };
		bind_unit(p_norm, name, i);
	}
	bind_unit(p_norm, "pMean", BC_UNIT_SUM);
	bind_unit(p_norm, "pInv", BC_UNIT_SQ);
	bind_unit(p_norm, "wts", BC_UNIT_WTS);
	glUseProgram(p_classify);
	for (int i = 0; i < BC_PLANES; i++) {
		char name[4] = { 's', (char)('0' + i), '\0', '\0' };
		bind_unit(p_classify, name, i);
	}
	bind_unit(p_classify, "wts", BC_UNIT_WTS);

	/* An explicit VAO rather than relying on the default one. The draw uses no
	   attributes at all -- the vertex shader synthesises its triangle from
	   gl_VertexID -- but core profiles require a bound VAO for any draw. */
	/*
	 * GL_DITHER is enabled by default, and the label texture is RGBA8 -- the one
	 * buffer that IS the answer. classify encodes a class as float(best)/255.0,
	 * which is not exactly on a UNORM8 level in f32, so a conforming dither
	 * implementation is free to pick either bracketing value and shift a voxel
	 * by one class, position-dependently. No tested stack does (Chromium/ANGLE
	 * and Firefox agree at 33/710/94), but Mesa and llvmpipe are untested and
	 * are what this backend exists for. One call removes the freedom.
	 */
	glDisable(GL_DITHER);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	/* The weight texture never moves off unit 4. */
	glActiveTexture(GL_TEXTURE0 + BC_UNIT_WTS);
	glBindTexture(GL_TEXTURE_2D, weight_tex);
	if (gl_failed("program setup")) goto cleanup;

	/* ------------------------------------------------------------ the model */

	/*
	 * P is always "the current activation" and Q the convolution's scratch:
	 *
	 *   conv    : P -> Q     (or input -> Q for layer 0)
	 *   moments : Q          -> partials -> mean, inverse stddev
	 *   norm    : Q -> P     normalised, affine, activated
	 *
	 * so the invariant is simple and no third set is needed. Write traffic is
	 * the same as an in-place norm; only the destination differs.
	 */
	const int *dilation = bc_accel_spec()->dilation;
	/*
	 * DRAW-LOOP GL calls only -- the per-slice bindFramebuffer/uniform/draw and
	 * the moments binds. It deliberately does NOT count the per-layer program
	 * and uniform setup, so it is an undercount of the process's total and is
	 * not trying to be otherwise: what it measures is the thing that scales with
	 * the volume -- the one question the per-slice design had to answer before it
	 * was worth building, since it drives tens of thousands of GL calls through
	 * emscripten's shims: 39204 for MindGrab and 20388 for model16 at one voxel
	 * per fragment, and 29988 / 15780 in the shipping VOX=2 build. About three
	 * quarters, NOT half, because only the hidden convolutions are blocked --
	 * conv_first, the three moments passes, norm and classify are not. The
	 * savings are exactly 24 and 12 full-volume passes of 384 (9216 and 4608),
	 * which is a check on both figures rather than a coincidence. The counter
	 * below prints the real number rather than leaving this comment to rot. A
	 * local, not a file static -- nothing outside this function has ever read it.
	 */
	long gl_calls = 0;
	double t_start = emscripten_get_now();
	double t_conv = 0, t_moments = 0, t_norm = 0, t_classify = 0;
#if BC_WEBGL2_PROFILE
#define BC_STAGE_MARK(acc, fbo, type) do { gl_sync(fbo, type); \
		double now = emscripten_get_now(); (acc) += now - mark; mark = now; } while (0)
	double mark = t_start;
#else
#define BC_STAGE_MARK(acc, fbo, type) ((void)0)
#endif

	for (int layer = 0; layer < BC_MODEL_NHIDDEN; layer++) {
#if BC_STAGE_CONV
		GLuint conv;
		int vox = 1;
		if (layer == 0) {
			/* Layer 0 stays one voxel per fragment. It is one pass in 25 and
			   reads a single input channel, so it carries almost none of the
			   weight traffic VOX=2 exists to halve. */
			conv = p_conv_first;
			glUseProgram(conv);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, input_tex);
		} else {
			vox = use_vox2 ? 2 : 1;
			conv = use_vox2 ? p_conv_hidden2 : p_conv_hidden;
			glUseProgram(conv);
			for (int i = 0; i < BC_PLANES; i++) {
				glActiveTexture(GL_TEXTURE0 + i);
				glBindTexture(GL_TEXTURE_3D, act[P][i]);
			}
		}
		glUniform1i(glGetUniformLocation(conv, "uDil"), dilation[layer]);
		glUniform1i(glGetUniformLocation(conv, "uWQ0"),
			(GLint)(offsets[layer].kernel_off / 4));
		GLint u_z = glGetUniformLocation(conv, "uZ");
		glViewport(0, 0, BC_DIM, BC_DIM);
		for (int k = 0; k < BC_DIM / vox; k++) {
			glBindFramebuffer(GL_FRAMEBUFFER,
				vox == 2 ? vox2_fbo[k] : act_fbo[Q][k]);
			glUniform1i(u_z, k * vox);
			glDrawArrays(GL_TRIANGLES, 0, 3);
			gl_calls += 3;
		}
		BC_STAGE_MARK(t_conv, act_fbo[Q][0], GL_FLOAT);
#endif

#if BC_WEBGL2_PHASE1
		if (layer == 0) {
			/* Verify layer 0 before anything else touches it. Kept as the
			   pipeline grew: it is the one stage with an exact reference. */
			plane = (float *)malloc((size_t)BC_DIM * BC_DIM * 4 * sizeof(*plane));
			gpu_slice = (float *)malloc((size_t)BC_DIM * BC_DIM * BC_CS * sizeof(*gpu_slice));
			ref_slice = (float *)malloc((size_t)BC_DIM * BC_DIM * BC_CS * sizeof(*ref_slice));
			ref_kernel = (float *)malloc((size_t)27 * BC_CS * sizeof(*ref_kernel));
			if (!plane || !gpu_slice || !ref_slice || !ref_kernel) {
				printfx("webgl2: out of memory for the layer-0 comparison\n");
				goto cleanup;
			}
			for (size_t i = 0; i < (size_t)27 * BC_CS; i++)
#if BC_WGL_F32
				ref_kernel[i] = packed[offsets[0].kernel_off + i];
#else
				ref_kernel[i] = bc_f16_to_f32(packed[offsets[0].kernel_off + i]);
#endif
			/*
			 * z=0 exercises the low-z zero padding, DIM/2 is fully interior,
			 * and 5*DIM/8 is a second interior slice.
			 *
			 * More than one interior slice because z=0 can be VACUOUS and
			 * silently so: with model16's layer-0 dilation of 1 it samples only
			 * z in {0, 1}, which for a conformed brain is background, so the
			 * reference is identically zero and comparing against it proves
			 * nothing. That case is named rather than reported as agreement.
			 * Padding is covered regardless by
			 * test/webgl2_conv_first_check.c, which uses random input.
			 */
			int slices[3] = { 0, BC_DIM / 2, 5 * BC_DIM / 8 };
			int failed = 0, with_signal = 0;
			for (int s = 0; s < 3; s++) {
				int z = slices[s];
				glBindFramebuffer(GL_FRAMEBUFFER, act_fbo[Q][z]);
				for (int p = 0; p < BC_PLANES; p++) {
					glReadBuffer(g_draw_buffers[p]);
					glReadPixels(0, 0, BC_DIM, BC_DIM, GL_RGBA, GL_FLOAT, plane);
					if (gl_failed("layer-0 readPixels")) goto cleanup;
					for (int i = 0; i < BC_DIM * BC_DIM; i++)
						for (int k = 0; k < 4; k++)
							gpu_slice[(size_t)i * BC_CS + p * 4 + k] =
								plane[(size_t)i * 4 + k];
				}
				bc_webgl2_ref_conv_first_slice(ref_input, ref_kernel,
					dilation[0], z, BC_DIM, BC_CS, ref_slice);
				bc_slice_stats stats;
				compare_slice(gpu_slice, ref_slice, &stats);
				report_slice(z, &stats);
				if (stats.max_value > 0.0) with_signal++;
				if (stats.max_ulp > 4) {
					printfx("webgl2 layer-0: z=%d exceeds the 4 ulp gate\n", z);
					failed = 1;
				}
			}
			if (!with_signal) {
				printfx("webgl2 layer-0: every compared slice was identically "
					"zero; this input proves nothing about the convolution\n");
				failed = 1;
			}
			if (failed) goto cleanup;
			printfx("webgl2 layer-0: layer 0 agrees with the CPU reference\n");
		}
#endif

#if BC_STAGE_MOMENTS
		/* Pass A: one partial per (x, y) column, per channel group. Reads the
		   activation exactly once. The group's texture is bound to the same
		   sampler and the viewport moved to its rows, which avoids indexing an
		   array of samplers by a runtime value -- forbidden in GLSL ES 3.00. */
		glUseProgram(p_moments_a);
		glBindFramebuffer(GL_FRAMEBUFFER, part_fbo);
		GLint u_rowbase = glGetUniformLocation(p_moments_a, "uRowBase");
		for (int g = 0; g < BC_PLANES; g++) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, act[Q][g]);
			glViewport(0, g * BC_DIM, BC_DIM, BC_DIM);
			glUniform1i(u_rowbase, g * BC_DIM);
			glDrawArrays(GL_TRIANGLES, 0, 3);
			gl_calls += 5;
		}

		glActiveTexture(GL_TEXTURE0 + BC_UNIT_SUM);
		glBindTexture(GL_TEXTURE_2D, part_sum);
		glActiveTexture(GL_TEXTURE0 + BC_UNIT_SQ);
		glBindTexture(GL_TEXTURE_2D, part_sq);
		glUseProgram(p_moments_b);
		glBindFramebuffer(GL_FRAMEBUFFER, red_fbo);
		glViewport(0, 0, BC_DIM, BC_PLANES);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		glActiveTexture(GL_TEXTURE0 + BC_UNIT_SUM);
		glBindTexture(GL_TEXTURE_2D, red_sum);
		glActiveTexture(GL_TEXTURE0 + BC_UNIT_SQ);
		glBindTexture(GL_TEXTURE_2D, red_sq);
		glUseProgram(p_moments_finish);
		glBindFramebuffer(GL_FRAMEBUFFER, mom_fbo);
		glViewport(0, 0, 1, BC_PLANES);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		gl_calls += 12;
		BC_STAGE_MARK(t_moments, mom_fbo, GL_FLOAT);
#endif

#if BC_STAGE_NORM
		glUseProgram(p_norm);
		for (int i = 0; i < BC_PLANES; i++) {
			glActiveTexture(GL_TEXTURE0 + i);
			glBindTexture(GL_TEXTURE_3D, act[Q][i]);
		}
		glActiveTexture(GL_TEXTURE0 + BC_UNIT_SUM);
		glBindTexture(GL_TEXTURE_2D, mom_mean);
		glActiveTexture(GL_TEXTURE0 + BC_UNIT_SQ);
		glBindTexture(GL_TEXTURE_2D, mom_inv);
#if BC_MODEL_HAS_AFFINE
		glUniform1i(glGetUniformLocation(p_norm, "uAffQ"),
			(GLint)(offsets[layer].affine_off / 4));
		glUniform1i(glGetUniformLocation(p_norm, "uBiasQ"),
			(GLint)(offsets[layer].bias_off / 4));
#endif
		GLint u_nz = glGetUniformLocation(p_norm, "uZ");
		glViewport(0, 0, BC_DIM, BC_DIM);
		for (int z = 0; z < BC_DIM; z++) {
			glBindFramebuffer(GL_FRAMEBUFFER, act_fbo[P][z]);
			glUniform1i(u_nz, z);
			glDrawArrays(GL_TRIANGLES, 0, 3);
			gl_calls += 3;
		}
		BC_STAGE_MARK(t_norm, act_fbo[P][0], GL_FLOAT);
#endif
		if (gl_failed("layer draw")) goto cleanup;
	}

#if BC_HAS_PVE
    if (options && options->tissue_planes) {
        /* Detach all old targets before deleting the dead activation set. */
        for (int s=0;s<fbo_sets;s++) glDeleteFramebuffers(BC_DIM,act_fbo[s]);
        fbo_sets=0;
        if(vox2_fbos)glDeleteFramebuffers(vox2_fbos,vox2_fbo);
        vox2_fbos=0;
        for(int i=0;i<BC_PLANES;i++) {
            glDeleteTextures(1,&act[1-P][i]);act[1-P][i]=0;
        }
        if(bc_webgl2_prepare_priors(act[P],weight_tex,classifier_off,classifier_bias_off,
                options->tissue_planes,options->tissue_priors_prepared!=NULL))goto cleanup;
        if(options->tissue_priors_prepared)*options->tissue_priors_prepared=1;
        status=0;goto cleanup;
    }
#endif

#if BC_STAGE_CLASSIFY
	glUseProgram(p_classify);
	for (int i = 0; i < BC_PLANES; i++) {
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_3D, act[P][i]);
	}
	glUniform1i(glGetUniformLocation(p_classify, "uWCls"), (GLint)classifier_off);
	glUniform1i(glGetUniformLocation(p_classify, "uWBias"),
		(GLint)classifier_bias_off);
	glBindFramebuffer(GL_FRAMEBUFFER, label_fbo);
	glViewport(0, 0, BC_WGL_LAB_W, BC_WGL_LAB_W);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	gl_calls += 4;
	BC_STAGE_MARK(t_classify, label_fbo, GL_UNSIGNED_BYTE);
	if (gl_failed("classify")) goto cleanup;
#endif

	/*
	 * The readback, and the only place bytes leave the GPU.
	 *
	 * GL_RGBA/GL_UNSIGNED_BYTE from a normalized attachment is the one
	 * combination every implementation must support, which is why the labels
	 * are an RGBA8 texture rather than an integer one. Four consecutive x
	 * voxels per texel means the buffer's byte order IS the voxel order, so no
	 * de-interleaving is needed.
	 */
	glBindFramebuffer(GL_FRAMEBUFFER, label_fbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, BC_WGL_LAB_W, BC_WGL_LAB_W, GL_RGBA, GL_UNSIGNED_BYTE, labels);
	if (gl_failed("label readback")) goto cleanup;
	for (size_t i = 0; i < BC_NVOX; i++) result[i] = (float)labels[i];

	{
		double total = emscripten_get_now() - t_start;
		printfx("webgl2: %.2f s on the GPU, %ld draw-loop GL calls, %d voxel%s per "
			"fragment in the hidden convolution\n", total / 1000.0, gl_calls,
			use_vox2 ? 2 : 1, use_vox2 ? "s" : "");
#if BC_WEBGL2_PROFILE
		printfx("webgl2: conv %.3fs, moments %.3fs, norm %.3fs, classify %.3fs\n",
			t_conv / 1000.0, t_moments / 1000.0, t_norm / 1000.0,
			t_classify / 1000.0);
#else
		(void)t_conv; (void)t_moments; (void)t_norm; (void)t_classify;
#endif
	}
	status = 0;

cleanup:
	for (int s = 0; s < fbo_sets; s++) glDeleteFramebuffers(BC_DIM, act_fbo[s]);
	if (vox2_fbos) glDeleteFramebuffers(vox2_fbos, vox2_fbo);
	if (part_fbo) glDeleteFramebuffers(1, &part_fbo);
	if (red_fbo) glDeleteFramebuffers(1, &red_fbo);
	if (mom_fbo) glDeleteFramebuffers(1, &mom_fbo);
	if (label_fbo) glDeleteFramebuffers(1, &label_fbo);
	for (int s = 0; s < 2; s++)
		for (int i = 0; i < BC_PLANES; i++)
			if (act[s][i]) glDeleteTextures(1, &act[s][i]);
	if (part_sum) glDeleteTextures(1, &part_sum);
	if (part_sq) glDeleteTextures(1, &part_sq);
	if (red_sum) glDeleteTextures(1, &red_sum);
	if (red_sq) glDeleteTextures(1, &red_sq);
	if (mom_mean) glDeleteTextures(1, &mom_mean);
	if (mom_inv) glDeleteTextures(1, &mom_inv);
	if (label_tex) glDeleteTextures(1, &label_tex);
	if (weight_tex) glDeleteTextures(1, &weight_tex);
	if (input_tex) glDeleteTextures(1, &input_tex);
	if (vao) glDeleteVertexArrays(1, &vao);
	if (p_conv_first) glDeleteProgram(p_conv_first);
	if (p_conv_hidden) glDeleteProgram(p_conv_hidden);
	if (p_conv_hidden2) glDeleteProgram(p_conv_hidden2);
	if (p_moments_a) glDeleteProgram(p_moments_a);
	if (p_moments_b) glDeleteProgram(p_moments_b);
	if (p_moments_finish) glDeleteProgram(p_moments_finish);
	if (p_norm) glDeleteProgram(p_norm);
	if (p_classify) glDeleteProgram(p_classify);
	free(normalized);
	free(ref_input);
	free(input16);
	free(packed);
	free(labels);
	free(plane);
	free(gpu_slice);
	free(ref_slice);
	free(ref_kernel);
	return status;
}

const bc_backend bc_webgl2_backend = {
	/* No adapter hooks: WebGL2 gives no way to enumerate or choose a GPU --
	   the browser picks one when the context is created. */
	"webgl2", webgl2_available, webgl2_segment, NULL, NULL,
};
