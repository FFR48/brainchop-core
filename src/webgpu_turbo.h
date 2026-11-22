/*
 * WebGPU turbo: the 8-bit-activation MeshNet of metal_turbo.h, in WGSL.
 * Included only by backend_webgpu.c, after its helpers; the fp16 kernels in
 * webgpu_kernels.h stay behind --no-turbo and --pve.
 *
 * Same codes and statistics as Metal: u8 activations with square-law codes
 * (a = qs * ((q - 32) / 16)^2 signed, qs folded into the next layer's weights),
 * pre-norm conv output as u8 scaled per block, moments fused into the conv,
 * fp16 arithmetic (Metal's hidden convs accumulate in fp32 via matrices).
 * A u32 word holds 4 channels of one voxel. Differences:
 * the block is a 64-lane workgroup (128 voxels) reduced through workgroup
 * memory, because subgroups are not available everywhere; weights are the
 * f16 copies the fp16 path uploads. Rows follow the turbo_cone.h schedule.
 */
#ifndef BRAINCHOPC_WEBGPU_TURBO_H
#define BRAINCHOPC_WEBGPU_TURBO_H

/* Accumulators a<voxel><quad>: named, not an array (see webgpu_kernels.h). */
#define BC_TB_DECL(n, b) "  var a0" n " = vec4<f16>(0.0h); var a1" n " = vec4<f16>(0.0h);\n"
#define BC_TB_FMA(n, b, c) \
	"      { let w = weff[wq + " c "u * CS4 + " n "u];\n" \
	"        a0" n " = fma(w, vec4<f16>(s0[" c "]), a0" n "); a1" n " = fma(w, vec4<f16>(s1[" c "]), a1" n "); }\n"
#define BC_TB_FMA0(n, b) \
	"      { let w = wts4[(tp.w_off / 4u) + tap * CS4 + " n "u];\n" \
	"        a0" n " = fma(w, vec4<f16>(s0), a0" n "); a1" n " = fma(w, vec4<f16>(s1), a1" n "); }\n"
#define BC_TB_FMAJ(n, b) BC_TB_FMA(n, b, "0") BC_TB_FMA(n, b, "1") BC_TB_FMA(n, b, "2") BC_TB_FMA(n, b, "3")
/* Block reduction of quad n, then its u8 codes and (lane 0) the moments. */
#define BC_TB_EPI(n, b) \
	"  { let x0 = vec4<f32>(a0" n "); let x1 = vec4<f32>(a1" n ");\n" \
	"    rs[lid] = x0 + x1; rq[lid] = x0 * x0 + x1 * x1; rmn[lid] = min(x0, x1); rmx[lid] = max(x0, x1);\n" \
	"    workgroupBarrier();\n" \
	"    for (var st = 32u; st > 0u; st = st >> 1u) {\n" \
	"      if (lid < st) { rs[lid] += rs[lid + st]; rq[lid] += rq[lid + st];\n" \
	"        rmn[lid] = min(rmn[lid], rmn[lid + st]); rmx[lid] = max(rmx[lid], rmx[lid + st]); }\n" \
	"      workgroupBarrier();\n" \
	"    }\n" \
	"    let mn = rmn[0]; let sp = rmx[0] - mn; let inv = select(vec4<f32>(0.0), 1.0 / sp, sp > vec4<f32>(1e-30));\n" \
	"    dst[v0 * CS4 + " n "u] = pack4x8unorm((x0 - mn) * inv);\n" \
	"    dst[(v0 + 1u) * CS4 + " n "u] = pack4x8unorm((x1 - mn) * inv);\n" \
	"    if (lid == 0u) { for (var j = 0u; j < 4u; j++) {\n" \
	"      mom[(" n "u * 4u + j) * NBLK + blk] = vec4<f32>(rs[0][j], rq[0][j], mn[j], sp[j]); } }\n" \
	"    workgroupBarrier(); }\n"

#define BC_TB_CONV_HEAD(NAME) \
	"@compute @workgroup_size(64, 1, 1)\n" \
	"fn " NAME "(@builtin(local_invocation_index) lid : u32, @builtin(workgroup_id) wid : vec3<u32>) {\n" \
	"  let ri = wid.z * 256u + wid.y;\n" \
	"  if (ri >= tp.count) { return; }\n" \
	"  let row = list[tp.list_off + ri];\n" \
	"  let y = i32(row % 256u); let z = i32(row / 256u); let x0 = i32(wid.x * 128u + lid * 2u); let d = tp.dil;\n" \
	"  let v0 = row * 256u + u32(x0); let blk = row * 2u + wid.x;\n" \
	BC_WG_ACC_N(BC_TB_DECL) \
	"  var tap = 0u;\n" \
	"  for (var dz = -1; dz <= 1; dz++) { for (var dy = -1; dy <= 1; dy++) {\n" \
	"    let sz = z + dz * d; let sy = y + dy * d;\n" \
	"    let ok = sz >= 0 && sz < 256 && sy >= 0 && sy < 256;\n" \
	"    let plane = (u32(clamp(sz, 0, 255)) * 256u + u32(clamp(sy, 0, 255))) * 256u;\n" \
	"    for (var dx = -1; dx <= 1; dx++) {\n" \
	"     if (ok) {\n" \
	"      let sx0 = x0 + dx * d; let sx1 = sx0 + 1;\n" \
	"      let p0 = plane + u32(clamp(sx0, 0, 255)); let p1 = plane + u32(clamp(sx1, 0, 255));\n" \
	"      let m0 = select(0.0, 1.0, u32(sx0) < 256u); let m1 = select(0.0, 1.0, u32(sx1) < 256u);\n"
#define BC_TB_CONV_TAIL \
	"     }\n" \
	"     tap++;\n" \
	"  }}}\n" \
	BC_WG_ACC_N(BC_TB_EPI) \
	"}\n"

static const char bc_wgsl_turbo_source[] = BC_WG_ENABLE_F16
"const CS : u32 = " BC_WG_STR(BC_WG_CS) "u;\n"
"const CS4 : u32 = CS / 4u;\n"
"const CHAN : u32 = " BC_WG_STR(BC_MODEL_CHAN) "u;\n"
"const NCLASS : u32 = " BC_WG_STR(BC_MODEL_NCLASS) "u;\n"
"const AFFINE : bool = " BC_WG_STR(BC_MODEL_HAS_AFFINE) " != 0;\n"
"const CBIAS : bool = " BC_WG_STR(BC_MODEL_CLASSIFIER_BIAS) " != 0;\n"
"const NV : u32 = 16777216u;\n"
"const BLK : u32 = 128u;\n"
"const NBLK : u32 = NV / BLK;\n"
"const ZP : f32 = 32.0;\n"
"const EPS : f32 = 1e-5;\n"
BC_WGSL_GELU
/* Rows come from the turbo_cone.h schedule (see metal_turbo.h's Layer). */
"struct TP { dil : i32, w_off : u32, aff_off : u32, bias_off : u32,\n"
"  next_w_off : u32, cls_off : u32, cls_bias_off : u32, list_off : u32,\n"
"  count : u32, pure_off : u32, npure : u32, rep : u32, rows_off : u32, nrows : u32 };\n"
"const ROW : u32 = 256u * CS4;\n"
"@group(0) @binding(0) var<storage, read> src : array<u32>;\n"
"@group(0) @binding(1) var<storage, read_write> dst : array<u32>;\n"
"@group(0) @binding(2) var<storage, read_write> mom : array<vec4<f32>>;\n"
"@group(0) @binding(3) var<storage, read> weff : array<vec4<f16>>;\n"
"@group(0) @binding(4) var<uniform> tp : TP;\n"
"@group(0) @binding(5) var<storage, read> wts : array<f16>;\n"
"@group(0) @binding(6) var<storage, read_write> stats : array<vec4<f32>>;\n"
"@group(0) @binding(7) var<storage, read_write> weffw : array<f16>;\n"
"@group(0) @binding(8) var<storage, read> inp : array<f16>;\n"
"@group(0) @binding(9) var<storage, read_write> labels : array<u32>;\n"
"@group(0) @binding(10) var<storage, read> wts4 : array<vec4<f16>>;\n"
"@group(0) @binding(11) var<storage, read> list : array<u32>;\n"
"@group(0) @binding(12) var<storage, read> rw : array<f32>;\n"
"var<workgroup> rs : array<vec4<f32>, 64>;\n"
"var<workgroup> rq : array<vec4<f32>, 64>;\n"
"var<workgroup> rmn : array<vec4<f32>, 64>;\n"
"var<workgroup> rmx : array<vec4<f32>, 64>;\n"
"var<workgroup> fr : array<vec4<f32>, 256>;\n"
BC_TB_CONV_HEAD("conv0")
"      let s0 = f16(select(0.0, f32(inp[p0]), m0 > 0.0)); let s1 = f16(select(0.0, f32(inp[p1]), m1 > 0.0));\n"
BC_WG_ACC_N(BC_TB_FMA0)
BC_TB_CONV_TAIL
BC_TB_CONV_HEAD("conv")
"      for (var i4 = 0u; i4 < CS4; i4++) {\n"
/* code q decodes to (q/16 - 2)^2 signed; unpack gives q/255. */
"        let u0 = vec4<f16>((unpack4x8unorm(src[p0 * CS4 + i4]) * 15.9375 - 2.0) * m0);\n"
"        let u1 = vec4<f16>((unpack4x8unorm(src[p1 * CS4 + i4]) * 15.9375 - 2.0) * m1);\n"
"        let s0 = u0 * abs(u0); let s1 = u1 * abs(u1);\n"
"        let wq = (tap * CS + i4 * 4u) * CS4;\n"
BC_WG_ACC_N(BC_TB_FMAJ)
"      }\n"
BC_TB_CONV_TAIL
/* One workgroup per channel: fold block moments; stats[c] = (scale, shift, 1/qs, qs). */
"@compute @workgroup_size(256, 1, 1)\n"
"fn finish(@builtin(local_invocation_index) lid : u32, @builtin(workgroup_id) wid : vec3<u32>) {\n"
"  let c = wid.x;\n"
"  var r = vec4<f32>(0.0, 0.0, 3.0e38, -3.0e38);\n"
"  for (var i = lid; i < tp.count; i += 256u) { let w = rw[tp.list_off + i];\n"
"    for (var b = 0u; b < 2u; b++) { let m = mom[c * NBLK + list[tp.list_off + i] * 2u + b];\n"
"      r = vec4<f32>(r.x + w * m.x, r.y + w * m.y, min(r.z, m.z), max(r.w, m.z + m.w)); } }\n"
"  fr[lid] = r;\n"
"  workgroupBarrier();\n"
"  for (var st = 128u; st > 0u; st = st >> 1u) {\n"
"    if (lid < st) { let o = fr[lid + st]; let t = fr[lid];\n"
"      fr[lid] = vec4<f32>(t.x + o.x, t.y + o.y, min(t.z, o.z), max(t.w, o.w)); }\n"
"    workgroupBarrier();\n"
"  }\n"
"  if (lid != 0u) { return; }\n"
"  let t = fr[0];\n"
"  let mean = t.x / f32(NV); let vr = max(t.y / f32(NV) - mean * mean, 0.0);\n"
"  let inv = inverseSqrt(vr + EPS);\n"
"  var gamma = 1.0; var beta = 0.0;\n"
"  if (AFFINE) { gamma = f32(wts[tp.aff_off + c]); beta = f32(wts[tp.bias_off + c]); }\n"
"  let scale = inv * gamma; let shift = beta - mean * scale;\n"
"  let vmax = max(0.25, max(gelu(t.z * scale + shift), gelu(t.w * scale + shift)));\n"
"  var qs = 1.0;\n"
"  if (c < CHAN) { qs = max(vmax / (223.0 * 223.0), 0.1705 / (ZP * ZP)); }\n"
"  stats[c] = vec4<f32>(scale, shift, 1.0 / qs, qs);\n"
"}\n"
/* Next layer's weights times its input channel's qs (x256 for the /16 decode). */
"@compute @workgroup_size(64, 1, 1)\n"
"fn prep(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
"  let i = gid.x;\n"
"  if (i < 27u * CS * CS) { weffw[i] = f16(f32(wts[tp.next_w_off + i]) * stats[(i / CS) % CS].w * 256.0); }\n"
"}\n"
"@compute @workgroup_size(64, 1, 1)\n"
"fn norm(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
"  let ri = gid.z * 256u + gid.y;\n"
"  if (ri >= tp.count) { return; }\n"
"  let i = list[tp.list_off + ri] * ROW + gid.x;\n"
"  let v = i / CS4; let blk = v / BLK; let c0 = (i % CS4) * 4u;\n"
"  let q = unpack4x8unorm(dst[i]);\n"
"  var r = vec4<f32>(0.0);\n"
"  for (var j = 0u; j < 4u; j++) {\n"
"    let m = mom[(c0 + j) * NBLK + blk]; let st = stats[c0 + j];\n"
"    let a = gelu((q[j] * m.w + m.z) * st.x + st.y) * st.z;\n"
"    r[j] = clamp(round(sign(a) * sqrt(abs(a)) + ZP), 0.0, 255.0);\n"
"  }\n"
"  dst[i] = pack4x8unorm(r * (1.0 / 255.0));\n"
"}\n"
"@compute @workgroup_size(64, 1, 1)\n"
"fn fill(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
"  let ri = gid.z * 256u + gid.y;\n"
"  if (ri < tp.npure) { dst[list[tp.pure_off + ri] * ROW + gid.x] = dst[tp.rep * ROW + gid.x]; }\n"
"}\n"
"@compute @workgroup_size(64, 1, 1)\n"
"fn fill_labels(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
"  let ri = gid.z * 256u + gid.y;\n"
"  if (ri < tp.npure) { labels[list[tp.pure_off + ri] * 256u + gid.x] = labels[tp.rep * 256u + gid.x]; }\n"
"}\n"
/* Last GroupNorm + GELU, unquantized, then 1x1 conv and argmax (first max). */
"@compute @workgroup_size(64, 1, 1)\n"
"fn classify(@builtin(global_invocation_id) gid : vec3<u32>) {\n"
"  let ri = gid.z * 256u + gid.y;\n"
"  if (ri >= tp.nrows) { return; }\n"
"  let v = list[tp.rows_off + ri] * 256u + gid.x; let blk = v / BLK;\n"
"  var a : array<f32, CS>;\n"
"  for (var o = 0u; o < CS4; o++) { let q = unpack4x8unorm(src[v * CS4 + o]);\n"
"    for (var j = 0u; j < 4u; j++) { let c = o * 4u + j; let m = mom[c * NBLK + blk]; let st = stats[c];\n"
"      a[c] = gelu((q[j] * m.w + m.z) * st.x + st.y); } }\n"
"  var best = 0u; var bv = -3.0e38;\n"
"  for (var k = 0u; k < NCLASS; k++) {\n"
"    var acc = 0.0;\n"
"    if (CBIAS) { acc = f32(wts[tp.cls_bias_off + k]); }\n"
"    for (var c = 0u; c < CHAN; c++) { acc += a[c] * f32(wts[tp.cls_off + c * NCLASS + k]); }\n"
"    if (acc > bv) { bv = acc; best = k; }\n"
"  }\n"
"  labels[v] = best;\n"
"}\n";

typedef struct {
	int32_t dil;
	uint32_t w_off, aff_off, bias_off, next_w_off, cls_off, cls_bias_off, list_off,
		count, pure_off, npure, rep, rows_off, nrows;
} bc_turbo_params;

#include "turbo_cone.h"

/* A bind group for one auto-layout pipeline: `n` (binding, buffer) pairs. */
static WGPUBindGroup turbo_group(WGPUDevice device, WGPUComputePipeline pipe,
		int n, const uint32_t *binding, const WGPUBuffer *buffer) {
	WGPUBindGroupEntry entries[8] = {0};
	for (int i = 0; i < n; i++) {
		entries[i].binding = binding[i];
		entries[i].buffer = buffer[i];
		entries[i].size = wgpuBufferGetSize(buffer[i]);
	}
	WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipe, 0);
	WGPUBindGroupDescriptor descriptor = {0};
	descriptor.layout = layout;
	descriptor.entryCount = (size_t)n;
	descriptor.entries = entries;
	WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &descriptor);
	wgpuBindGroupLayoutRelease(layout);
	return group;
}

/* Returns 0 on success. */
static int webgpu_turbo(WGPUDevice device, WGPUQueue queue, const unsigned char *conformed,
		float *result, const float *weights) {
	enum { P_CONV0, P_CONV, P_FINISH, P_PREP, P_NORM, P_CLASSIFY, P_FILL, P_FILL_LABELS, P_N };
	enum { G_CONV0, G_CONV_AB, G_CONV_BA, G_FINISH, G_PREP, G_NORM_A, G_NORM_B, G_CLS_A, G_CLS_B,
		G_FILL_A, G_FILL_B, G_FILL_LABELS, G_N };
	static const char *const entry[P_N] = { "conv0", "conv", "finish", "prep", "norm", "classify",
		"fill", "fill_labels" };
	const uint64_t act_bytes = (uint64_t)BC_NVOX * BC_CS, mom_bytes = (uint64_t)BC_NVOX / 128 * BC_CS * 16;
	int status = 1;
	WGPUShaderModule module = NULL;
	WGPUComputePipeline pipe[P_N] = {0};
	WGPUBindGroup group[G_N] = {0};
	WGPUBuffer input_buf = NULL, act_a = NULL, act_b = NULL, weight_buf = NULL, param_buf = NULL,
		mom_buf = NULL, stats_buf = NULL, weff_buf = NULL, label_buf = NULL, readback = NULL,
		list_buf = NULL, rw_buf = NULL;
	bc_tc_plan plan = {0};
	bc_store *packed = NULL, *input16 = NULL;
	float *normalized = NULL;
	double t_start = bc_now(), t_ready = 0;

	wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
	wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
	int scope_pushed = 2;

	WGPUShaderSourceWGSL wgsl = {0};
	wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
	wgsl.code = view(bc_wgsl_turbo_source);
	WGPUShaderModuleDescriptor module_descriptor = {0};
	module_descriptor.nextInChain = &wgsl.chain;
	module = wgpuDeviceCreateShaderModule(device, &module_descriptor);
	if (!module) { printfx("webgpu: turbo shader module failed\n"); goto cleanup; }
	for (int k = 0; k < P_N; k++) {
		pipe[k] = make_pipeline(device, NULL, module, entry[k]);
		if (!pipe[k]) { printfx("webgpu: turbo pipeline %s failed\n", entry[k]); goto cleanup; }
	}

	bc_layer_offsets offsets[BC_MODEL_NHIDDEN];
	uint32_t cls_off = 0, cls_bias_off = 0;
	size_t weight_count = 0;
	packed = bc_pack_weights(weights, offsets, &cls_off, &cls_bias_off, &weight_count, 4);
	if (bc_tc_make(conformed, brainchop_model_dilation, &plan)) goto cleanup;
	normalized = bc_accel_normalize_input(conformed);
	input16 = (bc_store *)malloc(BC_NVOX * sizeof(*input16));
	if (!packed || !normalized || !input16) goto cleanup;
	for (size_t i = 0; i < BC_NVOX; i++) input16[i] = BC_STORE(normalized[i]);
	free(normalized);
	normalized = NULL;

	const WGPUBufferUsage st = WGPUBufferUsage_Storage;
	input_buf = make_buffer(device, BC_NVOX * sizeof(bc_store), st | WGPUBufferUsage_CopyDst);
	act_a = make_buffer(device, act_bytes, st);
	act_b = make_buffer(device, act_bytes, st);
	/* Whole vec4s (pad_to 4 above): wts is also bound as vec4<f16>. */
	const size_t weight_bytes = ((weight_count + 3) & ~(size_t)3) * sizeof(bc_store);
	weight_buf = make_buffer(device, weight_bytes, st | WGPUBufferUsage_CopyDst);
	param_buf = make_buffer(device, sizeof(bc_turbo_params), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
	mom_buf = make_buffer(device, mom_bytes, st);
	stats_buf = make_buffer(device, BC_CS * 16, st);
	weff_buf = make_buffer(device, 27 * BC_CS * BC_CS * sizeof(uint16_t), st);
	label_buf = make_buffer(device, BC_NVOX * sizeof(uint32_t), st | WGPUBufferUsage_CopySrc);
	readback = make_buffer(device, BC_NVOX * sizeof(uint32_t), WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
	list_buf = make_buffer(device, plan.n * sizeof(uint32_t), st | WGPUBufferUsage_CopyDst);
	rw_buf = make_buffer(device, plan.n * sizeof(float), st | WGPUBufferUsage_CopyDst);
	if (!input_buf || !act_a || !act_b || !weight_buf || !param_buf || !mom_buf || !stats_buf ||
			!weff_buf || !label_buf || !readback || !list_buf || !rw_buf) {
		printfx("webgpu: turbo buffer allocation failed (needs about %llu MiB)\n",
			(unsigned long long)((2 * act_bytes + mom_bytes) >> 20));
		goto cleanup;
	}
	wgpuQueueWriteBuffer(queue, weight_buf, 0, packed, weight_bytes);
	wgpuQueueWriteBuffer(queue, input_buf, 0, input16, BC_NVOX * sizeof(bc_store));
	wgpuQueueWriteBuffer(queue, list_buf, 0, plan.list, plan.n * sizeof(uint32_t));
	wgpuQueueWriteBuffer(queue, rw_buf, 0, plan.weight, plan.n * sizeof(float));

	{
		static const uint32_t b_conv0[] = { 1, 2, 4, 8, 10, 11 }, b_conv[] = { 0, 1, 2, 3, 4, 11 },
			b_finish[] = { 2, 4, 5, 6, 11, 12 }, b_prep[] = { 4, 5, 6, 7 }, b_norm[] = { 1, 2, 4, 6, 11 },
			b_cls[] = { 0, 2, 4, 5, 6, 9, 11 }, b_fill[] = { 1, 4, 11 }, b_fill_labels[] = { 4, 9, 11 };
		WGPUBuffer conv0[] = { act_a, mom_buf, param_buf, input_buf, weight_buf, list_buf };
		WGPUBuffer ab[] = { act_a, act_b, mom_buf, weff_buf, param_buf, list_buf };
		WGPUBuffer ba[] = { act_b, act_a, mom_buf, weff_buf, param_buf, list_buf };
		WGPUBuffer fin[] = { mom_buf, param_buf, weight_buf, stats_buf, list_buf, rw_buf };
		WGPUBuffer prep[] = { param_buf, weight_buf, stats_buf, weff_buf };
		WGPUBuffer na[] = { act_a, mom_buf, param_buf, stats_buf, list_buf };
		WGPUBuffer nb[] = { act_b, mom_buf, param_buf, stats_buf, list_buf };
		WGPUBuffer ca[] = { act_a, mom_buf, param_buf, weight_buf, stats_buf, label_buf, list_buf };
		WGPUBuffer cb[] = { act_b, mom_buf, param_buf, weight_buf, stats_buf, label_buf, list_buf };
		WGPUBuffer fa[] = { act_a, param_buf, list_buf }, fb[] = { act_b, param_buf, list_buf };
		WGPUBuffer fl[] = { param_buf, label_buf, list_buf };
		group[G_CONV0] = turbo_group(device, pipe[P_CONV0], 6, b_conv0, conv0);
		group[G_CONV_AB] = turbo_group(device, pipe[P_CONV], 6, b_conv, ab);
		group[G_CONV_BA] = turbo_group(device, pipe[P_CONV], 6, b_conv, ba);
		group[G_FINISH] = turbo_group(device, pipe[P_FINISH], 6, b_finish, fin);
		group[G_PREP] = turbo_group(device, pipe[P_PREP], 4, b_prep, prep);
		group[G_NORM_A] = turbo_group(device, pipe[P_NORM], 5, b_norm, na);
		group[G_NORM_B] = turbo_group(device, pipe[P_NORM], 5, b_norm, nb);
		group[G_CLS_A] = turbo_group(device, pipe[P_CLASSIFY], 7, b_cls, ca);
		group[G_CLS_B] = turbo_group(device, pipe[P_CLASSIFY], 7, b_cls, cb);
		group[G_FILL_A] = turbo_group(device, pipe[P_FILL], 3, b_fill, fa);
		group[G_FILL_B] = turbo_group(device, pipe[P_FILL], 3, b_fill, fb);
		group[G_FILL_LABELS] = turbo_group(device, pipe[P_FILL_LABELS], 3, b_fill_labels, fl);
		for (int g = 0; g < G_N; g++)
			if (!group[g]) { printfx("webgpu: turbo bind group %d failed\n", g); goto cleanup; }
	}
	t_ready = bc_now();

	/* Layer l writes act_a when l is even. Params change per layer, so each
	   layer is its own submission (the queue write orders against it). */
	for (int l = 0; l < BC_MODEL_NHIDDEN; l++) {
		const int last = l + 1 == BC_MODEL_NHIDDEN, even = !(l & 1);
		const bc_tc_layer *t = &plan.layer[l];
		bc_turbo_params p = { brainchop_model_dilation[l], offsets[l].kernel_off,
			offsets[l].affine_off, offsets[l].bias_off, last ? 0 : offsets[l + 1].kernel_off,
			cls_off, cls_bias_off, t->list_off, t->count, t->pure_off, t->npure, t->rep,
			plan.cls_off, plan.ncls };
		/* Row lists span workgroup y (256) and z. */
#define BC_TB_ZN(n) (((n) + 255) / 256)
		wgpuQueueWriteBuffer(queue, param_buf, 0, &p, sizeof p);
		WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
		WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
#define BC_TB_RUN(P, G, X, Y, Z) do { wgpuComputePassEncoderSetPipeline(pass, pipe[P]); \
		wgpuComputePassEncoderSetBindGroup(pass, 0, group[G], 0, NULL); \
		wgpuComputePassEncoderDispatchWorkgroups(pass, X, Y, Z); } while (0)
		BC_TB_RUN(l ? P_CONV : P_CONV0, l ? (even ? G_CONV_BA : G_CONV_AB) : G_CONV0, 2, 256,
			BC_TB_ZN(t->count));
		BC_TB_RUN(P_FINISH, G_FINISH, BC_CS, 1, 1);
		if (!last) {
			BC_TB_RUN(P_PREP, G_PREP, (27 * BC_CS * BC_CS + 63) / 64, 1, 1);
			BC_TB_RUN(P_NORM, even ? G_NORM_A : G_NORM_B, BC_CS, 256, BC_TB_ZN(t->count));
			if (t->npure)
				BC_TB_RUN(P_FILL, even ? G_FILL_A : G_FILL_B, BC_CS, 256, BC_TB_ZN(t->npure));
		} else {
			BC_TB_RUN(P_CLASSIFY, even ? G_CLS_A : G_CLS_B, 4, 256, BC_TB_ZN(plan.ncls));
			if (t->npure)
				BC_TB_RUN(P_FILL_LABELS, G_FILL_LABELS, 4, 256, BC_TB_ZN(t->npure));
		}
#undef BC_TB_RUN
		wgpuComputePassEncoderEnd(pass);
		if (last) wgpuCommandEncoderCopyBufferToBuffer(encoder, label_buf, 0, readback, 0,
			BC_NVOX * sizeof(uint32_t));
		WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
		wgpuQueueSubmit(queue, 1, &commands);
		wgpuCommandBufferRelease(commands);
		wgpuComputePassEncoderRelease(pass);
		wgpuCommandEncoderRelease(encoder);
	}

	scope_pushed = 0;
	if (bc_wg_pop_and_map(device, readback, BC_NVOX * sizeof(uint32_t))) goto cleanup;
	const uint32_t *labels = (const uint32_t *)wgpuBufferGetConstMappedRange(readback, 0,
		BC_NVOX * sizeof(uint32_t));
	if (!labels) { printfx("webgpu: could not map the readback buffer\n"); goto cleanup; }
	for (size_t i = 0; i < BC_NVOX; i++) result[i] = (float)labels[i];
	wgpuBufferUnmap(readback);
	if (bc_profiling())
		printfx("webgpu: turbo setup %.3fs, gpu+readback %.3fs\n",
			t_ready - t_start, bc_now() - t_ready);
	status = 0;

cleanup:
	bc_wg_drop_scopes(device, scope_pushed);
	free(packed);
	free(normalized);
	free(input16);
	bc_tc_free(&plan);
	for (int g = 0; g < G_N; g++) wgpuBindGroupRelease(group[g]);
	WGPUBuffer buffers[] = { input_buf, act_a, act_b, weight_buf, param_buf, mom_buf,
		stats_buf, weff_buf, label_buf, readback, list_buf, rw_buf };
	for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++) {
		if (buffers[i]) wgpuBufferDestroy(buffers[i]);
		wgpuBufferRelease(buffers[i]);
	}
	for (int k = 0; k < P_N; k++) wgpuComputePipelineRelease(pipe[k]);
	wgpuShaderModuleRelease(module);
	return status;
}

#endif
