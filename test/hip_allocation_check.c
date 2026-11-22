/* Compile with section GC: exercise the C-only loader's actual allocation
 * helper without a HIP driver, GPU, or large allocation. */
#include "../src/backend_hip_driver.c"
#include <assert.h>

static int calls, fail_at, query_mode, queries;
static hipError_t allocate_stub(HIPdeviceptr *ptr, size_t bytes) {
    assert(bytes); calls++;
    if (calls == fail_at) return 2;
    *ptr = (void *)(uintptr_t)calls;
    return 0;
}
static hipError_t memory_stub(size_t *available, size_t *total) {
    queries++;
    *available = (size_t)1 << 30;
    *total = (size_t)8 << 30;
    return query_mode == 2 ? 1 : 0;
}
static const char *error_stub(hipError_t status) { (void)status; return "hipErrorOutOfMemory"; }
int main(int argc, char **argv) {
    assert(argc == 4);
    fail_at = atoi(argv[1]); query_mode = atoi(argv[2]);
    int pve = atoi(argv[3]);
    const bc_hip_profile *p = &bc_hip_artifact_profile;
    uint64_t voxels = (uint64_t)p->dimension * p->dimension * p->dimension;
    uint64_t sizes[3] = {p->arena_bytes, p->input_bytes, pve ? p->pve_output_bytes : voxels};
    assert(!pve || p->supports_pve);
    HIPdeviceptr buffers[3] = {NULL, NULL, NULL};
    d.malloc_fn = allocate_stub;
    d.mem_get_info = query_mode ? memory_stub : NULL;
    d.error_name = d.error_string = error_stub;
    assert(allocate_buffers(buffers, sizes) == !!fail_at);
    assert(calls == (fail_at ? fail_at : 3));
    assert(queries == (fail_at && query_mode ? 1 : 0));
    for (int i = 0; i < 3; i++) assert(!!buffers[i] == (!fail_at || i < fail_at - 1));
    puts("PASS: actual C loader allocation boundary");
    return 0;
}
