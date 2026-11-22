/* Linux/glibc test-only symbol auditing. Fail hipMalloc in the child process
 * while leaving real adapter enumeration and hipMemGetInfo available. No GPU
 * memory pressure or changes to the driver/model server are required. */
#define _GNU_SOURCE
#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
unsigned int la_version(unsigned int version) { (void)version; return LAV_CURRENT; }
unsigned int la_objopen(struct link_map *map, Lmid_t namespace_id, uintptr_t *cookie) {
    (void)map; (void)namespace_id; (void)cookie;
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}
static int fail_allocation(void **pointer, size_t bytes) {
    (void)bytes; *pointer = NULL; return 2; /* hipErrorOutOfMemory */
}
uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index,
        uintptr_t *reference, uintptr_t *definition, unsigned int *flags, const char *name) {
    (void)index; (void)reference; (void)definition; (void)flags;
    if (!strcmp(name, "hipMalloc")) return (uintptr_t)fail_allocation;
    return symbol->st_value;
}
