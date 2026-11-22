#ifndef BRAINCHOPC_PAYLOAD_H
#define BRAINCHOPC_PAYLOAD_H

#include "model_specs.h"
#include <stddef.h>

/*
 * This executable's weights, embedded by payload.S: model.bin verbatim, in
 * read-only data. There is no models/ directory to find, no model.json to parse
 * and no file to open at run time, which is what makes the executable and its
 * model inseparable.
 *
 * Declared as arrays rather than pointers: the linker supplies the addresses of
 * these symbols, and an `extern const unsigned char *` would instead declare a
 * pointer variable stored at that address.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char bc_model_payload_begin[];
extern const unsigned char bc_model_payload_end[];

#ifdef __cplusplus
}
#endif

#endif
