#ifndef BRAINCHOPC_BWLABEL_H
#define BRAINCHOPC_BWLABEL_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Retain the largest nonzero component. `connectivity` is 6 (face neighbours)
   or 26 (face, edge and corner); any other value is rejected. If binary_output
   is nonzero, retained values become 1; otherwise their original labels remain.
   The two differ in practice: 26 bridges structures that touch only at a
   corner, which is why CAT-lite's support cleanup asks for 6. */
int bwlabel_largest(float *image, const size_t dimensions[3], int binary_output,
		int connectivity);
/* 18-class browser cleanup: 6-connected brain mask, then largest component
   of classes 1, 2, 5, 6, 13. Suppressed components inherit the most common
   surviving class on their face boundary (ties: lowest class). Background
   and other suppressed components never vote. All other classes are kept.
   Input must contain integer labels 0..17. Returns nonzero on error. */
int bwlabel_cleanup18(float *image, const size_t dimensions[3]);
/* 104-class browser cleanup (Desikan-Killiany): 6-connected brain mask to drop
   disconnected exterior and resection-cavity islands, then largest 6-connected
   component per class (1..103; ties keep the later component). Suppressed
   components inherit the most common surviving face-neighbor class, restricted
   first to the same cortical hemisphere family (1..34 LH, 35..68 RH) with
   unrestricted fallback when no same-family neighbor exists (ties: lowest class).
   Input must contain integer labels 0..103. Returns nonzero on error. */
int bwlabel_cleanup104(float *image, const size_t dimensions[3]);
/* Validate raw categorical conformed output before cleanup can hide a failed
   inference. Does not modify image. Reports a diagnostic and returns nonzero
   for invalid labels, empty/near-filled foreground or excessive fragmentation. */
int bwlabel_validate(const float *image, const size_t dimensions[3], unsigned classes);
#ifdef __cplusplus
}
#endif
#endif
