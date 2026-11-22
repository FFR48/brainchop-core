#ifndef BRAINCHOP_CUDA_VARIANTS_H
#define BRAINCHOP_CUDA_VARIANTS_H

/*
 * Authoritative catalog of CUDA convolution kernel variants.
 *
 * Keep this data-only: both the in-tree CUDA runtime backend and the
 * checked-in Driver API module generator expand these lists. Sharing the
 * catalog makes it impossible for tuning to select a variant that was not
 * shipped in the native module.
 */
#define BC_CUDA_SCALAR_VARIANTS(X) \
	X(1, 128) X(1, 256) X(1, 512) X(2, 128) X(2, 256) X(2, 512) \
	X(3, 128) X(3, 256) X(3, 512) X(4, 128) X(4, 256) X(4, 512) \
	X(6, 128) X(6, 256) X(6, 512) X(8, 128) X(8, 256) X(8, 512)

#define BC_CUDA_WMMA_VARIANTS(X) \
	X(2) X(4) X(8) X(16) X(24) X(32)

#endif
