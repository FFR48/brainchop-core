#ifndef BRAINCHOP_CUDA_GEOMETRY_H
#define BRAINCHOP_CUDA_GEOMETRY_H

/* Data-only offline launch/layout contract. The generator records these
 * literals in the manifest/catalog; cuda_module.cu checks the actual compiled
 * macros against them. The driver consumes the generated catalog, not these
 * source defaults, so its geometry always travels with the shipped module. */
#if (defined(BC_MODEL_IS_MINDMAP) && BC_MODEL_IS_MINDMAP) || \
    (defined(BC_MODEL_IS_MINDSNAP) && BC_MODEL_IS_MINDSNAP)
#include "cuda_geometry_mindmap.h"
#else
#define BC_CUDA_GEOMETRY_DIM 256
#define BC_CUDA_GEOMETRY_MAX_CHANNELS 16
#define BC_CUDA_GEOMETRY_CS 16
#define BC_CUDA_GEOMETRY_THREADS 128
#define BC_CUDA_GEOMETRY_REDUCE_THREADS 256
#define BC_CUDA_GEOMETRY_REDUCE_BLOCKS 256
#define BC_CUDA_GEOMETRY_CONV_VPT 3
#define BC_CUDA_GEOMETRY_CONV_THREADS 512
#define BC_CUDA_GEOMETRY_WMMA_TILE 16
#define BC_CUDA_GEOMETRY_WMMA_WARPS 8
#define BC_CUDA_GEOMETRY_WMMA_SCRATCH 1024

#endif /* model geometry */
#endif
