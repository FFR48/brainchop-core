#ifndef BRAINCHOPC_H
#define BRAINCHOPC_H

#include "nifti_io.h"
#include <stddef.h>

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define LOAD_MAT44(A,a,b,c,d,e,f,g,h,i,j,k,l) \
	((A).m[0][0]=(a),(A).m[0][1]=(b),(A).m[0][2]=(c),(A).m[0][3]=(d), \
	 (A).m[1][0]=(e),(A).m[1][1]=(f),(A).m[1][2]=(g),(A).m[1][3]=(h), \
	 (A).m[2][0]=(i),(A).m[2][1]=(j),(A).m[2][2]=(k),(A).m[2][3]=(l), \
	 (A).m[3][0]=0,(A).m[3][1]=0,(A).m[3][2]=0,(A).m[3][3]=1)

typedef struct { float v[4]; } vec4;

int nii_mul_size(size_t a, size_t b, size_t *out);
void *nii_malloc(size_t count, size_t size);
vec4 setVec4(float x, float y, float z);
vec4 nifti_vect44mat44_mul(vec4 v, mat44 m);
int brainchopc_convert(nifti_image *nim, int datatype);
int brainchopc_comply(nifti_image *nim);
int brainchopc_conform(nifti_image *nim);
enum { BC_RESAMPLE_NEAREST = 0, BC_RESAMPLE_LINEAR = 1, BC_RESAMPLE_LABELS = 2 };
/* Onto a grid within 1e-3 voxel of the source's, this copies (see do_reslice):
   LINEAR then keeps the edge faces the gather would fill with the minimum. */
int brainchopc_reslice(nifti_image *source, const nifti_image *target, int linear);
#endif
