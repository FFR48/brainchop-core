#ifndef BRAINCHOPC_PREPROCESS_H
#define BRAINCHOPC_PREPROCESS_H

#include "nifti_io.h"

typedef struct {
	int x_low, x_high;
	int y_low, y_high;
	int z_low, z_high;
} bc_crop_box;

int brainchopc_hounsfield_to_cormack(nifti_image *image);

int brainchopc_crop_box(const unsigned char *volume, int nx, int ny, int nz,
		double percentile, bc_crop_box *box);

/* Euclidean mask growth in physical millimeters, over a port of niimath's EDT
 * (BSD-2-Clause; see THIRD_PARTY_NOTICES.md). */
int brainchopc_grow_mask(float *mask, int nx, int ny, int nz,
		float dx, float dy, float dz, float millimeters);

#endif
