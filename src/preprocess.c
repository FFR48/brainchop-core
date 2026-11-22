#include "preprocess.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int brainchopc_hounsfield_to_cormack(nifti_image *image) {
	if (!image || image->datatype != DT_FLOAT32 || !image->data) return 1;
	/* Lossless in the diagnostic range; values below air are intentionally clamped. */
	const float minimum = -1000.0f;
	const float dark_units = 900.0f;
	const float mid_units = 200.0f;
	const float scale_ratio = 10.0f;
	float *values = (float *)image->data;
	for (size_t i = 0; i < (size_t)image->nvox; i++) {
		float value = fmaxf(values[i], minimum) - minimum;
		float boost = fminf(fmaxf(value - dark_units, 0.0f), mid_units);
		values[i] = value + boost * (scale_ratio - 1.0f);
	}
	image->scl_slope = 1.0f;
	image->scl_inter = 0.0f;
	return 0;
}

static double u8_percentile(const int64_t *cumulative, size_t count, double q) {
	double rank = q * (double)(count - 1);
	double base = floor(rank);
	double fraction = rank - base;
	size_t index = (size_t)base;
	int low = 0;
	while (low < 255 && cumulative[low] <= (int64_t)index) low++;
	int high = low;
	while (high < 255 && cumulative[high] <= (int64_t)(index + 1)) high++;
	return (double)low + (double)(high - low) * fraction;
}

int brainchopc_crop_box(const unsigned char *volume, int nx, int ny, int nz,
		double percentile, bc_crop_box *box) {
	if (!volume || !box || nx < 1 || ny < 1 || nz < 1) return 1;
	size_t count = (size_t)nx * ny * nz;
	int64_t histogram[256] = {0}, cumulative[256], run = 0;
	for (size_t i = 0; i < count; i++) histogram[volume[i]]++;
	for (int i = 0; i < 256; i++) {
		run += histogram[i];
		cumulative[i] = run;
	}
	double cutoff = u8_percentile(cumulative, count, percentile / 100.0);
	box->x_low = nx;
	box->x_high = -1;
	box->y_low = ny;
	box->y_high = -1;
	/* brainchop-cli deliberately retains the complete first NumPy axis (NIfTI z). */
	box->z_low = 0;
	box->z_high = nz - 1;
	for (int z = 0; z < nz; z++)
		for (int y = 0; y < ny; y++)
			for (int x = 0; x < nx; x++) {
				size_t index = (size_t)x + (size_t)y * nx + (size_t)z * nx * ny;
				if ((double)volume[index] <= cutoff) continue;
				if (x < box->x_low) box->x_low = x;
				if (x > box->x_high) box->x_high = x;
				if (y < box->y_low) box->y_low = y;
				if (y > box->y_high) box->y_high = y;
			}
	if (box->x_high < box->x_low || box->y_high < box->y_low) {
		box->x_low = box->y_low = box->z_low = 0;
		box->x_high = box->y_high = box->z_high = -1;
	}
	return 0;
}

/* Lower envelope of parabolas: the Felzenszwalb/Huttenlocher EDT, ported from
 * niimath's vx()/edt() in src/coreFLT.c. BSD-2-Clause, Chris Rorden's Lab --
 * see THIRD_PARTY_NOTICES.md, which is where that notice lives. */
static float intersection(const float *values, int left, int right) {
	if (left == right) return INFINITY;
	float result = ((values[right] + right * right) -
		(values[left] + left * left)) / (2.0f * right - 2.0f * left);
	return isnan(result) ? INFINITY : result;
}

static void edt_line(float *values, int length, float spacing,
		float *distance, float *limits, int *vertices) {
	if (!(spacing > 0.0f)) spacing = 1.0f;
	if (spacing != 1.0f) {
		float inverse_square = 1.0f / (spacing * spacing);
		for (int i = 0; i < length; i++) values[i] *= inverse_square;
	}
	int envelope = 0;
	vertices[0] = 0;
	limits[0] = -INFINITY;
	limits[1] = INFINITY;
	for (int right = 1; right < length; right++) {
		float crossing = intersection(values, vertices[envelope], right);
		while (envelope > 0 && crossing <= limits[envelope]) {
			envelope--;
			crossing = intersection(values, vertices[envelope], right);
		}
		envelope++;
		vertices[envelope] = right;
		limits[envelope] = crossing;
		limits[envelope + 1] = INFINITY;
	}
	envelope = 0;
	for (int position = 0; position < length; position++) {
		while (limits[envelope + 1] < position) envelope++;
		float delta = (float)(position - vertices[envelope]);
		distance[position] = delta * delta + values[vertices[envelope]];
	}
	float square = spacing * spacing;
	for (int i = 0; i < length; i++) values[i] = distance[i] * square;
}

int brainchopc_grow_mask(float *mask, int nx, int ny, int nz,
		float dx, float dy, float dz, float millimeters) {
	if (!mask || nx < 2 || ny < 2 || nz < 1 || !(millimeters > 0.0f)) return 1;
	size_t count = (size_t)nx * ny * nz;
	float *field = (float *)malloc(count * sizeof(*field));
	int maximum = nx > ny ? nx : ny;
	if (nz > maximum) maximum = nz;
	float *line = (float *)malloc((size_t)maximum * sizeof(*line));
	float *distance = (float *)malloc((size_t)(maximum + 2) * sizeof(*distance));
	float *limits = (float *)malloc((size_t)(maximum + 2) * sizeof(*limits));
	int *vertices = (int *)malloc((size_t)(maximum + 2) * sizeof(*vertices));
	if (!field || !line || !distance || !limits || !vertices) {
		free(field); free(line); free(distance); free(limits); free(vertices);
		return 1;
	}
	for (size_t i = 0; i < count; i++) field[i] = mask[i] > 0.0f ? 0.0f : INFINITY;

	for (int z = 0; z < nz; z++)
		for (int y = 0; y < ny; y++) {
			float *row = field + ((size_t)z * ny + y) * nx;
			memcpy(line, row, (size_t)nx * sizeof(*line));
			edt_line(line, nx, fabsf(dx), distance, limits, vertices);
			memcpy(row, line, (size_t)nx * sizeof(*line));
		}
	for (int z = 0; z < nz; z++)
		for (int x = 0; x < nx; x++) {
			for (int y = 0; y < ny; y++)
				line[y] = field[(size_t)x + (size_t)y * nx + (size_t)z * nx * ny];
			edt_line(line, ny, fabsf(dy), distance, limits, vertices);
			for (int y = 0; y < ny; y++)
				field[(size_t)x + (size_t)y * nx + (size_t)z * nx * ny] = line[y];
		}
	for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++) {
			for (int z = 0; z < nz; z++)
				line[z] = field[(size_t)x + (size_t)y * nx + (size_t)z * nx * ny];
			edt_line(line, nz, fabsf(dz), distance, limits, vertices);
			for (int z = 0; z < nz; z++)
				field[(size_t)x + (size_t)y * nx + (size_t)z * nx * ny] = line[z];
		}
	for (size_t i = 0; i < count; i++)
		mask[i] = mask[i] > 0.0f || sqrtf(field[i]) < millimeters ? 1.0f : 0.0f;
	free(field); free(line); free(distance); free(limits); free(vertices);
	return 0;
}
