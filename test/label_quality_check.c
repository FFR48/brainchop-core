#include "bwlabel.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
	size_t dim[3] = {100, 1, 1};
	float image[100] = {0};
	assert(bwlabel_validate(image, dim, 18));
	for (int i = 0; i < 95; i++) image[i] = 1;
	assert(!bwlabel_validate(image, dim, 18));
	image[95] = 1;
	assert(bwlabel_validate(image, dim, 18));
	image[95] = 0;
	const float invalid[] = {NAN, INFINITY, -INFINITY, -1, 18, 1.5f};
	for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
		image[0] = invalid[i];
		assert(bwlabel_validate(image, dim, 18));
	}
	assert(bwlabel_validate(NULL, dim, 18));
	assert(bwlabel_validate(image, NULL, 18));
	size_t huge[3] = {SIZE_MAX, 2, 1};
	assert(bwlabel_validate(image, huge, 18));

	/* Exactly the cap, then cap+1; labels touch only across background. */
	size_t line[3] = {200002, 1, 1};
	float *noise = calloc(line[0] * 2, sizeof(float));
	assert(noise);
	for (size_t i = 0; i < 200000; i += 2) noise[i] = 1;
	assert(!bwlabel_validate(noise, line, 18));
	noise[200000] = 1;
	assert(bwlabel_validate(noise, line, 18));
	/* Every branch joins on the second row. The cheap root upper bound
	   exceeds the cap, but the actual component count is just one. */
	line[1] = 2;
	for (size_t i = line[0]; i < line[0] * 2; i++) noise[i] = 1;
	assert(!bwlabel_validate(noise, line, 18));
	for (size_t i = 0; i < line[0]; i++) assert(noise[i] == (i % 2 == 0));
	free(noise);

	/* Same-label connectivity, not just binary foreground connectivity. */
	size_t cube[3] = {64, 64, 64};
	noise = calloc(64 * 64 * 64, sizeof(float));
	assert(noise);
	for (size_t z = 0; z < 48; z++)
	for (size_t y = 0; y < 64; y++)
	for (size_t x = 0; x < 64; x++) noise[x + y*64 + z*4096] = 1 + (x+y+z)%2;
	assert(bwlabel_validate(noise, cube, 18));
	free(noise);
	puts("categorical quality: validity, occupancy, fragmentation and boundary gates PASS");
	return 0;
}
