#include "bwlabel.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void selftest_18(void) {
	size_t dim[3] = {9, 1, 1};
	/* One connected brain; first tied class-1 component survives. Both
	   class-3 components survive. The suppressed 1 takes class 3. */
	float img[] = {1,1,3,1,1,3,0,0,0};
	float want[] = {1,1,3,3,3,3,0,0,0};
	assert(!bwlabel_cleanup18(img, dim));
	assert(!memcmp(img, want, sizeof(img)));
	float empty[9] = {0};
	assert(!bwlabel_cleanup18(empty, dim));
	assert(!bwlabel_cleanup18(empty, dim));
	const float invalid[] = {NAN, INFINITY, -1, 18, 1.5f};
	for (size_t i = 0; i < sizeof(invalid)/sizeof(*invalid); i++) {
		empty[0] = invalid[i];
		assert(bwlabel_cleanup18(empty, dim));
	}
	assert(bwlabel_cleanup18(NULL, dim));
	assert(bwlabel_cleanup18(img, NULL));
	size_t huge[3] = {SIZE_MAX, 2, 2};
	assert(bwlabel_cleanup18(img, huge));
}

static void selftest_104(void) {
	/* 1. Resection-cavity fragment test: a disconnected island inside a cavity
	   carrying a unique class (77, hippocampus) that appears nowhere else MUST
	   be dropped to 0 by Step 1 binary 6-connected main-component gating. */
	{
		size_t dim[3] = {10, 1, 1};
		/* Main brain mass at x=0..4 (size 5); air gap at x=5..6; disconnected
		   resection-cavity fragment at x=7..8 (size 2, unique class 77). */
		float img[]  = {85, 85, 1, 1, 2, 0, 0, 77, 77, 0};
		float want[] = {85, 85, 1, 1, 2, 0, 0,  0,  0, 0};
		assert(!bwlabel_cleanup104(img, dim));
		assert(!memcmp(img, want, sizeof(img)));
	}
	/* 2. Per-class tie-break: 104-class keeps the LAST tied component
	   (largerIndexWins = true, matching brainchop-test/bwlabels.js). */
	{
		size_t dim[3] = {9, 1, 1};
		/* Main brain mass x=0..5: two equal-size components of class 85
		   separated by class 70 (kept). Later class-85 component (x=3..4) wins,
		   earlier class-85 component (x=0..1) relabels to kept neighbor 70. */
		float img[]  = {85, 85, 70, 85, 85, 70, 0, 0, 0};
		/* Class 70 has two 1-voxel components at x=2 and x=5; later (x=5) wins.
		   So kept components are: class 85 at x=3..4 and class 70 at x=5.
		   Suppressed x=2 (class 70) touches kept 85 -> becomes 85.
		   Suppressed x=0..1 (class 85) touches only suppressed x=2 -> becomes 0! */
		float want[] = {0, 0, 85, 85, 85, 70, 0, 0, 0};
		assert(!bwlabel_cleanup104(img, dim));
		assert(!memcmp(img, want, sizeof(img)));
	}
	/* 3. Cortical family gating (1..34 LH cortex vs 85 WM): a suppressed LH
	   cortical fragment (class 3, fam 1) resting on 4 WM faces (class 85, fam 0)
	   and touching 1 neighboring LH cortical face (class 4, fam 1) MUST take
	   class 4 (same cortical hemisphere), NOT class 85 (WM). */
	{
		size_t dim[3] = {5, 3, 1};
		/* y=0: kept WM (85) across x=0..4 (size 5)
		   y=1: x=0..1 kept LH parcel 3 (size 2), x=2 kept LH parcel 4 (size 2 with y=2),
		        x=3 suppressed LH parcel 3 (size 1, touches 85 at y=0, 85 at x=4,
		        and 4 at x=2 -> 2 WM contacts vs 1 LH-cortex contact!)
		        x=4 kept WM (85) */
		float img[15] = {
			85, 85, 85, 85, 85,
			 3,  3,  4,  3, 85,
			 0,  0,  4, 85, 85
		};
		/* At (3,1): neighbors are (2,1)=4 [fam 1], (4,1)=85 [fam 0], (3,0)=85 [fam 0],
		   (3,2)=85 [fam 0]. Three WM contacts vs one LH-cortex contact -> family
		   gate MUST pick 4! */
		assert(!bwlabel_cleanup104(img, dim));
		assert(img[1 * 5 + 3] == 4.0f);
		assert(img[1 * 5 + 0] == 3.0f);
		assert(img[1 * 5 + 1] == 3.0f);
	}
	/* 4. Cross-hemisphere cortical isolation: suppressed LH parcel (class 3, fam 1)
	   touching 3 faces of RH cortex (class 40, fam 2) and 1 face of LH cortex
	   (class 5, fam 1) MUST take LH cortex (class 5). */
	{
		size_t dim[3] = {5, 3, 1};
		float img[15] = {
			40, 40, 40, 40, 40,
			 3,  3,  5,  3, 40,
			 0,  0,  5, 40, 40
		};
		assert(!bwlabel_cleanup104(img, dim));
		assert(img[1 * 5 + 3] == 5.0f);
	}
	/* 5. Family fallback: suppressed LH cortical speck (class 3, fam 1) completely
	   buried inside WM (class 85, fam 0) has no same-family neighbor -> falls
	   back to unrestricted vote and takes 85 (WM). */
	{
		size_t dim[3] = {5, 3, 1};
		float img[15] = {
			85, 85, 85, 85, 85,
			 3,  3, 85,  3, 85,
			85, 85, 85, 85, 85
		};
		assert(!bwlabel_cleanup104(img, dim));
		assert(img[1 * 5 + 3] == 85.0f);
	}
	/* 6. Empty and invalid inputs for 104-class cleanup. */
	{
		size_t dim[3] = {9, 1, 1};
		float empty[9] = {0};
		assert(!bwlabel_cleanup104(empty, dim));
		const float invalid[] = {NAN, INFINITY, -1, 104, 1.5f};
		for (size_t i = 0; i < sizeof(invalid)/sizeof(*invalid); i++) {
			empty[0] = invalid[i];
			assert(bwlabel_cleanup104(empty, dim));
		}
		assert(bwlabel_cleanup104(NULL, dim));
		assert(bwlabel_cleanup104(empty, NULL));
		size_t huge[3] = {SIZE_MAX, 2, 2};
		assert(bwlabel_cleanup104(empty, huge));
	}
}

int main(int argc, char **argv) {
	if (argc == 2 && !strcmp(argv[1], "--selftest")) {
		selftest_18();
		selftest_104();
		puts("label cleanup: 18-class & 104-class invariants and invalid inputs PASS");
		return 0;
	}
	int mode104 = 0;
	int arg_off = 1;
	if (argc == 7 && !strcmp(argv[1], "--104")) {
		mode104 = 1;
		arg_off = 2;
	} else if (argc != 6) {
		return 2;
	}
	size_t dim[3], n = 1;
	for (int i = 0; i < 3; i++) {
		char *end;
		unsigned long v = strtoul(argv[arg_off + 2 + i], &end, 10);
		if (*end || !v || v > 256) return 2;
		dim[i] = v; n *= v;
	}
	float *img = malloc(n * sizeof(*img));
	FILE *f = fopen(argv[arg_off], "rb");
	if (!img || !f || fread(img, sizeof(*img), n, f) != n) return 1;
	if (fgetc(f) != EOF || fclose(f)) return 1;
	if (mode104 ? bwlabel_cleanup104(img, dim) : bwlabel_cleanup18(img, dim)) return 1;
	f = fopen(argv[arg_off + 1], "wb");
	if (!f || fwrite(img, sizeof(*img), n, f) != n || fclose(f)) return 1;
	free(img);
	return 0;
}
