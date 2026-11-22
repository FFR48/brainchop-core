#include "bwlabel.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Keep the largest nonzero component of a 3-D label image. The general label
 * image and bubble-filling machinery from niimath is deliberately omitted.
 * This is an independent queue flood fill, not niimath's Andersson/SPM code.
 */
int bwlabel_largest(float *img, const size_t dim[3], int binary_output,
		int connectivity) {
	if (!img || !dim || dim[0] < 1 || dim[1] < 1 || dim[2] < 1)
		return 1;
	if (connectivity != 6 && connectivity != 26) return 1;
	if (dim[0] > SIZE_MAX / dim[1] || dim[0] * dim[1] > SIZE_MAX / dim[2])
		return 1;
	const size_t nxy = dim[0] * dim[1], n = nxy * dim[2];
	uint32_t *labels = calloc(n, sizeof(*labels));
	size_t *queue = malloc(n * sizeof(*queue));
	if (!labels || !queue) {
		fprintf(stderr, "bwlabel: out of memory\n");
		free(labels);
		free(queue);
		return 1;
	}
	uint32_t label = 0, best_label = 0;
	size_t best_count = 0;
	for (size_t seed = 0; seed < n; seed++) {
		if (img[seed] <= 0.0f || labels[seed]) continue;
		label++;
		size_t head = 0, tail = 0;
		queue[tail++] = seed;
		labels[seed] = label;
		while (head < tail) {
			size_t p = queue[head++];
			size_t z = p / nxy, rem = p - z * nxy;
			size_t y = rem / dim[0], x = rem - y * dim[0];
			for (int dz = -1; dz <= 1; dz++)
			for (int dy = -1; dy <= 1; dy++)
			for (int dx = -1; dx <= 1; dx++) {
				if (!(dx || dy || dz)) continue;
				if (connectivity == 6 &&
						(dx != 0) + (dy != 0) + (dz != 0) != 1) continue;
				int64_t xx = (int64_t)x + dx, yy = (int64_t)y + dy, zz = (int64_t)z + dz;
				if (xx < 0 || yy < 0 || zz < 0 ||
				    xx >= (int64_t)dim[0] || yy >= (int64_t)dim[1] || zz >= (int64_t)dim[2])
					continue;
				size_t q = (size_t)xx + (size_t)yy * dim[0] + (size_t)zz * nxy;
				if (img[q] > 0.0f && !labels[q]) {
					labels[q] = label;
					queue[tail++] = q;
				}
			}
		}
		if (tail > best_count) {
			best_count = tail;
			best_label = label;
		}
	}
	for (size_t i = 0; i < n; i++) {
		if (labels[i] != best_label || best_label == 0) img[i] = 0.0f;
		else if (binary_output) img[i] = 1.0f;
	}
	free(labels);
	free(queue);
	return 0;
}

/* Independent flood-fill implementation of brainchop-test's 18-class path.
   Two uint32 volumes (128 MiB at 256^3) are reused for both labeling passes.
   Work is linear in voxels/contacts, including fragmented inputs: no pairwise
   component scans or per-component full-volume scans. */
typedef struct {
	uint32_t seed, count;
	unsigned char klass, kept, replacement;
} bc_component;

static int face_neighbors(uint32_t p, const size_t dim[3], uint32_t q[6]) {
	size_t xy = dim[0] * dim[1], z = p / xy;
	size_t y = (p % xy) / dim[0], x = p % dim[0];
	int n = 0;
	if (x) q[n++] = p - 1;
	if (x + 1 < dim[0]) q[n++] = p + 1;
	if (y) q[n++] = p - (uint32_t)dim[0];
	if (y + 1 < dim[1]) q[n++] = p + (uint32_t)dim[0];
	if (z) q[n++] = p - (uint32_t)xy;
	if (z + 1 < dim[2]) q[n++] = p + (uint32_t)xy;
	return n;
}

static uint32_t flood6(const float *img, const size_t dim[3], uint32_t *map,
		uint32_t *queue, uint32_t seed, uint32_t id, int binary) {
	uint32_t head = 0, tail = 1;
	queue[0] = seed;
	map[seed] = id;
	while (head < tail) {
		uint32_t q[6];
		int nq = face_neighbors(queue[head++], dim, q);
		for (int j = 0; j < nq; j++) {
			uint32_t p = q[j];
			if (!map[p] && (binary ? img[p] > 0 : img[p] == img[seed])) {
				map[p] = id;
				queue[tail++] = p;
			}
		}
	}
	return tail;
}

int bwlabel_validate(const float *img, const size_t dim[3], unsigned classes) {
	if (!img || !dim || !classes || !dim[0] || !dim[1] || !dim[2] ||
			dim[0] > SIZE_MAX / dim[1] || dim[0] * dim[1] > SIZE_MAX / dim[2]) {
		fprintf(stderr, "brainchopc: invalid categorical volume dimensions\n");
		return 1;
	}
	const size_t xy = dim[0] * dim[1], n = xy * dim[2];
	if (n > UINT32_MAX || n > SIZE_MAX / sizeof(uint32_t)) return 1;
	size_t foreground = 0, roots = 0;
	for (size_t i = 0; i < n; i++) {
		float v = img[i];
		if (!isfinite(v) || v < 0 || (double)v >= classes || v != floorf(v)) {
			fprintf(stderr, "brainchopc: invalid categorical label at voxel %zu: %g\n", i, (double)v);
			return 1;
		}
		if (!v) continue;
		foreground++;
		/* Every six-connected same-label component has a first voxel with no
		   preceding face neighbor. Their count is an UPPER bound on components:
		   disconnected branches can merge later. Healthy output usually proves
		   the cap without allocating or running another whole-volume flood fill. */
		if ((i % dim[0] == 0 || img[i-1] != v) &&
				((i % xy) < dim[0] || img[i-dim[0]] != v) &&
				(i < xy || img[i-xy] != v)) roots++;
	}
	if (!foreground || (double)foreground > (double)n * 0.95) {
		fprintf(stderr, "brainchopc: categorical foreground is %zu/%zu voxels; %s\n",
			foreground, n, foreground ? "segmentation is near-filled" : "segmentation is empty");
		return 1;
	}
	size_t cap = n / 100;
	if (cap < 100000) cap = 100000;
	if (roots <= cap) return 0;
	unsigned char *visited = calloc(n, 1);
	uint32_t *queue = malloc(foreground * sizeof(*queue));
	if (!visited || !queue) {
		free(visited); free(queue);
		fprintf(stderr, "brainchopc: unable to allocate categorical quality workspace\n");
		return 1;
	}
	size_t components = 0;
	for (size_t i = 0; i < n; i++) {
		if (!img[i] || visited[i]) continue;
		if (++components > cap) break;
		size_t head = 0, tail = 1;
		queue[0] = (uint32_t)i; visited[i] = 1;
		while (head < tail) {
			uint32_t q[6];
			int nq = face_neighbors(queue[head++], dim, q);
			for (int j = 0; j < nq; j++)
				if (!visited[q[j]] && img[q[j]] == img[i]) {
					visited[q[j]] = 1;
					queue[tail++] = q[j];
				}
		}
	}
	free(visited); free(queue);
	if (components > cap)
		fprintf(stderr, "brainchopc: categorical segmentation is excessively fragmented "
			"(more than %zu six-connected regions)\n", cap);
	return components > cap;
}

static int targeted18(unsigned klass) {
	return klass == 1 || klass == 2 || klass == 5 || klass == 6 || klass == 13;
}

static int targeted_all(unsigned klass) {
	return klass > 0;
}

static unsigned dk104_cortical_family(unsigned klass) {
	if (klass >= 1 && klass <= 34) return 1;
	if (klass >= 35 && klass <= 68) return 2;
	return 0;
}

static int bwlabel_cleanup_hierarchical(float *img, const size_t dim[3],
		unsigned nclass, int (*is_targeted)(unsigned), int larger_index_wins,
		unsigned (*family_of)(unsigned)) {
	if (!img || !dim || !nclass || nclass > 256 || !dim[0] || !dim[1] || !dim[2] ||
			dim[0] > SIZE_MAX / dim[1] || dim[0] * dim[1] > SIZE_MAX / dim[2])
		return 1;
	size_t n = dim[0] * dim[1] * dim[2];
	/* Reserve the high bit as a visited flag for the boundary-voting pass. */
	if (n >= UINT32_C(0x80000000) || n > SIZE_MAX / sizeof(uint32_t)) return 1;
	for (size_t i = 0; i < n; i++)
		if (!isfinite(img[i]) || img[i] < 0 || (double)img[i] >= (double)nclass ||
				img[i] != (float)(unsigned)img[i]) return 1;
	uint32_t *map = calloc(n, sizeof(*map));
	uint32_t *queue = malloc(n * sizeof(*queue));
	bc_component *comp = NULL;
	int failed = 1;
	if (!map || !queue) goto done;
	uint32_t count = 0, best = 0, best_count = 0;
	for (size_t i = 0; i < n; i++) {
		if (!img[i] || map[i]) continue;
		uint32_t size = flood6(img, dim, map, queue, (uint32_t)i, ++count, 1);
		/* Browser binary LCC keeps the LAST tie. */
		if (size >= best_count) { best = count; best_count = size; }
	}
	for (size_t i = 0; i < n; i++)
		if (map[i] != best) img[i] = 0;
	memset(map, 0, n * sizeof(*map));
	count = 0;
	size_t capacity = 0;
	uint32_t largest[256] = {0};
	for (size_t i = 0; i < n; i++) {
		if (!img[i] || map[i]) continue;
		if ((size_t)count + 1 >= capacity) {
			size_t next = capacity ? capacity * 2 : 1024;
			if (next > n + 1) next = n + 1;
			if (next > SIZE_MAX / sizeof(*comp)) goto done;
			bc_component *grown = realloc(comp, next * sizeof(*comp));
			if (!grown) goto done;
			comp = grown;
			if (!capacity) memset(&comp[0], 0, sizeof(*comp));
			capacity = next;
		}
		unsigned klass = (unsigned)img[i];
		uint32_t size = flood6(img, dim, map, queue, (uint32_t)i, ++count, 0);
		comp[count] = (bc_component){(uint32_t)i, size, (unsigned char)klass, 0, 0};
		/* 18-class filter_clusters keeps the FIRST tie; 104-class keeps the LAST. */
		if (!largest[klass] || (larger_index_wins ? size >= comp[largest[klass]].count
		                                          : size > comp[largest[klass]].count))
			largest[klass] = count;
	}
	for (uint32_t c = 1; c <= count; c++) {
		unsigned klass = comp[c].klass;
		if (!is_targeted(klass) || largest[klass] == c)
			comp[c].kept = comp[c].replacement = (unsigned char)klass;
	}
	const uint32_t visited = UINT32_C(0x80000000);
	for (uint32_t c = 1; c <= count; c++) {
		if (comp[c].kept) continue;
		uint64_t votes[256] = {0};
		uint32_t head = 0, tail = 1;
		queue[0] = comp[c].seed;
		map[queue[0]] |= visited;
		while (head < tail) {
			uint32_t q[6];
			int nq = face_neighbors(queue[head++], dim, q);
			for (int j = 0; j < nq; j++) {
				uint32_t id = map[q[j]];
				if (id == c) {
					map[q[j]] |= visited;
					queue[tail++] = q[j];
				} else {
					unsigned klass = comp[id & ~visited].kept;
					if (klass) votes[klass]++;
				}
			}
		}
		unsigned winner = 0;
		uint64_t best_votes = 0;
		unsigned fam = family_of ? family_of(comp[c].klass) : 0;
		if (fam) {
			for (unsigned k = 1; k < nclass; k++)
				if (family_of(k) == fam && votes[k] > best_votes) {
					best_votes = votes[k];
					winner = k;
				}
		}
		if (!winner) {
			best_votes = 0;
			for (unsigned k = 1; k < nclass; k++)
				if (votes[k] > best_votes) {
					best_votes = votes[k];
					winner = k;
				}
		}
		comp[c].replacement = (unsigned char)winner;
	}
	for (size_t i = 0; i < n; i++)
		img[i] = count ? comp[map[i] & ~visited].replacement : 0;
	failed = 0;
done:
	if (failed) fprintf(stderr, "bwlabel: unable to allocate label-cleanup workspace\n");
	free(comp); free(queue); free(map);
	return failed;
}

int bwlabel_cleanup18(float *img, const size_t dim[3]) {
	return bwlabel_cleanup_hierarchical(img, dim, 18, targeted18, 0, NULL);
}

int bwlabel_cleanup104(float *img, const size_t dim[3]) {
	return bwlabel_cleanup_hierarchical(img, dim, 104, targeted_all, 1,
		dk104_cortical_family);
}
