#ifndef BRAINCHOPC_CAT_LITE_H
#define BRAINCHOPC_CAT_LITE_H

#include "model_specs.h"
#include "model_capabilities.h"
#include <stddef.h>

/*
 * CAT-lite partial-volume estimation, ported from ../brainchop-test/cat-lite.js.
 *
 * Availability is an explicit model capability. The tissue groups were tuned
 * and validated for MindMap; sharing its 18-class label count is not enough to
 * establish that another model supports the CAT-lite fit.
 */
#define BC_HAS_PVE BC_MODEL_SUPPORTS_PVE

/*
 * --pve and --crop cannot coexist, and the coupling is otherwise invisible.
 * The engine fills `planes` only inside the inference region, so with an ROI
 * the blur would smear uninitialised heap across the rest of the volume and the
 * fit would model it -- a brain-shaped wrong answer, not a crash. The two
 * A PVE-capable model must declare "crop": false, so assert it
 * so a model that declares both fails to compile rather than at a reader's desk.
 */
#if BC_HAS_PVE
_Static_assert(!BC_MODEL_SUPPORTS_CROP,
	"a model cannot offer both --pve and --crop: the PVE planes are only "
	"written inside the inference region");
#endif

#if BC_HAS_PVE

/* Layout of the buffer the engine fills and this module consumes. */
#define BC_PVE_GM 0
#define BC_PVE_WM 1
#define BC_PVE_CSF 2
#define BC_PVE_SUPPORT 3
#include "pve_constants.h"
#define BC_PVE_PLANES BC_PRIOR_PLANES
/* Gaussian sigma of the prior blur, in voxels. A double literal: the Metal
   head stringifies it, and cat_lite.c's double arithmetic must not change. */
#define BC_PVE_BLUR_SIGMA 0.65

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The two halves of CAT-lite, split at the boundary cat-lite.js itself draws.
 *
 * `planes` holds BC_PVE_PLANES volumes of `count` voxels: the three grouped
 * tissue priors as the engine produced them, then the brain support.
 *
 * bc_cat_lite_priors is the page's WGSL tail -- a separable blur, then the
 * support gate -- and rewrites the first three planes in place.
 * bc_cat_lite_fit is applyCatLitePartialVolume: it overwrites those three with
 * the CAT-lite GM, WM and CSF fractions in [0, 1], reading `intensity`, the
 * normalized conformed volume the network saw.
 *
 * They are separate because brainchopc.c writes BRAINCHOPC_PVE_DUMP between
 * them and test/cat_lite_check.c drives the FIT alone against the JS on
 * identical priors, which is what tests the port rather than the fp16 network
 * feeding it.
 *
 * Both return 0 on success. A refusal to FIT (too few confident voxels, an
 * implausible tissue ordering) leaves the support-gated priors in place and
 * still returns 0: a plausible fallback beats failing a segmentation over a fit.
 */
int bc_cat_lite_priors(float *planes, const size_t dim[3]);
int bc_cat_lite_fit(float *planes, const float *intensity, const size_t dim[3]);

/* Optional fit diagnostics for the browser adapter. Fixed-width fields and
   six leading doubles make the wasm layout explicit (64 bytes total). */
typedef struct {
    double mean[3], sigma[3];
    unsigned int component_count, removed_voxels;
    int applied, refusal; /* refusal: 1=too few voxels, 2=tissue ordering */
} bc_cat_lite_report;
_Static_assert(sizeof(bc_cat_lite_report) == 64 &&
        offsetof(bc_cat_lite_report, component_count) == 48,
        "CAT-lite browser report layout must remain stable");
int bc_cat_lite_fit_report(float *planes, const float *intensity,
        const size_t dim[3], bc_cat_lite_report *report);

#ifdef __cplusplus
}
#endif
#endif /* BC_HAS_PVE */
#endif
