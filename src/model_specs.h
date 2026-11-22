#ifndef BRAINCHOPC_MODEL_SPECS_H
#define BRAINCHOPC_MODEL_SPECS_H

/*
 * This build's model, as constants.
 *
 * Nothing here is written by hand. scripts/gen_model_def.py derives it from
 * models/<name>/model.json -- the layer topology and the weights manifest --
 * checks the derived layout against the byte size of model.bin, and folds in
 * models/<name>/model_meta.json for what model.json cannot express (the
 * normalization quantiles, which GELU approximation, deliberate divergences
 * from the checkpoint, and which CLI options the model supports).
 * `make check-generated` fails the build if a checked-in header has drifted
 * from its model.
 *
 * The Makefile compiles each executable with -DBC_MODEL_IS_<NAME>, which is
 * what turns that model's namespaced constants into the generic BC_MODEL_*
 * names every shared source is written against. Adding a model that reuses an
 * existing GELU flavour therefore touches no code at all: drop its directory
 * into models/, run the generator, and add one block to the Makefile.
 */

#include "model_def.h"

#define BRAINCHOP_MODEL_DIM 256

/*
 * Which options this executable offers, straight from the model's
 * model_meta.json. These gate the help text as well as validation, so a binary
 * never advertises an option it would refuse.
 */
#define BC_SUPPORTS_CROP BC_MODEL_SUPPORTS_CROP
#define BC_SUPPORTS_MASK BC_MODEL_SUPPORTS_MASK
#define BC_SUPPORTS_BORDER BC_MODEL_SUPPORTS_BORDER
#define BC_SUPPORTS_SAVE_CONFORM BC_MODEL_SUPPORTS_SAVE_CONFORM


#ifdef __cplusplus
extern "C" {
#endif

/* This build's dilation schedule, defined once in model_specs.c. */
extern const int brainchop_model_dilation[BC_MODEL_NHIDDEN];

#ifdef __cplusplus
}
#endif

#endif
