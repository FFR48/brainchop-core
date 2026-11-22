#ifndef BRAINCHOP_PVE_CONSTANTS_H
#define BRAINCHOP_PVE_CONSTANTS_H
/* Shared with the upstream 18-class CAT-lite probability head. Labels 0
 * (unknown) and 13 (brain stem) deliberately belong to no tissue group. */
#define BC_PRIOR_TEMPERATURE 2.0f
#define BC_PRIOR_SUPPORT_TEMPERATURE 1.0f
#define BC_PRIOR_GM_MASK ((1u<<2)|(1u<<6)|(1u<<7)|(1u<<8)|(1u<<9)|(1u<<10)| \
                        (1u<<14)|(1u<<15)|(1u<<16)|(1u<<17))
#define BC_PRIOR_WM_MASK ((1u<<1)|(1u<<5))
#define BC_PRIOR_CSF_MASK ((1u<<3)|(1u<<4)|(1u<<11)|(1u<<12))
#define BC_PRIOR_PLANES 4
#endif
