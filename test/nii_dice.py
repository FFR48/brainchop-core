#!/usr/bin/env python3
"""Gate a turbo (8-bit) label map against a reference, as brainchop-test does.

usage: nii_dice.py REF TEST MIN_MACRO_DICE MIN_FG_AGREE_PCT

Macro Dice over foreground classes present in either map; foreground agreement
is exact matches over voxels labelled foreground in either (compareLabels in
brainchop-test tests/fixtures/meshnet-v4-benchmark-page.js). uint8 labels only.
"""

import gzip
import struct
import sys
from collections import Counter


def labels(path):
    blob = (gzip.open if path.endswith(".gz") else open)(path, "rb").read()
    if struct.unpack_from("<h", blob, 70)[0] != 2:
        sys.exit(f"nii_dice: {path}: not uint8")
    return blob[int(struct.unpack_from("<f", blob, 108)[0]):]


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    ref, test = labels(sys.argv[1]), labels(sys.argv[2])
    if len(ref) != len(test):
        sys.exit("nii_dice: volumes differ in size")
    pairs = Counter(zip(ref, test))
    inter, n_ref, n_test = Counter(), Counter(), Counter()
    union = same = 0
    for (r, t), n in pairs.items():
        n_ref[r] += n
        n_test[t] += n
        if r == t:
            inter[r] += n
        if r or t:
            union += n
            same += n if r == t else 0
    classes = [c for c in set(n_ref) | set(n_test) if c]
    dice = [2 * inter[c] / (n_ref[c] + n_test[c]) for c in classes]
    macro = sum(dice) / max(1, len(dice))
    fg = 100 * same / max(1, union)
    print(f"nii_dice: macro Dice {macro:.5f} (min {min(dice, default=1):.4f}), "
          f"fg agree {fg:.3f}%, {union - same} fg voxels differ")
    if macro < float(sys.argv[3]) or fg < float(sys.argv[4]):
        sys.exit(f"nii_dice: below gate ({sys.argv[3]}, {sys.argv[4]}%)")


main()
