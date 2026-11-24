#!/usr/bin/env python3
"""Check that MindGrab preserves scaled integer NIfTI intensities.

Covers both positive-slope int16 (preventing >32767 clipping) and negative-slope
int64 with adjacent 64-bit maximum values that tie in double precision, and
verifies that both produce the exact same brain mask.
"""

from array import array
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from nii_diff import HEADER_BYTES, read, voxels


def to_little_endian_bytes(values):
    out = values if sys.byteorder == "little" else array(values.typecode, values)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def from_little_endian_bytes(blob, typecode):
    values = array(typecode)
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def run_mindgrab(executable, temp, name, header, payload):
    input_path = Path(temp) / f"{name}_in.nii"
    output_path = Path(temp) / f"{name}_out.nii"
    mask_path = Path(temp) / f"{name}_mask.nii"
    input_path.write_bytes(header + payload)
    subprocess.run(
        [executable, "-cpu", "0", "--mask", str(mask_path), "-o", str(output_path), str(input_path)],
        check=True,
    )
    return read(str(output_path)), voxels(read(str(mask_path)), mask_path)


def main(executable, fixture):
    source = read(fixture)
    original = voxels(source, fixture)
    minimum = min(original)

    with TemporaryDirectory(prefix="brainchop-mindgrab-scaling-") as temp:
        # 1. Positive-slope int16: decoded intensities exceed 32767.
        bg_i16 = -32768
        slope_i16, inter_i16 = 512.0, 512.0 * -bg_i16
        stored_i16 = array("h", (v - minimum + bg_i16 for v in original))
        hdr_i16 = bytearray(source[:HEADER_BYTES])
        struct.pack_into("<2h", hdr_i16, 70, 4, 16)
        struct.pack_into("<2f", hdr_i16, 112, slope_i16, inter_i16)

        res_i16, mask_i16 = run_mindgrab(
            executable, temp, "pos_i16", hdr_i16, to_little_endian_bytes(stored_i16)
        )
        assert set(mask_i16) == {0, 1}, "expected brain and background voxels"
        assert any(keep and raw * slope_i16 + inter_i16 > 32767
                   for raw, keep in zip(stored_i16, mask_i16))
        assert struct.unpack_from("<2h", res_i16, 70) == (4, 16), "int16 output datatype changed"
        assert struct.unpack_from("<2f", res_i16, 112) == (slope_i16, inter_i16), "int16 scaling changed"
        out_i16 = voxels(res_i16, "pos_i16_out.nii")
        assert len(out_i16) == len(stored_i16) == len(mask_i16), "int16 output size changed"
        assert all(value == (raw if keep else bg_i16)
                   for value, raw, keep in zip(out_i16, stored_i16, mask_i16)), "int16 voxel values changed"

        # 2. Negative-slope int64 with adjacent 64-bit extremes (top - 1 vs top tie in double).
        # Exact power-of-two scaling decodes to the same float32 intensities as Case 1.
        top_i64 = 1 << 60
        step_i64 = 1 << 45
        slope_i64 = -512.0 / float(step_i64)
        inter_i64 = 512.0 * float(top_i64 // step_i64)
        stored_i64 = array("q", (top_i64 - (v - minimum) * step_i64 for v in original))
        first_min = next(i for i, v in enumerate(original) if v == minimum)
        assert any(i != first_min and v == minimum for i, v in enumerate(original))
        stored_i64[first_min] = top_i64 - 1
        hdr_i64 = bytearray(source[:HEADER_BYTES])
        struct.pack_into("<2h", hdr_i64, 70, 1024, 64)
        struct.pack_into("<2f", hdr_i64, 112, slope_i64, inter_i64)

        res_i64, mask_i64 = run_mindgrab(
            executable, temp, "neg_i64", hdr_i64, to_little_endian_bytes(stored_i64)
        )
        assert mask_i64 == mask_i16, "negative-slope int64 mask diverged from positive-slope int16 mask"
        assert struct.unpack_from("<2h", res_i64, 70) == (1024, 64), "int64 output datatype changed"
        assert struct.unpack_from("<2f", res_i64, 112) == (slope_i64, inter_i64), "int64 scaling changed"
        out_i64 = from_little_endian_bytes(res_i64[HEADER_BYTES:], "q")
        assert len(out_i64) == len(stored_i64) == len(mask_i64), "int64 output size changed"
        assert all(value == (raw if keep else top_i64)
                   for value, raw, keep in zip(out_i64, stored_i64, mask_i64)), "int64 voxel values changed"

    print("MindGrab scaling: positive int16 and negative int64 voxel values and scaling preserved")


if __name__ == "__main__":
    main(*sys.argv[1:])
