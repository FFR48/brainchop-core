#!/usr/bin/env python3
"""GPU-free tests for the HIP artifact contract."""
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hip_artifacts", ROOT / "scripts/hip_artifacts.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pack(value):
    if value is None:
        return b"\xc0"
    if value is True:
        return b"\xc3"
    if value is False:
        return b"\xc2"
    if isinstance(value, int):
        if 0 <= value < 128:
            return bytes([value])
        if value < 256:
            return b"\xcc" + bytes([value])
        if value < 65536:
            return b"\xcd" + value.to_bytes(2, "big")
        return b"\xce" + value.to_bytes(4, "big")
    if isinstance(value, str):
        data = value.encode()
        return (bytes([0xa0 + len(data)]) if len(data) < 32 else b"\xd9" + bytes([len(data)])) + data
    if isinstance(value, list):
        return bytes([0x90 + len(value)]) + b"".join(pack(v) for v in value)
    if isinstance(value, dict):
        return bytes([0x80 + len(value)]) + b"".join(pack(k) + pack(v) for k, v in value.items())
    raise TypeError(type(value))


def code_object(entry="kernel", target="gfx1151", static_lds=64, private_bytes=0, vgpr_spills=0, sgpr_spills=0):
    arguments = [
        {".address_space": "global", ".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
        {".address_space": "global", ".offset": 8, ".size": 8, ".value_kind": "global_buffer"},
    ]
    note = {"amdhsa.kernels": [{".args": arguments, ".group_segment_fixed_size": static_lds,
                                ".kernarg_segment_size": 16, ".name": entry,
                                ".private_segment_fixed_size": private_bytes,
                                ".symbol": entry + ".kd",
                                ".vgpr_spill_count": vgpr_spills,
                                ".sgpr_spill_count": sgpr_spills}],
            "amdhsa.target": "amdgcn-amd-amdhsa--" + target,
            "amdhsa.version": [1, 2]}
    elf = bytearray(b"\x7fELF\x02\x01" + bytes(58))
    elf[16:18] = (3).to_bytes(2, "little")
    return bytes(elf) + b"AMDGPU\x00" + pack(note)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = b"model graph"
        (self.root / "model.onnx").write_bytes(self.source)
        self.kernel = code_object()
        (self.root / "kernels.bin").write_bytes(self.kernel)
        self.manifest = {
            "schema_version": 1, "abi_version": 1,
            "model": {"name": "mindgrab", "descriptor_sha256": "1" * 64,
                      "dimension": 256, "channels": 15,
                      "hidden_layers": 25, "classes": 2, "has_affine": False,
                      "classifier_bias": False},
            "target": {"architecture": "gfx1151", "num_compute_units": 40,
                       "num_chiplets": 1, "wavefront_size": 32},
            "toolchain": {"rocm": "7.2.2", "migraphx": "2.18", "tinygrad": "0.14.0",
                          "compile_options": {"fp16": True}},
            "sources": {"model.onnx": sha(self.source)},
            "bundle": {"file": "kernels.bin", "size": len(self.kernel), "sha256": sha(self.kernel)},
            "buffers": [{"id": "arena", "bytes": 4096, "alignment": 4096, "kind": "arena"},
                        {"id": "input", "bytes": 32, "alignment": 16, "kind": "input"},
                        {"id": "output", "bytes": 16, "alignment": 16, "kind": "output"}],
            "kernels": [{"id": "kernel", "offset": 0, "size": len(self.kernel),
                         "sha256": sha(self.kernel), "entry": "kernel", "symbol": "kernel.kd",
                         "producer": "tinygrad", "target": "gfx1151", "kernarg_segment_size": 16,
                         "static_lds_bytes": 64,
                         "arguments": [{"offset": 0, "size": 8, "value_kind": "global_buffer"},
                                       {"offset": 8, "size": 8, "value_kind": "global_buffer"}]}],
            "dispatches": [{"kernel": "kernel", "grid": [1, 1, 1], "block": [32, 1, 1],
                            "dynamic_lds_bytes": 0,
                            "args": [{"kind": "buffer", "buffer": "input", "offset": 0,
                                      "bytes": 32, "access": "read"},
                                     {"kind": "buffer", "buffer": "output", "offset": 0,
                                      "bytes": 16, "access": "write"}]}],
            "weights": [{"tensor": "conv0", "buffer": "arena", "offset": 0, "bytes": 128,
                         "source_dtype": "f32", "target_dtype": "f16", "layout": "dhwio"}],
        }
        self.write()

    def write(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def verify(self):
        self.write()
        return m.verify(self.root, self.root)

    def test_valid_artifact(self):
        self.assertEqual(self.verify()["model"]["name"], "mindgrab")

    def test_corrupt_bundle(self):
        (self.root / "kernels.bin").write_bytes(self.kernel + b"x")
        with self.assertRaisesRegex(ValueError, "corrupt HIP kernel bundle"):
            self.verify()

    def test_model_and_architecture_are_bound(self):
        original = copy.deepcopy(self.manifest)
        for path, value in ((["model", "descriptor_sha256"], "bad"),
                            (["target", "architecture"], "gfx!")):
            with self.subTest(path=path):
                bad = copy.deepcopy(original)
                bad[path[0]][path[1]] = value
                self.manifest = bad
                with self.assertRaises(ValueError):
                    self.verify()
        self.manifest = original

    def test_kernel_target_mismatch(self):
        bad = copy.deepcopy(self.manifest)
        bad["kernels"][0]["target"] = "gfx1100"
        self.manifest = bad
        with self.assertRaisesRegex(ValueError, "producer or target"):
            self.verify()

    def test_elf_metadata_mismatch(self):
        for key, value in (("entry", "wrong"), ("static_lds_bytes", 0),
                           ("kernarg_segment_size", 24)):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.manifest)
                bad["kernels"][0][key] = value
                (self.root / "manifest.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "AMDGPU metadata mismatch"):
                    m.verify(self.root, self.root)

    def test_relocatable_kernel_is_rejected(self):
        kernel = bytearray(self.kernel)
        kernel[16:18] = (1).to_bytes(2, "little")
        self.kernel = bytes(kernel)
        (self.root / "kernels.bin").write_bytes(self.kernel)
        self.manifest["bundle"].update(
            size=len(self.kernel), sha256=sha(self.kernel))
        self.manifest["kernels"][0].update(
            size=len(self.kernel), sha256=sha(self.kernel))
        with self.assertRaisesRegex(ValueError, "not a loadable ELF shared object"):
            self.verify()

    def test_static_lds_is_not_dynamic_lds(self):
        self.assertEqual(self.manifest["kernels"][0]["static_lds_bytes"], 64)
        self.assertEqual(self.manifest["dispatches"][0]["dynamic_lds_bytes"], 0)
        self.verify()

    def test_buffer_bounds(self):
        self.manifest["dispatches"][0]["args"][0]["offset"] = 1
        with self.assertRaisesRegex(ValueError, "invalid buffer argument"):
            self.verify()

    def test_kernel_argument_count(self):
        self.manifest["dispatches"][0]["args"].pop()
        with self.assertRaisesRegex(ValueError, "argument count"):
            self.verify()

    def test_overlapping_kernel_slices(self):
        other = copy.deepcopy(self.manifest["kernels"][0])
        other["id"] = "other"
        self.manifest["kernels"].append(other)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.verify()

    def test_path_traversal_is_rejected(self):
        self.manifest["bundle"]["file"] = "../kernels.bin"
        with self.assertRaisesRegex(ValueError, "basename"):
            self.verify()

    def test_unknown_fields_are_rejected(self):
        self.manifest["surprise"] = True
        with self.assertRaisesRegex(ValueError, "manifest keys"):
            self.verify()

    def test_source_drift_is_rejected(self):
        (self.root / "model.onnx").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "stale artifact source"):
            self.verify()

    def test_schema_two_dispatch_modes_and_pve_capability(self):
        self.manifest["schema_version"] = 2
        self.manifest["abi_version"] = 2
        self.manifest["model"].update(
            dimension=2, classes=18, supports_pve=True)
        self.manifest["buffers"][2]["bytes"] = 128
        self.manifest["dispatches"][0]["mode"] = "labels"
        self.manifest["dispatches"][0]["args"][1]["bytes"] = 8
        pve = copy.deepcopy(self.manifest["dispatches"][0])
        pve["mode"] = "pve"
        pve["args"][1]["bytes"] = 128
        self.manifest["dispatches"].append(pve)
        self.assertTrue(self.verify()["model"]["supports_pve"])
        self.manifest["dispatches"].pop()
        with self.assertRaisesRegex(ValueError, "dispatch modes"):
            self.verify()

    def test_schema_one_allows_only_known_contract_source_evolution(self):
        source = self.root / "scripts" / "build_hip_artifacts.py"
        source.parent.mkdir()
        source.write_bytes(b"new packer")
        self.manifest["sources"] = {
            "scripts/build_hip_artifacts.py": sha(b"old packer")}
        self.verify()

    def test_legacy_high_class_count_remains_compatible(self):
        # Old full-head artifacts, including PR #34's 104-class MindSnap,
        # must remain readable before replacing their tail. Class count alone
        # does not identify a fused artifact or its schema contract.
        self.manifest["model"].update(dimension=2, classes=104)
        self.verify()
        self.manifest.update(schema_version=2, abi_version=2)
        self.manifest["model"]["supports_pve"] = False
        self.manifest["buffers"][2]["bytes"] = 8
        self.manifest["dispatches"][0].update(mode="labels")
        self.manifest["dispatches"][0]["args"][1]["bytes"] = 8
        self.verify()

    def test_bounded_classifier_rejects_private_scratch_or_spills(self):
        for kwargs in ({"private_bytes": 16}, {"vgpr_spills": 1}, {"sgpr_spills": 1}):
            with self.subTest(**kwargs):
                blob = code_object(entry="bc_hip_classify", **kwargs)
                with self.assertRaisesRegex(ValueError, "spill to private memory"):
                    m.code_object_metadata(blob)


if __name__ == "__main__":
    unittest.main()
