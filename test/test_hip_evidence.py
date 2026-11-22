#!/usr/bin/env python3
"""Privacy, timing integrity and determinism checks for published evidence."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_hip_evidence import build_bundle


class EvidenceTests(unittest.TestCase):
    def test_sanitization_integrity_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input'; source.mkdir()
            (source / 'run.log').write_text('/home/alice/work/results alice@example.com 0x000012345678\n')
            (source / 'trace.csv').write_text('Thread_Id,Address,Start_Timestamp,End_Timestamp\n123,0x123400000000,100,140\n123,0x123400000000,150,170\n')
            (source / 'allocation.csv').write_text('Start_Timestamp,End_Timestamp\n80,90\n')
            (source / 'patient.nii.gz').write_bytes(b'never publish this')
            (source / 'source').mkdir()
            (source / 'source/private.json').write_text('{"private":true}')
            a, b = root / 'a.tar.gz', root / 'b.tar.gz'
            digest, count = build_bundle({'run': source}, a, {'/home/alice/work': 'REPO'})
            build_bundle({'run': source}, b, {'/home/alice/work': 'REPO'})
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(digest, hashlib.sha256(a.read_bytes()).hexdigest())
            self.assertEqual(count, 3)
            with tarfile.open(a) as archive:
                self.assertNotIn('run/patient.nii.gz', archive.getnames())
                text = archive.extractfile('run/run.log').read().decode()
                self.assertNotIn('alice', text); self.assertNotIn('0x000012345678', text)
                self.assertIn('${REPO}', text)
                rows = list(csv.DictReader(io.StringIO(archive.extractfile('run/trace.csv').read().decode())))
                self.assertEqual(int(rows[0]['End_Timestamp']) - int(rows[0]['Start_Timestamp']), 40)
                self.assertEqual(rows[0]['Start_Timestamp'], '20')
                self.assertEqual(rows[0]['Address'], rows[1]['Address'])
                metadata = json.loads(archive.extractfile('MANIFEST.json').read())
                for item in metadata['files']:
                    self.assertEqual(item['sha256'], hashlib.sha256(archive.extractfile(item['path']).read()).hexdigest())

    def test_credentials_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'bad.log').write_text('Authorization: Bearer fake-test-token')
            with self.assertRaisesRegex(ValueError, 'possible credential'):
                build_bundle({'run': root}, root / 'public.tar.gz', {})
            self.assertFalse((root / 'public.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
