"""Regression checks for moving a cached wheel between hardware contexts."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging'))
from brainchop_cli import adaptive as a


class RuntimeCache(unittest.TestCase):
    def setUp(self):
        self.current=dict(abi=a.abi(),cpu='avx2-fma',devices=[])
        self.manifest=dict(runtime_environment=copy.deepcopy(self.current),cpu_target='avx2-fma',
            build_policy=dict(target='auto',cpu='auto',settings={}),options=['CUDA=0','OPENCL=0'],artifacts={},
            artifact_policy=dict(cuda_architectures=[60,70,75,80,86,89,90],cuda_minimum_driver=12000,hip_architectures=['gfx1151']))

    def test_cpu_wheel_adapts_when_gpu_appears(self):
        for device in (dict(backend='cuda',arch=89,driver=13000),dict(backend='hip',arch='gfx1151'),dict(backend='opencl',arch='gpu'),dict(backend='metal',arch='apple')):
            with self.subTest(device=device):
                self.current['devices']=[device]
                self.assertFalse(a.compatible(self.manifest,self.current))

    def test_shared_cuda_wheel_rebuilds_for_best_available_cubin(self):
        self.manifest['artifacts']={'cuda_modules/mindgrab.fatbin':{'architectures':[80]}}
        self.current['devices']=[dict(backend='cuda',arch=86,driver=13000)]
        self.assertFalse(a.compatible(self.manifest,self.current))
        self.manifest['artifacts']['cuda_modules/mindgrab.fatbin']['architectures']=[86]
        self.assertTrue(a.compatible(self.manifest,self.current))

    def test_driver_loss_and_unsupported_gpus_do_not_cause_build_loops(self):
        for devices in ([],[dict(backend='cuda',arch=100,driver=13000)],[dict(backend='cuda',arch=89,driver=11000)],[dict(backend='hip',arch='gfx9999')]):
            self.current['devices']=devices
            self.assertTrue(a.compatible(self.manifest,self.current))

    def test_fixed_gpu_target_stays_fixed_but_unsafe_cpu_or_abi_rebuilds(self):
        self.manifest['build_policy']['target']='sm89'
        self.current['devices']=[dict(backend='cuda',arch=75,driver=13000)]
        self.assertTrue(a.compatible(self.manifest,self.current))
        self.current['cpu']='baseline'
        self.assertFalse(a.compatible(self.manifest,self.current))
        self.manifest['cpu_target']='baseline'
        self.assertTrue(a.compatible(self.manifest,self.current))
        self.current['abi']['libc']=['glibc','other']
        self.assertFalse(a.compatible(self.manifest,self.current))

    def test_signature_deduplicates_ordinals_but_preserves_driver(self):
        first=dict(cpu='baseline',devices=[dict(backend='cuda',arch=89,driver=13000,index=0)])
        second=copy.deepcopy(first);second['devices']+=[dict(backend='cuda',arch=89,driver=13000,index=7)]
        self.assertEqual(a.signature(first),a.signature(second))
        second['devices'][0]['driver']=13001
        self.assertNotEqual(a.signature(first),a.signature(second))

    def test_cache_integrity_and_read_only_hit(self):
        with tempfile.TemporaryDirectory() as temp:
            package=Path(temp)/'installed'; package.mkdir(); cache=Path(temp)/'cache';cache.mkdir()
            capsule=b'source';digest=hashlib.sha256(capsule).hexdigest()
            self.manifest['source_capsule']=dict(file='source.tar.gz',sha256=digest)
            (package/'source.tar.gz').write_bytes(capsule)
            self.current['devices']=[dict(backend='cuda',arch=89,driver=13000)]
            key=hashlib.sha256(json.dumps(dict(source=digest,hardware=self.current,policy=self.manifest['build_policy']),sort_keys=True).encode()).hexdigest()
            runtime=cache/key;(runtime/'bin').mkdir(parents=True)
            record=copy.deepcopy(self.manifest);record['runtime_environment']=copy.deepcopy(self.current)
            record['executables']={'brainchop-mindgrab':dict(sha256=hashlib.sha256(b'binary').hexdigest())}
            (runtime/'bin/brainchop-mindgrab').write_bytes(b'binary');(runtime/'build.json').write_text(json.dumps(record))
            with mock.patch.object(a,'probe',return_value=self.current),mock.patch.object(a,'cache_root',return_value=cache),mock.patch.object(a.tempfile,'TemporaryDirectory',side_effect=AssertionError('cache hit must not write')):
                self.assertEqual(a.select(package,self.manifest),(runtime,record))
            (runtime/'bin/brainchop-mindgrab').write_bytes(b'corrupted')
            self.assertIsNone(a.cached(runtime,digest,self.current,self.manifest['build_policy']))

    def test_source_rejects_traversal_and_links(self):
        for name,kind in [('root/../../bad',tarfile.REGTYPE),('/root/bad',tarfile.REGTYPE),('root/link',tarfile.SYMTYPE),('root/ok',tarfile.LNKTYPE)]:
            with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'source.tar.gz'
                with tarfile.open(path,'w:gz') as archive:
                    item=tarfile.TarInfo(name);item.type=kind;archive.addfile(item,io.BytesIO())
                with self.assertRaises(ValueError): a.unpack_source(path,Path(temp)/'output')

    def test_prebuilt_wheel_portable_abi_across_os_versions(self):
        # Windows Server 2022 -> Windows 10/11 (same 10.0.* UCRT ABI)
        win_built=dict(system='Windows',machine='x86_64',libc=[],os_version='10.0.20348')
        win_11=dict(system='Windows',machine='x86_64',libc=[],os_version='10.0.26100')
        win_10=dict(system='Windows',machine='x86_64',libc=[],os_version='10.0.19045')
        self.assertTrue(a.abi_compatible(win_built,win_11,portable=True))
        self.assertTrue(a.abi_compatible(win_built,win_10,portable=True))
        self.assertFalse(a.abi_compatible(win_built,win_11,portable=False))
        # macOS deployment target 13.2
        mac_built=dict(system='Darwin',machine='arm64',libc=[],os_version='15.4')
        self.assertTrue(a.abi_compatible(mac_built,dict(system='Darwin',machine='arm64',libc=[],os_version='14.7'),portable=True))
        self.assertTrue(a.abi_compatible(mac_built,dict(system='Darwin',machine='arm64',libc=[],os_version='26.1'),portable=True))
        self.assertFalse(a.abi_compatible(mac_built,dict(system='Darwin',machine='arm64',libc=[],os_version='12.7'),portable=True))
        # Linux glibc forward compatibility only
        lin_built=dict(system='Linux',machine='x86_64',libc=['glibc','2.39'],os_version='')
        self.assertTrue(a.abi_compatible(lin_built,dict(system='Linux',machine='x86_64',libc=['glibc','2.41'],os_version=''),portable=True))
        self.assertFalse(a.abi_compatible(lin_built,dict(system='Linux',machine='x86_64',libc=['glibc','2.35'],os_version=''),portable=True))
        self.assertFalse(a.abi_compatible(lin_built,dict(system='Linux',machine='x86_64',libc=['musl','1.2'],os_version=''),portable=True))

    def test_prebuilt_baseline_select_skips_probe_when_abi_compatible(self):
        win_built=dict(system='Windows',machine='x86_64',libc=[],os_version='10.0.20348')
        win_11=dict(system='Windows',machine='x86_64',libc=[],os_version='10.0.26100')
        manifest=copy.deepcopy(self.manifest)
        manifest['build_policy']['target']='all'
        manifest['cpu_target']='baseline'
        manifest['runtime_environment']['abi']=win_built
        manifest['source_capsule']=dict(file='source.tar.gz',sha256='0'*64)
        with mock.patch.object(a,'abi',return_value=win_11),mock.patch.object(a,'probe',side_effect=AssertionError('prebuilt baseline wheel must not spawn hardware-probe')):
            self.assertEqual(a.select(Path('/fake/pkg'),manifest),(Path('/fake/pkg'),manifest))

if __name__=='__main__': unittest.main()
