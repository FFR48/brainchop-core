#!/usr/bin/env python3
"""Exercise shared NFS pip-wheel and hardware-runtime caches on real GPUs.

Prime the pip cache on sm75, then consume it on other allocated GPU nodes.
The package install command needs no product-specific pip flags.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from hip_native_check import compare
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--prime',action='store_true');p.add_argument('--pip-wheel',type=Path);a=p.parse_args()
root=a.root.resolve();out=root/('shared-'+a.label);out.mkdir(exist_ok=False)
env=dict(os.environ,PIP_CACHE_DIR=str(root/'shared-pip-cache'),PIP_DISABLE_PIP_VERSION_CHECK='1',BC_NUM_THREADS='4',XDG_CACHE_HOME=str(root/'shared-runtime-cache'),BC_TIME_GPU='1',BRAINCHOPC_CUDA_PROFILE='1')
subprocess.run([sys.executable,'-m','venv',*(['--without-pip'] if a.pip_wheel else []),str(out/'venv')],check=True)
python=out/'venv/bin/python'
if a.pip_wheel:
    code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
    subprocess.run([str(python),'-c',code,str(a.pip_wheel.resolve()),'install','--no-index',str(a.pip_wheel.resolve())],check=True)
version=(Path(__file__).resolve().parents[1]/'packaging/PYPI_VERSION').read_text().strip()
cmd=[str(python),'-m','pip','install','--no-index','--find-links',str(root/f'brainchop-{version}.tar.gz'),'brainchop']
proc=subprocess.run(cmd,env=env,capture_output=True,text=True)
log=proc.stdout+proc.stderr;(out/'install.log').write_text(log);assert proc.returncode==0,log
if a.prime: assert 'Building wheel for brainchop' in log or 'Using cached brainchop-' in log,log
else: assert 'Using cached brainchop-' in log and 'Building wheel for brainchop' not in log,log
package=Path(subprocess.check_output([str(python),'-c','import brainchop_cli,pathlib;print(pathlib.Path(brainchop_cli.__file__).parent)'],text=True).strip())
manifest=json.loads((package/'build.json').read_text());assert manifest['artifacts']['cuda_modules/mindgrab.fatbin']['architectures']==[75],manifest
# Concurrent first callers must publish one complete, pruned runtime. Checking
# selection separately keeps native launch calibration from racing this test.
code="import json;from pathlib import Path;import brainchop_cli;from brainchop_cli.adaptive import select;p=Path(brainchop_cli.__file__).parent;print(select(p,json.loads((p/'build.json').read_text()))[0])"
processes=[subprocess.Popen([str(python),'-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
selected=[];builds=0
for i,process in enumerate(processes):
    stdout,stderr=process.communicate(timeout=660);(out/f'select-{i}.log').write_text(stdout+stderr)
    assert process.returncode==0,stderr
    selected.append(Path(stdout.strip()));builds+=stderr.count('building a native runtime')
assert selected[0]==selected[1]
assert builds==(0 if a.prime else 1),builds
runtime=selected[0];record=json.loads((runtime/'build.json').read_text())
actual=record['artifacts']['cuda_modules/mindgrab.fatbin']['architectures']
if not a.prime: assert actual!=[75],actual
# Once built, no compiler or make may be needed by later users of this cache.
stubs=out/'forbidden';stubs.mkdir()
for name in ('cc','gcc','clang','make','nvcc','hipcc','tinygrad','torchrun'):
    path=stubs/name;path.write_text('#!/bin/sh\nexit 97\n');path.chmod(0o755)
env['PATH']=str(stubs)+os.pathsep+env['PATH']
source=Path(__file__).resolve().parent/'t1_crop.nii.gz'
for mode in ('forced','auto'):
    output=out/(mode+'.nii.gz');command=[str(out/'venv/bin/brainchop-mindgrab'),'-o',str(output),str(source)]
    if mode=='forced': command+=['-backend','cuda']
    proc=subprocess.run(command,env=env,capture_output=True,text=True,timeout=600)
    log=proc.stdout+proc.stderr;(out/(mode+'.log')).write_text(log)
    assert proc.returncode==0 and 'using cuda backend' in log and 'CUDA stages (' in log and 'falling back' not in log and 'building a native runtime' not in log,log
    different=compare(source.parent/'bet.nii.gz',output,64)
assert gzip.decompress((out/'forced.nii.gz').read_bytes())==gzip.decompress((out/'auto.nii.gz').read_bytes())
subprocess.run([str(python),str(source.parent/'pypi_audit.py'),'--runtime',str(runtime),'--output',str(out/'audit.json')],check=True,env=env)
(out/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
(out/'results.json').write_text(json.dumps(dict(prime=a.prime,pip_cached='Using cached brainchop-' in (out/'install.log').read_text(),installed_architectures=[75],runtime_architectures=actual,concurrent_builds=builds,runtime=str(runtime),gpu_different_voxels=different,auto_equal=True,warm_without_compiler=True),indent=2)+'\n')
print('PASS: cached pip wheel, concurrent hardware adaptation, warm CUDA without compilers, artifact pruning')
