#!/usr/bin/env python3
"""Real HIP first-use rebuild, concurrent cache selection and compiler-free reuse.

Run with the CPU-hidden lifecycle's cold environment after drivers are exposed.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import brainchop_cli
from hip_native_check import compare
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
package=Path(brainchop_cli.__file__).parent;original=json.loads((package/'build.json').read_text());assert original['hardware']['devices']==[]
stubs=out/'forbidden';stubs.mkdir();calls=out/'forbidden.log'
def forbid(names):
    for name in names:
        tool=stubs/name;tool.write_text('#!/bin/sh\necho '+name+' >> "'+str(calls)+'"\nexit 97\n');tool.chmod(0o755)
forbid(('hipcc','nvcc','tinygrad','torchrun','migraphx-driver','python','python3'))
env=dict(os.environ,PATH=str(stubs)+os.pathsep+os.environ['PATH'],XDG_CACHE_HOME=str(out/'cache'),BC_TIME_GPU='1',BC_HIP_MEMORY='1',BC_NUM_THREADS='4')
code="import brainchop_cli,json;from pathlib import Path;from brainchop_cli.adaptive import select;p=Path(brainchop_cli.__file__).parent;print(select(p,json.loads((p/'build.json').read_text()))[0])"
processes=[subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
paths=[];builds=0
for i,process in enumerate(processes):
    stdout,stderr=process.communicate(timeout=660);(out/f'select-{i}.log').write_text(stdout+stderr)
    assert process.returncode==0,stderr
    paths.append(Path(stdout.strip()));builds+=stderr.count('building a native runtime')
assert paths[0]==paths[1] and builds==1,(paths,builds)
runtime=paths[0];record=json.loads((runtime/'build.json').read_text())
assert 'HIP=1' in record['options'] and 'CUDA=0' in record['options'],record
assert len([k for k in record['artifacts'] if k.startswith('hip_artifacts/')])==4
assert {v['architecture'] for k,v in record['artifacts'].items() if k.startswith('hip_artifacts/')}=={'gfx1151'}
assert not calls.exists()
# Selection must persist across independent Python processes and public launches.
forbid(('cc','gcc','clang','make'))
again=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,check=True)
assert Path(again.stdout.strip())==runtime and 'building a native runtime' not in again.stderr
fixtures=Path(__file__).resolve().parent;scripts=Path(sysconfig.get_path('scripts'));rows=[]
for mode in ('hip','auto'):
    output=out/(mode+'.nii.gz')
    result=subprocess.run([str(scripts/'brainchop-mindgrab'),'-backend',mode,'-o',str(output),str(fixtures/'t1_crop.nii.gz')],env=env,capture_output=True,text=True,timeout=600)
    log=result.stdout+result.stderr;(out/(mode+'.log')).write_text(log)
    dispatch=re.search(r'HIP (\d+) dispatches in ([0-9.]+) s',log)
    assert result.returncode==0 and 'using hip backend' in log and 'falling back' not in log and dispatch and int(dispatch[1])>0,log
    assert 'building a native runtime' not in log,log
    rows.append(dict(mode=mode,dispatches=int(dispatch[1]),gpu_seconds=float(dispatch[2]),different_voxels=compare(fixtures/'bet.nii.gz',output,64)))
assert gzip.decompress((out/'hip.nii.gz').read_bytes())==gzip.decompress((out/'auto.nii.gz').read_bytes())
assert not calls.exists(),calls.read_text() if calls.exists() else ''
subprocess.run([sys.executable,str(fixtures/'pypi_audit.py'),'--runtime',str(runtime),'--output',str(out/'audit.json')],env=env,check=True)
(out/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
(out/'runtime-path.txt').write_text(str(runtime)+'\n')
(out/'results.json').write_text(json.dumps(dict(concurrent_builds=builds,selected_architecture='gfx1151',warm_without_compilers=True,no_forbidden_tools=True,auto_equal=True,runs=rows),indent=2)+'\n')
print('PASS: CPU wheel adapted once to gfx1151, warm compiler-free HIP dispatch and artifact pruning')
