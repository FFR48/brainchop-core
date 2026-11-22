#!/usr/bin/env python3
"""Real GPU-hidden fixed sm89 build, SDK-free OpenCL compilation, baseline CPU gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pip-wheel',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();out=root/'fixed-sm89';out.mkdir(exist_ok=False)
hidden=out/'hidden';hidden.mkdir();source=hidden/'hidden.c';source.write_text('int hidden_driver(void){return 0;}\n')
for name in ('libcuda.so.1','libOpenCL.so.1','libamdhip64.so'):
    subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(hidden/name)],check=True)
env=dict(os.environ,LD_LIBRARY_PATH=str(hidden),PYTHONPATH=str(root/'packaging'))
script="import brainchop_build as b;print(b.build_wheel('fixed-sm89',{'brainchop.target':'sm89'}))"
with (out/'build.log').open('w') as log: subprocess.run([sys.executable,'-c',script],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,'-m','venv','--without-pip',str(out/'venv')],check=True)
python=out/'venv/bin/python';wheel=next(out.glob('brainchop*.whl'))
bootstrap="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
subprocess.run([str(python),'-c',bootstrap,str(a.pip_wheel.resolve()),'install','--no-index',str(wheel)],check=True)
code="import brainchop_cli;from pathlib import Path;print(Path(brainchop_cli.__file__).parent)"
package=Path(subprocess.check_output([str(python),'-c',code],text=True).strip());record=json.loads((package/'build.json').read_text())
assert record['hardware']['devices']==[] and record['cpu_target']=='baseline' and record['build_policy']['target']=='sm89',record
assert record['artifacts']['cuda_modules/mindgrab.fatbin']['architectures']==[89],record
subprocess.run([str(python),str(root/'test/pypi_installed.py'),'--cpu-fallback-only'],check=True,env=dict(os.environ,BC_NUM_THREADS='4'))
subprocess.run([str(python),str(root/'test/pypi_audit.py'),'--output',str(out/'audit.json')],check=True)
# On the priming RTX sm75 host the requested fixed sm89 selection must remain
# fixed; no automatic retargeting is permitted even though a compiler is here.
code="import brainchop_cli,json;from pathlib import Path;from brainchop_cli.adaptive import select;p=Path(brainchop_cli.__file__).parent;assert select(p,json.loads((p/'build.json').read_text()))[0]==p"
subprocess.run([str(python),'-c',code],check=True)
(out/'results.json').write_text(json.dumps(dict(gpu_hidden_build=True,target='sm89',cpu='baseline',fixed_selection_retained=True,baseline_all_models_passed=True),indent=2)+'\n')
print('PASS: GPU-hidden fixed sm89 build, baseline CPU parity, fixed runtime selection')
