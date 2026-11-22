#!/usr/bin/env python3
"""Installed OpenCL parity and first-run calibration/cache gate (two supported models)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sysconfig
from hip_native_check import compare

p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
fixtures=Path(__file__).resolve().parent; scripts=Path(sysconfig.get_path('scripts'))
env=dict(os.environ,XDG_CACHE_HOME=str(a.output/'calibration-cache'),BC_NUM_THREADS='4')
rows=[]
for model,fixture,limit in [('mindgrab','bet.nii.gz',64),('16chan18cls','labels16.nii.gz',1024)]:
    outputs=[]
    for iteration in range(2):
        label=f'{model}-{iteration}'; output=a.output/(label+'.nii.gz')
        command=[str(scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model))),'-backend','opencl','-o',str(output)]
        if model!='mindgrab': command+=['--save-conform']
        result=subprocess.run(command+[str(fixtures/'t1_crop.nii.gz')],env=env,capture_output=True,text=True,timeout=600)
        log=result.stdout+result.stderr; (a.output/(label+'.log')).write_text(log)
        assert result.returncode==0 and 'using opencl backend' in log and 'falling back' not in log,log
        assert ('tuning OpenCL convolution' in log)==(iteration==0),log
        different=compare(fixtures/fixture,output,limit)
        outputs.append(output); rows.append(dict(model=model,iteration=iteration,different_voxels=different,limit=limit,tuned=iteration==0))
        print(json.dumps(rows[-1]),flush=True)
    compare(outputs[0],outputs[1],0)
assert (a.output/'calibration-cache/brainchop/tune.txt').is_file()
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
