#!/usr/bin/env python3
"""Generate tissue references with the CPU executable; preserve all gates."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=True)
rows=[]
for model in ('mindmap',):
    for fixture in ('t1_crop','t1_2mm'):
        for conform in (True,False):
            key=f'{model}-{fixture}-'+('conformed' if conform else 'native')
            args=[str(root/'src'/f'brainchop-{model}'),'-cpu','4','--pve','-o',str(out/(key+'.nii.gz'))]
            if conform: args+=['--save-conform']
            args+=[str(root/'test'/(fixture+'.nii.gz'))]
            p=subprocess.run(args,capture_output=True,text=True)
            (out/(key+'.log')).write_text(p.stdout+p.stderr)
            if p.returncode:raise RuntimeError(f'{key} failed: {p.stderr}')
            files={t:f'{key}_{t}.nii.gz' for t in ('gm','wm','csf','brain')}
            rows.append(dict(key=key,model=model,fixture=fixture,conform=conform,files=files,
                sha256={t:hashlib.sha256((out/f).read_bytes()).hexdigest() for t,f in files.items()}))
            print(key,flush=True)
            (out/'references.json').write_text(json.dumps(rows,indent=2)+'\n')
