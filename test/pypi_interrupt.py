#!/usr/bin/env python3
"""Installed POSIX launchers forward interruption without Python tracebacks."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sysconfig
import tempfile
import time
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert os.name=='posix'
scripts=Path(sysconfig.get_path('scripts'));fixture=Path(__file__).parent/'t1_crop.nii.gz';rows=[]
with tempfile.TemporaryDirectory(prefix='brainchop-interrupt-') as temp:
    for name,flags in [('brainchop-mindgrab',['-cpu','1']),('brainchop',['--skull-strip','--threads','1'])]:
        log=Path(temp)/(name+'.log')
        with log.open('w') as output:
            process=subprocess.Popen([str(scripts/name),str(fixture),*flags,'-o',str(Path(temp)/(name+'.nii.gz'))],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+60
                while 'using cpu backend' not in log.read_text():
                    assert process.poll() is None,log.read_text()
                    assert time.monotonic()<deadline,'native runtime did not start'
                    time.sleep(.05)
                os.killpg(process.pid,signal.SIGINT);code=process.wait(timeout=10)
            finally:
                if process.poll() is None: os.killpg(process.pid,signal.SIGKILL);process.wait()
        text=log.read_text();assert 'Traceback' not in text,text
        assert code==(-signal.SIGINT if name!='brainchop' else 130),(name,code,text)
        rows.append(dict(command=name,exit_code=code,no_traceback=True))
a.output.write_text(json.dumps(rows,indent=2)+'\n');print('PASS: direct exec and compatibility launcher interrupt cleanly')
