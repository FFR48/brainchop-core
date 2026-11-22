#!/usr/bin/env python3
"""Actual installed CPU wheel: missing compiler, read-only cache and corruption."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import brainchop_cli
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(exist_ok=False)
package=Path(brainchop_cli.__file__).parent;manifest=json.loads((package/'build.json').read_text());assert manifest['hardware']['devices']==[]
stubs=a.output/'forbidden';stubs.mkdir()
for name in ('cc','gcc','clang','make','nvcc','hipcc'):
    tool=stubs/name;tool.write_text('#!/bin/sh\nexit 97\n');tool.chmod(0o755)
env=dict(os.environ,PATH=str(stubs)+os.pathsep+os.environ['PATH'])
command=[str(Path(sysconfig.get_path('scripts'))/'brainchop-mindgrab'),'-a','-1']
rows=[]
def run(label,cache,expected):
    result=subprocess.run(command,env=dict(env,XDG_CACHE_HOME=str(cache)),capture_output=True,text=True,timeout=120)
    log=result.stdout+result.stderr;(a.output/(label+'.log')).write_text(log)
    assert (result.returncode==0)==expected,log
    assert 'Traceback' not in log,log
    if expected: assert 'building a native runtime' not in log,log
    else: assert 'native rebuild failed' in log and 'using cpu backend' not in log,log
    rows.append(dict(case=label,exit_code=result.returncode,expected_success=expected))
run('missing-compiler',a.output/'empty-cache',False)
cache=a.output/'read-only-cache';shutil.copytree(a.cache.resolve(),cache)
for path in sorted(cache.rglob('*'),reverse=True):
    path.chmod(0o555 if path.is_dir() or path.parent.name=='bin' else 0o444)
cache.chmod(0o555)
try: run('read-only-hit',cache,True)
finally:
    cache.chmod(0o755)
    for path in cache.rglob('*'): path.chmod(0o755 if path.is_dir() or path.parent.name=='bin' else 0o644)
corrupt=a.output/'corrupt-cache';shutil.copytree(a.cache.resolve(),corrupt)
binary=next(corrupt.glob('brainchop/runtimes/*/bin/brainchop-mindgrab'));binary.write_bytes(b'corrupt')
run('corrupt-binary',corrupt,False)
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS: missing compiler errors, read-only cache reuse, corrupt cache rebuild refusal without fallback')
