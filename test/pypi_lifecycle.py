#!/usr/bin/env python3
"""Actual sdist installs via a localhost /simple index, including pip cache reuse.

Index and cache paths are isolated test infrastructure, not user requirements.
The legacy upgrade uses --no-deps to avoid installing the retired ML toolchain.
"""
import argparse
import functools
import gzip
import shlex
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdist',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pip-wheel',type=Path,help='Test infrastructure: bootstrap pip offline on nodes without ensurepip')
    parser.add_argument('--upgrade',action='store_true')
    parser.add_argument('--verbose',action='store_true',help='retain compiler commands in pip build logs')
    parser.add_argument('--hide-gpu-for-cold',action='store_true',help='Linux-only: driver stubs hide GPUs only while building the first wheel')
    parser.add_argument('--forbid-build-tools',action='store_true',help='Fail if the build invokes GPU compilers, ML tools or bare Python commands')
    args=parser.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    index=out/'simple/brainchop'; index.mkdir(parents=True)
    shutil.copy2(args.sdist,index/args.sdist.name)
    (index/'index.html').write_text(f'<a href="{args.sdist.name}">{args.sdist.name}</a>')
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(out)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    index_url=f'http://127.0.0.1:{server.server_port}/simple'
    env=dict(os.environ,PIP_CACHE_DIR=str(out/'cache'),PIP_DISABLE_PIP_VERSION_CHECK='1',XDG_CACHE_HOME=str(out/'runtime-cache'))
    results=[]
    if args.forbid_build_tools:
        stubs=out/'forbidden-tools'; stubs.mkdir()
        for name in ('python','python3','hipcc','nvcc','tinygrad','torchrun','migraphx-driver'):
            path=stubs/name
            path.write_text('#!/bin/sh\necho '+shlex.quote(name)+' >> '+shlex.quote(str(out/'forbidden.log'))+'\nexit 97\n')
            path.chmod(0o755)
        env['PATH']=str(stubs)+os.pathsep+env['PATH']
    original_library_path=env.get('LD_LIBRARY_PATH')
    if args.hide_gpu_for_cold:
        assert sys.platform.startswith('linux')
        shim=out/'hidden-drivers'; shim.mkdir()
        source=shim/'hidden.c'; source.write_text('int hidden_driver(void){return 0;}\n')
        for name in ('libamdhip64.so','libOpenCL.so.1','libcuda.so.1'):
            subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(shim/name)],check=True)
    def run(label, command, **kwargs):
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT,check=True,**kwargs)
    try:
        for case in ('cold','warm'):
            if args.hide_gpu_for_cold:
                if case=='cold': env['LD_LIBRARY_PATH']=str(shim)
                elif original_library_path is None: env.pop('LD_LIBRARY_PATH',None)
                else: env['LD_LIBRARY_PATH']=original_library_path
            venv=out/case
            run(case+'-venv',[sys.executable,'-m','venv',*(['--without-pip'] if args.pip_wheel else []),venv])
            python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            if args.pip_wheel:
                wheel=args.pip_wheel.resolve()
                code="import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
                run(case+'-bootstrap',[python,'-c',code,wheel,'install','--no-index',wheel])
            if case=='cold' and args.upgrade:
                run('legacy-install',[python,'-m','pip','install','--no-deps','brainchop==0.2.5'])
            run(case+'-install',[python,'-m','pip','install',*(['-v'] if args.verbose else []),'--index-url',index_url,'--upgrade','brainchop'])
            code='import brainchop_cli, pathlib; print(pathlib.Path(brainchop_cli.__file__).parent)'
            package=Path(subprocess.check_output([str(python),'-c',code],text=True).strip())
            manifest=json.loads((package/'build.json').read_text())
            run(case+'-help',[python,'-m','brainchop_cli','--help'])
            run(case+'-list',[python,'-m','brainchop_cli','--list'])
            run(case+'-check',[python,'-m','pip','check'])
            log=(out/(case+'-install.log')).read_text()
            if case=='warm': assert 'Using cached brainchop-' in log and 'Building wheel for brainchop' not in log, log
            assert not list(package.rglob('*.fatbin')) and not list(package.rglob('kernels.bin'))
            assert not any((package/name).exists() for name in ('src','models','torch','tinygrad'))
            results.append(dict(case=case,package=str(package),manifest=manifest,cached='Using cached brainchop-' in log))
        if args.hide_gpu_for_cold:
            assert all(row['manifest']['hardware']['devices']==[] for row in results),results
            # Pip reused the CPU wheel; first use must adapt to visible hardware.
            import re
            for iteration in range(2):
                output=out/f'warm-gpu-{iteration}.nii.gz'
                run(f'warm-visible-gpu-{iteration}',[python,'-m','brainchop_cli',Path(__file__).parent/'t1_crop.nii.gz','--skull-strip','-o',output])
                log=(out/f'warm-visible-gpu-{iteration}.log').read_text()
                assert re.search(r'using (cuda|hip|opencl) backend',log) and 'falling back' not in log,log
                assert ('building a native runtime' in log)==(iteration==0),log
            assert gzip.decompress((out/'warm-gpu-0.nii.gz').read_bytes())==gzip.decompress((out/'warm-gpu-1.nii.gz').read_bytes())
        assert not (out/'forbidden.log').exists(), 'a forbidden build tool ran'
        # Uninstall warm instance and verify pip owns every installed binary.
        run('uninstall',[python,'-m','pip','uninstall','-y','brainchop'])
        assert not (package/'bin').exists()
        assert not (venv/('Scripts' if os.name=='nt' else 'bin')/('brainchop.exe' if os.name=='nt' else 'brainchop')).exists()
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results,indent=2))
    finally: server.shutdown()
if __name__=='__main__': main()
