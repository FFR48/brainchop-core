#!/usr/bin/env python3
"""Direct make must select a runtime-compatible compiler, including on Windows."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
from brainchop_hardware import tool_environment
env=tool_environment()
suffix='.exe' if os.name=='nt' else ''
# No explicit CC: exercise the Makefile auto-selection which PEP 517 pins.
command=['make','-C',str(ROOT/'src'),'-j2','CUDA=0','HIP=0','OPENCL=0','METAL=0','ZSTD=0','brainchop-mindgrab'+suffix]
subprocess.run(command,env=env,check=True)
subprocess.run([str(ROOT/'src'/('brainchop-mindgrab'+suffix)),'--help'],env=env,check=True)
# The same probe must reject an MSVC ABI, even on a runner without VS clang.
with tempfile.TemporaryDirectory() as temp:
    result=subprocess.run(['gcc' if os.name=='nt' else 'cc','-D_WIN32=1','-U__MINGW32__','-c',str(ROOT/'src/compiler_probe.c'),'-o',str(Path(temp)/'probe.o')],env=env,capture_output=True,text=True)
    assert result.returncode and 'MinGW-targeting compiler' in result.stderr,result.stderr
print('PASS: direct make selects a POSIX-compatible toolchain; MSVC ABI is rejected')
