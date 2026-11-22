#!/usr/bin/env python3
"""Prove a clean HIP C build cannot invoke compilers or Python/ML tooling.

Copy sources into an isolated build tree, put failing tool stubs first on PATH,
install a Python import guard, build with gcc, and inspect runtime dependencies.
This gate intentionally excludes maintainer artifact regeneration/check targets.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.output = a.output.resolve()
    source, stubs = a.output / 'source', a.output / 'stubs'
    stubs.mkdir()
    calls = a.output / 'forbidden-tools.log'
    for name in ('python', 'python3', 'hipcc', 'nvcc', 'migraphx-driver',
                 'tinygrad', 'torchrun', 'pip', 'pip3'):
        path = stubs / name
        path.write_text('#!/bin/sh\necho "$0 $*" >> "' + str(calls) + '"\nexit 97\n')
        path.chmod(0o755)
    (stubs / 'sitecustomize.py').write_text('''import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tinygrad', 'torch', 'migraphx', 'onnx'):
            raise RuntimeError('forbidden ML import: ' + fullname)
sys.meta_path.insert(0, Guard())
''')
    def ignore(directory, names):
        return [n for n in names if n in ('.git', 'build', 'dist', '__pycache__', '.build-config')
                or n.endswith(('.o', '.exe')) or (n.startswith('brainchop-') and Path(directory).name == 'src')]
    shutil.copytree(ROOT, source, ignore=ignore)
    env = dict(PATH=str(stubs) + ':/usr/bin:/bin', HOME=str(a.output),
               PYTHONPATH=str(stubs), TMPDIR=str(a.output), LC_ALL='C')
    command = ['make', '-C', str(source / 'src'), '-j4', 'CC=gcc', 'CUDA=driver',
               'HIP=1', 'OPENCL=0', 'ZSTD=0', 'models']
    with (a.output / 'build.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # PR #34's incremental-build regression: changing artifact availability
    # must change .build-config and rebuild backend.c without manual cleaning.
    artifact = source / 'src/hip_artifacts/mindsnap/gfx1151'
    hidden = artifact.with_name('temporarily-hidden')
    artifact.rename(hidden)
    build_one = command[:-1] + ['brainchop-mindsnap']
    with (a.output / 'stamp-remove.log').open('w') as log:
        subprocess.run(build_one, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = source / 'src/brainchop-mindsnap'
    listing = subprocess.check_output([str(binary), '-a', '-1'], env=env, stderr=subprocess.STDOUT, text=True)
    assert 'hip:' not in listing, listing
    hidden.rename(artifact)
    with (a.output / 'stamp-restore.log').open('w') as log:
        subprocess.run(build_one, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    listing = subprocess.check_output([str(binary), '-a', '-1'], env=env, stderr=subprocess.STDOUT, text=True)
    assert 'hip:' in listing, listing
    assert not calls.exists(), calls.read_text() if calls.exists() else ''
    rows = []
    for name in ('mindgrab', 'mindmap-fast', 'mindmap', 'mindsnap'):
        binary = source / 'src' / ('brainchop-' + name)
        links = subprocess.check_output(['ldd', str(binary)], env=env, text=True)
        assert not any(word in links.lower() for word in ('python', 'torch', 'tinygrad', 'migraphx', 'amdhip', 'cuda'))
        subprocess.run([str(binary), '--help'], env=env, capture_output=True, check=True)
        rows.append(dict(model=name, dynamic_dependencies=links))
    (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('PASS: clean gcc HIP/CUDA-driver build; all Python/ML/GPU compiler stubs uncalled; C-only linkage')


if __name__ == '__main__':
    main()
