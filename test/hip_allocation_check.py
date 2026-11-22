#!/usr/bin/env python3
"""Linux allocation diagnostics: mock every buffer failure, then real HIP OOM injection."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from hip_memory import memory_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--real-hip', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.output = a.output.resolve()
    rows = []
    for path in sorted((ROOT / 'src/hip_artifacts').glob('*/*/manifest.json')):
        m = json.loads(path.read_text())
        name, arch = m['model']['name'], m['target']['architecture']
        executable = a.output / (name + '-' + arch)
        command = ['gcc', '-O2', '-std=c11', '-ffunction-sections', '-fdata-sections',
                   '-I.', '-I' + str(ROOT / 'src/models' / name),
                   '-DBC_MODEL_IS_' + name.upper() + '=1',
                   '-DBC_HIP_CATALOG_HEADER="hip_artifacts/' + name + '/' + arch + '/catalog.h"',
                   str(ROOT / 'test/hip_allocation_check.c'), '-Wl,--gc-sections', '-ldl', '-lm',
                   '-o', str(executable)]
        subprocess.run(command, cwd=ROOT / 'src', check=True)
        for mode in (('labels', 'pve') if m['model'].get('supports_pve') else ('labels',)):
            plan = memory_plan(m, mode)
            sizes = [plan[k] for k in ('arena_bytes', 'input_bytes', 'output_bytes')]
            for fail_at in range(4):
                for query in range(3):
                    result = subprocess.run([str(executable), str(fail_at), str(query), str(int(mode == 'pve'))],
                                            capture_output=True, text=True, check=True)
                    log = result.stdout + result.stderr
                    if fail_at:
                        assert f'request={sizes[fail_at-1]} bytes' in log, log
                        assert f'planned persistent={sum(sizes)} bytes' in log, log
                        assert f'already allocated={sum(sizes[:fail_at-1])} bytes' in log, log
                        assert ('driver free snapshot=1073741824 of 8589934592 bytes' if query == 1
                                else 'driver free snapshot unavailable') in log, log
                    else:
                        assert 'HIP memory:' not in log, log
                    label = f'{name}-{arch}-{mode}-failure{fail_at}-query{query}'
                    (a.output / (label + '.log')).write_text(log)
                    rows.append(dict(case=label, passed=True))
    if a.real_hip:
        audit = a.output / 'oom-audit.so'
        subprocess.run(['gcc', '-shared', '-fPIC', str(ROOT / 'test/hip_oom_audit.c'), '-o', str(audit)], check=True)
        for name in ('mindgrab', 'mindmap-fast', 'mindmap', 'mindsnap'):
            for mode in (('labels', 'pve') if name == 'mindmap' else ('labels',)):
                output = a.output / (name + '-' + mode + '.nii.gz')
                command = [str(ROOT / 'src' / ('brainchop-' + name)), '-backend', 'hip', '-o', str(output)]
                if mode == 'pve': command += ['--pve']
                command += [str(ROOT / 'test/t1_2mm.nii.gz')]
                result = subprocess.run(command, env=dict(os.environ, LD_AUDIT=str(audit)),
                                        capture_output=True, text=True, timeout=60)
                log = result.stdout + result.stderr
                (a.output / (name + '-' + mode + '-real-oom.log')).write_text(log)
                assert result.returncode and not list(output.parent.glob(output.name[:-7] + '*.nii.gz')), log
                assert 'using hip backend' in log and 'hipErrorOutOfMemory' in log, log
                assert 'planned persistent=' in log and 'driver free snapshot=' in log, log
                assert 'host MemAvailable snapshot=' in log, log
                rows.append(dict(case=name + '-' + mode + '-real-oom', passed=True))
        # A failed GPU allocation in automatic mode must still use the CPU
        # reference correctly. This process-local injection never pressures GPU memory.
        output = a.output / 'mindsnap-auto-oom.nii.gz'
        command = [str(ROOT / 'src/brainchop-mindsnap'), '-backend', 'auto',
                   '-o', str(output), str(ROOT / 'test/t1_2mm.nii.gz')]
        result = subprocess.run(command, env=dict(os.environ, LD_AUDIT=str(audit), BC_NUM_THREADS='4'),
                                capture_output=True, text=True, timeout=120)
        log = result.stdout + result.stderr
        (a.output / 'mindsnap-auto-real-oom.log').write_text(log)
        assert result.returncode == 0 and 'falling back to CPU' in log, log
        assert 'planned persistent=' in log and 'driver free snapshot=' in log, log
        assert gzip.decompress(output.read_bytes()) == gzip.decompress(
            (ROOT / 'test/labels104_native_majority.nii.gz').read_bytes())
        rows.append(dict(case='mindsnap-auto-real-oom', passed=True, cpu_reference_exact=True))
    (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'PASS: {len(rows)} allocation diagnostic cases; forced failure has no output; auto fallback exact')


if __name__ == '__main__':
    main()
