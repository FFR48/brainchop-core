#!/usr/bin/env python3
"""Maintainer-only real-GPU smoke tests for the SAME offline fused label head.

No ML imports. Compiles small shapes with hipcc; production users consume the
checked-in code object and never run this developer test.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hipcc', default='/opt/rocm/bin/hipcc')
    p.add_argument('--gfx', default='gfx1151')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    hip = C.CDLL('libamdhip64.so')
    def bind(name, args):
        f = getattr(hip, name)
        f.argtypes, f.restype = args, C.c_int
        def call(*values):
            status = f(*values)
            if status:
                raise RuntimeError(f'{name} failed: {status}')
        return call
    pointer = C.c_void_p
    init = bind('hipInit', [C.c_uint]); init(0)
    device = bind('hipSetDevice', [C.c_int]); device(0)
    allocate = bind('hipMalloc', [C.POINTER(pointer), C.c_size_t])
    free = bind('hipFree', [pointer])
    upload = bind('hipMemcpyHtoD', [pointer, pointer, C.c_size_t])
    download = bind('hipMemcpyDtoH', [pointer, pointer, C.c_size_t])
    load = bind('hipModuleLoadData', [C.POINTER(pointer), pointer])
    unload = bind('hipModuleUnload', [pointer])
    function = bind('hipModuleGetFunction', [C.POINTER(pointer), pointer, C.c_char_p])
    launch = bind('hipModuleLaunchKernel', [pointer] + [C.c_uint] * 7 +
                  [pointer, C.POINTER(pointer), pointer])
    synchronize = bind('hipDeviceSynchronize', [])
    rows = []
    n = 7 ** 3  # exercises the partial final launch block
    for channels, classes in ((15, 2), (16, 18), (24, 104)):
        for bias in (False, True):
            directory = a.output / f'c{channels}-k{classes}-bias{int(bias)}'
            command = [sys.executable, str(ROOT / 'scripts/compile_hip_support.py'),
                '--hipcc', a.hipcc, '--source', str(ROOT / 'scripts/hip_classify.hip'),
                '--output', str(directory), '--gfx', a.gfx, '--dimension', '7',
                '--include', str(ROOT / 'src'), '--channels', str(channels), '--classes', str(classes)]
            if bias:
                command.append('--classifier-bias')
            subprocess.run(command, check=True)
            blob = C.create_string_buffer((directory / 'classify.hsaco').read_bytes())
            module, kernel = pointer(), pointer()
            load(C.byref(module), blob)
            try:
                function(C.byref(kernel), module, b'bc_hip_classify')
                for case in ('negative', 'tie', 'last-class', 'bias', 'fp32-accumulation', 'layout'):
                    rng = random.Random(104)
                    # Extra physical planes simulate padding; logical channels
                    # determine weight rows. Sentinel pad must never be read.
                    physical = 16 if channels == 15 else channels
                    x = [1.] * (channels * n) + [60000.] * ((physical - channels) * n)
                    w = [-3.] * (channels * classes)
                    b = [0.] * classes
                    if case == 'negative':
                        w[channels:2 * channels] = [-1.] * channels
                    elif case == 'tie':
                        w[:channels] = [1.] * channels
                        w[-channels:] = [1.] * channels
                    elif case == 'last-class':
                        w[-channels:] = [1.] * channels
                    elif case == 'bias':
                        w = [0.] * len(w)
                        b = [-2.] * classes
                        b[-1] = 2.
                    elif case == 'fp32-accumulation':
                        w = [-1.] * len(w)
                        w[:2 * channels] = [0.] * (2 * channels)
                        w[0:3] = [2048., 1., -2048.]
                        w[channels] = .5
                    else:
                        x[:channels * n] = [rng.randint(-8, 8) / 8 for _ in range(channels * n)]
                        w = [rng.randint(-8, 8) / 8 for _ in w]
                        b = [rng.randint(-8, 8) / 8 for _ in b]
                    expected = []
                    for i in range(n):
                        scores = [(b[k] if bias else 0.) + sum(x[c * n + i] * w[k * channels + c]
                                   for c in range(channels)) for k in range(classes)]
                        expected.append(max(range(classes), key=scores.__getitem__))
                    host = [struct.pack('<' + 'e' * len(v), *v) for v in ([x, w, b] if bias else [x, w])]
                    allocations = []
                    try:
                        for data in host + [bytes([255]) * n]:
                            ptr = pointer(); allocate(C.byref(ptr), len(data)); allocations.append(ptr)
                            upload(ptr, C.create_string_buffer(data), len(data))
                        args = (pointer * len(allocations))(*[C.cast(C.byref(v), pointer) for v in allocations])
                        launch(kernel, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, args, None)
                        synchronize()
                        result = C.create_string_buffer(n)
                        download(result, allocations[-1], n)
                        assert list(result.raw) == expected, (channels, classes, bias, case)
                        rows.append(dict(channels=channels, physical_channels=physical, classes=classes,
                                         bias=bias, case=case, voxels=n, differing=0))
                    finally:
                        for ptr in allocations:
                            free(ptr)
            finally:
                unload(module)
    (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'PASS: {len(rows)} real HIP classifier cases; no CPU fallback')


if __name__ == '__main__':
    main()
