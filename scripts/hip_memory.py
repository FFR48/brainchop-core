#!/usr/bin/env python3
"""Promotion/runtime-test memory budgets for every HIP artifact, without ML tools.

The base budget is 256 bytes per voxel (4 GiB at 256 cubed), including all
persistent runtime allocations. PVE-capable artifacts retain shared full logits;
allow that class-volume and the larger selected output explicitly. This policy
is separate from ABI verification so historical artifacts remain upgrade inputs.
"""
import argparse
import json
from pathlib import Path


def memory_plan(manifest, mode='labels'):
    model = manifest['model']
    voxels = model['dimension'] ** 3
    pve = model.get('supports_pve', False)
    if mode not in ('labels', 'pve') or (mode == 'pve' and not pve):
        raise ValueError('unsupported HIP memory mode')
    buffers = manifest['buffers']
    if [(b['id'], b['kind']) for b in buffers] != [
            ('arena', 'arena'), ('input', 'input'), ('output', 'output')]:
        raise ValueError('HIP runtime permits only arena, input and output allocations')
    expected_output = 16 * voxels if pve else voxels
    if buffers[1]['bytes'] != 2 * voxels or buffers[2]['bytes'] != expected_output:
        raise ValueError('HIP input/output memory does not match model geometry')
    output = 16 * voxels if mode == 'pve' else voxels
    base = 256 * voxels
    shared_logits = 4 * model['classes'] * voxels if pve else 0
    extra_output = output - voxels
    return dict(model=model['name'], architecture=manifest['target']['architecture'],
                mode=mode, arena_bytes=buffers[0]['bytes'], input_bytes=2 * voxels,
                output_bytes=output,
                persistent_bytes=buffers[0]['bytes'] + 2 * voxels + output,
                budget_bytes=base + shared_logits + extra_output,
                base_budget_bytes=base, shared_pve_logits_allowance_bytes=shared_logits,
                pve_output_allowance_bytes=extra_output)


def check_memory(manifest, mode='labels', observed_bytes=None):
    plan = memory_plan(manifest, mode)
    if plan['persistent_bytes'] > plan['budget_bytes']:
        raise ValueError(f"{plan['model']} {mode}: HIP memory plan {plan['persistent_bytes']} "
                         f"exceeds budget {plan['budget_bytes']} bytes")
    if observed_bytes is not None and observed_bytes != plan['persistent_bytes']:
        raise ValueError(f"{plan['model']} {mode}: actual HIP allocations {observed_bytes} "
                         f"do not match declared plan {plan['persistent_bytes']} bytes")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='src/hip_artifacts directory')
    args = parser.parse_args()
    paths = sorted(args.root.glob('*/*/manifest.json'))
    if not paths:
        parser.error('no HIP artifact manifests found')
    for path in paths:
        from hip_artifacts import verify
        manifest = verify(path.parent)
        modes = ('labels', 'pve') if manifest['model'].get('supports_pve') else ('labels',)
        for mode in modes:
            print(json.dumps(check_memory(manifest, mode)), flush=True)


if __name__ == '__main__':
    main()
