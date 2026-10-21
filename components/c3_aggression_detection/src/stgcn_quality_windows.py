"""Version existing windows with the same current-observation guard as runtime.

Old archives and labels remain unchanged. All roles use the identical pose-only
rule. This is a separate matched-cohort experiment, not a new test split.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.advanced_experiment import load_windows
from src.foundation_io import sha256, write_csv, write_json
from src.pose_dataset import sequence_window
from src.pose_sequences import load_cache


def guard_subset(data, current_observed_joints, minimum=6):
    counts = np.asarray(current_observed_joints)
    n = len(data['role'])
    if counts.shape != (n,) or not np.isfinite(counts).all() or (counts < 0).any() or minimum < 1:
        raise ValueError('One nonnegative current-observation count is required per window')
    keep = counts >= minimum
    result = {k: value[keep].copy() for k, value in data.items()}
    result['original_window_index'] = np.flatnonzero(keep)
    result['current_observed_joints'] = counts[keep].astype(np.int16)
    return result, keep


def build(args):
    folder = Path(args.data)
    data, meta = load_windows(folder)
    entries = {}
    for split in ('train','val'):
        path = folder.parent/(split+'_index.json')
        if sha256(path) != meta['index_sha256'][split]:
            raise ValueError('Parent cache index changed')
        entries.update(json.loads(path.read_text())['entries'])
    current = np.zeros(len(data['role']), dtype=np.int16)
    for key in sorted(set(data['video_id'].tolist())):
        entry = entries[key]
        if sha256(entry['path']) != entry['cache_sha256']:
            raise ValueError('Parent pose cache changed')
        arrays, provenance = load_cache(entry['path'])
        if provenance['video_id'] != key or provenance['source_sha256'] != entry['source_sha256']:
            raise ValueError('Pose cache identity changed')
        for index in np.flatnonzero(data['video_id'] == key):
            i = int(np.searchsorted(arrays['t'], data['cutoff_s'][index]))
            if i >= len(arrays['t']) or abs(arrays['t'][i] - data['cutoff_s'][index]) > 1e-7:
                raise ValueError('Window cutoff missing from original cache')
            sample = sequence_window(arrays, i)
            if not np.array_equal(sample['sk'], data['sk'][index]):
                raise ValueError('Original feature vector no longer reproducible')
            current[index] = int(sample['observed_mask'][-1].sum(0).max())
    values, keep = guard_subset(data, current)
    summary = {}
    for role in ('fit','dev','val'):
        ix = values['role'] == role
        summary[role] = dict(windows=int(ix.sum()), videos=len(set(values['video_id'][ix].tolist())),
            supervised=values['target_mask'][ix].sum(0).tolist(),
            positives=(values['targets'][ix] * values['target_mask'][ix]).sum(0).astype(int).tolist())
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'windows.npz').open('xb') as stream:
        np.savez_compressed(stream, **values)
    new_meta = dict(meta, summary=summary, windows_sha256=sha256(output/'windows.npz'),
        parent_windows_sha256=meta['windows_sha256'], parent_data_directory=str(folder.resolve()),
        current_observation_policy=dict(min_current_observed_joints=6, source='raw causal pose cache at cutoff',
            identical_across_roles=True, label_selection=False, masked_or_imputed_points_count_as_observed=False),
        windows_removed=int((~keep).sum()), test_sources_used=0)
    write_json(output/'metadata.json', new_meta)
    write_csv(output/'selection.csv', [dict(original_window_index=i,video_id=str(data['video_id'][i]),
        cutoff_s=float(data['cutoff_s'][i]),role=str(data['role'][i]),
        current_observed_joints=int(current[i]),retained=bool(keep[i])) for i in range(len(keep))])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/pose_cache/advanced_v1/windows')
    parser.add_argument('--out-dir', default='data/pose_cache/diagnosis_guarded_v1/windows')
    build(parser.parse_args())
