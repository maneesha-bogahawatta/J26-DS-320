"""Verify C3 preservation, locked assignments and reused cache identities."""
import argparse
import json
from pathlib import Path
import subprocess

from src.foundation_io import sha256, write_json
from src.lock_splits import load_locked, validate_locked
from src.pose_sequences import load_cache


ALLOWED_EXISTING_EDITS = {'README.md', 'RUN_C3_DEMO.ps1', 'src/predict_advanced.py',
                          'src/evaluate_advanced_runtime.py'}


def verify(args):
    root = Path(__file__).resolve().parents[1]
    baseline = Path(args.baseline)
    protected = json.loads((baseline/'preservation_before.json').read_text())
    changed = []; missing = []
    for relative, digest in protected.items():
        path = root/relative
        if not path.is_file(): missing.append(relative)
        elif sha256(path) != digest: changed.append(relative.replace('\\', '/'))
    unexpected = sorted(set(changed)-ALLOWED_EXISTING_EDITS)
    media_before = json.loads((baseline/'dataset_stat_before.json').read_text())
    media_now = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in (root/'data/dataset').rglob('*') if p.is_file()}
    dataset_equal = media_before == media_now
    locked = root/'data/foundation/v1/ubi_locked.csv'
    rows = load_locked(locked); by_id = {r['video_id']:r for r in rows}
    split_report = validate_locked(rows)
    caches = []
    for name, expected_split in (('train_index.json', 'train'), ('val_index.json', 'val')):
        index = json.loads((root/'data/pose_cache/advanced_v1'/name).read_text())
        if index['locked_manifest_sha256'] != sha256(locked):
            raise ValueError('Cache index manifest changed')
        for key, entry in index['entries'].items():
            row = by_id[key]
            if row['split'] != expected_split or sha256(entry['path']) != entry['cache_sha256']:
                raise ValueError('Indexed cache split/hash mismatch')
            _, meta = load_cache(entry['path'])
            if meta['video_id'] != key or meta['source_sha256'] != row['content_sha256']:
                raise ValueError('Indexed cache/source identity mismatch')
            caches.append(key)
    git_status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
        cwd=root, text=True)
    outside = [line for line in git_status.splitlines()
        if line[3:] and not line[3:].startswith('components/c3_aggression_detection/')]
    report = dict(schema='c3_completion_preservation_v1', protected_files=len(protected),
        unchanged_protected_files=len(protected)-len(changed)-len(missing),
        intended_existing_changes=sorted(changed), unexpected_changes=unexpected, missing_files=missing,
        original_dataset_files=len(media_before), dataset_sizes_and_mtimes_unchanged=dataset_equal,
        dataset_verification_scope='Exact path/size/mtime inventory; existing locked source hashes preserved. Full raw media was not rehashed.',
        locked_splits=split_report, indexed_pose_caches_verified=len(caches),
        status_entries_outside_component=outside, commit_performed=False, push_performed=False,
        passed=not (unexpected or missing or outside) and dataset_equal)
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report['passed']: raise SystemExit(1)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='outputs/development_completion_v1')
    parser.add_argument('--out', required=True)
    verify(parser.parse_args())
