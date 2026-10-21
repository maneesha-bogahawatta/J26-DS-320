"""Bounded pose-visible context selection and explicit weak first-onset targets.

Weak forecast means an original UBI first annotated onset within three seconds.
It does NOT mean an independently verified aggression precursor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.foundation_io import sha256, write_json
from src.lock_splits import load_locked
from src.pose_sequences import CACHE_SCHEMA, extract_video, load_cache
from src.pose_dataset import sequence_window

PROTOCOL = 'ubi_weak_first_onset_within_3s_v1'


def weak_targets(t, intervals, horizon=3., boundary=.1):
    """Unknown post-onset forecast time is masked; active fight uses weak binary labels."""
    ranges = sorted(intervals)
    fight = float(any(a <= t < b for a, b in ranges))
    mask_fight = not any(abs(t-edge) < boundary for pair in ranges for edge in pair)
    if not ranges:
        return np.array([0., 0.], np.float32), np.array([True, True])
    onset = ranges[0][0]
    remaining = onset-t
    mask_pre = remaining > boundary and abs(remaining-horizon) >= boundary
    return np.array([float(0 < remaining <= horizon), fight], np.float32), np.array([mask_pre, mask_fight])


def choose_candidates(rows, split, positive, excluded, limit=40, seed=73):
    candidates = [r for r in rows if r['split'] == split and r['leakage_group'] not in excluded
                  and bool(json.loads(r['intervals'])) == positive and float(r['duration_s']) >= 8]
    if positive:
        candidates = [r for r in candidates if 5 <= json.loads(r['intervals'])[0][0] <= float(r['duration_s'])-3]
    return sorted(candidates, key=lambda r: hashlib.sha256(f'{seed}:{r["video_id"]}'.encode()).hexdigest())[:limit]


def expand(args):
    # All model/settings files remain inside C3 and implicit downloads are disabled.
    output = Path(args.out_dir).resolve()
    (output/'.ultralytics/Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(output/'.ultralytics')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    from ultralytics import YOLO
    rows = load_locked(args.locked)
    indexes = {s: json.loads(Path(p).read_text()) for s, p in
               (('train', args.train_index), ('val', args.val_index))}
    by_id = {r['video_id']: r for r in rows}
    excluded = {by_id[k]['leakage_group'] for idx in indexes.values() for k in idx['entries']}
    model = YOLO(str(Path('yolo11n-pose.pt').resolve()))
    choices = []
    probes = []
    for split, positive, quota in (('train', False, 6), ('val', False, 4), ('val', True, 4)):
        accepted = 0
        for row in choose_candidates(rows, split, positive, excluded):
            ranges = json.loads(row['intervals'])
            start = ranges[0][0]-5 if positive else max(0., (float(row['duration_s'])-8)/2)
            cap = cv2.VideoCapture(row['video'])
            visible = []
            for t in (start+2, start+5):
                cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
                ok, frame = cap.read()
                count = 0
                if ok:
                    result = model.predict(frame, imgsz=640, conf=.3, device='cpu', verbose=False)[0]
                    if result.keypoints is not None:
                        conf = result.keypoints.conf
                        if conf is not None:
                            count = int(((conf.cpu().numpy() >= .3).sum(1) >= 10).sum())
                visible.append(count)
            cap.release()
            probes.append(dict(video_id=row['video_id'], split=split, positive=positive, visible_people=visible))
            if min(visible) < 2:
                continue
            choices.append(dict(video_id=row['video_id'], split=split, start_s=start, end_s=start+8,
                                reason='weak fight context' if positive else 'pose-visible weak no-fight context'))
            excluded.add(row['leakage_group'])
            accepted += 1
            print(f'Selected {split}: {row["video_id"]} people={visible}', flush=True)
            if accepted >= quota:
                break
    write_json(output/'selection.json', dict(schema='c3_advanced_context_selection_v1',
        locked_manifest_sha256=sha256(args.locked), jobs=choices, probes=probes,
        note='Retrospective pose-visible diagnostic sampling; not a blind population benchmark'))
    for job in choices:
        row = by_id[job['video_id']]
        print(f'Extracting bounded 8 s context: {row["video_id"]}', flush=True)
        path, meta = extract_video(row, output/'new_cache', start=job['start_s'], end=job['end_s'])
        indexes[job['split']]['entries'][row['video_id']] = dict(path=str(path.resolve()),
            cache_sha256=sha256(path), source_sha256=meta['source_sha256'])
        # Per-completed-video indexes survive interruption, without rewriting earlier snapshots.
        write_json(output/'completed'/(row['video_id']+'.json'), dict(split=job['split'],
            video_id=row['video_id'], entry=indexes[job['split']]['entries'][row['video_id']]))
    for split, index in indexes.items():
        write_json(output/(split+'_index.json'), index)
    write_json(output/'expansion_summary.json', dict(jobs_completed=len(choices),
        new_source_seconds=sum(j['end_s']-j['start_s'] for j in choices),
        train_sources=len(indexes['train']['entries']), val_sources=len(indexes['val']['entries']),
        test_sources_used=0, labels_remain_weak=True))


def internal_partition(rows, entries, seed=73):
    """Internal development groups come ONLY from locked training sources."""
    by_id = {r['video_id']: r for r in rows}
    groups = {}
    for key in entries:
        row = by_id[key]
        if row['split'] != 'train':
            raise ValueError('Internal fitting/development partition requires locked train sources')
        groups.setdefault(row['leakage_group'], bool(json.loads(row['intervals'])))
    dev = set()
    for positive in (False, True):
        candidates = sorted([g for g, label in groups.items() if label == positive],
                            key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
        n = max(1, round(.2*len(candidates))) if len(candidates) > 1 else 0
        dev.update(candidates[:n])
    return {key: 'dev' if by_id[key]['leakage_group'] in dev else 'fit' for key in entries}


def materialize(args):
    rows = load_locked(args.locked)
    by_id = {r['video_id']: r for r in rows}
    indexes = {s: json.loads(Path(p).read_text()) for s, p in
               (('train', args.train_index), ('val', args.val_index))}
    expected = sha256(args.locked)
    if any(idx.get('schema') != CACHE_SCHEMA or idx.get('locked_manifest_sha256') != expected for idx in indexes.values()):
        raise ValueError('Indices differ from locked splits')
    roles = internal_partition(rows, indexes['train']['entries'])
    fields = {key: [] for key in ('sk', 'joint_mask', 'temporal_mask', 'frame_times', 'targets', 'target_mask', 'quality', 'cutoff_s', 'video_id', 'role')}
    coverage = []
    contract = None
    used_groups = {'train': set(), 'val': set()}
    for split, index in indexes.items():
        for key, entry in sorted(index['entries'].items()):
            row = by_id[key]
            if row['split'] != split or split == 'test' or sha256(entry['path']) != entry['cache_sha256']:
                raise ValueError('Changed, wrong-split or test cache')
            arrays, meta = load_cache(entry['path'])
            if meta['video_id'] != key or meta['source_sha256'] != row['content_sha256']:
                raise ValueError('Cache/source identity mismatch')
            cfg = {k: meta.get(k) for k in ('schema', 'preprocessing', 'sample_fps', 'weights_sha256', 'tracker_sha256', 'libraries')}
            if contract is not None and cfg != contract:
                raise ValueError('Mixed preprocessing contract')
            contract = cfg
            used_groups[split].add(row['leakage_group'])
            intervals = json.loads(row['intervals'])
            candidates = {}
            last = -np.inf
            for i, t in enumerate(arrays['t']):
                if t-last < args.step_s-1e-8:
                    continue
                last = t
                y, mask = weak_targets(float(t), intervals)
                group = tuple(y.astype(int)) + tuple(mask.astype(int))
                candidates.setdefault(group, []).append(i)
            selected = set()
            for group, indices in candidates.items():
                limit = args.per_bin if split == 'train' else len(indices)
                selected.update(indices[j] for j in np.unique(np.linspace(0, len(indices)-1, min(limit, len(indices))).round().astype(int)))
            eligible = 0
            for i in sorted(selected):
                sample = sequence_window(arrays, i, frames=48)
                if sample['temporal_mask'].sum() < 8 or sample['quality'] < .45:
                    continue
                y, mask = weak_targets(sample['cutoff_s'], intervals)
                if not mask.any():
                    continue
                eligible += 1
                values = dict(sample, targets=y, target_mask=mask, role=roles[key] if split == 'train' else 'val', video_id=key)
                for name in fields:
                    fields[name].append(values[name])
            coverage.append(dict(video_id=key, split=split, role=roles.get(key, 'val'),
                                 candidate_windows=len(selected), eligible_windows=eligible))
            if len(coverage) % 20 == 0: print(f'Materialized sources: {len(coverage)}', flush=True)
    if used_groups['train'] & used_groups['val']:
        raise ValueError('Source-group leakage')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    values = {key: np.asarray(value) for key, value in fields.items()}
    with (output/'windows.npz').open('xb') as stream:
        np.savez_compressed(stream, **values)
    summary = {role: dict(windows=int((values['role'] == role).sum()),
        videos=len(set(values['video_id'][values['role'] == role].tolist())),
        supervised=values['target_mask'][values['role'] == role].sum(0).tolist(),
        positives=(values['targets'][values['role'] == role]*values['target_mask'][values['role'] == role]).sum(0).astype(int).tolist())
        for role in ('fit', 'dev', 'val')}
    write_json(output/'metadata.json', dict(schema='c3_advanced_windows_v1', protocol=PROTOCOL,
        target_heads=['weak_first_onset_within_3s', 'weak_active_fight'],
        locked_manifest_sha256=expected, index_sha256={s: sha256(p) for s, p in
            (('train', args.train_index), ('val', args.val_index))},
        feature_contract=contract, windows_sha256=sha256(output/'windows.npz'),
        per_bin=args.per_bin, step_s=args.step_s, seed=73, summary=summary,
        coverage=coverage, roles=roles, test_sources_used=0,
        locked_manifest_path=str(Path(args.locked).resolve()),
        source_groups={k: by_id[k]['leakage_group'] for idx in indexes.values() for k in idx['entries']},
        label_policy='Original weak binary UBI first onset; no verified or AI precursor targets',
        forecast_policy='Positive: first annotated onset in next 3s; pre-onset negatives outside horizon; after first onset forecast masked. 0.1s boundary guard.',
        validation_policy='Locked val used only after train-only internal development selection; historical pilot exposure disclosed.'))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['expand', 'materialize'])
    p.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--train-index', default='data/pose_cache/advanced_v1/train_recovered_index.json')
    p.add_argument('--val-index', default='data/pose_cache/quick_v1/pilot_val_index.json')
    p.add_argument('--out-dir', default='data/pose_cache/advanced_v1')
    p.add_argument('--step-s', type=float, default=.4)
    p.add_argument('--per-bin', type=int, default=6)
    args = p.parse_args()
    if args.step_s <= 0 or args.per_bin < 1:
        p.error('Positive sampling step/per-bin limit required')
    (expand if args.command == 'expand' else materialize)(args)


if __name__ == '__main__':
    main()
