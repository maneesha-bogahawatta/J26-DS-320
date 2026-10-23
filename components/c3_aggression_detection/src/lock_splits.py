"""Lock source videos, exact duplicates, sampled re-encodes and known groups together.

Perceptual matching is conservative, not an exhaustive partial-footage search.
Human-discovered shared footage must be added using --groups and a new version.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.foundation_io import read_csv, sha256, video_id, write_csv, write_json

SPLITS = {'train', 'val', 'test'}


def validate_locked(rows, test_ids=()):
    errors = []
    for field in ('video_id', 'group', 'source_group', 'leakage_group', 'perceptual_group', 'content_sha256', 'video'):
        seen = defaultdict(set)
        for row in rows:
            value = row.get(field, '')
            if value:
                seen[value.casefold()].add(row.get('split'))
        errors.extend(f'{field} crosses splits: {key}' for key, splits in seen.items() if len(splits) > 1)
    identities = [r['video_id'] for r in rows]
    if len(set(identities)) != len(identities):
        errors.append('Repeated video identity in locked manifest')
    by_id = {r['video_id']: r for r in rows}
    for key in test_ids:
        if key not in by_id or by_id[key]['split'] != 'test':
            errors.append(f'Reserved test identity missing or reassigned: {key}')
    for row in rows:
        if row.get('split') not in SPLITS or row.get('split_locked') != 'true':
            errors.append(f'Unlocked/invalid assignment: {row["video_id"]}')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'passed': True, 'videos': len(rows), 'counts': dict(Counter(r['split'] for r in rows)),
            'checks': ['video_identity', 'source_path', 'exact_content_hash', 'known_group',
                       'leakage_component', 'reserved_test_membership', 'locked_assignments']}


def load_locked(path):
    path = Path(path)
    rows = read_csv(path)
    metadata = json.loads(path.with_name('split_metadata.json').read_text(encoding='utf-8'))
    if sha256(path) != metadata['locked_manifest_sha256']:
        raise ValueError('Locked manifest changed; build a new audited version')
    validate_locked(rows, metadata['reserved_test_ids'])
    return rows


def companion_report(rows, report_dir):
    companions = []
    for row in rows:
        if row['split']=='test' and row['test_list_member']!='true':
            companions.append({'video_id':row['video_id'],'reason':'Shares locked duplicate/source component with test footage',
                               'component':row['leakage_group'],
                               'test_list_relatives':[r['video_id'] for r in rows if r['test_list_member']=='true' and
                                   r['leakage_group']==row['leakage_group']]})
    write_json(Path(report_dir)/'test_companions.json',{'additional_test_companions':companions,
               'policy':'Preserve all reserved test identities and hold out their duplicate/source companions'})


def perceptual_signature(path, samples=8):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    try:
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or n <= 0 or fps <= 0:
            raise ValueError(f'Cannot fingerprint video: {path}')
        hashes = []
        textured = 0
        for position in np.linspace(.05, .95, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(n-1, int(position*n)))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Fingerprint frame decode failed: {path}')
            gray = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY).astype('float32')
            textured += int(gray.std() > 8)
            low = cv2.dct(gray)[:8, :8].reshape(-1)[1:]
            hashes.append(np.packbits(low > np.median(low)).tobytes().hex())
        return {'duration_s': n/fps, 'hashes': hashes, 'textured_frames': textured}
    finally:
        cap.release()


def similar(a, b):
    if min(a['textured_frames'], b['textured_frames']) < 6:
        return False
    if abs(a['duration_s']-b['duration_s']) > max(.15, .01*min(a['duration_s'], b['duration_s'])):
        return False
    distances = [(int(x, 16)^int(y, 16)).bit_count() for x, y in zip(a['hashes'], b['hashes'])]
    return max(distances) <= 8 and sum(distances)/len(distances) <= 3


def assign(rows, test_ids, known_groups=None, seed=42, val_fraction=.15):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    rows = [dict(r, video_id=video_id(r['video'])) for r in rows]
    ids = [r['video_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Ambiguous duplicate video identities')
    if set(test_ids)-set(ids):
        raise ValueError(f'Test list identities missing from manifest: {sorted(set(test_ids)-set(ids))}')
    parent = {key:key for key in ids}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def join(a, b):
        a, b = root(a), root(b)
        parent[max(a,b)] = min(a,b)
    aliases = {}
    for row in rows:
        if row.get('split', '') not in SPLITS | {''}:
            raise ValueError('Invalid existing split')
        row['source_group'] = (known_groups or {}).get(row['video_id'], row.get('source_group', ''))
        for field in ('group', 'source_group', 'content_sha256', 'perceptual_group'):
            if row.get(field):
                key = (field, row[field].casefold())
                if key in aliases:
                    join(row['video_id'], aliases[key])
                aliases[key] = row['video_id']
    components = defaultdict(list)
    for row in rows:
        components[root(row['video_id'])].append(row)
    pending = defaultdict(list)
    for key, group in components.items():
        named = {r['split'] for r in group if r.get('split')}
        if any(r['video_id'] in test_ids for r in group):
            named.add('test')
        if len(named) > 1:
            raise ValueError(f'Conflicting existing/reserved split assignments in duplicate group {key}: {named}')
        if named:
            split = named.pop()
            for row in group:
                row['split'] = split
        else:
            stratum = bool(any(json.loads(r.get('intervals') or '[]') for r in group))
            pending[stratum].append(key)
    for keys in pending.values():
        keys.sort(key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())
        nval = round(len(keys)*val_fraction)
        for i,key in enumerate(keys):
            for row in components[key]:
                row['split'] = 'val' if i < nval else 'train'
    for key, group in components.items():
        for row in group:
            row.update(leakage_group=key, split_locked='true',
                       test_list_member='true' if row['video_id'] in test_ids else 'false')
    rows.sort(key=lambda r:r['video_id'])
    validate_locked(rows, test_ids)
    return rows


def run(args):
    rows = read_csv(args.manifest)
    tests = {video_id(s.strip().split(',')[0].strip('"')) for s in
             Path(args.test_list).read_text(encoding='utf-8-sig').splitlines() if s.strip()}
    assignment_audit = []
    existing = defaultdict(set)
    for path in sorted(Path(args.manifest).parent.glob('*.csv')):
        if path.name.startswith('EXAMPLE_'):
            continue
        rr = read_csv(path)
        count = 0
        for row in rr:
            if row.get('video') and row.get('split'):
                existing[video_id(row['video'])].add(row['split'])
                count += 1
        assignment_audit.append({'manifest':str(path), 'rows':len(rr), 'named_splits':count})
    signatures = {}
    for i,row in enumerate(rows,1):
        key = video_id(row['video'])
        named = existing[key]
        if len(named) > 1:
            raise ValueError(f'Existing manifests disagree about split: {key}')
        if named:
            row['split'] = next(iter(named))
        row['content_sha256'] = sha256(row['video'])
        signatures[key] = perceptual_signature(row['video'])
        if i % 100 == 0:
            print(f'Hashed and fingerprinted {i}/{len(rows)}', flush=True)
    known = {}
    if args.groups:
        for row in read_csv(args.groups):
            key = video_id(row['video'])
            if key not in signatures or not row.get('source_group'):
                raise ValueError('Unknown video or empty source_group in --groups')
            if key in known and known[key] != row['source_group']:
                raise ValueError(f'Conflicting source groups: {key}')
            known[key] = row['source_group']
    # Union similar whole-video signatures before random assignment.
    pairs = []
    parent = {key:key for key in signatures}
    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key
    keys = sorted(signatures)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if similar(signatures[a], signatures[b]):
                pairs.append([a,b])
                x,y = root(a), root(b)
                parent[max(x,y)] = min(x,y)
    for row in rows:
        row['perceptual_group'] = root(video_id(row['video']))
    locked = assign(rows, tests, known, args.seed, args.val_fraction)
    overlap = sorted(video_id(r['video']) for r in read_csv(args.review_batch) if video_id(r['video']) in tests)
    report = validate_locked(locked, tests)
    exact = defaultdict(list)
    for row in locked:
        exact[row['content_sha256']].append(row['video_id'])
    report.update(reserved_test_videos=len(tests), test_review_batch_overlap=overlap,
                  exact_duplicate_sets=[v for v in exact.values() if len(v)>1],
                  perceptual_duplicate_pairs=pairs, existing_manifest_audit=assignment_audit,
                  known_source_groups=known,
                  limitations=['Whole-video sampled fingerprints do not establish absence of partial overlaps, crops or time-shifted duplicates.',
                               'Add known shared footage with --groups; audit remaining source relationships before scientific evaluation.',
                               'Prior heuristic/test review exposure is preserved and must be disclosed.'])
    out = Path(args.out_dir)
    write_csv(out/'ubi_locked.csv', locked)
    metadata = {'schema':'c3_locked_splits_v1', 'seed':args.seed, 'validation_fraction_of_unreserved_groups':args.val_fraction,
                'input_sha256':sha256(args.manifest), 'test_list_sha256':sha256(args.test_list),
                'reserved_test_ids':sorted(tests), 'locked_manifest_sha256':sha256(out/'ubi_locked.csv'),
                'fingerprint_policy':'8 DCT frames at 5..95%; textured>=6; duration delta<=1% or .15s; mean Hamming<=3,max<=8',
                'group_map_sha256':sha256(args.groups) if args.groups else None}
    write_json(out/'split_metadata.json', metadata)
    write_json(out/'video_fingerprints.json', signatures)
    write_json(Path(args.report_dir)/'split_validation.json', report)
    companion_report(locked,args.report_dir)
    print(json.dumps({k:v for k,v in report.items() if k not in ('existing_manifest_audit','known_source_groups')}, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifests/ubi_review.csv')
    p.add_argument('--test-list', default='data/dataset/UBI_FIGHTS/test_videos.csv')
    p.add_argument('--review-batch', default='outputs/v4_evaluation/training_review_batch/review_annotations.csv')
    p.add_argument('--groups', help='Optional CSV with video,source_group for known shared/cropped footage')
    p.add_argument('--out-dir', default='data/foundation/v1')
    p.add_argument('--report-dir', default='outputs/phase1_foundation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=.15)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    if args.validate_only:
        rows=load_locked(Path(args.out_dir)/'ubi_locked.csv')
        print(json.dumps(validate_locked(rows),indent=2))
        companion_report(rows,args.report_dir)
    else:
        run(args)


if __name__ == '__main__':
    main()
