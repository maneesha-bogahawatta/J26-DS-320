"""Export a balanced, UNLABELED human-review batch for C3 early-aggression research.

Run from components/c3_aggression_detection:
    python -m src.export_v4_training_review

This tool reads prior offline alert timing diagnostics; it NEVER converts
heuristic alerts or the dataset's fight-onset annotation to precursor labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path('outputs/v4_evaluation/fight_alert_lead_times.csv')
DEFAULT_SHORTLIST = Path('outputs/v4_evaluation/fight_alert_review_shortlist.csv')
DEFAULT_MANIFEST = Path('data/manifests/ubi_early_review_candidates.csv')
DEFAULT_OUTPUT = Path('outputs/v4_evaluation/training_review_batch')


def read_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def basename(value: str) -> str:
    return str(value or '').replace('\\', '/').split('/')[-1]


def choose_balanced(rows: list[dict], seed: int = 42, target: int = 4,
                    very_early: int = 3, late: int = 2,
                    no_alert: int = 2, burst: int = 2) -> list[tuple[dict, str]]:
    """Balance *audit categories*, not class labels. One video per batch."""
    rng = random.Random(seed)
    selected: list[tuple[dict, str]] = []
    seen: set[str] = set()

    groups = [
        ('first_1to3s', '1_to_3s_early', target),
        ('first_gt3s', 'more_than_3s_early', very_early),
        ('first_lt1s', 'less_than_1s_early', late),
        ('no_alert', 'no_alert', no_alert),
    ]
    for label, value, quota in groups:
        candidates = [r for r in rows if r.get('first_alert_category') == value
                      and basename(r.get('video', '')) not in seen]
        rng.shuffle(candidates)
        for row in candidates[:max(0, quota)]:
            key = basename(row.get('video', ''))
            selected.append((row, label))
            seen.add(key)

    # Dense distinct-pair bursts are a separate TRACKING audit, not positives.
    if burst:
        burst_candidates = [r for r in rows
                            if basename(r.get('video', '')) not in seen
                            and int(float(r.get('max_distinct_pairs_in_1s') or 0)) >= 4]
        burst_candidates.sort(key=lambda r: (-int(float(r.get('max_distinct_pairs_in_1s') or 0)),
                                             basename(r.get('video', ''))))
        for row in burst_candidates[:burst]:
            selected.append((row, 'pair_burst_audit'))
            seen.add(basename(row.get('video', '')))
    return selected


def video_index(manifest: Path, roots: list[Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if manifest.exists():
        for row in read_csv(manifest):
            raw = row.get('video', '')
            if not raw:
                continue
            name = basename(raw)
            p = Path(raw)
            if p.is_file():
                paths[name] = p
            else:
                for root in roots:
                    direct = root / name
                    if direct.is_file():
                        paths[name] = direct
                        break
    # Handles datasets organized into several subfolders; index only once.
    for root in roots:
        if root.is_dir():
            for p in root.rglob('*.mp4'):
                paths.setdefault(p.name, p)
    return paths


def clip_bounds(onset: float, duration: float, pre: float = 5.0,
                post: float = 1.0) -> tuple[float, float]:
    if not math.isfinite(onset) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid onset or duration')
    if onset < 0 or pre <= 0 or post < 0:
        raise ValueError('Invalid onset/pre/post')
    start = max(0., onset-pre)
    end = min(duration, onset+post)
    if end <= start:
        raise ValueError('Empty review clip range')
    return start, end


def export_video(source: Path, dest: Path, onset: float,
                 pre: float, post: float, width: int = 960) -> tuple[float, float, int]:
    import cv2
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {source}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (math.isfinite(fps) and 0 < fps <= 240 and nframes > 0):
            raise ValueError(f'Invalid video metadata for {source}')
        start, end = clip_bounds(onset, nframes / fps, pre, post)
        first = max(0, int(math.floor(start * fps)))
        last = min(nframes, int(math.ceil(end * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_w < 1 or orig_h < 1:
            raise ValueError('Invalid frame geometry')
        scale = min(1., width / orig_w)
        new_w, new_h = max(2, int(orig_w*scale)//2*2), max(2, int(orig_h*scale)//2*2)
        dest.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_w,new_h))
        if not writer.isOpened():
            raise RuntimeError(f'Could not create MP4 writer: {dest}')
        wrote = 0
        try:
            for i in range(first, last):
                ok, frame = cap.read()
                if not ok:
                    break
                if (orig_w, orig_h) != (new_w, new_h):
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # Only exported REVIEW copies get timestamp labels; original video is unchanged.
                secs = i/fps
                cv2.rectangle(frame, (0, 0), (min(new_w, 620), 34), (0,0,0), -1)
                cv2.putText(frame, f'Original t={secs:.3f}s | annotated onset={onset:.3f}s',
                            (6, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
                writer.write(frame)
                wrote += 1
        finally:
            writer.release()
        if not wrote:
            dest.unlink(missing_ok=True)
            raise ValueError('No frames decoded')
        return round(first/fps, 6), round(min(last,first+wrote)/fps, 6), wrote
    finally:
        cap.release()


def run(args: argparse.Namespace) -> int:
    rows = read_csv(Path(args.input))
    if not rows:
        raise ValueError(f'No rows in {args.input}')
    shortlist = {}
    if Path(args.shortlist).exists():
        shortlist = {basename(r['video']): r for r in read_csv(Path(args.shortlist)) if r.get('video')}
    selection = choose_balanced(rows, args.seed, args.target, args.very_early,
                                args.late, args.no_alert, args.burst)
    roots = [Path(args.videos_root), Path('data/dataset/UBI_FIGHTS/videos')]
    index = video_index(Path(args.manifest), roots)
    out = Path(args.output)
    previous = out / 'review_annotations.csv'
    if previous.exists():
        existing = read_csv(previous)
        if any(str(r.get('review_label', '')).strip().lower() not in ('', 'unreviewed') for r in existing):
            raise ValueError(
                f'{previous} contains human review labels; refusing to overwrite. '
                'Back it up and choose a new --output directory.'
            )
    clips = out / 'clips'
    clips.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (row, group) in enumerate(selection, 1):
        name = basename(row.get('video', ''))
        onset = float(row['fight_onset_s'])
        source = index.get(name)
        exported = ''
        actual_start = actual_end = ''
        status = 'source_missing'
        error = ''
        if source is not None:
            dest = clips / f'{Path(name).stem}_review.mp4'
            if args.dry_run:
                status = 'dry_run'
            else:
                try:
                    actual_start, actual_end, n = export_video(
                        source, dest, onset, args.pre, args.post, args.width)
                    exported = str(dest)
                    status = 'exported'
                except Exception as exc:
                    status = 'error'
                    error = str(exc)
        extra = shortlist.get(name, {})
        records.append({
            'video': name,
            'source_video': '' if source is None else str(source),
            'clip_path': exported,
            'clip_start_s': actual_start,
            'clip_end_s': actual_end,
            'fight_onset_s': onset,
            'selection_group': group,
            'first_alert_category': row.get('first_alert_category', ''),
            'first_alert_s': row.get('first_alert_s', ''),
            'first_alert_lead_s': row.get('first_alert_lead_s', ''),
            'first_target_alert_s': row.get('first_target_alert_s', ''),
            'max_distinct_pairs_in_1s': row.get('max_distinct_pairs_in_1s', ''),
            'review_reason': extra.get('review_reason', ''),
            # Intentionally blank: NEVER train on model outputs as labels.
            'review_label': 'unreviewed',
            'precursor_start_s': '',
            'precursor_end_s': '',
            'first_physical_contact_s': '',
            'reviewer_1': '',
            'reviewer_2': '',
            'review_notes': '',
            'export_status': status,
            'export_error': error,
        })
        print(f'[{i}/{len(selection)}] {name}: {group} / {status}')
    csv_path = out / 'review_annotations.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    summary = {
        'selected': len(records),
        'groups': dict(Counter(r['selection_group'] for r in records)),
        'statuses': dict(Counter(r['export_status'] for r in records)),
        'labels': 'All unreviewed; not eligible as verified positive training labels',
        'source_rows': len(rows),
    }
    (out / 'export_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved:', csv_path)
    print('IMPORTANT: v4 alerts are suggestions, NOT verified precursor labels.')
    return 0 if not any(r['export_status'] in ('error','source_missing') for r in records) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=str(DEFAULT_INPUT))
    p.add_argument('--shortlist', default=str(DEFAULT_SHORTLIST))
    p.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    p.add_argument('--videos-root', default='data/dataset/UBI_FIGHTS/videos')
    p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target', type=int, default=4, help='First alert 1–3s before onset')
    p.add_argument('--very-early', type=int, default=3)
    p.add_argument('--late', type=int, default=2)
    p.add_argument('--no-alert', type=int, default=2)
    p.add_argument('--burst', type=int, default=2)
    p.add_argument('--pre', type=float, default=5)
    p.add_argument('--post', type=float, default=1)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if any(getattr(args, n) < 0 for n in ('target','very_early','late','no_alert','burst')):
        p.error('Selection quotas must be nonnegative')
    if args.width < 200 or args.pre <= 0 or args.post < 0:
        p.error('Invalid width/pre/post')
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
