"""Auto-triage Component 03 v3 no-fight control clips without inventing ground truth.

Run from the Component 03 root:
    python -m src.auto_triage_control_reviews

Only AUTO fields are filled; never overwrites review_label, reviewer or notes.
The v3 detector is NOT an independent classifier of aggressive/non-aggressive behavior.
"""

import argparse
import csv
import json
import math
import ntpath
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_REVIEW = Path('outputs/v3_evaluation/control_review_labels.csv')
DEFAULT_RESULTS = Path('outputs/v3_evaluation/no_fight_control_results.csv')
AUTO_FIELDS = [
    'auto_flag', 'auto_priority', 'auto_proposal_count',
    'auto_proposal_duration_s', 'auto_first_proposal_s',
    'auto_reason', 'auto_verification',
]


def read_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f'Missing CSV header: {path}')
        return list(reader), list(reader.fieldnames)


def basename(path):
    return ntpath.basename(str(path).replace('\\', '/'))


def numeric(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def row_key(row):
    """Use basename AND time bounds; avoids mismatching multiple windows of one video."""
    start = row.get('window_start_s', row.get('control_window_start_s', ''))
    end = row.get('window_end_s', row.get('control_window_end_s', ''))
    return (basename(row.get('video', '')).lower(), round(numeric(start, -1), 3), round(numeric(end, -1), 3))


def parse_intervals(value):
    try:
        intervals = json.loads(value or '[]')
        if not isinstance(intervals, list):
            return [], True
        result = []
        for pair in intervals:
            if not isinstance(pair, list) or len(pair) != 2:
                return [], True
            a, b = map(float, pair)
            if not (math.isfinite(a) and math.isfinite(b) and b > a):
                return [], True
            result.append((a, b))
        return result, False
    except (ValueError, TypeError, json.JSONDecodeError):
        return [], True


def decision(row):
    status = (row.get('v3_status') or '').strip()
    score = numeric(row.get('v3_max_score'))
    quality = numeric(row.get('v3_quality'))
    intervals, bad_json = parse_intervals(row.get('v3_proposals'))
    start = numeric(row.get('window_start_s'), numeric(row.get('control_window_start_s'), -1))
    end = numeric(row.get('window_end_s'), numeric(row.get('control_window_end_s'), -1))
    # Only count motion events intersecting the sampled control window.
    clipped = [(max(a, start), min(b, end)) for a, b in intervals]
    clipped = [(a, b) for a, b in clipped if b > a]
    total_duration = sum(b-a for a, b in clipped)

    if bad_json or status == 'error':
        flag, priority = 'data_or_inference_error', 'high'
        reason = 'Detector/error or invalid interval JSON; inspect results and video'
    elif status == 'proposal':
        if not clipped:
            flag, priority = 'proposal_outside_window', 'high'
            reason = 'Detector proposed a window that does not overlap the sampled control clip'
        elif quality < .55:
            flag, priority = 'low_quality_alarm', 'high'
            reason = f'Pose-based alarm with low tracking quality ({quality:.2f}); artifact possible'
        elif total_duration >= .8 and quality >= .65:
            flag, priority = 'sustained_alarm', 'high'
            reason = f'Pose-based alarm lasts ~{total_duration:.2f}s despite no labeled fight'
        elif score >= .75 and quality >= .65:
            flag, priority = 'high_score_alarm', 'high'
            reason = f'High heuristic score ({score:.2f}) despite no labeled fight'
        elif score < .45:
            flag, priority = 'borderline_alarm', 'medium'
            reason = f'Heuristic score ({score:.2f}) is near default threshold; audit ordinary motion'
        else:
            flag, priority = 'motion_alarm', 'medium'
            reason = 'Pose-based alarm in no-labeled-fight clip; visually distinguish threats from ordinary activity'
    elif status == 'no_proposal':
        flag, priority = 'no_detector_alarm', 'audit'
        reason = 'No v3 alert; audit samples for missed threatening behavior'
    else:
        flag, priority = 'unknown_status', 'high'
        reason = f'Unexpected detector status: {status or "empty"}'

    return {
        'auto_flag': flag,
        'auto_priority': priority,
        'auto_proposal_count': str(len(clipped)),
        'auto_proposal_duration_s': f'{total_duration:.3f}',
        'auto_first_proposal_s': f'{min(a for a, _ in clipped):.3f}' if clipped else '',
        'auto_reason': reason,
        'auto_verification': 'REQUIRES_HUMAN_REVIEW',
    }


def update(review_path, results_path, *, dry_run=False):
    if not review_path.is_file():
        raise FileNotFoundError(
            f'{review_path} not found. First run:\n'
            'python -m src.export_control_review_clips --status proposal\n'
            'python -m src.export_control_review_clips --status no_proposal --limit 4'
        )
    if not results_path.is_file():
        raise FileNotFoundError(f'{results_path} not found. Run v3 scan-controls first.')
    reviews, fields = read_csv(review_path)
    results, _ = read_csv(results_path)
    result_map = {row_key(r): r for r in results}
    matched, unmatched = 0, []
    counts = Counter()
    for row in reviews:
        original = result_map.get(row_key(row))
        if original is None:
            unmatched.append((basename(row.get('video', '')), row.get('window_start_s','')))
            data = row
        else:
            data = dict(original)
            data.update(row)
            # Keep authoritative pose values from the scan output, not stale review export.
            for col in ('v3_status', 'v3_proposals', 'v3_max_score', 'v3_quality'):
                data[col] = original.get(col, '')
            matched += 1
        recommendation = decision(data)
        row.update(recommendation)
        counts[recommendation['auto_flag']] += 1
        # Deliberately preserve review_label, reviewer, notes and other manual columns.
    if not dry_run:
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = review_path.with_name(f'{review_path.stem}.backup_{suffix}{review_path.suffix}')
        shutil.copy2(review_path, backup)
        extra = [c for c in AUTO_FIELDS if c not in fields]
        with review_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields+extra, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(reviews)
        print('Backup:', backup)
        print('Updated:', review_path)
    print(f'Rows: {len(reviews)} | Matching detector results: {matched} | Unmatched: {len(unmatched)}')
    for flag, count in sorted(counts.items()):
        print(f'  {flag}: {count}')
    if unmatched:
        print('Warning: some clips did not match control scan windows; verify them:', unmatched[:5])
    print('Only auto_* columns were changed. review_label remains human-verified/unreviewed.')
    return reviews


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-csv', type=Path, default=DEFAULT_REVIEW)
    p.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    update(args.review_csv, args.results, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
