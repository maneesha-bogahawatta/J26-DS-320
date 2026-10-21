"""Audit v4 alerts with an optional camera-wide notification cooldown.

This is a post-processing and evaluation tool, not a trained classifier.
The detector is untouched. Fight onset is used ONLY for retrospective metrics.
Run: python -m src.audit_v4_actionable_alerts --cooldown-s 3
"""
import argparse
import csv
import json
import math
from pathlib import Path, PureWindowsPath
from collections import Counter

DEFAULT_FIGHTS = Path('outputs/v4_evaluation/fight_candidates_v4.csv')
DEFAULT_CONTROLS = Path('outputs/v4_evaluation/no_fight_control_results_v4.csv')
DEFAULT_LABELS = Path('outputs/v3_evaluation/control_review_labels.csv')
DEFAULT_OUT = Path('outputs/v4_evaluation/actionable_audit')


def read_csv(path):
    with Path(path).open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def filename(value):
    return PureWindowsPath(str(value).replace('/', '\\')).name


def alert_events(row):
    """Read v4 emitted alerts. Ignore records that lack finite timestamps."""
    raw = row.get('v4_alerts') or '[]'
    events = json.loads(raw)
    if not isinstance(events, list):
        raise ValueError('v4_alerts must be a JSON list')
    accepted = []
    for event in events:
        if not isinstance(event, dict) or 'alert_s' not in event:
            raise ValueError('Malformed event: missing alert_s')
        timestamp = float(event['alert_s'])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid alert_s')
        accepted.append({**event, 'alert_s': timestamp})
    return sorted(accepted, key=lambda e: e['alert_s'])


def cooldown_events(events, seconds):
    """Causal camera-wide cooldown; first notification always retained.

    All person pairs within the same video share the cooldown.
    This cannot eliminate the first alarm in a false-positive video.
    """
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError('Cooldown must be a finite nonnegative number')
    emitted = []
    for event in events:
        if not emitted or event['alert_s'] - emitted[-1]['alert_s'] >= seconds - 1e-9:
            emitted.append(event)
    return emitted


def onset_s(row):
    if row.get('fight_onset_s') not in (None, ''):
        return float(row['fight_onset_s'])
    intervals = json.loads(row.get('intervals') or '[]')
    if not intervals:
        raise ValueError('Fight row lacks onset and intervals')
    return min(float(i[0]) for i in intervals)


def lead_band(lead):
    if lead is None:
        return 'no_alert'
    if lead > 3:
        return 'more_than_3s_early'
    if lead >= 1:
        return '1to3s_early'
    if lead >= 0:
        return 'less_than_1s_early'
    return 'post_onset'


def score_rows(rows, kind, cooldown, labels=None):
    labels = labels or {}
    scored = []
    errors = []
    for row in rows:
        name = filename(row.get('video', ''))
        try:
            raw = alert_events(row)
            filtered = cooldown_events(raw, cooldown)
            onset = onset_s(row) if kind == 'fights' else None
            first = filtered[0]['alert_s'] if filtered else None
            lead = (onset - first) if first is not None and onset is not None else None
            any_target = (any(1 <= onset - a['alert_s'] <= 3 for a in filtered)
                          if onset is not None else False)
            raw_target = (any(1 <= onset - a['alert_s'] <= 3 for a in raw)
                          if onset is not None else False)
            result = {
                'video': name,
                'dataset': kind,
                'review_label': labels.get(name, 'unreviewed') if kind == 'controls' else '',
                'fight_onset_s': '' if onset is None else round(onset, 6),
                'raw_alert_count': len(raw),
                'actionable_alert_count': len(filtered),
                'suppressed_notifications': len(raw) - len(filtered),
                'raw_first_alert_s': '' if not raw else round(raw[0]['alert_s'], 6),
                'actionable_first_alert_s': '' if first is None else round(first, 6),
                'first_alert_lead_s': '' if lead is None else round(lead, 6),
                'first_alert_band': lead_band(lead) if onset is not None else ('has_alert' if first is not None else 'no_alert'),
                'raw_any_in_1to3s': int(raw_target),
                'actionable_any_in_1to3s': int(any_target),
                'actionable_timestamps_s': json.dumps([round(e['alert_s'], 6) for e in filtered]),
            }
            scored.append(result)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append({'video': name, 'dataset': kind, 'error': str(exc)})
    return scored, errors


FIELDS = [
    'video', 'dataset', 'review_label', 'fight_onset_s', 'raw_alert_count',
    'actionable_alert_count', 'suppressed_notifications', 'raw_first_alert_s',
    'actionable_first_alert_s', 'first_alert_lead_s', 'first_alert_band',
    'raw_any_in_1to3s', 'actionable_any_in_1to3s', 'actionable_timestamps_s',
]


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(scored):
    return {
        'windows': len(scored),
        'windows_with_alarms': sum(int(r['raw_alert_count'] > 0) for r in scored),
        'raw_alerts': sum(r['raw_alert_count'] for r in scored),
        'after_cooldown_alerts': sum(r['actionable_alert_count'] for r in scored),
        'suppressed_notifications': sum(r['suppressed_notifications'] for r in scored),
        'first_alert_bands': dict(Counter(r['first_alert_band'] for r in scored)),
        'raw_any_in_1to3s': sum(r['raw_any_in_1to3s'] for r in scored),
        'after_cooldown_any_in_1to3s': sum(r['actionable_any_in_1to3s'] for r in scored),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fight-csv', type=Path, default=DEFAULT_FIGHTS)
    ap.add_argument('--control-csv', type=Path, default=DEFAULT_CONTROLS)
    ap.add_argument('--label-csv', type=Path, default=DEFAULT_LABELS)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--cooldown-s', type=float, default=3.0)
    args = ap.parse_args(argv)
    if args.cooldown_s < 0 or not math.isfinite(args.cooldown_s):
        ap.error('--cooldown-s must be finite and nonnegative')
    try:
        fights = read_csv(args.fight_csv)
        controls = read_csv(args.control_csv)
    except (OSError, csv.Error) as exc:
        ap.error(str(exc))
    labels = {}
    if args.label_csv.is_file():
        labels = {filename(r.get('video', '')): r.get('review_label', 'unreviewed')
                  for r in read_csv(args.label_csv)}
    fight_out, fight_err = score_rows(fights, 'fights', args.cooldown_s)
    control_out, control_err = score_rows(controls, 'controls', args.cooldown_s, labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.out_dir / 'fight_actionable_alerts.csv', fight_out, FIELDS)
    save_csv(args.out_dir / 'control_actionable_alerts.csv', control_out, FIELDS)
    save_csv(args.out_dir / 'parse_errors.csv', fight_err + control_err, ['video', 'dataset', 'error'])
    summary = {
        'cooldown_s': args.cooldown_s,
        'fight_candidates': summarize(fight_out),
        'no_fight_controls': summarize(control_out),
        'reviewed_non_aggressive_controls': sum(r['review_label'] == 'non_aggressive' for r in control_out),
        'reviewed_non_aggressive_controls_with_alarms': sum(
            r['review_label'] == 'non_aggressive' and r['raw_alert_count'] > 0 for r in control_out),
        'parse_errors': len(fight_err) + len(control_err),
        'caveat': 'Alarm suppression reduces notifications, not initial control alarm incidence. '
                  'These are retrospective timing and heuristic counts, not verified accuracy.',
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('Saved:', args.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
