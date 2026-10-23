"""Offline diagnostic report for C3 v4 alert emission lead times.

This does NOT estimate accuracy/sensitivity and does NOT validate precursor labels.
Run from c3_aggression_detection root:
    python -m src.evaluate_v4_fight_alerts
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_IN = Path('outputs/v4_evaluation/fight_candidates_v4.csv')
DEFAULT_DIR = Path('outputs/v4_evaluation')


def onset_from_row(row):
    for key in ('fight_onset_s', 'v4_fight_onset_s'):
        value = str(row.get(key, '') or '').strip()
        if value:
            return float(value)
    values = json.loads(row.get('intervals') or '[]')
    if not values:
        raise ValueError('Missing annotated onset/intervals')
    return min(float(pair[0]) for pair in values)


def parse_alerts(row):
    value = json.loads(row.get('v4_alerts') or '[]')
    if not isinstance(value, list):
        raise ValueError('v4_alerts is not a JSON array')
    alerts = []
    for alert in value:
        if not isinstance(alert, dict) or 'alert_s' not in alert:
            raise ValueError('An alert lacks alert_s')
        t = float(alert['alert_s'])
        pair = alert.get('pair_ids') or []
        pair = tuple(sorted(map(str, pair))) if isinstance(pair, list) else ()
        alerts.append({'t': t, 'pair': pair,
                       'evidence_start_s': alert.get('evidence_start_s')})
    return sorted(alerts, key=lambda a: a['t'])


def classify_first_lead(lead):
    if lead is None:
        return 'no_alert'
    if lead > 3:
        return 'more_than_3s_early'
    if lead >= 1:
        return '1_to_3s_early'
    if lead >= 0:
        return 'less_than_1s_early'
    return 'post_onset'


def densest_one_second(alerts):
    """Max number of alert emissions and distinct pairs in an inclusive 1s window."""
    max_count, max_pairs = 0, 0
    for i, alert in enumerate(alerts):
        within = [a for a in alerts[i:] if a['t'] <= alert['t'] + 1.0 + 1e-9]
        max_count = max(max_count, len(within))
        max_pairs = max(max_pairs, len({a['pair'] for a in within if a['pair']}))
    return max_count, max_pairs


def evaluate_row(row):
    onset = onset_from_row(row)
    alerts = parse_alerts(row)
    first = alerts[0]['t'] if alerts else None
    lead = onset - first if first is not None else None
    target_alerts = [a for a in alerts if 1.0 <= onset - a['t'] <= 3.0]
    first_target = target_alerts[0]['t'] if target_alerts else None
    max_burst, burst_pairs = densest_one_second(alerts)
    emission_before_evidence = 0
    for a in alerts:
        if a['evidence_start_s'] not in (None, ''):
            if a['t'] + 1e-8 < float(a['evidence_start_s']):
                emission_before_evidence += 1
    video = str(row.get('video', '')).replace('\\', '/').split('/')[-1]
    return {
        'video': video,
        'fight_onset_s': round(onset, 6),
        'v4_status': row.get('v4_status', ''),
        'alert_count': len(alerts),
        'first_alert_s': '' if first is None else round(first, 6),
        'first_alert_lead_s': '' if lead is None else round(lead, 6),
        'first_alert_category': classify_first_lead(lead),
        'any_alert_in_1to3s': bool(target_alerts),
        'target_alert_count': len(target_alerts),
        'first_target_alert_s': '' if first_target is None else round(first_target, 6),
        'max_alerts_in_1s': max_burst,
        'max_distinct_pairs_in_1s': burst_pairs,
        'total_distinct_pairs': len({a['pair'] for a in alerts if a['pair']}),
        'emission_before_evidence_count': emission_before_evidence,
        'v4_error': row.get('v4_error', ''),
    }


def summarize(evaluated):
    categories = Counter(row['first_alert_category'] for row in evaluated)
    return {
        'videos': len(evaluated),
        'statuses': dict(sorted(Counter(row['v4_status'] for row in evaluated).items())),
        'videos_with_any_alert': sum(r['alert_count'] > 0 for r in evaluated),
        'first_alert_in_1to3s': categories.get('1_to_3s_early', 0),
        'any_alert_in_1to3s': sum(r['any_alert_in_1to3s'] for r in evaluated),
        'first_alert_more_than_3s_early': categories.get('more_than_3s_early', 0),
        'first_alert_less_than_1s_early': categories.get('less_than_1s_early', 0),
        'first_alert_post_onset': categories.get('post_onset', 0),
        'no_alert': categories.get('no_alert', 0),
        'total_alert_emissions': sum(r['alert_count'] for r in evaluated),
        'videos_with_5plus_alerts_in_1s': sum(r['max_alerts_in_1s'] >= 5 for r in evaluated),
        'videos_with_4plus_distinct_pairs_in_1s': sum(r['max_distinct_pairs_in_1s'] >= 4 for r in evaluated),
        'emission_before_evidence_count': sum(r['emission_before_evidence_count'] for r in evaluated),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DEFAULT_IN))
    parser.add_argument('--out-dir', default=str(DEFAULT_DIR))
    args = parser.parse_args(argv)
    with Path(args.input).open(newline='', encoding='utf-8-sig') as f:
        source = list(csv.DictReader(f))
    results = []
    problems = []
    for i, row in enumerate(source, 2):
        try:
            results.append(evaluate_row(row))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            problems.append(f'CSV line {i}: {exc}')
    if problems:
        print('ERROR: Could not evaluate all rows. No reports written:')
        for issue in problems[:10]:
            print(' -', issue)
        raise SystemExit(2)
    if not results:
        raise SystemExit('ERROR: CSV has no data rows.')

    out_dir = Path(args.out_dir)
    write_csv(out_dir / 'fight_alert_lead_times.csv', results, list(results[0]))
    shortlist = []
    for row in results:
        if row['max_distinct_pairs_in_1s'] >= 4 or row['max_alerts_in_1s'] >= 5:
            why = 'multiple_near_simultaneous_pair_alerts_audit'
        elif row['any_alert_in_1to3s']:
            why = 'candidate_1to3s_pre_onset_alert'
        elif row['video'] == 'F_66_1_2_0_0.mp4':
            why = 'F66_late_alert_diagnostic'
        elif row['v4_status'] in ('no_proposal', 'unobservable'):
            why = 'possible_missed_early_cue_audit'
        else:
            continue
        shortlist.append(dict(row, review_reason=why,
                              review_label='unreviewed'))
    priority = {'multiple_near_simultaneous_pair_alerts_audit': 0,
                'candidate_1to3s_pre_onset_alert': 1,
                'F66_late_alert_diagnostic': 2,
                'possible_missed_early_cue_audit': 3}
    shortlist.sort(key=lambda r: (priority[r['review_reason']], r['video']))
    shortlist = shortlist[:30]
    write_csv(out_dir / 'fight_alert_review_shortlist.csv', shortlist,
              list(results[0]) + ['review_reason', 'review_label'])
    stats = summarize(results)
    with (out_dir / 'fight_alert_summary.json').open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print('FIGHT CANDIDATE v4 ALERT EMISSION REPORT')
    for key, value in stats.items():
        print(f'{key}: {value}')
    print('Saved:', out_dir / 'fight_alert_lead_times.csv')
    print('Saved:', out_dir / 'fight_alert_review_shortlist.csv')
    print('Saved:', out_dir / 'fight_alert_summary.json')
    print('CAUTION: time alignment to annotated fight onset; not verified aggression or model sensitivity.')
    return stats


if __name__ == '__main__':
    main()
