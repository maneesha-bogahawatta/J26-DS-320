"""Normalize review evidence without changing originals or inventing supervision."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.foundation_io import read_csv, sha256, video_id, write_csv, write_json
from src.lock_splits import load_locked


def intervals(value, duration):
    value = json.loads(value or '[]') if isinstance(value, str) else value or []
    if not isinstance(value, list):
        raise ValueError('Intervals must be a list')
    out = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('Expected [start,end] interval')
        a,b = map(float, pair)
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration+.001):
            raise ValueError('Interval outside original-video duration')
        out.append([a,b])
    return out


def covered_seconds(ranges):
    total = 0.
    end = -1.
    for a,b in sorted(ranges):
        total += max(0., b-max(a,end))
        end = max(end,b)
    return total


def normalize(original, base, location, kind):
    r = original
    duration = float(base['duration_s'])
    original_label = str(r.get('review_label') or r.get('review_decision') or
                         r.get('decision') or r.get('annotation_status') or 'unreviewed').strip().lower()
    aliases = {'verified_precursor':'precursor', 'precursor_verified':'precursor',
               'potential_missed_precursor':'potential_precursor', 'negative_verified':'non_aggressive'}
    label = aliases.get(original_label, original_label)
    if label in ('verified','approve'):
        label = 'approved_annotation'
    reviewers = [str(r.get(a) or r.get(b) or '').strip() for a,b in
                 (('reviewer_a','reviewer_1'), ('reviewer_b','reviewer_2'))]
    if not any(reviewers) and r.get('reviewer'):
        reviewers[0] = r['reviewer'].strip()
    distinct = bool(all(reviewers) and reviewers[0].casefold() != reviewers[1].casefold())
    issues = []
    coverage, cues, fight = [], [], []
    contact = None
    try:
        if kind == 'manifest':
            coverage = [[0.,duration]]
        elif kind == 'v4':
            coverage = intervals([[float(r['clip_start_s']), float(r['clip_end_s'])]], duration)
        elif kind == 'control':
            coverage = intervals([[float(r['window_start_s']), float(r['window_end_s'])]], duration)
        elif r.get('reviewed_ranges_s'):
            coverage = intervals(r['reviewed_ranges_s'], duration)
        cues = intervals(r.get('verified_precursors') or r.get('precursors'), duration)
        if label == 'approved_annotation':
            label = 'precursor' if cues else ('active_fight' if json.loads(r.get('intervals') or '[]') else 'non_aggressive')
        if r.get('precursor_start_s') not in ('', None) or r.get('precursor_end_s') not in ('', None):
            cues = intervals([[float(r['precursor_start_s']),float(r['precursor_end_s'])]], duration)
        if r.get('first_physical_contact_s') not in ('', None):
            contact = float(r['first_physical_contact_s'])
        # Only explicit approved annotations can provide verified contact/fight bounds.
        explicit = original_label == 'approve' or r.get('annotation_status') in ('verified','negative_verified')
        if kind == 'manifest' and explicit:
            fight = intervals(r.get('onset_override_s') or r.get('intervals'), duration)
            contact = min((a for a,b in fight), default=contact)
        elif r.get('verified_fight_intervals'):
            fight = intervals(r['verified_fight_intervals'], duration)
        if contact is not None and (not math.isfinite(contact) or not 0 <= contact <= duration):
            raise ValueError('Contact time outside original video')
        if cues and (contact is None or any(b > contact for a,b in cues)):
            raise ValueError('Precursors need a confirmed later contact time')
        if any(not any(x <= a and b <= y for x,y in coverage) for a,b in cues):
            raise ValueError('Precursors outside reviewed coverage')
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        issues.append(str(exc))
    if kind == 'manifest' and not distinct:
        status = 'weak'
    elif label in ('unreviewed',''):
        status = 'unreviewed'
    elif distinct and not issues and label in ('precursor','non_aggressive','active_fight','already_violent'):
        status = 'independently_verified'
    else:
        status = 'provisional_single_review'
    if status == 'independently_verified':
        if not coverage or (label == 'precursor' and (not cues or contact is None)):
            issues.append('Verified precursor requires reviewed bounds, explicit cues and confirmed contact')
            status = 'provisional_single_review'
        if label in ('active_fight','already_violent') and not fight:
            issues.append('Verified active-fight training requires explicit fight intervals')
            status = 'provisional_single_review'
    return {'record_id':location, 'video_id':base['video_id'], 'split':base['split'],
            'label':label, 'original_label':original_label, 'review_status':status,
            'reviewers':reviewers, 'coverage_s':coverage, 'precursors':cues,
            'contact_s':contact, 'fight_intervals':fight, 'issues':issues,
            'timestamp_basis':'original_video_seconds', 'original':dict(original),
            'usage':'held_out_evaluation_only' if base['split']=='test' else base['split'],
            'eligible_for_fitting':status=='independently_verified' and base['split']=='train'}


def reconcile(locked, sources):
    by_id = {r['video_id']:r for r in locked}
    records = [normalize(r,r,f'locked_manifest:{i+2}','manifest') for i,r in enumerate(locked)]
    unmatched = []
    for kind,path,rows in sources:
        for i,row in enumerate(rows):
            key = video_id(row.get('source_video') or row.get('video',''))
            if key not in by_id:
                unmatched.append({'source':str(path), 'row':i+1, 'video':key})
                continue
            records.append(normalize(row,by_id[key],f'{path}:{i+1}',kind))
    if unmatched:
        raise ValueError(f'Reviews reference unknown videos: {unmatched}')
    return records


def coverage_report(records):
    groups = defaultdict(list)
    for r in records:
        if r['review_status'] != 'unreviewed':
            groups[(r['split'],r['review_status'],r['video_id'])].extend(r['coverage_s'])
    coverage = defaultdict(float)
    for (split,status,key), bounds in groups.items():
        coverage[f'{split}/{status}'] += covered_seconds(bounds)
    return {'records':len(records), 'review_status_counts':dict(Counter(r['review_status'] for r in records)),
            'labels_by_status':{status:dict(Counter(r['label'] for r in records if r['review_status']==status))
                                for status in sorted({r['review_status'] for r in records})},
            'records_by_split':{s:dict(Counter(r['review_status'] for r in records if r['split']==s)) for s in ('train','val','test')},
            'coverage_seconds_by_split_and_status':dict(coverage),
            'verified_precursor_videos_by_split':{s:len({r['video_id'] for r in records if r['split']==s and
                r['review_status']=='independently_verified' and r['label']=='precursor' and r['precursors']})
                for s in ('train','val','test')},
            'records_with_issues':[{'record_id':r['record_id'],'issues':r['issues']} for r in records if r['issues']],
            'notes':['Review record counts overlap across artifacts; they are not independent examples.',
                     'Coverage is scoped to explicit original-video bounds. Uncertain/unreviewed regions provide no supervised targets.',
                     'Imported fight labels and heuristic proposals do not establish verified precursors.']}


def run(args):
    locked = load_locked(args.locked)
    sources = [('manual_json',args.manual_json,json.loads(Path(args.manual_json).read_text(encoding='utf-8'))),
               ('v4',args.review_batch,read_csv(args.review_batch)),
               ('control',args.controls,read_csv(args.controls))]
    records = reconcile(locked, sources)
    report = coverage_report(records)
    out = Path(args.out_dir)
    write_json(out/'annotations.json', {'schema':'c3_review_evidence_v1',
        'locked_manifest_sha256':sha256(args.locked),
        'source_sha256':{str(path):sha256(path) for kind,path,rr in sources}, 'records':records})
    # Selection for additional TRAINING review cannot include validation or test videos.
    queue = [{'video':r['video'], 'video_id':r['video_id'], 'split':r['split'],
              'leakage_group':r['leakage_group'], 'review_record_ids':json.dumps(
                  [x['record_id'] for x in records if x['video_id']==r['video_id'] and not x['record_id'].startswith('locked_manifest:')])}
             for r in locked if r['split']=='train' and json.loads(r.get('intervals') or '[]')]
    write_csv(out/'training_review_queue.csv', queue,
              ['video','video_id','split','leakage_group','review_record_ids'])
    write_json(Path(args.report_dir)/'annotation_coverage.json', report)
    print(json.dumps(report,indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--manual-json', default='data/reviewed_annotations/ubi_manual_reviews.json')
    p.add_argument('--review-batch', default='outputs/v4_evaluation/training_review_batch/review_annotations.csv')
    p.add_argument('--controls', default='outputs/v3_evaluation/control_review_labels.csv')
    p.add_argument('--out-dir', default='data/foundation/v1')
    p.add_argument('--report-dir', default='outputs/phase1_foundation')
    run(p.parse_args())


if __name__ == '__main__':
    main()
