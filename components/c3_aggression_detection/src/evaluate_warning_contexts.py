"""Separate weak onset, weak fight and blocked verified pre-contact evaluation.

Only locked validation pose caches are replayed. These selected short contexts
are not full recordings; no population FAR/hour or verified accuracy is inferred.
"""
import argparse
import json
from pathlib import Path

from src.advanced_data import weak_targets
from src.advanced_experiment import metrics
from src.foundation_io import sha256, write_json, write_csv
from src.lock_splits import load_locked
from src.predict_advanced import score_cache


def onset_context(onset, records, extracted_start, extracted_end):
    if not extracted_start <= onset < extracted_end:
        return None
    candidates = [r for r in records if r['valid_pose'] and
        r['candidate_status'] == 'WEAK_ONSET_WITHIN_3S_ALERT' and 0 < onset-r['t_s'] <= 3.]
    notified = [r for r in candidates if r['alert_emitted']]
    candidate_lead = max((onset-r['t_s'] for r in candidates), default=None)
    notification_lead = max((onset-r['t_s'] for r in notified), default=None)
    return dict(weak_annotated_first_onset_s=onset, onset_is_verified_contact=False,
        full_three_second_preonset_context=extracted_start <= onset-3.,
        usable_preonset_decisions=sum(r['valid_pose'] for r in records if onset-3 <= r['t_s'] < onset),
        candidate_lead_s=candidate_lead, notified_lead_s=notification_lead,
        candidate_to_notification_lead_difference_s=candidate_lead-notification_lead
            if candidate_lead is not None and notification_lead is not None else None,
        candidate_detected=bool(candidates), notification_detected=bool(notified),
        candidate_recall_at_lead={str(h): any(onset-r['t_s']+1e-8 >= h for r in candidates) for h in (1,2,3)},
        notification_recall_at_lead={str(h): any(onset-r['t_s']+1e-8 >= h for r in notified) for h in (1,2,3)},
        candidate_without_notification=bool(candidates and not notified))


def summarize_contexts(events):
    n = len(events)
    return dict(contexts=n, detected_candidates=sum(e['candidate_detected'] for e in events),
        detected_notifications=sum(e['notification_detected'] for e in events),
        candidate_without_notification=sum(e['candidate_without_notification'] for e in events),
        candidate_recall_at_lead={str(h): sum(e['candidate_recall_at_lead'][str(h)] for e in events)/n if n else None for h in (1,2,3)},
        notification_recall_at_lead={str(h): sum(e['notification_recall_at_lead'][str(h)] for e in events)/n if n else None for h in (1,2,3)},
        full_three_second_contexts=sum(e['full_three_second_preonset_context'] for e in events))


def evaluate(args):
    rows = {r['video_id']: r for r in load_locked(args.locked)}
    index = json.loads(Path(args.index).read_text())
    if index['locked_manifest_sha256'] != sha256(args.locked):
        raise ValueError('Index does not match locked splits')
    # Validate the entire requested cohort before any model evaluation.
    if any(key not in rows or rows[key]['split'] != 'val' for key in index['entries']):
        raise ValueError('Only locked validation caches are permitted; test sources are never evaluated here')
    checkpoint = args.checkpoint or json.loads(Path('outputs/advanced_v1/deployment_selection.json').read_text())['checkpoint']
    event_rows = []; scored = []; no_fight_notifications = 0; coverage = []; ck = None
    for key, entry in sorted(index['entries'].items()):
        if sha256(entry['path']) != entry['cache_sha256']:
            raise ValueError('Pose cache changed after indexing')
        records, alerts, meta, ck, _ = score_cache(checkpoint, entry['path'], research_alerts=True,
            expected_source=dict(video_id=key, source_sha256=rows[key]['content_sha256']))
        if meta['video_id'] != key or meta['source_sha256'] != rows[key]['content_sha256']:
            raise ValueError('Cache source identity mismatch')
        intervals = json.loads(rows[key]['intervals'])
        for row in records:
            target, mask = weak_targets(row['t_s'], intervals)
            scored.append(dict(row, video_id=key, onset_target=int(target[0]), onset_mask=bool(mask[0]),
                active_target=int(target[1]), active_mask=bool(mask[1])))
        if intervals:
            event = onset_context(intervals[0][0], records, meta['extracted_start_s'], meta['extracted_end_s'])
            if event: event_rows.append(dict(video_id=key, **event))
        else:
            no_fight_notifications += len(alerts)
        coverage.append(dict(video_id=key, decisions=len(records), usable=sum(r['valid_pose'] for r in records)))
    head_reports = {}
    for h, name, score_name, target_name, mask_name in (
            (0, 'weak_first_onset_forecasting', 'weak_onset_score', 'onset_target', 'onset_mask'),
            (1, 'weak_active_fight_classification', 'fight_score', 'active_target', 'active_mask')):
        usable = [r for r in scored if r['valid_pose'] and r[mask_name]]
        head_reports[name] = dict(window_metrics=metrics([r[target_name] for r in usable],
            [r[score_name] for r in usable], ck['thresholds'][h] if ck else .5),
            research_alert_head_enabled=bool(ck and ck.get('research_alert_heads_enabled', [False,False])[h]),
            targets='Original weak UBI binary annotations, not verified cue/contact labels')
    annotations = json.loads(Path(args.annotations).read_text())
    if annotations.get('locked_manifest_sha256') != sha256(args.locked):
        raise ValueError('Annotation evidence does not match locked split version')
    positives = {r['video_id'] for r in annotations['records'] if r['review_status'] == 'independently_verified'
        and r.get('label') == 'precursor' and r.get('precursors') and not r.get('issues') and r.get('contact_s') is not None}
    total = len(scored); valid = sum(r['valid_pose'] for r in scored)
    report = dict(schema='c3_three_task_context_evaluation_v1', checkpoint=checkpoint,
        checkpoint_sha256=sha256(checkpoint), locked_manifest_sha256=sha256(args.locked),
        cache_index_sha256=sha256(args.index), annotation_evidence_sha256=sha256(args.annotations),
        selection_changed=False, test_sources_used=0, operational_alerts_enabled=False,
        verified_early_aggression_claim=False, tasks=head_reports,
        verified_pre_contact_aggression=dict(status='BLOCKED_PENDING_INDEPENDENT_VERIFIED_EVALUATION',
            preserved_verified_precursor_positive_videos=len(positives), accuracy=None,
            event_recall_at_1_2_3_seconds=None,
            reason='Explicit reviewed cue, observed first physical contact, independent agreement, source coverage and prospective evaluation are required. Weak onset is not a contact annotation.'),
        weak_onset_context_metrics=summarize_contexts(event_rows), events=event_rows,
        decisions=total, usable_decisions=valid, usable_fraction=valid/total if total else None,
        no_fight_context_research_notifications=no_fight_notifications,
        false_alarms_per_camera_hour=None, per_video=coverage,
        suppression=dict(policy='Per-kind/identity cooldown, identity-reset confirmation, pending candidates retained, independent active escalation',
            suppressed_decisions=sum(r['notification_suppression_reason'] == 'cooldown_same_kind' for r in scored),
            identity_reset_decisions=sum(r['identity_reset'] for r in scored),
            decision_reason_counts={str(reason): sum(r['notification_suppression_reason'] == reason for r in scored)
                for reason in sorted({r['notification_suppression_reason'] for r in scored}, key=str)},
            note='Candidate and notification recall are both reported; suppression can consume useful lead time.'),
        limitations=['Selected short weak-label contexts; overlapping windows are not independent events.',
            'Unobservable event contexts remain in the denominator as undetected.',
            '1/2/3-second metrics refer to weak annotation times, never independently verified physical contact.',
            'External validation has prior exploratory exposure; no tuning or test evaluation occurs here.',
            'No full-recording exposure, population false-alarm, calibrated probability or real-time claim.'])
    output = Path(args.out_dir)
    write_json(output/'evaluation.json', report)
    write_csv(output/'context_scores.csv', scored)
    print(json.dumps({k: report[k] for k in ('weak_onset_context_metrics','decisions','usable_decisions','suppression')}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--index', default='data/pose_cache/advanced_v1/val_index.json')
    p.add_argument('--annotations', default='data/foundation/ai_visual_v1/annotations.json')
    p.add_argument('--checkpoint')
    p.add_argument('--out-dir', required=True)
    evaluate(p.parse_args())
