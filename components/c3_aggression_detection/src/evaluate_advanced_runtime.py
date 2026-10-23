"""Evaluate the final current-observation guard on locked validation contexts."""
import argparse
import json
from pathlib import Path

from src.foundation_io import sha256, write_json, write_csv
from src.predict_advanced import score_cache
from src.advanced_data import weak_targets
from src.advanced_experiment import metrics
from src.lock_splits import load_locked


def run(args):
    rows = {r['video_id']: r for r in load_locked(args.locked)}
    index = json.loads(Path(args.index).read_text())
    if index['locked_manifest_sha256'] != sha256(args.locked):
        raise ValueError('Runtime validation index differs from locked splits')
    if not args.checkpoint:
        args.checkpoint = json.loads(Path('outputs/advanced_v1/deployment_selection.json').read_text())['checkpoint']
    all_rows = []; coverage = []; events = []; negative_alerts = 0
    for key, entry in index['entries'].items():
        if key not in rows or rows[key]['split'] != 'val':
            raise ValueError('This diagnostic permits locked validation sources only')
        if sha256(entry['path']) != entry['cache_sha256']:
            raise ValueError('Runtime validation cache changed')
        records, alerts, meta, ck, _ = score_cache(args.checkpoint, entry['path'], research_alerts=True,
            expected_source=dict(video_id=key, source_sha256=rows[key]['content_sha256']))
        if meta['video_id'] != key or meta['source_sha256'] != rows[key]['content_sha256']:
            raise ValueError('Source/cache identity mismatch')
        intervals = json.loads(rows[key]['intervals'])
        onset = intervals[0][0] if intervals else None
        for record in records:
            y, mask = weak_targets(record['t_s'], intervals)
            all_rows.append(dict(record, video_id=key, weak_onset_target=int(y[0]),
                weak_onset_mask=bool(mask[0]), weak_fight_target=int(y[1]), weak_fight_mask=bool(mask[1])))
        coverage.append(dict(video_id=key, decisions=len(records), valid=sum(r['valid_pose'] for r in records), alerts=len(alerts)))
        if onset is not None and meta['extracted_start_s'] <= onset < meta['extracted_end_s']:
            before = [r for r in records if onset-3 <= r['t_s'] < onset]
            hits = [r for r in alerts if r['candidate_status'] == 'WEAK_ONSET_WITHIN_3S_ALERT' and onset-3 <= r['t_s'] < onset]
            events.append(dict(video_id=key, weak_first_onset_s=onset,
                usable_preonset_decisions=sum(r['valid_pose'] for r in before), detected=bool(hits),
                lead_s=onset-hits[0]['t_s'] if hits else None))
        elif not intervals:
            negative_alerts += len(alerts)
    head_metrics = {}
    for h, score_name, target, mask_name in ((0,'weak_onset_score','weak_onset_target','weak_onset_mask'),
                                           (1,'fight_score','weak_fight_target','weak_fight_mask')):
        usable = [r for r in all_rows if r['valid_pose'] and r[mask_name]]
        head_metrics[['weak_first_onset_within_3s','weak_active_fight'][h]] = metrics(
            [r[target] for r in usable], [r[score_name] for r in usable], ck['thresholds'][h])
    report = dict(schema='c3_runtime_quality_v3_diagnostic', checkpoint=args.checkpoint,
        notification_policy='per-kind/identity pending cooldown with identity-reset confirmation',
        current_observation_guard=6, decisions=len(all_rows), valid_decisions=sum(r['valid_pose'] for r in all_rows),
        usable_coverage=sum(r['valid_pose'] for r in all_rows)/len(all_rows), per_video=coverage,
        raw_weak_label_metrics_on_observable_windows=head_metrics, weak_event_contexts=events,
        weak_event_contexts_detected=sum(e['detected'] for e in events), weak_event_contexts_total=len(events),
        weak_negative_context_alerts=negative_alerts, test_sources_used=0,
        operational_alerts_enabled=False, verified_early_aggression_claim=False,
        limitations=['Small retrospectively selected weak-label contexts, not full recordings or verified precursor events.',
            'Unobservable event contexts remain counted as undetected; no-pose time is abstention, never a negative prediction.',
            'No false-alarms-per-hour, calibration or operational accuracy claim. Research alert eligibility comes only from internal development.'])
    output = Path(args.out_dir)
    write_json(output/'runtime_evaluation.json', report)
    write_csv(output/'runtime_validation_scores.csv', all_rows)
    print(json.dumps({k: report[k] for k in ('decisions','valid_decisions','usable_coverage',
        'weak_event_contexts_detected','weak_event_contexts_total','weak_negative_context_alerts')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--index', default='data/pose_cache/advanced_v1/val_index.json')
    p.add_argument('--checkpoint')
    p.add_argument('--out-dir', default='outputs/final_c3_v2')
    run(p.parse_args())
