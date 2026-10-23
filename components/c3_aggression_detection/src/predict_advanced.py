"""Causal cached replay or bounded new-video extraction with quality abstention."""
import argparse
from bisect import bisect_right
from collections import deque
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from src.advanced_experiment import SCHEMA, build_model, forward as model_forward
from src.foundation_io import sha256, video_id, write_json, write_csv
from src.lock_splits import load_locked
from src.pose_sequences import extract_video, load_cache
from src.pose_dataset import sequence_window, collate_sequences
from src.incident_events import build_incidents, utc_timestamp


class CausalDecision:
    def __init__(self, thresholds, alpha=.5, cooldown_s=5.):
        if len(thresholds) != 2 or not np.isfinite(thresholds).all() or not 0 < alpha <= 1 or cooldown_s < 0:
            raise ValueError('Two finite thresholds, positive smoothing and nonnegative cooldown required')
        self.thresholds = thresholds; self.alpha = alpha
        self.state = None; self.history = deque(maxlen=3); self.last_segment = None
        self.last_time = None; self.last_alert = -math.inf; self.previous = 'WARMUP'
        self.cooldown_s = cooldown_s; self.last_by_kind = {}; self.pending = None; self.audit = {}
        self.last_identity = None

    def update(self, t, scores, valid, segment, identity=None):
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Decision timestamps must increase')
        scene_changed = self.last_segment is not None and self.last_segment != segment
        identity = tuple(sorted(identity)) if identity is not None else None
        identity_changed = self.last_identity is not None and identity != self.last_identity
        reset = self.last_segment != segment or identity_changed or (self.last_time is not None and t-self.last_time > 1.)
        self.last_segment = segment; self.last_time = t
        self.last_identity = identity
        if scene_changed:
            self.last_by_kind.clear()
        if reset or not valid:
            self.state = None; self.history.clear(); self.previous = 'WARMUP'; self.pending = None
        self.audit = dict(notification_candidate=False, notification_suppression_reason='low_pose_quality', identity_reset=identity_changed)
        if not valid: return 'ABSTAIN_LOW_POSE_QUALITY', None, False
        values = np.asarray(scores, float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError('Two finite research scores required')
        self.state = values if self.state is None else self.alpha*values+(1-self.alpha)*self.state
        self.history.append(self.state >= self.thresholds)
        hits = np.asarray(self.history).sum(0)
        confirmed = hits >= 2 if len(self.history) == 3 else np.zeros(2, bool)
        status = 'WEAK_ACTIVE_FIGHT_ALERT' if confirmed[1] else ('WEAK_ONSET_WITHIN_3S_ALERT' if confirmed[0] else 'BELOW_SELECTED_THRESHOLDS')
        candidate = status.endswith('_ALERT')
        if candidate and status != self.previous:
            self.pending = status
        elif not candidate:
            self.pending = None
        ready = candidate and self.pending == status
        notification_key = (status, identity)
        alert = ready and t-self.last_by_kind.get(notification_key, -math.inf)+1e-9 >= self.cooldown_s
        reason = ('cooldown_same_kind' if ready else 'already_notified_episode') if candidate else 'below_selected_thresholds'
        if alert:
            self.last_alert = t; self.last_by_kind[notification_key] = t; self.pending = None; reason = None
        self.audit = dict(notification_candidate=bool(ready), notification_suppression_reason=reason, identity_reset=identity_changed)
        self.previous = status
        return status, self.state.copy(), bool(alert)


def pose_evidence(arrays, sample, index):
    """Describe selected currently visible tracks using only the supplied prefix."""
    selected = sample['selected_ids']
    begin, end = arrays['frame_offsets'][index:index+2]
    current = [j for j in range(begin, end) if arrays['track_ids'][j] in selected and arrays['observed'][j].any()]
    ids = [int(arrays['track_ids'][j]) for j in current]
    boxes = arrays['boxes'][current]
    bbox = ([int(math.floor(boxes[:, 0].min())), int(math.floor(boxes[:, 1].min())),
             int(math.ceil(boxes[:, 2].max())), int(math.ceil(boxes[:, 3].max()))] if len(boxes) else None)
    signals = [dict(signal='observed_pose_joints', value=int(sample['observed_mask'][-1].sum()), units='joint_count'),
               dict(signal='window_track_quality', value=float(sample['quality']), units='extractor_quality_index')]
    observed = sample['observed_mask']; sk = sample['sk']; times = sample['frame_times']
    # Never mix imputed joints into motion evidence or divide across missing times.
    if sample['temporal_mask'][-2:].all():
        dt = float(times[-1]-times[-2])
        valid = observed[-1] & observed[-2]
        wrist = valid[[9, 10]]
        if dt > 0 and wrist.any():
            speed = np.linalg.norm(sk[-1, [9, 10], :, :2]-sk[-2, [9, 10], :, :2], axis=-1)/dt
            signals.append(dict(signal='observed_wrist_motion', value=float(speed[wrist].max()), units='frame_normalized_distance_per_second'))
        pair_centers = []
        for pos in (-2, -1):
            centers = []
            for person in range(2):
                mask = observed[pos, :, person]
                if mask.sum() < 6: break
                centers.append(sk[pos, mask, person, :2].mean(0))
            if len(centers) == 2: pair_centers.append(float(np.linalg.norm(centers[0]-centers[1])))
        if len(pair_centers) == 2 and dt > 0:
            signals.extend([dict(signal='observed_person_center_distance', value=pair_centers[1], units='frame_normalized_distance'),
                dict(signal='observed_closing_speed', value=(pair_centers[0]-pair_centers[1])/dt, units='frame_normalized_distance_per_second')])
    return dict(track_ids_json=json.dumps(ids), bounding_box_json=json.dumps(bbox),
                evidence_signals_json=json.dumps(signals, allow_nan=False), source_frame_number=int(arrays['frame_index'][index]))


def at_or_before(records, t):
    position = bisect_right([r['t_s'] for r in records], t+1e-8)-1
    return records[position] if position >= 0 else None


def score_cache(checkpoint, cache, step_s=.2, research_alerts=False, expected_source=None):
    if step_s <= 0: raise ValueError('Positive decision step required')
    torch.set_num_threads(2)
    ck = torch.load(checkpoint, weights_only=True, map_location='cpu')
    if ck.get('schema') != SCHEMA or ck.get('verified_early_aggression_claim') is not False:
        raise ValueError('Expected an explicitly weak-supervised research checkpoint')
    arrays, meta = load_cache(cache)
    if expected_source is not None and (meta['video_id'] != expected_source['video_id'] or
            meta['source_sha256'] != expected_source['source_sha256']):
        raise ValueError('Cache source identity differs from the expected locked/development source')
    cfg = {k: meta.get(k) for k in ck['feature_contract']}
    if cfg != ck['feature_contract']:
        raise ValueError('Pose cache preprocessing differs from checkpoint')
    model = build_model(ck['variant']).eval(); model.load_state_dict(ck['state'], strict=True)
    return score_arrays(model, ck, arrays, meta, step_s, research_alerts)


def score_arrays(model, ck, arrays, meta, step_s=.2, research_alerts=False):
    """Shared causal replay; callers must separately authenticate cache/experiment provenance."""
    enabled = ck.get('research_alert_heads_enabled', [False, False])
    decision = CausalDecision([t if good else 1.000001 for t, good in zip(ck['thresholds'], enabled)])
    records = []; alerts = []; previous = -math.inf
    started = time.monotonic()
    with torch.no_grad():
        for i, t in enumerate(arrays['t']):
            if t-previous < step_s-1e-8: continue
            previous = t
            sample = sequence_window(arrays, i)
            # Historical track quality alone cannot establish a current observation.
            current_joints = int(sample['observed_mask'][-1].sum(0).max())
            valid = bool(sample['temporal_mask'].sum() >= 8 and sample['quality'] >= .45 and current_joints >= 6)
            scores = None
            if valid:
                batch = collate_sequences([dict(sample, targets=np.zeros(2, np.float32),
                    target_mask=np.zeros(2, bool), video_id=meta['video_id'], split='inference', leakage_group='inference')])
                scores = model_forward(model, batch, ck['variant'])[0].sigmoid().tolist()
            evidence = pose_evidence(arrays, sample, i)
            status, smoothed, alert = decision.update(float(t), scores, valid, int(arrays['segment_id'][i]),
                identity=json.loads(evidence['track_ids_json']))
            candidate_status = status
            if valid and not research_alerts:
                status = 'RESEARCH_SCORES_ONLY'; alert = False
            row = dict(t_s=float(t), quality=float(sample['quality']), valid_pose=valid,
                observed_history_frames=int(sample['temporal_mask'].sum()),
                current_observed_joints=current_joints,
                weak_onset_score=scores[0] if scores else None, fight_score=scores[1] if scores else None,
                smoothed_onset_score=float(smoothed[0]) if smoothed is not None else None,
                smoothed_fight_score=float(smoothed[1]) if smoothed is not None else None,
                status=status, candidate_status=candidate_status, alert_emitted=alert, segment_id=int(arrays['segment_id'][i]),
                **decision.audit, **evidence)
            if valid and not research_alerts and candidate_status.endswith('_ALERT'):
                row['notification_suppression_reason'] = 'research_notifications_not_requested'
            records.append(row)
            if alert: alerts.append(row)
    timing = dict(model_replay_seconds=round(time.monotonic()-started, 6),
                  decisions=len(records), valid_decisions=sum(r['valid_pose'] for r in records),
                  pose_extraction_included=False)
    return records, alerts, meta, ck, timing


def render_geometry(width, height):
    if width <= 0 or height <= 0: raise ValueError('Positive source image dimensions required')
    scale = min(960/max(1, width), max(1., 640/max(1, width)))
    w = round(width*scale)//2*2; h = round(height*scale)//2*2
    if min(w, h) < 2: raise ValueError('Rendered image must have nonzero dimensions')
    return dict(source_width=int(width), source_height=int(height), rendered_width=w,
        content_height=h, rendered_height=h+116, score_footer_height=116)


def render(records, meta, output, pose_arrays=None):
    import cv2
    source = Path(meta['source_video'])
    if sha256(source) != meta['source_sha256']:
        raise ValueError('Source recording changed after cache extraction')
    cap = cv2.VideoCapture(str(source)); fps = cap.get(cv2.CAP_PROP_FPS)
    first = round(meta['extracted_start_s']*fps); last = round(meta['extracted_end_s']*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geometry = render_geometry(width, height)
    w, h = geometry['rendered_width'], geometry['content_height']
    work = Path(output)/'render_work.mp4'
    writer = cv2.VideoWriter(str(work), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h+116))
    if not cap.isOpened() or not writer.isOpened(): raise RuntimeError('Cannot create demonstration video')
    try:
        for index in range(first, last):
            ok, frame = cap.read()
            if not ok: raise RuntimeError('Source decode failed during rendering')
            canvas = np.zeros((h+116, w, 3), np.uint8)
            canvas[:h] = cv2.resize(frame, (w, h))
            t = index/fps; record = at_or_before(records, t)
            if pose_arrays is not None:
                pos = int(np.searchsorted(pose_arrays['t'], t+1e-8, side='right'))-1
                if pos >= 0 and t-pose_arrays['t'][pos] < .2:
                    from src.occlusion import draw_pose_evidence
                    drawing=dict(pose_arrays,_source_width=width,_source_height=height)
                    draw_pose_evidence(canvas,drawing,pos,w,h)
            status = record['status'] if record else 'WARMUP'
            pre = record['smoothed_onset_score'] if record else None
            fight = record['smoothed_fight_score'] if record else None
            values = 'No usable pose score' if pre is None else f'Weak onset <=3s: {pre:.3f} | Active fight: {fight:.3f}'
            color = (60,160,255) if 'ALERT' in status else (220,220,220)
            lines = [(f'Original time {t:.2f}s | {status}', color), (values, (220,220,220)),
                     ('Green: observed | Orange hollow: estimated | Missing: not plotted', (80,210,240)),
                     ('Not validated early-aggression ground truth', (80,210,240))]
            for row, (text, col) in enumerate(lines):
                cv2.putText(canvas, text, (8, h+22+row*27), cv2.FONT_HERSHEY_SIMPLEX, .49, col, 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        cap.release(); writer.release()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg: raise FileNotFoundError('Local FFmpeg needed for browser-compatible demonstration')
    target = Path(output)/'demo.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(work),
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(target)], check=True, timeout=120, capture_output=True)
    return target


def run(args):
    component_root = Path(__file__).resolve().parents[1]
    output = Path(args.out_dir).resolve()
    if not output.is_relative_to(component_root):
        raise ValueError('Prediction output must stay inside Component 03')
    if not math.isfinite(args.start) or not math.isfinite(args.seconds) or args.start < 0 or args.seconds <= 0:
        raise ValueError('Finite positive duration and nonnegative start required')
    processed_at = utc_timestamp()
    camera_id = getattr(args, 'camera_id', 'c3_demo')
    recording_start = getattr(args, 'recording_start', None)
    if recording_start: utc_timestamp(recording_start)
    if not args.checkpoint:
        args.checkpoint = json.loads(Path('outputs/advanced_v1/deployment_selection.json').read_text())['checkpoint']
    output.mkdir(parents=True, exist_ok=False)
    extraction_seconds = 0.
    lookup_seconds = 0.
    cache_reused = True
    if args.video:
        source = Path(args.video).resolve()
        row = dict(video=str(source), video_id=video_id(source), content_sha256=sha256(source))
        # Metadata-only holdout check also rejects renamed or uploaded exact duplicates.
        if any(r['split'] == 'test' and (r['video_id'] == row['video_id'] or r['content_sha256'] == row['content_sha256'])
               for r in load_locked(args.locked)):
            raise ValueError('Locked test sources are excluded from demonstration/model development')
        started = time.monotonic()
        cache_folder = Path('data/pose_cache/inference')/row['video_id']
        existing = {p.resolve() for p in cache_folder.glob('*.npz')}
        cache, _ = extract_video(row, cache_folder,
                                start=args.start, end=args.start+args.seconds)
        elapsed = time.monotonic()-started
        cache_reused = cache.resolve() in existing
        extraction_seconds = elapsed if not cache_reused else 0.
        lookup_seconds = elapsed if cache_reused else 0.
    else:
        rows = {r['video_id']: r for r in load_locked(args.locked)}
        index = json.loads(Path(args.index).read_text())
        if index['locked_manifest_sha256'] != sha256(args.locked): raise ValueError('Index/split mismatch')
        key = video_id(args.video_id)
        if rows[key]['split'] == 'test':
            raise ValueError('Locked test sources are excluded from demonstration/model development')
        entry = index['entries'][key]
        if sha256(entry['path']) != entry['cache_sha256'] or entry['source_sha256'] != rows[key]['content_sha256']:
            raise ValueError('Cache/source mismatch')
        cache = entry['path']
    expected_source = (dict(video_id=row['video_id'], source_sha256=row['content_sha256']) if args.video else
        dict(video_id=key, source_sha256=rows[key]['content_sha256']))
    records, alerts, meta, ck, timing = score_cache(args.checkpoint, cache,
        research_alerts=args.research_alerts, expected_source=expected_source)
    if not args.video and (meta['video_id'] != key or meta['source_sha256'] != rows[key]['content_sha256']):
        raise ValueError('Prediction cache metadata differs from the locked source')
    write_csv(output/'scores.csv', records)
    write_json(output/'alerts.json', dict(alerts=alerts, supervision=ck['supervision'],
        verified_early_aggression_claim=False, threshold_source='train-only internal development'))
    report = dict(video_id=meta['video_id'], source_sha256=meta['source_sha256'],
        checkpoint_sha256=sha256(args.checkpoint), variant=ck['variant'],
        thresholds=ck['thresholds'], supervision=ck['supervision'],
        research_alert_heads_enabled=ck.get('research_alert_heads_enabled', [False, False]),
        research_alerts_requested=args.research_alerts, operational_alerts_enabled=False,
        verified_early_aggression_claim=False, probability_calibration='UNVALIDATED',
        timing=timing, extraction_seconds=round(extraction_seconds, 6),
        cache_lookup_seconds=round(lookup_seconds, 6), pose_cache_reused=cache_reused,
        runtime_quality_policy=dict(version='current_observation_guard_with_identity_reset_v3',min_history_frames=8,
            min_window_track_quality=.45,min_current_observed_joints_in_one_person=6),
        status_counts={status:sum(r['status'] == status for r in records) for status in sorted({r['status'] for r in records})},
        alerts=len(alerts), source_context=[meta['extracted_start_s'], meta['extracted_end_s']],
        camera_id=camera_id, processed_at=processed_at, source_video=meta['source_video'],
        frame_wh=meta['frame_wh'], source_fps=meta['source_fps'],
        score_semantics='uncalibrated sigmoid research scores, not probabilities',
        notification_policy=dict(version='per_kind_identity_pending_cooldown_v3', cooldown_s=5.,
            active_escalation_has_separate_cooldown=True, suppressed_candidates_retained=True),
        caveat='Cached causal replay; model timing excludes pose extraction. CPU end-to-end real-time performance is not claimed.')
    if args.render:
        arrays, _ = load_cache(cache)
        video = render(records, meta, output, arrays)
        report['demo_video'] = str(video); report['demo_sha256'] = sha256(video)
        report['media_geometry'] = render_geometry(*meta['frame_wh'])
    envelope, notifications = build_incidents(records, meta, ck, report['checkpoint_sha256'],
        camera_id=camera_id, processed_at=processed_at, recording_start=recording_start,
        annotated_video='demo.mp4' if args.render else None)
    write_json(output/'incidents.json', envelope)
    write_json(output/'notifications.json', dict(notifications=notifications, research_only=True, operational_alerts_enabled=False))
    report.update(incidents_file='incidents.json', notifications_file='notifications.json',
        incident_candidates=len(envelope['incidents']), research_notifications=len(notifications))
    write_json(output/'prediction.json', report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--video'); group.add_argument('--video-id')
    p.add_argument('--checkpoint', help='Default: internal-development guarded runtime selection')
    p.add_argument('--index', default='data/pose_cache/advanced_v1/val_index.json')
    p.add_argument('--locked', default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--render', action='store_true')
    p.add_argument('--camera-id', default='c3_demo')
    p.add_argument('--recording-start', help='Optional original recording start ISO-8601 with UTC offset; otherwise incident timestamps are processing time')
    p.add_argument('--research-alerts', action='store_true', help='Explicit opt-in; only internally guarded heads can alert. Not operational validation.')
    args = p.parse_args()
    if not np.isfinite([args.start, args.seconds]).all() or args.start < 0 or args.seconds <= 0:
        p.error('Finite positive duration/nonnegative start required')
    run(args)


if __name__ == '__main__': main()
