"""C3 v2: causal, explainable UBI-Fights precursor *proposals* (not labels).
Run at the c3_aggression_detection root: python -m src.auto_annotate_precursors_v2 --limit 5
Requires: ultralytics, opencv-python, numpy. Saves NEW v2 files only.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

INPUT = Path('data/manifests/ubi_early_review_candidates.csv')
OUTPUT = Path('data/manifests/ubi_auto_precursors_v2.csv')
REVIEW = Path('data/manifests/ubi_review_priority_v2.csv')


def load_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def parse_onset(row):
    if row.get('fight_onset_s') not in (None, ''):
        return float(row['fight_onset_s'])
    intervals = json.loads(row.get('intervals') or '[]')
    if not intervals:
        raise ValueError('no fight intervals')
    return float(intervals[0][0])


def persons_from(result):
    if result.boxes is None or result.boxes.id is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.int().cpu().numpy()
    coords = result.keypoints.xy.cpu().numpy()
    conf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
            else np.ones(coords.shape[:2], dtype=np.float32))
    people = []
    for i, pid in enumerate(ids):
        box = boxes[i]
        h = max(float(box[3] - box[1]), 20.)
        center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=np.float32)
        people.append({'id': int(pid), 'center': center, 'height': h,
                       'kp': coords[i], 'conf': conf[i]})
    return people


class TemporalCues:
    """Online state. No future frames; isolated per video; distances in body-heights."""
    def __init__(self, alpha=0.45, max_gap=0.8):
        self.alpha, self.max_gap = alpha, max_gap
        self.state = {}
        self.last_time = None

    def update(self, persons, t):
        cur = {}
        for p in persons:
            pid = p['id']
            prev = self.state.get(pid)
            center = p['center'].astype(float)
            if prev is not None and 0 < t-prev['t'] <= self.max_gap:
                center = self.alpha*center + (1-self.alpha)*prev['center']
            cur[pid] = {**p, 'center': center, 't': t}
        signals = set()
        speeds = []
        for pid, p in cur.items():
            old = self.state.get(pid)
            if old is not None and 0 < t-old['t'] <= self.max_gap:
                speed = np.linalg.norm(p['center'] - old['center']) / p['height'] / (t-old['t'])
                speeds.append(float(speed))
            # raised arm alone is ambiguous; require a dyadic context for proposals
            for shoulder, wrist in ((5, 9), (6, 10)):
                if p['conf'][shoulder] >= .4 and p['conf'][wrist] >= .4:
                    if (p['kp'][shoulder, 1] - p['kp'][wrist, 1])/p['height'] > .22:
                        signals.add('raised_arm')
        if speeds and max(speeds) > .8:
            signals.add('rapid_movement')
        dyadic = False
        pairs = list(cur.values())
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                a,b = pairs[i],pairs[j]
                olda,oldb = self.state.get(a['id']),self.state.get(b['id'])
                if olda is None or oldb is None: continue
                dt = t-max(olda['t'], oldb['t'])
                if dt <= 0 or dt > self.max_gap: continue
                height = max((a['height']+b['height'])/2,20)
                gap = np.linalg.norm(a['center']-b['center'])/height
                old_gap = np.linalg.norm(olda['center']-oldb['center'])/height
                closing = (old_gap-gap)/dt
                if closing > .25 and gap < 2.0:
                    dyadic = True
                    signals.add('rapid_approach')
        # Explicitly retain uncertainty rather than treating absence as a negative.
        score = .0
        if dyadic: score += .40
        if 'rapid_movement' in signals: score += .18
        if 'raised_arm' in signals: score += .17
        if dyadic and 'raised_arm' in signals: score += .15
        if dyadic and 'rapid_movement' in signals: score += .10
        score = min(score, 1.)
        self.state = cur  # no stale tracks linger across extended gaps
        self.last_time = t
        return {'timestamp': float(t), 'score': round(score,4), 'signals': sorted(signals),
                'tracked_people': len(cur), 'dyadic': dyadic}


def propose_intervals(obs, threshold=.42, min_steps=2, sample_fps=5., onset=None):
    """Consecutive sampled frames only, never extend past onset."""
    out, active = [], []
    def flush():
        if len(active) >= min_steps:
            start = active[0]['timestamp']
            end = active[-1]['timestamp'] + 1./sample_fps
            if onset is not None: end = min(end, onset)
            if end > start: out.append([round(start,3), round(end,3)])
        active.clear()
    last_t = None
    for o in obs:
        if last_t is not None and o['timestamp']-last_t > 1.6/sample_fps:
            flush()
        if o['score'] >= threshold and o['dyadic']:
            active.append(o)
        else:
            flush()
        last_t = o['timestamp']
    flush()
    return out


def analyze(path, onset, weights, sample_fps, pre_seconds, threshold, min_steps, device):
    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError('Video could not be opened')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: raise RuntimeError('Invalid FPS')
        begin = max(0., onset-pre_seconds)
        first = int(math.ceil(begin*fps))
        last = int(math.ceil(onset*fps))
        stride = max(1,round(fps/sample_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        model = YOLO(weights)  # resets ByteTrack per video; no cross-video ID leakage
        cues = TemporalCues()
        obs = []
        for idx in range(first,last):
            ok, frame = cap.read()
            if not ok: break
            if (idx-first) % stride: continue
            # Sequential reading, sampled inference. No seeking repeatedly.
            opts = dict(persist=True, tracker='bytetrack.yaml', classes=[0], conf=.25, verbose=False)
            if device: opts['device'] = device
            result = model.track(frame, **opts)[0]
            obs.append(cues.update(persons_from(result), idx/fps))
        actual_fps = fps/stride
        intervals = propose_intervals(obs,threshold,min_steps,actual_fps,onset)
        return obs, intervals
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default=str(INPUT))
    ap.add_argument('--weights', default='yolo11n-pose.pt')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sample-fps', type=float, default=5.)
    ap.add_argument('--pre-seconds', type=float, default=5.)
    ap.add_argument('--threshold', type=float, default=.42)
    ap.add_argument('--min-steps', type=int, default=2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--negative-audit', type=int, default=15)
    args = ap.parse_args()
    if args.sample_fps <= 0 or args.pre_seconds <= 0 or not 0 <= args.threshold <= 1:
        ap.error('sample-fps/pre-seconds must be positive; threshold must be in [0,1]')
    rows = load_csv(Path(args.manifest))
    if args.limit > 0: rows = rows[:args.limit]
    results = []
    for i, row in enumerate(rows, 1):
        rec = dict(row)
        rec.update(v2_proposals='[]', v2_max_score='', v2_signals='[]', v2_review_priority='none',
                   v2_status='unverified', v2_error='', v2_quality='', v2_method='causal_pose_heuristic_v2')
        try:
            path = Path(row['video'])
            onset = parse_onset(row)
            if not path.is_file(): raise FileNotFoundError(str(path))
            obs, intervals = analyze(path,onset,args.weights,args.sample_fps,args.pre_seconds,
                                     args.threshold,args.min_steps,args.device)
            rec['v2_proposals'] = json.dumps(intervals)
            rec['v2_max_score'] = max((x['score'] for x in obs), default=0.)
            rec['v2_signals'] = json.dumps(sorted({s for x in obs for s in x['signals']}))
            rec['v2_quality'] = round(sum(x['tracked_people']>=2 for x in obs)/len(obs),3) if obs else 0.
            rec['v2_status'] = 'proposal' if intervals else ('unobservable' if not obs else 'no_proposal')
            rec['v2_review_priority'] = 'high' if intervals else ('medium' if float(rec['v2_quality'])<.5 else 'audit')
        except Exception as exc:
            rec['v2_status'] = 'error'; rec['v2_error'] = str(exc)
            rec['v2_review_priority'] = 'high'
        results.append(rec)
        print(f'[{i}/{len(rows)}] {Path(row.get("video", "missing")).name}: {rec["v2_status"]}')
    fields = list(rows[0].keys()) + ['v2_proposals','v2_max_score','v2_signals','v2_review_priority',
                                     'v2_status','v2_error','v2_quality','v2_method'] if rows else []
    write_csv(OUTPUT, results, fields or ['v2_status'])
    priority = [r for r in results if r['v2_review_priority'] in ('high','medium')]
    pool = [r for r in results if r['v2_review_priority']=='audit']
    sample = random.Random(args.seed).sample(pool,min(args.negative_audit,len(pool)))
    write_csv(REVIEW, priority+sample,fields or ['v2_status'])
    print(f'Processed: {len(results)} | Proposals: {sum(r["v2_status"]=="proposal" for r in results)} '
          f'| Errors: {sum(r["v2_status"]=="error" for r in results)} '
          f'| Review queue: {len(priority)+len(sample)}')
    print('Proposals:', OUTPUT, '| Review:', REVIEW)
    print('NOTE: all proposals remain UNVERIFIED. Do not use v2 scores as probabilities.')

if __name__ == '__main__':
    main()
