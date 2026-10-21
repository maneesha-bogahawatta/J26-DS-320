"""C3 v4 EXPERIMENTAL causal video-only early-warning proposal generator.

Unlike v3, the core detection receives no fight-onset timestamp. Each alert has
its actual emission time (alert_s), not just the beginning of supporting motion.
Offline scan-fights selects a pre-onset diagnostic window using the annotation;
this is retrospective benchmarking only, not a real-time performance estimate.

Requires: numpy, opencv-python, ultralytics; optional hardware acceleration.
Uses YOLO11-Pose + ByteTrack. It does NOT provide calibrated probabilities.
"""
from __future__ import annotations

import argparse
import csv
from collections import deque
import json
import math
from pathlib import Path

import cv2
import numpy as np

FIGHT_MANIFEST = Path('data/manifests/ubi_early_review_candidates.csv')
CONTROL_WINDOWS = Path('outputs/v3_evaluation/no_fight_control_windows.csv')
OUT = Path('outputs/v4_evaluation')


def read_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fight_onset(row):
    value = row.get('fight_onset_s', '')
    if value not in ('', None):
        return float(value)
    intervals = json.loads(row.get('intervals') or '[]')
    return min((float(a) for a, b in intervals), default=None)


def person_from_pose(person, confidence=.35):
    """Extract reliable COCO-17 torso/arm landmarks from v3 detection format."""
    xy = np.asarray(person['xy'], dtype=float)
    conf = np.asarray(person['conf'], dtype=float)
    if xy.shape != (17, 2) or conf.shape != (17,):
        raise ValueError('Expected COCO-17 pose data')
    valid = np.isfinite(xy).all(axis=1) & (conf >= confidence)
    # Wrist motion requires shoulders, torso location requires >=2 of shoulders/hips.
    torso_indices = [j for j in (5, 6, 11, 12) if valid[j]]
    torso = np.mean(xy[torso_indices], axis=0) if len(torso_indices) >= 2 else None
    shoulder = {9:5, 10:6}
    usable_wrists = [w for w, s in shoulder.items() if valid[w] and valid[s]]
    return {'id': int(person['id']), 'center': np.asarray(person['center'], dtype=float),
            'height': max(float(person['height']), 20.0),
            'torso': torso, 'xy': xy, 'valid': valid,
            'wrists': usable_wrists, 'quality': float(valid.mean())}


def interaction_evidence(a, b, old_a, old_b, dt):
    """One-pair, one-timepoint *candidate* evidence, based on visible pose only.

    Pair distance, relative wrist-to-shoulder movement and wrist-to-opponent
    movement are invariant to common camera translation (not camera rotation).
    """
    scale = max((a['height'] + b['height']) / 2.0, 20.)
    distance = float(np.linalg.norm(a['center'] - b['center']) / scale)
    close = distance < 1.45
    if not close or a['torso'] is None or b['torso'] is None:
        return None
    closing = 0.0
    if old_a is not None and old_b is not None and dt is not None and dt > 0:
        old_scale = max((old_a['height'] + old_b['height']) / 2.0, 20.)
        old_d = float(np.linalg.norm(old_a['center'] - old_b['center']) / old_scale)
        closing = (old_d - distance) / dt
    approach = closing > .16
    hand_motion = False
    hand_targets = 0
    if old_a is not None and old_b is not None and dt is not None and dt > 0:
        for actor, target, old_actor, old_target in ((a,b,old_a,old_b),(b,a,old_b,old_a)):
            if old_actor['torso'] is None or old_target['torso'] is None:
                continue
            for wrist in actor['wrists']:
                if wrist not in old_actor['wrists']:
                    continue
                shoulder = 5 if wrist == 9 else 6
                # Arm movement relative to its own shoulder discounts walking / camera translation.
                rel = actor['xy'][wrist] - actor['xy'][shoulder]
                rel_old = old_actor['xy'][wrist] - old_actor['xy'][shoulder]
                rel_speed = float(np.linalg.norm(rel-rel_old) / scale / dt)
                wrist_to_target = float(np.linalg.norm(actor['xy'][wrist]-target['torso']) / scale)
                old_dist = float(np.linalg.norm(old_actor['xy'][wrist]-old_target['torso']) / old_scale)
                approaching_hand = (old_dist-wrist_to_target) / dt
                # A near, rapidly advancing wrist is stronger than mere proximity.
                if wrist_to_target < .55 and approaching_hand > .18 and rel_speed > .36:
                    hand_motion = True
                    hand_targets += 1
    quality = min(a['quality'],b['quality'])
    score = min(1.0, .25*close + .25*approach + .40*hand_motion
                + .10*min(1., max(0.,closing)))
    return {'pair_ids': [a['id'],b['id']], 'quality': round(quality,4),
            'distance': round(distance,4), 'approach': bool(approach),
            'arm_intent': bool(hand_motion), 'score': round(score,4),
            'hand_targets': hand_targets}


class CausalAlertGate:
    """Requires a persistent pair and repeated arm cues before emitting alerts.

    No future observation is accessed. Alarm time == time condition was first met.
    """
    def __init__(self, min_pair_frames=3, min_arm_hits=2, window=.65,
                 cooldown=2.0, quality_min=.55, threshold=.65, max_gap=.25):
        self.min_pair_frames = min_pair_frames
        self.min_arm_hits = min_arm_hits
        self.window = window
        self.cooldown = cooldown
        self.quality_min = quality_min
        self.threshold = threshold
        self.max_gap = max_gap
        self.pairs = {}
        self.alerts = []

    def reset_tracking(self):
        self.pairs.clear()

    def update(self, evidence, t, scene_cut=False):
        if scene_cut:
            self.reset_tracking()
            return []
        emitted = []
        seen = set()
        for e in evidence:
            ids = tuple(sorted(e['pair_ids']))
            seen.add(ids)
            current = self.pairs.get(ids)
            if current is None or t-current['last'] > self.max_gap or t <= current['last']:
                current = {'last': t, 'frames': 0, 'history': deque(), 'last_alert': -math.inf}
            current['last'] = t
            current['frames'] += 1
            while current['history'] and t-current['history'][0][0] > self.window:
                current['history'].popleft()
            ok = (e['quality'] >= self.quality_min and e['score'] >= self.threshold)
            if ok and e['arm_intent']:
                current['history'].append((t,bool(e['approach']),float(e['score'])))
            enough = (current['frames'] >= self.min_pair_frames and
                      len(current['history']) >= self.min_arm_hits and
                      any(x[1] for x in current['history']) and
                      (t-current['history'][-1][0] <= self.max_gap))
            if enough and t-current['last_alert'] >= self.cooldown:
                event = {'pair_ids': list(ids), 'evidence_start_s': round(current['history'][0][0],3),
                         'alert_s': round(t,3), 'score': round(max(x[2] for x in current['history']),4),
                         'quality': e['quality'], 'signals':['persistent_pair','approach','directed_arm_motion']}
                self.alerts.append(event)
                emitted.append(event)
                current['last_alert'] = t
                current['history'].clear()
            self.pairs[ids] = current
        self.pairs = {k:v for k,v in self.pairs.items()
                      if k in seen and t-v['last'] <= self.max_gap}
        return emitted


class CausalPoseV4:
    def __init__(self, quality_min=.55, min_arm_hits=2, min_pair_frames=3):
        self.prev = {}
        self.prev_t = None
        self.gate = CausalAlertGate(quality_min=quality_min,
                                   min_arm_hits=min_arm_hits,
                                   min_pair_frames=min_pair_frames)
        self.max_quality = 0.0
        self.pair_samples = 0

    def update(self, people, t, scene_cut=False):
        if scene_cut:
            self.prev = {}
            self.prev_t = None
            return self.gate.update([], t, scene_cut=True)
        cur = {int(p['id']): person_from_pose(p) for p in people}
        dt = (t-self.prev_t) if self.prev_t is not None else None
        if dt is None or dt <= 0 or dt > .35:
            self.prev = {}
            self.gate.reset_tracking()
            dt = None
        evidences = []
        persons = list(cur.values())
        for i,a in enumerate(persons):
            for b in persons[i+1:]:
                ev = interaction_evidence(a,b,self.prev.get(a['id']),
                                          self.prev.get(b['id']),dt)
                if ev is not None:
                    self.max_quality = max(self.max_quality, ev['quality'])
                    self.pair_samples += 1
                    evidences.append(ev)
        alerts = self.gate.update(evidences,t)
        self.prev = cur
        self.prev_t = t
        return alerts


class SceneChangeGuard:
    """Conservative hard-cut detection, not a complete camera-motion model."""
    def __init__(self):
        self.prev_gray = None
        self.prev_hist = None
        self.cuts = 0

    def update(self, frame):
        small = cv2.resize(frame,(128,72),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray],[0],None,[32],[0,256]); cv2.normalize(hist,hist)
        cut = False
        if self.prev_gray is not None:
            difference = float(np.mean(cv2.absdiff(gray,self.prev_gray)) / 255.)
            corr = float(cv2.compareHist(hist,self.prev_hist,cv2.HISTCMP_CORREL))
            cut = difference > .35 and corr < .35
        self.prev_gray, self.prev_hist = gray, hist
        if cut: self.cuts += 1
        return cut


def scan_video(path, weights='yolo11n-pose.pt', sample_fps=15.,
               start_s=0., end_s=None, device=None, quality_min=.55,
               min_arm_hits=2, min_pair_frames=3):
    """Streaming/causal scan; never receives an onset or label. end_s only bounds reading."""
    if sample_fps <= 0 or start_s < 0 or (end_s is not None and end_s <= start_s):
        raise ValueError('Invalid time range or FPS')
    from ultralytics import YOLO
    # Import v3 formatting only. No modification or call to v3 scanning logic.
    from src.auto_annotate_precursors_v3 import detections
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {path}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: raise ValueError(f'Invalid FPS: {path}')
        stride = max(1,round(fps/sample_fps))
        beginning = int(math.ceil(start_s*fps))
        end_frame = None if end_s is None else int(math.ceil(end_s*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES,beginning)
        model = YOLO(weights)  # tracker state isolated across videos
        detector = CausalPoseV4(quality_min=quality_min, min_arm_hits=min_arm_hits,
                                min_pair_frames=min_pair_frames)
        scene = SceneChangeGuard()
        frames = 0
        for frame_idx in range(beginning, end_frame if end_frame is not None else 10**12):
            ok, frame = cap.read()
            if not ok: break
            if (frame_idx-beginning) % stride:
                continue
            t = frame_idx/fps
            cut = scene.update(frame)
            if cut:
                detector.update([],t,scene_cut=True)
                # Isolate ByteTrack state as well across cuts. Instantiating fresh YOLO is
                # heavier but avoids carrying identities from unrelated scenes.
                model = YOLO(weights)
                frames += 1
                continue
            kwargs = {'persist':True, 'tracker':'bytetrack.yaml',
                      'classes':[0], 'conf':.25, 'verbose':False}
            if device: kwargs['device'] = device
            result = model.track(frame,**kwargs)[0]
            detector.update(detections(result),t)
            frames += 1
        return {'v4_alerts':detector.gate.alerts,
                'sampled_frames':frames, 'scene_cuts':scene.cuts,
                'pair_samples':detector.pair_samples,
                'v4_max_quality':round(detector.max_quality,4)}
    finally:
        cap.release()


def scan_rows(rows, kind, args):
    output = []
    for i,row in enumerate(rows,1):
        rec = dict(row)
        rec.update(v4_status='error',v4_alerts='[]',v4_alert_count=0,
                   v4_first_alert_s='',v4_1to3s_alert='',
                   v4_max_quality='',v4_scene_cuts='',v4_sampled_frames='',v4_error='')
        try:
            if kind == 'controls':
                start = float(row['control_window_start_s'])
                end = float(row['control_window_end_s'])
                onset = None
            else:
                # The onset is used ONLY to choose the retrospective evaluation window.
                # It never enters scan_video() or CausalPoseV4.
                onset = fight_onset(row)
                if onset is None: raise ValueError('No annotated fight onset')
                start = max(0.,onset-args.pre_seconds)
                end = onset
            data = scan_video(row['video'],weights=args.weights,
                              sample_fps=args.sample_fps,start_s=start,end_s=end,
                              device=args.device,quality_min=args.quality_min,
                              min_arm_hits=args.min_arm_hits,
                              min_pair_frames=args.min_pair_frames)
            alerts = data['v4_alerts']
            rec.update(v4_status='proposal' if alerts else 'no_proposal',
                       v4_alerts=json.dumps(alerts),v4_alert_count=len(alerts),
                       v4_first_alert_s=alerts[0]['alert_s'] if alerts else '',
                       v4_max_quality=data['v4_max_quality'],
                       v4_scene_cuts=data['scene_cuts'],
                       v4_sampled_frames=data['sampled_frames'])
            if onset is not None:
                rec['v4_1to3s_alert'] = int(any(1 <= onset-a['alert_s'] <= 3 for a in alerts))
        except Exception as exc:
            rec['v4_error'] = str(exc)
        output.append(rec)
        print(f'[{i}/{len(rows)}] {Path(row.get("video","unknown")).name}: '
              f'{rec["v4_status"]} alerts={rec["v4_alert_count"]}')
    return output



def compare_results(v3_rows, v4_rows, review_rows=None, kind='controls'):
    """Join scans by video, reporting counts, not model precision or recall.

    Comparison is only valid when v3/v4 rows refer to the same time windows.
    Explicitly check window bounds for control samples.
    """
    v3 = {Path(r.get('video','')).name: r for r in v3_rows}
    reviewed = {Path(r.get('video','')).name: r for r in (review_rows or [])}
    result = []
    for new in v4_rows:
        name = Path(new.get('video','')).name
        old = v3.get(name)
        if old is None:
            continue
        if kind == 'controls':
            for f in ('control_window_start_s','control_window_end_s'):
                if f in old and f in new and abs(float(old[f])-float(new[f])) > .02:
                    raise ValueError(f'Mismatched control windows for {name}: {f}')
        review = reviewed.get(name,{})
        result.append({'video': name, 'v3_status': old.get('v3_status',''),
                       'v4_status': new.get('v4_status',''),
                       'v3_max_score':old.get('v3_max_score',''),
                       'v4_alert_count':new.get('v4_alert_count',''),
                       'v4_first_alert_s':new.get('v4_first_alert_s',''),
                       'v4_1to3s_alert':new.get('v4_1to3s_alert',''),
                       'review_label':review.get('review_label','unreviewed'),
                       'review_notes':review.get('review_notes',''),
                       'v4_error':new.get('v4_error','')})
    return result


def summarize_comparison(rows, kind):
    checked = [r for r in rows if r['v4_status'] in ('proposal','no_proposal')]
    v3_hits = sum(r['v3_status']=='proposal' for r in checked)
    v4_hits = sum(r['v4_status']=='proposal' for r in checked)
    result={'matched':len(rows),'v4_scanned':len(checked),
            'v3_alarm_windows':v3_hits,'v4_alarm_windows':v4_hits,
            'v4_errors':sum(r['v4_status']=='error' for r in rows),
            'kind':kind}
    if kind=='controls':
        reviewed = [r for r in checked if r['review_label']=='non_aggressive']
        result.update(reviewed_non_aggressive=len(reviewed),
                      v3_alarms_in_reviewed=sum(r['v3_status']=='proposal' for r in reviewed),
                      v4_alarms_in_reviewed=sum(r['v4_status']=='proposal' for r in reviewed))
    else:
        result['v4_1to3s_alarms']=sum(str(r['v4_1to3s_alert'])=='1' for r in checked)
    return result


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    scan = sub.add_parser('scan-video', help='Scan video from beginning, or specified interval; no labels')
    scan.add_argument('--video',required=True)
    scan.add_argument('--start',type=float,default=0.)
    scan.add_argument('--end',type=float,default=None)
    scan.add_argument('--out',default=str(OUT/'single_video_v4.json'))
    fights = sub.add_parser('scan-fights', help='OFFLINE annotated pre-onset diagnostic only')
    fights.add_argument('--manifest',default=str(FIGHT_MANIFEST))
    fights.add_argument('--pre-seconds',type=float,default=5.)
    fights.add_argument('--video-id',default=None)
    fights.add_argument('--limit',type=int,default=5,help='Default 5; use 0 for all')
    fights.add_argument('--out',default=str(OUT/'fight_diagnostic_v4.csv'))
    controls = sub.add_parser('scan-controls',help='Scan v3 sampled no-fight control windows with v4')
    controls.add_argument('--windows',default=str(CONTROL_WINDOWS))
    controls.add_argument('--limit',type=int,default=20,help='Default 20; use 0 for all')
    controls.add_argument('--out',default=str(OUT/'no_fight_control_results_v4.csv'))
    compare = sub.add_parser('compare',help='Compare already generated v3 and v4 CSVs')
    compare.add_argument('--kind',choices=['controls','fights'],default='controls')
    compare.add_argument('--v3',default=None,help='Path to existing v3 control/fight results')
    compare.add_argument('--v4',default=None,help='Path to v4 control/fight results')
    compare.add_argument('--reviews',default='outputs/v3_evaluation/control_review_labels.csv')
    compare.add_argument('--out',default=None)
    for cmd in (scan,fights,controls):
        cmd.add_argument('--weights',default='yolo11n-pose.pt')
        cmd.add_argument('--sample-fps',type=float,default=15.)
        cmd.add_argument('--device',default=None)
        cmd.add_argument('--quality-min',type=float,default=.55)
        cmd.add_argument('--min-arm-hits',type=int,default=2)
        cmd.add_argument('--min-pair-frames',type=int,default=3)
    args = p.parse_args(argv)
    if args.command == 'compare':
        is_ctrl=args.kind=='controls'
        v3_path=args.v3 or ('outputs/v3_evaluation/no_fight_control_results.csv'
                              if is_ctrl else 'data/manifests/ubi_auto_precursors_v3.csv')
        v4_path=args.v4 or str(OUT / ('no_fight_control_results_v4.csv'
                                     if is_ctrl else 'fight_diagnostic_v4.csv'))
        dest=Path(args.out or str(OUT / ('comparison_controls_v3_v4.csv'
                                         if is_ctrl else 'comparison_fights_v3_v4.csv')))
        reviews=read_csv(args.reviews) if is_ctrl and Path(args.reviews).is_file() else []
        comp=compare_results(read_csv(v3_path),read_csv(v4_path),reviews,args.kind)
        fields=list(comp[0]) if comp else ['video','v3_status','v4_status']
        write_csv(dest,comp,fields)
        stats=summarize_comparison(comp,args.kind)
        print(json.dumps(stats,indent=2))
        print('Saved:',dest)
        print('Counts only, NOT verified precision/recall or population false-positive rate.')
        return
    if (args.sample_fps <= 0 or not 0 <= args.quality_min <= 1 or
            args.min_arm_hits < 1 or args.min_pair_frames < 1):
        p.error('Invalid detector settings')
    if args.command == 'scan-video':
        data = scan_video(args.video, weights=args.weights,sample_fps=args.sample_fps,
                          start_s=args.start,end_s=args.end,device=args.device,
                          quality_min=args.quality_min,min_arm_hits=args.min_arm_hits,
                          min_pair_frames=args.min_pair_frames)
        dest=Path(args.out); dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps({'video':args.video,'range_start_s':args.start,
                                    'range_end_s':args.end,**data},indent=2),encoding='utf-8')
        print(f'Alerts: {len(data["v4_alerts"])} | Scene cuts: {data["scene_cuts"]} | Saved: {dest}')
        return
    if args.command == 'scan-fights':
        if args.pre_seconds <= 0 or args.limit < 0: p.error('Invalid pre-seconds or limit')
        rows = read_csv(args.manifest)
        if args.video_id:
            rows = [r for r in rows if Path(r['video']).stem == args.video_id.removesuffix('.mp4')]
            if not rows: p.error('video-id not found in manifest')
        if args.limit: rows=rows[:args.limit]
        kind='fights'
    else:
        if args.limit < 0: p.error('Invalid limit')
        rows=read_csv(args.windows)
        if args.limit: rows=rows[:args.limit]
        kind='controls'
    output=scan_rows(rows,kind,args)
    fields=list(dict.fromkeys([k for row in output for k in row]))
    if not fields: fields=['video','v4_status','v4_alerts','v4_error']
    write_csv(args.out,output,fields)
    proposed=sum(r['v4_status']=='proposal' for r in output)
    errs=sum(r['v4_status']=='error' for r in output)
    print(f'Processed: {len(output)} | v4 proposed: {proposed} | errors: {errs} | Saved: {args.out}')
    if kind=='fights':
        target=sum(r['v4_1to3s_alert']==1 for r in output)
        print(f'OFFLINE 1-3s alert-timestamp overlaps: {target}; NOT verified sensitivity.')
    else:
        print('No-fight footage alarms are CANDIDATE false alarms, not a population FPR.')
    print('Experimental heuristic, UNVERIFIED; v4 scores are NOT probabilities.')


if __name__=='__main__':
    cli()
