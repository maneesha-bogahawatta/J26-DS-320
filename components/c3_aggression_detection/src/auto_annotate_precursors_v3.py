"""C3 v3 causal short-duration *proposal* generator; not verified labels.
Run from component root: python -m src.auto_annotate_precursors_v3 --video-id F_66_1_2_0_0
For all: python -m src.auto_annotate_precursors_v3
Requires: ultralytics, opencv-python, numpy. Does not write to v1/v2 outputs.
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
import cv2
import numpy as np

DEFAULT_MANIFEST = Path('data/manifests/ubi_early_review_candidates.csv')
DEFAULT_OUTPUT = Path('data/manifests/ubi_auto_precursors_v3.csv')
DEFAULT_REVIEW = Path('data/manifests/ubi_review_priority_v3.csv')


def load_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def onset_from(row):
    if row.get('fight_onset_s') not in (None, ''):
        return float(row['fight_onset_s'])
    intervals = json.loads(row.get('intervals') or '[]')
    if not intervals:
        raise ValueError('No annotated fight intervals')
    return float(intervals[0][0])


def detections(result):
    if result.boxes is None or result.boxes.id is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.int().cpu().numpy()
    xy = result.keypoints.xy.cpu().numpy()
    conf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
            else np.ones(xy.shape[:2], np.float32))
    people = []
    for i, tid in enumerate(ids):
        box = boxes[i]
        h = max(float(box[3]-box[1]), 20.0)
        people.append({'id':int(tid), 'center':np.array([(box[0]+box[2])/2, (box[1]+box[3])/2],float),
                       'height':h, 'xy':xy[i].astype(float), 'conf':conf[i].astype(float)})
    return people


class CausalInteractionCues:
    """Per-video state; only current and previous pose data are read."""
    def __init__(self, alpha=0.6, confidence=0.35, gap=0.30):
        self.alpha, self.confidence, self.gap = alpha, confidence, gap
        self.state = {}

    def update(self, persons, t):
        cur = {}
        pair_candidates = []
        signals = set()
        quality_joint = []
        for p in persons:
            old = self.state.get(p['id'])
            raw_xy = np.asarray(p['xy'],float)
            conf = np.asarray(p['conf'],float)
            valid = conf >= self.confidence
            qual = float(valid.mean())
            quality_joint.append(qual)
            xy = raw_xy.copy()
            center = np.asarray(p['center'],float).copy()
            speed = 0.0; wrist_speed = 0.0; wrist_accel = 0.0
            if old is not None and 0 < t-old['t'] <= self.gap:
                dt = t-old['t']
                center = self.alpha*center + (1-self.alpha)*old['center']
                xy[valid] = self.alpha*xy[valid]+(1-self.alpha)*old['xy'][valid]
                speed = float(np.linalg.norm(center-old['center'])/p['height']/dt)
                for w in (9,10):
                    if valid[w] and old['valid'][w]:
                        v = float(np.linalg.norm(xy[w]-old['xy'][w])/p['height']/dt)
                        wrist_speed = max(wrist_speed,v)
                        if old['wrist_speed'] is not None:
                            wrist_accel = max(wrist_accel,abs(v-old['wrist_speed'])/dt)
            raised = any(valid[sh] and valid[wr] and (xy[sh,1]-xy[wr,1])/p['height']>.18
                         for sh,wr in ((5,9),(6,10)))
            if raised: signals.add('raised_arm')
            if speed > .85: signals.add('rapid_movement')
            if wrist_speed > 1.0: signals.add('fast_wrist')
            if wrist_accel > 5.0: signals.add('wrist_acceleration')
            cur[p['id']] = {**p,'center':center,'xy':xy,'valid':valid,'t':float(t),
                            'speed':speed,'wrist_speed':wrist_speed if old is not None else None,
                            'wrist_accel':wrist_accel,'raised':raised,'quality':qual}

        # Pair-level interaction reduces false positives due to unrelated people.
        tracked = list(cur.values())
        for i,a in enumerate(tracked):
            for b in tracked[i+1:]:
                prev_a = self.state.get(a['id']); prev_b = self.state.get(b['id'])
                scale = max((a['height']+b['height'])/2,20.)
                distance = float(np.linalg.norm(a['center']-b['center'])/scale)
                closing = 0.0
                if prev_a is not None and prev_b is not None:
                    dt = t-max(prev_a['t'],prev_b['t'])
                    if 0 < dt <= self.gap:
                        old_d = np.linalg.norm(prev_a['center']-prev_b['center'])/scale
                        closing = float((old_d-distance)/dt)
                close = distance < 1.65
                approach = close and closing > .20
                active_arm = a['raised'] or b['raised'] or max(a['wrist_speed'] or 0,b['wrist_speed'] or 0)>.85
                # nearest-wrist to other's shoulders/torso (normalized body height)
                near_arm = False
                for attacker,target in ((a,b),(b,a)):
                    wrists = [j for j in (9,10) if attacker['valid'][j]]
                    torso = [j for j in (5,6,11,12) if target['valid'][j]]
                    if wrists and torso:
                        d = min(np.linalg.norm(attacker['xy'][w]-target['xy'][k])/scale
                                for w in wrists for k in torso)
                        near_arm = near_arm or d < .35
                if approach: signals.add('rapid_approach')
                if close and near_arm: signals.add('arm_torso_proximity')
                score = (0.38*approach + 0.20*(close and active_arm)
                         + 0.22*(close and near_arm)
                         + 0.12*(close and max(a['speed'],b['speed'])>.85)
                         + 0.08*(close and max(a['wrist_accel'],b['wrist_accel'])>5))
                q = min(a['quality'],b['quality'])
                pair_candidates.append({'score':float(score),'quality':float(q),
                                        'ids':[a['id'],b['id']],'distance':distance,'closing':closing})

        best = max(pair_candidates,key=lambda v:v['score'],default=None)
        quality = float(best['quality']) if best else (float(np.mean(quality_joint)) if quality_joint else 0.)
        # No pair means no multi-person aggression proposal. Low quality is retained for audit.
        score = float(best['score']) if best else 0.
        self.state = cur
        return {'timestamp':float(t),'score':round(score,4),'quality':round(quality,4),
                'signals':sorted(signals),'tracked_people':len(cur),
                'pair_ids':best['ids'] if best else [],'pair_visible':bool(best)}


def intervals_from_observations(observations, onset, threshold=.36, min_steps=2, expected_dt=1/15):
    out=[]; current=[]; last=None
    def flush():
        if len(current)>=min_steps:
            start=current[0]['timestamp']
            end=min(onset,current[-1]['timestamp']+expected_dt)
            if end>start: out.append([round(start,3),round(end,3)])
        current.clear()
    for o in observations:
        if last is not None and o['timestamp']-last > expected_dt*1.7:
            flush()
        if o['pair_visible'] and o['score']>=threshold and o['quality']>=.25:
            current.append(o)
        else:
            flush()
        last=o['timestamp']
    flush()
    return out


def analyze(path, onset, weights, sample_fps, pre_seconds, threshold, min_steps, device):
    from ultralytics import YOLO
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError('Cannot open video')
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if fps<=0: raise ValueError('Invalid FPS')
        first=int(math.ceil(max(0.,onset-pre_seconds)*fps))
        stop=int(math.ceil(onset*fps))
        stride=max(1,int(round(fps/sample_fps)))
        actual_dt=stride/fps
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        model=YOLO(weights)  # new per video -> tracker IDs cannot leak between clips
        cues=CausalInteractionCues(gap=max(.30,actual_dt*2.2))
        obs=[]
        for frame_index in range(first,stop):
            ok,frame=cap.read()
            if not ok: break
            if (frame_index-first)%stride: continue
            kwargs={'persist':True,'tracker':'bytetrack.yaml','classes':[0],
                    'conf':.25,'verbose':False}
            if device: kwargs['device']=device
            result=model.track(frame,**kwargs)[0]
            obs.append(cues.update(detections(result),frame_index/fps))
        return obs, intervals_from_observations(obs,onset,threshold,min_steps,actual_dt)
    finally:
        cap.release()


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def detector_main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    parser.add_argument('--out',default=str(DEFAULT_OUTPUT))
    parser.add_argument('--review-out',default=str(DEFAULT_REVIEW))
    parser.add_argument('--weights',default='yolo11n-pose.pt')
    parser.add_argument('--video-id',default=None,help='Video stem (e.g. F_66_1_2_0_0) for diagnostic run')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--sample-fps',type=float,default=15.)
    parser.add_argument('--pre-seconds',type=float,default=5.)
    parser.add_argument('--threshold',type=float,default=.36)
    parser.add_argument('--min-steps',type=int,default=2)
    parser.add_argument('--device',default=None)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--audit-sample',type=int,default=15)
    args=parser.parse_args(argv)
    if args.sample_fps<=0 or args.pre_seconds<=0 or not 0<=args.threshold<=1 or args.min_steps<1:
        parser.error('Invalid sampling/threshold/min-steps')
    rows=load_rows(Path(args.manifest))
    if args.video_id:
        rows=[r for r in rows if Path(r.get('video','')).stem==args.video_id.removesuffix('.mp4')]
        if not rows: parser.error(f'No match for --video-id {args.video_id}')
    if args.limit>0: rows=rows[:args.limit]
    output=[]
    for i,row in enumerate(rows,1):
        rec=dict(row)
        rec.update(v3_proposals='[]',v3_status='unverified',v3_max_score='',v3_quality='',
                   v3_signals='[]',v3_review_priority='none',v3_error='',v3_method='causal_pose_heuristic_v3')
        try:
            video=Path(row['video']); onset=onset_from(row)
            if not video.is_file(): raise FileNotFoundError(str(video))
            obs,windows=analyze(video,onset,args.weights,args.sample_fps,args.pre_seconds,
                                args.threshold,args.min_steps,args.device)
            rec['v3_proposals']=json.dumps(windows)
            rec['v3_max_score']=round(max((x['score'] for x in obs),default=0.),4)
            rec['v3_quality']=round(sum(x['quality'] for x in obs)/len(obs),4) if obs else 0.
            rec['v3_signals']=json.dumps(sorted({s for o in obs for s in o['signals']}))
            rec['v3_status']='proposal' if windows else ('unobservable' if not obs else 'no_proposal')
            rec['v3_review_priority']='high' if windows else ('medium' if float(rec['v3_quality'])<.45 else 'audit')
        except Exception as exc:
            rec['v3_status']='error';rec['v3_error']=str(exc);rec['v3_review_priority']='high'
        output.append(rec)
        print(f'[{i}/{len(rows)}] {Path(row.get("video","unknown")).name}: {rec["v3_status"]}')
    extra=['v3_proposals','v3_status','v3_max_score','v3_quality','v3_signals','v3_review_priority','v3_error','v3_method']
    fields=list(rows[0])+extra if rows else extra
    write_rows(Path(args.out),output,fields)
    priority=[r for r in output if r['v3_review_priority'] in ('high','medium')]
    pool=[r for r in output if r['v3_review_priority']=='audit']
    audits=random.Random(args.seed).sample(pool,min(args.audit_sample,len(pool)))
    write_rows(Path(args.review_out),priority+audits,fields)
    print(f'Processed: {len(output)} | Proposals: {sum(r["v3_status"]=="proposal" for r in output)} '
          f'| Errors: {sum(r["v3_status"]=="error" for r in output)} | Review queue: {len(priority)+len(audits)}')
    print('CAUTION: heuristic proposals are UNVERIFIED; v3 scores are not fight probabilities.')
    print('Saved:',args.out,'and',args.review_out)

# ======================================================================
# Evaluation integration (formerly src/evaluate_precursor_v3.py).
# Subcommands: compare, sample-controls, scan-controls.
# No changes to v1/v2 manifests or human-reviewed annotation labels.
# ======================================================================

V2_DEFAULT = 'data/manifests/ubi_auto_precursors_v2.csv'
V3_DEFAULT = 'data/manifests/ubi_auto_precursors_v3.csv'
FULL_DEFAULT = 'data/manifests/ubi_review.csv'
OUTDIR_DEFAULT = 'outputs/v3_evaluation'


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def basename(value):
    return str(value).replace('\\', '/').rsplit('/', 1)[-1]


def pairs(value):
    try:
        data = json.loads(value or '[]')
        if isinstance(data, list):
            return [(float(p[0]), float(p[1])) for p in data if isinstance(p, (list, tuple)) and len(p) == 2 and float(p[1]) > float(p[0])]
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        pass
    return []


def onset(row):
    val = row.get('fight_onset_s', '')
    if val not in ('', None):
        try:
            return float(val)
        except ValueError:
            pass
    ivs = pairs(row.get('intervals', '[]'))
    return min((s for s, _ in ivs), default=None)


def overlap_seconds(windows, start, stop):
    """Total seconds of window coverage in [start, stop], de-duplicating overlaps."""
    sections = sorted((max(a, start), min(b, stop)) for a, b in windows if min(b, stop) > max(a, start))
    if not sections:
        return 0.0
    merged = []
    for a, b in sections:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return round(sum(b-a for a, b in merged), 3)


def numeric(value, fallback=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return fallback


def build_comparison(v2_rows, v3_rows):
    lookup = {basename(r.get('video', '')): r for r in v2_rows}
    result = []
    for r in v3_rows:
        name = basename(r.get('video', ''))
        other = lookup.get(name, {})
        t = onset(r)
        intervals = pairs(r.get('v3_proposals', '[]'))
        # 1-3 seconds of lead: v3 proposal must intersect [onset-3, onset-1].
        target_overlap = overlap_seconds(intervals, max(0., t-3), max(0., t-1)) if t is not None else 0.
        quality = numeric(r.get('v3_quality'))
        score = numeric(r.get('v3_max_score'))
        result.append({
            'video': name,
            'fight_onset_s': '' if t is None else round(t, 3),
            'v2_status': other.get('v2_status', 'missing'),
            'v3_status': r.get('v3_status', ''),
            'v3_max_score': r.get('v3_max_score', ''),
            'v3_quality': r.get('v3_quality', ''),
            'v3_interval_count': len(intervals),
            'v3_first_proposal_s': min((p[0] for p in intervals), default=''),
            'v3_target_1to3s_overlap_s': target_overlap,
            'v3_proposals': r.get('v3_proposals', '[]'),
            'reason': '',
            '_ranking': quality * score,
        })
    return result


def shortlist(rows):
    out, chosen = [], set()

    def add(items, reason, k):
        added = 0
        for r in items:
            if added >= k:
                break
            if r['video'] in chosen:
                continue
            chosen.add(r['video'])
            out.append({**r, 'reason': reason})
            added += 1

    by_rank = lambda r: (-r['_ranking'], -numeric(r['v3_target_1to3s_overlap_s']), r['video'])
    add((r for r in rows if r['video'] == 'F_66_1_2_0_0.mp4'), 'F66 diagnostic (unverified)', 1)
    add(sorted((r for r in rows if r['v3_status']=='proposal' and r['v2_status']!='proposal'
                and r['v3_target_1to3s_overlap_s'] >= .13),key=by_rank),
        'v3-only; 1-3s target window', 12)
    add(sorted((r for r in rows if r['v3_status']=='proposal' and r['v2_status']!='proposal'
                and r['v3_target_1to3s_overlap_s'] < .13),key=by_rank),
        'v3-only; outside 1-3s window', 5)
    add(sorted((r for r in rows if r['v3_status']=='no_proposal'),
               key=lambda r:(-numeric(r['v3_quality']),r['video'])),
        'v3 no-proposal; missed-cue audit', 6)
    add(sorted((r for r in rows if r['v3_status']=='proposal' and r['v2_status']=='proposal'),key=by_rank),
        'v2 and v3 agreement', 3)
    add(sorted((r for r in rows if r['v3_status']=='proposal'),
               key=lambda r:(numeric(r['v3_quality']),r['video'])),
        'v3 low-quality; false-cue audit', 5)
    return out


def do_compare(args):
    v2 = read_csv(args.v2)
    v3 = read_csv(args.v3)
    comp = build_comparison(v2, v3)
    review = shortlist(comp)
    out = Path(args.outdir)
    cols = ['video','fight_onset_s','v2_status','v3_status','v3_max_score',
            'v3_quality','v3_interval_count','v3_first_proposal_s',
            'v3_target_1to3s_overlap_s','v3_proposals']
    write_csv(out/'version_comparison.csv', comp, cols)
    write_csv(out/'review_shortlist.csv', review, ['reason']+cols)
    count_v3 = sum(r['v3_status']=='proposal' for r in comp)
    count_new = sum(r['v3_status']=='proposal' and r['v2_status']!='proposal' for r in comp)
    count_target = sum(r['v3_status']=='proposal' and r['v3_target_1to3s_overlap_s']>=.13 for r in comp)
    print(f'Compared: {len(comp)} | v3 proposed: {count_v3} | v3-only vs v2: {count_new}')
    print(f'v3 proposals intersecting 1-3s pre-onset window (>=0.13s): {count_target}')
    print(f'Targeted manual review: {len(review)} videos')
    print(f'Saved: {out / "version_comparison.csv"} and {out / "review_shortlist.csv"}')
    print('NOT model accuracy, sensitivity or precision; all proposals remain unverified.')


def make_control_rows(manifest_rows, n, seed):
    rng = random.Random(seed)
    eligible = []
    for r in manifest_rows:
        if pairs(r.get('intervals')):
            continue
        duration = numeric(r.get('duration_s'), -1)
        if duration <= 5.5:
            continue
        eligible.append((r,duration))
    picked = rng.sample(eligible, min(len(eligible), n))
    result=[]
    for row, duration in picked:
        end = round(rng.uniform(5.1, duration-0.1), 3)
        result.append({'video':row.get('video', ''), 'duration_s':row.get('duration_s',''),
                       'control_window_start_s':round(end-5,3), 'control_window_end_s':end,
                       'annotation_intervals':row.get('intervals','[]'),
                       'label_note':'annotation has no labeled fight; manual verification still needed'})
    return result


def do_sample_controls(args):
    rows=make_control_rows(read_csv(args.manifest), args.number, args.seed)
    dest=Path(args.outdir)/'no_fight_control_windows.csv'
    write_csv(dest,rows,['video','duration_s','control_window_start_s','control_window_end_s','annotation_intervals','label_note'])
    print(f'Chosen {len(rows)} no-labeled-fight windows. Saved: {dest}')
    print('These are negative controls, not verified non-aggression labels.')


def do_scan_controls(args):
    inp=Path(args.outdir)/'no_fight_control_windows.csv'
    rows=read_csv(inp)
    if args.limit:
        rows=rows[:args.limit]
    results=[]
    for i,r in enumerate(rows,1):
        name=basename(r['video'])
        record=dict(r)
        record.update(v3_status='error',v3_proposals='[]',v3_max_score='',v3_quality='',error='')
        try:
            observations, proposals=analyze(Path(r['video']),float(r['control_window_end_s']),
                                            args.weights,args.sample_fps,5.0,args.threshold,
                                            args.min_steps,args.device)
            record['v3_proposals']=json.dumps(proposals)
            record['v3_status']='proposal' if proposals else 'no_proposal'
            record['v3_max_score']=max((x['score'] for x in observations),default=0.0)
            record['v3_quality']=round(sum(x['quality'] for x in observations)/len(observations),4) if observations else 0.0
        except Exception as err:
            record['error']=str(err)
        results.append(record)
        print(f'[{i}/{len(rows)}] {name}: {record["v3_status"]}')
    dest=Path(args.outdir)/'no_fight_control_results.csv'
    write_csv(dest,results,list(results[0]) if results else ['video','v3_status','error'])
    positives=sum(x['v3_status']=='proposal' for x in results)
    errors=sum(x['v3_status']=='error' for x in results)
    print(f'No-fight control windows: {len(results)} | proposals: {positives} | errors: {errors}')
    print('Proposals in annotated no-fight footage are candidate false alarms, not a population FPR.')
    print('Saved:',dest)


def evaluation_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('compare', help='Compare existing v2/v3 outputs and make a review shortlist')
    a.add_argument('--v2',default=V2_DEFAULT)
    a.add_argument('--v3',default=V3_DEFAULT)
    a.set_defaults(action=do_compare)
    b=sub.add_parser('sample-controls',help='Sample no-fight windows; no pose inference yet')
    b.add_argument('--manifest',default=FULL_DEFAULT)
    b.add_argument('--number',type=int,default=20)
    b.add_argument('--seed',type=int,default=42)
    b.set_defaults(action=do_sample_controls)
    c=sub.add_parser('scan-controls',help='Pose inference on sampled no-fight windows')
    c.add_argument('--limit',type=int,default=20)
    c.add_argument('--weights',default='yolo11n-pose.pt')
    c.add_argument('--sample-fps',type=float,default=15.)
    c.add_argument('--threshold',type=float,default=.36)
    c.add_argument('--min-steps',type=int,default=2)
    c.add_argument('--device',default=None)
    c.set_defaults(action=do_scan_controls)
    for command in (a, b, c):
        command.add_argument('--outdir', default=OUTDIR_DEFAULT,
                             help='Directory for generated evaluation CSV files')
    args=p.parse_args(argv)
    if args.task=='sample-controls' and args.number<1: p.error('--number must be positive')
    if args.task=='scan-controls' and (args.limit<0 or args.sample_fps<=0 or args.min_steps<1
                                       or not 0 <= args.threshold <= 1):
        p.error('invalid scan arguments')
    args.action(args)

# ---- Unified CLI: existing video analysis plus offline v3 validation ----
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('compare', 'sample-controls', 'scan-controls'):
        return evaluation_main(argv)
    return detector_main(argv)


if __name__ == '__main__':
    main()
