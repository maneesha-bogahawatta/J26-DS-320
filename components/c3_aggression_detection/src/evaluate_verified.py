"""Coverage-scoped verified evaluation using frozen policy and existing tools.

No threshold tuning. Locked test input is rejected. Early matching reuses
evaluate_events; window targets reuse pose_dataset; decision replay reuses the
actual CausalDecision. Unreviewed time never becomes negative recording hours.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from src.advanced_experiment import metrics
from src.data_quality import checked_document, strict_records
from src.diagnose_stgcn import ranking
from src.evaluate_events import score_events
from src.foundation_io import read_csv, sha256, write_json
from src.pose_dataset import labels_at
from src.pose_sequences import load_cache
from src.predict_advanced import CausalDecision


def union_ranges(ranges):
    result=[]
    for a,b in sorted(ranges):
        if not math.isfinite(a) or not math.isfinite(b) or a>=b:raise ValueError('Invalid coverage interval')
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([float(a),float(b)])
    return result


def subtract_ranges(ranges,excluded):
    result=union_ranges(ranges)
    for x,y in union_ranges(excluded):
        following=[]
        for a,b in result:
            if y<=a or x>=b:following.append([a,b]);continue
            if a<x:following.append([a,x])
            if y<b:following.append([y,b])
        result=following
    return result


def within(t,ranges):return any(a<=t<b for a,b in ranges)


def boolean(value):
    if value is True or value=='True' or value=='true':return True
    if value is False or value=='False' or value=='false':return False
    raise ValueError('Explicit boolean pose validity required')


def validate_timeline(rows,expected_times):
    times=np.array([float(r['t_s']) for r in rows]);expected=np.asarray(expected_times)
    if not np.isfinite(times).all() or (np.diff(times)<=0).any():raise ValueError('Duplicate/nonmonotonic decision timestamp')
    if len(times)!=len(expected) or not np.allclose(times,expected,atol=1e-6,rtol=0):
        raise ValueError('Incomplete or misaligned decision timeline; false-alarm exposure is not established')


def window_report(y,p,threshold):
    if not y:return dict(n=0,positives=0,precision=None,recall=None,f1=None,auroc=None,
        average_precision=None,pr_auc_trapezoid=None,threshold=float(threshold),tp=0,fp=0,tn=0,fn=0)
    result=ranking(y,p);result.update(metrics(y,p,threshold))
    if not result['tp']+result['fp']:result['precision']=None
    if not result['positives']:result['recall']=None;result['f1']=None
    return result


def evaluate(records,scores,contexts,thresholds,enabled,split='val'):
    if split not in ('train','val'):raise ValueError('Locked test is excluded from this development evaluator')
    if len(thresholds)!=2 or len(enabled)!=2 or not np.isfinite(thresholds).all() or any(type(v)!=bool for v in enabled):
        raise ValueError('Two finite thresholds and explicit eligibility booleans required')
    grouped=defaultdict(list)
    for r in records:
        if r['split']==split:grouped[r['video_id']].append(r)
    by_video=defaultdict(list)
    for r in scores:by_video[r['video_id']].append(r)
    heads=[dict(y=[],p=[],known_decisions=0,abstentions=0),dict(y=[],p=[],known_decisions=0,abstentions=0)]
    negative_seconds=0.;negative_alerts=[0,0];event_rows=[];eligible_events=0;excluded_events=0;total_decisions=0;valid_decisions=0;provenance=[]
    for key,table in by_video.items():
        rr=grouped[key];lo,hi=contexts[key];known=union_ranges([b for r in rr for b in r['coverage_s']])
        negative=union_ranges([[max(a,lo),min(b,hi)] for r in rr if r['label']=='non_aggressive' for a,b in r['coverage_s'] if max(a,lo)<min(b,hi)])
        # Conservatively exclude overlapping non-negative reviews from FAR exposure.
        other=[b for r in rr if r['label']!='non_aggressive' for b in r['coverage_s']]
        negative=subtract_ranges(negative,other);seconds=sum(b-a for a,b in negative);negative_seconds+=seconds
        contacts=sorted(set(float(r['contact_s']) for r in rr if r['label']=='precursor' and r['contact_s'] is not None))
        eligible=[t for t in contacts if lo<=t<hi and any(a<=t-3 and t<=b for a,b in known) and t-3>=lo]
        excluded_events+=len(contacts)-len(eligible);eligible_events+=len(eligible)
        decision=CausalDecision([t if flag else 1.000001 for t,flag in zip(thresholds,enabled)])
        early_alerts=[];times=[]
        fight_ranges=union_ranges([b for r in rr for b in r['fight_intervals']])
        for row in table:
            t=float(row['t_s']);valid=boolean(row['valid_pose']);total_decisions+=1;valid_decisions+=valid;times.append(t)
            values=[row.get('weak_onset_score'),row.get('fight_score')]
            p=[float(v) for v in values] if valid else None
            if valid and (not np.isfinite(p).all() or any(v<0 or v>1 for v in p)):raise ValueError('Finite sigmoid scores required')
            _,_,alert=decision.update(t,p,valid,int(row.get('segment_id',0)),json.loads(row.get('track_ids_json','[]')))
            y,mask=labels_at(t,rr)
            for h in range(2):
                if mask[h]:
                    heads[h]['known_decisions']+=1;heads[h]['abstentions']+=not valid
                    if valid:heads[h]['y'].append(int(y[h]));heads[h]['p'].append(p[h])
            if not alert:continue
            h=1 if decision.previous=='WEAK_ACTIVE_FIGHT_ALERT' else 0
            if within(t,negative):negative_alerts[h]+=1
            if h==0 and ((mask[0] and (within(t,negative) or any(0<contact-t<=3 for contact in eligible))) or within(t,fight_ranges)):
                early_alerts.append(dict(timestamp_s=t))
        # Reuse one-to-one event matching, but never its whole-recording FAR denominator.
        event_rows.append(dict(video=key,times=times,alerts=early_alerts,
            intervals=[[t,max([t+1e-7]+[b for a,b in fight_ranges if a>=t])] for t in eligible],duration_s=0.))
        provenance.append(dict(video_id=key,negative_reviewed_seconds=seconds,eligible_contacts=len(eligible),
            excluded_contacts=len(contacts)-len(eligible),context=[lo,hi]))
    event=score_events(event_rows)
    tp=event['early_detected_onsets'];false=event['false_early_alerts']+event['late_alerts_during_fights'];fn=eligible_events-tp
    event_precision=tp/(tp+false) if tp+false else None;event_recall=tp/eligible_events if eligible_events else None
    event_f1=2*tp/(2*tp+false+fn) if eligible_events else None
    per_head={}
    for h,name in enumerate(('verified_precursor','verified_active_fight')):
        report=window_report(heads[h]['y'],heads[h]['p'],thresholds[h]);report.update(
            eligible_policy_head=enabled[h],known_decisions_including_abstention=heads[h]['known_decisions'],
            pose_abstentions=heads[h]['abstentions'],observable_coverage=len(heads[h]['y'])/heads[h]['known_decisions'] if heads[h]['known_decisions'] else None,
            false_alerts_on_reviewed_negative_time=negative_alerts[h],
            reviewed_negative_camera_hours=negative_seconds/3600,
            false_alarms_per_reviewed_negative_hour=negative_alerts[h]*3600/negative_seconds if negative_seconds else None)
        per_head[name]=report
    reasons=[]
    if not eligible_events:reasons.append('No strict verified contact/precursor events with complete reviewed three-second prehistory')
    if negative_seconds==0:reasons.append('No strict reviewed negative time with a complete authenticated score timeline')
    return dict(status='blocked' if reasons else 'descriptive_verified_evaluation',split=split,
        heads=per_head,event_precision=event_precision,event_recall=event_recall,event_f1=event_f1,
        eligible_contact_events=eligible_events,excluded_insufficient_context_events=excluded_events,
        true_positive_events=tp,false_positive_early_alerts=false,false_negative_events=fn,
        late_early_head_alerts=event['late_alerts_during_fights'],
        median_lead_seconds=event['median_lead_seconds'],lead_recall=event['early_lead_recall'],
        event_matches=event['per_video'],sources=provenance,decisions=total_decisions,usable_decisions=valid_decisions,
        blockers=reasons,operational_alerts_enabled=False,test_sources_used=0,
        limitations=['PR metrics are conditional window-level discrimination on explicit verified labels, not event PR-AUC.',
            'Event metrics use the frozen deployed confirmation/identity/cooldown policy; no-pose events remain missed.',
            'FAR denominator is unique reviewed negative seconds inside authenticated cached contexts, including pose abstention.',
            'Overlapping/duplicate review ranges are counted once; unknown/uncertain time supplies no exposure.',
            'Balanced prioritized diagnostic clips cannot establish population camera-hour false alarms.',
            'Disabled heads cannot establish useful event sensitivity even if raw score ranking is measurable.'])


def run(args):
    rows,doc=checked_document(args.annotations,args.locked);by_id={r['video_id']:r for r in rows}
    records,rejected=strict_records(doc);index=json.loads(Path(args.index).read_text())
    if index.get('locked_manifest_sha256')!=sha256(args.locked):raise ValueError('Cache index differs from locked splits')
    policy=json.loads(Path(args.policy).read_text())
    if policy.get('schema')!='c3_frozen_verified_evaluation_policy_v1' or policy.get('locked_manifest_sha256')!=sha256(args.locked):raise ValueError('Invalid frozen policy provenance')
    if policy.get('operational_alerts_enabled') is not False:raise ValueError('Research evaluator does not enable operational alerts')
    if sha256(policy['checkpoint'])!=policy['checkpoint_sha256']:raise ValueError('Referenced model changed')
    scores=read_csv(args.scores);groups=defaultdict(list);contexts={}
    for row in scores:
        key=row['video_id']
        if key not in by_id or by_id[key]['split']!=args.split:raise ValueError('Test, unknown, or wrong-split prediction source')
        groups[key].append(row)
    annotated_sources={r['video_id'] for r in records if r['split']==args.split}
    unavailable=sorted(annotated_sources-set(index['entries']))
    needed=annotated_sources & set(index['entries'])
    if needed-set(groups):raise ValueError('Verified sources are missing score timelines; do not drop missed events')
    for key,table in groups.items():
        if key not in index['entries']:raise ValueError('Missing authenticated pose cache')
        entry=index['entries'][key]
        if sha256(entry['path'])!=entry['cache_sha256']:raise ValueError('Pose cache changed')
        arrays,meta=load_cache(entry['path'])
        if meta['video_id']!=key or meta['source_sha256']!=by_id[key]['content_sha256']:raise ValueError('Prediction/cache source mismatch')
        expected=[];last=-math.inf
        for t in arrays['t']:
            if t-last>=policy['decision_step_s']-1e-8:expected.append(t);last=t
        validate_timeline(table,expected);contexts[key]=[meta['extracted_start_s'],meta['extracted_end_s']]
    report=evaluate(records,scores,contexts,policy['thresholds'],policy['research_heads_enabled'],args.split)
    report['verified_sources_without_cache_index']=unavailable
    if unavailable:
        report['blockers'].append('Some verified sources have no authenticated cache; reported metrics cover only the explicit cached subset')
        report['status']='blocked'
    report.update(annotations_sha256=sha256(args.annotations),scores_sha256=sha256(args.scores),
        frozen_policy_sha256=sha256(args.policy),unsupported_verified_records=rejected,
        policy=policy,threshold_tuning_performed=False)
    write_json(args.out,report);print(json.dumps({k:report[k] for k in ('status','eligible_contact_events','event_precision','event_recall','blockers')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked',default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--annotations',default='data/foundation/ai_visual_v1/annotations.json')
    p.add_argument('--index',default='outputs/data_quality_v1/val_index.json')
    p.add_argument('--scores',required=True);p.add_argument('--policy',default='outputs/data_quality_v1/frozen_evaluation_policy.json')
    p.add_argument('--split',choices=('train','val'),default='val');p.add_argument('--out',required=True)
    run(p.parse_args())
