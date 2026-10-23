"""Event-level EARLY-warning evaluation on complete, held-out recordings.

Validation is the only permitted split for threshold selection. A test split is
read-only and reports early event recall, lead times, false alert / camera-hour,
and audio availability. Alerts after physical onset never count as an early hit.
"""
import argparse,json,statistics
from pathlib import Path
import numpy as np

def debounce(times,scores,quality=None,threshold=.5,k=2,n=3,cooldown=5.,min_quality=.45):
    ts=np.asarray(times,dtype=float);sc=np.asarray(scores,dtype=float)
    if len(ts)!=len(sc):raise ValueError('time / score length mismatch')
    q=np.ones(len(ts)) if quality is None else np.asarray(quality,dtype=float)
    if len(q)!=len(ts):raise ValueError('time / quality length mismatch')
    votes=[];alerts=[];last=-float('inf')
    for t,s,quality_at_t in zip(ts,sc,q):
        if not np.isfinite(s):votes.append(False)
        else:votes.append(bool(s>=threshold and quality_at_t>=min_quality))
        votes=votes[-n:]
        if len(votes)==n and sum(votes)>=k and t-last>=cooldown and votes[-1]:
            alerts.append({'timestamp_s':float(t),'risk':float(s)});last=float(t)
    return alerts

def score_events(rows,horizon=3.,leads=(1.,2.,3.)):
    """Rows: {times, alerts:[{timestamp_s,..}], intervals:[[onset,end],..]}; one-to-one matching."""
    tp=0;all_events=0;false=0;late=0;hours=0.;matched_leads=[];lead_hit={L:0 for L in leads};eligible={L:0 for L in leads}
    per_video=[]
    for r in rows:
        times=np.asarray(r['times'],float);start_time=float(times[0]) if len(times) else 0.
        duration=float(r.get('duration_s',times[-1] if len(times) else 0.))
        # Conservative: report full camera-hour duration of recording, including first inference warmup.
        hours+=duration/3600.
        events=sorted([(float(a),float(b)) for a,b in r['intervals']])
        alerts=sorted([float(a['timestamp_s']) for a in r['alerts']])
        all_events+=len(events);used=set();row_hits=[]
        for j,(onset,end) in enumerate(events):
            lead_candidates=[(idx,onset-t) for idx,t in enumerate(alerts)
                             if idx not in used and 0<onset-t<=horizon+1e-6]
            if lead_candidates:
                idx,delta=max(lead_candidates,key=lambda c:c[1]);used.add(idx)
                tp+=1;matched_leads.append(delta);row_hits.append({'onset_s':onset,'first_early_alert_s':alerts[idx],
                    'lead_time_s':round(delta,3)})
            else:row_hits.append({'onset_s':onset,'first_early_alert_s':None,'lead_time_s':None})
            for L in leads:
                # Do not punish impossibility of 3s lead if clip starts <3s before fight.
                if len(times) and onset - L >= start_time-1e-6:
                    eligible[L]+=1
                    if any(h['onset_s']==onset and h['lead_time_s'] is not None and h['lead_time_s']>=L-1e-6 for h in row_hits):lead_hit[L]+=1
        for idx,t in enumerate(alerts):
            if idx in used:continue
            if any(start<=t<=end for start,end in events):late+=1
            else:false+=1
        per_video.append(dict(video=r.get('video'),n_onsets=len(events),n_alerts=len(alerts),event_matches=row_hits,
                              n_unmatched_early_false=sum(1 for idx,t in enumerate(alerts) if idx not in used and not any(s<=t<=e for s,e in events))))
    precision=tp/max(tp+false+late,1)
    out={'videos':len(rows),'fight_onsets':all_events,'early_detected_onsets':tp,'early_event_recall':tp/all_events if all_events else None,
         'early_alert_precision':precision if tp+false+late else None,'false_early_alerts':false,
         'late_alerts_during_fights':late,'recorded_camera_hours':hours,
         'false_alerts_per_camera_hour':false/hours if hours>0 else None,
         'median_lead_seconds':statistics.median(matched_leads) if matched_leads else None,
         'early_lead_recall':{str(int(L) if L.is_integer() else L):{'eligible_onsets':eligible[L],
                                   'detected':lead_hit[L],'recall':lead_hit[L]/eligible[L] if eligible[L] else None} for L in leads},
         'per_video':per_video,'warnings':[]}
    if hours<1:out['warnings'].append('Less than 1 camera-hour tested; false-alert/h is statistically unstable.')
    if not all_events:out['warnings'].append('No positive fight onsets in this split; recall undefined.')
    return out

def make_model_predictions(manifest,index_file,ckpt,split='test',seed=42,min_quality=.45):
    import torch
    from src.manifest import load_manifest,split_groups
    from src.lock_splits import load_locked
    from c3_model import RiskNet
    load_locked(manifest)
    rows=load_manifest(manifest);sets=dict(zip(('train','val','test'),split_groups(rows,seed)))
    selected=sets[split]
    with open(index_file) as f:cache_index=json.load(f)
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    from src.causal_pose import PREPROCESSING_VERSION
    if ck.get('preprocessing_version')!=PREPROCESSING_VERSION:
        raise ValueError('Legacy/incompatible checkpoint preprocessing; regenerate causal features and retrain')
    model=RiskNet(*ck['dims'],d=ck.get('hidden_dim',192));model.load_state_dict(ck['state']);model.eval()
    results=[]
    for row in selected:
        key=row.get('video') or row.get('audio')
        if key not in cache_index:raise FileNotFoundError(f'Cache index missing {key}')
        with np.load(cache_index[key]) as z:
            if 'preprocessing_version' not in z or str(z['preprocessing_version'].item())!=PREPROCESSING_VERSION:
                raise ValueError('Legacy/non-causal evaluation cache rejected')
            # Refuse mixed legacy/new audio encoders. A trained model expects exactly one configuration.
            backend=str(z['audio_events'].item()) if 'audio_events' in z else 'none'
            if backend!=ck.get('audio_events','none'):raise ValueError(f'Audio backend mismatch: cache {backend} != checkpoint {ck.get("audio_events","none")}')
            if z['aud'].shape[-1]!=ck['dims'][2]:raise ValueError('Audio feature dimensions differ from checkpoint')
            x={k:torch.tensor(((z[k]-ck['norm'][k][0])/ck['norm'][k][1])[None],dtype=torch.float32) for k in ('vis','pose','aud')}
            mods=z['mods'].copy()
            if ck['mode']=='video':mods[:,2]=0
            with torch.no_grad():
                pre,_,_,_=model(x['vis'],x['pose'],x['aud'],torch.tensor(mods[None],dtype=torch.float32),
                                torch.tensor(z['sk'][None],dtype=torch.float32))
            results.append(dict(video=key,times=z['t'].astype(float).tolist(),scores=torch.sigmoid(pre)[0].numpy().astype(float).tolist(),
                         quality=np.where(mods[:,1]>0,z['qual'][:,4],1.).astype(float).tolist(),
                         intervals=row['intervals'],duration_s=float(z['t'][-1]),audio_present=bool(mods[:,2].any()),
                         group=row['group'],source=row['source']))
    return results

def eval_at(rows,threshold,horizon=3.):
    pred=[]
    for r in rows:
        pred.append({**r,'alerts':debounce(r['times'],r['scores'],r['quality'],threshold=threshold)})
    return score_events(pred,horizon=horizon)

def calibrate_val(rows,max_false_per_hour=.5):
    if not rows:raise ValueError('No validation videos')
    if not any(x['intervals'] for x in rows):raise ValueError('Need positive validation onsets')
    if not any(not x['intervals'] for x in rows):raise ValueError('Need complete negative validation videos')
    candidates=[]
    for thr in np.arange(.2,.951,.025):
        r=eval_at(rows,float(thr));candidates.append((float(thr),r))
    good=[(t,r) for t,r in candidates if r['false_alerts_per_camera_hour'] is not None and r['false_alerts_per_camera_hour']<=max_false_per_hour]
    if good:
        thr,report=max(good,key=lambda v:(v[1]['early_event_recall'] or 0,v[1]['early_alert_precision'] or 0,v[0]));status='val_target_met'
    else:
        thr,report=candidates[-1];status='val_target_unmet'
    return dict(threshold_watch=round(thr,4),status=status,val_summary={k:v for k,v in report.items() if k!='per_video'},
                target_false_alerts_per_camera_hour=max_false_per_hour)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--index',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--out',default='outputs/c3_event_eval.json')
    p.add_argument('--calibrate',action='store_true');p.add_argument('--calibration',help='Validation calibration JSON used on locked test')
    p.add_argument('--max-far',type=float,default=.5);p.add_argument('--threshold',type=float)
    a=p.parse_args()
    if a.calibrate and a.split!='val':p.error('Calibration is validation-only, never on test')
    if a.calibration and a.calibrate:p.error('Choose --calibrate OR --calibration')
    rows=make_model_predictions(a.manifest,a.index,a.weights,a.split)
    if a.calibrate:
        result=calibrate_val(rows,a.max_far)
    else:
        threshold=a.threshold
        if a.calibration:
            cal=json.loads(Path(a.calibration).read_text());threshold=float(cal['threshold_watch'])
        if threshold is None:raise ValueError('Provide --calibration path or explicit --threshold (uncalibrated diagnostic)')
        result=eval_at(rows,threshold);result['threshold_watch']=threshold
        result['calibrated_from_validation']=bool(a.calibration)
        if not a.calibration:result['warnings'].append('Threshold supplied manually; not calibrated on held-out validation.')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_video','warnings')},indent=2));print('Saved:',out)
    if 'warnings' in result:
        for w in result['warnings']:print('WARNING:',w)
