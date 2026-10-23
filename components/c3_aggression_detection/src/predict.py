"""Offline causal feature-cache inference; mask audio when unavailable."""
import json,uuid,datetime,argparse
import numpy as np
import torch
from c3_model import RiskNet
from src.features import DV,DP,DA
from src.audio_events import cue_report

def predict(cache,weights,camera_id='cam_01',out=None,min_quality=.45,calibration=None):
    import pathlib
    z=np.load(cache);ck=torch.load(weights,map_location='cpu',weights_only=False)
    from src.causal_pose import PREPROCESSING_VERSION
    if ('preprocessing_version' not in z or str(z['preprocessing_version'].item())!=PREPROCESSING_VERSION
            or ck.get('preprocessing_version')!=PREPROCESSING_VERSION):
        raise ValueError('Legacy/incompatible preprocessing rejected; use a matching causal cache and checkpoint')
    if calibration:
        cal=json.loads(pathlib.Path(calibration).read_text())
        ck['threshold_watch']=float(cal['threshold_watch'])
        ck['threshold_high']=max(ck['threshold_watch']+.15,ck['threshold_high'])
    cache_audio=str(z['audio_events'].item()) if 'audio_events' in z else 'none'
    if cache_audio!=ck.get('audio_events','none'):
        raise ValueError('Cached audio-event backend differs from checkpoint. Re-cache or use matching weights.')
    dv,dp,da=ck['dims'];model=RiskNet(dv,dp,da,d=ck.get('hidden_dim',192));model.load_state_dict(ck['state']);model.eval()
    x={k:torch.from_numpy(((z[k]-ck['norm'][k][0])/ck['norm'][k][1]).astype('float32'))[None] for k in ('vis','pose','aud')}
    mods=z['mods'].astype('float32').copy()
    if ck['mode']=='video':mods[:,2]=0
    x['mods']=torch.from_numpy(mods)[None]
    x['sk']=torch.from_numpy(z['sk'].astype('float32'))[None]
    with torch.no_grad():
        pre,fight,tto,_=model(x['vis'],x['pose'],x['aud'],x['mods'],x['sk'])
    pre=torch.sigmoid(pre)[0].numpy();fight=torch.sigmoid(fight)[0].numpy()
    assessments=[];events=[];votes=[];last=-1e6
    for i,t in enumerate(z['t']):
        q=float(z['qual'][i,4]);risk=float(pre[i]);tier='HIGH' if risk>=ck['threshold_high'] else 'WATCH' if risk>=ck['threshold_watch'] else 'LOW'
        if q<min_quality and mods[i,1]>0:tier='INSUFFICIENT_QUALITY'
        assessments.append(dict(timestamp_s=float(t),pre_attack_risk=risk,active_fight_probability=float(fight[i]),
                                tier=tier,track_quality=q,modality='audio_video' if mods[i,2]>0 else 'video_only',
                                audio_cues=json.dumps(cue_report(z['aud'][i]) if mods[i,2]>0 else {})))
        votes.append(tier in ('WATCH','HIGH'));votes=votes[-3:]
        if len(votes)==3 and sum(votes)>=2 and votes[-1] and float(t)-last>=5:
            last=float(t)
            events.append(dict(schema_version='1.0',component='c3_early_aggression',event_id=str(uuid.uuid4()),
                               camera_id=camera_id,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               source_time_s=float(t),person_ids=[],bounding_boxes=[],threat_score=risk,
                               priority_tier=tier,uncertainty=float(max(1-q, .1)),track_quality=q,
                               explanation_signals=([dict(signal='learned_temporal_pre_attack',source='c3_risknet')] +
                                 [dict(signal=k,source='yamnet_sound_event',value=v) for k,v in
                                  (cue_report(z['aud'][i]) if mods[i,2]>0 else {}).items() if v>=.35]),
                               modality=assessments[-1]['modality'],model_version=ck['version'],
                               human_review_required=True))
    from src.output_paths import output_path
    path=output_path(camera_id,out,'c3_events.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf8') as f:
        for ev in events:f.write(json.dumps(ev)+'\n')
    import csv
    with open(path.with_name(path.stem+'_step_assessments.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,assessments[0].keys());w.writeheader();w.writerows(assessments)
    return dict(steps=len(assessments),alerts=len(events),path=str(path))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cache',required=True);a.add_argument('--weights',required=True)
    a.add_argument('--camera-id',default='cam_01');a.add_argument('--out')
    a.add_argument('--calibration');args=a.parse_args();print(predict(args.cache,args.weights,args.camera_id,args.out,calibration=args.calibration))
