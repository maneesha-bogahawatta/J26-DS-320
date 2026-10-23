"""Audit whether pretrained scream/yell cues correlate with VERIFIED early aggression.

This is a descriptive audit, not an aggression detector and not a source of
verified annotations. Only same-recording synchronized audio is valid.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from src.audio_events import CUE_NAMES
from src.manifest import load_manifest

def audit(manifest,index_path,out='outputs/audio_cue_audit.csv',threshold=.35,horizon=3.):
    with open(index_path) as f:index=json.load(f)
    records=[]
    for r in load_manifest(manifest):
        key=r.get('video') or r.get('audio')
        if key not in index:continue
        with np.load(index[key]) as z:
            if 'audio_events' not in z or str(z['audio_events'].item())!='yamnet':continue
            if z['aud'].shape[-1]<24:continue
            t=z['t'];aud=z['aud'][:,16:24];available=z['mods'][:,2]>0
            for i,time in enumerate(t):
                if not available[i]:continue
                cls='baseline'
                for onset,end in r['intervals']:
                    if onset<=time<end:cls='active_fight';break
                    if onset-horizon<=time<onset:cls='pre_onset_3s'
                for a,b in r.get('precursors',[]):
                    if a<=time<=b and cls=='pre_onset_3s':cls='reviewed_precursor';break
                records.append({'video':key,'source':r['source'],'label':cls,'time_s':round(float(time),2),
                         **{n.lower().replace(', sobbing','').replace(' ','_'):round(float(v),4) for n,v in zip(CUE_NAMES,aud[i])}})
    if not records:raise ValueError('No synchronized YAMNet event-feature caches found. Run cache_dataset --audio-events yamnet first.')
    dest=Path(out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)
    print('Saved',len(records),'audio windows to',dest)
    for cls in ('baseline','reviewed_precursor','pre_onset_3s','active_fight'):
        subset=[r for r in records if r['label']==cls]
        if subset:print(cls,'windows',len(subset),'screaming>='+str(threshold),
                        round(sum(r['screaming']>=threshold for r in subset)/len(subset),4))
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True)
    p.add_argument('--index',required=True);p.add_argument('--out',default='outputs/audio_cue_audit.csv')
    a=p.parse_args();audit(a.manifest,a.index,a.out)
