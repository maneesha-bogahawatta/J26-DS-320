"""Add bounded weak-label TRAIN contexts; freeze every existing fit/dev/val role.

Source selection uses a fixed hash order and pose visibility, never model scores.
Existing validation windows and the locked test remain unchanged. New labels are
original UBI weak labels, not independently verified aggression/contact labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.advanced_data import weak_targets
from src.advanced_experiment import load_windows
from src.foundation_io import sha256, write_json
from src.lock_splits import load_locked
from src.pose_dataset import sequence_window
from src.pose_sequences import extract_video, load_cache


def validate_new_sources(items, by_id, original_groups):
    groups = set(original_groups)
    for item in items:
        row = by_id[item['video_id']]
        if row['split'] != 'train' or item['role'] not in ('fit', 'dev'):
            raise ValueError('Expansion permits fitting/internal-development train sources only')
        if row['leakage_group'] in groups:
            raise ValueError('Expansion source group overlaps an existing or new source')
        if bool(json.loads(row['intervals'])) != item['positive']:
            raise ValueError('Expansion class disagrees with original weak annotations')
        groups.add(row['leakage_group'])


def prepare(args):
    root=Path(args.out_dir).resolve()
    (root/'.ultralytics/Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(root/'.ultralytics')
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    original,meta=load_windows(args.parent_data)
    rows=load_locked(meta['locked_manifest_path'])
    by_id={r['video_id']:r for r in rows}
    used=set(meta['source_groups'].values())
    plan=dict(schema='c3_training_source_expansion_v1',seed=4243,
        parent_windows_sha256=meta['windows_sha256'],normal_quota=args.normal,fight_quota=args.fight,
        context_seconds=8,selection='fixed hash order of unused locked train groups; >=1 pose-visible person in two fixed probes',
        development='each fourth newly accepted source per class; all original roles frozen',
        unchanged_validation=True,test_used=False,labels='original weak UBI binary intervals, no new verified labels')
    write_json(root/'expansion_plan.json',plan)
    model=YOLO(str(Path('yolo11n-pose.pt').resolve()))
    completed=[]
    for path in sorted((root/'completed').glob('*.json')):
        completed.append(json.loads(path.read_text()))
    used.update(by_id[j['video_id']]['leakage_group'] for j in completed)
    for positive,quota in ((False,args.normal),(True,args.fight)):
        accepted=sum(j['positive']==positive for j in completed)
        candidates=[r for r in rows if r['split']=='train' and r['leakage_group'] not in used
            and bool(json.loads(r['intervals']))==positive and float(r['duration_s'])>=8]
        candidates.sort(key=lambda r:hashlib.sha256(f'4243:{r["video_id"]}'.encode()).hexdigest())
        for row in candidates[:100]:
            if accepted>=quota:break
            if row['leakage_group'] in used:continue
            intervals=json.loads(row['intervals'])
            start=max(0.,min(float(row['duration_s'])-8,intervals[0][0]-4)) if positive else max(0.,(float(row['duration_s'])-8)/2)
            probe=root/'probes'/(row['video_id']+'.json')
            if probe.exists():
                visible=json.loads(probe.read_text())['visible_people']
            else:
                cap=cv2.VideoCapture(row['video']); visible=[]
                for t in (start+1.5,start+5.5):
                    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read(); count=0
                    if ok:
                        result=model.predict(frame,imgsz=640,conf=.3,device='cpu',verbose=False)[0]
                        if result.keypoints is not None and result.keypoints.conf is not None:
                            count=int(((result.keypoints.conf.cpu().numpy()>=.3).sum(1)>=6).sum())
                    visible.append(count)
                cap.release()
                write_json(probe,dict(video_id=row['video_id'],split='train',start_s=start,visible_people=visible))
            if min(visible)<1:continue
            path,provenance=extract_video(row,root/'new_cache',start=start,end=start+8)
            arrays,_=load_cache(path)
            if not any(sequence_window(arrays,i)['quality']>=.45 for i in range(7,len(arrays['t']),6)):
                continue
            role='dev' if accepted%4==0 else 'fit'
            item=dict(video_id=row['video_id'],positive=positive,role=role,start_s=start,
                entry=dict(path=str(path.resolve()),cache_sha256=sha256(path),source_sha256=provenance['source_sha256']))
            write_json(root/'completed'/(row['video_id']+'.json'),item)
            completed.append(item);used.add(row['leakage_group']);accepted+=1
            print(json.dumps(dict(completed=len(completed),video=row['video_id'],positive=positive,role=role,visible=visible)),flush=True)
        if accepted<quota:print(f'Coverage shortfall: positive={positive}: {accepted}/{quota}',flush=True)
    # Append to the already quality-matched archive; old rows keep their exact values.
    additions={k:[] for k in original}
    roles=dict(meta['roles']); groups=dict(meta['source_groups'])
    coverage=[]
    validate_new_sources(completed,by_id,meta['source_groups'].values())
    for item in sorted(completed,key=lambda j:j['video_id']):
        row=by_id[item['video_id']]
        if row['split']!='train' or row['leakage_group'] in set(meta['source_groups'].values()):
            raise ValueError('Expansion source crossed original split/role boundaries')
        arrays,provenance=load_cache(item['entry']['path'])
        config={k:provenance.get(k) for k in meta['feature_contract']}
        if config!=meta['feature_contract'] or sha256(item['entry']['path'])!=item['entry']['cache_sha256'] or provenance['source_sha256']!=row['content_sha256']:
            raise ValueError('Expansion feature contract/cache mismatch')
        intervals=json.loads(row['intervals']); bins={}; last=-1e9
        for i,t in enumerate(arrays['t']):
            if t-last<.4-1e-8:continue
            last=t;y,known=weak_targets(float(t),intervals)
            bins.setdefault(tuple(y.astype(int))+tuple(known.astype(int)),[]).append(i)
        chosen=set()
        for indices in bins.values():
            chosen.update(indices[j] for j in np.unique(np.linspace(0,len(indices)-1,min(6,len(indices))).round().astype(int)))
        kept=0
        for i in sorted(chosen):
            sample=sequence_window(arrays,i)
            current=int(sample['observed_mask'][-1].sum(0).max())
            if sample['temporal_mask'].sum()<8 or sample['quality']<.45 or current<6:continue
            y,known=weak_targets(sample['cutoff_s'],intervals)
            if not known.any():continue
            values=dict(sample,targets=y,target_mask=known,role=item['role'],video_id=row['video_id'],
                original_window_index=-1,current_observed_joints=current)
            for key in additions:additions[key].append(values[key])
            kept+=1
        roles[row['video_id']]=item['role'];groups[row['video_id']]=row['leakage_group']
        coverage.append(dict(video_id=row['video_id'],role=item['role'],positive=item['positive'],windows=kept))
    values={k:np.concatenate([v,np.asarray(additions[k],dtype=v.dtype)]) for k,v in original.items()}
    output=root/'windows';output.mkdir(parents=True,exist_ok=True)
    with (output/'windows.npz').open('xb') as stream:np.savez_compressed(stream,**values)
    summary={role:dict(windows=int((values['role']==role).sum()),videos=len(set(values['video_id'][values['role']==role].tolist())),
        supervised=values['target_mask'][values['role']==role].sum(0).tolist(),
        positives=(values['targets'][values['role']==role]*values['target_mask'][values['role']==role]).sum(0).astype(int).tolist())
        for role in ('fit','dev','val')}
    new_meta=dict(meta,roles=roles,source_groups=groups,summary=summary,windows_sha256=sha256(output/'windows.npz'),
        expansion_plan_sha256=sha256(root/'expansion_plan.json'),expansion_sources=coverage,
        parent_windows_sha256=meta['windows_sha256'],parent_data_directory=str(Path(args.parent_data).resolve()),
        original_window_index_policy='Inherited original indices; newly appended sources have -1; source/time remains unique')
    write_json(output/'metadata.json',new_meta)
    for key in original:
        assert np.array_equal(original[key][original['role']=='val'],values[key][values['role']=='val']),key
    write_json(root/'expansion_summary.json',dict(summary=summary,new_sources=coverage,
        added_sources=len(completed),added_windows=len(values['role'])-len(original['role']),validation_unchanged=True,test_used=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-data',default='data/pose_cache/diagnosis_guarded_v1/windows')
    p.add_argument('--out-dir',default='data/pose_cache/model_improvement_expanded_v1')
    p.add_argument('--normal',type=int,default=24);p.add_argument('--fight',type=int,default=8)
    prepare(p.parse_args())
