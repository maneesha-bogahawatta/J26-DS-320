"""Trailing pose windows with split enforcement and coverage-scoped supervision."""
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.foundation_io import sha256
from src.lock_splits import load_locked
from src.pose_sequences import CACHE_SCHEMA, load_cache


def sequence_window(arrays, end_index, frames=48, people=2):
    if frames<1 or people!=2 or not 0<=end_index<len(arrays['t']):
        raise ValueError('Baseline needs positive frames, two person slots, valid cutoff')
    segment = arrays['segment_id'][end_index]
    begin = end_index
    while begin>0 and end_index-begin+1<frames and arrays['segment_id'][begin-1]==segment:
        if arrays['t'][begin]-arrays['t'][begin-1]>.35:
            break
        begin -= 1
    indices = list(range(begin,end_index+1))
    scores = defaultdict(float)
    offsets = arrays['frame_offsets']
    for i in indices:
        for j in range(offsets[i],offsets[i+1]):
            box = arrays['boxes'][j]
            scores[int(arrays['track_ids'][j])] += float((box[2]-box[0])*(box[3]-box[1]))
    selected = sorted(scores,key=lambda pid:(-scores[pid],pid))[:people]
    sk = np.zeros((frames,17,people,3),np.float32)
    joint_mask = np.zeros((frames,17,people),bool)
    observed = np.zeros_like(joint_mask)
    imputed = np.zeros_like(joint_mask)
    person_mask = np.zeros((frames,people),bool)
    temporal_mask = np.zeros(frames,bool)
    times = np.full(frames,-1.,np.float64)
    frame_indices = np.full(frames,-1,np.int64)
    qs = []
    for pos,i in enumerate(indices,start=frames-len(indices)):
        temporal_mask[pos]=True;times[pos]=arrays['t'][i];frame_indices[pos]=arrays['frame_index'][i]
        for j in range(offsets[i],offsets[i+1]):
            pid=int(arrays['track_ids'][j])
            if pid not in selected:
                continue
            slot=selected.index(pid)
            sk[pos,:,slot]=arrays['keypoints'][j]
            observed[pos,:,slot]=arrays['observed'][j]
            imputed[pos,:,slot]=arrays['imputed'][j]
            joint_mask[pos,:,slot]=observed[pos,:,slot]|imputed[pos,:,slot]
            person_mask[pos,slot]=joint_mask[pos,:,slot].any()
            qs.append(arrays['quality'][j,4])
    return dict(sk=sk,joint_mask=joint_mask,observed_mask=observed,imputed_mask=imputed,
                person_mask=person_mask,temporal_mask=temporal_mask,frame_times=times,frame_indices=frame_indices,
                selected_ids=np.asarray(selected+[-1]*(people-len(selected)),np.int64),
                quality=np.float32(np.mean(qs) if qs else 0.),cutoff_s=float(arrays['t'][end_index]))


def labels_at(t, records, weak_intervals=None):
    """No default background labels: unknown and conflicting regions are masked."""
    votes=[set(),set()]
    for r in records:
        if r['review_status']!='independently_verified' or r.get('issues'):
            continue
        if not any(a<=t<b for a,b in r['coverage_s']):
            continue
        if r['label']=='non_aggressive':
            votes[0].add(0.);votes[1].add(0.)
        if r['label']=='precursor' and r['contact_s'] is not None:
            if 0<r['contact_s']-t<=3 and any(a<=t<b for a,b in r['precursors']):
                votes[0].add(1.)
        if any(a<=t<b for a,b in r['fight_intervals']):
            votes[1].add(1.)
    if weak_intervals is not None and not votes[1]:
        votes[1].add(float(any(a<=t<b for a,b in weak_intervals)))
    targets=np.asarray([next(iter(v)) if len(v)==1 else 0. for v in votes],np.float32)
    mask=np.asarray([len(v)==1 for v in votes],bool)
    if mask[1] and targets[1]>0:
        mask[0]=False
    return targets,mask


class PoseSequenceDataset(Dataset):
    def __init__(self, locked, annotations, index, split='train', purpose='fitting',
                 frames=48, step_s=.2, min_frames=8, min_quality=.45, supervision='verified'):
        if split not in ('train','val','test') or (split=='test' and purpose!='evaluation'):
            raise ValueError('Test data is available only for explicit evaluation, never fitting/tuning')
        if frames<1 or step_s<=0 or not 1<=min_frames<=frames or not 0<=min_quality<=1:
            raise ValueError('Invalid sequence configuration')
        if supervision not in ('verified','weak-fight'):
            raise ValueError('supervision must be verified or explicitly defined weak-fight')
        manifest=load_locked(locked)
        expected=sha256(locked)
        ann=json.loads(Path(annotations).read_text(encoding='utf-8'))
        idx=json.loads(Path(index).read_text(encoding='utf-8'))
        if ann.get('schema')!='c3_review_evidence_v1' or idx.get('schema')!=CACHE_SCHEMA:
            raise ValueError('Invalid annotation/cache index schema')
        if ann['locked_manifest_sha256']!=expected or idx['locked_manifest_sha256']!=expected:
            raise ValueError('Annotations/cache index belong to a different locked split version')
        self.rows={r['video_id']:r for r in manifest}
        self.records=defaultdict(list)
        for r in ann['records']:
            if r['video_id'] not in self.rows or r['split']!=self.rows[r['video_id']]['split']:
                raise ValueError('Annotation split does not match the locked source')
            self.records[r['video_id']].append(r)
        self.frames,self.min_frames,self.min_quality,self.supervision=frames,min_frames,min_quality,supervision
        self.entries={}
        self.windows=[]
        self._last_cache=None
        feature_contract=None
        for key,entry in sorted(idx['entries'].items()):
            if key not in self.rows:
                raise ValueError('Cache index references an unknown video')
            if self.rows[key]['split']!=split:
                continue
            if sha256(entry['path'])!=entry['cache_sha256']:
                raise ValueError('Cache content changed after index creation')
            arrays,meta=load_cache(entry['path'])
            if meta['video_id']!=key or meta['source_sha256']!=self.rows[key]['content_sha256']:
                raise ValueError('Cache/source identity mismatch')
            contract={k:meta.get(k) for k in ('schema','preprocessing','sample_fps','weights_sha256','tracker_sha256','libraries')}
            if feature_contract is not None and contract!=feature_contract:
                raise ValueError('Mixed pose preprocessing/weights/sampling configurations in one dataset')
            feature_contract=contract
            self.entries[key]=entry
            last=-float('inf')
            for i,t in enumerate(arrays['t']):
                if t-last>=step_s-1e-8:
                    self.windows.append((key,i));last=float(t)
        self.split=split
        self.provenance={'locked_sha256':expected,'annotations_sha256':sha256(annotations),'index_sha256':sha256(index),
                         'split':split,'frames':frames,'step_s':step_s,'supervision':supervision,
                         'min_frames':min_frames,'min_quality':min_quality}
        self.provenance['feature_contract_sha256']=hashlib.sha256(json.dumps(feature_contract,sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        key,i=self.windows[index]
        if self._last_cache is None or self._last_cache[0]!=key:
            self._last_cache=(key,*load_cache(self.entries[key]['path']))
        arrays=self._last_cache[1]
        result=sequence_window(arrays,i,self.frames)
        weak=json.loads(self.rows[key]['intervals']) if self.supervision=='weak-fight' else None
        target,mask=labels_at(result['cutoff_s'],self.records[key],weak)
        if result['temporal_mask'].sum()<self.min_frames or result['quality']<self.min_quality:
            mask[:]=False
        return dict(result,targets=target,target_mask=mask,video_id=key,split=self.split,
                    leakage_group=self.rows[key]['leakage_group'])


def collate_sequences(samples):
    if not samples or len({s['split'] for s in samples})!=1:
        raise ValueError('A batch must contain samples from exactly one split')
    tensors=('sk','joint_mask','observed_mask','imputed_mask','person_mask','temporal_mask',
             'frame_times','frame_indices','selected_ids','quality','targets','target_mask')
    result={k:torch.from_numpy(np.stack([s[k] for s in samples])) for k in tensors}
    result.update({k:[s[k] for s in samples] for k in ('video_id','split','leakage_group','cutoff_s')})
    return result


def readiness(dataset):
    counts=np.zeros(2,int);positives=np.zeros(2,int)
    for sample in dataset:
        counts+=sample['target_mask']
        positives+=(sample['targets']>.5)&sample['target_mask']
    return {'split':dataset.split,'windows':len(dataset),'supervised_windows_per_head':counts.tolist(),
            'positive_windows_per_head':positives.tolist(),'heads':['precursor','active_fight'],
            'early_training_blocked':not bool(positives[0]),
            'reason':'No verified positive precursor windows meeting cache/quality requirements' if not positives[0] else None}
