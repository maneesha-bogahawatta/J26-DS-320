"""Training-fit-only automatic proposals; no verified labels or new training.

Reuses weak_targets, trained score helpers, original caches and the existing v4
observed-pose interaction evidence. Teacher scores on fitting sources are not
independent supervision and must never validate themselves.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import Windows, load_windows, score
from src.auto_annotate_precursors_v4 import person_from_pose, interaction_evidence
from src.foundation_io import sha256, write_csv, write_json
from src.improve_model import audit_roles
from src.lock_splits import load_locked
from src.optimize_model import load_model
from src.pose_sequences import load_cache


def proposal_kind(onset,active,known,source_has_weak_fight,active_score,motion):
    if not known[1]:return 'uncertain_boundary',None
    if active:return ('weak_active_supported' if active_score>=.5 else 'weak_active_teacher_disagreement'),1
    if known[0] and onset:
        return ('provisional_preonset_motion' if motion else 'weak_preonset_context_only'),None
    if not source_has_weak_fight and active_score<.35 and not motion:return 'provisional_quiet_negative',0
    return 'uncertain_or_teacher_disagreement',None


def run(args):
    data,meta=load_windows('data/pose_cache/model_improvement_expanded_v1/windows');audit_roles(data,meta)
    locked={r['video_id']:r for r in load_locked(meta['locked_manifest_path'])}
    ix=np.flatnonzero(data['role']=='fit')
    if any(locked[k]['split']!='train' for k in data['video_id'][ix]):raise ValueError('Only locked train fitting sources permitted')
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    config=dict(schema='c3_automatic_proposals_v1',source_role='fit',locked_split='train',
        data_sha256=meta['windows_sha256'],weak_targets='Original binary fight and first annotated onset <=3s; not physical contact',
        source_selection='All quality-matched fit windows; dev/val/test excluded',
        teacher_scores_are_independent=False,verified_precursor_ground_truth_created=False,
        training_performed=False,threshold_tuning=False,
        eligibility=dict(verified_evaluation=False,strict_training=False),
        rules='Fixed diagnostic .5 active-support/.35 quiet-negative thresholds; existing v4 observed-only approach/arm evidence')
    write_json(out/'protocol.json',config);torch.set_num_threads(2)
    paths=dict(baseline='weights/baseline_weak_onset_guarded_v1.pt',candidate='weights/model_optimization_v1/selected_validation_reference.pt')
    predictions={}
    for name,path in paths.items():
        model,ck=load_model(path);scored=score(model,DataLoader(Windows(data,'fit'),batch_size=32),ck['variant'])
        assert np.array_equal(scored['indices'].numpy(),ix)
        predictions[name]=scored['logits'].sigmoid().numpy()
    entries=json.loads(Path('data/pose_cache/advanced_v1/train_index.json').read_text())['entries']
    for p in Path('data/pose_cache/model_improvement_expanded_v1/completed').glob('*.json'):
        d=json.loads(p.read_text());entries[d['video_id']]=d['entry']
    records=[];current=None;arrays=None
    for n,i in enumerate(ix):
        key=str(data['video_id'][i]);t=float(data['cutoff_s'][i]);entry=entries[key]
        if current!=key:
            if sha256(entry['path'])!=entry['cache_sha256']:raise ValueError('Source cache changed')
            arrays,cache_meta=load_cache(entry['path']);current=key
            if cache_meta['video_id']!=key or cache_meta['source_sha256']!=locked[key]['content_sha256']:
                raise ValueError('Cache/locked source identity differs')
        j=int(np.searchsorted(arrays['t'],t+1e-7,side='right')-1);persons=[];before={}
        if j<0 or arrays['t'][j]>t+1e-7:raise ValueError('Proposal lacks a past/current pose frame')
        def people(frame):
            a,b=arrays['frame_offsets'][frame:frame+2];result=[]
            for k in range(a,b):
                kp=arrays['raw_keypoints'][k];box=arrays['boxes'][k]
                result.append(person_from_pose(dict(id=int(arrays['track_ids'][k]),xy=kp[:,:2],
                    conf=arrays['raw_confidence'][k],center=(box[:2]+box[2:])/2,height=box[3]-box[1])))
            return result
        persons=people(j)
        dt=float(arrays['t'][j]-arrays['t'][j-1]) if j else None
        if j and dt<=.25 and arrays['segment_id'][j]==arrays['segment_id'][j-1]:before={p['id']:p for p in people(j-1)}
        evidence=[]
        for a_pos,a in enumerate(persons):
            for b in persons[a_pos+1:]:
                e=interaction_evidence(a,b,before.get(a['id']),before.get(b['id']),dt)
                if e:evidence.append(e)
        motion=any(e['quality']>=.55 and (e['approach'] or e['arm_intent']) for e in evidence)
        y=data['targets'][i];mask=data['target_mask'][i]
        kind,target=proposal_kind(bool(y[0]),bool(y[1]),mask,bool(json.loads(locked[key]['intervals'])),float(predictions['candidate'][n,1]),motion)
        records.append(dict(proposal_id=f'{key}:{t:.6f}',video_id=key,split='train',internal_role='fit',
            cutoff_s=t,proposal_kind=kind,provisional_active_target=target,
            label_status='provisional_automatic',source_annotation_status='weak',
            weak_onset_target=int(y[0]),weak_onset_known=bool(mask[0]),weak_active_target=int(y[1]),weak_active_known=bool(mask[1]),
            baseline_onset_score=float(predictions['baseline'][n,0]),candidate_active_score=float(predictions['candidate'][n,1]),
            observed_motion_cue=motion,pose_evidence=json.dumps(evidence),
            verified_first_contact_s=None,human_review=False,eligible_for_verified_evaluation=False,eligible_for_strict_training=False,
            source_sha256=locked[key]['content_sha256'],cache_sha256=entry['cache_sha256']))
    write_json(out/'proposals.json',dict(schema=config['schema'],records=records,reviewer_decisions=[],
        scientific_status='Provisional research proposals; no human/verified labels; not independent teacher evidence'))
    write_csv(out/'proposals.csv',records)
    summary=dict(windows=len(records),unique_fit_sources=len(set(r['video_id'] for r in records)),
        counts=dict(Counter(r['proposal_kind'] for r in records)),teacher_checkpoint_sha256={k:sha256(p) for k,p in paths.items()},
        new_verified_labels=0,training_performed=False,test_sources_used=0,
        decision='No self-training: teacher predictions reuse fitting sources and lack independent quality validation')
    write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='outputs/pp1_v1/provisional_labels');run(p.parse_args())
