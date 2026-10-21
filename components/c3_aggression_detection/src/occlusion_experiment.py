"""Fixed occlusion comparisons on existing weak validation caches, no fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.advanced_data import weak_targets
from src.causal_pose import CausalPosePreprocessor
from src.diagnose_stgcn import ranking
from src.foundation_io import sha256, write_csv, write_json
from src.lock_splits import load_locked
from src.occlusion import GuardedPosePreprocessor
from src.optimize_model import load_model
from src.pose_enrichment import box_iou
from src.pose_sequences import DETECTION_ARRAYS, load_cache
from src.predict_advanced import score_arrays


BASELINE='weights/baseline_weak_onset_guarded_v1.pt'
CANDIDATE='weights/model_optimization_v1/selected_validation_reference.pt'
SCENARIOS=('one_joint','multiple_joints','lower_body','complete_occlusion','overlap','fast_motion','id_change','long_gap')


def frames_from_cache(arrays):
    frames=[]
    for i,t in enumerate(arrays['t']):
        a,b=arrays['frame_offsets'][i:i+2]
        raw=arrays['raw_keypoints'][a:b].copy();raw[:,:,2]=arrays['raw_confidence'][a:b]
        frames.append(dict(t=float(t),frame_index=int(arrays['frame_index'][i]),keypoints=raw,
            boxes=arrays['boxes'][a:b].copy(),track_ids=arrays['track_ids'][a:b].copy(),scene_cut=bool(arrays['scene_cut'][i])))
    return frames


def reprocess(arrays,meta,mode='default',scenario=None):
    frames=frames_from_cache(arrays);overlap_frames=0;applied_frames=0
    processor=CausalPosePreprocessor() if mode=='default' else GuardedPosePreprocessor(motion=mode=='motion')
    values={key:[] for key in DETECTION_ARRAYS};offsets=[0];audits=[];kinds=[];errors=[];speed=[];previous={}
    start=len(frames)//2;stop=start+6
    for i,frame in enumerate(frames):
        raw=frame['keypoints'];ids=frame['track_ids'];boxes=frame['boxes'];reference=raw.copy()
        overlapping=any(box_iou(a,b)>.2 for j,a in enumerate(boxes) for b in boxes[j+1:]);overlap_frames+=overlapping
        hidden=np.zeros(raw.shape[:2],bool)
        if scenario and len(ids):
            if scenario=='id_change' and i>=start:frame['track_ids']=ids+100000
            elif scenario=='complete_occlusion' and start<=i<stop:hidden[:]=True
            elif scenario=='long_gap' and start<=i<start+12:hidden[:,9]=True
            elif scenario=='one_joint' and start<=i<stop:hidden[:,9]=True
            elif scenario=='multiple_joints' and start<=i<stop:hidden[:,[7,8,9,10]]=True
            elif scenario=='lower_body' and start<=i<start+10:hidden[:,11:17]=True
            elif scenario=='overlap' and overlapping and i%12 in (5,6):hidden[:,[9,10]]=True
            elif scenario=='fast_motion':
                for j,pid in enumerate(ids):
                    old=previous.get(int(pid));height=max(1.,boxes[j,3]-boxes[j,1])
                    if old is not None and np.linalg.norm(raw[j,9,:2]-old)>height*.02:hidden[j,9]=True
        for j,pid in enumerate(ids):previous[int(pid)]=reference[j,9,:2].copy()
        if scenario=='id_change' and i==start:
            # A newly assigned identity with a hidden wrist must never inherit old coordinates.
            hidden[:,9]=True
        if hidden.any():raw[:,:,2][hidden]=0;applied_frames+=1
        started=time.perf_counter();out=processor.update(raw,boxes,frame['track_ids'],frame['t'],meta['frame_wh'],frame['scene_cut']);speed.append(time.perf_counter()-started)
        out['raw_keypoints']=raw
        for key in values:values[key].append(out[key])
        offsets.append(offsets[-1]+len(ids));kinds.append(out.get('estimate_kind',np.where(out['observed'],1,np.where(out['imputed'],2,0))))
        if mode!='default':audits.extend(processor.audit['resets'])
        # Artificially held-out high-confidence detector outputs are proxy targets,
        # not motion capture/human keypoint ground truth.
        proxy=hidden&(reference[:,:,2]>=.6)&out['imputed']
        if proxy.any():errors.extend((np.linalg.norm(out['pixel_keypoints'][:,:,:2]-reference[:,:,:2],axis=-1)/np.maximum(1.,boxes[:,3]-boxes[:,1])[:,None])[proxy].tolist())
    result={key:np.concatenate(value) if value else np.empty_like(arrays[key]) for key,value in values.items()}
    result.update({key:arrays[key].copy() for key in ('t','frame_index','segment_id','scene_cut')});result['frame_offsets']=np.array(offsets,np.int64)
    observed=result['observed'];estimated=result['imputed'];visible=observed|estimated
    info=dict(raw_observed_joint_fraction=float(observed.mean()) if observed.size else 0.,
        estimated_joint_fraction=float(estimated.mean()) if estimated.size else 0.,
        missing_joint_fraction=float((~visible).mean()) if visible.size else 1.,
        raw_confident_hidden_joint_proxy_errors=len(errors),proxy_error_body_height_mean=float(np.mean(errors)) if errors else None,
        preprocessing_ms_per_frame=1000*sum(speed)/len(speed),frames=len(frames),
        perturbed_frames=applied_frames,overlapping_box_frames=overlap_frames,
        resets=audits,motion_estimates=int((np.concatenate(kinds)==3).sum()) if kinds else 0,
        finite_features=bool(np.isfinite(result['keypoints']).all()),
        masks_disjoint=not bool((observed&estimated).any()))
    # Temporal stability is a descriptive jump between same-ID, currently visible
    # output joints in image-normalized coordinates (no cross-person derivative).
    jumps=[];last={}
    for i,t in enumerate(result['t']):
        a,b=result['frame_offsets'][i:i+2]
        for j in range(a,b):
            pid=int(result['track_ids'][j]);kp=result['keypoints'][j];mask=visible[j];old=last.get(pid)
            if old is not None:
                common=mask&old[1]
                if common.any():jumps.extend(np.linalg.norm(kp[common,:2]-old[0][common,:2],axis=-1).tolist())
            last[pid]=(kp,mask)
    info['mean_same_track_normalized_joint_step']=float(np.mean(jumps)) if jumps else None
    return result,info


def summarize(records,intervals,thresholds):
    result={}
    for h,name in enumerate(('weak_onset','active_fight')):
        values=[]
        for r in records:
            y,m=weak_targets(r['t_s'],intervals[r['video_id']])
            if m[h]:values.append((int(y[h]),r))
        valid=[(y,r) for y,r in values if r['valid_pose']]
        if valid:
            yy=np.array([y for y,r in valid]);pp=np.array([r['weak_onset_score' if h==0 else 'fight_score'] for y,r in valid])
            from src.advanced_experiment import metrics
            report=ranking(yy,pp);report.update(metrics(yy,pp,thresholds[h]))
        else:report=dict(n=0)
        report.update(known_decisions_including_abstentions=len(values),pose_abstentions=len(values)-len(valid),
            positive_abstentions=sum(y for y,r in values if not r['valid_pose']),
            coverage=len(valid)/max(1,len(values)))
        result[name]=report
    return result


def run(args):
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    manifest=load_locked('data/foundation/v1/ubi_locked.csv');rows={r['video_id']:r for r in manifest}
    index=json.loads(Path('data/pose_cache/advanced_v1/val_index.json').read_text())
    if index['locked_manifest_sha256']!=sha256('data/foundation/v1/ubi_locked.csv'):
        raise ValueError('Validation index differs from locked split manifest')
    protocol=dict(schema='c3_occlusion_comparison_v1',modes=['default','guarded','motion'],
        params=dict(min_current_anchors_for_carry=6,min_motion_anchors=8,motion_max_age_s=.134,
            max_gap_s=.25,box_discontinuity_iou=.05,max_motion_body_diagonals_per_second=2),
        threshold_tuning=False,weights_changed=False,old_cache_overwrite=False,test_sources_used=0,
        stress_sources=['f_101_1_2_0_0','f_147_1_0_1_0','n_360_0_0_1_0','f_9_1_2_0_0'],
        stress_scenarios=SCENARIOS,selection='Frozen diagnostic sources; no F9-specific parameters',
        caution='Masked detector keypoints are proxy targets, not independently measured pose ground truth')
    write_json(out/'protocol.json',protocol);torch.set_num_threads(2)
    models={name:load_model(path) for name,path in [('baseline',BASELINE),('candidate',CANDIDATE)]}
    data={};infos=[];score_rows=[];all_real={};stress=[]
    for key,entry in index['entries'].items():
        if rows[key]['split']!='val' or sha256(entry['path'])!=entry['cache_sha256']:raise ValueError('Bad validation identity/cache')
        arrays,meta=load_cache(entry['path']);data[key]=(arrays,meta)
        if meta['video_id']!=key or meta['source_sha256']!=rows[key]['content_sha256']:
            raise ValueError('Cache and locked source identity differ')
        for mode in protocol['modes']:
            derived,info=reprocess(arrays,meta,mode);info.update(video_id=key,mode=mode,scenario='native');infos.append(info)
            if mode=='default':
                delta=float(np.max(np.abs(derived['keypoints']-arrays['keypoints']))) if len(arrays['keypoints']) else 0.
                if delta>1e-6:raise ValueError('Default raw replay disagrees with original cache')
                info['default_cache_max_abs_difference']=delta
            for name,(model,ck) in models.items():
                records,_,_,_,timing=score_arrays(model,ck,derived,meta)
                for r in records:r.update(video_id=key,mode=mode,model=name,scenario='native')
                score_rows.extend(records);all_real.setdefault((mode,name),[]).extend(records)
        print(f'Native occlusion comparison {key}',flush=True)
    intervals={key:json.loads(rows[key]['intervals']) for key in data}
    reports={mode:{name:summarize(all_real[(mode,name)],intervals,ck['thresholds']) for name,(model,ck) in models.items()} for mode in protocol['modes']}
    for key in protocol['stress_sources']:
        arrays,meta=data[key]
        for scenario in SCENARIOS:
            for mode in protocol['modes']:
                derived,info=reprocess(arrays,meta,mode,scenario);info.update(video_id=key,scenario=scenario,mode=mode);infos.append(info)
                model,ck=models['baseline'];records,_,_,_,timing=score_arrays(model,ck,derived,meta)
                for r in records:r.update(video_id=key,scenario=scenario,mode=mode,model='baseline')
                stress.append(dict(video_id=key,scenario=scenario,mode=mode,
                    decisions=len(records),usable_decisions=sum(r['valid_pose'] for r in records),
                    identity_resets=sum(r['identity_reset'] for r in records),model_replay_seconds=timing['model_replay_seconds'],
                    metrics=summarize(records,{key:intervals[key]},ck['thresholds'])))
        print(f'Stress scenarios complete {key}',flush=True)
    # Matched native cutoff metrics prevent increased abstention from masquerading as better accuracy.
    common={}
    for name,(model,ck) in models.items():
        maps={mode:{(r['video_id'],r['t_s']):r for r in all_real[(mode,name)]} for mode in protocol['modes']}
        keys=[k for k,r in maps['default'].items() if all(maps[m][k]['valid_pose'] for m in maps)]
        common[name]={m:summarize([maps[m][k] for k in keys],intervals,ck['thresholds']) for m in maps}
    write_json(out/'native_evaluation.json',dict(dynamic_coverage=reports,matched_observable_cutoffs=common,
        default_changed=False,experimental_preprocessing_promoted=False,test_sources_used=0))
    write_json(out/'pose_diagnostics.json',infos);write_json(out/'stress_evaluation.json',stress)
    write_csv(out/'native_scores.csv',score_rows)
    print(json.dumps(dict(native=reports,common=common)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='outputs/occlusion_v1');run(p.parse_args())
