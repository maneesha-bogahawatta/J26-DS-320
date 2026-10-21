"""Audit source/cached evidence, trace F9, and plot frozen optimization results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.advanced_experiment import Windows, load_windows, metrics, score
from src.diagnose_stgcn import independent_targets, ranking
from src.foundation_io import read_csv, sha256, write_csv, write_json
from src.lock_splits import load_locked
from src.optimize_model import (BASELINE, HEADWISE, HISTORICAL_ADVANCED, HEADS, F9,
    CONFIGS, evaluate_head, load_model)
from src.pose_dataset import sequence_window
from src.pose_sequences import load_cache
from src.predict_advanced import CausalDecision, score_cache


def label_and_cache_audit(out):
    meta=json.loads(Path('data/pose_cache/advanced_v1/windows/metadata.json').read_text())
    rows=load_locked(meta['locked_manifest_path']);by_id={r['video_id']:r for r in rows}
    reports={}
    for name,folder in [('original','data/pose_cache/advanced_v1/windows'),
                        ('expanded','data/pose_cache/model_improvement_expanded_v1/windows')]:
        data,metadata=load_windows(folder);mismatches=0;future_frames=0;counts={}
        for i,(key,t) in enumerate(zip(data['video_id'],data['cutoff_s'])):
            y,mask=independent_targets(float(t),json.loads(by_id[key]['intervals']))
            mismatches+=int(not np.array_equal(y,data['targets'][i]) or not np.array_equal(mask,data['target_mask'][i]))
            times=data['frame_times'][i,data['temporal_mask'][i]]
            future_frames+=int(bool((times>t+1e-7).any()))
            if len(times)>1 and (np.diff(times)<=0).any():raise ValueError('Nonmonotonic window time')
        for role in ('fit','dev','val'):
            use=data['role']==role
            counts[role]=dict(windows=int(use.sum()),sources=len(set(data['video_id'][use].tolist())),
                heads={h:dict(known=int(data['target_mask'][use,j].sum()),positive=int((data['targets'][use,j]*data['target_mask'][use,j]).sum())) for j,h in enumerate(HEADS)})
        if mismatches or future_frames:raise ValueError('Label or causal alignment failure')
        reports[name]=dict(label_mismatches=mismatches,future_frame_windows=future_frames,counts=counts,
            dataset_sha256=metadata['windows_sha256'])
    entries=[]
    for split in ('train','val'):
        index=json.loads(Path(f'data/pose_cache/advanced_v1/{split}_index.json').read_text())
        entries.extend((key,split,value) for key,value in index['entries'].items())
    for path in Path('data/pose_cache/model_improvement_expanded_v1/completed').glob('*.json'):
        record=json.loads(path.read_text());entries.append((record['video_id'],'train',record['entry']))
    for key,split,entry in entries:
        if by_id[key]['split']!=split or split=='test':raise ValueError('Wrong source role')
        if sha256(entry['path'])!=entry['cache_sha256']:raise ValueError('Cache hash changed')
        arrays,metadata=load_cache(entry['path'])
        if metadata['video_id']!=key or metadata['source_sha256']!=by_id[key]['content_sha256']:raise ValueError('Source differs')
    reports['cache_integrity']=dict(index_entries_checked=len(entries),unique_caches=len(set(e[2]['path'] for e in entries)),
        schema_and_arrays_valid=True,test_sources_used=0)
    reports['f9_membership']={k:by_id[F9][k] for k in ('video_id','split','split_locked','test_list_member','content_sha256','leakage_group','duration_s','intervals')}
    write_json(out/'data_audit.json',reports)
    return by_id


def image_alignment(parent,derived):
    """Descriptive image-only crop alignment; no labels or fitting involved."""
    import cv2
    def small(frame):return cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(80,45)).astype(float)
    clip=cv2.VideoCapture(str(derived));fps=clip.get(cv2.CAP_PROP_FPS);anchors=[]
    for t in (0.,2.,6.):
        clip.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=clip.read()
        if not ok:raise ValueError('Clip anchor decode failed')
        anchors.append(small(frame))
    clip.release();cap=cv2.VideoCapture(str(parent));source_fps=cap.get(cv2.CAP_PROP_FPS);source=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        source.append(small(frame))
    cap.release();source=np.stack(source);offsets=[round(t*source_fps) for t in (0.,2.,6.)]
    errors=sum(np.abs(source[offset:len(source)-max(offsets)+offset]-anchor).mean((1,2)) for offset,anchor in zip(offsets,anchors))/len(anchors)
    best=int(errors.argmin());order=np.argsort(errors)
    return dict(method='Minimum mean absolute grayscale pixel error at three anchors (0,2,6 seconds), original frames searched',
        estimated_parent_offset_s=best/source_fps,source_frame=best,mae_8bit=float(errors[best]),
        next_best_offset_s=float(order[1]/source_fps),next_best_mae=float(errors[order[1]]),
        source_fps=source_fps,derived_fps=fps,
        interpretation='Image-matched estimate, not an authoritative edit-decision provenance record; derived clip is not independent data')


def simulation(records,thresholds):
    decision=CausalDecision(thresholds);result=[]
    for row in records:
        scores=[row['weak_onset_score'],row['fight_score']] if row['valid_pose'] else None
        status,smoothed,notification=decision.update(row['t_s'],scores,row['valid_pose'],row['segment_id'],json.loads(row['track_ids_json']))
        result.append(dict(status=status,notification=notification,smoothed=smoothed))
    return result


def trace_f9(out,by_id,evaluation):
    index=json.loads(Path('data/pose_cache/advanced_v1/val_index.json').read_text())
    paths=dict(original=Path(index['entries'][F9]['path']),
        derived=next(Path('data/pose_cache/inference/b96502dd_f_9_1_2_0_0_review').glob('*.npz')))
    _,derived_meta=load_cache(paths['derived'])
    alignment=image_alignment(by_id[F9]['video'],derived_meta['source_video'])
    write_json(out/'f9'/'image_alignment.json',alignment)
    summary={};model_paths=dict(baseline=BASELINE,advanced=HISTORICAL_ADVANCED,headwise=HEADWISE,readout=evaluation['reference_checkpoint'])
    for context,cache in paths.items():
        arrays,meta=load_cache(cache);summary[context]={}
        for name,checkpoint in model_paths.items():
            records,alerts,_,ck,timing=score_cache(checkpoint,cache,research_alerts=False,
                expected_source=dict(video_id=meta['video_id'],source_sha256=meta['source_sha256']))
            policies=evaluation['cohorts']['quality_matched'][name]
            tuned=[policies[head]['policies']['fpr10']['selected_without_f9']['threshold'] for head in HEADS]
            reference=simulation(records,ck['thresholds'])
            counterfactual=simulation(records,tuned)
            enriched=[]
            for row,ref,cf in zip(records,reference,counterfactual):
                i=int(np.searchsorted(arrays['t'],row['t_s']+1e-8,side='right')-1)
                sample=sequence_window(arrays,i);begin,end=arrays['frame_offsets'][i:i+2]
                item=dict(row,source_frame=int(arrays['frame_index'][i]),
                    detected_tracks=int(end-begin),selected_track_ids=json.dumps(sample['selected_ids'].tolist()),
                    currently_observed_people=int((sample['observed_mask'][-1].sum(0)>=6).sum()),
                    imputed_selected_joints=int(sample['imputed_mask'][-1].sum()),
                    history_gate=bool(row['observed_history_frames']>=8),quality_gate=bool(row['quality']>=.45),
                    current_observation_gate=bool(row['current_observed_joints']>=6),
                    reference_onset_threshold=float(ck['thresholds'][0]),reference_active_threshold=float(ck['thresholds'][1]),
                    effective_active_threshold=float(ck['thresholds'][1] if ck['research_alert_heads_enabled'][1] else 1.000001),
                    active_head_enabled=bool(ck['research_alert_heads_enabled'][1]),
                    reference_both_heads_enabled_status=ref['status'],
                    validation_without_f9_onset_threshold=float(tuned[0]),validation_without_f9_active_threshold=float(tuned[1]),
                    counterfactual_both_heads_enabled_status=cf['status'],
                    estimated_parent_time_s=float(row['t_s']+(alignment['estimated_parent_offset_s'] if context=='derived' else 0)),
                    parent_time_is_estimated=context=='derived')
                if context=='original':
                    targets,masks=independent_targets(row['t_s'],json.loads(by_id[F9]['intervals']))
                    item.update(onset_y=targets[0],active_y=targets[1],onset_known=masks[0],active_known=masks[1])
                enriched.append(item)
            write_csv(out/'f9'/f'{context}_{name}_stages.csv',enriched)
            valid=[r for r in records if r['valid_pose']]
            summary[context][name]=dict(timing=timing,cache_sha256=sha256(cache),source_sha256=meta['source_sha256'],
                source_context=[meta['extracted_start_s'],meta['extracted_end_s']],
                unique_track_ids=len(set(arrays['track_ids'].tolist())),identity_resets=sum(r['identity_reset'] for r in records),
                scene_cuts=int(arrays['scene_cut'].sum()),
                history_gate_failures=sum(not r['history_gate'] for r in enriched),
                quality_gate_failures=sum(not r['quality_gate'] for r in enriched),
                current_observation_gate_failures=sum(not r['current_observation_gate'] for r in enriched),
                valid_coverage=len(valid)/len(records),
                peak_raw_scores=[max(r[k] for r in valid) for k in ('weak_onset_score','fight_score')],
                peak_smoothed_scores=[max(r[k] for r in valid) for k in ('smoothed_onset_score','smoothed_fight_score')],
                research_heads_enabled=ck['research_alert_heads_enabled'],reference_thresholds=ck['thresholds'],
                effective_thresholds=[t if enabled else 1.000001 for t,enabled in zip(ck['thresholds'],ck['research_alert_heads_enabled'])],
                actual_candidate_decisions=sum(r['candidate_status'].endswith('_ALERT') for r in records),
                actual_notifications=len(alerts),
                counterfactual_reference_both_enabled_candidate_decisions=sum(r['status'].endswith('_ALERT') for r in reference),
                counterfactual_validation_without_f9_thresholds=tuned,
                counterfactual_validation_both_enabled_candidate_decisions=sum(r['status'].endswith('_ALERT') for r in counterfactual),
                note='Counterfactual simulations are diagnostic only; no notification sent, no eligibility/default modified')
    write_json(out/'f9'/'summary.json',summary)
    return summary


def ablations(out):
    data,meta=load_windows('data/pose_cache/model_improvement_expanded_v1/windows');summary={};rows=[]
    for name,loss,groups,frames in CONFIGS:
        table=read_csv(out/'runs'/name/'predictions.csv')
        val=[r for r in table if r['role']=='val' and r['known']=='True']
        yy=np.array([int(r['y']) for r in val]);pp=np.array([float(r['score']) for r in val]);sources=np.array([r['video_id'] for r in val])
        report,_=evaluate_head(yy,pp,sources,.5);summary[name]=report
        for policy,entry in report['policies'].items():
            rows.append(dict(config=name,loss=loss,groups=','.join(groups),frames=frames,policy=policy,
                auroc=report['auroc'],average_precision=report['average_precision'],pr_auc_trapezoid=report['pr_auc_trapezoid'],
                **entry['source_crossfit']['metrics']))
    write_json(out/'ablation_validation.json',summary);write_csv(out/'ablation_validation.csv',rows)
    return summary


def candidate_runtime_audit(out):
    """Every small arm also receives a real full-window inference/load check."""
    from torch.utils.data import DataLoader
    selection=json.loads((out/'selection.json').read_text())
    model,_=load_model(selection['checkpoint'])
    data,meta=load_windows('data/pose_cache/model_improvement_expanded_v1/windows')
    weight_dir=Path(selection['checkpoint']).parent;results={}
    for name,loss,groups,frames in CONFIGS:
        saved=torch.load(weight_dir/(name+'.pt'),map_location='cpu',weights_only=True)['readout']
        for dest,key in [('readout_weight','weights'),('readout_bias','biases'),('feature_mean','mean'),('feature_scale','scale'),('feature_mask','mask')]:
            getattr(model,dest).copy_(saved[key])
        model.history_frames.fill_(frames)
        started=time.monotonic()
        scored=score(model,DataLoader(Windows(data,'val'),batch_size=32), 'optimized_readout')
        elapsed=time.monotonic()-started;p=scored['logits'][:,1].sigmoid().numpy()
        cached=[r for r in read_csv(out/'runs'/name/'predictions.csv') if r['role']=='val']
        expected=np.array([float(r['score']) for r in cached]);difference=float(np.max(np.abs(expected-p)))
        if difference>2e-6:raise ValueError('Frozen feature readout and runtime disagree')
        results[name]=dict(seconds=elapsed,windows=len(p),ms_per_window=1000*elapsed/len(p),
            batch_size=32,threads=2,history_frames=frames,validation_prevalence=float(data['targets'][data['role']=='val',1].mean()),
            eligible_coverage=204/320,pose_extraction_included=False,
            cached_vs_runtime_max_abs_score_difference=difference,
            active_reference_half=ranking(data['targets'][data['role']=='val',1],p))
    write_json(out/'candidate_runtime.json',results)
    return results


def lead_times(out,by_id):
    rows=read_csv(out/'validation_predictions.csv');results={}
    for model in ('baseline','advanced','headwise','readout'):
        use=[r for r in rows if r['cohort']=='quality_matched' and r['model']==model and r['head']==HEADS[0]]
        summaries={}
        for policy in ('f1','fpr10','precision50'):
            values=[];positive_sources=set();warned=set()
            for r in use:
                if int(r['y'])!=1:continue
                positive_sources.add(r['video_id'])
                if r[policy+'_crossfit_prediction']!='True':continue
                onset=json.loads(by_id[r['video_id']]['intervals'])[0][0]
                lead=onset-float(r['cutoff_s']);assert 0<lead<=3
                values.append(lead);warned.add(r['video_id'])
            summaries[policy]=dict(positive_source_count=len(positive_sources),warned_sources=len(warned),
                true_positive_windows=len(values),lead_time_s_quantiles=np.quantile(values,[0,.5,1]).tolist() if values else None,
                semantics='Window threshold crossings before first weak binary annotation, not confirmed candidate alerts or verified pre-contact warnings')
        results[model]=summaries
    write_json(out/'weak_onset_lead_times.json',results)


def plot_results(out,evaluation,ablations_result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    plots=out/'plots';plots.mkdir(exist_ok=True)
    data=read_csv(out/'validation_predictions.csv');models=('baseline','advanced','headwise','readout')
    colors=dict(zip(models,('#4765a5','#c88e28','#7e51a8','#15866c')))
    def values(model,head):
        rows=[r for r in data if r['cohort']=='quality_matched' and r['model']==model and r['head']==head]
        return np.array([int(r['y']) for r in rows]),np.array([float(r['score']) for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for h,ax in zip(HEADS,axes):
        for model in models:
            y,p=values(model,h);order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1]
            tp=y[order].cumsum()[ends];recall=tp/y.sum();precision=tp/(ends+1)
            m=evaluation['cohorts']['quality_matched'][model][h]
            ax.plot(np.r_[0,recall],np.r_[1,precision],color=colors[model],label=f"{model}: AP {m['average_precision']:.3f}")
        ax.axhline(y.mean(),ls='--',color='gray',label=f'prevalence {y.mean():.3f}')
        ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02),title=h.replace('weak_','').replace('_',' '));ax.legend(fontsize=8)
    fig.suptitle('Unchanged quality-matched validation: 13 sources; exploratory weak labels');fig.tight_layout();fig.savefig(plots/'precision_recall.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,model in zip(axes.flat,models):
        y,p=values(model,HEADS[1]);ax.hist([p[y==0],p[y==1]],bins=np.linspace(0,1,26),label=['weak non-fight','weak fight'],color=['#6584b6','#cc6654'],alpha=.8)
        m=evaluation['cohorts']['quality_matched'][model][HEADS[1]]
        for policy,style in [('f1','--'),('fpr10',':')]:ax.axvline(m['policies'][policy]['selected_without_f9']['threshold'],ls=style,color='black',label=policy+' threshold (no F9)')
        ax.set(title=model,xlabel='Uncalibrated active-fight score',ylabel='Windows');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'score_distributions.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for ax,model in zip(axes.flat,models):
        m=evaluation['cohorts']['quality_matched'][model][HEADS[1]]['policies']['fpr10']['source_crossfit']['metrics']
        matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=162)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',fontsize=17,color='white' if v>90 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['negative','positive'],yticklabels=['negative','positive'],xlabel='Predicted',ylabel='Weak annotation',title=f"{model}: F1 {m['f1']:.3f}, recall {m['recall']:.3f}")
    fig.suptitle('Active fight: source-held-out thresholds tuned to FPR <= 10%\nConstraint is on tuning sources; held-out FPR can exceed it');fig.tight_layout();fig.savefig(plots/'confusion_matrices.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    old=json.loads(Path('outputs/model_improvement_expanded_v1/stgcn/advanced_class_weighted/training.json').read_text())
    for role in ('fit','development'):
        axes[0].plot([r['epoch'] for r in old['history']],[r[role]['heads'][HEADS[1]]['bce'] for r in old['history']],label=role)
    axes[0].set(title='Previous full ST-GCN: active unweighted BCE',xlabel='Epoch',ylabel='BCE');axes[0].legend()
    selected=json.loads((out/'selection.json').read_text())['selected']
    for seed in (73,101,137):
        report=json.loads((out/'runs'/selected/f'seed_{seed}.json').read_text());hist=report['history']
        axes[1].plot([r['epoch'] for r in hist],[r['fit']['bce'] for r in hist],label=f'fit {seed}')
        axes[1].plot([r['epoch'] for r in hist],[r['dev']['bce'] for r in hist],ls='--',label=f'dev {seed}')
    axes[1].set(title='Frozen encoder + selected weighted readout',xlabel='Epoch',ylabel='Unweighted BCE (comparable diagnostic)');axes[1].legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(plots/'training_validation_loss.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,h in zip(axes,HEADS):
        ax.plot([0,1],[0,1],ls='--',color='gray')
        for model in models:
            bins=evaluation['cohorts']['quality_matched'][model][h]['calibration']['bins'];nonempty=[b for b in bins if b['n']]
            ax.plot([b['mean_score'] for b in nonempty],[b['observed_fraction'] for b in nonempty],marker='o',color=colors[model],label=model)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean raw score',ylabel='Weak positive fraction',title=h.replace('weak_','').replace('_',' '));ax.legend(fontsize=8)
    fig.suptitle('Raw score reliability (10 fixed bins); sparse bins are uncertain');fig.tight_layout();fig.savefig(plots/'calibration.png');plt.close(fig)
    selection=json.loads((out/'selection.json').read_text());names=[c[0] for c in CONFIGS];fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(names));dev=[selection['runs'][n]['mean_seed_dev_ap'] for n in names];err=[selection['runs'][n]['sd_seed_dev_ap'] for n in names]
    ax.bar(x-.18,dev,.36,yerr=err,label='Mean seed internal-dev AP (SD)',color='#4765a5')
    ax.bar(x+.18,[ablations_result[n]['average_precision'] for n in names],.36,label='Ensemble validation AP (descriptive only)',color='#15866c')
    ax.axhline(42/204,color='gray',ls='--',label='Validation prevalence');ax.set(xticks=x,xticklabels=names,ylabel='Average precision',title='Frozen encoder: controlled loss / temporal / feature readout ablations')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(plots/'ablation_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    for ax,context in zip(axes,('original','derived')):
        for model in ('baseline','headwise','readout'):
            rows=read_csv(out/'f9'/f'{context}_{model}_stages.csv');good=[r for r in rows if r['valid_pose']=='True']
            ax.plot([float(r['t_s']) for r in good],[float(r['fight_score']) for r in good],label=model,color=colors[model])
            if model=='readout':
                ax.axhline(float(rows[0]['validation_without_f9_active_threshold']),ls='--',color=colors[model],label='Readout threshold from other validation sources')
        invalid=[float(r['t_s']) for r in rows if r['valid_pose']!='True'];ax.scatter(invalid,[.02]*len(invalid),marker='x',color='red',label='Pose abstention')
        ax.set(ylim=(0,1),xlabel='Original source time (s)' if context=='original' else 'Derived clip-relative time (s)',ylabel='Active-fight raw score',title='F9 '+context+' context');ax.legend(fontsize=8,ncol=3)
    fig.tight_layout();fig.savefig(plots/'f9_scores.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='outputs/model_optimization_v1');args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out_dir);evaluation=json.loads((out/'evaluation.json').read_text())
    by_id=label_and_cache_audit(out);print('Data/cache audit complete',flush=True)
    summary=trace_f9(out,by_id,evaluation);print(json.dumps(dict(stage='F9',summary=summary)),flush=True)
    ablation=ablations(out);candidate_runtime_audit(out);lead_times(out,by_id);plot_results(out,evaluation,ablation)
    print('Ablations, weak lead times and seven diagnostic plots complete',flush=True)


if __name__=='__main__':main()
