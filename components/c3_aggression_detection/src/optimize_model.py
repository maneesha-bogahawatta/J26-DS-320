"""Controlled frozen-encoder experiments and source-held-out threshold analysis.

Run prepare, train, then evaluate. Fits/normalization/selection use locked train
fit/dev only. Validation thresholds explicitly exclude F9 and each scored source.
No locked test labels or runtime defaults are used for optimization.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.advanced_experiment import SCHEMA, Windows, build_model, load_windows, metrics, score
from src.diagnose_stgcn import ranking
from src.foundation_io import sha256, write_csv, write_json
from src.improve_model import audit_roles
from src.optimized_readout import OptimizedReadoutC3, feature_mask, normalize_features
from src.reliability_audit import fit_temperature, reliability_metrics, temperature_scale


HEADS = ('weak_first_onset_within_3s', 'weak_active_fight')
F9 = 'f_9_1_2_0_0'
BACKBONE = 'weights/model_improvement_expanded_v1/stgcn/advanced_class_weighted.pt'
BASELINE = 'weights/baseline_weak_onset_guarded_v1.pt'
HISTORICAL_ADVANCED = 'weights/advanced_weak_onset_guarded_v1.pt'
HEADWISE = 'weights/model_improvement_expanded_v1/headwise_guarded.pt'
SEEDS = (73, 101, 137)
ALL = ('joint', 'bone', 'velocity', 'geometry')
CONFIGS = (
    ('full_bce_48', 'bce', ALL, 48),
    ('full_weighted_48', 'weighted', ALL, 48),
    ('full_focal_48', 'focal', ALL, 48),
    ('full_bce_24', 'bce', ALL, 24),
    ('joint_bce_48', 'bce', ('joint',), 48),
    ('joint_bone_bce_48', 'bce', ('joint', 'bone'), 48),
    ('no_geometry_bce_48', 'bce', ('joint', 'bone', 'velocity'), 48),
)


def choose_threshold(y, p, policy='f1', *, role):
    """Exact tied-score cutoffs; only validation tuning is permitted here."""
    if role != 'val':
        raise ValueError('This protocol only permits validation thresholds')
    y = np.asarray(y, int); p = np.asarray(p, float)
    if y.ndim != 1 or p.shape != y.shape or len(np.unique(y)) != 2:
        raise ValueError('Two aligned validation classes required')
    if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all() or ((p<0)|(p>1)).any():
        raise ValueError('Finite binary labels and sigmoid scores required')
    if policy not in ('f1', 'fpr10', 'precision50'):
        raise ValueError('Unknown operating policy')
    candidates = []
    for threshold in np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]:
        row = metrics(y, p, threshold)
        row['false_positive_rate'] = row['fp']/max(1, row['fp']+row['tn'])
        allowed = policy == 'f1' or (policy == 'fpr10' and row['false_positive_rate'] <= .1+1e-12) or (
            policy == 'precision50' and row['precision'] >= .5 and row['tp']+row['fp'] > 0)
        if allowed:
            candidates.append(row)
    if not candidates:
        row = metrics(y, p, np.nextafter(p.max(), np.inf))
        row['false_positive_rate'] = 0.
        return dict(threshold=row['threshold'], feasible=False, metrics=row)
    best = max(candidates, key=lambda r: (r['f1'], r['balanced_accuracy'], r['threshold']))
    return dict(threshold=best['threshold'], feasible=True, metrics=best)


def cross_source_thresholds(y, p, sources, policy):
    """Each held-out source's labels cannot affect its threshold. F9 never tunes."""
    sources = np.asarray(sources); y = np.asarray(y); p = np.asarray(p)
    predictions = np.zeros(len(y), bool); thresholds = np.zeros(len(y)); folds = []
    for source in sorted(set(sources.tolist())):
        held = sources == source
        tune = (sources != source) & (sources != F9)
        point = choose_threshold(y[tune], p[tune], policy, role='val')
        predictions[held] = p[held] >= point['threshold']; thresholds[held] = point['threshold']
        folds.append(dict(source=source, tuning_sources=sorted(set(sources[tune].tolist())),
                          tuning_n=int(tune.sum()), held_n=int(held.sum()), **point))
    report = metrics(y, predictions.astype(float), .5)
    # Binary decisions do not define a comparable model ranking.
    report.pop('auroc'); report.pop('average_precision'); report.pop('bce')
    report['threshold'] = 'source-specific; see folds'
    report['false_positive_rate'] = report['fp']/max(1, report['fp']+report['tn'])
    return dict(metrics=report, folds=folds), predictions, thresholds


def active_loss(logits, y, loss_name, pos_weight):
    if loss_name not in ('bce', 'weighted', 'focal'):
        raise ValueError('Unknown controlled loss')
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none',
        pos_weight=pos_weight if loss_name == 'weighted' else None)
    if loss_name == 'focal':
        # gamma=2, no alpha/class weighting or balanced sampling compounded with it.
        pt = torch.where(y.bool(), logits.sigmoid(), (-logits).sigmoid())
        bce = (1-pt).square()*bce
    return bce.mean()


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(ck['variant']).eval(); model.load_state_dict(ck['state'], strict=True)
    return model, ck


def features(backbone, data, frames):
    values = []
    with torch.no_grad():
        for role in ('fit', 'dev', 'val'):
            for batch in DataLoader(Windows(data, role), batch_size=32, shuffle=False):
                x = backbone.window_features(batch['sk'], batch['temporal_mask'], batch['joint_mask'], batch['frame_times'], frames)
                values.append((batch['index'].numpy(), x.numpy()))
    result = np.zeros((len(data['role']), 224), np.float32)
    for ix, x in values: result[ix] = x
    if not np.isfinite(result).all(): raise ValueError('Nonfinite encoder features')
    return result


def prepare(args):
    out = Path(args.out_dir)
    if (out/'protocol.json').exists(): raise FileExistsError('Use a new experiment version')
    data, meta = load_windows(args.data); audit_roles(data, meta)
    protocol = dict(schema='c3_model_optimization_v1',
        windows_sha256=meta['windows_sha256'], locked_manifest_sha256=meta['locked_manifest_sha256'],
        encoder=dict(path=BACKBONE, sha256=sha256(BACKBONE),
            reason='Highest active AUROC on expanded internal development among prior advanced checkpoints; no new validation family selection'),
        baseline_sha256=sha256(BASELINE),
        configs=[dict(name=n,loss=l,groups=g,frames=f) for n,l,g,f in CONFIGS], seeds=SEEDS,
        training=dict(epochs=80,patience=10,batch_size=128,lr=.003,weight_decay=.01,
            sampler='Uniform shuffled fitting windows without replacement',augmentation='None: identical frozen features in every controlled arm',
            normalization='Fitting-window mean/SD only',focal_gamma=2,focal_alpha=None,
            selection='Per seed max internal-dev active average precision, tie lower unweighted BCE; config max mean-seed dev AP, tie lower BCE',
            ensemble='Three selected seed heads averaged in probability space; frozen encoder'),
        validation=dict(threshold_policies=['f1','fpr10','precision50'],
            excluded_from_all_tuning=[F9],
            crossfit='Leave one source out of threshold tuning; F9 always excluded. Original and quality-matched distributions retained.',
            selection_bias='Previously inspected small validation cohort. Full-cohort tuned scores are selection statistics, not independent estimates.'),
        promotion_criteria=dict(quality_matched_active_source_crossfit_fpr10=dict(
            f1_gain_vs_rethresholded_baseline=.05,recall_gain=.10,precision_min=.5,fpr_max=.1,
            paired_source_bootstrap_f1_delta_lower_95_strictly_positive=True),
            active_auroc_not_below_existing_headwise=True,onset='Historical baseline unchanged',
            operational_eligibility='Never enabled by this experiment; separate independent event validation required'),
        test_sources_used=0,operational_alerts_enabled=False,verified_early_aggression_claim=False)
    write_json(out/'protocol.json',protocol)  # Frozen before computing any new result.
    backbone, ck = load_model(BACKBONE)
    if ck['feature_contract'] != meta['feature_contract']: raise ValueError('Feature contract differs')
    for frames in (48,24):
        started = time.monotonic(); values = features(backbone, data, frames)
        path = out/f'features_{frames}.npz'
        with path.open('xb') as stream: np.savez_compressed(stream, x=values)
        write_json(out/f'features_{frames}.json',dict(sha256=sha256(path),rows=len(values),
            data_sha256=meta['windows_sha256'],encoder_sha256=sha256(BACKBONE),seconds=time.monotonic()-started,
            history_frames=frames,labels_used_for_encoding=False,test_sources_used=0))
        print(json.dumps(dict(stage='features',frames=frames,rows=len(values),seconds=time.monotonic()-started)),flush=True)


def train_readout(x, y, fit, dev, loss_name, seed):
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    pos_weight = (1-y[fit]).sum()/y[fit].sum()
    best = (-np.inf, -np.inf); state = None; history = []; stale = 0; selected_epoch = None
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1,81):
        order = fit[torch.randperm(len(fit), generator=generator)]; losses=[]
        model.train()
        for indices in order.split(128):
            logits = model(x[indices])[:,0]; loss=active_loss(logits,y[indices],loss_name,pos_weight)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        with torch.no_grad():
            p = model(x)[:,0].sigmoid().numpy()
        fit_m = ranking(y[fit].numpy(),p[fit]); dev_m = ranking(y[dev].numpy(),p[dev])
        history.append(dict(epoch=epoch,optimization_loss=float(np.mean(losses)),fit=fit_m,dev=dev_m))
        key = (dev_m['average_precision'], -dev_m['bce'])
        if key > best:
            state = copy.deepcopy(model.state_dict()); best=key; stale=0; selected_epoch=epoch
        else: stale+=1
        if stale >= 10: break
    model.load_state_dict(state)
    with torch.no_grad(): p=model(x)[:,0].sigmoid().numpy()
    return model,p,dict(seed=seed,selected_epoch=selected_epoch,epochs=len(history),history=history,
        dev=ranking(y[dev].numpy(),p[dev]),fit_positive_weight=float(pos_weight))


def train(args):
    out=Path(args.out_dir); protocol=json.loads((out/'protocol.json').read_text())
    data,meta=load_windows(args.data);audit_roles(data,meta)
    if protocol['windows_sha256'] != meta['windows_sha256']:raise ValueError('Training data changed')
    fit=torch.from_numpy(np.flatnonzero((data['role']=='fit')&data['target_mask'][:,1]))
    dev=torch.from_numpy(np.flatnonzero((data['role']=='dev')&data['target_mask'][:,1]))
    if any(F9 in data['video_id'][ix].tolist() for ix in (fit,dev)):raise ValueError('F9 fitting prohibited')
    y=torch.from_numpy(data['targets'][:,1]); rows=[]; runs={}; candidates={}
    for name,loss_name,groups,frames in CONFIGS:
        feature_path=out/f'features_{frames}.npz'
        fm=json.loads((out/f'features_{frames}.json').read_text())
        if sha256(feature_path)!=fm['sha256']:raise ValueError('Feature cache changed')
        with np.load(feature_path,allow_pickle=False) as archive:x=torch.from_numpy(archive['x'].copy())
        mean=x[fit].mean(0);scale=x[fit].std(0,unbiased=False).clamp_min(1e-4);mask=feature_mask(groups)
        normalized=normalize_features(x,mean,scale,mask)
        weights=[];biases=[];probabilities=[];reports=[]
        for seed in SEEDS:
            started=time.monotonic();head,p,report=train_readout(normalized,y,fit,dev,loss_name,seed)
            report['seconds']=time.monotonic()-started
            write_json(out/'runs'/name/f'seed_{seed}.json',report)
            weights.append(head.weight.detach()[0]);biases.append(head.bias.detach()[0]);probabilities.append(p);reports.append(report)
            rows.append(dict(config=name,loss=loss_name,frames=frames,groups=','.join(groups),seed=seed,
                selected_epoch=report['selected_epoch'],seconds=report['seconds'],**report['dev']))
            print(json.dumps(dict(stage='train',config=name,seed=seed,dev_ap=report['dev']['average_precision'],
                dev_auc=report['dev']['auroc'],epoch=report['selected_epoch'])),flush=True)
        average=np.mean(probabilities,axis=0)
        candidates[name]=dict(weights=torch.stack(weights),biases=torch.stack(biases),mean=mean,scale=scale,mask=mask,
            frames=frames,probabilities=average)
        runs[name]=dict(config=dict(loss=loss_name,groups=groups,frames=frames),
            mean_seed_dev_ap=float(np.mean([r['dev']['average_precision'] for r in reports])),
            sd_seed_dev_ap=float(np.std([r['dev']['average_precision'] for r in reports])),
            mean_seed_dev_bce=float(np.mean([r['dev']['bce'] for r in reports])),
            ensemble_dev=ranking(y[dev].numpy(),average[dev]),
            selected_epochs=[r['selected_epoch'] for r in reports])
        path=Path(args.weights_dir)/(name+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:torch.save(dict(readout=candidates[name] | {'probabilities':None},protocol=protocol),stream)
        write_csv(out/'runs'/name/'predictions.csv',[
            dict(index=i,video_id=str(data['video_id'][i]),role=str(data['role'][i]),cutoff_s=float(data['cutoff_s'][i]),
                y=int(y[i]),known=bool(data['target_mask'][i,1]),score=float(average[i])) for i in range(len(y))])
    selected=max(runs,key=lambda n:(runs[n]['mean_seed_dev_ap'],-runs[n]['mean_seed_dev_bce']))
    chosen=candidates[selected];model=OptimizedReadoutC3()
    backbone,_=load_model(BACKBONE);onset,old=load_model(BASELINE)
    model.backbone.load_state_dict(backbone.state_dict());model.onset.load_state_dict(onset.state_dict())
    for dest,key in [('readout_weight','weights'),('readout_bias','biases'),('feature_mean','mean'),('feature_scale','scale'),('feature_mask','mask')]:
        getattr(model,dest).copy_(chosen[key])
    model.history_frames.fill_(chosen['frames'])
    ck=dict(schema=SCHEMA,variant='optimized_readout',state=model.state_dict(),
        thresholds=[old['thresholds'][0],1.000001],research_alert_heads_enabled=[False,False],
        operational_alerts_enabled=False,verified_early_aggression_claim=False,calibration='UNVALIDATED',
        supervision='weak_ubi_first_onset_and_binary_fight_only',trained_heads=list(HEADS),
        feature_contract=meta['feature_contract'],windows_sha256=meta['windows_sha256'],
        locked_manifest_sha256=meta['locked_manifest_sha256'],protocol=protocol,
        selected_config=selected,selection_source='internal dev only; no validation model selection')
    path=Path(args.weights_dir)/'selected_dev_readout.pt'
    with path.open('xb') as stream:torch.save(ck,stream)
    write_json(out/'selection.json',dict(selected=selected,runs=runs,checkpoint=str(path.resolve()),
        checkpoint_sha256=sha256(path),validation_used_for_selection=False,test_used=False))
    write_csv(out/'development_comparison.csv',rows)
    print(json.dumps(dict(stage='selected',selected=selected,runs=runs)),flush=True)


def evaluate_head(y,p,sources,reference):
    result=ranking(y,p); result['calibration']=reliability_metrics(y,p)
    result['positive_scores']=np.quantile(p[y==1],[0,.1,.5,.9,1]).tolist()
    result['negative_scores']=np.quantile(p[y==0],[0,.1,.5,.9,1]).tolist()
    result['policies']={};predictions={}
    tune=np.asarray(sources)!=F9
    for policy in ('f1','fpr10','precision50'):
        point=choose_threshold(y[tune],p[tune],policy,role='val')
        cross,pred,thresholds=cross_source_thresholds(y,p,sources,policy)
        result['policies'][policy]=dict(selected_without_f9=point,
            full_cohort_at_selected=metrics(y,p,point['threshold']),source_crossfit=cross)
        predictions[policy]=(pred,thresholds)
    result['reference_threshold']=metrics(y,p,reference)
    result['reference_threshold']['false_positive_rate']=result['reference_threshold']['fp']/max(1,result['reference_threshold']['fp']+result['reference_threshold']['tn'])
    return result,predictions


def cluster_delta(y,before,after,sources,draws=2000):
    keys=sorted(set(sources.tolist()));rng=np.random.default_rng(73);values=[]
    for _ in range(draws):
        ix=np.concatenate([np.flatnonzero(sources==key) for key in rng.choice(keys,len(keys),replace=True)])
        if len(np.unique(y[ix]))<2:continue
        b=metrics(y[ix],before[ix].astype(float));a=metrics(y[ix],after[ix].astype(float))
        values.append([a[k]-b[k] for k in ('f1','recall','precision')])
    interval=np.quantile(values,[.025,.975],axis=0)
    return dict(sources=len(keys),valid_draws=len(values),draws=draws,
        delta_95={k:interval[:,i].tolist() for i,k in enumerate(('f1','recall','precision'))},
        limitation='Source cluster bootstrap of fixed crossfit predictions; does not refit thresholds or account for model/validation reuse')


def evaluate(args):
    out=Path(args.out_dir);selection=json.loads((out/'selection.json').read_text())
    if sha256(selection['checkpoint'])!=selection['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    results={};prediction_rows=[];candidate_folds={};latencies={};coverage={}
    for cohort,folder in [('original','data/pose_cache/advanced_v1/windows'),('quality_matched',args.data)]:
        data,meta=load_windows(folder);audit_roles(data,meta);results[cohort]={};candidate_folds[cohort]={}
        ix=np.flatnonzero(data['role']=='val')
        coverage[cohort]=dict(materialized_windows=len(ix),source_count=len(set(data['video_id'][ix].tolist())),
            candidate_windows=sum(r['candidate_windows'] for r in meta['coverage'] if r['role']=='val'),
            eligibility_windows=sum(r['eligible_windows'] for r in meta['coverage'] if r['role']=='val'))
        for name,path in [('baseline',BASELINE),('advanced',HISTORICAL_ADVANCED),('headwise',HEADWISE),('readout',selection['checkpoint'])]:
            model,ck=load_model(path);started=time.monotonic()
            scored=score(model,DataLoader(Windows(data,'val'),batch_size=32),ck['variant'])
            latencies[cohort+'/'+name]=dict(seconds=time.monotonic()-started,windows=len(ix),batch_size=32,threads=2,pose_extraction_included=False)
            p=scored['logits'].sigmoid().numpy();y=scored['targets'].numpy();mask=scored['masks'].numpy()
            assert np.array_equal(scored['indices'].numpy(),ix)
            results[cohort][name]={};candidate_folds[cohort][name]={}
            for h,head in enumerate(HEADS):
                use=mask[:,h];yy=y[use,h].astype(int);pp=p[use,h];sources=data['video_id'][ix][use]
                result,preds=evaluate_head(yy,pp,sources,ck['thresholds'][h]);results[cohort][name][head]=result
                candidate_folds[cohort][name][head]=dict(y=yy,p=pp,sources=sources,preds=preds)
                for j,index in enumerate(ix[use]):
                    prediction_rows.append(dict(cohort=cohort,model=name,head=head,index=int(index),
                        video_id=str(sources[j]),cutoff_s=float(data['cutoff_s'][index]),y=int(yy[j]),score=float(pp[j]),
                        **{policy+'_crossfit_prediction':bool(v[0][j]) for policy,v in preds.items()},
                        **{policy+'_crossfit_threshold':float(v[1][j]) for policy,v in preds.items()}))
        # Calibration transform fitted on train-only internal dev, not validation.
        model,ck=load_model(selection['checkpoint'])
        dev=score(model,DataLoader(Windows(data,'dev'),batch_size=32),ck['variant'])
        for h,head in enumerate(HEADS):
            known=dev['masks'][:,h].numpy();dp=dev['logits'].sigmoid().numpy()[known,h];dy=dev['targets'].numpy()[known,h]
            temp=fit_temperature(dy,dp,role='dev');entry=candidate_folds[cohort]['readout'][head]
            results[cohort]['readout'][head]['development_temperature_diagnostic']=dict(fit=temp,
                validation=reliability_metrics(entry['y'],temperature_scale(entry['p'],temp['temperature'])))
    entry=candidate_folds['quality_matched'];head=HEADS[1];a=entry['readout'][head];b=entry['baseline'][head]
    uncertainty={policy:cluster_delta(a['y'],b['preds'][policy][0],a['preds'][policy][0],a['sources']) for policy in ('f1','fpr10','precision50')}
    am=results['quality_matched']['readout'][head]['policies']['fpr10']['source_crossfit']['metrics']
    bm=results['quality_matched']['baseline'][head]['policies']['fpr10']['source_crossfit']['metrics']
    criteria=dict(f1_gain=am['f1']>=bm['f1']+.05,recall_gain=am['recall']>=bm['recall']+.10,
        minimum_precision=am['precision']>=.5,fpr_cap=am['false_positive_rate']<=.1,
        source_bootstrap_lower_positive=uncertainty['fpr10']['delta_95']['f1'][0]>0,
        active_auroc_nonregression=results['quality_matched']['readout'][head]['auroc']>=results['quality_matched']['headwise'][head]['auroc'])
    # New reference thresholds only, research policy stays disabled. No default edits.
    selected_model,ck=load_model(selection['checkpoint'])
    ck['thresholds']=[results['quality_matched']['readout'][h]['policies']['fpr10']['selected_without_f9']['threshold'] for h in HEADS]
    ck['threshold_source']='Validation excluding F9; exploratory reference only; source crossfit evaluated separately'
    ck['promotion_criteria']=criteria;ck['research_alert_heads_enabled']=[False,False]
    reference_path=Path(args.weights_dir)/'selected_validation_reference.pt'
    with reference_path.open('xb') as stream:torch.save(ck,stream)
    write_json(out/'evaluation.json',dict(cohorts=results,coverage=coverage,latency=latencies,uncertainty=uncertainty,
        acceptance_criteria=criteria,acceptance_passed=all(criteria.values()),
        reference_checkpoint=str(reference_path.resolve()),reference_sha256=sha256(reference_path),
        default_changed=False,operational_alerts_enabled=False,test_used=False))
    write_csv(out/'validation_predictions.csv',prediction_rows)
    print(json.dumps(dict(stage='evaluation',criteria=criteria,
        baseline_crossfit=bm,readout_crossfit=am,uncertainty=uncertainty)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','train','evaluate'))
    p.add_argument('--data',default='data/pose_cache/model_improvement_expanded_v1/windows')
    p.add_argument('--out-dir',default='outputs/model_optimization_v1')
    p.add_argument('--weights-dir',default='weights/model_optimization_v1')
    args=p.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    dict(prepare=prepare,train=train,evaluate=evaluate)[args.command](args)


if __name__=='__main__':main()
