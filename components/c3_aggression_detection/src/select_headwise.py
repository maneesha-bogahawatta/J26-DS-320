"""Freeze task-specific model choices using expanded internal development only.

The ensemble grid contains only nonnegative convex weights, so no score direction
can be inverted. External validation is read by the separate evaluate command.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import SCHEMA, Windows, build_model, load_windows, metrics, score
from src.diagnose_stgcn import tune_threshold
from src.foundation_io import sha256, write_json, write_csv
from src.headwise_model import HeadwiseC3
from src.improve_model import audit_roles, HEADS
from src.research_operating_point import select_point


def convex_grid():
    return [np.array([a,b,4-a-b],dtype=float)/4 for a in range(5) for b in range(5-a)]


def select(args):
    torch.set_num_threads(2)
    data,meta=load_windows(args.data);audit_roles(data,meta)
    out=Path(args.out_dir)
    plan=dict(schema='c3_headwise_selection_v1', data_sha256=meta['windows_sha256'],
        component_selection='baseline family maximizes development onset AUROC; advanced family maximizes development active-fight AUROC; kinematic family maximizes mean AUROC; lower matching BCE breaks ties',
        head_selection='per-head maximum development AUROC over positive quarter-step convex weights; advanced onset excluded because its historical research forecast gate failed',
        thresholds='development F1 and FPR<=0.10; minimum recall .20 and balanced accuracy .55 for research eligibility',
        original_cohort_acceptance='Both heads ranking non-regression vs runtime baseline; active ranking improvement vs historical advanced; guarded active F1 non-regression vs runtime baseline',
        operational_alerts_enabled=False,test_used=False,external_validation_used=False,
        prior_validation_exposure='Exploratory existing cohort; earlier v1 candidate was inspected and rejected')
    write_json(out/'selection_plan.json',plan)
    paths=dict(baseline=['weights/baseline_weak_onset_guarded_v1.pt',str(Path(args.trained_dir)/'baseline_class_weighted.pt')],
        advanced=['weights/advanced_weak_onset_guarded_v1.pt',str(Path(args.trained_dir)/'advanced_class_weighted.pt')],
        kinematic=[str(Path(args.kinematic_dir)/'kinematic_linear_window.pt'),str(Path(args.kinematic_dir)/'kinematic_linear_source.pt')])
    for family in ('baseline','advanced'):
        paths[family].extend([f'weights/diagnosis_v1/{family}_class_weighted.pt',
            f'weights/diagnosis_v1/{family}_unweighted.pt',
            f'weights/diagnosis_guarded_v1/{family}_class_weighted.pt'])
    chosen={};probabilities=[];states={}
    for family,options in paths.items():
        candidates=[]
        for path in options:
            ck=torch.load(path,map_location='cpu',weights_only=True)
            model=build_model(ck['variant']);model.load_state_dict(ck['state'])
            scored=score(model,DataLoader(Windows(data,'dev'),batch_size=32),ck['variant'])
            p=scored['logits'].sigmoid().numpy();y=scored['targets'].numpy();mask=scored['masks'].numpy()
            heads=[metrics(y[mask[:,h],h],p[mask[:,h],h]) for h in range(2)]
            primary = [heads[0]] if family=='baseline' else [heads[1]] if family=='advanced' else heads
            candidates.append((float(np.mean([v['auroc'] for v in primary])),-float(np.mean([v['bce'] for v in primary])),path,ck,p,heads))
        winner=max(candidates,key=lambda v:v[:2])
        chosen[family]=dict(path=winner[2],sha256=sha256(winner[2]),development=winner[5],
            candidates=[dict(path=v[2],selection_auroc=v[0],selection_bce=-v[1]) for v in candidates])
        states[family]=winner[3]['state'];probabilities.append(winner[4])
    blend=np.stack(probabilities,-1);weights=[];heads=[];points=[];f1=[]
    for h in range(2):
        use=mask[:,h];yy=y[use,h]
        def key(w):
            report=metrics(yy,blend[use,h]@w)
            return report['auroc'],-report['bce']
        grid = [w for w in convex_grid() if h==1 or w[1]==0]
        w=max(grid,key=key);pp=blend[use,h]@w
        weights.append(w.tolist());heads.append(metrics(yy,pp));points.append(select_point(yy,pp))
        f1.append(tune_threshold(yy,pp)['threshold'])
    model=HeadwiseC3()
    for family,state in states.items():getattr(model,family).load_state_dict(state)
    model.blend_weights.copy_(torch.tensor(weights))
    ck=dict(schema=SCHEMA,variant='headwise',state=model.state_dict(),
        thresholds=[p['metrics']['threshold'] for p in points],
        threshold_comparisons=dict(development_f1=f1,development_fpr10=[p['metrics']['threshold'] for p in points]),
        research_alert_heads_enabled=[p['research_alert_eligible'] for p in points],
        feature_contract=meta['feature_contract'],windows_sha256=meta['windows_sha256'],
        locked_manifest_sha256=meta['locked_manifest_sha256'],trained_heads=list(HEADS),
        supervision='weak_ubi_first_onset_and_binary_fight_only',calibration='UNVALIDATED',
        verified_early_aggression_claim=False,operational_alerts_enabled=False,
        operating_point_policy=dict(source='locked train expanded internal development only',max_development_fpr=.1),
        protocol=plan,component_checkpoints=chosen,blend_weights=weights)
    path=Path(args.out_checkpoint);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(ck,stream)
    write_json(out/'selection.json',dict(checkpoint=str(path.resolve()),checkpoint_sha256=sha256(path),
        components=chosen,blend_weights=weights,development=heads,points=points,
        frozen_before_external_validation=True,test_used=False))
    print(json.dumps(dict(blend_weights=weights,development=heads,points=points)),flush=True)


def evaluate(args):
    torch.set_num_threads(2)
    out=Path(args.out_dir);selection=json.loads((out/'selection.json').read_text())
    if sha256(selection['checkpoint'])!=selection['checkpoint_sha256']:raise ValueError('Selected model changed')
    reports={};rows=[]
    for cohort,folder in [('original','data/pose_cache/advanced_v1/windows'),('quality_matched',args.data)]:
        data,meta=load_windows(folder);audit_roles(data,meta);results={}
        for name,path in [('baseline','weights/baseline_weak_onset_guarded_v1.pt'),('advanced','weights/advanced_weak_onset_guarded_v1.pt'),('candidate',selection['checkpoint'])]:
            ck=torch.load(path,map_location='cpu',weights_only=True);model=build_model(ck['variant']);model.load_state_dict(ck['state'])
            scored=score(model,DataLoader(Windows(data,'val'),batch_size=32),ck['variant'])
            p=scored['logits'].sigmoid().numpy();y=scored['targets'].numpy();mask=scored['masks'].numpy()
            policies=dict(guarded=ck['thresholds'])
            if name=='candidate':policies['development_f1']=ck['threshold_comparisons']['development_f1']
            results[name]={policy:{head:metrics(y[mask[:,h],h],p[mask[:,h],h],thresholds[h]) for h,head in enumerate(HEADS)} for policy,thresholds in policies.items()}
            rows.extend(dict(cohort=cohort,model=name,video_id=str(data['video_id'][i]),cutoff_s=float(data['cutoff_s'][i]),
                onset_y=int(yy[0]),onset_mask=bool(mm[0]),onset_score=float(pp[0]),active_y=int(yy[1]),active_mask=bool(mm[1]),active_score=float(pp[1]))
                for i,yy,mm,pp in zip(scored['indices'].numpy(),y,mask,p))
        reports[cohort]=results
    r=reports['original'];b=r['baseline']['guarded'];a=r['advanced']['guarded'];c=r['candidate']['guarded']
    passed=bool(all(c[h]['auroc']>=b[h]['auroc'] for h in HEADS) and c[HEADS[1]]['auroc']>a[HEADS[1]]['auroc'] and c[HEADS[1]]['f1']>=b[HEADS[1]]['f1'])
    write_json(out/'evaluation.json',dict(cohorts=reports,predeclared_original_cohort_acceptance_passed=passed,
        selection_used_external_validation=False,test_used=False,operational_alerts_enabled=False,
        note='Exploratory weak-label overlapping windows, not verified pre-contact accuracy; acceptance is descriptive and does not establish statistical or operational validity.'))
    write_csv(out/'validation_predictions.csv',rows)
    print(json.dumps(dict(acceptance_passed=passed,cohorts=reports)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('select','evaluate'))
    p.add_argument('--data',default='data/pose_cache/model_improvement_expanded_v1/windows')
    p.add_argument('--out-dir',default='outputs/model_improvement_expanded_v1/headwise')
    p.add_argument('--trained-dir',default='weights/model_improvement_expanded_v1/stgcn')
    p.add_argument('--kinematic-dir',default='weights/model_improvement_expanded_v1/kinematic')
    p.add_argument('--out-checkpoint',default='weights/model_improvement_expanded_v1/headwise_guarded.pt')
    args=p.parse_args();(select if args.command=='select' else evaluate)(args)
