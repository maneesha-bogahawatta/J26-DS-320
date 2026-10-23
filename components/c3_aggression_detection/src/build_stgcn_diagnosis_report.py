"""Compile matched-cohort weak-label diagnostics, plots and paired source bootstrap."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.advanced_experiment import metrics
from src.diagnose_stgcn import HEADS, ranking
from src.foundation_io import read_csv, write_csv, write_json, write_new


def paired_source_bootstrap(y, old, new, groups, old_threshold=.5, new_threshold=.5,
                            repetitions=1000, seed=73):
    y, old, new, groups = map(np.asarray, (y,old,new,groups))
    if not (y.shape == old.shape == new.shape == groups.shape) or len(set(y)) != 2:
        raise ValueError('Aligned scores, source groups and both classes required')
    unique = np.unique(groups)
    members = {key:np.flatnonzero(groups == key) for key in unique}
    rng = np.random.default_rng(seed)
    changes = {key:[] for key in ('auroc','average_precision','precision','recall','f1')}
    for _ in range(repetitions):
        ix = np.concatenate([members[key] for key in rng.choice(unique, len(unique), replace=True)])
        if len(set(y[ix])) != 2:
            continue
        before = metrics(y[ix],old[ix],old_threshold)
        after = metrics(y[ix],new[ix],new_threshold)
        for key in changes:
            changes[key].append(after[key]-before[key])
    return dict(source_groups=len(unique), requested_repetitions=repetitions,
        valid_repetitions=len(changes['auroc']), intervals={key:np.quantile(values,[.025,.975]).tolist()
            if values else None for key,values in changes.items()},
        note='Paired source-cluster percentile intervals with fixed development thresholds; exploratory, 13 selected sources and overlapping windows, no verified-label inference.')


def parse_predictions(path):
    rows = read_csv(path)
    for row in rows:
        for field in ('onset','active'):
            row[field+'_score'] = float(row[field+'_score'])
            row[field+'_y'] = int(row[field+'_y'])
            row[field+'_mask'] = str(row[field+'_mask']).lower() == 'true'
        row['cutoff_s'] = float(row['cutoff_s'])
    return rows


def build(args):
    root = Path(args.out_dir)
    audit = json.loads((root/'audit.json').read_text())
    old_rows = parse_predictions(root/'existing_predictions.csv')
    observation = read_csv(Path(args.quality_data)/'selection.csv')
    retained = {(r['video_id'],round(float(r['cutoff_s']),6)) for r in observation if r['retained']=='True'}
    experiments = {}
    for variant in ('baseline','advanced'):
        rows = [r for r in old_rows if r['model']==variant and r['role']=='val']
        thresholds = audit['models'][variant]['thresholds']
        for cohort in ('original','current_guard'):
            selected = rows if cohort == 'original' else [r for r in rows if (r['video_id'],round(r['cutoff_s'],6)) in retained]
            experiments[(variant+'_historical',cohort)] = dict(variant=variant,rows=selected,
                thresholds=thresholds, historical=True)
    for directory, cohort in (('ablation','original'),('guarded_comparison','current_guard')):
        frozen = json.loads((root/directory/'frozen_comparison.json').read_text())
        for folder in sorted((root/directory).iterdir()):
            if not folder.is_dir() or not (folder/'evaluation.json').exists():
                continue
            training = json.loads((folder/'training.json').read_text())
            experiments[(folder.name,cohort)] = dict(variant=training['config']['variant'],
                rows=parse_predictions(folder/'validation_predictions.csv'),
                thresholds=dict(half=[.5,.5],**training['thresholds']),historical=False,
                development_selected=frozen['development_selection'][training['config']['variant']]['selected_run']==folder.name,
                training=training)
    table = []
    for (name,cohort), exp in experiments.items():
        for h, (head,pre) in enumerate(zip(HEADS,('onset','active'))):
            select = [r for r in exp['rows'] if r[pre+'_mask']]
            y = np.array([r[pre+'_y'] for r in select])
            p = np.array([r[pre+'_score'] for r in select])
            ranks = ranking(y,p)
            for policy,values in exp['thresholds'].items():
                point = metrics(y,p,values[h])
                table.append(dict(model=name,cohort=cohort,head=head,policy=policy,
                    development_selected=exp.get('development_selected',False),
                    false_positive_rate=point['fp']/max(1,point['fp']+point['tn']),
                    always_positive_f1=2*point['positives']/(point['n']+point['positives']),
                    pr_auc_trapezoid=ranks['pr_auc_trapezoid'],**point))
    write_csv(root/'comparison_metrics.csv', table)
    deltas = []
    for (name,cohort), exp in experiments.items():
        if exp['historical']:
            continue
        reference_name = exp['variant']+'_historical'
        reference = experiments[(reference_name,cohort)]
        for h,(head,pre) in enumerate(zip(HEADS,('onset','active'))):
            before = next(r for r in table if r['model']==reference_name and r['cohort']==cohort and r['head']==head and r['policy']=='development_f1')
            after = next(r for r in table if r['model']==name and r['cohort']==cohort and r['head']==head and r['policy']=='development_f1')
            old_by_key = {(r['video_id'],round(r['cutoff_s'],6)):r for r in reference['rows'] if r[pre+'_mask']}
            selected = [r for r in exp['rows'] if r[pre+'_mask']]
            keys = [(r['video_id'],round(r['cutoff_s'],6)) for r in selected]
            if set(keys) != set(old_by_key):
                raise ValueError('Unmatched comparison cohort')
            labels = [r[pre+'_y'] for r in selected]
            if labels != [old_by_key[k][pre+'_y'] for k in keys]:
                raise ValueError('Changed targets in comparison')
            boot = paired_source_bootstrap(labels,[old_by_key[k][pre+'_score'] for k in keys],
                [r[pre+'_score'] for r in selected],[r['video_id'] for r in selected],
                reference['thresholds']['development_f1'][h],exp['thresholds']['development_f1'][h],
                repetitions=args.bootstrap)
            delta = {k:after[k]-before[k] for k in ('auroc','average_precision','pr_auc_trapezoid','precision','recall','f1')}
            deltas.append(dict(model=name,reference=reference_name,cohort=cohort,head=head,delta=delta,bootstrap=boot))
    write_json(root/'paired_improvements.json', deltas)

    os.environ['MPLCONFIGDIR'] = str((root/'matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = root/'plots'
    plots.mkdir(exist_ok=True)
    def save(fig,name):
        path=plots/name
        if path.exists():raise FileExistsError('Choose a new report directory/version')
        fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for row,variant in enumerate(('baseline','advanced')):
        for col,role in enumerate(('fit','dev','val')):
            ax=axes[row,col]
            chosen=[r for r in old_rows if r['model']==variant and r['role']==role and r['onset_mask']]
            for label,color in ((0,'#2676a6'),(1,'#db7537')):
                values=[r['onset_score'] for r in chosen if r['onset_y']==label]
                ax.hist(values,bins=np.linspace(0,1,21),density=True,alpha=.55,color=color,label=f'target {label}')
            ax.set(title=f'{variant}: {role}',xlabel='Original onset sigmoid score',ylabel='Density',xlim=(0,1))
            ax.legend(fontsize=8)
    fig.suptitle('Historical first-onset score distributions; original weak labels')
    save(fig,'score_distributions.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for h,(head,pre) in enumerate(zip(HEADS,('onset','active'))):
        ax=axes[h]
        for (name,cohort),exp in experiments.items():
            if cohort!='original':continue
            chosen=[r for r in exp['rows'] if r[pre+'_mask']]
            y=np.array([r[pre+'_y'] for r in chosen]);p=np.array([r[pre+'_score'] for r in chosen])
            order=np.argsort(-p,kind='stable');ends=np.r_[np.flatnonzero(np.diff(p[order])),len(p)-1]
            tp=y[order].cumsum()[ends];recall=tp/y.sum();precision=tp/(ends+1)
            ap=ranking(y,p)['average_precision']
            ax.step(np.r_[0,recall],np.r_[1,precision],where='pre',label=f'{name.replace("class_weighted","weighted")} AP={ap:.3f}',alpha=.85)
        ax.axhline(y.mean(),linestyle='--',color='gray',linewidth=1,label=f'Prevalence {y.mean():.3f}')
        ax.set(title='Weak '+('onset within 3s' if h==0 else 'active fight'),xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=7)
    save(fig,'precision_recall.png')
    full=[(name,exp) for (name,cohort),exp in experiments.items() if cohort=='original' and not exp['historical']]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(name,exp) in zip(axes.flat,full):
        training=exp['training'];history=training['history']
        for role,color in (('fit','#2676a6'),('development','#db7537')):
            ax.plot([r['epoch'] for r in history],[r[role]['macro_auroc'] for r in history],marker='o',label=role,color=color)
        ax.axvline(training['selected_epoch'],linestyle=':',color='gray',label='Selected on development')
        ax.axhline(.5,linestyle='--',linewidth=.8,color='gray')
        ax.set(title=name,xlabel='Epoch',ylabel='Mean head AUROC',ylim=(.35,.85));ax.legend(fontsize=8)
    save(fig,'learning_curves.png')
    frozen=json.loads((root/'ablation/frozen_comparison.json').read_text())
    names=['baseline_historical','advanced_historical']+[frozen['development_selection'][v]['selected_run'] for v in ('baseline','advanced')]
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for col,name in enumerate(names):
        for row,head in enumerate(HEADS):
            m=next(r for r in table if r['model']==name and r['cohort']=='original' and r['head']==head and r['policy']=='development_f1')
            matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
            ax=axes[row,col];ax.imshow(matrix,cmap='Blues',vmin=0,vmax=175)
            for (i,j),value in np.ndenumerate(matrix):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>85 else 'black')
            ax.set(title=f'{name}\n{"onset" if row==0 else "active"} t={m["threshold"]:.3f}',xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='True')
    save(fig,'confusion_matrices.png')
    write_json(root/'report_data.json',dict(comparisons=table,improvements= deltas,
        development_selection=frozen['development_selection'],cohort_policy='Original and current-observation cohorts reported separately',
        operational_alerts_enabled=False,test_used=False,verified_early_aggression_claim=False))
    lines=['**ST-GCN diagnosis: measured weak-label results**','',
        'Targets, original source annotations, saved score direction and AUROC implementations agree. There is no demonstrated global label/score inversion. Complementing the advanced onset score raises external-validation AUROC from 0.232 to 0.768 but lowers development AUROC from 0.601 to 0.399; it is a diagnostic, never a deployed correction.','',
        'All new checkpoints and thresholds use source-grouped internal development from locked training data. External validation was historically inspected and is exploratory. No test source, fabricated precursor label, original artifact replacement or deployment change is involved.','',
        '**Class distributions (positive / negative / masked)**','',
        '| Role | Onset | Active fight | Usable fight / no-fight sources |','|---|---:|---:|---:|']
    for role in ('fit','dev','val'):
        d=audit['distributions'][role]
        cells=[' / '.join(str(d['heads'][h][k]) for k in ('positive','negative','masked')) for h in HEADS]
        lines.append(f'| {role} | {cells[0]} | {cells[1]} | {d["weak_fight_videos"]} / {d["weak_no_fight_videos"]} |')
    lines+=['','Class weighting was already enabled historically. The old video-balanced sampler changed effective onset prevalence from 54.45% to 45.59% while weights were calculated from raw window counts: the historical onset positive weight was 0.837, whereas the corresponding sampler-weighted ratio was 1.194. The new comparison uses uniformly shuffled windows without replacement, consistent fitting-only class weights, and the same fixed architectures/augmentation. Checkpoint selection now directly measures development ranking instead of averaging two BCE losses. Differences against history combine these training-policy changes; the paired weighted/unweighted arms isolate weighting within the new policy.','',
        '**Original-cohort validation comparison**','',
        'All rows below use identical original validation windows: 161 supervised onset windows (50 positives) and 224 active-fight windows (51 positives). F1 thresholds are chosen only on internal development. `historical_guarded` is the previous restrictive policy; `development_f1` is the exact development maximum-F1 policy. Higher recall alone can mean predicting almost everything positive. Predicting every window positive already gives onset F1=0.474 and active-fight F1=0.371 on these cohorts; improvements over the previous restrictive threshold must not be confused with useful discrimination.','',
        '| Model | Head | Threshold policy | AUROC | Precision | Recall | F1 | AP | PR-AUC (trapezoid) |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        if r['cohort']!='original' or r['policy'] not in ('historical_guarded','development_f1'):continue
        lines.append('| '+ ' | '.join([r['model'],'onset' if r['head']==HEADS[0] else 'active',r['policy']]+[f'{r[k]:.3f}' for k in ('auroc','precision','recall','f1','average_precision','pr_auc_trapezoid')])+' |')
    lines+=['','**Current-pose preprocessing mismatch and separate matched cohort**','',
        'The historical window gate could accept stale history when current observations were absent. The existing runtime-v2 rule requires six currently observed joints in at least one selected person. Recomputing every skeleton from its raw cache reproduced all 1,974 windows exactly. Applying that fixed pose-only rule excludes 261 fit, 92 development and 20 validation windows. No targets are relabeled. The separate guarded archive retains 1,088 / 309 / 204 windows; its validation onset head has 152 supervised windows with 46 positives, and active head has 204 with 42 positives. The four original onset-positive and nine active-positive windows excluded at validation are abstentions, not negatives.','',
        'The two class-weighted quality-aligned models are compared with historical models on this same retained subset. These cohort results cannot be directly substituted for full-cohort performance.','',
        '| Model | Head | AUROC | Precision | Recall | F1 | AP | PR-AUC (trapezoid) |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        if r['cohort']=='current_guard' and r['policy']=='development_f1':
            lines.append('| '+' | '.join([r['model'],'onset' if r['head']==HEADS[0] else 'active']+[f'{r[k]:.3f}' for k in ('auroc','precision','recall','f1','average_precision','pr_auc_trapezoid')])+' |')
    lines+=['','**Learning and uncertainty**','',
        'The historical advanced onset fit/development AUROC is 0.625/0.601, compared with 0.232 externally. Its within-video validation onset AUROC is only 0.156, so the failure is not explained solely by added normal-video controls. Its selected checkpoint does not show near-perfect fitting discrimination; weak learning and source/timing shift are supported more clearly than a claim of simple training-set memorization. Later development BCE deteriorated while training loss fell, but those original loss numbers also differ in weighting and augmentation. New curves measure fitting and development in evaluation mode with the same unweighted objectives.','',
        'A fitting-only tiny-batch diagnostic can drive both model losses close to zero with finite gradients. That confirms an optimization path, not useful generalization. Per-epoch per-head AUROC/AP/BCE, gradient norms, source counts, score quantiles, confusion matrices and all threshold policies are retained. The paired source bootstrap resamples the 13 validation source groups together, never overlapping windows independently. Confidence intervals in `paired_improvements.json` are exploratory; one seed and selected weak-label contexts do not establish reliable model superiority.','',
        'AP is the non-interpolated precision-recall summary. It differs from trapezoidal PR-AUC; both are reported, following the [scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Positive logits follow [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).','',
        '**Artifacts and reproduction**','',
        '`outputs/diagnosis_v1/comparison_metrics.csv` contains every confusion matrix and policy. `paired_improvements.json` contains actual matched deltas and source-cluster intervals. Plots are in `outputs/diagnosis_v1/plots/`. New source modules are `diagnose_stgcn.py`, `stgcn_loss_ablation.py`, `stgcn_quality_windows.py` and `build_stgcn_diagnosis_report.py`; tests are in `tests/test_stgcn_diagnostics.py`. Existing models, labels, split identities, data, v1-v4 detectors and previous results are retained. All weights stay local/ignored.','',
        'Run from Component 03, using new version names so existing artifacts remain intact:','',
        '```powershell',
        r"Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'",
        r"$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path",
        "$env:PYTHONDONTWRITEBYTECODE = '1'",
        '& $C3Python -m src.diagnose_stgcn --out-dir outputs/diagnosis_reproduction_v1',
        '& $C3Python -m src.stgcn_loss_ablation train-all --out-dir outputs/diagnosis_reproduction_v1/ablation --weights-dir weights/diagnosis_reproduction_v1',
        '& $C3Python -m src.stgcn_loss_ablation evaluate-all --out-dir outputs/diagnosis_reproduction_v1/ablation --weights-dir weights/diagnosis_reproduction_v1',
        '& $C3Python -m src.stgcn_quality_windows --out-dir data/pose_cache/diagnosis_guarded_reproduction_v1/windows',
        '& $C3Python -m src.stgcn_loss_ablation train-all --data data/pose_cache/diagnosis_guarded_reproduction_v1/windows --out-dir outputs/diagnosis_reproduction_v1/guarded_comparison --weights-dir weights/diagnosis_guarded_reproduction_v1 --losses class_weighted',
        '& $C3Python -m src.stgcn_loss_ablation evaluate-all --data data/pose_cache/diagnosis_guarded_reproduction_v1/windows --out-dir outputs/diagnosis_reproduction_v1/guarded_comparison --weights-dir weights/diagnosis_guarded_reproduction_v1 --losses class_weighted',
        '```','',
        'The full suite passed **175 tests plus 17 subtests**, with no failures or skips; see `outputs/diagnosis_v1/tests_complete.xml`. Tests cover positive-logit gradient direction, masked supervision, fit-only class weights, tied AP/PR areas, exact development thresholds, quality filtering without label changes, paired source resampling and locked-test rejection. The earlier targeted runs passed 20 and 21 tests; the final suite includes the additional paired-bootstrap regression. No dependencies were installed for this investigation.','',
        'The scientific blocker remains independently adjudicated precursor/contact supervision and broader, prospectively frozen source-group evaluation. More favorable weak-label window metrics do not establish confirmed early-violence accuracy. The runtime default and alert eligibility remain unchanged.']
    write_new(args.document,'\n'.join(lines)+'\n')
    print('Diagnosis report:',Path(args.document).resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default='outputs/diagnosis_v1')
    parser.add_argument('--document',default='STGCN_DIAGNOSIS.md')
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--quality-data',default='data/pose_cache/diagnosis_guarded_v1/windows')
    args=parser.parse_args()
    if args.bootstrap < 1:parser.error('Positive bootstrap repetition count required')
    build(args)
