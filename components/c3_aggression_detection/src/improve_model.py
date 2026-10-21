"""Frozen, fit/dev-only kinematic comparison, followed by a separate evaluation.

The four candidates are fixed before validation. Independent heads use different
best epochs but one architecture is selected by mean internal-development AUROC.
No label inversion, test fitting, fabricated precursors or runtime promotion.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import SCHEMA, Windows, build_model, load_windows, metrics, score
from src.foundation_io import sha256, write_csv, write_json
from src.kinematic_model import KinematicTemporalModel, causal_features
from src.lock_splits import load_locked
from src.research_operating_point import select_point
from src.diagnose_stgcn import tune_threshold

HEADS = ('weak_first_onset_within_3s', 'weak_active_fight')
CANDIDATES = ((0, False), (0, True), (32, False), (32, True))


def audit_roles(data, meta):
    locked = {row['video_id']: row for row in load_locked(meta['locked_manifest_path'])}
    for key, role in zip(data['video_id'], data['role']):
        if key not in locked or role not in ('fit', 'dev', 'val'):
            raise ValueError('Unknown source or data role')
        if locked[key]['split'] != ('val' if role == 'val' else 'train'):
            raise ValueError('Locked test or wrong-role source prohibited')


def features_for(data, role):
    ds = Windows(data, role)
    values = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=48, shuffle=False):
            values.append(causal_features(batch['sk'], batch['temporal_mask'], batch['joint_mask'], batch['frame_times']))
    return torch.cat(values), ds.indices


def fit_weights(y, masks, sources, source_balanced):
    weights = np.zeros_like(y, dtype=np.float32)
    for h in range(2):
        use = masks[:, h]
        if set(y[use, h].tolist()) != {0., 1.}:
            raise ValueError('Each fitting head needs both weak classes')
        base = np.ones(len(y), np.float64)
        if source_balanced:
            for source in set(sources[use].tolist()):
                same = use & (sources == source)
                base[same] = 1. / same.sum()
        for label in (0, 1):
            same = use & (y[:, h] == label)
            weights[same, h] = base[same] / base[same].sum() / 2.
    return torch.tensor(weights)


def head_metrics(y, mask, probabilities, thresholds=None):
    return {name: metrics(y[mask[:, h], h], probabilities[mask[:, h], h],
                          .5 if thresholds is None else thresholds[h]) for h, name in enumerate(HEADS)}


def train(args):
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    data, meta = load_windows(args.data); audit_roles(data, meta)
    output, weights_dir = Path(args.out_dir), Path(args.weights_dir)
    plan = dict(schema='c3_kinematic_improvement_v1', seed=73, epochs=args.epochs,
        candidates=[dict(hidden=hidden, source_balanced=balanced) for hidden, balanced in CANDIDATES],
        features='24 masked body-relative motion/pose/interaction channels; trailing 1/3/6s mean/max/std plus current',
        normalization='fit-only mean and standard deviation, input clip +/-5',
        loss='per-head class-balanced BCE; source-balanced arms give equal total source weight within each class',
        optimizer='AdamW, learning rate 0.003, weight decay 0.05',
        epoch_selection='independent maximum development AUROC per head, lower BCE breaks ties',
        model_selection='maximum mean head development AUROC, then lower mean development BCE',
        patience=20, minimum_epochs=30, threshold_selection='internal-development max F1 and guarded FPR<=0.10',
        evaluation='original validation cohort and same current-observation-matched cohort; both reported',
        data=str(Path(args.data).resolve()), windows_sha256=meta['windows_sha256'],
        locked_manifest_sha256=meta['locked_manifest_sha256'], external_validation_used=False,
        test_used=False, verified_precursor_positive_videos=0, operational_alerts_enabled=False)
    if (output/'selection.json').exists():
        raise FileExistsError('Choose a new experiment version')
    write_json(output/'experiment_plan.json', plan)
    x_fit, fit_ix = features_for(data, 'fit'); x_dev, dev_ix = features_for(data, 'dev')
    y = torch.tensor(data['targets'][fit_ix]); mask = data['target_mask'][fit_ix]
    dev_y, dev_mask = data['targets'][dev_ix], data['target_mask'][dev_ix]
    mean, scale = x_fit.mean(0), x_fit.std(0, unbiased=False).clamp_min(.01)
    results = []
    started = time.monotonic()
    for hidden, balanced in CANDIDATES:
        name = f'kinematic_{"mlp" if hidden else "linear"}_{"source" if balanced else "window"}'
        torch.manual_seed(73); np.random.seed(73)
        model = KinematicTemporalModel(hidden)
        model.feature_mean.copy_(mean); model.feature_scale.copy_(scale)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.05)
        sample_weights = fit_weights(data['targets'][fit_ix], mask, data['video_id'][fit_ix], balanced)
        best_keys = [(-np.inf, -np.inf)] * 2; states = [None, None]; epochs = [0, 0]
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train(); optimizer.zero_grad()
            logits = model.feature_logits(x_fit)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss = (losses * sample_weights).sum(0).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
            model.eval()
            with torch.no_grad():
                probabilities = model.feature_logits(x_dev).sigmoid().numpy()
            report = head_metrics(dev_y, dev_mask, probabilities)
            history.append(dict(epoch=epoch, fit_loss=float(loss.detach()), development=report))
            for h, head in enumerate(HEADS):
                key = (report[head]['auroc'], -report[head]['bce'])
                if key > best_keys[h]:
                    best_keys[h] = key; states[h] = copy.deepcopy(model.heads[h].state_dict()); epochs[h] = epoch
            if epoch >= 30 and epoch - max(epochs) >= 20:
                break
        for h in range(2): model.heads[h].load_state_dict(states[h])
        with torch.no_grad(): probabilities = model.feature_logits(x_dev).sigmoid().numpy()
        points = [select_point(dev_y[dev_mask[:, h], h], probabilities[dev_mask[:, h], h]) for h in range(2)]
        f1 = [tune_threshold(dev_y[dev_mask[:, h], h], probabilities[dev_mask[:, h], h])['threshold'] for h in range(2)]
        threshold = [p['metrics']['threshold'] for p in points]
        dev_report = head_metrics(dev_y, dev_mask, probabilities)
        ck = dict(schema=SCHEMA, variant='kinematic_mlp' if hidden else 'kinematic_linear',
            state=model.state_dict(), thresholds=threshold, threshold_comparisons=dict(development_f1=f1, development_fpr10=threshold),
            research_alert_heads_enabled=[p['research_alert_eligible'] for p in points],
            feature_contract=meta['feature_contract'], windows_sha256=meta['windows_sha256'],
            locked_manifest_sha256=meta['locked_manifest_sha256'], trained_heads=list(HEADS),
            supervision='weak_ubi_first_onset_and_binary_fight_only', calibration='UNVALIDATED',
            verified_early_aggression_claim=False, operational_alerts_enabled=False, protocol=plan,
            selected_epochs=epochs, operating_point_policy=dict(source='locked train internal development only', max_development_fpr=.1))
        target = weights_dir/(name+'.pt'); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: torch.save(ck, stream)
        row = dict(name=name, checkpoint=str(target.resolve()), checkpoint_sha256=sha256(target), selected_epochs=epochs,
            development=dev_report, points=points, thresholds=threshold, f1_thresholds=f1,
            macro_auroc=float(np.mean([dev_report[h]['auroc'] for h in HEADS])),
            macro_bce=float(np.mean([dev_report[h]['bce'] for h in HEADS])))
        write_json(output/name/'training.json', dict(**row, history=history))
        results.append(row)
        print(json.dumps(dict(candidate=name, epochs=epochs, dev_auroc=[dev_report[h]['auroc'] for h in HEADS], elapsed_s=round(time.monotonic()-started,2))), flush=True)
    chosen = max(results, key=lambda row:(row['macro_auroc'], -row['macro_bce']))
    write_json(output/'selection.json',dict(selected=chosen, candidates=results, selection_used_external_validation=False,
        frozen_before_external_validation=True, elapsed_s=round(time.monotonic()-started,3)))


def evaluate(args):
    torch.set_num_threads(2)
    output = Path(args.out_dir)
    selection = json.loads((output/'selection.json').read_text())
    frozen = {row['checkpoint']:row['checkpoint_sha256'] for row in selection['candidates']}
    if any(sha256(p)!=digest for p,digest in frozen.items()):
        raise ValueError('Frozen checkpoint changed')
    chosen = selection['selected']
    checkpoint = torch.load(chosen['checkpoint'], map_location='cpu', weights_only=True)
    models = [('baseline', 'weights/baseline_weak_onset_guarded_v1.pt'),
              ('advanced', 'weights/advanced_weak_onset_guarded_v1.pt'),
              (chosen['name'], chosen['checkpoint'])]
    report = dict(selection=chosen['name'], selection_used_external_validation=False, test_used=False,
        external_validation_previously_inspected=True, claims='Exploratory weak-label window metrics, not verified early-aggression or camera-hour accuracy', cohorts={})
    predictions = []
    for cohort, folder in [('original',args.original_data), ('quality_matched',args.data)]:
        data, meta = load_windows(folder); audit_roles(data,meta)
        ix = np.flatnonzero(data['role']=='val')
        cohort_reports = {}
        for name, path in models:
            ck = torch.load(path,map_location='cpu',weights_only=True)
            if name==chosen['name']:
                model=KinematicTemporalModel(32 if ck['variant']=='kinematic_mlp' else 0)
                model.load_state_dict(ck['state']); model.eval()
                x, indices=features_for(data,'val')
                with torch.no_grad(): p=model.feature_logits(x).sigmoid().numpy()
            else:
                model=build_model(ck['variant']); model.load_state_dict(ck['state'])
                scored=score(model,DataLoader(Windows(data,'val'),batch_size=32),ck['variant'])
                p=scored['logits'].sigmoid().numpy(); indices=scored['indices'].numpy()
            assert np.array_equal(ix,indices)
            y, mask=data['targets'][ix],data['target_mask'][ix]
            policies=dict(guarded=ck['thresholds'])
            if name==chosen['name']: policies.update(development_f1=chosen['f1_thresholds'])
            cohort_reports[name]=dict(checkpoint_sha256=sha256(path), policies={policy:head_metrics(y,mask,p,thresholds) for policy,thresholds in policies.items()},
                sources=len(set(data['video_id'][ix].tolist())), windows=len(ix))
            predictions.extend(dict(cohort=cohort,model=name,video_id=str(data['video_id'][i]),cutoff_s=float(data['cutoff_s'][i]),
                onset_y=int(yy[0]),onset_mask=bool(mm[0]),onset_score=float(pp[0]),
                active_y=int(yy[1]),active_mask=bool(mm[1]),active_score=float(pp[1]))
                for i,yy,mm,pp in zip(ix,y,mask,p))
        report['cohorts'][cohort]=cohort_reports
    write_csv(output/'validation_predictions.csv',predictions)
    write_json(output/'evaluation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','evaluate'))
    p.add_argument('--data',default='data/pose_cache/diagnosis_guarded_v1/windows')
    p.add_argument('--original-data',default='data/pose_cache/advanced_v1/windows')
    p.add_argument('--out-dir',default='outputs/model_improvement_v1')
    p.add_argument('--weights-dir',default='weights/model_improvement_v1')
    p.add_argument('--epochs',type=int,default=160)
    args=p.parse_args()
    if args.epochs<1:p.error('Positive epoch limit required')
    (train if args.command=='train' else evaluate)(args)
