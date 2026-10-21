"""Frozen four-run loss/sampling comparison on existing weak-label windows.

No extraction, new labels, test access, runtime replacement or validation tuning.
All four fits finish before the separate external-validation evaluation command.
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import (SCHEMA, Windows, build_model, forward,
                                     load_windows, score)
from src.diagnose_stgcn import (HEADS, evaluate_scores, result_rows,
                                scored_probabilities, tune_threshold)
from src.foundation_io import sha256, write_csv, write_json


def fit_class_weights(data):
    ix = data['role'] == 'fit'
    positives = (data['targets'][ix] * data['target_mask'][ix]).sum(0)
    negatives = data['target_mask'][ix].sum(0) - positives
    if (positives <= 0).any() or (negatives <= 0).any():
        raise ValueError('Each fitting head requires both weak classes')
    return torch.tensor(np.clip(negatives / positives, .25, 4), dtype=torch.float32)


def masked_bce(logits, targets, masks, pos_weight):
    if logits.shape != targets.shape or targets.shape != masks.shape:
        raise ValueError('Head/target/mask shape mismatch')
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none')
    count = masks.sum(0)
    valid = count > 0
    means = (losses * masks).sum(0) / count.clamp_min(1)
    # No supervised heads means a differentiable zero, never invented negatives.
    return means[valid].mean() if valid.any() else logits.sum() * 0.


def development_summary(scored):
    probabilities = scored['logits'].sigmoid().numpy()
    heads = evaluate_scores(scored, probabilities, {})
    macro_auroc = float(np.mean([heads[h]['auroc'] for h in HEADS]))
    macro_bce = float(np.mean([heads[h]['bce'] for h in HEADS]))
    return dict(macro_auroc=macro_auroc, macro_bce=macro_bce,
                heads={h:{k: heads[h][k] for k in ('auroc','average_precision','pr_auc_trapezoid','bce','n','positives')}
                       for h in HEADS})


def train_one(args, variant, loss_name):
    run_name = variant + '_' + loss_name
    output = Path(args.out_dir) / run_name
    checkpoint_path = Path(args.weights_dir) / (run_name + '.pt')
    if checkpoint_path.exists() or (output / 'training.json').exists():
        raise FileExistsError('Select a new version directory for a new training run')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, meta = load_windows(args.data)
    if meta['test_sources_used']:
        raise ValueError('Test sources forbidden')
    weights = fit_class_weights(data) if loss_name == 'class_weighted' else torch.ones(2)
    fit = Windows(data, 'fit', augment=True)
    fit_plain = Windows(data, 'fit')
    dev = Windows(data, 'dev')
    for h in range(2):
        y = data['targets'][dev.indices, h][data['target_mask'][dev.indices, h]]
        if len(set(y.tolist())) != 2:
            raise ValueError('Both development classes required')
    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True, generator=generator)
    eval_loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=False) for ds in (fit_plain, dev)]
    config = dict(variant=variant, loss=loss_name, pos_weight=weights.tolist(), seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=.0003, weight_decay=.01,
        sampler='uniform shuffled fitting windows without replacement',
        augmentation='same person swap, horizontal reflection and 2.5% joint dropout as historical run',
        selection='maximum internal-development mean head AUROC; lower unweighted BCE breaks ties',
        patience=3, threshold_source='locked-train internal development only',
        score_direction='sigmoid positive-class logits; never complemented',
        windows_sha256=meta['windows_sha256'], locked_manifest_sha256=meta['locked_manifest_sha256'],
        test_sources_used=0, labels='original weak UBI onset-within-3s and active fight only')
    write_json(output / 'protocol.json', config)
    model = build_model(variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    best_key = (-math.inf, -math.inf)
    best_state = None
    best_epoch = None
    history = []
    stale = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        gradient_norms = []
        batches = 0
        for batch in fit_loader:
            logits = forward(model, batch, variant)
            loss = masked_bce(logits, batch['targets'], batch['target_mask'], weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite optimization loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradients')
            gradient_norms.append(float(norm))
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        fit_scores, dev_scores = [score(model, loader, variant) for loader in eval_loaders]
        fit_report, dev_report = map(development_summary, (fit_scores, dev_scores))
        row = dict(epoch=epoch, optimization_loss=total / batches,
            fit=fit_report, development=dev_report,
            gradient_norm_mean=float(np.mean(gradient_norms)), gradient_norm_max=max(gradient_norms),
            elapsed_s=round(time.monotonic() - started, 3))
        history.append(row)
        write_json(output / 'epochs' / f'{epoch:02}.json', row)
        print(json.dumps(dict(run=run_name, epoch=epoch, fit_auc=fit_report['macro_auroc'],
            dev_auc=dev_report['macro_auroc'], dev_bce=dev_report['macro_bce'], elapsed_s=row['elapsed_s'])), flush=True)
        key = (dev_report['macro_auroc'], -dev_report['macro_bce'])
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(best_state)
    dev_scores = score(model, eval_loaders[1], variant)
    probabilities = dev_scores['logits'].sigmoid().numpy()
    thresholds = {}
    for name, cap in (('development_f1', None), ('development_fpr10', .1)):
        thresholds[name] = []
        for h in range(2):
            known = dev_scores['masks'][:, h].numpy()
            thresholds[name].append(tune_threshold(dev_scores['targets'][:, h].numpy()[known],
                probabilities[known, h], cap)['threshold'])
    ck = dict(schema=SCHEMA, state=best_state, variant=variant, protocol=config,
        thresholds=thresholds['development_f1'], threshold_comparisons=thresholds,
        trained_heads=list(HEADS), history=history, selected_epoch=best_epoch,
        windows_sha256=meta['windows_sha256'], locked_manifest_sha256=meta['locked_manifest_sha256'],
        feature_contract=meta['feature_contract'], research_alert_heads_enabled=[False,False],
        operational_alerts_enabled=False, verified_early_aggression_claim=False,
        supervision='weak_ubi_first_onset_and_binary_fight_only', calibration='UNVALIDATED')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open('xb') as stream:
        torch.save(ck, stream)
    reports = {}
    rows = []
    for role, loader in zip(('fit', 'dev'), eval_loaders):
        scored = score(model, loader, variant)
        p = scored['logits'].sigmoid().numpy()
        reports[role] = evaluate_scores(scored, p, dict(half=[.5,.5], **thresholds))
        rows.extend(result_rows(data, scored, p, run_name, role))
    write_csv(output / 'fit_dev_predictions.csv', rows)
    write_json(output / 'training.json', dict(config=config, history=history, selected_epoch=best_epoch,
        thresholds=thresholds, results=reports, checkpoint=str(checkpoint_path.resolve()),
        checkpoint_sha256=sha256(checkpoint_path), elapsed_s=round(time.monotonic()-started,3),
        external_validation_used=False, test_used=False, verified_early_aggression_claim=False))


def evaluate_all(args):
    torch.set_num_threads(2)
    data, meta = load_windows(args.data)
    # Freeze every checkpoint before forwarding any external-validation window.
    runs = [variant+'_'+loss for variant in args.variants for loss in args.losses]
    paths = [Path(args.weights_dir)/(name+'.pt') for name in runs]
    if not all(p.is_file() for p in paths):
        raise ValueError('All predefined fits must finish before external validation')
    frozen = {str(p):sha256(p) for p in paths}
    development_selection = {}
    for variant in args.variants:
        candidates = {}
        for loss in args.losses:
            name = variant + '_' + loss
            training = json.loads((Path(args.out_dir)/name/'training.json').read_text())
            selected = next(row for row in training['history'] if row['epoch'] == training['selected_epoch'])
            candidates[name] = selected['development']
        winner = max(candidates, key=lambda name: (candidates[name]['macro_auroc'], -candidates[name]['macro_bce']))
        development_selection[variant] = dict(selected_run=winner, candidates=candidates)
    write_json(Path(args.out_dir)/'frozen_comparison.json', dict(checkpoints=frozen,
        data_sha256=meta['windows_sha256'], external_validation_previously_inspected=True,
        development_selection=development_selection, test_used=False, deployment_changed=False))
    comparison = {}
    for run_name, path in zip(runs, paths):
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert ck['windows_sha256'] == meta['windows_sha256']
        model = build_model(ck['variant'])
        model.load_state_dict(ck['state'], strict=True)
        scored, p = scored_probabilities(model, data, 'val', ck['variant'])
        result = evaluate_scores(scored, p, dict(half=[.5,.5], **ck['threshold_comparisons']))
        output = Path(args.out_dir)/run_name
        write_csv(output/'validation_predictions.csv', result_rows(data, scored, p, run_name, 'val'))
        write_json(output/'evaluation.json', dict(results=result, checkpoint_sha256=sha256(path),
            thresholds_selected_using='locked-train internal development only',
            external_validation_status='Previously inspected; exploratory weak-label windows',
            test_used=False, verified_early_aggression_claim=False))
        comparison[run_name] = result
        print(json.dumps(dict(run=run_name, onset_auc=result[HEADS[0]]['auroc'],
            onset_f1=result[HEADS[0]]['thresholds']['development_f1']['f1'],
            onset_ap=result[HEADS[0]]['average_precision'], active_auc=result[HEADS[1]]['auroc'])), flush=True)
    write_json(Path(args.out_dir)/'ablation_comparison.json', comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train-all','evaluate-all'])
    parser.add_argument('--data', default='data/pose_cache/advanced_v1/windows')
    parser.add_argument('--out-dir', default='outputs/diagnosis_v1/ablation')
    parser.add_argument('--weights-dir', default='weights/diagnosis_v1')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--variants', nargs='+', choices=['baseline','advanced'], default=['baseline','advanced'])
    parser.add_argument('--losses', nargs='+', choices=['unweighted','class_weighted'], default=['unweighted','class_weighted'])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error('Positive epoch count and batch size >=2 required')
    if args.command == 'train-all':
        for variant in args.variants:
            for loss in args.losses:
                train_one(args, variant, loss)
    else:
        evaluate_all(args)


if __name__ == '__main__':
    main()
