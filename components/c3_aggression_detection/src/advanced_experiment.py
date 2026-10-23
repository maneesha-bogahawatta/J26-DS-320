"""Train-only development selection and held-out weak-label diagnostics.

No verified precursor claim: the forecast target is explicitly the next weak
UBI first annotated onset, not a human-verified precursor label.
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.advanced_stgcn import AdvancedSTGCN
from src.advanced_data import PROTOCOL
from src.stgcn_baseline import JointOnlySTGCN
from src.foundation_io import sha256, write_json, write_csv

SCHEMA = 'c3_weak_onset_stgcn_v1'


def metrics(y, score, threshold=.5):
    y = np.asarray(y, int); score = np.asarray(score, float)
    if not len(y): return dict(n=0)
    if not np.isfinite(score).all(): raise ValueError('Nonfinite scores')
    pred = score >= threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    precision = tp/max(1, tp+fp); recall = tp/max(1, tp+fn)
    auroc = ap = None
    if 0 < y.sum() < len(y):
        order = np.argsort(-score, kind='stable'); yy = y[order]; ss = score[order]
        ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
        positives = yy.cumsum()[ends]; negatives = (1-yy).cumsum()[ends]
        recalls = positives/y.sum(); fprs = negatives/(len(y)-y.sum())
        auroc = float(np.trapezoid(np.r_[0., recalls], np.r_[0., fprs]))
        ap = float((np.diff(np.r_[0., recalls]) * positives/(ends+1)).sum())
    p = score.clip(1e-7, 1-1e-7)
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()), tp=tp, tn=tn,
        fp=fp, fn=fn, accuracy=(tp+tn)/len(y), precision=precision, recall=recall,
        f1=2*precision*recall/max(1e-9, precision+recall),
        balanced_accuracy=.5*(recall+tn/max(1, tn+fp)) if 0 < y.sum() < len(y) else None, auroc=auroc, average_precision=ap,
        bce=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
        threshold=float(threshold), always_negative_accuracy=float((1-y).mean()))


class Windows(Dataset):
    def __init__(self, data, role, augment=False):
        self.data = data; self.indices = np.flatnonzero(data['role'] == role)
        self.augment = augment
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        keys = ('sk', 'temporal_mask', 'joint_mask', 'frame_times', 'targets', 'target_mask')
        item = {k: torch.from_numpy(self.data[k][j].copy()) for k in keys}
        if self.augment:
            if torch.rand(()) < .5:
                item['sk'] = item['sk'].flip(2); item['joint_mask'] = item['joint_mask'].flip(2)
            if torch.rand(()) < .5: item['sk'][..., 0] *= -1
            dropped = torch.rand(item['joint_mask'].shape) < .025
            item['joint_mask'] &= ~dropped
            item['sk'][~item['joint_mask']] = 0
        item['index'] = j
        return item


def load_windows(folder):
    folder = Path(folder)
    meta = json.loads((folder/'metadata.json').read_text())
    if meta.get('protocol') != PROTOCOL or sha256(folder/'windows.npz') != meta['windows_sha256']:
        raise ValueError('Invalid/changed weak-supervision window cache')
    if sha256(meta['locked_manifest_path']) != meta['locked_manifest_sha256']:
        raise ValueError('Locked split manifest changed')
    with np.load(folder/'windows.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    groups = {role: set(data['video_id'][data['role'] == role].tolist()) for role in ('fit', 'dev', 'val')}
    if any(groups[a] & groups[b] for a,b in (('fit','dev'),('fit','val'),('dev','val'))):
        raise ValueError('Video leakage in materialized roles')
    source_groups = {role: {meta['source_groups'][k] for k in keys} for role, keys in groups.items()}
    if any(source_groups[a] & source_groups[b] for a,b in (('fit','dev'),('fit','val'),('dev','val'))):
        raise ValueError('Shared source footage across fitting/development/validation')
    if not set(data['role'].tolist()) <= {'fit','dev','val'} or meta['test_sources_used']:
        raise ValueError('Test/unknown fitting role prohibited')
    return data, meta


def forward(model, batch, variant):
    args = [batch['sk'], batch['temporal_mask'], batch['joint_mask']]
    return model(*args, *([batch['frame_times']] if variant != 'baseline' else []))['logits']


def build_model(variant):
    if variant == 'baseline': return JointOnlySTGCN()
    if variant == 'advanced': return AdvancedSTGCN()
    if variant in ('kinematic_linear', 'kinematic_mlp'):
        from src.kinematic_model import KinematicTemporalModel
        return KinematicTemporalModel(32 if variant == 'kinematic_mlp' else 0)
    if variant == 'headwise':
        from src.headwise_model import HeadwiseC3
        return HeadwiseC3()
    if variant == 'optimized_readout':
        from src.optimized_readout import OptimizedReadoutC3
        return OptimizedReadoutC3()
    raise ValueError(f'Unknown C3 model variant: {variant}')


def score(model, loader, variant):
    model.eval(); logits = []; targets = []; masks = []; indices = []
    with torch.no_grad():
        for batch in loader:
            out = forward(model, batch, variant)
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite model output')
            logits.append(out); targets.append(batch['targets']); masks.append(batch['target_mask']); indices.append(batch['index'])
    return dict(logits=torch.cat(logits), targets=torch.cat(targets), masks=torch.cat(masks), indices=torch.cat(indices))


def head_reports(scored, thresholds):
    probs = scored['logits'].sigmoid().numpy(); y = scored['targets'].numpy(); mask = scored['masks'].numpy()
    return {name: metrics(y[mask[:, h], h], probs[mask[:, h], h], thresholds[h])
            for h, name in enumerate(('weak_first_onset_within_3s', 'weak_active_fight'))}


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    data, meta = load_windows(args.data)
    # No external-validation sample is read by a fitting loader or threshold selection.
    fit = Windows(data, 'fit', augment=True); dev = Windows(data, 'dev')
    for role, ds in (('fit', fit), ('dev', dev)):
        for h in range(2):
            target = data['targets'][ds.indices, h][data['target_mask'][ds.indices, h]]
            if not (len(target) and target.min() == 0 and target.max() == 1):
                raise ValueError(f'{role} needs both weak classes for head {h}')
    if Path(args.out).exists(): raise FileExistsError('Checkpoint exists; select a new experiment version')
    counts = {k: int((data['video_id'][fit.indices] == k).sum()) for k in set(data['video_id'][fit.indices].tolist())}
    weights = [1/counts[k] for k in data['video_id'][fit.indices]]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights, len(fit), replacement=True, generator=generator)
    loaders = [DataLoader(fit, batch_size=args.batch_size, sampler=sampler, num_workers=0),
               DataLoader(dev, batch_size=args.batch_size, shuffle=False, num_workers=0)]
    positive = (data['targets'][fit.indices]*data['target_mask'][fit.indices]).sum(0)
    negative = data['target_mask'][fit.indices].sum(0)-positive
    pos_weight = torch.tensor(negative/positive, dtype=torch.float32).clamp(.25, 4)
    config = dict(variant=args.variant, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=.0003, weight_decay=.01, pos_weight=pos_weight.tolist(),
        protocol=PROTOCOL, development_policy='Only source-grouped subset of locked train',
        threshold_policy='Maximize F1 on train-only development windows at thresholds 0.05..0.95',
        windows_sha256=meta['windows_sha256'], test_sources_used=0)
    write_json(Path(args.report_dir)/'protocol.json', config)
    model = build_model(args.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    history = []; best = math.inf; best_state = None; stale = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train(); total = n = 0
        for batch in loaders[0]:
            logits = forward(model, batch, args.variant)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['targets'],
                pos_weight=pos_weight, reduction='none')
            mask = batch['target_mask'].float()
            loss = ((loss*mask).sum(0)/mask.sum(0).clamp_min(1)).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total += float(loss.detach()); n += 1
        scored = score(model, loaders[1], args.variant)
        losses = torch.nn.functional.binary_cross_entropy_with_logits(scored['logits'], scored['targets'], reduction='none')
        objective = float(((losses*scored['masks']).sum(0)/scored['masks'].sum(0).clamp_min(1)).mean())
        row = dict(epoch=epoch+1, train_balanced_bce=total/n, development_bce=objective,
                   elapsed_s=round(time.monotonic()-started, 2))
        history.append(row); print(json.dumps(row), flush=True)
        if objective < best:
            best = objective; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= 3: break
    model.load_state_dict(best_state)
    scored = score(model, loaders[1], args.variant)
    thresholds = []
    for h in range(2):
        mask = scored['masks'][:, h].numpy()
        y = scored['targets'][:, h].numpy()[mask]; p = scored['logits'][:, h].sigmoid().numpy()[mask]
        thresholds.append(float(max(np.arange(.05, .951, .025), key=lambda t: metrics(y, p, t)['f1'])))
    checkpoint = dict(schema=SCHEMA, state=best_state, variant=args.variant, protocol=config,
        thresholds=thresholds, trained_heads=meta['target_heads'], history=history,
        windows_sha256=meta['windows_sha256'], locked_manifest_sha256=meta['locked_manifest_sha256'],
        feature_contract=meta['feature_contract'], verified_early_aggression_claim=False,
        supervision='weak_ubi_first_onset_and_binary_fight_only', calibration='UNVALIDATED')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open('xb') as stream: torch.save(checkpoint, stream)
    write_json(Path(args.report_dir)/'training.json', dict(history=history, thresholds=thresholds,
        selected_epoch=min(history, key=lambda r:r['development_bce'])['epoch'],
        elapsed_s=round(time.monotonic()-started, 2), development_metrics=head_reports(scored, thresholds),
        model_trained=True, supervision=checkpoint['supervision'], verified_early_aggression_claim=False))


def evaluate(args):
    torch.set_num_threads(2)
    data, meta = load_windows(args.data)
    ck = torch.load(args.out, map_location='cpu', weights_only=True)
    if ck.get('schema') != SCHEMA or ck['windows_sha256'] != meta['windows_sha256']:
        raise ValueError('Checkpoint/data contract mismatch')
    model = build_model(ck['variant']); model.load_state_dict(ck['state'], strict=True)
    val = Windows(data, 'val'); loader = DataLoader(val, batch_size=32, shuffle=False)
    scored = score(model, loader, ck['variant'])
    report = dict(schema='c3_advanced_weak_evaluation_v1', variant=ck['variant'],
        checkpoint_sha256=sha256(args.out), protocol=PROTOCOL, threshold_source='train-only internal development',
        metrics=head_reports(scored, ck['thresholds']), test_sources_used=0,
        verified_early_aggression_claim=False, calibration='UNVALIDATED',
        summary=meta['summary'], note='Weak annotation agreement on retrospective selected contexts; overlapping windows; not verified accuracy or full-recording event evaluation.')
    probabilities = scored['logits'].sigmoid().numpy()
    rows = []
    for index, prob, y, mask in zip(scored['indices'].numpy(), probabilities, scored['targets'].numpy(), scored['masks'].numpy()):
        rows.append(dict(video_id=str(data['video_id'][index]), cutoff_s=float(data['cutoff_s'][index]),
            weak_onset_score=float(prob[0]), fight_score=float(prob[1]),
            weak_onset_target=int(y[0]), weak_onset_mask=bool(mask[0]),
            weak_fight_target=int(y[1]), weak_fight_mask=bool(mask[1])))
    seen = set(json.loads(Path('data/pose_cache/quick_v1/pilot_val_index.json').read_text())['entries'])
    for label, select in (('historically_seen_pilot', [r for r in rows if r['video_id'] in seen]),
                          ('new_context_sources', [r for r in rows if r['video_id'] not in seen])):
        report[label] = {name: metrics([r[target] for r in select if r[mask]],
            [r[score_name] for r in select if r[mask]], ck['thresholds'][h]) for h, name, target, mask, score_name in
            ((0,'weak_first_onset_within_3s','weak_onset_target','weak_onset_mask','weak_onset_score'),
             (1,'weak_active_fight','weak_fight_target','weak_fight_mask','fight_score'))}
    write_json(Path(args.report_dir)/'evaluation.json', report)
    write_csv(Path(args.report_dir)/'predictions.csv', rows)
    print(json.dumps(report['metrics'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train','evaluate'])
    p.add_argument('--data', default='data/pose_cache/advanced_v1/windows')
    p.add_argument('--variant', choices=['baseline','advanced'], default='advanced')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--out', default='weights/advanced_weak_onset_v1.pt')
    p.add_argument('--report-dir', default='outputs/advanced_v1/advanced')
    args = p.parse_args()
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__': main()
