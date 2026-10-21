"""Audit existing weak ST-GCN experiments without changing labels or checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import Windows, build_model, load_windows, metrics, score
from src.foundation_io import read_csv, sha256, write_csv, write_json
from src.lock_splits import load_locked

HEADS = ('weak_first_onset_within_3s', 'weak_active_fight')


def independent_targets(t, intervals):
    """Independent scalar implementation of the documented weak supervision contract."""
    intervals = sorted(intervals)
    active = int(any(start <= t < end for start, end in intervals))
    active_known = all(abs(t - edge) >= .1 for pair in intervals for edge in pair)
    if not intervals:
        return [0, 0], [True, True]
    delta = intervals[0][0] - t
    onset = int(0 < delta <= 3.)
    onset_known = delta > .1 and abs(delta - 3.) >= .1
    return [onset, active], [onset_known, active_known]


def ranking(y, probabilities):
    """AP and trapezoidal PR-AUC are distinct; positive means y=1 everywhere."""
    y = np.asarray(y, int)
    p = np.asarray(probabilities, float)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all() or not set(y) <= {0, 1}:
        raise ValueError('Expected finite aligned binary labels and scores')
    result = metrics(y, p)
    if len(set(y)) < 2:
        result.update(pr_auc_trapezoid=None, independent_pairwise_auroc=None)
        return result
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = y[order].cumsum()[ends]
    recall = tp / y.sum()
    precision = tp / (ends + 1)
    result['pr_auc_trapezoid'] = float(np.trapezoid(np.r_[1., precision], np.r_[0., recall]))
    positive = p[y == 1, None]
    negative = p[y == 0][None, :]
    result['independent_pairwise_auroc'] = float(((positive > negative) + .5 * (positive == negative)).mean())
    if not np.isclose(result['auroc'], result['independent_pairwise_auroc']):
        raise AssertionError('AUROC implementations disagree')
    return result


def tune_threshold(y, probabilities, max_fpr=None):
    """Use development inputs only; no external validation labels are accepted here."""
    y = np.asarray(y, int)
    p = np.asarray(probabilities, float)
    if y.shape != p.shape or len(set(y)) != 2 or not np.isfinite(p).all():
        raise ValueError('Threshold selection needs both development classes')
    if max_fpr is not None and not 0 <= max_fpr <= 1:
        raise ValueError('FPR cap must be in [0,1]')
    candidates = np.unique(np.r_[p, np.nextafter(p.max(), np.inf)])
    candidates_reports = [metrics(y, p, float(t)) for t in candidates]
    if max_fpr is not None:
        candidates_reports = [r for r in candidates_reports if r['fp'] / (r['fp'] + r['tn']) <= max_fpr + 1e-12]
    # Identical predictions use the highest threshold. No label/score inversion is fitted.
    return max(candidates_reports, key=lambda r: (r['f1'], r['balanced_accuracy'], r['threshold']))


def scored_probabilities(model, data, role, variant, batch_size=32):
    scored = score(model, DataLoader(Windows(data, role), batch_size=batch_size, shuffle=False), variant)
    return scored, scored['logits'].sigmoid().numpy()


def result_rows(data, scored, probabilities, model_name, role):
    return [dict(model=model_name, role=role, index=int(index), video_id=str(data['video_id'][index]),
                 cutoff_s=float(data['cutoff_s'][index]), quality=float(data['quality'][index]),
                 onset_y=int(y[0]), onset_mask=bool(mask[0]), onset_score=float(prob[0]),
                 active_y=int(y[1]), active_mask=bool(mask[1]), active_score=float(prob[1]))
            for index, prob, y, mask in zip(scored['indices'].numpy(), probabilities,
                scored['targets'].numpy(), scored['masks'].numpy())]


def evaluate_scores(scored, probabilities, thresholds):
    reports = {}
    for h, head in enumerate(HEADS):
        mask = scored['masks'][:, h].numpy()
        y = scored['targets'][:, h].numpy()[mask]
        p = probabilities[mask, h]
        report = ranking(y, p)
        report['thresholds'] = {name: metrics(y, p, value[h]) for name, value in thresholds.items()}
        report['reversed_score_diagnostic_only'] = ranking(y, 1. - p)
        report['score_quantiles_by_target'] = {str(label): np.quantile(p[y == label], [0, .1, .25, .5, .75, .9, 1]).tolist()
                                             if (y == label).any() else [] for label in (0, 1)}
        reports[head] = report
    return reports


def dataset_audit(data, meta):
    locked = {r['video_id']: r for r in load_locked(meta['locked_manifest_path'])}
    mismatch = []
    raw_sources = {}
    for i, (key, t) in enumerate(zip(data['video_id'], data['cutoff_s'])):
        row = locked[str(key)]
        if row['split'] != ('val' if data['role'][i] == 'val' else 'train'):
            raise ValueError('Test or wrong-split source in materialized windows')
        expected, known = independent_targets(float(t), json.loads(row['intervals']))
        if not np.array_equal(data['targets'][i], expected) or not np.array_equal(data['target_mask'][i], known):
            mismatch.append(i)
        if str(key) not in raw_sources:
            # Original annotation vectors, independent of the normalized manifest targets.
            binary = np.loadtxt(row['source_annotation'], dtype=np.int8, ndmin=1)
            if not set(np.unique(binary).tolist()) <= {0, 1}:
                raise ValueError('Original annotation is not binary')
            edges = np.diff(np.r_[0, binary, 0])
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            ranges = json.loads(row['intervals'])
            # Dataset videos are 30 Hz here; verify each independently via video headers.
            import cv2
            cap = cv2.VideoCapture(row['video'])
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            converted = np.array(list(zip(starts / fps, ends / fps))) if len(starts) else np.empty((0, 2))
            matching = converted.shape == np.asarray(ranges).reshape(-1, 2).shape and np.allclose(converted, np.asarray(ranges).reshape(-1, 2), atol=1e-5)
            raw_sources[str(key)] = dict(split=row['split'], positive_frames=int(binary.sum()), frames=len(binary), fps=fps,
                intervals_match=bool(matching), frame_count_match=frame_count == len(binary),
                annotation_sha256=sha256(row['source_annotation']))
    assert not mismatch, mismatch
    assert all(r['intervals_match'] and r['frame_count_match'] for r in raw_sources.values())
    distributions = {}
    for role in ('fit', 'dev', 'val'):
        ix = data['role'] == role
        ids, counts = np.unique(data['video_id'][ix], return_counts=True)
        local_weight = {key: 1. / count for key, count in zip(ids, counts)}
        weights = np.array([local_weight[key] for key in data['video_id'][ix]])
        entry = dict(windows=int(ix.sum()), videos=len(ids), weak_fight_videos=sum(str(key).startswith('f_') for key in ids),
                     weak_no_fight_videos=sum(str(key).startswith('n_') for key in ids), heads={})
        for h, head in enumerate(HEADS):
            y = data['targets'][ix, h]
            known = data['target_mask'][ix, h]
            positive = y[known].sum()
            rate = positive / known.sum()
            sampler_rate = (weights * y * known).sum() / (weights * known).sum()
            pw = float(np.clip((known.sum() - positive) / positive, .25, 4))
            entry['heads'][head] = dict(positive=int(positive), negative=int(known.sum() - positive), masked=int((~known).sum()),
                prevalence=float(rate), old_video_sampler_effective_prevalence=float(sampler_rate),
                raw_count_pos_weight=pw,
                pos_weight_for_old_video_sampler=float((weights * (1-y) * known).sum() / (weights*y*known).sum()))
        distributions[role] = entry
    real = data['temporal_mask']
    dt = np.diff(data['frame_times'], axis=1)[real[:, 1:] & real[:, :-1]]
    assert np.isfinite(data['sk']).all() and np.isfinite(data['quality']).all()
    assert (dt > 0).all()
    assert not (data['joint_mask'] & ~real[:, :, None, None]).any()
    assert (data['sk'][~data['joint_mask']] == 0).all()
    return dict(distributions=distributions, independently_recomputed_target_mismatches=mismatch,
        original_annotation_sources=raw_sources, positive_class_definition='1=next original first fight onset within 3s; active head 1=inside original weak fight interval',
        masked_forecast_policy='After first onset and boundary guards remain masked; never background',
        preprocessing=dict(finite=True, no_valid_joints_in_padding=True, masked_joints_zero=True,
            sampled_dt_quantiles=np.quantile(dt, [0, .5, 1]).tolist(), skeleton_shape=list(data['sk'].shape)),
        independent_precursor_labels_created=0, test_sources_used=0)


def audit(args):
    torch.set_num_threads(2)
    data, meta = load_windows(args.data)
    report = dataset_audit(data, meta)
    all_rows = []
    per_video = []
    report['models'] = {}
    for variant in ('baseline', 'advanced'):
        path = f'weights/{variant}_weak_onset_guarded_v1.pt'
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert ck['trained_heads'] == list(HEADS)
        assert ck['windows_sha256'] == meta['windows_sha256']
        model = build_model(variant)
        model.load_state_dict(ck['state'], strict=True)
        dev, dev_p = scored_probabilities(model, data, 'dev', variant)
        tuned = {}
        for policy, cap in (('development_f1', None), ('development_fpr10', .1)):
            tuned[policy] = [tune_threshold(dev['targets'][:, h].numpy()[dev['masks'][:, h].numpy()],
                dev_p[dev['masks'][:, h].numpy(), h], cap)['threshold'] for h in range(2)]
        thresholds = dict(half=[.5, .5], historical_guarded=ck['thresholds'], **tuned)
        results = {}
        for role in ('fit', 'dev', 'val'):
            scored, prob = (dev, dev_p) if role == 'dev' else scored_probabilities(model, data, role, variant)
            results[role] = evaluate_scores(scored, prob, thresholds)
            rows = result_rows(data, scored, prob, variant, role)
            all_rows.extend(rows)
            for key in sorted({r['video_id'] for r in rows}):
                select = [r for r in rows if r['video_id'] == key]
                for head, prefix in zip(HEADS, ('onset', 'active')):
                    eligible = [r for r in select if r[prefix + '_mask']]
                    if eligible:
                        y = [r[prefix + '_y'] for r in eligible]
                        p = [r[prefix + '_score'] for r in eligible]
                        per_video.append(dict(model=variant, role=role, video_id=key, head=head,
                            mean_score=float(np.mean(p)), **ranking(y, p)))
            if role == 'val':
                old = read_csv(f'outputs/advanced_v1/{variant}_guarded/predictions.csv')
                assert len(old) == len(rows)
                deltas = [abs(float(a[field]) - b[new]) for a, b in zip(old, rows)
                    for field, new in (('weak_onset_score', 'onset_score'), ('fight_score', 'active_score'))]
                results['saved_probability_max_abs_difference'] = max(deltas)
                assert max(deltas) < 1e-6
        report['models'][variant] = dict(checkpoint_sha256=sha256(path), thresholds=thresholds, results=results,
            probability_direction='sigmoid(logit[h])=P(target[h]==1), no softmax or complement',
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            original_training=json.loads(Path(f'outputs/advanced_v1/{variant}/training.json').read_text()))
    report['score_inversion_policy'] = 'Complement scores are diagnostic only; no direction is selected from external validation.'
    report['evaluation_status'] = 'Previously inspected external validation; exploratory weak-label window diagnostics, not independent final performance.'
    write_json(Path(args.out_dir) / 'audit.json', report)
    write_csv(Path(args.out_dir) / 'existing_predictions.csv', all_rows)
    write_csv(Path(args.out_dir) / 'existing_per_video.csv', per_video)
    print(json.dumps(dict(distributions=report['distributions'], target_mismatches=len(report['independently_recomputed_target_mismatches']),
        directions={v:{r:report['models'][v]['results'][r][HEADS[0]]['auroc'] for r in ('fit','dev','val')} for v in report['models']}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/pose_cache/advanced_v1/windows')
    parser.add_argument('--out-dir', default='outputs/diagnosis_v1')
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
