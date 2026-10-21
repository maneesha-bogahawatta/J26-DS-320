"""Reproducible weak-score reliability diagnostics from existing cached predictions.

Fits an explicitly diagnostic positive temperature on internal development only.
No model, threshold, runtime selection, annotation or source recording is changed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.foundation_io import read_csv, sha256, write_csv, write_json, write_new

HEADS = {'onset': 'weak_first_onset_within_3s', 'active': 'weak_active_fight'}
ROLES = ('fit', 'dev', 'val')


def aligned_scores(labels, scores):
    y = np.asarray(labels, dtype=float)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Expected aligned finite one-dimensional labels and scores')
    if not np.isin(y, (0., 1.)).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Binary labels and sigmoid scores in [0,1] required')
    return y, p


def temperature_scale(scores, temperature):
    p = np.asarray(scores, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Finite scores in [0,1] required')
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive; score inversion prohibited')
    # Numeric clipping is explicit. Interior scores preserve ordering and the .5 cutoff.
    bounded = p.clip(1e-8, 1 - 1e-8)
    logits = (np.log(bounded) - np.log1p(-bounded)) / temperature
    return np.exp(-np.logaddexp(0., -logits))


def reliability_metrics(labels, scores, bins=10, reference_prevalence=None):
    y, p = aligned_scores(labels, scores)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError('Positive integer reliability-bin count required')
    if reference_prevalence is not None and (not np.isfinite(reference_prevalence) or not 0 <= reference_prevalence <= 1):
        raise ValueError('Reference prevalence must be in [0,1]')
    if not len(y):
        return dict(n=0, positives=0, prevalence=None, brier=None, nll=None,
                    ece=None, maximum_bin_gap=None, bins=[], reference_brier=None)
    clipped = p.clip(1e-8, 1 - 1e-8)
    allocation = np.minimum((p * bins).astype(int), bins - 1)
    entries = []
    ece = 0.
    for index in range(bins):
        selected = allocation == index
        count = int(selected.sum())
        mean_score = float(p[selected].mean()) if count else None
        observed = float(y[selected].mean()) if count else None
        gap = abs(mean_score - observed) if count else None
        if count:
            ece += count / len(y) * gap
        entries.append(dict(bin=index, lower=index / bins, upper=(index + 1) / bins,
                            upper_inclusive=index == bins - 1, n=count,
                            mean_score=mean_score, observed_fraction=observed, absolute_gap=gap))
    return dict(n=len(y), positives=int(y.sum()), prevalence=float(y.mean()),
                brier=float(np.square(p - y).mean()),
                nll=float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
                ece=float(ece), maximum_bin_gap=max(entry['absolute_gap'] or 0. for entry in entries),
                bins=entries, reference_brier=float(np.square(reference_prevalence - y).mean())
                if reference_prevalence is not None else None)


def fit_temperature(labels, scores, *, role):
    """A positive transform is selected from dev only, never validation/test input."""
    if role != 'dev':
        raise ValueError('Temperature fitting accepts internal-development role dev only')
    y, p = aligned_scores(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError('Temperature fitting requires both development classes')
    # Frozen deterministic grid. Include identity and break equivalent fits toward it.
    candidates = np.unique(np.r_[1., np.exp(np.linspace(np.log(.05), np.log(20.), 241))])
    objective = [(reliability_metrics(y, temperature_scale(p, value))['nll'],
                  abs(np.log(value)), float(value)) for value in candidates]
    loss, _, selected = min(objective)
    return dict(temperature=selected, development_nll=loss, source_role='dev',
                grid_min=.05, grid_max=20., grid_count=len(candidates),
                boundary_solution=bool(np.isclose(selected, .05) or np.isclose(selected, 20.)),
                external_validation_used_for_fitting=False,
                interpretation='Diagnostic weak-label score scaling; not validated operational calibration')


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ('True', 'true', '1', 1):
        return True
    if value in ('False', 'false', '0', 0):
        return False
    raise ValueError('Explicit boolean mask required')


def validate_cohort(rows, metadata):
    """Reject test/unknown roles, duplicate windows, and shared source groups."""
    if metadata.get('test_sources_used') != 0:
        raise ValueError('A test-free materialized cohort is required')
    expected_roles = {entry['video_id']: entry['role'] for entry in metadata['coverage']}
    source_groups = metadata['source_groups']
    result = []
    windows = set()
    groups = {role: set() for role in ROLES}
    for row in rows:
        item = dict(row)
        role, key = item['role'], item['video_id']
        if role not in ROLES or expected_roles.get(key) != role or key not in source_groups:
            raise ValueError('Test, unknown or wrong-role source in score table')
        groups[role].add(source_groups[key])
        item['index'] = int(item['index'])
        identity = (item['model'], item['index'])
        if not item['model'] or identity in windows:
            raise ValueError('Duplicate window or missing model identity')
        windows.add(identity)
        item['cutoff_s'] = float(item['cutoff_s'])
        if not np.isfinite(item['cutoff_s']) or item['cutoff_s'] < 0:
            raise ValueError('Nonnegative finite prediction timestamp required')
        for prefix in HEADS:
            item[prefix + '_mask'] = _boolean(item[prefix + '_mask'])
            y, p = aligned_scores([item[prefix + '_y']], [item[prefix + '_score']])
            item[prefix + '_y'], item[prefix + '_score'] = int(y[0]), float(p[0])
        result.append(item)
    for a, b in (('fit', 'dev'), ('fit', 'val'), ('dev', 'val')):
        if groups[a] & groups[b]:
            raise ValueError('Source-group leakage between model roles')
    if not result:
        raise ValueError('No scored windows available')
    return result, dict(test_sources_used=0, cross_role_source_groups=0,
                       source_groups={role: len(value) for role, value in groups.items()},
                       scored_rows=len(result),
                       fitting_policy='Models fitted on fit; diagnostic temperatures selected on dev only')


def audit_scores(rows, metadata, bins=10):
    rows, integrity = validate_cohort(rows, metadata)
    models = {}
    comparison = []
    bin_rows = []
    for model in sorted({row['model'] for row in rows}):
        model_rows = [row for row in rows if row['model'] == model]
        models[model] = {}
        for prefix, head in HEADS.items():
            selected = {role: [row for row in model_rows if row['role'] == role and row[prefix + '_mask']]
                        for role in ROLES}
            dev = selected['dev']
            y_dev = [row[prefix + '_y'] for row in dev]
            p_dev = [row[prefix + '_score'] for row in dev]
            transform = fit_temperature(y_dev, p_dev, role='dev')
            development_prevalence = float(np.mean(y_dev))
            roles = {}
            for role, eligible in selected.items():
                y = [row[prefix + '_y'] for row in eligible]
                raw = [row[prefix + '_score'] for row in eligible]
                scaled = temperature_scale(raw, transform['temperature'])
                roles[role] = dict(supervised_windows=len(eligible),
                    supervised_source_groups=len({metadata['source_groups'][row['video_id']] for row in eligible}),
                    masked_windows=sum(row['role'] == role for row in model_rows) - len(eligible),
                    threshold_half_decisions_unchanged=bool(np.array_equal(np.asarray(raw) >= .5, scaled >= .5)))
                for policy, scores in (('raw_sigmoid', raw), ('dev_temperature_diagnostic', scaled)):
                    measured = reliability_metrics(y, scores, bins=bins, reference_prevalence=development_prevalence)
                    roles[role][policy] = measured
                    comparison.append(dict(model=model, head=head, role=role, policy=policy,
                        **{key: value for key, value in measured.items() if key != 'bins'},
                        temperature=1. if policy == 'raw_sigmoid' else transform['temperature'],
                        reference_prevalence_source='dev', reference_prevalence=development_prevalence))
                    bin_rows.extend(dict(model=model, head=head, role=role, policy=policy, **entry)
                                    for entry in measured['bins'])
            models[model][head] = dict(transform=transform, roles=roles)
    report = dict(schema='c3_weak_reliability_diagnostic_v1', cohort_integrity=integrity,
        models=models, score_semantics='Uncalibrated sigmoid scores for original weak-label targets',
        task_definitions={HEADS['onset']: 'First ORIGINAL ANNOTATED onset in next 3 seconds; not confirmed physical contact',
                          HEADS['active']: 'Inside original weak UBI fight interval',
                          'verified_pre_contact_aggression': 'BLOCKED: no independently verified precursor-positive labels'},
        runtime_calibration_changed=False, runtime_thresholds_changed=False,
        operational_alerts_enabled=False, verified_early_aggression_claim=False,
        limitations=['Development has already selected model epochs and thresholds; its calibration fit is optimistic.',
                     'External validation was previously inspected; results are exploratory diagnostics.',
                     'Overlapping windows are correlated; bin statistics are descriptive, with no IID confidence intervals.',
                     'A positive temperature cannot repair weak ranking, label semantics or source-domain shift.',
                     'ECE depends on binning and prevalence; calibrated diagnostic scores are not validated violence probabilities.',
                     'No locked-test recordings or labels are used; no full-recording false-alert rate is inferred.'])
    return report, comparison, bin_rows


def run(args):
    metadata = json.loads(Path(args.metadata).read_text(encoding='utf-8'))
    report, comparison, bin_rows = audit_scores(read_csv(args.predictions), metadata, bins=args.bins)
    report['inputs'] = {name: dict(path=str(Path(path).resolve()), sha256=sha256(path))
                        for name, path in (('predictions', args.predictions), ('metadata', args.metadata))}
    destination = Path(args.out_dir)
    write_json(destination / 'reliability.json', report)
    write_csv(destination / 'reliability_metrics.csv', comparison)
    write_csv(destination / 'reliability_bins.csv', bin_rows)
    lines = ['Weak-model calibration diagnostics', '',
        'All temperatures below were selected on locked-training internal development only. '
        'They are diagnostic transformations and are not loaded by inference. '
        'No checkpoint, alert threshold or label was changed.', '',
        '| Model | Head | Dev temperature | Validation raw Brier | Diagnostic Brier | Raw ECE | Diagnostic ECE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model, heads in report['models'].items():
        for head, result in heads.items():
            raw = result['roles']['val']['raw_sigmoid']
            diagnostic = result['roles']['val']['dev_temperature_diagnostic']
            values = [result['transform']['temperature'], raw['brier'], diagnostic['brier'], raw['ece'], diagnostic['ece']]
            lines.append('| ' + ' | '.join([model, head] + [f'{value:.6f}' if value is not None else 'unavailable' for value in values]) + ' |')
    lines += ['', 'A lower Brier score or ECE here does not establish verified pre-contact accuracy. '
        'Calibration may regress under source/prevalence shift, and overlapping validation windows do not provide independent evidence.', '',
        'See reliability.json for fit/dev/val reliability bins, masks, source counts and limitations.']
    write_new(destination / 'comparison.md', '\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str((destination / 'reliability.json').resolve()),
                          models=list(report['models']), test_sources_used=0,
                          runtime_calibration_changed=False), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='outputs/diagnosis_v1/existing_predictions.csv')
    parser.add_argument('--metadata', default='data/pose_cache/advanced_v1/windows/metadata.json')
    parser.add_argument('--out-dir', default='outputs/development_completion_v1/model_reliability')
    parser.add_argument('--bins', type=int, default=10)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
