"""Summarize frozen weak-label comparison and paired source-cluster uncertainty."""
import argparse
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from src.advanced_experiment import metrics
from src.foundation_io import read_csv, write_csv, write_json, write_new
from src.improve_model import HEADS


def paired_bootstrap(y, before, after, sources, before_threshold, after_threshold, draws=1000, seed=73):
    y=np.asarray(y);before=np.asarray(before);after=np.asarray(after);sources=np.asarray(sources)
    keys=sorted(set(sources.tolist()));rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        selected=rng.choice(keys,len(keys),replace=True)
        ix=np.concatenate([np.flatnonzero(sources==key) for key in selected])
        if len(set(y[ix].tolist()))<2:continue
        b=metrics(y[ix],before[ix],before_threshold);a=metrics(y[ix],after[ix],after_threshold)
        samples.append([a[k]-b[k] for k in ('auroc','f1','balanced_accuracy')])
    if not samples:raise ValueError('Insufficient source diversity for paired bootstrap')
    intervals=np.quantile(samples,[.025,.975],axis=0)
    return dict(independent_sources=len(keys),draws=draws,valid_draws=len(samples),seed=seed,
        deltas_95_percent={k:[float(v) for v in intervals[:,i]] for i,k in enumerate(('auroc','f1','balanced_accuracy'))},
        method='paired resampling of entire source-video clusters; exploratory intervals, overlapping windows retained within source')


def build(args):
    root=Path(args.out_dir);evaluation=json.loads((root/'evaluation.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    rows=read_csv(root/'validation_predictions.csv')
    uncertainty={};comparison=[]
    for cohort in ('original','quality_matched'):
        uncertainty[cohort]={}
        for h,head in enumerate(HEADS):
            label='onset' if h==0 else 'active'
            b=[r for r in rows if r['cohort']==cohort and r['model']=='baseline' and r[label+'_mask']=='True']
            a=[r for r in rows if r['cohort']==cohort and r['model']=='candidate' and r[label+'_mask']=='True']
            assert [(r['video_id'],r['cutoff_s'],r[label+'_y']) for r in a]==[(r['video_id'],r['cutoff_s'],r[label+'_y']) for r in b]
            baseline=evaluation['cohorts'][cohort]['baseline']['guarded'][head]
            candidate=evaluation['cohorts'][cohort]['candidate']['guarded'][head]
            uncertainty[cohort][head]=paired_bootstrap([int(r[label+'_y']) for r in a],
                [float(r[label+'_score']) for r in b],[float(r[label+'_score']) for r in a],
                [r['video_id'] for r in a],baseline['threshold'],candidate['threshold'])
        for model,policies in evaluation['cohorts'][cohort].items():
            for policy,heads in policies.items():
                for head,m in heads.items():
                    comparison.append(dict(cohort=cohort,model=model,threshold_policy=policy,head=head,**m,
                        false_positive_rate=m['fp']/max(1,m['fp']+m['tn'])))
    write_json(root/'paired_bootstrap.json',uncertainty)
    write_csv(root/'comparison.csv',comparison)
    data=json.loads(Path('data/pose_cache/model_improvement_expanded_v1/expansion_summary.json').read_text())
    tree=ET.parse('outputs/model_improvement_v1/release_tests.xml').getroot()
    suites=list(tree.iter('testsuite'))
    xml_tests=sum(int(s.get('tests',0)) for s in suites);failures=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    passed=re.search(r'(?m)^(\d+) passed,.*?(\d+) subtests passed',Path('outputs/model_improvement_v1/release_tests.log').read_text())
    if not passed:raise ValueError('Final pytest summary is missing')
    tests,subtests=map(int,passed.groups())
    if tests+subtests!=xml_tests:raise ValueError('Pytest/JUnit counts disagree')
    lines=['# Component 03 test and model improvement summary','',
        'Completed locally on 21 October 2026. These are exploratory weak-label results, not independently verified early-aggression accuracy.','',
        '## Tests','',f'- Initial suite: 225 tests and 62 subtests passed.',
        f'- Final C3/backend/optional suite: {tests} tests + {subtests} subtests passed; {failures} failures/errors.',
        '- Frontend: 13 tests passed. C3 production build passed.',
        '', '## Work completed','',
        '- Locked dataset/split guards, causal YOLO-Pose/ByteTrack caches and trailing pose windows.',
        '- Baseline and multi-stream ST-GCN, separate weak-onset and active-fight heads, quality abstention and provenance.',
        '- Original shared dashboard: one Recorded video upload, queued C3 inference, annotated playback, scores, review and export.',
        '- C3 research tab and obsolete dashboard copies removed. C4-only backend modules archived separately.',
        '- Team-guide component entry point: run.py. Research logbook and daily diary PDFs prepared.',
        '', '## Improvement work in this run','',
        '- Implemented 240-dimensional masked motion/geometry features with linear and MLP learned heads.',
        '- Tested person-order/translation/reflection/scale invariance, masked and padded values, causality and source leakage guards.',
        '- Initial four-candidate experiment was rejected: active ranking improved modestly, but guarded F1 and onset results regressed.',
        f'- Added {data["added_sources"]} locked-training contexts: 24 weak no-fight and 8 weak fight; {data["added_windows"]} additional quality-matched windows.',
        '- All original rows, labels and fit/dev/val roles were preserved. The 204 quality-matched validation windows are byte-identical.',
        '- Trained four additional kinematic candidates and two ST-GCN candidates on expanded data.',
        '- Selected model family members and per-head positive ensemble weights on internal development only; thresholds frozen before external validation.',
        '', '## Same original validation cohort','',
        '| Head / model | N | Accuracy | Balanced accuracy | AUROC | F1 | Recall | FPR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for head in HEADS:
        for model in ('baseline','advanced','candidate'):
            m=evaluation['cohorts']['original'][model]['guarded'][head]
            lines.append(f'| {head} / {model} | {m["n"]} | {m["accuracy"]:.3f} | {m["balanced_accuracy"]:.3f} | {m["auroc"]:.3f} | {m["f1"]:.3f} | {m["recall"]:.3f} | {m["fp"]/max(1,m["fp"]+m["tn"]):.3f} |')
    lines+=['','Thresholds use train-only development. F1 above uses guarded reference thresholds; old baseline active-fight notifications were disabled, so it is a score-level comparison, not measured deployed alert accuracy.',
        '',f'Predeclared descriptive acceptance passed: **{evaluation["predeclared_original_cohort_acceptance_passed"]}**.',
        '', 'The runtime default remains the existing guarded baseline. The candidate improves ranking point estimates but regresses guarded active-fight F1, so it was not promoted. No default checkpoint, threshold or operational notification policy was changed.',
        '',f'Candidate checkpoint: `{selection["checkpoint"]}`',f'Candidate SHA-256: `{selection["checkpoint_sha256"]}`',
        '',f'Per-head weights [baseline, advanced, kinematic]: `{selection["blend_weights"]}`.',
        '', '## Practical limits','',
        '- Zero independently verified precursor-positive videos. Confirmed 1-3 second pre-contact prediction remains unproven.',
        '- Validation has only 13 contributing source videos and was inspected in earlier experiments. Windows overlap; no fresh final-test accuracy claim is made.',
        '- comparison.csv includes both original and quality-matched cohorts. paired_bootstrap.json reports source-cluster uncertainty; a small point improvement is not necessarily statistically established.',
        '- Raw accuracy must be compared with the always-negative baseline because weak labels are imbalanced.',
        '- No test source was used for fitting, source selection, normalization, epoch/model selection or threshold selection. Original weights and source data are preserved.',
        '- Operational notifications remain disabled. Scores remain uncalibrated.',
        '', '## References','',
        '- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): class weighting controls precision/recall trade-offs; it does not prove accuracy.',
        '- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): source groups must not overlap between fitting and evaluation.',
        '']
    smoke=root/'recorded_video_smoke/prediction.json'
    if smoke.is_file():
        run=json.loads(smoke.read_text())
        lines+=['## Recorded-video smoke check','',
            f'- Existing F_9 review clip: {run["timing"]["valid_decisions"]}/{run["timing"]["decisions"]} usable decisions; compatible pose cache reused.',
            f'- Candidate returned {run["incident_candidates"]} incident candidates. The reported violence-detection miss is still unresolved.',
            '- Annotated MP4, scores, incident JSON and timing/provenance were produced successfully.',
            '- This single clip was used only after freezing selection/evaluation, as a runtime smoke test; it was not used for fitting or threshold tuning.','']
    write_new(root/'SUMMARY.md','\n'.join(lines))
    print(json.dumps(dict(tests=tests,subtests=subtests,failures=failures,acceptance=evaluation['predeclared_original_cohort_acceptance_passed'],uncertainty=uncertainty),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',default='outputs/model_improvement_expanded_v1/headwise')
    build(p.parse_args())
