**ST-GCN diagnosis: measured weak-label results**

Targets, original source annotations, saved score direction and AUROC implementations agree. There is no demonstrated global label/score inversion. Complementing the advanced onset score raises external-validation AUROC from 0.232 to 0.768 but lowers development AUROC from 0.601 to 0.399; it is a diagnostic, never a deployed correction.

All new checkpoints and thresholds use source-grouped internal development from locked training data. External validation was historically inspected and is exploratory. No test source, fabricated precursor label, original artifact replacement or deployment change is involved.

**Class distributions (positive / negative / masked)**

| Role | Onset | Active fight | Usable fight / no-fight sources |
|---|---:|---:|---:|
| fit | 251 / 210 / 888 | 444 / 905 / 0 | 87 / 5 |
| dev | 92 / 76 / 233 | 113 / 288 / 0 | 22 / 1 |
| val | 50 / 111 / 63 | 51 / 173 / 0 | 8 / 5 |

Class weighting was already enabled historically. The old video-balanced sampler changed effective onset prevalence from 54.45% to 45.59% while weights were calculated from raw window counts: the historical onset positive weight was 0.837, whereas the corresponding sampler-weighted ratio was 1.194. The new comparison uses uniformly shuffled windows without replacement, consistent fitting-only class weights, and the same fixed architectures/augmentation. Checkpoint selection now directly measures development ranking instead of averaging two BCE losses. Differences against history combine these training-policy changes; the paired weighted/unweighted arms isolate weighting within the new policy.

**Original-cohort validation comparison**

All rows below use identical original validation windows: 161 supervised onset windows (50 positives) and 224 active-fight windows (51 positives). F1 thresholds are chosen only on internal development. `historical_guarded` is the previous restrictive policy; `development_f1` is the exact development maximum-F1 policy. Higher recall alone can mean predicting almost everything positive. Predicting every window positive already gives onset F1=0.474 and active-fight F1=0.371 on these cohorts; improvements over the previous restrictive threshold must not be confused with useful discrimination.

| Model | Head | Threshold policy | AUROC | Precision | Recall | F1 | AP | PR-AUC (trapezoid) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| baseline_historical | onset | development_f1 | 0.608 | 0.303 | 0.860 | 0.448 | 0.446 | 0.438 |
| baseline_historical | onset | historical_guarded | 0.608 | 0.571 | 0.160 | 0.250 | 0.446 | 0.438 |
| baseline_historical | active | development_f1 | 0.534 | 0.232 | 0.706 | 0.350 | 0.373 | 0.366 |
| baseline_historical | active | historical_guarded | 0.534 | 0.423 | 0.216 | 0.286 | 0.373 | 0.366 |
| advanced_historical | onset | development_f1 | 0.232 | 0.297 | 0.940 | 0.452 | 0.215 | 0.209 |
| advanced_historical | onset | historical_guarded | 0.232 | 0.000 | 0.000 | 0.000 | 0.215 | 0.209 |
| advanced_historical | active | development_f1 | 0.666 | 0.229 | 1.000 | 0.372 | 0.445 | 0.438 |
| advanced_historical | active | historical_guarded | 0.666 | 0.750 | 0.059 | 0.109 | 0.445 | 0.438 |
| advanced_class_weighted | onset | development_f1 | 0.292 | 0.229 | 0.540 | 0.321 | 0.233 | 0.225 |
| advanced_class_weighted | active | development_f1 | 0.507 | 0.229 | 1.000 | 0.372 | 0.369 | 0.363 |
| advanced_unweighted | onset | development_f1 | 0.377 | 0.295 | 0.760 | 0.425 | 0.275 | 0.268 |
| advanced_unweighted | active | development_f1 | 0.555 | 0.230 | 0.980 | 0.373 | 0.290 | 0.277 |
| baseline_class_weighted | onset | development_f1 | 0.585 | 0.306 | 0.980 | 0.467 | 0.402 | 0.393 |
| baseline_class_weighted | active | development_f1 | 0.341 | 0.228 | 1.000 | 0.371 | 0.169 | 0.165 |
| baseline_unweighted | onset | development_f1 | 0.626 | 0.311 | 1.000 | 0.474 | 0.514 | 0.509 |
| baseline_unweighted | active | development_f1 | 0.361 | 0.206 | 0.686 | 0.317 | 0.173 | 0.169 |

**Current-pose preprocessing mismatch and separate matched cohort**

The historical window gate could accept stale history when current observations were absent. The existing runtime-v2 rule requires six currently observed joints in at least one selected person. Recomputing every skeleton from its raw cache reproduced all 1,974 windows exactly. Applying that fixed pose-only rule excludes 261 fit, 92 development and 20 validation windows. No targets are relabeled. The separate guarded archive retains 1,088 / 309 / 204 windows; its validation onset head has 152 supervised windows with 46 positives, and active head has 204 with 42 positives. The four original onset-positive and nine active-positive windows excluded at validation are abstentions, not negatives.

The two class-weighted quality-aligned models are compared with historical models on this same retained subset. These cohort results cannot be directly substituted for full-cohort performance.

| Model | Head | AUROC | Precision | Recall | F1 | AP | PR-AUC (trapezoid) |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_historical | onset | 0.591 | 0.291 | 0.848 | 0.433 | 0.436 | 0.428 |
| baseline_historical | active | 0.479 | 0.197 | 0.643 | 0.302 | 0.311 | 0.304 |
| advanced_historical | onset | 0.241 | 0.289 | 0.935 | 0.441 | 0.218 | 0.210 |
| advanced_historical | active | 0.644 | 0.207 | 1.000 | 0.343 | 0.381 | 0.365 |
| advanced_class_weighted | onset | 0.353 | 0.196 | 0.435 | 0.270 | 0.260 | 0.249 |
| advanced_class_weighted | active | 0.475 | 0.208 | 1.000 | 0.344 | 0.221 | 0.210 |
| baseline_class_weighted | onset | 0.687 | 0.411 | 0.804 | 0.544 | 0.438 | 0.428 |
| baseline_class_weighted | active | 0.565 | 0.209 | 0.929 | 0.341 | 0.338 | 0.328 |

**Learning and uncertainty**

The historical advanced onset fit/development AUROC is 0.625/0.601, compared with 0.232 externally. Its within-video validation onset AUROC is only 0.156, so the failure is not explained solely by added normal-video controls. Its selected checkpoint does not show near-perfect fitting discrimination; weak learning and source/timing shift are supported more clearly than a claim of simple training-set memorization. Later development BCE deteriorated while training loss fell, but those original loss numbers also differ in weighting and augmentation. New curves measure fitting and development in evaluation mode with the same unweighted objectives.

A fitting-only tiny-batch diagnostic can drive both model losses close to zero with finite gradients. That confirms an optimization path, not useful generalization. Per-epoch per-head AUROC/AP/BCE, gradient norms, source counts, score quantiles, confusion matrices and all threshold policies are retained. The paired source bootstrap resamples the 13 validation source groups together, never overlapping windows independently. Confidence intervals in `paired_improvements.json` are exploratory; one seed and selected weak-label contexts do not establish reliable model superiority.

AP is the non-interpolated precision-recall summary. It differs from trapezoidal PR-AUC; both are reported, following the [scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Positive logits follow [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html).

**Artifacts and reproduction**

`outputs/diagnosis_v1/comparison_metrics.csv` contains every confusion matrix and policy. `paired_improvements.json` contains actual matched deltas and source-cluster intervals. Plots are in `outputs/diagnosis_v1/plots/`. New source modules are `diagnose_stgcn.py`, `stgcn_loss_ablation.py`, `stgcn_quality_windows.py` and `build_stgcn_diagnosis_report.py`; tests are in `tests/test_stgcn_diagnostics.py`. Existing models, labels, split identities, data, v1-v4 detectors and previous results are retained. All weights stay local/ignored.

Run from Component 03, using new version names so existing artifacts remain intact:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
& $C3Python -m src.diagnose_stgcn --out-dir outputs/diagnosis_reproduction_v1
& $C3Python -m src.stgcn_loss_ablation train-all --out-dir outputs/diagnosis_reproduction_v1/ablation --weights-dir weights/diagnosis_reproduction_v1
& $C3Python -m src.stgcn_loss_ablation evaluate-all --out-dir outputs/diagnosis_reproduction_v1/ablation --weights-dir weights/diagnosis_reproduction_v1
& $C3Python -m src.stgcn_quality_windows --out-dir data/pose_cache/diagnosis_guarded_reproduction_v1/windows
& $C3Python -m src.stgcn_loss_ablation train-all --data data/pose_cache/diagnosis_guarded_reproduction_v1/windows --out-dir outputs/diagnosis_reproduction_v1/guarded_comparison --weights-dir weights/diagnosis_guarded_reproduction_v1 --losses class_weighted
& $C3Python -m src.stgcn_loss_ablation evaluate-all --data data/pose_cache/diagnosis_guarded_reproduction_v1/windows --out-dir outputs/diagnosis_reproduction_v1/guarded_comparison --weights-dir weights/diagnosis_guarded_reproduction_v1 --losses class_weighted
```

The full suite passed **175 tests plus 17 subtests**, with no failures or skips; see `outputs/diagnosis_v1/tests_complete.xml`. Tests cover positive-logit gradient direction, masked supervision, fit-only class weights, tied AP/PR areas, exact development thresholds, quality filtering without label changes, paired source resampling and locked-test rejection. The earlier targeted runs passed 20 and 21 tests; the final suite includes the additional paired-bootstrap regression. No dependencies were installed for this investigation.

The scientific blocker remains independently adjudicated precursor/contact supervision and broader, prospectively frozen source-group evaluation. More favorable weak-label window metrics do not establish confirmed early-violence accuracy. The runtime default and alert eligibility remain unchanged.

**Final assessment after all six fits**

On the original, unchanged validation cohort, the development-selected unweighted
baseline improves onset AUROC by 0.0178 and AP by 0.0675. Its F1-maximizing
development threshold predicts every validation onset window positive, so its
recall of 1.0 and F1 of 0.474 are not an operational improvement. The corresponding
advanced run improves onset AUROC by 0.1447 and AP by 0.0600, but AUROC remains
0.377 and its F1 decreases from 0.452 to 0.425 when both checkpoints use the same
development maximum-F1 policy. The original reported advanced F1 of zero was at
the restrictive research guard, not its maximum-F1 point.

Class weighting does not consistently help: under the same new training policy,
onset AUROC falls from 0.626 to 0.585 for baseline and from 0.377 to 0.292 for
advanced. Active-fight ranking also regresses in the unweighted original-cohort
runs: baseline AUROC 0.534 to 0.361 and advanced 0.666 to 0.555. All regressions
remain visible rather than selecting a favorable head after viewing validation.

On the separate **matched current-observation cohort**, the quality-aligned
baseline improves onset AUROC from 0.5913 to 0.6874, precision from 0.2910 to
0.4111 and F1 from 0.4333 to 0.5441. Recall decreases from 0.8478 to 0.8043;
AP changes only from 0.4364 to 0.4385, and trapezoidal PR-AUC from 0.4283 to
0.4280. Its confusion matrix is TN=53, FP=53, FN=9, TP=37, so the false-positive
rate is still **50%**. The paired source-bootstrap 95% interval for its AUROC
change is **[-0.230, +0.453]**, and for F1 change **[-0.107, +0.347]**. These
wide intervals do not establish a robust improvement. The excluded four onset
positives remain documented abstentions; even counting all 50 original positives,
only 37 would be detected at this point (74% window recall, not event recall).

The investigation therefore supports a real eligibility mismatch, training/
development source imbalance and weak out-of-source timing generalization.
It does not support complementing the scores, inventing precursor supervision,
or promoting a newly fitted model to operational use. Preserve the test holdout,
broaden representative train/development source coverage, adjudicate cue/contact
times and freeze the next evaluation protocol before making further model-quality
claims. This stage checked all 531 pre-existing protected files without changes.

The final test invocation was:

```powershell
$C3Root = (Get-Location).Path
$env:PYTHONPATH = "$C3Root\data\phase1_work\testdeps;$C3Root;$C3Root\legacy"
$env:TEMP = (Resolve-Path 'data\phase1_work\tmp').Path
$env:TMP = $env:TEMP
& $C3Python -m pytest tests tests_optional legacy/test_c3_aggression.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_diagnosis_final --junitxml outputs/diagnosis_v1/tests_complete.xml
```

Use a fresh basename/output path for another run. The final report builder can
be rerun after the reproduction fits above with:

```powershell
& $C3Python -m src.build_stgcn_diagnosis_report --out-dir outputs/diagnosis_reproduction_v1 --quality-data data/pose_cache/diagnosis_guarded_reproduction_v1/windows --document STGCN_DIAGNOSIS_REPRODUCTION_V1.md
```

Manual final assessment and verification notes follow the generated numerical
tables; all source metrics and bootstrap intervals remain in the versioned JSONs.
