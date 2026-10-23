# Component 03 test and model improvement summary

Completed locally on 21 October 2026. These are exploratory weak-label results, not independently verified early-aggression accuracy.

## Tests

- Initial suite: 225 tests and 62 subtests passed.
- Final C3/backend/optional suite: 237 tests + 62 subtests passed; 0 failures/errors.
- Frontend: 13 tests passed. C3 production build passed.

## Work completed

- Locked dataset/split guards, causal YOLO-Pose/ByteTrack caches and trailing pose windows.
- Baseline and multi-stream ST-GCN, separate weak-onset and active-fight heads, quality abstention and provenance.
- Original shared dashboard: one Recorded video upload, queued C3 inference, annotated playback, scores, review and export.
- C3 research tab and obsolete dashboard copies removed. C4-only backend modules archived separately.
- Team-guide component entry point: run.py. Research logbook and daily diary PDFs prepared.

## Improvement work in this run

- Implemented 240-dimensional masked motion/geometry features with linear and MLP learned heads.
- Tested person-order/translation/reflection/scale invariance, masked and padded values, causality and source leakage guards.
- Initial four-candidate experiment was rejected: active ranking improved modestly, but guarded F1 and onset results regressed.
- Added 32 locked-training contexts: 24 weak no-fight and 8 weak fight; 210 additional quality-matched windows.
- All original rows, labels and fit/dev/val roles were preserved. The 204 quality-matched validation windows are byte-identical.
- Trained four additional kinematic candidates and two ST-GCN candidates on expanded data.
- Selected model family members and per-head positive ensemble weights on internal development only; thresholds frozen before external validation.

## Same original validation cohort

| Head / model | N | Accuracy | Balanced accuracy | AUROC | F1 | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| weak_first_onset_within_3s / baseline | 161 | 0.702 | 0.553 | 0.608 | 0.250 | 0.160 | 0.054 |
| weak_first_onset_within_3s / advanced | 161 | 0.671 | 0.486 | 0.232 | 0.000 | 0.000 | 0.027 |
| weak_first_onset_within_3s / candidate | 161 | 0.677 | 0.546 | 0.665 | 0.278 | 0.200 | 0.108 |
| weak_active_fight / baseline | 224 | 0.754 | 0.564 | 0.534 | 0.286 | 0.216 | 0.087 |
| weak_active_fight / advanced | 224 | 0.781 | 0.527 | 0.666 | 0.109 | 0.059 | 0.006 |
| weak_active_fight / candidate | 224 | 0.777 | 0.524 | 0.672 | 0.107 | 0.059 | 0.012 |

Thresholds use train-only development. F1 above uses guarded reference thresholds; old baseline active-fight notifications were disabled, so it is a score-level comparison, not measured deployed alert accuracy.

Predeclared descriptive acceptance passed: **False**.

The runtime default remains the existing guarded baseline. The candidate improves ranking point estimates but regresses guarded active-fight F1, so it was not promoted. No default checkpoint, threshold or operational notification policy was changed.

Candidate checkpoint: `C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection\weights\model_improvement_expanded_v1\headwise_guarded.pt`
Candidate SHA-256: `553510326d5c171273c6aa21dcb0c8a34b2cb87eeb737631364b457bac9e3bba`

Per-head weights [baseline, advanced, kinematic]: `[[0.75, 0.0, 0.25], [0.25, 0.5, 0.25]]`.

## Practical limits

- Zero independently verified precursor-positive videos. Confirmed 1-3 second pre-contact prediction remains unproven.
- Validation has only 13 contributing source videos and was inspected in earlier experiments. Windows overlap; no fresh final-test accuracy claim is made.
- comparison.csv includes both original and quality-matched cohorts. paired_bootstrap.json reports source-cluster uncertainty; a small point improvement is not necessarily statistically established.
- Raw accuracy must be compared with the always-negative baseline because weak labels are imbalanced.
- No test source was used for fitting, source selection, normalization, epoch/model selection or threshold selection. Original weights and source data are preserved.
- Operational notifications remain disabled. Scores remain uncalibrated.

## References

- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): class weighting controls precision/recall trade-offs; it does not prove accuracy.
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): source groups must not overlap between fitting and evaluation.

## Recorded-video smoke check

- Existing F_9 review clip: 35/40 usable decisions; compatible pose cache reused.
- Candidate returned 0 incident candidates. The reported violence-detection miss is still unresolved.
- Annotated MP4, scores, incident JSON and timing/provenance were produced successfully.
- This single clip was used only after freezing selection/evaluation, as a runtime smoke test; it was not used for fitting or threshold tuning.
