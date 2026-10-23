# Component 03 — Model optimization results

Measured locally on 21 October 2026. All numerical results use existing weak UBI labels. They do not establish independently verified pre-contact aggression prediction.

## Decision

The new frozen ST-GCN readout improves active-fight ranking point estimates on the unchanged quality-matched cohort: AUROC **0.719**, versus **0.648** for the previous headwise candidate and **0.479** for the runtime baseline. The predeclared constrained-recall/F1 promotion criteria **failed**. Keep the original baseline and its thresholds as the default. No operational alerts or new eligibility flags were enabled.

At source-held-out thresholds selected to limit tuning-source FPR to 10%, new active-fight recall is 0.167, with 17 false-positive windows. The rethresholded baseline has the same recall and 2 false positives. Better ranking has not solved the required precision/recall operating point.

## What was inspected and preserved

Read MODEL_IMPROVEMENT.md, MODEL_RELIABILITY.md, STGCN_DIAGNOSIS.md, WEAK_FIGHT_PILOT.md and FINAL_C3.md completely, plus actual model, training, data, loss, threshold, inference and event-adapter code. Reviewed the previous eight kinematic fits and two expanded ST-GCN fits before freezing this protocol.

Previous findings: the expanded linear-window kinematic model was stronger on internal-dev onset; the MLP-window model had modest active ranking but weaker onset. Full ST-GCN refits showed unstable development curves and late overfitting. The existing expanded advanced checkpoint had the strongest active internal-dev AUROC among prior advanced checkpoints (0.557), so it was reused as the frozen encoder. No expensive pose extraction was repeated.

Preservation audit: **2149 pre-existing files are byte-identical**; the one already-running dashboard request log appended independently, with its entire original byte prefix hash-verified intact. The initial strict audit recorded that live-log change; preservation_final.json resolves it as append-only. All old weights, result data and tracked files outside C3 are unchanged. 163 indexed pose caches passed hashes, source identity, schema and array validation. Components 01, 02, 04 and shared dashboard sources are unchanged. No dataset/annotation/split/default/checkpoint overwrite, commit or push was performed.

## Current working system

Component 03 currently includes locked source splits and provenance, causal YOLO11-Pose/ByteTrack caches, quality-aware trailing windows, baseline and joint/bone/velocity ST-GCN models, separate weak-onset and active-fight scoring, versioned experiments, incident/evidence export and the original shared dashboard integration. Recorded-video uploads reuse compatible caches or queue extraction, produce annotated playback and expose missing-pose/disabled-head diagnostics. The obsolete C3 research-video tab was already removed. This optimization adds controlled readout experiments and diagnostics to that working pipeline.

## Protocol and data

- Locked source split remains 791 train / 141 validation / 68 test identities. No test source was used for fitting, normalization, epoch/model selection, thresholds or diagnostics.
- Expanded fitting: 1,240 windows / 115 sources; internal development: 367 / 31. Active fitting labels: 376 positive, 864 negative (positive weight 2.298). Dev prevalence 91/367 = 24.80%. Uniform shuffled windows without replacement; no resampling compounded with loss weighting.
- Original validation: active 224 windows / 51 positive; onset 161 known / 50 positive. Quality-matched validation: active 204 / 42; onset 152 known / 46. Both cohorts contain 13 contributing sources. All compared models receive identical rows, labels and masks within each cohort.
- Source and leakage groups remain disjoint across fit/dev/val. Independent scalar label reconstruction found zero target/mask mismatches and zero windows containing a future timestamp in both materialized archives.
- Original candidate coverage: 224/320 decisions (70.0%). Current-observation guard retains 204/320 (63.75%), or 204/224 (91.07%) of previously eligible rows. The 20 exclusions include 9 active positives and 4 onset positives. Exclusions are abstentions; they are not relabeled negatives. Quality-matched conditional metrics are reported separately from original-cohort results.
- F9 is locked **validation**, not test. It was excluded from every new threshold fitting set and all weight/configuration selection. Each scored validation source is also excluded from its own threshold tuning fold. Other validation sources had been inspected historically, so this remains exploratory source-crossfit analysis, not a fresh independent benchmark.
- Full-cohort metrics at a threshold selected on the other 12 sources are reported as selection statistics. Source-held-out results use different thresholds per source and are the primary comparison. Window-level FPR is not event false alarms/hour.

## Root causes supported by measurements

1. **Threshold and eligibility are distinct causes.** Previous headwise active ranking improved, but its guarded reference threshold 0.590869 gave only 3 TP / 48 FN on the original cohort. The active head was independently disabled: its effective runtime threshold was 1.000001. Lowering a reference threshold alone would not enable candidates. Research-score mode separately suppresses notifications, not candidate scoring.
2. **Score overlap remains substantial.** F1 optimization trades missed positives for many negative-window crossings. On the quality-matched cohort, the new F1 crossfit operating point has 26 TP / 16 FN / 49 FP / 113 TN; FPR 30.25%. A constraint fitted on other sources does not guarantee its held-out FPR. Even the minimum-precision-50% tuning policy produces 0 TP / 11 FP for the new model after source holdout. This is source instability, not a useful high-precision policy.
3. **Imbalance and distribution shift affect scores and calibration.** Fitting active prevalence is 30.32%, versus 24.80% internal dev and 20.59% quality-matched validation. Weighted BCE improves dev AP slightly but moves scores upward; it does not make scores probabilities. Uniform sampling avoids the historical mismatch between source-balanced sampling and raw-window class weights.
4. **Overfitting and limited source diversity.** Frozen-readout fit BCE falls while dev BCE stays around 0.71–0.75; earlier full-ST-GCN development curves were also unstable. Early stopping and source-separated selection limit the damage. Thousands of overlapping windows are not thousands of independent examples.
5. **Pose coverage and tracking affect usable history.** The current observation guard excludes 20 originally eligible validation windows. F9 has 35/40 usable decisions and 9–10 observed-identity resets per eight-second replay. Resets intentionally discard confirmation history. Anonymous ID churn suggests an input consistency issue; it is not independently measured tracker error without tracking ground truth.
6. **Weak global labels limit learning.** Labels describe a source time, while inputs select up to two tracks by past box area; these are not verified identities of the fighting people. Short bursts, boundaries and occlusion can mismatch the selected pair. This is a plausible limitation rather than a newly verified label error; deterministic encoding/alignment tests found no implementation inversion or timestamp leak.
7. **Architecture constraints.** Existing joint/bone/backward-velocity encoders use symmetric two-person pooling and eight interaction summaries. Joint masks merge observed and short causal imputations; there is no separately learned observed-vs-imputed input channel. A new readout cannot recover motion missing from the cache. No evidence here justifies replacing the complete backbone or claiming an acceleration/identity feature gain.

## Controlled experiments

Seven small readout configurations, each with seeds 73, 101 and 137 (21 small fits, zero new full-backbone retrains). Frozen 224-dimensional advanced ST-GCN representation; fitting-only mean/SD normalization; linear active head; AdamW lr=0.003, weight decay=0.01, batch 128, max 80 epochs, patience 10. Select epoch by internal-dev active average precision, tie by unweighted BCE. Select configuration by mean seed dev AP, tie lower BCE. Average three seed probabilities for inference. The onset output is exactly the historical baseline output.

BCE, positive-weighted BCE and focal gamma=2 were tested independently. Focal used no alpha balancing or class-weight/resampling combination. Feature ablations refit a readout on retained frozen feature blocks; they are not end-to-end stream-removal retrains. The 24-frame condition reads only the latest 24 observations with the same cutoff/targets; it does not relabel or change the validation cohort. Acceleration and new tracking-mask channels were reviewed but not added without supporting evidence.

| Configuration | Mean seed dev AP ± SD | Val AUROC | Val AP | Val trap PR-AUC | F1 crossfit F1 | FPR10 crossfit P / R / F1 |
|---|---:|---:|---:|---:|---:|---:|
| full_bce_48 | 0.2732 ± 0.0029 | 0.701 | 0.395 | 0.379 | 0.400 | 0.296 / 0.190 / 0.232 |
| full_weighted_48 | 0.2800 ± 0.0040 | 0.719 | 0.406 | 0.390 | 0.444 | 0.292 / 0.167 / 0.212 |
| full_focal_48 | 0.2778 ± 0.0065 | 0.708 | 0.403 | 0.387 | 0.369 | 0.348 / 0.190 / 0.246 |
| full_bce_24 | 0.2723 ± 0.0016 | 0.632 | 0.296 | 0.282 | 0.349 | 0.300 / 0.143 / 0.194 |
| joint_bce_48 | 0.2710 ± 0.0184 | 0.500 | 0.281 | 0.268 | 0.188 | 0.250 / 0.048 / 0.080 |
| joint_bone_bce_48 | 0.2585 ± 0.0012 | 0.542 | 0.273 | 0.260 | 0.348 | 0.000 / 0.000 / 0.000 |
| no_geometry_bce_48 | 0.2549 ± 0.0027 | 0.583 | 0.292 | 0.279 | 0.274 | 0.368 / 0.167 / 0.230 |

Selected on internal development: **full_weighted_48**, epochs [15, 18, 18]. Full weighted dev AP 0.2800 versus unweighted 0.2732; this small difference is not proof of a loss-family population advantage. Joint-only seed AP SD is larger than the full-feature conditions. Bone/velocity/geometry benefits vary across this frozen representation; do not infer universal causal feature importance from seven exploratory arms.

## Same-cohort model ranking and original reference thresholds

| Cohort | Model / head | N / positive | AUROC | AP | Trap PR-AUC | Reference threshold | P / R / F1 | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | advanced / active | 224 / 51 | 0.666 | 0.445 | 0.438 | 0.601315 | 0.750 / 0.059 / 0.109 | 172 / 1 / 48 / 3 |
| original | advanced / onset | 161 / 50 | 0.232 | 0.215 | 0.209 | 0.551531 | 0.000 / 0.000 / 0.000 | 108 / 3 / 50 / 0 |
| original | baseline / active | 224 / 51 | 0.534 | 0.373 | 0.366 | 0.611068 | 0.423 / 0.216 / 0.286 | 158 / 15 / 40 / 11 |
| original | baseline / onset | 161 / 50 | 0.608 | 0.446 | 0.438 | 0.603839 | 0.571 / 0.160 / 0.250 | 105 / 6 / 42 / 8 |
| original | headwise / active | 224 / 51 | 0.672 | 0.389 | 0.382 | 0.590869 | 0.600 / 0.059 / 0.107 | 171 / 2 / 48 / 3 |
| original | headwise / onset | 161 / 50 | 0.665 | 0.408 | 0.399 | 0.691749 | 0.455 / 0.200 / 0.278 | 99 / 12 / 40 / 10 |
| original | readout / active | 224 / 51 | 0.696 | 0.414 | 0.401 | disabled (1.000001) | 0.000 / 0.000 / 0.000 | 173 / 0 / 51 / 0 |
| original | readout / onset | 161 / 50 | 0.608 | 0.446 | 0.438 | 0.603839 | 0.571 / 0.160 / 0.250 | 105 / 6 / 42 / 8 |
| quality_matched | advanced / active | 204 / 42 | 0.644 | 0.381 | 0.365 | 0.601315 | 0.000 / 0.000 / 0.000 | 162 / 0 / 42 / 0 |
| quality_matched | advanced / onset | 152 / 46 | 0.241 | 0.218 | 0.210 | 0.551531 | 0.000 / 0.000 / 0.000 | 103 / 3 / 46 / 0 |
| quality_matched | baseline / active | 204 / 42 | 0.479 | 0.311 | 0.304 | 0.611068 | 0.318 / 0.167 / 0.219 | 147 / 15 / 35 / 7 |
| quality_matched | baseline / onset | 152 / 46 | 0.591 | 0.436 | 0.428 | 0.603839 | 0.571 / 0.174 / 0.267 | 100 / 6 / 38 / 8 |
| quality_matched | headwise / active | 204 / 42 | 0.648 | 0.343 | 0.337 | 0.590869 | 1.000 / 0.048 / 0.091 | 162 / 0 / 40 / 2 |
| quality_matched | headwise / onset | 152 / 46 | 0.673 | 0.414 | 0.403 | 0.691749 | 0.500 / 0.217 / 0.303 | 96 / 10 / 36 / 10 |
| quality_matched | readout / active | 204 / 42 | 0.719 | 0.406 | 0.390 | disabled (1.000001) | 0.000 / 0.000 / 0.000 | 162 / 0 / 42 / 0 |
| quality_matched | readout / onset | 152 / 46 | 0.591 | 0.436 | 0.428 | 0.603839 | 0.571 / 0.174 / 0.267 | 100 / 6 / 38 / 8 |

Reference-threshold entries are score-level comparisons, not deployed event accuracy. The new dev-selected checkpoint initially disables active thresholding entirely; its separate versioned validation-reference checkpoint below records the exploratory validation threshold while retaining disabled eligibility.

## Validation-only operating-point comparison

Policies: maximize F1; maximize F1 with tuning FPR ≤ 10%; maximize F1 with tuning precision ≥ 50%. Exact unique tied-score cutoffs are used, with a no-positive fallback if precision is infeasible. No test labels enter this calculation.

| Model / head | Policy | Selected threshold excluding F9 | Source-crossfit P | R | F1 | FPR | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline / onset | f1 | 0.424076 | 0.393 | 0.522 | 0.449 | 0.349 | 69 / 37 / 22 / 24 |
| baseline / onset | fpr10 | 0.567236 | 0.611 | 0.239 | 0.344 | 0.066 | 99 / 7 / 35 / 11 |
| baseline / onset | precision50 | 0.567236 | 0.306 | 0.239 | 0.268 | 0.236 | 81 / 25 / 35 / 11 |
| baseline / active | f1 | 0.034656 | 0.091 | 0.262 | 0.135 | 0.679 | 52 / 110 / 31 / 11 |
| baseline / active | fpr10 | 0.644028 | 0.778 | 0.167 | 0.275 | 0.012 | 160 / 2 / 35 / 7 |
| baseline / active | precision50 | 0.644028 | 0.778 | 0.167 | 0.275 | 0.012 | 160 / 2 / 35 / 7 |
| advanced / onset | f1 | 0.300583 | 0.273 | 0.848 | 0.413 | 0.981 | 2 / 104 / 7 / 39 |
| advanced / onset | fpr10 | 0.448031 | 0.143 | 0.022 | 0.038 | 0.057 | 100 / 6 / 45 / 1 |
| advanced / onset | precision50 | 0.568198 | 0.000 | 0.000 | 0.000 | 0.047 | 101 / 5 / 46 / 0 |
| advanced / active | f1 | 0.495997 | 0.297 | 0.452 | 0.358 | 0.278 | 117 / 45 / 23 / 19 |
| advanced / active | fpr10 | 0.527451 | 0.478 | 0.262 | 0.338 | 0.074 | 150 / 12 / 31 / 11 |
| advanced / active | precision50 | 0.530445 | 0.333 | 0.119 | 0.175 | 0.062 | 152 / 10 / 37 / 5 |
| headwise / onset | f1 | 0.555317 | 0.367 | 0.478 | 0.415 | 0.358 | 68 / 38 / 24 / 22 |
| headwise / onset | fpr10 | 0.684956 | 0.478 | 0.239 | 0.319 | 0.113 | 94 / 12 / 35 / 11 |
| headwise / onset | precision50 | 0.570864 | 0.045 | 0.022 | 0.029 | 0.198 | 85 / 21 / 45 / 1 |
| headwise / active | f1 | 0.493288 | 0.300 | 0.643 | 0.409 | 0.389 | 99 / 63 / 15 / 27 |
| headwise / active | fpr10 | 0.569196 | 0.240 | 0.143 | 0.179 | 0.117 | 143 / 19 / 36 / 6 |
| headwise / active | precision50 | 0.589588 | 0.375 | 0.071 | 0.120 | 0.031 | 157 / 5 / 39 / 3 |
| readout / onset | f1 | 0.424076 | 0.393 | 0.522 | 0.449 | 0.349 | 69 / 37 / 22 / 24 |
| readout / onset | fpr10 | 0.567236 | 0.611 | 0.239 | 0.344 | 0.066 | 99 / 7 / 35 / 11 |
| readout / onset | precision50 | 0.567236 | 0.306 | 0.239 | 0.268 | 0.236 | 81 / 25 / 35 / 11 |
| readout / active | f1 | 0.463545 | 0.347 | 0.619 | 0.444 | 0.302 | 113 / 49 / 16 / 26 |
| readout / active | fpr10 | 0.669162 | 0.292 | 0.167 | 0.212 | 0.105 | 145 / 17 / 35 / 7 |
| readout / active | precision50 | 0.753929 | 0.000 | 0.000 | 0.000 | 0.068 | 151 / 11 / 42 / 0 |

The single threshold column is fitted on all non-F9 validation sources for future reference. The adjacent crossfit metrics use a separately fitted threshold for every held-out source; they must not be interpreted as evaluation of that single fixed threshold. Full-cohort fixed-threshold metrics and every fold threshold are in evaluation.json. This prevents threshold selection on the source being scored, but does not erase earlier validation exposure or establish deployment generalization.

### Calibration and score direction

| Model | Active Brier | Active NLL | Active ECE | Positive score median | Negative score median |
|---|---:|---:|---:|---:|---:|
| baseline | 0.2579 | 0.7484 | 0.3033 | 0.4603 | 0.4923 |
| advanced | 0.2284 | 0.6495 | 0.2687 | 0.4984 | 0.4743 |
| headwise | 0.2304 | 0.6533 | 0.2791 | 0.5193 | 0.4782 |
| readout | 0.1990 | 0.5776 | 0.2197 | 0.5341 | 0.4197 |

Positive temperature fitted on internal dev only: T=3.7551; validation Brier 0.1990 → 0.2251, NLL 0.5776 → 0.6421. This is a diagnostic, not operational calibration, and was not applied to checkpoints. Sigmoid(logit) always denotes y=1; no score complement or validation-fitted sign inversion was used. Pairwise and rank-integrated AUROC agree. ECE uses 10 fixed bins; sparse-bin estimates are uncertain.

### Source uncertainty and errors

- original: readout AUROC 95% source-cluster interval [0.5380996405813903, 0.827868164929015]; paired AUROC delta vs baseline [-0.11434672984672982, 0.420229053912725], vs headwise [-0.1326568798175519, 0.1633776965431681].
- quality_matched: readout AUROC 95% source-cluster interval [0.5389668855571762, 0.8563390380966281]; paired AUROC delta vs baseline [-0.01114484540057061, 0.4840015016901862], vs headwise [-0.0729692044482465, 0.19566026695344135].

For quality-matched FPR10 crossfit active F1, readout-minus-baseline 95% interval is [-0.582335176085176, 0.3877627170271093]. All 2,000 draws resample whole sources. Threshold-crossfit bootstrap intervals condition on the fitted folds and do not repeat threshold/model selection, so are exploratory rather than full nested-procedure uncertainty.

At the quality-matched FPR10 source-crossfit point, all seven new true-positive windows come from just two of the eight positive sources: f_72 (6) and f_79 (1). Six positive sources have zero true-positive windows. The 17 false positives are concentrated in f_72 (8), f_79 (2), n_360 (6) and n_234 (1). This source concentration explains why global ranking gains do not ensure reliable event recall. These counts are relative to weak annotations, not independently verified incidents.

per_source_errors.csv lists positive/negative counts, TP/FP/FN/TN and operating policies per source. validation_predictions.csv contains every timestamp, score, weak label and held-out threshold, allowing false-positive and missed-window review.

## F9 complete inference diagnostic

Original F_9_1_2_0_0.mp4 is 96 seconds, locked validation, first weak onset 31.166667 seconds. Its cached context is 26.2–34.2 seconds. The uploaded review is a separately encoded 8-second clip on a 0–8 second local clock, not another validation source.

Three-anchor image matching estimates parent offset 26.166667s (original frame 785; MAE 0.726/255 versus next-best 3.651). This is image-derived alignment, not an authoritative edit record, and derived rows were not assigned new ground-truth labels or used for selection. Original/derived sample phases differ by one source frame; tracker initialization and input phases can change selected identities and scores. The observed score divergence is documented rather than assumed numerically equivalent.

| Context / model | Valid / decisions | Raw active peak | Reference active threshold | Effective active threshold | Identity resets | Actual candidates | Counterfactual tuned candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| derived / advanced | 35 / 40 | 0.522490 | 0.601315 | 1.000001 | 10 | 0 | 0 |
| derived / baseline | 35 / 40 | 0.550248 | 0.611068 | 1.000001 | 10 | 0 | 0 |
| derived / headwise | 35 / 40 | 0.563864 | 0.590869 | 1.000001 | 10 | 0 | 0 |
| derived / readout | 35 / 40 | 0.764853 | 0.669162 | 1.000001 | 10 | 0 | 0 |
| original / advanced | 35 / 40 | 0.532252 | 0.601315 | 1.000001 | 9 | 0 | 0 |
| original / baseline | 35 / 40 | 0.575874 | 0.611068 | 1.000001 | 9 | 0 | 0 |
| original / headwise | 35 / 40 | 0.565136 | 0.590869 | 1.000001 | 9 | 0 | 0 |
| original / readout | 35 / 40 | 0.595615 | 0.669162 | 1.000001 | 9 | 0 | 0 |

Pipeline findings:

- Video/source hashes match their caches. YOLO11n-Pose/ByteTrack outputs and strictly past windows were reused; no pose extraction was repeated. Cached source frames/timestamps remain monotonic.
- Each replay has 3 decisions with insufficient history, 2 with too few currently observed joints, and zero failures of the 0.45 historical quality threshold. These five decisions abstain. Selected pair IDs and detected/currently observed people counts are included per timestamp.
- The baseline and previous headwise active peaks remain below their respective reference thresholds. Their active eligibility is also disabled. Candidate confirmation and cooldown therefore cannot recover those misses.
- New derived-clip readout raw crossings: 2; smoothed crossings: 2; maximum positive hits retained in confirmation history: 2. At clip time 7.2s an identity reset leaves one history decision; 7.4s has two decisions/two hits; 7.6s abstains and clears history. Confirmation needs three history decisions and two positive smoothed hits. These crossings map approximately to 33.37–33.57s under image-derived alignment, after the first weak 31.166667–32.0s burst, so they are not evidence of a correctly timed fight detection. The actual new checkpoint also disables both heads. No F9-specific threshold or exception was introduced.
- All actual and counterfactual confirmed candidate counts remain zero. Cooldown suppression is not the primary cause here. No incident/notification was emitted. The F9 missed detection remains unresolved; a higher peak score alone is not a fix.

f9/*_stages.csv trace raw and smoothed scores, history/current-observation/quality gates, selected/current IDs, source frame, reference/effective thresholds, identity resets, notification suppression and counterfactual status at each decision. f9/confirmation_stages.csv exposes confirmation-history length and hit counts.

## Weak onset lead-time diagnostic

The new onset head is exactly the baseline, so onset rankings and crossings are unchanged. These are threshold-crossing windows before the first weak fight annotation; they are not confirmed candidate events or independently verified pre-contact warnings.

| Policy | Weak positive sources | Sources with TP crossings | TP windows | Lead min / median / max (s) |
|---|---:|---:|---:|---|
| f1 | 8 | 5 | 24 | [0.16666700000000034, 1.3666670000000032, 2.599999666666667] |
| fpr10 | 8 | 3 | 11 | [0.1666670000000181, 0.9666669999999726, 2.5666669999999954] |
| precision50 | 8 | 3 | 11 | [0.1666670000000181, 0.9666669999999726, 2.5666669999999954] |

Verified 1–3-second pre-contact aggression remains a separate blocked task: zero independently verified positive precursor videos are available. Weak first-onset forecasting, weak active-fight recognition and verified precursor prediction must not be pooled into one accuracy claim.

## Checkpoints, recommendation and runtime regression

- Keep default: `weights/baseline_weak_onset_guarded_v1.pt`, SHA-256 `954559460c10b85c25b532b5f7aa7d28edb141f38b308ebaeb199aca6c4d7820`; onset/active reference thresholds 0.603839457 / 0.611068130; research head eligibility [true, false]. The active threshold is diagnostic while that head is disabled.
- New dev-selected candidate: `C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection\weights\model_optimization_v1\selected_dev_readout.pt`, SHA-256 `3347375dab23fb8ffe15e9e72177c8c183769c480c7fa01243b1b3a11085bb8f`. All three seed readouts and frozen encoders are stored in one strict-load checkpoint.
- New separate validation-reference version: `C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection\weights\model_optimization_v1\selected_validation_reference.pt`, SHA-256 `c4592375a33fea5bef34e039956c3d93c1513df97c1f2822e2fd54a7f7e5c517`; onset/active reference thresholds 0.567235708 / 0.669162214, both research heads disabled. These are exploratory score-review reference values, not recommended operational alarm thresholds.
- Predeclared checks: `{"active_auroc_nonregression": true, "f1_gain": false, "fpr_cap": false, "minimum_precision": false, "recall_gain": false, "source_bootstrap_lower_positive": false}`. No default replacement is recommended. F1-optimal lowering would raise review load (49 FP windows on 162 negatives); constrained source-held-out results fail the practical precision/recall target.

| Model | Validation model pass ms/window (batch 32) | Derived replay ms/usable decision |
|---|---:|---:|
| baseline | 2.23 | 8.46 |
| advanced | 6.13 | 16.94 |
| headwise | 9.57 | 52.69 |
| readout | 8.04 | 23.69 |

Single measured CPU passes with two Torch threads; includes Python/loader/window overhead as applicable. Pose extraction and warmup-repeat benchmark methodology are not included. No end-to-end real-time claim is made.

Every one of the seven configurations also passed real full-window inference against its frozen feature predictions (maximum absolute score difference below 2e-6). candidate_runtime.json records per-configuration latency, 204-window sample count, 20.59% active prevalence and common 63.75% candidate-decision coverage. These runtime checks do not select a different model.

## Tests and artifacts

- C3/backend/optional suite: **246 tests + 62 subtests passed**, zero failures/skips; one existing Starlette/httpx deprecation warning.
- Nine new regression tests verify exact historical advanced logits, trailing-window history exclusion, padding exclusion, feature-group masks, finite/correct-direction loss gradients, tied-score constrained thresholds, infeasible precision fallback, source/F9 holdout and strict optimized-model loading/onset preservation.
- Existing causal-prefix, cache/source isolation, model loading, incident adapter and shared backend integration tests passed. Frontend: 13/13 tests; C3 TypeScript/Vite production build passed.
- One recorded-video runtime smoke run generated annotated MP4, scores and empty non-operational incident/notification envelopes using the new versioned checkpoint. The shared dashboard was not restarted or modified.

Artifacts under outputs/model_optimization_v1: protocol.json, selection.json, evaluation.json, development_comparison.csv, ablation_validation.csv/json, validation_predictions.csv, per_source_errors.csv, data_audit.json, ranking_uncertainty.json, weak_onset_lead_times.json, f9/, plots/, tests.log/xml, frontend logs, runtime_smoke/ and preservation manifests.

## Exact Windows PowerShell commands

Use the project Python interpreter. For a repeat experiment choose unused output/weights version directories; existing artifacts deliberately refuse overwrite. Commands below reproduce this run with a new suffix. All script stages must use the same suffix.

```powershell
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$py = '..\..\.venv\Scripts\python.exe'
$out = 'outputs/model_optimization_v2'
$weights = 'weights/model_optimization_v2'
& $py -B -m src.optimize_model prepare --out-dir $out --weights-dir $weights
& $py -B -m src.optimize_model train --out-dir $out --weights-dir $weights
& $py -B -m src.optimize_model evaluate --out-dir $out --weights-dir $weights
$env:MPLCONFIGDIR = Join-Path (Get-Location) ($out + '/mplconfig')
& $py -B -m src.optimization_diagnostics --out-dir $out
```

Run the existing new candidate (choose a fresh output directory each time):

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
& .\.venv\Scripts\python.exe -B components/c3_aggression_detection/run.py `
  --source dashboard/backend/uploads/b96502dd_F_9_1_2_0_0_review.mp4 `
  --seconds 8 --checkpoint weights/model_optimization_v1/selected_validation_reference.pt `
  --out-dir outputs/model_optimization_v1/manual_review_01
```

Run tests:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
$env:PYTHONDONTWRITEBYTECODE = '1'
$c3 = Join-Path (Get-Location) 'components/c3_aggression_detection'
$env:PYTHONPATH = @((Join-Path $c3 'data/phase1_work/testdeps'), (Join-Path $c3 'data/dashboard_deps_v1'), $c3, (Join-Path $c3 'legacy'), (Join-Path (Get-Location) 'dashboard/backend')) -join ';'
& .\.venv\Scripts\python.exe -B -m pytest components/c3_aggression_detection/tests `
  components/c3_aggression_detection/tests_optional dashboard/backend/test_c3_integration.py `
  -q -p no:cacheprovider --basetemp components/c3_aggression_detection/data/phase1_work/manual_optimization_tests_01
npm --prefix dashboard/frontend test
npm --prefix dashboard/frontend run build:c3
```

Normal user terminals can use npm as shown. This tool session used the bundled Node executable and the existing isolated npm CLI because npm was absent from its PATH; frontend logs record the successful result.

## Remaining limitations and next evidence needed

The point ranking gain is encouraging, but the constrained recall/F1 failure and wide source intervals prevent a default switch. Collect additional source-separated fight and hard-negative contexts, independently review contact/precursor timing, and evaluate a frozen operating policy on new sources with event-level false alarms/hour and missed-event counts. Tracking/selected-pair annotation is needed to distinguish detector/association errors from model limitations. Preserve the locked test for an explicitly frozen final evaluation. These experiments do not justify operational alerts or verified early-warning claims.

## Diagnostic figures

![precision recall](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/precision_recall.png)

![score distributions](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/score_distributions.png)

![confusion matrices](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/confusion_matrices.png)

![training validation loss](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/training_validation_loss.png)

![calibration](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/calibration.png)

![ablation comparison](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/ablation_comparison.png)

![f9 scores](C:/Users/Kulanjana/Downloads/J26-DS-320/components/c3_aggression_detection/outputs/model_optimization_v1/plots/f9_scores.png)

## Primary references

- [scikit-learn decision thresholds](https://scikit-learn.org/1.5/modules/classification_threshold.html): ranking scores and the decision cutoff are separate; tuning/evaluation data must be separated.
- [scikit-learn precision–recall example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html): AP and trapezoidal PR integration differ; this report names both.
- [Original focal-loss paper](https://arxiv.org/abs/1708.02002) and [Torchvision focal-loss definition](https://docs.pytorch.org/vision/0.12/generated/torchvision.ops.sigmoid_focal_loss.html): focal modulation was tested as an independent controlled arm.
