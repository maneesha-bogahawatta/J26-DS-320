# Component 03 — skeleton occlusion review

22 October 2026. Existing causal preprocessing was inspected before implementing alternatives. COCO-17 mapping, confidence filtering, same-track short-gap carry and quality abstention already existed. KNN exists only in a legacy noncausal imputer. The new conservative guard/motion variants are opt-in research tools; the original cache contract, preprocessing default, ST-GCN checkpoint and dashboard integration remain unchanged.

## Evidence-based implementation status before this work

| Feature | Status | Located implementation and tests |
|---|---|---|
| COCO 17-keypoint mapping | IMPLEMENTED | `c3_model.py`: COCO_EDGES, COCO_PARENT, build_adj; `src/pose_sequences.py`: pack_frames/validate_cache assert 17×3; `src/advanced_stgcn.py`: pose_streams uses this topology. `tests/test_advanced_pipeline.py` covers pose features, permutation and padding; current `tests/test_occlusion.py` checks mapping names/rendering. Explicit standard joint names are now in occlusion.COCO_KEYPOINT_NAMES. |
| Keypoint confidence threshold | IMPLEMENTED | `src/causal_pose.py`: PoseConfig.confidence=.30; CausalPosePreprocessor.update rejects nonfinite keypoints and low-confidence observation. `tests/test_causal_pose.py::CausalPoseTests.test_bad_time_ids_and_nan_masks` and missing-joint tests. YOLO box/person confidence .25 is separate. |
| Missing joints / original visibility masks | IMPLEMENTED | update stores observed/imputed/raw_confidence; missing is the complement and zeroed, not an inferred observation. pose_sequences stores masks; `src/pose_dataset.py::sequence_window` preserves them. `test_expiration_is_from_last_observation`, normalization/empty and advanced all-missing tests. |
| Bounded same-track temporal nearest observation | IMPLEMENTED | CausalPosePreprocessor.update carries the last observed joint on the same ID only while age<.25s; carry does not extend the observation clock. Confidence is original confidence × .5 × (1−age/.25). Observed EMA uses alpha=.6. Future-prefix, expiration and scene/new-ID tests exist. This is causal temporal nearest-observation carry, not KNN regression. |
| Nearest neighbor / KNN estimation | MISSING in active causal path; IMPLEMENTED in legacy only | `legacy/run.py::KNNPoseImputer` searches a k=5 bank including a template and visible-joint weighted distances; `impute_sequence` first adds the entire sequence to the bank and uses left/right interpolation. This can use future data. `src/causal_pose.py::process_persons_causal` explicitly avoids that imputer. Legacy presence is not evidence of online causal KNN support. |
| Anatomical neighboring-joint reconstruction | MISSING in original active path | COCO edges and bone features describe connectivity; they do not reconstruct hidden wrists/knees from neighbors. No such active reconstruction code/test was found. New motion vetoes use same-person historical bone-length bounds, a PARTIAL constraint only, not anatomical ground-truth recovery. |
| Person-track continuity | PARTIAL | `src/pose_sequences.py::extract_video` calls local YOLO11-Pose `model.track(... persist=True, tracker=bytetrack.yaml)`; scene resets rebuild the tracker. `src/pose_enrichment.py::match_pose_to_tracks` is one-to-one greedy box IoU for external IDs. `sequence_window` selects two IDs by past box area and fixes slots within a window; scene/gap breaks and CausalDecision identity resets exist. `tests/test_c2_compatibility.py` checks IDs, ambiguous one-to-one mapping and unmatched zero-confidence output. No identity ground truth proves robustness to undetected same-ID switches. |
| Low-quality rejection / abstention | IMPLEMENTED | `src/predict_advanced.py::score_cache/score_arrays` requires ≥8 history frames, average track quality≥.45 and ≥6 currently observed joints in one selected person. `CausalDecision.update` clears confirmation history on invalid pose/identity/scene gaps. Advanced all-missing, causal replay and decision tests exist. |

Standard COCO landmark ordering agrees with the [official Ultralytics COCO pose documentation](https://docs.ultralytics.com/datasets/pose/coco/). Tracker API and persist behavior are described in [official tracking documentation](https://docs.ultralytics.com/modes/track/); actual repository settings above were inspected rather than inferred from the docs. ByteTrack continuity is a tracker hypothesis, not proof of identity under overlap ([original paper](https://arxiv.org/abs/2110.06864)).

## Improvements implemented, scope and causal constraints

`src/occlusion.py::GuardedPosePreprocessor` subclasses the existing processor, avoiding duplicate extraction/feature/model modules. It is used only by the explicit experiment, never silently substituted when reading old v1 caches.

- **Guarded carry** clears history when fewer than six current observed anchors remain, when a same-ID bounding box is discontinuous (IoU<.05 plus center displacement>.5 prior-box diagonal or area ratio outside .25–4), or when the track gap exceeds .25s. Scene cuts clear history. A new ID cannot inherit another ID's joints. This vetoes unsupported all-body carry and gross same-ID jumps; an overlapping identity switch that produces a plausible box can still escape detection.
- **Optional short-gap motion** uses two past observed positions of the same joint and track only. It requires ≥8 current anchors, current/previous box IoU≥.3, observation separation .04–.2s, estimate age≤.134s, both observed confidences≥.5, speed≤2 box diagonals/s, extrapolation≤.15 diagonal and a point inside the expanded current box. Historical same-person connected-bone lengths veto ratios outside .5–1.5 when a neighboring joint is observed. No joint is copied from another track.
- Motion confidence is decaying carry confidence ×.7. History updates only from observed joints, never from previous estimates. Long missing gaps expire from the last true observation. No future frames, symmetric interpolation, population KNN bank or fabricated observed landmarks are used.
- Explicit `observed`, `estimated`, `missing` and estimate-kind masks distinguish observed, temporal carry and motion estimates. Original `raw_confidence`, `observed` and `imputed` semantics remain. Missing coordinates stay zero with false masks; the combined mask is not a claim of detection. The runtime's observed-joint gate remains active.
- Anatomical position reconstruction and KNN were not added. Under partial 2D occlusion, several anatomically plausible poses are possible; existing caches supply detector outputs, not independently verified hidden-joint truth. Future-sequence KNN would leak information and same-frame cross-person nearest neighbors would violate identity. The limited historical bone constraint is testable without claiming to resolve that ambiguity.

The 17 names are nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee and left/right ankle, in standard COCO order. Existing advanced features retain hip-center/visible-center normalization, joint/bone/backward-velocity/interaction streams and masks. Base preprocessing/deployed inference remain v1 because the experiment did not establish robust model improvement.

## Experimental method

`src/occlusion_experiment.py` writes its fixed protocol before processing. It authenticates all 16 existing locked-validation pose caches; uses their raw detector keypoints/boxes/IDs/timestamps and the original weak intervals; performs zero new extraction, fitting, threshold tuning or test evaluation. Replay with default preprocessing reconstructs every original cached keypoint exactly: maximum absolute difference **0.0**. Existing caches are not overwritten or relabeled as a new preprocessing version.

Native comparison: 16×120 sampled frames = 1,920 real cached frames per mode, .2s model decision spacing, baseline and candidate frozen weights/reference thresholds. Known-active sample N=412, positives82/negatives330, 218/630 known-active cutoffs abstain (18 positive). Known-onset N=315, positives92/negatives223, 197/512 known-onset cutoffs abstain. Both dynamic-coverage and common-observable-cutoff reports are saved; coverage did not change across methods, so common comparisons agree here.

Stress tests: four predeclared diagnostic sources (f_101_1_2_0_0, f_147_1_0_1_0, n_360_0_0_1_0, f_9_1_2_0_0), eight scenarios and three modes = 96 runs on existing real pose sequences; baseline inference only for stress runs. Sources/parameters are common, not F9-specific. Dropouts hold out one wrist, four elbow/wrist joints, six lower-body joints, all joints for six frames, wrists during actual box overlap, moving wrists, the first wrist after a simulated ID reassignment, or a wrist for twelve frames (.8s). Fast movement uses measured detector motion; overlapping boxes are only a proxy for physical overlap. Complete temporary occlusion retains a detector box with no observed joints; tests also cover complete track absence and scene reset.

High-confidence original detector coordinates artificially hidden by the stress test are proxy reference points. They are not human-labeled occluded joints, motion capture or pose ground truth. Reported proxy error is Euclidean detector-coordinate error divided by box height, only for hidden confidence≥.6 joints that the method actually estimates. Guard rejection changes that denominator; lower error after rejecting difficult joints cannot alone prove better reconstruction. Raw masks and error Ns are retained. Temporal stability is mean same-ID visible-joint step in image-normalized x/y coordinates, not meters, per-second speed or ground-truth smoothness. Latencies are descriptive CPU timings, not a controlled throughput benchmark.

## Before / after results

### Native active-fight model performance

| Preprocessing | Checkpoint | AUROC | AP | PR-AUC | Precision | Recall | F1 | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|
| default | baseline | 0.4782 | 0.3031 | 0.2988 | 0.3333 | 0.1707 | 0.2258 | 14/28/302/68 |
| default | candidate | 0.6879 | 0.3638 | 0.3558 | 0.3947 | 0.1829 | 0.2500 | 15/23/307/67 |
| guarded | baseline | 0.4824 | 0.3038 | 0.2996 | 0.3488 | 0.1829 | 0.2400 | 15/28/302/67 |
| guarded | candidate | 0.6861 | 0.3615 | 0.3536 | 0.4103 | 0.1951 | 0.2645 | 16/23/307/66 |
| motion | baseline | 0.4826 | 0.3040 | 0.2997 | 0.3488 | 0.1829 | 0.2400 | 15/28/302/67 |
| motion | candidate | 0.6813 | 0.3588 | 0.3509 | 0.4103 | 0.1951 | 0.2645 | 16/23/307/66 |

Thresholds stay baseline .611068 and candidate .669162; these are score-level diagnostic classifications. Candidate head eligibility stays disabled. Native onset AUROC is .5793 / .5789 / .5790 (default / guarded / motion); its fixed-reference F1 stays .2479 baseline / .3359 readout. The onset path consumes the skeleton coordinates and confidence channels, so changed preprocessing can change its scores. The candidate's onset predictor is still the copied baseline predictor within each mode. All detailed per-head metrics, abstentions and matched cohorts are in `native_evaluation.json`.

Guarded/motion variants each add one active TP at the fixed reference in each model, but candidate ranking falls .6879 → .6861 → .6813 and PR-AUC also falls (0.3558 → 0.3536 → 0.3509; use the saved JSON for full precision). This small, inconsistent change on a reused weak cohort does not demonstrate improved violence detection or justify changing the default. No promotion was performed.

### Pose and processing diagnostics on unmodified cached sequences

| Policy | Preprocess ms/frame | Mean normalized step | Motion estimates | All usable cutoffs |
|---|---|---|---|---|
| default | 0.367 | 0.005721 | 0 | 421 / 640 |
| guarded | 0.566 | 0.005810 | 0 | 421 / 640 |
| motion | 0.877 | 0.005850 | 196 | 421 / 640 |

All 144 native/stress pose reprocessing runs produced finite keypoints with disjoint observed/estimated masks; 192 model replays completed without nonfinite output (native uses two models, stress one). The new guard logged 63 low-anchor resets and ten expired-gap resets in native sequences. It lowers carried-joint coverage, but does not recover a currently unobserved person's true pose. Motion mode makes 196 extra motion-position estimates in native data while keeping the same estimated-mask count and inference coverage. Average per-source estimated-joint fraction is .02122 default and .01947 guarded/motion; these are descriptive source means rather than pooled joint counts. Higher estimated coverage would not be inherently better.

### Artificial dropout proxy error and inference continuity

| Scenario | Default proxy error (N) | Guarded proxy error (N) | Motion proxy error (N) | Usable cutoffs (each mode) |
|---|---|---|---|---|
| one_joint | 0.0676 (37) | 0.0611 (34) | 0.0627 (34) | 134/160 |
| multiple_joints | 0.0498 (141) | 0.0419 (127) | 0.0474 (127) | 134/160 |
| lower_body | 0.0355 (244) | 0.0274 (216) | 0.0287 (216) | 134/160 |
| complete_occlusion | 0.0342 (568) | N/A (0) | N/A (0) | 128/160 |
| overlap | 0.0643 (154) | 0.0643 (154) | 0.0593 (154) | 134/160 |
| fast_motion | 0.0732 (353) | 0.0732 (346) | 0.0728 (346) | 134/160 |
| id_change | N/A (0) | N/A (0) | N/A (0) | 111/160 |
| long_gap | 0.0676 (37) | 0.0611 (34) | 0.0627 (34) | 134/160 |

Every scenario has 160 decisions (4 sources×40) per mode. Across all eight scenarios each mode has **1,043/1,280** usable decisions and 204 decision-identity resets. Full-occlusion dropouts abstain rather than manufacture current evidence. ID changes reduce usable windows to 111/160 because the new track cannot inherit history; long joint gaps expire without extending fill. The guard estimates zero artificially hidden joints under complete occlusion, so N/A is not zero error. Partial and overlapping scenarios preserve finite inference without claiming every missing landmark is recovered.

Motion increases proxy error versus guarded carry for one/multiple/lower-body dropouts and reduces it for the overlap/fast-motion proxies in this small selection. Average normalized step also rises slightly, not evidence of improved stability. KNN superiority and improved detection have not been established. The current engineering outcome is explicit provenance, bounded opt-in safety guards, tested alternatives and clearer visualization, not a validated new estimator or default policy.

![Measured weak validation and occlusion diagnostics](outputs/pp1_v1/research_comparison.png)

## Dashboard visualization

`src/occlusion.py::draw_pose_evidence`, called from the existing `predict_advanced.render`, draws observed joints solid green, estimates as hollow amber circles and amber connected edges, missing joints nowhere, and O/E/M counts per tracked box. Below-six-current-joint or track-quality<.45 skeletons get LOW POSE labels. The footer legend explains colors; abstention remains visible. This rendering uses existing masks, not AI assertions of visibility. The original team dashboard replays this genuine annotated video and retains source-clock scores/evidence. Missing joint locations are omitted rather than falsely plotted at inferred anatomical positions. Text can overlap in crowded scenes, and a low-quality box flag is a visibility gate rather than measured 3D occlusion.

Browser inspection and renderer unit tests confirm different observed/estimated symbols, omitted missing joints, counts and low track-quality labels. Both the recorded original-context and fresh-upload videos decoded and were served successfully. A final F_147 rerender with the latest renderer reproduces the demo MP4 and score CSV byte-for-byte (`renderer_final_check.json`); no re-extraction was needed. The new variants remain offline opt-in; the demo uses the unchanged default causal carry with improved explanatory rendering.

## Tests, preservation and remaining limitations

Full current suite: **274 tests + 65 subtests passed**, one existing dependency deprecation warning. `tests/test_occlusion.py` adds thirteen tests including three partial-dropout subcases. Existing future-prefix, decay, bad confidence/NaN, scene/new-ID reset, all-missing abstention, mask/padding invariance and C2 external-ID compatibility tests also pass. Frontend 13 tests pass; TypeScript/Vite production build passes; lint has six pre-existing warnings in unchanged files.

No models, pose caches, annotations, datasets, deployment defaults, shared backend or teammate sources were changed. The protected-file audit passed (2663 files, no missing/changed protected file; one authenticated log append). No commit or push. The current default SHA-256 remains `954559460c10b85c25b532b5f7aa7d28edb141f38b308ebaeb199aca6c4d7820`.

Remaining research needs: actual landmark and identity annotations under genuine occlusion; more held-out source diversity; reliable full-body tracking when people overlap/leave/re-enter; uncertainty for ambiguous same-ID switches; full negative hours; independently verified cue/contact labels. Confidence decay indicates an estimate, not calibrated geometric uncertainty. Detector caches cannot measure hidden-joint truth. A strict ban on undetected identity switches cannot be guaranteed from anonymous boxes/IDs alone; the experiment detects only the specified discontinuities and abstains when evidence fails.

Reproduce with the exact PowerShell commands in PP1_MODEL_PROGRESS.md (`src.occlusion_experiment --out-dir` pointing to a new directory). Machine-readable protocol, native score rows, matched/dynamic metrics, per-source masks/latencies/reset counts and all stress metrics remain in `outputs/occlusion_v1/`. No new fit is necessary.
