# Component 03 — PP1 model progress

Completed locally on 22 October 2026. Genuine weak-supervision research and the original shared dashboard are demonstrable today. No new model training, human reviews, consensus labels, verified prediction claims, commits or pushes were performed.

## Completed implementation

The project already contains local YOLO11n-Pose extraction, ByteTrack identities, COCO-17 confidence/visibility masks, causal pose caches, source-separated fit/dev/validation archives, baseline and multi-stream ST-GCN training, model diagnostics, immutable human/contact-review tooling, event evaluation, and the shared-dashboard C3 bridge. Existing modules and caches were reused.

This continuation adds:

- `src/pp1_evaluate.py`: fresh frozen-checkpoint validation scoring, full confusion/error CSVs, score distributions, AP and trapezoidal PR-AUC. Every rescored historical probability matches exactly: maximum difference **0.0**. No threshold search or fit occurs here.
- `src/provisional_labels.py`: separate train-fit-only proposals using existing weak targets, checkpoint scores and observed-pose v4 interaction features. Source/cache identities and past-frame cutoffs are checked. No strict supervised loader consumes these files.
- `src/occlusion.py` and `src/occlusion_experiment.py`: opt-in continuity guards and constrained short-gap motion estimates; all-16-source native validation and eight controlled occlusion scenarios over real cached detector outputs. See SKELETON_OCCLUSION_REVIEW.md for the decision to retain default preprocessing.
- `src/predict_advanced.py`: reuse of its unchanged numerical replay loop via `score_arrays`; clearer observed/estimated/missing joint rendering, track counts and low-pose labels. Cached input/model defaults remain unchanged.
- Additive C3 frontend controls inside the original `dashboard/frontend/`: explicitly labeled authentic-result replay and score graphs in ordinary recorded-video results. The original team layout, recorded/live navigation, shared backend and teammate components are preserved. No separate C3 research-video tab was introduced.
- `src/prepare_pp1_demo.py`, `src/pp1_dashboard_check.py` and `RUN_PP1.ps1`: authentic F_147 artifacts, isolated job history, export/media/provenance checks and a repeatable launcher.

## Frozen model comparison — weak labels only

The locked split remains 791 train / 141 validation / 68 test sources. Fitting and internal development are source-separated subsets of locked train. This task reads existing development caches and metadata; it does not open test media or use test examples for fitting, thresholds or selection. Only a small previously selected validation cache cohort is evaluated, not the full validation population.

Active target: cutoff inside an original UBI binary fight interval. Onset target: first original annotated fight onset in the next three seconds, with the existing post-onset/boundary masks. These weak intervals do not establish physical contact or independently verified threatening cues. Overlapping windows are correlated observations; clip filenames are not labels. Unknown targets and pose abstentions are not negative ground truth.

The historical archive samples approximately every 0.4s in selected eight-second contexts. Both models receive identical eligible inputs within each table. The quality-matched archive removes low-quality-current-pose windows; it is reported separately. The 16 cached sources include three with no materialized eligible windows, leaving 13 represented sources.

| Archive | Sources with materialized windows | Materialized / candidate cutoffs | Known active + / - | Known onset + / - |
|---|---|---|---|---|
| original | 13 | 224 / 320 | 51 / 173 | 50 / 111 |
| quality_matched | 13 | 204 / 320 | 42 / 162 | 46 / 106 |

Original pose eligibility is 224/320 cutoffs (70%); quality-matched materialization is 204/320 (63.75%). These are coverage proportions, not classification accuracy. Onset eligibility also loses unknown post-onset targets.

### Original cohort: active fight (224 windows; 51 positive, 173 negative)

| Checkpoint | Threshold | AUROC | AP | PR-AUC (trap.) | Precision | Recall | F1 | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.611068 | 0.5341 | 0.3729 | 0.3664 | 0.4231 | 0.2157 | 0.2857 | 11/15/158/40 |
| headwise | 0.590869 | 0.6723 | 0.3894 | 0.3824 | 0.6000 | 0.0588 | 0.1071 | 3/2/171/48 |
| readout | 0.669162 | 0.6961 | 0.4144 | 0.4009 | 0.4737 | 0.1765 | 0.2571 | 9/10/163/42 |

### Quality-matched cohort: active fight (204 windows; 42 positive, 162 negative)

| Checkpoint | Threshold | AUROC | AP | PR-AUC (trap.) | Precision | Recall | F1 | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.611068 | 0.4788 | 0.3107 | 0.3037 | 0.3182 | 0.1667 | 0.2187 | 7/15/147/35 |
| headwise | 0.590869 | 0.6480 | 0.3430 | 0.3367 | 1.0000 | 0.0476 | 0.0909 | 2/0/162/40 |
| readout | 0.669162 | 0.7193 | 0.4062 | 0.3905 | 0.4444 | 0.1905 | 0.2667 | 8/10/152/34 |

### Original cohort: weak onset (161 windows; 50 positive, 111 negative)

| Checkpoint | Threshold | AUROC | AP | PR-AUC (trap.) | Precision | Recall | F1 | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.603839 | 0.6081 | 0.4461 | 0.4384 | 0.5714 | 0.1600 | 0.2500 | 8/6/105/42 |
| headwise | 0.691749 | 0.6652 | 0.4082 | 0.3988 | 0.4545 | 0.2000 | 0.2778 | 10/12/99/40 |
| readout | 0.567236 | 0.6081 | 0.4461 | 0.4384 | 0.6316 | 0.2400 | 0.3478 | 12/7/104/38 |

### Quality-matched cohort: weak onset (152 windows; 46 positive, 106 negative)

| Checkpoint | Threshold | AUROC | AP | PR-AUC (trap.) | Precision | Recall | F1 | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.603839 | 0.5913 | 0.4364 | 0.4283 | 0.5714 | 0.1739 | 0.2667 | 8/6/100/38 |
| headwise | 0.691749 | 0.6727 | 0.4145 | 0.4034 | 0.5000 | 0.2174 | 0.3030 | 10/10/96/36 |
| readout | 0.567236 | 0.5913 | 0.4364 | 0.4283 | 0.6316 | 0.2609 | 0.3692 | 12/7/99/34 |

AP is stepwise average precision; PR-AUC is trapezoidal integration of the tied-score precision-recall curve. Both are reported rather than interchanging their names. Confusion order in all tables is TP/FP/TN/FN (equivalent matrix rows true negative/positive: [[TN,FP],[FN,TP]]). Per-window false positives and false negatives, source IDs, scores and cutoffs are in `outputs/pp1_v1/weak_evaluation/predictions.csv`.

The readout candidate's active AUROC improvement **0.6723 → 0.6961** is reproduced from the prior headwise model. At its already stored validation-reference threshold, candidate active recall is only 0.1765 and F1 0.2571, below baseline F1 0.2857 on the original cohort. The candidate's onset weights are the unchanged baseline onset weights; its different onset Precision/Recall/F1 come from a different reference threshold, not a newly improved onset predictor.

Threshold provenance matters: baseline/headwise thresholds originated in train-only development; the selected readout file contains previously explored validation-reference thresholds excluding F9. Its displayed reference-threshold scores therefore are exploratory diagnostics on a reused cohort, not an independent operating-point estimate. Both readout research candidate heads are disabled. The baseline permits onset research candidates and disables active-fight candidates. All operational notifications remain disabled. A diagnostic threshold crossing is not an operational alert.

### Why Recall and F1 remain limited

- Score distributions overlap substantially. On quality-matched active examples, baseline negative/positive median scores are 0.492/0.460; readout medians are 0.420/0.534. A high cutoff suppresses many positives as well as negatives.
- Prior seven configurations across three seeds improved ranking but did not satisfy promotion criteria. Existing leave-one-source-out validation threshold diagnostics excluded the scored source and always excluded F9 from tuning: under the FPR≤10% tuning policy, quality-matched baseline TP/FP/FN = 7/2/35 (F1 .2745), readout = 7/17/35 (F1 .2121, realized held-source FPR .1049). The readout F1-oriented policy reaches recall .6190 with 49 FP (FPR .3025). This trade-off does not justify enabling active alerts. These reused-validation analyses are preserved, not new independent evidence.
- Weak boundaries may describe already ongoing fighting rather than visible pre-contact aggression. Only 42 quality-matched active positives from 13 represented sources support the relevant small-cohort comparison. Noisy targets, source shift and correlated windows limit generalization; these are plausible explanations, not independently proven causes.
- Pose extraction can miss or fragment identities under occlusion. The fresh current-policy runtime replay retains **421/640** decisions (65.78%); missing/current-joint/history quality causes abstention. On known active targets there are 412/630 usable cutoffs, including 82 positive windows; 18 positive cutoffs abstain. Do not omit those abstentions when discussing overall detection coverage.
- EMA score smoothing, two-of-three confirmation, identity resets and per-kind cooldown further separate raw model scores from emitted research candidates. Low-pose decisions clear confirmation history. These safeguards remain intact.

The fresh 0.2s runtime experiment is a different cohort from the 0.4s archive tables: 16 source contexts, 640 decisions, 421 usable; weak-onset metrics use 315/512 known decisions and active metrics 412/630. Default baseline active AUROC .4782, AP .3031, PR-AUC .2988, Precision .3333, Recall .1707, F1 .2258 (TP14/FP28/TN302/FN68). The candidate at its frozen reference has AUROC .6879, AP .3638, PR-AUC .3558, Precision .3947, Recall .1829, F1 .2500 (TP15/FP23/TN307/FN67). This does not replace the original-cohort comparison. Onset research event replay flags 2/8 weak-onset contexts and produces three candidates in weak-negative contexts; these are not verified event Recall or FAR/hour.

## Automatic proposals — isolated and provisional

`outputs/pp1_v1/provisional_labels_validated/` contains **1240 windows from 115 locked train-fit sources**. No internal-dev, validation or test window is proposed. The repeated run with explicit source/past-frame checks reproduces the first proposal file byte-for-byte.

| Provisional category | Windows |
|---|---|
| provisional_preonset_motion | 110 |
| provisional_quiet_negative | 50 |
| uncertain_or_teacher_disagreement | 562 |
| weak_active_supported | 285 |
| weak_active_teacher_disagreement | 91 |
| weak_preonset_context_only | 142 |

The fixed .5 active-support and .35 quiet-negative rules only prioritize weak contexts; the observed-only motion cues reuse `auto_annotate_precursors_v4.person_from_pose` / `interaction_evidence`. Past-frame evidence is bounded to the same segment and ≤.25s. A motion-supported pre-onset context still has no confirmed first contact and no verified precursor target. Teacher agreement is not independence: these teachers were trained on fitting sources. Therefore no self-training or new model fit was justified today.

All records use a distinct `c3_automatic_proposals_v1` schema, `label_status=provisional_automatic`, `human_review=false`, `eligible_for_verified_evaluation=false`, `eligible_for_strict_training=false`, and null verified contact. Reviewer decisions are empty. Existing weak, AI, single-review, uncertain and independently verified partitions remain separate and unchanged. These outputs cannot be described as 1,240 new verified training examples.

## F_9 limitation

F_9 is locked validation. The original 26.2–34.2s context freshly reproduces every historical score/validity/identity/candidate decision (maximum difference 0.0). It has 35/40 usable decisions, 16 observed track IDs, nine selected-identity changes, three history-gate failures and two current-observation failures. Peak baseline onset/active scores .5556/.5759 remain below reference thresholds .603839/.611068; active candidate generation is also disabled. Even the prior diagnostic reference-both-enabled replay produces zero candidate decisions. F9-specific thresholds, parameters or tracker fixes were not introduced.

The uploaded F9 derivative's source offset is only an approximate image alignment, not authoritative edit provenance or a new independent source. Weak onset 31.166667s is not confirmed first contact. This remains a documented miss; F_147 is a legitimate tested demonstration example, not a substitute evaluation result for F9.

## Demonstration on the original shared dashboard

The actual server is `dashboard/backend/main.py` with existing C3 routes/bridge, serving the built original `dashboard/frontend/`. `RUN_PP1.ps1` selects port 8002 and `outputs/pp1_v1/dashboard_state` so previous local jobs are not overwritten. Existing port-8000 processes were left alone.

1. Open `http://127.0.0.1:8002/`, choose **Check a recorded video**. The original Recorded video / Live camera navigation remains.
2. For the repeatable presentation path, select **f_147_1_0_1_0 · 2026-10-21T20:10:58.879828Z** under **Replay recorded C3 results**. The page explicitly says recorded inference replay and no new inference. It plays authentic newly rendered output with numerically identical scores to the previously tested F_147 run, source time 13.43–21.43s.
3. Show detected person identities, solid green observed joints, hollow amber estimated joints, O/E/M counts, missing joints omitted, uncalibrated onset/active curves and quality gaps. Clicking a candidate seeks the rendered clip at source-relative time; evidence retains original-video seconds. The score graph also seeks the recording.
4. Open a candidate to inspect evidence, original source time, current observed joints, research policy and review controls. Active scores can be high while the active head is disabled; the UI explains that policy instead of declaring no violence.
5. Record operator workflow assessments and export the C3 ZIP. These assessments are non-training audit records, not independent annotations. An actual UI software test used shift ID PP1_UI_TEST and an explicit note that no human annotation/ground truth was performed; it is exported with `eligible_for_training=false` and `verified_precursor_annotation=false`.
6. To show a new upload, choose `outputs/pp1_v1/upload_source/F_147_pp1_review.mp4`. This is a silent eight-second derivative of the same legitimate source, not an independent sample. The tested real browser file-chooser → upload → YOLO/ByteTrack → ST-GCN → annotated media flow took about 18.7s end-to-end on CPU, including 12.922s pose extraction and .484s model replay. It returned 37/40 usable decisions and two authentic onset research candidates. A new extraction/initialization/re-encoding can change scores/timing versus original-cache replay; exact equality is not promised for uploads.

The recorded original-clock run also contains two authentic candidates, 37/40 valid decisions, .406s model replay and zero operational notifications. Replay requires no new model run; cached model timing excludes pose extraction. No end-to-end CPU real-time claim is made. No C1/C2/C4 detector model is registered in this isolated demo; the UI explicitly reports that and no teammate output was fabricated. Live-camera code is retained, but live C3 temporal violence inference is not asserted as validated by this demo.

Actual checks: both result APIs, 40-row scores, two genuine incidents per job, correct default checkpoint hash, byte-range media HTTP206, real ZIP download and ZIP artifact/review contents passed. Browser controls, original layout, evidence, recorded label and curves were inspected. The browser-downloaded uploaded-job ZIP is `C:/Users/Kulanjana/Downloads/c3_cb235c158ad74722a52e4f7c9afd1fb4.zip`. `dashboard_checks_final.json`, `browser_checks.json`, screenshots and job folders retain the evidence. Use a maximized desktop browser for presentation; this run's small embedded viewport horizontally crops the pre-existing desktop result columns.

![Authentic recorded result in the shared dashboard](outputs/pp1_v1/dashboard_replay.png)

## Validation and preservation

- Full C3, optional and original shared-backend integration suite: **274 passed + 65 subtests passed**, one existing Starlette/httpx deprecation warning; no failures/skips. Thirteen new tests cover causal prefix invariance, partial/full occlusion, decay, long absence, ID/scene reset, overlapping tracks, same-ID box jump, conservative motion/fallback, mask/render semantics, low track-quality labeling and proposal restrictions. `tests_final.log` and `tests_final.xml` contain the final result.
- Frontend Node tests: **13/13 passed**. TypeScript + Vite `build:c3` passed (61 modules). Oxlint exits successfully with six pre-existing warnings in unchanged shared components/hooks; no new warning from the C3 additions.
- Fresh archive predictions match historical scores exactly; default raw-cache reprocessing also matches all 16 caches exactly. Existing F9 regression check passes. Strict verified evaluator completes with status blocked and null unsupported metrics, as intended.
- Protected snapshot: 2663 existing files checked; 2662 byte-identical, one already-running server log append only with its original byte prefix authenticated. Zero changed protected files and zero missing files. Checkpoints, annotations, caches, datasets, teammate sources, shared backend, defaults and deployment selection are preserved. Only C3-owned implementation and additive C3 UI files changed; all generated artifacts stay inside C3 except the requested browser download. No commit/push.

Default checkpoint SHA-256: `954559460c10b85c25b532b5f7aa7d28edb141f38b308ebaeb199aca6c4d7820`. Candidate SHA-256: `c4592375a33fea5bef34e039956c3d93c1513df97c1f2822e2fd54a7f7e5c517`. Full evidence lives under `outputs/pp1_v1/` and `outputs/occlusion_v1/`; detailed error rows and native/stress score/quality files are retained. Early failed preparation/check invocations are retained in their logs; final successful checks are the final manifest/dashboard-check files and passing test report.

## Strict early-warning research blocker and next work

Strict train and validation each still have **zero verified precursor-positive sources, zero verified negative sources, zero confirmed first-contact sources and zero eligible supervised positive/negative windows**. Fresh `verified_evaluation.json` reports zero eligible contact events and zero verified negative camera-hours. Verified Precision, Recall, F1, PR-AUC, lead time and false alarms/hour remain null. Weak window FPs cannot be divided by retrospectively selected clip duration to invent operational FAR.

When reviewers become available, reuse the existing independent A/B contact-review pack: observe visible cues and first physical contact separately, retain uncertainty/single-review disagreements, and validate linked attested votes before forming a strict training view. More source-diverse verified positives and full reviewed negative recordings are needed; keep the locked test untouched until the final frozen protocol. Validate pose variants on actual landmark/identity ground truth and multiple held-out cohorts before promoting preprocessing. Human review is a future research requirement, not a prerequisite for today's authentic weak-supervision PP1 demonstration.

## Exact PowerShell commands

Build and launch in terminal 1 (dependencies are already installed locally; this uses the tested bundled Node):

```powershell
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
$node = 'C:\Users\Kulanjana\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$npmCli = 'components/c3_aggression_detection/data/dashboard_node_tools_v1/package/bin/npm-cli.js'
$env:PATH = (Split-Path -Parent $node) + ';' + $env:PATH
& $node $npmCli --prefix dashboard/frontend run build:c3
if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
.\components\c3_aggression_detection\RUN_PP1.ps1 -Port 8002
```

In terminal 2, or directly open the URL if the server is already running:

```powershell
Start-Process 'http://127.0.0.1:8002/'
```

Reproduce validation, occlusion and provisional engineering in unused version directories; no training command is needed:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$py = '..\..\.venv\Scripts\python.exe'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$runOut = "outputs/pp1_recheck_$runTag"
& $py -B -m src.pp1_evaluate --out-dir "$runOut/weak"
& $py -B -m src.occlusion_experiment --out-dir "$runOut/occlusion"
& $py -B -m src.provisional_labels --out-dir "$runOut/proposals"
& $py -B -m src.evaluate_advanced_runtime --out-dir "$runOut/runtime"
& $py -B -m src.evaluate_verified --scores "$runOut/runtime/runtime_validation_scores.csv" --out "$runOut/verified_evaluation.json"
# Existing PP1 state must include the actual browser-upload test for this check:
& $py -B -m src.pp1_dashboard_check --port 8002 --out "$runOut/dashboard_checks.json"
```

Optionally prepare an additional authentic replay state, preserving the supplied PP1 state:

```powershell
$demoOut = "outputs/pp1_demo_$runTag"
& $py -B -m src.prepare_pp1_demo --out-dir $demoOut
& .\RUN_C3_DASHBOARD.ps1 -Port 8003 -StateDirectory "$demoOut/dashboard_state"
```

Run the complete tests and frontend checks:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
$c3 = Join-Path (Get-Location) 'components/c3_aggression_detection'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = @((Join-Path $c3 'data/phase1_work/testdeps'), (Join-Path $c3 'data/dashboard_deps_v1'), $c3, (Join-Path $c3 'legacy'), (Join-Path (Get-Location) 'dashboard/backend')) -join ';'
$testTag = Get-Date -Format 'yyyyMMdd_HHmmss'
& .\.venv\Scripts\python.exe -B -m pytest components/c3_aggression_detection/tests `
  components/c3_aggression_detection/tests_optional dashboard/backend/test_c3_integration.py `
  -q -p no:cacheprovider --basetemp "components/c3_aggression_detection/data/phase1_work/pp1_$testTag"
$node = 'C:\Users\Kulanjana\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$npmCli = 'components/c3_aggression_detection/data/dashboard_node_tools_v1/package/bin/npm-cli.js'
$env:PATH = (Split-Path -Parent $node) + ';' + $env:PATH
& $node $npmCli --prefix dashboard/frontend test
& $node $npmCli --prefix dashboard/frontend run lint
& $node $npmCli --prefix dashboard/frontend run build:c3
```
