**Component 03: runnable final research pipeline and measured limits**

The component now has reproducible split/annotation infrastructure, two trained
weak-supervised ST-GCN variants, source-grouped internal development, conservative
research operating points, inference on a new video, current-pose abstention,
annotated videos, CSV/JSON outputs, and a local interactive report. This is a
working research implementation. A validated operational early-aggression
predictor has **not** been achieved; the measured results below explain why.

Open the prepared dashboard from the component directory:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
.\RUN_C3_DEMO.ps1 -Serve
```

Visit `http://127.0.0.1:8765` in your browser. The server binds only to loopback;
Ctrl+C stops it. The report is `outputs/final_c3_v2/index.html`. It includes four
annotated videos, score plots, model comparisons and direct CSV/JSON links.
Earlier `final_c3_v1` outputs are retained; **v2** adds the current-observation
guard. No external scripts or uploads are used.

If local PowerShell policy prevents a script from running, the equivalent direct
command is available without changing execution policy:

```powershell
& '..\..\.venv\Scripts\python.exe' -m http.server 8765 --bind 127.0.0.1 --directory outputs/final_c3_v2
```

**Run a new video**

```powershell
.\RUN_C3_DEMO.ps1 -Video 'C:\path\your_video.mp4' -Seconds 8
```

The launcher saves a new version under `outputs/user_run_<UTC timestamp>/`.
It reuses the existing local YOLO11-Pose weights and ByteTrack pipeline, exports
`demo.mp4`, `scores.csv`, `alerts.json`, and `prediction.json`, and preserves all
old results. Output remains inside Component 03. New-video extraction is bounded
to eight seconds by default. Existing compatible caches are reused on reruns.

Equivalent Python commands, including an already cached example:

```powershell
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
& $C3Python -m src.predict_advanced --video 'C:\path\your_video.mp4' --seconds 8 --out-dir outputs/my_video_v1 --render
& $C3Python -m src.predict_advanced --video-id f_79_1_0_0_0 --out-dir outputs/cached_replay_v1 --render
```

Default output is **research scores only**, with operational alerts disabled.
`--research-alerts` (or launcher `-ResearchAlerts`) explicitly enables only heads
which pass the internal-development research guard. It does not validate them for
operational use. Scores are uncalibrated. An abstention means the system lacks a
usable current pose, not that the scene is non-aggressive.

**What changed beyond the first pilot**

The original first pilot's train caches were almost entirely fight sources and
its usable weak no-fight validation poses came from one source. The new run:

- Recovered 109 completed training caches without repeating their extraction.
- Selected pose-visible contexts from distinct locked source groups and extracted
  only 112 additional source seconds: six train weak no-fight, four validation
  weak no-fight and four validation weak fight contexts.
- Kept all 67 original test-list identities and the duplicate companion in the
  locked test split. No test source was used for fitting, threshold selection,
  label selection or this evaluation.
- Audited all 131 selected source annotations: binary vector lengths and
  FPS-to-seconds conversion match the locked intervals. This does not resolve
  semantic ambiguity between visible contact/aggression and weak annotations.
- Used 92 training videos for fitting and 23 different training source groups for
  internal development. External validation has 16 indexed contexts, 13 with
  eligible materialized windows. All source groups are disjoint across roles.
- Materialized 1,349 fit, 401 internal-development and 224 validation windows.
  The validation forecast head has 161 supervised windows, including 50 positives;
  the active head has 224, including 51 positives. Windows overlap.

Original locked dataset totals remain 791 train / 141 validation / 68 test.
This experiment uses a subset of those sources, not a new global split.
The extra test identity is `N_770_0_0_1_0`, an exact duplicate of reserved
`N_527_0_0_1_0`. The five old training-review overlaps (`F_0`, `F_120`, `F_38`,
`F_41`, `F_95`) remain test-only; their full identities and the duplicate audit
are recorded in `PHASE1.md` and `outputs/phase1_foundation/`.

**Explicit supervision policy**

Protocol `ubi_weak_first_onset_within_3s_v1` predicts whether the **original UBI
first annotated fight onset** is in the next three seconds. Before that first
onset, times outside the horizon are weak negatives; after the first onset the
forecast target is masked. Original weak binary intervals supervise active fight.
A 0.1-second boundary guard masks ambiguous annotation boundaries.

These are new, explicitly defined weak forecasting targets, not verified
aggression-precursor labels. No AI proposals or v3/v4 heuristic outputs were
promoted to human ground truth or used as verified targets. The 20 AI reviews
remain `provisional_ai_review`, and independently verified precursor labels
remain zero. Future annotation times are used only to construct offline targets;
features and predictions use current/past input only.

| Annotation record provenance | Records |
|---|---:|
| Original weak manifest evidence | 1,000 |
| Provisional single-review decisions | 21 |
| Unreviewed proposals | 19 |
| Provisional AI visual reviews | 20 |
| Independently verified precursor-positive videos | **0** |

The 1,060 records overlap in video/coverage and are not 1,060 independent
training examples. Original notes and decisions remain intact. The full status
and split breakdown is `data/foundation/ai_visual_v1/annotation_coverage.json`.

**Models and validation**

The baseline reuses the existing joint-only blocks. The advanced model reuses the
existing ST-GCN blocks for shared per-person joint/bone/backward-velocity streams,
adds body-relative normalization and symmetric interaction fusion, and uses
GroupNorm to avoid reliance on camera-specific BatchNorm running statistics.
It is invariant to swapping the two person slots. Augmentation, video-balanced
sampling, class weights and checkpoint selection use fitting/internal-development
sources only. Existing v1-v4 detectors and `c3_model.py` were not rebuilt or edited.

Both runs were capped at eight epochs with internal-development early stopping.
Baseline stopped at epoch 5 and selected epoch 2; advanced completed eight and
selected epoch 6. The advanced minimum development BCE was 0.699596 versus
baseline 0.710298. Thresholds use internal development only, with a predeclared
maximum sampled weak-label FPR of 0.10 and minimum recall/balanced-accuracy guards.
These guards are not population false-alarm guarantees.

| Same expanded validation contexts | Baseline | Advanced |
|---|---:|---:|
| Weak first-onset AUROC | 0.608 | 0.232 |
| Weak first-onset F1 at guarded threshold | 0.250 | 0.000 |
| Weak active-fight AUROC | 0.534 | 0.666 |
| Weak active-fight F1 at guarded threshold | 0.286 | 0.109 |

The advanced model improves active-fight ranking but fails the forecast research
guard. The engineering runtime default therefore falls back to the baseline,
whose forecast head passes that **internal** research guard; its active-fight
head remains disabled for alerts. Both candidates and the original minimum-BCE
selection record are preserved. Final fallback selection uses internal eligibility
and development BCE, not validation metrics. Historical validation exposure and
new-context-only metrics are disclosed in the evaluation JSON.

Do not compare the old 40.2% pilot agreement with these numbers as an accuracy
improvement: the dataset and target protocol changed. Neither expanded model has
adequate demonstrated operational early-warning performance.

**Final runtime quality and timing**

QA exposed that historical average track quality could remain high after people
disappeared. Runtime v2 additionally requires six **currently observed** joints in
at least one person, eight history frames and window quality >=0.45. The old
outputs remain intact. Missing current poses now cause abstention immediately.

Across the 16 selected validation contexts, v2 has 421 usable decisions out of
640 (65.8%). Opt-in research replay detects only **1 of 8** weak first-onset
contexts and emits **2** alerts across weak no-fight contexts. Unobservable event
contexts remain counted as undetected. This is not full-recording event recall,
verified precursor recall, false alarms/hour, calibration or test-set performance.
Operational alerts stay disabled by default because these results are insufficient.

A fresh two-second source extraction/replay/render smoke test completed; pose
extraction took 8.25 seconds on the current CPU runtime. Cached model replay is
much faster, but its timing excludes pose extraction. End-to-end real-time
performance on this machine is not claimed. No GPU drivers or shared virtual
environment packages were changed.

**Verification and artifacts**

The final full suite passed **164 tests plus 17 subtests** in 9.28 seconds after
the current-observation fix, including a regression case proving that a missing
current pose abstains despite valid history. Results are saved in
`outputs/advanced_v1/tests_final.xml`. No tests failed or were skipped.
Tests cover split grouping, weak-target policy, past-only features, person-order
invariance, temporal padding, finite gradients, operating-point ties, quality
abstention and end-to-end cached-replay prefix invariance. Actual video extraction,
checkpoint reload, rendering and validation were exercised separately.

The final new-video CLI cache-hit smoke test also passed: the two-second context
was reused with zero extraction time and a 0.296-second cache lookup. Its outputs
are in `outputs/final_c3_v2/cache_reuse_smoke/`.

Video files were decoded with FFmpeg; the dashboard JavaScript and PowerShell
launcher were syntax checked. Comparison plots and rendered video frames were
visually inspected. Browser interaction is unverified: the app browser's earlier
local-file preview was blocked by its URL policy, and no automated browser
workaround was attempted.

New source modules: `advanced_data.py`, `advanced_stgcn.py`,
`advanced_experiment.py`, `research_operating_point.py`, `predict_advanced.py`,
`evaluate_advanced_runtime.py`, and `build_c3_report.py`. New tests are in
`tests/test_advanced_pipeline.py`; the launcher is `RUN_C3_DEMO.ps1`.
The existing dataset, labels, v1-v4 source modules, previous outputs, and model
weights are preserved. Components 01, 02 and 04 were not changed. No commit/push
was performed. Data, caches, outputs and model weights remain ignored.

The earlier continuation also added `src/quick_pose_setup.py`,
`src/human_review.py`, `src/human_review_ui.html`, `src/ai_review_evidence.py`,
their three corresponding test files, and `FAST_START.md`, `HUMAN_REVIEW.md`,
`AI_VISUAL_REVIEW.md`, and `WEAK_FIGHT_PILOT.md`. The only currently modified
tracked file is C3's `.gitignore`, which adds exclusions for versioned review
exports. The original Phase 1 module/change inventory is preserved in
`PHASE1.md`. The exact current uncommitted file inventory and hashes are saved
in `outputs/advanced_v1/implementation_files.json`.

Final integrity checks matched all 170 files in the pre-stage preservation
snapshot, verified all 131 indexed cache hashes and the materialized-window
hash, and found zero cross-split groups. Reports are
`outputs/advanced_v1/preservation_final.json` and
`outputs/advanced_v1/integrity_final_v2.json`. The earlier integrity report's
role-summary omitted indexed validation groups; v2 includes all 16 and
supersedes that summary without replacing the original report.

Important local artifacts:

- `outputs/final_c3_v2/`: dashboard, four demos, score CSVs, runtime evaluation.
- `outputs/advanced_v1/`: alignment audit, both model evaluations, internal
  operating points, selection records, tests, provenance and preservation reports.
- `data/pose_cache/advanced_v1/`: 115-train/16-val indices and versioned window
  cache; old caches remain reusable.
- `weights/baseline_weak_onset_guarded_v1.pt`: default research runtime checkpoint.
- `weights/advanced_weak_onset_guarded_v1.pt`: advanced comparison checkpoint.

**Reproduce without overwriting**

To re-evaluate the final runtime into a new result directory:

```powershell
& $C3Python -m src.evaluate_advanced_runtime --out-dir outputs/runtime_recheck_v1
```

Retrain from the prepared windows using new checkpoint/report names:

```powershell
& $C3Python -m src.advanced_experiment train --variant baseline --epochs 8 --out weights/baseline_reproduction_v1.pt --report-dir outputs/reproduction_v1/baseline
& $C3Python -m src.advanced_experiment train --variant advanced --epochs 8 --out weights/advanced_reproduction_v1.pt --report-dir outputs/reproduction_v1/advanced
& $C3Python -m src.research_operating_point --checkpoint weights/advanced_reproduction_v1.pt --out weights/advanced_reproduction_guarded_v1.pt --report outputs/reproduction_v1/advanced_guarded/operating_point.json
& $C3Python -m src.advanced_experiment evaluate --out weights/advanced_reproduction_guarded_v1.pt --report-dir outputs/reproduction_v1/advanced_guarded
```

Recreate data selection/extraction and materialization in new versioned paths:

```powershell
& $C3Python -m src.advanced_data expand --out-dir data/pose_cache/advanced_reproduction_v1
& $C3Python -m src.advanced_data materialize --train-index data/pose_cache/advanced_reproduction_v1/train_index.json --val-index data/pose_cache/advanced_reproduction_v1/val_index.json --out-dir data/pose_cache/advanced_reproduction_v1/windows
```

Run all tests with the existing isolated dependencies:

```powershell
$C3Root = (Get-Location).Path
$env:PYTHONPATH = "$C3Root\data\phase1_work\testdeps;$C3Root;$C3Root\legacy"
$env:TEMP = (Resolve-Path 'data\phase1_work\tmp').Path
$env:TMP = $env:TEMP
& $C3Python -m pytest tests tests_optional legacy/test_c3_aggression.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_final_recheck --junitxml outputs/final_recheck/tests.xml
```

The remaining scientific blocker is reliable precursor/contact supervision and
adequate independent validation, especially during occlusion and active physical
contact. Larger model complexity, more epochs or full filesystem access cannot
substitute for those observations. Further model-quality claims require better
adjudicated data and prospectively frozen evaluation; this package does not
manufacture that evidence.
