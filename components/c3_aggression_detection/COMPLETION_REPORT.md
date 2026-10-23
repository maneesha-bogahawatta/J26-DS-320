# Component 03 development completion — 21 October 2026

The remaining independently achievable engineering work is implemented inside
Component 03. This is a working university research demonstration; verified
pre-contact prediction, operational deployment and validated audio fusion remain
blocked. The initial audit is in `IMPLEMENTATION_GAPS.md`. Earlier experiments,
models, labels, reviews and static dashboard are preserved.

## Completed requirements

| Task | Result |
|---|---|
| Current implementation audit | Existing source, Git state, PHASE1, FINAL_C3, later STGCN diagnosis and artifacts inspected before major changes. No architecture rebuild. |
| Model reliability | Reused six existing diagnostic fits and predictions; added masked source-role integrity, Brier/NLL/ECE/reliability-bin diagnostics and development-only positive-temperature analysis. No validation-driven model promotion. |
| Three research tasks | Weak first-onset, weak active fight and blocked verified pre-contact evaluation reported separately. 1/2/3-second context metrics retain unobservable cases. |
| Safe annotation workflow | Explicit first threatening cue, personally observed defined first physical contact, uncertainty and ongoing violence; exact two-human agreement, append-only provenance, test exclusion. Twenty existing review clips reused. |
| Video inference | New prerecorded files and indexed caches, existing detector/tracker/ST-GCN, camera/source timestamps, current-pose abstention, annotated MP4, scores, evidence and incidents. |
| Correctness fixes | Reject wrong-source cache metadata before model construction/scoring. Reset confirmation when visible tracked identities change. Separate notification cooldowns by kind/identity and retain pending/suppressed candidates. |
| Explainable incidents | Stable IDs, C3/camera/time, kind, uncalibrated score/reliability, tracks, measured signals, source-frame/video references, model/hash and operator status. Evidence describes observations, not intent or causal model attribution. |
| Shared contract | Loads the existing unmodified `IncidentEvent` dataclass. Active-fight projection uses existing `fighting`. Generic forecast has no exact shared type and is explicitly blocked from projection until team agreement. No new transport/API invented. |
| Interactive demonstration | Local upload/select, async serial inference, score/video/evidence display, append-only operator status review, export ZIP, byte-range video serving, loopback/same-origin protections. Original static report retained. |
| Optional audio | Existing YAMNet/FFT/RiskNet interface audited and documented. Pose-only workbench remains supported; no unvalidated AV branch is advertised. |
| Tests/reproducibility | Baseline and final suites, meaningful regressions, actual video/cache replay, actual browser demonstration, split/cache/preservation checks and exact commands retained. |

## Files created and modified

Existing files edited in this stage:

- `README.md`: authoritative current entry points, separated historical workflow.
- `RUN_C3_DEMO.ps1`: additive `-Interactive` mode, existing modes preserved.
- `src/predict_advanced.py`: provenance/evidence/incidents, identity-aware
  notification state, cache preflight, scoped outputs and recording-time option.
- `src/evaluate_advanced_runtime.py`: source preflight and explicit v3 policy.

New source files: `incident_events.py`, `evaluate_warning_contexts.py`,
`verify_completion.py`, `reliability_audit.py`, `contact_review.py`,
`contact_review_ui.html`, `demo_server.py`, `demo_ui.html` (all under `src/`).
New tests: `test_incident_events.py`, `test_warning_contexts.py`,
`test_reliability_audit.py`, `test_contact_review.py`, `test_demo_server.py`.
New documents: this report, `IMPLEMENTATION_GAPS.md`, `MODEL_RELIABILITY.md`,
`CONTACT_REVIEW.md`, `AUDIO_INTERFACE.md`, `DEMO_SERVER.md`, `PP1_GUIDE.md`.
Reports, blank forms and demonstration runs are versioned under
`outputs/development_completion_v1/`; workbench state is in its own selected
`outputs/` directory. User work already uncommitted before this stage remains
uncommitted. The existing `.gitignore` change belongs to that prior state.

## Model findings and measured results

No new model was promoted or existing checkpoint overwritten. The runtime still
uses `weights/baseline_weak_onset_guarded_v1.pt`, SHA256
`954559460c10b85c25b532b5f7aa7d28edb141f38b308ebaeb199aca6c4d7820`.
Its active-fight notification head remains disabled by its existing research gate.

The later `STGCN_DIAGNOSIS.md` already established no global score-direction bug,
sampler/class-weight inconsistency, current-pose eligibility mismatch, weak
source/timing generalization and uncertain results from six corrective fits.
These were reused, not repeated. In the separate matched quality cohort, a
baseline candidate reached onset AUROC .687 and F1 .544, but with 50% FPR and
wide source-bootstrap intervals; this did not justify deployment or changing
the runtime default. See that document for unchanged-cohort and regression rows.

New development-only diagnostic temperatures produce the following validation
Brier scores (lower is better):

| Existing model | Weak onset raw → diagnostic | Weak active raw → diagnostic |
|---|---|---|
| Baseline | .220845 → .227378 | .254121 → .240943 |
| Advanced | .240421 → .242363 | .229811 → .248810 |

Calibration does not generalize consistently. No diagnostic transform is applied
at runtime. See `MODEL_RELIABILITY.md` for masks, source groups and exact inputs.

The final runtime policy was replayed on the same 16 indexed validation contexts,
without extraction or threshold/model selection. `context_evaluation_final/`
is authoritative for the completed code; earlier reports are retained:

- 640 decisions, 421 usable (65.8%); missing poses remain abstentions.
- Raw observable-window weak onset AUROC .5793, F1 .2479, precision .5172,
  recall .1630 (315 supervised decisions, 92 positive).
- Raw observable-window weak active-fight AUROC .4782, F1 .2258, precision .3333,
  recall .1707 (412 supervised decisions, 82 positive). Notifications disabled
  for this head; a raw threshold metric is not permission to activate it.
- Two of eight weak onset contexts have some pre-onset research notification,
  but notification recall at leads ≥1, ≥2 and ≥3 seconds is **0/8 each**.
  Confirmed candidate-state recall at ≥1/≥2 seconds is 1/8 and ≥3 seconds 0/8.
- Three research notifications occur across selected weak no-fight contexts.
  No population false-alerts/hour is calculated from retrospectively selected
  short clips. Fifty-six identity-reset decisions and every suppression reason
  are retained. One candidate occurs about 2.97 seconds before weak onset while
  an in-horizon notification occurs only about .17 seconds before it; the
  candidate/notification distinction must not be hidden.

The identity-aware policy changes notification counts/times, not raw model
scores. The prior report had one of eight contexts and two no-fight alerts;
the new two-of-eight/three-no-fight result is not a scientific improvement.
All metrics concern weak annotation boundaries, not verified physical contact.
External validation was historically inspected and remains exploratory. No test
media was used for fitting, threshold selection or these diagnostic evaluations.

## Validation and preservation

Fresh baseline: **175 tests + 17 subtests passed**, 29.89 seconds.
Authoritative release suite, including the final dashboard status projection:
**220 tests + 52 subtests passed**, zero failed/skipped, 23.42 seconds,
`outputs/development_completion_v1/tests_release.xml`. Earlier full passing
runs are retained. The final suite and browser result are additionally recorded
in `completion.json`.
One newly added cooldown regression initially failed on floating-point boundary
rounding; the comparison tolerance was corrected, and subsequent suites pass.
The initial failure report is retained rather than replaced.

Coverage includes causal prefix invariance, input shapes/masks, finite gradients,
missing current pose, correct cache identity before model construction, swapped
person handling, notification escalation/pending/identity changes, actual shared
dataclass roundtrip, strict cue/contact agreement and provenance, HTTP upload/
origin/path/range/job/review/export behavior and unchanged source artifacts.

An actual new-video CLI run reused the existing two-second pose cache: seven of
ten decisions usable, no notification, zero extraction time, .469-second cache
lookup, .047-second model replay, annotated MP4/CSV/JSON/incident exports.
This exercises the file path; a previous real YOLO/ByteTrack extraction smoke
is preserved. Re-extracting identical media was unnecessary. The eight-second
indexed fight context also replays/renders with the current checkpoint.

Browser QA exercises actual local upload, model subprocess, playable annotated
video, score chart, evidence seeking, append-only operator decision and export.
The final default-mode job `b3b17835d2274ab29ac2a1eb801202e5` reuses its cached
eight-second context: 37/40 usable decisions, two research candidates, zero
notifications and zero extraction time. The API/UI projects the latest operator
status while preserving the original incident file. The exact summary is
`dashboard_smoke/qa_summary.json`, screenshot `dashboard_review_final.jpg` and
verified export `final_reviewed_run_export.zip`. Earlier runs/policies remain
intact. Browser warnings/errors were empty. QA operator decisions are explicitly
demonstration records, ineligible for training. Browser tab closed and server
8766 stopped after verification; the normal launch default remains 8765.

`integrity_final.json` verifies all 575 protected pre-stage files: 571 unchanged
and exactly the four intended edits above. No original labels, dataset, model,
old research experiment or teammate source changes. All 131 indexed cache hashes
and identities pass; split counts remain 791/141/68 with zero cross-split groups.
The 6,915 original dataset files retain the same path/size/mtime inventory. Raw
media was not fully rehashed; this limit is explicit. No commit/push/PR occurred.

## Demonstration commands

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'

# Interactive upload/run/review/export workbench, loopback only:
.\RUN_C3_DEMO.ps1 -Interactive
# Equivalent if script execution is restricted:
& $C3Python -m src.demo_server --port 8765
# Open http://127.0.0.1:8765 and use Ctrl+C to stop the server.

# Preserved static research report:
.\RUN_C3_DEMO.ps1 -Serve

# New prerecorded video (creates a fresh versioned result folder):
.\RUN_C3_DEMO.ps1 -Video 'C:\path\your_video.mp4' -Seconds 8

# Cached demonstration, choose a NEW output name on each run:
& $C3Python -m src.predict_advanced --video-id f_147_1_0_1_0 --camera-id c3_pp1 --render --out-dir outputs/pp1_replay_new_v1
# --research-alerts opts into supported research notifications only.

# Strict cue/contact audit and blank forms, reuse existing media:
& $C3Python -m src.contact_review prepare --out-dir outputs/contact_review_new_v1
& $C3Python -m src.contact_review audit --out outputs/contact_review_new_v1/audit.json

# Diagnostic reproduction, no new fitting or extraction:
& $C3Python -m src.reliability_audit --out-dir outputs/reliability_new_v1
& $C3Python -m src.evaluate_warning_contexts --out-dir outputs/context_evaluation_new_v1
& $C3Python -m src.verify_completion --out outputs/preservation_recheck_new_v1.json

# Full suite using the existing isolated dependency set:
$C3TaskRoot = (Get-Location).Path
$env:PYTHONPATH = "$C3TaskRoot\data\phase1_work\testdeps;$C3TaskRoot;$C3TaskRoot\legacy"
$env:TEMP = (Resolve-Path 'data\phase1_work\tmp').Path
$env:TMP = $env:TEMP
& $C3Python -m pytest tests tests_optional legacy/test_c3_aggression.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_pp1_new_v1 --junitxml outputs/pp1_checks_new_v1/tests.xml
```

The launcher changes directory to C3. Versioned writers refuse different content
at an existing output path. Uploads and results stay local. No dependencies,
GPU drivers or shared virtual-environment packages were changed in this stage.

## Completion checklist, blockers and PP1 readiness

- [x] Reuse existing data, caches, architectures, checkpoints and later diagnosis.
- [x] Three-task reporting, diagnostic calibration and transparent notification effects.
- [x] New-video inference, descriptive evidence, timestamps and quality abstention.
- [x] Interactive video selection/upload/run, operator review and result export.
- [x] Explicit independent cue/contact review forms and blocked readiness.
- [x] Existing shared contract projection without shared-code changes.
- [x] Tests, real video/browser checks, preservation and exact Windows commands.
- [ ] Actual independent verified precursor/contact positive and negative labels.
- [ ] Frozen prospective evaluation with enough independent events/normal hours.
- [ ] Reliable demonstrated 1–3-second pre-contact model and calibrated scores.
- [ ] Team agreement for a generic potential-aggression shared event type and
  actual shared transport/scheduler integration.
- [ ] Trained synchronized AV comparison and end-to-end real-time validation.

PP1 readiness: suitable to demonstrate engineering progress and defend the
research design and measured limitations. It is not ready to claim reliable
pre-contact prediction or deploy autonomous safety actions. See `PP1_GUIDE.md`
for a seven-slide narrative and five-minute demonstration.

Suggested commit message, after the user's approval:
`[C3] Complete local research demo, incident evidence and cue-contact review`
