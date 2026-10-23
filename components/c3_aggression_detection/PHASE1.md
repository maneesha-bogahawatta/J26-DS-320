**Phase 1: locked data, reviewed-label provenance, and a causal joint-only ST-GCN interface**

This extension continues the existing C3 implementation. The v1–v4 detectors, `c3_model.py`, legacy imputer, original datasets, original reviews, and previous evaluation outputs are preserved. No other component is modified. No real-data predictor has been trained.

**Current dataset state**

`data/foundation/v1/ubi_locked.csv` locks all 1,000 UBI sources into 791 training, 141 validation, and 68 test videos. All 67 identities in the existing `test_videos.csv` remain test identities. One extra video, `N_770_0_0_1_0.mp4`, is held out because its bytes are identical to reserved test video `N_527_0_0_1_0.mp4`. Its promotion prevents leakage; no reserved identity was removed.

The five test-list identities in the old training-review batch are:

- `F_0_1_0_0_0.mp4`
- `F_120_1_0_0_0.mp4`
- `F_38_1_2_0_0.mp4`
- `F_41_0_0_0_0.mp4`
- `F_95_0_0_0_0.mp4`

They are retained as held-out evidence and excluded from the new training-review queue. Prior exploratory detector scans and reviews of test footage must be disclosed; this is not an untouched historical benchmark.

Split locking uses full SHA-256 hashes, existing `group`/`source_group` metadata, optional explicit source groups, and conservative whole-video DCT fingerprints. The audit found 27 exact duplicate pairs and 39 sampled perceptual matching pairs, including exact duplicates and transitive matches. All members of each connected group share a split. Assignment is stratified by whether a group contains annotated fight intervals, ordered by a seed-derived hash; validation receives approximately 15% of unreserved groups. Existing named assignments are respected; a conflict with reserved test membership fails rather than silently reassigning it. All non-example manifests were inspected for existing assignments.

This does not prove the absence of arbitrary cropped, shifted, or partially shared footage. Add known relationships using a CSV with `video,source_group` columns and `--groups`, then build a new split version. Same-scene metadata must represent a verified source relationship, not an assumption based on a filename suffix.

`data/foundation/v1/annotations.json` preserves 1,040 source records: 1,000 weak manifest records, 21 provisional review records, and 19 unreviewed records. Provisional records comprise 12 uncertain assessments, one potential missed precursor, two already-violent assessments, and six non-aggressive controls. Original dictionaries, notes, review decisions, and source locations remain attached. Multiple records may concern the same video.

There are **zero independently verified precursor-positive videos** in train, validation, or test. Existing provisional review coverage totals approximately 78.10 seconds where explicit bounds are available; missing ranges are not invented from prose. The six assessed controls remain clip-scoped and do not become verified negative full videos.

**Modules and contracts**

| Module | Role |
|---|---|
| `src/foundation_io.py` | Stable IDs, content hashes, deterministic artifact writers; refuses to replace different existing content. |
| `src/lock_splits.py` | Source/duplicate grouping, fixed splits, manifest integrity checks, overlap and companion reports. |
| `src/reconcile_reviews.py` | Normalizes manifest, manual JSON, v4 batch, and control evidence without changing originals. |
| `src/causal_pose.py` | Causal same-track joint filling, confidence decay, EMA, masks, quality, and RiskNet compatibility adapter. |
| `src/pose_sequences.py` | Label-independent YOLO11-Pose + ByteTrack extraction, scene resets, versioned NPZ caches. |
| `src/pose_dataset.py` | Trailing windows, fixed person slots, padding, masks, coverage-scoped labels, batching, split guards. |
| `src/stgcn_baseline.py` | `JointOnlySTGCN`, reusing the existing `STGCNStream` and `build_adj`. |
| `src/train_stgcn.py` | Readiness checks and a guarded train/validation loop. Never loads test data. |

The raw cache stores per-frame time/index/segment and ragged detection offsets, normalized and raw COCO-17 joints, raw confidence, observed/imputed masks, boxes, IDs, and five quality indicators. Coordinates use `x/width-0.5, y/height-0.5`; unknown joints stay zero and masked. Missing joints may carry their last same-track observation for at most 0.25 seconds, with confidence decay. Filling never renews the last-observed timestamp. A scene cut or expired identity resets the state. Normalization is per frame and does not use later poses.

The first configuration targets 15 Hz with 48-frame trailing windows, decisions every 0.2 seconds, at least eight decoded frames, and mean selected-track quality of 0.45. Actual source timestamps remain available. Windows do not cross scene cuts or sampled gaps over 0.35 seconds. Two person IDs are selected by accumulated box area within the available window with deterministic ID tie-breaking. Person slots remain fixed within that window; bystander selection is a documented limitation of this initial baseline.

Dataset skeletons have shape `[B,T,17,2,3]`. Masks include `[B,T]` decoded-time validity and `[B,T,17,2]` usable-joint validity, plus observed, imputed, and person masks. `collate_sequences` rejects mixed splits. The model converts to the existing two-person 34-node graph, removes temporal padding before normalization/convolution/pooling, and zeros masked joints. Its two outputs are precursor and active-fight logits, accompanied by `valid_window`. An empty/unsupported window must not be treated as a usable prediction.

The graph classifies the **whole trailing window at its cutoff**. Symmetric convolutions operate within that already available window; this is not a claim of framewise causal hidden states. It is the local C3 graph variant, not an exact reproduction of the original ST-GCN paper.

No heuristic output contributes a verified target. Verified precursor supervision requires a reviewed cue interval, a confirmed contact time, explicit reviewed coverage, and two distinct reviewer IDs with an approving label. Review IDs are provenance fields, not proof of independent human work. Unknown, uncertain, conflicting, and out-of-coverage time is masked rather than background. The existing finalizer's verified-manifest schema and the v4 review schema are both accepted. Timestamps must be in original-video seconds, not relative to an exported clip.

The optional `weak-fight` policy uses the imported binary annotation to supervise **active fight only**. It never supplies precursor targets. The default early task rejects weak-fight supervision. Training requires eligible positive and negative samples in both train and validation, and compatible cache configurations. Checkpoints record which head was trained and remain uncalibrated; no operational early-prediction claim is made automatically.

**Existing paths updated**

`src/features.py` and `src/live.py` use the causal compatibility adapter. The original `legacy/run.py` remains unchanged. New RiskNet caches and checkpoints carry `c3_causal_pose_v1`; training, prediction, and learned-model event evaluation reject legacy/incompatible preprocessing. Legacy reference experiments can still use the preserved legacy code. The feature extractor also retains decoded dimensions when its final read reaches EOF.

The RiskNet trainer now requires a locked manifest, loads only train/validation features during fitting, and leaves test evaluation to a separate invocation. Existing checkpoints and cache indexes are protected against overwrite. These changes mean old README commands that bypass split locking must be updated to use the Phase 1 contract before a new training run.

**Windows PowerShell setup and verification**

Run from the actual repository component. Test dependencies are isolated under C3, leaving the team's `.venv` unchanged:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$componentRoot = (Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force -Path '.\data\phase1_work\tmp' | Out-Null
$env:TEMP = (Resolve-Path '.\data\phase1_work\tmp').Path
$env:TMP = $env:TEMP
& $C3Python -m pip install --no-cache-dir --target '.\data\phase1_work\testdeps' -r '.\requirements-phase1-test.txt'
$env:PYTHONPATH = "$componentRoot\data\phase1_work\testdeps;$componentRoot;$componentRoot\legacy"
$env:YOLO_CONFIG_DIR = "$componentRoot\data\phase1_work\ultralytics"
$env:YOLO_AUTOINSTALL = 'false'
& $C3Python -m src.lock_splits --validate-only
& $C3Python -m pytest tests tests_optional legacy/test_c3_aggression.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_next --junitxml outputs/phase1_next/tests.xml
```

The `legacy` path is needed by the preserved legacy test's historical `from run import *`. Initially pytest/pandas/scipy were missing. They were installed only into `data/phase1_work/testdeps`. An initial full-suite collection attempt without the legacy path failed; that failure is retained in `outputs/phase1_foundation/tests_initial.xml`. Subsequent full-suite runs include the legacy and optional model tests. Synthetic training tests write only temporary fixture checkpoints.

To reproduce split locking and reconciliation into a fresh version without replacing the prepared v1 artifacts:

```powershell
& $C3Python -m src.lock_splits --out-dir data/foundation/reproduction_v1 --report-dir outputs/phase1_reproduction
& $C3Python -m src.reconcile_reviews --locked data/foundation/reproduction_v1/ubi_locked.csv --out-dir data/foundation/reproduction_v1 --report-dir outputs/phase1_reproduction
```

The duplicate audit reads all source bytes and samples video frames. A validation-only check is much faster. To incorporate known shared sources, append `--groups data/reviewed_annotations/source_groups.csv` to split creation and use a new directory/version. Unknown IDs and conflicting groups/assignments fail validation.

To reproduce the tested two-second pilot, use a fresh cache index and report name:

```powershell
& $C3Python -m src.pose_sequences --locked data/foundation/v1/ubi_locked.csv --video-id F_100_1_2_0_0 --end 2 --out-dir data/pose_cache/phase1_next --index data/pose_cache/phase1_next/pilot_train_index.json
& $C3Python -m src.train_stgcn --train-index data/pose_cache/phase1_next/pilot_train_index.json --check-only --report outputs/phase1_next/readiness.json
```

The pilot yields 30 sampled frames, 95 tracked detections, 1,320 observed joints, and 77 carried joints. Repeated CPU extraction was bitwise equal for the cached arrays. It supplies ten trailing decision windows and zero supervised early targets. `--check-only` reports this blocker without training or writing a checkpoint. A normal training invocation exits with an explanatory blocker when supervision is absent.

For full **training/validation cache preparation**, keep the locked v1 assignments and select only development splits:

```powershell
& $C3Python -m src.pose_sequences --split train --limit 0 --out-dir data/pose_cache/full_v1 --index data/pose_cache/full_v1/train_index.json
& $C3Python -m src.pose_sequences --split val --limit 0 --out-dir data/pose_cache/full_v1 --index data/pose_cache/full_v1/val_index.json
& $C3Python -m src.train_stgcn --train-index data/pose_cache/full_v1/train_index.json --val-index data/pose_cache/full_v1/val_index.json --check-only --report outputs/phase1_full/readiness.json
```

Full extraction was not run in this phase and can be expensive on CPU. Long recordings may need range-based pilot extraction while planning storage; partial caches do not constitute full-recording evaluation data. Cache keys include source/weight/tracker hashes, source range, seed, implementation hashes, library versions, and configuration. Existing matching caches are validated and reused. Existing differing artifacts and indexes are never overwritten; choose new output names.

**Next supervised stage**

Use `data/foundation/v1/training_review_queue.csv` for train-only label work. Conduct validation review separately using the locked validation split. Save new independent review forms under a new filename with the v4 review CSV columns; leave all existing source reviews intact. Explicitly fill original-video cue/contact times and reviewer provenance. Keep ambiguous cases uncertain. Do not use held-out test evidence to choose training labels, model settings, or thresholds.

After independently reviewed train and validation examples exist, reconcile into a new annotation version. For example, with a genuinely completed `independent_reviews.csv`:

```powershell
& $C3Python -m src.reconcile_reviews --review-batch data/reviewed_annotations/independent_reviews.csv --out-dir data/foundation/reviews_v2 --report-dir outputs/phase1_reviews_v2
& $C3Python -m src.train_stgcn --annotations data/foundation/reviews_v2/annotations.json --train-index data/pose_cache/full_v1/train_index.json --val-index data/pose_cache/full_v1/val_index.json --check-only --report outputs/stgcn_v1/readiness.json
& $C3Python -m src.train_stgcn --annotations data/foundation/reviews_v2/annotations.json --train-index data/pose_cache/full_v1/train_index.json --val-index data/pose_cache/full_v1/val_index.json --epochs 10 --out weights/stgcn_joint_v1.pt --report outputs/stgcn_v1/readiness.json
```

The final command remains blocked until suitable positive and negative windows exist. Only after that should the next stage implement baseline checkpoint inference, validation-only event-threshold calibration, and a separate locked-test event evaluation adapter using the existing metric functions. No baseline event-performance result is supplied by Phase 1.

Prepared reports are in `outputs/phase1_foundation`: split validation, test companion explanation, annotation coverage, reproducibility, training readiness, and JUnit test results. All generated data, caches, test dependencies, outputs, and weights are excluded by C3's ignore rules. No commit or push is part of this workflow.

**Completed validation and file inventory**

The final full suite passed **127 tests**: the 92 existing tests, including optional and legacy tests, plus 35 tests across six new files. No final test failures or missing test dependencies remain in the isolated test setup. `outputs/phase1_foundation/tests_complete.xml` is the final JUnit result. `pipeline_current.json` records a real-video batch of `[2,48,17,2,3]`, finite `[2,2]` model logits from random initialization, bitwise repeatability, and zero test identities in the 149-row training-review queue. `training_readiness_current.json` records the supervised-training blocker. `preservation.json` checks 84 pre-existing source/review/result/weight files and records no unexpected changes.

New source modules: `foundation_io.py`, `lock_splits.py`, `reconcile_reviews.py`, `causal_pose.py`, `pose_sequences.py`, `pose_dataset.py`, `stgcn_baseline.py`, and `train_stgcn.py`, all under `src/`.

New test files under `tests/`: `test_foundation_splits.py`, `test_foundation_reviews.py`, `test_causal_pose.py`, `test_pose_dataset.py`, `test_causal_compatibility.py`, and `test_stgcn_training_guards.py`. New documentation/setup files: this `PHASE1.md` and `requirements-phase1-test.txt`.

Modified existing files: `.gitignore`, `README.md`, `src/features.py`, `src/cache_dataset.py`, `src/live.py`, `src/train.py`, `src/predict.py`, and `src/evaluate_events.py`. These changes connect causal preprocessing, enforce locked fitting, protect artifacts, and describe the new workflow. No existing detector, graph block, dataset, review decision, or previous result was replaced.

The five generated foundation artifacts are `ubi_locked.csv`, `split_metadata.json`, `video_fingerprints.json`, `annotations.json`, and `training_review_queue.csv`, under `data/foundation/v1/`. Generated caches, reports, library settings, and isolated test dependencies remain local and ignored. Counts retain duplicate source files; future event evaluation should also report independent source-group counts.
