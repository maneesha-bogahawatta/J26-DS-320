**Completed AI visual labeling of the 20-clip cached pilot**

The requested AI review is saved separately from every existing human decision.
Raw decisions, notes, observed timestamps and contact uncertainty are in
`data/reviewed_annotations/ai_visual_v1/labels.json`. The readable CSV is
`data/foundation/ai_visual_v1/ai_review_summary.csv`. The new normalized annotation
version is `data/foundation/ai_visual_v1/annotations.json`; all 1,040 parent
records are retained unchanged, followed by 20 `provisional_ai_review` records.

| Proposed label | Train | Validation | Total |
|---|---:|---:|---:|
| Precursor candidate | 4 | 0 | 4 |
| Active fight | 4 | 1 | 5 |
| Non-aggressive scoped interval | 1 | 3 | 4 |
| Uncertain | 3 | 4 | 7 |

These are **AI-provisional observations**, not independently verified human
labels, calibrated probabilities, or final ground truth. Every new record has
`eligible_for_fitting=false` and `requires_human_adjudication=true`. The verified
loader masks all of them. Their file schema differs from the human-review form;
the human importer rejects the AI file. No second human/model reviewer was
simulated and no training was run. A source filename, weak onset, or subsequent
fight was never enough by itself to assign a verified precursor.

The review inspected chronological contact sheets with 17 frames per eight-second
clip: 0.5-second spacing plus the last frame. Nine detail sheets examined selected
uncertain or onset regions; several were sampled every 0.1 seconds. This was
frame-sampled visual inspection, **not full-frame-rate playback or audio review**.
Frames between samples can contain missed events. Original weak-onset-based
selection and source IDs were available, so this is neither blinded annotation
nor an unbiased validation benchmark. Contact sheets and frame indices are under
`outputs/ai_visual_review_v1/`; raw decisions include their image hashes.

The four draft precursor proposals use original-video seconds:

| Clip | Proposed cue | Proposed contact | Contact uncertainty |
|---|---|---:|---|
| train_001 / F_103_1_2_0_0 | 3.7-5.9 s | 6.2 s | 6.0-6.4 s |
| train_004 / F_135_1_2_0_0 | 0.4-1.2 s | 1.5 s | 1.4-1.6 s |
| train_007 / F_155_0_0_1_0 | 3.3-3.7 s | 3.9 s | 3.8-4.0 s |
| train_012 / F_205_0_1_0_0 | 0.0-2.6 s | 3.3 s | 2.8-3.4 s |

These contact values are estimates of apparent striking/pushing contact. Earlier
light touching or grazing contact is ambiguous in some clips. Proposed cue ends
precede the earliest plausible contact bracket. In train_012, an attempted punch
appears around 2.9 s and clearer counter-contact around 3.3 s, hence the wider
bracket. Human adjudication must resolve the operational contact definition and
timing; the importer does not turn these estimates into human-confirmed events.

Seven clips remain uncertain because contact or relevant gestures cannot be
resolved: train_005, train_009, train_010, val_006, val_015, val_016 and val_017.
The shop entrance gesture and animated conversation do not establish precursors
without confirmed subsequent contact. Dense crowd motion is not automatically a
negative. The train_003 negative covers **only 12.0-14.5 seconds**; later occluded
time is not labeled negative. The street/vehicle negatives in val_019 and val_020
have no eligible pose windows under current quality guards.

No validation precursor candidate was confidently proposed in the available
eight-second contexts. Thus a future explicitly defined AI weak-supervision
experiment would still need suitable validation precursor examples. Independently
verified early supervision remains blocked by the original lack of human labels;
the AI review does not remove that blocker. It supplies reviewable draft evidence
and identifies uncertain contexts without manufacturing an early predictor.

**Exact Windows PowerShell commands**

The current import and check have already been run. To reproduce into a new output
version using the saved AI decisions, without pose extraction:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
& $C3Python -m src.ai_review_evidence --out-dir data/foundation/ai_visual_reproduction_v1
& $C3Python -m src.train_stgcn --annotations data/foundation/ai_visual_reproduction_v1/annotations.json --train-index data/pose_cache/quick_v1/pilot_train_index.json --val-index data/pose_cache/quick_v1/pilot_val_index.json --check-only --report outputs/ai_visual_reproduction_v1/readiness.json
```

The check reports the verified-supervision blocker and writes no checkpoint.
Future human reviews use the existing separate `HUMAN_REVIEW.md` workflow, with
real reviewer IDs and actual human decisions; do not enter an AI review under a
fabricated human identity.

Full regression tests use the existing isolated dependencies:

```powershell
$C3Root = (Get-Location).Path
$env:PYTHONPATH = "$C3Root\data\phase1_work\testdeps;$C3Root;$C3Root\legacy"
$env:TEMP = (Resolve-Path 'data\phase1_work\tmp').Path
$env:TMP = $env:TEMP
& $C3Python -m pytest tests tests_optional legacy/test_c3_aggression.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_ai_visual_next --junitxml outputs/ai_visual_next/tests.xml
```

Added source: `src/ai_review_evidence.py` and
`tests/test_ai_review_evidence.py`; this guide documents the result. The only
existing-file edit adds an ignore rule for local `ai_visual_*` review directories.
All detectors, original labels, caches and prior evaluation outputs are preserved.
No commit or push was performed.
