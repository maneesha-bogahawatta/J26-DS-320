**Next step: review the existing cached pilot**

The prepared pack is `data/reviewed_annotations/quick_human_v1/`. It contains
12 train and 8 validation clips, at most eight seconds each. Exporting these
silent browser-compatible review copies performs no YOLO or pose extraction.
The original recordings, cache archives, locked splits, reviews, and results
are preserved. No model has been trained and no human decisions were generated.

Open `train/review.html` in Edge or Chrome. A reviewer watches each clip, enters
their own ID, attests to personal independent review, and downloads their JSON.
Repeat separately with `val/review.html`. A second human starts each page blank
and reviews independently before seeing the first person's decisions. Save the
four downloaded files under a new directory such as
`data/reviewed_annotations/completed_human_v1/`, named `train_A.json`,
`train_B.json`, `val_A.json`, and `val_B.json`. The names A/B do not substitute for
real distinct reviewer IDs. Software cannot authenticate human independence.
Export before closing the page; resume only your own saved JSON using its file
picker. Each export preserves notes and decisions. Keep every original export.

Use the **original-video clock** above the player, not the player's relative
clock. Scope reviewed coverage to the interval actually assessed. For a positive
precursor, mark observable aggressive behavior before confirmed visible physical
contact; do not infer one from a fight annotation or eventual outcome. The current
early head supervises only cue timestamps within three seconds before contact.
Some very short source videos begin in violence or have too little preceding
context; choose already violent or uncertain as appropriate. Do not force a
positive class to satisfy training. Active-fight decisions need explicit fight
intervals. Non-aggressive decisions need aggression-free coverage and no asserted
cue/contact/fight inside that record. An individual video can have separately
reviewed positive and negative intervals in later versioned packs; a whole fight
video is never automatically a negative example outside its weak annotations.

The UI shows no heuristic alerts, annotated onset, or proposed class labels.
Selection used weak onsets retrospectively to find review context; filenames
and source material may still identify the source class. This is a diagnostic
pilot, not an unbiased validation benchmark. Validation is reviewed separately
and is never imported into the training split. No test source enters this pack.
Review copies are silent; judgments apply to visible evidence only.

The importer requires exact agreement on label, reviewed coverage, precursor
intervals, contact time, and fight intervals. Notes may differ and are retained.
One completed review stays provisional. Disagreement, uncertainty, or unreviewed
time produces no verified target. To resolve disagreements, retain the original
files and conduct an explicit separate adjudication in a new version; do not
change another person's independent observations to manufacture agreement.
The importer never averages boundaries or promotes heuristic proposals.

**Windows PowerShell commands**

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'

# The pack is already prepared. Open the training review page:
Start-Process '.\data\reviewed_annotations\quick_human_v1\train\review.html'
# Review validation independently of training:
Start-Process '.\data\reviewed_annotations\quick_human_v1\val\review.html'

# Only after the humans have completed and saved these four files:
& $C3Python -m src.human_review import --reviews data/reviewed_annotations/completed_human_v1/train_A.json data/reviewed_annotations/completed_human_v1/train_B.json data/reviewed_annotations/completed_human_v1/val_A.json data/reviewed_annotations/completed_human_v1/val_B.json --out-dir data/foundation/reviews_human_v2

# Check actual supervised coverage, cache quality, and the training blocker:
& $C3Python -m src.train_stgcn --annotations data/foundation/reviews_human_v2/annotations.json --train-index data/pose_cache/quick_v1/pilot_train_index.json --val-index data/pose_cache/quick_v1/pilot_val_index.json --check-only --report outputs/human_review_v2/readiness.json
```

If only one reviewer is available, import just that person's files; these remain
provisional. Do not enter a fabricated second ID. Readiness requires positive
and negative eligible early windows in both train and validation. Passing these
minimum execution guards does not establish adequate sample size, independent
event diversity, calibration, or deployment performance. If insufficient labels
remain, use the coverage report to request more human-reviewed contexts; only
then extract additional selected intervals as necessary.

After readiness reports `ready`, a research-only pilot fit can use:

```powershell
& $C3Python -m src.train_stgcn --annotations data/foundation/reviews_human_v2/annotations.json --train-index data/pose_cache/quick_v1/pilot_train_index.json --val-index data/pose_cache/quick_v1/pilot_val_index.json --epochs 10 --out weights/stgcn_human_pilot_v1.pt --report outputs/human_review_fit_v1/readiness.json
```

This is a tiny diagnostic experiment. It does not establish a trained and
validated operational early-aggression predictor. Do not select training labels,
thresholds, or model settings using the locked test identities.

To prepare a different new version without re-extracting poses:

```powershell
& $C3Python -m src.human_review prepare --out-dir data/reviewed_annotations/quick_human_v2
```

Requires the existing NumPy environment and local FFmpeg with libx264. No
packages, models, or videos are downloaded. Existing pack directories refuse
reuse. New imports write a new annotation file, retaining every parent record
and raw submitted review. Matching reruns are allowed; different existing output
content is never overwritten. Generated review media remain local dataset files
and must not be committed. No commit or push is part of these commands.

**Verification of this prepared version**

The full suite passed 146 tests plus 10 subtests using the existing isolated test
dependencies. Both generated pages passed JavaScript syntax checks; all 20 H.264
clips decoded successfully and matched their eight-second ranges. Browser visual
verification was blocked by the app browser's policy against `file://` URLs;
no alternate browser route was attempted. Open the local pages yourself using
the PowerShell commands above.

Importing the blank templates preserved all 1,040 parent evidence records and
added only 20 unreviewed records in a separate smoke-test version,
`data/foundation/human_template_check_v1`. Actual loader readiness still reports
zero supervised early windows and writes no checkpoint. The 127 pre-existing
source/review/result files in the preservation snapshot and 107 recovered cache
hashes remained unchanged. This change only adds the human-review source,
HTML template, tests, guide, generated pack/reports, and two local ignore rules.

Quality checks found usable windows inside 16 of 20 review contexts: 411 train
and 164 validation decision windows meet the current frame/quality guards.
These are overlapping unlabeled windows, not independent training examples.
`F_195_1_0_1_0`, `N_312_0_0_1_0`, `N_553_1_0_1_0`, and `N_707_1_0_1_0`
have no eligible windows inside their review clips under current settings.
Their human observations can still document data/pose issues, but cannot supply
training targets with this cache configuration. Do not lower quality guards just
to force readiness. Detailed reports are in `outputs/human_review_v1/`.

Reproduce the full suite from the component directory:

```powershell
$C3Root = (Get-Location).Path
$env:PYTHONPATH = "$C3Root\data\phase1_work\testdeps;$C3Root;$C3Root\legacy"
$env:TEMP = (Resolve-Path 'data\phase1_work\tmp').Path
$env:TMP = $env:TEMP
& $C3Python -m pytest tests tests_optional legacy/test_c3_aggression.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_human_next --junitxml outputs/human_review_next/tests.xml
```

The legacy path is required by the unchanged historical `from run import *`
test. An initial collection attempt without it failed; the corrected complete
result is `outputs/human_review_v1/tests_complete.xml`. Earlier unit-test fixture
failures were corrected before exporting the pack; those initial reports remain
preserved alongside the passing reports.
