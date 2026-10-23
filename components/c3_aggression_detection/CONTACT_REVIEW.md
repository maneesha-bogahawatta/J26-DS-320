**Visible-contact review workflow for Component 03**

The existing independent human-review workflow already preserves raw decisions,
scoped coverage, locked splits, matching reviewer agreement and immutable media.
The additive `src/contact_review.py` workflow reuses those review copies and adds
explicit semantics needed for a pre-contact research claim. No dataset, previous
human decision, provisional AI review, model or locked split is changed.

The audit of `data/foundation/ai_visual_v1/annotations.json` finds 1,060 overlapping
evidence records: 1,000 weak, 21 provisional single-review, 19 unreviewed and 20
provisional AI reviews. Strict independently verified precursor-positive and
non-aggressive video counts are zero in both development splits. Verified
early-warning training remains blocked; verified accuracy and recall are `null`.
See `outputs/development_completion_v1/annotation_contact_audit.json`.

**The three observations**

| Observation | Required evidence | What it supervises |
|---|---|---|
| First threatening cue | Explicit observed cue intervals and their first start | A candidate precursor, only when independently confirmed visible later contact is available |
| First physical contact | Visible first strike, forceful push, kick or grappling contact; explicit personal-observation and no-inferred-dataset-onset attestations | The visible contact reference for the pre-contact task |
| Ongoing physical violence | Explicit reviewed fight intervals | Active-fight recognition; this does not establish the unobserved first contact |

Contact definition is versioned as `visible_strike_push_kick_or_grapple_v1`.
Ordinary touching alone does not meet this operational definition. This is the
prototype's annotation protocol, not a legal determination of aggression.
If timing or visibility is uncertain, retain an earliest/latest bracket and choose
uncertain. A bracket does not become a confirmed contact timestamp. If the clip
starts during violence, mark already in progress and leave first contact blank.

All times use original-video seconds. Reviewed, cue and fight intervals must be
finite, non-overlapping, in bounds and coverage scoped. Cue intervals must end
strictly before contact, with some observed cue coverage in the last three seconds
before contact to supply this model's early target. The first-threatening-cue
marker is the first cue interval's start. The form shows the original clock above
the player; browser controls retain the relative clip clock.

The original UBI binary onset remains a **weak annotation boundary**. It must not
be assumed equal to first physical contact. Historical `review_annotations.py`
and approved-manifest normalization can derive contact from approved interval
onset; those artifacts and behavior are preserved. They do not count toward the
new stricter contact-semantics readiness report. AI/v3/v4 proposals remain
provisional; the new importer accepts only its human form schema.

**Prepared independent forms**

The prepared pages contain 12 training clips and eight separate validation clips,
reusing the existing silent media without pose extraction or new video export:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'

Start-Process '.\outputs\development_completion_v1\contact_review_pack\train\review.html'
Start-Process '.\outputs\development_completion_v1\contact_review_pack\val\review.html'
```

Each human starts blank and reviews independently without seeing another
reviewer's decisions. Enter your own reviewer ID and independent-review
attestation, watch the footage, scope coverage and export JSON. Resume only your
own file. The page stores no server-side decisions; download before closing.
Use genuine distinct human IDs; software cannot authenticate independence.

The importer requires exact semantic agreement, including contact definition,
observation state, first cue, coverage, cue/contact/fight timing and uncertainty.
Notes may differ and are preserved in each raw submission. A single review stays
provisional. Disagreements and uncertain decisions never produce a verified
precursor target. Keep every original export. Resolve disagreements through a
new explicit adjudication version; never edit another review to manufacture
agreement. Reviewed decisions can be appended in new versions; existing outputs
refuse changed-content replacement.

Save human exports under a new local folder, for example
`data/reviewed_annotations/completed_human_contact_v1/`. After real independent
reviews exist, import them into a new version:

```powershell
& $C3Python -m src.contact_review import --reviews data/reviewed_annotations/completed_human_contact_v1/train_A_contact.json data/reviewed_annotations/completed_human_contact_v1/train_B_contact.json data/reviewed_annotations/completed_human_contact_v1/val_A_contact.json data/reviewed_annotations/completed_human_contact_v1/val_B_contact.json --out-dir data/foundation/contact_human_v1

# Confirm compatible pose quality and eligible labeled windows before training:
& $C3Python -m src.train_stgcn --annotations data/foundation/contact_human_v1/annotations.json --train-index data/pose_cache/quick_v1/pilot_train_index.json --val-index data/pose_cache/quick_v1/pilot_val_index.json --check-only --report outputs/contact_human_v1/readiness.json
```

New imports preserve every record in the default parent
`data/foundation/ai_visual_v1/annotations.json`. To append a later review version,
pass `--annotations data/foundation/contact_human_v1/annotations.json` and choose
a new output directory. The pack binds media/source hashes and locked splits,
while the output records the chosen parent hash and each submitted form hash.

Strict annotation readiness requires positive and negative independently verified
videos in **both train and validation**. This is only a minimum label-presence
guard. Compatible eligible cache windows, independent event diversity, sufficient
negative recording hours and a predefined evaluation protocol are still required.
The existing training loader masks conflicting and unknown coverage. Readiness
does not establish scientific accuracy, calibration or operational suitability.

The review pack prohibits test sources. Test metadata already in parent evidence
is preserved but excluded from development readiness. Do not use test video
review, test labels or test results to select training labels or model settings.
Pack selection used retrospective weak onsets and visible source material may
reveal source class; this diagnostic pack is not a blinded unbiased benchmark.

**Reproduce and integrate**

```powershell
# New blank pages, using the same existing immutable media:
& $C3Python -m src.contact_review prepare --out-dir outputs/contact_review_reproduction_v1

# Read-only provenance/label audit into a new report name:
& $C3Python -m src.contact_review audit --out outputs/contact_review_reproduction_v1/annotation_audit.json

$C3TaskRoot = (Get-Location).Path
$env:PYTHONPATH = "$C3TaskRoot\data\phase1_work\testdeps;$C3TaskRoot;$C3TaskRoot\legacy"
& $C3Python -m pytest tests/test_contact_review.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_contact_review_next --junitxml outputs/contact_review_reproduction_v1/tests.xml
```

Library entry points `validate_contact_record`, `blank_contact_form`,
`import_contact_evidence` and `readiness_report` are available to Component 03
interfaces. CLI import additionally authenticates immutable media, the manifest
and parent provenance before writing a new normalized annotation version.
No shared group API or teammate source is changed.

The contact workflow has **15 tests plus 19 subtests**. The final combined
contact, existing human-review and foundation-review run passed **37 tests plus
29 subtests**. Both generated
page scripts passed `node --check`. Tests exercise exact consensus, raw-note and
parent preservation, disagreement, uncertain contact, no-onset inference,
strict timing, negative scoping, ongoing violence without early labels, test
exclusion, minimum readiness and refusal to overwrite changed outputs. This
verification does not substitute for a completed human review or browser visual
inspection. No human decisions were generated in this development step.
