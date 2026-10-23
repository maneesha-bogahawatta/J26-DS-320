# PP1 demonstration and presentation guidance

Present this as an implemented, measured research prototype. The engineering
demonstration is ready; independently validated 1–3-second pre-contact prediction
has not been demonstrated. No score is a calibrated aggression probability.

Suggested seven-slide narrative:

1. **Research problem:** assist a CCTV operator in spotting potentially escalating
   interactions. Distinguish a warning before observed contact from recognition
   of violence already underway. The human makes the incident decision.
2. **Architecture:** video → existing YOLO11-Pose/ByteTrack → causal trailing
   skeleton windows → baseline/advanced ST-GCN → quality abstention → research
   incident/evidence → operator review. Optional audio is separate and unvalidated.
3. **Data and integrity:** 1,000 UBI sources; locked 791/141/68 split; duplicates
   grouped; cached pose features reused. Weak annotation onset is not independently
   confirmed first physical contact. Keep the test set out of demonstration tuning.
4. **Three tasks:** weak annotated onset within three seconds, weak active-fight
   classification, and blocked verified cue/contact prediction. Show explicit
   human review forms and explain the need for two independent actual reviewers.
5. **Measured results:** use `MODEL_RELIABILITY.md`, `STGCN_DIAGNOSIS.md` and the
   final context evaluation. State cohort and threshold policy with every table.
   The runtime has 421/640 usable decisions, two of eight weak onset contexts with
   some pre-onset notification, but zero notification recall at leads ≥1, ≥2 and
   ≥3 seconds. Three notifications occur across selected weak no-fight contexts.
6. **Working demonstration:** upload/select → run → inspect video/trace/evidence →
   append operator decision → export. Explain that missing pose means abstention.
7. **Limits and next experiment:** actual independent cue/contact annotation,
   broader source representation, prospectively frozen evaluation and sufficient
   normal recording hours. Do not promise a scientifically unsupported accuracy.

Five-minute live demonstration:

- Start `RUN_C3_DEMO.ps1 -Interactive` and open `http://127.0.0.1:8765`.
- Select cached `f_147_1_0_1_0` to avoid CPU extraction delays. Run in the default
  scores-only mode. The selected context is validation material with prior
  exploratory exposure, not a test demonstration or blinded benchmark.
- Point out separate onset/fight curves, the first warmup/abstention period,
  quality coverage and source-video timestamps. A candidate record is research
  evidence even when research notifications are disabled.
- Open a candidate's measured evidence, seek to the frame and append a decision
  using an honest demonstration reviewer ID and note. This never becomes a
  precursor/contact training label.
- Export the run ZIP. Show the preserved static model comparison report.
- If time permits, show a normal context (`n_545_1_0_1_0`) and the upload picker.
  New-video extraction can exceed playback time; prepare a real consenting sample
  in advance. Do not represent a cached replay as end-to-end real-time inference.

If asked whether the model improved, distinguish engineering correctness from
scientific generalization. Historical quality-aligned baseline onset AUROC rose
from .591 to .687 on a changed matched cohort, but the threshold had 50% FPR and
wide source-bootstrap intervals; it was not promoted. Development-only score
temperature diagnostics do not consistently improve validation Brier. The
original guarded baseline/checkpoint/thresholds remain selected.

If the browser cannot start during PP1, use the saved annotated MP4, CSV and JSON
under `outputs/development_completion_v1/` and the original static dashboard.
Explain the fallback honestly. No automatic intervention or shared-team delivery
is demonstrated. Generic potential-aggression events need an agreed shared type;
only active-fight events currently have an exact `IncidentEvent` projection.
