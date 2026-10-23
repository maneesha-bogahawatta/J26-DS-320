# Component 03 completion audit — 21 October 2026

Audit performed before major changes against `PHASE1.md`, `FINAL_C3.md`, the later
`STGCN_DIAGNOSIS.md`, actual source, saved reports, local datasets and Git status.
Existing uncommitted work belongs to the user and is preserved. All development
in this stage stays within Component 03; there is no commit, push or shared edit.

| Requirement | Existing evidence | Gap / action in this stage |
|---|---|---|
| Dataset preparation | 1,000 locked UBI sources, 791/141/68 split; hash/source grouping | Preserve splits and original datasets; reject test identities in demonstration selection. |
| Pose extraction | YOLO11-Pose + ByteTrack, deterministic versioned NPZ caches | Reuse existing caches; add evidence references to inference without changing extraction. |
| Temporal preprocessing | Past-only joint filling, trailing windows, scene reset, current observation guard | Keep working preprocessing; regression checks for timestamps and abstention. |
| Model training | Baseline/advanced and RiskNet architectures; historical fits plus six later diagnosis fits | Do not retrain already investigated arms. Add development-only temperature/reliability diagnostics; preserve runtime selection. |
| Weak first-onset forecasting | Explicit next-three-second weak UBI onset target; scored separately | Complete 1/2/3-second context-level summaries and disclose deduplication effects. Weak onset is not confirmed physical contact. |
| Active-fight recognition | Separate model head and weak labels | Preserve separate scores and head eligibility; do not activate failed research gates. |
| Verified pre-contact prediction | Annotation provenance, conservative masked targets, human review packs | Zero independently verified positives. Add stricter explicit cue/contact observation workflow; training/accuracy remain blocked. |
| Evaluation | Exact development thresholds, original/quality-matched cohorts, source bootstrap | Add calibration diagnostics and explicit three-task/readiness report. Never tune on test. |
| Video inference | New video CLI, cache reuse, research scores, MP4/CSV/JSON, abstention | Add camera/time/track/frame evidence, standardized research incidents and transparent notification suppression. |
| Explainability | Pose/interaction heuristics; old RiskNet signal dictionaries | Add measured descriptive pose evidence, not causal attribution, with reliability and model provenance. |
| Dashboard | Static report, score plots, four rendered demos | Add local upload/selection, asynchronous inference, evidence/review, export; preserve original report. |
| Shared event integration | `shared.schemas.event_schema.IncidentEvent` and allowed C3 types exist | Add C3-owned adapter using this exact dataclass and existing event types; metadata stays in a C3 envelope. No invented group transport. |
| Testing/reproducibility | Latest saved diagnosis suite: 175 tests + 17 subtests; older final doc says 164 | Run fresh baseline/final suites, add meaningful event/server/review regressions, record hashes and commands. |
| Optional audio | YAMNet/FFT cues, RiskNet AV interface and missing-modality masking | Preserve as optional; no trained validated AV checkpoint or synchronized evaluation evidence exists. Do not attach audio to pose-only ST-GCN or claim fusion complete. |

The later diagnosis already found sampler/class-weight mismatch and stale-history
eligibility mismatch, tested fixes and documented uncertain/regressing results.
Changing score direction after seeing validation is unjustified. The historical
guarded baseline remains the runtime choice; scores remain uncalibrated and
operational notifications disabled. The small selected validation contexts and
their overlapping windows cannot establish population accuracy or false alarms/hour.

Planned additions extend existing code: incident adapter and evidence export,
strict cue/contact review, reliability diagnostics, context-level evaluation and
interactive demo server. None substitutes for independently adjudicated labels.
