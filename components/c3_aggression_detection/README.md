# Component 03 — Early aggression research prototype

## Team onboarding layout

Component 03 lives in the guide's assigned folder,
`components/c3_aggression_detection/`. All C3 model, data preparation,
evaluation and research artifacts stay within this component:

```text
components/c3_aggression_detection/
├── __init__.py             # Component package, like the other team components
├── run.py                  # Team-guide --source entry point
├── src/                    # Data preparation, pose extraction, models, evaluation
├── c3_model.py             # Original C3 model used by existing modules
├── tests/                  # C3 regression tests
├── tests_optional/         # Optional model/dependency smoke checks
├── data/                   # Local manifests, annotations and pose caches
├── weights/                # Local trained checkpoints (Git-ignored)
├── outputs/                # Local experiments, inference jobs and audit history
├── output/pdf/             # Generated research logbook and daily diary
├── tmp/                    # Local document previews and temporary files
├── legacy/                 # Earlier implementation kept for provenance
├── requirements*.txt       # C3 dependency sets
├── RUN_C3_DASHBOARD.ps1     # Launch the shared team dashboard with C3
├── RUN_C3_DEMO.ps1          # Standalone research tools
└── *.md                    # C3 research and integration documentation
```

From the repository root with the project's Python environment active:

```powershell
python components/c3_aggression_detection/run.py --source "C:\clips\video.mp4" --seconds 8
```

`--source` is resolved from the calling terminal. Results default to a new
timestamped folder in C3 `outputs/`; a relative `--out-dir` is relative to this
component. The entry point calls the existing pipeline, checkpoint selection,
quality guards and split checks. It accepts recorded files, not live cameras.

The common UI and API remain in `dashboard/frontend/` and `dashboard/backend/`.
There is one Recorded video upload for the team; it includes C3 temporal inference
when C3 is enabled. The separate C3 research tab and home shortcut have been removed.
The guide does not prescribe an internal C3 layout; existing data/cache and model
paths are retained so experiment provenance and inference remain valid.

## Current status

Latest test and model comparison: [MODEL_IMPROVEMENT.md](MODEL_IMPROVEMENT.md).
The 32-source training expansion and new motion/interaction ensemble improve
exploratory ranking, but guarded active-fight F1 regresses. The new candidate is
retained for research comparison; the supported runtime default remains unchanged.
Run a candidate explicitly with `run.py --checkpoint` for comparison.

The original team React dashboard at `dashboard/frontend/`, restored from
`origin/develop`, is integrated through `dashboard/backend/` and the C3 research API.
See [DASHBOARD_INTEGRATION.md](DASHBOARD_INTEGRATION.md). Launch it with
`RUN_C3_DASHBOARD.ps1` (default `http://127.0.0.1:8000`); the earlier C3-only
workbench and static report remain available through `RUN_C3_DEMO.ps1`.

Current continuation: [COMPLETION_REPORT.md](COMPLETION_REPORT.md) and
[DEMO_SERVER.md](DEMO_SERVER.md). Start the interactive video workbench with
`RUN_C3_DEMO.ps1 -Interactive`; `-Serve` retains the original static comparison.
The workbench accepts a local video, reuses compatible pose caches, runs the
existing guarded ST-GCN and exports annotated video, scores, descriptive evidence,
research incidents, notification audit and append-only operator decisions.

Three separate tasks are reported: weak annotated first-onset forecasting,
weak active-fight recognition, and verified pre-contact aggression prediction.
The third remains blocked: independently verified positive precursor labels are
unavailable. Operational notifications remain disabled; scores are uncalibrated.
An incident review is not a training label or independent scientific validation.

Read [IMPLEMENTATION_GAPS.md](IMPLEMENTATION_GAPS.md) for the initial audit,
[STGCN_DIAGNOSIS.md](STGCN_DIAGNOSIS.md) for the later six-fit investigation,
[MODEL_RELIABILITY.md](MODEL_RELIABILITY.md) for calibration diagnostics,
[CONTACT_REVIEW.md](CONTACT_REVIEW.md) for explicit cue/contact annotation,
and [AUDIO_INTERFACE.md](AUDIO_INTERFACE.md) for optional audio status.
Existing data, reviews, experiments and models are preserved. Dashboard changes
are confined to the original `dashboard/` application. Obsolete root dashboard
copies were archived outside the repository at the user's request; root npm
shortcuts target `dashboard/frontend`. The older shared event dataclass remains unchanged. Its
generic forecast restriction still applies, while the actual dashboard v1.1
contract supports cautious `aggression`/`violence` candidates through the existing
shared event store. No shared contract fields or enums were changed.

## Historical v2 workflow

The following original instructions are retained as research history. Dataset and
training status claims below describe that earlier package, not today's workspace.
Use the newer documents above and locked preprocessing/split contracts for new runs.

### Reviewed Early Aggression Prediction + Optional Screaming-Aware Audio

**Phase 1 continuation:** See [PHASE1.md](PHASE1.md) for locked UBI splits, preserved review provenance, causal pose caches, the joint-only ST-GCN interface, current training blockers, and exact Windows commands. New training requires locked splits and matching causal preprocessing; the historical workflow below is retained for context.

**J26-DS-320 | Windows / VS Code | standalone, no changes to teammate source files**

This update extends the original uploaded `c3_model.py` (ST-GCN/DIFEM + causal GRU) and `legacy/run.py` (pose/tracking/quality). It adds **UBI-Fights annotation import -> two-person review -> onset + precursors -> cached features -> training -> validation threshold -> held-out event-level test**. A YAMNet **sound-event cue branch** adds learned audio context for screaming, yelling, shouting, crying, cheering, laughter, music and speech on cameras WITH sound. Without sound, all audio inputs are masked and the video model continues.

### What this does NOT claim

- The **downloaded UBI-Fights videos, ground-truth files, XD-Violence videos, or Kaggle datasets are NOT in the ZIP**, so no actual dataset was reviewed, no model was trained on real video, and no real-data accuracy / lead time is reported.
- UBI-Fights frame-level annotation identifies the annotated **physical fight onset** (roughly first contact). It **does not prove** that any behavior 1–3 seconds earlier is aggression. Before training, reviewers must check actual preceding cues and supply `precursors` intervals. Unknown pre-onset periods are **ignored by the precursor loss**; no fake precursor labels are generated.
- **A scream alone is not violence**; sports, celebration, accidents, music, films and audio not belonging to a visible person may cause false alarms. YAMNet probabilities are *sound-class hypotheses*, not violence probabilities or evidence of intent.
- Official XD-Violence test annotations must not be used for training or threshold calibration. Its training labels are weak; add independently verified onsets if using it for early-event supervised training.
- AV model requires *genuinely synchronized audio-video recordings* with real onset/precursor annotations. Unrelated audio clips and videos must never be paired to fabricate synchronized training examples.
- Real-time RTSP audio decoder is best-effort, **not a production-level timestamp synchronizer**. Calibrate latencies (video PTS + audio PTS / PTP) before relying on cross-stream cues. This prototype does not claim multi-camera robustness or ready-for-deployment performance.
- Permission/license, independent reviewers, and privacy-impact assessment are essential; output is for a human-operated safety review dashboard, not automatic interventions.

## 0. Prepare Windows Python environment

```powershell
cd C3_EarlyViolence_VSCode
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
# First install Torch for your CUDA version or CPU, from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
# For pretrained screaming, shouting and other sound event scores:
pip install -r requirements-audio.txt
ffmpeg -version
python -m pytest tests -q
```

FFmpeg and ffprobe must be on PATH. YAMNet may download pretrained weights from TF Hub at first use; caching is local to your environment, not included in this ZIP. CPU-only mode is possible but slow for full datasets and YOLO pose extraction. If TensorFlow+torch cannot coexist in an environment, run the audio feature caching in a separate Python 3.11 environment, and use precomputed NPZ for training.

## 1. Download the datasets under the official terms

- [UBI-Fights](https://socia-lab.di.ubi.pt/EventDetection) — preferred contact-onset ground truth and normal footage; obtain original annotation folder **and** videos.
- [XD-Violence](https://roc-ng.github.io/XD-Violence/) — synchronized audio/video with weak training supervision. Obtain separate real-onset training reviews for early prediction. Keep publisher test set locked.
- [RWF-2000](https://www.kaggle.com/datasets/vulamnguyen/rwf2000) and [Real-Life Violence](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — class-level only, suitable for optional later visual pretraining or independent onset review; **not automatically early-supervised**.
- [Audio-Based Violence Detection](https://www.kaggle.com/datasets/fangfangz/audio-based-violence-detection-dataset) — optional *separate* audio cue study; inspect actual classes and licensing. It cannot supply an onset timestamp for an unrelated CCTV fight video.

**Do not put datasets or `.pt` weights into the team Git repository.** Folder suggestion: `data/downloads/UBI-Fights/videos` and `data/downloads/UBI-Fights/annotation` (adjust to actual extracted layout).

## 2. Import the official UBI-Fights timestamps as candidates

First, **open an annotation text file** to identify whether it contains (a) one 0/1 flag **per frame**, (b) start/end frame pairs, (c) intervals in seconds, or (d) `.npy` 0/1 arrays. The importer intentionally refuses to guess the format.

**Example if original annotations are one binary label per frame:**

```powershell
python -m src.ubi_annotations --videos data/downloads/UBI-Fights/videos --annotations data/downloads/UBI-Fights/annotation --format frame_binary --out data/manifests/ubi_review.csv
```

Other choices: `--format npy_binary`, `--format interval_frames`, `--format interval_seconds`. Optional `--test-list data/downloads/UBI-Fights/test_videos.csv` **preserves the publisher test list**. Importer uses each video's actual FPS, checks frame counts and bounds, writes `ubi_review.errors.csv`. It **never auto-verifies annotation_status**.

### Two independent reviews (mandatory for `verified`)

Make short clips around each contact onset:

```powershell
python -m src.review_clips --manifest data/manifests/ubi_review.csv --out data/review_clips
```

Open `data/manifests/ubi_review.csv` in Excel/VS Code (save as ordinary UTF-8 CSV). For each video, review original video **as well as the contextual clip** and fill:

- `reviewer_a`, `reviewer_b`: distinct human reviewer identifiers, independently agreeing on the event.
- `decision`: `approve` / `reject`.
- `onset_override_s`: optional JSON `[[contact_start_s, physical_fight_end_s]]` corrected after review; blank uses imported fight intervals.
- `precursors`: **observed** pre-contact cues as JSON seconds, e.g. `[[4.5,5.9]]` before verified fight onset `6.0`. If no cues are observable, enter `[]`. DO NOT fabricate cues just because fight follows.
- `audio_cue_review`: context notes e.g. 'scream before contact? cheer? speech? audio mismatch?'. `notes`: visibility, occlusion, confidence, whether clip begins during fight, privacy/copyright caveats.

Each observed precursor must end before contact and be within six seconds of an onset. Review clips may contain violence and could be distressing; view only with the necessary research safeguards. **The two reviewer IDs are not cryptographic proof of agreement** — keep separate annotation forms and an adjudication log for dissertation-quality evidence.

Then:

```powershell
python -m src.review_annotations --reviewed data/manifests/ubi_review.csv --out data/manifests/ubi_verified.csv
```

`--allow-no-precursor` permits verified contact-only examples for active-fight training, but does *not* invent early supervision. Positive early-training windows require actual `precursors`.

**Negative videos** need manual approval too. Include *hard negatives*: shouting crowds, cheering stadiums, excited talking, arguments without physical fights, celebrations, playgrounds, hugging, running, raised-arm dancing. This is crucial when audio is available.

## 3. Prepare synchronized features

For the optional pretrained sound-event branch, run:

```powershell
python -m src.cache_dataset --manifest data/manifests/ubi_verified.csv --out-dir data/cache --index data/cache_index.json --audio-events yamnet
```

This produces 16-D basic log FFT energy bands plus **8-D YAMNet class confidence features** (`Screaming`, `Yell`, `Shout`, `Crying, sobbing`, `Cheering`, `Laughter`, `Music`, `Speech`). Video features = 16-D RGB summaries + pose kinematics = 22-D + 16×17×2×3 skeleton. If running on a no-audio-only dataset and you do not want TensorFlow, use `--audio-events none`. Do not mix NPZ feature dimensions or event backends across a single training run; cache the same way for val/test/live.

**Why audio?** Temporal shouting, screams or cries *combined with* rapid closing distance / forceful arm gestures may improve probability estimates for the next 1–3 seconds. The model must learn this association on genuine synchronized examples; there is **no fixed rule** `screaming => imminent fight`.

### Audit whether screams actually correlate with an onset

```powershell
python -m src.audio_cue_audit --manifest data/manifests/ubi_verified.csv --index data/cache_index.json --out outputs/audio_cue_audit.csv
```

This outputs per-second scream/yell/etc probabilities and compares reviewed pre-onset, other pre-onset, baseline and active-fight audio. If the videos contain no usable audio, **there will be no sound-audit measurements**. For audio experiments with XD-Violence, produce independent onset reviews and a combined or separate genuine AV manifest; don't use publisher test labels as training data.

## 4. Train separate Video-only and Audio+Video risk models

```powershell
python -m src.train --mode video --manifest data/manifests/ubi_verified.csv --index data/cache_index.json --out weights/c3_video.pt --steps 1000
python -m src.train --mode av --manifest data/manifests/ubi_verified.csv --index data/cache_index.json --out weights/c3_av.pt --steps 1000
```

The AV command **requires at least some reviewed positive samples with actual audio**. If UBI videos have no sound, create the independently onset-reviewed XD (or consented field recording) AV training manifest and re-cache it. Video-only remains fully usable with no audio. The training script rejects missing independent review precursors and uses missing-modality masking to improve robustness.

Train/val/test are grouped by `group` and locked `split` assignments when available. Same scene must NOT appear in multiple splits, including duplicate crops, download re-encodes, or audio isolated from the original video. Each video has `intervals` (verified physical fight contact) and `precursors` (verified early cue windows) in seconds.

## 5. Calibrate on validation and evaluate early EVENTS on test

```powershell
# Select threshold using validation only; max 0.5 false early alerts/hour target
python -m src.evaluate_events --manifest data/manifests/ubi_verified.csv --index data/cache_index.json --weights weights/c3_video.pt --split val --calibrate --max-far 0.5 --out outputs/video_val_calibration.json

# Locked test! Event recall @1/2/3 seconds, lead time, false alerts per camera hour
python -m src.evaluate_events --manifest data/manifests/ubi_verified.csv --index data/cache_index.json --weights weights/c3_video.pt --split test --calibration outputs/video_val_calibration.json --out outputs/video_test_metrics.json
```

For AV repeat the same two commands with `weights/c3_av.pt`, and use **different output filenames**. `--split test --calibrate` is explicitly prohibited. If the validation target cannot be met, calibration JSON reports `val_target_unmet`. Evaluator treats post-contact alerts as *late* not early detections. If <1 video-hour is available, FAR/h is unstable and warning is produced.

**Warning:** AV-vs-video results can only be compared fairly when evaluated on the same eligible recorded videos/split. Separate UBI vs XD results are not directly comparable.

## 6. Run local files, IP cameras, or CCTV without microphone

```powershell
python -m src.live --source 0 --weights weights/c3_video.pt --camera-id c3_webcam --audio off --show --calibration outputs/video_val_calibration.json
python -m src.live --source data/videos/scene_with_audio.mp4 --weights weights/c3_av.pt --camera-id c3_av_01 --show --calibration outputs/av_val_calibration.json
python -m src.live --source rtsp://YOUR_CAMERA_URL --weights weights/c3_av.pt --camera-id c3_rtsp_01 --calibration outputs/av_val_calibration.json
```

No microphone / audio decode failure => `mods.audio=0` and no fake screaming score. For webcam `0`, no microphone is assumed. In live RTSP mode, audio/video timestamp sync is best-effort and must be properly engineered before deployment. Very low pose quality suppresses C3 alerts, increases uncertainty; C3 events always require human review.

## 7. Integrate with Component 02 — no duplicated trackers

```python
from src.shared_adapter import C3UpstreamAdapter

# one instance per camera; no extra YOLO instance:
c3 = C3UpstreamAdapter('weights/c3_video.pt', camera_id='cam_01')
# from shared upstream tracker at ~8 fps:
c3.push(timestamp_s, frame_bgr, keypoints_Nx17x3, boxes_Nx4_xyxy, byte_track_ids)
# at ~1 Hz:
step_assessment, optional_event = c3.assess(timestamp_s, width, height, audio_wave=None)
```

C3 accepts the shared perception output without importing C2 loitering rules. C3 owns only its own **per-camera recurrent model state** and emits `component='c3_early_aggression'`, `camera_id`, event UUID, UTC timestamp, `threat_score`, `priority_tier`, `uncertainty`, source cues and `human_review_required=True`. **Finalize event schema with C1, C2 and C4 before merge**. The adapter and the standalone demo cannot be run on the same camera simultaneously without duplicate work; choose one ingestion mode.

## Research validation checklist

1. Confirm fps and contact onset for every UBI-Fights event, document annotator agreement, discard no-context fight clips from early evaluation.
2. Time-align audio and frame timestamps; document audio provenance and where sound does not match visible fight.
3. Evaluate: `(V+Pose)` vs `(V+Pose+FFT Audio)` vs `(V+Pose+YAMNet cues)`, on the **same held-out AV scenes**.
4. Report event recall at lead >=1s, >=2s, >=3s, event precision, false alerts / camera-hour, median early lead seconds; include null/empty metrics rather than inventing precision on empty samples.
5. Test camera domains: low light, occlusion, variable person count, sports, celebration, speech/screaming without violence; measure calibration, inference latency and audio dropout.
6. Do not claim 'fight predicted 3 seconds before occurrence' until the held-out event-level results demonstrate it.

## Source materials / learning references

- UBI-Fights original research: https://socia-lab.di.ubi.pt/EventDetection
- XD-Violence benchmark: https://roc-ng.github.io/XD-Violence/
- TensorFlow YAMNet: https://www.tensorflow.org/hub/tutorials/yamnet
- RWF-2000: https://www.kaggle.com/datasets/vulamnguyen/rwf2000
- Risk of false alarms in audio aggression detectors: https://projects.propublica.org/graphics/aggression-detector-data-analysis


### Unit testing without datasets

```powershell
python -m pytest tests -q
# Optional slower Torch forward-pass smoke test, where Torch imports successfully:
python -m pytest tests_optional -q
```

The `data/manifests/EXAMPLE_ONLY_DO_NOT_TRAIN.csv` has **fabricated illustrative numbers** and references deliberately nonexistent video paths. Never use it for real experiments.

## Component 02 compatibility supplement (adapter v2.1)

The provided Component 02 **description** says it exports ByteTrack IDs and `xyxy`
bounding boxes, but not 17-joint pose coordinates. Therefore the original adapter
requiring `(N,17,3)` keypoints cannot consume a C2-only track output unaided.
`src/pose_enrichment.py` now supports one **pose estimator** without a second
tracker, and matches pose outputs back to C2 person boxes using one-to-one IoU.

```python
from src.shared_adapter import C3UpstreamAdapter

# Create ONCE per camera; C2 already owns tracking.
c3 = C3UpstreamAdapter('weights/c3_video.pt',camera_id='cam_01',
                       pose_weights='yolo11n-pose.pt')

# From a shared video reader and C2 tracker, approximately 8FPS:
c3.push_from_c2(timestamp_s,frame_bgr,c2_boxes_xyxy,c2_track_ids)
# C2 remains the source of IDs; C3 uses pose predict, not pose track.

# Approximately 1Hz; 16kHz mono waveform only if synchronized and available:
assessment,event=c3.assess(timestamp_s,frame_width,frame_height,audio_wave=None)
```

If the shared upstream already produces **COCO-17** poses, omit
`pose_weights` and use `c3.push(timestamp_s,frame_bgr,kps,boxes,ids)`.
Frame timestamps must be strictly increasing *per camera*. C2's exported CSV
alone does not contain pixels and cannot serve as a pose-estimation source: pass
the actual synchronized video frames. Bounding box / pose matching is heuristic,
not guaranteed for overlapping subjects: unmatched joints get zero confidence
and quality should suppress unsupported risk alerts.

C3 has isolated per-camera pose imputation state (the original `legacy/run.py`
used a module-global mutable imputer), per-camera recurrent state, and by
default writes into `outputs/c3/<camera_id>/`, never into shared teammate output
folders. The integration event schema and camera scheduler still **must** be
verified against the real C1/C2/C4 source, which was not accessible during this
revision; no claim of fully merged repository compatibility is made.

Do NOT run C3 standalone `src.live` on a camera at the same time as the shared
C2 ingestion. Use either the standalone tracker or the shared adapter, not both.
Do not load `.pt` checkpoints from untrusted sources (`torch.load`).
