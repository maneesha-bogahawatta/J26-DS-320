# Research Project Periodic Log Entry — Component 03

**Draft chronology:** The reporting periods and suggested dates follow the team
member's example, as requested. They do not independently establish when the
student performed each activity. Confirm the dates and earlier research/design
entries against your own records before submission. October implementation and
test results reflect the current Component 03 artifacts. No supervisor comment,
signature, independent human decision or Git push is fabricated.

**Group ID:** J26-DS-320  
**Student ID:** [Your Student ID]  
**Student Name:** [Your Full Name]  
**Individual Component:** Component 03 — Early Aggression and Violence Prediction with Explainable Human-in-the-Loop Alerting.

## Periodic Log Entry Form 1

**Period: 25 July–10 August 2026**

| Suggested date | Work — draft entry |
|---|---|
| 28-07-2026 | Reviewed the Component 03 objectives and defined the scope of CCTV-based aggression and violence analysis. Identified the need to distinguish potential pre-contact warnings from recognition of violence already in progress. |
| 03-08-2026 | Studied person detection, multi-object tracking and pose-based action-recognition approaches. Identified human skeletal motion and interactions between people as relevant inputs for the proposed aggression-analysis pipeline. |
| 10-08-2026 | Defined the research requirements for early-warning evaluation, including observable pre-contact cues, first physical contact, false alerts and operator review. Recognized that a fight annotation alone does not verify an earlier aggressive cue. |

## Periodic Log Entry Form 2

**Period: 11–24 August 2026**

| Suggested date | Work — draft entry |
|---|---|
| 12-08-2026 | Identified dataset requirements covering fight and non-fight recordings, pre-event context, occlusion and different camera conditions. Included the need for challenging non-violent movements and interactions when assessing false warnings. |
| 18-08-2026 | Reviewed available dataset options, including UBI-Fights, RWF-2000, Real-Life Violence and XD-Violence. Distinguished video-level violence labels from temporal annotations needed for onset forecasting and verified pre-contact evaluation. |
| 24-08-2026 | Designed the proposed processing flow using YOLO11-Pose, ByteTrack, temporal pose features, ST-GCN models and a human-operated review interface. Kept optional audio analysis separate from the primary video-only implementation. |

## Periodic Log Entry Form 3

**Period: 25 August–7 September 2026**

| Suggested date | Work — draft entry |
|---|---|
| 26-08-2026 | Specified the dataset organization and source-group separation strategy. Planned duplicate detection and locked training, validation and test assignments to reduce data leakage between related recordings. |
| 01-09-2026 | Designed annotation fields for reviewed coverage, threatening-cue intervals, physical-contact timing and ongoing fight intervals. Defined the distinction between weak, provisional and independently verified evidence. |
| 07-09-2026 | Defined baseline and advanced skeleton-model designs. The baseline uses joint sequences, while the advanced design incorporates joints, bones, past motion and person-interaction features to investigate temporal aggression patterns. |

## Periodic Log Entry Form 4

**Period: 8–21 September 2026**

| Suggested date | Work — draft entry |
|---|---|
| 09-09-2026 | Designed causal temporal preprocessing using currently available and past frames. Specified trailing pose windows, missing-joint masks and scene-change resets to prevent future observations from influencing earlier predictions. |
| 15-09-2026 | Defined pose-quality checks for unreliable detections, missing keypoints and insufficient observation history. Established abstention as the appropriate output when usable pose evidence is unavailable. |
| 21-09-2026 | Prepared the evaluation protocol for separate weak-onset forecasting and active-fight tasks. Included precision, recall, F1, AUROC, warning lead time and explicit reporting of false alerts and pose coverage. |

## Periodic Log Entry Form 5

**Period: 22 September–5 October 2026**

| Suggested date | Work — draft entry |
|---|---|
| 23-09-2026 | Planned Component 03 modules for feature extraction, caching, training, evaluation and inference. Defined versioned outputs so that new experiments would preserve original datasets, manually reviewed annotations and previous results. |
| 28-09-2026 | Reviewed the project repository structure and shared incident-event contract. Identified the boundaries between Component 03, shared perception and the other team components, including the need for compatible event fields. |
| 05-10-2026 | Prepared the implementation plan for reusable pose caches, separate model heads and evidence-based research outputs. Defined requirements for camera identifiers, source timestamps, tracked identities and human review. |

## Periodic Log Entry Form 6

**Period: 6–21 October 2026**

| Suggested date | Work — draft entry |
|---|---|
| 09-10-2026 | Developed the initial Component 03 video-processing structure, configuration and inference workflow. Organized the component so subsequent pose, model and evaluation modules could be added without rebuilding the team project. |
| 16-10-2026 | Developed the aggression-analysis pipeline using person tracking and pose sequences. Integrated the processing stages needed to extract and reuse skeletal features from prerecorded CCTV videos. |
| 18-10-2026 | Added temporal pose and person-interaction features, explanatory research signals, track-quality assessment and uncertainty-related information. Kept heuristic detections separate from independently verified training labels. |
| 19-10-2026 | Extended causal feature processing, model-training guards and event-level evaluation. Separated early-warning supervision from active-fight labels and prevented test data from being used for fitting or threshold selection. |
| 20-10-2026 | Prepared the locked UBI-Fights foundation with 1,000 sources split into 791 training, 141 validation and 68 test videos. Added duplicate/source-group protection, annotation provenance and compatible causal pose-cache checks. |
| 21-10-2026 | Trained and compared weakly supervised Baseline and Advanced ST-GCN models, investigated score direction and class imbalance, and evaluated threshold and calibration behavior. Retained the guarded baseline because reliable pre-contact generalization was not demonstrated. |
| 21-10-2026 | Completed local video upload and inference, annotated outputs, timestamped incident evidence, operator status review and ZIP export. Added strict cue/contact review forms and cache/identity regression checks. The final suite passed 220 tests and 52 subtests; verified 1–3-second pre-contact prediction remains unvalidated. |

## Footer for each reporting form

Supervisor Comment: _________________________________________________

Student progressing well in research project: Yes / No [Supervisor to complete]

If No, project work required before the next evaluation: _______________

Signature: _______________________    Date: _______________________
