# Proactive Anomaly Detection and Security Alerting System

> **An AI-Powered Behavioral Surveillance Layer for Existing CCTV Infrastructure**
>
> *"The system flags. The human judges."*

## Project Overview

| Field | Detail |
|---|---|
| Institution | Sri Lanka Institute of Information Technology (SLIIT) |
| Programme | BSc (Hons) IT — Data Science |
| Project ID | J26-DS320 |
| Type | Final-Year Group Research |
| Members | 4 |

## Team

| Member | Component | IT Number |
|---|---|---|
| <<Member 1 Name>> | C1 — Spatial Anomaly Detection | <<IT Number>> |
| <<Member 2 Name>> | C2 — Predictive Loitering Detection | <<IT Number>> |
| <<Member 3 Name>> | C3 — Aggression & Violence Recognition | <<IT Number>> |
| Bogahawatta M.O | C4 — Fall & Abandoned-Object Detection | IT23242418 |

## Supervisors

- **Supervisor:** Ms. Malithi Nawarathne
- **Co-Supervisor:** Dr. Mahima Weerasinghe
- **External Supervisor:** Mr. Asiri Gawesha

## Repository Structure

```
proactive-anomaly-detection/
│
├── shared/                    # Shared perception layer (ALL members use this)
│   ├── detection/             # YOLOv8 person & object detection
│   ├── tracking/              # ByteTrack / DeepSORT multi-object tracking
│   ├── pose/                  # MediaPipe / OpenPose 2D pose estimation
│   ├── schemas/               # Common event schema (JSON contract)
│   └── utils/                 # Shared utilities (video I/O, drawing, logging)
│
├── components/                # Individual member components
│   ├── c1_spatial_anomaly/    # Member 1: intrusion, tailgating, crowding
│   ├── c2_predictive_loitering/ # Member 2: suspicious dwell detection
│   ├── c3_aggression_detection/ # Member 3: pose-based violence recognition
│   └── c4_critical_incidents/ # Member 4: fall detection + abandoned objects
│
├── dashboard/                 # Operator interface
│   ├── backend/               # FastAPI alert API
│   └── frontend/              # React / Streamlit UI
│
├── integration/               # System-level integration
│   ├── alert_fusion/          # Alert merging, dedup, prioritisation
│   └── pipeline/              # End-to-end processing pipeline
│
├── datasets/                  # Data directory (NOT committed — see .gitignore)
│   ├── raw/                   # Original downloaded datasets
│   ├── processed/             # Preprocessed frames / clips
│   └── annotations/           # Ground-truth labels
│
├── experiments/               # Experiment tracking (NOT committed)
│   ├── logs/                  # MLflow / TensorBoard logs
│   └── checkpoints/           # Saved model weights
│
├── docs/                      # Documentation
│   ├── proposal/              # Proposal documents
│   ├── guides/                # Setup guides, API docs
│   └── diagrams/              # Architecture diagrams (draw.io files)
│
├── tests/                     # Test suites
│   ├── unit/                  # Unit tests per component
│   └── integration/           # End-to-end integration tests
│
├── scripts/                   # Utility scripts (download data, run eval, etc.)
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment (alternative)
├── .gitignore                 # Files to exclude from Git
├── .env.example               # Environment variable template
└── README.md                  # This file
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<<your-org>>/proactive-anomaly-detection.git
cd proactive-anomaly-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment variables
cp .env.example .env
# Edit .env with your local paths

# 5. Test shared perception layer
python shared/detection/detect.py --source sample.mp4

# 6. Run your component
python components/c4_critical_incidents/run.py --source sample.mp4
```

## Branch Strategy

```
main                           # Protected — only merge tested code here
├── develop                    # Integration branch — merge components here first
├── member1/spatial            # Member 1 working branch
├── member2/loitering          # Member 2 working branch
├── member3/aggression         # Member 3 working branch
├── member4/critical-incidents # Member 4 working branch
└── shared/perception          # Shared layer updates (any member, reviewed by all)
```

### Rules

1. **Never push directly to `main` or `develop`** — always use Pull Requests
2. **Each member works on their own branch** — `member4/critical-incidents`
3. **Shared layer changes go through `shared/perception`** branch and need review from at least 1 other member
4. **Merge to `develop` first** — test integration there before merging to `main`
5. **Write a clear commit message** — `[C4] Add CNN-LSTM fall detection model`

## Commit Message Format

```
[TAG] Short description of change

Tags:
[C1]      — Component 1 (spatial anomaly)
[C2]      — Component 2 (loitering)
[C3]      — Component 3 (aggression)
[C4]      — Component 4 (fall & abandoned objects)
[SHARED]  — Shared perception layer
[DASH]    — Dashboard / API
[INTEG]   — Integration / pipeline
[DATA]    — Dataset preparation
[DOCS]    — Documentation
[FIX]     — Bug fix
[TEST]    — Test code

Examples:
[C4] Add CNN-LSTM fall detection model
[SHARED] Upgrade YOLOv8 to v8.1 for better indoor accuracy
[DATA] Add Le2i fall dataset annotation script
[FIX] Fix ByteTrack ID switch on occluded persons
[DOCS] Update architecture diagram with alert fusion layer
```

## Event Schema (Shared Contract)

All 4 components output events in this JSON format:

```json
{
  "timestamp": "2026-10-01T14:32:05.123Z",
  "camera_id": "cam_lobby_01",
  "person_id": "P-042",
  "object_id": null,
  "event_type": "fall_detected",
  "threat_score": 0.94,
  "bounding_box": [120, 340, 280, 520],
  "zone": "main_corridor",
  "explanation_signals": [
    "vertical_velocity_exceeded",
    "aspect_ratio_below_threshold",
    "temporal_confirmation_10_frames"
  ],
  "model_version": "c4_fall_v1.2"
}
```

## License

This project is developed as part of an academic research programme at SLIIT.
All rights reserved. Not for commercial use without permission.
