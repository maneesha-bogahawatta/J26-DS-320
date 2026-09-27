# Component 4 — Vision-Only Fall and Abandoned-Object Detection

**Owner:** Bogahawatta M.O (IT23242418)

## What This Component Does

Detects two types of critical incidents from ordinary indoor CCTV:

1. **Falls / collapse** — a person transitions from upright to fallen
2. **Abandoned objects** — an object is left unattended by its owner

Both incident types produce events using the shared `IncidentEvent` schema
and are sent to the operator dashboard for human review.

## Directory Structure

```
c4_critical_incidents/
├── fall_branch/           # Fall detection pipeline
│   ├── feature_extract.py # Extract pose + motion features per frame
│   ├── temporal_model.py  # CNN-LSTM temporal classifier
│   ├── baseline.py        # Rule-based baseline (aspect ratio + motion)
│   └── train.py           # Training script
│
├── object_branch/         # Abandoned-object pipeline
│   ├── owner_assoc.py     # Owner–object association logic
│   ├── persistence.py     # Stationary-duration tracker
│   ├── baseline.py        # Rule-based baseline (distance + timer)
│   └── train.py           # Training script
│
├── models/                # Saved model weights (NOT in Git)
├── data/                  # Component-specific data scripts
├── eval/                  # Evaluation scripts and results
├── configs/               # Hyperparameter configs (YAML)
├── run.py                 # Main entry point
└── README.md              # This file
```

## Datasets

| Dataset | Type | Size | Use |
|---|---|---|---|
| Le2i | Falls | 190 videos | Training + test |
| URFD | Falls | 70 videos | Training + test |
| GMDCSA-24 | Falls | 160 clips | Training + test |
| ABODA | Abandoned objects | 11 sequences | Training + test |
| PETS2006 | Abandoned luggage | 7 sequences | Test |
| CCTV-KD | Person + luggage | Images | Supplementary |

## Baselines

| Task | Baseline Method | Expected F1 |
|---|---|---|
| Fall detection | Bounding-box aspect ratio + motion energy | 65–70% |
| Abandoned objects | Fixed distance (<1.5m) + fixed timer (>120s) | 60–65% |

## Advanced Models

| Task | Method | Expected F1 |
|---|---|---|
| Fall detection | CNN (3-layer) + LSTM (2-layer, 128 units) | 85–90% |
| Abandoned objects | Owner association + temporal persistence | 80–85% |

## Usage

```bash
# Run fall detection baseline
python components/c4_critical_incidents/fall_branch/baseline.py \
  --source datasets/raw/le2i/ \
  --output experiments/logs/c4_fall_baseline/

# Run abandoned-object baseline
python components/c4_critical_incidents/object_branch/baseline.py \
  --source datasets/raw/aboda/ \
  --output experiments/logs/c4_object_baseline/

# Train CNN-LSTM fall model
python components/c4_critical_incidents/fall_branch/train.py \
  --config configs/fall_cnn_lstm.yaml

# Run full component
python components/c4_critical_incidents/run.py \
  --source sample_video.mp4
```

## Event Types Produced

- `fall_detected` (priority 1 — medical emergency)
- `abandoned_object` (priority 2 — security concern)

## Key References

- [Sal22b] Salimi et al. (2022) — Pose-based fall detection with CNN
- [Vrs25] Vrsalović et al. (2025) — Real-time abandoned luggage detection
- [Gay24] Gayathri & Mohanapriya (2024) — Fall detection systematic review
