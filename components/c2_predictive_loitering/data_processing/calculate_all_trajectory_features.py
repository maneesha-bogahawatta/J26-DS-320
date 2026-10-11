from pathlib import Path

import os
import numpy as np
import pandas as pd


# ==================================================
# 1. Paths and configuration
# ==================================================

COMPONENT_ROOT = Path(__file__).resolve().parents[1]

WORK_ROOT = Path(
    os.environ.get(
        "C2_WORK_ROOT",
        COMPONENT_ROOT,
    )
)

INPUT_CSV = (
    WORK_ROOT
    / "data"
    / "ground_truth"
    / "people_ground_truth_trajectories.csv"
)

OUTPUT_FEATURES_CSV = (
    WORK_ROOT
    / "data"
    / "processed"
    / "all_trajectory_features.csv"
)

OUTPUT_SUMMARY_CSV = (
    WORK_ROOT
    / "data"
    / "processed"
    / "trajectory_summary.csv"
)

# Speed is measured in normalized image units per second.
PAUSE_SPEED_THRESHOLD = 0.02


# ==================================================
# 2. Validate input
# ==================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Combined trajectory file was not found:\n{INPUT_CSV}"
    )

OUTPUT_FEATURES_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ==================================================
# 3. Feature-calculation function
# ==================================================

def calculate_track_features(track):
    track = track.sort_values("frame_number").copy()

    # Smooth minor annotation movements.
    track["smooth_x"] = (
        track["centre_x"]
        .rolling(
            window=5,
            center=True,
            min_periods=1,
        )
        .median()
    )

    track["smooth_y"] = (
        track["centre_y"]
        .rolling(
            window=5,
            center=True,
            min_periods=1,
        )
        .median()
    )

    # Time difference between observations.
    track["delta_time"] = track["timestamp"].diff()

    track.loc[
        track["delta_time"] <= 0,
        "delta_time",
    ] = np.nan
