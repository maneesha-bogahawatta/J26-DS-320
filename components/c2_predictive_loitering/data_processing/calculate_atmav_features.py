"""Calculate person-level trajectory features from ATMA-V tracks."""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


WORK_ROOT = Path(
    os.environ.get(
        "C2_WORK_ROOT",
        ".",
    )
)

PAUSE_SPEED_THRESHOLD = 0.02
DIRECTION_CHANGE_THRESHOLD = 45.0
MINIMUM_OBSERVATIONS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate ATMA-V trajectory features."
    )

    parser.add_argument(
        "--video-id",
        default="12",
        help="Video ID matching the trajectory CSV filename.",
    )

    return parser.parse_args()


def angle_difference(first_angle, second_angle):
    """Return the smallest difference between two angles."""

    difference = abs(first_angle - second_angle)

    return min(
        difference,
        360.0 - difference,
    )


def calculate_track_features(track):
    """Calculate features for one ByteTrack trajectory."""

    track = track.sort_values(
        "frame_number"
    ).copy()

    track["delta_x"] = (
        track["centre_x"].diff()
    )

    track["delta_y"] = (
        track["centre_y"].diff()
    )

    track["delta_time"] = (
        track["timestamp"].diff()
    )

    track["frame_gap"] = (
        track["frame_number"].diff()
    )

    track["step_distance"] = np.sqrt(
        track["delta_x"] ** 2
        + track["delta_y"] ** 2
    )

    valid_time = (
        track["delta_time"] > 0
    )

    track["speed"] = 0.0

    track.loc[
        valid_time,
        "speed",
    ] = (
        track.loc[
            valid_time,
            "step_distance",
        ]
        / track.loc[
            valid_time,
            "delta_time",
        ]
    )

    track["is_pause"] = (
        track["speed"]
        <= PAUSE_SPEED_THRESHOLD
    )

    directions = np.degrees(
        np.arctan2(
            track["delta_y"],
            track["delta_x"],
        )
    )

    direction_changes = 0

    valid_directions = (
        directions.dropna().tolist()
    )

    for index in range(
        1,
        len(valid_directions),
    ):
        change = angle_difference(
            valid_directions[index - 1],
            valid_directions[index],
        )

        if (
            change
            >= DIRECTION_CHANGE_THRESHOLD
        ):
            direction_changes += 1

    first_row = track.iloc[0]
    last_row = track.iloc[-1]

    direct_displacement = math.dist(
        (
            first_row["centre_x"],
            first_row["centre_y"],
        ),
        (
            last_row["centre_x"],
            last_row["centre_y"],
        ),
    )

    total_path_length = float(
        track["step_distance"].sum()
    )

    if total_path_length > 0:
        path_efficiency = (
            direct_displacement
            / total_path_length
        )
    else:
        path_efficiency = 0.0

    observation_count = len(track)

    first_frame = int(
        track["frame_number"].min()
    )

    last_frame = int(
        track["frame_number"].max()
    )

    expected_observations = (
        last_frame - first_frame + 1
    )

    coverage_ratio = min(
        observation_count
        / max(expected_observations, 1),
        1.0,
    )

    mean_confidence = float(
        track[
            "detection_confidence"
        ].mean()
    )

    track_quality = (
        coverage_ratio
        * mean_confidence
    )

    dwell_time = max(
        float(
            last_row["timestamp"]
            - first_row["timestamp"]
        ),
        0.0,
    )

    maximum_displacement = 0.0

    if observation_count > 1:
        origin_x = float(
            first_row["centre_x"]
        )

        origin_y = float(
            first_row["centre_y"]
        )

        displacement_from_origin = np.sqrt(
            (
                track["centre_x"]
                - origin_x
            ) ** 2
            + (
                track["centre_y"]
                - origin_y
            ) ** 2
        )

        maximum_displacement = float(
            displacement_from_origin.max()
        )

    frame_gaps = (
        track["frame_gap"]
        .fillna(1)
    )

    missing_frame_count = int(
        np.maximum(
            frame_gaps - 1,
            0,
        ).sum()
    )

    eligible_for_analysis = (
        observation_count
        >= MINIMUM_OBSERVATIONS
        and track_quality >= 0.30
    )

    summary = {
        "dataset_source": "ATMA-V",
        "video_id": str(
            first_row["video_id"]
        ),
        "track_id": int(
            first_row["track_id"]
        ),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "observation_count": observation_count,
        "expected_observations": (
            expected_observations
        ),
        "missing_frame_count": (
            missing_frame_count
        ),
        "dwell_time": round(
            dwell_time,
            4,
        ),
        "mean_speed": round(
            float(track["speed"].mean()),
            6,
        ),
        "maximum_speed": round(
            float(track["speed"].max()),
            6,
        ),
        "maximum_displacement": round(
            maximum_displacement,
            6,
        ),
        "total_path_length": round(
            total_path_length,
            6,
        ),
        "path_efficiency": round(
            path_efficiency,
            6,
        ),
        "pause_ratio": round(
            float(
                track["is_pause"].mean()
            ),
            6,
        ),
        "direction_changes": (
            direction_changes
        ),
        "coverage_ratio": round(
            coverage_ratio,
            6,
        ),
        "mean_detection_confidence": round(
            mean_confidence,
            6,
        ),
        "track_quality": round(
            track_quality,
            6,
        ),
        "eligible_for_analysis": (
            eligible_for_analysis
        ),
    }

    return track, summary


def main():
    arguments = parse_arguments()

    input_csv = (
        WORK_ROOT
        / "data"
        / "raw_tracks"
        / (
            f"{arguments.video_id}"
            "_trajectories.csv"
        )
    )

    output_directory = (
        WORK_ROOT
        / "data"
        / "processed"
    )

    frame_output = (
        output_directory
        / (
            f"{arguments.video_id}"
            "_trajectory_features.csv"
        )
    )

    summary_output = (
        output_directory
        / (
            f"{arguments.video_id}"
            "_trajectory_summary.csv"
        )
    )

    if not input_csv.exists():
        raise FileNotFoundError(
            "Trajectory file was not found:\n"
            f"{input_csv}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectories = pd.read_csv(
        input_csv
    )

    required_columns = {
        "video_id",
        "frame_number",
        "timestamp",
        "track_id",
        "centre_x",
        "centre_y",
        "detection_confidence",
    }

    missing_columns = (
        required_columns
        - set(trajectories.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame_features = []
    summary_rows = []

    grouped_tracks = trajectories.groupby(
        [
            "video_id",
            "track_id",
        ],
        sort=True,
    )

    for (
        _,
        track,
    ) in grouped_tracks:
        enhanced_track, summary = (
            calculate_track_features(
                track
            )
        )

        frame_features.append(
            enhanced_track
        )

        summary_rows.append(
            summary
        )

    if frame_features:
        all_frame_features = pd.concat(
            frame_features,
            ignore_index=True,
        )
    else:
        all_frame_features = pd.DataFrame()

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    all_frame_features.to_csv(
        frame_output,
        index=False,
    )

    summary_dataframe.to_csv(
        summary_output,
        index=False,
    )

    eligible_count = int(
        summary_dataframe[
            "eligible_for_analysis"
        ].sum()
    )

    print()
    print("ATMA-V feature calculation completed")
    print("------------------------------------")
    print(
        f"Input trajectory rows: "
        f"{len(trajectories)}"
    )
    print(
        f"Person tracks: "
        f"{len(summary_dataframe)}"
    )
    print(
        f"Eligible tracks: "
        f"{eligible_count}"
    )
    print(
        f"Rejected/uncertain tracks: "
        f"{len(summary_dataframe) - eligible_count}"
    )
    print(
        f"Frame features: "
        f"{frame_output}"
    )
    print(
        f"Track summary: "
        f"{summary_output}"
    )


if __name__ == "__main__":
    main()