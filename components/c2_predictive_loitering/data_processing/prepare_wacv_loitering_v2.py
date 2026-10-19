"""Prepare robust, behaviour-rich WACV24 trajectory sequences.

Version 2 preserves the official train/test split while addressing two
limitations found during baseline error analysis: extreme perspective-coordinate
outliers and insufficient explicit behavioural context for short trajectories.
All clipping, pause and scaling parameters are fitted on training data only.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WACV_ROOT = Path(os.environ.get("WACV_LOITERING_ROOT", "RS-WACV24_Loitering"))
DEFAULT_WORK_ROOT = Path(os.environ.get("C2_WORK_ROOT", "."))
SEQUENCE_LENGTH = 120
SMOOTHING_WINDOW = 5
LOWER_QUANTILE = 0.001
UPPER_QUANTILE = 0.999
SCALER_EPSILON = 1e-8
DIRECTION_CHANGE_RADIANS = np.deg2rad(45.0)

FEATURE_COLUMNS = [
    "absolute_x",
    "absolute_y",
    "relative_x",
    "relative_y",
    "delta_x",
    "delta_y",
    "speed",
    "acceleration",
    "direction_sin",
    "direction_cos",
    "angular_change",
    "cumulative_path_log",
    "displacement_log",
    "path_efficiency",
    "activity_width_log",
    "activity_height_log",
    "confinement_area_log",
    "pause_indicator",
    "cumulative_pause_ratio",
    "cumulative_direction_changes_log",
    "revisit_count_log",
    "occlusion",
    "cumulative_occlusion_ratio",
    "elapsed_frames_log",
    "observed_fraction",
]

REQUIRED_FRAME_COLUMNS = {
    "folder", "clip", "frame", "id", "class", "x", "y", "width",
    "height", "occlusion", "x_transform", "y_transform", "loitering",
}
REQUIRED_LABEL_COLUMNS = {"id", "loitering"}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare robust WACV24 loitering sequences (feature version 2)."
    )
    parser.add_argument("--wacv-root", type=Path, default=DEFAULT_WACV_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--smoothing-window", type=int, default=SMOOTHING_WINDOW)
    parser.add_argument("--lower-quantile", type=float, default=LOWER_QUANTILE)
    parser.add_argument("--upper-quantile", type=float, default=UPPER_QUANTILE)
    return parser.parse_args()


def validate_columns(dataframe, required_columns, source_name):
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"{source_name} is missing columns: {sorted(missing)}")


def load_split(dataset_root, split_name):
    frame_path = dataset_root / f"X_{split_name}.csv"
    label_path = dataset_root / f"{split_name}_df.csv"
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame file was not found: {frame_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file was not found: {label_path}")

    frames = pd.read_csv(frame_path)
    labels = pd.read_csv(label_path)
    validate_columns(frames, REQUIRED_FRAME_COLUMNS, str(frame_path))
    validate_columns(labels, REQUIRED_LABEL_COLUMNS, str(label_path))

    if frames["id"].isna().any() or frames["frame"].isna().any():
        raise ValueError(f"{split_name}: missing trajectory IDs or frame numbers.")
    if frames[["x_transform", "y_transform"]].isna().any().any():
        raise ValueError(f"{split_name}: missing transformed coordinates.")
    if not np.isfinite(frames[["x_transform", "y_transform"]]).all().all():
        raise ValueError(f"{split_name}: non-finite transformed coordinates.")
    if (~frames["loitering"].isin([0, 1])).any():
        raise ValueError(f"{split_name}: invalid frame labels.")
    if (~labels["loitering"].isin([0, 1])).any():
        raise ValueError(f"{split_name}: invalid trajectory labels.")
    if labels["id"].duplicated().any():
        raise ValueError(f"{split_name}: duplicate IDs in label file.")

    frame_ids = set(frames["id"].astype(int))
    label_ids = set(labels["id"].astype(int))
    if frame_ids != label_ids:
        raise ValueError(f"{split_name}: frame/label trajectory IDs do not match.")
    if (frames.groupby("id")["loitering"].nunique() > 1).any():
        raise ValueError(f"{split_name}: inconsistent frame labels.")
    if (frames.groupby("id")["clip"].nunique() > 1).any():
        raise ValueError(f"{split_name}: an ID appears in multiple clips.")

    frame_labels = frames.groupby("id")["loitering"].first().astype(int)
    official = labels.set_index("id")["loitering"].astype(int).reindex(frame_labels.index)
    if (frame_labels != official).any():
        raise ValueError(f"{split_name}: frame and official labels disagree.")
    return frames.sort_values(["id", "frame"]).reset_index(drop=True), labels


def fit_coordinate_parameters(train_frames, lower_quantile, upper_quantile):
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Coordinate quantiles must satisfy 0 <= lower < upper <= 1.")
    bounds = {}
    for column in ("x_transform", "y_transform"):
        bounds[column] = {
            "lower": float(train_frames[column].quantile(lower_quantile)),
            "upper": float(train_frames[column].quantile(upper_quantile)),
        }
    return bounds


def clipped_smoothed_coordinates(track, coordinate_bounds, smoothing_window):
    x = track["x_transform"].astype(float).clip(
        coordinate_bounds["x_transform"]["lower"],
        coordinate_bounds["x_transform"]["upper"],
    )
    y = track["y_transform"].astype(float).clip(
        coordinate_bounds["y_transform"]["lower"],
        coordinate_bounds["y_transform"]["upper"],
    )
    if smoothing_window > 1:
        x = x.rolling(smoothing_window, center=True, min_periods=1).median()
        y = y.rolling(smoothing_window, center=True, min_periods=1).median()
    return x.reset_index(drop=True), y.reset_index(drop=True)


def calculate_motion(track, coordinate_bounds, smoothing_window):
    ordered = track.sort_values("frame").reset_index(drop=True)
    x, y = clipped_smoothed_coordinates(
        ordered, coordinate_bounds, smoothing_window
    )
    frame_gap = ordered["frame"].astype(float).diff().fillna(1.0).clip(lower=1.0)
    delta_x = x.diff().fillna(0.0) / frame_gap
    delta_y = y.diff().fillna(0.0) / frame_gap
    speed = np.sqrt(delta_x ** 2 + delta_y ** 2)
    return ordered, x, y, frame_gap, delta_x, delta_y, speed


def fit_behaviour_parameters(train_frames, coordinate_bounds, smoothing_window):
    positive_speeds = []
    for _, track in train_frames.groupby("id", sort=False):
        *_, speed = calculate_motion(track, coordinate_bounds, smoothing_window)
        values = speed.to_numpy(dtype=np.float64)
        positive_speeds.append(values[values > SCALER_EPSILON])
    usable = [values for values in positive_speeds if len(values)]
    if not usable:
        raise ValueError("No positive training movement was available.")
    speeds = np.concatenate(usable)
    pause_threshold = float(np.quantile(speeds, 0.20))

    x_iqr = float(
        train_frames["x_transform"].clip(
            coordinate_bounds["x_transform"]["lower"],
            coordinate_bounds["x_transform"]["upper"],
        ).quantile(0.75)
        - train_frames["x_transform"].clip(
            coordinate_bounds["x_transform"]["lower"],
            coordinate_bounds["x_transform"]["upper"],
        ).quantile(0.25)
    )
    y_iqr = float(
        train_frames["y_transform"].clip(
            coordinate_bounds["y_transform"]["lower"],
            coordinate_bounds["y_transform"]["upper"],
        ).quantile(0.75)
        - train_frames["y_transform"].clip(
            coordinate_bounds["y_transform"]["lower"],
            coordinate_bounds["y_transform"]["upper"],
        ).quantile(0.25)
    )
    spatial_scale = max(float(np.hypot(x_iqr, y_iqr)), SCALER_EPSILON)
    return {
        "pause_speed_threshold": max(pause_threshold, SCALER_EPSILON),
        "revisit_radius": 0.01 * spatial_scale,
        "minimum_turn_speed": max(pause_threshold, SCALER_EPSILON),
    }


def create_track_features(
    track,
    coordinate_bounds,
    behaviour_parameters,
    smoothing_window,
    sequence_length,
):
    ordered, x, y, frame_gap, delta_x, delta_y, speed = calculate_motion(
        track, coordinate_bounds, smoothing_window
    )
    count = len(ordered)
    acceleration = speed.diff().fillna(0.0) / frame_gap
    direction = np.arctan2(delta_y.to_numpy(), delta_x.to_numpy())
    direction_sin = np.sin(direction)
    direction_cos = np.cos(direction)

    raw_angle_change = np.zeros(count, dtype=np.float64)
    if count > 1:
        difference = np.diff(direction)
        wrapped = np.abs((difference + np.pi) % (2.0 * np.pi) - np.pi)
        raw_angle_change[1:] = wrapped

    moving = speed.to_numpy() > behaviour_parameters["minimum_turn_speed"]
    previous_moving = np.concatenate(([False], moving[:-1]))
    valid_turn = moving & previous_moving
    angular_change = np.where(valid_turn, raw_angle_change / np.pi, 0.0)
    direction_change = valid_turn & (raw_angle_change >= DIRECTION_CHANGE_RADIANS)
    cumulative_direction_changes = np.cumsum(direction_change.astype(np.float64))

    step_distance = np.sqrt(x.diff().fillna(0.0) ** 2 + y.diff().fillna(0.0) ** 2)
    cumulative_path = step_distance.cumsum().to_numpy(dtype=np.float64)
    relative_x = x - float(x.iloc[0])
    relative_y = y - float(y.iloc[0])
    displacement = np.sqrt(relative_x ** 2 + relative_y ** 2).to_numpy(dtype=np.float64)
    path_efficiency = np.divide(
        displacement,
        cumulative_path,
        out=np.zeros(count, dtype=np.float64),
        where=cumulative_path > SCALER_EPSILON,
    )

    activity_width = (x.cummax() - x.cummin()).to_numpy(dtype=np.float64)
    activity_height = (y.cummax() - y.cummin()).to_numpy(dtype=np.float64)
    confinement_area = activity_width * activity_height

    pause_indicator = (
        speed.to_numpy() <= behaviour_parameters["pause_speed_threshold"]
    ).astype(np.float64)
    pause_indicator[0] = 0.0
    denominator = np.arange(1, count + 1, dtype=np.float64)
    cumulative_pause_ratio = np.cumsum(pause_indicator) / denominator

    revisit_candidate = (
        (displacement <= behaviour_parameters["revisit_radius"])
        & (cumulative_path >= 2.0 * behaviour_parameters["revisit_radius"])
    )
    revisit_event = revisit_candidate & ~np.concatenate(([False], revisit_candidate[:-1]))
    revisit_count = np.cumsum(revisit_event.astype(np.float64))

    occlusion = ordered["occlusion"].astype(float).clip(0.0, 1.0).to_numpy()
    cumulative_occlusion_ratio = np.cumsum(occlusion) / denominator
    elapsed_frames = (
        ordered["frame"].astype(float) - float(ordered["frame"].iloc[0])
    ).to_numpy()
    observed_fraction = np.minimum(denominator / float(sequence_length), 1.0)

    features = pd.DataFrame({
        "absolute_x": x.to_numpy(),
        "absolute_y": y.to_numpy(),
        "relative_x": relative_x.to_numpy(),
        "relative_y": relative_y.to_numpy(),
        "delta_x": delta_x.to_numpy(),
        "delta_y": delta_y.to_numpy(),
        "speed": speed.to_numpy(),
        "acceleration": acceleration.to_numpy(),
        "direction_sin": direction_sin,
        "direction_cos": direction_cos,
        "angular_change": angular_change,
        "cumulative_path_log": np.log1p(cumulative_path),
        "displacement_log": np.log1p(displacement),
        "path_efficiency": np.clip(path_efficiency, 0.0, 1.0),
        "activity_width_log": np.log1p(activity_width),
        "activity_height_log": np.log1p(activity_height),
        "confinement_area_log": np.log1p(confinement_area),
        "pause_indicator": pause_indicator,
        "cumulative_pause_ratio": cumulative_pause_ratio,
        "cumulative_direction_changes_log": np.log1p(cumulative_direction_changes),
        "revisit_count_log": np.log1p(revisit_count),
        "occlusion": occlusion,
        "cumulative_occlusion_ratio": cumulative_occlusion_ratio,
        "elapsed_frames_log": np.log1p(elapsed_frames),
        "observed_fraction": observed_fraction,
    })
    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            f"Non-finite features for trajectory {int(ordered['id'].iloc[0])}."
        )
    return features


def build_feature_cache(
    frames,
    split_name,
    coordinate_bounds,
    behaviour_parameters,
    smoothing_window,
    sequence_length,
):
    cache = {}
    manifest_rows = []
    for trajectory_id, track in frames.groupby("id", sort=True):
        track = track.sort_values("frame")
        features = create_track_features(
            track,
            coordinate_bounds,
            behaviour_parameters,
            smoothing_window,
            sequence_length,
        )
        frame_gaps = track["frame"].astype(int).diff().fillna(1)
        cache[int(trajectory_id)] = features[FEATURE_COLUMNS].to_numpy(np.float32)
        label = int(track["loitering"].iloc[0])
        manifest_rows.append({
            "dataset_source": "WACV24_Loitering",
            "feature_version": "v2_robust_behavioural",
            "split": split_name,
            "trajectory_id": int(trajectory_id),
            "folder": str(track["folder"].iloc[0]),
            "clip": str(track["clip"].iloc[0]),
            "label": label,
            "label_name": "potential_loitering" if label == 1 else "normal",
            "observation_count": len(track),
            "first_frame": int(track["frame"].min()),
            "last_frame": int(track["frame"].max()),
            "missing_frame_count": int(np.maximum(frame_gaps.to_numpy() - 1, 0).sum()),
            "mean_occlusion": round(float(track["occlusion"].mean()), 6),
        })
    return cache, pd.DataFrame(manifest_rows)


def fit_robust_scaler(feature_cache):
    if not feature_cache:
        raise ValueError("The training feature cache is empty.")
    values = np.concatenate(list(feature_cache.values()), axis=0).astype(np.float64)
    median = np.median(values, axis=0)
    q1 = np.quantile(values, 0.25, axis=0)
    q3 = np.quantile(values, 0.75, axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr < SCALER_EPSILON, 1.0, iqr)
    return median.astype(np.float32), iqr.astype(np.float32)


def create_padded_arrays(feature_cache, manifest, median, iqr, sequence_length):
    track_count = len(manifest)
    sequences = np.zeros(
        (track_count, sequence_length, len(FEATURE_COLUMNS)), dtype=np.float32
    )
    masks = np.zeros((track_count, sequence_length), dtype=np.float32)
    labels = np.zeros(track_count, dtype=np.int64)
    trajectory_ids = np.zeros(track_count, dtype=np.int64)
    lengths = np.zeros(track_count, dtype=np.int64)

    for index, row in enumerate(manifest.itertuples(index=False)):
        trajectory_id = int(row.trajectory_id)
        raw = feature_cache[trajectory_id]
        retained = min(len(raw), sequence_length)
        scaled = (raw[:retained] - median) / iqr
        scaled = np.clip(scaled, -10.0, 10.0)
        sequences[index, :retained] = scaled
        masks[index, :retained] = 1.0
        labels[index] = int(row.label)
        trajectory_ids[index] = trajectory_id
        lengths[index] = retained

    return {
        "sequences": sequences,
        "masks": masks,
        "labels": labels,
        "trajectory_ids": trajectory_ids,
        "lengths": lengths,
        "feature_names": np.asarray(FEATURE_COLUMNS),
    }


def count_labels(labels):
    labels = np.asarray(labels)
    return {
        "normal": int(np.sum(labels == 0)),
        "potential_loitering": int(np.sum(labels == 1)),
    }


def main():
    arguments = parse_arguments()
    dataset_root = arguments.wacv_root / "dataset"
    output_directory = (
        arguments.work_root / "data" / "processed" / "wacv_loitering_v2"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Loading official WACV train split...")
    train_frames, _ = load_split(dataset_root, "train")
    print("Loading official WACV test split...")
    test_frames, _ = load_split(dataset_root, "test")

    train_ids = set(train_frames["id"].astype(int))
    test_ids = set(test_frames["id"].astype(int))
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Data leakage: IDs in both splits: {sorted(overlap)[:10]}")

    print("Fitting training-only coordinate clipping parameters...")
    coordinate_bounds = fit_coordinate_parameters(
        train_frames, arguments.lower_quantile, arguments.upper_quantile
    )
    print("Fitting training-only behavioural thresholds...")
    behaviour_parameters = fit_behaviour_parameters(
        train_frames, coordinate_bounds, arguments.smoothing_window
    )

    print("Calculating v2 training trajectory features...")
    train_cache, train_manifest = build_feature_cache(
        train_frames,
        "train",
        coordinate_bounds,
        behaviour_parameters,
        arguments.smoothing_window,
        arguments.sequence_length,
    )
    print("Calculating v2 test trajectory features...")
    test_cache, test_manifest = build_feature_cache(
        test_frames,
        "test",
        coordinate_bounds,
        behaviour_parameters,
        arguments.smoothing_window,
        arguments.sequence_length,
    )

    print("Fitting robust feature scaler using training data only...")
    median, iqr = fit_robust_scaler(train_cache)
    train_arrays = create_padded_arrays(
        train_cache, train_manifest, median, iqr, arguments.sequence_length
    )
    test_arrays = create_padded_arrays(
        test_cache, test_manifest, median, iqr, arguments.sequence_length
    )

    train_npz = output_directory / "wacv_train_sequences_v2.npz"
    test_npz = output_directory / "wacv_test_sequences_v2.npz"
    train_manifest_path = output_directory / "wacv_train_manifest_v2.csv"
    test_manifest_path = output_directory / "wacv_test_manifest_v2.csv"
    scaler_path = output_directory / "wacv_feature_scaler_v2.json"
    metadata_path = output_directory / "wacv_preparation_metadata_v2.json"

    np.savez_compressed(train_npz, **train_arrays)
    np.savez_compressed(test_npz, **test_arrays)
    train_manifest.to_csv(train_manifest_path, index=False)
    test_manifest.to_csv(test_manifest_path, index=False)

    scaler_information = {
        "fit_split": "train_only",
        "method": "median_iqr",
        "post_scale_clip": [-10.0, 10.0],
        "feature_names": FEATURE_COLUMNS,
        "median": dict(zip(FEATURE_COLUMNS, map(float, median))),
        "interquartile_range": dict(zip(FEATURE_COLUMNS, map(float, iqr))),
    }
    scaler_path.write_text(json.dumps(scaler_information, indent=2), encoding="utf-8")

    clipped_train_rows = (
        (train_frames["x_transform"] < coordinate_bounds["x_transform"]["lower"])
        | (train_frames["x_transform"] > coordinate_bounds["x_transform"]["upper"])
        | (train_frames["y_transform"] < coordinate_bounds["y_transform"]["lower"])
        | (train_frames["y_transform"] > coordinate_bounds["y_transform"]["upper"])
    )
    metadata = {
        "dataset": "RS-WACV24_Loitering",
        "feature_version": "v2_robust_behavioural",
        "source_repository": str(arguments.wacv_root.resolve()),
        "official_split_preserved": True,
        "identity_overlap_count": 0,
        "sequence_length": arguments.sequence_length,
        "smoothing_method": "centred_rolling_median",
        "smoothing_window": arguments.smoothing_window,
        "coordinate_clipping_fit_split": "train_only",
        "coordinate_clipping_quantiles": [
            arguments.lower_quantile, arguments.upper_quantile
        ],
        "coordinate_bounds": coordinate_bounds,
        "training_rows_outside_bounds": int(clipped_train_rows.sum()),
        "behaviour_parameters_fit_split": "train_only",
        "behaviour_parameters": behaviour_parameters,
        "feature_names": FEATURE_COLUMNS,
        "scaler": "median_iqr_train_only",
        "train_frame_rows": len(train_frames),
        "test_frame_rows": len(test_frames),
        "train_trajectories": len(train_manifest),
        "test_trajectories": len(test_manifest),
        "train_distribution": count_labels(train_arrays["labels"]),
        "test_distribution": count_labels(test_arrays["labels"]),
        "privacy_note": (
            "Prepared data uses anonymous trajectory IDs and no facial identity."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nWACV v2 loitering preparation completed")
    print("----------------------------------------")
    print(f"Training frame rows: {len(train_frames):,}")
    print(f"Test frame rows: {len(test_frames):,}")
    print(f"Training trajectories: {len(train_manifest):,}")
    print(f"Test trajectories: {len(test_manifest):,}")
    print(f"Training distribution: {count_labels(train_arrays['labels'])}")
    print(f"Test distribution: {count_labels(test_arrays['labels'])}")
    print(f"Sequence shape: {train_arrays['sequences'].shape}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Training rows clipped: {int(clipped_train_rows.sum()):,}")
    print(f"Pause threshold: {behaviour_parameters['pause_speed_threshold']:.6f}")
    print(f"Revisit radius: {behaviour_parameters['revisit_radius']:.6f}")
    print(f"Training arrays: {train_npz}")
    print(f"Test arrays: {test_npz}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
