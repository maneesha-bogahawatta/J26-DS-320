"""Prepare the WACV24 loitering trajectories for sequence modelling.

The script:
1. preserves the official WACV train/test split;
2. validates one label and one clip per trajectory ID;
3. smooths perspective-transformed coordinates;
4. derives movement features;
5. fits standardisation statistics using training data only;
6. creates padded NumPy arrays with validity masks;
7. exports manifests and reproducibility metadata.

Raw videos are not required because the repository already provides
frame-level trajectories and trajectory-level loitering labels.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WACV_ROOT = Path(
    os.environ.get(
        "WACV_LOITERING_ROOT",
        "RS-WACV24_Loitering",
    )
)

DEFAULT_WORK_ROOT = Path(
    os.environ.get(
        "C2_WORK_ROOT",
        ".",
    )
)

SEQUENCE_LENGTH = 120
SMOOTHING_WINDOW = 5
SCALER_EPSILON = 1e-8

FEATURE_COLUMNS = [
    "relative_x",
    "relative_y",
    "delta_x",
    "delta_y",
    "speed",
    "acceleration",
    "turn_sin",
    "turn_cos",
    "occlusion",
    "progress",
]

REQUIRED_FRAME_COLUMNS = {
    "folder",
    "clip",
    "frame",
    "id",
    "class",
    "x",
    "y",
    "width",
    "height",
    "occlusion",
    "x_transform",
    "y_transform",
    "loitering",
}

REQUIRED_LABEL_COLUMNS = {
    "id",
    "loitering",
}


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the official WACV24 loitering "
            "train/test trajectories."
        )
    )

    parser.add_argument(
        "--wacv-root",
        type=Path,
        default=DEFAULT_WACV_ROOT,
        help="Root of the RS-WACV24_Loitering repository.",
    )

    parser.add_argument(
        "--work-root",
        type=Path,
        default=DEFAULT_WORK_ROOT,
        help="Component data/output root.",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Maximum padded sequence length.",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=SMOOTHING_WINDOW,
        help="Centred rolling-median smoothing window.",
    )

    return parser.parse_args()


def validate_columns(dataframe, required_columns, source_name):
    """Ensure that all required columns are available."""

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def load_split(dataset_root, split_name):
    """Load and validate one official WACV split."""

    frame_path = (
        dataset_root
        / f"X_{split_name}.csv"
    )

    label_path = (
        dataset_root
        / f"{split_name}_df.csv"
    )

    if not frame_path.exists():
        raise FileNotFoundError(
            f"Frame file was not found: {frame_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file was not found: {label_path}"
        )

    frames = pd.read_csv(frame_path)
    labels = pd.read_csv(label_path)

    validate_columns(
        frames,
        REQUIRED_FRAME_COLUMNS,
        str(frame_path),
    )

    validate_columns(
        labels,
        REQUIRED_LABEL_COLUMNS,
        str(label_path),
    )

    if frames["id"].isna().any():
        raise ValueError(
            f"{split_name}: missing trajectory IDs."
        )

    if frames["frame"].isna().any():
        raise ValueError(
            f"{split_name}: missing frame numbers."
        )

    if frames[
        ["x_transform", "y_transform"]
    ].isna().any().any():
        raise ValueError(
            f"{split_name}: missing transformed coordinates."
        )

    invalid_frame_labels = (
        ~frames["loitering"].isin([0, 1])
    )

    invalid_track_labels = (
        ~labels["loitering"].isin([0, 1])
    )

    if invalid_frame_labels.any():
        raise ValueError(
            f"{split_name}: frame data contains "
            "labels other than 0 and 1."
        )

    if invalid_track_labels.any():
        raise ValueError(
            f"{split_name}: label data contains "
            "labels other than 0 and 1."
        )

    duplicated_label_ids = (
        labels["id"].duplicated()
    )

    if duplicated_label_ids.any():
        duplicate_ids = (
            labels.loc[
                duplicated_label_ids,
                "id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{split_name}: duplicate IDs in label file: "
            f"{duplicate_ids}"
        )

    frame_ids = set(
        frames["id"].astype(int)
    )

    label_ids = set(
        labels["id"].astype(int)
    )

    if frame_ids != label_ids:
        missing_from_frames = sorted(
            label_ids - frame_ids
        )[:10]

        missing_from_labels = sorted(
            frame_ids - label_ids
        )[:10]

        raise ValueError(
            f"{split_name}: frame/label IDs do not match. "
            f"Missing from frames: {missing_from_frames}; "
            f"missing from labels: {missing_from_labels}"
        )

    frame_label_counts = (
        frames.groupby("id")["loitering"]
        .nunique()
    )

    inconsistent_ids = (
        frame_label_counts[
            frame_label_counts > 1
        ]
        .index
        .tolist()
    )

    if inconsistent_ids:
        raise ValueError(
            f"{split_name}: trajectories with inconsistent "
            f"frame labels: {inconsistent_ids[:10]}"
        )

    frame_labels = (
        frames.groupby("id")["loitering"]
        .first()
        .astype(int)
    )

    official_labels = (
        labels.set_index("id")["loitering"]
        .astype(int)
    )

    official_labels = official_labels.reindex(
        frame_labels.index
    )

    mismatch_mask = (
        frame_labels != official_labels
    )

    if mismatch_mask.any():
        mismatch_ids = (
            frame_labels.index[
                mismatch_mask
            ]
            .tolist()[:10]
        )

        raise ValueError(
            f"{split_name}: labels disagree between "
            f"X_{split_name}.csv and {split_name}_df.csv "
            f"for IDs: {mismatch_ids}"
        )

    multi_clip_counts = (
        frames.groupby("id")["clip"]
        .nunique()
    )

    multi_clip_ids = (
        multi_clip_counts[
            multi_clip_counts > 1
        ]
        .index
        .tolist()
    )

    if multi_clip_ids:
        raise ValueError(
            f"{split_name}: IDs appearing in multiple clips: "
            f"{multi_clip_ids[:10]}"
        )

    frames = frames.sort_values(
        ["id", "frame"]
    ).reset_index(drop=True)

    return frames, labels


def calculate_angle_features(delta_x, delta_y):
    """Return sine and cosine of movement direction."""

    magnitude = np.sqrt(
        delta_x ** 2
        + delta_y ** 2
    )

    turn_sin = np.zeros(
        len(delta_x),
        dtype=np.float64,
    )

    turn_cos = np.ones(
        len(delta_x),
        dtype=np.float64,
    )

    moving = magnitude > 0

    turn_sin[moving] = (
        delta_y[moving]
        / magnitude[moving]
    )

    turn_cos[moving] = (
        delta_x[moving]
        / magnitude[moving]
    )

    return turn_sin, turn_cos


def create_track_features(track, smoothing_window):
    """Create frame-level features for one trajectory."""

    track = track.sort_values(
        "frame"
    ).copy()

    x_coordinates = (
        track["x_transform"]
        .astype(float)
    )

    y_coordinates = (
        track["y_transform"]
        .astype(float)
    )

    if smoothing_window > 1:
        smoothed_x = (
            x_coordinates
            .rolling(
                window=smoothing_window,
                center=True,
                min_periods=1,
            )
            .median()
        )

        smoothed_y = (
            y_coordinates
            .rolling(
                window=smoothing_window,
                center=True,
                min_periods=1,
            )
            .median()
        )
    else:
        smoothed_x = x_coordinates.copy()
        smoothed_y = y_coordinates.copy()

    relative_x = (
        smoothed_x
        - float(smoothed_x.iloc[0])
    )

    relative_y = (
        smoothed_y
        - float(smoothed_y.iloc[0])
    )

    frame_gap = (
        track["frame"]
        .astype(float)
        .diff()
        .fillna(1.0)
    )

    frame_gap = frame_gap.clip(
        lower=1.0
    )

    delta_x = (
        smoothed_x.diff()
        .fillna(0.0)
        / frame_gap
    )

    delta_y = (
        smoothed_y.diff()
        .fillna(0.0)
        / frame_gap
    )

    speed = np.sqrt(
        delta_x ** 2
        + delta_y ** 2
    )

    acceleration = (
        speed.diff()
        .fillna(0.0)
        / frame_gap
    )

    turn_sin, turn_cos = (
        calculate_angle_features(
            delta_x.to_numpy(),
            delta_y.to_numpy(),
        )
    )

    observation_count = len(track)

    if observation_count > 1:
        progress = np.linspace(
            0.0,
            1.0,
            observation_count,
            dtype=np.float64,
        )
    else:
        progress = np.zeros(
            observation_count,
            dtype=np.float64,
        )

    features = pd.DataFrame(
        {
            "relative_x": relative_x.to_numpy(),
            "relative_y": relative_y.to_numpy(),
            "delta_x": delta_x.to_numpy(),
            "delta_y": delta_y.to_numpy(),
            "speed": speed.to_numpy(),
            "acceleration": acceleration.to_numpy(),
            "turn_sin": turn_sin,
            "turn_cos": turn_cos,
            "occlusion": (
                track["occlusion"]
                .astype(float)
                .clip(0.0, 1.0)
                .to_numpy()
            ),
            "progress": progress,
        }
    )

    values = features[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite feature values found for "
            f"trajectory ID {track['id'].iloc[0]}."
        )

    return features


def build_feature_cache(
    frames,
    split_name,
    smoothing_window,
):
    """Calculate features and metadata for every trajectory."""

    cache = {}
    manifest_rows = []

    grouped_tracks = frames.groupby(
        "id",
        sort=True,
    )

    for trajectory_id, track in grouped_tracks:
        track = track.sort_values(
            "frame"
        )

        features = create_track_features(
            track,
            smoothing_window,
        )

        label = int(
            track["loitering"].iloc[0]
        )

        folder = str(
            track["folder"].iloc[0]
        )

        clip = str(
            track["clip"].iloc[0]
        )

        frame_gaps = (
            track["frame"]
            .astype(int)
            .diff()
            .fillna(1)
        )

        missing_frames = int(
            np.maximum(
                frame_gaps.to_numpy() - 1,
                0,
            ).sum()
        )

        cache[int(trajectory_id)] = (
            features[
                FEATURE_COLUMNS
            ].to_numpy(dtype=np.float32)
        )

        manifest_rows.append(
            {
                "dataset_source": (
                    "WACV24_Loitering"
                ),
                "split": split_name,
                "trajectory_id": int(
                    trajectory_id
                ),
                "folder": folder,
                "clip": clip,
                "label": label,
                "label_name": (
                    "potential_loitering"
                    if label == 1
                    else "normal"
                ),
                "observation_count": len(track),
                "first_frame": int(
                    track["frame"].min()
                ),
                "last_frame": int(
                    track["frame"].max()
                ),
                "missing_frame_count": (
                    missing_frames
                ),
                "mean_occlusion": round(
                    float(
                        track["occlusion"].mean()
                    ),
                    6,
                ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    return cache, manifest


def fit_training_scaler(feature_cache):
    """Fit mean and standard deviation using training data only."""

    if not feature_cache:
        raise ValueError(
            "The training feature cache is empty."
        )

    all_training_values = np.concatenate(
        list(feature_cache.values()),
        axis=0,
    ).astype(np.float64)

    mean = all_training_values.mean(
        axis=0
    )

    standard_deviation = (
        all_training_values.std(
            axis=0
        )
    )

    standard_deviation = np.where(
        standard_deviation
        < SCALER_EPSILON,
        1.0,
        standard_deviation,
    )

    return (
        mean.astype(np.float32),
        standard_deviation.astype(np.float32),
    )


def create_padded_arrays(
    feature_cache,
    manifest,
    mean,
    standard_deviation,
    sequence_length,
):
    """Create padded model arrays and validity masks."""

    number_of_tracks = len(manifest)
    number_of_features = len(
        FEATURE_COLUMNS
    )

    sequences = np.zeros(
        (
            number_of_tracks,
            sequence_length,
            number_of_features,
        ),
        dtype=np.float32,
    )

    masks = np.zeros(
        (
            number_of_tracks,
            sequence_length,
        ),
        dtype=np.float32,
    )

    labels = np.zeros(
        number_of_tracks,
        dtype=np.int64,
    )

    trajectory_ids = np.zeros(
        number_of_tracks,
        dtype=np.int64,
    )

    lengths = np.zeros(
        number_of_tracks,
        dtype=np.int64,
    )

    for row_index, row in enumerate(
        manifest.itertuples(index=False)
    ):
        trajectory_id = int(
            row.trajectory_id
        )

        raw_features = (
            feature_cache[
                trajectory_id
            ]
        )

        retained_length = min(
            len(raw_features),
            sequence_length,
        )

        selected_features = (
            raw_features[
                :retained_length
            ]
        )

        standardised_features = (
            selected_features - mean
        ) / standard_deviation

        sequences[
            row_index,
            :retained_length,
            :,
        ] = standardised_features

        masks[
            row_index,
            :retained_length,
        ] = 1.0

        labels[row_index] = int(
            row.label
        )

        trajectory_ids[row_index] = (
            trajectory_id
        )

        lengths[row_index] = (
            retained_length
        )

    return {
        "sequences": sequences,
        "masks": masks,
        "labels": labels,
        "trajectory_ids": trajectory_ids,
        "lengths": lengths,
        "feature_names": np.asarray(
            FEATURE_COLUMNS
        ),
    }


def save_npz(output_path, arrays):
    """Save one prepared dataset split."""

    np.savez_compressed(
        output_path,
        **arrays,
    )


def count_labels(labels):
    """Return binary-label counts."""

    labels = np.asarray(labels)

    return {
        "normal": int(
            np.sum(labels == 0)
        ),
        "potential_loitering": int(
            np.sum(labels == 1)
        ),
    }


def main():
    """Prepare both official WACV splits."""

    arguments = parse_arguments()

    dataset_root = (
        arguments.wacv_root
        / "dataset"
    )

    output_directory = (
        arguments.work_root
        / "data"
        / "processed"
        / "wacv_loitering"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading official WACV train split..."
    )

    train_frames, _ = load_split(
        dataset_root,
        "train",
    )

    print(
        "Loading official WACV test split..."
    )

    test_frames, _ = load_split(
        dataset_root,
        "test",
    )

    train_ids = set(
        train_frames["id"].astype(int)
    )

    test_ids = set(
        test_frames["id"].astype(int)
    )

    overlapping_ids = (
        train_ids & test_ids
    )

    if overlapping_ids:
        raise ValueError(
            "Data leakage detected: trajectory IDs "
            "appear in both splits: "
            f"{sorted(overlapping_ids)[:10]}"
        )

    print(
        "Calculating training trajectory features..."
    )

    train_cache, train_manifest = (
        build_feature_cache(
            train_frames,
            "train",
            arguments.smoothing_window,
        )
    )

    print(
        "Calculating test trajectory features..."
    )

    test_cache, test_manifest = (
        build_feature_cache(
            test_frames,
            "test",
            arguments.smoothing_window,
        )
    )

    print(
        "Fitting feature scaler using training data only..."
    )

    mean, standard_deviation = (
        fit_training_scaler(
            train_cache
        )
    )

    train_arrays = create_padded_arrays(
        train_cache,
        train_manifest,
        mean,
        standard_deviation,
        arguments.sequence_length,
    )

    test_arrays = create_padded_arrays(
        test_cache,
        test_manifest,
        mean,
        standard_deviation,
        arguments.sequence_length,
    )

    train_npz = (
        output_directory
        / "wacv_train_sequences.npz"
    )

    test_npz = (
        output_directory
        / "wacv_test_sequences.npz"
    )

    train_manifest_path = (
        output_directory
        / "wacv_train_manifest.csv"
    )

    test_manifest_path = (
        output_directory
        / "wacv_test_manifest.csv"
    )

    scaler_path = (
        output_directory
        / "wacv_feature_scaler.json"
    )

    metadata_path = (
        output_directory
        / "wacv_preparation_metadata.json"
    )

    save_npz(
        train_npz,
        train_arrays,
    )

    save_npz(
        test_npz,
        test_arrays,
    )

    train_manifest.to_csv(
        train_manifest_path,
        index=False,
    )

    test_manifest.to_csv(
        test_manifest_path,
        index=False,
    )

    scaler_information = {
        "fit_split": "train_only",
        "feature_names": FEATURE_COLUMNS,
        "mean": {
            feature_name: float(value)
            for feature_name, value in zip(
                FEATURE_COLUMNS,
                mean,
            )
        },
        "standard_deviation": {
            feature_name: float(value)
            for feature_name, value in zip(
                FEATURE_COLUMNS,
                standard_deviation,
            )
        },
    }

    with scaler_path.open(
        "w",
        encoding="utf-8",
    ) as scaler_stream:
        json.dump(
            scaler_information,
            scaler_stream,
            indent=2,
        )

    metadata = {
        "dataset": "RS-WACV24_Loitering",
        "source_repository": str(
            arguments.wacv_root.resolve()
        ),
        "official_split_preserved": True,
        "identity_overlap_count": 0,
        "sequence_length": (
            arguments.sequence_length
        ),
        "smoothing_method": (
            "centred_rolling_median"
        ),
        "smoothing_window": (
            arguments.smoothing_window
        ),
        "coordinate_source": [
            "x_transform",
            "y_transform",
        ],
        "feature_names": FEATURE_COLUMNS,
        "scaler_fit_split": "train_only",
        "train_frame_rows": int(
            len(train_frames)
        ),
        "test_frame_rows": int(
            len(test_frames)
        ),
        "train_trajectories": int(
            len(train_manifest)
        ),
        "test_trajectories": int(
            len(test_manifest)
        ),
        "train_distribution": count_labels(
            train_arrays["labels"]
        ),
        "test_distribution": count_labels(
            test_arrays["labels"]
        ),
        "privacy_note": (
            "The prepared dataset uses anonymous "
            "trajectory IDs and contains no facial identity."
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_stream:
        json.dump(
            metadata,
            metadata_stream,
            indent=2,
        )

    print()
    print("WACV loitering preparation completed")
    print("------------------------------------")
    print(
        f"Training frame rows: "
        f"{len(train_frames):,}"
    )
    print(
        f"Test frame rows: "
        f"{len(test_frames):,}"
    )
    print(
        f"Training trajectories: "
        f"{len(train_manifest):,}"
    )
    print(
        f"Test trajectories: "
        f"{len(test_manifest):,}"
    )
    print(
        "Training distribution: "
        f"{count_labels(train_arrays['labels'])}"
    )
    print(
        "Test distribution: "
        f"{count_labels(test_arrays['labels'])}"
    )
    print(
        f"Sequence shape: "
        f"{train_arrays['sequences'].shape}"
    )
    print(
        f"Identity overlap: "
        f"{len(overlapping_ids)}"
    )
    print(
        f"Training arrays: {train_npz}"
    )
    print(
        f"Test arrays: {test_npz}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()