"""Train and evaluate a lightweight multi-scale trajectory Transformer.

The official test set is retained for final evaluation. A stratified
validation subset is created only from the official training split.

The script supports:
- padded variable-length trajectory sequences;
- masked self-attention at short, medium and long temporal scales;
- class-weighted binary loss;
- validation-based early stopping;
- validation-based decision-threshold selection;
- precision, recall, F1, PR-AUC and confusion-matrix reporting;
- reproducible experiment metadata.
"""

import argparse
import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


WORK_ROOT = Path(
    os.environ.get(
        "C2_WORK_ROOT",
        ".",
    )
)

DEFAULT_SEED = 42


def parse_arguments():
    """Parse training options."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a class-weighted multi-scale Transformer using "
            "robust, behaviour-rich WACV v2 trajectories."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            WORK_ROOT
            / "data"
            / "processed"
            / "wacv_loitering_v2"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            WORK_ROOT
            / "outputs"
            / "wacv_multiscale_transformer"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--feedforward-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--minimum-length",
        type=int,
        default=2,
        help=(
            "Exclude trajectories shorter than this "
            "number of observations."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--exclude-occlusion-features",
        action="store_true",
        help=(
            "Remove occlusion and cumulative-occlusion features "
            "for a dataset-shortcut ablation experiment."
        ),
    )

    return parser.parse_args()


def set_reproducible_seed(seed):
    """Set reproducible random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
    except Exception:
        pass


def load_prepared_split(path, minimum_length):
    """Load and validate one prepared NPZ split."""

    if not path.exists():
        raise FileNotFoundError(
            f"Prepared split was not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    required_arrays = {
        "sequences",
        "masks",
        "labels",
        "trajectory_ids",
        "lengths",
        "feature_names",
    }

    missing_arrays = (
        required_arrays
        - set(data.files)
    )

    if missing_arrays:
        raise ValueError(
            f"{path} is missing arrays: "
            f"{sorted(missing_arrays)}"
        )

    sequences = data[
        "sequences"
    ].astype(np.float32)

    masks = data[
        "masks"
    ].astype(np.float32)

    labels = data[
        "labels"
    ].astype(np.int64)

    trajectory_ids = data[
        "trajectory_ids"
    ].astype(np.int64)

    lengths = data[
        "lengths"
    ].astype(np.int64)

    feature_names = data[
        "feature_names"
    ].astype(str)

    number_of_samples = len(labels)

    if sequences.ndim != 3:
        raise ValueError(
            "Sequences must have shape "
            "[samples, time, features]."
        )

    if masks.shape != sequences.shape[:2]:
        raise ValueError(
            "Mask shape does not match sequence shape."
        )

    if not (
        len(trajectory_ids)
        == len(lengths)
        == number_of_samples
        == len(sequences)
    ):
        raise ValueError(
            "Prepared arrays have inconsistent sample counts."
        )

    if not np.isfinite(sequences).all():
        raise ValueError(
            f"Non-finite sequence values found in {path}."
        )

    if not set(np.unique(labels)).issubset(
        {0, 1}
    ):
        raise ValueError(
            f"Labels other than 0/1 found in {path}."
        )

    expected_lengths = masks.sum(
        axis=1
    ).astype(np.int64)

    if not np.array_equal(
        expected_lengths,
        lengths,
    ):
        raise ValueError(
            f"Mask-derived lengths disagree with "
            f"stored lengths in {path}."
        )

    keep = (
        lengths >= minimum_length
    )

    return {
        "sequences": sequences[keep],
        "masks": masks[keep],
        "labels": labels[keep],
        "trajectory_ids": (
            trajectory_ids[keep]
        ),
        "lengths": lengths[keep],
        "feature_names": feature_names,
        "excluded_count": int(
            np.sum(~keep)
        ),
    }


def remove_named_features(prepared_data, excluded_names):
    """Return prepared arrays without selected feature columns."""

    names = prepared_data["feature_names"].tolist()
    missing = sorted(set(excluded_names) - set(names))
    if missing:
        raise ValueError(
            f"Cannot exclude missing features: {missing}"
        )

    keep_indices = [
        index
        for index, name in enumerate(names)
        if name not in excluded_names
    ]

    updated = dict(prepared_data)
    updated["sequences"] = prepared_data["sequences"][:, :, keep_indices]
    updated["feature_names"] = prepared_data["feature_names"][keep_indices]
    return updated


def calculate_length_group_metrics(labels, probabilities, lengths, threshold):
    """Calculate test metrics for predefined trajectory-length groups."""

    groups = [
        (2, 10, "2-10"),
        (11, 20, "11-20"),
        (21, 40, "21-40"),
        (41, 60, "41-60"),
        (61, 90, "61-90"),
        (91, 120, "91-120"),
    ]
    rows = []
    for lower, upper, label in groups:
        selected = (lengths >= lower) & (lengths <= upper)
        if not selected.any():
            continue
        metrics = calculate_metrics(
            labels[selected], probabilities[selected], threshold
        )
        rows.append({
            "length_group": label,
            "minimum_length": lower,
            "maximum_length": upper,
            **metrics,
        })
    return rows


class TrajectoryDataset(Dataset):
    """PyTorch dataset containing padded trajectories."""

    def __init__(
        self,
        sequences,
        labels,
        lengths,
        trajectory_ids,
        indices=None,
    ):
        if indices is None:
            indices = np.arange(
                len(labels)
            )

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        self.sequences = torch.from_numpy(
            sequences[indices]
        ).float()

        self.labels = torch.from_numpy(
            labels[indices]
        ).float()

        self.lengths = torch.from_numpy(
            lengths[indices]
        ).long()

        self.trajectory_ids = (
            torch.from_numpy(
                trajectory_ids[indices]
            ).long()
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.labels[index],
            self.lengths[index],
            self.trajectory_ids[index],
        )


class SinusoidalPositionEncoding(nn.Module):
    """Add deterministic temporal-position information to a sequence."""

    def __init__(self, model_size, maximum_length=120):
        super().__init__()

        positions = torch.arange(
            maximum_length,
            dtype=torch.float32,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_size, 2, dtype=torch.float32)
            * (-np.log(10000.0) / model_size)
        )

        encoding = torch.zeros(maximum_length, model_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, sequence):
        return sequence + self.encoding[:, : sequence.shape[1]]


class MultiScaleTransformerClassifier(nn.Module):
    """Masked Transformer over native, 2x and 4x temporal resolutions."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_heads,
        feedforward_size,
        dropout,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden-size must be divisible by num-heads."
            )

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.position_encoding = SinusoidalPositionEncoding(hidden_size)
        self.scales = (1, 2, 4)

        def create_encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(
                layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

        self.encoders = nn.ModuleList(
            [create_encoder() for _ in self.scales]
        )
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                hidden_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size // 2,
                1,
            ),
        )

    @staticmethod
    def create_valid_mask(lengths, sequence_length):
        positions = torch.arange(
            sequence_length,
            device=lengths.device,
        ).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    def downsample(self, sequence, lengths, stride):
        if stride == 1:
            return sequence, lengths

        batch_size, sequence_length, feature_count = sequence.shape
        padded_length = int(np.ceil(sequence_length / stride) * stride)
        padding = padded_length - sequence_length
        if padding:
            sequence = nn.functional.pad(sequence, (0, 0, 0, padding))

        valid = self.create_valid_mask(lengths, padded_length).unsqueeze(-1)
        sequence = sequence.reshape(
            batch_size, padded_length // stride, stride, feature_count
        )
        valid = valid.reshape(
            batch_size, padded_length // stride, stride, 1
        )
        counts = valid.sum(dim=2).clamp_min(1)
        pooled = (sequence * valid).sum(dim=2) / counts
        pooled_lengths = torch.div(
            lengths + stride - 1,
            stride,
            rounding_mode="floor",
        )
        return pooled, pooled_lengths

    def forward(self, sequences, lengths):
        projected = self.input_projection(sequences)
        scale_vectors = []

        for stride, encoder in zip(self.scales, self.encoders):
            scaled, scaled_lengths = self.downsample(
                projected, lengths, stride
            )
            scaled = self.position_encoding(scaled)
            valid_mask = self.create_valid_mask(
                scaled_lengths, scaled.shape[1]
            )
            encoded = encoder(
                scaled,
                src_key_padding_mask=~valid_mask,
            )
            valid_values = valid_mask.unsqueeze(-1)
            pooled = (encoded * valid_values).sum(dim=1) / (
                valid_values.sum(dim=1).clamp_min(1)
            )
            scale_vectors.append(pooled)

        stacked_scales = torch.stack(scale_vectors, dim=1)
        scale_weights = torch.softmax(
            self.scale_attention(stacked_scales),
            dim=1,
        )
        fused = (stacked_scales * scale_weights).sum(dim=1)

        logits = (
            self.classifier(
                fused
            )
            .squeeze(1)
        )

        return logits


def calculate_positive_weight(labels):
    """Calculate training-only positive-class weight."""

    negative_count = int(
        np.sum(labels == 0)
    )

    positive_count = int(
        np.sum(labels == 1)
    )

    if positive_count == 0:
        raise ValueError(
            "Training subset contains no positive samples."
        )

    return (
        negative_count
        / positive_count
    )


def run_training_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    """Train for one epoch."""

    model.train()

    running_loss = 0.0
    sample_count = 0

    for (
        sequences,
        labels,
        lengths,
        _,
    ) in data_loader:
        sequences = sequences.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            sequences,
            lengths,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = len(labels)

        running_loss += (
            float(loss.item())
            * batch_size
        )

        sample_count += batch_size

    return (
        running_loss
        / max(sample_count, 1)
    )


def collect_predictions(
    model,
    data_loader,
    criterion,
    device,
):
    """Collect probabilities, labels and IDs."""

    model.eval()

    running_loss = 0.0
    sample_count = 0

    all_probabilities = []
    all_labels = []
    all_ids = []
    all_lengths = []

    with torch.no_grad():
        for (
            sequences,
            labels,
            lengths,
            trajectory_ids,
        ) in data_loader:
            sequences = sequences.to(
                device
            )

            labels_device = labels.to(
                device
            )

            logits = model(
                sequences,
                lengths,
            )

            loss = criterion(
                logits,
                labels_device,
            )

            probabilities = torch.sigmoid(
                logits
            )

            batch_size = len(labels)

            running_loss += (
                float(loss.item())
                * batch_size
            )

            sample_count += batch_size

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy().astype(
                    np.int64
                )
            )

            all_ids.append(
                trajectory_ids.numpy()
            )

            all_lengths.append(
                lengths.numpy()
            )

    return {
        "loss": (
            running_loss
            / max(sample_count, 1)
        ),
        "probabilities": np.concatenate(
            all_probabilities
        ),
        "labels": np.concatenate(
            all_labels
        ),
        "trajectory_ids": np.concatenate(
            all_ids
        ),
        "lengths": np.concatenate(
            all_lengths
        ),
    }


def select_f1_threshold(
    labels,
    probabilities,
):
    """Select threshold using validation F1 only."""

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            labels,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        return 0.5, 0.0

    numerator = (
        2.0
        * precision_values[:-1]
        * recall_values[:-1]
    )

    denominator = (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-12
    )

    f1_values = (
        numerator
        / denominator
    )

    best_index = int(
        np.nanargmax(
            f1_values
        )
    )

    return (
        float(thresholds[best_index]),
        float(f1_values[best_index]),
    )


def calculate_metrics(
    labels,
    probabilities,
    threshold,
):
    """Calculate binary-classification metrics."""

    predictions = (
        probabilities
        >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    true_negative = int(
        matrix[0, 0]
    )

    false_positive = int(
        matrix[0, 1]
    )

    false_negative = int(
        matrix[1, 0]
    )

    true_positive = int(
        matrix[1, 1]
    )

    specificity = (
        true_negative
        / max(
            true_negative
            + false_positive,
            1,
        )
    )

    return {
        "threshold": round(
            float(threshold),
            6,
        ),
        "accuracy": round(
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),
            6,
        ),
        "precision": round(
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            6,
        ),
        "recall": round(
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            6,
        ),
        "f1_score": round(
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),
            6,
        ),
        "pr_auc": round(
            float(
                average_precision_score(
                    labels,
                    probabilities,
                )
            ),
            6,
        ),
        "specificity": round(
            float(specificity),
            6,
        ),
        "true_negative": (
            true_negative
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
        "true_positive": (
            true_positive
        ),
        "sample_count": int(
            len(labels)
        ),
        "positive_count": int(
            np.sum(labels == 1)
        ),
        "negative_count": int(
            np.sum(labels == 0)
        ),
    }


def benchmark_inference(
    model,
    data_loader,
    device,
):
    """Measure average CPU/GPU model-inference latency."""

    model.eval()

    elapsed_seconds = 0.0
    sample_count = 0

    with torch.no_grad():
        for (
            sequences,
            _,
            lengths,
            _,
        ) in data_loader:
            sequences = sequences.to(
                device
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = (
                time.perf_counter()
            )

            _ = model(
                sequences,
                lengths,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_seconds += (
                time.perf_counter()
                - start_time
            )

            sample_count += len(
                sequences
            )

    milliseconds_per_trajectory = (
        elapsed_seconds
        / max(sample_count, 1)
        * 1000.0
    )

    trajectories_per_second = (
        sample_count
        / max(
            elapsed_seconds,
            1e-12,
        )
    )

    return {
        "total_seconds": round(
            elapsed_seconds,
            6,
        ),
        "milliseconds_per_trajectory": round(
            milliseconds_per_trajectory,
            6,
        ),
        "trajectories_per_second": round(
            trajectories_per_second,
            3,
        ),
        "sample_count": int(
            sample_count
        ),
    }


def main():
    """Train and evaluate the multi-scale Transformer."""

    arguments = parse_arguments()

    set_reproducible_seed(
        arguments.seed
    )

    if (
        arguments.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    device = torch.device(
        arguments.device
    )

    train_path = (
        arguments.data_root
        / "wacv_train_sequences_v2.npz"
    )

    test_path = (
        arguments.data_root
        / "wacv_test_sequences_v2.npz"
    )

    arguments.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading prepared WACV data...")

    training_data = load_prepared_split(
        train_path,
        arguments.minimum_length,
    )

    test_data = load_prepared_split(
        test_path,
        arguments.minimum_length,
    )

    if not np.array_equal(
        training_data["feature_names"],
        test_data["feature_names"],
    ):
        raise ValueError(
            "Training and test feature names or ordering differ."
        )

    excluded_feature_names = []
    if arguments.exclude_occlusion_features:
        excluded_feature_names = [
            "occlusion",
            "cumulative_occlusion_ratio",
        ]
        training_data = remove_named_features(
            training_data,
            excluded_feature_names,
        )
        test_data = remove_named_features(
            test_data,
            excluded_feature_names,
        )

    training_ids = set(
        training_data[
            "trajectory_ids"
        ].tolist()
    )

    test_ids = set(
        test_data[
            "trajectory_ids"
        ].tolist()
    )

    overlap = (
        training_ids
        & test_ids
    )

    if overlap:
        raise ValueError(
            "Train/test trajectory-ID leakage detected."
        )

    all_training_indices = np.arange(
        len(
            training_data[
                "labels"
            ]
        )
    )

    (
        train_indices,
        validation_indices,
    ) = train_test_split(
        all_training_indices,
        test_size=(
            arguments.validation_size
        ),
        random_state=arguments.seed,
        stratify=(
            training_data[
                "labels"
            ]
        ),
    )

    train_labels = (
        training_data[
            "labels"
        ][train_indices]
    )

    validation_labels = (
        training_data[
            "labels"
        ][validation_indices]
    )

    positive_weight = (
        calculate_positive_weight(
            train_labels
        )
    )

    train_dataset = (
        TrajectoryDataset(
            training_data[
                "sequences"
            ],
            training_data[
                "labels"
            ],
            training_data[
                "lengths"
            ],
            training_data[
                "trajectory_ids"
            ],
            train_indices,
        )
    )

    validation_dataset = (
        TrajectoryDataset(
            training_data[
                "sequences"
            ],
            training_data[
                "labels"
            ],
            training_data[
                "lengths"
            ],
            training_data[
                "trajectory_ids"
            ],
            validation_indices,
        )
    )

    test_dataset = TrajectoryDataset(
        test_data[
            "sequences"
        ],
        test_data[
            "labels"
        ],
        test_data[
            "lengths"
        ],
        test_data[
            "trajectory_ids"
        ],
    )

    generator = torch.Generator()
    generator.manual_seed(
        arguments.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            arguments.batch_size
        ),
        shuffle=True,
        num_workers=(
            arguments.num_workers
        ),
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            arguments.batch_size
        ),
        shuffle=False,
        num_workers=(
            arguments.num_workers
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=(
            arguments.batch_size
        ),
        shuffle=False,
        num_workers=(
            arguments.num_workers
        ),
    )

    input_size = int(
        training_data[
            "sequences"
        ].shape[2]
    )

    model = MultiScaleTransformerClassifier(
        input_size=input_size,
        hidden_size=(
            arguments.hidden_size
        ),
        num_layers=(
            arguments.num_layers
        ),
        num_heads=(
            arguments.num_heads
        ),
        feedforward_size=(
            arguments.feedforward_size
        ),
        dropout=arguments.dropout,
    ).to(device)

    pos_weight_tensor = torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=device,
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=(
                pos_weight_tensor
            )
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=(
            arguments.weight_decay
        ),
    )

    best_state = None
    best_validation_pr_auc = (
        float("-inf")
    )

    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []

    print()
    print("WACV multi-scale Transformer feature training")
    print("-----------------------------")
    print(
        f"Device: {device}"
    )
    print(
        f"Training samples: "
        f"{len(train_dataset):,}"
    )
    print(
        f"Validation samples: "
        f"{len(validation_dataset):,}"
    )
    print(
        f"Official test samples: "
        f"{len(test_dataset):,}"
    )
    print(
        f"Input features: "
        f"{input_size}"
    )
    print(
        "Temporal scales: 1x, 2x, 4x"
    )
    print(
        "Occlusion features: "
        + (
            "excluded (ablation)"
            if excluded_feature_names
            else "included"
        )
    )
    print(
        f"Positive class weight: "
        f"{positive_weight:.4f}"
    )

    training_start_time = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        arguments.epochs + 1,
    ):
        epoch_start_time = (
            time.perf_counter()
        )

        training_loss = (
            run_training_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        validation_output = (
            collect_predictions(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        validation_pr_auc = (
            average_precision_score(
                validation_output[
                    "labels"
                ],
                validation_output[
                    "probabilities"
                ],
            )
        )

        (
            validation_threshold,
            validation_f1,
        ) = select_f1_threshold(
            validation_output[
                "labels"
            ],
            validation_output[
                "probabilities"
            ],
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_start_time
        )

        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": (
                    training_loss
                ),
                "validation_loss": (
                    validation_output[
                        "loss"
                    ]
                ),
                "validation_pr_auc": (
                    validation_pr_auc
                ),
                "validation_f1": (
                    validation_f1
                ),
                "validation_threshold": (
                    validation_threshold
                ),
                "epoch_seconds": (
                    epoch_seconds
                ),
            }
        )

        print(
            f"Epoch {epoch:02d}/{arguments.epochs} | "
            f"train loss={training_loss:.4f} | "
            f"val loss={validation_output['loss']:.4f} | "
            f"val PR-AUC={validation_pr_auc:.4f} | "
            f"val F1={validation_f1:.4f} | "
            f"{epoch_seconds:.1f}s"
        )

        if (
            validation_pr_auc
            > best_validation_pr_auc
            + 1e-6
        ):
            best_validation_pr_auc = (
                float(
                    validation_pr_auc
                )
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= arguments.patience
        ):
            print(
                "Early stopping activated."
            )
            break

    total_training_seconds = (
        time.perf_counter()
        - training_start_time
    )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model."
        )

    model.load_state_dict(
        best_state
    )

    validation_output = (
        collect_predictions(
            model,
            validation_loader,
            criterion,
            device,
        )
    )

    (
        selected_threshold,
        validation_best_f1,
    ) = select_f1_threshold(
        validation_output[
            "labels"
        ],
        validation_output[
            "probabilities"
        ],
    )

    validation_metrics = (
        calculate_metrics(
            validation_output[
                "labels"
            ],
            validation_output[
                "probabilities"
            ],
            selected_threshold,
        )
    )

    test_output = collect_predictions(
        model,
        test_loader,
        criterion,
        device,
    )

    test_metrics = (
        calculate_metrics(
            test_output[
                "labels"
            ],
            test_output[
                "probabilities"
            ],
            selected_threshold,
        )
    )

    length_group_metrics = calculate_length_group_metrics(
        test_output["labels"],
        test_output["probabilities"],
        test_output["lengths"],
        selected_threshold,
    )

    inference_benchmark = (
        benchmark_inference(
            model,
            test_loader,
            device,
        )
    )

    model_path = (
        arguments.output_root
        / "wacv_multiscale_transformer_best.pt"
    )

    metrics_path = (
        arguments.output_root
        / "wacv_multiscale_transformer_metrics.json"
    )

    history_path = (
        arguments.output_root
        / "wacv_multiscale_transformer_training_history.csv"
    )

    predictions_path = (
        arguments.output_root
        / "wacv_multiscale_transformer_test_predictions.csv"
    )

    length_metrics_path = (
        arguments.output_root
        / "wacv_multiscale_transformer_length_metrics.csv"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "model_class": (
                "MultiScaleTransformerClassifier"
            ),
            "input_size": input_size,
            "hidden_size": (
                arguments.hidden_size
            ),
            "num_layers": (
                arguments.num_layers
            ),
            "num_heads": (
                arguments.num_heads
            ),
            "feedforward_size": (
                arguments.feedforward_size
            ),
            "temporal_scales": [1, 2, 4],
            "dropout": (
                arguments.dropout
            ),
            "decision_threshold": (
                selected_threshold
            ),
            "feature_names": (
                training_data[
                    "feature_names"
                ].tolist()
            ),
            "minimum_length": (
                arguments.minimum_length
            ),
            "seed": arguments.seed,
        },
        model_path,
    )

    history_dataframe = (
        pd.DataFrame(
            history_rows
        )
    )

    history_dataframe.to_csv(
        history_path,
        index=False,
    )

    test_predictions = (
        test_output[
            "probabilities"
        ]
        >= selected_threshold
    ).astype(np.int64)

    prediction_dataframe = pd.DataFrame(
        {
            "trajectory_id": (
                test_output[
                    "trajectory_ids"
                ]
            ),
            "sequence_length": (
                test_output[
                    "lengths"
                ]
            ),
            "actual_label": (
                test_output[
                    "labels"
                ]
            ),
            "predicted_probability": (
                test_output[
                    "probabilities"
                ]
            ),
            "predicted_label": (
                test_predictions
            ),
            "decision_threshold": (
                selected_threshold
            ),
        }
    )

    prediction_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    pd.DataFrame(length_group_metrics).to_csv(
        length_metrics_path,
        index=False,
    )

    experiment_results = {
        "model": (
            "Multi-scale Transformer with robust behavioural trajectory features"
        ),
        "dataset": (
            "RS-WACV24_Loitering"
        ),
        "official_test_split_preserved": (
            True
        ),
        "official_test_used_for_threshold_selection": (
            False
        ),
        "feature_version": "v2_robust_behavioural",
        "included_feature_names": training_data[
            "feature_names"
        ].tolist(),
        "excluded_feature_names": excluded_feature_names,
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_pr_auc": round(
            best_validation_pr_auc,
            6,
        ),
        "validation_best_f1": round(
            validation_best_f1,
            6,
        ),
        "selected_threshold": round(
            selected_threshold,
            6,
        ),
        "positive_class_weight": round(
            positive_weight,
            6,
        ),
        "training_seconds": round(
            total_training_seconds,
            3,
        ),
        "train_sample_count": len(
            train_dataset
        ),
        "validation_sample_count": len(
            validation_dataset
        ),
        "test_sample_count": len(
            test_dataset
        ),
        "excluded_short_training_tracks": (
            training_data[
                "excluded_count"
            ]
        ),
        "excluded_short_test_tracks": (
            test_data[
                "excluded_count"
            ]
        ),
        "configuration": {
            "epochs_requested": (
                arguments.epochs
            ),
            "epochs_completed": len(
                history_rows
            ),
            "batch_size": (
                arguments.batch_size
            ),
            "hidden_size": (
                arguments.hidden_size
            ),
            "num_layers": (
                arguments.num_layers
            ),
            "num_heads": (
                arguments.num_heads
            ),
            "feedforward_size": (
                arguments.feedforward_size
            ),
            "temporal_scales": [1, 2, 4],
            "dropout": (
                arguments.dropout
            ),
            "learning_rate": (
                arguments.learning_rate
            ),
            "weight_decay": (
                arguments.weight_decay
            ),
            "validation_size": (
                arguments.validation_size
            ),
            "patience": (
                arguments.patience
            ),
            "minimum_length": (
                arguments.minimum_length
            ),
            "seed": arguments.seed,
            "device": str(device),
            "exclude_occlusion_features": (
                arguments.exclude_occlusion_features
            ),
        },
        "validation_metrics": (
            validation_metrics
        ),
        "official_test_metrics": (
            test_metrics
        ),
        "official_test_metrics_by_length": (
            length_group_metrics
        ),
        "inference_benchmark": (
            inference_benchmark
        ),
        "environment": {
            "python": (
                platform.python_version()
            ),
            "platform": (
                platform.platform()
            ),
            "pytorch": (
                torch.__version__
            ),
            "numpy": (
                np.__version__
            ),
            "pandas": (
                pd.__version__
            ),
            "scikit_learn": (
                sklearn.__version__
            ),
        },
        "research_note": (
            "This model predicts observable "
            "potential-loitering behaviour from "
            "anonymous trajectories. It does not "
            "determine criminal intent."
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as metrics_stream:
        json.dump(
            experiment_results,
            metrics_stream,
            indent=2,
        )

    print()
    print("Multi-scale Transformer evaluation completed")
    print("----------------------------")
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Selected validation threshold: "
        f"{selected_threshold:.4f}"
    )
    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test precision: "
        f"{test_metrics['precision']:.4f}"
    )
    print(
        f"Test recall: "
        f"{test_metrics['recall']:.4f}"
    )
    print(
        f"Test F1-score: "
        f"{test_metrics['f1_score']:.4f}"
    )
    print(
        f"Test PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}"
    )
    print(
        "Confusion matrix: "
        f"TN={test_metrics['true_negative']}, "
        f"FP={test_metrics['false_positive']}, "
        f"FN={test_metrics['false_negative']}, "
        f"TP={test_metrics['true_positive']}"
    )
    print(
        "Inference latency: "
        f"{inference_benchmark['milliseconds_per_trajectory']:.4f} "
        "ms/trajectory"
    )
    print(
        f"Model: {model_path}"
    )
    print(
        f"Metrics: {metrics_path}"
    )
    print(
        f"Predictions: {predictions_path}"
    )
    print(
        f"Length metrics: {length_metrics_path}"
    )


if __name__ == "__main__":
    main()
