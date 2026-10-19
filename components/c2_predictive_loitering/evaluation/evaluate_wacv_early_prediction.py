"""Evaluate trained WACV models on incomplete trajectory prefixes.

The evaluator preserves the official WACV test split and uses each model's
validation-selected decision threshold. It never fits or tunes on test data.
Predictions are measured at 25%, 50%, 75% and 100% observation ratios.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from components.c2_predictive_loitering.models.train_wacv_gru_v2 import (  # noqa: E402
    GRULoiteringClassifier,
)
from components.c2_predictive_loitering.models.train_wacv_lstm_v2 import (  # noqa: E402
    LSTMLoiteringClassifier,
)
from components.c2_predictive_loitering.models.train_wacv_multiscale_transformer import (  # noqa: E402
    MultiScaleTransformerClassifier,
)


WORK_ROOT = Path(os.environ.get("C2_WORK_ROOT", "."))
OBSERVATION_RATIOS = (0.25, 0.50, 0.75, 1.00)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate WACV models at incomplete observation ratios."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=(
            WORK_ROOT
            / "data"
            / "processed"
            / "wacv_loitering_v2"
            / "wacv_test_sequences_v2.npz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORK_ROOT / "outputs" / "wacv_early_prediction",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class PrefixDataset(Dataset):
    def __init__(self, sequences, labels, lengths, trajectory_ids):
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        self.lengths = torch.from_numpy(lengths).long()
        self.trajectory_ids = torch.from_numpy(trajectory_ids).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.labels[index],
            self.lengths[index],
            self.trajectory_ids[index],
        )


def load_test_data(path, minimum_length):
    if not path.exists():
        raise FileNotFoundError(f"Prepared test data not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "sequences",
            "labels",
            "lengths",
            "trajectory_ids",
            "feature_names",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Prepared test data is missing: {sorted(missing)}")

        result = {name: np.asarray(data[name]) for name in required}

    selected = result["lengths"] >= minimum_length
    for name in ("sequences", "labels", "lengths", "trajectory_ids"):
        result[name] = result[name][selected]

    result["sequences"] = result["sequences"].astype(np.float32)
    result["labels"] = result["labels"].astype(np.int64)
    result["lengths"] = result["lengths"].astype(np.int64)
    result["trajectory_ids"] = result["trajectory_ids"].astype(np.int64)
    return result


def checkpoint_paths():
    return {
        "lstm_v2": WORK_ROOT / "outputs" / "wacv_lstm_v2" / "wacv_lstm_v2_best.pt",
        "gru_v2": WORK_ROOT / "outputs" / "wacv_gru_v2" / "wacv_gru_v2_best.pt",
        "multiscale_transformer": (
            WORK_ROOT
            / "outputs"
            / "wacv_multiscale_transformer"
            / "wacv_multiscale_transformer_best.pt"
        ),
    }


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def construct_model(model_name, checkpoint):
    common = {
        "input_size": int(checkpoint["input_size"]),
        "hidden_size": int(checkpoint["hidden_size"]),
        "num_layers": int(checkpoint["num_layers"]),
        "dropout": float(checkpoint["dropout"]),
    }
    if model_name == "lstm_v2":
        return LSTMLoiteringClassifier(**common)
    if model_name == "gru_v2":
        return GRULoiteringClassifier(**common)
    if model_name == "multiscale_transformer":
        return MultiScaleTransformerClassifier(
            **common,
            num_heads=int(checkpoint["num_heads"]),
            feedforward_size=int(checkpoint["feedforward_size"]),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def align_features(test_data, checkpoint_feature_names):
    available = test_data["feature_names"].astype(str).tolist()
    required = [str(name) for name in checkpoint_feature_names]
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"Checkpoint features unavailable in test data: {missing}")
    indices = [available.index(name) for name in required]
    return test_data["sequences"][:, :, indices]


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def collect_predictions(model, loader, device):
    model.eval()
    probability_parts = []
    label_parts = []
    length_parts = []
    id_parts = []
    start = time.perf_counter()

    with torch.inference_mode():
        for sequences, labels, lengths, trajectory_ids in loader:
            logits = model(sequences.to(device), lengths.to(device))
            probability_parts.append(torch.sigmoid(logits).cpu().numpy())
            label_parts.append(labels.numpy())
            length_parts.append(lengths.numpy())
            id_parts.append(trajectory_ids.numpy())

    elapsed = time.perf_counter() - start
    probabilities = np.concatenate(probability_parts)
    return {
        "probabilities": probabilities,
        "labels": np.concatenate(label_parts),
        "lengths": np.concatenate(length_parts),
        "trajectory_ids": np.concatenate(id_parts),
        "milliseconds_per_trajectory": elapsed * 1000.0 / len(probabilities),
    }


def main():
    arguments = parse_arguments()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(arguments.device)
    arguments.output_root.mkdir(parents=True, exist_ok=True)

    paths = checkpoint_paths()
    checkpoints = {
        name: load_checkpoint(path, device) for name, path in paths.items()
    }
    minimum_lengths = {
        int(checkpoint.get("minimum_length", 2))
        for checkpoint in checkpoints.values()
    }
    if len(minimum_lengths) != 1:
        raise ValueError("Models use different minimum trajectory lengths.")
    minimum_length = minimum_lengths.pop()
    test_data = load_test_data(arguments.data_path, minimum_length)

    summary_rows = []
    prediction_rows = []
    print("WACV early-prediction evaluation")
    print("--------------------------------")
    print(f"Device: {device}")
    print(f"Official test samples: {len(test_data['labels']):,}")
    print("Observation ratios: 25%, 50%, 75%, 100%")
    print("Threshold source: validation-selected model checkpoints")

    for model_name, checkpoint in checkpoints.items():
        model = construct_model(model_name, checkpoint).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = float(checkpoint["decision_threshold"])
        sequences = align_features(test_data, checkpoint["feature_names"])

        print()
        print(f"Model: {model_name}")
        for ratio in OBSERVATION_RATIOS:
            prefix_lengths = np.ceil(test_data["lengths"] * ratio).astype(np.int64)
            prefix_lengths = np.clip(
                prefix_lengths,
                minimum_length,
                test_data["lengths"],
            )
            dataset = PrefixDataset(
                sequences,
                test_data["labels"],
                prefix_lengths,
                test_data["trajectory_ids"],
            )
            loader = DataLoader(
                dataset,
                batch_size=arguments.batch_size,
                shuffle=False,
                num_workers=arguments.num_workers,
            )
            output = collect_predictions(model, loader, device)
            metrics = calculate_metrics(
                output["labels"], output["probabilities"], threshold
            )
            ratio_percent = int(round(ratio * 100))
            summary_rows.append(
                {
                    "model": model_name,
                    "observation_ratio": ratio,
                    "observation_percent": ratio_percent,
                    "sample_count": len(output["labels"]),
                    "mean_prefix_length": float(np.mean(output["lengths"])),
                    "decision_threshold": threshold,
                    "milliseconds_per_trajectory": output[
                        "milliseconds_per_trajectory"
                    ],
                    **metrics,
                }
            )
            predictions = (
                output["probabilities"] >= threshold
            ).astype(np.int64)
            for index in range(len(predictions)):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "observation_percent": ratio_percent,
                        "trajectory_id": int(output["trajectory_ids"][index]),
                        "prefix_length": int(output["lengths"][index]),
                        "full_length": int(test_data["lengths"][index]),
                        "actual_label": int(output["labels"][index]),
                        "predicted_probability": float(
                            output["probabilities"][index]
                        ),
                        "predicted_label": int(predictions[index]),
                    }
                )
            print(
                f"  {ratio_percent:>3}% | "
                f"F1={metrics['f1_score']:.4f} | "
                f"PR-AUC={metrics['pr_auc']:.4f} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f}"
            )

    summary_path = arguments.output_root / "wacv_early_prediction_metrics.csv"
    predictions_path = (
        arguments.output_root / "wacv_early_prediction_predictions.csv"
    )
    metadata_path = arguments.output_root / "wacv_early_prediction_metadata.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    metadata = {
        "dataset": "RS-WACV24_Loitering official test split",
        "test_data_path": str(arguments.data_path),
        "test_sample_count": int(len(test_data["labels"])),
        "minimum_length": int(minimum_length),
        "observation_ratios": list(OBSERVATION_RATIOS),
        "threshold_policy": (
            "Reuse each checkpoint's validation-selected threshold; "
            "no test-set threshold tuning."
        ),
        "checkpoint_paths": {name: str(path) for name, path in paths.items()},
        "device": str(device),
        "batch_size": int(arguments.batch_size),
        "research_note": (
            "Predictions describe observable potential-loitering behaviour "
            "from anonymous trajectories and do not determine intent."
        ),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print()
    print("Early-prediction evaluation completed")
    print(f"Metrics: {summary_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
