"""Calibrate WACV loitering models and quantify predictive uncertainty.

Platt calibration is fitted only on the validation subset recreated from the
official training split. The official test split is used only for evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
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
EPSILON = 1e-7


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calibrate WACV probabilities using validation data only."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=WORK_ROOT / "data" / "processed" / "wacv_loitering_v2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORK_ROOT / "outputs" / "wacv_calibration_uncertainty",
    )
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--maximum-alert-uncertainty",
        type=float,
        default=0.65,
        help="Suppress positive alerts above this normalized entropy.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=10,
    )
    return parser.parse_args()


class TrajectoryDataset(Dataset):
    def __init__(self, sequences, labels, lengths, trajectory_ids, indices=None):
        if indices is None:
            indices = np.arange(len(labels))
        indices = np.asarray(indices, dtype=np.int64)
        self.sequences = torch.from_numpy(sequences[indices]).float()
        self.labels = torch.from_numpy(labels[indices]).long()
        self.lengths = torch.from_numpy(lengths[indices]).long()
        self.trajectory_ids = torch.from_numpy(trajectory_ids[indices]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.labels[index],
            self.lengths[index],
            self.trajectory_ids[index],
        )


def load_split(path, minimum_length):
    if not path.exists():
        raise FileNotFoundError(f"Prepared data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        names = (
            "sequences",
            "labels",
            "lengths",
            "trajectory_ids",
            "feature_names",
        )
        missing = set(names) - set(data.files)
        if missing:
            raise ValueError(f"Prepared data is missing: {sorted(missing)}")
        result = {name: np.asarray(data[name]) for name in names}

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
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def construct_model(name, checkpoint):
    common = {
        "input_size": int(checkpoint["input_size"]),
        "hidden_size": int(checkpoint["hidden_size"]),
        "num_layers": int(checkpoint["num_layers"]),
        "dropout": float(checkpoint["dropout"]),
    }
    if name == "lstm_v2":
        return LSTMLoiteringClassifier(**common)
    if name == "gru_v2":
        return GRULoiteringClassifier(**common)
    if name == "multiscale_transformer":
        return MultiScaleTransformerClassifier(
            **common,
            num_heads=int(checkpoint["num_heads"]),
            feedforward_size=int(checkpoint["feedforward_size"]),
        )
    raise ValueError(f"Unsupported model: {name}")


def align_features(split, checkpoint_names):
    available = split["feature_names"].astype(str).tolist()
    required = [str(name) for name in checkpoint_names]
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"Features missing from prepared data: {missing}")
    indices = [available.index(name) for name in required]
    return split["sequences"][:, :, indices]


def collect_logits(model, loader, device):
    model.eval()
    logits = []
    labels = []
    lengths = []
    trajectory_ids = []
    with torch.inference_mode():
        for sequences, batch_labels, batch_lengths, batch_ids in loader:
            output = model(sequences.to(device), batch_lengths.to(device))
            logits.append(output.cpu().numpy())
            labels.append(batch_labels.numpy())
            lengths.append(batch_lengths.numpy())
            trajectory_ids.append(batch_ids.numpy())
    return {
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
        "lengths": np.concatenate(lengths),
        "trajectory_ids": np.concatenate(trajectory_ids),
    }


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def normalized_entropy(probabilities):
    probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    entropy = -(
        probabilities * np.log(probabilities)
        + (1.0 - probabilities) * np.log(1.0 - probabilities)
    )
    return entropy / np.log(2.0)


def expected_calibration_error(labels, probabilities, number_of_bins):
    edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    total = len(labels)
    error = 0.0
    rows = []
    for index in range(number_of_bins):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (
            (probabilities >= lower)
            & (probabilities < upper if index < number_of_bins - 1 else probabilities <= upper)
        )
        count = int(selected.sum())
        if count == 0:
            accuracy = 0.0
            confidence = 0.0
        else:
            accuracy = float(labels[selected].mean())
            confidence = float(probabilities[selected].mean())
            error += count / total * abs(accuracy - confidence)
        rows.append(
            {
                "bin": index + 1,
                "lower_probability": lower,
                "upper_probability": upper,
                "sample_count": count,
                "observed_positive_rate": accuracy,
                "mean_probability": confidence,
                "absolute_gap": abs(accuracy - confidence),
            }
        )
    return float(error), rows


def select_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], EPSILON
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def classification_metrics(labels, probabilities, threshold):
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


def probability_metrics(labels, probabilities, bins):
    ece, reliability = expected_calibration_error(labels, probabilities, bins)
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return {
        "brier_score": float(brier_score_loss(labels, clipped)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "expected_calibration_error": ece,
        "reliability_rows": reliability,
    }


def main():
    arguments = parse_arguments()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if not 0.0 <= arguments.maximum_alert_uncertainty <= 1.0:
        raise ValueError("maximum-alert-uncertainty must be between 0 and 1.")
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
    seeds = {int(checkpoint.get("seed", 42)) for checkpoint in checkpoints.values()}
    if len(minimum_lengths) != 1 or len(seeds) != 1:
        raise ValueError("Checkpoints use inconsistent split settings.")
    minimum_length = minimum_lengths.pop()
    seed = seeds.pop()

    training_data = load_split(
        arguments.data_root / "wacv_train_sequences_v2.npz", minimum_length
    )
    test_data = load_split(
        arguments.data_root / "wacv_test_sequences_v2.npz", minimum_length
    )
    all_indices = np.arange(len(training_data["labels"]))
    _, validation_indices = train_test_split(
        all_indices,
        test_size=arguments.validation_size,
        random_state=seed,
        stratify=training_data["labels"],
    )

    metric_rows = []
    reliability_rows = []
    prediction_rows = []
    calibrator_records = {}

    print("WACV calibration and uncertainty evaluation")
    print("-------------------------------------------")
    print(f"Device: {device}")
    print(f"Validation samples: {len(validation_indices):,}")
    print(f"Official test samples: {len(test_data['labels']):,}")
    print("Calibration method: Platt logistic calibration")
    print("Calibration fitting source: validation subset only")

    for model_name, checkpoint in checkpoints.items():
        model = construct_model(model_name, checkpoint).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation_sequences = align_features(
            training_data, checkpoint["feature_names"]
        )
        test_sequences = align_features(test_data, checkpoint["feature_names"])
        validation_loader = DataLoader(
            TrajectoryDataset(
                validation_sequences,
                training_data["labels"],
                training_data["lengths"],
                training_data["trajectory_ids"],
                validation_indices,
            ),
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=arguments.num_workers,
        )
        test_loader = DataLoader(
            TrajectoryDataset(
                test_sequences,
                test_data["labels"],
                test_data["lengths"],
                test_data["trajectory_ids"],
            ),
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=arguments.num_workers,
        )
        validation_output = collect_logits(model, validation_loader, device)
        test_output = collect_logits(model, test_loader, device)

        calibrator = LogisticRegression(C=1_000_000.0, solver="lbfgs")
        calibrator.fit(
            validation_output["logits"].reshape(-1, 1),
            validation_output["labels"],
        )
        validation_calibrated = calibrator.predict_proba(
            validation_output["logits"].reshape(-1, 1)
        )[:, 1]
        raw_test = sigmoid(test_output["logits"])
        calibrated_test = calibrator.predict_proba(
            test_output["logits"].reshape(-1, 1)
        )[:, 1]
        calibrated_threshold = select_f1_threshold(
            validation_output["labels"], validation_calibrated
        )
        raw_threshold = float(checkpoint["decision_threshold"])
        uncertainty = normalized_entropy(calibrated_test)

        raw_probability_metrics = probability_metrics(
            test_output["labels"], raw_test, arguments.ece_bins
        )
        calibrated_probability_metrics = probability_metrics(
            test_output["labels"], calibrated_test, arguments.ece_bins
        )
        raw_classification = classification_metrics(
            test_output["labels"], raw_test, raw_threshold
        )
        calibrated_classification = classification_metrics(
            test_output["labels"], calibrated_test, calibrated_threshold
        )

        for probability_type, probability_values, probability_result, class_result, threshold in (
            (
                "raw",
                raw_test,
                raw_probability_metrics,
                raw_classification,
                raw_threshold,
            ),
            (
                "calibrated",
                calibrated_test,
                calibrated_probability_metrics,
                calibrated_classification,
                calibrated_threshold,
            ),
        ):
            metric_rows.append(
                {
                    "model": model_name,
                    "probability_type": probability_type,
                    "decision_threshold": threshold,
                    "brier_score": probability_result["brier_score"],
                    "log_loss": probability_result["log_loss"],
                    "expected_calibration_error": probability_result[
                        "expected_calibration_error"
                    ],
                    **class_result,
                }
            )
            for row in probability_result["reliability_rows"]:
                reliability_rows.append(
                    {
                        "model": model_name,
                        "probability_type": probability_type,
                        **row,
                    }
                )

        alert_candidate = calibrated_test >= calibrated_threshold
        alert_generated = alert_candidate & (
            uncertainty <= arguments.maximum_alert_uncertainty
        )
        for index in range(len(calibrated_test)):
            if alert_generated[index]:
                state = "potential_loitering"
            elif alert_candidate[index]:
                state = "monitor_high_uncertainty"
            elif uncertainty[index] > arguments.maximum_alert_uncertainty:
                state = "uncertain"
            else:
                state = "normal_or_low_risk"
            prediction_rows.append(
                {
                    "model": model_name,
                    "trajectory_id": int(test_output["trajectory_ids"][index]),
                    "sequence_length": int(test_output["lengths"][index]),
                    "actual_label": int(test_output["labels"][index]),
                    "raw_probability": float(raw_test[index]),
                    "calibrated_probability": float(calibrated_test[index]),
                    "uncertainty": float(uncertainty[index]),
                    "calibrated_threshold": calibrated_threshold,
                    "alert_candidate": bool(alert_candidate[index]),
                    "alert_generated": bool(alert_generated[index]),
                    "behaviour_state": state,
                }
            )

        calibrator_records[model_name] = {
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "raw_threshold": raw_threshold,
            "calibrated_threshold": calibrated_threshold,
            "maximum_alert_uncertainty": arguments.maximum_alert_uncertainty,
        }
        print()
        print(f"Model: {model_name}")
        print(
            "  Brier: "
            f"{raw_probability_metrics['brier_score']:.4f} -> "
            f"{calibrated_probability_metrics['brier_score']:.4f}"
        )
        print(
            "  ECE: "
            f"{raw_probability_metrics['expected_calibration_error']:.4f} -> "
            f"{calibrated_probability_metrics['expected_calibration_error']:.4f}"
        )
        print(
            "  Calibrated threshold: "
            f"{calibrated_threshold:.4f} | "
            f"F1={calibrated_classification['f1_score']:.4f}"
        )

    metrics_path = arguments.output_root / "wacv_calibration_metrics.csv"
    reliability_path = arguments.output_root / "wacv_reliability_bins.csv"
    predictions_path = arguments.output_root / "wacv_uncertainty_predictions.csv"
    calibrators_path = arguments.output_root / "wacv_platt_calibrators.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(reliability_rows).to_csv(reliability_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    with calibrators_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "method": "Platt logistic calibration",
                "fit_source": "validation subset recreated from official training split",
                "official_test_used_for_fitting": False,
                "validation_size": arguments.validation_size,
                "seed": seed,
                "ece_bins": arguments.ece_bins,
                "models": calibrator_records,
            },
            stream,
            indent=2,
        )

    print()
    print("Calibration and uncertainty evaluation completed")
    print(f"Metrics: {metrics_path}")
    print(f"Reliability bins: {reliability_path}")
    print(f"Uncertainty predictions: {predictions_path}")
    print(f"Calibrators: {calibrators_path}")


if __name__ == "__main__":
    main()
