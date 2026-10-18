"""Explainable rule-based baseline for potential-loitering analysis.

This baseline is intended for proof-of-concept evaluation. It is not a
trained model and does not determine criminal intent.
"""

import argparse
import json
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

ALERT_THRESHOLD = 0.65
WATCH_THRESHOLD = 0.40
MINIMUM_TRACK_QUALITY = 0.30
MODEL_VERSION = "c2_rule_baseline_v1.1"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate explainable potential-loitering "
            "risk scores from trajectory features."
        )
    )

    parser.add_argument(
        "--video-id",
        default="12",
        help=(
            "Video ID used by the "
            "trajectory-summary CSV."
        ),
    )

    return parser.parse_args()


def clip_score(value):
    """Limit a score to the range 0–1."""

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def value_is_true(value):
    """Safely convert CSV Boolean values."""

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def calculate_risk(row):
    """Calculate risk, uncertainty and explanations for one track."""

    dwell_time = float(
        row["dwell_time"]
    )

    pause_ratio = float(
        row["pause_ratio"]
    )

    maximum_displacement = float(
        row["maximum_displacement"]
    )

    path_efficiency = float(
        row["path_efficiency"]
    )

    direction_changes = int(
        row["direction_changes"]
    )

    track_quality = float(
        row["track_quality"]
    )

    eligible = value_is_true(
        row["eligible_for_analysis"]
    )

    duration_for_rate = max(
        dwell_time,
        1.0,
    )

    direction_change_rate = (
        direction_changes
        / duration_for_rate
    )

    dwell_score = clip_score(
        (dwell_time - 10.0)
        / 20.0
    )

    pause_score = clip_score(
        (pause_ratio - 0.15)
        / 0.60
    )

    confinement_score = clip_score(
        1.0
        - (
            maximum_displacement
            / 0.40
        )
    )

    inefficient_path_score = clip_score(
        1.0
        - path_efficiency
    )

    # The feature script now calculates direction changes
    # from smoothed and sampled trajectory coordinates.
    direction_score = clip_score(
        direction_change_rate
        / 3.0
    )

    raw_risk = (
        0.35 * dwell_score
        + 0.20 * pause_score
        + 0.20 * confinement_score
        + 0.15 * inefficient_path_score
        + 0.10 * direction_score
    )

    risk_score = clip_score(
        raw_risk
        * track_quality
    )

    uncertainty = clip_score(
        1.0
        - track_quality
    )

    explanation_signals = []

    if dwell_score >= 0.50:
        explanation_signals.append(
            "prolonged_presence"
        )

    if pause_score >= 0.40:
        explanation_signals.append(
            "frequent_or_long_pauses"
        )

    if confinement_score >= 0.60:
        explanation_signals.append(
            "movement_confined_to_small_area"
        )

    if inefficient_path_score >= 0.60:
        explanation_signals.append(
            "indirect_or_repeated_movement"
        )

    if direction_score >= 0.60:
        explanation_signals.append(
            "frequent_direction_changes"
        )

    if not explanation_signals:
        explanation_signals.append(
            "insufficient_loitering_signals"
        )

    alert_generated = (
        eligible
        and track_quality
        >= MINIMUM_TRACK_QUALITY
        and risk_score
        >= ALERT_THRESHOLD
    )

    if not eligible:
        behaviour_state = (
            "insufficient_track_quality"
        )
    elif alert_generated:
        behaviour_state = (
            "potential_loitering"
        )
    elif risk_score >= WATCH_THRESHOLD:
        behaviour_state = "monitor"
    else:
        behaviour_state = (
            "normal_or_low_risk"
        )

    return {
        "risk_score": round(
            risk_score,
            4,
        ),
        "uncertainty": round(
            uncertainty,
            4,
        ),
        "behaviour_state": (
            behaviour_state
        ),
        "alert_generated": (
            alert_generated
        ),
        "explanation_signals": (
            explanation_signals
        ),
        "dwell_score": round(
            dwell_score,
            4,
        ),
        "pause_score": round(
            pause_score,
            4,
        ),
        "confinement_score": round(
            confinement_score,
            4,
        ),
        "path_score": round(
            inefficient_path_score,
            4,
        ),
        "direction_score": round(
            direction_score,
            4,
        ),
    }


def create_assessment(row, analysis):
    """Create a privacy-preserving track-assessment record."""

    return {
        "record_type": (
            "track_assessment"
        ),
        "camera_id": (
            f"atma_video_{row['video_id']}"
        ),
        "person_id": (
            f"P-{int(row['track_id']):04d}"
        ),
        "source_video_id": str(
            row["video_id"]
        ),
        "first_frame": int(
            row["first_frame"]
        ),
        "last_frame": int(
            row["last_frame"]
        ),
        "dwell_time_seconds": round(
            float(row["dwell_time"]),
            4,
        ),
        "behaviour_state": (
            analysis[
                "behaviour_state"
            ]
        ),
        "risk_score": (
            analysis[
                "risk_score"
            ]
        ),
        "uncertainty": (
            analysis[
                "uncertainty"
            ]
        ),
        "track_quality": round(
            float(
                row["track_quality"]
            ),
            4,
        ),
        "alert_generated": (
            analysis[
                "alert_generated"
            ]
        ),
        "explanation_signals": (
            analysis[
                "explanation_signals"
            ]
        ),
        "model_version": (
            MODEL_VERSION
        ),
        "requires_operator_review": (
            analysis[
                "alert_generated"
            ]
        ),
        "privacy_note": (
            "Anonymous tracking ID; "
            "no facial identification."
        ),
    }


def main():
    arguments = parse_arguments()

    input_csv = (
        WORK_ROOT
        / "data"
        / "processed"
        / (
            f"{arguments.video_id}"
            "_trajectory_summary.csv"
        )
    )

    output_directory = (
        WORK_ROOT
        / "outputs"
        / "loitering_baseline"
    )

    output_csv = (
        output_directory
        / (
            f"{arguments.video_id}"
            "_loitering_risk.csv"
        )
    )

    assessment_json = (
        output_directory
        / (
            f"{arguments.video_id}"
            "_track_assessments.json"
        )
    )

    event_json = (
        output_directory
        / (
            f"{arguments.video_id}"
            "_loitering_events.json"
        )
    )

    if not input_csv.exists():
        raise FileNotFoundError(
            "Trajectory summary was not found:\n"
            f"{input_csv}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.read_csv(
        input_csv
    )

    required_columns = {
        "video_id",
        "track_id",
        "first_frame",
        "last_frame",
        "dwell_time",
        "pause_ratio",
        "maximum_displacement",
        "path_efficiency",
        "direction_changes",
        "track_quality",
        "eligible_for_analysis",
    }

    missing_columns = (
        required_columns
        - set(summary.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if summary.empty:
        raise ValueError(
            "The trajectory-summary CSV is empty."
        )

    results = []
    json_assessments = []
    json_events = []

    for row in summary.to_dict(
        orient="records"
    ):
        analysis = calculate_risk(
            row
        )

        result = {
            **row,
            **analysis,
            "explanation_signals": "|".join(
                analysis[
                    "explanation_signals"
                ]
            ),
            "model_version": (
                MODEL_VERSION
            ),
        }

        results.append(
            result
        )

        assessment = create_assessment(
            row,
            analysis,
        )

        json_assessments.append(
            assessment
        )

        # Only confirmed alert records are written to the
        # event file. Monitor and normal assessments remain
        # in the separate assessment file.
        if analysis["alert_generated"]:
            event = {
                **assessment,
                "record_type": (
                    "alert_event"
                ),
                "event_type": (
                    "potential_loitering"
                ),
                "requires_operator_review": (
                    True
                ),
            }

            json_events.append(
                event
            )

    results_dataframe = pd.DataFrame(
        results
    )

    results_dataframe = (
        results_dataframe.sort_values(
            "risk_score",
            ascending=False,
        )
    )

    results_dataframe.to_csv(
        output_csv,
        index=False,
    )

    with assessment_json.open(
        "w",
        encoding="utf-8",
    ) as json_stream:
        json.dump(
            json_assessments,
            json_stream,
            indent=2,
        )

    with event_json.open(
        "w",
        encoding="utf-8",
    ) as json_stream:
        json.dump(
            json_events,
            json_stream,
            indent=2,
        )

    alert_count = int(
        results_dataframe[
            "alert_generated"
        ].sum()
    )

    watch_count = int(
        (
            results_dataframe[
                "behaviour_state"
            ]
            == "monitor"
        ).sum()
    )

    print()
    print(
        "Loitering baseline completed"
    )
    print(
        "----------------------------"
    )
    print(
        f"Tracks analysed: "
        f"{len(results_dataframe)}"
    )
    print(
        f"Potential alerts: "
        f"{alert_count}"
    )
    print(
        f"Monitor states: "
        f"{watch_count}"
    )
    print(
        f"Risk CSV: "
        f"{output_csv}"
    )
    print(
        f"Assessment JSON: "
        f"{assessment_json}"
    )
    print(
        f"Alert-event JSON: "
        f"{event_json}"
    )


if __name__ == "__main__":
    main()