"""Real-time explainable predictive-loitering prototype.

Uses YOLO11 person detection and ByteTrack anonymous tracking. For every
trajectory, the prototype calculates dwell time, movement speed, pause ratio,
path efficiency, direction changes, track quality, uncertainty and an
explainable rule-based loitering-risk score.

This is a proof-of-concept baseline, not a trained loitering classifier.
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


WORK_ROOT = Path(os.environ.get("C2_WORK_ROOT", "."))
OUTPUT_ROOT = WORK_ROOT / "outputs" / "realtime_loitering"

TRAIL_LENGTH = 100
MAX_HISTORY_LENGTH = 3000

ALERT_THRESHOLD = 0.65
WATCH_THRESHOLD = 0.40
MINIMUM_TRACK_QUALITY = 0.30
MINIMUM_ANALYSIS_SECONDS = 3.0

PAUSE_SPEED_THRESHOLD = 0.015
DIRECTION_CHANGE_DEGREES = 45.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time explainable predictive-loitering prototype."
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Camera number such as 0, or a complete video path. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--camera-id",
        default="webcam_01",
        help="Anonymous camera identifier used in generated events.",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or local model path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum person-detection confidence.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="YOLO input image size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device: cpu, 0, 1, etc.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=ALERT_THRESHOLD,
        help="Risk threshold required for an alert.",
    )
    parser.add_argument(
        "--watch-threshold",
        type=float,
        default=WATCH_THRESHOLD,
        help="Risk threshold required for a monitor state.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=MINIMUM_TRACK_QUALITY,
        help="Minimum track quality required for an alert.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the annotated webcam/video output.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening an OpenCV display window.",
    )

    return parser.parse_args()


def resolve_source(source_argument):
    try:
        return int(source_argument)
    except ValueError:
        source_path = Path(source_argument)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Video source was not found: {source_path}"
            )

        return str(source_path.resolve())


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def vector_angle_degrees(first_vector, second_vector):
    first_length = math.hypot(first_vector[0], first_vector[1])
    second_length = math.hypot(second_vector[0], second_vector[1])

    if first_length <= 1e-9 or second_length <= 1e-9:
        return 0.0

    dot_product = (
        first_vector[0] * second_vector[0]
        + first_vector[1] * second_vector[1]
    )

    cosine_value = dot_product / (first_length * second_length)
    cosine_value = max(-1.0, min(1.0, cosine_value))

    return math.degrees(math.acos(cosine_value))


def calculate_features(history):
    observations = list(history)

    if len(observations) < 2:
        return {
            "observation_count": len(observations),
            "dwell_time": 0.0,
            "mean_speed": 0.0,
            "pause_ratio": 1.0,
            "maximum_displacement": 0.0,
            "path_length": 0.0,
            "path_efficiency": 1.0,
            "direction_changes": 0,
            "mean_confidence": (
                observations[0]["confidence"] if observations else 0.0
            ),
            "track_quality": 0.0,
        }

    first_observation = observations[0]
    last_observation = observations[-1]

    dwell_time = max(
        0.0,
        last_observation["timestamp"] - first_observation["timestamp"],
    )

    speeds = []
    movement_vectors = []
    path_length = 0.0

    first_x = first_observation["x"]
    first_y = first_observation["y"]
    maximum_displacement = 0.0

    for previous, current in zip(observations[:-1], observations[1:]):
        delta_x = current["x"] - previous["x"]
        delta_y = current["y"] - previous["y"]
        delta_time = current["timestamp"] - previous["timestamp"]

        distance = math.hypot(delta_x, delta_y)
        path_length += distance

        if delta_time > 0:
            speeds.append(distance / delta_time)

        if distance > 0.001:
            movement_vectors.append((delta_x, delta_y))

        displacement_from_start = math.hypot(
            current["x"] - first_x,
            current["y"] - first_y,
        )

        maximum_displacement = max(
            maximum_displacement,
            displacement_from_start,
        )

    mean_speed = float(np.mean(speeds)) if speeds else 0.0

    pause_count = sum(
        speed <= PAUSE_SPEED_THRESHOLD for speed in speeds
    )
    pause_ratio = pause_count / len(speeds) if speeds else 1.0

    direct_distance = math.hypot(
        last_observation["x"] - first_x,
        last_observation["y"] - first_y,
    )

    path_efficiency = (
        direct_distance / path_length if path_length > 1e-9 else 1.0
    )

    direction_changes = 0

    for first_vector, second_vector in zip(
        movement_vectors[:-1],
        movement_vectors[1:],
    ):
        angle = vector_angle_degrees(first_vector, second_vector)

        if angle >= DIRECTION_CHANGE_DEGREES:
            direction_changes += 1

    mean_confidence = float(
        np.mean(
            [
                observation["confidence"]
                for observation in observations
            ]
        )
    )

    frame_span = max(
        1,
        last_observation["frame_number"]
        - first_observation["frame_number"]
        + 1,
    )
    coverage_ratio = len(observations) / frame_span

    duration_quality = clamp(dwell_time / 5.0)
    observation_quality = clamp(len(observations) / 30.0)

    track_quality = clamp(
        0.35 * coverage_ratio
        + 0.30 * mean_confidence
        + 0.20 * observation_quality
        + 0.15 * duration_quality
    )

    return {
        "observation_count": len(observations),
        "dwell_time": dwell_time,
        "mean_speed": mean_speed,
        "pause_ratio": pause_ratio,
        "maximum_displacement": maximum_displacement,
        "path_length": path_length,
        "path_efficiency": clamp(path_efficiency),
        "direction_changes": direction_changes,
        "mean_confidence": mean_confidence,
        "track_quality": track_quality,
    }


def calculate_risk(features, alert_threshold, watch_threshold, min_quality):
    dwell_score = clamp(features["dwell_time"] / 30.0)
    pause_score = clamp(features["pause_ratio"] / 0.70)

    confinement_score = 1.0 - clamp(
        features["maximum_displacement"] / 0.35
    )

    inefficient_path_score = 1.0 - clamp(
        features["path_efficiency"]
    )

    direction_score = clamp(
        features["direction_changes"] / 20.0
    )

    raw_risk = (
        0.35 * dwell_score
        + 0.20 * pause_score
        + 0.20 * confinement_score
        + 0.15 * inefficient_path_score
        + 0.10 * direction_score
    )

    track_quality = features["track_quality"]
    risk_score = clamp(raw_risk * track_quality)
    uncertainty = clamp(1.0 - track_quality)

    explanation_signals = []

    if features["dwell_time"] >= 15.0:
        explanation_signals.append("prolonged presence")

    if features["pause_ratio"] >= 0.50:
        explanation_signals.append("frequent or long pauses")

    if features["maximum_displacement"] <= 0.20:
        explanation_signals.append("movement confined to a small area")

    if features["path_efficiency"] <= 0.45:
        explanation_signals.append("indirect or repeated movement")

    if features["direction_changes"] >= 10:
        explanation_signals.append("frequent direction changes")

    if not explanation_signals:
        explanation_signals.append("insufficient suspicious movement evidence")

    sufficient_duration = (
        features["dwell_time"] >= MINIMUM_ANALYSIS_SECONDS
    )
    sufficient_quality = track_quality >= min_quality

    alert_generated = (
        sufficient_duration
        and sufficient_quality
        and risk_score >= alert_threshold
    )

    if not sufficient_duration or not sufficient_quality:
        behaviour_state = "INSUFFICIENT EVIDENCE"
    elif alert_generated:
        behaviour_state = "POTENTIAL LOITERING"
    elif risk_score >= watch_threshold:
        behaviour_state = "MONITOR"
    else:
        behaviour_state = "NORMAL / LOW RISK"

    return {
        "risk_score": risk_score,
        "uncertainty": uncertainty,
        "behaviour_state": behaviour_state,
        "alert_generated": alert_generated,
        "explanation_signals": explanation_signals,
    }


def state_colour(behaviour_state):
    if behaviour_state == "POTENTIAL LOITERING":
        return (0, 0, 255)

    if behaviour_state == "MONITOR":
        return (0, 165, 255)

    if behaviour_state == "INSUFFICIENT EVIDENCE":
        return (180, 180, 180)

    return (0, 200, 0)


def draw_text(frame, text, position, colour, scale=0.48):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def create_event(
    camera_id,
    track_id,
    features,
    risk_result,
):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "event_type": "potential_loitering",
        "person_id": f"anonymous_track_{track_id}",
        "threat_score": round(risk_result["risk_score"], 4),
        "uncertainty": round(risk_result["uncertainty"], 4),
        "track_quality": round(features["track_quality"], 4),
        "dwell_time_seconds": round(features["dwell_time"], 4),
        "behaviour_state": risk_result["behaviour_state"],
        "explanation_signals": risk_result["explanation_signals"],
        "model_version": "c2_rule_baseline_realtime_v1.0",
        "requires_operator_review": True,
    }


def run_realtime(arguments):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    source = resolve_source(arguments.source)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {arguments.source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))

    if source_fps <= 0:
        source_fps = 20.0

    output_video_path = (
        OUTPUT_ROOT / f"{arguments.camera_id}_realtime_demo.mp4"
    )
    event_path = (
        OUTPUT_ROOT / f"{arguments.camera_id}_events.jsonl"
    )

    writer = None

    if arguments.save_video:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                f"Could not create output video: {output_video_path}"
            )

    model = YOLO(arguments.model)

    histories = defaultdict(
        lambda: deque(maxlen=MAX_HISTORY_LENGTH)
    )
    alert_recorded = set()
    frame_number = 0
    start_time = time.perf_counter()

    print("Real-time predictive-loitering prototype")
    print("----------------------------------------")
    print(f"Source: {arguments.source}")
    print(f"Camera ID: {arguments.camera_id}")
    print(f"Device: {arguments.device}")
    print(f"Alert threshold: {arguments.alert_threshold}")
    print("Press Q in the video window to stop.")

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1
            current_timestamp = time.perf_counter() - start_time

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=arguments.confidence,
                imgsz=arguments.image_size,
                device=arguments.device,
                verbose=False,
            )[0]

            annotated_frame = frame.copy()
            active_track_count = 0
            alert_count = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().numpy()

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    active_track_count += 1

                    x1, y1, x2, y2 = [
                        float(value) for value in box
                    ]

                    centre_x = ((x1 + x2) / 2.0) / width
                    centre_y = ((y1 + y2) / 2.0) / height

                    histories[track_id].append(
                        {
                            "frame_number": frame_number,
                            "timestamp": current_timestamp,
                            "x": centre_x,
                            "y": centre_y,
                            "confidence": float(confidence),
                        }
                    )

                    features = calculate_features(
                        histories[track_id]
                    )
                    risk_result = calculate_risk(
                        features,
                        arguments.alert_threshold,
                        arguments.watch_threshold,
                        arguments.min_quality,
                    )

                    if risk_result["alert_generated"]:
                        alert_count += 1

                        if track_id not in alert_recorded:
                            event = create_event(
                                arguments.camera_id,
                                track_id,
                                features,
                                risk_result,
                            )

                            with event_path.open(
                                "a",
                                encoding="utf-8",
                            ) as event_file:
                                event_file.write(
                                    json.dumps(event) + "\n"
                                )

                            alert_recorded.add(track_id)

                            print(
                                f"ALERT: anonymous track {track_id}, "
                                f"risk={risk_result['risk_score']:.3f}"
                            )

                    colour = state_colour(
                        risk_result["behaviour_state"]
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        colour,
                        2,
                    )

                    trail_points = [
                        (
                            int(item["x"] * width),
                            int(item["y"] * height),
                        )
                        for item in list(
                            histories[track_id]
                        )[-TRAIL_LENGTH:]
                    ]

                    for index in range(
                        1,
                        len(trail_points),
                    ):
                        cv2.line(
                            annotated_frame,
                            trail_points[index - 1],
                            trail_points[index],
                            colour,
                            2,
                            cv2.LINE_AA,
                        )

                    text_x = max(5, int(x1))
                    text_y = max(20, int(y1) - 55)

                    draw_text(
                        annotated_frame,
                        (
                            f"Anonymous ID {track_id} | "
                            f"{risk_result['behaviour_state']}"
                        ),
                        (text_x, text_y),
                        colour,
                    )

                    draw_text(
                        annotated_frame,
                        (
                            f"Dwell {features['dwell_time']:.1f}s | "
                            f"Risk {risk_result['risk_score']:.2f}"
                        ),
                        (text_x, text_y + 18),
                        colour,
                    )

                    draw_text(
                        annotated_frame,
                        (
                            f"Quality {features['track_quality']:.2f} | "
                            f"Uncertainty {risk_result['uncertainty']:.2f}"
                        ),
                        (text_x, text_y + 36),
                        colour,
                    )

                    explanation = (
                        risk_result["explanation_signals"][0]
                    )

                    draw_text(
                        annotated_frame,
                        f"Evidence: {explanation}",
                        (text_x, text_y + 54),
                        colour,
                    )

            panel_text = (
                f"Camera: {arguments.camera_id} | "
                f"Tracks: {active_track_count} | "
                f"Alerts: {alert_count} | "
                "Q: quit"
            )

            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (width, 34),
                (0, 0, 0),
                -1,
            )

            draw_text(
                annotated_frame,
                panel_text,
                (10, 23),
                (255, 255, 255),
                0.55,
            )

            if writer is not None:
                writer.write(annotated_frame)

            if not arguments.no_display:
                cv2.imshow(
                    "C2 Predictive Loitering Prototype",
                    annotated_frame,
                )

                if cv2.waitKey(1) & 0xFF in (
                    ord("q"),
                    ord("Q"),
                ):
                    break

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    print()
    print("Real-time session completed")
    print("---------------------------")
    print(f"Frames processed: {frame_number}")
    print(f"Track IDs observed: {len(histories)}")
    print(f"Alerts recorded: {len(alert_recorded)}")
    print(f"Event file: {event_path}")

    if arguments.save_video:
        print(f"Demo video: {output_video_path}")


def main():
    arguments = parse_arguments()
    run_realtime(arguments)


if __name__ == "__main__":
    main()