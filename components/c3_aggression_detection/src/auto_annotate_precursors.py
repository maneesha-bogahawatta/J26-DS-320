
import csv
import json
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


INPUT = Path("data/manifests/ubi_early_review_candidates.csv")
OUTPUT = Path("data/manifests/ubi_auto_precursors.csv")
REVIEW = Path("data/manifests/ubi_manual_review_queue.csv")

PRE_SECONDS = 5.0
SAMPLE_FPS = 5.0
POSE_CONF = 0.30

# Heuristic thresholds, NOT trained probabilities
SCORE_THRESHOLD = 0.45
MIN_CONSECUTIVE = 2


def read_manifest(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def get_persons(result):
    """Extract tracked person boxes and COCO pose keypoints."""
    persons = []

    if result.boxes is None or result.boxes.id is None:
        return persons

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.int().cpu().tolist()

    if result.keypoints is None:
        return persons

    xy = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        conf = result.keypoints.conf.cpu().numpy()
    else:
        conf = np.ones(xy.shape[:2], dtype=np.float32)

    for i, track_id in enumerate(ids):
        x1, y1, x2, y2 = boxes[i]
        height = max(float(y2 - y1), 1.0)

        center = np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2
        ], dtype=float)

        persons.append({
            "id": int(track_id),
            "center": center,
            "height": height,
            "keypoints": xy[i],
            "confidence": conf[i]
        })

    return persons


def calculate_signals(persons, previous, dt):
    """
    Returns movement features based only on past/current frames.
    All thresholds are experimental heuristics.
    """
    signals = []
    dt = max(float(dt), 1e-3)

    # 1. Raised-arm cue
    for p in persons:
        kp = p["keypoints"]
        conf = p["confidence"]
        h = p["height"]

        for shoulder, wrist in [(5, 9), (6, 10)]:
            if (
                conf[shoulder] > POSE_CONF
                and conf[wrist] > POSE_CONF
            ):
                arm_raise = (
                    kp[shoulder, 1] - kp[wrist, 1]
                ) / h

                if arm_raise > 0.20:
                    signals.append("raised_arm")

    # 2. Person movement speed
    speeds = []

    for p in persons:
        pid = p["id"]

        if pid not in previous:
            continue

        old = previous[pid]

        distance = np.linalg.norm(
            p["center"] - old["center"]
        )

        normalized_speed = distance / p["height"] / dt
        speeds.append(float(normalized_speed))

    if speeds and max(speeds) > 0.9:
        signals.append("rapid_movement")

    # 3. Two-person approach
    if len(persons) >= 2:
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a = persons[i]
                b = persons[j]

                if a["id"] not in previous:
                    continue
                if b["id"] not in previous:
                    continue

                avg_height = max(
                    (a["height"] + b["height"]) / 2,
                    1.0
                )

                current_distance = (
                    np.linalg.norm(a["center"] - b["center"])
                    / avg_height
                )

                old_a = previous[a["id"]]
                old_b = previous[b["id"]]

                previous_distance = (
                    np.linalg.norm(old_a["center"] - old_b["center"])
                    / avg_height
                )

                closing_speed = (
                    previous_distance - current_distance
                ) / dt

                if (
                    closing_speed > 0.35
                    and current_distance < 2.0
                ):
                    signals.append("rapid_approach")
                    break

    # Avoid scoring the same signal repeatedly
    signals = sorted(set(signals))

    return signals


def calculate_score(signals):
    """
    Explainable cue score.
    Not a calibrated aggression probability.
    """
    weights = {
        "raised_arm": 0.25,
        "rapid_movement": 0.25,
        "rapid_approach": 0.40
    }

    score = sum(weights.get(s, 0.0) for s in signals)

    # More than one cue provides stronger evidence
    if len(signals) >= 2:
        score += 0.15

    return round(min(score, 1.0), 4)


def analyze_video(video_path, onset_s, model):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS")

    start_s = max(0.0, onset_s - PRE_SECONDS)
    start_frame = int(math.ceil(start_s * fps))
    end_frame = int(math.ceil(onset_s * fps))

    step = max(1, int(round(fps / SAMPLE_FPS)))

    # Reset tracking state for each independent video.
    # A new model instance is created per video below.
    previous = {}
    previous_time = None

    observations = []

    try:
        for frame_idx in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ok, frame = cap.read()

            if not ok:
                continue

            timestamp = frame_idx / fps

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.25,
                verbose=False
            )[0]

            persons = get_persons(result)

            if previous_time is None:
                dt = 1.0 / SAMPLE_FPS
            else:
                dt = timestamp - previous_time

            signals = calculate_signals(
                persons,
                previous,
                dt
            )

            score = calculate_score(signals)

            observations.append({
                "timestamp": timestamp,
                "score": score,
                "signals": signals,
                "persons": len(persons)
            })

            previous = {
                p["id"]: {
                    "center": p["center"].copy()
                }
                for p in persons
            }

            previous_time = timestamp

    finally:
        cap.release()

    return observations


def propose_intervals(observations):
    """
    Turn consecutive high-score observations into
    provisional precursor intervals.
    """
    intervals = []
    active = []

    def flush():
        if len(active) >= MIN_CONSECUTIVE:
            start = active[0]["timestamp"]
            end = active[-1]["timestamp"] + (1 / SAMPLE_FPS)

            intervals.append([
                round(start, 3),
                round(end, 3)
            ])

        active.clear()

    for observation in observations:
        if observation["score"] >= SCORE_THRESHOLD:
            active.append(observation)
        else:
            flush()

    flush()
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(INPUT)
    )
    parser.add_argument(
        "--weights",
        default="yolo11n-pose.pt"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))

    if args.limit > 0:
        rows = rows[:args.limit]

    results = []
    review_queue = []

    for index, row in enumerate(rows, start=1):
        video_path = Path(row["video"])
        onset = float(row["fight_onset_s"])

        print(
            f"[{index}/{len(rows)}] "
            f"Analyzing {video_path.name}"
        )

        output = dict(row)

        output["auto_precursors"] = "[]"
        output["auto_max_score"] = ""
        output["auto_signals"] = "[]"
        output["auto_status"] = "needs_review"
        output["auto_error"] = ""

        try:
            # Reset ByteTrack between independent videos
            model = YOLO(args.weights)

            observations = analyze_video(
                video_path,
                onset,
                model
            )

            intervals = propose_intervals(observations)

            signals = sorted({
                s
                for obs in observations
                for s in obs["signals"]
            })

            max_score = max(
                (o["score"] for o in observations),
                default=0.0
            )

            output["auto_precursors"] = json.dumps(intervals)
            output["auto_max_score"] = round(max_score, 4)
            output["auto_signals"] = json.dumps(signals)

            if not observations:
                output["auto_status"] = "no_valid_frames"
            elif intervals:
                output["auto_status"] = "proposed_precursor"
            else:
                output["auto_status"] = "no_precursor_proposed"

        except Exception as exc:
            output["auto_status"] = "error"
            output["auto_error"] = str(exc)
            print("ERROR:", exc)

        results.append(output)

        # All predictions remain unverified;
        # prioritize proposed and failed cases.
        if output["auto_status"] in (
            "proposed_precursor",
            "no_valid_frames",
            "error"
        ):
            review_queue.append(output)

    save_csv(OUTPUT, results)
    save_csv(REVIEW, review_queue)

    print("\n===== AUTOMATIC ANNOTATION SUMMARY =====")
    print("Processed:", len(results))

    print(
        "Proposed precursor clips:",
        sum(
            r["auto_status"] == "proposed_precursor"
            for r in results
        )
    )

    print("Review priority queue:", len(review_queue))
    print("Results:", OUTPUT)
    print("Review queue:", REVIEW)


if __name__ == "__main__":
    main()
