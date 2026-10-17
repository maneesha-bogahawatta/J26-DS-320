"""Extract anonymous person trajectories from an ATMA-V video.

Uses a pretrained YOLO11 person detector with ByteTrack. The script writes
frame-level tracking coordinates to CSV and an annotated demonstration video.
"""

import argparse
import os
from collections import defaultdict, deque
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


DEFAULT_ATMA_ROOT = Path(
    os.environ.get("ATMA_V_ROOT", "ATMA-V Dataset")
)

DEFAULT_WORK_ROOT = Path(
    os.environ.get("C2_WORK_ROOT", ".")
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract ATMA-V person trajectories "
            "with YOLO11 and ByteTrack."
        )
    )

    parser.add_argument(
        "--video",
        default="12.mp4",
        help=(
            "Video filename inside ATMA_V_ROOT/videos "
            "or a complete video path."
        ),
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
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Use cpu, 0 for the first GPU, or another device.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Optional frame limit for a quick test. "
            "Use 0 to process the complete video."
        ),
    )

    return parser.parse_args()


def resolve_video_path(video_argument):
    """Locate the requested video."""

    requested_path = Path(video_argument)

    if requested_path.exists():
        return requested_path.resolve()

    dataset_path = (
        DEFAULT_ATMA_ROOT
        / "videos"
        / video_argument
    )

    if dataset_path.exists():
        return dataset_path.resolve()

    raise FileNotFoundError(
        "ATMA-V video was not found. Checked:\n"
        f"  {requested_path}\n"
        f"  {dataset_path}"
    )


def safe_fps(capture):
    """Return the video FPS or a safe fallback."""

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if fps > 0:
        return fps

    return 25.0


def draw_track_trail(frame, points):
    """Draw the recent movement trail for one person."""

    if len(points) < 2:
        return

    for index in range(1, len(points)):
        cv2.line(
            frame,
            points[index - 1],
            points[index],
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def extract_tracks(
    video_path,
    model_name,
    confidence,
    image_size,
    device,
    max_frames,
):
    """Detect and track people in one video."""

    output_root = (
        DEFAULT_WORK_ROOT
        / "outputs"
        / "atma_tracking"
    )

    csv_root = (
        DEFAULT_WORK_ROOT
        / "data"
        / "raw_tracks"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_root
        / f"{video_path.stem}_tracked.mp4"
    )

    output_csv = (
        csv_root
        / f"{video_path.stem}_trajectories.csv"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = safe_fps(capture)

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not create video: {output_video}"
        )

    model = YOLO(model_name)

    trajectory_rows = []

    trails = defaultdict(
        lambda: deque(maxlen=75)
    )

    frame_number = 0

    print(f"Input video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Declared frames: {total_frames}")
    print(f"Device: {device}")

    try:
        while True:
            # Stop before reading an extra frame.
            if (
                max_frames > 0
                and frame_number >= max_frames
            ):
                break

            success, frame = capture.read()

            if not success:
                break

            frame_number += 1

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
            )[0]

            annotated_frame = frame.copy()
            active_tracks = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                for (
                    box,
                    track_id,
                    track_confidence,
                ) in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    x1, y1, x2, y2 = [
                        float(value)
                        for value in box
                    ]

                    centre_x = (
                        x1 + x2
                    ) / 2.0

                    centre_y = (
                        y1 + y2
                    ) / 2.0

                    normalized_x = (
                        centre_x / width
                    )

                    normalized_y = (
                        centre_y / height
                    )

                    active_tracks += 1

                    trajectory_rows.append(
                        {
                            "dataset_source": "ATMA-V",
                            "video_id": video_path.stem,
                            "frame_number": frame_number,
                            "timestamp": round(
                                (frame_number - 1) / fps,
                                4,
                            ),
                            "track_id": int(track_id),
                            "x1": round(x1, 3),
                            "y1": round(y1, 3),
                            "x2": round(x2, 3),
                            "y2": round(y2, 3),
                            "centre_x": round(
                                normalized_x,
                                6,
                            ),
                            "centre_y": round(
                                normalized_y,
                                6,
                            ),
                            "box_width": round(
                                (x2 - x1) / width,
                                6,
                            ),
                            "box_height": round(
                                (y2 - y1) / height,
                                6,
                            ),
                            "detection_confidence": round(
                                float(
                                    track_confidence
                                ),
                                6,
                            ),
                            "detector": model_name,
                            "tracker": "ByteTrack",
                        }
                    )

                    centre = (
                        int(centre_x),
                        int(centre_y),
                    )

                    trails[
                        int(track_id)
                    ].append(centre)

                    draw_track_trail(
                        annotated_frame,
                        trails[int(track_id)],
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (
                            int(x1),
                            int(y1),
                        ),
                        (
                            int(x2),
                            int(y2),
                        ),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated_frame,
                        (
                            f"Person {track_id} "
                            f"{track_confidence:.2f}"
                        ),
                        (
                            int(x1),
                            max(
                                20,
                                int(y1) - 8,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                annotated_frame,
                (
                    f"Frame {frame_number} | "
                    f"Active tracks: "
                    f"{active_tracks}"
                ),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                annotated_frame
            )

            if frame_number % 100 == 0:
                print(
                    f"Processed "
                    f"{frame_number} frames"
                )

    finally:
        capture.release()
        writer.release()

    columns = [
        "dataset_source",
        "video_id",
        "frame_number",
        "timestamp",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "centre_x",
        "centre_y",
        "box_width",
        "box_height",
        "detection_confidence",
        "detector",
        "tracker",
    ]

    trajectories = pd.DataFrame(
        trajectory_rows,
        columns=columns,
    )

    trajectories.to_csv(
        output_csv,
        index=False,
    )

    if trajectories.empty:
        unique_tracks = 0
    else:
        unique_tracks = (
            trajectories["track_id"]
            .nunique()
        )

    print()
    print("Tracking completed")
    print("------------------")
    print(
        f"Processed frames: "
        f"{frame_number}"
    )
    print(
        f"Trajectory rows: "
        f"{len(trajectories)}"
    )
    print(
        f"Unique track IDs: "
        f"{unique_tracks}"
    )
    print(
        f"Trajectory CSV: "
        f"{output_csv}"
    )
    print(
        f"Annotated video: "
        f"{output_video}"
    )


def main():
    arguments = parse_arguments()

    video_path = resolve_video_path(
        arguments.video
    )

    extract_tracks(
        video_path=video_path,
        model_name=arguments.model,
        confidence=arguments.confidence,
        image_size=arguments.image_size,
        device=arguments.device,
        max_frames=arguments.max_frames,
    )


if __name__ == "__main__":
    main()