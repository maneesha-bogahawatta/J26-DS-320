"""
Component 4 — Vision-Only Fall and Abandoned-Object Detection
Main entry point.

This script runs both the fall branch and the object branch
on a video source and emits IncidentEvent objects.

Usage:
    python run.py --source video.mp4
    python run.py --source 0          # webcam
    python run.py --source rtsp://... # live CCTV
"""

import argparse
import sys
import os

# Add project root to path so we can import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.utils.video_io import read_video_frames, get_video_info
from shared.schemas.event_schema import IncidentEvent, EVENT_TYPES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Component 4: Fall & Abandoned-Object Detection"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Video file path, camera index, or RTSP URL"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["fall", "object", "both"],
        help="Which branch to run"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Minimum confidence to emit an event"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/logs/c4_run",
        help="Directory for evidence frames and logs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get video info
    try:
        source = int(args.source)  # webcam index
    except ValueError:
        source = args.source       # file path or URL

    print(f"[C4] Starting Component 4: Fall & Abandoned-Object Detection")
    print(f"[C4] Source: {source}")
    print(f"[C4] Mode: {args.mode}")
    print(f"[C4] Confidence threshold: {args.confidence_threshold}")
    print()

    # TODO: Initialize shared perception layer
    # from shared.detection.detect import PersonDetector
    # from shared.tracking.tracker import MultiObjectTracker
    # from shared.pose.estimator import PoseEstimator
    #
    # detector = PersonDetector()
    # tracker = MultiObjectTracker()
    # pose_estimator = PoseEstimator()

    # TODO: Initialize fall branch
    # from fall_branch.temporal_model import FallDetector
    # fall_detector = FallDetector.load("models/fall_cnn_lstm_v1.pt")

    # TODO: Initialize object branch
    # from object_branch.owner_assoc import AbandonedObjectDetector
    # object_detector = AbandonedObjectDetector.load("models/object_assoc_v1.pt")

    # Process frames
    print("[C4] Processing frames...")
    for frame_num, frame in read_video_frames(source):

        # Step 1: Shared perception layer
        # detections = detector(frame)
        # tracks = tracker.update(detections)
        # poses = pose_estimator(frame, detections)

        # Step 2: Fall branch
        if args.mode in ("fall", "both"):
            # fall_result = fall_detector(poses, tracks)
            # if fall_result and fall_result.confidence > args.confidence_threshold:
            #     event = IncidentEvent(...)
            #     print(event.to_json())
            pass

        # Step 3: Object branch
        if args.mode in ("object", "both"):
            # object_result = object_detector(tracks, detections)
            # if object_result and object_result.confidence > args.confidence_threshold:
            #     event = IncidentEvent(...)
            #     print(event.to_json())
            pass

        # Progress indicator
        if frame_num % 100 == 0:
            print(f"[C4] Processed frame {frame_num}")

    print("[C4] Done.")


if __name__ == "__main__":
    main()
