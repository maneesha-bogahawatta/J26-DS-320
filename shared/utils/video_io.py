"""
Shared video I/O utilities.

All members use these functions to read video, extract frames,
and save evidence clips. This prevents 4 different implementations
of the same thing.
"""

import cv2
import os
from typing import Generator, Tuple, Optional


def read_video_frames(
    source: str,
    resize: Optional[Tuple[int, int]] = None,
    skip_frames: int = 0,
) -> Generator[Tuple[int, "cv2.Mat"], None, None]:
    """
    Read frames from a video file or camera stream.

    Args:
        source: path to video file, or camera index (0, 1, ...)
                or RTSP URL for live CCTV stream
        resize: optional (width, height) to resize frames
        skip_frames: process every Nth frame (0 = every frame)

    Yields:
        (frame_number, frame) tuples

    Usage:
        for frame_num, frame in read_video_frames("mall_cam.mp4"):
            detections = detector(frame)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            if resize:
                frame = cv2.resize(frame, resize)

            yield frame_count, frame
            frame_count += 1
    finally:
        cap.release()


def get_video_info(source: str) -> dict:
    """
    Get metadata about a video file.

    Returns:
        dict with keys: fps, width, height, total_frames, duration_sec
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def save_evidence_frame(
    frame,
    output_dir: str,
    camera_id: str,
    frame_number: int,
    event_type: str,
) -> str:
    """
    Save a single frame as evidence for an alert.

    Returns:
        path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{camera_id}_frame{frame_number}_{event_type}.jpg"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, frame)
    return path


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        info = get_video_info(sys.argv[1])
        print(f"Video info: {info}")
        for i, (fn, frame) in enumerate(read_video_frames(sys.argv[1])):
            print(f"Frame {fn}: shape={frame.shape}")
            if i >= 4:
                break
    else:
        print("Usage: python video_io.py <video_file>")
