import os
import sys
import glob
import time
from pathlib import Path

# Add component root directory ('components/c1_spatial_anomaly') to sys.path
COMPONENT_ROOT = Path(__file__).resolve().parent.parent
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
from src.spatial_logic import SpatialAnomalyEngine

class SequenceLoader:
    """Loads video frames from an MP4 file or an image folder directory."""
    def __init__(self, source_path):
        self.is_dir = os.path.isdir(source_path)
        if self.is_dir:
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            self.files = []
            for p in patterns:
                self.files.extend(glob.glob(os.path.join(source_path, p)))
            self.files = sorted(self.files)
            self.idx = 0
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(source_path)
            self.files = []

    def read_frame(self):
        if self.is_dir:
            if self.idx >= len(self.files):
                return False, None
            frame = cv2.imread(self.files[self.idx])
            self.idx += 1
            return (frame is not None), frame
        else:
            return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

def run_benchmark(dataset_root, line_y=320, max_sequences=5):
    model = YOLO("yolov8n.pt")
    engine = SpatialAnomalyEngine(
        portal_line=((100, line_y), (750, line_y)),
        tailgating_time_threshold=1.0,
        crowd_density_limit=4
    )

    # Detect either subdirectories (e.g., 01_001) or video files (.mp4)
    entries = sorted([
        os.path.join(dataset_root, d) for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) or d.endswith(".mp4")
    ])[:max_sequences]

    if not entries:
        print(f"[ERROR] No valid sequences found in {dataset_root}")
        return

    total_frames = 0
    total_latency_ms = 0.0
    frame_latencies = []
    alerts_summary = {"TAILGATING": 0, "ZONE_INTRUSION": 0, "CROWD_FORMATION": 0}

    print(f"[INFO] Benchmarking {len(entries)} sequence(s) from {dataset_root}...")

    for seq_path in entries:
        loader = SequenceLoader(seq_path)
        seq_name = os.path.basename(seq_path)
        seq_frames = 0
        print(f" -> Processing sequence: {seq_name}")

        while True:
            ret, frame = loader.read_frame()
            if not ret:
                break

            total_frames += 1
            seq_frames += 1

            t_start = time.perf_counter()

            # ByteTrack inference on person class 0
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                verbose=False
            )

            tracks = []
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                tracks = list(zip(ids, boxes))

            # Process spatial anomaly engine
            alerts = engine.process_frame(tracks, camera_id=seq_name)

            dt_ms = (time.perf_counter() - t_start) * 1000.0
            total_latency_ms += dt_ms
            frame_latencies.append(dt_ms)

            for alert in alerts:
                event_type = alert.get("event_type", "UNKNOWN")
                alerts_summary[event_type] = alerts_summary.get(event_type, 0) + 1

        loader.release()
        print(f"    Completed {seq_frames} frames.")

    avg_latency = total_latency_ms / total_frames if total_frames > 0 else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    p95_latency = sorted(frame_latencies)[int(len(frame_latencies) * 0.95)] if frame_latencies else 0.0

    print("\n================ COMPONENT 1 BENCHMARK REPORT ================")
    print(f"Total Video Frames Processed: {total_frames}")
    print(f"Average Frame Latency:        {avg_latency:.2f} ms")
    print(f"95th Percentile Latency:      {p95_latency:.2f} ms")
    print(f"Inference Throughput:         {fps:.1f} FPS")
    print(f"Total Detected Anomalies:     {sum(alerts_summary.values())}")
    print("Detected Events Breakdown:    ", alerts_summary)
    print("==============================================================")

if __name__ == "__main__":
    benchmark_dir = r"D:\J26-DS-320\datasets\c1_spatial\raw_videos\public_benchmarks\shanghaitech"
    run_benchmark(benchmark_dir)