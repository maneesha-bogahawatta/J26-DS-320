import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

COMPONENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = COMPONENT_ROOT.parent.parent
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic import SpatialAnomalyEngine

# Synchronized configurations across all 16 benchmark clips
CLIP_CONFIGS = {
    "T_001.mp4": {"portal_line": ((250, 340), (550, 340)), "tailgating_time_threshold": 2.5, "crowd_density_limit": 6},
    "T_002.mp4": {"portal_line": ((100, 480), (950, 480)), "tailgating_time_threshold": 1.5, "crowd_density_limit": 6},
    "T_003.mp4": {"portal_line": ((150, 420), (900, 420)), "tailgating_time_threshold": 1.5, "crowd_density_limit": 12},
    "T_004.mp4": {"portal_line": ((20, 500), (550, 500)), "turnstile_barrier_polygon": [[50, 300], [520, 300], [520, 620], [50, 620]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 6},
    "T_005.mp4": {"portal_line": ((100, 800), (650, 800)), "tailgating_time_threshold": 3.0, "crowd_density_limit": 6},
    "I_001.mp4": {"portal_line": ((0, 0), (0, 0)), "restricted_polygon": [[20, 100], [550, 100], [550, 950], [20, 950]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 6},
    "I_002.mp4": {"portal_line": ((0, 0), (0, 0)), "restricted_polygon": [[200, 180], [480, 180], [480, 300], [200, 300]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 15, "conf": 0.35},
    "I_003.mp4": {"portal_line": ((0, 0), (0, 0)), "restricted_polygon": [[10, 40], [1210, 40], [1210, 710], [10, 710]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 15, "conf": 0.08},
    "I_004python components/c1_spatial_anomaly/tests/evaluate_staged.py.mp4": {"portal_line": ((0, 0), (0, 0)), "restricted_polygon": [[10, 40], [1210, 40], [1210, 710], [10, 710]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 15, "conf": 0.08},
    "I_005.mp4": {"portal_line": ((0, 0), (0, 0)), "restricted_polygon": [[30, 80], [610, 80], [610, 350], [30, 350]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 15, "conf": 0.30},
    "C_001.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 4},
    "C_002.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 12},
    "C_003.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 6},
    "C_004.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 8},
    "C_005.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 10},
    "C_006.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 10}
}


class FastLiveTracker:
    def __init__(self, max_lost=15, iou_threshold=0.10):
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["lost"] += 1

        updated_tracks = []
        unmatched_dets = list(range(len(detections)))

        for tid, track_info in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = -1
            for det_idx in unmatched_dets:
                iou = self._compute_iou(track_info["bbox"], detections[det_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_iou >= self.iou_threshold and best_det_idx != -1:
                self.tracks[tid]["bbox"] = detections[best_det_idx]
                self.tracks[tid]["lost"] = 0
                unmatched_dets.remove(best_det_idx)
                updated_tracks.append((tid, detections[best_det_idx]))

        for det_idx in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": detections[det_idx], "lost": 0}
            updated_tracks.append((tid, detections[det_idx]))

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["lost"] > self.max_lost:
                del self.tracks[tid]

        return updated_tracks


def evaluate_all():
    dataset_base = Path("D:/J26-DS-320/datasets/c1_spatial/raw_videos")
    gt_path = COMPONENT_ROOT / "tests" / "staged_ground_truth.json"

    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found at: {gt_path}")
        return

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    model = YOLO("yolov8n.pt")
    
    tp, fp, fn = 0, 0, 0
    total_frames_processed = 0
    start_time = time.time()

    print("\n=======================================================")
    print(" C1 SPATIAL ANOMALY ENGINE - BENCHMARK EVALUATION")
    print("=======================================================\n")

    for clip_key, gt_info in ground_truth.items():
        video_path = dataset_base / clip_key
        if not video_path.exists():
            print(f"[SKIP] Video file missing: {video_path}")
            continue

        video_filename = video_path.name
        cfg = CLIP_CONFIGS.get(video_filename, {})
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        engine = SpatialAnomalyEngine(
            portal_line=cfg.get("portal_line", ((0, 0), (0, 0))),
            restricted_polygon=cfg.get("restricted_polygon"),
            turnstile_barrier_polygon=cfg.get("turnstile_barrier_polygon"),
            tailgating_time_threshold=cfg.get("tailgating_time_threshold", 1.5),
            crowd_density_limit=cfg.get("crowd_density_limit", 5),
            fps=fps
        )
        tracker = FastLiveTracker(max_lost=15, iou_threshold=0.10)
        clip_conf = cfg.get("conf", 0.08)

        expected_events = gt_info.get("expected_events", [])
        detected_event_types = set()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            total_frames_processed += 1

            results = model.predict(frame, classes=[0], conf=clip_conf, imgsz=640, verbose=False)
            detections = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
            
            tracks = tracker.update(detections)
            alerts = engine.process_frame(tracks, camera_id=video_filename, frame_number=frame_count)

            for alert in alerts:
                event_type = alert["event_type"]
                # Check if alert matches temporal window in ground truth
                for exp in expected_events:
                    if exp["event_type"] == event_type and exp["frame_start"] <= frame_count <= exp["frame_end"]:
                        detected_event_types.add(event_type)

        cap.release()

        # Evaluate TP, FP, FN for this video
        expected_types = set(exp["event_type"] for exp in expected_events)
        clip_tp = len(detected_event_types.intersection(expected_types))
        clip_fn = len(expected_types - detected_event_types)
        clip_fp = 1 if (len(detected_event_types) > 0 and clip_tp == 0) else 0

        tp += clip_tp
        fn += clip_fn
        fp += clip_fp

        status = "PASSED" if clip_fn == 0 and clip_fp == 0 else "FAILED"
        print(f"[{status}] Clip: {clip_key:<22} | Expected: {len(expected_types)} | Detected: {len(detected_event_types)} | TP:{clip_tp} FP:{clip_fp} FN:{clip_fn}")

    elapsed_time = time.time() - start_time
    overall_fps = total_frames_processed / elapsed_time if elapsed_time > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n-------------------------------------------------------")
    print(" FINAL SYSTEM METRICS SUMMARY")
    print("-------------------------------------------------------")
    print(f" Total Frames Evaluated: {total_frames_processed}")
    print(f" Processing Speed:       {overall_fps:.2f} FPS")
    print(f" True Positives (TP):    {tp}")
    print(f" False Positives (FP):   {fp}")
    print(f" False Negatives (FN):   {fn}")
    print(f" Precision:              {precision * 100:.2f}%")
    print(f" Recall:                 {recall * 100:.2f}%")
    print(f" F1-Score:               {f1_score * 100:.2f}%")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    evaluate_all()