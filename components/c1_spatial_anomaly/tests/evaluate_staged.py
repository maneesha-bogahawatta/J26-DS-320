import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Resolve paths
COMPONENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = COMPONENT_ROOT.parent.parent
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic import SpatialAnomalyEngine

# Per-clip geometry calibrations matched to video filenames
CLIP_CONFIGS = {
    "T_001.mp4": {
        "portal_line": ((250, 340), (550, 340)),
        "restricted_polygon": None,
        "tailgating_time_threshold": 2.5,
        "crowd_density_limit": 6
    },
    "T_002.mp4": {
        "portal_line": ((100, 480), (950, 480)),
        "restricted_polygon": None,
        "tailgating_time_threshold": 1.5,
        "crowd_density_limit": 6
    },
    "T_003.mp4": {
        "portal_line": ((150, 420), (900, 420)),
        "restricted_polygon": None,
        "tailgating_time_threshold": 1.5,
        "crowd_density_limit": 12
    },
    "T_004.mp4": {
        "portal_line": ((20, 500), (550, 500)),
        "restricted_polygon": None,
        "turnstile_barrier_polygon": [[50, 300], [520, 300], [520, 620], [50, 620]],
        "tailgating_time_threshold": 1.5,
        "crowd_density_limit": 6
    },
    "T_005.mp4": {
        "portal_line": ((100, 800), (650, 800)),
        "restricted_polygon": None,
        "tailgating_time_threshold": 3.0,
        "crowd_density_limit": 6
    },
    "I_001.mp4": {
        "portal_line": ((0, 0), (0, 0)),
        "restricted_polygon": [[20, 100], [550, 100], [550, 950], [20, 950]],
        "tailgating_time_threshold": 1.5,
        "crowd_density_limit": 6
    }
}

DEFAULT_CONFIG = {
    "portal_line": ((200, 450), (900, 450)),
    "restricted_polygon": None,
    "tailgating_time_threshold": 1.5,
    "crowd_density_limit": 6
}


def evaluate_staged_clips(base_raw_videos_dir, ground_truth_json):
    if not os.path.exists(ground_truth_json):
        print(f"[ERROR] Ground truth JSON not found at: {ground_truth_json}")
        return

    with open(ground_truth_json, "r") as f:
        gt_data = json.load(f)

    model = YOLO("yolov8n.pt")
    
    tp = 0
    fp = 0
    fn = 0

    print("\n================ STARTING REAL-WORLD STAGED EVALUATION ================\n")

    for rel_video_path, gt_info in gt_data.items():
        clean_rel_path = rel_video_path.replace("/", os.sep)
        full_video_path = os.path.join(base_raw_videos_dir, clean_rel_path)
        
        if not os.path.exists(full_video_path):
            print(f"[WARN] Skipping missing file: {full_video_path}")
            continue

        print(f" -> Processing Clip: {rel_video_path}")
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        # Extract base filename (e.g. "T_001.mp4") to match CLIP_CONFIGS keys accurately
        filename_key = os.path.basename(rel_video_path)
        cfg = CLIP_CONFIGS.get(filename_key, DEFAULT_CONFIG)

        engine = SpatialAnomalyEngine(
            portal_line=cfg["portal_line"],
            restricted_polygon=cfg.get("restricted_polygon"),
            turnstile_barrier_polygon=cfg.get("turnstile_barrier_polygon"),
            tailgating_time_threshold=cfg["tailgating_time_threshold"],
            crowd_density_limit=cfg["crowd_density_limit"],
            fps=fps
        )

        frame_num = 0
        expected_events = gt_info.get("expected_events", [])
        detected_event_windows = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=0.4, verbose=False
            )

            tracks = []
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                tracks = list(zip(ids, boxes))

            alerts = engine.process_frame(tracks, camera_id=rel_video_path, frame_number=frame_num)
            
            for alert in alerts:
                detected_event_windows.append((frame_num, alert["event_type"]))

        cap.release()

        # Match predictions against ground truth
        triggered_gt_indices = set()
        matched_detections = set()

        for gt_idx, gt_evt in enumerate(expected_events):
            e_type = gt_evt["event_type"]
            f_start = gt_evt["frame_start"] - 30
            f_end = gt_evt["frame_end"] + 30

            match = False
            for det_idx, (f_idx, det_type) in enumerate(detected_event_windows):
                if det_type == e_type and f_start <= f_idx <= f_end:
                    match = True
                    matched_detections.add(det_idx)
                    break
            
            if match:
                tp += 1
                triggered_gt_indices.add(gt_idx)
            else:
                fn += 1

        # Detections outside expected windows count as False Positives
        unmatched_fp_types = set()
        for det_idx, (f_idx, det_type) in enumerate(detected_event_windows):
            if det_idx in matched_detections:
                continue
            
            is_valid = any(
                gt_evt["event_type"] == det_type and 
                (gt_evt["frame_start"] - 30) <= f_idx <= (gt_evt["frame_end"] + 30)
                for gt_evt in expected_events
            )
            if not is_valid and det_type not in unmatched_fp_types:
                fp += 1
                unmatched_fp_types.add(det_type)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n================ ACCURACY METRICS REPORT ================")
    print(f"True Positives  (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision:           {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Recall:              {recall:.4f} ({recall * 100:.1f}%)")
    print(f"F1-Score:            {f1_score:.4f} ({f1_score * 100:.1f}%)")
    print("========================================================\n")


if __name__ == "__main__":
    base_raw_dir = r"D:\J26-DS-320\datasets\c1_spatial\raw_videos"
    gt_json = r"components\c1_spatial_anomaly\tests\staged_ground_truth.json"
    evaluate_staged_clips(base_raw_dir, gt_json)