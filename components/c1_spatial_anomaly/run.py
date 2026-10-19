import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Resolve paths
COMPONENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = COMPONENT_ROOT.parent.parent
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic import SpatialAnomalyEngine

# Calibrated geometry lines per video canvas resolution
CLIP_CONFIGS = {
    "T_001.mp4": {
        "portal_line": ((250, 300), (550, 440)),
        "restricted_polygon": None,
        "tailgating_time_threshold": 0.1,
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
        # Door floor threshold for 720x1280 vertical resolution
        "portal_line": ((100, 800), (650, 800)),
        "restricted_polygon": None,
        "tailgating_time_threshold": 3.0,          # 3.0s window for exit button press & door swing
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


def main():
    parser = argparse.ArgumentParser(description="Run Spatial Anomaly Detection Engine")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file or stream")
    args = parser.parse_args()

    video_path = args.source
    if not os.path.exists(video_path):
        print(f"[ERROR] Source video not found: {video_path}")
        return

    video_filename = os.path.basename(video_path)
    cfg = CLIP_CONFIGS.get(video_filename, DEFAULT_CONFIG)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    print(f"[C1] Opened '{video_filename}' ({width}x{height} @ {fps:.1f} FPS)")

    model = YOLO("yolov8n.pt")
    engine = SpatialAnomalyEngine(
        portal_line=cfg["portal_line"],
        restricted_polygon=cfg.get("restricted_polygon"),
        turnstile_barrier_polygon=cfg.get("turnstile_barrier_polygon"),
        tailgating_time_threshold=cfg["tailgating_time_threshold"],
        crowd_density_limit=cfg["crowd_density_limit"],
        fps=fps
    )

    frame_count = 0
    window_name = f"Component 1 - Spatial Anomaly Engine [{video_filename}]"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            tracks = list(zip(ids, boxes))

        alerts = engine.process_frame(tracks, camera_id=video_filename, frame_number=frame_count)

        # 1. Render Portal Line
        p1, p2 = cfg["portal_line"]
        if p1 != (0, 0) or p2 != (0, 0):
            cv2.line(frame, p1, p2, (255, 255, 0), 2)
            cv2.putText(frame, "PORTAL THRESHOLD", (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 2. Render Pedestrian Footprints & Bounding Boxes
        flagged_ids = engine.flagged_tailgaters.union(engine.active_intrusions)
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)
            color = (0, 0, 255) if track_id in flagged_ids else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"ID:{track_id} ({foot_x},{foot_y})", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Print Console Alerts & Render GUI HUD
        if alerts:
            for alert in alerts:
                signals_str = ", ".join(alert.get("explanation_signals", []))
                person_str = alert.get("person_id", "P-UNKNOWN")
                print(f"[C1 ALERT {alert['event_type'].upper()}] {person_str} -> {signals_str}")

            cv2.putText(frame, f"ANOMALY: {alerts[0]['event_type'].upper()}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[C1] Video playback completed.")


if __name__ == "__main__":
    main()