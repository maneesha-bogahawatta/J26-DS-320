import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

COMPONENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = COMPONENT_ROOT.parent.parent
if str(COMPONENT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPONENT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_logic import SpatialAnomalyEngine


class FastLiveTracker:
    """Lightweight real-time tracker for live CCTV streams."""
    def __init__(self, max_lost=10, iou_threshold=0.10):
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


CLIP_CONFIGS = {
    "T_001.mp4": {"portal_line": ((250, 340), (550, 340)), "tailgating_time_threshold": 2.5, "crowd_density_limit": 6},
    "T_002.mp4": {"portal_line": ((100, 480), (950, 480)), "tailgating_time_threshold": 1.5, "crowd_density_limit": 6},
    "T_003.mp4": {"portal_line": ((150, 420), (900, 420)), "tailgating_time_threshold": 1.5, "crowd_density_limit": 12},
    "T_004.mp4": {"portal_line": ((20, 500), (550, 500)), "turnstile_barrier_polygon": [[50, 300], [520, 300], [520, 620], [50, 620]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 6},
    "T_005.mp4": {"portal_line": ((100, 800), (650, 800)), "tailgating_time_threshold": 3.0, "crowd_density_limit": 6},
    "I_001.mp4": {"portal_line": ((0, 0), (0, 0)), "restricted_polygon": [[20, 100], [550, 100], [550, 950], [20, 950]], "tailgating_time_threshold": 1.5, "crowd_density_limit": 6},
    "C_001.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 5},
    "C_002.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 12},
    "C_003.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 10},
    "C_004.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 8},
    "C_005.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 10},
    "C_006.mp4": {"portal_line": ((0, 0), (0, 0)), "crowd_density_limit": 10}
}

DEFAULT_CONFIG = {
    "portal_line": ((0, 0), (0, 0)),
    "restricted_polygon": None,
    "tailgating_time_threshold": 1.5,
    "crowd_density_limit": 5
}


def main():
    parser = argparse.ArgumentParser(description="Run Spatial Anomaly Engine for Live CCTV Streams")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or RTSP CCTV stream")
    args = parser.parse_args()

    video_path = args.source
    if not os.path.exists(video_path) and not video_path.startswith("rtsp://"):
        print(f"[ERROR] Source stream/file not found: {video_path}")
        return

    video_filename = os.path.basename(video_path)
    cfg = CLIP_CONFIGS.get(video_filename, DEFAULT_CONFIG)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    print(f"[C1 LIVE] Opened '{video_filename}' ({width}x{height} @ {fps:.1f} FPS)")

    # Lightweight nano model for true 30+ FPS real-time processing
    model = YOLO("yolov8n.pt")
    tracker = FastLiveTracker(max_lost=10, iou_threshold=0.10)
    
    engine = SpatialAnomalyEngine(
        portal_line=cfg["portal_line"],
        restricted_polygon=cfg.get("restricted_polygon"),
        turnstile_barrier_polygon=cfg.get("turnstile_barrier_polygon"),
        tailgating_time_threshold=cfg.get("tailgating_time_threshold", 1.5),
        crowd_density_limit=cfg.get("crowd_density_limit", 5),
        fps=fps
    )

    frame_count = 0
    window_name = f"Component 1 - Live Spatial Engine [{video_filename}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Fast 640px inference with conf=0.08 to capture low-contrast sprinters at real-time speeds
        results = model.predict(
            frame, 
            classes=[0], 
            conf=0.08, 
            imgsz=640, 
            verbose=False
        )

        detections = []
        if len(results[0].boxes) > 0:
            detections = results[0].boxes.xyxy.cpu().numpy()

        tracks = tracker.update(detections)
        alerts = engine.process_frame(tracks, camera_id=video_filename, frame_number=frame_count)

        # 1. Render Portal Line
        p1, p2 = cfg["portal_line"]
        if p1 != (0, 0) or p2 != (0, 0):
            cv2.line(frame, p1, p2, (255, 255, 0), 2)

        # 2. Render Bounding Boxes & IDs
        flagged_ids = engine.flagged_tailgaters.union(engine.active_intrusions)
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            foot_x, foot_y = int((x1 + x2) / 2), int(y2)
            color = (0, 0, 255) if track_id in flagged_ids else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (foot_x, foot_y), 4, (0, 255, 255), -1)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Render HUD Banner
        if alerts:
            for alert in alerts:
                signals_str = ", ".join(alert.get("explanation_signals", []))
                person_str = alert.get("person_id", "P-UNKNOWN")
                print(f"[C1 ALERT {alert['event_type'].upper()}] {person_str} -> {signals_str}")

            cv2.putText(frame, f"ANOMALY: {alerts[0]['event_type'].upper()} ({len(tracks)} PEOPLE)", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[C1] Live stream processing completed.")


if __name__ == "__main__":
    main()