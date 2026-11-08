import os
import sys
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Path Resolution Setup
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

for path in [PROJECT_ROOT, BACKEND_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from detect import video_detector
except ImportError:
    def video_detector(name: str):
        def decorator(func):
            return func
        return decorator

# ---------------------------------------------------------------------------
# FastLiveTracker Implementation (IoU Tracker with Frame Persistence)
# ---------------------------------------------------------------------------
class Track:
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float]):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.lost_frames = 0
        self.has_crossed = False

    def update(self, bbox: Tuple[float, float, float, float]):
        self.bbox = bbox
        self.lost_frames = 0

class FastLiveTracker:
    def __init__(self, iou_threshold: float = 0.15, max_lost: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: List[Track] = []

    @staticmethod
    def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update(self, detections: List[Tuple[float, float, float, float]]) -> List[Track]:
        for track in self.tracks:
            track.lost_frames += 1

        unmatched_detections = list(range(len(detections)))

        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = self.compute_iou(track.bbox, det)

            matched_tracks = set()
            matched_dets = set()

            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_val = iou_matrix[max_idx]

                if max_val < self.iou_threshold:
                    break

                t_idx, d_idx = max_idx
                if t_idx not in matched_tracks and d_idx not in matched_dets:
                    self.tracks[t_idx].update(detections[d_idx])
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

            unmatched_detections = [d for d in range(len(detections)) if d not in matched_dets]

        for det_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[det_idx])
            self.next_id += 1
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]
        return self.tracks

# ---------------------------------------------------------------------------
# Geometric Helper Functions
# ---------------------------------------------------------------------------
def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# ---------------------------------------------------------------------------
# Lazy Load Model
# ---------------------------------------------------------------------------
_model_cache = None

def get_yolo_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO("yolov8n.pt")
    return _model_cache

# ---------------------------------------------------------------------------
# Plugin Entrypoint
# ---------------------------------------------------------------------------
@video_detector("c1_spatial")
def process_c1_video(
    video_path: str,
    source_id: Optional[str] = None,
    user_polygon: Optional[List[Tuple[float, float]]] = None,
    crowd_limit: int = 3,
    **kwargs
) -> List[Dict[str, Any]]:

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    filename = os.path.basename(video_path)

    # 1. Convert user drawn normalized polygon [0.0 - 1.0] to video pixel dimensions
    scaled_polygon = []
    if user_polygon and len(user_polygon) >= 3:
        scaled_polygon = [(int(pt[0] * width), int(pt[1] * height)) for pt in user_polygon]
    else:
        # Fallback default zone if user hasn't drawn one
        scaled_polygon = [
            (int(width * 0.05), int(height * 0.05)),
            (int(width * 0.95), int(height * 0.05)),
            (int(width * 0.95), int(height * 0.95)),
            (int(width * 0.05), int(height * 0.95))
        ]

    model = get_yolo_model()
    tracker = FastLiveTracker(iou_threshold=0.15, max_lost=15)

    events: List[Dict[str, Any]] = []
    triggered_intrusion_ids = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_timestamp = float(frame_idx / fps)

        results = model.predict(frame, conf=0.10, classes=[0], verbose=False)
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                detections.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))

        active_tracks = tracker.update(detections)
        persons_in_restricted_zone = 0
        total_headcount = len(active_tracks)

        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox
            footprint = ((x1 + x2) / 2.0, y2)

            # --- ALERT 1: Restricted Zone Intrusion ---
            if scaled_polygon and point_in_polygon(footprint, scaled_polygon):
                persons_in_restricted_zone += 1
                if track.track_id not in triggered_intrusion_ids:
                    triggered_intrusion_ids.add(track.track_id)
                    events.append({
                        "event_id": f"c1_zone_{track.track_id}_{frame_idx}",
                        "component_id": "component_1",
                        "incident_type": "zone_intrusion",
                        "camera_id": str(source_id or filename),
                        "timestamp_detected": datetime.now().isoformat(),
                        "media_offset_seconds": round(current_timestamp, 2),
                        "priority": "high",
                        "summary": f"Unauthorized intrusion detected by Track P-{track.track_id} in restricted zone.",
                        "model_name": "YOLOv8n",
                        "model_version": "1.0.0",
                        "evidence": [
                            {"name": "track_id", "value": f"P-{track.track_id}"},
                            {"name": "signal", "value": "point_in_polygon_positive"},
                            {"name": "confidence", "value": "0.95"}
                        ]
                    })

        # --- ALERT 2: Crowding Head Count Limit ---
        if total_headcount > crowd_limit:
            # Throttle crowding alerts to max 1 per 3 seconds
            if not any(e["incident_type"] == "crowd_density" and abs(e["media_offset_seconds"] - current_timestamp) < 3.0 for e in events):
                events.append({
                    "event_id": f"c1_crowd_{frame_idx}",
                    "component_id": "component_1",
                    "incident_type": "crowd_density",
                    "camera_id": str(source_id or filename),
                    "timestamp_detected": datetime.now().isoformat(),
                    "media_offset_seconds": round(current_timestamp, 2),
                    "priority": "medium",
                    "summary": f"Crowd threshold exceeded! Live headcount is {total_headcount} (Max Capacity Limit: {crowd_limit}).",
                    "model_name": "YOLOv8n",
                    "model_version": "1.0.0",
                    "evidence": [
                        {"name": "headcount", "value": str(total_headcount)},
                        {"name": "crowd_limit", "value": str(crowd_limit)}
                    ]
                })

    cap.release()
    return events