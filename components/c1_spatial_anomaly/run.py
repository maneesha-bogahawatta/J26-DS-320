import argparse
import glob
import json
import os
import cv2
import numpy as np
from ultralytics import YOLO
from src.spatial_logic import SpatialAnomalyEngine

class FrameReader:
    """Helper class to abstract reading from both VideoCapture and Image Folders."""
    def __init__(self, source):
        self.is_image_folder = os.path.isdir(source)
        if self.is_image_folder:
            # Find and sort all image files inside directory
            patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            self.image_files = []
            for pattern in patterns:
                self.image_files.extend(glob.glob(os.path.join(source, pattern)))
            self.image_files = sorted(self.image_files)
            self.idx = 0
            self.cap = None
            if not self.image_files:
                raise ValueError(f"No image files found in folder: {source}")
            # Determine dimensions from first image
            sample = cv2.imread(self.image_files[0])
            self.height, self.width = sample.shape[:2]
        else:
            if source.isdigit():
                self.cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {source}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        if self.is_image_folder:
            if self.idx >= len(self.image_files):
                return False, None
            frame = cv2.imread(self.image_files[self.idx])
            self.idx += 1
            return (frame is not None), frame
        else:
            return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description="Run Component 1: Spatial Anomaly Detection")
    parser.add_argument("--source", type=str, required=True, help="Video file path, camera index, or folder of images")
    parser.add_argument("--config", type=str, default="components/c1_spatial_anomaly/configs/doorway_config.json")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = json.load(f)

    portal_line = tuple(tuple(pt) for pt in cfg["portal_threshold_line"])
    restricted_poly = cfg.get("restricted_zone_polygon", None)

    # Initialize Engine
    engine = SpatialAnomalyEngine(
        portal_line=portal_line,
        restricted_polygon=restricted_poly,
        tailgating_time_threshold=cfg.get("tailgating_time_threshold_sec", 1.0),
        crowd_density_limit=cfg.get("crowd_density_limit", 4)
    )

    model = YOLO("yolov8n.pt")

    try:
        reader = FrameReader(args.source)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"[C1] Successfully opened stream from '{args.source}' ({reader.width}x{reader.height})")

    while True:
        ret, frame = reader.read()
        if not ret:
            print("[C1] Finished reading frames from source.")
            break

        # Run ByteTrack for human class 0
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            tracks = list(zip(ids, boxes))

        # Process spatial alerts
        alerts = engine.process_frame(tracks, camera_id=cfg["camera_id"])
        for alert in alerts:
            print(f"[C1 ALERT {alert['event_type']}] {alert['evidence']['explanation']}")

        # 1. Draw Doorway Portal Line (Cyan)
        cv2.line(frame, portal_line[0], portal_line[1], (255, 255, 0), 2)
        cv2.putText(frame, "PORTAL LINE", (portal_line[0][0], portal_line[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 2. Draw Restricted Zone Polygon (Orange)
        if restricted_poly:
            pts = np.array(restricted_poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(frame, "RESTRICTED ZONE", (restricted_poly[0][0], restricted_poly[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 3. Draw Pedestrian Bounding Boxes and IDs
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            if track_id in engine.flagged_tailgaters or track_id in engine.active_intrusions:
                color = (0, 0, 255)
                label = f"ID:{track_id} [FLAGGED BREACH]"
            elif track_id == engine.last_leader_id:
                color = (0, 255, 0)
                label = f"ID:{track_id} [LEADER]"
            else:
                color = (255, 255, 255)
                label = f"ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("C1 - Spatial Anomaly Evaluator", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    reader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()