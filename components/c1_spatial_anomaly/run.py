import argparse
import json
import cv2
import numpy as np
from ultralytics import YOLO
from src.spatial_logic import SpatialAnomalyEngine

def main():
    parser = argparse.ArgumentParser(description="Run Component 1: Spatial Anomaly Detection")
    parser.add_argument("--source", type=str, default="0", help="Video file or camera stream")
    parser.add_argument("--config", type=str, default="components/c1_spatial_anomaly/configs/doorway_config.json")
    args = parser.parse_args()

    # Load zone configuration
    with open(args.config, "r") as f:
        cfg = json.load(f)

    portal_line = tuple(tuple(pt) for pt in cfg["portal_threshold_line"])
    restricted_poly = cfg["restricted_zone_polygon"]

    # Initialize Engine
    engine = SpatialAnomalyEngine(
        portal_line=portal_line,
        restricted_polygon=restricted_poly,
        tailgating_time_threshold=cfg["tailgating_time_threshold_sec"],
        crowd_density_limit=cfg["crowd_density_limit"]
    )

    model = YOLO("yolov8n.pt")
    if args.source.isdigit():
    # cv2.CAP_DSHOW forces Windows DirectShow, which fixes camera indexing issues
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: '{args.source}'. Check if file exists!")
        return

    # Check if first frame can be read
    ret, frame = cap.read()
    if not ret:
        print(f"[ERROR] Opened '{args.source}' but failed to read the first frame. Invalid video format/codec or empty file.")
        cap.release()
        return

    # Reset position back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"[C1] Successfully opened video '{args.source}' ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
    

    print(f"[C1] Initialized Spatial Engine on {args.source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            tracks = list(zip(ids, boxes))

        # Process spatial logic
        alerts = engine.process_frame(tracks, camera_id=cfg["camera_id"])
        for alert in alerts:
            print(f"[C1 ALERT {alert['event_type']}] {alert['evidence']['explanation']}")

        # 1. Draw Doorway Portal Line (Cyan)
        cv2.line(frame, portal_line[0], portal_line[1], (255, 255, 0), 2)
        cv2.putText(frame, "PORTAL LINE", (portal_line[0][0], portal_line[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # 2. Draw Restricted Zone Polygon (Yellow/Orange dashed or solid)
        pts = np.array(restricted_poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.putText(frame, "RESTRICTED ZONE", (restricted_poly[0][0], restricted_poly[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 3. Render Pedestrian Bounding Boxes with Roles
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            if track_id in engine.flagged_tailgaters or track_id in engine.active_intrusions:
                color = (0, 0, 255)  # Red for violators
                label = f"ID:{track_id} [FLAGGED BREACH]"
            elif track_id == engine.last_leader_id:
                color = (0, 255, 0)  # Green for leader
                label = f"ID:{track_id} [LEADER]"
            else:
                color = (255, 255, 255)  # White for neutral
                label = f"ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("C1 - Spatial Anomaly Evaluator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()