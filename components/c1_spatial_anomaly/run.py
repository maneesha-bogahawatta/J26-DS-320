import argparse
import cv2
from ultralytics import YOLO
from src.spatial_logic import SpatialAnomalyEngine

def main():
    parser = argparse.ArgumentParser(description="Run Component 1: Spatial Anomaly Detection")
    parser.add_argument("--source", type=str, default="0", help="Path to video file or webcam index")
    args = parser.parse_args()

    # Load shared detection model
    model = YOLO("yolov8n.pt")
    engine = SpatialAnomalyEngine(line_y=450, time_threshold=1.0)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    print(f"[C1] Running Spatial Anomaly Engine on source: {args.source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 with ByteTrack for class 0 (person)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

        tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()
            tracks = list(zip(ids, boxes))

        alerts = engine.process_tracks(tracks)
        for alert in alerts:
            print(f"[C1 ALERT] Tailgating detected! Breacher ID: {alert['breacher_id']} trailed Leader ID: {alert['leader_id']} in {alert['delta_t']}s")

        # Visualization
        cv2.line(frame, (0, 450), (frame.shape[1], 450), (255, 255, 0), 2)
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if track_id in engine.flagged_tailgaters else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("C1 - Spatial Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()