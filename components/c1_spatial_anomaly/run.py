import argparse
import cv2
from ultralytics import YOLO
from src.spatial_logic import SpatialAnomalyEngine

def main():
    parser = argparse.ArgumentParser(description="Component 1 - Spatial Anomaly Evaluator")
    parser.add_argument("--source", type=str, default="0", help="Video file path or camera index")
    parser.add_argument("--line-y", type=int, default=450, help="Y-coordinate for portal threshold")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")
    engine = SpatialAnomalyEngine(line_y=args.line_y, time_threshold=1.0)

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

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

        alerts = engine.process_tracks(tracks)
        for alert in alerts:
            print("[C1 ALERT]", alert["evidence"]["explanation"])

        # Display portal line
        cv2.line(frame, (0, args.line_y), (frame.shape[1], args.line_y), (255, 255, 0), 2)
        cv2.imshow("C1 Spatial Anomaly", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()