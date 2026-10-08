import argparse
from collections import deque
import cv2
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

def load_model(weights_path, device):
    print(f"Loading weights from: {weights_path}")
    model = r2plus1d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2)
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def run_inference(source, weights_path, threshold=0.70):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    model = load_model(weights_path, device)

    # Convert numeric camera string (e.g. '0') to integer
    video_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source '{source}'")
        return

    frame_buffer = deque(maxlen=16)
    frame_count = 0
    stride = 4
    current_status = "Scanning..."
    status_color = (0, 255, 0)

    print("Video stream opened. Press 'q' on the preview window to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize for the 3D-CNN buffer
        resized = cv2.resize(frame, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        frame_buffer.append(tensor_frame)

        # Run inference when buffer has 16 frames
        if len(frame_buffer) == 16 and (frame_count % stride == 0):
            input_tensor = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                conf, pred_class = torch.max(probs, dim=0)

            conf_val = conf.item()
            if pred_class.item() == 1 and conf_val >= threshold:
                current_status = f"VIOLENCE / AGGRESSION ({conf_val:.1%})"
                status_color = (0, 0, 255)
            else:
                current_status = f"Normal Activity ({probs[0].item():.1%})"
                status_color = (0, 255, 0)

        # Overlay detection status on preview window
        cv2.putText(frame, f"C3 Status: {current_status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.imshow("Component 03 - Aggression Detection Baseline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference session closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C3 Aggression Detection Baseline")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index (0)")
    parser.add_argument("--weights", type=str, default="weights/best_aggression_model.pt", help="Path to weights")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold")
    args = parser.parse_args()

    run_inference(args.source, args.weights, args.threshold)
