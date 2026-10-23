
from pathlib import Path
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--start", type=float, default=4.5)
parser.add_argument("--end", type=float, default=6.5)
parser.add_argument("--step", type=float, default=0.1)
args = parser.parse_args()

video = Path(args.video)
output = Path("outputs/precursor_frames") / video.stem
output.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise RuntimeError("Invalid video FPS")

time_s = args.start
count = 0

while time_s <= args.end + 1e-6:
    frame_number = round(time_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ok, frame = cap.read()

    if ok:
        filename = output / f"frame_{time_s:.2f}s.jpg"
        cv2.imwrite(str(filename), frame)
        count += 1

    time_s += args.step

cap.release()
print(f"Saved {count} frames to {output}")
