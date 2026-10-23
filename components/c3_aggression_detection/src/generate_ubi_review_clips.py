
import csv
import json
import subprocess
from pathlib import Path

MANIFEST = Path("data/manifests/ubi_review.csv")
OUTPUT = Path("outputs/review_clips")

PRE_SECONDS = 5
POST_SECONDS = 3

OUTPUT.mkdir(parents=True, exist_ok=True)

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

generated = 0
skipped = 0

for row in rows:
    video = Path(row["video"])
    intervals = json.loads(row["intervals"] or "[]")

    if not video.is_file() or not intervals:
        skipped += 1
        continue

    # Generate a clip around the first annotated fight onset.
    onset = float(intervals[0][0])

    start = max(0.0, onset - PRE_SECONDS)
    duration = onset + POST_SECONDS - start

    output = OUTPUT / f"{video.stem}_review.mp4"

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-i", str(video),
        "-t", str(duration),
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-y", str(output)
    ]

    try:
        subprocess.run(command, check=True)
        generated += 1
        print(f"[OK] {video.name} | onset={onset:.2f}s")
    except subprocess.CalledProcessError:
        skipped += 1
        print(f"[ERROR] {video.name}")

print(f"\nGenerated: {generated}")
print(f"Skipped/failed: {skipped}")

