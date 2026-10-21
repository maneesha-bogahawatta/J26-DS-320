"""Export short CCTV control windows for blinded human review.

Run from the Component 03 project root:
    python -m src.export_control_review_clips --status proposal
    python -m src.export_control_review_clips --status no_proposal --limit 4

Reads existing CSV; never modifies detector outputs or verified annotations.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2

DEFAULT_CSV = Path("outputs/v3_evaluation/no_fight_control_results.csv")
DEFAULT_OUT = Path("outputs/v3_evaluation/control_review_clips")
REVIEW_CSV = Path("outputs/v3_evaluation/control_review_labels.csv")

FIELDS = [
    "video", "clip_path", "window_start_s", "window_end_s", "v3_status",
    "v3_proposals", "v3_max_score", "v3_quality", "review_label",
    "observed_cue_start_s", "first_contact_s", "person_pair_visible",
    "camera_motion", "reviewer", "notes",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value, name):
    try:
        return float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc


def export_video_window(source, target, start_s, end_s, max_width=960):
    """Export original video time span with preserved (source) FPS."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            raise ValueError(f"Unexpected FPS {fps} for {source}")
        if end_s <= start_s or start_s < 0:
            raise ValueError(f"Invalid time window {start_s}-{end_s}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first = max(0, int(round(start_s * fps)))
        stop = int(round(end_s * fps))
        if count > 0:
            stop = min(stop, count)
        if stop <= first:
            raise ValueError(f"Empty clip range {start_s}-{end_s} for {source}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        exported = 0
        for _ in range(first, stop):
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if max_width > 0 and width > max_width:
                scaled = max(2, int(round(height * max_width / width)))
                width = max_width - (max_width % 2)
                height = scaled + (scaled % 2)
                frame = cv2.resize(frame, (width, height))
            elif width % 2 or height % 2:
                width -= width % 2
                height -= height % 2
                frame = cv2.resize(frame, (width, height))
            if writer is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not write MP4: {target}")
            writer.write(frame)
            exported += 1
        if exported == 0:
            raise RuntimeError(f"No frames decoded from {source}")
        return exported
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def make_review_row(source_row, output_path):
    return {
        "video": source_row.get("video", ""),
        "clip_path": str(output_path),
        "window_start_s": source_row.get("control_window_start_s", ""),
        "window_end_s": source_row.get("control_window_end_s", ""),
        "v3_status": source_row.get("v3_status", ""),
        "v3_proposals": source_row.get("v3_proposals", "[]"),
        "v3_max_score": source_row.get("v3_max_score", ""),
        "v3_quality": source_row.get("v3_quality", ""),
        "review_label": "unreviewed",  # non_aggressive / threatening_cue / physical_fight / uncertain
        "observed_cue_start_s": "",   # original-video seconds
        "first_contact_s": "",       # original-video seconds
        "person_pair_visible": "",    # yes / no / uncertain
        "camera_motion": "",         # stable / moving / uncertain
        "reviewer": "",
        "notes": "",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--review-csv", type=Path, default=REVIEW_CSV)
    parser.add_argument("--status", choices=("proposal", "no_proposal", "all"), default="proposal")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--max-width", type=int, default=960)
    args = parser.parse_args(argv)
    if args.limit < 0 or args.max_width < 0:
        parser.error("--limit and --max-width cannot be negative")
    if not args.csv.is_file():
        parser.error(f"Input CSV not found: {args.csv}")

    selected = [r for r in read_csv(args.csv)
                if args.status == "all" or r.get("v3_status") == args.status]
    if args.limit:
        selected = selected[: args.limit]

    # Preserve existing manual review work when exporting a second batch.
    old_rows = read_csv(args.review_csv) if args.review_csv.is_file() else []
    reviews = {r["clip_path"]: r for r in old_rows if r.get("clip_path")}
    success = 0
    for i, row in enumerate(selected, 1):
        source = Path(row.get("video", ""))
        start = safe_float(row.get("control_window_start_s"), "control_window_start_s")
        end = safe_float(row.get("control_window_end_s"), "control_window_end_s")
        target = args.outdir / f"{source.stem}_{start:.3f}-{end:.3f}.mp4"
        try:
            if not source.is_file():
                raise FileNotFoundError(str(source))
            frames = export_video_window(source, target, start, end, args.max_width)
            record = make_review_row(row, target)
            reviews.setdefault(str(target), record)
            success += 1
            print(f"[{i}/{len(selected)}] {source.name}: {frames} frames -> {target.name}")
        except Exception as exc:
            if target.exists() and target.stat().st_size == 0:
                target.unlink()
            print(f"[{i}/{len(selected)}] ERROR {source.name}: {exc}")
    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviews.values())
    print(f"Exported: {success}/{len(selected)} | Clips folder: {args.outdir}")
    print(f"Review labels CSV: {args.review_csv}")
    print("Review labels begin UNREVIEWED; no automatic false-positive labels were assigned.")
    return success == len(selected)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
