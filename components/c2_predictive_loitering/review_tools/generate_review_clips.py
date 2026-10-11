from pathlib import Path
import os

import cv2
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
COMPONENT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    os.environ.get(
        "C2_DATASET_ROOT",
        COMPONENT_ROOT / "CAVIAR_Dataset" / "INRIA",
    )
)

WORK_ROOT = Path(
    os.environ.get(
        "C2_WORK_ROOT",
        COMPONENT_ROOT,
    )
)

TRAJECTORY_FILE = (
    WORK_ROOT
    / "data"
    / "ground_truth"
    / "people_ground_truth_trajectories.csv"
)

SUMMARY_FILE = (
    WORK_ROOT
    / "data"
    / "processed"
    / "trajectory_summary.csv"
)

VIDEO_ROOT = DATASET_ROOT

OUTPUT_DIR = (
    WORK_ROOT
    / "outputs"
    / "trajectory_review_clips"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
DEFAULT_FPS = 25.0
CONTEXT_SECONDS = 1.0
TRAIL_LENGTH = 75
BOX_COLOUR = (0, 255, 0)
TRAIL_COLOUR = (0, 255, 255)


def find_video(video_id):
    """
    Use the original MPG because XML coordinates correspond to
    the original 384 x 288 CAVIAR video.
    """
    scenario_folder = VIDEO_ROOT / video_id

    preferred_video = scenario_folder / f"{video_id}.mpg"

    if preferred_video.exists():
        return preferred_video

    mpg_files = list(scenario_folder.glob("*.mpg"))

    if mpg_files:
        return mpg_files[0]

    return None


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def draw_information(frame, video_id, track_id, summary_row):
    overlay_lines = [
        f"Video: {video_id}",
        f"Target person: Track {track_id}",
        f"Dwell time: {summary_row['dwell_time']:.2f} seconds",
        f"Mean speed: {summary_row['mean_speed']:.4f}",
        f"Pause ratio: {summary_row['pause_ratio']:.3f}",
        (
            "Path efficiency: "
            f"{summary_row['final_path_efficiency']:.3f}"
        ),
        f"Situations: {safe_text(summary_row['situations'])}",
        f"Roles: {safe_text(summary_row['roles'])}",
    ]

    panel_height = 22 * len(overlay_lines) + 12

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (frame.shape[1], panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    y_position = 22

    for line in overlay_lines:
        cv2.putText(
            frame,
            line,
            (8, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y_position += 22


def generate_track_clip(
    video_path,
    video_id,
    track_id,
    track_data,
    summary_row,
):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return False

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = DEFAULT_FPS

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    context_frames = int(fps * CONTEXT_SECONDS)

    first_annotated_frame = int(track_data["frame_number"].min())
    last_annotated_frame = int(track_data["frame_number"].max())

    clip_start = max(0, first_annotated_frame - context_frames)
    clip_end = last_annotated_frame + context_frames

    output_path = (
        OUTPUT_DIR
        / f"{video_id}_track_{track_id}_review.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print(f"ERROR: Could not create {output_path}")
        capture.release()
        return False

    # Fast lookup: frame number -> annotation row
    annotation_lookup = {
        int(row.frame_number): row
        for row in track_data.itertuples()
    }

    # Centre points for drawing the trajectory trail
    centres = {
        int(row.frame_number): (
            int((row.x1 + row.x2) / 2),
            int((row.y1 + row.y2) / 2),
        )
        for row in track_data.itertuples()
    }

    capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

    current_frame = clip_start

    while current_frame <= clip_end:
        success, frame = capture.read()

        if not success:
            break

        # Draw recent trajectory trail
        trail_start = max(
            first_annotated_frame,
            current_frame - TRAIL_LENGTH,
        )

        trail_points = [
            centres[frame_number]
            for frame_number in range(
                trail_start,
                current_frame + 1,
            )
            if frame_number in centres
        ]

        for point_index in range(1, len(trail_points)):
            cv2.line(
                frame,
                trail_points[point_index - 1],
                trail_points[point_index],
                TRAIL_COLOUR,
                2,
                cv2.LINE_AA,
            )

        # Draw target-person bounding box
        if current_frame in annotation_lookup:
            annotation = annotation_lookup[current_frame]

            x1 = max(0, int(annotation.x1))
            y1 = max(0, int(annotation.y1))
            x2 = min(width - 1, int(annotation.x2))
            y2 = min(height - 1, int(annotation.y2))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BOX_COLOUR,
                2,
            )

            label = f"TARGET ID {track_id}"

            label_y = max(18, y1 - 8)

            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                BOX_COLOUR,
                2,
                cv2.LINE_AA,
            )

            centre = centres[current_frame]

            cv2.circle(
                frame,
                centre,
                4,
                (0, 0, 255),
                -1,
            )

        draw_information(
            frame,
            video_id,
            track_id,
            summary_row,
        )

        cv2.putText(
            frame,
            f"Frame: {current_frame}",
            (8, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        current_frame += 1

    writer.release()
    capture.release()

    print(f"Created: {output_path.name}")
    return True


def main():
    print("Reading trajectories:")
    print(TRAJECTORY_FILE)

    trajectories = pd.read_csv(TRAJECTORY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    required_columns = {
        "video_id",
        "track_id",
        "frame_number",
        "x1",
        "y1",
        "x2",
        "y2",
    }

    missing_columns = required_columns - set(trajectories.columns)

    if missing_columns:
        raise ValueError(
            f"Missing trajectory columns: {missing_columns}"
        )

    created_count = 0
    failed_count = 0

    grouped_tracks = trajectories.groupby(
        ["video_id", "track_id"],
        sort=True,
    )

    for (video_id, track_id), track_data in grouped_tracks:
        video_path = find_video(video_id)

        if video_path is None:
            print(f"WARNING: Video not found for {video_id}")
            failed_count += 1
            continue

        matching_summary = summary[
            (summary["video_id"] == video_id)
            & (summary["track_id"] == track_id)
        ]

        if matching_summary.empty:
            print(
                f"WARNING: Summary missing for "
                f"{video_id}, track {track_id}"
            )
            failed_count += 1
            continue

        summary_row = matching_summary.iloc[0]

        print(
            f"Processing {video_id}, track {track_id} "
            f"using {video_path.name}"
        )

        completed = generate_track_clip(
            video_path=video_path,
            video_id=video_id,
            track_id=track_id,
            track_data=track_data.sort_values("frame_number"),
            summary_row=summary_row,
        )

        if completed:
            created_count += 1
        else:
            failed_count += 1

    print()
    print("Review-clip generation completed")
    print("--------------------------------")
    print(f"Created clips: {created_count}")
    print(f"Failed clips: {failed_count}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
