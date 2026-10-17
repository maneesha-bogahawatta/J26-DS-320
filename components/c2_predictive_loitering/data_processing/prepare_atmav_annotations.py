"""Prepare and validate ATMA-V temporal annotations."""

import os
from pathlib import Path

import pandas as pd


COMPONENT_ROOT = Path(__file__).resolve().parents[1]

ATMA_ROOT = Path(
    os.environ.get(
        "ATMA_V_ROOT",
        COMPONENT_ROOT / "ATMA-V Dataset",
    )
)

WORK_ROOT = Path(
    os.environ.get(
        "C2_WORK_ROOT",
        COMPONENT_ROOT,
    )
)

LABEL_FILE = ATMA_ROOT / "labels" / "labels.txt"
VIDEO_ROOT = ATMA_ROOT / "videos"

OUTPUT_DIRECTORY = WORK_ROOT / "data" / "processed"
MANIFEST_OUTPUT = OUTPUT_DIRECTORY / "atma_video_manifest.csv"
INTERVAL_OUTPUT = OUTPUT_DIRECTORY / "atma_temporal_intervals.csv"


def parse_annotation_line(line):
    """Parse one ATMA-V temporal-annotation line."""
    parts = line.strip().split()

    if len(parts) < 5:
        raise ValueError(
            f"Invalid annotation line: {line.strip()}"
        )

    video_name = parts[0]
    total_frames = int(parts[1])
    fps = float(parts[2])
    interval_values = [int(value) for value in parts[3:]]

    annotation_warning = ""

    # The source labels contain an incomplete final interval
    # for 93.mp4. Complete pairs are preserved, while the
    # unmatched value is recorded for manual investigation.
    if len(interval_values) % 2 != 0:
        orphan_frame = interval_values.pop()

        annotation_warning = (
            "Incomplete final interval in source annotation; "
            f"unpaired frame value: {orphan_frame}"
        )

    intervals = []

    for index in range(0, len(interval_values), 2):
        start_frame = interval_values[index]
        end_frame = interval_values[index + 1]

        if start_frame == -1 and end_frame == -1:
            continue

        if start_frame < 0 or end_frame < 0:
            raise ValueError(
                f"Invalid negative range for {video_name}: "
                f"{start_frame}-{end_frame}"
            )

        if start_frame > end_frame:
            raise ValueError(
                f"Start frame exceeds end frame for "
                f"{video_name}: {start_frame}-{end_frame}"
            )

        if end_frame > total_frames:
            previous_warning = (
                f"{annotation_warning}; "
                if annotation_warning
                else ""
            )

            annotation_warning = (
                previous_warning
                + f"interval {start_frame}-{end_frame} exceeds "
                + f"declared frame count {total_frames}"
            )

        intervals.append(
            (
                start_frame,
                end_frame,
            )
        )

    return {
        "video_name": video_name,
        "total_frames": total_frames,
        "fps": fps,
        "intervals": intervals,
        "annotation_warning": annotation_warning,
    }


def main():
    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"ATMA-V label file was not found:\n{LABEL_FILE}"
        )

    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(
            f"ATMA-V video folder was not found:\n{VIDEO_ROOT}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []
    interval_rows = []

    with LABEL_FILE.open(
        "r",
        encoding="utf-8",
    ) as label_stream:
        for line_number, line in enumerate(
            label_stream,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = parse_annotation_line(line)
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: {error}"
                ) from error

            video_name = record["video_name"]
            total_frames = record["total_frames"]
            fps = record["fps"]
            intervals = record["intervals"]
            annotation_warning = record[
                "annotation_warning"
            ]

            if fps <= 0:
                raise ValueError(
                    f"Line {line_number}: invalid FPS "
                    f"for {video_name}: {fps}"
                )

            if annotation_warning:
                print(
                    f"WARNING: {video_name}: "
                    f"{annotation_warning}"
                )

            video_path = VIDEO_ROOT / video_name
            video_exists = video_path.exists()

            source_label = (
                "contains_anomaly"
                if intervals
                else "normal_video"
            )

            anomalous_frames = sum(
                end_frame - start_frame + 1
                for start_frame, end_frame in intervals
            )

            manifest_rows.append(
                {
                    "dataset_source": "ATMA-V",
                    "video_id": Path(video_name).stem,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "video_exists": video_exists,
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration_seconds": round(
                        total_frames / fps,
                        4,
                    ),
                    "source_label": source_label,
                    "anomaly_interval_count": len(
                        intervals
                    ),
                    "anomalous_frames": anomalous_frames,
                    "loitering_label": "",
                    "label_status":
                        "requires_manual_review",
                    "annotation_warning":
                        annotation_warning,
                }
            )

            if intervals:
                for interval_number, (
                    start_frame,
                    end_frame,
                ) in enumerate(
                    intervals,
                    start=1,
                ):
                    interval_rows.append(
                        {
                            "dataset_source": "ATMA-V",
                            "video_id":
                                Path(video_name).stem,
                            "video_name": video_name,
                            "interval_number":
                                interval_number,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "start_seconds": round(
                                start_frame / fps,
                                4,
                            ),
                            "end_seconds": round(
                                end_frame / fps,
                                4,
                            ),
                            "source_label":
                                "general_anomaly",
                            "loitering_label": "",
                            "label_status":
                                "requires_manual_review",
                            "annotation_warning":
                                annotation_warning,
                        }
                    )
            else:
                interval_rows.append(
                    {
                        "dataset_source": "ATMA-V",
                        "video_id":
                            Path(video_name).stem,
                        "video_name": video_name,
                        "interval_number": 0,
                        "start_frame": 1,
                        "end_frame": total_frames,
                        "start_seconds": 0.0,
                        "end_seconds": round(
                            total_frames / fps,
                            4,
                        ),
                        "source_label": "normal",
                        "loitering_label": 0,
                        "label_status":
                            "source_verified_normal",
                        "annotation_warning":
                            annotation_warning,
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    temporal_intervals = pd.DataFrame(interval_rows)

    manifest.to_csv(
        MANIFEST_OUTPUT,
        index=False,
    )

    temporal_intervals.to_csv(
        INTERVAL_OUTPUT,
        index=False,
    )

    missing_videos = manifest[
        manifest["video_exists"] == False  # noqa: E712
    ]

    warning_count = int(
        manifest["annotation_warning"]
        .fillna("")
        .str.strip()
        .ne("")
        .sum()
    )

    print()
    print("ATMA-V preparation completed")
    print("----------------------------")
    print(f"Annotated videos: {len(manifest)}")

    normal_count = (
        manifest["source_label"] == "normal_video"
    ).sum()

    anomaly_count = (
        manifest["source_label"]
        == "contains_anomaly"
    ).sum()

    print(f"Normal videos: {normal_count}")
    print(
        "Videos containing anomalies: "
        f"{anomaly_count}"
    )
    print(
        f"Temporal rows: {len(temporal_intervals)}"
    )
    print(
        f"Missing video files: {len(missing_videos)}"
    )
    print(
        f"Annotation warnings: {warning_count}"
    )
    print(f"\nManifest:\n{MANIFEST_OUTPUT}")
    print(
        f"\nTemporal intervals:\n{INTERVAL_OUTPUT}"
    )

    if not missing_videos.empty:
        print("\nMissing videos:")

        print(
            missing_videos[
                [
                    "video_name",
                    "video_path",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()