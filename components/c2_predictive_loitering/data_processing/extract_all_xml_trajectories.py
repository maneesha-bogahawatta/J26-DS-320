import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


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

OUTPUT_CSV = (
    WORK_ROOT
    / "data"
    / "ground_truth"
    / "all_ground_truth_trajectories.csv"
)

SUMMARY_CSV = (
    WORK_ROOT
    / "data"
    / "ground_truth"
    / "scenario_summary.csv"
)

FPS = 25.0
VIDEO_WIDTH = 384
VIDEO_HEIGHT = 288

SELECTED_SCENARIOS = [
    "Browse1",
    "Browse2",
    "Browse3",
    "Browse4",
    "Browse_WhileWaiting1",
    "Browse_WhileWaiting2",
    "Meet_Crowd",
    "Meet_Split_3rdGuy",
    "Meet_WalkSplit",
    "LeftBag",
    "LeftBag_AtChair",
    "LeftBag_BehindChair",
    "LeftBag_PickedUp",
    "LeftBox",
]


def get_text(parent, path, default="unknown"):
    element = parent.find(path)

    if element is None or element.text is None:
        return default

    value = element.text.strip()
    return value if value else default


OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

all_rows = []
scenario_summaries = []

for video_id in SELECTED_SCENARIOS:
    scenario_folder = DATASET_ROOT / video_id
    xml_files = list(scenario_folder.glob("*.xml"))

    if not scenario_folder.exists():
        print(f"Skipped missing folder: {video_id}")
        continue

    if len(xml_files) == 0:
        print(f"Skipped: no XML found for {video_id}")
        continue

    if len(xml_files) > 1:
        print(
            f"Skipped: multiple XML files found for {video_id}: "
            f"{[file.name for file in xml_files]}"
        )
        continue

    xml_path = xml_files[0]

    print(f"Processing {video_id}: {xml_path.name}")