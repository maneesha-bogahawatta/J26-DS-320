"""
EXAMPLE — not loaded (the leading underscore skips it).

Copy to `c4_fall.py` (no underscore), replace `my_model` with your pipeline,
restart uvicorn, and the dashboard's Check video screen will use it when
VITE_DETECT_MODE=http.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from detect import frame_detector, video_detector
from schemas import EvidenceSignal, IncidentEvent, utc_now_iso


def _fall_event(source_id: str, seconds: float | None, box: dict | None, score: float) -> IncidentEvent:
    return IncidentEvent(
        event_id=f"component_4-{uuid4().hex[:8]}",
        component_id="component_4",
        incident_type="fall",
        camera_id=source_id,
        timestamp_detected=utc_now_iso(),
        priority="high",
        status="new",
        summary="Potential fall — review required",
        evidence=[
            EvidenceSignal(name="downward_motion", value=0.8, unit=None),
            EvidenceSignal(name="low_posture_duration", value=4.0, unit="seconds"),
        ],
        model_name="c4-fall-example",
        model_version="0.0.1",
        confidence=score,
        score_calibrated=False,
        bounding_box=box,  # {"x_min":..,"y_min":..,"x_max":..,"y_max":..}, all 0..1
        track_id="person-1",
        media_offset_seconds=seconds,
    )


@video_detector("c4_fall_example")
def detect_falls_in_video(video_path: Path, source_id: str) -> list[IncidentEvent]:
    # hits = my_model.run(str(video_path))
    hits: list[tuple[float, dict, float]] = []  # (seconds, normalised box, score)
    return [_fall_event(source_id, s, box, score) for s, box, score in hits]


@frame_detector("c4_fall_example")
def detect_falls_in_frame(jpeg: bytes, source_id: str, t: float) -> list[IncidentEvent]:
    # Keep per-source state yourself if your model needs several frames
    # (e.g. a dict of rolling buffers keyed by source_id).
    return []
