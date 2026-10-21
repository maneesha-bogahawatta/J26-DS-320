"""
Component 4 (Bogahawatta M.O.) - Fall + Abandoned-Object detector for the shared dashboard.

Registers one VIDEO detector (upload check) and one FRAME detector (live camera):

  fall   : MediaPipe pose -> 6 features -> 6 s window every 1 s -> CNN-LSTM ensemble (3 seeds)
  object : classical static-object monitor (background model + persistence + unattended timer)

Setup (see C4_INTEGRATION.md):
  pip install -r requirements_c4.txt
  weights in  ~/c4_weights/fall_cnn_lstm_gmdcsa_final_s{0,1,2}.pt   (or set C4_WEIGHTS_DIR)

Environment variables (all optional):
  C4_WEIGHTS_DIR   folder with the three .pt files            (default ~/c4_weights)
  C4_FALL_THR      fall probability threshold                  (default 0.40)
  C4_FALL_K        consecutive windows above threshold         (default 1)
  C4_T_UNATTENDED  seconds before an object counts as left     (default 10)
  C4_OBJECT_PRESET caviar (default, validated) or none            
  C4_FALL=0        switch the fall branch off
  C4_OBJECT=0      switch the object branch off
  C4_LIVE_HZ       pose rate for live frames                   (default 5)

If the weights or MediaPipe are missing, the fall branch is skipped (logged) and the
object branch still runs, so the API never goes down because of C4.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from detect import frame_detector, video_detector
from schemas import EvidenceSignal, IncidentEvent, RelatedTrack, utc_now_iso  # noqa: F401

log = logging.getLogger("c4")

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "c4_core"

FALL_ON = os.environ.get("C4_FALL", "1") != "0"
OBJECT_ON = os.environ.get("C4_OBJECT", "1") != "0"
THR = float(os.environ.get("C4_FALL_THR", "0.40"))
K = int(os.environ.get("C4_FALL_K", "1"))
T_UNATTENDED = float(os.environ.get("C4_T_UNATTENDED", "10"))
LIVE_HZ = float(os.environ.get("C4_LIVE_HZ", "5"))
# The object branch is only validated on the CAVIAR lobby camera, so that ROI mask is the default.
# Set C4_OBJECT_PRESET=none for any other camera view (it will need its own tuning).
CAVIAR_MASK = [(0.0, 0.72, 0.62, 1.0)] if os.environ.get("C4_OBJECT_PRESET", "caviar") == "caviar" else []
MODEL_VERSION = "0.1.0-pp1"

_shared: dict = {}          # pose + ensemble, loaded once
_live: dict[str, dict] = {}  # per live source: its own monitors


# ------------------------------------------------------------------ shared models
def _get_fall_parts():
    """Load MediaPipe + the 3 trained weights once. Returns (pose, models) or None."""
    if not FALL_ON:
        return None
    if "fall" in _shared:
        return _shared["fall"]
    try:
        from c4_core.fall_stream import MediaPipePose, load_ensemble
        weights = os.path.expanduser(os.environ.get("C4_WEIGHTS_DIR", "~/c4_weights"))
        models = load_ensemble(weights)
        pose = MediaPipePose(str(CORE / "pose_landmarker_lite.task"))
        _shared["fall"] = (pose, models)
        log.info("C4 fall branch ready (weights: %s)", weights)
    except Exception as exc:  # missing weights / mediapipe: keep the API alive
        log.warning("C4 fall branch disabled: %s", exc)
        _shared["fall"] = None
    return _shared["fall"]


def _make_monitors(pose_hz: float):
    from c4_core.fall_stream import FallMonitor
    from c4_core.object_stream import StaticObjectMonitor
    parts = _get_fall_parts()
    fall = FallMonitor(parts[0], parts[1], thr=THR, k=K, pose_hz=pose_hz) if parts else None
    obj = StaticObjectMonitor(t_alert=T_UNATTENDED, mask_rects=CAVIAR_MASK) if OBJECT_ON else None
    return fall, obj


# ------------------------------------------------------------------ event builders
def _norm_box(bbox, w: int, h: int):
    """[x0,y0,x1,y1] in pixels -> contract box (0..1, with area) or None."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = sorted((min(max(x0 / w, 0.0), 1.0), min(max(x1 / w, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0 / h, 0.0), 1.0), min(max(y1 / h, 0.0), 1.0)))
    if x1 - x0 < 0.01:
        x0, x1 = max(0.0, x0 - 0.01), min(1.0, x1 + 0.01)
    if y1 - y0 < 0.01:
        y0, y1 = max(0.0, y0 - 0.01), min(1.0, y1 + 0.01)
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}


def _fall_event(a: dict, source_id: str, w: int, h: int) -> IncidentEvent:
    p = float(a["p"])
    return IncidentEvent(
        event_id=f"component_4-{uuid4().hex[:8]}",
        component_id="component_4",
        incident_type="fall",
        camera_id=source_id,
        timestamp_detected=utc_now_iso(),
        priority="high",
        status="new",
        summary="Someone may have fallen - please check the clip",
        evidence=[
            EvidenceSignal(name="fall_probability", value=round(p, 3), unit=None,
                           description="Average of 3 trained models over the last 6 seconds of body movement."),
            EvidenceSignal(name="window_seconds", value=6.0, unit="seconds",
                           description="Length of the movement window the models looked at."),
            EvidenceSignal(name="pose_frames_in_window", value=int(a["n_pose"]), unit=None,
                           description="How many frames in the window had a clearly visible body."),
        ],
        model_name="c4-pose-cnn-lstm-ensemble",
        model_version=MODEL_VERSION,
        confidence=min(max(p, 0.0), 1.0),
        score_calibrated=False,
        bounding_box=_norm_box(a.get("bbox"), w, h),
        track_id="person-1",
        media_offset_seconds=max(0.0, float(a["t"]) - 3.0),
    )


def _object_event(a: dict, source_id: str, w: int, h: int) -> IncidentEvent:
    left_at = max(0.0, float(a["t"]) - float(a["unattended_s"]))
    secs = int(round(a["unattended_s"]))
    return IncidentEvent(
        event_id=f"component_4-{uuid4().hex[:8]}",
        component_id="component_4",
        incident_type="abandoned_object",
        camera_id=source_id,
        timestamp_detected=utc_now_iso(),
        priority="medium",
        status="new",
        summary=f"An item may have been left on its own for about {secs} seconds",
        evidence=[
            EvidenceSignal(name="unattended_seconds", value=round(float(a["unattended_s"]), 1), unit="seconds",
                           description="Time since anyone was last seen close to the item."),
            EvidenceSignal(name="placed_at_seconds", value=round(float(a["placed_t"]), 1), unit="seconds",
                           description="When the item stopped moving and became part of the scene."),
            EvidenceSignal(name="touched_fraction", value=round(float(a["touch_frac"]), 2), unit=None,
                           description="How often something moved right on top of the item (low = it just sat there)."),
        ],
        model_name="c4-static-object-baseline",
        model_version=MODEL_VERSION,
        confidence=None,           # rule-based: there is no probability, so none is claimed
        score_calibrated=False,
        bounding_box=_norm_box(a.get("bbox"), w, h),
        track_id=f"object-{a['object_id']}",
        media_offset_seconds=left_at,
    )


def _overlaps(a, b, frac: float = 0.5) -> bool:
    """True if the smaller of two contract boxes is mostly inside the other (same item reported twice)."""
    ix = max(0.0, min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"]))
    iy = max(0.0, min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"]))
    area = lambda r: (r["x_max"] - r["x_min"]) * (r["y_max"] - r["y_min"])
    return ix * iy >= frac * min(area(a), area(b))


def _keep_new_objects(events: list[IncidentEvent], seen: list[dict]) -> list[IncidentEvent]:
    """Drop an abandoned-object event whose box sits on one already reported (image noise can split one item in two)."""
    kept = []
    for e in events:
        box = e.bounding_box.model_dump() if e.bounding_box else None
        if e.incident_type == "abandoned_object" and box is not None:
            if any(_overlaps(box, b) for b in seen):
                continue
            seen.append(box)
        kept.append(e)
    return kept


# ------------------------------------------------------------------ uploaded video
@video_detector("c4_fall_and_object")
def detect_video(video_path: Path, source_id: str) -> list[IncidentEvent]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("C4 could not open %s", video_path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fall, obj = _make_monitors(pose_hz=15.0)
    events: list[IncidentEvent] = []
    seen: list[dict] = []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            h, w = frame.shape[:2]
            if obj is not None:
                events += _keep_new_objects([_object_event(a, source_id, w, h) for a in obj.feed(frame, t)], seen)
            if fall is not None:
                small = frame if w <= 640 else cv2.resize(frame, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
                for a in fall.feed(small, t):
                    sh, sw = small.shape[:2]
                    events.append(_fall_event(a, source_id, sw, sh))
    finally:
        cap.release()
    return events


# ------------------------------------------------------------------ live camera frames
@frame_detector("c4_fall_and_object")
def detect_frame(jpeg: bytes, source_id: str, t: float) -> list[IncidentEvent]:
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return []
    st = _live.get(source_id)
    if st is None or t < st["last_t"] - 1.0:      # first frame, or the camera was restarted
        fall, obj = _make_monitors(pose_hz=LIVE_HZ)
        st = _live[source_id] = {"fall": fall, "obj": obj, "last_t": t, "seen": []}
    st["last_t"] = t
    h, w = frame.shape[:2]
    events: list[IncidentEvent] = []
    if st["obj"] is not None:
        events += _keep_new_objects([_object_event(a, source_id, w, h) for a in st["obj"].feed(frame, t)], st["seen"])
    if st["fall"] is not None:
        events += [_fall_event(a, source_id, w, h) for a in st["fall"].feed(frame, t)]
    return events
