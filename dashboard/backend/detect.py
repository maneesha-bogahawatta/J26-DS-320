"""
Detection endpoints — where each component's trained model plugs in.

The dashboard's "Check video" screen (with VITE_DETECT_MODE=http) sends:

    POST /api/v1/detect/video?source_id=upload-lobby&filename=lobby.mp4
         body: the raw video file
    POST /api/v1/detect/frame?source_id=live-camera&t=12.50
         body: one JPEG from the live camera (about one per second)

Both reply with the same shape:

    { "events": [IncidentEvent, ...], "detectors_run": ["c4_fall", ...] }

Every event returned is also stored, so it appears in Alerts and the shift log.

---------------------------------------------------------------------------
HOW TO PLUG IN YOUR MODEL (Members 1-4)
---------------------------------------------------------------------------

Write a function that takes the video path (or the JPEG bytes) and returns a
list of IncidentEvent, then register it with the decorator. That is all.

    # dashboard/backend/detectors/my_detector.py (optional plugin)
    from detect import video_detector
    from schemas import IncidentEvent

    @video_detector("c4_fall")
    def detect_falls(video_path, source_id):
        events = []
        for hit in my_model.run(video_path):          # your pipeline
            events.append(IncidentEvent(
                event_id=f"component_4-{uuid4().hex[:8]}",
                component_id="component_4",
                incident_type="fall",
                camera_id=source_id,
                timestamp_detected=utc_now_iso(),
                priority="high",
                status="new",
                summary="Potential fall — review required",
                evidence=[...],
                model_name="c4-fall-stgcn", model_version="0.3.0",
                bounding_box=hit.box_normalised,
                media_offset_seconds=hit.seconds,      # so the dashboard jumps there
                ...
            ))
        return events

Any .py file in dashboard/backend/detectors/ is imported at start-up when that
optional folder exists. The C3 setup uses c3_api.py/c3_bridge.py and does not
need a detector plugin here. Adding a team plugin needs no change to main.py.

Rules: keep the event contract (docs/EVENT_CONTRACT.md); never put a name or
identity in track_id; send media_offset_seconds for uploads so the dashboard
can jump to the moment. If your model is slow, that is fine for uploads; for
live frames, aim to answer in well under a second or skip frames.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import re
import time
import uuid
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from schemas import IncidentEvent
from store import DuplicateEventError, EventStore

log = logging.getLogger("detect")

HERE = Path(__file__).parent
UPLOAD_DIR = HERE / "uploads"
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_FRAME_BYTES = 8 * 1024 * 1024

VideoDetector = Callable[[Path, str], list[IncidentEvent]]
FrameDetector = Callable[[bytes, str, float], list[IncidentEvent]]

VIDEO_DETECTORS: dict[str, VideoDetector] = {}
FRAME_DETECTORS: dict[str, FrameDetector] = {}


def video_detector(name: str) -> Callable[[VideoDetector], VideoDetector]:
    """Register a function(video_path, source_id) -> list[IncidentEvent]."""

    def register(fn: VideoDetector) -> VideoDetector:
        VIDEO_DETECTORS[name] = fn
        return fn

    return register


def frame_detector(name: str) -> Callable[[FrameDetector], FrameDetector]:
    """Register a function(jpeg_bytes, source_id, seconds_since_start) -> list[IncidentEvent]."""

    def register(fn: FrameDetector) -> FrameDetector:
        FRAME_DETECTORS[name] = fn
        return fn

    return register


def load_detectors() -> None:
    """Load optional team detector plugins; an absent directory is supported."""
    package_dir = HERE / "detectors"
    if not package_dir.is_dir():
        return
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"detectors.{info.name}")
            log.info("loaded detector module %s", info.name)
        except Exception:  # one broken detector must not take the API down
            log.exception("could not load detector module %s", info.name)


class DetectResponse(BaseModel):
    events: list[IncidentEvent]
    detectors_run: list[str]
    note: str | None = None


def _store_all(store: EventStore, events: list[IncidentEvent]) -> list[IncidentEvent]:
    stored = []
    for event in events:
        try:
            stored.append(store.add_event(event))
        except DuplicateEventError:
            stored.append(event)
    return stored


def _run(name: str, fn: Callable[..., list[IncidentEvent]], *args) -> list[IncidentEvent]:
    started = time.perf_counter()
    try:
        result = fn(*args) or []
    except Exception:
        log.exception("detector %s failed", name)
        return []
    log.info("detector %s returned %d event(s) in %.2fs", name, len(result), time.perf_counter() - started)
    return [e if isinstance(e, IncidentEvent) else IncidentEvent.model_validate(e) for e in result]


def build_router(store: EventStore) -> APIRouter:
    router = APIRouter(prefix="/api/v1/detect", tags=["detect"])

    @router.get("/detectors")
    def list_detectors() -> dict[str, list[str]]:
        """Which models are plugged in right now."""
        return {"video": sorted(VIDEO_DETECTORS), "frame": sorted(FRAME_DETECTORS)}

    @router.post("/video", response_model=DetectResponse)
    async def detect_video(
        request: Request,
        source_id: str = Query(default="upload", max_length=64),
        filename: str = Query(default="video.mp4", max_length=200),
    ) -> DetectResponse:
        """
        Check an uploaded video with every registered video detector.

        Send the raw file as the request body (not multipart).
        """
        UPLOAD_DIR.mkdir(exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)[-80:] or "video.mp4"
        path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"

        size = 0
        with path.open("wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    out.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="That video is larger than 500 MB.")
                out.write(chunk)
        if size == 0:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="The video arrived empty.")

        events: list[IncidentEvent] = []
        for name, fn in VIDEO_DETECTORS.items():
            for event in _run(name, fn, path, source_id):
                if not event.media_ref:
                    event.media_ref = f"uploads/{path.name}"
                events.append(event)

        note = None
        if not VIDEO_DETECTORS:
            note = (
                "No video detectors are registered yet, so nothing was checked. "
                "Optional team plugins go in dashboard/backend/detectors/ — see detect.py. "
                "C3 temporal inference uses the separate /api/v1/c3/ job endpoints."
            )
        return DetectResponse(events=_store_all(store, events), detectors_run=sorted(VIDEO_DETECTORS), note=note)

    @router.post("/frame", response_model=DetectResponse)
    async def detect_frame(
        request: Request,
        source_id: str = Query(default="live-camera", max_length=64),
        t: float = Query(default=0.0, ge=0),
    ) -> DetectResponse:
        """Check one live-camera frame (JPEG body) with every registered frame detector."""
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="The frame arrived empty.")
        if len(body) > MAX_FRAME_BYTES:
            raise HTTPException(status_code=413, detail="That frame is too large.")

        events: list[IncidentEvent] = []
        for name, fn in FRAME_DETECTORS.items():
            events.extend(_run(name, fn, body, source_id, t))

        note = None if FRAME_DETECTORS else "No frame detectors are registered yet."
        return DetectResponse(events=_store_all(store, events), detectors_run=sorted(FRAME_DETECTORS), note=note)

    return router
