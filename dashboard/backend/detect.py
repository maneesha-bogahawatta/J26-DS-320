"""
Detection endpoints — where each component's trained model plugs in.

The dashboard's "Check video" screen (with VITE_DETECT_MODE=http) sends:

    POST /api/v1/detect/video?source_id=upload-lobby&filename=lobby.mp4
        body: the raw video file
    POST /api/v1/detect/frame?source_id=live-camera&t=12.50
        body: one JPEG from the live camera (about one per second)

Both reply with the same shape:

    { "events": [IncidentEvent, ...], "detectors_run": ["c1_spatial", ...] }
"""


from __future__ import annotations

import importlib
import json
import logging
import os
import pkgutil
import time
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

# Import C1 detector function
from detectors.c1_spatial import process_c1_video
from schemas import IncidentEvent
from store import DuplicateEventError, EventStore

log = logging.getLogger("detect")

HERE = Path(__file__).parent
UPLOAD_DIR = HERE / "uploads"
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_FRAME_BYTES = 8 * 1024 * 1024    # 8 MB

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
    """Import every module in backend/detectors/ so their decorators run."""
    package_dir = HERE / "detectors"
    if not package_dir.is_dir():
        return
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"detectors.{info.name}")
            log.info("loaded detector module %s", info.name)
        except Exception:
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


def _run(name: str, fn: Callable[..., list[IncidentEvent]], *args, **kwargs) -> list[IncidentEvent]:
    started = time.perf_counter()
    try:
        result = fn(*args, **kwargs) or []
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

    @router.post("/video")
    async def detect_video(
        request: Request,
        source_id: str,
        filename: str,
        crowd_limit: int = 3,
        polygon: Optional[str] = None
    ):
        # 1. Ensure uploads directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        video_path = os.path.join(UPLOAD_DIR, filename)

        # 2. Save incoming raw video binary to disk
        body = await request.body()
        if body:
            with open(video_path, "wb") as f:
                f.write(body)

        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Uploaded video file could not be found at {video_path}"
            )

        # 3. Parse optional user polygon
        parsed_polygon = None
        if polygon:
            try:
                parsed_polygon = json.loads(polygon)
            except Exception:
                parsed_polygon = None

        # 4. Run C1 Spatial Engine
        raw_events = process_c1_video(
            video_path=video_path,
            source_id=source_id,
            user_polygon=parsed_polygon,
            crowd_limit=crowd_limit
        )

        validated_events = [
            e if isinstance(e, IncidentEvent) else IncidentEvent.model_validate(e)
            for e in raw_events
        ]
        stored_events = _store_all(store, validated_events)

        return {"events": stored_events, "detectors_run": ["c1_spatial"]}
    

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