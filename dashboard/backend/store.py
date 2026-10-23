"""
In-memory event store with optional JSON persistence.

Deliberately not a database. For the prototype a dict plus a JSON file is
enough, and it keeps the setup cost for teammates at zero â€” no Postgres, no
migrations, no credentials. When the group decides it needs real persistence,
replace this module; the API layer in main.py does not care how events are
stored.

Thread safety: uvicorn may handle requests on different threads, and a
detection pipeline POSTing while an operator PATCHes is exactly the race this
project will hit. A single lock around mutations is cheap insurance.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schemas import (
    ACTION_TARGET_STATUS,
    ActionRequest,
    AuditEntry,
    CameraSource,
    ComponentId,
    IncidentEvent,
    utc_now_iso,
)

HERE = Path(__file__).parent
SEED_FILE = HERE / "seed_events.json"
PERSIST_FILE = HERE / "events_store.json"


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _render(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _shift_timestamps(payload: dict, generated_at: str | None) -> dict:
    """
    Move every seeded timestamp forward by (now - generated_at).

    Without this, a demo three weeks after the fixture was written shows a
    queue of three-week-old alerts and the "4 min ago" column reads as
    nonsense. Shifting keeps the relative spacing between events â€” which is
    what the response-time column measures â€” while making the queue look
    current. Set SHIFT_SEED_TIMES=0 to keep the literal times in the file.
    """
    if not generated_at or os.environ.get("SHIFT_SEED_TIMES", "1") == "0":
        return payload

    try:
        delta = datetime.now(timezone.utc) - _parse(generated_at)
    except ValueError:
        return payload

    if delta < timedelta(0):
        return payload

    for event in payload.get("events", []):
        if isinstance(event.get("timestamp_detected"), str):
            event["timestamp_detected"] = _render(_parse(event["timestamp_detected"]) + delta)
        for entry in event.get("audit", []) or []:
            if isinstance(entry.get("at"), str):
                entry["at"] = _render(_parse(entry["at"]) + delta)

    return payload


class DuplicateEventError(Exception):
    """Raised when a pipeline POSTs an event_id that already exists."""


class EventNotFoundError(Exception):
    """Raised when an operator action targets an event that is not here."""


class EventStore:
    def __init__(self, persist: bool = False, *, seed: bool = True,
                 persist_file: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, IncidentEvent] = {}
        self._cameras: list[CameraSource] = []
        self._persist = persist
        self._seed = seed
        self._persist_file = Path(persist_file) if persist_file is not None else PERSIST_FILE
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        source = self._persist_file if (self._persist and self._persist_file.exists()) else SEED_FILE if self._seed else None
        if source is None:
            return
        if not source.exists():
            return

        raw = json.loads(source.read_text(encoding="utf-8"))

        # Only the pristine seed gets its timestamps shifted. A persisted store
        # holds real operator actions with real times â€” leave those alone.
        if source == SEED_FILE:
            raw = _shift_timestamps(raw, raw.get("generated_at"))

        for item in raw.get("cameras", []):
            self._cameras.append(CameraSource.model_validate(item))

        for item in raw.get("events", []):
            event = IncidentEvent.model_validate(item)
            self._events[event.event_id] = event

    def _save(self) -> None:
        if not self._persist:
            return
        payload = {
            "cameras": [c.model_dump() for c in self._cameras],
            "events": [e.model_dump(mode="json") for e in self._events.values()],
        }
        self._persist_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._persist_file.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self._persist_file)

    def upsert_camera(self, camera: CameraSource) -> None:
        """Keep recorded C3 results discoverable without any seeded cameras."""
        with self._lock:
            self._cameras = [c for c in self._cameras if c.camera_id != camera.camera_id]
            self._cameras.append(camera)
            self._save()

    # ------------------------------------------------------------------ read

    def list_events(
        self,
        status: str | None = None,
        camera_id: str | None = None,
        component_id: str | None = None,
        incident_type: str | None = None,
    ) -> list[IncidentEvent]:
        events = list(self._events.values())

        if status:
            events = [e for e in events if e.status.value == status]
        if camera_id:
            events = [e for e in events if e.camera_id == camera_id]
        if component_id:
            events = [e for e in events if e.component_id.value == component_id]
        if incident_type:
            events = [e for e in events if e.incident_type == incident_type]

        # Newest first. The dashboard re-sorts for the queue, but a sensible
        # default matters for anyone calling the API with curl.
        return sorted(events, key=lambda e: e.timestamp_detected, reverse=True)

    def get_event(self, event_id: str) -> IncidentEvent:
        event = self._events.get(event_id)
        if event is None:
            raise EventNotFoundError(event_id)
        return event

    def list_cameras(self) -> list[CameraSource]:
        return list(self._cameras)

    def components_reporting(self) -> list[ComponentId]:
        return sorted({e.component_id for e in self._events.values()}, key=lambda c: c.value)

    # ----------------------------------------------------------------- write

    def add_event(self, event: IncidentEvent) -> IncidentEvent:
        with self._lock:
            if event.event_id in self._events:
                raise DuplicateEventError(event.event_id)

            if not event.audit:
                event.audit = [
                    AuditEntry(
                        action="created",
                        at=utc_now_iso(),
                        operator_id=None,
                        note=f"Raised by {event.model_name} v{event.model_version}.",
                        from_status=None,
                        to_status=event.status,
                    )
                ]

            self._events[event.event_id] = event
            self._save()
            return event

    def apply_action(self, event_id: str, request: ActionRequest) -> IncidentEvent:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise EventNotFoundError(event_id)

            target = ACTION_TARGET_STATUS[request.action]
            entry = AuditEntry(
                action=request.action.value,
                at=utc_now_iso(),
                operator_id=request.operator_id,
                note=request.note,
                from_status=event.status,
                to_status=target,
            )

            # Note the event is REPLACED, never removed. A false alarm stays in
            # the record â€” it is a hard negative and part of the false-alarm rate.
            updated = event.model_copy(update={"status": target, "audit": [*event.audit, entry]})
            self._events[event_id] = updated
            self._save()
            return updated
