"""
J26-DS-320 — mock alerting API.

Implements the group's agreed REST contract so that:
  * Members 1-3 can POST real events from their pipelines today, against a
    server that validates the contract properly, instead of waiting for the
    final backend;
  * the dashboard can run in VITE_API_MODE=http end to end;
  * nobody has to agree on a database before they can agree on a format.

Run it:
    cd backend_mock
    pip install -r requirements.txt
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for an interactive API explorer that
Members 1-3 can POST from directly in the browser.

This is a PROTOTYPE server. No authentication, in-memory storage, single
process. Do not put it on a public network.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    ActionRequest,
    CameraSource,
    HealthStatus,
    IncidentEvent,
    utc_now_iso,
)
from store import DuplicateEventError, EventNotFoundError, EventStore
from detect import build_router, load_detectors

API_VERSION = "1.1.0"

app = FastAPI(
    title="Security Alerts API",
    version=API_VERSION,
    description=(
        "Shared incident-event API for the Proactive Anomaly Detection and Security "
        "Alerting System. Detection components POST events; the operator dashboard "
        "reads them and PATCHes operator decisions back."
    ),
)

# The Vite dev server runs on a different port, so the browser treats it as a
# different origin. Allowing it here is what lets the dashboard talk to this
# API without a proxy. Tighten this before the system ever leaves a laptop.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = EventStore(persist=os.environ.get("PERSIST_EVENTS", "0") == "1")

# The Check video screen's endpoints. Teammates' models register themselves
# from backend_mock/detectors/ — see detect.py.
load_detectors()
app.include_router(build_router(store))


@app.exception_handler(EventNotFoundError)
async def _event_not_found(_: Request, exc: EventNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f'No event with id "{exc.args[0]}".'})


@app.exception_handler(DuplicateEventError)
async def _duplicate_event(_: Request, exc: DuplicateEventError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": (
                f'An event with id "{exc.args[0]}" already exists. '
                "Each alert needs a unique event_id — try f'{component_id}-{uuid4().hex[:8]}'."
            )
        },
    )


# --------------------------------------------------------------------- health


@app.get("/api/v1/health", response_model=HealthStatus, tags=["system"])
def health() -> HealthStatus:
    """Is the service up, and which components have reported at least once?"""
    return HealthStatus(
        ok=True,
        mode="mock",
        api_version=API_VERSION,
        components_reporting=store.components_reporting(),
        checked_at=utc_now_iso(),
    )


# -------------------------------------------------------------------- cameras


@app.get("/api/v1/cameras", response_model=list[CameraSource], tags=["sources"])
def list_cameras() -> list[CameraSource]:
    """Registered cameras and recorded clips."""
    return store.list_cameras()


# --------------------------------------------------------------------- events


@app.get("/api/v1/events", response_model=list[IncidentEvent], tags=["events"])
def list_events(
    status: str | None = Query(default=None, description="new | acknowledged | false_positive | escalated | resolved"),
    camera_id: str | None = Query(default=None),
    component_id: str | None = Query(default=None, description="component_1 … component_4"),
    incident_type: str | None = Query(default=None),
) -> list[IncidentEvent]:
    """The alert queue, newest first. All filters are optional and combine with AND."""
    return store.list_events(
        status=status,
        camera_id=camera_id,
        component_id=component_id,
        incident_type=incident_type,
    )


@app.get("/api/v1/events/{event_id}", response_model=IncidentEvent, tags=["events"])
def get_event(event_id: str) -> IncidentEvent:
    """One alert with its evidence signals and full audit trail."""
    return store.get_event(event_id)


@app.post("/api/v1/events", response_model=IncidentEvent, status_code=201, tags=["events"])
def submit_event(event: IncidentEvent) -> IncidentEvent:
    """
    **Detection pipelines POST here.**

    The body is one incident event in contract v1.1. A malformed payload comes
    back as a 422 naming the exact field, so check the response body — it will
    tell you precisely what to fix.
    """
    return store.add_event(event)


@app.patch("/api/v1/events/{event_id}", response_model=IncidentEvent, tags=["events"])
def update_event(event_id: str, request: ActionRequest) -> IncidentEvent:
    """
    Record an operator decision.

    The dashboard calls this — pipelines normally do not. Note that a
    `false_alarm` changes the status but never removes the event: rejected
    alerts are needed for the false-alarm rate and the hard-negative set.
    """
    event = store.get_event(event_id)
    if event.status == request.status:
        raise HTTPException(status_code=409, detail=f'Event is already marked "{request.status.value}".')
    return store.apply_action(event_id, request)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "J26-DS-320 Alerting API (mock)",
        "version": API_VERSION,
        "docs": "/docs",
        "events": "/api/v1/events",
        "detectors": "/api/v1/detect/detectors",
    }
