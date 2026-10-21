"""
Pydantic models for the J26-DS-320 incident event contract (v1.1).

This is the server-side twin of `frontend/src/types/events.ts`. Keep the two
in step — if you add a field here, add it there, and update
`docs/EVENT_CONTRACT.md`.

These models are deliberately strict. A teammate whose pipeline sends pixel
coordinates instead of normalised ones, or a local timestamp instead of UTC,
gets a clear 422 naming the field — which is far kinder than a dashboard that
silently draws the box in the wrong place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComponentId(str, Enum):
    """Which group component produced the event."""

    C1_SPATIAL = "component_1"
    C2_LOITERING = "component_2"
    C3_AGGRESSION = "component_3"
    C4_CRITICAL = "component_4"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class OperatorAction(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    FALSE_ALARM = "false_alarm"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


#: Which status each operator action moves an event into.
ACTION_TARGET_STATUS: dict[OperatorAction, EventStatus] = {
    OperatorAction.ACKNOWLEDGE: EventStatus.ACKNOWLEDGED,
    OperatorAction.FALSE_ALARM: EventStatus.FALSE_POSITIVE,
    OperatorAction.ESCALATE: EventStatus.ESCALATED,
    OperatorAction.RESOLVE: EventStatus.RESOLVED,
}


class EvidenceSignal(BaseModel):
    """One behavioural signal that contributed to the alert."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description='snake_case, e.g. "owner_separation"')
    value: float | str | bool | None = Field(..., description="Measurement, or null if the signal carries none")
    unit: str | None = Field(default=None, description='e.g. "seconds". Null for unitless scores.')
    description: str | None = Field(
        default=None,
        description="Plain-language explanation shown to the operator — the explainability layer.",
    )


class RelatedTrack(BaseModel):
    """
    A second track involved in the same incident.

    Some incidents are about one track (a person falling). Others are about a
    RELATIONSHIP between tracks and cannot be expressed with a single track_id:
    an abandoned object has an object track and a likely owner track;
    tailgating has a leader and a follower.
    """

    model_config = ConfigDict(extra="forbid")

    track_id: str = Field(..., min_length=1, description='Anonymous label, e.g. "person-31"')
    role: str = Field(
        ...,
        min_length=1,
        description="likely_owner | follower | leader | counterpart | object | bystander",
    )
    note: str | None = Field(default=None, description="How the association was inferred")
    association_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in the association. CCTV cannot prove ownership, only infer it.",
    )

    @field_validator("track_id")
    @classmethod
    def _not_a_name(cls, value: str) -> str:
        parts = value.split()
        if len(parts) == 2 and all(p.istitle() for p in parts):
            raise ValueError('Related track IDs are anonymous labels too, e.g. "person-31".')
        return value


class BoundingBox(BaseModel):
    """Normalised bounding box. All values in 0..1 — never pixels."""

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)


class AuditEntry(BaseModel):
    """Append-only record of what happened to an event."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["created", "acknowledge", "false_alarm", "escalate", "resolve"]
    at: str
    operator_id: str | None = None
    note: str | None = None
    from_status: EventStatus | None = None
    to_status: EventStatus


class IncidentEvent(BaseModel):
    """The alert a detection pipeline POSTs to /api/v1/events."""

    model_config = ConfigDict(extra="forbid")

    # --- required ---
    event_id: str = Field(..., min_length=1, description="Unique across the whole system")
    component_id: ComponentId
    incident_type: str = Field(..., min_length=1, description='e.g. "fall", "loitering", "tailgating"')
    camera_id: str = Field(..., min_length=1)
    timestamp_detected: str = Field(..., description='UTC ISO-8601, e.g. "2026-10-18T09:15:22Z"')
    priority: Priority
    status: EventStatus = EventStatus.NEW
    summary: str = Field(..., min_length=1, description="Cautious wording. Never claims proof.")
    evidence: list[EvidenceSignal] = Field(default_factory=list)
    model_name: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)

    # --- optional: send null when unavailable ---
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bounding_box: BoundingBox | None = None
    track_id: str | None = Field(
        default=None,
        description='Anonymous label for the SUBJECT of the alert, e.g. "person-12". '
        "For an abandoned object this is the object track; the owner goes in related_tracks.",
    )
    related_tracks: list[RelatedTrack] | None = Field(
        default=None, description="Other tracks involved in the same incident, with their role"
    )
    media_ref: str | None = None
    media_offset_seconds: float | None = Field(default=None, ge=0.0)
    score_calibrated: bool = False
    zone_id: str | None = None
    audit: list[AuditEntry] = Field(default_factory=list)

    @field_validator("timestamp_detected")
    @classmethod
    def _must_be_iso(cls, value: str) -> str:
        """Accept ISO-8601; normalise a trailing Z so downstream parsing is uniform."""
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                'Must be ISO-8601 UTC, e.g. "2026-10-18T09:15:22Z". '
                "In Python: datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')"
            ) from exc
        return value

    @field_validator("bounding_box")
    @classmethod
    def _box_must_have_area(cls, box: BoundingBox | None) -> BoundingBox | None:
        if box is None:
            return None
        if box.x_min >= box.x_max or box.y_min >= box.y_max:
            raise ValueError("min values must be smaller than max values — the box has no area")
        return box

    @field_validator("track_id")
    @classmethod
    def _track_id_is_not_a_name(cls, value: str | None) -> str | None:
        """Privacy guard: the contract says track IDs are not personal identities."""
        if value and len(value.split()) == 2 and all(part.istitle() for part in value.split()):
            raise ValueError(
                'track_id looks like a person\'s name. Use an anonymous tracker label such as "person-12". '
                "The system holds no identities."
            )
        return value


class CameraSource(BaseModel):
    """A camera or recorded clip the system can show."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    label: str
    kind: Literal["recorded_clip", "live_stream"] = "recorded_clip"
    media_ref: str | None = None
    source_note: str | None = None


class ActionRequest(BaseModel):
    """Body of PATCH /api/v1/events/{event_id} — the operator's decision."""

    model_config = ConfigDict(extra="forbid")

    status: EventStatus
    action: OperatorAction
    operator_id: str | None = None
    note: str | None = None


class HealthStatus(BaseModel):
    ok: bool
    mode: Literal["mock", "live"]
    api_version: str
    components_reporting: list[ComponentId]
    checked_at: str


def utc_now_iso() -> str:
    """The one place a timestamp is minted, so the format never drifts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
