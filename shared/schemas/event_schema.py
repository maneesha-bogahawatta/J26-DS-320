"""
Common incident-event schema used by ALL 4 components.

Every component must output events using the IncidentEvent dataclass.
This is the contract that makes integration possible.

DO NOT MODIFY THIS FILE without agreement from all 4 members.
Changes here affect everyone.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime
import json


@dataclass
class IncidentEvent:
    """
    Shared event format for all components.

    Every detection module (C1-C4) creates one of these
    and sends it to the alert fusion layer.
    """

    # ---- REQUIRED FIELDS (every event must have these) ----
    timestamp: str                          # ISO 8601: "2026-10-01T14:32:05.123Z"
    camera_id: str                          # e.g. "cam_lobby_01"
    event_type: str                         # see EVENT_TYPES below
    threat_score: float                     # 0.0 to 1.0 (confidence)

    # ---- IDENTITY (at least one must be set) ----
    person_id: Optional[str] = None         # tracked person ID e.g. "P-042"
    object_id: Optional[str] = None         # tracked object ID e.g. "O-017"

    # ---- LOCATION ----
    bounding_box: Optional[List[int]] = None  # [x1, y1, x2, y2] in pixels
    zone: Optional[str] = None               # named zone e.g. "main_entrance"

    # ---- EXPLAINABILITY ----
    explanation_signals: List[str] = field(default_factory=list)
    # e.g. ["vertical_velocity_exceeded", "aspect_ratio_below_0.5"]

    # ---- METADATA ----
    model_version: str = "unknown"          # e.g. "c4_fall_v1.2"
    frame_number: Optional[int] = None      # frame index in the video stream
    evidence_frame: Optional[str] = None    # path to saved evidence frame (jpg)

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IncidentEvent":
        """Create event from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, json_str: str) -> "IncidentEvent":
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ============================================================
# VALID EVENT TYPES — use ONLY these strings
# ============================================================
# Adding a new type? Discuss with all members first.

EVENT_TYPES = {
    # Component 1 — Spatial anomaly
    "zone_intrusion":       "Unauthorized entry into restricted zone",
    "tailgating":           "Person following another through secure door",
    "abnormal_proximity":   "Unusual closeness between persons",
    "crowding":             "Sudden crowd formation in monitored zone",

    # Component 2 — Loitering
    "suspicious_loitering": "Person dwelling suspiciously near target area",

    # Component 3 — Aggression
    "aggressive_approach":  "Sudden rapid movement toward another person",
    "fighting":             "Physical altercation detected",
    "raised_arm_gesture":   "Threatening arm movement detected",

    # Component 4 — Critical incidents
    "fall_detected":        "Person fall or collapse detected",
    "abandoned_object":     "Object left unattended beyond threshold",
}

# ============================================================
# PRIORITY LEVELS — used by alert fusion
# ============================================================

PRIORITY = {
    "fall_detected":        1,   # highest — medical emergency
    "fighting":             1,
    "aggressive_approach":  2,
    "abandoned_object":     2,
    "zone_intrusion":       3,
    "tailgating":           3,
    "suspicious_loitering": 4,
    "abnormal_proximity":   4,
    "crowding":             4,
    "raised_arm_gesture":   3,
}


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Example: Component 4 creates a fall event
    event = IncidentEvent(
        timestamp=datetime.utcnow().isoformat() + "Z",
        camera_id="cam_corridor_02",
        event_type="fall_detected",
        threat_score=0.94,
        person_id="P-042",
        bounding_box=[120, 340, 280, 520],
        zone="main_corridor",
        explanation_signals=[
            "vertical_velocity_exceeded",
            "aspect_ratio_below_threshold",
            "temporal_confirmation_10_frames",
        ],
        model_version="c4_fall_v1.0",
        frame_number=1847,
    )

    print("Event JSON:")
    print(event.to_json())
    print()
    print(f"Priority: {PRIORITY.get(event.event_type, 5)}")
