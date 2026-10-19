import time
import sys
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.schemas.event_schema import IncidentEvent


class SpatialAnomalyEngine:
    def __init__(self, portal_line=((300, 450), (900, 450)), 
                 restricted_polygon=None,
                 turnstile_barrier_polygon=None,
                 ground_y_level=None,
                 tailgating_time_threshold=1.5, 
                 crowd_density_limit=5,
                 fps=30.0):
        
        self.portal_line = portal_line
        self.restricted_polygon = np.array(restricted_polygon, np.int32) if restricted_polygon else None
        self.turnstile_barrier_polygon = np.array(turnstile_barrier_polygon, np.int32) if turnstile_barrier_polygon else None
        self.ground_y_level = ground_y_level if ground_y_level is not None else portal_line[0][1]
        self.tailgating_time_threshold = tailgating_time_threshold
        self.crowd_density_limit = crowd_density_limit
        self.fps = fps

        self.track_history = {}       # track_id -> list of footprint (x, y)
        self.crossed_tracks = set()   # Tracks that have already triggered a portal crossing
        self.last_entry_time = None
        self.last_leader_id = None
        self.flagged_tailgaters = set()
        self.active_intrusions = set()
        self.active_barrier_breaches = set()
        self.last_crowd_alert_time = 0

    def get_footprint(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def _check_portal_crossing(self, track_id, history):
        """Checks if a footprint trajectory transitioned across the portal threshold line."""
        if track_id in self.crossed_tracks or len(history) < 2:
            return False

        line_y = self.portal_line[0][1]
        line_x_min = min(self.portal_line[0][0], self.portal_line[1][0]) - 60
        line_x_max = max(self.portal_line[0][0], self.portal_line[1][0]) + 60

        p1, p2 = history[-2], history[-1]

        # Check if footprint is within horizontal lane bounds and crosses line_y
        if line_x_min <= p2[0] <= line_x_max:
            if (p1[1] < line_y <= p2[1]) or (p2[1] < line_y <= p1[1]) or abs(p2[1] - line_y) <= 12:
                self.crossed_tracks.add(track_id)
                return True
        return False

    def _is_vaulting(self, bbox, threshold_px=35):
        _, _, _, y2 = bbox
        return (self.ground_y_level - y2) > threshold_px

    def process_frame(self, tracks, camera_id="CAM_01", frame_number=None):
        # Calculate video timestamp if frame_number is provided, else fallback to wall-clock
        if frame_number is not None:
            current_time = frame_number / self.fps
        else:
            current_time = time.time()

        alerts = []

        for track_id, bbox in tracks:
            foot_pt = self.get_footprint(bbox)

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(foot_pt)

            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # ----------------------------------------------------
            # 1. PORTAL CROSSING & TAILGATING LOGIC
            # ----------------------------------------------------
            if self._check_portal_crossing(track_id, self.track_history[track_id]):
                # Check turnstile vaulting
                if self._is_vaulting(bbox):
                    event = IncidentEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        camera_id=camera_id,
                        event_type="zone_intrusion",
                        threat_score=0.98,
                        person_id=f"P-{track_id}",
                        bounding_box=list(map(int, bbox)),
                        zone="turnstile_gate",
                        explanation_signals=[
                            "turnstile_vaulting_detected",
                            "elevated_footprint_crossing"
                        ],
                        model_version="c1_spatial_v1.0",
                        frame_number=frame_number
                    )
                    alerts.append(event.to_dict())

                # Check temporal tailgating slipstream
                if self.last_entry_time is not None:
                    delta_t = current_time - self.last_entry_time
                    if delta_t <= self.tailgating_time_threshold and track_id != self.last_leader_id:
                        self.flagged_tailgaters.add(track_id)
                        leader_id_val = int(self.last_leader_id) if self.last_leader_id is not None else -1
                        
                        event = IncidentEvent(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            camera_id=camera_id,
                            event_type="tailgating",
                            threat_score=0.95,
                            person_id=f"P-{track_id}",
                            bounding_box=list(map(int, bbox)),
                            zone="secure_portal",
                            explanation_signals=[
                                f"authorized_leader_P-{leader_id_val}",
                                f"time_delta_{delta_t:.2f}s",
                                "unauthorized_slipstream_entry"
                            ],
                            model_version="c1_spatial_v1.0",
                            frame_number=frame_number
                        )
                        alerts.append(event.to_dict())
                    else:
                        self.last_leader_id = track_id
                        self.last_entry_time = current_time
                else:
                    self.last_leader_id = track_id
                    self.last_entry_time = current_time

            # ----------------------------------------------------
            # 2. RESTRICTED ZONE INTRUSION LOGIC
            # ----------------------------------------------------
            if self.restricted_polygon is not None:
                dist = cv2.pointPolygonTest(self.restricted_polygon, foot_pt, False)
                if dist >= 0:
                    if track_id not in self.active_intrusions:
                        self.active_intrusions.add(track_id)
                        event = IncidentEvent(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            camera_id=camera_id,
                            event_type="zone_intrusion",
                            threat_score=0.88,
                            person_id=f"P-{track_id}",
                            bounding_box=list(map(int, bbox)),
                            zone="restricted_perimeter",
                            explanation_signals=["point_in_polygon_positive"],
                            model_version="c1_spatial_v1.0",
                            frame_number=frame_number
                        )
                        alerts.append(event.to_dict())
                else:
                    self.active_intrusions.discard(track_id)

        # ----------------------------------------------------
        # 3. CROWD DENSITY LOGIC (Throttled per 5 seconds)
        # ----------------------------------------------------
        if len(tracks) >= self.crowd_density_limit:
            if current_time - self.last_crowd_alert_time > 5.0:
                self.last_crowd_alert_time = current_time
                anchor_bbox = list(map(int, tracks[0][1])) if tracks else None
                event = IncidentEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    camera_id=camera_id,
                    event_type="crowding",
                    threat_score=0.80,
                    bounding_box=anchor_bbox,
                    zone="monitored_hallway",
                    explanation_signals=[f"cluster_density_count_{len(tracks)}"],
                    model_version="c1_spatial_v1.0",
                    frame_number=frame_number
                )
                alerts.append(event.to_dict())

        return alerts