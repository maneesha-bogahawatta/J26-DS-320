import time
import sys
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add project root (J26-DS-320) to sys.path to resolve shared imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.schemas.event_schema import IncidentEvent


class SpatialAnomalyEngine:
    def __init__(self, portal_line=((300, 450), (900, 450)), 
                 restricted_polygon=None, 
                 tailgating_time_threshold=1.0, 
                 crowd_density_limit=4):
        """
        portal_line: tuple of ((x1, y1), (x2, y2))
        restricted_polygon: list of points [(x1, y1), (x2, y2), ...]
        tailgating_time_threshold: float (seconds)
        crowd_density_limit: int (max persons allowed simultaneously)
        """
        self.portal_line = portal_line
        self.restricted_polygon = np.array(restricted_polygon, np.int32) if restricted_polygon else None
        self.tailgating_time_threshold = tailgating_time_threshold
        self.crowd_density_limit = crowd_density_limit

        self.track_history = {}       # track_id -> list of footprint (x, y)
        self.last_entry_time = None
        self.last_leader_id = None
        self.flagged_tailgaters = set()
        self.active_intrusions = set()
        self.last_crowd_alert_time = 0

    def get_footprint(self, bbox):
        """Calculates bottom-center coordinate of bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def _has_crossed_portal(self, p1, p2):
        """Checks if movement vector from p1 to p2 crossed the portal line downwards."""
        line_y = self.portal_line[0][1]
        line_x_min = min(self.portal_line[0][0], self.portal_line[1][0])
        line_x_max = max(self.portal_line[0][0], self.portal_line[1][0])

        # Check if x is within line boundaries and y transitioned downwards
        if line_x_min <= p2[0] <= line_x_max:
            if p1[1] < line_y <= p2[1]:
                return True
        return False

    def process_frame(self, tracks, camera_id="CAM_01", frame_number=None):
        """
        tracks: list of (track_id, [x1, y1, x2, y2])
        camera_id: camera identifier string
        frame_number: optional int specifying frame index in stream
        
        Returns: list of alert dictionaries conforming to shared IncidentEvent schema
        """
        current_time = time.time()
        alerts = []
        current_footprints = []

        for track_id, bbox in tracks:
            foot_pt = self.get_footprint(bbox)
            current_footprints.append((track_id, foot_pt, bbox))

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(foot_pt)

            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # ----------------------------------------------------
            # 1. TAILGATING DETECTION LOGIC
            # ----------------------------------------------------
            if len(self.track_history[track_id]) >= 2:
                prev_pt = self.track_history[track_id][-2]
                curr_pt = self.track_history[track_id][-1]

                if self._has_crossed_portal(prev_pt, curr_pt):
                    if self.last_entry_time is not None:
                        delta_t = current_time - self.last_entry_time
                        if delta_t < self.tailgating_time_threshold and track_id != self.last_leader_id:
                            self.flagged_tailgaters.add(track_id)
                            leader_id_val = int(self.last_leader_id) if self.last_leader_id is not None else -1
                            
                            # Construct shared IncidentEvent for tailgating
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
            # 2. RESTRICTED ZONE INTRUSION LOGIC (Point-in-Polygon)
            # ----------------------------------------------------
            if self.restricted_polygon is not None:
                dist = cv2.pointPolygonTest(self.restricted_polygon, foot_pt, False)
                if dist >= 0:
                    if track_id not in self.active_intrusions:
                        self.active_intrusions.add(track_id)
                        
                        # Construct shared IncidentEvent for zone intrusion
                        event = IncidentEvent(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            camera_id=camera_id,
                            event_type="zone_intrusion",
                            threat_score=0.88,
                            person_id=f"P-{track_id}",
                            bounding_box=list(map(int, bbox)),
                            zone="restricted_perimeter",
                            explanation_signals=[
                                "point_in_polygon_positive",
                                f"footprint_x{foot_pt[0]}_y{foot_pt[1]}"
                            ],
                            model_version="c1_spatial_v1.0",
                            frame_number=frame_number
                        )
                        alerts.append(event.to_dict())
                else:
                    self.active_intrusions.discard(track_id)

        # ----------------------------------------------------
        # 3. SUDDEN CROWDING DENSITY LOGIC
        # ----------------------------------------------------
        if len(tracks) >= self.crowd_density_limit:
            # Throttle crowding alerts to once every 5 seconds
            if current_time - self.last_crowd_alert_time > 5.0:
                self.last_crowd_alert_time = current_time
                
                # Bounding box of the primary cluster (use first track box as anchor)
                anchor_bbox = list(map(int, tracks[0][1])) if tracks else None
                
                # Construct shared IncidentEvent for crowding
                event = IncidentEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    camera_id=camera_id,
                    event_type="crowding",
                    threat_score=0.80,
                    bounding_box=anchor_bbox,
                    zone="monitored_hallway",
                    explanation_signals=[
                        f"cluster_density_count_{len(tracks)}",
                        f"density_threshold_{self.crowd_density_limit}"
                    ],
                    model_version="c1_spatial_v1.0",
                    frame_number=frame_number
                )
                alerts.append(event.to_dict())

        return alerts