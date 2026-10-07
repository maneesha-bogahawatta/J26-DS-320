import time

class SpatialAnomalyEngine:
    def __init__(self, line_y=450, time_threshold=1.0):
        self.line_y = line_y
        self.time_threshold = time_threshold
        self.track_history = {}
        self.last_entry_time = None
        self.last_leader_id = None
        self.flagged_tailgaters = set()

    def get_footprint(self, bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)

    def process_tracks(self, tracks):
        """
        tracks: list of (track_id, [x1, y1, x2, y2])
        Returns a list of anomaly alerts for the current frame
        """
        current_time = time.time()
        alerts = []

        for track_id, bbox in tracks:
            foot_x, foot_y = self.get_footprint(bbox)

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append((foot_x, foot_y))

            # Retain a rolling buffer of 30 frames
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Check boundary line crossing
            if len(self.track_history[track_id]) >= 2:
                prev_y = self.track_history[track_id][-2][1]
                curr_y = foot_y

                # Downward crossing across the threshold
                if prev_y < self.line_y <= curr_y:
                    if self.last_entry_time is not None:
                        delta_t = current_time - self.last_entry_time
                        if delta_t < self.time_threshold and track_id != self.last_leader_id:
                            self.flagged_tailgaters.add(track_id)
                            alerts.append({
                                "anomaly_type": "TAILGATING",
                                "leader_id": self.last_leader_id,
                                "breacher_id": track_id,
                                "delta_t": round(delta_t, 2),
                                "bbox": bbox
                            })
                        else:
                            self.last_leader_id = track_id
                            self.last_entry_time = current_time
                    else:
                        self.last_leader_id = track_id
                        self.last_entry_time = current_time

        return alerts