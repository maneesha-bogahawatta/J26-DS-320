import cv2
import yaml
import numpy as np
from ultralytics import YOLO

class AggressionDetector:
    """
    Component 03: Early Aggression and Violence Recognition Detector
    - Uses YOLOv8-Pose for real-time skeleton keypoint extraction.
    - Implements temporal sliding window and debouncing for threat alerts.
    """
    def __init__(self, config_path="configs/aggression_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_cfg = self.config.get('model', {})
        self.inference_cfg = self.config.get('inference', {})
        
        # Load YOLOv8 Pose Model
        self.pose_model = YOLO("yolov8n-pose.pt")
        self.conf_thresh = self.model_cfg.get('confidence_threshold', 0.70)
        
        # Sliding Window Parameters
        self.stride = self.inference_cfg.get('stride', 4)
        self.required_consecutive = self.inference_cfg.get('consecutive_alerts_required', 3)
        self.cooldown_frames = self.inference_cfg.get('cooldown_frames', 30)
        
        self.alert_counter = 0
        self.cooldown_timer = 0

    def process_frame(self, frame):
        """
        Processes a single CCTV frame, extracts 2D keypoints, 
        and evaluates early aggression risk.
        """
        h, w = frame.shape[:2]
        results = self.pose_model(frame, conf=0.15, verbose=False)[0]
        
        annotated_frame = results.plot() if results.keypoints is not None else frame
        threat_detected = False
        
        if results.keypoints is not None and len(results.keypoints.data) > 0:
            # Basic kinematic check placeholder (velocity/proximity checks)
            # Full multi-stream ST-GCN inference integration point
            threat_detected = True # Placeholder logic for demonstration

        # Debouncing & Temporal Smoothing logic
        if threat_detected:
            if self.cooldown_timer == 0:
                self.alert_counter += 1
        else:
            self.alert_counter = max(0, self.alert_counter - 1)

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

        is_final_alert = False
        if self.alert_counter >= self.required_consecutive:
            is_final_alert = True
            self.cooldown_timer = self.cooldown_frames
            self.alert_counter = 0

        return annotated_frame, is_final_alert