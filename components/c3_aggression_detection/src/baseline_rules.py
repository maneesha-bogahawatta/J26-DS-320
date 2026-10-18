import numpy as np

class ExplainableAggressionBaseline:
    """
    Component 03 (IT23423992): Explainable Rule-Based Pre-Attack Aggression Baseline (SO6 & FR8)
    Evaluates kinematic DIFEM features to emit calibrated risk scores, track quality,
    uncertainty estimates, and human-readable explanation signals for operator review.
    """
    def __init__(self, watch_threshold=0.48, high_threshold=0.72):
        self.watch_threshold = watch_threshold
        self.high_threshold = high_threshold

        # Reference kinematic thresholds (normalized by bounding-box height per second)
        self.ref_closing_speed = 0.85
        self.ref_wrist_speed = 2.20
        self.ref_arm_raise = 0.12
        self.ref_min_dist = 0.65
        self.ref_limb_accel = 4.50

    def compute_track_quality_and_uncertainty(self, joint_confidences, imputed_ratio=0.0):
        """
        Computes pose tracking quality [0, 1] and epistemic uncertainty [0, 1]
        based on visible keypoint confidence and occlusion imputation fraction.
        """
        if joint_confidences is None or len(joint_confidences) == 0:
            return 0.0, 1.0
        mean_conf = float(np.clip(np.mean(joint_confidences), 0.0, 1.0))
        track_quality = round(mean_conf * (1.0 - 0.5 * float(imputed_ratio)), 3)
        uncertainty = round(1.0 - track_quality, 3)
        return track_quality, uncertainty

    def evaluate_window(self, difem_metrics, joint_confidences=None, imputed_ratio=0.0):
        """
        Evaluates a sliding window of kinematic features and returns structured assessment.
        difem_metrics: dict with keys ['min_dist', 'closing_speed', 'max_wrist_speed',
                                       'max_arm_raise', 'max_limb_accel', 'max_iou']
        """
        clues = []
        score = 0.05

        closing_spd = float(difem_metrics.get("closing_speed", 0.0))
        wrist_spd = float(difem_metrics.get("max_wrist_speed", 0.0))
        arm_raise = float(difem_metrics.get("max_arm_raise", -0.5))
        min_dist = float(difem_metrics.get("min_dist", 4.0))
        limb_acc = float(difem_metrics.get("max_limb_accel", 0.0))
        box_iou = float(difem_metrics.get("max_iou", 0.0))

        if closing_spd >= self.ref_closing_speed:
            score += 0.25
            clues.append({"signal": "rapid_interpersonal_approach", "value": round(closing_spd, 3)})

        if wrist_spd >= self.ref_wrist_speed:
            score += 0.28
            clues.append({"signal": "fast_arm_or_wrist_strike_motion", "value": round(wrist_spd, 3)})

        if arm_raise >= self.ref_arm_raise:
            score += 0.18
            clues.append({"signal": "raised_arm_threat_posture", "value": round(arm_raise, 3)})

        if limb_acc >= self.ref_limb_accel:
            score += 0.15
            clues.append({"signal": "sudden_limb_acceleration_spike", "value": round(limb_acc, 3)})

        if min_dist <= self.ref_min_dist or box_iou >= 0.25:
            score += 0.14
            clues.append({"signal": "close_physical_proximity_or_overlap", "value": round(min_dist, 3)})

        risk_score = round(float(np.clip(score, 0.0, 1.0)), 3)
        if risk_score >= self.high_threshold:
            tier = "HIGH"
        elif risk_score >= self.watch_threshold:
            tier = "WATCH"
        else:
            tier = "LOW"

        track_q, uncert = self.compute_track_quality_and_uncertainty(joint_confidences, imputed_ratio)

        return {
            "risk_score": risk_score,
            "priority_tier": tier,
            "track_quality": track_q,
            "uncertainty": uncert,
            "explanation_signals": clues
        }
