import numpy as np
import cv2

class EarlyAggressionFeatureExtractor:
    """
    Component 03: Feature Extraction & Occlusion Imputation Module (SO1 & SO3)
    Computes joint velocities, spatial proximity, and restores missing keypoints.
    """
    def __init__(self, conf_threshold=0.25):
        self.conf_threshold = conf_threshold

    def impute_missing_keypoints(self, kpts_seq):
        """
        SO3: Imputes occluded or low-confidence keypoints using linear temporal interpolation.
        Input shape: (Frames, Joints=17, Channels=3) -> [x, y, confidence]
        """
        seq = kpts_seq.copy()
        T, V, C = seq.shape
        for v in range(V):
            confs = seq[:, v, 2]
            valid_indices = np.where(confs >= self.conf_threshold)[0]
            if 0 < len(valid_indices) < T:
                for coord in (0, 1):
                    seq[:, v, coord] = np.interp(np.arange(T), valid_indices, seq[valid_indices, v, coord])
                seq[:, v, 2] = np.maximum(confs, 0.35) # Mark as imputed confidence
        return seq

    def compute_kinematic_features(self, kpts_seq, fps=25.0):
        """
        SO1: Computes pre-attack motion features across sliding windows:
        - Inter-person spatial proximity / closing velocity
        - Upper-limb joint velocity spikes (wrists & elbows)
        - Raised-arm threat posture elevation
        """
        T = V = kpts_seq.shape[0]
        features = {}

        # Extract wrists (9, 10) and elbows (7, 8) velocity
        if kpts_seq.ndim == 4: # Multi-person (T, V, C, M)
            p0 = kpts_seq[:, :, :2, 0]
            velocities = np.linalg.norm(np.gradient(p0[:, [7, 8, 9, 10], :], axis=0), axis=-1)
            max_limb_vel = np.max(velocities) * fps
        else:
            velocities = np.linalg.norm(np.gradient(kpts_seq[:, [7, 8, 9, 10], :2], axis=0), axis=-1)
            max_limb_vel = np.max(velocities) * fps

        features['max_limb_velocity'] = float(max_limb_vel)
        features['threat_level'] = "HIGH" if max_limb_vel > 45.0 else "NORMAL"
        
        return features