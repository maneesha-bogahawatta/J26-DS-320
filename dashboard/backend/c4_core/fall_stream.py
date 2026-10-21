"""Streaming fall monitor: MediaPipe pose -> 6 features -> 6 s sliding window (every 1 s) -> CNN-LSTM ensemble.

Identical feature definition, windowing (6 s, 1 s step, 100 time-resampled steps) and model to the
Colab notebooks (FallCNNLSTM, trained on all 160 GMDCSA-24 clips). Alert rule: p >= thr in k consecutive windows.
"""
import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn

FEATURE_NAMES = ["nose_y", "shoulder_y", "hip_y", "aspect_ratio", "width", "height"]
WIN, STEP, T = 6.0, 1.0, 100
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


class FallCNNLSTM(nn.Module):               # copied from the notebook - must match the saved weights
    def __init__(self, num_features=6, cnn_channels=[32, 64, 128], lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(cnn_channels[2], lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(lstm_hidden, 64); self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        _, (h_n, _) = self.lstm(x.permute(0, 2, 1))
        return self.fc2(self.relu(self.fc1(h_n[-1])))


def load_ensemble(weights_dir, seeds=(0, 1, 2), device="cpu"):
    models = []
    for s in seeds:
        p = os.path.join(weights_dir, f"fall_cnn_lstm_gmdcsa_final_s{s}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing weights file: {p}")
        m = FallCNNLSTM().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)
    return models


def ensemble_proba(models, arr, device="cpu"):
    x = torch.tensor(arr[None]).to(device)
    with torch.no_grad():
        return float(np.mean([torch.softmax(m(x), dim=1)[0, 1].item() for m in models]))


class MediaPipePose:
    """Same landmark features as the notebooks (Cell 3). IMAGE mode, one person (the strongest)."""
    def __init__(self, model_path="pose_landmarker_lite.task"):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise SystemExit("mediapipe is not installed (try Python 3.12: pip install mediapipe)") from e
        if not os.path.exists(model_path):
            print("downloading", POSE_URL)
            urllib.request.urlretrieve(POSE_URL, model_path)
        self.mp = mp
        self.det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path), output_segmentation_masks=False))

    def detect(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not res.pose_landmarks:
            return None, None
        lm = res.pose_landmarks[0]
        xs, ys = [l.x for l in lm], [l.y for l in lm]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        feat = {"nose_y": lm[0].y, "shoulder_y": (lm[11].y + lm[12].y) / 2, "hip_y": (lm[23].y + lm[24].y) / 2,
                "aspect_ratio": w / h if h > 0.01 else 0, "width": w, "height": h}
        H, W = bgr.shape[:2]
        bbox = [max(0, min(xs)) * W, max(0, min(ys)) * H, min(1, max(xs)) * W, min(1, max(ys)) * H]
        return feat, bbox


class FallMonitor:
    def __init__(self, pose, models, thr=0.40, k=1, pose_hz=15.0, device="cpu"):
        self.pose, self.models, self.thr, self.k, self.device = pose, models, thr, k, device
        self.dt = 1.0 / pose_hz
        self.buf = []                       # (t, feat or None)
        self.last_pose_t = -1e9
        self.t_first = None
        self.next_eval = None
        self.streak, self.in_alert = 0, False
        self.last_p, self.last_bbox, self.pose_ok = None, None, False
        self.p_hist = []
        self.status = "warming up"

    def feed(self, frame_bgr, t):
        """Call for every frame. Returns a list of new fall alert dicts (usually empty)."""
        if self.t_first is None:
            self.t_first = t
            self.next_eval = t + WIN
        if t - self.last_pose_t >= self.dt * 0.999:
            self.last_pose_t = t
            feat, bbox = self.pose.detect(frame_bgr)
            if feat is not None:
                feat = dict(feat); feat["t"] = t
                self.last_bbox = bbox
            self.pose_ok = feat is not None
            self.buf.append((t, feat))
            self.buf = [b for b in self.buf if b[0] >= t - WIN - 1.0]
        events = []
        if t >= self.next_eval:
            t1, t0 = t, t - WIN
            self.next_eval += STEP
            valid = [f for tt, f in self.buf if f is not None and t0 <= tt <= t1]
            if len(valid) < 5:               # not enough pose -> no decision (same rule as the notebook)
                self.last_p, self.streak, self.in_alert = None, 0, False
                self.status = "no person / pose"
            else:
                ts = np.array([f["t"] for f in valid])
                grid = np.linspace(ts.min(), ts.max(), T)
                arr = np.stack([np.interp(grid, ts, [f[n] for f in valid]) for n in FEATURE_NAMES], axis=1).astype(np.float32)
                p = ensemble_proba(self.models, arr, self.device)
                self.last_p = p
                self.p_hist.append(p); del self.p_hist[:-40]
                self.status = "monitoring"
                if p >= self.thr:
                    self.streak += 1
                    if self.streak >= self.k and not self.in_alert:
                        self.in_alert = True
                        events.append(dict(t=t, p=p, window=[t0, t1], bbox=self.last_bbox, n_pose=len(valid)))
                else:
                    self.streak, self.in_alert = 0, False
        return events
