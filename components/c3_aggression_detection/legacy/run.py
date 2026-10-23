# C3 - Early Aggression and Violence Recognition (IT23423992) - real-time prototype
# usage:  python run.py --source 0              (webcam)
#         python run.py --source sample.mp4 --out-dir outputs
# Engines: --engine rules (explainable baseline, no weights needed) | --engine risknet --weights risk_model_v3.pt (pose-only streaming; needs c3_model.py + torch)
# Outputs: step_assessments.csv (continuous), incident_alerts.jsonl (debounced alerts), pose_kinematics.csv, evidence_clips/*.mp4
import os, sys, json, math, time, uuid, copy, datetime, argparse, collections
import numpy as np, pandas as pd, cv2

SEED = 42
H_S, STEP_S, POSE_FPS = 3.0, 1.0, 8
SK_T, MAX_P = 16, 2
POSE_DIM, QUAL_DIM = 11, 5
EMA_A, REFRACTORY_S = 0.6, 5.0
CAMERA_ID, SCHEMA_VERSION = 'cam_01', '1.0'
MODEL_VERSION = 'c3_aggression_v3'

# ---- v3 additions (shared with run.py) ----
SMOOTH_MODE = 'ema'          # 'ema' = causal (use for training AND live); 'savgol' = offline analysis only (non-causal: uses future frames)
EMA_ALPHA_KP, EMA_ALPHA_BOX = 0.6, 0.5
SAVGOL_WIN, SAVGOL_ORDER = 7, 2
KP_CONF_THR = 0.30           # joint confidence below this = occluded -> imputed
ALERT_N, ALERT_K, ALERT_K_WATCH, ALERT_K_HIGH = 3, 2, 2, 2    # k-of-n confirmation over the last n steps
CONTACT_DIST = 0.6           # body-heights; below this people are treated as in contact (time-to-contact estimate)

POSE_NAMES = ['n_persons', 'min_dist', 'closing_speed', 'max_speed', 'mean_speed', 'max_wrist_speed',
              'max_arm_raise', 'max_iou', 'max_wrist_accel', 'max_torso_rot', 'max_elbow_speed']
QUAL_NAMES = ['coverage', 'visible_joint_ratio', 'imputed_joint_ratio', 'box_stability', 'track_quality']
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16)]

def _centroid(kp, box):
    hips = kp[[11, 12]]; ok = hips[:, 2] > 0.3
    if ok.any(): return hips[ok, :2].mean(0)
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)

def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def pose_sequence_feats(persons, fps):
    # persons: list over frames of (kps (N,17,3) [x,y,conf], boxes (N,4) xyxy, ids). All features are scale-normalized by box height.
    T = len(persons); feats = np.zeros((T, POSE_DIM), np.float32); feats[:, 1] = 4.0
    mind = np.full(T, np.nan); prev = None
    for t, pr in enumerate(persons):
        kps, boxes = pr[0], pr[1]; n = len(kps)
        if n == 0: prev = None; continue
        cents = np.stack([_centroid(kps[i], boxes[i]) for i in range(n)])
        hts = np.maximum(boxes[:, 3] - boxes[:, 1], 20.0); s = float(np.median(hts))
        feats[t, 0] = min(n, 10)
        if n >= 2:
            d_min, iou_max = 4.0, 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d_min = min(d_min, float(np.linalg.norm(cents[i] - cents[j]) / s))
                    iou_max = max(iou_max, _iou(boxes[i], boxes[j]))
            mind[t] = d_min; feats[t, 1] = d_min; feats[t, 7] = iou_max
        raise_max = -1.0
        for i in range(n):
            for sh, wr in ((5, 9), (6, 10)):
                if kps[i, sh, 2] > 0.3 and kps[i, wr, 2] > 0.3:
                    raise_max = max(raise_max, float((kps[i, sh, 1] - kps[i, wr, 1]) / hts[i]))
        feats[t, 6] = np.clip(raise_max, -1, 2)
        if prev is not None:
            pk, pc = prev; used = set(); sp, ws, tr, es = [], [0.0], [0.0], [0.0]
            for i in range(n):
                d = np.linalg.norm(pc - cents[i], axis=1)
                for u in used: d[u] = np.inf
                j = int(np.argmin(d))
                if d[j] < 1.5 * s:
                    used.add(j); sp.append(float(d[j] / s * fps))
                    for w in (9, 10):
                        if kps[i, w, 2] > 0.3 and pk[j, w, 2] > 0.3:
                            ws.append(float(np.linalg.norm(kps[i, w, :2] - pk[j, w, :2]) / s * fps))
                    for e in (7, 8):
                        if kps[i, e, 2] > 0.3 and pk[j, e, 2] > 0.3:
                            es.append(float(np.linalg.norm(kps[i, e, :2] - pk[j, e, :2]) / s * fps))
                    if min(kps[i, 5, 2], kps[i, 6, 2], pk[j, 5, 2], pk[j, 6, 2]) > 0.3:
                        a1 = np.arctan2(kps[i, 6, 1] - kps[i, 5, 1], kps[i, 6, 0] - kps[i, 5, 0])
                        a0 = np.arctan2(pk[j, 6, 1] - pk[j, 5, 1], pk[j, 6, 0] - pk[j, 5, 0])
                        tr.append(abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi) * fps)
            if sp: feats[t, 3] = min(max(sp), 15.0); feats[t, 4] = min(float(np.mean(sp)), 15.0)
            feats[t, 5] = min(max(ws), 30.0); feats[t, 9] = min(max(tr), 30.0); feats[t, 10] = min(max(es), 30.0)
        prev = (kps, cents)
    for t in range(1, T):
        if not np.isnan(mind[t]) and not np.isnan(mind[t - 1]):
            feats[t, 2] = np.clip(-(mind[t] - mind[t - 1]) * fps, -5, 5)
        feats[t, 8] = min(abs(feats[t, 5] - feats[t - 1, 5]) * fps, 200.0)
    return feats

def aggregate_pose(feats, ft, ts, step=STEP_S):
    out = np.zeros((len(ts), 2 * POSE_DIM), np.float32)
    for k, t in enumerate(ts):
        m = (ft > t - step) & (ft <= t)
        if m.any():
            out[k, :POSE_DIM] = feats[m].mean(0); out[k, POSE_DIM:] = feats[m].max(0)
            out[k, POSE_DIM + 1] = feats[m][:, 1].min()
        else:
            out[k, 1] = 4.0; out[k, POSE_DIM + 1] = 4.0
    return out

# ---------------- occluded keypoint imputation: temporal interpolation + KNN spatial (SO3 / FR6) ----------------
IMPUTED_CONF = 0.35
TEMPLATE = np.array([[0, -.42], [-.02, -.44], [.02, -.44], [-.05, -.42], [.05, -.42], [-.11, -.30], [.11, -.30], [-.14, -.12],
                     [.14, -.12], [-.14, .04], [.14, .04], [-.07, .02], [.07, .02], [-.08, .22], [.08, .22], [-.08, .45], [.08, .45]], np.float32)

def _norm_pose(kp, box):
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32); h = max(float(box[3] - box[1]), 20.0)
    return (kp[:, :2] - c) / h, c, h

class KNNPoseImputer:
    def __init__(self, k=5, thr=KP_CONF_THR, cap=6000):
        self.k, self.thr, self.cap = k, thr, cap
        self.bank = [TEMPLATE.reshape(-1).copy()]; self.B = None; self.dirty = True; self.rs = np.random.RandomState(SEED)
    def add(self, kps, boxes):
        for kp, box in zip(kps, boxes):
            if (kp[:, 2] > 0.5).all():
                q = _norm_pose(kp, box)[0].reshape(-1)
                if len(self.bank) < self.cap: self.bank.append(q)
                else: self.bank[self.rs.randint(self.cap)] = q
                self.dirty = True
    def _mat(self):
        if self.dirty or self.B is None: self.B = np.stack(self.bank).reshape(-1, 17, 2); self.dirty = False
        return self.B
    def impute(self, kp, box):
        vis = kp[:, 2] >= self.thr
        if vis.all() or vis.sum() < 4: return kp, 0
        B = self._mat(); q, c, h = _norm_pose(kp, box); qv = q[vis]
        d = (((B[:, vis] - B[:, vis].mean(1, keepdims=True)) - (qv - qv.mean(0))[None]) ** 2).sum((1, 2))
        idx = np.argsort(d)[:self.k]; w = 1.0 / (d[idx] + 1e-4)
        pred = (B[idx] * w[:, None, None]).sum(0) / w.sum()
        pred = pred - pred[vis].mean(0) + qv.mean(0)
        out = kp.copy(); out[~vis, :2] = pred[~vis] * h + c; out[~vis, 2] = IMPUTED_CONF
        return out, int((~vis).sum())

IMPUTER = KNNPoseImputer()

def impute_sequence(persons, imputer, max_gap=4):
    out = [(p[0].copy(), p[1], p[2]) for p in persons]
    for kps, boxes, ids in out: imputer.add(kps, boxes)
    tracks = {}
    for t, (kps, boxes, ids) in enumerate(out):
        for i, pid in enumerate(ids):
            if pid >= 0: tracks.setdefault(int(pid), []).append((t, i))
    n_t = 0
    for pid, occ in tracks.items():
        if len(occ) < 3: continue
        fr = np.array([o[0] for o in occ])
        for j in range(17):
            conf = np.array([out[t][0][i, j, 2] for t, i in occ]); vis = conf >= imputer.thr
            if vis.all() or vis.sum() < 2: continue
            xs = np.array([out[t][0][i, j, 0] for t, i in occ]); ys = np.array([out[t][0][i, j, 1] for t, i in occ])
            for k_ in np.where(~vis)[0]:
                left, right = np.where(vis[:k_])[0], np.where(vis[k_ + 1:])[0] + k_ + 1
                if len(left) == 0 or len(right) == 0: continue
                l, r = left[-1], right[0]
                if fr[r] - fr[l] > max_gap + 1: continue
                a = (fr[k_] - fr[l]) / (fr[r] - fr[l]); t, i = occ[k_]
                out[t][0][i, j, 0] = xs[l] * (1 - a) + xs[r] * a; out[t][0][i, j, 1] = ys[l] * (1 - a) + ys[r] * a
                out[t][0][i, j, 2] = IMPUTED_CONF; n_t += 1
    n_s = 0
    for kps, boxes, ids in out:
        for i in range(len(kps)):
            kps[i], c = imputer.impute(kps[i], boxes[i]); n_s += c
    return out, dict(temporal=n_t, spatial=n_s)

# ---------------- jitter smoothing (applied AFTER imputation, BEFORE derivatives) ----------------
def ema_smooth_sequence(persons, a_kp=EMA_ALPHA_KP, a_box=EMA_ALPHA_BOX, max_age=3):
    # causal: output at frame t only depends on frames <= t. Smooths x,y of confident joints and the boxes, per ByteTrack id.
    out = [(p[0].copy(), p[1].copy(), p[2]) for p in persons]; state = {}
    for t, (kps, boxes, ids) in enumerate(out):
        for i, pid in enumerate(ids):
            pid = int(pid)
            if pid < 0: continue
            m = (kps[i, :, 2] >= KP_CONF_THR)[:, None]; cur = kps[i, :, :2].copy()
            if pid in state and t - state[pid][2] <= max_age:
                pk, pb, _ = state[pid]
                new = np.where(m, a_kp * cur + (1 - a_kp) * pk, cur); boxes[i] = a_box * boxes[i] + (1 - a_box) * pb
            else:
                new = cur
            kps[i, :, :2] = new; state[pid] = (np.where(m, new, state[pid][0]) if pid in state else new.copy(), boxes[i].copy(), t)
    return out

def savgol_smooth_sequence(persons, win=SAVGOL_WIN, order=SAVGOL_ORDER):
    # NON-causal (centered window = look-ahead): for offline analysis only, do not mix with a live system.
    from scipy.signal import savgol_filter
    out = [(p[0].copy(), p[1].copy(), p[2]) for p in persons]; tracks = {}
    for t, (kps, boxes, ids) in enumerate(out):
        for i, pid in enumerate(ids):
            if pid >= 0: tracks.setdefault(int(pid), []).append((t, i))
    for pid, occ in tracks.items():
        if len(occ) < win: continue
        for j in range(17):
            for c in (0, 1):
                s = np.array([out[t][0][i, j, c] for t, i in occ]); s = savgol_filter(s, win, order)
                for (t, i), v in zip(occ, s): out[t][0][i, j, c] = v
        for c in range(4):
            s = savgol_filter(np.array([out[t][1][i, c] for t, i in occ]), win, order)
            for (t, i), v in zip(occ, s): out[t][1][i, c] = v
    return out

def smooth_sequence(persons, mode=SMOOTH_MODE):
    if mode == 'ema': return ema_smooth_sequence(persons)
    if mode == 'savgol': return savgol_smooth_sequence(persons)
    return persons

# ---------------- track quality (per frame -> per step) ----------------
def track_quality_frames(raw, imputed, thr=KP_CONF_THR):
    # columns: n_tracked, visible_joint_ratio (before imputation), imputed_joint_ratio, box_stability (IoU with previous box), track_quality
    T = len(raw); q = np.zeros((T, 5), np.float32); prev_boxes = {}
    for t in range(T):
        kps, boxes, ids = raw[t]; ik = imputed[t][0]; n = len(ids)
        if n == 0: continue
        vis = (kps[:, :, 2] >= thr).mean(1); imp = ((ik[:, :, 2] >= thr) & (kps[:, :, 2] < thr)).mean(1)
        stab = [(_iou(boxes[i], prev_boxes[int(p)]) if int(p) in prev_boxes else 0.5) for i, p in enumerate(ids)]
        for i, p in enumerate(ids): prev_boxes[int(p)] = boxes[i]
        q[t, 0], q[t, 1], q[t, 2], q[t, 3] = n, vis.mean(), imp.mean(), float(np.mean(stab))
        q[t, 4] = np.clip(0.5 * q[t, 1] + 0.3 * q[t, 3] + 0.2 * (1.0 - min(1.0, 2.0 * q[t, 2])), 0.0, 1.0)
    return q

def aggregate_quality(qf, ft, ts, step=STEP_S):
    # columns: coverage, visible_joint_ratio, imputed_joint_ratio, box_stability, track_quality
    out = np.zeros((len(ts), QUAL_DIM), np.float32)
    for k, t in enumerate(ts):
        m = (ft > t - step) & (ft <= t)
        if not m.any(): continue
        has = qf[m, 0] > 0; out[k, 0] = has.mean()
        if has.any(): out[k, 1:] = qf[m][has][:, 1:].mean(0)
    return out

def skeleton_windows(persons, pt, ts, frame_wh, T=SK_T, M=MAX_P):
    K = len(ts); W, Hh = frame_wh
    sk = np.zeros((K, T, 17, M, 3), np.float32); ids_o = np.full((K, M), -1, np.int32); bx_o = np.zeros((K, M, 4), np.float32)
    for k, t in enumerate(ts):
        idx = np.where(pt <= t)[0][-T:]
        if len(idx) < T // 2: continue
        score = {}
        for i in idx:
            kps, boxes, ids = persons[i]
            for p_ in range(len(ids)):
                score[int(ids[p_])] = score.get(int(ids[p_]), 0.0) + float((boxes[p_, 2] - boxes[p_, 0]) * (boxes[p_, 3] - boxes[p_, 1]))
        if not score: continue
        sel = sorted(score, key=score.get, reverse=True)[:M]
        cs, hs = [], []
        for i in idx:
            kps, boxes, ids = persons[i]
            for p_ in range(len(ids)):
                if int(ids[p_]) in sel: cs.append(_centroid(kps[p_], boxes[p_])); hs.append(boxes[p_, 3] - boxes[p_, 1])
        c, s = np.mean(cs, 0), max(float(np.median(hs)), 20.0)
        for jj, i in enumerate(idx):
            pos = T - len(idx) + jj; kps, boxes, ids = persons[i]
            for m, pid in enumerate(sel):
                w = np.where(ids == pid)[0]
                if len(w):
                    p_ = int(w[0]); sk[k, pos, :, m, 0] = (kps[p_][:, 0] - c[0]) / s; sk[k, pos, :, m, 1] = (kps[p_][:, 1] - c[1]) / s
                    sk[k, pos, :, m, 2] = kps[p_][:, 2]
                    ids_o[k, m] = pid; bx_o[k, m] = boxes[p_] / np.array([W, Hh, W, Hh], np.float32)
    return sk, ids_o, bx_o

def process_persons(persons, ft, ts, frame_wh, smooth_mode=SMOOTH_MODE, imputer=None):
    # Component-03 integration-safe: no shared mutable imputer between cameras.
    # Callers may pass a dedicated per-camera imputer; offline callers get a fresh one.
    imp, info = impute_sequence(persons, imputer if imputer is not None else KNNPoseImputer())
    sm = smooth_sequence(imp, smooth_mode)
    ff = pose_sequence_feats(sm, POSE_FPS); qf = track_quality_frames(persons, imp)
    sk, ids, boxes = skeleton_windows(sm, ft, ts, frame_wh)
    return dict(pose=aggregate_pose(ff, ft, ts), qual=aggregate_quality(qf, ft, ts), sk=sk, ids=ids, boxes=boxes,
                frame_feats=ff, frame_q=qf, impute=info)

# ---------------- structured kinematics export (outputs/pose_kinematics.csv) ----------------
def kinematics_frame_df(ff, qf, ft):
    df = pd.DataFrame(ff, columns=POSE_NAMES); df.insert(0, 'time_s', np.round(np.asarray(ft, dtype=float), 3))
    for j, nm in enumerate(['n_tracked', 'visible_joint_ratio', 'imputed_joint_ratio', 'box_stability', 'track_quality']): df[nm] = qf[:, j]
    return df

def kinematics_window_df(agg, qual, ts):
    cols = [f'{n}_mean' for n in POSE_NAMES] + [('min_dist_min' if i == 1 else f'{n}_max') for i, n in enumerate(POSE_NAMES)]
    df = pd.DataFrame(agg, columns=cols); df.insert(0, 'time_s', np.round(np.asarray(ts, dtype=float), 3))
    for j, nm in enumerate(QUAL_NAMES): df[nm] = qual[:, j]
    return df

TIERS = ['LOW', 'WATCH', 'HIGH']

def tier_from_score(s, thr_watch, thr_high): return 2 if s >= thr_high else 1 if s >= thr_watch else 0

def uncertainty_score(risk, track_quality):
    # HEURISTIC (not epistemic): poor tracking and mid-range risk (0.5) are the most uncertain situations
    return float(np.clip(0.6 * (1.0 - track_quality) + 0.4 * 4.0 * risk * (1.0 - risk), 0.0, 1.0))

def apply_platt(x, calib):
    if calib is None: return float(x)
    a, b = calib; x = float(np.clip(x, 1e-4, 1 - 1e-4)); return float(1.0 / (1.0 + np.exp(-(a * np.log(x / (1 - x)) + b))))

class RuleBasedAggression:
    # Explainable pre-attack baseline (SO6). Inputs: the aggregated step vector [mean | max] from aggregate_pose (scale-normalized, body-heights).
    # Each signal -> activation in [0,1] (linear ramp lo->hi), combined with a noisy-OR. DEFAULT lo/hi are initial heuristics: call fit() on normal data.
    def __init__(self):
        D = POSE_DIM
        self.sig = [  # name, index in vector, weight, lo, hi, needs two persons
            ['rapid approach', D + 2, 0.90, 0.30, 1.50, True],
            ['very close proximity', D + 1, 0.55, 1.20, 0.50, True],
            ['sudden body movement', D + 3, 0.50, 1.00, 5.00, False],
            ['wrist velocity spike', D + 5, 0.80, 4.00, 15.0, False],
            ['elbow velocity spike', D + 10, 0.45, 3.00, 12.0, False],
            ['limb acceleration spike', D + 8, 0.60, 30.0, 120.0, False],
            ['raised arm posture', D + 6, 0.60, 0.10, 0.60, False],
            ['rapid torso rotation', D + 9, 0.40, 2.00, 8.00, False],
            ['bodies overlapping', D + 7, 0.50, 0.05, 0.30, True]]
    def fit(self, normal_agg):
        D = POSE_DIM; X = np.asarray(normal_agg, dtype=np.float64)
        if len(X) < 30: return self
        pair = X[:, D + 0] >= 2
        for s in self.sig:
            v = X[pair, s[1]] if s[5] else X[:, s[1]]
            if len(v) < 20: continue
            if s[3] > s[4]: lo, hi = np.percentile(v, 10), np.percentile(v, 0.5)       # inverted signal (smaller = more aggressive)
            else: lo, hi = np.percentile(v, 90), np.percentile(v, 99.5)
            if abs(hi - lo) > 1e-6 and (lo > hi) == (s[3] > s[4]): s[3], s[4] = float(lo), float(hi)
        return self
    def activations(self, v):
        two = v[POSE_DIM] >= 2; out = []
        for name, i, w, lo, hi, pair in self.sig:
            x = float(v[i]); a = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0)) if (two or not pair) else 0.0
            out.append((name, x, a, w))
        return out
    def score(self, v):
        acts = self.activations(v); p = 1.0
        for _, _, a, w in acts: p *= (1.0 - w * a)
        raw = 1.0 - p
        if v[POSE_DIM] < 2: raw *= 0.5            # a single person cannot show a pre-attack interaction
        return float(np.clip(raw, 0.0, 1.0)), acts
    def score_matrix(self, M): return np.array([self.score(v)[0] for v in M], dtype=np.float64)
    def explain(self, v, min_act=0.3, top=5):
        _, acts = self.score(v)
        return [dict(signal=n, source='pose_rules', value=round(x, 3), activation=round(a, 2))
                for n, x, a, w in sorted(acts, key=lambda z: -z[2]) if a >= min_act][:top]
    def estimate_ttc(self, v):
        d, cs = float(v[POSE_DIM + 1]), float(v[POSE_DIM + 2])
        if v[POSE_DIM] < 2: return None
        if d <= CONTACT_DIST: return 0.0
        return round(min((d - CONTACT_DIST) / cs, 2 * H_S), 1) if cs > 0.1 else None
    def phase(self, v):
        close = float(v[POSE_DIM + 1]) <= CONTACT_DIST * 1.2
        hit = any(a >= 0.5 for n, x, a, w in self.activations(v) if n in ('wrist velocity spike', 'limb acceleration spike', 'elbow velocity spike'))
        return 'contact_or_fight' if (close and hit) else 'pre_attack'
    def save(self, path, **extra):
        json.dump(dict(signals=self.sig, **extra), open(path, 'w'), indent=1)
    @staticmethod
    def load(path):
        d = json.load(open(path)); r = RuleBasedAggression(); r.sig = d['signals']; return r, d

RULES = RuleBasedAggression()

def make_step_record(t, risk, tier, uncertainty, track_quality, ttc, ids, boxes, signals, phase, engine, modality=None):
    pids, bbs = [], []
    if ids is not None:
        for m in range(len(ids)):
            if int(ids[m]) >= 0:
                pids.append(int(ids[m])); bbs.append(dict(person_id=int(ids[m]), bbox_xyxy_norm=[round(float(x), 4) for x in boxes[m]]))
    return dict(time_s=round(float(t), 3), risk_score=round(float(risk), 4), priority_tier=tier, uncertainty=round(float(uncertainty), 4),
                track_quality=round(float(track_quality), 4), est_time_to_contact_s=(None if ttc is None else round(float(ttc), 1)),
                person_ids=pids, bounding_boxes=bbs, explanation_signals=signals, phase=phase, engine=engine,
                modality_status=modality or dict(vision=False, pose=True, audio=False))

class RulesEngine:
    # per-step assessment from the explainable rule baseline (works without any trained weights)
    name = 'rule_baseline'
    def __init__(self, rules=None, calib=None, thr_watch=0.35, thr_high=0.60, ema_a=EMA_A):
        self.rules = rules or RULES; self.calib = calib; self.tw, self.th, self.a = thr_watch, thr_high, ema_a; self.s = None
    def reset(self): self.s = None
    def assess(self, t, agg, qual, ids, boxes):
        raw, _ = self.rules.score(agg); r = apply_platt(raw, self.calib)
        self.s = r if self.s is None else self.a * r + (1 - self.a) * self.s
        risk, tq = float(self.s), float(qual[4])
        return make_step_record(t, risk, TIERS[tier_from_score(risk, self.tw, self.th)], uncertainty_score(risk, tq), tq,
                                self.rules.estimate_ttc(agg), ids, boxes, self.rules.explain(agg), self.rules.phase(agg), self.name)

class AlertDebouncer:
    # Turns the continuous per-step tiers into discrete alerts: k-of-n confirmation + cooldown/refractory + escalation.
    def __init__(self, n=ALERT_N, k_watch=ALERT_K_WATCH, k_high=ALERT_K_HIGH, cooldown=REFRACTORY_S):
        self.n, self.kw, self.kh, self.cd = n, k_watch, k_high, cooldown
        self.hist, self.last_tier, self.last_t = [], 0, -1e9
    def reset(self): self.hist, self.last_tier, self.last_t = [], 0, -1e9
    def update(self, t, tier):
        self.hist = (self.hist + [int(tier)])[-self.n:]
        conf = 2 if sum(h >= 2 for h in self.hist) >= self.kh else 1 if sum(h >= 1 for h in self.hist) >= self.kw else 0
        if conf == 0:
            if len(self.hist) >= self.n and all(h == 0 for h in self.hist): self.last_tier = 0      # re-arm after n quiet steps
            return None
        if conf > self.last_tier or (t - self.last_t) >= self.cd:
            self.last_tier, self.last_t = max(conf, self.last_tier if (t - self.last_t) < self.cd else conf), t
            return conf
        return None

def _isnum(x): return isinstance(x, (int, float)) and not isinstance(x, bool)

EVENT_SCHEMA = dict(schema_version=str, event_id=str, event_type=str, timestamp=str, video_time_s=_isnum, camera_id=str, module=str,
                    threat_score=_isnum, priority_tier=str, uncertainty=_isnum, track_quality=_isnum, person_ids=list,
                    bounding_boxes=list, explanation_signals=list, modality_status=dict, model_version=str)

def validate_event(ev):
    for k, ty in EVENT_SCHEMA.items():
        assert k in ev, f'missing field: {k}'
        assert (ty(ev[k]) if ty is _isnum else isinstance(ev[k], ty)), f'bad type for {k}'
    assert ev['priority_tier'] in ('WATCH', 'HIGH'), 'priority_tier must be WATCH or HIGH for an alert'
    for k in ('threat_score', 'uncertainty', 'track_quality'): assert 0.0 <= ev[k] <= 1.0, f'{k} out of [0,1]'
    assert ev['estimated_time_to_contact_s'] is None or _isnum(ev['estimated_time_to_contact_s']), 'bad estimated_time_to_contact_s'
    for bb in ev['bounding_boxes']:
        assert set(bb) >= {'person_id', 'bbox_xyxy_norm'} and len(bb['bbox_xyxy_norm']) == 4, 'bad bounding box'
    for s in ev['explanation_signals']: assert 'signal' in s, 'explanation signal needs a name'
    datetime.datetime.fromisoformat(ev['timestamp'])
    return True

def build_alert_event(rec, tier, camera_id, ts_iso, model_version=MODEL_VERSION, evidence_clip=None):
    ev = dict(schema_version=SCHEMA_VERSION, event_id=str(uuid.uuid4()),
              event_type='physical_violence_in_progress' if rec['phase'] == 'contact_or_fight' else 'pre_attack_early_warning',
              timestamp=ts_iso, video_time_s=round(float(rec['time_s']), 2), camera_id=camera_id, module='early_aggression_violence',
              threat_score=rec['risk_score'], priority_tier=TIERS[tier], uncertainty=rec['uncertainty'], track_quality=rec['track_quality'],
              estimated_time_to_contact_s=rec['est_time_to_contact_s'], person_ids=rec['person_ids'], bounding_boxes=rec['bounding_boxes'],
              explanation_signals=rec['explanation_signals'], modality_status=rec['modality_status'], model_version=model_version,
              source=rec['engine'], privacy='skeleton_keypoints_only')
    if evidence_clip: ev['evidence_clip'] = evidence_clip
    validate_event(ev); return ev

def records_to_events(recs, camera_id=CAMERA_ID, start_time=None, model_version=MODEL_VERSION, debouncer=None):
    start_time = start_time or datetime.datetime.now(datetime.timezone.utc); deb = debouncer or AlertDebouncer(); events = []
    for rec in recs:
        conf = deb.update(rec['time_s'], TIERS.index(rec['priority_tier']))
        if conf is not None:
            ts = (start_time + datetime.timedelta(seconds=rec['time_s'])).isoformat()
            events.append(build_alert_event(rec, conf, camera_id, ts, model_version))
    return events

def records_to_df(recs):
    df = pd.DataFrame(recs)
    for c in ('person_ids', 'bounding_boxes', 'explanation_signals', 'modality_status'):
        if c in df: df[c] = df[c].map(json.dumps)
    return df

def write_jsonl(events, path):
    with open(path, 'w') as f:
        for ev in events: f.write(json.dumps(ev)); f.write(chr(10))

def make_yolo_pose(weights='yolov8n-pose.pt'):
    from ultralytics import YOLO
    m = YOLO(weights)
    def fn(frame, persist):
        r = m.track(frame, persist=persist, tracker='bytetrack.yaml', imgsz=640, conf=0.35, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return (np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
        ids = r.boxes.id.int().cpu().numpy().astype(np.int64) if r.boxes.id is not None else -1 - np.arange(len(r.boxes))
        return (r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), ids)
    return fn

TIER_BGR = {'LOW': (80, 200, 80), 'WATCH': (0, 200, 255), 'HIGH': (0, 0, 255)}

def draw_hud(canvas, persons, rec, fps_txt=''):
    kps, boxes, ids = persons
    for i in range(len(kps)):
        col = tuple(int(c) for c in (np.array([37, 91, 53]) * (int(ids[i]) % 7 + 1) % 200 + 55))
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]; cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.putText(canvas, f'ID {int(ids[i])}', (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        for a, b in COCO_EDGES:
            if kps[i, a, 2] >= KP_CONF_THR and kps[i, b, 2] >= KP_CONF_THR:
                cv2.line(canvas, (int(kps[i, a, 0]), int(kps[i, a, 1])), (int(kps[i, b, 0]), int(kps[i, b, 1])), col, 2)
        for j in range(17):
            if kps[i, j, 2] >= KP_CONF_THR: cv2.circle(canvas, (int(kps[i, j, 0]), int(kps[i, j, 1])), 3, (255, 255, 255), -1)
    if rec is not None:
        tier = rec['priority_tier']; cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), TIER_BGR[tier], -1)
        ttc = rec['est_time_to_contact_s']
        cv2.putText(canvas, f"{tier}  risk {rec['risk_score']:.2f}  unc {rec['uncertainty']:.2f}  TQ {rec['track_quality']:.2f}"
                    + (f'  TTC {ttc}s' if ttc is not None else '') + f'  {fps_txt}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        for n, s in enumerate(rec['explanation_signals'][:4]):
            cv2.putText(canvas, '- ' + s['signal'], (8, 56 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return canvas

class RiskNetEngine:
    # optional learned engine (pose-only streaming inference: V-JEPA and audio streams are masked out). Needs torch + c3_model.py.
    name = 'risk_net_pose'
    def __init__(self, weights, device=None):
        import torch
        from c3_model import RiskNet
        self.torch = torch; self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ck = torch.load(weights, map_location=self.dev, weights_only=False); dv, dp, da = ck['dims']
        self.model = RiskNet(dv, dp, da).to(self.dev); self.model.load_state_dict(ck['state']); self.model.eval()
        self.norm, self.calib, self.tw, self.th = ck['norm'], ck['calib'], ck['thr_watch'], ck['thr_high']
        self.dims = (dv, dp, da); self.h = None; self.s = None
    def reset(self): self.h = None; self.s = None
    def assess(self, t, agg, qual, ids, boxes, sk):
        torch = self.torch; dv, dp, da = self.dims; mu, sd = self.norm['pose']
        with torch.no_grad():
            x = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)
            pose = x((agg - mu) / sd)[None, None]; skt = x(sk)[None, None]
            pre, fight, tto, self.h = self.model(torch.zeros(1, 1, dv, device=self.dev), pose, torch.zeros(1, 1, da, device=self.dev),
                                                 x(np.array([[[0, 1, 0]]])), skt, self.h)
        pre, fight = float(torch.sigmoid(pre)), float(torch.sigmoid(fight)); r = apply_platt(max(pre, fight), self.calib)
        self.s = r if self.s is None else EMA_A * r + (1 - EMA_A) * self.s
        phase = 'contact_or_fight' if fight >= pre else 'pre_attack'
        ttc = None if phase != 'pre_attack' else round(float(np.clip(float(tto), 0, 2)) * H_S, 1)
        return make_step_record(t, self.s, TIERS[tier_from_score(self.s, self.tw, self.th)], uncertainty_score(self.s, float(qual[4])),
                                float(qual[4]), ttc, ids, boxes, RULES.explain(agg), phase, self.name)

def run_realtime(source=0, engine='rules', weights=None, rules_json=None, camera_id=CAMERA_ID, out_dir='outputs', show=False,
                 privacy='skeleton', yolo_weights='yolov8n-pose.pt', max_seconds=None, pose_fn=None, pre_roll_s=5.0, post_roll_s=3.0,
                 thr_watch=0.35, thr_high=0.60, buf_s=6.0):
    os.makedirs(os.path.join(out_dir, 'evidence_clips'), exist_ok=True)
    is_cam = str(source).isdigit(); cap = cv2.VideoCapture(int(source) if is_cam else source)
    assert cap.isOpened(), f'cannot open source {source}'
    fps = cap.get(cv2.CAP_PROP_FPS); fps = fps if fps and fps > 1 else 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_fn = pose_fn or make_yolo_pose(yolo_weights)
    if engine == 'risknet':
        eng = RiskNetEngine(weights)
    else:
        rules, calib = RULES, None
        if rules_json and os.path.exists(rules_json):
            rules, d = RuleBasedAggression.load(rules_json); calib = d.get('calib'); thr_watch, thr_high = d.get('thr_watch', thr_watch), d.get('thr_high', thr_high)
        eng = RulesEngine(rules, calib, thr_watch, thr_high)
    deb = AlertDebouncer(); t0 = time.time(); buf, ring = [], collections.deque(maxlen=int(pre_roll_s * fps))
    step_recs, events, kin_frames = [], [], []
    alerts_f = open(os.path.join(out_dir, 'incident_alerts.jsonl'), 'w')
    next_pose, next_step, last_kin_t, rec, latest, fi = 0.0, STEP_S, -1.0, None, (np.zeros((0, 17, 3)), np.zeros((0, 4)), np.zeros(0, int)), 0
    writer, rec_until, rec_start = None, -1.0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            t = (time.time() - t0) if is_cam else fi / fps; fi += 1
            if max_seconds and t > max_seconds: break
            if t >= next_pose:
                latest = pose_fn(frame, persist=len(buf) > 0); buf.append((t,) + tuple(latest)); next_pose += 1.0 / POSE_FPS
                while buf and buf[0][0] < t - buf_s: buf.pop(0)
            if t >= next_step and len(buf) >= SK_T // 2:
                next_step += STEP_S; ft = np.array([b[0] for b in buf]); persons = [(b[1], b[2], b[3]) for b in buf]
                res = process_persons(persons, ft, np.array([ft[-1]]), (W, H))
                if engine == 'risknet': rec = eng.assess(t, res['pose'][0], res['qual'][0], res['ids'][0], res['boxes'][0], res['sk'][0])
                else: rec = eng.assess(t, res['pose'][0], res['qual'][0], res['ids'][0], res['boxes'][0])
                step_recs.append(rec)
                m = ft > last_kin_t; last_kin_t = ft[-1]
                if m.any(): kin_frames.append(kinematics_frame_df(res['frame_feats'][m], res['frame_q'][m], ft[m]))
                conf = deb.update(t, TIERS.index(rec['priority_tier']))
                if rec['priority_tier'] != 'LOW': rec_until = t + post_roll_s
                if conf is not None:
                    eid_path = None
                    ev = build_alert_event(rec, conf, camera_id, datetime.datetime.now(datetime.timezone.utc).isoformat(), MODEL_VERSION)
                    if writer is None:
                        eid_path = os.path.join('evidence_clips', ev['event_id'] + '.mp4'); ev['evidence_clip'] = eid_path
                        writer = cv2.VideoWriter(os.path.join(out_dir, eid_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H)); rec_start = t
                        for f_ in ring: writer.write(f_)
                    events.append(ev); alerts_f.write(json.dumps(ev)); alerts_f.write(chr(10)); alerts_f.flush()
            canvas = np.zeros_like(frame) if privacy == 'skeleton' else frame.copy()
            draw_hud(canvas, (np.asarray(latest[0]), np.asarray(latest[1]), np.asarray(latest[2])), rec, f'{fi / max(time.time() - t0, 1e-6):.0f}fps')
            ring.append(canvas)
            if writer is not None:
                writer.write(canvas)
                if t > rec_until or t - rec_start > 60: writer.release(); writer = None
            if show:
                cv2.imshow('C3 aggression (q to quit)', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release(); alerts_f.close()
        if writer is not None: writer.release()
        if show: cv2.destroyAllWindows()
        if step_recs: records_to_df(step_recs).to_csv(os.path.join(out_dir, 'step_assessments.csv'), index=False)
        if kin_frames: pd.concat(kin_frames).to_csv(os.path.join(out_dir, 'pose_kinematics.csv'), index=False)
    return dict(steps=len(step_recs), alerts=len(events), out_dir=out_dir)

def main():
    ap = argparse.ArgumentParser(description='C3 early aggression recognition - real-time prototype')
    ap.add_argument('--source', default='0', help='0 = webcam, or a video path / RTSP url')
    ap.add_argument('--engine', default='rules', choices=['rules', 'risknet'])
    ap.add_argument('--weights', default=None, help='risk_model_v3.pt (for --engine risknet)')
    ap.add_argument('--rules-json', default=None, help='rules_v3.json saved by the notebook (fitted thresholds + calibration)')
    ap.add_argument('--camera-id', default=CAMERA_ID); ap.add_argument('--out-dir', default='outputs')
    ap.add_argument('--yolo', default='yolov8n-pose.pt'); ap.add_argument('--show', action='store_true')
    ap.add_argument('--privacy', default='skeleton', choices=['skeleton', 'off'], help='skeleton = evidence clips show skeletons on black (no faces)')
    ap.add_argument('--max-seconds', type=float, default=None); ap.add_argument('--thr-watch', type=float, default=0.35); ap.add_argument('--thr-high', type=float, default=0.60)
    a = ap.parse_args()
    print(run_realtime(a.source, a.engine, a.weights, a.rules_json, a.camera_id, a.out_dir, a.show, a.privacy, a.yolo, a.max_seconds,
                       thr_watch=a.thr_watch, thr_high=a.thr_high))

if __name__ == '__main__':
    main()
