"""Past-only COCO-17 preprocessing shared by offline caches and online callers."""
from dataclasses import asdict, dataclass
import numpy as np

from src.pose_enrichment import box_iou

PREPROCESSING_VERSION = 'c3_causal_pose_v1'


@dataclass(frozen=True)
class PoseConfig:
    confidence: float = .30
    max_gap_s: float = .25
    ema_alpha: float = .6

    def __post_init__(self):
        if not (0 < self.confidence <= 1 and self.max_gap_s > 0 and 0 < self.ema_alpha <= 1):
            raise ValueError('Invalid causal pose configuration')

    def metadata(self):
        return {'version':PREPROCESSING_VERSION, **asdict(self),
                'normalization':'x/width-0.5,y/height-0.5',
                'imputation':'last observed same-track joint; confidence decays; never future interpolation'}


class CausalPosePreprocessor:
    def __init__(self, config=None):
        self.config = config or PoseConfig()
        self.state = {}
        self.last_t = None

    def update(self, keypoints, boxes, track_ids, t, frame_wh, scene_cut=False):
        t = float(t)
        if not np.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('Timestamps must be finite and strictly increasing')
        width,height = map(float,frame_wh)
        if not np.isfinite([width,height]).all() or min(width,height) <= 0:
            raise ValueError('Invalid frame dimensions')
        kp = np.asarray(keypoints,np.float32).copy()
        boxes = np.asarray(boxes,np.float32).copy()
        ids = np.asarray(track_ids,np.int64)
        if kp.shape != (len(ids),17,3) or boxes.shape != (len(ids),4):
            raise ValueError('Expected keypoints[N,17,3], boxes[N,4], IDs[N]')
        if (ids < 0).any() or len(set(ids.tolist())) != len(ids):
            raise ValueError('Tracked IDs must be unique nonnegative integers')
        if not np.isfinite(boxes).all() or ((boxes[:,2:] - boxes[:,:2]) <= 0).any():
            raise ValueError('Invalid bounding boxes')
        if scene_cut:
            self.state.clear()
        raw_conf = np.where(np.isfinite(kp).all(axis=-1),np.clip(kp[...,2],0,1),0)
        observed = raw_conf >= self.config.confidence
        filled = np.zeros_like(observed)
        pixels = np.zeros_like(kp)
        quality = np.zeros((len(ids),5),np.float32)
        for i,pid in enumerate(ids):
            old = self.state.get(int(pid))
            if old is None or t-old['t'] > self.config.max_gap_s:
                old = {'xy':np.zeros((17,2),np.float32), 'joint_t':np.full(17,-np.inf),
                       'conf':np.zeros(17,np.float32), 'smooth':np.zeros((17,2),np.float32),
                       'valid':np.zeros(17,bool), 'box':boxes[i], 't':t}
            xy = np.zeros((17,2),np.float32)
            xy[observed[i]] = kp[i,observed[i],:2]
            smooth = observed[i] & old['valid']
            xy[smooth] = self.config.ema_alpha*xy[smooth]+(1-self.config.ema_alpha)*old['smooth'][smooth]
            age = t-old['joint_t']
            carry = ~observed[i] & (age < self.config.max_gap_s)
            filled[i] = carry
            xy[carry] = old['xy'][carry]
            conf = np.where(observed[i],raw_conf[i],0).astype(np.float32)
            conf[carry] = old['conf'][carry]*.5*(1-age[carry]/self.config.max_gap_s)
            pixels[i,:,:2],pixels[i,:,2] = xy,conf
            valid = observed[i] | carry
            stability = box_iou(boxes[i],old['box']) if int(pid) in self.state else .5
            vis,imp = observed[i].mean(),carry.mean()
            tq = np.clip(.5*vis+.3*stability+.2*(1-min(1.,2*imp)),0,1) if valid.any() else 0.
            quality[i] = [vis,imp,stability,valid.mean(),tq]
            last_xy,last_t,last_conf = old['xy'].copy(),old['joint_t'].copy(),old['conf'].copy()
            last_xy[observed[i]] = xy[observed[i]]
            last_t[observed[i]] = t
            last_conf[observed[i]] = raw_conf[i,observed[i]]
            self.state[int(pid)] = dict(xy=last_xy,joint_t=last_t,conf=last_conf,
                                       smooth=xy,valid=valid,box=boxes[i].copy(),t=t)
        self.state = {pid:s for pid,s in self.state.items() if t-s['t'] <= self.config.max_gap_s}
        self.last_t = t
        normalized = pixels.copy()
        normalized[...,:2] = pixels[...,:2]/np.array([width,height],np.float32)-.5
        normalized[~(observed|filled)] = 0
        return dict(keypoints=normalized,pixel_keypoints=pixels,raw_confidence=raw_conf,
                    observed=observed,imputed=filled,quality=quality,boxes=boxes,track_ids=ids)


def process_persons_causal(persons, ft, ts, frame_wh):
    """Compatibility adapter for RiskNet's completed feature/window aggregators.

    Each invocation replays its supplied past history from a fresh state. Offline
    preprocessing and live sliding buffers therefore never prefill a KNN bank.
    The legacy imputer and legacy evaluation code remain unchanged for reference.
    """
    from legacy.run import (pose_sequence_feats, aggregate_pose, track_quality_frames,
                            aggregate_quality, skeleton_windows)
    processor = CausalPosePreprocessor()
    clean = []
    for raw,t in zip(persons,ft):
        kp,boxes,ids = raw
        # Untracked detections cannot acquire a synthetic temporal identity.
        keep = np.asarray(ids) >= 0
        kp,boxes,ids = np.asarray(kp)[keep],np.asarray(boxes)[keep],np.asarray(ids)[keep]
        out = processor.update(kp,boxes,ids,t,frame_wh)
        clean.append((out['pixel_keypoints'],boxes.copy(),ids.copy()))
    if len(clean) != len(persons) or len(ft) != len(persons):
        raise ValueError('Pose/time length mismatch')
    # Kinematic derivatives use actual sampled time intervals, not a guessed FPS.
    dt = np.diff(np.asarray(ft,dtype=float),prepend=float(ft[0])-1/8.) if len(ft) else np.array([])
    ff = pose_sequence_feats(clean,1.)
    for j,limit in ((2,5.),(3,15.),(4,15.),(5,30.),(9,30.),(10,30.)):
        ff[:,j] = np.clip(ff[:,j]/dt,-limit if j==2 else 0,limit)
    if len(ff)>1:
        ff[1:,8] = np.minimum(np.abs(np.diff(ff[:,5]))/dt[1:],200.)
    filtered_raw = [(np.asarray(k)[np.asarray(ids)>=0],np.asarray(b)[np.asarray(ids)>=0],np.asarray(ids)[np.asarray(ids)>=0])
                    for k,b,ids in persons]
    qf = track_quality_frames(filtered_raw,clean)
    sk,ids,boxes = skeleton_windows(clean,np.asarray(ft),np.asarray(ts),frame_wh)
    return dict(pose=aggregate_pose(ff,np.asarray(ft),np.asarray(ts)),
                qual=aggregate_quality(qf,np.asarray(ft),np.asarray(ts)),sk=sk,ids=ids,boxes=boxes,
                frame_feats=ff,frame_q=qf,impute={'method':PREPROCESSING_VERSION})
