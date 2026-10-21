"""Enrich Component 02 ByteTrack bounding boxes with 17 COCO pose joints.

Component 02's stated contract is person detections/tracks, not pose. This
module runs **pose estimation without a second ByteTrack tracker** and matches
pose detections to the upstream track IDs by one-to-one box overlap.

The matching is a baseline: crowded, occluded, or overlapping people may be
ambiguous. Unmatched persons receive zero-confidence joints (not fabricated
pose estimates), allowing track-quality gating to suppress unsupported alerts.
"""
import numpy as np


def box_iou(a, b):
    a=np.asarray(a,dtype=np.float32).reshape(4)
    b=np.asarray(b,dtype=np.float32).reshape(4)
    x1=max(float(a[0]),float(b[0]));y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]));y2=min(float(a[3]),float(b[3]))
    intersection=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,float(a[2]-a[0]))*max(0.,float(a[3]-a[1]))
    bb=max(0.,float(b[2]-b[0]))*max(0.,float(b[3]-b[1]))
    union=aa+bb-intersection
    return intersection/union if union>0 else 0.


def match_pose_to_tracks(track_boxes, pose_boxes, pose_keypoints, min_iou=0.15):
    """Return (N_tracks,17,3) in upstream TRACK order, retaining IDs externally."""
    track_boxes=np.asarray(track_boxes,dtype=np.float32).reshape(-1,4)
    pose_boxes=np.asarray(pose_boxes,dtype=np.float32).reshape(-1,4)
    pose_keypoints=np.asarray(pose_keypoints,dtype=np.float32).reshape(-1,17,3)
    if len(pose_boxes)!=len(pose_keypoints):
        raise ValueError('Pose box count must match pose keypoint count')
    if not 0<min_iou<=1:raise ValueError('min_iou must be between 0 and 1')
    out=np.zeros((len(track_boxes),17,3),dtype=np.float32)
    pairs=[]
    for i,tb in enumerate(track_boxes):
        for j,pb in enumerate(pose_boxes):
            score=box_iou(tb,pb)
            if score>=min_iou:pairs.append((score,i,j))
    used_tracks=set();used_poses=set()
    for score,i,j in sorted(pairs,key=lambda x:-x[0]):
        if i in used_tracks or j in used_poses:continue
        out[i]=pose_keypoints[j]
        used_tracks.add(i);used_poses.add(j)
    return out


class PoseEnricher:
    """Optional YOLO pose inference *without tracking*. Initialize once per camera worker."""
    def __init__(self,weights='yolo11n-pose.pt',min_iou=0.15):
        from ultralytics import YOLO
        self.pose_model=YOLO(weights)
        self.min_iou=min_iou

    def estimate(self,frame_bgr,upstream_boxes):
        boxes=np.asarray(upstream_boxes,dtype=np.float32).reshape(-1,4)
        if not len(boxes):return np.zeros((0,17,3),np.float32)
        # `.predict`, not `.track`: Component 02 remains the sole owner of ByteTrack IDs.
        res=self.pose_model.predict(frame_bgr,classes=[0],verbose=False)[0]
        if res.boxes is None or not len(res.boxes) or res.keypoints is None:
            return np.zeros((len(boxes),17,3),np.float32)
        pboxes=res.boxes.xyxy.cpu().numpy().astype(np.float32)
        xy=res.keypoints.xy.cpu().numpy().astype(np.float32)
        conf=res.keypoints.conf
        conf=conf.cpu().numpy().astype(np.float32) if conf is not None else np.zeros(xy.shape[:2],np.float32)
        return match_pose_to_tracks(boxes,pboxes,np.concatenate([xy,conf[...,None]],axis=-1),self.min_iou)
