"""Label-independent, deterministic YOLO11-Pose/ByteTrack sequence caches."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from src.causal_pose import CausalPosePreprocessor, PoseConfig, PREPROCESSING_VERSION
from src.foundation_io import json_text, sha256, video_id, write_json
from src.lock_splits import load_locked

CACHE_SCHEMA = 'c3_pose_sequences_v1'
DETECTION_ARRAYS = {'keypoints':(17,3), 'raw_keypoints':(17,3), 'raw_confidence':(17,),
                    'observed':(17,), 'imputed':(17,), 'boxes':(4,), 'quality':(5,), 'track_ids':()}


def pack_frames(frames, frame_wh, config=None):
    processor = CausalPosePreprocessor(config)
    values = {k:[] for k in DETECTION_ARRAYS}
    times,indices,segments,cuts,offsets = [],[],[],[],[0]
    segment = 0
    for frame in frames:
        if frame.get('scene_cut',False):
            segment += 1
        out = processor.update(frame['keypoints'],frame['boxes'],frame['track_ids'],
                               frame['t'],frame_wh,frame.get('scene_cut',False))
        out['raw_keypoints'] = np.nan_to_num(np.asarray(frame['keypoints'],np.float32),nan=0.,posinf=0.,neginf=0.)
        for key in values:
            values[key].append(out[key])
        times.append(frame['t']); indices.append(frame['frame_index'])
        segments.append(segment); cuts.append(frame.get('scene_cut',False))
        offsets.append(offsets[-1]+len(out['track_ids']))
    result = {}
    for key,tail in DETECTION_ARRAYS.items():
        dtype = bool if key in ('observed','imputed') else np.int64 if key=='track_ids' else np.float32
        result[key] = np.concatenate(values[key]).astype(dtype) if values[key] else np.empty((0,*tail),dtype=dtype)
    result.update(t=np.asarray(times,np.float64),frame_index=np.asarray(indices,np.int64),
                  segment_id=np.asarray(segments,np.int64),scene_cut=np.asarray(cuts,bool),
                  frame_offsets=np.asarray(offsets,np.int64))
    return result


def validate_cache(arrays, metadata):
    if metadata.get('schema') != CACHE_SCHEMA or metadata.get('preprocessing',{}).get('version') != PREPROCESSING_VERSION:
        raise ValueError('Incompatible or legacy pose cache')
    n = len(arrays['t'])
    if not n or not np.isfinite(arrays['t']).all() or (np.diff(arrays['t'])<=0).any():
        raise ValueError('Cache needs strictly increasing finite timestamps')
    if any(len(arrays[k])!=n for k in ('frame_index','segment_id','scene_cut')):
        raise ValueError('Frame metadata length mismatch')
    offsets = arrays['frame_offsets']
    count = len(arrays['track_ids'])
    if len(offsets)!=n+1 or offsets[0]!=0 or offsets[-1]!=count or (np.diff(offsets)<0).any():
        raise ValueError('Invalid ragged frame offsets')
    for key,tail in DETECTION_ARRAYS.items():
        if arrays[key].shape != (count,*tail) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid cache array: {key}')
    if (arrays['observed'] & arrays['imputed']).any():
        raise ValueError('A joint cannot be both observed and imputed')
    if ((arrays['keypoints'][...,2]<0)|(arrays['keypoints'][...,2]>1)).any():
        raise ValueError('Keypoint confidence outside [0,1]')
    for i in range(n):
        ids=arrays['track_ids'][offsets[i]:offsets[i+1]]
        if (ids<0).any() or len(np.unique(ids))!=len(ids):
            raise ValueError('Invalid/repeated tracked identity')


def save_cache(path, arrays, metadata):
    validate_cache(arrays,metadata)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream,**arrays,metadata=np.asarray(json_text(metadata)))


def load_cache(path):
    with np.load(path,allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        arrays = {key:archive[key] for key in archive.files if key!='metadata'}
    validate_cache(arrays,metadata)
    return arrays,metadata


def extract_video(row, out_dir, weights='yolo11n-pose.pt', sample_fps=15., start=0., end=None, device='cpu', seed=42):
    import os
    # Keep third-party settings inside the selected C3 cache directory.
    config_root=Path(out_dir).resolve()/'.ultralytics'
    (config_root/'Ultralytics').mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(config_root)
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2
    import random
    import torch
    import ultralytics
    from ultralytics import YOLO
    from src.auto_annotate_precursors_v4 import SceneChangeGuard
    if not np.isfinite([sample_fps,start,*([end] if end is not None else [])]).all() or sample_fps<=0 or start<0 or (end is not None and end<=start):
        raise ValueError('Invalid sampling range')
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError('Provide existing local pose weights; implicit downloads are disabled')
    source = Path(row['video']).resolve()
    content_hash = sha256(source)
    if content_hash != row['content_sha256']:
        raise ValueError('Source video changed after split locking')
    tracker = Path(ultralytics.__file__).parent/'cfg/trackers/bytetrack.yaml'
    config = PoseConfig()
    metadata = {'schema':CACHE_SCHEMA,'source_video':str(source),'video_id':row['video_id'],
                'source_sha256':content_hash,'weights_sha256':sha256(weights),
                'tracker_sha256':sha256(tracker),'sample_fps':sample_fps,'start_s':start,'requested_end_s':end,
                'seed':seed,'device':device,'preprocessing':config.metadata(),
                'implementation_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                    ('pose_sequences.py','causal_pose.py','auto_annotate_precursors_v4.py')},
                'libraries':{name:importlib.metadata.version(name) for name in ('numpy','torch','ultralytics','opencv-python')},
                'quality_columns':['observed_ratio','imputed_ratio','box_iou','usable_ratio','track_quality'],
                'sampler':'sequential decode; first available source frame at/after each target timestamp',
                'determinism':'fixed seeds, deterministic torch algorithms; unsupported nondeterministic operations fail'}
    key = hashlib.sha256(json_text(metadata).encode()).hexdigest()[:24]
    target = Path(out_dir)/(key+'.npz')
    if target.exists():
        arrays,saved=load_cache(target)
        if any(saved.get(k)!=v for k,v in metadata.items()):
            raise ValueError('Existing cache metadata does not match requested configuration')
        return target,saved
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open {source}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wh = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if fps<=0 or n<=0:
            raise ValueError('Invalid video metadata')
        first = int(math.ceil(start*fps))
        last = min(n,int(math.ceil(end*fps))) if end is not None else n
        if first>=last:
            raise ValueError('Empty extraction range')
        cap.set(cv2.CAP_PROP_POS_FRAMES,first)
        model = YOLO(str(weights))
        scene = SceneChangeGuard()
        def frames():
            nonlocal model
            next_sample = first/fps
            for index in range(first,last):
                ok,frame = cap.read()
                if not ok:
                    raise ValueError(f'Unexpected decode failure at frame {index}')
                t = index/fps
                if t+1e-9 < next_sample:
                    continue
                while next_sample <= t+1e-9:
                    next_sample += 1/sample_fps
                cut = scene.update(frame)
                if cut:
                    model = YOLO(str(weights))
                res = model.track(frame,persist=True,tracker=str(tracker),classes=[0],conf=.25,
                                  verbose=False,device=device)[0]
                if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
                    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                    ids = res.boxes.id.cpu().numpy().astype(np.int64)
                    xy = res.keypoints.xy.cpu().numpy().astype(np.float32)
                    conf = res.keypoints.conf
                    conf = conf.cpu().numpy().astype(np.float32) if conf is not None else np.zeros(xy.shape[:2],np.float32)
                    kp = np.concatenate([xy,conf[...,None]],axis=-1)
                else:
                    boxes=np.zeros((0,4),np.float32);ids=np.zeros(0,np.int64);kp=np.zeros((0,17,3),np.float32)
                yield dict(keypoints=kp,boxes=boxes,track_ids=ids,t=t,frame_index=index,scene_cut=cut)
        arrays = pack_frames(frames(),wh,config)
        metadata.update(source_fps=fps,source_frames=n,frame_wh=wh,source_duration_s=n/fps,
                        extracted_start_s=first/fps,extracted_end_s=last/fps)
        save_cache(target,arrays,metadata)
        return target,metadata
    finally:
        cap.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked',default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--out-dir',default='data/pose_cache/v1')
    p.add_argument('--index',default='data/pose_cache/v1/train_index.json')
    p.add_argument('--split',choices=['train','val','test'],default='train')
    p.add_argument('--purpose',choices=['development','evaluation'],default='development')
    p.add_argument('--video-id')
    p.add_argument('--limit',type=int,default=1,help='Default pilot=1; zero extracts all eligible sources')
    p.add_argument('--weights',default='yolo11n-pose.pt')
    p.add_argument('--sample-fps',type=float,default=15.)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--end',type=float)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.split=='test' and args.purpose!='evaluation':
        p.error('Test extraction requires --purpose evaluation; never use test for fitting/tuning')
    if args.limit<0:
        p.error('limit must be nonnegative')
    rows=[r for r in load_locked(args.locked) if r['split']==args.split and
          (not args.video_id or r['video_id']==video_id(args.video_id))]
    if args.limit:
        rows=rows[:args.limit]
    if not rows:
        p.error('No eligible source videos')
    entries={}
    for row in rows:
        path,meta=extract_video(row,args.out_dir,args.weights,args.sample_fps,args.start,args.end,args.device)
        entries[row['video_id']]={'path':str(path.resolve()),'cache_sha256':sha256(path),
                                  'source_sha256':meta['source_sha256']}
        print('CACHED',row['video_id'],path,flush=True)
    write_json(args.index,{'schema':CACHE_SCHEMA,'locked_manifest_sha256':sha256(args.locked),'entries':entries})


if __name__=='__main__':
    main()
