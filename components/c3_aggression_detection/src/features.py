"""Causal low-cost RGB, pose, and sound feature extraction; no look-ahead."""
import numpy as np
import cv2
from pathlib import Path

DV, DP, DA = 16, 22, 16
AUDIO_CUE_DIM = 8

def visual_feature(frame):
    img=cv2.cvtColor(cv2.resize(frame,(64,64)),cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
    tiles=[img[y:y+16,x:x+16] for y in (0,16,32,48) for x in (0,16,32,48)]
    return np.asarray([v.mean() for v in tiles],np.float32)

def audio_feature(wave, sr=16000):
    """16 log-power band summaries. All features calculated on available past audio."""
    x=np.asarray(wave,np.float32).reshape(-1)
    if len(x)<256:return np.zeros(DA,np.float32)
    x=x-np.mean(x)
    nfft=512; hop=256
    # cap processing to the most recent 1 second
    x=x[-sr:]
    if len(x)<nfft:x=np.pad(x,(nfft-len(x),0))
    frames=np.lib.stride_tricks.sliding_window_view(x,nfft)[::hop]
    p=np.abs(np.fft.rfft(frames*np.hanning(nfft),axis=-1))**2
    bins=np.array_split(np.arange(1,p.shape[-1]),DA)
    return np.asarray([np.log1p(p[:, b].mean()) for b in bins],np.float32)

def wav_segment(media, start, end, sr=16000):
    """ffmpeg must exist. Handles embedded audio or separate audio files."""
    import subprocess
    duration=max(0.05,end-start)
    cmd=['ffmpeg','-nostdin','-v','error','-ss',str(max(0,start)),'-i',str(media),'-t',str(duration),'-vn','-ac','1','-ar',str(sr),'-f','f32le','pipe:1']
    p=subprocess.run(cmd,capture_output=True)
    if p.returncode or not p.stdout:return None
    return np.frombuffer(p.stdout,dtype='<f4').copy()

def extract_one(row, output_dir, pose_weights='yolo11n-pose.pt', sample_fps=8, step=1., audio_events='none'):
    """Slow one-time extraction with YOLO-Pose+ByteTrack. Produces fixed-size NPZ."""
    from ultralytics import YOLO
    from src.causal_pose import process_persons_causal as process_persons, PREPROCESSING_VERSION
    video=row.get('video',''); source=video or row.get('audio','')
    cap=cv2.VideoCapture(video) if video else None
    if cap is not None and not cap.isOpened():raise RuntimeError(f'Cannot open video: {video}')
    fps=(cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0) or 25
    n=(cap.get(cv2.CAP_PROP_FRAME_COUNT) if cap is not None else 0)
    if cap is None:
        import subprocess, json
        d=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',source]))
        duration=float(d['format']['duration'])
    else:duration=n/fps
    if duration<step:raise ValueError(f'Clip shorter than one step: {source}')
    times=np.arange(step,duration+1e-4,step,dtype=np.float32)
    raw=[]; ft=[]; vf=[]; step_visual=[]; next_sample=0.; decoded_wh=None
    yolo=YOLO(pose_weights) if cap is not None else None
    if cap is not None:
        fi=0
        try:
            while True:
                ok,fr=cap.read()
                if not ok:break
                decoded_wh=(fr.shape[1],fr.shape[0])
                t=fi/fps;fi+=1
                if t+1e-6<next_sample:continue
                next_sample+=1/sample_fps
                res=yolo.track(fr,persist=True,tracker='bytetrack.yaml',verbose=False)[0]
                if res.boxes is not None and len(res.boxes) and res.keypoints is not None:
                    boxes=res.boxes.xyxy.cpu().numpy().astype(np.float32)
                    xy=res.keypoints.xy.cpu().numpy().astype(np.float32)
                    conf=res.keypoints.conf
                    conf=conf.cpu().numpy().astype(np.float32) if conf is not None else np.ones(xy.shape[:2],np.float32)
                    kps=np.concatenate((xy,conf[...,None]),-1)
                    ids=res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else -1-np.arange(len(boxes))
                else:
                    boxes=np.zeros((0,4),np.float32);kps=np.zeros((0,17,3),np.float32);ids=np.zeros(0,int)
                raw.append((kps,boxes,ids));ft.append(t);vf.append(visual_feature(fr))
        finally:cap.release()
        if not raw:raise RuntimeError(f'No decoded video frames: {video}')
        W,H=decoded_wh
        feats=process_persons(raw,np.asarray(ft),times,(W,H))
        pose=feats['pose'];sk=feats['sk'];qual=feats['qual']
        for t in times:
            ix=np.where((np.asarray(ft)>t-step)&(np.asarray(ft)<=t))[0]
            step_visual.append(np.stack([vf[i] for i in ix]).mean(0) if len(ix) else np.zeros(DV,np.float32))
        vis=np.stack(step_visual)
    else:
        vis=np.zeros((len(times),DV),np.float32);pose=np.zeros((len(times),DP),np.float32)
        sk=np.zeros((len(times),16,17,2,3),np.float32);qual=np.zeros((len(times),5),np.float32)
    from src.audio_events import make_detector
    detector=make_detector(audio_events) if (row.get('audio') or row.get('has_audio')) else None
    da=DA+(AUDIO_CUE_DIM if audio_events=='yamnet' else 0)
    aud=np.zeros((len(times),da),np.float32);mods=np.zeros((len(times),3),np.float32)
    if video:mods[:,:2]=1
    media=row.get('audio','') or (video if row['has_audio'] else '')
    if media:
        got=0
        for i,t in enumerate(times):
            wave=wav_segment(media,float(t-step),float(t))
            if wave is not None and len(wave)>256:
                aud[i,:DA]=audio_feature(wave)
                if detector is not None:aud[i,DA:]=detector(wave)
                mods[i,2]=1;got+=1
        if row['has_audio'] and got==0:print('WARNING: expected audio absent:',media)
    stem=__import__('hashlib').sha256((source+row.get('group','')+PREPROCESSING_VERSION).encode()).hexdigest()[:16]
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    path=output/(stem+'.npz')
    if path.exists():raise FileExistsError(f'Refusing to overwrite feature cache: {path}')
    with path.open('xb') as stream:
        np.savez_compressed(stream,t=times,vis=vis,pose=pose,sk=sk,aud=aud,mods=mods,qual=qual,
                            audio_events=np.asarray(audio_events),preprocessing_version=np.asarray(PREPROCESSING_VERSION))
    return str(path)
