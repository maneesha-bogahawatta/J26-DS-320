"""Live / RTSP / video-file inference. Audio is optional, read from same RTSP/MP4 via ffmpeg.
Do not instantiate a second shared tracker if upstream already supplies timestamped poses.
"""
import argparse,collections,datetime,json,subprocess,threading,time,uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from c3_model import RiskNet
from src.causal_pose import process_persons_causal as process_persons, PREPROCESSING_VERSION
from src.features import visual_feature,audio_feature,DV,DA
from src.audio_events import make_detector, cue_report

class AudioRing:
    def __init__(self,source,sr=16000,seconds=10):
        self.sr=sr;self.buf=collections.deque(maxlen=seconds*sr);self.lock=threading.Lock()
        # RTSP and local files: separate audio decoder; stdout read in independent thread.
        self.proc=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-i',str(source),'-vn','-ac','1',
                                    '-ar',str(sr),'-f','f32le','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        self.alive=True;self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()
    def _read(self):
        while self.alive:
            block=self.proc.stdout.read(4096)
            if not block:break
            with self.lock:self.buf.extend(np.frombuffer(block,dtype='<f4').tolist())
    def latest(self,seconds=1):
        with self.lock:
            if len(self.buf)<self.sr*0.25:return None
            return np.asarray(list(self.buf)[-int(self.sr*seconds):],np.float32)
    def close(self):
        self.alive=False;self.proc.terminate()
        try:self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:self.proc.kill()

class C3Live:
    def __init__(self,weights,pose_weights='yolo11n-pose.pt',camera_id='cam_01'):
        self.ck=torch.load(weights,map_location='cpu',weights_only=False)
        if self.ck.get('preprocessing_version')!=PREPROCESSING_VERSION:
            raise ValueError('Checkpoint preprocessing is incompatible; retrain with causal features or use the legacy reference runner')
        self.model=RiskNet(*self.ck['dims'],d=self.ck.get('hidden_dim',192));self.model.load_state_dict(self.ck['state']);self.model.eval()
        self.audio_events=self.ck.get('audio_events','none')
        self.detector=make_detector(self.audio_events) if self.ck.get('mode')=='av' else None
        self.yolo=YOLO(pose_weights) if pose_weights else None
        self.id=camera_id;self.h=None;self.buf=collections.deque(maxlen=48)
        self.vbuf=collections.deque(maxlen=24);self.last=-1e9;self.votes=collections.deque(maxlen=3)
    def append_frame(self,t,frame):
        if self.yolo is None:raise RuntimeError('Standalone pose tracker disabled; feed frames using append_tracked instead')
        r=self.yolo.track(frame,persist=True,tracker='bytetrack.yaml',verbose=False)[0]
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            bb=r.boxes.xyxy.cpu().numpy().astype('float32'); xy=r.keypoints.xy.cpu().numpy().astype('float32')
            conf=r.keypoints.conf.cpu().numpy().astype('float32') if r.keypoints.conf is not None else np.ones(xy.shape[:2],np.float32)
            kp=np.concatenate([xy,conf[...,None]],axis=-1)
            ids=r.boxes.id.cpu().numpy().astype('int32') if r.boxes.id is not None else -1-np.arange(len(bb))
        else:
            kp=np.zeros((0,17,3),np.float32);bb=np.zeros((0,4),np.float32);ids=np.zeros(0,np.int32)
        self.append_tracked(t,frame,kp,bb,ids)
    def append_tracked(self,t,frame,kp,bb,ids):
        # Shared perception contract: raw frame + timestamp + (N,17,3) COCO pose,
        # (N,4) xyxy boxes, and ByteTrack IDs. Does not call a second YOLO model.
        kp=np.asarray(kp,np.float32);bb=np.asarray(bb,np.float32);ids=np.asarray(ids,np.int32)
        if kp.shape!=(len(ids),17,3) or bb.shape!=(len(ids),4):
            raise ValueError('Expected kps[N,17,3], boxes[N,4], ids[N] with matching N')
        if self.buf and t<=self.buf[-1][0]:raise ValueError('Per-camera timestamps must strictly increase')
        self.buf.append((float(t),kp,bb,ids))
        self.vbuf.append((float(t),visual_feature(frame)))
    def assess(self,t,frame_wh,audio_wave=None):
        if len(self.buf)<8:return None,None
        hist=list(self.buf);ft=np.asarray([r[0] for r in hist],np.float32)
        persons=[(r[1],r[2],r[3]) for r in hist]
        z=process_persons(persons,ft,np.asarray([t],np.float32),frame_wh)
        pose=z['pose'][0];sk=z['sk'][0];q=float(z['qual'][0,4]);norm=self.ck['norm']
        vf=np.stack([v for ts,v in self.vbuf if t-1.<ts<=t]) if any(t-1.<ts<=t for ts,_ in self.vbuf) else np.zeros((1,DV),np.float32)
        vis=vf.mean(0)
        aud=np.zeros(self.ck['dims'][2],np.float32)
        if audio_wave is not None:
            aud[:DA]=audio_feature(audio_wave)
            if self.detector is not None:aud[DA:]=self.detector(audio_wave)
        has_audio=int(audio_wave is not None and self.ck.get('mode')=='av')
        mods=np.asarray([1,1,has_audio],np.float32)
        x=lambda arr:torch.from_numpy(np.asarray(arr,dtype=np.float32))[None,None]
        with torch.no_grad():
            pre,fight,tto,self.h=self.model(x((vis-norm['vis'][0])/norm['vis'][1]),x((pose-norm['pose'][0])/norm['pose'][1]),
                                           x((aud-norm['aud'][0])/norm['aud'][1]),x(mods),x(sk),self.h)
        risk=float(torch.sigmoid(pre)[0,-1]);active=float(torch.sigmoid(fight)[0,-1]);tier='HIGH' if risk>=self.ck['threshold_high'] else 'WATCH' if risk>=self.ck['threshold_watch'] else 'LOW'
        if q<.45:tier='INSUFFICIENT_QUALITY'
        rec=dict(camera_id=self.id,source_time_s=float(t),pre_attack_risk=risk,active_fight_probability=active,
                 priority_tier=tier,track_quality=q,uncertainty=float(max(.1,1-q)),
                 modality='audio_video' if has_audio else 'video_only',
                 audio_cues=cue_report(aud) if has_audio else {},
                 person_ids=[int(i) for i in z['ids'][0] if i>=0],
                 bounding_boxes=z['boxes'][0].astype(float).tolist())
        self.votes.append(tier in ('WATCH','HIGH'))
        event=None
        if len(self.votes)==3 and sum(self.votes)>=2 and self.votes[-1] and t-self.last>=5:
            self.last=t
            event=dict(schema_version='1.0',component='c3_early_aggression',event_id=str(uuid.uuid4()),
                       timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       camera_id=self.id,person_ids=rec['person_ids'],bounding_boxes=[dict(person_id=int(pid),xyxy=box) for pid,box in zip(rec['person_ids'],rec['bounding_boxes'])],
                       threat_score=risk,priority_tier=tier,uncertainty=rec['uncertainty'],track_quality=q,
                       explanation_signals=([dict(signal='temporal_pose_audio_risk',source='c3_risknet')] +
                                            [dict(signal=k,source='yamnet_sound_event',value=v) for k,v in rec['audio_cues'].items() if v>=.35]),
                       modality=rec['modality'],model_version=self.ck['version'],human_review_required=True)
        return rec,event

def run(source,weights,out=None,camera_id='cam_01',audio='auto',show=False,max_seconds=0,calibration=None):
    model=C3Live(weights,camera_id=camera_id)
    if calibration:
        cal=json.loads(Path(calibration).read_text());model.ck['threshold_watch']=float(cal['threshold_watch'])
        model.ck['threshold_high']=max(model.ck['threshold_watch']+.15,model.ck['threshold_high'])
    cam=str(source).isdigit();cap=cv2.VideoCapture(int(source) if cam else source)
    if not cap.isOpened():raise RuntimeError(f'Failed to open {source}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.
    # For offline local files, audio is extracted per timestamp for exact alignment.
    # For RTSP, the decoder uses a bounded live audio ring. Camera 0 defaults to video-only.
    isfile=Path(str(source)).is_file()
    ring=None
    if audio!='off' and not cam and not isfile:
        try:ring=AudioRing(source)
        except (FileNotFoundError,OSError):print('Audio unavailable; running video-only')
    from src.features import wav_segment
    from src.output_paths import output_path
    output=output_path(camera_id,out,'live_events.jsonl');output.parent.mkdir(parents=True,exist_ok=True)
    fi=0;t0=time.monotonic();next_pose=0.;next_step=1.
    with output.open('w',encoding='utf8') as f:
        try:
            while True:
                ok,fr=cap.read()
                if not ok:break
                t=(time.monotonic()-t0) if cam or not isfile else fi/fps
                fi+=1
                if max_seconds and t>max_seconds:break
                if t>=next_pose:
                    model.append_frame(t,fr);next_pose+=1/8.
                if t>=next_step:
                    next_step+=1.
                    wave=None
                    if audio!='off' and model.ck.get('mode')=='av':
                        if isfile:wave=wav_segment(source,max(0,t-1),t)
                        elif ring:wave=ring.latest(1.)
                    if wave is not None and len(wave)<400:wave=None
                    rec,ev=model.assess(t,(fr.shape[1],fr.shape[0]),wave)
                    if rec:
                        print(json.dumps({k:rec[k] for k in ('source_time_s','pre_attack_risk','priority_tier','modality')}))
                    if ev:f.write(json.dumps(ev)+'\n');f.flush()
                if show:
                    cv2.imshow('C3 early violence risk',fr)
                    if cv2.waitKey(1)&255==ord('q'):break
        finally:
            cap.release()
            if ring:ring.close()
            if show:cv2.destroyAllWindows()
    print('Events saved:',output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--camera-id',default='cam_01');p.add_argument('--audio',choices=['auto','off'],default='auto')
    p.add_argument('--out');p.add_argument('--show',action='store_true');p.add_argument('--max-seconds',type=float,default=0)
    p.add_argument('--calibration',help='Validation-only calibrated threshold JSON')
    a=p.parse_args();run(a.source,a.weights,a.out,a.camera_id,a.audio,a.show,a.max_seconds,a.calibration)
