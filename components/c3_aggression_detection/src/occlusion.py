"""Opt-in causal occlusion research over the existing same-track processor.

No cache/checkpoint contract is silently changed. Guards veto unsupported carry;
the optional motion estimate uses two past same-joint observations only. It is
an estimate, never a directly observed coordinate or a verified body joint.
"""
import numpy as np

from src.causal_pose import CausalPosePreprocessor
from src.pose_enrichment import box_iou

COCO_KEYPOINT_NAMES=('nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
    'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle')


class GuardedPosePreprocessor(CausalPosePreprocessor):
    """Conservative experimental policy; existing default v1 stays unchanged."""
    def __init__(self,config=None,motion=False):
        super().__init__(config);self.motion=motion;self.history={};self.audit={}

    def update(self,keypoints,boxes,track_ids,t,frame_wh,scene_cut=False):
        kp=np.asarray(keypoints,np.float32);boxes=np.asarray(boxes,np.float32);ids=np.asarray(track_ids,np.int64)
        # Let the established validator reject invalid dimensions/times first.
        if kp.shape!=(len(ids),17,3) or boxes.shape!=(len(ids),4):
            return super().update(keypoints,boxes,track_ids,t,frame_wh,scene_cut)
        if scene_cut:self.history.clear()
        resets=[];anchors=[]
        for i,pid in enumerate(ids):
            old=self.state.get(int(pid));observed=np.isfinite(kp[i]).all(-1)&(kp[i,:,2]>=self.config.confidence)
            reason=None
            if observed.sum()<6:reason='insufficient_current_anchors'
            elif old is not None:
                diagonal=max(1.,float(np.linalg.norm(old['box'][2:]-old['box'][:2])))
                distance=np.linalg.norm((boxes[i,2:]+boxes[i,:2]-old['box'][2:]-old['box'][:2])/2)/diagonal
                scale=np.prod(boxes[i,2:]-boxes[i,:2])/max(1.,np.prod(old['box'][2:]-old['box'][:2]))
                if box_iou(boxes[i],old['box'])<.05 and (distance>.5 or scale>4 or scale<.25):
                    reason='same_id_box_discontinuity'
                elif float(t)-old['t']>self.config.max_gap_s:reason='expired_track_gap'
            if reason:
                self.state.pop(int(pid),None);self.history.pop(int(pid),None);resets.append(dict(track_id=int(pid),reason=reason))
            anchors.append(int(observed.sum()))
        out=super().update(kp,boxes,ids,t,frame_wh,scene_cut)
        kind=np.where(out['observed'],1,np.where(out['imputed'],2,0)).astype(np.uint8)
        for i,pid in enumerate(ids):
            history=self.history.get(int(pid));pixel=out['pixel_keypoints'][i]
            if self.motion and history is not None and anchors[i]>=8 and box_iou(boxes[i],history['box'])>=.3:
                finite=np.isfinite(history['t_last'])&np.isfinite(history['t_prev'])
                dt=np.ones(17);np.subtract(history['t_last'],history['t_prev'],out=dt,where=finite)
                age=np.ones(17);np.subtract(float(t),history['t_last'],out=age,where=np.isfinite(history['t_last']))
                good=out['imputed'][i]&finite&(dt>=.04)&(dt<=.2)&(age>0)&(age<=.134)&(history['conf_last']>=.5)&(history['conf_prev']>=.5)
                speed=(history['last']-history['prev'])/np.maximum(dt[:,None],1e-6)
                diag=max(1.,float(np.linalg.norm(boxes[i,2:]-boxes[i,:2])))
                displacement=speed*age[:,None]
                predicted=history['last']+displacement
                good&=(np.linalg.norm(speed,axis=-1)<=2*diag)&(np.linalg.norm(displacement,axis=-1)<=.15*diag)
                good&=(predicted>=boxes[i,:2]-.15*(boxes[i,2:]-boxes[i,:2])).all(-1)
                good&=(predicted<=boxes[i,2:]+.15*(boxes[i,2:]-boxes[i,:2])).all(-1)
                from c3_model import COCO_EDGES
                # Historical same-person bone lengths veto implausible extrapolation.
                for joint in np.flatnonzero(good):
                    neighbors=[b if a==joint else a for a,b in COCO_EDGES if a==joint or b==joint]
                    for neighbor in neighbors:
                        if out['observed'][i,neighbor] and history['conf_last'][neighbor]>=.5:
                            old_length=np.linalg.norm(history['last'][joint]-history['last'][neighbor])
                            new_length=np.linalg.norm(predicted[joint]-pixel[neighbor,:2])
                            if old_length>2 and not .5*old_length<=new_length<=1.5*old_length:good[joint]=False
                pixel[good,:2]=predicted[good];pixel[good,2]*=.7;kind[i,good]=3
                out['keypoints'][i,good,:2]=pixel[good,:2]/np.asarray(frame_wh,np.float32)-.5
                out['keypoints'][i,good,2]=pixel[good,2]
            if history is None:
                history=dict(last=np.zeros((17,2),np.float32),prev=np.zeros((17,2),np.float32),
                    t_last=np.full(17,-np.inf),t_prev=np.full(17,-np.inf),conf_last=np.zeros(17),conf_prev=np.zeros(17))
            observed=out['observed'][i]
            history['prev'][observed]=history['last'][observed];history['t_prev'][observed]=history['t_last'][observed]
            history['conf_prev'][observed]=history['conf_last'][observed]
            history['last'][observed]=pixel[observed,:2];history['t_last'][observed]=float(t)
            history['conf_last'][observed]=out['raw_confidence'][i,observed];history['box']=boxes[i].copy()
            self.history[int(pid)]=history
        self.history={pid:value for pid,value in self.history.items() if pid in self.state}
        out.update(estimated=out['imputed'].copy(),missing=~(out['observed']|out['imputed']),estimate_kind=kind)
        self.audit=dict(policy='guarded_motion_v1' if self.motion else 'guarded_hold_v1',resets=resets,
            temporal_estimated_joints=int((kind==2).sum()),motion_estimated_joints=int((kind==3).sum()))
        return out


def draw_pose_evidence(canvas,arrays,index,width,height):
    """Observed solid green; estimated hollow amber; missing locations omitted."""
    import cv2
    from c3_model import COCO_EDGES
    begin,end=arrays['frame_offsets'][index:index+2]
    for j in range(begin,end):
        observed=arrays['observed'][j];estimated=arrays['imputed'][j];visible=observed|estimated
        pixels=((arrays['keypoints'][j,:,:2]+.5)*np.array([width,height])).round().astype(int)
        for a,b in COCO_EDGES:
            if visible[a] and visible[b]:
                color=(50,210,80) if observed[a] and observed[b] else (30,170,255)
                cv2.line(canvas,tuple(pixels[a]),tuple(pixels[b]),color,1,cv2.LINE_AA)
        for k in np.flatnonzero(visible):
            cv2.circle(canvas,tuple(pixels[k]),3 if estimated[k] else 2,
                (30,170,255) if estimated[k] else (50,210,80),1 if estimated[k] else -1)
        box=arrays['boxes'][j]
        # Text is placed at a known box, never at an invented missing-joint position.
        x=int(np.clip((box[0]/max(1,arrays.get('_source_width',width)))*width,0,width-1))
        y=int(np.clip((box[1]/max(1,arrays.get('_source_height',height)))*height,12,height-1))
        track_quality=float(arrays['quality'][j,4]) if 'quality' in arrays else 1.
        low=int(observed.sum())<6 or track_quality<.45
        label=f'ID {int(arrays["track_ids"][j])} O:{int(observed.sum())} E:{int(estimated.sum())} M:{int((~visible).sum())}'
        if low:label+=' LOW POSE'
        cv2.putText(canvas,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.32,(150,150,150) if low else (220,230,240),1,cv2.LINE_AA)
