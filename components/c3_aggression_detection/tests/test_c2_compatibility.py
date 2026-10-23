import numpy as np
import pytest
from src.pose_enrichment import match_pose_to_tracks,box_iou

def test_exact_matching_keeps_upstream_order_and_ids_external():
    tracks=np.array([[10,10,50,100],[100,10,140,100]],np.float32)
    poses=tracks[::-1].copy()
    kps=np.zeros((2,17,3),np.float32)
    kps[0,:,0]=222;kps[0,:,2]=.9
    kps[1,:,0]=111;kps[1,:,2]=.9
    out=match_pose_to_tracks(tracks,poses,kps)
    assert out.shape==(2,17,3)
    assert out[0,0,0]==111 and out[1,0,0]==222

def test_zero_confidence_for_unmatched_tracks_never_invent_pose():
    out=match_pose_to_tracks([[0,0,20,20]],[[40,40,60,60]],np.ones((1,17,3)))
    assert not np.any(out)
    assert match_pose_to_tracks(np.zeros((0,4)),np.zeros((0,4)),np.zeros((0,17,3))).shape==(0,17,3)

def test_ambiguous_pairs_are_one_to_one():
    out=match_pose_to_tracks(np.array([[0,0,20,20],[0,0,19,19]]),
                             np.array([[0,0,20,20]]),np.ones((1,17,3)))
    assert np.sum(np.any(out!=0,axis=(1,2)))==1
    assert box_iou([0,0,10,10],[0,0,10,10])==1
    assert box_iou([0,0,10,10],[20,20,30,30])==0
    with pytest.raises(ValueError,match='count'):match_pose_to_tracks([[0,0,10,10]],[[0,0,10,10]],np.zeros((0,17,3)))

def test_per_camera_default_output_paths_are_separate_and_safe():
    from src.output_paths import output_path
    assert output_path('cam1',None,'events.jsonl')!=output_path('cam2',None,'events.jsonl')
    assert '..' not in str(output_path('../cam3',None,'events.jsonl').parts[-2])
    assert output_path('cam1','custom.jsonl','events.jsonl').name=='custom.jsonl'
    with pytest.raises(ValueError):output_path('..',None,'events.jsonl')

def test_c2_adapter_forwards_original_ids_without_own_tracker():
    from src.shared_adapter import C3UpstreamAdapter
    class StubEngine:
        def __init__(self):self.frames=[]
        def append_tracked(self,t,frame,kps,boxes,ids):self.frames.append((t,kps,boxes,ids))
        def assess(self,t,wh,wave):return {'t':t,'wh':wh,'audio':wave},None
    class StubEnricher:
        def estimate(self,frame,boxes):
            result=np.zeros((len(boxes),17,3),np.float32)
            result[:,:,2]=.9
            return result
    a=C3UpstreamAdapter.__new__(C3UpstreamAdapter)
    a.engine=StubEngine();a.pose_enricher=StubEnricher();a.camera_id='cam_02'
    boxes=np.array([[1,2,31,52]],np.float32);ids=np.array([74],np.int32)
    a.push_from_c2(1.0,np.zeros((100,100,3),np.uint8),boxes,ids)
    assert a.engine.frames[0][-1].tolist()==[74]
    assert a.engine.frames[0][1].shape==(1,17,3)
    rec,ev=a.assess(2.0,100,100,None)
    assert rec['wh']==(100,100) and rec['audio'] is None and ev is None
