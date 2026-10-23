"""Safety properties for the opt-in processor; no violence-label assertions."""
import unittest
from unittest.mock import patch

import numpy as np

from src.causal_pose import CausalPosePreprocessor
from src.occlusion import COCO_KEYPOINT_NAMES, GuardedPosePreprocessor, draw_pose_evidence
from src.provisional_labels import proposal_kind


def frame(ids=(1,), shift=0., hidden=()):
    kp=np.zeros((len(ids),17,3),np.float32)
    kp[:,:,:2]=[40+shift,50];kp[:,:,2]=.9
    kp[:,7,:2]=[35+shift,50];kp[:,9,:2]=[40+shift,50]
    kp[:,hidden,2]=0
    return kp,np.tile([0,0,100,100],(len(ids),1)).astype(np.float32),np.array(ids)


class OcclusionTests(unittest.TestCase):
    def test_no_missing_native_pose_matches_default(self):
        baseline=CausalPosePreprocessor();candidate=GuardedPosePreprocessor(motion=True)
        for i in range(5):
            a=baseline.update(*frame(shift=i),i/15,(100,100))
            b=candidate.update(*frame(shift=i),i/15,(100,100))
            for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_causal_prefix_is_unchanged_by_future(self):
        def replay(last):
            p=GuardedPosePreprocessor(motion=True)
            frames=[frame(),frame(shift=1),frame(hidden=(9,)),frame(shift=last)]
            return [p.update(*f,i/15,(100,100)) for i,f in enumerate(frames)]
        a,b=replay(5),replay(50)
        for i in range(3):
            for key in a[i]:np.testing.assert_array_equal(a[i][key],b[i][key])

    def test_partial_missing_masks_and_decay(self):
        for hidden in ((9,),(7,8,9,10),tuple(range(11,17))):
            with self.subTest(hidden=hidden):
                p=GuardedPosePreprocessor();p.update(*frame(),0,(100,100))
                a=p.update(*frame(hidden=hidden),.1,(100,100))
                b=p.update(*frame(hidden=hidden),.2,(100,100))
                c=p.update(*frame(hidden=hidden),.3,(100,100))
                self.assertTrue(a['estimated'][0,list(hidden)].all())
                self.assertFalse((a['observed']&a['estimated']).any())
                self.assertTrue((b['keypoints'][0,list(hidden),2]<a['keypoints'][0,list(hidden),2]).all())
                self.assertTrue(c['missing'][0,list(hidden)].all())
                self.assertFalse(c['keypoints'][0,list(hidden)].any())

    def test_complete_occlusion_vetoes_carried_skeleton(self):
        p=GuardedPosePreprocessor(motion=True);p.update(*frame(),0,(100,100))
        result=p.update(*frame(hidden=tuple(range(17))),.1,(100,100))
        self.assertTrue(result['missing'].all());self.assertFalse(result['estimated'].any())
        self.assertEqual(p.audit['resets'][0]['reason'],'insufficient_current_anchors')

    def test_new_identity_and_scene_cut_never_inherit(self):
        for new_id,cut in ((2,False),(1,True)):
            p=GuardedPosePreprocessor(motion=True);p.update(*frame(),0,(100,100))
            result=p.update(*frame(ids=(new_id,),hidden=(9,)),.1,(100,100),scene_cut=cut)
            self.assertTrue(result['missing'][0,9]);self.assertFalse(result['estimated'].any())

    def test_overlapping_persons_keep_independent_histories(self):
        p=GuardedPosePreprocessor();f=frame(ids=(1,2));f[0][1,:,0]=80
        p.update(*f,0,(100,100));f=frame(ids=(1,2),hidden=(9,))
        r=p.update(*f,.1,(100,100))
        self.assertEqual(r['pixel_keypoints'][0,9,0],40)
        self.assertEqual(r['pixel_keypoints'][1,9,0],80)

    def test_long_absence_does_not_fill(self):
        p=GuardedPosePreprocessor(motion=True);p.update(*frame(),0,(100,100))
        r=p.update(*frame(hidden=(9,)),.4,(100,100))
        self.assertTrue(r['missing'][0,9])

    def test_same_id_large_box_jump_vetoes_fill(self):
        p=GuardedPosePreprocessor();p.update(*frame(),0,(100,100))
        f=frame(shift=500,hidden=(9,));f[1][:,[0,2]]+=500
        r=p.update(*f,.1,(1000,1000))
        self.assertTrue(r['missing'][0,9])
        self.assertEqual(p.audit['resets'][0]['reason'],'same_id_box_discontinuity')

    def test_motion_uses_past_only_and_lower_confidence(self):
        p=GuardedPosePreprocessor(motion=True)
        p.update(*frame(),0,(100,100));p.update(*frame(shift=1),1/15,(100,100))
        r=p.update(*frame(hidden=(9,)),2/15,(100,100))
        self.assertEqual(r['estimate_kind'][0,9],3)
        self.assertGreater(r['pixel_keypoints'][0,9,0],40.6)
        self.assertLess(r['pixel_keypoints'][0,9,2],.9)
        self.assertFalse(r['observed'][0,9])

    def test_fast_motion_falls_back_to_bounded_carry(self):
        p=GuardedPosePreprocessor(motion=True)
        p.update(*frame(),0,(100,100));p.update(*frame(shift=90),1/15,(100,100))
        r=p.update(*frame(hidden=(9,)),2/15,(100,100))
        self.assertEqual(r['estimate_kind'][0,9],2)
        self.assertTrue(np.isfinite(r['keypoints']).all())

    def test_render_distinguishes_estimates_and_omits_missing(self):
        kp=np.zeros((1,17,3),np.float32);observed=np.zeros((1,17),bool);estimated=observed.copy()
        observed[0,5]=True;estimated[0,7]=True
        arrays=dict(frame_offsets=np.array([0,1]),keypoints=kp,observed=observed,imputed=estimated,
                    boxes=np.array([[0,0,100,100]]),track_ids=np.array([1]))
        with patch('cv2.circle') as circles,patch('cv2.putText') as text:
            draw_pose_evidence(np.zeros((100,100,3),np.uint8),arrays,0,100,100)
        self.assertEqual(circles.call_count,2)
        self.assertEqual(circles.call_args_list[0].args[-1],-1)
        self.assertEqual(circles.call_args_list[1].args[-1],1)
        self.assertIn('O:1 E:1 M:15 LOW POSE',text.call_args.args[1])
        self.assertEqual(len(COCO_KEYPOINT_NAMES),17)

    def test_proposals_never_assert_verified_precursors(self):
        self.assertEqual(proposal_kind(True,False,[True,True],True,.9,True),('provisional_preonset_motion',None))
        self.assertEqual(proposal_kind(False,False,[True,False],False,.1,False),('uncertain_boundary',None))
        self.assertEqual(proposal_kind(False,False,[True,True],True,.1,False),('uncertain_or_teacher_disagreement',None))
        self.assertEqual(proposal_kind(False,False,[True,True],False,.1,False),('provisional_quiet_negative',0))

    def test_render_flags_low_track_quality_despite_visible_joints(self):
        arrays=dict(frame_offsets=np.array([0,1]),keypoints=np.zeros((1,17,3),np.float32),
            observed=np.ones((1,17),bool),imputed=np.zeros((1,17),bool),
            quality=np.array([[1,0,0,1,.4]]),boxes=np.array([[0,0,100,100]]),track_ids=np.array([1]))
        with patch('cv2.putText') as text:
            draw_pose_evidence(np.zeros((100,100,3),np.uint8),arrays,0,100,100)
        self.assertIn('O:17 E:0 M:0 LOW POSE',text.call_args.args[1])


if __name__=='__main__':unittest.main()
