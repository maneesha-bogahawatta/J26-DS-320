import unittest
import numpy as np
from src.causal_pose import CausalPosePreprocessor, PoseConfig


def frame(x=30., confidence=1., pid=1):
    kp=np.zeros((1,17,3),np.float32)
    kp[...,:2]=[x,50]
    kp[...,2]=confidence
    return kp,np.array([[0,0,100,100]],np.float32),np.array([pid])


class CausalPoseTests(unittest.TestCase):
    def test_future_perturbation_and_prefix_invariance(self):
        def process(last):
            p=CausalPosePreprocessor()
            seq=[frame(30),frame(0,0),frame(last)]
            return [p.update(*f,i*.1,(100,100)) for i,f in enumerate(seq)]
        a,b=process(40),process(90)
        for i in range(2):
            for key in ('keypoints','pixel_keypoints','quality','imputed','observed'):
                np.testing.assert_array_equal(a[i][key],b[i][key])
        self.assertEqual(a[1]['pixel_keypoints'][0,9,0],30)

    def test_expiration_is_from_last_observation(self):
        p=CausalPosePreprocessor(PoseConfig(max_gap_s=.25))
        p.update(*frame(),0,(100,100))
        a=p.update(*frame(0,0),.1,(100,100))
        b=p.update(*frame(0,0),.2,(100,100))
        c=p.update(*frame(0,0),.3,(100,100))
        self.assertTrue(a['imputed'].all())
        self.assertTrue((b['keypoints'][...,2]<a['keypoints'][...,2]).all())
        self.assertFalse(c['imputed'].any())
        self.assertFalse(c['keypoints'].any())

    def test_scene_cut_and_new_id_never_carry_joints(self):
        for options in ({'scene_cut':True},{'new_id':True}):
            p=CausalPosePreprocessor()
            p.update(*frame(),0,(100,100))
            result=p.update(*frame(0,0,2 if options.get('new_id') else 1),.1,(100,100),
                            scene_cut=options.get('scene_cut',False))
            self.assertFalse(result['imputed'].any())

    def test_bad_time_ids_and_nan_masks(self):
        p=CausalPosePreprocessor()
        f=frame(); f[0][0,2,:]=np.nan
        r=p.update(*f,0,(100,100))
        self.assertFalse(r['observed'][0,2])
        self.assertTrue(np.isfinite(r['keypoints']).all())
        with self.assertRaisesRegex(ValueError,'increasing'):
            p.update(*frame(),0,(100,100))
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            CausalPosePreprocessor().update(*frame(pid=-1),0,(100,100))

    def test_normalization_quality_empty_frame(self):
        p=CausalPosePreprocessor()
        r=p.update(*frame(25),0,(100,100))
        np.testing.assert_allclose(r['keypoints'][0,0],[-.25,0,1])
        self.assertEqual(r['quality'].shape,(1,5))
        empty=p.update(np.zeros((0,17,3)),np.zeros((0,4)),np.array([],int),.1,(100,100))
        self.assertEqual(empty['keypoints'].shape,(0,17,3))


if __name__ == '__main__':
    unittest.main()
