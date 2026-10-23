import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.causal_pose import PREPROCESSING_VERSION, process_persons_causal
from src.features import extract_one


class CompatibilityTests(unittest.TestCase):
    def test_risknet_features_causal_with_irregular_timestamps(self):
        def frames(future):
            result=[]
            for i in range(8):
                kp=np.ones((1,17,3),np.float32)
                kp[...,:2]=[20+i,50]
                if i==2:
                    kp[0,9]=[0,0,0]
                if i>4:
                    kp[...,:2]+=future
                result.append((kp,np.array([[0,0,100,100]],np.float32),np.array([1])))
            return result
        t=np.array([0,.12,.24,.36,.48,.61,.73,.85])
        a=process_persons_causal(frames(0),t,np.array([.36,.48]),(100,100))
        b=process_persons_causal(frames(100),np.r_[t[:5],[1.1,1.3,1.8]],np.array([.36,.48]),(100,100))
        for key in ('pose','qual','sk'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_existing_feature_extractor_empty_detections_and_metadata(self):
        class EmptyYOLO:
            def __init__(self,*args,**kwargs):pass
            def track(self,*args,**kwargs):
                return [SimpleNamespace(boxes=None,keypoints=None)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); video=tmp/'synthetic.mp4'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10.,(100,80))
            if not writer.isOpened():self.skipTest('MP4 encoder unavailable')
            for i in range(20):writer.write(np.full((80,100,3),i,dtype=np.uint8))
            writer.release()
            row=dict(video=str(video),group='fixture',has_audio=False)
            with patch('ultralytics.YOLO',EmptyYOLO):
                path=extract_one(row,tmp/'cache')
                with np.load(path) as cache:
                    self.assertEqual(cache['sk'].shape,(2,16,17,2,3))
                    self.assertEqual(cache['preprocessing_version'].item(),PREPROCESSING_VERSION)
                    self.assertFalse(cache['sk'].any())
                with self.assertRaises(FileExistsError):extract_one(row,tmp/'cache')

    def test_legacy_feature_cache_cannot_enter_training(self):
        from src.train import load_cached
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.npz'
            np.savez(path,t=np.arange(3))
            with self.assertRaisesRegex(ValueError,'non-causal'):
                load_cached([{'video':'fixture'}],{'fixture':str(path)})


if __name__=='__main__':
    unittest.main()
