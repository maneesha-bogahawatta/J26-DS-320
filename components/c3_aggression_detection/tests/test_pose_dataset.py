import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.causal_pose import PoseConfig
from src.foundation_io import sha256, write_csv, write_json
from src.lock_splits import assign
from src.pose_dataset import PoseSequenceDataset, collate_sequences, labels_at, sequence_window
from src.pose_sequences import CACHE_SCHEMA, load_cache, pack_frames, save_cache
from src.reconcile_reviews import normalize
from src.stgcn_baseline import JointOnlySTGCN


def fixture_frames(n=20):
    out=[]
    for i in range(n):
        kp=np.ones((2,17,3),np.float32)
        kp[0,:,:2]=[30+i,40];kp[1,:,:2]=[70-i,40]
        out.append(dict(t=i/10,frame_index=i,keypoints=kp,
                        boxes=np.array([[10,10,50,90],[50,10,90,90]],np.float32),track_ids=np.array([8,3]),scene_cut=i==12))
    return out


class PoseDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cache_roundtrip_masks_and_no_overwrite(self):
        arrays=pack_frames(fixture_frames(),(100,100))
        meta={'schema':CACHE_SCHEMA,'preprocessing':PoseConfig().metadata()}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'cache.npz'
            save_cache(path,arrays,meta)
            got,saved=load_cache(path)
            self.assertEqual(saved,meta)
            np.testing.assert_array_equal(got['keypoints'],arrays['keypoints'])
            with self.assertRaises(FileExistsError):save_cache(path,arrays,meta)

    def test_cache_prefix_invariant_and_cut_boundary(self):
        frames=fixture_frames()
        a=pack_frames(frames[:10],(100,100))
        frames[15]['keypoints'][:,:,:2]+=100
        b=pack_frames(frames,(100,100))
        for key in ('keypoints','observed','imputed','quality'):
            np.testing.assert_array_equal(a[key],b[key][:20])
        before=sequence_window(b,9,48)
        np.testing.assert_array_equal(before['sk'],sequence_window(a,9,48)['sk'])
        after=sequence_window(b,13,48)
        self.assertEqual(after['temporal_mask'].sum(),2)
        self.assertTrue((after['frame_times'][after['temporal_mask']]<=after['cutoff_s']).all())
        self.assertEqual(set(after['selected_ids']),{3,8})

    def test_unknown_scope_and_conflicts_masked(self):
        base=dict(video_id='f_fixture',split='train',duration_s='20')
        review=dict(review_label='verified_precursor',clip_start_s='0',clip_end_s='2',
                    precursor_start_s='.5',precursor_end_s='1.4',first_physical_contact_s='1.5',
                    reviewer_1='A',reviewer_2='B')
        r=normalize(review,base,'fixture','v4')
        target,mask=labels_at(1,[r]);self.assertTrue(mask[0]);self.assertEqual(target[0],1)
        self.assertFalse(labels_at(4,[r])[1].any())
        negative=dict(r,label='non_aggressive')
        self.assertFalse(labels_at(1,[r,negative])[1][0])
        weak,mask=labels_at(1,[],[[.5,2]])
        self.assertEqual(weak.tolist(),[0,1]);self.assertEqual(mask.tolist(),[False,True])

    def make_dataset(self,temp,split='train'):
        temp=Path(temp)
        locked=temp/'ubi_locked.csv'
        rows=assign([dict(video='F_fixture.mp4',group='fixture',intervals='[[1,2]]',
                          duration_s='2',split=split,content_sha256='fixture_hash')],set())
        write_csv(locked,rows)
        write_json(temp/'split_metadata.json',dict(locked_manifest_sha256=sha256(locked),reserved_test_ids=[]))
        arrays=pack_frames(fixture_frames(),(100,100))
        cache=temp/'cache.npz'
        save_cache(cache,arrays,dict(schema=CACHE_SCHEMA,preprocessing=PoseConfig().metadata(),
                                   video_id='f_fixture',source_sha256='fixture_hash'))
        write_json(temp/'index.json',dict(schema=CACHE_SCHEMA,locked_manifest_sha256=sha256(locked),
            entries={'f_fixture':dict(path=str(cache),cache_sha256=sha256(cache))}))
        write_json(temp/'annotations.json',dict(schema='c3_review_evidence_v1',locked_manifest_sha256=sha256(locked),records=[]))
        return locked,temp/'annotations.json',temp/'index.json'

    def test_loading_batch_model_forward_and_gradient(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.make_dataset(temp)
            ds=PoseSequenceDataset(*args,min_frames=2)
            self.assertTrue(len(ds))
            self.assertFalse(ds[3]['target_mask'].any())
            batch=collate_sequences([ds[3],ds[4]])
            self.assertEqual(batch['sk'].shape,(2,48,17,2,3))
            net=JointOnlySTGCN()
            out=net(batch['sk'],batch['temporal_mask'],batch['joint_mask'])
            self.assertEqual(out['logits'].shape,(2,2))
            out['logits'].square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
            weak=PoseSequenceDataset(*args,min_frames=2,supervision='weak-fight')
            self.assertFalse(weak[6]['target_mask'][0])

    def test_test_guard_and_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.make_dataset(temp,'test')
            with self.assertRaisesRegex(ValueError,'evaluation'):
                PoseSequenceDataset(*args,split='test')
            ds=PoseSequenceDataset(*args,split='test',purpose='evaluation')
            self.assertTrue(len(ds))
            args[0].write_text(args[0].read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'changed'):
                PoseSequenceDataset(*args,split='test',purpose='evaluation')

    def test_masked_padding_values_do_not_change_output(self):
        net=JointOnlySTGCN().eval()
        x=torch.randn(1,8,17,2,3)
        padded=torch.cat([torch.randn(1,12,17,2,3)*100,x],1)
        mask=torch.ones((1,8),dtype=torch.bool)
        padded_mask=torch.cat([torch.zeros((1,12),dtype=torch.bool),mask],1)
        with torch.no_grad():
            a=net(x,mask,torch.ones((1,8,17,2),dtype=torch.bool))['logits']
            b=net(padded,padded_mask,torch.ones((1,20,17,2),dtype=torch.bool))['logits']
        torch.testing.assert_close(a,b)


if __name__=='__main__':
    unittest.main()
