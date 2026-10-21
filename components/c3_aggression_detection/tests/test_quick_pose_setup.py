import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.causal_pose import PoseConfig
from src.pose_sequences import CACHE_SCHEMA,pack_frames,save_cache
from src.quick_pose_setup import recover_entries,select_reused_pilot,validation_plan,run_bounded


class QuickSetupTests(unittest.TestCase):
    def test_plan_is_short_reproducible_grouped_and_no_test(self):
        rows=[dict(video_id=f'f_{i}',split='test' if i==0 else 'val',leakage_group=f'g{i//2}',
                   intervals='[[10,20]]' if i%2 else '[]',duration_s='30') for i in range(24)]
        a=validation_plan(rows)
        self.assertEqual(a,validation_plan(rows[::-1]))
        self.assertLessEqual(sum(x['end_s']-x['start_s'] for x in a),64)
        self.assertTrue(all(x['split']=='val' and x['video_id']!='f_0' for x in a))
        groups={r['video_id']:r['leakage_group'] for r in rows}
        self.assertEqual(len(a),len({groups[j['video_id']] for j in a}))

    def test_budget_expiry_stops_before_starting_next_job(self):
        state=[0.];called=[]
        def execute(job,remaining):
            called.append(job);state[0]+=6
            return {'status':'cached'}
        result=run_bounded([{'video_id':str(i)} for i in range(10)],execute,10,lambda:state[0])
        self.assertEqual(len(called),2)
        self.assertEqual(len(result),2)

    def test_timeout_keeps_prior_success(self):
        def execute(job,remaining):
            if job['video_id']=='b':raise subprocess.TimeoutExpired('fixture',remaining)
            return {'status':'cached'}
        result=run_bounded([{'video_id':x} for x in ('a','b','c')],execute,15)
        self.assertEqual([r['status'] for r in result],['cached','time_budget_reached'])

    def test_recover_valid_archives_and_skip_other_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for i,split in enumerate(('train','test')):
                key=f'f_{i}'
                kp=np.ones((1,17,3),np.float32)
                arrays=pack_frames([dict(t=0.,frame_index=0,keypoints=kp,
                    boxes=np.array([[0,0,10,10]],np.float32),track_ids=np.array([1]))],(20,20))
                meta={'schema':CACHE_SCHEMA,'preprocessing':PoseConfig().metadata(),'video_id':key,
                      'source_sha256':key,'source_duration_s':10.,'sample_fps':15.}
                save_cache(Path(tmp)/(key+'.npz'),arrays,meta)
                rows.append(dict(video_id=key,content_sha256=key,split=split,leakage_group=key))
            entries,metadata,skipped=recover_entries(tmp,rows)
            self.assertEqual(set(entries),{'f_0'})
            self.assertEqual(len(skipped),1)
            pilot=select_reused_pilot(rows,entries,metadata)
            self.assertEqual(set(pilot),{'f_0'})
            self.assertEqual(len(list(Path(tmp).glob('*.npz'))),2)

    def test_interrupted_archive_is_not_deleted_or_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'interrupted.npz'
            path.write_bytes(b'PK\x03\x04unfinished')
            entries,metadata,skipped=recover_entries(tmp,[])
            self.assertFalse(entries)
            self.assertEqual(len(skipped),1)
            self.assertEqual(path.read_bytes(),b'PK\x03\x04unfinished')


if __name__=='__main__':
    unittest.main()
