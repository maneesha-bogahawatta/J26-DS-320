import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.auto_annotate_precursors_v3 import CausalInteractionCues, intervals_from_observations, onset_from

def person(tid,cx,wrist_x=None):
    kp=np.zeros((17,2),float)
    kp[:]=[cx,100]
    kp[5]=[cx-12,65];kp[6]=[cx+12,65]
    kp[9]=[cx-18 if wrist_x is None else wrist_x,90]
    kp[10]=[cx+18,90]
    kp[11]=[cx-10,125];kp[12]=[cx+10,125]
    return {'id':tid,'center':np.array([cx,100.]),'height':100.,'xy':kp,'conf':np.ones(17)}

class TestV3(unittest.TestCase):
    def test_onset(self):
        self.assertAlmostEqual(onset_from({'intervals':'[[6.366667,7.2],[7.233,32.0]]'}),6.366667)
    def test_no_pair(self):
        cue=CausalInteractionCues()
        self.assertEqual(cue.update([person(1,100)],0.0)['score'],0.)
    def test_pair_approach(self):
        cue=CausalInteractionCues()
        cue.update([person(1,100),person(2,220)],0.)
        result=cue.update([person(1,120),person(2,190)],.1)
        self.assertIn('rapid_approach',result['signals'])
        self.assertGreater(result['score'],.3)
    def test_short_event_and_onset_clip(self):
        obs=[{'timestamp':6.0+i/15.,'score':.5,'quality':.9,'pair_visible':True} for i in range(3)]
        intervals=intervals_from_observations(obs,6.2,min_steps=2)
        self.assertEqual(len(intervals),1)
        self.assertLessEqual(intervals[0][1],6.2)
    def test_one_spike_rejected(self):
        obs=[{'timestamp':6.0,'score':.6,'quality':.9,'pair_visible':True},
             {'timestamp':6.067,'score':0.,'quality':.9,'pair_visible':True}]
        self.assertEqual(intervals_from_observations(obs,6.2,min_steps=2),[])
    def test_low_quality_rejected(self):
        obs=[{'timestamp':6+i/15,'score':.8,'quality':.1,'pair_visible':True} for i in range(3)]
        self.assertEqual(intervals_from_observations(obs,6.3),[])
    def test_causal(self):
        a=CausalInteractionCues(); b=CausalInteractionCues()
        p=[person(1,100),person(2,220)]
        self.assertEqual(a.update(p,0),b.update(p,0))
        r1=a.update([person(1,120),person(2,195)],.1)
        r2=b.update([person(1,120),person(2,195)],.1)
        self.assertEqual(r1,r2)
        b.update([person(1,5),person(2,250)],.2)
        self.assertEqual(r1,r2)

# Evaluation helpers integrated into the same v3 module.
from src.auto_annotate_precursors_v3 import (
    pairs, onset, overlap_seconds, build_comparison, shortlist, make_control_rows,
    read_csv, main,
)
from tempfile import TemporaryDirectory
from unittest.mock import patch
import csv
import json
import contextlib
import io

class EvaluationTests(unittest.TestCase):
    def test_pairs(self):
        self.assertEqual(pairs('[[1,2],[3,4]]'),[(1.,2.),(3.,4.)])
        self.assertEqual(pairs('[]'),[])
    def test_overlap_deduplicated(self):
        self.assertAlmostEqual(overlap_seconds([(1,3),(2,4)],1.5,3.5),2.)
    def test_onset(self):
        self.assertEqual(onset({'intervals':'[[6.4,7.2]]'}),6.4)
    def test_early_interval(self):
        a=build_comparison([{'video':'C:/v/A.mp4','v2_status':'no_proposal'}],[{
            'video':'C:/v/A.mp4','intervals':'[[10,12]]','v3_status':'proposal',
            'v3_proposals':'[[7.2,8.3]]','v3_quality':'0.9','v3_max_score':'0.8'}])
        self.assertAlmostEqual(a[0]['v3_target_1to3s_overlap_s'],1.1)
        self.assertEqual(len(shortlist(a)),1)
    def test_controls_no_fight(self):
        rows=[{'video':'no.mp4','duration_s':'12','intervals':'[]'},
              {'video':'fight.mp4','duration_s':'12','intervals':'[[2,4]]'}]
        x=make_control_rows(rows,20,42)
        self.assertEqual(len(x),1)
        self.assertGreaterEqual(x[0]['control_window_start_s'],0)
        self.assertNotIn('fight_onset_s',x[0])




class TestEvaluationCli(unittest.TestCase):
    def test_integrated_compare_does_not_run_pose_model(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            v2=root/'v2.csv'; v3=root/'v3.csv'; out=root/'out'
            with v2.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=['video','v2_status'])
                w.writeheader(); w.writerow({'video':'a/F_66_1_2_0_0.mp4','v2_status':'no_proposal'})
            with v3.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=['video','intervals','v3_status','v3_proposals','v3_quality','v3_max_score'])
                w.writeheader(); w.writerow({'video':'a/F_66_1_2_0_0.mp4',
                  'intervals':'[[6.366667,7.2]]','v3_status':'proposal',
                  'v3_proposals':'[[4.467,4.733],[4.8,5.333],[5.467,6.367]]',
                  'v3_quality':'0.8079','v3_max_score':'1.0'})
            with contextlib.redirect_stdout(io.StringIO()):
                main(['compare','--v2',str(v2),'--v3',str(v3),'--outdir',str(out)])
            self.assertTrue((out/'version_comparison.csv').exists())
            self.assertTrue((out/'review_shortlist.csv').exists())
            comp=read_csv(out/'version_comparison.csv')
            self.assertEqual(comp[0]['v3_status'],'proposal')
            self.assertEqual(comp[0]['v3_interval_count'],'3')

    def test_integrated_sample_controls_uses_unlabeled_fight_only(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=root/'all.csv'; out=root/'out'
            with manifest.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=['video','duration_s','intervals'])
                w.writeheader()
                w.writerow({'video':'not_labeled.mp4','duration_s':'20','intervals':'[]'})
                w.writerow({'video':'fight.mp4','duration_s':'20','intervals':'[[10,12]]'})
            with contextlib.redirect_stdout(io.StringIO()):
                main(['sample-controls','--manifest',str(manifest), '--number','20', '--outdir',str(out)])
            ctrls=read_csv(out/'no_fight_control_windows.csv')
            self.assertEqual(len(ctrls),1)
            self.assertEqual(ctrls[0]['video'],'not_labeled.mp4')
            self.assertAlmostEqual(float(ctrls[0]['control_window_end_s'])-float(ctrls[0]['control_window_start_s']),5.0,places=2)

    def test_integrated_control_scanner_mock_analyzer(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/'out'; out.mkdir()
            input_csv=out/'no_fight_control_windows.csv'
            with input_csv.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=['video','duration_s','control_window_start_s','control_window_end_s','annotation_intervals','label_note'])
                w.writeheader()
                w.writerow({'video':'synthetic.mp4','duration_s':'20','control_window_start_s':'7','control_window_end_s':'12','annotation_intervals':'[]','label_note':'unverified'})
            with patch('src.auto_annotate_precursors_v3.analyze',return_value=([{'score':.45,'quality':.9}], [[8.0,8.3]])) as fake:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(['scan-controls','--outdir',str(out),'--limit','1'])
            self.assertTrue(fake.called)
            self.assertEqual(read_csv(out/'no_fight_control_results.csv')[0]['v3_status'],'proposal')


if __name__ == '__main__':
    unittest.main()
