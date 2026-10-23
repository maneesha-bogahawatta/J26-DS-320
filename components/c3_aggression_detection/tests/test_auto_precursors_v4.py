"""Unit tests. Synthetic signals, not a real-world performance benchmark."""
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.auto_annotate_precursors_v4 import (
    CausalAlertGate, CausalPoseV4, SceneChangeGuard, person_from_pose,
    interaction_evidence, fight_onset,
)


def person(tid, cx, wrist_shift=0., q=1.):
    xy = np.tile(np.array([cx,100.]), (17,1))
    xy[5]=[cx-12,75];xy[6]=[cx+12,75]
    xy[11]=[cx-10,125];xy[12]=[cx+10,125]
    xy[9]=[cx-15+wrist_shift,95]
    xy[10]=[cx+15+wrist_shift,95]
    return {'id':tid,'center':np.array([cx,100.]),'height':100.,'xy':xy,
            'conf':np.ones(17)*q}


def cue(approach=True, arm=True, q=.9, score=.9):
    return {'pair_ids':[2,1], 'quality':q, 'approach':approach,
            'arm_intent':arm,'score':score}


class TestV4(unittest.TestCase):
    def test_onset_for_offline_only(self):
        self.assertAlmostEqual(fight_onset({'intervals':'[[6.366667,7.2],[7.23,32]]'}),6.366667)

    def test_single_spike_rejected(self):
        g=CausalAlertGate()
        self.assertEqual(g.update([cue()],0),[])
        self.assertEqual(g.update([cue(arm=False)],.067),[])
        self.assertEqual(len(g.alerts),0)

    def test_alert_emission_is_causal_and_not_backdated(self):
        g=CausalAlertGate()
        g.update([cue()],1.0)
        g.update([cue()],1.067)
        events=g.update([cue()],1.134)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['alert_s'],1.134)
        self.assertEqual(events[0]['evidence_start_s'],1.)

    def test_low_quality_rejected(self):
        g=CausalAlertGate()
        for i in range(7):
            self.assertFalse(g.update([cue(q=.44)],i/15))

    def test_pair_switch_no_shared_evidence(self):
        g=CausalAlertGate()
        for i in range(2): g.update([cue()],i/15)
        other=cue();other['pair_ids']=[2,3]
        self.assertEqual(g.update([other],2/15),[])

    def test_no_arm_motion_no_alarm_despite_approach(self):
        g=CausalAlertGate()
        for i in range(15):
            self.assertEqual(g.update([cue(arm=False)],i/15),[])

    def test_scene_cut_resets_evidence(self):
        g=CausalAlertGate()
        g.update([cue()],0)
        g.update([cue()],.067)
        g.update([], .134, scene_cut=True)
        self.assertEqual(g.update([cue()],.201),[])

    def test_cooldown_no_retrigger(self):
        g=CausalAlertGate()
        for i in range(7): g.update([cue()],i/15)
        self.assertEqual(len(g.alerts),1)

    def test_pose_low_confidence(self):
        p=person_from_pose(person(1,120,q=.2))
        self.assertEqual(p['wrists'],[])
        self.assertIsNone(p['torso'])

    def test_side_by_side_without_arm_extension(self):
        a=person_from_pose(person(1,100))
        b=person_from_pose(person(2,180))
        prior_a=person_from_pose(person(1,90))
        prior_b=person_from_pose(person(2,195))
        ev=interaction_evidence(a,b,prior_a,prior_b,.067)
        self.assertFalse(ev['arm_intent'])
        self.assertTrue(ev['approach'])

    def test_frame_gap_discards_old_pair(self):
        detector=CausalPoseV4()
        detector.update([person(1,100),person(2,180)],0)
        detector.update([person(1,105),person(2,175)],2.0)
        self.assertEqual(detector.gate.alerts,[])

    def test_synthetic_cut_detected(self):
        s=SceneChangeGuard()
        self.assertFalse(s.update(np.zeros((80,100,3),dtype=np.uint8)))
        self.assertTrue(s.update(np.full((80,100,3),255,dtype=np.uint8)))
        self.assertEqual(s.cuts,1)

    def test_evidence_hand_reaches_opponent(self):
        # Move one wrist toward the opponent's torso while close and approaching.
        old_a=person_from_pose(person(1,100,wrist_shift=0))
        old_b=person_from_pose(person(2,175))
        a=person_from_pose(person(1,110,wrist_shift=30))
        b=person_from_pose(person(2,167))
        ev=interaction_evidence(a,b,old_a,old_b,.067)
        self.assertIsNotNone(ev)
        self.assertTrue(ev['arm_intent'])


class TestComparison(unittest.TestCase):
    def test_control_comparison(self):
        from src.auto_annotate_precursors_v4 import compare_results, summarize_comparison
        v3=[{'video':'C:/x/N_A.mp4','v3_status':'proposal',
             'control_window_start_s':'2.0','control_window_end_s':'7.0'}]
        v4=[{'video':'D:/y/N_A.mp4','v4_status':'no_proposal', 'v4_alert_count':'0',
             'control_window_start_s':'2.0','control_window_end_s':'7.0'}]
        labels=[{'video':'N_A.mp4','review_label':'non_aggressive'}]
        comp=compare_results(v3,v4,labels,kind='controls')
        stats=summarize_comparison(comp,'controls')
        self.assertEqual(stats['v3_alarms_in_reviewed'],1)
        self.assertEqual(stats['v4_alarms_in_reviewed'],0)
        self.assertEqual(stats['reviewed_non_aggressive'],1)

    def test_mismatched_windows_rejected(self):
        from src.auto_annotate_precursors_v4 import compare_results
        a=[{'video':'N_X.mp4','control_window_start_s':'2','control_window_end_s':'7'}]
        b=[{'video':'N_X.mp4','control_window_start_s':'3','control_window_end_s':'8'}]
        with self.assertRaises(ValueError): compare_results(a,b,kind='controls')

class TestRowScanning(unittest.TestCase):
    def test_retrospective_onset_not_passed_to_model(self):
        from src.auto_annotate_precursors_v4 import scan_rows
        from types import SimpleNamespace
        from unittest.mock import patch
        row={'video':'C:/dataset/F_66_1_2_0_0.mp4','intervals':'[[6.366667,7.2]]'}
        args=SimpleNamespace(pre_seconds=5, weights='dummy.pt',sample_fps=15,
                             device=None,quality_min=.55,min_arm_hits=2,min_pair_frames=3)
        fake={'v4_alerts':[{'alert_s':4.7,'pair_ids':[1,2]}],
              'sampled_frames':72,'scene_cuts':0,'pair_samples':55,'v4_max_quality':.9}
        with patch('src.auto_annotate_precursors_v4.scan_video',return_value=fake) as spy:
            rows=scan_rows([row],'fights',args)
        self.assertEqual(rows[0]['v4_1to3s_alert'],1)
        self.assertEqual(rows[0]['v4_status'],'proposal')
        kwargs=spy.call_args.kwargs
        self.assertNotIn('onset', kwargs)
        self.assertAlmostEqual(kwargs['start_s'],1.366667,places=5)
        self.assertAlmostEqual(kwargs['end_s'],6.366667,places=5)

    def test_controls_no_annotation_used(self):
        from src.auto_annotate_precursors_v4 import scan_rows
        from types import SimpleNamespace
        from unittest.mock import patch
        args=SimpleNamespace(weights='dummy',sample_fps=15,device=None,
                             quality_min=.55,min_arm_hits=2,min_pair_frames=3)
        row={'video':'N_01.mp4','control_window_start_s':'3','control_window_end_s':'8'}
        fake={'v4_alerts':[],'sampled_frames':75,'scene_cuts':0,
              'pair_samples':20,'v4_max_quality':.8}
        with patch('src.auto_annotate_precursors_v4.scan_video',return_value=fake) as spy:
            rows=scan_rows([row],'controls',args)
        self.assertEqual(rows[0]['v4_status'],'no_proposal')
        self.assertEqual(rows[0]['v4_1to3s_alert'],'')
        self.assertEqual(spy.call_args.kwargs['start_s'],3.)
        self.assertEqual(spy.call_args.kwargs['end_s'],8.)

if __name__ == "__main__":
    unittest.main()
