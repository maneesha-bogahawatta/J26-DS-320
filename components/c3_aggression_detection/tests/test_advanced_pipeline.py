import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from src.advanced_data import weak_targets, internal_partition, choose_candidates
from src.advanced_stgcn import AdvancedSTGCN, pose_streams
from src.advanced_experiment import metrics
from src.predict_advanced import CausalDecision, at_or_before
from src.predict_advanced import score_cache
from src.research_operating_point import select_point
from src.advanced_experiment import SCHEMA
from src.causal_pose import PoseConfig
from src.pose_sequences import CACHE_SCHEMA, pack_frames, save_cache


class AdvancedDataTests(unittest.TestCase):
    def test_operating_point_controls_development_false_positive_rate(self):
        report = select_point([0,0,0,0,0,0,0,0,0,0,1,1], [.1]*10+[.8,.9])
        self.assertLessEqual(report['development_fpr'], .1)
        self.assertTrue(report['research_alert_eligible'])
        tied = select_point([0,0,1,1], [.5]*4)
        self.assertFalse(tied['research_alert_eligible'])
        self.assertEqual(tied['metrics']['recall'], 0.)
    def test_decisions_abstain_reset_and_never_use_future_scores(self):
        decision = CausalDecision([.5, .5])
        self.assertFalse(decision.update(0., [.9,.1], True, 0)[2])
        decision.update(.2, [.9,.1], True, 0)
        self.assertEqual(decision.update(.4, [.9,.1], True, 0)[0], 'WEAK_ONSET_WITHIN_3S_ALERT')
        self.assertEqual(decision.update(.6, None, False, 0)[0], 'ABSTAIN_LOW_POSE_QUALITY')
        self.assertEqual(len(decision.history), 0)
        self.assertEqual(decision.update(.8, [.1,.9], True, 1)[0], 'BELOW_SELECTED_THRESHOLDS')
        with self.assertRaises(ValueError): decision.update(.7, [.1,.9], True, 1)
        rows = [dict(t_s=1., score=.1), dict(t_s=2., score=.9)]
        self.assertEqual(at_or_before(rows, 1.5)['score'], .1)
        self.assertIsNone(at_or_before(rows, .5))
        self.assertEqual(at_or_before(rows+[dict(t_s=3.,score=1.)], 1.5), at_or_before(rows, 1.5))

    def test_ranking_metrics_handle_ties_and_single_class(self):
        perfect = metrics([0,1,0,1], [.1,.9,.2,.8])
        self.assertEqual(perfect['auroc'], 1.)
        self.assertEqual(perfect['average_precision'], 1.)
        tied = metrics([0,1,0,1], [.5,.5,.5,.5])
        self.assertEqual(tied['auroc'], .5)
        self.assertEqual(tied['average_precision'], .5)
        self.assertIsNone(metrics([0,0], [.1,.3])['auroc'])
        self.assertIsNone(metrics([0,0], [.1,.3])['balanced_accuracy'])
        with self.assertRaises(ValueError): metrics([0,1], [np.nan,.3])
    def test_weak_forecast_explicit_horizon_and_post_onset_mask(self):
        self.assertEqual(weak_targets(8, [[10, 12]])[0].tolist(), [1, 0])
        self.assertTrue(weak_targets(8, [[10, 12]])[1][0])
        self.assertEqual(weak_targets(5, [[10, 12]])[0].tolist(), [0, 0])
        self.assertFalse(weak_targets(11, [[10, 12]])[1][0])
        self.assertEqual(weak_targets(11, [[10, 12]])[0][1], 1)
        self.assertFalse(weak_targets(10, [[10, 12]])[1].any())
        np.testing.assert_array_equal(weak_targets(8, [])[1], [True, True])

    def test_internal_dev_is_grouped_train_only(self):
        rows = [dict(video_id=f'v{i}', split='train', leakage_group=f'g{i//2}', intervals='[[10,12]]') for i in range(20)]
        roles = internal_partition(rows, {r['video_id']: {} for r in rows})
        self.assertEqual(roles, internal_partition(rows, {r['video_id']: {} for r in rows}))
        for i in range(0, 20, 2): self.assertEqual(roles[f'v{i}'], roles[f'v{i+1}'])
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError): internal_partition(rows, {'v0': {}})

    def test_candidate_selection_never_crosses_locked_role(self):
        rows = [dict(video_id=str(i), split=s, leakage_group=str(i), intervals='[]', duration_s='10') for i,s in enumerate(('train','val','test'))]
        self.assertEqual([r['video_id'] for r in choose_candidates(rows, 'train', False, set())], ['0'])


class AdvancedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def fixture(self):
        torch.manual_seed(73)
        sk = torch.rand(2, 16, 17, 2, 3)*.4-.2
        sk[..., 2] = .8
        mask = torch.ones(sk.shape[:-1], dtype=torch.bool)
        times = torch.arange(16).repeat(2, 1)/15
        return sk, mask, times

    def test_person_permutation_invariance_and_finite_gradient(self):
        x, m, t = self.fixture(); net = AdvancedSTGCN().eval()
        a = net(x, torch.ones(x.shape[:2], dtype=torch.bool), m, t)['logits']
        b = net(x.flip(3), torch.ones(x.shape[:2], dtype=torch.bool), m.flip(3), t)['logits']
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        a.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
        self.assertFalse(any(isinstance(n, torch.nn.BatchNorm2d) for n in net.modules()))

    def test_padding_and_hidden_values_cannot_affect_output(self):
        x, m, t = self.fixture(); net = AdvancedSTGCN().eval()
        pad = torch.cat([torch.randn(2, 8, 17, 2, 3)*100, x], 1)
        tm = torch.cat([torch.zeros(2, 8, dtype=torch.bool), torch.ones(2, 16, dtype=torch.bool)], 1)
        masks = torch.cat([torch.zeros(2, 8, 17, 2, dtype=torch.bool), m], 1)
        times = torch.cat([torch.full((2, 8), -1), t], 1)
        with torch.no_grad():
            a = net(x, torch.ones(x.shape[:2], dtype=torch.bool), m, t)['logits']
            b = net(pad, tm, masks, times)['logits']
        torch.testing.assert_close(a, b)

    def test_frame_features_are_past_only_and_translation_invariant(self):
        x, m, t = self.fixture()
        prefix = pose_streams(x[:, :8], m[:, :8], t[:, :8])[0]
        changed = x.clone(); changed[:, 8:, ..., :2] += 100
        whole = pose_streams(changed, m, t)[0]
        for a,b in zip(prefix, whole): torch.testing.assert_close(a, b[:, :8])
        translated = x.clone(); translated[..., :2] += .2
        for a,b in zip(pose_streams(x, m, t)[0], pose_streams(translated, m, t)[0]):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_all_missing_input_abstains(self):
        x, m, t = self.fixture(); m[:] = False
        out = AdvancedSTGCN().eval()(x, torch.ones(x.shape[:2], dtype=torch.bool), m, t)
        self.assertFalse(out['valid_window'].any())
        self.assertTrue(torch.isfinite(out['logits']).all())

    def test_end_to_end_cached_replay_is_prefix_invariant_and_score_only(self):
        frames = []
        for i in range(20):
            kp = np.ones((2,17,3), np.float32); kp[..., :2] *= 30+i
            if i > 14: kp[..., :2] += 100
            frames.append(dict(t=i/10, frame_index=i, keypoints=kp,
                boxes=np.array([[0,0,60,90],[20,0,90,95]],np.float32),track_ids=np.array([1,2])))
        frames[10] = dict(t=1., frame_index=10, keypoints=np.zeros((0,17,3),np.float32),
            boxes=np.zeros((0,4),np.float32),track_ids=np.zeros(0,np.int64))
        config = dict(schema=CACHE_SCHEMA, preprocessing=PoseConfig().metadata(),sample_fps=10.,
            video_id='fixture', source_sha256='fixture')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); checkpoint = root/'fixture.pt'
            torch.save(dict(schema=SCHEMA, variant='advanced', state=AdvancedSTGCN().state_dict(),
                feature_contract={k:config[k] for k in ('schema','preprocessing','sample_fps')},
                thresholds=[.5,.5], verified_early_aggression_claim=False), checkpoint)
            a = root/'prefix.npz'; b = root/'whole.npz'
            save_cache(a, pack_frames(frames[:12], (100,100)), config)
            save_cache(b, pack_frames(frames, (100,100)), config)
            prefix = score_cache(checkpoint, a)[0]; full = score_cache(checkpoint, b)[0]
            self.assertEqual(prefix, full[:len(prefix)])
            self.assertFalse(any(row['alert_emitted'] for row in full))
            self.assertTrue(all(row['status'] in ('RESEARCH_SCORES_ONLY','ABSTAIN_LOW_POSE_QUALITY') for row in full))
            absent = [row for row in full if abs(row['t_s']-1.) < 1e-8][0]
            self.assertEqual(absent['status'], 'ABSTAIN_LOW_POSE_QUALITY')
            self.assertEqual(absent['current_observed_joints'], 0)


if __name__ == '__main__': unittest.main()
