import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.kinematic_model import FEATURE_DIM, KinematicTemporalModel, causal_features, frame_channels
from src.improve_model import audit_roles, features_for, fit_weights
from src.expand_training import validate_new_sources
from src.advanced_experiment import build_model, forward
from src.headwise_model import HeadwiseC3
from src.select_headwise import convex_grid
from src.improvement_report import paired_bootstrap


class KinematicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        torch.manual_seed(73)
        sk = torch.rand(2, 24, 17, 2, 3) * .6 - .3
        sk[..., 2] = .8
        mask = torch.ones(sk.shape[:-1], dtype=torch.bool)
        tm = torch.ones(sk.shape[:2], dtype=torch.bool)
        times = torch.arange(24).repeat(2, 1).float() / 8.
        return sk, tm, mask, times

    def test_person_order_translation_reflection_and_scale_invariance(self):
        sk, tm, mask, times = self.fixture()
        expected = causal_features(sk, tm, mask, times)
        self.assertEqual(expected.shape, (2, FEATURE_DIM))
        for transformed, transformed_mask in [(sk.flip(3), mask.flip(3))]:
            torch.testing.assert_close(expected, causal_features(transformed, tm, transformed_mask, times), atol=2e-5, rtol=2e-5)
        for scale, shift in ((1., .4), (2., 0.)):
            moved = sk.clone(); moved[..., :2] = moved[..., :2] * scale + shift
            torch.testing.assert_close(expected, causal_features(moved, tm, mask, times), atol=2e-5, rtol=2e-5)
        reflected = sk.clone(); reflected[..., 0] *= -1
        torch.testing.assert_close(expected, causal_features(reflected, tm, mask, times), atol=2e-5, rtol=2e-5)

    def test_padding_and_masked_nan_values_do_not_change_features(self):
        sk, tm, mask, times = self.fixture()
        mask[:, :, 2] = False
        expected = causal_features(sk, tm, mask, times)
        changed = sk.clone(); changed[~mask] = float('nan')
        pad = torch.full((2, 8, 17, 2, 3), float('nan'))
        got = causal_features(torch.cat([pad, changed], 1),
            torch.cat([torch.zeros(2, 8, dtype=torch.bool), tm], 1),
            torch.cat([torch.zeros(2, 8, 17, 2, dtype=torch.bool), mask], 1),
            torch.cat([torch.full((2, 8), float('nan')), times], 1))
        torch.testing.assert_close(expected, got)

    def test_frame_motion_has_no_future_dependency(self):
        sk, tm, mask, times = self.fixture()
        before = frame_channels(sk[:, :10], tm[:, :10], mask[:, :10], times[:, :10])
        sk[:, 10:, ..., :2] += 100
        after = frame_channels(sk, tm, mask, times)
        for a, b in zip(before, after): torch.testing.assert_close(a, b[:, :10])
        with self.assertRaises(ValueError):
            frame_channels(sk, tm, mask, times.flip(1))

    def test_empty_pose_abstains_and_model_gradients_are_finite(self):
        sk, tm, mask, times = self.fixture()
        for hidden in (0, 32):
            model = KinematicTemporalModel(hidden).eval()
            result = model(sk, tm, mask, times)
            result['logits'].sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            result = model(sk, torch.zeros_like(tm), torch.zeros_like(mask), times)
            self.assertFalse(result['valid_window'].any())
            self.assertTrue(torch.isfinite(result['logits']).all())

    def test_fit_features_ignore_all_validation_values(self):
        sk, tm, mask, times = self.fixture()
        data = dict(sk=sk.numpy(), temporal_mask=tm.numpy(), joint_mask=mask.numpy(), frame_times=times.numpy(),
            role=np.array(['fit', 'val']), targets=np.zeros((2, 2), np.float32), target_mask=np.ones((2, 2), bool))
        before, ix = features_for(data, 'fit')
        data['sk'][1] = 123.; data['targets'][1] = 1.
        after, second = features_for(data, 'fit')
        torch.testing.assert_close(before, after); np.testing.assert_array_equal(ix, second)

    def test_source_weights_balance_each_head_without_inventing_labels(self):
        y = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], np.float32)
        masks = np.array([[1, 1], [1, 0], [1, 1], [1, 1]], bool)
        w = fit_weights(y, masks, np.array(['a', 'a', 'b', 'c']), True).numpy()
        self.assertEqual(w[1, 1], 0.)
        np.testing.assert_allclose(w.sum(0), [1., 1.])
        for h in range(2):
            for label in (0, 1): self.assertAlmostEqual(w[y[:, h] == label, h].sum(), .5)

    def test_locked_test_and_wrong_role_sources_are_rejected(self):
        with patch('src.improve_model.load_locked', return_value=[dict(video_id='v', split='test')]):
            with self.assertRaises(ValueError):
                audit_roles(dict(video_id=np.array(['v']), role=np.array(['fit'])), dict(locked_manifest_path='fixture'))

    def test_expansion_preserves_group_boundary_and_original_weak_labels(self):
        row = dict(split='train', leakage_group='new', intervals='[]')
        item = dict(video_id='v', role='fit', positive=False)
        validate_new_sources([item], {'v':row}, ['old'])
        for changed in (dict(row, split='test'), dict(row, leakage_group='old'), dict(row, intervals='[[2,4]]')):
            with self.assertRaises(ValueError): validate_new_sources([item], {'v':changed}, ['old'])
        with self.assertRaises(ValueError): validate_new_sources([item,item], {'v':row}, [])
        with self.assertRaises(ValueError): validate_new_sources([dict(item,role='val')], {'v':row}, [])

    def test_model_dispatch_and_unknown_variant_rejection(self):
        sk, tm, mask, times = self.fixture()
        batch = dict(sk=sk, temporal_mask=tm, joint_mask=mask, frame_times=times)
        for variant in ('kinematic_linear', 'kinematic_mlp', 'headwise'):
            self.assertEqual(forward(build_model(variant).eval(), batch, variant).shape, (2, 2))
        with self.assertRaises(ValueError): build_model('misspelled_model')

    def test_headwise_blending_matches_one_hot_baseline_and_rejects_inversion(self):
        sk, tm, mask, times = self.fixture()
        model = HeadwiseC3().eval()
        with torch.no_grad():
            expected = model.baseline(sk, tm, mask)['logits']
            actual = model(sk, tm, mask, times)['logits']
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-5)
        model.blend_weights[0] = torch.tensor([-1., 2., 0.])
        with self.assertRaises(ValueError): model(sk, tm, mask, times)

    def test_ensemble_grid_and_source_bootstrap_preserve_pairing(self):
        grid = convex_grid()
        self.assertEqual(len(grid), 15)
        self.assertTrue(all((w >= 0).all() and np.isclose(w.sum(), 1.) for w in grid))
        result = paired_bootstrap([0, 1, 0, 1], [.1, .9, .2, .8], [.1, .9, .2, .8],
            ['a', 'a', 'b', 'b'], .5, .5, draws=40)
        self.assertEqual(result['valid_draws'], 40)
        self.assertTrue(all(bounds == [0., 0.] for bounds in result['deltas_95_percent'].values()))


if __name__ == '__main__': unittest.main()
