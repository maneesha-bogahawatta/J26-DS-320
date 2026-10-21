import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.diagnose_stgcn import dataset_audit, independent_targets, ranking, tune_threshold
from src.stgcn_loss_ablation import fit_class_weights, masked_bce
from src.stgcn_quality_windows import guard_subset
from src.build_stgcn_diagnosis_report import paired_source_bootstrap


class DiagnosticTests(unittest.TestCase):
    def test_paired_source_bootstrap_preserves_identical_model_differences(self):
        report = paired_source_bootstrap([0,1,0,1],[.2,.8,.1,.9],[.2,.8,.1,.9],
                                        ['a','a','b','b'], repetitions=30)
        self.assertEqual(report['valid_repetitions'], 30)
        self.assertEqual(report['source_groups'], 2)
        for interval in report['intervals'].values():self.assertEqual(interval, [0.,0.])
        with self.assertRaises(ValueError): paired_source_bootstrap([0,1],[.2,.8],[.2,.8],['a'])

    def test_current_guard_keeps_original_labels_masks_and_indices(self):
        data = dict(role=np.array(['fit','dev','val']), targets=np.array([[1,0],[0,1],[1,1]]),
                    target_mask=np.array([[1,1],[1,0],[0,1]],bool))
        result, keep = guard_subset(data, [17,0,6])
        np.testing.assert_array_equal(keep, [True,False,True])
        np.testing.assert_array_equal(result['targets'], data['targets'][[0,2]])
        np.testing.assert_array_equal(result['target_mask'], data['target_mask'][[0,2]])
        np.testing.assert_array_equal(result['original_window_index'], [0,2])
        result['targets'][:] = 0
        self.assertEqual(data['targets'][0,0], 1)
        with self.assertRaises(ValueError): guard_subset(data, [1,np.nan,2])
        with self.assertRaises(ValueError): guard_subset(data, [1,2])

    def test_positive_logit_direction_and_masked_gradient(self):
        logits = torch.zeros(3, 2, requires_grad=True)
        targets = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        mask = torch.tensor([[True, True], [True, True], [False, False]])
        loss = masked_bce(logits, targets, mask, torch.ones(2))
        loss.backward()
        self.assertLess(logits.grad[0, 0], 0)
        self.assertGreater(logits.grad[0, 1], 0)
        self.assertTrue(torch.equal(logits.grad[2], torch.zeros(2)))
        new_probabilities = (logits.detach() - logits.grad).sigmoid()
        self.assertGreater(new_probabilities[0, 0], .5)
        self.assertLess(new_probabilities[0, 1], .5)

    def test_empty_supervision_is_zero_without_manufactured_negatives(self):
        logits = torch.zeros(3, 2, requires_grad=True)
        loss = masked_bce(logits, torch.ones_like(logits), torch.zeros_like(logits, dtype=torch.bool), torch.ones(2))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_class_weight_increases_positive_loss_only(self):
        logits = torch.zeros(2, 2)
        target = torch.tensor([[1., 0.], [0., 1.]])
        mask = torch.tensor([[True, False], [False, False]])
        self.assertAlmostEqual(float(masked_bce(logits, target, mask, torch.tensor([3., 1.]))),
                               3 * float(masked_bce(logits, target, mask, torch.ones(2))), places=6)

    def test_class_weights_ignore_development_validation_and_masked_targets(self):
        data = dict(role=np.array(['fit','fit','fit','dev','val']),
            targets=np.array([[1.,0.],[0.,1.],[1.,1.],[1.,1.],[1.,1.]],np.float32),
            target_mask=np.array([[1,1],[1,1],[0,1],[1,1],[1,1]],bool))
        expected = fit_class_weights(data)
        torch.testing.assert_close(expected, torch.tensor([1., .5]))
        data['targets'][3:] = 0
        data['targets'][2,0] = 0
        torch.testing.assert_close(expected, fit_class_weights(data))

    def test_independent_label_contract(self):
        self.assertEqual(independent_targets(8., [[10.,12.]]), ([1,0], [True,True]))
        self.assertEqual(independent_targets(5., [[10.,12.]]), ([0,0], [True,True]))
        self.assertEqual(independent_targets(11., [[10.,12.]]), ([0,1], [False,True]))
        self.assertEqual(independent_targets(7., [[10.,12.]])[1][0], False)
        self.assertEqual(independent_targets(1., []), ([0,0], [True,True]))

    def test_auroc_complement_is_diagnostic_not_automatic_correction(self):
        y = np.array([0,0,1,1]); p = np.array([.9,.7,.6,.2])
        a, b = ranking(y,p), ranking(y,1-p)
        self.assertEqual(a['auroc'], 0.)
        self.assertEqual(b['auroc'], 1.)
        self.assertEqual(a['independent_pairwise_auroc'], a['auroc'])
        self.assertEqual(a['auroc'] + b['auroc'], 1.)

    def test_ap_and_trapezoidal_pr_area_are_explicit_and_tie_correct(self):
        report = ranking([0,1,0,1], [.5,.5,.5,.5])
        self.assertEqual(report['average_precision'], .5)
        self.assertEqual(report['pr_auc_trapezoid'], .75)
        self.assertEqual(report['auroc'], .5)
        with self.assertRaises(ValueError): ranking([0,2], [.1,.9])
        with self.assertRaises(ValueError): ranking([0,1], [.1,np.nan])

    def test_threshold_f1_and_false_positive_cap(self):
        y = [0,0,0,1,1]
        p = [.1,.2,.8,.6,.7]
        report = tune_threshold(y,p)
        self.assertEqual(report['threshold'], .6)
        self.assertEqual(report['recall'], 1.)
        self.assertEqual(report['tp'], 2)
        capped = tune_threshold(y,p,max_fpr=0)
        self.assertEqual(capped['fp'], 0)
        self.assertEqual(capped['recall'], 0)
        self.assertEqual(tune_threshold([0,1],[.5,.5])['recall'], 1.)
        with self.assertRaises(ValueError): tune_threshold([1,1],[.4,.5])

    def test_audit_rejects_test_before_annotation_or_video_access(self):
        data = dict(video_id=np.array(['test_source']), cutoff_s=np.array([8.]), role=np.array(['fit']))
        with patch('src.diagnose_stgcn.load_locked', return_value=[dict(video_id='test_source',split='test')]):
            with self.assertRaisesRegex(ValueError, 'Test or wrong-split'):
                dataset_audit(data, dict(locked_manifest_path='unused'))


if __name__ == '__main__':
    unittest.main()
