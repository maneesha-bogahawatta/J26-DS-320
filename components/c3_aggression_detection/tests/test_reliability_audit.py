import copy
import unittest

import numpy as np

from src.reliability_audit import (audit_scores, fit_temperature, reliability_metrics,
                                   temperature_scale, validate_cohort)


class ReliabilityAuditTests(unittest.TestCase):
    def fixture(self):
        rows = []
        coverage = []
        source_groups = {}
        for index, role in enumerate(('fit', 'dev', 'val')):
            key = role + '_source'
            coverage.append(dict(video_id=key, role=role))
            source_groups[key] = role + '_group'
            for j in range(3):
                rows.append(dict(model='baseline', role=role, video_id=key, index=index * 3 + j,
                    cutoff_s=j * .4, onset_y=j % 2, active_y=j % 2,
                    onset_mask=j != 2, active_mask=True,
                    onset_score=(.2, .8, 1.)[j], active_score=(.2, .8, .1)[j]))
        return rows, dict(coverage=coverage, source_groups=source_groups, test_sources_used=0)

    def test_reliability_matches_known_brier_nll_and_bins(self):
        report = reliability_metrics([0, 1], [.25, .75], bins=2, reference_prevalence=.5)
        self.assertAlmostEqual(report['brier'], .0625)
        self.assertAlmostEqual(report['nll'], -np.log(.75))
        self.assertAlmostEqual(report['ece'], .25)
        self.assertEqual(report['reference_brier'], .25)
        endpoints = reliability_metrics([0, 1], [0, 1], bins=10)
        self.assertEqual(sum(item['n'] for item in endpoints['bins']), 2)
        self.assertEqual(endpoints['bins'][0]['n'], 1)
        self.assertEqual(endpoints['bins'][-1]['n'], 1)
        self.assertEqual(endpoints['ece'], 0.)

    def test_empty_or_single_class_metrics_do_not_invent_accuracy(self):
        empty = reliability_metrics([], [])
        self.assertEqual(empty['n'], 0)
        self.assertIsNone(empty['brier'])
        self.assertEqual(empty['bins'], [])
        self.assertEqual(reliability_metrics([0, 0], [.1, .2])['positives'], 0)
        for labels, scores in (([.5], [.2]), ([1], [np.nan]), ([1], [-.1]), ([0, 1], [.2])):
            with self.assertRaises(ValueError):
                reliability_metrics(labels, scores)

    def test_temperature_positive_preserves_direction_order_and_half_cutoff(self):
        p = np.array([.01, .2, .5, .7, .99])
        for temperature in (.05, .5, 1., 20.):
            scaled = temperature_scale(p, temperature)
            self.assertTrue((np.diff(scaled) >= 0).all())
            np.testing.assert_array_equal(scaled >= .5, p >= .5)
        np.testing.assert_allclose(temperature_scale(p, 1.), p)
        for temperature in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                temperature_scale(p, temperature)

    def test_temperature_selection_rejects_validation_and_test(self):
        for role in ('fit', 'val', 'test', 'unknown'):
            with self.assertRaises(ValueError):
                fit_temperature([0, 1], [.1, .9], role=role)
        with self.assertRaises(ValueError):
            fit_temperature([1, 1], [.8, .9], role='dev')
        fit = fit_temperature([0, 1, 0, 1], [.1, .6, .9, .8], role='dev')
        original = reliability_metrics([0, 1, 0, 1], [.1, .6, .9, .8])['nll']
        self.assertLessEqual(fit['development_nll'], original)
        self.assertFalse(fit['external_validation_used_for_fitting'])

    def test_cohort_rejects_duplicate_windows_wrong_roles_and_source_leakage(self):
        rows, metadata = self.fixture()
        validate_cohort(rows, metadata)
        scenarios = []
        duplicate = copy.deepcopy(rows); duplicate.append(duplicate[0]); scenarios.append((duplicate, metadata))
        wrong = copy.deepcopy(rows); wrong[0]['role'] = 'test'; scenarios.append((wrong, metadata))
        leaked = copy.deepcopy(metadata); leaked['source_groups']['val_source'] = 'dev_group'; scenarios.append((rows, leaked))
        unknown = copy.deepcopy(rows); unknown[0]['video_id'] = 'unrecorded'; scenarios.append((unknown, metadata))
        test_data = dict(metadata, test_sources_used=1); scenarios.append((rows, test_data))
        for data, meta in scenarios:
            with self.assertRaises(ValueError):
                validate_cohort(data, meta)

    def test_masking_determinism_and_validation_cannot_select_temperature(self):
        rows, metadata = self.fixture()
        snapshot = copy.deepcopy(rows)
        first, comparison, bins = audit_scores(rows, metadata)
        second, _, _ = audit_scores(rows, metadata)
        self.assertEqual(first, second)
        self.assertEqual(rows, snapshot)
        head = first['models']['baseline']['weak_first_onset_within_3s']
        self.assertEqual(head['roles']['val']['supervised_windows'], 2)
        self.assertEqual(head['roles']['val']['masked_windows'], 1)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row['role'] == 'val':
                row['onset_y'] = 1 - row['onset_y']
                row['onset_score'] = .99
        other, _, _ = audit_scores(changed, metadata)
        self.assertEqual(head['transform'], other['models']['baseline']['weak_first_onset_within_3s']['transform'])
        self.assertFalse(first['runtime_calibration_changed'])
        self.assertFalse(first['verified_early_aggression_claim'])
        self.assertTrue(comparison)
        self.assertTrue(bins)


if __name__ == '__main__':
    unittest.main()
