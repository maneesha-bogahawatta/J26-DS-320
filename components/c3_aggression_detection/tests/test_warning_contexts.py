import unittest
from src.evaluate_warning_contexts import onset_context, summarize_contexts


class WarningContextTests(unittest.TestCase):
    def row(self, t, valid=True, alert=False, candidate=True):
        return dict(t_s=t, valid_pose=valid, alert_emitted=alert,
            candidate_status='WEAK_ONSET_WITHIN_3S_ALERT' if candidate else 'BELOW_SELECTED_THRESHOLDS')

    def test_lead_metrics_separate_candidates_notifications_and_post_contact(self):
        event = onset_context(10., [self.row(7.), self.row(8.), self.row(9., alert=True), self.row(10., alert=True)], 6., 12.)
        self.assertEqual(event['candidate_recall_at_lead'], {'1':True,'2':True,'3':True})
        self.assertEqual(event['notification_recall_at_lead'], {'1':True,'2':False,'3':False})
        self.assertEqual(event['candidate_lead_s'], 3.)
        self.assertEqual(event['notified_lead_s'], 1.)
        self.assertEqual(event['candidate_to_notification_lead_difference_s'], 2.)
        self.assertFalse(event['onset_is_verified_contact'])

    def test_missing_pose_remains_undetected_in_denominator(self):
        missing = onset_context(10., [self.row(8., valid=False, alert=True)], 7., 11.)
        hit = onset_context(10., [self.row(8., alert=True)], 7., 11.)
        report = summarize_contexts([missing, hit])
        self.assertEqual(report['contexts'], 2)
        self.assertEqual(report['notification_recall_at_lead']['2'], .5)
        self.assertFalse(missing['notification_detected'])

    def test_empty_uncovered_and_only_late_contexts_are_explicit(self):
        self.assertIsNone(onset_context(10., [], 0., 5.))
        self.assertIsNone(summarize_contexts([])['notification_recall_at_lead']['1'])
        late = onset_context(10., [self.row(10., alert=True)], 9., 11.)
        self.assertFalse(late['notification_detected'])
        self.assertFalse(late['full_three_second_preonset_context'])


if __name__ == '__main__': unittest.main()
