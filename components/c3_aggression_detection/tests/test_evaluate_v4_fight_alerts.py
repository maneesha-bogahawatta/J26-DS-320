import json
import unittest
from src.evaluate_v4_fight_alerts import (classify_first_lead, evaluate_row,
                                        onset_from_row, summarize, densest_one_second)


class LeadTimeTests(unittest.TestCase):
    def sample(self, timestamps, onset=6.366667):
        return {'video': r'C:\dataset\F_66_1_2_0_0.mp4',
                'intervals': json.dumps([[onset, 7.2], [7.233333, 32]]),
                'v4_status': 'proposal' if timestamps else 'no_proposal',
                'v4_alerts': json.dumps([{'alert_s': t,
                                          'evidence_start_s': t - 0.067,
                                          'pair_ids': [1, 3]} for t in timestamps])}

    def test_onset(self):
        self.assertAlmostEqual(onset_from_row(self.sample([])), 6.366667)

    def test_f66_late(self):
        r = evaluate_row(self.sample([6.2]))
        self.assertEqual(r['first_alert_category'], 'less_than_1s_early')
        self.assertAlmostEqual(r['first_alert_lead_s'], .166667)
        self.assertFalse(r['any_alert_in_1to3s'])

    def test_first_early_later_target(self):
        r = evaluate_row(self.sample([2.8, 4.5, 6.2]))
        self.assertEqual(r['first_alert_category'], 'more_than_3s_early')
        self.assertTrue(r['any_alert_in_1to3s'])
        self.assertEqual(r['target_alert_count'], 1)

    def test_first_in_target(self):
        r = evaluate_row(self.sample([4.2, 6.2]))
        self.assertEqual(r['first_alert_category'], '1_to_3s_early')
        self.assertEqual(r['first_target_alert_s'], 4.2)

    def test_no_alert(self):
        r = evaluate_row(self.sample([]))
        self.assertEqual(r['first_alert_category'], 'no_alert')
        self.assertEqual(r['alert_count'], 0)

    def test_burst(self):
        alerts = [{'t': t, 'pair': p} for t, p in [(1,(1,2)),(1.1,(1,2)),(1.2,(3,4)),(2.4,(5,6))]]
        self.assertEqual(densest_one_second(alerts), (3,2))

    def test_summary(self):
        results = [evaluate_row(self.sample([4.2])), evaluate_row(self.sample([6.2])), evaluate_row(self.sample([]))]
        s = summarize(results)
        self.assertEqual(s['videos'], 3)
        self.assertEqual(s['first_alert_in_1to3s'], 1)
        self.assertEqual(s['first_alert_less_than_1s_early'], 1)
        self.assertEqual(s['no_alert'], 1)

    def test_malformed_alerts_rejected(self):
        row = self.sample([4.2]); row['v4_alerts'] = '[{}]'
        with self.assertRaises(ValueError):
            evaluate_row(row)

    def test_post_onset(self):
        self.assertEqual(classify_first_lead(-0.1), 'post_onset')


if __name__ == '__main__':
    unittest.main()
