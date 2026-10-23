import csv
import json
import tempfile
import unittest
from pathlib import Path
from src.audit_v4_actionable_alerts import alert_events, cooldown_events, score_rows, summarize, main


def row(name='F_66_1_2_0_0.mp4', times=(4.0, 4.1, 6.2), onset=6.366667):
    return {'video': name, 'intervals': json.dumps([[onset, 7.2]]),
            'v4_alerts': json.dumps([{'alert_s': t, 'pair_ids': [i, i + 1]} for i, t in enumerate(times)])}


class Tests(unittest.TestCase):
    def test_cooldown_camera_wide(self):
        events = alert_events(row())
        self.assertEqual([e['alert_s'] for e in cooldown_events(events, 2)], [4.0, 6.2])

    def test_first_unchanged(self):
        r, _ = score_rows([row()], 'fights', 3.0)
        self.assertEqual(r[0]['raw_first_alert_s'], r[0]['actionable_first_alert_s'])
        self.assertEqual(r[0]['suppressed_notifications'], 2)

    def test_target_may_be_suppressed(self):
        r, _ = score_rows([row(times=(2.9, 5.0), onset=6.366667)], 'fights', 3)
        self.assertEqual(r[0]['raw_any_in_1to3s'], 1)
        self.assertEqual(r[0]['actionable_any_in_1to3s'], 0)

    def test_control_still_has_first_alarm(self):
        r, _ = score_rows([row(name='N_121.mp4', times=(2, 2.1, 2.2))],
                          'controls', 3, {'N_121.mp4': 'non_aggressive'})
        self.assertEqual(r[0]['actionable_alert_count'], 1)
        self.assertEqual(r[0]['review_label'], 'non_aggressive')
        self.assertEqual(summarize(r)['windows_with_alarms'], 1)

    def test_no_alarm(self):
        r, _ = score_rows([row(times=())], 'fights', 3)
        self.assertEqual(r[0]['first_alert_band'], 'no_alert')

    def test_windows_video_basename(self):
        r, _ = score_rows([row(name=r'C:\folder\F_66.mp4')], 'fights', 3)
        self.assertEqual(r[0]['video'], 'F_66.mp4')

    def test_bad_event_reported(self):
        r = row()
        r['v4_alerts'] = json.dumps([{'pair_ids': [1, 2]}])
        valid, errors = score_rows([r], 'fights', 3)
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(errors), 1)

    def test_cli_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for key, records in [('fights', [row()]), ('controls', [row(name='N_1.mp4', times=(1, 1.1))])]:
                path = d / f'{key}.csv'
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['video', 'intervals', 'v4_alerts'])
                    w.writeheader(); w.writerows(records)
            ret = main(['--fight-csv', str(d / 'fights.csv'),
                        '--control-csv', str(d / 'controls.csv'),
                        '--label-csv', str(d / 'absent.csv'),
                        '--out-dir', str(d / 'out'), '--cooldown-s', '3'])
            self.assertEqual(ret, 0)
            self.assertTrue((d / 'out' / 'summary.json').exists())
            summary = json.loads((d / 'out' / 'summary.json').read_text())
            self.assertEqual(summary['fight_candidates']['raw_alerts'], 3)
            self.assertEqual(summary['fight_candidates']['after_cooldown_alerts'], 1)

if __name__ == '__main__':
    unittest.main()
