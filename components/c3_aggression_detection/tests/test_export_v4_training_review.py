import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.export_v4_training_review import (
    basename, choose_balanced, clip_bounds, run, video_index,
)


def sample_rows():
    categories = ['1_to_3s_early']*6 + ['more_than_3s_early']*6 + ['less_than_1s_early']*4 + ['no_alert']*4
    return [{
        'video': f'F_{i}.mp4', 'fight_onset_s': '10.0',
        'first_alert_category': c, 'first_alert_s': '8' if c != 'no_alert' else '',
        'max_distinct_pairs_in_1s': '6' if i in [5, 11, 15, 19] else '0',
    } for i,c in enumerate(categories)]


class SelectionTests(unittest.TestCase):
    def test_video_windows_backslash(self):
        self.assertEqual(basename(r'folder\sub\F_66.mp4'), 'F_66.mp4')

    def test_balanced_groups(self):
        selected = choose_balanced(sample_rows(), seed=42, target=4, very_early=3,
                                   late=2, no_alert=2, burst=2)
        self.assertEqual(len(selected), 13)
        self.assertEqual(len({basename(r['video']) for r,_ in selected}), 13)
        from collections import Counter
        self.assertEqual(dict(Counter(g for _,g in selected)), {
            'first_1to3s':4, 'first_gt3s':3, 'first_lt1s':2, 'no_alert':2,
            'pair_burst_audit':2,
        })

    def test_reproducible(self):
        a = choose_balanced(sample_rows())
        b = choose_balanced(sample_rows())
        self.assertEqual(a, b)

    def test_zero_and_missing_categories(self):
        x = choose_balanced(sample_rows()[:3], target=4, very_early=3, late=2,
                            no_alert=2, burst=0)
        self.assertEqual(len(x), 3)

    def test_clip_bounds(self):
        self.assertEqual(clip_bounds(6.366, 32., 5., 1.), (1.3659999999999997, 7.366))
        self.assertEqual(clip_bounds(.8, 2.0, 5., 1.), (0., 1.8))

    def test_clip_bounds_invalid(self):
        with self.assertRaises(ValueError):
            clip_bounds(-1, 10)
        with self.assertRaises(ValueError):
            clip_bounds(8, 0)

    def test_index_existing_video_in_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = root/'F_0.mp4'; p.touch()
            manifest = root/'manifest.csv'
            with manifest.open('w',newline='') as f:
                wr = csv.DictWriter(f, fieldnames=['video']); wr.writeheader(); wr.writerow({'video': str(p)})
            got = video_index(manifest, [root])
            self.assertEqual(got['F_0.mp4'], p)

    def test_dry_run_csv_never_labels_positive(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); videos=root/'videos'; videos.mkdir()
            v=videos/'F_0.mp4'; v.touch()
            src=root/'in.csv'
            with src.open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(sample_rows()[0])); wr.writeheader()
                wr.writerow({'video':'F_0.mp4','fight_onset_s':'10',
                             'first_alert_category':'1_to_3s_early',
                             'first_alert_s':'8','max_distinct_pairs_in_1s':'0'})
            args=Namespace(input=str(src),shortlist=str(root/'missing.csv'),
                           manifest=str(root/'missing_manifest.csv'),
                           videos_root=str(videos),output=str(root/'out'),
                           seed=42,target=1,very_early=0,late=0,no_alert=0,burst=0,
                           pre=5,post=1,width=960,dry_run=True)
            self.assertEqual(run(args),0)
            with (root/'out'/'review_annotations.csv').open(encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['review_label'],'unreviewed')
            self.assertEqual(rows[0]['export_status'],'dry_run')
            self.assertEqual(rows[0]['precursor_start_s'],'')
            self.assertEqual(json.loads((root/'out'/'export_summary.json').read_text())['selected'],1)


    def test_refuses_to_overwrite_reviewed_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            src=root/'in.csv'
            with src.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['video','fight_onset_s','first_alert_category'])
                w.writeheader(); w.writerow({'video':'F.mp4','fight_onset_s':'2','first_alert_category':'no_alert'})
            out=root/'out'; out.mkdir()
            (out/'review_annotations.csv').write_text('video,review_label\nF.mp4,verified_precursor\n')
            args=Namespace(input=str(src),shortlist=str(root/'missing.csv'),manifest=str(root/'missing.csv'),
                           videos_root=str(root/'missing'),output=str(out),seed=42,
                           target=0,very_early=0,late=0,no_alert=1,burst=0,
                           pre=5,post=1,width=960,dry_run=True)
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):
                run(args)
            self.assertIn('verified_precursor',(out/'review_annotations.csv').read_text())

    def test_export_real_synthetic_video(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('opencv-python and numpy unavailable')
        from src.export_v4_training_review import export_video
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            source=root/'input.mp4'; dest=root/'clip.mp4'
            w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10.,(160,120))
            if not w.isOpened():
                self.skipTest('MP4 encoder not available')
            for i in range(30):
                w.write(np.full((120,160,3),i*2,dtype=np.uint8))
            w.release()
            start,end,n=export_video(source,dest,onset=2,pre=5,post=1,width=320)
            self.assertEqual((start,end,n),(0.0,3.0,30))
            self.assertTrue(dest.is_file())
            cap=cv2.VideoCapture(str(dest))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),30)
            cap.release()


if __name__=='__main__':
    unittest.main()
