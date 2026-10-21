import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from src.foundation_io import sha256, write_csv, write_json
from src.human_review import (FORM_SCHEMA, PACK_SCHEMA, blank_form, import_evidence,
                             import_forms, plan_clips, validate_record)
from src.lock_splits import assign
from src.pose_dataset import labels_at
from src.reconcile_reviews import normalize


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(video_id='f_1', video='F_1.mp4', split='train',
            leakage_group='source1', duration_s='20', intervals='[[10,15]]',
            content_sha256='source_hash')]
        self.clip = dict(clip_id='train_001', video_id='f_1', split='train',
            leakage_group='source1', source_video='F_1.mp4', source_sha256='source_hash',
            clip_start_s=5., clip_end_s=13.)
        self.pack = dict(schema=PACK_SCHEMA, clips=[self.clip])
        self.base = dict(records=[normalize(dict(notes='keep me', annotation_status='weak'),
                         self.rows[0], 'original', 'manifest')])
        self.raw = dict(clip_id='train_001', review_label='precursor',
            reviewed_ranges_s=[[5, 13]], precursors=[[8, 9]], first_physical_contact_s=10,
            verified_fight_intervals=[[10, 13]], review_notes='Observed visible arm movement')

    def form(self, reviewer='Alice', raw=None):
        return dict(schema=FORM_SCHEMA, pack_sha256='pack_hash', split='train',
            reviewer=reviewer, independent_human_review=True,
            records=[copy.deepcopy(self.raw if raw is None else raw)])

    def run_import(self, forms):
        return import_evidence(self.pack, 'pack_hash', self.rows, self.base, forms)

    def test_plan_uses_cached_context_and_never_assigns_labels(self):
        a = plan_clips(self.rows, {'f_1': (0., 20.)})
        self.assertEqual(a, plan_clips(self.rows, {'f_1': (0., 20.)}))
        self.assertEqual((a[0]['clip_start_s'], a[0]['clip_end_s']), (5, 13))
        self.assertNotIn('review_label', a[0])
        self.assertEqual(self.rows[0]['intervals'], '[[10,15]]')

    def test_plan_rejects_test_and_shared_sources(self):
        self.rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Test'):
            plan_clips(self.rows, {'f_1': (0., 20.)})
        self.rows[0]['split'] = 'train'
        self.rows.append(dict(self.rows[0], video_id='f_2'))
        with self.assertRaisesRegex(ValueError, 'source groups'):
            plan_clips(self.rows, {'f_1': (0., 20.), 'f_2': (0., 20.)})

    def test_blank_forms_create_no_supervision(self):
        form = blank_form(self.pack, 'pack_hash', 'train')
        records, report = self.run_import([('blank', form)])
        self.assertEqual(report['agreed_clips'], [])
        self.assertEqual(report['unreviewed_clips'], ['train_001'])
        self.assertEqual(records[-1]['review_status'], 'unreviewed')
        self.assertFalse(records[-1]['eligible_for_fitting'])

    def test_single_human_review_is_provisional_and_notes_preserved(self):
        records, report = self.run_import([('a', self.form())])
        self.assertEqual(records[-1]['review_status'], 'provisional_single_review')
        self.assertEqual(records[-1]['submitted'], self.raw)
        self.assertEqual(report['single_review_clips'], ['train_001'])
        self.assertFalse(records[-1]['eligible_for_fitting'])

    def test_matching_independent_reviews_append_scoped_supervision(self):
        snapshot = copy.deepcopy(self.base)
        records, report = self.run_import([('a', self.form()), ('b', self.form('Bob'))])
        self.assertEqual(self.base, snapshot)
        self.assertEqual(records[:len(self.base['records'])], self.base['records'])
        self.assertEqual(report['agreed_clips'], ['train_001'])
        self.assertTrue(records[-1]['eligible_for_fitting'])
        target, mask = labels_at(8.5, records[1:])
        self.assertTrue(mask[0]); self.assertEqual(target[0], 1)
        self.assertFalse(labels_at(4., records[1:])[1].any())
        # A reviewed precursor clip does not label all its background time negative.
        self.assertFalse(labels_at(6., records[1:])[1].any())

    def test_matching_non_aggression_is_only_coverage_scoped(self):
        raw = dict(self.raw, review_label='non_aggressive', reviewed_ranges_s=[[5, 7]],
                   precursors=[], first_physical_contact_s=None, verified_fight_intervals=[])
        records, report = self.run_import([('a', self.form(raw=raw)), ('b', self.form('Bob', raw))])
        self.assertEqual(report['agreed_clips'], ['train_001'])
        self.assertEqual(labels_at(6, records[1:])[0].tolist(), [0, 0])
        self.assertTrue(labels_at(6, records[1:])[1].all())
        self.assertFalse(labels_at(9, records[1:])[1].any())

    def test_disagreement_or_uncertainty_never_verifies(self):
        altered = dict(self.raw, precursors=[[8.1, 9]])
        records, report = self.run_import([('a', self.form()), ('b', self.form('Bob', altered))])
        self.assertEqual(report['disagreement_clips'], ['train_001'])
        self.assertFalse(any(r.get('eligible_for_fitting') for r in records))
        uncertain = dict(self.raw, review_label='uncertain')
        records, report = self.run_import([('a', self.form(raw=uncertain)), ('b', self.form('Bob', uncertain))])
        self.assertEqual(report['agreed_clips'], [])

    def test_duplicate_reviewer_cannot_become_second_independent_vote(self):
        with self.assertRaisesRegex(ValueError, 'same reviewer'):
            self.run_import([('a', self.form()), ('b', self.form('alice'))])

    def test_attestation_and_identity_are_required(self):
        for field, value in [('reviewer', ''), ('independent_human_review', False),
                             ('independent_human_review', 'true')]:
            form = self.form(); form[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'attestation'):
                self.run_import([('a', form)])

    def test_out_of_clip_future_contact_and_malformed_ranges_rejected(self):
        for change in (dict(reviewed_ranges_s=[[0, 13]]), dict(precursors=[[9, 11]]),
                       dict(first_physical_contact_s=14), dict(precursors=[[8, float('nan')]]),
                       dict(verified_fight_intervals=[[12, 15]])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_record(dict(self.raw, **change), self.clip, 20.)

    def test_negative_cannot_also_assert_aggression(self):
        with self.assertRaisesRegex(ValueError, 'Non-aggressive'):
            validate_record(dict(self.raw, review_label='non_aggressive'), self.clip, 20.)

    def test_wrong_pack_unknown_clip_and_wrong_role_rejected(self):
        for field, value in [('pack_sha256', 'wrong'), ('split', 'val')]:
            form = self.form(); form[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_import([('a', form)])
        form = self.form(); form['records'][0]['clip_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.run_import([('a', form)])
        self.rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Test'):
            self.run_import([('a', self.form())])

    def test_duplicate_clip_in_form_is_rejected(self):
        form = self.form(); form['records'].append(copy.deepcopy(form['records'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate clip'):
            self.run_import([('a', form)])

    def test_cli_import_preserves_base_checks_media_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = assign([dict(video='F_1.mp4', group='source1', split='train',
                duration_s='20', intervals='[[10,15]]', content_sha256='source_hash')], set())
            locked = root/'ubi_locked.csv'; write_csv(locked, rows)
            write_json(root/'split_metadata.json', dict(locked_manifest_sha256=sha256(locked), reserved_test_ids=[]))
            base = root/'base.json'
            write_json(base, dict(self.base, schema='c3_review_evidence_v1', locked_manifest_sha256=sha256(locked)))
            clip = dict(self.clip, leakage_group=rows[0]['leakage_group'], media='clip.mp4')
            (root/'clip.mp4').write_bytes(b'fixture review bytes')
            clip['media_sha256'] = sha256(root/'clip.mp4')
            pack = root/'pack.json'
            write_json(pack, dict(self.pack, clips=[clip], locked_manifest_sha256=sha256(locked), base_annotations_sha256=sha256(base)))
            form = self.form(); form['pack_sha256'] = sha256(pack)
            path = root/'a.json'; write_json(path, form)
            args = Namespace(locked=str(locked), annotations=str(base), pack=str(pack), reviews=[str(path)], out_dir=str(root/'new'))
            original = base.read_bytes()
            import_forms(args)
            self.assertEqual(base.read_bytes(), original)
            got = json.loads((root/'new/annotations.json').read_text())
            self.assertEqual(got['records'][0], self.base['records'][0])
            self.assertEqual(got['schema'], 'c3_review_evidence_v1')
            # An identical rerun is safe; a changed review cannot replace an artifact.
            import_forms(args)
            form['records'][0]['review_notes'] = 'New preserved note'
            path.write_text(json.dumps(form))
            with self.assertRaises(FileExistsError):
                import_forms(args)
            (root/'clip.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'media changed'):
                import_forms(args)


if __name__ == '__main__':
    unittest.main()
