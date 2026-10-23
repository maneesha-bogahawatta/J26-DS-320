import copy
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from src.contact_review import (CONTACT_DEFINITION, FORM_SCHEMA, blank_contact_form,
    import_contact_evidence, import_forms, prepare, readiness_report, validate_contact_record)
from src.foundation_io import sha256, write_csv, write_json
from src.human_review import PACK_SCHEMA
from src.lock_splits import assign
from src.pose_dataset import labels_at
from src.reconcile_reviews import normalize


class ContactReviewTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(video_id='f_1', video='F_1.mp4', split='train', duration_s='20',
            content_sha256='source_hash', leakage_group='source1', intervals='[[9,15]]')
        self.clip = dict(clip_id='train_001', video_id='f_1', split='train',
            source_sha256='source_hash', leakage_group='source1', clip_start_s=5., clip_end_s=13.)
        self.pack = dict(schema=PACK_SCHEMA, clips=[self.clip])
        self.raw = dict(clip_id='train_001', review_label='precursor',
            timestamp_basis='original_video_seconds', reviewed_ranges_s=[[5., 13.]],
            precursors=[[8., 9.]], first_threatening_cue_s=8., first_physical_contact_s=10.,
            verified_fight_intervals=[[10., 13.]], contact_observation='confirmed_visible',
            contact_uncertainty_s=None, contact_definition=CONTACT_DEFINITION,
            contact_personally_observed=True, contact_not_inferred_from_dataset_onset=True,
            review_notes='Visible raised fist before forceful push. Dataset onset is not contact evidence.')
        self.base = dict(records=[normalize(self.row, self.row, 'parent', 'manifest')])

    def form(self, reviewer='Alice', raw=None, split='train'):
        return dict(schema=FORM_SCHEMA, pack_sha256='pack_hash', reviewer=reviewer,
            split=split, independent_human_review=True,
            records=[copy.deepcopy(self.raw if raw is None else raw)])

    def import_review(self, forms):
        return import_contact_evidence(self.pack, 'pack_hash', [self.row], self.base, forms)

    def test_blank_forms_do_not_generate_or_infer_labels(self):
        form = blank_contact_form(self.pack, 'pack_hash', 'train')
        records, report = self.import_review([('blank', form)])
        self.assertEqual(report['agreed_clips'], [])
        self.assertEqual(records[-1]['review_status'], 'unreviewed')
        self.assertIsNone(records[-1]['contact_s'])
        self.assertIsNone(records[-1]['contact_semantics']['first_threatening_cue_s'])
        self.assertEqual(report['strict_contact_readiness']['status'], 'blocked')
        with self.assertRaisesRegex(ValueError, 'test'):
            blank_contact_form(self.pack, 'pack_hash', 'test')

    def test_two_matching_reviews_append_without_editing_parents(self):
        original = copy.deepcopy(self.base)
        records, report = self.import_review([('a', self.form()), ('b', self.form('Bob'))])
        self.assertEqual(self.base, original)
        self.assertEqual(records[0], original['records'][0])
        self.assertEqual(report['agreed_clips'], ['train_001'])
        self.assertEqual(records[-1]['review_status'], 'independently_verified')
        self.assertTrue(records[-1]['eligible_for_fitting'])
        self.assertEqual(records[-1]['contact_s'], 10.)  # weak boundary is 9, never inherited
        self.assertEqual(records[-1]['contact_semantics']['first_threatening_cue_s'], 8.)
        self.assertEqual(records[1]['submitted'], self.raw)
        targets, mask = labels_at(8.5, records)
        self.assertTrue(mask[0]); self.assertEqual(targets[0], 1.)
        self.assertFalse(labels_at(6., records)[1].any())

    def test_single_review_is_provisional(self):
        records, report = self.import_review([('a', self.form())])
        self.assertEqual(records[-1]['review_status'], 'provisional_single_review')
        self.assertFalse(records[-1]['eligible_for_fitting'])
        self.assertEqual(report['single_review_clips'], ['train_001'])

    def test_uncertain_contact_never_creates_precursor_target(self):
        raw = dict(self.raw, review_label='uncertain', first_physical_contact_s=None,
            contact_observation='uncertain', contact_uncertainty_s=[9.8, 10.2],
            contact_personally_observed=False)
        records, report = self.import_review([('a', self.form(raw=raw)), ('b', self.form('Bob', raw))])
        self.assertEqual(report['agreed_clips'], [])
        self.assertFalse(labels_at(8.5, records)[1].any())
        self.assertEqual(records[-1]['contact_semantics']['contact_uncertainty_s'], [9.8, 10.2])

    def test_contact_requires_explicit_visible_observation_and_no_inferred_onset(self):
        for field, value in [('contact_personally_observed', False),
            ('contact_not_inferred_from_dataset_onset', False),
            ('contact_not_inferred_from_dataset_onset', 'true'),
            ('contact_definition', ''), ('timestamp_basis', 'clip_relative_seconds'),
            ('contact_uncertainty_s', [9.9, 10.1])]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_contact_record(dict(self.raw, **{field: value}), self.clip, 20.)

    def test_missing_reviewer_attestation_or_repeated_identity_rejected(self):
        for change in [dict(reviewer=''), dict(independent_human_review=False),
                       dict(independent_human_review='true')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.import_review([('a', dict(self.form(), **change))])
        with self.assertRaisesRegex(ValueError, 'second independent'):
            self.import_review([('a', self.form()), ('b', self.form('alice'))])

    def test_contact_semantics_disagreement_not_averaged(self):
        other = dict(self.raw, first_physical_contact_s=10.1, verified_fight_intervals=[[10.1, 13]])
        records, report = self.import_review([('a', self.form()), ('b', self.form('Bob', other))])
        self.assertEqual(report['disagreement_clips'], ['train_001'])
        self.assertFalse(any(r.get('eligible_for_fitting') for r in records))

    def test_invalid_time_order_overlap_and_coverage_rejected(self):
        cases = [dict(first_threatening_cue_s=8.2), dict(precursors=[[8, 10]]),
            dict(precursors=[[8, 9], [8.5, 9.5]]), dict(reviewed_ranges_s=[[5, 10], [9, 13]]),
            dict(verified_fight_intervals=[[9, 13]]), dict(first_physical_contact_s=14),
            dict(first_physical_contact_s=float('nan')), dict(first_physical_contact_s=True),
            dict(precursors=[[5, 6]], first_threatening_cue_s=5),
            dict(reviewed_ranges_s=[[5, 8], [10, 13]])]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_contact_record(dict(self.raw, **change), self.clip, 20.)

    def test_ongoing_violence_without_observed_onset_is_not_early_label(self):
        raw = dict(self.raw, review_label='already_violent', precursors=[],
            first_threatening_cue_s=None, first_physical_contact_s=None,
            contact_observation='already_in_progress', contact_personally_observed=False,
            contact_not_inferred_from_dataset_onset=False, verified_fight_intervals=[[5, 13]])
        records, report = self.import_review([('a', self.form(raw=raw)), ('b', self.form('Bob', raw))])
        self.assertEqual(report['agreed_clips'], ['train_001'])
        targets, mask = labels_at(8.5, records)
        self.assertFalse(mask[0]); self.assertTrue(mask[1]); self.assertEqual(targets[1], 1.)
        self.assertIsNone(records[-1]['contact_s'])

    def test_non_aggressive_coverage_is_scoped_and_cannot_contain_contact(self):
        raw = dict(self.raw, review_label='non_aggressive', reviewed_ranges_s=[[5, 7]],
            precursors=[], first_threatening_cue_s=None, first_physical_contact_s=None,
            verified_fight_intervals=[], contact_observation='not_observed',
            contact_personally_observed=False, contact_not_inferred_from_dataset_onset=False)
        records, report = self.import_review([('a', self.form(raw=raw)), ('b', self.form('Bob', raw))])
        self.assertTrue(labels_at(6, records)[1].all())
        self.assertFalse(labels_at(8, records)[1].any())
        self.assertEqual(report['strict_contact_readiness']['by_split']['train']['non_aggressive_videos'], 1)
        with self.assertRaises(ValueError):
            validate_contact_record(dict(self.raw, review_label='non_aggressive'), self.clip, 20.)

    def test_test_or_wrong_split_identity_and_duplicate_clips_rejected(self):
        self.row['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'held-out test'):
            self.import_review([('a', self.form())])
        self.row['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'remain separate'):
            self.import_review([('a', self.form(split='val'))])
        form = self.form(); form['records'].append(copy.deepcopy(form['records'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate clip'):
            self.import_review([('a', form)])

    def test_readiness_blocks_legacy_and_test_counts_and_has_null_accuracy(self):
        records, _ = self.import_review([('a', self.form()), ('b', self.form('Bob'))])
        positive = records[-1]
        legacy = dict(positive); legacy.pop('source_schema')
        result = readiness_report([legacy, dict(positive, split='test')])
        self.assertEqual(result['by_split']['train']['precursor_positive_videos'], 0)
        self.assertEqual(result['excluded_legacy_verified_records'], 1)
        self.assertEqual(result['status'], 'blocked')
        self.assertIsNone(result['verified_early_warning_accuracy'])
        self.assertFalse(result['scientific_validation_achieved'])

    def test_forged_strict_status_without_contact_semantics_does_not_count(self):
        record = dict(self.base['records'][0], source_schema=FORM_SCHEMA,
            review_status='independently_verified', label='precursor', issues=[])
        result = readiness_report([record])
        self.assertEqual(result['rejected_malformed_strict_records'], 1)
        self.assertEqual(result['status'], 'blocked')

    def test_annotation_minimum_is_separate_from_cache_and_scientific_readiness(self):
        positives, _ = self.import_review([('a', self.form()), ('b', self.form('Bob'))])
        negative = dict(self.raw, review_label='non_aggressive', reviewed_ranges_s=[[5, 7]],
            precursors=[], first_threatening_cue_s=None, first_physical_contact_s=None,
            verified_fight_intervals=[], contact_observation='not_observed',
            contact_personally_observed=False, contact_not_inferred_from_dataset_onset=False)
        negatives, _ = self.import_review([('a', self.form(raw=negative)), ('b', self.form('Bob', negative))])
        records = [positives[-1], negatives[-1], dict(positives[-1], split='val', video_id='f_val'),
                   dict(negatives[-1], split='val', video_id='n_val')]
        result = readiness_report(records)
        self.assertEqual(result['status'], 'annotation_minimum_met_cache_checks_required')
        self.assertEqual(result['blockers'], [])
        self.assertFalse(result['scientific_validation_achieved'])
        self.assertFalse(result['model_trained'])
        self.assertIsNone(result['verified_early_warning_accuracy'])

    def test_cli_preserves_parents_checks_media_and_refuses_changed_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = assign([dict(self.row, group='source1')], set())
            locked = root/'ubi_locked.csv'; write_csv(locked, rows)
            write_json(root/'split_metadata.json', dict(locked_manifest_sha256=sha256(locked), reserved_test_ids=[]))
            base = root/'base.json'
            write_json(base, dict(self.base, schema='c3_review_evidence_v1', locked_manifest_sha256=sha256(locked)))
            (root/'clip.mp4').write_bytes(b'immutable media fixture')
            clip = dict(self.clip, leakage_group=rows[0]['leakage_group'],
                media='clip.mp4', media_sha256=sha256(root/'clip.mp4'))
            pack = root/'pack.json'; write_json(pack, dict(self.pack, clips=[clip], locked_manifest_sha256=sha256(locked)))
            form = self.form(); form['pack_sha256'] = sha256(pack)
            path = root/'Alice.json'; write_json(path, form)
            args = Namespace(locked=str(locked), annotations=str(base), pack=str(pack),
                reviews=[str(path)], out_dir=str(root/'new'))
            snapshot = base.read_bytes()
            import_forms(args); import_forms(args)
            self.assertEqual(base.read_bytes(), snapshot)
            got = json.loads((root/'new/annotations.json').read_text())
            self.assertEqual(got['records'][0], self.base['records'][0])
            form['records'][0]['review_notes'] = 'Preserve revised note separately.'
            path.write_text(json.dumps(form))
            with self.assertRaises(FileExistsError):
                import_forms(args)
            (root/'clip.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'media changed'):
                import_forms(args)


if __name__ == '__main__':
    unittest.main()
