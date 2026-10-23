import copy
import unittest

from src.ai_review_evidence import AI_SCHEMA, AI_STATUS, normalize_ai_reviews
from src.human_review import PACK_SCHEMA, import_evidence
from src.pose_dataset import labels_at


class AIReviewEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(video_id='f_1', video='F_1.mp4', split='train', duration_s='20',
                         content_sha256='source_hash', leakage_group='group1')]
        self.pack = dict(schema=PACK_SCHEMA, clips=[dict(clip_id='train_001', video_id='f_1',
            split='train', source_video='F_1.mp4', source_sha256='source_hash', leakage_group='group1',
            clip_start_s=5., clip_end_s=13.)])
        self.raw = dict(clip_id='train_001', review_label='precursor', reviewed_ranges_s=[[5, 13]],
            proposed_precursors=[[8, 9]], proposed_contact_s=10., contact_uncertainty_s=[9.8, 10.2],
            observed_fight_intervals=[[10.5, 12]], observed_frame_times_s=[5, 8, 9, 10, 12.5],
            review_notes='Visible arm wind-up followed by contact; AI-provisional estimate.')
        self.evidence = dict(schema=AI_SCHEMA, pack_sha256='pack_hash', reviewer_type='ai',
            independent_human_review=False, review_method='Chronological sampled frames',
            sampling_limitations=['Not full-rate playback; onset estimated'], records=[self.raw])

    def normalize(self):
        return normalize_ai_reviews(self.evidence, self.pack, 'pack_hash', self.rows, 'fixture')

    def test_positive_ai_review_never_grants_verified_supervision(self):
        records = self.normalize()
        self.assertEqual(records[0]['review_status'], AI_STATUS)
        self.assertFalse(records[0]['eligible_for_fitting'])
        self.assertTrue(records[0]['requires_human_adjudication'])
        self.assertFalse(labels_at(8.5, records)[1].any())
        self.assertEqual(records[0]['submitted'], self.raw)

    def test_ai_schema_cannot_be_imported_as_independent_human_review(self):
        with self.assertRaisesRegex(ValueError, 'form does not match'):
            import_evidence(self.pack, 'pack_hash', self.rows, {'records': []}, [('ai', self.evidence)])

    def test_human_attestation_or_verified_status_claim_is_rejected(self):
        self.evidence['independent_human_review'] = True
        with self.assertRaisesRegex(ValueError, 'cannot claim'):
            self.normalize()
        self.evidence['independent_human_review'] = False
        self.raw['review_status'] = 'independently_verified'
        with self.assertRaisesRegex(ValueError, 'provisional'):
            self.normalize()

    def test_contact_uncertainty_is_mandatory_and_precontact_bound_conservative(self):
        for bracket in (None, [10.1, 10.2], [8.5, 10.2], [9.8, float('nan')]):
            self.raw['contact_uncertainty_s'] = bracket
            with self.subTest(bracket=bracket), self.assertRaises(ValueError):
                self.normalize()

    def test_test_sources_and_changed_pack_are_rejected(self):
        self.rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Test'):
            self.normalize()
        self.rows[0]['split'] = 'train'
        self.evidence['pack_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.normalize()

    def test_all_clips_notes_and_sample_provenance_required(self):
        original = copy.deepcopy(self.evidence)
        for field, value in [('records', []), ('review_method', ''), ('sampling_limitations', [])]:
            self.evidence = dict(original, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.normalize()
        self.evidence = copy.deepcopy(original)
        self.evidence['records'][0]['observed_frame_times_s'] = [14]
        with self.assertRaisesRegex(ValueError, 'sampled'):
            self.normalize()

    def test_negative_ai_review_also_stays_masked(self):
        self.raw.update(review_label='non_aggressive', proposed_precursors=[],
            proposed_contact_s=None, contact_uncertainty_s=None, observed_fight_intervals=[])
        records = self.normalize()
        self.assertFalse(labels_at(8.5, records)[1].any())


if __name__ == '__main__':
    unittest.main()
