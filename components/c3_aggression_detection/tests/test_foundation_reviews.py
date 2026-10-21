import unittest
from src.reconcile_reviews import normalize, covered_seconds, coverage_report


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(video_id='f_1',split='train',duration_s='20')
        self.review = dict(review_label='verified_precursor', reviewer_1='A', reviewer_2='B',
                           clip_start_s='3',clip_end_s='11',precursor_start_s='7',
                           precursor_end_s='9',first_physical_contact_s='10',review_notes='keep this')

    def test_verified_requires_real_reviewers_and_explicit_bounds(self):
        r = normalize(self.review,self.base,'fixture','v4')
        self.assertEqual(r['review_status'],'independently_verified')
        self.assertEqual(r['precursors'],[[7.,9.]])
        self.review['reviewer_2']='a'
        self.assertEqual(normalize(self.review,self.base,'fixture','v4')['review_status'],'provisional_single_review')

    def test_no_heuristic_promotion(self):
        r = normalize(dict(review_label='unreviewed',auto_precursors='[[7,9]]',
            clip_start_s='3',clip_end_s='11'),self.base,'fixture','v4')
        self.assertEqual(r['review_status'],'unreviewed')
        self.assertEqual(r['precursors'],[])

    def test_notes_and_test_exclusion(self):
        self.base['split']='test'
        r = normalize(self.review,self.base,'fixture','v4')
        self.assertEqual(r['original'],self.review)
        self.assertFalse(r['eligible_for_fitting'])
        self.assertEqual(r['usage'],'held_out_evaluation_only')

    def test_uncertain_even_with_two_ids_not_verified(self):
        self.review['review_label']='uncertain'
        self.assertEqual(normalize(self.review,self.base,'fixture','v4')['review_status'],'provisional_single_review')

    def test_invalid_range_masks_verification(self):
        self.review['precursor_end_s']='12'
        r = normalize(self.review,self.base,'fixture','v4')
        self.assertTrue(r['issues'])
        self.assertFalse(r['eligible_for_fitting'])

    def test_control_scope_not_whole_video(self):
        r=normalize(dict(review_label='non_aggressive',window_start_s='5',window_end_s='10'),self.base,'fixture','control')
        self.assertEqual(r['coverage_s'],[[5,10]])
        self.assertEqual(r['review_status'],'provisional_single_review')
        self.assertEqual(covered_seconds([[1,3],[2,4],[8,9]]),4)

    def test_finalized_manifest_with_real_precursor_is_supported(self):
        original=dict(annotation_status='verified',reviewer_a='A',reviewer_b='B',
                      intervals='[[10,15]]',precursors='[[7,9]]')
        r=normalize(original,self.base,'fixture','manifest')
        self.assertEqual(r['review_status'],'independently_verified')
        self.assertEqual(r['contact_s'],10.)

    def test_malformed_review_is_preserved_and_ineligible(self):
        original=dict(annotation_status='verified',reviewer_a='A',reviewer_b='B',
                      intervals='[[10,15]]',precursors='not JSON')
        r=normalize(original,self.base,'fixture','manifest')
        self.assertTrue(r['issues'])
        self.assertFalse(r['eligible_for_fitting'])
        self.assertEqual(r['original'],original)


if __name__ == '__main__':
    unittest.main()
