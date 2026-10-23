"""Regression checks for the user's supplied v1.1 dashboard projection."""
import copy
import json
import unittest

from src.dashboard_adapter import adapt_run, normalized_render_box
from src.predict_advanced import render_geometry


class DashboardAdapterTests(unittest.TestCase):
    def fixture(self):
        prediction = dict(operational_alerts_enabled=False, verified_early_aggression_claim=False,
            variant='baseline', supervision='weak_ubi_first_onset_and_binary_fight_only',
            source_context=[13., 21.], media_geometry=render_geometry(1280,720))
        incident = dict(event_id='local-uuid', incident_kind='potential_aggression',
            research_only=True, operational_notification=False, model_score=.71,
            video_timestamp_s=17., camera_id='cam_lobby', timestamp='2026-10-21T12:00:00+05:30',
            timestamp_origin='processing_utc_not_recording_time', track_ids=[12,15], model_version='baseline:abc',
            reliability=dict(supervision='weak',quality=.8,current_observed_joints=17,observed_history_frames=40),
            evidence_signals=[dict(signal='observed_closing_speed',value=.12,units='frame_normalized_distance_per_second')],
            evidence=dict(source_frame_number=510,annotated_video_offset_s=4.,annotated_video='demo.mp4',
                bounding_box=[0,0,1280,720]), notification=dict(emitted=False))
        return incident, prediction

    def test_projects_exact_fields_with_cautious_uncalibrated_research_semantics(self):
        incident, prediction = self.fixture(); event = adapt_run([incident], prediction, 'job-1')[0]
        self.assertEqual(event['component_id'], 'component_3')
        self.assertEqual(event['incident_type'], 'aggression')
        self.assertFalse(event['score_calibrated'])
        self.assertEqual(event['confidence'], .71)
        self.assertIn('weak onset candidate', event['summary'])
        self.assertEqual(event['status'], 'new')
        self.assertEqual(event['timestamp_detected'], '2026-10-21T06:30:00Z')
        self.assertEqual(event['track_id'], 'person-12')
        self.assertEqual(event['related_tracks'][0]['track_id'], 'person-15')
        self.assertIsNone(event['related_tracks'][0]['association_score'])
        self.assertEqual(set(event), {'event_id','component_id','incident_type','camera_id','timestamp_detected',
            'priority','status','summary','evidence','model_name','model_version','confidence','bounding_box',
            'track_id','related_tracks','media_ref','media_offset_seconds','score_calibrated','zone_id','audit'})
        signals={s['name']:s for s in event['evidence']}
        self.assertTrue(signals['c3_research_only']['value'])
        self.assertFalse(signals['operational_alerts_enabled']['value'])
        self.assertIsNone(signals['occlusion_ratio']['value'])
        json.dumps(event, allow_nan=False)

    def test_bbox_and_offset_follow_annotated_media_not_original_video(self):
        incident, prediction=self.fixture(); event=adapt_run([incident],prediction,'job-1')[0]
        self.assertEqual(event['media_ref'],'/api/v1/c3/media/job-1/demo.mp4')
        self.assertEqual(event['media_offset_seconds'],4.)
        self.assertEqual(event['bounding_box']['x_max'],1.)
        self.assertAlmostEqual(event['bounding_box']['y_max'],540/656)
        self.assertLess(event['bounding_box']['y_max'],1.)
        self.assertEqual(prediction['media_geometry']['rendered_height'],656)
        self.assertEqual(prediction['media_geometry']['source_width'],1280)

    def test_legacy_reports_omit_overlay_without_inventing_dimensions(self):
        incident,prediction=self.fixture(); prediction.pop('media_geometry')
        event=adapt_run([incident],prediction,'job-1')[0]
        self.assertIsNone(event['bounding_box'])
        self.assertTrue(any(s['name']=='source_bounding_box_xyxy' for s in event['evidence']))

    def test_run_scoping_is_idempotent_and_independent_across_reruns(self):
        incident,prediction=self.fixture(); before=copy.deepcopy((incident,prediction))
        a=adapt_run([incident],prediction,'job-1'); b=adapt_run([incident],prediction,'job-1')
        c=adapt_run([incident],prediction,'job-2')
        self.assertEqual(a,b);self.assertNotEqual(a[0]['event_id'],c[0]['event_id'])
        self.assertNotEqual(a[0]['media_ref'],c[0]['media_ref'])
        self.assertEqual((incident,prediction),before)

    def test_active_head_is_separate_without_operational_claim(self):
        incident,prediction=self.fixture();incident['incident_kind']='active_violence'
        event=adapt_run([incident],prediction,'job-1')[0]
        self.assertEqual(event['incident_type'],'violence')
        self.assertIn('weak-model candidate',event['summary'])
        self.assertFalse(event['score_calibrated'])

    def test_rejects_mismatched_timestamps_and_duplicate_local_ids(self):
        incident,prediction=self.fixture()
        with self.assertRaises(ValueError):adapt_run([incident,incident],prediction,'job-1')
        incident['evidence']['annotated_video_offset_s']=17.
        with self.assertRaises(ValueError):adapt_run([incident],prediction,'job-1')
        incident,prediction=self.fixture();incident['timestamp']='2026-10-21T12:00:00'
        with self.assertRaises(ValueError):adapt_run([incident],prediction,'job-1')

    def test_rejects_invalid_research_provenance_and_nonfinite_scores(self):
        for field,value in (('research_only',False),('operational_notification',True),('model_score',float('nan')),
                             ('model_score',2.),('track_ids',[True]),('track_ids',[-1]),('incident_kind','verified_fight')):
            incident,prediction=self.fixture();incident[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):adapt_run([incident],prediction,'job-1')
        incident,prediction=self.fixture();prediction['operational_alerts_enabled']=True
        with self.assertRaises(ValueError):adapt_run([incident],prediction,'job-1')
        with self.assertRaises(ValueError):adapt_run([],prediction,'../../outside')

    def test_render_geometry_rejects_zero_dimensions_and_box_area(self):
        with self.assertRaises(ValueError):render_geometry(0,720)
        _,prediction=self.fixture()
        with self.assertRaises(ValueError):normalized_render_box([2,2,1,1],prediction)
        self.assertIsNone(normalized_render_box([1400,900,1600,1000],prediction))


if __name__ == '__main__':unittest.main()
