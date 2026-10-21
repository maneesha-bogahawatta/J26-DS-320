import json
import unittest
from unittest.mock import patch

import numpy as np

from src.incident_events import build_incidents, project_shared_event, shared_contract, utc_timestamp
from src.predict_advanced import CausalDecision, pose_evidence, score_cache
from src.advanced_experiment import SCHEMA as MODEL_SCHEMA
from src.pose_sequences import pack_frames
from src.pose_dataset import sequence_window


class IncidentEventsTests(unittest.TestCase):
    def record(self, t=2., status='WEAK_ONSET_WITHIN_3S_ALERT', emitted=False):
        return dict(t_s=t, candidate_status=status, valid_pose=True, segment_id=0,
            quality=.8, current_observed_joints=17, observed_history_frames=30,
            smoothed_onset_score=.7, smoothed_fight_score=.8, alert_emitted=emitted,
            track_ids_json='[7, 9]', bounding_box_json='[1, 2, 30, 40]',
            source_frame_number=60, evidence_signals_json='[{"signal":"observed_pose_joints","value":34}]',
            notification_suppression_reason='research_notifications_not_requested')

    def build(self, rows, **kwargs):
        meta = dict(source_sha256='abc', source_video='recording.mp4', extracted_start_s=1.)
        return build_incidents(rows, meta, dict(variant='baseline', supervision='weak'), 'def',
            processed_at='2026-10-21T12:00:00Z', **kwargs)

    def test_research_incident_is_stable_and_never_claims_recording_time_or_probability(self):
        a, notes = self.build([self.record(), self.record(2.2)])
        b, _ = self.build([self.record(), self.record(2.2)])
        self.assertEqual(a, b); self.assertEqual(len(a['incidents']), 1); self.assertEqual(notes, [])
        event = a['incidents'][0]
        self.assertEqual(event['timestamp_origin'], 'processing_utc_not_recording_time')
        self.assertEqual(event['evidence']['annotated_video_offset_s'], 1.)
        self.assertEqual(event['operator_review_status'], 'pending')
        self.assertIsNone(event['shared_event'])
        self.assertFalse(event['operational_notification'])
        json.dumps(a, allow_nan=False)

    def test_active_projection_roundtrips_actual_shared_contract(self):
        result, notes = self.build([self.record(status='WEAK_ACTIVE_FIGHT_ALERT', emitted=True)],
            camera_id='cam_test', recording_start='2026-10-21T10:30:00+05:30')
        event = result['incidents'][0]
        self.assertEqual(event['timestamp'], '2026-10-21T05:00:02Z')
        shared = event['shared_event']; contract = shared_contract()
        self.assertEqual(contract.IncidentEvent.from_dict(shared).to_dict(), shared)
        self.assertEqual(shared['event_type'], 'fighting')
        self.assertEqual(shared['person_id'], 'P-7')
        self.assertEqual(len(notes), 1)
        event['incident_kind'] = 'potential_aggression'
        with self.assertRaises(ValueError): project_shared_event(event)

    def test_suppressed_episode_tracks_later_emission_and_delay(self):
        a = self.record(); a['notification_suppression_reason'] = 'cooldown_same_kind'
        result, notes = self.build([a, self.record(t=3., emitted=True)])
        self.assertEqual(len(result['incidents']), 1)
        self.assertEqual(result['incidents'][0]['notification']['delay_from_first_candidate_s'], 1.)
        self.assertEqual(len(notes), 1)

    def test_abstention_never_generates_incident(self):
        row = self.record(); row['valid_pose'] = False
        self.assertEqual(self.build([row])[0]['incidents'], [])
        row = self.record(); row['track_ids_json'] = '[]'
        self.assertEqual(self.build([row])[0]['incidents'], [])
        with self.assertRaises(ValueError): utc_timestamp('2026-10-21T10:00:00')
        row = self.record(); row['smoothed_onset_score'] = float('nan')
        with self.assertRaises(ValueError): self.build([row])

    def test_separate_cooldowns_do_not_hide_active_escalation(self):
        engine = CausalDecision([.5, .5], alpha=1.)
        for t in (0., .2, .4): result = engine.update(t, [.9, .1], True, 0)
        self.assertTrue(result[2])
        engine.update(.6, [.9, .9], True, 0)
        self.assertTrue(engine.update(.8, [.9, .9], True, 0)[2])

    def test_suppressed_new_episode_can_notify_when_cooldown_expires(self):
        engine = CausalDecision([.5, .5], alpha=1., cooldown_s=1.)
        for t in (0., .2, .4): first = engine.update(t, [.9, .1], True, 0)
        self.assertTrue(first[2])
        for t in (.6, .8): engine.update(t, [.1, .1], True, 0)
        engine.update(1., [.9, .1], True, 0)
        self.assertFalse(engine.update(1.2, [.9, .1], True, 0)[2])
        self.assertEqual(engine.audit['notification_suppression_reason'], 'cooldown_same_kind')
        self.assertTrue(engine.update(1.4, [.9, .1], True, 0)[2])
        self.assertFalse(engine.update(1.6, [.9, .1], True, 0)[2])

    def test_scene_change_has_independent_notification_state(self):
        engine = CausalDecision([.5, .5], alpha=1.)
        for t in (0., .2, .4): engine.update(t, [.9, .1], True, 0)
        for t in (.6, .8, 1.): result = engine.update(t, [.9, .1], True, 1)
        self.assertTrue(result[2])
        with self.assertRaises(ValueError): engine.update(float('nan'), [.9, .1], True, 1)

    def test_identity_change_requires_fresh_votes_and_does_not_reuse_other_pair_cooldown(self):
        engine = CausalDecision([.5, .5], alpha=1.)
        for t in (0., .2, .4): first = engine.update(t, [.9, .1], True, 0, identity=(1,2))
        self.assertTrue(first[2])
        self.assertFalse(engine.update(.6, [.9, .1], True, 0, identity=(3,4))[2])
        self.assertTrue(engine.audit['identity_reset'])
        engine.update(.8, [.9, .1], True, 0, identity=(3,4))
        self.assertTrue(engine.update(1., [.9, .1], True, 0, identity=(4,3))[2])
        result, _ = self.build([self.record(), dict(self.record(2.2), track_ids_json='[3, 4]')])
        self.assertEqual(len(result['incidents']), 2)
        self.assertNotEqual(result['incidents'][0]['event_id'], result['incidents'][1]['event_id'])

    def test_wrong_source_is_rejected_before_model_construction_or_scoring(self):
        ck = dict(schema=MODEL_SCHEMA, verified_early_aggression_claim=False)
        with patch('src.predict_advanced.torch.load', return_value=ck), \
             patch('src.predict_advanced.load_cache', return_value=({},dict(video_id='reserved', source_sha256='test'))), \
             patch('src.predict_advanced.build_model') as model:
            with self.assertRaises(ValueError):
                score_cache('unused.pt', 'unused.npz', expected_source=dict(video_id='val', source_sha256='allowed'))
            model.assert_not_called()

    def test_motion_evidence_is_prefix_only_and_absent_tracks_not_exported(self):
        frames = []
        for i in range(12):
            kp = np.ones((2, 17, 3), np.float32); kp[..., :2] *= 20+i
            frames.append(dict(t=i/15, frame_index=i, keypoints=kp,
                boxes=np.array([[0,0,60,90],[20,0,90,95]],np.float32), track_ids=np.array([1,2])))
        a = pack_frames(frames[:10], (100,100)); b = pack_frames(frames, (100,100))
        self.assertEqual(pose_evidence(a, sequence_window(a, 9), 9), pose_evidence(b, sequence_window(b, 9), 9))
        self.assertEqual(json.loads(pose_evidence(a, sequence_window(a, 9), 9)['track_ids_json']), [1,2])
        missing = dict(t=12/15, frame_index=12, keypoints=np.zeros((0,17,3),np.float32),
            boxes=np.zeros((0,4),np.float32),track_ids=np.zeros(0,np.int64))
        c = pack_frames(frames+[missing], (100,100))
        self.assertEqual(json.loads(pose_evidence(c, sequence_window(c, 12), 12)['track_ids_json']), [])


if __name__ == '__main__': unittest.main()
