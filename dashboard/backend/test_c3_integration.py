"""C3 dashboard integration tests with fixture inference, not model claims."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]/'components/c3_aggression_detection'))

from fastapi.testclient import TestClient
from main import create_app
from schemas import ActionRequest, IncidentEvent
from store import EventStore
from src.demo_server import DemoWorkbench, write_json


def fixture_adapter(incidents, prediction, job_id):
    return [dict(event_id=f'component_3-{job_id}-{item["event_id"]}', component_id='component_3',
        incident_type='aggression', camera_id='c3_fixture', timestamp_detected='2026-10-21T05:30:00Z',
        priority='medium', status='new', summary='Research weak-onset candidate; review required',
        evidence=[dict(name='research_only', value=True)], model_name='C3 ST-GCN', model_version='fixture',
        confidence=.7, score_calibrated=False, track_id='person-7',
        media_ref=f'/api/v1/c3/media/{job_id}/demo.mp4', media_offset_seconds=1.) for item in incidents]


class C3IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.test_bytes = b'locked-test-recording'
        locked = self.root/'data/foundation/v1/ubi_locked.csv'
        locked.parent.mkdir(parents=True)
        with locked.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['video_id','split','content_sha256'])
            writer.writeheader()
            writer.writerows([dict(video_id='val_fixture',split='val',content_sha256='val'),
                dict(video_id='test_fixture',split='test',content_sha256=hashlib.sha256(self.test_bytes).hexdigest())])
        index = self.root/'data/pose_cache/advanced_v1/val_index.json'
        index.parent.mkdir(parents=True)
        write_json(index,dict(entries={'val_fixture':{},'test_fixture':{}}))
        self.frontend = self.root/'frontend'
        self.frontend.mkdir()
        (self.frontend/'index.html').write_text('<!doctype html><title>Imported dashboard</title>',encoding='utf-8')
        self.state = self.root/'outputs/dashboard'
        self.wb = DemoWorkbench(self.root,self.state,runner=self.inference,max_upload_bytes=128)
        self.app = create_app(c3_enabled=True,workbench=self.wb,adapter=fixture_adapter,
            frontend_dir=self.frontend,state_dir=self.state)
        self.client = TestClient(self.app,base_url='http://127.0.0.1:8000')
        self.client.__enter__()
        self.headers = {'X-C3-Token':self.client.get('/api/v1/c3/bootstrap').json()['token']}

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.tmp.cleanup()

    def inference(self, job, output):
        output.mkdir()
        write_json(output/'prediction.json',dict(source_context=[5.,13.],camera_id='c3_fixture',
            timing=dict(valid_decisions=3,decisions=4),operational_alerts_enabled=False,
            probability_calibration='UNVALIDATED'))
        write_json(output/'incidents.json',dict(incidents=[dict(event_id='local-1',
            incident_kind='potential_aggression',operator_review_status='pending')]))
        write_json(output/'notifications.json',dict(notifications=[],operational_alerts_enabled=False))
        (output/'demo.mp4').write_bytes(b'0123456789')
        (output/'scores.csv').write_text('t_s,weak_onset_score,fight_score\n6,.7,.2\n',encoding='utf-8')

    def job(self):
        response = self.client.post('/api/v1/c3/jobs',headers=self.headers,
            json=dict(source_id='cache:val_fixture',camera_id='c3_fixture'))
        self.assertEqual(response.status_code,202,response.text)
        key = response.json()['job_id']
        end = time.monotonic()+4
        while time.monotonic()<end:
            detail = self.client.get(f'/api/v1/c3/jobs/{key}')
            self.assertEqual(detail.status_code,200,detail.text)
            if detail.json()['status'] not in ('queued','running'):
                self.assertEqual(detail.json()['status'],'completed',detail.text)
                return key,detail.json()
            time.sleep(.01)
        self.fail('Fixture job timed out')

    def test_c3_mode_starts_without_seeded_events_or_cameras(self):
        self.assertEqual(self.client.get('/api/v1/events').json(),[])
        self.assertEqual(self.client.get('/api/v1/cameras').json(),[])
        self.assertEqual(self.client.get('/api/v1/health').json()['mode'],'live')
        boot = self.client.get('/api/v1/c3/bootstrap').json()
        self.assertFalse(boot['capabilities']['live_frame'])
        self.assertEqual([s['source_id'] for s in boot['sources']],['cache:val_fixture'])

    def test_loopback_origin_and_token_mutation_controls(self):
        self.assertEqual(self.client.get('/api/v1/c3/bootstrap',headers={'Host':'evil.example'}).status_code,403)
        self.assertEqual(self.client.post('/api/v1/c3/jobs',json={}).status_code,403)
        self.assertEqual(self.client.post('/api/v1/c3/jobs',json={},
            headers=dict(self.headers,Origin='https://evil.example')).status_code,403)
        self.assertEqual(self.client.post('/api/v1/c3/jobs',json=[],headers=self.headers).status_code,400)

    def test_upload_bounds_test_duplicate_and_content_reuse(self):
        headers = dict(self.headers,**{'X-C3-File-Name':'sample.mp4'})
        one = self.client.post('/api/v1/c3/uploads',content=b'video',headers=headers)
        two = self.client.post('/api/v1/c3/uploads',content=b'video',headers=headers)
        self.assertEqual(one.status_code,201,one.text)
        self.assertEqual(one.json()['source_id'],two.json()['source_id'])
        self.assertEqual(self.client.post('/api/v1/c3/uploads',content=self.test_bytes,headers=headers).status_code,400)
        self.assertEqual(self.client.post('/api/v1/c3/uploads',content=b'x'*129,headers=headers).status_code,413)
        bad = dict(self.headers,**{'X-C3-File-Name':'..%2Fescape.mp4'})
        self.assertEqual(self.client.post('/api/v1/c3/uploads',content=b'video',headers=bad).status_code,400)

    def test_jobs_materialize_valid_events_once_and_register_camera(self):
        key,detail = self.job()
        self.assertEqual(len(detail['events']),1)
        event = IncidentEvent.model_validate(detail['events'][0])
        self.assertFalse(event.score_calibrated)
        self.assertEqual(event.component_id.value,'component_3')
        self.client.get(f'/api/v1/c3/jobs/{key}')
        self.assertEqual(len(self.client.get('/api/v1/events').json()),1)
        self.assertEqual(len(self.client.get('/api/v1/events').json()[0]['audit']),1)
        self.assertEqual(self.client.get('/api/v1/cameras').json()[0]['camera_id'],'c3_fixture')

    def test_operator_actions_preserve_incidents_and_append_non_training_review(self):
        key,detail = self.job(); event = detail['events'][0]
        path = self.wb.folder(key)/'result/incidents.json'; original = path.read_bytes()
        for action,status in (('acknowledge','acknowledged'),('escalate','escalated'),('resolve','resolved')):
            response = self.client.patch('/api/v1/events/'+event['event_id'],headers=self.headers,
                json=dict(action=action,status=status,operator_id='qa_operator',note='Fixture review only'))
            self.assertEqual(response.status_code,200,response.text)
        reviews = self.wb.reviews(key)
        self.assertEqual([r['status'] for r in reviews],['acknowledged']*3)
        self.assertTrue(all(r['eligible_for_training'] is False for r in reviews))
        self.assertEqual(path.read_bytes(),original)
        current = self.client.get('/api/v1/events/'+event['event_id']).json()
        self.assertEqual(current['status'],'resolved')
        self.assertEqual(len(current['audit']),4)
        self.assertTrue((self.state/'events_store.json').is_file())
        reloaded = EventStore(persist=True,seed=False,persist_file=self.state/'events_store.json')
        self.assertEqual(reloaded.get_event(event['event_id']).status.value,'resolved')
        mismatch = self.client.patch('/api/v1/events/'+event['event_id'],headers=self.headers,
            json=dict(action='acknowledge',status='false_positive'))
        self.assertEqual(mismatch.status_code,400)

    def test_media_ranges_export_and_traversal(self):
        key,detail = self.job()
        response = self.client.get(detail['media_url'],headers={'Range':'bytes=2-5'})
        self.assertEqual((response.status_code,response.content),(206,b'2345'))
        bad = self.client.get(f'/api/v1/c3/media/{key}/%2e%2e%2fjob.json')
        self.assertEqual(bad.status_code,400)
        archive = self.client.get(detail['export_url'])
        self.assertEqual(archive.status_code,200)
        names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
        self.assertIn('result/demo.mp4',names)
        self.assertFalse(any('uploads/' in name for name in names))
        self.assertEqual(self.client.get('/api/v1/not-a-route').status_code,404)

    def test_c3_runs_without_optional_c4_detector_package(self):
        registry = self.client.get('/api/v1/detect/detectors').json()
        self.assertEqual(registry,dict(video=[],frame=[]))
        response = self.client.post('/api/v1/detect/frame',content=b'fixture')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['events'],[])
        self.assertEqual(response.json()['detectors_run'],[])
        self.assertEqual(response.json()['note'],'No frame detectors are registered yet.')
        self.assertTrue(self.client.get('/api/v1/health').json()['ok'])
        self.assertFalse(self.client.get('/api/v1/c3/bootstrap').json()['capabilities']['live_frame'])

    def test_c1_c2_c4_pipeline_post_and_operator_actions_remain_available(self):
        for number,kind in ((1,'tailgating'),(2,'loitering'),(4,'fall')):
            with self.subTest(component=number):
                event = fixture_adapter([dict(event_id='external')],{},'external')[0]
                event.update(event_id=f'component_{number}-external',component_id=f'component_{number}',incident_type=kind)
                submitted = self.client.post('/api/v1/events',json=event)
                self.assertEqual(submitted.status_code,201,submitted.text)
                self.assertEqual(self.client.post('/api/v1/events',json=event).status_code,409)
                action = self.client.patch('/api/v1/events/'+event['event_id'],json=dict(action='acknowledge',status='acknowledged',operator_id='team_qa'))
                self.assertEqual(action.status_code,200,action.text)
                self.assertEqual(len(action.json()['audit']),2)
                self.assertEqual(self.client.get('/api/v1/events',params={'component_id':f'component_{number}'}).json()[0]['status'],'acknowledged')

    def test_registered_team_video_and_frame_detectors_still_store_results(self):
        from unittest.mock import patch
        from detect import VIDEO_DETECTORS, FRAME_DETECTORS
        def fake(*args):
            event = fixture_adapter([dict(event_id='team')],{},str(len(args)))[0]
            event.update(component_id='component_4',event_id=f'component_4-{len(args)}',incident_type='fall',media_ref=None)
            return [IncidentEvent.model_validate(event)]
        with patch('detect.UPLOAD_DIR', self.root/'team_uploads'), patch.dict(VIDEO_DETECTORS,{'team_fixture':fake},clear=True), patch.dict(FRAME_DETECTORS,{'team_fixture':fake},clear=True):
            for endpoint in ('video','frame'):
                response = self.client.post('/api/v1/detect/'+endpoint,content=b'fixture',params={'filename':'qa.mp4'})
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(response.json()['detectors_run'],['team_fixture'])
                self.assertEqual(response.json()['events'][0]['component_id'],'component_4')

    def test_serves_imported_frontend_without_replacing_api(self):
        self.assertIn('Imported dashboard',self.client.get('/').text)
        self.assertEqual(self.client.get('/missing.js').status_code,404)
        self.assertEqual(self.client.get('/api/v1/events').json(),[])

    def test_original_mock_mode_remains_explicitly_available(self):
        app = create_app(c3_enabled=False,frontend_dir=self.frontend)
        with TestClient(app,base_url='http://127.0.0.1:8000') as client:
            self.assertEqual(client.get('/api/v1/health').json()['mode'],'mock')
            self.assertGreater(len(client.get('/api/v1/events').json()),0)



if __name__ == '__main__':
    unittest.main()
