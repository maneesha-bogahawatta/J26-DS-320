"""Local workbench integration tests; inference is a fixture, not a model claim."""
import csv
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import zipfile

from src.demo_server import DemoHTTPServer, DemoWorkbench, write_json


class DemoServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.test_bytes = b'locked-test-source'
        locked = self.root/'data/foundation/v1/ubi_locked.csv'
        locked.parent.mkdir(parents=True)
        with locked.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=['video_id','split','content_sha256'])
            writer.writeheader()
            writer.writerows([dict(video_id='val_1',split='val',content_sha256='v'),
                dict(video_id='test_1',split='test',content_sha256=hashlib.sha256(self.test_bytes).hexdigest())])
        index = self.root/'data/pose_cache/advanced_v1/val_index.json'
        index.parent.mkdir(parents=True)
        write_json(index,dict(entries={'val_1':{},'test_1':{}}))
        self.called = []
        self.app = DemoWorkbench(self.root,runner=self.fake_inference,max_upload_bytes=128)

    def tearDown(self):
        self.app.close()
        self.tmp.cleanup()

    def fake_inference(self,job,out):
        self.called.append(job)
        out.mkdir()
        write_json(out/'prediction.json',dict(source_context=[5,13],
            operational_alerts_enabled=False,verified_early_aggression_claim=False,
            timing=dict(valid_decisions=2,decisions=3)))
        write_json(out/'incidents.json',dict(schema='c3_incident_envelope_v1',incidents=[
            dict(event_id='event-1',incident_kind='potential_aggression',model_score=.7,
                video_timestamp_s=7.,research_only=True,shared_event=None)]))
        write_json(out/'notifications.json',dict(notifications=[]))
        (out/'scores.csv').write_text('t_s,weak_onset_score,fight_score\n7,0.7,0.2\n',encoding='utf-8')
        (out/'demo.mp4').write_bytes(b'0123456789')

    def done_job(self):
        job = self.app.submit(dict(source_id='cache:val_1'))
        deadline = time.monotonic()+3
        while self.app.details(job['job_id'])['status'] in ('queued','running'):
            if time.monotonic() > deadline:
                self.fail('Fixture inference worker did not finish')
            time.sleep(.005)
        self.assertEqual(self.app.details(job['job_id'])['status'],'completed')
        return job['job_id']

    def test_cache_source_list_excludes_locked_test_without_opening_test_media(self):
        sources = self.app.public_sources()
        self.assertEqual([source['source_id'] for source in sources],['cache:val_1'])
        with self.assertRaisesRegex(ValueError,'known'):
            self.app.submit(dict(source_id='cache:test_1'))

    def test_upload_reuses_content_address_and_rejects_test_duplicate(self):
        one = self.app.upload('camera.mp4',io.BytesIO(b'video'),5)
        two = self.app.upload('other_name.mov',io.BytesIO(b'video'),5)
        self.assertEqual(one['source_id'],two['source_id'])
        self.assertEqual(len(list(self.app.uploads.glob('*/source.mp4'))),1)
        with self.assertRaisesRegex(ValueError,'locked test'):
            self.app.upload('renamed.mp4',io.BytesIO(self.test_bytes),len(self.test_bytes))
        self.assertEqual(list(self.app.uploads.glob('pending_*')),[])

    def test_upload_rejects_traversal_extension_size_and_truncation(self):
        for name in ('../video.mp4','C:\\video.mp4','video.py',''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.app.upload(name,io.BytesIO(b'data'),4)
        for length in (0,129):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.app.upload('video.mp4',io.BytesIO(b'data'),length)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            self.app.upload('video.mp4',io.BytesIO(b'data'),8)
        self.assertEqual(list(self.app.uploads.glob('pending_*')),[])

    def test_jobs_validate_duration_camera_and_source_types(self):
        for body in (dict(source_id={}),dict(source_id='cache:val_1',seconds=float('nan')),
                     dict(source_id='cache:val_1',seconds=True),dict(source_id='cache:val_1',seconds=61),
                     dict(source_id='cache:val_1',camera_id='../cam'),
                     dict(source_id='cache:val_1',research_alerts='yes')):
            with self.subTest(body=body),self.assertRaises(ValueError):
                self.app.submit(body)

    def test_async_inference_and_versioned_results(self):
        first,second = self.done_job(),self.done_job()
        self.assertNotEqual(first,second)
        result = self.app.details(first)
        self.assertEqual(result['scores'][0]['t_s'],'7')
        self.assertEqual(result['incidents'][0]['shared_event'],None)
        self.assertIn('demo.mp4',result['files'])
        self.assertFalse(self.called[0]['research_alerts'])
        self.assertFalse(result['operational_alerts_enabled'])

    def test_reviews_are_append_only_and_do_not_modify_incidents_or_labels(self):
        job_id = self.done_job()
        original = (self.app.folder(job_id)/'result/incidents.json').read_bytes()
        for status in ('confirmed','dismissed'):
            self.app.add_review(job_id,dict(event_id='event-1',status=status,reviewer='human',note='Observed context'))
        reviews = self.app.reviews(job_id)
        self.assertEqual([review['status'] for review in reviews],['confirmed','dismissed'])
        self.assertTrue(all(not review['eligible_for_training'] for review in reviews))
        self.assertEqual(original,(self.app.folder(job_id)/'result/incidents.json').read_bytes())
        effective = self.app.details(job_id)['incidents'][0]
        self.assertEqual(effective['operator_review_status'],'dismissed')
        self.assertEqual(effective['source_operator_review_status'],'pending')
        self.assertEqual(effective['latest_operator_review_id'],reviews[-1]['review_id'])
        for body in (dict(event_id='missing',status='confirmed',reviewer='human'),
                     dict(event_id={},status='confirmed',reviewer='human'),
                     dict(event_id='event-1',status={},reviewer='human'),
                     dict(event_id='event-1',status='confirmed',reviewer='')):
            with self.subTest(body=body),self.assertRaises(ValueError):
                self.app.add_review(job_id,body)

    def test_export_contains_results_and_review_history_but_no_source(self):
        job_id = self.done_job()
        self.app.add_review(job_id,dict(event_id='event-1',status='uncertain',reviewer='human'))
        archive = zipfile.ZipFile(io.BytesIO(self.app.export(job_id)))
        self.assertIn('result/demo.mp4',archive.namelist())
        self.assertIn('incident_reviews.jsonl',archive.namelist())
        self.assertFalse(any('uploads' in name or 'source.mp4' in name for name in archive.namelist()))

    def test_worker_failure_is_visible_and_preserves_job_record(self):
        def fail(job,out):
            raise RuntimeError('fixture inference failed')
        self.app.runner = fail
        job = self.app.submit(dict(source_id='cache:val_1'))
        deadline = time.monotonic()+3
        while self.app.details(job['job_id'])['status'] in ('queued','running'):
            if time.monotonic()>deadline: self.fail('Failure worker timed out')
            time.sleep(.005)
        result = self.app.details(job['job_id'])
        self.assertEqual(result['status'],'failed')
        self.assertIn('fixture inference failed',result['error'])

    def test_restart_marks_unfinished_job_interrupted(self):
        folder = self.app.jobs_dir/('a'*32); folder.mkdir()
        write_json(folder/'job.json',dict(job_id='a'*32,status='running'))
        another = DemoWorkbench(self.root,runner=self.fake_inference)
        try:
            self.assertEqual(another.jobs['a'*32]['status'],'interrupted')
        finally:
            another.close()

    def test_loopback_bind_is_required(self):
        with self.assertRaisesRegex(ValueError,'127.0.0.1'):
            DemoHTTPServer(('0.0.0.0',0),self.app)

    def test_http_security_upload_job_review_export_and_video_range(self):
        server = DemoHTTPServer(('127.0.0.1',0),self.app)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        port = server.server_address[1]
        def call(method,path,body=None,headers=None):
            connection = http.client.HTTPConnection('127.0.0.1',port,timeout=3)
            connection.request(method,path,body=body,headers=headers or {})
            response = connection.getresponse(); raw = response.read()
            result = (response.status,raw,dict(response.getheaders()))
            connection.close(); return result
        try:
            status,raw,_ = call('GET','/api/bootstrap')
            self.assertEqual(status,200); token = json.loads(raw)['token']
            self.assertEqual(call('GET','/api/bootstrap',headers={'Host':'evil.example'})[0],403)
            self.assertEqual(call('POST','/api/jobs','{}',{'Content-Type':'application/json'})[0],403)
            bad = {'X-C3-Token':token,'Origin':'https://evil.example','Content-Type':'application/json'}
            self.assertEqual(call('POST','/api/jobs','{}',bad)[0],403)
            good = {'X-C3-Token':token,'Content-Type':'application/json'}
            self.assertEqual(call('POST','/api/jobs','[]',good)[0],400)
            status,raw,_ = call('POST','/api/uploads',b'data',dict(good,**{'X-C3-File-Name':'camera.mp4'}))
            self.assertEqual(status,201)
            self.assertTrue(json.loads(raw)['source_id'].startswith('upload:'))
            job_id = self.done_job()
            status,raw,headers = call('GET',f'/files/{job_id}/demo.mp4',headers={'Range':'bytes=2-5'})
            self.assertEqual((status,raw),(206,b'2345'))
            self.assertEqual(headers['Content-Range'],'bytes 2-5/10')
            self.assertEqual(call('GET',f'/files/{job_id}/demo.mp4',headers={'Range':'bytes=99-'})[0],416)
            self.assertEqual(call('GET',f'/files/{job_id}/../../job.json')[0],400)
            self.assertEqual(call('GET','/report/../../README.md')[0],400)
            status,_,_ = call('POST',f'/api/jobs/{job_id}/reviews',json.dumps(dict(event_id='event-1',status='uncertain',reviewer='operator')),good)
            self.assertEqual(status,201)
            status,raw,_ = call('GET',f'/api/jobs/{job_id}/export.zip')
            self.assertEqual(status,200)
            self.assertIn('incident_reviews.jsonl',zipfile.ZipFile(io.BytesIO(raw)).namelist())
            self.assertEqual(call('OPTIONS','/api/jobs')[0],405)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
