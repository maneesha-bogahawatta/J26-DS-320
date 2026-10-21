"""Loopback-only C3 workbench over the existing video research pipeline.

This server never trains models or changes source annotations. Incident operator
reviews are append-only demonstration records, not independent training labels.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import unquote, urlparse
import uuid
import zipfile


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
REVIEW_STATUSES = {'pending', 'acknowledged', 'confirmed', 'dismissed', 'uncertain'}
FILE_EXTENSIONS = {'.mp4', '.csv', '.json', '.jpg', '.jpeg', '.png', '.txt', '.log'}
ID_PATTERN = re.compile(r'^[a-f0-9]{32}$')


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path must remain inside its result directory')
    return path


class DemoWorkbench:
    """One serial inference worker, reusable uploads and versioned job results."""
    def __init__(self, component_root, work_dir=None, runner=None, max_upload_bytes=256*1024*1024):
        self.root = Path(component_root).resolve()
        self.work = within(work_dir or self.root/'outputs/demo_workbench_v1', self.root)
        self.uploads = self.work/'uploads'
        self.jobs_dir = self.work/'jobs'
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max_upload_bytes
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='c3-inference')
        self.runner = runner or self._run_inference
        self.jobs = {}
        self.sources = {}
        self.test_hashes = set()
        self.cached_index = self.root/'data/pose_cache/advanced_v1/val_index.json'
        self._load_sources()
        # A previous server cannot still own these jobs after process restart.
        for folder in self.jobs_dir.iterdir():
            if not folder.is_dir() or not ID_PATTERN.fullmatch(folder.name):
                continue
            state = read_json(folder/'job.json')
            if state:
                if state['status'] in ('queued', 'running'):
                    state.update(status='interrupted', error='Previous server stopped; submit a new job to retry.')
                    write_json(folder/'job.json', state)
                self.jobs[folder.name] = state

    def _load_sources(self):
        locked_path = self.root/'data/foundation/v1/ubi_locked.csv'
        locked = {}
        if locked_path.is_file():
            with locked_path.open(encoding='utf-8-sig', newline='') as stream:
                locked = {row['video_id']: row for row in csv.DictReader(stream)}
            self.test_hashes = {row['content_sha256'] for row in locked.values() if row['split'] == 'test'}
        index = read_json(self.cached_index, {})
        for key in index.get('entries', {}):
            row = locked.get(key)
            if row and row['split'] in ('train', 'val'):
                self.sources['cache:'+key] = dict(source_id='cache:'+key, name=key,
                    kind='cached_context', video_id=key, split=row['split'])
        for folder in self.uploads.iterdir():
            meta = read_json(folder/'upload.json') if folder.is_dir() else None
            if meta and meta.get('sha256') not in self.test_hashes:
                source = folder/meta['stored_name']
                if source.is_file():
                    self.sources[meta['source_id']] = dict(meta, path=str(source.resolve()))

    def public_sources(self):
        with self.lock:
            return [{k:v for k,v in source.items() if k not in ('path', 'stored_name')}
                    for source in sorted(self.sources.values(), key=lambda source: source['name'])]

    def upload(self, name, stream, length):
        if not isinstance(name, str) or len(name) > 180 or any(c in name for c in ('/', '\\', '\x00')):
            raise ValueError('Use a plain video filename without a directory')
        if name in ('', '.', '..') or Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError('Supported video extensions: mp4, avi, mov, mkv, webm, m4v')
        if not isinstance(length, int) or not 0 < length <= self.max_upload_bytes:
            raise ValueError('Video upload is empty or exceeds the configured upload limit')
        staging = self.uploads/('pending_'+uuid.uuid4().hex)
        digest = hashlib.sha256()
        try:
            with staging.open('xb') as dest:
                remaining = length
                while remaining:
                    block = stream.read(min(1024*1024, remaining))
                    if not block:
                        raise ValueError('Incomplete video upload')
                    digest.update(block); dest.write(block); remaining -= len(block)
            value = digest.hexdigest()
            if value in self.test_hashes:
                raise ValueError('This video matches a locked test source; development inference is blocked')
            folder = self.uploads/value
            with self.lock:
                # Identical uploaded bytes always use the same source path so
                # existing pose-cache identity remains reusable on later jobs.
                existing = read_json(folder/'upload.json')
                if existing:
                    source = dict(existing, path=str((folder/existing['stored_name']).resolve()))
                    if not Path(source['path']).is_file():
                        raise ValueError('Stored upload is incomplete')
                else:
                    folder.mkdir(exist_ok=True)
                    filename = 'source'+Path(name).suffix.lower()
                    staging.rename(folder/filename)
                    source = dict(source_id='upload:'+value, name=name, kind='uploaded_video',
                        sha256=value, size_bytes=length, stored_name=filename, uploaded_at=utc_now())
                    write_json(folder/'upload.json', source)
                    source['path'] = str((folder/filename).resolve())
                self.sources[source['source_id']] = source
            return {k:v for k,v in source.items() if k not in ('path', 'stored_name')}
        finally:
            if staging.exists():
                staging.unlink()

    def submit(self, body):
        source_id = body.get('source_id')
        if not isinstance(source_id, str) or source_id not in self.sources:
            raise ValueError('Select a known uploaded video or non-test cached source')
        seconds = body.get('seconds', 8)
        if isinstance(seconds, bool) or not isinstance(seconds, (int,float)) or not math.isfinite(seconds) or not 1 <= seconds <= 60:
            raise ValueError('Choose an extraction duration from 1 to 60 seconds')
        camera = body.get('camera_id', 'c3_demo')
        if not isinstance(camera, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', camera):
            raise ValueError('Camera ID must contain 1–80 letters, digits, underscores or hyphens')
        research = body.get('research_alerts', False)
        if not isinstance(research, bool):
            raise ValueError('research_alerts must be a JSON boolean')
        with self.lock:
            if sum(j['status'] in ('queued','running') for j in self.jobs.values()) >= 4:
                raise ValueError('The four-job queue is full; wait for an existing job')
            job_id = uuid.uuid4().hex
            folder = self.jobs_dir/job_id
            folder.mkdir(exist_ok=False)
            job = dict(job_id=job_id, status='queued', source_id=source_id,
                source_name=self.sources[source_id]['name'], seconds=float(seconds),
                camera_id=camera, research_alerts=research, created_at=utc_now(),
                operational_alerts_enabled=False)
            self.jobs[job_id] = job
            write_json(folder/'job.json', job)
            self.executor.submit(self._execute, job_id)
            return dict(job)

    def _execute(self, job_id):
        folder = self.jobs_dir/job_id
        with self.lock:
            job = self.jobs[job_id]
            job.update(status='running', started_at=utc_now())
            write_json(folder/'job.json', job)
        try:
            self.runner(dict(job), folder/'result')
            if not (folder/'result/prediction.json').is_file():
                raise RuntimeError('Inference returned without prediction.json')
            with self.lock:
                job.update(status='completed', completed_at=utc_now())
        except Exception as exc:
            with self.lock:
                job.update(status='failed', completed_at=utc_now(), error=str(exc)[:1000])
        finally:
            with self.lock:
                write_json(folder/'job.json', job)

    def _run_inference(self, job, output):
        source = self.sources[job['source_id']]
        args = [sys.executable, '-m', 'src.predict_advanced', '--out-dir', str(output),
                '--render', '--camera-id', job['camera_id']]
        if source['kind'] == 'cached_context':
            args += ['--video-id', source['video_id'], '--index', str(self.cached_index)]
        else:
            args += ['--video', source['path'], '--seconds', str(job['seconds'])]
        if job['research_alerts']:
            args.append('--research-alerts')
        env = os.environ.copy()
        local_paths = [str(self.root), str(self.root/'legacy')]
        isolated_deps = self.root/'data/phase1_work/testdeps'
        if isolated_deps.is_dir():
            local_paths.insert(0, str(isolated_deps))
        env['PYTHONPATH'] = os.pathsep.join(local_paths+[env.get('PYTHONPATH','')])
        env['PYTHONDONTWRITEBYTECODE'] = '1'; env['YOLO_AUTOINSTALL'] = 'false'
        env['YOLO_CONFIG_DIR'] = str(self.root/'data/phase1_work/ultralytics')
        with (output.parent/'inference.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(args, cwd=self.root, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=3600, check=False)
        if result.returncode:
            raise RuntimeError(f'Inference exited with code {result.returncode}; see inference.log')

    def folder(self, job_id):
        if not ID_PATTERN.fullmatch(job_id) or job_id not in self.jobs:
            raise KeyError('Unknown job')
        return within(self.jobs_dir/job_id, self.jobs_dir)

    def details(self, job_id):
        folder = self.folder(job_id)
        with self.lock:
            result = dict(self.jobs[job_id])
            result['reviews'] = self.reviews(job_id)
        if result['status'] == 'completed':
            output = folder/'result'
            result['prediction'] = read_json(output/'prediction.json', {})
            incidents = read_json(output/'incidents.json', {'incidents':[]})
            result['incidents'] = incidents.get('incidents',[]) if isinstance(incidents,dict) else incidents
            # The generated model artifact stays immutable. API consumers see
            # the latest operator status derived from the append-only history.
            latest_reviews = {review['event_id']: review for review in result['reviews']}
            for incident in result['incidents']:
                incident['source_operator_review_status'] = incident.get('operator_review_status','pending')
                review = latest_reviews.get(incident['event_id'])
                if review:
                    incident['operator_review_status'] = review['status']
                    incident['latest_operator_review_id'] = review['review_id']
            result['notifications'] = read_json(output/'notifications.json', {})
            scores = output/'scores.csv'
            result['scores'] = []
            if scores.is_file():
                with scores.open(encoding='utf-8-sig', newline='') as stream:
                    result['scores'] = list(csv.DictReader(stream))
            result['files'] = sorted(str(p.relative_to(output)).replace('\\','/')
                for p in output.rglob('*') if p.is_file() and p.suffix.lower() in FILE_EXTENSIONS)
        return result

    def reviews(self, job_id):
        path = self.folder(job_id)/'incident_reviews.jsonl'
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []

    def add_review(self, job_id, body):
        details = self.details(job_id)
        if details['status'] != 'completed':
            raise ValueError('Only completed inference incidents can be reviewed')
        known = {item['event_id'] for item in details['incidents']}
        if not isinstance(body.get('event_id'), str) or body['event_id'] not in known:
            raise ValueError('Unknown incident ID')
        status, reviewer, note = body.get('status'), body.get('reviewer'), body.get('note','')
        if not isinstance(status,str) or status not in REVIEW_STATUSES:
            raise ValueError('Unsupported incident review status')
        if not isinstance(reviewer,str) or not 1 <= len(reviewer.strip()) <= 120:
            raise ValueError('A reviewer identifier of 1–120 characters is required')
        if not isinstance(note,str) or len(note) > 4000:
            raise ValueError('Review note must contain at most 4000 characters')
        review = dict(review_id=uuid.uuid4().hex, event_id=body['event_id'], status=status,
            reviewer=reviewer.strip(), note=note, timestamp=utc_now(),
            record_kind='operator_incident_review', eligible_for_training=False,
            verified_precursor_annotation=False)
        with self.lock:
            with (self.folder(job_id)/'incident_reviews.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(review, ensure_ascii=True, allow_nan=False)+'\n')
        return review

    def export(self, job_id):
        folder = self.folder(job_id)
        if self.jobs[job_id]['status'] != 'completed':
            raise ValueError('Only completed jobs can be exported')
        content = io.BytesIO()
        with self.lock, zipfile.ZipFile(content,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder/'job.json','job.json')
            for path in (folder/'result').rglob('*'):
                if path.is_file() and path.suffix.lower() in FILE_EXTENSIONS:
                    within(path,folder/'result')
                    archive.write(path, str(Path('result')/path.relative_to(folder/'result')))
            for name in ('incident_reviews.jsonl','inference.log'):
                if (folder/name).is_file():
                    archive.write(folder/name,name)
            archive.writestr('README.txt', 'C3 research demonstration export. Weak-model scores are uncalibrated.\n'
                'Operator incident reviews are not independent precursor training labels.\n'
                'Effective current review status is the last record for an event in incident_reviews.jsonl; generated model artifacts are immutable.\n'
                'Original uploaded source video and dataset media are excluded.\n')
        return content.getvalue()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, workbench):
        if address[0] != '127.0.0.1':
            raise ValueError('The C3 demo server must bind to 127.0.0.1')
        self.workbench = workbench
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server_version = 'C3LocalDemo/1'

    def _trusted(self, mutation=False):
        port = self.server.server_address[1]
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:
            raise PermissionError('Untrusted Host header')
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://'+host for host in allowed}:
            raise PermissionError('Cross-origin requests are not accepted')
        if mutation and not secrets.compare_digest(self.headers.get('X-C3-Token',''),self.server.workbench.token):
            raise PermissionError('A current local workbench token is required')

    def _send(self, status, data, mime='application/json', extra=None):
        if mime == 'application/json':
            data = json.dumps(data, ensure_ascii=True, allow_nan=False).encode()
        elif isinstance(data,str):
            data = data.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        for key,value in (extra or {}).items():
            self.send_header(key,value)
        self.end_headers(); self.wfile.write(data)

    def _error(self, exc):
        self.close_connection = True
        status = 403 if isinstance(exc,PermissionError) else 404 if isinstance(exc,(KeyError,FileNotFoundError)) else 400
        self._send(status,dict(error=str(exc)))

    def _body_length(self, maximum):
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('Chunked request bodies are not supported')
        length = int(self.headers.get('Content-Length','0'))
        if not 0 < length <= maximum:
            raise ValueError('Request is empty or exceeds the configured size limit')
        return length

    def _json(self):
        length = self._body_length(65536)
        body = json.loads(self.rfile.read(length))
        if not isinstance(body,dict):
            raise ValueError('A JSON object is required')
        return body

    def _file(self, path):
        if not path.is_file():
            raise FileNotFoundError('File not found')
        size = path.stat().st_size
        begin, end, status = 0, size-1, 200
        value = self.headers.get('Range')
        if value:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)',value)
            if not match:
                self._send(416,dict(error='Only a single byte range is supported')); return
            begin = int(match[1]); end = min(int(match[2]) if match[2] else size-1,size-1)
            if begin > end or begin >= size:
                self._send(416,dict(error='Range outside file')); return
            status = 206
        self.send_response(status)
        self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(max(0,end-begin+1)))
        self.send_header('Accept-Ranges','bytes'); self.send_header('X-Content-Type-Options','nosniff')
        if status == 206:
            self.send_header('Content-Range',f'bytes {begin}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(begin); remaining = end-begin+1
            while remaining > 0:
                block = stream.read(min(1024*1024,remaining))
                if not block: break
                self.wfile.write(block); remaining -= len(block)

    def do_GET(self):
        try:
            self._trusted()
            app = self.server.workbench
            path = unquote(urlparse(self.path).path)
            if path == '/':
                self._send(200,Path(__file__).with_name('demo_ui.html').read_text(encoding='utf-8'),'text/html; charset=utf-8')
            elif path == '/api/bootstrap':
                with app.lock:
                    jobs = [dict(job) for job in app.jobs.values()]
                self._send(200,dict(token=app.token,sources=app.public_sources(),jobs=jobs,
                    max_upload_bytes=app.max_upload_bytes,verified_early_prediction=False,
                    operational_alerts_enabled=False))
            elif path.startswith('/api/jobs/'):
                parts = path.strip('/').split('/')
                if len(parts) == 4 and parts[3] == 'export.zip':
                    self._send(200,app.export(parts[2]),'application/zip',{'Content-Disposition':f'attachment; filename="c3_{parts[2]}.zip"'})
                elif len(parts) == 3:
                    self._send(200,app.details(parts[2]))
                else: raise FileNotFoundError('Unknown route')
            elif path.startswith('/files/'):
                parts = path.strip('/').split('/')
                if len(parts) < 3: raise FileNotFoundError('Unknown result file')
                base = app.folder(parts[1])
                relative = '/'.join(parts[2:])
                if relative == 'inference.log':
                    target = base/'inference.log'
                else:
                    target = within(base/'result'/relative,base/'result')
                if target.suffix.lower() not in FILE_EXTENSIONS:
                    raise PermissionError('Unsupported result-file type')
                self._file(target)
            elif path.startswith('/report/'):
                base = app.root/'outputs/final_c3_v2'
                target = within(base/(path[len('/report/'):] or 'index.html'),base)
                if target.suffix.lower() not in FILE_EXTENSIONS | {'.html'}:
                    raise PermissionError('Unsupported report-file type')
                self._file(target)
            else:
                raise FileNotFoundError('Unknown route')
        except (ValueError,KeyError,PermissionError,FileNotFoundError,json.JSONDecodeError) as exc:
            self._error(exc)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self._trusted(mutation=True)
            app = self.server.workbench
            path = unquote(urlparse(self.path).path)
            if path == '/api/uploads':
                length = self._body_length(app.max_upload_bytes)
                name = unquote(self.headers.get('X-C3-File-Name',''))
                self._send(201,app.upload(name,self.rfile,length))
            elif path == '/api/jobs':
                self._send(202,app.submit(self._json()))
            elif path.startswith('/api/jobs/') and path.endswith('/reviews'):
                parts = path.strip('/').split('/')
                if len(parts) != 4: raise FileNotFoundError('Unknown route')
                self._send(201,app.add_review(parts[2],self._json()))
            else: raise FileNotFoundError('Unknown route')
        except (ValueError,KeyError,PermissionError,FileNotFoundError,json.JSONDecodeError) as exc:
            self._error(exc)

    def do_OPTIONS(self):
        self._send(405,dict(error='Cross-origin access is disabled'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--work-dir',help='New workbench state, inside Component 03')
    parser.add_argument('--max-upload-mb',type=int,default=256)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 1 <= args.max_upload_mb <= 2048:
        parser.error('Choose port 1024–65535 and an upload limit of 1–2048 MB')
    app = DemoWorkbench(Path(__file__).resolve().parents[1],args.work_dir,
        max_upload_bytes=args.max_upload_mb*1024*1024)
    server = DemoHTTPServer(('127.0.0.1',args.port),app)
    print(f'C3 local workbench: http://127.0.0.1:{args.port} — Ctrl+C stops it.',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping server; waiting for any running inference to finish.',flush=True)
    finally:
        server.server_close(); app.close()


if __name__ == '__main__':
    main()
