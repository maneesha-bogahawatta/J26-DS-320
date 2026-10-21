"""Additive C3 routes for the original shared FastAPI application.

The existing C3 workbench owns inference, caches, artifacts and operator reviews.
The shared store owns contract v1.1 events; C1/C2/C4 routes remain independent.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import re
import secrets
import sys
import tempfile
from urllib.parse import unquote, urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from c3_bridge import C3Bridge

REPOSITORY = Path(__file__).resolve().parents[2]
C3_ROOT = REPOSITORY / 'components/c3_aggression_detection'


def install_c3(app, store, *, state_dir=None, workbench=None, adapter=None):
    work = Path(state_dir or C3_ROOT/'outputs/original_dashboard_integration_v1/workbench').resolve()
    if workbench is None:
        if not work.is_relative_to(C3_ROOT.resolve()):
            raise ValueError('C3 state must remain within Component 03')
        if str(C3_ROOT) not in sys.path:
            sys.path.insert(0, str(C3_ROOT))
        from src.demo_server import DemoWorkbench
        workbench = DemoWorkbench(C3_ROOT, work)
    bridge = C3Bridge(workbench, store, adapter)
    app.state.c3 = bridge
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            try:
                yield
            finally:
                await run_in_threadpool(bridge.close)
    app.router.lifespan_context = lifespan

    @app.middleware('http')
    async def c3_local_mutations(request: Request, call_next):
        path = request.url.path
        protected = path.startswith('/api/v1/c3/')
        # Protect C3-owned operator actions without changing other pipelines' API.
        protected |= request.method == 'PATCH' and path.startswith('/api/v1/events/component_3-')
        if protected:
            host = request.headers.get('host', '')
            if not re.fullmatch(r'(127\.0\.0\.1|localhost)(:\d{1,5})?', host):
                return JSONResponse(status_code=403, content={'detail': 'C3 requires a loopback Host'})
            origin = request.headers.get('origin')
            if origin:
                try:
                    parsed = urlsplit(origin)
                    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.port not in (request.url.port or 80, 5173, 4173):
                        raise ValueError('Cross-origin request')
                except ValueError:
                    return JSONResponse(status_code=403, content={'detail': 'C3 rejects this Origin'})
            if request.method not in ('GET', 'HEAD', 'OPTIONS') and not secrets.compare_digest(request.headers.get('x-c3-token', ''), workbench.token):
                return JSONResponse(status_code=403, content={'detail': 'A current X-C3-Token from /api/v1/c3/bootstrap is required'})
        response = await call_next(request)
        if protected:
            response.headers['Cache-Control'] = 'no-store'
            response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/api/v1/c3/bootstrap', tags=['C3 research'])
    def bootstrap():
        return bridge.bootstrap()

    @app.post('/api/v1/c3/uploads', status_code=201, tags=['C3 research'])
    async def upload(request: Request):
        name = unquote(request.headers.get('x-c3-file-name', ''))
        length = request.headers.get('content-length')
        if length is not None and (not length.isdigit() or not 0 < int(length) <= workbench.max_upload_bytes):
            raise HTTPException(413, 'Video upload is empty or exceeds the configured limit')
        count = 0
        with tempfile.SpooledTemporaryFile(max_size=8*1024*1024, dir=workbench.uploads) as stream:
            async for chunk in request.stream():
                count += len(chunk)
                if count > workbench.max_upload_bytes:
                    raise HTTPException(413, 'Video upload exceeds the configured limit')
                stream.write(chunk)
            stream.seek(0)
            return await run_in_threadpool(workbench.upload, name, stream, count)

    @app.post('/api/v1/c3/jobs', status_code=202, tags=['C3 research'])
    async def submit_job(request: Request):
        if len(await request.body()) > 65536:
            raise HTTPException(413, 'Job request is too large')
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, 'A JSON object is required')
        return bridge.submit(body)

    @app.get('/api/v1/c3/jobs/{job_id}', tags=['C3 research'])
    def job(job_id: str):
        return bridge.details(job_id)

    @app.get('/api/v1/c3/jobs/{job_id}/export.zip', tags=['C3 research'])
    def export(job_id: str):
        bridge.details(job_id)
        return Response(workbench.export(job_id), media_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="c3_{job_id}.zip"'})

    @app.get('/api/v1/c3/media/{job_id}/{filename:path}', tags=['C3 research'])
    def media(job_id: str, filename: str):
        target = bridge.media_file(job_id, filename)
        if not target.is_file():
            raise HTTPException(404, 'Result file not found')
        return FileResponse(target)
    return bridge
