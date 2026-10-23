"""C3-owned model jobs projected into the team dashboard's existing API.

No model, dataset, annotation or shared dataclass is modified. Operator actions
remain audit records and never become independent precursor training labels.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading
from typing import Callable, Any

from schemas import ActionRequest, CameraSource, IncidentEvent
from store import DuplicateEventError, EventNotFoundError, EventStore


class C3Bridge:
    def __init__(self, workbench: Any, store: EventStore, adapter: Callable | None = None):
        self.workbench, self.store, self.adapter = workbench, store, adapter
        self.lock = threading.RLock()
        self.materialized: set[str] = set()
        self.locations: dict[str, tuple[str, str]] = {}

    def bootstrap(self) -> dict:
        self.refresh()
        with self.workbench.lock:
            jobs = [dict(job) for job in self.workbench.jobs.values()]
        return dict(token=self.workbench.token, sources=self.workbench.public_sources(), jobs=jobs,
            max_upload_bytes=self.workbench.max_upload_bytes,
            capabilities=dict(video=True, cached_sources=True, live_frame=False, research_only=True),
            operational_alerts_enabled=False, verified_early_prediction=False,
            note='C3 weak-model research only. Recorded videos; scores uncalibrated; no operational alerts.')

    def upload(self, name: str, body: bytes) -> dict:
        return self.workbench.upload(name, BytesIO(body), len(body))

    def submit(self, body: dict) -> dict:
        return self.workbench.submit(body)

    def refresh(self):
        with self.workbench.lock:
            completed = [key for key, value in self.workbench.jobs.items() if value['status'] == 'completed']
        for key in completed:
            self.details(key)

    def details(self, job_id: str) -> dict:
        detail = self.workbench.details(job_id)
        detail['events'] = []
        detail['media_url'] = f'/api/v1/c3/media/{job_id}/demo.mp4'
        detail['export_url'] = f'/api/v1/c3/jobs/{job_id}/export.zip'
        detail['result_files'] = [dict(name=name, url=f'/api/v1/c3/media/{job_id}/{name}')
            for name in detail.get('files', [])]
        if detail['status'] != 'completed':
            return detail
        with self.lock:
            adapter = self.adapter
            if adapter is None:
                from src.dashboard_adapter import adapt_run
                adapter = adapt_run
            # Validate the full projection before storing any candidate.
            projected = adapter(detail.get('incidents', []), detail['prediction'], job_id)
            events = [IncidentEvent.model_validate(event) for event in projected]
            if job_id not in self.materialized:
                for event, incident in zip(events, detail.get('incidents', [])):
                    try:
                        self.store.add_event(event)
                    except DuplicateEventError:
                        pass
                    self.locations.setdefault(event.event_id, (job_id, incident['event_id']))
                self.materialized.add(job_id)
                self.store.upsert_camera(CameraSource(camera_id=detail['camera_id'],
                    label=f"C3 recorded result - {detail['source_name']}", media_ref=detail['media_url'],
                    kind='recorded_clip', source_note='C3 weak supervision; annotated model replay, not a live feed.'))
            detail['events'] = [self.store.get_event(event.event_id).model_dump(mode='json') for event in events]
        return detail

    def apply_action(self, event_id: str, action: ActionRequest) -> IncidentEvent:
        self.refresh()
        with self.lock:
            from schemas import ACTION_TARGET_STATUS
            if ACTION_TARGET_STATUS[action.action] != action.status:
                raise ValueError("Action and status do not match")
            event = self.store.get_event(event_id)
            if event.status == action.status:
                raise ValueError(f'Event is already marked "{action.status.value}"')
            location = self.locations.get(event_id)
            if location:
                # Escalation/resolution are workflow choices, not ground truth.
                status = 'dismissed' if action.action.value == 'false_alarm' else 'acknowledged'
                note = f'Dashboard action={action.action.value}; status={action.status.value}. '
                note += action.note or ''
                self.workbench.add_review(location[0], dict(event_id=location[1], status=status,
                    reviewer=action.operator_id or 'operator_unspecified', note=note[:4000]))
            return self.store.apply_action(event_id, action)

    def media_file(self, job_id: str, filename: str) -> Path:
        folder = self.workbench.folder(job_id)
        if filename == 'inference.log':
            return folder/'inference.log'
        target = (folder/'result'/filename).resolve()
        if not target.is_relative_to((folder/'result').resolve()):
            raise ValueError('Result path escaped its job directory')
        if target.suffix.lower() not in {'.mp4', '.csv', '.json', '.jpg', '.jpeg', '.png', '.txt', '.log'}:
            raise ValueError('Unsupported result-file type')
        return target

    def close(self):
        self.workbench.close()
