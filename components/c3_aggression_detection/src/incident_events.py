"""C3 research incidents and projections to the existing shared dataclass.

Pose evidence describes observations, not model attribution or human intent.
No network delivery or operational notification is implemented here.
"""
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid

SCHEMA = 'c3_incident_envelope_v1'
KINDS = {'WEAK_ONSET_WITHIN_3S_ALERT': 'potential_aggression',
         'WEAK_ACTIVE_FIGHT_ALERT': 'active_violence'}


@lru_cache(maxsize=1)
def shared_contract():
    """Load the team's actual, unmodified contract without copying its fields."""
    path = Path(__file__).resolve().parents[3]/'shared/schemas/event_schema.py'
    spec = importlib.util.spec_from_file_location('_c3_shared_incident_contract', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Existing shared IncidentEvent contract is unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_timestamp(value=None):
    instant = datetime.now(timezone.utc) if value is None else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if instant.tzinfo is None:
        raise ValueError('Timestamp must include a UTC offset')
    return instant.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def project_shared_event(incident):
    """Only project semantically representable events; generic onset has no type."""
    if incident['incident_kind'] != 'active_violence':
        raise ValueError('The group contract has no generic potential-aggression type; team agreement required')
    contract = shared_contract()
    if 'fighting' not in contract.EVENT_TYPES:
        raise ValueError('Shared contract no longer supports fighting')
    track_ids = incident['track_ids']
    if not track_ids:
        raise ValueError('Shared incidents require a tracked identity')
    event = contract.IncidentEvent(timestamp=incident['timestamp'], camera_id=incident['camera_id'],
        event_type='fighting', threat_score=incident['model_score'], person_id=f'P-{track_ids[0]}',
        bounding_box=incident['evidence'].get('bounding_box'),
        explanation_signals=['research_only_weak_active_fight', 'uncalibrated_score',
            *[signal['signal'] for signal in incident['evidence_signals']]],
        model_version=incident['model_version'], frame_number=incident['evidence']['source_frame_number'],
        evidence_frame=incident['evidence'].get('evidence_frame'))
    return event.to_dict()


def _decoded(row, name, default):
    value = row.get(name, default)
    return json.loads(value) if isinstance(value, str) else value


def build_incidents(records, metadata, checkpoint, checkpoint_sha256, camera_id='c3_demo',
                    processed_at=None, recording_start=None, annotated_video=None):
    if not isinstance(camera_id, str) or not camera_id.strip() or len(camera_id) > 120:
        raise ValueError('Camera ID must contain 1–120 characters')
    processed_at = utc_timestamp(processed_at)
    recording = datetime.fromisoformat(utc_timestamp(recording_start).replace('Z', '+00:00')) if recording_start else None
    incidents = []; current = {}; notifications = []
    for row in records:
        status = row.get('candidate_status')
        kind = KINDS.get(status)
        if not kind or not row['valid_pose']:
            current.clear()
            continue
        t = float(row['t_s'])
        score_key = 'smoothed_onset_score' if kind == 'potential_aggression' else 'smoothed_fight_score'
        score = row[score_key]
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Incident score must be finite and bounded')
        tracks = [int(v) for v in _decoded(row, 'track_ids_json', [])]
        key = (int(row['segment_id']), kind, tuple(sorted(tracks)))
        # Each confirmed episode has one incident, including a suppressed candidate.
        if key not in current:
            current.clear()
            if not tracks:
                continue
            event_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                f"c3:{metadata['source_sha256']}:{checkpoint_sha256}:{camera_id}:{key}:{t:.9f}"))
            signals = _decoded(row, 'evidence_signals_json', [])
            incident = dict(event_id=event_id, component_id='C3', camera_id=camera_id,
                timestamp=utc_timestamp((recording+timedelta(seconds=t)).isoformat()) if recording else processed_at,
                timestamp_origin='recording_start_plus_offset' if recording else 'processing_utc_not_recording_time',
                video_timestamp_s=t, incident_kind=kind,
                model_score=float(score), score_semantics='uncalibrated sigmoid research score; not a probability',
                research_only=True, operational_notification=False, verified_early_aggression_claim=False,
                reliability=dict(supervision=checkpoint.get('supervision', 'weak'), calibration='UNVALIDATED',
                    quality=float(row['quality']), current_observed_joints=int(row['current_observed_joints']),
                    observed_history_frames=int(row['observed_history_frames']),
                    human_review_required=True, verified_precursor_model=False),
                track_ids=tracks, evidence_signals=signals,
                evidence=dict(source_video=metadata['source_video'], source_sha256=metadata['source_sha256'],
                    source_frame_number=int(row['source_frame_number']), source_timestamp_s=t,
                    annotated_video=annotated_video,
                    annotated_video_offset_s=t-metadata['extracted_start_s'],
                    bounding_box=_decoded(row, 'bounding_box_json', None),
                    explanation_scope='Measured pose observations; not causal model attribution or intent'),
                model_version=f"{checkpoint['variant']}:{checkpoint_sha256[:16]}",
                checkpoint_sha256=checkpoint_sha256, operator_review_status='pending',
                notification=dict(eligible=True, emitted=False,
                    suppression_reason=row.get('notification_suppression_reason', 'scores_only')))
            if kind == 'active_violence':
                incident['shared_event'] = project_shared_event(incident)
                incident['shared_export_status'] = 'compatible_projection_research_only_no_delivery'
            else:
                incident['shared_event'] = None
                incident['shared_export_status'] = 'blocked_no_exact_shared_type'
            incidents.append(incident); current[key] = incident
        incident = current[key]
        if row.get('alert_emitted'):
            incident['notification'].update(emitted=True, emitted_at_video_s=t,
                delay_from_first_candidate_s=t-incident['video_timestamp_s'], suppression_reason=None)
            notifications.append(dict(event_id=incident['event_id'], video_timestamp_s=t,
                incident_kind=kind, research_only=True, operational_notification=False))
    return dict(schema=SCHEMA, incidents=incidents, component_id='C3',
        processed_at=processed_at, operational_alerts_enabled=False,
        verified_early_aggression_claim=False,
        shared_integration='Exact IncidentEvent projection for active fight; generic forecast type awaits team agreement; no delivery API'), notifications
