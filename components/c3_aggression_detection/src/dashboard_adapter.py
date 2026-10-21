"""Project C3 research envelopes onto the supplied dashboard v1.1 contract.

The imported dashboard contract is separate from shared/schemas/event_schema.py.
This adapter neither edits that contract nor sends operational notifications.
"""
import json
import math
import re

from src.incident_events import utc_timestamp


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite numeric value')
    return float(value)


def normalized_render_box(box, prediction):
    """Convert source pixels onto the rendered image INCLUDING its score footer.

    Older prediction reports lack verified rendering geometry: omit their box
    rather than overlaying source-normalized coordinates on different media.
    """
    if box is None:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Expected source pixel xyxy box')
    values = [_finite(v, 'box coordinate') for v in box]
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError('Source box must have positive area')
    geometry = prediction.get('media_geometry')
    if not geometry:
        return None
    width = _finite(geometry['source_width'], 'source width')
    height = _finite(geometry['source_height'], 'source height')
    rendered_width = _finite(geometry['rendered_width'], 'rendered width')
    content_height = _finite(geometry['content_height'], 'content height')
    rendered_height = _finite(geometry['rendered_height'], 'rendered height')
    if min(width, height, rendered_width, content_height, rendered_height) <= 0 or content_height > rendered_height:
        raise ValueError('Invalid annotated-media geometry')
    x1, y1, x2, y2 = values
    result = dict(x_min=max(0., min(1., x1/width)),
        y_min=max(0., min(content_height/rendered_height, y1/height*content_height/rendered_height)),
        x_max=max(0., min(1., x2/width)),
        y_max=max(0., min(content_height/rendered_height, y2/height*content_height/rendered_height)))
    if result['x_min'] >= result['x_max'] or result['y_min'] >= result['y_max']:
        return None
    return result


def _signal(name, value, unit=None, description=None):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite dashboard evidence')
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError('Dashboard evidence must be a scalar or null')
    return dict(name=name, value=value, unit=unit, description=description)


def adapt_run(incidents, prediction, job_id, media_base='/api/v1/c3/media'):
    """Return exact IncidentEvent dictionaries; caller validates/imports schema.

    IDs are scoped to this run so repeated clips do not inherit earlier operator
    decisions or media. Re-ingesting one run remains idempotent.
    """
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', job_id):
        raise ValueError('Invalid dashboard run ID')
    if not isinstance(media_base, str) or not media_base.startswith('/') or '..' in media_base.split('/'):
        raise ValueError('Media route must be a safe same-origin absolute path')
    if prediction.get('operational_alerts_enabled') is not False or prediction.get('verified_early_aggression_claim') is not False:
        raise ValueError('Only explicitly non-operational weak research runs may be projected')
    if not isinstance(incidents, list):
        raise ValueError('C3 incidents must be a list')
    result = []; seen = set()
    for incident in incidents:
        kind = incident['incident_kind']
        if kind not in ('potential_aggression', 'active_violence'):
            raise ValueError('Unknown C3 research incident kind')
        if incident.get('research_only') is not True or incident.get('operational_notification') is not False:
            raise ValueError('Dashboard projection requires explicit research-only incident provenance')
        local_id = incident['event_id']
        if not isinstance(local_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,150}', local_id) or local_id in seen:
            raise ValueError('Invalid/repeated local incident ID')
        seen.add(local_id)
        score = _finite(incident['model_score'], 'uncalibrated model score')
        if not 0 <= score <= 1:
            raise ValueError('Model score outside [0,1]')
        source_time = _finite(incident['video_timestamp_s'], 'source video time')
        if source_time < 0: raise ValueError('Negative source video time')
        evidence = incident['evidence']
        offset = _finite(evidence['annotated_video_offset_s'], 'annotated video time')
        if offset < 0: raise ValueError('Negative annotated-media offset')
        context = prediction.get('source_context')
        if context is not None:
            if len(context) != 2 or not context[0] <= source_time < context[1] or abs(source_time-context[0]-offset) > 1e-5:
                raise ValueError('Source and clip-relative incident timestamps disagree')
        raw_tracks = incident.get('track_ids', [])
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in raw_tracks) or len(set(raw_tracks)) != len(raw_tracks):
            raise ValueError('Anonymous numeric track IDs required')
        tracks = [f'person-{value}' for value in raw_tracks]
        reliability = incident.get('reliability', {})
        signals = [_signal('c3_research_only', True, description='Weak-model research candidate; human review required.'),
            _signal('c3_task', 'weak_first_onset_forecasting' if kind == 'potential_aggression' else 'weak_active_fight_recognition'),
            _signal('c3_verified_pre_contact_prediction', False, description='Independently verified pre-contact performance remains unvalidated.'),
            _signal('operational_alerts_enabled', False),
            _signal('c3_run_id', job_id), _signal('c3_local_event_id', local_id),
            _signal('source_timestamp_seconds', source_time, 'seconds', 'Offset in the original recording.'),
            _signal('source_frame_number', evidence.get('source_frame_number'), 'frame'),
            _signal('timestamp_origin', incident.get('timestamp_origin', 'processing_utc_not_recording_time'),
                description='Processing time is not the original recording date or contact time.'),
            _signal('supervision', reliability.get('supervision', prediction.get('supervision'))),
            _signal('occlusion_ratio', None, description='Not measured. Missing joints and track quality are not a measured scene occlusion ratio.'),
            _signal('notification_emitted', bool(incident.get('notification', {}).get('emitted', False)),
                description='Research notification only; candidates also remain visible when notifications are disabled.')]
        for signal in incident.get('evidence_signals', []):
            signals.append(_signal(signal['signal'], signal.get('value'), signal.get('units'),
                'Descriptive pose observation; not causal attribution or proof of intent.'))
        for field, unit in (('quality', 'extractor_quality_index'), ('current_observed_joints', 'joint_count'),
                            ('observed_history_frames', 'frames')):
            if field in reliability:
                signals.append(_signal('c3_'+field, reliability[field], unit))
        if evidence.get('bounding_box') is not None:
            signals.append(_signal('source_bounding_box_xyxy', json.dumps(evidence['bounding_box'], allow_nan=False), 'source_pixels'))
        media = f"{media_base.rstrip('/')}/{job_id}/demo.mp4" if evidence.get('annotated_video') else None
        result.append(dict(event_id=f'component_3-{job_id}-{local_id}', component_id='component_3',
            incident_type='aggression' if kind == 'potential_aggression' else 'violence',
            camera_id=incident['camera_id'], timestamp_detected=utc_timestamp(incident['timestamp']),
            priority='medium', status='new',
            summary=('Potential aggression - weak onset candidate; human review required' if kind == 'potential_aggression'
                else 'Potential active violence - weak-model candidate; human review required'),
            evidence=signals, model_name=f"c3-{prediction.get('variant', 'unknown')}-stgcn",
            model_version=incident['model_version'], confidence=score,
            score_calibrated=False, bounding_box=normalized_render_box(evidence.get('bounding_box'), prediction) if media else None,
            track_id=tracks[0] if tracks else None,
            related_tracks=[dict(track_id=track, role='co_observed_track', association_score=None,
                note='Selected by the pose pipeline; participation in violence is not established.') for track in tracks[1:]] or None,
            media_ref=media, media_offset_seconds=offset if media else None, zone_id=None, audit=[]))
    # Forbid accidental non-finite data before the HTTP schema sees it.
    json.dumps(result, allow_nan=False)
    return result
