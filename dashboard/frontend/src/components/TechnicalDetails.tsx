/**
 * Everything machine-readable, collapsed by default.
 *
 * This is the part an officer never opens and the project team always does:
 * raw scores, model name and version, event ID, box coordinates, UTC
 * timestamps, raw signal names and values, which component produced it.
 *
 * Nothing was deleted when the interface was simplified for officers — it was
 * moved here. That matters for the project: the evidence the examiners and
 * the pipeline owners need is still one click away on every single alert.
 */

import type { CameraSource, IncidentEvent } from '../types/events';
import { NOT_PROVIDED, formatScore, formatSignalValue, formatTimestamp, scoreLabel } from '../lib/format';
import { COMPONENTS, describeIncident } from '../lib/incidentTypes';

function Row({ label, value, missing = false }: { label: string; value: string; missing?: boolean }) {
  return (
    <>
      <span className="meta__key">{label}</span>
      <span className="meta__value" data-missing={String(missing)}>
        {value}
      </span>
    </>
  );
}

export function TechnicalDetails({
  event,
  camera,
}: {
  event: IncidentEvent;
  camera: CameraSource | undefined;
}) {
  const incident = describeIncident(event.incident_type, event.component_id);
  const component = COMPONENTS[event.component_id];

  return (
    <details className="more">
      <summary className="more__summary">Technical details — for the project team</summary>

      <div className="more__body">
        <div className="meta">
          <Row label="Event ID" value={event.event_id} />
          <Row label="Incident type" value={event.incident_type} />
          <Row
            label="Registered type"
            value={incident.known ? 'Yes' : 'No — not in the dashboard registry'}
            missing={!incident.known}
          />
          <Row
            label="Component"
            value={component ? `${component.tag} — ${component.name}` : event.component_id}
          />
          <Row
            label="Raw score"
            value={`${formatScore(event.confidence)} — ${scoreLabel(event.score_calibrated)}`}
            missing={event.confidence === null || event.confidence === undefined}
          />
          <Row label="Model" value={`${event.model_name} v${event.model_version}`} />
          <Row label="Detected (UTC)" value={event.timestamp_detected} />
          <Row label="Detected (local)" value={formatTimestamp(event.timestamp_detected)} />
          <Row
            label="Camera"
            value={`${event.camera_id}${camera ? ` — ${camera.label}` : ' — not registered'}`}
          />
          <Row label="Zone" value={event.zone_id ?? NOT_PROVIDED} missing={!event.zone_id} />
          <Row label="Track ID" value={event.track_id ?? NOT_PROVIDED} missing={!event.track_id} />

          {(event.related_tracks ?? []).map((r) => (
            <Row
              key={`tech-${r.role}-${r.track_id}`}
              label={`Related (${r.role})`}
              value={
                r.association_score !== null && r.association_score !== undefined
                  ? `${r.track_id} · association ${r.association_score.toFixed(2)}`
                  : r.track_id
              }
            />
          ))}

          <Row
            label="Bounding box"
            value={
              event.bounding_box
                ? `${event.bounding_box.x_min.toFixed(3)}, ${event.bounding_box.y_min.toFixed(3)} → ${event.bounding_box.x_max.toFixed(3)}, ${event.bounding_box.y_max.toFixed(3)} (normalised)`
                : NOT_PROVIDED
            }
            missing={!event.bounding_box}
          />
          <Row
            label="Media"
            value={`${event.media_ref ?? NOT_PROVIDED}${
              event.media_offset_seconds !== null && event.media_offset_seconds !== undefined
                ? ` @ ${event.media_offset_seconds}s`
                : ''
            }`}
            missing={!event.media_ref}
          />
        </div>

        {event.evidence.length > 0 ? (
          <>
            <h4 className="more__heading">Raw evidence signals</h4>
            <div className="meta">
              {event.evidence.map((signal, i) => (
                <Row key={`raw-${signal.name}-${i}`} label={signal.name} value={formatSignalValue(signal)} />
              ))}
            </div>
          </>
        ) : null}

        <p className="muted-note" style={{ marginTop: 'var(--sp-3)' }}>
          Track IDs are anonymous tracker labels. The system holds no names, faces or biometric
          identifiers.
        </p>
      </div>
    </details>
  );
}
