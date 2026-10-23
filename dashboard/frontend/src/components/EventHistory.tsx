/**
 * Event history — every alert and what the operator did about it.
 *
 * This view is doing double duty. For the operator it is "what happened on my
 * shift". For the research it is the evaluation log: the response-time column
 * is the gap between the pipeline raising an alert and the operator's first
 * decision, which is exactly the measure RQ4 asks about ("does an explainable
 * real-time alert layer measurably improve operator detection rate and
 * response time?").
 *
 * The CSV export exists so that data can leave the prototype and go into the
 * analysis without anyone retyping it.
 */

import { useMemo, useState } from 'react';
import type { AuditEntry, CameraSource, IncidentEvent } from '../types/events';
import { ACTION_LABELS } from '../lib/format';
import { COMPONENTS, describeIncident } from '../lib/incidentTypes';
import { plainCamera, plainClock, plainDuration, plainTimeAgo } from '../lib/plainLanguage';
import { urgencyOf } from '../lib/urgency';
import { EmptyState } from './States';
import { C3_ENABLED } from '../api';

/** The same colour language as the alert cards, in badge form. */
function UrgencyBadge({ event }: { event: IncidentEvent }) {
  const urgency = urgencyOf(event);
  return (
    <span className="badge" data-urgency={urgency.key} title={urgency.description}>
      {urgency.word}
    </span>
  );
}

/** Seconds from detection to the operator's FIRST decision. Null if untouched. */
export function responseSeconds(event: IncidentEvent): number | null {
  const actions = (event.audit ?? []).filter((a) => a.action !== 'created');
  if (actions.length === 0) return null;
  const first = actions.reduce((earliest, a) =>
    new Date(a.at).getTime() < new Date(earliest.at).getTime() ? a : earliest,
  );
  const delta = (new Date(first.at).getTime() - new Date(event.timestamp_detected).getTime()) / 1000;
  return Number.isFinite(delta) ? Math.max(0, Math.round(delta)) : null;
}

function firstAction(event: IncidentEvent): AuditEntry | null {
  const actions = (event.audit ?? []).filter((a) => a.action !== 'created');
  if (actions.length === 0) return null;
  return actions.reduce((earliest, a) =>
    new Date(a.at).getTime() < new Date(earliest.at).getTime() ? a : earliest,
  );
}

function formatDuration(seconds: number | null): string {
  if (seconds === null) return 'Not looked at';
  return plainDuration(seconds);
}

function toCsv(events: IncidentEvent[]): string {
  const header = [
    'event_id',
    'component_id',
    'component_tag',
    'incident_type',
    'camera_id',
    'timestamp_detected_utc',
    'priority',
    'final_status',
    'confidence',
    'score_calibrated',
    'track_id',
    'first_action',
    'first_action_at_utc',
    'response_seconds',
    'operator_id',
    'operator_note',
    'model_name',
    'model_version',
  ];

  const escape = (value: unknown): string => {
    if (value === null || value === undefined) return '';
    const s = String(value);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };

  const rows = events.map((e) => {
    const action = firstAction(e);
    return [
      e.event_id,
      e.component_id,
      COMPONENTS[e.component_id]?.tag ?? '',
      e.incident_type,
      e.camera_id,
      e.timestamp_detected,
      e.priority,
      e.status,
      e.confidence ?? '',
      e.score_calibrated === true,
      e.track_id ?? '',
      action?.action ?? '',
      action?.at ?? '',
      responseSeconds(e) ?? '',
      action?.operator_id ?? '',
      action?.note ?? '',
      e.model_name,
      e.model_version,
    ].map(escape);
  });

  return [header.join(','), ...rows.map((r) => r.join(','))].join('\n');
}

function downloadCsv(events: IncidentEvent[]) {
  const blob = new Blob([toCsv(events)], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `j26ds320_event_log_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

interface EventHistoryProps {
  events: IncidentEvent[];
  cameras: CameraSource[];
  onOpenEvent: (eventId: string) => void;
}

export function EventHistory({ events, cameras, onOpenEvent }: EventHistoryProps) {
  const [search, setSearch] = useState('');
  const [reviewedOnly, setReviewedOnly] = useState(false);

  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return events
      .filter((e) => (reviewedOnly ? (e.audit ?? []).some((a) => a.action !== 'created') : true))
      .filter((e) => {
        if (!needle) return true;
        return (
          e.event_id.toLowerCase().includes(needle) ||
          e.summary.toLowerCase().includes(needle) ||
          e.incident_type.toLowerCase().includes(needle) ||
          e.camera_id.toLowerCase().includes(needle) ||
          (e.track_id ?? '').toLowerCase().includes(needle) ||
          (e.audit ?? []).some((a) => (a.note ?? '').toLowerCase().includes(needle))
        );
      })
      .sort((a, b) => new Date(b.timestamp_detected).getTime() - new Date(a.timestamp_detected).getTime());
  }, [events, search, reviewedOnly]);

  const reviewed = events.filter((e) => (e.audit ?? []).some((a) => a.action !== 'created'));
  const responses = reviewed.map(responseSeconds).filter((n): n is number => n !== null);
  const medianResponse =
    responses.length > 0 ? [...responses].sort((a, b) => a - b)[Math.floor(responses.length / 2)] : null;

  return (
    <div className="history">
      <div className="history__controls">
        <input
          type="search"
          className="search"
          placeholder="Search alerts and notes…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Search the shift log"
        />
        <label className="toggle">
          <input type="checkbox" checked={reviewedOnly} onChange={(e) => setReviewedOnly(e.target.checked)} />
          Only ones that were acted on
        </label>

        <div style={{ flex: 1 }} />

        <div className="stat">
          <span className="stat__label">Looked at</span>
          <span className="stat__value">
            {reviewed.length} of {events.length}
          </span>
        </div>
        <div className="stat">
          <span className="stat__label">Typical response</span>
          <span className="stat__value">{formatDuration(medianResponse)}</span>
        </div>

        <button type="button" className="btn btn--small" onClick={() => downloadCsv(rows)} disabled={rows.length === 0}>
          Export for the project team
        </button>
      </div>

      {rows.length === 0 ? (
        <EmptyState
          glyph="⌸"
          title="Nothing in the log yet"
          detail="Every alert and what was done about it appears here, including the ones marked as nothing wrong."
        />
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Time</th>
                <th scope="col">What came in</th>
                <th scope="col">Where</th>
                <th scope="col">Status</th>
                <th scope="col">Time to respond</th>
                <th scope="col">Note</th>
                <th scope="col"></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((event) => {
                const incident = describeIncident(event.incident_type, event.component_id);
                const action = firstAction(event);
                const camera = cameras.find((c) => c.camera_id === event.camera_id);
                return (
                  <tr key={event.event_id}>
                    <td style={{ whiteSpace: 'nowrap' }}>
                      {plainClock(event.timestamp_detected)}
                      <div className="table__sub">{plainTimeAgo(event.timestamp_detected)}</div>
                    </td>
                    <td>{incident.label}</td>
                    <td>{plainCamera(camera?.label, event.camera_id)}</td>
                    <td>
                      <UrgencyBadge event={event} />
                      {action ? (
                        <div className="table__sub">{ACTION_LABELS[action.action] ?? action.action}</div>
                      ) : null}
                    </td>
                    <td>{formatDuration(responseSeconds(event))}</td>
                    <td className="table__note">{action?.note ?? '—'}</td>
                    <td>
                      <button type="button" className="table__link" onClick={() => onOpenEvent(event.event_id)}>
                        Open
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="inline-note">
        {C3_ENABLED ? 'Time to respond measures the gap between processing an alert and its first operator decision. Decisions remain in the audit; they do not become independent precursor/contact labels or automatically retrain the model.' : 'Time to respond is measured from the alert arriving to the first decision recorded against it. Dismissed alerts stay in this log for later review.'}
      </p>
    </div>
  );
}
