/**
 * Integration tab — the handoff surface for Members 1-3.
 *
 * Why this is a dashboard tab and not just a markdown file: a teammate can
 * paste the JSON their pipeline actually produces and immediately see the
 * errors the dashboard would raise, then push it into the live queue and
 * watch it render. That turns "integrate with the dashboard" from a meeting
 * into a two-minute self-service check, days before any HTTP call is written.
 */

import { useMemo, useState } from 'react';
import type { IncidentEvent } from '../types/events';
import { validateEvent } from '../lib/validateEvent';
import type { ValidationResult } from '../lib/validateEvent';
import { COMPONENTS, COMPONENT_IDS, INCIDENT_TYPES } from '../lib/incidentTypes';
import { API_BASE_URL } from '../api';

const SAMPLE_EVENT = `{
  "event_id": "evt-c1-0101",
  "component_id": "component_1",
  "incident_type": "tailgating",
  "camera_id": "clip_03",
  "timestamp_detected": "${new Date().toISOString().replace(/\.\d{3}Z$/, 'Z')}",
  "priority": "high",
  "status": "new",
  "summary": "Possible tailgating at the controlled door - review required.",
  "evidence": [
    { "name": "entry_gap", "value": 1.2, "unit": "seconds",
      "description": "Time between the two tracks crossing the door line." },
    { "name": "follower_distance", "value": 0.8, "unit": "metres (est.)" }
  ],
  "bounding_box": { "x_min": 0.18, "y_min": 0.30, "x_max": 0.46, "y_max": 0.92 },
  "track_id": "person-77",
  "media_ref": "clips/clip_03.mp4",
  "media_offset_seconds": 2.0,
  "confidence": 0.81,
  "score_calibrated": false,
  "zone_id": "door_restricted_a",
  "model_name": "spatial_rel_gnn",
  "model_version": "0.1"
}`;

const PYTHON_SNIPPET = `# components/<your_component>/emit.py
import uuid, datetime, requests

API = "${API_BASE_URL}/api/v1/events"

def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def emit(incident_type, camera_id, summary, evidence, *,
         component_id, model_name, model_version,
         confidence=None, bbox=None, track_id=None,
         media_ref=None, media_offset_seconds=None,
         priority="medium", zone_id=None):
    """Send one incident event to the shared dashboard."""
    event = {
        "event_id": f"{component_id}-{uuid.uuid4().hex[:8]}",
        "component_id": component_id,
        "incident_type": incident_type,
        "camera_id": camera_id,
        "timestamp_detected": utc_now(),
        "priority": priority,
        "status": "new",
        "summary": summary,
        "evidence": evidence,
        "bounding_box": bbox,          # normalised 0..1, or None
        "track_id": track_id,          # "person-12", never a name
        "media_ref": media_ref,        # or None if no clip was saved
        "media_offset_seconds": media_offset_seconds,
        "confidence": confidence,      # 0..1, or None for rule-based
        "score_calibrated": False,
        "zone_id": zone_id,
        "model_name": model_name,
        "model_version": model_version,
    }
    r = requests.post(API, json=event, timeout=5)
    r.raise_for_status()
    return r.json()`;

const ENDPOINTS = [
  { verb: 'GET', path: '/api/v1/health', note: 'Is the service up, and is it serving mock or live data?' },
  { verb: 'GET', path: '/api/v1/cameras', note: 'Registered cameras and recorded clips.' },
  { verb: 'GET', path: '/api/v1/events', note: 'The alert queue. Optional ?status= &camera_id= &component_id= filters.' },
  { verb: 'GET', path: '/api/v1/events/{id}', note: 'One alert with its evidence and audit trail.' },
  { verb: 'POST', path: '/api/v1/events', note: 'YOUR PIPELINE POSTS HERE. Body = one incident event.' },
  { verb: 'PATCH', path: '/api/v1/events/{id}', note: 'Operator records a decision. The dashboard calls this, not you.' },
];

function IssueList({ result }: { result: ValidationResult }) {
  const issues = [
    ...result.errors.map((i) => ({ ...i, kind: 'error' as const })),
    ...result.warnings.map((i) => ({ ...i, kind: 'warning' as const })),
  ];

  if (issues.length === 0) {
    return (
      <ul className="issues">
        <li className="issue" style={{ borderColor: 'var(--accent-line)' }}>
          <span className="issue__field">✓</span>
          <span>Valid against contract v1.1. No errors, no warnings.</span>
        </li>
      </ul>
    );
  }

  return (
    <ul className="issues">
      {issues.map((issue, i) => (
        <li className="issue" data-kind={issue.kind} key={`${issue.field}-${i}`}>
          <span className="issue__field">{issue.field}</span>
          <span>{issue.message}</span>
        </li>
      ))}
    </ul>
  );
}

interface IntegrationPanelProps {
  onSubmit: (event: IncidentEvent) => Promise<void>;
  apiMode: 'mock' | 'http';
}

export function IntegrationPanel({ onSubmit, apiMode }: IntegrationPanelProps) {
  const [draft, setDraft] = useState(SAMPLE_EVENT);
  const [submitState, setSubmitState] = useState<{ kind: 'idle' | 'ok' | 'error'; message: string }>({
    kind: 'idle',
    message: '',
  });

  const parsed = useMemo(() => {
    try {
      return { ok: true as const, value: JSON.parse(draft) as unknown };
    } catch (err) {
      return { ok: false as const, message: err instanceof Error ? err.message : 'Invalid JSON' };
    }
  }, [draft]);

  const result = useMemo<ValidationResult | null>(() => {
    if (!parsed.ok) return null;
    return validateEvent(parsed.value);
  }, [parsed]);

  async function handleSubmit() {
    if (!result?.valid || !result.event) return;
    try {
      await onSubmit(result.event);
      setSubmitState({
        kind: 'ok',
        message: `Added ${result.event.event_id} to the queue. Open the Live queue tab to see it.`,
      });
    } catch (err) {
      setSubmitState({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Could not submit that event.',
      });
    }
  }

  return (
    <div className="integration">
      <div className="integration__inner">
        {/* ---------------- left column: the live checker ---------------- */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-5)' }}>
          <div className="card">
            <h2 className="card__title">Check your pipeline's output</h2>
            <p className="card__lede">
              Paste one event exactly as your detector produces it. You get the same errors the
              dashboard would raise, before you write a single HTTP call. Then push it into the live
              queue and watch it render.
            </p>

            <label className="visually-hidden" htmlFor="event-json">
              Event JSON
            </label>
            <textarea
              id="event-json"
              className="json-input"
              spellCheck={false}
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                setSubmitState({ kind: 'idle', message: '' });
              }}
            />

            {!parsed.ok ? (
              <ul className="issues">
                <li className="issue" data-kind="error">
                  <span className="issue__field">JSON</span>
                  <span>{parsed.message}</span>
                </li>
              </ul>
            ) : result ? (
              <IssueList result={result} />
            ) : null}

            <div style={{ display: 'flex', gap: 'var(--sp-2)', marginTop: 'var(--sp-3)', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn--primary"
                disabled={!result?.valid}
                onClick={() => void handleSubmit()}
              >
                Send to the alert queue
              </button>
              <button type="button" className="btn" onClick={() => setDraft(SAMPLE_EVENT)}>
                Reset to sample
              </button>
            </div>

            {submitState.kind !== 'idle' ? (
              <p
                className="inline-note"
                style={{ color: submitState.kind === 'ok' ? 'var(--sig-ok)' : 'var(--sig-high)' }}
              >
                {submitState.message}
              </p>
            ) : null}

            <p className="inline-note">
              {apiMode === 'mock'
                ? 'Mock mode: the event is added to this browser session only. Nothing is written to a server.'
                : 'Live mode: the event is POSTed to the API and becomes visible to every operator.'}
            </p>
          </div>

          <div className="card">
            <h2 className="card__title">Minimal Python emitter</h2>
            <p className="card__lede">
              Copy this into your component, call <span className="mono">emit(...)</span> when your
              detector fires. It is the whole integration.
            </p>
            <pre className="code">{PYTHON_SNIPPET}</pre>
          </div>
        </div>

        {/* ---------------- right column: the reference ---------------- */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--sp-5)' }}>
          <div className="card">
            <h2 className="card__title">Who owns what</h2>
            <p className="card__lede">
              Four components, one event format, one dashboard. Any component can ship without
              waiting for the others.
            </p>
            <div className="legend">
              {COMPONENT_IDS.map((id) => {
                const meta = COMPONENTS[id];
                const types = Object.values(INCIDENT_TYPES).filter((t) => t.component === id);
                return (
                  <div className="legend__row" key={id}>
                    <span className="chip__swatch" style={{ background: `var(${meta.colorVar})`, marginTop: 6 }} />
                    <div>
                      <div className="legend__name">
                        <strong className="mono">{meta.tag}</strong> — {meta.name}
                      </div>
                      <div className="legend__owner">
                        {meta.owner} · <span className="mono">{id}</span>
                      </div>
                      <div className="legend__owner mono">
                        {types.length > 0 ? types.map((t) => t.type).join(', ') : 'no types registered yet'}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="inline-note">
              Sending an <span className="mono">incident_type</span> that is not listed here still
              works — the dashboard renders it with a readable label and marks it "Unregistered
              type". Add it to <span className="mono">src/lib/incidentTypes.ts</span> to give it a
              proper label and operator hint.
            </p>
          </div>

          <div className="card">
            <h2 className="card__title">API endpoints</h2>
            <p className="card__lede">
              Base URL: <span className="mono">{API_BASE_URL}</span>
            </p>
            <ul className="endpoint-list">
              {ENDPOINTS.map((e) => (
                <li key={`${e.verb}-${e.path}`}>
                  <div className="endpoint">
                    <span className="endpoint__verb">{e.verb}</span>
                    <span className="endpoint__path">{e.path}</span>
                    <span className="endpoint__note">{e.note}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="card">
            <h2 className="card__title">The rules that are not negotiable</h2>
            <ul style={{ margin: 0, paddingLeft: '1.1rem', fontSize: 'var(--fs-small)', color: 'var(--text-2)', display: 'flex', flexDirection: 'column', gap: 'var(--sp-2)' }}>
              <li>
                Timestamps are <strong>UTC</strong>, ISO-8601, trailing <span className="mono">Z</span>.
              </li>
              <li>
                Bounding boxes are <strong>normalised 0..1</strong>, not pixels. Divide by frame width
                and height.
              </li>
              <li>
                <span className="mono">track_id</span> is an anonymous tracker label such as{' '}
                <span className="mono">person-12</span>. Never a name, never a face ID.
              </li>
              <li>
                Missing value means <span className="mono">null</span>. Do not invent a zero, an
                empty string or a fake confidence.
              </li>
              <li>
                <span className="mono">confidence</span> is a <strong>score</strong>, not a
                probability, unless you have calibrated it and set{' '}
                <span className="mono">score_calibrated: true</span>.
              </li>
              <li>
                Summaries stay cautious: "Potential fall — review required", never "Fall confirmed"
                or "Medical emergency".
              </li>
              <li>
                <span className="mono">event_id</span> must be unique. A duplicate is rejected with a
                409.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
