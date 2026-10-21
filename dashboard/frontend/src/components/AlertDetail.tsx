/**
 * One alert, full screen.
 *
 * A single column an officer reads top to bottom, in the order they need it:
 *
 *   1. What happened, where, when      (big, coloured by urgency)
 *   2. The video
 *   3. What the camera saw             (plain sentences)
 *   4. What to do                      (one suggested first step)
 *   5. Four big coloured buttons
 *   6. Everything else                 (collapsed — history, technical)
 *
 * No tabs, no side panels, no badges. One alert, one screen, one decision.
 */

import type { CameraSource, IncidentEvent, OperatorAction } from '../types/events';
import { describeIncident } from '../lib/incidentTypes';
import { matchStrength, plainCamera, plainRole, plainSignal, plainTimeAgo, plainTrack } from '../lib/plainLanguage';
import { urgencyOf } from '../lib/urgency';
import { DecisionButtons } from './DecisionButtons';
import { TechnicalDetails } from './TechnicalDetails';
import { AuditTrail } from './AuditTrail';
import { VideoPanel } from './VideoPanel';

/** What the camera saw, one plain sentence per signal. */
function Observations({ event }: { event: IncidentEvent }) {
  if (event.component_id === 'component_3' && event.evidence.some(signal => signal.name === 'c3_research_only' && signal.value === true)) {
    const sourceTime = event.evidence.find(signal => signal.name === 'source_timestamp_seconds')?.value;
    const joints = event.evidence.find(signal => signal.name === 'observed_pose_joints')?.value;
    const observed = [
      typeof sourceTime === 'number' ? `This moment is ${sourceTime.toFixed(2)} seconds into the original recording.` : 'Use the annotated clip to inspect this recorded moment.',
      typeof joints === 'number' ? `${joints} body joints were currently observed across the selected tracks.` : 'A usable pose sequence was available for this research score.',
      'Movement and proximity measurements describe the scene; they do not establish aggression or intent.',
      'Scene occlusion was not measured. Judge visibility from the video.',
    ];
    if (event.evidence.some(signal => signal.name === 'timestamp_origin' && signal.value === 'processing_utc_not_recording_time')) {
      observed.push('The alert clock records processing time; the original recording date is unknown.');
    }
    return <ul className="observations">{observed.map(message => <li className="observation" key={message} data-kind="view"><span className="observation__dot" aria-hidden="true" /><span className="observation__text">{message}</span></li>)}</ul>;
  }
  if (event.evidence.length === 0) {
    return (
      <p className="muted-note">
        The system did not say why it raised this. Judge it from the video, and let the project team
        know.
      </p>
    );
  }

  return (
    <ul className="observations">
      {event.evidence.map((signal, i) => {
        const plain = plainSignal(signal);
        return (
          <li
            className="observation"
            key={`${signal.name}-${i}`}
            data-kind={plain.reliability ? 'view' : plain.reassuring ? 'against' : 'for'}
          >
            <span className="observation__dot" aria-hidden="true" />
            <span className="observation__text">{plain.text}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** Who was involved — readable labels, still anonymous. */
function People({ event }: { event: IncidentEvent }) {
  const subject = plainTrack(event.track_id);
  const related = event.related_tracks ?? [];
  const research = event.evidence.some(signal => signal.name === 'c3_research_only' && signal.value === true);
  if (!subject && related.length === 0) return null;

  return (
    <div className="people">
      {subject ? (
        <span className="people__chip">
          <span className="people__role">{research ? 'Observed track' : 'Main subject'}</span>
          <span className="people__who">{subject}</span>
        </span>
      ) : null}
      {related.map((r) => (
        <span className="people__chip" key={`${r.role}-${r.track_id}`}>
          <span className="people__role">{research && r.role === 'co_observed_track' ? 'Other observed track' : plainRole(r.role)}</span>
          <span className="people__who">
            {plainTrack(r.track_id)}
            {r.association_score !== null && r.association_score !== undefined && r.association_score < 0.7
              ? ' (not certain)'
              : ''}
          </span>
        </span>
      ))}
    </div>
  );
}

interface AlertDetailProps {
  event: IncidentEvent;
  cameras: CameraSource[];
  pending: boolean;
  onAction: (action: OperatorAction, operatorId: string, note: string | null) => void;
  /** Shown only on narrow screens, where the list is a separate screen. */
  onBack: () => void;
}

export function AlertDetail({ event, cameras, pending, onAction, onBack }: AlertDetailProps) {
  const incident = describeIncident(event.incident_type, event.component_id);
  const urgency = urgencyOf(event);
  const camera = cameras.find((c) => c.camera_id === event.camera_id);
  const strength = matchStrength(event);
  const c3Research = event.component_id === 'component_3' && event.evidence.some(signal => signal.name === 'c3_research_only' && signal.value === true);

  return (
    <div className="detail" data-urgency={urgency.key}>
      <header className="detail__header">
        <button type="button" className="detail__back" onClick={onBack}>
          ← All alerts
        </button>

        <div className="detail__headline">
          <span className="detail__icon" aria-hidden="true">
            {incident.glyph}
          </span>
          <div className="detail__titles">
            <h2 className="detail__title">{incident.label}</h2>
            <p className="detail__where">
              {plainCamera(camera?.label, event.camera_id)} · {plainTimeAgo(event.timestamp_detected)}
            </p>
          </div>
          <span className="detail__tag">{urgency.word}</span>
        </div>
      </header>

      <div className="detail__scroll">
        {c3Research ? <div className="c3-notice"><strong>C3 research candidate - operator review required</strong><p>Weak annotated-onset and active-fight tasks do not establish verified pre-contact aggression. Scores are uncalibrated; operational notifications remain disabled. This decision does not become a training label.</p></div> : null}
        <VideoPanel event={event} />

        <section className="block">
          <h3 className="block__title">What the camera saw</h3>
          <Observations event={event} />
          <People event={event} />
        </section>

        <section className="block">
          <h3 className="block__title">{c3Research ? 'Research model score' : 'How sure is the system?'}</h3>
          <div className="sureness" data-band={strength.band}>
            <span className="sureness__word">{c3Research ? 'Unvalidated research output' : strength.word}</span>
            <span className="sureness__caveat">{c3Research ? `Uncalibrated score: ${event.confidence?.toFixed(3) ?? 'not supplied'}. This is not a probability or verified aggression label.` : strength.caveat}</span>
          </div>
        </section>

        <section className="block block--advice">
          <h3 className="block__title">What to do</h3>
          <p className="advice">{c3Research ? 'Review this recorded moment and record what is visible. Follow your own site procedure for any real-world action.' : incident.whatToDo}</p>
          <p className="muted-note">{c3Research ? 'A pose pattern or high model score does not confirm aggression. Full measurements remain available in Technical details.' : incident.operatorHint}</p>
          <p className="muted-note">Your own site procedure always comes first.</p>
        </section>

        <section className="block">
          <h3 className="block__title">Your decision</h3>
          <DecisionButtons event={event} pending={pending} onAction={onAction} />
        </section>

        <details className="more">
          <summary className="more__summary">What's been done so far</summary>
          <div className="more__body">
            <AuditTrail entries={event.audit ?? []} />
          </div>
        </details>

        <TechnicalDetails event={event} camera={camera} />
      </div>
    </div>
  );
}
