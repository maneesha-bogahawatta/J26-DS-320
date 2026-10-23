/**
 * The short version of an alert, shown under the video on the upload and live
 * screens: what it is, what the camera saw, and what to do. The full alert —
 * with the decision buttons and the record of what was done — is one tap away
 * in Alerts, where every found item is also listed.
 */

import type { Found } from '../../detect';
import { describeIncident } from '../../lib/incidentTypes';
import { matchStrength, plainSignal } from '../../lib/plainLanguage';
import { urgencyOf } from '../../lib/urgency';

interface FoundSummaryProps {
  found: Found;
  when: string;
  onOpenAlert: (eventId: string) => void;
}

export function FoundSummary({ found, when, onOpenAlert }: FoundSummaryProps) {
  const { event } = found;
  const meta = describeIncident(event.incident_type, event.component_id);
  const urgency = urgencyOf(event);
  const strength = matchStrength(event);
  const research = event.evidence.some(signal => signal.name === 'c3_research_only' && signal.value === true);
  const signals = research ? event.evidence.filter(signal => ['source_timestamp_seconds', 'c3_current_observed_joints', 'c3_quality'].includes(signal.name)).map(plainSignal) : event.evidence.map(plainSignal);

  return (
    <section className="found" data-urgency={urgency.key} aria-label="Selected item">
      <header className="found__head">
        <span className="found__icon" aria-hidden="true">
          {meta.glyph}
        </span>
        <div className="found__titles">
          <h3 className="found__title">{meta.label}</h3>
          <p className="found__when">{when}</p>
        </div>
        <span className="found__tag">{urgency.word}</span>
      </header>

      {signals.length > 0 ? (
        <ul className="found__saw">
          {signals.map((s, i) => (
            <li key={i} data-kind={s.reassuring ? 'against' : s.reliability ? 'view' : 'for'}>
              {s.text}
            </li>
          ))}
        </ul>
      ) : null}

      <p className="found__do">
        <strong>What to do:</strong> {research ? 'Review this research candidate. Weak-supervision scores do not confirm violence or verified early prediction.' : meta.whatToDo}
      </p>

      <div className="found__foot">
        <span className="found__strength" title={strength.caveat}>
          {research ? `Uncalibrated model score: ${event.confidence?.toFixed(3) ?? 'unavailable'}` : strength.word}
        </span>
        <button type="button" className="btn btn--primary" onClick={() => onOpenAlert(event.event_id)}>
          Open to decide
        </button>
      </div>
    </section>
  );
}
