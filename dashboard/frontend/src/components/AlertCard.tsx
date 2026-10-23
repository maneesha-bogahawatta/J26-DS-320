/**
 * One alert, as a card.
 *
 * Four things, in the order an officer asks them:
 *   What happened?   Where?   How long ago?   How urgent?
 *
 * Nothing else. No IDs, no scores, no model names, no component tags. Those
 * exist and matter, but they belong in the technical panel — putting them
 * here makes an officer read past three pieces of machine vocabulary to find
 * the word "Lobby".
 *
 * The whole card is one button, so the target is enormous and there is no way
 * to click "near" an alert and miss.
 */

import type { CameraSource, IncidentEvent } from '../types/events';
import { describeIncident } from '../lib/incidentTypes';
import { plainCamera, plainTimeAgo } from '../lib/plainLanguage';
import { needsLook, urgencyOf } from '../lib/urgency';

interface AlertCardProps {
  event: IncidentEvent;
  cameras: CameraSource[];
  selected: boolean;
  onOpen: () => void;
  /** Replaces "place · time ago", e.g. "At 0:12 in the video". */
  where?: string;
  /** A still of the moment (live camera), shown instead of the icon. */
  thumb?: string | null;
}

export function AlertCard({ event, cameras, selected, onOpen, where, thumb }: AlertCardProps) {
  const incident = describeIncident(event.incident_type, event.component_id);
  const urgency = urgencyOf(event);
  const camera = cameras.find((c) => c.camera_id === event.camera_id);
  const place = plainCamera(camera?.label, event.camera_id);
  const whereText = where ?? `${place} · ${plainTimeAgo(event.timestamp_detected)}`;

  return (
    <button
      type="button"
      className="card-alert"
      data-urgency={urgency.key}
      data-selected={String(selected)}
      data-fresh={String(needsLook(event))}
      onClick={onOpen}
      aria-label={`${incident.label}. ${whereText}. ${urgency.description}.`}
    >
      <span className="card-alert__stripe" aria-hidden="true" />

      {thumb ? (
        <img className="card-alert__thumb" src={thumb} alt="" aria-hidden="true" />
      ) : (
        <span className="card-alert__icon" aria-hidden="true">
          {incident.glyph}
        </span>
      )}

      <span className="card-alert__body">
        <span className="card-alert__title">{incident.label}</span>
        <span className="card-alert__where">{whereText}</span>
      </span>

      <span className="card-alert__tag">{urgency.word}</span>
    </button>
  );
}
