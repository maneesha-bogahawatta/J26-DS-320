/**
 * A bar the length of the video with a coloured mark wherever something was
 * found. Tap a mark to jump straight to that moment.
 */

import type { Found } from '../../detect';
import { describeIncident } from '../../lib/incidentTypes';
import { urgencyOf } from '../../lib/urgency';
import { videoClock } from '../../lib/geometry';

interface TimelineProps {
  duration: number;
  /** Original-video start of a rendered excerpt; ordinary uploads start at zero. */
  startSeconds?: number;
  current: number;
  found: Found[];
  selectedId: string | null;
  onPick: (f: Found) => void;
  onSeek: (t: number) => void;
}

export function Timeline({ duration, startSeconds = 0, current, found, selectedId, onPick, onSeek }: TimelineProps) {
  if (!duration) return null;
  const pct = (t: number) => `${Math.min(100, Math.max(0, ((t - startSeconds) / duration) * 100))}%`;

  return (
    <div className="timeline">
      <div
        className="timeline__track"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(startSeconds + ((e.clientX - r.left) / r.width) * duration);
        }}
        role="presentation"
      >
        <span className="timeline__played" style={{ width: pct(current) }} />
        {found.filter(f => f.endSeconds >= startSeconds && f.startSeconds <= startSeconds + duration).map((f) => {
          const meta = describeIncident(f.event.incident_type, f.event.component_id);
          const width = Math.max(1.2, ((Math.min(f.endSeconds, startSeconds + duration) - Math.max(f.startSeconds, startSeconds)) / duration) * 100);
          return (
            <button
              key={f.event.event_id}
              type="button"
              className="timeline__mark"
              data-urgency={urgencyOf(f.event).key}
              data-selected={String(selectedId === f.event.event_id)}
              style={{ left: pct(f.startSeconds), width: `${width}%` }}
              title={`${meta.label} at ${videoClock(f.startSeconds)}`}
              aria-label={`${meta.label} at ${videoClock(f.startSeconds)}`}
              onClick={(e) => {
                e.stopPropagation();
                onPick(f);
              }}
            />
          );
        })}
        <span className="timeline__head" style={{ left: pct(current) }} />
      </div>
      <div className="timeline__clock">
        <span className="mono">{videoClock(current)}</span>
        <span className="mono">{videoClock(startSeconds + duration)}</span>
      </div>
    </div>
  );
}
