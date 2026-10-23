/**
 * Urgency: the single idea the whole interface is colour-coded around.
 *
 * An alert's colour answers one question — "does this need me right now?" —
 * and it combines two things the officer should not have to combine in their
 * head: how urgent the detector said it was, and whether anyone has dealt
 * with it yet.
 *
 * A high-priority alert that has been resolved is GREEN, not red. A
 * low-priority one nobody has looked at is still blue, not invisible. That
 * merge is the whole point: the officer reads the colour, not the two fields.
 */

import type { IncidentEvent } from '../types/events';

export type UrgencyKey = 'now' | 'soon' | 'later' | 'done' | 'closed';

export interface Urgency {
  key: UrgencyKey;
  /** The word on the card. Colour is never the only signal. */
  word: string;
  /** Spoken form for screen readers and tooltips. */
  description: string;
}

const URGENCIES: Record<UrgencyKey, Urgency> = {
  now: { key: 'now', word: 'NOW', description: 'Needs attention now' },
  soon: { key: 'soon', word: 'SOON', description: 'Needs attention soon' },
  later: { key: 'later', word: 'WHEN YOU CAN', description: 'Look at this when you have time' },
  done: { key: 'done', word: 'DEALT WITH', description: 'This has been handled' },
  closed: { key: 'closed', word: 'NOTHING WRONG', description: 'Marked as a false alarm' },
};

/** The one function that decides an alert's colour. */
export function urgencyOf(event: IncidentEvent): Urgency {
  // Already handled? Then it is not urgent any more, whatever the detector said.
  if (event.status === 'resolved') return URGENCIES.done;
  if (event.status === 'false_positive') return URGENCIES.closed;

  // Help is on the way, but it is still live — keep it red.
  if (event.status === 'escalated') return URGENCIES.now;

  // Someone has seen it but not closed it: step it down one level, so it stops
  // competing with things nobody has looked at.
  if (event.status === 'acknowledged') {
    return event.priority === 'high' ? URGENCIES.soon : URGENCIES.later;
  }

  // Nobody has looked at it yet — the detector's priority stands.
  if (event.priority === 'high') return URGENCIES.now;
  if (event.priority === 'medium') return URGENCIES.soon;
  return URGENCIES.later;
}

/** Sort order for the list: red first, then amber, then blue, then finished. */
const SORT_WEIGHT: Record<UrgencyKey, number> = {
  now: 5,
  soon: 4,
  later: 3,
  done: 2,
  closed: 1,
};

export function sortByUrgency(events: IncidentEvent[]): IncidentEvent[] {
  return [...events].sort((a, b) => {
    const weight = SORT_WEIGHT[urgencyOf(b).key] - SORT_WEIGHT[urgencyOf(a).key];
    if (weight !== 0) return weight;
    return new Date(b.timestamp_detected).getTime() - new Date(a.timestamp_detected).getTime();
  });
}

/** True when nobody has made a decision on this alert yet. */
export function needsLook(event: IncidentEvent): boolean {
  return event.status === 'new';
}
