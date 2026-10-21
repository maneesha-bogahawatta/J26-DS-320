/**
 * Formatting helpers.
 *
 * Every function here assumes a value may be missing. The dashboard shows
 * "Not provided" rather than a blank, a zero, or a crash, because the contract
 * explicitly allows pipelines to send `null`.
 */

import type { EvidenceSignal, EventStatus, Priority } from '../types/events';

export const NOT_PROVIDED = 'Not provided';

/** Local wall-clock time for the operator, e.g. "18 Oct 2026, 14:32:07". */
export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return NOT_PROVIDED;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return `Invalid timestamp (${iso})`;
  return d.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

/** Just the clock portion, for the dense queue rows. */
export function formatClock(iso: string | null | undefined): string {
  if (!iso) return '--:--:--';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '--:--:--';
  return d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

/** "4 min ago" — helps an operator see what is fresh in the queue. */
export function formatRelative(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return NOT_PROVIDED;
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return NOT_PROVIDED;

  const seconds = Math.round((now - t) / 1000);
  if (seconds < 0) return 'in the future';
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return `${days} d ago`;
}

/**
 * Format a model score. Deliberately returns the raw 0..1 value as a
 * two-decimal number, NOT a percentage — a percentage reads as a calibrated
 * probability, which these scores are not.
 */
export function formatScore(confidence: number | null | undefined): string {
  if (confidence === null || confidence === undefined) return NOT_PROVIDED;
  if (Number.isNaN(confidence)) return NOT_PROVIDED;
  return confidence.toFixed(2);
}

/** Label that goes next to the score, so it is never mistaken for a probability. */
export function scoreLabel(calibrated: boolean | undefined): string {
  return calibrated ? 'Calibrated probability' : 'Model score (uncalibrated)';
}

/** Render one evidence signal's value with its unit. */
export function formatSignalValue(signal: EvidenceSignal): string {
  const { value, unit } = signal;
  if (value === null || value === undefined) return NOT_PROVIDED;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') {
    const rendered = Number.isInteger(value) ? String(value) : value.toFixed(2);
    return unit ? `${rendered} ${unit}` : rendered;
  }
  return unit ? `${value} ${unit}` : String(value);
}

/**
 * Labels shown to security officers.
 *
 * The machine values stay as the group agreed (`false_positive`,
 * `escalated`…) — those are the contract and the CSV columns. These are the
 * words on screen. An officer should never have to learn that "escalated"
 * is this system's word for "I sent someone".
 */
export const STATUS_LABELS: Record<EventStatus, string> = {
  new: 'Needs a look',
  acknowledged: 'Seen',
  false_positive: 'Nothing wrong',
  escalated: 'Help sent',
  resolved: 'Dealt with',
};

/** Priority as an instruction, not a severity grade. */
export const PRIORITY_LABELS: Record<Priority, string> = {
  low: 'When you can',
  medium: 'Soon',
  high: 'Now',
};

export const ACTION_LABELS = {
  acknowledge: 'Marked as seen',
  false_alarm: 'Marked nothing wrong',
  escalate: 'Sent help',
  resolve: 'Marked dealt with',
  created: 'Alert came in',
} as const;

/** The machine values, for the technical panel and the CSV export. */
export const STATUS_MACHINE_LABELS: Record<EventStatus, string> = {
  new: 'new',
  acknowledged: 'acknowledged',
  false_positive: 'false_positive',
  escalated: 'escalated',
  resolved: 'resolved',
};

/** Sort weight so high-priority items can float to the top of the queue. */
export const PRIORITY_WEIGHT: Record<Priority, number> = {
  high: 3,
  medium: 2,
  low: 1,
};

/** Shorten a media path for display without hiding which file it is. */
export function shortenPath(path: string | null | undefined, max = 38): string {
  if (!path) return NOT_PROVIDED;
  if (path.length <= max) return path;
  const file = path.split('/').pop() ?? path;
  return file.length <= max ? `…/${file}` : `…${file.slice(-(max - 1))}`;
}
