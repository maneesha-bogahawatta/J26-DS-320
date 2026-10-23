/**
 * Plain-language layer.
 *
 * The people using this are security officers, not data scientists. A row
 * reading `low_posture_duration: 4.1 seconds` is correct and useless; what an
 * officer needs is "Stayed down for 4.1 seconds".
 *
 * The system underneath stays as complex as it needs to be. This module is the
 * single place where machine vocabulary is turned into something a person can
 * act on in two seconds, under pressure, possibly at 3am.
 *
 * Rules for every phrase written here:
 *   - Say what the camera SAW, never what it means. "Has not moved since" is
 *     an observation; "unconscious" is a diagnosis we cannot make.
 *   - No jargon, no units the officer would not use, no model vocabulary.
 *   - Keep the uncertainty. "Looks like", "about", "may have".
 *
 * The raw machine values are not deleted — they live in the "Technical
 * details" section for the project team.
 */

import type { EvidenceSignal, IncidentEvent } from '../types/events';

/* ------------------------------------------------------------------ time */

/** 92 -> "1 min 32 sec". Officers read clock time, not decimal seconds. */
export function plainDuration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s} sec`;
  const minutes = Math.floor(s / 60);
  const rest = s % 60;
  if (minutes < 60) return rest === 0 ? `${minutes} min` : `${minutes} min ${rest} sec`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min`;
}

/** "Just now", "4 minutes ago" — the only time format in the operator view. */
export function plainTimeAgo(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return 'Time unknown';
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return 'Time unknown';

  const seconds = Math.round((now - t) / 1000);
  if (seconds < 0) return 'Just now';
  if (seconds < 60) return 'Just now';
  const minutes = Math.round(seconds / 60);
  if (minutes === 1) return '1 minute ago';
  if (minutes < 60) return `${minutes} minutes ago`;
  const hours = Math.round(minutes / 60);
  if (hours === 1) return '1 hour ago';
  if (hours < 24) return `${hours} hours ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? 'Yesterday' : `${days} days ago`;
}

/** The clock time an officer would write in a logbook. */
export function plainClock(iso: string | null | undefined): string {
  if (!iso) return '--:--';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '--:--';
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
}

/* --------------------------------------------------------------- tracks */

/**
 * "person-12" -> "Person 12".
 *
 * Still an anonymous tracker label — we are only making it readable. The
 * system holds no names, and this does not change that.
 */
export function plainTrack(trackId: string | null | undefined): string | null {
  if (!trackId) return null;
  const match = /^([a-z]+)[-_]?(\d+)$/i.exec(trackId.trim());
  if (!match) return trackId;
  const [, kind, number] = match;
  const label = kind.charAt(0).toUpperCase() + kind.slice(1).toLowerCase();
  return `${label} ${number}`;
}

/** "likely_owner" -> "Person who left it" */
export const ROLE_LABELS: Record<string, string> = {
  likely_owner: 'Person who may have left it',
  follower: 'Person following',
  leader: 'Person in front',
  counterpart: 'Other person involved',
  object: 'The object',
  bystander: 'Someone nearby',
};

export function plainRole(role: string): string {
  return ROLE_LABELS[role] ?? role.replace(/[_-]+/g, ' ');
}

/* ---------------------------------------------------- match strength ---- */

export interface MatchStrength {
  /** One word an officer can read at a glance. */
  word: string;
  /** Which band, for styling. */
  band: 'strong' | 'moderate' | 'weak' | 'none';
  /** The sentence that keeps them from over-trusting it. */
  caveat: string;
}

/**
 * Turn a raw score into words.
 *
 * Deliberately NOT a percentage. "82%" reads as "82% chance this is real",
 * which an uncalibrated model score absolutely does not mean. A word band
 * carries the same information with none of the false precision.
 */
export function matchStrength(event: IncidentEvent): MatchStrength {
  const score = event.confidence;

  if (score === null || score === undefined || Number.isNaN(score)) {
    return {
      word: 'Not rated',
      band: 'none',
      caveat: 'This check works on fixed rules and gives no rating. Judge it from what the camera saw.',
    };
  }

  if (event.score_calibrated) {
    const percent = Math.round(score * 100);
    return {
      word: `${percent}% likely`,
      band: score >= 0.7 ? 'strong' : score >= 0.4 ? 'moderate' : 'weak',
      caveat: 'This figure has been checked against real cases. It is still a guide, not a decision.',
    };
  }

  if (score >= 0.8) {
    return {
      word: 'Strong match',
      band: 'strong',
      caveat: 'Closely matches the pattern the system looks for. A strong match is still not proof — check the video.',
    };
  }
  if (score >= 0.6) {
    return {
      word: 'Fair match',
      band: 'moderate',
      caveat: 'Partly matches the pattern. Worth a look.',
    };
  }
  if (score >= 0.4) {
    return {
      word: 'Weak match',
      band: 'weak',
      caveat: 'Only loosely matches the pattern. Often turns out to be nothing.',
    };
  }
  return {
    word: 'Very weak match',
    band: 'weak',
    caveat: 'Barely matches the pattern. Most alerts like this are false alarms.',
  };
}

/* ------------------------------------------------------- evidence ------ */

export interface PlainSignal {
  /** The sentence the officer reads. */
  text: string;
  /** True when this signal argues AGAINST the alert being real. */
  reassuring?: boolean;
  /** True when this signal means the camera's view was poor. */
  reliability?: boolean;
}

function num(value: EvidenceSignal['value']): number | null {
  return typeof value === 'number' && !Number.isNaN(value) ? value : null;
}

/**
 * Translate one machine signal into one plain sentence.
 *
 * Each entry answers: what would a person watching the screen have noticed?
 */
const TRANSLATORS: Record<string, (s: EvidenceSignal) => PlainSignal | null> = {
  // ---------- C4: falls ----------
  downward_motion: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: v >= 0.6 ? 'Dropped down suddenly' : 'Moved downwards, not sharply' };
  },
  posture_aspect_ratio: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 1.5
      ? { text: 'Ended up lying flat' }
      : { text: 'Still more upright than flat', reassuring: true };
  },
  torso_angle: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 60 ? { text: 'Body almost horizontal' } : { text: 'Body still fairly upright', reassuring: true };
  },
  low_posture_duration: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 3
      ? { text: `Stayed down for ${plainDuration(v)}` }
      : { text: `Was only down for ${plainDuration(v)}`, reassuring: true };
  },
  post_fall_movement: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v <= 0.2
      ? { text: 'Has not got up since' }
      : { text: 'Got up and moved again', reassuring: true };
  },

  // ---------- C4: abandoned objects ----------
  object_dwell_time: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Left sitting there for ${plainDuration(v)}` };
  },
  owner_separation: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 4
      ? { text: `The person who left it is now about ${Math.round(v)} m away` }
      : { text: `The person who left it is still close by, about ${Math.round(v)} m`, reassuring: true };
  },
  object_displacement: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v <= 0.1 ? { text: 'The item has not moved at all' } : { text: 'The item has shifted slightly' };
  },
  owner_association_score: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    if (v >= 0.7) return { text: 'Fairly clear who put it down' };
    if (v >= 0.4) return { text: 'Not certain who put it down' };
    return { text: 'Could not work out who put it down' };
  },

  // ---------- C1: spatial ----------
  entry_gap: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `A second person came through ${plainDuration(v)} behind the first` };
  },
  follower_distance: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Following about ${v.toFixed(1)} m behind` };
  },
  person_count: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `${v} people in view` };
  },
  baseline_count: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Normally about ${v} people here` };
  },
  zone_dwell: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `In the restricted area for ${plainDuration(v)}` };
  },
  rise_window: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `The crowd built up in ${plainDuration(v)}` };
  },

  // ---------- C2: loitering ----------
  dwell_time: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Has been there ${plainDuration(v)}` };
  },
  path_reversals: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 4
      ? { text: `Turned back on themselves ${v} times` }
      : { text: `Changed direction ${v} time${v === 1 ? '' : 's'}`, reassuring: v <= 1 };
  },
  trajectory_entropy: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 0.6
      ? { text: 'Walking about with no clear purpose' }
      : { text: 'Moving in a fairly direct line', reassuring: true };
  },
  queue_alignment: (s) =>
    s.value === true
      ? { text: 'Standing in the queue like everyone else', reassuring: true }
      : { text: 'Not standing in the queue' },

  // ---------- C3: aggression ----------
  approach_velocity: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 2 ? { text: 'Moved at speed towards another person' } : { text: 'Walked towards another person' };
  },
  closing_distance: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Came within about ${v.toFixed(1)} m of them` };
  },
  torso_lean: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v >= 0.35 ? { text: 'Leaning forward towards them' } : { text: 'Upright posture', reassuring: true };
  },
  inter_skeleton_velocity: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: 'Fast movement between two people' };
  },
  limb_contact_events: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Arms or legs appeared to make contact ${v} time${v === 1 ? '' : 's'}` };
  },
  hand_to_torso_events: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Hand moved to body ${v} time${v === 1 ? '' : 's'}` };
  },
  dwell_at_shelf: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Stood at the shelf for ${plainDuration(v)}` };
  },
  // ---------- practice detector (upload + live screens) ----------
  motion_area: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    const pct = Math.max(1, Math.round(v * 100));
    return { text: `Movement covered about ${pct}% of the picture` };
  },
  motion_duration: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return { text: `Movement went on for ${plainDuration(v)}` };
  },
  position_spread: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v < 0.15 ? { text: 'Stayed in almost exactly the same spot' } : { text: 'Moved around within a small area' };
  },
  object_size: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    return v < 0.01 ? { text: 'Something small appeared and stayed' } : { text: 'Something appeared and stayed' };
  },

  occlusion_ratio: (s) => {
    const v = num(s.value);
    if (v === null) return null;
    if (v >= 0.4)
      return {
        text: 'The camera view was partly blocked, so this is less reliable',
        reliability: true,
      };
    return { text: 'The camera had a clear view', reliability: true };
  },
};

/** Turn one machine signal into a sentence, or fall back to something readable. */
export function plainSignal(signal: EvidenceSignal): PlainSignal {
  const translator = TRANSLATORS[signal.name];
  if (translator) {
    const result = translator(signal);
    if (result) return result;
  }

  // Unknown signal from a teammate's new detector. Still make it readable
  // rather than dumping snake_case at an officer.
  const name = signal.name.replace(/[_-]+/g, ' ');
  const readable = name.charAt(0).toUpperCase() + name.slice(1);
  if (signal.value === null || signal.value === undefined) return { text: readable };
  if (typeof signal.value === 'boolean') return { text: `${readable}: ${signal.value ? 'yes' : 'no'}` };
  const value =
    typeof signal.value === 'number' && !Number.isInteger(signal.value)
      ? signal.value.toFixed(2)
      : String(signal.value);
  return { text: `${readable}: ${value}${signal.unit ? ` ${signal.unit}` : ''}` };
}

/* ------------------------------------------------------ camera names --- */

/**
 * "Clip 03 — Corridor (synthetic)" -> "Corridor".
 *
 * An officer knows their building by its places, not by clip filenames. Falls
 * back to the raw id so a camera nobody has labelled still shows something.
 */
export function plainCamera(label: string | undefined, cameraId: string): string {
  if (!label) return cameraId;
  const afterDash = label.includes('—') ? label.split('—').slice(1).join('—') : label;
  return afterDash.replace(/\(synthetic\)/i, '').replace(/^\s*[-–—]\s*/, '').trim() || cameraId;
}
