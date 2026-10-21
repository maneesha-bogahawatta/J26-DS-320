/**
 * Runtime validation of an incoming incident event.
 *
 * Why this exists: TypeScript types vanish at runtime. When a teammate's
 * Python pipeline POSTs an event, the only thing standing between a malformed
 * payload and a broken dashboard is this function. It is also wired into the
 * Integration tab so Members 1-3 can paste their pipeline's JSON output and
 * get the same errors the dashboard would raise — before they write the
 * HTTP call.
 *
 * Philosophy: reject on structural problems, warn on quality problems.
 * A warning still renders; an error does not.
 */

import type {
  BoundingBox,
  ComponentId,
  EventStatus,
  EvidenceSignal,
  IncidentEvent,
  Priority,
  RelatedTrack,
} from '../types/events';
import { COMPONENT_IDS } from './incidentTypes';

export interface ValidationIssue {
  field: string;
  message: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  /** Present only when `valid` is true. */
  event?: IncidentEvent;
}

const STATUSES: EventStatus[] = ['new', 'acknowledged', 'false_positive', 'escalated', 'resolved'];
const PRIORITIES: Priority[] = ['low', 'medium', 'high'];

/** Phrases that overclaim what CCTV can prove. Warned about, not blocked. */
const OVERCLAIMING_PHRASES = [
  'confirmed',
  'proven',
  'criminal',
  'thief',
  'medical emergency',
  'is stealing',
  'guilty',
];

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isIsoUtc(value: string): boolean {
  if (!/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(value)) return false;
  return !Number.isNaN(new Date(value).getTime());
}

function validateBoundingBox(box: unknown, errors: ValidationIssue[], warnings: ValidationIssue[]): BoundingBox | null {
  if (box === null || box === undefined) return null;
  if (!isPlainObject(box)) {
    errors.push({ field: 'bounding_box', message: 'Must be an object with x_min, y_min, x_max, y_max — or null.' });
    return null;
  }

  const keys: (keyof BoundingBox)[] = ['x_min', 'y_min', 'x_max', 'y_max'];
  const out = {} as BoundingBox;

  for (const key of keys) {
    const v = box[key];
    if (typeof v !== 'number' || Number.isNaN(v)) {
      errors.push({ field: `bounding_box.${key}`, message: 'Must be a number.' });
      return null;
    }
    if (v < 0 || v > 1) {
      errors.push({
        field: `bounding_box.${key}`,
        message: `Must be normalised to 0..1 (got ${v}). Divide pixel coordinates by frame width/height.`,
      });
      return null;
    }
    out[key] = v;
  }

  if (out.x_min >= out.x_max || out.y_min >= out.y_max) {
    warnings.push({
      field: 'bounding_box',
      message: 'Box has zero or negative area — min values should be smaller than max values.',
    });
  }
  return out;
}

function validateEvidence(raw: unknown, errors: ValidationIssue[], warnings: ValidationIssue[]): EvidenceSignal[] {
  if (raw === undefined || raw === null) {
    errors.push({ field: 'evidence', message: 'Required. Send an empty array [] if the detector produced no signals.' });
    return [];
  }
  if (!Array.isArray(raw)) {
    errors.push({ field: 'evidence', message: 'Must be an array of {name, value, unit?, description?} objects.' });
    return [];
  }
  if (raw.length === 0) {
    warnings.push({
      field: 'evidence',
      message: 'No evidence signals. The operator will see an alert with no explanation of why it fired.',
    });
  }

  const signals: EvidenceSignal[] = [];
  raw.forEach((item, i) => {
    if (!isPlainObject(item)) {
      errors.push({ field: `evidence[${i}]`, message: 'Each signal must be an object.' });
      return;
    }
    if (typeof item.name !== 'string' || !item.name.trim()) {
      errors.push({ field: `evidence[${i}].name`, message: 'Required non-empty string, e.g. "owner_separation".' });
      return;
    }
    const value = item.value;
    const validValue =
      value === null || typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean';
    if (!validValue) {
      errors.push({ field: `evidence[${i}].value`, message: 'Must be a number, string, boolean or null.' });
      return;
    }
    signals.push({
      name: item.name,
      value: value as EvidenceSignal['value'],
      unit: typeof item.unit === 'string' ? item.unit : null,
      description: typeof item.description === 'string' ? item.description : null,
    });
  });

  return signals;
}

const NAME_LIKE = /^[A-Z][a-z]+ [A-Z][a-z]+$/;

function validateRelatedTracks(
  raw: unknown,
  errors: ValidationIssue[],
  warnings: ValidationIssue[],
): RelatedTrack[] | null {
  if (raw === null || raw === undefined) return null;
  if (!Array.isArray(raw)) {
    errors.push({
      field: 'related_tracks',
      message: 'Must be an array of {track_id, role} objects — or null.',
    });
    return null;
  }

  const tracks: RelatedTrack[] = [];
  raw.forEach((item, i) => {
    if (!isPlainObject(item)) {
      errors.push({ field: `related_tracks[${i}]`, message: 'Each entry must be an object.' });
      return;
    }
    if (typeof item.track_id !== 'string' || !item.track_id.trim()) {
      errors.push({
        field: `related_tracks[${i}].track_id`,
        message: 'Required anonymous tracker label, e.g. "person-31".',
      });
      return;
    }
    if (typeof item.role !== 'string' || !item.role.trim()) {
      errors.push({
        field: `related_tracks[${i}].role`,
        message: 'Required. Suggested: likely_owner, follower, leader, counterpart, object, bystander.',
      });
      return;
    }
    if (NAME_LIKE.test(item.track_id)) {
      warnings.push({
        field: `related_tracks[${i}].track_id`,
        message: 'This looks like a person\'s name. Related tracks are anonymous labels too.',
      });
    }
    const score = item.association_score;
    if (score !== undefined && score !== null) {
      if (typeof score !== 'number' || Number.isNaN(score) || score < 0 || score > 1) {
        errors.push({
          field: `related_tracks[${i}].association_score`,
          message: 'Must be a number in 0..1, or null.',
        });
        return;
      }
    }
    tracks.push({
      track_id: item.track_id,
      role: item.role,
      note: typeof item.note === 'string' ? item.note : null,
      association_score: typeof score === 'number' ? score : null,
    });
  });

  return tracks.length > 0 ? tracks : null;
}

/** Validate an unknown payload against the v1.1 contract. */
export function validateEvent(raw: unknown): ValidationResult {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];

  if (!isPlainObject(raw)) {
    return { valid: false, errors: [{ field: '(root)', message: 'Event must be a JSON object.' }], warnings };
  }

  // --- required strings ---
  const requiredStrings = ['event_id', 'incident_type', 'camera_id', 'summary', 'model_name', 'model_version'] as const;
  for (const field of requiredStrings) {
    const v = raw[field];
    if (typeof v !== 'string' || !v.trim()) {
      errors.push({ field, message: 'Required non-empty string.' });
    }
  }

  // --- component_id ---
  if (!COMPONENT_IDS.includes(raw.component_id as ComponentId)) {
    errors.push({
      field: 'component_id',
      message: `Must be one of: ${COMPONENT_IDS.join(', ')}.`,
    });
  }

  // --- timestamp ---
  if (typeof raw.timestamp_detected !== 'string' || !isIsoUtc(raw.timestamp_detected)) {
    errors.push({
      field: 'timestamp_detected',
      message: 'Required UTC ISO-8601 string, e.g. "2026-10-18T09:15:22Z".',
    });
  } else if (!raw.timestamp_detected.endsWith('Z') && !/[+-]\d{2}:\d{2}$/.test(raw.timestamp_detected)) {
    warnings.push({
      field: 'timestamp_detected',
      message: 'No timezone marker. The group agreed on UTC with a trailing "Z" — this will be read as local time.',
    });
  }

  // --- priority / status ---
  if (!PRIORITIES.includes(raw.priority as Priority)) {
    errors.push({ field: 'priority', message: `Must be one of: ${PRIORITIES.join(', ')}.` });
  }
  if (!STATUSES.includes(raw.status as EventStatus)) {
    errors.push({ field: 'status', message: `Must be one of: ${STATUSES.join(', ')}. New alerts should send "new".` });
  }

  // --- confidence ---
  let confidence: number | null = null;
  if (raw.confidence !== undefined && raw.confidence !== null) {
    if (typeof raw.confidence !== 'number' || Number.isNaN(raw.confidence)) {
      errors.push({ field: 'confidence', message: 'Must be a number in 0..1, or null if your detector has no score.' });
    } else if (raw.confidence < 0 || raw.confidence > 1) {
      errors.push({ field: 'confidence', message: `Must be in 0..1 (got ${raw.confidence}).` });
    } else {
      confidence = raw.confidence;
    }
  } else {
    warnings.push({
      field: 'confidence',
      message: 'No score sent. The alert will show "Not provided" — acceptable for rule-based detectors.',
    });
  }

  // --- nested structures ---
  const bounding_box = validateBoundingBox(raw.bounding_box, errors, warnings);
  const evidence = validateEvidence(raw.evidence, errors, warnings);
  const related_tracks = validateRelatedTracks(raw.related_tracks, errors, warnings);

  // --- optional fields ---
  if (raw.track_id !== undefined && raw.track_id !== null && typeof raw.track_id !== 'string') {
    errors.push({ field: 'track_id', message: 'Must be a string like "person-12", or null.' });
  }
  if (raw.media_ref !== undefined && raw.media_ref !== null && typeof raw.media_ref !== 'string') {
    errors.push({ field: 'media_ref', message: 'Must be a string path, or null when no clip was saved.' });
  }
  if (
    raw.media_offset_seconds !== undefined &&
    raw.media_offset_seconds !== null &&
    (typeof raw.media_offset_seconds !== 'number' || raw.media_offset_seconds < 0)
  ) {
    errors.push({ field: 'media_offset_seconds', message: 'Must be a non-negative number of seconds, or null.' });
  }

  // --- privacy and wording checks (warnings) ---
  if (typeof raw.track_id === 'string' && NAME_LIKE.test(raw.track_id)) {
    warnings.push({
      field: 'track_id',
      message: 'This looks like a person\'s name. Track IDs must be anonymous tracker labels such as "person-12".',
    });
  }
  if (typeof raw.summary === 'string') {
    const lower = raw.summary.toLowerCase();
    const hit = OVERCLAIMING_PHRASES.find((p) => lower.includes(p));
    if (hit) {
      warnings.push({
        field: 'summary',
        message: `Summary contains "${hit}". CCTV evidence cannot prove this — use cautious wording such as "Potential …, review required".`,
      });
    }
  }
  // Relationship incidents are about two tracks. An alert that names only one
  // leaves the operator unable to check the claim the detector is making.
  const RELATIONSHIP_TYPES: Record<string, string> = {
    abandoned_object: 'likely_owner',
    tailgating: 'leader',
    unauthorized_proximity: 'counterpart',
    violence: 'counterpart',
    rapid_approach: 'counterpart',
  };
  const expectedRole = RELATIONSHIP_TYPES[String(raw.incident_type)];
  if (expectedRole && (!related_tracks || related_tracks.length === 0)) {
    warnings.push({
      field: 'related_tracks',
      message: `A "${raw.incident_type}" alert is about a relationship between tracks. Send the other track with role "${expectedRole}" so the operator can check it.`,
    });
  }

  if (confidence !== null && confidence < 0.4) {
    warnings.push({
      field: 'confidence',
      message: 'Low score. Consider whether this should reach the operator queue or only the evaluation log.',
    });
  }

  if (errors.length > 0) {
    return { valid: false, errors, warnings };
  }

  const event: IncidentEvent = {
    event_id: raw.event_id as string,
    component_id: raw.component_id as ComponentId,
    incident_type: raw.incident_type as string,
    camera_id: raw.camera_id as string,
    timestamp_detected: raw.timestamp_detected as string,
    priority: raw.priority as Priority,
    status: raw.status as EventStatus,
    summary: raw.summary as string,
    evidence,
    model_name: raw.model_name as string,
    model_version: raw.model_version as string,
    confidence,
    bounding_box,
    track_id: typeof raw.track_id === 'string' ? raw.track_id : null,
    related_tracks,
    media_ref: typeof raw.media_ref === 'string' ? raw.media_ref : null,
    media_offset_seconds: typeof raw.media_offset_seconds === 'number' ? raw.media_offset_seconds : null,
    score_calibrated: raw.score_calibrated === true,
    zone_id: typeof raw.zone_id === 'string' ? raw.zone_id : null,
    audit: Array.isArray(raw.audit) ? (raw.audit as IncidentEvent['audit']) : [],
  };

  return { valid: true, errors, warnings, event };
}
