/**
 * J26-DS-320 — Shared Incident Event Contract (v1.1)
 *
 * This file is the TypeScript mirror of `shared/schemas/event_schema.py`.
 * EVERY detection component (C1-C4) emits objects in this shape.
 *
 * Rules agreed by the group:
 *   - All timestamps are UTC, ISO-8601, with a trailing "Z".
 *   - Bounding boxes use NORMALISED coordinates in the range 0..1.
 *   - `track_id` is an anonymous tracker label (e.g. "person-12"). It is NOT
 *     a person's name, identity, or any biometric reference.
 *   - If a value is unavailable, send `null`. Never invent a value.
 *   - `confidence` is a MODEL SCORE, not a probability, unless the owning
 *     component has explicitly calibrated it and sets `score_calibrated: true`.
 *
 * Do not change this file without team agreement — it is the integration
 * contract between all four components and the dashboard.
 */

/** Which group component produced the event. */
export type ComponentId =
  | 'component_1' // C1 — Spatial Anomaly Detection
  | 'component_2' // C2 — Predictive Loitering Detection
  | 'component_3' // C3 — Aggression & Violence Recognition
  | 'component_4'; // C4 — Fall & Abandoned-Object Detection

/** Lifecycle state of an alert, as recorded by the operator. */
export type EventStatus =
  | 'new'
  | 'acknowledged'
  | 'false_positive'
  | 'escalated'
  | 'resolved';

/** Agreed priority labels. Set by the producing component. */
export type Priority = 'low' | 'medium' | 'high';

/** Actions an operator can record against an alert. */
export type OperatorAction =
  | 'acknowledge'
  | 'false_alarm'
  | 'escalate'
  | 'resolve';

/**
 * `incident_type` is intentionally an open string, not a closed union.
 * The dashboard must render an event from a component it has never seen
 * before rather than crashing. Known values are listed in
 * `src/lib/incidentTypes.ts`; anything else degrades to a readable fallback.
 */
export type IncidentType = string;

/**
 * One behavioural signal that contributed to the alert.
 * Example: { name: "low_posture_duration", value: 3.4, unit: "seconds" }
 */
export interface EvidenceSignal {
  /** snake_case machine name, e.g. "owner_separation". */
  name: string;
  /** Numeric, textual or boolean value. `null` when the signal fired but carries no measurement. */
  value: number | string | boolean | null;
  /** Unit of measurement, e.g. "seconds", "metres", "px". `null` for unitless scores. */
  unit?: string | null;
  /** Optional plain-language explanation shown to the operator (the XAI layer). */
  description?: string | null;
}

/**
 * A second track involved in the same incident.
 *
 * Some incidents are about ONE track (a person falling). Others are
 * inherently about a RELATIONSHIP between tracks, and a single `track_id`
 * cannot express them:
 *
 *   - abandoned_object : the object track AND its likely owner track
 *   - tailgating       : the leading track AND the follower
 *   - violence         : both parties
 *
 * In those cases `track_id` holds the subject of the alert (the object, the
 * follower) and `related_tracks` holds the others with their role. The role
 * is an open string so a component can name a relationship the dashboard has
 * not seen before.
 */
export interface RelatedTrack {
  /** Anonymous tracker label, e.g. "person-31". Never a name. */
  track_id: string;
  /** Suggested: likely_owner | follower | leader | counterpart | object | bystander */
  role: string;
  /** Optional note, e.g. how the association was inferred. */
  note?: string | null;
  /**
   * How confident the component is in this association, 0..1, or null.
   * For abandoned objects this is the owner-association estimate — it matters
   * because CCTV cannot prove ownership, only infer it.
   */
  association_score?: number | null;
}

/** Normalised bounding box. All four values are in the range 0..1. */
export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

/** One entry in the immutable audit trail for an event. */
export interface AuditEntry {
  /** `created` is written by the pipeline; the rest are operator actions. */
  action: OperatorAction | 'created';
  /** UTC ISO-8601 timestamp of the action. */
  at: string;
  /** Operator label, e.g. "OP-01". Free text; not linked to any surveilled person. */
  operator_id: string | null;
  /** Optional free-text note the operator added. */
  note: string | null;
  from_status: EventStatus | null;
  to_status: EventStatus;
}

/**
 * The incident event. This is what a detection pipeline POSTs to
 * `/api/v1/events` and what the dashboard renders.
 */
export interface IncidentEvent {
  // ---- Required: every pipeline must supply these ----

  /** Unique across the whole system. Suggested form: "<component>-<uuid4>". */
  event_id: string;
  component_id: ComponentId;
  incident_type: IncidentType;
  /** Camera or recorded-clip identifier, e.g. "clip_03" or "cam_lobby_01". */
  camera_id: string;
  /** UTC ISO-8601, e.g. "2026-10-18T09:15:22Z". */
  timestamp_detected: string;
  priority: Priority;
  status: EventStatus;
  /** Short, cautious description. "Potential fall — review required", never "Fall confirmed". */
  summary: string;
  /** Signals that led to the alert. May be an empty array, never missing. */
  evidence: EvidenceSignal[];
  model_name: string;
  model_version: string;

  // ---- Optional: send null when unavailable ----

  /** Model/rule score in 0..1, or null if the detector produces no score. */
  confidence?: number | null;
  /** Normalised box around the subject of the alert, or null. */
  bounding_box?: BoundingBox | null;
  /**
   * Anonymous tracker label for the SUBJECT of the alert, or null when
   * tracking was not available. For an abandoned object this is the object
   * track; the likely owner goes in `related_tracks`.
   */
  track_id?: string | null;
  /** Other tracks involved in the same incident, with their role. */
  related_tracks?: RelatedTrack[] | null;
  /** Path or URL to the clip/frame, relative to the media root. Null if no media was saved. */
  media_ref?: string | null;

  // ---- v1.1 additions (optional, backwards compatible) ----

  /**
   * Seconds into `media_ref` where the event occurs, so the dashboard can seek
   * straight to the moment instead of making the operator scrub. Null if unknown.
   */
  media_offset_seconds?: number | null;
  /**
   * Set to `true` ONLY if your component has calibrated its score against a
   * held-out set. When false/absent the dashboard labels it "model score
   * (uncalibrated)" and refuses to present it as a probability.
   */
  score_calibrated?: boolean;
  /** Zone or region label the event occurred in, e.g. "atm_zone". Null if not applicable. */
  zone_id?: string | null;
  /** Operator action history. The dashboard maintains this; pipelines may omit it. */
  audit?: AuditEntry[];
}

/** A camera or recorded clip the system can show. */
export interface CameraSource {
  camera_id: string;
  /** Human-readable label shown in the UI, e.g. "Clip 03 — Bank lobby". */
  label: string;
  /** `recorded_clip` for the prototype; `live_stream` only once a feed is really connected. */
  kind: 'recorded_clip' | 'live_stream';
  /** Path to the media file, or null when the clip has not been placed on disk. */
  media_ref: string | null;
  /** Where the footage came from. Shown to the operator for provenance. */
  source_note?: string | null;
}

/** Response of `GET /api/v1/health`. */
export interface HealthStatus {
  ok: boolean;
  /** "mock" when serving seeded data, "live" when backed by real pipelines. */
  mode: 'mock' | 'live';
  api_version: string;
  /** Number of detection components that have POSTed at least one event this session. */
  components_reporting: ComponentId[];
  checked_at: string;
}

/** Filters applied to the alert queue. */
export interface EventFilters {
  incidentTypes: string[];
  components: ComponentId[];
  cameras: string[];
  statuses: EventStatus[];
  priorities: Priority[];
  /** Free-text match across summary, event_id and track_id. */
  search: string;
}

export const EMPTY_FILTERS: EventFilters = {
  incidentTypes: [],
  components: [],
  cameras: [],
  statuses: [],
  priorities: [],
  search: '',
};

/** Statuses that still need an operator to look at them. */
export const UNREVIEWED_STATUSES: EventStatus[] = ['new'];

/** Which status an operator action moves an event into. */
export const ACTION_TARGET_STATUS: Record<OperatorAction, EventStatus> = {
  acknowledge: 'acknowledged',
  false_alarm: 'false_positive',
  escalate: 'escalated',
  resolve: 'resolved',
};
