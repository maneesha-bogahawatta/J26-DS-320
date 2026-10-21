/**
 * The practice detector.
 *
 * Feed it frames in time order — from an uploaded video being stepped through,
 * or from a live camera — and it reports anything that matches one of five
 * simple patterns, one for each component's territory:
 *
 *   C1  zone_activity     something moved in view for a while
 *   C2  loitering         movement that stayed in one spot for a long time
 *   C3  aggression        a lot of fast movement at once
 *   C4  fall              movement that dropped, then a long flat shape stayed still
 *   C4  abandoned_object  a small new thing appeared and stayed after the mover left
 *
 * These are rules of thumb on a 64x36 grid, NOT the trained models. They are
 * here so the upload and live screens work end to end today. Every event they
 * produce is labelled as a practice check, carries `score_calibrated: false`,
 * and reports only what was actually measured.
 *
 * The same class runs offline (upload) and in real time (live): it only ever
 * looks at the frame it is given and what it remembers from earlier ones.
 */

import type { BoundingBox, ComponentId, EvidenceSignal, IncidentEvent, Priority } from '../types/events';
import { overlap } from '../lib/geometry';
import type { Found, PathPoint } from './engine';
import {
  Background,
  FrameSampler,
  activeCells,
  countCells,
  dilate,
  ghostCells,
  lightStats,
  largestRegion,
  subtract,
  CELL_COUNT,
} from './motion';

export const PRACTICE_MODEL = 'practice-motion-check';

/**
 * Tuning aid. In the browser console run
 *     window.__PRACTICE_DEBUG = []
 * then check a video: every decision the detector makes is pushed into that
 * array with the numbers behind it. Leave it unset normally — it costs nothing.
 */
function debug(info: Record<string, unknown>): void {
  const sink = (globalThis as { __PRACTICE_DEBUG?: unknown[] }).__PRACTICE_DEBUG;
  if (Array.isArray(sink)) sink.push(info);
}
export const PRACTICE_VERSION = '0.1.0';

/* ----------------------------------------------------------- tuning --- */

/** More of the picture than this changing at once means the camera moved. */
const CAMERA_MOVED_FRACTION = 0.42;
/** A moving region must be at least this many cells. */
const MIN_MOTION_CELLS = 5;
/** A still-change region must be at least this many cells. */
const MIN_STATIC_CELLS = 2;
/** No movement for this long ends a burst of movement. */
const SEGMENT_GAP = 0.8;
/** Movement shorter than this is ignored altogether. */
const MIN_SEGMENT = 1.2;
/** How long a new still thing must stay before it counts. */
const STILL_SECONDS = 3;
/** Movement covering this much of the picture is "a lot". */
const BIG_MOTION = 0.11;
/** ...for at least this long. */
const BIG_MOTION_SECONDS = 1.5;
/** A patch this bright (0..255) that got brighter is glare, not an object. */
const GLARE_LEVEL = 200;
/** Background half-life, seconds. */
const BACKGROUND_TAU = 20;

interface Options {
  /** 'live' adds a cooldown so a person waving does not raise twenty alerts. */
  mode: 'upload' | 'live';
  sourceId: string;
  mediaRef: string | null;
  /** Loitering threshold. Shorter for uploads, which are usually short clips. */
  loiterSeconds: number;
  /** Live only: minimum seconds between two alerts of the same kind. */
  cooldownSeconds: number;
}

interface Segment {
  start: number;
  end: number;
  points: (PathPoint & { area: number })[];
  peakArea: number;
  bigSince: number | null;
  /** Something more specific than "movement" was raised from this segment. */
  claimed: boolean;
  loiterRaised: boolean;
  aggressionRaised: boolean;
}

interface StillThing {
  box: BoundingBox;
  area: number;
  since: number;
  last: number;
  raised: boolean;
}

export interface PushResult {
  moving: BoundingBox | null;
  cameraMoving: boolean;
  found: Found[];
}

function iso(date = new Date()): string {
  return date.toISOString().replace(/\.\d{3}Z$/, 'Z');
}

function shortId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function round(n: number, places = 2): number {
  const f = 10 ** places;
  return Math.round(n * f) / f;
}

function centre(box: BoundingBox): { x: number; y: number } {
  return { x: (box.x_min + box.x_max) / 2, y: (box.y_min + box.y_max) / 2 };
}

/** Width over height in real pixels, given the frame's own shape. */
function pixelAspect(box: BoundingBox, frameAspect: number): number {
  const w = (box.x_max - box.x_min) * frameAspect;
  const h = box.y_max - box.y_min;
  return h > 0 ? w / h : 0;
}

/** How far the centre wandered during a burst of movement, 0..~1.4. */
function spread(points: PathPoint[]): number {
  if (points.length < 2) return 0;
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const p of points) {
    const c = centre(p.box);
    minX = Math.min(minX, c.x);
    maxX = Math.max(maxX, c.x);
    minY = Math.min(minY, c.y);
    maxY = Math.max(maxY, c.y);
  }
  return Math.hypot(maxX - minX, maxY - minY);
}

interface Kind {
  type: string;
  component: ComponentId;
  priority: Priority;
  track: 'person' | 'object' | 'movement';
  summary: string;
}

const KINDS: Record<string, Kind> = {
  zone_activity: {
    type: 'zone_activity',
    component: 'component_1',
    priority: 'low',
    track: 'movement',
    summary: 'Practice check: movement in view — review the video',
  },
  loitering: {
    type: 'loitering',
    component: 'component_2',
    priority: 'low',
    track: 'person',
    summary: 'Practice check: movement stayed in one spot for a long time — review required',
  },
  aggression: {
    type: 'aggression',
    component: 'component_3',
    priority: 'medium',
    track: 'person',
    summary: 'Practice check: a lot of fast movement at once — review required',
  },
  fall: {
    type: 'fall',
    component: 'component_4',
    priority: 'high',
    track: 'person',
    summary: 'Practice check: possible fall — someone may be down, review now',
  },
  abandoned_object: {
    type: 'abandoned_object',
    component: 'component_4',
    priority: 'medium',
    track: 'object',
    summary: 'Practice check: something new appeared and stayed — review required',
  },
};

export class PracticeAnalyzer {
  private sampler = new FrameSampler();
  private prev: Uint8Array | null = null;
  private background: Background | null = null;
  private lastT: number | null = null;

  private segment: Segment | null = null;
  /** Finished segments, kept a while so a later still-thing can look back at them. */
  private recent: Segment[] = [];
  /** Finished segments waiting to see whether something more specific claims them. */
  private pending: { segment: Segment; decideAt: number }[] = [];
  private still: StillThing | null = null;
  /** Where falls and left items have already been reported. */
  private reportedSpots: BoundingBox[] = [];

  private lastRaised: Record<string, number> = {};
  private counters = { person: 0, object: 0, movement: 0 };
  private opts: Options;

  constructor(opts: Partial<Options> & Pick<Options, 'mode' | 'sourceId'>) {
    this.opts = {
      mediaRef: null,
      loiterSeconds: opts.mode === 'live' ? 12 : 8,
      cooldownSeconds: opts.mode === 'live' ? 12 : 0,
      ...opts,
    };
  }

  /**
   * Look at one frame.
   *
   * @param source       the video element (or anything drawable) at time `t`
   * @param t            seconds into the video, or since the camera started
   * @param frameAspect  the frame's width / height, for shape checks
   */
  push(source: CanvasImageSource, t: number, frameAspect: number): PushResult {
    const frame = this.sampler.grab(source);
    const dt = this.lastT === null ? 0 : t - this.lastT;
    this.lastT = t;

    if (!this.prev || !this.background) {
      this.prev = frame;
      this.background = new Background(frame);
      return { moving: null, cameraMoving: false, found: [] };
    }

    const motionCells = activeCells(frame, this.prev);
    this.prev = frame;
    const motionFraction = countCells(motionCells) / CELL_COUNT;

    // The whole picture changed: the camera was moved or the lights changed.
    // Start again from this frame rather than raising nonsense.
    if (motionFraction > CAMERA_MOVED_FRACTION) {
      this.background.reset(frame);
      this.segment = null;
      this.still = null;
      return { moving: null, cameraMoving: true, found: this.decidePending(t) };
    }

    const motion = largestRegion(motionCells, MIN_MOTION_CELLS);
    // Everything within two squares of movement counts as "moving", so the
    // edges of a walking person never look like something left behind.
    const near = dilate(dilate(motionCells));
    this.background.update(frame, near, dt, BACKGROUND_TAU);

    const found: Found[] = [];
    found.push(...this.trackMovement(motion, t));
    found.push(...this.trackStillThings(frame, near, motion, t, frameAspect));
    found.push(...this.decidePending(t));

    return { moving: motion?.box ?? null, cameraMoving: false, found };
  }

  /** End of video: close any open burst of movement and decide everything waiting. */
  finish(t: number): Found[] {
    const found: Found[] = [];
    if (this.segment) {
      this.closeSegment(this.segment);
      this.segment = null;
    }
    found.push(...this.decidePending(Number.POSITIVE_INFINITY));
    void t;
    return found;
  }

  /* --------------------------------------------------- movement bursts */

  private trackMovement(motion: ReturnType<typeof largestRegion>, t: number): Found[] {
    const found: Found[] = [];

    if (motion) {
      if (!this.segment) {
        this.segment = {
          start: t,
          end: t,
          points: [],
          peakArea: 0,
          bigSince: null,
          claimed: false,
          loiterRaised: false,
          aggressionRaised: false,
        };
      }
      const seg = this.segment;
      seg.end = t;
      seg.points.push({ t, box: motion.box, area: motion.area });
      seg.peakArea = Math.max(seg.peakArea, motion.area);

      // C2 — movement that has stayed in one spot for a long time.
      const duration = t - seg.start;
      if (!seg.loiterRaised && duration >= this.opts.loiterSeconds && spread(seg.points) < 0.3) {
        seg.loiterRaised = true;
        seg.claimed = true;
        const f = this.raise('loitering', seg.start, t, seg.points, [
          { name: 'dwell_time', value: round(duration, 1), unit: 'seconds' },
          { name: 'position_spread', value: round(spread(seg.points)), unit: null },
        ], 0.5 + Math.min(0.25, (duration - this.opts.loiterSeconds) / 40));
        if (f) found.push(f);
      }

      // C3 — a lot of fast movement, held for a moment.
      if (motion.area >= BIG_MOTION) {
        if (seg.bigSince === null) seg.bigSince = t;
        if (!seg.aggressionRaised && t - seg.bigSince >= BIG_MOTION_SECONDS) {
          seg.aggressionRaised = true;
          seg.claimed = true;
          const f = this.raise('aggression', seg.bigSince, t, seg.points.filter((p) => p.t >= (seg.bigSince ?? 0)), [
            { name: 'motion_area', value: round(seg.peakArea), unit: 'fraction' },
            { name: 'motion_duration', value: round(t - seg.bigSince, 1), unit: 'seconds' },
          ], 0.45 + Math.min(0.3, seg.peakArea));
          if (f) found.push(f);
        }
      } else {
        seg.bigSince = null;
      }
    } else if (this.segment && t - this.segment.end > SEGMENT_GAP) {
      this.closeSegment(this.segment);
      this.segment = null;
    }

    return found;
  }

  private closeSegment(seg: Segment): void {
    this.recent.push(seg);
    // Keep the last 30 s of history.
    this.recent = this.recent.filter((s) => seg.end - s.end < 30);
    if (seg.end - seg.start >= MIN_SEGMENT) {
      // Wait long enough for a fall or a left item to be confirmed first,
      // so the same moment is not reported twice under two names.
      this.pending.push({ segment: seg, decideAt: seg.end + STILL_SECONDS + 1 });
    }
  }

  private decidePending(t: number): Found[] {
    const found: Found[] = [];
    const remaining: typeof this.pending = [];
    for (const item of this.pending) {
      if (item.decideAt > t) {
        remaining.push(item);
        continue;
      }
      const seg = item.segment;
      if (seg.claimed) continue;
      const f = this.raise('zone_activity', seg.start, seg.end, seg.points, [
        { name: 'motion_duration', value: round(seg.end - seg.start, 1), unit: 'seconds' },
        { name: 'motion_area', value: round(seg.peakArea), unit: 'fraction' },
      ], 0.35 + Math.min(0.2, (seg.end - seg.start) / 30));
      if (f) found.push(f);
    }
    this.pending = remaining;
    return found;
  }

  /* ------------------------------------------------------ still things */

  private trackStillThings(
    frame: Uint8Array,
    near: Uint8Array,
    motion: ReturnType<typeof largestRegion>,
    t: number,
    frameAspect: number,
  ): Found[] {
    if (!this.background) return [];
    let changed = subtract(this.background.changedCells(frame), near);
    // Ghosts — spots something LEFT, not something new — are accepted as
    // background straight away.
    const ghosts = ghostCells(changed, near, frame);
    if (countCells(ghosts) > 0) {
      this.background.absorbCells(frame, ghosts);
      changed = subtract(changed, ghosts);
    }
    const region = largestRegion(changed, MIN_STATIC_CELLS);

    if (!region) {
      if (this.still && t - this.still.last > 1) this.still = null;
      return [];
    }

    if (this.still && overlap(this.still.box, region.box) > 0.3) {
      this.still.last = t;
      this.still.box = region.box;
      this.still.area = region.area;
    } else {
      this.still = { box: region.box, area: region.area, since: t, last: t, raised: false };
    }

    const still = this.still;
    const touchingMotion = motion ? overlap(motion.box, still.box) > 0.2 : false;
    if (still.raised || touchingMotion || t - still.since < STILL_SECONDS) return [];

    // Light, not a thing. Two cases, both measured on real CCTV:
    //   - glare: a sunlit window or reflection that got brighter still
    //   - featureless before and after: only the brightness moved
    const light = lightStats(this.background.pixels, frame, still.box);
    const glare = light.meanNow >= GLARE_LEVEL && light.meanNow > light.meanBefore + 10;
    const featureless = light.stdBefore < 7 && light.stdNow < 7;
    const note = { t, box: still.box, area: round(still.area, 4), aspect: round(pixelAspect(still.box, frameAspect)), ...light };
    if (glare || featureless) {
      debug({ ...note, decision: glare ? 'ignored: glare' : 'ignored: lighting' });
      this.background.absorbBox(frame, still.box);
      this.still = null;
      return [];
    }
    // Already reported something here; do not report the same item twice.
    if (this.reportedSpots.some((b) => overlap(b, still.box) > 0.4)) {
      debug({ ...note, decision: 'ignored: already reported' });
      still.raised = true;
      return [];
    }

    still.raised = true;
    const before = this.lastSegmentTouching(still.box);
    const aspect = pixelAspect(still.box, frameAspect);
    const stillFor = t - still.since;

    // C4 — went down, and a long flat shape has stayed there.
    if (before && aspect >= 1.3 && still.area >= 0.006 && this.droppedNear(before, still.box)) {
      before.claimed = true;
      debug({ ...note, decision: 'fall' });
      this.reportedSpots.push(still.box);
      const f = this.raise('fall', before.end, t, [{ t: before.end, box: still.box }, { t, box: still.box }], [
        { name: 'downward_motion', value: 0.7, unit: null },
        { name: 'posture_aspect_ratio', value: round(aspect), unit: null },
        { name: 'low_posture_duration', value: round(stillFor, 1), unit: 'seconds' },
        { name: 'post_fall_movement', value: 0, unit: null },
      ], 0.55 + Math.min(0.2, (aspect - 1.3) / 4));
      return f ? [f] : [];
    }

    // C4 — a small new thing stayed, AND whatever brought it moved away.
    // The second half matters: a person who walks in and stops also leaves a
    // still, new shape — but their movement ended where they stand. With a
    // left item, the movement carried on somewhere else.
    const lastMove = before ? before.points[before.points.length - 1] : null;
    const moverLeft = lastMove !== null && overlap(lastMove.box, still.box) < 0.1;
    const personShaped = aspect < 0.5; // tall and thin: someone standing still
    if (still.area <= 0.05 && moverLeft && !personShaped) {
      if (before) before.claimed = true;
      debug({ ...note, decision: 'abandoned_object' });
      this.reportedSpots.push(still.box);
      const f = this.raise('abandoned_object', still.since, t, [{ t: still.since, box: still.box }, { t, box: still.box }], [
        { name: 'object_dwell_time', value: round(stillFor, 1), unit: 'seconds' },
        { name: 'object_size', value: round(still.area, 3), unit: 'fraction' },
        { name: 'object_displacement', value: 0, unit: null },
      ], 0.55);
      return f ? [f] : [];
    }

    // A person who stopped, or a big change that stayed — someone sat down, a
    // door opened. Not worth an alert on its own.
    debug({
      ...note,
      decision: personShaped ? 'ignored: person standing still' : !moverLeft ? 'ignored: nobody moved away' : 'ignored: too big',
    });
    return [];
  }

  private lastSegmentTouching(box: BoundingBox): Segment | null {
    const candidates = [...this.recent, ...(this.segment ? [this.segment] : [])];
    for (let i = candidates.length - 1; i >= 0; i -= 1) {
      const seg = candidates[i];
      if (seg.points.some((p) => overlap(p.box, box) > 0.2)) return seg;
    }
    return null;
  }

  /** Did the movement near this box come downwards towards the end? */
  private droppedNear(seg: Segment, box: BoundingBox): boolean {
    const nearby = seg.points.filter((p) => overlap(p.box, box) > 0.2);
    if (nearby.length === 0) return false;
    const last = nearby[nearby.length - 1];
    const window = nearby.filter((p) => last.t - p.t <= 2.5);
    const topStart = Math.min(...window.map((p) => p.box.y_min));
    const topEnd = last.box.y_min;
    // The top of the moving shape came down, or the shape was already low and wide.
    return topEnd - topStart >= 0.06 || centre(last.box).y > 0.55;
  }

  /* ----------------------------------------------------------- events */

  private raise(
    kindKey: keyof typeof KINDS,
    start: number,
    end: number,
    path: PathPoint[],
    evidence: EvidenceSignal[],
    confidence: number,
  ): Found | null {
    const kind = KINDS[kindKey];
    if (this.opts.cooldownSeconds > 0) {
      const last = this.lastRaised[kind.type];
      if (last !== undefined && end - last < this.opts.cooldownSeconds) return null;
    }
    this.lastRaised[kind.type] = end;

    this.counters[kind.track] += 1;
    const keyPoint = path.length > 0 ? path[path.length - 1] : null;
    const offset = this.opts.mode === 'upload' ? round(Math.max(0, start), 2) : null;

    const event: IncidentEvent = {
      event_id: `practice-${kind.type}-${shortId()}`,
      component_id: kind.component,
      incident_type: kind.type,
      camera_id: this.opts.sourceId,
      timestamp_detected: iso(),
      priority: kind.priority,
      status: 'new',
      summary: kind.summary,
      evidence,
      model_name: PRACTICE_MODEL,
      model_version: PRACTICE_VERSION,
      confidence: round(Math.min(0.79, confidence)),
      score_calibrated: false,
      bounding_box: keyPoint ? keyPoint.box : null,
      track_id: `${kind.track}-${this.counters[kind.track]}`,
      related_tracks: null,
      media_ref: this.opts.mediaRef,
      media_offset_seconds: offset,
      zone_id: null,
    };

    return {
      event,
      startSeconds: Math.max(0, start),
      endSeconds: Math.max(start, end),
      path: path.map((p) => ({ t: p.t, box: p.box })),
    };
  }
}
