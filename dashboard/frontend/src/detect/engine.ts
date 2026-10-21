/**
 * The detection seam.
 *
 * The dashboard asks for detections through ONE interface, `DetectionEngine`.
 * Two implementations exist:
 *
 *   browser : the practice detector. Runs in the page, needs no server, and
 *             measures real movement — so its boxes land on what actually
 *             moved — but it does not know WHAT moved. It exists so the
 *             upload and live screens can be built and demonstrated before
 *             the four trained models are ready.
 *
 *   http    : sends the video, or live frames, to the backend, which runs
 *             whatever detectors the team has registered there
 *             (backend_mock/detect.py) and returns events in the shared
 *             contract.
 *
 * Switch with VITE_DETECT_MODE in .env. Either way the screens receive the
 * same `IncidentEvent` objects as the alert list, so a teammate's model goes
 * live on every screen at once with no UI change.
 */

import type { BoundingBox, IncidentEvent } from '../types/events';

/** Where a detection's box was at one moment, for following it during playback. */
export interface PathPoint {
  t: number;
  box: BoundingBox;
}

/** One thing found, plus what the screens need to show it on the video. */
export interface Found {
  event: IncidentEvent;
  /** Seconds into the video (upload) or since the camera started (live). */
  startSeconds: number;
  endSeconds: number;
  /** The box over time. May be a single point. */
  path: PathPoint[];
  /** Small still of the moment, as a data URL. Live only. */
  snapshot?: string | null;
}

export interface AnalyzeProgress {
  /** 0..1, or null when the work has no measurable progress (waiting on a server). */
  fraction: number | null;
  /** Plain words for the officer, e.g. "Watching the video". */
  stage: string;
}

export interface AnalyzeOptions {
  /** Camera/source id written into every event, e.g. "upload-lobby". */
  sourceId: string;
  /** Local URL the video is playing from. Written into events as media_ref. */
  mediaUrl: string;
  onProgress?: (p: AnalyzeProgress) => void;
  signal?: AbortSignal;
}

export interface LiveTick {
  /** What is moving right now, for the live outline. Null when nothing is. */
  moving: BoundingBox | null;
  /** True when the whole picture changed — the camera itself was moved. */
  cameraMoving: boolean;
  /** Anything that crossed the line on this tick. Usually empty. */
  found: Found[];
}

export interface LiveSession {
  /** Look at the current frame. Call a few times a second. */
  check(video: HTMLVideoElement, elapsedSeconds: number): Promise<LiveTick>;
  stop(): void;
}

export interface DetectionEngine {
  readonly mode: 'browser' | 'http';
  /** Shown on screen so nobody mistakes the practice detector for a trained model. */
  readonly description: string;
  analyzeVideo(file: File, opts: AnalyzeOptions): Promise<Found[]>;
  startLive(opts: { sourceId: string }): LiveSession;
}

/** Thrown when the officer cancels a check. Not an error worth showing. */
export class Cancelled extends Error {
  constructor() {
    super('Cancelled');
    this.name = 'Cancelled';
  }
}
