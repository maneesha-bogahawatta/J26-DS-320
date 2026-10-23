/**
 * Sends videos and live frames to the backend, which runs the detectors the
 * team has registered in dashboard/backend/detect.py and replies with events in
 * the shared contract.
 *
 *   POST /api/v1/detect/video?source_id=…&filename=…   body: the video file
 *   POST /api/v1/detect/frame?source_id=…&t=…          body: one JPEG
 *   -> { "events": IncidentEvent[], "detectors_run": string[] }
 */

import type { IncidentEvent } from '../types/events';
import type { AnalyzeOptions, DetectionEngine, Found, LiveSession } from './engine';
import { Cancelled } from './engine';
import { c3Request } from '../api/c3Client';
import { checkRecordedVideo } from './recordedVideo';

interface DetectResponse {
  events: IncidentEvent[];
  detectors_run?: string[];
}

/** C4 needs ~5 frames per second: its object timer and fall window assume that rate (1/s is too sparse). */
const FRAME_INTERVAL_S = 0.2;
const FRAME_WIDTH = 640;

function toFound(event: IncidentEvent, at: number, mediaUrl: string | null): Found {
  const start = typeof event.media_offset_seconds === 'number' ? event.media_offset_seconds : at;
  return {
    // The server's media_ref points at its own copy. Play the local one.
    event: mediaUrl ? { ...event, media_ref: mediaUrl } : event,
    startSeconds: start,
    endSeconds: start + 3,
    path: event.bounding_box ? [{ t: start, box: event.bounding_box }] : [],
  };
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string') return body.detail;
  } catch {
    /* not JSON */
  }
  return `The detection server answered ${res.status}.`;
}

export function createHttpEngine(baseUrl: string, { c3Enabled = false } = {}): DetectionEngine {
  const root = baseUrl.replace(/\/+$/, '');

  return {
    mode: 'http',
    description: 'Videos and frames are sent to the detection server, which runs the team’s registered models.',

    async analyzeVideo(file: File, options: AnalyzeOptions): Promise<Found[]> {
      const { sourceId, mediaUrl, onProgress, onReport, signal } = options;
      onProgress?.({ fraction: null, stage: 'Sending the video to the detection server' });
      const shortName = file.name.length > 120 ? file.name.slice(0, 100) + file.name.slice(file.name.lastIndexOf('.')) : file.name;
      const url = `${root}/api/v1/detect/video?source_id=${encodeURIComponent(sourceId)}&filename=${encodeURIComponent(shortName)}`;
      const teamCheck = async (): Promise<DetectResponse> => {
        let res: Response;
        try {
          res = await fetch(url, {
            method: 'POST',
            body: file,
            headers: { 'Content-Type': file.type || 'application/octet-stream' },
            signal,
          });
        } catch (err) {
          if (signal?.aborted) throw new Cancelled();
          throw new Error(`Cannot reach the detection server at ${root}. Is it running? (${String(err)})`);
        }
        if (!res.ok) throw new Error(await readError(res));
        return await res.json() as DetectResponse;
      };
      if (c3Enabled) {
        const result = await checkRecordedVideo(file, options, { team: teamCheck, c3: c3Request });
        onProgress?.({ fraction: 1, stage: 'Video checks finished; inspect model coverage' });
        return [...result.teamEvents.map(event => toFound(event, 0, mediaUrl)), ...result.c3Found]
          .sort((a, b) => a.startSeconds - b.startSeconds);
      }
      const body = await teamCheck();
      onReport?.({ team: { status: 'completed', detectors: body.detectors_run ?? [] } });
      onProgress?.({ fraction: 1, stage: 'Done' });
      return (body.events ?? []).map((e) => toFound(e, 0, mediaUrl)).sort((a, b) => a.startSeconds - b.startSeconds);
    },

    startLive({ sourceId }): LiveSession {
      const canvas = document.createElement('canvas');
      let lastSent = -Infinity;
      let inFlight = false;
      const controller = new AbortController();

      return {
        async check(video, elapsedSeconds) {
          const idle = { moving: null, cameraMoving: false, found: [] as Found[] };
          if (inFlight || elapsedSeconds - lastSent < FRAME_INTERVAL_S) return idle;
          if (video.readyState < 2 || !video.videoWidth) return idle;

          lastSent = elapsedSeconds;
          inFlight = true;
          try {
            canvas.width = FRAME_WIDTH;
            canvas.height = Math.round((FRAME_WIDTH * video.videoHeight) / video.videoWidth);
            canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height);
            const jpeg = await new Promise<Blob | null>((r) => canvas.toBlob(r, 'image/jpeg', 0.8));
            if (!jpeg) return idle;

            const res = await fetch(
              `${root}/api/v1/detect/frame?source_id=${encodeURIComponent(sourceId)}&t=${elapsedSeconds.toFixed(2)}`,
              { method: 'POST', body: jpeg, headers: { 'Content-Type': 'image/jpeg' }, signal: controller.signal },
            );
            if (!res.ok) throw new Error(await readError(res));
            const body = (await res.json()) as DetectResponse;
            return { ...idle, found: (body.events ?? []).map((e) => toFound(e, elapsedSeconds, null)) };
          } finally {
            inFlight = false;
          }
        },
        stop() {
          controller.abort();
        },
      };
    },
  };
}
