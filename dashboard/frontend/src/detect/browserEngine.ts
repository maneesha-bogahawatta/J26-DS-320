/**
 * Runs the practice detector inside the page. No server, no upload — the
 * video never leaves the computer or phone it was chosen on.
 */

import type { AnalyzeOptions, DetectionEngine, Found, LiveSession } from './engine';
import { Cancelled } from './engine';
import { PracticeAnalyzer } from './practiceAnalyzer';

/** At most this many frames are examined per video, however long it is. */
const MAX_SAMPLES = 480;
/** And never more often than this. */
const MIN_STEP = 0.25;

function waitFor(el: HTMLMediaElement, event: string, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      resolve(); // carry on with whatever frame is showing rather than hang
    }, timeoutMs);
    const onEvent = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error('unplayable'));
    };
    function cleanup() {
      window.clearTimeout(timer);
      el.removeEventListener(event, onEvent);
      el.removeEventListener('error', onError);
    }
    el.addEventListener(event, onEvent, { once: true });
    el.addEventListener('error', onError, { once: true });
  });
}

/**
 * Some recordings (phone screen-recorders, MediaRecorder output) report an
 * infinite duration until the browser has seen the end of the file. Seeking
 * far past the end forces it to work the real length out.
 */
async function realDuration(video: HTMLVideoElement): Promise<number> {
  if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
  video.currentTime = 1e7;
  await waitFor(video, 'durationchange', 3000);
  const d = video.duration;
  video.currentTime = 0;
  await waitFor(video, 'seeked', 2000);
  return Number.isFinite(d) && d > 0 ? d : 0;
}

export function createBrowserEngine(): DetectionEngine {
  return {
    mode: 'browser',
    description:
      'Practice detector. It spots movement, drops, things left behind and people staying put — but it is a simple rule of thumb, not your team’s trained models.',

    async analyzeVideo(_file: File, { sourceId, mediaUrl, onProgress, signal }: AnalyzeOptions): Promise<Found[]> {
      const video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.preload = 'auto';
      video.src = mediaUrl;

      try {
        onProgress?.({ fraction: 0, stage: 'Opening the video' });
        try {
          await waitFor(video, 'loadeddata', 15000);
        } catch {
          throw new Error(
            'This browser cannot play this video’s format. Try opening the dashboard in Chrome, Edge or Safari, or convert the video to MP4 or WebM first.',
          );
        }

        const duration = await realDuration(video);
        if (!duration) throw new Error('Could not tell how long this video is, so it cannot be checked.');

        const step = Math.max(MIN_STEP, duration / MAX_SAMPLES);
        const aspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 16 / 9;
        const analyzer = new PracticeAnalyzer({ mode: 'upload', sourceId, mediaRef: mediaUrl });
        const found: Found[] = [];

        for (let t = 0; t <= duration; t += step) {
          if (signal?.aborted) throw new Cancelled();
          video.currentTime = Math.min(t, Math.max(0, duration - 0.04));
          await waitFor(video, 'seeked', 3000);
          found.push(...analyzer.push(video, t, aspect).found);
          onProgress?.({ fraction: Math.min(1, t / duration), stage: 'Watching the video' });
        }

        found.push(...analyzer.finish(duration));
        onProgress?.({ fraction: 1, stage: 'Done' });
        return found.sort((a, b) => a.startSeconds - b.startSeconds);
      } finally {
        video.removeAttribute('src');
        video.load();
      }
    },

    startLive({ sourceId }): LiveSession {
      const analyzer = new PracticeAnalyzer({ mode: 'live', sourceId });
      return {
        async check(video, elapsedSeconds) {
          if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
            return { moving: null, cameraMoving: false, found: [] };
          }
          const result = analyzer.push(video, elapsedSeconds, video.videoWidth / video.videoHeight);
          return { moving: result.moving, cameraMoving: result.cameraMoving, found: result.found };
        },
        stop() {
          /* nothing to release — the camera belongs to the screen, not the detector */
        },
      };
    },
  };
}
