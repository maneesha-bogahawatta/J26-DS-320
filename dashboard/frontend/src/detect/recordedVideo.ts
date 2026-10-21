/** Run the original team's video check and the existing C3 temporal job together. */
import type { IncidentEvent } from '../types/events';
import type { C3Job } from '../api/c3Client';
import type { AnalyzeOptions, Found, VideoCheckReport } from './engine';
import { Cancelled } from './engine.ts';

export interface TeamVideoResult { events: IncidentEvent[]; detectors_run?: string[] }
export interface RecordedDependencies {
  team: () => Promise<TeamVideoResult>;
  c3: <T>(path: string, options?: RequestInit) => Promise<T>;
  wait?: (signal?: AbortSignal) => Promise<void>;
  maxPolls?: number;
}

function abortIfNeeded(signal?: AbortSignal) { if (signal?.aborted) throw new Cancelled(); }
function message(error: unknown) { return error instanceof Error ? error.message : String(error); }

export function emptyVideoSummary(report: VideoCheckReport | null, c3Expected: boolean): string {
  if ((c3Expected && !report?.c3) || report?.c3?.status === 'failed' || report?.team.status === 'failed') return 'Video check incomplete';
  const timing = report?.c3?.job?.prediction?.timing;
  if (timing && timing.valid_decisions === 0) return 'Insufficient pose evidence for C3';
  if (report && !report.c3 && report.team.detectors.length === 0) return 'No models checked this video';
  return 'No candidates returned';
}

function waitForPoll(signal?: AbortSignal): Promise<void> {
  abortIfNeeded(signal);
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new Cancelled()); };
    const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, 1000);
    signal?.addEventListener('abort', abort, { once: true });
  });
}

/** Keep original-source timeline and rendered-media geometry separate. */
export function c3Found(event: IncidentEvent, contextStart: number): Found {
  const sourceTime = event.evidence.find(item => item.name === 'source_timestamp_seconds')?.value;
  const start = typeof sourceTime === 'number' && Number.isFinite(sourceTime)
    ? sourceTime : contextStart + (event.media_offset_seconds ?? 0);
  // The shared event keeps its annotated video and its footer-normalized box.
  // That box must never be drawn on the unmodified upload in this screen.
  return { event, startSeconds: start, endSeconds: start + .5, path: [] };
}

export async function checkRecordedVideo(file: File, options: AnalyzeOptions,
  dependencies: RecordedDependencies): Promise<{ teamEvents: IncidentEvent[]; c3Found: Found[]; report: VideoCheckReport }> {
  const { signal, sourceId, onProgress } = options;
  abortIfNeeded(signal);
  const teamWork = async (): Promise<VideoCheckReport['team'] & { events: IncidentEvent[] }> => {
    try {
      const result = await dependencies.team();
      return { status: 'completed', detectors: result.detectors_run ?? [], events: result.events ?? [] };
    } catch (error) {
      abortIfNeeded(signal);
      return { status: 'failed', detectors: [], events: [], error: message(error) };
    }
  };
  const c3Work = async (): Promise<NonNullable<VideoCheckReport['c3']>> => {
    try {
      onProgress?.({ fraction: null, stage: 'Uploading video for C3 temporal inference and registered team models' });
      const source = await dependencies.c3<{ source_id: string }>('/api/v1/c3/uploads', {
        method: 'POST', body: file, signal,
        headers: { 'Content-Type': 'application/octet-stream', 'X-C3-File-Name': encodeURIComponent(file.name) },
      });
      abortIfNeeded(signal);
      let job = await dependencies.c3<C3Job>('/api/v1/c3/jobs', {
        method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: source.source_id, camera_id: sourceId, seconds: 60, research_alerts: false }),
      });
      for (let polls = 0; job.status === 'queued' || job.status === 'running'; polls++) {
        if (polls >= (dependencies.maxPolls ?? 3600)) throw new Error(`C3 job ${job.job_id} is still pending on the server. This video check has not finished.`);
        onProgress?.({ fraction: null, stage: job.status === 'queued'
          ? 'C3 queued; waiting for the temporal inference worker'
          : 'C3 running pose tracking and temporal violence research scores' });
        await (dependencies.wait ?? waitForPoll)(signal);
        abortIfNeeded(signal);
        job = await dependencies.c3<C3Job>(`/api/v1/c3/jobs/${encodeURIComponent(job.job_id)}`, { signal });
      }
      if (job.status !== 'completed') return { status: 'failed', job, error: job.error ?? `C3 job ${job.status}` };
      if (!job.prediction) throw new Error('C3 completed without prediction provenance; the check is incomplete.');
      return { status: 'completed', job };
    } catch (error) {
      abortIfNeeded(signal);
      return { status: 'failed', error: message(error) };
    }
  };
  const [team, c3] = await Promise.all([teamWork(), c3Work()]);
  abortIfNeeded(signal);
  const report: VideoCheckReport = { team: { status: team.status, detectors: team.detectors, error: team.error }, c3 };
  options.onReport?.(report);
  if (team.status === 'failed' && c3.status === 'failed') throw new Error(`Video checks failed. C3: ${c3.error}. Team models: ${team.error}`);
  return { teamEvents: team.events, c3Found: (c3.status === 'completed' ? c3.job?.events ?? [] : []).map(event => c3Found(event, c3.job?.prediction?.source_context[0] ?? 0)), report };
}
