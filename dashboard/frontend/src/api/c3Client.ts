/** Real C3 jobs. No practice-data fallback is permitted on this path. */
import { API_BASE_URL } from './index';
import type { IncidentEvent } from '../types/events';

export interface C3Source {
  source_id: string;
  name: string;
  kind: 'cached_context' | 'uploaded_video';
  split?: string;
}

export interface C3Score {
  t_s: number | string;
  weak_onset_score: number | string | null;
  fight_score: number | string | null;
  valid_pose?: boolean | string;
  status?: string;
}

export interface C3Prediction {
  camera_id?: string;
  source_context: [number, number];
  timing: { valid_decisions: number; decisions: number };
  pose_cache_reused?: boolean;
  extraction_seconds?: number;
  model_version?: string;
  variant?: string;
  score_semantics?: string;
  research_notifications?: number;
  operational_alerts_enabled?: boolean;
  thresholds?: [number, number];
  research_alert_heads_enabled?: [boolean, boolean];
}

export interface C3Job {
  job_id: string;
  source_id: string;
  source_name: string;
  camera_id?: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';
  created_at: string;
  error?: string;
  prediction?: C3Prediction;
  scores?: C3Score[];
  events?: IncidentEvent[];
  files?: string[];
}

export interface C3Bootstrap {
  token: string;
  sources: C3Source[];
  jobs: C3Job[];
  max_upload_bytes?: number;
  capabilities?: { video: boolean; cached_sources: boolean; live_frame: boolean; research_only: boolean };
}

let token: string | null = null;
let tokenRequest: Promise<string> | null = null;

export function invalidateC3Token(): void { token = null; }

export function c3Url(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json() as { detail?: unknown; error?: unknown };
    const value = body.detail ?? body.error;
    if (typeof value === 'string') return value;
    if (value) return JSON.stringify(value);
  } catch { /* retain the HTTP status */ }
  return `The C3 service answered ${response.status}.`;
}

export async function c3Bootstrap(signal?: AbortSignal): Promise<C3Bootstrap> {
  const response = await fetch(c3Url('/api/v1/c3/bootstrap'), { signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  const body = await response.json() as C3Bootstrap;
  token = body.token;
  return body;
}

export async function c3MutationHeaders(): Promise<Record<string, string>> {
  if (!token) {
    tokenRequest ??= c3Bootstrap().then(body => body.token).finally(() => { tokenRequest = null; });
    token = await tokenRequest;
  }
  return { 'X-C3-Token': token };
}

export async function c3Request<T>(path: string, options: RequestInit = {}, retry = true): Promise<T> {
  const mutation = options.method && options.method !== 'GET';
  const headers = new Headers(options.headers);
  if (mutation) {
    for (const [key, value] of Object.entries(await c3MutationHeaders())) headers.set(key, value);
  }
  const response = await fetch(c3Url(path), { ...options, headers });
  if (mutation && response.status === 403 && retry) {
    token = null;
    return c3Request<T>(path, options, false);
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  return await response.json() as T;
}

export function c3MediaUrl(jobId: string, file = 'demo.mp4'): string {
  return c3Url(`/api/v1/c3/media/${encodeURIComponent(jobId)}/${file.split('/').map(encodeURIComponent).join('/')}`);
}

export function c3ExportUrl(jobId: string): string {
  return c3Url(`/api/v1/c3/jobs/${encodeURIComponent(jobId)}/export.zip`);
}
