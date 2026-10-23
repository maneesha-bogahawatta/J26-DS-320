/**
 * Chooses the data layer from environment config, once, at module load.
 *
 * `.env` (copy from `.env.example`):
 *   VITE_API_MODE=mock | http
 *   VITE_API_BASE_URL=http://127.0.0.1:8000
 *   VITE_MEDIA_BASE_URL=/            (where clips are served from)
 */

import type { EventApi } from './client';
import { createMockClient } from './mockClient';
import { createHttpClient } from './httpClient';

const mode = (import.meta.env.VITE_API_MODE ?? 'mock').toLowerCase();
const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export const api: EventApi = mode === 'http' ? createHttpClient(baseUrl) : createMockClient();

export const API_BASE_URL = baseUrl;

/** Where clip files are served from. Defaults to the Vite `public/` root. */
export const MEDIA_BASE_URL = (import.meta.env.VITE_MEDIA_BASE_URL ?? import.meta.env.BASE_URL ?? '/').replace(
  /\/*$/,
  '/',
);

/** Resolve an event's `media_ref` into a URL the <video> element can load. */
export function resolveMediaUrl(mediaRef: string | null | undefined): string | null {
  if (!mediaRef) return null;
  // Full URLs, and local files the officer just uploaded (blob:), play as-is.
  if (/^(https?:|blob:)/i.test(mediaRef)) return mediaRef;
  return `${MEDIA_BASE_URL}${mediaRef.replace(/^\/+/, '')}`;
}

export type { EventApi } from './client';
export { ApiError } from './client';

/** Additive C3 workflow, enabled only in the explicit C3 configuration. */
export const C3_ENABLED = import.meta.env.VITE_C3_ENABLED === 'true';
