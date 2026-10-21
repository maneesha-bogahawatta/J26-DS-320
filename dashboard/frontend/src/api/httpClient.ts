/**
 * HTTP implementation of EventApi — talks to the real REST API.
 *
 * Endpoints (agreed group contract v1):
 *   GET    /api/v1/health
 *   GET    /api/v1/cameras
 *   GET    /api/v1/events
 *   GET    /api/v1/events/{event_id}
 *   POST   /api/v1/events            <- detection pipelines submit alerts here
 *   PATCH  /api/v1/events/{event_id} <- operator records a decision
 *
 * Point it at a backend with VITE_API_BASE_URL (default http://127.0.0.1:8000).
 * `backend_mock/` in this repo implements exactly these routes.
 */

import type { ActionRequest, EventApi } from './client';
import { ApiError } from './client';
import type { CameraSource, HealthStatus, IncidentEvent } from '../types/events';
import { ACTION_TARGET_STATUS } from '../types/events';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    });
  } catch {
    // A network-level failure. The most common cause by far is that the
    // backend simply is not running, so say that instead of "Failed to fetch".
    throw new ApiError(
      `Could not reach the API at ${url}. Is the backend running? (cd backend_mock && uvicorn main:app --reload)`,
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* response had no JSON body — keep the status text */
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function createHttpClient(baseUrl: string): EventApi {
  const root = baseUrl.replace(/\/+$/, '');
  const api = `${root}/api/v1`;

  return {
    mode: 'http',

    listEvents() {
      return request<IncidentEvent[]>(`${api}/events`);
    },

    listCameras() {
      return request<CameraSource[]>(`${api}/cameras`);
    },

    health() {
      return request<HealthStatus>(`${api}/health`);
    },

    applyAction({ eventId, action, operatorId, note }: ActionRequest) {
      return request<IncidentEvent>(`${api}/events/${encodeURIComponent(eventId)}`, {
        method: 'PATCH',
        body: JSON.stringify({
          status: ACTION_TARGET_STATUS[action],
          action,
          operator_id: operatorId || null,
          note: note && note.trim() ? note.trim() : null,
        }),
      });
    },

    submitEvent(event: IncidentEvent) {
      return request<IncidentEvent>(`${api}/events`, {
        method: 'POST',
        body: JSON.stringify(event),
      });
    },
  };
}
