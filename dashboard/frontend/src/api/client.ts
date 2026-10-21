/**
 * The dashboard's data layer.
 *
 * There is exactly ONE place the UI talks to data: the `EventApi` interface
 * below. Two implementations satisfy it —
 *
 *   mockClient : in-memory seeded events, no backend needed
 *   httpClient : real REST calls to backend_mock/ or the group's final backend
 *
 * Switch with `VITE_API_MODE` in `.env`:
 *   VITE_API_MODE=mock   (default — works offline, good for UI work)
 *   VITE_API_MODE=http   (talks to VITE_API_BASE_URL)
 *
 * This is the seam that lets Member 4 build the whole UI before any pipeline
 * exists, then swap in real detectors with no UI rework.
 */

import type {
  CameraSource,
  HealthStatus,
  IncidentEvent,
  OperatorAction,
} from '../types/events';

export interface ActionRequest {
  eventId: string;
  action: OperatorAction;
  operatorId: string;
  note: string | null;
}

export interface EventApi {
  /** Mode label shown in the top bar so nobody mistakes seeded data for live data. */
  readonly mode: 'mock' | 'http';
  listEvents(): Promise<IncidentEvent[]>;
  listCameras(): Promise<CameraSource[]>;
  health(): Promise<HealthStatus>;
  /** Record an operator decision and return the updated event. */
  applyAction(req: ActionRequest): Promise<IncidentEvent>;
  /** Used by the Integration tab to inject a pipeline's event by hand. */
  submitEvent(event: IncidentEvent): Promise<IncidentEvent>;
}

export class ApiError extends Error {
  /** HTTP status when the failure came from the server; undefined for network errors. */
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}
