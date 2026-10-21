/**
 * MOCK implementation of EventApi.
 *
 * Holds events in memory, applies operator actions, and simulates a little
 * network latency so loading states are actually exercised during development.
 * No backend required — `npm run dev` works on a laptop with no Python set up.
 */

import type { ActionRequest, EventApi } from './client';
import { ApiError } from './client';
import type {
  AuditEntry,
  CameraSource,
  ComponentId,
  HealthStatus,
  IncidentEvent,
} from '../types/events';
import { ACTION_TARGET_STATUS } from '../types/events';
import { SEED_CAMERAS, buildSeedEvents } from './seedData';

const LATENCY_MS = 220;

function delay<T>(value: T, ms = LATENCY_MS): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

export function createMockClient(): EventApi {
  let events: IncidentEvent[] = buildSeedEvents();
  const cameras: CameraSource[] = [...SEED_CAMERAS];

  return {
    mode: 'mock',

    async listEvents() {
      return delay(events.map((e) => ({ ...e })));
    },

    async listCameras() {
      return delay(cameras.map((c) => ({ ...c })));
    },

    async health() {
      const reporting = Array.from(new Set(events.map((e) => e.component_id))) as ComponentId[];
      const status: HealthStatus = {
        ok: true,
        mode: 'mock',
        api_version: '1.1.0-mock',
        components_reporting: reporting,
        checked_at: nowIso(),
      };
      return delay(status, 80);
    },

    async applyAction({ eventId, action, operatorId, note }: ActionRequest) {
      const index = events.findIndex((e) => e.event_id === eventId);
      if (index === -1) {
        throw new ApiError(`No event with id "${eventId}".`, 404);
      }

      const current = events[index];
      const toStatus = ACTION_TARGET_STATUS[action];

      if (current.status === toStatus) {
        throw new ApiError(`Event is already marked "${toStatus}".`, 409);
      }

      const entry: AuditEntry = {
        action,
        at: nowIso(),
        operator_id: operatorId || null,
        note: note && note.trim() ? note.trim() : null,
        from_status: current.status,
        to_status: toStatus,
      };

      const updated: IncidentEvent = {
        ...current,
        status: toStatus,
        audit: [...(current.audit ?? []), entry],
      };

      events = events.map((e, i) => (i === index ? updated : e));
      return delay({ ...updated }, 160);
    },

    async submitEvent(event: IncidentEvent) {
      if (events.some((e) => e.event_id === event.event_id)) {
        throw new ApiError(
          `Duplicate event_id "${event.event_id}". Each alert needs a unique id.`,
          409,
        );
      }
      const stored: IncidentEvent = {
        ...event,
        audit: [
          {
            action: 'created',
            at: nowIso(),
            operator_id: null,
            note:
              event.model_name === 'practice-motion-check'
                ? 'Found by the practice detector on the Check screen.'
                : 'Submitted through the Project team tab.',
            from_status: null,
            to_status: event.status,
          },
          ...(event.audit ?? []),
        ],
      };
      events = [stored, ...events];
      return delay({ ...stored }, 120);
    },
  };
}
