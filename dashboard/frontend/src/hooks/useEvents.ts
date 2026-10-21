/**
 * Loads events, cameras and health; records operator actions; polls for new
 * alerts.
 *
 * Polling, not WebSockets: a periodic refresh is enough for a prototype and
 * costs a few lines instead of a streaming stack. Swap in a socket later if
 * the group decides it is needed.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { ApiError } from '../api/client';
import type { CameraSource, HealthStatus, IncidentEvent, OperatorAction } from '../types/events';

const POLL_INTERVAL_MS = 15_000;

export interface EventsState {
  events: IncidentEvent[];
  cameras: CameraSource[];
  health: HealthStatus | null;
  loading: boolean;
  error: string | null;
  lastSyncedAt: string | null;
  refresh: () => Promise<void>;
  recordAction: (
    eventId: string,
    action: OperatorAction,
    operatorId: string,
    note: string | null,
  ) => Promise<void>;
  addEvent: (event: IncidentEvent) => Promise<IncidentEvent>;
  actionPending: string | null;
}

export function useEvents(): EventsState {
  const [events, setEvents] = useState<IncidentEvent[]>([]);
  const [cameras, setCameras] = useState<CameraSource[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null);
  const [actionPending, setActionPending] = useState<string | null>(null);

  // Avoids a state update after unmount when the poll resolves late.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const load = useCallback(async (showSpinner: boolean) => {
    if (showSpinner) setLoading(true);
    try {
      const [nextEvents, nextCameras, nextHealth] = await Promise.all([
        api.listEvents(),
        api.listCameras(),
        api.health(),
      ]);
      if (!mounted.current) return;
      setEvents(nextEvents);
      setCameras(nextCameras);
      setHealth(nextHealth);
      setError(null);
      setLastSyncedAt(new Date().toISOString());
    } catch (err) {
      if (!mounted.current) return;
      setError(err instanceof ApiError ? err.message : 'Unexpected error while loading alerts.');
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(true);
    const id = window.setInterval(() => void load(false), POLL_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [load]);

  const refresh = useCallback(async () => {
    await load(false);
  }, [load]);

  const recordAction = useCallback(
    async (eventId: string, action: OperatorAction, operatorId: string, note: string | null) => {
      setActionPending(eventId);
      try {
        const updated = await api.applyAction({ eventId, action, operatorId, note });
        if (!mounted.current) return;
        setEvents((prev) => prev.map((e) => (e.event_id === updated.event_id ? updated : e)));
        setError(null);
      } catch (err) {
        if (!mounted.current) return;
        setError(err instanceof ApiError ? err.message : 'Could not record that action.');
      } finally {
        if (mounted.current) setActionPending(null);
      }
    },
    [],
  );

  const addEvent = useCallback(async (event: IncidentEvent) => {
    const stored = await api.submitEvent(event);
    if (mounted.current) {
      setEvents((prev) => [stored, ...prev]);
    }
    return stored;
  }, []);

  return {
    events,
    cameras,
    health,
    loading,
    error,
    lastSyncedAt,
    refresh,
    recordAction,
    addEvent,
    actionPending,
  };
}
