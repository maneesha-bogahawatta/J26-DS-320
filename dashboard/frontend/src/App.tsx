/**
 * Application shell.
 *
 * Four screens:
 *
 *   Check video   upload a recorded video, or watch a live camera (phone or
 *                 webcam), and see anything unusual it finds. The start screen.
 *   Alerts        the list, and one alert open at a time
 *   Shift log     what happened earlier, and the export for the study
 *   Project team  the integration surface for Members 1-3
 *
 * Everything found on the Check screen is added to Alerts as an ordinary
 * event in the shared contract, so the decision buttons, the shift log and
 * the CSV export all work on it with no special case.
 *
 * On a wide screen the list and the open alert sit side by side. On a narrow
 * one they are separate screens with a back button. Either way there are
 * never more than two things on screen at once — that is the whole layout
 * rule.
 *
 * All data comes through `useEvents`, which talks to the swappable API layer,
 * so nothing here knows whether the alerts are practice data or real ones.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import { useEvents } from './hooks/useEvents';
import type { CameraSource, IncidentEvent, OperatorAction } from './types/events';
import { detector } from './detect';
import { needsLook } from './lib/urgency';
import { AlertDetail } from './components/AlertDetail';
import { AlertList, applyListFilter } from './components/AlertList';
import type { ListFilter } from './components/AlertList';
import { AppHeader } from './components/AppHeader';
import type { Screen } from './components/AppHeader';
import { CheckScreen } from './components/detect/CheckScreen';
import type { CheckMode } from './components/detect/CheckScreen';
import { LIVE_SOURCE_ID } from './components/detect/LiveCheck';
import { EventHistory } from './components/EventHistory';
import { IntegrationPanel } from './components/IntegrationPanel';
import { EmptyState, ErrorState } from './components/States';

/** Below this, the list and the open alert are separate screens. */
const WIDE_QUERY = '(min-width: 1000px)';

function useIsWide(): boolean {
  const [wide, setWide] = useState(() =>
    typeof window !== 'undefined' ? window.matchMedia(WIDE_QUERY).matches : true,
  );

  useEffect(() => {
    const mq = window.matchMedia(WIDE_QUERY);
    const onChange = (e: MediaQueryListEvent) => setWide(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  return wide;
}

export default function App() {
  const { events, cameras, health, loading, error, refresh, recordAction, addEvent, actionPending } =
    useEvents();

  const wide = useIsWide();
  const [screen, setScreen] = useState<Screen>('check');
  const [checkMode, setCheckMode] = useState<CheckMode>('home');
  const [liveOn, setLiveOn] = useState(false);
  /** Uploaded files and the live camera, so alerts from them have a readable place name. */
  const [checkSources, setCheckSources] = useState<CameraSource[]>([]);
  const [filter, setFilter] = useState<ListFilter>('new');
  const [openId, setOpenId] = useState<string | null>(null);

  const allCameras = useMemo(() => [...cameras, ...checkSources], [cameras, checkSources]);

  const visible = useMemo(() => applyListFilter(events, filter), [events, filter]);

  const counts = useMemo(
    () => ({
      new: applyListFilter(events, 'new').length,
      handling: applyListFilter(events, 'handling').length,
      done: applyListFilter(events, 'done').length,
      all: events.length,
    }),
    [events],
  );

  const unreviewedCount = useMemo(() => events.filter(needsLook).length, [events]);

  /**
   * On a wide screen something is always open, so the right pane is never
   * blank. On a narrow one nothing is open until the officer taps a card.
   */
  const openEvent = useMemo(() => {
    if (openId) {
      const found = events.find((e) => e.event_id === openId);
      if (found) return found;
    }
    return wide ? (visible.length > 0 ? visible[0] : null) : null;
  }, [events, visible, openId, wide]);

  const handleAction = useCallback(
    (action: OperatorAction, operatorId: string, note: string | null) => {
      if (!openEvent) return;
      void recordAction(openEvent.event_id, action, operatorId, note);
    },
    [recordAction, openEvent],
  );

  const handleOpenFromLog = useCallback((eventId: string) => {
    setOpenId(eventId);
    setFilter('all');
    setScreen('alerts');
  }, []);

  /** Something was found on the Check screen: put it in Alerts. */
  const handleFound = useCallback(
    (found: IncidentEvent[], source: { id: string; label: string }) => {
      setCheckSources((prev) =>
        prev.some((c) => c.camera_id === source.id)
          ? prev
          : [
              ...prev,
              {
                camera_id: source.id,
                label: source.label,
                kind: source.id === LIVE_SOURCE_ID ? 'live_stream' : 'recorded_clip',
                media_ref: null,
                source_note: 'Added from the Check screen.',
              },
            ],
      );
      if (found.length === 0) return;
      if (detector.mode === 'http') {
        // The detection server has already stored them; just fetch.
        void refresh();
      } else {
        for (const event of found) void addEvent(event).catch(() => undefined);
      }
    },
    [addEvent, refresh],
  );

  const handleOpenAlert = useCallback((eventId: string) => {
    setOpenId(eventId);
    setFilter('all');
    setScreen('alerts');
  }, []);

  const handleSubmitEvent = useCallback(
    async (event: IncidentEvent) => {
      await addEvent(event);
    },
    [addEvent],
  );

  // On narrow screens the open alert IS the screen, so the list is hidden.
  const showList = wide || !openEvent;
  const showDetail = Boolean(openEvent) && (wide || Boolean(openId));

  return (
    <div className="app">
      <AppHeader
        mode={api.mode}
        health={health}
        unreviewedCount={unreviewedCount}
        screen={screen}
        onScreenChange={setScreen}
        onRefresh={() => void refresh()}
        refreshing={loading}
        liveOn={liveOn}
      />

      <main className="main">
        {error ? (
          <div className="banner" role="alert">
            Cannot reach the alert system right now. Keep watching your cameras as normal.
            <button type="button" className="banner__retry" onClick={() => void refresh()}>
              Try again
            </button>
          </div>
        ) : null}

        {/* Kept mounted (just hidden) so a camera keeps watching while the
            officer looks at Alerts. */}
        <div className="screen-check" hidden={screen !== 'check'}>
          <CheckScreen
            mode={checkMode}
            onModeChange={setCheckMode}
            onFound={handleFound}
            onOpenAlert={handleOpenAlert}
            onLiveChange={setLiveOn}
            needLook={unreviewedCount}
            onOpenAlerts={() => setScreen('alerts')}
          />
        </div>

        {screen === 'alerts' ? (
          <div className="screen" data-split={String(wide)}>
            {showList ? (
              error && events.length === 0 ? (
                <ErrorState
                  title="No alerts can be shown"
                  detail="The system that sends alerts cannot be reached. Tell the project team."
                  onRetry={() => void refresh()}
                />
              ) : (
                <AlertList
                  events={visible}
                  cameras={allCameras}
                  filter={filter}
                  onFilterChange={setFilter}
                  counts={counts}
                  selectedId={openEvent?.event_id ?? null}
                  onOpen={setOpenId}
                  loading={loading}
                />
              )
            ) : null}

            {showDetail && openEvent ? (
              <AlertDetail
                event={openEvent}
                cameras={allCameras}
                pending={actionPending === openEvent.event_id}
                onAction={handleAction}
                onBack={() => setOpenId(null)}
              />
            ) : wide ? (
              <EmptyState
                glyph="◻"
                title="Pick an alert"
                detail="Choose one from the list to see the video and what the camera saw."
              />
            ) : null}
          </div>
        ) : null}

        {screen === 'log' ? (
          <EventHistory events={events} cameras={allCameras} onOpenEvent={handleOpenFromLog} />
        ) : null}

        {screen === 'team' ? <IntegrationPanel onSubmit={handleSubmitEvent} apiMode={api.mode} /> : null}
      </main>
    </div>
  );
}
