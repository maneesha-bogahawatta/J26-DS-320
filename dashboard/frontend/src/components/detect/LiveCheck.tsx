/**
 * Use a live camera — a phone, a laptop webcam, or any camera the browser can
 * see.
 *
 * The camera feed is checked four times a second. A thin white outline shows
 * whatever is moving right now, so the officer can see the system is
 * watching; when something crosses the line, a coloured box and a card appear
 * and the item goes into Alerts.
 *
 * Nothing is recorded. Each found item keeps one small still so it can be
 * recognised in the list — that is all that is kept.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { BoundingBox, IncidentEvent } from '../../types/events';
import { detector } from '../../detect';
import type { Found } from '../../detect';
import type { LiveSession } from '../../detect/engine';
import { describeIncident } from '../../lib/incidentTypes';
import { plainClock } from '../../lib/plainLanguage';
import { urgencyOf } from '../../lib/urgency';
import { videoClock } from '../../lib/geometry';
import { AlertCard } from '../AlertCard';
import { DetectStage } from './DetectStage';
import type { StageBox } from './DetectStage';
import { FoundSummary } from './FoundSummary';

export const LIVE_SOURCE_ID = 'live-camera';

const TICK_MS = 250;
/** How long a found item's box stays on the live picture. */
const SHOW_FOUND_S = 4;
const SNAPSHOT_W = 192;

type Phase = 'idle' | 'starting' | 'live';
type Facing = 'environment' | 'user';

interface LiveCheckProps {
  onFound: (events: IncidentEvent[], source: { id: string; label: string }) => void;
  onOpenAlert: (eventId: string) => void;
  onLiveChange: (on: boolean) => void;
}

function cameraProblem(err: unknown): string {
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return 'Camera access was refused. Allow the camera in the browser (the icon in the address bar), then try again.';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.';
  if (name === 'NotReadableError')
    return 'The camera is being used by another app. Close that app (video call, camera app) and try again.';
  return `The camera could not be started. ${err instanceof Error ? err.message : ''}`.trim();
}

/**
 * True when the page is embedded somewhere whose permissions policy forbids
 * the camera (a preview frame, a shared page). Asking anyway would fail with a
 * misleading "access refused", so the screen explains instead.
 */
function cameraBlockedByHost(): boolean {
  if (typeof document === 'undefined') return false;
  const doc = document as Document & {
    permissionsPolicy?: { allowsFeature?: (f: string) => boolean };
    featurePolicy?: { allowsFeature?: (f: string) => boolean };
  };
  const policy = doc.permissionsPolicy ?? doc.featurePolicy;
  try {
    return policy?.allowsFeature ? !policy.allowsFeature('camera') : false;
  } catch {
    return false;
  }
}

function snapshot(video: HTMLVideoElement, canvas: HTMLCanvasElement): string | null {
  if (!video.videoWidth) return null;
  canvas.width = SNAPSHOT_W;
  canvas.height = Math.round((SNAPSHOT_W * video.videoHeight) / video.videoWidth);
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  try {
    return canvas.toDataURL('image/jpeg', 0.7);
  } catch {
    return null;
  }
}

export function LiveCheck({ onFound, onOpenAlert, onLiveChange }: LiveCheckProps) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [facing, setFacing] = useState<Facing>('environment');
  const [error, setError] = useState<string | null>(null);
  const [moving, setMoving] = useState<BoundingBox | null>(null);
  const [cameraMoving, setCameraMoving] = useState(false);
  const [found, setFound] = useState<Found[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const sessionRef = useRef<LiveSession | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const startedAt = useRef(0);
  const snapCanvas = useRef<HTMLCanvasElement | null>(null);

  const secure = typeof window !== 'undefined' && window.isSecureContext;
  const supported = typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia);
  const embeddedBlocked = cameraBlockedByHost();

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    sessionRef.current?.stop();
    sessionRef.current = null;
    setStream(null);
    setMoving(null);
    setCameraMoving(false);
    setPhase('idle');
    onLiveChange(false);
  }, [onLiveChange]);

  const start = useCallback(
    async (which: Facing) => {
      setError(null);
      setPhase('starting');
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: which }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        // If the camera is unplugged or permission is revoked, end cleanly.
        s.getVideoTracks()[0]?.addEventListener('ended', stop);
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = s;
        sessionRef.current?.stop();
        sessionRef.current = detector.startLive({ sourceId: LIVE_SOURCE_ID });
        startedAt.current = performance.now();
        setElapsed(0);
        setStream(s);
        setPhase('live');
        onLiveChange(true);
      } catch (err) {
        setError(cameraProblem(err));
        setPhase('idle');
      }
    },
    [stop, onLiveChange],
  );

  // Release the camera when the screen is closed.
  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      sessionRef.current?.stop();
    },
    [],
  );

  // The watching loop.
  useEffect(() => {
    if (phase !== 'live') return;
    let busy = false;
    const id = window.setInterval(async () => {
      const video = videoRef.current;
      const session = sessionRef.current;
      if (busy || !video || !session) return;
      busy = true;
      const t = (performance.now() - startedAt.current) / 1000;
      setElapsed(t);
      try {
        const tick = await session.check(video, t);
        setMoving(tick.moving);
        setCameraMoving(tick.cameraMoving);
        if (tick.found.length > 0) {
          if (!snapCanvas.current) snapCanvas.current = document.createElement('canvas');
          const still = snapshot(video, snapCanvas.current);
          const fresh = tick.found.map((f) => ({ ...f, snapshot: still }));
          setFound((prev) => [...fresh.reverse(), ...prev]);
          onFound(
            fresh.map((f) => f.event),
            { id: LIVE_SOURCE_ID, label: 'Live camera' },
          );
        }
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'The detector stopped responding.');
      } finally {
        busy = false;
      }
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [phase, onFound]);

  const boxes: StageBox[] = useMemo(() => {
    const out: StageBox[] = [];
    if (moving) out.push({ key: 'moving', box: moving, label: 'Moving', tone: 'moving' });
    for (const f of found) {
      if (elapsed - f.endSeconds > SHOW_FOUND_S) continue;
      const last = f.path[f.path.length - 1];
      if (!last) continue;
      out.push({
        key: f.event.event_id,
        box: last.box,
        label: describeIncident(f.event.incident_type, f.event.component_id).label,
        tone: urgencyOf(f.event).key,
      });
    }
    return out;
  }, [moving, found, elapsed]);

  const selected = found.find((f) => f.event.event_id === selectedId) ?? null;
  const whenText = (f: Found) =>
    `Seen at ${plainClock(f.event.timestamp_detected)} · ${videoClock(f.startSeconds)} into watching`;

  /* ---------------------------------------------------- before starting */

  if (phase !== 'live') {
    const blocked = !secure || !supported || embeddedBlocked;
    return (
      <div className="upload">
        <div className="drop drop--live">
          <span className="drop__icon" aria-hidden="true">
            ◉
          </span>
          <p className="drop__title">Watch a live camera</p>

          {blocked ? (
            <div className="live__blocked" role="alert">
              <p>
                <strong>The browser will not open the camera on this page.</strong>
              </p>
              <p>
                {!supported
                  ? 'This browser does not support cameras. Use a recent Chrome, Safari or Edge.'
                  : embeddedBlocked
                    ? 'This copy of the dashboard is shown inside another page, which does not allow cameras. Checking a recorded video works here; for the live camera, open the dashboard itself on your computer or phone.'
                    : 'Cameras only work on a secure (https) page or on this computer’s own address (localhost).'}
              </p>
              {supported && (!secure || embeddedBlocked) ? (
                <p className="live__setup">
                  For whoever set this up: start the dashboard with <code>npm run dev:phone</code> and open the{' '}
                  <code>https://</code> address it prints.
                </p>
              ) : null}
            </div>
          ) : (
            <>
              <div className="segmented" role="group" aria-label="Which camera">
                <button type="button" aria-pressed={facing === 'environment'} onClick={() => setFacing('environment')}>
                  Back camera
                </button>
                <button type="button" aria-pressed={facing === 'user'} onClick={() => setFacing('user')}>
                  Front camera
                </button>
              </div>
              <button
                type="button"
                className="btn btn--primary btn--big"
                onClick={() => void start(facing)}
                disabled={phase === 'starting'}
              >
                {phase === 'starting' ? 'Starting the camera…' : 'Start the camera'}
              </button>
              <p className="drop__hint">
                On a phone, open this page on the phone itself. Prop it up so it stays still — moving the camera
                pauses the checks.
              </p>
            </>
          )}

          {error ? (
            <p className="upload__error" role="alert">
              {error}
            </p>
          ) : null}
        </div>

        {found.length > 0 ? (
          <p className="live__earlier">
            {found.length} {found.length === 1 ? 'thing was' : 'things were'} found last time. They are in Alerts.
          </p>
        ) : null}
      </div>
    );
  }

  /* -------------------------------------------------------- while live */

  return (
    <div className="check" data-phase="live">
      <div className="check__main">
        <div className="check__titlebar">
          <p className="live__status">
            <span className="live__dot" aria-hidden="true" />
            <strong>Watching</strong>
            <span className="mono">{videoClock(elapsed)}</span>
          </p>
          <div className="check__actions">
            <button
              type="button"
              className="btn btn--small"
              onClick={() => {
                const next: Facing = facing === 'environment' ? 'user' : 'environment';
                setFacing(next);
                void start(next);
              }}
            >
              Switch camera
            </button>
            <button type="button" className="btn btn--small btn--stop" onClick={stop}>
              Stop
            </button>
          </div>
        </div>

        <DetectStage videoRef={videoRef} stream={stream} boxes={boxes} />

        <p className="live__hint" data-warn={String(cameraMoving)} aria-live="polite">
          {cameraMoving
            ? 'The camera is moving — hold it still or prop it up. Checks are paused until it settles.'
            : moving
              ? 'Something is moving. The white outline shows where.'
              : 'Nothing is moving right now.'}
        </p>

        {error ? (
          <p className="upload__error" role="alert">
            {error}
          </p>
        ) : null}

        {selected ? <FoundSummary found={selected} when={whenText(selected)} onOpenAlert={onOpenAlert} /> : null}
      </div>

      <aside className="check__side" aria-label="Found while watching">
        <h2 className="check__side-title">
          {found.length === 0 ? 'Nothing found yet' : `Found ${found.length} so far`}
        </h2>
        {found.length === 0 ? (
          <p className="check__side-lead">Anything unusual will appear here as it happens, newest at the top.</p>
        ) : (
          <ul className="list__items">
            {found.map((f) => (
              <li key={f.event.event_id}>
                <AlertCard
                  event={f.event}
                  cameras={[]}
                  selected={f.event.event_id === selectedId}
                  onOpen={() => setSelectedId(f.event.event_id)}
                  where={whenText(f)}
                  thumb={f.snapshot}
                />
              </li>
            ))}
          </ul>
        )}
      </aside>
    </div>
  );
}
