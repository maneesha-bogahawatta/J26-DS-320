/**
 * The "Check" section: the home choice, recorded video, and live camera.
 *
 * Recorded and live stay mounted once opened, so switching between them — or
 * going to Alerts and back — does not lose a finished check or stop a camera
 * that is watching.
 */

import { useState } from 'react';
import type { IncidentEvent } from '../../types/events';
import { C3_ENABLED } from '../../api';
import { detector } from '../../detect';
import { CheckHome } from './CheckHome';
import { LiveCheck } from './LiveCheck';
import { UploadCheck } from './UploadCheck';

export type CheckMode = 'home' | 'upload' | 'live';

interface CheckScreenProps {
  mode: CheckMode;
  onModeChange: (mode: CheckMode) => void;
  onFound: (events: IncidentEvent[], source: { id: string; label: string }) => void;
  onOpenAlert: (eventId: string) => void;
  onLiveChange: (on: boolean) => void;
  needLook: number;
  onOpenAlerts: () => void;
}

export function CheckScreen({
  mode,
  onModeChange,
  onFound,
  onOpenAlert,
  onLiveChange,
  needLook,
  onOpenAlerts,
}: CheckScreenProps) {
  const [opened, setOpened] = useState({ upload: false, live: false });
  if (mode !== 'home' && !opened[mode]) setOpened({ ...opened, [mode]: true });

  return (
    <div className="checkscreen">
      {mode === 'home' ? (
        <CheckHome onChoose={onModeChange} needLook={needLook} onOpenAlerts={onOpenAlerts} />
      ) : (
        <div className="subbar">
          <button type="button" className="subbar__back" onClick={() => onModeChange('home')}>
            ← Back
          </button>
          <div className="segmented segmented--bar" role="tablist" aria-label="What to check">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'upload'}
              aria-pressed={mode === 'upload'}
              onClick={() => onModeChange('upload')}
            >
              Recorded video
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'live'}
              aria-pressed={mode === 'live'}
              onClick={() => onModeChange('live')}
            >
              Live camera
            </button>
          </div>
          {detector.mode === 'browser' ? (
            <p className="subbar__note" title={detector.description}>
              Practice detector — a rough guide, not the trained models
            </p>
          ) : (
            <p className="subbar__note">{C3_ENABLED && mode === 'upload' ? 'C3 temporal research + registered team video detectors' : 'Registered team detectors; C3 is unavailable on single live frames'}</p>
          )}
        </div>
      )}

      {opened.upload ? (
        <div className="checkscreen__pane" hidden={mode !== 'upload'}>
          <UploadCheck onFound={onFound} onOpenAlert={onOpenAlert} />
        </div>
      ) : null}
      {opened.live ? (
        <div className="checkscreen__pane" hidden={mode !== 'live'}>
          <LiveCheck onFound={onFound} onOpenAlert={onOpenAlert} onLiveChange={onLiveChange} />
        </div>
      ) : null}
    </div>
  );
}
