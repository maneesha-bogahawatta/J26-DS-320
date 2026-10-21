/**
 * The header. One line, three things.
 *
 * The count of alerts needing a look is the biggest element on the screen
 * after the video, and it changes colour when it is not zero. An officer
 * walking back to the desk should be able to tell from the doorway whether
 * anything happened.
 *
 * Two honesty rules live here, and they are not cosmetic:
 *   1. It says whether these are practice alerts or a real feed. Nobody
 *      should ever demo seeded events believing they are real detections.
 *   2. "Live camera on" appears ONLY while a camera is actually streaming on
 *      the Check screen. Nothing else in the interface may claim to be live.
 */

import type { HealthStatus } from '../types/events';

export type Screen = 'check' | 'alerts' | 'log' | 'team';

interface AppHeaderProps {
  mode: 'mock' | 'http';
  health: HealthStatus | null;
  unreviewedCount: number;
  screen: Screen;
  onScreenChange: (screen: Screen) => void;
  onRefresh: () => void;
  refreshing: boolean;
  /** True while the live camera is streaming. */
  liveOn: boolean;
}

export function AppHeader({
  mode,
  health,
  unreviewedCount,
  screen,
  onScreenChange,
  onRefresh,
  refreshing,
  liveOn,
}: AppHeaderProps) {
  const connected = health?.ok ?? false;

  return (
    <header className="header">
      <div className="header__brand">
        <span className="header__mark" aria-hidden="true">
          ◉
        </span>
        <span className="header__name">Security Alerts</span>
      </div>

      <div className="header__count" data-attention={String(unreviewedCount > 0)}>
        <span className="header__count-num">{unreviewedCount}</span>
        <span className="header__count-word">
          {unreviewedCount === 1 ? 'needs a look' : 'need a look'}
        </span>
      </div>

      <nav className="header__nav" aria-label="Sections">
        <button
          type="button"
          className="navbtn"
          aria-current={screen === 'check'}
          onClick={() => onScreenChange('check')}
        >
          <span className="navbtn__long">Check video</span>
          <span className="navbtn__short">Check</span>
        </button>
        <button
          type="button"
          className="navbtn"
          aria-current={screen === 'alerts'}
          onClick={() => onScreenChange('alerts')}
        >
          Alerts
        </button>
        <button
          type="button"
          className="navbtn"
          aria-current={screen === 'log'}
          onClick={() => onScreenChange('log')}
        >
          <span className="navbtn__long">Shift log</span>
          <span className="navbtn__short">Log</span>
        </button>
        {/* Not an officer's tab. It is the integration surface for the project
            team, so it is dimmed and sits apart from the two that are. */}
        <button
          type="button"
          className="navbtn navbtn--quiet navbtn--team"
          aria-current={screen === 'team'}
          onClick={() => onScreenChange('team')}
        >
          Project team
        </button>
      </nav>

      <div className="header__right">
        <span
          className="pill"
          title={
            mode === 'mock'
              ? 'These are practice alerts for testing. They are not real detections.'
              : 'Alerts are arriving from the detection system.'
          }
        >
          <span className="pill__dot" data-ok={String(connected)} aria-hidden="true" />
          {mode === 'mock' ? 'Practice data' : connected ? 'Connected' : 'Disconnected'}
        </span>
        {liveOn ? (
          <span className="pill pill--live" title="A camera is streaming on the Check screen right now.">
            <span className="pill__dot pill__dot--live" aria-hidden="true" />
            Live camera on
          </span>
        ) : null}
        <button type="button" className="btn" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? 'Checking…' : 'Check now'}
        </button>
      </div>
    </header>
  );
}
