/**
 * The first screen: two big choices and nothing else that needs reading.
 *
 * Below them, one line per component saying what the system looks for — the
 * four parts of the project in words an officer would use.
 */

interface CheckHomeProps {
  onChoose: (mode: 'upload' | 'live') => void;
  needLook: number;
  onOpenAlerts: () => void;
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 48 48" width="48" height="48" aria-hidden="true">
      <rect x="6" y="10" width="36" height="28" rx="4" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <path d="M20 19v10l9-5z" fill="currentColor" />
      <path d="M38 4v10M33 9l5-5 5 5" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 48 48" width="48" height="48" aria-hidden="true">
      <rect x="14" y="4" width="20" height="40" rx="4" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <circle cx="24" cy="18" r="5" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <circle cx="24" cy="18" r="1.6" fill="currentColor" />
      <path d="M21 38h6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      <path
        d="M6 16a14 14 0 0 1 0 12M42 16a14 14 0 0 0 0 12"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

const LOOKS_FOR = [
  { glyph: '⇥', text: 'People going where they should not, or following someone through a door', tag: 'C1' },
  { glyph: '◷', text: 'Someone hanging around in one place for too long', tag: 'C2' },
  { glyph: '✸', text: 'Fights and aggressive behaviour', tag: 'C3' },
  { glyph: '↯', text: 'Someone falling, and bags or items left behind', tag: 'C4' },
];

export function CheckHome({ onChoose, needLook, onOpenAlerts }: CheckHomeProps) {
  return (
    <section className="home">
      <h1 className="home__title">What would you like to check?</h1>
      <p className="home__lead">
        The system watches the video and points out anything unusual. You decide what to do about it.
      </p>

      <div className="home__choices">
        <button type="button" className="choice" onClick={() => onChoose('upload')}>
          <span className="choice__icon">
            <UploadIcon />
          </span>
          <span className="choice__title">Check a recorded video</span>
          <span className="choice__text">Upload a clip from a CCTV recorder or a phone.</span>
          <span className="choice__cta">Choose a video →</span>
        </button>

        <button type="button" className="choice" onClick={() => onChoose('live')}>
          <span className="choice__icon">
            <CameraIcon />
          </span>
          <span className="choice__title">Use a live camera</span>
          <span className="choice__text">Point a phone or webcam at the scene and watch as it happens.</span>
          <span className="choice__cta">Start the camera →</span>
        </button>
      </div>

      {needLook > 0 ? (
        <button type="button" className="home__pending" onClick={onOpenAlerts}>
          <span className="home__pending-num">{needLook}</span>
          {needLook === 1 ? 'alert needs a look' : 'alerts need a look'} →
        </button>
      ) : null}

      <div className="home__looks">
        <h2 className="home__looks-title">What it looks for</h2>
        <ul className="home__looks-list">
          {LOOKS_FOR.map((item) => (
            <li key={item.tag}>
              <span className="home__looks-glyph" aria-hidden="true">
                {item.glyph}
              </span>
              <span>{item.text}</span>
              <span className="home__looks-tag">{item.tag}</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
