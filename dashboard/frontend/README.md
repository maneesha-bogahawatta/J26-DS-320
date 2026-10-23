# Operator Console — frontend

Current repository location: `dashboard/frontend/`. The original navigation,
recorded-video and live-camera workflows remain. The single Recorded video upload
also runs C3 temporal inference when enabled, using the same Alerts, Shift log,
review buttons and exports. There is no separate C3 research tab or home shortcut.

```powershell
# From this directory, after installing the existing lockfile dependencies:
npm run build:c3
# Then, from the repository root:
.\components\c3_aggression_detection\RUN_C3_DASHBOARD.ps1
```

For frontend development use `npm run dev:c3` with that backend on port 8000.
The existing `npm run dev`, `npm run build`, `.env.example` and `.env.c4`
configurations remain available. `build:c3` selects `.env.c3` (HTTP, same-origin
API/media, C3 enabled); ordinary builds retain the original practice defaults.
Obsolete root dashboard copies were archived outside the repository at the
user's request. Root npm shortcuts now target this original frontend. See
`../../components/c3_aggression_detection/DASHBOARD_INTEGRATION.md`.
Historical paths below predate moving the team application under `dashboard/`.

The shared operator dashboard for J26-DS-320. One UI, four detection components, one event
format.

Built with React 19 + TypeScript + Vite. No UI framework, no CSS framework — plain CSS with
design tokens, so the whole team can change it without learning a styling library.

---

## Run it

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open http://localhost:5173.

It works immediately with **no backend and no Python**. `VITE_API_MODE=mock` serves seeded
events from the browser, and `VITE_DETECT_MODE=browser` runs the practice detector in the
page — so uploading a video and using a webcam both work straight away.

### On a phone, as a live camera

```bash
npm run dev:phone
```

Open the `https://192.168…:5173` address it prints on the phone (same Wi-Fi), accept the
certificate warning, then **Use a live camera**. HTTPS is required — browsers refuse
camera access on plain `http://` from another device. Details in `docs/VSCODE_GUIDE.md`.

### Against the real API

```bash
# terminal 1
cd backend_mock && pip install -r requirements.txt && uvicorn main:app --reload

# terminal 2 — set VITE_API_MODE=http in frontend/.env first
cd frontend && npm run dev
```

### Other commands

| Command | What it does |
|---|---|
| `npm run dev` | dev server with hot reload |
| `npm run dev:phone` | same, over HTTPS on your network, so a phone can use its camera |
| `npm run build` | type-check and build to `dist/` |
| `npm run preview` | serve the production build |
| `npm run lint` | oxlint |

---

## What is in it

**Check video** — the start screen. Two choices: **check a recorded video** (drop a file;
the system watches it and marks anything unusual on a timeline) or **use a live camera**
(a phone or webcam; a white outline shows what is moving, a coloured box and a card appear
when something crosses the line). Everything found goes into Alerts as an ordinary event.

**Alerts** — what an officer uses to decide. A list of colour-coded cards on the left, one open
alert on the right. Below 1000px they become separate screens with a back button. Never
more than two things on screen at once.

**Shift log** — every alert and the decision taken on it, including the rejected ones,
with a response-time column (alert → first decision) and a CSV export for the
operator-performance study.

**Project team** — paste an event your pipeline produces, get the exact errors the
dashboard would raise, then push it into the live list. This is the self-service handoff
for Members 1–3. It is dimmed and sits apart, because it is not an officer's tab.

---

## Where things live

```
src/
  types/events.ts        THE CONTRACT. Mirrors backend_mock/schemas.py.
                         Do not change without team agreement.
  lib/
    plainLanguage.ts     Machine vocabulary -> words an officer reads.
                         Change wording HERE.
    urgency.ts           The one function that decides an alert's colour.
    incidentTypes.ts     Component registry, cautious labels, and the
                         "what to check / what to do" guidance per type.
    format.ts            Formatting. Every function handles null.
    validateEvent.ts     Runtime validation, used by the Project team tab.
  api/
    client.ts            The EventApi interface — the one seam to the data.
    mockClient.ts        In-memory seeded data. No backend needed.
    httpClient.ts        Real REST calls.
    seedData.ts          The demo events.
    index.ts             Picks mock or http from VITE_API_MODE.
  detect/
    engine.ts            The DetectionEngine interface — the seam for models.
    browserEngine.ts     Practice detector, runs in the page. No server.
    httpEngine.ts        Sends video / live frames to backend_mock/detect.py.
    practiceAnalyzer.ts  The practice detector's rules. Thresholds at the top.
    motion.ts            Frame differencing, background, ghost and glare checks.
    index.ts             Picks browser or http from VITE_DETECT_MODE.
  hooks/useEvents.ts     Loading, polling, operator actions.
  components/            One file per panel. Start at App.tsx.
    detect/              Check video: home, upload, live camera, timeline.
  styles/
    tokens.css           All colours, spacing and type. Change things HERE.
    app.css              Layout and component styles. No hard-coded colours.
public/clips/            Video clips. Gitignored except the synthetic demo ones.
```

### Adding a new incident type

Your detector can send any `incident_type` — the dashboard renders unknown ones with a
readable label, so you are never blocked. To give it a proper label, a "what to check"
line and a "what to do" line, add an entry to `src/lib/incidentTypes.ts`. That is the
only change needed.

---

## Design rules this UI follows

These are not style preferences. They come from who uses this — a security officer, not
an engineer — and from the proposal's ethics and accessibility sections.

**The two layout rules**

- **Never more than two things on screen at once.** A list and one open alert. That is
  the whole layout. If a third pane seems necessary, something belongs in a collapsed
  section instead.
- **One alert, one screen, one decision.** The open alert is a single column read top to
  bottom: what happened → the video → what the camera saw → what to do → four buttons.

**The colour rule**

- **Colour means urgency, and nothing else.** Red now, amber soon, blue when you can,
  green dealt with, grey nothing wrong. Nothing else in the interface is coloured, which
  is what lets an officer read the screen from across the room.
- **Never colour alone.** Every card also carries the urgency as a word. An officer with
  colour-vision deficiency loses the glance but loses no information.

**The language rules**

- **No machine vocabulary in the officer's view.** Not `low_posture_duration: 4.1` but
  "Stayed down for 4.1 seconds". The translation lives in `src/lib/plainLanguage.ts`,
  and there is a test that fails if snake_case leaks into the officer's screen.
- **A score is not a probability.** An uncalibrated score shows as a word — "Strong
  match" — never as `82%`, which reads as "82% chance this is real". The raw number is in
  the technical panel. A percentage appears only when a component sets
  `score_calibrated: true`.
- **Cautious wording everywhere.** "Someone may have fallen", never "Fall detected".
  "Item left on its own", never "Abandoned bag". The owner of an item is always "the
  person who may have left it".
- **Say what the camera saw, never what it means.** "Has not got up since" is an
  observation. "Unconscious" is a diagnosis this system cannot make.

**The honesty rules**

- **Missing data is shown, not hidden.** A missing clip gets an explanation, not a blank
  box. Null fields say so.
- **No identities.** Anonymous track labels only — "Person 12". No names, no faces, no
  biometrics. The API rejects anything in `track_id` that looks like a personal name.
- **Honest labelling.** The header always says whether the alerts are practice or real.
  "Live camera on" appears only while a camera is actually streaming. The Check screen
  says "Practice detector" whenever the in-page detector is in use, and its events are
  capped below "Strong match" — a rule of thumb must never look as sure as a trained model.
- **Nothing is recorded.** The practice detector reads the video in the page; uploads
  never leave the device. Live mode keeps one small still per found item, and nothing else.
- **Rejected alerts are kept.** "Nothing wrong" changes the status; it never deletes.
  Those rows are the hard negatives and the false-alarm rate.
- **Evidence against the alert is as visible as evidence for it.** Signals that argue
  against are green, so an officer dismissing something can see the grounds for
  dismissing it.

---

## Video clips

`public/clips/` holds four **synthetic** placeholder clips generated by
`tools/make_demo_clips.py`. They are drawn from scratch — a plain room and a figure made
of rectangles, with "SYNTHETIC DEMO CLIP — NOT CCTV FOOTAGE" stamped on every frame.

Each clip's motion matches the alert that references it (a figure drops and stays down; a
bag is put down and its owner walks away), and the seed events carry the figure's **real**
pixel position converted to normalised coordinates. That is deliberate: it means the
highlight overlay can be checked against a position we know exactly, which is the only
way to prove the box lands where the detector said it did.

Each clip ships as both `.mp4` and `.webm`. Some Linux builds of Chromium and Firefox have
no H.264 and refuse the mp4 silently; the player offers both sources so the demo works on
any machine in the lab.

To use your own footage, drop the files in `public/clips/` and point `media_ref` at them.

**Do not commit real CCTV footage or downloaded social-media video to this repository.**
`.gitignore` is set up to keep `public/clips/` out except the synthetic demo files. Check
with the supervisor before using any downloaded video at all, and blur identifiable faces
in anything that goes into a presentation.

`clip_05` is deliberately absent so the "Video unavailable" state can be demonstrated.

---

## The practice detector

Until the four trained models are plugged in, the Check screen uses a small rule-based
detector that runs in the browser (`src/detect/practiceAnalyzer.ts`). It works on a 64x36
grid of the picture and raises one pattern per component:

| Component | Pattern | Raised when |
|---|---|---|
| C1 | Movement spotted | something moves for over 1.2 s and nothing more specific applies |
| C2 | Someone hanging around | movement stays within a small area for 8 s (upload) / 12 s (live) |
| C3 | Possible aggressive behaviour | movement covers over 11% of the picture for 1.5 s |
| C4 | Someone may have fallen | movement comes down, then a wide, flat new shape stays still for 3 s |
| C4 | Item left on its own | a small new thing stays for 3 s **and** whatever brought it moved away |

It also rejects the usual false alarms: a camera being moved, "ghosts" (the spot a person
stood in when the check started), glare on bright windows, and a person who simply stopped.

Tested on the four synthetic clips and on two real CAVIAR clips from the project folder
(`LeftBag`, `LeftBag_PickedUp`): it finds the bag left at 0:38 in `LeftBag` and raises no
false item in either. It is still a rule of thumb — expect misses and false alarms on
crowded scenes, moving cameras and tiny distant objects.

To tune it, run `window.__PRACTICE_DEBUG = []` in the browser console, check a video, and
read `window.__PRACTICE_DEBUG`: every decision is listed with the numbers behind it.

