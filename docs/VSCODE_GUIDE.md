# Working on the dashboard in VS Code

Follows on from the Team Onboarding Guide. That one covers Git, Python and the repo; this
one covers the dashboard and the alerting API specifically.

---

## 1. One-time setup

You already have VS Code, Git and Python from the onboarding guide. Two more things:

### Node.js

The dashboard needs Node 20 or newer. Check:

```bash
node --version
```

Nothing, or a number below 20? Install the **LTS** version from <https://nodejs.org>, then
close VS Code and reopen it so it picks up the new PATH.

### Install the dashboard's packages

Open the repo in VS Code, press **Ctrl+`** (the backtick key, top-left) for a terminal:

```bash
cd frontend
npm install
cp .env.example .env
```

`npm install` takes a few minutes the first time. You only do it once, and again whenever
someone adds a package.

When you open the repo, VS Code offers a set of recommended extensions in the bottom-right
corner. Click **Install** — they are listed in `.vscode/extensions.json` and are the ones
this project actually uses.

---

## 2. Running it

**The easy way:** `Ctrl+Shift+P` → type `Run Task` → **Start everything**. That starts the
dashboard and the API side by side in two terminal panels.

**Or by hand**, in two terminals (split with the ⌄ icon next to the `+` in the terminal
panel):

```bash
# terminal 1 — the dashboard
cd frontend
npm run dev

# terminal 2 — the API (only needed when VITE_API_MODE=http)
cd backend_mock
uvicorn main:app --reload
```

Then **Ctrl+click** the `http://localhost:5173` link in the terminal. The first
screen is **Check video**: upload a recording, or start a live camera.

### Using your phone as the live camera

Browsers only open a camera on a secure (`https`) page or on `localhost`. Your
phone is neither when it opens your laptop's address, so there is a second
command for it:

```bash
cd frontend
npm run dev:phone
```

1. Your phone and laptop must be on the **same Wi-Fi**.
2. The terminal prints a line like `Network: https://192.168.1.20:5173/` — open
   **that** address on the phone (type it in, including `https://`).
3. The phone warns that the connection is not private (the certificate is
   home-made). Tap **Advanced → Proceed**. This is expected, and only for
   development.
4. Tap **Check video → Use a live camera → Start the camera**, allow the camera.
5. Prop the phone up. Moving the camera pauses the checks.

University or public Wi-Fi often blocks phone-to-laptop connections. If the
phone cannot load the page, use a phone hotspot with the laptop connected to
it.

Leave it running while you work. Save a file and the browser updates by itself — no
restart, no refresh.

### Mock mode vs API mode

`frontend/.env` decides where alerts come from:

| Setting | What happens |
|---|---|
| `VITE_API_MODE=mock` | Practice alerts generated in the browser. No Python needed. **Use this for UI work.** |
| `VITE_API_MODE=http` | Real alerts from `backend_mock`. Use this when testing a pipeline. |

Changing `.env` needs a restart of `npm run dev` — Vite reads it once at startup.

---

## 3. Where to make changes

| You want to… | Open this |
|---|---|
| Change wording an officer reads | `frontend/src/lib/plainLanguage.ts` |
| Add a new alert type, or its guidance | `frontend/src/lib/incidentTypes.ts` |
| Change colours, text size, spacing | `frontend/src/styles/tokens.css` |
| Change the layout of a panel | the matching file in `frontend/src/components/` |
| Change what the alert list shows | `frontend/src/components/AlertList.tsx` |
| Change the upload / live camera screens | `frontend/src/components/detect/` |
| Tune the practice detector | `frontend/src/detect/practiceAnalyzer.ts` (thresholds at the top) |
| Plug your model into upload / live | `backend_mock/detectors/` — copy `_example_c4.py` |
| Change the event format | `frontend/src/types/events.ts` **and** `backend_mock/schemas.py` **and** `docs/EVENT_CONTRACT.md` — all three, with the team's agreement |
| Change an API endpoint | `backend_mock/main.py` |

Start at `frontend/src/App.tsx` if you are trying to find your way around — it is the top
of the tree and every panel is imported there by name.

**Jump to a file fast:** `Ctrl+P` then start typing the filename.
**Jump to a function:** `Ctrl+T` then type its name.
**See where something is used:** right-click it → *Find All References*.

---

## 4. The two commands worth knowing

```bash
# In frontend/ — does it compile? Run this before every push.
npm run build

# In backend_mock/ — does the API still honour the contract? 18 checks.
python smoke_test.py
```

If `npm run build` prints errors, VS Code is showing you the same ones: `Ctrl+Shift+M`
opens the Problems panel. Red squiggles are errors, yellow are warnings. A red squiggle
will fail the build, so fix those before pushing.

---

## 5. Everyday Git in VS Code

Same as the onboarding guide; the Source Control icon in the left sidebar is all you need.

1. **Morning:** Source Control → `...` → **Pull**
2. **As you work:** stage with `+`, write a message, click ✓, then `...` → **Push**
3. **Message format:** `[C4] Make the fall alert wording clearer`

Check the bottom-left corner shows your branch (`member4/critical-incidents`) before you
commit.

**Never commit:** `node_modules/`, `frontend/dist/`, `.env`, or real CCTV footage. All
four are in `.gitignore` already — if you see any of them in the Source Control list,
something is wrong, so ask before committing.

---

## 6. When something breaks

**`npm` is not recognised**
Node is not installed, or VS Code was open when you installed it. Close VS Code completely
and reopen.

**`npm run dev` says the port is in use**
It is already running in another terminal. Either use that one, or run `npm run dev -- --port 5174`.

**The dashboard loads but shows no alerts**
Check `frontend/.env`. If `VITE_API_MODE=http`, the API must be running — open
<http://127.0.0.1:8000/api/v1/health> in a browser. If that fails, start the API. Switch
to `mock` if you just want to work on the UI.

**"Could not reach the API" banner**
Same cause. The dashboard is fine; the backend is not running.

**Red squiggles everywhere right after cloning**
You skipped `npm install`. Run it in `frontend/`.

**TypeScript errors that the build does not report**
VS Code may be using its own TypeScript. `Ctrl+Shift+P` → *TypeScript: Select TypeScript
Version* → **Use Workspace Version**.

**"The browser will not open the camera on this page"**
You opened the dashboard on another device over `http://`. Use `npm run dev:phone`
and the `https://` address (see *Using your phone as the live camera*).

**"Camera access was refused"**
The browser remembered a "Block". Tap the camera/lock icon in the address bar,
set Camera to Allow, reload.

**An uploaded video says the browser cannot play its format**
Old CCTV exports (`.mpg`, `.avi`, `.dav`) often need converting first. With
ffmpeg: `ffmpeg -i input.mpg -c:v libx264 -pix_fmt yuv420p output.mp4`.

**Tuning the practice detector**
In the browser console type `window.__PRACTICE_DEBUG = []`, check a video, then
type `window.__PRACTICE_DEBUG` — every decision is listed with the numbers
behind it (why something was raised, or why it was ignored as glare, a person
standing still, and so on).

**The video does not play**
The clips live in `frontend/public/clips/`. If they are missing, regenerate the synthetic
placeholders: `python3 tools/make_demo_clips.py`.

---

## 7. Testing a teammate's pipeline output

The fastest loop, and it needs nothing from Member 4:

**Option A — your model runs inside the API** (the upload and live screens use
it directly): copy `backend_mock/detectors/_example_c4.py` to a new file without
the underscore, fill in your model, restart uvicorn, and set
`VITE_DETECT_MODE=http` in `frontend/.env`. Section 7 of `docs/EVENT_CONTRACT.md`
has the details.

**Option B — your pipeline runs on its own** and sends events:

1. Start the dashboard (`npm run dev`).
2. Open the **Project team** tab.
3. Paste one event your detector produced into the box.
4. Read the errors and warnings — each names the exact field.
5. Fix your emitter, paste again.
6. When it is clean, click **Send to the alert queue** and open the **Alerts** tab to see
   how an officer would see it.

Once that works, point your pipeline at `POST http://127.0.0.1:8000/api/v1/events` with the
same JSON. Section 6 of `docs/EVENT_CONTRACT.md` has a copy-paste Python function.
