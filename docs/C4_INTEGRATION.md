# Component 4 in the dashboard (Fall + Abandoned Object)

What was added (nothing else in the repo was changed except one number in `httpEngine.ts`):

| File | What it does |
| --- | --- |
| `backend_mock/detectors/c4.py` | Registers the C4 video + live-frame detector. Returns contract v1.1 events (`fall`, `abandoned_object`). |
| `backend_mock/c4_core/` | The trained fall pipeline (MediaPipe pose -> CNN-LSTM ensemble) and the object monitor. |
| `backend_mock/requirements_c4.txt` | Extra packages (numpy, opencv-python, torch, mediapipe). |
| `frontend/.env.c4` | Env file that switches the dashboard to the real backend and detectors. |
| `frontend/src/detect/httpEngine.ts` | Live frames are now sent 5 times a second instead of once (C4 timers assume ~5 fps). |

## One-time setup (Python 3.12)

```bash
cd backend_mock
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements_c4.txt

mkdir -p ~/c4_weights        # put the 3 files here (from Drive: J26-DS-320/datasets/)
#   fall_cnn_lstm_gmdcsa_final_s0.pt  _s1.pt  _s2.pt
```

Add to the repo `.gitignore` so weights are never committed:

```
*.pt
*.task
backend_mock/uploads/
```

## Run (two terminals)

```bash
# terminal 1 - backend
cd backend_mock && source .venv/bin/activate
uvicorn main:app --port 8000
# look for:  "C4 fall branch ready"  (first upload/frame triggers the load)

# terminal 2 - frontend
cd frontend && cp .env.c4 .env && npm install && npm run dev
```

Open http://localhost:5173 -> Check a recorded video, or Live camera (laptop webcam).
Check http://127.0.0.1:8000/api/v1/detect/detectors lists `c4_fall_and_object`.

## Videos for the demo

Browsers cannot play `.mpg`, so convert the CAVIAR clips first (quality matters - this setting was tested):

```bash
ffmpeg -i LeftBag.mpg -c:v libx264 -crf 18 -pix_fmt yuv420p -an LeftBag.mp4
```

Upload `LeftBag.mp4` -> one "item left on its own" alert at ~0:40 (owner left at 0:40, alert fires 10 s later).
Upload any Human Fall clip -> a "may have fallen" alert.

## Settings (environment variables, all optional)

`C4_WEIGHTS_DIR` (~/c4_weights), `C4_FALL_THR` (0.40), `C4_FALL_K` (1), `C4_T_UNATTENDED` (10),
`C4_FALL=0` / `C4_OBJECT=0` to switch a branch off, `C4_OBJECT_PRESET=none` for non-CAVIAR cameras.

## Honest limits (say these out loud)

- Fall branch: one person, trained on GMDCSA-24, wild test 8/9 with 6 false alarms. Live mode feeds it ~5 frames/s instead of the ~15 it was run at on video; that live rate has not been evaluated.
- Object branch: validated only on the 5 CAVIAR lobby clips (overhead camera). Other views need retuning. No owner identity.
- If weights or MediaPipe are missing the fall branch is skipped (see the backend log) and the object branch still runs.
