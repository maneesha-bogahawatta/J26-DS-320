# Mock Alerting API

The shared event API for J26-DS-320. Detection components POST incident events here; the
operator dashboard reads them and PATCHes operator decisions back.

It exists so Members 1–3 can integrate **today**, against a server that validates the
contract properly, without waiting for anyone to choose a database.

---

## Run it

```bash
cd backend_mock
pip install -r requirements.txt
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000
- **Interactive explorer: http://127.0.0.1:8000/docs** — you can POST a test event from the
  browser with the "Try it out" button, no code needed.

### Check it honours the contract

```bash
python smoke_test.py
```

Eighteen checks covering the whole flow: a valid POST, a duplicate rejection, a pixel
bounding box rejection, a personal name in `track_id` rejection, a bad timestamp
rejection, the operator decision flow, and the audit trail. If this passes, your pipeline
can integrate.

---

## Endpoints

| Method | Path | Who calls it |
|---|---|---|
| `GET` | `/api/v1/health` | anyone |
| `GET` | `/api/v1/cameras` | dashboard |
| `GET` | `/api/v1/events` | dashboard. Filters: `?status=` `?camera_id=` `?component_id=` `?incident_type=` |
| `GET` | `/api/v1/events/{event_id}` | dashboard |
| **`POST`** | **`/api/v1/events`** | **detection pipelines** |
| `PATCH` | `/api/v1/events/{event_id}` | dashboard |
| `POST` | `/api/v1/detect/video` | dashboard's **Check video** screen — body is the raw video file |
| `POST` | `/api/v1/detect/frame` | dashboard's **live camera** — body is one JPEG, about once a second |
| `GET` | `/api/v1/detect/detectors` | anyone — which models are plugged in |

The two `detect` endpoints run every model registered in `detectors/` and store
what they return, so it shows up in Alerts. Used when the dashboard has
`VITE_DETECT_MODE=http`. **How to plug your model in:** copy
`detectors/_example_c4.py` to a new file without the leading underscore, fill in
your model, restart uvicorn. Full instructions are at the top of `detect.py`.

Full field reference: [`../docs/EVENT_CONTRACT.md`](../docs/EVENT_CONTRACT.md)

---

## What it validates, and why

The strictness is deliberate. Each of these is a bug that would otherwise surface as a
confusing dashboard rather than a clear error.

| Rule | Why |
|---|---|
| Bounding boxes must be 0..1 | Pixel coordinates draw the box in the wrong place, silently |
| Timestamps must be ISO-8601 | A local timestamp is 5h30m out and the queue order breaks |
| `track_id` must not look like a name | Our ethics section says the system holds no identities |
| `confidence` must be 0..1 | `81` and `0.81` mean very different things to the score meter |
| `event_id` must be unique | Duplicate alerts inflate every metric we report |
| Box must have positive area | A zero-area box is a coordinate-order mistake |

A rejection returns `422` with the exact field named in the response body. Read it — it
tells you precisely what to fix.

---

## Files

```
main.py            The API. Routes and error handling.
schemas.py         Pydantic models — the server-side twin of
                   frontend/src/types/events.ts. Keep them in step.
store.py           In-memory store with optional JSON persistence.
seed_events.json   Demo events so the queue is not empty on first run.
smoke_test.py      End-to-end contract test.
detect.py          The Check video endpoints, and the @video_detector /
                   @frame_detector decorators your model registers with.
detectors/         One file per model. Imported at start-up. Files starting
                   with "_" are skipped (that is how the example stays off).
uploads/           Videos sent for checking. Gitignored — never commit.
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PERSIST_EVENTS` | `0` | `1` writes to `events_store.json` so events survive a restart |
| `SHIFT_SEED_TIMES` | `1` | Shifts seeded timestamps so a demo shows a recent queue. `0` keeps the literal times in the file. |

---

## Limits — read before anyone demos this

This is a **prototype server**: no authentication, in-memory storage, single process, CORS
open to localhost. It is fine for development and for the project demo. It must not go on a
public network, and it is not the final backend. When the group needs real persistence,
replace `store.py` — `main.py` does not care how events are stored.
