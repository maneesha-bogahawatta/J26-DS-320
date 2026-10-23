# Incident Event Contract — v1.1

**Project:** J26-DS-320 — Proactive Anomaly Detection and Security Alerting System
**Status:** agreed interface between all four components and the operator dashboard
**Owner of this document:** Member 4 (Bogahawatta M.O) — changes need team agreement

---

## 1. Why this exists

Each of us is building a different detector. If C1 returns `{"type": "tailgate", "conf": 81}`
and C3 returns `{"event": "AGGRESSION", "probability": 0.81}`, the dashboard has to grow a
special case per member, and integration becomes a week of arguing about field names in
the last fortnight of the project.

One format fixes that. You emit this shape; the dashboard renders it. You never touch
dashboard code, and I never touch your model code.

**You do not need to wait for anyone.** Build your detector, make it emit this shape, and it
appears in the queue.

---

## 2. The shape

```jsonc
{
  // ---------- required: every component must send these ----------
  "event_id":           "component_1-a3f9c2b1",   // unique across the whole system
  "component_id":       "component_1",            // component_1 … component_4
  "incident_type":      "tailgating",             // your detector's event name
  "camera_id":          "clip_03",                // camera or recorded-clip id
  "timestamp_detected": "2026-10-18T09:15:22Z",   // UTC, ISO-8601, trailing Z
  "priority":           "high",                   // low | medium | high
  "status":             "new",                    // always "new" from a pipeline
  "summary":            "Possible tailgating at the controlled door — review required.",
  "evidence":           [ /* see below */ ],      // [] is allowed, missing is not
  "model_name":         "spatial_rel_gnn",
  "model_version":      "0.1",

  // ---------- optional: send null when you do not have it ----------
  "confidence":           0.81,                   // 0..1 score, or null
  "bounding_box":         { "x_min": 0.18, "y_min": 0.30, "x_max": 0.46, "y_max": 0.92 },
  "track_id":             "person-44",            // the SUBJECT of the alert, or null
  "related_tracks":       [ /* see below */ ],    // other tracks involved, or null
  "media_ref":            "clips/clip_03.mp4",    // path to the clip, or null
  "media_offset_seconds": 2.0,                    // where in the clip it happens, or null
  "score_calibrated":     false,                  // true only if you calibrated it
  "zone_id":              "door_restricted_a"     // or null
}
```

### Evidence signals

This is the explainability layer. Each signal is one reason the alert fired, and the
operator sees them in plain language next to the clip.

```jsonc
"evidence": [
  { "name": "entry_gap",         "value": 1.4, "unit": "seconds",
    "description": "Time between the two tracks crossing the door line." },
  { "name": "follower_distance", "value": 0.9, "unit": "metres (est.)" },
  { "name": "queue_alignment",   "value": false }
]
```

- `name` — snake_case, required
- `value` — number, string, boolean or `null`
- `unit` — optional, e.g. `"seconds"`. Omit or `null` for unitless scores.
- `description` — optional but **please write one**. It is the difference between an
  operator trusting the alert and ignoring it, and it is the part the examiners will look at
  when we claim the system is explainable.

### Related tracks

Some incidents are about **one** track — a person falling. Others are about a
**relationship between tracks**, and a single `track_id` cannot express them. An abandoned
object has an object track *and* a likely owner. Tailgating has a leader *and* a follower.
Violence has both parties.

In those cases `track_id` holds the subject of the alert and `related_tracks` holds the
rest:

```jsonc
"track_id": "object-07",                       // the subject: the object itself
"related_tracks": [
  { "track_id": "person-31",
    "role": "likely_owner",                    // likely_owner | follower | leader |
                                               // counterpart | object | bystander
    "note": "Last track in sustained contact with the object before separation.",
    "association_score": 0.69 }                // how confident the association is, or null
]
```

`association_score` matters for abandoned objects specifically: the dashboard shows it next
to the owner track so the operator can see that ownership was **inferred from proximity**,
not established. Video cannot prove ownership, and the UI must not imply it did.

The Integration tab warns you if you send a relationship incident (`abandoned_object`,
`tailgating`, `unauthorized_proximity`, `violence`, `rapid_approach`) with no related track.
It is a warning, not an error — the event still renders.

---

## 3. The seven rules

1. **Timestamps are UTC, ISO-8601, trailing `Z`.**
   `datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")`
   Local time will be read as UTC and your alerts will be 5h30m out.

2. **Bounding boxes are normalised 0..1, not pixels.**
   `x_min = x_px / frame_width`, `y_min = y_px / frame_height`.
   The API rejects pixel values with a 422. This is so the box lands correctly whatever
   size the operator's window is.

3. **`track_id` is an anonymous tracker label** — `"person-12"`, `"object-07"`.
   Never a name, never a face ID. The API rejects anything that looks like a personal name.
   Our ethics section says the system holds no identities; this is where that is enforced.

4. **Missing means `null`.** Not `0`, not `""`, not a guessed value. The dashboard shows
   "Not provided" and the operator knows where they stand. A fabricated `0.0` confidence
   reads as a real measurement.

5. **`confidence` is a score, not a probability.** The dashboard labels it
   "Model score (uncalibrated)" and shows `0.81`, never `81%`. If you genuinely calibrate
   against a held-out set, set `score_calibrated: true` and say so in your write-up.

6. **Summaries stay cautious.** "Possible tailgating — review required", not "Tailgating
   detected" or "Intruder". CCTV does not prove intent, ownership or identity. The API warns
   on overclaiming words.

7. **`event_id` must be unique.** `f"{component_id}-{uuid.uuid4().hex[:8]}"` is fine.
   A duplicate gets a 409.

---

## 4. Incident types

Use these if they fit. If your detector finds something else, **just send it** — the
dashboard renders unknown types with a readable label and an "Unregistered type" marker,
so you are never blocked. Tell Member 4 and it gets a proper label and operator hint.

| Component | `component_id` | Suggested `incident_type` values |
|---|---|---|
| C1 — Spatial Anomaly Detection | `component_1` | `tailgating`, `restricted_zone_intrusion`, `unauthorized_proximity`, `crowding` |
| C2 — Predictive Loitering | `component_2` | `loitering` |
| C3 — Aggression & Violence | `component_3` | `aggression`, `violence`, `concealment`, `rapid_approach` |
| C4 — Fall & Abandoned Object | `component_4` | `fall`, `abandoned_object` |

### Evidence signals worth sending, by component

These are suggestions, not requirements — send what your model actually computes.

- **C1:** `entry_gap`, `follower_distance`, `person_count`, `baseline_count`, `zone_dwell`
- **C2:** `dwell_time`, `path_reversals`, `trajectory_entropy`, `queue_alignment`
- **C3:** `approach_velocity`, `limb_contact_events`, `torso_lean`, `occlusion_ratio`
- **C4 falls:** `downward_motion`, `posture_aspect_ratio`, `torso_angle`,
  `low_posture_duration`, `post_fall_movement`
  *(the last two are what separate a fall from sitting, bending, kneeling, lying down or
  picking something up — send them, they are the discrimination the operator needs)*
- **C4 objects:** `object_dwell_time`, `owner_separation`, `object_displacement`,
  `owner_association_score`, plus the owner in `related_tracks`

`occlusion_ratio` is worth sending from any component that can compute it — occlusion is a
named weakness in our literature review, and showing the operator when a detection was
half-hidden is a real contribution.

---

## 5. The API

Base URL in development: `http://127.0.0.1:8000`

| Method | Path | Who calls it |
|---|---|---|
| `GET` | `/api/v1/health` | anyone, to check the service is up |
| `GET` | `/api/v1/cameras` | dashboard |
| `GET` | `/api/v1/events` | dashboard. Filters: `?status=` `?camera_id=` `?component_id=` `?incident_type=` |
| `GET` | `/api/v1/events/{event_id}` | dashboard |
| **`POST`** | **`/api/v1/events`** | **your pipeline** |
| `PATCH` | `/api/v1/events/{event_id}` | dashboard, when the operator decides |
| `POST` | `/api/v1/detect/video` | dashboard's Check video screen; runs **your registered model** — see section 7 |
| `POST` | `/api/v1/detect/frame` | dashboard's live camera; one JPEG per second — see section 7 |

Interactive explorer with a "Try it out" button: **http://127.0.0.1:8000/docs**

### Status codes you should expect

| Code | Meaning | What to do |
|---|---|---|
| `201` | accepted | nothing |
| `422` | a field is wrong | read the response body — it names the exact field |
| `409` | duplicate `event_id` | generate a fresh id |

---

## 6. Copy this into your component

```python
# components/<your_component>/emit.py
import uuid, datetime, requests

API = "http://127.0.0.1:8000/api/v1/events"
COMPONENT_ID = "component_1"        # <- change to yours
MODEL_NAME = "spatial_rel_gnn"      # <- change to yours
MODEL_VERSION = "0.1"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_box(x1, y1, x2, y2, frame_w, frame_h) -> dict:
    """Pixel box -> the normalised box the contract requires."""
    return {
        "x_min": max(0.0, min(1.0, x1 / frame_w)),
        "y_min": max(0.0, min(1.0, y1 / frame_h)),
        "x_max": max(0.0, min(1.0, x2 / frame_w)),
        "y_max": max(0.0, min(1.0, y2 / frame_h)),
    }


def emit(incident_type, camera_id, summary, evidence, *,
         confidence=None, bbox=None, track_id=None, media_ref=None,
         media_offset_seconds=None, priority="medium", zone_id=None):
    event = {
        "event_id": f"{COMPONENT_ID}-{uuid.uuid4().hex[:8]}",
        "component_id": COMPONENT_ID,
        "incident_type": incident_type,
        "camera_id": camera_id,
        "timestamp_detected": utc_now(),
        "priority": priority,
        "status": "new",
        "summary": summary,
        "evidence": evidence,
        "bounding_box": bbox,
        "track_id": track_id,
        "media_ref": media_ref,
        "media_offset_seconds": media_offset_seconds,
        "confidence": confidence,
        "score_calibrated": False,
        "zone_id": zone_id,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }
    response = requests.post(API, json=event, timeout=5)
    if response.status_code == 422:
        print("Event rejected. The server says exactly what is wrong:")
        print(response.json())
    response.raise_for_status()
    return response.json()


# Example call from inside your detector loop:
# emit(
#     incident_type="tailgating",
#     camera_id="clip_03",
#     summary="Possible tailgating at the controlled door - review required.",
#     evidence=[
#         {"name": "entry_gap", "value": 1.4, "unit": "seconds",
#          "description": "Time between the two tracks crossing the door line."},
#         {"name": "follower_distance", "value": 0.9, "unit": "metres (est.)"},
#     ],
#     confidence=0.81,
#     bbox=normalise_box(86, 81, 221, 248, frame_w=480, frame_h=270),
#     track_id="person-44",
#     media_ref="clips/clip_03.mp4",
#     media_offset_seconds=2.0,
#     priority="high",
#     zone_id="door_restricted_a",
# )
```

---

## 7. Check your output before you write any HTTP code

Open the dashboard, go to the **Integration** tab, paste one event your detector produces,
and read the result. You get the same errors and warnings the real API would give — field
by field — and a button that pushes the event straight into the alert queue so you can see
how it looks to an operator.

That is the fastest way to know you are integrated. It takes about two minutes and needs
nothing from anyone else.

---

## 8. Version history

| Version | Change |
|---|---|
| 1.0 | Initial contract: core fields, evidence array, normalised boxes, track IDs |
| 1.1 | Added optional `media_offset_seconds`, `score_calibrated`, `zone_id`, `related_tracks`, `audit`. All backwards compatible — a v1.0 event is still valid. |

Changing this document means changing `frontend/src/types/events.ts`,
`backend_mock/schemas.py` and `shared/schemas/event_schema.py` together. Raise it in the
standup first.

---

## 7. Plugging your model into "Check video"

The dashboard's start screen lets an officer **upload a recorded video** or
**point a live camera** (a phone or webcam) at a scene. Two ways for your model
to take part — pick whichever suits it:

**A. The server calls your model** (recommended for the demo). Write one
function, register it, and the dashboard's upload and live screens use it when
`VITE_DETECT_MODE=http`:

```python
# backend_mock/detectors/c2_loitering.py
from detect import video_detector, frame_detector

@video_detector("c2_loitering")
def run_on_video(video_path, source_id):      # a whole uploaded file
    return [ ...IncidentEvent... ]             # set media_offset_seconds!

@frame_detector("c2_loitering")
def run_on_frame(jpeg_bytes, source_id, t):   # one live frame, ~1 per second
    return [ ...IncidentEvent... ]             # keep your own per-source buffer
```

A complete example is `backend_mock/detectors/_example_c4.py`. Check what is
loaded at `GET /api/v1/detect/detectors`.

**B. Your pipeline runs on its own** and POSTs to `/api/v1/events` (section 6).
Nothing changes — events appear in Alerts as before.

Either way the event shape is this contract. For uploads, always send
`media_offset_seconds` — the dashboard uses it to put a mark on the timeline and
jump the video to the moment.

**Until the models are ready**, the dashboard ships a *practice detector*
(`VITE_DETECT_MODE=browser`, the default) that runs in the page. It measures
real movement — boxes land on what actually moved — and raises one simple
pattern per component (movement, staying put, a lot of fast movement, a drop
then lying still, an item left after its carrier moved away). Its events say
`model_name: "practice-motion-check"`, are capped below "strong match", and the
screen labels it as a practice check. It is a stand-in, not a baseline for the
evaluation.

