# Frontend Handoff — what exists, and what each member does next

**From:** Member 4 (Bogahawatta M.O, IT23242418) — C4 Fall & Abandoned-Object Detection
**Re:** the shared operator dashboard and alerting layer
**Date:** 18 October 2026

---

## 1. What this is

The operator dashboard from the proposal is built and running. It is the **shared** alerting
layer for the whole project — not a Component 4 app. All four components feed into it
through one event format.

Three things are in the repository now:

| Folder | What it is |
|---|---|
| `frontend/` | The operator console. React + TypeScript. Runs with no backend. |
| `backend_mock/` | The event API your pipelines POST to. FastAPI, three dependencies. |
| `docs/EVENT_CONTRACT.md` | The one format we all emit. Read this before you write your emitter. |

It works today on seeded demo data, so the UI, the operator workflow and the audit trail
can be reviewed and criticised now rather than in the last fortnight.

---

## 2. What it does not claim

Worth being blunt about this, because it is what a marker will probe:

- It runs on **recorded clips**, not a live CCTV stream. The top bar says so, permanently,
  until someone actually connects and tests a feed.
- The demo data is **seeded**, and the top bar says "Mock data" whenever it is. Nobody
  should ever demo seeded events believing they are real detections.
- The clips in the repo are **synthetic placeholders** — generated video with a moving
  rectangle, not CCTV footage.
- No detector is finished. The dashboard being done proves the dashboard is done. Nothing
  else.

---

## 3. What this unblocks

The thing that normally kills a four-person integration is everyone waiting on everyone.
This removes that:

- **You do not wait for the dataset.** Your detector emits events; the dashboard renders
  them. It does not care how good your model is.
- **You do not wait for me.** Send an `incident_type` I have never seen and it still
  renders, with a readable label and an "Unregistered type" marker.
- **I do not wait for you.** The dashboard is already finished against mock events and
  swaps to real pipelines with a one-line config change.
- **Nobody has to agree on a database** before agreeing on a format.

---

## 4. What each of you does

### Everyone — 15 minutes, this week

1. Pull `develop`.
2. `cd frontend && npm install && cp .env.example .env && npm run dev`
3. Open http://localhost:5173 and click around. Look at the **Integration** tab.
4. Read [`docs/EVENT_CONTRACT.md`](EVENT_CONTRACT.md).
5. Bring any objection to the format to the standup **now**, while changing it is cheap.

### Member 1 — Spatial Anomaly Detection (`component_1`)

Emit `tailgating`, `restricted_zone_intrusion`, `unauthorized_proximity`, `crowding`.
Useful evidence signals: `entry_gap`, `follower_distance`, `person_count`,
`baseline_count`, `zone_dwell`.

Your vision-only tailgating claim is a selling point — send `entry_gap` and
`follower_distance` as evidence so the operator can see the alert was made from camera
behaviour alone, with no badge data.

### Member 2 — Predictive Loitering (`component_2`)

Emit `loitering`. Useful evidence signals: `dwell_time`, `path_reversals`,
`trajectory_entropy`, `queue_alignment`.

Your novelty is telling legitimate lingering from suspicious loitering, so **send the shape
signals, not just the timer**. `dwell_time` alone is the fixed-threshold baseline you are
arguing against; `path_reversals` and `trajectory_entropy` are the evidence that you beat
it. Seed event `evt-c2-0020` in the dashboard is deliberately a timer-only false alarm —
that is the contrast your evaluation needs.

### Member 3 — Aggression & Violence (`component_3`)

Emit `aggression`, `violence`, `concealment`, `rapid_approach`. Useful evidence signals:
`approach_velocity`, `limb_contact_events`, `torso_lean`, `occlusion_ratio`.

Please always send `occlusion_ratio`. Occlusion is a named weakness in our literature
review; showing the operator when a detection was half-hidden turns a limitation into a
contribution.

### Member 4 (me) — Fall & Abandoned Object (`component_4`)

Emitting `fall` and `abandoned_object`, plus the dashboard, the API, the audit trail, the
latency work and the operator-performance study.

---

## 5. Questions I need one answer to from the group

From the planning notes, still open. Please answer in the group chat:

1. Is the first demo on recorded clips, a webcam, or a real CCTV stream? *(The dashboard
   currently says recorded clips. If that changes, the label changes.)*
2. Are the four `incident_type` sets above right, or do you want different names?
3. Do we all agree on `new / acknowledged / false_positive / escalated / resolved` as the
   statuses, and `low / medium / high` as the priorities?
4. Who owns the backend long term — does `backend_mock/` become the real one, or does
   someone replace it?
5. Which fields can your pipeline realistically supply, and which will be `null`?
   Specifically: can you give me a **bounding box** and a **media reference**? Without them
   the operator sees evidence numbers but no picture.
6. Does anyone object to contract v1.1 as written? Say so now.

---

## 6. Message to paste into the group chat

> The shared operator dashboard is built and in the repo on `develop` — `frontend/` for the
> UI, `backend_mock/` for the event API, `docs/EVENT_CONTRACT.md` for the format.
>
> To see it: `cd frontend && npm install && cp .env.example .env && npm run dev`, then open
> localhost:5173. No Python or backend needed, it runs on demo data.
>
> What I need from each of you: make your pipeline emit the JSON format in
> `docs/EVENT_CONTRACT.md`. There is a copy-paste Python function in section 6 of that
> document — it is about ten lines in your detector loop.
>
> Quickest way to check you have it right: open the dashboard, go to the **Integration**
> tab, paste one event your detector produces. It tells you field by field what is wrong,
> then lets you push it into the live queue to see how it looks to an operator. Two minutes,
> and you do not need anything from me.
>
> Three rules that will bite you if you miss them: timestamps in **UTC** with a trailing Z;
> bounding boxes **normalised 0–1**, not pixels; `track_id` is an **anonymous label** like
> `person-12`, never a name. The API rejects all three with a message naming the field.
>
> If your detector produces an incident type I have not listed, **just send it** — it still
> renders, it is just marked "Unregistered type" until I add a label. You are not blocked
> on me for anything.
>
> Please have a look before the standup and tell me if you want the format changed. Changing
> it now is free; changing it in December is not.

---

## 7. Supervisor note on the video files

The project folder contains downloaded social-media and CCTV clips. None of them are in
this repository and none should be — `.gitignore` keeps `public/clips/` out except the
synthetic placeholders.

Two things to settle with the supervisor before any of that footage is used in the research
at all, not just before it is committed:

1. Whether downloaded social-media video can be used as research data, given that we have
   no consent from the people in it and our ethics section commits to consented or
   public-domain sources.
2. If it can, whether faces must be blurred in anything that appears in the thesis, the
   presentation or the demo.

Worth asking early. If the answer is no, the dataset plan needs to lean on the public
benchmarks (UCF-Crime, CUHK Avenue, ShanghaiTech) plus our own consented recordings, and
that is better known in October than in March.
