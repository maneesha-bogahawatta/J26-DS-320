# Original team dashboard + Component 03

Completed locally on 21 October 2026 (Asia/Colombo).

- Frontend: `dashboard/frontend/` — original React/TypeScript/Vite application.
- Backend: `dashboard/backend/` — shared FastAPI event API, C3 jobs and optional team detector registry.
- Inference: `components/c3_aggression_detection/` — existing DemoWorkbench and predict_advanced.
- Prepared dashboard: http://127.0.0.1:8000/

The launcher uses only the original application under `dashboard/`.

## C3-only backend cleanup (user-authorized, 21 October 2026)

`dashboard/backend/c4_core/`, `dashboard/backend/detectors/` and the C4-only
`requirements_c4.txt` are not dependencies of C3 inference. They were removed
from this local project and archived with seven SHA-256-verified files at:

`C:/Users/Kulanjana/Downloads/J26-DS-320-unused-c4-backend-20261021-223454/`

This removes local C4 fall/object execution. C3 jobs, playback/export and the
shared event POST/PATCH API remain. The optional plugin loader already supports
an absent `detectors/` directory; a fresh backend process has an empty team detector list.
The running server still has the old C4 functions loaded until a manual restart:
automatic process stop/restart was blocked by approval policy. C3 does not
support single-frame live inference. The actual `components/c4_critical_incidents/`
member folder and shared schemas were not changed. C4 preservation claims in the
historical restoration sections below describe the state before this cleanup.

## Obsolete dashboard cleanup (user-authorized, 21 October 2026)

Repository-root `frontend/` and `backend_mock/` were removed from the repository
by moving them intact to this external archive:

`C:/Users/Kulanjana/Downloads/J26-DS-320-obsolete-dashboard-20261021-190438/`

All 814 archived files were verified by SHA-256. No permanent deletion was used.
The archive includes pre-cleanup root npm/README files and its own manifest.
The original `dashboard/`, C3 data/models and running services remain. Root
`frontend:*` npm shortcuts target `dashboard/frontend`; `frontend:dev` selects
the C3 configuration. Evidence is in `outputs/dashboard_cleanup_v1/`.
Preservation notes below describe the earlier restore stage, before this cleanup.

## Recorded-video correction (21 October 2026)

The original Recorded video upload initially invoked only registered C4 models,
while C3 required its separate research tab. This caused an uploaded violence clip
to show an empty result without running C3. In the C3-enabled configuration,
ordinary recorded uploads now run both the existing team endpoint and the existing
C3 asynchronous upload/job workflow. C3 checks up to the first 60 seconds; the
actual processed range and usable-pose decisions are shown in the original results.
The result includes model scores, pose coverage, annotated-video links and export
in the same Recorded video screen. Single-frame/live C3 remains unsupported.

Empty results no longer show the green “Nothing unusual found” message. Failures,
missing poses, no registered models, and no returned candidates have distinct
wording. Raw per-head peaks, reference thresholds and checkpoint candidate policy
are visible. Model weights, thresholds and candidate policy were not changed.

The reported F_9 review clip completed with 35/40 usable pose decisions, peak
onset score 0.519887 and peak fight score 0.550248. Its checkpoint reference
thresholds are 0.603839/0.611068 and active-fight candidate generation is disabled
by the existing reliability policy. It still returns zero candidates; this is a
remaining model limitation, not evidence that the clip contains no violence.
The routing fix does not establish improved model accuracy.

Targeted frontend tests: 13 passed (including routing, failed/partial checks,
abstention, cancellation, source/render coordinates and head diagnostics).
C3 production build passed; lint retains the six existing warning sites.
Evidence and pre-edit snapshots: `outputs/recorded_c3_fix_v1/`.

## 1. Restored files

Fetched origin and verified `origin/develop`, commit `ceb8c70` (full SHA in manifest). Backed up and then restored all **81 upstream dashboard files** with the requested `git restore --source origin/develop --worktree -- dashboard`. All restored files matched upstream Git blobs after accounting for CRLF conversion. Pre-restore local files had no content differences beyond line endings, and no extra dashboard files.

The complete restored-file list, pre-restore tracked/untracked inventory, hashes and upstream SHA are in `outputs/original_dashboard_integration_v1/restore_backup_manifest.json`. Restoration verification is in `restore_verification.json`.

This local branch is `member3/early-aggression-detection`. Its older HEAD tracks only the dashboard package initializer files, so most restored dashboard files remain **untracked** locally. Worktree restoration did not stage files or merge develop.

## 2. Backup, quarantine and duplicates

Verified external backup:

`C:/Users/Kulanjana/Downloads/J26-DS-320-dashboard-backup-20261021-175431/`

It contains all 81 pre-restore dashboard files, a SHA-256 manifest, original Git status and overwrite comparison. Every copied file was verified before restore.

Two seven-byte `fixture` uploads created by integration tests were moved into that backup's `quarantine/generated-test-uploads/`. Their paths/hashes are recorded in `quarantine_manifest.json`. Tests now put these uploads in temporary fixture directories.

No obsolete generated dashboard files existed before restoration. At that stage the root frontend/backend copies were retained unchanged. They were subsequently archived under the cleanup authorization above; all original bytes remain recoverable. No uncertain teammate files, source videos, models, old caches or annotations were deleted.

## 3. Backend

The original backend has no database or authentication service: it is a local prototype using a locked in-memory store with optional JSON persistence.

Changes to the existing application:

- `main.py`: app factory, additive C3 lifecycle/refresh, original event endpoints and original detector loading retained; serves the original frontend build and uploaded-video media.
- `store.py`: backwards-compatible seed/persistence-path options, atomic JSON replacement and camera upsert. Connected C3 mode starts with an empty persistent store inside C3 outputs.
- New `c3_api.py`: queued upload/inference/results/media/export routes, loopback/origin checks and C3 mutation token.
- New `c3_bridge.py`: uses existing C3 Python methods, validates event projections into the original shared store, and appends non-training operator reviews.
- New backend integration tests and isolated API dependency lock.

At the original restoration stage, `schemas.py`, `detect.py`, `detectors/c4.py`,
`c4_core/` and seed events were preserved. The later C3-only cleanup above removes
the C4 plugin and its core from this local installation. C1/C2 have no registered
inference modules in this upstream dashboard; shared event transport remains available.

Existing C1/C2/C4 event POSTs, filters, PATCHes and audits still work. Set `C3_DASHBOARD_ENABLED=0` before a direct uvicorn launch for the original seeded prototype mode; use a frontend configuration with C3 disabled alongside it.

## 4. Frontend

The original Check video / Alerts / Shift log / Project team navigation and design remain. Existing recorded-video and live-camera screens are preserved.

The user requested the same navigation as the other team members. The separate
**C3 research video** tab, its home shortcut and `C3ResearchPanel.tsx` have been
removed. The Check screen offers only Recorded video and Live camera. Ordinary
recorded uploads still use the existing queued C3 workflow and show pose coverage,
head diagnostics, annotated-video links and result exports. Candidates use the
original Alerts/review/Shift log workflow. Standalone research history and cached
source tools remain available through `RUN_C3_DEMO.ps1 -Interactive`.

C3 wording distinguishes a research score from probability and review from ground truth. Other components retain their original decision wording. Retained modules include `c3Client.ts`, `C3ModelStatus.tsx` and `c3Presentation.ts`. Styling uses existing tokens; no replacement framework/chart library was added. Pre-removal source snapshots and verification are in `outputs/onboarding_alignment_v1/`.

`.env.c3`, `dev:c3` and `build:c3` select same-origin HTTP with C3 enabled. Ordinary builds retain the original practice defaults; `.env.c4` and the upstream dependency lockfile remain unchanged. Final `dist/` uses `build:c3`.

## 5. C3 integration and contract

```text
original dashboard frontend
 -> original FastAPI + additive C3 routes
 -> existing DemoWorkbench serial worker
 -> existing predict_advanced subprocess / pose cache / guarded ST-GCN
 -> immutable C3 artifacts
 -> contract adapter -> original shared EventStore -> Alerts/review/export
```

No ML pipeline was copied into the backend. One backend process owns the workbench and starts existing Python inference subprocesses; no separate competing dashboard is needed.

| Route | Purpose |
|---|---|
| GET /api/v1/c3/bootstrap | Sources, history, capabilities, current token |
| POST /api/v1/c3/uploads | Raw video bytes; basename in X-C3-File-Name |
| POST /api/v1/c3/jobs | Queued temporal inference |
| GET /api/v1/c3/jobs/{id} | Status, scores, provenance, shared events |
| GET /api/v1/c3/jobs/{id}/export.zip | Results and review history |
| GET /api/v1/c3/media/{id}/{file} | Artifacts with video byte-range support |
| Existing /api/v1/events GET/POST/PATCH | All-component queue and review audit |

C3 mutations require X-C3-Token from bootstrap. The launcher binds IPv4 loopback only. Existing team video/JPEG routes remain independent; C3 is unavailable on a single JPEG.

The actual team contract is v1.1 in `dashboard/backend/schemas.py`, mirrored by `dashboard/frontend/src/types/events.ts` and documented in `docs/EVENT_CONTRACT.md`. It already supports the open incident_type string and suggested aggression/violence types. **No shared-contract change was required or made.**

| C3 | Existing contract |
|---|---|
| C3 / potential_aggression / active_violence | component_3 / aggression / violence, cautious research summaries |
| Raw model score | confidence; score_calibrated=false |
| Local incident UUID | Job-scoped event_id; original UUID in evidence |
| Anonymous IDs | track_id and related_tracks; co-observation does not prove participation |
| Source pixels | Normalized rendered-media box including the 116px score footer |
| Source time/provenance/pose quality | Existing evidence signals |
| Model version | Existing model fields |
| Review | Existing status/audit plus append-only C3 non-training review |

Media offsets subtract context start; original source time remains in evidence. Missing render geometry yields no overlay. Actual scene occlusion is unavailable and remains null; missing joints are not substituted for occlusion.

The older `shared/schemas/event_schema.py` dataclass has a narrower, different type set and remains unchanged. Its generic forecast restriction still applies.

Reviews never modify precursor/contact labels or incident source records. Shared JSON and C3 review JSONL are separate writes, not a transactional database. Exports exclude original source videos and include annotated result media.

## 6. Verification

Evidence: `outputs/original_dashboard_integration_v1/`.

- Full C3 + original-backend suite: **225 tests and 62 subtests passed**, no failures/skips, 16.54s (`tests_full.xml`).
- Final backend check after isolating test uploads: **11 tests and 3 subtests passed** (`backend_tests.xml`). Included in the full suite count, not additional distinct coverage.
- Original unchanged backend HTTP smoke: **18 passed, 0 failed** against the running backend. Its C4 event is a smoke_test_detector fixture, not model accuracy evidence.
- Frontend presentation tests: **4 passed**. Original default build and C3 build both passed.
- Lint: **0 errors, 6 existing React warnings** at upstream warning sites.
- One test dependency warning: deprecated Starlette/httpx transport; tests pass.

Actual browser checks:

| Run | Result |
|---|---|
| Cached validation f_147_1_0_1_0 | 2 research events, 37/40 usable decisions, cache reused, extraction 0s |
| Synthetic clip_01.mp4 upload, 1s | Fresh extraction 4.375s, 0/4 usable poses, abstained, no candidates |
| Playback | 640x476; 8s cached / 1s upload; readyState 4, no video error |
| Seek | Cached evidence at annotated offset 4.8s |
| Review | Acknowledged; persisted in Alerts, Shift log, shared audit and non-training C3 review |
| Exports | Original CSV and C3 ZIP downloaded; ZIP content verified |
| Media | HTTP 206 ranges passed for both results |
| Console | No browser warnings/errors observed |
| Existing screens | Original recorded-video and live-camera controls remain available |

Cached job: `0f790ddc2c5843c0870667f22349a696`.
Upload job: `1e3bc75be79f466c890e9dfdfc2a9188`.
See `browser_smoke/verification.json`, exact job API responses and screenshot. These are integration/abstention checks, not independent model evaluation.

Root frontend hashes, C1/C2/C4/shared source, protected model/foundation files, original C4 dashboard code/schema/types/media and preserved scientific state were checked. New uploads/results/cache entries use C3-owned paths; no old feature cache/model was replaced.

## 7. Exact PowerShell commands

The prepared local application needs one command for all required services:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
.\components\c3_aggression_detection\RUN_C3_DASHBOARD.ps1
```

Open http://127.0.0.1:8000/. Ctrl+C stops it. The current demonstration already uses port 8000. After stopping it, reopen the same QA history with:

```powershell
.\components\c3_aggression_detection\RUN_C3_DASHBOARD.ps1 -StateDirectory 'outputs/original_dashboard_integration_v1/browser_smoke'
```

Equivalent Python command if PowerShell script execution is restricted:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
& '..\..\.venv\Scripts\python.exe' -X utf8 -B -m src.serve_dashboard --port 8000
```

Do not run two workbenches over the same state directory. A different port also needs a distinct state directory if another process remains running.

Build/test using ordinary trusted Node/npm:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\dashboard\frontend'
npm ci
npm run test
npm run build:c3
npm run lint
```

For HMR: `npm run dev:c3` with the original backend on port 8000. Vite proxies /api and /uploads. Do not start the standalone C3 workbench for this integration.

This machine used its existing bundled Node/npm and trusted HTTPS mirror because npmjs TLS interception was not trusted. Certificate verification remained enabled; root frontend was never used:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
$C3Node = 'C:\Users\Kulanjana\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$C3Npm = (Resolve-Path 'components/c3_aggression_detection/data/dashboard_node_tools_v1/package/bin/npm-cli.js').Path
$env:Path = (Split-Path $C3Node) + ';' + $env:Path
Set-Location dashboard/frontend
& $C3Node $C3Npm ci --registry=https://registry.yarnpkg.com/ --replace-registry-host=always --cache '../../components/c3_aggression_detection/data/dashboard_npm_cache_v1' --ignore-scripts --audit=false --fund=false
& $C3Node $C3Npm run build:c3
```

API dependencies are already installed into a C3-only target. Fresh setup command:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
& '.\.venv\Scripts\python.exe' -m pip install --target components/c3_aggression_detection/data/dashboard_deps_v1 -r dashboard/backend/requirements-c3.lock.txt
```

The existing C3 scientific environment/checkpoints/video tools are also required; this command installs API dependencies only.

Regression commands with the existing isolated test dependencies:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320'
$env:PYTHONPATH = 'components/c3_aggression_detection/data/phase1_work/testdeps;components/c3_aggression_detection/data/dashboard_deps_v1;dashboard/backend;components/c3_aggression_detection'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\.venv\Scripts\python.exe' -X utf8 -B -m pytest components/c3_aggression_detection/tests dashboard/backend/test_c3_integration.py -q
& '.\.venv\Scripts\python.exe' -X utf8 -B dashboard/backend/smoke_test.py
```

The HTTP smoke test needs a running backend and appends a clearly named fixture event.

## 8. Demonstration steps

1. Open the original dashboard and select **Component 03 research — check a recorded video**.
2. Select cached f_147_1_0_1_0, enter camera ID and run inference. This is historically inspected validation material, not prospective test footage.
3. Inspect annotated video, two raw score traces, gaps and pose coverage.
4. Seek evidence, inspect tracks, open the candidate in Alerts and record a shift label/note.
5. Acknowledge/dismiss, inspect Shift log, export CSV and the run ZIP.
6. Upload a new recorded video for a fresh extraction check. New-video duration is 1–60s from its beginning; CPU extraction may be slower than playback. Known locked test IDs and byte-identical test uploads are blocked.
7. Ordinary Recorded video uploads now include C3 in C3-enabled builds. Live camera remains available for the original registered frame detectors; C3 needs temporal video jobs.

## 9. Limits and Git status

C3 weights, thresholds and calibration are unchanged. Checkpoint SHA-256:
`954559460c10b85c25b532b5f7aa7d28edb141f38b308ebaeb199aca6c4d7820`.

Weak supervision remains. Verified 1–3s pre-contact prediction, calibrated probabilities and operational reliability are not established. Operational notifications remain disabled. Abstention or an empty list never proves safety.

Original C4 source/registry were preserved and transport-tested, but real C4 fall-model accuracy was not checked: its default ~/c4_weights directory is absent. C1/C2 have no registered inference modules here; their shared transport was tested. Physical webcam streaming/permissions were not exercised. No production authentication, transactional DB, public deployment or external alert delivery was added.

See the stage-specific Git status, preservation and release manifests for inventories. Existing C3 changes and restored dashboard files remain local, mostly untracked on the older branch. Nothing was staged, committed or pushed. Root duplicates are now recoverable from the authorized external archive; `outputs/dashboard_cleanup_v1/` records the latest cleanup state.
