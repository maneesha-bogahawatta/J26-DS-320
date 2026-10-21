**C3 interactive local demonstration**

The existing `outputs/final_c3_v2/` comparison dashboard and `-Serve` launcher
mode remain available. The additive workbench runs the existing
`src.predict_advanced` pipeline; it does not train or replace a model.

From Windows PowerShell:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
.\RUN_C3_DEMO.ps1 -Interactive
```

Open `http://127.0.0.1:8765`. The equivalent command, without changing
PowerShell execution policy, is:

```powershell
& '..\..\.venv\Scripts\python.exe' -m src.demo_server --port 8765
```

The server binds only to IPv4 loopback. Ctrl+C stops new requests and waits for
an existing inference job to finish. Run a second server with a different port
and `--work-dir` if needed; do not run two servers over the same work directory.

1. Choose an existing cached validation context, or upload a new CCTV video.
   A file-picker upload stays on this computer. Upload limit is 256 MB by default
   (`--max-upload-mb` can set 1–2048 MB). Known locked test identities and exact
   test-source SHA-256 duplicates are blocked without decoding test media. This
   protection does not identify arbitrary cropped or re-encoded test footage.
2. Choose a camera ID and 1–60 seconds for a new source; eight seconds is the
   default. Cached contexts use their full existing extracted range. Keep
   research notifications disabled for the normal demonstration.
3. Run inference. A single background worker processes jobs, with at most four
   queued/running jobs. CPU pose extraction may be slower than the source video.
4. View the annotated video, original-video score timeline, usable-pose coverage,
   candidate incidents, evidence signals and reliability. Click the timeline or
   an incident's **Seek evidence** button to move the video.
5. Enter an operator ID and append an incident decision. All earlier decisions
   remain in the review history. An operator incident decision does not supply
   independent precursor or first-contact ground truth.
6. Export a ZIP with result files and operator review history. Original uploaded
   video and dataset media are excluded; annotated result video is included.

The preserved comparison dashboard is linked at `/report/`. Weak onset and
active-fight scores are uncalibrated model scores, not validated probabilities.
Candidate incidents are research records. Operational notifications stay disabled.
Opt-in research notifications can only use internally supported model heads, and
their confirmation/suppression fields remain visible. A score below a threshold,
no incident, or pose abstention never establishes that a scene is safe.

**State and provenance**

New workbench state is under `outputs/demo_workbench_v1/`:

- `uploads/<source SHA-256>/`: original uploaded bytes and source metadata.
  Identical upload content reuses the same source path and compatible pose cache.
- `jobs/<UUID>/job.json`: job request, camera, timestamps and status.
- `jobs/<UUID>/result/`: new output directory for the existing inference pipeline.
- `jobs/<UUID>/inference.log`: subprocess diagnostic output.
- `jobs/<UUID>/incident_reviews.jsonl`: append-only operator review history.

Jobs interrupted by server restart are marked `interrupted`; submit a new job to
retry. Failed jobs retain their status and log. No old run, source annotation,
dataset split or existing experiment is overwritten. Uploads are for a local
demonstration, not a training-data ingestion process.

**Local API**

`GET /api/bootstrap` returns a fresh server token, eligible sources and job
history. Mutations require `X-C3-Token`, a trusted loopback Host and, when present,
a matching loopback Origin. No cross-origin access is enabled.

| Endpoint | Purpose |
|---|---|
| `POST /api/uploads` | Raw video bytes; original basename in percent-encoded `X-C3-File-Name` |
| `POST /api/jobs` | JSON `source_id`, `seconds`, `camera_id`, `research_alerts` |
| `GET /api/jobs/<id>` | Status, prediction provenance, score trace, incidents and operator history |
| `POST /api/jobs/<id>/reviews` | JSON `event_id`, `status`, `reviewer`, optional `note` |
| `GET /api/jobs/<id>/export.zip` | Completed result and append-only review export |
| `GET /files/<id>/<file>` | Result artifacts; video supports a single HTTP byte range |

Review statuses are `pending`, `acknowledged`, `confirmed`, `dismissed` and
`uncertain`. Each decision has a new review UUID and UTC timestamp, with
`eligible_for_training=false` and `verified_precursor_annotation=false`.

**Verification**

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& '..\..\.venv\Scripts\python.exe' -m unittest tests.test_demo_server -v
```

The tests exercise real loopback HTTP routes with fixture inference, upload
bounds, cached-source filtering, locked test duplicate rejection, same-origin
mutation controls, traversal prevention, job errors/restart state, video ranges,
append-only reviews, exports and preservation of incident source records. Fixture
inference tests do not establish model accuracy. Real cached inference and browser
verification should be reported separately in the final completion evidence.
