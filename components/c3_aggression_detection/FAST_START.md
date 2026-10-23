**Fast development path: reuse completed caches, then process a small pilot**

Full-dataset extraction is optional during infrastructure development. Keep the
original locked splits and every completed `.npz` file. If a full run is still
active in your terminal, Ctrl+C stops that run; completed archives remain saved.
An interrupted current video might need extraction later. No completed video
needs to be regenerated to create an index.

`src/quick_pose_setup.py` validates existing caches against the locked manifest,
recovers a reusable train index, and selects up to 12 short cached training
videos from distinct source groups. It performs no YOLO inference on those
training videos. A validation pilot selects four weak annotated-fight and four
weak annotated-no-fight sources, all in the existing validation split and in
distinct source groups. It processes at most eight source seconds per video:
64 seconds of new footage. The existing 15 Hz configuration is retained.

This selection is retrospective diagnostic sampling, not an unbiased validation
benchmark. It supplies no verified positive precursor labels. Reviewed or
uncertain decisions and notes are preserved; independently verified supervision
is still required for early-warning training. No model is trained by quick setup.

The 15-minute budget covers recovery and launching extraction jobs. Each child
extractor receives the remaining wall-clock budget as a timeout. A timeout stops
only that child and preserves earlier completed work. The tool does not kill
other terminals or other Python jobs. Incomplete archives are reported and left
intact. Different existing index/report content is never overwritten; choose a
new output version on a later recovery run if additional caches have appeared.

**Commands for the prepared quick pilot**

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
& $C3Python -m src.train_stgcn --train-index data/pose_cache/quick_v1/pilot_train_index.json --val-index data/pose_cache/quick_v1/pilot_val_index.json --check-only --report outputs/quick_v1/readiness.json
```

This checks the prepared pilot without extracting more video. It will report the
verified-label training blocker rather than invent labels or write a checkpoint.

The full recovered index is `data/pose_cache/quick_v1/recovered_train_index.json`;
the small training subset uses `pilot_train_index.json`. New validation archives
and their combined index are under the same quick directory. Reports are under
`outputs/quick_v1/`.

**Repeat recovery later without any new extraction**

```powershell
& $C3Python -m src.quick_pose_setup --recover-only --out-dir data/pose_cache/recovered_v2 --report-dir outputs/recovered_v2
```

**Create a fresh bounded pilot later**

```powershell
& $C3Python -m src.quick_pose_setup --minutes 15 --out-dir data/pose_cache/quick_v2 --report-dir outputs/quick_v2
```

There is no need to install CUDA or change drivers for this short CPU pilot.
GPU setup, larger extraction, threshold calibration, and full-recording event
evaluation remain separate stages. All original datasets, reviewed annotations,
detectors, and previous results remain untouched.
