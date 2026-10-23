**A three-epoch ST-GCN active-fight pilot has completed**

The existing joint-only baseline trained on the already cached pilot, using the
original UBI weak binary fight intervals as explicitly defined active-fight
supervision. No AI precursor proposal was used as a target. The checkpoint is
`weights/stgcn_weak_fight_pilot_v1.pt`. Only the active-fight head is supervised;
the precursor head is untrained. `early_prediction_claim=false` and probability
calibration remains `UNVALIDATED`. Do not interpret its precursor output as an
early-warning prediction or connect this checkpoint to operational alerts.

No extra YOLO/pose extraction was performed. The data uses the locked train and
validation groups and no test source. Of 12 indexed training and 8 indexed
validation videos, 11 and 5 respectively have eligible windows after the existing
frame/quality guards. There are 1,006 training windows (553 weak positives and
453 weak negatives) and 164 validation windows (59 positives and 105 negatives).
Windows overlap and are not independent examples. The validation selection is
the earlier retrospective eight-second diagnostic pilot, not a full benchmark.

Training used the existing implementation unchanged: seed 42, three epochs,
batch size 8, AdamW learning rate 0.0002, weight decay 0.01, 48-frame trailing
windows, and the fixed original quality guards. The lowest validation BCE selected
the saved state. This is a research checkpoint, not an operational model.

| Epoch | Training BCE | Weak-label validation BCE |
|---|---:|---:|
| 1 (selected) | 0.619632 | 1.902250 |
| 2 | 0.573880 | 2.339369 |
| 3 | 0.547973 | 2.370199 |

Validation agreement against **weak annotations**, at a fixed threshold of 0.5:

| Measure | Result |
|---|---:|
| Window agreement / accuracy | 40.24% |
| Precision | 34.65% |
| Recall | 74.58% |
| F1 | 47.31% |
| Balanced accuracy | 47.76% |
| Always-no-fight baseline agreement | 64.02% |
| Train-prevalence constant-score BCE | 0.726086 |
| Checkpoint validation BCE | 1.902250 |

The model fails both simple comparisons. Training loss decreases while validation
loss increases. At threshold 0.5, it assigns the same binary class to every eligible
window within each of the five validation sources; this does not follow the weak
annotation's changing event boundaries. High agreement on one mostly positive
clip does not establish successful event detection. These observations indicate
that more epochs alone are not a justified next experiment.

Validation negatives with usable poses come from only one weak no-fight source,
`N_534_0_1_1_0`; all 37 of its eligible windows are false positives against the weak
labels. Three other weak no-fight clips have no quality-eligible pose windows.
Earlier AI review also found ambiguous or apparently inconsistent weak event
semantics/timing in several selected fight clips. Weak annotation agreement is
not verified human accuracy. No full-recording event recall, false alarms/hour,
calibration, test performance, or early-warning lead time has been measured.

The next data work is to audit weak event semantics/timing against visible contact,
collect more distinct pose-visible non-aggressive activities and confirmed
precursor contexts, and review validation separately. Human-verified precursor
labels remain absent. The four AI precursor candidates are provisional and all
belong to train; no validation precursor candidate was established in these clips.
Existing source files, annotations, AI review decisions, caches, and previous
results must remain versioned and preserved during that work.

**What was checked**

The weak-label readiness guards passed. The saved checkpoint was strictly reloaded
and produced finite logits and state. Its reloaded BCE reproduced selected epoch
1 within numerical tolerance. Train/validation source groups do not overlap, and
zero test sources were loaded. All 127 source/review/result file hashes from the
existing preservation snapshot and 115 indexed cache archives remained unchanged.
Original and AI review parent records remain preserved. Existing source modules
were not edited in this stage; the previous full-suite result of 153 tests plus
17 subtests was not rerun. This run exercised actual fitting, checkpoint reload,
and validation on real cached data.

Reports and a reproducible evaluation script are in `outputs/weak_fight_pilot_v1/`:
`readiness.json`, `evaluation.json`, `validation_predictions.csv`,
`per_video_weak_agreement.csv`, `preservation.json`, and `completion.json`.
`readiness.json` is the pre-fit check and correctly has `model_trained=false`;
`completion.json` and the checkpoint record completed fitting. No commit or push
was performed. Weights and experiment outputs remain ignored.

**Exact Windows PowerShell reproduction commands**

Use new filenames for a fresh run so the existing checkpoint and results remain
intact. The same seeds/configuration are deterministic on the current CPU runtime;
different library/runtime versions can change numerical behavior.

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'

& $C3Python -m src.train_stgcn --task active-fight --supervision weak-fight --train-index data/pose_cache/quick_v1/pilot_train_index.json --val-index data/pose_cache/quick_v1/pilot_val_index.json --epochs 3 --batch-size 8 --out weights/stgcn_weak_fight_reproduction_v1.pt --report outputs/weak_fight_reproduction_v1/readiness.json

& $C3Python outputs/weak_fight_pilot_v1/evaluate_checkpoint.py --checkpoint weights/stgcn_weak_fight_reproduction_v1.pt --out-dir outputs/weak_fight_reproduction_v1
```

Evaluation does not fit thresholds, calibrate probabilities, alter labels, or read
test caches. It compares the exact checkpoint provenance with the loaded datasets
and rejects mismatched source artifacts. Re-evaluating the original checkpoint
without fitting uses:

```powershell
& $C3Python outputs/weak_fight_pilot_v1/evaluate_checkpoint.py --out-dir outputs/weak_fight_recheck_v1
```
