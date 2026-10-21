The existing model investigation is preserved in [STGCN_DIAGNOSIS.md](STGCN_DIAGNOSIS.md). It is newer than the earlier 164-test completion report: its full suite passed 175 tests plus 17 subtests. The six predefined weak-label fits, original comparisons, quality-matched comparisons and source-bootstrap intervals already exist. This continuation reuses those results and cached prediction tables; it does not repeat extraction or claim that another fit fixes the scientific blocker.

The three tasks remain separate:

| Task | Target | Current evidence |
|---|---|---|
| Weak first-onset forecasting | Original first annotated UBI onset in the next three seconds | Trained models and exploratory source-separated weak-label diagnostics |
| Active-fight recognition | Inside an original weak fight interval | Trained models and separate classification metrics |
| Verified pre-contact aggression | Independently reviewed threatening cue before independently confirmed first physical contact | Training and accuracy claims blocked: zero independently verified precursor-positive videos |

An annotated fight onset is not automatically the first physical contact. No heuristic v3/v4 output, provisional AI review or later event is promoted into a verified precursor target. Current and past frames form the input; future annotation times construct only offline weak targets.

The training/data audit found no demonstrated global positive-class inversion: sigmoid of head h's logit means the score for target h equal to one. Original binary annotations, target masks and independently computed AUROC agree. The advanced onset score has development AUROC 0.601 and external-validation AUROC 0.232. Complementing it helps external validation retrospectively while reducing development AUROC to 0.399; selecting such a correction from validation would be invalid. Source groups are disjoint across fit/development/validation, and test sources remain excluded.

The older video-balanced sampler had an effective prevalence inconsistent with the raw-count class weights. The existing diagnosis already compares uniform shuffled fitting windows with consistent fit-only class weights and unweighted losses. It also identifies a current-observation mismatch: historical training windows could have usable history despite absent current poses. Its separate quality-matched archive preserves labels while applying the runtime observation rule. No favorable subset is substituted for the original cohort.

| Comparison on unchanged original validation windows | Onset AUROC | Active-fight AUROC | Interpretation |
|---|---:|---:|---|
| Historical baseline | 0.608 | 0.534 | Current research runtime candidate |
| Historical advanced | 0.232 | 0.666 | Advanced onset generalization fails |
| Existing diagnosis baseline, unweighted | 0.626 | 0.361 | Small onset-ranking gain, active regression |
| Existing diagnosis advanced, unweighted | 0.377 | 0.555 | Onset remains below chance, active regression |

The quality-aligned baseline in the separate matched cohort has onset AUROC 0.687 and F1 0.544, versus historical baseline 0.591 and 0.433 on the same retained windows. Its false-positive rate is still 50%, and source-cluster 95% intervals include no improvement. The original-cohort unweighted baseline's maximum-F1 development threshold predicts all validation onset windows positive. These results do not justify changing the supported runtime checkpoint or alert eligibility. More complete confusion matrices, threshold policies, source counts and uncertainty are in `outputs/diagnosis_v1/comparison_metrics.csv` and `paired_improvements.json`.

The remaining calibration gap is now measured by `src/reliability_audit.py`. It reads the existing fit/dev/val prediction table, rejects test/unknown or wrong-role sources and source-group overlap, honors target masks, and reports Brier score, negative log likelihood, ten fixed reliability bins, ECE and a constant development-prevalence reference. It fits a positive temperature separately per head on internal development only, using a frozen deterministic grid from 0.05 to 20. Positive scaling cannot reverse score direction or repair poor ranking. Identity scaling is included in the grid. The `.5` decisions are checked for invariance.

All transformed results are isolated diagnostics. Runtime checkpoints, thresholds, calibration status and default alert policy are unchanged. Development previously selected model epochs and thresholds, so its calibration statistics are optimistic; external validation was historically inspected and remains exploratory. Scores should continue to be described as uncalibrated weak-model scores, not probabilities that violence will occur.

| Existing model | Head | Dev temperature | Validation Brier raw → diagnostic | Validation ECE raw → diagnostic |
|---|---|---:|---:|---:|
| Baseline | Weak onset | 2.999469 | 0.220845 → 0.227378 | 0.216696 → 0.162937 |
| Baseline | Active fight | 2.714418 | 0.254121 → 0.240943 | 0.281145 → 0.245755 |
| Advanced | Weak onset | 2.011748 | 0.240421 → 0.242363 | 0.234618 → 0.153762 |
| Advanced | Active fight | 20.000000 | 0.229811 → 0.248810 | 0.252968 → 0.271341 |

Lower is better for Brier and ECE. Onset ECE improves while Brier worsens in both models. Advanced active-fight temperature reaches the grid's upper boundary and external reliability regresses. Baseline active-fight calibration improves these descriptive measures, but remains worse than the development-prevalence reference Brier of 0.178770 on this validation cohort. There is no evidence here supporting calibrated operational probabilities. Source/prevalence shift, weak labels, narrow source coverage and correlated windows remain material limitations.

New outputs are under `outputs/development_completion_v1/model_reliability/`:

- `reliability.json`: both heads, all three roles, masks, source counts, bins, calibration policy and input hashes.
- `reliability_metrics.csv`: raw and diagnostic scores compared using the same windows.
- `reliability_bins.csv`: all populated and empty bins with observed weak-target fractions.
- `comparison.md`: generated summary table.
- `tests.xml`: the targeted six-test run, zero failures.

The six tests cover exact Brier/NLL/bin arithmetic, score endpoints and empty cohorts, invalid inputs, positive temperature direction, rejection of validation/test fitting, source-role/leakage checks, masks, deterministic reruns and invariance of selected temperature to changed validation outcomes. The actual report command below used cached predictions and completed without source-video extraction or additional dependency installation.

Run from Component 03. Use a new output directory for a different audit configuration; existing differing artifacts are never overwritten:

```powershell
Set-Location 'C:\Users\Kulanjana\Downloads\J26-DS-320\components\c3_aggression_detection'
$C3Python = (Resolve-Path '..\..\.venv\Scripts\python.exe').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
& $C3Python -m src.reliability_audit --out-dir outputs/development_completion_v1/model_reliability

$C3Root = (Get-Location).Path
$env:PYTHONPATH = "$C3Root\data\phase1_work\testdeps;$C3Root;$C3Root\legacy"
& $C3Python -m pytest tests/test_reliability_audit.py -q -p no:cacheprovider --basetemp data/phase1_work/pytest_reliability_recheck --junitxml outputs/development_completion_v1/model_reliability/tests_recheck.xml
```

The next scientific step is representative independent cue/contact annotation in locked training and validation groups, followed by a prospectively frozen source-group protocol. Use training for fitting and development/validation for selection; retain the locked test for a separate final evaluation. Confirmed early-warning accuracy, full-recording false alerts per camera-hour and operational real-time performance remain unproven.
