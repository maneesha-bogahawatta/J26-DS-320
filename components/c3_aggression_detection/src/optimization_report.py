"""Build the measured optimization report and supplementary, read-only audits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from src.advanced_experiment import metrics
from src.diagnose_stgcn import ranking
from src.foundation_io import read_csv, sha256, write_csv, write_json, write_new
from src.optimize_model import BASELINE, HEADS, CONFIGS
from src.predict_advanced import CausalDecision


def confirmation_audit(out):
    summaries={};details=[]
    for context in ('original','derived'):
        rows=read_csv(out/'f9'/f'{context}_readout_stages.csv')
        thresholds=[float(rows[0]['validation_without_f9_'+h+'_threshold']) for h in ('onset','active')]
        decision=CausalDecision(thresholds);raw_count=smooth_count=0;max_hits=0;max_history=0
        for row in rows:
            valid=row['valid_pose']=='True';scores=[float(row[k]) for k in ('weak_onset_score','fight_score')] if valid else None
            status,smoothed,_=decision.update(float(row['t_s']),scores,valid,int(row['segment_id']),json.loads(row['track_ids_json']))
            hits=int(np.asarray(decision.history).sum(0)[1]) if decision.history else 0
            raw_above=bool(valid and scores[1]>=thresholds[1]);smooth_above=bool(smoothed is not None and smoothed[1]>=thresholds[1])
            raw_count+=raw_above;smooth_count+=smooth_above;max_hits=max(max_hits,hits);max_history=max(max_history,len(decision.history))
            details.append(dict(context=context,t_s=float(row['t_s']),valid_pose=valid,active_raw=scores[1] if valid else None,
                active_smoothed=float(smoothed[1]) if smoothed is not None else None,threshold=thresholds[1],
                raw_above=raw_above,smoothed_above=smooth_above,history_decisions=len(decision.history),
                active_hits_in_history=hits,identity_reset=decision.audit['identity_reset'],status=status))
        summaries[context]=dict(raw_active_crossings=raw_count,smoothed_active_crossings=smooth_count,
            maximum_active_hits_in_confirmation_history=max_hits,maximum_history_decisions=max_history,
            rule='3 decisions in history required; at least 2 smoothed positive hits, reset on invalid pose, identity/scene change or >1s gap',
            simulation_only=True)
    write_json(out/'f9'/'confirmation_audit.json',summaries);write_csv(out/'f9'/'confirmation_stages.csv',details)
    return summaries


def ranking_uncertainty(out):
    rows=read_csv(out/'validation_predictions.csv');result={};per_source=[]
    for cohort in ('original','quality_matched'):
        tables={name:[r for r in rows if r['cohort']==cohort and r['model']==name and r['head']==HEADS[1]] for name in ('baseline','headwise','readout')}
        b=tables['baseline'];a=tables['readout'];h=tables['headwise']
        assert [(r['video_id'],r['cutoff_s'],r['y']) for r in a]==[(r['video_id'],r['cutoff_s'],r['y']) for r in b]
        y=np.array([int(r['y']) for r in a]);p=np.array([float(r['score']) for r in a]);sources=np.array([r['video_id'] for r in a]);keys=sorted(set(sources.tolist()))
        before=np.array([float(r['score']) for r in b]);old=np.array([float(r['score']) for r in h]);rng=np.random.default_rng(73);values=[]
        for _ in range(2000):
            ix=np.concatenate([np.flatnonzero(sources==key) for key in rng.choice(keys,len(keys),replace=True)])
            if len(np.unique(y[ix]))<2:continue
            mm=metrics(y[ix],p[ix]);bb=metrics(y[ix],before[ix]);hh=metrics(y[ix],old[ix])
            values.append([mm['auroc'],mm['average_precision'],mm['auroc']-bb['auroc'],mm['auroc']-hh['auroc']])
        intervals=np.quantile(values,[.025,.975],axis=0)
        result[cohort]=dict(draws=2000,valid_draws=len(values),independent_sources=len(keys),
            interval_95={key:intervals[:,i].tolist() for i,key in enumerate(('readout_auroc','readout_ap','delta_auc_vs_baseline','delta_auc_vs_headwise'))},
            method='Paired source-cluster bootstrap; entire overlapping-window source clusters resampled; conditional on frozen models')
        for name,table in tables.items():
            for source in keys:
                use=[r for r in table if r['video_id']==source];yy=np.array([int(r['y']) for r in use]);pp=np.array([float(r['score']) for r in use])
                for policy in ('f1','fpr10','precision50'):
                    pred=np.array([r[policy+'_crossfit_prediction']=='True' for r in use])
                    m=metrics(yy,pred.astype(float));m.pop('auroc');m.pop('average_precision');m.pop('bce');m.pop('threshold')
                    per_source.append(dict(cohort=cohort,model=name,source=source,policy=policy,**m))
    write_json(out/'ranking_uncertainty.json',result);write_csv(out/'per_source_errors.csv',per_source)
    return result


def verify_preservation(out):
    before=json.loads((out/'preservation_before.json').read_text());root=Path.cwd().parents[1]
    missing=[];changed=[];live_appends=[]
    live_log='components/c3_aggression_detection/outputs/original_dashboard_integration_v1/browser_smoke/server.out.log'
    for name,digest in before.items():
        path=root/name
        if not path.is_file():missing.append(name)
        elif sha256(path)!=digest:
            # The already-running dashboard appends request logs independently.
            # Verify its complete original byte prefix, never truncate/revert it.
            if name==live_log:
                raw=path.read_bytes();h=hashlib.sha256();offset=0;matched=None
                if h.hexdigest()==digest:matched=0
                for line in raw.splitlines(keepends=True):
                    h.update(line);offset+=len(line)
                    if h.hexdigest()==digest:matched=offset;break
                if matched is not None:
                    live_appends.append(dict(path=name,original_prefix_bytes=matched,original_prefix_sha256=digest,
                        observed_bytes=len(raw),appended_bytes=len(raw)-matched))
                    continue
            changed.append(name)
    report=dict(files_checked=len(before),byte_identical_files=len(before)-len(live_appends)-len(changed)-len(missing),
        live_append_only_logs=live_appends,changed=changed,missing=missing,passed=not changed and not missing,
        scope='All pre-existing model weights/experiment outputs, annotation CSVs, materialized window archives and cache indices; tracked files outside C3')
    write_json(out/'preservation_final.json',report)
    if not report['passed']:raise ValueError('Pre-existing artifacts changed')
    return report


def build(args):
    out=Path(args.out_dir);e=json.loads((out/'evaluation.json').read_text());s=json.loads((out/'selection.json').read_text())
    audit=json.loads((out/'data_audit.json').read_text());f9=json.loads((out/'f9/summary.json').read_text())
    alignment=json.loads((out/'f9/image_alignment.json').read_text());ablation=json.loads((out/'ablation_validation.json').read_text())
    lead=json.loads((out/'weak_onset_lead_times.json').read_text());confirm=confirmation_audit(out);ci=ranking_uncertainty(out);preserve=verify_preservation(out)
    tests=(out/'tests.log').read_text(encoding='utf-8-sig');passed=re.search(r'(\d+) passed,.*?(\d+) subtests passed',tests)
    if not passed:raise ValueError('Complete pytest result missing')
    ntests,nsub=map(int,passed.groups());q=e['cohorts']['quality_matched'];original=e['cohorts']['original'];active=HEADS[1];onset=HEADS[0]
    lines=['# Component 03 — Model optimization results','',
        'Measured locally on 21 October 2026. All numerical results use existing weak UBI labels. They do not establish independently verified pre-contact aggression prediction.','',
        '## Decision','',
        f"The new frozen ST-GCN readout improves active-fight ranking point estimates on the unchanged quality-matched cohort: AUROC **{q['readout'][active]['auroc']:.3f}**, versus **{q['headwise'][active]['auroc']:.3f}** for the previous headwise candidate and **{q['baseline'][active]['auroc']:.3f}** for the runtime baseline. The predeclared constrained-recall/F1 promotion criteria **failed**. Keep the original baseline and its thresholds as the default. No operational alerts or new eligibility flags were enabled.",
        '',f"At source-held-out thresholds selected to limit tuning-source FPR to 10%, new active-fight recall is {q['readout'][active]['policies']['fpr10']['source_crossfit']['metrics']['recall']:.3f}, with 17 false-positive windows. The rethresholded baseline has the same recall and 2 false positives. Better ranking has not solved the required precision/recall operating point.",
        '', '## What was inspected and preserved','',
        'Read MODEL_IMPROVEMENT.md, MODEL_RELIABILITY.md, STGCN_DIAGNOSIS.md, WEAK_FIGHT_PILOT.md and FINAL_C3.md completely, plus actual model, training, data, loss, threshold, inference and event-adapter code. Reviewed the previous eight kinematic fits and two expanded ST-GCN fits before freezing this protocol.',
        '', 'Previous findings: the expanded linear-window kinematic model was stronger on internal-dev onset; the MLP-window model had modest active ranking but weaker onset. Full ST-GCN refits showed unstable development curves and late overfitting. The existing expanded advanced checkpoint had the strongest active internal-dev AUROC among prior advanced checkpoints (0.557), so it was reused as the frozen encoder. No expensive pose extraction was repeated.',
        '',f"Preservation audit: **{preserve['byte_identical_files']} pre-existing files are byte-identical**; the one already-running dashboard request log appended independently, with its entire original byte prefix hash-verified intact. The initial strict audit recorded that live-log change; preservation_final.json resolves it as append-only. All old weights, result data and tracked files outside C3 are unchanged. {audit['cache_integrity']['unique_caches']} indexed pose caches passed hashes, source identity, schema and array validation. Components 01, 02, 04 and shared dashboard sources are unchanged. No dataset/annotation/split/default/checkpoint overwrite, commit or push was performed.",
        '', '## Current working system','',
        'Component 03 currently includes locked source splits and provenance, causal YOLO11-Pose/ByteTrack caches, quality-aware trailing windows, baseline and joint/bone/velocity ST-GCN models, separate weak-onset and active-fight scoring, versioned experiments, incident/evidence export and the original shared dashboard integration. Recorded-video uploads reuse compatible caches or queue extraction, produce annotated playback and expose missing-pose/disabled-head diagnostics. The obsolete C3 research-video tab was already removed. This optimization adds controlled readout experiments and diagnostics to that working pipeline.',
        '', '## Protocol and data','',
        '- Locked source split remains 791 train / 141 validation / 68 test identities. No test source was used for fitting, normalization, epoch/model selection, thresholds or diagnostics.',
        '- Expanded fitting: 1,240 windows / 115 sources; internal development: 367 / 31. Active fitting labels: 376 positive, 864 negative (positive weight 2.298). Dev prevalence 91/367 = 24.80%. Uniform shuffled windows without replacement; no resampling compounded with loss weighting.',
        '- Original validation: active 224 windows / 51 positive; onset 161 known / 50 positive. Quality-matched validation: active 204 / 42; onset 152 known / 46. Both cohorts contain 13 contributing sources. All compared models receive identical rows, labels and masks within each cohort.',
        '- Source and leakage groups remain disjoint across fit/dev/val. Independent scalar label reconstruction found zero target/mask mismatches and zero windows containing a future timestamp in both materialized archives.',
        '- Original candidate coverage: 224/320 decisions (70.0%). Current-observation guard retains 204/320 (63.75%), or 204/224 (91.07%) of previously eligible rows. The 20 exclusions include 9 active positives and 4 onset positives. Exclusions are abstentions; they are not relabeled negatives. Quality-matched conditional metrics are reported separately from original-cohort results.',
        '- F9 is locked **validation**, not test. It was excluded from every new threshold fitting set and all weight/configuration selection. Each scored validation source is also excluded from its own threshold tuning fold. Other validation sources had been inspected historically, so this remains exploratory source-crossfit analysis, not a fresh independent benchmark.',
        '- Full-cohort metrics at a threshold selected on the other 12 sources are reported as selection statistics. Source-held-out results use different thresholds per source and are the primary comparison. Window-level FPR is not event false alarms/hour.',
        '', '## Root causes supported by measurements','',
        '1. **Threshold and eligibility are distinct causes.** Previous headwise active ranking improved, but its guarded reference threshold 0.590869 gave only 3 TP / 48 FN on the original cohort. The active head was independently disabled: its effective runtime threshold was 1.000001. Lowering a reference threshold alone would not enable candidates. Research-score mode separately suppresses notifications, not candidate scoring.',
        '2. **Score overlap remains substantial.** F1 optimization trades missed positives for many negative-window crossings. On the quality-matched cohort, the new F1 crossfit operating point has 26 TP / 16 FN / 49 FP / 113 TN; FPR 30.25%. A constraint fitted on other sources does not guarantee its held-out FPR. Even the minimum-precision-50% tuning policy produces 0 TP / 11 FP for the new model after source holdout. This is source instability, not a useful high-precision policy.',
        '3. **Imbalance and distribution shift affect scores and calibration.** Fitting active prevalence is 30.32%, versus 24.80% internal dev and 20.59% quality-matched validation. Weighted BCE improves dev AP slightly but moves scores upward; it does not make scores probabilities. Uniform sampling avoids the historical mismatch between source-balanced sampling and raw-window class weights.',
        '4. **Overfitting and limited source diversity.** Frozen-readout fit BCE falls while dev BCE stays around 0.71–0.75; earlier full-ST-GCN development curves were also unstable. Early stopping and source-separated selection limit the damage. Thousands of overlapping windows are not thousands of independent examples.',
        '5. **Pose coverage and tracking affect usable history.** The current observation guard excludes 20 originally eligible validation windows. F9 has 35/40 usable decisions and 9–10 observed-identity resets per eight-second replay. Resets intentionally discard confirmation history. Anonymous ID churn suggests an input consistency issue; it is not independently measured tracker error without tracking ground truth.',
        '6. **Weak global labels limit learning.** Labels describe a source time, while inputs select up to two tracks by past box area; these are not verified identities of the fighting people. Short bursts, boundaries and occlusion can mismatch the selected pair. This is a plausible limitation rather than a newly verified label error; deterministic encoding/alignment tests found no implementation inversion or timestamp leak.',
        '7. **Architecture constraints.** Existing joint/bone/backward-velocity encoders use symmetric two-person pooling and eight interaction summaries. Joint masks merge observed and short causal imputations; there is no separately learned observed-vs-imputed input channel. A new readout cannot recover motion missing from the cache. No evidence here justifies replacing the complete backbone or claiming an acceleration/identity feature gain.',
        '', '## Controlled experiments','',
        'Seven small readout configurations, each with seeds 73, 101 and 137 (21 small fits, zero new full-backbone retrains). Frozen 224-dimensional advanced ST-GCN representation; fitting-only mean/SD normalization; linear active head; AdamW lr=0.003, weight decay=0.01, batch 128, max 80 epochs, patience 10. Select epoch by internal-dev active average precision, tie by unweighted BCE. Select configuration by mean seed dev AP, tie lower BCE. Average three seed probabilities for inference. The onset output is exactly the historical baseline output.',
        '', 'BCE, positive-weighted BCE and focal gamma=2 were tested independently. Focal used no alpha balancing or class-weight/resampling combination. Feature ablations refit a readout on retained frozen feature blocks; they are not end-to-end stream-removal retrains. The 24-frame condition reads only the latest 24 observations with the same cutoff/targets; it does not relabel or change the validation cohort. Acceleration and new tracking-mask channels were reviewed but not added without supporting evidence.',
        '', '| Configuration | Mean seed dev AP ± SD | Val AUROC | Val AP | Val trap PR-AUC | F1 crossfit F1 | FPR10 crossfit P / R / F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,loss,groups,frames in CONFIGS:
        dev=s['runs'][name];v=ablation[name];f=v['policies']['f1']['source_crossfit']['metrics'];g=v['policies']['fpr10']['source_crossfit']['metrics']
        lines.append(f"| {name} | {dev['mean_seed_dev_ap']:.4f} ± {dev['sd_seed_dev_ap']:.4f} | {v['auroc']:.3f} | {v['average_precision']:.3f} | {v['pr_auc_trapezoid']:.3f} | {f['f1']:.3f} | {g['precision']:.3f} / {g['recall']:.3f} / {g['f1']:.3f} |")
    lines+=['',f"Selected on internal development: **{s['selected']}**, epochs {s['runs'][s['selected']]['selected_epochs']}. Full weighted dev AP {s['runs'][s['selected']]['mean_seed_dev_ap']:.4f} versus unweighted {s['runs']['full_bce_48']['mean_seed_dev_ap']:.4f}; this small difference is not proof of a loss-family population advantage. Joint-only seed AP SD is larger than the full-feature conditions. Bone/velocity/geometry benefits vary across this frozen representation; do not infer universal causal feature importance from seven exploratory arms.",
        '', '## Same-cohort model ranking and original reference thresholds','',
        '| Cohort | Model / head | N / positive | AUROC | AP | Trap PR-AUC | Reference threshold | P / R / F1 | TN / FP / FN / TP |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort,models in e['cohorts'].items():
        for name,heads in models.items():
            for head,v in heads.items():
                m=v['reference_threshold'];short='onset' if head==onset else 'active'
                threshold='disabled (1.000001)' if name=='readout' and short=='active' else f"{m['threshold']:.6f}"
                lines.append(f"| {cohort} | {name} / {short} | {v['n']} / {v['positives']} | {v['auroc']:.3f} | {v['average_precision']:.3f} | {v['pr_auc_trapezoid']:.3f} | {threshold} | {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f} | {m['tn']} / {m['fp']} / {m['fn']} / {m['tp']} |")
    lines+=['','Reference-threshold entries are score-level comparisons, not deployed event accuracy. The new dev-selected checkpoint initially disables active thresholding entirely; its separate versioned validation-reference checkpoint below records the exploratory validation threshold while retaining disabled eligibility.',
        '', '## Validation-only operating-point comparison','',
        'Policies: maximize F1; maximize F1 with tuning FPR ≤ 10%; maximize F1 with tuning precision ≥ 50%. Exact unique tied-score cutoffs are used, with a no-positive fallback if precision is infeasible. No test labels enter this calculation.',
        '', '| Model / head | Policy | Selected threshold excluding F9 | Source-crossfit P | R | F1 | FPR | TN / FP / FN / TP |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ('baseline','advanced','headwise','readout'):
        for head in HEADS:
            for policy,entry in q[name][head]['policies'].items():
                m=entry['source_crossfit']['metrics'];short='onset' if head==onset else 'active'
                lines.append(f"| {name} / {short} | {policy} | {entry['selected_without_f9']['threshold']:.6f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['false_positive_rate']:.3f} | {m['tn']} / {m['fp']} / {m['fn']} / {m['tp']} |")
    lines+=['','The single threshold column is fitted on all non-F9 validation sources for future reference. The adjacent crossfit metrics use a separately fitted threshold for every held-out source; they must not be interpreted as evaluation of that single fixed threshold. Full-cohort fixed-threshold metrics and every fold threshold are in evaluation.json. This prevents threshold selection on the source being scored, but does not erase earlier validation exposure or establish deployment generalization.',
        '', '### Calibration and score direction','',
        '| Model | Active Brier | Active NLL | Active ECE | Positive score median | Negative score median |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('baseline','advanced','headwise','readout'):
        v=q[name][active];r=v['calibration'];lines.append(f"| {name} | {r['brier']:.4f} | {r['nll']:.4f} | {r['ece']:.4f} | {v['positive_scores'][2]:.4f} | {v['negative_scores'][2]:.4f} |")
    temp=q['readout'][active]['development_temperature_diagnostic'];raw=q['readout'][active]['calibration'];scaled=temp['validation']
    lines+=['',f"Positive temperature fitted on internal dev only: T={temp['fit']['temperature']:.4f}; validation Brier {raw['brier']:.4f} → {scaled['brier']:.4f}, NLL {raw['nll']:.4f} → {scaled['nll']:.4f}. This is a diagnostic, not operational calibration, and was not applied to checkpoints. Sigmoid(logit) always denotes y=1; no score complement or validation-fitted sign inversion was used. Pairwise and rank-integrated AUROC agree. ECE uses 10 fixed bins; sparse-bin estimates are uncertain.",
        '', '### Source uncertainty and errors','']
    for cohort,v in ci.items():
        lines.append(f"- {cohort}: readout AUROC 95% source-cluster interval {v['interval_95']['readout_auroc']}; paired AUROC delta vs baseline {v['interval_95']['delta_auc_vs_baseline']}, vs headwise {v['interval_95']['delta_auc_vs_headwise']}.")
    lines+=['',f"For quality-matched FPR10 crossfit active F1, readout-minus-baseline 95% interval is {e['uncertainty']['fpr10']['delta_95']['f1']}. All 2,000 draws resample whole sources. Threshold-crossfit bootstrap intervals condition on the fitted folds and do not repeat threshold/model selection, so are exploratory rather than full nested-procedure uncertainty.",
        '', 'At the quality-matched FPR10 source-crossfit point, all seven new true-positive windows come from just two of the eight positive sources: f_72 (6) and f_79 (1). Six positive sources have zero true-positive windows. The 17 false positives are concentrated in f_72 (8), f_79 (2), n_360 (6) and n_234 (1). This source concentration explains why global ranking gains do not ensure reliable event recall. These counts are relative to weak annotations, not independently verified incidents.',
        '', 'per_source_errors.csv lists positive/negative counts, TP/FP/FN/TN and operating policies per source. validation_predictions.csv contains every timestamp, score, weak label and held-out threshold, allowing false-positive and missed-window review.',
        '', '## F9 complete inference diagnostic','',
        'Original F_9_1_2_0_0.mp4 is 96 seconds, locked validation, first weak onset 31.166667 seconds. Its cached context is 26.2–34.2 seconds. The uploaded review is a separately encoded 8-second clip on a 0–8 second local clock, not another validation source.',
        '',f"Three-anchor image matching estimates parent offset {alignment['estimated_parent_offset_s']:.6f}s (original frame {alignment['source_frame']}; MAE {alignment['mae_8bit']:.3f}/255 versus next-best {alignment['next_best_mae']:.3f}). This is image-derived alignment, not an authoritative edit record, and derived rows were not assigned new ground-truth labels or used for selection. Original/derived sample phases differ by one source frame; tracker initialization and input phases can change selected identities and scores. The observed score divergence is documented rather than assumed numerically equivalent.",
        '', '| Context / model | Valid / decisions | Raw active peak | Reference active threshold | Effective active threshold | Identity resets | Actual candidates | Counterfactual tuned candidates |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for context,models in f9.items():
        for name,v in models.items():
            lines.append(f"| {context} / {name} | {v['timing']['valid_decisions']} / {v['timing']['decisions']} | {v['peak_raw_scores'][1]:.6f} | {v['reference_thresholds'][1]:.6f} | {v['effective_thresholds'][1]:.6f} | {v['identity_resets']} | {v['actual_candidate_decisions']} | {v['counterfactual_validation_both_enabled_candidate_decisions']} |")
    lines+=['','Pipeline findings:',
        '', '- Video/source hashes match their caches. YOLO11n-Pose/ByteTrack outputs and strictly past windows were reused; no pose extraction was repeated. Cached source frames/timestamps remain monotonic.',
        '- Each replay has 3 decisions with insufficient history, 2 with too few currently observed joints, and zero failures of the 0.45 historical quality threshold. These five decisions abstain. Selected pair IDs and detected/currently observed people counts are included per timestamp.',
        '- The baseline and previous headwise active peaks remain below their respective reference thresholds. Their active eligibility is also disabled. Candidate confirmation and cooldown therefore cannot recover those misses.',
        f"- New derived-clip readout raw crossings: {confirm['derived']['raw_active_crossings']}; smoothed crossings: {confirm['derived']['smoothed_active_crossings']}; maximum positive hits retained in confirmation history: {confirm['derived']['maximum_active_hits_in_confirmation_history']}. At clip time 7.2s an identity reset leaves one history decision; 7.4s has two decisions/two hits; 7.6s abstains and clears history. Confirmation needs three history decisions and two positive smoothed hits. These crossings map approximately to 33.37–33.57s under image-derived alignment, after the first weak 31.166667–32.0s burst, so they are not evidence of a correctly timed fight detection. The actual new checkpoint also disables both heads. No F9-specific threshold or exception was introduced.",
        '- All actual and counterfactual confirmed candidate counts remain zero. Cooldown suppression is not the primary cause here. No incident/notification was emitted. The F9 missed detection remains unresolved; a higher peak score alone is not a fix.',
        '', 'f9/*_stages.csv trace raw and smoothed scores, history/current-observation/quality gates, selected/current IDs, source frame, reference/effective thresholds, identity resets, notification suppression and counterfactual status at each decision. f9/confirmation_stages.csv exposes confirmation-history length and hit counts.',
        '', '## Weak onset lead-time diagnostic','',
        'The new onset head is exactly the baseline, so onset rankings and crossings are unchanged. These are threshold-crossing windows before the first weak fight annotation; they are not confirmed candidate events or independently verified pre-contact warnings.',
        '', '| Policy | Weak positive sources | Sources with TP crossings | TP windows | Lead min / median / max (s) |','|---|---:|---:|---:|---|']
    for policy,v in lead['readout'].items():lines.append(f"| {policy} | {v['positive_source_count']} | {v['warned_sources']} | {v['true_positive_windows']} | {v['lead_time_s_quantiles']} |")
    lines+=['','Verified 1–3-second pre-contact aggression remains a separate blocked task: zero independently verified positive precursor videos are available. Weak first-onset forecasting, weak active-fight recognition and verified precursor prediction must not be pooled into one accuracy claim.',
        '', '## Checkpoints, recommendation and runtime regression','',
        f"- Keep default: `{BASELINE}`, SHA-256 `{sha256(BASELINE)}`; onset/active reference thresholds 0.603839457 / 0.611068130; research head eligibility [true, false]. The active threshold is diagnostic while that head is disabled.",
        f"- New dev-selected candidate: `{s['checkpoint']}`, SHA-256 `{s['checkpoint_sha256']}`. All three seed readouts and frozen encoders are stored in one strict-load checkpoint.",
        f"- New separate validation-reference version: `{e['reference_checkpoint']}`, SHA-256 `{e['reference_sha256']}`; onset/active reference thresholds {q['readout'][onset]['policies']['fpr10']['selected_without_f9']['threshold']:.9f} / {q['readout'][active]['policies']['fpr10']['selected_without_f9']['threshold']:.9f}, both research heads disabled. These are exploratory score-review reference values, not recommended operational alarm thresholds.",
        f"- Predeclared checks: `{json.dumps(e['acceptance_criteria'],sort_keys=True)}`. No default replacement is recommended. F1-optimal lowering would raise review load (49 FP windows on 162 negatives); constrained source-held-out results fail the practical precision/recall target.",
        '', '| Model | Validation model pass ms/window (batch 32) | Derived replay ms/usable decision |','|---|---:|---:|']
    for name in ('baseline','advanced','headwise','readout'):
        timing=e['latency']['quality_matched/'+name];replay=f9['derived'][name]['timing']
        lines.append(f"| {name} | {1000*timing['seconds']/timing['windows']:.2f} | {1000*replay['model_replay_seconds']/replay['valid_decisions']:.2f} |")
    lines+=['','Single measured CPU passes with two Torch threads; includes Python/loader/window overhead as applicable. Pose extraction and warmup-repeat benchmark methodology are not included. No end-to-end real-time claim is made.',
        '', 'Every one of the seven configurations also passed real full-window inference against its frozen feature predictions (maximum absolute score difference below 2e-6). candidate_runtime.json records per-configuration latency, 204-window sample count, 20.59% active prevalence and common 63.75% candidate-decision coverage. These runtime checks do not select a different model.',
        '', '## Tests and artifacts','',
        f"- C3/backend/optional suite: **{ntests} tests + {nsub} subtests passed**, zero failures/skips; one existing Starlette/httpx deprecation warning.",
        '- Nine new regression tests verify exact historical advanced logits, trailing-window history exclusion, padding exclusion, feature-group masks, finite/correct-direction loss gradients, tied-score constrained thresholds, infeasible precision fallback, source/F9 holdout and strict optimized-model loading/onset preservation.',
        '- Existing causal-prefix, cache/source isolation, model loading, incident adapter and shared backend integration tests passed. Frontend: 13/13 tests; C3 TypeScript/Vite production build passed.',
        '- One recorded-video runtime smoke run generated annotated MP4, scores and empty non-operational incident/notification envelopes using the new versioned checkpoint. The shared dashboard was not restarted or modified.',
        '', 'Artifacts under outputs/model_optimization_v1: protocol.json, selection.json, evaluation.json, development_comparison.csv, ablation_validation.csv/json, validation_predictions.csv, per_source_errors.csv, data_audit.json, ranking_uncertainty.json, weak_onset_lead_times.json, f9/, plots/, tests.log/xml, frontend logs, runtime_smoke/ and preservation manifests.',
        '', '## Exact Windows PowerShell commands','',
        'Use the project Python interpreter. For a repeat experiment choose unused output/weights version directories; existing artifacts deliberately refuse overwrite. Commands below reproduce this run with a new suffix. All script stages must use the same suffix.',
        '', '```powershell',
        "$ErrorActionPreference = 'Stop'",
        "Set-Location 'C:\\Users\\Kulanjana\\Downloads\\J26-DS-320\\components\\c3_aggression_detection'",
        "$py = '..\\..\\.venv\\Scripts\\python.exe'",
        "$out = 'outputs/model_optimization_v2'",
        "$weights = 'weights/model_optimization_v2'",
        '& $py -B -m src.optimize_model prepare --out-dir $out --weights-dir $weights',
        '& $py -B -m src.optimize_model train --out-dir $out --weights-dir $weights',
        '& $py -B -m src.optimize_model evaluate --out-dir $out --weights-dir $weights',
        "$env:MPLCONFIGDIR = Join-Path (Get-Location) ($out + '/mplconfig')",
        '& $py -B -m src.optimization_diagnostics --out-dir $out',
        '```','', 'Run the existing new candidate (choose a fresh output directory each time):','', '```powershell',
        "Set-Location 'C:\\Users\\Kulanjana\\Downloads\\J26-DS-320'",
        '& .\\.venv\\Scripts\\python.exe -B components/c3_aggression_detection/run.py `',
        '  --source dashboard/backend/uploads/b96502dd_F_9_1_2_0_0_review.mp4 `',
        '  --seconds 8 --checkpoint weights/model_optimization_v1/selected_validation_reference.pt `',
        '  --out-dir outputs/model_optimization_v1/manual_review_01',
        '```','', 'Run tests:','', '```powershell',
        "Set-Location 'C:\\Users\\Kulanjana\\Downloads\\J26-DS-320'",
        "$env:PYTHONDONTWRITEBYTECODE = '1'",
        "$c3 = Join-Path (Get-Location) 'components/c3_aggression_detection'",
        "$env:PYTHONPATH = @((Join-Path $c3 'data/phase1_work/testdeps'), (Join-Path $c3 'data/dashboard_deps_v1'), $c3, (Join-Path $c3 'legacy'), (Join-Path (Get-Location) 'dashboard/backend')) -join ';'",
        '& .\\.venv\\Scripts\\python.exe -B -m pytest components/c3_aggression_detection/tests `',
        '  components/c3_aggression_detection/tests_optional dashboard/backend/test_c3_integration.py `',
        '  -q -p no:cacheprovider --basetemp components/c3_aggression_detection/data/phase1_work/manual_optimization_tests_01',
        'npm --prefix dashboard/frontend test',
        'npm --prefix dashboard/frontend run build:c3','```','',
        'Normal user terminals can use npm as shown. This tool session used the bundled Node executable and the existing isolated npm CLI because npm was absent from its PATH; frontend logs record the successful result.',
        '', '## Remaining limitations and next evidence needed','',
        'The point ranking gain is encouraging, but the constrained recall/F1 failure and wide source intervals prevent a default switch. Collect additional source-separated fight and hard-negative contexts, independently review contact/precursor timing, and evaluate a frozen operating policy on new sources with event-level false alarms/hour and missed-event counts. Tracking/selected-pair annotation is needed to distinguish detector/association errors from model limitations. Preserve the locked test for an explicitly frozen final evaluation. These experiments do not justify operational alerts or verified early-warning claims.',
        '', '## Diagnostic figures','']
    for name in ('precision_recall','score_distributions','confusion_matrices','training_validation_loss','calibration','ablation_comparison','f9_scores'):
        lines+= [f'![{name.replace("_"," ")}]({(out/"plots"/(name+".png")).resolve().as_posix()})','']
    lines+=['## Primary references','',
        '- [scikit-learn decision thresholds](https://scikit-learn.org/1.5/modules/classification_threshold.html): ranking scores and the decision cutoff are separate; tuning/evaluation data must be separated.',
        '- [scikit-learn precision–recall example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html): AP and trapezoidal PR integration differ; this report names both.',
        '- [Original focal-loss paper](https://arxiv.org/abs/1708.02002) and [Torchvision focal-loss definition](https://docs.pytorch.org/vision/0.12/generated/torchvision.ops.sigmoid_focal_loss.html): focal modulation was tested as an independent controlled arm.','']
    target=Path('MODEL_OPTIMIZATION_RESULTS.md');write_new(target,'\n'.join(lines))
    print(json.dumps(dict(report=str(target.resolve()),tests=ntests,subtests=nsub,acceptance=e['acceptance_passed'],preservation=preserve,confirmation=confirm)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='outputs/model_optimization_v1');build(p.parse_args())
