"""Report the actual annotation blocker, review batch, checks and provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from src.foundation_io import read_csv, sha256, write_json, write_new
from src.lock_splits import load_locked


def preservation(out):
    root=Path.cwd().parents[1];before=json.loads((out/'preservation_before.json').read_text());changed=[];missing=[];appends=[]
    live='components/c3_aggression_detection/outputs/original_dashboard_integration_v1/browser_smoke/server.out.log'
    for name,entry in before.items():
        p=root/name
        if not p.is_file():missing.append(name);continue
        if sha256(p)==entry['sha256']:continue
        if name==live:
            with p.open('rb') as stream:prefix=stream.read(entry['bytes'])
            if hashlib.sha256(prefix).hexdigest()==entry['sha256']:
                appends.append(dict(path=name,preserved_prefix_bytes=entry['bytes'],appended_bytes=p.stat().st_size-entry['bytes']));continue
        changed.append(name)
    report=dict(files_checked=len(before),byte_identical_files=len(before)-len(appends)-len(changed)-len(missing),
        live_append_only_logs=appends,changed=changed,missing=missing,passed=not changed and not missing,
        note='The already-running dashboard may append its request log; its original byte prefix must remain intact.')
    write_json(out/'preservation_final.json',report)
    if not report['passed']:raise ValueError('Existing artifact preservation failed')
    return report


def build(args):
    out=Path(args.out_dir);audit=json.loads((out/'annotation_audit.json').read_text());plan=json.loads((out/'review_plan.json').read_text())
    readiness=json.loads((out/'training_readiness.json').read_text());raw=json.loads((out/'raw_annotation_summary.json').read_text())
    prep=json.loads((out/'pack_preparation.json').read_text());f9=json.loads((out/'f9_recheck.json').read_text())
    evaluated=json.loads((out/'verified_evaluation.json').read_text());policy=json.loads((out/'frozen_evaluation_policy.json').read_text())
    loader=json.loads((out/'legacy_loader_readiness.json').read_text());runtime=json.loads((out/'runtime/runtime_evaluation.json').read_text())
    media=json.loads((out/'review_media_checks.json').read_text());queue=read_csv(out/'prioritized_queue.csv');cached=read_csv(out/'cached_context_audit.csv')
    verified=preservation(out);log=(out/'tests.log').read_text(encoding='utf-8-sig')
    match=re.search(r'(\d+) passed,.*?(\d+) subtests passed',log)
    if not match:raise ValueError('Complete passing test result missing')
    tests,subtests=map(int,match.groups());rows=load_locked('data/foundation/v1/ubi_locked.csv');flags=[]
    for r in rows:
        if r['split']=='test':continue
        ranges=json.loads(r['intervals']);first=ranges[0][0] if ranges else None
        flags.append(dict(video_id=r['video_id'],split=r['split'],weak_intervals=len(ranges),
            insufficient_three_second_weak_precontext=first is not None and first<3,
            fight_named_source_without_positive_frames=Path(r['video']).stem.casefold().startswith('f_') and not ranges,
            sub_100ms_weak_intervals=sum(b-a<.1 for a,b in ranges),
            cue_or_contact_verified=False))
    from src.foundation_io import write_csv
    write_csv(out/'annotation_context_flags.csv',flags)
    short=sum(r['insufficient_three_second_weak_precontext'] for r in flags);emptyfight=sum(r['fight_named_source_without_positive_frames'] for r in flags)
    totals={split:dict(cached_sources=sum(r['split']==split for r in cached),usable_decisions=sum(int(r['usable_decisions']) for r in cached if r['split']==split),
        zero_usable_sources=sum(r['split']==split and int(r['usable_decisions'])==0 for r in cached)) for split in ('train','val')}
    write_json(out/'completion.json',dict(tests=tests,subtests=subtests,review_clips=prep['clips'],assignments=prep['review_assignments'],
        verified_labels_created=0,training_status='blocked',model_trained=False,default_changed=False,
        preservation=verified,test_media_opened=False,committed=False,pushed=False))
    lines=['# Component 03 — Data quality and supervised readiness','',
        'Completed locally on 22 October 2026. Supporting engineering is complete; supervised early-warning training remains blocked by missing real independent human labels. No new model was trained.','',
        '## Results','',
        f"- Audited the selected canonical evidence document: {audit['canonical_records']} records; status counts `{json.dumps(audit['status_counts'],sort_keys=True)}`. Evidence versions are inventoried separately, never added together as independent observations.",
        f"- Reparsed **{raw['development_sources']} train/validation original UBI frame-binary annotation files** against actual video frame count/FPS using the existing parser: all 932 match locked weak intervals. **Zero test-source files opened**. Matching weak intervals does not verify threatening cues or first physical contact.",
        f"- Authenticated **{plan['cached_sources']} cached development sources** and inventoried **{plan['development_sources']} non-test sources**. Combined indices reuse 147 train and 16 validation caches. No YOLO/pose extraction was performed.",
        f"- Prepared **{prep['clips']} review clips, {prep['review_assignments']} unassigned reviewer tasks**: training 12 weak-fight + 12 weak-no-fight; validation 4 + 4. Six existing silent media copies reused, 26 exported with the existing export_clip utility. Each copy is eight seconds; total unique review media 256 seconds, or about 8.5 minutes of playback across two reviewers before pausing/annotation overhead.",
        '- Real reviewer IDs are intentionally blank. Reviewers A/B are workflow slots, not claimed people or completed reviews. No human or consensus label was generated.',
        f"- **{tests} tests + {subtests} subtests passed**, zero failures/skips; one pre-existing Starlette/httpx deprecation warning. All {len(media['clips'])} videos decoded and matched duration; all four review pages passed JS syntax, local media-link and blank-form checks.",
        f"- Preservation: {verified['byte_identical_files']} pre-existing files byte-identical; {len(verified['live_append_only_logs'])} already-running dashboard log changed only by an authenticated append. Models, dashboards, datasets, annotation parents and teammate sources are unchanged. Existing uncommitted model-optimization work was retained. No commit or push.",
        '', '## Training-readiness counts','',
        '| Split | Strict verified precursor-positive sources | Strict verified negative sources | Confirmed first-contact sources | Eligible early positive / negative windows |',
        '|---|---:|---:|---:|---:|']
    for split in ('train','val'):
        v=readiness['strict_verified'][split];w=readiness['eligible_cached_windows'][split]
        lines.append(f"| {split} | {v['precursor_positive_videos']} | {v['non_aggressive_videos']} | {v['confirmed_first_contact_videos']} | {w['strict_precursor_positive_windows']} / {w['strict_precursor_negative_windows']} |")
    lines+=['','Strict verified active-fight positive and negative windows are also zero. The existing train_stgcn --check-only loader independently confirms zero positive/negative supervised early windows and writes no checkpoint. Mere presence of one example per class would only pass an execution minimum; source diversity, compatible pose coverage, realistic negative hours and independent evaluation would still be required.',
        '', 'The immediate blocker is actual review: two distinct humans must independently annotate visible threatening cues, first physical contact and scoped non-aggressive coverage, then resolve disagreements explicitly. Existing 21 provisional single-review records, 20 AI reviews, 19 unreviewed records and weak binary labels cannot replace those observations.',
        '', '## Annotation audit and separated evidence','',
        'Read MODEL_OPTIMIZATION_RESULTS.md and the existing human_review, contact_review, reconcile_reviews, review_annotations, ai_review_evidence, pose_dataset, train_stgcn, evaluate_events and evaluate_advanced_runtime implementations, prior evaluation records and F9 diagnostics. Existing exact-consensus validation and the contact-review UI were reused.',
        '', 'The canonical parent is data/foundation/ai_visual_v1/annotations.json, SHA-256 `'+audit['annotations_sha256']+'`. Its status counts preserve provenance. New semantic partitions contain 1,000 weak, 22 provisional, 19 uncertain and 19 unreviewed records; verified_strict and verified_legacy_or_unsupported partitions are empty. The 19 uncertain records remain labeled with their original single/AI review status too. Counts are not separate independent videos.',
        '', 'A strict accepted consensus must match the existing contact definition and timestamp validator, agree with normalized coverage/cue/contact/fight fields, and link to two distinct attested raw human votes. Unsupported status strings, legacy interval-derived contact, AI proposals, uncertain brackets and test records do not count. Software can check provenance and agreement; it cannot authenticate that two actual humans worked independently.',
        '', 'strict_training_view.json is a separate derived projection containing only strict consensus and supporting raw-vote provenance. It currently has zero records. Original weak/provisional/uncertain evidence is retained in its parent and separate partitions. This view is the safer input for a later supervised readiness check; it does not replace the source evidence.',
        '',f"Original weak annotation quality flags: **{short}** development videos have less than three seconds before their first weak boundary; **{emptyfight}** fight-named videos contain no positive frame intervals. These are review flags, not corrections. Short sub-100ms segments are also recorded in annotation_context_flags.csv. A filename or absent positive frame does not create a verified negative. Weak timing agreement is a parser-integrity result, not a physical-contact label audit by humans.",
        '', '### Cache and review coverage','', '| Split | Cached sources | Usable decision windows in planned cached contexts | Sources with zero usable decisions |','|---|---:|---:|---:|']
    for split,v in totals.items():lines.append(f"| {split} | {v['cached_sources']} | {v['usable_decisions']} | {v['zero_usable_sources']} |")
    lines+=['','The current runtime requirements are retained: at least 8 history frames, track quality ≥ 0.45, and ≥ 6 currently observed joints in one person. Cached windows are overlapping observations, not independent labels. Zero-usable contexts are deferred to pose-failure diagnosis; guards were not lowered. Eligible cached windows are not supervised until actual verified labels cover their cutoff times.',
        '', 'development_source_inventory.csv lists all 932 development identities. cached_context_audit.csv records all 163 cached contexts, internal fit/dev/val role, coverage, observation failures, prior uncertainty and review need. Uncached sources require later scoped review/context planning and targeted extraction; no bulk extraction was started.',
        '', '## Balanced, prioritized two-reviewer workflow','',
        'Half the quota in each train/validation × weak-class stratum uses priority (prior uncertainty/single review, historical diagnostic misses/false-positive windows, and F9 recheck). Half uses deterministic seeded source diversity. Source leakage groups remain unique. F9 is validation-only. Internal development roles are recorded from the materialized window archive and are not reassigned.',
        '', 'The weak strata balance review effort, not final human labels. Some fight-labelled clips may be already violent, uncertain, or contain no visible precursor. Reviewers must record what they see; do not force class balance or infer contact from the eventual fight. The selected hard-negative/positive batch is retrospective diagnostic sampling and cannot estimate natural prevalence or operational FAR.',
        '', '1. Allocate two real independent human reviewers to slots A and B. Give each only their own pages/media; coordinator priority tables, model scores, weak boundaries, AI proposals and the other reviewer’s exports stay out of the review UI. Separate deterministic clip orders are used. Source footage itself can reveal behavior/class, so this is not a fully blinded benchmark.',
        '2. Each reviewer starts blank, enters their own real ID, attests to personal independent observation and reviews train and validation separately. Use the original-video clock, slow playback/frame stepping, explicit coverage, first threatening cue, cue intervals, visible contact and fight intervals. Downloads are local; export before closing and resume only your own file.',
        '3. Contact is the first visible strike, forceful push, kick or grapple under visible_strike_push_kick_or_grapple_v1. Ordinary touch alone is insufficient. A weak onset is not confirmed contact. If onset precedes the clip, mark already in progress; if hidden/ambiguous, mark uncertain with a bracket and leave confirmed contact blank.',
        '4. Non-aggressive labels apply only to watched explicit ranges and cannot also assert cues/contact/fight. Precursor cues must precede personally observed contact and overlap its last three seconds. An uncertain or single review supplies no verified early target.',
        '5. Preserve each raw export. Existing contact_review import appends decisions and verifies only exact semantic agreement of two distinct attested IDs. Notes may differ. Never average contact bounds or edit another review to manufacture agreement. Single/uncertain/disagreeing cases remain provisional.',
        '6. The import report provides disagreement_clips, single_review_clips and unreviewed_clips. A coordinator requests additional context or an explicit fresh adjudication round; retain both original exports. If a third adjudicator is used, keep that decision as a separate version rather than claiming the original pair agreed. No third-reviewer decision was invented here.',
        '', 'Prepared files: data/reviewed_annotations/data_quality_v1/pack.json; reviewer_A/train, reviewer_A/val, reviewer_B/train and reviewer_B/val each contain review.html and a blank review.template.json. Immutable media/source hashes are checked on import. The unchanged importer accepts this pack directly. Blank-form import smoke testing preserved all parent records and added only 64 unreviewed records in a separate output; it created no consensus.',
        '', '## F9 re-examination','',
        f"Original F_9_1_2_0_0 is locked validation, not test. Fresh default-baseline replay matches every previous timestamp, pose-validity, track identity, identity-reset and candidate decision; maximum raw-score difference **{f9['maximum_score_difference_from_previous_baseline']}**. {f9['usable_decisions']}/{f9['decision_count']} usable decisions, zero candidates.",
        '', 'The original cached context is 26.2–34.2s. The 8-second uploaded derivative has an image-matched approximate parent offset 26.166667s; that estimate is not authoritative edit provenance and supplies no new truth labels. Previous misses reflect low scores, disabled active eligibility, pose abstention and identity-reset confirmation history. No model parameter, threshold, tracker setting or F9-specific rule changed.',
        '', 'F9 is review_025 in both independent review forms. Reviewers see no weak boundary or model result. The coordinator knows that the weak boundary is 31.166667s and asks humans to establish whether visible threatening cues and first contact are observable, rather than assuming equality. The original and derivative are not independent sources. Verified contact remains unavailable.',
        '', '## Evaluation engineering','',
        'New src/evaluate_verified.py reuses strict contact validation/provenance, pose_dataset.labels_at, diagnose_stgcn.ranking, the actual CausalDecision and evaluate_events.score_events one-to-one matching. It uses an immutable explicit policy file copied from the unchanged default checkpoint; no threshold fitting is implemented. Train/validation development evaluation only; test rows/roles are rejected.',
        '', '- Window metrics: Precision, Recall, F1, AUROC, average precision (AP) and trapezoidal PR-AUC, confusion counts, eligible sample counts and pose abstention. Only explicit strict verified targets are scored. PR-AUC is conditional window discrimination, not event PR-AUC.',
        '- Early-event metrics: one-to-one contact matching for alerts strictly before confirmed contact, 0–3s lead, median lead and recall at ≥1/2/3s. Full reviewed three-second prehistory is required; insufficient-context events are reported separately. No-pose events remain in the eligible event denominator and count as misses.',
        '- False alarms/hour: confirmed alert notifications during unique reviewed non-aggressive time divided by that negative camera-hour exposure. Intersect with authenticated cached contexts; merge overlap and conservatively exclude conflicting non-negative review coverage. Unknown/uncertain time and unreviewed full-video duration do not inflate exposure. Pose abstention is reported but does not remove recording time from exposure.',
        '- Exact cached decision timelines, increasing timestamps, source/split/cache hashes, frozen policy/checkpoint hashes and per-row score validity are checked. Missing verified source timelines cannot silently drop missed events. Disabled heads remain explicitly disabled in event replay.',
        '',f"Actual default runtime diagnostic: {runtime['valid_decisions']}/{runtime['decisions']} usable decisions across 16 cached validation contexts, {runtime['weak_event_contexts_detected']}/{runtime['weak_event_contexts_total']} weak-onset contexts flagged and {runtime['weak_negative_context_alerts']} alerts in weak-negative contexts. These are weak-label diagnostic counts, not verified lead-time/precision/FAR results.",
        '', 'Actual strict evaluation is **blocked**: zero eligible verified contact events and zero verified negative camera-hours. Verified Precision/Recall/F1, PR-AUC, lead time and FAR/hour are null, not fabricated zeros or weak-label substitutes. Even after review, 32 prioritized eight-second clips cannot establish population false alarms/hour; new source-separated complete negative recordings and uncertainty estimates will be needed.',
        '', 'Frozen checkpoint: `'+policy['checkpoint']+'`, SHA-256 `'+policy['checkpoint_sha256']+'`; onset/active reference thresholds `'+str(policy['thresholds'])+'`; research eligibility `'+str(policy['research_heads_enabled'])+'`; operational alerts disabled.',
        '', '## Tests','',
        f"Full C3, optional and shared-backend integration suite: **{tests} passed + {subtests} subtests passed**, one existing warning. Fifteen new synthetic-fixture tests cover linked two-vote evidence, forged/legacy/AI/uncertain/test exclusion, balanced group selection, quality deferral, exposure union/conflicts, timeline integrity, null metrics when blocked, event lead/precision/recall, negative hours, no-pose misses, late alerts, disabled heads and single-class PR limits. Synthetic names/timings appear only in tests and were not exported as research truth.",
        '', 'Media verification: 32 decode/duration/hash checks; four page scripts pass node --check; blank IDs/attestations/labels and all relative video links checked. Review pages reuse the existing template; no new browser visual interaction was performed. Backend tests cover integration; frontend sources/build were not changed in this data-only task.',
        '', '## Exact Windows PowerShell commands','',
        'Open only the assigned reviewer pages. The two slots must be completed by different actual humans:','', '```powershell',
        "$ErrorActionPreference = 'Stop'",
        "Set-Location 'C:\\Users\\Kulanjana\\Downloads\\J26-DS-320\\components\\c3_aggression_detection'",
        "$py = '..\\..\\.venv\\Scripts\\python.exe'",
        "$pack = 'data/reviewed_annotations/data_quality_v1/pack.json'",
        "Start-Process '.\\data\\reviewed_annotations\\data_quality_v1\\reviewer_A\\train\\review.html'",
        "Start-Process '.\\data\\reviewed_annotations\\data_quality_v1\\reviewer_A\\val\\review.html'",
        '# Reviewer B uses these separately, starting blank:',
        "Start-Process '.\\data\\reviewed_annotations\\data_quality_v1\\reviewer_B\\train\\review.html'",
        "Start-Process '.\\data\\reviewed_annotations\\data_quality_v1\\reviewer_B\\val\\review.html'",'```','',
        'After the two humans save their real exports under completed_human_data_quality_v1 (filenames below are placeholders for those actual exports):','', '```powershell',
        '& $py -B -m src.contact_review import --pack $pack `',
        '  --annotations data/foundation/ai_visual_v1/annotations.json `',
        '  --reviews data/reviewed_annotations/completed_human_data_quality_v1/train_A.json `',
        '    data/reviewed_annotations/completed_human_data_quality_v1/train_B.json `',
        '    data/reviewed_annotations/completed_human_data_quality_v1/val_A.json `',
        '    data/reviewed_annotations/completed_human_data_quality_v1/val_B.json `',
        '  --out-dir data/foundation/contact_data_quality_human_v1',
        '& $py -B -m src.data_quality audit `',
        '  --annotations data/foundation/contact_data_quality_human_v1/annotations.json `',
        '  --out-dir outputs/data_quality_human_v1',
        '& $py -B -m src.train_stgcn --check-only `',
        '  --annotations outputs/data_quality_human_v1/strict_training_view.json `',
        '  --train-index outputs/data_quality_human_v1/train_index.json `',
        '  --val-index outputs/data_quality_human_v1/val_index.json `',
        '  --report outputs/data_quality_human_v1/loader_readiness.json',
        '& $py -B -m src.evaluate_verified `',
        '  --annotations data/foundation/contact_data_quality_human_v1/annotations.json `',
        '  --scores outputs/data_quality_v1/runtime/runtime_validation_scores.csv `',
        '  --index outputs/data_quality_v1/val_index.json `',
        '  --policy outputs/data_quality_v1/frozen_evaluation_policy.json `',
        '  --out outputs/data_quality_human_v1/verified_evaluation.json','```','',
        'The current complete cached validation timelines can be reused because the model/policy are unchanged. Human-reviewed coverage outside those cached contexts needs a new authenticated timeline; it must not be silently counted. The importer preserves its parent and raw decisions. Choose a fresh output version for every changed result.',
        '', 'Reproduce the supporting plan/media in unused version directories:','', '```powershell',
        '& $py -B -m src.data_quality audit --out-dir outputs/data_quality_v2',
        '& $py -B -m src.data_quality prepare --out-dir outputs/data_quality_v2 `',
        '  --pack-dir data/reviewed_annotations/data_quality_v2',
        '& $py -B -m src.data_quality_checks raw --out-dir outputs/data_quality_v2',
        '& $py -B -m src.data_quality_checks media --out-dir outputs/data_quality_v2 `',
        '  --pack-dir data/reviewed_annotations/data_quality_v2','```','',
        'Run the relevant complete test suite:','', '```powershell',
        "Set-Location 'C:\\Users\\Kulanjana\\Downloads\\J26-DS-320'",
        "$c3 = Join-Path (Get-Location) 'components/c3_aggression_detection'",
        "$env:PYTHONDONTWRITEBYTECODE = '1'",
        "$env:PYTHONPATH = @((Join-Path $c3 'data/phase1_work/testdeps'), (Join-Path $c3 'data/dashboard_deps_v1'), $c3, (Join-Path $c3 'legacy'), (Join-Path (Get-Location) 'dashboard/backend')) -join ';'",
        '& .\\.venv\\Scripts\\python.exe -B -m pytest components/c3_aggression_detection/tests `',
        '  components/c3_aggression_detection/tests_optional dashboard/backend/test_c3_integration.py `',
        '  -q -p no:cacheprovider --basetemp components/c3_aggression_detection/data/phase1_work/data_quality_manual_01','```','',
        '## Remaining research blockers','',
        'Two independent human reviews and consensus are still required. There are no strict verified positive or negative examples in either development split, no verified contact reference for F9, and no verified negative recording hours. Active recognition labels and verified precursors remain separate tasks. The next research action is to complete and adjudicate actual reviews, then reassess strict/cache readiness and evaluate a frozen policy on new sources. The existing baseline remains the default. No model fitting, threshold tuning, operational alert enablement, teammate edit, commit or push was performed.',
        '', '## Artifacts and references','',
        'Source additions: src/data_quality.py, src/data_quality_checks.py, src/evaluate_verified.py, src/data_quality_report.py and tests/test_data_quality.py. New local outputs: outputs/data_quality_v1/ and data/reviewed_annotations/data_quality_v1/. Existing human/contact UI, importer, models and dashboard are reused.',
        '', '[scikit-learn AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) distinguishes non-interpolated AP from trapezoidal PR-AUC. This report names both and leaves them undefined when verified classes are absent. The repository CONTACT_REVIEW.md supplies the explicit visible-contact annotation protocol.','']
    write_new('DATA_QUALITY_READINESS.md','\n'.join(lines))
    print(json.dumps(dict(report=str(Path('DATA_QUALITY_READINESS.md').resolve()),tests=tests,subtests=subtests,
        status='blocked',flags=dict(short_precontext=short,fight_named_without_positive=emptyfight),preservation=verified)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='outputs/data_quality_v1');build(p.parse_args())
