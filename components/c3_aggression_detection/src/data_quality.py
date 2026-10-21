"""Audit and prioritize existing evidence; reuse strict two-human contact tools.

This module creates no human decisions, trains no model, and excludes locked
test media. Weak labels and scores are coordinator-only selection information.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from src.contact_review import (FORM_SCHEMA, blank_contact_form, import_forms,
    readiness_report, validate_contact_record)
from src.foundation_io import read_csv, sha256, write_csv, write_json, write_new
from src.human_review import PACK_SCHEMA, cached_spans, export_clip, plan_clips
from src.lock_splits import load_locked
from src.pose_dataset import labels_at, sequence_window
from src.pose_sequences import load_cache
from src.reconcile_reviews import coverage_report, covered_seconds


def checked_document(path, locked):
    rows=load_locked(locked);doc=json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('schema')!='c3_review_evidence_v1' or doc.get('locked_manifest_sha256')!=sha256(locked):
        raise ValueError('Annotation document differs from locked manifest')
    by_id={r['video_id']:r for r in rows};seen=set()
    for r in doc['records']:
        if r['record_id'] in seen:raise ValueError('Repeated evidence record ID')
        seen.add(r['record_id'])
        if r['video_id'] not in by_id or r['split']!=by_id[r['video_id']]['split']:
            raise ValueError('Annotation identity/split mismatch')
    return rows,doc


def strict_records(doc):
    """Count only canonical strict consensus linked to two attested raw votes.

    Reuses contact validation. IDs/attestations cannot authenticate real humans.
    Legacy verified and unsupported claimed-verified records remain separate.
    """
    index={r['record_id']:r for r in doc['records']};accepted=[];rejected=[]
    for r in doc['records']:
        if r.get('split') not in ('train','val') or r.get('review_status')!='independently_verified':continue
        try:
            if r.get('source_schema')!=FORM_SCHEMA or r.get('issues'):raise ValueError('Legacy schema or issues')
            coverage=r['coverage_s'];canonical=validate_contact_record(r['contact_semantics'],
                dict(clip_start_s=min(a for a,b in coverage),clip_end_s=max(b for a,b in coverage)),max(b for a,b in coverage))
            if canonical['review_label'] not in ('precursor','non_aggressive','active_fight','already_violent'):
                raise ValueError('Uncertain/unreviewed consensus is not a target')
            if (canonical['review_label']!=r['label'] or canonical['precursors']!=r['precursors'] or
                canonical['first_physical_contact_s']!=r['contact_s'] or canonical['reviewed_ranges_s']!=coverage or
                canonical['verified_fight_intervals']!=r['fight_intervals']):raise ValueError('Semantic mismatch')
            refs=r['original']['evidence_record_ids']
            if len(refs)!=2 or len(set(refs))!=2:raise ValueError('Two raw evidence votes required')
            voters=[]
            for ref in refs:
                raw=index[ref];original=raw['original'];voter=str(original.get('reviewer') or '').strip().casefold()
                if not voter or original.get('independent_human_review') is not True:raise ValueError('Unattested raw vote')
                if raw.get('source_schema')!=FORM_SCHEMA or raw.get('contact_semantics')!=canonical:raise ValueError('Raw votes disagree')
                if raw['video_id']!=r['video_id'] or raw['split']!=r['split']:raise ValueError('Raw vote identity differs')
                voters.append(voter)
            if len(set(voters))!=2 or set(voters)!={str(v).strip().casefold() for v in r['reviewers']}:
                raise ValueError('Distinct matching reviewer identities required')
            accepted.append(r)
        except (KeyError,ValueError,TypeError) as exc:
            rejected.append(dict(record_id=r['record_id'],reason=str(exc)))
    return accepted,rejected


def evidence_audit(doc):
    accepted,rejected=strict_records(doc);valid={r['record_id'] for r in accepted}
    partitions=defaultdict(list)
    for record in doc['records']:
        status=record['review_status']
        if record['record_id'] in valid:state='verified_strict'
        elif status=='independently_verified':state='verified_legacy_or_unsupported'
        elif record.get('label')=='uncertain':state='uncertain'
        elif status=='weak':state='weak'
        elif status=='unreviewed':state='unreviewed'
        else:state='provisional'
        partitions[state].append(record)
    counts={}
    for split in ('train','val'):
        rr=[r for r in accepted if r['split']==split]
        counts[split]=dict(precursor_positive_videos=len({r['video_id'] for r in rr if r['label']=='precursor'}),
            non_aggressive_videos=len({r['video_id'] for r in rr if r['label']=='non_aggressive'}),
            active_fight_videos=len({r['video_id'] for r in rr if r['fight_intervals']}),
            confirmed_first_contact_videos=len({r['video_id'] for r in rr if r['contact_s'] is not None}),
            coverage_seconds=covered_seconds_by_source(rr))
    return dict(schema='c3_data_quality_audit_v1',canonical_records=len(doc['records']),
        status_counts=dict(Counter(r['review_status'] for r in doc['records'])),
        partition_counts={k:len(v) for k,v in partitions.items()},
        original_coverage=coverage_report(doc['records']),strict_verified=counts,
        rejected_claimed_verified=rejected,contact_readiness=readiness_report(accepted),
        test_media_reviewed=False,test_records_excluded_from_readiness=True,
        notes=['Evidence versions overlap: only the selected canonical annotation document contributes counts.',
            'Weak full-video annotation coverage is not human-reviewed coverage.',
            'Uncertain is a separate semantic partition; original review status/provenance is retained.',
            'Reviewer IDs and attestations cannot authenticate independence.']),dict(partitions)


def covered_seconds_by_source(records):
    groups=defaultdict(list)
    for r in records:groups[r['video_id']].extend(r['coverage_s'])
    return sum(covered_seconds(ranges) for ranges in groups.values())


def merged_indexes(args,out):
    indices={s:json.loads(Path(p).read_text()) for s,p in [('train',args.train_index),('val',args.val_index)]}
    for record_path in sorted(Path(args.expansion).glob('*.json')):
        r=json.loads(record_path.read_text());key=r['video_id']
        if key in indices['train']['entries']:raise ValueError('Expansion source already indexed')
        indices['train']['entries'][key]=r['entry']
    paths={}
    for split,index in indices.items():
        paths[split]=out/(split+'_index.json');write_json(paths[split],index)
    rows,spans=cached_spans(args.locked,paths)
    return rows,spans,indices,paths


def source_candidates(rows,spans,indices,doc):
    by_id={r['video_id']:r for r in rows};existing=defaultdict(list)
    for r in doc['records']:
        if r['review_status']!='weak':existing[r['video_id']].append(r)
    errors=defaultdict(lambda:dict(fp=0,fn=0))
    for r in read_csv('outputs/model_optimization_v1/per_source_errors.csv'):
        if r['cohort']=='quality_matched' and r['model']=='readout' and r['policy']=='fpr10':
            errors[r['source']]=dict(fp=int(r['fp']),fn=int(r['fn']))
    from src.advanced_experiment import load_windows
    windows,_=load_windows('data/pose_cache/model_improvement_expanded_v1/windows')
    roles={str(key):str(role) for key,role in zip(windows['video_id'],windows['role'])}
    verified,_=strict_records(doc);verified_by=defaultdict(list)
    for r in verified:verified_by[r['video_id']].append(r)
    candidates=[];clipplans={c['video_id']:c for c in plan_clips(rows,spans)}
    for key,(lo,hi) in sorted(spans.items()):
        row=by_id[key];clip=clipplans[key];a,m=load_cache(indices[row['split']]['entries'][key]['path'])
        valid=0;total=0;last=-1e9;currentfails=0;histfails=0;qualityfails=0;verified_windows=[0,0];verified_positive=[0,0]
        for i,t in enumerate(a['t']):
            if t<clip['clip_start_s'] or t>=clip['clip_end_s'] or t-last<.2-1e-8:continue
            last=t;sample=sequence_window(a,i);total+=1
            h=int(sample['temporal_mask'].sum())>=8;q=sample['quality']>=.45;o=sample['observed_mask'][-1].sum(0).max()>=6
            histfails+=not h;qualityfails+=not q;currentfails+=not o
            if not (h and q and o):continue
            valid+=1;y,mask=labels_at(float(t),verified_by[key])
            for head in range(2):verified_windows[head]+=int(mask[head]);verified_positive[head]+=int(mask[head] and y[head]>0)
        prior=existing[key];uncertain=any(r['label']=='uncertain' for r in prior)
        provisional=any(r['review_status'].startswith('provisional') for r in prior)
        priority=10+5*uncertain+3*provisional+min(12,errors[key]['fp']+errors[key]['fn'])
        if key=='f_9_1_2_0_0':priority+=20
        candidates.append(dict(video_id=key,split=row['split'],leakage_group=row['leakage_group'],
            internal_role=roles.get(key,'no_materialized_windows'),weak_stratum='weak_fight' if json.loads(row['intervals']) else 'weak_no_fight',
            clip_start_s=clip['clip_start_s'],clip_end_s=clip['clip_end_s'],candidate_decisions=total,
            usable_decisions=valid,pose_coverage=valid/max(1,total),history_failures=histfails,quality_failures=qualityfails,current_observation_failures=currentfails,
            prior_nonweak_records=len(prior),prior_uncertain=uncertain,prior_provisional=provisional,
            strict_precursor_positive_windows=verified_positive[0],strict_precursor_negative_windows=verified_windows[0]-verified_positive[0],
            strict_active_positive_windows=verified_positive[1],strict_active_negative_windows=verified_windows[1]-verified_positive[1],
            diagnostic_fp_windows=errors[key]['fp'],diagnostic_fn_windows=errors[key]['fn'],priority=priority,
            selection_reason='F9 descriptive contact/timing recheck' if key=='f_9_1_2_0_0' else 'Existing uncertainty/single review/error coverage plus seeded diversity',
            review_action='two_independent_reviews' if valid else 'pose_failure_diagnostic_defer_supervised_batch',
            cache_path=indices[row['split']]['entries'][key]['path']))
    return candidates,clipplans


def select_balanced(candidates,quotas):
    selected=[];groups=set();shortfalls=[]
    for split,per_class in quotas.items():
        for stratum in ('weak_fight','weak_no_fight'):
            pool=[c for c in candidates if c['split']==split and c['weak_stratum']==stratum and c['usable_decisions']>0]
            # Half priority/error/uncertainty, half deterministic source diversity.
            priority=sorted(pool,key=lambda c:(-c['priority'],c['video_id']))
            chosen=[]
            for c in priority[:(per_class+1)//2]:
                if c['leakage_group'] not in groups:chosen.append(c);groups.add(c['leakage_group'])
            remaining=sorted(pool,key=lambda c:hashlib.sha256(('data-quality-v1:73:'+c['video_id']).encode()).hexdigest())
            for c in remaining:
                if len(chosen)>=per_class:break
                if c['leakage_group'] not in groups:chosen.append(c);groups.add(c['leakage_group'])
            selected.extend(chosen)
            if len(chosen)<per_class:shortfalls.append(dict(split=split,stratum=stratum,requested=per_class,available=len(chosen)))
    if any(c['split']=='test' for c in selected):raise ValueError('Test source selected')
    return selected,shortfalls


def audit(args):
    out=Path(args.out_dir);rows,doc=checked_document(args.annotations,args.locked)
    report,partitions=evidence_audit(doc)
    report.update(annotations_path=str(Path(args.annotations).resolve()),annotations_sha256=sha256(args.annotations))
    write_json(out/'annotation_audit.json',report)
    for name in ('verified_strict','verified_legacy_or_unsupported','weak','provisional','uncertain','unreviewed'):
        write_json(out/'evidence_partitions'/(name+'.json'),dict(records=partitions.get(name,[]),source_sha256=sha256(args.annotations)))
    verified,_=strict_records(doc)
    supporting={ref for r in verified for ref in r['original']['evidence_record_ids']}
    training_view=dict(doc,records=[r for r in doc['records'] if r['record_id'] in supporting or r in verified],
        parent_annotations_sha256=sha256(args.annotations),
        purpose='Strict consensus and its raw provenance only; no weak, AI, uncertain or test targets')
    write_json(out/'strict_training_view.json',training_view)
    inventory=[]
    for p in sorted(Path('data/foundation').rglob('annotations.json')):
        d=json.loads(p.read_text());inventory.append(dict(path=str(p),sha256=sha256(p),records=len(d.get('records',[])),
            status_counts=dict(Counter(r.get('review_status') for r in d.get('records',[]))),counted_as_canonical=p.resolve()==Path(args.annotations).resolve()))
    write_json(out/'annotation_version_inventory.json',inventory)
    rows,spans,indices,paths=merged_indexes(args,out)
    candidates,clips=source_candidates(rows,spans,indices,doc)
    chosen,shortfalls=select_balanced(candidates,dict(train=args.train_per_class,val=args.val_per_class))
    write_csv(out/'cached_context_audit.csv',candidates);write_csv(out/'prioritized_queue.csv',chosen)
    by_key={r['video_id']:r for r in candidates};eligible=[]
    for row in rows:
        if row['split']=='test':continue
        c=by_key.get(row['video_id']);eligible.append(dict(video_id=row['video_id'],split=row['split'],leakage_group=row['leakage_group'],
            cache_available=c is not None,usable_cached_decisions=c['usable_decisions'] if c else 0,
            strict_verification_required=True,action=c['review_action'] if c else 'uncached_context_future_review_then_targeted_extraction'))
    write_csv(out/'development_source_inventory.csv',eligible)
    readiness=dict(status='blocked' if report['contact_readiness']['blockers'] else 'minimum_labels_present_not_scientifically_validated',
        strict_verified=report['strict_verified'],eligible_cached_windows={s:{k:sum(r[k] for r in candidates if r['split']==s) for k in (
            'strict_precursor_positive_windows','strict_precursor_negative_windows','strict_active_positive_windows','strict_active_negative_windows')} for s in ('train','val')},
        blockers=report['contact_readiness']['blockers'],new_verified_labels=0,model_trained=False,
        verified_precision=None,verified_recall=None,verified_f1=None,verified_pr_auc=None,verified_false_alarms_per_hour=None)
    write_json(out/'training_readiness.json',readiness)
    plan=dict(schema='c3_balanced_contact_plan_v1',canonical_annotations_sha256=sha256(args.annotations),
        locked_manifest_sha256=sha256(args.locked),cached_sources=len(candidates),development_sources=len(eligible),
        selected=[c['video_id'] for c in chosen],selected_by_split_stratum=dict(Counter(c['split']+'/'+c['weak_stratum'] for c in chosen)),
        shortfalls=shortfalls,reviewer_slots=['A','B'],reviewer_identities_assigned=False,
        target_balance='Weak-source selection strata only; actual human labels must not be balanced artificially',
        selection='Half priority, half seeded source diversity per split/weak stratum; source-group disjoint',
        validation_usage='Diagnostic, retrospectively selected contexts; no new threshold/model selection; not a prevalence/FAR benchmark',
        test_sources_selected=0,new_pose_extraction=False,new_ground_truth_created=False)
    write_json(out/'review_plan.json',plan)
    print(json.dumps(dict(status=readiness['status'],counts=report['status_counts'],selected=plan['selected_by_split_stratum'],shortfalls=shortfalls)),flush=True)


def prepare(args):
    out=Path(args.out_dir);plan=json.loads((out/'review_plan.json').read_text());rows,doc=checked_document(args.annotations,args.locked)
    if sha256(args.annotations)!=plan['canonical_annotations_sha256']:raise ValueError('Annotations changed since planning')
    # Use exactly the authenticated contexts frozen by the audit.
    indices={s:json.loads((out/(s+'_index.json')).read_text()) for s in ('train','val')}
    _,spans=cached_spans(args.locked,{s:out/(s+'_index.json') for s in indices})
    clip_by={c['video_id']:c for c in plan_clips(rows,spans)}
    pack_root=Path(args.pack_dir);pack_root.mkdir(parents=True,exist_ok=False)
    write_new(pack_root/'.gitignore','*\n')
    old_path=Path('data/reviewed_annotations/quick_human_v1/pack.json');old=json.loads(old_path.read_text())
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise FileNotFoundError('Existing FFmpeg required')
    clips=[];reused=0
    for i,key in enumerate(plan['selected'],1):
        clip=copy.deepcopy(clip_by[key]);clip['clip_id']=f'review_{i:03d}';clip['media']=f'media/{clip["clip_id"]}.mp4'
        target=pack_root/clip['media'];target.parent.mkdir(exist_ok=True)
        previous=next((c for c in old['clips'] if c['video_id']==key and abs(c['clip_start_s']-clip['clip_start_s'])<1e-6 and abs(c['clip_end_s']-clip['clip_end_s'])<1e-6),None)
        if previous:
            source=old_path.parent/previous['media']
            if sha256(source)!=previous['media_sha256']:raise ValueError('Existing review media changed')
            shutil.copyfile(source,target);clip['media_sha256']=sha256(target);clip['media_reused_from']=str(source);reused+=1
        else:clip['media_sha256']=export_clip(clip,target,ffmpeg)
        clips.append(clip);print(f'Prepared review media {i}/{len(plan["selected"])}',flush=True)
    pack=dict(schema=PACK_SCHEMA,clips=clips,locked_manifest_sha256=sha256(args.locked),
        base_annotations_sha256=sha256(args.annotations),selection=plan['selection'],supervision='Unreviewed blank human contact forms only')
    write_json(pack_root/'pack.json',pack);digest=sha256(pack_root/'pack.json')
    template=Path(__file__).with_name('contact_review_ui.html').read_text(encoding='utf-8');assignments=[]
    for slot in ('A','B'):
        for split in ('train','val'):
            form=blank_contact_form(pack,digest,split)
            # Separate seeded orders; no other reviewer's decisions, proposals or scores.
            form['records'].sort(key=lambda r:hashlib.sha256((slot+':'+r['clip_id']).encode()).hexdigest())
            lookup={c['clip_id']:c for c in clips};folder=pack_root/('reviewer_'+slot)/split
            ui_clips=[dict(clip_id=r['clip_id'],split=split,clip_start_s=lookup[r['clip_id']]['clip_start_s'],
                clip_end_s=lookup[r['clip_id']]['clip_end_s'],media=Path(os.path.relpath(pack_root/lookup[r['clip_id']]['media'],folder)).as_posix()) for r in form['records']]
            encoded=json.dumps(dict(form=form,clips=ui_clips),ensure_ascii=True,allow_nan=False).replace('<','\\u003c')
            write_new(folder/'review.html',template.replace('__REVIEW_CONFIG__',encoded));write_json(folder/'review.template.json',form)
            assignments.extend(dict(reviewer_slot=slot,reviewer_identity='',clip_id=r['clip_id'],split=split,status='unassigned_unreviewed') for r in form['records'])
    write_csv(out/'review_assignments.csv',assignments)
    write_json(out/'pack_preparation.json',dict(clips=len(clips),review_assignments=len(assignments),existing_media_reused=reused,
        silent_review_copies_exported=len(clips)-reused,pose_extraction_performed=False,labels_created=0,pack_sha256=digest))
    print(json.dumps(dict(pack=str((pack_root/'pack.json').resolve()),clips=len(clips),reused=reused,labels_created=0)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('audit','prepare'))
    p.add_argument('--locked',default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--annotations',default='data/foundation/ai_visual_v1/annotations.json')
    p.add_argument('--train-index',default='data/pose_cache/advanced_v1/train_index.json')
    p.add_argument('--val-index',default='data/pose_cache/advanced_v1/val_index.json')
    p.add_argument('--expansion',default='data/pose_cache/model_improvement_expanded_v1/completed')
    p.add_argument('--out-dir',default='outputs/data_quality_v1')
    p.add_argument('--pack-dir',default='data/reviewed_annotations/data_quality_v1')
    p.add_argument('--train-per-class',type=int,default=12);p.add_argument('--val-per-class',type=int,default=4)
    args=p.parse_args()
    if args.train_per_class<1 or args.val_per_class<1:p.error('Positive per-stratum counts required')
    dict(audit=audit,prepare=prepare)[args.command](args)


if __name__=='__main__':main()
