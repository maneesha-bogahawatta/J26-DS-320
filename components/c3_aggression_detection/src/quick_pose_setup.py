"""Reuse finished pose caches and make a small, time-bounded validation pilot.

No training is performed. The source annotations remain weak unless independently
reviewed. A diagnostic eight-second clip is not a complete-recording benchmark.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from src.foundation_io import sha256, write_json
from src.lock_splits import load_locked
from src.pose_sequences import CACHE_SCHEMA, load_cache


def recover_entries(folder, rows, split='train'):
    """Index only complete, valid archives; leave every existing file intact."""
    by_id={r['video_id']:r for r in rows}
    entries={}; metadata={}; skipped=[]; contract=None
    for i,path in enumerate(sorted(Path(folder).glob('*.npz')),1):
        try:
            before=path.stat()
            # The explicit owner also closes truncated archives if NumPy raises
            # before constructing its archive context manager.
            with path.open('rb') as stream:
                arrays,meta=load_cache(stream)
            key=meta['video_id']
            if key not in by_id or meta['source_sha256']!=by_id[key]['content_sha256']:
                raise ValueError('Source identity/hash differs from locked manifest')
            if by_id[key]['split']!=split:
                skipped.append({'path':str(path),'reason':'different locked split'})
                continue
            config={k:meta.get(k) for k in ('schema','preprocessing','sample_fps','weights_sha256','tracker_sha256','libraries')}
            if contract is not None and config!=contract:
                raise ValueError('Mixed cache configurations; select a separate source folder')
            digest=sha256(path)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Cache was changing during recovery; retry once extraction stops')
            if key in entries:
                raise ValueError('Multiple archives for one source; choose a source folder with one cache per video')
            contract=config
            entries[key]={'path':str(path.resolve()),'cache_sha256':digest,'source_sha256':meta['source_sha256']}
            metadata[key]=meta
        except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as exc:
            skipped.append({'path':str(path),'reason':str(exc)})
        if i%20==0:
            print(f'Validated existing caches: {i}',flush=True)
    return entries,metadata,skipped


def select_reused_pilot(rows,entries,metadata,limit=12):
    """Use short already-cached training sources; do not rerun their extractor."""
    by_id={r['video_id']:r for r in rows}
    result={};groups=set()
    for key in sorted(entries,key=lambda k:(metadata[k]['source_duration_s'],k)):
        if by_id[key]['split']!='train':
            raise ValueError('Training pilot cannot include held-out identities')
        group=by_id[key]['leakage_group']
        if group in groups:
            continue
        result[key]=entries[key];groups.add(group)
        if len(result)>=limit:
            break
    return result


def validation_plan(rows,per_class=4,seconds=8.,seed=42):
    """Bounded diagnostic selection; no test sources and no verified-label claim."""
    jobs=[];groups=set()
    candidates=sorted((r for r in rows if r['split']=='val'),
                      key=lambda r:hashlib.sha256(f'{seed}:{r["video_id"]}'.encode()).hexdigest())
    for fight in (True,False):
        selected=0
        for row in candidates:
            intervals=json.loads(row.get('intervals') or '[]')
            duration=float(row['duration_s'])
            if bool(intervals)!=fight or duration<=0 or row['leakage_group'] in groups:
                continue
            onset=float(intervals[0][0]) if intervals else None
            if fight and onset<5:
                continue
            start=max(0.,onset-5.) if fight else max(0.,(duration-seconds)/2.)
            end=min(duration,start+seconds)
            jobs.append({'video_id':row['video_id'],'split':'val','start_s':start,'end_s':end,
                         'source_label':'weak_annotated_fight' if fight else 'weak_annotated_no_fight',
                         'selection':'retrospective diagnostic context; not verified early-warning supervision'})
            groups.add(row['leakage_group']);selected+=1
            if selected>=per_class:
                break
    return jobs


def run_bounded(jobs,run_one,budget_seconds,clock=time.monotonic):
    if not math.isfinite(budget_seconds) or budget_seconds<=0:
        raise ValueError('Budget must be positive and finite')
    deadline=clock()+budget_seconds
    results=[]
    for job in jobs:
        remaining=deadline-clock()
        if remaining<=0:
            break
        try:
            result=run_one(job,remaining)
            results.append(dict(job,**result))
        except subprocess.TimeoutExpired:
            results.append(dict(job,status='time_budget_reached'))
            break
    return results


def index_document(locked,entries):
    return {'schema':CACHE_SCHEMA,'locked_manifest_sha256':sha256(locked),'entries':entries}


def run(args):
    started=time.monotonic()
    if not 0<args.minutes<=15:
        raise ValueError('Quick mode allows at most 15 minutes; use a positive budget')
    rows=load_locked(args.locked)
    output=Path(args.out_dir).resolve();reports=Path(args.report_dir).resolve()
    entries,metadata,skipped=recover_entries(args.cached_train,rows)
    if not entries:
        raise ValueError('No valid training caches recovered; no new extraction started')
    write_json(output/'recovered_train_index.json',index_document(args.locked,entries))
    pilot=select_reused_pilot(rows,entries,metadata)
    write_json(output/'pilot_train_index.json',index_document(args.locked,pilot))
    jobs=validation_plan(rows)
    write_json(reports/'validation_plan.json',{'jobs':jobs,'new_source_seconds':sum(j['end_s']-j['start_s'] for j in jobs),
               'note':'Small retrospective diagnostic pilot, not full validation or verified labels'})
    print(f'Recovered {len(entries)} caches; training pilot reuses {len(pilot)} without YOLO inference.',flush=True)
    val_entries={}
    def run_one(job,remaining):
        job_index=output/'jobs'/(job['video_id']+'.json')
        command=[sys.executable,'-m','src.pose_sequences','--locked',str(Path(args.locked).resolve()),
                 '--split','val','--video-id',job['video_id'],'--start',str(job['start_s']),
                 '--end',str(job['end_s']),'--out-dir',str(output/'val_cache'),'--index',str(job_index)]
        print(f'Short validation clip: {job["video_id"]} ({job["end_s"]-job["start_s"]:.1f}s)',flush=True)
        process=subprocess.run(command,capture_output=True,text=True,timeout=remaining)
        if process.returncode:
            return {'status':'failed','error':process.stderr[-2000:] or process.stdout[-2000:]}
        generated=json.loads(job_index.read_text(encoding='utf-8'))
        val_entries.update(generated['entries'])
        return {'status':'cached'}
    remaining=max(.001,args.minutes*60-(time.monotonic()-started))
    results=[] if args.recover_only else run_bounded(jobs,run_one,remaining)
    if val_entries:
        write_json(output/'pilot_val_index.json',index_document(args.locked,val_entries))
    report={'existing_train_caches_recovered':len(entries),'reused_pilot_train_videos':len(pilot),
            'new_validation_clips':len(val_entries),'planned_source_seconds':sum(j['end_s']-j['start_s'] for j in jobs),
            'budget_minutes':args.minutes,'elapsed_seconds':round(time.monotonic()-started,2),
            'skipped_archives':skipped,'validation_jobs':results,
            'train_index':str(output/'recovered_train_index.json'),
            'pilot_train_index':str(output/'pilot_train_index.json'),
            'pilot_val_index':str(output/'pilot_val_index.json') if val_entries else None,
            'trained_model':False,'training_blocker':'No independently verified precursor positives/negative coverage yet',
            'limitations':['Pilot clips do not estimate full-recording accuracy or false alarms/hour.',
                           'Original caches, source data, labels and locked test identities are preserved.']}
    write_json(reports/'quick_setup.json',report)
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked',default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--cached-train',default='data/pose_cache/full_v1')
    p.add_argument('--out-dir',default='data/pose_cache/quick_v1')
    p.add_argument('--report-dir',default='outputs/quick_v1')
    p.add_argument('--minutes',type=float,default=15.)
    p.add_argument('--recover-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':
    main()
