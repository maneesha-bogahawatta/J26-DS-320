"""Read-only raw-annotation, review-media and F9 regression checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np

from src.contact_review import FORM_SCHEMA
from src.foundation_io import read_csv, sha256, write_csv, write_json
from src.lock_splits import load_locked
from src.ubi_annotations import load_annotation, probe_video


def raw_annotations(args):
    rows=load_locked(args.locked);reports=[]
    for i,row in enumerate(r for r in rows if r['split']!='test'):
        path=Path(row['source_annotation']);record=dict(video_id=row['video_id'],split=row['split'],source_annotation=row['source_annotation'])
        try:
            if path.is_file():
                n,fps,duration=probe_video(row['video'])
                parsed=load_annotation(path,'frame_binary',n,fps)
                expected=json.loads(row['intervals'])
                matches=len(parsed)==len(expected) and np.allclose(np.array(parsed),np.array(expected),atol=1e-6,rtol=0)
                record.update(status='matches_locked_weak_intervals' if matches else 'interval_mismatch',
                    frames=n,fps=fps,intervals=len(parsed),source_sha256=sha256(path))
            elif row['source_annotation']=='normal_name_candidate' and not json.loads(row['intervals']):
                record.update(status='weak_normal_name_only_no_verified_negative',intervals=0)
            else:record.update(status='missing_original_annotation')
        except (ValueError,RuntimeError,OSError) as exc:record.update(status='invalid_annotation',error=str(exc))
        reports.append(record)
        if (i+1)%100==0:print(f'Raw annotation metadata checks {i+1}/932',flush=True)
    out=Path(args.out_dir);write_csv(out/'raw_annotation_audit.csv',reports)
    write_json(out/'raw_annotation_summary.json',dict(development_sources=len(reports),status_counts=dict(Counter(r['status'] for r in reports)),
        format='Explicit existing frame_binary parser; positive=1; end-exclusive frame/FPS intervals',
        test_source_files_opened=0,labels_promoted_to_verified=0,original_files_modified=False))


def review_media(args):
    import cv2
    root=Path(args.pack_dir);pack=json.loads((root/'pack.json').read_text());clips=[]
    for c in pack['clips']:
        path=root/c['media']
        if sha256(path)!=c['media_sha256']:raise ValueError('Review clip checksum changed')
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT);ok,frame=cap.read();cap.release()
        if not ok or fps<=0 or abs(frames/fps-(c['clip_end_s']-c['clip_start_s']))>.15:raise ValueError('Review media decode/duration failed')
        clips.append(dict(clip_id=c['clip_id'],duration_s=frames/fps,decodes=True))
    node=Path(args.node);pages=[]
    for page in root.glob('reviewer_*/*/review.html'):
        source=page.read_text(encoding='utf-8');script=re.search(r'<script>(.*?)</script>',source,re.S).group(1)
        config=json.loads(re.search(r'const config=(.*);\nconst \$',script).group(1))
        if config['form']['schema']!=FORM_SCHEMA or any(r['review_label']!='unreviewed' for r in config['form']['records']):raise ValueError('Nonblank human form')
        if config['form']['reviewer'] or config['form']['independent_human_review']:raise ValueError('Reviewer impersonation')
        for clip in config['clips']:
            if set(clip)-{'clip_id','split','clip_start_s','clip_end_s','media'}:raise ValueError('Coordinator hints leaked into review UI')
            if not (page.parent/clip['media']).resolve().is_file():raise ValueError('Broken relative review media link')
        with tempfile.TemporaryDirectory(dir=Path(args.out_dir)) as folder:
            js=Path(folder)/'page.js';js.write_text(script,encoding='utf-8')
            subprocess.run([str(node),'--check',str(js)],check=True,capture_output=True,text=True)
        pages.append(dict(path=str(page),js_syntax_passed=True,blank_records=len(config['form']['records']),coordinator_hints_excluded=True))
    write_json(Path(args.out_dir)/'review_media_checks.json',dict(clips=clips,pages=pages,
        browser_visual_interaction_performed=False,template='Unmodified existing contact_review_ui.html'))


def f9(args):
    out=Path(args.out_dir);fresh=[r for r in read_csv(out/'runtime/runtime_validation_scores.csv') if r['video_id']=='f_9_1_2_0_0']
    previous=read_csv('outputs/model_optimization_v1/f9/original_baseline_stages.csv')
    if len(fresh)!=len(previous):raise ValueError('F9 decision count changed')
    max_delta=0.
    for a,b in zip(fresh,previous):
        for name in ('t_s','valid_pose','candidate_status','identity_reset','track_ids_json'):
            if a[name]!=b[name]:raise ValueError('F9 runtime decision changed: '+name)
        for name in ('weak_onset_score','fight_score'):
            if a[name]:max_delta=max(max_delta,abs(float(a[name])-float(b[name])))
    raw_summary=json.loads(Path('outputs/model_optimization_v1/f9/summary.json').read_text())
    write_json(out/'f9_recheck.json',dict(video_id='f_9_1_2_0_0',locked_split='val',test_member=False,
        source_context_s=[26.2,34.2],decision_count=len(fresh),usable_decisions=sum(r['valid_pose']=='True' for r in fresh),
        maximum_score_difference_from_previous_baseline=max_delta,
        same_validity_identity_and_candidate_decisions=True,
        actual_candidates=sum(r['candidate_status'].endswith('_ALERT') for r in fresh),
        existing_diagnostic=raw_summary['original']['baseline'],
        verified_first_contact_available=False,
        review_question='Review original video clock for visible cue and first physical contact; binary onset 31.166667s is weak, not confirmed contact.',
        review_clip_id=next(c['clip_id'] for c in json.loads((Path(args.pack_dir)/'pack.json').read_text())['clips'] if c['video_id']=='f_9_1_2_0_0'),
        model_weights_thresholds_or_tracking_settings_modified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('raw','media','f9'))
    p.add_argument('--locked',default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--pack-dir',default='data/reviewed_annotations/data_quality_v1')
    p.add_argument('--out-dir',default='outputs/data_quality_v1')
    p.add_argument('--node',default='C:/Users/Kulanjana/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
    a=p.parse_args();dict(raw=raw_annotations,media=review_media,f9=f9)[a.command](a)
