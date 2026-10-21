"""Prepare real model artifacts and a separate original-dashboard replay history."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from src.foundation_io import read_csv, sha256, write_json
from src.human_review import export_clip


def run(args):
    root=Path.cwd();out=Path(args.out_dir).resolve()
    if not out.is_relative_to(root):raise ValueError('PP1 output stays inside C3')
    previous=root/'outputs/original_dashboard_integration_v1/browser_smoke/jobs/0f790ddc2c5843c0870667f22349a696/result'
    old=json.loads((previous/'prediction.json').read_text())
    if old['video_id']!='f_147_1_0_1_0' or old['checkpoint_sha256']!=sha256('weights/baseline_weak_onset_guarded_v1.pt'):
        raise ValueError('Known tested demonstration provenance changed')
    target=out/'authentic_run'
    if target.exists():
        if not args.resume:raise FileExistsError('Choose an unused version or explicitly resume the completed authentic run')
    else:
        subprocess.run([sys.executable,'-B','-m','src.predict_advanced','--video-id',old['video_id'],
            '--out-dir',str(target),'--render','--camera-id','c3_pp1_recorded'],check=True)
    prediction=json.loads((target/'prediction.json').read_text());inc=json.loads((target/'incidents.json').read_text())
    scores=read_csv(target/'scores.csv');old_scores=read_csv(previous/'scores.csv')
    if len(scores)!=len(old_scores):raise ValueError('Authentic replay decision count changed')
    for a,b in zip(scores,old_scores):
        for name in ('t_s','weak_onset_score','fight_score','candidate_status','valid_pose'):
            if a[name]!=b[name]:raise ValueError('Fresh default inference differs from known tested recording')
    if prediction['incident_candidates']!=len(inc['incidents']) or not inc['incidents']:
        raise ValueError('Chosen legitimate clip no longer has its tested candidate evidence')
    work=out/'dashboard_state';job_id=uuid.uuid5(uuid.NAMESPACE_URL,'c3-pp1:'+sha256(target/'prediction.json')).hex
    result=work/'jobs'/job_id/'result';result.mkdir(parents=True,exist_ok=args.resume)
    hashes={}
    for p in target.iterdir():
        if p.is_file() and p.name!='render_work.mp4':
            if (result/p.name).exists():
                if sha256(p)!=sha256(result/p.name):raise ValueError('Existing replay artifact differs')
            else:shutil.copyfile(p,result/p.name)
            hashes[p.name]=sha256(p)
    job=dict(job_id=job_id,status='completed',source_id='cache:'+prediction['video_id'],
        source_name=prediction['video_id'],seconds=8.,camera_id=prediction['camera_id'],
        research_alerts=False,created_at=prediction['processed_at'],completed_at=prediction['processed_at'],
        operational_alerts_enabled=False,result_origin='recorded_authentic_inference',
        original_result=str(target),artifact_sha256=hashes)
    write_json(result.parent/'job.json',job)
    source=out/'upload_source/F_147_pp1_review.mp4'
    if not source.exists():
        export_clip(dict(source_video=prediction['source_video'],source_sha256=prediction['source_sha256'],
            clip_start_s=prediction['source_context'][0],clip_end_s=prediction['source_context'][1]),source,shutil.which('ffmpeg'))
    write_json(out/'demo_manifest_final.json',dict(job_id=job_id,state_dir=str(work),source_id=job['source_id'],
        checkpoint_sha256=prediction['checkpoint_sha256'],source_sha256=prediction['source_sha256'],
        source_context=prediction['source_context'],recorded_at=prediction['processed_at'],
        source_clip=str(source),
        actual_candidates=len(inc['incidents']),valid_decisions=prediction['timing']['valid_decisions'],
        decisions=prediction['timing']['decisions'],authentic_scores_match_prior=True,
        artifact_sha256=hashes,weak_research_only=True,new_inference_during_replay=False))
    write_json(out/'upload_demo_source.json',dict(path=str(source.resolve()),sha256=sha256(source),
        original_source=prediction['video_id'],parent_offset_s=prediction['source_context'][0],
        note='Legitimate derived review clip; upload is a new extraction/initialization and may differ from cached original-clock replay'))
    print(json.dumps(dict(job_id=job_id,authentic_candidates=len(inc['incidents']),state_dir=str(work)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',default='outputs/pp1_v1');p.add_argument('--resume',action='store_true');run(p.parse_args())
