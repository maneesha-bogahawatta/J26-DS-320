"""Read-only checks of authentic completed jobs on the original local dashboard."""
import argparse
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

from src.foundation_io import write_json


def run(args):
    base=f'http://127.0.0.1:{args.port}'
    def get(path,headers=None):
        with urlopen(Request(base+path,headers=headers or {}),timeout=30) as response:
            return response.status,dict(response.headers),response.read()
    boot=json.loads(get('/api/v1/c3/bootstrap')[2]);manifest=json.loads(Path('outputs/pp1_v1/demo_manifest_final.json').read_text())
    recorded=manifest['job_id'];jobs=[j for j in boot['jobs'] if j['status']=='completed']
    jobs=[j for j in jobs if j['job_id']==recorded or j['source_name']=='F_147_pp1_review.mp4']
    if len(jobs)<2:raise ValueError('Run the real F_147 upload through the UI first')
    results=[]
    for job in jobs:
        key=job['job_id'];details=json.loads(get(f'/api/v1/c3/jobs/{key}')[2]);prediction=details['prediction']
        assert prediction['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert not prediction['operational_alerts_enabled'] and not prediction['verified_early_aggression_claim']
        assert len(details['scores'])==40 and prediction['timing']['valid_decisions']==37
        assert len(details['events'])==2
        status,headers,part=get(f'/api/v1/c3/media/{key}/demo.mp4',{'Range':'bytes=0-1023'})
        assert status==206 and len(part)==1024
        status,headers,body=get(f'/api/v1/c3/jobs/{key}/export.zip')
        archive=zipfile.ZipFile(io.BytesIO(body));names=archive.namelist()
        for required in ('result/prediction.json','result/scores.csv','result/incidents.json'):
            assert required in names
        assert not any(Path(n).name=='source.mp4' for n in names)
        assert json.loads(archive.read('result/prediction.json'))==prediction
        reviews=[json.loads(line) for line in archive.read('incident_reviews.jsonl').decode().splitlines()] if 'incident_reviews.jsonl' in names else []
        assert all(not r['eligible_for_training'] and not r['verified_precursor_annotation'] for r in reviews)
        results.append(dict(job_id=key,source_name=job['source_name'],recorded_replay=key==recorded,
            decisions=len(details['scores']),valid_decisions=prediction['timing']['valid_decisions'],
            research_candidates=len(details['events']),operational_notifications=prediction['research_notifications'],
            pose_cache_reused=prediction.get('pose_cache_reused'),extraction_seconds=prediction.get('extraction_seconds'),
            model_replay_seconds=prediction['timing']['model_replay_seconds'],media_range_status=206,
            export_files=names,operator_review_export=reviews))
    proposals=json.loads(Path('outputs/pp1_v1/provisional_labels_validated/proposals.json').read_text())
    assert proposals['reviewer_decisions']==[]
    assert all(r['label_status']=='provisional_automatic' and not r['human_review'] and
        not r['eligible_for_verified_evaluation'] and not r['eligible_for_strict_training'] and
        r['verified_first_contact_s'] is None and r['split']=='train' and r['internal_role']=='fit'
        for r in proposals['records'])
    report=dict(base_url=base,jobs=results,provisional_records_checked=len(proposals['records']),
        scope='Read-only API/artifact checks; manual browser observations are recorded separately',
        browser_evidence_record='outputs/pp1_v1/browser_checks.json',
        registered_team_detectors=json.loads(get('/api/v1/detect/detectors')[2]),
        human_ground_truth_created=False,passed=True)
    write_json(args.out,report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8002)
    p.add_argument('--out',default='outputs/pp1_v1/dashboard_checks.json');run(p.parse_args())
