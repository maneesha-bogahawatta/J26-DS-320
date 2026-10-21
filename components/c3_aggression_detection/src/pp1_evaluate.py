"""Re-score unchanged validation archives/checkpoints without fitting or tuning."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.advanced_experiment import Windows, load_windows, score, metrics
from src.diagnose_stgcn import ranking
from src.foundation_io import sha256, write_json, write_csv
from src.improve_model import audit_roles
from src.optimize_model import load_model, HEADS


def run(args):
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    historical=list(csv.DictReader(Path('outputs/model_optimization_v1/validation_predictions.csv').open()))
    checkpoints=dict(baseline='weights/baseline_weak_onset_guarded_v1.pt',
        headwise='weights/model_improvement_expanded_v1/headwise_guarded.pt',
        readout='weights/model_optimization_v1/selected_validation_reference.pt')
    results={};predictions=[];max_delta=0.
    for cohort,folder in [('original','data/pose_cache/advanced_v1/windows'),
                          ('quality_matched','data/pose_cache/model_improvement_expanded_v1/windows')]:
        data,meta=load_windows(folder);audit_roles(data,meta);ix=np.flatnonzero(data['role']=='val')
        rows=[r for r in meta['coverage'] if r['role']=='val']
        results[cohort]=dict(materialized_windows=len(ix),sources=len(set(data['video_id'][ix].tolist())),
            candidate_windows=sum(r['candidate_windows'] for r in rows),eligible_windows=sum(r['eligible_windows'] for r in rows),
            windows_sha256=meta['windows_sha256'],models={})
        for name,path in checkpoints.items():
            model,ck=load_model(path);started=time.monotonic()
            scored=score(model,DataLoader(Windows(data,'val'),batch_size=32),ck['variant'])
            if not np.array_equal(scored['indices'].numpy(),ix):raise ValueError('Scored indices differ')
            p=scored['logits'].sigmoid().numpy();y=scored['targets'].numpy();mask=scored['masks'].numpy()
            report=dict(checkpoint_sha256=sha256(path),seconds=time.monotonic()-started,heads={})
            for h,head in enumerate(HEADS):
                use=mask[:,h];yy=y[use,h].astype(int);pp=p[use,h]
                stat=ranking(yy,pp);stat.update(metrics(yy,pp,ck['thresholds'][h]))
                stat['research_candidate_head_enabled']=bool(ck['research_alert_heads_enabled'][h])
                stat['score_quantiles']={str(c):np.quantile(pp[yy==c],[0,.1,.5,.9,1]).tolist() for c in (0,1)}
                report['heads'][head]=stat
                old={(int(r['index'])):float(r['score']) for r in historical if r['cohort']==cohort and r['model']==name and r['head']==head}
                if set(old)!=set(ix[use].tolist()):raise ValueError('Historical validation keys differ')
                delta=max(abs(float(pp[j])-old[int(i)]) for j,i in enumerate(ix[use]))
                max_delta=max(max_delta,delta)
                for j,i in enumerate(ix[use]):
                    pred=bool(pp[j]>=ck['thresholds'][h]);label=int(yy[j])
                    predictions.append(dict(cohort=cohort,model=name,head=head,video_id=str(data['video_id'][i]),
                        cutoff_s=float(data['cutoff_s'][i]),label=label,score=float(pp[j]),threshold=float(ck['thresholds'][h]),
                        diagnostic_prediction=pred,error='FP' if pred and not label else 'FN' if label and not pred else 'correct'))
            results[cohort]['models'][name]=report
    if max_delta>1e-6:raise ValueError(f'Predictions changed: {max_delta}')
    write_json(out/'evaluation.json',dict(cohorts=results,max_difference_from_historical_scores=max_delta,
        threshold_tuning=False,training_performed=False,test_sources_used=0,default_changed=False,
        verified_early_warning_results=None,label_status='Source-separated weak UBI labels; no independently verified precursors'))
    write_csv(out/'predictions.csv',predictions)
    print(json.dumps(dict(max_difference_from_historical_scores=max_delta,cohorts=results),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',default='outputs/pp1_v1/weak_evaluation')
    run(parser.parse_args())
