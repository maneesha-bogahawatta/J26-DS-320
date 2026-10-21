"""Train/validate the joint-only baseline, with explicit supervision blockers.

Default task is verified early-warning supervision. --task active-fight with
--supervision weak-fight defines an annotation-based active-fight experiment;
it never trains or validates the precursor head.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.foundation_io import write_json
from src.pose_dataset import PoseSequenceDataset, collate_sequences, readiness
from src.stgcn_baseline import JointOnlySTGCN


def training_blockers(train_report, val_report, task='early'):
    head=0 if task=='early' else 1
    blockers=[]
    for label,report in (('train',train_report),('val',val_report)):
        if report is None:
            blockers.append(f'Missing {label} cache/index')
            continue
        count=report['supervised_windows_per_head'][head]
        positive=report['positive_windows_per_head'][head]
        if positive==0:
            blockers.append(f'No eligible positive {task} windows in {label}')
        if count-positive==0:
            blockers.append(f'No eligible negative {task} windows in {label}')
    return blockers


def fit_model(train_data,val_data,out,task='early',epochs=10,batch_size=8,seed=42):
    if train_data.split!='train' or val_data.split!='val':
        raise ValueError('Fitting requires train data and validation requires val data; test is prohibited')
    if train_data.provenance.get('feature_contract_sha256')!=val_data.provenance.get('feature_contract_sha256'):
        raise ValueError('Train/validation pose cache configurations differ')
    if epochs<1 or batch_size<1:
        raise ValueError('epochs and batch_size must be positive')
    blockers=training_blockers(readiness(train_data),readiness(val_data),task)
    if blockers:
        raise ValueError('Training blocked: '+'; '.join(blockers))
    out=Path(out)
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    head=0 if task=='early' else 1
    sets=[]
    for ds in (train_data,val_data):
        indices=[i for i in range(len(ds)) if ds[i]['target_mask'][head]]
        sets.append(Subset(ds,indices))
    generator=torch.Generator().manual_seed(seed)
    loaders=[DataLoader(ds,batch_size=batch_size,shuffle=(i==0),num_workers=0,
                        collate_fn=collate_sequences,generator=generator) for i,ds in enumerate(sets)]
    model=JointOnlySTGCN()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    best=float('inf');best_state=None
    history=[]
    for epoch in range(epochs):
        losses=[]
        for phase,loader in enumerate(loaders):
            model.train(phase==0)
            total=n=0
            with torch.set_grad_enabled(phase==0):
                for batch in loader:
                    logits=model(batch['sk'],batch['temporal_mask'],batch['joint_mask'])['logits'][:,head]
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['targets'][:,head])
                    if phase==0:
                        optimizer.zero_grad();loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step()
                    total+=float(loss.detach())*len(logits);n+=len(logits)
            losses.append(total/n)
        history.append({'epoch':epoch+1,'train_bce':losses[0],'val_bce':losses[1]})
        print(json.dumps(history[-1]),flush=True)
        if losses[1]<best:
            best=losses[1];best_state=copy.deepcopy(model.state_dict())
    checkpoint={'schema':'c3_joint_only_stgcn_v1','state':best_state,'embedding_dim':64,
                'trained_heads':['precursor'] if task=='early' else ['active_fight'],
                'task':task,'seed':seed,'history':history,'train_provenance':train_data.provenance,
                'val_provenance':val_data.provenance,'probability_calibration':'UNVALIDATED',
                'early_prediction_claim':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:
        torch.save(checkpoint,stream)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--locked',default='data/foundation/v1/ubi_locked.csv')
    p.add_argument('--annotations',default='data/foundation/v1/annotations.json')
    p.add_argument('--train-index',default='data/pose_cache/v1/train_index.json')
    p.add_argument('--val-index')
    p.add_argument('--task',choices=['early','active-fight'],default='early')
    p.add_argument('--supervision',choices=['verified','weak-fight'],default='verified')
    p.add_argument('--frames',type=int,default=48)
    p.add_argument('--step-s',type=float,default=.2)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--report',default='outputs/phase1_foundation/training_readiness.json')
    p.add_argument('--out',default='weights/stgcn_joint_v1.pt')
    args=p.parse_args()
    if args.task=='early' and args.supervision!='verified':
        p.error('Weak fight annotations cannot supervise early precursors')
    train_data=PoseSequenceDataset(args.locked,args.annotations,args.train_index,split='train',
                                  frames=args.frames,step_s=args.step_s,supervision=args.supervision)
    val_data=PoseSequenceDataset(args.locked,args.annotations,args.val_index,split='val',
                                frames=args.frames,step_s=args.step_s,supervision=args.supervision) if args.val_index else None
    tr=readiness(train_data);va=readiness(val_data) if val_data is not None else None
    blockers=training_blockers(tr,va,args.task)
    report={'task':args.task,'supervision':args.supervision,'train':tr,'val':va,
            'training_blockers':blockers,'status':'blocked' if blockers else 'ready',
            'model_trained':False,'provenance':train_data.provenance}
    write_json(args.report,report)
    print(json.dumps(report,indent=2))
    if args.check_only:
        return
    if blockers:
        raise SystemExit('Training blocked; see readiness report. No checkpoint written.')
    fit_model(train_data,val_data,args.out,args.task,args.epochs,args.batch_size)


if __name__=='__main__':
    main()
