"""Train two separate RiskNet checkpoints: with audio and video-only."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from c3_model import RiskNet
from src.manifest import load_manifest,split_groups
from src.labels import targets
from src.causal_pose import PREPROCESSING_VERSION

FIELDS=('vis','pose','aud')

def load_cached(rows,cache_index):
    items=[]
    for row in rows:
        key=row.get('video') or row.get('audio')
        if key not in cache_index:continue
        z=np.load(cache_index[key])
        if 'preprocessing_version' not in z or str(z['preprocessing_version'].item())!=PREPROCESSING_VERSION:
            raise ValueError('Legacy/non-causal feature cache rejected; rebuild in a new cache directory')
        t=z['t']
        y=targets(t,row['intervals'],precursors=row.get('precursors',[]))
        items.append(dict(row=row,**{k:z[k] for k in (*FIELDS,'mods','sk','qual')},**y))
    return items

def fit_stats(items):
    stats={}
    for j,key in enumerate(FIELDS):
        parts=[x[key][x['mods'][:,j]>0] for x in items if (x['mods'][:,j]>0).any()]
        dim=items[0][key].shape[-1]
        v=np.concatenate(parts) if parts else np.zeros((1,dim),np.float32)
        stats[key]=(v.mean(0).astype('float32'),np.maximum(v.std(0),1e-4).astype('float32'))
    return stats

def make_batch(items,stats,mode,seq=12,batch=4,device='cpu'):
    xs={k:[] for k in (*FIELDS,'mods','sk','pre','pre_mask','fight','fight_mask','tto','tto_mask')}
    candidate_weights=[3.0 if (it['pre']>0).any() else 1.0 for it in items]
    for it in random.choices(items,weights=candidate_weights,k=batch):
        n=len(it['pre'])
        positive_idx=np.flatnonzero(it['pre']>0)
        if len(positive_idx) and random.random()<.65:
            end=min(n,int(random.choice(positive_idx))+random.randrange(1,5))
        else:end=random.randrange(1,n+1)
        begin=max(0,end-seq)
        valid=end-begin
        for key in FIELDS:
            val=(it[key][begin:end]-stats[key][0])/stats[key][1]
            xs[key].append(np.pad(val,((seq-valid,0),(0,0))).astype('float32'))
        mods=it['mods'][begin:end].copy()
        if mode=='video':mods[:,2]=0
        elif mode=='av' and random.random()<0.35:mods[:,2]=0 # missing-audio robustness
        xs['mods'].append(np.pad(mods,((seq-valid,0),(0,0))))
        xs['sk'].append(np.pad(it['sk'][begin:end],((seq-valid,0),(0,0),(0,0),(0,0),(0,0))))
        for k in ('pre','pre_mask','fight','tto','tto_mask'):
            xs[k].append(np.pad(it[k][begin:end],(seq-valid,0)))
        xs['fight_mask'].append(np.pad(np.ones(valid,dtype=np.float32),(seq-valid,0)))
    return {k:torch.as_tensor(np.stack(v),device=device,dtype=torch.float32) for k,v in xs.items()}

def evaluate(model,items,stats,mode,device):
    model.eval(); tp=fp=fn=0
    with torch.no_grad():
        for it in items:
            x={k:torch.tensor(((it[k]-stats[k][0])/stats[k][1])[None],device=device,dtype=torch.float32) for k in FIELDS}
            x['mods']=torch.tensor(it['mods'][None],device=device,dtype=torch.float32)
            if mode=='video':x['mods'][:,:,2]=0
            x['sk']=torch.tensor(it['sk'][None],device=device,dtype=torch.float32)
            pred,_,_,_=model(x['vis'],x['pose'],x['aud'],x['mods'],x['sk'])
            p=(torch.sigmoid(pred)[0].cpu().numpy()>=.5)
            y=it['pre']>.5;valid=it['pre_mask']>.5
            tp+=int((p&y&valid).sum());fp+=int((p&~y&valid).sum());fn+=int((~p&y&valid).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return 2*precision*recall/max(precision+recall,1e-8)

def train(mode,manifest,index_path,out,steps=500,seed=42,device=None):
    from src.lock_splits import load_locked
    load_locked(manifest)  # No implicit/random reassignment during fitting.
    if Path(out).exists():raise FileExistsError('Refusing to overwrite an existing checkpoint')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    rows=load_manifest(manifest)
    if mode=='av': rows=[r for r in rows if r.get('video')] # AV can include video-only negatives
    if mode=='video': rows=[r for r in rows if r.get('video')]
    if not rows:raise ValueError('No eligible onset/negative verified samples. Label your manifest first.')
    with open(index_path) as f:index=json.load(f)
    tr,va,_=split_groups(rows,seed)
    train_items=load_cached(tr,index);val_items=load_cached(va,index)
    if not train_items or not val_items:raise ValueError('Need cached locked train and validation groups.')
    if not any(x['pre'].sum()>0 for x in train_items):raise ValueError('No independently reviewed pre-onset CUE windows in train set. Onset timestamp alone is insufficient for early aggression training.')
    if mode=='av' and not any(x['mods'][:,2].any() for x in train_items):raise ValueError('No audio examples for AV training')
    stats=fit_stats(train_items);dv,dp,da=(train_items[0][k].shape[-1] for k in FIELDS)
    backend='none'
    with np.load(index[train_items[0]['row'].get('video') or train_items[0]['row'].get('audio')]) as first:
        if 'audio_events' in first: backend=str(first['audio_events'].item())
    for it in train_items+val_items:
        if any(it[k].shape[-1]!=d for k,d in zip(FIELDS,(dv,dp,da))):raise ValueError('Mixed feature dimensions. Re-cache with the same --audio-events backend.')
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=RiskNet(dv,dp,da,d=96).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    best=-1;path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    for step in range(1,steps+1):
        model.train();b=make_batch(train_items,stats,mode,device=device)
        pre,fight,tto,_=model(b['vis'],b['pose'],b['aud'],b['mods'],b['sk'])
        pre_bce=torch.nn.functional.binary_cross_entropy_with_logits(pre,b['pre'],reduction='none')
        l1=(pre_bce*b['pre_mask']).sum()/b['pre_mask'].sum().clamp(min=1)
        fight_bce=torch.nn.functional.binary_cross_entropy_with_logits(fight,b['fight'],reduction='none')
        l2=(fight_bce*b['fight_mask']).sum()/b['fight_mask'].sum().clamp(min=1)
        l3=(((tto-b['tto'])**2)*b['tto_mask']).sum() / b['tto_mask'].sum().clamp(min=1)
        loss=2*l1+l2+.1*l3
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%max(1,steps//10)==0 or step==steps:
            score=evaluate(model,val_items,stats,mode,device)
            print(f'{mode} step={step} loss={loss.item():.4f} val_pre_F1={score:.3f}')
            if score>best:
                best=score
                torch.save(dict(state=model.state_dict(),dims=(dv,dp,da),hidden_dim=96,norm=stats,mode=mode,
                                threshold_watch=.35,threshold_high=.65,threshold_status='UNVALIDATED',
                                audio_events=backend,preprocessing_version=PREPROCESSING_VERSION,
                                version='c3_vscode_v2_onset_reviewed'),path)
    print('Training complete; no test features loaded. Run separate validation calibration and locked-test evaluation.')
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['av','video'],required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--index',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=500);a=p.parse_args();train(a.mode,a.manifest,a.index,a.out,a.steps)
