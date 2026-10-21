import csv, json, os
from pathlib import Path

def load_manifest(path, allow_weak=False):
    rows=[]
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            video = r.get('video','').strip()
            audio = r.get('audio','').strip()
            if not video and not audio: raise ValueError('Each row needs video or audio path')
            for key in ('video','audio'):
                if r.get(key):
                    p=Path(r[key]); r[key]=str(p if p.is_absolute() else (Path(path).resolve().parent / p).resolve())
                    if not Path(r[key]).is_file(): raise FileNotFoundError(r[key])
            r['intervals'] = json.loads(r.get('intervals') or '[]')
            r['precursors'] = json.loads(r.get('precursors') or '[]')
            r['split']=(r.get('split') or '').strip().lower()
            if r['split'] not in ('','train','val','test'):raise ValueError('split must be train, val, test or blank')
            r['annotation_status']=(r.get('annotation_status') or 'weak').lower()
            if r['annotation_status'] not in ('verified','weak','negative_verified'):
                raise ValueError('annotation_status must be verified / weak / negative_verified')
            r['has_audio'] = bool(audio) or str(r.get('has_audio','')).lower() in ('1','true','yes')
            r['group'] = r.get('group') or Path(video or audio).stem
            r['source'] = r.get('source','unknown')
            if r['annotation_status']=='verified' and not r['intervals']:
                raise ValueError('verified onset samples must contain intervals; use negative_verified for clean negatives')
            if r['annotation_status']=='negative_verified' and r['intervals']:
                raise ValueError('negative_verified sample cannot contain positive intervals')
            if not allow_weak and r['annotation_status']=='weak': continue
            rows.append(r)
    return rows

def split_groups(rows, seed=42):
    """Groups never appear in more than one split."""
    import random
    groups={}
    for r in rows: groups.setdefault(r['group'],[]).append(r)
    reserved={}
    for group,rr in groups.items():
        named={r.get('split','') for r in rr if r.get('split','')}
        if len(named)>1:raise ValueError(f'Group {group} appears in multiple locked splits')
        if named:reserved[group]=named.pop()
    unlocked=[g for g in groups if g not in reserved];random.Random(seed).shuffle(unlocked)
    counts={s:sum(x==s for x in reserved.values()) for s in ('train','val','test')}
    n=len(groups);targets={'train':int(.70*n),'val':int(.15*n),'test':n-int(.70*n)-int(.15*n)}
    for g in unlocked:
        deficits={s:targets[s]-counts[s] for s in targets}
        choice=max(deficits,key=lambda s:(deficits[s],s=='train'))
        reserved[g]=choice;counts[choice]+=1
    return [[r for g,rr in groups.items() if reserved[g]==s for r in rr] for s in ('train','val','test')]
