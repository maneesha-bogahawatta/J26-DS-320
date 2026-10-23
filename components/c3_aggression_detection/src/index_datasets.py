"""Index downloaded datasets without inventing onset timestamps.
Requires review of weak labels before primary early-prediction training.
"""
import argparse,csv,json
from pathlib import Path
VIDEO={'.mp4','.avi','.mov','.mkv','.webm'}
AUDIO={'.wav','.flac','.mp3','.ogg','.m4a'}

def classify(p):
    parts=[a.lower() for a in p.parts]
    if any(t in parts for t in ('nonfight','nonviolence','normal','nonviolent')):return 'weak' # never auto-verify a folder name
    if any(t in parts for t in ('fight','violence','violent','abnormal')):return 'weak'
    return 'weak'

def main(root,source,out):
    root=Path(root).resolve(); files=sorted(p for p in root.rglob('*') if p.suffix.lower() in VIDEO|AUDIO)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',encoding='utf8',newline='') as f:
        field=['video','audio','source','group','intervals','annotation_status','has_audio']
        w=csv.DictWriter(f,fieldnames=field);w.writeheader()
        for p in files:
            is_audio=p.suffix.lower() in AUDIO
            # Safer grouping is one file per scene; replace if multiple crops originate from a shared source.
            w.writerow(dict(video='' if is_audio else str(p),audio=str(p) if is_audio else '',
                            source=source,group=f'{source}_{p.stem}',intervals='[]',
                            annotation_status=classify(p),has_audio='true' if is_audio else 'false'))
    print(f'Indexed {len(files)} files at {out}. Confirm labels and intervals by manual review.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.root,a.source,a.out)
