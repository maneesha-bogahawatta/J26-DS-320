import argparse,json
from pathlib import Path
from src.manifest import load_manifest
from src.features import extract_one

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out-dir',default='data/cache')
    p.add_argument('--pose-weights',default='yolo11n-pose.pt');p.add_argument('--index',default='data/cache_index.json')
    p.add_argument('--audio-events',choices=['none','yamnet'],default='none')
    a=p.parse_args()
    if Path(a.index).exists():raise FileExistsError('Refusing to overwrite cache index; select a new --index')
    rows=load_manifest(a.manifest,allow_weak=True);idx={}
    for r in rows:
        key=r.get('video') or r.get('audio')
        try:
            idx[key]=extract_one(r,a.out_dir,a.pose_weights,audio_events=a.audio_events)
            print('CACHED',key)
        except Exception as e:print('FAILED',key,repr(e))
    out=Path(a.index);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(idx,indent=2))
    print('Cached',len(idx),'of',len(rows))
