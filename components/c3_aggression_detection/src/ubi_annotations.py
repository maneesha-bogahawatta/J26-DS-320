"""Prepare *candidate*, never automatically verified, UBI-Fights intervals.

UBI-Fights frame-label encoding is intentionally explicit. Use --format frame_binary
for a 0/1 per-frame vector, --format interval_frames for start/end frame pairs,
or --format interval_seconds for start/end seconds. Inspect the dataset files and
choose the correct format: automatic guesses can silently corrupt timestamps.
"""
import argparse, csv, json, re, subprocess
from pathlib import Path
import numpy as np

VIDEO_EXT={'.mp4','.avi','.mov','.mkv','.webm'}

def has_audio_stream(p):
    try:
        result=subprocess.run(['ffprobe','-v','error','-select_streams','a:0',
           '-show_entries','stream=codec_type','-of','csv=p=0',str(p)],capture_output=True,text=True,timeout=12)
        return result.returncode==0 and 'audio' in result.stdout
    except (OSError,subprocess.TimeoutExpired):
        return False  # Unknown treated as unavailable; do not invent audio.


def probe_video(p):
    try:
        import cv2
        cap=cv2.VideoCapture(str(p))
        if not cap.isOpened():raise RuntimeError('Cannot open video')
        fps=float(cap.get(cv2.CAP_PROP_FPS));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps<=0 or n<=0:raise ValueError('Invalid video FPS or frame count')
        return n,fps,n/fps
    except ImportError:
        raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
             'stream=avg_frame_rate,nb_frames,duration','-of','json',str(p)])
        data=json.loads(raw)['streams'][0];parts=data['avg_frame_rate'].split('/')
        fps=float(parts[0])/float(parts[1]);n=int(data.get('nb_frames') or round(float(data['duration'])*fps))
        return n,fps,n/fps

def read_text_numbers(p):
    lines=[]
    for s in Path(p).read_text(encoding='utf-8-sig').splitlines():
        s=s.split('#',1)[0].strip()
        if s:lines.append([float(x) for x in re.split(r'[,;\s]+',s) if x])
    return lines

def binary_to_intervals(labels,fps):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.all(np.isin(labels,[0,1])):raise ValueError('Expected a 0/1 label per frame')
    padded=np.r_[0,labels.astype(int),0];starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
    return [[round(int(a)/fps,6),round(int(b)/fps,6)] for a,b in zip(starts,ends)]

def load_annotation(p,format,nframes,fps):
    if format=='npy_binary':
        arr=np.load(p,allow_pickle=False).reshape(-1)
        if abs(len(arr)-nframes)>1:raise ValueError(f'Frame label count mismatch {len(arr)} != {nframes}')
        return binary_to_intervals(arr,fps)
    lines=read_text_numbers(p)
    if format=='frame_binary':
        vals=[v for line in lines for v in line]
        if abs(len(vals)-nframes)>1:raise ValueError(f'Frame label count mismatch {len(vals)} != {nframes}')
        return binary_to_intervals(vals,fps)
    if format in ('interval_frames','interval_seconds'):
        if any(len(line)!=2 for line in lines):raise ValueError('Expected exactly two numbers (start,end) per row')
        out=[]
        for a,b in lines:
            if format=='interval_frames':
                if not (a.is_integer() and b.is_integer()):raise ValueError('Frame indices must be integers')
                if a<0 or b>nframes+1:raise ValueError('Frame indices out of bounds')
                a,b=a/fps,b/fps  # end is exclusive, convert for inclusive frame sets before use
            if a<0 or b<=a or b>nframes/fps+1/fps+1e-3:raise ValueError('Invalid interval bounds')
            out.append([round(a,6),round(b,6)])
        return out
    raise ValueError('Unknown format')

def make_candidates(video_root,annotation_root,out,fmt='frame_binary',test_list=None,min_context=3.0):
    video_root=Path(video_root).resolve();annotation_root=Path(annotation_root).resolve()
    videos=sorted(p for p in video_root.rglob('*') if p.suffix.lower() in VIDEO_EXT)
    if not videos:raise ValueError(f'No video files found in {video_root}')
    test_stems=set()
    if test_list:
        text=Path(test_list).read_text(encoding='utf-8-sig')
        test_stems={Path(s.split(',')[0].strip().strip('"')).stem.casefold() for s in text.splitlines() if s.strip()}
    anns={}
    allowed={'.npy'} if fmt=='npy_binary' else {'.txt','.csv'}
    for p in annotation_root.rglob('*'):
        if p.suffix.lower() in allowed:
            # Support F_123.txt and F_123.mp4.txt, without guessing arbitrary names.
            ann_key=Path(p.stem).stem.casefold()
            anns.setdefault(ann_key,[]).append(p)
    fields=['video','audio','source','group','intervals','annotation_status','has_audio','precursors',
            'split','duration_s','source_annotation','reviewer_a','reviewer_b','decision','onset_override_s',
            'notes','audio_cue_review','precontext_s']
    records=[];errors=[]
    for p in videos:
        try:
            n,fps,duration=probe_video(p)
            key=p.stem.casefold();matching=anns.get(key,[])
            if len(matching)>1:raise ValueError('Ambiguous matching annotation files')
            if matching:intervals=load_annotation(matching[0],fmt,n,fps)
            elif p.stem.lower().startswith('n_') or 'normal' in [x.lower() for x in p.parts]:intervals=[]
            else:raise ValueError('Fight video missing annotation; not safe to assume negative')
            prec_s=max((min_context if start>=min_context else start) for start,_ in intervals) if intervals else ''
            records.append(dict(video=str(p),audio='',source='ubi_fights',group='ubi_'+p.stem,
                         intervals=json.dumps(intervals),annotation_status='weak',
                         has_audio='true' if has_audio_stream(p) else 'false',
                         precursors='[]',split='test' if key in test_stems else '',duration_s=round(duration,3),
                         source_annotation=str(matching[0]) if matching else 'normal_name_candidate',
                         reviewer_a='',reviewer_b='',decision='',onset_override_s='',notes='',
                         audio_cue_review='',precontext_s=prec_s))
        except Exception as e:errors.append((str(p),str(e)))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(records)
    err=out.with_suffix('.errors.csv')
    with err.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['video','error']);w.writerows(errors)
    print(f'Wrote {len(records)} UNVERIFIED annotation candidates to {out}; errors={len(errors)} at {err}')
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--format',choices=['frame_binary','npy_binary','interval_frames','interval_seconds'],required=True)
    p.add_argument('--test-list');p.add_argument('--out',default='data/manifests/ubi_review.csv')
    a=p.parse_args();make_candidates(a.videos,a.annotations,a.out,a.format,a.test_list)
