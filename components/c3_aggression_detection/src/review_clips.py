"""Generate short clips AROUND onset candidates for human labeling; requires ffmpeg."""
import argparse,csv,json,re,subprocess
from pathlib import Path

def clips(manifest,out_dir,before=5.,after=2.):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
    with open(manifest,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            for i,(start,end) in enumerate(json.loads(r['intervals'])):
                s=max(0.,float(start)-before);dur=(float(start)+after)-s
                name=re.sub(r'[^\w.-]','_',Path(r['video']).stem)
                dest=out/f'{name}_onset{i}.mp4'
                cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(s),'-i',r['video'],
                     '-t',str(dur),'-c:v','libx264','-preset','veryfast','-c:a','aac',str(dest)]
                result=subprocess.run(cmd,capture_output=True)
                if result.returncode:print('WARN clip failed:',r['video'],result.stderr.decode(errors='replace')[-300:]);continue
                rows.append(dict(clip=str(dest),original_video=r['video'],onset_s=start,clip_start_s=s,
                                 seconds_before=round(float(start)-s,3),reviewed='false'))
    with (out/'clip_index.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,['clip','original_video','onset_s','clip_start_s','seconds_before','reviewed']);w.writeheader();w.writerows(rows)
    print('Review clips:',len(rows),'at',out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',default='data/review_clips');a=p.parse_args()
    clips(a.manifest,a.out)
