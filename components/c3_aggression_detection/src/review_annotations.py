"""Convert *independently reviewed* UBI candidates into verified training manifest.

Fill reviewer_a, reviewer_b, decision (approve/reject), and precursors (JSON pairs
in seconds, or [] when no observed precursors). Both reviewers must agree about
the physical-contact onset and audio/visual evidence. No automatic approvals.
For positives, onset_override_s can be a JSON list of revised [start,end] pairs.
"""
import argparse,csv,json
from pathlib import Path

def _intervals(s,duration,kind):
    v=json.loads(s or '[]')
    if not isinstance(v,list):raise ValueError(f'{kind} must be a JSON list')
    for q in v:
        if not isinstance(q,list) or len(q)!=2:raise ValueError(f'{kind} requires [[start,end],...]')
        x,y=map(float,q)
        if x<0 or x>=y or y>duration+0.1:raise ValueError(f'{kind} out of bounds {q}')
    return v

def finalize(input_csv,output_csv,require_precursor=True):
    accepted=[];rejected=[]
    with open(input_csv,encoding='utf-8-sig',newline='') as f:
        for n,r in enumerate(csv.DictReader(f),start=2):
            try:
                dec=r.get('decision','').strip().lower();reviewers=[r.get('reviewer_a','').strip(),r.get('reviewer_b','').strip()]
                if dec not in ('approve','reject'):raise ValueError('Requires approve or reject decision')
                if not all(reviewers) or reviewers[0].casefold()==reviewers[1].casefold():raise ValueError('Two distinct reviewers required')
                if dec=='reject':continue
                duration=float(r['duration_s']);onset=r.get('onset_override_s','').strip() or r['intervals']
                intervals=_intervals(onset,duration,'intervals');cues=_intervals(r.get('precursors','[]'),duration,'precursors')
                if intervals:
                    if require_precursor and not cues:raise ValueError('Positive sample needs reviewed precursor interval; use --allow-no-precursor to include only active fight supervision')
                    for x,y in cues:
                        if not any(float(y)<=float(start) and float(y)>float(start)-6.1 for start,_ in intervals):
                            raise ValueError('Each precursor interval must end at/before an onset and within 6 seconds')
                    status='verified'
                else:
                    if cues:raise ValueError('Negative video must not have precursors')
                    status='negative_verified'
                output={k:r.get(k,'') for k in ('video','audio','source','group','has_audio','split')}
                output.update(intervals=json.dumps(intervals),precursors=json.dumps(cues),annotation_status=status,
                              reviewer_a=reviewers[0],reviewer_b=reviewers[1],annotation_source=r.get('source_annotation',''),
                              review_notes=r.get('notes',''),audio_cue_review=r.get('audio_cue_review',''))
                accepted.append(output)
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:rejected.append((n,str(e)))
    if not accepted:raise ValueError('No approved verified examples. Review your candidate CSV first.')
    out=Path(output_csv);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(accepted[0]));w.writeheader();w.writerows(accepted)
    print(f'Verified manifest rows={len(accepted)}; rejected or incomplete={len(rejected)}')
    for line,err in rejected:print(f'  line {line}: {err}')
    return accepted,rejected

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed',required=True);p.add_argument('--out',required=True)
    p.add_argument('--allow-no-precursor',action='store_true')
    a=p.parse_args();finalize(a.reviewed,a.out,not a.allow_no_precursor)
