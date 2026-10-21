"""Causal supervision: fight onset != observed precursor.

A positive early label requires TWO independent facts: a verified future contact
onset within `horizon`, and a *reviewed* preceding aggression cue interval. We
mask ambiguous pre-onset time from the early loss instead of inventing cues.
"""
import numpy as np

def targets(times, intervals, horizon=3., precursors=None):
    t=np.asarray(times,np.float32)
    fight=np.zeros(len(t),np.float32)
    time_to=np.zeros(len(t),np.float32)
    pre=np.zeros(len(t),np.float32)
    pre_mask=np.ones(len(t),np.float32)
    mask=np.zeros(len(t),np.float32)
    for start,end in intervals:
        start,end=float(start),float(end)
        if start<0 or end<=start:raise ValueError(f'Bad onset interval: {(start,end)}')
        fight[(t>=start)&(t<end)]=1.
        ambiguous=(t>=max(0.,start-horizon))&(t<start)
        pre_mask[ambiguous]=0.
        # Cue observations are required for positive precursors, not just closeness to contact.
        for a,b in (precursors or []):
            a,b=float(a),float(b)
            if a<0 or b<=a:raise ValueError('Bad precursor interval')
            positive=ambiguous&(t>=a)&(t<=b)
            pre[positive]=1.
            pre_mask[positive]=1.
            delta=start-t[positive]
            time_to[positive]=delta/horizon
            mask[positive]=1.
    pre_mask[fight>0]=0.
    return dict(pre=pre,fight=fight,tto=time_to,tto_mask=mask,pre_mask=pre_mask)
