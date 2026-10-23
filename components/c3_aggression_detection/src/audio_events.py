"""Optional pretrained sound-event cues for AV risk fusion.

Screaming/yelling can occur during sport, celebration, or accidents: scores are
context features, never standalone allegations of imminent violence. YAMNet is
optional and is downloaded (on first call) only when explicitly enabled.
"""
import csv
import numpy as np

CUE_NAMES=('Screaming','Yell','Shout','Crying, sobbing','Cheering','Laughter','Music','Speech')

class YAMNetCues:
    def __init__(self):
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
        except ImportError as e:
            raise RuntimeError('Optional YAMNet requires tensorflow and tensorflow-hub; see README') from e
        self.tf=tf
        self.model=hub.load('https://tfhub.dev/google/yamnet/1')
        with tf.io.gfile.GFile(self.model.class_map_path().numpy().decode()) as f:
            table=list(csv.DictReader(f))
        mp={row['display_name']:int(row['index']) for row in table}
        self.ix=[mp.get(name,-1) for name in CUE_NAMES]
        missing=[name for name,j in zip(CUE_NAMES,self.ix) if j<0]
        if missing:raise RuntimeError(f'Missing YAMNet event class(es): {missing}')
    def __call__(self,wave,sr=16000):
        if sr!=16000:raise ValueError('YAMNet expects 16 kHz mono waveform')
        x=np.asarray(wave,dtype=np.float32).reshape(-1)
        if not len(x):return np.zeros(len(CUE_NAMES),np.float32)
        # Last 1s of past audio; pad leading silence, never read future audio.
        x=x[-16000:]
        if len(x)<16000:x=np.pad(x,(16000-len(x),0))
        scores,_,_=self.model(self.tf.convert_to_tensor(x,dtype=self.tf.float32))
        values=scores.numpy()
        return np.asarray(values[:,self.ix].max(axis=0),np.float32)

def make_detector(backend='none'):
    if backend=='none':return None
    if backend=='yamnet':return YAMNetCues()
    raise ValueError('Audio event backend must be none or yamnet')

def cue_report(feat):
    """Diagnostic model outputs only; NOT ground truth or aggression probabilities."""
    f=np.asarray(feat).reshape(-1)
    if len(f)<16+len(CUE_NAMES):return {}
    return {name.lower().replace(', sobbing','').replace(' ','_'):round(float(v),4)
            for name,v in zip(CUE_NAMES,f[16:16+len(CUE_NAMES)])}
