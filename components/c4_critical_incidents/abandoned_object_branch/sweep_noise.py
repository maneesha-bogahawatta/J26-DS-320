"""Robustness check: add +-amp grey-level uniform noise to every decoded frame (mimics colour-conversion
differences between OpenCV/ffmpeg builds) and see whether the verdict per clip changes."""
import json, numpy as np, cv2
import static_object as so
M=[(0.0,0.72,0.62,1.0)]
REAL = cv2.VideoCapture
class Cap:
    def __init__(self, path, seed, amp):
        self.c=REAL(path); self.rng=np.random.default_rng(seed); self.amp=amp
    def read(self):
        ok,f=self.c.read()
        if ok and self.amp>0:
            f=np.clip(f.astype(np.float32)+self.rng.uniform(-self.amp,self.amp,f.shape),0,255).astype(np.uint8)
        return ok,f
    def get(self,k): return self.c.get(k)
    def release(self): self.c.release()
gt=json.load(open("object_gt.json"))["scenarios"]
res={}
for name,g in gt.items():
    for amp in (0.0,0.5,1.0,2.0):
        for seed in ((0,) if amp==0 else (1,2,3)):
            cv2.VideoCapture=lambda p,s=seed,a=amp: Cap(p,s,a)
            for T in (5.0,10.0):
                r=so.run(f"raw/{g['video']}", mask_rects=M, t_alert=T)
                o=[x for x in r['objects'] if abs(x['xy'][0]-g['xy'][0])<=25 and abs(x['xy'][1]-g['xy'][1])<=25 and abs(x['t_static']-g['place'])<=6]
                other=[x for x in r['objects'] if x['t_alert'] is not None and x not in o]
                res.setdefault(name,{}).setdefault(str(T),{}).setdefault(str(amp),[]).append(dict(alert=[x['t_alert'] for x in o], other_alerts=len(other)))
            cv2.VideoCapture=REAL
        print(name,"amp",amp,"done",flush=True)
json.dump(res,open("noise_sweep.json","w"),indent=1)
print("ALL DONE")
