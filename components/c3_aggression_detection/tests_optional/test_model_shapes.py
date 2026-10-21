import torch
from c3_model import RiskNet

def test_risknet_accepts_audio_with_optional_events_and_masks():
    for d_audio in (16,24):
        model=RiskNet(16,22,d_audio,d=32)
        model.eval()
        with torch.no_grad():
            v=torch.zeros(1,3,16)
            pose=torch.zeros(1,3,22)
            a=torch.zeros(1,3,d_audio)
            mods=torch.tensor([[[1.,1.,1.],[1.,1.,0.],[1.,1.,1.]]])
            sk=torch.zeros(1,3,16,17,2,3)
            pre,fight,tto,h=model(v,pose,a,mods,sk)
            assert pre.shape==fight.shape==tto.shape==(1,3)
            assert torch.isfinite(pre).all()
