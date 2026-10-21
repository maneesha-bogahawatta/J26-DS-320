"""Joint-only interface around C3's existing custom two-person ST-GCN blocks.

Scores classify an entire trailing window at its cutoff. Internal symmetric
convolutions stay inside that available window; no framewise causality is claimed.
"""
import torch
from torch import nn
from c3_model import STGCNStream, build_adj


class JointOnlySTGCN(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoder=STGCNStream(3,torch.tensor(build_adj(M=2)),d=embedding_dim)
        self.heads=nn.Linear(embedding_dim,2)

    def forward(self,sk,temporal_mask=None,joint_mask=None):
        if sk.ndim!=5 or sk.shape[2:]!=(17,2,3):
            raise ValueError('Expected skeletons[B,T,17,2,3]')
        b,t=sk.shape[:2]
        if temporal_mask is None:
            temporal_mask=torch.ones((b,t),dtype=torch.bool,device=sk.device)
        if temporal_mask.shape!=(b,t):
            raise ValueError('Temporal mask must be [B,T]')
        if joint_mask is None:
            joint_mask=sk[...,2]>0
        if joint_mask.shape!=(b,t,17,2):
            raise ValueError('Joint mask must be [B,T,17,2]')
        # Remove padding before BatchNorm/convolution/pooling. Masked pixel values
        # and left-padding length cannot change an otherwise identical sample.
        lengths=temporal_mask.sum(1)
        embeddings=sk.new_zeros((b,self.heads.in_features))
        for length in torch.unique(lengths):
            n=int(length)
            if not n:
                continue
            which=torch.nonzero(lengths==length).flatten()
            usable=[]
            for i in which:
                x=sk[i,temporal_mask[i]]
                mask=joint_mask[i,temporal_mask[i]]
                usable.append(torch.where(mask[...,None],x,torch.zeros_like(x)))
            x=torch.stack(usable).permute(0,4,1,3,2).reshape(len(which),3,n,34)
            embeddings=embeddings.index_copy(0,which,self.encoder(x))
        logits=self.heads(embeddings)
        return {'precursor_logits':logits[:,0],'fight_logits':logits[:,1],
                'logits':logits,'valid_window':(lengths>0)&(joint_mask&temporal_mask[:,:,None,None]).flatten(1).any(1)}
