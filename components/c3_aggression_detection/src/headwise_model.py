"""Positive-weight, per-task ensemble of existing causal C3 model families."""
import torch
from torch import nn

from src.stgcn_baseline import JointOnlySTGCN
from src.advanced_stgcn import AdvancedSTGCN
from src.kinematic_model import KinematicTemporalModel


class HeadwiseC3(nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline = JointOnlySTGCN()
        self.advanced = AdvancedSTGCN()
        self.kinematic = KinematicTemporalModel()
        self.register_buffer('blend_weights', torch.tensor([[1., 0., 0.], [1., 0., 0.]]))

    def forward(self, sk, temporal_mask, joint_mask, frame_times):
        if (self.blend_weights < 0).any() or not torch.allclose(self.blend_weights.sum(1), self.blend_weights.new_ones(2)):
            raise ValueError('Each task requires nonnegative model weights summing to one')
        outputs = [self.baseline(sk, temporal_mask, joint_mask),
            self.advanced(sk, temporal_mask, joint_mask, frame_times),
            self.kinematic(sk, temporal_mask, joint_mask, frame_times)]
        probabilities = torch.stack([out['logits'].sigmoid() for out in outputs], -1)
        probabilities = (probabilities * self.blend_weights[None]).sum(-1).clamp(1e-6, 1-1e-6)
        logits = torch.logit(probabilities)
        return dict(logits=logits, weak_onset_logits=logits[:, 0], fight_logits=logits[:, 1],
            valid_window=outputs[0]['valid_window'] & (temporal_mask.sum(1) >= 8))
