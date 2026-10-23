"""Versioned frozen ST-GCN encoder plus independently learned fight readouts.

The onset head is the unmodified historical baseline. All weights live in one
checkpoint; no runtime dependency on external experiment checkpoint paths.
"""
import torch
from torch import nn

from src.advanced_stgcn import AdvancedSTGCN
from src.stgcn_baseline import JointOnlySTGCN


FEATURE_GROUPS = dict(joint=(0, 64), bone=(64, 128), velocity=(128, 192), geometry=(192, 224))


def feature_mask(groups):
    if not groups or not set(groups) <= set(FEATURE_GROUPS):
        raise ValueError('Known nonempty feature groups required')
    mask = torch.zeros(224)
    for group in groups:
        a, b = FEATURE_GROUPS[group]
        mask[a:b] = 1
    return mask


def normalize_features(features, mean, scale, mask):
    return ((features-mean)/scale.clamp_min(1e-4))*mask


class OptimizedReadoutC3(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AdvancedSTGCN()
        self.onset = JointOnlySTGCN()
        self.register_buffer('feature_mean', torch.zeros(224))
        self.register_buffer('feature_scale', torch.ones(224))
        self.register_buffer('feature_mask', torch.ones(224))
        self.register_buffer('history_frames', torch.tensor(48))
        self.register_buffer('readout_weight', torch.zeros(3, 224))
        self.register_buffer('readout_bias', torch.zeros(3))
        self.requires_grad_(False)

    def active_from_features(self, features):
        x = normalize_features(features, self.feature_mean, self.feature_scale, self.feature_mask)
        probability = (x@self.readout_weight.T+self.readout_bias).sigmoid().mean(1)
        return torch.logit(probability.clamp(1e-7, 1-1e-7))

    def forward(self, sk, temporal_mask, joint_mask, frame_times):
        features = self.backbone.window_features(sk, temporal_mask, joint_mask, frame_times, int(self.history_frames))
        original = self.onset(sk, temporal_mask, joint_mask)
        logits = torch.stack((original['logits'][:, 0], self.active_from_features(features)), 1)
        return dict(logits=logits, weak_onset_logits=logits[:, 0], fight_logits=logits[:, 1],
                    valid_window=original['valid_window'])
