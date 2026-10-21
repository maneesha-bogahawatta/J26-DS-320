"""Body-relative joint/bone/past-velocity ST-GCN with symmetric person fusion.

Reuses the existing ST-GCN blocks; old models/checkpoints are unchanged. Each
forward scores only its supplied trailing window. No future input is requested.
"""
import torch
from torch import nn

from c3_model import STGCNStream, COCO_PARENT, build_adj


def replace_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            groups = 4 if child.num_features % 4 == 0 else 1
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_batch_norm(child)


def pose_streams(sk, mask, times):
    """Per-frame normalization and backward differences: no lookahead imputation."""
    xy = sk[..., :2]
    conf = sk[..., 2:3].clamp(0, 1)*mask[..., None]
    valid = mask.any(2)  # B,T,persons
    hip_mask = mask[:, :, 11:13]
    hip = (xy[:, :, 11:13]*hip_mask[..., None]).sum(2)/hip_mask.sum(2).clamp_min(1)[..., None]
    fallback = (xy*mask[..., None]).sum(2)/mask.sum(2).clamp_min(1)[..., None]
    center = torch.where(hip_mask.any(2)[..., None], hip, fallback)
    upper = xy.masked_fill(~mask[..., None], -1e4).amax(2)
    lower = xy.masked_fill(~mask[..., None], 1e4).amin(2)
    scale = (upper-lower).amax(-1).clamp_min(.02)
    scale = torch.where(valid, scale, torch.ones_like(scale))
    relative = ((xy-center[:, :, None])/scale[:, :, None, :, None]).clamp(-3, 3)
    relative = relative*mask[..., None]
    joint = torch.cat((relative, conf), -1)
    parents = torch.tensor(COCO_PARENT, device=sk.device)
    bone_mask = mask & mask[:, :, parents]
    bone = torch.cat(((relative-relative[:, :, parents])*bone_mask[..., None], conf*bone_mask[..., None]), -1)
    dt = (times[:, 1:]-times[:, :-1]).to(sk.dtype).clamp_min(1/240)
    velocity = torch.zeros_like(relative)
    motion_valid = mask[:, 1:] & mask[:, :-1]
    velocity[:, 1:] = ((relative[:, 1:]-relative[:, :-1])/dt[:, :, None, None, None]).clamp(-10, 10)*motion_valid[..., None]
    vel = torch.cat((velocity/10, conf), -1)
    pair_valid = valid.all(-1)
    pair_scale = scale.mean(-1).clamp_min(.02)
    distance = (center[:, :, 0]-center[:, :, 1]).norm(dim=-1)/pair_scale
    distance = distance.clamp(0, 20)*pair_valid
    closing = torch.zeros_like(distance)
    closing[:, 1:] = ((distance[:, :-1]-distance[:, 1:])/dt).clamp(-10, 10)*(pair_valid[:, 1:] & pair_valid[:, :-1])
    denom = pair_valid.sum(1).clamp_min(1)
    minimum = distance.masked_fill(~pair_valid, 20).amin(1)
    minimum = torch.where(pair_valid.any(1), minimum, torch.zeros_like(minimum))
    wrist_motion = velocity[:, :, [9, 10]].norm(dim=-1)
    geometry = torch.stack((distance.sum(1)/denom, minimum,
        closing.clamp_min(0).sum(1)/denom, closing.amax(1),
        wrist_motion.mean((1, 2, 3)), wrist_motion.amax((1, 2, 3)),
        pair_valid.float().mean(1), mask.float().mean((1, 2, 3))), -1)
    return (joint, bone, vel), geometry, valid.any(1)


class AdvancedSTGCN(nn.Module):
    def __init__(self, embedding_dim=32, dropout=.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.streams = nn.ModuleList([STGCNStream(3, torch.tensor(build_adj(M=1)), d=embedding_dim) for _ in range(3)])
        for stream in self.streams:
            replace_batch_norm(stream)
        self.geometry = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 32), nn.GELU())
        self.fusion = nn.Sequential(nn.LayerNorm(6*embedding_dim+32),
            nn.Linear(6*embedding_dim+32, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 2))

    def encode_features(self, sk, joint_mask, times):
        """Frozen-encoder ablations reuse the exact existing pre-fusion features."""
        values, geometry, people = pose_streams(sk, joint_mask, times)
        features = []
        b, t = sk.shape[:2]
        for encoder, value in zip(self.streams, values):
            x = value.permute(0, 3, 4, 1, 2).reshape(b*2, 3, t, 17)
            z = encoder(x).reshape(b, 2, self.embedding_dim)*people[..., None]
            average = z.sum(1)/people.sum(1).clamp_min(1)[:, None]
            difference = (z[:, 0]-z[:, 1]).abs()
            features.extend((average, difference))
        return torch.cat(features+[self.geometry(geometry)], -1)

    def encode(self, sk, joint_mask, times):
        return self.fusion(self.encode_features(sk, joint_mask, times))

    def window_features(self, sk, temporal_mask, joint_mask, frame_times, frames=48):
        """Return trailing-window features; padding and earlier history are excluded."""
        if frames < 8 or frames > sk.shape[1]:
            raise ValueError('Feature history must contain 8..T frames')
        result = sk.new_zeros((len(sk), 6*self.embedding_dim+32))
        lengths = temporal_mask.sum(1).clamp_max(frames)
        for length in torch.unique(lengths):
            n = int(length)
            if not n:
                continue
            which = torch.where(lengths == length)[0]
            values = torch.stack([sk[i, temporal_mask[i]][-n:] for i in which])
            masks = torch.stack([joint_mask[i, temporal_mask[i]][-n:] for i in which])
            times = torch.stack([frame_times[i, temporal_mask[i]][-n:] for i in which])
            result = result.index_copy(0, which, self.encode_features(values, masks, times))
        return result

    def forward(self, sk, temporal_mask, joint_mask, frame_times):
        if sk.ndim != 5 or sk.shape[2:] != (17, 2, 3):
            raise ValueError('Expected skeleton[B,T,17,2,3]')
        if temporal_mask.shape != sk.shape[:2] or joint_mask.shape != sk.shape[:-1] or frame_times.shape != sk.shape[:2]:
            raise ValueError('Mask/time shape mismatch')
        result = sk.new_zeros((len(sk), 2))
        lengths = temporal_mask.sum(1)
        for length in torch.unique(lengths):
            n = int(length)
            if n == 0:
                continue
            which = torch.where(lengths == length)[0]
            values = torch.stack([sk[i, temporal_mask[i]] for i in which])
            masks = torch.stack([joint_mask[i, temporal_mask[i]] for i in which])
            times = torch.stack([frame_times[i, temporal_mask[i]] for i in which])
            logits = self.encode(values, masks, times)
            result = result.index_copy(0, which, logits)
        valid = (lengths >= 8) & (joint_mask & temporal_mask[:, :, None, None]).flatten(1).any(1)
        return dict(logits=result, weak_onset_logits=result[:, 0], fight_logits=result[:, 1], valid_window=valid)
