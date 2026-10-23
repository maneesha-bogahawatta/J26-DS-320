"""Causal, masked body-relative motion features with two learned weak-label heads.

Features describe motion and geometry, never rule-based violence labels. Each
prediction sees only the supplied trailing window. Missing joints are excluded
from reductions and both person orders have the same representation.
"""
import torch
from torch import nn

FEATURE_DIM = 240  # 24 channels: mean/max/std over 1/3/6 seconds + current value


def _reduce(values, valid):
    axes = tuple(range(2, values.ndim))
    count = valid.sum(axes)
    mean = torch.where(valid, values, 0.).sum(axes) / count.clamp_min(1)
    maximum = values.masked_fill(~valid, -1e6).amax(axes)
    return mean, torch.where(count > 0, maximum, 0.), count > 0


def frame_channels(sk, temporal_mask, joint_mask, times):
    if sk.ndim != 5 or sk.shape[2:] != (17, 2, 3):
        raise ValueError('Expected skeleton[B,T,17,2,3]')
    if temporal_mask.shape != sk.shape[:2] or joint_mask.shape != sk.shape[:-1] or times.shape != sk.shape[:2]:
        raise ValueError('Mask/time shape mismatch')
    mask = joint_mask & temporal_mask[:, :, None, None]
    xy = torch.where(mask[..., None], sk[..., :2], 0.)
    if not torch.isfinite(xy).all() or not torch.isfinite(times[temporal_mask]).all():
        raise ValueError('Observed coordinates and frame times must be finite')
    hips = mask[:, :, 11:13]
    hip = (xy[:, :, 11:13] * hips[..., None]).sum(2) / hips.sum(2).clamp_min(1)[..., None]
    fallback = xy.sum(2) / mask.sum(2).clamp_min(1)[..., None]
    center = torch.where(hips.any(2)[..., None], hip, fallback)
    high = xy.masked_fill(~mask[..., None], -1e4).amax(2)
    low = xy.masked_fill(~mask[..., None], 1e4).amin(2)
    scale = (high - low).amax(-1).clamp_min(.02)
    person = mask.sum(2) >= 6
    scale = torch.where(person, scale, 1.)
    relative = torch.where(mask[..., None], (xy - center[:, :, None]) / scale[:, :, None, :, None], 0.).clamp(-3, 3)
    dt = times[:, 1:] - times[:, :-1]
    consecutive = temporal_mask[:, 1:] & temporal_mask[:, :-1]
    if (dt[consecutive] <= 0).any():
        raise ValueError('Observed timestamps must increase')
    step = consecutive & (dt > 0) & (dt <= .5)
    safe_dt = torch.where(step, dt, 1.)
    velocity = torch.zeros_like(relative)
    motion_mask = torch.zeros_like(mask)
    motion_mask[:, 1:] = mask[:, 1:] & mask[:, :-1] & step[:, :, None, None]
    velocity[:, 1:] = (relative[:, 1:] - relative[:, :-1]) / safe_dt[:, :, None, None, None]
    speed = velocity.norm(dim=-1).clamp(max=10.) / 10.
    values, known = [], []

    def add_pair(value, valid):
        mean, maximum, usable = _reduce(value, valid)
        values.extend((mean, maximum)); known.extend((usable, usable))

    for joints in (list(range(17)), [9, 10], [7, 8], [15, 16]):
        add_pair(speed[:, :, joints], motion_mask[:, :, joints])
    for a, b in (([9, 10], [5, 6]), ([9, 10], [11, 12]), ([15, 16], [11, 12])):
        distance = (relative[:, :, a] - relative[:, :, b]).norm(dim=-1)
        add_pair(distance, mask[:, :, a] & mask[:, :, b])
    height = relative[:, :, [5, 6], :, 1] - relative[:, :, [9, 10], :, 1]
    add_pair(height, mask[:, :, [5, 6]] & mask[:, :, [9, 10]])

    pair_valid = person.all(-1)
    pair_scale = scale.mean(-1).clamp_min(.02)
    delta = center[:, :, 0] - center[:, :, 1]
    distance = (delta.norm(dim=-1) / pair_scale).clamp(max=20.) / 20.
    values.append(distance); known.append(pair_valid)
    pair_step = pair_valid[:, 1:] & pair_valid[:, :-1] & step
    closing = torch.zeros_like(distance)
    closing[:, 1:] = (distance[:, :-1] - distance[:, 1:]) / safe_dt
    closing_known = torch.zeros_like(pair_valid); closing_known[:, 1:] = pair_step
    values.extend((closing.clamp(0, 1), (-closing).clamp(0, 1)))
    known.extend((closing_known, closing_known))

    def contact(a, b):
        directions, available = [], []
        for p, q in ((0, 1), (1, 0)):
            diff = xy[:, :, a, p, None, :] - xy[:, :, None, b, q, :]
            valid = mask[:, :, a, p, None] & mask[:, :, None, b, q] & pair_valid[:, :, None, None]
            d = (diff.norm(dim=-1) / pair_scale[:, :, None, None]).clamp(max=20.) / 20.
            directions.append(d.masked_fill(~valid, 1e6).amin((2, 3)))
            available.append(valid.any((2, 3)))
        valid = torch.stack(available, -1).any(-1)
        value = torch.stack(directions, -1).amin(-1)
        return torch.where(valid, value, 0.), valid

    for a, b in (([9, 10], [0, 1, 2, 3, 4]), ([9, 10], [9, 10]), ([9, 10], [11, 12]), (list(range(17)), list(range(17)))):
        value, valid = contact(a, b); values.append(value); known.append(valid)
    pair_speed = torch.zeros_like(distance)
    pair_speed[:, 1:] = ((delta[:, 1:] - delta[:, :-1]).norm(dim=-1) / safe_dt / pair_scale[:, 1:]).clamp(max=10.) / 10.
    values.append(pair_speed); known.append(closing_known)
    return torch.stack(values, -1), torch.stack(known, -1)


def causal_features(sk, temporal_mask, joint_mask, frame_times):
    values, known = frame_channels(sk, temporal_mask, joint_mask, frame_times)
    # Padded coordinates and padded timestamps cannot alter the selected history.
    cutoff = frame_times.masked_fill(~temporal_mask, -1e6).amax(1)
    safe_times = torch.where(temporal_mask, frame_times, cutoff[:, None])
    age = cutoff[:, None] - safe_times
    features = []
    for horizon in (1., 3., 6.):
        valid = known & (age <= horizon + 1e-7)[:, :, None]
        count = valid.sum(1)
        mean = torch.where(valid, values, 0.).sum(1) / count.clamp_min(1)
        variance = torch.where(valid, (values - mean[:, None]).square(), 0.).sum(1) / count.clamp_min(1)
        maximum = values.masked_fill(~valid, -1e6).amax(1)
        features.extend((mean, torch.where(count > 0, maximum, 0.), variance.clamp_min(0).sqrt()))
    current = known & temporal_mask[:, :, None] & (age.abs() < 1e-6)[:, :, None]
    features.append(torch.where(current, values, 0.).sum(1))
    result = torch.cat(features, -1)
    if result.shape[1] != FEATURE_DIM or not torch.isfinite(result).all():
        raise ValueError('Invalid kinematic feature vector')
    return result


class KinematicTemporalModel(nn.Module):
    def __init__(self, hidden=0):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(FEATURE_DIM))
        self.register_buffer('feature_scale', torch.ones(FEATURE_DIM))
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(), nn.Dropout(.25), nn.Linear(hidden, 1))
            if hidden else nn.Linear(FEATURE_DIM, 1) for _ in range(2)])

    def feature_logits(self, features):
        normalized = ((features - self.feature_mean) / self.feature_scale.clamp_min(.01)).clamp(-5, 5)
        return torch.cat([head(normalized) for head in self.heads], -1)

    def forward(self, sk, temporal_mask, joint_mask, frame_times):
        logits = self.feature_logits(causal_features(sk, temporal_mask, joint_mask, frame_times))
        valid = (temporal_mask.sum(1) >= 8) & (joint_mask & temporal_mask[:, :, None, None]).flatten(1).any(1)
        return dict(logits=logits, weak_onset_logits=logits[:, 0], fight_logits=logits[:, 1], valid_window=valid)
