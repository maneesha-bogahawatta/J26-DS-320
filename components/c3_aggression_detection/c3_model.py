# Multi-stream ST-GCN (Joint + Bone + Velocity) + DIFEM branch + causal GRU fusion (RiskNet)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
MAX_P = 2
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16)]
COCO_PARENT = [5, 0, 0, 1, 2, 11, 5, 5, 6, 7, 8, 11, 11, 11, 12, 13, 14]     # bone = joint - parent joint

def build_adj(M=MAX_P):
    V = 17 * M; A = np.zeros((V, V), np.float32)
    for m in range(M):
        o = 17 * m
        for i, j in COCO_EDGES: A[o + i, o + j] = A[o + j, o + i] = 1
    if M == 2:
        for j in range(17): A[j, 17 + j] = A[17 + j, j] = 1                      # same joint in the other person
        for a in (0, 9, 10, 15, 16):                                              # head/wrists/ankles <-> other person's torso
            for b in (5, 6, 11, 12):
                A[a, 17 + b] = A[17 + b, a] = 1; A[17 + a, b] = A[b, 17 + a] = 1
    A = A + np.eye(V, dtype=np.float32); d = A.sum(1)
    return A / np.sqrt(d[:, None] * d[None, :])

class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, p=0.1):
        super().__init__()
        self.register_buffer('A', A); self.edge_w = nn.Parameter(torch.ones_like(A))
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(), nn.Conv2d(cout, cout, (5, 1), (stride, 1), (2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(p))
        self.res = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
    def forward(self, x):                                   # x: (N, C, T, V)
        y = torch.einsum('nctv,vw->nctw', self.gcn(x), self.A * self.edge_w)
        return F.relu(self.tcn(y) + self.res(x))

class STGCNStream(nn.Module):
    def __init__(self, cin, A, d=64):
        super().__init__()
        self.bn = nn.BatchNorm2d(cin)
        self.blocks = nn.Sequential(STGCNBlock(cin, 32, A), STGCNBlock(32, 64, A, stride=2), STGCNBlock(64, d, A, stride=2))
    def forward(self, x): return self.blocks(self.bn(x)).mean(dim=(2, 3))

class MultiStreamSTGCN(nn.Module):
    # streams: Joint (x,y,conf), Bone (child-parent vector), Velocity (frame difference)
    def __init__(self, d_out=192, M=MAX_P):
        super().__init__()
        A = torch.tensor(build_adj(M))
        self.register_buffer('parent', torch.tensor([m * 17 + p for m in range(M) for p in COCO_PARENT], dtype=torch.long))
        self.joint, self.bone, self.vel = STGCNStream(3, A), STGCNStream(3, A), STGCNStream(3, A)
        self.proj = nn.Sequential(nn.Linear(3 * 64, d_out), nn.GELU())
    def forward(self, sk):                                   # sk: (N, T, 17, M, 3)
        N, T, V, M, C = sk.shape
        x = sk.permute(0, 4, 1, 3, 2).reshape(N, C, T, M * V)
        xy, conf = x[:, :2], x[:, 2:3]
        bone = torch.cat([xy - xy[..., self.parent], conf], 1)
        vel = torch.zeros_like(xy); vel[:, :, 1:] = xy[:, :, 1:] - xy[:, :, :-1]
        vel = torch.cat([vel * 5.0, conf], 1)
        return self.proj(torch.cat([self.joint(x), self.bone(bone), self.vel(vel)], -1))

class RiskNet(nn.Module):
    def __init__(self, dv, dp, da, d=192, p=0.2):
        super().__init__()
        enc = lambda i: nn.Sequential(nn.LayerNorm(i), nn.Linear(i, d), nn.GELU(), nn.Dropout(p))
        self.ev, self.ea, self.difem = enc(dv), enc(da), enc(dp)       # difem: interaction-feature branch
        self.stgcn = MultiStreamSTGCN(d_out=d)
        self.pose_fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p))
        self.miss = nn.Parameter(torch.zeros(3, d))
        self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(p))
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True, dropout=p)       # causal
        self.h_pre, self.h_fight, self.h_tto = nn.Linear(d, 1), nn.Linear(d, 1), nn.Linear(d, 1)
    def forward(self, vis, pose, aud, mods, sk, h=None):
        B, L = vis.shape[:2]
        sk_e = self.stgcn(sk.reshape(B * L, *sk.shape[2:])).reshape(B, L, -1)
        zp = self.pose_fuse(torch.cat([self.difem(pose), sk_e], -1))
        zs = [self.ev(vis), zp, self.ea(aud)]; ms = mods.unbind(-1)
        z = [m.unsqueeze(-1) * x + (1 - m.unsqueeze(-1)) * self.miss[i] for i, (x, m) in enumerate(zip(zs, ms))]
        y, h = self.gru(self.fuse(torch.cat(z, -1)), h)
        return self.h_pre(y).squeeze(-1), self.h_fight(y).squeeze(-1), self.h_tto(y).squeeze(-1), h
