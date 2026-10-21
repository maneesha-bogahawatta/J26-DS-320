"""Streaming abandoned-object monitor (same algorithm as static_object.py, but frame by frame).

Feed it every frame with a timestamp; it subsamples to `work_fps`, learns a background from the
first `t_init` seconds, then tracks static objects and an unattended timer. Returns alert dicts.
Validated only on the overhead CAVIAR lobby clips; other camera views need the size parameters retuned.
"""
import cv2
import numpy as np


class StaticObjectMonitor:
    def __init__(self, work_fps=5.0, width=384, mask_rects=(), t_init=8.0, t_static=3.0,
                 t_alert=10.0, radius_frac=0.16, min_area=18, max_area=260, person_area=300,
                 th_bg=28, th_stab=14, min_life=6.0, max_touch=0.25, t_remove=1.5):
        self.dt = 1.0 / work_fps
        self.width, self.mask_rects = width, tuple(mask_rects)
        self.t_init, self.t_static, self.t_alert = t_init, t_static, t_alert
        self.radius_frac, self.min_area, self.max_area, self.person_area = radius_frac, min_area, max_area, person_area
        self.th_bg, self.th_stab, self.min_life, self.max_touch, self.t_remove = th_bg, th_stab, min_life, max_touch, t_remove
        self.k_init = max(3, int(t_init / self.dt))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._last_t = -1e9
        self.bg = None
        self.buf = []
        self.objs, self.nxt = [], 0
        self.prev = None
        self.count = None
        self.scale = 1.0
        self.status = "warming up"

    # ---------------------------------------------------------------- public
    def feed(self, frame_bgr, t):
        """Call for every frame. Returns a list of new alert dicts (usually empty)."""
        if t - self._last_t < self.dt * 0.999:
            return []
        self._last_t = t
        g = self._prep(frame_bgr)
        if self.bg is None:
            self.buf.append((g, t))
            if len(self.buf) < self.k_init:
                self.status = f"learning background {len(self.buf) * self.dt:.0f}/{self.t_init:.0f}s"
                return []
            self.bg = np.median(np.stack([b[0] for b in self.buf]), axis=0).astype(np.float32)
            self.count = np.zeros(self.bg.shape, np.int32)
            self.prev = self.buf[0][0]
            out = []
            for gg, tt in self.buf:           # replay the warm-up frames (same as the batch version)
                out += self._step(gg, tt)
            self.buf = []
            self.status = "monitoring"
            return out
        return self._step(g, t)

    def tracked(self):
        """Objects worth drawing: [{id, bbox(px in original frame), state, unattended_s}]."""
        out = []
        for o in self.objs:
            if o['removed'] is not None or not self._valid(o, self._last_t):
                continue
            x, y, w, h = o['bbox']
            s = self.scale
            un = max(0.0, self._last_t - o['last_person_t'])
            state = 'alert' if o['alerted'] is not None else ('unattended' if un > 1.0 else 'static')
            out.append(dict(id=o['id'], bbox=[x * s, y * s, (x + w) * s, (y + h) * s], state=state, unattended_s=un))
        return out

    # --------------------------------------------------------------- internal
    def _prep(self, f):
        h0, w0 = f.shape[:2]
        self.scale = w0 / self.width
        f = cv2.resize(f, (self.width, int(round(h0 / self.scale))), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        H, W = g.shape
        for (rx0, ry0, rx1, ry1) in self.mask_rects:
            g[int(ry0 * H):int(ry1 * H), int(rx0 * W):int(rx1 * W)] = 0
        return g

    def _valid(self, o, t):
        life = t - o['t_static']
        touch = o.get('touch', 0) / max(1, o.get('frames', 1))
        return life >= self.min_life and touch <= self.max_touch and o['t_static'] >= self.t_init + 1.0

    def _step(self, g, t):
        H, W = g.shape
        R = self.radius_frac * W
        d_bg = np.abs(g - self.bg) > self.th_bg
        d_pr = np.abs(g - self.prev) > self.th_stab
        moving = cv2.morphologyEx(d_pr.astype(np.uint8), cv2.MORPH_CLOSE, self.kernel)
        moving = cv2.dilate(moving, self.kernel, iterations=2)
        if d_bg.mean() > 0.25 or d_pr.mean() > 0.25:     # exposure flicker / camera shake
            self.count[:] = 0
            self.prev = g
            return []
        stable = d_bg & (~d_pr)
        self.count = np.where(stable, self.count + 1, 0)
        static = (self.count >= int(self.t_static / self.dt)).astype(np.uint8)
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, self.kernel)
        n_c, lab, stats, cen = cv2.connectedComponentsWithStats(static, 8)
        cands, persons = [], []
        for c in range(1, n_c):
            a = stats[c, cv2.CC_STAT_AREA]
            if a >= self.person_area:
                persons.append(tuple(cen[c]))
            elif self.min_area <= a <= self.max_area:
                cands.append((tuple(cen[c]), int(a), tuple(int(v) for v in stats[c, :4])))
        live = [o for o in self.objs if o['removed'] is None]
        persons = [p for p in persons if not any(np.hypot(p[0] - o['xy'][0], p[1] - o['xy'][1]) <= 10 for o in live)]
        n_m, _, st_m, cen_m = cv2.connectedComponentsWithStats(moving, 8)
        mov_c = [tuple(cen_m[c]) for c in range(1, n_m) if st_m[c, cv2.CC_STAT_AREA] >= 40]
        persons += mov_c
        used = set()
        for o in self.objs:
            best, bd = None, 1e9
            for j, (c, a, bb) in enumerate(cands):
                d = np.hypot(c[0] - o['xy'][0], c[1] - o['xy'][1])
                if d < bd and j not in used:
                    best, bd = j, d
            if best is not None and bd <= 6:
                used.add(best)
                o['xy'] = cands[best][0]; o['bbox'] = cands[best][2]
        for j, (c, a, bb) in enumerate(cands):
            if j in used:
                continue
            self.objs.append(dict(id=self.nxt, xy=c, bbox=bb, t_static=t - self.t_static, last_person_t=t,
                                  alerted=None, emitted=False, removed=None, area=a, touch=0, frames=0,
                                  near_hist=[], gone_cnt=0))
            self.nxt += 1
        events = []
        for o in self.objs:
            if o['removed'] is not None:
                continue
            o['frames'] += 1
            if any(np.hypot(p[0] - o['xy'][0], p[1] - o['xy'][1]) <= 12 for p in mov_c):
                o['touch'] += 1
            near_now = any(np.hypot(p[0] - o['xy'][0], p[1] - o['xy'][1]) <= R for p in persons)
            o['near_hist'].append(near_now); del o['near_hist'][:-3]
            if sum(o['near_hist']) >= 2:                  # debounce: >= 2 of the last 3 frames
                o['last_person_t'] = t
            x, y, w, h = o['bbox']
            reg = d_bg[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1]
            frac = float(reg.mean()) if reg.size else 0.0
            o['gone_cnt'] = o['gone_cnt'] + 1 if frac < 0.15 else 0
            if o['gone_cnt'] * self.dt >= self.t_remove:   # pixels went back to the background
                o['removed'] = t - self.t_remove
            elif o['alerted'] is None and (t - o['last_person_t']) >= self.t_alert:
                o['alerted'] = t
            if o['alerted'] is not None and not o['emitted'] and self._valid(o, t):
                o['emitted'] = True
                s = self.scale
                events.append(dict(t=t, object_id=o['id'], bbox=[x * s, y * s, (x + w) * s, (y + h) * s],
                                   unattended_s=t - o['last_person_t'], placed_t=o['t_static'],
                                   touch_frac=o['touch'] / max(1, o['frames'])))
        self.prev = g
        return events
