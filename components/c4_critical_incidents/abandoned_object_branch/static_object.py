"""Classical abandoned-object baseline for fixed cameras (Component 4, object branch).

Idea (matches Fig 4.4 states: carried -> placed/static -> owner away -> abandoned):
  1. Slow background  B_slow  (long memory)   -> what the scene looked like before.
  2. Persistence map: pixels that differ from B_slow AND have stopped changing frame-to-frame
     are 'static foreground'; a counter says how long each pixel has been static.
  3. Connected components of the static mask, size-filtered, are STATIC OBJECT candidates.
     Big static blobs (a person standing still) count as 'person present', not as objects.
  4. A person is 'near' an object if a moving blob (or big static blob) is within R pixels.
  5. Unattended timer: seconds since a person was last near the object. Alert when >= T_ALERT.
  6. If the candidate disappears before the alert -> 'removed' (picked up / owner returned): no alert.

Limits (state them in the report): no identity - ANY person near the object resets the timer
(the proposal's owner_track_id needs a tracker); fixed camera only; tiny overhead targets.
"""
import argparse, json, os
import cv2
import numpy as np


def run(path, work_fps=5.0, width=384, crop=None, mask_rects=(), t_init=8.0,
        t_static=3.0, t_alert=10.0, radius_frac=0.16, min_area=18, max_area=260,
        person_area=300, th_bg=28, th_stab=14, min_life=6.0, max_touch=0.25, debug_dir=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / work_fps)))
    dt = step / fps
    frames, times = [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if crop:
                x0, y0, x1, y1 = crop
                f = f[y0:y1, x0:x1]
            h0, w0 = f.shape[:2]
            s = width / w0
            f = cv2.resize(f, (width, int(round(h0 * s))), interpolation=cv2.INTER_AREA)
            g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
            for (rx0, ry0, rx1, ry1) in mask_rects:       # fractional rects to blank out overlays
                H, W = g.shape
                g[int(ry0 * H):int(ry1 * H), int(rx0 * W):int(rx1 * W)] = 0
            frames.append(g); times.append(i / fps)
        i += 1
    cap.release()
    n = len(frames)
    H, W = frames[0].shape
    R = radius_frac * W
    k_init = max(3, int(t_init / dt))
    bg = np.median(np.stack(frames[:k_init]), axis=0).astype(np.float32)   # B_slow init
    count = np.zeros((H, W), np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    objs = []       # tracked static objects
    log = []
    nxt = 0
    prev = frames[0]
    for t_idx in range(n):
        g, t = frames[t_idx], times[t_idx]
        d_bg = np.abs(g - bg) > th_bg
        d_pr = np.abs(g - prev) > th_stab
        moving = cv2.morphologyEx(d_pr.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        moving = cv2.dilate(moving, kernel, iterations=2)
        # global change (exposure flicker) -> don't learn from or trust this frame
        if d_bg.mean() > 0.25 or d_pr.mean() > 0.25:
            count[:] = 0; prev = g; continue
        stable = d_bg & (~d_pr)
        count = np.where(stable, count + 1, 0)
        static = (count >= int(t_static / dt)).astype(np.uint8)
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, kernel)
        n_c, lab, stats, cen = cv2.connectedComponentsWithStats(static, 8)
        cands, persons = [], []
        for c in range(1, n_c):
            a = stats[c, cv2.CC_STAT_AREA]
            if a >= person_area:
                persons.append(tuple(cen[c]))
            elif min_area <= a <= max_area:
                cands.append((tuple(cen[c]), int(a), tuple(int(v) for v in stats[c, :4])))
        # a big static blob centred on a tracked object is the object (+shadow), not a person
        persons = [p for p in persons if not any(np.hypot(p[0] - o['xy'][0], p[1] - o['xy'][1]) <= 10
                                                 for o in objs if o['removed'] is None)]
        # moving blobs = people in motion
        n_m, _, st_m, cen_m = cv2.connectedComponentsWithStats(moving, 8)
        for c in range(1, n_m):
            if st_m[c, cv2.CC_STAT_AREA] >= 40:
                persons.append(tuple(cen_m[c]))
        # match candidates to tracked objects (nearest, <= 6 px)
        used = set()
        for o in objs:
            best, bd = None, 1e9
            for j, (c, a, bb) in enumerate(cands):
                d = np.hypot(c[0] - o['xy'][0], c[1] - o['xy'][1])
                if d < bd and j not in used:
                    best, bd = j, d
            if best is not None and bd <= 6:
                used.add(best)
                o['xy'] = cands[best][0]; o['bbox'] = cands[best][2]; o['last_seen'] = t; o['miss'] = 0
            else:
                o['miss'] = o.get('miss', 0) + 1
        for j, (c, a, bb) in enumerate(cands):
            if j in used:
                continue
            # first time we see it static: who was nearest (the probable owner) -> keep for the log
            near = [np.hypot(p[0] - c[0], p[1] - c[1]) for p in persons]
            objs.append(dict(id=nxt, xy=c, bbox=bb, t_static=t - t_static, last_seen=t, miss=0,
                             last_person_t=t, alerted=None, removed=None, area=a, touch=0, frames=0,
                             n_near_at_start=int(sum(d <= R for d in near))))
            nxt += 1
        mov_c = [tuple(cen_m[c]) for c in range(1, n_m) if st_m[c, cv2.CC_STAT_AREA] >= 40]
        for o in objs:
            if o['removed'] is not None:
                continue
            o['frames'] = o.get('frames', 0) + 1
            if any(np.hypot(p[0] - o['xy'][0], p[1] - o['xy'][1]) <= 12 for p in mov_c):
                o['touch'] = o.get('touch', 0) + 1
            if any(np.hypot(p[0] - o['xy'][0], p[1] - o['xy'][1]) <= R for p in persons):
                o['last_person_t'] = t
            x, y, w, h = o['bbox']
            reg = d_bg[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1]
            frac = float(reg.mean()) if reg.size else 0.0
            o['gone_cnt'] = o.get('gone_cnt', 0) + 1 if frac < 0.15 else 0
            gone = o['gone_cnt'] * dt >= 1.5          # pixels went back to the background
            if gone:
                o['removed'] = t - 1.5
            elif o['alerted'] is None and (t - o['last_person_t']) >= t_alert:
                o['alerted'] = t
        prev = g
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            vis = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for o in objs:
                if o['removed'] is None:
                    x, y, w, h = o['bbox']
                    col = (0, 0, 255) if o['alerted'] is not None else (0, 255, 255)
                    cv2.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2), col, 1)
            if t_idx % 5 == 0:
                cv2.imwrite(f"{debug_dir}/{t_idx:04d}.jpg", vis)
    out = []
    for o in objs:
        # drop flicker: needs to have lived at least 2 s static
        life = (o['removed'] if o['removed'] is not None else times[-1]) - o['t_static']
        touch_frac = o.get('touch', 0) / max(1, o.get('frames', 1))
        if life < min_life or touch_frac > max_touch or o['t_static'] < t_init + 1.0:
            continue
        out.append(dict(id=o['id'], xy=[round(float(o['xy'][0]), 1), round(float(o['xy'][1]), 1)],
                        area=o['area'], touch_frac=round(touch_frac, 2), t_static=round(o['t_static'], 2),
                        t_owner_left=round(o['last_person_t'], 2),
                        t_alert=None if o['alerted'] is None else round(o['alerted'], 2),
                        t_removed=None if o['removed'] is None else round(o['removed'], 2)))
    return dict(video=os.path.basename(path), duration=round(times[-1], 2), frame_wh=[W, H],
                params=dict(t_static=t_static, t_alert=t_alert, radius_px=round(R, 1), min_life=min_life, max_touch=max_touch), objects=out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t-alert", type=float, default=10.0)
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.video, t_alert=a.t_alert, debug_dir=a.debug), indent=1))
