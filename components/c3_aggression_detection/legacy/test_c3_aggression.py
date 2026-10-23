# pytest tests for C3 (run from this folder's parent:  pytest tests/ -q)
import os, sys, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from run import *

def _mk_person(cx, cy, h=250.0, jit=0.0, rs=None, raise_arm=False):
    rs = rs or np.random.RandomState(0)
    kp = np.zeros((17, 3), np.float32); kp[:, :2] = np.array([cx, cy]) + (TEMPLATE + rs.randn(17, 2) * jit) * h; kp[:, 2] = 0.9
    if raise_arm: kp[9, 1] = kp[5, 1] - 0.3 * h
    return kp, np.array([cx - 0.2 * h, cy - 0.5 * h, cx + 0.2 * h, cy + 0.5 * h], np.float32)

def _seq(n=24, approach=True, scale=1.0, raise_from=None, jit=0.0, seed=0):
    rs = np.random.RandomState(seed); frames = []
    for f in range(n):
        gap = (400 - 14 * f) if approach else 400
        k1, b1 = _mk_person(100 * scale, 300 * scale, 250 * scale, jit, rs, raise_arm=(raise_from is not None and f >= raise_from))
        k2, b2 = _mk_person((100 + gap) * scale, 300 * scale, 250 * scale, jit, rs)
        frames.append((np.stack([k1, k2]), np.stack([b1, b2]), np.array([1, 2])))
    return frames

def test_ema_smoothing_reduces_jitter_and_is_causal():
    clean = _seq(30, jit=0.0); noisy = _seq(30, jit=0.03, seed=1)
    sm = ema_smooth_sequence(noisy)
    def rough(p): return np.mean([np.abs(np.diff(np.array([f[0][0, 9, :2] for f in p]), 2, axis=0)).mean()])
    assert rough(sm) < rough(noisy)
    a = [(f[0].copy(), f[1].copy(), f[2]) for f in noisy]; b = [(f[0].copy(), f[1].copy(), f[2]) for f in noisy]
    b[20][0][:, :, :2] += 500.0                                      # change ONLY the future
    sa, sb = ema_smooth_sequence(a), ema_smooth_sequence(b)
    for t in range(20): assert np.allclose(sa[t][0], sb[t][0]) and np.allclose(sa[t][1], sb[t][1])   # causal: past unaffected

def test_savgol_smoothing_reduces_jitter():
    noisy = _seq(30, jit=0.03, seed=2); sm = savgol_smooth_sequence(noisy, 7, 2)
    r = lambda p: np.abs(np.diff(np.array([f[0][0, 10, :2] for f in p]), 2, axis=0)).mean()
    assert r(sm) < r(noisy)

def test_knn_imputation_beats_box_centre_baseline():
    rs = np.random.RandomState(0); imp = KNNPoseImputer()
    for _ in range(300):
        k, b = _mk_person(rs.uniform(100, 500), rs.uniform(150, 400), rs.uniform(150, 350), 0.03, rs); imp.add(k[None], b[None])
    e_knn, e_ctr = [], []
    for _ in range(100):
        kp, box = _mk_person(rs.uniform(100, 500), rs.uniform(150, 400), rs.uniform(150, 350), 0.03, rs); h = box[3] - box[1]
        hide = rs.choice(17, 5, replace=False); k2 = kp.copy(); k2[hide, 2] = 0.05; k2[hide, :2] = 0
        out, n = imp.impute(k2, box); assert n == 5 and np.allclose(out[~np.isin(np.arange(17), hide)], kp[~np.isin(np.arange(17), hide)])
        assert (out[hide, 2] == IMPUTED_CONF).all()
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        e_knn.append(np.linalg.norm(out[hide, :2] - kp[hide, :2], axis=1).mean() / h); e_ctr.append(np.linalg.norm(c - kp[hide, :2], axis=1).mean() / h)
    assert np.mean(e_knn) < 0.5 * np.mean(e_ctr)

def test_temporal_interpolation_restores_short_gap():
    fr = _seq(20); gt = [f[0][0, 9, :2].copy() for f in fr]
    for f in (6, 7, 8): fr[f][0][0, 9, 2] = 0.05; fr[f][0][0, 9, :2] = 0
    out, info = impute_sequence(fr, KNNPoseImputer())
    assert info['temporal'] >= 3
    assert all(np.linalg.norm(out[f][0][0, 9, :2] - gt[f]) < 2.0 for f in (6, 7, 8))

def test_difem_feature_bounds():
    ff = pose_sequence_feats(_seq(24, raise_from=10), POSE_FPS)
    assert ff.shape == (24, POSE_DIM) and np.isfinite(ff).all()
    assert ff[:, 0].max() <= 10 and ff[:, 1].min() >= 0 and ff[:, 1].max() <= 4.0
    assert ff[:, 2].min() >= -5 and ff[:, 2].max() <= 5 and ff[:, 3].max() <= 15 and ff[:, 4].max() <= 15
    assert ff[:, 5].max() <= 30 and ff[:, 6].min() >= -1 and ff[:, 6].max() <= 2 and 0 <= ff[:, 7].min() and ff[:, 7].max() <= 1
    assert ff[:, 8].max() <= 200 and ff[:, 9].max() <= 30 and ff[:, 10].max() <= 30
    assert ff[12:, 2].mean() > 0.2 and ff[12:, 6].max() > 0.15          # approaching + raised arm are detected

def test_difem_features_are_scale_invariant():
    a = pose_sequence_feats(_seq(24, scale=1.0, raise_from=10), POSE_FPS); b = pose_sequence_feats(_seq(24, scale=2.0, raise_from=10), POSE_FPS)
    assert np.allclose(a, b, atol=1e-3)

def test_track_quality_drops_with_occlusion():
    clean = _seq(16); occl = _seq(16)
    for f in occl: f[0][:, 5:, 2] = 0.05
    qa = track_quality_frames(clean, clean); imp, _ = impute_sequence(occl, KNNPoseImputer()); qb = track_quality_frames(occl, imp)
    for q in (qa, qb): assert (q >= 0).all() and (q[:, 1:] <= 1.0 + 1e-6).all()          # ratios and quality score live in [0, 1]
    assert qb[:, 4].mean() < qa[:, 4].mean() and qb[:, 2].mean() > qa[:, 2].mean()

def test_process_persons_end_to_end():
    fr = _seq(24); ft = np.arange(24) / POSE_FPS; ts = np.array([2.0, 2.9])
    r = process_persons(fr, ft, ts, (640, 360))
    assert r['pose'].shape == (2, 2 * POSE_DIM) and r['qual'].shape == (2, QUAL_DIM) and r['sk'].shape == (2, SK_T, 17, MAX_P, 3)
    assert np.isfinite(r['pose']).all() and set(r['ids'][1]) == {1, 2}

def test_rule_baseline_is_monotonic_and_explainable():
    def vec(closing, dist=0.7, wrist=0.0):
        v = np.zeros(2 * POSE_DIM); v[POSE_DIM] = 2; v[0] = 2; v[1] = dist; v[POSE_DIM + 1] = dist; v[POSE_DIM + 2] = closing; v[POSE_DIM + 5] = wrist; return v
    r0, r1, r2 = [RULES.score(vec(c))[0] for c in (0.0, 0.8, 2.0)]
    assert 0 <= r0 < r1 < r2 <= 1
    sig = RULES.explain(vec(2.0, 0.6, 14.0)); names = [s['signal'] for s in sig]
    assert 'rapid approach' in names and 'wrist velocity spike' in names
    assert RULES.score(vec(2.0, 3.0))[0] > 0
    one = vec(2.0); one[POSE_DIM] = 1; one[0] = 1; assert RULES.score(one)[0] < RULES.score(vec(2.0))[0]
    assert RULES.estimate_ttc(vec(1.0, 1.6)) == 1.0 and RULES.estimate_ttc(vec(0.0, 1.6)) is None

def test_uncertainty_and_priority_tier_bounds():
    for r in (0.0, 0.3, 0.5, 1.0):
        for q in (0.0, 0.5, 1.0): assert 0.0 <= uncertainty_score(r, q) <= 1.0
    assert uncertainty_score(0.5, 1.0) > uncertainty_score(0.0, 1.0) and uncertainty_score(0.0, 0.1) > uncertainty_score(0.0, 0.9)
    assert [TIERS[tier_from_score(s, 0.35, 0.6)] for s in (0.1, 0.4, 0.7)] == ['LOW', 'WATCH', 'HIGH']

def test_debouncer_k_of_n_cooldown_and_escalation():
    d = AlertDebouncer(n=3, k_watch=2, k_high=2, cooldown=5.0)
    assert [d.update(t, x) for t, x in enumerate([0, 1, 0, 0])] == [None, None, None, None]          # single spike never alerts
    d.reset(); out = [d.update(t, x) for t, x in enumerate([0, 1, 1, 1, 1])]
    assert out == [None, None, 1, None, None]                                                          # confirmed WATCH, then cooldown
    out = [d.update(t, x) for t, x in enumerate([2, 2, 2], start=5)]; assert out[0] == 2 or out[1] == 2   # escalation is immediate
    d.reset(); fired = [d.update(t, 1) for t in range(12)]; assert fired[1] == 1 and sum(f is not None for f in fired) >= 2   # repeat after cooldown
    d.reset(); [d.update(t, 1) for t in range(3)]; [d.update(t, 0) for t in range(3, 6)]
    assert d.update(6, 1) is None and d.update(7, 1) == 1                                               # re-armed after quiet steps

def _rec(tier='WATCH', risk=0.5, t=3.0):
    return make_step_record(t, risk, tier, 0.3, 0.8, 1.5, np.array([4, 7]), np.array([[.1, .1, .3, .9], [.4, .1, .6, .9]]),
                            [dict(signal='rapid approach', source='pose_rules', value=1.2, activation=0.8)], 'pre_attack', 'rule_baseline')

def test_incident_event_matches_json_contract():
    ev = build_alert_event(_rec(), 1, 'cam_07', '2026-10-18T10:00:00+00:00')
    assert validate_event(ev) and ev['priority_tier'] == 'WATCH' and ev['person_ids'] == [4, 7] and ev['camera_id'] == 'cam_07'
    json.loads(json.dumps(ev))                                                                         # JSON serializable
    for k in ('event_id', 'timestamp', 'camera_id', 'person_ids', 'bounding_boxes', 'threat_score', 'priority_tier', 'uncertainty', 'track_quality', 'explanation_signals'):
        bad = dict(ev); bad.pop(k)
        try: validate_event(bad); assert False, f'{k} should be required'
        except AssertionError as e: assert 'missing field' in str(e)
    for patch in (dict(priority_tier='LOW'), dict(threat_score=1.5), dict(timestamp='not-a-time'), dict(bounding_boxes=[dict(person_id=1)])):
        try: validate_event({**ev, **patch}); assert False, f'{patch} should fail'
        except (AssertionError, ValueError): pass

def test_step_assessments_are_separate_from_debounced_alerts():
    tiers = ['LOW'] * 3 + ['WATCH', 'LOW', 'WATCH', 'WATCH', 'WATCH', 'HIGH', 'HIGH', 'LOW', 'LOW', 'LOW']
    recs = [_rec(tr, {'LOW': 0.1, 'WATCH': 0.45, 'HIGH': 0.8}[tr], t=float(i)) for i, tr in enumerate(tiers)]
    events = records_to_events(recs, 'cam_01', datetime.datetime(2026, 10, 18, tzinfo=datetime.timezone.utc))
    df = records_to_df(recs)
    assert len(df) == len(recs) and (df.priority_tier != 'LOW').sum() == 6          # every step is assessed ...
    assert 1 <= len(events) < 6 and [e['priority_tier'] for e in events][-1] == 'HIGH' # ... but only confirmed alerts are emitted
    assert all(validate_event(e) for e in events)
