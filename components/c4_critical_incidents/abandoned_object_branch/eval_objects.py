"""Score the static-object baseline against object_gt.json for several unattended timers T.
A clip is 'expected to alert' at T iff owner_gone + T is inside the clip, before the owner returns,
with a 2 s margin for label uncertainty. TP if the detector alerts on the right object within +-4 s of
the expected time; FP if it alerts when it should not (or on a different object); FN if it never alerts."""
import json, sys, os
from static_object import run
CAVIAR_MASK = [(0.0, 0.72, 0.62, 1.0)]
MARGIN, TOL = 2.0, 4.0
gt = json.load(open("object_gt.json"))["scenarios"]
out = {"mask": CAVIAR_MASK, "margin_s": MARGIN, "tolerance_s": TOL, "runs": {}, "summary": {}}
for T in (3.0, 5.0, 10.0):
    rows, tp = [], dict(TP=0, FP=0, FN=0, TN=0, BORDERLINE=0)
    lat = []
    for name, g in gt.items():
        r = run(os.path.join(os.environ.get("VIDEO_DIR", "raw"), g["video"]), mask_rects=CAVIAR_MASK, t_alert=T)
        end = r["duration"]
        stop = min(end, g["owner_returns"] if g["owner_returns"] else end)
        exp, borderline = None, False
        if g["owner_gone"] is not None:
            t_exp = g["owner_gone"] + T
            if t_exp <= stop - MARGIN: exp = t_exp                 # clearly alertable
            elif t_exp < stop + MARGIN: borderline = True          # within label uncertainty of the boundary
        match = [o for o in r["objects"] if abs(o["xy"][0] - g["xy"][0]) <= 25 and abs(o["xy"][1] - g["xy"][1]) <= 25
                 and abs(o["t_static"] - g["place"]) <= 6]
        alerts_match = [o["t_alert"] for o in match if o["t_alert"] is not None]
        alerts_other = [o for o in r["objects"] if o["t_alert"] is not None and o not in match]
        got = alerts_match[0] if alerts_match else None
        if borderline: res = "BORDERLINE"
        elif exp is not None and got is not None and abs(got - exp) <= TOL: res = "TP"; lat.append(got - g["owner_gone"])
        elif exp is not None and got is None: res = "FN"
        elif exp is None and got is None and not alerts_other: res = "TN"
        else: res = "FP"
        if alerts_other and res == "TN": res = "FP"
        tp[res] += 1
        rows.append(dict(clip=name, T=T, expected_alert=exp, borderline=borderline, detector_alert=got, other_alerts=len(alerts_other),
                         detector_found_object=bool(match), t_static=(match[0]["t_static"] if match else None), outcome=res))
    out["runs"][str(T)] = rows
    out["summary"][str(T)] = dict(tp, mean_latency_after_owner_gone=round(sum(lat) / len(lat), 1) if lat else None)
json.dump(out, open("object_baseline_results.json", "w"), indent=1)
for T, rows in out["runs"].items():
    print(f"\nT = {T} s   {out['summary'][T]}")
    for r in rows: print("  ", {k: r[k] for k in ("clip", "expected_alert", "detector_alert", "other_alerts", "detector_found_object", "outcome")})
