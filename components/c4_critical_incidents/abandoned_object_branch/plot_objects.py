import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
gt = json.load(open("object_gt.json"))["scenarios"]
res = json.load(open("object_baseline_results.json"))["runs"]["10.0"]
dur = {"LeftBag":57.4,"LeftBag_AtChair":44.6,"LeftBag_BehindChair":42.6,"LeftBag_PickedUp":54.2,"LeftBox":34.4}
fig, ax = plt.subplots(figsize=(9, 3.6))
names = list(gt)
for i, n in enumerate(names):
    y = len(names) - 1 - i; g = gt[n]
    ax.barh(y, dur[n], color="#e8e8e8", height=0.55, zorder=1)
    if g["owner_gone"] is not None:
        end = g["owner_returns"] or dur[n]
        ax.barh(y, end - g["owner_gone"], left=g["owner_gone"], color="#f2c14e", height=0.55, zorder=2)
    ax.plot([g["place"]], [y], marker="v", color="k", zorder=3)
    r = [x for x in res if x["clip"] == n][0]
    if r["detector_alert"] is not None:
        ax.plot([r["detector_alert"]], [y], marker="*", ms=14, color="#c0392b", zorder=4)
    if r["expected_alert"] is not None:
        ax.plot([r["expected_alert"]], [y], marker="|", ms=16, color="#1f4e79", mew=2, zorder=4)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1]); ax.set_xlabel("time in clip (s)")
ax.set_title("Static-object baseline, unattended timer T = 10 s  (provisional ground truth)", fontsize=10)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="v",color="k",ls="",label="object placed"),
    plt.Rectangle((0,0),1,1,color="#f2c14e",label="owner away"),
    Line2D([],[],marker="|",color="#1f4e79",ls="",ms=12,mew=2,label="alert expected"),
    Line2D([],[],marker="*",color="#c0392b",ls="",ms=12,label="detector alert")], loc="lower right", fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig("object_baseline_timeline.png", dpi=150)
