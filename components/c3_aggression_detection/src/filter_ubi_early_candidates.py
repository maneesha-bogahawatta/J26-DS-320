
import csv
import json
from pathlib import Path

INPUT = Path("data/manifests/ubi_review.csv")
OUTPUT = Path("data/manifests/ubi_early_review_candidates.csv")

MIN_PRECONTEXT = 5.0

eligible = []
normal = 0
too_early = 0
invalid = 0

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    try:
        intervals = json.loads(row["intervals"] or "[]")

        if not intervals:
            normal += 1
            continue

        # First annotated fight onset
        onset = float(intervals[0][0])

        if onset >= MIN_PRECONTEXT:
            row["fight_onset_s"] = onset
            eligible.append(row)
        else:
            too_early += 1

    except (ValueError, TypeError, IndexError, KeyError):
        invalid += 1
        print("Invalid annotation:", row.get("video"))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if eligible:
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eligible[0].keys())
        writer.writeheader()
        writer.writerows(eligible)

print("\n===== UBI-FIGHTS REVIEW SUMMARY =====")
print("Total videos:", len(rows))
print("Videos without fight intervals:", normal)
print("Fight onset before 5 seconds:", too_early)
print("Eligible for 5-second review:", len(eligible))
print("Parsing errors:", invalid)
print("Output:", OUTPUT)
