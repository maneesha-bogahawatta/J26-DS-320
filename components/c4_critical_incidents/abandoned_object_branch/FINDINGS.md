# First findings: object clips (CAVIAR left-bag/box set)

Clips: LeftBag, LeftBag_AtChair, LeftBag_BehindChair, LeftBag_PickedUp, LeftBox (384x288, overhead fisheye).
Ground truth provisional (+-2 s); detector = classical static-object baseline, ROI-masked.

| T (unattended) | Correct alerts | Missed | False alarms | Correct no-alert | Borderline* | Mean delay after owner leaves |
|---|---|---|---|---|---|---|
| 3 s  | 3 | 1 | 0 | 0 | 1 | 3.4 s |
| 5 s  | 2 | 1 | 0 | 0 | 2 | 5.0 s |
| 10 s | 2 | 1 | 0 | 2 | 0 | 10.0 s |

*Borderline = the expected alert time falls within my +-2 s labelling error of the end of the owner's absence (or of the clip), so it is not counted as right or wrong.
"Expected alert" = owner_gone + T inside the clip and before the owner comes back.

## What worked
- LeftBag: static at 37.4 s (placed ~37.8), alert 50.4 s at T=10 (expected 50).
- LeftBag_AtChair: static 18.6 s, alert 31.6 s (expected 32), removed 37.9 s when the owner returns.
- LeftBag_PickedUp: object tracked and flagged removed at ~41 s. The owner is only away ~5 s, so there is no alert at T>=5; at T=3 it alerts at 36.0 s (borderline).
- LeftBox: found at 22.0 s; the clip ends only ~5 s after the owner leaves, so a 10 s timer cannot fire.

## What failed / is weak
- LeftBag_BehindChair missed at every T: dark bag, partly hidden behind a chair, against dark furniture (occlusion / low contrast, RQ3).
- Identity-less "person near" would suppress a real alert whenever a bystander walks past.
- Static people (standing still) produce short-lived candidates.

## Robustness to decoder differences (found on 18 Oct)
Two machines running the same OpenCV version decoded the .mpg files with frame means ~0.5 grey levels apart (132.85 vs 133.34). The first version of the detector gave a different verdict on LeftBag_AtChair on each machine.
Cause: a single-frame moving blob 61.0 px from the bag (radius 61.4 px) reset the 10 s timer.
Fix: a person counts as "near" only if seen near in >= 2 of the last 3 frames.
Check (sweep_noise.py / noise_sweep.json): +-0.5, 1, 2 grey-level uniform noise on every frame, 3 seeds each, 5 clips x T in {5, 10}: the verdict per clip is identical in every run.
Ground-truth correction: LeftBag_PickedUp was first labelled "owner stays near"; a radius-overlay crop (verify/PickedUp_owner_distance.jpg) shows he is outside the radius ~31-37 s. The label was changed from the crop, not from the detector output.

## Honest status
Filters, ROI, the debounce and two label fixes were made while looking at these five clips. Treat the table as a development result; it needs held-out clips (ABODA, PETS2006, AVSS2007, your own recordings) and long normal footage for false alarms per camera-hour.

## Other uploaded clips (not scored)
- grocery_theft.mp4: shoplifting, not an abandoned object.
- aitoe_unattended.mp4, videoplayback.mp4, videoplayback1.mp4: vendor demos with detector boxes/captions burnt in.
- airport_unattended.mp4: handheld phone video; a background model does not apply.
- cctv_unattended.mp4: bag set down in the first 4 s (no empty-scene window); caption overlay.
- tiktok_7252.mp4: multi-scene edit with captions; qualitative only.
