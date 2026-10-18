# First findings: object clips (CAVIAR left-bag/box set)

Clips: LeftBag, LeftBag_AtChair, LeftBag_BehindChair, LeftBag_PickedUp, LeftBox (384x288, overhead fisheye).
Ground truth provisional (+-2 s); detector = classical static-object baseline, ROI-masked.

| T (unattended) | TP | FP | FN | TN | Mean delay after owner leaves |
|---|---|---|---|---|---|
| 3 s  | 3 | 0 | 1 | 1 | 3.7 s |
| 5 s  | 2 | 0 | 1 | 2 | 5.6 s |
| 10 s | 2 | 0 | 1 | 2 | 10.6 s |

"Expected alert" = owner_gone + T inside the clip, before the owner returns, with a 2 s margin.
Short clips mean a long T makes some clips un-alertable (LeftBox ends 5 s after the owner leaves).

## What worked
- LeftBag: static at 37.4 s (placed ~37.8), alert 51.6 s at T=10 (expected 50).
- LeftBag_AtChair: static 18.6 s, alert 31.6 s (expected 32), removed 37.9 s when the owner returns.
- LeftBag_PickedUp: object tracked, flagged removed at 41.1 s, no alert (owner stays near).
- LeftBox: found at 22.0 s, correct no-alert at T>=5 because the clip ends before T elapses.

## What failed / is weak
- LeftBag_BehindChair missed at every T: dark bag, partly hidden behind a chair, against dark furniture.
  This is the occlusion / low-contrast case in RQ3.
- Bug found and fixed on LeftBox: the box + its shadow merged into a blob big enough to be treated as a
  person, so the timer kept resetting. Rule now ignores static blobs centred on a tracked object.
- Identity-less "person near" would suppress a real alert whenever a bystander walks past (a passer-by
  near LeftBag at 52-54 s shows how easily this happens).
- Static people (standing still) produce short-lived candidates (e.g. PickedUp 28.8-35 s); they did not
  alert here only because a person was always near them.

## Other uploaded clips (not scored)
- grocery_theft.mp4: shoplifting, not an abandoned object.
- aitoe_unattended.mp4, videoplayback.mp4, videoplayback1.mp4: vendor demos with detector boxes/captions burnt in.
- airport_unattended.mp4: handheld phone video; a background model does not apply.
- cctv_unattended.mp4: bag set down in the first 4 s (no empty-scene window); caption overlay.
- tiktok_7252.mp4: multi-scene edit with captions; qualitative only.
