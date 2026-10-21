# Component 4 - abandoned-object baseline (first pass)

Classical fixed-camera baseline for the object half of Component 4. It is the object-side
counterpart of the fall pipeline and writes results in the same spirit (per-clip events, scored
against a ground-truth file). It is a BASELINE to beat with the YOLO + tracker + owner-linking
method in your proposal, not the final method.

## Run (Mac or Colab; needs only opencv-python, numpy, matplotlib)
    pip install opencv-python numpy matplotlib
    export VIDEO_DIR="/path/to/folder/with/the/five/.mpg/files"   # LeftBag.mpg, LeftBox.mpg, ...
    python eval_objects.py        # -> object_baseline_results.json  (T = 3, 5, 10 s)
    python plot_objects.py        # -> object_baseline_timeline.png

## How it works (static_object.py)
1. Background = median of the first 8 s (so an object placed in the first ~9 s cannot be detected).
2. A pixel is "static foreground" if it differs from the background and has stopped changing.
   Components of the right size (18-260 px at 384 px width) are static-object candidates.
3. A person is "near" if a moving blob (or a large static blob) is within R = 16 % of the frame width, seen in >= 2 of the last 3 frames (debounce against one-frame noise blobs).
4. Unattended timer = time since a person was last near. Alert at T. If the object's pixels return to
   the background, it is "removed" (picked up / owner returned): no alert.
5. Filters: lifetime >= 6 s, not constantly touched by motion, appears after the background window,
   fixed ROI mask over the glass wall / plants in the bottom-left (auto-exposure flicker).

## Known limits (put these in the report)
- No identity: ANY person near the object resets the timer. The proposal's owner_track_id needs a tracker.
- Fixed camera only; dark-on-dark and partly hidden objects are missed (LeftBag_BehindChair).
- Filters, ROI, a debounce and two label corrections were developed while looking at these same 5 clips: results are a
  development result, NOT a held-out test.
- Ground truth is provisional (read by eye from 3x crops, +-2 s). Confirm in CVAT.
- 5 clips, no long normal footage: false alarms per camera-hour cannot be estimated yet.
