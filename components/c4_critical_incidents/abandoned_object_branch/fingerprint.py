import os, sys, cv2, numpy as np, platform
d = os.environ.get("VIDEO_DIR", "raw")
print("python", platform.python_version(), "| cv2", cv2.__version__, "| numpy", np.__version__)
for b in ["LeftBag_AtChair", "LeftBag"]:
    c = cv2.VideoCapture(os.path.join(d, b + ".mpg"))
    print(b, "fps", c.get(cv2.CAP_PROP_FPS), "frames", int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
    for k in (100, 500, 900):
        c.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = c.read()
        print("  frame", k, "ok", ok, "mean", round(float(f.mean()), 3) if ok else None)
