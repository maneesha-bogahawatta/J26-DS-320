"""C3 outputs stay in per-camera folders and cannot collide with C1/C2/C4."""
import re
from pathlib import Path

def output_path(camera_id,provided,filename):
    if provided is not None:return Path(provided)
    raw=str(camera_id).strip()
    if not raw or raw in {'.','..'}:raise ValueError('Invalid camera_id')
    safe=re.sub(r'[^A-Za-z0-9_-]','_',raw)
    return Path('outputs')/'c3'/safe/filename
