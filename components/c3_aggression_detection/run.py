"""Team-guide entry point for C3 recorded-video inference.

From the repository root:
    python components/c3_aggression_detection/run.py --source C:/clips/video.mp4

Delegates to the existing causal pose and guarded ST-GCN pipeline. Source paths
are relative to the calling terminal; output paths are relative to this component.
"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Local recorded video; live cameras are not supported by this entry point')
    parser.add_argument('--seconds', type=float, default=8., help='Length of the recorded context (default: 8 seconds)')
    parser.add_argument('--start', type=float, default=0., help='Start on the original-video clock')
    parser.add_argument('--out-dir', help='New result directory inside Component 03; default: outputs/run_<UTC time>')
    parser.add_argument('--camera-id', default='c3_recorded')
    parser.add_argument('--checkpoint', help='Optional candidate checkpoint, relative to Component 03 or an absolute path')
    parser.add_argument('--no-render', action='store_true', help='Skip annotated MP4 rendering')
    parser.add_argument('--research-alerts', action='store_true', help='Opt in to existing eligible research candidates; operational alerts remain disabled')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isfinite(args.start) or args.start < 0:
        parser.error('--seconds must be positive and --start nonnegative; both must be finite')
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        parser.error('--source must be an existing local recorded video')
    component = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    output = Path(args.out_dir).expanduser() if args.out_dir else Path('outputs') / f'run_{stamp}'
    output = (component / output).resolve()
    if output == component or not output.is_relative_to(component):
        parser.error('--out-dir must be a new directory inside Component 03')
    if output.exists():
        parser.error('--out-dir already exists; choose a new result directory')
    command = [sys.executable, '-m', 'src.predict_advanced', '--video', str(source),
               '--seconds', str(args.seconds), '--start', str(args.start),
               '--out-dir', str(output), '--camera-id', args.camera_id]
    if not args.no_render:
        command.append('--render')
    if args.research_alerts:
        command.append('--research-alerts')
    if args.checkpoint:
        checkpoint = (component / Path(args.checkpoint).expanduser()).resolve()
        if not checkpoint.is_file():
            parser.error('--checkpoint must name an existing local checkpoint')
        command.extend(['--checkpoint', str(checkpoint)])
    return subprocess.run(command, cwd=component, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
