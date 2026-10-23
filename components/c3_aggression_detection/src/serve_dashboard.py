"""Launch the original team dashboard and real C3 API on one loopback origin."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--state-dir', help='Separate C3-local directory for jobs and operator audit')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Choose a port from 1024 to 65535')
    component = Path(__file__).resolve().parents[1]
    repository = component.parents[1]
    dependencies = component/'data/dashboard_deps_v1'
    frontend = repository/'dashboard/frontend/dist'
    if not (frontend/'index.html').is_file():
        parser.error('Frontend build missing. Run npm ci and npm run build:c3 in the dashboard/frontend directory; see DASHBOARD_INTEGRATION.md.')
    for path in (str(dependencies), str(component), str(repository/'dashboard/backend')):
        if path not in sys.path: sys.path.insert(0, path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['C3_DASHBOARD_ENABLED'] = '1'
    os.environ['PYTHONPATH'] = os.pathsep.join([str(dependencies), str(component), str(repository/'dashboard/backend'),
        os.environ.get('PYTHONPATH', '')])
    if args.state_dir:
        os.environ['C3_DASHBOARD_STATE_DIR'] = args.state_dir
    try:
        import uvicorn
        from main import app
    except ImportError as exc:
        parser.error(f'Isolated dashboard dependencies are missing ({exc}). See DASHBOARD_INTEGRATION.md for the local --target installation command.')
    print(f'C3 integrated dashboard: http://127.0.0.1:{args.port}', flush=True)
    print('Recorded-video research inference; uncalibrated scores; operational alerts disabled. Ctrl+C stops the server.', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1)


if __name__ == '__main__': main()
