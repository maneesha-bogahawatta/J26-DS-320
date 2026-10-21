"""Deterministic, non-overwriting I/O for versioned C3 training artifacts."""
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def video_id(value):
    return str(value).replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0].casefold()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def write_new(path, content):
    """An identical rerun is allowed; different existing content is never replaced."""
    path = Path(path)
    if isinstance(content, str):
        content = content.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise FileExistsError(f'Refusing to overwrite {path}; choose a new version directory')


def write_json(path, value):
    write_new(path, json_text(value))


def write_csv(path, rows, fields=None):
    import io
    stream = io.StringIO(newline='')
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    write_new(path, stream.getvalue())
