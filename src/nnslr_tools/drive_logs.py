"""The comma's logs of a drive for the drive review (Data › Drive review): where they come from, how they are kept, read
and joined to the video.

Where they come from. The comma writes an rlog (every message) and a qlog (a decimated copy) for every minute of a
drive. ``helpers/fetch_comma_segments.sh`` (the comma pill's Download new drives) already copies the rlog of every
segment it downloads next to its video, into the shared folder (COMMA_VIDEOS_DIR, /root/comma-videos), and
``nnslr sync``/``link-videos`` link them into ``<data root>/raw/routes/<route>/<segment>/``. ``discover`` finds them
there; nothing new connects to the comma. Logs downloaded elsewhere (comma connect, a USB copy) are added from the
portal (Upload logs) and kept, unchanged, in ``<data root>/raw/drive_logs/<route>/<segment>/<kind>-<sha16><ext>``:
the name holds the content's hash, so the same file uploaded twice is stored once.

What is kept. Every file is recorded in the drive review's database (drive_review.DriveReviewStore, table log_files)
by the drive's original id, the segment, the kind (rlog or qlog) and the SHA-256 of its content, with where it came
from (comma-download or upload), its size, and its coverage (the span of its messages and their count per service,
once read). A segment may have several files: the scan reads the best one (an rlog before a qlog, then the one with
more messages) and records which.

How they are read. ``nnslr_tools.drive_extract`` reads one log in a child process with the Cap'n Proto schema of the
runtime branch (speed-limits of cristianku/sunnypilot: speedVisionSP, longitudinalPlanSP, liveMapDataSP), pinned
below by commit and SHA-256 and downloaded once into ~/.cache/comma-fetch-log (NNSLR_DRIVE_SCHEMA: a folder with
log.capnp, custom.capnp, deprecated.capnp, include/c++.capnp and car.capnp instead). Its result is cached per file
and EXTRACT_VERSION in ``<data root>/derived/drive_logs/<route>/<segment>.<sha16>.v<version>.json.gz``: a scan reads
again only the segments that are new or changed.

How they are joined. ``merge`` puts the segments of a drive in order on the comma's monotonic clock (logMonoTime runs
on across the segments of a drive): a missing segment ends every series with None (no value is carried over a hole),
a segment whose clock goes back is left out and reported. ``sync`` joins that clock to the drive's video
(drive_video: the road camera's minutes joined into one MP4): the encode index of each segment gives, for every video
frame, the start of its exposure on the same clock. The civil time (the GPS's) is only the date shown; file times and
names are never used to align anything.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from nnslr_tools.drive_extract import EXTRACT_VERSION

ROUTE = re.compile(r'[0-9a-f]{8}--[0-9a-f]{10}')
# A log file as the comma names it.
LOG_FILE = re.compile(r'(rlog|qlog)(\.zst|\.bz2)?')
# A log named with its drive and segment, as comma connect downloads it, or the comma's own folder name before it.
NAMED_LOG = re.compile(r'(?:.*/)?(?P<route>[0-9a-f]{8}--[0-9a-f]{10})--(?P<segment>[0-9]{1,4})(?:--|/)'
                       r'(?P<kind>rlog|qlog)(?P<ext>\.zst|\.bz2)?')
KIND_RANK = {'rlog': 2, 'qlog': 1}
MAX_SEGMENT = 9999
# The largest log accepted from the portal (an rlog minute is about 10-30 MB, a bz2 one more).
MAX_UPLOAD_BYTES = 256 * 1024 * 1024
EXTRACT_TIMEOUT = 300

# The schema of the runtime branch speed-limits (cristianku/sunnypilot): its last change to openpilot/cereal.
SCHEMA_REPO = 'cristianku/sunnypilot'
SCHEMA_COMMIT = 'ea7485e3884fc9d618a04075a098b3a4b5c588b4'
# car.capnp of opendbc, the same pin as helpers/comma_fetch_log.py.
CAR_REPO = 'cristianku/opendbc'
CAR_COMMIT = '6810f033b9e60738365cfecb8504c980dfaad137'
_RAW = 'https://raw.githubusercontent.com'
SCHEMA_FILES = (
    ('log.capnp', f'{_RAW}/{SCHEMA_REPO}/{SCHEMA_COMMIT}/openpilot/cereal/log.capnp',
     '4f7e5f7352d8eeb32d4e4bd6fd1db3204dd18057cfcef9fae78c870bbdc9d555'),
    ('custom.capnp', f'{_RAW}/{SCHEMA_REPO}/{SCHEMA_COMMIT}/openpilot/cereal/custom.capnp',
     'ece368ee632e949c81173531b89f7b31378ba6c857a0b72a1796c10ea3b13ab9'),
    ('deprecated.capnp', f'{_RAW}/{SCHEMA_REPO}/{SCHEMA_COMMIT}/openpilot/cereal/deprecated.capnp',
     'b3aa5c13de40001ed7376cef858500fb003f7698f96131cd0db18d878fa1af0d'),
    ('include/c++.capnp', f'{_RAW}/{SCHEMA_REPO}/{SCHEMA_COMMIT}/openpilot/cereal/include/c++.capnp',
     'fb306076cd38c27af1aed20fac6395e9a46fbe5b5df6a248b5f1b6845a079c44'),
    ('car.capnp', f'{_RAW}/{CAR_REPO}/{CAR_COMMIT}/opendbc/car/car.capnp',
     'b4a6502704b0315b8277fb7fe125c9780ec3aae47d4ba3dfa9b3d665160b545a'),
)
SCHEMA_ENV = 'NNSLR_DRIVE_SCHEMA'
# Video frames a second of the comma's road camera (drive_video.FPS).
FPS = 20
# How much two consecutive minutes of a log may overlap (the logger's hand-over takes a few tens of ms).
OVERLAP_NS = 5_000_000_000


class DriveLogError(RuntimeError):
    def __init__(self, reason: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.status = status


# ---------- schema ----------

def schema_dir(fetch=None) -> Path:
    """The folder with the schema files (NNSLR_DRIVE_SCHEMA, else the pinned ones, downloaded and checked once)."""
    value = os.environ.get(SCHEMA_ENV)
    if value:
        folder = Path(value).expanduser()
        if not (folder / 'log.capnp').is_file() or not (folder / 'car.capnp').is_file():
            raise DriveLogError('schema_missing', f'{SCHEMA_ENV} has no log.capnp and car.capnp', 500)
        return folder
    folder = Path.home() / '.cache' / 'comma-fetch-log' / f'{SCHEMA_COMMIT[:12]}-{CAR_COMMIT[:12]}'
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    for name, url, sha in SCHEMA_FILES:
        path = folder / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            continue
        try:
            data = fetch(url)
        except OSError as exc:
            raise DriveLogError('schema_download', f'the log schema could not be downloaded ({name}): {exc}', 503) from exc
        if hashlib.sha256(data).hexdigest() != sha:
            raise DriveLogError('schema_mismatch', f'{name}: the downloaded schema does not match its pinned SHA-256', 500)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + '.part')
        part.write_bytes(data)
        part.replace(path)
    return folder


# ---------- files ----------

def videos_dir() -> Path:
    from nnslr_tools.sync import DEFAULT_VIDEOS_DIR
    return Path(os.environ.get('COMMA_VIDEOS_DIR') or DEFAULT_VIDEOS_DIR)


def upload_dir(root: Path, route: str, segment: int) -> Path:
    return Path(root) / 'raw' / 'drive_logs' / route / str(segment)


def cache_dir(root: Path, route: str) -> Path:
    return Path(root) / 'derived' / 'drive_logs' / route


def check_route(route: str) -> str:
    if not isinstance(route, str) or not ROUTE.fullmatch(route):
        raise DriveLogError('invalid_route', 'Not a drive id.', 404)
    return route


def known_route(root: Path, route: str) -> bool:
    """A drive the portal has: its folder of the comma's files, or its copy in the shared folder."""
    check_route(route)
    if (Path(root) / 'raw' / 'routes' / route).is_dir():
        return True
    base = videos_dir()
    return base.is_dir() and any(base.glob(route + '--*'))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def discover(root: Path, route: str) -> list[dict]:
    """The logs of a drive on this server: {segment, kind, path, source}. The comma's own (raw/routes, the shared
    folder) and the uploaded ones (raw/drive_logs); a file reached by two paths is listed once (its real path)."""
    check_route(route)
    found: dict[str, dict] = {}

    def add(segment: int, kind: str, path: Path, source: str) -> None:
        try:
            real = str(path.resolve(strict=True))
        except OSError:
            return
        if Path(real).is_file() and Path(real).stat().st_size > 0:
            found.setdefault(real, {'segment': segment, 'kind': kind, 'path': real, 'source': source})

    base = Path(root) / 'raw' / 'routes' / route
    if base.is_dir():
        for folder in base.iterdir():
            if folder.name.isdigit() and folder.is_dir():
                for item in folder.iterdir():
                    match = LOG_FILE.fullmatch(item.name)
                    if match:
                        add(int(folder.name), match.group(1), item, 'comma-download')
    shared = videos_dir()
    if shared.is_dir():
        for folder in shared.glob(route + '--*'):
            tail = folder.name[len(route) + 2:]
            if tail.isdigit() and folder.is_dir():
                for item in folder.iterdir():
                    match = LOG_FILE.fullmatch(item.name)
                    if match:
                        add(int(tail), match.group(1), item, 'comma-download')
    uploads = Path(root) / 'raw' / 'drive_logs' / route
    if uploads.is_dir():
        for folder in uploads.iterdir():
            if folder.name.isdigit() and folder.is_dir():
                for item in folder.iterdir():
                    match = re.fullmatch(r'(rlog|qlog)-[0-9a-f]{16}(\.zst|\.bz2)?', item.name)
                    if match:
                        add(int(folder.name), match.group(1), item, 'upload')
    return sorted(found.values(), key=lambda item: (item['segment'], -KIND_RANK[item['kind']], item['path']))


def parse_upload_name(route: str, name: str, segment=None) -> tuple[int, str, str]:
    """(segment, kind, extension) of a log sent from the portal: its relative path or name holds the drive and the
    segment ("<route>--4/rlog.zst", "<route>--4--rlog.zst"), or the page says the segment and the name is rlog.zst."""
    check_route(route)
    text = str(name or '').replace('\\', '/')
    match = NAMED_LOG.fullmatch(text)
    if match:
        if match.group('route') != route:
            raise DriveLogError('wrong_drive', f'{text.rsplit("/", 1)[-1]} is a log of another drive ({match.group("route")}).')
        number = int(match.group('segment'))
        kind, ext = match.group('kind'), match.group('ext') or ''
    else:
        plain = LOG_FILE.fullmatch(text.rsplit('/', 1)[-1])
        if not plain or segment is None:
            raise DriveLogError('unnamed_log', 'Name the file <drive>--<segment>--rlog.zst (or qlog), or pick the '
                                               'segment folders as the comma writes them (<drive>--<segment>/rlog.zst).')
        try:
            number = int(segment)
        except (TypeError, ValueError):
            raise DriveLogError('bad_segment', 'The segment is not a number.') from None
        kind, ext = plain.group(1), plain.group(2) or ''
    if not 0 <= number <= MAX_SEGMENT:
        raise DriveLogError('bad_segment', 'The segment is out of range.')
    return number, kind, ext


def store_upload(root: Path, route: str, segment: int, kind: str, ext: str, stream, length: int) -> dict:
    """Writes an uploaded log under raw/drive_logs, named by its content's hash; the same content already there is not
    written again. Returns {segment, kind, path, source, sha256, size, new}."""
    if length <= 0:
        raise DriveLogError('empty_file', 'The file is empty.')
    if length > MAX_UPLOAD_BYTES:
        raise DriveLogError('too_large', 'The file is larger than a comma log can be.', 413)
    folder = upload_dir(root, route, segment)
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    handle, partial = tempfile.mkstemp(prefix='.incoming-', dir=folder)
    try:
        left = length
        with os.fdopen(handle, 'wb') as out:
            while left > 0:
                block = stream.read(min(1 << 20, left))
                if not block:
                    raise DriveLogError('incomplete', 'The upload stopped before the end of the file.')
                digest.update(block)
                out.write(block)
                left -= len(block)
        sha = digest.hexdigest()
        target = folder / f'{kind}-{sha[:16]}{ext}'
        new = not target.exists()
        if new:
            os.replace(partial, target)
        return {'segment': segment, 'kind': kind, 'path': str(target.resolve()), 'source': 'upload', 'sha256': sha,
                'size': length, 'new': new}
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


# ---------- reading ----------

def extract_path(root: Path, route: str, segment: int, sha256: str) -> Path:
    return cache_dir(root, route) / f'{segment}.{sha256[:16]}.v{EXTRACT_VERSION}.json.gz'


def load_extract(root: Path, route: str, segment: int, sha256: str) -> dict | None:
    try:
        return json.loads(gzip.decompress(extract_path(root, route, segment, sha256).read_bytes()))
    except (OSError, ValueError):
        return None


def extract(root: Path, route: str, segment: int, sha256: str, path: str, *, schema: Path | None = None) -> dict:
    """The series of one log file, from the cache or read now in a child process. Raises DriveLogError when the file
    cannot be read (its reason is recorded with the file)."""
    cached = load_extract(root, route, segment, sha256)
    if cached is not None:
        return cached
    folder = schema or schema_dir()
    try:
        answer = subprocess.run([sys.executable, '-m', 'nnslr_tools.drive_extract', str(folder), str(folder), path],
                                capture_output=True, text=True, timeout=EXTRACT_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        raise DriveLogError('unreadable', 'reading the log took too long') from None
    try:
        value = json.loads(answer.stdout or '{}')
    except ValueError:
        value = {'error': (answer.stderr or 'no answer').strip()[-300:]}
    if answer.returncode != 0 or value.get('error') or 'series' not in value:
        raise DriveLogError('unreadable', value.get('error') or (answer.stderr or 'unreadable log').strip()[-300:])
    target = extract_path(root, route, segment, sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + f'.{os.getpid()}.part')
    partial.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':')).encode('utf-8'), 6))
    os.replace(partial, target)
    return value


# ---------- joining ----------

def merge(extracts: dict[int, dict]) -> dict:
    """The segments of a drive as one drive on the monotonic clock: {series, events, gps, frames: {segment: ...},
    segments: [...], spans: {segment: [first, last]}, holes: [[after, before]], left_out: [{segment, reason}], t0,
    span, meta}."""
    order = sorted(extracts)
    kept: list[int] = []
    left_out = []
    for segment in order:
        span = extracts[segment].get('span') or [None, None]
        if span[0] is None:
            left_out.append({'segment': segment, 'reason': 'no messages'})
            continue
        # The logger closes a minute a few tens of ms after the next one has begun: a larger overlap, or a start
        # before the previous one's, is another start of the comma (its clock began again).
        if kept and (span[0] <= extracts[kept[-1]]['span'][0] or extracts[kept[-1]]['span'][1] - span[0] > OVERLAP_NS):
            left_out.append({'segment': segment, 'reason': "its clock goes back: another start of the comma"})
            continue
        kept.append(segment)
    series: dict[str, list] = {}
    events: list = []
    gps: list = []
    holes = []
    frames = {}
    previous = None
    for segment in kept:
        part = extracts[segment]
        if previous is not None and segment != previous + 1:
            holes.append([previous, segment])
            # Nothing is known over a missing segment: every series ends there.
            end = extracts[previous]['span'][1] + 1
            for points in series.values():
                if points and points[-1][1] is not None:
                    points.append([end, None])
        # A service missing from this minute (it did not run): its series has no value here.
        for name, line in series.items():
            if name not in part.get('series', {}) and line and line[-1][1] is not None:
                line.append([max(part['span'][0], line[-1][0]), None])
        for name, points in part.get('series', {}).items():
            line = series.setdefault(name, [])
            for t, value in points:
                if line and line[-1][1] == value:
                    continue
                # A message of the overlap with the previous minute: kept, at the last time already in the series.
                line.append([max(t, line[-1][0]) if line else t, value])
        events.extend(part.get('events', []))
        gps.extend(part.get('gps', []))
        if part.get('frames', {}).get('ids'):
            frames[segment] = part['frames']
        previous = segment
    events.sort(key=lambda item: item[0])
    gps.sort(key=lambda item: item[0])
    starts = [min(frames[s]['sof']) for s in frames] + [extracts[s]['span'][0] for s in kept]
    t0 = min(starts) if starts else None
    end = max(extracts[s]['span'][1] for s in kept) if kept else None
    meta = {}
    for segment in kept:
        for key, value in (extracts[segment].get('meta') or {}).items():
            meta.setdefault(key, value)
    spans = {segment: extracts[segment]['span'] for segment in kept}
    return {'series': series, 'events': events, 'gps': gps, 'frames': frames, 'segments': kept, 'spans': spans,
            'holes': holes, 'left_out': left_out, 't0': t0, 'span': [t0, end], 'meta': meta}


def sync(merged: dict, video_index: dict | None) -> list[dict]:
    """Per segment, how a time of the logs becomes a time of the drive's video: the exposure start of frame 0
    (seconds from t0) and the frame period fitted on the encode index (frame id -> exposure start), the largest
    deviation of a frame from that line, frames missing from the index, and where the segment starts in the video.
    precision: 'frame' (within half a frame), 'degraded' (the index has holes or the video another count of frames,
    the time is right to within max_error_ms), 'no_video' (logs only), 'no_index' (no encode index: no sync)."""
    t0 = merged.get('t0') or 0
    video = {item['segment']: item for item in (video_index or {}).get('segments', [])}
    out = []
    for segment in merged.get('segments', []):
        info = {'segment': segment, 'frames_log': 0, 'frames_video': None, 'video_start': None, 'start': None,
                'period': None, 'max_error_ms': None, 'missing_ids': 0, 'precision': 'no_index', 'exact': None}
        frames = merged['frames'].get(segment)
        if frames and frames.get('ids'):
            pairs = sorted(zip(frames['ids'], frames['sof']))
            ids = [p[0] for p in pairs]
            times = [(p[1] - t0) / 1e9 for p in pairs]
            count = len(ids)
            mean_id = sum(ids) / count
            mean_t = sum(times) / count
            spread = sum((i - mean_id) ** 2 for i in ids)
            period = (sum((i - mean_id) * (t - mean_t) for i, t in zip(ids, times)) / spread) if spread else 1 / FPS
            start = mean_t - period * mean_id
            error = max(abs(t - (start + period * i)) for i, t in zip(ids, times))
            info.update(frames_log=count, start=round(start, 4), period=round(period, 6),
                        max_error_ms=round(error * 1000, 1), missing_ids=(max(ids) + 1 - count) if ids else 0)
            # Frames that drift from the line: their own times, so the page can place them exactly.
            if error * 1000 > 500 / FPS:
                info['exact'] = [[i, round(t, 3)] for i, t in zip(ids, times)]
            made = video.get(segment)
            if made is None:
                info['precision'] = 'no_video'
            else:
                info.update(frames_video=made['frames'], video_start=made['start'])
                same = abs(made['frames'] - count) <= 1 and info['missing_ids'] == 0
                info['precision'] = 'frame' if same and error * 1000 <= 500 / FPS else 'degraded'
        out.append(info)
    return out


def civil_start(merged: dict) -> int | None:
    """The civil time (unix ms) of the drive's t0, from the GPS (the satellites' time): None without a fix."""
    for t, _lat, _lon, unix_ms in merged.get('gps', []):
        if unix_ms:
            return int(unix_ms - (t - (merged.get('t0') or t)) / 1e6)
    return None
