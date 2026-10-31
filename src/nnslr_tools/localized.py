"""Precise positions of the frames, from the localizer of openpilot.distill.

The comma's GPS fix (gpsLocationExternal, about 5 Hz) is a few metres off and late; openpilot.distill's localizer
(``localizer/`` there) smooths the raw GNSS of a segment with the car's motion, wheel speed and calibration into a
position and a velocity for every road-camera frame. ``nnslr localize`` runs it on the drives of this data root: for
each segment with rlog.zst, fcamera.hevc and ecamera.hevc in ``raw/routes/<route>/<segment>/`` it runs
``localize_child.py`` with distill's Python, which writes ``derived/localizer/<route>/<segment>/localizer.safetensors``
(with frame_info.safetensors and localizer.ionex). Segments distill already localized for itself
(``<openpilot.distill>/data/<route without dashes><segment:02d>/``) are read from there, never copied or written. A
segment the localizer cannot use (the car never above 18 km/h, no fix, too few satellites, no calibration yet:
distill's exit code 3) gets a ``localizer.failed`` note and is not tried again (``--again`` does); any other failure is
reported and tried again next time. The localizer downloads the GNSS orbits and ionosphere maps of a day the first time
(laika, from comma's gnss-data mirror, into its cache /tmp/gnss).

``nnslr gps`` then takes, for a frame of the road camera, the localizer's state at that very frame
(``decoded_frame_index``: frame_states has one row per frame of fcamera.hevc), instead of the nearest GPS fix:
``frame_fix``; nnslr_tools.gps keeps the GPS fix where the track disagrees with it. The rest of nnslr reads only this
module, with the standard library: the safetensors file is parsed by hand (8 bytes of header length, a JSON header,
little-endian float64 tensors) and the ECEF position and velocity are turned into latitude, longitude, ground speed
and bearing on the WGS84 ellipsoid.
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Callable

from nnslr_tools.manifest import parse_route_id

LOCALIZER_RELATIVE_DIR = Path('derived') / 'localizer'
TRACK = 'localizer.safetensors'
FAILED = 'localizer.failed'
CHILD = Path(__file__).resolve().parent / 'localize_child.py'
INPUTS = ('rlog.zst', 'fcamera.hevc', 'ecamera.hevc')
DISTILL_ENV = 'NNSLR_DISTILL_ROOT'
UNUSABLE_EXIT = 3
# A segment takes about 20 s of a core and 220 MB; more when the machine is busy (the child runs with nice 10).
LOCALIZE_TIMEOUT_S = 600
# The columns of a row of frame_states (distill's localizer.localizer: position ECEF, quaternion, velocity ECEF).
POSITION = slice(0, 3)
VELOCITY = slice(7, 10)
# Below this ground speed (m/s) the direction of the velocity is noise: the frame takes the bearing of the nearest
# frame before it (else after it) where the car moved, as the GPS keeps its last heading while the car stands.
MOVING_MPS = 1.0
# The largest header read, as the safetensors library has it: a longer one is not such a file.
MAX_HEADER = 100_000_000
# WGS84
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


class LocalizerError(RuntimeError):
    pass


def distill_root() -> Path:
    """NNSLR_DISTILL_ROOT, else openpilot.distill next to this checkout."""
    if os.environ.get(DISTILL_ENV):
        return Path(os.environ[DISTILL_ENV])
    return Path(__file__).resolve().parents[3] / 'openpilot.distill'


def distill_python(distill: Path | None = None) -> str:
    """The Python of openpilot.distill (its .venv), which runs the localizer; LocalizerError without it."""
    python = (distill or distill_root()) / '.venv' / 'bin' / 'python'
    if not os.access(python, os.X_OK):
        raise LocalizerError(f'no Python of openpilot.distill at {python} (see docs/INSTALL.md)')
    return str(python)


def output_dir(root: Path, route: str, segment: int) -> Path:
    parse_route_id(route)
    return Path(root) / LOCALIZER_RELATIVE_DIR / route / str(int(segment))


def distill_dir(route: str, segment: int, distill: Path | None = None) -> Path:
    """Where distill keeps a segment of its own: data/<route without dashes><segment, two digits>."""
    parse_route_id(route)
    return (distill or distill_root()) / 'data' / f'{route.replace("--", "")}{int(segment):02d}'


def track_path(root: Path, route: str, segment: int, distill: Path | None = None) -> Path | None:
    """The localizer output of a segment: ours, else distill's own, else None."""
    for directory in (output_dir(root, route, segment), distill_dir(route, segment, distill)):
        if (directory / TRACK).is_file():
            return directory / TRACK
    return None


def _counts(value: object, length: int | None = None) -> list[int]:
    """``value`` when it is a list of non-negative integers (of ``length`` of them), else ValueError."""
    if not isinstance(value, list) or (length is not None and len(value) != length) \
            or not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value):
        raise ValueError(value)
    return value


def read_safetensors(path: Path, names: tuple[str, ...]) -> dict[str, tuple[list[int], array]]:
    """{name: (shape, flat float64 array)} of the F64 tensors ``names`` of a safetensors file; only their bytes are
    read. Anything else than such a file is a LocalizerError."""
    with open(path, 'rb') as handle:
        total = os.fstat(handle.fileno()).st_size
        head = handle.read(8)
        if len(head) < 8:
            raise LocalizerError(f'{path}: not a safetensors file')
        size = struct.unpack('<Q', head)[0]
        if size > min(total - 8, MAX_HEADER):
            raise LocalizerError(f'{path}: not a safetensors file')
        try:
            header = json.loads(handle.read(size))
        except ValueError as exc:
            raise LocalizerError(f'{path}: unreadable header') from exc
        if not isinstance(header, dict):
            raise LocalizerError(f'{path}: unreadable header')
        out = {}
        for name in names:
            tensor = header.get(name)
            if not isinstance(tensor, dict) or tensor.get('dtype') != 'F64':
                raise LocalizerError(f'{path}: no float64 tensor {name}')
            try:
                shape = _counts(tensor.get('shape'))
                begin, end = _counts(tensor.get('data_offsets'), 2)
            except ValueError:
                raise LocalizerError(f'{path}: {name} has no shape or data offsets') from None
            if end - begin != 8 * math.prod(shape):
                raise LocalizerError(f'{path}: {name} does not match its shape')
            if 8 + size + end > total:
                raise LocalizerError(f'{path}: {name} is truncated')
            handle.seek(8 + size + begin)
            data = handle.read(end - begin)
            if len(data) != end - begin:
                raise LocalizerError(f'{path}: {name} is truncated')
            values = array('d')
            values.frombytes(data)
            if sys.byteorder != 'little':
                values.byteswap()
            out[name] = (shape, values)
    return out


def geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """Latitude and longitude in degrees of an ECEF point (WGS84)."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - _E2))
    for _ in range(5):
        n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - _E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lon)


def ground_motion(lat_deg: float, lon_deg: float, vx: float, vy: float, vz: float) -> tuple[float, float]:
    """Ground speed (m/s) and bearing (degrees from north, clockwise) of an ECEF velocity at a place."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    east = -math.sin(lon) * vx + math.cos(lon) * vy
    north = -math.sin(lat) * math.cos(lon) * vx - math.sin(lat) * math.sin(lon) * vy + math.cos(lat) * vz
    return math.hypot(north, east), math.degrees(math.atan2(east, north)) % 360.0


@lru_cache(maxsize=64)
def _track(path: str, mtime_ns: int) -> tuple[int, array]:
    tensors = read_safetensors(Path(path), ('frame_states',))
    shape, values = tensors['frame_states']
    if len(shape) != 2 or shape[1] < 10:
        raise LocalizerError(f'{path}: frame_states has shape {shape}')
    return shape[0], values


def load_track(path: Path) -> tuple[int, array]:
    """(frames, flat frame_states) of a localizer.safetensors, cached while the file stays the same."""
    return _track(str(path), Path(path).stat().st_mtime_ns)


def _motion(track: tuple[int, array], frame_index: int) -> tuple[float, float, float, float] | None:
    """(latitude, longitude, ground speed, bearing) of a frame of the track, None when its state is not a place."""
    frames, values = track
    start = frame_index * (len(values) // frames)
    row = values[start:start + 10]
    x, y, z = row[POSITION]
    vx, vy, vz = row[VELOCITY]
    if not all(math.isfinite(value) for value in (x, y, z, vx, vy, vz)) or math.hypot(x, y, z) < 6.3e6:
        return None
    lat, lon = geodetic(x, y, z)
    return (lat, lon) + ground_motion(lat, lon, vx, vy, vz)


def _moving_bearing(track: tuple[int, array], frame_index: int) -> float | None:
    """The bearing of the nearest frame before ``frame_index`` (else after it) where the car moved, None if none."""
    for others in (range(frame_index - 1, -1, -1), range(frame_index + 1, track[0])):
        for other in others:
            motion = _motion(track, other)
            if motion is not None and motion[2] >= MOVING_MPS:
                return motion[3]
    return None


def frame_fix(track: tuple[int, array], frame_index: int) -> dict | None:
    """The position of the road camera's frame ``frame_index`` of the segment, as nnslr gps writes it, or None."""
    if not 0 <= frame_index < track[0]:
        return None
    motion = _motion(track, frame_index)
    if motion is None:
        return None
    lat, lon, speed, bearing = motion
    if speed < MOVING_MPS:
        moving = _moving_bearing(track, frame_index)
        bearing = bearing if moving is None else moving
    return {'latitude': round(lat, 7), 'longitude': round(lon, 7), 'speed_mps': round(speed, 2),
            'bearing_deg': round(bearing, 1) % 360.0}


# -- running the localizer ---------------------------------------------------------------------------------------------

def _segments(root: Path, route: str) -> list[int]:
    directory = Path(root) / 'raw' / 'routes' / route
    if not directory.is_dir():
        return []
    return sorted(int(entry.name) for entry in directory.iterdir() if entry.name.isdigit() and entry.is_dir())


def all_routes(root: Path) -> list[str]:
    """The routes of the data root with segment folders in raw/routes."""
    base = Path(root) / 'raw' / 'routes'
    if not base.is_dir():
        return []
    found = []
    for entry in sorted(base.iterdir()):
        try:
            parse_route_id(entry.name)
        except ValueError:
            continue
        if _segments(root, entry.name):
            found.append(entry.name)
    return found


def localize_segment(root: Path, route: str, segment: int, *, distill: Path | None = None, again: bool = False,
                     timeout: float = LOCALIZE_TIMEOUT_S) -> dict:
    """Localize one segment unless done; {'segment', 'state': done|new|unusable|no_input|failed, 'message'?}."""
    distill = distill or distill_root()
    out = output_dir(root, route, segment)
    if track_path(root, route, segment, distill) is not None:
        return {'segment': segment, 'state': 'done'}
    if (out / FAILED).is_file() and not again:
        return {'segment': segment, 'state': 'unusable'}
    source = Path(root) / 'raw' / 'routes' / route / str(segment)
    if not all((source / name).is_file() for name in INPUTS):
        return {'segment': segment, 'state': 'no_input'}
    try:
        python = distill_python(distill)
    except LocalizerError as exc:
        return {'segment': segment, 'state': 'failed', 'message': str(exc)}
    # One thread each: several segments run side by side. -I: none of this folder's modules on the child's path.
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    try:
        proc = subprocess.run(['nice', '-n', '10', python, '-I', str(CHILD), str(distill), str(source), str(out)],
                              capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except subprocess.TimeoutExpired:
        return {'segment': segment, 'state': 'failed', 'message': f'more than {int(timeout)} s'}
    message = (proc.stderr.strip().splitlines() or [''])[-1][:300]
    if proc.returncode == UNUSABLE_EXIT:
        out.mkdir(parents=True, exist_ok=True)
        (out / FAILED).write_text(json.dumps({'reason': 'unusable', 'message': message}) + '\n')
        return {'segment': segment, 'state': 'unusable', 'message': message}
    if proc.returncode != 0 or not (out / TRACK).is_file():
        return {'segment': segment, 'state': 'failed', 'message': message or f'exit {proc.returncode}'}
    (out / FAILED).unlink(missing_ok=True)
    return {'segment': segment, 'state': 'new'}


def _summary(root: Path, route: str, results: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in results:
        counts[item['state']] = counts.get(item['state'], 0) + 1
    summary = {'route_id': route, 'segments': len(results), **counts,
               'failed_segments': {str(item['segment']): item.get('message')
                                   for item in sorted(results, key=lambda item: item['segment'])
                                   if item['state'] == 'failed'}}
    if not results:
        # A drive named by hand that is not linked into the data root (yet), or a typo.
        summary['error'] = f'no segment folders in {Path(root) / "raw" / "routes" / route}'
    return summary


def localize_routes(root: Path, routes: list[str], *, workers: int = 6, again: bool = False,
                    distill: Path | None = None, on_route: Callable[[dict], None] | None = None) -> list[dict]:
    """Every segment of the routes through the localizer, ``workers`` at a time across the routes; per route, in
    the order given, counts by state and the failures. ``on_route`` gets each route's as soon as it is complete."""
    routes = list(dict.fromkeys(routes))
    for route in routes:
        parse_route_id(route)
    segments = {route: _segments(root, route) for route in routes}
    results: dict[str, list[dict]] = {route: [] for route in routes}
    summaries: dict[str, dict] = {}

    def finish(route: str) -> None:
        summaries[route] = _summary(root, route, results[route])
        if on_route is not None:
            on_route(summaries[route])

    for route in routes:
        if not segments[route]:
            finish(route)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        try:
            jobs = {pool.submit(localize_segment, root, route, segment, distill=distill, again=again):
                    (route, segment) for route in routes for segment in segments[route]}
            for job in as_completed(jobs):
                route, segment = jobs[job]
                try:
                    item = job.result()
                except (LocalizerError, OSError, ValueError) as exc:
                    item = {'segment': segment, 'state': 'failed', 'message': str(exc)[:300]}
                results[route].append(item)
                if len(results[route]) == len(segments[route]):
                    finish(route)
        except BaseException:
            # Ctrl-C: the segments not started yet are dropped, not localized one after the other for nobody (the
            # running children got the Ctrl-C too).
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return [summaries[route] for route in routes]


def localize_route(root: Path, route: str, *, workers: int = 6, again: bool = False,
                   distill: Path | None = None) -> dict:
    """Every segment of a route through the localizer; counts by state, and the failures."""
    return localize_routes(root, [route], workers=workers, again=again, distill=distill)[0]


def newest_track_ns(root: Path, route: str, distill: Path | None = None) -> int:
    """The mtime of the newest localizer output of a route (ours or distill's), 0 when there is none."""
    newest = 0
    for segment in _segments(root, route):
        path = track_path(root, route, segment, distill)
        if path is not None:
            newest = max(newest, path.stat().st_mtime_ns)
    return newest
