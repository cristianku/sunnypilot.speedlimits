"""GPS position and country of the frames of a route, from the comma logs.

``nnslr gps --route <id>`` (or ``--all``) reads ``raw/routes/<route>/<segment>/rlog.zst`` (or the qlog) and
writes ``derived/gps/<route>/positions.jsonl``: one line per frame of the route's current pre-annotation
report, ``{"frame_key", "segment_index", "output_index", "latitude", "longitude", "speed_mps", "bearing_deg",
"position_from", "country", "country_from"}``, latitude to bearing null when the frame has no position; the
country then comes from the nearest frame with a fix (country_from "nearest_fix", else "gps").

The frame k of a segment is the first video frame at k seconds (1 frame a second, nnslr_tools.media), so
its time is the time of the segment's first encoded road frame plus k seconds; the nearest fix
(``gpsLocationExternal``, else ``gpsLocation``) within GPS_MAX_GAP_S gives its position (position_from "gps"). The
logs are read by the Python of the comma fetch (openpilot's LogReader, or pycapnp with NNSLR_OPENPILOT_ROOT:
helpers/comma_fetch_log.py). A frame of the road camera (fcamera) of a segment that openpilot.distill's localizer
went through (``nnslr localize``, nnslr_tools.localized) takes the localizer's position at that very frame instead
of its GPS fix (position_from "localizer"), unless the track disagrees with that fix (LOCALIZER_MAX_GAP_M plus the
way the car drove between the fix and the frame, and LOCALIZER_MAX_SPEED_GAP_MPS for a fix within
LOCALIZER_CHECK_GAP_S of the frame) or with too many of the fixes within LOCALIZER_CHECK_GAP_S of its segment's
frames (LOCALIZER_MAX_REFUSED): a track can go wrong where the raw GNSS is poor (its speed metres a second off, its
bearing turned round). A frame without a GPS fix keeps no position, as before, even where the track has one:
nnslr_tools.road_types takes a slow stretch without GPS for a garage.

The country comes from the borders in ``data/countries.json.gz`` (Natural Earth 1:10 million, public domain;
scripts/make_country_borders.py), Europe only: elsewhere it is None. Natural Earth is coarse where a border
follows a river: along the Rhine from Wallbach to Laufenburg it puts Bad Säckingen and Rheinfelden (Baden) in
Switzerland. ``<data root>/derived/gps/borders.json.gz``, when there (GISCO 1:1 million, EuroGeographics:
non-commercial use, so never in git; ``scripts/make_country_borders.py --gisco``), takes its place. ``nnslr gps
--countries`` gives the frames of the routes already read their country again from those borders. It is a
proposal: the reviewer confirms or corrects it with each frame (``country`` of the review).
"""
from __future__ import annotations

import bisect
import gzip
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from nnslr_tools import localized
from nnslr_tools.annotations import atomic_write, jsonl_bytes, load_jsonl
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.sign_reads import frame_order, frame_second

GPS_RELATIVE_DIR = Path('derived') / 'gps'
GPS_MAX_GAP_S = 2.0
# The localizer's position of a frame is kept when it is within this of the frame's GPS fix, in metres (plus the way
# driven at the fix's speed between the fix and the frame) and in m/s of speed; a segment whose track is refused at more
# than this share of its frames checked is not used at all.
LOCALIZER_MAX_GAP_M = 25.0
LOCALIZER_MAX_SPEED_GAP_MPS = 2.0
LOCALIZER_MAX_REFUSED = 0.2
# Only a fix this close in time to the frame checks the track's speed and counts for its segment (the GPS gives 5 a
# second): one further away, where the GPS dropped out, is where the car was up to GPS_MAX_GAP_S before or after.
LOCALIZER_CHECK_GAP_S = 0.5
BORDERS = Path(__file__).resolve().parent / 'data' / 'countries.json.gz'
# Finer borders of the data root (scripts/make_country_borders.py --gisco), used instead when there.
DETAILED_BORDERS = GPS_RELATIVE_DIR / 'borders.json.gz'
FETCH_PYTHON_ENV = 'COMMA_FETCH_PYTHON'
HELPERS = Path(__file__).resolve().parents[2] / 'helpers'

# Run by the log Python: per segment log, the first encoded road frame and the GPS fixes, as one JSON line.
_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import comma_fetch_log
from pathlib import Path
ENCODE = ('roadEncodeIdx', 'narrowRoadEncodeIdx')
GPS = ('gpsLocationExternal', 'gpsLocation')
for name in sys.argv[2:]:
    first = None
    fixes = {service: [] for service in GPS}
    try:
        for message in comma_fetch_log._messages(Path(name)):
            which = message.which()
            if which in ENCODE:
                payload = getattr(message, which)
                if int(payload.segmentId) == 0 and first is None:
                    first = int(message.logMonoTime)
            elif which in GPS:
                g = getattr(message, which)
                if getattr(g, 'hasFix', True) and (g.latitude or g.longitude):
                    fixes[which].append([int(message.logMonoTime), float(g.latitude), float(g.longitude),
                                         float(g.speed), float(g.bearingDeg)])
        error = None
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:200]
    chosen = fixes['gpsLocationExternal'] or fixes['gpsLocation']
    print(json.dumps({'log': name, 'first_frame_mono': first, 'fixes': chosen, 'error': error}), flush=True)
"""


class GpsError(RuntimeError):
    pass


def log_python() -> str:
    """The Python that reads the comma logs: COMMA_FETCH_PYTHON, else the .venv of openpilot.distill next to
    this checkout, else this one (with the [log-reader] extra and NNSLR_OPENPILOT_ROOT)."""
    if os.environ.get(FETCH_PYTHON_ENV):
        return os.environ[FETCH_PYTHON_ENV]
    distill = Path(__file__).resolve().parents[3] / 'openpilot.distill/.venv/bin/python'
    if os.access(distill, os.X_OK):
        return str(distill)
    return sys.executable


def segment_log(root: Path, route: str, segment: int) -> Path | None:
    directory = Path(root) / 'raw' / 'routes' / route / str(segment)
    for name in ('rlog.zst', 'rlog.bz2', 'rlog', 'qlog.zst', 'qlog.bz2', 'qlog'):
        if (directory / name).is_file():
            return directory / name
    return None


def read_logs(paths: list[Path], *, python: str | None = None) -> dict[str, dict]:
    """{log path: {'first_frame_mono', 'fixes': [[mono, lat, lon, speed, bearing], ...], 'error'}}."""
    if not paths:
        return {}
    proc = subprocess.run([python or log_python(), '-c', _CHILD, str(HELPERS)] + [str(p) for p in paths],
                          capture_output=True, text=True, check=False, timeout=600)
    if proc.returncode != 0:
        raise GpsError('log reader failed: ' + proc.stderr.strip()[-300:])
    result = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            item = json.loads(line)
            result[item['log']] = item
    return result


def _nearest_fix(segment_log_data: dict | None, second: float) -> tuple[list, float] | None:
    """The fix nearest to the frame at ``second`` of the segment (sign_reads.frame_second) and its distance in time
    (s), within GPS_MAX_GAP_S."""
    if not segment_log_data or segment_log_data.get('first_frame_mono') is None or not segment_log_data['fixes']:
        return None
    fixes = segment_log_data['fixes']
    when = segment_log_data['first_frame_mono'] + int(second * 1e9)
    times = [fix[0] for fix in fixes]
    at = bisect.bisect_left(times, when)
    best = None
    for index in (at - 1, at):
        if 0 <= index < len(fixes) and (best is None or abs(fixes[index][0] - when) < abs(best[0] - when)):
            best = fixes[index]
    if best is None or abs(best[0] - when) > GPS_MAX_GAP_S * 1e9:
        return None
    return best, abs(best[0] - when) / 1e9


def _fix_position(fix: list) -> dict:
    return {'latitude': round(fix[1], 6), 'longitude': round(fix[2], 6), 'speed_mps': round(fix[3], 2),
            'bearing_deg': round(fix[4], 1)}


def frame_position(segment_log_data: dict | None, second: float) -> dict | None:
    """The fix nearest to the frame at ``second`` of the segment, within GPS_MAX_GAP_S, or None."""
    nearest = _nearest_fix(segment_log_data, second)
    return None if nearest is None else _fix_position(nearest[0])


@lru_cache(maxsize=4)
def _read_borders(path: str, stamp: int | None) -> tuple:
    return tuple(json.loads(gzip.decompress(Path(path).read_bytes()))['countries'])


def _borders(root: Path | None = None) -> tuple:
    """The countries of the finer borders of the data root ``root`` when there, else of the shipped ones."""
    if root is not None:
        detailed = Path(root) / DETAILED_BORDERS
        try:
            return _read_borders(str(detailed), detailed.stat().st_mtime_ns)
        except (OSError, ValueError, KeyError):
            pass
    return _read_borders(str(BORDERS), None)


def _inside(lon: float, lat: float, ring: list) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def country_of(latitude: float | None, longitude: float | None, root: Path | None = None) -> str | None:
    """ISO 3166-1 alpha-2 code of the country at a position ("CH", "IT", "PL"...), None outside the borders
    (Europe) or without a position; with the finer borders of the data root ``root`` when there."""
    if latitude is None or longitude is None:
        return None
    for country in _borders(root):
        west, south, east, north = country['bbox']
        if not (west <= longitude <= east and south <= latitude <= north):
            continue
        for polygon in country['polygons']:
            if _inside(longitude, latitude, polygon[0]) and not any(_inside(longitude, latitude, hole)
                                                                      for hole in polygon[1:]):
                return country['code']
    return None


def country_names() -> dict[str, str]:
    """{code: English name} of the countries the shipped borders know (the finer ones have the same codes)."""
    return {country['code']: country['name'] for country in _borders()}


# Every country of the world, coarser (Natural Earth 1:50m, public domain), each with its continent
# (scripts/make_country_borders.py --world): where a photo of the Mapillary Traffic Sign Dataset was taken.
WORLD_BORDERS = Path(__file__).resolve().parent / 'data' / 'world.json.gz'
# The continents, in the order the portal lists them, with their names.
CONTINENTS = {'europe': 'Europe', 'asia': 'Asia', 'africa': 'Africa', 'north_america': 'North America',
              'south_america': 'South America', 'oceania': 'Oceania', 'antarctica': 'Antarctica'}
# The countries of the European borders that the world borders do not draw.
EXTRA_CONTINENTS = {'GI': 'europe'}


def _world() -> tuple:
    return _read_borders(str(WORLD_BORDERS), None)


def _country_in(countries: tuple, latitude: float, longitude: float) -> dict | None:
    for country in countries:
        west, south, east, north = country['bbox']
        if not (west <= longitude <= east and south <= latitude <= north):
            continue
        for polygon in country['polygons']:
            if _inside(longitude, latitude, polygon[0]) and not any(_inside(longitude, latitude, hole)
                                                                      for hole in polygon[1:]):
                return country
    return None


# A photo on a coast the coarse world borders cut off (in the sea on their map) takes the country nearest to it, up to
# this far (degrees of latitude: about 10 km).
COAST_DEGREES = 0.1


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
    x, y = ax + t * dx - px, ay + t * dy - py
    return x * x + y * y


def _nearest_country(countries: tuple, latitude: float, longitude: float, within: float = COAST_DEGREES) -> dict | None:
    """The country whose border is nearest to a position outside every country, if at most ``within`` away (the
    longitude scaled to the latitude)."""
    import math
    scale = max(0.1, math.cos(math.radians(latitude)))
    best, best_distance = None, (within * within)
    for country in countries:
        west, south, east, north = country['bbox']
        reach = within / scale
        if not (west - reach <= longitude <= east + reach and south - within <= latitude <= north + within):
            continue
        for polygon in country['polygons']:
            ring = polygon[0]
            for (ax, ay), (bx, by) in zip(ring, ring[1:]):
                if min(ax, bx) - reach > longitude or max(ax, bx) + reach < longitude:
                    continue
                if min(ay, by) - within > latitude or max(ay, by) + within < latitude:
                    continue
                distance = _segment_distance(longitude * scale, latitude, ax * scale, ay, bx * scale, by)
                if distance <= best_distance:
                    best, best_distance = country, distance
    return best


def continent_of(code: str | None) -> str | None:
    """The continent key (CONTINENTS) of an ISO country code, as Natural Earth has it (Russia in Europe, Turkey and
    Cyprus in Asia), or None for a code it does not know."""
    if not code:
        return None
    for country in _world():
        if country['code'] == code:
            return country['continent']
    return EXTRA_CONTINENTS.get(code)


def world_place(latitude: float | None, longitude: float | None, root: Path | None = None) -> tuple[str | None, str | None]:
    """(country code, continent key) of a position anywhere: the European borders first (country_of, finer), then
    the world's, then the country nearest to a coast they cut (COAST_DEGREES); (None, None) at sea, or without a
    position."""
    if latitude is None or longitude is None:
        return None, None
    code = country_of(latitude, longitude, root)
    if code is None:
        found = _country_in(_world(), latitude, longitude) or _nearest_country(_world(), latitude, longitude)
        if found is None:
            return None, None
        return found['code'], found['continent']
    return code, continent_of(code)


def world_names() -> dict[str, str]:
    """{code: English name} of every country of the world borders (and of the European ones)."""
    names = {country['code']: country['name'] for country in _world()}
    names.update({code: name for code, name in country_names().items() if code not in names})
    return names


def positions_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / GPS_RELATIVE_DIR / route / 'positions.jsonl'


def _agrees(precise: dict, fix: dict, gap_s: float) -> bool:
    """Whether the localizer's position of a frame is within LOCALIZER_MAX_GAP_M of its GPS fix ``gap_s`` seconds away,
    plus the way driven at the fix's speed in that time, and, for a fix within LOCALIZER_CHECK_GAP_S, within
    LOCALIZER_MAX_SPEED_GAP_MPS of its speed."""
    north = math.radians(precise['latitude'] - fix['latitude']) * 6371000.0
    east = math.radians(precise['longitude'] - fix['longitude']) * 6371000.0 * math.cos(math.radians(fix['latitude']))
    if math.hypot(north, east) > LOCALIZER_MAX_GAP_M + fix['speed_mps'] * gap_s:
        return False
    return gap_s > LOCALIZER_CHECK_GAP_S or abs(precise['speed_mps'] - fix['speed_mps']) <= LOCALIZER_MAX_SPEED_GAP_MPS


def localizer_positions(root: Path, route: str, rows: list[dict], fixes: list[dict | None], gaps: list[float | None],
                        errors: list[str], distill: Path | None = None) -> tuple[dict[int, dict], list[int]]:
    """{index of a row: the localizer's position} for the rows of the road camera with a GPS fix (``fixes``, at
    ``gaps`` seconds from the frame) whose segment has a localizer track (localized.frame_fix at their
    decoded_frame_index), where it agrees with that fix (_agrees); and the segments whose track disagrees with too
    many of the fixes within LOCALIZER_CHECK_GAP_S to be used (LOCALIZER_MAX_REFUSED)."""
    by_segment: dict[int, list[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        frame = row.get('decoded_frame_index')
        if row.get('camera_stream') == 'narrow_road' and isinstance(frame, int) and not isinstance(frame, bool):
            by_segment.setdefault(row['segment_index'], []).append((index, frame))
    precise: dict[int, dict] = {}
    refused = []
    for segment, frames in sorted(by_segment.items()):
        path = localized.track_path(root, route, segment, distill)
        if path is None:
            continue
        try:
            track = localized.load_track(path)
        except (localized.LocalizerError, OSError, ValueError) as exc:
            errors.append(f'segment {segment}: {exc}')
            continue
        found = {}
        for index, frame in frames:
            fix = localized.frame_fix(track, frame) if fixes[index] is not None else None
            if fix is not None:
                found[index] = fix
        disagree = {index for index in found if not _agrees(found[index], fixes[index], gaps[index])}
        checked = [index for index in found if gaps[index] <= LOCALIZER_CHECK_GAP_S]
        if len(disagree.intersection(checked)) > LOCALIZER_MAX_REFUSED * len(checked):
            refused.append(segment)
            continue
        precise.update({index: fix for index, fix in found.items() if index not in disagree})
    return precise, refused


def build_positions(root: Path, route: str, rows: list[dict], *, python: str | None = None,
                    workers: int = 8, distill: Path | None = None) -> dict:
    """Write positions.jsonl for the report ``rows`` of a route; returns counts."""
    segments = sorted({row['segment_index'] for row in rows})
    logs = {segment: segment_log(root, route, segment) for segment in segments}
    found = [path for path in logs.values() if path is not None]
    groups = [found[i::workers] for i in range(workers) if found[i::workers]]
    data = {}
    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        for part in pool.map(lambda group: read_logs(group, python=python), groups):
            data.update(part)
    ordered = sorted(rows, key=frame_order)
    fixes = []
    gaps = []
    errors = []
    for row in ordered:
        path = logs.get(row['segment_index'])
        segment_data = data.get(str(path)) if path is not None else None
        if segment_data and segment_data.get('error'):
            errors.append(f"segment {row['segment_index']}: {segment_data['error']}")
        nearest = _nearest_fix(segment_data, frame_second(row))
        fixes.append(None if nearest is None else _fix_position(nearest[0]))
        gaps.append(None if nearest is None else nearest[1])
    precise, refused = localizer_positions(root, route, ordered, fixes, gaps, errors, distill)
    lines = []
    located = 0
    countries: dict[str, int] = {}
    for index, row in enumerate(ordered):
        position, source = (precise[index], 'localizer') if index in precise else (fixes[index], 'gps')
        country = country_of(position['latitude'], position['longitude'], root) if position else None
        line = {'frame_key': row['frame_key'], 'segment_index': row['segment_index'],
                'output_index': row['output_index'], 'latitude': None, 'longitude': None, 'speed_mps': None,
                'bearing_deg': None, 'position_from': None, 'country': country,
                'country_from': 'gps' if country else None}
        if position:
            line.update(position)
            line['position_from'] = source
            located = located + 1
        if country:
            countries[country] = countries.get(country, 0) + 1
        lines.append(line)
    _nearest_countries(lines)
    target = positions_path(root, route)
    atomic_write(target, jsonl_bytes(lines))
    return {'route_id': route, 'frames': len(lines), 'located': located, 'from_localizer': len(precise),
            'localizer_refused_segments': refused, 'countries': countries,
            'segments_without_log': [segment for segment, path in logs.items() if path is None],
            'errors': sorted(set(errors)), 'output': str(target)}


def positions_current(root: Path, route: str, distill: Path | None = None) -> bool:
    """Whether positions.jsonl of a route is there and need not be read again for the localizer: its lines say where
    their position comes from (position_from) and no localizer track of the route is newer; or the route has no
    track at all (the GPS would give the same). Only its first line is read."""
    path = positions_path(root, route)
    try:
        stamp = path.stat().st_mtime_ns
        with path.open('rb') as handle:
            first = handle.readline()
    except OSError:
        return False
    newest = localized.newest_track_ns(root, route, distill)
    if not newest:
        return True
    return (not first or b'"position_from"' in first) and newest <= stamp


def _nearest_countries(lines: list[dict]) -> None:
    """A frame without a fix (the GPS still searching, a tunnel) takes the country of the nearest frame that has
    one: a route rarely crosses a border, and the reviewer sees where the proposal comes from."""
    located_at = [index for index, line in enumerate(lines) if line['country'] and line['country_from'] == 'gps']
    for index, line in enumerate(lines):
        if line['country_from'] == 'gps':
            continue
        line['country'], line['country_from'] = None, None
        if located_at:
            nearest = min(located_at, key=lambda other: abs(other - index))
            line['country'] = lines[nearest]['country']
            line['country_from'] = 'nearest_fix'


def recountry_positions(root: Path, route: str) -> dict:
    """The country of every frame of positions.jsonl again, from its position and the borders now (country_of):
    the logs are not read again. Returns counts, with the frames whose country changed."""
    path = positions_path(root, route)
    if not path.is_file():
        raise GpsError(f'{route}: no positions yet: run nnslr gps --route {route} first')
    lines = [line for line in load_jsonl(path) if isinstance(line, dict) and 'frame_key' in line]
    before = [line.get('country') for line in lines]
    countries: dict[str, int] = {}
    for line in lines:
        country = country_of(line.get('latitude'), line.get('longitude'), root)
        line['country'], line['country_from'] = country, 'gps' if country else None
        if country:
            countries[country] = countries.get(country, 0) + 1
    _nearest_countries(lines)
    changed = {}
    for old, line in zip(before, lines):
        if old != line['country']:
            pair = f'{old}->{line["country"]}'
            changed[pair] = changed.get(pair, 0) + 1
    if changed:
        atomic_write(path, jsonl_bytes(lines))
    return {'route_id': route, 'frames': len(lines), 'countries': countries, 'changed': changed,
            'output': str(path)}


def load_positions(root: Path, route: str) -> dict:
    """{frame_key: position line} of a route, {} when ``nnslr gps`` has not run for it."""
    path = positions_path(root, route)
    if not path.is_file():
        return {}
    return {line['frame_key']: line for line in load_jsonl(path) if isinstance(line, dict) and 'frame_key' in line}
