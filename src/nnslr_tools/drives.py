"""A route as the drive it was: when (the day and the time it started, how long), and from where to where.

- When: the time of the last segment video copied from the comma (each video is written as its minute ends); the
  drive started one minute per segment before (segments are numbered from 0, one a minute). The first videos can
  carry an older date (the comma's clock before its GPS time), so only the last one is trusted. Swiss time.
- Where: the first and the last GPS position of the route (derived/gps/<route>/positions.jsonl, nnslr gps), named
  by the nearest place of derived/osm/places.jsonl (scripts/osm_places.py, every extract of raw/osm), a city or town
  before a village; a hamlet only when no other place is near (FALLBACK).

Everything is read on the server; a route without videos, positions or places just has less to say.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LOCAL = ZoneInfo('Europe/Zurich')
# How much a kind of place counts against its distance: a town 2 km away beats a village 1.5 km away.
WEIGHT = {'city': 3.0, 'town': 2.0, 'village': 1.0, 'suburb': 0.3, 'hamlet': 0.5}
# The places that name a position only when no other is within MAX_WEIGHTED_KM: a hamlet, a few houses at a crossroad
# (some 20,000 in the extracts of the drives, mostly in the US, often all there is there), never before a village.
FALLBACK = ('hamlet',)
# Further than this from any place (km, weighted): no name.
MAX_WEIGHTED_KM = 6.0
# The places by squares of this size (degrees), the box place_name looks in.
BOX_LAT, BOX_LON = 0.1, 0.15


@lru_cache(maxsize=2)
def _places(path: str, mtime: float) -> tuple[tuple[dict, ...], dict[tuple[int, int], list[dict]]]:
    """The places of the file and the same by squares of BOX_LAT x BOX_LON: the map of several countries has some
    30,000."""
    items = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            items.append(json.loads(line))
    squares: dict[tuple[int, int], list[dict]] = {}
    for item in items:
        squares.setdefault((math.floor(item['lat'] / BOX_LAT), math.floor(item['lon'] / BOX_LON)), []).append(item)
    return tuple(items), squares


def _read_places(root: Path) -> tuple[tuple[dict, ...], dict[tuple[int, int], list[dict]]]:
    path = Path(root) / 'derived' / 'osm' / 'places.jsonl'
    try:
        return _places(str(path), path.stat().st_mtime)
    except OSError:
        return (), {}


def places(root: Path) -> tuple[dict, ...]:
    return _read_places(root)[0]


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return 6371.0 * math.hypot(x, math.radians(lat2 - lat1))


def place_name(root: Path, latitude: float, longitude: float) -> str | None:
    """The place a position is in or near: the smallest distance divided by WEIGHT, a FALLBACK place only when no other
    is near enough."""
    squares = _read_places(root)[1]
    a, b = math.floor(latitude / BOX_LAT), math.floor(longitude / BOX_LON)
    best = {False: (MAX_WEIGHTED_KM, None), True: (MAX_WEIGHTED_KM, None)}
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for item in squares.get((a + da, b + db), ()):
                if abs(item['lat'] - latitude) > BOX_LAT or abs(item['lon'] - longitude) > BOX_LON:
                    continue
                fallback = item['place'] in FALLBACK
                score = _km(latitude, longitude, item['lat'], item['lon']) / WEIGHT.get(item['place'], 1.0)
                if score < best[fallback][0]:
                    best[fallback] = score, item['name']
    return best[False][1] or best[True][1]


def _ends(root: Path, route: str) -> tuple[dict | None, dict | None]:
    path = Path(root) / 'derived' / 'gps' / route / 'positions.jsonl'
    first = last = None
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None, None
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get('latitude') is not None and item.get('longitude') is not None:
            first = first or item
            last = item
    return first, last


def _times(root: Path, route: str) -> tuple[datetime | None, datetime | None]:
    last_segment, last_time = -1, None
    for folder in (Path(root) / 'raw' / 'routes' / route).glob('*'):
        video = folder / 'fcamera.hevc'
        if folder.name.isdigit() and video.exists() and int(folder.name) > last_segment:
            last_segment, last_time = int(folder.name), video.stat().st_mtime
    if last_time is None:
        return None, None
    end = datetime.fromtimestamp(last_time, timezone.utc)
    return end - timedelta(minutes=last_segment + 1), end


def drive_of(root: Path, route: str) -> dict[str, Any]:
    """{'when': "Thu 23 Oct 2026 · 08:11", 'minutes', 'start' (ISO), 'from', 'to', 'title'} of a route; the parts
    that cannot be known are None ('title' is "Olten → Zürich", "around Olten" or None)."""
    start, end = _times(root, route)
    first, last = _ends(root, route)
    origin = place_name(root, first['latitude'], first['longitude']) if first else None
    destination = place_name(root, last['latitude'], last['longitude']) if last else None
    title = None
    if origin and destination:
        title = 'around ' + origin if origin == destination else origin + ' → ' + destination
    elif origin or destination:
        title = origin or destination
    local = start.astimezone(LOCAL) if start else None
    return {'when': local.strftime('%a %-d %b %Y · %H:%M') if local else None,
            'minutes': round((end - start).total_seconds() / 60) if start and end else None,
            'start': start.isoformat(timespec='minutes') if start else None,
            'from': origin, 'to': destination, 'title': title}
