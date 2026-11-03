#!/usr/bin/env python3
"""The maps the drives need: the Geofabrik extracts that cover the GPS positions of the drives, downloaded when missing.

  .venv/bin/python scripts/osm_maps.py [--data-root DIR] [--dry-run] [--max-gb 5]

Reads every ``derived/gps/<route>/positions.jsonl`` of the data root, keeps one position of each square of about a
kilometre (GRID_DEGREES) and looks for it in the extracts of ``raw/osm`` (the outline of each in Geofabrik's index, by
the name of its file: ``illinois-latest.osm.pbf`` is us/illinois). A position outside all of them needs a map: the
smallest extract of the index that holds it (a US state, not the US; a German state or one of its districts, not
Germany), and an extract chosen of about the same size takes the positions it holds too (FOLD_AREA: Alsace into
Baden-Württemberg, never a state into us-midwest). A region of regions (us-midwest, the US, Europe) chosen for a few
squares between the outlines of its parts (GAP_SQUARES: a lake, the sea, a GPS fix that jumped) is not downloaded:
those squares are said like the ones on no map. A file of raw/osm the index does not know covers nothing here, and is
never deleted.

The missing extracts are downloaded one at a time, the most needed first, into ``raw/osm/.<name>.part``, checked against
Geofabrik's md5 (``<url>.md5`` of the file of the day that ``-latest`` leads to) and moved to ``raw/osm/<name>`` only
when whole: a part left by a run that stopped is started again, never resumed (one younger than FRESH_PART_S belongs to
a download still running elsewhere and is left alone; a run stopped from the portal removes its own). A lock
(``raw/osm/.osm_maps.lock``) keeps two runs from downloading at the same time. All together they may take
at most --max-gb (default 5 GB, from the sizes Geofabrik gives, and while downloading) and must leave DISK_MARGIN_GB
free: a bigger one is not downloaded, and the run says what and how big. A download that fails is said, the others go
on. The new file is newer than ``derived/osm/roads.jsonl``: helpers/road_types.sh reads the map again.

Geofabrik's index (https://download.geofabrik.de/index-v1.json, about 4 MB with the outlines) is kept in
``raw/osm/geofabrik-index-v1.json`` and fetched again when older than INDEX_DAYS; without the network the one kept is
used, and the run says so. Only Geofabrik is asked (download.geofabrik.de: the index, the size and the md5 of an extract,
the extract; a redirect elsewhere is refused): no position of the drives leaves the server.

Writes ``derived/osm/maps.json`` for Data › Roads: ``{"checked_at", "index": {"date", "offline"} or null, "maps": [the
extracts of raw/osm: {"file", "id" (null: not in the index), "name", "bytes", "date"}], "needed": [the extracts the
drives need that were missing: {"id", "name", "file", "url", "bytes" (null: unknown), "drives", "positions", "state"
(downloaded, to download, too big, failed, busy), "reason"}], "outside": [the drives with positions outside every map
now: {"route", "positions"}], "nowhere": [the drives with positions in no extract at all, or only in a region of
regions chosen for a few squares], "downloads": [the last
downloads: {"id", "name", "file", "bytes", "at"}], "max_gb", "dry_run"}``.

--dry-run: what is missing, its size and the drives that need it; nothing is downloaded (the index is fetched and kept
as in a run). Exit code 0 when every map the drives need is there (or nothing was needed, or only a dry run), 1 when
some drives still miss theirs (a download failed or was too big, another run was downloading, no index), 2 for a wrong
argument. OpenStreetMap data © OpenStreetMap contributors (ODbL), extracts by Geofabrik.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import urllib.request

HOST = 'https://download.geofabrik.de/'
INDEX_URL = HOST + 'index-v1.json'
INDEX_NAME = 'geofabrik-index-v1.json'
# Geofabrik makes the extracts every day, the outlines change seldom.
INDEX_DAYS = 30
# One position of each square of 0.01° (about 1.1 km of latitude, 0.8 km of longitude here): the outlines of
# Geofabrik reach some kilometres past the borders.
GRID_DEGREES = 0.01
# An extract chosen gives its squares to another one chosen at most this many times its area: a neighbour of about its
# size (Alsace or Freiburg into Baden-Württemberg, 4×), never a region of regions (a US state into us-midwest, 8× and
# more), where one stray square would pull the squares of every state into a download of gigabytes.
FOLD_AREA = 5
# A region of regions (an extract that holds one present or chosen FOLD_AREA times smaller: us-midwest, the US, Europe)
# chosen for this many squares or fewer: they lie between the outlines of its parts (a lake, the sea, a GPS fix that
# jumped), not worth its gigabytes. Not downloaded, said like the squares on no map.
GAP_SQUARES = 5
# A part file written to in the last ten minutes: a download still running (by hand, another program).
FRESH_PART_S = 600
# Left free on the disk of raw/osm after a download (GB): osm_roads.py keeps its node index next to the data root.
DISK_MARGIN_GB = 2.0
KEEP_DOWNLOADS = 20
CHUNK = 1 << 20
USER_AGENT = 'nnslr-osm-maps/1 (sunnypilot.speedlimits; one extract at a time)'
STATUS_RELATIVE = Path('derived') / 'osm' / 'maps.json'
MAPS_RELATIVE = Path('raw') / 'osm'
SUFFIX = '-latest.osm.pbf'


class MapError(RuntimeError):
    """A map that could not be downloaded: the message says why."""


class _GeofabrikOnly(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only on Geofabrik's server (``-latest`` to the file of the day), never to another one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith(HOST):
            if fp is not None:
                fp.close()
            raise MapError(f'Geofabrik sent {req.full_url} elsewhere ({newurl}): not followed')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_GeofabrikOnly)


def _open(url: str, method: str = 'GET'):
    """A response of Geofabrik's server; nothing else is ever asked, not even by a redirect."""
    if not url.startswith(HOST):
        raise MapError(f'not a Geofabrik address: {url}')
    request = urllib.request.Request(url, method=method, headers={'User-Agent': USER_AGENT})
    return _OPENER.open(request, timeout=60)


def _gb(size: int | None) -> str:
    if size is None:
        return 'size unknown'
    return f'{size / 1e9:.1f} GB' if size >= 1e9 else f'{max(1, round(size / 1e6))} MB'


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec='seconds')


def display_name(properties: dict) -> str:
    """The name of an extract for a person: "Baden-Württemberg", "Illinois" (the index calls the US states
    "us/illinois"), "Nord-Norge" (without its English name in brackets)."""
    name = str(properties.get('name') or properties.get('id') or '').split('<br', 1)[0].strip()
    if '/' in name and ' ' not in name:
        words = name.rsplit('/', 1)[1].split('-')
        name = ' '.join('US' if word == 'us' else word if word in ('of', 'and') else word.capitalize() for word in words)
    return name


def stem_name(file: str) -> str:
    """A file of raw/osm the index does not name: "my-region-latest.osm.pbf" -> "My Region"."""
    stem = file.removesuffix(SUFFIX).removesuffix('.osm.pbf')
    return ' '.join(word.capitalize() for word in stem.replace('_', '-').split('-') if word)


def extracts(index: dict) -> list[dict]:
    """The extracts of Geofabrik's index with a pbf on download.geofabrik.de: {'id', 'name', 'url', 'file' (its name in
    raw/osm: the name of the url, or the id with '-' for '/' when two extracts share it, us/georgia and Georgia),
    'geometry' (shapely), 'area'}."""
    from shapely.geometry import shape

    found = []
    for feature in index.get('features') or ():
        properties = feature.get('properties') or {}
        url = (properties.get('urls') or {}).get('pbf')
        if not isinstance(url, str) or not url.startswith(HOST) or not url.endswith('.osm.pbf') \
                or not feature.get('geometry') or not properties.get('id'):
            continue
        geometry = shape(feature['geometry'])
        if geometry.is_empty:
            continue
        found.append({'id': str(properties['id']), 'name': display_name(properties), 'url': url,
                      'file': url.rsplit('/', 1)[1], 'geometry': geometry, 'area': geometry.area})
    shared = {}
    for extract in found:
        shared[extract['file']] = shared.get(extract['file'], 0) + 1
    for extract in found:
        if shared[extract['file']] > 1:
            extract['file'] = extract['id'].replace('/', '-') + SUFFIX
    return found


def grid_positions(root: Path) -> dict[tuple[int, int], dict]:
    """One position of each square of GRID_DEGREES the drives went through: {(a, b): {'lat', 'lon', 'routes'}}."""
    cells: dict[tuple[int, int], dict] = {}
    for path in sorted((Path(root) / 'derived' / 'gps').glob('*/positions.jsonl')):
        route = path.parent.name
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            lat, lon = item.get('latitude'), item.get('longitude')
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)) or isinstance(lat, bool):
                continue
            cell = cells.setdefault((int(lat // GRID_DEGREES), int(lon // GRID_DEGREES)),
                                    {'lat': float(lat), 'lon': float(lon), 'routes': set()})
            cell['routes'].add(route)
    return cells


def present_maps(folder: Path, by_file: dict[str, dict]) -> list[dict]:
    """The extracts in raw/osm, oldest first: {'file', 'id' (None: a file the index does not name), 'name', 'bytes',
    'date'}."""
    found = []
    for path in folder.glob('*.osm.pbf'):
        try:
            stat = path.stat()
        except OSError:
            continue
        extract = by_file.get(path.name)
        found.append({'file': path.name, 'id': extract['id'] if extract else None,
                      'name': extract['name'] if extract else stem_name(path.name), 'bytes': stat.st_size,
                      'date': _iso(stat.st_mtime), 'mtime': stat.st_mtime})
    found.sort(key=lambda item: (item['mtime'], item['file']))
    for item in found:
        del item['mtime']
    return found


def plan(cells: dict, all_extracts: list[dict], present: set[str]) -> tuple[dict[str, dict], set, dict[str, set]]:
    """Which extracts the positions outside the maps present need: ({id: {'extract', 'cells', 'routes'}}, the cells
    no extract holds or only a region of regions chosen for GAP_SQUARES or fewer, {route: its cells outside every map
    present}). Each cell goes to the smallest extract that holds it; a region of regions chosen for a few cells is
    dropped (they lie between its parts); then an extract whose cells all lie in other ones chosen, at most FOLD_AREA
    times the area of each cell's smallest, is dropped too (one download fewer)."""
    from shapely import STRtree, points

    if not cells:
        return {}, set(), {}
    keys = list(cells)
    tree = STRtree([extract['geometry'] for extract in all_extracts])
    inside: dict[tuple, list[int]] = {key: [] for key in keys}
    pairs = tree.query(points([[cells[key]['lon'], cells[key]['lat']] for key in keys]), predicate='within')
    for at, extract in zip(*pairs):
        inside[keys[at]].append(int(extract))
    outside = [key for key in keys if not any(all_extracts[index]['file'] in present for index in inside[key])]
    nowhere = {key for key in outside if not inside[key]}
    smallest: dict[tuple, int] = {}
    chosen: dict[int, set] = {}
    for key in outside:
        if inside[key]:
            smallest[key] = min(inside[key], key=lambda index: (all_extracts[index]['area'], all_extracts[index]['id']))
            chosen.setdefault(smallest[key], set()).add(key)
    # A region of regions chosen for a few cells, the smallest first: it holds a map present or chosen much smaller, and
    # the cells lie between the outlines of its parts (Lake Michigan is only in us-midwest, the sea only in the US).
    there = [index for index, extract in enumerate(all_extracts) if extract['file'] in present]
    for index in sorted(chosen, key=lambda item: all_extracts[item]['area']):
        region = all_extracts[index]
        if len(chosen[index]) > GAP_SQUARES:
            continue
        if any(part != index and all_extracts[part]['area'] * FOLD_AREA < region['area']
               and region['geometry'].contains(all_extracts[part]['geometry'].representative_point())
               for part in there + list(chosen)):
            nowhere |= chosen.pop(index)
    for index in sorted(chosen, key=lambda item: (len(chosen[item]), -all_extracts[item]['area'])):
        others = sorted((other for other in chosen if other != index), key=lambda item: all_extracts[item]['area'])
        # Only into a neighbour of about the size of the cell's own: never every state into the region around them.
        moves = {key: next((other for other in others if other in inside[key] and all_extracts[other]['area']
                            <= FOLD_AREA * all_extracts[smallest[key]]['area']), None) for key in chosen[index]}
        if others and all(target is not None for target in moves.values()):
            for key, target in moves.items():
                chosen[target].add(key)
            del chosen[index]
    needed = {}
    for index, keys_of in chosen.items():
        extract = all_extracts[index]
        needed[extract['id']] = {'extract': extract, 'cells': keys_of,
                                 'routes': sorted({route for key in keys_of for route in cells[key]['routes']})}
    by_route: dict[str, set] = {}
    for key in outside:
        for route in cells[key]['routes']:
            by_route.setdefault(route, set()).add(key)
    return needed, nowhere, by_route


def load_index(folder: Path, refresh: bool = True) -> tuple[dict | None, bool, str]:
    """Geofabrik's index, from raw/osm while younger than INDEX_DAYS, else fetched and kept there: (index or None,
    offline: the kept one used because the network failed, what to say)."""
    path = folder / INDEX_NAME
    kept = None
    if path.is_file():
        try:
            kept = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            kept = None
    if kept is not None and time.time() - path.stat().st_mtime < INDEX_DAYS * 86400:
        return kept, False, f'Geofabrik index of {_iso(path.stat().st_mtime)[:10]}'
    if refresh:
        try:
            with _open(INDEX_URL) as response:
                data = response.read()
            index = json.loads(data)
            if not isinstance(index, dict) or not isinstance(index.get('features'), list):
                raise ValueError('not a GeoJSON FeatureCollection')
            part = folder / ('.' + INDEX_NAME + '.part')
            part.write_bytes(data)
            os.replace(part, path)
            return index, False, 'Geofabrik index fetched'
        except (OSError, ValueError, MapError) as exc:
            problem = str(exc) or type(exc).__name__
            if kept is not None:
                return kept, True, (f'Geofabrik index not fetched ({problem}): offline, with the one of '
                                    f'{_iso(path.stat().st_mtime)[:10]}')
            return None, False, f'Geofabrik index not fetched ({problem}) and none kept: no map can be chosen'
    if kept is not None:
        return kept, False, f'Geofabrik index of {_iso(path.stat().st_mtime)[:10]}'
    return None, False, 'no Geofabrik index kept'


def remote_size(url: str) -> int | None:
    """The size Geofabrik gives for a file, None when it does not say."""
    try:
        with _open(url, 'HEAD') as response:
            value = response.headers.get('Content-Length')
        return int(value) if value and value.isdigit() else None
    except (OSError, ValueError, MapError):
        return None


def download(extract: dict, folder: Path, limit: int) -> int:
    """Downloads one extract into raw/osm, checked against Geofabrik's md5; returns its size. Raises MapError (with
    nothing new left in raw/osm) when it cannot."""
    target = folder / extract['file']
    part = folder / ('.' + extract['file'] + '.part')
    if target.exists():
        raise MapError(f'{target.name} is already there')
    if part.exists():
        if time.time() - part.stat().st_mtime < FRESH_PART_S:
            raise MapError(f'{part.name} is being written by another download: left alone')
        # A part of a run that stopped: never resumed, started again.
        part.unlink()
    try:
        with _open(extract['url']) as response, open(part, 'wb') as handle:
            # -latest leads to the file of the day (us-midwest-261102.osm.pbf): its own md5, as the md5 of -latest
            # may already be of the next day's around Geofabrik's update.
            dated = response.geturl()
            with _open(dated + '.md5') as answer:
                words = answer.read(4096).decode('ascii', errors='replace').split()
            want = words[0].lower() if words else ''
            if not re.fullmatch(r'[0-9a-f]{32}', want):
                raise MapError('Geofabrik gave no md5 for it')
            digest = hashlib.md5()
            written = 0
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise MapError(f'bigger than the {_gb(limit)} left of --max-gb')
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != want:
            raise MapError(f'md5 {digest.hexdigest()}, Geofabrik says {want}: the file is not kept')
        # No overwrite: a file of that name that appeared meanwhile stays.
        os.link(part, target)
        part.unlink()
        return written
    except (OSError, ValueError) as exc:
        raise MapError(str(exc) or type(exc).__name__) from exc
    finally:
        with contextlib.suppress(OSError):
            if part.exists():
                part.unlink()


@contextlib.contextmanager
def download_lock(folder: Path):
    """True while this run holds the lock of the downloads, False when another run holds it."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / '.osm_maps.lock', 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _previous(path: Path) -> list[dict]:
    try:
        downloads = json.loads(path.read_text(encoding='utf-8')).get('downloads')
    except (OSError, ValueError, AttributeError):
        return []
    return [item for item in downloads if isinstance(item, dict)] if isinstance(downloads, list) else []


def write_status(root: Path, status: dict) -> Path:
    target = Path(root) / STATUS_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    temporary.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default=os.environ.get('NNSLR_DATA_ROOT') or str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--dry-run', action='store_true', help='say what is missing and how big; download nothing')
    parser.add_argument('--max-gb', type=float, default=5.0, help='the most the downloads of a run may take (default 5)')
    args = parser.parse_args(argv)
    if args.max_gb <= 0:
        parser.error('--max-gb is more than 0')
    root = Path(args.data_root).resolve()
    folder = root / MAPS_RELATIVE
    folder.mkdir(parents=True, exist_ok=True)
    status_path = root / STATUS_RELATIVE
    cells = grid_positions(root)
    status = {'checked_at': _iso(time.time()), 'index': None, 'maps': [], 'needed': [], 'outside': [], 'nowhere': [],
              'downloads': _previous(status_path)[-KEEP_DOWNLOADS:], 'max_gb': args.max_gb, 'dry_run': args.dry_run}
    index, offline, said = load_index(folder)
    print(said)
    if index is None:
        status['maps'] = present_maps(folder, {})
        write_status(root, status)
        # Without the outlines nothing is known: a drive with positions may need a map.
        return 1 if cells else 0
    all_extracts = extracts(index)
    by_file = {extract['file']: extract for extract in all_extracts}
    index_path = folder / INDEX_NAME
    status['index'] = {'date': _iso(index_path.stat().st_mtime) if index_path.is_file() else None, 'offline': offline}
    maps = present_maps(folder, by_file)
    print('Maps: ' + (', '.join(item['name'] for item in maps) or 'none'))
    needed, nowhere, by_route = plan(cells, all_extracts, {item['file'] for item in maps})
    if not cells:
        print('No GPS positions yet (nnslr gps --all): no map needed.')
    elif not needed and not nowhere:
        print(f'Every position of the drives is on a map ({len(cells)} squares of about a km).')
    # The most needed first: what the drives need most is downloaded before a limit is reached.
    order = sorted(needed.values(), key=lambda item: (-len(item['cells']), item['extract']['id']))
    limit = int(args.max_gb * 1e9)
    total = 0
    rows = []
    for item in order:
        extract = item['extract']
        size = remote_size(extract['url'])
        row = {'id': extract['id'], 'name': extract['name'], 'file': extract['file'], 'url': extract['url'],
               'bytes': size, 'drives': item['routes'], 'positions': len(item['cells']), 'state': 'to download',
               'reason': None}
        if size is not None and total + size > limit:
            row.update(state='too big', reason=f'{_gb(size)}, beyond the {_gb(limit)} of --max-gb '
                                               f'({_gb(total + size)} with the maps before it)')
        else:
            total += size or 0
        rows.append(row)
        print(f'{extract["name"]} ({extract["id"]}, {_gb(size)}): needed by {", ".join(item["routes"])}'
              + (f' — not downloaded: {row["reason"]}' if row['state'] == 'too big' else ''))
    if nowhere:
        routes = sorted({route for key in nowhere for route in cells[key]['routes']})
        status['nowhere'] = routes
        print(f'{len(nowhere)} squares of positions on no map to download (in no extract of Geofabrik, or only between '
              f'the parts of a region: a lake, the sea, a GPS fix that jumped): {", ".join(routes)}')
    status['needed'] = rows
    if args.dry_run:
        if rows:
            todo = [row for row in rows if row['state'] == 'to download']
            known = [row['bytes'] for row in todo if row['bytes'] is not None]
            size = ('' if not todo else ', size unknown' if not known else ', ' + _gb(sum(known))
                    + (f' and {len(todo) - len(known)} of unknown size' if len(known) < len(todo) else ''))
            print(f'Dry run: {len(todo)} to download{size}; nothing downloaded.')
    elif any(row['state'] == 'to download' for row in rows):
        with download_lock(folder) as held:
            if not held:
                print('Another run is downloading the maps now: this one goes on with the maps present.')
            spent = 0
            for row in rows:
                if row['state'] != 'to download':
                    continue
                if not held:
                    row.update(state='busy', reason='another run was downloading the maps')
                    continue
                free = shutil.disk_usage(folder).free
                if row['bytes'] is not None and free - row['bytes'] < DISK_MARGIN_GB * 1e9:
                    row.update(state='failed', reason=f'{_gb(free)} free on the disk: {_gb(row["bytes"])} would leave '
                                                      f'less than {DISK_MARGIN_GB:g} GB')
                    print(f'{row["name"]}: not downloaded, {row["reason"]}')
                    continue
                print(f'Downloading {row["name"]} ({_gb(row["bytes"])}) from {row["url"]}')
                started = time.time()
                try:
                    size = download(by_file[row['file']], folder, limit - spent)
                except MapError as exc:
                    row.update(state='failed', reason=str(exc))
                    print(f'{row["name"]}: download failed, {exc}; the other maps go on')
                    continue
                spent += size
                row.update(state='downloaded', bytes=size)
                status['downloads'] = (status['downloads'] + [{'id': row['id'], 'name': row['name'],
                                                               'file': row['file'], 'bytes': size,
                                                               'at': _iso(time.time())}])[-KEEP_DOWNLOADS:]
                print(f'{row["name"]}: {_gb(size)} in {round(time.time() - started)} s, md5 ok')
        maps = present_maps(folder, by_file)
        needed, nowhere, by_route = plan(cells, all_extracts, {item['file'] for item in maps})
    failed = any(row['state'] in ('too big', 'failed', 'busy') for row in rows)
    status['maps'] = maps
    status['outside'] = [{'route': route, 'positions': len(keys)} for route, keys in sorted(by_route.items())]
    target = write_status(root, status)
    if status['outside']:
        print(f'{len(status["outside"])} drives with positions outside the maps: '
              + ', '.join(item['route'] for item in status['outside']))
    print(f'Status in {target}')
    return 1 if failed and not args.dry_run else 0


if __name__ == '__main__':
    # Stopped from the portal (SIGTERM to the job's group): an exit, so that the download's finally removes its part
    # and the next run does not take it for another download still writing.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
