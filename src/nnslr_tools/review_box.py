"""The machine of the outside reviewers, kept apart from this server: "the review box".

People from outside work on a copy of the site that holds only the drives given to them: no training, no model, no key
towards this server. Nothing there can reach here, so everything moves from this side, over rsync, on an account that
can only copy files inside one folder of the box (rrsync, no shell).

    this server   nnslr review-box sync    Data › Reviewers, by hand and on a timer
    the box       nnslr review-box tick    a timer of its own, every minute

rsync never touches the live databases of the box. A sync leaves a package in inbox/, and the tick, the only program
there that writes both sides, installs it; the tick writes in outbox/ one package per drive with what the people did,
and a sync reads only that. A package is the same on both machines (reviews_of): the rows of the signs and of the
speeds, and the lines of the marks of Sign distance.

What a drive takes there: its frames, its video, its report and the reduced logs of the pages. The raw logs stay here.
The positions are not sent: a reduced log travels with its track moved to the middle of nowhere (the turns of the track
stay, Speed review reads them), without the names of the roads; of derived/gps only the country of each frame travels;
a saved review travels without the position of its frame, and takes it back here when it returns; the rest loses its
places as the answers to a guest do (scrubbed). The video can still show where a drive was.

What comes back comes from a machine open to the internet and becomes training labels, so it is read as a stranger's:
only the drives on the box, only the kinds given, only under the names of the people who hold them. Each sync compares
three things: what was sent (the base, web/review_box/sent), what the box holds and what this server holds now. A frame,
or the whole of a store for the others, changed only on the box is taken; changed on both sides, this server's stays and
the other is kept aside in web/review_box/aside.

"Review completed" of a reviewer ends that work: the sync takes its results, takes the work back, and when nobody holds
work on a drive any more the drive leaves the box, one sync later so that a last save is not lost.
"""
from __future__ import annotations

import contextlib
import fcntl
import gzip
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from nnslr_tools import drive_review, drive_video, sign_distance, speed_review, video_frames
from nnslr_tools.annotations import atomic_write
from nnslr_tools.drive_logs import EXTRACT_VERSION, cache_dir
from nnslr_tools.gps import load_positions, positions_path
from nnslr_tools.review_access import DRIVE, GUEST_PREFIX, KIND_KEYS, Assignments, scrubbed
from nnslr_tools.review_auth import auth_dir, write_private
from nnslr_tools.review_db import ReviewRejected, ReviewStore, db_path, validate_reviewed_frame
from nnslr_tools.review_service import GPS_FIELDS
from nnslr_tools.sign_reads import CAMERA_FPS, READ_INDEX_BASE, is_read_frame

FORMAT = 1
# This server's own files: the link, the state, the bases and what the last sync read.
OWN_RELATIVE = Path('web') / 'review_box'
# On the box, next to the data root (NNSLR_DATA_ROOT, /srv/reviews/data).
INBOX = 'inbox'
OUTBOX = 'outbox'
PEOPLE_FILE = 'people.json'
INSTALLED_FILE = 'installed.json'
PACKAGE = re.compile(DRIVE + r'\.json')
REMOVAL = re.compile(DRIVE + r'\.remove')
# The folders and files of a drive that go as they are; those with places go through stage_without_places.
AS_THEY_ARE = ('derived/preannotations/{route}', 'derived/frames/{route}', 'derived/road_type_frames/{route}',
               'derived/car_motion/{route}.json', 'derived/road_type_signs/{route}.json', 'derived/video/{route}.mp4',
               'derived/video/{route}.json')
# The map's road types and limits of the frames: Map now and the Server today lane of the player.
ROADS = 'derived/road_types/{route}.jsonl'
# The map's junctions, sent until Roads went (7 Nov 2026): still deleted with a drive.
JUNCTIONS = 'derived/junctions/{route}.json'
# What a removal deletes on the box, besides the rows and the marks.
ON_THE_BOX = (*AS_THEY_ARE, JUNCTIONS, ROADS, 'derived/gps/{route}', 'derived/drive_logs/{route}',
              f'{video_frames.CACHE_DIR}/{{route}}', 'raw/routes/{route}')
# A package read from the box is never bigger than this: a full drive of reviews is a few MB.
MAX_PACKAGE = '256M'
SSH_TIMEOUT = 20
RSYNC_TIMEOUT = 300
# A drive nobody holds leaves the box this long after the last sync that told the box: the tick has run by then, so the
# last package holds the last save.
LEAVING_SECONDS = 150
# The drives that left, in Data › Reviewers.
FINISHED_KEPT = 30
TIME_TEXT = re.compile(r'[0-9TZ:+.-]{10,35}')

REVIEW_COLUMNS = ('route_id', 'run_id', 'frame_key', 'segment_index', 'output_index', 'payload', 'reviewer', 'confirmed',
                  'revision', 'updated_at', 'reviewer_kind', 'channel')
REVIEW_HISTORY_COLUMNS = ('route_id', 'run_id', 'frame_key', 'revision', 'payload', 'reviewer', 'confirmed', 'saved_at',
                          'client', 'reviewer_kind', 'channel')
# A package from before 8 Nov 2026 has no blocks: none. The stretches are read by nothing since, and still go with their
# drive.
SPEED_TABLES = {'stretches': ('speed_stretches', ('route', 'start_ns', 'end_ns', 'kph', 'state', 'known_ns', 'note',
                                                  'reviewer', 'revision')),
                'blocks': ('speed_blocks', ('route', 'start_ns', 'end_ns', 'kph', 'note', 'reviewer', 'revision')),
                'works': ('speed_works', ('route', 'start_ns', 'end_ns', 'reviewer', 'revision')),
                'checked': ('speed_works_checked', ('route', 'end_ns', 'reviewer', 'revision'))}
SPEED_HISTORY_COLUMNS = ('route', 'revision', 'action', 'start_ns', 'end_ns', 'kph', 'state', 'known_ns', 'note',
                         'reviewer', 'at')
# The stores made of lines, and the kind of work that writes each.
LINE_STORES = {'distance': 'distance'}
DRIVE_TABLES = ('scans', 'log_files', 'episodes')
# The columns that must be numbers in a package from the box, whatever else it holds.
INTEGERS = frozenset({'revision', 'start_ns', 'end_ns'})
OPEN_ENDED = SPEED_TABLES['works'][1]


class BoxError(RuntimeError):
    """What stops a sync, in words for Data › Reviewers."""


class Busy(BoxError):
    """Another round is running: this one has nothing to do, as when the timer fires during a long send."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA busy_timeout = 30000')
    return connection


def _schemas(root: Path) -> None:
    """The databases exist, with their tables: a box starts empty."""
    ReviewStore(db_path(root))
    speed_review.SpeedReviewStore(root)
    drive_review.DriveReviewStore(root)


def _rows(connection: sqlite3.Connection, sql: str, *args) -> list[dict]:
    return [dict(row) for row in connection.execute(sql, args)]


def _insert(connection: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: list[dict], verb='INSERT') -> None:
    """The columns are this file's, never the keys of a package."""
    sql = f'{verb} INTO {table} ({", ".join(columns)}) VALUES ({", ".join("?" * len(columns))})'
    connection.executemany(sql, [[row[name] for name in columns] for row in rows])


@contextlib.contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    connection.execute('BEGIN IMMEDIATE')
    try:
        yield connection
    except BaseException:
        connection.execute('ROLLBACK')
        raise
    connection.execute('COMMIT')


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """One sync, or one tick, at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy('Another copy with the review machine is running.') from None
        yield


# ---------- a package: what people did on a drive, the same on both machines ----------

def _lines(path: Path) -> list[str]:
    try:
        return [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def _line_paths(root: Path, route: str) -> dict[str, Path]:
    return {'distance': sign_distance.marks_path(root, route)}


def _write_lines(path: Path, lines: list[str]) -> None:
    if not lines and not path.exists():
        return
    atomic_write(path, ''.join(line + '\n' for line in lines).encode('utf-8'))
    path.chmod(0o600)


def _signs_of(connection: sqlite3.Connection, route: str) -> dict:
    order = 'ORDER BY run_id, frame_key'
    columns = ', '.join(REVIEW_HISTORY_COLUMNS)
    return {'reviews': _rows(connection, f'SELECT {", ".join(REVIEW_COLUMNS)} FROM frame_reviews WHERE route_id = ? {order}', route),
            'history': _rows(connection, f'SELECT {columns} FROM frame_review_history WHERE route_id = ? '
                                         f'{order}, revision, history_id', route)}


def _speed_of(connection: sqlite3.Connection, route: str) -> dict:
    found = {key: _rows(connection, f'SELECT {", ".join(columns)} FROM {table} WHERE route = ? ORDER BY {columns[1]}', route)
             for key, (table, columns) in SPEED_TABLES.items()}
    found['history'] = _rows(connection, f'SELECT {", ".join(SPEED_HISTORY_COLUMNS)} FROM speed_history WHERE route = ? '
                                         'ORDER BY revision', route)
    return found


def reviews_of(root: Path, route: str) -> dict:
    with contextlib.closing(_open(db_path(root))) as signs, contextlib.closing(_open(speed_review.db_path(root))) as speed:
        package = {'format': FORMAT, 'route': route, 'signs': _signs_of(signs, route), 'speed': _speed_of(speed, route)}
    for key, path in _line_paths(root, route).items():
        package[key] = _lines(path)
    return package


# ---------- no places ----------

def _extracts(root: Path, route: str) -> list[Path]:
    return sorted(cache_dir(root, route).glob(f'*.v{EXTRACT_VERSION}.json.gz'))


def _origin(paths: list[Path]) -> tuple[float, float] | None:
    """The first fix of a drive: what its whole track is moved by."""
    for path in sorted(paths, key=lambda item: int(item.name.split('.')[0])):
        gps = json.loads(gzip.decompress(path.read_bytes())).get('gps') or []
        if gps:
            return gps[0][1], gps[0][2]
    return None


def without_places(extract: dict, origin: tuple[float, float] | None) -> dict:
    """A reduced log with its track moved around 0, 0, in the same scale, and no road names. The headings stay: the
    longitude is scaled here as a heading scales it at the drive's latitude."""
    moved = dict(extract)
    moved['series'] = {key: value for key, value in (extract.get('series') or {}).items() if key != 'road_name'}
    moved['gps'] = []
    if origin is not None:
        scale = math.cos(math.radians(origin[0]))
        moved['gps'] = [[t, round(latitude - origin[0], 7), round((longitude - origin[1]) * scale, 7), unix]
                        for t, latitude, longitude, unix in extract.get('gps') or []]
    return moved


def _stage(stage: Path, relative: str, text: str) -> None:
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')


def _payload_text(payload: dict) -> str:
    """As ReviewStore.save_review writes a payload: the same review reads the same on both machines."""
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _without_positions(signs: dict) -> dict:
    """The sign reviews of a drive without the position each one saved of its frame."""
    def plain(row: dict) -> dict:
        payload = json.loads(row['payload'])
        return {**row, 'payload': _payload_text({**payload, 'gps': None} if payload.get('gps') else payload)}

    return {key: [plain(row) for row in rows] for key, rows in signs.items()}


def _with_position(row: dict, payload: dict, located: dict) -> dict:
    """A row from the box with the position of its frame, which only this server knows: for a frame between the
    seconds the one of the nearest frame of its segment, as ReviewService._added_rows gives it."""
    segment, index = (int(part) for part in row['frame_key'].split('/'))
    position = located.get(row['frame_key'])
    if position is None and index >= READ_INDEX_BASE:
        second = (index - READ_INDEX_BASE) / CAMERA_FPS
        position = min((line for line in located.values() if line.get('segment_index') == segment),
                       key=lambda line: abs(line.get('output_index', 0) - second), default=None)
    position = position or {}
    place = {name: position.get(name) for name in GPS_FIELDS} if position.get('latitude') is not None else None
    return {**row, 'payload': _payload_text({**payload, 'gps': place})}


def stage_without_places(root: Path, route: str, stage: Path) -> None:
    """The files of a drive that hold places, without them, in a folder laid out as the data root."""
    paths = _extracts(root, route)
    origin = _origin(paths)
    for path in paths:
        extract = json.loads(gzip.decompress(path.read_bytes()))
        target = stage / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(_text(without_places(extract, origin)).encode('utf-8'), 6))
    source = root / ROADS.format(route=route)
    if source.is_file():
        _stage(stage, ROADS.format(route=route),
               ''.join(_text(scrubbed(json.loads(line))) + '\n' for line in _lines(source)))
    source = positions_path(root, route)
    if source.is_file():
        lines = [json.loads(line) for line in _lines(source)]
        _stage(stage, str(source.relative_to(root)),
               ''.join(_text({name: line.get(name) for name in ('frame_key', 'country', 'country_from')}) + '\n'
                       for line in lines))


def drive_rows(root: Path, route: str) -> dict:
    """The rows of the logs' scan of a drive, which the pages of the box read in place of the logs."""
    with contextlib.closing(_open(drive_review.db_path(root))) as connection:
        rows = {table: _rows(connection, f'SELECT * FROM {table} WHERE route = ?', route) for table in DRIVE_TABLES}
    for row in rows['scans']:
        row['summary'] = _text(scrubbed(json.loads(row['summary']))) if row['summary'] else row['summary']
    for row in rows['log_files']:
        row['path'] = Path(row['path']).name
    for row in rows['episodes']:
        row['payload'] = _text(scrubbed(json.loads(row['payload'])))
    return rows


def files_of(root: Path, route: str) -> list[str]:
    """What goes as it is, relative to the data root: the folders and files of the drive and its road video alone."""
    found = [item.format(route=route) for item in AS_THEY_ARE if (root / item.format(route=route)).exists()]
    for _, video in drive_video.minutes(root, route):
        found.append(str(video.relative_to(root)))
        if video.with_name(video_frames.WIDE_VIDEO).is_file():
            found.append(str(video.with_name(video_frames.WIDE_VIDEO).relative_to(root)))
    return found


# ---------- the box's side ----------

def install_drive(root: Path, package: dict) -> None:
    """A drive arrives: its rows and its marks as this server holds them."""
    route = package['route']
    with contextlib.closing(_open(db_path(root))) as connection, _transaction(connection):
        connection.execute('DELETE FROM frame_reviews WHERE route_id = ?', (route,))
        _insert(connection, 'frame_reviews', REVIEW_COLUMNS, package['signs']['reviews'])
        known = {(row['run_id'], row['frame_key'], row['revision'])
                 for row in _rows(connection, 'SELECT run_id, frame_key, revision FROM frame_review_history WHERE route_id = ?',
                                  route)}
        _insert(connection, 'frame_review_history', REVIEW_HISTORY_COLUMNS,
                [row for row in package['signs']['history'] if (row['run_id'], row['frame_key'], row['revision']) not in known])
    with contextlib.closing(_open(speed_review.db_path(root))) as connection, _transaction(connection):
        _replace_speed(connection, route, package['speed'])
        _insert(connection, 'speed_history', SPEED_HISTORY_COLUMNS, package['speed']['history'], 'INSERT OR IGNORE')
    with contextlib.closing(_open(drive_review.db_path(root))) as connection, _transaction(connection):
        for table, rows in package['drive'].items():
            if table not in DRIVE_TABLES or not rows:
                continue
            connection.execute(f'DELETE FROM {table} WHERE route = ?', (route,))
            _insert(connection, table, tuple(rows[0]), rows)
    for key, path in _line_paths(root, route).items():
        _write_lines(path, package[key])


def _replace_speed(connection: sqlite3.Connection, route: str, speed: dict) -> None:
    for key, (table, columns) in SPEED_TABLES.items():
        connection.execute(f'DELETE FROM {table} WHERE route = ?', (route,))
        _insert(connection, table, columns, speed.get(key, []))


def remove_drive(root: Path, route: str) -> None:
    """A drive leaves: its files, its rows and its marks. The histories only grow (their triggers), and hold no place."""
    for item in ON_THE_BOX:
        path = root / item.format(route=route)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    shutil.rmtree(cache_dir(root, route), ignore_errors=True)
    for path in _line_paths(root, route).values():
        path.unlink(missing_ok=True)
    with contextlib.closing(_open(db_path(root))) as connection, _transaction(connection):
        connection.execute('DELETE FROM frame_reviews WHERE route_id = ?', (route,))
    with contextlib.closing(_open(speed_review.db_path(root))) as connection, _transaction(connection):
        for table, _ in SPEED_TABLES.values():
            connection.execute(f'DELETE FROM {table} WHERE route = ?', (route,))
    with contextlib.closing(_open(drive_review.db_path(root))) as connection, _transaction(connection):
        for table in DRIVE_TABLES:
            connection.execute(f'DELETE FROM {table} WHERE route = ?', (route,))


def _write_if_changed(path: Path, text: str) -> bool:
    try:
        if path.read_text(encoding='utf-8') == text:
            return False
    except FileNotFoundError:
        pass
    atomic_write(path, text.encode('utf-8'))
    return True


def tick(root: Path, box: Path, log: Callable[[str], None] = print) -> None:
    """One pass on the box: what arrived in the inbox, then the packages of the drives and of the people in the outbox."""
    if read_link(root) is not None:
        raise BoxError('This is the server, not the review machine: a tick here would delete its drives.')
    inbox, outbox = box / INBOX, box / OUTBOX
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    with _locked(box / 'tick.lock'):
        _schemas(root)
        installed_path = box / INSTALLED_FILE
        installed = set(json.loads(installed_path.read_text())) if installed_path.is_file() else set()
        for marker in sorted(inbox.iterdir()):
            match = REMOVAL.fullmatch(marker.name)
            if match:
                remove_drive(root, match.group(1))
                installed.discard(match.group(1))
                (outbox / (match.group(1) + '.json')).unlink(missing_ok=True)
                marker.unlink()
                log(f'{match.group(1)}: removed')
        for path in sorted(inbox.iterdir()):
            match = PACKAGE.fullmatch(path.name)
            if match:
                # A drive already here is being worked on: a package sent twice never overwrites that.
                if match.group(1) not in installed:
                    install_drive(root, json.loads(path.read_text(encoding='utf-8')))
                    installed.add(match.group(1))
                    log(f'{match.group(1)}: installed')
                path.unlink()
        _write_if_changed(installed_path, _text(sorted(installed)))
        people = inbox / PEOPLE_FILE
        assignments = Assignments(auth_dir(root))
        if people.is_file():
            given = json.loads(people.read_text(encoding='utf-8'))
            assignments.replace_work(given['people'], given['work'])
            people.unlink()
        for route in sorted(installed):
            _write_if_changed(outbox / (route + '.json'), _text(reviews_of(root, route)))
        _write_if_changed(outbox / PEOPLE_FILE, _text(assignments.visits()))


# ---------- what comes back ----------

class _Refused(ValueError):
    """A package, or a part of it, that this server does not take."""


def _is_row(row, columns: tuple[str, ...], key: str, route: str) -> bool:
    # Roadworks with no end yet have none.
    return (isinstance(row, dict) and all(name in row for name in columns) and row[key] == route
            and all(type(row[name]) is int or (columns is OPEN_ENDED and name == 'end_ns' and row[name] is None)
                    for name in columns if name in INTEGERS)
            and ('reviewer' not in columns or isinstance(row['reviewer'], str)))


def _good_speed(kph) -> bool:
    return kph is None or (type(kph) is int and 0 <= kph <= speed_review.MAX_KPH)


def _read_package(path: Path, route: str) -> dict:
    try:
        package = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise _Refused(f'unreadable package: {exc}') from exc
    good = (isinstance(package, dict) and package.get('format') == FORMAT and package.get('route') == route
            and isinstance(package.get('signs'), dict) and isinstance(package.get('speed'), dict))
    if not good:
        raise _Refused('this is not a package of this drive')
    signs, speed = package['signs'], package['speed']
    tables = ((signs.get('reviews'), REVIEW_COLUMNS, 'route_id'), (signs.get('history'), REVIEW_HISTORY_COLUMNS, 'route_id'),
              (speed.get('history'), SPEED_HISTORY_COLUMNS, 'route'),
              *((speed.get(key, []), columns, 'route') for key, (_, columns) in SPEED_TABLES.items()))
    for rows, columns, key in tables:
        if not isinstance(rows, list) or not all(_is_row(row, columns, key, route) for row in rows):
            raise _Refused('a table of the package is malformed')
    if not all(_good_speed(row['kph']) for key in ('stretches', 'blocks') for row in speed.get(key, [])):
        raise _Refused('a speed of the package is out of range')
    for key in LINE_STORES:
        lines = package.get(key)
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise _Refused('the lines of the package are malformed')
    return package


def _merge_signs(root: Path, connection: sqlite3.Connection, route: str, theirs: dict, base: dict, allowed: set[str]) -> dict:
    """Frame by frame. Only frames that changed on the box are looked at; those changed here too keep this server's."""
    def by_frame(rows: list[dict]) -> dict:
        return {(row['run_id'], row['frame_key']): row for row in rows}

    before, ours = by_frame(base['reviews']), by_frame(_without_positions(_signs_of(connection, route))['reviews'])
    located = load_positions(root, route)
    report = {'taken': 0, 'conflicts': 0, 'refused': 0}
    for key, row in by_frame(theirs['reviews']).items():
        old, mine = before.get(key), ours.get(key)
        if row == old or row == mine:
            continue
        if mine != old:
            report['conflicts'] += 1
            continue
        try:
            payload = json.loads(row['payload'])
            if (row['reviewer'] not in allowed or row['reviewer_kind'] != 'human' or row['channel'] != 'web'
                    or not isinstance(payload, dict) or payload.get('reviewer') != row['reviewer']
                    or payload.get('route_id') != route or payload.get('frame_key') != row['frame_key']):
                raise _Refused('another name')
            row = _with_position(row, payload, located)
            if row['confirmed']:
                if is_read_frame(row):
                    # A frame between the seconds: the box took its picture out of the video, this server does the same.
                    video_frames.materialize(root, route, row['segment_index'], row['output_index'] - READ_INDEX_BASE)
                validate_reviewed_frame(json.loads(row['payload']), root)
            since = old['revision'] if old else 0
            lines = [_with_position(line, json.loads(line['payload']), located) for line in theirs['history']
                     if (line['run_id'], line['frame_key']) == key and line['revision'] > since]
        except (_Refused, ReviewRejected, ValueError, TypeError):
            report['refused'] += 1
            continue
        connection.execute('DELETE FROM frame_reviews WHERE route_id = ? AND run_id = ? AND frame_key = ?', (route, *key))
        _insert(connection, 'frame_reviews', REVIEW_COLUMNS, [row])
        _insert(connection, 'frame_review_history', REVIEW_HISTORY_COLUMNS, lines)
        report['taken'] += 1
    return report


def _speed_rows(speed: dict) -> dict:
    return {key: speed.get(key, []) for key in SPEED_TABLES}


def _new_names(speed: dict, base: dict) -> set:
    return {row['reviewer'] for key in SPEED_TABLES for row in speed.get(key, []) if row not in base.get(key, [])}


def _merge_speed(connection: sqlite3.Connection, route: str, theirs: dict, base: dict, allowed: set[str]) -> str:
    """The blocks and marks of a drive move together: all of the box's, or none. A box on code from before 8 Nov 2026
    sends no blocks: they stay as they were sent to it."""
    theirs = {**{key: base.get(key, []) for key in SPEED_TABLES}, **theirs}
    ours = _speed_of(connection, route)
    if _speed_rows(theirs) == _speed_rows(base) or _speed_rows(theirs) == _speed_rows(ours):
        return 'same'
    if _speed_rows(ours) != _speed_rows(base):
        return 'conflict'
    if not _new_names(theirs, base) <= allowed:
        return 'refused'
    _replace_speed(connection, route, theirs)
    top = max((row['revision'] for row in base['history']), default=0)
    _insert(connection, 'speed_history', SPEED_HISTORY_COLUMNS, [row for row in theirs['history'] if row['revision'] > top],
            'INSERT OR IGNORE')
    return 'taken'


def _merge_lines(path: Path, theirs: list[str], base: list[str], allowed: set[str]) -> str:
    """An undo rewrites these files, so a whole file comes back or nothing does."""
    ours = _lines(path)
    if theirs in (base, ours):
        return 'same'
    if ours != base:
        return 'conflict'
    try:
        lines = [json.loads(line) for line in theirs if line not in base]
        if not all(isinstance(line, dict) and line.get('reviewer') in allowed for line in lines):
            return 'refused'
    except (ValueError, TypeError):
        return 'refused'
    _write_lines(path, theirs)
    return 'taken'


def merge_drive(root: Path, route: str, theirs: dict, base: dict, allowed: dict[str, set[str]]) -> dict:
    """What the box did on a drive into this server's stores. allowed: by kind of work, who may have written."""
    with contextlib.closing(_open(db_path(root))) as connection, _transaction(connection):
        signs = _merge_signs(root, connection, route, theirs['signs'], base['signs'], allowed['signs'])
    with contextlib.closing(_open(speed_review.db_path(root))) as connection, _transaction(connection):
        speed = _merge_speed(connection, route, theirs['speed'], base['speed'], allowed['speed'])
    report = {'signs': signs, 'speed': speed}
    for key, path in _line_paths(root, route).items():
        report[key] = _merge_lines(path, theirs[key], base[key], allowed[LINE_STORES[key]])
    return report


# ---------- this server's side ----------

def own_dir(root: Path) -> Path:
    return Path(root) / OWN_RELATIVE


def sent_path(root: Path, route: str) -> Path:
    return own_dir(root) / 'sent' / (route + '.json')


def read_state(root: Path) -> dict:
    try:
        state = json.loads((own_dir(root) / 'state.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    return {'drives': {}, 'finished': [], 'people': [], 'work': {}, 'last': None, **state}


def _save_state(root: Path, state: dict) -> None:
    write_private(own_dir(root) / 'state.json', json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + '\n')


def read_link(root: Path) -> dict | None:
    try:
        return json.loads((own_dir(root) / 'link.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def link(root: Path, remote: str, fingerprint: str) -> None:
    """Remembers the box and its host key, read now and compared with the fingerprint its administrator gave."""
    host = remote.rsplit('@', 1)[-1]
    scanned = subprocess.run(['ssh-keyscan', '-T', str(SSH_TIMEOUT), '-t', 'ed25519', host], capture_output=True, text=True,
                             check=False).stdout
    told = subprocess.run(['ssh-keygen', '-lf', '-'], input=scanned, capture_output=True, text=True, check=False).stdout
    if fingerprint not in told.split():
        raise BoxError(f'The host key of {host} is not {fingerprint}: nothing was saved.')
    write_private(own_dir(root) / 'known_hosts', scanned)
    write_private(own_dir(root) / 'link.json', _text({'remote': remote, 'linked_at': _now()}) + '\n')


def _at(remote: str, path: str) -> str:
    """Where a path of the box is. A remote that is a folder is a box on this disk, for the tests."""
    return f'{remote}:{path}' if '@' in remote else f'{remote}/{path}'


def _rsync(root: Path, *arguments: str) -> None:
    known = own_dir(root) / 'known_hosts'
    ssh = (f'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile={known} '
           f'-o ConnectTimeout={SSH_TIMEOUT}')
    answer = subprocess.run(['rsync', '-e', ssh, f'--timeout={RSYNC_TIMEOUT}', *arguments], capture_output=True, text=True,
                            check=False)
    if answer.returncode != 0:
        raise BoxError('rsync: ' + (answer.stderr.strip()[-300:] or f'code {answer.returncode}'))


def _send_file(root: Path, remote: str, path: Path, target: str) -> None:
    """Whole or not at all: rsync writes a hidden file and renames it, and the tick reads only the names it knows."""
    _rsync(root, '-t', str(path), _at(remote, target))


def _send_drive(root: Path, remote: str, route: str, log: Callable[[str], None]) -> None:
    if drive_video.minutes(root, route) and drive_video.index(root, route) is None:
        log(f'{route}: making its video')
        drive_video.make(root, route)
    package = reviews_of(root, route)
    package['signs'] = _without_positions(package['signs'])
    base = dict(package)
    package['drive'] = drive_rows(root, route)
    with tempfile.TemporaryDirectory(dir=own_dir(root), prefix='stage-') as folder:
        stage = Path(folder)
        listing = stage / 'files'
        listing.write_text(''.join(item + '\n' for item in files_of(root, route)), encoding='utf-8')
        log(f'{route}: sending its files')
        _rsync(root, '-rtL', '--partial', f'--files-from={listing}', str(root) + '/', _at(remote, 'data/'))
        places = stage / 'places'
        places.mkdir()
        stage_without_places(root, route, places)
        _rsync(root, '-rt', str(places) + '/', _at(remote, 'data/'))
        # Last, and only now the tick sees the drive: its files are already there.
        arrival = stage / (route + '.json')
        arrival.write_text(_text(package), encoding='utf-8')
        _send_file(root, remote, arrival, f'{INBOX}/{route}.json')
    write_private(sent_path(root, route), _text(base))


def _given_routes(work: dict) -> set[str]:
    return {route for drives in work.values() for route in drives}


def _hold(state: dict, work: dict) -> None:
    """Who held what on each drive: for as long as the drive is on the box, so that what someone saved just before
    their work ended still counts."""
    for route, item in state['drives'].items():
        writers = item.setdefault('writers', {})
        for name, drives in work.items():
            for kind in drives.get(route, ()):
                names = writers.setdefault(kind, [])
                if name not in names:
                    names.append(name)


def _allowed(item: dict, people: set[str]) -> dict[str, set[str]]:
    """By kind, the names on the labels of those who held it here and are still on the list of people."""
    writers = item.get('writers', {})
    return {kind: {GUEST_PREFIX + name for name in writers.get(kind, ()) if name in people} for kind in KIND_KEYS}


def _take_results(root: Path, state: dict, pulled: Path, log: Callable[[str], None]) -> None:
    people = {person['name'] for person in Assignments(auth_dir(root)).view()['people']}
    for route, item in state['drives'].items():
        path = pulled / (route + '.json')
        base_path = sent_path(root, route)
        if not path.is_file() or not base_path.is_file():
            continue
        try:
            theirs = _read_package(path, route)
        except _Refused as exc:
            item['problem'] = str(exc)
            log(f'{route}: {exc}')
            continue
        base = json.loads(base_path.read_text(encoding='utf-8'))
        # A box on code from before 8 Nov 2026 sends no blocks: those sent to it stay in the base for the next round too,
        # else that round finds them only here and takes every later change of the box for a conflict.
        theirs['speed'] = {**{key: rows for key, rows in base['speed'].items() if key in SPEED_TABLES}, **theirs['speed']}
        if {key: theirs[key] for key in base} == base:
            continue
        try:
            report = merge_drive(root, route, theirs, base, _allowed(item, people))
        except sqlite3.Error as exc:
            item['problem'] = f'refused by the database: {exc}'
            log(f'{route}: {item["problem"]}')
            continue
        write_private(base_path, _text({key: theirs[key] for key in base}))
        whole = [key for key in report if key != 'signs']
        saved = item.setdefault('saved', {'frames': 0, 'stores': 0, 'conflicts': 0, 'refused': 0})
        saved['frames'] += report['signs']['taken']
        saved['stores'] += sum(report[key] == 'taken' for key in whole)
        saved['conflicts'] += report['signs']['conflicts'] + sum(report[key] == 'conflict' for key in whole)
        saved['refused'] += report['signs']['refused'] + sum(report[key] == 'refused' for key in whole)
        item['results_at'] = _now()
        item.pop('problem', None)
        if report['signs']['conflicts'] or report['signs']['refused'] or {'conflict', 'refused'} & {report[key] for key in whole}:
            aside = own_dir(root) / 'aside' / f'{route}-{item["results_at"].replace(":", "")}.json'
            write_private(aside, _text({'report': report, 'theirs': theirs}))
        log(f'{route}: results taken {_text(report)}')


def _take_people(root: Path, state: dict, pulled: Path, log: Callable[[str], None]) -> None:
    """The visits of the people, and the work they called completed, which ends here."""
    try:
        people = json.loads((pulled / PEOPLE_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    if not isinstance(people, dict):
        return
    assignments = Assignments(auth_dir(root))
    visits, done = people.get('people'), people.get('done')
    if isinstance(visits, dict):
        assignments.take_visits(visits, state['people'])
    if not isinstance(done, dict):
        return
    work = assignments.view()['work']
    for name, drives in done.items():
        for route, kinds in (drives.items() if isinstance(drives, dict) else ()):
            for kind, at in (kinds.items() if isinstance(kinds, dict) else ()):
                if (kind in work.get(name, {}).get(route, ()) and route in state['drives']
                        and isinstance(at, str) and TIME_TEXT.fullmatch(at)):
                    assignments.set_work(name, route, kind, False)
                    state['drives'][route].setdefault('completed', []).append({'person': name, 'kind': kind, 'at': at})
                    log(f'{route}: {kind} completed by {name}')


def _send_people(root: Path, remote: str, state: dict) -> None:
    """The people and their work on the drives that are there, when they changed: the site of the box writes that
    file too, so the less it is touched the better."""
    view = Assignments(auth_dir(root)).view()
    people = [person['name'] for person in view['people']]
    work = {name: {route: kinds for route, kinds in drives.items() if route in state['drives']}
            for name, drives in view['work'].items()}
    if people == state['people'] and work == state['work']:
        return
    with tempfile.TemporaryDirectory(dir=own_dir(root), prefix='stage-') as folder:
        path = Path(folder) / PEOPLE_FILE
        path.write_text(_text({'people': people, 'work': work}), encoding='utf-8')
        _send_file(root, remote, path, f'{INBOX}/{PEOPLE_FILE}')
    state['people'], state['work'] = people, work


def _leave(root: Path, remote: str, state: dict, wanted: set[str], pulled: Path, log: Callable[[str], None]) -> None:
    """A drive nobody holds any more. The box was told at the first sync without it, and has shown its last package
    after LEAVING_SECONDS: only then it is asked to delete the drive. It has left when the box stops showing it, so a
    drive given again in the meantime waits and never lands on a removal."""
    now = datetime.now(timezone.utc)
    for route, item in list(state['drives'].items()):
        if item.get('removing'):
            if not (pulled / (route + '.json')).exists():
                sent_path(root, route).unlink(missing_ok=True)
                kept = {key: value for key, value in item.items() if key not in ('removing', 'leaving_at', 'writers')}
                state['finished'] = [{'route': route, 'left_at': _now(), **kept}, *state['finished']][:FINISHED_KEPT]
                del state['drives'][route]
                log(f'{route}: left the review machine')
            continue
        if route in wanted:
            item.pop('leaving_at', None)
            continue
        told = datetime.strptime(item.setdefault('leaving_at', _now()), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        if now - told < timedelta(seconds=LEAVING_SECONDS):
            continue
        with tempfile.TemporaryDirectory(dir=own_dir(root), prefix='stage-') as folder:
            marker = Path(folder) / (route + '.remove')
            marker.touch()
            _send_file(root, remote, marker, f'{INBOX}/{route}.remove')
        item['removing'] = True


def sync(root: Path, log: Callable[[str], None] = print) -> dict:
    """One round with the box: its results here first, then what it should hold now. Raises BoxError when the link is
    missing or the box cannot be reached; a drive that fails stays for the next round and is said in the state."""
    root = Path(root)
    linked = read_link(root)
    if linked is None:
        raise BoxError('The review machine is not linked yet (nnslr review-box link).')
    remote = linked['remote']
    with _locked(own_dir(root) / 'sync.lock'):
        _schemas(root)
        state = read_state(root)
        started = time.monotonic()
        problems = []
        pulled = own_dir(root) / 'pulled'
        pulled.mkdir(parents=True, exist_ok=True)
        try:
            _rsync(root, '-rt', '--delete', f'--max-size={MAX_PACKAGE}', _at(remote, OUTBOX + '/'), str(pulled) + '/')
            _take_results(root, state, pulled, log)
            _take_people(root, state, pulled, log)
            wanted = _given_routes(Assignments(auth_dir(root)).view()['work'])
            for route in sorted(wanted - set(state['drives'])):
                try:
                    _send_drive(root, remote, route, log)
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, sqlite3.Error) as exc:
                    problems.append(f'{route}: {exc}')
                    log(f'{route}: not sent: {exc}')
                    continue
                state['drives'][route] = {'sent_at': _now()}
                log(f'{route}: on the review machine')
            _hold(state, Assignments(auth_dir(root)).view()['work'])
            _send_people(root, remote, state)
            _leave(root, remote, state, wanted, pulled, log)
        except BoxError as exc:
            problems.append(str(exc))
            log(str(exc))
        finally:
            state['last'] = {'at': _now(), 'ok': not problems, 'seconds': round(time.monotonic() - started),
                             'problems': problems[:5]}
            _save_state(root, state)
    if problems:
        raise BoxError('; '.join(problems[:3]))
    return state


def view(root: Path) -> dict:
    """What Data › Reviewers shows of the box."""
    state = read_state(root)
    return {'linked': read_link(root) is not None, 'last': state['last'],
            'drives': [{'route': route, **item} for route, item in sorted(state['drives'].items())],
            'finished': state['finished']}


__all__ = ['BoxError', 'Busy', 'link', 'sync', 'tick', 'view', 'reviews_of', 'merge_drive', 'without_places', 'files_of',
           'stage_without_places', 'drive_rows', 'install_drive', 'remove_drive', 'read_state', 'read_link', 'own_dir']
