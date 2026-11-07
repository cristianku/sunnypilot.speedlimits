"""Speed review (Data › Speed review): the speed a person wants the car to hold, stretch by stretch of a drive.

What it is for. The brain, the one network that will decide the speed, needs a target, and the limit in the logs cannot
be it: it holds the very errors to remove (an 80 that never ended, a limit read on the ramp). So a person goes along a
drive and says what the speed to hold really was. Nothing here runs on the car and nothing here trains: this only
collects the labels.

The label. kph is the cap on a free road, on a grid of 5 km/h: queues, curves and stops stay with the planner, which
slows down further on its own. A label holds on a stretch [start_ns, end_ns) of logMonoTime, the comma's clock, which
stays the same if a new scan moves the start of the drive:

- start_ns: where the car should begin to obey it: at the sign, for a drop as for a raise, the first frame where the
  sign is no longer in the picture. The brain says a drop ahead with its metres, and the car slows down for it before.
- known_ns (optional): from when the past is enough to work the limit out. A missed end of a limit starts at the sign
  nobody saw, and is known only further on, where the scene and the traffic tell it.
- state: 'confirmed' (the proposal stands), 'corrected' (another value) or 'undetermined' (a person looked and cannot
  say: kph is None). A stretch nobody looked at has no row: 'not_examined' is only ever computed, so a proposal left
  alone never counts as approved.

The proposal (proposed) is the OpenStreetMap limit of the drive's frames, as nnslr road-types read it on the server
(map_runs, "Read the map" on Speed review), in runs rounded to 5 km/h, a run under MERGE_SECONDS joined to the one
before (the first one of a drive to the one after). Where the map has no number (no way found, "none", a general limit
like "FR:urban"), the limit of the signs the comma read stands in (sign_runs), and every stretch says which of the two it
comes from. The map's ways and the comma's reads change some way before the sign, so every change of the proposal moves
to the sign the comma read nearby, where the car passes it (at_the_signs). The map is wrong in places, so it is only
ever a proposal: it is worked out when a drive is shown and never
stored, and it is an answer only once a person says yes to it. Next to it the page shows the limit of the signs read
and marks red where the two differ (differences): there the map may be wrong. The signs are the numbers the comma
confirmed reading, not its vision limit: on a drive with the brain that is the brain's own answer, never a proposal.

The writes. Only a person on the site writes them: signed in, or the person at the server itself on a site served only
on 127.0.0.1 without sign-in (reviewer 'local'); never an agent (no MCP tool, no REST write, no CLI). One write (apply)
sets or clears [start_ns, end_ns): the rows under it are clipped or removed, in one transaction with the check of the
revision the page read (a stale one is refused with 409, so two tabs cannot overwrite each other). The drive's revision
is the number of the last write of that drive in speed_history, a table that only grows (triggers refuse the rest): a
drive's speed_stretches can always be rebuilt by playing its history again.

Roadworks. Two taps of a person on the frame shown: here they start, here they end. A start with no end yet runs to the
end of the drive. A mark says nothing of the speed and cuts no stretch: it lies next to the labels, so that the training
can tell the brain where the works were and measure its error there. It is never an input of the car. The taps are
writes of the drive like the others, same revision and same rules (works_start, works_end, works_removed in
speed_history, without a speed), and speed_works is rebuilt by playing them again. A stretch with no mark says nothing
about works: no mark does not mean no works. Only a third tap does: "checked to here" says that a person went through the
drive from its start to the frame shown and marked all the works there (works_checked in speed_history, the last one
holds, speed_works_checked keeps it), so what lies there outside the marks is free of works. Saying where works are not
is what lets the brain learn them at all (nnslr_tools.brain_targets).

``annotations/speed_review.sqlite`` is a file of its own, readable only by its owner like review.sqlite:
drive_review.sqlite holds what a scan can make again, this holds what only a person can.
"""
from __future__ import annotations

import bisect
import contextlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from nnslr_tools import drive_logs
from nnslr_tools.drive_review import DriveReviewStore, drives_view, utc_now
from nnslr_tools.road_types import map_road_types, maxspeed_limit

DB_RELATIVE_PATH = Path('annotations') / 'speed_review.sqlite'
SCHEMA_VERSION = 1
NS = 1_000_000_000
# A run of a limit shorter than this joins the one before, or at the start of a drive the one after: it is a flicker,
# not a stretch of road.
MERGE_SECONDS = 5
# The map's frames are a second apart: a longer gap between two is a hole of the map, where it says nothing.
FRAME_GAP_S = 1.5
MPH_KPH = 1.609344
# A sign read holds this long after it was last seen (sign_runs).
SIGN_HOLD_SECONDS = 60
# A number read once, with no second read of it this close, is a misreading, not a sign (sign_runs).
READ_AGAIN_SECONDS = 2
# The reads of one sign as the car comes to it are never further apart than this (sign_passages).
PASSAGE_GAP_SECONDS = 3
# After its last read a sign leaves the picture: the car is at it this much later (sign_passages).
PASSAGE_SECONDS = 0.5
# A change of the proposal this close to the reads of a sign moves to where the car passes it (at_the_signs).
SNAP_SECONDS = 10
# The classes of a sign that sets a limit, and of one that ends it: a minimum speed or a limit at some hours sets none.
LIMIT_CLASSES = ('maximum_speed:', 'variable_display:')
ENDS_OF_LIMIT = ('cancellation', 'end_of_limit')
# Shorter than this, a sign's limit other than the map's is no disagreement: a sign and the end of its way in the map
# are seldom at the same place.
DIFFER_SECONDS = 8
GRID_KPH = 5
MAX_KPH = 200
MAX_NOTE = 500
STATES = ('confirmed', 'corrected', 'undetermined')
WORKS_TAPS = ('start', 'end', 'remove', 'checked')
NO_LABEL = {'kph': None, 'state': None, 'known_ns': None, 'note': ''}
# The drives to review first: whole drives chosen for what the brain must learn. Together 169 minutes
# with 93 % of the seconds where the limit looks stuck, the roadworks and the low limits on the carriageway.
FIRST_DRIVES = ('00000160--ea56ae0f33', '0000015a--e47d11b0b0', '00000154--fa7e8f49aa', '00000151--7adc20003e',
                '00000153--44d221e479')

SCHEMA_STATEMENTS = (
    'CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)',
    '''CREATE TABLE IF NOT EXISTS speed_stretches (
        route TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, kph INTEGER, state TEXT NOT NULL,
        known_ns INTEGER, note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL, revision INTEGER NOT NULL,
        PRIMARY KEY (route, start_ns))''',
    '''CREATE TABLE IF NOT EXISTS speed_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, route TEXT NOT NULL, revision INTEGER NOT NULL, action TEXT NOT NULL,
        start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, kph INTEGER, state TEXT, known_ns INTEGER,
        note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL, at TEXT NOT NULL, UNIQUE (route, revision))''',
    '''CREATE TABLE IF NOT EXISTS speed_works (
        route TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER, reviewer TEXT NOT NULL, revision INTEGER NOT NULL,
        PRIMARY KEY (route, start_ns))''',
    '''CREATE TABLE IF NOT EXISTS speed_works_checked (
        route TEXT PRIMARY KEY, end_ns INTEGER NOT NULL, reviewer TEXT NOT NULL, revision INTEGER NOT NULL)''',
    '''CREATE TRIGGER IF NOT EXISTS speed_history_no_update BEFORE UPDATE ON speed_history
        BEGIN SELECT RAISE(ABORT, 'speed_history only grows'); END''',
    '''CREATE TRIGGER IF NOT EXISTS speed_history_no_delete BEFORE DELETE ON speed_history
        BEGIN SELECT RAISE(ABORT, 'speed_history only grows'); END''',
)


class SpeedReviewError(ValueError):
    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def db_path(root: Path) -> Path:
    return Path(root) / DB_RELATIVE_PATH


def _has_span(merged: dict | None) -> bool:
    return bool(merged) and merged.get('t0') is not None and merged['span'][1] is not None and merged['span'][1] > merged['t0']


def _runs(points: list, start: int, end: int) -> list[dict]:
    """A limit over [start, end) as stretches in time order, from the points (t, km/h) where it changes: rounded to
    GRID_KPH, kph None where there is none (None or 0), a run under MERGE_SECONDS joined to the one before (the first one
    of the drive to the one after)."""
    points = [(t, value) for t, value in points if start <= t < end]
    if not points or points[0][0] > start:
        points.insert(0, (start, None))
    runs: list[list] = []
    for index, (t, value) in enumerate(points):
        stop = points[index + 1][0] if index + 1 < len(points) else end
        kph = int(round(value / GRID_KPH) * GRID_KPH) if value else None
        if runs and runs[-1][2] == kph:
            runs[-1][1] = stop
        else:
            runs.append([t, stop, kph])
    joined: list[list] = []
    for first, stop, kph in runs:
        if joined and (stop - first < MERGE_SECONDS * NS or joined[-1][2] == kph):
            joined[-1][1] = stop
        else:
            joined.append([first, stop, kph])
    if len(joined) > 1 and joined[0][1] - joined[0][0] < MERGE_SECONDS * NS:
        joined[1][0] = joined[0][0]
        del joined[0]
    return [{'start_ns': first, 'end_ns': stop, 'kph': kph} for first, stop, kph in joined]


def _at(points: list, t: int):
    """The value of a series of changes [[t, value], ...] at a time: its last change at or before it, None before."""
    at = bisect.bisect_right(points, t, key=lambda point: point[0]) - 1
    return points[at][1] if at >= 0 else None


def sign_runs(merged: dict | None) -> list[dict]:
    """The limit of the signs the comma read, from the start of the drive to its end: kph None where none is known. It is
    the last number the comma confirmed reading (a signRead, with the raw reading of its frame: rawSignClass and
    rawSignSpeed), until the next one, an end of a limit, or SIGN_HOLD_SECONDS after the number was last seen: past that a
    sign says nothing of the road (a motorway entered, a limit ended with no sign seen). A number read in one frame alone
(none of the same within READ_AGAIN_SECONDS) is a misreading: a sign is read again and again as the car comes to it. Not the vision limit: with the
    brain that is the brain's answer, and the memory's held its limits for good. Of a frame with two numbers (a gantry)
    the log keeps the surest. A drive from before rawSignClass was logged (31 Oct 2026) has only the memory's vision limit:
    that is taken. Empty for a drive with nothing read."""
    if not _has_span(merged):
        return []
    series = merged['series']
    classes, speeds = series.get('vis_raw_class') or [], series.get('vis_raw_speed') or []
    if not any(isinstance(name, str) for _, name in classes):
        return _runs(series.get('vis_limit') or [], merged['t0'], merged['span'][1])
    read: list[tuple[int, float | None]] = []
    for event in merged.get('events') or ():
        name = _at(classes, event[0]) if event[1] == 'sign' else None
        if name in ENDS_OF_LIMIT:
            read.append((event[0], None))
        elif isinstance(name, str) and name.startswith(LIMIT_CLASSES) and _at(speeds, event[0]):
            read.append((event[0], _at(speeds, event[0])))
    again = READ_AGAIN_SECONDS * NS
    read = [(t, kph) for index, (t, kph) in enumerate(read)
            if any(abs(other - t) <= again and value == kph for other, value in read[max(0, index - 20):index + 21]
                   if other != t)]
    points: list[tuple[int, float | None]] = []
    for index, (t, kph) in enumerate(read):
        points.append((t, kph))
        following = read[index + 1][0] if index + 1 < len(read) else None
        if kph is not None and (following is None or following - t > SIGN_HOLD_SECONDS * NS):
            points.append((t + SIGN_HOLD_SECONDS * NS, None))
    return _runs(points, merged['t0'], merged['span'][1])


def map_runs(merged: dict | None, rows: list[dict]) -> list[dict]:
    """The OpenStreetMap limit of the drive's frames as nnslr road-types read it on the server (one row a second,
    derived/road_types), from the start of the drive to its end: kph None where the way found has no number (no way,
    "none", "FR:urban": no general limit is made up), a limit in mph turned into km/h. A frame holds for its second, and
    a missing one ends the limit there. A second of a segment is placed as the page places it (secondOf in
    drive-review.ts): on the segment's frame line when its encode index has one, else from the segment's start in the
    logs. Empty without rows."""
    if not _has_span(merged) or not rows:
        return []
    lines = {item['segment']: item for item in drive_logs.sync(merged, None)}
    points = []
    for row in rows:
        segment, second = row.get('segment_index'), row.get('output_index')
        if not isinstance(segment, int) or not isinstance(second, (int, float)):
            continue
        line = lines.get(segment)
        if line and line['start'] is not None and line['period']:
            t = merged['t0'] + round((line['start'] + second * drive_logs.FPS * line['period']) * NS)
        elif segment in merged['spans']:
            t = merged['spans'][segment][0] + round(second * NS)
        else:
            continue
        limit = maxspeed_limit(row.get('maxspeed'))
        points.append((t, None if limit is None else limit[0] * (MPH_KPH if limit[1] == 'mph' else 1)))
    points.sort(key=lambda point: point[0])
    held = []
    for index, (t, kph) in enumerate(points):
        held.append((t, kph))
        if index + 1 == len(points) or points[index + 1][0] - t > FRAME_GAP_S * NS:
            held.append((t + NS, None))
    return _runs(held, merged['t0'], merged['span'][1])


def sign_passages(merged: dict | None) -> list[tuple[int, int]]:
    """The signs the comma read on the drive, as (first read, passed): a sign is read again and again as the car comes to
    it, reads at most PASSAGE_GAP_SECONDS apart, and the car is at it PASSAGE_SECONDS after the last read, when the sign has
    left the picture. Every confirmed read counts, whatever it says: the proposal only needs where the signs stand."""
    if not _has_span(merged):
        return []
    reads = sorted(event[0] for event in merged.get('events') or () if event[1] == 'sign')
    gap = PASSAGE_GAP_SECONDS * NS
    out = []
    for index, t in enumerate(reads):
        if index == 0 or t - reads[index - 1] > gap:
            first = t
        if index + 1 == len(reads) or reads[index + 1] - t > gap:
            out.append((first, t + round(PASSAGE_SECONDS * NS)))
    return out


def at_the_signs(proposal: list[dict], passages: list[tuple[int, int]]) -> list[dict]:
    """The proposal with every change at its sign, as a person cuts it: a change within SNAP_SECONDS of the reads of a
    sign (sign_passages) moves to where the car passes the nearest one. A change with no sign near stays where it is, and
    a move that would leave less than a second to a stretch is not made."""
    out = [dict(item) for item in proposal]
    snap = SNAP_SECONDS * NS
    for index in range(1, len(out)):
        cut = out[index]['start_ns']
        near = [(max(first - cut, cut - passed, 0), passed) for first, passed in passages
                if first - snap <= cut <= passed + snap]
        if not near:
            continue
        passed = min(near)[1]
        if out[index - 1]['start_ns'] + NS <= passed <= out[index]['end_ns'] - NS:
            out[index - 1]['end_ns'] = out[index]['start_ns'] = passed
    return out


def proposed(signs: list[dict], mapped: list[dict], span: tuple[int, int]) -> list[dict]:
    """The proposal of the drive: the map's limit (source 'map'), and where the map has no number the signs' (source
    'signs', sign_runs). A piece of the signs' under MERGE_SECONDS joins the stretch before, neighbours of one speed are one
    stretch, and a stretch says the source of most of it."""
    out: list[dict] = []
    top = [{**item, 'source': 'map'} for item in mapped if item['kph'] is not None]
    for item in partition(signs, top, span):
        length = item['end_ns'] - item['start_ns']
        from_map = item.get('source') == 'map'
        if out and (out[-1]['kph'] == item['kph'] or not from_map and length < MERGE_SECONDS * NS):
            out[-1]['end_ns'] = item['end_ns']
            out[-1]['map_ns'] += length if from_map else 0
        else:
            out.append({'start_ns': item['start_ns'], 'end_ns': item['end_ns'], 'kph': item['kph'],
                        'map_ns': length if from_map else 0})
    return [{'start_ns': item['start_ns'], 'end_ns': item['end_ns'], 'kph': item['kph'],
             'source': 'map' if 2 * item['map_ns'] >= item['end_ns'] - item['start_ns'] else 'signs'} for item in out]


def differences(signs: list[dict], mapped: list[dict]) -> list[dict]:
    """Where the signs read gave another limit than the map's, both known, for at least DIFFER_SECONDS: the stretches the
    page marks red, for a person to see at once where the map may be wrong."""
    out: list[dict] = []
    for sign in signs:
        for way in mapped:
            if sign['kph'] is None or way['kph'] is None or sign['kph'] == way['kph']:
                continue
            first, last = max(sign['start_ns'], way['start_ns']), min(sign['end_ns'], way['end_ns'])
            if last - first >= DIFFER_SECONDS * NS:
                out.append({'start_ns': first, 'end_ns': last, 'map_kph': way['kph'], 'sign_kph': sign['kph']})
    return sorted(out, key=lambda item: item['start_ns'])


def partition(proposed: list[dict], saved: list[dict], span: tuple[int, int]) -> list[dict]:
    """The whole drive as stretches in time order: the saved ones, and in the gaps between them the proposals cut to
    the gap, not examined."""
    out: list[dict] = []

    def fill(start: int, stop: int) -> None:
        for item in proposed:
            first, last = max(start, item['start_ns']), min(stop, item['end_ns'])
            if first < last:
                out.append({**item, 'start_ns': first, 'end_ns': last, 'state': 'not_examined', 'known_ns': None,
                            'note': '', 'reviewer': None})

    at = span[0]
    for row in saved:
        if row['start_ns'] > at:
            fill(at, row['start_ns'])
        out.append(row)
        at = max(at, row['end_ns'])
    if at < span[1]:
        fill(at, span[1])
    return out


def _integer(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpeedReviewError(400, 'bad_value', f'{name} must be a whole number.')
    return value


def checked(body: dict, span: tuple[int, int]) -> tuple[int, int, dict | None, int]:
    """(start_ns, end_ns, label, base_revision) of a request, label None to clear the stretch; refused with 400 when
    it does not fit the drive."""
    start, end = _integer(body.get('start_ns'), 'start_ns'), _integer(body.get('end_ns'), 'end_ns')
    base = _integer(body.get('base_revision'), 'base_revision')
    if not span[0] <= start < end <= span[1]:
        raise SpeedReviewError(400, 'bad_stretch', 'The stretch is not inside the drive.')
    state = body.get('state')
    if state is None:
        return start, end, None, base
    if state not in STATES:
        raise SpeedReviewError(400, 'bad_state', 'state must be confirmed, corrected or undetermined.')
    kph, known = body.get('kph'), body.get('known_ns')
    if state == 'undetermined':
        if kph is not None or known is not None:
            raise SpeedReviewError(400, 'bad_value', 'A stretch nobody can tell has no speed.')
    else:
        kph = _integer(kph, 'kph')
        if not GRID_KPH <= kph <= MAX_KPH or kph % GRID_KPH:
            raise SpeedReviewError(400, 'bad_value', f'kph must be a multiple of {GRID_KPH} up to {MAX_KPH}.')
        if known is not None and not start <= _integer(known, 'known_ns') < end:
            raise SpeedReviewError(400, 'bad_value', 'known_ns must be inside the stretch.')
    note = body.get('note', '')
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise SpeedReviewError(400, 'bad_value', f'A note is text of up to {MAX_NOTE} characters.')
    return start, end, {'kph': kph, 'state': state, 'known_ns': known, 'note': note.strip()}, base


def works_tap(body: dict, span: tuple[int, int]) -> tuple[str, int, int]:
    """(tap, at_ns, base_revision) of a request on the roadworks; refused with 400 when it does not fit the drive."""
    tap = body.get('tap')
    if tap not in WORKS_TAPS:
        raise SpeedReviewError(400, 'bad_tap', 'tap must be start, end, remove or checked.')
    at, base = _integer(body.get('at_ns'), 'at_ns'), _integer(body.get('base_revision'), 'base_revision')
    if not span[0] <= at < span[1]:
        raise SpeedReviewError(400, 'bad_stretch', 'This frame is outside the logs of the drive.')
    return tap, at, base


class SpeedReviewStore:
    """The labels' database. Every call opens its own connection (the site's threads share the store); writes are
    serialized by a lock."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = db_path(self.root)
        self._write = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout = 30000')
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))
        with self._write:
            connection = self._connect()
            try:
                connection.execute('PRAGMA journal_mode = WAL')
                connection.execute('BEGIN IMMEDIATE')
                for statement in SCHEMA_STATEMENTS:
                    connection.execute(statement)
                version = connection.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
                if version is not None and version > SCHEMA_VERSION:
                    connection.execute('ROLLBACK')
                    raise RuntimeError(f'{self.path} has schema version {version}; this nnslr knows {SCHEMA_VERSION}')
                connection.execute('INSERT OR IGNORE INTO schema_version (version) VALUES (?)', (SCHEMA_VERSION,))
                connection.execute('COMMIT')
            finally:
                connection.close()

    def _rows(self, sql: str, args=()) -> list[sqlite3.Row]:
        connection = self._connect()
        try:
            return connection.execute(sql, args).fetchall()
        finally:
            connection.close()

    def revision(self, route: str) -> int:
        return self._rows('SELECT COALESCE(MAX(revision), 0) AS revision FROM speed_history WHERE route = ?',
                          (route,))[0]['revision']

    def stretches(self, route: str) -> list[dict]:
        """The saved stretches of a drive in time order."""
        return [dict(row) for row in self._rows('SELECT start_ns, end_ns, kph, state, known_ns, note, reviewer, revision '
                                                'FROM speed_stretches WHERE route = ? ORDER BY start_ns', (route,))]

    def history(self, route: str) -> list[dict]:
        return [dict(row) for row in self._rows('SELECT * FROM speed_history WHERE route = ? ORDER BY revision', (route,))]

    def examined(self) -> dict[str, int]:
        """The seconds with a saved label per drive, in ns."""
        return {row['route']: row['total'] for row in
                self._rows('SELECT route, SUM(end_ns - start_ns) AS total FROM speed_stretches GROUP BY route')}

    @contextlib.contextmanager
    def _writing(self, route: str, base_revision: int) -> Iterator[tuple[sqlite3.Connection, int]]:
        """One write of a drive, all or nothing, if the drive is still at base_revision: gives the connection and the
        new revision."""
        with self._write:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    revision = connection.execute('SELECT COALESCE(MAX(revision), 0) FROM speed_history WHERE route = ?',
                                                  (route,)).fetchone()[0]
                    if revision != base_revision:
                        raise SpeedReviewError(409, 'stale_revision',
                                               'This drive changed in another window: its stretches are read again.')
                    yield connection, revision + 1
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()

    @staticmethod
    def _log(connection: sqlite3.Connection, route: str, revision: int, action: str, start_ns: int, end_ns: int, item: dict,
             reviewer: str) -> None:
        connection.execute('INSERT INTO speed_history (route, revision, action, start_ns, end_ns, kph, state, known_ns, note, '
                           'reviewer, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (route, revision, action, start_ns, end_ns, item['kph'], item['state'], item['known_ns'],
                            item['note'], reviewer, utc_now()))

    def apply(self, route: str, start_ns: int, end_ns: int, label: dict | None, base_revision: int, reviewer: str) -> int:
        """Sets [start_ns, end_ns) to label, or clears it (None: back to not examined), if the drive is still at
        base_revision; the stretches under it keep what stays outside it. Returns the new revision."""
        with self._writing(route, base_revision) as (connection, revision):
            under = connection.execute('SELECT * FROM speed_stretches WHERE route = ? AND start_ns < ? AND end_ns > ?',
                                       (route, end_ns, start_ns)).fetchall()
            insert = ('INSERT INTO speed_stretches (route, start_ns, end_ns, kph, state, known_ns, note, reviewer, '
                      'revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
            connection.executemany('DELETE FROM speed_stretches WHERE route = ? AND start_ns = ?',
                                   [(route, row['start_ns']) for row in under])
            for row in under:
                for first, last in ((row['start_ns'], start_ns), (end_ns, row['end_ns'])):
                    if first < last:
                        known = row['known_ns']
                        known = None if known is None or known < first else min(known, last - 1)
                        connection.execute(insert, (route, first, last, row['kph'], row['state'], known, row['note'],
                                                    row['reviewer'], row['revision']))
            if label is not None:
                connection.execute(insert, (route, start_ns, end_ns, label['kph'], label['state'], label['known_ns'],
                                            label['note'], reviewer, revision))
            self._log(connection, route, revision, 'saved' if label else 'cleared', start_ns, end_ns, label or NO_LABEL, reviewer)
        return revision

    def works(self, route: str) -> list[dict]:
        """The roadworks marked on a drive in time order, end_ns None while a mark has no end yet."""
        return [dict(row) for row in self._rows('SELECT start_ns, end_ns, reviewer, revision FROM speed_works WHERE route = ? '
                                                'ORDER BY start_ns', (route,))]

    def works_checked(self, route: str) -> int | None:
        """Up to when a person went through a drive for roadworks, from its start; None when nobody did."""
        rows = self._rows('SELECT end_ns FROM speed_works_checked WHERE route = ?', (route,))
        return rows[0]['end_ns'] if rows else None

    def tap_works(self, route: str, tap: str, at_ns: int, drive_end: int, base_revision: int, reviewer: str) -> int:
        """A tap on the roadworks at a frame: start a mark, end the one with no end yet, remove the one on this frame,
        say that the drive is checked up to it. Marks do not overlap, one at a time has no end, and a drive is not
        checked past a mark with no end. Returns the new revision."""
        with self._writing(route, base_revision) as (connection, revision):
            marks = connection.execute('SELECT * FROM speed_works WHERE route = ? ORDER BY start_ns', (route,)).fetchall()
            here = next((row for row in marks if row['start_ns'] <= at_ns and (row['end_ns'] is None or at_ns < row['end_ns'])), None)
            unended = next((row for row in marks if row['end_ns'] is None), None)
            if tap == 'start':
                if unended is not None or here is not None:
                    raise SpeedReviewError(400, 'bad_works', 'End the roadworks already open first.' if unended is not None
                                           else 'This frame is already inside roadworks.')
                connection.execute('INSERT INTO speed_works (route, start_ns, end_ns, reviewer, revision) VALUES (?, ?, NULL, ?, ?)',
                                   (route, at_ns, reviewer, revision))
                self._log(connection, route, revision, 'works_start', at_ns, drive_end, NO_LABEL, reviewer)
            elif tap == 'end':
                if unended is None:
                    raise SpeedReviewError(400, 'bad_works', 'No roadworks are waiting for their end.')
                later = [row['start_ns'] for row in marks if row['start_ns'] > unended['start_ns']]
                if at_ns <= unended['start_ns'] or (later and at_ns > later[0]):
                    raise SpeedReviewError(400, 'bad_works', 'The end must come after the start, before the next roadworks.')
                connection.execute('UPDATE speed_works SET end_ns = ?, reviewer = ?, revision = ? WHERE route = ? AND start_ns = ?',
                                   (at_ns, reviewer, revision, route, unended['start_ns']))
                self._log(connection, route, revision, 'works_end', unended['start_ns'], at_ns, NO_LABEL, reviewer)
            elif tap == 'checked':
                if unended is not None and unended['start_ns'] <= at_ns:
                    raise SpeedReviewError(400, 'bad_works', 'End the roadworks already open first.')
                connection.execute('INSERT OR REPLACE INTO speed_works_checked (route, end_ns, reviewer, revision) VALUES (?, ?, ?, ?)',
                                   (route, at_ns, reviewer, revision))
                self._log(connection, route, revision, 'works_checked', at_ns, at_ns, NO_LABEL, reviewer)
            else:
                if here is None:
                    raise SpeedReviewError(400, 'bad_works', 'No roadworks on this frame.')
                connection.execute('DELETE FROM speed_works WHERE route = ? AND start_ns = ?', (route, here['start_ns']))
                self._log(connection, route, revision, 'works_removed', here['start_ns'], here['end_ns'] or drive_end, NO_LABEL,
                          reviewer)
        return revision


_stores: dict[str, SpeedReviewStore] = {}
_stores_lock = threading.Lock()


def store_for(root: Path) -> SpeedReviewStore:
    """The store of a data root, one per process."""
    key = str(Path(root).resolve())
    with _stores_lock:
        if key not in _stores:
            _stores[key] = SpeedReviewStore(Path(root))
        return _stores[key]


def _span_of(drives: DriveReviewStore, route: str) -> tuple[int, int] | None:
    """The drive on the comma's clock as the last scan found it; None before a scan."""
    summary = drives.summary(route)
    if summary is None or not summary.get('span_ns') or summary['span_ns'][1] is None:
        return None
    return summary['t0_ns'], summary['span_ns'][1]


def drive_view(store: SpeedReviewStore, drives: DriveReviewStore, route: str) -> dict:
    """A drive for the page: its revision, its stretches from start to end, its roadworks, how much of it is examined,
    and next to them the map's limit (map, map_read: whether the map was read for the drive at all), the limit of the
    signs the comma read (signs) and where the two differ (differ). A drive whose logs are not scanned has no stretches."""
    span = _span_of(drives, route)
    if span is None:
        return {'ok': True, 'route': route, 'revision': store.revision(route), 'stretches': [], 'works': [],
                'works_checked_ns': None, 'examined_s': 0, 'total_s': 0, 'map_read': False, 'map': [], 'signs': [],
                'differ': []}
    saved = store.stretches(route)
    merged = drives.merged(route)
    rows = map_road_types(store.root, route)
    signs, mapped = sign_runs(merged), map_runs(merged, rows)
    return {'ok': True, 'route': route, 'revision': store.revision(route),
            'stretches': partition(at_the_signs(proposed(signs, mapped, span), sign_passages(merged)), saved, span),
            'map_read': bool(rows), 'map': mapped, 'signs': signs, 'differ': differences(signs, mapped),
            'works': store.works(route),
            'works_checked_ns': store.works_checked(route),
            'examined_s': round(sum(row['end_ns'] - row['start_ns'] for row in saved) / NS, 1),
            'total_s': round((span[1] - span[0]) / NS, 1)}


def drives_list(store: SpeedReviewStore, drives: DriveReviewStore) -> list[dict]:
    """The drives for the page, which puts them in order: each with its share examined and whether it is one to review
    first; no proposals here (they need the drive's logs)."""
    examined = store.examined()
    rows = []
    for drive in drives_view(drives):
        span = _span_of(drives, drive['route'])
        rows.append({**drive, 'first': drive['route'] in FIRST_DRIVES, 'examined_s': round(examined.get(drive['route'], 0) / NS, 1),
                     'total_s': round((span[1] - span[0]) / NS, 1) if span else 0})
    return rows


def save(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's write: checks it against the drive, applies it, and answers with the drive as it is now."""
    span = _span_of(drives, route)
    if span is None:
        raise SpeedReviewError(404, 'not_scanned', 'This drive has no scanned logs.')
    start, end, label, base = checked(body, span)
    store.apply(route, start, end, label, base, reviewer)
    return drive_view(store, drives, route)


def save_works(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's tap on the roadworks: checks it against the drive, applies it, and answers with the drive as it is now."""
    span = _span_of(drives, route)
    if span is None:
        raise SpeedReviewError(404, 'not_scanned', 'This drive has no scanned logs.')
    tap, at, base = works_tap(body, span)
    store.tap_works(route, tap, at, span[1], base, reviewer)
    return drive_view(store, drives, route)
