"""Speed review (Data › Speed review): the speed the brain learns to hold on a drive, the driver's own unless a person
blocks it.

What it is for. The brain, the one network that decides the speed, needs an answer at every frame. The answer is the
speed the driver held, as the dashboard showed it (drive_logs.dashboard_speed), in whole km/h, at the brain's steps
(driver_speed): there is nothing to confirm, every drive with logs is labelled by being driven. A person only corrects
it, with a block: from one moment to another the car should have held another speed (the driver too fast or too slow for
the road there), or those moments are left out of the training (a speed that is no answer, and that nobody can give).
Nothing here runs on the car and nothing here trains: this only collects the corrections.

A block holds on [start_ns, end_ns) of logMonoTime, the comma's clock, which stays the same if a new scan moves the start
of the drive, at least MIN_BLOCK_SECONDS long. kph is a whole number from 1 to MAX_KPH, or None to leave the moments out.
A new block over others clips them: what they say outside it stays, unless it is shorter than MIN_BLOCK_SECONDS.

The writes. Only a person on the site writes them: signed in, or the person at the server itself on a site served only
on 127.0.0.1 without sign-in (reviewer 'local'); never an agent (no MCP tool, no REST write, no CLI), with two
exceptions, neither of which writes over a person's mark, removal or "checked to here", and both definitive, nobody
confirms them (the tunnels asked on 8 Nov 2026, the labeler on 9 Nov): the tunnels of the map where it is sure
(nnslr_tools.tunnels, reviewer 'osm'), and the road labeler's Prepare Speed review (nnslr_tools.road_labeler, reviewer
net/<run>: marks of the kinds of road and of the roadworks, only of the heads that passed its gate). Both are answers of
the brain like a person's marks; a person corrects them only where they are wrong, with the same taps. A newer run of
the labeler takes out the marks of the older one that no person kept (a mark that starts before its kind's "checked to
here") and writes its own, in one write of many revisions (batch: each tap a row of speed_history, all or nothing), with
a row types_labeled for its layer of road types (road_types.write_labeler_types). One write sets a block or removes one,
in one transaction with the check of the revision the page read (a stale one is refused with 409, so two tabs cannot
overwrite each other). The drive's revision is the number of the last write of that drive in speed_history, a table that
only grows (triggers refuse the rest): a drive's speed_blocks can always be rebuilt by playing its history again
(block_saved, block_moved, block_removed). A block whose ends a person drags on the page is moved in one write:
block_moved, with the start it had in its column known_ns.

Roadworks. Two taps of a person on the frame shown: here they start, here they end. A start with no end yet runs to the
end of the drive. A mark says nothing of the speed: it lies next to the answers, so that the training can tell the brain
where the works were and measure its error there. It is never an input of the car. The taps are writes of the drive like
the others, same revision and same rules (works_start, works_end, works_removed in speed_history, without a speed), and
speed_works is rebuilt by playing them again. No mark does not mean no works. Only a third tap does: "checked to here"
says that a person went through the drive from its start to the frame shown and marked all the works there
(works_checked in speed_history, the last one holds, speed_works_checked keeps it), so what lies there outside the marks
is free of works. Saying where works are not is what lets the brain learn them at all (nnslr_tools.brain_targets).

The kinds of road. The same taps mark what the road is, kind by kind: a narrow road, a country road, a gravel road,
potholes (FIRST_ROAD_KINDS), and any kind a person adds from the page (speed_road_kinds, which only grows: a kind keeps
its marks). Like the roadworks they say nothing of the speed and are never an input of the car: the brain learns to tell
them (nnslr_tools.brain_net's road_head), and that changes the memory the speed shares. One stretch can be of several
kinds at once, so the marks of one kind never overlap and those of two kinds may. "Checked to here" is a kind's own
(speed_road_checked): a person went through the drive for that kind, and a kind added later starts with nothing checked.
The taps go in speed_history as road_start, road_end, road_removed and road_checked, with the kind in its column state,
and speed_road is rebuilt by playing them again, as speed_works.

The road types. Town, rural, expressway, motorway… (road_types.ROAD_TYPES) come from the map, one a second (Read the map
in Data › Prepare, nnslr road-types), with the road labeler's layer over it where it is sure (Prepare Speed review), and
a person corrects them here where the type shown is wrong: a type, or "can't tell", on the whole stretch of the frame
shown or from one frame to another. The page shows the type as the brain takes it (road_types.load_road_types) and
says, stretch by stretch, whose it is: the map's, the labeler's (net) or a person's. The correction is kept with
road_types.correct (annotations/road_types/<route>.jsonl), over both layers: never an input of the car, the brain
learns the type aside (brain_net's type_head). It is a write of the drive like the others, same revision and same
rules: type_corrected in speed_history, the type in its column state ('cant_tell' for none) and its frames in note.

The old stretches. Until 8 Nov 2026 a person said the speed stretch by stretch, on a grid of 5 km/h, from a proposal of
the OpenStreetMap limit: speed_stretches and its rows of speed_history (saved, cleared) stay in the file, and nothing
reads them any more.

The old switch. On 9 Nov 2026 a person marked a drive "OK for brain training" (speed_training_ok,
speed_training_history), and the same evening the switch went again: every dashcam drive enters the brain training,
whole (brain_targets.training_routes), since the signs the brain is given are the networks' anyway. Its tables stay in
the file, and nothing reads them any more.

``annotations/speed_review.sqlite`` is a file of its own, readable only by its owner like review.sqlite:
drive_review.sqlite holds what a scan can make again, this holds what only a person can.
"""
from __future__ import annotations

import bisect
import contextlib
import json
import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from nnslr_tools import drive_logs, road_types, video_frames
from nnslr_tools.drive_review import DriveReviewStore, drives_view, utc_now
from nnslr_tools.road_type_signs import STEPS_PER_SECOND

DB_RELATIVE_PATH = Path('annotations') / 'speed_review.sqlite'
SCHEMA_VERSION = 2
NS = 1_000_000_000
MAX_KPH = 200
MAX_NOTE = 500
# A block shorter than this is no stretch of road: a slip of the finger.
MIN_BLOCK_SECONDS = 1
WORKS_TAPS = ('start', 'end', 'remove', 'checked')
# The kinds of road there are before a person adds one: (kind, label), in the order the page and the brain take them.
FIRST_ROAD_KINDS = (('narrow', 'Narrow road'), ('country', 'Country road'), ('gravel', 'Gravel road'), ('potholes', 'Potholes'))
MAX_KIND_LABEL = 40
# The names a kind of road may not take: the page's own roadworks ("works", its taps go to speed_works), a name every
# object of the page has already ("constructor"), and the fields a guest's copy drops as places
# (review_access.PLACE_FIELDS: its "checked to here" would be lost on the box).
RESERVED_KINDS = frozenset({'works', 'constructor', 'gps', 'latitude', 'longitude', 'lat', 'lon', 'lng', 'road_name',
                            'title', 'way_id'})
# The labels a kind may not take: the roadworks' own, as the page names them.
RESERVED_LABELS = frozenset({'roadworks', 'works'})
NO_LABEL = {'kph': None, 'state': None, 'known_ns': None, 'note': ''}
# A frame of the map this long after the one before starts a new stretch of road type: the frames between are missing.
ROAD_TYPE_GAP_NS = 2 * NS
# openpilot driving, any of them True once: selfdriveState, MADS, carControl.
ENGAGED_SERIES = ('sd_enabled', 'sd_active', 'mads_active', 'lat_active', 'long_active')

SCHEMA_STATEMENTS = (
    'CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)',
    # Read by nothing since 8 Nov 2026: kept for its history.
    '''CREATE TABLE IF NOT EXISTS speed_stretches (
        route TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, kph INTEGER, state TEXT NOT NULL,
        known_ns INTEGER, note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL, revision INTEGER NOT NULL,
        PRIMARY KEY (route, start_ns))''',
    '''CREATE TABLE IF NOT EXISTS speed_blocks (
        route TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, kph INTEGER,
        note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL, revision INTEGER NOT NULL,
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
    '''CREATE TABLE IF NOT EXISTS speed_road_kinds (
        kind TEXT PRIMARY KEY, label TEXT NOT NULL, position INTEGER NOT NULL, reviewer TEXT NOT NULL, at TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS speed_road (
        route TEXT NOT NULL, kind TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER, reviewer TEXT NOT NULL,
        revision INTEGER NOT NULL, PRIMARY KEY (route, kind, start_ns))''',
    '''CREATE TABLE IF NOT EXISTS speed_road_checked (
        route TEXT NOT NULL, kind TEXT NOT NULL, end_ns INTEGER NOT NULL, reviewer TEXT NOT NULL, revision INTEGER NOT NULL,
        PRIMARY KEY (route, kind))''',
    '''CREATE TRIGGER IF NOT EXISTS speed_road_kinds_no_delete BEFORE DELETE ON speed_road_kinds
        BEGIN SELECT RAISE(ABORT, 'a kind of road keeps its marks'); END''',
    '''CREATE TRIGGER IF NOT EXISTS speed_history_no_update BEFORE UPDATE ON speed_history
        BEGIN SELECT RAISE(ABORT, 'speed_history only grows'); END''',
    '''CREATE TRIGGER IF NOT EXISTS speed_history_no_delete BEFORE DELETE ON speed_history
        BEGIN SELECT RAISE(ABORT, 'speed_history only grows'); END''',
    # "OK for brain training", the drive's switch of 9 Nov 2026, gone the same evening: nothing reads these any more.
    '''CREATE TABLE IF NOT EXISTS speed_training_ok (
        route TEXT PRIMARY KEY, ok INTEGER NOT NULL, reviewer TEXT NOT NULL, at TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS speed_training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT, route TEXT NOT NULL, ok INTEGER NOT NULL, reviewer TEXT NOT NULL,
        at TEXT NOT NULL)''',
    '''CREATE TRIGGER IF NOT EXISTS speed_training_history_no_update BEFORE UPDATE ON speed_training_history
        BEGIN SELECT RAISE(ABORT, 'speed_training_history only grows'); END''',
    '''CREATE TRIGGER IF NOT EXISTS speed_training_history_no_delete BEFORE DELETE ON speed_training_history
        BEGIN SELECT RAISE(ABORT, 'speed_training_history only grows'); END''',
)


class SpeedReviewError(ValueError):
    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def db_path(root: Path) -> Path:
    return Path(root) / DB_RELATIVE_PATH


def whole_kph(value: float | None) -> int | None:
    """A speed of the dashboard as an answer: whole km/h, half up, at most MAX_KPH; None stays None."""
    return None if value is None else min(MAX_KPH, max(0, math.floor(value + 0.5)))


def driver_speed(merged: dict | None, ns) -> list[int | None]:
    """The driver's speed at each of the times ns (logMonoTime, in order), as the dashboard showed it, in whole km/h
    (whole_kph): the brain's answer where no block says otherwise. None where it is unknown, before the logs' first
    speed among them."""
    points = drive_logs.dashboard_speed(merged) if merged else []
    times = [point[0] for point in points]
    out = []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        out.append(whole_kph(points[at][1]) if at >= 0 else None)
    return out


def speed_curve(merged: dict | None, span: tuple[int, int]) -> list[list]:
    """The driver's speed over the drive at the brain's steps (STEPS_PER_SECOND), as [t_ns, kph] where it changes:
    what the page draws, the same numbers the training takes (driver_speed)."""
    step = NS // STEPS_PER_SECOND
    times = range(span[0], span[1], step)
    out: list[list] = []
    for t, kph in zip(times, driver_speed(merged, times)):
        if not out or out[-1][1] != kph:
            out.append([t, kph])
    return out


def machine_held(merged: dict | None, ns) -> list[bool]:
    """Whether a machine held the speed at each of the times ns, not his foot: openpilot's longitudinal was active
    (carControl.longActive), or the car's cruise was on (carState.cruiseState.enabled) with the gas not pressed, since his
    foot over the cruise is his speed. The brain learns no speed of his there; a block still says one. Unknown is not
    held."""
    series = (merged or {}).get('series') or {}

    def values(name: str) -> list:
        points = series.get(name) or []
        times = [point[0] for point in points]
        found = []
        for t in ns:
            at = bisect.bisect_right(times, int(t)) - 1
            found.append(points[at][1] if at >= 0 else None)
        return found
    out = []
    for active, on, gas in zip(values('long_active'), values('cruise_enabled'), values('gas')):
        held = active is True or (on is True and gas is not True)
        out.append(held)
    return out


def dashcam_drive(summary: dict | None) -> bool:
    """Whether the brain learns from a drive, by its scan's summary (DriveReviewStore.summary): openpilot was passive in
    every carParams of it, so he drove alone, and selfdriveState never said engaged. Passive is sunnylink › Toggles ›
    "Enable sunnypilot" off (OpenpilotEnabledToggle) or a car in dashcamOnly (selfdrive/car/card.py), read once a route.
    An extract older than carParams.passive (drive_extract, before 8 Nov 2026) says dashcamOnly instead: the toggle was on
    in every log until then, so it is the same. A drive without carParams is not one. The sign classifier learns from
    every drive; the brain and Speed review's list only from these."""
    summary = summary or {}
    car = (summary.get('meta') or {}).get('car') or {}
    passive = car.get('passive', car.get('dashcam_only'))
    return passive is True and not (summary.get('coverage') or {}).get('engaged')


def engaged_series(merged: dict | None) -> list[str]:
    """The series of openpilot driving (ENGAGED_SERIES) that say True anywhere in a drive's joined logs: none on a
    dashcam drive (dashcam_drive), since passive keeps selfdriveState, MADS and controlsd off."""
    series = (merged or {}).get('series') or {}
    return [name for name in ENGAGED_SERIES if any(point[1] is True for point in series.get(name) or ())]


def held_spans(merged: dict | None, span: tuple[int, int]) -> list[list[int]]:
    """Where a machine held the speed (machine_held) at the brain's steps, as [from_ns, to_ns]: what the page greys."""
    step = NS // STEPS_PER_SECOND
    times = range(span[0], span[1], step)
    out: list[list[int]] = []
    for t, held in zip(times, machine_held(merged, times)):
        if held and out and out[-1][1] == t:
            out[-1][1] = t + step
        elif held:
            out.append([t, t + step])
    return out


def _integer(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpeedReviewError(400, 'bad_value', f'{name} must be a whole number.')
    return value


def block_of(body: dict, span: tuple[int, int]) -> tuple[int, int, int | None, str, int]:
    """(start_ns, end_ns, kph, note, base_revision) of a request to set a block; refused with 400 when it does not fit the
    drive. kph None leaves the moments out of the training."""
    start, end = _integer(body.get('start_ns'), 'start_ns'), _integer(body.get('end_ns'), 'end_ns')
    base = _integer(body.get('base_revision'), 'base_revision')
    if not span[0] <= start < end <= span[1]:
        raise SpeedReviewError(400, 'bad_block', 'The block is not inside the drive.')
    if end - start < MIN_BLOCK_SECONDS * NS:
        raise SpeedReviewError(400, 'bad_block', f'A block lasts at least {MIN_BLOCK_SECONDS} s.')
    kph = body.get('kph')
    if kph is not None and not 1 <= _integer(kph, 'kph') <= MAX_KPH:
        raise SpeedReviewError(400, 'bad_value', f'kph is a whole number from 1 to {MAX_KPH}.')
    note = body.get('note', '')
    if not isinstance(note, str) or len(note) > MAX_NOTE:
        raise SpeedReviewError(400, 'bad_value', f'A note is text of up to {MAX_NOTE} characters.')
    return start, end, kph, note.strip(), base


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
    """The corrections' database. Every call opens its own connection (the site's threads share the store); writes are
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
                connection.executemany('INSERT OR IGNORE INTO speed_road_kinds (kind, label, position, reviewer, at) '
                                       'VALUES (?, ?, ?, ?, ?)', [(kind, label, position, 'nnslr', utc_now())
                                                                  for position, (kind, label) in enumerate(FIRST_ROAD_KINDS)])
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

    def blocks(self, route: str) -> list[dict]:
        """The blocks of a drive in time order."""
        return [dict(row) for row in self._rows('SELECT start_ns, end_ns, kph, note, reviewer, revision FROM speed_blocks '
                                                'WHERE route = ? ORDER BY start_ns', (route,))]

    def counts(self) -> dict[str, dict[str, int]]:
        """Per drive, how many blocks, roadworks marks and marks of a kind of road it has."""
        out: dict[str, dict[str, int]] = {}
        for table, key in (('speed_blocks', 'blocks'), ('speed_works', 'works'), ('speed_road', 'road')):
            for row in self._rows(f'SELECT route, COUNT(*) AS n FROM {table} GROUP BY route'):
                out.setdefault(row['route'], {'blocks': 0, 'works': 0, 'road': 0})[key] = row['n']
        return out

    def history(self, route: str) -> list[dict]:
        return [dict(row) for row in self._rows('SELECT * FROM speed_history WHERE route = ? ORDER BY revision', (route,))]

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
                                               'This drive changed in another window: it is read again.')
                    yield connection, revision + 1
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()

    @contextlib.contextmanager
    def batch(self, route: str, base_revision: int) -> Iterator['Batch']:
        """Several writes of a drive as one, all or nothing, if the drive is still at base_revision (409 otherwise):
        each a row of speed_history at the revisions after it, in order. Nothing is kept when the block raises. The
        road labeler's Prepare Speed review: the older run's marks out and the new run's in, at once."""
        with self._writing(route, base_revision) as (connection, revision):
            yield Batch(self, connection, route, revision)

    @staticmethod
    def _log(connection: sqlite3.Connection, route: str, revision: int, action: str, start_ns: int, end_ns: int, item: dict,
             reviewer: str) -> None:
        connection.execute('INSERT INTO speed_history (route, revision, action, start_ns, end_ns, kph, state, known_ns, note, '
                           'reviewer, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (route, revision, action, start_ns, end_ns, item['kph'], item['state'], item['known_ns'],
                            item['note'], reviewer, utc_now()))

    def set_block(self, route: str, start_ns: int, end_ns: int, kph: int | None, note: str, base_revision: int,
                  reviewer: str, moved_from: int | None = None) -> int:
        """Sets the block [start_ns, end_ns) to kph (None: left out), if the drive is still at base_revision; the blocks
        under it keep what stays outside it, a piece shorter than MIN_BLOCK_SECONDS goes (the page could only remove it).
        With moved_from, the block that started there goes first: its ends dragged to these, in the same write. Returns
        the new revision."""
        with self._writing(route, base_revision) as (connection, revision):
            if moved_from is not None:
                if connection.execute('DELETE FROM speed_blocks WHERE route = ? AND start_ns = ?', (route, moved_from)).rowcount != 1:
                    raise SpeedReviewError(400, 'bad_block', 'No block starts here.')
            under = connection.execute('SELECT * FROM speed_blocks WHERE route = ? AND start_ns < ? AND end_ns > ?',
                                       (route, end_ns, start_ns)).fetchall()
            insert = 'INSERT INTO speed_blocks (route, start_ns, end_ns, kph, note, reviewer, revision) VALUES (?, ?, ?, ?, ?, ?, ?)'
            connection.executemany('DELETE FROM speed_blocks WHERE route = ? AND start_ns = ?',
                                   [(route, row['start_ns']) for row in under])
            for row in under:
                for first, last in ((row['start_ns'], start_ns), (end_ns, row['end_ns'])):
                    if last - first >= MIN_BLOCK_SECONDS * NS:
                        connection.execute(insert, (route, first, last, row['kph'], row['note'], row['reviewer'], row['revision']))
            connection.execute(insert, (route, start_ns, end_ns, kph, note, reviewer, revision))
            self._log(connection, route, revision, 'block_saved' if moved_from is None else 'block_moved', start_ns, end_ns,
                      {**NO_LABEL, 'kph': kph, 'note': note, 'known_ns': moved_from}, reviewer)
        return revision

    def remove_block(self, route: str, start_ns: int, base_revision: int, reviewer: str) -> int:
        """Removes the block that starts at start_ns, if the drive is still at base_revision: the driver's speed holds
        there again. Returns the new revision."""
        with self._writing(route, base_revision) as (connection, revision):
            row = connection.execute('SELECT * FROM speed_blocks WHERE route = ? AND start_ns = ?', (route, start_ns)).fetchone()
            if row is None:
                raise SpeedReviewError(400, 'bad_block', 'No block starts here.')
            connection.execute('DELETE FROM speed_blocks WHERE route = ? AND start_ns = ?', (route, start_ns))
            self._log(connection, route, revision, 'block_removed', row['start_ns'], row['end_ns'], NO_LABEL, reviewer)
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
        return self._tap(route, None, tap, at_ns, drive_end, base_revision, reviewer)

    def road_kinds(self) -> list[dict]:
        """The kinds of road, [{kind, label}] in their order: the page's and the brain's (nnslr_tools.brain_targets)."""
        return [dict(row) for row in self._rows('SELECT kind, label FROM speed_road_kinds ORDER BY position, kind')]

    def road(self, route: str) -> list[dict]:
        """The marks of the kinds of road on a drive, by kind then time, end_ns None while a mark has no end yet."""
        return [dict(row) for row in self._rows('SELECT kind, start_ns, end_ns, reviewer, revision FROM speed_road '
                                                'WHERE route = ? ORDER BY kind, start_ns', (route,))]

    def road_checked(self, route: str) -> dict[str, int]:
        """{kind: up to when a person went through the drive for that kind, from its start}; a kind nobody did is not there."""
        return {row['kind']: row['end_ns'] for row in self._rows('SELECT kind, end_ns FROM speed_road_checked WHERE route = ?',
                                                                 (route,))}

    def add_road_kind(self, label: str, reviewer: str) -> str:
        """A new kind of road with this label, after the others: returns its kind, a name made of the label."""
        label = ' '.join(label.split())
        if not label or len(label) > MAX_KIND_LABEL:
            raise SpeedReviewError(400, 'bad_value', f'A kind of road is a name of 1 to {MAX_KIND_LABEL} characters.')
        base = ''.join(char if char.isascii() and char.isalnum() else '_' for char in label.lower()).strip('_') or 'kind'
        with self._write:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                rows = connection.execute('SELECT kind, label, position FROM speed_road_kinds').fetchall()
                if label.lower() in RESERVED_LABELS or any(row['label'].lower() == label.lower() for row in rows):
                    connection.execute('ROLLBACK')
                    raise SpeedReviewError(400, 'kind_exists', f'There is a kind "{label}" already.')
                taken = {row['kind'] for row in rows} | RESERVED_KINDS
                kind, number = base, 2
                while kind in taken:
                    kind, number = f'{base}_{number}', number + 1
                connection.execute('INSERT INTO speed_road_kinds (kind, label, position, reviewer, at) VALUES (?, ?, ?, ?, ?)',
                                   (kind, label, max((row['position'] for row in rows), default=-1) + 1, reviewer, utc_now()))
                connection.execute('COMMIT')
            finally:
                connection.close()
        return kind

    def tap_road(self, route: str, kind: str, tap: str, at_ns: int, drive_end: int, base_revision: int, reviewer: str) -> int:
        """A tap on a kind of road at a frame, as tap_works on the roadworks, with that kind's marks and its own "checked to
        here". Returns the new revision."""
        label = next((row['label'] for row in self.road_kinds() if row['kind'] == kind), None)
        if label is None:
            raise SpeedReviewError(400, 'bad_kind', 'No such kind of road.')
        return self._tap(route, kind, tap, at_ns, drive_end, base_revision, reviewer, label)

    def _tap(self, route: str, kind: str | None, tap: str, at_ns: int, drive_end: int, base_revision: int,
             reviewer: str, label: str = '') -> int:
        """tap_works (kind None) and tap_road: the same rules, on speed_works or on the marks of one kind in speed_road
        (label: the kind's, for the messages)."""
        with self._writing(route, base_revision) as (connection, revision):
            self._tap_on(connection, route, revision, kind, tap, at_ns, drive_end, reviewer, label)
        return revision

    def _tap_on(self, connection: sqlite3.Connection, route: str, revision: int, kind: str | None, tap: str, at_ns: int,
                drive_end: int, reviewer: str, label: str = '', exact: bool = False) -> None:
        """One tap of _tap at the revision given, inside a write already open: it checks before it changes anything, so
        a tap refused (SpeedReviewError) leaves the write as it was. exact (remove only): the mark that starts at at_ns,
        never another one whose span covers it (an open mark started earlier covers every later time)."""
        if kind is None:
            marks_table, checked_table, action, what = 'speed_works', 'speed_works_checked', 'works', 'roadworks'
            keys, item, bad = {'route': route}, NO_LABEL, 'bad_works'
        else:
            marks_table, checked_table, action, what = 'speed_road', 'speed_road_checked', 'road', f'"{label}" mark'
            keys, item, bad = {'route': route, 'kind': kind}, {**NO_LABEL, 'state': kind}, 'bad_road'
        of = ' AND '.join(f'{name} = ?' for name in keys)
        names = ', '.join(keys)
        holes = ', '.join('?' for _ in keys)
        values = tuple(keys.values())
        marks = connection.execute(f'SELECT * FROM {marks_table} WHERE {of} ORDER BY start_ns', values).fetchall()
        here = next((row for row in marks if row['start_ns'] <= at_ns and (row['end_ns'] is None or at_ns < row['end_ns'])), None)
        if exact:
            here = next((row for row in marks if row['start_ns'] == at_ns), None)
        unended = next((row for row in marks if row['end_ns'] is None), None)
        if tap == 'start':
            if unended is not None or here is not None:
                raise SpeedReviewError(400, bad, f'End the {what} already open first.' if unended is not None
                                       else f'This frame is already inside a {what}.' if kind else
                                       'This frame is already inside roadworks.')
            connection.execute(f'INSERT INTO {marks_table} ({names}, start_ns, end_ns, reviewer, revision) '
                               f'VALUES ({holes}, ?, NULL, ?, ?)', (*values, at_ns, reviewer, revision))
            self._log(connection, route, revision, f'{action}_start', at_ns, drive_end, item, reviewer)
        elif tap == 'end':
            if unended is None:
                raise SpeedReviewError(400, bad, f'No {what} is waiting for its end.' if kind else
                                       'No roadworks are waiting for their end.')
            later = [row['start_ns'] for row in marks if row['start_ns'] > unended['start_ns']]
            if at_ns <= unended['start_ns'] or (later and at_ns > later[0]):
                raise SpeedReviewError(400, bad, f'The end must come after the start, before the next {what}.')
            connection.execute(f'UPDATE {marks_table} SET end_ns = ?, reviewer = ?, revision = ? WHERE {of} AND start_ns = ?',
                               (at_ns, reviewer, revision, *values, unended['start_ns']))
            self._log(connection, route, revision, f'{action}_end', unended['start_ns'], at_ns, item, reviewer)
        elif tap == 'checked':
            if unended is not None and unended['start_ns'] <= at_ns:
                raise SpeedReviewError(400, bad, f'End the {what} already open first.')
            connection.execute(f'INSERT OR REPLACE INTO {checked_table} ({names}, end_ns, reviewer, revision) '
                               f'VALUES ({holes}, ?, ?, ?)', (*values, at_ns, reviewer, revision))
            self._log(connection, route, revision, f'{action}_checked', at_ns, at_ns, item, reviewer)
        else:
            if here is None:
                raise SpeedReviewError(400, bad, f'No {what} on this frame.')
            connection.execute(f'DELETE FROM {marks_table} WHERE {of} AND start_ns = ?', (*values, here['start_ns']))
            self._log(connection, route, revision, f'{action}_removed', here['start_ns'], here['end_ns'] or drive_end, item,
                      reviewer)

    def set_road_type(self, route: str, first: str, last: str, start_ns: int, end_ns: int, road_type: str | None,
                      base_revision: int, reviewer: str) -> int:
        """A person's road type (None: they cannot tell) on the frames of the map from first to last (frame keys), from
        start_ns to end_ns on the page, if the drive is still at base_revision: kept with road_types.correct, last, so
        that a correction refused writes nothing. Returns the new revision."""
        with self._writing(route, base_revision) as (connection, revision):
            self._log(connection, route, revision, 'type_corrected', start_ns, end_ns,
                      {**NO_LABEL, 'state': road_type or 'cant_tell', 'note': f'{first} to {last}'}, reviewer)
            try:
                road_types.correct(self.root, route, first, last, road_type, reviewer)
            except road_types.CorrectionError as exc:
                raise SpeedReviewError(400, 'bad_road_type', str(exc)) from exc
        return revision



class Batch:
    """The writes of SpeedReviewStore.batch, on its open transaction: what a drive has, read inside it, and the taps,
    each at the next revision. revision: the last one written (the base one before any)."""

    def __init__(self, store: SpeedReviewStore, connection: sqlite3.Connection, route: str, revision: int) -> None:
        self.store, self.connection, self.route = store, connection, route
        self.next = revision

    @property
    def revision(self) -> int:
        return self.next - 1

    def kinds(self) -> dict[str, str]:
        """{kind: label} of the kinds of road."""
        return {row['kind']: row['label'] for row in self.connection.execute('SELECT kind, label FROM speed_road_kinds')}

    def marks(self, kind: str | None) -> list[dict]:
        """The marks of a kind of road (None: the roadworks), in time order."""
        if kind is None:
            rows = self.connection.execute('SELECT start_ns, end_ns, reviewer, revision FROM speed_works WHERE route = ? '
                                           'ORDER BY start_ns', (self.route,))
        else:
            rows = self.connection.execute('SELECT start_ns, end_ns, reviewer, revision FROM speed_road WHERE route = ? '
                                           'AND kind = ? ORDER BY start_ns', (self.route, kind))
        return [dict(row) for row in rows]

    def checked(self, kind: str | None) -> int | None:
        """Up to when a person went through the drive for a kind (None: the roadworks), None when nobody did."""
        if kind is None:
            row = self.connection.execute('SELECT end_ns FROM speed_works_checked WHERE route = ?', (self.route,)).fetchone()
        else:
            row = self.connection.execute('SELECT end_ns FROM speed_road_checked WHERE route = ? AND kind = ?',
                                          (self.route, kind)).fetchone()
        return None if row is None else row['end_ns']

    def history(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute('SELECT * FROM speed_history WHERE route = ? ORDER BY revision',
                                                             (self.route,))]

    def tap(self, kind: str | None, tap: str, at_ns: int, drive_end: int, reviewer: str) -> None:
        """One tap as tap_works (kind None) or tap_road, at the next revision; refused (SpeedReviewError), it changes
        nothing."""
        label = ''
        if kind is not None:
            label = self.kinds().get(kind)
            if label is None:
                raise SpeedReviewError(400, 'bad_kind', 'No such kind of road.')
        self.store._tap_on(self.connection, self.route, self.next, kind, tap, at_ns, drive_end, reviewer, label)
        self.next += 1

    def remove(self, kind: str | None, start_ns: int, drive_end: int, reviewer: str) -> None:
        """The mark of a kind (None: the roadworks) that starts at start_ns out, at the next revision, as a remove tap
        would, but that very mark: never a mark of someone else whose span covers start_ns. Refused (SpeedReviewError)
        when there is none."""
        label = ''
        if kind is not None:
            label = self.kinds().get(kind)
            if label is None:
                raise SpeedReviewError(400, 'bad_kind', 'No such kind of road.')
        self.store._tap_on(self.connection, self.route, self.next, kind, 'remove', start_ns, drive_end, reviewer, label,
                           exact=True)
        self.next += 1

    def mark(self, kind: str | None, start_ns: int, end_ns: int, drive_end: int, reviewer: str) -> bool:
        """A whole mark, its start and its end, or nothing of it: False when the store refuses either."""
        first = self.next
        self.connection.execute('SAVEPOINT mark')
        try:
            self.tap(kind, 'start', start_ns, drive_end, reviewer)
            self.tap(kind, 'end', end_ns, drive_end, reviewer)
        except SpeedReviewError:
            self.connection.execute('ROLLBACK TO mark')
            self.connection.execute('RELEASE mark')
            self.next = first
            return False
        self.connection.execute('RELEASE mark')
        return True

    def log(self, action: str, start_ns: int, end_ns: int, reviewer: str, state: str | None = None, note: str = '') -> None:
        """A row of speed_history at the next revision, for a write outside the store (the labeler's road types)."""
        self.store._log(self.connection, self.route, self.next, action, start_ns, end_ns,
                        {**NO_LABEL, 'state': state, 'note': note[:MAX_NOTE]}, reviewer)
        self.next += 1

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


def _writable_span(drives: DriveReviewStore, route: str) -> tuple[int, int]:
    """The span of a drive Speed review writes on (blocks, roadworks, kinds of road, road types): only a dashcam drive
    (dashcam_drive), the only ones the brain learns from; on the others Speed review is off (#69): its Signs side stays."""
    span = _span_of(drives, route)
    if span is None:
        raise SpeedReviewError(404, 'not_scanned', 'This drive has no scanned logs.')
    if not dashcam_drive(drives.summary(route)):
        raise SpeedReviewError(409, 'not_dashcam', 'sunnypilot was on in this drive: Speed review is off on it, the '
                               'brain learns only drives with sunnypilot off.')
    return span


def road_type_frames(root: Path, route: str, merged: dict | None) -> list[tuple[int, dict]]:
    """The road type of each frame of the map, one a second, as the brain takes it (road_types.load_road_types: the
    map's, the road labeler's layer over it, a person's corrections over both), with the time the road camera exposed
    that frame on the comma's clock (its decoded frame in the drive's encode index, drive_logs.frame_time), in time
    order: [(ns, line)]. A frame the encode index lacks is left out; [] before Read the map."""
    from nnslr_tools.road_type_data import FRAMES_RELATIVE
    lines = road_types.load_road_types(root, route)
    if not lines or not merged:
        return []
    decoded = {}
    for path in (Path(root) / FRAMES_RELATIVE / route).glob('*/frames.jsonl'):
        for text in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(text)
                decoded[f"{row['segment_index']}/{row['output_index']}"] = int(row['decoded_frame_index'])
            except (ValueError, TypeError, KeyError):
                continue
    index = drive_logs.frame_index(merged)
    found = []
    for line in lines:
        segment, second = line.get('segment_index'), line.get('output_index')
        if not isinstance(segment, int) or not isinstance(second, int):
            continue
        ns = drive_logs.frame_time(index, segment, decoded.get(line['frame_key'], second * drive_logs.FPS))
        if ns is not None:
            found.append((ns, line))
    found.sort(key=lambda item: item[0])
    return found


def _type_source(line: dict) -> tuple[str, str | None, str | None]:
    """(source: 'person', 'net' or 'map'; the person's reviewer or None; the labeler's run or None) of a line of
    road_type_frames."""
    source = road_types.source_of(line)
    if line.get('reviewed') is True:
        return 'person', line.get('reviewer'), None
    if road_types.by_labeler(source):
        return 'net', None, source[len(road_types.NET_PREFIX):]
    return 'map', None, None


def road_type_stretches(frames: list[tuple[int, dict]], drive_end: int) -> list[dict]:
    """The drive as stretches of one road type for the page, from road_type_frames: [{road_type (None: the map has none,
    or a person cannot tell), source ('map', 'net': the road labeler's, 'person'), reviewer (the person's, None
    otherwise), run (the labeler's, None otherwise), from, to (frame keys of its first and last frame), from_ns, to_ns}].
    Frames of two sources (or two persons, or two runs) are never one stretch. A stretch runs to the next one's first
    frame, the last one a second past its last frame (never past drive_end), and frames missing (ROAD_TYPE_GAP_NS) end
    one."""
    found: list[dict] = []
    last_ns = None
    for ns, line in frames:
        source, reviewer, run = _type_source(line)
        current = found[-1] if found else None
        gap = last_ns is not None and ns - last_ns >= ROAD_TYPE_GAP_NS
        if (current and not gap and current['road_type'] == line.get('road_type') and current['source'] == source
                and current['reviewer'] == reviewer and current['run'] == run):
            current['to'] = line['frame_key']
        else:
            if current:
                current['to_ns'] = last_ns + NS if gap else ns
            found.append({'road_type': line.get('road_type'), 'source': source, 'reviewer': reviewer, 'run': run,
                          'from': line['frame_key'], 'to': line['frame_key'], 'from_ns': ns, 'to_ns': None})
        last_ns = ns
    if found:
        found[-1]['to_ns'] = last_ns + NS
    for stretch in found:
        stretch['to_ns'] = max(stretch['from_ns'] + 1, min(stretch['to_ns'], drive_end))
    return found


def drive_view(store: SpeedReviewStore, drives: DriveReviewStore, route: str) -> dict:
    """A drive for the page: its revision, the driver's speed along it (speed: speed_curve), where a machine held it
    (held: held_spans), its blocks, its roadworks, the kinds of road (road_kinds), their marks (road) and how far each was
    gone through (road_checked_ns, by kind), the road types of the map with a person's corrections (road_types:
    road_type_stretches; road_type_names, in their order), its first and last ns (span_ns, where a block may go), its
    length and whether it is a drive the brain learns from (dashcam: dashcam_drive; each one does, whole:
    brain_targets.training_routes), and the reading of its logs (scan: state, progress, error of
    DriveReviewStore.scan_state). A drive whose logs are not scanned has no speed, no span, no road types and
    dashcam None."""
    span = _span_of(drives, route)
    scan = drives.scan_state(route)
    view = {'ok': True, 'route': route, 'revision': store.revision(route), 'blocks': store.blocks(route),
            'works': store.works(route), 'works_checked_ns': store.works_checked(route), 'road_kinds': store.road_kinds(),
            'road': store.road(route), 'road_checked_ns': store.road_checked(route), 'road_types': [],
            'road_type_names': list(road_types.ROAD_TYPES), 'speed': [], 'held': [],
            'span_ns': None, 'total_s': 0, 'max_kph': MAX_KPH,
            'dashcam': None if span is None else dashcam_drive(drives.summary(route)),
            'scan': {key: scan[key] for key in ('state', 'progress', 'error')}}
    if span is not None:
        merged = drives.merged(route)
        view.update(speed=speed_curve(merged, span), held=held_spans(merged, span), span_ns=list(span),
                    total_s=round((span[1] - span[0]) / NS, 1),
                    road_types=road_type_stretches(road_type_frames(store.root, route, merged), span[1]))
    return view


def drives_list(store: SpeedReviewStore, drives: DriveReviewStore) -> list[dict]:
    """The drives for the page, which puts them in order: each with its length, how many blocks, roadworks marks and
    marks of a kind of road it has, and whether the brain learns from it (dashcam: dashcam_drive, None before its logs are
    read). The page lists only the ones not False; Read the logs and Regression take them all."""
    counts = store.counts()
    with drives.reading():
        return _drives_list(drives, counts)


def _drives_list(drives: DriveReviewStore, counts: dict) -> list[dict]:
    rows = []
    for drive in drives_view(drives):
        span = _span_of(drives, drive['route'])
        rows.append({**drive, **counts.get(drive['route'], {'blocks': 0, 'works': 0, 'road': 0}),
                     'total_s': round((span[1] - span[0]) / NS, 1) if span else 0,
                     'dashcam': None if span is None else dashcam_drive(drives.summary(drive['route'])),
                     'car': car_name(drives.summary(drive['route']))})
    return rows


def car_name(summary: dict | None) -> str | None:
    """The car of a drive in words, from the fingerprint its logs gave ('PSA_PEUGEOT_3008' -> 'Peugeot 3008',
    'VOLKSWAGEN_GOLF_MK7' -> 'Volkswagen Golf Mk7'): what the page filters the drives by. None before its logs are read."""
    car = ((summary or {}).get('meta') or {}).get('car') or {}
    fingerprint = str(car.get('fingerprint') or '').strip()
    if not fingerprint:
        return None
    words = fingerprint.split('_')
    if len(words) > 1 and words[0] == 'PSA':  # the group, before the brand
        words = words[1:]
    return ' '.join(word if any(ch.isdigit() for ch in word) and not word.startswith('MK') else word.capitalize()
                    for word in words)


def frame_jpeg(root: Path, route: str, segment: int, decoded: int) -> bytes:
    """Frame `decoded` of minute `segment` of the drive's road video at its full definition (the comma's fcamera.hevc,
    1344x760 or 2688x1520): what the player shows over its smaller video while it stands still, with the HD switch on,
    the same pixels Sign review labels. Frame k of a minute of the player's video is frame k of that minute's
    fcamera.hevc (nnslr_tools.drive_video). SpeedReviewError 404 when the server has no such frame."""
    try:
        data = video_frames.preview(root, route, segment, decoded).read_bytes()
    except (video_frames.VideoFrameError, OSError) as exc:
        raise SpeedReviewError(404, 'no_video_frame', 'This frame of the video cannot be shown.') from exc
    # The GOP after it (or before) decoded in the background: a step into it finds its frames ready.
    video_frames.read_ahead(root, route, segment, decoded)
    return data


def save_block(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's block, set ({start_ns, end_ns, kph, note}), moved (the same and from_start_ns, the start it had) or
    removed ({remove: true, start_ns}): checks it against the drive, applies it, and answers with the drive as it is now."""
    span = _writable_span(drives, route)
    if body.get('remove') is True:
        store.remove_block(route, _integer(body.get('start_ns'), 'start_ns'), _integer(body.get('base_revision'), 'base_revision'),
                           reviewer)
    else:
        start, end, kph, note, base = block_of(body, span)
        moved = body.get('from_start_ns')
        store.set_block(route, start, end, kph, note, base, reviewer,
                        None if moved is None else _integer(moved, 'from_start_ns'))
    return drive_view(store, drives, route)


def save_works(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's tap on the roadworks: checks it against the drive, applies it, and answers with the drive as it is now."""
    span = _writable_span(drives, route)
    tap, at, base = works_tap(body, span)
    store.tap_works(route, tap, at, span[1], base, reviewer)
    return drive_view(store, drives, route)


def save_road(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's tap on a kind of road ({kind, tap, at_ns, base_revision}, as save_works): checks it against the drive,
    applies it, and answers with the drive as it is now."""
    span = _writable_span(drives, route)
    kind = body.get('kind')
    if not isinstance(kind, str):
        raise SpeedReviewError(400, 'bad_kind', 'kind is the name of a kind of road.')
    tap, at, base = works_tap(body, span)
    store.tap_road(route, kind, tap, at, span[1], base, reviewer)
    return drive_view(store, drives, route)


def save_road_type(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's road type ({road_type: one of road_types.ROAD_TYPES, or null when they cannot tell, base_revision}) on
    the whole stretch of the page the frame shown is in ({at_ns}: road_type_stretches), or from one frame to another
    ({from_ns, to_ns}, in either order: the frames of the map shown at those times, both kept): checks it against the
    drive, keeps it (SpeedReviewStore.set_road_type), and answers with the drive as it is now."""
    span = _writable_span(drives, route)
    road_type = body.get('road_type')
    if 'road_type' not in body or (road_type is not None and road_type not in road_types.ROAD_TYPES):
        raise SpeedReviewError(400, 'bad_road_type', 'road_type is one of ' + ', '.join(road_types.ROAD_TYPES)
                               + ', or null when it cannot be told.')
    base = _integer(body.get('base_revision'), 'base_revision')
    stretch = 'at_ns' in body
    if stretch == ('from_ns' in body or 'to_ns' in body):
        raise SpeedReviewError(400, 'bad_stretch', 'Say at_ns for the stretch of a frame, or from_ns and to_ns.')
    times = [_integer(body.get(name), name) for name in (('at_ns',) if stretch else ('from_ns', 'to_ns'))]
    if not all(span[0] <= ns <= span[1] for ns in times):
        raise SpeedReviewError(400, 'bad_stretch', 'This frame is outside the logs of the drive.')
    frames = road_type_frames(store.root, route, drives.merged(route))
    if not frames:
        raise SpeedReviewError(404, 'no_road_types', 'This drive has no road types yet: Read the map in Data › Prepare.')
    if stretch:
        here = next((item for item in road_type_stretches(frames, span[1])
                     if item['from_ns'] <= times[0] < item['to_ns']), None)
        if here is None:
            raise SpeedReviewError(400, 'bad_stretch', 'The map has no frame here.')
        first, last, start, end = here['from'], here['to'], here['from_ns'], here['to_ns']
    else:
        # The frame shown at each time: the last one at or before it, the first one before the map starts.
        at = [max(0, bisect.bisect_right(frames, ns, key=lambda item: item[0]) - 1) for ns in sorted(times)]
        if frames[at[1]][0] > max(times):
            raise SpeedReviewError(400, 'bad_stretch', 'The map has no frame from here to there.')
        (start, line_first), (last_ns, line_last) = frames[at[0]], frames[at[1]]
        first, last, end = line_first['frame_key'], line_last['frame_key'], min(last_ns + NS, span[1])
    store.set_road_type(route, first, last, start, max(end, start + 1), road_type, base, reviewer)
    return drive_view(store, drives, route)


def save_road_kind(store: SpeedReviewStore, body: dict, reviewer: str) -> dict:
    """A person's new kind of road ({label}): answers with its kind and all the kinds."""
    label = body.get('label')
    if not isinstance(label, str):
        raise SpeedReviewError(400, 'bad_value', 'label is the name of the kind of road.')
    kind = store.add_road_kind(label, reviewer)
    return {'ok': True, 'kind': kind, 'road_kinds': store.road_kinds()}
