"""Speed review (Data › Speed review): the speed a person wants the car to hold, stretch by stretch of a drive.

What it is for. The brain, the one network that will decide the speed, needs a target, and the limit in the logs cannot
be it: it holds the very errors to remove (an 80 that never ended, a limit read on the ramp). So a person goes along a
drive and says what the speed to hold really was. Nothing here runs on the car and nothing here trains: this only
collects the labels.

The label. kph is the cap on a free road, on a grid of 5 km/h: queues, curves and stops stay with the planner, which
slows down further on its own. A label holds on a stretch [start_ns, end_ns) of logMonoTime, the comma's clock, which
stays the same if a new scan moves the start of the drive:

- start_ns: where the car should begin to obey it: for a drop when the sign can be read, for a raise at the sign.
- known_ns (optional): from when the past is enough to work the limit out. A missed end of a limit starts at the sign
  nobody saw, and is known only further on, where the scene and the traffic tell it.
- state: 'confirmed' (the proposal stands), 'corrected' (another value) or 'undetermined' (a person looked and cannot
  say: kph is None). A stretch nobody looked at has no row: 'not_examined' is only ever computed, so a proposal left
  alone never counts as approved.

The proposal (proposals) is the limit the comma's vision held, in runs rounded to 5 km/h, a run under MERGE_SECONDS
joined to the one before (the first one of a drive to the one after). On free road the driver's own speed goes over it
(driven, with_driven): no car ahead, no curve, the speed steady for STEADY_SECONDS, and the proposal is the nearest limit
(78 km/h held proposes 80). A car ahead, a curve or a series not in the log is no free road: there the vision limit stays.
It is worked out when a drive is shown and never stored, so a proposal is an answer only once a person says yes to it.
OpenStreetMap is never a proposal.

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
import math
import os
import sqlite3
import statistics
import threading
from pathlib import Path
from typing import Iterator

from nnslr_tools.drive_review import DriveReviewStore, drives_view, utc_now

DB_RELATIVE_PATH = Path('annotations') / 'speed_review.sqlite'
SCHEMA_VERSION = 1
NS = 1_000_000_000
# A run of the vision limit shorter than this joins the one before, or at the start of a drive the one after: it is a
# flicker, not a stretch of road.
MERGE_SECONDS = 5
# The driven proposal: how long and how steady a speed has to be, and what is still straight (the sideways acceleration
# the GPS track gives, m/s^2). The track is read over TURN_FIXES fixes on each side, about a second apart; a gap in it
# longer than MAX_TRACK_GAP_S says nothing of the road.
STEADY_SECONDS = 15
STEADY_KPH = 6
MIN_DRIVEN_KPH = 25
STRAIGHT_LATERAL = 1.0
TURN_FIXES = 2
MAX_TRACK_GAP_S = 3
LIMITS_KPH = range(30, 140, 10)
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


def proposals(merged: dict | None) -> list[dict]:
    """The stretches the comma's vision limit proposes, from the start of the drive to its end, in time order: kph
    None where it held none (not sure, not in the log). Empty for a drive with nothing read."""
    if merged is None or merged.get('t0') is None or merged['span'][1] is None or merged['span'][1] <= merged['t0']:
        return []
    start, end = merged['t0'], merged['span'][1]
    points = [(t, value) for t, value in merged['series'].get('vis_limit') or () if start <= t < end]
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


def _turn_rates(gps: list[list]) -> list[tuple[int, float]]:
    """(t, rad/s) along the GPS track: the heading into a fix against the heading out of it."""
    def heading(a: list, b: list) -> float:
        return math.atan2((b[2] - a[2]) * math.cos(math.radians(a[1])), b[1] - a[1])

    rates = []
    for before, here, after in zip(gps, gps[TURN_FIXES:], gps[2 * TURN_FIXES:]):
        seconds = (after[0] - before[0]) / NS
        if seconds <= 2 * TURN_FIXES * MAX_TRACK_GAP_S:
            turn = (heading(here, after) - heading(before, here) + math.pi) % (2 * math.pi) - math.pi
            rates.append((here[0], abs(turn) / (seconds / 2)))
    return rates


def _steady_runs(seconds: list[tuple[int, float | None]]) -> Iterator[list[tuple[int, float]]]:
    """The runs of (t, kph) that stay inside STEADY_KPH; a second with no speed ends one."""
    run: list[tuple[int, float]] = []
    low = high = 0.0
    for t, kph in seconds:
        if run and (kph is None or max(high, kph) - min(low, kph) > STEADY_KPH):
            yield run
            run = []
        if kph is not None:
            low, high = (min(low, kph), max(high, kph)) if run else (kph, kph)
            run.append((t, kph))
    if run:
        yield run


def driven(merged: dict | None) -> list[dict]:
    """The stretches of free road where the driver held a steady speed, each proposing the limit nearest to it (a tie
    goes to the lower one, kph is a cap). What the log does not say counts as not free."""
    if merged is None or merged.get('t0') is None or merged['span'][1] is None:
        return []
    start, end = merged['t0'], merged['span'][1]
    speeds, leads = merged['series'].get('v_ego') or [], merged['series'].get('lead_present') or []
    rates = _turn_rates(merged.get('gps') or [])
    speed_times, lead_times, rate_times = [p[0] for p in speeds], [p[0] for p in leads], [p[0] for p in rates]

    def free_speed(t: int) -> float | None:
        at_speed = bisect.bisect_right(speed_times, t) - 1
        at_lead = bisect.bisect_right(lead_times, t) - 1
        at_rate = bisect.bisect_right(rate_times, t) - 1
        if min(at_speed, at_lead, at_rate) < 0 or leads[at_lead][1] is not False:
            return None
        kph = speeds[at_speed][1]
        if kph is None or kph < MIN_DRIVEN_KPH or t - rate_times[at_rate] > MAX_TRACK_GAP_S * NS:
            return None
        return kph if kph / 3.6 * rates[at_rate][1] <= STRAIGHT_LATERAL else None

    return [{'start_ns': run[0][0], 'end_ns': min(run[-1][0] + NS, end), 'source': 'driven',
             'kph': min(LIMITS_KPH, key=lambda limit, kph=statistics.median(kph for _, kph in run): (abs(limit - kph), limit))}
            for run in _steady_runs([(t, free_speed(t)) for t in range(start, end, NS)]) if len(run) >= STEADY_SECONDS]


def with_driven(proposed: list[dict], runs: list[dict], span: tuple[int, int]) -> list[dict]:
    """The proposals with the runs of driven on top: what the vision limit keeps under MERGE_SECONDS joins the stretch
    before, neighbours of one speed are one stretch, and that one says driven only if nothing of it comes from vision."""
    out: list[dict] = []
    for item in partition(proposed, runs, span):
        sliver = 'source' not in item and item['end_ns'] - item['start_ns'] < MERGE_SECONDS * NS
        if out and (sliver or out[-1]['kph'] == item['kph']):
            out[-1]['end_ns'] = item['end_ns']
            if not sliver and 'source' not in item:
                out[-1].pop('source', None)
        else:
            out.append(dict(item))
    return out


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
    """A drive for the page: its revision, its stretches from start to end, its roadworks, how much of it is examined. A
    drive whose logs are not scanned has no stretches."""
    span = _span_of(drives, route)
    if span is None:
        return {'ok': True, 'route': route, 'revision': store.revision(route), 'stretches': [], 'works': [],
                'works_checked_ns': None, 'examined_s': 0, 'total_s': 0}
    saved = store.stretches(route)
    merged = drives.merged(route)
    return {'ok': True, 'route': route, 'revision': store.revision(route),
            'stretches': partition(with_driven(proposals(merged), driven(merged), span), saved, span),
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
