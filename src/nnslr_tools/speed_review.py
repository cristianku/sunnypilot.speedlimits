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
joined to the one before. It is worked out when a drive is shown and never stored. OpenStreetMap is never a proposal.

The writes. Only a person on the site writes them: signed in, or the person at the server itself on a site served only
on 127.0.0.1 without sign-in (reviewer 'local'); never an agent (no MCP tool, no REST write, no CLI). One write (apply)
sets or clears [start_ns, end_ns): the rows under it are clipped or removed, in one transaction with the check of the
revision the page read (a stale one is refused with 409, so two tabs cannot overwrite each other). The drive's revision
is the number of the last write of that drive in speed_history, a table that only grows (triggers refuse the rest): a
drive's speed_stretches can always be rebuilt by playing its history again.

``annotations/speed_review.sqlite`` is a file of its own, readable only by its owner like review.sqlite:
drive_review.sqlite holds what a scan can make again, this holds what only a person can.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

from nnslr_tools.drive_review import DriveReviewStore, drives_view, utc_now

DB_RELATIVE_PATH = Path('annotations') / 'speed_review.sqlite'
SCHEMA_VERSION = 1
NS = 1_000_000_000
# A run of the vision limit shorter than this joins the one before: it is a flicker, not a stretch of road.
MERGE_SECONDS = 5
GRID_KPH = 5
MAX_KPH = 200
MAX_NOTE = 500
STATES = ('confirmed', 'corrected', 'undetermined')
# The drives to review first, in this order: whole drives chosen for what the brain must learn. Together 169 minutes
# with 93 % of the seconds where the limit looks stuck, the roadworks and the low limits on the carriageway.
FIRST_DRIVES = ('00000160--ea56ae0f33', '0000015a--e47d11b0b0', '00000154--fa7e8f49aa', '00000151--7adc20003e',
                '00000153--44d221e479')
# The drive to review last: it is the test drive, and its logs have no vision limit to propose from.
LAST_DRIVES = ('0000012d--6552a45378',)

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
    return [{'start_ns': first, 'end_ns': stop, 'kph': kph} for first, stop, kph in joined]


def partition(proposed: list[dict], saved: list[dict], span: tuple[int, int]) -> list[dict]:
    """The whole drive as stretches in time order: the saved ones, and in the gaps between them the proposals cut to
    the gap, not examined."""
    out: list[dict] = []

    def fill(start: int, stop: int) -> None:
        for item in proposed:
            first, last = max(start, item['start_ns']), min(stop, item['end_ns'])
            if first < last:
                out.append({'start_ns': first, 'end_ns': last, 'kph': item['kph'], 'state': 'not_examined',
                            'known_ns': None, 'note': '', 'reviewer': None})

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

    def apply(self, route: str, start_ns: int, end_ns: int, label: dict | None, base_revision: int, reviewer: str) -> int:
        """Sets [start_ns, end_ns) to label, or clears it (None: back to not examined), if the drive is still at
        base_revision; the stretches under it keep what stays outside it. Returns the new revision."""
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
                    under = connection.execute('SELECT * FROM speed_stretches WHERE route = ? AND start_ns < ? AND end_ns > ?',
                                               (route, end_ns, start_ns)).fetchall()
                    insert = ('INSERT INTO speed_stretches (route, start_ns, end_ns, kph, state, known_ns, note, reviewer, '
                              'revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)')
                    connection.executemany('DELETE FROM speed_stretches WHERE route = ? AND start_ns = ?',
                                           [(route, row['start_ns']) for row in under])
                    for row in under:
                        for first, last in ((row['start_ns'], start_ns), (end_ns, row['end_ns'])):
                            if first < last:
                                known = row['known_ns'] if row['known_ns'] is not None and first <= row['known_ns'] < last else None
                                connection.execute(insert, (route, first, last, row['kph'], row['state'], known, row['note'],
                                                            row['reviewer'], row['revision']))
                    revision += 1
                    if label is not None:
                        connection.execute(insert, (route, start_ns, end_ns, label['kph'], label['state'], label['known_ns'],
                                                    label['note'], reviewer, revision))
                    item = label or {'kph': None, 'state': None, 'known_ns': None, 'note': ''}
                    connection.execute('INSERT INTO speed_history (route, revision, action, start_ns, end_ns, kph, state, '
                                       'known_ns, note, reviewer, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                       (route, revision, 'saved' if label else 'cleared', start_ns, end_ns, item['kph'],
                                        item['state'], item['known_ns'], item['note'], reviewer, utc_now()))
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
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
    """A drive for the page: its revision, its stretches from start to end, how much of it is examined. A drive whose
    logs are not scanned has no stretches."""
    span = _span_of(drives, route)
    if span is None:
        return {'ok': True, 'route': route, 'revision': store.revision(route), 'stretches': [], 'examined_s': 0, 'total_s': 0}
    saved = store.stretches(route)
    return {'ok': True, 'route': route, 'revision': store.revision(route),
            'stretches': partition(proposals(drives.merged(route)), saved, span),
            'examined_s': round(sum(row['end_ns'] - row['start_ns'] for row in saved) / NS, 1),
            'total_s': round((span[1] - span[0]) / NS, 1)}


def drives_list(store: SpeedReviewStore, drives: DriveReviewStore) -> list[dict]:
    """The drives for the page: the first ones in their order, the others from the newest on, the last ones at the end.
    Each with its share examined; no proposals here (they need the drive's logs)."""
    examined = store.examined()
    rows = []
    for drive in drives_view(drives):
        span = _span_of(drives, drive['route'])
        rows.append({**drive, 'first': drive['route'] in FIRST_DRIVES, 'examined_s': round(examined.get(drive['route'], 0) / NS, 1),
                     'total_s': round((span[1] - span[0]) / NS, 1) if span else 0})

    def place(row: dict) -> tuple[int, int]:
        if row['first']:
            return 0, FIRST_DRIVES.index(row['route'])
        return (2, 0) if row['route'] in LAST_DRIVES else (1, 0)
    return sorted(rows, key=place)


def save(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's write: checks it against the drive, applies it, and answers with the drive as it is now."""
    span = _span_of(drives, route)
    if span is None:
        raise SpeedReviewError(404, 'not_scanned', 'This drive has no scanned logs.')
    start, end, label, base = checked(body, span)
    store.apply(route, start, end, label, base, reviewer)
    return drive_view(store, drives, route)
