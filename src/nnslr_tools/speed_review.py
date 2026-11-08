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
on 127.0.0.1 without sign-in (reviewer 'local'); never an agent (no MCP tool, no REST write, no CLI). One write sets a
block or removes one, in one transaction with the check of the revision the page read (a stale one is refused with 409,
so two tabs cannot overwrite each other). The drive's revision is the number of the last write of that drive in
speed_history, a table that only grows (triggers refuse the rest): a drive's speed_blocks can always be rebuilt by
playing its history again (block_saved, block_removed).

Roadworks. Two taps of a person on the frame shown: here they start, here they end. A start with no end yet runs to the
end of the drive. A mark says nothing of the speed: it lies next to the answers, so that the training can tell the brain
where the works were and measure its error there. It is never an input of the car. The taps are writes of the drive like
the others, same revision and same rules (works_start, works_end, works_removed in speed_history, without a speed), and
speed_works is rebuilt by playing them again. No mark does not mean no works. Only a third tap does: "checked to here"
says that a person went through the drive from its start to the frame shown and marked all the works there
(works_checked in speed_history, the last one holds, speed_works_checked keeps it), so what lies there outside the marks
is free of works. Saying where works are not is what lets the brain learn them at all (nnslr_tools.brain_targets).

The old stretches. Until 8 Nov 2026 a person said the speed stretch by stretch, on a grid of 5 km/h, from a proposal of
the OpenStreetMap limit: speed_stretches and its rows of speed_history (saved, cleared) stay in the file, and nothing
reads them any more.

``annotations/speed_review.sqlite`` is a file of its own, readable only by its owner like review.sqlite:
drive_review.sqlite holds what a scan can make again, this holds what only a person can.
"""
from __future__ import annotations

import bisect
import contextlib
import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from nnslr_tools import drive_logs, video_frames
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
NO_LABEL = {'kph': None, 'state': None, 'known_ns': None, 'note': ''}

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
        """Per drive, how many blocks and roadworks marks it has."""
        out: dict[str, dict[str, int]] = {}
        for table, key in (('speed_blocks', 'blocks'), ('speed_works', 'works')):
            for row in self._rows(f'SELECT route, COUNT(*) AS n FROM {table} GROUP BY route'):
                out.setdefault(row['route'], {'blocks': 0, 'works': 0})[key] = row['n']
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

    @staticmethod
    def _log(connection: sqlite3.Connection, route: str, revision: int, action: str, start_ns: int, end_ns: int, item: dict,
             reviewer: str) -> None:
        connection.execute('INSERT INTO speed_history (route, revision, action, start_ns, end_ns, kph, state, known_ns, note, '
                           'reviewer, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (route, revision, action, start_ns, end_ns, item['kph'], item['state'], item['known_ns'],
                            item['note'], reviewer, utc_now()))

    def set_block(self, route: str, start_ns: int, end_ns: int, kph: int | None, note: str, base_revision: int,
                  reviewer: str) -> int:
        """Sets the block [start_ns, end_ns) to kph (None: left out), if the drive is still at base_revision; the blocks
        under it keep what stays outside it, a piece shorter than MIN_BLOCK_SECONDS goes (the page could only remove it).
        Returns the new revision."""
        with self._writing(route, base_revision) as (connection, revision):
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
            self._log(connection, route, revision, 'block_saved', start_ns, end_ns, {**NO_LABEL, 'kph': kph, 'note': note}, reviewer)
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
    """A drive for the page: its revision, the driver's speed along it (speed: speed_curve), its blocks, its roadworks,
    its first and last ns (span_ns, where a block may go) and its length. A drive whose logs are not scanned has no
    speed and no span."""
    span = _span_of(drives, route)
    view = {'ok': True, 'route': route, 'revision': store.revision(route), 'blocks': store.blocks(route),
            'works': store.works(route), 'works_checked_ns': store.works_checked(route), 'speed': [], 'span_ns': None,
            'total_s': 0, 'max_kph': MAX_KPH}
    if span is not None:
        view.update(speed=speed_curve(drives.merged(route), span), span_ns=list(span),
                    total_s=round((span[1] - span[0]) / NS, 1))
    return view


def drives_list(store: SpeedReviewStore, drives: DriveReviewStore) -> list[dict]:
    """The drives for the page, which puts them in order: each with its length and how many blocks and roadworks marks
    it has."""
    counts = store.counts()
    rows = []
    for drive in drives_view(drives):
        span = _span_of(drives, drive['route'])
        rows.append({**drive, **counts.get(drive['route'], {'blocks': 0, 'works': 0}),
                     'total_s': round((span[1] - span[0]) / NS, 1) if span else 0})
    return rows


def frame_jpeg(root: Path, route: str, segment: int, decoded: int) -> bytes:
    """Frame `decoded` of minute `segment` of the drive's road video at its full definition (the comma's fcamera.hevc,
    1344x760 or 2688x1520): what the player shows over its smaller video while it stands still, with the HD switch on,
    the same pixels Sign review labels. Frame k of a minute of the player's video is frame k of that minute's
    fcamera.hevc (nnslr_tools.drive_video). SpeedReviewError 404 when the server has no such frame."""
    try:
        return video_frames.preview(root, route, segment, decoded).read_bytes()
    except (video_frames.VideoFrameError, OSError) as exc:
        raise SpeedReviewError(404, 'no_video_frame', 'This frame of the video cannot be shown.') from exc


def save_block(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's block, set ({start_ns, end_ns, kph, note}) or removed ({remove: true, start_ns}): checks it against the
    drive, applies it, and answers with the drive as it is now."""
    span = _span_of(drives, route)
    if span is None:
        raise SpeedReviewError(404, 'not_scanned', 'This drive has no scanned logs.')
    if body.get('remove') is True:
        store.remove_block(route, _integer(body.get('start_ns'), 'start_ns'), _integer(body.get('base_revision'), 'base_revision'),
                           reviewer)
    else:
        start, end, kph, note, base = block_of(body, span)
        store.set_block(route, start, end, kph, note, base, reviewer)
    return drive_view(store, drives, route)


def save_works(store: SpeedReviewStore, drives: DriveReviewStore, route: str, body: dict, reviewer: str) -> dict:
    """A person's tap on the roadworks: checks it against the drive, applies it, and answers with the drive as it is now."""
    span = _span_of(drives, route)
    if span is None:
        raise SpeedReviewError(404, 'not_scanned', 'This drive has no scanned logs.')
    tap, at, base = works_tap(body, span)
    store.tap_works(route, tap, at, span[1], base, reviewer)
    return drive_view(store, drives, route)
