"""The comma's own logs of the recorded drives, read on the server: what the car, openpilot, the vision and the map
did, second by second, for the drive player (Speed review, Regression, the guests' reviews), Sign distance and the
brain's training. Nothing is judged here. The scan also finds the episodes where the driver intervened or the vision
and the map disagreed (drive_episodes): the Episodes lane of Regression shows them. (Data › Drive review, the page
that listed them, was removed on 7 Nov 2026.)

Everything lives in ``<data root>/annotations/drive_review.sqlite`` next to the frame review's database, and nothing
here touches that one, the dataset or a training:

- log_files: every log of a drive the portal knows, by drive, segment, kind and the SHA-256 of its content, with
  where it came from and what it covers once read (drive_logs). Importing the same files again adds nothing.
- log_errors: the logs of a drive found on this server that could not be opened (a file of another user, unreadable
  for this server): never read until they can be, and shown as such (state unreadable_logs when they are all it has).
- scans: the last scan of each drive: its state (queued, scanning, done, failed), progress, the scanner's and the
  extractor's versions, the files it read, what the logs cover and the date of the drive.
- episodes: what the scanner found (drive_episodes). A scan again keeps the id of an episode it finds again (the
  same channel, overlapping in time); one no longer found is kept, inactive.
- annotations, annotation_history: the verdicts of the page's first version, when episodes were judged here. Kept
  in the file, never read or written any more.

The scan runs in one background thread of the site (one drive at a time, the logs read by a few child processes at
low priority): CPU only, no image, no network.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import queue
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools import drive_episodes, drive_logs
from nnslr_tools.drive_extract import EXTRACT_VERSION
from nnslr_tools.drive_logs import DriveLogError

DB_RELATIVE_PATH = Path('annotations') / 'drive_review.sqlite'
SCHEMA_VERSION = 1
EXTRACT_WORKERS = 3
NS = 1_000_000_000
# Episodes of two scans are the same when they overlap within this margin.
MATCH_NS = 5 * NS

SCHEMA_STATEMENTS = (
    'CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)',
    '''CREATE TABLE IF NOT EXISTS log_files (
        route TEXT NOT NULL, segment INTEGER NOT NULL, kind TEXT NOT NULL, sha256 TEXT NOT NULL,
        size INTEGER NOT NULL, source TEXT NOT NULL, path TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'new',
        error TEXT, first_ns INTEGER, last_ns INTEGER, messages INTEGER, counts TEXT, extract_version INTEGER,
        added_at TEXT NOT NULL, read_at TEXT,
        PRIMARY KEY (route, segment, kind, sha256))''',
    '''CREATE TABLE IF NOT EXISTS log_errors (
        route TEXT NOT NULL, path TEXT NOT NULL, segment INTEGER NOT NULL, kind TEXT NOT NULL, error TEXT NOT NULL,
        seen_at TEXT NOT NULL, PRIMARY KEY (route, path))''',
    '''CREATE TABLE IF NOT EXISTS file_hashes (
        path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS scans (
        route TEXT PRIMARY KEY, state TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, scanner_version INTEGER,
        extract_version INTEGER, started_at TEXT, finished_at TEXT, error TEXT, files TEXT, summary TEXT)''',
    '''CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY, route TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL,
        anchor_ns INTEGER NOT NULL, channel TEXT NOT NULL, kinds TEXT NOT NULL, priority INTEGER NOT NULL,
        payload TEXT NOT NULL, active INTEGER NOT NULL, scanner_version INTEGER NOT NULL,
        first_seen TEXT NOT NULL, last_seen TEXT NOT NULL)''',
    'CREATE INDEX IF NOT EXISTS episodes_route ON episodes (route, start_ns)',
    '''CREATE TABLE IF NOT EXISTS annotations (
        id TEXT PRIMARY KEY, route TEXT NOT NULL, episode_id TEXT, revision INTEGER NOT NULL, payload TEXT NOT NULL,
        author TEXT NOT NULL, author_kind TEXT NOT NULL, origin TEXT NOT NULL, created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0)''',
    'CREATE INDEX IF NOT EXISTS annotations_route ON annotations (route)',
    'CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)',
    '''CREATE TABLE IF NOT EXISTS annotation_history (
        id TEXT NOT NULL, revision INTEGER NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL, author TEXT NOT NULL,
        author_kind TEXT NOT NULL, origin TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY (id, revision))''',
)


# Drives before this day are left out by Read the logs: the vision network on the car was not yet the one worth
# judging (Cristian, 31 Oct 2026). Each is still read from its player; a stored '' takes every drive.
DEFAULT_REVIEW_FROM = '2026-10-30'


class DriveReviewError(ValueError):
    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def db_path(root: Path) -> Path:
    return Path(root) / DB_RELATIVE_PATH


def _loads(text, default=None):
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def _dumps(value) -> str:
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


class DriveReviewStore:
    """The logs' database and its scan worker. Every call opens its own connection (the site's threads share
    the store); writes are serialized by a lock."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = db_path(self.root)
        self._write = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._queued: set[str] = set()
        self._running: str | None = None
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._merged: dict[str, tuple] = {}
        # Per cached extract (its path): (mtime_ns, stale) - drive_logs.stale_extract, read once per version of it.
        self._stale: dict[str, tuple[int, bool]] = {}
        # The connection of a thread inside reading(): the queries of a whole list on it.
        self._local = threading.local()
        self._initialize()

    # ---------- database ----------

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout = 30000')
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                # A scan the site was running when it stopped never ends: it is reported as interrupted.
                connection.execute("UPDATE scans SET state = 'failed', error = 'interrupted: the site restarted', "
                                   "finished_at = ? WHERE state IN ('queued', 'scanning')", (utc_now(),))
                connection.execute('COMMIT')
            finally:
                connection.close()

    def _run(self, statements) -> None:
        """Runs [(sql, args)] in one write transaction."""
        with self._write:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    for sql, args in statements:
                        connection.execute(sql, args)
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()

    @contextmanager
    def reading(self):
        """The reads of a whole list (drives_view, a list of the routes) on one connection, not one per query."""
        if getattr(self._local, 'connection', None) is not None:
            yield
            return
        connection = self._connect()
        self._local.connection = connection
        try:
            yield
        finally:
            self._local.connection = None
            connection.close()

    def _rows(self, sql: str, args=()) -> list[sqlite3.Row]:
        shared = getattr(self._local, 'connection', None)
        if shared is not None:
            return shared.execute(sql, args).fetchall()
        connection = self._connect()
        try:
            return connection.execute(sql, args).fetchall()
        finally:
            connection.close()

    # ---------- the logs ----------

    def _sha(self, path: str) -> tuple[str, int]:
        info = os.stat(path)
        rows = self._rows('SELECT sha256 FROM file_hashes WHERE path = ? AND size = ? AND mtime_ns = ?',
                          (path, info.st_size, info.st_mtime_ns))
        if rows:
            return rows[0]['sha256'], info.st_size
        sha = drive_logs.sha256_file(Path(path))
        self._run([('INSERT OR REPLACE INTO file_hashes (path, size, mtime_ns, sha256) VALUES (?, ?, ?, ?)',
                    (path, info.st_size, info.st_mtime_ns, sha))])
        return sha, info.st_size

    def import_logs(self, route: str) -> dict:
        """Records the logs of a drive found on this server (drive_logs.discover); the same content is recorded once.
        A file that cannot be opened is recorded in log_errors (and forgotten there once it can, or is gone).
        {found, new, files}."""
        drive_logs.check_route(route)
        found = drive_logs.discover(self.root, route)
        known = {(row['segment'], row['kind'], row['sha256']): row for row in
                 self._rows('SELECT * FROM log_files WHERE route = ?', (route,))}
        statements, new = [], 0
        now = utc_now()
        failed: list[tuple] = []
        for item in found:
            try:
                sha, size = self._sha(item['path'])
            except OSError as exc:
                failed.append((route, item['path'], item['segment'], item['kind'],
                               f'{type(exc).__name__}: {exc.strerror or "cannot be read"}'[:200], now))
                continue
            key = (item['segment'], item['kind'], sha)
            row = known.get(key)
            if row is None:
                new += 1
                known[key] = {'path': item['path']}
                statements.append(('INSERT INTO log_files (route, segment, kind, sha256, size, source, path, '
                                   'added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                                   (route, item['segment'], item['kind'], sha, size, item['source'], item['path'], now)))
            elif not os.path.exists(row['path']):
                # The same content, moved: read it where it is now.
                statements.append(('UPDATE log_files SET path = ?, source = ? WHERE route = ? AND segment = ? '
                                   'AND kind = ? AND sha256 = ?', (item['path'], item['source'], route, *key)))
        statements.append(('DELETE FROM log_errors WHERE route = ?', (route,)))
        statements.extend(('INSERT INTO log_errors (route, path, segment, kind, error, seen_at) '
                           'VALUES (?, ?, ?, ?, ?, ?)', row) for row in failed)
        self._run(statements)
        return {'found': len(found), 'new': new, 'files': len(known)}

    def log_errors(self, route: str) -> list[dict]:
        """The logs of a drive found but not readable by this server: [{segment, kind, error, seen_at}] (no path)."""
        return [dict(row) for row in self._rows('SELECT segment, kind, error, seen_at FROM log_errors WHERE route = ? '
                                                'ORDER BY segment, kind', (route,))]

    def add_upload(self, route: str, segment: int, kind: str, ext: str, stream, length: int) -> dict:
        """A log sent from the portal: kept under raw/drive_logs, recorded, and the drive queued for a scan. The same
        content already known (from the comma or an earlier upload) is not kept twice."""
        if not drive_logs.known_route(self.root, route):
            raise DriveReviewError(404, 'unknown_route', 'This drive is not on the server: download it from the comma '
                                                         'first (its video), then add its logs.')
        stored = drive_logs.store_upload(self.root, route, segment, kind, ext, stream, length)
        present = self._rows('SELECT path FROM log_files WHERE route = ? AND segment = ? AND kind = ? AND sha256 = ?',
                             (route, segment, kind, stored['sha256']))
        duplicate = bool(present) and present[0]['path'] != stored['path'] and os.path.exists(present[0]['path'])
        if duplicate and stored['new']:
            os.unlink(stored['path'])
        if not present:
            self._run([('INSERT INTO log_files (route, segment, kind, sha256, size, source, path, added_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (route, segment, kind, stored['sha256'], stored['size'], 'upload', stored['path'], utc_now()))])
        self.queue_scan(route)
        return {'segment': segment, 'kind': kind, 'sha256': stored['sha256'], 'size': stored['size'],
                'added': not present}

    def log_files(self, route: str) -> list[dict]:
        out = []
        for row in self._rows('SELECT * FROM log_files WHERE route = ? ORDER BY segment, kind DESC, size DESC',
                              (route,)):
            item = dict(row)
            item['counts'] = _loads(item['counts'], {})
            item['present'] = os.path.exists(item['path'])
            item.pop('path')
            out.append(item)
        return out

    # ---------- settings ----------

    def review_from(self) -> str:
        """The first day (YYYY-MM-DD) whose drives are reviewed; '' for every drive."""
        rows = self._rows("SELECT value FROM settings WHERE key = 'review_from'")
        return rows[0]['value'] if rows else DEFAULT_REVIEW_FROM

    # ---------- the scan ----------

    def queue_scan(self, route: str) -> dict:
        drive_logs.check_route(route)
        with self._worker_lock:
            if route not in self._queued:
                self._queued.add(route)
                # A drive being scanned now is scanned again after, with what was added meanwhile.
                if route != self._running:
                    self._run([("INSERT INTO scans (route, state, progress) VALUES (?, 'queued', 0) ON CONFLICT(route) "
                                "DO UPDATE SET state = 'queued', progress = 0, error = NULL", (route,))])
                self._queue.put(route)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name='drive-review-scan', daemon=True)
                self._worker.start()
        return self.scan_state(route)

    def _work(self) -> None:
        while True:
            try:
                route = self._queue.get(timeout=30)
            except queue.Empty:
                with self._worker_lock:
                    if self._queue.empty():
                        self._worker = None
                        return
                continue
            with self._worker_lock:
                self._queued.discard(route)
                self._running = route
            try:
                self.scan(route)
            except Exception as exc:  # recorded for the page; the worker goes on with the next drive
                message = exc.message if isinstance(exc, DriveLogError) else f'{type(exc).__name__}: {exc}'
                self._run([("UPDATE scans SET state = 'failed', error = ?, finished_at = ? WHERE route = ?",
                            (message[:500], utc_now(), route))])
            finally:
                with self._worker_lock:
                    self._running = None

    def _progress(self, route: str, share: float) -> None:
        self._run([('UPDATE scans SET progress = ? WHERE route = ?', (round(share, 3), route))])

    def scan(self, route: str) -> dict:
        """Imports the drive's logs, reads the best log of each segment (cached), joins them and finds the episodes;
        records all of it. Returns the scan's state."""
        self.import_logs(route)
        self._run([("INSERT INTO scans (route, state, progress, started_at) VALUES (?, 'scanning', 0, ?) "
                    "ON CONFLICT(route) DO UPDATE SET state = 'scanning', progress = 0, started_at = excluded.started_at,"
                    " error = NULL", (route, utc_now()))])
        rows = self._rows('SELECT * FROM log_files WHERE route = ?', (route,))
        by_segment: dict[int, list] = {}
        for row in rows:
            # A file unreadable for an older extractor is tried again.
            retry = row['state'] != 'unreadable' or row['extract_version'] != EXTRACT_VERSION
            if retry and os.path.exists(row['path']):
                by_segment.setdefault(row['segment'], []).append(dict(row))
        seen = sorted({row['sha256'] for row in rows})
        if not by_segment:
            self._run([("UPDATE scans SET state = 'done', progress = 1, finished_at = ?, scanner_version = ?, "
                        "extract_version = ?, files = '{}', summary = ? WHERE route = ?",
                        (utc_now(), drive_episodes.SCANNER_VERSION, EXTRACT_VERSION,
                         _dumps({'seen': seen, 'unreadable': sorted({row['segment'] for row in rows})}), route))])
            self._set_episodes(route, [])
            return self.scan_state(route)
        schema = drive_logs.schema_dir()
        done = [0]
        lock = threading.Lock()

        def read(segment: int):
            candidates = sorted(by_segment[segment], key=lambda r: (-drive_logs.KIND_RANK[r['kind']], -r['size']))
            for row in candidates:
                try:
                    value = drive_logs.extract(self.root, route, segment, row['sha256'], row['path'], schema=schema)
                except DriveLogError as exc:
                    self._run([("UPDATE log_files SET state = 'unreadable', error = ?, read_at = ?, extract_version = ? "
                                "WHERE route = ? AND segment = ? AND kind = ? AND sha256 = ?",
                                (exc.message[:300], utc_now(), EXTRACT_VERSION, route, segment, row['kind'],
                                 row['sha256']))])
                    continue
                self._run([("UPDATE log_files SET state = 'read', error = NULL, first_ns = ?, last_ns = ?, "
                            "messages = ?, counts = ?, extract_version = ?, read_at = COALESCE(read_at, ?) "
                            "WHERE route = ? AND segment = ? AND kind = ? AND sha256 = ?",
                            (value['span'][0], value['span'][1], value['messages'], _dumps(value['counts']),
                             EXTRACT_VERSION, utc_now(), route, segment, row['kind'], row['sha256']))])
                with lock:
                    done[0] += 1
                    self._progress(route, 0.95 * done[0] / len(by_segment))
                return segment, row, value
            with lock:
                done[0] += 1
            return segment, None, None

        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            results = list(pool.map(read, sorted(by_segment)))
        extracts = {segment: value for segment, row, value in results if value is not None}
        files = {str(segment): {'kind': row['kind'], 'sha256': row['sha256'], 'source': row['source']}
                 for segment, row, value in results if row is not None}
        merged = drive_logs.merge(extracts)
        kinds = {segment: row['kind'] for segment, row, value in results if row is not None}
        found = drive_episodes.scan(route, merged, kinds)
        summary = {
            't0_ns': merged['t0'], 'span_ns': merged['span'], 'segments': merged['segments'], 'holes': merged['holes'],
            'left_out': merged['left_out'], 'meta': merged['meta'], 'civil_ms': drive_logs.civil_start(merged),
            'gps': [merged['gps'][0], merged['gps'][-1]] if merged['gps'] else None,
            'counts': found['counts'], 'coverage': found['coverage'], 'dashcam': found['dashcam'],
            'unreadable': sorted(set(by_segment) - set(extracts)), 'seen': seen,
            'scanner_version': drive_episodes.SCANNER_VERSION, 'extract_version': EXTRACT_VERSION,
        }
        self._set_episodes(route, found['episodes'])
        self._run([("UPDATE scans SET state = 'done', progress = 1, finished_at = ?, scanner_version = ?, "
                    "extract_version = ?, files = ?, summary = ? WHERE route = ?",
                    (utc_now(), drive_episodes.SCANNER_VERSION, EXTRACT_VERSION, _dumps(files), _dumps(summary),
                     route))])
        self._merged.pop(route, None)
        return self.scan_state(route)

    def _set_episodes(self, route: str, found: list[dict]) -> None:
        """The episodes of a new scan, matched to the ones of earlier scans (same channel, overlapping): a match
        keeps its id; an old one not found again stays, inactive."""
        old = [dict(row) for row in self._rows('SELECT id, start_ns, end_ns, channel, active FROM episodes '
                                               'WHERE route = ?', (route,))]
        used: set[str] = set()
        now = utc_now()
        statements = []
        for episode in found:
            family = episode['channel'] == 'spot_check'
            best, overlap = None, None
            for item in old:
                if item['id'] in used or (item['channel'] == 'spot_check') != family:
                    continue
                shared = (min(item['end_ns'], episode['end_ns']) + MATCH_NS) - (max(item['start_ns'],
                                                                                    episode['start_ns']) - MATCH_NS)
                if shared > 0 and (overlap is None or shared > overlap):
                    best, overlap = item, shared
            values = (episode['start_ns'], episode['end_ns'], episode['anchor_ns'], episode['channel'],
                      _dumps(episode['kinds']), episode['priority'], _dumps(episode), drive_episodes.SCANNER_VERSION,
                      now)
            if best is not None:
                used.add(best['id'])
                statements.append(('UPDATE episodes SET start_ns = ?, end_ns = ?, anchor_ns = ?, channel = ?, '
                                   'kinds = ?, priority = ?, payload = ?, scanner_version = ?, last_seen = ?, '
                                   'active = 1 WHERE id = ?', (*values, best['id'])))
            else:
                ident = hashlib.sha256(f'{route}|{family}|{episode["start_ns"]}'.encode()).hexdigest()[:16]
                if any(item['id'] == ident for item in old):
                    ident = uuid.uuid4().hex[:16]
                used.add(ident)
                statements.append(('INSERT INTO episodes (start_ns, end_ns, anchor_ns, channel, kinds, priority, '
                                   'payload, scanner_version, last_seen, id, route, active, first_seen) '
                                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)', (*values, ident, route, now)))
        for item in old:
            if item['id'] not in used and item['active']:
                statements.append(('UPDATE episodes SET active = 0 WHERE id = ?', (item['id'],)))
        if statements:
            self._run(statements)

    def _stale_files(self, route: str, files: dict) -> bool:
        """Any cached extract a scan read (summary files) is stale (drive_logs.stale_extract): the scan reads it again.
        Remembered per file and modification time, so the extract written again by that scan is looked at anew."""
        for segment, item in (files or {}).items():
            path = drive_logs.extract_path(self.root, route, int(segment), item['sha256'])
            try:
                mtime = path.stat().st_mtime_ns
            except OSError:
                continue
            known = self._stale.get(str(path))
            if known is None or known[0] != mtime:
                known = (mtime, drive_logs.stale_extract(drive_logs.load_extract(self.root, route, int(segment),
                                                                                  item['sha256'])))
                self._stale[str(path)] = known
            if known[1]:
                return True
        return False

    def scan_state(self, route: str) -> dict:
        """What the page says of a drive's logs and scan: state no_logs, unreadable_logs (logs found that this server
        cannot open, and no other), not_scanned (logs imported or only found on this server, not read yet), queued, scanning, failed, outdated (new logs, a newer scanner or
        extractor, or a cached extract to read again), no_episodes or done. unreadable_files: the logs found that
        cannot be opened, in any state (a drive can have some of each)."""
        files = self._rows("SELECT segment, kind, sha256, state FROM log_files WHERE route = ?", (route,))
        rows = self._rows('SELECT * FROM scans WHERE route = ?', (route,))
        scan = dict(rows[0]) if rows else None
        summary = _loads(scan['summary']) if scan else None
        errors = self.log_errors(route)
        base = {'route': route, 'files': len(files), 'segments': len({row['segment'] for row in files}),
                'kinds': sorted({row['kind'] for row in files}), 'progress': scan['progress'] if scan else 0,
                'error': scan['error'] if scan else None, 'finished_at': scan['finished_at'] if scan else None,
                'scanner_version': scan['scanner_version'] if scan else None,
                'coverage': (summary or {}).get('coverage'), 'counts': (summary or {}).get('counts'),
                'dashcam': (summary or {}).get('dashcam'), 'unreadable': (summary or {}).get('unreadable', []),
                'unreadable_files': errors}
        if scan and scan['state'] in ('queued', 'scanning', 'failed'):
            return {**base, 'state': scan['state']}
        if not files:
            # Logs on this server not imported yet (a drive just copied from the comma): to read, not missing.
            unreadable = {row['path'] for row in self._rows('SELECT path FROM log_errors WHERE route = ?', (route,))}
            if any(item['path'] not in unreadable for item in drive_logs.discover(self.root, route)):
                return {**base, 'state': 'not_scanned'}
            return {**base, 'state': 'unreadable_logs' if errors else 'no_logs'}
        if scan is None or scan['state'] != 'done':
            return {**base, 'state': 'not_scanned'}
        seen = set((summary or {}).get('seen') or ())
        newer = (scan['scanner_version'] != drive_episodes.SCANNER_VERSION
                 or scan['extract_version'] != EXTRACT_VERSION
                 or any(row['sha256'] not in seen for row in files)
                 or self._stale_files(route, _loads(scan['files'], {})))
        if newer:
            return {**base, 'state': 'outdated'}
        if (summary or {}).get('t0_ns') is None:
            return {**base, 'state': 'failed', 'error': 'None of the logs of this drive could be read.'}
        count = self._rows("SELECT COUNT(*) FROM episodes WHERE route = ? AND active = 1 AND channel != 'spot_check'",
                           (route,))[0][0]
        return {**base, 'state': 'done' if count else 'no_episodes', 'episodes': count}

    def needs_scan(self, route: str, state: dict | None = None) -> bool:
        """The one rule of every caller (Read the logs on Speed review, Sign distance, nnslr sign-reads) for a drive to
        scan now: never scanned, out of date, a scan that stopped (interrupted, an error), or logs on this server not
        imported yet, or recorded as unreadable and readable now (a real open: never the mode bits, which an NFS share
        does not honour as root). A drive whose logs were all read and none could be read is not scanned again."""
        state = state or self.scan_state(route)
        name = state['state']
        if name in ('queued', 'scanning'):
            return False
        if name in ('not_scanned', 'outdated'):
            return True
        if name == 'failed':
            rows = self._rows('SELECT state FROM scans WHERE route = ?', (route,))
            if rows and rows[0]['state'] == 'failed':
                return True
        errors = {row['path'] for row in self._rows('SELECT path FROM log_errors WHERE route = ?', (route,))}
        if any(_readable(path) for path in errors):
            return True
        if name in ('no_logs', 'unreadable_logs'):
            return any(item['path'] not in errors for item in drive_logs.discover(self.root, route))
        return False

    def summary(self, route: str) -> dict | None:
        rows = self._rows('SELECT summary, files FROM scans WHERE route = ?', (route,))
        if not rows or not rows[0]['summary']:
            return None
        summary = _loads(rows[0]['summary'], {})
        if summary.get('t0_ns') is None:
            return None
        return {**summary, 'files': _loads(rows[0]['files'], {})}

    # ---------- the joined drive ----------

    def merged(self, route: str, *, keep: bool = True) -> dict | None:
        """The drive as the last scan read it (the cached extracts of its files), joined; None before a scan. keep:
        remembered for the next call (the last few drives); False for a pass over many drives."""
        summary = self.summary(route)
        if summary is None:
            return None
        signature = _dumps(summary['files'])
        cached = self._merged.get(route)
        if cached is not None and cached[0] == signature:
            return cached[1]
        extracts = {}
        for segment, item in summary['files'].items():
            value = drive_logs.load_extract(self.root, route, int(segment), item['sha256'])
            if value is not None:
                extracts[int(segment)] = value
        merged = drive_logs.merge(extracts)
        if not keep:
            return merged
        if len(self._merged) > 4:
            self._merged.pop(next(iter(self._merged)))
        self._merged[route] = (signature, merged)
        return merged

    # ---------- episodes ----------

    def episodes(self, route: str | None = None, *, active: bool = True) -> list[dict]:
        sql = 'SELECT * FROM episodes WHERE 1 = 1'
        args: list = []
        if route is not None:
            sql += ' AND route = ?'
            args.append(route)
        if active:
            sql += ' AND active = 1'
        out = []
        for row in self._rows(sql + ' ORDER BY route, start_ns', args):
            payload = _loads(row['payload'], {})
            payload.update(id=row['id'], route=row['route'], active=bool(row['active']), first_seen=row['first_seen'],
                           last_seen=row['last_seen'], scanner_version=row['scanner_version'])
            out.append(payload)
        return out


def _readable(path: str) -> bool:
    """The file opens and reads for this process now."""
    try:
        with open(path, 'rb') as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _seconds(t, t0) -> float | None:
    return None if t is None or t0 is None else round((t - t0) / NS, 3)


TIMELINE_SERIES = ('v_ego', 'v_cruise', 'cruise_speed', 'cruise_enabled', 'v_target', 'a_target', 'gas', 'brake',
                   'steer_pressed', 'sd_state', 'sd_enabled', 'long_active', 'lat_active', 'mads_active', 'alert', 'icbm_send',
                   'res_limit', 'res_source', 'res_final', 'sla_state', 'vis_limit', 'vis_model',
                   'vis_state', 'vis_reason', 'vis_raw_class', 'vis_raw_speed', 'vis_raw_conf', 'vis_raw_taken',
                   'vis_sign_dist', 'vis_sign_size', 'map_valid', 'map_limit', 'map_ahead', 'map_ahead_dist', 'road_name')


def timeline(store: DriveReviewStore, route: str, video: dict | None) -> dict:
    """Everything the episode view draws, on seconds from the drive's first video frame (t0): the comma's series,
    its signs and events, the sync of each segment with the video and the episodes."""
    drive_logs.check_route(route)
    state = store.scan_state(route)
    summary = store.summary(route)
    merged = store.merged(route) if summary else None
    if merged is None or merged.get('t0') is None:
        return {'route': route, 'scan': state, 't0_ns': None, 'segments': [], 'series': {}, 'events': [],
                'episodes': []}
    t0 = merged['t0']
    series = {}
    for name in TIMELINE_SERIES:
        points = merged['series'].get(name)
        if points:
            series[name] = [[_seconds(t, t0), value] for t, value in points]
    events = []
    for event in merged['events']:
        if event[1] == 'sign':
            events.append([_seconds(event[0], t0), 'sign', event[2]])
        elif event[1] == 'button' and event[3]:
            events.append([_seconds(event[0], t0), 'button', event[2]])
        elif event[1] == 'onroad':
            flagged = [[name, flags] for name, flags in event[3]
                       if set(flags) & {'userDisable', 'softDisable', 'immediateDisable', 'overrideLateral',
                                        'overrideLongitudinal', 'noEntry'}]
            if flagged:
                events.append([_seconds(event[0], t0), 'onroad', flagged])
    sync = drive_logs.sync(merged, (video or {}).get('index'))
    segments = []
    for item in sync:
        span = merged['spans'][item['segment']]
        segments.append({**item, 'log': summary['files'].get(str(item['segment'])),
                         'from_s': _seconds(span[0], t0), 'to_s': _seconds(span[1], t0)})
    episodes = []
    for episode in store.episodes(route, active=False):
        if episode['channel'] == 'spot_check':
            continue
        episodes.append({**episode, 'start_s': _seconds(episode['start_ns'], t0),
                         'end_s': _seconds(episode['end_ns'], t0), 'anchor_s': _seconds(episode['anchor_ns'], t0)})
    return {'route': route, 'scan': state, 't0_ns': t0, 'span_s': [0, _seconds(merged['span'][1], t0)],
            'civil_ms': summary.get('civil_ms'), 'meta': merged['meta'], 'holes': merged['holes'],
            'left_out': merged['left_out'], 'segments': segments, 'series': series, 'events': events,
            'episodes': episodes,
            'gps': [[_seconds(t, t0), lat, lon] for t, lat, lon, _unix in merged['gps'][::5]]}


def inference_lane(root: Path, route: str) -> dict | None:
    """The speed signs our networks found on the server in the frames of the drive (its latest pre-annotation
    report, one frame a second): new inference made on the server, with the models that made it. None without one."""
    from nnslr_tools.preannotate import PreannotationError
    from nnslr_tools.review import latest_report, report_rows
    try:
        directory = latest_report(Path(root), route)
        rows = report_rows(directory)
    except (PreannotationError, ValueError, OSError):
        return None
    model = {}
    frames = []
    for row in rows:
        model = model or row.get('model') or {}
        signs = [{'family': item.get('family'), 'value_kph': item.get('value_kph'), 'score': item.get('score')}
                 for item in row.get('detections') or ()
                 if item.get('family') in ('maximum_speed', 'zone', 'cancellation', 'variable_display')]
        if signs:
            frames.append({'segment': row.get('segment_index'), 'second': row.get('media_time_s'), 'signs': signs})
    return {'run_id': directory.name, 'cutter': model.get('cutter'), 'classifier': model.get('classifier'),
            'frames': frames}


# The fields of a box the drive player draws and lists (review/catalog.ts shortName, describe): the rest stays in Sign
# review.
SIGN_BOX_FIELDS = ('bbox_xyxy', 'family', 'applicability', 'value_kph', 'sign_unit', 'sign_type', 'painted', 'plate_hours',
                   'supplementary_panel', 'score')
# A frame added at a sign the comma read is 1000 + its frame in the minute's video (sign_reads, review/order.ts).
READ_INDEX_BASE = 1000
CAMERA_FPS = 20


def _frame_second(row: dict) -> float | None:
    """The second of its segment's video a frame of Sign review was taken at: its media time, else its frame of the
    video, else its place among the pictures at one a second; None when the report does not say."""
    for value, scale in ((row.get('media_time_s'), 1), (row.get('decoded_frame_index'), CAMERA_FPS)):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return round(value / scale, 3)
    fps = row.get('requested_sample_fps')
    index = row.get('output_index')
    if (isinstance(fps, (int, float)) and fps > 0 and isinstance(index, int) and 0 <= index < READ_INDEX_BASE):
        return round(index / fps, 3)
    return None


def _sign_box(item: dict) -> dict:
    """A box as the drive player needs it: its fields of SIGN_BOX_FIELDS, its corners to a tenth of a pixel and its score
    to three places (a drive has thousands of frames)."""
    box = {name: item[name] for name in SIGN_BOX_FIELDS if name in item}
    box['bbox_xyxy'] = [round(value, 1) if isinstance(value, float) else value for value in item['bbox_xyxy']]
    if isinstance(box.get('score'), float):
        box['score'] = round(box['score'], 3)
    return box


def sign_frames(run_id: str, rows: list[dict], reviews: list[dict]) -> dict:
    """The boxes of Sign review on every frame of a drive, for Speed review's picture, its Signs lane and its Signs
    panel (GET /api/speed-review/drives/<route>/signs): where a frame is confirmed, the boxes saved with it (none, when
    the reviewer found no sign: never the proposals then); elsewhere the model's proposals of the report. Only reading:
    the boxes are confirmed and corrected in Sign review. A frame whose second is not known is left out (it cannot be
    placed on the drive), as is a saved frame no longer in the report. A frame the site's AI reviewer saved and no
    person confirmed since (review_db.waits_for_a_person) shows its boxes, but as not confirmed (needs_person), as Sign
    review counts it."""
    from nnslr_tools.review_db import waits_for_a_person
    saved = {review['frame_key']: review for review in reviews if review.get('confirmed')}
    frames = []
    for row in rows:
        second = _frame_second(row)
        if second is None or not isinstance(row.get('segment_index'), int):
            continue
        review = saved.get(row.get('frame_key'))
        boxes = (review['payload'].get('detections') if review is not None else row.get('detections')) or ()
        frames.append({
            'frame_key': row['frame_key'], 'segment': row['segment_index'], 'second': second,
            'width': row.get('width'), 'height': row.get('height'),
            'confirmed': review is not None and not waits_for_a_person(review),
            'needs_person': review is not None and waits_for_a_person(review),
            'reviewer': review.get('reviewer') if review is not None else None,
            'reviewer_kind': review.get('reviewer_kind') if review is not None else None,
            'boxes': [_sign_box(item) for item in boxes if isinstance(item, dict)
                      and isinstance(item.get('bbox_xyxy'), (list, tuple)) and len(item['bbox_xyxy']) == 4],
        })
    frames.sort(key=lambda item: (item['segment'], item['second']))
    return {'run_id': run_id, 'frames': frames}


def map_today_lane(root: Path, route: str) -> dict | None:
    """The road type and the OpenStreetMap limit of every frame as the map says today (nnslr road-types): data
    reconstructed now, not what the comma knew while driving. Runs of [segment, from second, to second, road type,
    maxspeed]."""
    from nnslr_tools.road_types import ROAD_TYPES_RELATIVE
    path = Path(root) / ROAD_TYPES_RELATIVE / (route + '.jsonl')
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    runs: list[list] = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        segment, second = row.get('segment_index'), row.get('output_index')
        if segment is None or second is None:
            continue
        value = (row.get('road_type'), row.get('maxspeed'))
        last = runs[-1] if runs else None
        if last and last[0] == segment and last[2] == second - 1 and (last[3], last[4]) == value:
            last[2] = second
        else:
            runs.append([segment, second, second, *value])
    return {'source': 'OpenStreetMap, as read on the server (not what the comma knew)', 'runs': runs}


def all_routes(root: Path) -> list[str]:
    """The drives the portal has: their folder of the comma's files, or a copy in the shared folder."""
    found = set()
    base = Path(root) / 'raw' / 'routes'
    if base.is_dir():
        found.update(entry.name for entry in base.iterdir() if drive_logs.ROUTE.fullmatch(entry.name))
    shared = drive_logs.videos_dir()
    if shared.is_dir():
        for entry in shared.iterdir():
            name = entry.name.rsplit('--', 1)[0]
            if drive_logs.ROUTE.fullmatch(name):
                found.add(name)
    return sorted(found)


def _when(civil_ms: int | None, drive: dict) -> tuple[str | None, bool]:
    """The date and time a drive started: the GPS's (certain) when the logs have a fix, else the file times of its
    videos (drives.drive_of: uncertain)."""
    if civil_ms:
        from nnslr_tools.drives import LOCAL
        local = datetime.fromtimestamp(civil_ms / 1000, timezone.utc).astimezone(LOCAL)
        return local.strftime('%a %-d %b %Y · %H:%M'), True
    return drive.get('when'), False


def drive_day(civil_ms: int | None, drive: dict) -> str | None:
    """The local day (YYYY-MM-DD) a drive started: the GPS's, else its videos' file times; None when unknown."""
    if civil_ms:
        from nnslr_tools.drives import LOCAL
        return datetime.fromtimestamp(civil_ms / 1000, timezone.utc).astimezone(LOCAL).date().isoformat()
    return (drive.get('start') or '')[:10] or None


def before_review(day: str | None, review_from: str) -> bool:
    """A drive before the first day reviewed; a drive whose day is unknown is reviewed."""
    return bool(review_from and day and day < review_from)


def early_routes(store: 'DriveReviewStore') -> set[str]:
    """The drives before the first day reviewed (Read the logs skips them)."""
    from nnslr_tools.drives import drive_of
    review_from = store.review_from()
    if not review_from:
        return set()
    out = set()
    for route in all_routes(store.root):
        summary = store.summary(route) or {}
        if before_review(drive_day(summary.get('civil_ms'), drive_of(store.root, route)), review_from):
            out.add(route)
    return out


def _title(root: Path, ends: list) -> str | None:
    """"Olten → Zürich" from the first and the last GPS fix of the logs (drives.place_name)."""
    from nnslr_tools.drives import place_name
    origin = place_name(root, ends[0][1], ends[0][2])
    destination = place_name(root, ends[1][1], ends[1][2])
    if origin and destination:
        return 'around ' + origin if origin == destination else origin + ' → ' + destination
    return origin or destination


_stores: dict[str, DriveReviewStore] = {}
_stores_lock = threading.Lock()


def store_for(root: Path) -> DriveReviewStore:
    """The store of a data root, one per process (its scan worker included)."""
    key = str(Path(root).resolve())
    with _stores_lock:
        if key not in _stores:
            _stores[key] = DriveReviewStore(Path(root))
        return _stores[key]


def drives_view(store: DriveReviewStore) -> list[dict]:
    """Every drive of the portal (the rows of Speed review's list): its name, when, its logs and scan, whether Read
    the logs reads it (needs_scan, the store's one rule), how many episodes it has."""
    with store.reading():
        return _drives_view(store)


def _drives_view(store: DriveReviewStore) -> list[dict]:
    from nnslr_tools import drive_video
    from nnslr_tools.drives import drive_of
    episodes = store.episodes()
    review_from = store.review_from()
    out = []
    for route in all_routes(store.root):
        drive = drive_of(store.root, route)
        summary = store.summary(route) or {}
        when, certain = _when(summary.get('civil_ms'), drive)
        day = drive_day(summary.get('civil_ms'), drive)
        if not drive.get('title') and summary.get('gps'):
            drive['title'] = _title(store.root, summary['gps'])
        state = store.scan_state(route)
        found = (len(drive_logs.discover(store.root, route)) if state['state'] in ('no_logs', 'unreadable_logs')
                 else None)
        out.append({'route': route, 'title': drive.get('title'), 'when': when, 'when_certain': certain, 'day': day,
                    'before_review': before_review(day, review_from),
                    'minutes': drive.get('minutes'), 'scan': state, 'logs_found': found,
                    'needs_scan': store.needs_scan(route, state),
                    'video_segments': len(drive_video.minutes(store.root, route)),
                    'episodes': sum(1 for item in episodes
                                    if item['route'] == route and item['channel'] != 'spot_check')})
    out.sort(key=lambda item: item['route'], reverse=True)
    return out
