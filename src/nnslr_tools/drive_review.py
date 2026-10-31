"""The drive review (Data › Drive review): the episodes of a recorded drive where the driver intervened or the
comma's vision and the map disagreed, shown with the video and the comma's own logs, and what a person says of them.

Everything lives in ``<data root>/annotations/drive_review.sqlite`` next to the frame review's database, and nothing
here touches that one, the dataset or a training:

- log_files: every log of a drive the portal knows, by drive, segment, kind and the SHA-256 of its content, with
  where it came from and what it covers once read (drive_logs). Importing the same files again adds nothing.
- scans: the last scan of each drive: its state (queued, scanning, done, failed), progress, the scanner's and the
  extractor's versions, the files it read, what the logs cover and the date of the drive.
- episodes: what the scanner found (drive_episodes). A scan again keeps the id of an episode it finds again (the
  same channel, overlapping in time), so what a person said of it stays with it; one no longer found is kept,
  inactive.
- annotations, annotation_history: what a person said of an episode or of any stretch of a drive, each change kept.
  An annotation holds its own times on the comma's monotonic clock (not the episode's), so it outlives scans,
  new logs of the same drive and new models. Its author is a person (author_kind human); rule, model and auto are
  reserved for labels made otherwise, which never count as a person's.

The scan runs in one background thread of the site (one drive at a time, the logs read by a few child processes at
low priority): CPU only, no image, no network.
"""
from __future__ import annotations

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

VERDICTS = ('wrong_limit', 'correct_limit', 'undeterminable')
# Why the limit was right but the driver intervened anyway.
OTHER_REASONS = ('traffic', 'road_or_curve', 'weather_or_visibility', 'driver_choice', 'test', 'other')
# Why a person cannot tell.
UNDETERMINABLE_REASONS = ('video_missing', 'data_missing', 'sign_not_visible', 'sign_ambiguous',
                          'conflicting_information', 'other')
# What starts the limit of an interval.
CHANGES = ('sign', 'end_of_limit', 'scene', 'carried', 'other')
LIMIT_STATES = ('known', 'unknown')
AUTHOR_KINDS = ('human', 'rule', 'model', 'auto')
MAX_INTERVALS = 50
MAX_TEXT = 1000

SCHEMA_STATEMENTS = (
    'CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)',
    '''CREATE TABLE IF NOT EXISTS log_files (
        route TEXT NOT NULL, segment INTEGER NOT NULL, kind TEXT NOT NULL, sha256 TEXT NOT NULL,
        size INTEGER NOT NULL, source TEXT NOT NULL, path TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'new',
        error TEXT, first_ns INTEGER, last_ns INTEGER, messages INTEGER, counts TEXT, extract_version INTEGER,
        added_at TEXT NOT NULL, read_at TEXT,
        PRIMARY KEY (route, segment, kind, sha256))''',
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
    '''CREATE TABLE IF NOT EXISTS annotation_history (
        id TEXT NOT NULL, revision INTEGER NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL, author TEXT NOT NULL,
        author_kind TEXT NOT NULL, origin TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY (id, revision))''',
)


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
    """The drive review's database and its scan worker. Every call opens its own connection (the site's threads share
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

    def _rows(self, sql: str, args=()) -> list[sqlite3.Row]:
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
        {found, new, files}."""
        drive_logs.check_route(route)
        found = drive_logs.discover(self.root, route)
        known = {(row['segment'], row['kind'], row['sha256']): row for row in
                 self._rows('SELECT * FROM log_files WHERE route = ?', (route,))}
        statements, new = [], 0
        now = utc_now()
        for item in found:
            try:
                sha, size = self._sha(item['path'])
            except OSError:
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
        if statements:
            self._run(statements)
        return {'found': len(found), 'new': new, 'files': len(known)}

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

    def scan_state(self, route: str) -> dict:
        """What the page says of a drive's logs and scan: state no_logs, not_scanned, queued, scanning, failed,
        outdated (new logs, or a newer scanner), no_episodes or done."""
        files = self._rows("SELECT segment, kind, sha256, state FROM log_files WHERE route = ?", (route,))
        rows = self._rows('SELECT * FROM scans WHERE route = ?', (route,))
        scan = dict(rows[0]) if rows else None
        summary = _loads(scan['summary']) if scan else None
        base = {'route': route, 'files': len(files), 'segments': len({row['segment'] for row in files}),
                'kinds': sorted({row['kind'] for row in files}), 'progress': scan['progress'] if scan else 0,
                'error': scan['error'] if scan else None, 'finished_at': scan['finished_at'] if scan else None,
                'scanner_version': scan['scanner_version'] if scan else None,
                'coverage': (summary or {}).get('coverage'), 'counts': (summary or {}).get('counts'),
                'dashcam': (summary or {}).get('dashcam'), 'unreadable': (summary or {}).get('unreadable', [])}
        if scan and scan['state'] in ('queued', 'scanning', 'failed'):
            return {**base, 'state': scan['state']}
        if not files:
            return {**base, 'state': 'no_logs'}
        if scan is None or scan['state'] != 'done':
            return {**base, 'state': 'not_scanned'}
        seen = set((summary or {}).get('seen') or ())
        newer = (scan['scanner_version'] != drive_episodes.SCANNER_VERSION
                 or scan['extract_version'] != EXTRACT_VERSION
                 or any(row['sha256'] not in seen for row in files))
        if newer:
            return {**base, 'state': 'outdated'}
        if (summary or {}).get('t0_ns') is None:
            return {**base, 'state': 'failed', 'error': 'None of the logs of this drive could be read.'}
        count = self._rows("SELECT COUNT(*) FROM episodes WHERE route = ? AND active = 1 AND channel != 'spot_check'",
                           (route,))[0][0]
        return {**base, 'state': 'done' if count else 'no_episodes', 'episodes': count}

    def summary(self, route: str) -> dict | None:
        rows = self._rows('SELECT summary, files FROM scans WHERE route = ?', (route,))
        if not rows or not rows[0]['summary']:
            return None
        summary = _loads(rows[0]['summary'], {})
        if summary.get('t0_ns') is None:
            return None
        return {**summary, 'files': _loads(rows[0]['files'], {})}

    # ---------- the joined drive ----------

    def merged(self, route: str) -> dict | None:
        """The drive as the last scan read it (the cached extracts of its files), joined; None before a scan."""
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

    def episode(self, episode_id: str) -> dict | None:
        rows = self._rows('SELECT route FROM episodes WHERE id = ?', (episode_id,))
        if not rows:
            return None
        return next((item for item in self.episodes(rows[0]['route'], active=False) if item['id'] == episode_id),
                    None)

    # ---------- annotations ----------

    def annotations(self, route: str | None = None, *, deleted: bool = False) -> list[dict]:
        sql = 'SELECT * FROM annotations WHERE 1 = 1'
        args: list = []
        if route is not None:
            sql += ' AND route = ?'
            args.append(route)
        if not deleted:
            sql += ' AND deleted = 0'
        return [_public_annotation(row) for row in self._rows(sql + ' ORDER BY created_at', args)]

    def history(self, annotation_id: str) -> list[dict]:
        return [{**dict(row), 'payload': _loads(row['payload'], {})} for row in
                self._rows('SELECT * FROM annotation_history WHERE id = ? ORDER BY revision', (annotation_id,))]

    def save_annotation(self, route: str, body: dict, *, author: str, author_kind: str = 'human',
                        origin: str = 'portal') -> dict:
        """Creates ({route, episode_id?, verdict, ...}) or changes ({id, base_revision, ...}) an annotation. A change
        on another revision than the one read is refused (409): read it again."""
        drive_logs.check_route(route)
        if author_kind not in AUTHOR_KINDS:
            raise DriveReviewError(400, 'bad_author', 'Unknown kind of author.')
        payload = validate_annotation(body, self.summary(route))
        episode_id = body.get('episode_id')
        if episode_id is not None:
            rows = self._rows('SELECT route FROM episodes WHERE id = ?', (str(episode_id),))
            if not rows or rows[0]['route'] != route:
                raise DriveReviewError(404, 'unknown_episode', 'This episode is not of this drive.')
        payload['context'] = self._context(route, payload, episode_id, body.get('id'))
        now = utc_now()
        ident = body.get('id')
        if ident is None:
            ident = uuid.uuid4().hex
            revision = 1
            statements = [('INSERT INTO annotations (id, route, episode_id, revision, payload, author, author_kind, '
                           'origin, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)',
                           (ident, route, episode_id, _dumps(payload), author, author_kind, origin, now, now))]
            action = 'create'
        else:
            current = self._rows('SELECT * FROM annotations WHERE id = ?', (str(ident),))
            if not current or current[0]['route'] != route or current[0]['deleted']:
                raise DriveReviewError(404, 'unknown_annotation', 'This annotation does not exist (any more).')
            if body.get('base_revision') != current[0]['revision']:
                raise DriveReviewError(409, 'conflict', 'This annotation was changed meanwhile: read it again.')
            revision = current[0]['revision'] + 1
            statements = [('UPDATE annotations SET revision = ?, payload = ?, author = ?, author_kind = ?, origin = ?, '
                           'updated_at = ?, episode_id = ? WHERE id = ? AND revision = ?',
                           (revision, _dumps(payload), author, author_kind, origin, now,
                            episode_id if episode_id is not None else current[0]['episode_id'], ident,
                            current[0]['revision']))]
            action = 'update'
        statements.append(('INSERT INTO annotation_history (id, revision, action, payload, author, author_kind, '
                           'origin, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                           (ident, revision, action, _dumps(payload), author, author_kind, origin, now)))
        self._run(statements)
        return _public_annotation(self._rows('SELECT * FROM annotations WHERE id = ?', (ident,))[0])

    def delete_annotation(self, route: str, ident: str, base_revision, *, author: str, author_kind: str = 'human',
                          origin: str = 'portal') -> dict:
        current = self._rows('SELECT * FROM annotations WHERE id = ?', (str(ident),))
        if not current or current[0]['route'] != route or current[0]['deleted']:
            raise DriveReviewError(404, 'unknown_annotation', 'This annotation does not exist (any more).')
        if base_revision != current[0]['revision']:
            raise DriveReviewError(409, 'conflict', 'This annotation was changed meanwhile: read it again.')
        revision = current[0]['revision'] + 1
        now = utc_now()
        self._run([('UPDATE annotations SET deleted = 1, revision = ?, updated_at = ? WHERE id = ? AND revision = ?',
                    (revision, now, ident, current[0]['revision'])),
                   ('INSERT INTO annotation_history (id, revision, action, payload, author, author_kind, origin, at) '
                    "VALUES (?, ?, 'delete', ?, ?, ?, ?, ?)",
                    (ident, revision, current[0]['payload'], author, author_kind, origin, now))])
        return {'id': ident, 'deleted': True, 'revision': revision}

    def _context(self, route: str, payload: dict, episode_id, ident) -> dict:
        """What the comma recorded where each interval starts, and the limit a person gave just before it: kept with
        the annotation, so it keeps its place in the drive whatever a later scan finds."""
        summary = self.summary(route) or {}
        merged = self.merged(route)
        drive = drive_episodes.Drive(merged) if merged else None
        earlier = []
        for item in self.annotations(route):
            if item['id'] == ident or item['author_kind'] != 'human':
                continue
            for interval in item['payload'].get('intervals', []):
                earlier.append((interval['start_ns'], interval, item['id']))
        intervals = []
        for interval in payload.get('intervals', []):
            start = interval['start_ns']
            before = [entry for entry in earlier if entry[0] < start]
            last = max(before, key=lambda entry: entry[0]) if before else None
            intervals.append({
                'recorded': drive.state(start) if drive else None,
                'previous_vision_change': drive_episodes.previous_change(drive, start) if drive else None,
                'previous_person_limit': ({'annotation': last[2], 'start_ns': last[0], 'limit': last[1]['limit']}
                                          if last else None),
            })
        return {'episode_id': episode_id, 'scanner_version': summary.get('scanner_version'),
                'extract_version': summary.get('extract_version'), 'files': summary.get('files'),
                't0_ns': summary.get('t0_ns'), 'civil_ms': summary.get('civil_ms'), 'intervals': intervals}

    # ---------- the export ----------

    def export(self, out) -> int:
        """Writes the annotations as JSON lines (one per interval) after a header with their meaning; then a line per
        scanned drive with what its logs cover. Returns the number of intervals."""
        out.write(_dumps({'kind': 'header', 'format': 'nnslr-drive-review', 'schema': SCHEMA_VERSION,
                          'semantics': EXPORT_SEMANTICS}) + '\n')
        count = 0
        for item in self.annotations():
            payload = item['payload']
            base = {'kind': 'interval', 'route': item['route'], 'annotation': item['id'], 'revision': item['revision'],
                    'episode': item['episode_id'], 'verdict': payload['verdict'], 'reason': payload.get('reason'),
                    'undeterminable_reason': payload.get('undeterminable_reason'), 'author': item['author'],
                    'author_kind': item['author_kind'], 'origin': item['origin'], 'updated_at': item['updated_at']}
            intervals = payload.get('intervals') or []
            if not intervals:
                out.write(_dumps({**base, 'interval': None, 'use': 'excluded', 'excluded_because': 'no interval'})
                          + '\n')
                continue
            for interval in intervals:
                use, why = 'target', None
                if item['author_kind'] != 'human':
                    use, why = 'excluded', 'not a person\'s label (a rule, model or auto-acceptance only proposes)'
                elif payload['verdict'] == 'undeterminable':
                    use, why = 'excluded', 'the reviewer could not determine it'
                elif not interval.get('certain', True):
                    use, why = 'excluded', 'the reviewer marked it uncertain'
                out.write(_dumps({**base, 'interval': interval, 'use': use, 'excluded_because': why}) + '\n')
                count += 1
        for row in self._rows("SELECT route, summary, files FROM scans WHERE state = 'done' AND summary IS NOT NULL"):
            summary = _loads(row['summary'], {})
            out.write(_dumps({'kind': 'drive', 'route': row['route'], 'segments': summary.get('segments'),
                              'holes': summary.get('holes'), 'span_ns': summary.get('span_ns'),
                              'civil_ms': summary.get('civil_ms'), 'coverage': summary.get('coverage'),
                              'files': _loads(row['files'], {})}) + '\n')
        return count


EXPORT_SEMANTICS = {
    'times': 'start_ns/end_ns are the comma\'s monotonic clock (logMonoTime) of the drive; end_ns null: until the '
             'next interval or the end of the drive',
    'limit.state known': 'the legal limit a person established for the interval (value_kph)',
    'limit.state unknown': 'a person reviewed the stretch and the information, earlier history included, cannot '
                           'establish the limit ("Limite sconosciuto"); not the same as a sign missing from a frame',
    'not in the log': 'a stretch with no vision, map or state in the comma\'s logs: drive lines, coverage and holes',
    'not reviewed': 'a stretch with no interval: nothing is said of it',
    'undeterminable': 'verdict undeterminable, with its reason: excluded from certain targets',
    'sign_read_kph': 'the number on the sign that was read, which may differ from the applicable limit',
    'scene': 'the kind of road (road_types.ROAD_TYPES), never a limit; a driver\'s or cautious speed is never a limit',
    'author_kind': 'human: a person in the portal; rule, model, auto: proposals, never used as targets; '
                   'auto-acceptance needs a reliability measured on independent drives and is not enabled',
    'use': 'target only for a person\'s certain interval of a determined verdict',
}


def _public_annotation(row) -> dict:
    return {'id': row['id'], 'route': row['route'], 'episode_id': row['episode_id'], 'revision': row['revision'],
            'payload': _loads(row['payload'], {}), 'author': row['author'], 'author_kind': row['author_kind'],
            'origin': row['origin'], 'created_at': row['created_at'], 'updated_at': row['updated_at'],
            'deleted': bool(row['deleted'])}


def _text(value, name: str) -> str | None:
    if value is None or value == '':
        return None
    if not isinstance(value, str) or len(value) > MAX_TEXT:
        raise DriveReviewError(400, 'bad_' + name, f'{name} must be a text of at most {MAX_TEXT} characters.')
    return value.strip() or None


def _choice(value, choices, name: str, required: bool = False):
    if value is None and not required:
        return None
    if value not in choices:
        raise DriveReviewError(400, 'bad_' + name, f'{name} must be one of: {", ".join(choices)}.')
    return value


def _kph(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 5 <= value <= 200:
        raise DriveReviewError(400, 'bad_' + name, f'{name} must be a whole number of km/h (5-200), or null.')
    return value


def _ns(value, name: str, span) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DriveReviewError(400, 'bad_' + name, f'{name} must be a time of the drive (ns).')
    if span and span[0] is not None and not span[0] - 600 * NS <= value <= span[1] + 600 * NS:
        raise DriveReviewError(400, 'bad_' + name, f'{name} is outside the drive.')
    return value


def validate_annotation(body: dict, summary: dict | None) -> dict:
    """The payload of an annotation from a request, checked: verdict, its reasons, and the intervals of the limit
    (each with its start, its end or null, the limit known with its value or unknown, what changed it, the sign's
    number when it differs, the scene, and whether the person is certain)."""
    if not isinstance(body, dict):
        raise DriveReviewError(400, 'invalid_json', 'An object is expected.')
    from nnslr_tools.road_types import ROAD_TYPES
    verdict = _choice(body.get('verdict'), VERDICTS, 'verdict', required=True)
    reason = _choice(body.get('reason'), OTHER_REASONS, 'reason')
    undeterminable = _choice(body.get('undeterminable_reason'), UNDETERMINABLE_REASONS, 'undeterminable_reason')
    if verdict == 'undeterminable' and undeterminable is None:
        raise DriveReviewError(400, 'reason_required', 'Say why it cannot be determined.')
    if verdict != 'undeterminable':
        undeterminable = None
    span = (summary or {}).get('span_ns')
    raw = body.get('intervals') or []
    if not isinstance(raw, list) or len(raw) > MAX_INTERVALS:
        raise DriveReviewError(400, 'bad_intervals', f'intervals: a list of at most {MAX_INTERVALS}.')
    intervals = []
    for item in raw:
        if not isinstance(item, dict):
            raise DriveReviewError(400, 'bad_intervals', 'Every interval is an object.')
        start = _ns(item.get('start_ns'), 'start', span)
        end = None if item.get('end_ns') is None else _ns(item.get('end_ns'), 'end', span)
        if end is not None and end <= start:
            raise DriveReviewError(400, 'bad_end', 'An interval must end after it starts.')
        limit = item.get('limit') or {}
        state = _choice(limit.get('state') if isinstance(limit, dict) else None, LIMIT_STATES, 'limit', required=True)
        value = _kph(limit.get('value_kph'), 'limit')
        if state == 'known' and value is None:
            raise DriveReviewError(400, 'value_required', 'A known limit needs its value.')
        if state == 'unknown':
            value = None
        certain = item.get('certain', True)
        if not isinstance(certain, bool):
            raise DriveReviewError(400, 'bad_certain', 'certain must be true or false.')
        intervals.append({'start_ns': start, 'end_ns': end, 'limit': {'state': state, 'value_kph': value},
                          'change': _choice(item.get('change'), CHANGES, 'change'),
                          'sign_read_kph': _kph(item.get('sign_read_kph'), 'sign_read'),
                          'scene': _choice(item.get('scene'), ROAD_TYPES, 'scene'), 'certain': certain,
                          'note': _text(item.get('note'), 'note')})
    intervals.sort(key=lambda item: item['start_ns'])
    for first, second in zip(intervals, intervals[1:]):
        # An open interval (end null) runs until the next one starts.
        if first['end_ns'] is not None and first['end_ns'] > second['start_ns']:
            raise DriveReviewError(400, 'overlap', 'Two intervals of the same annotation overlap.')
    if verdict == 'wrong_limit' and not intervals:
        raise DriveReviewError(400, 'interval_required', 'A wrong limit needs the right one: its value, where it '
                                                         'starts and until when it holds.')
    return {'verdict': verdict, 'reason': reason if verdict == 'correct_limit' else None,
            'undeterminable_reason': undeterminable, 'note': _text(body.get('note'), 'note'), 'intervals': intervals}


# ---------- what the page shows ----------

def _seconds(t, t0) -> float | None:
    return None if t is None or t0 is None else round((t - t0) / NS, 3)


TIMELINE_SERIES = ('v_ego', 'v_cruise', 'cruise_speed', 'cruise_enabled', 'v_target', 'a_target', 'gas', 'brake',
                   'steer_pressed', 'sd_state', 'sd_enabled', 'long_active', 'lat_active', 'mads_active', 'alert', 'icbm_send',
                   'res_limit', 'res_source', 'res_final', 'sla_state', 'vis_limit', 'vis_road', 'vis_model',
                   'map_valid', 'map_limit', 'map_ahead', 'map_ahead_dist', 'road_name')


def timeline(store: DriveReviewStore, route: str, video: dict | None) -> dict:
    """Everything the episode view draws, on seconds from the drive's first video frame (t0): the comma's series,
    its signs and events, the sync of each segment with the video, the episodes and the annotations."""
    drive_logs.check_route(route)
    state = store.scan_state(route)
    summary = store.summary(route)
    merged = store.merged(route) if summary else None
    if merged is None or merged.get('t0') is None:
        return {'route': route, 'scan': state, 't0_ns': None, 'segments': [], 'series': {}, 'events': [],
                'episodes': [], 'annotations': store.annotations(route)}
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
        episodes.append({**episode, 'start_s': _seconds(episode['start_ns'], t0),
                         'end_s': _seconds(episode['end_ns'], t0), 'anchor_s': _seconds(episode['anchor_ns'], t0)})
    return {'route': route, 'scan': state, 't0_ns': t0, 'span_s': [0, _seconds(merged['span'][1], t0)],
            'civil_ms': summary.get('civil_ms'), 'meta': merged['meta'], 'holes': merged['holes'],
            'left_out': merged['left_out'], 'segments': segments, 'series': series, 'events': events,
            'episodes': episodes, 'annotations': store.annotations(route),
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


def episode_status(episode_id: str, annotations: list[dict]) -> str:
    """open (nobody said anything), reviewed, or undeterminable (the latest verdict of a person)."""
    mine = [item for item in annotations if item['episode_id'] == episode_id and item['author_kind'] == 'human']
    if not mine:
        return 'open'
    latest = max(mine, key=lambda item: item['updated_at'])
    return 'undeterminable' if latest['payload']['verdict'] == 'undeterminable' else 'reviewed'


def _when(civil_ms: int | None, drive: dict) -> tuple[str | None, bool]:
    """The date and time a drive started: the GPS's (certain) when the logs have a fix, else the file times of its
    videos (drives.drive_of: uncertain)."""
    if civil_ms:
        from nnslr_tools.drives import LOCAL
        local = datetime.fromtimestamp(civil_ms / 1000, timezone.utc).astimezone(LOCAL)
        return local.strftime('%a %-d %b %Y · %H:%M'), True
    return drive.get('when'), False


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
    """Every drive of the portal for the page: its name, when, its logs and scan, its episodes by status."""
    from nnslr_tools import drive_video
    from nnslr_tools.drives import drive_of
    annotations = store.annotations()
    episodes = store.episodes()
    out = []
    for route in all_routes(store.root):
        drive = drive_of(store.root, route)
        summary = store.summary(route) or {}
        when, certain = _when(summary.get('civil_ms'), drive)
        if not drive.get('title') and summary.get('gps'):
            drive['title'] = _title(store.root, summary['gps'])
        state = store.scan_state(route)
        found = len(drive_logs.discover(store.root, route)) if state['state'] == 'no_logs' else None
        mine = [item for item in episodes if item['route'] == route]
        statuses = {'open': 0, 'reviewed': 0, 'undeterminable': 0}
        for item in mine:
            if item['channel'] != 'spot_check':
                statuses[episode_status(item['id'], annotations)] += 1
        out.append({'route': route, 'title': drive.get('title'), 'when': when, 'when_certain': certain,
                    'minutes': drive.get('minutes'), 'scan': state, 'logs_found': found,
                    'video_segments': len(drive_video.minutes(store.root, route)),
                    'episodes': statuses, 'spot_checks': sum(1 for item in mine if item['channel'] == 'spot_check'),
                    'annotations': sum(1 for item in annotations if item['route'] == route)})
    out.sort(key=lambda item: item['route'], reverse=True)
    return out


QUEUE_KINDS = ('intervention', 'disagreement', 'both', 'spot_check', *drive_episodes.INTERVENTIONS)
QUEUE_DATA = ('vision_and_map', 'no_vision', 'no_map', 'no_video', 'qlog')
QUEUE_STATUS = ('open', 'reviewed', 'undeterminable')


def queue_view(store: DriveReviewStore, *, route: str | None = None, kind: str | None = None,
               data: str | None = None, status: str | None = None) -> list[dict]:
    """The episodes to review, the ones with more agreeing clues first: filtered by drive, kind (a channel or a
    clue), data available and status."""
    from nnslr_tools import drive_video
    annotations = store.annotations()
    videos: dict[str, set] = {}
    out = []
    for item in store.episodes(route):
        if kind and kind != item['channel'] and kind not in item['kinds']:
            continue
        if item['route'] not in videos:
            videos[item['route']] = {segment for segment, _path in drive_video.minutes(store.root, item['route'])}
        segments = item.get('segments') or [None, None]
        has_video = segments[0] is not None and all(number in videos[item['route']]
                                                    for number in range(segments[0], segments[1] + 1))
        available = item['data']
        if data == 'vision_and_map' and not (available['vision'] and available['map']):
            continue
        if data == 'no_vision' and available['vision']:
            continue
        if data == 'no_map' and available['map']:
            continue
        if data == 'no_video' and has_video:
            continue
        if data == 'qlog' and available.get('log') not in ('qlog', 'mixed'):
            continue
        current = episode_status(item['id'], annotations)
        if status and current != status:
            continue
        out.append({'id': item['id'], 'route': item['route'], 'start_ns': item['start_ns'], 'end_ns': item['end_ns'],
                    'channel': item['channel'], 'kinds': item['kinds'], 'priority': item['priority'],
                    'reasons': item['reasons'], 'hint': item['hint'], 'before': item['before'], 'data': available,
                    'video': has_video, 'status': current, 'segments': item.get('segments'),
                    'duration_s': round((item['end_ns'] - item['start_ns']) / NS, 1)})
    out.sort(key=lambda item: (item['status'] != 'open', -item['priority'], item['route'], item['start_ns']))
    return out
