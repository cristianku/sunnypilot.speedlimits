"""Review database of the web review: one SQLite file under the data root.

Standard library only. The review server writes here and nowhere else; the
pre-annotation reports stay read-only. A confirmed frame is stored only after it
passed the same checks as ``nnslr import-annotations`` (the shared functions in
:mod:`nnslr_tools.annotations`), and every save is also appended to a history
table that cannot be updated or deleted.

``nnslr import-annotations --from-db`` turns the confirmed frames back into the
review JSONL lines the page export would write and imports them through the
normal import path.

Every review records who made it: ``reviewer_kind`` is ``human`` or ``ai`` and
``channel`` is ``web`` (the review page), ``rest`` (``/api/v1``) or ``mcp``. A
frame confirmed by an AI keeps ``label_basis = 'ai_review'``, ``reviewer_kind``
and ``review_channel`` in its payload; :func:`import_from_db` imports it like a
human review, and the dataset rows keep that provenance. Only confirmed frames
are imported: withdrawn reviews and unreviewed proposals never are.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from nnslr_tools.annotations import (
    AI_REVIEWER_PREFIX,
    LABEL_KEYS,
    REVIEW_CHANNELS,
    frame_identity,
    import_review_sources,
    jsonl_bytes,
    load_jsonl,
    require,
    reviewed_frame_annotations,
    validate_dataset,
)
from nnslr_tools.carry_over import unchecked_boxes
from nnslr_tools.sign_types import painted_problem, sign_type_problem
from nnslr_tools.manifest import parse_route_id
from speed_vision_core.types import MAX_VALUE_KPH, MIN_VALUE_KPH

# Version 2 added reviewer_kind and channel to frame_reviews and frame_review_history.
SCHEMA_VERSION = 2
DB_RELATIVE_PATH = Path('annotations') / 'review.sqlite'

# Fields the reviewer can change on a box: the label fields of the import plus
# the notes. Everything else on a proposal (score, model label, class, evidence)
# comes from the read-only report on the server.
EDITABLE_DETECTION_KEYS = LABEL_KEYS + ('notes', 'sign_type', 'painted')

# Every box of a saved frame is checked against the original image (hash and
# PNG decode), so the number of boxes in one request is bounded.
MAX_DETECTIONS_PER_FRAME = 100

# The country of a reviewed frame: ISO 3166-1 alpha-2.
COUNTRY_CODE = re.compile(r'[A-Z]{2}')

# The reviewer name is stored with every save and printed on the server terminal.
MAX_REVIEWER_LENGTH = 100

# The annotation id only depends on frame and box identity, never on this hash,
# so a placeholder is enough to find which reviewed frames are already imported.
PLACEHOLDER_SOURCE_HASH = '0' * 64

REVIEWER_KINDS = ('human', 'ai')
# REVIEW_CHANNELS and AI_REVIEWER_PREFIX (the stored name of an AI review is "ai/<name>",
# so it never reads as a person) come from annotations, which checks them in the dataset.
AI_LABEL_BASIS = 'ai_review'
# What the site's own AI reviewer saves (Settings › AI reviewer, Review with AI: an LLM server such as llama or strata,
# reviewer "ai/<provider>/<model>") waits for a person: the page shows it to confirm, and the import leaves it out until
# a person confirms it. The AI assistants that review through the REST API or MCP (ai/Claude Code, ai/Codex,
# ai/ChatGPT) count as confirmed, as before.
SITE_AI_REVIEWER = re.compile(r'ai/[^/]+/.+')
# The same in SQL: confirmed frames that count as confirmed leave these out.
NOT_WAITING_SQL = " AND NOT (reviewer_kind = 'ai' AND reviewer GLOB 'ai/*/*')"


def waits_for_a_person(review: dict) -> bool:
    """A confirmed frame the site's own AI reviewer saved, not a person since (SITE_AI_REVIEWER)."""
    return bool(review.get('confirmed')) and review.get('reviewer_kind') == 'ai' \
        and isinstance(review.get('reviewer'), str) and SITE_AI_REVIEWER.fullmatch(review['reviewer']) is not None

SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL,
        applied_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS frame_reviews (
        route_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        frame_key TEXT NOT NULL,
        segment_index INTEGER NOT NULL,
        output_index INTEGER NOT NULL,
        payload TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
        revision INTEGER NOT NULL,
        updated_at TEXT NOT NULL,
        reviewer_kind TEXT NOT NULL DEFAULT 'human' CHECK (reviewer_kind IN ('human', 'ai')),
        channel TEXT NOT NULL DEFAULT 'web' CHECK (channel IN ('web', 'rest', 'mcp')),
        PRIMARY KEY (route_id, run_id, frame_key)
    )""",
    """CREATE TABLE IF NOT EXISTS frame_review_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        frame_key TEXT NOT NULL,
        revision INTEGER NOT NULL,
        payload TEXT NOT NULL,
        reviewer TEXT NOT NULL,
        confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
        saved_at TEXT NOT NULL,
        client TEXT NOT NULL,
        reviewer_kind TEXT NOT NULL DEFAULT 'human' CHECK (reviewer_kind IN ('human', 'ai')),
        channel TEXT NOT NULL DEFAULT 'web' CHECK (channel IN ('web', 'rest', 'mcp'))
    )""",
    """CREATE INDEX IF NOT EXISTS frame_review_history_frame
        ON frame_review_history (route_id, run_id, frame_key)""",
    """CREATE TRIGGER IF NOT EXISTS frame_review_history_no_update
        BEFORE UPDATE ON frame_review_history
        BEGIN SELECT RAISE(ABORT, 'frame_review_history is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS frame_review_history_no_delete
        BEFORE DELETE ON frame_review_history
        BEGIN SELECT RAISE(ABORT, 'frame_review_history is append-only'); END""",
)

# Columns added by schema version 2. A version 1 file gets them with ALTER TABLE:
# its reviews were all saved by the review page as human reviews, which is what
# the defaults say.
VERSION_2_COLUMNS = (
    ('reviewer_kind', "TEXT NOT NULL DEFAULT 'human' CHECK (reviewer_kind IN ('human', 'ai'))"),
    ('channel', "TEXT NOT NULL DEFAULT 'web' CHECK (channel IN ('web', 'rest', 'mcp'))"),
)

# English text shown by the review page for each import-annotations reason code.
REASON_MESSAGES = {
    'invalid_json': 'Invalid JSON data.',
    'invalid_schema': 'Invalid review structure.',
    'unknown_enum': 'Value not allowed in one of the choice fields.',
    'unreviewed_proposal': 'The frame is not confirmed.',
    'missing_provenance': 'The provenance data of the review is missing.',
    'invalid_review_timestamp': 'Invalid review date.',
    'invalid_identity': 'Invalid identity of the frame or of a box.',
    'invalid_route_id': 'Invalid route id.',
    'invalid_dimensions': 'Invalid frame dimensions.',
    'invalid_group': 'Physical site and encounter must be empty or an identifier (not “unknown”).',
    'capture_time_inconsistency': 'Inconsistent time data of the frame.',
    'illegal_speed': f'The value must be a whole number between {MIN_VALUE_KPH} and {MAX_VALUE_KPH} km/h.',
    'value_family_mismatch': 'This type of sign cannot have a value in km/h.',
    'value_state_inconsistency': 'Value in km/h inconsistent with the chosen type.',
    'invalid_bbox': 'Invalid box: it must be inside the image and have a width and a height.',
    'invalid_score': 'Invalid proposal score.',
    'missing_image': 'The original image of the frame is not on the server.',
    'unsafe_image_path': 'Image path not allowed.',
    'absolute_path': 'Image path not allowed.',
    'remote_path': 'Image path not allowed.',
    'path_traversal': 'Image path not allowed.',
    'hash_mismatch': 'The original image changed after the extraction: rebuild the report.',
    'invalid_image': 'The original image is damaged.',
    'image_dimensions_mismatch': 'The image dimensions do not match the report.',
    'duplicate_identity': 'The same box appears twice in the frame.',
    'contradictory_frame_labels': 'The frame is marked both as empty and as having boxes.',
    'empty_dataset': 'No annotations to save.',
}

HUMAN_REVIEW_EXISTS_MESSAGE = ('This frame is already confirmed by a human reviewer: an AI reviewer does not '
                               'change or withdraw it without override_human_review=true, to use only if the '
                               'owner asked for it for this frame.')


class ReviewRejected(ValueError):
    """A save the server refuses; ``message`` is the English text for the page."""

    def __init__(self, status: int, reason: str, message: str):
        self.status = status
        self.reason = reason
        self.message = message
        super().__init__(f'{reason}: {message}')


class ReviewStoreError(ValueError):
    pass


def utc_now() -> str:
    """UTC time in the same form as the page's Date.toISOString()."""
    text = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
    return text.replace('+00:00', 'Z')


def db_path(root: Path) -> Path:
    return Path(root) / DB_RELATIVE_PATH


def italian_message(reason: str, detail: str = '') -> str:
    # validate_dataset reports "reason: detail"; keep only the detail.
    if detail.startswith(reason + ':'):
        detail = detail[len(reason) + 1:].strip()
    if reason == 'missing_provenance' and 'reviewer' in detail:
        text = 'Give a name or pseudonym of the reviewer.'
    elif reason in REASON_MESSAGES:
        text = REASON_MESSAGES[reason]
    else:
        text = 'Review not accepted.'
    if detail:
        return f'{text} (reason: {reason}; {detail})'
    return f'{text} (reason: {reason})'


class ReviewStore:
    """Current review per (route, run, frame) plus an append-only history.

    Every call opens its own SQLite connection, so the store can be shared by
    the threads of the HTTP server. Writes are serialized by a lock and run in
    an IMMEDIATE transaction; the file uses WAL so reads never wait for writes.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA busy_timeout = 30000')
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('PRAGMA journal_mode = WAL')
                connection.execute('BEGIN IMMEDIATE')
                try:
                    for statement in SCHEMA_STATEMENTS:
                        connection.execute(statement)
                    version = connection.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
                    if version is not None and version > SCHEMA_VERSION:
                        raise ReviewStoreError(f'review database {self.path} has schema version {version}; '
                                               f'this nnslr understands up to {SCHEMA_VERSION}')
                    for table in ('frame_reviews', 'frame_review_history'):
                        present = set()
                        for column in connection.execute(f'PRAGMA table_info({table})'):
                            present.add(column['name'])
                        for name, definition in VERSION_2_COLUMNS:
                            if name not in present:
                                connection.execute(f'ALTER TABLE {table} ADD COLUMN {name} {definition}')
                    if version is None or version < SCHEMA_VERSION:
                        connection.execute('INSERT INTO schema_version (version, applied_at) VALUES (?, ?)',
                                           (SCHEMA_VERSION, utc_now()))
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()

    def schema_version(self) -> int:
        connection = self._connect()
        try:
            return connection.execute('SELECT MAX(version) FROM schema_version').fetchone()[0]
        finally:
            connection.close()

    def save_review(self, *, route_id: str, run_id: str, frame_key: str, segment_index: int,
                    output_index: int, payload: dict, reviewer: str, client: str,
                    base_revision: int | None = None, reviewer_kind: str = 'human', channel: str = 'web',
                    protect_human: bool = False) -> dict:
        """Store a confirmed frame. The payload must already be validated.

        ``protect_human`` refuses (409) to change a frame a human confirmed.
        """
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        now = utc_now()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    existing = _select_review(connection, route_id, run_id, frame_key)
                    current = 0
                    if existing is not None:
                        current = existing['revision']
                    _check_base_revision(base_revision, current)
                    _check_human_review(protect_human, existing)
                    revision = current + 1
                    if current == 0:
                        connection.execute(
                            'INSERT INTO frame_reviews (route_id, run_id, frame_key, segment_index, output_index, '
                            'payload, reviewer, confirmed, revision, updated_at, reviewer_kind, channel) '
                            'VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)',
                            (route_id, run_id, frame_key, segment_index, output_index, text, reviewer, revision, now,
                             reviewer_kind, channel))
                    else:
                        connection.execute(
                            'UPDATE frame_reviews SET payload = ?, reviewer = ?, confirmed = 1, revision = ?, '
                            'updated_at = ?, reviewer_kind = ?, channel = ? '
                            'WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                            (text, reviewer, revision, now, reviewer_kind, channel, route_id, run_id, frame_key))
                    _append_history(connection, route_id, run_id, frame_key, revision, text, reviewer, 1, now, client,
                                    reviewer_kind, channel)
                    saved = _select_review(connection, route_id, run_id, frame_key)
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return saved

    def withdraw_review(self, *, route_id: str, run_id: str, frame_key: str, reviewer: str, client: str,
                        base_revision: int | None = None, reviewer_kind: str = 'human', channel: str = 'web',
                        protect_human: bool = False) -> dict:
        """Mark a saved frame as no longer confirmed; its last payload is kept."""
        now = utc_now()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    existing = _select_review(connection, route_id, run_id, frame_key)
                    if existing is None:
                        raise ReviewRejected(404, 'not_found',
                                             'No saved review to withdraw for this frame.')
                    _check_base_revision(base_revision, existing['revision'])
                    _check_human_review(protect_human, existing)
                    payload = json.dumps(existing['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False)
                    revision = existing['revision'] + 1
                    connection.execute(
                        'UPDATE frame_reviews SET reviewer = ?, confirmed = 0, revision = ?, updated_at = ?, '
                        'reviewer_kind = ?, channel = ? WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                        (reviewer, revision, now, reviewer_kind, channel, route_id, run_id, frame_key))
                    _append_history(connection, route_id, run_id, frame_key, revision, payload, reviewer, 0,
                                    now, client, reviewer_kind, channel)
                    saved = _select_review(connection, route_id, run_id, frame_key)
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return saved

    def reclassify_as_ai(self, reviewer: str, *, client: str, dry_run: bool = False) -> list[dict]:
        """Mark every current review saved under ``reviewer`` as a human review as an AI review.

        For reviews an AI saved through the review page before reviewer_kind
        existed. Each frame gets a new revision with the same boxes and review
        time: reviewer "ai/<name>", reviewer_kind "ai", label_basis "ai_review",
        recorded in the history. Returns the frames changed (or to change, with dry_run).
        """
        now = utc_now()
        changed = []
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    rows = connection.execute(
                        "SELECT * FROM frame_reviews WHERE reviewer = ? AND reviewer_kind = 'human' "
                        'ORDER BY route_id, run_id, segment_index, output_index', (reviewer,)).fetchall()
                    for row in rows:
                        review = _review_from_row(row)
                        name = _stored_reviewer(reviewer, 'ai')
                        payload = copy.deepcopy(review['payload'])
                        payload['reviewer'] = name
                        payload['label_basis'] = AI_LABEL_BASIS
                        payload['reviewer_kind'] = 'ai'
                        payload['review_channel'] = review['channel']
                        payload['reclassified_from'] = {'label_basis': review['payload'].get('label_basis'),
                                                        'reviewer': reviewer, 'at': now}
                        revision = review['revision'] + 1
                        changed.append({'route_id': review['route_id'], 'run_id': review['run_id'],
                                        'frame_key': review['frame_key'], 'confirmed': review['confirmed'],
                                        'revision': revision, 'reviewer': name})
                        if dry_run:
                            continue
                        text = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
                        connection.execute(
                            "UPDATE frame_reviews SET payload = ?, reviewer = ?, reviewer_kind = 'ai', revision = ?, "
                            'updated_at = ? WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                            (text, name, revision, now, review['route_id'], review['run_id'], review['frame_key']))
                        _append_history(connection, review['route_id'], review['run_id'], review['frame_key'],
                                        revision, text, name, 1 if review['confirmed'] else 0, now, client, 'ai',
                                        review['channel'])
                    connection.execute('ROLLBACK' if dry_run else 'COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return changed

    def withdraw_ai_reviews(self, *, client: str, dry_run: bool = False) -> list[dict]:
        """Mark every frame an AI confirmed (reviewer_kind 'ai', all runs) as no longer confirmed, as withdraw_review
        does: the AI's boxes stay as the frame's last payload, a new revision is recorded in the history, and the next
        import takes the frame out of the dataset. Frames a person confirmed are left alone. Returns the frames
        changed (or to change, with dry_run)."""
        now = utc_now()
        changed = []
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    rows = connection.execute(
                        "SELECT * FROM frame_reviews WHERE reviewer_kind = 'ai' AND confirmed = 1 "
                        'ORDER BY route_id, run_id, segment_index, output_index').fetchall()
                    for row in rows:
                        review = _review_from_row(row)
                        revision = review['revision'] + 1
                        changed.append({'route_id': review['route_id'], 'run_id': review['run_id'],
                                        'frame_key': review['frame_key'], 'reviewer': review['reviewer'],
                                        'revision': revision})
                        if dry_run:
                            continue
                        connection.execute(
                            'UPDATE frame_reviews SET confirmed = 0, revision = ?, updated_at = ? '
                            'WHERE route_id = ? AND run_id = ? AND frame_key = ?',
                            (revision, now, review['route_id'], review['run_id'], review['frame_key']))
                        _append_history(connection, review['route_id'], review['run_id'], review['frame_key'],
                                        revision, row['payload'], review['reviewer'], 0, now, client, 'ai',
                                        review['channel'])
                    connection.execute('ROLLBACK' if dry_run else 'COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return changed

    def review(self, route_id: str, run_id: str, frame_key: str) -> dict | None:
        connection = self._connect()
        try:
            return _select_review(connection, route_id, run_id, frame_key)
        finally:
            connection.close()

    def reviews(self, route_id: str, run_id: str, *, confirmed_only: bool = False,
                since: str | None = None) -> list[dict]:
        """The reviews of a route's run; with ``since`` (an updated_at) only those saved after it."""
        query = 'SELECT * FROM frame_reviews WHERE route_id = ? AND run_id = ?'
        values: tuple = (route_id, run_id)
        if confirmed_only:
            query = query + ' AND confirmed = 1'
        if since is not None:
            query = query + ' AND updated_at > ?'
            values = values + (since,)
        query = query + ' ORDER BY segment_index, output_index'
        connection = self._connect()
        try:
            rows = connection.execute(query, values).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            result.append(_review_from_row(row))
        return result

    def confirmed_count(self, route_id: str, run_id: str, *, reviewer_kind: str | None = None) -> int:
        """Confirmed frames of a run; with reviewer_kind ('human' or 'ai'), only those of that kind of reviewer. The
        frames the site's AI reviewer saved wait for a person (waiting_count) and are not counted."""
        query = 'SELECT COUNT(*) FROM frame_reviews WHERE route_id = ? AND run_id = ? AND confirmed = 1' + NOT_WAITING_SQL
        values: tuple = (route_id, run_id)
        if reviewer_kind is not None:
            query += ' AND reviewer_kind = ?'
            values = values + (reviewer_kind,)
        connection = self._connect()
        try:
            return connection.execute(query, values).fetchone()[0]
        finally:
            connection.close()

    def waiting_count(self, route_id: str, run_id: str) -> int:
        """The frames of a run the site's AI reviewer saved, waiting for a person to confirm them."""
        connection = self._connect()
        try:
            return connection.execute(
                "SELECT COUNT(*) FROM frame_reviews WHERE route_id = ? AND run_id = ? AND confirmed = 1 "
                "AND reviewer_kind = 'ai' AND reviewer GLOB 'ai/*/*'", (route_id, run_id)).fetchone()[0]
        finally:
            connection.close()

    def confirmed_count_in_other_runs(self, route_id: str, run_id: str) -> int:
        """The frames confirmed on older runs of a route only: one Propose again carried onto ``run_id``
        (carry_confirmed) is reviewed there, and counts there."""
        connection = self._connect()
        try:
            return connection.execute(
                'SELECT COUNT(DISTINCT frame_key) FROM frame_reviews WHERE route_id = ? AND run_id <> ? '
                'AND confirmed = 1 AND frame_key NOT IN '
                '(SELECT frame_key FROM frame_reviews WHERE route_id = ? AND run_id = ?)',
                (route_id, run_id, route_id, run_id)).fetchone()[0]
        finally:
            connection.close()

    def carry_confirmed(self, *, route_id: str, from_run: str, to_run: str, frame_keys, client: str) -> int:
        """Propose again on a drive in review: the frames confirmed on ``from_run`` stay confirmed on ``to_run``, its
        new report. Each is copied as it is (its boxes, who confirmed it and how, when), for the frames ``to_run``
        has (``frame_keys``) and holds no review of yet, with its line of history; the payload keeps the run it was
        reviewed on, so the dataset's provenance stays true. One transaction: every frame or none. Returns how many
        were copied."""
        keys = set(frame_keys)
        now = utc_now()
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                try:
                    copied = 0
                    rows = connection.execute(
                        'SELECT * FROM frame_reviews WHERE route_id = ? AND run_id = ? AND confirmed = 1 '
                        'ORDER BY segment_index, output_index', (route_id, from_run)).fetchall()
                    for row in rows:
                        if row['frame_key'] not in keys or _select_review(connection, route_id, to_run,
                                                                          row['frame_key']) is not None:
                            continue
                        connection.execute(
                            'INSERT INTO frame_reviews (route_id, run_id, frame_key, segment_index, output_index, '
                            'payload, reviewer, confirmed, revision, updated_at, reviewer_kind, channel) '
                            'VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)',
                            (route_id, to_run, row['frame_key'], row['segment_index'], row['output_index'],
                             row['payload'], row['reviewer'], row['updated_at'], row['reviewer_kind'], row['channel']))
                        _append_history(connection, route_id, to_run, row['frame_key'], 1, row['payload'],
                                        row['reviewer'], 1, now, client, row['reviewer_kind'], row['channel'])
                        copied = copied + 1
                    connection.execute('COMMIT')
                except BaseException:
                    connection.execute('ROLLBACK')
                    raise
            finally:
                connection.close()
        return copied

    def route_ids(self) -> list[str]:
        connection = self._connect()
        try:
            rows = connection.execute('SELECT DISTINCT route_id FROM frame_reviews ORDER BY route_id').fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            result.append(row[0])
        return result

    def history(self, route_id: str, run_id: str, frame_key: str) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                'SELECT * FROM frame_review_history WHERE route_id = ? AND run_id = ? AND frame_key = ? '
                'ORDER BY history_id', (route_id, run_id, frame_key)).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            entry = dict(row)
            entry['payload'] = json.loads(entry['payload'])
            entry['confirmed'] = entry['confirmed'] == 1
            result.append(entry)
        return result


def _select_review(connection: sqlite3.Connection, route_id: str, run_id: str, frame_key: str) -> dict | None:
    row = connection.execute(
        'SELECT * FROM frame_reviews WHERE route_id = ? AND run_id = ? AND frame_key = ?',
        (route_id, run_id, frame_key)).fetchone()
    if row is None:
        return None
    return _review_from_row(row)


def _check_base_revision(base_revision: int | None, current: int) -> None:
    if base_revision is None:
        return
    if base_revision != current:
        raise ReviewRejected(409, 'revision_conflict',
                             'This frame was saved by another page in the meantime: '
                             'reload the page to see the current version.')


def _check_human_review(protect_human: bool, existing: dict | None) -> None:
    if protect_human and existing is not None and existing['confirmed'] and existing['reviewer_kind'] == 'human':
        raise ReviewRejected(409, 'human_review_exists', HUMAN_REVIEW_EXISTS_MESSAGE)


def _append_history(connection: sqlite3.Connection, route_id: str, run_id: str, frame_key: str, revision: int,
                    payload: str, reviewer: str, confirmed: int, saved_at: str, client: str,
                    reviewer_kind: str, channel: str) -> None:
    connection.execute(
        'INSERT INTO frame_review_history (route_id, run_id, frame_key, revision, payload, reviewer, confirmed, '
        'saved_at, client, reviewer_kind, channel) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (route_id, run_id, frame_key, revision, payload, reviewer, confirmed, saved_at, client, reviewer_kind,
         channel))


def _review_from_row(row: sqlite3.Row) -> dict:
    return {
        'route_id': row['route_id'],
        'run_id': row['run_id'],
        'frame_key': row['frame_key'],
        'segment_index': row['segment_index'],
        'output_index': row['output_index'],
        'payload': json.loads(row['payload']),
        'reviewer': row['reviewer'],
        'confirmed': row['confirmed'] == 1,
        'revision': row['revision'],
        'updated_at': row['updated_at'],
        'reviewer_kind': row['reviewer_kind'],
        'channel': row['channel'],
    }


def public_review(review: dict) -> dict:
    """What the page needs to restore one saved frame."""
    payload = review['payload']
    return {
        'route_id': review['route_id'],
        'run_id': review['run_id'],
        'frame_key': review['frame_key'],
        'confirmed': review['confirmed'],
        'revision': review['revision'],
        'reviewer': review['reviewer'],
        'reviewer_kind': review['reviewer_kind'],
        'channel': review['channel'],
        'updated_at': review['updated_at'],
        'needs_person': waits_for_a_person(review),
        'review_timestamp': payload.get('review_timestamp'),
        'detections': payload.get('detections', []),
        'site_group': payload.get('site_group'),
        'encounter_id': payload.get('encounter_id'),
        # Missing in reviews saved before the country existed: the page then shows the GPS proposal.
        **({'country': payload['country']} if 'country' in payload else {}),
    }


def _reviewed_detection(item, originals: dict):
    """One reviewed box: the original proposal (or a manual box) plus the reviewer's edits."""
    if not isinstance(item, dict):
        # Left as it is: the import checks reject it with their own reason.
        return item
    detection_id = item.get('detection_id')
    if isinstance(detection_id, str) and detection_id in originals:
        detection = copy.deepcopy(originals[detection_id])
    else:
        detection = {'score': None, 'model_label': None, 'class_id': None}
    for key in EDITABLE_DETECTION_KEYS:
        if key in item:
            detection[key] = copy.deepcopy(item[key])
    if 'detection_id' in item:
        detection['detection_id'] = detection_id
    detection['review_state'] = 'reviewed'
    if detection.get('sign_value') is not None and detection.get('value_kph') is None:
        detection['value_kph'] = detection['sign_value']  # written digits, never mph converted to km/h
    if detection.get('sign_value') is None and detection.get('value_kph') is None:
        detection['value_state'] = 'unknown'
    else:
        detection['value_state'] = 'reviewed'
    return detection


def build_reviewed_frame(report_row: dict, body: dict, timestamp: str, *, reviewer_kind: str = 'human',
                         channel: str = 'web') -> dict:
    """The reviewed frame line, the same as the page's "Scarica revisioni JSONL" writes it.

    Frame identity, image, hashes and the original proposals come from the
    report on the server; the request only brings the reviewer's decisions.
    A human review is exactly the line of the page; an AI review says so in
    label_basis, reviewer_kind and review_channel.
    """
    originals = {}
    proposals = report_row.get('detections')
    if isinstance(proposals, list):
        for proposal in proposals:
            if isinstance(proposal, dict) and isinstance(proposal.get('detection_id'), str):
                originals[proposal['detection_id']] = proposal
    detections = []
    for item in body['detections']:
        detections.append(_reviewed_detection(item, originals))
    reviewer = body.get('reviewer')
    if isinstance(reviewer, str):
        reviewer = reviewer.strip()
    site_group = body.get('site_group')
    if site_group is None:
        site_group = report_row.get('site_group')
    encounter_id = body.get('encounter_id')
    if encounter_id is None:
        encounter_id = report_row.get('encounter_id')
    # The country of the frame: the reviewer's, else the one its GPS position gives (nnslr_tools.gps).
    country = body.get('country') if 'country' in body else report_row.get('country_proposed')
    if country is not None and (not isinstance(country, str) or not COUNTRY_CODE.fullmatch(country)):
        raise ReviewRejected(400, 'invalid_country',
                             'country must be a two-letter ISO 3166-1 code in capitals ("CH", "IT", "PL"), or null.')
    frame = copy.deepcopy(report_row)
    frame['country'] = country
    frame['reviewer'] = reviewer
    frame['review_timestamp'] = timestamp
    frame['site_group'] = site_group
    frame['encounter_id'] = encounter_id
    frame['kind'] = 'reviewed_frame'
    frame['ground_truth'] = False
    frame['training_ready'] = False
    frame['review_state'] = 'reviewed'
    frame['label_basis'] = 'human_review'
    if reviewer_kind == 'ai':
        frame['label_basis'] = AI_LABEL_BASIS
        frame['reviewer_kind'] = 'ai'
        frame['review_channel'] = channel
    frame['source_proposals'] = copy.deepcopy(proposals)
    frame['detections'] = detections
    return frame


def validate_reviewed_frame(frame: dict, root: Path) -> None:
    """Apply the import-annotations rules to one reviewed frame; raise ReviewRejected(400).

    An AI review goes through the same rules as a human one, with its own provenance.
    """
    try:
        source_hash = hashlib.sha256(jsonl_bytes([frame])).hexdigest()
        rows = reviewed_frame_annotations(frame, source_hash, dataset_kind='gold', name='web review')
        report = validate_dataset(rows, root, dataset_kind='gold')
    except (ValueError, OSError) as exc:
        reason = getattr(exc, 'reason', 'invalid_schema')
        detail = getattr(exc, 'detail', str(exc))
        raise ReviewRejected(400, reason, italian_message(reason, detail)) from exc
    if not report['valid']:
        error = report['errors'][0]
        raise ReviewRejected(400, error['reason'], italian_message(error['reason'], error.get('detail', '')))


def _check_reviewer_text(reviewer) -> None:
    """Refuse a reviewer name that is too long or has control or invisible characters.

    A missing name is left to the import checks (missing_provenance).
    """
    if not isinstance(reviewer, str):
        return
    name = reviewer.strip()
    if len(name) > MAX_REVIEWER_LENGTH:
        raise ReviewRejected(400, 'invalid_reviewer',
                             f'The reviewer name can have at most {MAX_REVIEWER_LENGTH} characters.')
    if not name.isprintable():
        raise ReviewRejected(400, 'invalid_reviewer',
                             'The reviewer name contains characters that are not allowed (control or invisible characters).')


def _reviewer_kind(body: dict, channel: str, require_versioning: bool) -> str:
    """'human' or 'ai'. The page may omit it (its reviews are human); /api/v1 must say it."""
    if 'reviewer_kind' not in body and not require_versioning:
        return 'human'
    kind = body.get('reviewer_kind')
    if kind not in REVIEWER_KINDS:
        raise ReviewRejected(400, 'invalid_reviewer_kind',
                             'reviewer_kind must be "human" (a person) or "ai" (Claude, Codex, other models).')
    return kind


def _stored_reviewer(reviewer, kind: str):
    """The reviewer name as stored: "ai/<name>" for an AI, never a bare name that reads as a person."""
    if not isinstance(reviewer, str):
        return reviewer
    name = reviewer.strip()
    if kind == 'ai':
        if name and not name.lower().startswith(AI_REVIEWER_PREFIX):
            name = AI_REVIEWER_PREFIX + name
    elif name.lower().startswith(AI_REVIEWER_PREFIX):
        raise ReviewRejected(400, 'invalid_reviewer',
                             'A name that starts with "ai/" means an AI reviewer: use reviewer_kind "ai".')
    _check_reviewer_text(name)
    return name


def _check_versioning(body: dict, run_id: str) -> None:
    """Clients of /api/v1 and MCP must name the report and the revision they edited."""
    if not isinstance(body.get('run_id'), str) or not body['run_id']:
        raise ReviewRejected(400, 'missing_run_id',
                             'run_id is missing: read the frame again and send the run_id of the report you looked at.')
    if body['run_id'] != run_id:
        raise ReviewRejected(409, 'stale_report',
                             'The report of this route was rebuilt: read the frame again.')
    if 'base_revision' not in body or body['base_revision'] is None:
        raise ReviewRejected(428, 'base_revision_required',
                             'base_revision is missing: send the revision read with the frame (0 if it was not saved).')


def save_frame_review(store: ReviewStore, root: Path, *, route_id: str, run_id: str, report_row: dict,
                      body, client: str, channel: str = 'web', require_versioning: bool = False) -> dict:
    """Handle one save: confirm (validated, stored) or withdraw a frame.

    The review page, /api/v1 and MCP all come here. ``channel`` records which one;
    ``require_versioning`` (external clients) makes run_id and base_revision mandatory.
    """
    if channel not in REVIEW_CHANNELS:
        raise ValueError(f'unknown review channel {channel!r}')
    # A drive Propose again is working on is read only: its report is about to be replaced.
    from nnslr_tools.reproposing import refuse_if_busy
    refuse_if_busy(root, route_id)
    if not isinstance(body, dict):
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'the request must be an object'))
    if require_versioning:
        _check_versioning(body, run_id)
    if body.get('run_id') != run_id:
        raise ReviewRejected(409, 'stale_report',
                             'The report of this route was rebuilt: reload the page.')
    confirmed = body.get('confirmed')
    if type(confirmed) is not bool:
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'confirmed must be true or false'))
    base_revision = body.get('base_revision')
    if base_revision is not None:
        if type(base_revision) is not int or base_revision < 0:
            raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'base_revision'))
    override = body.get('override_human_review', False)
    if type(override) is not bool:
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'override_human_review'))
    _check_reviewer_text(body.get('reviewer'))
    kind = _reviewer_kind(body, channel, require_versioning)
    reviewer = _stored_reviewer(body.get('reviewer'), kind)
    # An AI never changes a human confirmation unless the owner asked for it.
    protect_human = kind == 'ai' and not override
    frame_key = report_row['frame_key']
    if not confirmed:
        if not isinstance(reviewer, str) or not reviewer:
            raise ReviewRejected(400, 'missing_provenance', italian_message('missing_provenance', 'reviewer'))
        return store.withdraw_review(route_id=route_id, run_id=run_id, frame_key=frame_key,
                                     reviewer=reviewer, client=client, base_revision=base_revision,
                                     reviewer_kind=kind, channel=channel, protect_human=protect_human)
    if not isinstance(body.get('detections'), list):
        raise ReviewRejected(400, 'invalid_schema', italian_message('invalid_schema', 'detections must be a list'))
    if len(body['detections']) > MAX_DETECTIONS_PER_FRAME:
        raise ReviewRejected(400, 'too_many_boxes',
                             f'Too many boxes in the frame (at most {MAX_DETECTIONS_PER_FRAME}).')
    for number, item in enumerate(body['detections'], start=1):
        problem = sign_type_problem(item) if isinstance(item, dict) else None
        if problem is not None:
            raise ReviewRejected(400, 'invalid_sign_type', f'Box {number}: {problem}.')
        problem = painted_problem(item) if isinstance(item, dict) else None
        if problem is not None:
            raise ReviewRejected(400, 'invalid_painted', f'Box {number}: {problem}.')
    # "Apply previous" placed these boxes where a sign should be: someone must have looked at them.
    unchecked = unchecked_boxes(body['detections'])
    if unchecked:
        raise ReviewRejected(400, 'position_not_checked',
                             ('Box ' if len(unchecked) == 1 else 'Boxes ')
                             + ', '.join(str(number) for number in unchecked)
                             + (' was' if len(unchecked) == 1 else ' were') + ' carried from the previous frame to '
                             'where the sign should be: check it on this image (move it onto the sign, or remove '
                             'it if the sign is gone), then save it without check_position.')
    named = dict(body)
    named['reviewer'] = reviewer
    frame = build_reviewed_frame(report_row, named, utc_now(), reviewer_kind=kind, channel=channel)
    validate_reviewed_frame(frame, root)
    return store.save_review(route_id=route_id, run_id=run_id, frame_key=frame_key,
                             segment_index=frame['segment_index'], output_index=frame['output_index'],
                             payload=frame, reviewer=frame['reviewer'], client=client,
                             base_revision=base_revision, reviewer_kind=kind, channel=channel,
                             protect_human=protect_human)


def _comparable(row: dict) -> dict:
    """A canonical row without who/when/which file, to compare labels only."""
    result = dict(row)
    result.pop('reviewer', None)
    result.pop('review_timestamp', None)
    provenance = result.get('provenance')
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        provenance.pop('source_review_sha256', None)
        result['provenance'] = provenance
    return result


def _existing_frames(output: Path) -> dict:
    """Rows already in the dataset, grouped by frame: frame identity -> {annotation_id: row}."""
    frames = {}
    if not output.exists():
        return frames
    for row in load_jsonl(output):
        try:
            identity = frame_identity(row)
            if identity not in frames:
                frames[identity] = {}
        except (KeyError, TypeError):
            # Not a canonical row: the validation of the import reports it.
            continue
        frames[identity][row.get('annotation_id')] = row
    return frames


def _import_state(rows: list[dict], existing_frames: dict) -> str:
    """'new' (frame not in the dataset), 'imported' (same boxes and labels) or 'changed'.

    The frame decides, not the single box: a frame already in the dataset with
    a box added, removed or drawn again is 'changed', never 'new'.
    """
    identity = frame_identity(rows[0])
    old_rows = existing_frames.get(identity)
    if old_rows is None:
        return 'new'
    new_ids = set()
    for row in rows:
        new_ids.add(row['annotation_id'])
    if new_ids != set(old_rows):
        return 'changed'
    for row in rows:
        if _comparable(old_rows[row['annotation_id']]) != _comparable(row):
            return 'changed'
    return 'imported'


def _dataset_line(review: dict) -> dict:
    """The review JSONL line of one confirmed frame, as the page export writes it.

    An AI review stays an AI review: if the database row and its payload disagree,
    the line says ai_review, never human_review.
    """
    payload = review['payload']
    if review['reviewer_kind'] != 'ai' and payload.get('label_basis') != AI_LABEL_BASIS:
        return payload
    line = dict(payload)
    line['label_basis'] = AI_LABEL_BASIS
    line['reviewer_kind'] = 'ai'
    if line.get('review_channel') not in REVIEW_CHANNELS:
        line['review_channel'] = review['channel']
    return line


def _with_position(line: dict, root: Path, cache: dict) -> dict:
    """The line with gps and country: a review saved before they existed takes them from the positions of its
    route (nnslr gps), when there are any; the reviewer's country, when given, stays."""
    if 'gps' in line and 'country' in line:
        return line
    route = line.get('route_id')
    if route not in cache:
        from nnslr_tools.gps import load_positions
        try:
            cache[route] = load_positions(root, route)
        except (ValueError, OSError):
            cache[route] = {}
    key = str(line.get('segment_index')) + '/' + str(line.get('output_index'))
    position = cache[route].get(key)
    if position is None:
        # Nothing to add: the line stays exactly the one the page export writes.
        return line
    line = dict(line)
    if 'gps' not in line:
        line['gps'] = None
        if position is not None and position.get('latitude') is not None:
            line['gps'] = {name: position.get(name) for name in ('latitude', 'longitude', 'speed_mps', 'bearing_deg')}
    if 'country' not in line:
        line['country'] = position.get('country')
    return line


def _review_identity(review: dict) -> tuple | None:
    """Frame identity (annotations.frame_identity) of a stored review, from its payload."""
    payload = review['payload']
    if not isinstance(payload, dict):
        return None
    keys = ('route_id', 'segment_index', 'camera_stream', 'decoded_frame_index', 'output_index')
    return frame_identity({key: payload.get(key) for key in keys})


def _frame_summary(review: dict) -> dict:
    """How a report of the import names a frame of the review database."""
    return {'route_id': review['route_id'], 'run_id': review['run_id'], 'frame_key': review['frame_key'],
            'reviewer': review['reviewer'], 'updated_at': review['updated_at']}


def _imported_from_run(rows: dict, run_id: str) -> bool:
    """True when every dataset row of a frame came from the given pre-annotation run."""
    for row in rows.values():
        provenance = row.get('provenance')
        if not isinstance(provenance, dict) or provenance.get('run_id') != run_id:
            return False
    return True


def import_from_db(root: Path, output: Path, *, route: str | None = None, dataset_kind: str = 'gold',
                   store: ReviewStore | None = None, dry_run: bool = False) -> dict:
    """Import the confirmed frames of each route's current report from the review database.

    The frames become the same JSONL lines the page export writes and go through
    import_review_sources, so validation, archive and output files are those of
    a JSONL import. Frames confirmed by a person and by an AI reviewer (REST API
    or MCP) are both imported; AI rows keep label_basis ai_review, reviewer
    "ai/<name>" and their channel. Frames already in the dataset with the same
    boxes and labels are skipped, so running it again imports only what is new.

    The dataset follows the current review of every frame of the current reports:
    a frame already in the dataset whose review changed after the import (box
    added, removed, drawn again, relabelled, or reclassified as an AI review)
    replaces all its old rows (same frame identity) and is listed under
    changed_after_import; a frame imported and then withdrawn (confirmed = 0)
    has its rows taken out and is listed under removed_withdrawn. The rows taken
    out are archived in annotations/replaced/<sha256>.jsonl, and the validation
    runs on the whole new dataset before anything is written. Frames the current
    reports do not review again stay as they are, and so do the rows of a
    withdrawn frame that came from an older pre-annotation run of the route.

    ``dry_run`` reports what would be new, replaced and removed (validation
    included) and writes nothing.
    """
    from nnslr_tools.preannotate import PreannotationError
    from nnslr_tools.review import latest_report

    if store is None:
        store = ReviewStore(db_path(root))
    if route is not None:
        parse_route_id(route)
        routes = [route]
    else:
        routes = store.route_ids()
    warnings = []
    selected = []
    withdrawn = []
    waiting = []
    unconfirmed_out = 0
    for route_id in routes:
        try:
            run_id = latest_report(root, route_id).name
        except (PreannotationError, ValueError, OSError) as exc:
            warnings.append(f'{route_id}: no current pre-annotation report ({exc}); its reviews were not imported')
            continue
        for review in store.reviews(route_id, run_id):
            if waits_for_a_person(review):
                # Saved by the site's AI reviewer: out of the dataset until a person confirms it (and out of it again
                # when it was imported before this rule).
                waiting.append(review)
            elif review['confirmed']:
                selected.append(review)
            else:
                withdrawn.append(review)
        older = store.confirmed_count_in_other_runs(route_id, run_id)
        if older:
            warnings.append(f'{route_id}: {older} confirmed frames belong to older pre-annotation runs '
                            f'and were not imported')
    existing = _existing_frames(output)
    # Frame identities whose rows leave the dataset: replaced by a changed review, withdrawn, or waiting for a person.
    drop = set()
    removed = []
    for review in withdrawn + waiting:
        identity = _review_identity(review)
        old_rows = existing.get(identity)
        if old_rows is None:
            continue
        # A frame carried onto this run by Propose again (carry_confirmed) keeps the run it was reviewed on.
        reviewed_on = review['payload'].get('run_id') if isinstance(review['payload'], dict) else None
        if not _imported_from_run(old_rows, review['run_id']) and not (
                reviewed_on and _imported_from_run(old_rows, reviewed_on)):
            # Imported from an older run of the route: this withdrawal did not put them there.
            continue
        drop.add(identity)
        if review in withdrawn:
            removed.append(_frame_summary(review))
        else:
            unconfirmed_out = unconfirmed_out + 1
    require(bool(selected) or bool(removed) or bool(unconfirmed_out), 'empty_dataset',
            'no confirmed reviews in the review database')
    new_frames = []
    new_count = 0
    new_ai_frames = 0
    already_imported = 0
    changed = []
    positions = {}
    for review in selected:
        line = _with_position(_dataset_line(review), root, positions)
        rows = reviewed_frame_annotations(line, PLACEHOLDER_SOURCE_HASH,
                                          dataset_kind=dataset_kind, name='review database')
        state = _import_state(rows, existing)
        if state == 'imported':
            already_imported = already_imported + 1
            continue
        # New and changed frames are both written; a changed one replaces its old rows.
        new_frames.append(line)
        if state == 'new':
            new_count = new_count + 1
            if line.get('label_basis') == AI_LABEL_BASIS:
                new_ai_frames = new_ai_frames + 1
        else:
            drop.add(frame_identity(rows[0]))
            changed.append(_frame_summary(review))
    counts = {'imported_frame_count': new_count, 'imported_ai_frame_count': new_ai_frames,
              'waiting_for_a_person_frame_count': len(waiting), 'waiting_removed_frame_count': unconfirmed_out,
              'already_imported_frame_count': already_imported,
              'replaced_frame_count': len(changed), 'changed_after_import': changed,
              'removed_withdrawn_frame_count': len(removed), 'removed_withdrawn': removed}
    if not new_frames and not drop:
        return {'valid': True, 'source': 'review-db', 'dry_run': dry_run, 'imported_count': 0, 'removed_count': 0,
                **counts, 'errors': [], 'warnings': warnings, 'output': str(output)}
    sources = []
    if new_frames:
        sources.append(('review-db', jsonl_bytes(new_frames)))
    report = import_review_sources(sources, root, output, dataset_kind=dataset_kind, drop_frames=drop,
                                   dry_run=dry_run)
    report['source'] = 'review-db'
    report['dry_run'] = dry_run
    report.update(counts)
    report['warnings'] = list(report.get('warnings', [])) + warnings
    return report
