"""Review operations shared by the review page, the REST API (``/api/v1``) and the MCP server.

Standard library only (Pillow, when installed, only for cropped or resized
images). The reports stay read-only; every save goes through
:func:`nnslr_tools.review_db.save_frame_review`, so the page, REST and MCP apply
the same checks and write the same SQLite database. Errors are
:class:`~nnslr_tools.review_db.ReviewRejected` with an HTTP status, a reason
code and an English message; the HTTP layer and the MCP tools only translate them.

Frames are ordered by (segment, image) as on the page timeline. Indices need
not be consecutive: "previous" and "next" are the neighbours in that order,
across segment boundaries.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import re
import threading
from pathlib import Path

from nnslr_tools.annotations import ENUMS, LABEL_KEYS
from nnslr_tools.carry_over import carry_over, unchecked_boxes
from nnslr_tools.drives import drive_of
from nnslr_tools.sign_types import SIGN_TYPES
from nnslr_tools.gps import load_positions, positions_path
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.preannotate import PreannotationError, _inside
from nnslr_tools.review_db import (AI_REVIEWER_PREFIX, MAX_DETECTIONS_PER_FRAME, MAX_REVIEWER_LENGTH,
                                   REVIEW_CHANNELS, REVIEWER_KINDS, ReviewRejected, public_review,
                                   save_frame_review)
from nnslr_tools.review_hints import frame_sizes_of, load_hints, suspicious_frames
from nnslr_tools import video_frames
from nnslr_tools.sign_reads import CAMERA_FPS, READ_INDEX_BASE, frame_order, frame_second
from speed_vision_core.types import MAX_VALUE_KPH, MIN_VALUE_KPH

GPS_FIELDS = ('latitude', 'longitude', 'speed_mps', 'bearing_deg')
FRAME_FILTERS = ('unreviewed', 'reviewed', 'suspicious', 'with_proposals', 'all')
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
MAX_CONTEXT = 10
IMAGE_VARIANTS = ('preview', 'original')
MIN_CROP_SIDE = 4
MIN_LONG_SIDE = 32
MAX_LONG_SIDE = 2048
# Families of a reviewed box (the page's "Tipo"); the dataset calls road_marking_candidate road_marking.
DETECTION_FAMILIES = ('maximum_speed', 'cancellation', 'zone', 'variable_display', 'road_marking_candidate',
                      'other_sign', 'unreadable', 'not_a_sign')
VALUE_FAMILIES = ('maximum_speed', 'cancellation', 'zone', 'variable_display', 'road_marking_candidate')
FRAME_KEY_PATTERN = re.compile(r'([0-9]{1,9})/([0-9]{1,9})')
UNKNOWN_ROUTE_MESSAGE = 'Unknown route, or route without a pre-annotation report.'


def frame_key_of(value) -> str:
    """Canonical "<segment>/<image>" of a frame key given as text; ReviewRejected(400) when malformed."""
    if isinstance(value, str):
        match = FRAME_KEY_PATTERN.fullmatch(value.strip())
        if match is not None:
            return str(int(match.group(1))) + '/' + str(int(match.group(2)))
    raise ReviewRejected(400, 'invalid_frame_key', 'Invalid frame key: use "<segment>/<image>", '
                                                   'for example "3/28".')


def _bounded(value, name: str, minimum: int, maximum: int, default: int) -> int:
    if value is None or value == '':
        return default
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if type(value) is not int or value < minimum or value > maximum:
        raise ReviewRejected(400, 'invalid_parameter', f'{name} must be an integer between {minimum} and {maximum}.')
    return value


def _choice(value, name: str, choices: tuple, default: str) -> str:
    if value is None or value == '':
        return default
    if value not in choices:
        raise ReviewRejected(400, 'invalid_parameter', f'{name} must be one of: ' + ', '.join(choices) + '.')
    return value


class RouteSnapshot:
    """One consistent view of a route: report rows in timeline order, reviews, hints, suspicious flags."""

    def __init__(self, route: str, directory: Path, rows: list[dict], by_key: dict, reviews: dict,
                 hints_by_frame: dict, hint_stats: dict, suspicious: dict):
        self.route = route
        self.directory = directory
        self.run_id = directory.name
        self.by_key = by_key
        self.reviews = reviews
        self.hints_by_frame = hints_by_frame
        self.hint_stats = hint_stats
        self.suspicious = suspicious
        ordered = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = row.get('frame_key')
            segment = row.get('segment_index')
            output = row.get('output_index')
            if not isinstance(key, str) or type(segment) is not int or type(output) is not int:
                continue
            # In time order: the frames of the comma's reads between the pictures at 1 a second.
            ordered.append((*frame_order(row), key))
        ordered.sort()
        self.order = []
        for _segment, _second, _output, key in ordered:
            self.order.append(key)
        self.position = {}
        for index in range(len(self.order)):
            self.position[self.order[index]] = index

    def status(self, key: str) -> str:
        review = self.reviews.get(key)
        if review is None:
            return 'unreviewed'
        if review['confirmed']:
            return 'confirmed'
        return 'withdrawn'

    def matches(self, key: str, frame_filter: str) -> bool:
        if frame_filter == 'all':
            return True
        if frame_filter == 'unreviewed':
            return self.status(key) != 'confirmed'
        if frame_filter == 'reviewed':
            return self.status(key) == 'confirmed'
        if frame_filter == 'suspicious':
            return key in self.suspicious
        detections = self.by_key[key].get('detections')
        return isinstance(detections, list) and len(detections) > 0

    def row(self, key: str) -> dict:
        if key not in self.position:
            raise ReviewRejected(404, 'unknown_frame', 'Frame not in the report of this route.')
        return self.by_key[key]


class ReviewService:
    """The review operations on one data root and review database, safe to share between threads."""

    def __init__(self, root: Path, store, *, log=None):
        self.root = Path(root)
        self.store = store
        self._log = log
        self._reports = {}
        self._reports_lock = threading.Lock()

    def log(self, text: str) -> None:
        if self._log is not None:
            self._log(text)

    # ---------- reports (read-only) ----------

    def report(self, route: str) -> tuple[Path, list[dict], dict]:
        """Current report of a route: directory, frame rows, rows by frame_key. Run directories never change.

        Each row also has ``gps`` (latitude, longitude, speed, bearing, or None) and ``country_proposed`` (from
        the GPS, or None) when ``nnslr gps`` has run for the route (nnslr_tools.gps); they follow that file.
        """
        from nnslr_tools.review import latest_report
        directory = latest_report(self.root, route)
        positions = positions_path(self.root, route)
        try:
            stamp = positions.stat().st_mtime_ns
        except OSError:
            stamp = None
        key = (str(directory), stamp, video_frames.manifest_stamp(self.root, route))
        with self._reports_lock:
            cached = self._reports.get(key)
        if cached is None:
            from nnslr_tools.review import report_rows
            rows = report_rows(directory)
            located = load_positions(self.root, route) if stamp is not None else {}
            by_key = {}
            for row in rows:
                if isinstance(row, dict):
                    position = located.get(row.get('frame_key'))
                    row['gps'] = None
                    row['country_proposed'] = None
                    row['country_from'] = None
                    if position is not None and position.get('latitude') is not None:
                        row['gps'] = {name: position.get(name) for name in GPS_FIELDS}
                    if position is not None:
                        row['country_proposed'] = position.get('country')
                        row['country_from'] = position.get('country_from')
                by_key[row.get('frame_key')] = row
            added = self._added_rows(route, directory, rows, by_key)
            if added:
                rows = sorted(rows + added, key=frame_order)
                for row in added:
                    by_key[row['frame_key']] = row
            cached = (rows, by_key)
            with self._reports_lock:
                # One entry per report: the rows of an older positions file go.
                for old in [item for item in self._reports if item[0] == key[0]]:
                    del self._reports[old]
                self._reports[key] = cached
        return directory, cached[0], cached[1]

    def _added_rows(self, route: str, directory: Path, rows: list[dict], by_key: dict) -> list[dict]:
        """The frames between the seconds saved in Sign review since the report was made (nnslr_tools.video_frames),
        as rows of the report: no proposals, the GPS and the country of the nearest frame of their segment."""
        from nnslr_tools.preannotate import frame_row
        added = []
        for raw in video_frames.added_rows(self.root, route):
            try:
                row = frame_row(self.root, route, raw.get('segment_index'), raw, Path(video_frames.VIDEO_MANIFEST))
            except (PreannotationError, OSError, ValueError, TypeError):
                continue
            if row['frame_key'] in by_key or not isinstance(row['media_time_s'], (int, float)):
                continue
            near = min((item for item in rows if item.get('segment_index') == row['segment_index']),
                       key=lambda item: abs(frame_second(item) - row['media_time_s']), default={})
            row.update(schema_version=1, kind='preannotated_frame', run_id=directory.name, width=raw.get('width'),
                       height=raw.get('height'),
                       preview_path=f'video/{row["segment_index"]}/{row["decoded_frame_index"]}.jpg',
                       detections=[], review_state='pending', ground_truth=False, model=None, road_model=None,
                       ocr=None, inference=None, gps=near.get('gps'), country_proposed=near.get('country_proposed'),
                       country_from='nearest_fix' if near.get('country_proposed') else None)
            added.append(row)
        return added

    def video_gop(self, route, segment: int, decoded: int) -> dict:
        """The frames of the road video around one, a GOP (nnslr_tools.video_frames), for the arrows of Sign review:
        each with its key (the report's row when it is one), their previews ready in the cache. ``frame_count``: the
        frames of the segment's video; ``seconds``: the seconds kept of a segment kept only at a car park's speed
        (None: every second)."""
        directory, rows, _by_key = self.known_report(route)
        try:
            video = video_frames.video_of(self.root, route, segment)
            first, last = video_frames.gop_of(video, decoded)
            video_frames.preview(self.root, route, segment, first)
            count = video_frames.frame_count(video)
            # Ready for a save of one of these frames (video_frames.materialize).
            video_frames.read_timings_soon(video)
        except (video_frames.VideoFrameError, OSError) as exc:
            raise ReviewRejected(404, 'no_video_frame', 'This frame of the video cannot be shown: ' + str(exc)) from exc
        seconds = video_frames.kept_seconds(self.root, route, segment)
        keys = {}
        for row in rows:
            if row.get('segment_index') == segment and type(row.get('decoded_frame_index')) is int:
                keys[row['decoded_frame_index']] = row['frame_key']
        frames = []
        for index in range(first, last + 1):
            if seconds is None or int(index / CAMERA_FPS) in seconds:
                frames.append({'decoded': index, 'media_time_s': index / CAMERA_FPS,
                               'frame_key': keys.get(index, f'{segment}/{READ_INDEX_BASE + index}')})
        return {'route_id': route, 'run_id': directory.name, 'segment_index': segment, 'frame_count': count,
                'gop': [first, last], 'seconds': seconds, 'frames': frames}

    def video_frame_row(self, route, frame_key: str, body) -> tuple[dict | None, bool]:
        """(report row, made now) of a frame between the seconds the page confirms (with its boxes, or as no sign):
        its picture taken out of the video now (nnslr_tools.video_frames.materialize) and added to the report. (None, False): not such a
        frame (another unknown key, or a frame that is already a row under another key)."""
        segment, index = (int(part) for part in frame_key.split('/'))
        if index < READ_INDEX_BASE:
            return None, False
        _directory, rows, _by_key = self.known_report(route)
        decoded = index - READ_INDEX_BASE
        if any(row.get('segment_index') == segment and row.get('decoded_frame_index') == decoded for row in rows):
            return None, False
        if not isinstance(body, dict) or body.get('confirmed') is not True:
            return None, False
        from nnslr_tools.reproposing import refuse_if_busy
        refuse_if_busy(self.root, route)
        try:
            _row, fresh = video_frames.materialize(self.root, route, segment, decoded)
        except (video_frames.VideoFrameError, OSError, ValueError) as exc:
            raise ReviewRejected(404, 'no_video_frame', 'This frame of the video cannot be saved: ' + str(exc)) from exc
        return self.report(route)[2].get(frame_key), fresh

    def video_stand_in(self, snapshot: RouteSnapshot, key: str) -> dict:
        """A frame of the video between the seconds not saved yet (video_gop gives its key), as Apply previous and Fit
        box see it before it is a row of the report: its place and time, the size of the nearest picture of its
        segment, no proposals. 404 for any other key."""
        segment, index = (int(part) for part in key.split('/'))
        decoded = index - READ_INDEX_BASE
        rows = [snapshot.by_key[other] for other in snapshot.order
                if snapshot.by_key[other]['segment_index'] == segment]
        valid = index > READ_INDEX_BASE and bool(rows) \
            and not any(row.get('decoded_frame_index') == decoded for row in rows)
        if valid:
            try:
                video_frames.gop_of(video_frames.video_of(self.root, snapshot.route, segment), decoded)
                seconds = video_frames.kept_seconds(self.root, snapshot.route, segment)
                valid = seconds is None or int(decoded / CAMERA_FPS) in seconds
            except (video_frames.VideoFrameError, OSError):
                valid = False
        if not valid:
            raise ReviewRejected(404, 'unknown_frame', 'Frame not in the report of this route.')
        second = decoded / CAMERA_FPS
        near = min(rows, key=lambda row: abs(frame_second(row) - second))
        return {'frame_key': key, 'segment_index': segment, 'output_index': index, 'decoded_frame_index': decoded,
                'media_time_s': second, 'width': near.get('width'), 'height': near.get('height'), 'detections': []}

    def video_frame(self, route, frame_key) -> tuple[dict, str, Path]:
        """(video_stand_in, run id, its picture: the preview JPEG of the video cache, the size of the report's pictures)
        of a frame between the seconds not saved yet, for Fit box."""
        key = frame_key_of(frame_key)
        snapshot = self.snapshot(route)
        row = self.video_stand_in(snapshot, key)
        try:
            path = video_frames.preview(self.root, snapshot.route, row['segment_index'], row['decoded_frame_index'])
        except (video_frames.VideoFrameError, OSError) as exc:
            raise ReviewRejected(404, 'image_not_found', 'This frame of the video cannot be shown: ' + str(exc)) from exc
        return row, snapshot.run_id, path

    def car_reading(self, route, frame_key) -> dict:
        """What the car's networks read on a frame (nnslr_tools.car_reading), for the button "Car networks" of Sign
        review: a row of the report or a frame of the video between the seconds not saved yet (video_stand_in), its
        boxes in the pixels of the frame's picture. Nothing is saved. 404 for a frame that is not one of a road video
        here, 503 when the sunnypilot clone has no release of the networks."""
        from nnslr_tools import car_reading
        key = frame_key_of(frame_key)
        snapshot = self.snapshot(route)
        row = snapshot.by_key.get(key) or self.video_stand_in(snapshot, key)
        segment, decoded = row.get('segment_index'), row.get('decoded_frame_index')
        if type(segment) is not int or type(decoded) is not int:
            raise ReviewRejected(404, 'no_video_frame',
                                 "This frame has no place in a road video: the car's networks read only those.")
        try:
            found = car_reading.read_frame(self.root, snapshot.route, segment, decoded, width=row.get('width'),
                                           height=row.get('height'))
        except (video_frames.VideoFrameError, OSError) as exc:
            raise ReviewRejected(404, 'no_video_frame',
                                 "The car's networks cannot read this frame: " + str(exc)) from exc
        except car_reading.CarReadingError as exc:
            raise ReviewRejected(503, 'no_car_networks', str(exc)) from exc
        return {'route_id': snapshot.route, 'run_id': snapshot.run_id, 'frame_key': key, **found}

    def videos(self, route: str, rows: list[dict]) -> dict[str, int | None]:
        """{segment: frames of its road video} for the segments that have one: Sign review counts the frames of each
        second (a cell of its timeline is confirmed when they all are). Counted only where a frame between the seconds
        was saved, the only segments a cell can be complete in; None elsewhere (the page learns the count as it goes
        through the video)."""
        saved = set()
        segments = set()
        for row in rows:
            if type(row.get('segment_index')) is int:
                segments.add(row['segment_index'])
                if str(row.get('preview_path') or '').startswith('video/'):
                    saved.add(row['segment_index'])
        found = {}
        for segment in sorted(segments):
            try:
                video = video_frames.video_of(self.root, route, segment)
                found[str(segment)] = video_frames.frame_count(video) if segment in saved else None
            except (video_frames.VideoFrameError, OSError):
                continue
        return found

    def forget_video_frame(self, route, frame_key: str) -> None:
        """Undo video_frame_row when the save it was for was refused."""
        segment, index = (int(part) for part in frame_key.split('/'))
        video_frames.forget(self.root, route, segment, index - READ_INDEX_BASE)

    def hints(self, route: str, rows: list[dict]) -> tuple[dict, dict]:
        """Second-opinion hints of a route, read again at every call (the files may change)."""
        return load_hints(self.root, route, frame_sizes_of(rows))

    def suspicious(self, route: str, run_id: str, rows: list[dict], hints_by_frame: dict) -> dict:
        """Suspicious frames of a route given the reviews confirmed so far."""
        confirmed = {}
        for review in self.store.reviews(route, run_id, confirmed_only=True):
            confirmed[review['frame_key']] = review['payload'].get('detections', [])
        return suspicious_frames(rows, hints_by_frame, confirmed, hints_available=bool(hints_by_frame))

    def known_report(self, route) -> tuple[Path, list[dict], dict]:
        """report() for a route named by a client: every failure is the same 404."""
        try:
            if not isinstance(route, str):
                raise ValueError('route id must be text')
            parse_route_id(route)
            return self.report(route)
        except (PreannotationError, ValueError, OSError) as exc:
            raise ReviewRejected(404, 'unknown_route', UNKNOWN_ROUTE_MESSAGE) from exc

    def snapshot(self, route) -> RouteSnapshot:
        directory, rows, by_key = self.known_report(route)
        reviews = {}
        confirmed = {}
        for review in self.store.reviews(route, directory.name):
            reviews[review['frame_key']] = review
            if review['confirmed']:
                confirmed[review['frame_key']] = review['payload'].get('detections', [])
        hints_by_frame, hint_stats = self.hints(route, rows)
        suspicious = suspicious_frames(rows, hints_by_frame, confirmed, hints_available=bool(hints_by_frame))
        return RouteSnapshot(route, directory, rows, by_key, reviews, hints_by_frame, hint_stats, suspicious)

    def route_ids(self) -> list[str]:
        from nnslr_tools.review import list_reports
        routes = []
        for route, directory in list_reports(self.root):
            routes.append(route)
        return routes

    # ---------- routes and progress ----------

    def route_summary(self, snapshot: RouteSnapshot) -> dict:
        confirmed = 0
        by_ai = 0
        withdrawn = 0
        with_proposals = 0
        for key in snapshot.order:
            status = snapshot.status(key)
            if status == 'confirmed':
                confirmed = confirmed + 1
                if snapshot.reviews[key]['reviewer_kind'] == 'ai':
                    by_ai = by_ai + 1
            elif status == 'withdrawn':
                withdrawn = withdrawn + 1
            if snapshot.matches(key, 'with_proposals'):
                with_proposals = with_proposals + 1
        total = len(snapshot.order)
        return {
            'route_id': snapshot.route,
            'run_id': snapshot.run_id,
            'frame_count': total,
            'confirmed_count': confirmed,
            'confirmed_by_human': confirmed - by_ai,
            'confirmed_by_ai': by_ai,
            'missing_count': total - confirmed,
            'withdrawn_count': withdrawn,
            'suspicious_count': len(snapshot.suspicious),
            'frames_with_proposals': with_proposals,
            'hint_frames': snapshot.hint_stats['frames'],
            # The route as the drive it was, as the route list names it: title ("Herznach → Zürich"), when,
            # minutes, start, from, to (nnslr_tools.drives; None for what cannot be known).
            'drive': drive_of(self.root, snapshot.route),
        }

    def list_routes(self) -> list[dict]:
        summaries = []
        for route in self.route_ids():
            try:
                summaries.append(self.route_summary(self.snapshot(route)))
            except ReviewRejected:
                continue
        return summaries

    def progress(self, route, *, suspicious_limit: int = 100) -> dict:
        """Counts of a route plus the suspicious frames and the confirmed frames with doubts."""
        snapshot = self.snapshot(route)
        result = self.route_summary(snapshot)
        suspicious_limit = _bounded(suspicious_limit, 'suspicious_limit', 0, 1000, 100)
        listed = []
        for key in snapshot.order:
            if key in snapshot.suspicious and len(listed) < suspicious_limit:
                listed.append({'frame_key': key, 'reasons': snapshot.suspicious[key]})
        doubts = {'unknown_value': [], 'unreadable': [], 'unresolved_road': []}
        for key in snapshot.order:
            if snapshot.status(key) != 'confirmed':
                continue
            flags = set()
            for detection in snapshot.reviews[key]['payload'].get('detections', []):
                if not isinstance(detection, dict):
                    continue
                family = detection.get('family')
                if family in ('maximum_speed', 'zone', 'variable_display', 'road_marking_candidate') \
                        and detection.get('value_kph') is None and detection.get('sign_value') is None:
                    flags.add('unknown_value')
                if family == 'unreadable':
                    flags.add('unreadable')
                if family in ('maximum_speed', 'zone', 'variable_display') \
                        and detection.get('applicability', 'unresolved') == 'unresolved':
                    flags.add('unresolved_road')
            for flag in flags:
                doubts[flag].append(key)
        result['suspicious'] = listed
        result['suspicious_truncated'] = len(snapshot.suspicious) > len(listed)
        result['confirmed_with_doubts'] = doubts
        result['first_unreviewed'] = None
        for key in snapshot.order:
            if snapshot.matches(key, 'unreviewed'):
                result['first_unreviewed'] = key
                break
        return result

    # ---------- frames ----------

    def frame_summary(self, snapshot: RouteSnapshot, key: str) -> dict:
        row = snapshot.by_key[key]
        review = snapshot.reviews.get(key)
        detections = row.get('detections')
        proposals = 0
        if isinstance(detections, list):
            proposals = len(detections)
        summary = {
            'frame_key': key,
            'segment_index': row['segment_index'],
            'output_index': row['output_index'],
            'position': snapshot.position[key],
            'status': snapshot.status(key),
            'revision': 0,
            'reviewer': None,
            'reviewer_kind': None,
            'proposal_count': proposals,
            'second_reading_count': len(snapshot.hints_by_frame.get(key, [])),
            'suspicious': key in snapshot.suspicious,
        }
        if review is not None:
            summary['revision'] = review['revision']
            summary['reviewer'] = review['reviewer']
            summary['reviewer_kind'] = review['reviewer_kind']
            if review['confirmed']:
                summary['saved_box_count'] = len(review['payload'].get('detections', []))
        return summary

    def _cursor(self, snapshot: RouteSnapshot, frame_filter: str, key: str) -> str:
        data = json.dumps({'run_id': snapshot.run_id, 'filter': frame_filter, 'after': key},
                          separators=(',', ':')).encode('utf-8')
        return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')

    def _after_cursor(self, snapshot: RouteSnapshot, frame_filter: str, cursor: str) -> int:
        try:
            padded = cursor + '=' * (-len(cursor) % 4)
            data = json.loads(base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8'))
            run_id, saved_filter, key = data['run_id'], data['filter'], data['after']
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise ReviewRejected(400, 'invalid_cursor', 'Invalid cursor: request the first page again.') from exc
        if run_id != snapshot.run_id:
            raise ReviewRejected(409, 'stale_report', 'The report of this route was rebuilt: '
                                                      'request the first page again.')
        if saved_filter != frame_filter or key not in snapshot.position:
            raise ReviewRejected(400, 'invalid_cursor', 'Invalid cursor for this filter.')
        return snapshot.position[key]

    def list_frames(self, route, *, frame_filter=None, cursor=None, limit=None) -> dict:
        """One page of frames matching a filter, in timeline order. next_cursor is None on the last page."""
        frame_filter = _choice(frame_filter, 'filter', FRAME_FILTERS, 'unreviewed')
        limit = _bounded(limit, 'limit', 1, MAX_PAGE_SIZE, DEFAULT_PAGE_SIZE)
        snapshot = self.snapshot(route)
        start = 0
        if cursor:
            start = self._after_cursor(snapshot, frame_filter, cursor) + 1
        matching = 0
        frames = []
        next_cursor = None
        for index in range(len(snapshot.order)):
            key = snapshot.order[index]
            if not snapshot.matches(key, frame_filter):
                continue
            matching = matching + 1
            if index < start:
                continue
            if len(frames) == limit:
                if next_cursor is None:
                    next_cursor = self._cursor(snapshot, frame_filter, frames[-1]['frame_key'])
                continue
            frames.append(self.frame_summary(snapshot, key))
        return {'route_id': snapshot.route, 'run_id': snapshot.run_id, 'filter': frame_filter, 'total': matching,
                'frames': frames, 'next_cursor': next_cursor}

    def frame_detail(self, snapshot: RouteSnapshot, key: str) -> dict:
        row = snapshot.row(key)
        detail = self.frame_summary(snapshot, key)
        metadata = {}
        for name, value in row.items():
            if name not in ('detections', 'image_path', 'preview_path'):
                metadata[name] = copy.deepcopy(value)
        review = snapshot.reviews.get(key)
        saved = None
        if review is not None:
            saved = public_review(review)
        position = snapshot.position[key]
        previous_key = None
        next_key = None
        if position > 0:
            previous_key = snapshot.order[position - 1]
        if position + 1 < len(snapshot.order):
            next_key = snapshot.order[position + 1]
        base = '/api/v1/routes/' + snapshot.route + '/frames/' + key
        proposals = row.get('detections')
        if not isinstance(proposals, list):
            proposals = []
        reasons = []
        for reason in snapshot.suspicious.get(key, []):
            reasons.append(reason['text'])
        detail.update({
            'route_id': snapshot.route,
            'run_id': snapshot.run_id,
            'frame_count': len(snapshot.order),
            'width': row.get('width'),
            'height': row.get('height'),
            'image': {'preview_url': base + '/image?variant=preview', 'original_url': base + '/image?variant=original',
                      'preview_format': 'jpeg', 'original_format': 'png',
                      'coordinates': 'bbox_xyxy = [x1, y1, x2, y2] in pixels of the original image'},
            'proposals': copy.deepcopy(proposals),
            'gps': copy.deepcopy(row.get('gps')),
            'country_proposed': row.get('country_proposed'),
            'second_reading': copy.deepcopy(snapshot.hints_by_frame.get(key, [])),
            'suspicious_reasons': reasons,
            'review': saved,
            'previous_frame_key': previous_key,
            'next_frame_key': next_key,
            'metadata': metadata,
            'save': {'run_id': snapshot.run_id, 'base_revision': detail['revision']},
        })
        return detail

    def get_frame(self, route, frame_key) -> dict:
        key = frame_key_of(frame_key)
        snapshot = self.snapshot(route)
        snapshot.row(key)
        return self.frame_detail(snapshot, key)

    def next_frame(self, route, *, frame_filter=None, after=None) -> dict:
        """The first frame matching the filter after ``after`` (or from the start), wrapping once.

        With the default filter "unreviewed" and no ``after`` this is where the
        work stopped: confirmed frames are skipped whoever confirmed them.
        """
        frame_filter = _choice(frame_filter, 'filter', FRAME_FILTERS, 'unreviewed')
        snapshot = self.snapshot(route)
        start = 0
        after_key = None
        if after:
            after_key = frame_key_of(after)
            start = snapshot.position.get(after_key, -1) + 1
            if after_key not in snapshot.position:
                snapshot.row(after_key)
        matching = []
        for key in snapshot.order:
            if snapshot.matches(key, frame_filter):
                matching.append(key)
        found = None
        wrapped = False
        for key in matching:
            if snapshot.position[key] >= start:
                found = key
                break
        if found is None and after_key is not None:
            for key in matching:
                if key != after_key:
                    found = key
                    wrapped = True
                    break
        frame = None
        if found is not None:
            frame = self.frame_detail(snapshot, found)
        return {'route_id': snapshot.route, 'run_id': snapshot.run_id, 'filter': frame_filter,
                'remaining': len(matching), 'wrapped': wrapped, 'frame': frame}

    def frame_context(self, route, frame_key, *, before=None, after=None) -> dict:
        """Up to ``before`` frames before and ``after`` frames after a frame, in timeline order."""
        key = frame_key_of(frame_key)
        before = _bounded(before, 'before', 0, MAX_CONTEXT, 2)
        after = _bounded(after, 'after', 0, MAX_CONTEXT, 2)
        snapshot = self.snapshot(route)
        snapshot.row(key)
        center = snapshot.position[key]
        segment = snapshot.by_key[key]['segment_index']
        frames = []
        for index in range(max(0, center - before), min(len(snapshot.order), center + after + 1)):
            other = snapshot.order[index]
            summary = self.frame_summary(snapshot, other)
            summary['offset'] = index - center
            summary['same_segment'] = snapshot.by_key[other]['segment_index'] == segment
            frames.append(summary)
        return {'route_id': snapshot.route, 'run_id': snapshot.run_id, 'frame_key': key,
                'frame_count': len(snapshot.order), 'frames': frames}

    def history(self, route, frame_key) -> dict:
        key = frame_key_of(frame_key)
        snapshot = self.snapshot(route)
        snapshot.row(key)
        entries = []
        for entry in self.store.history(snapshot.route, snapshot.run_id, key):
            entries.append({
                'revision': entry['revision'],
                'confirmed': entry['confirmed'],
                'reviewer': entry['reviewer'],
                'reviewer_kind': entry['reviewer_kind'],
                'channel': entry['channel'],
                'saved_at': entry['saved_at'],
                'detections': entry['payload'].get('detections', []),
            })
        return {'route_id': snapshot.route, 'run_id': snapshot.run_id, 'frame_key': key, 'history': entries}

    # ---------- images ----------

    def image_file(self, route, frame_key, variant=None) -> tuple[Path, str, dict]:
        """(file, MIME type, frame row) of the preview JPEG or of the original PNG of a frame.

        The preview is the report's ``images/NNNNNNNN.jpg`` (same checks as the
        page); the original is the extracted frame under
        ``derived/frames/<route>/``, never a symlink, with the hash of the report.
        """
        from nnslr_tools.review import public_file
        from nnslr_tools.annotations import sha256_of
        key = frame_key_of(frame_key)
        variant = _choice(variant, 'variant', IMAGE_VARIANTS, 'preview')
        directory, rows, by_key = self.known_report(route)
        row = by_key.get(key)
        if row is None:
            raise ReviewRejected(404, 'unknown_frame', 'Frame not in the report of this route.')
        if variant == 'preview':
            preview = row.get('preview_path')
            path = None
            if isinstance(preview, str) and preview.startswith('video/'):
                # A frame between the seconds saved in Sign review: its preview is in the video cache.
                try:
                    path = video_frames.preview(self.root, route, row['segment_index'], row['decoded_frame_index'])
                except (video_frames.VideoFrameError, OSError):
                    path = None
            elif isinstance(preview, str):
                path = public_file(directory, '/' + preview)
            if path is None:
                raise ReviewRejected(404, 'image_not_found', 'Frame preview not found.')
            return path, 'image/jpeg', row
        relative = row.get('image_path')
        frames = self.root / 'derived' / 'frames' / route
        try:
            if not isinstance(relative, str) or not relative.endswith('.png') or Path(relative).is_absolute() \
                    or '..' in Path(relative).parts:
                raise ValueError('image_path')
            candidate = self.root / relative
            base = _inside(self.root, frames)
            resolved = _inside(base, candidate)
            if candidate.is_symlink() or not resolved.is_file():
                raise ValueError('image_path')
        except (PreannotationError, ValueError, OSError) as exc:
            raise ReviewRejected(404, 'image_not_found', 'Original frame image not found.') from exc
        if sha256_of(resolved) != row.get('frame_sha256'):
            raise ReviewRejected(409, 'image_changed', 'The original image changed after the extraction: '
                                                       'rebuild the report.')
        return resolved, 'image/png', row

    def image(self, route, frame_key, *, variant=None, crop=None, long_side=None) -> dict:
        """Bytes of a frame image, optionally cropped (original pixel coordinates) and resized.

        Cropping and resizing need Pillow; without it only whole images are served.
        Returns {'data', 'mime_type', 'width', 'height', 'crop', 'variant', 'frame_key'}.
        """
        path, mime_type, row = self.image_file(route, frame_key, variant)
        variant = _choice(variant, 'variant', IMAGE_VARIANTS, 'preview')
        long_side = _bounded(long_side, 'long_side', MIN_LONG_SIDE, MAX_LONG_SIDE, 0) if long_side else None
        box = None
        if crop is not None:
            box = self._crop_box(crop, row)
        result = {'frame_key': frame_key_of(frame_key), 'variant': variant, 'crop': box,
                  'width': row.get('width'), 'height': row.get('height')}
        if box is None and long_side is None:
            result['data'] = path.read_bytes()
            result['mime_type'] = mime_type
            return result
        try:
            from PIL import Image
        except ImportError as exc:
            raise ReviewRejected(501, 'image_processing_unavailable',
                                 'Cropping and resizing need Pillow on the server '
                                 '(pip install Pillow): ask for the whole image.') from exc
        try:
            with Image.open(path) as opened:
                picture = opened.convert('RGB')
        except OSError as exc:
            raise ReviewRejected(409, 'invalid_image', 'The frame image cannot be read.') from exc
        scale_x = picture.width / row['width']
        scale_y = picture.height / row['height']
        if box is not None:
            picture = picture.crop((round(box[0] * scale_x), round(box[1] * scale_y),
                                    round(box[2] * scale_x), round(box[3] * scale_y)))
        if long_side is not None:
            factor = long_side / max(picture.width, picture.height)
            size = (max(1, round(picture.width * factor)), max(1, round(picture.height * factor)))
            picture = picture.resize(size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        picture.save(output, format='PNG')
        result['data'] = output.getvalue()
        result['mime_type'] = 'image/png'
        result['width'] = picture.width
        result['height'] = picture.height
        return result

    def _crop_box(self, crop, row: dict) -> list[int]:
        width = row.get('width')
        height = row.get('height')
        if isinstance(crop, str):
            parts = crop.split(',')
            try:
                crop = [int(round(float(part))) for part in parts]
            except ValueError:
                crop = None
        valid = isinstance(crop, (list, tuple)) and len(crop) == 4
        if valid:
            for value in crop:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    valid = False
        if not valid:
            raise ReviewRejected(400, 'invalid_crop', 'crop must be [x1, y1, x2, y2] in pixels of the original image.')
        x1 = max(0, min(int(round(crop[0])), width))
        y1 = max(0, min(int(round(crop[1])), height))
        x2 = max(0, min(int(round(crop[2])), width))
        y2 = max(0, min(int(round(crop[3])), height))
        if x2 - x1 < MIN_CROP_SIDE or y2 - y1 < MIN_CROP_SIDE:
            raise ReviewRejected(400, 'invalid_crop', f'The crop must be inside the image ({width}×{height}) '
                                                      f'and at least {MIN_CROP_SIDE} pixels on each side.')
        return [x1, y1, x2, y2]

    # ---------- apply previous ----------

    def apply_previous(self, route, frame_key, body=None) -> dict:
        """"Apply previous": the boxes confirmed on the frame before, carried onto this one. Saves nothing.

        ``body`` may hold ``run_id`` (checked when given) and ``detections``, the boxes of this frame as
        they are now (default: the model proposals). See :mod:`nnslr_tools.carry_over`: a box placed
        where the sign should be has ``check_position: true`` and no save accepts it until that is gone.
        A box keeps its labels only when the reviewer changed it on screen: one as the model proposed it, or as
        this frame was confirmed before (a drive reviewed again), takes those of the previous frame. A frame of the
        video between the seconds not saved yet (video_stand_in) works too: it has no proposals, every box carried
        there is to check (Fit box or "Here").
        """
        key = frame_key_of(frame_key)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ReviewRejected(400, 'invalid_schema', 'The request must be a JSON object.')
        snapshot = self.snapshot(route)
        if key in snapshot.position:
            row = snapshot.row(key)
            position = snapshot.position[key]
        else:
            # A frame of the video between the seconds, not saved yet: the frame before is the last one of the report
            # before it in time.
            row = self.video_stand_in(snapshot, key)
            place = frame_order(row)
            position = sum(1 for other in snapshot.order if frame_order(snapshot.by_key[other]) < place)
        if body.get('run_id') is not None and body.get('run_id') != snapshot.run_id:
            raise ReviewRejected(409, 'stale_report', 'The report of this route was rebuilt: read the frame again.')
        source_key = snapshot.order[position - 1] if position > 0 else None
        # The frame before in the timeline, in this segment or at the end of the one before.
        if source_key is None or snapshot.by_key[source_key]['segment_index'] < row['segment_index'] - 1:
            raise ReviewRejected(409, 'no_previous_frame', 'This frame has no frame just before it.')
        source = snapshot.reviews.get(source_key)
        if source is None or not source['confirmed']:
            raise ReviewRejected(409, 'previous_not_confirmed',
                                 f'The previous frame ({source_key}) is not confirmed: confirm it first.')
        proposals = row.get('detections')
        if not isinstance(proposals, list):
            proposals = []
        current = body.get('detections')
        if current is None:
            current = proposals
        if not isinstance(current, list) or len(current) > MAX_DETECTIONS_PER_FRAME:
            raise ReviewRejected(400, 'invalid_schema',
                                 f'detections must be a list of at most {MAX_DETECTIONS_PER_FRAME} boxes.')
        for item in current:
            if not isinstance(item, dict):
                raise ReviewRejected(400, 'invalid_schema', 'Every box must be a JSON object.')
        review = snapshot.reviews.get(key)
        # What the boxes were before the reviewer touched them: the proposals, then this frame's saved review.
        untouched = list(proposals)
        if review is not None and review['confirmed']:
            untouched = untouched + [item for item in review['payload'].get('detections', []) if isinstance(item, dict)]
        result = carry_over(source['payload'].get('detections', []), current, width=row.get('width') or 0,
                            height=row.get('height') or 0, source_key=source_key, proposals=untouched)
        to_check = unchecked_boxes(result['detections'])
        return {
            'route_id': snapshot.route,
            'run_id': snapshot.run_id,
            'frame_key': key,
            'source_frame_key': source_key,
            'source_revision': source['revision'],
            'source_reviewer_kind': source['reviewer_kind'],
            'detections': result['detections'],
            'matched': result['matched'],
            'estimated': result['estimated'],
            'dropped': result['dropped'],
            'kept': result['kept'],
            'duplicates': result['duplicates'],
            'to_check': to_check,
            'message': _carry_message(source_key, result),
            'save': {'run_id': snapshot.run_id, 'base_revision': review['revision'] if review else 0},
        }

    # ---------- saving ----------

    def save_review(self, route, frame_key, body, *, client: str, channel: str) -> dict:
        """Save one frame for an external client (REST or MCP): run_id and base_revision are mandatory.

        Boxes without detection_id get "manual-<channel>-<n>". The rest is
        save_frame_review(), the function of the review page.
        """
        if channel not in REVIEW_CHANNELS:
            raise ValueError(f'unknown review channel {channel!r}')
        key = frame_key_of(frame_key)
        directory, rows, by_key = self.known_report(route)
        report_row = by_key.get(key)
        if report_row is None:
            raise ReviewRejected(404, 'unknown_frame', 'Frame not in the report of this route.')
        if isinstance(body, dict) and isinstance(body.get('detections'), list):
            body = dict(body)
            body['detections'] = _with_detection_ids(body['detections'], channel)
        review = save_frame_review(self.store, self.root, route_id=route, run_id=directory.name,
                                   report_row=report_row, body=body, client=client, channel=channel,
                                   require_versioning=True)
        self.log_save(route, key, review)
        snapshot = self.snapshot(route)
        summary = self.route_summary(snapshot)
        following = None
        position = snapshot.position[key]
        for index in range(position + 1, len(snapshot.order)):
            if snapshot.matches(snapshot.order[index], 'unreviewed'):
                following = snapshot.order[index]
                break
        return {'ok': True, 'review': public_review(review), 'status': snapshot.status(key),
                'confirmed_count': summary['confirmed_count'], 'missing_count': summary['missing_count'],
                'next_unreviewed_frame_key': following}

    def log_save(self, route: str, key: str, review: dict) -> None:
        from nnslr_tools.review import terminal_text
        reviewer = terminal_text(review['reviewer'])
        if review['confirmed']:
            self.log(f'Salvato: {route} fotogramma {key}, revisione {review["revision"]} ({reviewer}, '
                     f'{review["reviewer_kind"]}, {review["channel"]})')
        else:
            self.log(f'Conferma annullata: {route} fotogramma {key} ({reviewer}, {review["reviewer_kind"]}, '
                     f'{review["channel"]})')


def _carry_message(source_key: str, result: dict) -> str:
    """"From 3/27: 2 signs found here …" for the page and for an agent."""
    parts = []
    if result['matched']:
        parts.append(f"{result['matched']} {'sign' if result['matched'] == 1 else 'signs'} found here and "
                     f"labelled as there")
    if result['kept']:
        parts.append(f"{result['kept']} already changed here, left as {'it is' if result['kept'] == 1 else 'they are'}")
    if result['estimated']:
        parts.append(f"{result['estimated']} placed where {'it' if result['estimated'] == 1 else 'they'} should be: "
                     f"check {'it' if result['estimated'] == 1 else 'each'} (move it onto the sign, or remove it "
                     f"if it is gone)")
    if result['dropped']:
        parts.append(f"{result['dropped']} left the picture")
    if result.get('duplicates'):
        parts.append(f"{result['duplicates']} model {'box' if result['duplicates'] == 1 else 'boxes'} on a sign "
                     f"that already has one removed")
    if not parts:
        return f'From {source_key}: no sign to carry over.'
    return f'From {source_key}: ' + '; '.join(parts) + '.'


def _with_detection_ids(detections: list, channel: str) -> list:
    """The boxes with a detection_id each: a new box without one gets "manual-<channel>-<n>"."""
    taken = set()
    for item in detections:
        if isinstance(item, dict) and isinstance(item.get('detection_id'), str):
            taken.add(item['detection_id'])
    result = []
    number = 0
    for item in detections:
        if isinstance(item, dict) and 'detection_id' not in item:
            item = dict(item)
            number = number + 1
            candidate = f'manual-{channel}-{number}'
            while candidate in taken:
                number = number + 1
                candidate = f'manual-{channel}-{number}'
            taken.add(candidate)
            item['detection_id'] = candidate
        result.append(item)
    return result


def review_schema() -> dict:
    """What a client may send when saving a frame, with the allowed values (from the import rules)."""
    return {
        'save_body': {
            'run_id': 'string, required: run_id of the frame as read (409 stale_report if the report was rebuilt)',
            'base_revision': 'integer, required: "revision" of the frame as read, 0 if never saved '
                             '(409 revision_conflict if someone saved in between)',
            'reviewer': f'string, required, at most {MAX_REVIEWER_LENGTH} characters: who reviewed. An AI name is '
                        f'stored as "{AI_REVIEWER_PREFIX}<name>"; never the name of a person for an AI review',
            'reviewer_kind': 'REST: required, one of ' + ', '.join(REVIEWER_KINDS) + '. MCP: always "ai"',
            'confirmed': 'true saves the boxes below; false withdraws the confirmation (only if asked)',
            'detections': f'list of at most {MAX_DETECTIONS_PER_FRAME} boxes; [] confirms that the frame has no '
                          f'relevant sign. A proposal left out of the list is recorded as rejected',
            'country': 'ISO 3166-1 alpha-2 code of the country of the frame ("CH", "IT", "PL"...), or null; '
                       'omitted: country_proposed of the frame, from its GPS position',
            'site_group': 'string or null (optional)',
            'encounter_id': 'string or null (optional)',
            'override_human_review': 'boolean, default false. An AI review never changes or withdraws a frame a '
                                     'human confirmed (409 human_review_exists) unless this is true, which only '
                                     'the owner may ask for',
        },
        'detection': {
            'detection_id': 'id of the proposal being kept or corrected (from "proposals"); omit it for a new '
                            'box. Fields left out are copied from that proposal',
            'bbox_xyxy': '[x1, y1, x2, y2] in pixels of the original image, inside it, x1 < x2 and y1 < y2',
            'family': list(DETECTION_FAMILIES),
            'value_kph': f'legacy alias of the written number, never unit-converted; integer between {MIN_VALUE_KPH} and {MAX_VALUE_KPH}, or null when the number cannot be '
                         f'read. Only with ' + ', '.join(VALUE_FAMILIES) + '; null with the other families',
            'sign_value': 'optional original written integer; preferred over legacy value_kph. If both are '
                          'present they must agree. 25 mph is 25, never 40',
            'sign_unit': 'optional km/h, mph or unknown; separate from the classifier numeric class',
            'applicability': sorted(ENUMS['applicability']),
            'visibility': sorted(ENUMS['visibility']),
            'occlusion': sorted(ENUMS['occlusion']),
            'supplementary_panel': sorted(ENUMS['supplementary_panel']),
            'temporary': 'true, false or null',
            'variable_display': 'true, false or null',
            'notes': 'string, possibly empty; never null',
            'sign_type': 'with family other_sign: what the sign is, one of ' + ', '.join(SIGN_TYPES)
                         + '; with family not_a_sign: on_vehicle for a sign-like disc or panel on a vehicle (the '
                         'top-speed sticker on the back of a truck); null or omitted when not said',
            'painted': 'with family other_sign: true when the sign is painted on the road (the children\'s triangle '
                       'with "SCHULE" in front of a school); its sign_type stays what it means. Null or omitted '
                       'otherwise',
            'check_position': 'only in the answer of apply_previous: true on a box placed where a sign of the '
                              'previous frame should be. A save with it still true is refused '
                              '(400 position_not_checked): look at the image, fix bbox_xyxy or drop the box, '
                              'then send it without check_position',
            'carried_from': 'only in the answer of apply_previous: the frame the labels come from (ignored when '
                            'saving)',
        },
        'label_fields': list(LABEL_KEYS) + ['notes'],
        'frame_filters': list(FRAME_FILTERS),
        'rules': [
            'Give a value only when the number is readable in this frame; never copy it from neighbours or from '
            'the second reading by hand. Unreadable number: value_kph null (family maximum_speed); not even the '
            'kind of sign: family "unreadable" with a note.',
            'apply_previous carries the labels confirmed on the previous frame onto the same signs here (the '
            'boxes move and grow as the car gets closer). It is a starting point: check every carried box in '
            'this image (still there, same value, same road) and fix or drop it before saving.',
            'End of limit ("Fine limitazione"): family "cancellation", value_kph null, crossed number in notes.',
            'Zone ("Zona 30"): family "zone". Variable or lit panels: "variable_display"; off, blank or '
            'overexposed: value_kph null and a note.',
            'applicability: own_road for our road, other_road for e.g. a side access, unresolved only when it '
            'cannot be decided.',
            'A limit with a plate of hours under it (22-6...): supplementary_panel "time" and one box around the disc '
            'and the plate, never the disc alone: with its number it is a class of its own. With any other panel the '
            'box stays on the disc.',
            'Other traffic signs (give way, stop, direction signs, town entry and exit, main road, motorway, '
            'no entry, no overtaking, weight, height or width limits, minimum speed, danger signs...) get a box '
            'too: family "other_sign" with their sign_type (get_review_schema lists them), so the model learns '
            'not to take them for limits. Logos, advertising and shop signs stay out of detections. Duplicates '
            'on the same sign: keep one box with the right value.',
            'A frame without traffic signs is confirmed with detections [] (negative examples matter).',
            'Look at the image before saving; model proposals and the second reading are only hints. Save only '
            'when the owner authorized saving for this route.',
            'A saved confirmation is final, by a person or by an AI (no second approval): the next dataset import '
            'takes it as labels, with its provenance (an AI review stays "ai/<name>", label_basis ai_review).',
        ],
        'errors': {
            '400': 'invalid request or value not allowed (error tells which)',
            '401': 'missing or wrong token (HTTP only)',
            '404': 'unknown route or frame',
            '409': 'stale_report, revision_conflict or human_review_exists: read the frame again; apply_previous: '
                   'previous_not_confirmed or no_previous_frame',
            '428': 'base_revision_required',
        },
        'instructions': 'review_instructions.md in the repository (sections 4-5) has the full rules.',
    }
