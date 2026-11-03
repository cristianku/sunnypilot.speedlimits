"""Sign distance (Data › Sign distance): how late or early the comma's vision says the sign ahead is reached.

What it is for. speedvisiond shows on the car's screen the distance to the sign ahead (speedVisionSP.signDistance,
vision.SignDistance: a round sign, a number or the end of a limit, by its size on the camera, or a motorway or
expressway sign or its end by its height; then counted down with the metres driven). A person marks, for
each sign, the frame of the road video they call "zero" (the sign is about to leave the picture); the page shows how
far from it vision's own zero comes. Only the comma's camera-side logs (speedVisionSP), the car's speed (carState) and
the person's marks: never the map.

The signs (find_signs). A sign is a run of the distance from its first sample the screen shows (round(d) > 0 in the
car's unit: metres, or feet, round(d * 3.28084), when the logs say the car shows imperial units) to its screen zero, the
first sample where it rounds to 0 (the screen hides the number there). d == 0 itself is only shown as information
(d0_s). A run that stops without a zero (the logs stop, a hole) is 'cut'. Vision's zero is compared to the person's:

- late_s, the main figure: vision's screen zero minus the person's zero, in seconds; + means vision's zero comes after
  theirs (late), - before (early);
- vision_m: what vision said at the person's zero (it should be 0 there), and the speed then (v_kmh);
- late_m, secondary: vision's last distance before its zero plus the metres driven from the person's zero to it
  (signed). It includes speedvisiond's processing delay times the speed (its message is sent after the frame).

A sign is left out of the figures (excluded, with its reason) when vision's zero is not an approach to compare:
'cut' (no screen zero), 'dropped' (no sighting for vision's DISTANCE_SECONDS: the zero is its time-out), 'coarse' (a
qlog: speedVisionSP decimated), 'no_index' (no encode index around it), 'no_video' (no road video linked yet: run
Prepare), 'slow_only' (the video around it was kept only at a car park's speed and none of its seconds is: those
seconds are never shown).

The marks. Only a person on the site writes them: signed in, or the person at the server itself on a site served only
on 127.0.0.1 without sign-in (reviewer 'local'); never an agent (no MCP tool, no REST write, no CLI, no automatic
mark). Each one is a line appended to ``annotations/sign_distance/<route>.jsonl`` ({'sign', 'mark': 'zero' |
'cant_tell' | 'clear', 'segment', 'frame', 't_ns' (the frame's exposure start from the server's encode index, never
the browser's), 'sign_t_ns', 'sync', 'reviewer', 'at'}); the last line of a sign wins and 'clear' takes its mark away.
The file is never rewritten: Undo is a line too ('mark': 'undo', with who and when, 'undoes': the number of the line it
takes back, the last line of that one sign still in effect, and that line's 'sign', 't_ns' and 'sign_t_ns'). A mark
whose sign is no longer found (the logs read again) is matched again by the time of its sign (the run that holds it),
else by its frame's time (the window that holds it; among several, the one whose zero is nearest, a tie matching none),
else listed as an orphan: never dropped. A mark on a segment whose video and logs differ (sync 'degraded'), or whose
video cannot be counted ('unknown'), is kept and left out of the medians.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from nnslr_tools import drive_logs, drive_review, video_frames
from nnslr_tools.annotations import atomic_write, jsonl_bytes, strict_json_loads
from nnslr_tools.sign_reads import CAMERA_FPS

NS = 1_000_000_000
MARKS = ('zero', 'cant_tell', 'clear')
# The lines of the file: the marks a person sends, and the Undo lines the server writes.
UNDO = 'undo'
LINES = MARKS + (UNDO,)
MARKS_RELATIVE = Path('annotations') / 'sign_distance'
# The frames shown around a sign: from WINDOW_S before its last sighting to WINDOW_S after vision's zero.
WINDOW_S = 5.0
# vision.SignDistance: the distance is 0 after DISTANCE_SECONDS without a sighting; speedvisiond runs every PERIOD_S;
# a sighting more than TRACK_SECONDS after the last one, or smaller than TRACK_SHRINK of it, starts another track.
DISTANCE_SECONDS = 15.0
PERIOD_S = 0.2
TRACK_SECONDS = 2.0
TRACK_SHRINK = 0.8
FEET = 3.28084
# A rise of the distance by more than this (metres, or this share of it) is a jump (a new estimate).
JUMP_M = 5.0
JUMP_SHARE = 0.25
# The commit that added signDistance to speedVisionSP (cristianku/sunnypilot) and its day: a build older has none.
SIGN_DISTANCE_COMMIT = 'e3cd38a88e56095d541528741f349d7b811eaade'
SIGN_DISTANCE_DAY = '2026-10-31'
SUNNYPILOT_ENV = 'NNSLR_SUNNYPILOT_CLONE'
SUNNYPILOT_DEFAULT = '/root/sunnypilot'
COMMIT_RE = re.compile(r'[0-9a-f]{7,40}')
ROUTE_RE = re.compile(r'[0-9a-f]{8}--[0-9a-f]{10}')
EXCLUDED = ('cut', 'dropped', 'coarse', 'no_index', 'no_video', 'slow_only')
EXCLUDED_MESSAGES = {
    'cut': 'The logs stop before vision reaches zero.',
    'dropped': 'Vision lost the sign: its zero is the time-out, not the approach.',
    'coarse': 'Only a qlog here: vision is logged too rarely to time its zero.',
    'no_index': 'No encode index around this sign: its frames cannot be found.',
    'no_video': 'Video not linked yet: run Prepare.',
    'slow_only': 'The video around this sign was kept only at a car park\'s speed: not shown.',
}
STATUS_MESSAGES = {
    'reading': 'Reading the logs of this drive…',
    'unread': 'The logs of this drive have not been read yet.',
    'unreadable': 'Logs found, but this server cannot open them (their file permissions).',
    'no_logs': 'No logs of this drive on this server.',
    'build_too_old': 'Driven with a build older than the sign distance (e3cd38a88): its logs have none.',
    'no_sign_seen': 'Vision showed no distance to a sign in this drive.',
    'no_video': 'Logs only, no video yet: run Prepare.',
    'signs': None,
}
# 'unread' of a drive read before, to read again (new logs, a newer reader, or its distance to the sign not exact yet).
OUTDATED_MESSAGE = 'The logs of this drive are to be read again (new logs, or a newer reader).'
LATE_M_NOTE = "Includes speedvisiond's processing delay times the speed."

_MARKING = threading.Lock()
_LAYERS: dict[tuple[str, str], tuple[str, dict]] = {}
_LAYERS_LOCK = threading.Lock()
_VERDICTS: dict[tuple[str, str], bool] = {}
# A commit the clone could not answer for (unknown to it, a time-out): (clone, commit) -> when it was asked
# (time.monotonic), not asked again for UNKNOWN_TTL_S, so a list of drives does not start a git per drive each time.
_UNKNOWN: dict[tuple[str, str], float] = {}
UNKNOWN_TTL_S = 600.0
GIT_TIMEOUT_S = 2.0


class SignDistanceError(ValueError):
    """A request Sign distance refuses: status, reason and message for the page."""

    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


# ---------- the series ----------

def _times(points: list) -> list[int]:
    return [point[0] for point in points]


def _at(points: list, times: list[int], t: int):
    """The value of a step series at a time (the last sample at or before it), None before the first."""
    position = bisect.bisect_right(times, t) - 1
    return points[position][1] if position >= 0 else None


def _slice(points: list, times: list[int], first: int, last: int) -> list:
    """The samples of a step series from first to last, with the one in force at first."""
    start = max(0, bisect.bisect_right(times, first) - 1)
    return [list(point) for point in points[start:bisect.bisect_right(times, last)]]


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def shown(distance, factor: float = 1.0) -> int | None:
    """The number the car's screen shows for a distance (metres): round(d) m, or round(d * 3.28084) ft (factor
    FEET); 0 means nothing is shown. None for no value."""
    if not _number(distance):
        return None
    return int(round(distance * factor))


def driven_m(speed: list, first: int, last: int) -> float | None:
    """The metres driven from first to last (signed: negative when last is before first) on a step series of the
    speed in km/h; None when the speed is unknown for part of it."""
    if last < first:
        value = driven_m(speed, last, first)
        return None if value is None else -value
    position = bisect.bisect_right(_times(speed), first) - 1
    if position < 0 and last > first:
        return None
    total = 0.0
    t = first
    while t < last:
        value = speed[position][1]
        until = min(speed[position + 1][0] if position + 1 < len(speed) else last, last)
        if until > t:
            if not _number(value):
                return None
            total += value / 3.6 * (until - t) / NS
            t = until
        position += 1
    return total


def find_signs(merged: dict, factor: float = 1.0) -> list[dict]:
    """The runs of vision's distance to a sign in a joined drive (see the module's text): [{'key', 'start_ns',
    'zero_ns', 'd0_ns', 'end_ns', 'end' ('zero' or 'cut'), 'last_positive': [t, d], 'last_seen_ns', 'last_seen_m',
    'tracks', 'jumps', 'raw_class'}] in time order."""
    series = merged.get('series') or {}
    dist = series.get('vis_sign_dist') or []
    dist_times = _times(dist)
    size = series.get('vis_sign_size') or []
    size_times = _times(size)
    raw = series.get('vis_raw_class') or []
    raw_times = _times(raw)
    end_of_logs = (merged.get('span') or [None, None])[1]
    runs = []
    index = 0
    while index < len(dist):
        t, d = dist[index]
        if not (shown(d, factor) or 0) > 0:
            index += 1
            continue
        start = t
        jumps = 0
        previous = d
        last_positive = [t, d]
        end, zero_ns, d0_ns, end_ns = 'cut', None, None, None
        index += 1
        while index < len(dist):
            t, d = dist[index]
            if not _number(d):
                end_ns = t
                break
            value = shown(d, factor)
            if value == 0:
                end, zero_ns, end_ns = 'zero', t, t
                # d == 0 itself (information): at the zero, or on the samples just after it that still show nothing.
                scan = index
                while scan < len(dist) and _number(dist[scan][1]) and shown(dist[scan][1], factor) == 0:
                    if dist[scan][1] == 0:
                        d0_ns = dist[scan][0]
                        break
                    scan += 1
                index += 1
                break
            if d - previous > max(JUMP_M, JUMP_SHARE * previous):
                jumps += 1
            previous = d
            last_positive = [t, d]
            index += 1
        if end_ns is None:
            end_ns = end_of_logs if end_of_logs is not None else last_positive[0]
        # The sightings: the steps of the sign's size above 0 during the run.
        last_seen = None
        tracks = 0
        previous_size = None
        previous_end = None
        position = max(0, bisect.bisect_right(size_times, start) - 1)
        while position < len(size) and size[position][0] <= end_ns:
            t_size, value = size[position]
            following = size[position + 1][0] if position + 1 < len(size) else end_ns
            if _number(value) and value > 0:
                begin = max(t_size, start)
                if (previous_end is None or begin - previous_end > TRACK_SECONDS * NS
                        or (previous_size and value < TRACK_SHRINK * previous_size)):
                    tracks += 1
                # The last message with this size: just before the next change, within the run (also where the gap
                # to the next sighting starts: vision starts another track after TRACK_SECONDS without one).
                last_seen = max(begin, min(following, end_ns) - int(PERIOD_S * NS))
                previous_size = value
                previous_end = last_seen
            position += 1
        runs.append({'key': str(start), 'start_ns': start, 'zero_ns': zero_ns, 'd0_ns': d0_ns, 'end_ns': end_ns,
                     'end': end, 'last_positive': last_positive, 'last_seen_ns': last_seen,
                     'last_seen_m': _at(dist, dist_times, last_seen) if last_seen is not None else None,
                     'tracks': tracks, 'jumps': jumps,
                     'raw_class': _at(raw, raw_times, last_seen) if last_seen is not None else None})
    return runs


# ---------- the build ----------

def _has_sign_distance(commit: str) -> bool | None:
    """The commit has signDistance in speedVisionSP (git grep in the sunnypilot clone, NNSLR_SUNNYPILOT_CLONE); None
    for a commit the clone does not know. Only what the clone has: the clone is partial (blob:none), and a missing
    object is never fetched (GIT_NO_LAZY_FETCH: no network, nothing written into it, from a page of the site)."""
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        return None
    clone = os.environ.get(SUNNYPILOT_ENV, SUNNYPILOT_DEFAULT)
    key = (clone, commit)
    if key in _VERDICTS:
        return _VERDICTS[key]
    asked = _UNKNOWN.get(key)
    if asked is not None and time.monotonic() - asked < UNKNOWN_TTL_S:
        return None
    try:
        answer = subprocess.run(['git', '-C', clone, 'grep', '-q', 'signDistance', commit, '--',
                                 'openpilot/cereal/custom.capnp'], capture_output=True, timeout=GIT_TIMEOUT_S,
                                check=False, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'})
    except (OSError, subprocess.SubprocessError):
        answer = None
    found = {0: True, 1: False}.get(answer.returncode) if answer is not None else None
    if found is None:
        _UNKNOWN[key] = time.monotonic()
    else:
        _VERDICTS[key] = found
    return found


def build_verdict(meta: dict, day: str | None) -> str:
    """'new' (the build has the sign distance), 'old' (it has not) or 'unknown': its commit in the sunnypilot clone,
    else its commit's date, else the day of the drive, before SIGN_DISTANCE_DAY."""
    build = (meta or {}).get('build') or {}
    found = _has_sign_distance(build.get('commit'))
    if found is not None:
        return 'new' if found else 'old'
    match = re.search(r'\d{4}-\d{2}-\d{2}', str(build.get('commit_date') or ''))
    if match and match.group(0) < SIGN_DISTANCE_DAY:
        return 'old'
    if day and day < SIGN_DISTANCE_DAY:
        return 'old'
    return 'unknown'


# ---------- the log layer (remembered per drive while its scan is the same) ----------

def _segments_of(spans: dict, first: int, last: int) -> list[int]:
    return sorted(int(segment) for segment, span in spans.items()
                  if span and span[0] is not None and span[0] <= last and span[1] >= first)


def _layer(store: drive_review.DriveReviewStore, route: str, summary: dict) -> dict:
    """What Sign distance takes from a drive's logs: its signs with their windows of frames and their series, the
    unit, and the sync of each segment without the video (the video is looked at on every call)."""
    signature = json.dumps([summary.get('files'), summary.get('t0_ns'), summary.get('span_ns')], sort_keys=True)
    key = (str(store.root), route)
    with _LAYERS_LOCK:
        cached = _LAYERS.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    merged = store.merged(route, keep=False) or {}
    meta = merged.get('meta') or summary.get('meta') or {}
    metric = meta.get('is_metric')
    factor = FEET if metric is False else 1.0
    t0 = merged.get('t0') or 0
    series = merged.get('series') or {}
    dist, size, speed = (series.get(name) or [] for name in ('vis_sign_dist', 'vis_sign_size', 'v_ego'))
    dist_times, size_times, speed_times = _times(dist), _times(size), _times(speed)
    index = drive_logs.frame_index(merged)
    spans = merged.get('spans') or {}
    kinds = {int(segment): item.get('kind') for segment, item in (summary.get('files') or {}).items()}
    signs = []
    for run in find_signs(merged, factor):
        first = (run['last_seen_ns'] or run['start_ns']) - int(WINDOW_S * NS)
        last = (run['zero_ns'] or run['end_ns']) + int(WINDOW_S * NS)
        frames = []
        for _first, segment, sof, ids in index:
            if sof[-1] < first or sof[0] > last:
                continue
            for at in range(bisect.bisect_left(sof, first), bisect.bisect_right(sof, last)):
                frames.append((segment, ids[at], sof[at]))
        frames.sort(key=lambda item: item[2])
        touched = _segments_of(spans, run['start_ns'], run['end_ns'])
        signs.append({**run, 'window': [first, last], 'frames': frames, 'segments': touched,
                      'qlog': any(kinds.get(segment) == 'qlog' for segment in touched),
                      'dist': _slice(dist, dist_times, first, last), 'size': _slice(size, size_times, first, last),
                      'speed': _slice(speed, speed_times, first, last)})
    syncs = {item['segment']: {name: item[name] for name in ('frames_log', 'missing_ids', 'max_error_ms')}
             for item in drive_logs.sync(merged, None)} if merged else {}
    layer = {'t0': t0, 'factor': factor, 'unit': 'ft' if metric is False else 'm', 'unit_known': metric is not None,
             'meta': meta, 'signs': signs, 'sync': syncs}
    with _LAYERS_LOCK:
        _LAYERS[key] = (signature, layer)
    return layer


# ---------- the video and the seconds that may be shown (looked at on every call) ----------

class _Video:
    """The videos of a drive's segments and their seconds kept, read once per call."""

    def __init__(self, root: Path, route: str) -> None:
        self.root, self.route = Path(root), route
        self._videos: dict[int, Path | None] = {}
        self._seconds: dict[int, set | None] = {}
        self._counts: dict[int, int | None] = {}

    def video(self, segment: int) -> Path | None:
        if segment not in self._videos:
            try:
                self._videos[segment] = video_frames.video_of(self.root, self.route, segment)
            except (video_frames.VideoFrameError, OSError):
                self._videos[segment] = None
        return self._videos[segment]

    def allowed(self, segment: int, decoded: int) -> bool:
        """A frame that may be shown: its segment has a video and, in a segment kept only at a car park's speed, its
        second is one kept (the rule of review_service: never a second that was not kept)."""
        if self.video(segment) is None:
            return False
        if segment not in self._seconds:
            seconds = video_frames.kept_seconds(self.root, self.route, segment)
            self._seconds[segment] = None if seconds is None else set(seconds)
        seconds = self._seconds[segment]
        return seconds is None or int(decoded / CAMERA_FPS) in seconds

    def frame_count(self, segment: int) -> int | None:
        if segment not in self._counts:
            video = self.video(segment)
            try:
                self._counts[segment] = video_frames.frame_count(video) if video is not None else None
            except (video_frames.VideoFrameError, OSError, ValueError, subprocess.SubprocessError):
                self._counts[segment] = None
        return self._counts[segment]


def _excluded(sign: dict, video: _Video) -> tuple[str | None, list[tuple]]:
    """Why a sign is left out (None when it is not), and the frames of its window that may be shown."""
    if sign['end'] != 'zero':
        return 'cut', []
    unseen = sign['zero_ns'] - sign['last_seen_ns'] if sign['last_seen_ns'] is not None else 0
    if unseen >= (DISTANCE_SECONDS - PERIOD_S) * NS:
        return 'dropped', []
    if sign['qlog']:
        return 'coarse', []
    if not sign['frames']:
        return 'no_index', []
    if not any(video.video(segment) is not None for segment in {item[0] for item in sign['frames']}):
        return 'no_video', []
    visible = [item for item in sign['frames'] if video.allowed(item[0], item[1])]
    if not visible:
        return 'slow_only', []
    return None, visible


def _sync_of(layer: dict, video: _Video, segment: int) -> dict:
    """The sync of a segment now: drive_logs.precision with the video's frames counted ('unknown' when they cannot
    be)."""
    base = layer['sync'].get(segment) or {'frames_log': 0, 'missing_ids': 0, 'max_error_ms': None}
    count = video.frame_count(segment)
    info = {'segment': segment, **base, 'frames_video': count}
    info['precision'] = drive_logs.precision(info) if count is not None or video.video(segment) is None else 'unknown'
    return info


# ---------- the marks ----------

def marks_path(root: Path, route: str) -> Path:
    if not isinstance(route, str) or not ROUTE_RE.fullmatch(route):
        raise SignDistanceError(404, 'unknown_route', 'Unknown drive.')
    return Path(root) / MARKS_RELATIVE / f'{route}.jsonl'


def load_marks(root: Path, route: str) -> list[dict]:
    """The lines of a drive's marks in the order they were written, each with '_line', its number in the file (the
    file only grows, so the number never changes); a line that cannot be read is skipped, never removed."""
    path = marks_path(root, route)
    if not path.is_file():
        return []
    found = []
    for number, text in enumerate(path.read_text(encoding='utf-8', errors='replace').split('\n')):
        if not text.strip():
            continue
        try:
            line = strict_json_loads(text)
        except (ValueError, TypeError):
            continue
        if not isinstance(line, dict) or not isinstance(line.get('sign'), str) or line.get('mark') not in LINES:
            continue
        if line['mark'] == UNDO and type(line.get('undoes')) is not int:
            continue
        found.append({**line, '_line': number})
    return found


def _resolve(lines: list[dict], signs: list[dict]) -> list[str]:
    """The sign each line is about now: its own key; else the sign whose run holds the time of its old key (the start
    of the sign it was marked on); else the sign whose window holds the time of a frame marked with that key, the one
    whose zero is nearest that frame when the windows of several do (a tie: none, never a guess); the key itself (an
    orphan) when none does."""
    keys = {sign['key']: sign for sign in signs}
    moved: dict[str, str] = {}
    for line in lines:
        old = line['sign']
        if old in keys or old in moved:
            continue
        same = [item for item in lines if item['sign'] == old]
        started = int(old) if old.isdigit() else next(
            (int(item['sign_t_ns']) for item in same if _number(item.get('sign_t_ns'))), None)
        found = None
        if started is not None:
            found = next((sign['key'] for sign in signs if sign['start_ns'] <= started <= sign['end_ns']), None)
        if found is None:
            times = [item['t_ns'] for item in same if _number(item.get('t_ns'))]
            near: dict[str, float] = {}
            for sign in signs:
                held = [t for t in times if sign['window'][0] <= t <= sign['window'][1]]
                if held:
                    zero = sign['zero_ns'] if sign['zero_ns'] is not None else sign['end_ns']
                    near[sign['key']] = min(abs(zero - t) for t in held)
            if near:
                best = min(near.values())
                nearest = [key for key, value in near.items() if value == best]
                found = nearest[0] if len(nearest) == 1 else None
        moved[old] = found or old
    return [line['sign'] if line['sign'] in keys else moved[line['sign']] for line in lines]


def _undone(lines: list[dict]) -> set[int]:
    """The numbers of the lines an Undo line took back."""
    return {line['undoes'] for line in lines if line['mark'] == UNDO}


def _in_effect(lines: list[dict], resolved: list[str], key: str) -> dict | None:
    """The last line of a sign that no Undo took back (a 'clear' included), None without one."""
    undone = _undone(lines)
    return next((line for line, item in zip(reversed(lines), reversed(resolved))
                 if item == key and line['mark'] != UNDO and line['_line'] not in undone), None)


def _current_marks(lines: list[dict], resolved: list[str]) -> dict[str, dict]:
    """{sign: its last line in effect} without the signs whose last line in effect is 'clear'."""
    undone = _undone(lines)
    last: dict[str, dict] = {}
    for line, key in zip(lines, resolved):
        if line['mark'] != UNDO and line['_line'] not in undone:
            last[key] = line
    return {key: line for key, line in last.items() if line['mark'] != 'clear'}


def _seconds(t, t0) -> float | None:
    return round((t - t0) / NS, 3) if _number(t) and t0 is not None else None


def _metres(value) -> float | None:
    """A distance for the page, to the millimetre: the logs keep it unrounded (float32), every rule here takes it so
    (shown), only what is sent is rounded."""
    return round(value, 3) if _number(value) else None


def _metrics(sign: dict, line: dict) -> dict | None:
    """The figures of a 'zero' mark against vision's zero (see the module's text)."""
    t_z = line.get('t_ns')
    if line['mark'] != 'zero' or sign['zero_ns'] is None or not _number(t_z):
        return None
    dist, speed = sign['dist'], sign['speed']
    vision = _at(dist, _times(dist), t_z)
    v_kmh = _at(speed, _times(speed), t_z)
    t_lp, d_lp = sign['last_positive']
    driven = driven_m(speed, t_z, t_lp)
    return {'late_s': round((sign['zero_ns'] - t_z) / NS, 3),
            'vision_m': _metres(vision),
            'vision_shown': shown(vision, sign.get('factor', 1.0)),
            'v_kmh': round(v_kmh, 1) if _number(v_kmh) else None,
            'late_m': round(d_lp + driven, 2) if driven is not None else None,
            'late_m_note': LATE_M_NOTE, 'speed_missing': driven is None}


def _public_mark(line: dict, t0) -> dict:
    return {'mark': line['mark'], 'segment': line.get('segment'), 'frame': line.get('frame'),
            't_s': _seconds(line.get('t_ns'), t0), 'sync': line.get('sync'), 'reviewer': line.get('reviewer'),
            'at': line.get('at')}


# ---------- the views ----------

def _route_known(root: Path, route: str) -> None:
    if not isinstance(route, str) or not ROUTE_RE.fullmatch(route) or not drive_logs.known_route(root, route):
        raise SignDistanceError(404, 'unknown_route', 'Unknown drive.')


def _drive(store: drive_review.DriveReviewStore, route: str, *, describe: bool = True) -> dict:
    """A drive as Sign distance sees it: its status and, when it has signs, everything about them (internal: times
    in ns). describe: its name and when it was driven too (its GPS positions are read: not for every frame)."""
    from nnslr_tools.drives import drive_of
    root = store.root
    summary = store.summary(route)
    civil = (summary or {}).get('civil_ms')
    drive = drive_of(root, route) if describe or not civil else {}
    when, certain = drive_review._when(civil, drive) if describe else (None, False)
    day = drive_review.drive_day(civil, drive)
    if describe and not drive.get('title') and (summary or {}).get('gps'):
        drive['title'] = drive_review._title(root, summary['gps'])
    state = store.scan_state(route)
    out = {'route': route, 'title': drive.get('title'), 'when': when, 'when_certain': certain, 'day': day,
           'scan': state['state'], 'unreadable_files': state.get('unreadable_files') or [], 'unit': 'm',
           'unit_known': False, 'build': None, 'signs': [], 'layer': None}
    if state['state'] in ('queued', 'scanning'):
        status = 'reading'
    elif store.needs_scan(route, state):
        status = 'unread'
    elif state['state'] == 'unreadable_logs' or (state['state'] == 'failed' and summary is None):
        status = 'unreadable'
    elif state['state'] == 'no_logs' or summary is None:
        status = 'no_logs'
    else:
        # The logs first: a distance in them shows the build has it, whatever its commit or its date guess.
        layer = _layer(store, route, summary)
        meta = summary.get('meta') or {}
        build = meta.get('build') or {}
        verdict = 'new' if layer['signs'] else build_verdict(meta, day)
        out['build'] = {'version': build.get('version'), 'branch': build.get('branch'), 'commit': build.get('commit'),
                        'commit_date': build.get('commit_date'), 'verdict': verdict}
        out.update(unit=layer['unit'], unit_known=layer['unit_known'], layer=layer)
        if verdict == 'old':
            status = 'build_too_old'
        elif not layer['signs']:
            status = 'no_sign_seen'
        elif not any(_Video(root, route).video(segment) is not None
                     for segment in {item[0] for sign in layer['signs'] for item in sign['frames']}
                     | {segment for sign in layer['signs'] for segment in sign['segments']}):
            status = 'no_video'
        else:
            status = 'signs'
    message = STATUS_MESSAGES[status]
    if status == 'unread' and state['state'] == 'outdated':
        message = OUTDATED_MESSAGE
    if status == 'unread' and state.get('error'):
        message += ' Last error: ' + str(state['error'])[:200]
    if status == 'no_sign_seen' and (out['build'] or {}).get('verdict') == 'unknown':
        commit = (out['build'] or {}).get('commit')
        message += (f" Its build ({commit[:9] if isinstance(commit, str) and commit else 'no commit'}) is not known "
                    'here: it may be older than the sign distance.')
    out.update(status=status, message=message)
    return out


def _signs_view(store: drive_review.DriveReviewStore, route: str, drive: dict) -> tuple[list[dict], list[dict], dict]:
    """The public signs of a drive with their marks and figures, its orphan marks, and the internal signs by key
    (with their visible frames)."""
    layer = drive['layer']
    if layer is None:
        orphans = [{'sign': line['sign'], **_public_mark(line, None)}
                   for line in _current_marks(*_marks_of(store.root, route, [])).values()]
        return [], orphans, {}
    t0 = layer['t0']
    video = _Video(store.root, route)
    lines, resolved = _marks_of(store.root, route, layer['signs'])
    current = _current_marks(lines, resolved)
    keys = {sign['key'] for sign in layer['signs']}
    public, internal = [], {}
    for number, sign in enumerate(layer['signs'], 1):
        reason, visible = _excluded(sign, video)
        line = current.get(sign['key'])
        metrics = _metrics({**sign, 'factor': layer['factor']}, line) if line else None
        internal[sign['key']] = {**sign, 'excluded': reason, 'visible': visible}
        public.append({
            'key': sign['key'], 'number': number, 'start_s': _seconds(sign['start_ns'], t0),
            'zero_s': _seconds(sign['zero_ns'], t0), 'd0_s': _seconds(sign['d0_ns'], t0),
            'end_s': _seconds(sign['end_ns'], t0), 'end': sign['end'],
            'last_seen_s': _seconds(sign['last_seen_ns'], t0), 'last_seen_m': _metres(sign['last_seen_m']),
            'last_positive_m': _metres(sign['last_positive'][1]), 'raw_class': sign['raw_class'],
            'tracks': sign['tracks'],
            'jumps': sign['jumps'], 'segments': sign['segments'], 'included': reason is None, 'excluded': reason,
            'excluded_message': EXCLUDED_MESSAGES.get(reason), 'frames': len(visible),
            'mark': _public_mark(line, t0) if line else None, 'metrics': metrics})
    orphans = [{'sign': key, **_public_mark(line, t0)} for key, line in current.items() if key not in keys]
    return public, orphans, internal


def _marks_of(root: Path, route: str, signs: list[dict]) -> tuple[list[dict], list[str]]:
    lines = load_marks(root, route)
    return lines, _resolve(lines, signs)


def _counts(signs: list[dict], orphans: list[dict]) -> dict:
    excluded: dict[str, int] = {}
    for sign in signs:
        if sign['excluded']:
            excluded[sign['excluded']] = excluded.get(sign['excluded'], 0) + 1
    marked = [sign['mark']['mark'] for sign in signs if sign['mark']]
    return {'signs': len(signs), 'included': sum(1 for sign in signs if sign['included']), 'excluded': excluded,
            'zero': marked.count('zero'), 'cant_tell': marked.count('cant_tell'), 'orphans': len(orphans)}


def _public_drive(drive: dict, signs: list[dict], orphans: list[dict]) -> dict:
    return {'route': drive['route'], 'title': drive['title'], 'when': drive['when'],
            'when_certain': drive['when_certain'], 'day': drive['day'], 'status': drive['status'],
            'message': drive['message'], 'scan': drive['scan'], 'unreadable_files': drive['unreadable_files'],
            'unit': drive['unit'], 'unit_known': drive['unit_known'], 'build': drive['build'],
            'counts': _counts(signs, orphans)}


def drive_view(store: drive_review.DriveReviewStore, route: str) -> dict:
    """One drive for the page: its status, its signs with their marks and figures, its orphan marks."""
    _route_known(store.root, route)
    drive = _drive(store, route)
    signs, orphans, _internal = _signs_view(store, route, drive)
    return {'ok': True, **_public_drive(drive, signs, orphans), 'signs': signs, 'orphans': orphans}


def _stats(values: list[float]) -> dict:
    if not values:
        return {'median': None, 'count': 0, 'min': None, 'max': None}
    return {'median': round(statistics.median(values), 3), 'count': len(values), 'min': min(values),
            'max': max(values)}


def overview(store: drive_review.DriveReviewStore) -> dict:
    """Every drive of the portal with its status and counts, and the figures of all of them: {'drives', 'summary'}.
    The medians are over the 'zero' marks of included signs on segments in sync to the frame."""
    drives, late_s, late_m = [], [], []
    degraded = orphans_count = 0
    excluded: dict[str, int] = {}
    statuses: dict[str, int] = {}
    by_drive = []
    for route in drive_review.all_routes(store.root):
        drive = _drive(store, route)
        signs, orphans, _internal = _signs_view(store, route, drive)
        public = _public_drive(drive, signs, orphans)
        drives.append(public)
        statuses[drive['status']] = statuses.get(drive['status'], 0) + 1
        orphans_count += len(orphans)
        for reason, count in public['counts']['excluded'].items():
            excluded[reason] = excluded.get(reason, 0) + count
        mine = []
        for sign in signs:
            metrics = sign['metrics']
            if not sign['included'] or metrics is None:
                continue
            if sign['mark']['sync'] != 'frame':
                degraded += 1
                continue
            mine.append(metrics['late_s'])
            late_s.append(metrics['late_s'])
            if metrics['late_m'] is not None:
                late_m.append(metrics['late_m'])
        if signs:
            by_drive.append({'route': route, 'title': drive['title'], 'included': public['counts']['included'],
                             'zero': public['counts']['zero'], 'late_s': _stats(mine)})
    drives.sort(key=lambda item: item['route'], reverse=True)
    summary = {'drives': len(drives), 'statuses': statuses,
               'signs': sum(item['counts']['signs'] for item in drives),
               'included': sum(item['counts']['included'] for item in drives), 'excluded': excluded,
               'zero': sum(item['counts']['zero'] for item in drives),
               'cant_tell': sum(item['counts']['cant_tell'] for item in drives),
               'late_s': _stats(late_s), 'late_m': {**_stats(late_m), 'note': LATE_M_NOTE},
               'degraded_marks': degraded, 'orphans': orphans_count, 'by_drive': by_drive}
    return {'ok': True, 'drives': drives, 'summary': summary}


def _sign_of(store: drive_review.DriveReviewStore, route: str, key, *, describe: bool = True
             ) -> tuple[dict, list[dict], dict, dict]:
    _route_known(store.root, route)
    drive = _drive(store, route, describe=describe)
    signs, orphans, internal = _signs_view(store, route, drive)
    if not isinstance(key, str) or key not in internal:
        raise SignDistanceError(404, 'unknown_sign', 'This drive has no such sign.')
    return drive, signs, internal, next(sign for sign in signs if sign['key'] == key)


def _warnings(sign: dict, syncs: list[dict]) -> list[dict]:
    out = []
    for info in syncs:
        if info['precision'] == 'degraded':
            out.append({'code': 'sync_degraded', 'segment': info['segment'],
                        'message': f"Minute {info['segment']}: its video and its logs differ (sync to within "
                                   f"{info['max_error_ms']} ms): a mark is kept, out of the medians."})
        elif info['precision'] == 'unknown':
            out.append({'code': 'sync_unknown', 'segment': info['segment'],
                        'message': f"Minute {info['segment']}: its video cannot be counted: a mark is kept, out of "
                                   'the medians.'})
        elif info['precision'] == 'no_video':
            out.append({'code': 'sync_no_video', 'segment': info['segment'],
                        'message': f"Minute {info['segment']}: video not linked yet (run Prepare): its frames are "
                                   'not shown.'})
    if sign['tracks'] > 1:
        out.append({'code': 'several_tracks', 'message': f"Vision followed this sign as {sign['tracks']} tracks."})
    if sign['jumps']:
        out.append({'code': 'distance_jumps', 'message': f"Vision's distance jumped up {sign['jumps']} times."})
    return out


def sign_view(store: drive_review.DriveReviewStore, route: str, key: str) -> dict:
    """One sign for the page: its figures, the frames of its window that may be shown (none for a sign left out),
    vision's distance, the sign's size and the speed over the window, the sync of its minutes, the signs before and
    after it to mark."""
    drive, signs, internal, public = _sign_of(store, route, key)
    layer = drive['layer']
    t0, factor = layer['t0'], layer['factor']
    sign = internal[key]
    video = _Video(store.root, route)
    dist, dist_times = sign['dist'], _times(sign['dist'])
    frames = [{'segment': segment, 'frame': decoded, 't_s': _seconds(sof, t0),
               'vision_m': _metres(_at(dist, dist_times, sof)), 'shown': shown(_at(dist, dist_times, sof), factor)}
              for segment, decoded, sof in sign['visible']]
    # Every minute of the window, a minute whose frames cannot be shown (no video yet) included.
    syncs = [_sync_of(layer, video, segment) for segment in sorted({item[0] for item in sign['frames']})]
    included = [item['key'] for item in signs if item['included']]
    position = included.index(key) if key in included else None
    before = [item['key'] for item in signs if item['included'] and int(item['key']) < int(key)]
    after = [item['key'] for item in signs if item['included'] and int(item['key']) > int(key)]

    def points(name: str, digits: int) -> list:
        return [[_seconds(t, t0), round(value, digits) if _number(value) else value] for t, value in sign[name]]

    return {'ok': True, 'route': route, 'unit': layer['unit'], 'unit_known': layer['unit_known'], 'sign': public,
            'window': {'from_s': _seconds(sign['window'][0], t0), 'to_s': _seconds(sign['window'][1], t0),
                       'frames': frames, 'wide_segments': _wide_segments(store.root, route, frames)},
            'series': {'vision_m': points('dist', 2), 'size_px': points('size', 2), 'speed_kmh': points('speed', 1)},
            'sync': [{name: info[name] for name in ('segment', 'precision', 'max_error_ms', 'frames_log',
                                                    'frames_video')} for info in syncs],
            'warnings': _warnings(sign, syncs),
            'prev': before[-1] if before else None, 'next': after[0] if after else None,
            'position': position + 1 if position is not None else None, 'of': len(included)}


def _wide_segments(root: Path, route: str, frames: list[dict]) -> list[int]:
    """The minutes of a window whose wide road camera can be shown beside the road one (video_frames.wide_video_of)."""
    found = []
    for segment in sorted({frame['segment'] for frame in frames}):
        try:
            video_frames.wide_video_of(root, route, segment)
        except (video_frames.VideoFrameError, OSError):
            continue
        found.append(segment)
    return found


def frame_allowed(store: drive_review.DriveReviewStore, route: str, segment: int, decoded: int) -> bool:
    """A frame of the road video the page may show: in the window of an included sign, never a second that was not
    kept. Light, as it runs for every picture: the drive's summary and its cached signs, then the video and its kept
    seconds (looked at live) only for the signs whose window holds this frame; no scan state, no marks."""
    _route_known(store.root, route)
    summary = store.summary(route)
    if summary is None:
        return False
    wanted = (segment, decoded)
    video = _Video(store.root, route)
    for sign in _layer(store, route, summary)['signs']:
        if not any(wanted == (item[0], item[1]) for item in sign['frames']):
            continue
        reason, visible = _excluded(sign, video)
        if reason is None and any(wanted == (item[0], item[1]) for item in visible):
            return True
    return False


def frame_jpeg(store: drive_review.DriveReviewStore, route: str, segment: int, decoded: int, *,
               wide: bool = False) -> bytes:
    """The JPEG of a frame frame_allowed lets through (wide: the same frame of the wide road camera); SignDistanceError
    404 for any other."""
    if not frame_allowed(store, route, segment, decoded):
        raise SignDistanceError(404, 'unknown_frame', 'This frame is not one of the frames of a sign.')
    try:
        return video_frames.preview(store.root, route, segment, decoded, wide=wide).read_bytes()
    except (video_frames.VideoFrameError, OSError) as exc:
        raise SignDistanceError(404, 'no_video_frame', 'This frame of the video cannot be shown.') from exc


def mark(store: drive_review.DriveReviewStore, route: str, body, reviewer: str) -> dict:
    """A person's mark on a sign ({'sign', 'mark', 'segment', 'frame'}: the frame of the zero, or the frame they
    could not tell on; segment and frame null for 'clear'), appended to the drive's marks. The time is the frame's
    exposure start from the server's encode index. Returns the sign again, the line kept and warnings."""
    if not isinstance(body, dict) or set(body) != {'sign', 'mark', 'segment', 'frame'}:
        raise SignDistanceError(400, 'bad_body', 'Send {"sign", "mark", "segment", "frame"} and nothing else.')
    kind = body['mark']
    if kind not in MARKS or not isinstance(body['sign'], str):
        raise SignDistanceError(400, 'bad_body', 'mark is "zero", "cant_tell" or "clear"; sign is the key of a sign.')
    if kind == 'clear':
        if body['segment'] is not None or body['frame'] is not None:
            raise SignDistanceError(400, 'bad_body', 'A clear has segment and frame null.')
    elif not all(type(body[name]) is int and body[name] >= 0 for name in ('segment', 'frame')):
        raise SignDistanceError(400, 'bad_body', 'segment and frame are the numbers of a frame of the sign.')
    drive, _signs, internal, _public = _sign_of(store, route, body['sign'], describe=False)
    sign = internal[body['sign']]
    layer = drive['layer']
    warnings: list[dict] = []
    line = {'sign': sign['key'], 'mark': kind, 'segment': body['segment'], 'frame': body['frame'], 't_ns': None,
            'sign_t_ns': sign['start_ns'], 'sync': None, 'reviewer': reviewer,
            'at': datetime.now().astimezone().isoformat(timespec='seconds')}
    if kind != 'clear':
        if sign['excluded'] is not None:
            raise SignDistanceError(409, 'sign_excluded',
                                    'This sign is left out: ' + EXCLUDED_MESSAGES[sign['excluded']])
        found = next((item for item in sign['visible'] if (item[0], item[1]) == (body['segment'], body['frame'])),
                     None)
        if found is None:
            raise SignDistanceError(400, 'bad_frame', 'This frame is not one of the frames of the sign.')
        info = _sync_of(layer, _Video(store.root, route), found[0])
        line.update(t_ns=found[2], sync=info['precision'])
        warnings = [item for item in _warnings(sign, [info]) if item['code'].startswith('sync_')]
    path = marks_path(store.root, route)
    with _MARKING:
        before = path.read_bytes() if path.is_file() else b''
        if before and not before.endswith(b'\n'):
            before += b'\n'
        atomic_write(path, before + jsonl_bytes([line]))
    _drive_again, _signs_again, _internal, public = _sign_of(store, route, sign['key'], describe=False)
    return {'ok': True, 'sign': public, 'line': {name: line[name] for name in ('sign', 'mark', 'segment', 'frame',
                                                                               'sync', 'reviewer', 'at')},
            'warnings': warnings}


def undo(store: drive_review.DriveReviewStore, route: str, body, reviewer: str) -> dict:
    """Takes back the last line in effect of one sign ({'sign'}: a sign of the drive, or an orphan mark's key) with an
    Undo line appended (the file is never rewritten). Returns the sign again (None for an orphan) and what was
    undone."""
    if not isinstance(body, dict) or set(body) != {'sign'} or not isinstance(body['sign'], str):
        raise SignDistanceError(400, 'bad_body', 'Send {"sign"}: the key of the sign to undo.')
    _route_known(store.root, route)
    drive = _drive(store, route, describe=False)
    signs = (drive['layer'] or {}).get('signs') or []
    key = body['sign']
    path = marks_path(store.root, route)
    with _MARKING:
        lines = load_marks(store.root, route)
        resolved = _resolve(lines, signs)
        if key not in {sign['key'] for sign in signs} and key not in resolved:
            raise SignDistanceError(404, 'unknown_sign', 'This drive has no such sign.')
        target = _in_effect(lines, resolved, key)
        if target is None:
            raise SignDistanceError(409, 'nothing_to_undo', 'This sign has no mark to undo.')
        # The line of its target's sign and times: read again later, it goes with it to the same sign.
        line = {'sign': target['sign'], 'mark': UNDO, 'segment': None, 'frame': None, 't_ns': target.get('t_ns'),
                'sign_t_ns': target.get('sign_t_ns'), 'sync': None, 'reviewer': reviewer,
                'at': datetime.now().astimezone().isoformat(timespec='seconds'), 'undoes': target['_line']}
        before = path.read_bytes()
        if before and not before.endswith(b'\n'):
            before += b'\n'
        atomic_write(path, before + jsonl_bytes([line]))
    public = None
    if key in {sign['key'] for sign in signs}:
        public = _sign_of(store, route, key, describe=False)[3]
    return {'ok': True, 'sign': public, 'undone': {'sign': key, 'mark': target['mark']}}
