"""Sign distance (Data › Sign distance, nnslr_tools.sign_distance): vision's screen zero of a sign against the frame a
person marks, the marks themselves, the seconds never shown, and the drive statuses.

Every log here is SYNTHETIC: written by these tests with the made-up schema of the drive review tests
(fixtures/drive_schema) for a made-up drive of two 10-second minutes. Nothing here is a real drive or a person's mark.

The made-up drive (seconds from its start; 18 km/h, 5 m/s, throughout; speedVisionSP 5 times a second):
- sign A: seen from 2.0 s to 6.0 s (its size 40 to 80 px); vision's distance (7.08 - s) * 5 m from 2.0 s, so 1 m
  steps (below the old 2 m tolerance), 0.4 m at 7.0 s (the metric screen's zero: round(0.4) == 0; 1.3 ft still shows
  1 on an imperial screen) and 0 at 7.2 s;
- sign B: seen from 15 s, its distance still 5 m when the logs stop at 20 s: cut, left out.
"""
from __future__ import annotations

import functools
import gzip
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

capnp = pytest.importorskip('capnp')
zstandard = pytest.importorskip('zstandard')

from nnslr_tools import drive_logs, drive_review, sign_distance  # noqa: E402
from nnslr_tools.sync import SLOW_ONLY_MARK  # noqa: E402
from review_helpers import ADMIN, ROUTE, TOKEN, RunningServer, json_of, sign_in  # noqa: E402

SCHEMA = Path(__file__).parent / 'fixtures' / 'drive_schema'
NS = 1_000_000_000
BOOT_NS = 7_345 * NS
SECONDS = 10
SEGMENTS = (0, 1)
SPEED_KMH = 18.0
NEW_COMMIT = '1' * 40
# The exposure start of frame 130 of minute 0 (6.5 s), and vision's screen zero of sign A (7.0 s + 5 ms).
MARK_FRAME = 130
MARK_NS = BOOT_NS + 6_500_000_000
ZERO_NS = BOOT_NS + 7_005_000_000


# ---------- the synthetic logs ----------

def _distance(s: float) -> float:
    if 2.0 <= s < 7.4:
        return max(0.0, (7.08 - s) * SPEED_KMH / 3.6)
    if s >= 15.0:
        return 30.0 - (s - 15.0) * 5.0
    return 0.0


def _size(s: float) -> float:
    if 2.0 <= s <= 6.0:
        return 40.0 + 10.0 * (s - 2.0)
    return 50.0 if s >= 15.0 else 0.0


@functools.lru_cache(maxsize=None)
def _schema():
    return capnp.load(str(SCHEMA / 'log.capnp'), imports=[str(SCHEMA)])


@functools.lru_cache(maxsize=None)
def segment_log(segment: int, metric: bool | None = True, commit: str = NEW_COMMIT,
                commit_date: str = "'1793350800 2026-11-02 08:00:00 +0000'") -> bytes:
    """A SYNTHETIC rlog of one 10-second minute of the made-up drive, zstd-compressed as the comma writes it."""
    schema = _schema()
    start = BOOT_NS + segment * SECONDS * NS
    messages: list[tuple[int, bytes]] = []

    def put(t: int, fill) -> None:
        event = schema.Event.new_message()
        event.logMonoTime = t
        event.valid = True
        fill(event)
        messages.append((t, event.to_bytes()))

    def init(event):
        data = event.init('initData')
        data.version, data.gitBranch, data.gitCommit, data.deviceType = '0.0-synthetic', 'synthetic', commit, 'mici'
        data.gitCommitDate = commit_date
        if metric is not None:
            entries = data.init('params').init('entries', 1)
            entries[0].key, entries[0].value = 'IsMetric', b'1' if metric else b'0'
    put(start, init)
    for k in range(SECONDS * 10):
        t = start + k * NS // 10

        def car_state(event):
            car = event.init('carState')
            car.vEgo = SPEED_KMH / 3.6
            car.vCruise = 30
            cruise = car.init('cruiseState')
            cruise.enabled, cruise.available, cruise.speed = True, True, 30 / 3.6
        put(t + 1_000_000, car_state)
    for k in range(SECONDS * 5):
        t = start + k * NS // 5
        s = (t - BOOT_NS) / NS

        def speed_vision(event, s=s):
            sign = event.init('speedVisionSP')
            sign.speedLimit, sign.modelName, sign.roadType = 30 / 3.6, 'synthetic', 'urban'
            sign.speedLimitState, sign.speedLimitReason = 'limit', 'sign'
            sign.rawSignClass = 'maximum_speed:30'
            sign.signDistance = _distance(s)
            sign.signSize = _size(s)
        put(t + 5_000_000, speed_vision)
    for frame in range(SECONDS * 20):
        sof = start + frame * NS // 20

        def encode(event, frame=frame, sof=sof):
            index = event.init('roadEncodeIdx')
            index.frameId, index.segmentNum, index.segmentId = segment * SECONDS * 20 + frame, segment, frame
            index.timestampSof, index.timestampEof = sof, sof + 20_000_000
        put(sof + 25_000_000, encode)
    messages.sort(key=lambda item: item[0])
    return zstandard.ZstdCompressor(level=3).compress(b''.join(data for _t, data in messages))


def comma_drive(root: Path, route: str = ROUTE, segments=SEGMENTS, *, video: bool = True, **options) -> None:
    """The made-up drive as the comma's download leaves it: per minute its rlog and a placeholder road video (not a
    video: it is never decoded, its frames cannot be counted)."""
    for segment in segments:
        folder = root / 'raw' / 'routes' / route / str(segment)
        folder.mkdir(parents=True, exist_ok=True)
        if video:
            (folder / 'fcamera.hevc').write_bytes(b'synthetic hevc ' * (segment + 1))
        (folder / 'rlog.zst').write_bytes(segment_log(segment, **options))


def real_video(root: Path, segment: int = 0, route: str = ROUTE, name: str = 'fcamera.hevc') -> None:
    """A real HEVC road video of a minute, 200 frames (the encode index's), a keyframe every 20 (name ecamera.hevc:
    the wide road camera's)."""
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required for local media integration')
    video = root / 'raw/routes' / route / str(segment) / name
    video.parent.mkdir(parents=True, exist_ok=True)
    made = subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc=s=64x48:r=20:d={SECONDS}', '-pix_fmt', 'yuv420p',
        '-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20',
        '-f', 'hevc', str(video)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr


def slow_only(root: Path, segment: int, seconds, route: str = ROUTE) -> None:
    """A minute kept only at a car park's speed, its pictures at 1 a second only at these seconds."""
    (root / 'raw/routes' / route / str(segment) / SLOW_ONLY_MARK).write_text('')
    folder = root / 'derived/frames' / route / str(segment)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'frames.jsonl').write_text(''.join(json.dumps({'segment_index': segment, 'output_index': second}) + '\n'
                                                 for second in seconds))


@pytest.fixture(autouse=True)
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv(drive_logs.SCHEMA_ENV, str(SCHEMA))
    shared = tmp_path / 'shared'
    shared.mkdir()
    monkeypatch.setenv('COMMA_VIDEOS_DIR', str(shared))
    # Never the server's sunnypilot clone: a commit is unknown unless a test makes a clone.
    monkeypatch.setenv(sign_distance.SUNNYPILOT_ENV, str(tmp_path / 'no-clone'))
    sign_distance._VERDICTS.clear()
    sign_distance._UNKNOWN.clear()
    sign_distance._LAYERS.clear()


def scanned(root: Path, route: str = ROUTE) -> drive_review.DriveReviewStore:
    store = drive_review.store_for(root)
    assert store.scan(route)['state'] in ('done', 'no_episodes')
    return store


def key_of(view: dict, number: int) -> str:
    return next(sign['key'] for sign in view['signs'] if sign['number'] == number)


def mark_body(view: dict, mark: str = 'zero', segment=0, frame=MARK_FRAME, number: int = 1) -> dict:
    return {'sign': key_of(view, number), 'mark': mark, 'segment': segment, 'frame': frame}


def post(server, path, body, headers=None):
    status, _response, data = server.request('POST', path, body=body, headers=headers)
    return status, json_of(data)


def get(server, path, headers=None):
    status, response, data = server.request('GET', path, headers=headers)
    return status, response, data


# ---------- the signs, from hand-made joined drives ----------

def merged_of(dist, size=(), speed=((0, 18.0),), span_end=None, frames=None) -> dict:
    return {'series': {'vis_sign_dist': [list(point) for point in dist], 'vis_sign_size': [list(p) for p in size],
                       'v_ego': [list(point) for point in speed]},
            'span': [0, span_end if span_end is not None else (dist[-1][0] if dist else 0)],
            'frames': frames or {}, 'meta': {}}


def test_screen_zero_is_the_first_sample_that_rounds_to_zero_and_d0_is_information():
    s = NS // 5
    dist = [(0, 0.0), (s, 12.0), (2 * s, 7.0), (3 * s, 3.0), (4 * s, 0.4), (5 * s, 0.0), (6 * s, 9.0), (7 * s, None)]
    signs = sign_distance.find_signs(merged_of(dist, size=[(s, 30.0), (3 * s, 0.0)]))
    first, second = signs
    assert (first['start_ns'], first['zero_ns'], first['d0_ns'], first['end']) == (s, 4 * s, 5 * s, 'zero')
    assert first['last_positive'] == [3 * s, 3.0]
    # Sighted from s to just before the size went to 0 (3 s): its last message, one period before.
    assert first['last_seen_ns'] == 3 * s - int(sign_distance.PERIOD_S * NS) and first['tracks'] == 1
    assert first['last_seen_m'] == 7.0
    # Another run from 9 m, cut by the logs stopping (None): no zero.
    assert (second['start_ns'], second['zero_ns'], second['end'], second['end_ns']) == (6 * s, None, 'cut', 7 * s)


def test_imperial_screen_shows_feet_and_reaches_zero_later():
    s = NS // 5
    dist = [(0, 0.0), (s, 5.0), (2 * s, 0.4), (3 * s, 0.1), (4 * s, 0.0)]
    metric = sign_distance.find_signs(merged_of(dist))[0]
    imperial = sign_distance.find_signs(merged_of(dist), sign_distance.FEET)[0]
    assert metric['zero_ns'] == 2 * s
    # 0.4 m is 1.3 ft (1 on the screen), 0.1 m is 0.33 ft (nothing): the zero is a sample later.
    assert imperial['zero_ns'] == 3 * s and imperial['d0_ns'] == 4 * s
    assert sign_distance.shown(0.5) == 0 and sign_distance.shown(1.5) == 2 and sign_distance.shown(None) is None


def test_jumps_and_tracks_are_counted():
    s = NS // 5
    dist = [(0, 40.0), (s, 30.0), (2 * s, 45.0), (3 * s, 20.0), (4 * s, 0.0)]
    size = [(0, 30.0), (s, 34.0), (2 * s, 20.0), (3 * s, 0.0)]
    sign = sign_distance.find_signs(merged_of(dist, size))[0]
    # 30 -> 45 is a jump (more than 5 m and 25 %); a size below 80 % of the last one is another track.
    assert sign['jumps'] == 1 and sign['tracks'] == 2


def test_a_new_track_starts_two_seconds_after_the_last_message_of_a_sighting():
    dist = [(0, 40.0), (5 * NS, 0.0)]
    # Seen until 1 s (its last message at 0.8 s), again from 2.9 s: 2.1 s without a sighting, another track (from
    # the change at 1 s it would be 1.9 s, the same track).
    size = [(0, 30.0), (NS, 0.0), (2_900_000_000, 30.0), (4 * NS, 0.0)]
    sign = sign_distance.find_signs(merged_of(dist, size))[0]
    assert sign['tracks'] == 2 and sign['last_seen_ns'] == 3_800_000_000
    size = [(0, 30.0), (NS, 0.0), (2_700_000_000, 30.0), (4 * NS, 0.0)]
    assert sign_distance.find_signs(merged_of(dist, size))[0]['tracks'] == 1


def test_driven_metres_are_signed_and_unknown_without_speed():
    speed = [[0, 36.0], [NS, 72.0]]
    assert sign_distance.driven_m(speed, 0, 2 * NS) == pytest.approx(30.0)
    assert sign_distance.driven_m(speed, 2 * NS, 0) == pytest.approx(-30.0)
    assert sign_distance.driven_m([[0, 36.0], [NS, None]], 0, 2 * NS) is None
    assert sign_distance.driven_m([[NS, 36.0]], 0, 2 * NS) is None


class _NoVideo:
    def __init__(self, videos=(), seconds=None):
        self.videos, self.seconds = set(videos), seconds

    def video(self, segment):
        return Path('/v') if segment in self.videos else None

    def allowed(self, segment, decoded):
        return segment in self.videos and (self.seconds is None or int(decoded / 20) in self.seconds)


def test_reasons_a_sign_is_left_out_in_order():
    base = {'end': 'zero', 'zero_ns': 20 * NS, 'last_seen_ns': 19 * NS, 'qlog': False,
            'frames': [(0, 10, 15 * NS), (0, 30, 16 * NS)]}
    video = _NoVideo({0})
    assert sign_distance._excluded({**base, 'end': 'cut'}, video)[0] == 'cut'
    # Vision's DISTANCE_SECONDS without a sighting: the zero is its time-out.
    assert sign_distance._excluded({**base, 'last_seen_ns': 5 * NS}, video)[0] == 'dropped'
    assert sign_distance._excluded({**base, 'qlog': True}, video)[0] == 'coarse'
    assert sign_distance._excluded({**base, 'frames': []}, video)[0] == 'no_index'
    assert sign_distance._excluded(base, _NoVideo())[0] == 'no_video'
    assert sign_distance._excluded(base, _NoVideo({0}, seconds={3}))[0] == 'slow_only'
    reason, visible = sign_distance._excluded(base, _NoVideo({0}, seconds={1}))
    assert reason is None and visible == [(0, 30, 16 * NS)]
    assert sign_distance.EXCLUDED_MESSAGES['no_video'] == 'Video not linked yet: run Prepare.'


def test_a_commit_is_looked_up_without_the_network_and_an_unknown_one_not_every_call(monkeypatch):
    calls = []

    class Answer:
        returncode = 128

    def run(command, **options):
        calls.append((command, options))
        return Answer()
    monkeypatch.setattr(sign_distance.subprocess, 'run', run)
    commit = 'abcdef1234567'
    assert sign_distance._has_sign_distance(commit) is None
    assert sign_distance._has_sign_distance(commit) is None
    # Once in UNKNOWN_TTL_S for a commit the clone does not know; never a fetch of a missing object, never a prompt,
    # and a time limit.
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:2] == ['git', '-C'] and 'grep' in command and commit in command
    assert options['env']['GIT_NO_LAZY_FETCH'] == '1' and options['env']['GIT_TERMINAL_PROMPT'] == '0'
    assert options['timeout'] == sign_distance.GIT_TIMEOUT_S
    Answer.returncode = 0
    assert sign_distance._has_sign_distance('1234567abcdef') is True and len(calls) == 2
    assert sign_distance._has_sign_distance('1234567abcdef') is True and len(calls) == 2


def _clone(tmp_path: Path) -> tuple[Path, str, str]:
    """A small sunnypilot clone: a commit without signDistance, then one with it."""
    clone = tmp_path / 'clone'
    schema = clone / 'openpilot/cereal/custom.capnp'
    schema.parent.mkdir(parents=True)
    git = ['git', '-C', str(clone), '-c', 'user.name=t', '-c', 'user.email=t@t']
    subprocess.run(['git', 'init', '-q', str(clone)], check=True)
    schema.write_text('struct SpeedVisionSP { speedLimit @0 :Float32; }\n')
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-qm', 'old'], check=True)
    old = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    schema.write_text('struct SpeedVisionSP { speedLimit @0 :Float32; signDistance @1 :Float32; }\n')
    subprocess.run([*git, 'commit', '-qam', 'new'], check=True)
    new = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    return clone, old, new


def test_build_verdict_from_the_clone_then_the_dates(tmp_path, monkeypatch):
    clone, old, new = _clone(tmp_path)
    monkeypatch.setenv(sign_distance.SUNNYPILOT_ENV, str(clone))
    assert sign_distance.build_verdict({'build': {'commit': new}}, None) == 'new'
    assert sign_distance.build_verdict({'build': {'commit': old[:12]}}, '2026-11-02') == 'old'
    unknown = {'build': {'commit': 'abcdef1234567'}}
    assert sign_distance.build_verdict(unknown, '2026-11-02') == 'unknown'
    assert sign_distance.build_verdict({'build': {'commit': 'abcdef1234567',
                                                  'commit_date': "'1792000000 2026-10-17 10:00:00 +0000'"}},
                                       '2026-11-02') == 'old'
    assert sign_distance.build_verdict(unknown, '2026-10-29') == 'old'
    # Never a command line: a commit that is not hexadecimal is not looked up.
    assert sign_distance.build_verdict({'build': {'commit': '--output=/tmp/x'}}, None) == 'unknown'


# ---------- a drive read from its logs ----------

def test_a_drive_has_its_signs_and_vision_zero_from_exact_samples(tmp_path):
    comma_drive(tmp_path)
    store = scanned(tmp_path)
    merged = store.merged(ROUTE)
    # Every sample of the distance is kept (steps of 1 m: the old 2 m tolerance dropped them), unrounded (the float32
    # of the log), and the extract says so.
    run = [value for t, value in merged['series']['vis_sign_dist'] if BOOT_NS + 2 * NS <= t <= BOOT_NS + 7_300_000_000]
    assert len(run) == 27 and run[-2:] == [pytest.approx(0.4, abs=1e-6), 0.0] and run[-2] != 0.4
    assert merged['meta'][drive_logs.SIGN_DIST_EXACT] == drive_logs.SIGN_DIST_VERSION == 2
    assert merged['meta']['is_metric'] is True
    assert merged['meta']['build']['commit_date'].startswith("'1793350800")
    view = sign_distance.drive_view(store, ROUTE)
    assert view['status'] == 'signs' and view['message'] is None
    assert (view['unit'], view['unit_known']) == ('m', True)
    first, second = view['signs']
    assert first['included'] and first['end'] == 'zero'
    assert (first['start_s'], first['zero_s'], first['d0_s'], first['last_seen_s']) == (2.005, 7.005, 7.205, 6.005)
    assert first['segments'] == [0] and first['raw_class'] == 'maximum_speed:30'
    # The window: 5 s before the last sighting to 5 s after the zero, across the next minute.
    assert first['frames'] == len(range(21, 200)) + len(range(0, 41))
    assert (second['included'], second['excluded'], second['end']) == (False, 'cut', 'cut')
    assert second['excluded_message'] == sign_distance.EXCLUDED_MESSAGES['cut']
    assert view['counts'] == {'signs': 2, 'included': 1, 'excluded': {'cut': 1}, 'zero': 0, 'cant_tell': 0,
                              'orphans': 0}
    # A commit the clone does not know, but a distance in its logs: the build has it.
    assert view['build']['commit'] == NEW_COMMIT and view['build']['verdict'] == 'new'


def test_an_imperial_drive_reaches_zero_a_sample_later(tmp_path):
    comma_drive(tmp_path, metric=False)
    view = sign_distance.drive_view(scanned(tmp_path), ROUTE)
    assert (view['unit'], view['unit_known']) == ('ft', True)
    assert view['signs'][0]['zero_s'] == 7.205


def test_a_drive_without_the_unit_in_its_logs_is_metric_and_says_so(tmp_path):
    comma_drive(tmp_path, metric=None)
    view = sign_distance.drive_view(scanned(tmp_path), ROUTE)
    assert (view['unit'], view['unit_known']) == ('m', False)


def test_sign_view_lists_its_frames_series_and_sync(tmp_path):
    comma_drive(tmp_path)
    store = scanned(tmp_path)
    view = sign_distance.drive_view(store, ROUTE)
    sign = sign_distance.sign_view(store, ROUTE, key_of(view, 1))
    window = sign['window']
    assert (window['from_s'], window['to_s']) == (1.005, 12.005)
    frame = next(item for item in window['frames'] if (item['segment'], item['frame']) == (0, MARK_FRAME))
    # At 6.5 s vision shows the sample of 6.4 s: 3.4 m, 3 on the screen.
    assert frame == {'segment': 0, 'frame': MARK_FRAME, 't_s': 6.5, 'vision_m': 3.4, 'shown': 3}
    assert sign['series']['speed_kmh'][0][1] == 18.0 and sign['series']['vision_m']
    # The placeholder videos cannot be counted: their sync is unknown, which the page says.
    assert {item['precision'] for item in sign['sync']} == {'unknown'}
    assert {item['code'] for item in sign['warnings']} == {'sync_unknown'}
    assert (sign['prev'], sign['next'], sign['position'], sign['of']) == (None, None, 1, 1)
    assert window['wide_segments'] == []
    excluded = sign_distance.sign_view(store, ROUTE, key_of(view, 2))
    assert excluded['window']['frames'] == [] and excluded['sign']['excluded'] == 'cut'


# ---------- the marks through the site ----------

def test_a_zero_mark_gives_late_s_vision_at_the_zero_speed_and_late_m(tmp_path):
    comma_drive(tmp_path)
    store = scanned(tmp_path)
    server = RunningServer(tmp_path)
    try:
        status, _response, data = get(server, f'/api/sign-distance/drives/{ROUTE}')
        view = json_of(data)
        assert status == 200 and view['ok'] and view['status'] == 'signs'
        status, answer = post(server, f'/api/sign-distance/drives/{ROUTE}/marks', mark_body(view))
        assert status == 200 and answer['ok'], answer
        metrics = answer['sign']['metrics']
        # Vision's zero (7.005 s) comes 0.505 s after the person's (6.5 s): late.
        assert metrics['late_s'] == 0.505
        assert (metrics['vision_m'], metrics['vision_shown'], metrics['v_kmh']) == (3.4, 3, 18.0)
        # Its last distance (1.4 m at 6.805 s) plus the 1.525 m driven from 6.5 s to then.
        assert metrics['late_m'] == pytest.approx(2.925, abs=0.01) and metrics['speed_missing'] is False
        assert metrics['late_m_note'] == sign_distance.LATE_M_NOTE
        assert answer['line'] == {**answer['line'], 'sign': key_of(view, 1), 'mark': 'zero', 'segment': 0,
                                  'frame': MARK_FRAME, 'sync': 'unknown', 'reviewer': 'local'}
        assert [item['code'] for item in answer['warnings']] == ['sync_unknown']
        # The time kept is the server's encode index's, never the browser's.
        line = json.loads(sign_distance.marks_path(tmp_path, ROUTE).read_text().splitlines()[0])
        assert line['t_ns'] == MARK_NS and line['sign_t_ns'] == int(key_of(view, 1))
        # Not in sync to the frame: kept, out of the medians.
        status, _response, data = get(server, '/api/sign-distance/summary')
        summary = json_of(data)['summary']
        assert summary['zero'] == 1 and summary['degraded_marks'] == 1 and summary['late_s']['count'] == 0
        status, _response, data = get(server, '/api/sign-distance/drives')
        drives = json_of(data)['drives']
        assert [(item['route'], item['status'], item['counts']['zero']) for item in drives] == [(ROUTE, 'signs', 1)]
        assert ZERO_NS - MARK_NS == 505_000_000
    finally:
        server.close()
    assert store.scan_state(ROUTE)['state'] in ('done', 'no_episodes')


def test_the_last_line_wins_clear_takes_it_away_and_undo_is_per_sign(tmp_path):
    comma_drive(tmp_path)
    scanned(tmp_path)
    server = RunningServer(tmp_path)
    marks = f'/api/sign-distance/drives/{ROUTE}/marks'
    undo = f'/api/sign-distance/drives/{ROUTE}/undo'
    try:
        view = json_of(get(server, f'/api/sign-distance/drives/{ROUTE}')[2])
        first, second = key_of(view, 1), key_of(view, 2)
        assert post(server, marks, mark_body(view))[0] == 200
        status, answer = post(server, marks, mark_body(view, 'cant_tell', frame=140))
        assert status == 200 and answer['sign']['mark']['mark'] == 'cant_tell' and answer['sign']['metrics'] is None
        # A zero on a sign left out is refused; a clear is always allowed.
        status, answer = post(server, marks, mark_body(view, number=2))
        assert status == 409 and answer['error'] == 'sign_excluded'
        status, answer = post(server, marks, {'sign': second, 'mark': 'clear', 'segment': None, 'frame': None})
        assert status == 200 and answer['sign']['mark'] is None
        status, answer = post(server, marks, {'sign': first, 'mark': 'clear', 'segment': None, 'frame': None})
        assert status == 200 and answer['sign']['mark'] is None
        # Undo of sign 1 takes back its last line (the clear), never sign 2's: a line appended, the file never
        # rewritten.
        status, answer = post(server, undo, {'sign': first})
        assert status == 200 and answer['undone'] == {'sign': first, 'mark': 'clear'}
        assert answer['sign']['mark']['mark'] == 'cant_tell'
        assert post(server, undo, {'sign': first})[1]['undone'] == {'sign': first, 'mark': 'cant_tell'}
        status, answer = post(server, undo, {'sign': first})
        assert status == 200 and answer['undone'] == {'sign': first, 'mark': 'zero'} and answer['sign']['mark'] is None
        status, answer = post(server, undo, {'sign': first})
        assert status == 409 and answer['error'] == 'nothing_to_undo'
        status, answer = post(server, undo, {'sign': '42'})
        assert status == 404 and answer['error'] == 'unknown_sign'
        lines = [json.loads(text) for text in sign_distance.marks_path(tmp_path, ROUTE).read_text().splitlines()]
        assert [(line['sign'], line['mark'], line.get('undoes')) for line in lines] == [
            (first, 'zero', None), (first, 'cant_tell', None), (second, 'clear', None), (first, 'clear', None),
            (first, 'undo', 3), (first, 'undo', 1), (first, 'undo', 0)]
        assert all(line['reviewer'] == 'local' and line['at'] for line in lines)
        assert lines[5]['t_ns'] == lines[1]['t_ns'] and lines[5]['sign_t_ns'] == int(first)
        # Sign 2 keeps its clear; a new mark of sign 1 after the undos is in effect, and Undo takes it back.
        view = json_of(get(server, f'/api/sign-distance/drives/{ROUTE}')[2])
        assert view['signs'][1]['mark'] is None and view['counts']['zero'] == 0
        assert post(server, marks, mark_body(view))[0] == 200
        assert post(server, undo, {'sign': first})[1]['undone'] == {'sign': first, 'mark': 'zero'}
        lines = [json.loads(text) for text in sign_distance.marks_path(tmp_path, ROUTE).read_text().splitlines()]
        assert lines[-1]['undoes'] == 7
    finally:
        server.close()


def test_a_mark_body_is_exactly_its_four_keys_and_a_frame_of_the_sign(tmp_path):
    comma_drive(tmp_path)
    scanned(tmp_path)
    server = RunningServer(tmp_path)
    marks = f'/api/sign-distance/drives/{ROUTE}/marks'
    try:
        view = json_of(get(server, f'/api/sign-distance/drives/{ROUTE}')[2])
        # The time is never the browser's: a body with t_ns is refused.
        for body in ({**mark_body(view), 't_ns': 1}, {**mark_body(view), 'frame': '130'},
                     {**mark_body(view), 'mark': 'here'}, {**mark_body(view), 'mark': 'undo'},
                     {**mark_body(view), 'frame': True},
                     {'sign': key_of(view, 1), 'mark': 'clear', 'segment': 0, 'frame': None}, [1, 2]):
            status, answer = post(server, marks, body)
            assert (status, answer['error']) == (400, 'bad_body'), body
        # Outside the window of the sign (frame 10 is at 0.5 s), or past the minute's frames.
        for segment, frame in ((0, 10), (1, 199), (2, 0)):
            status, answer = post(server, marks, mark_body(view, segment=segment, frame=frame))
            assert (status, answer['error']) == (400, 'bad_frame')
        status, answer = post(server, marks, {**mark_body(view), 'sign': '123'})
        assert (status, answer['error']) == (404, 'unknown_sign')
        status, answer = post(server, '/api/sign-distance/drives/00000009--abcdef1234/marks', mark_body(view))
        assert (status, answer['error']) == (404, 'unknown_route')
        status, _response, data = get(server, marks)
        assert status == 405
        assert not sign_distance.marks_path(tmp_path, ROUTE).exists()
    finally:
        server.close()


def test_only_a_signed_in_person_writes_marks_and_the_api_only_reads(tmp_path):
    comma_drive(tmp_path)
    scanned(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    marks = f'/api/sign-distance/drives/{ROUTE}/marks'
    agent = {'X-Review-Token': TOKEN}
    bearer = {'Authorization': 'Bearer ' + TOKEN}
    try:
        view = json_of(get(server, f'/api/sign-distance/drives/{ROUTE}', agent)[2])
        assert view['status'] == 'signs'
        # The access token alone (an agent): never a mark.
        status, answer = post(server, marks, mark_body(view), agent)
        assert (status, answer['error']) == (403, 'session_required')
        status, answer = post(server, f'/api/sign-distance/drives/{ROUTE}/undo', {'sign': key_of(view, 1)}, agent)
        assert (status, answer['error']) == (403, 'session_required')
        session = sign_in(server)
        status, answer = post(server, marks, mark_body(view), {**session, 'Origin': 'http://elsewhere.example'})
        assert (status, answer['error']) == (403, 'cross_origin')
        status, _response, data = server.request('POST', marks, body=json.dumps(mark_body(view)).encode(),
                                                 headers={**session, 'Content-Type': 'text/plain'})
        assert (status, json_of(data)['error']) == (415, 'unsupported_media_type')
        server.app.panel.actions_enabled = False
        status, answer = post(server, marks, mark_body(view), session)
        assert (status, answer['error']) == (403, 'actions_disabled')
        server.app.panel.actions_enabled = True
        status, answer = post(server, marks, mark_body(view), session)
        assert status == 200 and answer['line']['reviewer'] == ADMIN
        # REST: reading only, with the Bearer token; no write there.
        status, _response, data = get(server, '/api/v1/sign-distance', bearer)
        payload = json_of(data)
        assert status == 200 and payload['ok'] and payload['summary']['zero'] == 1 and payload['drives']
        status, _response, data = get(server, f'/api/v1/sign-distance/{ROUTE}', bearer)
        assert status == 200 and json_of(data)['signs'][0]['mark']['mark'] == 'zero'
        assert get(server, '/api/v1/sign-distance')[0] == 401
        for path in ('/api/v1/sign-distance', f'/api/v1/sign-distance/{ROUTE}',
                     f'/api/v1/sign-distance/{ROUTE}/marks'):
            status, answer = post(server, path, mark_body(view), bearer)
            assert (status, answer['error']) == (404, 'not_found')
        assert len(sign_distance.marks_path(tmp_path, ROUTE).read_text().splitlines()) == 1
        # The page itself, without a session: the sign-in page.
        status, _response, data = get(server, '/sign-distance')
        assert status == 401 and b'<html' in data.lower()
        status, _response, data = get(server, '/sign-distance', session)
        assert status == 200 and b'<html' in data.lower()
    finally:
        server.close()


def test_marks_of_a_sign_read_again_are_matched_by_time_or_listed(tmp_path):
    comma_drive(tmp_path)
    store = scanned(tmp_path)
    view = sign_distance.drive_view(store, ROUTE)
    first = key_of(view, 1)
    path = sign_distance.marks_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    lines = [
        # A key of an older reading of the logs whose frame is in sign 1's window: sign 1.
        {'sign': str(int(first) - 7), 'mark': 'zero', 'segment': 0, 'frame': MARK_FRAME, 't_ns': MARK_NS,
         'sign_t_ns': int(first) - 7, 'sync': 'frame', 'reviewer': 'local', 'at': '2026-11-03T10:00:00+01:00'},
        # A sign that is no longer found anywhere: kept and listed.
        {'sign': '99', 'mark': 'cant_tell', 'segment': 0, 'frame': 1, 't_ns': 99, 'sign_t_ns': 99, 'sync': 'frame',
         'reviewer': 'local', 'at': '2026-11-03T10:00:01+01:00'},
    ]
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines) + 'not json\n')
    view = sign_distance.drive_view(store, ROUTE)
    assert view['signs'][0]['mark']['mark'] == 'zero' and view['signs'][0]['metrics']['late_s'] == 0.505
    assert [(item['sign'], item['mark']) for item in view['orphans']] == [('99', 'cant_tell')]
    assert view['counts']['orphans'] == 1
    # Undo of the orphan takes back its own line, with a line of its own; sign 1 keeps its mark.
    answer = sign_distance.undo(store, ROUTE, {'sign': '99'}, 'local')
    assert answer['undone'] == {'sign': '99', 'mark': 'cant_tell'} and answer['sign'] is None
    view = sign_distance.drive_view(store, ROUTE)
    assert view['orphans'] == [] and view['signs'][0]['mark']['mark'] == 'zero'
    with pytest.raises(sign_distance.SignDistanceError) as caught:
        sign_distance.undo(store, ROUTE, {'sign': '99'}, 'local')
    assert caught.value.reason == 'nothing_to_undo'
    # The line that is not JSON stays in the file, and so do the lines taken back.
    text = path.read_text().splitlines()
    assert text[2] == 'not json' and len(text) == 4
    assert json.loads(text[3])['mark'] == 'undo' and json.loads(text[3])['undoes'] == 1
    # The old key's undo goes with its target: an undo of sign 1 takes back the moved zero.
    assert sign_distance.undo(store, ROUTE, {'sign': first}, 'local')['undone'] == {'sign': first, 'mark': 'zero'}
    assert sign_distance.drive_view(store, ROUTE)['signs'][0]['mark'] is None


def test_a_mark_moves_with_the_run_of_its_old_key_before_the_nearest_window():
    signs = [{'key': '100', 'start_ns': 100, 'end_ns': 1000, 'zero_ns': 1000, 'window': [0, 2000]},
             {'key': '1500', 'start_ns': 1500, 'end_ns': 3000, 'zero_ns': 3000, 'window': [1000, 4000]}]
    lines = [
        # Its old key (the start of the sign it was marked on) is inside the second run: that one, though its frame
        # is nearer the first one's zero.
        {'sign': '1510', 'mark': 'zero', 't_ns': 1100, 'sign_t_ns': 1510, '_line': 1},
        # No run holds its key: the window whose zero is nearest its frame.
        {'sign': '50', 'mark': 'zero', 't_ns': 1800, 'sign_t_ns': 50, '_line': 2},
        # As near the zero of both windows that hold its frame: none, never a guess.
        {'sign': '60', 'mark': 'cant_tell', 't_ns': 2000, 'sign_t_ns': 60, '_line': 3},
        # In no window at all: an orphan.
        {'sign': '70', 'mark': 'zero', 't_ns': 9000, 'sign_t_ns': 70, '_line': 4},
    ]
    assert sign_distance._resolve(lines, signs) == ['1500', '100', '60', '70']


# ---------- the seconds never shown, and the video ----------

def test_slow_only_seconds_are_never_listed_nor_served(tmp_path):
    comma_drive(tmp_path)
    store = scanned(tmp_path)
    (tmp_path / 'raw/routes' / ROUTE / '1' / 'fcamera.hevc').unlink()
    slow_only(tmp_path, 0, [3])
    view = sign_distance.drive_view(store, ROUTE)
    sign = view['signs'][0]
    assert sign['included'] and sign['frames'] == 20
    viewed = sign_distance.sign_view(store, ROUTE, sign['key'])
    frames = viewed['window']['frames']
    assert {item['frame'] // 20 for item in frames} == {3} and {item['segment'] for item in frames} == {0}
    # Minute 1 of its window has no video: said, its frames not shown.
    assert {(item['code'], item.get('segment')) for item in viewed['warnings']} == {('sync_unknown', 0),
                                                                                    ('sync_no_video', 1)}
    assert sign_distance.frame_allowed(store, ROUTE, 0, 70)
    assert not sign_distance.frame_allowed(store, ROUTE, 0, MARK_FRAME)
    server = RunningServer(tmp_path)
    try:
        status, _response, data = get(server, f'/api/sign-distance/drives/{ROUTE}/frames/0/{MARK_FRAME}.jpg')
        assert (status, json_of(data)['error']) == (404, 'unknown_frame')
        status, answer = post(server, f'/api/sign-distance/drives/{ROUTE}/marks', mark_body(view))
        assert (status, answer['error']) == (400, 'bad_frame')
    finally:
        server.close()
    # No second of the window kept: the sign is left out, none of its frames served.
    slow_only(tmp_path, 0, [0])
    view = sign_distance.drive_view(store, ROUTE)
    assert view['signs'][0]['excluded'] == 'slow_only' and view['signs'][0]['frames'] == 0
    assert not sign_distance.frame_allowed(store, ROUTE, 0, 10)
    assert sign_distance.sign_view(store, ROUTE, view['signs'][0]['key'])['window']['frames'] == []


def test_a_frame_is_allowed_from_the_signs_and_the_video_only(tmp_path, monkeypatch):
    comma_drive(tmp_path)
    store = scanned(tmp_path)

    def never(*_args, **_kwargs):
        raise AssertionError('not for a picture')
    # Every picture asks: never the scan state, the scan rule nor the marks.
    monkeypatch.setattr(store, 'scan_state', never)
    monkeypatch.setattr(store, 'needs_scan', never)
    monkeypatch.setattr(sign_distance, 'load_marks', never)
    assert sign_distance.frame_allowed(store, ROUTE, 0, MARK_FRAME)
    # Before the window of sign A (5 s before its last sighting) and in the window of B (cut): not shown.
    assert not sign_distance.frame_allowed(store, ROUTE, 0, 5)
    assert not sign_distance.frame_allowed(store, ROUTE, 1, 190)
    # The kept seconds are looked at on every call.
    slow_only(tmp_path, 0, [3])
    assert not sign_distance.frame_allowed(store, ROUTE, 0, MARK_FRAME)


def test_a_drive_with_logs_only_says_run_prepare(tmp_path):
    comma_drive(tmp_path, video=False)
    view = sign_distance.drive_view(scanned(tmp_path), ROUTE)
    assert view['status'] == 'no_video' and view['message'] == 'Logs only, no video yet: run Prepare.'
    assert view['signs'][0]['excluded'] == 'no_video'
    assert view['signs'][0]['excluded_message'] == 'Video not linked yet: run Prepare.'


def test_a_real_video_in_sync_counts_in_the_medians_and_serves_its_frames(tmp_path):
    comma_drive(tmp_path)
    real_video(tmp_path, 0)
    real_video(tmp_path, 1)
    real_video(tmp_path, 0, name='ecamera.hevc')
    store = scanned(tmp_path)
    server = RunningServer(tmp_path)
    try:
        view = json_of(get(server, f'/api/sign-distance/drives/{ROUTE}')[2])
        sign = json_of(get(server, f'/api/sign-distance/drives/{ROUTE}/signs/{key_of(view, 1)}')[2])
        assert {item['precision'] for item in sign['sync']} == {'frame'} and sign['warnings'] == []
        # The wide road camera only where the minute has its video, frame for frame.
        assert sign['window']['wide_segments'] == [0]
        status, answer = post(server, f'/api/sign-distance/drives/{ROUTE}/marks', mark_body(view))
        assert status == 200 and answer['line']['sync'] == 'frame' and answer['warnings'] == []
        summary = json_of(get(server, '/api/sign-distance/summary')[2])['summary']
        assert summary['late_s'] == {'median': 0.505, 'count': 1, 'min': 0.505, 'max': 0.505}
        assert summary['late_m']['count'] == 1 and summary['degraded_marks'] == 0
        assert summary['by_drive'][0]['late_s']['median'] == 0.505
        status, response, data = get(server, f'/api/sign-distance/drives/{ROUTE}/frames/0/{MARK_FRAME}.jpg')
        assert status == 200 and data[:2] == b'\xff\xd8'
        assert response.getheader('Content-Type') == 'image/jpeg'
        assert response.getheader('Cache-Control') == 'private, max-age=3600'
        status, response, wide = get(server, f'/api/sign-distance/drives/{ROUTE}/frames/0/{MARK_FRAME}-wide.jpg')
        assert status == 200 and wide[:2] == b'\xff\xd8' and response.getheader('Content-Type') == 'image/jpeg'
        status, _response, data = get(server, f'/api/sign-distance/drives/{ROUTE}/frames/1/10-wide.jpg')
        assert (status, json_of(data)['error']) == (404, 'no_video_frame')
        # A frame outside every window of an included sign is not served.
        status, _response, data = get(server, f'/api/sign-distance/drives/{ROUTE}/frames/0/5.jpg')
        assert (status, json_of(data)['error']) == (404, 'unknown_frame')
        status, _response, data = get(server, f'/api/sign-distance/drives/{ROUTE}/frames/0/5-wide.jpg')
        assert (status, json_of(data)['error']) == (404, 'unknown_frame')
    finally:
        server.close()
    assert store.merged(ROUTE)['segments'] == [0, 1]


# ---------- the drive statuses and the rule of scanning ----------

def test_statuses_before_and_after_reading_the_logs(tmp_path):
    comma_drive(tmp_path)
    store = drive_review.store_for(tmp_path)
    view = sign_distance.drive_view(store, ROUTE)
    assert view['status'] == 'unread' and store.needs_scan(ROUTE)
    assert [item['needs_scan'] for item in drive_review.drives_view(store)] == [True]
    store.scan(ROUTE)
    assert not store.needs_scan(ROUTE)
    assert [item['needs_scan'] for item in drive_review.drives_view(store)] == [False]
    (tmp_path / 'raw/routes' / '00000002--abcdef1234' / '0').mkdir(parents=True)
    overview = sign_distance.overview(store)
    assert [(item['route'], item['status']) for item in overview['drives']] == [
        ('00000002--abcdef1234', 'no_logs'), (ROUTE, 'signs')]
    assert overview['summary']['statuses'] == {'signs': 1, 'no_logs': 1}
    with pytest.raises(sign_distance.SignDistanceError) as caught:
        sign_distance.drive_view(store, '00000009--abcdef1234')
    assert caught.value.status == 404 and caught.value.reason == 'unknown_route'


OLD_DATE = "'1792000000 2026-10-17 10:00:00 +0000'"


def test_a_distance_in_the_logs_lists_the_signs_whatever_the_build_date(tmp_path):
    # Dated before signDistance by its commit (a cherry-pick, a clock), yet the logs have the distance: its signs.
    comma_drive(tmp_path, commit='abcdef1234567', commit_date=OLD_DATE)
    view = sign_distance.drive_view(scanned(tmp_path), ROUTE)
    assert view['status'] == 'signs' and view['build']['verdict'] == 'new' and view['counts']['included'] == 1


def test_a_build_too_old_and_a_drive_with_no_sign(tmp_path):
    # No distance in the logs and a build dated before signDistance: too old.
    old = tmp_path / 'old'
    folder = old / 'raw/routes' / ROUTE / '0'
    folder.mkdir(parents=True)
    (folder / 'rlog.zst').write_bytes(_no_sign_log(OLD_DATE))
    view = sign_distance.drive_view(scanned(old), ROUTE)
    assert view['status'] == 'build_too_old' and view['signs'] == [] and view['build']['verdict'] == 'old'
    # A build not known here, dated after signDistance, and no distance at all: no sign seen, and the message says
    # the build may be the reason (its commit's first 9 characters).
    empty = tmp_path / 'empty'
    folder = empty / 'raw/routes' / ROUTE / '0'
    folder.mkdir(parents=True)
    (folder / 'rlog.zst').write_bytes(_no_sign_log())
    view = sign_distance.drive_view(scanned(empty), ROUTE)
    assert view['status'] == 'no_sign_seen' and view['build']['verdict'] == 'unknown'
    assert 'abcdef123)' in view['message'] and 'may be older' in view['message']


def _no_sign_log(commit_date: str = "'1793350800 2026-11-02 08:00:00 +0000'") -> bytes:
    """A minute whose vision never shows a distance: segment 0 with its sign A taken away."""
    schema = _schema()
    events = []
    for event in schema.Event.read_multiple_bytes(zstandard.ZstdDecompressor().decompress(
            segment_log(0, commit='abcdef1234567', commit_date=commit_date), max_output_size=1 << 26)):
        builder = event.as_builder()
        if builder.which() == 'speedVisionSP':
            builder.speedVisionSP.signDistance = 0.0
            builder.speedVisionSP.signSize = 0.0
        events.append(builder.to_bytes())
    return zstandard.ZstdCompressor(level=3).compress(b''.join(events))


def test_logs_that_cannot_be_read_are_their_own_state_and_not_scanned_again(tmp_path, monkeypatch):
    comma_drive(tmp_path)
    store = drive_review.store_for(tmp_path)
    real = drive_logs.sha256_file

    def refused(path):
        raise PermissionError(13, 'Permission denied', str(path))
    monkeypatch.setattr(drive_logs, 'sha256_file', refused)
    monkeypatch.setattr(drive_review, '_readable', lambda path: False)
    assert store.scan(ROUTE)['state'] == 'unreadable_logs'
    state = store.scan_state(ROUTE)
    assert [item['error'] for item in state['unreadable_files']] == ['PermissionError: Permission denied'] * 2
    assert not store.needs_scan(ROUTE)
    view = sign_distance.drive_view(store, ROUTE)
    assert view['status'] == 'unreadable' and len(view['unreadable_files']) == 2
    assert view['message'] == 'Logs found, but this server cannot open them (their file permissions).'
    # Find episodes of every drive skips it (the one rule of needs_scan) ...
    server = RunningServer(tmp_path)
    try:
        status, answer = post(server, '/api/drive-review/scan-all', {})
        assert status == 200 and answer['queued'] == []
    finally:
        server.close()
    # ... until it can be opened.
    monkeypatch.setattr(drive_review, '_readable', lambda path: True)
    assert store.needs_scan(ROUTE)
    monkeypatch.setattr(drive_logs, 'sha256_file', real)
    assert store.scan(ROUTE)['state'] in ('done', 'no_episodes')
    assert store.scan_state(ROUTE)['unreadable_files'] == []


def test_an_extract_from_before_the_exact_distance_is_read_again(tmp_path):
    comma_drive(tmp_path)
    store = scanned(tmp_path)
    files = store.summary(ROUTE)['files']
    item = files['0']
    path = drive_logs.extract_path(tmp_path, ROUTE, 0, item['sha256'])
    value = json.loads(gzip.decompress(path.read_bytes()))
    value['meta'].pop(drive_logs.SIGN_DIST_EXACT)
    value['series']['vis_sign_dist'] = [[t, round(d, 2)] for t, d in value['series']['vis_sign_dist']]
    time.sleep(0.01)
    path.write_bytes(gzip.compress(json.dumps(value).encode()))
    assert drive_logs.stale_extract(value)
    assert store.scan_state(ROUTE)['state'] == 'outdated' and store.needs_scan(ROUTE)
    view = sign_distance.drive_view(store, ROUTE)
    assert view['status'] == 'unread' and view['message'] == sign_distance.OUTDATED_MESSAGE
    assert store.scan(ROUTE)['state'] in ('done', 'no_episodes')
    again = json.loads(gzip.decompress(path.read_bytes()))
    assert again['meta'][drive_logs.SIGN_DIST_EXACT] == drive_logs.SIGN_DIST_VERSION
    assert store.scan_state(ROUTE)['state'] != 'outdated'
    # An old extract without any distance (an older build) is not read again.
    assert not drive_logs.stale_extract({'meta': {}, 'series': {'vis_sign_dist': [[1, 0.0]]}})
    # The first marker (True: every change kept, rounded to the centimetre) is read again, a newer one is not.
    positive = {'series': {'vis_sign_dist': [[1, 3.0], [2, 0.0]]}}
    assert drive_logs.stale_extract({**positive, 'meta': {drive_logs.SIGN_DIST_EXACT: True}})
    assert drive_logs.stale_extract({**positive, 'meta': {drive_logs.SIGN_DIST_EXACT: 1}})
    assert not drive_logs.stale_extract({**positive, 'meta': {drive_logs.SIGN_DIST_EXACT: 2}})
