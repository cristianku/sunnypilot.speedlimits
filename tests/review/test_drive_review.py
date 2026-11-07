"""The drive review (Data › Drive review): the comma's logs of a recorded drive added to the portal, joined to its video,
scanned for episodes and annotated by a person (nnslr_tools.drive_logs, drive_extract, drive_episodes, drive_review).

Every log here is SYNTHETIC: written by these tests with a made-up schema (fixtures/drive_schema, the field names of the
runtime branch only) for a made-up drive of three minutes. Nothing here is a real drive or a person's review.

The made-up drive (seconds from its start; openpilot engaged throughout, 85 km/h, set speed 100):
- the comma's vision limit 80 km/h; "not sure" (0) from 120 s; not logged at all from 150 s;
- the map's limit 80 km/h, 120 km/h from 20 s to 30 s and from 120 s on: a disagreement without an intervention at
  20-30 s; none after 120 s, where the vision is missing;
- the brake at 58.5 s (the first minute) and the accelerator at 62 s (the second): one episode across two minutes;
- the road camera's encode index at 20 frames a second; the GPS without a fix (no civil time) unless a test gives one.
"""
from __future__ import annotations

import functools
import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

capnp = pytest.importorskip('capnp')
zstandard = pytest.importorskip('zstandard')

from nnslr_tools import drive_episodes, drive_logs, drive_review, drive_video  # noqa: E402
from review_helpers import ROUTE, OTHER_ROUTE, RunningServer, json_of, make_report  # noqa: E402

SCHEMA = Path(__file__).parent / 'fixtures' / 'drive_schema'
NS = 1_000_000_000
# The comma's monotonic clock when the made-up drive starts: any value (it counts from the comma's boot).
BOOT_NS = 7_345 * NS
SEGMENT_NS = 60 * NS
SEGMENTS = (0, 1, 2)
NO_LOGS_TEXT = 'Logs not available: import the logs to look for the episodes of the recorded drive.'
CONTROL_JS = Path(__file__).resolve().parents[2] / 'src' / 'nnslr_tools' / 'web' / 'js' / 'control.js'


# ---------- the synthetic logs ----------

def _vision(s: float) -> float | None:
    """The comma's vision limit (km/h) at s; 0: not sure; None: speedVisionSP not in the log."""
    return 80 if s < 120 else 0 if s < 150 else None


def _map(s: float) -> float:
    return 120 if 20 <= s < 30 or s >= 120 else 80


def _pressed(s: float, spans) -> bool:
    return any(a <= s < b for a, b in spans)


BRAKE = ((58.5, 59.0),)
GAS = ((62.0, 63.0),)
LEAD = ((10.0, 20.0),)


@functools.lru_cache(maxsize=None)
def _schema():
    return capnp.load(str(SCHEMA / 'log.capnp'), imports=[str(SCHEMA)])


@functools.lru_cache(maxsize=None)
def segment_log(segment: int, gps_unix_ms: int | None = None, skip_frames: tuple = ()) -> bytes:
    """A SYNTHETIC rlog of one minute of the made-up drive, zstd-compressed as the comma writes it."""
    schema = _schema()
    start = BOOT_NS + segment * SEGMENT_NS
    messages: list[tuple[int, bytes]] = []

    def put(t: int, fill) -> None:
        event = schema.Event.new_message()
        event.logMonoTime = t
        event.valid = True
        fill(event)
        messages.append((t, event.to_bytes()))

    def init(event):
        data = event.init('initData')
        data.version, data.gitBranch, data.gitCommit, data.deviceType = '0.0-synthetic', 'synthetic', '0' * 40, 'mici'
    put(start, init)

    def params(event):
        car = event.init('carParams')
        car.brand, car.carFingerprint, car.dashcamOnly, car.openpilotLongitudinalControl = 'synthetic', 'SYNTHETIC', False, True
    put(start, params)
    for k in range(600):
        t = start + k * NS // 10
        s = (t - BOOT_NS) / NS

        def car_state(event, s=s):
            car = event.init('carState')
            car.vEgo = 85 / 3.6
            car.vEgoCluster = 88 / 3.6
            car.vCruise = 100
            car.gasPressed = _pressed(s, GAS)
            car.brakePressed = _pressed(s, BRAKE)
            car.steeringPressed = False
            cruise = car.init('cruiseState')
            cruise.enabled, cruise.available, cruise.speed = True, True, 100 / 3.6

        def selfdrive(event):
            state = event.init('selfdriveState')
            state.state, state.enabled, state.active = 'enabled', True, True

        def control(event):
            control = event.init('carControl')
            control.latActive, control.longActive = True, True
        put(t + 1_000_000, car_state)
        put(t + 2_000_000, selfdrive)
        put(t + 3_000_000, control)
        if k % 5 == 0:
            def plan(event, s=s):
                plan = event.init('longitudinalPlanSP')
                plan.vTarget = 85 / 3.6
                resolver = plan.init('speedLimit').init('resolver')
                vision = _vision(s)
                resolver.source = 'vision' if vision else 'map'
                resolver.speedLimit = (vision or _map(s)) / 3.6
            put(t + 4_000_000, plan)

            def radar(event, s=s):
                lead = event.init('radarState').init('leadOne')
                # Without a car ahead the numbers stay in the message, meaning nothing.
                lead.dRel, lead.vRel, lead.vLead = 30.0, -5.0, 20.0
                lead.present = _pressed(s, LEAD)
            put(t + 4_500_000, radar)
        if k % 10 == 0:
            vision = _vision(s)
            if vision is not None:
                def speed_vision(event, vision=vision, s=s):
                    sign = event.init('speedVisionSP')
                    sign.speedLimit, sign.modelName, sign.roadType = vision / 3.6, 'synthetic', 'rural'
                    sign.signRead = s == 5
                    # The first 3 s from a build before these fields: none of them written.
                    if s < 3:
                        return
                    sign.speedLimitState = 'limit' if vision else 'unknown'
                    sign.speedLimitReason = 'sign' if s >= 5 else ''
                    # An 80 read and taken at 5 s, a 120 read at 30 s that the memory dropped (too unsure).
                    if s >= 5:
                        taken = s < 30
                        sign.rawSignClass = 'maximum_speed:80' if taken else 'maximum_speed:120'
                        sign.rawSignSpeed, sign.rawSignConfidence, sign.rawSignTaken = (80, 0.93, True) if taken else (120, 0.4, False)
                    # The 80 ahead, nearer each second until it is passed at 5 s.
                    sign.signDistance = max(0.0, (5 - s) * 85 / 3.6)
                    sign.signSize = 40.0 if s < 5 else 0.0
                put(t + 5_000_000, speed_vision)

            def live_map(event, s=s):
                data = event.init('liveMapDataSP')
                data.speedLimitValid, data.speedLimit, data.roadName = True, _map(s) / 3.6, 'Synthetic road'
            put(t + 6_000_000, live_map)

            def gps(event, s=s):
                fix = event.init('gpsLocationExternal')
                fix.hasFix = gps_unix_ms is not None
                if gps_unix_ms is not None:
                    fix.latitude, fix.longitude = 47.0, 8.0
                    fix.unixTimestampMillis = gps_unix_ms + round(s * 1000)
            put(t + 7_000_000, gps)
    for frame in range(1200):
        if frame in skip_frames:
            continue
        sof = start + frame * NS // 20

        def encode(event, frame=frame, sof=sof):
            index = event.init('roadEncodeIdx')
            index.frameId, index.segmentNum, index.segmentId = segment * 1200 + frame, segment, frame
            index.timestampSof, index.timestampEof = sof, sof + 20_000_000
        put(sof + 25_000_000, encode)
    messages.sort(key=lambda item: item[0])
    return zstandard.ZstdCompressor(level=3).compress(b''.join(data for _t, data in messages))


def comma_drive(root: Path, route: str = ROUTE, segments=SEGMENTS, *, logs: bool = True, **options) -> None:
    """The made-up drive as the comma's download leaves it: per minute a road video (a placeholder: never decoded
    here) and, with logs, its rlog."""
    for segment in segments:
        folder = root / 'raw' / 'routes' / route / str(segment)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'fcamera.hevc').write_bytes(b'synthetic hevc ' * (segment + 1))
        if logs:
            (folder / 'rlog.zst').write_bytes(segment_log(segment, **options))


def fake_video(root: Path, route: str = ROUTE) -> None:
    """The index of the drive's video as drive_video.make writes it, 1200 frames a minute (no video is made)."""
    parts = drive_video.minutes(root, route)
    video, index = drive_video.paths(root, route)
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b'synthetic mp4')
    segments = [{'segment': segment, 'start': 60.0 * position, 'frames': 1200} for position, (segment, _p) in enumerate(parts)]
    index.write_text(json.dumps({'route': route, 'fps': 20, 'width': 960, 'segments': segments,
                                 'duration': 60.0 * len(parts), 'source': drive_video._signature(parts)}))


@pytest.fixture(autouse=True)
def synthetic_schema(tmp_path, monkeypatch):
    monkeypatch.setenv(drive_logs.SCHEMA_ENV, str(SCHEMA))
    # Never the server's real shared folder of comma downloads.
    shared = tmp_path / 'shared'
    shared.mkdir()
    monkeypatch.setenv('COMMA_VIDEOS_DIR', str(shared))
    return shared


def settled(store: drive_review.DriveReviewStore, route: str = ROUTE, timeout: float = 90) -> dict:
    """The scan state once the background worker has nothing left for the drive."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = store.scan_state(route)
        if route not in store._queued and store._running != route and state['state'] not in ('queued', 'scanning'):
            return state
        time.sleep(0.05)
    raise AssertionError('the scan did not finish: ' + json.dumps(store.scan_state(route)))


def upload(server, route: str, name: str, data: bytes, segment=None):
    query = 'name=' + name.replace('/', '%2F') + ('' if segment is None else f'&segment={segment}')
    return server.request('POST', f'/api/drive-review/drives/{route}/logs?{query}', body=data,
                          headers={'Content-Type': 'application/octet-stream'})


def active(store, route: str = ROUTE) -> list[dict]:
    return [item for item in store.episodes(route) if item['channel'] != 'spot_check']


# ---------- import ----------

def test_a_drive_without_logs_says_so_and_its_video_review_still_works(tmp_path):
    make_report(tmp_path)
    comma_drive(tmp_path, logs=False)
    fake_video(tmp_path)
    store = drive_review.store_for(tmp_path)
    assert store.scan_state(ROUTE)['state'] == 'no_logs'
    # Scanning a drive without logs finds nothing and invents nothing.
    assert store.scan(ROUTE)['state'] == 'no_logs' and store.episodes(ROUTE) == []
    server = RunningServer(tmp_path)
    try:
        drives = json_of(server.request('GET', '/api/drive-review/drives')[2])['drives']
        drive = next(item for item in drives if item['route'] == ROUTE)
        assert drive['scan']['state'] == 'no_logs' and drive['logs_found'] == 0 and drive['video_segments'] == 3
        view = json_of(server.request('GET', f'/api/drive-review/drives/{ROUTE}/timeline')[2])
        # The video plays without logs; there is no timeline of the comma and no episode.
        assert view['video']['state'] == 'ready' and view['t0_ns'] is None and view['episodes'] == []
        assert view['scan']['state'] == 'no_logs' and view['when_certain'] is False
        status, _, data = server.request('GET', '/drive-review')
        assert status == 200 and '<title>Speed limits · Drive review</title>' in data.decode()
        # The frame review of the same drive is untouched.
        assert server.request('GET', f'/api/route/{ROUTE}/frames')[0] in (200, 404)
    finally:
        server.close()
    assert NO_LOGS_TEXT in CONTROL_JS.read_text(encoding='utf-8')


def test_import_states_from_found_to_scanned_to_outdated_and_unreadable(tmp_path):
    comma_drive(tmp_path, segments=(0, 1))
    store = drive_review.store_for(tmp_path)
    assert store.scan_state(ROUTE)['state'] == 'no_logs'
    assert store.import_logs(ROUTE) == {'found': 2, 'new': 2, 'files': 2}
    assert store.scan_state(ROUTE)['state'] == 'not_scanned'
    assert store.scan(ROUTE)['state'] == 'done'
    # A minute added later: the scan is out of date until it runs again.
    comma_drive(tmp_path, segments=(2,))
    store.import_logs(ROUTE)
    assert store.scan_state(ROUTE)['state'] == 'outdated'
    assert store.scan(ROUTE)['state'] == 'done' and store.summary(ROUTE)['segments'] == [0, 1, 2]
    # A drive whose only log cannot be read: failed, with why, never "no episodes".
    broken = tmp_path / 'raw' / 'routes' / OTHER_ROUTE / '0'
    broken.mkdir(parents=True)
    (broken / 'rlog.zst').write_bytes(b'not a log')
    state = store.scan(OTHER_ROUTE)
    assert state['state'] == 'failed' and 'could be read' in state['error']
    assert [item['state'] for item in store.log_files(OTHER_ROUTE)] == ['unreadable']


def test_logs_added_later_from_the_portal_are_kept_once_and_scanned(tmp_path):
    make_report(tmp_path)
    comma_drive(tmp_path, logs=False)
    store = drive_review.store_for(tmp_path)
    server = RunningServer(tmp_path)
    try:
        # As the comma names its folders, as comma connect names its files, and a bare rlog with its segment.
        for name, segment, minute in ((f'{ROUTE}--0/rlog.zst', None, 0), (f'{ROUTE}--1--rlog.zst', None, 1),
                                      ('rlog.zst', 2, 2)):
            status, _, data = upload(server, ROUTE, name, segment_log(minute), segment)
            assert status == 200 and json_of(data)['added'] is True, data
        assert settled(store)['state'] == 'done'
        assert [(item['segment'], item['source'], item['state']) for item in store.log_files(ROUTE)] == [
            (0, 'upload', 'read'), (1, 'upload', 'read'), (2, 'upload', 'read')]
        # The same file again: recorded and stored once.
        status, _, data = upload(server, ROUTE, f'{ROUTE}--0/rlog.zst', segment_log(0))
        assert status == 200 and json_of(data)['added'] is False
        assert len(list((tmp_path / 'raw' / 'drive_logs' / ROUTE / '0').iterdir())) == 1
        assert settled(store)['state'] == 'done' and len(store.log_files(ROUTE)) == 3
        # Refused: a log of another drive, a drive the server does not have, a log without its segment, not raw.
        assert upload(server, ROUTE, f'{OTHER_ROUTE}--0--rlog.zst', b'x')[0] == 400
        assert upload(server, OTHER_ROUTE, f'{OTHER_ROUTE}--0--rlog.zst', b'x')[0] == 404
        assert upload(server, ROUTE, 'rlog.zst', b'x')[0] == 400
        assert server.request('POST', f'/api/drive-review/drives/{ROUTE}/logs?name=rlog.zst&segment=0',
                              body={'x': 1})[0] == 415
        drives = json_of(server.request('GET', '/api/drive-review/drives')[2])['drives']
        assert next(item for item in drives if item['route'] == ROUTE)['episodes'] == 2
        queue = json_of(server.request('GET', '/api/drive-review/episodes?kind=disagreement')[2])['episodes']
        assert [item['channel'] for item in queue] == ['disagreement']
        assert server.request('GET', '/api/drive-review/episodes?kind=nonsense')[0] == 400
    finally:
        server.close()


def test_importing_again_records_nothing_twice(tmp_path, synthetic_schema):
    comma_drive(tmp_path)
    # The same minute also in the shared folder of comma downloads: another path, the same content.
    copy = synthetic_schema / f'{ROUTE}--0'
    copy.mkdir()
    (copy / 'rlog.zst').write_bytes(segment_log(0))
    store = drive_review.store_for(tmp_path)
    assert store.import_logs(ROUTE) == {'found': 4, 'new': 3, 'files': 3}
    assert store.import_logs(ROUTE)['new'] == 0
    store.scan(ROUTE)
    first = sorted(item['id'] for item in store.episodes(ROUTE))
    # Uploading a file the comma's download already has keeps no second copy.
    added = store.add_upload(ROUTE, 1, 'rlog', '.zst', io.BytesIO(segment_log(1)), len(segment_log(1)))
    assert added['added'] is False and not list((tmp_path / 'raw' / 'drive_logs' / ROUTE / '1').glob('rlog-*'))
    settled(store)
    store.scan(ROUTE)
    assert len(store.log_files(ROUTE)) == 3
    assert sorted(item['id'] for item in store.episodes(ROUTE, active=False)) == first


# ---------- sync ----------

def test_the_video_is_joined_on_the_monotonic_clock_not_on_file_times(tmp_path):
    make_report(tmp_path)
    comma_drive(tmp_path)
    # File times that lie: the last minute "oldest", all in 1999. Nothing may follow them.
    for position, segment in enumerate(SEGMENTS):
        for name in ('rlog.zst', 'fcamera.hevc'):
            moment = 915_000_000 - position * 3600
            os.utime(tmp_path / 'raw' / 'routes' / ROUTE / str(segment) / name, (moment, moment))
    fake_video(tmp_path)
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    server = RunningServer(tmp_path)
    try:
        view = json_of(server.request('GET', f'/api/drive-review/drives/{ROUTE}/timeline')[2])
    finally:
        server.close()
    assert view['t0_ns'] == BOOT_NS
    assert [(item['segment'], item['start'], item['video_start'], item['precision'], item['max_error_ms'])
            for item in view['segments']] == [(0, 0.0, 0.0, 'frame', 0.0), (1, 60.0, 60.0, 'frame', 0.0),
                                              (2, 120.0, 120.0, 'frame', 0.0)]
    # No GPS fix in the logs: no civil time, and the date shown says it is uncertain.
    assert view['civil_ms'] is None and view['when_certain'] is False


def test_the_timeline_says_why_the_vision_limit_is_what_it_is(tmp_path):
    make_report(tmp_path)
    comma_drive(tmp_path)
    fake_video(tmp_path)
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    server = RunningServer(tmp_path)
    try:
        view = json_of(server.request('GET', f'/api/drive-review/drives/{ROUTE}/timeline')[2])
    finally:
        server.close()
    series = view['series']

    def at(name, s):
        points = [value for t, value in series[name] if t <= s]
        return points[-1] if points else None
    # Not written by the build: not in the log, not an empty reason.
    assert at('vis_state', 1.5) is None and at('vis_reason', 1.5) is None and at('vis_sign_dist', 1.5) is None
    assert at('vis_state', 10) == 'limit' and at('vis_reason', 3.5) == '' and at('vis_reason', 10) == 'sign'
    assert (at('vis_raw_class', 10), at('vis_raw_speed', 10), at('vis_raw_conf', 10), at('vis_raw_taken', 10)) == \
        ('maximum_speed:80', 80, 0.93, True)
    assert (at('vis_raw_class', 40), at('vis_raw_speed', 40), at('vis_raw_conf', 40), at('vis_raw_taken', 40)) == \
        ('maximum_speed:120', 120, 0.4, False)
    assert 30 < at('vis_sign_dist', 3.5) < 50 and at('vis_sign_dist', 10) == 0
    assert at('vis_sign_size', 3.5) == 40 and at('vis_sign_size', 10) == 0
    # Where speedVisionSP stops (150 s on), its series end: a missing value is never carried on.
    assert at('vis_reason', 170) is None and at('vis_raw_class', 170) is None


def test_the_car_ahead_has_numbers_only_while_it_is_there(tmp_path):
    comma_drive(tmp_path, segments=(0,))
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    series = store.merged(ROUTE)['series']

    def at(name, s):
        return [value for t, value in series[name] if t <= BOOT_NS + s * NS][-1]
    assert at('lead_present', 5) is False and at('lead_present', 15) is True
    assert (at('lead_d', 15), at('lead_v', 15), at('lead_v_rel', 15)) == (30.0, 72.0, -18.0)
    # No car ahead: nothing known, never 0 or the numbers of the message.
    for name in ('lead_d', 'lead_v', 'lead_v_rel'):
        assert at(name, 5) is None and at(name, 25) is None


def test_the_dashboard_speed_is_the_logs_own_or_vego_plus_the_cars_offset(tmp_path):
    comma_drive(tmp_path, segments=(0,))
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    merged = store.merged(ROUTE)
    assert {value for _, value in merged['series']['v_ego_cluster']} == {88.0}
    assert {value for _, value in drive_logs.dashboard_speed(merged)} == {88.0}
    peugeot = {'meta': {'car': {'fingerprint': 'PSA_PEUGEOT_3008'}}}
    own = [[0, 0.2], [10, 50.0], [20, 80.0], [30, None]]
    # A log from before vEgoCluster was kept, and one where it is a copy of vEgo: vEgo + 3 on the Peugeot, 0 when still.
    for series in ({'v_ego': own}, {'v_ego': own, 'v_ego_cluster': own}):
        assert drive_logs.dashboard_speed({**peugeot, 'series': series}) == [[0, 0.0], [10, 53.0], [20, 83.0], [30, None]]
    # A car with no offset known stays at vEgo.
    assert drive_logs.dashboard_speed({'series': {'v_ego': own}}) == [[0, 0.0], [10, 50.0], [20, 80.0], [30, None]]
    # Its own (the car's cluster speed): taken as it is, and vEgo + 3 only on a minute extracted without it.
    cluster = [[0, 0.0], [10, 53.0], [15, None]]
    assert drive_logs.dashboard_speed({**peugeot, 'series': {'v_ego': own, 'v_ego_cluster': cluster}}) == [
        [0, 0.0], [10, 53.0], [20, 83.0], [30, None]]
    toyota = [[0, 0.0], [10, 50.75], [15, None]]
    assert drive_logs.dashboard_speed({**peugeot, 'series': {'v_ego': own, 'v_ego_cluster': toyota}})[1] == [10, 50.75]


def test_holes_in_the_encode_index_degrade_the_sync_and_the_gps_gives_the_date(tmp_path):
    comma_drive(tmp_path, segments=(0,), gps_unix_ms=1_790_000_000_000)
    comma_drive(tmp_path, segments=(1,), gps_unix_ms=1_790_000_000_000, skip_frames=tuple(range(100, 110)))
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    merged = store.merged(ROUTE)
    index = {'segments': [{'segment': 0, 'start': 0.0, 'frames': 1200}, {'segment': 1, 'start': 60.0, 'frames': 1200}]}
    sync = {item['segment']: item for item in drive_logs.sync(merged, index)}
    assert sync[0]['precision'] == 'frame'
    assert sync[1]['precision'] == 'degraded' and sync[1]['frames_log'] == 1190 and sync[1]['start'] == 60.0
    # The civil time is the satellites' (the fix 0.007 s after the drive's start carries its own time).
    assert abs(drive_logs.civil_start(merged) - 1_790_000_000_000) <= 10
    assert drive_review._when(drive_logs.civil_start(merged), {})[1] is True
    assert drive_logs.sync(merged, None)[0]['precision'] == 'no_video'


# ---------- the scan ----------

def test_the_scan_runs_on_the_cpu_without_any_neural_network(tmp_path, monkeypatch):
    # The extractor runs in a child process: a neural network library imported there fails, as here.
    blocked = tmp_path / 'blocked'
    blocked.mkdir()
    for name in ('torch', 'onnxruntime', 'tensorflow', 'tinygrad'):
        (blocked / f'{name}.py').write_text("raise ImportError('the drive review scan must not use a neural network')\n")
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(blocked), os.environ.get('PYTHONPATH', '')]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    comma_drive(tmp_path)
    store = drive_review.store_for(tmp_path)
    state = store.scan(ROUTE)
    assert state['state'] == 'done' and state['unreadable'] == [] and len(active(store)) == 2


def test_clues_across_two_minutes_make_one_episode(tmp_path):
    comma_drive(tmp_path)
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    episode = next(item for item in active(store) if item['channel'] == 'intervention')
    assert episode['kinds'] == ['brake', 'gas'] and episode['segments'] == [0, 1]
    assert [round((clue['t_ns'] - BOOT_NS) / NS, 1) for clue in episode['clues']] == [58.5, 62.0]
    assert episode['data']['log'] == 'rlog' and episode['before']['vision_kph'] == 80


def test_a_disagreement_without_intervention_and_missing_data_that_is_none(tmp_path):
    comma_drive(tmp_path)
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    runs = [item for item in active(store) if item['channel'] == 'disagreement']
    assert len(runs) == 1
    run = runs[0]
    assert run['kinds'] == ['disagreement'] and run['clues'][0]['detail']['pairs'][0][:2] == [80, 120]
    assert 9 <= run['clues'][0]['detail']['seconds'] <= 11
    assert 19 <= (run['start_ns'] - BOOT_NS) / NS <= 21
    # The driver's speed is a hint of which limit was right, never a limit.
    assert run['hint'] == {'speed_kph': 85, 'source': "driver's speed during the disagreement", 'closer_to': 'vision'}
    # After 120 s the map says 120 while the vision is "not sure", then not logged: nothing is found there.
    assert all(item['start_ns'] < BOOT_NS + 120 * NS for item in active(store))
    coverage = store.scan_state(ROUTE)['coverage']
    assert coverage['vision_limit'] < coverage['vision'] < coverage['seconds'] and coverage['buttons'] is False


def test_the_vision_ahead_of_the_map_is_no_disagreement():
    """Merged drives made up in place (no log): the vision reads the 40 sign before the map's 40 begins."""
    def scan(map_at_s, vision_back=False):
        vision = [[0, 60.0], [30 * NS, 40.0]] + ([[map_at_s * NS, 60.0]] if vision_back else [])
        merged = {'series': {'vis_limit': vision, 'map_valid': [[0, True]],
                             'map_limit': [[0, 60.0], [map_at_s * NS, 40.0], [100 * NS, 40.0]]},
                  'events': [], 'spans': {0: [0, 120 * NS]}, 'meta': {}}
        return drive_episodes.scan(ROUTE, merged, {0: 'rlog'})

    found = scan(35)
    assert not [item for item in found['episodes'] if item['channel'] != 'spot_check']
    assert found['counts'] == {'vision_ahead': 1} and found['coverage']['disagreement'] == 0
    # The map 15 s later is a disagreement; so is a vision that went back to 60 as the map came to 40.
    assert [item['channel'] for item in scan(45)['episodes'] if item['channel'] != 'spot_check'] == ['disagreement']
    assert scan(35, vision_back=True)['counts'].get('disagreement') == 1


def test_spot_checks_stay_away_from_episodes_and_are_the_same_at_every_scan():
    """A merged drive of 40 minutes made up in place (no log): vision and map agree throughout."""
    minutes = 40
    merged = {'series': {'vis_limit': [[0, 80.0]], 'map_valid': [[0, True]], 'map_limit': [[0, 80.0]],
                         'sd_enabled': [[0, True]], 'brake': [[0, False], [600 * NS, True], [601 * NS, False]]},
              'events': [], 'spans': {0: [0, minutes * 60 * NS]}, 'meta': {}}
    first = drive_episodes.scan(ROUTE, merged, {0: 'rlog'})
    again = drive_episodes.scan(ROUTE, merged, {0: 'rlog'})
    checks = [item for item in first['episodes'] if item['channel'] == 'spot_check']
    assert len(checks) == 2 and checks == [item for item in again['episodes'] if item['channel'] == 'spot_check']
    for item in checks:
        assert item['end_ns'] - item['start_ns'] == 30 * NS
        assert item['end_ns'] <= 540 * NS or item['start_ns'] >= 661 * NS


def test_a_set_speed_changed_with_icbm_idle_is_the_drivers_even_without_buttons():
    """Merged drives made up in place (no log), as the PSA logs them: no buttonEvents, the set speed from the car."""
    base = {'sd_enabled': [[0, True]], 'v_cruise': [[0, 50.0], [20 * NS, 51.0], [40 * NS, 52.0]]}

    def kinds(**series):
        merged = {'series': {**base, **series}, 'events': [], 'spans': {0: [0, 60 * NS]}, 'meta': {}}
        clues = drive_episodes.interventions(drive_episodes.Drive(merged))
        return [item['kind'] for item in clues if item['kind'].startswith('set_speed')]
    # ICBM idle throughout: the driver's stalk.
    assert kinds(icbm_state=[[0, 'inactive']], icbm_send=[[0, 'none']]) == ['set_speed_manual'] * 2
    # ICBM pressing a button at the first change: the car's; the second, with ICBM idle again, the driver's.
    assert kinds(icbm_state=[[0, 'inactive'], [19 * NS, 'increasing'], [22 * NS, 'inactive']],
                 icbm_send=[[0, 'none'], [19_500_000_000, 'increase'], [20_500_000_000, 'none']]) == [
        'set_speed_auto', 'set_speed_manual']
    # ICBM active without a button in the log, or no ICBM state at all (an older log): the log does not say.
    assert kinds(icbm_state=[[0, 'holding']]) == ['set_speed_unknown'] * 2
    assert kinds() == ['set_speed_unknown'] * 2


# ---------- only a view ----------

def test_the_drive_review_only_shows_and_sends_to_sign_review(tmp_path, monkeypatch):
    make_report(tmp_path)
    comma_drive(tmp_path, segments=(0, 1))
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    run = next(item for item in active(store) if item['channel'] == 'disagreement')
    server = RunningServer(tmp_path)
    try:
        view = json_of(server.request('GET', f'/api/drive-review/drives/{ROUTE}/timeline')[2])
        # Nothing is judged here: no verdicts, no spot checks; the signs are reviewed in Sign review.
        assert 'annotations' not in view and view['sign_review'] is True
        assert all(item['channel'] != 'spot_check' for item in view['episodes'])
        for path in ('/api/drive-review/annotations', f'/api/drive-review/annotations/{run["id"]}/delete'):
            assert server.request('POST', path, body={'route': ROUTE, 'verdict': 'correct_limit'})[0] == 404
        queue = json_of(server.request('GET', '/api/drive-review/episodes')[2])['episodes']
        assert queue and all('status' not in item and item['channel'] != 'spot_check' for item in queue)
    finally:
        server.close()
    # A new scanner keeps the episode's id: a link to it still opens it.
    monkeypatch.setattr(drive_episodes, 'SCANNER_VERSION', drive_episodes.SCANNER_VERSION + 1)
    assert store.scan(ROUTE)['state'] == 'done'
    assert next(item for item in active(store) if item['channel'] == 'disagreement')['id'] == run['id']
    # Both ways between the two pages: Drive review to the frame of a moment, Sign review back to the drive.
    script = CONTROL_JS.read_text(encoding='utf-8')
    assert '?segment=' in script and '&second=' in script
    assert '/drive-review?drive=' in (CONTROL_JS.parent / 'review.js').read_text(encoding='utf-8')


def test_drives_before_the_first_day_reviewed_stay_out_of_the_queue(tmp_path):
    make_report(tmp_path)
    # ROUTE's GPS says 21 Sep 2026, before the default first day; OTHER_ROUTE has no fix: its videos' day (today).
    comma_drive(tmp_path, segments=(0,), gps_unix_ms=1_790_000_000_000)
    comma_drive(tmp_path, OTHER_ROUTE, segments=(0,))
    store = drive_review.store_for(tmp_path)
    store.scan(ROUTE)
    store.scan(OTHER_ROUTE)
    assert store.review_from() == drive_review.DEFAULT_REVIEW_FROM == '2026-10-30'
    assert drive_review.early_routes(store) == {ROUTE}
    queue = drive_review.queue_view(store)
    assert {item['route'] for item in queue} == {OTHER_ROUTE}
    # Each episode says when it was: here from the videos' times only (no fix), so not certain.
    assert all(item['when'] and item['when_certain'] is False for item in queue)
    view = {item['route']: item for item in drive_review.drives_view(store)}
    assert view[ROUTE]['day'] == '2026-09-21' and view[ROUTE]['before_review'] is True
    assert view[OTHER_ROUTE]['before_review'] is False
    server = RunningServer(tmp_path)
    try:
        # Out of date with a new scanner: Find episodes searches only the drives reviewed.
        store._run([("UPDATE scans SET scanner_version = 0", ())])
        status, _, data = server.request('POST', '/api/drive-review/scan-all', body={})
        assert status == 200 and json_of(data)['queued'] == [OTHER_ROUTE]
        settled(store, OTHER_ROUTE)
        status, _, data = server.request('POST', '/api/drive-review/settings', body={'review_from': '30 Oct'})
        assert status == 400 and json_of(data)['error'] == 'bad_date'
        status, _, data = server.request('POST', '/api/drive-review/settings', body={'review_from': ''})
        assert status == 200 and json_of(data)['review_from'] == ''
        listing = json_of(server.request('GET', '/api/drive-review/drives')[2])
        assert listing['review_from'] == '' and not any(item['before_review'] for item in listing['drives'])
    finally:
        server.close()
    # Every drive again: both drives' episodes are in the queue; ROUTE's say their time from the GPS.
    queue = drive_review.queue_view(store)
    assert {item['route'] for item in queue} == {ROUTE, OTHER_ROUTE}
    assert all(item['when'].endswith('2026 · ' + item['when'][-5:]) and item['when_certain']
               for item in queue if item['route'] == ROUTE)
    assert any('21 Sep 2026' in item['when'] for item in queue if item['route'] == ROUTE)


def test_the_command_line_scans_every_drive_with_logs(tmp_path, capsys):
    from nnslr_tools.cli import main
    comma_drive(tmp_path, segments=(0, 1))
    comma_drive(tmp_path, OTHER_ROUTE, segments=(0,), logs=False)
    assert main(['drive-review', 'scan', '--all', '--data-root', str(tmp_path)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(line['route'], line['state'], line['episodes']) for line in lines] == [(ROUTE, 'done', 2)]
    # Nothing new: nothing to scan again.
    assert main(['drive-review', 'scan', '--all', '--data-root', str(tmp_path)]) == 0
    assert capsys.readouterr().out == ''
    # Nothing to export: the drive review is only a view.
    with pytest.raises(SystemExit):
        main(['drive-review', 'export', '--data-root', str(tmp_path)])
