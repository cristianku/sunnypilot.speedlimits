"""openpilot.distill's localizer: its tracks read with the standard library, the position of a frame from them,
nnslr gps taking it for the road camera's frames (or keeping the GPS), and nnslr localize running a segment.

Everything is synthetic: hand-written safetensors files, a fake log reader and a fake localizer child; no real drive
and no openpilot.distill."""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

from nnslr_tools import gps, localized
from nnslr_tools.cli import main
from review_helpers import OTHER_ROUTE, ROUTE

_A = 6378137.0
_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)
COLUMNS = 43
# Zürich, 400 m up.
LAT, LON, HEIGHT = 47.3769, 8.5417, 400.0


def ecef(lat: float, lon: float, height: float = HEIGHT) -> tuple[float, float, float]:
    la, lo = math.radians(lat), math.radians(lon)
    n = _A / math.sqrt(1 - _E2 * math.sin(la) ** 2)
    return ((n + height) * math.cos(la) * math.cos(lo), (n + height) * math.cos(la) * math.sin(lo),
            (n * (1 - _E2) + height) * math.sin(la))


def ecef_velocity(lat: float, lon: float, speed: float, bearing: float) -> tuple[float, float, float]:
    la, lo = math.radians(lat), math.radians(lon)
    east, north = speed * math.sin(math.radians(bearing)), speed * math.cos(math.radians(bearing))
    return (-math.sin(lo) * east - math.sin(la) * math.cos(lo) * north,
            math.cos(lo) * east - math.sin(la) * math.sin(lo) * north,
            math.cos(la) * north)


def state(lat: float, lon: float, speed: float, bearing: float) -> list[float]:
    """A row of frame_states: ECEF position, a quaternion, ECEF velocity, the rest zero."""
    row = [0.0] * COLUMNS
    row[0:3] = ecef(lat, lon)
    row[3:7] = [1.0, 0.0, 0.0, 0.0]
    row[7:10] = ecef_velocity(lat, lon, speed, bearing)
    return row


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> Path:
    header, blob = {'__metadata__': {'schema_version': '1'}}, b''
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(blob), len(blob) + len(data)]}
        blob += data
    head = json.dumps(header).encode()
    head += b' ' * (-len(head) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(head)) + head + blob)
    return path


def write_track(path: Path, rows: list[list[float]]) -> Path:
    data = struct.pack(f'<{len(rows) * COLUMNS}d', *[value for row in rows for value in row])
    return write_safetensors(path, {'frame_states': ('F64', [len(rows), COLUMNS], data),
                                    't': ('F64', [len(rows)], struct.pack(f'<{len(rows)}d', *range(len(rows))))})


def north_of(metres: float, lat: float = LAT) -> float:
    return lat + math.degrees(metres / 6371000.0)


def east_of(metres: float, lon: float = LON, lat: float = LAT) -> float:
    return lon + math.degrees(metres / (6371000.0 * math.cos(math.radians(lat))))


# -- reading a track ---------------------------------------------------------------------------------------------------

def test_a_frame_of_the_track_gives_its_place_speed_and_bearing(tmp_path):
    track = localized.load_track(write_track(tmp_path / 'localizer.safetensors', [
        state(LAT, LON, 10.0, 90.0), state(north_of(5), LON, 25.0, 312.5)]))
    assert track[0] == 2
    first = localized.frame_fix(track, 0)
    assert first['latitude'] == pytest.approx(LAT, abs=1e-7)
    assert first['longitude'] == pytest.approx(LON, abs=1e-7)
    assert first['speed_mps'] == 10.0 and first['bearing_deg'] == 90.0
    second = localized.frame_fix(track, 1)
    assert second['latitude'] == pytest.approx(north_of(5), abs=1e-7)
    assert (second['speed_mps'], second['bearing_deg']) == (25.0, 312.5)
    assert set(first) == {'latitude', 'longitude', 'speed_mps', 'bearing_deg'}


def test_a_standing_car_keeps_the_bearing_it_last_moved_with(tmp_path):
    track = localized.load_track(write_track(tmp_path / 'localizer.safetensors', [
        state(LAT, LON, 0.3, 200.0),      # standing before it ever moved: the bearing of the first move after
        state(LAT, LON, 8.0, 45.0),
        state(LAT, LON, 0.4, 270.0),      # stopped: the velocity's direction is noise
        state(LAT, LON, 0.0, 0.0),
        state(LAT, LON, 6.0, 120.0)]))
    assert [localized.frame_fix(track, index)['bearing_deg'] for index in range(5)] == [45.0, 45.0, 45.0, 45.0, 120.0]
    assert localized.frame_fix(track, 2)['speed_mps'] == 0.4
    still = localized.load_track(write_track(tmp_path / 'still.safetensors', [state(LAT, LON, 0.2, 33.0)] * 3))
    assert localized.frame_fix(still, 1)['bearing_deg'] == 33.0


def test_a_frame_outside_the_track_or_not_a_place_has_no_position(tmp_path):
    nowhere = [0.0] * COLUMNS
    broken = state(LAT, LON, 5.0, 0.0)
    broken[1] = float('nan')
    track = localized.load_track(write_track(tmp_path / 'localizer.safetensors', [
        state(LAT, LON, 5.0, 0.0), nowhere, broken]))
    assert localized.frame_fix(track, 0) is not None
    assert localized.frame_fix(track, 1) is None
    assert localized.frame_fix(track, 2) is None
    assert localized.frame_fix(track, -1) is None
    assert localized.frame_fix(track, 3) is None


def test_the_geodetic_conversion_comes_back_to_the_same_place():
    for lat, lon in ((LAT, LON), (-33.9, 151.2), (64.1, -21.9)):
        back = localized.geodetic(*ecef(lat, lon, 1234.0))
        assert back == (pytest.approx(lat, abs=1e-9), pytest.approx(lon, abs=1e-9))


def test_a_broken_track_file_is_an_error(tmp_path):
    rows = 2 * COLUMNS
    good = struct.pack(f'<{rows}d', *([1.0] * rows))
    cases = {
        'short': None,
        'f32': {'frame_states': ('F32', [2, COLUMNS], good[: rows * 4])},
        'shape': {'frame_states': ('F64', [3, COLUMNS], good)},
        'missing': {'t': ('F64', [1], struct.pack('<d', 0.0))},
        'narrow': {'frame_states': ('F64', [2, 5], good[:80])},
    }
    for name, tensors in cases.items():
        path = tmp_path / f'{name}.safetensors'
        if tensors is None:
            path.write_bytes(b'\x01\x02')
        else:
            write_safetensors(path, tensors)
        with pytest.raises(localized.LocalizerError):
            localized.load_track(path)
    truncated = write_safetensors(tmp_path / 'truncated.safetensors', {'frame_states': ('F64', [2, COLUMNS], good)})
    truncated.write_bytes(truncated.read_bytes()[:-8])
    with pytest.raises(localized.LocalizerError, match='truncated'):
        localized.load_track(truncated)
    not_json = tmp_path / 'header.safetensors'
    not_json.write_bytes(struct.pack('<Q', 4) + b'nope')
    with pytest.raises(localized.LocalizerError, match='header'):
        localized.load_track(not_json)
    # Not a safetensors file at all: its first 8 bytes are no header length.
    text = tmp_path / 'text.safetensors'
    text.write_text('frame,latitude,longitude\n0,47.3,8.5\n')
    with pytest.raises(localized.LocalizerError, match='not a safetensors'):
        localized.load_track(text)
    shape = [2, COLUMNS]
    for name, tensor in {'no offsets': {'dtype': 'F64', 'shape': shape},
                         'null offsets': {'dtype': 'F64', 'shape': shape, 'data_offsets': None},
                         'negative offset': {'dtype': 'F64', 'shape': shape, 'data_offsets': [-8, len(good) - 8]},
                         'shape a number': {'dtype': 'F64', 'shape': 2, 'data_offsets': [0, len(good)]},
                         'shape of booleans': {'dtype': 'F64', 'shape': [True, COLUMNS],
                                               'data_offsets': [0, 8 * COLUMNS]},
                         'past the end': {'dtype': 'F64', 'shape': [10**12, COLUMNS],
                                          'data_offsets': [0, 8 * COLUMNS * 10**12]}}.items():
        head = json.dumps({'frame_states': tensor}).encode()
        path = tmp_path / f'{name}.safetensors'
        path.write_bytes(struct.pack('<Q', len(head)) + head + good)
        with pytest.raises(localized.LocalizerError):
            localized.load_track(path)


def test_a_track_is_ours_else_distill_s_own(tmp_path):
    root, distill = tmp_path / 'data', tmp_path / 'distill'
    assert localized.track_path(root, ROUTE, 4, distill) is None
    theirs = write_track(distill / 'data' / f'{ROUTE.replace("--", "")}04' / localized.TRACK, [state(LAT, LON, 1, 0)])
    assert localized.track_path(root, ROUTE, 4, distill) == theirs
    ours = write_track(root / 'derived/localizer' / ROUTE / '4' / localized.TRACK, [state(LAT, LON, 1, 0)])
    assert localized.track_path(root, ROUTE, 4, distill) == ours
    with pytest.raises(ValueError):
        localized.track_path(root, '../escape', 4, distill)


# -- nnslr gps with a track --------------------------------------------------------------------------------------------

def _rows(count: int, segment: int = 0, stream: str = 'narrow_road', start: int = 0) -> list[dict]:
    return [{'frame_key': f'{segment}/{index}', 'segment_index': segment, 'output_index': index,
             'camera_stream': stream, 'decoded_frame_index': 20 * index if stream == 'narrow_road' else None}
            for index in range(start, start + count)]


def _fake_logs(monkeypatch, root: Path, fixes_by_segment: dict[int, list]) -> None:
    """Every segment gets an rlog; read_logs returns ``fixes`` [(seconds, lat, lon, speed, bearing), ...]."""
    paths = {}
    for segment, fixes in fixes_by_segment.items():
        path = root / 'raw/routes' / ROUTE / str(segment) / 'rlog.zst'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'')
        paths[str(path)] = {'first_frame_mono': 0, 'error': None,
                            'fixes': [[int(t * 1e9), *fix] for t, *fix in fixes]}
    monkeypatch.setattr(gps, 'read_logs', lambda group, python=None: {str(p): paths[str(p)] for p in group})


def _lines(root: Path) -> dict[str, dict]:
    lines = gps.positions_path(root, ROUTE).read_text().splitlines()
    return {line['frame_key']: line for line in map(json.loads, lines)}


def test_the_road_camera_s_frames_take_the_localizer_s_position(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    # The GPS: 4 m east of the track and late; the track moves 10 m north a second.
    _fake_logs(monkeypatch, root, {0: [(k, north_of(10 * k - 8), LON + 5e-5, 10.0, 0.0) for k in range(5)],
                                   1: [(k, north_of(10 * k), LON, 10.0, 0.0) for k in range(5)]})
    write_track(root / 'derived/localizer' / ROUTE / '0' / localized.TRACK,
                [state(north_of(0.5 * frame), LON, 10.0, 0.0) for frame in range(81)])
    rows = _rows(4) + [{**_rows(1, start=4)[0], 'decoded_frame_index': None}] + _rows(2, segment=1)
    rows.append({'frame_key': 'wide/0/3', 'segment_index': 0, 'output_index': 3, 'camera_stream': 'wide_road',
                 'decoded_frame_index': 60})
    rows.append({'frame_key': '0/9', 'segment_index': 0, 'output_index': 9, 'camera_stream': 'narrow_road',
                 'decoded_frame_index': 180})
    result = gps.build_positions(root, ROUTE, rows, workers=1)
    assert result['from_localizer'] == 4 and result['located'] == 8
    assert result['localizer_refused_segments'] == [] and result['errors'] == []
    lines = _lines(root)
    for index in range(4):
        line = lines[f'0/{index}']
        assert line['position_from'] == 'localizer'
        assert line['latitude'] == pytest.approx(north_of(10 * index), abs=1e-7)
        assert line['longitude'] == pytest.approx(LON, abs=1e-7)
        assert (line['speed_mps'], line['bearing_deg']) == (10.0, 0.0)
        assert line['country'] == 'CH'
    # No frame number, another segment without a track, another camera: the GPS fix as before.
    assert lines['0/4']['position_from'] == 'gps' and lines['0/4']['longitude'] == round(LON + 5e-5, 6)
    assert lines['1/0']['position_from'] == 'gps' and lines['1/1']['position_from'] == 'gps'
    assert lines['wide/0/3']['position_from'] == 'gps' and lines['wide/0/3']['longitude'] == round(LON + 5e-5, 6)
    # Past the end of the track and of the GPS: no position, the country of the nearest frame with one.
    assert lines['0/9']['position_from'] is None and lines['0/9']['latitude'] is None
    assert lines['0/9']['country'] == 'CH' and lines['0/9']['country_from'] == 'nearest_fix'
    assert set(lines['0/0']) == {'frame_key', 'segment_index', 'output_index', 'latitude', 'longitude', 'speed_mps',
                                 'bearing_deg', 'position_from', 'country', 'country_from'}


def test_the_localizer_is_kept_out_where_it_disagrees_with_the_gps(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    fixes = [(k, north_of(10 * k), LON, 10.0, 0.0) for k in range(6)]
    _fake_logs(monkeypatch, root, {0: fixes})
    track = [state(north_of(0.5 * frame), LON, 10.0, 0.0) for frame in range(101)]
    track[40] = state(north_of(20 + 60), LON, 10.0, 0.0)        # frame 0/2: 60 m off
    track[80] = state(north_of(40), LON, 13.0, 0.0)             # frame 0/4: 3 m/s too fast
    path = write_track(root / 'derived/localizer' / ROUTE / '0' / localized.TRACK, track)
    result = gps.build_positions(root, ROUTE, _rows(6), workers=1)
    lines = _lines(root)
    # 2 of 6 frames refused: more than LOCALIZER_MAX_REFUSED, the segment keeps the GPS.
    assert result['localizer_refused_segments'] == [0] and result['from_localizer'] == 0
    assert {line['position_from'] for line in lines.values()} == {'gps'}
    # 1 of 6 refused: that frame keeps the GPS, the others take the localizer.
    track[80] = state(north_of(40), LON, 10.0, 0.0)
    write_track(path, track)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    result = gps.build_positions(root, ROUTE, _rows(6), workers=1)
    lines = _lines(root)
    assert result['localizer_refused_segments'] == [] and result['from_localizer'] == 5
    assert lines['0/2']['position_from'] == 'gps' and lines['0/2']['latitude'] == round(north_of(20), 6)
    assert [lines[f'0/{k}']['position_from'] for k in (0, 1, 3, 4, 5)] == ['localizer'] * 5


def test_the_track_is_checked_against_the_east_too_and_a_fifth_of_frames_may_disagree(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    fixes = [(k, north_of(10 * k), LON, 10.0, 0.0) for k in range(5)]
    fixes[1] = (1, north_of(10), east_of(20), 10.0, 0.0)       # 20 m east: within LOCALIZER_MAX_GAP_M
    fixes[3] = (3, north_of(30), east_of(30), 10.0, 0.0)       # 30 m east: too far
    _fake_logs(monkeypatch, root, {0: fixes})
    write_track(root / 'derived/localizer' / ROUTE / '0' / localized.TRACK,
                [state(north_of(0.5 * frame), LON, 10.0, 0.0) for frame in range(81)])
    result = gps.build_positions(root, ROUTE, _rows(5), workers=1)
    lines = _lines(root)
    # 1 of 5 frames refused is LOCALIZER_MAX_REFUSED, not more: the segment is used.
    assert result['localizer_refused_segments'] == [] and result['from_localizer'] == 4
    assert lines['0/1']['position_from'] == 'localizer' and lines['0/1']['longitude'] == pytest.approx(LON, abs=1e-7)
    assert lines['0/3']['position_from'] == 'gps' and lines['0/3']['longitude'] == round(east_of(30), 6)


def test_a_fix_seconds_away_where_the_gps_dropped_out_does_not_check_the_track(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    # 30 m/s north, the GPS exact 5 times a second but silent from 1.2 s to 4 s (its last fix 5 m/s slow); the track
    # exact at every frame.
    fixes = [(t / 5, north_of(6 * t), LON, 30.0, 0.0) for t in (*range(7), *range(20, 26))]
    fixes[6] = (1.2, north_of(36), LON, 25.0, 0.0)
    _fake_logs(monkeypatch, root, {0: fixes})
    track = [state(north_of(1.5 * frame), LON, 30.0, 0.0) for frame in range(101)]
    path = write_track(root / 'derived/localizer' / ROUTE / '0' / localized.TRACK, track)
    result = gps.build_positions(root, ROUTE, _rows(6), workers=1)
    lines = _lines(root)
    # Frames 0/2 and 0/3 have a fix 0.8 s and 1 s from them, 24 and 30 m behind or ahead, 0/2's slower: the track is
    # not refused for it.
    assert result['localizer_refused_segments'] == [] and result['from_localizer'] == 6
    assert lines['0/2']['position_from'] == 'localizer'
    assert lines['0/2']['latitude'] == pytest.approx(north_of(60), abs=1e-7)
    assert lines['0/3']['latitude'] == pytest.approx(north_of(90), abs=1e-7)
    # The track 100 m off at both: farther than the car drove since the fix. Those frames keep the GPS, but a fix that
    # far in time does not count against the segment, whose other frames keep the localizer.
    track[40] = state(north_of(60 + 100), LON, 30.0, 0.0)
    track[60] = state(north_of(90 - 100), LON, 30.0, 0.0)
    write_track(path, track)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    result = gps.build_positions(root, ROUTE, _rows(6), workers=1)
    lines = _lines(root)
    assert result['localizer_refused_segments'] == [] and result['from_localizer'] == 4
    assert [lines[f'0/{k}']['position_from'] for k in range(6)] == ['localizer'] * 2 + ['gps'] * 2 + ['localizer'] * 2
    assert lines['0/2']['latitude'] == round(north_of(36), 6)


def test_a_frame_without_a_gps_fix_keeps_no_position(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    # A fix at the first second only (an underground garage after it): the track has every frame, but road_types takes
    # a slow stretch without GPS for a garage, so those frames stay without a position.
    _fake_logs(monkeypatch, root, {0: [(0, LAT, LON, 10.0, 0.0)]})
    write_track(root / 'derived/localizer' / ROUTE / '0' / localized.TRACK,
                [state(north_of(0.5 * frame), LON, 10.0, 0.0) for frame in range(101)])
    result = gps.build_positions(root, ROUTE, _rows(6), workers=1)
    lines = _lines(root)
    # Frames 0/1 and 0/2 have the fix within GPS_MAX_GAP_S: theirs is the localizer's position, unchecked.
    assert result['from_localizer'] == 3 and result['located'] == 3
    assert lines['0/2']['position_from'] == 'localizer'
    assert lines['0/2']['latitude'] == pytest.approx(north_of(20), abs=1e-7)
    for index in (3, 4, 5):
        line = lines[f'0/{index}']
        assert line['position_from'] is None and line['latitude'] is None and line['bearing_deg'] is None
        assert line['country_from'] == 'nearest_fix'


@pytest.mark.parametrize('content', [b'\x00' * 4, b'not a track: some text that is no safetensors file\n'])
def test_a_broken_track_is_reported_and_the_gps_kept(tmp_path, monkeypatch, content):
    root = tmp_path / 'data'
    _fake_logs(monkeypatch, root, {0: [(k, LAT, LON, 0.0, 0.0) for k in range(3)]})
    broken = root / 'derived/localizer' / ROUTE / '0' / localized.TRACK
    broken.parent.mkdir(parents=True)
    broken.write_bytes(content)
    result = gps.build_positions(root, ROUTE, _rows(3), workers=1)
    assert result['from_localizer'] == 0 and result['located'] == 3
    assert result['errors'] and 'segment 0' in result['errors'][0]


def test_positions_are_read_again_only_when_a_track_is_newer(tmp_path):
    root = tmp_path / 'data'
    (root / 'raw/routes' / ROUTE / '0').mkdir(parents=True)
    target = gps.positions_path(root, ROUTE)
    assert not gps.positions_current(root, ROUTE)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'frame_key': '0/0', 'latitude': LAT}) + '\n')
    # No track: the GPS would give the same positions.
    assert gps.positions_current(root, ROUTE)
    track = write_track(root / 'derived/localizer' / ROUTE / '0' / localized.TRACK, [state(LAT, LON, 1, 0)])
    stamp = track.stat().st_mtime_ns
    os.utime(target, ns=(stamp + 10**9, stamp + 10**9))
    # Written before the localizer was known: no position_from.
    assert not gps.positions_current(root, ROUTE)
    target.write_text(json.dumps({'frame_key': '0/0', 'position_from': 'gps'}) + '\n')
    os.utime(target, ns=(stamp + 10**9, stamp + 10**9))
    assert gps.positions_current(root, ROUTE)
    os.utime(track, ns=(stamp + 2 * 10**9, stamp + 2 * 10**9))
    assert not gps.positions_current(root, ROUTE)
    # The same time is current; a later segment localized after the positions were read is not.
    os.utime(target, ns=(stamp + 2 * 10**9, stamp + 2 * 10**9))
    assert gps.positions_current(root, ROUTE)
    (root / 'raw/routes' / ROUTE / '1').mkdir()
    later = write_track(root / 'derived/localizer' / ROUTE / '1' / localized.TRACK, [state(LAT, LON, 1, 0)])
    os.utime(later, ns=(stamp + 3 * 10**9, stamp + 3 * 10**9))
    assert localized.newest_track_ns(root, ROUTE) == stamp + 3 * 10**9
    assert not gps.positions_current(root, ROUTE)


def test_gps_missing_takes_the_routes_with_a_newer_track(tmp_path, monkeypatch, capsys):
    from nnslr_tools import review
    root = tmp_path / 'data'
    for route in (ROUTE, OTHER_ROUTE):
        (root / 'raw/routes' / route / '0').mkdir(parents=True)
        target = gps.positions_path(root, route)
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({'frame_key': '0/0', 'position_from': 'gps'}) + '\n')
    track = write_track(root / 'derived/localizer' / OTHER_ROUTE / '0' / localized.TRACK, [state(LAT, LON, 1, 0)])
    later = gps.positions_path(root, OTHER_ROUTE).stat().st_mtime_ns + 10**9
    os.utime(track, ns=(later, later))
    done = []
    monkeypatch.setattr(review, 'list_reports', lambda data: [(ROUTE, None), (OTHER_ROUTE, None)])
    monkeypatch.setattr(review, 'latest_report', lambda data, route: route)
    monkeypatch.setattr(review, 'report_rows', lambda report: [])
    monkeypatch.setattr(gps, 'build_positions', lambda data, route, rows: done.append(route) or {'route_id': route})
    assert main(['gps', '--all', '--missing', '--data-root', str(root)]) == 0
    assert done == [OTHER_ROUTE]
    assert json.loads(capsys.readouterr().out) == {'route_id': OTHER_ROUTE}


# -- nnslr localize ----------------------------------------------------------------------------------------------------

FAKE_CHILD = '''
import json, os, sys, time
from pathlib import Path
distill, source, out = (Path(arg) for arg in sys.argv[1:4])
mode = (source / 'mode').read_text().strip()
if mode == 'sleep':
    time.sleep(60)
if mode == 'unusable':
    print('UnusableSegment: the car never went above 18 km/h', file=sys.stderr)
    sys.exit(3)
if mode == 'fail':
    print('Traceback ...', file=sys.stderr)
    print('RuntimeError: no ionosphere map', file=sys.stderr)
    sys.exit(1)
out.mkdir(parents=True, exist_ok=True)
(out / 'localizer.safetensors').write_bytes((source / 'track').read_bytes())
threads = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
(out / 'call.json').write_text(json.dumps({'argv': sys.argv[1:], 'isolated': sys.flags.isolated, 'nice': os.nice(0),
                                           'threads': threads}))
print('frames=1 ionex=test')
'''


@pytest.fixture
def fake_distill(tmp_path, monkeypatch):
    """openpilot.distill with a .venv whose Python is this one, and a localizer child that does what a segment's
    ``mode`` file says."""
    distill = Path(os.environ['NNSLR_DISTILL_ROOT'])
    (distill / '.venv/bin').mkdir(parents=True)
    (distill / '.venv/bin/python').symlink_to(sys.executable)
    child = tmp_path / 'child' / 'fake_child.py'
    child.parent.mkdir()
    child.write_text(FAKE_CHILD)
    monkeypatch.setattr(localized, 'CHILD', child)
    return distill


def _segment(root: Path, route: str, segment: int, mode: str | None, *, inputs: bool = True) -> Path:
    directory = root / 'raw/routes' / route / str(segment)
    directory.mkdir(parents=True, exist_ok=True)
    if inputs:
        for name in localized.INPUTS:
            (directory / name).write_bytes(b'')
    if mode:
        (directory / 'mode').write_text(mode)
        write_track(directory / 'track', [state(LAT, LON, 3.0, 10.0)])
    return directory


def test_a_segment_goes_through_the_localizer_once(tmp_path, fake_distill):
    root = tmp_path / 'data'
    _segment(root, ROUTE, 0, 'ok')
    assert localized.localize_segment(root, ROUTE, 0) == {'segment': 0, 'state': 'new'}
    out = localized.output_dir(root, ROUTE, 0)
    assert localized.frame_fix(localized.load_track(out / localized.TRACK), 0)['bearing_deg'] == 10.0
    assert localized.localize_segment(root, ROUTE, 0) == {'segment': 0, 'state': 'done'}


def test_the_child_runs_isolated_niced_on_one_thread_and_is_stopped_when_too_long(tmp_path, fake_distill):
    root = tmp_path / 'data'
    source = _segment(root, ROUTE, 0, 'ok')
    assert localized.localize_segment(root, ROUTE, 0)['state'] == 'new'
    out = localized.output_dir(root, ROUTE, 0)
    call = json.loads((out / 'call.json').read_text())
    assert call['argv'] == [str(fake_distill), str(source), str(out)]
    assert call['isolated'] == 1 and call['nice'] == min(19, os.nice(0) + 10)
    assert call['threads'] == {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    _segment(root, ROUTE, 1, 'sleep')
    started = time.monotonic()
    assert localized.localize_segment(root, ROUTE, 1, timeout=1) == {'segment': 1, 'state': 'failed',
                                                                    'message': 'more than 1 s'}
    assert time.monotonic() - started < 30


def test_a_segment_the_localizer_cannot_use_is_not_tried_again(tmp_path, fake_distill):
    root = tmp_path / 'data'
    source = _segment(root, ROUTE, 1, 'unusable')
    first = localized.localize_segment(root, ROUTE, 1)
    assert first['state'] == 'unusable' and '18 km/h' in first['message']
    note = json.loads((localized.output_dir(root, ROUTE, 1) / localized.FAILED).read_text())
    assert note['reason'] == 'unusable'
    (source / 'mode').write_text('ok')
    assert localized.localize_segment(root, ROUTE, 1) == {'segment': 1, 'state': 'unusable'}
    assert localized.localize_segment(root, ROUTE, 1, again=True) == {'segment': 1, 'state': 'new'}
    assert not (localized.output_dir(root, ROUTE, 1) / localized.FAILED).exists()


def test_a_failure_is_reported_and_tried_again_next_time(tmp_path, fake_distill):
    root = tmp_path / 'data'
    source = _segment(root, ROUTE, 2, 'fail')
    failed = localized.localize_segment(root, ROUTE, 2)
    assert failed == {'segment': 2, 'state': 'failed', 'message': 'RuntimeError: no ionosphere map'}
    (source / 'mode').write_text('ok')
    assert localized.localize_segment(root, ROUTE, 2)['state'] == 'new'


def test_a_segment_without_its_files_or_already_localized_by_distill(tmp_path, fake_distill):
    root = tmp_path / 'data'
    _segment(root, ROUTE, 3, 'ok', inputs=False)
    assert localized.localize_segment(root, ROUTE, 3) == {'segment': 3, 'state': 'no_input'}
    theirs = localized.distill_dir(ROUTE, 3) / localized.TRACK
    write_track(theirs, [state(LAT, LON, 1, 0)])
    assert localized.localize_segment(root, ROUTE, 3) == {'segment': 3, 'state': 'done'}
    assert not localized.output_dir(root, ROUTE, 3).exists()


def test_without_distill_s_python_nothing_runs(tmp_path):
    root = tmp_path / 'data'
    _segment(root, ROUTE, 0, 'ok')
    with pytest.raises(localized.LocalizerError):
        localized.distill_python()
    result = localized.localize_segment(root, ROUTE, 0)
    assert result['state'] == 'failed' and 'openpilot.distill' in result['message']
    assert main(['localize', ROUTE, '--data-root', str(root)]) == 2
    assert not localized.output_dir(root, ROUTE, 0).exists()


def test_localize_prints_a_line_per_route_and_fails_when_a_segment_failed(tmp_path, fake_distill, capsys):
    root = tmp_path / 'data'
    for segment, mode in enumerate(('ok', 'unusable', 'ok')):
        _segment(root, ROUTE, segment, mode)
    _segment(root, OTHER_ROUTE, 0, 'ok')
    (root / 'raw/routes/not-a-route/0').mkdir(parents=True)
    assert main(['localize', '--all', '--workers', '2', '--data-root', str(root)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    by_route = {line['route_id']: line for line in lines}
    assert set(by_route) == {ROUTE, OTHER_ROUTE} and len(lines) == 2
    assert by_route[ROUTE] == {'route_id': ROUTE, 'segments': 3, 'new': 2, 'unusable': 1, 'failed_segments': {}}
    assert by_route[OTHER_ROUTE] == {'route_id': OTHER_ROUTE, 'segments': 1, 'new': 1, 'failed_segments': {}}
    _segment(root, ROUTE, 3, 'fail')
    assert main(['localize', ROUTE, '--data-root', str(root)]) == 1
    line = json.loads(capsys.readouterr().out)
    assert line['done'] == 2 and line['unusable'] == 1 and line['failed'] == 1
    assert line['failed_segments'] == {'3': 'RuntimeError: no ionosphere map'}


def test_localize_without_drives_or_with_a_wrong_name(tmp_path, fake_distill, capsys):
    root = tmp_path / 'data'
    assert main(['localize', '--all', '--data-root', str(root)]) == 0
    assert main(['localize', '--data-root', str(root)]) == 2
    assert main(['localize', '../escape', '--data-root', str(root)]) == 2
    assert capsys.readouterr().out == ''
    # A drive named that is not in raw/routes: said, and exit 1.
    assert main(['localize', ROUTE, '--data-root', str(root)]) == 1
    line = json.loads(capsys.readouterr().out)
    assert line['segments'] == 0 and 'no segment folders' in line['error']


def _routes(root: Path, count: int, segments: int = 1) -> list[str]:
    routes = [f'{index + 10:08x}--abcdef1234' for index in range(count)]
    for route in routes:
        for segment in range(segments):
            (root / 'raw/routes' / route / str(segment)).mkdir(parents=True)
    return routes


def test_segments_run_workers_at_a_time(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    routes = _routes(root, 2, segments=4)
    running, most = [0], [0]
    lock = threading.Lock()

    def fake_segment(root, route, segment, **kwargs):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return {'segment': segment, 'state': 'new'}

    monkeypatch.setattr(localized, 'localize_segment', fake_segment)
    summaries = localized.localize_routes(root, routes, workers=2)
    assert [summary['new'] for summary in summaries] == [4, 4] and most[0] == 2


def test_ctrl_c_drops_the_segments_not_started(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    routes = _routes(root, 8)
    started = []

    def fake_segment(root, route, segment, **kwargs):
        started.append(route)
        time.sleep(0.05)
        return {'segment': segment, 'state': 'new'}

    def interrupt(summary):
        raise KeyboardInterrupt

    monkeypatch.setattr(localized, 'localize_segment', fake_segment)
    with pytest.raises(KeyboardInterrupt):
        localized.localize_routes(root, routes, workers=1, on_route=interrupt)
    assert len(started) <= 2


def test_positions_cover_every_frame_of_the_report(tmp_path):
    root = tmp_path / 'data'
    rows = [{'frame_key': '0/0'}, {'frame_key': '0/1'}]
    assert not gps.positions_cover(root, ROUTE, rows)
    target = gps.positions_path(root, ROUTE)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'frame_key': '0/0'}) + '\n')
    assert not gps.positions_cover(root, ROUTE, rows)
    target.write_text(json.dumps({'frame_key': '0/0'}) + '\n' + json.dumps({'frame_key': '0/1'}) + '\n')
    assert gps.positions_cover(root, ROUTE, rows)
