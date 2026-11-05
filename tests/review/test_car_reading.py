"""What the car's networks read on the frame on screen in Sign review (nnslr_tools.car_reading): the car's own
read_signs on the frame of the road video as NV12, each box with what the car does with it and as a box of the
review, on a picture of the report and on a frame between the seconds."""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from nnslr_tools import car_reading, road_type_signs, video_frames
from nnslr_tools.review_db import ReviewRejected
from review_helpers import ROUTE, RUN_ID, json_of
from test_video_frames import drive  # noqa: F401 - the fixture: a 3-second road video and its report

CLASSES = ['cancellation', 'maximum_speed:80', 'not_a_sign', 'other_sign:give_way', 'other_sign:motorway_start',
           'other_sign:motorway_direction']


def car_vision():
    package = road_type_signs.package_path()
    if not (package / 'vision.py').is_file():
        pytest.skip('no sunnypilot clone with the car code')
    return road_type_signs.load_vision(package)


def finder(x):
    """SignNet's output with three peaks: 0.9, 0.5 and 0.2 (below MIN_BOX_SCORE), each 80 band pixels wide."""
    out = np.zeros((1, 5, x.shape[2] // 8, x.shape[3] // 8), dtype=np.float32)
    for (iy, ix), heat in (((5, 10), 0.9), ((5, 40), 0.5), ((20, 80), 0.2)):
        out[0, 0, iy, ix] = heat
        out[0, 1:3, iy, ix] = np.log(80.0)
    return out


class Classifier:
    """The first box a sure 80, the second a give way not sure enough (about 0.35)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, _x):
        logits = np.zeros(len(CLASSES), dtype=np.float32)
        logits[1 if self.calls % 2 == 0 else 3] = 10.0 if self.calls % 2 == 0 else 1.0
        self.calls += 1
        return logits[None]


def stub_networks(vision):
    return {'vision': vision, 'finder': finder, 'classifier': Classifier(), 'classes': CLASSES, 'stamp': ('stub',),
            'release': {'name': 'speed-limits test'}}


def test_each_box_says_what_the_car_does_with_it():
    vision = car_vision()
    width, height = 1344, 760
    frame = np.full(width * height * 3 // 2, 128, dtype=np.uint8).tobytes()
    signs, surest = car_reading.read_nv12(stub_networks(vision), frame, width, height)
    assert surest == 0
    # The third peak is below MIN_BOX_SCORE: the car never sees it.
    assert [sign['score'] for sign in signs] == [0.9, 0.5]
    sure, unsure = signs
    # 80 band pixels at 1344 / 896 = 1.5 frame pixels each, centred on (10 * 8, 5 * 8) band pixels.
    assert sure['box'] == [60.0, 0.0, 180.0, 120.0]
    assert sure['class'] == 'maximum_speed:80' and sure['confidence'] > 0.99 and sure['speed'] == 80.0
    assert sure['use'] == 'speed' and sure['review'] == {'family': 'maximum_speed', 'value_kph': 80}
    assert unsure['class'] == 'other_sign:give_way' and unsure['confidence'] < vision.MIN_CONFIDENCE
    assert unsure['use'] == 'unsure' and unsure['speed'] is None
    assert unsure['review'] == {'family': 'other_sign', 'value_kph': None, 'sign_type': 'give_way'}

    def use(name, confidence=0.9):
        return car_reading.use_of(vision, {'class': name, 'confidence': confidence,
                                           'speed': vision.class_speed(name)})
    assert use('cancellation') == 'speed'  # the end of a limit: 0 for the memory
    assert use('other_sign:motorway_start') == 'road'
    assert use('other_sign:motorway_direction') == 'board'
    assert use('other_sign:give_way') == use('not_a_sign') == use('maximum_speed:unknown') == 'nothing'
    assert use('maximum_speed:80', vision.MIN_CONFIDENCE - 0.01) == 'unsure'


def test_the_frame_s_reading_is_the_car_s_on_two_signs_both_1_0_once_rounded():
    """Two sure signs on the two sides of the road: the car's logs give the one read surer, even when both show as
    1.0 here."""
    vision = car_vision()

    class Sure:
        def __init__(self):
            self.calls = 0

        def __call__(self, _x):
            logits = np.zeros(len(CLASSES), dtype=np.float32)
            logits[1 if self.calls == 0 else 0] = 13.5 if self.calls == 0 else 16.0
            self.calls += 1
            return logits[None]

    loaded = {**stub_networks(vision), 'classifier': Sure()}
    frame = np.full(1344 * 760 * 3 // 2, 128, dtype=np.uint8).tobytes()
    signs, surest = car_reading.read_nv12(loaded, frame, 1344, 760)
    assert [(sign['class'], sign['confidence']) for sign in signs] == [('maximum_speed:80', 1.0), ('cancellation', 1.0)]
    assert surest == 1


def car_package(tmp_path, *, networks=True):
    """A copy of the car's package of the sunnypilot clone (its networks linked, not copied), or one without them."""
    source = car_reading.road_type_signs.package_path()
    if not (source / 'models' / car_reading.MANIFEST).is_file():
        pytest.skip('no release of the car\'s networks in the sunnypilot clone')
    base = tmp_path / 'sunnypilot'
    package = base / car_reading.road_type_signs.PACKAGE_RELATIVE
    (package / 'models').mkdir(parents=True)
    shutil.copy(source / 'vision.py', package / 'vision.py')
    for path in (source / 'models').iterdir():
        if path.suffix == '.json':
            shutil.copy(path, package / 'models' / path.name)
        elif networks and path.suffix == '.onnx':
            (package / 'models' / path.name).symlink_to(path)
    return base, package


def test_a_release_without_its_networks_is_no_car_networks(drive, tmp_path, monkeypatch):
    _root, _video, _rows, server = drive
    base, _package = car_package(tmp_path, networks=False)
    monkeypatch.setenv('NNSLR_SUNNYPILOT_CLONE', str(base))
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/car/0/1')
    assert status == 503 and json_of(data)['error'] == 'no_car_networks'


def test_a_new_release_leaves_the_readings_under_way_their_networks(tmp_path, monkeypatch):
    _base, package = car_package(tmp_path)
    monkeypatch.setattr(car_reading, '_loaded', {})
    before = car_reading.networks(package)
    assert car_reading.networks(package) is before
    manifest = package / 'models' / car_reading.MANIFEST
    stat = manifest.stat()
    os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    after = car_reading.networks(package)
    assert after is not before and after['stamp'] != before['stamp']
    assert set(before) == set(after) == {'vision', 'finder', 'classifier', 'classes', 'stamp', 'release'}


def test_a_class_becomes_a_box_of_the_review():
    assert car_reading.review_box('road_marking:30') == {'family': 'road_marking_candidate', 'value_kph': 30}
    assert car_reading.review_box('maximum_speed:unknown') == {'family': 'maximum_speed', 'value_kph': None}
    assert car_reading.review_box('zone:30') == {'family': 'zone', 'value_kph': 30}
    assert car_reading.review_box('maximum_speed_hours:30:22-6') == {
        'family': 'maximum_speed', 'value_kph': 30, 'supplementary_panel': 'time', 'plate_hours': '22-6'}
    assert car_reading.review_box('variable_display:unknown') == {'family': 'variable_display', 'value_kph': None}
    assert car_reading.review_box('other_sign') == {'family': 'other_sign', 'value_kph': None}
    for name in ('cancellation', 'unreadable', 'not_a_sign'):
        assert car_reading.review_box(name) == {'family': name, 'value_kph': None}
    assert car_reading.review_box('maximum_speed_hours:30') == {'family': 'maximum_speed', 'value_kph': 30,
                                                                'supplementary_panel': 'time'}


def test_the_nv12_of_a_gop_is_the_frame_of_a_decode_of_the_whole_video(drive):
    _root, video, _rows, _server = drive
    whole = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vsync', '0', '-pix_fmt', 'nv12', '-f',
                            'rawvideo', '-'], capture_output=True, check=True).stdout
    size = 64 * 48 * 3 // 2
    for decoded in (0, 19, 27, 59):
        data, width, height = video_frames.nv12_frame(video, decoded)
        assert (width, height) == (64, 48)
        assert data == whole[decoded * size:(decoded + 1) * size]
    with pytest.raises(video_frames.VideoFrameError):
        video_frames.nv12_frame(video, 60)


def test_the_car_reads_a_picture_of_the_report_and_a_frame_between_the_seconds(drive, monkeypatch):
    root, _video, _rows, server = drive
    vision = car_vision()
    monkeypatch.setattr(car_reading, 'networks', lambda package=None: stub_networks(vision))
    for key in ('0/1', '0/1025'):
        found = server.app.car_reading(ROUTE, key)
        assert found['frame_key'] == key and found['run_id'] == RUN_ID and (found['width'], found['height']) == (64, 48)
        assert [sign['use'] for sign in found['signs']] == ['speed', 'unsure'] and found['surest'] == 0
    # The picture at 1 a second under another key, past the end of the video: not frames to read.
    for key in ('0/1020', '0/1060'):
        with pytest.raises(ReviewRejected) as refused:
            server.app.car_reading(ROUTE, key)
        assert refused.value.status == 404
    # A video of another size than the frame's picture: its boxes would not be on it.
    with pytest.raises(video_frames.VideoFrameError):
        car_reading.read_frame(root, ROUTE, 0, 5, width=128, height=96)


def test_the_page_asks_the_server_what_the_car_reads(drive, monkeypatch):
    _root, _video, _rows, server = drive
    vision = car_vision()
    monkeypatch.setattr(car_reading, 'networks', lambda package=None: stub_networks(vision))
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/car/0/1025')
    assert status == 200, data
    found = json_of(data)
    assert found['ok'] and found['frame_key'] == '0/1025' and found['release'] == {'name': 'speed-limits test'}
    assert [(sign['class'], sign['use']) for sign in found['signs']] == [('maximum_speed:80', 'speed'),
                                                                        ('other_sign:give_way', 'unsure')]
    assert server.request('GET', f'/api/route/{ROUTE}/car/0/1060')[0] == 404

    def missing(package=None):
        raise car_reading.CarReadingError('no release of the car\'s networks')
    monkeypatch.setattr(car_reading, 'networks', missing)
    status, _response, data = server.request('GET', f'/api/route/{ROUTE}/car/0/1026')
    assert status == 503 and json_of(data)['error'] == 'no_car_networks'
