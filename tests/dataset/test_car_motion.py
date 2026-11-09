"""The car's own motion as an input of the road-type network (nnslr_tools.car_motion) and the car's turn as a plain
rule (nnslr_tools.road_type_metrics.turns, turn_events), on synthetic samples: no log is read. numpy and the modules
that need it are imported inside the tests: collecting the tests must leave numpy out
(tests/cli/test_import_isolation.py)."""
import json
import math

import pytest


def _pose(seconds: float, yaw_dps: float, *, hz: float = 20.0, valid: bool = True, start: float = 0.0) -> list:
    """deviceMotion samples turning left at yaw_dps: the device z points down, so z = -yaw."""
    count = int(round(seconds * hz))
    return [[start + (i + 1) / hz, -math.radians(yaw_dps), int(valid)] for i in range(count)]


def _car(seconds: float, *, steer: float = 0.0, left: bool = False, right: bool = False, v_ego: float = 10.0,
         yaw_rate: float = 0.0, hz: float = 20.0, start: float = 0.0) -> list:
    count = int(round(seconds * hz))
    return [[start + (i + 1) / hz, v_ego, yaw_rate, steer, int(left), int(right)] for i in range(count)]


def test_a_second_is_its_heading_change_its_steering_and_its_blinkers():
    import numpy as np

    from nnslr_tools import car_motion
    # 2 s straight, then 3 s turning left at 20 deg/s with the wheel at 90 deg and the left blinker, then straight.
    pose = _pose(2, 0.0) + _pose(3, 20.0, start=2) + _pose(3, 0.0, start=5)
    car = (_car(2, v_ego=8.0) + _car(3, steer=90.0, left=True, v_ego=6.0, start=2)
           + _car(3, v_ego=7.0, start=5))
    items = car_motion.summarise(np.arange(1.0, 9.0), np.array(pose), np.array(car))
    assert [round(item['yaw_dps'], 3) for item in items] == [0.0, 0.0, 20.0, 20.0, 20.0, 0.0, 0.0, 0.0]
    assert {item['yaw_source'] for item in items} == {'device'}
    assert [round(item['steer_deg']) for item in items] == [0, 0, 90, 90, 90, 0, 0, 0]
    # The blinker counts for the 2 s after its last sample (5.0 s: on at 6 s, off at 7 s).
    assert [item['left'] for item in items] == [False, False, True, True, True, True, False, False]
    assert not any(item['right'] for item in items)
    assert [item['v_ego'] for item in items[:3]] == [8.0, 8.0, 6.0]
    features = car_motion.motion_features(items[3])
    assert len(features) == car_motion.MOTION_INPUTS == len(car_motion.MOTION_FIELDS)
    assert features == pytest.approx((20 / 30, 20 / 30, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0))


def test_without_the_device_the_cars_yaw_rate_and_without_carstate_nothing_but_the_yaw():
    import numpy as np

    from nnslr_tools import car_motion
    # The pose is not valid: the car's own yawRate (rad/s, left positive) is taken.
    pose = _pose(3, 30.0, valid=False)
    car = _car(3, yaw_rate=math.radians(-10.0))
    items = car_motion.summarise(np.arange(1.0, 4.0), np.array(pose), np.array(car))
    assert [item['yaw_source'] for item in items] == ['car'] * 3
    assert round(items[1]['yaw_dps'], 3) == -10.0
    # A car whose yawRate is always 0 has no sensor: the yaw is unknown.
    items = car_motion.summarise(np.arange(1.0, 3.0), np.array(pose), np.array(_car(2)))
    assert items[0]['yaw_dps'] is None and items[0]['car_known'] is True
    assert car_motion.motion_features(items[0])[:3] == (0.0, 0.0, 0.0)
    # No carState: the yaw only; steering and blinkers unknown, their flags 0.
    items = car_motion.summarise(np.arange(1.0, 3.0), np.array(_pose(2, 5.0)), np.zeros((0, 6)))
    features = car_motion.motion_features(items[1])
    assert features[2] == 1.0 and features[3:] == (0.0,) * 5
    assert items[1]['v_ego'] is None
    # Too few samples in a step (a gap in the log): unknown.
    sparse = _pose(2, 5.0, hz=4.0)
    assert car_motion.summarise(np.array([2.0]), np.array(sparse), np.zeros((0, 6)))[0]['yaw_dps'] is None
    assert car_motion.motion_features(None) == car_motion.NO_MOTION


def test_a_drives_record_gives_the_motion_of_every_frame(tmp_path):
    from nnslr_tools import car_motion
    logs = {0: tmp_path / '0' / 'rlog.zst', 1: tmp_path / '1' / 'rlog.zst'}
    for path in logs.values():
        path.parent.mkdir(parents=True)
        path.write_bytes(b'log')
    # logMonoTime runs on across the segments: the second starts 60 s after the first.
    read = {str(logs[0]): {'first_frame_mono': int(1e9), 'error': None,
                           'pose': [[int(t * 1e9), z, v] for t, z, v in _pose(61, 0.0)],
                           'car': [[int(t * 1e9), *rest] for t, *rest in _car(61)]},
            str(logs[1]): {'first_frame_mono': int(61e9), 'error': None,
                           'pose': [[int(t * 1e9), z, v] for t, z, v in _pose(60, 10.0, start=61)],
                           'car': [[int(t * 1e9), *rest] for t, *rest in _car(60, right=True, start=61)]}}
    record = car_motion.build_route(tmp_path, 'r', logs, read)
    assert record['coverage']['seconds'] == 120 and record['coverage']['yaw_device'] == 120
    record = json.loads(json.dumps(record))
    # Frame k of a segment is the step (first frame + k - 1 s, first frame + k s]: frame 0 of segment 1 is the last
    # second of segment 0's samples, joined across the segments.
    assert car_motion.frame_motion(record, 0, 5)['yaw_dps'] == 0.0
    assert car_motion.frame_motion(record, 1, 0)['yaw_dps'] == 0.0
    assert car_motion.frame_motion(record, 1, 3)['yaw_dps'] == pytest.approx(10.0)
    assert car_motion.frame_motion(record, 1, 3)['right'] == 1
    assert car_motion.frame_motion(record, 2, 0) is None and car_motion.frame_motion(None, 0, 0) is None


def test_a_turn_is_its_heading_change_on_a_tight_radius_never_a_speed_threshold():
    import numpy as np

    from nnslr_tools.road_type_metrics import turn_events, turns
    nan = float('nan')
    # A right turn of 90 deg in 4 s at 6 m/s (24 m of path: radius 15 m); later a long bend of 70 deg at 25 m/s.
    yaw = np.array([0, 0, -20, -25, -25, -20, 0, 0] + [10] * 7 + [0, 1, nan, 0], dtype=float)
    speed = np.array([6.0] * 8 + [25.0] * 7 + [25.0] * 4)
    found = turns(yaw, speed)
    assert [(t['start'], t['end'], t['degrees']) for t in found] == [(2, 6, -90.0), (8, 15, 70.0)]
    assert round(found[0]['radius_m'], 1) == round(24 / math.radians(90), 1)
    assert found[1]['radius_m'] > 100
    # The event is the second after the turn; the bend's radius is too large to be an intersection.
    assert turn_events(yaw, speed) == [6]
    assert turn_events(yaw, speed, radius_m=200.0) == [6, 15]
    assert turn_events(yaw, speed, degrees=95.0) == []
