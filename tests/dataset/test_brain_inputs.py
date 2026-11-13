"""The car's file of the brain's inputs, loaded as the training loads it: what a frame hands to the network, and the rules
that must be the same on the server and on the car."""
import ast
from datetime import datetime, timedelta, timezone

import pytest
from zoneinfo import ZoneInfo


def _car():
    from nnslr_tools.brain_targets import BRAIN_INPUTS, VISION
    return BRAIN_INPUTS, VISION


def fed(*frames):
    """SignInputs after the frames (now, signs...)."""
    brain, vision = _car()
    signs = brain.SignInputs(vision)
    for now, *read in frames:
        signs.add(now, list(read))
    return signs


def kph(row):
    return round(float(row[0]) * 130), round(float(row[8]) * 130)


def test_the_file_of_the_car_needs_nothing_of_openpilot_and_reads_no_map():
    from nnslr_tools.brain_targets import BRAIN_INPUTS_PATH
    tree = ast.parse(BRAIN_INPUTS_PATH.read_text(encoding='utf-8'))
    imported = {name.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= {'datetime', 'math', 'numpy'}
    code = ''.join(ast.unparse(node) for node in tree.body[1:])
    assert 'liveMapData' not in code and 'roadName' not in code


def test_one_reading_is_a_single_reading_and_a_second_within_the_time_confirms_it():
    one = fed((10.0, ('maximum_speed:80', .9)))
    assert kph(one.features(10.0)) == (0, 80)
    two = fed((10.0, ('maximum_speed:80', .9)), (10.2, ('maximum_speed:80', .9)))
    assert kph(two.features(10.2)) == (80, 80)
    late = fed((10.0, ('maximum_speed:80', .9)), (12.0, ('maximum_speed:80', .9)))
    assert kph(late.features(12.0)) == (0, 80)


def test_a_number_read_once_leaves_the_confirmed_one_where_it_is():
    signs = fed((10.0, ('maximum_speed:80', .9)), (10.2, ('maximum_speed:80', .9)), (30.0, ('maximum_speed:60', .9)))
    assert kph(signs.features(30.0)) == (80, 60)


@pytest.mark.parametrize('name, flag', [('variable_display:80', 2), ('zone:30', 3), ('road_marking:30', 4)])
def test_every_kind_of_number_counts_and_says_its_kind(name, flag):
    row = fed((10.0, (name, .9)), (10.2, (name, .9))).features(10.2)
    assert row[1] == 1 and [at for at in (2, 3, 4) if row[at]] == [flag]


def test_a_reading_not_sure_enough_or_no_number_is_nothing():
    brain, vision = _car()
    doubt = vision.MIN_CONFIDENCE - .01
    signs = fed((10.0, ('maximum_speed:80', doubt)), (10.2, ('maximum_speed:80', doubt)),
                (10.4, ('other_sign:give_way', .9)), (10.6, ('not_a_sign', .9)))
    assert not signs.features(11.0).any()


def test_an_end_is_confirmed_as_a_number_is_and_the_number_stays_beside_it():
    number = ((10.0, ('maximum_speed:80', .9)), (10.2, ('maximum_speed:80', .9)))
    once = fed(*number, (20.0, ('cancellation', .9)))
    assert once.features(20.0)[6] == 0 and once.features(20.0)[8] == 0 and once.features(20.0)[9] == 1
    ended = fed(*number, (20.0, ('cancellation', .9)), (20.2, ('cancellation', .9)))
    assert ended.features(20.2)[6] == 1 and kph(ended.features(20.2))[0] == 80
    again = fed(*number, (20.0, ('cancellation', .9)), (20.2, ('cancellation', .9)), (30.0, ('maximum_speed:60', .9)),
                (30.2, ('maximum_speed:60', .9)))
    assert again.features(30.2)[6] == 0


def test_the_same_frames_give_the_same_answer_in_any_order_of_the_signs():
    a, b = ('maximum_speed:80', .9), ('maximum_speed:60', .9)
    first = fed((10.0, a, b), (10.2, a, b)).features(10.2)
    second = fed((10.0, b, a), (10.2, b, a)).features(10.2)
    assert (first == second).all()
    assert kph(first)[0] == 60  # the car's rule: the lower on a tie
    assert kph(first)[1] == 80  # same confidence: the class name decides, never the order


def test_the_surest_of_a_frame_is_its_single_reading():
    signs = fed((10.0, ('maximum_speed:80', .75), ('maximum_speed:120', .95)))
    assert kph(signs.features(10.0))[1] == 120


def test_a_number_read_twice_in_a_frame_is_its_surest_reading():
    display, sign = ('variable_display:80', .95), ('maximum_speed:80', .75)
    assert fed((10.0, display, sign), (10.2, display, sign)).features(10.2)[2] == 1
    assert fed((10.0, sign, display), (10.2, sign, display)).features(10.2)[2] == 1


def test_read_in_the_last_second_and_the_age_are_of_the_last_reading_of_the_number():
    number = ((10.0, ('maximum_speed:80', .9)), (10.2, ('maximum_speed:80', .9)))
    assert fed(*number).features(11.0)[7] == 1
    assert fed(*number).features(11.3)[7] == 0
    later = fed(*number, (20.0, ('maximum_speed:80', .9)))
    assert later.features(20.0)[5] == 0 and later.features(20.5)[7] == 1
    assert fed(*number).features(20.0)[5] > 0


def test_a_plate_is_unknown_until_the_reader_says_it():
    brain, _ = _car()
    number = (10.0, ('maximum_speed:30', .9)), (10.2, ('maximum_speed:30', .9))
    assert list(fed(*number).plate_row()) == [0, 0, 0, 1]
    assert list(fed().plate_row()) == [0, 0, 0, 1]


def test_the_plate_is_of_the_number_confirmed_and_its_last_reading_wins():
    read = lambda now, plate, number=30: (now, (f'maximum_speed:{number}', .9, plate))  # noqa: E731
    assert list(fed(read(10.0, 'time'), read(10.2, None)).plate_row()) == [0, 1, 0, 0]
    assert list(fed(read(10.0, 'time'), read(10.2, 'none')).plate_row()) == [1, 0, 0, 0]
    assert list(fed(read(10.0, 'none'), read(10.2, 'none'), read(30.0, 'time'), read(30.2, None)).plate_row()) == [0, 1, 0, 0]


def test_the_plate_of_another_sign_never_stays():
    read = lambda now, plate, number=30: (now, (f'maximum_speed:{number}', .9, plate))  # noqa: E731
    assert list(fed(read(10.0, 'time'), read(10.2, 'time'), read(30.0, None, 50), read(30.2, None, 50)).plate_row()) == [0, 0, 0, 1]
    assert list(fed(read(10.0, 'time'), read(10.2, 'time'), read(30.0, 'none', 50)).plate_row()) == [0, 1, 0, 0]
    assert list(fed(read(10.0, 'time'), read(10.2, 'time'), read(30.0, None), read(30.2, None)).plate_row()) == [0, 1, 0, 0]
    ended = (read(10.0, 'time'), read(10.2, 'time'), (20.0, ('cancellation', .9)), (20.2, ('cancellation', .9)),
             read(30.0, None), read(30.2, None))
    assert list(fed(*ended).plate_row()) == [0, 0, 0, 1]


def test_a_plate_for_wet_roads_has_a_group_of_its_own_and_leaves_the_plate_group_as_it_was():
    read = lambda now, plate, number=80: (now, (f'maximum_speed:{number}', .9, plate))  # noqa: E731
    wet = fed(read(10.0, 'wet'), read(10.2, 'wet'))
    assert list(wet.plate_wet_row()) == [1] and list(wet.plate_row()) == [0, 0, 0, 1]
    assert list(fed(read(10.0, 'time'), read(10.2, 'time')).plate_wet_row()) == [0]
    assert list(fed(read(10.0, None), read(10.2, None)).plate_wet_row()) == [0]
    assert list(fed(read(10.0, 'wet'), read(10.2, 'wet'), read(30.0, None, 50), read(30.2, None, 50)).plate_wet_row()) == [0]


def test_the_hours_of_a_plate_are_those_of_the_number_confirmed():
    brain, _ = _car()
    read = lambda now, hours, number=30: (now, (f'maximum_speed_hours:{number}', .9, 'time', hours))  # noqa: E731
    row = fed(read(10.0, (22, 6)), read(10.2, None)).plate_hours_row()
    assert row[:2] == pytest.approx(brain.hour_row(22)[:2], abs=1e-6) and row[2:4] == pytest.approx(brain.hour_row(6)[:2], abs=1e-6)
    assert row[4] == 1
    assert not fed().plate_hours_row().any()
    assert not fed(read(10.0, None), read(10.2, None)).plate_hours_row().any()
    assert not fed(read(10.0, (22, 6)), read(10.2, None), read(30.0, None, 50), read(30.2, None, 50)).plate_hours_row().any()


def test_hours_change_no_other_group():
    read = lambda now, *said: (now, ('maximum_speed_hours:30', .9, *said))  # noqa: E731
    plain, hours = fed(read(10.0, 'time'), read(10.2, 'time')), fed(read(10.0, 'time', (22, 6)), read(10.2, 'time', (22, 6)))
    assert list(plain.features(11.0)) == list(hours.features(11.0)) and list(plain.plate_row()) == list(hours.plate_row())


@pytest.mark.parametrize('country, zone', [('CH', 'Europe/Zurich'), ('DE', 'Europe/Berlin'), ('FR', 'Europe/Paris'),
                                           ('IT', 'Europe/Rome'), ('AT', 'Europe/Vienna')])
def test_the_local_hour_is_the_one_of_the_time_zones_for_every_hour_of_these_years(country, zone):
    brain, _ = _car()
    place = ZoneInfo(zone)
    moment = datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
    while moment.year < 2031:
        there = moment.astimezone(place)
        expected = there.hour + there.minute / 60
        assert brain.local_hour(moment.timestamp(), country) == pytest.approx(expected), moment
        moment += timedelta(hours=1)


@pytest.mark.parametrize('moment', [datetime(2026, 3, 29, 0, 59, 59, tzinfo=timezone.utc), datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc),
                                    datetime(2026, 10, 25, 0, 59, 59, tzinfo=timezone.utc), datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)])
def test_the_change_of_the_clock_falls_on_its_second(moment):
    brain, _ = _car()
    there = moment.astimezone(ZoneInfo('Europe/Zurich'))
    assert brain.local_hour(moment.timestamp(), 'CH') == pytest.approx(there.hour + there.minute / 60 + there.second / 3600)


def test_with_no_time_or_a_country_outside_the_table_the_hour_is_unknown():
    brain, _ = _car()
    assert brain.local_hour(None, 'CH') is None
    assert brain.local_hour(1_750_000_000, 'US') is None
    assert brain.local_hour(1_750_000_000, None) is None
    assert list(brain.hour_row(None)) == [0, 0, 0]


def test_a_minute_to_midnight_is_next_to_midnight():
    brain, _ = _car()
    before, after = brain.hour_row(23.99), brain.hour_row(0.0)
    assert abs(before - after).max() < .01 and before[2] == 1


def test_a_group_has_the_size_it_says():
    brain, _ = _car()
    sizes = brain.INPUT_SIZES
    signs = fed((10.0, ('maximum_speed:80', .9)))
    rows = {'signs': signs.features(10.0), 'plate': signs.plate_row(), 'hour': brain.hour_row(12.0),
            'lead': brain.lead_row(10.0, 50.0), 'country': brain.country_row('CH'), 'pedals': brain.PedalInputs().row(10.0),
            'speed': brain.speed_row(20.0), 'plate_hours': signs.plate_hours_row(), 'plate_wet': signs.plate_wet_row(),
            'engaged': brain.EngagedInputs().row(10.0), 'sign_distance': brain.sign_distance_row(80.0),
            'motion': brain.MotionInputs().row(10.0), 'road_signs': signs.road_signs_row(10.0),
            'sign_boxes': sign_boxes().row(10.0), 'light_boxes': light_boxes().row(10.0)}
    assert {name: len(row) for name, row in rows.items()} == sizes
    assert brain.pack_inputs(rows, tuple(sizes)).shape == (sum(sizes.values()),)


def test_the_pedals_on_the_car_are_the_ones_of_the_training():
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import NS, _pressed
    brain, _ = _car()
    points = [[2 * NS, False], [3 * NS, True], [5 * NS, False], [9 * NS, True], [10 * NS, False]]
    seconds = list(range(1, 13))
    pressed, known = _pressed(points, numpy.array(seconds) * NS)
    pedals, said = brain.PedalInputs(), []
    for tick in range(1301):  # carState at 100 Hz
        at = tick / 100
        state = [value for t, value in points if t <= tick * NS // 100]
        if state:
            pedals.add(at, state[-1], False)
        if tick % 100 == 0 and tick // 100 in seconds:
            said.append(pedals.row(at))
    assert [row[0] for row in said] == list(pressed)
    assert [row[2] for row in said] == list(known)


def engaged_after(*events):
    """EngagedInputs after (now, 'on' | 'off' | 'gas' | 'brake' | 'up'), a pedal staying down until 'up'."""
    brain, _ = _car()
    engaged, gas, brake = brain.EngagedInputs(), False, False
    for now, what in events:
        if what in ('on', 'off'):
            engaged.add_enabled(now, what == 'on')
        else:
            gas, brake = what == 'gas', what == 'brake'
            engaged.add_pedals(now, gas, brake)
    return engaged


def cause(row):
    return [name for name, at in (('brake', 3), ('gas', 4), ('other', 5)) if row[at]]


def test_openpilot_on_and_never_off_has_no_age_and_no_cause():
    row = engaged_after((1.0, 'on')).row(5.0)
    assert list(row) == [1, 0, 0, 0, 0, 0, 1]
    assert list(engaged_after().row(5.0)) == [0] * 7


def test_braking_and_then_rolling_with_no_pedal_still_says_openpilot_went_off_by_the_brake():
    engaged = engaged_after((1.0, 'on'), (10.0, 'brake'), (10.0, 'off'), (10.5, 'up'))
    row = engaged.row(40.0)
    assert row[0] == 0 and row[1] == 1 and cause(row) == ['brake'] and row[6] == 1
    assert row[2] == pytest.approx(__import__('math').log1p(30.0) / _car()[0].SIGN_AGE_SCALE)


def test_the_brake_seen_a_little_after_openpilot_goes_off_is_still_its_cause():
    engaged = engaged_after((1.0, 'on'), (10.0, 'off'))
    assert cause(engaged.row(10.0)) == ['other']
    engaged.add_pedals(10.8, False, True)
    assert cause(engaged.row(11.0)) == ['brake']
    engaged.add_pedals(30.0, True, False)
    assert cause(engaged.row(31.0)) == ['brake']


def test_a_pedal_long_before_or_after_is_not_the_cause_and_a_held_brake_is():
    late = engaged_after((1.0, 'on'), (5.0, 'brake'), (5.2, 'up'), (10.0, 'off'), (12.0, 'brake'))
    assert cause(late.row(13.0)) == ['other']
    held = engaged_after((1.0, 'on'), (5.0, 'brake'), (10.0, 'off'))
    assert cause(held.row(10.0)) == ['brake']
    gas = engaged_after((1.0, 'on'), (9.9, 'gas'), (10.0, 'off'))
    assert cause(gas.row(10.0)) == ['gas']


def test_engaged_again_keeps_the_last_going_off():
    row = engaged_after((1.0, 'on'), (10.0, 'brake'), (10.0, 'off'), (10.3, 'up'), (20.0, 'on')).row(25.0)
    assert row[0] == 1 and row[1] == 1 and cause(row) == ['brake']


def test_the_engaged_group_on_the_car_is_the_one_of_the_training():
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import NS, engaged_inputs
    brain, _ = _car()
    series = {'sd_enabled': [[1 * NS, False], [2 * NS, True], [10 * NS, False], [15 * NS, True], [20 * NS, False]],
              'brake': [[1 * NS, False], [9 * NS, True], [9 * NS + NS // 2, False], [20 * NS + NS // 2, True],
                        [21 * NS, False]],
              'gas': [[1 * NS, False]]}
    seconds = list(range(1, 26))
    trained = engaged_inputs(series, numpy.array(seconds) * NS)
    car, said = brain.EngagedInputs(), []
    state = lambda name, tick: [value for t, value in series[name] if t <= tick * NS // 100]
    for tick in range(2601):  # carState and selfdriveState at 100 Hz, the pedals first
        at = tick / 100
        gas, brake, enabled = state('gas', tick), state('brake', tick), state('sd_enabled', tick)
        if gas or brake:
            car.add_pedals(at, bool(gas and gas[-1]), bool(brake and brake[-1]))
        if enabled:
            car.add_enabled(at, enabled[-1])
        if tick % 100 == 0 and tick // 100 in seconds:
            said.append(car.row(at))
    assert numpy.allclose(numpy.array(said), trained, atol=1e-6)
    assert cause(trained[seconds.index(12)]) == ['brake'] and cause(trained[seconds.index(21)]) == ['brake']


def test_a_frame_with_no_time_has_the_engaged_group_unknown():
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import NS, engaged_inputs
    rows = engaged_inputs({'sd_enabled': [[1 * NS, True]]}, numpy.array([-1, 2 * NS]))
    assert list(rows[0]) == [0] * 7 and rows[1][0] == 1


def test_the_speed_on_the_car_is_the_one_of_the_road_type_network():
    from nnslr_tools.road_type_data import speed_features
    brain, _ = _car()
    for speed in (None, float('nan'), -1.0, 0.0, 13.9, 75.0):
        assert tuple(brain.speed_row(speed)) == pytest.approx(speed_features(speed))


def test_the_brain_on_the_car_keeps_loading_after_the_motion_group():
    brain, _ = _car()
    assert 'c2303311c6b92f362273754c88fe87321ac0a3ac71967c259c9ad1161be1366d' in brain.SAME_GROUPS


def test_a_flashing_blinker_is_on_and_its_age_counts_from_its_last_flash():
    brain, _ = _car()
    motion = brain.MotionInputs()
    for tick in range(0, 300):  # carState at 100 Hz, the lamp flashing 0.4 s on, 0.4 s off for 2 s
        at = tick / 100
        motion.add_car_state(at, 12.0, at < 2.0 and (tick // 40) % 2 == 0, False, 0.0)
    row = motion.row(3.2)  # the lamp went off with the message of 2.00 s
    assert row[0] == 0 and row[1] == 0 and row[7] == 1
    assert row[2] == pytest.approx((3.2 - 2.0) / brain.BLINKER_AGE_MAX) and row[3] == 1
    assert motion.row(2.9)[0] == 1  # off for less than a second between two flashes counts as on
    assert row[4] == pytest.approx(12.0 / 180)
    assert motion.row(1.9)[0] == 1
    # no device yaw and a car without the sensor: yaw unknown; with deviceMotion, its yaw, left positive
    assert row[6] == 0
    motion.add_device_motion(2.99, -0.5, True)
    assert motion.row(2.99)[5] == pytest.approx(__import__('math').degrees(0.5) / 30) and motion.row(2.99)[6] == 1
    assert list(motion.row(10.0)) == [0, 0, 1, 1, 0, 0, 0, 0]


def test_the_motion_group_on_the_car_is_the_one_of_the_training():
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import NS, motion_inputs
    brain, _ = _car()
    math = __import__('math')
    car, series = brain.MotionInputs(), {name: [] for name in ('blinker_l', 'blinker_r', 'steer', 'yaw_car', 'yaw_dev')}

    def put(name, t, value):  # drive_extract: a change only
        if not series[name] or series[name][-1][1] != value:
            series[name].append([t, value])

    frames, said = list(range(25, 1200, 25)), []  # 4 Hz steps over 12 s
    for tick in range(1200):
        t, at = tick * NS // 100, tick / 100
        left = 300 <= tick < 520 and (tick // 37) % 2 == 0
        right = 800 <= tick < 830
        steer = float(round(30 * math.sin(tick / 200)))
        alive = not 600 <= tick < 780  # carState stops for 1.8 s
        if alive:
            car.add_car_state(at, steer, left, right, 0.0)
            for name, value in (('blinker_l', left), ('blinker_r', right), ('steer', steer), ('yaw_car', 0.0)):
                put(name, t, value)
        elif tick == 649:  # drive_extract's gap: None 0.5 s after the last message (5.99 s)
            for name in ('blinker_l', 'blinker_r', 'steer', 'yaw_car'):
                put(name, t, None)
        if tick % 5 == 0:  # deviceMotion at 20 Hz, valid after 2 s
            yaw = round(-math.radians(steer / 3), 4)
            car.add_device_motion(at, yaw, tick >= 200)
            put('yaw_dev', t, -math.degrees(yaw) if tick >= 200 else None)
        if tick in frames:
            said.append(car.row(at))
    found = motion_inputs(series, numpy.array(frames) * NS // 100)
    numpy.testing.assert_allclose(found, numpy.array(said), atol=1e-6)
    assert found[:, 0].any() and found[:, 1].any() and not found[:, 7].all() and found[:, 6].any()


# -- the boxes of the signs (BoxInputs, the group sign_boxes) ---------------------------------------------------------------
FRAME = (1344, 760)


def sign_boxes():
    brain, vision = _car()
    return brain.BoxInputs(vision, brain.SIGN_BOX_CHANNELS, brain.sign_box_channel(vision))


def grid(row):
    """A row of sign_boxes as (channel, column, [width, height, centre y]), and its known and age."""
    brain, _ = _car()
    return row[:-2].reshape(brain.SIGN_BOX_CHANNELS, brain.BOX_COLUMNS, brain.BOX_NUMBERS), row[-2], row[-1]


def test_the_boxes_are_fifty_numbers_and_all_zero_before_the_first_sign():
    brain, _ = _car()
    assert (brain.BOX_COLUMNS, brain.BOX_NUMBERS, brain.SIGN_BOX_CHANNELS) == (8, 3, 2)
    assert brain.INPUT_SIZES['sign_boxes'] == brain.box_size(2) == 50
    boxes = sign_boxes()
    assert not boxes.row(10.0).any()
    boxes.add(10.0, [], *FRAME)
    assert not boxes.row(10.0).any()


def test_a_box_goes_to_the_column_of_its_centre_with_its_size_over_the_frame():
    boxes = sign_boxes()
    boxes.add(10.0, [('maximum_speed:80', .9, (1008.0, 190.0, 1050.0, 232.0))], *FRAME)
    cells, known, age = grid(boxes.row(10.0))
    assert cells[0, 6] == pytest.approx([42 / 1344, 42 / 760, 211 / 760])  # centre x 1029 of 1344: the seventh of 8
    assert cells.sum() == pytest.approx(cells[0, 6].sum()) and (known, age) == (1, 0)


@pytest.mark.parametrize('name, channel', [
    ('maximum_speed:80', 0), ('maximum_speed_hours:30:22-6', 0), ('maximum_speed_wet:80', 0), ('zone:30', 0),
    ('road_marking:30', 0), ('cancellation', 0), ('variable_display:100', 0), ('variable_display:unknown', 0),
    ('other_sign:stop', 1), ('other_sign:motorway_start', 1), ('other_sign', 1),
    ('maximum_speed:unknown', 0), ('zone:unknown', 0), ('unreadable', 1),
    ('other_sign:traffic_light_red', None), ('other_sign:road_user_car', None), ('not_a_sign', None)])
def test_a_speed_sign_and_any_other_sign_have_a_channel_each_and_a_light_or_a_road_user_none(name, channel):
    brain, vision = _car()
    assert brain.sign_box_channel(vision)(name) == channel
    boxes = sign_boxes()
    boxes.add(10.0, [(name, .9, (100.0, 100.0, 140.0, 140.0))], *FRAME)
    cells, known, _ = grid(boxes.row(10.0))
    assert [bool(cells[at].any()) for at in (0, 1)] == [channel == 0, channel == 1] and known == (channel is not None)


def test_of_two_signs_in_one_column_the_tallest_is_kept_and_another_channel_or_column_keeps_its_own():
    boxes = sign_boxes()
    boxes.add(10.0, [('maximum_speed:80', .99, (700.0, 200.0, 720.0, 220.0)),  # the best score first, as read_signs
                     ('maximum_speed:60', .9, (800.0, 150.0, 840.0, 200.0)),  # the same column (672 to 840), taller
                     ('other_sign:give_way', .9, (760.0, 300.0, 790.0, 330.0)),  # the same column, the other channel
                     ('cancellation', .9, (20.0, 100.0, 50.0, 130.0))], *FRAME)
    cells, _, _ = grid(boxes.row(10.0))
    assert cells[0, 4] == pytest.approx([40 / 1344, 50 / 760, 175 / 760])
    assert cells[1, 4] == pytest.approx([30 / 1344, 30 / 760, 315 / 760])
    assert cells[0, 0] == pytest.approx([30 / 1344, 30 / 760, 115 / 760])
    assert (cells.reshape(-1, 3).any(axis=1)).sum() == 3


def test_a_box_read_under_the_confidence_is_not_taken():
    brain, vision = _car()
    boxes = sign_boxes()
    boxes.add(10.0, [('maximum_speed:80', vision.MIN_CONFIDENCE - .01, (700.0, 200.0, 720.0, 220.0))], *FRAME)
    assert not boxes.row(10.0).any()
    boxes.add(10.2, [('maximum_speed:80', vision.MIN_CONFIDENCE, (700.0, 200.0, 720.0, 220.0))], *FRAME)
    assert boxes.row(10.2)[-2] == 1


def test_when_the_sign_leaves_the_columns_empty_and_the_age_grows():
    brain, _ = _car()
    boxes = sign_boxes()
    boxes.add(10.0, [('maximum_speed:80', .9, (1300.0, 100.0, 1340.0, 140.0))], *FRAME)
    held = boxes.row(10.1)  # a step between two readings: the last reading's columns
    assert grid(held)[0][0, 7].all() and held[-1] == pytest.approx(__import__('math').log1p(.1) / brain.SIGN_AGE_SCALE)
    boxes.add(10.25, [], *FRAME)
    for now, age in ((10.25, .25), (40.0, 30.0)):
        cells, known, said = grid(boxes.row(now))
        assert not cells.any() and known == 1
        assert said == pytest.approx(__import__('math').log1p(age) / brain.SIGN_AGE_SCALE)
    # a reading with only a sign of no channel is a reading with no box
    boxes.add(41.0, [('other_sign:road_user_car', .9, (100.0, 100.0, 300.0, 300.0))], *FRAME)
    assert grid(boxes.row(41.0))[2] == pytest.approx(__import__('math').log1p(31.0) / brain.SIGN_AGE_SCALE)


def test_the_two_definitions_of_the_frame_give_the_same_fractions():
    numpy = pytest.importorskip('numpy')
    signs = [('maximum_speed:80', .9, (1008.3, 190.1, 1050.7, 232.9)), ('other_sign:stop', .9, (167.9, 300.0, 168.1, 380.0)),
             ('cancellation', .8, (-20.0, -10.0, 30.5, 41.5)), ('maximum_speed:50', .9, (1330.0, 700.0, 1400.0, 790.0))]
    low, full = sign_boxes(), sign_boxes()
    low.add(10.0, signs, 1344, 760)
    full.add(10.0, [(name, confidence, tuple(2 * value for value in box)) for name, confidence, box in signs], 2688, 1520)
    assert numpy.array_equal(low.row(10.0), full.row(10.0)) and numpy.count_nonzero(low.row(10.0)[:-2]) == 12


def test_a_box_over_the_edge_counts_for_its_part_inside_the_frame():
    boxes = sign_boxes()
    boxes.add(10.0, [('cancellation', .9, (-20.0, -10.0, 30.0, 40.0)), ('maximum_speed:50', .9, (1330.0, 700.0, 1400.0, 790.0)),
                     ('other_sign:stop', .9, (1350.0, 100.0, 1400.0, 150.0))], *FRAME)  # the last one all outside
    cells, _, _ = grid(boxes.row(10.0))
    assert cells[0, 0] == pytest.approx([30 / 1344, 40 / 760, 20 / 760])
    assert cells[0, 7] == pytest.approx([14 / 1344, 60 / 760, 730 / 760])
    assert not cells[1].any() and cells.max() <= 1 and cells.min() >= 0


def as_the_car(readings, boxes, times):
    """The rows the car gives at times: every frame read given to BoxInputs as brain.BrainReader.add_signs gives it (one
    the record has no reading at had no sign), then the row at that frame, where the brain steps."""
    numpy = pytest.importorskip('numpy')
    signs = {seconds: [(name, confidence, box) for (name, confidence), box in zip(read, places)]
             for (seconds, read), places in zip(readings, boxes)}
    car, rows = sign_boxes(), []
    for now in times:
        car.add(now, signs.get(now, []), *FRAME)
        rows.append(car.row(now))
    return numpy.array(rows)


def test_the_boxes_on_the_car_are_the_ones_of_the_training():
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import box_inputs
    times = [n * .25 for n in range(4 * 30)]  # every frame read of 30 s, each one a step of the brain
    readings, boxes = [], []
    for at, now in enumerate(times):
        read = []
        if 8 <= at < 40:  # a number that comes nearer on the right, lost on two frames
            if at not in (20, 21):
                side = 10.0 + at
                read.append((['maximum_speed:80', .7 + at / 200], [900.0 + 9 * at, 300.0 - 4 * at, 900.0 + 9 * at + side, 300.0 - 4 * at + side]))
        if 30 <= at < 60:  # a give way on the left meanwhile, and a light the group does not take
            read.append((['other_sign:give_way', .95], [300.0 - 8 * (at - 30), 350.0, 330.0 - 8 * (at - 30), 380.0]))
            read.append((['other_sign:traffic_light_red', .95], [600.0, 100.0, 620.0, 150.0]))
        if read:
            readings.append([now, [item[0] for item in read]])
            boxes.append([item[1] for item in read])
    found = box_inputs(numpy.array(times), readings, boxes)
    assert numpy.array_equal(found, as_the_car(readings, boxes, times))
    assert found[7].sum() == 0 and found[8][-2] == 1 and found[20][:-2].sum() == 0 and found[20][-1] > 0
    assert found[35][:24].any() and found[35][24:48].any() and found[-1][:-2].sum() == 0 and found[-1][-2] == 1
    # one frame a second (the summary's rate) sees the same rows at its frames
    assert numpy.array_equal(box_inputs(numpy.array(times[::4]), readings, boxes), found[::4])
    # a drive not read, or read before the boxes were kept: zeros
    for unread in ((None, None), (readings, None), ([], [])):
        assert not box_inputs(numpy.array(times), *unread).any()


# -- the boxes of the traffic lights (BoxInputs, the group light_boxes) -----------------------------------------------------
def light_boxes():
    brain, vision = _car()
    return brain.BoxInputs(vision, len(brain.LIGHTS), brain.light_box_channel)


def test_the_lights_are_ninety_eight_numbers_a_channel_for_what_is_lit():
    brain, _ = _car()
    assert brain.LIGHTS == ('red', 'amber', 'green', 'off')
    assert brain.INPUT_SIZES['light_boxes'] == brain.box_size(4) == 98
    assert not light_boxes().row(10.0).any()


@pytest.mark.parametrize('name, channel', [
    ('other_sign:traffic_light_red', 0), ('other_sign:traffic_light_amber', 1), ('other_sign:traffic_light_green', 2),
    ('other_sign:traffic_light_off', 3), ('other_sign:traffic_light_blue', None), ('other_sign:stop', None),
    ('other_sign:road_user_car', None), ('maximum_speed:80', None), ('unreadable', None), ('not_a_sign', None)])
def test_a_light_goes_to_the_channel_of_what_is_lit_and_its_column_and_nothing_else_goes_there(name, channel):
    brain, _ = _car()
    assert brain.light_box_channel(name) == channel
    lights = light_boxes()
    lights.add(10.0, [(name, .9, (1008.0, 100.0, 1028.0, 150.0))], *FRAME)
    row = lights.row(10.0)
    cells = row[:-2].reshape(len(brain.LIGHTS), brain.BOX_COLUMNS, brain.BOX_NUMBERS)
    if channel is None:
        assert not row.any()
    else:
        assert cells[channel, 6] == pytest.approx([20 / 1344, 50 / 760, 125 / 760]) and row[-2] == 1
        assert cells.sum() == pytest.approx(cells[channel, 6].sum())


def test_a_light_read_under_the_confidence_is_not_taken_and_a_sign_is_no_light():
    brain, vision = _car()
    lights = light_boxes()
    lights.add(10.0, [('other_sign:traffic_light_red', vision.MIN_CONFIDENCE - .01, (700.0, 100.0, 720.0, 150.0)),
                      ('maximum_speed:80', .99, (700.0, 200.0, 740.0, 240.0))], *FRAME)
    assert not lights.row(10.0).any()


def test_the_lights_on_the_car_are_the_ones_of_the_training():
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import box_inputs
    times = [n * .25 for n in range(4 * 20)]
    readings, boxes = [], []
    for at, now in enumerate(times):
        read = []
        if 10 <= at < 50:  # a red light that comes nearer, green from frame 30, and a speed sign the group does not take
            light = 'other_sign:traffic_light_red' if at < 30 else 'other_sign:traffic_light_green'
            read.append(([light, .95], [600.0 + 4 * at, 120.0 - at, 620.0 + 5 * at, 160.0 + at]))
            read.append((['maximum_speed:50', .95], [1100.0, 300.0, 1140.0, 340.0]))
        if read:
            readings.append([now, [item[0] for item in read]])
            boxes.append([item[1] for item in read])
    found = box_inputs(numpy.array(times), readings, boxes, 'light_boxes')
    signs = {seconds: [(name, confidence, box) for (name, confidence), box in zip(read, places)]
             for (seconds, read), places in zip(readings, boxes)}
    car, said = light_boxes(), []
    for now in times:
        car.add(now, signs.get(now, []), *FRAME)
        said.append(car.row(now))
    assert numpy.array_equal(found, numpy.array(said))
    assert found[20][:24].any() and not found[20][24:96].any()  # red: the first channel only
    assert found[40][48:72].any() and not found[40][:48].any()  # green: the third
    assert found[-1][-2] == 1 and not found[-1][:-2].any()
    # the signs' own group is unchanged by the lights
    assert numpy.array_equal(box_inputs(numpy.array(times), readings, boxes), as_the_car(readings, boxes, times))


def test_the_boxes_of_a_real_drive_are_on_the_car_what_they_are_in_the_training():
    """One record of the reader (data/derived/road_type_signs), every frame it read: 4 a second, the brain's steps."""
    import json
    from pathlib import Path
    numpy = pytest.importorskip('numpy')
    from nnslr_tools.brain_targets import INPUT_SIZES, box_inputs
    from nnslr_tools.road_type_signs import SEGMENT_SECONDS, SIGNS_RELATIVE, VIDEO_FPS, VIDEO_STEP
    folder = Path(__file__).resolve().parents[2] / 'data' / SIGNS_RELATIVE
    record = next((found for found in (json.loads(path.read_text(encoding='utf-8'))
                                       for path in sorted(folder.glob('releases/*/*.json'))[:5])
                   if found.get('boxes') and len(found['boxes']) > 100), None)
    if record is None:
        pytest.skip('no drive read with its boxes on this machine')
    times = [int(segment) * SEGMENT_SECONDS + at * VIDEO_STEP / VIDEO_FPS
             for segment, frames in sorted(record['frames'].items(), key=lambda item: int(item[0])) for at in range(frames)]
    assert {seconds for seconds, _ in record['readings']} <= set(times)  # every reading is at a frame the brain steps at
    found = box_inputs(numpy.array(times), record['readings'], record['boxes'])
    assert found.shape == (len(times), INPUT_SIZES['sign_boxes']) and found.dtype == numpy.float32
    assert numpy.array_equal(found, as_the_car(record['readings'], record['boxes'], times))
    with_a_box = (found[:, :-2] != 0).any(axis=1)
    assert with_a_box.sum() > 100 and found[:, :-2].min() >= 0 and found[:, :-2].max() <= 1
