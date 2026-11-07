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
            'engaged': brain.EngagedInputs().row(10.0), 'sign_distance': brain.sign_distance_row(80.0)}
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
