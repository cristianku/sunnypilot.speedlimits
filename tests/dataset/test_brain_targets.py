"""What Speed review and Sign review become for the training: the answers, what stays out of the loss, the split, the
inputs. numpy comes in only inside the tests: collecting them must leave it out (tests/cli/test_import_isolation.py)."""
import json
import sqlite3

import pytest

NS = 1_000_000_000
ROUTE = '00000001--abcdef0123'


def _np():
    import numpy
    return numpy


def _targets():
    from nnslr_tools import brain_targets
    return brain_targets


def block(start, end, kph):
    return {'start_ns': start * NS, 'end_ns': end * NS, 'kph': kph}


def mark(start, end):
    return {'start_ns': start * NS, 'end_ns': None if end is None else end * NS}


def seconds(*values):
    return (_np().array(values, dtype=_np().float64) * NS).astype(_np().int64)


def test_a_label_is_a_class_of_one_kilometre_and_below_one_there_is_none():
    np, targets = _np(), _targets()
    assert targets.speed_class(1) == 0 and targets.speed_class(100) == 99 and targets.speed_class(200) == 199
    assert targets.speed_class(250) == 199 and targets.speed_class(52.6) == 52
    assert targets.speed_class(0) == -1 and targets.speed_class(0.6) == -1 and targets.speed_class(None) == -1
    assert targets.class_kph(targets.speed_class(123)) == 123
    assert targets.speed_class(np.array([0, 1, 1.4, 50, 300, np.nan])).tolist() == [-1, 0, 0, 49, 199, -1]
    assert targets.class_kph(np.arange(3)).tolist() == [1, 2, 3]
    assert targets.SPEED_CLASSES == 200 and targets.NONE_AHEAD == 200 and not hasattr(targets, 'GRID_KPH')


def test_the_kph_of_the_classes_come_out_of_a_tensor_too():
    torch = pytest.importorskip('torch')
    assert _targets().class_kph(torch.arange(3)).tolist() == [1, 2, 3]


def test_his_speed_is_the_answer_where_no_block_says_otherwise():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(5, 15, 25, 35), [80, None, 0, 120], [])
    assert list(classes) == [79, -1, -1, 119]
    assert list(out) == [0, targets.NO_SPEED, targets.STOPPED, 0]


def test_a_block_says_the_speed_and_the_moments_around_its_edges_are_out():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(5, 8, 9, 11, 15, 19, 21, 22), [80] * 8, [block(10, 20, 100)])
    assert list(classes) == [79, 79, -1, -1, 99, -1, -1, 79]
    boundary = targets.BOUNDARY
    assert list(out) == [0, 0, boundary, boundary, 0, boundary, boundary, 0]


def test_a_block_with_no_speed_takes_its_moments_out():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(5, 9, 15, 21, 25), [80] * 5, [block(10, 20, None)])
    assert list(classes) == [79, -1, -1, -1, 79]
    assert list(out) == [0, targets.BOUNDARY, targets.BLOCKED_OUT, targets.BOUNDARY, 0]


def test_a_block_of_his_own_speed_is_no_jump_and_neither_is_his_speed_changing():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(9, 11, 19, 21), [100, 100, 100, 100], [block(10, 20, 100)])
    assert list(out) == [0, 0, 0, 0] and list(classes) == [99] * 4
    classes, out = targets.speed_answers(seconds(9, 10, 11), [80, 90, 100], [])
    assert list(out) == [0, 0, 0] and list(classes) == [79, 89, 99]
    # Two blocks of one speed side by side are one, and an edge with no moment on one side of it is no jump.
    _classes, out = targets.speed_answers(seconds(1, 9, 11, 19), [80] * 4, [block(0, 10, 100), block(10, 20, 100)])
    assert list(out) == [0, 0, 0, 0]


def test_a_block_of_naught_is_a_car_that_stands_and_has_no_class():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(5, 15), [80, 80], [block(10, 30, 0)])
    assert list(classes) == [79, -1] and out[1] == targets.STOPPED


def test_where_a_machine_held_the_speed_his_foot_gave_no_answer_but_his_block_still_does():
    targets = _targets()
    held = [False, True, True, True, False]
    classes, out = targets.speed_answers(seconds(5, 15, 25, 35, 45), [80] * 5, [], held)
    assert list(classes) == [79, -1, -1, -1, 79] and list(out) == [0] + [targets.CRUISE] * 3 + [0]
    classes, out = targets.speed_answers(seconds(5, 15, 25, 35, 45), [80] * 5, [block(20, 30, 100)], held)
    assert list(classes) == [79, -1, 99, -1, 79] and list(out) == [0, targets.CRUISE, 0, targets.CRUISE, 0]
    # No speed stays no speed.
    assert list(targets.speed_answers(seconds(5), [None], [], [True])[1]) == [targets.NO_SPEED]
    assert targets.OUT_REASONS[targets.CRUISE] == 'cruise'


def test_a_moment_with_no_time_has_no_answer_from_a_block_either():
    targets = _targets()
    ns = _np().array([-1, 15 * NS], dtype=_np().int64)
    classes, out = targets.speed_answers(ns, [None, 80], [block(0, 20, 100)])
    assert list(classes) == [-1, 99] and list(out) == [targets.NO_SPEED, 0]


def test_no_mark_is_no_answer_at_all():
    target, weight = _targets().works_answers(seconds(5, 50, 500), [], None)
    assert weight.sum() == 0 and target.sum() == 0


def test_inside_a_mark_is_works_also_past_where_the_drive_was_checked():
    target, weight = _targets().works_answers(seconds(20, 30, 40), [mark(10, 50)], None)
    assert list(target) == [1, 1, 1]
    assert list(weight) == [1, 1, 1]


def test_outside_the_marks_is_no_works_only_up_to_where_a_person_looked():
    target, weight = _targets().works_answers(seconds(5, 20, 60, 90, 120), [mark(10, 50)], 100 * NS)
    assert list(target) == [0, 1, 0, 0, 0]
    assert list(weight) == [1, 1, 1, 1, 0]


def test_a_mark_with_no_end_tells_nothing_from_its_start_on():
    target, weight = _targets().works_answers(seconds(5, 20, 60), [mark(10, None)], 100 * NS)
    assert list(target) == [0, 0, 0]
    assert list(weight) == [1, 0, 0]


def test_a_tap_is_only_so_exact():
    _target, weight = _targets().works_answers(seconds(9, 11, 13, 47, 49, 51, 53), [mark(10, 50)], 100 * NS)
    assert list(weight) == [0, 0, 1, 1, 0, 0, 1]


def kind_mark(kind, start, end):
    return {**mark(start, end), 'kind': kind}


def test_each_kind_of_road_is_the_roadworks_of_its_own_marks_and_its_own_check():
    targets = _targets()
    ns = seconds(5, 20, 60, 90)
    road = [kind_mark('narrow', 10, 50), kind_mark('gravel', 0, 100), kind_mark('gone', 0, 100)]
    target, weight = targets.road_answers(ns, road, {'narrow': 70 * NS}, ('narrow', 'country', 'gravel'))
    assert target.shape == weight.shape == (4, 3)
    # Narrow: inside its mark, checked to 70 s.
    assert target[:, 0].tolist() == [0, 1, 0, 0] and weight[:, 0].tolist() == [1, 1, 1, 0]
    # Country: no mark and nobody looked, nothing to say.
    assert not target[:, 1].any() and not weight[:, 1].any()
    # Gravel at the same time as narrow: a road can be both.
    assert target[:, 2].tolist() == [1, 1, 1, 1] and target[1].tolist() == [1, 0, 1]
    # A kind not asked for is not read.
    assert targets.road_answers(ns, road, {}, ())[0].shape == (4, 0)


def test_the_kinds_of_road_are_read_in_their_order_and_a_file_without_them_has_the_first_ones(tmp_path):
    import sqlite3 as sql
    from nnslr_tools.speed_review import FIRST_ROAD_KINDS, SpeedReviewStore, db_path
    targets = _targets()
    first = tuple(kind for kind, _label in FIRST_ROAD_KINDS)
    assert targets.road_kinds(tmp_path) == first
    SpeedReviewStore(tmp_path)
    connection = sql.connect(db_path(tmp_path))
    connection.execute("INSERT INTO speed_road_kinds (kind, label, position, reviewer, at) VALUES ('cobbles', 'Cobbles', -1, 'a', 'now')")
    connection.commit()
    connection.close()
    assert targets.road_kinds(tmp_path) == ('cobbles', *first)
    # A file of before the kinds of road.
    old = tmp_path / 'old'
    path = db_path(old)
    path.parent.mkdir(parents=True)
    connection = sql.connect(path)
    connection.execute('CREATE TABLE speed_works (route TEXT, start_ns INTEGER, end_ns INTEGER, reviewer TEXT, revision INTEGER)')
    connection.close()
    assert targets.road_kinds(old) == first


def test_the_end_of_works_leaves_the_speed_alone():
    targets = _targets()
    ns = seconds(20, 60)
    classes, _out = targets.speed_answers(ns, [80, 80], [])
    target, _weight = targets.works_answers(ns, [mark(10, 50)], 100 * NS)
    assert list(target) == [1, 0]
    assert list(classes) == [targets.speed_class(80)] * 2



SPLIT = {'cells': {'10,10': 'train', '20,20': 'test', '30,30': 'validation'},
         'partitions': {'old': ['train', 'test']}, 'frame_keys': {'old': ['0/0', '0/1']}}


def at_cell(key, cell):
    from nnslr_tools.road_type_data import PLACE_CELL
    if cell is None:
        return {'key': key, 'latitude': None, 'longitude': None}
    return {'key': key, 'latitude': (cell[0] + .5) * PLACE_CELL[0], 'longitude': (cell[1] + .5) * PLACE_CELL[1]}


def test_a_frozen_drive_keeps_its_split_frame_by_frame():
    frames = [at_cell('0/0', (10, 10)), at_cell('0/1', (20, 20))]
    assert _targets().frame_partitions('old', frames, SPLIT) == ['train', 'test']


def test_a_frozen_drive_with_other_frames_stops_loudly():
    with pytest.raises(ValueError):
        _targets().frame_partitions('old', [at_cell('0/0', (10, 10))], SPLIT)
    with pytest.raises(ValueError):
        _targets().frame_partitions('old', [at_cell('0/0', (10, 10)), at_cell('0/9', (20, 20))], SPLIT)


def test_a_later_drive_goes_by_the_area_of_each_frame():
    frames = [at_cell('0/0', (10, 10)), at_cell('0/1', (20, 20)), at_cell('0/2', (30, 30)),
              at_cell('0/3', (21, 21)), at_cell('0/4', (50, 50))]
    assert _targets().frame_partitions('new', frames, SPLIT) == ['train', 'test', 'validation', 'buffer', 'train']


def test_a_later_drive_without_any_position_is_held_back():
    assert _targets().frame_partitions('new', [at_cell('0/0', None)], SPLIT) == ['buffer']


def site_drive(partitions, cells, marks):
    return {'ns': _np().arange(len(partitions), dtype=_np().int64) * 30 * NS, 'partitions': list(partitions), 'cells': list(cells),
            'marks': marks}


def test_a_site_in_test_has_no_frame_in_train_approach_and_exit_included():
    drive = site_drive(['train', 'train', 'test', 'train', 'train', 'train', 'train', 'train'],
                       [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)], [mark(60, 90)])
    _targets().keep_sites_whole([drive])
    near = [i for i, t in enumerate(drive['ns']) if 0 * NS <= t < 150 * NS]
    assert all(drive['partitions'][i] != 'train' for i in near)
    assert drive['partitions'][-1] == 'train'


def test_the_same_site_on_another_drive_leaves_train_too():
    held = site_drive(['test'] * 4, [(1, 1), (1, 2), (1, 3), (1, 4)], [mark(30, 60)])
    again = site_drive(['train'] * 4, [(1, 1), (1, 2), (1, 3), (1, 4)], [mark(30, 60)])
    other = site_drive(['train'] * 4, [(9, 1), (9, 2), (9, 3), (9, 4)], [mark(30, 60)])
    _targets().keep_sites_whole([held, again, other])
    assert set(again['partitions']) == {'buffer'}
    assert set(other['partitions']) == {'train'}


def test_a_site_all_in_train_stays():
    drive = site_drive(['train'] * 4, [(1, 1), (1, 2), (1, 3), (1, 4)], [mark(30, 60)])
    _targets().keep_sites_whole([drive])
    assert set(drive['partitions']) == {'train'}


def test_the_rest_of_a_drive_stays_in_the_sequence_and_only_its_loss_is_masked():
    numpy = _np()
    drive = {'partitions': ['train', 'test', 'train'], 'speed_weight': numpy.ones(3, dtype=numpy.float32),
             'works_weight': numpy.array([1, 1, 0], dtype=numpy.float32)}
    speed, works = _targets().loss_weights(drive, 'train')
    assert list(speed) == [1, 0, 1]
    assert list(works) == [1, 0, 0]


def test_a_drive_goes_in_order_in_pieces_covering_every_frame_once():
    chunks = _targets().drive_chunks
    assert chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert chunks(10, 4, 3) == [(0, 3), (3, 7), (7, 10)]
    assert chunks(3, 4) == [(0, 3)]
    assert chunks(0, 4) == []


def test_a_piece_that_learns_has_only_train_frames():
    pieces = _targets().learning_pieces
    names = ['train'] * 5 + ['buffer'] + ['train'] * 3
    assert pieces(names, 4) == [(0, 4, True, True), (4, 5, True, False), (5, 6, False, False), (6, 9, True, False)]
    assert pieces(names, 4, 2) == [(0, 2, True, True), (2, 5, True, False), (5, 6, False, False), (6, 8, True, False),
                                   (8, 9, True, False)]
    assert pieces([], 4) == []


def test_a_held_out_frame_is_in_no_piece_and_the_memory_starts_again_after_it():
    pieces = _targets().learning_pieces
    names = ['train'] * 3 + ['test'] * 2 + ['buffer'] + ['validation'] + ['train'] * 2
    assert pieces(names, 4) == [(0, 3, True, True), (5, 6, False, True), (7, 9, True, True)]
    assert pieces(['test'] * 3, 2) == []


def test_the_memory_starts_again_at_a_cut_inside_a_stretch():
    pieces = _targets().learning_pieces
    assert pieces(['train'] * 6, 4, cuts={2}) == [(0, 2, True, True), (2, 6, True, True)]
    assert pieces(['train'] * 6, 4, cuts={4}) == [(0, 4, True, True), (4, 6, True, True)]


def test_a_gap_in_the_frames_cuts_the_memory():
    targets = _targets()
    assert targets.memory_cuts(seconds(0, 1, 2, 8, 9)) == {3}
    assert targets.memory_cuts(seconds(0, 1, 6)) == set()
    assert targets.memory_cuts(_np().array([0, -1, -1, 9 * NS, 10 * NS])) == {3}


def checked(**changes):
    numpy = _np()
    return {'route': 'r1', 'frames': [{}] * 3, 'partitions': ['train', 'validation', 'test'], 'signs_read': True,
            'reader_current': True, 'speed_weight': numpy.ones(3, dtype=numpy.float32),
            'works_weight': numpy.zeros(3, dtype=numpy.float32), **changes}


def test_the_training_runs_on_drives_read_by_todays_reader():
    _targets().check_drives([checked()])


@pytest.mark.parametrize('change, says', [({'signs_read': False, 'reader_current': False}, 'no reading of the signs'),
                                          ({'reader_current': False}, 'another reader'), ({'frames': []}, 'no frame')])
def test_the_training_refuses_what_would_go_wrong_without_a_word(change, says):
    targets = _targets()
    with pytest.raises(ValueError, match=says) as found:
        targets.check_drives([checked(), checked(route='r2', **change)])
    if 'reader' in str(found.value) or 'reading' in str(found.value):
        assert 'r2' in str(found.value) and 'r1' not in str(found.value) and targets.READ_COMMAND in str(found.value)


def test_a_sign_read_counts_from_its_second_on_and_never_before():
    numpy = _np()
    readings = [[10.0, [['maximum_speed:100', .9]]], [10.2, [['maximum_speed:100', .9]]], [20.0, [['cancellation', .9]]],
                [20.2, [['cancellation', .9]]], [30.0, [['variable_display:80', .9]]], [30.2, [['variable_display:80', .9]]],
                [31.0, [['other_sign:give_way', .9]]]]
    found, plates, _hours, _wet = _targets().sign_inputs(numpy.array([9.0, 10.0, 11.0, 15.0, 21.0, 31.0, 40.0]), readings)
    assert found[0].sum() == 0
    assert found[1][:2].sum() == 0 and found[1][8] == pytest.approx(100 / 130)
    assert found[2][:3] == pytest.approx([100 / 130, 1, 0])
    assert found[2][7] == 1 and found[3][7] == 0
    assert found[3][6] == 0 and found[4][6] == 1
    assert found[5][:3] == pytest.approx([80 / 130, 1, 1]) and found[5][6] == 0
    assert (plates[:, -1] == 1).all()


def test_a_drive_not_read_yet_has_its_signs_unknown():
    numpy = _np()
    signs, plates, _hours, _wet = _targets().sign_inputs(numpy.array([1.0, 2.0]), None)
    assert signs.sum() == 0 and (plates[:, -1] == 1).all() and plates.sum() == 2


def inputs_of(ns, series, readings=None, gps=()):
    frames = [{'segment': 0, 'index': int(t // NS)} for t in ns]
    return _targets().drive_inputs(frames, ns, {'series': series, 'gps': gps}, readings, None)


def test_no_input_of_a_frame_knows_what_comes_after_it():
    numpy = _np()
    ns = seconds(10, 20, 30)
    past = {'gas': [[1 * NS, False], [9 * NS, True], [10 * NS, False]], 'brake': [[1 * NS, False]],
            'v_ego': [[1 * NS, 50.0]], 'lead_d': [[1 * NS, 40.0]], 'lead_v': [[1 * NS, 50.0]]}
    future = {name: [*points, [21 * NS, points[-1][1] if name == 'brake' else 99.0]] for name, points in past.items()}
    future['gas'] = [*past['gas'], [21 * NS, True]]
    future['brake'] = [*past['brake'], [21 * NS, True]]
    readings = [[5.0, [['maximum_speed:80', .9]]], [5.2, [['maximum_speed:80', .9]]]]
    later = [*readings, [21.0, [['maximum_speed:120', .9]]]]
    fixes = [[5 * NS, 47.0, 8.0, 1_750_000_000_000]]
    before, after = inputs_of(ns, past, readings, fixes), inputs_of(ns, future, later, [*fixes, [21 * NS, 47.0, 8.0, 1_760_000_000_000]])
    for name in before:
        assert numpy.array_equal(before[name][:2], after[name][:2]), name
        assert before[name].shape[1] == _targets().INPUT_SIZES[name]
    assert not numpy.array_equal(before['signs'][2], after['signs'][2])


def test_gas_pressed_in_the_last_second_counts_and_unknown_is_not_released():
    ns = seconds(10, 20)
    pedals = inputs_of(ns, {'gas': [[1 * NS, False], [9 * NS + 500_000_000, True], [10 * NS - 1, False]],
                            'brake': [[1 * NS, False]]})['pedals']
    assert list(pedals[0]) == [1, 0, 1]
    assert list(pedals[1]) == [0, 0, 1]
    assert list(inputs_of(ns, {})['pedals'][0]) == [0, 0, 0]


def test_the_lead_and_the_speed_are_unknown_until_they_are_known():
    ns = seconds(5, 15)
    found = inputs_of(ns, {'lead_d': [[1 * NS, None], [10 * NS, 300.0]], 'lead_v': [[1 * NS, None], [10 * NS, 65.0]],
                           'v_ego': [[10 * NS, 36.0]]})
    assert list(found['lead'][0]) == [0, 0, 0]
    assert list(found['lead'][1]) == pytest.approx([2, .5, 1])
    assert found['speed'][0][1] == 0 and found['speed'][1][1] == 1


def test_a_country_outside_the_list_is_one_of_its_own(monkeypatch):
    targets = _targets()
    monkeypatch.setattr(targets, 'frame_country', lambda latitude, _longitude, _root: {1.0: 'CH', 2.0: 'SM'}.get(latitude))
    gps = [[1 * NS, 1.0, 8.0, 1000], [2 * NS, 2.0, 8.0, 2000], [3 * NS, 3.0, 8.0, 3000]]
    frames = [{'segment': 0, 'index': second} for second in (1, 2, 3)]
    found = targets.drive_inputs(frames, seconds(1, 2, 3), {'series': {}, 'gps': gps}, None, None)['country']
    assert found[0][targets.COUNTRIES.index('CH')] == 1
    assert found[1][-1] == 1 and found[2][-1] == 1
    assert found.sum() == 3


def test_a_frame_has_the_country_of_its_last_fix_and_never_of_a_later_one(monkeypatch):
    numpy, targets = _np(), _targets()
    monkeypatch.setattr(targets, 'frame_country', lambda latitude, _longitude, _root: {1.0: 'DE', 2.0: 'CH'}[latitude])
    gps = [[10 * NS, 1.0, 8.0, 1000], [20 * NS, 2.0, 8.0, 2000]]
    assert targets.countries_so_far(gps, seconds(5, 10, 19, 20, 30), None) == [None, 'DE', 'DE', 'CH', 'CH']
    assert targets.countries_so_far(gps, numpy.array([-1]), None) == [None]
    assert targets.countries_so_far([], seconds(5), None) == [None]


def test_before_the_first_fix_neither_the_country_nor_the_hour_is_known():
    targets = _targets()
    found = inputs_of(seconds(5, 11), {}, gps=[[10 * NS, 47.0, 8.0, 1_750_000_000_000]])
    assert found['country'][0][-1] == 1 and not found['hour'][0].any()
    assert found['country'][1][targets.COUNTRIES.index('CH')] == 1 and found['hour'][1][2] == 1


def test_the_gps_clock_runs_on_from_its_last_fix_and_says_nothing_before_the_first():
    numpy = _np()
    gps = [[10 * NS, 47.0, 8.0, 1_000_000], [20 * NS, 47.0, 8.0, None], [30 * NS, 47.0, 8.0, 1_020_000]]
    assert _targets().unix_at(gps, numpy.array([5 * NS, 10 * NS, 25 * NS, 31 * NS, -1])) == [None, 1000, 1015, 1021, None]


def test_the_hour_of_a_frame_is_the_local_one_from_the_gps_clock():
    ns = seconds(10, 20)
    summer_noon_utc = 1_751_371_200_000  # 1 July 2025 12:00 UTC
    hour = inputs_of(ns, {}, gps=[[15 * NS, 47.0, 8.0, summer_noon_utc]])['hour']
    assert list(hour[0]) == [0, 0, 0]
    assert hour[1][2] == 1
    assert hour[1][0] == pytest.approx(-.5, abs=.01)  # 14:05 on the circle


def test_a_comparison_can_leave_out_the_driver():
    numpy = _np()
    targets = _targets()
    inputs = {name: numpy.zeros((2, size), dtype=numpy.float32) for name, size in targets.INPUT_SIZES.items()}
    without = tuple(name for name in targets.INPUT_SIZES if name not in targets.DRIVER_INPUTS)
    assert targets.pack_inputs(inputs, without).shape == (2, sum(targets.INPUT_SIZES[name] for name in without))
    assert targets.pack_inputs(inputs, tuple(targets.INPUT_SIZES)).shape == (2, sum(targets.INPUT_SIZES.values()))


def test_engaged_is_made_and_never_learned():
    """openpilot on or off only confused the network (7 Nov 2026): the group is still made as on the car, no brain learns it."""
    targets = _targets()
    assert 'engaged' in targets.INPUT_SIZES and targets.LEFT_OUT == ('engaged',)
    assert 'engaged' not in targets.DRIVER_INPUTS


def write_drive(root, route, frames):
    """frames: [(segment, index)], each with a picture and a decoded frame index of index * 20."""
    for segment in {segment for segment, _index in frames}:
        folder = root / 'derived' / 'frames' / route / str(segment)
        folder.mkdir(parents=True, exist_ok=True)
        lines = [{'segment_index': s, 'output_index': i, 'decoded_frame_index': i * 20} for s, i in frames if s == segment]
        (folder / 'frames.jsonl').write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')
        for _s, i in [(s, i) for s, i in frames if s == segment]:
            (folder / f'frame_{i:08d}.png').write_bytes(b'x')


def label_db(root, blocks=(), works=(), checked=(), road=(), road_checked=()):
    from nnslr_tools.speed_review import SpeedReviewStore, db_path
    SpeedReviewStore(root)
    connection = sqlite3.connect(db_path(root))
    for route, kind, start, end, *who in road:
        connection.execute('INSERT INTO speed_road (route, kind, start_ns, end_ns, reviewer, revision) VALUES (?, ?, ?, ?, ?, ?)',
                           (route, kind, start, end, who[0] if who else 'a', 1))
    for route, kind, end in road_checked:
        connection.execute('INSERT INTO speed_road_checked (route, kind, end_ns, reviewer, revision) VALUES (?, ?, ?, ?, ?)',
                           (route, kind, end, 'a', 1))
    for route, start, end, kph in blocks:
        connection.execute('INSERT INTO speed_blocks (route, start_ns, end_ns, kph, note, reviewer, revision) '
                           'VALUES (?, ?, ?, ?, ?, ?, ?)', (route, start, end, kph, '', 'a', 1))
    for route, start, end in works:
        connection.execute('INSERT INTO speed_works (route, start_ns, end_ns, reviewer, revision) VALUES (?, ?, ?, ?, ?)',
                           (route, start, end, 'a', 1))
    for route, end in checked:
        connection.execute('INSERT INTO speed_works_checked (route, end_ns, reviewer, revision) VALUES (?, ?, ?, ?)', (route, end, 'a', 1))
    connection.commit()
    connection.close()


def encoded(seconds, missing=()):
    """The encode index of one segment, an exposure every decoded frame (20 a second), without the decoded frames missing."""
    ids = [i for i in range(seconds * 20) if i not in missing]
    return {'0': {'sof': [i * NS // 20 for i in ids], 'ids': ids}}


def test_a_block_set_on_top_of_another_clips_it_and_the_answers_follow(tmp_path):
    from nnslr_tools.speed_review import SpeedReviewStore
    targets = _targets()
    store = SpeedReviewStore(tmp_path)
    revision = store.set_block(ROUTE, 0, 100 * NS, 120, '', 0, 'a')
    store.set_block(ROUTE, 10 * NS, 20 * NS, None, '', revision, 'a')
    classes, out = targets.speed_answers(seconds(5, 15, 40), [80] * 3, targets.read_labels(tmp_path)[ROUTE]['blocks'])
    assert list(classes) == [119, -1, 119] and out[1] == targets.BLOCKED_OUT


def test_from_the_database_to_the_answers_of_a_drive(tmp_path, monkeypatch):
    targets = _targets()
    borders = []
    monkeypatch.setattr(targets, 'frame_country', lambda _latitude, _longitude, root: borders.append(root) or 'CH')
    frames = [(0, i) for i in range(40)]
    write_drive(tmp_path, ROUTE, frames)
    for second in (22, 30):
        (tmp_path / 'derived' / 'frames' / ROUTE / '0' / f'frame_{second:08d}.png').unlink()
    label_db(tmp_path, blocks=[(ROUTE, 10 * NS, 20 * NS, 100), (ROUTE, 25 * NS, 28 * NS, None)],
             works=[(ROUTE, 5 * NS, 15 * NS)], checked=[(ROUTE, 25 * NS)],
             road=[(ROUTE, 'narrow', 5 * NS, 15 * NS), (ROUTE, 'gravel', 10 * NS, None)],
             road_checked=[(ROUTE, 'narrow', 25 * NS)])
    # His speed from the third second on: vEgo + 3 km/h on a Peugeot log without its own vEgoCluster, 80 on the dashboard.
    # openpilot held the speed from 32 to 34 s: not his foot.
    merged = {'frames': encoded(40, missing={23 * 20}),
              'series': {'v_ego': [[3 * NS, 77.0]], 'long_active': [[32 * NS, True], [34 * NS, False]]},
              'meta': {'car': {'fingerprint': 'PSA_PEUGEOT_3008'}}, 'span': [0, 40 * NS],
              'gps': [[1 * NS, 47.0, 8.0, 1_750_000_000_000]]}
    split = {'cells': {}, 'partitions': {}, 'frame_keys': {}}
    labels = targets.read_labels(tmp_path)
    assert labels[ROUTE]['checked_ns'] == 25 * NS and len(labels[ROUTE]['blocks']) == 2
    assert labels[ROUTE]['road_checked'] == {'narrow': 25 * NS} and len(labels[ROUTE]['road']) == 2
    drive = targets.drive_targets(tmp_path, ROUTE, labels[ROUTE], merged, split, kinds=('narrow', 'gravel'))
    assert [frame['key'] for frame in drive['frames']][:2] == ['0/0', '0/1']
    # The kinds of road, a column each, answered as the roadworks; a frame with no picture teaches none of them.
    assert drive['road_kinds'] == ['narrow', 'gravel'] and drive['road'].shape == drive['road_weight'].shape == (40, 2)
    assert drive['road'][10].tolist() == [1, 0] and drive['road_weight'][10].tolist() == [1, 0]
    assert drive['road'][20, 0] == 0 and drive['road_weight'][20, 0] == 1 and drive['road_weight'][22, 0] == 0
    assert drive['road_weight'][30, 0] == 0 and not drive['road_weight'][:, 1].any()
    assert list(drive['speed_out'][:3]) == [targets.NO_SPEED] * 3
    assert drive['speed'][5] == 79 and drive['speed'][15] == 99 and drive['speed'][35] == 79
    assert drive['speed_out'][8] == 0 and drive['speed_out'][9] == drive['speed_out'][10] == targets.BOUNDARY
    assert drive['speed_out'][26] == targets.BLOCKED_OUT
    assert drive['speed_out'][30] == targets.NO_FRAME
    assert drive['speed_out'][32] == drive['speed_out'][33] == targets.CRUISE and drive['speed'][32] == -1
    assert drive['speed_out'][34] == 0 and drive['speed_weight'][33] == 0
    assert drive['blocked'][15] and not drive['blocked'][26] and not drive['blocked'][5]
    assert drive['works'][10] == 1 and drive['works_weight'][10] == 1
    assert drive['works'][20] == 0 and drive['works_weight'][20] == 1
    assert drive['works_weight'][30] == 0
    # A frame with no picture or no time teaches nothing.
    assert drive['speed_out'][22] == drive['speed_out'][23] == targets.NO_FRAME
    assert drive['works_weight'][22] == drive['works_weight'][23] == 0 and drive['works_weight'][21] == 1
    assert borders == [tmp_path]
    assert drive['partitions'] == ['buffer'] * 40
    # The car's speed of the measure is the dashboard's, in the labels' units; the speed input stays vEgo, as the car
    # gives it.
    assert {kph for kph in drive['own_kph'].tolist() if kph == kph} == {80.0} and drive['ns'][23] == -1
    assert drive['inputs']['speed'][10].tolist() == _np().float32(targets.BRAIN_INPUTS.speed_row(77 / 3.6)).tolist()
    # No Sign review: no next limit to learn, and nothing to trace.
    assert not drive['next_weight'].any() and drive['sign_review']['fingerprint'] is None
    # Seconds with his speed and no picture: 22 and 30 without one, 23 without its time.
    assert drive['no_picture_s'] == 3
    found = targets.summary([drive])
    assert found['drives'][0]['works_in'] == drive['works'][drive['works_weight'] > 0].sum()
    assert found['partitions']['buffer']['speed'] == found['drives'][0]['speed']
    assert found['drives'][0]['blocks'] == 2 and found['drives'][0]['out']['left_out'] == 3
    assert found['road_kinds'] == ['narrow', 'gravel'] and set(found['drives'][0]['road']) == {'narrow'}
    narrow = found['partitions']['buffer']['road']['narrow']
    assert narrow['in'] == int(drive['road'][drive['road_weight'][:, 0] > 0, 0].sum()) and narrow['out'] > 0
    assert found['drives'][0]['out']['cruise'] == 2
    text = targets.summary_text(found)
    assert 'signs NOT read' in text and '3 seconds with a speed and no picture' in text and '2 blocks' in text
    assert drive['cuts'] == set() and drive['reader'] is None
    found['drives'][0].update(signs_read=True, reader_current=False)
    assert 'signs read by ANOTHER reader' in targets.summary_text(found)


def road_type_lines(root, route, types, corrections=()):
    """The map's road type of each second of segment 0 (None: the map does not know) and a person's corrections over it:
    [(from second, to second, road type or None)]."""
    folder = root / 'derived' / 'road_types'
    folder.mkdir(parents=True, exist_ok=True)
    lines = [{'frame_key': f'0/{second}', 'segment_index': 0, 'output_index': second, 'road_type': kind}
             for second, kind in enumerate(types)]
    (folder / f'{route}.jsonl').write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')
    if corrections:
        folder = root / 'annotations' / 'road_types'
        folder.mkdir(parents=True, exist_ok=True)
        rows = [{'from': f'0/{start}', 'to': f'0/{end}', 'road_type': kind, 'reviewer': 'a', 'at': 'x'}
                for start, end, kind in corrections]
        (folder / f'{route}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')


def test_with_the_road_types_only_the_seconds_with_one_answer_and_a_person_s_type_wins(tmp_path):
    from nnslr_tools.road_types import ROAD_TYPES
    targets = _targets()
    write_drive(tmp_path, ROUTE, [(0, i) for i in range(40)])
    label_db(tmp_path, works=[(ROUTE, 5 * NS, 15 * NS)], checked=[(ROUTE, 39 * NS)])
    road_type_lines(tmp_path, ROUTE, ['town'] * 20 + [None] * 10 + ['motorway'] * 10,
                    corrections=[(25, 27, 'rural'), (35, 36, None)])
    merged = {'frames': encoded(40), 'series': {'v_ego': [[3 * NS, 77.0]]}, 'meta': {}, 'span': [0, 40 * NS]}
    split = {'cells': {}, 'partitions': {}, 'frame_keys': {}}
    labels = targets.read_labels(tmp_path)[ROUTE]
    drive = targets.drive_targets(tmp_path, ROUTE, labels, merged, split, inputs=False, types=True)
    assert drive['types'] is True
    assert drive['road_type'][5] == ROAD_TYPES.index('town') and not drive['typed_by_person'][5]
    assert drive['road_type'][26] == ROAD_TYPES.index('rural') and drive['typed_by_person'][26]
    assert drive['road_type'][32] == ROAD_TYPES.index('motorway') and drive['type_weight'][32] == 1
    # The map does not know, or a person cannot tell: the second answers nothing, the speed included.
    for second in (21, 35):
        assert drive['road_type'][second] == -1 and drive['type_weight'][second] == 0
        assert drive['speed_out'][second] == targets.NO_TYPE and drive['speed'][second] == -1
        assert drive['speed_weight'][second] == 0 and drive['works_weight'][second] == 0
    assert drive['speed_out'][1] == targets.NO_SPEED and drive['speed_out'][10] == 0 and drive['works_weight'][10] == 1
    found = targets.summary([drive])
    assert found['types'] is True and found['road_types'] == list(ROAD_TYPES)
    assert found['partitions']['buffer']['types']['town'] == 20 and found['partitions']['buffer']['types']['rural'] == 3
    assert found['drives'][0]['typed_s'] == 40 - 10 + 3 - 2 and found['drives'][0]['typed_by_person_s'] == 3
    assert found['drives'][0]['out']['no_road_type'] == 7 + 2
    assert 'road types in buffer: town 20' in targets.summary_text(found)
    # Without the road types they are read all the same, and every second keeps its speed.
    plain = targets.drive_targets(tmp_path, ROUTE, labels, merged, split, inputs=False)
    assert plain['types'] is False and plain['speed_out'][21] == 0 and plain['speed'][21] == drive['speed'][19]
    assert (plain['road_type'] == drive['road_type']).all()


def test_a_drive_the_map_was_never_asked_about_has_no_road_type(tmp_path):
    targets = _targets()
    write_drive(tmp_path, ROUTE, [(0, i) for i in range(10)])
    found, person = targets.type_answers(tmp_path, ROUTE, [{'segment': 0, 'index': i} for i in range(10)])
    assert (found == -1).all() and not person.any()


def test_reading_the_labels_changes_nothing(tmp_path):
    from nnslr_tools.speed_review import db_path
    label_db(tmp_path, works=[(ROUTE, 1, 2)])
    before = db_path(tmp_path).read_bytes()
    assert _targets().read_labels(tmp_path)[ROUTE]['works'][0]['start_ns'] == 1
    assert db_path(tmp_path).read_bytes() == before


def test_a_database_of_before_the_blocks_still_reads(tmp_path):
    from nnslr_tools.speed_review import db_path
    path = db_path(tmp_path)
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE speed_works (route TEXT, start_ns INTEGER, end_ns INTEGER, reviewer TEXT, revision INTEGER)')
    connection.execute('INSERT INTO speed_works VALUES (?, 1, 2, ?, 1)', (ROUTE, 'a'))
    connection.commit()
    connection.close()
    assert _targets().read_labels(tmp_path) == {ROUTE: {'blocks': [], 'works': [{'route': ROUTE, 'start_ns': 1, 'end_ns': 2,
                                                                                    'reviewer': 'a', 'revision': 1}],
                                                        'checked_ns': None, 'road': [], 'road_checked': {}}}
    assert _targets().read_labels(tmp_path / 'nothing') == {}


# A drive the brain learns from: openpilot passive in every carParams and never engaged (speed_review.dashcam_drive).
DASHCAM = {'meta': {'car': {'brand': 'psa', 'dashcam_only': False, 'passive': True}}, 'coverage': {'engaged': 0}}


def scanned(root, route, summary=None):
    from nnslr_tools import drive_review
    path = drive_review.db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for statement in drive_review.SCHEMA_STATEMENTS:
        connection.execute(statement)
    summary = {'t0_ns': 0, 'span_ns': [0, 40 * NS], **DASHCAM} if summary is None else summary
    connection.execute("INSERT INTO scans (route, state, summary) VALUES (?, 'done', ?)", (route, json.dumps(summary)))
    connection.commit()
    connection.close()


def marked(root, *routes, ok=True):
    """A person's switch "OK for brain training" of Speed review on these drives, through the store as the site sets it."""
    from nnslr_tools.speed_review import SpeedReviewStore
    store = SpeedReviewStore(root)
    for route in routes:
        store.set_training_ok(route, ok, 'cristian')


def frozen(root, partitions=(), frame_keys=()):
    folder = root / 'splits'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'speed-brain-split.json').write_text(json.dumps({'sha256': 'x', 'cells': {}, 'partitions': dict(partitions)}))
    (folder / 'speed-brain-frames.json').write_text(json.dumps({'split_sha256': 'x', 'frames': dict(frame_keys)}))


def test_the_drives_are_those_scanned_with_their_frames_with_a_block_or_not(tmp_path):
    targets = _targets()
    other, unfinished, bare = '00000002--abcdef0123', '00000003--abcdef0123', '00000004--abcdef0123'
    for route in (ROUTE, other, unfinished):
        write_drive(tmp_path, route, [(0, i) for i in range(3)])
    scanned(tmp_path, ROUTE)
    scanned(tmp_path, other)
    scanned(tmp_path, unfinished, {'t0_ns': 0, 'span_ns': [0, None], **DASHCAM})
    scanned(tmp_path, bare)
    # Not marked OK for training yet: none, each with its reason; a drive with no frames or no end is no drive at all.
    left_out = {}
    assert targets.training_routes(tmp_path, left_out) == []
    assert left_out == {ROUTE: targets.NOT_MARKED, other: targets.NOT_MARKED}
    assert targets.dashcam_routes(tmp_path) == [ROUTE, other]
    marked(tmp_path, ROUTE, other, unfinished, bare)
    assert targets.training_routes(tmp_path) == [ROUTE, other]
    assert targets.training_routes(tmp_path / 'nothing') == []
    # Labels of a drive with no frames: said, and left out.
    label_db(tmp_path, blocks=[(bare, 0, NS, 50)])
    frozen(tmp_path, partitions={other: ['train', 'train']}, frame_keys={other: ['0/0', '0/1']})
    said = []
    merged = {'frames': encoded(3), 'series': {'v_ego': [[0, 50.0]]}, 'span': [0, 3 * NS], 'gps': []}
    drives = targets.dataset(tmp_path, lambda route: merged, log=said.append)
    # A drive with no block is a drive; one of the frozen split with other frames is left out with a word.
    assert [drive['route'] for drive in drives] == [ROUTE]
    assert drives[0]['speed'].tolist() == [49, 49, 49] and drives[0]['blocks'] == []
    assert any(other in line and 'frames changed' in line and 'left out' in line for line in said)
    assert any(bare in line and 'left out' in line for line in said)
    assert targets.dataset(tmp_path, lambda route: None, routes=[ROUTE], log=said.append) == []
    assert said[-1] == f'{ROUTE}: no scan, left out'


def test_the_brain_learns_only_from_dashcam_drives_and_says_why_it_leaves_the_others_out(tmp_path):
    targets = _targets()
    engaged, unknown, old, older, steered = (f'0000000{i}--abcdef0123' for i in range(5, 10))
    for route in (ROUTE, engaged, unknown, old, older, steered):
        write_drive(tmp_path, route, [(0, i) for i in range(3)])
    scanned(tmp_path, ROUTE)
    # sunnypilot on: passive False, or engaged once.
    scanned(tmp_path, engaged, {'t0_ns': 0, 'span_ns': [0, 40 * NS], 'meta': {'car': {'passive': True}},
                                'coverage': {'engaged': 3.5}})
    # No carParams: not a dashcam drive.
    scanned(tmp_path, unknown, {'t0_ns': 0, 'span_ns': [0, 40 * NS], 'meta': {}})
    # An extract before carParams.passive: dashcamOnly says it.
    scanned(tmp_path, old, {'t0_ns': 0, 'span_ns': [0, 40 * NS], 'meta': {'car': {'dashcam_only': True}}})
    scanned(tmp_path, older, {'t0_ns': 0, 'span_ns': [0, 40 * NS], 'meta': {'car': {'dashcam_only': False}}})
    scanned(tmp_path, steered)
    marked(tmp_path, ROUTE, old, steered)
    left_out = {}
    assert targets.training_routes(tmp_path, left_out) == [ROUTE, old, steered]
    assert sorted(left_out) == [engaged, unknown, older] and all('sunnypilot was on' in why for why in left_out.values())
    frozen(tmp_path)
    said = []
    plain = {'frames': encoded(3), 'series': {'v_ego': [[0, 50.0]]}, 'span': [0, 3 * NS], 'gps': []}
    # MADS steered though the summary says passive: the logs win.
    madsed = {**plain, 'series': {**plain['series'], 'mads_active': [[0, False], [NS, True]]}}
    drives = targets.dataset(tmp_path, lambda route: madsed if route == steered else plain, log=said.append)
    assert [drive['route'] for drive in drives] == [ROUTE, old]
    assert f'{steered}: openpilot drove (mads_active on), left out' in said
    assert all(any(route in line and 'sunnypilot was on' in line for line in said) for route in (engaged, unknown, older))


def test_only_the_drives_marked_ok_for_training_are_learnt_each_whole(tmp_path):
    """Speed review's switch "OK for brain training" (9 Nov 2026): the brain has a memory, so a drive goes in whole once
    a person said all its signs are right, and one without the switch stays out whole, said in the log and counted."""
    from nnslr_tools.speed_review import db_path
    targets = _targets()
    unmarked, unset, on = '00000002--abcdef0123', '00000003--abcdef0123', '00000004--abcdef0123'
    for route in (ROUTE, unmarked, unset, on):
        write_drive(tmp_path, route, [(0, i) for i in range(3)])
    scanned(tmp_path, ROUTE, {'t0_ns': 0, 'span_ns': [0, 1800 * NS], **DASHCAM})
    scanned(tmp_path, unmarked)
    scanned(tmp_path, unset)
    scanned(tmp_path, on, {'t0_ns': 0, 'span_ns': [0, 40 * NS], 'meta': {'car': {'passive': False}}})
    # No file of Speed review, or one without the switch (the site's code of before 9 Nov 2026): no drive is marked.
    assert targets.marked_routes(tmp_path) == set()
    db_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path(tmp_path)).close()
    assert targets.marked_routes(tmp_path) == set() and targets.training_routes(tmp_path) == []
    db_path(tmp_path).unlink()
    # Set, and set then unset: only the switch on counts.
    marked(tmp_path, ROUTE, unset)
    marked(tmp_path, unset, ok=False)
    assert targets.marked_routes(tmp_path) == {ROUTE}
    left_out = {}
    assert targets.training_routes(tmp_path, left_out) == [ROUTE]
    assert left_out == {unmarked: targets.NOT_MARKED, unset: targets.NOT_MARKED, on: targets.SUNNYPILOT_ON}
    counts = targets.training_counts(tmp_path)
    assert counts == {'drives': 1, 'hours': 0.5, 'waiting': 2, 'sunnypilot_on': 1}
    assert targets.training_counts_text(counts) == ('Drives marked OK for training (Speed review): 1, 0.5 h; '
                                                    '2 dashcam drives not marked yet, left out whole.')
    frozen(tmp_path)
    said = []
    plain = {'frames': encoded(3), 'series': {'v_ego': [[0, 50.0]]}, 'span': [0, 3 * NS], 'gps': []}
    drives = targets.dataset(tmp_path, lambda route: plain, log=said.append)
    assert [drive['route'] for drive in drives] == [ROUTE]
    assert f'{unmarked}: {targets.NOT_MARKED}, left out' in said and f'{unset}: {targets.NOT_MARKED}, left out' in said
    # Asked by name, an unmarked drive is still out; marked False (only running the brain along it) takes it.
    assert targets.dataset(tmp_path, lambda route: plain, routes=[unmarked], log=said.append) == []
    assert said[-1] == f'{unmarked}: {targets.NOT_MARKED}, left out'
    assert [drive['route'] for drive in targets.dataset(tmp_path, lambda route: plain, log=said.append, marked=False)] == [
        ROUTE, unmarked, unset]


def test_a_frozen_drive_whose_frames_changed_is_named():
    with pytest.raises(_targets().FramesChanged, match='frames changed'):
        _targets().frame_partitions('old', [at_cell('0/0', (10, 10))], SPLIT)


def test_a_second_of_a_drive_is_its_frame_and_the_readings_after_it(tmp_path):
    from nnslr_tools.road_type_signs import picture_path
    targets = _targets()
    picture = picture_path(tmp_path, ROUTE, 0)
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b'x')
    frames = [{'route': ROUTE, 'key': f'0/{second}', 'segment': 0, 'index': second, 'decoded': second * 20,
               'seconds': float(second), 'present': True} for second in (0, 1)]
    merged = {'frames': encoded(2, missing={10})}
    found, partitions, cells = targets.brain_frames(tmp_path, frames, ['train', 'test'], [(1, 1), None], merged,
                                                    {'pictures': {'0': 6}})
    assert [frame['key'] for frame in found] == ['0/0', '0/0.25', '0/0.5', '0/0.75', '0/1', '0/1.25', '0/1.5', '0/1.75']
    assert [frame['step'] for frame in found] == [0, 1, 2, 3] * 2 and [frame['picture'] for frame in found] == list(range(8))
    assert [frame['ns'] for frame in found][:4] == [0, NS // 4, None, 3 * NS // 4]
    assert [frame['present'] for frame in found] == [True] * 6 + [False] * 2
    assert partitions == ['train'] * 4 + ['test'] * 4 and cells == [(1, 1)] * 4 + [None] * 4
    assert not any(frame['present'] for frame in targets.brain_frames(tmp_path, frames, ['train'] * 2, [None] * 2,
                                                                       merged, None)[0])
    with pytest.raises(ValueError, match='off the readings'):
        targets.brain_frames(tmp_path, [{**frames[0], 'decoded': 7}], ['train'], [None], merged, None)


RUN = '20261108T120000Z-0123abcd'


def sign_box(family='maximum_speed', kph=None, where='own_road', panel='none', sign_type=None):
    return {'family': family, 'value_kph': kph, 'applicability': where, 'supplementary_panel': panel,
            'sign_type': sign_type, 'bbox_xyxy': [0, 0, 10, 10]}


def frame_review(second, *boxes, read=False, reviewer='cristian', kind='human', run=RUN, confirmed=1, revision=1):
    """A frame of Sign review: a 1 Hz picture at a whole second, or a reading of the comma (read) at any step."""
    decoded = int(round(second * 20))
    output = 1000 + decoded if read else int(second)
    payload = {'segment_index': 0, 'output_index': output, 'decoded_frame_index': decoded, 'detections': list(boxes)}
    return (ROUTE, run, f'0/{output}', 0, output, json.dumps(payload), reviewer, confirmed, revision,
            '2026-11-08T10:00:00', kind, 'web')


def sign_review_db(root, rows, run=RUN):
    from nnslr_tools.review_db import SCHEMA_STATEMENTS, db_path
    report = root / 'derived' / 'preannotations' / ROUTE
    (report / run).mkdir(parents=True, exist_ok=True)
    (report / run / 'index.html').write_text('<html></html>', encoding='utf-8')
    (report / 'latest.json').write_text(json.dumps({'run_id': run}), encoding='utf-8')
    db_path(root).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path(root))
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.executemany('INSERT OR REPLACE INTO frame_reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rows)
    connection.commit()
    connection.close()


def distance_marks(root, lines):
    folder = root / 'annotations' / 'sign_distance'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{ROUTE}.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines), encoding='utf-8')


def zero_at(second, sign='999'):
    return {'sign': sign, 'mark': 'zero', 'segment': 0, 'frame': int(second * 20), 't_ns': int(second * NS),
            'sign_t_ns': None, 'sync': None, 'reviewer': 'a', 'at': '2026-11-08T10:00:00+01:00'}


def sign_merged(seconds_long=120, kph=36.0):
    return {'frames': encoded(seconds_long), 'series': {'v_ego': [[0, kph]]}, 'span': [0, seconds_long * NS], 'meta': {},
            'gps': []}


def reviewed_drive(root):
    """A drive gone through in Sign review from 0 to 61 s, but for 30 s, which the site's AI reviewer still holds."""
    special = {
        10: [sign_box(kph=60)], 11: [sign_box(kph=60)], 12: [sign_box(kph=60)],
        20: [sign_box(kph=50)], 21: [sign_box(kph=50), sign_box(kph=70, where='other_road')],
        40: [sign_box('cancellation')], 41: [sign_box('cancellation')],
        50: [sign_box('zone', 30)], 51: [sign_box('zone', 30)],
        55: [sign_box(kph=80, panel='time')],
        58: [sign_box('other_sign', sign_type='town_entry')],
    }
    rows = [frame_review(second, *special.get(second, [])) for second in range(61) if second != 30]
    rows += [frame_review(30, sign_box(kph=80), reviewer='ai/llama/qwen', kind='ai'),
             frame_review(61, sign_box(kph=100), reviewer='ai/Claude Code', kind='ai'),
             frame_review(12.25, sign_box(kph=60), read=True),
             frame_review(45, sign_box(kph=90), run='20261101T120000Z-0123abcd'),
             frame_review(62, sign_box(kph=120), confirmed=0)]
    sign_review_db(root, rows)
    # A zero mark of Sign distance just after the 50, and one taken back by an Undo after the end.
    distance_marks(root, [zero_at(21.8), zero_at(41.2, '777'), {'sign': '777', 'mark': 'undo', 'undoes': 1,
                                                                 'reviewer': 'a', 'at': '2026-11-08T10:01:00+01:00'}])


def test_the_signs_of_a_drive_are_the_ones_a_person_confirmed_on_its_current_run(tmp_path):
    targets = _targets()
    reviewed_drive(tmp_path)
    found = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    # The 60 passed half a second after its last box (a reading of the comma's), the 50 at its zero mark, the end of
    # the limit with no mark in effect, and the 100 of an AI assistant: the site's AI reviewer, another run, a draft and
    # another road's sign are not there.
    assert found['signs'] == [(12_750_000_000, 60), (21_800_000_000, 50), (41_500_000_000, None), (61_500_000_000, 100)]
    assert found['masked'] == [(50 * NS, 51_500_000_000), (55 * NS, 55_500_000_000), (58 * NS, 58_500_000_000)]
    # The pictures gone through: the one still waiting for a person breaks them.
    assert found['covered'] == [(0, 29 * NS), (31 * NS, 61 * NS)]
    assert isinstance(found['fingerprint'], str) and len(found['fingerprint']) == 16
    again = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert again == found
    distance_marks(tmp_path, [zero_at(22.0)])
    moved = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert moved['signs'][1] == (22 * NS, 50) and moved['fingerprint'] != found['fingerprint']


def test_two_numbers_on_one_sign_are_two_signs_and_a_mark_goes_to_one_sign(tmp_path):
    targets = _targets()
    sign_review_db(tmp_path, [frame_review(10, sign_box(kph=60)), frame_review(11, sign_box(kph=50)),
                              frame_review(12, sign_box(kph=50)), frame_review(13, sign_box(kph=None))])
    # A mark long after the sign is not its passing.
    distance_marks(tmp_path, [zero_at(19.0)])
    found = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert found['signs'] == [(10_500_000_000, 60), (12_500_000_000, 50)] and found['masked'] == []
    # A sign nobody could read the number of says a limit with no number: its approach is masked.
    sign_review_db(tmp_path, [frame_review(10, sign_box(kph=None)), frame_review(11, sign_box('unreadable')),
                              frame_review(12, sign_box(kph=None)), frame_review(13, sign_box(kph=None))])
    found = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert found['signs'] == [] and found['masked'] == [(10 * NS, 13_500_000_000), (11 * NS, 11_500_000_000)]


def test_no_sign_review_is_no_sign_and_nothing_gone_through(tmp_path):
    empty = {'signs': [], 'covered': [], 'masked': [], 'fingerprint': None, 'read_by': None}
    assert _targets().reviewed_signs(tmp_path, ROUTE, sign_merged()) == empty
    sign_review_db(tmp_path, [frame_review(10, sign_box(kph=60))])
    (tmp_path / 'derived' / 'preannotations' / ROUTE / 'latest.json').unlink()
    assert _targets().reviewed_signs(tmp_path, ROUTE, sign_merged()) == empty


def test_reading_sign_review_changes_nothing(tmp_path):
    from nnslr_tools.review_db import db_path
    reviewed_drive(tmp_path)
    before = db_path(tmp_path).read_bytes()
    _targets().reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert db_path(tmp_path).read_bytes() == before


NETWORKS = {part: {'id': f'{part}:run/x', 'size': 1, 'mtime_ns': 1} for part in ('cutter', 'classifier', 'decider')}


def net_box(family='maximum_speed', kph=None, where='own_road', panel='unknown', label=None):
    """A box of our networks as their worker writes it (_chain_proposer): its class in model_label, no sign_type."""
    return {'family': family, 'value_kph': kph, 'applicability': where, 'supplementary_panel': panel,
            'model_label': label or (family if kph is None else f'{family}:{kph}'), 'bbox_xyxy': [0, 0, 10, 10]}


def network_read(root, boxes, monkeypatch, last=119, in_use=NETWORKS):
    """The drive read by our networks (brain_signs) on its pictures from 0 to last s, with boxes {second: [box, ...]};
    in_use: the networks in use now (brain_signs.stamp_in_use)."""
    from nnslr_tools import brain_signs
    monkeypatch.setattr(brain_signs, 'stamp_in_use', lambda _root: in_use)
    path = brain_signs.cache_path(root, ROUTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {'version': brain_signs.CACHE_VERSION, 'pictures': 'p1', 'confidence': 0.1, 'networks': NETWORKS}
    rows = [{'frame_key': f'0/{second}', 'segment_index': 0, 'decoded_frame_index': second * 20, 'output_index': second,
             'detections': boxes.get(second, [])} for second in range(last + 1)]
    path.write_text(''.join(json.dumps(line) + '\n' for line in [head, *rows]), encoding='utf-8')


def test_without_sign_review_the_signs_are_what_our_networks_read_on_the_whole_drive(tmp_path, monkeypatch):
    targets = _targets()
    network_read(tmp_path, {
        10: [net_box(kph=60)], 11: [net_box(kph=60)],
        # One sign: the applicability network gave it our road on a picture.
        15: [net_box(kph=40, where='unresolved')], 16: [net_box(kph=40)],
        30: [net_box(kph=80, where='other_road')],
        50: [net_box(kph=50, where='unresolved')],
        70: [net_box(kph=30, panel='time', label='maximum_speed_hours:30')],
        # Too far to read, then a 70: the 70's.
        78: [net_box('unreadable', where='unresolved')], 80: [net_box(kph=70)],
        86: [net_box('unreadable', where='unresolved')],
        90: [net_box('other_sign', where='unresolved', label='other_sign:town_entry')],
        95: [net_box('other_sign', where='unresolved', label='other_sign:direction')],
        100: [net_box('cancellation')]}, monkeypatch)
    found = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    # Another road's sign is not ours; one whose road the applicability network could not tell, a number with hours,
    # a sign nobody could read and a town's entry say a limit nobody knows: their approach is masked.
    assert found['signs'] == [(11_500_000_000, 60), (16_500_000_000, 40), (80_500_000_000, 70), (100_500_000_000, None)]
    assert found['masked'] == [(50 * NS, 50_500_000_000), (70 * NS, 70_500_000_000), (86 * NS, 86_500_000_000),
                               (90 * NS, 90_500_000_000)]
    assert found['covered'] == [(0, 119 * NS)]
    assert found['read_by'] == {'cutter': 'cutter:run/x', 'classifier': 'classifier:run/x', 'decider': 'decider:run/x'}
    assert isinstance(found['fingerprint'], str)


def test_a_reading_by_networks_no_longer_in_use_is_not_used(tmp_path, monkeypatch):
    targets = _targets()
    network_read(tmp_path, {40: [net_box(kph=100)], 41: [net_box(kph=100)]}, monkeypatch,
                 in_use={**NETWORKS, 'classifier': {'id': 'classifier:run/y', 'size': 1, 'mtime_ns': 2}})
    assert targets.reviewed_signs(tmp_path, ROUTE, sign_merged())['read_by'] is None
    network_read(tmp_path, {40: [net_box(kph=100)], 41: [net_box(kph=100)]}, monkeypatch, in_use=None)
    assert targets.reviewed_signs(tmp_path, ROUTE, sign_merged()) == {'signs': [], 'covered': [], 'masked': [],
                                                                      'fingerprint': None, 'read_by': None}


def test_where_he_confirmed_the_frames_his_signs_replace_the_networks(tmp_path, monkeypatch):
    targets = _targets()
    # He went through 0 to 25 s and 60 to 80 s: a 60 at 10-12, a 70 of another road at 24-25 and a 50 at 78-80.
    his = {10: [sign_box(kph=60)], 11: [sign_box(kph=60)], 12: [sign_box(kph=60)],
           24: [sign_box(kph=70, where='other_road')], 25: [sign_box(kph=70, where='other_road')],
           78: [sign_box(kph=50)], 79: [sign_box(kph=50)], 80: [sign_box(kph=50)]}
    sign_review_db(tmp_path, [frame_review(second, *his.get(second, [])) for second in [*range(26), *range(60, 81)]])
    network_read(tmp_path, {
        11: [net_box(kph=50)],                  # a frame he confirmed: his
        26: [net_box(kph=70)], 27: [net_box(kph=70)],  # his 70 is another road's
        40: [net_box(kph=100)], 41: [net_box(kph=100)],  # nobody went through it: the networks'
        81: [net_box(kph=50)], 82: [net_box(kph=80)]}, monkeypatch)  # his 50 longer to 81 s, then misread
    found = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert found['signs'] == [(12_500_000_000, 60), (41_500_000_000, 100), (81_500_000_000, 50)]
    assert found['masked'] == [] and found['covered'] == [(0, 119 * NS)]
    alone = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert alone == found
    (tmp_path / 'derived' / 'brain_signs' / f'{ROUTE}.jsonl').unlink()
    mine = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    assert mine['signs'] == [(12_500_000_000, 60), (80_500_000_000, 50)] and mine['read_by'] is None
    assert mine['covered'] == [(0, 25 * NS), (60 * NS, 80 * NS)] and mine['fingerprint'] != found['fingerprint']


def test_the_networks_answer_the_next_limit_where_nobody_went_through_the_frames(tmp_path, monkeypatch):
    targets = _targets()
    network_read(tmp_path, {40: [net_box(kph=100)], 41: [net_box(kph=100)]}, monkeypatch)
    found = targets.reviewed_signs(tmp_path, ROUTE, sign_merged())
    ns = seconds(5, 35, 60)
    following, weight, metres, _metres_weight = targets.ahead_from_signs(
        ns, found, targets.metres_driven([[0, 36.0]]), _np().ones(3, dtype=bool))
    # At 36 km/h: 65 m before the 100 its number; far before it and after it, none.
    assert following.tolist() == [targets.NONE_AHEAD, targets.speed_class(100), targets.NONE_AHEAD]
    assert weight.tolist() == [1, 1, 1] and round(float(metres[1]) * targets.METRES_SCALE) == 65


def ahead(times, signs=(), covered=((0, 100 * NS),), masked=(), asked=None):
    """The next limit at these seconds, at 36 km/h: 10 m a second."""
    targets = _targets()
    ns = seconds(*times)
    reviewed = {'signs': list(signs), 'covered': list(covered), 'masked': list(masked)}
    asked = _np().ones(len(ns), dtype=bool) if asked is None else _np().array(asked)
    return targets.ahead_from_signs(ns, reviewed, targets.metres_driven([[0, 36.0]]), asked)


def test_within_a_hundred_metres_the_next_sign_and_its_metres_beyond_the_margin_none():
    targets = _targets()
    following, weight, metres, metres_weight = ahead((10, 15, 18, 19, 21, 25, 29, 31, 60, 95), signs=[(30 * NS, 60)])
    none = targets.NONE_AHEAD
    assert following.tolist() == [none, none, none, none, 59, 59, 59, none, none, none]
    assert weight.tolist() == [1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
    assert metres.tolist() == pytest.approx([0, 0, 0, 0, .9, .5, .1, 0, 0, 0])
    assert metres_weight.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]


def test_an_end_ahead_has_no_number_to_learn():
    following, weight, _metres, metres_weight = ahead((10, 18, 21), signs=[(30 * NS, None)])
    assert weight.tolist() == [1, 0, 0] and not metres_weight.any() and following[0] == _targets().NONE_AHEAD


def test_none_ahead_only_where_a_person_went_through_the_metres_ahead():
    _following, weight, _metres, _metres_weight = ahead((5, 10, 25), covered=[(0, 20 * NS)])
    assert weight.tolist() == [1, 0, 0]


def test_a_masked_sign_ahead_says_nothing_and_neither_does_a_sign_behind_it():
    _following, weight, _metres, _metres_weight = ahead((31, 38, 45, 55, 68), masked=[(40 * NS, 50 * NS)])
    assert weight.tolist() == [1, 0, 0, 1, 1]
    following, weight, _metres, metres_weight = ahead((46, 51), signs=[(55 * NS, 80)], masked=[(40 * NS, 50 * NS)])
    assert weight.tolist() == [0, 1] and metres_weight.tolist() == [0, 1] and following[1] == 79


def test_a_frame_not_asked_learns_no_next_limit():
    _following, weight, _metres, metres_weight = ahead((21, 25), signs=[(30 * NS, 60)], asked=[False, True])
    assert weight.tolist() == [0, 1] and metres_weight.tolist() == [0, 1]
    numpy, targets = _np(), _targets()
    found = targets.ahead_from_signs(numpy.array([-1, 21 * NS]), {'signs': [(30 * NS, 60)], 'covered': [(0, 40 * NS)],
                                                                 'masked': []},
                                     targets.metres_driven([[0, 36.0]]), numpy.ones(2, dtype=bool))
    assert found[1].tolist() == [0, 1]


def test_a_repeat_of_the_limit_in_force_is_no_next_and_the_change_after_it_is():
    targets = _targets()
    following, weight, metres, metres_weight = ahead((25, 32, 44, 52), signs=[(20 * NS, 50), (30 * NS, 50),
                                                                              (40 * NS, 50), (55 * NS, 80)])
    # Past the first 50 the next 50s, 50 and 80 m on, are repeats: none ahead, then the margin, then the 80 30 m on.
    assert following.tolist() == [targets.NONE_AHEAD] * 3 + [targets.speed_class(80)] and metres[-1] == pytest.approx(.3)
    assert weight.tolist() == [1, 1, 0, 1] and metres_weight.tolist() == [0, 0, 0, 1]
    # An end is always a next, after it the first sign of all is, whatever its number, and past that one its repeat is not.
    following, weight, _metres, _metres_weight = ahead((12, 22, 32), signs=[(10 * NS, 50), (15 * NS, None), (25 * NS, 50),
                                                                              (35 * NS, 50)])
    assert weight.tolist() == [0, 1, 1] and following.tolist()[1:] == [targets.speed_class(50), targets.NONE_AHEAD]


def test_a_masked_sign_passed_leaves_nothing_in_force():
    targets = _targets()
    following, weight, _metres, _metres_weight = ahead((18, 32), signs=[(10 * NS, 50), (40 * NS, 50)],
                                                       masked=[(14 * NS, 15 * NS)])
    # Before the masked sign the 50 is a repeat; past it nothing is in force and the 50 is the next.
    assert weight.tolist() == [1, 1] and following[0] == targets.NONE_AHEAD and following[1] == targets.speed_class(50)


def test_a_number_ahead_needs_a_person_through_the_way_to_it_and_back_to_the_limit_in_force():
    targets = _targets()
    # The way from the frame to the 60 crosses frames nobody confirmed: no number.
    _following, weight, _metres, metres_weight = ahead((25,), signs=[(30 * NS, 60)], covered=[(0, 27 * NS)])
    assert weight.tolist() == [0] and metres_weight.tolist() == [0]
    # A gap since the 50 in force: the 50 ahead may be a change, nothing is said of it.
    _following, weight, _metres, _metres_weight = ahead((25,), signs=[(5 * NS, 50), (30 * NS, 50)],
                                                        covered=[(0, 10 * NS), (20 * NS, 100 * NS)])
    assert weight.tolist() == [0]
    # Without the gap it is a repeat, and with no other sign within the margin the answer is none.
    following, weight, _metres, _metres_weight = ahead((25,), signs=[(5 * NS, 50), (30 * NS, 50)])
    assert weight.tolist() == [1] and following[0] == targets.NONE_AHEAD
    # Not known what is in force, and no sign at all within the margin: none is still sure.
    following, weight, _metres, _metres_weight = ahead((25,), signs=[(5 * NS, 50), (60 * NS, 50)],
                                                       covered=[(0, 10 * NS), (20 * NS, 100 * NS)])
    assert weight.tolist() == [1] and following[0] == targets.NONE_AHEAD


def test_a_drive_takes_its_next_limit_from_sign_review(tmp_path):
    targets = _targets()
    write_drive(tmp_path, ROUTE, [(0, i) for i in range(70)])
    reviewed_drive(tmp_path)
    merged = sign_merged(70)
    drive = targets.drive_targets(tmp_path, ROUTE, targets.NO_LABELS, merged, {'cells': {}, 'partitions': {},
                                                                              'frame_keys': {}})
    assert drive['next'][5] == targets.speed_class(60) and drive['metres'][5] == pytest.approx(.775)
    assert drive['next_weight'][5] == drive['metres_weight'][5] == 1
    assert drive['next'][14] == targets.speed_class(50) and drive['next_weight'][14] == 1
    # 127.5 m from the 60: in the margin. 40 m from the end of what a person went through: nothing said of what is beyond.
    assert drive['next_weight'][0] == 0 and drive['next_weight'][25] == 0
    assert drive['sign_review']['fingerprint'] is not None
    found = targets.summary([drive])['drives'][0]
    assert found['signs'] == 4 and found['covered_s'] == 59 and found['next'] == int(drive['next_weight'].sum())
    # Without vEgo the car drives no metres: no frame has a limit ahead.
    merged['series'] = {'v_ego_cluster': [[0, 36.0]]}
    assert not targets.drive_targets(tmp_path, ROUTE, targets.NO_LABELS, merged,
                                     {'cells': {}, 'partitions': {}, 'frame_keys': {}})['next_weight'].any()


GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')


def plated(hours=(14.0, 14.0, 14.0), label=50, number=30, plate='time', ended=0, partitions=None):
    """A drive with a number and a plate on every frame, at these hours, his label on all of them."""
    np, targets = _np(), _targets()
    inputs = {name: np.zeros((len(hours), size), dtype=np.float32) for name, size in targets.INPUT_SIZES.items()}
    inputs['signs'][:, 0], inputs['signs'][:, 1], inputs['signs'][:, 6] = number / targets.BRAIN_INPUTS.SIGN_KPH_SCALE, 1, ended
    inputs['plate'][:, targets.BRAIN_INPUTS.PLATES.index(plate) if plate else -1] = 1
    inputs['hour'][:] = [targets.BRAIN_INPUTS.hour_row(hour) for hour in hours]
    return {'route': 'r', 'partitions': partitions or ['train'] * len(hours), 'inputs': inputs,
            'extra': targets.pack_inputs(inputs, GROUPS), 'speed': np.full(len(hours), targets.speed_class(label)),
            'speed_weight': np.ones(len(hours), dtype=np.float32), 'works': np.ones(len(hours), dtype=np.float32),
            'works_weight': np.ones(len(hours), dtype=np.float32)}


def in_an_epoch(drive, seed=0):
    import random
    return _targets().epoch_drives([drive], GROUPS, random.Random(seed))


def clock_of(drive):
    return _targets()._clock(drive['inputs']['hour'])


@pytest.mark.parametrize('hour, moved, after', [(10.0, 3.0, 13.0), (23.0, 2.0, 1.0), (1.0, -2.0, 23.0), (7.5, 24.0, 7.5)])
def test_a_moved_clock_is_what_the_car_says_at_that_hour(hour, moved, after):
    np, targets = _np(), _targets()
    row = targets.BRAIN_INPUTS.hour_row
    assert np.allclose(targets.other_hour(row(hour)[None], moved)[0], row(after), atol=1e-5)


def test_a_moved_clock_leaves_an_unknown_hour_unknown():
    np, targets = _np(), _targets()
    assert not targets.other_hour(targets.BRAIN_INPUTS.hour_row(None)[None], 5.0).any()


def test_a_drive_takes_one_clock_for_all_its_frames_and_nothing_else_changes():
    np, targets = _np(), _targets()
    drive = plated(hours=(10.0, 10.5, 11.0), plate=None)
    real, extra = drive['inputs']['hour'].copy(), drive['extra'].copy()
    (seen,) = in_an_epoch(drive)
    moved = clock_of(seen)
    assert not np.allclose(moved, clock_of(drive))
    assert np.allclose((moved - moved[0]) % 24, [0, .5, 1], atol=1e-3)
    assert seen['speed'] is drive['speed'] and seen['speed_weight'] is drive['speed_weight']
    assert np.array_equal(seen['extra'], targets.pack_inputs(seen['inputs'], GROUPS))
    assert np.array_equal(drive['inputs']['hour'], real) and np.array_equal(drive['extra'], extra)


def test_every_epoch_has_another_clock_and_a_seed_the_same_ones():
    import random
    np, targets = _np(), _targets()
    drive = plated(plate=None)
    rng = random.Random(0)
    first, second = (targets.epoch_drives([drive], GROUPS, rng)[0] for _ in range(2))
    assert not np.allclose(clock_of(first), clock_of(second))
    assert np.array_equal(first['extra'], in_an_epoch(drive)[0]['extra'])


def test_a_drive_with_a_plate_of_hours_keeps_its_real_clock():
    drive = plated()
    assert in_an_epoch(drive)[0] is drive


def test_a_plate_of_hours_read_by_day_goes_through_the_night_with_the_number_as_the_answer():
    np, targets = _np(), _targets()
    drive = plated(hours=(14.0, 14.5, 15.0))
    speed = drive['speed'].copy()
    for seed in range(30):
        _day, night = in_an_epoch(drive, seed)
        answered = night['speed_weight'] > 0
        assert answered[0] and targets._inside(clock_of(night), targets._plate_window(night))[answered].all()
        assert not targets._inside(clock_of(night), targets._plate_window(night))[~answered].any()
        assert np.allclose((clock_of(night) - clock_of(night)[0]) % 24, [0, .5, 1], atol=1e-3)
        assert set(night['speed'][answered]) == {targets.speed_class(30)}
        assert not night['works_weight'].any()
        assert np.array_equal(night['extra'], targets.pack_inputs(night['inputs'], GROUPS))
    assert np.array_equal(drive['speed'], speed)


@pytest.mark.parametrize('drive', [
    {'label': 30}, {'label': 20}, {'number': 33}, {'number': 0}, {'number': 250}, {'hours': (23.0, 23.1)}, {'hours': (5.5,)}, {'plate': 'other'},
    {'plate': 'none'}, {'ended': 1}, {'hours': (14.0, 14.0), 'partitions': ['validation', 'test']},
    {'hours': (14.0, 14.0), 'partitions': ['buffer', 'buffer']}], ids=str)
def test_no_night_copy_of_what_does_not_say_the_number_holds_at_night(drive):
    assert not _targets().night_frames(plated(**drive))[0].any()
    assert len(in_an_epoch(plated(**drive))) == 1


def test_a_plate_of_hours_outside_train_leaves_the_clock_of_train_moving():
    np = _np()
    drive = plated(hours=(14.0, 14.0), partitions=['validation', 'train'])
    drive['inputs']['plate'][1] = 0
    drive['inputs']['plate'][1, -1] = 1
    (seen,) = in_an_epoch(drive)
    assert not np.allclose(clock_of(seen), clock_of(drive))


def test_a_label_equal_to_the_number_takes_its_whole_passage_out():
    targets = _targets()
    drive = plated(hours=(14.0,) * 4)
    drive['speed'][2] = targets.speed_class(30)
    assert not targets.night_frames(drive)[0].any()


def test_a_label_equal_to_the_number_says_nothing_of_the_next_number():
    targets = _targets()
    drive = plated(hours=(14.0,) * 4)
    drive['inputs']['signs'][2:, 0] = 40 / targets.BRAIN_INPUTS.SIGN_KPH_SCALE
    drive['speed'][3] = targets.speed_class(40)
    assert list(targets.night_frames(drive)[0]) == [True, True, False, False]


def test_a_label_equal_to_the_number_says_nothing_of_another_plate_with_that_number():
    targets = _targets()
    drive = plated(hours=(14.0,) * 5)
    drive['inputs']['plate'][2] = 0
    drive['inputs']['plate'][2, -1] = 1
    drive['speed'][4] = targets.speed_class(30)
    assert list(targets.night_frames(drive)[0]) == [True, True, False, False, False]


def test_a_label_out_of_the_loss_says_nothing_of_its_passage():
    targets = _targets()
    drive = plated(hours=(14.0,) * 4)
    drive['speed'][2], drive['speed_weight'][2] = targets.speed_class(30), 0
    assert list(targets.night_frames(drive)[0]) == [True, True, False, True]


def test_a_label_equal_to_the_number_at_night_says_nothing_of_the_day():
    targets = _targets()
    drive = plated(hours=(5.0, 7.0, 7.5))
    drive['speed'][0] = targets.speed_class(30)
    assert list(targets.night_frames(drive)[0]) == [False, True, True]


def test_no_night_copy_at_an_unknown_hour():
    drive = plated()
    drive['inputs']['hour'][:] = 0
    assert len(in_an_epoch(drive)) == 1


def test_no_night_copy_of_a_frame_with_no_speed_of_his():
    drive = plated()
    drive['speed_weight'][:] = 0
    assert len(in_an_epoch(drive)) == 1


def test_a_night_copy_answers_only_its_frames():
    np, targets = _np(), _targets()
    drive = plated(hours=(14.0,) * 5, partitions=['train', 'train', 'validation', 'buffer', 'train'])
    drive['speed_weight'][1] = 0
    drive['road'], drive['road_weight'] = np.ones((5, 2), dtype=np.float32), np.ones((5, 2), dtype=np.float32)
    day, night = in_an_epoch(drive)
    assert list(night['speed_weight']) == [1, 0, 0, 0, 1]
    assert not night['road_weight'].any() and day['road_weight'].all()
    assert night['partitions'] == drive['partitions']
    assert night['speed'][1] == drive['speed'][1]


def test_a_frame_that_falls_after_the_night_has_no_answer():
    for seed in range(30):
        _day, night = in_an_epoch(plated(hours=(10.0, 19.0)), seed)
        assert list(night['speed_weight']) == [1, 0]


@pytest.mark.parametrize('label, copied', [(32, False), (28, False), (35, False), (36, True)])
def test_a_label_within_a_few_kilometres_of_the_number_is_the_number(label, copied):
    """His speed is the dashboard's to the kilometre: 32 past a 30 said the 30 holds by day too."""
    assert _targets().night_frames(plated(label=label))[0].all() == copied


def test_a_label_two_off_the_number_takes_its_whole_passage_out():
    targets = _targets()
    drive = plated(hours=(14.0,) * 4)
    drive['speed'][2] = targets.speed_class(32)
    assert not targets.night_frames(drive)[0].any()


@pytest.mark.parametrize('number', [32, 4, 205])
def test_a_number_no_sign_has_has_no_night_copy(number):
    assert not _targets().night_frames(plated(number=number, label=number + 20))[0].any()


def numbered(kphs, labels, *, singles=None, ended=None, partitions=None, plate=None):
    """A drive with this number in force on each frame (0: none) and these answers."""
    np, targets = _np(), _targets()
    drive = plated(hours=[14.0] * len(kphs), plate=plate, partitions=partitions)
    scale, signs = targets.BRAIN_INPUTS.SIGN_KPH_SCALE, drive['inputs']['signs']
    signs[:, 0], signs[:, 1] = np.array(kphs) / scale, np.array(kphs) > 0
    signs[:, 6] = ended or 0
    signs[:, 8] = np.array(singles or [0] * len(kphs)) / scale
    drive['speed'] = np.array([targets.speed_class(label) for label in labels])
    drive['extra'] = targets.pack_inputs(drive['inputs'], GROUPS)
    return drive


def numbers(drive):
    return _targets().sign_kph(drive['inputs']['signs']).tolist()


def answers(drive):
    return _targets().class_kph(drive['speed']).tolist()


def passages(drive, partition='train'):
    return [passage.tolist() for passage in _targets().swap_passages(drive, partition)]


def all_swapped(drive, partition='train'):
    return _targets().swapped(drive, _targets().swap_passages(drive, partition), GROUPS)


def test_a_passage_that_says_its_number_changes_with_its_answers_and_the_rest_stays():
    drive = numbered([100, 100, 0, 0], [100, 100, 90, 90])
    seen = all_swapped(drive)
    assert passages(drive) == [[0, 1]]
    assert numbers(seen) == [120, 120, 0, 0] and answers(seen) == [120, 120, 90, 90]
    assert _np().array_equal(seen['extra'], _targets().pack_inputs(seen['inputs'], GROUPS))
    assert numbers(drive) == [100, 100, 0, 0] and answers(drive) == [100, 100, 90, 90]


@pytest.mark.parametrize('number, other', [(30, 40), (40, 30), (50, 60), (60, 50), (70, 80), (80, 70), (100, 120), (120, 100)])
def test_the_numbers_of_a_family_change_into_each_other(number, other):
    seen = all_swapped(numbered([number] * 2, [number] * 2))
    assert numbers(seen) == [other] * 2 and answers(seen) == [other] * 2


@pytest.mark.parametrize('number', [20, 90, 110, 130])
def test_a_number_with_no_family_stays(number):
    assert passages(numbered([number] * 2, [number] * 2)) == []


def test_two_numbers_are_two_passages():
    seen = all_swapped(numbered([100, 100, 60, 60], [100, 100, 60, 60]))
    assert numbers(seen) == [120, 120, 50, 50] and answers(seen) == [120, 120, 50, 50]


def test_a_passage_with_one_answer_off_the_number_stays_as_a_mistake_of_the_sign():
    assert passages(numbered([100, 100, 100], [100, 100, 120])) == []


def test_a_passage_with_no_answer_stays():
    drive = numbered([100, 100], [100, 100])
    drive['speed_weight'][:] = 0
    assert passages(drive) == []


def test_only_the_answers_of_the_partition_count():
    drive = numbered([100, 100, 100], [100, 100, 90], partitions=['train', 'train', 'test'])
    assert passages(drive) == [[0, 1, 2]] and passages(drive, 'test') == []
    assert answers(all_swapped(drive)) == [120, 120, 90]
    assert passages(numbered([100, 100], [100, 100], partitions=['test', 'test'])) == []


def test_an_end_read_keeps_the_answer_after_it_and_asks_nothing_of_the_passage():
    seen = all_swapped(numbered([100, 100, 100], [100, 100, 130], ended=[0, 0, 1]))
    assert numbers(seen) == [120, 120, 120] and answers(seen) == [120, 120, 130]
    assert answers(all_swapped(numbered([100, 100, 100], [100, 100, 100], ended=[0, 0, 1]))) == [120, 120, 100]


def test_a_single_reading_changes_only_where_it_says_the_number():
    np, targets = _np(), _targets()
    seen = all_swapped(numbered([100, 100, 100, 0], [100, 100, 100, 90], singles=[100, 80, 0, 100]))
    assert np.rint(seen['inputs']['signs'][:, 8] * targets.BRAIN_INPUTS.SIGN_KPH_SCALE).tolist() == [120, 80, 0, 100]


def test_an_epoch_swaps_half_of_the_passages_with_a_new_clock_and_leaves_the_drive_as_it_is():
    drive = numbered([100, 100], [100, 100])
    seen = [in_an_epoch(drive, seed)[0] for seed in range(200)]
    swaps = [item for item in seen if numbers(item) == [120, 120]]
    assert 60 < len(swaps) < 140 and all(answers(item) == [120, 120] for item in swaps)
    assert all(answers(item) == [100, 100] for item in seen if numbers(item) == [100, 100])
    assert all(not _np().allclose(clock_of(item), 14.0) for item in swaps[:10])
    assert numbers(drive) == [100, 100]


def test_the_night_copy_takes_no_swap_and_the_day_of_its_drive_does():
    drive = numbered([30, 30, 60, 60], [50, 50, 60, 60], plate='time')
    seen = [in_an_epoch(drive, seed) for seed in range(40)]
    assert all(len(epoch) == 2 and numbers(epoch[1]) == [30, 30, 60, 60] for epoch in seen)
    assert {tuple(numbers(epoch[0])) for epoch in seen} == {(30, 30, 60, 60), (30, 30, 50, 50)}


def test_answers_a_few_kilometres_off_the_number_move_with_it_by_as_much():
    seen = all_swapped(numbered([100, 100, 100, 0], [98, 102, 100, 90]))
    assert numbers(seen) == [120, 120, 120, 0] and answers(seen) == [118, 122, 120, 90]
    assert answers(all_swapped(numbered([30, 30], [27, 32]))) == [37, 42]


@pytest.mark.parametrize('label, swaps', [(105, True), (95, True), (106, False), (94, False)])
def test_an_answer_off_the_number_by_more_than_the_tolerance_keeps_the_passage(label, swaps):
    assert passages(numbered([100, 100, 100], [100, label, 100])) == ([[0, 1, 2]] if swaps else [])


def test_a_passage_with_a_block_of_speed_review_stays():
    np = _np()
    drive = numbered([100, 100, 60, 60], [100, 100, 60, 60])
    drive['blocked'] = np.array([False, True, False, False])
    assert passages(drive) == [[2, 3]]
    assert answers(all_swapped(drive)) == [100, 100, 50, 50]


def test_the_next_limit_of_the_number_moves_with_it_and_another_stays():
    np, targets = _np(), _targets()
    drive = numbered([100, 100, 100, 0], [100, 100, 100, 90])
    drive['next'] = np.array([targets.speed_class(100), targets.speed_class(80), targets.NONE_AHEAD,
                              targets.speed_class(100)])
    seen = all_swapped(drive)
    assert seen['next'].tolist() == [targets.speed_class(120), targets.speed_class(80), targets.NONE_AHEAD,
                                     targets.speed_class(100)]
    assert drive['next'][0] == targets.speed_class(100)


def test_a_next_limit_of_the_other_number_is_a_repeat_once_swapped_and_says_nothing():
    np, targets = _np(), _targets()
    drive = numbered([100, 100, 0], [100, 100, 90])
    drive['next'] = np.array([targets.speed_class(120), targets.speed_class(80), targets.speed_class(120)])
    drive['next_weight'], drive['metres_weight'] = np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    seen = all_swapped(drive)
    assert seen['next_weight'].tolist() == [0, 1, 1] and seen['metres_weight'].tolist() == [0, 1, 1]
    assert drive['next_weight'].tolist() == [1, 1, 1]


def test_a_number_with_hours_gives_its_number_and_its_plate():
    numpy = _np()
    targets = _targets()
    read = [[10.0, [['maximum_speed_hours:30', .9]]], [10.2, [['maximum_speed_hours:30', .9]]]]
    found, plates, _hours, _wet = targets.sign_inputs(numpy.array([9.0, 11.0]), read)
    time = targets.BRAIN_INPUTS.PLATES.index('time')
    assert found[1][0] == pytest.approx(30 / 130) and plates[1][time] == 1 and plates[0][time] == 0
    plain = targets.sign_inputs(numpy.array([9.0, 11.0]), [[second, [['maximum_speed:30', .9]]] for second, _ in read])
    assert (plain[0] == found).all() and plain[1][1][time] == 0


def test_a_number_with_its_hours_gives_them_as_the_car_does():
    numpy, targets = _np(), _targets()
    read = [[10.0, [['maximum_speed_hours:30:22-6', .9]]], [10.2, [['maximum_speed_hours:30:22-6', .9]]]]
    found, plates, hours, _wet = targets.sign_inputs(numpy.array([9.0, 11.0]), read)
    row = targets.BRAIN_INPUTS.hour_row
    assert found[1][0] == pytest.approx(30 / 130) and plates[1][targets.BRAIN_INPUTS.PLATES.index('time')] == 1
    assert not hours[0].any() and hours[1][4] == 1
    assert numpy.allclose(hours[1][:2], row(22)[:2]) and numpy.allclose(hours[1][2:4], row(6)[:2])


def with_hours(drive, start, end):
    """The drive with these hours read on the plate of every frame."""
    row = _targets().BRAIN_INPUTS.hour_row
    drive['inputs']['plate_hours'][:] = [*row(start)[:2], *row(end)[:2], 1]
    return drive


@pytest.mark.parametrize('start, end, clock, copied', [(22, 6, 14.0, True), (7, 17, 19.0, True), (7, 17, 10.0, False),
                                                       (22, 6, 23.0, False), (13, 15, 14.0, False), (15, 13, 14.0, True)])
def test_a_copy_needs_a_frame_outside_the_hours_read(start, end, clock, copied):
    assert _targets().night_frames(with_hours(plated(hours=(clock, clock)), start, end))[0].all() == copied


@pytest.mark.parametrize('seed', range(8))
def test_the_copy_of_a_plate_goes_to_the_hours_read_with_the_number_as_the_answer(seed):
    np, targets = _np(), _targets()
    _day, copy = in_an_epoch(with_hours(plated(hours=(19.0, 19.0)), 7, 17), seed)
    assert ((clock_of(copy) >= 7) & (clock_of(copy) < 17)).all()
    assert (copy['speed'] == targets.speed_class(30)).all() and (copy['speed_weight'] == 1).all()


def test_a_frame_in_the_hours_read_says_nothing_of_the_frames_outside_them():
    np, targets = _np(), _targets()
    drive = with_hours(plated(hours=(16.0, 16.5, 19.0, 19.5)), 7, 17)
    drive['speed'][:2] = targets.speed_class(30)
    assert list(targets.night_frames(drive)[0]) == [False, False, True, True]


def test_a_brain_learns_the_hours_only_when_a_frame_of_train_has_them():
    targets = _targets()
    assert not targets.hours_in_train([plated()])
    assert targets.hours_in_train([with_hours(plated(), 22, 6)])
    assert not targets.hours_in_train([with_hours(plated(partitions=['validation'] * 3), 22, 6)])


def test_a_plate_counts_for_a_brain_only_on_frames_of_train():
    numpy = _np()
    targets = _targets()
    plate = numpy.zeros((2, targets.INPUT_SIZES['plate']), dtype=numpy.float32)
    plate[1, targets.BRAIN_INPUTS.PLATES.index('time')] = 1

    def drive(plates, partitions, answered=(1, 1)):
        weights = numpy.array(answered, dtype=numpy.float32)
        return {'inputs': {'plate': plates}, 'partitions': partitions, 'speed_weight': weights,
                'works_weight': numpy.zeros_like(weights)}

    plain = drive(numpy.zeros_like(plate), ['train', 'train'])
    assert targets.plates_in_train([drive(plate, ['validation', 'train']), plain])
    assert not targets.plates_in_train([drive(plate, ['train', 'validation']), plain])
    assert not targets.plates_in_train([plain])
    # A plate on a frame nobody gave a speed: its weights get no answer, though the frame is of train.
    assert not targets.plates_in_train([drive(plate, ['train', 'train'], answered=(1, 0))])


WET_GROUPS = (*GROUPS, 'plate_wet')


def wet_plated(label=100, number=80, hours=(14.0, 14.0, 14.0), **given):
    """A drive past a plate for wet roads."""
    drive = plated(hours=hours, label=label, number=number, plate=None, **given)
    drive['inputs']['plate_wet'][:] = 1
    return drive


def test_a_number_for_wet_roads_gives_its_number_and_its_plate():
    numpy, targets = _np(), _targets()
    read = [[10.0, [['maximum_speed_wet:80', .9]]], [10.2, [['maximum_speed_wet:80', .9]]]]
    found, plates, _hours, wet = targets.sign_inputs(numpy.array([9.0, 11.0]), read)
    assert found[1][0] == pytest.approx(80 / 130) and list(wet[:, 0]) == [0, 1]
    assert list(plates[1]) == [0, 0, 0, 1]


@pytest.mark.parametrize('seed', range(8))
def test_a_wet_plate_has_no_copy_since_the_picture_of_one_is_of_a_dry_road(seed):
    import random
    targets = _targets()
    seen = targets.epoch_drives([wet_plated()], WET_GROUPS, random.Random(seed))
    assert len(seen) == 1 and (seen[0]['speed'] == targets.speed_class(100)).all()


def test_a_brain_learns_the_wet_plate_only_when_a_frame_of_train_has_it():
    targets = _targets()
    assert targets.wet_in_train([wet_plated()])
    assert not targets.wet_in_train([plated()])
    assert not targets.wet_in_train([wet_plated(partitions=['validation'] * 3)])


def test_the_server_gives_a_number_with_hours_the_speed_of_the_car():
    from nnslr_tools import comma_trial
    for name in ('maximum_speed_hours:30', 'maximum_speed:30', 'cancellation', 'other_sign:give_way', 'not_a_sign'):
        assert comma_trial.speed_of(name) == _targets().VISION.class_speed(name), name


def by_labeler(start, end, run='net/run'):
    return {**mark(start, end), 'reviewer': run}


def test_a_mark_of_the_labeler_is_an_answer_like_a_person_s():
    targets = _targets()
    ns = seconds(5, 20, 30, 60, 80, 90, 120)
    marks = [by_labeler(10, 40), by_labeler(70, 110), {**mark(50, 65), 'reviewer': 'osm'}]
    target, weight = targets.works_answers(ns, marks, 100 * NS)
    # Nobody confirms them: inside every mark a yes, before the check a no, past it nothing.
    assert list(target) == [0, 1, 1, 1, 1, 1, 0]
    assert list(weight) == [1, 1, 1, 1, 1, 1, 0]
    target, weight = targets.works_answers(ns, marks, None)
    assert list(target) == [0, 1, 1, 1, 1, 1, 0] and list(weight) == [0, 1, 1, 1, 1, 1, 0]


def test_the_brain_takes_the_labeler_s_road_types_over_the_map_and_a_person_s_over_both(tmp_path):
    from nnslr_tools.road_types import ROAD_TYPES, write_labeler_types
    targets = _targets()
    write_drive(tmp_path, ROUTE, [(0, i) for i in range(20)])
    label_db(tmp_path, road=[(ROUTE, 'narrow', 2 * NS, 8 * NS, 'net/run')])
    road_type_lines(tmp_path, ROUTE, ['town'] * 20, corrections=[(15, 17, 'rural')])
    write_labeler_types(tmp_path, ROUTE, 'run', {f'0/{i}': ('motorway', 0.95) for i in range(10, 17)}, ['motorway'])
    merged = {'frames': encoded(20), 'series': {'v_ego': [[3 * NS, 77.0]]}, 'meta': {}, 'span': [0, 20 * NS]}
    split = {'cells': {}, 'partitions': {}, 'frame_keys': {}}
    labels = targets.read_labels(tmp_path)[ROUTE]
    drive = targets.drive_targets(tmp_path, ROUTE, labels, merged, split, inputs=False, types=True, kinds=('narrow',))
    assert drive['road_type'][5] == ROAD_TYPES.index('town') and not drive['typed_by_person'][5]
    assert drive['road_type'][12] == ROAD_TYPES.index('motorway') and not drive['typed_by_person'][12]
    assert drive['type_weight'][12] == 1
    assert drive['road_type'][16] == ROAD_TYPES.index('rural') and drive['typed_by_person'][16]
    # The labeler's mark of a kind of road is an answer with no "checked to here" of a person.
    column = drive['road_kinds'].index('narrow')
    assert drive['road'][5, column] == 1 and drive['road_weight'][5, column] == 1
