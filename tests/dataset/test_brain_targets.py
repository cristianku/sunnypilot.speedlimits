"""What a label of Speed review becomes for the training: the answers, what stays out of the loss, the split, the inputs.
numpy comes in only inside the tests: collecting them must leave it out (tests/cli/test_import_isolation.py)."""
import json
import sqlite3

import pytest

NS = 1_000_000_000


def _np():
    import numpy
    return numpy


def _targets():
    from nnslr_tools import brain_targets
    return brain_targets


def stretch(start, end, kph, state='confirmed', known=None):
    return {'start_ns': start * NS, 'end_ns': end * NS, 'kph': kph, 'state': state,
            'known_ns': None if known is None else known * NS}


def mark(start, end):
    return {'start_ns': start * NS, 'end_ns': None if end is None else end * NS}


def seconds(*values):
    return _np().array(values, dtype=_np().int64) * NS


def test_a_label_is_a_class_of_five_kilometres():
    targets = _targets()
    assert targets.speed_class(5) == 0
    assert targets.speed_class(100) == 19
    assert targets.class_kph(targets.speed_class(120)) == 120
    assert targets.SPEED_CLASSES == 40


def test_only_a_frame_a_person_looked_at_gives_a_speed():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(5, 15, 25, 35), [stretch(0, 20, 100), stretch(20, 30, None, 'undetermined')])
    assert list(classes) == [19, 19, -1, -1]
    assert list(out) == [0, 0, targets.UNDETERMINED, targets.NOT_EXAMINED]


def test_a_speed_not_known_yet_is_not_an_answer_before_it_can_be_known():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(5, 15), [stretch(0, 20, 100, known=10)])
    assert list(classes) == [-1, 19]
    assert out[0] == targets.NOT_KNOWN_YET


def test_the_frames_around_a_change_of_speed_are_out_and_a_cut_with_the_same_speed_is_no_change():
    targets = _targets()
    classes, out = targets.speed_answers(seconds(9, 11, 19, 21, 29, 31),
                                         [stretch(0, 10, 100), stretch(10, 20, 100), stretch(20, 40, 120)])
    assert list(classes) == [19, 19, -1, -1, 23, 23]
    assert list(out)[2:4] == [targets.BOUNDARY, targets.BOUNDARY]
    # The first frame and the last one are on the edge of what was examined.
    _classes, out = targets.speed_answers(seconds(1, 3), [stretch(0, 10, 100)])
    assert list(out) == [targets.BOUNDARY, 0]


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


def test_the_end_of_works_leaves_the_speed_alone():
    targets = _targets()
    ns = seconds(20, 60)
    labelled = [stretch(0, 100, 80)]
    classes, _out = targets.speed_answers(ns, labelled)
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


def write_drive(root, route, frames):
    """frames: [(segment, index)], each with a picture and a decoded frame index of index * 20."""
    for segment in {segment for segment, _index in frames}:
        folder = root / 'derived' / 'frames' / route / str(segment)
        folder.mkdir(parents=True, exist_ok=True)
        lines = [{'segment_index': s, 'output_index': i, 'decoded_frame_index': i * 20} for s, i in frames if s == segment]
        (folder / 'frames.jsonl').write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')
        for _s, i in [(s, i) for s, i in frames if s == segment]:
            (folder / f'frame_{i:08d}.png').write_bytes(b'x')


def label_db(root, stretches=(), works=(), checked=()):
    from nnslr_tools.speed_review import SpeedReviewStore, db_path
    SpeedReviewStore(root)
    connection = sqlite3.connect(db_path(root))
    columns = {row[1] for row in connection.execute('PRAGMA table_info(speed_stretches)')}
    for route, start, end, kph, state in stretches:
        row = {'route': route, 'start_ns': start, 'end_ns': end, 'kph': kph, 'state': state, 'known_ns': None, 'note': '',
               'reviewer': 'a', 'revision': 1}
        row = {key: value for key, value in row.items() if key in columns}
        connection.execute(f"INSERT INTO speed_stretches ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    for route, start, end in works:
        connection.execute('INSERT INTO speed_works (route, start_ns, end_ns, reviewer, revision) VALUES (?, ?, ?, ?, ?)',
                           (route, start, end, 'a', 1))
    for route, end in checked:
        connection.execute('INSERT INTO speed_works_checked (route, end_ns, reviewer, revision) VALUES (?, ?, ?, ?)', (route, end, 'a', 1))
    connection.commit()
    connection.close()


def test_a_piece_clipped_off_before_the_known_time_gives_no_answer(tmp_path):
    from nnslr_tools.speed_review import SpeedReviewStore
    targets, route = _targets(), '00000001--abcdef0123'
    store = SpeedReviewStore(tmp_path)
    wrong = {'kph': 120, 'state': 'corrected', 'known_ns': 30 * NS, 'note': ''}
    store.apply(route, 0, 100 * NS, wrong, 0, 'a')
    store.apply(route, 10 * NS, 20 * NS, {**wrong, 'kph': 80, 'known_ns': None}, 1, 'a')
    _classes, out = targets.speed_answers(seconds(5, 25, 40), store.stretches(route))
    assert list(out) == [targets.NOT_KNOWN_YET, targets.NOT_KNOWN_YET, 0]


def test_from_the_database_to_the_answers_of_a_drive(tmp_path, monkeypatch):
    targets = _targets()
    borders = []
    monkeypatch.setattr(targets, 'frame_country', lambda _latitude, _longitude, root: borders.append(root) or 'CH')
    route = '00000001--abcdef0123'
    frames = [(0, i) for i in range(40)]
    write_drive(tmp_path, route, frames)
    for second in (22, 30):
        (tmp_path / 'derived' / 'frames' / route / '0' / f'frame_{second:08d}.png').unlink()
    label_db(tmp_path, stretches=[(route, 0, 20 * NS, 100, 'corrected'), (route, 20 * NS, 40 * NS, 120, 'confirmed')],
             works=[(route, 5 * NS, 15 * NS)], checked=[(route, 25 * NS)])
    encoded = [i for i in range(800) if i != 23 * 20]
    merged = {'frames': {'0': {'sof': [i * NS // 20 for i in encoded], 'ids': encoded}}, 'series': {},
              'gps': [[1 * NS, 47.0, 8.0, 1_750_000_000_000]]}
    split = {'cells': {}, 'partitions': {}, 'frame_keys': {}}
    labels = targets.read_labels(tmp_path)
    assert labels[route]['checked_ns'] == 25 * NS
    drive = targets.drive_targets(tmp_path, route, labels[route], merged, split)
    assert [frame['key'] for frame in drive['frames']][:2] == ['0/0', '0/1']
    assert drive['speed'][10] == 19 and drive['speed'][25] == 23
    assert drive['speed_out'][20] == targets.BOUNDARY
    assert drive['speed_out'][30] == targets.NO_FRAME
    assert drive['works'][10] == 1 and drive['works_weight'][10] == 1
    assert drive['works'][20] == 0 and drive['works_weight'][20] == 1
    assert drive['works_weight'][30] == 0
    # Inside what a person went through too, a frame with no picture or no time teaches nothing.
    assert drive['speed_out'][22] == drive['speed_out'][23] == targets.NO_FRAME
    assert drive['works_weight'][22] == drive['works_weight'][23] == 0 and drive['works_weight'][21] == 1
    assert borders == [tmp_path]
    assert drive['partitions'] == ['buffer'] * 40
    assert len(drive['own_kph']) == 40 and all(kph != kph for kph in drive['own_kph'])
    found = targets.summary([drive])
    assert found['drives'][0]['works_in'] == drive['works'][drive['works_weight'] > 0].sum()
    assert found['partitions']['buffer']['speed'] == found['drives'][0]['speed']
    assert 'signs NOT read' in targets.summary_text(found)
    assert drive['cuts'] == set() and drive['reader'] is None
    found['drives'][0].update(signs_read=True, reader_current=False)
    assert 'signs read by ANOTHER reader' in targets.summary_text(found)


def test_reading_the_labels_changes_nothing(tmp_path):
    from nnslr_tools.speed_review import db_path
    label_db(tmp_path, works=[('00000001--abcdef0123', 1, 2)])
    before = db_path(tmp_path).read_bytes()
    assert _targets().read_labels(tmp_path)['00000001--abcdef0123']['works'][0]['start_ns'] == 1
    assert db_path(tmp_path).read_bytes() == before


def test_no_labels_yet_is_no_drive(tmp_path):
    assert _targets().read_labels(tmp_path) == {}


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
    _day, night = in_an_epoch(drive)
    assert list(night['speed_weight']) == [1, 0, 0, 0, 1]
    assert night['partitions'] == drive['partitions']
    assert night['speed'][1] == drive['speed'][1]


def test_a_frame_that_falls_after_the_night_has_no_answer():
    for seed in range(30):
        _day, night = in_an_epoch(plated(hours=(10.0, 19.0)), seed)
        assert list(night['speed_weight']) == [1, 0]


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


WET_GROUPS = (*GROUPS, 'plate_wet', 'wipers')


def wet_plated(wipers=0, label=100, number=80, hours=(14.0, 14.0, 14.0), **given):
    """A drive past a plate for wet roads, the wipers in this state on every frame."""
    drive = plated(hours=hours, label=label, number=number, plate=None, **given)
    drive['inputs']['plate_wet'][:] = 1
    drive['inputs']['wipers'][:] = _targets().BRAIN_INPUTS.wipers_row(wipers)
    return drive


def wet_copy(drive, groups=WET_GROUPS):
    import random
    return _targets().wet_copy(drive, groups, random.Random(0))


def test_a_number_for_wet_roads_gives_its_number_and_its_plate():
    numpy, targets = _np(), _targets()
    read = [[10.0, [['maximum_speed_wet:80', .9]]], [10.2, [['maximum_speed_wet:80', .9]]]]
    found, plates, _hours, wet = targets.sign_inputs(numpy.array([9.0, 11.0]), read)
    assert found[1][0] == pytest.approx(80 / 130) and list(wet[:, 0]) == [0, 1]
    assert list(plates[1]) == [0, 0, 0, 1]


def test_the_wipers_of_a_frame_are_those_of_its_moment_and_nothing_carries_over_a_gap():
    np, targets = _np(), _targets()
    frames = [{'segment': 0, 'index': index} for index in range(5)]
    series = [[2, 0], [3, 2], [4, None]]
    found = targets.drive_inputs(frames, np.array([1, 2, 3, 4, 5]), {'series': {}, 'gps': []}, None, None, series)['wipers']
    assert list(found[:, -1]) == [0, 1, 1, 0, 0]
    assert list(found[1]) == [1, 0, 0, 0, 1] and list(found[2]) == [0, 0, 1, 0, 1]
    assert not targets.drive_inputs(frames, np.array([1, 2, 3, 4, 5]), {'series': {}, 'gps': []}, None, None)['wipers'].any()


def test_the_copy_of_a_wet_plate_has_the_wipers_on_and_the_number_as_the_answer():
    targets = _targets()
    drive = wet_plated()
    copy = wet_copy(drive)
    assert (copy['speed'] == targets.speed_class(80)).all() and (copy['speed_weight'] == 1).all()
    wipers = copy['inputs']['wipers']
    assert (wipers[:, 0] == 0).all() and (wipers[:, 1:4].sum(axis=1) == 1).all() and (wipers[:, -1] == 1).all()
    assert (copy['extra'][:, -5:] == wipers).all()
    assert (drive['speed'] == targets.speed_class(100)).all() and (drive['inputs']['wipers'][:, 0] == 1).all()


def test_on_a_wet_copy_every_other_frame_keeps_its_answer_so_the_wipers_alone_teach_no_slowing():
    targets = _targets()
    drive = wet_plated()
    drive['inputs']['plate_wet'][0] = 0
    copy = wet_copy(drive)
    assert list(copy['speed']) == [targets.speed_class(100), targets.speed_class(80), targets.speed_class(80)]
    assert (copy['speed_weight'] == 1).all() and (copy['works_weight'] == 1).all()
    assert (copy['inputs']['wipers'][0] == copy['inputs']['wipers'][1]).all()


@pytest.mark.parametrize('why, drive', [
    ('the wipers were on: his label is already the answer', lambda: wet_plated(wipers=2)),
    ('the wipers are not known', lambda: wet_plated(wipers=None)),
    ('his speed is under the number', lambda: wet_plated(label=60)),
    ('his speed is the number: it holds dry too', lambda: wet_plated(label=80)),
    ('an end was read', lambda: wet_plated(ended=1)),
    ('no frame of train', lambda: wet_plated(partitions=['validation'] * 3)),
    ('no such class of the speed', lambda: wet_plated(number=83))])
def test_no_wet_copy(why, drive):
    assert wet_copy(drive()) is None, why


def test_a_frame_with_the_number_as_his_speed_cancels_the_wet_copy_of_its_passage():
    targets = _targets()
    drive = wet_plated()
    drive['speed'][0] = targets.speed_class(80)
    assert not targets.wet_frames(drive)[0].any()


def test_a_brain_without_both_inputs_has_no_wet_copy():
    assert wet_copy(wet_plated(), ('signs', 'plate_wet')) is None
    assert wet_copy(wet_plated(), ('signs', 'wipers')) is None


@pytest.mark.parametrize('seed', range(8))
def test_an_epoch_goes_through_the_drive_and_its_wet_copy(seed):
    import random
    np, targets = _np(), _targets()
    seen = targets.epoch_drives([wet_plated()], WET_GROUPS, random.Random(seed))
    assert len(seen) == 2 and (seen[0]['speed'] != targets.speed_class(80)).all()
    assert (seen[1]['speed'] == targets.speed_class(80)).all()
    assert np.allclose(seen[0]['inputs']['hour'], seen[1]['inputs']['hour'])


def test_a_brain_learns_the_wet_plate_only_when_a_frame_of_train_has_it():
    targets = _targets()
    assert targets.wet_in_train([wet_plated()])
    assert not targets.wet_in_train([plated()])
    assert not targets.wet_in_train([wet_plated(partitions=['validation'] * 3)])


def test_the_server_gives_a_number_with_hours_the_speed_of_the_car():
    from nnslr_tools import comma_trial
    for name in ('maximum_speed_hours:30', 'maximum_speed:30', 'cancellation', 'other_sign:give_way', 'not_a_sign'):
        assert comma_trial.speed_of(name) == _targets().VISION.class_speed(name), name
