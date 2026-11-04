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
    names = ['train'] * 5 + ['test'] * 2 + ['buffer'] + ['train'] * 3
    assert pieces(names, 4) == [(0, 4, True), (4, 5, True), (5, 8, False), (8, 11, True)]
    assert pieces(names, 4, 2) == [(0, 2, True), (2, 5, True), (5, 8, False), (8, 10, True), (10, 11, True)]
    assert pieces(['test'] * 3, 2) == [(0, 2, False), (2, 3, False)]
    assert pieces([], 4) == []


def test_a_sign_read_counts_from_its_second_on_and_never_before():
    numpy = _np()
    readings = [[10.0, [['maximum_speed:100', .9]]], [20.0, [['cancellation', .9]]], [30.0, [['variable_display:80', .9]]],
                [31.0, [['other_sign:give_way', .9]]]]
    found = _targets().sign_inputs(numpy.array([9.0, 10.0, 15.0, 20.0, 30.0, 40.0]), readings)
    assert found[0].sum() == 0
    assert found[1][:3] == pytest.approx([100 / 130, 1, 0])
    assert found[1][5] == 1 and found[2][5] == 0
    assert found[2][4] == 0 and found[3][4] == 1
    assert found[4][:3] == pytest.approx([80 / 130, 1, 1]) and found[4][4] == 0


def test_a_drive_not_read_yet_has_its_signs_unknown():
    numpy = _np()
    assert _targets().sign_inputs(numpy.array([1.0, 2.0]), None).sum() == 0


def inputs_of(ns, series, readings=None):
    frames = [{'segment': 0, 'index': int(t // NS), 'country': 'CH'} for t in ns]
    return _targets().drive_inputs(frames, ns, {'series': series}, readings)


def test_no_input_of_a_frame_knows_what_comes_after_it():
    numpy = _np()
    ns = seconds(10, 20, 30)
    past = {'gas': [[1 * NS, False], [9 * NS, True], [10 * NS, False]], 'brake': [[1 * NS, False]],
            'v_ego': [[1 * NS, 50.0]], 'lead_d': [[1 * NS, 40.0]], 'lead_v': [[1 * NS, 50.0]]}
    future = {name: [*points, [21 * NS, points[-1][1] if name == 'brake' else 99.0]] for name, points in past.items()}
    future['gas'] = [*past['gas'], [21 * NS, True]]
    future['brake'] = [*past['brake'], [21 * NS, True]]
    readings = [[5.0, [['maximum_speed:80', .9]]]]
    later = [*readings, [21.0, [['maximum_speed:120', .9]]]]
    before, after = inputs_of(ns, past, readings), inputs_of(ns, future, later)
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


def test_a_country_outside_the_list_is_one_of_its_own():
    targets = _targets()
    frames = [{'segment': 0, 'index': 1, 'country': 'CH'}, {'segment': 0, 'index': 2, 'country': 'SM'}, {'segment': 0, 'index': 3, 'country': None}]
    found = targets.drive_inputs(frames, seconds(1, 2, 3), {'series': {}}, None)['country']
    assert found[0][targets.COUNTRIES.index('CH')] == 1
    assert found[1][-1] == 1 and found[2][-1] == 1
    assert found.sum() == 3


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


def test_from_the_database_to_the_answers_of_a_drive(tmp_path):
    targets = _targets()
    route = '00000001--abcdef0123'
    frames = [(0, i) for i in range(40)]
    write_drive(tmp_path, route, frames)
    (tmp_path / 'derived' / 'frames' / route / '0' / 'frame_00000030.png').unlink()
    label_db(tmp_path, stretches=[(route, 0, 20 * NS, 100, 'corrected'), (route, 20 * NS, 40 * NS, 120, 'confirmed')],
             works=[(route, 5 * NS, 15 * NS)], checked=[(route, 25 * NS)])
    merged = {'frames': {'0': {'sof': [i * NS // 20 for i in range(800)], 'ids': list(range(800))}}, 'series': {}}
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
    assert drive['partitions'] == ['buffer'] * 40
    assert len(drive['own_kph']) == 40 and all(kph != kph for kph in drive['own_kph'])
    found = targets.summary([drive])
    assert found['drives'][0]['works_in'] == drive['works'][drive['works_weight'] > 0].sum()
    assert found['partitions']['buffer']['speed'] == found['drives'][0]['speed']
    assert 'signs NOT read' in targets.summary_text(found)


def test_reading_the_labels_changes_nothing(tmp_path):
    from nnslr_tools.speed_review import db_path
    label_db(tmp_path, works=[('00000001--abcdef0123', 1, 2)])
    before = db_path(tmp_path).read_bytes()
    assert _targets().read_labels(tmp_path)['00000001--abcdef0123']['works'][0]['start_ns'] == 1
    assert db_path(tmp_path).read_bytes() == before


def test_no_labels_yet_is_no_drive(tmp_path):
    assert _targets().read_labels(tmp_path) == {}
