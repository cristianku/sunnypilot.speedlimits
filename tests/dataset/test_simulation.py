"""Models › Drive (nnslr_tools.simulation): a result as the page reads it, a certified drive and the regression test
against it, the sets of networks, and the order a replay gives the car's loop its messages and frames in. The car's
loop itself is tested in the sunnypilot clone (speed_limits/tests/test_pipeline.py, international/test_daemon.py)."""
import json
from types import SimpleNamespace

import pytest

from nnslr_tools import simulation

ROUTE = '00000184--0d22c290e8'
T0 = 100_000_000_000


def record(limits, *, set_key='in-use', classes=None, seen=None, brain=None, frames=None, signs=None, made_at='2026-11-10T10:00:00+00:00'):
    """A result of len(limits) frames read 0.2 s apart from T0 + 1 s: the speed to hold (km/h) of each."""
    count = len(limits)
    out = {field: [unset] * count for field, unset in simulation.FIELDS.items()}
    out['speedLimit'] = [kph / 3.6 for kph in limits]
    out['rawSignClass'] = classes or [''] * count
    out['rawSignSeen'] = seen or [False] * count
    out['rawSignValue'] = [float(name.rsplit(':', 1)[1]) if name.startswith('maximum_speed:') else 0.0 for name in out['rawSignClass']]
    return {'version': simulation.VERSION, 'route': ROUTE, 'set': set_key, 'name': 'speed-limits test', 'made_at': made_at,
            'device': 'cpu', 'frame': [1344, 760], 'networks': [{'role': 'finder', 'file': 'signnet.onnx', 'id': 'cutter:a/signnet', 'sha256': 'f' * 64}],
            'code': {'pipeline.py': 'c' * 64}, 'settings': {}, 'videos': [],
            'frames': {'t_ns': [T0 + 1_000_000_000 + 200_000_000 * row for row in range(count)],
                       'segment': [0] * count, 'frame': frames or [4 * row for row in range(count)]},
            'out': out, 'signs': signs or {},
            'in': {**{name: [None] * count for name in simulation.INPUTS}, 'brain': brain or {}}}


def test_a_column_is_a_step_series_with_a_hole_where_the_frames_stop():
    assert simulation._steps([1.0, 1.2, 1.4, 1.6], [50, 50, 60, 60]) == [[1.0, 50], [1.4, 60]]
    # More than HOLE_SECONDS without a frame: the series ends, and starts again with the next frame.
    assert simulation._steps([1.0, 1.2, 9.0], [50, 50, 50]) == [[1.0, 50], [4.2, None], [9.0, 50]]


def test_the_page_reads_the_squares_as_the_timelines_series():
    made = record([0, 50, 50], classes=['', 'maximum_speed:50', 'maximum_speed:50'], seen=[False, True, False],
                  signs={'1': [[600.0, 200.0, 640.0, 240.0, 0.9, 'maximum_speed:50', 0.97, 'own_road']]})
    made['out']['signRead'] = [False, True, False]
    view = simulation.view(made, T0)
    assert view['series']['vis_limit'] == [[1.0, 0.0], [1.2, 50.0]]
    assert view['series']['vis_raw_class'] == [[1.0, ''], [1.2, 'maximum_speed:50']]
    assert view['series']['vis_raw_seen'] == [[1.0, False], [1.2, True], [1.4, False]]
    assert view['events'] == [[1.2, 'sign', 50.0]]
    # On frame 4 of minute 0: the page draws the boxes on that frame of the video.
    assert view['signs'] == [[1.2, [[600.0, 200.0, 640.0, 240.0, 'maximum_speed:50', 0.97, 'own_road']], 0, 4]]
    assert (view['set'], view['label'], view['frames'], view['frame']) == ('in-use', 'In use', 3, [1344, 760])


def test_only_a_set_without_a_brain_has_a_limit_ahead():
    # The rules (vision.LimitAhead) say the next limit and the metres to it; a brain says the speed to hold alone.
    made = record([50, 50, 50])
    made['out']['speedLimitAhead'] = [0.0, 30 / 3.6, 30 / 3.6]
    made['out']['speedLimitAheadDistance'] = [0.0, 80.0, 60.0]
    series = simulation.view(made, T0)['series']
    assert series['vis_ahead'] == [[1.0, 0.0], [1.2, 30.0]]
    assert series['vis_ahead_dist'] == [[1.0, 0.0], [1.2, 80.0], [1.4, 60.0]]
    # The same columns in the result of a set with a brain (one replayed before 12 Nov 2026, when its brain still said
    # them): not a series of the page.
    made['networks'] = made['networks'] + [{'role': 'brain', 'file': 'brain.onnx', 'id': 'brain:x', 'sha256': 'b' * 64}]
    series = simulation.view(made, T0)['series']
    assert 'vis_ahead' not in series and 'vis_ahead_dist' not in series
    assert series['vis_limit'] == [[1.0, 50.0]]


def test_a_result_is_written_whole_with_what_the_lists_show_beside_it(tmp_path):
    made = record([50, 50])
    path = simulation.write_result(tmp_path, made)
    assert path == tmp_path / 'derived' / 'simulation' / ROUTE / 'in-use.json.gz'
    assert simulation.read(tmp_path, ROUTE, 'in-use') == made
    meta = simulation.result_meta(tmp_path, ROUTE, 'in-use')
    assert (meta['frames'], meta['name'], meta['networks']) == (2, 'speed-limits test', made['networks'])
    assert simulation.read(tmp_path, ROUTE, 'car') is None and simulation.result_meta(tmp_path, ROUTE, 'car') is None
    # A result without its meta is read once for it.
    path.with_name(path.name + simulation.META_SUFFIX).unlink()
    assert simulation.result_meta(tmp_path, ROUTE, 'in-use')['frames'] == 2


def test_the_regression_test_says_where_a_result_differs_from_the_certified_one():
    was = record([50, 50, 50, 80, 80], classes=['', 'maximum_speed:50', '', '', ''], seen=[False, True, False, False, False],
                 brain={'0': [0.1, 0.2], '3': [0.5, 0.5]})
    now = record([50, 60, 60, 80, 80], classes=['', 'maximum_speed:60', '', '', ''], seen=[False, True, False, False, False],
                 brain={'0': [0.1, 0.2], '3': [0.5, 0.5]})
    found = simulation.compare(was, now)
    times = was['frames']['t_ns']
    assert found['limit'] == [[times[1], times[2], 50, 60]] and found['limit_frames'] == 2
    assert found['reading'] == [[times[1], times[1], '50', '60']] and found['reading_frames'] == 1
    assert (found['frames'], found['only_was'], found['only_now']) == (5, 0, 0)
    # The brain was given the same row at both frames: it is the brain that changed.
    assert (found['same_input'], found['other_input']) == (2, 0)
    page = simulation.differences(found, T0)
    assert page['limit'] == [[1.2, 1.4, 50, 60]] and page['limit_seconds'] == 0.4


def test_another_input_of_the_brain_points_before_the_brain():
    was = record([50, 50], brain={'0': [0.1, 0.2]})
    now = record([50, 30], brain={'0': [0.1, 0.9]})
    found = simulation.compare(was, now)
    assert (found['same_input'], found['other_input']) == (0, 1)


def brain_record(limits, rows, *, brain_sha='b' * 64, gap=5.0):
    made = record(limits, brain=rows)
    made['networks'] = made['networks'] + [{'role': 'brain', 'file': 'brain.onnx', 'id': 'brain:x', 'sha256': brain_sha}]
    made['code'] = {**made['code'], 'brain.py': 'd' * 64}
    made['brain_memory_gap'] = gap
    return made


def test_with_the_same_brain_every_other_speed_points_before_it():
    was = brain_record([50, 50], {'0': [0.1, 0.2], '1': [0.1, 0.2]})
    now = brain_record([50, 30], {'0': [0.1, 0.2], '1': [0.1, 0.2]})
    assert (simulation.compare(was, now)['same_input'], simulation.compare(was, now)['other_input']) == (0, 1)


def test_the_brain_carries_another_input_in_its_memory_until_it_starts_again():
    # Step 1 had another row: the brain's memory differs from there, so frames 2 and 3 with the same rows are still
    # other_input; nothing resets it in 0.6 s (memory_gap_seconds 5).
    rows_was = {'0': [0.1], '1': [0.2], '2': [0.3], '3': [0.4]}
    rows_now = {'0': [0.1], '1': [0.9], '2': [0.3], '3': [0.4]}
    was = brain_record([50, 50, 50, 50], rows_was)
    now = brain_record([50, 50, 60, 60], rows_now, brain_sha='e' * 64)
    found = simulation.compare(was, now)
    assert (found['same_input'], found['other_input']) == (0, 2)
    # A pause longer than its memory_gap_seconds starts it again in both: from there the same rows mean the brain changed.
    for made in (was, now):
        made['frames']['t_ns'][2:] = [t + 6_000_000_000 for t in made['frames']['t_ns'][2:]]
    found = simulation.compare(was, now)
    assert (found['same_input'], found['other_input']) == (2, 0)
    # Started again in one of the two only: the other still carries the other row.
    now['frames']['t_ns'][2:] = [t - 6_000_000_000 for t in now['frames']['t_ns'][2:]]
    found = simulation.compare(was, now)
    assert (found['same_input'], found['other_input']) == (0, 2)


def test_a_step_in_one_result_alone_counts_even_across_a_pause():
    # The certified result read frame 8 and stepped there, after the pause: its memory started on a row the replay never
    # gave, so the same rows at frame 12 do not make it the brain alone.
    was = brain_record([50, 50, 50], {'0': [0.1], '1': [0.7], '2': [0.3]}, gap=0.3)
    was['frames']['frame'], was['frames']['t_ns'][1:] = [0, 8, 12], [t + 1_000_000_000 for t in was['frames']['t_ns'][1:]]
    now = brain_record([50, 60], {'0': [0.1], '1': [0.3]}, brain_sha='e' * 64, gap=0.3)
    now['frames']['frame'], now['frames']['t_ns'][1:] = [0, 12], [t + 1_200_000_000 for t in now['frames']['t_ns'][1:]]
    found = simulation.compare(was, now)
    assert (found['frames'], found['same_input'], found['other_input']) == (2, 0, 1)


def test_brains_with_other_groups_are_compared_on_the_groups_both_take():
    was = brain_record([50, 50], {'0': [0.1, 0.2, 0.3], '1': [0.1, 0.2, 0.3]})
    was['brain_groups'] = [['signs', 1], ['speed', 2]]
    now = brain_record([50, 30], {'0': [0.1], '1': [0.1]}, brain_sha='e' * 64)
    now['brain_groups'] = [['signs', 1]]
    found = simulation.compare(was, now)
    assert (found['same_input'], found['other_input']) == (1, 0)
    now['in']['brain']['1'] = [0.4]
    found = simulation.compare(was, now)
    assert (found['same_input'], found['other_input']) == (0, 1)


def test_another_picture_points_before_the_brain():
    # The same rows, but the video of the minute is another file (or the code that makes the picture changed).
    was = brain_record([50, 50], {'0': [0.1], '1': [0.1]})
    now = brain_record([50, 30], {'0': [0.1], '1': [0.1]}, brain_sha='e' * 64)
    was['sources'] = {'video': [[0, 'a' * 64]], 'logs': []}
    now['sources'] = {'video': [[0, 'a' * 64]], 'logs': []}
    assert simulation.compare(was, now)['same_input'] == 1
    now['sources']['video'] = [[0, 'c' * 64]]
    assert (simulation.compare(was, now)['same_input'], simulation.compare(was, now)['other_input']) == (0, 1)
    now['sources']['video'] = [[0, 'a' * 64]]
    now['code'] = {**now['code'], 'vision.py': 'v' * 64}
    assert (simulation.compare(was, now)['same_input'], simulation.compare(was, now)['other_input']) == (0, 1)


def test_a_run_of_differences_ends_at_a_hole_in_the_frames():
    times = [0, 200_000_000, 9_000_000_000]
    assert simulation._runs([0, 1, 2], times, [('50', '60')] * 3) == [[0, 200_000_000, '50', '60'],
                                                                      [9_000_000_000, 9_000_000_000, '50', '60']]


def test_the_pulse_is_compared_not_the_confidence_or_the_metres():
    was = record([50, 50], classes=['maximum_speed:50'] * 2, seen=[True, True])
    now = record([50, 50], classes=['maximum_speed:50'] * 2, seen=[True, True])
    was['out']['signRead'], now['out']['signRead'] = [True, False], [False, True]
    now['out']['rawSignConfidence'], now['out']['signDistance'] = [0.81, 0.83], [12.0, 11.0]
    found = simulation.compare(was, now)
    times = was['frames']['t_ns']
    assert found['reading'] == [[times[0], times[0], '50 read', '50'], [times[1], times[1], '50', '50 read']]
    now['out']['signRead'] = [True, False]
    assert simulation.compare(was, now)['reading_frames'] == 0


def test_another_video_or_log_of_a_minute_is_never_identical(tmp_path):
    was = record([50, 50])
    was['sources'] = {'video': [[0, 'a' * 64]], 'logs': [[0, 'rlog', 'b' * 64]]}
    now = {**record([50, 50]), 'sources': {'video': [[0, 'c' * 64]], 'logs': [[0, 'rlog', 'b' * 64]]}}
    now['in']['v_ego'] = [10.0, None]
    found = simulation.compare(was, now)
    assert found['sources'] == ['video of minute 0'] and found['inputs_frames'] == 1 and found['limit_frames'] == 0
    simulation.write_result(tmp_path, was)
    entry = simulation.certify(tmp_path, ROUTE, 'in-use', 'cristianku')
    simulation.write_result(tmp_path, {**now, 'made_at': '2026-11-11T08:00:00+00:00'})
    counts = simulation.regression_counts(tmp_path, ROUTE, 'in-use', entry, simulation.result_meta(tmp_path, ROUTE, 'in-use'))
    assert counts['sources'] == ['video of minute 0'] and not counts['identical']
    # A result made before the sources were kept says nothing of them.
    assert simulation.compare(record([50, 50]), was)['sources'] == []


def test_frames_are_compared_by_their_place_in_the_video():
    was = record([50, 50, 50], frames=[0, 4, 8])
    now = record([50, 50, 50], frames=[0, 5, 8])  # the middle frame read is another one
    found = simulation.compare(was, now)
    assert (found['frames'], found['only_was'], found['only_now'], found['limit_frames']) == (2, 1, 1, 0)


def test_the_reading_square_in_a_word():
    out = record([0] * 5, classes=['', 'cancellation', 'other_sign:motorway_start', 'maximum_speed:80', 'maximum_speed:80'],
                 seen=[True, True, True, True, False])['out']
    out['rawSignRoad'] = ['', '', '', 'other_road', '']
    assert [simulation.reading_text(out, row) for row in range(5)] == ['?', 'END', 'motorway_start', '80 other road', '']
    out['signRead'][3] = True
    assert simulation.reading_text(out, 3) == '80 other road read'


def test_certify_keeps_the_replay_looked_at_and_refuses_one_replayed_since(tmp_path):
    made = record([50, 50, 80])
    simulation.write_result(tmp_path, made)
    with pytest.raises(simulation.SimulationError) as refused:
        simulation.certify(tmp_path, ROUTE, 'in-use', 'cristianku', made_at='2026-11-09T10:00:00+00:00')
    assert (refused.value.status, refused.value.reason) == (409, 'changed') and simulation.certified(tmp_path) == {}
    entry = simulation.certify(tmp_path, ROUTE, 'in-use', 'cristianku', made_at=made['made_at'])
    assert entry['made_at'] == made['made_at'] and simulation.reference(tmp_path, ROUTE) == made


def test_a_replay_of_an_older_format_is_not_there_and_cannot_be_certified(tmp_path):
    simulation.write_result(tmp_path, {**record([50, 50]), 'version': simulation.VERSION - 1})
    assert simulation.read(tmp_path, ROUTE, 'in-use') is None and simulation.result_meta(tmp_path, ROUTE, 'in-use') is None
    with pytest.raises(simulation.SimulationError) as refused:
        simulation.certify(tmp_path, ROUTE, 'in-use', 'cristianku')
    assert refused.value.reason == 'no_simulation'


def test_a_certified_drive_keeps_its_result_whole_and_is_the_reference(tmp_path):
    with pytest.raises(simulation.SimulationError) as refused:
        simulation.certify(tmp_path, ROUTE, 'in-use', 'cristianku')
    assert (refused.value.status, refused.value.reason) == (409, 'no_simulation')
    made = record([50, 50, 80])
    simulation.write_result(tmp_path, made)
    entry = simulation.certify(tmp_path, ROUTE, 'in-use', 'cristianku')
    assert (entry['who'], entry['set'], entry['frames'], entry['name']) == ('cristianku', 'in-use', 3, 'speed-limits test')
    assert simulation.certified(tmp_path) == {ROUTE: entry}
    assert simulation.reference(tmp_path, ROUTE) == made
    # Another replay of the drive does not touch what was certified.
    simulation.write_result(tmp_path, record([50, 60, 80], made_at='2026-11-11T08:00:00+00:00'))
    assert simulation.reference(tmp_path, ROUTE) == made
    states = {key: {'networks': made['networks'], 'ready': True} for key in simulation.SETS}
    summary = simulation.summary(tmp_path, ROUTE, states, made['code'])
    assert summary['certified'] == entry and summary['sets']['in-use']['current']
    assert summary['regression']['in-use']['limit_frames'] == 1 and not summary['regression']['in-use']['identical']
    kept = tmp_path / 'derived' / 'simulation' / ROUTE / ('in-use.json.gz' + simulation.REGRESSION_SUFFIX)
    assert json.loads(kept.read_text())['counts'] == summary['regression']['in-use']
    simulation.uncertify(tmp_path, ROUTE)
    assert simulation.certified(tmp_path) == {} and simulation.reference(tmp_path, ROUTE) is None
    assert simulation.summary(tmp_path, ROUTE, states, made['code'])['regression'] == {}


def release(directory, version='2026-11-10', checkpoint='cutter:a/signnet', settings=None):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 2, 'version': version, 'finder': {'file': 'signnet.onnx', 'contract': 'signnet.onnx.contract.json'},
                'classifier': {'file': 'classifier.onnx', 'contract': 'classifier.onnx.contract.json'},
                **({'settings': settings} if settings else {})}
    (directory / 'speed_limits.json').write_text(json.dumps(manifest))
    for name, model in (('signnet', checkpoint), ('classifier', 'classifier:b/classifier')):
        (directory / f'{name}.onnx').write_bytes(name.encode())
        (directory / f'{name}.onnx.contract.json').write_text(json.dumps({'checkpoint': model, 'onnx_sha256': name[0] * 64}))


def test_the_sets_say_their_networks_and_whether_they_are_ready(tmp_path):
    package = tmp_path / 'clone' / 'openpilot' / 'sunnypilot' / 'speed_limits'
    release(package / 'models')
    root = tmp_path / 'data'
    states = simulation.set_states(root, {'finder': 'cutter:a/signnet', 'classifier': 'classifier:b/classifier', 'brain': None},
                                   package=package)
    assert states['car']['ready'] and states['car']['name'] == 'speed-limits 2026-11-10'
    assert [(one['role'], one['id']) for one in states['car']['networks']] == [('finder', 'cutter:a/signnet'),
                                                                               ('classifier', 'classifier:b/classifier')]
    assert not states['in-use']['ready'] and 'Not exported yet' in states['in-use']['reason']
    # Exported from the versions in use with the car's settings of today: ready; a version in use changes: to export
    # again; the car's settings change: to export again too.
    from nnslr_tools import car_settings
    release(root / simulation.SETS_RELATIVE / 'in-use', version='in-use', settings=car_settings.values(root))
    (root / simulation.SETS_RELATIVE / 'in-use' / simulation.SOURCE_FILE).write_text(json.dumps(
        {'finder': 'cutter:a/signnet', 'classifier': 'classifier:b/classifier'}))
    wanted = {'finder': 'cutter:a/signnet', 'classifier': 'classifier:b/classifier', 'decider': None}
    # An export from before what an export writes changed (no stamp, or an older one): to export again.
    old = simulation.set_states(root, wanted, package=package)['in-use']
    assert not old['ready'] and 'export changed' in old['reason']
    stamp = root / simulation.SETS_RELATIVE / 'in-use' / simulation.STAMP_FILE
    stamp.write_text(json.dumps({'stamp': simulation.EXPORT_STAMP - 1}))
    assert not simulation.set_states(root, wanted, package=package)['in-use']['ready']
    stamp.write_text(json.dumps({'stamp': simulation.EXPORT_STAMP}))
    ready = simulation.set_states(root, wanted, package=package)['in-use']
    assert ready['ready'] and ready['settings'] == car_settings.values(root)
    changed = simulation.set_states(root, {**wanted, 'finder': 'cutter:c/signnet'}, package=package)['in-use']
    assert not changed['ready'] and 'versions in use changed' in changed['reason']
    release(root / simulation.SETS_RELATIVE / 'in-use', version='in-use', settings={**car_settings.values(root),
                                                                                     'min_confidence': 0.5})
    other = simulation.set_states(root, wanted, package=package)['in-use']
    assert not other['ready'] and "settings changed" in other['reason']
    with pytest.raises(simulation.SimulationError):
        simulation.set_directory(root, 'other')


def test_a_result_is_current_with_the_same_networks_code_and_settings():
    made = record([50])
    assert simulation.is_current(made, made['networks'], made['code'])
    assert simulation.is_current(made, made['networks'], made['code'], {})
    assert not simulation.is_current(made, [{**made['networks'][0], 'sha256': 'e' * 64}], made['code'])
    assert not simulation.is_current(made, made['networks'], {'pipeline.py': 'd' * 64})
    assert not simulation.is_current(made, None, made['code'])
    assert not simulation.is_current(made, made['networks'], made['code'], {'min_confidence': 0.5})



def test_an_out_of_date_result_says_what_changed():
    made = record([50])
    first = made['networks'][0]
    ready = {'ready': True, 'networks': made['networks'], 'settings': {}, 'type_head': None}
    assert simulation.changes(made, ready, made['code']) == []
    newer = {**first, 'id': first['id'] + '-new', 'sha256': 'e' * 64}
    assert simulation.changes(made, {**ready, 'networks': [newer, *made['networks'][1:]]}, made['code']) == \
        [{'what': first['role'], 'was': first['id'], 'now': first['id'] + '-new'}]
    assert [one['what'] for one in simulation.changes(made, {**ready, 'settings': {'min_confidence': 0.5}},
                                                      {'pipeline.py': 'd' * 64})] == ['code', 'settings']
    # not exported yet: the versions in use it waits for
    waiting = {'ready': False, 'networks': made['networks'], 'wanted': {first['role']: 'other'}}
    assert simulation.changes(made, waiting, made['code'])[0] == {'what': first['role'], 'was': first['id'], 'now': 'other'}

def test_a_replay_without_the_road_type_of_its_brain_is_not_current_once_the_head_is_there():
    made = record([50])
    head = 'a' * 64
    assert not simulation.is_current(made, made['networks'], made['code'], {}, head)
    assert simulation.is_current({**made, 'brain_type_head': head}, made['networks'], made['code'], {}, head)
    assert not simulation.is_current({**made, 'brain_type_head': 'b' * 64}, made['networks'], made['code'], {}, head)


def test_the_page_holds_the_road_type_of_the_brain_from_one_of_its_steps_to_the_next():
    made = record([50, 50, 50, 50])
    assert 'brain_type' not in simulation.view(made, T0)['series']
    made['brain_type'] = {'0': ['town', 0.912], '1': ['town', 0.904], '3': ['motorway', 0.6]}
    series = simulation.view(made, T0)['series']
    assert series['brain_type'] == [[1.0, 'town'], [1.6, 'motorway']]
    assert series['brain_type_p'] == [[1.0, 0.91], [1.2, 0.9], [1.6, 0.6]]


def brain_release(directory, sha):
    release(directory)
    manifest = json.loads((directory / 'speed_limits.json').read_text())
    manifest['brain'] = {'file': 'brain.onnx', 'contract': 'brain.onnx.contract.json'}
    (directory / 'speed_limits.json').write_text(json.dumps(manifest))
    (directory / 'brain.onnx').write_bytes(b'brain')
    (directory / 'brain.onnx.contract.json').write_text(json.dumps(
        {'checkpoint': 'brain:2026-11-10_2051-brain-s2/brain', 'checkpoint_sha256': sha, 'onnx_sha256': 'b' * 64}))


def write_head(root, sha, weight, names):
    import numpy as np
    path = simulation.type_head_path(root, {'checkpoint_sha256': sha})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        np.savez(handle, weight=np.array(weight, dtype=np.float32), bias=np.zeros(len(names), dtype=np.float32),
                 names=np.array(names))


def test_a_set_says_the_road_type_head_of_its_brain_once_it_is_written(tmp_path):
    package = tmp_path / 'clone' / 'openpilot' / 'sunnypilot' / 'speed_limits'
    sha = 'c' * 64
    brain_release(package / 'models', sha)
    root = tmp_path / 'data'
    assert simulation.set_states(root, package=package)['car']['type_head'] is None
    contract = simulation.brain_contract(package / 'models')
    assert contract['checkpoint_sha256'] == sha and simulation.load_type_head(root, contract) is None
    write_head(root, sha, [[1.0, 0.0], [0.0, 1.0]], ['town', 'motorway'])
    assert simulation.set_states(root, package=package)['car']['type_head'] == sha
    head = simulation.load_type_head(root, contract)
    assert head['names'] == ['town', 'motorway'] and head['weight'].shape == (2, 2) and head['sha256'] == sha
    # A set without a brain, a contract without its checkpoint's hash: no head.
    release(tmp_path / 'other')
    assert simulation.brain_contract(tmp_path / 'other') is None
    assert simulation.type_head_path(root, {'checkpoint_sha256': '../x'}) is None


def test_the_road_type_head_is_written_only_from_the_weights_the_contract_names(tmp_path):
    package = tmp_path / 'clone' / 'openpilot' / 'sunnypilot' / 'speed_limits'
    root = tmp_path / 'data'
    said = []
    release(package / 'models')
    assert simulation.save_type_head(root, 'car', package, log=said.append) is None and 'no brain' in said[-1]
    brain_release(package / 'models', 'c' * 64)
    assert simulation.save_type_head(root, 'car', package, log=said.append) is None and 'no weights' in said[-1]
    weights = root / 'runs' / '2026-11-10_2051-brain-s2' / 'brain-last.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'trained again since')
    assert simulation.save_type_head(root, 'car', package, log=said.append) is None and 'changed since' in said[-1]
    write_head(root, 'c' * 64, [[1.0, 0.0]], ['town'])
    assert simulation.save_type_head(root, 'car', package, log=said.append) is not None and 'ready' in said[-1]


def test_a_meta_without_the_settings_is_made_again_from_its_result(tmp_path):
    made = record([50, 50])
    made['settings'] = {'min_confidence': 0.7}
    path = simulation.write_result(tmp_path, made)
    meta_path = path.with_name(path.name + simulation.META_SUFFIX)
    older = json.loads(meta_path.read_text())
    del older['settings']
    meta_path.write_text(json.dumps(older))
    assert simulation.result_meta(tmp_path, ROUTE, 'in-use')['settings'] == {'min_confidence': 0.7}
    assert json.loads(meta_path.read_text())['settings'] == {'min_confidence': 0.7}


def test_one_drive_that_breaks_does_not_stop_the_others(tmp_path, monkeypatch):
    other = '00000185--0d22c290e8'
    monkeypatch.setattr(simulation, 'set_states', lambda root, package=None: {
        'in-use': {'networks': [], 'settings': {}, 'ready': True}})
    monkeypatch.setattr(simulation, 'code_hashes', lambda package=None: {})

    def replay(root, route, set_key, **kwargs):
        if route == ROUTE:
            raise OSError('Input/output error')
        return {**record([50, 50]), 'route': route}

    monkeypatch.setattr(simulation, 'replay', replay)
    said = []
    assert simulation.run(tmp_path, [ROUTE, other], 'in-use', log=said.append) == 1
    assert f'drive 1/2 {ROUTE}: FAILED, OSError: Input/output error' in said
    assert simulation.read(tmp_path, other, 'in-use') is not None


class Loop:
    """A pipeline.SpeedVision that reads a frame every PERIOD and notes what it was given, in order."""
    PERIOD = 0.2

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.next_frame = 0.0
        self.signs = []
        self.lead = None
        self.gps_clock = None
        self.gps_country = None

    def car_state(self, now, log_time, v_ego, *rest):
        self.calls.append(('car', round(now, 3), round(log_time, 3), v_ego))

    def enabled(self, now, enabled):
        self.calls.append(('enabled', round(now, 3), enabled))

    def device_motion(self, now, z, valid):
        self.calls.append(('motion', round(now, 3), z))

    def radar(self, present, metres, v_lead):
        self.calls.append(('radar', metres))
        self.lead = (metres, v_lead * 3.6) if present else None

    def due(self, now):
        return now >= self.next_frame

    def gps(self, fix, unix_millis):
        self.calls.append(('gps', round(fix.time, 3), unix_millis))
        self.gps_country = 'CH'

    def frame(self, now, y, uv, full=None):
        self.next_frame = now + self.PERIOD
        self.calls.append(('frame', round(now, 3), int(y[0, 0])))
        self.signs = [{'box': [1.0, 2.0, 3.0, 4.0], 'score': 0.5, 'class': 'maximum_speed:50', 'confidence': 0.9,
                       'road': 'own_road'}] if int(y[0, 0]) == 4 else []
        return {'speedLimit': 50 / 3.6, 'signRead': bool(self.signs), 'modelName': 'fake', 'rawSignSeen': bool(self.signs)}


def test_a_replay_gives_the_loop_what_speedvisiond_gets_in_its_order(tmp_path, monkeypatch):
    import numpy as np

    from nnslr_tools import road_type_signs
    width, height = 16, 16
    loops = []

    def speed_vision(*args, **kwargs):
        loops.append(Loop())
        return loops[-1]

    fix = lambda time, latitude, longitude, accuracy, valid: SimpleNamespace(time=time, valid=valid)  # noqa: E731
    car = SimpleNamespace(
        pipeline=SimpleNamespace(release=lambda directory: SimpleNamespace(
            name='speed-limits fake', settings={}, finder='f', classifier='c', brain=None, road_type=None, decider=None),
            SpeedVision=speed_vision, DEFAULT_COUNTRY='CH'),
        vision=SimpleNamespace(apply_settings=lambda settings: {}, FOCAL_LENGTHS={},
                               nv12_planes=lambda data, w, h, stride, offset: (data[:w * h].reshape(h, w), data[w * h:].reshape(h // 2, w))),
        applicability=SimpleNamespace(), brain=SimpleNamespace(), country=SimpleNamespace(GpsFix=fix))
    monkeypatch.setattr(simulation, 'car_modules', lambda package=None: car)
    monkeypatch.setattr(simulation, 'release_networks', lambda directory: [{'role': 'finder', 'file': 'f', 'id': 'x', 'sha256': 'a'}])
    monkeypatch.setattr(simulation, 'code_hashes', lambda package=None: {'pipeline.py': 'c'})
    monkeypatch.setattr(simulation, 'OneNetwork', lambda path, device: None)
    monkeypatch.setattr(road_type_signs, '_videos', lambda root, route: {0: tmp_path / 'fcamera.hevc'})
    monkeypatch.setattr(road_type_signs, '_video_size', lambda video: (width, height))
    monkeypatch.setattr(road_type_signs, '_video_stamp', lambda videos: [[0, 1, 2]])
    (tmp_path / 'fcamera.hevc').write_bytes(b'video')
    second = 1_000_000_000
    # Frames 0..9 came 50 ms apart from 10.000 s; frame 2 has no time in the log.
    times = [10 * second + 50_000_000 * index for index in range(10)]
    times[2] = None
    monkeypatch.setattr(simulation, 'read_logs', lambda root, route, workers=None: {
        'car': [(10 * second - 5_000_000, 3.0, 0.0, False, False, 0.0, False, False), (10 * second + 120_000_000, 4.0, 0.0, False, False, 0.0, False, False)],
        'enabled': [(10 * second - 1_000_000, True)],
        'motion': [(10 * second + 60_000_000, 0.25, True)],
        'radar': [(10 * second - 9_000_000, True, 30.0, 10.0), (10 * second - 2_000_000, True, 20.0, 10.0)],
        'gps': [(10 * second - 8_000_000, 47.0, 8.0, 5.0, True, 111), (10 * second - 3_000_000, 47.0, 8.0, 5.0, True, 222),
                (10 * second + 180_000_000, 47.0, 8.0, 5.0, True, 333)],
        'frames': {0: times}, 'files': [[0, 'rlog', 'b' * 64], [1, 'qlog', 'c' * 64]]})

    def decode(video, w, h, *, reduce, device):
        for index in range(10):
            yield bytes([index]) * (w * h * 3 // 2)

    monkeypatch.setattr(simulation, '_decode', decode)
    made = simulation.replay(tmp_path, ROUTE, 'car', device='cpu', log=lambda text: None)
    (loop,) = loops
    assert loop.calls == [
        ('car', 10.0, 9.995, 3.0), ('enabled', 10.0, True), ('radar', 20.0),  # before the first frame: the last radar only
        ('gps', 9.997, 222),                                               # and, as it is read, the last GPS message
        ('frame', 10.0, 0),
        # Frame 1 at 10.05 s is not due and nothing came before it; frame 2 has no time; frame 3 at 10.15 s is not due
        # either, but the messages that came before it are the loop's: every carState first, then the deviceMotion.
        ('car', 10.15, 10.12, 4.0), ('motion', 10.15, 0.25),
        ('gps', 10.18, 333), ('frame', 10.2, 4),                          # 0.2 s after the last one read
        ('frame', 10.4, 8)]
    assert made['frames'] == {'t_ns': [times[0], times[4], times[8]], 'segment': [0, 0, 0], 'frame': [0, 4, 8]}
    assert made['out']['speedLimit'] == [round(50 / 3.6, 4)] * 3 and made['out']['signRead'] == [False, True, False]
    assert made['out']['rawSignClass'] == ['', '', ''] and made['out']['speedLimitAhead'] == [0.0] * 3  # unset: as the car leaves them
    assert made['signs'] == {'1': [[1.0, 2.0, 3.0, 4.0, 0.5, 'maximum_speed:50', 0.9, 'own_road']]}
    assert made['in']['v_ego'] == [3.0, 4.0, 4.0] and made['in']['enabled'] == [True] * 3
    assert made['in']['lead_m'] == [20.0] * 3 and made['in']['yaw_device'] == [None, 0.25, 0.25]
    assert (made['set'], made['name'], made['frame'], made['device']) == ('car', 'speed-limits fake', [width, height], 'cpu')
    assert np.isclose(made['in']['lead_kph'][0], 36.0)
    # What was read, to say later that a replay read the same drive: the minute's video and log by SHA-256.
    import hashlib
    assert made['sources'] == {'video': [[0, hashlib.sha256(b'video').hexdigest()]], 'logs': [[0, 'rlog', 'b' * 64]]}
    assert made['brain_memory_gap'] is None


def test_a_replay_names_the_road_type_on_the_memory_of_the_cars_brain_at_each_of_its_steps(tmp_path, monkeypatch):
    import numpy as np

    from nnslr_tools import road_type_signs
    width, height = 16, 16
    sha = 'c' * 64
    network = SimpleNamespace(given=None)
    reader = SimpleNamespace(hidden=np.zeros((1, 2), dtype=np.float32), groups=[])
    # The car's brain steps at frames 0 and 8 only; its memory then says town, then motorway.
    memories = {0: [2.0, 0.0], 8: [0.0, 3.0]}

    class BrainLoop(Loop):
        def frame(self, now, y, uv, full=None):
            sent = super().frame(now, y, uv, full)
            index = int(y[0, 0])
            if index in memories:
                network.given = {'extra': np.array([[float(index)]])}
                reader.hidden = np.array([memories[index]], dtype=np.float32)
            return sent

    car = SimpleNamespace(
        pipeline=SimpleNamespace(release=lambda directory: SimpleNamespace(
            name='speed-limits fake', settings={}, finder='f', classifier='c', brain='b', road_type=None, decider=None,
            brain_contract={'checkpoint_sha256': sha, 'step': {'memory_gap_seconds': 5.0}}),
            SpeedVision=lambda *args, **kwargs: BrainLoop(), DEFAULT_COUNTRY='CH'),
        vision=SimpleNamespace(apply_settings=lambda settings: {}, FOCAL_LENGTHS={},
                               nv12_planes=lambda data, w, h, stride, offset: (data[:w * h].reshape(h, w), data[w * h:].reshape(h // 2, w))),
        applicability=SimpleNamespace(), country=SimpleNamespace(GpsFix=None),
        brain=SimpleNamespace(BrainReader=lambda *args: reader, brain_inputs=SimpleNamespace(INPUT_SIZES={})))
    monkeypatch.setattr(simulation, 'car_modules', lambda package=None: car)
    monkeypatch.setattr(simulation, 'release_networks', lambda directory: [{'role': 'brain', 'file': 'b', 'id': 'x', 'sha256': 'a'}])
    monkeypatch.setattr(simulation, 'code_hashes', lambda package=None: {'brain.py': 'c'})
    monkeypatch.setattr(simulation, 'OneNetwork', lambda path, device: None)
    monkeypatch.setattr(simulation, 'StepNetwork', lambda path, device: network)
    monkeypatch.setattr(road_type_signs, '_videos', lambda root, route: {0: tmp_path / 'fcamera.hevc'})
    monkeypatch.setattr(road_type_signs, '_video_size', lambda video: (width, height))
    monkeypatch.setattr(road_type_signs, '_video_stamp', lambda videos: [[0, 1, 2]])
    (tmp_path / 'fcamera.hevc').write_bytes(b'video')
    second = 1_000_000_000
    monkeypatch.setattr(simulation, 'read_logs', lambda root, route, workers=None: {
        'car': [], 'enabled': [], 'motion': [], 'radar': [], 'gps': [],
        'frames': {0: [10 * second + 50_000_000 * index for index in range(10)]}, 'files': []})
    monkeypatch.setattr(simulation, '_decode', lambda video, w, h, *, reduce, device: (
        bytes([index]) * (w * h * 3 // 2) for index in range(10)))
    without = simulation.replay(tmp_path, ROUTE, 'car', device='cpu', log=lambda text: None)
    assert 'brain_type' not in without and 'brain_type_head' not in without
    write_head(tmp_path, sha, [[1.0, 0.0], [0.0, 1.0]], ['town', 'motorway'])
    made = simulation.replay(tmp_path, ROUTE, 'car', device='cpu', log=lambda text: None)
    # Frames 0, 4 and 8 are read; the brain stepped at the first and the last.
    assert made['in']['brain'] == {'0': [0.0], '2': [8.0]} and made['brain_type_head'] == sha
    assert made['brain_type'] == {'0': ['town', round(float(np.exp(2) / (np.exp(2) + 1)), 3)],
                                  '2': ['motorway', round(float(np.exp(3) / (np.exp(3) + 1)), 3)]}
