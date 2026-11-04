"""The road-type network's data and measures (nnslr_tools.road_type_data, road_type_metrics) on a tiny synthetic data
root, and its ONNX step contract (nnslr_tools.road_type_net, with torch only). numpy and the modules that need it
are imported inside the tests: collecting the tests must leave numpy out (tests/cli/test_import_isolation.py)."""
import json
from functools import partial
from pathlib import Path

import pytest

from nnslr_tools.road_types import ROAD_TYPES

TEST_ROUTE = '00000001--aaaaaaaaaa'
OTHER_ROUTES = ('00000002--bbbbbbbbbb', '00000003--cccccccccc', '00000004--dddddddddd')
TOWN, RURAL, MOTORWAY, RAMP, PARKING = (ROAD_TYPES.index(n) for n in ('town', 'rural', 'motorway', 'ramp', 'parking'))


def _write_route(root: Path, route: str, types: list[tuple], start: tuple[float, float], step: tuple[float, float],
                 *, segments: int = 2, missing: set | None = None, picture: bool = True):
    """A drive of segments x 60 frames: types is a list of (road_type, sure, basis, highway) per frame, positions on a
    line from start by step per frame."""
    from PIL import Image

    from nnslr_tools import road_type_data as data
    lines, positions = [], []
    for i, (road_type, sure, basis, highway) in enumerate(types[:segments * 60]):
        segment, index = divmod(i, 60)
        key = f'{segment}/{index}'
        lines.append({'frame_key': key, 'segment_index': segment, 'output_index': index, 'road_type': road_type,
                      'sure': sure, 'basis': basis, 'highway': highway, 'map_type': road_type, 'way_id': 1,
                      'maxspeed': None})
        positions.append({'frame_key': key, 'segment_index': segment, 'output_index': index,
                          'latitude': start[0] + step[0] * i, 'longitude': start[1] + step[1] * i,
                          'speed_mps': 20.0, 'bearing_deg': 0.0, 'country': 'CH'})
        if picture and (segment, index) not in (missing or set()):
            path = data.frame_path(root, route, segment, index)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (64, 36), (i % 255, 40, 90)).save(path)
    folder = root / 'derived' / 'road_types'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{route}.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    gps = root / 'derived' / 'gps' / route
    gps.mkdir(parents=True, exist_ok=True)
    (gps / 'positions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in positions))


def _types(*parts):
    found = []
    for road_type, count, *rest in parts:
        sure = rest[0] if rest else True
        basis = rest[1] if len(rest) > 1 else 'maxspeed 50'
        highway = rest[2] if len(rest) > 2 else None
        found += [(road_type, sure, basis, highway)] * count
    return found


def _confirm_all(root: Path, route: str):
    """A person confirms every stretch the map proposes, as Data › Roads saves it."""
    from nnslr_tools import road_types
    for stretch in road_types.stretches(road_types.map_road_types(root, route)):
        if stretch['type'] is not None:
            road_types.correct(root, route, stretch['from'], stretch['to'], stretch['type'], 'someone')


@pytest.fixture
def proposed(tmp_path):
    """Four drives with the map's proposals, no person's decision yet."""
    root = tmp_path / 'data'
    # The test drive: town, a ramp, the motorway; in an area of its own.
    _write_route(root, TEST_ROUTE, _types(('town', 40), ('ramp', 20, False, 'highway motorway_link', 'motorway_link'),
                                          ('motorway', 60, True, 'highway motorway')), (47.0, 8.0), (0.0003, 0.0))
    # Far drives for train and validation, each through its own areas.
    _write_route(root, OTHER_ROUTES[0], _types(('rural', 60, False, 'maxspeed 60'), ('town', 60)), (47.5, 8.5),
                 (0.0003, 0.0))
    _write_route(root, OTHER_ROUTES[1], _types(('motorway', 50), ('ramp', 20, False, 'highway motorway_link',
                                                                  'motorway_link'), ('parking', 50)),
                 (48.0, 9.0), (0.0003, 0.0))
    _write_route(root, OTHER_ROUTES[2], _types((None, 10, False, 'no gps'), ('town', 110)), (48.5, 9.5),
                 (0.0003, 0.0), missing={(1, 30)})
    split = {'test_routes': [TEST_ROUTE]}
    (root / 'splits').mkdir(parents=True)
    (root / 'splits' / 'abc.json').write_text(json.dumps(split))
    (root / 'splits' / 'latest.json').write_text(json.dumps({'path': 'splits/abc.json', 'sha256': 'abc'}))
    return root


@pytest.fixture
def root(proposed):
    """The same drives with every proposal confirmed by a person."""
    for route in (TEST_ROUTE, *OTHER_ROUTES):
        _confirm_all(proposed, route)
    return proposed


def test_only_a_person_s_type_is_a_label():
    from nnslr_tools import road_type_data as data
    assert data.CLASSES == ('town', 'rural', 'expressway', 'motorway', 'ramp', 'parking', 'zone30', 'garage',
                            'merge')
    # No proposal is a label, however sure: the ramps of the map's links and the acceleration lane least of all.
    for proposal in ({'road_type': 'town', 'sure': True, 'basis': 'maxspeed 50'},
                     {'road_type': 'motorway', 'sure': True, 'basis': 'highway motorway'},
                     {'road_type': 'ramp', 'sure': True, 'basis': 'highway motorway_link'},
                     {'road_type': 'ramp', 'sure': False, 'basis': 'highway motorway_link'},
                     {'road_type': 'ramp', 'sure': False, 'basis': 'highway trunk_link'},
                     {'road_type': 'merge', 'sure': True, 'basis': 'acceleration lane (highway motorway)'},
                     {'road_type': 'merge', 'sure': False, 'basis': 'acceleration lane (highway motorway_link)'},
                     {'road_type': 'rural', 'sure': False, 'basis': 'maxspeed 60', 'reviewed': False},
                     {'road_type': 'town', 'sure': True, 'reviewed': 'yes'}):
        assert data.frame_label(proposal) == (-1, 0.0)
    assert data.frame_label({'road_type': 'ramp', 'sure': True, 'basis': 'reviewed', 'reviewed': True}) == (RAMP, 1.0)
    # Reviewed without a valid type: "Can't tell", or a type no more of the list.
    assert data.frame_label({'road_type': None, 'sure': True, 'reviewed': True}) == (-1, 0.0)
    assert data.frame_label({'road_type': 'lane', 'sure': True, 'reviewed': True}) == (-1, 0.0)
    assert data.speed_features(15.0) == (0.5, 1.0)
    assert data.speed_features(100.0) == (2.0, 1.0)
    assert data.speed_features(None) == (0.0, 0.0)


def test_frames_without_gps_take_the_nearest_area():
    from nnslr_tools import road_type_data as data
    frames = [{'latitude': None, 'longitude': None}, {'latitude': 47.0, 'longitude': 8.0},
              {'latitude': None, 'longitude': None}, {'latitude': None, 'longitude': None},
              {'latitude': 47.1, 'longitude': 8.1}]
    cells = data.frame_cells(frames)
    assert cells[0] == cells[1] == cells[2] and cells[3] == cells[4] and cells[1] != cells[4]


def test_split_by_place_keeps_the_test_drive_and_a_buffer(root):
    from nnslr_tools import road_type_data as data
    dataset = data.load_dataset(root, min_length=10)
    split = dataset['split']
    assert split['policy'] == data.SPLIT_POLICY and split['test_routes'] == [TEST_ROUTE]
    assert set(split['partitions'][TEST_ROUTE]) == {'test'}
    # No area is both test (or validation) and train, and every train area is away from the test areas.
    by_cell = {}
    for route, frames in dataset['frames'].items():
        for cell, partition in zip(data.frame_cells(frames), split['partitions'][route]):
            by_cell.setdefault(cell, set()).add(partition)
    assert all(len(partitions) == 1 for partitions in by_cell.values())
    test_cells = {cell for cell, p in by_cell.items() if p == {'test'}}
    train_cells = {cell for cell, p in by_cell.items() if p == {'train'}}
    assert not data._around(test_cells) & train_cells
    assert split['counts']['validation'] > 0 and split['counts']['train'] > 0
    assert dataset['split_sha256'] == data.split_sha256(split)


def test_the_validation_never_leaves_out_a_main_class(monkeypatch):
    from nnslr_tools import road_type_data as data
    town, rural, motorway = (data.CLASS_INDEX[name] for name in ('town', 'rural', 'motorway'))
    classes_of = {
        (0, 0): {town: 20, rural: 20},                 # a village and its country road, alone
        (10, 0): {town: 30, rural: 30, motorway: 40},  # a junction onto the motorway...
        (10, 1): {motorway: 150},                      # ...with more motorway beside it, lost to its buffer
        (20, 0): {motorway: 150},
        (30, 0): {town: 100, rural: 100},
    }
    weight_of = {cell: sum(counts.values()) for cell, counts in classes_of.items()}
    assert data._validation_cells(set(classes_of), weight_of, classes_of, .15, 1) == {(10, 0)}
    # Without the rule the village wins: closer to every share, nothing lost to its buffer, and no motorway.
    monkeypatch.setattr(data, 'MAIN_CLASS_SHARE', 2.0)
    monkeypatch.setattr(data, 'NEEDED_CLASSES', ())
    assert data._validation_cells(set(classes_of), weight_of, classes_of, .15, 1) == {(0, 0)}


def test_the_validation_grows_through_the_areas_without_a_person_s_types():
    from nnslr_tools import road_type_data as data
    town, motorway = data.CLASS_INDEX['town'], data.CLASS_INDEX['motorway']
    # Three areas in a row, the middle one driven but not reviewed: the cluster reaches the far one across it.
    classes_of = {(0, 0): {town: 100}, (0, 1): {}, (0, 2): {motorway: 100}, (9, 9): {town: 400, motorway: 400}}
    weight_of = {cell: sum(counts.values()) for cell, counts in classes_of.items()}
    chosen = data._validation_cells(set(classes_of), weight_of, classes_of, .2, 1)
    assert chosen == {(0, 0), (0, 1), (0, 2)}
    # A place without one confirmed frame is never where a cluster starts.
    assert data._validation_cells({(0, 1)}, {(0, 1): 0}, {(0, 1): {}}, .2, 1) == set()


def test_sequences_are_runs_in_drive_order_broken_at_gaps(root):
    import numpy as np

    from nnslr_tools import road_type_data as data
    frames = data.route_frames(root, OTHER_ROUTES[2])
    runs = data.sequences(frames, ['train'] * len(frames), min_length=10)
    # The missing picture (segment 1, frame 30) breaks the drive in two; positions follow the drive.
    assert [len(run['positions']) for run in runs] == [90, 29]
    assert all(np.all(np.diff(run['positions']) == 1) for run in runs)
    assert runs[0]['positions'][0] == 0 and frames[60]['key'] == '1/0'
    # A buffer frame breaks a run too, and short runs are dropped.
    partitions = ['train'] * 50 + ['buffer'] + ['train'] * (len(frames) - 51)
    assert [len(run['positions']) for run in data.sequences(frames, partitions, min_length=30)] == [50, 39]


def test_summary_counts_a_person_s_labels_apart_from_the_frames_to_review(root):
    from nnslr_tools import road_type_data as data
    summary = data.load_dataset(root, min_length=10)['summary']
    assert summary['label_policy'] == data.LABEL_POLICY == 'confirmed-only'
    assert summary['all_frames']['labelled']['ramp'] == 40
    assert summary['all_frames']['labelled']['rural'] == 60
    assert summary['all_frames']['cant_tell'] == 0 and summary['all_frames']['to_review'] == 10


# -- a road type enters a training only by a person: a proposal teaches nothing and measures nothing ------------------
def _labels(root, route) -> list[int]:
    from nnslr_tools import road_type_data as data
    return [frame['label'] for frame in data.route_frames(root, route)]


def test_proposals_are_no_labels_until_a_person_confirms_or_corrects_them(proposed):
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_types
    route = OTHER_ROUTES[1]  # the motorway, a ramp from its link, a car park
    frames = data.route_frames(proposed, route)
    assert {frame['label'] for frame in frames} == {-1} and {frame['weight'] for frame in frames} == {0.0}
    assert [frame['map_label'] for frame in frames] == [MOTORWAY] * 50 + [RAMP] * 20 + [PARKING] * 50
    summary = data.load_dataset(proposed, min_length=10)['summary']
    assert sum(summary['all_frames']['labelled'].values()) == 0 and summary['all_frames']['to_review'] == 480

    # Confirming the proposed ramp, and correcting the motorway to a town: the very same labels of full weight.
    road_types.correct(proposed, route, '0/50', '1/9', 'ramp', 'someone')
    road_types.correct(proposed, route, '0/0', '0/9', 'town', 'someone')
    frames = data.route_frames(proposed, route)
    assert [frame['label'] for frame in frames] == [TOWN] * 10 + [-1] * 40 + [RAMP] * 20 + [-1] * 50
    assert {frame['weight'] for frame in frames if frame['label'] >= 0} == {1.0}
    assert {frame['weight'] for frame in frames if frame['label'] < 0} == {0.0}


def test_a_confirmation_outlives_new_proposals_of_the_map(proposed):
    from nnslr_tools import road_types
    route = OTHER_ROUTES[1]
    road_types.correct(proposed, route, '0/50', '1/9', 'ramp', 'someone')
    kept = road_types.corrections_path(proposed, route).read_bytes()
    # The map read again: other types, and the frames confirmed are now the map's town.
    _write_route(proposed, route, _types(('town', 120)), (48.0, 9.0), (0.0003, 0.0))
    assert road_types.corrections_path(proposed, route).read_bytes() == kept
    assert _labels(proposed, route) == [-1] * 50 + [RAMP] * 20 + [-1] * 50


def test_cant_tell_is_reviewed_no_label_and_stays_in_the_sequence(proposed):
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_types
    route = OTHER_ROUTES[0]
    road_types.correct(proposed, route, '0/0', '0/19', None, 'someone')
    road_types.correct(proposed, route, '0/20', '0/39', 'rural', 'someone')
    frames = data.route_frames(proposed, route)
    assert [frame['label'] for frame in frames[:40]] == [-1] * 20 + [RURAL] * 20
    assert {frame['weight'] for frame in frames[:20]} == {0.0} and all(frame['reviewed'] for frame in frames[:20])
    # In the memory of the sequence all the same, with the proposals around them.
    runs = data.sequences(frames, ['train'] * len(frames), min_length=10)
    assert [len(run['positions']) for run in runs] == [120]
    summary = data.load_dataset(proposed, min_length=10)['summary']
    assert summary['all_frames']['cant_tell'] == 20 and summary['all_frames']['labelled']['rural'] == 20
    assert summary['all_frames']['to_review'] == 480 - 40


def test_the_measures_take_only_a_person_s_types_as_truth(proposed):
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_metrics as metrics
    from nnslr_tools import road_types
    route = OTHER_ROUTES[1]
    # A person confirmed the motorway's start and the car park's end: the proposals in between are no truth.
    road_types.correct(proposed, route, '0/0', '0/29', 'motorway', 'someone')
    road_types.correct(proposed, route, '1/30', '1/59', 'parking', 'someone')
    frames = data.route_frames(proposed, route)
    labels = np.array([frame['label'] for frame in frames])
    weights = np.array([frame['weight'] for frame in frames])
    # Right on the confirmed frames, flipping between town and rural on every frame of the proposals.
    predicted = np.where(labels >= 0, labels, np.arange(len(labels)) % 2 * (RURAL - TOWN) + TOWN)
    result = metrics.sequence_metrics([{'labels': labels, 'weights': weights, 'predicted': predicted}], data.CLASSES)
    # Only the confirmed frames are measured, right every one: the wrong answers on the map's ramp count for nothing.
    assert result['labelled']['classes']['motorway']['support'] == 30
    assert result['labelled']['classes']['parking']['support'] == 30
    assert result['labelled']['classes']['ramp']['support'] == 0
    assert sum(map(sum, result['confusion'])) == 60
    assert result['labelled']['classes']['motorway']['f1'] == result['labelled']['classes']['parking']['f1'] == 1.0
    # The proposals are a gap too long to be a transition, and the flips inside it are no false switches.
    assert metrics.change_points(labels, data.CLASSES) == []
    assert not any(kind['count'] for kind in result['transitions'].values())
    assert result['false_switches_per_hour'] == 0
    # What the map says there, were it the truth, is another measure: 40 frames more, and all the flips of the ramp.
    map_labels = np.array([frame['map_label'] for frame in frames])
    assert metrics.switches(predicted, map_labels)[0] > 0 and metrics.change_points(map_labels, data.CLASSES)


def test_enough_labels_name_what_each_partition_lacks():
    from nnslr_tools import road_type_data as data

    def summary(train=None, validation=None, test=None):
        parts = {name: {'labelled': {kind: 0 for kind in data.CLASSES} | (labelled or {})}
                 for name, labelled in (('train', train), ('validation', validation), ('test', test))}
        return {'partitions': parts}
    full = {name: 1000 for name in data.NEEDED_CLASSES}
    assert data.missing_labels(summary(full, full, full)) == []
    missing = data.missing_labels(summary(full, {**full, 'ramp': 0}, {**full, 'ramp': 29}))
    assert missing == ['validation has 0 of 30 ramp frames', 'test has 29 of 30 ramp frames']
    few = {name: 30 for name in data.NEEDED_CLASSES}
    assert data.missing_labels(summary(few, full, full)) == ['train has 120 of 2000 confirmed frames']
    assert 'test has 0 of 300 confirmed frames' in data.missing_labels(summary(full, full))


def test_too_few_confirmed_frames_stop_a_training_before_it_writes_anything(proposed, tmp_path, capsys):
    from nnslr_tools.cli import main
    run_dir = tmp_path / 'run'
    # The proposals alone would fill every partition: they are not counted, and nothing falls back on them.
    assert main(['train-road-type', '--dry-run', '--run-dir', str(run_dir), '--data-root', str(proposed)]) != 0
    message = capsys.readouterr().err
    assert 'a proposal of the map is never a label' in message and 'Data › Roads' in message
    assert 'train has 0 of 2000 confirmed frames' in message and 'validation has 0 of 30 ramp frames' in message
    assert not run_dir.exists()


def test_a_dry_run_on_a_person_s_types_records_its_label_policy(root, tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    # The thresholds and the shortest sequence are those of a real data root.
    monkeypatch.setattr(data, 'ENOUGH_LABELS', {'train': 1, 'validation': 1, 'test': 1})
    monkeypatch.setattr(data, 'NEEDED_CLASSES', ())
    monkeypatch.setattr(data, 'load_dataset', partial(data.load_dataset, min_length=10))
    run_dir = tmp_path / 'run'
    result = net.train_road_type(root, run_dir, dry_run=True, workers=0, log=lambda *_: None)
    assert result['config']['label_policy'] == data.LABEL_POLICY
    assert 'unsure_weight' not in result['config'] and 'ramp_weight' not in result['config']
    assert json.loads((run_dir / 'dataset.json').read_text())['label_policy'] == data.LABEL_POLICY
    assert '"label_policy": "confirmed-only"' in (run_dir / 'result.json').read_text()  # what run_road_type.sh looks for


def test_a_dry_run_takes_the_sequences_with_the_most_confirmed_frames(proposed):
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    from nnslr_tools import road_types
    road_types.correct(proposed, OTHER_ROUTES[1], '1/10', '1/29', 'parking', 'someone')
    frames = {route: data.route_frames(proposed, route) for route in (TEST_ROUTE, *OTHER_ROUTES)}
    sequences = [{**run, 'route': route, 'partition': 'train'} for route, found in frames.items()
                 for run in data.sequences(found, ['train'] * len(found), min_length=10)]
    runs = net._dry_runs({'frames': frames, 'sequences': sequences})
    assert runs[0]['route'] == OTHER_ROUTES[1] and len(runs[0]['positions']) == 40
    # It starts at the first of them, not at the start of the drive where nothing would be learned.
    assert runs[0]['positions'][0] == runs[0]['start'] == 70
    assert sum(frames[OTHER_ROUTES[1]][at]['label'] >= 0 for at in runs[0]['positions']) == 20


def _sampler_of(root, route, **options):
    import random

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    dataset = data.load_dataset(root, min_length=10)
    net.attach_signs(dataset, root)
    net.attach_motion(dataset, root)
    frames = dataset['frames'][route]
    runs = [{**run, 'route': route, 'partition': 'train'}
            for run in data.sequences(frames, ['train'] * len(frames), min_length=10)]
    return net.WindowSampler(dataset, runs, random.Random(1), merge_share=0.0, focus_share=0.0, junction_share=0.0,
                             **options)


def test_windows_go_around_the_confirmed_frames_and_keep_the_proposals_as_memory(proposed):
    from nnslr_tools import road_types
    route = OTHER_ROUTES[0]
    road_types.correct(proposed, route, '1/30', '1/39', 'town', 'someone')
    sampler = _sampler_of(proposed, route)
    windows = [sampler.window(20) for _ in range(100)]
    for window in windows:
        labels, weights = window[2], window[3]
        assert (labels >= 0).any()
        # The frames around them come along, for the memory, and weigh nothing.
        assert not weights[labels < 0].any() and (weights[labels >= 0] == 1.0).all()
    assert any((window[2] < 0).any() for window in windows)
    # The junction head's stage does not care for the road types: its windows are anywhere.
    anywhere = _sampler_of(proposed, route, around_labels=False)
    assert any(not (anywhere.window(20)[2] >= 0).any() for _ in range(100))


def test_a_drive_without_a_person_s_types_still_gives_windows(proposed):
    window = _sampler_of(proposed, OTHER_ROUTES[0]).window(20)
    assert (window[2] == -1).all() and not window[3].any()


def test_frame_cache_is_the_pil_bilinear_resize_and_is_reused(root):
    import numpy as np

    from nnslr_tools import road_type_data as data
    frames = {OTHER_ROUTES[2]: data.route_frames(root, OTHER_ROUTES[2])}
    paths = data.build_frame_cache(root, frames, workers=1, log=lambda _: None)
    array = np.load(paths[OTHER_ROUTES[2]])
    assert array.shape == (120, data.FRAME_HEIGHT, data.FRAME_WIDTH, 3) and array.dtype == np.uint8
    assert np.array_equal(array[5], data.resized_frame(frames[OTHER_ROUTES[2]][5]['image']))
    assert not array[90].any()  # the missing picture
    stamp = paths[OTHER_ROUTES[2]].stat().st_mtime_ns
    data.build_frame_cache(root, frames, workers=1, log=lambda _: None)
    assert paths[OTHER_ROUTES[2]].stat().st_mtime_ns == stamp


def test_per_class_f1_over_masked_frames():
    import numpy as np

    from nnslr_tools.road_type_metrics import per_class_f1
    classes = ('a', 'b', 'c')
    result = per_class_f1(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 0]), np.array([1, 1, 1, 1, 0], bool), classes)
    assert result['classes']['a'] == {'precision': 1.0, 'recall': 0.5, 'f1': 0.6667, 'support': 2, 'predicted': 1}
    assert result['classes']['c']['f1'] is None
    assert result['macro_f1'] == round((2 / 3 + 0.8) / 2, 4) and result['frames'] == 4


def test_transitions_their_delay_and_false_switches():
    import numpy as np

    from nnslr_tools.road_type_metrics import change_points, debounced, detection_delay, majority, switches
    classes = ROAD_TYPES
    labels = np.array([RAMP] * 20 + [-1] * 2 + [MOTORWAY] * 40)
    changes = change_points(labels, classes)
    assert changes == [{'time': 22, 'old': RAMP, 'new': MOTORWAY, 'kind': 'ramp_to_motorway'}]
    predicted = np.array([RAMP] * 25 + [MOTORWAY] * 37)
    assert detection_delay(predicted, changes[0]) == 3
    assert detection_delay(np.array([RAMP] * 62), changes[0]) is None
    # Early by 2 seconds.
    assert detection_delay(np.array([RAMP] * 20 + [MOTORWAY] * 42), changes[0]) == -2
    # A short run of fewer than 10 seconds is no transition.
    assert change_points(np.array([TOWN] * 20 + [RURAL] * 5 + [TOWN] * 20), classes) == []
    flicker = np.array([TOWN] * 30 + [RURAL] + [TOWN] * 29)
    assert switches(flicker, np.array([TOWN] * 60)) == (2, 40)
    assert switches(debounced(flicker), np.array([TOWN] * 60))[0] == 0
    assert list(debounced(np.array([0, 1, 1, 1, 0, 0, 0]))) == [0, 0, 0, 1, 1, 1, 0]
    assert list(majority(np.array([0, 1, 0, 1, 1]), window=3)) == [0, 1, 0, 1, 1]


def test_after_merge_accuracy_counts_the_first_minute_on_the_motorway():
    import numpy as np

    from nnslr_tools.road_type_metrics import after_merge, sequence_metrics
    classes = ROAD_TYPES
    labels = np.array([RAMP] * 10 + [MOTORWAY] * 80)
    # The frame network says rural for 20 seconds after the ramp; the memory keeps motorway.
    frame = np.array([RAMP] * 10 + [RURAL] * 20 + [MOTORWAY] * 60)
    memory = np.array([RAMP] * 10 + [MOTORWAY] * 80)
    without = after_merge([{'labels': labels, 'predicted': frame}], classes)
    with_memory = after_merge([{'labels': labels, 'predicted': memory}], classes)
    assert without['merges'] == 1 and without['frames'] == 60
    assert without['accuracy'] == round(40 / 60, 4) and without['said']['rural'] == 20
    assert with_memory['accuracy'] == 1.0
    metrics = sequence_metrics([{'labels': labels, 'weights': np.ones(90), 'predicted': memory}], classes)
    assert metrics['transitions']['ramp_to_motorway']['found'] == 1
    assert metrics['after_merge']['merges'] == 1


def test_onnx_step_contract(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False).eval()
    # The step's GRU written out by hand gives the numbers of torch's GRU over a sequence.
    x = torch.randn(1, 5, net.FEATURES + 2)
    with torch.no_grad():
        expected, last = model.gru(x)
        hidden = torch.zeros(1, net.HIDDEN)
        for t in range(5):
            hidden = net.gru_step(model.gru, x[:, t], hidden)
            assert torch.allclose(hidden, expected[:, t], atol=1e-5)
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru',
                'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['task'] == 'road_type' and contract['verification']['passed'] is True
    assert contract['verification']['rollout']['checked'] is True
    assert [i['name'] for i in contract['inputs']] == ['image', 'speed', 'hidden']
    assert contract['inputs'][0]['shape'] == [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH]
    assert [o['name'] for o in contract['outputs']] == ['probabilities', 'hidden_out']
    assert contract['output']['classes'] == list(data.CLASSES)
    import onnx
    graph = onnx.load(result['onnx']).graph
    assert not [node.op_type for node in graph.node if node.op_type in ('GRU', 'LSTM', 'RNN')]
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    hidden = np.zeros((1, net.HIDDEN), np.float32)
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    probabilities, hidden_out = session.run(None, {'image': image, 'speed': np.array([[0.5, 1.0]], np.float32),
                                                   'hidden': hidden})
    assert probabilities.shape == (1, len(data.CLASSES)) and abs(float(probabilities.sum()) - 1) < 1e-5
    assert hidden_out.shape == (1, net.HIDDEN) and np.abs(hidden_out).max() <= 1


# -- the motorway sign the car confirmed (nnslr_tools.road_type_signs) ------------------------------------------------
def test_sign_features_age_out_after_two_minutes():
    from nnslr_tools import road_type_signs as signs
    assert signs.sign_features(None) == signs.NO_SIGN == (2.0, 0.0)
    assert signs.sign_features(-1) == signs.NO_SIGN and signs.sign_features(float('inf')) == signs.NO_SIGN
    assert signs.sign_features(0) == (0.0, 1.0) and signs.sign_features(30) == (0.5, 1.0)
    assert signs.sign_features(120) == (2.0, 1.0) and signs.sign_features(120.5) == signs.NO_SIGN
    # A dataset frame is its segment's minute and its index; a sign counts from the frame it was confirmed at.
    frames = [{'segment': segment, 'index': index} for segment in range(4) for index in range(60)]
    found = signs.frame_sign_features(frames, [61.0, 10.0])
    assert found.dtype.name == 'float32' and found.shape == (240, 2)
    assert found[9].tolist() == [2.0, 0.0] and found[10].tolist() == [0.0, 1.0]
    assert found[60].tolist() == pytest.approx([50 / 60, 1.0]) and found[61].tolist() == [0.0, 1.0]
    assert found[181].tolist() == [2.0, 1.0] and found[182].tolist() == [2.0, 0.0]


def test_the_map_puts_the_motorway_sign_at_the_start_of_every_entry_ramp():
    from nnslr_tools import road_type_signs as signs
    road = [TOWN] * 30 + [RAMP] * 10 + [MOTORWAY] * 40 + [RAMP] * 10 + [MOTORWAY] * 30  # an entry, then an interchange
    frames = [{'segment': i // 60, 'index': i % 60, 'label': -1, 'map_label': label} for i, label in enumerate(road)]
    assert signs.map_sign_times(frames, ROAD_TYPES) == [30.0]
    ages = signs.sign_ages([29, 30, 31.5, 200], [30.0])
    assert ages[0] == float('inf') and list(ages[1:]) == [0.0, 1.5, 170.0]
    assert signs.features_at([29, 30, 150, 151], [30.0]).tolist() == [[2.0, 0.0], [0.0, 1.0], [2.0, 1.0], [2.0, 0.0]]
    assert signs.features_at([1, 2], []).tolist() == [[2.0, 0.0], [2.0, 0.0]]


def test_confirmations_are_those_of_the_cars_memory():
    from nnslr_tools import road_type_signs as signs
    package = signs.package_path()
    if not (package / 'vision.py').is_file():
        pytest.skip('no sunnypilot clone with the car code')
    vision = signs.load_vision(package)
    start = [['other_sign:motorway_start', 0.9]]
    readings = [(10.0, start), (10.2, start), (10.4, start),  # a sign read three times: confirmed at its 2nd and 3rd
                (30.0, start),  # once only: not confirmed
                (50.0, [['other_sign:town_entry', 0.9]]), (50.2, [['other_sign:town_entry', 0.9]]),  # not a motorway
                (70.0, start), (72.0, start)]  # twice but 2 s apart: not confirmed
    assert signs.confirmations(vision, readings) == [[10.2, 'motorway'], [10.4, 'motorway']]
    assert signs.sign_times({'confirmed': [[10.2, 'motorway'], [50.2, 'town']]}) == [10.2]


def test_board_confirmations_are_those_of_the_cars_memory_and_never_a_road_sign():
    from nnslr_tools import road_type_signs as signs
    package = signs.package_path()
    if not (package / 'vision.py').is_file():
        pytest.skip('no sunnypilot clone with the car code')
    vision = signs.load_vision(package)
    if not hasattr(vision, 'BOARD_SIGNS'):
        assert signs.board_confirmations(vision, [(1.0, [['other_sign:motorway_direction', 0.9]])]) == []
        pytest.skip('a car code without the green boards')
    board = [['other_sign:motorway_direction', 0.9]]
    readings = [(10.0, board), (10.2, board),  # twice within 1.5 s: confirmed at the 2nd
                (30.0, board),  # once only
                (50.0, board + [['maximum_speed:80', 0.9]]), (50.2, board + [['maximum_speed:80', 0.9]])]
    assert signs.board_confirmations(vision, readings) == [10.2, 50.2]
    # A board is never a motorway sign: the sign input stays empty.
    assert signs.confirmations(vision, readings) == []
    assert signs.board_times({'boards': [50.2, 10.2]}) == [10.2, 50.2] and signs.board_times(None) == []


def test_a_chosen_classifier_is_part_of_the_readings_fingerprint(tmp_path):
    from nnslr_tools import road_type_signs as signs
    package = tmp_path / 'package'
    (package / 'models').mkdir(parents=True)
    (package / 'vision.py').write_text('# car code\n')
    for file in signs.NETWORK_FILES.values():
        (package / 'models' / file).write_bytes(b'car ' + file.encode())
    other = tmp_path / 'other' / 'classifier.onnx'
    other.parent.mkdir()
    other.write_bytes(b'another classifier')
    assert signs.network_paths(package, other)['classifier'] == other
    assert signs.fingerprint(package) != signs.fingerprint(package, other)
    assert signs.fingerprint(package)['version'] == signs.CACHE_VERSION == 2


# -- the comma's rule on the merges (road_type_metrics) ---------------------------------------------------------------
def _entry_drive():
    """60 s of town, 20 s of ramp, 100 s of motorway."""
    import numpy as np
    return np.array([TOWN] * 60 + [RAMP] * 20 + [MOTORWAY] * 100)


def test_ramp_entries_find_the_ramp_and_what_came_before():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    labels = _entry_drive()
    assert metrics.ramp_entries(labels, ROAD_TYPES) == [{'ramp': 60, 'merge': 80, 'before': 'town'}]
    # A ramp of two seconds, or one that leaves the motorway, is no entry.
    assert metrics.ramp_entries(np.array([TOWN] * 5 + [RAMP] * 2 + [MOTORWAY] * 5), ROAD_TYPES) == []
    assert metrics.ramp_entries(np.array([MOTORWAY] * 5 + [RAMP] * 5 + [TOWN] * 5), ROAD_TYPES) == []


def test_comma_release_on_time_on_the_ramp_late_and_not_held():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    entry = {'ramp': 60, 'merge': 80, 'before': 'town'}
    no_sign = np.full(180, np.inf)
    # The road type says ramp at the number (second 62) and motorway from the merge: released 8 s later.
    found = metrics.comma_release(_entry_drive(), no_sign, entry, ROAD_TYPES)
    assert found == {'number_at': -18, 'held_by': 'ramp', 'release': 8, 'outcome': 'in_time'}
    # Motorway from second 65, still on the ramp: a false general limit on the ramp.
    early = _entry_drive()
    early[65:80] = MOTORWAY
    assert metrics.comma_release(early, no_sign, entry, ROAD_TYPES)['release'] == -7
    assert metrics.comma_release(early, no_sign, entry, ROAD_TYPES)['outcome'] == 'on_ramp'
    # The road type says town at the number and no sign: not the ramp's, but read off the carriageways it ends after
    # 10 s of motorway (vision.CARRIAGEWAY_LAG: the motorway began 18 s after it).
    town = _entry_drive()
    town[60:70] = TOWN
    assert metrics.comma_release(town, no_sign, entry, ROAD_TYPES) == {
        'number_at': -18, 'held_by': 'off', 'release': 10, 'outcome': 'in_time'}
    # Motorway from second 65, 3 s after the number: it may be the motorway's own, it holds.
    soon = town.copy()
    soon[65:] = MOTORWAY
    assert metrics.comma_release(soon, no_sign, entry, ROAD_TYPES)['outcome'] == 'late'
    # Read on the motorway with no sign: nothing to release.
    on_road = _entry_drive()
    on_road[60:] = MOTORWAY
    assert metrics.comma_release(on_road, no_sign, entry, ROAD_TYPES)['outcome'] == 'not_held'
    # A motorway sign 20 s before the number holds it; town after the merge breaks the 8 s, so it is late.
    sign = np.full(180, np.inf)
    sign[42:] = np.arange(180 - 42)
    late = town.copy()
    late[80:] = np.where(np.arange(100) % 6 == 5, TOWN, MOTORWAY)
    result = metrics.comma_release(late, sign, entry, ROAD_TYPES)
    assert result['held_by'] == 'sign' and result['release'] is None and result['outcome'] == 'late'
    both = metrics.comma_release(_entry_drive(), sign, entry, ROAD_TYPES)
    assert both['held_by'] == 'ramp+sign' and both['outcome'] == 'in_time'
    summary = metrics.release_summary([found, result, both, metrics.comma_release(town, no_sign, entry, ROAD_TYPES),
                                       metrics.comma_release(on_road, no_sign, entry, ROAD_TYPES)])
    assert summary['merges'] == 5 and summary['held'] == 4 and summary['held_by_sign'] == 2
    assert summary['held_by_off'] == 1
    assert (summary['in_time'], summary['late'], summary['not_held'], summary['on_ramp']) == (3, 1, 1, 0)
    assert summary['median_release'] == 8.0 and summary['releases'] == [8, None, 8, 10, None]



def test_the_acceleration_lane_ends_the_ramps_limit_and_is_the_carriageway_for_the_road_measures():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    lane = ROAD_TYPES.index('merge')
    labels = _entry_drive()
    labels[78:91] = lane
    # The merge is the first second of the lane; the ramp before it is still an entry.
    entry = {'ramp': 60, 'merge': 78, 'before': 'town'}
    assert metrics.ramp_entries(labels, ROAD_TYPES) == [entry]
    no_sign = np.full(180, np.inf)
    # The answer says merge from second 81. Without a motorway sign the carriageway is not known: the lane counts for
    # the 8 s of carriageway in a row (from second 80), released at 88 as by a network without the class.
    held = _entry_drive()
    held[81:88] = lane
    assert metrics.comma_release(held, no_sign, entry, ROAD_TYPES) == {
        'number_at': -16, 'held_by': 'ramp', 'release': 10, 'outcome': 'in_time'}
    assert metrics.comma_release(_entry_drive(), no_sign, entry, ROAD_TYPES)['release'] == 10
    # A motorway sign at the ramp's start names the carriageway: released on the lane's first second.
    sign = np.full(180, np.inf)
    sign[60:] = np.arange(120)
    assert metrics.comma_release(held, sign, entry, ROAD_TYPES) == {
        'number_at': -16, 'held_by': 'ramp+sign', 'release': 3, 'outcome': 'in_time'}
    # Said on the lane since the sign: the number is the carriageway's, the sign no longer holds it.
    on_lane = held.copy()
    on_lane[61] = lane
    assert metrics.comma_release(on_lane, sign, entry, ROAD_TYPES)['held_by'] == 'ramp'
    # The road measures: the lane is the motorway after it, in the labels and the answers.
    folded, answer = metrics.fold_merge(labels, held, ROAD_TYPES)
    assert (folded[78:91] == MOTORWAY).all() and (answer[81:88] == MOTORWAY).all()
    found = metrics.sequence_metrics([{'labels': labels, 'weights': np.ones(180), 'predicted': held}], ROAD_TYPES)
    assert found['transitions']['ramp_to_motorway']['count'] == 1
    assert found['transitions']['other']['count'] == 0 and found['after_merge']['merges'] == 1
    # The F1 keeps it apart: three of its seconds said, of thirteen.
    assert found['weight_1']['classes']['merge']['support'] == 13
    assert found['weight_1']['classes']['merge']['predicted'] == 7
    # An expressway's lane folds into the expressway; a lane at the end of a sequence into the motorway.
    road = _entry_drive()
    road[78:] = ROAD_TYPES.index('expressway')
    road[78:85] = lane
    assert (metrics.fold_labels(road, ROAD_TYPES)[78:85] == ROAD_TYPES.index('expressway')).all()
    assert (metrics.fold_labels(np.array([RAMP] * 5 + [lane] * 3), ROAD_TYPES)[5:] == MOTORWAY).all()



def test_comma_release_forgets_the_signs_carriageway_as_the_car_does():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    lane, expressway = ROAD_TYPES.index('merge'), ROAD_TYPES.index('expressway')
    # A motorway sign, 18 s of motorway, then a link to an expressway: the sign no longer names the carriageway, the
    # lane counts for the 8 s (test_vision: a ramp off the motorway does not take its limit to an expressway).
    held = np.array([MOTORWAY] * 18 + [RAMP] * 12 + [lane] + [expressway] * 40)
    sign = np.arange(len(held), dtype=float)
    entry = {'ramp': 18, 'merge': 30, 'before': 'motorway'}
    assert metrics.comma_release(held, sign, entry, ROAD_TYPES) == {
        'number_at': -10, 'held_by': 'ramp+sign', 'release': 8, 'outcome': 'in_time'}
    # An expressway's link to the motorway, the road type still the expressway's for 3 s after the sign (the
    # debounce): the sign names the motorway, released on the lane's first second.
    held = np.array([expressway] * 13 + [RAMP] * 17 + [lane] * 8 + [MOTORWAY] * 30)
    sign = np.full(len(held), np.inf)
    sign[10:] = np.arange(len(held) - 10)
    entry = {'ramp': 13, 'merge': 30, 'before': 'expressway'}
    assert metrics.comma_release(held, sign, entry, ROAD_TYPES)['release'] == 0
    # The drive's previous ramp's limit ended after the sign: the number is the carriageway's, the sign holds nothing.
    assert metrics.comma_release(held, sign, entry, ROAD_TYPES, joined_at=11)['held_by'] == 'ramp'
    assert metrics.comma_release(held, sign, entry, ROAD_TYPES, joined_at=9)['held_by'] == 'ramp+sign'
    # The car's dwell: a carriageway right after the lane counts from the lane's start.
    assert metrics._road_type_since(np.array([RAMP, lane, lane, MOTORWAY, MOTORWAY]), 4, lane) == 1

def test_the_lanes_lead_is_the_ramps_for_the_road_measures():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    lane = ROAD_TYPES.index('merge')
    labels = _entry_drive()
    labels[78:91] = lane
    lead = np.zeros(180, dtype=bool)
    lead[78:80] = True
    # The lane's first two seconds were the ramp's metres: the ramp for the road measures, the rest the motorway.
    folded = metrics.fold_labels(labels, ROAD_TYPES, lead)
    assert (folded[78:80] == RAMP).all() and (folded[80:91] == MOTORWAY).all()
    # "merge" said on the lead is the ramp there; said in town it stays an error (the motorway).
    said = labels.copy()
    said[10] = lane
    _, answer = metrics.fold_merge(labels, said, ROAD_TYPES, lead)
    assert (answer[78:80] == RAMP).all() and (answer[80:91] == MOTORWAY).all() and answer[10] == MOTORWAY
    # The entry: the merge at the lane's first second, the lead in the ramp's length.
    assert metrics.ramp_entries(labels, ROAD_TYPES, lead) == [{'ramp': 60, 'merge': 78, 'before': 'town'}]
    short = np.array([TOWN] * 5 + [RAMP] + [lane] * 6 + [MOTORWAY] * 5)
    short_lead = np.zeros(len(short), dtype=bool)
    short_lead[6:8] = True
    assert metrics.ramp_entries(short, ROAD_TYPES) == []
    assert metrics.ramp_entries(short, ROAD_TYPES, short_lead) == [{'ramp': 5, 'merge': 6, 'before': 'town'}]


def test_a_detection_needs_one_class_said_in_a_row():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    lane = ROAD_TYPES.index('merge')
    change = {'time': 20, 'new': MOTORWAY}
    raw = np.array([RAMP] * 20 + [lane, lane, MOTORWAY, MOTORWAY, MOTORWAY] + [MOTORWAY] * 20)
    folded = np.where(raw == lane, MOTORWAY, raw)
    # Folded, three seconds of motorway from 20; for the car merge, merge, motorway is no run: found at 22.
    assert metrics.detection_delay(folded, change) == 0
    assert metrics.detection_delay(folded, change, raw) == 2
    # The car's debounce first, then the fold: merge and motorway in turn on the lane are no answer the car holds; it
    # stays on the ramp until three seconds of one class (second 23 to 25), found 13 s after the merge.
    labels = np.array([RAMP] * 10 + [lane] * 5 + [MOTORWAY] * 40)
    flicker = {'labels': labels, 'weights': np.ones(55), 'predicted': np.array([RAMP] * 10 + [lane, MOTORWAY] * 7
                                                                               + [MOTORWAY] * 31)}
    found = metrics.after_merge([flicker], ROAD_TYPES)
    assert found['accuracy'] == 1.0 and found['accuracy_debounced'] == round(30 / 45, 4)
    assert metrics.sequence_metrics([flicker], ROAD_TYPES)['transitions']['ramp_to_motorway']['delays'] == [13]


def test_confusion_counts_the_masked_frames_by_truth_and_answer():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    truth = np.array([0, 0, 1, 1, 2, -1])
    answer = np.array([0, 1, 1, 1, 0, 2])
    mask = np.array([1, 1, 1, 0, 1, 1])
    assert metrics.confusion(truth, answer, mask, 3) == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_onnx_step_contract_with_the_sign_input(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False, signs=True).eval()
    assert model.gru.input_size == net.FEATURES + 2 + net.SIGN_INPUTS
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru', 'signs': True,
                'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['verification']['passed'] is True and contract['step']['signs'] is True
    assert [i['name'] for i in contract['inputs']] == ['image', 'speed', 'sign', 'hidden']
    assert any('_road_sign_at' in line for line in contract['car_input']['steps'])
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    feeds = {'image': image, 'speed': np.array([[0.5, 1.0]], np.float32),
             'hidden': np.zeros((1, net.HIDDEN), np.float32)}
    seen = session.run(None, {**feeds, 'sign': np.array([[0.1, 1.0]], np.float32)})[0]
    unseen = session.run(None, {**feeds, 'sign': np.array([[2.0, 0.0]], np.float32)})[0]
    assert seen.shape == (1, len(data.CLASSES)) and not np.allclose(seen, unseen)


def test_onnx_step_contract_with_the_board_input(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False, signs=True, motion=True, boards=True).eval()
    assert model.gru.input_size == net.FEATURES + 2 + net.SIGN_INPUTS + net.MOTION_INPUTS + net.BOARD_INPUTS
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru', 'signs': True,
                'motion': True, 'boards': True, 'board_signs': [net.BOARD_CLASS], 'classes': list(data.CLASSES),
                'state_dict': model.state_dict(), 'epoch': 1, 'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['verification']['passed'] is True and contract['step']['boards'] is True
    assert contract['step']['board_signs'] == ['other_sign:motorway_direction']
    assert [i['name'] for i in contract['inputs']] == ['image', 'speed', 'sign', 'motion', 'board', 'hidden']
    assert any('board_age' in line for line in contract['car_input']['steps'])
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    from nnslr_tools.car_motion import NO_MOTION
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    feeds = {'image': image, 'speed': np.array([[0.5, 1.0]], np.float32), 'sign': np.array([[2.0, 0.0]], np.float32),
             'motion': np.array([NO_MOTION], np.float32), 'hidden': np.zeros((1, net.HIDDEN), np.float32)}
    seen = session.run(None, {**feeds, 'board': np.array([[0.1, 1.0]], np.float32)})[0]
    unseen = session.run(None, {**feeds, 'board': np.array([[2.0, 0.0]], np.float32)})[0]
    assert not np.allclose(seen, unseen)


def test_a_start_without_the_board_input_goes_on_as_it_was(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    torch.manual_seed(0)
    old = net.build_road_type_net('gru', pretrained=False, signs=True, motion=True).eval()
    path = tmp_path / 'run' / 'road-type' / 'gru' / 'road-type-best.pt'
    path.parent.mkdir(parents=True)
    torch.save({'task': 'road_type', 'memory': 'gru', 'signs': True, 'motion': True, 'classes': list(data.CLASSES),
                'state_dict': old.state_dict(), 'epoch': 7}, path)
    new = net.build_road_type_net('gru', pretrained=False, signs=True, motion=True, boards=True).eval()
    start = net.road_type_start(torch, new, path, 'gru', {'test_routes': []})
    assert start['widened'] == ['gru.weight_ih_l0'] and start['fresh'] == []
    images = torch.rand(1, 3, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH)
    speed, sign = torch.rand(1, 3, 2), torch.rand(1, 3, 2)
    motion = torch.rand(1, 3, net.MOTION_INPUTS)
    with torch.no_grad():
        before = old(images, speed, None, sign, motion)[0]
        for board in (torch.tensor([[[0.1, 1.0]] * 3]), None):
            assert torch.allclose(new(images, speed, None, sign, motion, board=board)[0], before, atol=1e-6)
    # A start with other inputs is not widened: its memory starts fresh.
    other = net.build_road_type_net('gru', pretrained=False, signs=True, motion=False, boards=True)
    assert 'gru.weight_ih_l0' in net.road_type_start(torch, other, path, 'gru', {'test_routes': []})['fresh']


def test_a_person_s_motorway_over_the_map_s_ramp_takes_its_sign_away(proposed):
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_signs as signs
    from nnslr_tools import road_types
    route = OTHER_ROUTES[2]
    # The map's ramp beside the carriageway, and the car on the carriageway (00000168--2b00afdc52, frame 18/57).
    _write_route(proposed, route, _types(('town', 30), ('ramp', 30, False, 'highway motorway_link', 'motorway_link'),
                                         ('motorway', 60, True, 'highway motorway')), (48.5, 9.5), (0.0003, 0.0))
    # A proposal is no label, but it still says where the sign of the entry stands: an input, never an answer.
    assert signs.map_sign_times(data.route_frames(proposed, route), ROAD_TYPES) == [30.0]
    road_types.correct(proposed, route, '0/30', '0/59', 'motorway', 'someone')
    frames = data.route_frames(proposed, route)
    assert signs.map_sign_times(frames, ROAD_TYPES) == []
    assert (frames[57]['label'], frames[57]['weight']) == (MOTORWAY, 1.0)


def test_train_windows_carry_the_board_input(root):
    import random

    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    from nnslr_tools import road_type_signs as signs
    dataset = data.load_dataset(root)
    net.attach_signs(dataset, root)
    net.attach_motion(dataset, root)
    coverage = net.attach_boards(dataset, root)
    # Not read with the boards: none seen.
    assert coverage['read'] == 0 and coverage['boards'] == 0
    runs = [run for run in dataset['sequences'] if run['partition'] == 'train']
    route = runs[0]['route']
    runs = [run for run in runs if run['route'] == route]
    seconds = dataset['seconds'][route]
    dataset['board_times'][route] = [float(seconds[runs[0]['positions'][5]])]
    dataset['boards'][route] = signs.features_at(seconds, dataset['board_times'][route])
    sampler = net.WindowSampler(dataset, runs, random.Random(1))
    kinds = {'seen': 0, 'none': 0}
    for _ in range(200):
        board = sampler.window(20)[10]
        assert board.shape == (20, 2)
        kinds['seen' if board[:, 1].any() else 'none'] += 1
    assert kinds['seen'] and kinds['none']


# -- the junction head (nnslr_tools.junctions labels, road_type_metrics junction measure) -----------------------------
def test_junction_events_and_their_matching():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    p_passed = np.array([0, .8, .9, 0, .6, 0, .8, 0, 0, 0])
    p_in = np.array([0, .9, .9, .1, 0, 0, .9, .9, 0, 0])
    # A rising edge of "passed"; a second one within 3 s is the same intersection said twice.
    assert metrics.junction_events(p_passed, p_in, .5) == [1, 4]
    assert metrics.junction_events(p_passed, p_in, .7) == [1, 6]
    # The exit decode: the first second below the threshold after one at or above it.
    assert metrics.junction_events(p_passed, p_in, .5, 'exit') == [3, 8]
    with pytest.raises(ValueError):
        metrics.junction_events(p_passed, p_in, .5, 'enter')
    # One to one, the closest pairs first.
    assert metrics.match_events([10.0, 20.0], [11.0, 25.0], 2.0) == [(0, 0)]
    assert sorted(metrics.match_events([10.0, 20.0], [11.0, 25.0], 5.0)) == [(0, 0), (1, 1)]
    assert metrics.match_events([10.0, 10.5], [10.4], 2.0) == [(1, 0)]


def test_junction_metrics_count_false_passed_per_hour_and_per_10_km():
    import numpy as np

    from nnslr_tools import road_type_metrics as metrics
    # 360 s at 10 m/s (0.1 h, 3.6 km), town then rural; passed at 50 and 250.
    count = 360
    labels = np.array([TOWN] * 180 + [RURAL] * 180)
    passed = np.zeros(count, bool)
    passed[[50, 250]] = True
    p_passed = np.zeros(count)
    p_passed[51] = .9                 # found 1 s late
    p_passed[120] = p_passed[300] = .6  # false: one in town, one outside
    p_passed[330] = .9                # on a frame not judged (after a passage whose exit was not seen)
    ignore = np.zeros(count, bool)
    ignore[325:335] = True
    sequence = {'seconds': np.arange(count, dtype=float), 'p_in': np.zeros(count), 'p_passed': p_passed,
                'in': np.zeros(count, bool), 'passed': passed, 'known': np.ones(count, bool), 'ignore': ignore,
                'labels': labels, 'predicted': labels, 'speed': np.full(count, 10.0)}
    found = metrics.junction_metrics([sequence], ROAD_TYPES)
    assert (found['passed'], found['hours'], found['km']) == (2, 0.1, 3.6)
    at_half = found['decodes']['passed']['0.5']
    assert (at_half['events'], at_half['recall_2s'], at_half['recall_4s'], at_half['precision_2s']) == (3, .5, .5, .3333)
    assert at_half['false'] == 2 and at_half['false_per_hour'] == 20.0 and at_half['false_per_10km'] == 5.56
    assert at_half['town'] == {'false': 1, 'per_hour': 20.0, 'per_10km': 5.56}
    assert at_half['outside'] == {'false': 1, 'per_hour': 20.0, 'per_10km': 5.56}
    strict = found['decodes']['passed']['0.7']
    assert (strict['events'], strict['false'], strict['f1_2s']) == (1, 0, .6667)
    # No exit events at all: F1 0, not unknown.
    assert found['decodes']['exit']['0.5']['f1_2s'] == 0.0
    # The car's point, chosen on validation: the best F1 at 2 s.
    chosen = metrics.operating_point(found)
    assert (chosen['decode'], chosen['threshold'], chosen['false_per_hour']) == ('passed', .7, 0.0)


def test_route_frames_carry_the_junction_labels_with_a_persons_corrections(root):
    from nnslr_tools import road_type_data as data
    from nnslr_tools.junctions import VERSION, correct, junctions_path
    route = OTHER_ROUTES[0]
    # Without nnslr junctions: no junction label is known.
    assert not any(frame['junction_known'] for frame in data.route_frames(root, route))
    frames = [{'frame_key': f'{s}/{i}', 'segment_index': s, 'output_index': i, 'known': True, 'in_junction': False,
               'passed': False, 'junction': None, 'distance_m': float(20 * (60 * s + i))}
              for s in range(2) for i in range(60)]
    for i in (10, 11):
        frames[i].update(in_junction=True, junction='J1')
    frames[12].update(passed=True, junction='J1')
    frames[70].update(in_junction=True, junction='J2')
    frames[90]['known'] = False
    passages = [{'id': 'J1', 'kind': 'crossing', 't_in': 9.6, 't_out': 11.4, 'passed_key': '0/12', 'closest_m': 1.0,
                 'frames_inside': 2},
                # The GPS stops in this one: no "passed" labelled, so none judged up to 4 s after it.
                {'id': 'J2', 'kind': 'crossing', 't_in': 69.6, 't_out': 70.4, 'passed_key': None, 'closest_m': 1.0,
                 'frames_inside': 1}]
    path = junctions_path(root, route)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'route_id': route, 'version': VERSION, 'rules': {}, 'passages': passages,
                                'rejected': [], 'frames': frames}))
    found = data.route_frames(root, route)
    assert [i for i, f in enumerate(found) if f['junction_in']] == [10, 11, 70]
    assert [i for i, f in enumerate(found) if f['junction_passed']] == [12]
    assert [i for i, f in enumerate(found) if f['junction_ignore']] == list(range(70, 75))
    assert found[12]['junction'] == 'J1' and found[12]['distance_m'] == 240.0
    assert found[89]['junction_known'] and not found[90]['junction_known']
    # "Not a junction" on the review page: its frames are no junction for the training either.
    correct(root, route, '0/11', 'none', 'admin')
    found = data.route_frames(root, route)
    assert [i for i, f in enumerate(found) if f['junction_in'] or f['junction_passed']] == [70]


def test_onnx_step_contract_with_the_junction_head(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False, signs=True, junctions=True).eval()
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru', 'signs': True,
                'junctions': True, 'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    (tmp_path / 'result.json').write_text(json.dumps({'junction_operating': {'decode': 'exit', 'threshold': 0.3}}))
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['verification']['passed'] is True and contract['step']['junctions'] is True
    # The road type's outputs keep their names and order; the junction comes after them.
    assert [o['name'] for o in contract['outputs']] == ['probabilities', 'hidden_out', 'junction']
    junction = contract['outputs'][2]
    assert junction['names'] == ['in_junction', 'passed'] and junction['operating'] == {'decode': 'exit',
                                                                                       'threshold': 0.3}
    assert 'in_junction < 0.3' in junction['decode']
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    probabilities, hidden_out, crossing = session.run(None, {
        'image': image, 'speed': np.array([[0.5, 1.0]], np.float32), 'sign': np.array([[2.0, 0.0]], np.float32),
        'hidden': np.zeros((1, net.HIDDEN), np.float32)})
    assert probabilities.shape == (1, len(data.CLASSES)) and abs(float(probabilities.sum()) - 1) < 1e-5
    assert crossing.shape == (1, 2) and ((crossing >= 0) & (crossing <= 1)).all()


# -- the car's own motion (nnslr_tools.car_motion) and the junction head's own memory ---------------------------------
def test_onnx_step_contract_with_the_cars_motion_and_a_junction_memory_of_its_own(root, tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('torchvision')
    import numpy as np

    from nnslr_tools import car_motion
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    model = net.build_road_type_net('gru', pretrained=False, signs=True, junctions=True, motion=True).eval()
    model.separate_junctions()
    # The junction stage trains its copy of the road GRU on its own (train_junction_stage): here a nudge.
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in model.junction_gru.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    assert net.step_input_names(model) == ['image', 'speed', 'sign', 'motion', 'hidden', 'junction_hidden']
    assert net.step_output_names(model) == ['probabilities', 'hidden_out', 'junction', 'junction_hidden_out']
    checkpoint = tmp_path / 'road-type-best.pt'
    torch.save({'task': 'road_type', 'architecture': 'mobilenet_v3_small+gru', 'memory': 'gru', 'signs': True,
                'junctions': True, 'motion': True, 'junction_branch': 'separate', 'speed_source': 'v_ego',
                'classes': list(data.CLASSES), 'state_dict': model.state_dict(), 'epoch': 1,
                'split_sha256': 's', 'dataset_sha256': 'd'}, checkpoint)
    result = net.export_road_type_onnx(checkpoint, tmp_path / 'road-type.onnx', data_root=root)
    contract = json.loads(Path(result['contract']).read_text())
    assert contract['verification']['passed'] is True
    # Without the drives' logs the rollout runs with the motion unknown.
    assert contract['verification']['rollout']['checked'] is True
    assert contract['verification']['rollout']['motion_seconds'] == 0
    assert [i['name'] for i in contract['inputs']] == ['image', 'speed', 'sign', 'motion', 'hidden', 'junction_hidden']
    motion = contract['inputs'][3]
    assert motion['shape'] == [1, car_motion.MOTION_INPUTS] and motion['names'] == list(car_motion.MOTION_FIELDS)
    assert [o['name'] for o in contract['outputs']] == ['probabilities', 'hidden_out', 'junction',
                                                        'junction_hidden_out']
    step = contract['step']
    assert (step['motion'], step['junction_branch'], step['speed_source']) == (True, 'separate', 'v_ego')
    cereal = contract['car_input']['motion']['cereal']
    assert 'livePose.angularVelocityDevice.z' in cereal['yaw'] and 'carState.yawRate' in cereal['yaw']
    assert (cereal['steer'], cereal['speed']) == ('carState.steeringAngleDeg', 'carState.vEgo')
    assert 'map' in contract['car_input']['motion']['never']
    assert any('junction_hidden_out' in line for line in contract['car_input']['steps'])
    import onnxruntime
    session = onnxruntime.InferenceSession(result['onnx'], providers=['CPUExecutionProvider'])
    image = np.random.default_rng(0).random((1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), dtype=np.float32)
    turning = np.array([car_motion.motion_features({'yaw_dps': 25.0, 'yaw_max_dps': 30.0, 'steer_deg': 200.0,
                                                    'car_known': True, 'left': True, 'right': False})], np.float32)
    zeros = np.zeros((1, net.HIDDEN), np.float32)
    probabilities, hidden_out, crossing, junction_hidden = session.run(None, {
        'image': image, 'speed': np.array([[0.3, 1.0]], np.float32), 'sign': np.array([[0.0, 0.0]], np.float32),
        'motion': turning, 'hidden': zeros, 'junction_hidden': zeros})
    assert abs(float(probabilities.sum()) - 1) < 1e-5 and crossing.shape == (1, 2)
    # The junction memory is its own GRU's: it differs from the road type's.
    assert junction_hidden.shape == (1, net.HIDDEN) and not np.allclose(junction_hidden, hidden_out)


def test_train_windows_carry_the_motion_and_drop_it_now_and_then(root):
    import random

    import numpy as np

    from nnslr_tools import car_motion
    from nnslr_tools import road_type_data as data
    from nnslr_tools import road_type_net as net
    dataset = data.load_dataset(root)
    net.attach_signs(dataset, root)
    coverage = net.attach_motion(dataset, root)
    # No drive's logs were read: the motion unknown, the speed from the GPS where vEgo is missing.
    assert coverage['speed_source'] == 'v_ego'
    runs = [run for run in dataset['sequences'] if run['partition'] == 'train']
    # Every second turning left at 30 deg/s, the wheel at 90 deg, the left blinker on.
    turning = np.array(car_motion.motion_features({'yaw_dps': 30.0, 'yaw_max_dps': 30.0, 'steer_deg': 90.0,
                                                   'car_known': True, 'left': True, 'right': False}), np.float32)
    sampler = net.WindowSampler(dataset, runs, random.Random(3))
    for _, arrays in sampler.runs:
        arrays['motion'][:] = turning
    kinds = {'kept': 0, 'none': 0, 'no_yaw': 0, 'no_car': 0}
    for _ in range(400):
        window = sampler.window(20)
        assert len(window) == 11
        motion = window[9]
        assert motion.shape == (20, car_motion.MOTION_INPUTS)
        if (motion == turning).all():
            kinds['kept'] += 1
        elif not motion.any():
            kinds['none'] += 1
        elif not motion[:, 0:3].any():
            kinds['no_yaw'] += 1
        else:
            assert not motion[:, 3:].any() and (motion[:, 0:3] == turning[0:3]).all()
            kinds['no_car'] += 1
    assert 0.1 < kinds['none'] / 400 < 0.3
    assert kinds['no_yaw'] and kinds['no_car'] and kinds['kept'] > 250


def test_the_road_types_measure_of_an_epoch_leaves_the_junction_head_out():
    from nnslr_tools import road_type_net as net
    metrics = {'weight_1': {'macro_f1': 0.6}, 'after_merge': {'accuracy_debounced': 0.9},
               'false_switches_per_hour_debounced': 40.0}
    assert net.road_score(metrics) == round(0.6 + 0.1 * 0.9 - 0.0005 * 40, 4)
    worse_merge = {**metrics, 'after_merge': {'accuracy_debounced': 0.5}}
    assert net.road_score(worse_merge) < net.road_score(metrics)
    assert net.road_score({**metrics, 'junction': {'f1_2s': 0.9}}) == net.road_score(metrics)
    assert net.road_score({'weight_1': {'macro_f1': None}}) == 0.0


def test_the_test_with_the_motion_unknown_reads_side_by_side():
    from nnslr_tools import road_type_net as net

    def part(macro, passed_recall, false, town):
        numbers = {'recall_2s': passed_recall, 'false_per_hour': false, 'town': {'per_hour': town}}
        return {'weight_1': {'macro_f1': macro}, 'after_merge': {'accuracy_debounced': 0.96},
                'false_switches_per_hour_debounced': 42.0,
                'junctions': {'decodes': {'passed': {'0.5': numbers}}}}
    result = {'test': part(0.664, 0.229, 20.7, 47.0), 'test_motion_unknown': part(0.661, 0.217, 20.1, 45.0),
              'junction_operating': {'decode': 'passed', 'threshold': 0.5}}
    text = net.motion_unknown_text(result)
    assert 'macro F1 0.664 -> 0.661' in text
    assert 'junctions (passed >= 0.5) recall 2 s 0.229 -> 0.217' in text and '(town 47.000 -> 45.000)' in text
    assert net.motion_unknown_text({'test': result['test']}) == ''


def test_a_30_zone_is_in_town():
    from nnslr_tools import road_type_data as data
    from nnslr_tools.road_type_metrics import junction_area, transition_kind
    classes = data.CLASSES
    town, zone, rural = (classes.index(name) for name in ('town', 'zone30', 'rural'))
    # Into a 30 zone from the town is no town entry; out of it into the country is the town's exit.
    assert transition_kind(town, zone, classes) == 'other'
    assert transition_kind(rural, zone, classes) == 'town_entry'
    assert transition_kind(zone, rural, classes) == 'town_exit'
    assert junction_area(zone, -1, {town, zone}) == 'town'
    assert junction_area(-1, rural, {town, zone}) == 'outside'
