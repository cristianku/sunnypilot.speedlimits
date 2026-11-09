"""The road labeler without torch: its inputs, its answers and Prepare Speed review (the base layer it writes), on
synthetic drives."""
from __future__ import annotations

import json

import numpy as np
import pytest

from nnslr_tools import road_labeler, road_labeler_data as data, road_types
from nnslr_tools.road_types import ROAD_TYPES
from nnslr_tools.speed_review import NS, SpeedReviewStore

ROUTE = '00000001--0123456789'
T0 = 1_000 * NS
END = T0 + 600 * NS
SPAN = (T0, END)
AT = {name: index for index, name in enumerate(data.INPUT_NAMES)}


def at(seconds: float) -> int:
    return T0 + int(seconds * NS)


def steps_of(keys: list[str]) -> list[dict]:
    return [{'key': key, 'segment': 0, 'decoded': 20 * n, 'step': 0, 'frame': n, 'ns': at(n), 'image': None}
            for n, key in enumerate(keys)]


@pytest.fixture
def store(tmp_path):
    store = SpeedReviewStore(tmp_path)
    if 'tunnel' not in {row['kind'] for row in store.road_kinds()}:
        store.add_road_kind('tunnel', 'test')
    return store


def tunnel_of(store) -> str:
    return next(row['kind'] for row in store.road_kinds() if row['kind'] == 'tunnel')


# ---------- inputs ----------

def test_the_map_scene_is_one_hot_with_its_tunnel_and_maxspeed():
    line = {'road_type': 'motorway', 'highway': 'motorway', 'maxspeed': '120', 'basis': 'filled', 'sure': True,
            'way_id': 7}
    found = data.map_vector(line, {7})
    assert found[AT['map_type:motorway']] == 1 and found[AT['highway:motorway']] == 1
    assert found[AT['maxspeed']] == pytest.approx(1.2) and found[AT['maxspeed_known']] == 1
    assert found[AT['map_sure']] == 1 and found[AT['map_tunnel']] == 1 and found[AT['map_line']] == 1
    assert found[AT['basis:filled']] == 1
    # No line: "none" only, so a dropped map and a missing one look the same.
    empty = data.map_vector(None, set())
    assert empty.sum() == 2 and empty[AT['map_type:none']] == 1 and empty[AT['highway:none']] == 1
    assert data.map_vector({'road_type': 'town', 'maxspeed': '30 mph'}, set())[AT['maxspeed']] == \
        pytest.approx(30 * data.MPH / 100, rel=1e-5)


def test_the_gps_says_a_fix_and_how_far_the_fixes_are_both_ways():
    gps = [(at(0), 47.0, 8.0), (at(120), 47.1, 8.1)]
    found = data.gps_inputs(np.array([at(0.5), at(30), -1]), gps)
    assert found[0].tolist() == pytest.approx([1.0, 0.5 / 60, 1.0])
    assert found[1].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert found[2].tolist() == [0.0, 1.0, 1.0]


def test_the_speed_is_the_last_one_known():
    found = data.speed_inputs(np.array([at(-1), at(1.5), at(9)]), [(at(0), 50.0), (at(2), 80.0)])
    np.testing.assert_allclose(found, [[0, 0], [0.5, 1], [0.8, 1]])


# ---------- answers ----------

def test_a_person_s_road_type_is_strong_the_map_weak_and_cant_tell_masked():
    lines = [{'frame_key': f'0/{n}', 'road_type': kind, 'sure': sure}
             for n, (kind, sure) in enumerate([('town', True), ('town', False), ('rural', False), ('town', True),
                                               ('rural', True)])]
    corrections = [{'from': '0/2', 'to': '0/2', 'road_type': 'motorway', 'reviewer': 'cristian'},
                   {'from': '0/3', 'to': '0/3', 'road_type': None, 'reviewer': 'cristian'},
                   {'from': '0/4', 'to': '0/4', 'road_type': 'garage', 'reviewer': data.NET_PREFIX + 'run'}]
    found = data.type_answers(lines, corrections, steps_of([f'0/{n}' for n in range(5)] + ['0/9']))
    assert found['type_weight'].tolist() == pytest.approx([0.3, 0.1, 1.0, 0.0, 0.3, 0.0])
    assert found['type'].tolist() == [0, 0, ROAD_TYPES.index('motorway'), -1, 1, -1]
    assert found['type_person'].tolist() == [False, False, True, False, False, False]
    # The map's own type stays for the measure; the labeler's own correction is no answer, the map's line is.
    assert found['map_type'].tolist() == [0, 0, 1, 0, 1, -1]


def test_marks_answer_inside_and_outside_only_before_checked_and_the_labeler_s_own_never_teach_it():
    ns = np.array([at(s) for s in range(0, 100, 5)])
    labels = {'road': [{'kind': 'tunnel', 'start_ns': at(10), 'end_ns': at(20), 'reviewer': 'osm'},
                       {'kind': 'tunnel', 'start_ns': at(60), 'end_ns': at(70), 'reviewer': 'net/run'},
                       {'kind': 'tunnel', 'start_ns': at(30), 'end_ns': at(40), 'reviewer': 'net/run'}],
              'road_checked': {'tunnel': at(50)}, 'works': [], 'checked_ns': None}
    found = data.drive_answers(ns, labels, ('tunnel',))
    seconds = {int((t - T0) / NS): (found['kinds'][i, 0], found['kinds_weight'][i, 0]) for i, t in enumerate(ns)}
    assert seconds[15] == (1, 1)            # osm's mark is an answer
    # The labeler's own marks are the brain's answers, never its own: masked, kept by a person (before "checked to
    # here") or not, with EDGE_SECONDS around them, so the check past them does not teach it "outside" there either.
    assert seconds[35][1] == 0 and seconds[30][1] == 0 and seconds[65][1] == 0
    assert seconds[45] == (0, 1)            # outside, before the check: known
    assert seconds[25] == (0, 1)
    assert seconds[80][1] == 0              # outside, past the check: unknown
    assert found['works_weight'].sum() == 0


def test_a_kind_with_one_side_alone_leaves_the_loss_of_train():
    drive = {'partitions': ['train'] * 4 + ['validation'], 'kind_names': ['tunnel', 'narrow'],
             'kinds': np.array([[1, 1], [1, 0], [1, 0], [1, 1], [1, 0]], dtype=np.float32),
             'kinds_weight': np.ones((5, 2), dtype=np.float32), 'works': np.zeros(5, dtype=np.float32),
             'works_weight': np.zeros(5, dtype=np.float32)}
    sides = data.keep_both_sides([drive])
    assert sides['tunnel'] == {'inside': 4, 'outside': 0} and sides['narrow'] == {'inside': 2, 'outside': 2}
    assert drive['kinds_weight'][:, 0].tolist() == [0, 0, 0, 0, 1]   # validation still measures it
    assert drive['kinds_weight'][:, 1].tolist() == [1, 1, 1, 1, 1]


# ---------- windows and stretches ----------

def test_training_windows_stay_inside_one_stretch_of_train():
    partitions = ['train'] * 25 + ['test'] * 5 + ['train'] * 12
    pieces = road_labeler.train_pieces(partitions, 10)
    assert pieces == [(0, 10), (10, 20), (20, 25), (30, 40), (40, 42)]
    assert all(all(partitions[i] == 'train' for i in range(a, b))
               for a, b in road_labeler.train_pieces(partitions, 10, offset=3))


def test_prediction_windows_keep_each_step_once_with_its_margin():
    found = road_labeler.windows(1000, 200, 50)
    kept = [i for _, _, a, b in found for i in range(a, b)]
    assert kept == list(range(1000))
    for first, last, a, b in found:
        assert a - first == min(50, a) and last - b == min(50, 1000 - b) and last - first <= 200
    assert road_labeler.windows(150, 200, 50) == [(0, 150, 0, 150)]


def test_sure_stretches_join_close_ones_and_leave_short_ones_out():
    ns = np.array([at(s / 2) for s in range(60)])
    chance = np.zeros(60)
    chance[4:14] = 0.95     # 2 to 7 s
    chance[18:24] = 0.95    # 9 to 12 s: 2 s after, joined
    chance[40:42] = 0.99    # 1 s alone: too short
    assert road_labeler.sure_stretches(ns, chance) == [(at(2), at(12))]
    assert road_labeler.sure_stretches(ns, chance, join_s=1.0) == [(at(2), at(7)), (at(9), at(12))]


def test_a_hole_in_time_ends_a_sure_stretch():
    # 10 s sure, then a minute without times (no encode index), then 10 s sure again: two marks, never one over the hole
    ns = np.array([at(s / 2) for s in range(20)] + [-1] * 120 + [at(70 + s / 2) for s in range(20)])
    chance = np.full(len(ns), 0.95)
    assert road_labeler.sure_stretches(ns, chance) == [(at(0), at(10)), (at(70), at(80))]
    # the same with the steps of the hole missing altogether (a slow-only minute keeps few frames)
    ns = np.array([at(s / 2) for s in range(20)] + [at(70 + s / 2) for s in range(20)])
    assert road_labeler.sure_stretches(ns, np.full(len(ns), 0.95)) == [(at(0), at(10)), (at(70), at(80))]


# ---------- Prepare Speed review: the base layer ----------

def prepare(store, stretches, run='run', earned=None, types=None, types_earned=()):
    """road_labeler.prepare_drive on ROUTE with these sure stretches ({kind, None for the roadworks: [(start, end)]}),
    every kind of them earned unless earned says which."""
    earned = set(stretches) if earned is None else set(earned)
    return road_labeler.prepare_drive(store, store.root, ROUTE, run, SPAN, stretches, earned, types or {},
                                      list(types_earned))


def marks_of(store) -> list[tuple]:
    return [(row['kind'], (row['start_ns'] - T0) // NS, row['reviewer']) for row in store.road(ROUTE)] + \
        [(None, (row['start_ns'] - T0) // NS, row['reviewer']) for row in store.works(ROUTE)]


def test_marks_are_written_where_nobody_said_anything(store):
    kind = tunnel_of(store)
    result = prepare(store, {kind: [(at(10), at(20))]})
    assert result == {kind: {'written': 1, 'removed': 0, 'kept': 0, 'left': 0}, 'road_types': 0}
    assert [(row['start_ns'], row['end_ns'], row['reviewer']) for row in store.road(ROUTE)] == \
        [(at(10), at(20), 'net/run')]


def test_marks_never_go_over_a_person_s_mark_removal_or_check(store):
    kind = tunnel_of(store)
    store.tap_road(ROUTE, kind, 'start', at(10), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'end', at(20), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'start', at(100), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'end', at(110), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'remove', at(105), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'checked', at(200), END, store.revision(ROUTE), 'cristian')
    found = [(at(15), at(25)), (at(102), at(108)), (at(150), at(160)), (at(300), at(301)), (at(400), at(410))]
    # A clash, a removal, before the check, 1 s (too short): left; only the last is written.
    assert prepare(store, {kind: found})[kind] == {'written': 1, 'removed': 0, 'kept': 0, 'left': 4}
    assert [(row['start_ns'], row['reviewer']) for row in store.road(ROUTE)] == [(at(10), 'cristian'),
                                                                                (at(400), 'net/run')]


def test_the_labeler_s_own_removal_does_not_count_as_a_person_s(store):
    kind = tunnel_of(store)
    prepare(store, {kind: [(at(10), at(20))]}, run='a')
    store.tap_road(ROUTE, kind, 'remove', at(15), END, store.revision(ROUTE), 'net/a')
    assert prepare(store, {kind: [(at(10), at(20))]}, run='b')[kind]['written'] == 1


def test_roadworks_are_written_with_the_same_rules(store):
    store.tap_works(ROUTE, 'checked', at(50), END, store.revision(ROUTE), 'cristian')
    result = prepare(store, {None: [(at(10), at(20)), (at(60), at(70))]})
    assert result['works'] == {'written': 1, 'removed': 0, 'kept': 0, 'left': 1}
    assert [(row['start_ns'], row['end_ns']) for row in store.works(ROUTE)] == [(at(60), at(70))]


def test_a_newer_run_replaces_the_older_one_s_marks_that_no_person_kept(store):
    kind = tunnel_of(store)
    prepare(store, {kind: [(at(10), at(20)), (at(100), at(110))], None: [(at(300), at(310))]}, run='a')
    store.tap_road(ROUTE, kind, 'checked', at(50), END, store.revision(ROUTE), 'cristian')   # keeps the mark at 10
    store.tap_road(ROUTE, kind, 'start', at(200), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'end', at(210), END, store.revision(ROUTE), 'cristian')
    store.tap_road(ROUTE, kind, 'start', at(400), END, store.revision(ROUTE), 'osm')
    store.tap_road(ROUTE, kind, 'end', at(410), END, store.revision(ROUTE), 'osm')
    # Run b: the tunnel earned, the roadworks held back; the older run's roadworks go all the same.
    result = prepare(store, {kind: [(at(500), at(510))], None: [(at(300), at(310))]}, run='b', earned={kind})
    assert result[kind] == {'written': 1, 'removed': 1, 'kept': 1, 'left': 0}
    assert result['works'] == {'written': 0, 'removed': 1, 'kept': 0, 'left': 0}
    assert marks_of(store) == [(kind, 10, 'net/a'), (kind, 200, 'cristian'), (kind, 400, 'osm'), (kind, 500, 'net/b')]
    # The removals are the labeler's own undo, never a person's word: run a's stretch at 100 can come back.
    assert road_types.person_removals(store.history(ROUTE)) == []
    assert prepare(store, {kind: [(at(100), at(110))]}, run='c')[kind]['written'] == 1


def test_an_older_run_s_mark_goes_out_never_a_person_s_open_mark_over_it(store):
    kind = tunnel_of(store)
    prepare(store, {kind: [(at(100), at(120))], None: [(at(100), at(120))]}, run='a')
    # A person starts a tunnel and roadworks at 50 s with no end yet: open, they cover the labeler's marks at 100 s.
    store.tap_road(ROUTE, kind, 'start', at(50), END, store.revision(ROUTE), 'cristian')
    store.tap_works(ROUTE, 'start', at(50), END, store.revision(ROUTE), 'cristian')
    result = prepare(store, {}, run='b', earned=set())
    assert result[kind]['removed'] == 1 and result['works']['removed'] == 1
    assert marks_of(store) == [(kind, 50, 'cristian'), (None, 50, 'cristian')]
    assert [(row['action'], (row['start_ns'] - T0) // NS) for row in store.history(ROUTE)
            if row['action'].endswith('_removed')] == [('road_removed', 100), ('works_removed', 100)]


def test_the_same_run_again_changes_nothing(store):
    kind = tunnel_of(store)
    lines = map_lines(store.root, ['town'] * 10)
    types = road_labeler.layer_types(lines, chances_for(['rural'] * 10), ['rural'])
    first = prepare(store, {kind: [(at(10), at(20))]}, types=types, types_earned=['rural'])
    assert first[kind]['written'] == 1 and first['road_types'] == 10
    revision, layer = store.revision(ROUTE), road_types.labeler_types_path(store.root, ROUTE).read_bytes()
    again = prepare(store, {kind: [(at(10), at(20))]}, types=types, types_earned=['rural'])
    assert again[kind] == {'written': 0, 'removed': 0, 'kept': 0, 'left': 0}
    assert store.revision(ROUTE) == revision and marks_of(store) == [(kind, 10, 'net/run')]
    assert road_types.labeler_types_path(store.root, ROUTE).read_bytes() == layer


def test_a_prepare_is_one_write_in_the_history_with_its_road_types(store, monkeypatch):
    kind = tunnel_of(store)
    lines = map_lines(store.root, ['town'] * 10)
    prepare(store, {kind: [(at(10), at(20))]}, run='a')
    base = store.revision(ROUTE)
    types = road_labeler.layer_types(lines, chances_for(['rural'] * 10), ['rural'])
    prepare(store, {kind: [(at(30), at(40))]}, run='b', types=types, types_earned=['rural'])
    rows = [row for row in store.history(ROUTE) if row['revision'] > base]
    assert [(row['action'], row['reviewer']) for row in rows] == [
        ('road_removed', 'net/b'), ('road_start', 'net/b'), ('road_end', 'net/b'), ('types_labeled', 'net/b')]
    assert [row['revision'] for row in rows] == list(range(base + 1, base + 5))
    assert (rows[-1]['state'], rows[-1]['note']) == ('b', '10 frames of rural')
    # A layer that cannot be written takes the marks back with it: nothing of run c.
    def broken(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(road_types, 'write_labeler_types', broken)
    with pytest.raises(OSError):
        prepare(store, {kind: [(at(50), at(60))]}, run='c', types=types, types_earned=['rural'])
    assert marks_of(store) == [(kind, 30, 'net/b')] and store.revision(ROUTE) == base + 4
    assert road_types.load_labeler_types(store.root, ROUTE)[0] == 'b'


def map_lines(root, kinds: list[str]) -> list[dict]:
    lines = [{'frame_key': f'0/{n}', 'road_type': kind, 'sure': False} for n, kind in enumerate(kinds)]
    path = road_types.road_types_path(root, ROUTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines), encoding='utf-8')
    return lines


def chances_for(kinds: list[str | None], p: float = 0.95) -> dict[str, np.ndarray]:
    found = {}
    for n, kind in enumerate(kinds):
        if kind is not None:
            row = np.full(len(ROAD_TYPES), (1 - p) / (len(ROAD_TYPES) - 1))
            row[ROAD_TYPES.index(kind)] = p
            found[f'0/{n}'] = row
    return found


def test_the_layer_of_road_types_takes_sure_runs_of_earned_types_also_where_the_map_agrees(store):
    lines = map_lines(store.root, ['town'] * 20)
    said = chances_for(['rural'] * 7 + ['town'] * 6 + ['motorway'] * 5 + ['rural'] * 2)
    found = road_labeler.layer_types(lines, said, ['rural', 'town'])
    # 0-6 rural, 7-12 town as the map says: both written; motorway not earned; 18-19 too short.
    assert {key: kind for key, (kind, _p) in found.items()} == {
        **{f'0/{n}': 'rural' for n in range(7)}, **{f'0/{n}': 'town' for n in range(7, 13)}}
    assert road_labeler.layer_types(lines, chances_for(['rural'] * 20, p=0.6), ['rural']) == {}
    assert len(road_labeler.layer_types(lines, chances_for(['rural'] * 4 + [None] + ['rural'] * 4), ['rural'])) == 0
    # Written, the layer is over the map and under a person, each line with its source.
    road_types.correct(store.root, ROUTE, '0/10', '0/19', 'motorway', 'cristian')
    prepare(store, {}, types=found, types_earned=['rural', 'town'])
    shown = road_types.load_road_types(store.root, ROUTE)
    assert [(line['road_type'], road_types.source_of(line)) for line in shown[:1] + shown[7:8] + shown[10:11] + shown[13:14]] \
        == [('rural', 'net/run'), ('town', 'net/run'), ('motorway', 'cristian'), ('motorway', 'cristian')]
    assert [line['road_type'] for line in road_types.load_road_types(store.root, ROUTE, labeler=False)[:7]] == ['town'] * 7
    # A newer run without a type earned takes the layer away: the map shows through.
    prepare(store, {}, run='b')
    assert road_types.load_road_types(store.root, ROUTE)[0]['road_type'] == 'town'
    assert road_types.source_of(road_types.load_road_types(store.root, ROUTE)[0]) == road_types.MAP_SOURCE


def test_only_heads_trained_with_both_sides_and_through_the_gate_write():
    sides = {'tunnel': {'inside': 1960, 'outside': 0}, 'narrow': {'inside': 3, 'outside': 9},
             'works': {'inside': 0, 'outside': 0}}
    passed = {'why': None}
    earned = {'narrow': passed, 'tunnel': passed, 'works': passed, 'type:rural': passed,
              **{f'type:{name}': {'why': 'too few'} for name in ROAD_TYPES if name != 'rural'}}
    checkpoint = {'kinds': ['narrow', 'tunnel'], 'trained': road_labeler.trained_heads(sides), 'earned': earned}
    # A tunnel head masked in train says chances from its first weights: never a mark, even at chance 1.
    assert road_labeler.writable(checkpoint) == (['narrow'], False, ['rural'])
    held = road_labeler.held_back(checkpoint)
    assert held['tunnel'] == held['works'] == 'not trained with both sides' and held['type:town'] == 'too few'
    # A head that failed its gate writes nothing, and a checkpoint made before the gate writes nothing at all.
    assert road_labeler.writable({**checkpoint, 'earned': {**earned, 'narrow': {'why': 'right on 50%'}}})[0] == []
    assert road_labeler.writable({'kinds': ['narrow'], 'trained': ['narrow', 'works']}) == ([], False, [])
    assert road_labeler.held_back({'kinds': ['narrow'], 'trained': ['narrow']})['narrow'] == \
        'no gate of validation in the checkpoint'


def gate_drive(length: int, inside: int, said_right: int, said_wrong: int) -> tuple[dict, dict]:
    """A drive all of validation with one kind: the first `inside` steps inside, the network sure on said_right of them
    and on said_wrong steps outside."""
    truth = (np.arange(length) < inside).astype(np.float32)
    chance = np.zeros(length, dtype=np.float32)
    chance[:said_right] = 0.95
    chance[inside:inside + said_wrong] = 0.95
    drive = {'partitions': ['validation'] * length, 'kind_names': ['tunnel'], 'kinds': truth[:, None],
             'kinds_weight': np.ones((length, 1), dtype=np.float32), 'works': np.zeros(length, dtype=np.float32),
             'works_weight': np.zeros(length, dtype=np.float32), 'type': np.zeros(length, dtype=np.int64),
             'type_weight': np.ones(length, dtype=np.float32), 'type_person': np.ones(length, dtype=bool),
             'map_type': np.ones(length, dtype=np.int64)}
    types = np.full((length, len(ROAD_TYPES)), 0.01, dtype=np.float32)
    types[:, 0] = 0.95      # town everywhere, the map says rural: a write would go everywhere, and it is right
    return drive, {'kinds': chance[:, None], 'works': np.zeros(length, dtype=np.float32), 'types': types}


def test_a_head_writes_only_with_enough_answers_on_each_side_and_a_high_precision():
    need = road_labeler.GATE_MIN_ANSWERS
    drive, said = gate_drive(3 * need, need, road_labeler.GATE_MIN_SAID, 0)
    gate = road_labeler.earned_heads([drive], [said])
    assert gate['tunnel']['why'] is None and gate['tunnel']['precision'] == 1.0
    assert gate['tunnel']['inside'] == need and gate['tunnel']['outside'] == 2 * need
    # Too few steps inside.
    drive, said = gate_drive(3 * need, need - 1, road_labeler.GATE_MIN_SAID, 0)
    assert 'needed' in road_labeler.earned_heads([drive], [said])['tunnel']['why']
    # Sure too rarely to tell.
    drive, said = gate_drive(3 * need, need, road_labeler.GATE_MIN_SAID - 1, 0)
    assert road_labeler.earned_heads([drive], [said])['tunnel']['why'].startswith('sure on')
    # Wrong too often where it is sure.
    drive, said = gate_drive(3 * need, need, road_labeler.GATE_MIN_SAID, road_labeler.GATE_MIN_SAID)
    gate = road_labeler.earned_heads([drive], [said])
    assert gate['tunnel']['precision'] == 0.5 and gate['tunnel']['why'].startswith('right on 50%')
    # Answers of less than full weight (the map's) do not count, nor do the steps of train.
    drive, said = gate_drive(3 * need, need, road_labeler.GATE_MIN_SAID, 0)
    drive['kinds_weight'][need:] = data.TUNNEL_FREE_WEIGHT
    assert road_labeler.earned_heads([drive], [said])['tunnel']['outside'] == 0
    drive['partitions'] = ['train'] * len(drive['partitions'])
    assert road_labeler.earned_heads([drive], [said])['tunnel']['inside'] == 0
    # A road type: inside where a person set it, sure where it says it against the map; works with no answer at all.
    drive, said = gate_drive(3 * need, need, road_labeler.GATE_MIN_SAID, 0)
    drive['type'][2 * need:] = 1
    gate = road_labeler.earned_heads([drive], [said])
    assert gate['type:town']['inside'] == 2 * need and gate['type:town']['outside'] == need
    assert gate['type:town']['precision'] == pytest.approx(2 / 3, abs=1e-3) and gate['type:town']['why'].startswith('right on')
    assert gate['type:rural']['said'] == 0 and gate['works']['why'] is not None


# ---------- a start and its end are one write ----------

def refusing(store, monkeypatch, *taps):
    """The store refuses these taps (as a revision changed meanwhile)."""
    from nnslr_tools.speed_review import SpeedReviewError
    for name in ('tap_road', 'tap_works'):
        real = getattr(store, name)

        def tap(*args, real=real):
            if any(arg in taps for arg in args if isinstance(arg, str)):
                raise SpeedReviewError(409, 'stale', 'The drive changed.')
            return real(*args)

        monkeypatch.setattr(store, name, tap)


def test_a_mark_whose_end_is_refused_is_taken_back(store, monkeypatch):
    from nnslr_tools.speed_review import SpeedReviewError
    kind = tunnel_of(store)
    real = store._tap_on

    def tap_on(connection, route, revision, of, tap, *args):
        if tap == 'end':
            raise SpeedReviewError(400, 'bad_road', 'refused')
        return real(connection, route, revision, of, tap, *args)

    monkeypatch.setattr(store, '_tap_on', tap_on)
    result = prepare(store, {kind: [(at(10), at(20))], None: [(at(10), at(20))]})
    assert result[kind] == result['works'] == {'written': 0, 'removed': 0, 'kept': 0, 'left': 1}
    assert store.road(ROUTE) == [] and store.works(ROUTE) == []
    monkeypatch.undo()
    # Nothing of it was written, no start left open: the next run writes it.
    assert prepare(store, {kind: [(at(10), at(20))]})[kind]['written'] == 1


def test_a_tunnel_whose_end_is_refused_is_taken_back_and_written_next_time(store, monkeypatch):
    from nnslr_tools import tunnels
    refusing(store, monkeypatch, 'end')
    assert tunnels.write_tunnels(store, ROUTE, [(at(10), at(20))], SPAN) == {'written': 0, 'kept': 1}
    assert store.road(ROUTE) == []
    monkeypatch.undo()
    refusing(store, monkeypatch, 'end', 'remove')
    assert tunnels.write_tunnels(store, ROUTE, [(at(30), at(40))], SPAN) == {'written': 0, 'kept': 1, 'left_open': 1}
    monkeypatch.undo()
    store.tap_road(ROUTE, tunnel_of(store), 'remove', at(30), END, store.revision(ROUTE), 'osm')
    # Neither undo of its own is a person's removal.
    assert tunnels.write_tunnels(store, ROUTE, [(at(10), at(20)), (at(30), at(40))], SPAN) == {'written': 2, 'kept': 0}


def test_a_person_s_removal_still_keeps_the_tunnels_out(store):
    from nnslr_tools import tunnels
    kind = tunnel_of(store)
    assert tunnels.write_tunnels(store, ROUTE, [(at(10), at(20))], SPAN)['written'] == 1
    store.tap_road(ROUTE, kind, 'remove', at(15), END, store.revision(ROUTE), 'cristian')
    assert tunnels.write_tunnels(store, ROUTE, [(at(10), at(20))], SPAN) == {'written': 0, 'kept': 1}
    # Nor does the labeler write where a person removed the osm mark; osm's own removal would not stop it.
    assert prepare(store, {kind: [(at(12), at(18))]})[kind]['left'] == 1


# ---------- a person's removal and the tunnel's weak "no" ----------

def test_a_person_s_removal_of_a_mark_with_an_end_answers_no(store):
    kind = tunnel_of(store)
    tap = lambda name, t, who='cristian', of=kind: store.tap_road(ROUTE, of, name, at(t), END, store.revision(ROUTE), who)
    tap('start', 10), tap('end', 30), tap('remove', 20)                  # a person's removal: an answer
    tap('start', 100), tap('remove', 100)                                # removed before its end: a slip
    prepare(store, {kind: [(at(200), at(220))]})
    tap('remove', 210, 'net/run')                                        # the labeler's own undo
    prepare(store, {kind: [(at(300), at(320))]})
    tap('remove', 310)                                                   # a person removes the labeler's mark: an answer
    tap('start', 400), tap('end', 430), tap('remove', 410), tap('start', 405), tap('end', 425)   # a mark moved
    store.tap_works(ROUTE, 'start', at(500), END, store.revision(ROUTE), 'cristian')
    store.tap_works(ROUTE, 'end', at(520), END, store.revision(ROUTE), 'cristian')
    store.tap_works(ROUTE, 'remove', at(510), END, store.revision(ROUTE), 'cristian')
    removed = data.removed_marks(store.root, ROUTE)
    assert [(row['kind'], (row['start_ns'] - T0) // NS, (row['end_ns'] - T0) // NS) for row in removed] == \
        [(kind, 10, 30), (kind, 300, 320), (kind, 400, 430), (None, 500, 520)]
    ns = np.array([at(s) for s in range(0, 600)])
    labels = {'road': store.road(ROUTE), 'road_checked': {}, 'works': store.works(ROUTE), 'checked_ns': None}
    found = data.drive_answers(ns, labels, (kind,), removed)
    answer = lambda s: (found['kinds'][s, 0], found['kinds_weight'][s, 0])
    assert answer(15) == (0, 1) and answer(29) == (0, 1) and answer(310) == (0, 1)
    assert answer(150)[1] == 0 and answer(215)[1] == 0                   # the slip and the undo say nothing
    assert answer(415) == (1, 1)                                         # the mark set again is the answer
    assert answer(401) == (0, 1) and answer(428) == (0, 1)                 # the old stretch outside it: no
    assert answer(404)[1] == 0 and answer(426)[1] == 0                   # within 2 s of it: nothing
    assert found['works'][510] == 0 and found['works_weight'][510] == 1 and found['works_weight'][530] == 0
    assert data.drive_answers(ns, labels, (kind,))['kinds_weight'][15, 0] == 0


def test_the_tunnel_s_weak_no_needs_a_fix_on_a_road_of_the_map_that_is_no_tunnel():
    keys = [f'0/{n}' for n in range(60)]
    steps = steps_of(keys)
    ns = np.array([step['ns'] for step in steps])
    lines = {key: {'way_id': 1} for key in keys}
    for n in range(40, 45):
        lines[f'0/{n}'] = {'way_id': 2}     # the map's tunnel
    lines['0/20'] = {'way_id': None}        # no road of the map
    fix = np.ones(60)
    fix[25] = 0                             # no GPS fix
    marks = [{'kind': 'tunnel', 'start_ns': at(10), 'end_ns': at(12), 'reviewer': 'osm'},
             {'kind': 'tunnel', 'start_ns': at(57), 'end_ns': None, 'reviewer': 'net/run'}]
    free = data.tunnel_free(ns, steps, lines, {2}, fix, marks)
    assert free[:5].all() and not free[5:17].any() and free[17] and free[19]   # 5 s around the osm mark
    assert not free[20] and not free[25] and free[26]
    assert free[35] and not free[36:49].any() and free[49] and free[51]      # 5 s around the map's tunnel
    assert not free[52:].any()                                              # a mark with no end: to the drive's end


def test_the_tunnel_kind_is_found_by_its_kind_or_its_label(store):
    assert data.tunnel_kind(store.root) == 'tunnel'
