"""The road labeler without torch: its inputs, its answers and its writes into Speed review, on synthetic drives."""
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


def test_marks_answer_inside_and_outside_only_before_checked_and_the_labeler_s_own_are_masked():
    ns = np.array([at(s) for s in range(0, 100, 5)])
    labels = {'road': [{'kind': 'tunnel', 'start_ns': at(10), 'end_ns': at(20), 'reviewer': 'osm'},
                       {'kind': 'tunnel', 'start_ns': at(60), 'end_ns': at(70), 'reviewer': 'net/run'},
                       {'kind': 'tunnel', 'start_ns': at(30), 'end_ns': at(40), 'reviewer': 'net/run'}],
              'road_checked': {'tunnel': at(50)}, 'works': [], 'checked_ns': None}
    found = data.drive_answers(ns, labels, ('tunnel',))
    seconds = {int((t - T0) / NS): (found['kinds'][i, 0], found['kinds_weight'][i, 0]) for i, t in enumerate(ns)}
    assert seconds[15] == (1, 1)            # osm's mark is an answer
    assert seconds[35] == (1, 1)            # the labeler's own, before "checked to here": a person kept it
    assert seconds[65][1] == 0              # its own past the check: masked
    assert seconds[45] == (0, 1)            # outside, before the check: known
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


# ---------- writes into Speed review ----------

def test_marks_are_written_where_nobody_said_anything(store):
    kind = tunnel_of(store)
    result = road_labeler.write_marks(store, ROUTE, kind, [(at(10), at(20))], SPAN, 'net/run')
    assert result == {'written': 1, 'kept': 0}
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
    result = road_labeler.write_marks(store, ROUTE, kind, found, SPAN, 'net/run')
    # A clash, a removal, before the check: kept; 1 s: too short; only the last is written.
    assert result == {'written': 1, 'kept': 3}
    assert [(row['start_ns'], row['reviewer']) for row in store.road(ROUTE)] == [(at(10), 'cristian'),
                                                                                (at(400), 'net/run')]


def test_the_labeler_s_own_removal_does_not_count_as_a_person_s(store):
    kind = tunnel_of(store)
    road_labeler.write_marks(store, ROUTE, kind, [(at(10), at(20))], SPAN, 'net/a')
    store.tap_road(ROUTE, kind, 'remove', at(15), END, store.revision(ROUTE), 'net/a')
    assert road_labeler.write_marks(store, ROUTE, kind, [(at(10), at(20))], SPAN, 'net/b')['written'] == 1


def test_roadworks_are_written_with_the_same_rules(store):
    store.tap_works(ROUTE, 'checked', at(50), END, store.revision(ROUTE), 'cristian')
    result = road_labeler.write_marks(store, ROUTE, None, [(at(10), at(20)), (at(60), at(70))], SPAN, 'net/run')
    assert result == {'written': 1, 'kept': 1}
    assert [(row['start_ns'], row['end_ns']) for row in store.works(ROUTE)] == [(at(60), at(70))]


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


def test_road_types_are_written_only_where_the_map_differs_and_no_person_set_one(store):
    lines = map_lines(store.root, ['town'] * 20)
    lines = road_types.corrected(lines, [{'from': '0/10', 'to': '0/19', 'road_type': 'town', 'reviewer': 'cristian'}])
    said = chances_for(['rural'] * 7 + ['town'] * 2 + ['rural'] * 11)
    found = road_labeler.type_stretches(lines, said)
    # 0-6 differs from the map; 7-8 agrees; 9 alone is too short; 10-19 a person set.
    assert found == [('0/0', '0/6', 'rural')]
    assert road_labeler.type_stretches(lines, chances_for(['rural'] * 20, p=0.6)) == []
    ns_of = {f'0/{n}': at(n) for n in range(20)}
    assert road_labeler.write_types(store, ROUTE, found, ns_of, SPAN, 'net/run') == {'written': 1, 'kept': 0}
    kept = road_types.load_corrections(store.root, ROUTE)
    assert [(row['from'], row['to'], row['road_type'], row['reviewer']) for row in kept] == \
        [('0/0', '0/6', 'rural', 'net/run')]


def test_only_heads_trained_with_both_sides_write():
    sides = {'tunnel': {'inside': 1960, 'outside': 0}, 'narrow': {'inside': 3, 'outside': 9},
             'works': {'inside': 0, 'outside': 0}}
    checkpoint = {'kinds': ['narrow', 'tunnel'], 'trained': road_labeler.trained_heads(sides)}
    # A tunnel head masked in train says chances from its first weights: never a mark, even at chance 1.
    assert road_labeler.writable(checkpoint) == (['narrow'], False)
    assert road_labeler.writable({'kinds': ['tunnel']}) == ([], False)
