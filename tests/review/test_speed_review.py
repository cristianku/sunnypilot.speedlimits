"""Speed review (Data › Speed review, nnslr_tools.speed_review): the driver's speed as the answer, the blocks of a person
that correct it, the roadworks, and who may write them.

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools import review_mcp, speed_review  # noqa: E402
from nnslr_tools.pages import CONTROL_TABS  # noqa: E402
from nnslr_tools.review import PAGE_RE  # noqa: E402
from nnslr_tools.speed_review import NS, SpeedReviewError, SpeedReviewStore  # noqa: E402
from review_helpers import ROUTE, TOKEN, RunningServer, json_of, make_report, sign_in  # noqa: E402

T0 = 100 * NS
END = 400 * NS
SPAN = (T0, END)


class FakeDrives:
    """What the store needs of a drive review: the span of a drive, its joined logs with the car's speed in km/h."""

    def __init__(self, points=((T0, 80.0),)):
        self.points = [list(point) for point in points]

    def summary(self, route):
        return {'t0_ns': T0, 'span_ns': [T0, END]}

    def reading(self):
        return contextlib.nullcontext()

    def scan_state(self, route):
        return {'route': route, 'state': 'done', 'progress': 1, 'error': None, 'files': 1}

    def merged(self, route):
        return merged(self.points)


def merged(points, cluster=()):
    return {'t0': T0, 'span': [T0, END], 'series': {'v_ego': [list(point) for point in points],
                                                    'v_ego_cluster': [list(point) for point in cluster]}}


def at(seconds: float) -> int:
    return T0 + round(seconds * NS)


def blocks(store, route=ROUTE):
    return [((row['start_ns'] - T0) / NS, (row['end_ns'] - T0) / NS, row['kph']) for row in store.blocks(route)]


@pytest.fixture
def store(tmp_path):
    return SpeedReviewStore(tmp_path)


# ---------- the driver's speed ----------

def test_the_answer_is_the_dashboard_speed_in_whole_km_h():
    drive = merged([(at(0), 51.49), (at(1), 51.5), (at(2), 0.3), (at(3), 250.0)])
    times = [at(0) - 1, at(0), at(1.5), at(2), at(3)]
    # Half up, at most MAX_KPH, 0 below the dashboard's floor, None before the logs say a speed.
    assert speed_review.driver_speed(drive, times) == [None, 51, 52, 0, 200]


def test_the_cluster_speed_of_the_car_is_the_answer_where_the_car_has_its_own():
    # The cluster says another speed than vEgo while moving: it is real, so it is the dashboard's.
    drive = merged([(at(0), 50.0)], cluster=[(at(0), 53.4)])
    assert speed_review.driver_speed(drive, [at(5)]) == [53]


def test_the_curve_is_the_speed_at_the_brains_steps_where_it_changes():
    drive = merged([(at(0), 80.2), (at(10.1), 80.4), (at(20), 79.6), (at(30), 100.0)])
    curve = speed_review.speed_curve(drive, (at(0), at(40)))
    # 80.2, 80.4 and 79.6 are all 80: one point.
    assert curve == [[at(0), 80], [at(30), 100]]
    assert speed_review.speed_curve(merged([(at(0.1), 30.0)]), (at(0), at(1))) == [[at(0), None], [at(0.25), 30]]


def test_openpilot_or_the_cruise_held_the_speed_unless_his_foot_was_on_the_gas():
    series = {'long_active': [[at(10), True], [at(20), False]],
              'cruise_enabled': [[at(30), True], [at(60), False]], 'gas': [[at(40), True], [at(50), False]]}
    times = [at(5), at(15), at(25), at(35), at(45), at(55), at(65)]
    assert speed_review.machine_held({'series': series}, times) == [False, True, False, True, False, True, False]
    # Logs without these series: nothing is held.
    assert speed_review.machine_held({}, times[:2]) == [False, False]
    assert speed_review.machine_held(None, times[:2]) == [False, False]


def test_a_set_speed_alone_is_never_held_the_answer_is_how_he_drove():
    # The Peugeot's set speed (cruise_speed) and its cruise_available say nothing of his foot: only openpilot's
    # longitudinal or the cruise on in carState take a moment out.
    series = {'cruise_enabled': [[at(0), False]], 'cruise_available': [[at(0), True], [at(10), False]],
              'cruise_speed': [[at(0), 100.0], [at(10), 117.0]], 'gas': [[at(0), False]],
              'v_ego': [[at(0), 100.0], [at(10), 116.5]]}
    peugeot = {'series': series, 'meta': {'car': {'brand': 'psa'}}}
    assert speed_review.machine_held(peugeot, [at(5), at(15)]) == [False, False]


def test_a_dashcam_drive_is_passive_in_every_carparams_and_never_engaged():
    def summary(car, engaged=0.0):
        return {'meta': {'car': car} if car is not None else {}, 'coverage': {'engaged': engaged}}
    assert speed_review.dashcam_drive(summary({'passive': True, 'dashcam_only': False}))
    assert not speed_review.dashcam_drive(summary({'passive': False, 'dashcam_only': False}))
    assert not speed_review.dashcam_drive(summary({'passive': True}, engaged=1.5))
    # An extract before carParams.passive: dashcamOnly.
    assert speed_review.dashcam_drive(summary({'dashcam_only': True}))
    assert not speed_review.dashcam_drive(summary({'dashcam_only': False}))
    # No carParams, no summary: not one.
    assert not speed_review.dashcam_drive(summary(None)) and not speed_review.dashcam_drive(None)
    assert speed_review.engaged_series({'series': {'sd_enabled': [[0, False]], 'lat_active': [[0, None], [1, True]]}}) \
        == ['lat_active']
    assert speed_review.engaged_series(None) == []


def test_the_page_greys_where_a_machine_held_the_speed_at_the_brains_steps():
    drive = merged([(at(0), 80.0)])
    drive['series']['long_active'] = [[at(1.1), True], [at(2), False]]
    # 1.1 s falls between two steps: the first step it holds is 1.25 s, and the span ends at the step after the last.
    assert speed_review.held_spans(drive, (at(0), at(4))) == [[at(1.25), at(2)]]
    assert speed_review.held_spans(merged([(at(0), 80.0)]), (at(0), at(4))) == []


# ---------- the blocks ----------

def test_a_block_clips_the_blocks_it_covers(store):
    assert store.set_block(ROUTE, at(0), at(100), 80, '', 0, 'a') == 1
    store.set_block(ROUTE, at(40), at(60), None, 'queue', 1, 'a')
    assert blocks(store) == [(0, 40, 80), (40, 60, None), (60, 100, 80)]
    store.set_block(ROUTE, at(30), at(70), 50, '', 2, 'b')
    assert blocks(store) == [(0, 30, 80), (30, 70, 50), (70, 100, 80)]
    # What a block keeps outside the new one is still its own writer's.
    assert [row['reviewer'] for row in store.blocks(ROUTE)] == ['a', 'b', 'a']
    # A piece shorter than a block can be goes: the page could only remove it.
    store.set_block(ROUTE, at(30.5), at(99.5), 60, '', 3, 'b')
    assert blocks(store) == [(0, 30, 80), (30.5, 99.5, 60)]


def test_a_removed_block_gives_the_drivers_speed_back(store):
    store.set_block(ROUTE, at(0), at(100), 80, '', 0, 'a')
    assert store.remove_block(ROUTE, at(0), 1, 'a') == 2 and blocks(store) == []
    with pytest.raises(SpeedReviewError) as refused:
        store.remove_block(ROUTE, at(5), 2, 'a')
    assert (refused.value.status, refused.value.reason) == (400, 'bad_block') and store.revision(ROUTE) == 2


def test_a_block_dragged_to_new_ends_moves_in_one_write(store):
    store.set_block(ROUTE, at(0), at(100), 80, 'map', 0, 'a')
    store.set_block(ROUTE, at(100), at(150), 50, '', 1, 'a')
    # Shorter: nothing of it stays where it was.
    assert store.set_block(ROUTE, at(10), at(90), 80, 'map', 2, 'b', moved_from=at(0)) == 3
    assert blocks(store) == [(10, 90, 80), (100, 150, 50)]
    # Longer, onto the next one: that one keeps what lies outside it.
    store.set_block(ROUTE, at(10), at(120), 80, 'map', 3, 'b', moved_from=at(10))
    assert blocks(store) == [(10, 120, 80), (120, 150, 50)]
    row = store.history(ROUTE)[-1]
    assert (row['action'], row['known_ns'], row['start_ns'], row['end_ns']) == ('block_moved', at(10), at(10), at(120))
    # The edge two blocks share dragged back (a roll, as the page sends it): the one after grows, the one before is cut.
    store.set_block(ROUTE, at(60), at(150), 50, '', 4, 'b', moved_from=at(120))
    assert blocks(store) == [(10, 60, 80), (60, 150, 50)]
    with pytest.raises(SpeedReviewError) as refused:
        store.set_block(ROUTE, at(0), at(5), 80, '', 5, 'b', moved_from=at(3))
    assert (refused.value.status, refused.value.reason) == (400, 'bad_block') and store.revision(ROUTE) == 5
    assert blocks(store) == [(10, 60, 80), (60, 150, 50)]


def test_a_stale_revision_changes_nothing(store):
    store.set_block(ROUTE, at(0), at(100), 80, '', 0, 'a')
    with pytest.raises(SpeedReviewError) as refused:
        store.set_block(ROUTE, at(0), at(100), 120, '', 0, 'b')
    assert (refused.value.status, refused.value.reason) == (409, 'stale_revision')
    assert blocks(store) == [(0, 100, 80)] and store.revision(ROUTE) == 1
    # The revision is the drive's own: another drive starts from 0.
    assert store.set_block('00000000--0000000000', at(0), at(10), 80, '', 0, 'a') == 1


def test_the_history_only_grows_and_rebuilds_the_blocks(store, tmp_path):
    writes = [(0, 100, 80), (40, 60, None), (50, 90, 120)]
    for revision, (start, end, kph) in enumerate(writes):
        store.set_block(ROUTE, at(start), at(end), kph, 'note' if kph is None else '', revision, 'a')
    store.set_block(ROUTE, at(60), at(95), 120, '', 3, 'a', moved_from=at(50))
    store.remove_block(ROUTE, at(0), 4, 'a')
    history = store.history(ROUTE)
    assert [row['revision'] for row in history] == [1, 2, 3, 4, 5]
    assert [row['action'] for row in history] == ['block_saved', 'block_saved', 'block_saved', 'block_moved', 'block_removed']
    assert history[0]['reviewer'] == 'a' and history[0]['at'] and history[1]['note'] == 'note'
    # Played again on an empty database, the history gives the same blocks.
    replay = SpeedReviewStore(tmp_path / 'replay')
    for row in history:
        if row['action'] == 'block_removed':
            replay.remove_block(ROUTE, row['start_ns'], row['revision'] - 1, row['reviewer'])
        else:
            replay.set_block(ROUTE, row['start_ns'], row['end_ns'], row['kph'], row['note'], row['revision'] - 1, row['reviewer'],
                             row['known_ns'] if row['action'] == 'block_moved' else None)
    assert replay.blocks(ROUTE) == store.blocks(ROUTE)
    connection = sqlite3.connect(store.path)
    for sql in ('UPDATE speed_history SET kph = 1', 'DELETE FROM speed_history'):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(sql)
    connection.close()


def test_an_old_file_keeps_its_stretches_and_takes_the_blocks(tmp_path):
    path = speed_review.db_path(tmp_path)
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(path)
    connection.executescript('''CREATE TABLE schema_version (version INTEGER PRIMARY KEY); INSERT INTO schema_version VALUES (1);
        CREATE TABLE speed_stretches (route TEXT NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL, kph INTEGER,
        state TEXT NOT NULL, known_ns INTEGER, note TEXT NOT NULL DEFAULT '', reviewer TEXT NOT NULL, revision INTEGER NOT NULL,
        PRIMARY KEY (route, start_ns));''')
    connection.execute('INSERT INTO speed_stretches VALUES (?, 0, 10, 80, ?, NULL, ?, ?, 1)', (ROUTE, 'confirmed', '', 'a'))
    connection.commit()
    connection.close()
    store = SpeedReviewStore(tmp_path)
    store.set_block(ROUTE, at(0), at(10), 70, '', 0, 'a')
    connection = sqlite3.connect(path)
    assert connection.execute('SELECT COUNT(*) FROM speed_stretches').fetchone()[0] == 1
    assert connection.execute('SELECT MAX(version) FROM schema_version').fetchone()[0] == 2
    connection.close()


def test_the_labels_are_readable_only_by_their_owner(store):
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.path.name == 'speed_review.sqlite' and store.path.parent.name == 'annotations'


@pytest.mark.parametrize('change, reason', [
    ({'start_ns': T0 - 1}, 'bad_block'),
    ({'end_ns': END + 1}, 'bad_block'),
    ({'end_ns': at(10)}, 'bad_block'),
    ({'end_ns': at(10.5)}, 'bad_block'),
    ({'start_ns': float(T0)}, 'bad_value'),
    ({'start_ns': True}, 'bad_value'),
    ({'base_revision': None}, 'bad_value'),
    ({'kph': -1}, 'bad_value'),
    ({'kph': 0}, 'bad_value'),
    ({'kph': 201}, 'bad_value'),
    ({'kph': '80'}, 'bad_value'),
    ({'kph': 80.0}, 'bad_value'),
    ({'kph': True}, 'bad_value'),
    ({'note': 'x' * 501}, 'bad_value'),
    ({'note': 5}, 'bad_value'),
])
def test_a_block_that_does_not_fit_is_refused(change, reason):
    body = {'start_ns': at(10), 'end_ns': at(20), 'kph': 80, 'note': '', 'base_revision': 0, **change}
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.block_of(body, SPAN)
    assert (refused.value.status, refused.value.reason) == (400, reason)


def test_the_blocks_that_fit():
    body = {'start_ns': at(10), 'end_ns': at(11), 'base_revision': 3}
    assert speed_review.block_of({**body, 'kph': 1}, SPAN) == (at(10), at(11), 1, '', 3)
    assert speed_review.block_of({**body, 'kph': 83, 'note': '  works  '}, SPAN) == (at(10), at(11), 83, 'works', 3)
    # No speed: the moments are left out of the training.
    assert speed_review.block_of(body, SPAN)[2] is None


def test_a_drive_without_a_scan_has_no_speed_and_takes_no_block(store):
    class NoLogs(FakeDrives):
        def summary(self, route):
            return None

        def scan_state(self, route):
            return {'route': route, 'state': 'scanning', 'progress': 0.407, 'error': None, 'files': 14}
    view = speed_review.drive_view(store, NoLogs(), ROUTE)
    assert view['speed'] == [] and view['held'] == [] and view['blocks'] == [] and view['revision'] == 0
    assert view['total_s'] == 0
    assert view['span_ns'] is None
    # The page says the reading at work and how far, and follows it.
    assert view['scan'] == {'state': 'scanning', 'progress': 0.407, 'error': None}
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save_block(store, NoLogs(), ROUTE, {}, 'a')
    assert refused.value.status == 404


def test_a_save_answers_with_the_drive_as_it_is_now(store):
    drives = FakeDrives([(at(0), 80.0), (at(100), 120.0)])
    body = {'start_ns': at(100), 'end_ns': at(200), 'kph': 100, 'base_revision': 0}
    view = speed_review.save_block(store, drives, ROUTE, body, 'a')
    assert view['revision'] == 1 and view['total_s'] == 300 and view['max_kph'] == 200 and view['span_ns'] == [T0, END]
    assert view['speed'] == [[at(0), 80], [at(100), 120]]
    assert [(row['kph'], row['reviewer']) for row in view['blocks']] == [(100, 'a')]
    view = speed_review.save_block(store, drives, ROUTE, {'remove': True, 'start_ns': at(100), 'base_revision': 1}, 'a')
    assert view['revision'] == 2 and view['blocks'] == []


# ---------- the roadworks ----------

def marks(store, route=ROUTE):
    return [((row['start_ns'] - T0) / NS, None if row['end_ns'] is None else (row['end_ns'] - T0) / NS)
            for row in store.works(route)]


def test_two_taps_mark_roadworks_and_a_start_alone_runs_to_the_end_of_the_drive(store):
    assert store.tap_works(ROUTE, 'start', at(50), END, 0, 'a') == 1
    assert marks(store) == [(50, None)]
    assert store.tap_works(ROUTE, 'end', at(80), END, 1, 'a') == 2
    assert marks(store) == [(50, 80)]
    store.tap_works(ROUTE, 'start', at(100), END, 2, 'a')
    # Remove acts on the mark at the frame shown, ended or not.
    store.tap_works(ROUTE, 'remove', at(120), END, 3, 'a')
    store.tap_works(ROUTE, 'remove', at(79), END, 4, 'a')
    assert marks(store) == []
    assert [(row['action'], row['kph'], row['state']) for row in store.history(ROUTE)] == [
        ('works_start', None, None), ('works_end', None, None), ('works_start', None, None), ('works_removed', None, None),
        ('works_removed', None, None)]
    assert [(row['start_ns'], row['end_ns']) for row in store.history(ROUTE)][:2] == [(at(50), END), (at(50), at(80))]


def test_roadworks_cut_no_block_and_share_the_revision_of_the_drive(store):
    store.set_block(ROUTE, at(0), at(100), 80, '', 0, 'a')
    store.tap_works(ROUTE, 'start', at(50), END, 1, 'a')
    assert blocks(store) == [(0, 100, 80)] and store.revision(ROUTE) == 2
    # Two tabs: a write from before the tap is refused, and so is a tap from before a write.
    with pytest.raises(SpeedReviewError) as refused:
        store.set_block(ROUTE, at(0), at(100), 120, '', 1, 'a')
    assert refused.value.status == 409
    with pytest.raises(SpeedReviewError) as refused:
        store.tap_works(ROUTE, 'end', at(60), END, 1, 'a')
    assert refused.value.status == 409 and marks(store) == [(50, None)]


@pytest.mark.parametrize('tap, seconds', [('start', 15), ('start', 30), ('end', 25), ('remove', 25)])
def test_a_tap_that_does_not_fit_the_marks_changes_nothing(store, tap, seconds):
    store.tap_works(ROUTE, 'start', at(10), END, 0, 'a')
    store.tap_works(ROUTE, 'end', at(20), END, 1, 'a')
    if seconds == 30:
        store.tap_works(ROUTE, 'start', at(25), END, 2, 'a')
    before = store.revision(ROUTE), marks(store)
    with pytest.raises(SpeedReviewError) as refused:
        store.tap_works(ROUTE, tap, at(seconds), END, before[0], 'a')
    assert (refused.value.status, refused.value.reason) == (400, 'bad_works')
    assert (store.revision(ROUTE), marks(store)) == before


@pytest.mark.parametrize('seconds', [5, 40, 50])
def test_an_end_stays_after_its_start_and_before_the_next_roadworks(store, seconds):
    store.tap_works(ROUTE, 'start', at(30), END, 0, 'a')
    store.tap_works(ROUTE, 'end', at(35), END, 1, 'a')
    store.tap_works(ROUTE, 'start', at(10), END, 2, 'a')
    with pytest.raises(SpeedReviewError) as refused:
        store.tap_works(ROUTE, 'end', at(seconds), END, 3, 'a')
    assert refused.value.reason == 'bad_works' and marks(store) == [(10, None), (30, 35)]
    store.tap_works(ROUTE, 'end', at(30), END, 3, 'a')
    assert marks(store) == [(10, 30), (30, 35)]


@pytest.mark.parametrize('change, reason', [
    ({'tap': 'move'}, 'bad_tap'),
    ({'tap': None}, 'bad_tap'),
    ({'at_ns': T0 - 1}, 'bad_stretch'),
    ({'at_ns': END}, 'bad_stretch'),
    ({'at_ns': float(T0)}, 'bad_value'),
    ({'at_ns': True}, 'bad_value'),
    ({'base_revision': None}, 'bad_value'),
])
def test_a_tap_outside_the_drive_is_refused(change, reason):
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.works_tap({'tap': 'start', 'at_ns': at(10), 'base_revision': 0, **change}, SPAN)
    assert (refused.value.status, refused.value.reason) == (400, reason)


def test_a_tap_answers_with_the_drive_as_it_is_now(store):
    view = speed_review.save_works(store, FakeDrives(), ROUTE, {'tap': 'start', 'at_ns': at(50), 'base_revision': 0}, 'a')
    assert view['revision'] == 1 and view['blocks'] == []
    assert view['works'] == [{'start_ns': at(50), 'end_ns': None, 'reviewer': 'a', 'revision': 1}]


def test_checked_to_here_says_where_a_person_looked_and_the_last_tap_holds(store):
    assert store.works_checked(ROUTE) is None
    store.tap_works(ROUTE, 'start', at(50), END, 0, 'a')
    store.tap_works(ROUTE, 'end', at(80), END, 1, 'a')
    assert store.tap_works(ROUTE, 'checked', at(120), END, 2, 'a') == 3
    assert store.works_checked(ROUTE) == at(120)
    store.tap_works(ROUTE, 'checked', at(90), END, 3, 'a')
    assert store.works_checked(ROUTE) == at(90) and marks(store) == [(50, 80)]
    assert [(row['action'], row['start_ns'], row['kph']) for row in store.history(ROUTE)][-1] == ('works_checked', at(90), None)


def test_a_drive_is_not_checked_past_roadworks_with_no_end(store):
    store.tap_works(ROUTE, 'start', at(50), END, 0, 'a')
    with pytest.raises(speed_review.SpeedReviewError) as refused:
        store.tap_works(ROUTE, 'checked', at(60), END, 1, 'a')
    assert refused.value.reason == 'bad_works' and store.works_checked(ROUTE) is None
    store.tap_works(ROUTE, 'checked', at(40), END, 1, 'a')
    assert store.works_checked(ROUTE) == at(40)


def test_the_page_reads_up_to_where_the_drive_is_checked(store):
    view = speed_review.save_works(store, FakeDrives(), ROUTE, {'tap': 'checked', 'at_ns': at(50), 'base_revision': 0}, 'a')
    assert view['works_checked_ns'] == at(50) and view['works'] == []


def test_a_drive_without_a_scan_takes_no_roadworks(store):
    class NoLogs(FakeDrives):
        def summary(self, route):
            return None
    assert speed_review.drive_view(store, NoLogs(), ROUTE)['works'] == []
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save_works(store, NoLogs(), ROUTE, {}, 'a')
    assert refused.value.status == 404


def test_the_drives_say_their_blocks_and_roadworks(store, monkeypatch):
    routes = ['00000170--aaaaaaaaaa', '00000171--bbbbbbbbbb']
    monkeypatch.setattr(speed_review, 'drives_view', lambda _drives: [{'route': route} for route in routes])
    store.set_block(routes[1], at(0), at(30), 70, '', 0, 'a')
    store.set_block(routes[1], at(40), at(50), None, '', 1, 'a')
    store.tap_works(routes[1], 'start', at(60), END, 2, 'a')
    rows = speed_review.drives_list(store, FakeDrives())
    assert [(row['route'], row['blocks'], row['works'], row['total_s']) for row in rows] == [
        (routes[0], 0, 0, 300), (routes[1], 2, 1, 300)]
    # FakeDrives' summary has no carParams: no dashcam drive, the page leaves it out.
    assert [row['dashcam'] for row in rows] == [False, False]


def test_the_drives_say_whether_the_brain_learns_from_them(store, monkeypatch):
    class Drives(FakeDrives):
        def summary(self, route):
            if route.startswith('00000003'):
                return None
            return {'t0_ns': T0, 'span_ns': [T0, END], 'coverage': {'engaged': 0},
                    'meta': {'car': {'passive': route.startswith('00000001')}}}
    routes = ['00000001--aaaaaaaaaa', '00000002--bbbbbbbbbb', '00000003--cccccccccc']
    monkeypatch.setattr(speed_review, 'drives_view', lambda _drives: [{'route': route} for route in routes])
    # Not read yet: None, listed so that its logs can be read.
    assert [row['dashcam'] for row in speed_review.drives_list(store, Drives())] == [True, False, None]
    assert speed_review.drive_view(store, Drives(), routes[1])['dashcam'] is False



# ---------- a dashcam drive ----------

class DashcamDrives(FakeDrives):
    """A made-up dashcam drive: sunnypilot passive, never engaged."""

    def summary(self, route):
        return {'t0_ns': T0, 'span_ns': [T0, END], 'coverage': {'engaged': 0}, 'meta': {'car': {'passive': True}}}


def test_a_drive_has_no_switch_ok_for_training_any_more(store, monkeypatch):
    """The switch "OK for brain training" went on the evening of 9 Nov 2026: every dashcam drive enters the brain
    training, whole. The drive and the list say nothing of it, and its tables stay in the file, read by nothing."""
    assert 'training_ok' not in speed_review.drive_view(store, DashcamDrives(), ROUTE)
    monkeypatch.setattr(speed_review, 'drives_view', lambda _drives: [{'route': ROUTE}])
    assert 'training_ok' not in speed_review.drives_list(store, DashcamDrives())[0]
    assert not hasattr(speed_review, 'save_training') and not hasattr(store, 'set_training_ok')
    tables = {row[0] for row in sqlite3.connect(store.path).execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {'speed_training_ok', 'speed_training_history'} <= tables


# ---------- the kinds of road ----------

def road_marks(store, route=ROUTE):
    return [(row['kind'], (row['start_ns'] - T0) / NS, None if row['end_ns'] is None else (row['end_ns'] - T0) / NS)
            for row in store.road(route)]


def test_the_first_kinds_of_road_are_there_in_their_order(store):
    assert store.road_kinds() == [{'kind': kind, 'label': label} for kind, label in speed_review.FIRST_ROAD_KINDS]
    # Opening the file again seeds nothing twice.
    assert SpeedReviewStore(store.root).road_kinds() == store.road_kinds()


def test_marks_of_two_kinds_may_overlap_and_those_of_one_kind_may_not(store):
    store.tap_road(ROUTE, 'narrow', 'start', at(10), END, 0, 'a')
    store.tap_road(ROUTE, 'gravel', 'start', at(20), END, 1, 'a')
    store.tap_road(ROUTE, 'narrow', 'end', at(40), END, 2, 'a')
    assert road_marks(store) == [('gravel', 20, None), ('narrow', 10, 40)]
    with pytest.raises(SpeedReviewError) as refused:
        store.tap_road(ROUTE, 'narrow', 'start', at(30), END, 3, 'a')
    assert (refused.value.status, refused.value.reason) == (400, 'bad_road')
    # Roadworks are their own: a kind of road changes none of them, and all share the revision of the drive.
    store.tap_works(ROUTE, 'start', at(25), END, 3, 'a')
    assert marks(store) == [(25, None)] and store.revision(ROUTE) == 4
    store.tap_road(ROUTE, 'gravel', 'remove', at(50), END, 4, 'a')
    assert road_marks(store) == [('narrow', 10, 40)]
    assert [(row['action'], row['state']) for row in store.history(ROUTE)] == [
        ('road_start', 'narrow'), ('road_start', 'gravel'), ('road_end', 'narrow'), ('works_start', None),
        ('road_removed', 'gravel')]


def test_each_kind_is_checked_on_its_own(store):
    store.tap_road(ROUTE, 'potholes', 'start', at(10), END, 0, 'a')
    with pytest.raises(SpeedReviewError) as refused:
        store.tap_road(ROUTE, 'potholes', 'checked', at(20), END, 1, 'a')
    assert refused.value.reason == 'bad_road'
    store.tap_road(ROUTE, 'country', 'checked', at(200), END, 1, 'a')
    store.tap_road(ROUTE, 'potholes', 'checked', at(5), END, 2, 'a')
    assert store.road_checked(ROUTE) == {'country': at(200), 'potholes': at(5)} and store.works_checked(ROUTE) is None
    assert [(row['action'], row['state'], row['start_ns']) for row in store.history(ROUTE)][-1] == (
        'road_checked', 'potholes', at(5))


def test_a_tap_on_a_kind_that_is_not_there_is_refused(store):
    with pytest.raises(SpeedReviewError) as refused:
        store.tap_road(ROUTE, 'motorway', 'start', at(10), END, 0, 'a')
    assert (refused.value.status, refused.value.reason) == (400, 'bad_kind') and store.revision(ROUTE) == 0
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save_road(store, FakeDrives(), ROUTE, {'tap': 'start', 'at_ns': at(10), 'base_revision': 0}, 'a')
    assert refused.value.reason == 'bad_kind'


def test_a_person_adds_a_kind_of_road_after_the_others(store):
    answer = speed_review.save_road_kind(store, {'label': '  Strada   bianca '}, 'a')
    assert answer['kind'] == 'strada_bianca' and answer['road_kinds'][-1] == {'kind': 'strada_bianca', 'label': 'Strada bianca'}
    # The same name twice is refused, whatever its case; a name with the kind of another gets a number.
    with pytest.raises(SpeedReviewError) as refused:
        store.add_road_kind('strada BIANCA', 'a')
    assert (refused.value.status, refused.value.reason) == (400, 'kind_exists')
    assert store.add_road_kind('Strada-bianca', 'a') == 'strada_bianca_2'
    assert store.add_road_kind('Über', 'a') == 'ber'
    # The roadworks' names and those a page or a guest's copy reads otherwise are never a kind: a number is added.
    for label in ('Roadworks', 'works'):
        with pytest.raises(SpeedReviewError) as refused:
            store.add_road_kind(label, 'a')
        assert refused.value.reason == 'kind_exists'
    assert [store.add_road_kind(label, 'a') for label in ('Works!', 'Constructor', 'Title', 'GPS', 'Road name')] == \
        ['works_2', 'constructor_2', 'title_2', 'gps_2', 'road_name_2']
    for label in ('', '   ', 'x' * (speed_review.MAX_KIND_LABEL + 1)):
        with pytest.raises(SpeedReviewError) as refused:
            store.add_road_kind(label, 'a')
        assert refused.value.reason == 'bad_value'
    with pytest.raises(SpeedReviewError):
        speed_review.save_road_kind(store, {'label': 7}, 'a')
    store.tap_road(ROUTE, 'strada_bianca', 'start', at(10), END, 0, 'a')
    assert road_marks(store) == [('strada_bianca', 10, None)]


def test_no_kind_of_road_is_a_field_a_guest_s_copy_drops():
    from nnslr_tools.review_access import PLACE_FIELDS
    assert PLACE_FIELDS <= speed_review.RESERVED_KINDS


def test_a_kind_of_road_is_never_deleted(store):
    connection = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError):
        connection.execute("DELETE FROM speed_road_kinds WHERE kind = 'narrow'")
    connection.close()


def test_a_tap_on_a_kind_answers_with_the_drive_as_it_is_now(store):
    body = {'kind': 'country', 'tap': 'start', 'at_ns': at(50), 'base_revision': 0}
    view = speed_review.save_road(store, FakeDrives(), ROUTE, body, 'a')
    assert view['revision'] == 1 and view['works'] == [] and view['road_checked_ns'] == {}
    assert view['road'] == [{'kind': 'country', 'start_ns': at(50), 'end_ns': None, 'reviewer': 'a', 'revision': 1}]
    assert [row['kind'] for row in view['road_kinds']] == [kind for kind, _label in speed_review.FIRST_ROAD_KINDS]


def test_the_drives_say_their_marks_of_a_kind_of_road(store, monkeypatch):
    monkeypatch.setattr(speed_review, 'drives_view', lambda _drives: [{'route': ROUTE}])
    store.tap_road(ROUTE, 'narrow', 'start', at(10), END, 0, 'a')
    store.tap_road(ROUTE, 'gravel', 'start', at(10), END, 1, 'a')
    assert [(row['blocks'], row['works'], row['road']) for row in speed_review.drives_list(store, FakeDrives())] == [(0, 0, 2)]


# ---------- the road types ----------

# The map of the made-up drive, a frame a second of minute 0: town 0-9 s, rural 10-19 s, motorway 20-29 s, then no frames
# up to 35 s and motorway again to 39 s.
MAP_TYPES = [(second, 'town' if second < 10 else 'rural' if second < 20 else 'motorway')
             for second in (*range(30), *range(35, 40))]


class TypedDrives(FakeDrives):
    """A drive with the road camera's encode index: frame k of minute 0 exposed at T0 + k / 20 s, so the frame of second
    s of the map (decoded 20 s) at at(s)."""

    def merged(self, route):
        ids = list(range(0, 20 * 60))
        return {**merged(self.points), 'frames': {'0': {'sof': [T0 + frame * NS // 20 for frame in ids], 'ids': ids}}}


def write_map(root: Path, route=ROUTE, types=MAP_TYPES) -> None:
    from nnslr_tools.road_types import road_types_path
    path = road_types_path(root, route)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'frame_key': f'0/{second}', 'segment_index': 0, 'output_index': second,
                                        'road_type': kind, 'basis': 'test', 'sure': True}) + '\n'
                            for second, kind in types), encoding='utf-8')


def stretches(view):
    return [(row['road_type'], row['reviewer'], row['from'], row['to'], (row['from_ns'] - T0) / NS, (row['to_ns'] - T0) / NS)
            for row in view['road_types']]


def test_the_page_gets_the_road_types_of_the_map_as_stretches_on_the_comma_s_clock(store, tmp_path):
    write_map(tmp_path)
    view = speed_review.drive_view(store, TypedDrives(), ROUTE)
    assert view['road_type_names'][:4] == ['town', 'rural', 'expressway', 'motorway']
    # A stretch runs to the next one's first frame; the frames missing end one, though the type is the same.
    assert stretches(view) == [('town', None, '0/0', '0/9', 0, 10), ('rural', None, '0/10', '0/19', 10, 20),
                               ('motorway', None, '0/20', '0/29', 20, 30), ('motorway', None, '0/35', '0/39', 35, 40)]
    # No map yet, or no encode index: no stretches.
    assert speed_review.drive_view(store, FakeDrives(), ROUTE)['road_types'] == []
    assert speed_review.drive_view(store, TypedDrives(), '00000002--abcdef1234')['road_types'] == []


def test_a_road_type_on_the_frame_shown_corrects_its_whole_stretch(store, tmp_path):
    write_map(tmp_path)
    body = {'road_type': 'town', 'at_ns': at(14.5), 'base_revision': 0}
    view = speed_review.save_road_type(store, TypedDrives(), ROUTE, body, 'a')
    assert view['revision'] == 1
    assert stretches(view)[:3] == [('town', None, '0/0', '0/9', 0, 10), ('town', 'a', '0/10', '0/19', 10, 20),
                                   ('motorway', None, '0/20', '0/29', 20, 30)]
    # The map's own correction, the one every training reads; the drive's history says it too.
    from nnslr_tools import road_types
    assert [(line['from'], line['to'], line['road_type'], line['reviewer']) for line in road_types.load_corrections(tmp_path, ROUTE)] \
        == [('0/10', '0/19', 'town', 'a')]
    assert [line['road_type'] for line in road_types.load_road_types(tmp_path, ROUTE)][9:21] == ['town'] * 11 + ['motorway']
    row = store.history(ROUTE)[0]
    assert (row['action'], row['state'], row['note'], row['start_ns'], row['end_ns']) == ('type_corrected', 'town', '0/10 to 0/19',
                                                                                         at(10), at(20))


def test_from_one_frame_to_another_corrects_those_frames_in_either_order(store, tmp_path):
    write_map(tmp_path)
    body = {'road_type': None, 'from_ns': at(25.5), 'to_ns': at(21.2), 'base_revision': 0}
    view = speed_review.save_road_type(store, TypedDrives(), ROUTE, body, 'a')
    assert stretches(view)[2:5] == [('motorway', None, '0/20', '0/20', 20, 21), (None, 'a', '0/21', '0/25', 21, 26),
                                    ('motorway', None, '0/26', '0/29', 26, 30)]
    assert store.history(ROUTE)[0]['state'] == 'cant_tell'


def test_from_before_the_map_starts_is_from_its_first_frame(store, tmp_path):
    write_map(tmp_path, types=[(second, 'town') for second in range(2, 10)])
    body = {'road_type': 'rural', 'from_ns': T0, 'to_ns': at(4.5), 'base_revision': 0}
    view = speed_review.save_road_type(store, TypedDrives(), ROUTE, body, 'a')
    assert stretches(view) == [('rural', 'a', '0/2', '0/4', 2, 5), ('town', None, '0/5', '0/9', 5, 10)]
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save_road_type(store, TypedDrives(), ROUTE, {**body, 'to_ns': at(1), 'base_revision': 1}, 'a')
    assert refused.value.reason == 'bad_stretch'


@pytest.mark.parametrize('change, status, reason', [
    ({'road_type': 'highway'}, 400, 'bad_road_type'),
    ({'road_type': 'Town'}, 400, 'bad_road_type'),
    ({'road_type': 'missing'}, 400, 'bad_road_type'),
    ({'at_ns': str(at(5))}, 400, 'bad_value'),
    ({'at_ns': at(-1)}, 400, 'bad_stretch'),
    ({'at_ns': at(32)}, 400, 'bad_stretch'),
    ({'from_ns': at(1)}, 400, 'bad_stretch'),
    ({'at_ns': 'missing'}, 400, 'bad_stretch'),
    ({'at_ns': 'missing', 'from_ns': at(1)}, 400, 'bad_value'),
    ({'base_revision': 3}, 409, 'stale_revision'),
])
def test_a_road_type_that_does_not_fit_changes_nothing(store, tmp_path, change, status, reason):
    write_map(tmp_path)
    body = {'road_type': 'town', 'at_ns': at(5), 'base_revision': 0, **change}
    body = {key: value for key, value in body.items() if value != 'missing'}
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save_road_type(store, TypedDrives(), ROUTE, body, 'a')
    assert (refused.value.status, refused.value.reason) == (status, reason)
    from nnslr_tools import road_types
    assert store.revision(ROUTE) == 0 and road_types.load_corrections(tmp_path, ROUTE) == []


def test_a_drive_without_the_map_takes_no_road_type(store):
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save_road_type(store, TypedDrives(), ROUTE, {'road_type': 'town', 'at_ns': at(5), 'base_revision': 0}, 'a')
    assert (refused.value.status, refused.value.reason) == (404, 'no_road_types')
    assert store.revision(ROUTE) == 0


# ---------- who writes ----------

@pytest.fixture
def drives(monkeypatch):
    """The site's drive review answering for a made-up drive."""
    from nnslr_tools import drive_review
    fake = FakeDrives()
    monkeypatch.setattr(drive_review, 'store_for', lambda _root: fake)
    return fake


def test_only_a_person_signed_in_writes_a_block(tmp_path, drives):
    make_report(tmp_path)
    path = f'/api/speed-review/drives/{ROUTE}/blocks'
    body = {'start_ns': at(0), 'end_ns': at(10), 'kph': 80, 'base_revision': 0}
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    token = {'X-Review-Token': TOKEN}

    def post(headers, sent=body):
        status, _response, data = server.request('POST', path, body=sent, headers=headers)
        return status, json_of(data)

    try:
        status, answer = post(token)
        assert (status, answer['error']) == (403, 'session_required')
        session = sign_in(server)
        assert post({**session, 'Origin': 'http://elsewhere.example'})[1]['error'] == 'cross_origin'
        assert post({**session, 'Content-Type': 'text/plain'})[0] == 415
        server.app.panel.actions_enabled = False
        assert post(session)[1]['error'] == 'actions_disabled'
        server.app.panel.actions_enabled = True
        assert speed_review.store_for(tmp_path).revision(ROUTE) == 0
        status, answer = post(session)
        assert status == 200 and answer['revision'] == 1
        # Saved under the person's name, and a second write from the same read is refused.
        assert speed_review.store_for(tmp_path).history(ROUTE)[0]['reviewer'] == 'admin'
        status, answer = post(session)
        assert (status, answer['error']) == (409, 'stale_revision')
        status, answer = post(session, {**body, 'kph': 201, 'base_revision': 1})
        assert (status, answer['error']) == (400, 'bad_value')
        # Its ends dragged: the block that started at 0 s goes from 2 s to 8 s, and nothing stays where it was.
        status, answer = post(session, {**body, 'start_ns': at(2), 'end_ns': at(8), 'from_start_ns': at(0), 'base_revision': 1})
        assert status == 200 and [(row['start_ns'], row['end_ns']) for row in answer['blocks']] == [(at(2), at(8))]
        for moved, reason in ((str(at(2)), 'bad_value'), (at(5), 'bad_block')):
            status, answer = post(session, {**body, 'from_start_ns': moved, 'base_revision': 2})
            assert (status, answer['error']) == (400, reason)
        # Reading is open to the token too.
        status, _response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers=token)
        assert status == 200 and json_of(data)['revision'] == 2
        assert server.request('GET', path, headers=token)[0] == 405
        assert server.request('GET', '/api/speed-review/nothing', headers=token)[0] == 404
    finally:
        server.close()
        speed_review._stores.clear()


def test_only_a_person_signed_in_marks_roadworks(tmp_path, drives):
    make_report(tmp_path)
    path = f'/api/speed-review/drives/{ROUTE}/works'
    body = {'tap': 'start', 'at_ns': at(10), 'base_revision': 0}
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    token = {'X-Review-Token': TOKEN}

    def post(headers, sent=body):
        status, _response, data = server.request('POST', path, body=sent, headers=headers)
        return status, json_of(data)

    try:
        status, answer = post(token)
        assert (status, answer['error']) == (403, 'session_required')
        session = sign_in(server)
        assert post({**session, 'Origin': 'http://elsewhere.example'})[1]['error'] == 'cross_origin'
        assert post({**session, 'Content-Type': 'text/plain'})[0] == 415
        server.app.panel.actions_enabled = False
        assert post(session)[1]['error'] == 'actions_disabled'
        server.app.panel.actions_enabled = True
        assert speed_review.store_for(tmp_path).works(ROUTE) == []
        status, answer = post(session)
        assert status == 200 and answer['revision'] == 1 and answer['works'][0]['reviewer'] == 'admin'
        status, answer = post(session)
        assert (status, answer['error']) == (409, 'stale_revision')
        status, answer = post(session, {**body, 'base_revision': 1})
        assert (status, answer['error']) == (400, 'bad_works')
        assert server.request('GET', path, headers=token)[0] == 405
    finally:
        server.close()
        speed_review._stores.clear()



def test_only_a_person_signed_in_marks_a_kind_of_road_or_adds_one(tmp_path, drives):
    make_report(tmp_path)
    path = f'/api/speed-review/drives/{ROUTE}/road'
    body = {'kind': 'narrow', 'tap': 'start', 'at_ns': at(10), 'base_revision': 0}
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    token = {'X-Review-Token': TOKEN}

    def post(headers, sent=body, to=path):
        status, _response, data = server.request('POST', to, body=sent, headers=headers)
        return status, json_of(data)

    try:
        status, answer = post(token)
        assert (status, answer['error']) == (403, 'session_required')
        assert post(token, {'label': 'Strada bianca'}, '/api/speed-review/road-kinds')[1]['error'] == 'session_required'
        session = sign_in(server)
        assert post({**session, 'Origin': 'http://elsewhere.example'})[1]['error'] == 'cross_origin'
        server.app.panel.actions_enabled = False
        assert post(session, {'label': 'Strada bianca'}, '/api/speed-review/road-kinds')[1]['error'] == 'actions_disabled'
        server.app.panel.actions_enabled = True
        status, answer = post(session)
        assert status == 200 and answer['revision'] == 1 and answer['road'][0]['reviewer'] == 'admin'
        status, answer = post(session, {**body, 'base_revision': 1})
        assert (status, answer['error']) == (400, 'bad_road')
        status, answer = post(session, {'label': 'Strada bianca'}, '/api/speed-review/road-kinds')
        assert status == 200 and answer['kind'] == 'strada_bianca'
        assert speed_review.store_for(tmp_path).road_kinds()[-1]['label'] == 'Strada bianca'
        status, answer = post(session, {'label': 'Strada bianca'}, '/api/speed-review/road-kinds')
        assert (status, answer['error']) == (400, 'kind_exists')
        assert server.request('GET', path, headers=token)[0] == 405
        assert server.request('GET', '/api/speed-review/road-kinds', headers=token)[0] == 405
    finally:
        server.close()
        speed_review._stores.clear()

def test_only_a_person_signed_in_corrects_a_road_type(tmp_path, monkeypatch):
    from nnslr_tools import drive_review
    monkeypatch.setattr(drive_review, 'store_for', lambda _root: TypedDrives())
    make_report(tmp_path)
    write_map(tmp_path)
    path = f'/api/speed-review/drives/{ROUTE}/road-type'
    body = {'road_type': 'rural', 'at_ns': at(5), 'base_revision': 0}
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    token = {'X-Review-Token': TOKEN}

    def post(headers, sent=body):
        status, _response, data = server.request('POST', path, body=sent, headers=headers)
        return status, json_of(data)

    try:
        status, answer = post(token)
        assert (status, answer['error']) == (403, 'session_required')
        session = sign_in(server)
        assert post({**session, 'Origin': 'http://elsewhere.example'})[1]['error'] == 'cross_origin'
        server.app.panel.actions_enabled = False
        assert post(session)[1]['error'] == 'actions_disabled'
        server.app.panel.actions_enabled = True
        status, answer = post(session)
        assert status == 200 and answer['revision'] == 1
        assert [(row['road_type'], row['reviewer']) for row in answer['road_types']][0] == ('rural', 'admin')
        status, answer = post(session)
        assert (status, answer['error']) == (409, 'stale_revision')
        status, answer = post(session, {**body, 'road_type': 'autobahn', 'base_revision': 1})
        assert (status, answer['error']) == (400, 'bad_road_type')
        status, _response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers=token)
        assert status == 200 and json_of(data)['road_types'][0]['road_type'] == 'rural'
        assert server.request('GET', path, headers=token)[0] == 405
    finally:
        server.close()
        speed_review._stores.clear()


def test_the_switch_ok_for_training_is_gone_from_the_site(tmp_path, monkeypatch):
    from nnslr_tools import drive_review
    monkeypatch.setattr(drive_review, 'store_for', lambda _root: DashcamDrives())
    make_report(tmp_path)
    path = f'/api/speed-review/drives/{ROUTE}/training'
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        status, _response, data = server.request('POST', path, body={'ok': True}, headers=sign_in(server))
        assert (status, json_of(data)['error']) == (404, 'not_found')
        assert server.request('GET', path, headers={'X-Review-Token': TOKEN})[0] == 404
        status, _response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers={'X-Review-Token': TOKEN})
        assert status == 200 and 'training_ok' not in json_of(data)
    finally:
        server.close()
        speed_review._stores.clear()


@pytest.mark.parametrize('kind', ['blocks', 'works'])
@pytest.mark.parametrize('raw', [b'[]', b'null', b'"ok"', b'true'])
def test_a_speed_review_write_that_is_no_json_object_is_a_400(tmp_path, monkeypatch, kind, raw):
    """Valid JSON that is no object ([], null, a string, true) is refused with a 400 on every Speed review write, never
    a server error; nothing is written."""
    from nnslr_tools import drive_review
    monkeypatch.setattr(drive_review, 'store_for', lambda _root: DashcamDrives())
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        status, _response, data = server.request('POST', f'/api/speed-review/drives/{ROUTE}/{kind}', body=raw,
                                                 headers=sign_in(server))
        assert (status, json_of(data)['error']) == (400, 'invalid_json')
        store = speed_review.store_for(tmp_path)
        assert store.revision(ROUTE) == 0
    finally:
        server.close()
        speed_review._stores.clear()


def test_the_hd_switch_gets_the_frame_of_the_road_video_at_its_own_size(tmp_path):
    """Standing still with HD on, the player shows frame k of a minute of the comma's road video (fcamera.hevc): frame k of
    that minute of its own video. Served as a JPEG at the video's size, never scaled; a frame the video lacks is a 404."""
    from PIL import Image
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe required for local media integration')
    video = tmp_path / 'raw/routes' / ROUTE / '0' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    made = subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=s=96x54:r=20:d=2', '-pix_fmt', 'yuv420p',
        '-c:v', 'libx265', '-x265-params', 'pools=1:frame-threads=1:log-level=error:bframes=0:keyint=20:min-keyint=20',
        '-f', 'hevc', str(video)], capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    token = {'X-Review-Token': TOKEN}
    try:
        status, response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}/frames/0/25.jpg', headers=token)
        assert status == 200 and response.getheader('Content-Type') == 'image/jpeg'
        assert response.getheader('Cache-Control') == 'private, max-age=3600'
        assert Image.open(io.BytesIO(data)).size == (96, 54)
        for path in (f'/api/speed-review/drives/{ROUTE}/frames/0/40.jpg', f'/api/speed-review/drives/{ROUTE}/frames/1/0.jpg'):
            status, _response, data = server.request('GET', path, headers=token)
            assert (status, json_of(data)['error']) == (404, 'no_video_frame')
    finally:
        server.close()
        speed_review._stores.clear()


def test_an_agent_has_no_way_to_write_a_speed():
    assert 'speed_review' not in inspect.getsource(review_mcp)


def test_every_control_page_shows_the_sign_in_page_without_a_session():
    assert '/speed-review' in CONTROL_TABS
    assert [path for path in CONTROL_TABS if not PAGE_RE.fullmatch(path)] == []
