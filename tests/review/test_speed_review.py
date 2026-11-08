"""Speed review (Data › Speed review, nnslr_tools.speed_review): the driver's speed as the answer, the blocks of a person
that correct it, the roadworks, and who may write them.

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import inspect
import io
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


def test_the_peugeots_own_acc_and_limiter_are_held_though_carstate_says_the_cruise_is_off():
    # 0-10 s ACC at 100 (available, set speed), the gas pressed at 5-10 s; 10-20 s nothing set (255); 20-40 s the
    # limiter at 80 (not available), the car at 70, at 79, then over it at 90 (his kick-down); 40-50 s nothing known.
    series = {'cruise_enabled': [[at(0), False]],
              'cruise_available': [[at(0), True], [at(10), False], [at(40), None]],
              'cruise_speed': [[at(0), 100.0], [at(10), 255.0], [at(20), 80.0], [at(40), None]],
              'gas': [[at(0), False], [at(5), True], [at(10), False]],
              'v_ego': [[at(0), 100.0], [at(20), 70.0], [at(30), 79.0], [at(37), 90.0]]}
    times = [at(2), at(7), at(15), at(25), at(35), at(38), at(45)]
    peugeot = {'series': series, 'meta': {'car': {'brand': 'psa'}}}
    assert speed_review.machine_held(peugeot, times) == [True, False, False, False, True, False, False]
    # The rule is the Peugeot's: another car's cruise says itself in cruiseState.enabled.
    assert speed_review.machine_held({'series': series, 'meta': {'car': {'brand': 'subaru'}}}, times) == [False] * 7


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
    store.remove_block(ROUTE, at(0), 3, 'a')
    history = store.history(ROUTE)
    assert [row['revision'] for row in history] == [1, 2, 3, 4]
    assert [row['action'] for row in history] == ['block_saved', 'block_saved', 'block_saved', 'block_removed']
    assert history[0]['reviewer'] == 'a' and history[0]['at'] and history[1]['note'] == 'note'
    # Played again on an empty database, the history gives the same blocks.
    replay = SpeedReviewStore(tmp_path / 'replay')
    for row in history:
        if row['action'] == 'block_removed':
            replay.remove_block(ROUTE, row['start_ns'], row['revision'] - 1, row['reviewer'])
        else:
            replay.set_block(ROUTE, row['start_ns'], row['end_ns'], row['kph'], row['note'], row['revision'] - 1, row['reviewer'])
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
    view = speed_review.drive_view(store, NoLogs(), ROUTE)
    assert view['speed'] == [] and view['held'] == [] and view['blocks'] == [] and view['revision'] == 0
    assert view['total_s'] == 0
    assert view['span_ns'] is None
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
        # Reading is open to the token too.
        status, _response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers=token)
        assert status == 200 and json_of(data)['revision'] == 1
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
