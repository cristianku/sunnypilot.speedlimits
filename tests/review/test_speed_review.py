"""Speed review (Data › Speed review, nnslr_tools.speed_review): the proposals from the logged vision limit, the writes
of a person, and who may write them.

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import inspect
import sqlite3
import stat
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
    """What the store needs of a drive review: the span of a drive, its joined logs."""

    def __init__(self, points=((T0, 80.0),)):
        self.points = [list(point) for point in points]

    def summary(self, route):
        return {'t0_ns': T0, 'span_ns': [T0, END]}

    def merged(self, route):
        return {'t0': T0, 'span': [T0, END], 'series': {'vis_limit': self.points}}


def merged(points):
    return {'t0': T0, 'span': [T0, END], 'series': {'vis_limit': [list(point) for point in points]}}


def at(seconds: float) -> int:
    return T0 + round(seconds * NS)


def label(kph=100, state='corrected', known_ns=None, note=''):
    return {'kph': kph, 'state': state, 'known_ns': known_ns, 'note': note}


def spans(store, route=ROUTE):
    return [((row['start_ns'] - T0) / NS, (row['end_ns'] - T0) / NS, row['kph'], row['state'])
            for row in store.stretches(route)]


@pytest.fixture
def store(tmp_path):
    return SpeedReviewStore(tmp_path)


# ---------- the proposals ----------

def test_a_proposal_rounds_to_the_grid_and_a_flicker_joins_the_stretch_before():
    found = speed_review.proposals(merged([(at(0), 79.99), (at(100), 121.0), (at(102), 80.0), (at(200), 0.0)]))
    assert found == [{'start_ns': at(0), 'end_ns': at(200), 'kph': 80},
                     {'start_ns': at(200), 'end_ns': END, 'kph': None}]


def test_a_proposal_covers_the_drive_from_its_first_second_even_with_nothing_held():
    # Before the first message nothing is known; a drive without the series has one stretch with no value.
    assert speed_review.proposals(merged([(at(50), 60.0)])) == [
        {'start_ns': T0, 'end_ns': at(50), 'kph': None}, {'start_ns': at(50), 'end_ns': END, 'kph': 60}]
    assert speed_review.proposals(merged([])) == [{'start_ns': T0, 'end_ns': END, 'kph': None}]
    assert speed_review.proposals(None) == []
    assert speed_review.proposals({'t0': None, 'span': [None, None], 'series': {}}) == []


def test_a_flicker_at_the_start_of_a_drive_joins_the_stretch_after():
    # The first message comes a moment after the start of the logs: no stretch of a tenth of a second to answer for.
    assert speed_review.proposals(merged([(at(0.1), 80.0), (at(100), 60.0)])) == [
        {'start_ns': T0, 'end_ns': at(100), 'kph': 80}, {'start_ns': at(100), 'end_ns': END, 'kph': 60}]


def test_saved_stretches_cut_the_proposals_and_the_whole_drive_stays_covered():
    proposed = speed_review.proposals(merged([(at(0), 80.0), (at(100), 120.0)]))
    saved = [{'start_ns': at(50), 'end_ns': at(150), 'kph': 100, 'state': 'corrected', 'known_ns': None, 'note': '',
              'reviewer': 'a'}]
    whole = speed_review.partition(proposed, saved, SPAN)
    assert [(item['start_ns'], item['end_ns'], item['kph'], item['state']) for item in whole] == [
        (at(0), at(50), 80, 'not_examined'), (at(50), at(150), 100, 'corrected'), (at(150), END, 120, 'not_examined')]


# ---------- the writes ----------

def test_a_write_clips_what_it_covers_and_a_clear_brings_the_proposal_back(store):
    first = store.apply(ROUTE, at(0), at(100), label(80, 'confirmed'), 0, 'a')
    assert first == 1
    # Split inside a saved stretch: the other side stays as it was, with the revision that wrote it.
    assert store.apply(ROUTE, at(60), at(100), label(120), first, 'a') == 2
    assert spans(store) == [(0, 60, 80, 'confirmed'), (60, 100, 120, 'corrected')]
    assert [row['revision'] for row in store.stretches(ROUTE)] == [1, 2]
    # A write in the middle leaves two sides.
    store.apply(ROUTE, at(10), at(20), label(90), 2, 'a')
    assert spans(store) == [(0, 10, 80, 'confirmed'), (10, 20, 90, 'corrected'), (20, 60, 80, 'confirmed'),
                            (60, 100, 120, 'corrected')]
    # A write over several removes the ones inside.
    store.apply(ROUTE, at(5), at(70), label(100), 3, 'a')
    assert spans(store) == [(0, 5, 80, 'confirmed'), (5, 70, 100, 'corrected'), (70, 100, 120, 'corrected')]
    store.apply(ROUTE, at(5), at(70), None, 4, 'a')
    assert spans(store) == [(0, 5, 80, 'confirmed'), (70, 100, 120, 'corrected')]
    assert store.revision(ROUTE) == 5
    assert store.examined()[ROUTE] == 35 * NS


def test_a_clipped_stretch_loses_a_known_time_outside_it(store):
    store.apply(ROUTE, at(0), at(100), label(120, known_ns=at(70)), 0, 'a')
    store.apply(ROUTE, at(50), at(60), label(80), 1, 'a')
    left, _middle, right = store.stretches(ROUTE)
    assert left['known_ns'] is None and right['known_ns'] == at(70)
    store.apply(ROUTE, at(60), at(80), label(90), 2, 'a')
    assert store.stretches(ROUTE)[-1]['known_ns'] is None


def test_a_stale_revision_changes_nothing(store):
    store.apply(ROUTE, at(0), at(100), label(80, 'confirmed'), 0, 'a')
    with pytest.raises(SpeedReviewError) as refused:
        store.apply(ROUTE, at(0), at(100), label(120), 0, 'b')
    assert (refused.value.status, refused.value.reason) == (409, 'stale_revision')
    assert spans(store) == [(0, 100, 80, 'confirmed')] and store.revision(ROUTE) == 1
    # The revision is the drive's own: another drive starts from 0.
    assert store.apply('00000000--0000000000', at(0), at(10), label(), 0, 'a') == 1


def test_the_history_only_grows_and_rebuilds_the_stretches(store, tmp_path):
    writes = [(0, 100, label(80, 'confirmed')), (40, 60, label(120, known_ns=at(50))), (50, 90, None),
              (0, 30, label(None, 'undetermined'))]
    for revision, (start, end, value) in enumerate(writes):
        store.apply(ROUTE, at(start), at(end), value, revision, 'a')
    history = store.history(ROUTE)
    assert [row['revision'] for row in history] == [1, 2, 3, 4]
    assert [row['action'] for row in history] == ['saved', 'saved', 'cleared', 'saved']
    assert history[0]['reviewer'] == 'a' and history[0]['at']
    # Played again on an empty database, the history gives the same stretches.
    replay = SpeedReviewStore(tmp_path / 'replay')
    for row in history:
        value = None if row['action'] == 'cleared' else {key: row[key] for key in ('kph', 'state', 'known_ns', 'note')}
        replay.apply(ROUTE, row['start_ns'], row['end_ns'], value, row['revision'] - 1, row['reviewer'])
    assert replay.stretches(ROUTE) == store.stretches(ROUTE)
    connection = sqlite3.connect(store.path)
    for sql in ('UPDATE speed_history SET kph = 1', 'DELETE FROM speed_history'):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(sql)
    connection.close()


def test_the_labels_are_readable_only_by_their_owner(store):
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert store.path.name == 'speed_review.sqlite' and store.path.parent.name == 'annotations'


@pytest.mark.parametrize('change, reason', [
    ({'start_ns': T0 - 1}, 'bad_stretch'),
    ({'end_ns': END + 1}, 'bad_stretch'),
    ({'end_ns': at(10)}, 'bad_stretch'),
    ({'start_ns': float(T0)}, 'bad_value'),
    ({'start_ns': True}, 'bad_value'),
    ({'base_revision': None}, 'bad_value'),
    ({'state': 'not_examined'}, 'bad_state'),
    ({'kph': 82}, 'bad_value'),
    ({'kph': 0}, 'bad_value'),
    ({'kph': 205}, 'bad_value'),
    ({'kph': None}, 'bad_value'),
    ({'kph': '80'}, 'bad_value'),
    ({'kph': 80.0}, 'bad_value'),
    ({'state': 'undetermined'}, 'bad_value'),
    ({'kph': None, 'state': 'undetermined', 'known_ns': at(20)}, 'bad_value'),
    ({'known_ns': at(10) - 1}, 'bad_value'),
    ({'known_ns': at(20)}, 'bad_value'),
    ({'note': 'x' * 501}, 'bad_value'),
    ({'note': 5}, 'bad_value'),
])
def test_a_label_that_does_not_fit_is_refused(change, reason):
    body = {'start_ns': at(10), 'end_ns': at(20), 'kph': 80, 'state': 'confirmed', 'known_ns': None, 'note': '',
            'base_revision': 0, **change}
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.checked(body, SPAN)
    assert (refused.value.status, refused.value.reason) == (400, reason)


def test_the_labels_that_fit():
    body = {'start_ns': at(10), 'end_ns': at(20), 'base_revision': 3}
    assert speed_review.checked({**body, 'state': None}, SPAN)[2] is None
    assert speed_review.checked({**body, 'state': 'undetermined'}, SPAN)[2]['kph'] is None
    start, end, value, base = speed_review.checked({**body, 'state': 'corrected', 'kph': 130, 'known_ns': at(15),
                                                    'note': '  works  '}, SPAN)
    assert (start, end, base) == (at(10), at(20), 3)
    assert value == {'kph': 130, 'state': 'corrected', 'known_ns': at(15), 'note': 'works'}


def test_a_drive_without_a_scan_has_no_stretches_and_takes_no_label(store):
    class NoLogs(FakeDrives):
        def summary(self, route):
            return None
    view = speed_review.drive_view(store, NoLogs(), ROUTE)
    assert view['stretches'] == [] and view['revision'] == 0
    with pytest.raises(SpeedReviewError) as refused:
        speed_review.save(store, NoLogs(), ROUTE, {}, 'a')
    assert refused.value.status == 404


def test_a_save_answers_with_the_drive_as_it_is_now(store):
    drives = FakeDrives([(at(0), 80.0), (at(100), 120.0)])
    body = {'start_ns': at(100), 'end_ns': at(200), 'kph': 120, 'state': 'confirmed', 'base_revision': 0}
    view = speed_review.save(store, drives, ROUTE, body, 'a')
    assert view['revision'] == 1 and view['examined_s'] == 100 and view['total_s'] == 300
    assert [(item['kph'], item['state']) for item in view['stretches']] == [
        (80, 'not_examined'), (120, 'confirmed'), (120, 'not_examined')]


def test_the_drives_say_which_to_review_first(store, monkeypatch):
    routes = ['00000170--aaaaaaaaaa', *speed_review.FIRST_DRIVES]
    monkeypatch.setattr(speed_review, 'drives_view', lambda _drives: [{'route': route} for route in routes])
    store.apply(speed_review.FIRST_DRIVES[0], at(0), at(30), label(), 0, 'a')
    rows = speed_review.drives_list(store, FakeDrives())
    assert [row['route'] for row in rows] == routes
    assert [row['first'] for row in rows] == [False] + [True] * 5
    assert rows[1]['examined_s'] == 30 and rows[1]['total_s'] == 300


# ---------- who writes ----------

@pytest.fixture
def drives(monkeypatch):
    """The site's drive review answering for a made-up drive."""
    from nnslr_tools import drive_review
    fake = FakeDrives()
    monkeypatch.setattr(drive_review, 'store_for', lambda _root: fake)
    return fake


def test_only_a_person_signed_in_writes_a_speed(tmp_path, drives):
    make_report(tmp_path)
    path = f'/api/speed-review/drives/{ROUTE}/stretches'
    body = {'start_ns': at(0), 'end_ns': at(10), 'kph': 80, 'state': 'confirmed', 'base_revision': 0}
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
        status, answer = post(session, {**body, 'kph': 82, 'base_revision': 1})
        assert (status, answer['error']) == (400, 'bad_value')
        # Reading is open to the token too.
        status, _response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers=token)
        assert status == 200 and json_of(data)['revision'] == 1
        assert server.request('GET', path, headers=token)[0] == 405
        assert server.request('GET', '/api/speed-review/nothing', headers=token)[0] == 404
    finally:
        server.close()
        speed_review._stores.clear()


def test_an_agent_has_no_way_to_write_a_speed():
    assert 'speed_review' not in inspect.getsource(review_mcp)


def test_every_control_page_shows_the_sign_in_page_without_a_session():
    assert '/speed-review' in CONTROL_TABS
    assert [path for path in CONTROL_TABS if not PAGE_RE.fullmatch(path)] == []
