"""Speed review (Data › Speed review, nnslr_tools.speed_review): the proposals from the map and the signs the comma
read, the writes of a person, and who may write them.

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import inspect
import json
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

def test_a_limit_rounds_to_the_grid_and_a_flicker_joins_the_stretch_before():
    found = speed_review.sign_runs(merged([(at(0), 79.99), (at(100), 121.0), (at(102), 80.0), (at(200), 0.0)]))
    assert found == [{'start_ns': at(0), 'end_ns': at(200), 'kph': 80},
                     {'start_ns': at(200), 'end_ns': END, 'kph': None}]


def test_a_limit_covers_the_drive_from_its_first_second_even_with_nothing_held():
    # Before the first message nothing is known; a drive without the series has one stretch with no value.
    assert speed_review.sign_runs(merged([(at(50), 60.0)])) == [
        {'start_ns': T0, 'end_ns': at(50), 'kph': None}, {'start_ns': at(50), 'end_ns': END, 'kph': 60}]
    assert speed_review.sign_runs(merged([])) == [{'start_ns': T0, 'end_ns': END, 'kph': None}]
    assert speed_review.sign_runs(None) == []
    assert speed_review.sign_runs({'t0': None, 'span': [None, None], 'series': {}}) == []


def test_a_flicker_at_the_start_of_a_drive_joins_the_stretch_after():
    # The first message comes a moment after the start of the logs: no stretch of a tenth of a second to answer for.
    assert speed_review.sign_runs(merged([(at(0.1), 80.0), (at(100), 60.0)])) == [
        {'start_ns': T0, 'end_ns': at(100), 'kph': 80}, {'start_ns': at(100), 'end_ns': END, 'kph': 60}]


def read_signs(reads, once=(), brain_limit=130.0):
    """A drive with the brain: its vision limit is the brain's answer, the signs are the signReads with their frame's
    raw reading. Each of reads is logged twice, 0.2 s apart, as a sign is read again as the car comes to it; each of once
    in one frame alone."""
    logged = sorted([(seconds + again, name, kph) for seconds, name, kph in reads for again in (0, 0.2)] + list(once))
    series = {'vis_limit': [[T0, brain_limit]], 'vis_raw_class': [[at(seconds), name] for seconds, name, _ in logged],
              'vis_raw_speed': [[at(seconds), kph] for seconds, _, kph in logged]}
    return {'t0': T0, 'span': [T0, END], 'series': series,
            'events': [[at(seconds), 'sign', brain_limit] for seconds, _, _ in logged]}


def test_the_signs_are_the_numbers_read_until_the_next_an_end_or_a_minute_unseen():
    reads = [(10, 'maximum_speed:80', 80.0), (40, 'maximum_speed:80', 80.0), (90, 'cancellation', None),
             (120, 'variable_display:100', 100.0), (200, 'minimum_speed:60', 60.0), (210, 'maximum_speed_hours:30', 30.0)]
    # The brain's 130 is never a sign; the 80 read again at 40 s holds to the end; the 100 a minute after it was seen.
    assert speed_review.sign_runs(read_signs(reads)) == [run(0, 10, None), run(10, 90, 80), run(90, 120, None),
                                                          run(120, 180.2, 100), run(180.2, None, None)]
    # A raw reading with no signRead is not a sign.
    unconfirmed = read_signs([(10, 'maximum_speed:80', 80.0)])
    unconfirmed['events'] = []
    assert speed_review.sign_runs(unconfirmed) == [run(0, None, None)]


def test_a_number_read_once_is_a_misreading():
    # One frame of 20 between two reads of 80, no second 20 near it.
    drive = read_signs([(10, 'maximum_speed:80', 80.0), (30, 'maximum_speed:80', 80.0)], once=[(20, 'maximum_speed:20', 20.0)])
    assert speed_review.sign_runs(drive) == [run(0, 10, None), run(10, 90.2, 80), run(90.2, None, None)]


def test_a_drive_from_before_the_raw_reading_takes_the_vision_limit():
    assert speed_review.sign_runs(merged([(at(0), 80.0), (at(100), 60.0)])) == [run(0, 100, 80), run(100, None, 60)]


def drive_with_frames(held=()):
    """A drive of two segments: segment 0 with an encode index whose frame 0 is exposed 2 s after the start of the
    logs, segment 1 without one."""
    ids = list(range(0, 1200, 20))
    return {'t0': T0, 'span': [T0, END], 'series': {'vis_limit': [list(point) for point in held]}, 'segments': [0, 1],
            'frames': {0: {'ids': ids, 'sof': [at(2 + frame / 20) for frame in ids]}},
            'spans': {0: [T0, at(60)], 1: [at(60), at(120)]}}


def map_rows(segment, seconds, maxspeed):
    return [{'frame_key': f'{segment}/{second}', 'segment_index': segment, 'output_index': second, 'maxspeed': maxspeed}
            for second in seconds]


def test_the_map_limit_is_placed_on_the_frames_of_the_video_and_only_a_number_is_a_limit():
    rows = (map_rows(0, range(0, 30), '80') + map_rows(0, range(30, 40), '30 mph') + map_rows(0, range(40, 44), 'FR:urban')
            + map_rows(0, range(44, 47), 'none') + map_rows(0, range(47, 50), None) + map_rows(1, range(0, 10), '120'))
    # Segment 0 on its frame line (a second 2 s after the logs' start), segment 1 from its start in the logs; the hole
    # between the two and the end of the rows say nothing.
    assert speed_review.map_runs(drive_with_frames(), rows) == [
        {'start_ns': T0, 'end_ns': at(32), 'kph': 80}, {'start_ns': at(32), 'end_ns': at(42), 'kph': 50},
        {'start_ns': at(42), 'end_ns': at(60), 'kph': None}, {'start_ns': at(60), 'end_ns': at(70), 'kph': 120},
        {'start_ns': at(70), 'end_ns': END, 'kph': None}]
    assert speed_review.map_runs(drive_with_frames(), []) == []
    assert speed_review.map_runs(None, rows) == []


def run(first, last, kph):
    return {'start_ns': at(first), 'end_ns': END if last is None else at(last), 'kph': kph}


def test_the_proposal_is_the_map_and_the_signs_only_where_the_map_has_no_number():
    mapped = [run(0, 100, 80), run(100, 103, None), run(103, 150, 80), run(150, None, None)]
    laid = speed_review.proposed(speed_review.sign_runs(merged([(T0, 100.0)])), mapped, SPAN)
    # Three seconds of the signs between two of the map join the stretch before.
    assert laid == [{**run(0, 150, 80), 'source': 'map'}, {**run(150, None, 100), 'source': 'signs'}]


def test_a_stretch_of_one_speed_says_where_most_of_it_comes_from():
    mapped = [run(0, 50, 80), run(50, None, None)]
    laid = speed_review.proposed(speed_review.sign_runs(merged([(T0, 80.0)])), mapped, SPAN)
    assert laid == [{**run(0, None, 80), 'source': 'signs'}]
    assert speed_review.proposed(speed_review.sign_runs(merged([(T0, 80.0)])), [], SPAN) == [
        {**run(0, None, 80), 'source': 'signs'}]


def test_the_signs_and_the_map_differ_only_both_known_and_for_long():
    signs = [run(0, 100, 80), run(100, 200, 120), run(200, 250, 100), run(250, None, None)]
    mapped = [run(0, 106, 80), run(106, None, 120)]
    # The 120 read 6 s before the map's way of 120 is a sign and the end of a way at two places, not a disagreement.
    assert speed_review.differences(signs, mapped) == [
        {'start_ns': at(200), 'end_ns': at(250), 'map_kph': 120, 'sign_kph': 100}]


def test_a_sign_is_passed_half_a_second_after_its_last_read():
    reads = [10, 10.4, 11, 12.5, 30, 30.3]
    drive = {'t0': T0, 'span': [T0, END], 'series': {}, 'events': [[at(seconds), 'sign', 80.0] for seconds in reads]}
    assert speed_review.sign_passages(drive) == [(at(10), at(13)), (at(30), at(30.8))]
    assert speed_review.sign_passages(merged([])) == [] and speed_review.sign_passages(None) == []


def test_every_change_of_the_proposal_moves_to_its_sign():
    proposal = [run(0, 100, 80), run(100, 200, 120), run(200, 230, 100), run(230, None, 60)]
    passages = [(at(96), at(104.5)), (at(160), at(162)), (at(239), at(241))]
    # The map's 120 starts 4.5 s before the car passes its sign; the 100 has no sign within 10 s and stays; the 60 moves
    # 11 s on, as the reads of its sign began 9 s after it.
    assert speed_review.at_the_signs(proposal, passages) == [
        run(0, 104.5, 80), run(104.5, 200, 120), run(200, 241, 100), run(241, None, 60)]
    # A move that would leave a stretch less than a second is not made.
    short = [run(0, 100, 80), run(100, None, 120)]
    assert speed_review.at_the_signs(short, [(at(99), at(299.5))]) == short
    assert speed_review.at_the_signs(proposal, []) == proposal


def test_the_drive_proposes_its_changes_at_the_signs_and_never_moves_a_saved_cut(store):
    drives = FakeDrives()
    drives.merged = lambda route: {**merged([(T0, 80.0), (at(100), 120.0)]),
                                   'events': [[at(seconds), 'sign', 120.0] for seconds in (100, 100.3, 101)]}
    view = speed_review.drive_view(store, drives, ROUTE)
    assert [(item['start_ns'], item['kph']) for item in view['stretches']] == [(T0, 80), (at(101.5), 120)]
    body = {'start_ns': T0, 'end_ns': at(100), 'kph': 80, 'state': 'confirmed', 'base_revision': 0}
    view = speed_review.save(store, drives, ROUTE, body, 'someone')
    assert [(item['start_ns'], item['end_ns'], item['state']) for item in view['stretches']] == [
        (T0, at(100), 'confirmed'), (at(100), at(101.5), 'not_examined'), (at(101.5), END, 'not_examined')]


def test_a_map_proposal_is_shown_stored_nowhere_and_a_person_confirms_it(store):
    path = store.root / 'derived' / 'road_types' / (ROUTE + '.jsonl')
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in map_rows(0, range(0, 58), '80')), encoding='utf-8')
    drives = FakeDrives()
    drives.merged = lambda route: drive_with_frames([(T0, 120.0)])
    view = speed_review.drive_view(store, drives, ROUTE)
    assert view['map_read'] is True and view['map'] == [run(0, 60, 80), run(60, None, None)]
    assert view['signs'] == [run(0, None, 120)]
    assert view['differ'] == [{'start_ns': T0, 'end_ns': at(60), 'map_kph': 80, 'sign_kph': 120}]
    assert [(item['kph'], item['source'], item['state']) for item in view['stretches']] == [
        (80, 'map', 'not_examined'), (120, 'signs', 'not_examined')]
    assert store.stretches(ROUTE) == [] and store.revision(ROUTE) == 0 and view['examined_s'] == 0
    body = {'start_ns': T0, 'end_ns': at(60), 'kph': 80, 'state': 'confirmed', 'base_revision': 0}
    view = speed_review.save(store, drives, ROUTE, body, 'someone')
    saved = view['stretches'][0]
    assert (saved['kph'], saved['state'], saved['reviewer']) == (80, 'confirmed', 'someone') and 'source' not in saved


def test_a_drive_without_the_map_read_is_proposed_from_the_signs(store):
    view = speed_review.drive_view(store, FakeDrives(), ROUTE)
    assert view['map_read'] is False and view['map'] == [] and view['differ'] == []
    assert [(item['kph'], item['source']) for item in view['stretches']] == [(80, 'signs')]


def test_saved_stretches_cut_the_proposals_and_the_whole_drive_stays_covered():
    proposed = speed_review.sign_runs(merged([(at(0), 80.0), (at(100), 120.0)]))
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


def test_a_clipped_stretch_stays_not_known_up_to_its_time(store):
    store.apply(ROUTE, at(0), at(100), label(120, known_ns=at(70)), 0, 'a')
    store.apply(ROUTE, at(50), at(60), label(80), 1, 'a')
    left, _middle, right = store.stretches(ROUTE)
    # Before the time nothing of it is known, up to its last instant, and the time stays inside the piece.
    assert left['known_ns'] == at(50) - 1 and right['known_ns'] == at(70)
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


def test_roadworks_cut_no_stretch_and_share_the_revision_of_the_drive(store):
    store.apply(ROUTE, at(0), at(100), label(80, 'confirmed'), 0, 'a')
    store.tap_works(ROUTE, 'start', at(50), END, 1, 'a')
    assert spans(store) == [(0, 100, 80, 'confirmed')] and store.revision(ROUTE) == 2
    # Two tabs: a write from before the tap is refused, and so is a tap from before a write.
    with pytest.raises(SpeedReviewError) as refused:
        store.apply(ROUTE, at(0), at(100), label(120), 1, 'a')
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
    assert view['revision'] == 1 and view['examined_s'] == 0
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


def test_an_agent_has_no_way_to_write_a_speed():
    assert 'speed_review' not in inspect.getsource(review_mcp)


def test_every_control_page_shows_the_sign_in_page_without_a_session():
    assert '/speed-review' in CONTROL_TABS
    assert [path for path in CONTROL_TABS if not PAGE_RE.fullmatch(path)] == []
