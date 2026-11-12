"""The road users of the brain's training (nnslr_tools.road_users): the crossing rule on synthetic tracks (a vehicle
crossing, an oncoming one, a parked one while we drive straight, a parked one while we turn), what cannot be judged,
a person's verdicts, and the answers of users_head (an unread frame or drive is weight 0, never "nobody there").
numpy comes in only inside the tests: collecting them must leave it out (tests/cli/test_import_isolation.py)."""
import json
import math
import os

import pytest

NS = 1_000_000_000
ROUTE = '00000001--abcdef0123'
FOCAL = 1141.5
# The readings start a second into the drive: the yaw is known from 0.
START_NS = NS


def _users():
    from nnslr_tools import road_users
    return road_users


def car_box(x_m, z_m, *, height_m=1.5, length_m=4.0, centre_y=380.0, jitter=0.0):
    """A vehicle X m to the right and Z m ahead, as the road camera at 1344 x 760 sees it."""
    cx = 672 + FOCAL * x_m / z_m + jitter
    h = FOCAL * height_m / z_m
    w = FOCAL * length_m / z_m
    return [cx - w / 2, centre_y - h / 2, cx + w / 2, centre_y + h / 2]


def turned_box(bearing, z_m, *, height_m=1.5, length_m=4.0, centre_y=380.0):
    """A box at this bearing (radians, right positive) Z m ahead."""
    cx = 672 + FOCAL * math.tan(bearing)
    h = FOCAL * height_m / z_m
    w = FOCAL * length_m / z_m
    return [cx - w / 2, centre_y - h / 2, cx + w / 2, centre_y + h / 2]


def readings_of(rows, *, frames=None, segment=0):
    """{segment: reading data} as load_readings gives it: rows is {row: [(class name, score, box), ...]}."""
    np = __import__('numpy')
    users = _users()
    reading, cls, score, box = [], [], [], []
    for row, boxes in sorted(rows.items()):
        for name, value, coordinates in boxes:
            reading.append(row)
            cls.append(users.CLASSES.index(name))
            score.append(value)
            box.append(coordinates)
    count = frames if frames is not None else (max(rows) + 1 if rows else 0)
    return {segment: {'meta': {'video': [1000, 1]}, 'reading': np.array(reading, dtype=np.int32),
                      'cls': np.array(cls, dtype=np.uint8), 'score': np.array(score, dtype=np.float32),
                      'box': np.array(box, dtype=np.float32).reshape(-1, 4), 'frames': count}}


def exposures_of(frames, segment=0, *, missing=()):
    """The exposure of every reading's frame, 0.25 s apart from START_NS; rows in missing have none."""
    return {segment: {row * 5: START_NS + row * NS // 4 for row in range(frames) if row not in missing}}


def straight(rate=0.0):
    return {'series': {'yaw_dev': [[0, rate]]}}


def proposals_of(rows, frames, merged, **options):
    users = _users()
    return users.make_proposals(readings_of(rows, frames=frames), exposures_of(frames, **options), merged)


# --- the crossing rule ----------------------------------------------------------------------------------------------


def test_a_car_crossing_from_the_left_is_proposed_with_its_speed_across():
    """A car 20 m ahead crossing from the left at 5 m/s, we drive straight: one event, a vehicle, about 5 m/s."""
    rows = {row: [('car', 0.95, car_box(-8 + 5 * row / 4, 20))] for row in range(9)}
    proposals = proposals_of(rows, 9, straight())
    assert proposals['yaw'] == 'yaw_dev'
    [event] = proposals['events']
    assert event['kind'] == 'vehicle' and event['class'] == 'car'
    assert event['score'] == pytest.approx(5.0, abs=0.2)
    # From the reading where its track is old enough to judge (0.5 s, 3 readings) to its last: its boxes along it.
    assert event['from'] == '0/0.5' and event['to'] == '0/2'
    assert [box[0] for box in event['boxes']] == [f'0/{row / 4:g}' for row in range(2, 9)]
    assert event['id'] == f"0/0.5@{int(round((event['boxes'][0][1] + event['boxes'][0][3]) / 2))}"


def test_an_oncoming_car_is_not_proposed():
    """A car in the other lane, 3.5 m to the left, coming at us at 25 m/s closing speed: its offset in its own heights
    stays the same, a few pixels of jitter."""
    rows = {row: [('car', 0.9, car_box(-3.5, 60 - 25 * row / 4, jitter=3.0 * (-1) ** row))] for row in range(8)}
    proposals = proposals_of(rows, 8, straight())
    assert proposals['events'] == []


def test_a_parked_car_on_the_right_is_not_proposed_while_we_drive_straight():
    rows = {row: [('car', 0.9, car_box(3.0, 40 - 14 * row / 4)), ('truck', 0.8, car_box(3.5, 50 - 14 * row / 4,
                                                                                         height_m=3.0))]
            for row in range(8)}
    proposals = proposals_of(rows, 8, straight(0.4))
    assert proposals['events'] == []
    assert not any(key for key, mask in proposals['unjudged'].items() if mask == (1 << 8) - 1)


def test_a_turn_moves_a_parked_car_but_the_yaw_is_taken_off():
    """We turn left at 8 deg/s with a car parked at a junction on the left, 30 m ahead: the picture moves right,
    towards the centre; without the yaw it would be a crossing, with it it is not."""
    rate = 8.0
    rows = {}
    for row in range(9):
        turned = math.radians(rate) * (START_NS + row * NS / 4) / NS
        rows[row] = [('car', 0.9, turned_box(math.radians(-30) + turned, 30))]
    assert all(box[2] < 672 for boxes in rows.values() for _name, _score, box in boxes)
    assert proposals_of(rows, 9, straight(rate))['events'] == []
    # The same pictures with a yaw of 0 in the logs: proposed. The compensation is what keeps it out.
    assert proposals_of(rows, 9, straight(0.0))['events'] != []


def test_a_strong_turn_is_not_judged():
    users = _users()
    rows = {row: [('car', 0.95, car_box(-8 + 5 * row / 4, 20))] for row in range(9)}
    proposals = proposals_of(rows, 9, straight(users.MAX_YAW_DEG_S + 5))
    assert proposals['events'] == []
    assert set(proposals['unjudged']) == {f'0/{row / 4:g}' for row in range(9)}
    assert set(proposals['unjudged'].values()) == {(1 << users.COLUMNS) - 1}


def test_no_yaw_in_the_logs_and_a_reading_without_its_exposure_are_not_judged():
    users = _users()
    rows = {row: [('car', 0.95, car_box(-8 + 5 * row / 4, 20))] for row in range(9)}
    nothing = proposals_of(rows, 9, {'series': {'yaw_car': [[0, 0.0]]}})
    assert nothing['yaw'] is None and nothing['events'] == [] and len(nothing['unjudged']) == 9
    gap = proposals_of(rows, 9, straight(), missing=(4,))
    assert gap['unjudged']['0/1'] == (1 << users.COLUMNS) - 1


def test_a_young_track_leaves_its_column_unjudged():
    """The first readings of a track cannot say whether it crosses: its column is not judged there, the others are."""
    users = _users()
    box = car_box(3.0, 20)
    rows = {row: [('car', 0.9, box)] for row in range(5)}
    proposals = proposals_of(rows, 5, straight())
    column = users.column_of(box)
    assert proposals['unjudged'] == {'0/0': 1 << column, '0/0.25': 1 << column}


def test_the_bonnet_of_our_car_is_never_tracked():
    users = _users()
    rows = {row: [('car', 0.9, [180.0 + 60 * (-1) ** row, 672.0, 1180.0 - 80 * (-1) ** row, 755.0])] for row in range(8)}
    proposals = proposals_of(rows, 8, straight())
    assert proposals['events'] == [] and proposals['unjudged'] == {}
    assert users.track(users._timeline(readings_of(rows), exposures_of(8))) == []


def test_two_crossings_of_one_track_far_apart_are_two_events():
    """Readings of one track said crossing with a long hole between are two events; a lone reading is none."""
    users = _users()
    timeline = [{'key': f'0/{row / 4:g}', 't': row / 4} for row in range(20)]
    tracks = [[{'at': at, 't': at / 4, 'box': [600.0, 300.0, 700.0, 380.0], 'cls': 3} for at in range(20)]]
    events = users.events_of(timeline, tracks, {0: {2: 2.0, 3: 2.5, 5: 3.0, 12: 2.0, 17: 1.6}})
    assert [(event['from'], event['to'], event['score']) for event in events] == [('0/0.5', '0/1.25', 3.0)]
    events = users.events_of(timeline, tracks, {0: {2: 2.0, 3: 2.5, 12: 2.0, 13: 1.6}})
    assert [(event['from'], event['to']) for event in events] == [('0/0.5', '0/0.75'), ('0/3', '0/3.25')]


def test_the_yaw_integral_and_its_unknown_stretches():
    users = _users()
    yaw = users.Yaw([[0, None], [2 * NS, 10.0], [4 * NS, -10.0]])
    assert yaw.turned(NS, 3 * NS) is None
    assert math.degrees(yaw.turned(2 * NS, 3 * NS)) == pytest.approx(10.0)
    assert math.degrees(yaw.turned(3 * NS, 5 * NS)) == pytest.approx(0.0)
    assert yaw.turned(-NS, NS) is None and yaw.turned(None, NS) is None


# --- the files, the verdicts and the answers -------------------------------------------------------------------------


def _video(root, segment=0):
    path = root / 'raw' / 'routes' / ROUTE / str(segment) / 'fcamera.hevc'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\0' * 1000)
    return path


def _save(root, rows, frames, segment=0, **meta):
    users = _users()
    video = _video(root, segment)
    data = readings_of(rows, frames=frames, segment=segment)[segment]
    stamp = users.video_stamp(video)
    users.save_segment(users.segment_path(root, ROUTE, segment),
                       {'version': users.VERSION, 'detector': users.DETECTOR, 'score_floor': users.SCORE_FLOOR,
                        'video_step': 5, 'frame_size': [1344, 760], 'video': stamp, 'frames': frames, **meta},
                       data['reading'], data['cls'], data['score'], data['box'])


def _frames(*pairs):
    return [{'segment': segment, 'decoded': decoded} for segment, decoded in pairs]


PERSON = [100.0, 300.0, 130.0, 360.0]


def test_the_detector_s_persons_by_column_and_the_four_readings_of_a_second(tmp_path):
    users = _users()
    _save(tmp_path, {1: [('person', 0.9, PERSON)], 2: [('bicycle', 0.8, [700.0, 300.0, 740.0, 350.0])]}, 4)
    answers = users.drive_answers(tmp_path, ROUTE, _frames((0, 0), (0, 5), (0, 10), (0, 15)))
    person, wheeler = users.KINDS.index('person'), users.KINDS.index('two_wheeler')
    assert answers['read'].tolist() == [True] * 4
    assert answers['presence'][:, person].sum(axis=1).tolist() == [0, 1, 0, 0]
    assert answers['presence'][1, person, users.column_of(PERSON)] == 1
    assert answers['height'][1, person, users.column_of(PERSON)] == pytest.approx(60 / 760)
    assert answers['presence'][2, wheeler, 4] == 1
    # Read and nobody there: weight 1, presence 0.
    assert answers['weight'][0, person].tolist() == [1.0] * 8 and answers['presence'][0].sum() == 0


def test_read_but_below_the_bar_is_nobody_there(tmp_path):
    users = _users()
    _save(tmp_path, {0: [('person', 0.4, PERSON), ('person', 0.9, [100.0, 300.0, 110.0, 320.0])]}, 1)
    answers = users.drive_answers(tmp_path, ROUTE, _frames((0, 0)))
    assert answers['weight'][0, users.KINDS.index('person')].tolist() == [1.0] * 8
    assert answers['presence'].sum() == 0


@pytest.mark.parametrize('case', ['no file', 'past the frames read', 'stale video', 'other detector', 'other version',
                                  'off the readings', 'other segment'])
def test_what_the_detector_has_not_read_is_weight_0_never_nobody_there(tmp_path, case):
    users = _users()
    if case != 'no file':
        meta = {'other detector': {'detector': 'yolo'}, 'other version': {'version': users.VERSION + 1}}.get(case, {})
        _save(tmp_path, {0: [('person', 0.9, PERSON)]}, 2, **meta)
    if case == 'stale video':
        video = _video(tmp_path)
        video.write_bytes(b'\0' * 1001)
    frame = {'past the frames read': (0, 10), 'off the readings': (0, 3), 'other segment': (1, 0)}.get(case, (0, 0))
    answers = users.drive_answers(tmp_path, ROUTE, _frames(frame))
    assert answers['weight'].sum() == 0 and answers['presence'].sum() == 0 and not answers['read'].any()


def _proposed(tmp_path, *, unjudged=None):
    """A drive read, with one proposed crossing (its boxes at 0/0.5 and 0/0.75) and the rule's unjudged columns."""
    users = _users()
    _save(tmp_path, {}, 8)
    readings = users.load_readings(tmp_path, ROUTE)
    event = {'id': '0/0.5@400', 'kind': 'vehicle', 'class': 'car', 'from': '0/0.5', 'to': '0/0.75', 'score': 4.2,
             'boxes': [['0/0.5', 350.0, 300.0, 450.0, 380.0], ['0/0.75', 420.0, 300.0, 520.0, 380.0]]}
    proposals = {'version': users.RULE_VERSION, 'readings_version': users.VERSION, 'detector': users.DETECTOR,
                 'segments': users.segments_stamp(readings), 'yaw': 'yaw_dev', 'events': [event],
                 'unjudged': unjudged or {}, 'made': 'now'}
    users.save_proposals(tmp_path, ROUTE, proposals)
    return event


EIGHT = _frames(*((0, decoded) for decoded in range(0, 40, 5)))


def test_vehicles_are_weight_0_until_a_person_reviewed_every_proposal(tmp_path):
    users = _users()
    event = _proposed(tmp_path)
    vehicle = users.KINDS.index('vehicle')
    answers = users.drive_answers(tmp_path, ROUTE, EIGHT)
    assert answers['weight'][:, vehicle].sum() == 0 and not answers['reviewed']
    users.add_verdict(tmp_path, ROUTE, event, 'not_crossing', 'cristian')
    answers = users.drive_answers(tmp_path, ROUTE, EIGHT)
    assert answers['reviewed'] and answers['weight'][:, vehicle].sum() == 8 * 8
    assert answers['presence'][:, vehicle].sum() == 0


def test_a_confirmed_crossing_is_presence_at_its_boxes(tmp_path):
    users = _users()
    event = _proposed(tmp_path, unjudged={'0/0': 1 << 2})
    users.add_verdict(tmp_path, ROUTE, event, 'crossing', 'cristian')
    answers = users.drive_answers(tmp_path, ROUTE, EIGHT)
    vehicle = users.KINDS.index('vehicle')
    presence = answers['presence'][:, vehicle]
    # Both boxes have their centres in column 2 (400 and 470 px).
    assert presence.sum() == 2 and presence[2, 2] == 1 and presence[3, 2] == 1
    assert answers['height'][2, vehicle, 2] == pytest.approx(80 / 760)
    # Reviewed: 0 elsewhere, except where the rule could not judge.
    assert answers['weight'][0, vehicle].tolist() == [1, 1, 0, 1, 1, 1, 1, 1]
    assert answers['weight'][1:, vehicle].min() == 1 and answers['confirmed'] == 1


def test_a_confirmed_crossing_stays_when_the_proposals_change_but_the_rest_is_unreviewed(tmp_path):
    users = _users()
    event = _proposed(tmp_path)
    users.add_verdict(tmp_path, ROUTE, event, 'crossing', 'cristian')
    # Read again (one reading more): the proposals are not current any more.
    _save(tmp_path, {}, 9)
    answers = users.drive_answers(tmp_path, ROUTE, EIGHT)
    vehicle = users.KINDS.index('vehicle')
    assert not answers['reviewed']
    assert answers['presence'][:, vehicle].sum() == 2 and answers['weight'][:, vehicle].sum() == 2


def test_verdicts_last_wins_clear_takes_back_and_a_net_reviewer_is_never_read(tmp_path):
    users = _users()
    event = _proposed(tmp_path)
    users.add_verdict(tmp_path, ROUTE, event, 'crossing', 'cristian')
    users.add_verdict(tmp_path, ROUTE, event, 'not_crossing', 'cristian')
    assert users.load_verdicts(tmp_path, ROUTE)[event['id']]['verdict'] == 'not_crossing'
    users.add_verdict(tmp_path, ROUTE, event, 'clear', 'cristian')
    assert users.load_verdicts(tmp_path, ROUTE) == {}
    path = users.verdicts_path(tmp_path, ROUTE)
    with path.open('a') as handle:
        handle.write(json.dumps({'event': event['id'], 'verdict': 'crossing', 'boxes': event['boxes'],
                                 'reviewer': 'net/applicability', 'kind': 'vehicle'}) + '\n')
        handle.write('not json\n')
    assert users.load_verdicts(tmp_path, ROUTE) == {}
    with pytest.raises(users.RoadUsersError):
        users.add_verdict(tmp_path, ROUTE, event, 'crossing', 'net/applicability')
    with pytest.raises(users.RoadUsersError):
        users.add_verdict(tmp_path, ROUTE, event, 'maybe', 'cristian')


def test_a_verdict_on_an_old_revision_is_refused(tmp_path):
    users = _users()
    event = _proposed(tmp_path)
    assert users.verdicts_revision(tmp_path, ROUTE) == 0
    saved = users.save_verdict(tmp_path, ROUTE, {'event': event['id'], 'verdict': 'crossing', 'base_revision': 0}, 'cristian')
    assert saved['revision'] == 1 and saved['verdict']['verdict'] == 'crossing'
    with pytest.raises(users.RoadUsersError) as refused:
        users.save_verdict(tmp_path, ROUTE, {'event': event['id'], 'verdict': 'not_crossing', 'base_revision': 0}, 'cristian')
    assert refused.value.status == 409
    with pytest.raises(users.RoadUsersError) as unknown:
        users.save_verdict(tmp_path, ROUTE, {'event': 'nope', 'verdict': 'crossing', 'base_revision': 1}, 'cristian')
    assert unknown.value.status == 404


def test_the_review_s_view_of_a_drive_places_every_box_in_the_video(tmp_path):
    users = _users()
    event = _proposed(tmp_path)
    index = tmp_path / 'derived' / 'video' / f'{ROUTE}.json'
    index.parent.mkdir(parents=True, exist_ok=True)
    from nnslr_tools import drive_video
    source = drive_video._signature(drive_video.minutes(tmp_path, ROUTE))
    index.write_text(json.dumps({'segments': [{'segment': 0, 'start': 0.0, 'frames': 1200}], 'source': source}))
    (tmp_path / 'derived' / 'video' / f'{ROUTE}.mp4').write_bytes(b'')
    view = users.drive_view(tmp_path, ROUTE)
    assert view['frame_size'] == [1344, 760] and view['revision'] == 0 and view['current']
    [shown] = view['events']
    assert shown['from'] == {'key': '0/0.5', 'segment': 0, 'frame': 10, 'seconds': 0.5, 'video_seconds': 0.5}
    assert shown['boxes'][1]['box'] == event['boxes'][1][1:] and shown['boxes'][1]['frame'] == 15
    assert shown['verdict'] is None
    [listed] = users.drives_list(tmp_path)
    assert listed == {'route': ROUTE, 'proposed': 1, 'reviewed': 0, 'crossing': 0, 'made': 'now', 'yaw': 'yaw_dev'}


def test_a_drive_id_that_is_no_drive_is_refused(tmp_path):
    users = _users()
    for route in ('../etc', '00000001--abcdef0123/..', ''):
        with pytest.raises(users.RoadUsersError):
            users.verdicts_path(tmp_path, route)
        with pytest.raises(users.RoadUsersError):
            users.segment_path(tmp_path, route, 0)


def test_the_check_without_a_ticked_frame_says_so(tmp_path):
    users = _users()
    found = users.check(tmp_path, truth=[])
    assert found['frames'] == 0 and 'nothing to check yet' in users.check_text(found)


def test_the_check_counts_found_and_false_boxes_by_size(tmp_path):
    users = _users()
    _save(tmp_path, {0: [('person', 0.9, PERSON), ('car', 0.9, [600.0, 300.0, 700.0, 400.0]),
                         ('car', 0.9, [900.0, 300.0, 960.0, 350.0])]}, 1)
    truth = [{'route': ROUTE, 'segment': 0, 'decoded': 0, 'ignore': [],
              'users': [('person', PERSON), ('person', [400.0, 300.0, 410.0, 320.0]),
                        ('vehicle', [600.0, 300.0, 700.0, 400.0])]}]
    found = users.check(tmp_path, truth=truth)
    assert found['frames'] == 1 and found['not_read'] == 0
    assert found['by_kind']['person']['48-96'] == {'truth': 1, 'found': 1, 'false': 0, 'recall': 1.0}
    # A person below the size rule is neither missed nor false.
    assert found['by_kind']['person']['24-48']['truth'] == 0
    assert found['by_kind']['vehicle']['96+'] == {'truth': 1, 'found': 1, 'false': 0, 'recall': 1.0}
    assert found['by_kind']['vehicle']['48-96']['false'] == 1


def test_read_segment_saves_what_the_detector_says(tmp_path, monkeypatch):
    """The worker's save, with the detector replaced: the readings, the meta that keeps them current, NVDEC first."""
    users = _users()
    video = _video(tmp_path)
    calls = []

    def detect(path, *, nvdec=True, limit=None):
        calls.append(nvdec)
        return {'frames': 0, 'reading': [], 'cls': [], 'score': [], 'box': [], 'seconds': 0.0} if nvdec else \
            {'frames': 3, 'reading': [1], 'cls': [0], 'score': [0.9], 'box': [PERSON], 'seconds': 0.1}

    monkeypatch.setattr(users, 'detect_video', detect)
    monkeypatch.setattr(users, '_detector', lambda: {'name': 'card'})
    done = users.read_segment((ROUTE, 0, str(video), str(users.segment_path(tmp_path, ROUTE, 0))))
    assert calls == [True, False] and done['frames'] == 3 and not done['nvdec']
    readings = users.load_readings(tmp_path, ROUTE)
    assert readings[0]['frames'] == 3 and readings[0]['box'].tolist() == [PERSON]
    assert users.unread(tmp_path, ROUTE) == []
    assert not list((tmp_path / 'derived' / 'road_users' / ROUTE).glob('*.part'))


def test_no_card_stops_never_the_cpu(monkeypatch, tmp_path):
    users = _users()
    monkeypatch.setitem(os.environ, 'CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(RuntimeError, match='never on the CPU'):
        users.read_drives(tmp_path, [ROUTE])
