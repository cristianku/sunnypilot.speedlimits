"""The machine of the outside reviewers: what a drive takes there, what comes back and what is refused. Both machines
on this disk, the copies with the real rsync; the drive, the people and their work are made up."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from nnslr_tools import drive_review, drive_video, junctions, review_auth, review_box, road_types, video_frames, sign_distance, speed_review
from nnslr_tools.drive_logs import cache_dir
from nnslr_tools.drive_extract import EXTRACT_VERSION
from nnslr_tools.gps import load_positions
from nnslr_tools.review_access import PROXY_SECRET_HEADER, Assignments
from nnslr_tools.review_auth import auth_dir, write_private
from nnslr_tools.review_box import BoxError, own_dir, sent_path, sync, tick
from nnslr_tools.review_db import ReviewStore, db_path
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report

GUEST = 'guest/ana'
LATITUDE, LONGITUDE = 47.3769, 8.5417
LABEL = {'kph': 80, 'state': 'limit', 'known_ns': None, 'note': ''}


class Pair:
    """This server, and a box in a folder next to it."""

    def __init__(self, base: Path):
        self.root, self.box = base / 'server', base / 'box'
        self.data = self.box / 'data'
        self.root.mkdir()
        self.data.mkdir(parents=True)
        write_private(own_dir(self.root) / 'link.json', json.dumps({'remote': str(self.box)}))
        self.people = Assignments(auth_dir(self.root))
        self.there = Assignments(auth_dir(self.data))

    def round(self):
        """A sync, then the tick the box runs on its own."""
        state = sync(self.root, log=lambda _: None)
        tick(self.data, self.box, log=lambda _: None)
        return state

    def give(self, kind='signs', name='ana', route=ROUTE):
        self.people.add_person(name)
        self.people.set_work(name, route, kind, True)


def extract(tmp_root: Path, route: str, segment: int, fixes: list[list]) -> Path:
    path = cache_dir(tmp_root, route) / f'{segment}.{"0" * 16}.v{EXTRACT_VERSION}.json.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({'gps': fixes, 'series': {'road_name': [[0, 'Bahnhofstrasse']], 'speed': [[0, 9]]}}).encode()))
    return path


def fake_drive(root: Path, route: str = ROUTE) -> None:
    """A report with its frames, reduced logs with a track, a junction, and the raw logs that must not travel."""
    make_report(root, route=route)
    bearing = [[index * 10 ** 9, LATITUDE + index * 1e-4, LONGITUDE + (index ** 2) * 1e-5, 1_700_000_000 + index]
               for index in range(30)]
    extract(root, route, 0, bearing)
    video = root / 'raw/routes' / route / '0' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'road video')
    video.with_name('ecamera.hevc').write_bytes(b'wide video')
    (video.parent / 'rlog.zst').write_bytes(b'the whole log')
    gps = root / 'derived/gps' / route / 'positions.jsonl'
    gps.parent.mkdir(parents=True)
    gps.write_text(json.dumps({'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': LATITUDE, 'longitude': LONGITUDE, 'speed_mps': 12.0,
                               'bearing_deg': 90.0, 'country': 'CH', 'country_from': 'gps'}) + '\n')
    roads = root / f'derived/road_types/{route}.jsonl'
    roads.parent.mkdir(parents=True)
    roads.write_text(json.dumps({'frame_key': '0/0', 'road_type': 'town', 'way_id': 4242424242}) + '\n')
    place = root / f'derived/junctions/{route}.json'
    place.parent.mkdir(parents=True)
    place.write_text(json.dumps({'junctions': [{'frame_key': '0/1', 'latitude': LATITUDE, 'longitude': LONGITUDE}]}))
    store = drive_review.DriveReviewStore(root)
    del store
    with review_box._open(drive_review.db_path(root)) as connection:
        connection.execute('INSERT INTO scans (route, state, progress, scanner_version, extract_version, files, summary) '
                           'VALUES (?, ?, 1, 1, 1, ?, ?)',
                           (route, 'done', '{}', json.dumps({'gps': [[LATITUDE, LONGITUDE]], 'civil_ms': 5})))


def confirm(root: Path, reviewer=GUEST, frame_key='0/0', kind='human', channel='web', base=None, gps=None) -> dict:
    payload = {'reviewer': reviewer, 'detections': [], 'frame_key': frame_key, 'route_id': ROUTE, 'gps': gps}
    return ReviewStore(db_path(root)).save_review(route_id=ROUTE, run_id=RUN_ID, frame_key=frame_key, segment_index=0,
                                                  output_index=int(frame_key.split('/')[1]), payload=payload,
                                                  reviewer=reviewer, client='test', base_revision=base,
                                                  reviewer_kind=kind, channel=channel)


def reviews(root: Path) -> dict:
    with review_box._open(db_path(root)) as connection:
        return {row['frame_key']: dict(row) for row in connection.execute('SELECT * FROM frame_reviews WHERE route_id = ?', (ROUTE,))}


def history(root: Path) -> list:
    with review_box._open(db_path(root)) as connection:
        return [row['revision'] for row in connection.execute('SELECT revision FROM frame_review_history WHERE route_id = ?', (ROUTE,))]


@pytest.fixture(autouse=True)
def made_up_media(monkeypatch):
    """The rules of a frame and the making of a video have their own tests: these frames and minutes are made up."""
    monkeypatch.setattr(review_box, 'validate_reviewed_frame', lambda frame, root: None)
    monkeypatch.setattr(drive_video, 'index', lambda root, route: {'made': True})


@pytest.fixture
def pair(tmp_path):
    made = Pair(tmp_path)
    tick(made.data, made.box, log=lambda _: None)
    fake_drive(made.root)
    return made


def walk_text(folder: Path) -> str:
    found = []
    for path in folder.rglob('*'):
        if path.is_file():
            data = path.read_bytes()
            found.append(gzip.decompress(data).decode() if path.suffix == '.gz' else data.decode('utf-8', 'replace'))
    return '\n'.join(found)


def test_a_drive_arrives_without_its_places_and_raw_logs(pair):
    pair.give()
    pair.round()
    assert (pair.data / 'raw/routes' / ROUTE / '0' / 'fcamera.hevc').read_bytes() == b'road video'
    assert (pair.data / 'raw/routes' / ROUTE / '0' / 'ecamera.hevc').read_bytes() == b'wide video'
    assert (pair.data / 'derived/frames' / ROUTE / '0' / 'frame_00000000.png').is_file()
    assert not (pair.data / 'raw/routes' / ROUTE / '0' / 'rlog.zst').exists()
    assert load_positions(pair.data, ROUTE) == {'0/0': {'frame_key': '0/0', 'country': 'CH', 'country_from': 'gps'}}
    with review_box._open(drive_review.db_path(pair.data)) as connection:
        assert json.loads(connection.execute('SELECT summary FROM scans').fetchone()['summary']) == {'civil_ms': 5}
    text = walk_text(pair.data / 'derived')
    assert 'Bahnhofstrasse' not in text and '47.3' not in text and 'latitude' not in text and 'way_id' not in text
    assert pair.there.work_of('ana') == {ROUTE: ['signs']}


def test_a_saved_review_goes_without_its_position_and_comes_back_with_it(pair):
    place = {'latitude': LATITUDE, 'longitude': LONGITUDE, 'speed_mps': 12.0, 'bearing_deg': 90.0}
    confirm(pair.root, reviewer='cristianku', gps=place)
    pair.give()
    pair.round()
    with review_box._open(db_path(pair.data)) as connection:
        text = ' '.join(row[0] for table in ('frame_reviews', 'frame_review_history')
                        for row in connection.execute(f'SELECT payload FROM {table}'))
    assert 'latitude' not in text
    assert pair.round()['drives'][ROUTE].get('saved') is None
    assert json.loads(reviews(pair.root)['0/0']['payload'])['gps'] == place
    confirm(pair.data, base=1)
    tick(pair.data, pair.box, log=lambda _: None)
    assert pair.round()['drives'][ROUTE]['saved']['frames'] == 1
    assert reviews(pair.root)['0/0']['reviewer'] == GUEST
    assert json.loads(reviews(pair.root)['0/0']['payload'])['gps'] == place


def test_the_moved_track_keeps_its_turns_and_loses_its_place():
    fixes = [[index * 10 ** 9, LATITUDE + index * 1e-4, LONGITUDE + (index ** 2) * 1e-5, index] for index in range(40)]
    moved = review_box.without_places({'gps': fixes}, (LATITUDE, LONGITUDE))['gps']
    before, after = speed_review._turn_rates(fixes), speed_review._turn_rates(moved)
    assert len(before) == len(after) and before
    assert all(math.isclose(a[1], b[1], rel_tol=0.02, abs_tol=1e-3) for a, b in zip(before, after))
    assert max(abs(fix[1]) for fix in moved) < 0.01 and max(abs(fix[2]) for fix in moved) < 0.02


def test_what_a_person_does_comes_back_once(pair):
    pair.give()
    pair.round()
    confirm(pair.data)
    tick(pair.data, pair.box, log=lambda _: None)
    state = pair.round()
    assert reviews(pair.root)['0/0']['reviewer'] == GUEST
    assert history(pair.root) == [1]
    assert state['drives'][ROUTE]['saved']['frames'] == 1
    pair.round()
    assert history(pair.root) == [1]


def test_a_frame_between_the_seconds_takes_its_picture_here_too(pair, monkeypatch):
    made = []
    monkeypatch.setattr(video_frames, 'materialize', lambda root, route, segment, decoded: made.append((segment, decoded)))
    pair.give()
    pair.round()
    confirm(pair.data, frame_key='0/1863')
    tick(pair.data, pair.box, log=lambda _: None)
    assert pair.round()['drives'][ROUTE]['saved']['frames'] == 1
    assert made == [(0, 863)]
    assert reviews(pair.root)['0/1863']['reviewer'] == GUEST
    assert json.loads(reviews(pair.root)['0/1863']['payload'])['gps']['latitude'] == LATITUDE


def test_a_frame_between_the_seconds_without_its_video_here_is_refused(pair, monkeypatch):
    def no_video(*_):
        raise video_frames.VideoFrameError('no video')
    monkeypatch.setattr(video_frames, 'materialize', no_video)
    pair.give()
    pair.round()
    confirm(pair.data, frame_key='0/1863')
    tick(pair.data, pair.box, log=lambda _: None)
    assert pair.round()['drives'][ROUTE]['saved']['refused'] == 1
    assert '0/1863' not in reviews(pair.root)


def test_a_review_of_another_drive_inside_a_package_is_refused(pair):
    pair.give()
    pair.round()
    saved = confirm(pair.data)
    with review_box._open(db_path(pair.data)) as connection:
        payload = json.loads(connection.execute('SELECT payload FROM frame_reviews').fetchone()[0])
        connection.execute('UPDATE frame_reviews SET payload = ?', (json.dumps({**payload, 'route_id': OTHER_ROUTE}),))
        connection.commit()
    tick(pair.data, pair.box, log=lambda _: None)
    assert saved and pair.round()['drives'][ROUTE]['saved']['refused'] == 1
    assert reviews(pair.root) == {}


def test_every_store_comes_back(pair):
    for kind in ('speed', 'distance', 'roads'):
        pair.give(kind)
    pair.round()
    speed_review.SpeedReviewStore(pair.data).apply(ROUTE, 0, 10, LABEL, 0, GUEST)
    line = json.dumps({'reviewer': GUEST, 'sign': 1}) + '\n'
    for path in (sign_distance.marks_path(pair.data, ROUTE), road_types.corrections_path(pair.data, ROUTE),
                 junctions.corrections_path(pair.data, ROUTE)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(line)
    tick(pair.data, pair.box, log=lambda _: None)
    pair.round()
    assert [row['kph'] for row in speed_review.SpeedReviewStore(pair.root).stretches(ROUTE)] == [80]
    assert len(speed_review.SpeedReviewStore(pair.root).history(ROUTE)) == 1
    for path in (sign_distance.marks_path(pair.root, ROUTE), road_types.corrections_path(pair.root, ROUTE),
                 junctions.corrections_path(pair.root, ROUTE)):
        assert path.read_text() == line


@pytest.mark.parametrize('reviewer, kind, channel', [('guest/nobody', 'human', 'web'), ('local', 'human', 'web'),
                                                     (GUEST, 'ai', 'web'), (GUEST, 'human', 'rest')])
def test_a_frame_under_another_name_is_refused(pair, reviewer, kind, channel):
    pair.give()
    pair.round()
    confirm(pair.data, reviewer=reviewer, kind=kind, channel=channel)
    tick(pair.data, pair.box, log=lambda _: None)
    state = pair.round()
    assert reviews(pair.root) == {}
    assert state['drives'][ROUTE]['saved']['refused'] == 1
    assert list((own_dir(pair.root) / 'aside').iterdir())


def test_a_kind_not_given_is_refused(pair):
    pair.give('signs')
    pair.round()
    speed_review.SpeedReviewStore(pair.data).apply(ROUTE, 0, 10, LABEL, 0, GUEST)
    tick(pair.data, pair.box, log=lambda _: None)
    pair.round()
    assert speed_review.SpeedReviewStore(pair.root).stretches(ROUTE) == []


def test_what_was_saved_just_before_the_work_ended_still_counts(pair):
    pair.give()
    pair.round()
    confirm(pair.data)
    pair.people.set_work('ana', ROUTE, 'signs', False)
    pair.round()
    pair.round()
    assert reviews(pair.root)['0/0']['reviewer'] == GUEST


def test_someone_removed_from_the_list_counts_no_more(pair):
    pair.give()
    pair.round()
    confirm(pair.data)
    pair.people.remove_person('ana')
    tick(pair.data, pair.box, log=lambda _: None)
    pair.round()
    assert reviews(pair.root) == {}


def test_changed_on_both_sides_this_server_stays_and_the_other_is_kept(pair):
    pair.give()
    pair.round()
    confirm(pair.data)
    confirm(pair.root, reviewer='Cristian')
    tick(pair.data, pair.box, log=lambda _: None)
    state = pair.round()
    assert reviews(pair.root)['0/0']['reviewer'] == 'Cristian'
    assert state['drives'][ROUTE]['saved']['conflicts'] == 1
    assert GUEST in walk_text(own_dir(pair.root) / 'aside')


def test_a_frame_changed_only_here_is_not_a_conflict(pair):
    pair.give()
    pair.round()
    confirm(pair.data, frame_key='0/1')
    confirm(pair.root, reviewer='Cristian', frame_key='0/2')
    tick(pair.data, pair.box, log=lambda _: None)
    state = pair.round()
    assert sorted(reviews(pair.root)) == ['0/1', '0/2']
    assert state['drives'][ROUTE]['saved'] == {'frames': 1, 'stores': 0, 'conflicts': 0, 'refused': 0}


@pytest.mark.parametrize('package', [
    '[]', 'not json', '{"format": 1, "route": "00000002--abcdef1234"}',
    json.dumps({'format': 1, 'route': ROUTE, 'signs': {'reviews': [{'route_id': ROUTE}], 'history': []},
                'speed': {'stretches': [], 'works': [], 'checked': [], 'history': []},
                'distance': [], 'road_types': [], 'junctions': []})])
def test_a_broken_package_is_refused_and_said(pair, package):
    pair.give()
    pair.round()
    (pair.box / 'outbox' / f'{ROUTE}.json').write_text(package)
    state = sync(pair.root, log=lambda _: None)
    assert state['drives'][ROUTE]['problem']
    assert reviews(pair.root) == {}


def test_a_speed_out_of_range_is_refused(pair):
    pair.give('speed')
    pair.round()
    package = json.loads((pair.box / 'outbox' / f'{ROUTE}.json').read_text())
    package['speed']['stretches'] = [{'route': ROUTE, 'start_ns': 0, 'end_ns': 10, 'kph': 900, 'state': 'limit',
                                      'known_ns': None, 'note': '', 'reviewer': GUEST, 'revision': 1}]
    (pair.box / 'outbox' / f'{ROUTE}.json').write_text(json.dumps(package))
    sync(pair.root, log=lambda _: None)
    assert speed_review.SpeedReviewStore(pair.root).stretches(ROUTE) == []


def test_review_completed_takes_the_results_and_the_work_back(pair):
    pair.give()
    pair.round()
    confirm(pair.data)
    pair.there.complete('ana', ROUTE)
    tick(pair.data, pair.box, log=lambda _: None)
    state = pair.round()
    assert reviews(pair.root)['0/0']['reviewer'] == GUEST
    assert pair.people.work_of('ana') == {}
    assert [item['person'] for item in state['drives'][ROUTE]['completed']] == ['ana']
    assert pair.there.work_of('ana') == {}


def test_a_drive_nobody_holds_leaves_the_box(pair, monkeypatch):
    monkeypatch.setattr(review_box, 'LEAVING_SECONDS', 0)
    pair.give()
    pair.round()
    confirm(pair.data)
    pair.people.set_work('ana', ROUTE, 'signs', False)
    tick(pair.data, pair.box, log=lambda _: None)
    for _ in range(3):
        state = pair.round()
    assert state['drives'] == {}
    assert state['finished'][0]['route'] == ROUTE
    assert not (pair.data / 'derived/frames' / ROUTE).exists()
    assert not (pair.data / 'raw/routes' / ROUTE).exists()
    assert not cache_dir(pair.data, ROUTE).exists()
    assert reviews(pair.data) == {}
    assert not sent_path(pair.root, ROUTE).exists()
    assert reviews(pair.root)['0/0']['reviewer'] == GUEST
    assert (pair.root / 'raw/routes' / ROUTE / '0' / 'rlog.zst').exists()


def test_a_drive_given_again_while_leaving_waits_for_the_removal(pair, monkeypatch):
    monkeypatch.setattr(review_box, 'LEAVING_SECONDS', 0)
    pair.give()
    pair.round()
    pair.people.set_work('ana', ROUTE, 'signs', False)
    sync(pair.root, log=lambda _: None)
    sync(pair.root, log=lambda _: None)
    pair.people.set_work('ana', ROUTE, 'signs', True)
    state = sync(pair.root, log=lambda _: None)
    assert state['drives'][ROUTE].get('removing')
    tick(pair.data, pair.box, log=lambda _: None)
    pair.round()
    state = pair.round()
    assert not state['drives'][ROUTE].get('removing')
    assert (pair.data / 'derived/frames' / ROUTE).exists()


def test_visits_come_here_and_someone_removed_does_not_come_back(pair):
    pair.give()
    pair.round()
    pair.there.seen('ana')
    pair.there.seen('newcomer')
    tick(pair.data, pair.box, log=lambda _: None)
    pair.people.remove_person('ana')
    pair.round()
    names = [person['name'] for person in pair.people.view()['people']]
    assert names == ['newcomer']
    assert pair.people.view()['people'][0]['last_seen']


def test_work_on_a_drive_that_failed_to_go_is_not_told(pair):
    pair.give()
    pair.give(route=OTHER_ROUTE)
    (pair.root / 'derived/frames' / OTHER_ROUTE).mkdir(parents=True)
    original = review_box._send_drive

    def failing(root, remote, route, log):
        if route == OTHER_ROUTE:
            raise RuntimeError('no video')
        original(root, remote, route, log)

    review_box._send_drive = failing
    try:
        with pytest.raises(BoxError):
            sync(pair.root, log=lambda _: None)
    finally:
        review_box._send_drive = original
    tick(pair.data, pair.box, log=lambda _: None)
    assert pair.there.work_of('ana') == {ROUTE: ['signs']}
    assert review_box.read_state(pair.root)['last']['ok'] is False


def test_an_unreachable_box_changes_nothing_and_is_said(pair):
    pair.give()
    write_private(own_dir(pair.root) / 'link.json', json.dumps({'remote': str(pair.box / 'gone')}))
    with pytest.raises(BoxError):
        sync(pair.root, log=lambda _: None)
    state = review_box.read_state(pair.root)
    assert state['drives'] == {} and state['last']['ok'] is False


def test_a_tick_refuses_to_run_on_this_server(pair):
    with pytest.raises(BoxError):
        tick(pair.root, pair.box, log=lambda _: None)
    assert (pair.root / 'derived/frames' / ROUTE).exists()


def test_a_sync_without_a_link_says_so(tmp_path):
    with pytest.raises(BoxError):
        sync(tmp_path, log=lambda _: None)


@pytest.fixture
def reviewer_site(tmp_path, monkeypatch):
    monkeypatch.setattr(review_auth, 'SCRYPT_N', 2 ** 10)
    monkeypatch.setenv('NNSLR_PROXY_IPS', '127.0.0.1')
    monkeypatch.setenv('NNSLR_PROXY_SECRET', 'proxy-secret-0123456789')
    monkeypatch.setenv('NNSLR_PUBLIC_HOSTS', 'reviews.example')
    monkeypatch.setenv('NNSLR_REVIEW_BOX', '1')
    make_report(tmp_path)
    make_report(tmp_path, route=OTHER_ROUTE)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.assignments.add_person('ana')
    server.app.assignments.set_work('ana', ROUTE, 'signs', True)
    server.app.assignments.set_work('ana', ROUTE, 'speed', True)
    yield server
    server.close()


def guest_post(server, body):
    headers = {PROXY_SECRET_HEADER: 'proxy-secret-0123456789', 'X-authentik-username': 'ana', 'X-authentik-groups': 'reviewers',
               'Host': 'reviews.example', 'Origin': 'http://reviews.example', 'Content-Type': 'application/json'}
    return server.request('POST', '/api/reviewer/done', body=json.dumps(body).encode(), headers=headers)


def test_a_reviewer_says_a_review_is_completed(reviewer_site):
    status, _, data = guest_post(reviewer_site, {'route': ROUTE})
    answer = json_of(data)
    assert status == 200 and answer['can_complete'] is True and answer['drives'][0]['completed'] is True
    assert set(reviewer_site.app.assignments.visits()['done']['ana'][ROUTE]) == {'signs', 'speed'}


@pytest.mark.parametrize('body', [{'route': OTHER_ROUTE}, {'route': None}, {}, []])
def test_only_a_drive_of_ones_own_can_be_completed(reviewer_site, body):
    assert guest_post(reviewer_site, body)[0] == 403
    assert reviewer_site.app.assignments.visits()['done'] == {}


def test_the_portal_shows_the_machine(reviewer_site):
    root = reviewer_site.app.root
    write_private(own_dir(root) / 'link.json', '{"remote": "reviewsync@box"}')
    view = review_box.view(root)
    assert view['linked'] is True and view['drives'] == [] and view['last'] is None
