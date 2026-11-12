"""The site's road users endpoints (nnslr_tools.road_users behind review.py): the drives with proposed crossings, one
drive's proposals placed in its video, and a person's verdict with its revision check, behind the site's login.

Every drive, box and verdict here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools import road_users  # noqa: E402
from review_helpers import ROUTE, TOKEN, RunningServer, json_of, make_report, sign_in  # noqa: E402

EVENT = {'id': '0/0.5@400', 'kind': 'vehicle', 'class': 'car', 'from': '0/0.5', 'to': '0/0.75', 'score': 4.2,
         'boxes': [['0/0.5', 350.0, 300.0, 450.0, 380.0], ['0/0.75', 420.0, 300.0, 520.0, 380.0]]}


def proposed(root: Path) -> None:
    """A drive whose proposals hold one crossing: the endpoints read the file the rule writes and nothing else (the
    readings of the detector are not needed to review)."""
    road_users.save_proposals(root, ROUTE, {
        'version': road_users.RULE_VERSION, 'readings_version': road_users.VERSION, 'detector': road_users.DETECTOR,
        'segments': {}, 'yaw': 'yaw_dev', 'events': [EVENT], 'unjudged': {}, 'made': '2026-11-12T20:00:00+00:00'})


def test_the_drives_and_a_drive_s_proposals_are_read_behind_the_login(tmp_path):
    make_report(tmp_path)
    proposed(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        # Nobody signed in and no token: refused.
        assert server.request('GET', '/api/road-users/drives')[0] in (401, 403)
        token = {'X-Review-Token': TOKEN}
        status, _response, data = server.request('GET', '/api/road-users/drives', headers=token)
        assert status == 200
        [drive] = json_of(data)['drives']
        assert drive['route'] == ROUTE and drive['proposed'] == 1 and drive['reviewed'] == 0
        status, _response, data = server.request('GET', f'/api/road-users/drives/{ROUTE}', headers=sign_in(server))
        view = json_of(data)
        assert status == 200 and view['revision'] == 0 and view['frame_size'] == [1344, 760]
        assert view['fps'] == 20 and view['readings_per_second'] == 4
        [event] = view['events']
        assert event['id'] == EVENT['id'] and event['kind'] == 'vehicle' and event['score'] == 4.2
        assert event['from']['segment'] == 0 and event['from']['frame'] == 10 and event['from']['seconds'] == 0.5
        assert [box['box'] for box in event['boxes']] == [box[1:] for box in EVENT['boxes']]
        assert event['verdict'] is None
        # A drive without proposals, an id that is no drive.
        status, _response, data = server.request('GET', '/api/road-users/drives/00000002--abcdef0123', headers=token)
        assert (status, json_of(data)['error']) == (404, 'no_proposals')
        assert server.request('GET', '/api/road-users/drives/nothing', headers=token)[0] == 404
    finally:
        server.close()


def test_only_a_person_signed_in_gives_a_verdict_and_an_old_revision_is_refused(tmp_path):
    make_report(tmp_path)
    proposed(tmp_path)
    path = f'/api/road-users/drives/{ROUTE}/verdicts'
    body = {'event': EVENT['id'], 'verdict': 'crossing', 'base_revision': 0}
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)

    def post(headers, sent=body):
        status, _response, data = server.request('POST', path, body=sent, headers=headers)
        return status, json_of(data)

    try:
        # The token alone is no person: an agent never gives a verdict.
        status, answer = post({'X-Review-Token': TOKEN})
        assert (status, answer['error']) == (403, 'session_required')
        session = sign_in(server)
        assert post({**session, 'Origin': 'http://elsewhere.example'})[1]['error'] == 'cross_origin'
        assert post({**session, 'Content-Type': 'text/plain'})[0] == 415
        server.app.panel.actions_enabled = False
        assert post(session)[1]['error'] == 'actions_disabled'
        server.app.panel.actions_enabled = True
        assert road_users.verdicts_revision(tmp_path, ROUTE) == 0
        status, answer = post(session)
        assert status == 200 and answer['revision'] == 1 and answer['event'] == EVENT['id']
        assert answer['verdict']['verdict'] == 'crossing' and answer['verdict']['reviewer'] == 'admin'
        # A second write from the same read is refused, and nothing changes.
        status, answer = post(session, {**body, 'verdict': 'not_crossing'})
        assert (status, answer['error']) == (409, 'stale_revision')
        assert road_users.load_verdicts(tmp_path, ROUTE)[EVENT['id']]['verdict'] == 'crossing'
        # What does not fit.
        status, answer = post(session, {**body, 'event': 'nope', 'base_revision': 1})
        assert (status, answer['error']) == (404, 'unknown_event')
        status, answer = post(session, {**body, 'verdict': 'maybe', 'base_revision': 1})
        assert status == 400
        status, answer = post(session, {**body, 'base_revision': '1'})
        assert (status, answer['error']) == (400, 'invalid_base_revision')
        # His verdict shows on the drive and in the list; clear takes it back.
        token = {'X-Review-Token': TOKEN}
        view = json_of(server.request('GET', f'/api/road-users/drives/{ROUTE}', headers=token)[2])
        assert view['revision'] == 1 and view['events'][0]['verdict']['verdict'] == 'crossing'
        [listed] = json_of(server.request('GET', '/api/road-users/drives', headers=token)[2])['drives']
        assert listed['reviewed'] == 1 and listed['crossing'] == 1
        status, answer = post(session, {**body, 'verdict': 'clear', 'base_revision': 1})
        assert status == 200 and answer['revision'] == 2 and answer['verdict'] is None
        assert road_users.load_verdicts(tmp_path, ROUTE) == {}
        view = json_of(server.request('GET', f'/api/road-users/drives/{ROUTE}', headers=token)[2])
        assert view['revision'] == 2 and view['events'][0]['verdict'] is None
        # The verdicts are a file of lines that only grows: every write is there, with its reviewer and time.
        lines = road_users._verdict_lines(tmp_path, ROUTE)
        assert [line['verdict'] for line in lines] == ['crossing', 'clear']
        assert all(line['reviewer'] == 'admin' and line['at'] for line in lines)
        assert server.request('POST', f'/api/road-users/drives/{ROUTE}/nothing', body=body, headers=session)[0] == 404
    finally:
        server.close()
