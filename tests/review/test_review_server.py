"""Web review server: token/cookie, routes index, saving frames into the review database."""
from __future__ import annotations

import gzip
import re

import pytest

from nnslr_tools.pages import asset
from nnslr_tools.review import GZIP_MIN, accepts_gzip
from nnslr_tools.review_db import ReviewStore, db_path
from review_helpers import (OTHER_ROUTE, ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report,
                            page_body, sign_in)


@pytest.fixture
def report(tmp_path):
    rows = make_report(tmp_path)
    return tmp_path, rows


@pytest.fixture
def loopback(report):
    root, rows = report
    server = RunningServer(root)
    yield server, root, rows
    server.close()


@pytest.fixture
def tokened(report):
    root, rows = report
    server = RunningServer(root, token=TOKEN, require_token=True)
    yield server, root, rows
    server.close()


def cookie_header(server, value=TOKEN):
    return {'Cookie': f'{server.app.cookie_name}={value}'}


def test_loopback_needs_no_token_and_index_lists_routes_with_counts(loopback, tmp_path):
    server, root, rows = loopback
    make_report(root, route=OTHER_ROUTE, count=2)
    # The route list of Sign review (/routes) leads to Speed review's, which lists the drives of /api/routes.
    status, response, data = server.request('GET', '/routes')
    assert status == 303 and response.getheader('Location') == '/speed-review'
    status, response, data = server.request('GET', '/api/routes')
    routes = json_of(data)['routes']
    assert {ROUTE, OTHER_ROUTE} <= {r['route_id'] for r in routes}
    first = next(r for r in routes if r['route_id'] == ROUTE)
    assert first == {'route_id': ROUTE, 'run_id': RUN_ID, 'frame_count': 3, 'frames_with_proposals': 2,
                     'confirmed_count': 0, 'confirmed_by_ai': 0, 'waiting_for_person': 0, 'suspicious_count': 0,
                     'hint_frames': 0,
                     'url': '/route/' + ROUTE + '/',
                     # No videos or positions in the test: the drive has nothing to say.
                     'drive': {'when': None, 'minutes': None, 'start': None, 'from': None, 'to': None, 'title': None,
                               'full_definition': False},
                     'proposer': 'vietnam', 'dashcam': None}
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0]))
    assert status == 200, data
    routes = json_of(server.request('GET', '/api/routes')[2])['routes']
    first = next(r for r in routes if r['route_id'] == ROUTE)
    # Frame 0/1 has nothing while its confirmed neighbour 0/0 has a speed sign (rule d).
    assert first['confirmed_count'] == 1 and first['suspicious_count'] == 1


def test_without_a_session_pages_show_the_sign_in_and_calls_are_refused(tokened):
    server, root, rows = tokened
    # [site-login] - START
    # The sign-in page is a Vue app: its form is in the login script (web/src/login/LoginApp.vue).
    login_script = asset("js/login.js").encode()
    for path in ['/', f'/route/{ROUTE}/', '/panoramica', '/pannello', '/account']:
        status, response, data = server.request('GET', path)
        assert status in (200, 401) and login_script in data and b'"server": true' not in data, path
    for path in ['/api/routes', f'/route/{ROUTE}/images/00000000.jpg', f'/api/route/{ROUTE}/reviews']:
        status, response, data = server.request('GET', path)
        assert status == 401, path
    # The access token in a cookie (the old way) opens nothing any more.
    assert server.request('GET', '/api/routes', headers=cookie_header(server))[0] == 401
    # An old link with ?token=: back to the same address without it, and no cookie.
    for supplied in ('wrong-token', TOKEN):
        status, response, data = server.request('GET', f'/route/{ROUTE}/?token={supplied}')
        assert status == 303 and response.getheader('Location') == f'/route/{ROUTE}/'
        assert response.getheader('Set-Cookie') is None
    session = sign_in(server)
    status, response, data = server.request('GET', f'/route/{ROUTE}/', headers=session)
    assert status == 200 and b'"server": true' in data
    status, response, data = server.request('GET', f'/route/{ROUTE}/images/00000000.jpg', headers=session)
    assert status == 200 and data == b'jpeg preview'
    # Agents: the access token in a header, as for /api/v1.
    status, response, data = server.request('GET', '/api/routes', headers={'X-Review-Token': TOKEN})
    assert status == 200 and json_of(data)['routes'][0]['route_id'] == ROUTE
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0]))
    assert status == 401
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0]), headers=session)
    assert status == 200, data
    # [site-login] - END


def test_cookies_of_other_web_apps_do_not_hide_the_review_cookie(tokened):
    server, root, rows = tokened
    # Cookies of other apps on the same host come first and are not all valid for SimpleCookie.
    foreign = 'prefs={"theme":"dark"}; abc; path=x; Secure; =empty; other=a b\\c'
    session = sign_in(server)['Cookie'].split('=', 1)[1]
    status, response, data = server.request('GET', '/api/routes', headers={
        'Cookie': f'{foreign}; {server.app.cookie_name}={session}'})
    assert status == 200
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0]), headers={
        'Cookie': f'{foreign}; {server.app.cookie_name}={session}; later=1'})
    assert status == 200, data
    status, response, data = server.request('GET', '/api/routes', headers={
        'Cookie': f'{foreign}; {server.app.cookie_name}=wrong-token-000000000000'})
    assert status == 401
    status, response, data = server.request('GET', '/api/routes', headers={'Cookie': f'{foreign}; nnslr_review={session}'})
    assert status == 401


def test_idle_connections_are_closed_and_their_number_is_bounded(report, monkeypatch):
    import socket
    import time
    from nnslr_tools import review
    root, rows = report
    monkeypatch.setattr(review, 'REQUEST_TIMEOUT_SECONDS', 1.0)
    monkeypatch.setattr(review, 'MAX_CONNECTIONS', 2)
    server = RunningServer(root, token=TOKEN, require_token=True)
    idle = []
    try:
        for number in range(3):
            connection = socket.create_connection(('127.0.0.1', server.port), timeout=5)
            idle.append(connection)
        # The third connection finds no free slot and is closed at once, without a token check.
        started = time.monotonic()
        assert idle[2].recv(1) == b''
        assert time.monotonic() - started < 0.9
        # The first two send nothing and are closed by the server timeout.
        assert idle[0].recv(1) == b''
        assert idle[1].recv(1) == b''
        status = None
        deadline = time.monotonic() + 5
        while status != 401 and time.monotonic() < deadline:
            try:
                status = server.request('GET', '/api/routes')[0]
            except (ConnectionError, OSError):
                time.sleep(0.05)
        assert status == 401
        assert server.request('GET', '/api/routes', headers=sign_in(server))[0] == 200
    finally:
        for connection in idle:
            connection.close()
        server.close()


def test_token_redirect_never_leaves_the_server(tokened):
    server, root, rows = tokened
    for path in [f'//evil.example/?token={TOKEN}', f'/\\evil.example/?token={TOKEN}']:
        status, response, data = server.request('GET', path)
        assert status == 303
        location = response.getheader('Location')
        # A path on this server only: never "//host" (another site) and never a backslash.
        assert location.startswith('/') and not location.startswith('//') and '\\' not in location


def test_save_valid_frame_writes_current_row_and_history(loopback):
    server, root, rows = loopback
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], value_kph=50))
    assert status == 200, data
    saved = json_of(data)
    assert saved['ok'] and saved['confirmed_count'] == 1
    assert saved['review']['confirmed'] is True and saved['review']['revision'] == 1
    assert saved['review']['detections'][0]['value_kph'] == 50
    status, response, data = server.post_frame(ROUTE, '0/1', page_body(rows[1], manual_box=[2, 2, 10, 12]))
    assert status == 200, data
    store = ReviewStore(db_path(root))
    review = store.review(ROUTE, RUN_ID, '0/0')
    assert review['reviewer'] == 'Cristian' and review['confirmed']
    payload = review['payload']
    assert payload['kind'] == 'reviewed_frame' and payload['label_basis'] == 'human_review'
    assert payload['source_proposals'][0]['value_kph'] == 30
    assert payload['detections'][0]['value_kph'] == 50 and payload['detections'][0]['score'] == 0.8
    assert payload['review_timestamp'].endswith('Z')
    assert len(store.history(ROUTE, RUN_ID, '0/0')) == 1
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], value_kph=60, base_revision=1))
    assert status == 200, data
    assert json_of(data)['review']['revision'] == 2
    history = store.history(ROUTE, RUN_ID, '0/0')
    assert [h['revision'] for h in history] == [1, 2]
    assert history[0]['payload']['detections'][0]['value_kph'] == 50
    assert history[1]['payload']['detections'][0]['value_kph'] == 60
    assert history[1]['client'] == '127.0.0.1'


def test_saved_reviews_are_returned_for_the_page(loopback):
    server, root, rows = loopback
    server.post_frame(ROUTE, '0/2', page_body(rows[2], drop_proposals=True))
    status, response, data = server.request('GET', f'/api/route/{ROUTE}/reviews')
    assert status == 200
    body = json_of(data)
    assert body['run_id'] == RUN_ID and body['frame_count'] == 3 and body['confirmed_count'] == 1
    assert len(body['reviews']) == 1
    review = body['reviews'][0]
    assert review['frame_key'] == '0/2' and review['confirmed'] is True and review['detections'] == []
    assert review['reviewer'] == 'Cristian' and review['revision'] == 1


@pytest.mark.parametrize('change,reason,text', [
    ({'value_kph': 0}, 'illegal_speed', 'km/h'),
    ({'bbox_xyxy': [1, 2, 99, 20]}, 'invalid_bbox', 'Invalid box'),
    ({'family': 'advertisement'}, 'unknown_enum', 'Value not allowed'),
    ({'value_kph': 50, 'family': 'other_sign'}, 'value_family_mismatch', 'type of sign'),
])
def test_invalid_detection_is_rejected_with_italian_message(loopback, change, reason, text):
    server, root, rows = loopback
    body = page_body(rows[0])
    body['detections'][0].update(change)
    status, response, data = server.post_frame(ROUTE, '0/0', body)
    assert status == 400
    problem = json_of(data)
    assert problem['ok'] is False and problem['error'] == reason
    assert text in problem['message']
    assert ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0') is None


def test_missing_reviewer_and_bad_body_are_rejected(loopback):
    server, root, rows = loopback
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], reviewer='  '))
    assert status == 400 and 'reviewer' in json_of(data)['message']
    status, response, data = server.post_frame(ROUTE, '0/0', b'{"run_id": NaN}')
    assert status == 400 and json_of(data)['error'] == 'invalid_json'
    status, response, data = server.post_frame(ROUTE, '0/0', b'[1, 2]')
    assert status == 400
    body = page_body(rows[0])
    body['detections'] = 'none'
    status, response, data = server.post_frame(ROUTE, '0/0', body)
    assert status == 400
    status, response, data = server.post_frame(ROUTE, '0/0', b'{}', headers={'Content-Length': str(512 * 1024 + 1)})
    assert status == 413
    status, response, data = server.post_frame(ROUTE, '0/9', page_body(rows[0]))
    assert status == 404 and json_of(data)['error'] == 'unknown_frame'
    assert ReviewStore(db_path(root)).reviews(ROUTE, RUN_ID) == []


def test_too_many_boxes_are_rejected_before_validation(loopback):
    server, root, rows = loopback
    body = page_body(rows[1], manual_box=[3, 3, 12, 14])
    box = body['detections'][0]
    detections = []
    for number in range(101):
        item = dict(box)
        item['detection_id'] = f'manual-{number}'
        detections.append(item)
    body['detections'] = detections
    status, response, data = server.post_frame(ROUTE, '0/1', body)
    assert status == 400
    problem = json_of(data)
    assert problem['error'] == 'too_many_boxes' and '100' in problem['message']
    assert ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/1') is None


def test_reviewer_with_control_characters_or_too_long_is_rejected(loopback, capsys):
    from nnslr_tools.review import terminal_text
    server, root, rows = loopback
    for reviewer in ['\x1b[2J\x1b[31mPWNED\x1b[0m', 'Cristian\nSalvato: falso', 'x' * 101, 'a‮b']:
        status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], reviewer=reviewer))
        assert status == 400, reviewer
        assert json_of(data)['error'] == 'invalid_reviewer' and 'reviewer' in json_of(data)['message']
    assert ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0') is None
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], reviewer='  Zoë Rossi  '))
    assert status == 200, data
    assert json_of(data)['review']['reviewer'] == 'Zoë Rossi'
    status, response, data = server.post_frame(ROUTE, '0/0', {'run_id': RUN_ID, 'reviewer': '\x1b[0m',
                                                              'confirmed': False, 'base_revision': 1})
    assert status == 400 and json_of(data)['error'] == 'invalid_reviewer'
    assert ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0')['confirmed'] is True
    assert '\x1b' not in capsys.readouterr().out
    assert terminal_text('a\x1b[2Jb\nc') == 'a?[2Jb?c'
    assert terminal_text('y' * 150) == 'y' * 100 + '…'


def test_stale_report_and_concurrent_edit_conflicts(loopback):
    server, root, rows = loopback
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], run_id='20200101T000000Z-00000000'))
    assert status == 409 and 'rebuilt' in json_of(data)['message']
    assert server.post_frame(ROUTE, '0/0', page_body(rows[0], base_revision=0))[0] == 200
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], base_revision=0))
    assert status == 409 and json_of(data)['error'] == 'revision_conflict'


def test_withdraw_keeps_payload_and_is_not_confirmed(loopback):
    server, root, rows = loopback
    status, response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], confirmed=False))
    assert status == 404
    server.post_frame(ROUTE, '0/0', page_body(rows[0]))
    status, response, data = server.post_frame(ROUTE, '0/0', {'run_id': RUN_ID, 'reviewer': 'Anna',
                                                              'confirmed': False, 'base_revision': 1})
    assert status == 200, data
    review = json_of(data)['review']
    assert review['confirmed'] is False and review['revision'] == 2 and review['reviewer'] == 'Anna'
    store = ReviewStore(db_path(root))
    assert store.confirmed_count(ROUTE, RUN_ID) == 0
    assert [h['confirmed'] for h in store.history(ROUTE, RUN_ID, '0/0')] == [True, False]


def test_cross_origin_and_non_json_posts_are_refused(loopback):
    server, root, rows = loopback
    body = page_body(rows[0])
    port = server.port
    for headers in [{'Origin': 'http://evil.example'}, {'Origin': 'null'},
                    {'Origin': f'http://127.0.0.1:{port}', 'Sec-Fetch-Site': 'cross-site'}]:
        status, response, data = server.post_frame(ROUTE, '0/0', body, headers=headers)
        assert status == 403 and json_of(data)['error'] == 'cross_origin'
    status, response, data = server.post_frame(ROUTE, '0/0', body, headers={'Content-Type': 'text/plain'})
    assert status == 415
    assert ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0') is None
    status, response, data = server.post_frame(ROUTE, '0/0', body, headers={
        'Origin': f'http://127.0.0.1:{port}', 'Sec-Fetch-Site': 'same-origin'})
    assert status == 200, data


def test_loopback_server_refuses_foreign_host_names(loopback):
    server, root, rows = loopback
    status, response, data = server.request('GET', '/', headers={'Host': 'evil.example:8765'})
    assert status == 403
    status, response, data = server.request('GET', '/', headers={'Host': 'localhost:9000'})
    assert status == 200


def test_paths_outside_the_report_are_refused(loopback):
    server, root, rows = loopback
    directory = root / 'derived/preannotations' / ROUTE / RUN_ID
    (directory / 'images' / '00000099.jpg').symlink_to('/etc/passwd')
    (root / 'secret.txt').write_text('secret')
    for path in [f'/route/{ROUTE}/run.json', f'/route/{ROUTE}/preannotations.jsonl',
                 f'/route/{ROUTE}/images/00000099.jpg', f'/route/{ROUTE}/images/../run.json',
                 f'/route/{ROUTE}/images/..%2Frun.json', '/route/..%2F..%2Fsecret.txt/',
                 '/route/../secret.txt', '/route/not-a-route/', '/secret.txt', '/etc/passwd',
                 f'/api/route/..%2F{ROUTE}/reviews', '/images/00000000.jpg']:
        status, response, data = server.request('GET', path)
        assert status == 404, path
        assert b'secret' not in data and b'root:x:0:0' not in data
    status, response, data = server.post_frame('..%2F' + ROUTE, '0/0', page_body(rows[0]))
    assert status == 404


def test_route_page_is_rendered_without_touching_the_report(loopback):
    server, root, rows = loopback
    index = root / 'derived/preannotations' / ROUTE / RUN_ID / 'index.html'
    before = index.read_bytes()
    status, response, data = server.request('GET', f'/route/{ROUTE}')
    assert status == 303 and response.getheader('Location') == f'/route/{ROUTE}/'
    status, response, data = server.request('GET', f'/route/{ROUTE}/')
    assert status == 200
    csp = response.getheader('Content-Security-Policy')
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert f'<script nonce="{nonce}">'.encode() in data
    assert "connect-src 'self'" in csp and 'unsafe-inline' not in csp.split('script-src')[1].split(';')[0]
    assert b'__NNSLR_' not in data and b'"route_id": "' + ROUTE.encode() in data
    assert response.getheader('X-Frame-Options') == 'DENY'
    assert index.read_bytes() == before
    status, response, data = server.request('GET', f'/route/{ROUTE}/images/00000001.jpg')
    assert status == 200 and response.getheader('Content-Type') == 'image/jpeg'
    status, response, data = server.request('GET', f'/route/{OTHER_ROUTE}/')
    assert status == 404


def test_pages_and_json_go_gzipped_to_a_browser_that_takes_it(loopback):
    server, root, rows = loopback
    for path in (f'/route/{ROUTE}/', f'/api/route/{ROUTE}/page'):
        status, response, plain = server.request('GET', path)
        assert status == 200 and response.getheader('Content-Encoding') is None
        assert response.getheader('Vary') == 'Accept-Encoding'
        status, response, data = server.request('GET', path, headers={'Accept-Encoding': 'gzip, deflate, br'})
        assert status == 200 and response.getheader('Content-Encoding') == 'gzip'
        assert response.getheader('Vary') == 'Accept-Encoding'
        assert int(response.getheader('Content-Length')) == len(data) < len(plain)
        unzipped = gzip.decompress(data)
        if path.startswith('/api/'):
            assert json_of(unzipped) == json_of(plain)
        else:
            # The page differs only by its nonce, new at each request.
            nonce = re.compile(rb'nonce-?[=\'"]*[A-Za-z0-9_-]{24}')
            assert nonce.sub(b'', unzipped) == nonce.sub(b'', plain)
    # Not taken (q=0), or too small to gain: as it is.
    status, response, data = server.request('GET', '/api/routes', headers={'Accept-Encoding': 'gzip;q=0'})
    assert response.getheader('Content-Encoding') is None and json_of(data)['routes']
    status, response, data = server.request('GET', '/api/auth/state', headers={'Accept-Encoding': 'gzip'})
    assert status == 200 and len(data) <= GZIP_MIN and response.getheader('Content-Encoding') is None
    # Pictures go as they are.
    status, response, data = server.request('GET', f'/route/{ROUTE}/images/00000001.jpg', headers={'Accept-Encoding': 'gzip'})
    assert status == 200 and response.getheader('Content-Encoding') is None
    assert data == (root / 'derived/preannotations' / ROUTE / RUN_ID / 'images/00000001.jpg').read_bytes()


@pytest.mark.parametrize('header, taken', [(None, False), ('', False), ('identity', False), ('gzip', True),
                                           ('br, GZIP;q=0.5', True), ('gzip;q=0', False), ('gzip;q=0.000', False),
                                           ('deflate, br', False)])
def test_accept_encoding_is_read_with_its_quality(header, taken):
    assert accepts_gzip(header) is taken


def test_static_report_page_still_works_offline(report):
    root, rows = report
    page = (root / 'derived/preannotations' / ROUTE / RUN_ID / 'index.html').read_text()
    assert '"server": false' in page and '__NNSLR_' not in page
    # Nothing loaded from elsewhere (the Vue runtime only names its error page in a string).
    assert re.search(r'(src|href)\s*=\s*["\']?https?://', page) is None


def test_non_loopback_host_requires_a_token(tmp_path, monkeypatch):
    from nnslr_tools.review import create_review_server, is_loopback, review_token
    assert is_loopback('127.0.0.1') and is_loopback('localhost') and is_loopback('::1')
    assert not is_loopback('0.0.0.0') and not is_loopback('10.10.10.213')
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', 'short')
    from nnslr_tools.preannotate import PreannotationError
    with pytest.raises(PreannotationError):
        review_token()
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    assert review_token() == TOKEN
    monkeypatch.delenv('NNSLR_REVIEW_TOKEN')
    assert len(review_token()) >= 30 and review_token() != review_token()
    server, app = create_review_server(tmp_path, host='0.0.0.0', port=0)
    try:
        assert app.require_token and app.token and len(app.token) >= 30
    finally:
        server.server_close()
    server, app = create_review_server(tmp_path, host='127.0.0.1', port=0)
    try:
        assert not app.require_token and app.token is None
    finally:
        server.server_close()


def test_review_cli_defaults_allow_index_and_lan_options():
    from nnslr_tools.cli import build_parser
    args = build_parser().parse_args(['review'])
    assert args.route is None and args.host == '127.0.0.1' and args.port == 8765
    args = build_parser().parse_args(['review', '--host', '0.0.0.0', '--port', '9000', '--route', ROUTE])
    assert args.host == '0.0.0.0' and args.port == 9000 and args.route == ROUTE
