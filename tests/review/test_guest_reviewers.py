# [guest-reviewers] - START
"""The reviewers from outside: who counts as one, what they may open, what the server stamps on their work, what never
reaches them, and the list of the administrator. A made-up proxy on 127.0.0.1, made-up drives and people."""
from __future__ import annotations

import http.client
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from nnslr_tools import review, review_auth, sign_distance, speed_review
from nnslr_tools.gps import positions_path
from nnslr_tools.pages import CONTROL_TABS
from nnslr_tools.review_access import (PROXY_SECRET_HEADER, Assignments, GuestAccess, guest_rule, scrubbed)
from nnslr_tools.review_db import ReviewRejected
from review_helpers import OTHER_ROUTE, ROUTE, TOKEN, RunningServer, json_of, make_report, page_body, real_previews, sign_in

SECRET = 'proxy-secret-0123456789'
PUBLIC_HOST = 'reviews.example'
ADDRESS = '/root/somewhere/data/runs/cutter.onnx'


def proof(name='ana', groups='reviewers', secret=SECRET) -> dict:
    return {PROXY_SECRET_HEADER: secret, 'X-authentik-username': name, 'X-authentik-groups': groups}


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(review_auth, 'SCRYPT_N', 2 ** 10)


@pytest.fixture
def proxied(monkeypatch):
    monkeypatch.setenv('NNSLR_PROXY_IPS', '127.0.0.1')
    monkeypatch.setenv('NNSLR_PROXY_SECRET', SECRET)
    monkeypatch.setenv('NNSLR_PUBLIC_HOSTS', PUBLIC_HOST)


@pytest.fixture
def outside(tmp_path, proxied):
    rows = make_report(tmp_path)
    make_report(tmp_path, route=OTHER_ROUTE)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.assignments.add_person('ana')
    server.app.assignments.set_work('ana', ROUTE, 'signs', True)
    yield server, rows
    server.close()


def give_all(server, name='ana', route=ROUTE):
    for kind in ('signs', 'distance', 'speed'):
        server.app.assignments.set_work(name, route, kind, True)


def raw_get(server, path, pairs):
    """A request with repeated headers, which the helper's dict cannot send."""
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=10)
    try:
        connection.putrequest('GET', path, skip_host=False)
        for name, value in pairs:
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


# ---------- who is a guest ----------

def test_without_the_settings_nobody_is_a_guest_and_the_site_is_as_always(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        status, _, data = server.request('GET', '/api/auth/state', headers=proof())
        assert status == 200 and json_of(data) == {'configured': False, 'signed_in': False, 'local': False}
        assert server.request('GET', f'/api/route/{ROUTE}/reviews', headers=proof())[0] == 401
    finally:
        server.close()


@pytest.mark.parametrize('headers', [
    {'X-authentik-username': 'ana', 'X-authentik-groups': 'reviewers'},
    proof(secret='wrong-secret'),
    proof(secret=''),
])
def test_the_names_of_the_proxy_count_only_with_its_secret(outside, headers):
    server, _ = outside
    status, _, data = server.request('GET', '/api/auth/state', headers=headers)
    assert status == 200 and json_of(data)['signed_in'] is False and 'role' not in json_of(data)
    assert server.request('GET', f'/api/route/{ROUTE}/reviews', headers=headers)[0] == 401


def test_the_secret_counts_only_from_the_proxy(tmp_path, monkeypatch):
    make_report(tmp_path)
    monkeypatch.setenv('NNSLR_PROXY_IPS', '10.0.0.9')
    monkeypatch.setenv('NNSLR_PROXY_SECRET', SECRET)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        server.app.assignments.add_person('ana')
        server.app.assignments.set_work('ana', ROUTE, 'signs', True)
        assert server.request('GET', f'/api/route/{ROUTE}/reviews', headers=proof())[0] == 401
    finally:
        server.close()


def test_never_a_guest_on_the_site_of_this_computer(tmp_path, proxied):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=None, require_token=False)
    try:
        assert json_of(server.request('GET', '/api/auth/state', headers=proof())[2])['local'] is True
    finally:
        server.close()


@pytest.mark.parametrize('host', [PUBLIC_HOST, PUBLIC_HOST + ':443', PUBLIC_HOST.upper()])
def test_the_public_address_without_the_proxy_shows_no_local_sign_in(outside, host):
    server, _ = outside
    for path in ('/', '/reviews', '/api/auth/state', '/api/v1/routes'):
        assert server.request('GET', path, headers={'Host': host})[0] == 403


def test_the_site_no_longer_holds_the_public_page(outside):
    server, _ = outside
    admin = sign_in(server)
    for path in ('/product', '/product/', '/product/icon.svg'):
        assert server.request('GET', path, headers={'Host': PUBLIC_HOST})[0] == 403
        assert server.request('GET', path, headers=proof())[0] == 403
        assert server.request('GET', path, headers=admin)[0] == 404


def test_the_home_of_a_reviewer_is_their_list_and_slash_leads_there(outside):
    server, _ = outside
    status, response, _ = server.request('GET', '/?from=a-link', headers={'Host': PUBLIC_HOST, **proof()})
    assert status == 303 and response.getheader('Location') == '/reviews'
    status, _, data = server.request('GET', '/reviews', headers={'Host': PUBLIC_HOST, **proof()})
    assert status == 200 and b'data-guest="true"' in data and b'href="/reviews"' in data and b'href="/"' not in data


def test_on_the_lan_reviews_is_nobodys_page(outside):
    server, _ = outside
    assert server.request('GET', '/reviews')[0] == 401
    status, response, _ = server.request('GET', '/reviews', headers=sign_in(server))
    assert status == 303 and response.getheader('Location') == '/'


def test_a_made_up_name_without_the_secret_is_nobody(outside):
    server, _ = outside
    forged = {'Host': PUBLIC_HOST, 'X-authentik-username': 'ana', 'X-authentik-groups': 'reviewers'}
    assert server.request('GET', '/reviews', headers=forged)[0] == 403
    assert server.request('GET', f'/api/route/{ROUTE}/reviews', headers=forged)[0] == 403


def test_an_account_with_no_access_can_go_back_and_sign_out(outside):
    server, _ = outside
    status, _, data = server.request('GET', '/reviews', headers={'Host': PUBLIC_HOST, **proof(groups='farm')})
    assert status == 403 and b'<a class="btn back" href="/">' in data
    assert b'href="/outpost.goauthentik.io/sign_out"' in data


@pytest.mark.parametrize('headers', [
    {PROXY_SECRET_HEADER: SECRET},
    {PROXY_SECRET_HEADER: SECRET, 'X-authentik-username': 'ana'},
    {PROXY_SECRET_HEADER: SECRET, 'X-authentik-groups': 'reviewers'},
    proof(groups='farm'),
    proof(groups='reviewers-not|farm'),
    proof(groups=''),
    proof(name=''),
    proof(name='ana/../admin'),
    proof(name='.hidden'),
])
def test_through_the_proxy_without_a_reviewer_is_refused_never_the_local_site(outside, headers):
    server, _ = outside
    for path in ('/reviews', '/api/auth/state', '/account'):
        status, _, data = server.request('GET', path, headers=headers)
        assert status == 403 and b'/api/auth/login' not in data and b'href="/prepare"' not in data


@pytest.mark.parametrize('repeated', [PROXY_SECRET_HEADER, 'X-authentik-username', 'X-authentik-groups'])
def test_a_repeated_header_proves_nothing(outside, repeated):
    server, _ = outside
    pairs = list(proof().items()) + [(repeated, proof()[repeated])]
    status, _ = raw_get(server, f'/api/route/{ROUTE}/reviews', pairs)
    assert status in (401, 403)


def test_every_group_of_the_list_counts(outside):
    server, _ = outside
    assert server.request('GET', f'/api/route/{ROUTE}/reviews', headers=proof(groups='farm|reviewers'))[0] == 200


def test_the_group_is_the_one_set(monkeypatch):
    monkeypatch.setenv('NNSLR_PROXY_IPS', '127.0.0.1, ::1')
    monkeypatch.setenv('NNSLR_PROXY_SECRET', SECRET)
    monkeypatch.setenv('NNSLR_REVIEWER_GROUP', 'guests')
    access = GuestAccess.from_env()
    assert access.enabled and access.proxy_ips == {'127.0.0.1', '::1'} and access.group == 'guests'


def test_a_bare_ipv4_peer_written_as_ipv6_is_the_same_proxy(tmp_path):
    from email.message import Message
    headers = Message()
    for name, value in proof().items():
        headers[name] = value
    access = GuestAccess(proxy_ips=frozenset({'127.0.0.1'}), secret=SECRET)
    assert access.guest_of('::ffff:127.0.0.1', headers) == 'ana'
    assert access.guest_of('127.0.0.2', headers) is None


def test_the_same_connection_can_carry_different_people(outside):
    server, _ = outside
    server.app.assignments.add_person('bea')
    connection = http.client.HTTPConnection('127.0.0.1', server.port, timeout=10)
    try:
        seen = []
        for headers in (proof('ana'), proof('bea'), {PROXY_SECRET_HEADER: SECRET}, proof('ana'), {}):
            connection.request('GET', f'/api/route/{ROUTE}/reviews', headers=headers)
            response = connection.getresponse()
            response.read()
            seen.append(response.status)
    finally:
        connection.close()
    assert seen == [200, 403, 403, 200, 401]


# ---------- nothing of the administrator ----------

def test_the_cookie_and_the_token_of_the_administrator_count_for_nothing_on_the_way_in(outside):
    server, _ = outside
    cookie = sign_in(server)
    keys = {**cookie, 'X-Review-Token': TOKEN, 'Authorization': 'Bearer ' + TOKEN}
    assert server.request('GET', '/api/panel/status', headers=keys)[0] == 200
    assert server.request('GET', '/api/panel/status', headers={**keys, **proof()})[0] == 403
    state = json_of(server.request('GET', '/api/auth/state', headers={**keys, **proof()})[2])
    assert state['role'] == 'reviewer' and state['user'] == 'ana' and 'sessions' not in state
    # A group called as the administrator's gives nothing either.
    assert server.request('GET', '/api/panel/status', headers=proof(groups='reviewers|admin|farm'))[0] == 403


@pytest.mark.parametrize('path', ['/api/v1', '/api/v1/routes', '/mcp'])
def test_the_agents_addresses_are_closed(outside, path):
    server, _ = outside
    keys = {**proof(), 'Authorization': 'Bearer ' + TOKEN}
    assert server.request('GET', path, headers=keys)[0] == 403
    assert server.request('POST', path, body={}, headers=keys)[0] == 403


def addresses_of_the_site() -> set[str]:
    """Every address written in review.py, plus the pages: a page added tomorrow is in this walk by itself."""
    source = Path(review.__file__).read_text(encoding='utf-8')
    found = set(re.findall(r"'(/api/[a-z0-9_/.-]+)'", source))
    found.update(re.findall(r"r'(/api/[a-z0-9_/-]+)[/(']", source))
    found.update(CONTROL_TABS)
    found.update({'/', '/reviews', '/routes', '/account', '/index.html', '/prepare', '/panoramica', '/pannello'})
    drive = ('', '/segments', '/scan', '/logs', '/episodes', '/queue')
    for route in (ROUTE, OTHER_ROUTE):
        found.update(f'/route/{route}/{tail}' for tail in ('summary', 'car', 'report.jsonl', 'reviews.jsonl', 'run.json',
                                                           'preannotations.jsonl', '../../etc/passwd'))
        found.update(f'/api/route/{route}/{tail}' for tail in ('ai-trial', 'ai-live', 'car/0/0', 'car-logs', 'frame/0/0/ai-suggest',
                                                               'frame/0/0/ai-fit', 'car-settings',
                                                               'frame/0/0/classifier-crop'))
        found.update(f'/api/drive-review/drives/{route}{tail}' for tail in drive)
        found.add(f'/api/panel/drive-video/{OTHER_ROUTE}')
    return found


def test_a_signed_in_account_outside_the_group_gets_nothing_anywhere(outside):
    server, _ = outside
    give_all(server)
    for path in sorted(addresses_of_the_site()):
        for method in ('GET', 'POST'):
            status, _, data = server.request(method, path, body={} if method == 'POST' else None, headers=proof(groups='farm'))
            assert status == 403, (method, path, status)
            assert ROUTE.encode() not in data and OTHER_ROUTE.encode() not in data


def test_a_walk_of_the_site_finds_only_what_the_table_opens(outside):
    server, _ = outside
    give_all(server)
    walked = addresses_of_the_site()
    assert len(walked) > 60
    opened = []
    for path in sorted(walked):
        for method in ('GET', 'POST'):
            status, _, data = server.request(method, path, body={} if method == 'POST' else None, headers=proof())
            rule = guest_rule(method, path)
            if rule is None or (rule[0] is not None and rule[0] != ROUTE):
                assert status == 403, (method, path, status)
            else:
                opened.append((method, path))
    assert ('GET', '/api/auth/state') in opened and ('GET', '/reviews') in opened
    assert all(not path.startswith(('/api/ai', '/api/settings', '/api/regression')) for _, path in opened)
    assert all('/scan' not in path and '/logs' not in path for _, path in opened)


@pytest.mark.parametrize('method', ['PUT', 'DELETE', 'OPTIONS', 'PATCH'])
def test_other_methods_are_closed(outside, method):
    server, _ = outside
    assert server.request(method, f'/api/route/{ROUTE}/reviews', headers=proof())[0] in (403, 501)


@pytest.mark.parametrize('path', [
    f'/route/{ROUTE}/../{OTHER_ROUTE}/',
    f'/route/{ROUTE}%2f..%2f{OTHER_ROUTE}/',
    f'/api/route/{ROUTE}/../{OTHER_ROUTE}/reviews',
    f'//route/{OTHER_ROUTE}/',
    f'/route/{ROUTE}/images/../../{OTHER_ROUTE}/images/00000000.jpg',
    f'/route/{ROUTE}/images/00000000.jpg/../../summary',
])
def test_a_trick_in_the_address_opens_nothing(outside, path):
    server, _ = outside
    assert server.request('GET', path, headers=proof())[0] == 403


# ---------- only the work given ----------

def test_only_the_drives_and_the_kinds_given(outside):
    server, _ = outside
    assert server.request('GET', f'/route/{ROUTE}/', headers=proof())[0] == 200
    assert server.request('GET', f'/route/{ROUTE}/images/00000000.jpg', headers=proof())[0] == 200
    assert server.request('GET', f'/route/{ROUTE}/small/images/00000000.jpg', headers=proof())[0] == 403
    assert server.request('GET', f'/route/{OTHER_ROUTE}/', headers=proof())[0] == 403
    assert server.request('GET', f'/route/{OTHER_ROUTE}/images/00000000.jpg', headers=proof())[0] == 403
    assert server.request('GET', f'/api/route/{OTHER_ROUTE}/reviews', headers=proof())[0] == 403
    # The signs of a drive do not open its other work: the player of Speed review shows them (its Signs side), its
    # speed is not saved.
    assert server.request('GET', f'/api/panel/road-types/{ROUTE}', headers=proof())[0] == 403
    assert server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers=proof())[0] == 200
    assert server.request('POST', f'/api/speed-review/drives/{ROUTE}/blocks', body={}, headers=proof())[0] == 403
    assert server.request('GET', f'/api/sign-distance/drives/{ROUTE}', headers=proof())[0] == 403


def test_work_taken_back_closes_at_once_and_a_person_removed_has_nothing(outside):
    server, _ = outside
    path = f'/route/{ROUTE}/'
    assert server.request('GET', path, headers=proof())[0] == 200
    server.app.assignments.set_work('ana', ROUTE, 'signs', False)
    assert server.request('GET', path, headers=proof())[0] == 403
    give_all(server)
    server.app.assignments.remove_person('ana')
    assert server.request('GET', path, headers=proof())[0] == 403
    assert json_of(server.request('GET', '/api/reviewer/work', headers=proof())[2])['drives'] == []


def test_my_list_holds_my_drives_by_date_with_their_pages(outside, tmp_path):
    server, _ = outside
    real_previews(tmp_path)
    server.app.assignments.set_work('ana', ROUTE, 'speed', True)
    answer = json_of(server.request('GET', '/api/reviewer/work', headers=proof())[2])
    assert answer['user'] == 'ana' and [item['route'] for item in answer['drives']] == [ROUTE]
    assert [item['kind'] for item in answer['drives'][0]['work']] == ['signs', 'speed']
    assert answer['drives'][0]['work'][1]['href'] == f'/speed-review?drive={ROUTE}'
    # No road video of it here: its signs keep the page of Sign review.
    assert answer['drives'][0]['work'][0]['href'] == f'/route/{ROUTE}/'
    assert 'title' not in answer['drives'][0] and 'from' not in answer['drives'][0]


def test_the_lists_are_filtered(outside, monkeypatch):
    server, _ = outside
    server.app.assignments.set_work('ana', ROUTE, 'distance', True)
    drives = [{'route': ROUTE, 'status': 'signs', 'title': 'Olten → Zürich'}, {'route': OTHER_ROUTE, 'status': 'signs', 'title': 'Bern'}]
    monkeypatch.setattr(sign_distance, 'overview', lambda store: {'drives': drives, 'summary': {}})
    answer = json_of(server.request('GET', '/api/sign-distance/drives', headers=proof())[2])
    assert answer['drives'] == [{'route': ROUTE, 'status': 'signs'}]
    page = server.request('GET', f'/route/{ROUTE}/', headers=proof())[2].decode()
    assert OTHER_ROUTE not in page


# ---------- what the server stamps ----------

def test_a_frame_is_signed_by_the_server_never_by_the_page(outside):
    server, rows = outside
    body = page_body(rows[0], reviewer='Cristian')
    body.update(reviewer_kind='ai', override_human_review=True)
    status, _, data = server.post_frame(ROUTE, rows[0]['frame_key'], body, headers=proof())
    assert status == 200, data
    saved = json_of(data)
    saved = saved.get('review', saved)
    assert saved['reviewer'] == 'guest/ana' and saved['reviewer_kind'] == 'human'


def test_a_guest_writes_nothing_on_another_drive(outside):
    server, rows = outside
    assert server.post_frame(OTHER_ROUTE, rows[0]['frame_key'], page_body(rows[0]), headers=proof())[0] == 403


def test_ai_and_fit_on_a_frame(outside):
    server, rows = outside
    key = rows[0]['frame_key']
    assert server.request('POST', f'/api/route/{ROUTE}/frame/{key}/ai-suggest', body={}, headers=proof())[0] == 403
    assert server.request('POST', f'/api/route/{ROUTE}/frame/{key}/ai-fit', body={}, headers=proof())[0] == 403
    # The fit reaches its own checks: refused for the body, not for who asks.
    status, _, data = server.request('POST', f'/api/route/{ROUTE}/frame/{key}/fit', body={}, headers=proof())
    assert (status, json_of(data).get('reason')) != (403, 'session_required')


@pytest.mark.parametrize('headers', [
    {'Origin': 'https://evil.example'},
    {'Sec-Fetch-Site': 'cross-site'},
    {'Content-Type': 'text/plain'},
])
def test_the_saves_of_a_guest_have_the_protections_of_every_save(outside, headers):
    server, rows = outside
    status, _, _ = server.post_frame(ROUTE, rows[0]['frame_key'], page_body(rows[0]), headers={**proof(), **headers})
    assert status in (403, 415)


def test_sign_distance_and_speed_review_take_the_name_of_the_guest(outside, monkeypatch):
    server, _ = outside
    give_all(server)
    who = []

    def mark(store, route, body, reviewer):
        who.append(reviewer)
        return {'line': {'sign': '1', 'mark': 'zero'}}

    def save(store, drives, route, body, reviewer):
        who.append(reviewer)
        return {'revision': 1}

    monkeypatch.setattr(sign_distance, 'mark', mark)
    monkeypatch.setattr(speed_review, 'save_block', save)
    monkeypatch.setattr(speed_review, 'save_road', save)
    assert server.request('POST', f'/api/sign-distance/drives/{ROUTE}/marks', body={}, headers=proof())[0] == 200
    assert server.request('POST', f'/api/speed-review/drives/{ROUTE}/blocks', body={}, headers=proof())[0] == 200
    assert server.request('POST', f'/api/speed-review/drives/{ROUTE}/road', body={}, headers=proof())[0] == 200
    assert who == ['guest/ana'] * 3
    # A guest marks with the kinds of road of this server and adds none.
    assert server.request('POST', '/api/speed-review/road-kinds', body={'label': 'x'}, headers=proof())[0] == 403
    # The token alone still writes neither: as always, only a person.
    keys = {'X-Review-Token': TOKEN}
    assert server.request('POST', f'/api/sign-distance/drives/{ROUTE}/marks', body={}, headers=keys)[0] == 403
    assert server.request('POST', f'/api/speed-review/drives/{ROUTE}/blocks', body={}, headers=keys)[0] == 403


# ---------- where the drive was ----------

def test_scrubbed_takes_away_the_places_and_the_paths_at_any_depth():
    value = {'title': 'Olten → Zürich', 'frames': [{'gps': {'latitude': 47.3}, 'image_path': 'derived/frames/a.png',
                                                    'model': {'cutter_path': ADDRESS, 'repository': 'x'}}],
             'series': {'road_name': [[1, 'Bahnhofstrasse']], 'roads': ['primary']}, 'lat': 1, 'lon': 2}
    assert scrubbed(value) == {'frames': [{'image_path': 'derived/frames/a.png', 'model': {'repository': 'x'}}],
                               'series': {'roads': ['primary']}}


def test_the_review_page_of_a_guest_holds_no_place_and_no_path(tmp_path, proxied):
    rows = make_report(tmp_path)
    directory = tmp_path / 'derived/preannotations' / ROUTE / rows[0]['run_id']
    for row in rows:
        row['model']['cutter_path'] = ADDRESS
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    positions = positions_path(tmp_path, ROUTE)
    positions.parent.mkdir(parents=True)
    positions.write_text(''.join(json.dumps({'frame_key': row['frame_key'], 'latitude': 47.351, 'longitude': 7.903,
                                             'speed_mps': 20.0, 'bearing_deg': 90.0}) + '\n' for row in rows))
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        server.app.assignments.add_person('ana')
        server.app.assignments.set_work('ana', ROUTE, 'signs', True)
        guest = server.request('GET', f'/route/{ROUTE}/', headers=proof())[2].decode()
        admin = server.request('GET', f'/route/{ROUTE}/', headers={'X-Review-Token': TOKEN})[2].decode()
    finally:
        server.close()
    assert '47.351' in admin and ADDRESS in admin
    assert '47.351' not in guest and ADDRESS not in guest and '"gps"' not in guest
    assert 'data-guest="true"' in guest and 'href="/prepare"' not in guest


# [sign-review-retired] - START
def test_a_guest_reviews_the_signs_on_the_signs_side_of_speed_review(tmp_path, proxied):
    rows = make_report(tmp_path)
    (tmp_path / 'raw/routes' / ROUTE / '0').mkdir(parents=True)
    (tmp_path / 'raw/routes' / ROUTE / '0' / 'fcamera.hevc').write_bytes(b'video')
    directory = tmp_path / 'derived/preannotations' / ROUTE / rows[0]['run_id']
    for row in rows:
        row['model']['cutter_path'] = ADDRESS
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    positions = positions_path(tmp_path, ROUTE)
    positions.parent.mkdir(parents=True)
    positions.write_text(''.join(json.dumps({'frame_key': row['frame_key'], 'latitude': 47.351, 'longitude': 7.903,
                                             'speed_mps': 20.0, 'bearing_deg': 90.0}) + '\n' for row in rows))
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        server.app.assignments.add_person('ana')
        server.app.assignments.set_work('ana', ROUTE, 'signs', True)
        # The page of Sign review leads to the Signs side, at the frame asked.
        status, response, _ = server.request('GET', f'/route/{ROUTE}/?frame=0/1', headers=proof())
        assert status == 303 and response.getheader('Location') == f'/speed-review?drive={ROUTE}&side=signs&frame=0/1'
        # Its list links there.
        work = json_of(server.request('GET', '/api/reviewer/work', headers=proof())[2])
        assert work['drives'][0]['work'][0]['href'] == f'/speed-review?drive={ROUTE}&side=signs'
        # The editor's data, without the places and the paths of the server.
        guest = server.request('GET', f'/api/route/{ROUTE}/page', headers=proof())
        admin = server.request('GET', f'/api/route/{ROUTE}/page', headers={'X-Review-Token': TOKEN})
        assert guest[0] == 200 and admin[0] == 200
        assert b'47.351' in admin[2] and ADDRESS.encode() in admin[2]
        assert b'47.351' not in guest[2] and ADDRESS.encode() not in guest[2] and b'"gps"' not in guest[2]
        assert server.request('GET', f'/api/speed-review/drives/{ROUTE}/signs', headers=proof())[0] == 200
        # Given only the speed: no signs, no editor, no page of Sign review.
        server.app.assignments.set_work('ana', ROUTE, 'signs', False)
        server.app.assignments.set_work('ana', ROUTE, 'speed', True)
        assert server.request('GET', f'/api/route/{ROUTE}/page', headers=proof())[0] == 403
        assert server.request('GET', f'/api/speed-review/drives/{ROUTE}/signs', headers=proof())[0] == 403
        assert server.request('GET', f'/route/{ROUTE}/', headers=proof())[0] == 403
        assert server.request('GET', f'/api/speed-review/drives/{ROUTE}', headers=proof())[0] == 200
    finally:
        server.close()
# [sign-review-retired] - END


def test_the_json_of_a_guest_is_scrubbed(outside, monkeypatch):
    server, _ = outside
    server.app.assignments.set_work('ana', ROUTE, 'distance', True)
    monkeypatch.setattr(sign_distance, 'drive_view', lambda store, route: {
        'route': route, 'title': 'Olten → Zürich', 'gps': [1, 2], 'file_path': ADDRESS})
    assert json_of(server.request('GET', f'/api/sign-distance/drives/{ROUTE}', headers=proof())[2]) == {'route': ROUTE}


@pytest.mark.parametrize('path', sorted(['/reviews', '/speed-review', '/sign-distance']))
def test_the_pages_of_a_guest_show_their_bar_and_no_tab_of_the_administrator(outside, path):
    server, _ = outside
    status, _, data = server.request('GET', path, headers=proof())
    text = data.decode()
    bar = text[text.index('<header class="topbar"'):text.index('</header>')]
    assert status == 200 and 'data-guest="true"' in bar and 'My reviews' in bar
    for address in ('/prepare', '/train', '/models', '/routes', '/dataset', '/account'):
        assert f'href="{address}"' not in bar
    assert 'tabs-machine' not in bar and 'comma-pill' not in bar


def test_the_error_page_of_a_guest_is_theirs(outside):
    server, _ = outside
    status, _, data = server.request('GET', '/prepare', headers=proof())
    assert status == 403 and b'part of the work given to you' in data and b'href="/prepare"' not in data


def test_the_state_tells_the_role_and_the_way_out(outside):
    server, _ = outside
    state = json_of(server.request('GET', '/api/auth/state', headers=proof())[2])
    assert state == {'configured': True, 'signed_in': True, 'local': False, 'user': 'ana', 'role': 'reviewer',
                     'sign_out': '/outpost.goauthentik.io/sign_out'}
    admin = sign_in(server)
    assert json_of(server.request('GET', '/api/auth/state', headers=admin)[2])['role'] == 'admin'


# ---------- the page of the administrator ----------

def test_the_administrator_sees_people_work_and_drives(outside):
    server, _ = outside
    answer = json_of(server.request('GET', '/api/reviewers', headers=sign_in(server))[2])
    assert answer['open'] is True and answer['can_edit'] is True
    assert [person['name'] for person in answer['people']] == ['ana']
    assert answer['work'] == {'ana': {ROUTE: ['signs']}}
    assert {drive['route'] for drive in answer['drives']} >= {ROUTE, OTHER_ROUTE}
    assert 'signs' in next(drive for drive in answer['drives'] if drive['route'] == ROUTE)['kinds']
    assert [kind['key'] for kind in answer['kinds']] == ['signs', 'distance', 'speed']


def test_the_administrator_gives_and_takes_back(outside):
    server, _ = outside
    admin = sign_in(server)
    assert server.request('POST', '/api/reviewers/people', body={'name': 'bea'}, headers=admin)[0] == 200
    assign = {'user': 'bea', 'route': OTHER_ROUTE, 'kind': 'signs', 'on': True}
    answer = json_of(server.request('POST', '/api/reviewers/assign', body=assign, headers=admin)[2])
    assert answer['work']['bea'] == {OTHER_ROUTE: ['signs']}
    assert server.request('GET', f'/route/{OTHER_ROUTE}/', headers=proof('bea'))[0] == 200
    answer = json_of(server.request('POST', '/api/reviewers/assign', body={**assign, 'on': False}, headers=admin)[2])
    assert answer['work']['bea'] == {}
    assert server.request('GET', f'/route/{OTHER_ROUTE}/', headers=proof('bea'))[0] == 403
    answer = json_of(server.request('POST', '/api/reviewers/people/remove', body={'name': 'bea'}, headers=admin)[2])
    assert 'bea' not in answer['work'] and [person['name'] for person in answer['people']] == ['ana']


@pytest.mark.parametrize('body', [
    {'user': 'ana', 'route': 'not-a-route', 'kind': 'signs', 'on': True},
    {'user': 'ana', 'route': ROUTE, 'kind': 'training', 'on': True},
    {'user': 'ana', 'route': ROUTE, 'kind': 'signs', 'on': 'yes'},
    {'user': 'ana/../x', 'route': ROUTE, 'kind': 'signs', 'on': True},
    {'user': 'nobody', 'route': ROUTE, 'kind': 'signs', 'on': True},
])
def test_a_wrong_assignment_is_refused(outside, body):
    server, _ = outside
    assert server.request('POST', '/api/reviewers/assign', body=body, headers=sign_in(server))[0] in (400, 404)


def test_only_a_signed_in_browser_changes_who_works(outside):
    server, _ = outside
    assign = {'user': 'ana', 'route': OTHER_ROUTE, 'kind': 'signs', 'on': True}
    assert server.request('POST', '/api/reviewers/assign', body=assign, headers={'X-Review-Token': TOKEN})[0] == 403
    assert server.request('POST', '/api/reviewers/assign', body=assign)[0] == 401
    assert server.request('POST', '/api/reviewers/assign', body=assign, headers=proof())[0] == 403
    assert server.request('GET', '/api/reviewers', headers=proof())[0] == 403
    admin = sign_in(server)
    keys = {**admin, 'Origin': 'https://evil.example'}
    assert server.request('POST', '/api/reviewers/assign', body=assign, headers=keys)[0] == 403
    assert server.app.assignments.work_of('ana') == {ROUTE: ['signs']}


def test_the_page_of_the_administrator_tells_when_nobody_can_come_in(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        assert json_of(server.request('GET', '/api/reviewers', headers=sign_in(server))[2])['open'] is False
    finally:
        server.close()


# ---------- the file ----------

def test_the_file_is_private_and_survives(tmp_path):
    assignments = Assignments(tmp_path / 'auth')
    assignments.add_person('ana')
    assignments.set_work('ana', ROUTE, 'distance', True)
    assignments.set_work('ana', ROUTE, 'signs', True)
    assert stat.S_IMODE((tmp_path / 'auth' / 'assignments.json').stat().st_mode) == 0o600
    again = Assignments(tmp_path / 'auth')
    assert again.work_of('ana') == {ROUTE: ['signs', 'distance']}
    assert again.may('ana', ROUTE, ('distance',)) and not again.may('ana', ROUTE, ('speed',))
    assert not again.may('bea', ROUTE, ('distance',)) and not again.may('ana', OTHER_ROUTE, ('distance',))


def test_roads_given_before_7_nov_2026_are_left_out(tmp_path):
    assignments = Assignments(tmp_path / 'auth')
    assignments.add_person('ana')
    path = tmp_path / 'auth' / 'assignments.json'
    data = json.loads(path.read_text())
    data['work'] = {'ana': {ROUTE: ['signs', 'roads'], OTHER_ROUTE: ['roads']}}
    path.write_text(json.dumps(data))
    again = Assignments(tmp_path / 'auth')
    assert again.work_of('ana') == {ROUTE: ['signs']}
    assert not again.may('ana', ROUTE, ('roads',)) and not again.may('ana', OTHER_ROUTE, ('signs', 'roads'))


def test_a_damaged_file_opens_nothing(tmp_path):
    assignments = Assignments(tmp_path)
    (tmp_path / 'assignments.json').write_text('{not json')
    with pytest.raises(ReviewRejected):
        assignments.may('ana', ROUTE, ('signs',))


def test_a_person_joins_at_the_first_visit_and_is_written_at_most_hourly(tmp_path):
    now = [1000.0]
    assignments = Assignments(tmp_path, clock=lambda: now[0])
    assignments.seen('ana')
    first = assignments.view()['people'][0]
    assert first['name'] == 'ana' and first['last_seen'] is not None
    (tmp_path / 'assignments.json').unlink()
    now[0] += 60
    assignments.seen('ana')
    assert not (tmp_path / 'assignments.json').exists()
    now[0] += 3600
    assignments.seen('ana')
    assert (tmp_path / 'assignments.json').exists()


def test_a_first_visit_needs_nobody_to_have_added_the_person(outside):
    server, _ = outside
    assert server.request('GET', '/reviews', headers=proof('new.person'))[0] == 200
    assert [person['name'] for person in server.app.assignments.view()['people']] == ['ana', 'new.person']
    assert json_of(server.request('GET', '/api/reviewer/work', headers=proof('new.person'))[2])['drives'] == []
# [guest-reviewers] - END


SETUP_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/setup_guest_access.sh'
SETUP_OPTIONS = ['--proxy-ips', '192.168.200.161', '--public-hosts', 'speedlimits.example.org']


def run_setup(env_file, *options):
    return subprocess.run([str(SETUP_SCRIPT), *options], capture_output=True, text=True, check=False,
                          env={**os.environ, 'NNSLR_REVIEW_ENV_FILE': str(env_file)})


def settings_of(env_file):
    return dict(line.split('=', 1) for line in env_file.read_text().splitlines())


def test_the_setup_script_writes_the_settings_and_never_shows_the_secret(tmp_path):
    env_file = tmp_path / 'site.env'
    env_file.write_text('NNSLR_REVIEW_TOKEN=keep-me\n')
    answer = run_setup(env_file, *SETUP_OPTIONS)
    assert answer.returncode == 0, answer.stderr
    settings = settings_of(env_file)
    assert settings['NNSLR_REVIEW_TOKEN'] == 'keep-me' and settings['NNSLR_PROXY_IPS'] == '192.168.200.161'
    assert settings['NNSLR_PUBLIC_HOSTS'] == 'speedlimits.example.org' and len(settings['NNSLR_PROXY_SECRET']) == 64
    assert settings['NNSLR_PROXY_SECRET'] not in answer.stdout + answer.stderr
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_the_setup_script_keeps_the_secret_until_it_is_asked_for_a_new_one(tmp_path):
    env_file = tmp_path / 'site.env'
    env_file.write_text('NNSLR_REVIEW_TOKEN=keep-me\n')
    run_setup(env_file, *SETUP_OPTIONS)
    first = settings_of(env_file)['NNSLR_PROXY_SECRET']
    run_setup(env_file, '--proxy-ips', '10.0.0.1', '--public-hosts', 'other.example.org')
    assert settings_of(env_file)['NNSLR_PROXY_SECRET'] == first
    assert settings_of(env_file)['NNSLR_PROXY_IPS'] == '10.0.0.1'
    assert env_file.read_text().count('NNSLR_PROXY_IPS=') == 1
    run_setup(env_file, *SETUP_OPTIONS, '--new-secret')
    assert settings_of(env_file)['NNSLR_PROXY_SECRET'] != first


def test_the_setup_script_changes_nothing_on_a_dry_run_and_closes_with_off(tmp_path):
    env_file = tmp_path / 'site.env'
    env_file.write_text('NNSLR_REVIEW_TOKEN=keep-me\n')
    assert run_setup(env_file, *SETUP_OPTIONS, '--dry-run').returncode == 0
    assert env_file.read_text() == 'NNSLR_REVIEW_TOKEN=keep-me\n'
    run_setup(env_file, *SETUP_OPTIONS)
    assert run_setup(env_file, '--off').returncode == 0
    assert env_file.read_text() == 'NNSLR_REVIEW_TOKEN=keep-me\n'


@pytest.mark.parametrize('options', [['--proxy-ips', 'a b', '--public-hosts', 'x.org'],
                                     ['--proxy-ips', '10.0.0.1', '--public-hosts', 'x.org\nNNSLR_REVIEW_TOKEN=evil'],
                                     ['--public-hosts', 'x.org'], ['--proxy-ips', '10.0.0.1']])
def test_the_setup_script_refuses_what_is_not_an_address_or_a_name(tmp_path, options):
    env_file = tmp_path / 'site.env'
    env_file.write_text('NNSLR_REVIEW_TOKEN=keep-me\n')
    assert run_setup(env_file, *options).returncode == 2
    assert env_file.read_text() == 'NNSLR_REVIEW_TOKEN=keep-me\n'


def test_speed_review_is_given_only_on_a_drive_with_sunnypilot_off(outside, monkeypatch):
    # The brain learns only dashcam drives (speed_review.dashcam_drive): Speed review is no work on the others.
    from nnslr_tools import speed_review
    server, _ = outside
    rows = [{'route': ROUTE, 'video_segments': [0], 'dashcam': True},
            {'route': OTHER_ROUTE, 'video_segments': [0], 'dashcam': False}]
    monkeypatch.setattr(speed_review, 'drives_list', lambda *args, **kwargs: rows)
    kinds = {drive['route']: drive['kinds'] for drive in server.app.assignable_drives()}
    assert 'speed' in kinds[ROUTE] and 'speed' not in kinds.get(OTHER_ROUTE, [])
    # A drive not read yet (dashcam None) can still be given.
    rows[1]['dashcam'] = None
    assert 'speed' in {drive['route']: drive['kinds'] for drive in server.app.assignable_drives()}[OTHER_ROUTE]
