"""The list of drives with signs (GET /api/routes; the page /routes is gone since 9 Nov 2026) and the data of the review page: what the cards count, the AI's share, one link to set
up the AI reviewer, who proposed the signs, the drive's two lines in the route menu."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nnslr_tools.pages import render_index
from nnslr_tools.review_db import ReviewStore, db_path
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, RunningServer, make_report


def ai_save(app, route, key, detections):
    """A frame confirmed by an AI reviewer through the REST API."""
    body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
            'detections': detections}
    return app.save_review(route, key, body, client='test', channel='rest')


@pytest.fixture
def served(tmp_path):
    make_report(tmp_path, count=3)
    make_report(tmp_path, route=OTHER_ROUTE, count=2)
    server = RunningServer(tmp_path)
    yield server, tmp_path
    server.close()


def card_of(page: str, route: str) -> str:
    return next(item for item in page.split('<li>') if 'href="/route/' + route + '/"' in item)


def test_route_list_counts_proposals_while_to_review_and_the_share_of_the_ai(served):
    server, root = served
    ai_save(server.app, ROUTE, '0/1', [])
    ai_save(server.app, OTHER_ROUTE, '0/0', [{'detection_id': '0/0/0', 'applicability': 'own_road'}])
    ai_save(server.app, OTHER_ROUTE, '0/1', [])
    store = ReviewStore(db_path(root))
    assert store.confirmed_count(ROUTE, RUN_ID) == 1 and store.confirmed_count(ROUTE, RUN_ID, reviewer_kind='ai') == 1
    assert store.confirmed_count(ROUTE, RUN_ID, reviewer_kind='human') == 0
    # The list of Speed review's card "Signs to review" (the list page /routes is gone since 9 Nov 2026).
    routes = {item['route_id']: item for item in json.loads(server.request('GET', '/api/routes')[2])['routes']}
    doing, done = routes[ROUTE], routes[OTHER_ROUTE]
    # The model's proposals: two of the three frames; one confirmed, by the AI.
    assert (doing['frame_count'], doing['frames_with_proposals']) == (3, 2)
    assert (doing['confirmed_count'], doing['confirmed_by_ai']) == (1, 1)
    assert (done['frame_count'], done['confirmed_count'], done['confirmed_by_ai']) == (2, 2, 2)
    # The same rows still make the old cards (render_index): proposals said only while there is something to review.
    page = render_index(list(routes.values()), db_path(root), ai_reviewer=False)
    assert '2 with proposals' in card_of(page, ROUTE) and '<span class="by-ai">1 by AI</span>' in card_of(page, ROUTE)
    assert 'with proposals' not in card_of(page, OTHER_ROUTE) and '2 by AI' in card_of(page, OTHER_ROUTE)


def test_route_list_offers_review_with_ai_on_each_drive_left_once_a_reviewer_is_chosen():
    def summary(route, confirmed, by_ai=0):
        return {'route_id': route, 'run_id': RUN_ID, 'frame_count': 4, 'frames_with_proposals': 2,
                'confirmed_count': confirmed, 'confirmed_by_ai': by_ai, 'suspicious_count': 0, 'hint_frames': 0,
                'url': '/route/' + route + '/', 'drive': {'title': 'Herznach → Zürich', 'when': 'Fri 23 Oct 2026 · 08:01',
                                                          'minutes': 49}, 'proposer': 'qwen'}
    page = render_index([summary(ROUTE, 1), summary(OTHER_ROUTE, 4, by_ai=4)], Path('review.sqlite'), ai_reviewer=True)
    assert 'Set up AI review' not in page
    assert page.count('data-ai-route=') == 1 and 'data-ai-route="' + ROUTE + '"' in page
    assert 'signs by Qwen3-VL' in page and '4 by AI' in page
    # Without one, nobody is offered a button that cannot work.
    page = render_index([summary(ROUTE, 1)], Path('review.sqlite'), ai_reviewer=False)
    assert 'data-ai-route="' not in page and 'Set up AI review' in page
    # Our own networks, mid-sentence: in lower case.
    page = render_index([{**summary(ROUTE, 1), 'proposer': 'ours'}], Path('review.sqlite'))
    assert 'signs by our networks' in page
    # Every drive done: nothing to review with an AI, no link either.
    page = render_index([summary(OTHER_ROUTE, 4, by_ai=4)], Path('review.sqlite'), ai_reviewer=False)
    assert 'Set up AI review' not in page


def test_route_list_shows_the_latest_drive_first():
    def summary(route, start):
        return {'route_id': route, 'run_id': RUN_ID, 'frame_count': 4, 'frames_with_proposals': 0,
                'confirmed_count': 0, 'confirmed_by_ai': 0, 'suspicious_count': 0, 'hint_frames': 0,
                'url': '/route/' + route + '/', 'drive': {'title': None, 'when': None, 'minutes': None, 'start': start},
                'proposer': 'ours'}
    page = render_index([summary('0000011a--16ed4c88ac', '2026-10-27T08:00+00:00'),
                         summary('00000154--fa7e8f49aa', '2026-10-30T16:20+00:00'),
                         summary('00000150--a7e71fb876', None),
                         summary('00000153--44d221e479', '2026-10-30T13:30+00:00')], Path('review.sqlite'))
    order = list(dict.fromkeys(re.findall(r'href="/route/([^/"]+)/"', page)))
    # A drive whose start is unknown goes last.
    assert order == ['00000154--fa7e8f49aa', '00000153--44d221e479', '0000011a--16ed4c88ac', '00000150--a7e71fb876']


def payload_of(page: str) -> dict:
    found = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S)
    assert found
    return json.loads(found.group(1))


def test_review_page_knows_who_proposed_the_signs_and_the_drive_apart(served):
    server, root = served
    run = root / 'derived/preannotations' / ROUTE / RUN_ID / 'run.json'
    run.write_text(json.dumps({'run_id': RUN_ID, 'route_id': ROUTE, 'proposer': 'qwen'}))
    payload = payload_of(server.request('GET', f'/route/{ROUTE}/')[2].decode())
    assert payload['proposer'] == 'qwen'
    link = next(item for item in payload['routes'] if item['route_id'] == ROUTE)
    # The two lines of the route menu (nothing is known of the drive in the test).
    assert {'title', 'when', 'minutes', 'name'} <= set(link) and link['title'] is None
    assert payload_of(server.request('GET', f'/route/{OTHER_ROUTE}/')[2].decode())['proposer'] == 'vietnam'


def scan(root, route, passive):
    """A drive whose logs are read: sunnypilot off in all of it (passive True) or on."""
    import sqlite3
    from nnslr_tools import drive_review
    path = drive_review.db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for statement in drive_review.SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute("INSERT INTO scans (route, state, summary) VALUES (?, 'done', ?)",
                       (route, json.dumps({'t0_ns': 0, 'span_ns': [0, 40 * 10**9],
                                           'meta': {'car': {'brand': 'psa', 'passive': passive}}})))
    connection.commit()
    connection.close()


def test_a_dashcam_drive_says_dashcam_mode_on_the_list_and_in_the_route_menu(served):
    server, root = served
    scan(root, ROUTE, True)
    scan(root, OTHER_ROUTE, False)
    routes = json.loads(server.request('GET', '/api/routes')[2])['routes']
    assert {r['route_id']: r['dashcam'] for r in routes} == {ROUTE: True, OTHER_ROUTE: False}
    page = render_index(routes, db_path(root))
    assert 'DASHCAM MODE' in card_of(page, ROUTE) and 'DASHCAM MODE' not in card_of(page, OTHER_ROUTE)
    links = {item['route_id']: item for item in payload_of(server.request('GET', f'/route/{ROUTE}/')[2].decode())['routes']}
    assert links[ROUTE]['dashcam'] is True and links[OTHER_ROUTE]['dashcam'] is False


def test_a_drive_whose_logs_are_not_read_has_no_dashcam_mode(served):
    server, root = served
    routes = json.loads(server.request('GET', '/api/routes')[2])['routes']
    assert all(r['dashcam'] is None for r in routes)
    assert 'class="dashcam-badge"' not in render_index(routes, db_path(root))
