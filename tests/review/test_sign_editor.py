"""Sign review's editor on the Signs side of Speed review: GET /api/route/<route>/page gives it what the page of Sign
review starts from, and pages.sign_editor_css its style kept to .sign-editor (both in one place with Sign review's own).

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools import pages  # noqa: E402
from nnslr_tools.review_access import guest_rule  # noqa: E402
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, RunningServer, json_of, make_report, page_body  # noqa: E402


def test_the_editor_gets_the_data_of_the_review_page_without_the_menu_of_the_drives(tmp_path):
    rows = make_report(tmp_path)
    make_report(tmp_path, route=OTHER_ROUTE, count=2)
    server = RunningServer(tmp_path)
    try:
        status, response, data = server.request('GET', f'/api/route/{ROUTE}/page')
        assert status == 200 and response.getheader('Content-Type').startswith('application/json')
        answer = json_of(data)
        assert answer['ok'] is True and answer['run_id'] == RUN_ID and answer['route_id'] == ROUTE
        assert answer['server'] is True and answer['routes'] == []
        assert [row['frame_key'] for row in answer['frames']] == [row['frame_key'] for row in rows]
        # The pictures are relative to the drive's page: the editor puts /route/<drive>/ before them.
        assert all(row['preview_path'].startswith('images/') for row in answer['frames'])
        assert {'hints', 'hint_stats', 'suspicious', 'videos', 'busy', 'proposer', 'countries'} <= set(answer)
        # The same data as the page of Sign review embeds, menu apart.
        status, _response, page = server.request('GET', f'/route/{ROUTE}/')
        assert status == 200 and f'"run_id": "{RUN_ID}"'.encode() in page
        # An unknown drive: a JSON 404 the editor takes as "no editor here".
        status, _response, data = server.request('GET', '/api/route/00000009--abcdef1234/page')
        assert status == 404 and json_of(data)['ok'] is False
        # It reads only.
        assert server.request('POST', f'/api/route/{ROUTE}/page', body={})[0] in (403, 404, 405)
    finally:
        server.close()


def test_the_editor_saves_through_the_endpoint_of_sign_review_and_meets_its_conflicts(tmp_path):
    rows = make_report(tmp_path)
    server = RunningServer(tmp_path)
    try:
        status, _response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0], value_kph=60))
        assert status == 200
        revision = json_of(data)['review']['revision']
        # A second save made on the revision before it (another tab, another person): 409, the editor reads it again.
        stale = page_body(rows[0], value_kph=80)
        stale['base_revision'] = revision - 1
        status, _response, data = server.post_frame(ROUTE, '0/0', stale)
        assert status == 409 and json_of(data)['error'] == 'revision_conflict'
        status, _response, data = server.request('GET', f'/api/route/{ROUTE}/reviews')
        saved = {item['frame_key']: item for item in json_of(data)['reviews']}
        assert saved['0/0']['revision'] == revision and saved['0/0']['reviewer'] == 'Cristian'
    finally:
        server.close()


def test_a_guest_opens_the_editor_only_with_the_signs_given():
    # Signs are reviewed on the Signs side of Speed review (9 Nov 2026): the editor's data with the drive's signs given,
    # the player of the drive with its signs or its speed, the speed's saves with its speed only.
    assert guest_rule('GET', f'/api/route/{ROUTE}/page') == (ROUTE, ('signs',))
    assert guest_rule('GET', f'/api/speed-review/drives/{ROUTE}/signs') == (ROUTE, ('signs',))
    assert guest_rule('GET', f'/route/{ROUTE}/') == (ROUTE, ('signs',))
    assert guest_rule('GET', f'/api/speed-review/drives/{ROUTE}') == (ROUTE, ('speed', 'signs'))
    assert guest_rule('POST', f'/api/speed-review/drives/{ROUTE}/blocks') == (ROUTE, ('speed',))


def test_the_style_of_sign_review_is_kept_to_the_editor():
    css = pages.sign_editor_css()
    assert css.strip()
    # Every rule under .sign-editor: none reaches the page around it.
    for prelude in re.findall(r"(?:^|})\s*([^{}@\s][^{}]*)\{", css):
        for selector in prelude.split(','):
            selector = selector.strip()
            if selector and not re.match(r'(from|to|[0-9.]+%)$', selector):
                assert selector.startswith('.sign-editor'), selector
    # Its tokens on the editor, nothing for the page's body, no layout of the window's size.
    assert ':root' not in css and not re.search(r'(^|[\s,}])body\b', css)
    assert '.sign-editor {' in css and '--inspector:' in css
    for query in re.findall(r'@media([^{]*)\{', css):
        assert 'width' not in query and 'height' not in query, query
    assert '@media (pointer: coarse)' in css
    # The editor's own pieces are there: the photo, its boxes, the chips, the confirm block.
    for piece in ('.sign-editor #scene', '.sign-editor .box.selected', '.sign-editor .chip', '.sign-editor .confirm-area'):
        assert piece in css, piece
    # Animations stay as they are (their names are global).
    assert '@keyframes busy-breathe' in css


def test_the_pages_with_speed_review_carry_the_style_of_the_editor(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    try:
        status, _response, data = server.request('GET', '/speed-review')
        page = data.decode()
        assert status == 200 and pages.SIGN_EDITOR_MARK not in page and '__NNSLR_' not in page
        assert '.sign-editor #scene' in page
        # Sign review's own page keeps its style as it is.
        status, _response, data = server.request('GET', f'/route/{ROUTE}/')
        assert status == 200 and b'.sign-editor' not in data
    finally:
        server.close()
    # Before a server that knows the mark replaces it, it is a comment: the page's style stays valid.
    assert pages.SIGN_EDITOR_MARK in pages.asset('control.html')
    assert pages.SIGN_EDITOR_MARK.startswith('/*') and pages.SIGN_EDITOR_MARK.endswith('*/')
