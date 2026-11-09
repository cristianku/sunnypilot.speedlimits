"""The page of Sign review retired (9 Nov 2026): signs are reviewed on the Signs side of Speed review, and the old
addresses lead there at the same moment and frame. Only a route Speed review cannot show (no drive's name, or no road
video on the server) keeps its page, so that no frame is out of reach.

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from review_helpers import (
    OTHER_ROUTE,
    ROUTE,
    RunningServer,
    json_of,
    make_report,
    page_body,
)

from nnslr_tools import pages

SIGNS = f'/speed-review?drive={ROUTE}&side=signs'


def with_video(root: Path, route: str = ROUTE) -> None:
    """A minute of road video of the drive on the server: the player of Speed review can show it."""
    (root / 'raw/routes' / route / '0').mkdir(parents=True)
    (root / 'raw/routes' / route / '0' / 'fcamera.hevc').write_bytes(b'video')


@pytest.fixture
def served(tmp_path):
    rows = make_report(tmp_path)
    make_report(tmp_path, route=OTHER_ROUTE, count=2)
    with_video(tmp_path)
    server = RunningServer(tmp_path)
    yield server, rows
    server.close()


def location(server, path: str) -> str:
    status, response, _data = server.request('GET', path)
    assert status == 303, (path, status)
    return response.getheader('Location')


def test_the_address_of_the_signs_side():
    assert pages.signs_address(ROUTE) == SIGNS
    assert pages.signs_address(ROUTE, segment=3, second=12.5, frame='3/12') == SIGNS + '&segment=3&second=12.5&frame=3/12'
    assert pages.signs_address(ROUTE, segment=0, second=0) == SIGNS + '&segment=0&second=0'


def test_the_page_of_a_drive_leads_to_its_signs_side_at_the_moment_and_frame_asked(served):
    server, _rows = served
    assert location(server, f'/route/{ROUTE}/?segment=0&second=2&frame=0/2') == SIGNS + '&segment=0&second=2&frame=0/2'
    assert location(server, f'/route/{ROUTE}/?frame=0/1') == SIGNS + '&frame=0/1'
    assert location(server, f'/route/{ROUTE}/?segment=0&second=1.5') == SIGNS + '&segment=0&second=1.5'
    # What is not a moment or a frame is left out.
    assert location(server, f'/route/{ROUTE}/?segment=x&frame=../../etc&frame=0/1') == SIGNS + '&segment=0&second=0&frame=0/0'


def test_without_a_moment_it_opens_at_the_first_frame_still_to_review(served):
    server, rows = served
    first = SIGNS + '&segment=0&second=0&frame=0/0'
    assert location(server, f'/route/{ROUTE}/') == first
    # Every address of the page, with and without its slash.
    assert location(server, f'/route/{ROUTE}') == first
    assert location(server, f'/route/{ROUTE}/index.html') == first
    status, _response, data = server.post_frame(ROUTE, '0/0', page_body(rows[0]))
    assert status == 200, data
    assert location(server, f'/route/{ROUTE}/') == SIGNS + '&segment=0&second=1&frame=0/1'


def test_a_route_speed_review_cannot_show_keeps_its_page(served):
    server, _rows = served
    # No road video of it on the server: the page of Sign review, as before.
    status, _response, data = server.request('GET', f'/route/{OTHER_ROUTE}/')
    assert status == 200 and b'"run_id"' in data
    assert location(server, f'/route/{OTHER_ROUTE}') == f'/route/{OTHER_ROUTE}/'
    links = {item['route_id']: item['url'] for item in json_of(server.request('GET', '/api/routes')[2])['routes']}
    assert links == {ROUTE: SIGNS, OTHER_ROUTE: f'/route/{OTHER_ROUTE}/'}


def test_the_route_list_and_the_menu_lead_to_speed_review(served):
    server, _rows = served
    assert location(server, '/routes') == '/speed-review'
    assert location(server, '/index.html') == '/speed-review'
    assert location(server, '/routes?drive=' + ROUTE) == '/speed-review?drive=' + ROUTE
    # The menu of the built page has no entry to the list any more (/api/routes stays: the Signs card reads it).
    page = server.request('GET', '/speed-review')[2].decode()
    assert re.search(r'''["'`]/routes["'`?]''', page) is None
    assert "'/speed-review'" in page or '"/speed-review"' in page or '`/speed-review`' in page


def test_an_unknown_drive_leads_back_to_speed_review(served):
    server, _rows = served
    status, _response, data = server.request('GET', '/route/ffffffff--ffffffffff/')
    assert status == 404 and b'href="/speed-review">\xe2\x86\x90 Speed review</a>' in data


def test_the_style_of_review_with_ai_reaches_the_signs_side():
    css = pages.ai_sheet_css()
    assert '.ai-sheet' in css and '.ai-choices{' in css and '.ios-list' in css
    # Not the AI's tags of Sign review's photo, nor the page's own rules.
    selectors = re.findall(r'(?:^|[}\n])([^{}@]+)\{', css)
    assert '.ai-tag' not in css and not any(re.match(r'\s*(body|html|:root)\b', item) for item in selectors)
