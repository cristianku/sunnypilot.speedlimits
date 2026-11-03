"""Optional real-browser check of the review page served by the review server (skipped without a browser:
NNSLR_TEST_CHROMIUM, and NNSLR_TEST_BROWSER_PYTHON with Playwright): the keys of an open menu stay in the menu,
Del on a box not chosen yet, a box an AI added, the doubt filter shown under the photo, the same sign drawn twice,
a drive with nothing left to review, the bar under the photo between 1280 and 1700 px, the phone's Confirm."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, RunningServer, make_report

CODE = r'''
import sys
from playwright.sync_api import sync_playwright

base, chromium, route, other = sys.argv[1:5]
errors = []


def open_page(browser, path, width, height=1000):
    page = browser.new_page(viewport={'width': width, 'height': height})
    page.on('pageerror', lambda err: errors.append(str(err)))
    page.goto(base + path)
    page.wait_for_selector('body[data-server="ready"]')
    return page


def data(page, name):
    return page.evaluate('(name) => document.body.dataset[name] ?? null', name)


def until(page, expression):
    """wait_for_function evaluates a string, which the page's Content-Security-Policy refuses: poll instead."""
    for _ in range(100):
        if page.evaluate(expression):
            return
        page.wait_for_timeout(100)
    raise AssertionError('never true: ' + expression)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=chromium, headless=True)
    page = open_page(browser, '/route/' + route + '/', 1500)
    pos = lambda: page.locator('#pos').inner_text()
    assert pos() == '1 / 4', pos()
    assert page.locator('#frameData').inner_text().startswith('0:00 into the drive'), page.locator('#frameData').inner_text()
    assert page.locator('.route-title').inner_text() == route
    assert page.locator('#aiSetup').get_attribute('href') == '/account#ai-reviewer'
    page.locator('summary', has_text='More').click()
    assert 'Vietnamese roads' in page.locator('#proposedBy').inner_text()

    # The keys of the open Account menu are its own: the arrows move Appearance, Enter never confirms the frame.
    page.locator('#account-btn').click()
    page.wait_for_selector('#account-pop', state='visible')
    page.keyboard.press('ArrowRight')
    page.keyboard.press('ArrowRight')
    assert pos() == '1 / 4', pos()
    assert page.evaluate('document.documentElement.dataset.theme') == 'auto'
    page.keyboard.press('Enter')
    assert data(page, 'lastSaveResult') is None
    page.keyboard.press('Escape')
    page.wait_for_selector('#account-pop', state='hidden')
    page.keyboard.press('ArrowRight')
    assert pos() == '2 / 4', pos()

    # 0/1: confirmed by an AI with a box of its own, not "by hand".
    assert 'added by the AI (Claude)' in page.locator('#boxList').inner_text()
    assert page.locator('#evidence').inner_text() == 'Box added by the AI (Claude).'
    # Del on the box the page opened on: it says to pick it first, the box stays.
    page.keyboard.press('Delete')
    assert 'Click box 1 first' in page.locator('#saveMsg').inner_text()
    assert page.locator('#boxList .box-row').count() == 1

    # A doubt counter of the timeline in use is shown pressed under the photo; × goes back to To review.
    page.locator('#qUnknown').click()
    chip = page.locator('.filters .doubt-on')
    assert chip.inner_text().startswith('1 without a number') and chip.get_attribute('aria-pressed') == 'true'
    chip.click()
    assert page.locator('.filters .doubt-on').count() == 0
    assert page.locator('.filters [data-filter="todo"]').get_attribute('aria-pressed') == 'true'

    # 0/2 has the same sign twice: the first Confirm asks, the second saves both boxes.
    page.locator('#timeline button[data-key="0/2"]').click()
    page.locator('#reviewer').fill('Tester')
    page.locator('#confirm').click()
    assert data(page, 'lastSaveResult') == 'need-duplicate-check', data(page, 'lastSaveResult')
    assert 'Box 2 looks like the same sign as box 1' in page.locator('#saveMsg').inner_text()
    page.locator('#confirm').click()
    until(page, "document.body.dataset.lastSaveKey === '0/2' && document.body.dataset.lastSaveResult === 'ok'")

    # The state tag and the frame data are never cut, at any width of a computer.
    for width in (1024, 1280, 1440, 1500, 1600, 1610, 1700, 1920):
        page.set_viewport_size({'width': width, 'height': 1000})
        page.wait_for_timeout(150)
        sizes = page.evaluate("[...document.querySelectorAll('#statePill, #frameData')]"
                              ".map((e) => [e.id, e.scrollWidth, e.clientWidth])")
        assert all(scroll <= client + 1 for _, scroll, client in sizes), (width, sizes)

    # A drive with every frame confirmed opens on All and says so.
    done = open_page(browser, '/route/' + other + '/', 1500)
    until(done, "document.querySelector('.filters [data-filter=\"all\"]').getAttribute('aria-pressed') === 'true'")
    assert 'All 2 frames are confirmed (2 by AI).' in done.locator('#saveMsg').inner_text()
    assert done.locator('#confirmLabel').inner_text() == 'Already confirmed · next →'
    # On the phone the same button says where it goes, unlike ‹ › beside it.
    assert done.locator('#mConfirmLabel').inner_text() == 'Next frame →'

    # The phone: Confirm on a frame without boxes says what it does.
    phone = open_page(browser, '/route/' + route + '/', 390, 844)
    for _ in range(3):
        phone.keyboard.press('ArrowRight')
    assert phone.locator('#mConfirmLabel').inner_text() == 'Confirm no sign'
    browser.close()
assert not errors, errors
'''


def test_review_page_in_a_browser(tmp_path):
    python = os.environ.get('NNSLR_TEST_BROWSER_PYTHON', sys.executable)
    chromium = os.environ.get('NNSLR_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('optional browser environment not configured')
    if subprocess.run([python, '-c', 'import playwright.sync_api'], capture_output=True).returncode:
        pytest.skip('optional Playwright environment not configured')
    make_report(tmp_path, count=4)
    # 0/2 has its proposal twice: the same sign drawn twice.
    report = tmp_path / 'derived/preannotations' / ROUTE / RUN_ID / 'preannotations.jsonl'
    rows = [json.loads(line) for line in report.read_text().splitlines()]
    rows[2]['detections'].append(dict(rows[2]['detections'][0], detection_id='0/2/1', bbox_xyxy=[2, 2, 15, 20]))
    report.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    make_report(tmp_path, route=OTHER_ROUTE, count=2)
    server = RunningServer(tmp_path)
    try:
        def ai_save(route, key, detections):
            body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai',
                    'confirmed': True, 'detections': detections}
            server.app.save_review(route, key, body, client='test', channel='rest')
        # A box of the AI's own, its number not readable: a doubt "without a number".
        ai_save(ROUTE, '0/1', [{'bbox_xyxy': [2, 2, 12, 12], 'family': 'maximum_speed', 'value_kph': None,
                                'applicability': 'own_road'}])
        ai_save(OTHER_ROUTE, '0/0', [{'detection_id': '0/0/0', 'applicability': 'own_road'}])
        ai_save(OTHER_ROUTE, '0/1', [])
        result = subprocess.run([python, '-c', CODE, f'http://127.0.0.1:{server.port}', chromium, ROUTE, OTHER_ROUTE],
                                capture_output=True, text=True, timeout=180)
    finally:
        server.close()
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
