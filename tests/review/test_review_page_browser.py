"""Optional real-browser check of the review page served by the review server (skipped without a browser:
NNSLR_TEST_CHROMIUM, and NNSLR_TEST_BROWSER_PYTHON with Playwright): the keys of an open menu stay in the menu,
Del on a box not chosen yet, a box an AI added, the doubt filter shown under the photo, the same sign drawn twice,
a drive with nothing left to review, the bar under the photo between 1280 and 1700 px, the phone's Confirm, the box
as the classifier gets it."""
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

    # A doubt counter under the drive's bar in use is shown pressed under the photo; × goes back to To review.
    page.locator('#qUnknown').click()
    chip = page.locator('.filters .doubt-on')
    assert chip.inner_text().startswith('1 without a number') and chip.get_attribute('aria-pressed') == 'true'
    chip.click()
    assert page.locator('.filters .doubt-on').count() == 0
    assert page.locator('.filters [data-filter="todo"]').get_attribute('aria-pressed') == 'true'

    # 0/2 has the same sign twice: the first Confirm asks, the second saves both boxes.
    page.locator('#scrubber').evaluate("(el) => { el.value = '2'; el.dispatchEvent(new Event('input', {bubbles: true})); }")
    page.wait_for_function("() => document.querySelector('#scrubber').value === '2'")
    # The panel under the sign in one tap, and a second tap takes it back.
    hours = page.locator('#panels [data-panel="time"]')
    hours.click()
    assert hours.get_attribute('aria-pressed') == 'true' and page.locator('#supplementary_panel').input_value() == 'time'
    assert page.locator('#plateNote').inner_text() == 'The box takes the disc and the plate.'
    # The hours on the plate: the box takes them with both chosen, and From alone waits.
    page.locator('#hoursFrom').select_option('22')
    assert page.locator('#hoursFrom').input_value() == '22' and '22-6' not in page.locator('#boxList').inner_text()
    page.locator('#hoursTo').select_option('6')
    assert 'hours 22-6' in page.locator('#boxList').inner_text()
    # Fit box looks for the disc: a box with its plate stays as it is.
    page.locator('#fitLocal').click()
    assert 'The box takes the disc and the plate: kept as it is.' in page.locator('body').inner_text()
    hours.click()
    assert hours.get_attribute('aria-pressed') == 'false' and page.locator('#supplementary_panel').input_value() == 'unknown'
    assert page.locator('#plateNote').count() == 0 and page.locator('#plateHours').count() == 0
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
    # « » are in the box under the photo too, and the page stays as wide as the screen.
    assert phone.locator('#prevSecond').is_visible() and phone.locator('#nextSecond').is_visible()
    assert phone.evaluate('document.documentElement.scrollWidth - innerWidth') <= 0

    # Beside the zoom, the box as the classifier gets it: rounded outwards as on the car, following a corner dragged.
    # The fixture's preview is no picture, so only the crop is checked here, and that a bad photo breaks nothing.
    square = open_page(browser, '/route/' + route + '/', 1500)
    square.locator('#boxList .box-main').first.click()
    caption = square.locator('#netView figcaption')
    assert caption.inner_text() == 'Classifier · 14 × 18 px', caption.inner_text()
    assert square.evaluate("[netViewPicture.width, netViewPicture.height]") == [160, 160]
    corner = square.locator('#zoom .handle-hit.se').bounding_box()
    square.mouse.move(corner['x'] + corner['width'] / 2, corner['y'] + corner['height'] / 2)
    square.mouse.down()
    square.mouse.move(corner['x'] - 40, corner['y'] - 40, steps=4)
    assert caption.inner_text() != 'Classifier · 14 × 18 px', 'the square did not follow the corner while dragging'
    square.mouse.up()

    # Car networks: a number with hours says so, and the plain disc of the frame under it does not use up that reading.
    car = open_page(browser, '/route/' + route + '/', 1500)
    tall = {'box': [1, 2, 15, 40], 'score': .9, 'class': 'maximum_speed_hours:30', 'confidence': .95, 'speed': 30,
            'use': 'speed', 'review': {'family': 'maximum_speed', 'value_kph': 30, 'supplementary_panel': 'time'}}
    plain = {**tall, 'class': 'maximum_speed:30',
             'review': {'family': 'maximum_speed', 'value_kph': 30}}
    release = {'name': 'r', 'cutter': None, 'classifier': None, 'models_commit': None, 'sunnypilot_commit': None,
               'min_confidence': .5, 'min_box_score': .3, 'max_boxes': 8}
    car.route('**/car/**', lambda request: request.fulfill(json={
        'ok': True, 'frame_key': '0/0', 'signs': [tall, plain], 'surest': 0, 'width': 1344, 'height': 760, 'release': release,
        'ms': 5}))
    # Off a frame with its own box where the car finds one: none added there.
    car.keyboard.press('t')
    car.wait_for_timeout(400)
    before = car.locator('#boxList .box-main').count()
    assert before == 1
    # A frame not confirmed takes them straight in its list, whom they are for too, and stays to review.
    false_box = {**plain, 'box': [18, 22, 26, 30], 'class': 'not_a_sign', 'speed': None, 'use': 'nothing',
                 'review': {'family': 'not_a_sign', 'value_kph': None}}
    free = {**plain, 'box': [30, 22, 38, 30], 'applicability': 'own_road', 'applicability_score': .9}
    car.unroute('**/car/**')
    car.route('**/car/**', lambda request: request.fulfill(json={
        'ok': True, 'frame_key': '0/3', 'signs': [free, false_box], 'surest': 0, 'width': 1344, 'height': 760,
        'release': release, 'ms': 5}))
    # 0/1 and 0/2 are confirmed.
    for _ in range(3):
        car.keyboard.press('ArrowRight')
    placed = car.locator('#boxList .box-main', has_text='car networks')
    placed.nth(1).wait_for()
    assert 'OURS' in placed.first.inner_text() and 'Not a sign' in placed.nth(1).inner_text()
    assert car.locator('#carList').count() == 0
    # Deleted, it does not come back on its own.
    placed.nth(1).click()
    car.locator('#remove').click()
    assert placed.count() == 1
    car.keyboard.press('ArrowLeft')
    car.keyboard.press('ArrowRight')
    car.wait_for_timeout(400)
    assert placed.count() == 1
    # Confirm saves it with the frame.
    car.locator('#reviewer').fill('Tester')
    car.locator('#confirm').click()
    until(car, "document.body.dataset.lastSaveResult === 'ok'")
    # Apply previous fits what it carries, as C does.
    carried = {'bbox_xyxy': [30, 22, 38, 30], 'family': 'maximum_speed', 'value_kph': 30, 'value_state': 'reviewed',
               'score': None, 'model_label': None, 'class_id': None, 'review_state': 'pending',
               'applicability': 'own_road', 'supplementary_panel': 'unknown', 'detection_id': 'car-1'}
    fits = []
    car.route('**/apply-previous', lambda route: route.fulfill(json={
        'ok': True, 'detections': [carried], 'to_check': [], 'message': 'Boxes carried.'}))
    car.route('**/fit', lambda route: (fits.append(route.request.post_data_json['boxes']), route.fulfill(json={
        'ok': True, 'boxes': [[31, 22, 39, 30]], 'ms': 1})))
    car.keyboard.press('ArrowRight')
    car.wait_for_timeout(400)
    car.keyboard.press('a')
    car.wait_for_timeout(500)
    assert fits == [[carried['bbox_xyxy']]], fits
    assert 'Boxes carried.' in car.locator('body').inner_text()
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
    # 0/0 has a box between the pixels, as the model's are: the classifier gets it rounded outwards.
    rows[0]['detections'][0]['bbox_xyxy'] = [1.2, 2.5, 14.6, 19.4]
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
