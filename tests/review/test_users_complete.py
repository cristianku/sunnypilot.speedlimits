""""Road users complete": the tick of a frame in Sign review (J) that says every person and two-wheeler from 24 px tall
and every vehicle from 40 px tall has its box there. Saved with the boxes (page, /api/v1, MCP), read back, never carried
by Apply previous, and on every dataset row of a ticked frame; reviews saved before it read as not ticked."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
from nnslr_tools.review_db import SCHEMA_VERSION, ReviewStore, db_path
from nnslr_tools.review_mcp import ReviewTools
from nnslr_tools.review_service import ReviewService, review_schema
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

WEB = Path(__file__).resolve().parents[2] / 'web/src/review'
AUTH = {'Authorization': 'Bearer ' + TOKEN}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, count=3)
    running = RunningServer(tmp_path)
    yield running, {row['frame_key']: row for row in rows}
    running.close()


def _payload(running, key):
    return {review['frame_key']: review for review in running.app.store.reviews(ROUTE, RUN_ID)}[key]['payload']


def test_the_page_saves_the_tick_and_reads_it_back(server, tmp_path):
    running, rows = server
    body = page_body(rows['0/0'])
    body['detections'][0].update(family='other_sign', value_kph=None, sign_type='road_user_car')
    body['users_complete'] = True
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200 and json_of(data)['review']['users_complete'] is True
    assert _payload(running, '0/0')['users_complete'] is True
    # A frame left unticked, and one saved by a page that does not know the tick: no field stored, read as false.
    for key, sent in (('0/1', False), ('0/2', None)):
        body = page_body(rows[key])
        if sent is not None:
            body['users_complete'] = sent
        status, response, data = running.post_frame(ROUTE, key, body)
        assert status == 200 and json_of(data)['review']['users_complete'] is False
        assert 'users_complete' not in _payload(running, key)
    # Unticked again on the same frame: the field goes.
    body = page_body(rows['0/0'], base_revision=1)
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200 and json_of(data)['review']['users_complete'] is False
    assert 'users_complete' not in _payload(running, '0/0')
    # Anything but true or false is refused.
    body = page_body(rows['0/2'], base_revision=1)
    body['users_complete'] = 'yes'
    status, response, data = running.post_frame(ROUTE, '0/2', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_schema'
    # The page reads it with the reviews of the route.
    status, response, data = running.request('GET', f'/api/route/{ROUTE}/reviews')
    assert status == 200 and {r['frame_key']: r['users_complete'] for r in json_of(data)['reviews']} == {
        '0/0': False, '0/1': False, '0/2': False}


def test_old_reviews_read_as_not_ticked_and_the_database_keeps_its_version(tmp_path):
    """No new column, no new version: the tick lives in the review's payload, so a database of today is read as it is
    and an older nnslr still opens a database written by this one."""
    make_report(tmp_path)
    store = ReviewStore(db_path(tmp_path))
    service = ReviewService(tmp_path, store)
    body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
            'detections': []}
    service.save_review(ROUTE, '0/1', body, client='test', channel='rest')
    assert service.get_frame(ROUTE, '0/1')['review']['users_complete'] is False
    connection = sqlite3.connect(db_path(tmp_path))
    try:
        columns = {row[1] for row in connection.execute('PRAGMA table_info(frame_reviews)')}
    finally:
        connection.close()
    assert 'users_complete' not in columns and SCHEMA_VERSION == 2


def test_rest_and_mcp_save_the_tick(server, tmp_path):
    running, rows = server
    body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
            'detections': [], 'users_complete': True}
    status, response, data = running.request('POST', f'/api/v1/routes/{ROUTE}/frames/0/1/review', body=body,
                                             headers=AUTH)
    assert status == 200, data
    assert json_of(data)['review']['users_complete'] is True
    status, response, data = running.request('GET', f'/api/v1/routes/{ROUTE}/frames/0/1', headers=AUTH)
    assert status == 200 and json_of(data)['review']['users_complete'] is True
    assert 'users_complete' in review_schema()['save_body']

    service = ReviewService(tmp_path, ReviewStore(db_path(tmp_path)))
    tools = ReviewTools(service, client='test')
    saved = tools.save_frame_review(ROUTE, '0/0', run_id=RUN_ID, base_revision=0, reviewer='Claude Code',
                                    detections=[{'detection_id': '0/0/0', 'applicability': 'own_road'}],
                                    users_complete=True)
    assert saved['review']['users_complete'] is True and saved['review']['channel'] == 'mcp'
    # Left out, it is false: a save says the whole frame, the tick too.
    saved = tools.save_frame_review(ROUTE, '0/0', run_id=RUN_ID, base_revision=1, reviewer='Claude Code',
                                    detections=[{'detection_id': '0/0/0', 'applicability': 'own_road'}])
    assert saved['review']['users_complete'] is False


def test_apply_previous_never_carries_the_tick(server, tmp_path):
    running, rows = server
    body = page_body(rows['0/0'])
    body['users_complete'] = True
    status, response, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200 and json_of(data)['review']['users_complete'] is True
    service = ReviewService(tmp_path, ReviewStore(db_path(tmp_path)))
    carried = service.apply_previous(ROUTE, '0/1', {'run_id': RUN_ID})
    assert carried['detections'] and 'users_complete' not in carried
    # The frame after, saved with the carried boxes as the page sends them: not ticked.
    body = page_body(rows['0/1'])
    body['detections'] = [{k: v for k, v in d.items() if k not in ('check_position', 'carried_from')}
                          for d in carried['detections']]
    status, response, data = running.post_frame(ROUTE, '0/1', body)
    assert status == 200 and json_of(data)['review']['users_complete'] is False


def test_every_dataset_row_of_a_ticked_frame_says_so(server, tmp_path):
    running, rows = server
    # A person boxed, the proposal of the frame removed; and a frame with no box at all, ticked: no road user there.
    body = page_body(rows['0/0'], drop_proposals=True)
    body['detections'] = [{'bbox_xyxy': [2, 2, 10, 20], 'family': 'other_sign', 'sign_type': 'road_user_person',
                           'value_kph': None, 'applicability': 'unresolved', 'detection_id': 'manual-1'}]
    body['users_complete'] = True
    assert running.post_frame(ROUTE, '0/0', body)[0] == 200
    body = page_body(rows['0/1'])
    body['users_complete'] = True
    assert running.post_frame(ROUTE, '0/1', body)[0] == 200
    assert running.post_frame(ROUTE, '0/2', page_body(rows['0/2']))[0] == 200
    ticked = reviewed_frame_annotations(_payload(running, '0/0'), '0' * 64, dataset_kind='gold', name='test')
    # The person kept, the speed proposal rejected: both rows carry the tick.
    assert sorted(row['review_state'] for row in ticked) == ['corrected', 'rejected']
    assert all(row['users_complete'] is True for row in ticked)
    empty = reviewed_frame_annotations(_payload(running, '0/1'), '0' * 64, dataset_kind='gold', name='test')
    assert [(row['bbox_xyxy'], row['sign_family'], row['users_complete']) for row in empty] == [
        (None, 'not_a_sign', True)]
    plain = reviewed_frame_annotations(_payload(running, '0/2'), '0' * 64, dataset_kind='gold', name='test')
    assert all('users_complete' not in row for row in plain)
    assert validate_dataset(ticked + empty + plain, tmp_path, dataset_kind='gold')['valid'] is True
    # Only true or false.
    report = validate_dataset([{**plain[0], 'users_complete': 'yes'}], tmp_path, dataset_kind='gold')
    assert report['valid'] is False and report['errors'][0]['reason'] == 'invalid_schema'


def test_the_page_has_the_keys_and_the_toggle():
    keyboard = (WEB / 'use-keyboard.ts').read_text()
    assert "{w: 'road_user_person', y: 'road_user_bicycle', k: 'road_user_car'}" in keyboard
    assert "case 'j':" in keyboard and 'toggleUsersComplete()' in keyboard
    editor = (WEB / 'BoxEditor.vue').read_text()
    assert "road_user_person: 'W', road_user_bicycle: 'Y', road_user_car: 'K'" in editor
    help_keys = (WEB / 'ConfirmArea.vue').read_text()
    assert all(f'<kbd>{key}</kbd>' in help_keys for key in 'WYKJ')
    assert 'id="usersComplete"' in (WEB / 'Inspector.vue').read_text()
    # The tick goes with every save of the page, and Apply previous carries boxes only.
    assert "users_complete: s.users_complete === true," in (WEB / 'server.ts').read_text()
    assert 'users_complete' not in (WEB / 'carry.ts').read_text()
