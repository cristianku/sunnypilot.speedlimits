"""REST API /api/v1: token in a header, frames in timeline order, images, saves with run_id and base_revision."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nnslr_tools.pages import asset

from nnslr_tools.review_db import ReviewStore, db_path, save_frame_review
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

# Frames on three segments with gaps: 0/0 0/2 0/5 | 1/1 1/3 | 3/0. Even positions have a proposal.
KEYS = [(0, 0), (0, 2), (0, 5), (1, 1), (1, 3), (3, 0)]
ORDER = ['0/0', '0/2', '0/5', '1/1', '1/3', '3/0']
AUTH = {'Authorization': 'Bearer ' + TOKEN}


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, keys=KEYS)
    # The report lines in another order: the API follows segment and image, not the file.
    report = tmp_path / 'derived/preannotations' / ROUTE / RUN_ID / 'preannotations.jsonl'
    report.write_text('\n'.join(reversed(report.read_text().splitlines())) + '\n')
    by_key = {}
    for row in rows:
        by_key[row['frame_key']] = row
    server = RunningServer(tmp_path)
    yield server, tmp_path, by_key
    server.close()


def get(server, path, headers=AUTH):
    status, response, data = server.request('GET', path, headers=headers)
    return status, response, data


def get_json(server, path):
    status, response, data = get(server, path)
    assert status == 200, data
    return json_of(data)


def save(server, route, key, body, headers=AUTH):
    status, response, data = server.request('POST', f'/api/v1/routes/{route}/frames/{key}/review', body=body,
                                             headers=headers)
    return status, json_of(data)


def ai_body(run_id=RUN_ID, base_revision=0, detections=None, **extra):
    body = {'run_id': run_id, 'base_revision': base_revision, 'reviewer': 'Claude', 'reviewer_kind': 'ai',
            'confirmed': True, 'detections': [] if detections is None else detections}
    body.update(extra)
    return body


def test_rest_service_preserves_original_mph_number_with_legacy_alias(tmp_path):
    from nnslr_tools.review_service import ReviewService
    make_report(tmp_path)
    service=ReviewService(tmp_path,ReviewStore(db_path(tmp_path)))
    result=service.save_review(ROUTE,'0/0',ai_body(country='US',detections=[{
        'detection_id':'0/0/0','value_kph':None,'sign_value':25,'sign_unit':'mph',
        'applicability':'own_road'}]),client='unit-test',channel='rest')
    d=result['review']['detections'][0]
    assert (d['value_kph'],d['sign_value'],d['sign_unit'])==(25,25,'mph')
    assert service.progress(ROUTE)['confirmed_with_doubts']['unknown_value']==[]


def test_rest_service_saves_a_panel_of_hours(tmp_path):
    from nnslr_tools.review_service import ReviewService
    make_report(tmp_path)
    service=ReviewService(tmp_path,ReviewStore(db_path(tmp_path)))
    result=service.save_review(ROUTE,'0/0',ai_body(detections=[{
        'detection_id':'0/0/0','value_kph':30,'applicability':'own_road','supplementary_panel':'time'}]),
        client='unit-test',channel='rest')
    assert result['review']['detections'][0]['supplementary_panel']=='time'


def test_api_needs_the_token_in_a_header_even_on_loopback(api):
    server, root, rows = api
    assert server.app.token is None and not server.app.require_token
    # The page on 127.0.0.1 needs no token, the API does.
    assert server.request('GET', '/api/routes')[0] == 200
    for headers in [{}, {'Authorization': 'Bearer wrong-token-0000000000'}, {'X-Review-Token': 'nope'},
                    {'Cookie': f'{server.app.cookie_name}={TOKEN}'}, {'Authorization': 'Basic ' + TOKEN}]:
        status, response, data = get(server, '/api/v1/routes', headers)
        assert status == 401, headers
        assert json_of(data)['error'] == 'unauthorized'
        assert response.getheader('WWW-Authenticate').startswith('Bearer')
    status, response, data = get(server, f'/api/v1/routes?token={TOKEN}', {})
    assert status == 400 and json_of(data)['error'] == 'token_in_url'
    assert response.getheader('Set-Cookie') is None
    assert get(server, '/api/v1/routes', {'X-Review-Token': TOKEN})[0] == 200
    status, response, data = get(server, '/api/v1/routes')
    assert status == 200 and response.getheader('Cache-Control') == 'no-store'
    status, body = save(server, ROUTE, '0/0', ai_body(), headers={})
    assert status == 401
    assert ReviewStore(db_path(root)).reviews(ROUTE, RUN_ID) == []


def test_api_on_the_lan_server_uses_the_same_token(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    try:
        assert server.app.api_token == TOKEN
        assert get(server, '/api/v1/routes', {})[0] == 401
        # The page cookie is not an API credential.
        assert get(server, '/api/v1/routes', {'Cookie': f'{server.app.cookie_name}={TOKEN}'})[0] == 401
        assert get(server, '/api/v1/routes')[0] == 200
        assert get(server, '/api/routes', {'X-Review-Token': TOKEN})[0] == 200
    finally:
        server.close()


def test_api_refuses_other_origins(api):
    server, root, rows = api
    for headers in [{'Origin': 'http://evil.example'}, {'Origin': 'null'},
                    {'Origin': f'http://127.0.0.1:{server.port}', 'Sec-Fetch-Site': 'cross-site'}]:
        headers.update(AUTH)
        status, response, data = get(server, '/api/v1/routes', headers)
        assert status == 403 and json_of(data)['error'] == 'cross_origin'
        status, body = save(server, ROUTE, '0/0', ai_body(), headers=headers)
        assert status == 403
    same = dict(AUTH, Origin=f'http://127.0.0.1:{server.port}')
    assert get(server, '/api/v1/routes', same)[0] == 200
    status, response, data = server.request('GET', '/api/v1/routes', headers=dict(AUTH, Host='evil.example'))
    assert status == 403


def test_routes_and_progress(api):
    server, root, rows = api
    make_report(root, route=OTHER_ROUTE, count=2)
    routes = get_json(server, '/api/v1/routes')['routes']
    first = next(r for r in routes if r['route_id'] == ROUTE)
    # The drive as the site names it; no videos or positions in the test: it has nothing to say.
    assert first == {'route_id': ROUTE, 'run_id': RUN_ID, 'frame_count': 6, 'confirmed_count': 0,
                     'confirmed_by_human': 0, 'confirmed_by_ai': 0, 'missing_count': 6, 'withdrawn_count': 0,
                     'suspicious_count': 0, 'frames_with_proposals': 3, 'hint_frames': 0,
                     'drive': {'when': None, 'minutes': None, 'start': None, 'from': None, 'to': None, 'title': None}}
    assert server.post_frame(ROUTE, '0/0', page_body(rows['0/0']))[0] == 200
    assert save(server, ROUTE, '1/3', ai_body())[0] == 200
    progress = get_json(server, f'/api/v1/routes/{ROUTE}')
    assert progress['confirmed_count'] == 2 and progress['confirmed_by_human'] == 1
    assert progress['confirmed_by_ai'] == 1 and progress['missing_count'] == 4
    assert progress['drive']['title'] is None
    assert progress['first_unreviewed'] == '0/2'
    # 0/2 has nothing and its confirmed neighbour 0/0 a speed sign (rule d).
    assert [s['frame_key'] for s in progress['suspicious']] == ['0/2']
    assert progress['suspicious'][0]['reasons'][0]['rule'] == 'd'
    assert progress['confirmed_with_doubts'] == {'unknown_value': [], 'unreadable': [],
                                                 'unresolved_road': ['0/0']}
    # The page endpoints keep their exact form.
    legacy = json_of(server.request('GET', '/api/routes')[2])['routes']
    assert set(next(r for r in legacy if r['route_id'] == ROUTE)) == {
        'route_id', 'run_id', 'frame_count', 'frames_with_proposals', 'confirmed_count', 'confirmed_by_ai',
        'waiting_for_person', 'suspicious_count', 'hint_frames', 'url', 'drive', 'proposer'}
    assert next(r for r in legacy if r['route_id'] == ROUTE)['confirmed_by_ai'] == 1
    assert get(server, '/api/v1/routes/00000009--abcdef1234')[0] == 404
    assert get(server, '/api/v1/routes/not-a-route')[0] == 404


def test_frames_are_paged_in_timeline_order_with_a_stable_cursor(api):
    server, root, rows = api
    seen = []
    cursor = None
    pages = 0
    while True:
        path = f'/api/v1/routes/{ROUTE}/frames?filter=all&limit=4'
        if cursor:
            path = path + '&cursor=' + cursor
        page = get_json(server, path)
        assert page['total'] == 6 and page['run_id'] == RUN_ID and page['filter'] == 'all'
        for frame in page['frames']:
            seen.append(frame['frame_key'])
        pages = pages + 1
        cursor = page['next_cursor']
        if cursor is None:
            break
    assert seen == ORDER and pages == 2
    first = get_json(server, f'/api/v1/routes/{ROUTE}/frames?filter=all&limit=2')
    assert [f['position'] for f in first['frames']] == [0, 1]
    assert first['frames'][0]['proposal_count'] == 1 and first['frames'][1]['proposal_count'] == 0
    with_proposals = get_json(server, f'/api/v1/routes/{ROUTE}/frames?filter=with_proposals')
    assert [f['frame_key'] for f in with_proposals['frames']] == ['0/0', '0/5', '1/3']
    # A save between two pages does not move the cursor of the "unreviewed" filter.
    page = get_json(server, f'/api/v1/routes/{ROUTE}/frames?limit=2')
    assert [f['frame_key'] for f in page['frames']] == ['0/0', '0/2']
    assert save(server, ROUTE, '0/5', ai_body())[0] == 200
    following = get_json(server, f'/api/v1/routes/{ROUTE}/frames?limit=2&cursor=' + page['next_cursor'])
    assert [f['frame_key'] for f in following['frames']] == ['1/1', '1/3'] and following['total'] == 5
    reviewed = get_json(server, f'/api/v1/routes/{ROUTE}/frames?filter=reviewed')
    assert [f['frame_key'] for f in reviewed['frames']] == ['0/5']
    assert reviewed['frames'][0]['reviewer'] == 'ai/Claude' and reviewed['frames'][0]['reviewer_kind'] == 'ai'
    for query, reason in [('filter=nope', 'invalid_parameter'), ('limit=0', 'invalid_parameter'),
                          ('limit=201', 'invalid_parameter'), ('cursor=garbage', 'invalid_cursor'),
                          ('filter=all&cursor=' + page['next_cursor'], 'invalid_cursor')]:
        status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames?' + query)
        assert status == 400 and json_of(data)['error'] == reason, query
    # A cursor of an older report.
    latest = root / 'derived/preannotations' / ROUTE / 'latest.json'
    other_run = root / 'derived/preannotations' / ROUTE / '20261022T100000Z-0123abcd'
    (root / 'derived/preannotations' / ROUTE / RUN_ID).rename(other_run)
    latest.write_text(json.dumps({'run_id': other_run.name}))
    status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames?limit=2&cursor=' + page['next_cursor'])
    assert status == 409 and json_of(data)['error'] == 'stale_report'


def test_next_frame_resumes_after_an_interruption(api):
    server, root, rows = api
    following = get_json(server, f'/api/v1/routes/{ROUTE}/next')
    assert following['remaining'] == 6 and following['frame']['frame_key'] == '0/0'
    assert save(server, ROUTE, '0/0', ai_body())[0] == 200
    assert server.post_frame(ROUTE, '0/2', page_body(rows['0/2']))[0] == 200
    # A new server on the same database (the work was interrupted) starts where it stopped.
    server.close()
    again = RunningServer(root)
    try:
        following = get_json(again, f'/api/v1/routes/{ROUTE}/next')
        assert following['frame']['frame_key'] == '0/5' and following['remaining'] == 4
        assert following['frame']['save'] == {'run_id': RUN_ID, 'base_revision': 0}
        following = get_json(again, f'/api/v1/routes/{ROUTE}/next?after=1/1')
        assert following['frame']['frame_key'] == '1/3' and not following['wrapped']
        following = get_json(again, f'/api/v1/routes/{ROUTE}/next?after=3/0')
        assert following['frame']['frame_key'] == '0/5' and following['wrapped']
        following = get_json(again, f'/api/v1/routes/{ROUTE}/next?filter=reviewed&after=0/0')
        assert following['frame']['frame_key'] == '0/2'
        for key in ['0/5', '1/1', '1/3', '3/0']:
            assert save(again, ROUTE, key, ai_body())[0] == 200
        following = get_json(again, f'/api/v1/routes/{ROUTE}/next')
        assert following == {'route_id': ROUTE, 'run_id': RUN_ID, 'filter': 'unreviewed', 'remaining': 0,
                             'wrapped': False, 'frame': None}
        assert get(again, f'/api/v1/routes/{ROUTE}/next?after=9/9')[0] == 404
    finally:
        again.close()


def test_frame_detail_and_context_across_segments(api):
    server, root, rows = api
    hints = root / 'derived/hints' / ROUTE
    hints.mkdir(parents=True)
    hints.joinpath('second.jsonl').write_text(json.dumps({
        'route_id': ROUTE, 'frame_key': '0/2', 'source': 'test', 'signs': [
            {'type': 'maximum_speed', 'value_kph': 60, 'box': [2, 2, 9, 9], 'confidence': 'high'}]}) + '\n')
    frame = get_json(server, f'/api/v1/routes/{ROUTE}/frames/0/5')
    assert frame['frame_key'] == '0/5' and frame['run_id'] == RUN_ID and frame['revision'] == 0
    assert frame['width'] == 32 and frame['height'] == 24 and frame['status'] == 'unreviewed'
    assert frame['proposals'][0]['detection_id'] == '0/5/0' and frame['review'] is None
    assert frame['previous_frame_key'] == '0/2' and frame['next_frame_key'] == '1/1'
    assert frame['metadata']['frame_sha256'] == rows['0/5']['frame_sha256']
    assert 'image_path' not in frame['metadata'] and 'detections' not in frame['metadata']
    assert frame['image']['preview_url'] == f'/api/v1/routes/{ROUTE}/frames/0/5/image?variant=preview'
    hinted = get_json(server, f'/api/v1/routes/{ROUTE}/frames/0/2')
    assert hinted['second_reading'][0]['value_kph'] == 60
    assert hinted['suspicious'] and 'second reading sees 60' in hinted['suspicious_reasons'][0]
    context = get_json(server, f'/api/v1/routes/{ROUTE}/frames/1/1/context?before=3&after=2')
    assert [(f['frame_key'], f['offset'], f['same_segment']) for f in context['frames']] == [
        ('0/0', -3, False), ('0/2', -2, False), ('0/5', -1, False), ('1/1', 0, True), ('1/3', 1, True),
        ('3/0', 2, False)]
    edge = get_json(server, f'/api/v1/routes/{ROUTE}/frames/3/0/context?before=1&after=5')
    assert [f['frame_key'] for f in edge['frames']] == ['1/3', '3/0']
    assert get(server, f'/api/v1/routes/{ROUTE}/frames/1/1/context?before=11')[0] == 400
    assert get(server, f'/api/v1/routes/{ROUTE}/frames/0/1')[0] == 404
    assert get(server, f'/api/v1/routes/{ROUTE}/frames/0/x')[0] == 404
    assert get(server, f'/api/v1/routes/{ROUTE}/frames/0/1/context')[0] == 404


def test_images_are_served_and_checked(api):
    server, root, rows = api
    status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames/0/5/image')
    assert status == 200 and response.getheader('Content-Type') == 'image/jpeg' and data == b'jpeg preview'
    status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames/0/5/image?variant=original')
    assert status == 200 and response.getheader('Content-Type') == 'image/png'
    assert data == (root / rows['0/5']['image_path']).read_bytes()
    assert get(server, f'/api/v1/routes/{ROUTE}/frames/0/5/image?variant=raw')[0] == 400
    assert get(server, f'/api/v1/routes/{ROUTE}/frames/0/5/image', {})[0] == 401
    (root / rows['1/1']['image_path']).write_bytes(b'changed')
    status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames/1/1/image?variant=original')
    assert status == 409 and json_of(data)['error'] == 'image_changed'


def test_crop_and_resize_with_pillow(api):
    pytest.importorskip('PIL')
    from PIL import Image
    import io
    server, root, rows = api
    status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames/0/0/image?variant=original'
                                         '&crop=1,2,15,20&long_side=64')
    assert status == 200 and response.getheader('Content-Type') == 'image/png'
    picture = Image.open(io.BytesIO(data))
    assert picture.size == (50, 64)
    for crop in ['1,2,3', '1,2,2,3', 'a,b,c,d', '40,40,50,50']:
        status, response, data = get(server, f'/api/v1/routes/{ROUTE}/frames/0/0/image?crop={crop}')
        assert status == 400 and json_of(data)['error'] == 'invalid_crop', crop


def test_images_outside_the_report_are_refused(api, tmp_path):
    server, root, rows = api
    directory = root / 'derived/preannotations' / ROUTE / RUN_ID
    (directory / 'images' / '00000002.jpg').unlink()
    (directory / 'images' / '00000002.jpg').symlink_to('/etc/passwd')
    # A report row whose image leaves derived/frames/<route> (a new run pointing outside).
    secret = root / 'secret.png'
    secret.write_bytes(b'secret')
    lines = []
    for line in (directory / 'preannotations.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['frame_key'] == '1/1':
            row['image_path'] = 'secret.png'
            row['frame_sha256'] = hashlib.sha256(b'secret').hexdigest()
        if row['frame_key'] == '1/3':
            row['image_path'] = '../' + root.name + '/' + row['image_path']
        if row['frame_key'] == '0/0':
            row['preview_path'] = '../run.json'
        lines.append(json.dumps(row))
    run = '20261022T100000Z-0123abcd'
    (root / 'derived/preannotations' / ROUTE / run).mkdir()
    for name in ['index.html', 'run.json']:
        (root / 'derived/preannotations' / ROUTE / run / name).write_bytes((directory / name).read_bytes())
    (root / 'derived/preannotations' / ROUTE / run / 'images').symlink_to(directory / 'images')
    (root / 'derived/preannotations' / ROUTE / run / 'preannotations.jsonl').write_text('\n'.join(lines) + '\n')
    (root / 'derived/preannotations' / ROUTE / 'latest.json').write_text(json.dumps({'run_id': run}))
    for path in [f'/api/v1/routes/{ROUTE}/frames/1/1/image?variant=original',
                 f'/api/v1/routes/{ROUTE}/frames/1/3/image?variant=original',
                 f'/api/v1/routes/{ROUTE}/frames/0/0/image',
                 f'/api/v1/routes/{ROUTE}/frames/1/1/image',
                 f'/api/v1/routes/{ROUTE}/frames/0/5/image',
                 f'/api/v1/routes/..%2F{ROUTE}/frames/0/0/image',
                 f'/api/v1/routes/..%2F..%2Fsecret.png/frames/0/0/image',
                 f'/api/v1/routes/{ROUTE}/frames/0/0/../../../../secret.png']:
        status, response, data = get(server, path)
        assert status == 404, path
        assert b'secret' not in data and b'root:' not in data


def test_rest_save_needs_run_id_base_revision_and_reviewer_kind(api):
    server, root, rows = api
    store = ReviewStore(db_path(root))
    body = ai_body()
    del body['run_id']
    status, problem = save(server, ROUTE, '0/0', body)
    assert status == 400 and problem['ok'] is False and problem['error'] == 'missing_run_id'
    status, problem = save(server, ROUTE, '0/0', ai_body(run_id='20200101T000000Z-00000000'))
    assert status == 409 and problem['error'] == 'stale_report'
    body = ai_body()
    del body['base_revision']
    status, problem = save(server, ROUTE, '0/0', body)
    assert status == 428 and problem['error'] == 'base_revision_required'
    body = ai_body()
    del body['reviewer_kind']
    status, problem = save(server, ROUTE, '0/0', body)
    assert status == 400 and problem['error'] == 'invalid_reviewer_kind'
    status, problem = save(server, ROUTE, '0/0', ai_body(reviewer_kind='robot'))
    assert status == 400 and problem['error'] == 'invalid_reviewer_kind'
    status, problem = save(server, ROUTE, '0/0', ai_body(reviewer_kind='human', reviewer='ai/Claude'))
    assert status == 400 and problem['error'] == 'invalid_reviewer'
    assert store.reviews(ROUTE, RUN_ID) == []
    status, saved = save(server, ROUTE, '0/0', ai_body(detections=[{'detection_id': '0/0/0',
                                                                   'applicability': 'own_road'}]))
    assert status == 200, saved
    review = saved['review']
    assert review['reviewer'] == 'ai/Claude' and review['reviewer_kind'] == 'ai' and review['channel'] == 'rest'
    assert review['revision'] == 1 and saved['status'] == 'confirmed' and saved['next_unreviewed_frame_key'] == '0/2'
    detection = review['detections'][0]
    # Fields not sent come from the proposal.
    assert detection['value_kph'] == 30 and detection['score'] == 0.8 and detection['applicability'] == 'own_road'
    status, problem = save(server, ROUTE, '0/0', ai_body(base_revision=0))
    assert status == 409 and problem['error'] == 'revision_conflict'
    status, saved = save(server, ROUTE, '0/0', ai_body(base_revision=1, reviewer='ai/Claude'))
    assert status == 200 and saved['review']['revision'] == 2 and saved['review']['reviewer'] == 'ai/Claude'
    status, saved = save(server, ROUTE, '0/2', ai_body(reviewer='Anna', reviewer_kind='human'))
    assert status == 200 and saved['review']['reviewer'] == 'Anna' and saved['review']['reviewer_kind'] == 'human'
    assert store.review(ROUTE, RUN_ID, '0/2')['payload']['label_basis'] == 'human_review'


def test_new_boxes_corrections_and_no_sign(api):
    server, root, rows = api
    status, saved = save(server, ROUTE, '0/0', ai_body(detections=[
        {'detection_id': '0/0/0', 'value_kph': 60, 'applicability': 'own_road'},
        {'bbox_xyxy': [16, 2, 30, 20], 'family': 'zone', 'value_kph': 30, 'applicability': 'other_road'},
        {'bbox_xyxy': [2, 12, 8, 20], 'family': 'unreadable', 'value_kph': None, 'notes': 'troppo lontano'}]))
    assert status == 200, saved
    detections = saved['review']['detections']
    assert [d['detection_id'] for d in detections] == ['0/0/0', 'manual-rest-1', 'manual-rest-2']
    assert detections[0]['value_kph'] == 60 and detections[0]['value_state'] == 'reviewed'
    assert detections[2]['value_state'] == 'unknown' and detections[2]['notes'] == 'troppo lontano'
    # Proposal removed, no sign: a negative frame.
    status, saved = save(server, ROUTE, '0/5', ai_body(detections=[]))
    assert status == 200 and saved['review']['detections'] == []
    for detection, reason in [({'detection_id': '0/0/0', 'value_kph': 3}, 'illegal_speed'),
                              ({'bbox_xyxy': [1, 2, 99, 20], 'family': 'zone', 'value_kph': 30}, 'invalid_bbox'),
                              ({'bbox_xyxy': [1, 2, 9, 20], 'family': 'logo', 'value_kph': None}, 'unknown_enum'),
                              ({'bbox_xyxy': [1, 2, 9, 20], 'family': 'other_sign', 'value_kph': 50},
                               'value_family_mismatch')]:
        status, problem = save(server, ROUTE, '0/0', ai_body(base_revision=1, detections=[detection]))
        assert status == 400 and problem['error'] == reason, detection
    assert ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0')['revision'] == 1


def test_ai_never_overwrites_a_human_review_unless_asked(api):
    server, root, rows = api
    assert server.post_frame(ROUTE, '0/0', page_body(rows['0/0'], reviewer='cristianku'))[0] == 200
    status, problem = save(server, ROUTE, '0/0', ai_body(base_revision=1))
    assert status == 409 and problem['error'] == 'human_review_exists'
    status, problem = save(server, ROUTE, '0/0', ai_body(base_revision=1, confirmed=False))
    assert status == 409 and problem['error'] == 'human_review_exists'
    status, saved = save(server, ROUTE, '0/0', ai_body(base_revision=1, override_human_review=True))
    assert status == 200 and saved['review']['reviewer'] == 'ai/Claude'
    history = get_json(server, f'/api/v1/routes/{ROUTE}/frames/0/0/history')['history']
    assert [(h['revision'], h['reviewer'], h['reviewer_kind'], h['channel']) for h in history] == [
        (1, 'cristianku', 'human', 'web'), (2, 'ai/Claude', 'ai', 'rest')]
    assert history[0]['detections'][0]['value_kph'] == 30 and history[1]['detections'] == []
    # A human may change an AI review; the page records it as human.
    assert server.post_frame(ROUTE, '0/0', page_body(rows['0/0'], base_revision=2))[0] == 200
    review = json_of(server.request('GET', f'/api/route/{ROUTE}/reviews')[2])['reviews'][0]
    assert review['reviewer'] == 'Cristian' and review['reviewer_kind'] == 'human' and review['channel'] == 'web'


def test_bad_requests_are_structured_json(api):
    server, root, rows = api
    path = f'/api/v1/routes/{ROUTE}/frames/0/0/review'
    status, response, data = server.request('POST', path, body=b'{"run_id": NaN}', headers=AUTH)
    assert status == 400 and json_of(data)['error'] == 'invalid_json'
    status, response, data = server.request('POST', path, body=ai_body(), headers=dict(AUTH, **{
        'Content-Type': 'text/plain'}))
    assert status == 415 and json_of(data)['ok'] is False
    status, response, data = server.request('POST', path, body=b'{}', headers=dict(AUTH, **{
        'Content-Length': str(512 * 1024 + 1)}))
    assert status == 413
    status, response, data = server.request('GET', path, headers=AUTH)
    assert status == 405 and json_of(data)['error'] == 'method_not_allowed'
    status, response, data = server.request('DELETE', f'/api/v1/routes/{ROUTE}', headers=AUTH)
    assert status == 405
    assert get(server, '/api/v1/nothing')[0] == 404
    schema = get_json(server, '/api/v1/schema')
    assert 'maximum_speed' in schema['detection']['family'] and schema['detection']['applicability'] == [
        'other_road', 'own_road', 'unresolved']
    index = get_json(server, '/api/v1')
    assert index['version'] == 1 and isinstance(index['mcp'], bool)


def test_version_1_database_is_migrated_in_place(tmp_path):
    path = db_path(tmp_path)
    path.parent.mkdir(parents=True)
    connection = sqlite3.connect(str(path))
    connection.executescript("""
        CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL);
        INSERT INTO schema_version VALUES (1, '2026-10-21T00:00:00.000Z');
        CREATE TABLE frame_reviews (route_id TEXT NOT NULL, run_id TEXT NOT NULL, frame_key TEXT NOT NULL,
            segment_index INTEGER NOT NULL, output_index INTEGER NOT NULL, payload TEXT NOT NULL,
            reviewer TEXT NOT NULL, confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
            revision INTEGER NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (route_id, run_id, frame_key));
        CREATE TABLE frame_review_history (history_id INTEGER PRIMARY KEY AUTOINCREMENT, route_id TEXT NOT NULL,
            run_id TEXT NOT NULL, frame_key TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
            reviewer TEXT NOT NULL, confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
            saved_at TEXT NOT NULL, client TEXT NOT NULL);
        CREATE TRIGGER frame_review_history_no_update BEFORE UPDATE ON frame_review_history
            BEGIN SELECT RAISE(ABORT, 'frame_review_history is append-only'); END;
        INSERT INTO frame_reviews VALUES ('00000001--abcdef1234', 'r', '0/0', 0, 0, '{"detections": []}',
            'cristianku', 1, 1, '2026-10-21T00:00:00.000Z');
        INSERT INTO frame_review_history (route_id, run_id, frame_key, revision, payload, reviewer, confirmed,
            saved_at, client) VALUES ('00000001--abcdef1234', 'r', '0/0', 1, '{"detections": []}', 'cristianku',
            1, '2026-10-21T00:00:00.000Z', '10.0.0.1');
    """)
    connection.commit()
    connection.close()
    store = ReviewStore(path)
    assert store.schema_version() == 2
    review = store.review('00000001--abcdef1234', 'r', '0/0')
    assert review['reviewer'] == 'cristianku' and review['reviewer_kind'] == 'human' and review['channel'] == 'web'
    history = store.history('00000001--abcdef1234', 'r', '0/0')
    assert history[0]['reviewer_kind'] == 'human' and history[0]['client'] == '10.0.0.1'
    # Opening it again changes nothing.
    assert ReviewStore(path).schema_version() == 2
    connection = sqlite3.connect(str(path))
    try:
        assert connection.execute('SELECT COUNT(*) FROM schema_version').fetchone()[0] == 2
        with pytest.raises(sqlite3.DatabaseError, match='append-only'):
            connection.execute("UPDATE frame_review_history SET reviewer_kind = 'ai'")
    finally:
        connection.close()


def test_review_site_api_and_core_work_without_the_mcp_sdk(tmp_path):
    """Without the SDK: the page, /api/v1 and ReviewTools work, /mcp answers 501, nnslr mcp exits with a hint."""
    make_report(tmp_path)
    source = Path(__file__).resolve().parents[2] / 'src'
    tests = Path(__file__).resolve().parent
    script = f"""
import sys, json
sys.modules['mcp'] = None
sys.path[:0] = [{str(source)!r}, {str(tests)!r}]
from pathlib import Path
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer
from nnslr_tools.review_mcp import ReviewTools, sdk_available
assert not sdk_available()
server = RunningServer(Path({str(tmp_path)!r}), token=TOKEN, require_token=True)
try:
    auth = {{'Authorization': 'Bearer ' + TOKEN}}
    assert server.request('GET', '/', headers={{'X-Review-Token': TOKEN}})[0] == 200
    assert server.request('GET', '/api/v1/routes', headers=auth)[0] == 200
    status, response, data = server.request('POST', '/mcp', body={{'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}},
                                            headers=auth)
    assert status == 501 and json.loads(data)['error'] == 'mcp_unavailable', (status, data)
    tools = ReviewTools(server.app, client='test')
    frame = tools.next_frame(ROUTE)['frame']
    saved = tools.save_frame_review(ROUTE, frame['frame_key'], run_id=RUN_ID, base_revision=0, reviewer='Codex',
                                    detections=[])
    assert saved['review']['reviewer'] == 'ai/Codex' and saved['review']['channel'] == 'mcp'
finally:
    server.close()
from nnslr_tools import cli
assert cli.main(['mcp', '--data-root', {str(tmp_path)!r}]) == 2
print('ok')
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith('ok')


def test_review_page_shows_ai_reviews_as_final(api):
    """The page shows where an AI confirmation comes from (filter, striped ticks, save line), asks no approval."""
    server, root, rows = api
    assert save(server, ROUTE, '0/0', ai_body())[0] == 200
    status, response, page = server.request('GET', f'/route/{ROUTE}/')
    assert status == 200
    page = page.decode()
    # The page is a Vue app: its markup and logic are in the built script, inline in the page.
    script = asset('js/review.js')
    assert script in page
    assert 'data-filter' in script and 'tlToggle' in script and 'frameReviewerKind' in script and ' (AI' in script
    assert 'Approva' not in page and 'approv' not in page.lower()
    reviews = json_of(server.request('GET', f'/api/route/{ROUTE}/reviews')[2])['reviews']
    assert reviews[0]['reviewer_kind'] == 'ai' and reviews[0]['channel'] == 'rest'
    # A person may still correct it from the page: the correction is a human review.
    body = page_body(rows['0/0'], value_kph=50, base_revision=1)
    status, response, data = server.post_frame(ROUTE, '0/0', body)
    assert status == 200, data
    review = ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '0/0')
    assert review['reviewer_kind'] == 'human' and review['payload']['label_basis'] == 'human_review'
