"""The AI reviewer on the site: Settings (the providers, the reviewer, their models, Test), the AI state, one frame's
suggestion, the "Review with AI" button of the route list, and who may use them: a signed-in browser or the site on
127.0.0.1, never the access token alone. No answer, page or log ever holds a key.

Only fake OpenAI-compatible servers on 127.0.0.1 (fake_llm.FakeLlm) are called; the data roots are temporary.
"""
from __future__ import annotations

import copy
import stat

import pytest

from fake_llm import FakeLlm, answer_json
from nnslr_tools import ai_review, llm_providers, review_auth
from nnslr_tools.pages import _route_row
from review_helpers import (HEIGHT, ROUTE, RUN_ID, TOKEN, WIDTH, RunningServer, json_of, make_report, page_body,
                            sign_in)

KEY = 'sk-site-test-SECRET-0d1e2f3a4b'
SESSION_ONLY = (
    ('/api/settings/ai/providers', {'name': 'other', 'base_url': 'http://127.0.0.1:9/v1', 'model': 'm'}),
    ('/api/settings/ai/providers/fake/delete', {}),
    ('/api/settings/ai/reviewer', {'id': None}),
    ('/api/settings/ai/models', {'id': 'fake'}),
    ('/api/settings/ai/test', {'id': 'fake'}),
    (f'/api/route/{ROUTE}/frame/0/0/ai-suggest', {'run_id': RUN_ID}),
    (f'/api/route/{ROUTE}/frame/0/0/ai-fit', {'run_id': RUN_ID, 'box': [1, 2, 15, 20]}),
    ('/api/panel/run/ai_review', {'route': ROUTE}),
    ('/api/panel/run/measure_ai_reviewer', {}),
)
READ_ONLY = ('/api/settings/ai', '/api/ai/state', '/api/panel/ai-reviewers')
# Frame 0/0 of make_report: one proposal 0/0/0 at [1, 2, 15, 20] on a 32 x 24 frame. The AI corrects it to an 80 and
# adds a give-way sign, in its 0-1000 coordinates.
ANSWER = answer_json(
    boxes=[{'n': 1, 'verdict': 'correct', 'box': [100, 100, 500, 900], 'family': 'maximum_speed', 'value_kph': 80,
            'applicability': 'own_road', 'visibility': 'clear', 'notes': ''}],
    added=[{'box': [600, 100, 900, 500], 'family': 'other_sign', 'sign_type': 'give_way', 'value_kph': None,
            'applicability': 'own_road', 'visibility': 'clear', 'notes': ''}])


@pytest.fixture(autouse=True)
def private_providers(monkeypatch):
    """The providers file of the temporary data root, never the one of NNSLR_LLM_PROVIDERS; a fast password hash."""
    monkeypatch.delenv(llm_providers.PROVIDERS_ENV, raising=False)
    monkeypatch.setattr(review_auth, 'SCRYPT_N', 2 ** 10)


@pytest.fixture
def fake():
    with FakeLlm(ANSWER, require_key=KEY) as llm:
        yield llm


def passed_test(root, provider_id: str) -> None:
    """A Test that passed, as Settings records it: only such a provider may review."""
    llm_providers.record_test(root, provider_id, {'ok': True, 'answered_json': True, 'sees_images': True,
                                                  'message': 'Sees images · 1.0 s'})


def add_provider(root, base_url: str, *, reviewer: bool = True) -> dict:
    provider = llm_providers.save_provider(root, {'name': 'fake', 'base_url': base_url, 'model': 'fake-vl',
                                                  'key': KEY})
    if reviewer:
        passed_test(root, provider['id'])
        llm_providers.set_reviewer(root, provider['id'])
    return provider


@pytest.fixture
def lan(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    yield server, tmp_path
    server.close()


@pytest.fixture
def local(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    yield server, tmp_path
    server.close()


def test_the_token_reads_but_never_changes_or_asks(lan, fake):
    server, root = lan
    add_provider(root, fake.base_url)
    before = llm_providers.providers_path(root).read_bytes()
    token = {'X-Review-Token': TOKEN}
    for path in READ_ONLY:
        status, response, data = server.request('GET', path, headers=token)
        assert status == 200 and KEY.encode() not in data and b'"key"' not in data, path
    assert json_of(server.request('GET', '/api/settings/ai', headers=token)[2])['can_edit'] is False
    state = json_of(server.request('GET', '/api/ai/state', headers=token)[2])
    assert state['available'] is True and state['can_ask'] is False
    assert state['reason'] == 'Sign in on the site to use the AI reviewer.'
    for path, body in SESSION_ONLY:
        status, response, data = server.request('POST', path, body=body, headers=token)
        assert status == 403 and json_of(data)['error'] == 'session_required', (path, data)
    # Nothing changed, nothing was asked, nothing started.
    assert llm_providers.providers_path(root).read_bytes() == before
    assert fake.paths == [] and server.app.panel.jobs() == []
    # Without a session or the token: the sign-in, as for the rest of the site.
    assert server.request('GET', '/api/settings/ai')[0] == 401


def test_a_session_manages_the_providers_and_never_sees_a_key(lan, fake):
    server, root = lan
    session = sign_in(server)
    status, response, data = server.request('GET', '/api/settings/ai', headers=session)
    assert status == 200 and json_of(data) == {'providers': [], 'reviewer': None, 'can_edit': True}
    status, response, data = server.request('POST', '/api/settings/ai/providers', headers=session, body={
        'name': 'Fake', 'base_url': fake.base_url + '/', 'model': 'fake-vl', 'key': KEY})
    assert status == 200 and KEY.encode() not in data, data
    answer = json_of(data)
    assert answer['ok'] is True and answer['can_edit'] is True and answer['reviewer'] is None
    assert answer['provider']['id'] == 'fake' and answer['provider']['has_key'] is True
    assert answer['provider']['base_url'] == fake.base_url and 'key' not in answer['provider']
    # The file: the owner's only, in a folder of its own, under the data root.
    path = llm_providers.providers_path(root)
    assert path == root / 'web/private/llm-providers.json'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    # An edit without a key keeps it; a refused edit never repeats the key it was sent.
    status, response, data = server.request('POST', '/api/settings/ai/providers', headers=session, body={
        'id': 'fake', 'name': 'Fake', 'base_url': fake.base_url, 'model': 'fake-vl-2'})
    assert status == 200 and json_of(data)['provider']['has_key'] is True
    assert llm_providers.provider(root, 'fake')['key'] == KEY
    status, response, data = server.request('POST', '/api/settings/ai/providers', headers=session, body={
        'id': 'fake', 'name': 'Fake', 'base_url': 'ftp://' + KEY, 'model': 'fake-vl', 'key': KEY})
    assert status == 400 and json_of(data)['error'] == 'invalid_provider' and KEY.encode() not in data

    # Never tested (a new model clears the Test): it cannot review yet. Test it, then it can.
    status, response, data = server.request('POST', '/api/settings/ai/reviewer', headers=session, body={'id': 'fake'})
    assert status == 409 and json_of(data)['error'] == 'ai_not_tested'
    assert json_of(data)['message'].startswith('Test Fake in Settings first')
    status, response, data = server.request('POST', '/api/settings/ai/test', headers=session, body={'id': 'fake'})
    assert status == 200 and json_of(data)['result']['ok'] is True, data
    status, response, data = server.request('POST', '/api/settings/ai/reviewer', headers=session, body={'id': 'fake'})
    assert status == 200 and json_of(data)['reviewer'] == 'fake'
    assert json_of(data)['providers'][0]['is_reviewer'] is True
    status, response, data = server.request('POST', '/api/settings/ai/reviewer', headers=session, body={'id': 3})
    assert status == 400
    state = json_of(server.request('GET', '/api/ai/state', headers=session)[2])
    assert state == {'available': True, 'can_ask': True, 'reason': '',
                     'provider': {'id': 'fake', 'name': 'Fake', 'model': 'fake-vl-2',
                                  'reviewer_name': 'ai/fake/fake-vl-2'}}
    for read in READ_ONLY:
        assert KEY.encode() not in server.request('GET', read, headers=session)[2]

    # "" removes the key; deleting the reviewer leaves no reviewer.
    status, response, data = server.request('POST', '/api/settings/ai/providers', headers=session, body={
        'id': 'fake', 'name': 'Fake', 'base_url': fake.base_url, 'model': 'fake-vl-2', 'key': ''})
    assert status == 200 and json_of(data)['provider']['has_key'] is False
    status, response, data = server.request('POST', '/api/settings/ai/providers/fake/delete', headers=session, body={})
    assert status == 200 and json_of(data) == {'ok': True, 'providers': [], 'reviewer': None, 'can_edit': True}
    status, response, data = server.request('POST', '/api/settings/ai/providers/fake/delete', headers=session, body={})
    assert status == 404
    state = json_of(server.request('GET', '/api/ai/state', headers=session)[2])
    assert state['available'] is False and state['reason'] == 'Choose an AI reviewer in Settings.'


def test_ai_posts_keep_the_protections_of_a_save(local, fake):
    server, root = local
    add_provider(root, fake.base_url)
    for path, body in SESSION_ONLY[:6]:
        status, response, data = server.request('POST', path, body=body, headers={'Origin': 'http://evil.example'})
        assert status == 403 and json_of(data)['error'] == 'cross_origin', path
        status, response, data = server.request('POST', path, body=b'{}', headers={'Content-Type': 'text/plain'})
        assert status == 415, path
        assert server.request('GET', path)[0] == 405, path
    status, response, data = server.request('POST', '/api/settings/ai/reviewer', body=[1])
    assert status == 400 and json_of(data)['error'] == 'invalid_json'
    assert server.request('POST', '/api/settings/ai/providers/Not_An_Id/delete', body={})[0] == 404
    assert fake.paths == []


def test_a_suggestion_has_pixel_boxes_and_saves_nothing(local, fake):
    server, root = local
    add_provider(root, fake.base_url)
    status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-suggest',
                                            body={'run_id': RUN_ID})
    assert status == 200, data
    assert KEY.encode() not in data
    suggestion = json_of(data)
    assert suggestion['ok'] is True and suggestion['route_id'] == ROUTE and suggestion['frame_key'] == '0/0'
    assert suggestion['run_id'] == RUN_ID and suggestion['reviewer'] == 'ai/fake/fake-vl'
    assert suggestion['provider'] == {'id': 'fake', 'name': 'fake', 'model': 'fake-vl'}
    assert suggestion['complete'] is True and suggestion['no_sign'] is False and suggestion['deleted'] == []
    corrected, added = suggestion['detections']
    # The corrected proposal keeps its id; the boxes are pixels of the 32 x 24 frame, inside it.
    assert corrected['detection_id'] == '0/0/0' and corrected['ai_action'] == 'correct'
    assert corrected['bbox_xyxy'] == pytest.approx([3.2, 2.4, 16.0, 21.6])
    assert corrected['family'] == 'maximum_speed' and corrected['value_kph'] == 80
    assert added['ai_action'] == 'add' and added['detection_id'].startswith('ai-')
    assert added['bbox_xyxy'] == pytest.approx([19.2, 2.4, 28.8, 12.0])
    assert added['family'] == 'other_sign' and added['sign_type'] == 'give_way'
    for box in (corrected['bbox_xyxy'], added['bbox_xyxy']):
        assert 0 <= box[0] < box[2] <= WIDTH and 0 <= box[1] < box[3] <= HEIGHT
    # The key went to the provider, in the header only; the model saw the proposal, in 0-1000.
    assert fake.auth_headers == ['Bearer ' + KEY]
    assert KEY not in repr(fake.requests)
    # Nothing saved.
    assert server.app.store.confirmed_count(ROUTE, RUN_ID) == 0

    # Applied on the page, then confirmed by the person: a human review, as any save of the page.
    row = server.app.report(ROUTE)[2]['0/0']
    body = page_body(row, base_revision=0)
    body['detections'] = [{key: value for key, value in item.items() if key != 'ai_action'}
                          for item in copy.deepcopy(suggestion['detections'])]
    status, response, data = server.post_frame(ROUTE, '0/0', body)
    assert status == 200, data
    review = json_of(data)['review']
    assert review['reviewer'] == 'Cristian' and review.get('reviewer_kind', 'human') == 'human'


def test_fit_with_ai_sends_only_the_crop_around_the_box_and_gives_it_back_tight(local):
    server, root = local
    with FakeLlm('{"found": true, "box": [250, 250, 750, 750]}', require_key=KEY) as llm:
        add_provider(root, llm.base_url)
        status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-fit',
                                                body={'run_id': RUN_ID, 'box': [1, 2, 15, 20]})
        assert status == 200, data
        fitted = json_of(data)
        # The 32 x 24 frame: the crop is its 24 x 24 square on the left; the answer back in pixels of the frame.
        assert fitted['box'] == pytest.approx([6.0, 6.0, 18.0, 18.0])
        assert fitted['message'].startswith('Box fitted by fake-vl')
        content = llm.requests[0]['messages'][0]['content']
        # One picture, the crop enlarged (a PNG, not the frame's JPEG), and the rough box in 0-1000 of the crop.
        assert content[0]['image_url']['url'].startswith('data:image/png;base64,')
        assert '42 83 625 833' in content[1]['text']
        assert llm.requests[0]['max_tokens'] == ai_review.FIT_TOKENS
    # Nothing there: no box, and nothing saved either way.
    with FakeLlm('{"found": false}', require_key=KEY) as llm:
        add_provider(root, llm.base_url)
        status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-fit',
                                                body={'run_id': RUN_ID, 'box': [1, 2, 15, 20]})
        assert status == 200 and json_of(data)['box'] is None
    status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-fit',
                                            body={'run_id': RUN_ID, 'box': [40, 40, 41, 41]})
    assert status == 400 and json_of(data)['error'] == 'invalid_box'
    assert server.app.store.confirmed_count(ROUTE, RUN_ID) == 0


def test_the_crop_of_a_fit_is_a_square_around_the_box_inside_the_frame():
    assert ai_review.fit_crop([600, 300, 620, 330], 1344, 760) == [565, 270, 655, 360]
    assert ai_review.fit_crop([0, 0, 40, 30], 1344, 760) == [0, 0, 120, 120]
    assert ai_review.fit_crop([1300, 700, 1344, 760], 1344, 760) == [1164, 580, 1344, 760]


def test_suggestion_refusals(local, fake):
    server, root = local
    path = f'/api/route/{ROUTE}/frame/0/0/ai-suggest'
    # No reviewer chosen.
    status, response, data = server.request('POST', path, body={'run_id': RUN_ID})
    assert status == 409 and json_of(data)['error'] == 'no_ai_reviewer'
    add_provider(root, fake.base_url)
    status, response, data = server.request('POST', path, body={'run_id': '20200101T000000Z-00000000'})
    assert status == 409 and json_of(data)['error'] == 'stale_report'
    status, response, data = server.request('POST', path, body={'run_id': RUN_ID, 'detections': 'all'})
    assert status == 400 and json_of(data)['error'] == 'invalid_detections'
    status, response, data = server.request('POST', path, body={'run_id': 5})
    assert status == 400
    assert server.request('POST', f'/api/route/{ROUTE}/frame/0/99/ai-suggest', body={'run_id': RUN_ID})[0] == 404
    assert fake.requests == []
    # Two suggestions are already waiting for the model: the third is refused at once.
    assert server.app.ai_slots.acquire(blocking=False) and server.app.ai_slots.acquire(blocking=False)
    try:
        status, response, data = server.request('POST', path, body={'run_id': RUN_ID})
        assert status == 429 and json_of(data)['error'] == 'ai_busy'
        status, response, data = server.request('POST', '/api/settings/ai/test', body={'id': 'fake'})
        assert status == 429 and json_of(data)['error'] == 'ai_busy'
    finally:
        server.app.ai_slots.release()
        server.app.ai_slots.release()
    assert fake.requests == []
    # The slots are free again.
    assert server.request('POST', path, body={'run_id': RUN_ID})[0] == 200


def test_a_provider_error_never_repeats_the_key(local):
    server, root = local
    with FakeLlm(status=401, error_message='Incorrect API key provided: ' + KEY) as broken:
        add_provider(root, broken.base_url)
        status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-suggest',
                                                body={'run_id': RUN_ID})
        assert status == 502 and json_of(data)['error'] == 'ai_http_error', data
        assert KEY.encode() not in data and b'HTTP 401' in data
        status, response, data = server.request('POST', '/api/settings/ai/test', body={'id': 'fake'})
        assert status == 200 and KEY.encode() not in data
        assert json_of(data)['result']['ok'] is False
    # Its Test failed: it is not asked any more until it passes again.
    status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-suggest',
                                            body={'run_id': RUN_ID})
    assert status == 409 and json_of(data)['error'] == 'ai_not_tested' and KEY.encode() not in data
    passed_test(root, 'fake')
    # The server is gone: unreachable, still without the key.
    status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-suggest',
                                            body={'run_id': RUN_ID})
    assert status == 502 and json_of(data)['error'] == 'ai_unreachable' and KEY.encode() not in data
    status, response, data = server.request('POST', '/api/settings/ai/models', body={'id': 'fake'})
    assert status == 502 and KEY.encode() not in data


def test_models_and_test(local, fake):
    server, root = local
    add_provider(root, fake.base_url, reviewer=False)
    status, response, data = server.request('POST', '/api/settings/ai/models', body={'id': 'fake'})
    assert status == 200 and json_of(data) == {'ok': True, 'models': [{'id': 'fake-vl', 'sees_images': True}]}
    assert fake.auth_headers == ['Bearer ' + KEY]
    status, response, data = server.request('POST', '/api/settings/ai/test', body={})
    assert status == 400 and json_of(data)['error'] == 'invalid_provider'
    status, response, data = server.request('POST', '/api/settings/ai/test', body={'id': 'missing'})
    assert status == 404
    # A model that takes text only: Test says so without asking it anything.
    with FakeLlm(models=('fake-vl',), text_only=True, require_key=KEY) as text_only:
        llm_providers.save_provider(root, {'id': 'fake', 'name': 'fake', 'base_url': text_only.base_url,
                                           'model': 'fake-vl'})
        status, response, data = server.request('POST', '/api/settings/ai/test', body={'id': 'fake'})
        assert status == 200 and KEY.encode() not in data
        answer = json_of(data)
        assert answer['ok'] is True and answer['result']['ok'] is False
        assert answer['result']['sees_images'] is False and 'text only' in answer['result']['message']
        assert answer['provider']['id'] == 'fake' and answer['provider']['last_test'] == answer['result']
        assert text_only.requests == []


def test_a_provider_that_did_not_pass_its_test_never_reviews(local):
    server, root = local
    with FakeLlm(text_only=True, require_key=KEY) as blind:
        add_provider(root, blind.base_url, reviewer=False)
        status, response, data = server.request('POST', '/api/settings/ai/test', body={'id': 'fake'})
        assert status == 200 and json_of(data)['result']['ok'] is False
        status, response, data = server.request('POST', '/api/settings/ai/reviewer', body={'id': 'fake'})
        assert status == 409 and json_of(data)['error'] == 'ai_not_tested'
        assert json_of(data)['message'] == 'fake did not pass its last Test. Fix it, then test it again in Settings.'
        assert llm_providers.public_settings(root)['reviewer'] is None
        # Chosen when it passed, then failing again: Ask AI is off, with the reason, and the server refuses it.
        passed_test(root, 'fake')
        assert server.request('POST', '/api/settings/ai/reviewer', body={'id': 'fake'})[0] == 200
        llm_providers.record_test(root, 'fake', {'ok': False, 'message': 'It cannot see the frames.'})
        state = json_of(server.request('GET', '/api/ai/state')[2])
        assert state['available'] is False and state['can_ask'] is True and state['provider']['id'] == 'fake'
        assert state['reason'] == 'fake did not pass its last Test. Fix it, then test it again in Settings.'
        status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/0/ai-suggest',
                                                body={'run_id': RUN_ID})
        assert status == 409 and json_of(data)['error'] == 'ai_not_tested'
        assert blind.requests == []
        # "Don't use AI" is always possible.
        status, response, data = server.request('POST', '/api/settings/ai/reviewer', body={'id': None})
        assert status == 200 and json_of(data)['reviewer'] is None


def test_the_review_page_reads_the_last_trial(lan, fake):
    server, root = lan
    add_provider(root, fake.base_url)
    token = {'X-Review-Token': TOKEN}
    status, response, data = server.request('GET', f'/api/route/{ROUTE}/ai-trial', headers=token)
    assert status == 200 and json_of(data) == {'ok': True, 'trial': None}
    ai_review.review_route(server.app, ROUTE, llm_providers.provider(root, 'fake'), save=False, log=lambda text: None)
    # The access token reads it too: it holds no key and saves nothing.
    status, response, data = server.request('GET', f'/api/route/{ROUTE}/ai-trial', headers=token)
    assert status == 200 and KEY.encode() not in data
    trial = json_of(data)['trial']
    assert trial['run_id'] == RUN_ID and trial['provider'] == {'id': 'fake', 'name': 'fake', 'model': 'fake-vl'}
    assert [frame['frame_key'] for frame in trial['frames']] == ['0/0', '0/1', '0/2']
    assert trial['frames'][0]['detections'][0]['ai_action'] == 'correct'
    assert server.app.store.confirmed_count(ROUTE, RUN_ID) == 0
    assert server.request('GET', '/api/route/00000009--abcdef1234/ai-trial', headers=token)[0] == 404
    assert server.request('GET', f'/api/route/{ROUTE}/ai-trial')[0] == 401


def test_the_route_list_offers_the_ai_on_routes_not_done(local, fake):
    server, root = local
    # The route list is retired (9 Nov 2026): "Review with AI" is on the Signs side of Speed review, whose page carries
    # the style of its sheet (the sheet says itself when no AI reviewer is chosen).
    status, response, _data = server.request('GET', '/routes')
    assert status == 303 and response.getheader('Location') == '/speed-review'
    add_provider(root, fake.base_url)
    page = server.request('GET', '/speed-review')[2].decode()
    assert '.ai-choices{' in page and '.ai-sheet' in page and '__NNSLR_AI_SHEET_CSS__' not in page
    # The pages of Sign review (unserved but for a route Speed review cannot show) still render the button.
    summary = {'route_id': ROUTE, 'url': f'/route/{ROUTE}/', 'frame_count': 1200, 'confirmed_count': 200,
               'frames_with_proposals': 900, 'drive': {'title': 'Herznach → <Zürich>', 'when': 'Fri 23 Oct 2026 · 08:01',
                                                       'minutes': 20}}
    row = _route_row(summary)
    assert row.startswith('<li><a class="card-row doing"') and row.endswith('Review with AI</button></li>')
    assert 'data-ai-name="Herznach → &lt;Zürich&gt; · Fri 23 Oct 2026 · 08:01"' in row and 'data-ai-left="1000"' in row
    # A route every frame of which is confirmed has nothing left for the AI.
    assert 'ai-route' not in _route_row({**summary, 'confirmed_count': 1200})
