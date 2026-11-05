"""MCP tools: the same operations as /api/v1, real image content, AI provenance, stdio and HTTP transports.

The ReviewTools tests need no SDK. The others use the official MCP SDK client
and are skipped when the optional [mcp] dependencies are not installed.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest

from nnslr_tools.annotations import load_jsonl
from nnslr_tools.review_db import ReviewRejected, ReviewStore, db_path, import_from_db
from nnslr_tools.review_mcp import MAX_CONTEXT_IMAGES, ReviewTools
from nnslr_tools.review_service import ReviewService
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

KEYS = [(0, 0), (0, 1), (0, 4), (2, 0), (2, 3), (2, 7), (5, 2)]
TOOL_NAMES = {'list_routes', 'list_frames', 'next_frame', 'get_frame', 'get_frame_image', 'get_frame_context',
              'get_review_schema', 'save_frame_review', 'get_review_progress', 'get_review_history',
              'apply_previous_frame'}
SRC = Path(__file__).resolve().parents[2] / 'src'


@pytest.fixture
def service(tmp_path):
    make_report(tmp_path, keys=KEYS)
    return ReviewService(tmp_path, ReviewStore(db_path(tmp_path)))


def test_review_flow_with_the_plain_tools(service, tmp_path):
    tools = ReviewTools(service, client='test')
    routes = tools.list_routes()['routes']
    assert [r['route_id'] for r in routes] == [ROUTE] and routes[0]['frame_count'] == 7
    # The drive, as the site names the route (nothing known of it in the test).
    assert routes[0]['drive'] == {'when': None, 'minutes': None, 'start': None, 'from': None, 'to': None, 'title': None,
                                 'full_definition': False}
    following = tools.next_frame(ROUTE)
    frame = following['frame']
    assert frame['frame_key'] == '0/0' and frame['save'] == {'run_id': RUN_ID, 'base_revision': 0}
    info, image = tools.get_frame_image(ROUTE, frame['frame_key'])
    assert image['data'] == b'jpeg preview' and info['mime_type'] == 'image/jpeg' and info['bytes'] == 12
    info, image = tools.get_frame_image(ROUTE, frame['frame_key'], variant='original')
    assert image['data'][:8] == b'\x89PNG\r\n\x1a\n' and info['width'] == 32
    saved = tools.save_frame_review(ROUTE, frame['frame_key'], run_id=frame['run_id'],
                                    base_revision=frame['revision'], reviewer='Claude Code',
                                    detections=[{'detection_id': '0/0/0', 'applicability': 'own_road'}])
    assert saved['review']['reviewer'] == 'ai/Claude Code' and saved['review']['channel'] == 'mcp'
    assert saved['next_unreviewed_frame_key'] == '0/1'
    assert tools.next_frame(ROUTE)['frame']['frame_key'] == '0/1'
    progress = tools.get_review_progress(ROUTE)
    assert progress['confirmed_by_ai'] == 1 and progress['missing_count'] == 6
    history = tools.get_review_history(ROUTE, '0/0')['history']
    assert [(h['reviewer'], h['reviewer_kind'], h['channel']) for h in history] == [('ai/Claude Code', 'ai', 'mcp')]
    payload = service.store.review(ROUTE, RUN_ID, '0/0')['payload']
    assert payload['label_basis'] == 'ai_review' and service.store.history(ROUTE, RUN_ID, '0/0')[0]['client'] == 'test'
    with pytest.raises(ReviewRejected) as caught:
        tools.save_frame_review(ROUTE, '0/0', run_id=RUN_ID, base_revision=0, reviewer='Claude', detections=[])
    assert caught.value.status == 409 and caught.value.reason == 'revision_conflict'
    schema = tools.get_review_schema()
    assert 'override_human_review' in schema['save_body'] and schema['detection']['family'][0] == 'maximum_speed'


def test_mcp_saves_are_always_ai_and_never_touch_human_reviews(service, tmp_path):
    tools = ReviewTools(service, client='test')
    rows = service.report(ROUTE)[2]
    from nnslr_tools.review_db import save_frame_review
    save_frame_review(service.store, tmp_path, route_id=ROUTE, run_id=RUN_ID, report_row=rows['0/1'],
                      body=page_body(rows['0/1'], reviewer='cristianku'), client='test')
    with pytest.raises(ReviewRejected) as caught:
        tools.save_frame_review(ROUTE, '0/1', run_id=RUN_ID, base_revision=1, reviewer='cristianku', detections=[])
    assert caught.value.reason == 'human_review_exists'
    review = service.store.review(ROUTE, RUN_ID, '0/1')
    assert review['reviewer'] == 'cristianku' and review['reviewer_kind'] == 'human' and review['revision'] == 1
    saved = tools.save_frame_review(ROUTE, '0/4', run_id=RUN_ID, base_revision=0, reviewer='cristianku',
                                    detections=[])
    # A person's name given by an AI still reads as an AI review.
    assert saved['review']['reviewer'] == 'ai/cristianku' and saved['review']['reviewer_kind'] == 'ai'
    read_only = ReviewTools(service, client='test', read_only=True)
    with pytest.raises(ReviewRejected) as caught:
        read_only.save_frame_review(ROUTE, '2/0', run_id=RUN_ID, base_revision=0, reviewer='Claude', detections=[])
    assert caught.value.reason == 'read_only'
    assert service.store.review(ROUTE, RUN_ID, '2/0') is None


def test_context_crosses_segments_and_attaches_few_images(service):
    tools = ReviewTools(service, client='test')
    context, images = tools.get_frame_context(ROUTE, '2/0', before=3, after=3)
    assert [(f['frame_key'], f['offset']) for f in context['frames']] == [
        ('0/0', -3), ('0/1', -2), ('0/4', -1), ('2/0', 0), ('2/3', 1), ('2/7', 2), ('5/2', 3)]
    assert images == []
    context, images = tools.get_frame_context(ROUTE, '2/0', before=3, after=3, include_images=True)
    assert len(images) == MAX_CONTEXT_IMAGES
    assert [item['frame_key'] for item in images] == ['0/4', '2/3', '0/1', '2/7']
    assert context['images_included'] == ['0/4', '2/3', '0/1', '2/7']
    with pytest.raises(ReviewRejected):
        tools.get_frame_context(ROUTE, '2/1')


# ---------- with the official MCP SDK ----------

def sdk():
    return pytest.importorskip('mcp.server.mcpserver', reason='optional [mcp] dependencies not installed')


def error_of(result) -> dict:
    """The JSON error of a failed tool call: the SDK writes "Error executing tool <name>: {...}"."""
    assert result.is_error
    text = result.content[0].text
    return json.loads(text[text.index('{'):])


def run(coroutine_function):
    import anyio
    return anyio.run(coroutine_function)


def test_tools_have_schemas_and_return_real_images(service):
    sdk()
    from mcp.client.client import Client
    from nnslr_tools.review_mcp import build_server
    server = build_server(service, client='test')

    async def scenario():
        async with Client(server) as client:
            listed = (await client.list_tools()).tools
            names = set()
            for tool in listed:
                names.add(tool.name)
                assert tool.description and tool.input_schema['type'] == 'object'
            assert names == TOOL_NAMES
            by_name = {tool.name: tool for tool in listed}
            assert by_name['list_routes'].output_schema['properties']['routes']['type'] == 'array'
            assert by_name['save_frame_review'].annotations.read_only_hint is False
            assert by_name['get_frame'].annotations.read_only_hint is True
            assert set(by_name['save_frame_review'].input_schema['required']) == {
                'route_id', 'frame_key', 'run_id', 'base_revision', 'reviewer', 'detections'}
            following = await client.call_tool('next_frame', {'route_id': ROUTE})
            frame = following.structured_content['frame']
            assert frame['frame_key'] == '0/0'
            result = await client.call_tool('get_frame_image', {'route_id': ROUTE, 'frame_key': '0/0'})
            assert not result.is_error
            kinds = [item.type for item in result.content]
            assert kinds == ['text', 'image']
            assert result.content[1].mime_type == 'image/jpeg'
            assert base64.b64decode(result.content[1].data) == b'jpeg preview'
            result = await client.call_tool('get_frame_image', {'route_id': ROUTE, 'frame_key': '0/0',
                                                                'variant': 'original'})
            assert base64.b64decode(result.content[1].data)[:4] == b'\x89PNG'
            context = await client.call_tool('get_frame_context', {'route_id': ROUTE, 'frame_key': '2/0',
                                                                   'include_images': True})
            images = [item for item in context.content if item.type == 'image']
            assert len(images) == 4 and context.structured_content['frame_key'] == '2/0'
            saved = await client.call_tool('save_frame_review', {
                'route_id': ROUTE, 'frame_key': '0/0', 'run_id': frame['run_id'],
                'base_revision': frame['revision'], 'reviewer': 'Claude',
                'detections': [{'detection_id': '0/0/0', 'value_kph': 50, 'applicability': 'own_road'}]})
            assert not saved.is_error, saved.content
            assert saved.structured_content['review']['reviewer'] == 'ai/Claude'
            assert saved.structured_content['review']['detections'][0]['value_kph'] == 50
            again = await client.call_tool('save_frame_review', {
                'route_id': ROUTE, 'frame_key': '0/0', 'run_id': frame['run_id'], 'base_revision': 0,
                'reviewer': 'Claude', 'detections': []})
            assert again.is_error
            assert error_of(again)['error'] == 'revision_conflict'
            bad = await client.call_tool('save_frame_review', {
                'route_id': ROUTE, 'frame_key': '0/1', 'run_id': frame['run_id'], 'base_revision': 0,
                'reviewer': 'Claude', 'detections': [{'family': 'advertisement'}]})
            assert bad.is_error
            missing = await client.call_tool('get_frame', {'route_id': ROUTE, 'frame_key': '9/9'})
            assert missing.is_error and error_of(missing)['status'] == 404
            history = await client.call_tool('get_review_history', {'route_id': ROUTE, 'frame_key': '0/0'})
            assert history.structured_content['history'][0]['channel'] == 'mcp'
            progress = await client.call_tool('get_review_progress', {'route_id': ROUTE})
            assert progress.structured_content['confirmed_by_ai'] == 1

    run(scenario)
    assert service.store.review(ROUTE, RUN_ID, '0/1') is None


def test_read_only_server_offers_no_save_tool(service):
    sdk()
    from mcp.client.client import Client
    from nnslr_tools.review_mcp import build_server

    async def scenario():
        async with Client(build_server(service, client='test', read_only=True)) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == TOOL_NAMES - {'save_frame_review'}

    run(scenario)


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    make_report(tmp_path, keys=KEYS)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    yield server, tmp_path
    server.close()


def test_mcp_over_http_needs_the_token_and_the_same_origin(http_server):
    sdk()
    server, root = http_server
    ping = {'jsonrpc': '2.0', 'id': 1, 'method': 'ping', 'params': {}}
    accept = {'Accept': 'application/json, text/event-stream'}
    for headers in [{}, {'Authorization': 'Bearer wrong-token-000000000000'},
                    {'Cookie': f'{server.app.cookie_name}={TOKEN}'}]:
        headers.update(accept)
        status, response, data = server.request('POST', '/mcp', body=ping, headers=headers)
        assert status == 401 and response.getheader('WWW-Authenticate'), headers
    status, response, data = server.request('POST', f'/mcp?token={TOKEN}', body=ping, headers=accept)
    assert status == 400 and json_of(data)['error'] == 'token_in_url'
    auth = dict(accept, Authorization='Bearer ' + TOKEN)
    status, response, data = server.request('POST', '/mcp', body=ping, headers=dict(auth, Origin='http://evil.example'))
    assert status == 403 and json_of(data)['error'] == 'cross_origin'
    status, response, data = server.request('POST', '/mcp', body=b'x' * (512 * 1024 + 1), headers=auth)
    assert status == 413
    # Stateless server: no server stream and no session to end.
    for method in ['GET', 'DELETE']:
        status, response, data = server.request(method, '/mcp', headers=auth)
        assert status == 405 and response.getheader('Allow') == 'POST', method


def test_claude_style_client_over_streamable_http(http_server):
    sdk()
    import httpx2
    from mcp.client.client import Client
    from mcp.client.streamable_http import streamable_http_client
    server, root = http_server
    url = f'http://127.0.0.1:{server.port}/mcp'

    async def scenario():
        http = httpx2.AsyncClient(headers={'Authorization': 'Bearer ' + TOKEN})
        async with Client(streamable_http_client(url, http_client=http)) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == TOOL_NAMES
            routes = await client.call_tool('list_routes', {})
            assert routes.structured_content['routes'][0]['route_id'] == ROUTE
            following = await client.call_tool('next_frame', {'route_id': ROUTE, 'after': '2/7'})
            frame = following.structured_content['frame']
            assert frame['frame_key'] == '5/2'
            image = await client.call_tool('get_frame_image', {'route_id': ROUTE, 'frame_key': '5/2'})
            assert image.content[1].type == 'image'
            assert base64.b64decode(image.content[1].data) == b'jpeg preview'
            saved = await client.call_tool('save_frame_review', {
                'route_id': ROUTE, 'frame_key': '5/2', 'run_id': frame['run_id'], 'base_revision': 0,
                'reviewer': 'Claude Code', 'detections': []})
            assert not saved.is_error, saved.content
        await http.aclose()

    run(scenario)
    review = ReviewStore(db_path(root)).review(ROUTE, RUN_ID, '5/2')
    assert review['reviewer'] == 'ai/Claude Code' and review['channel'] == 'mcp'
    assert ReviewStore(db_path(root)).history(ROUTE, RUN_ID, '5/2')[0]['client'] == 'mcp-http'
    # The page sees the same database.
    reviews = json_of(server.request('GET', f'/api/route/{ROUTE}/reviews', headers={'X-Review-Token': TOKEN})[2])
    assert reviews['confirmed_count'] == 1 and reviews['reviews'][0]['reviewer_kind'] == 'ai'
    # The confirmation made over MCP is final: the import takes it as it is, as an AI review.
    report = import_from_db(root, root / 'objects.jsonl')
    assert report['imported_frame_count'] == 1 and report['imported_ai_frame_count'] == 1
    for record in load_jsonl(root / 'objects.jsonl'):
        assert record['reviewer'] == 'ai/Claude Code' and record['provenance']['review_channel'] == 'mcp'


def test_read_only_environment_hides_the_save_tool_over_http(tmp_path, monkeypatch):
    sdk()
    import httpx2
    from mcp.client.client import Client
    from mcp.client.streamable_http import streamable_http_client
    monkeypatch.setenv('NNSLR_MCP_READ_ONLY', '1')
    make_report(tmp_path, keys=KEYS)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)

    async def scenario():
        http = httpx2.AsyncClient(headers={'X-Review-Token': TOKEN})
        async with Client(streamable_http_client(f'http://127.0.0.1:{server.port}/mcp', http_client=http)) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert 'save_frame_review' not in names and 'get_frame_image' in names
        await http.aclose()

    try:
        run(scenario)
    finally:
        server.close()


def test_stdio_transport_of_nnslr_mcp(tmp_path):
    sdk()
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters
    make_report(tmp_path, keys=KEYS)
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(SRC) + os.pathsep + environment.get('PYTHONPATH', '')
    parameters = StdioServerParameters(command=sys.executable,
                                       args=['-m', 'nnslr_tools.cli', 'mcp', '--data-root', str(tmp_path)],
                                       env=environment)

    async def scenario():
        async with Client(parameters) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == TOOL_NAMES
            image = await client.call_tool('get_frame_image', {'route_id': ROUTE, 'frame_key': '0/4',
                                                               'variant': 'original'})
            assert image.content[1].type == 'image' and image.content[1].mime_type == 'image/png'
            saved = await client.call_tool('save_frame_review', {
                'route_id': ROUTE, 'frame_key': '0/4', 'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Codex',
                'detections': [{'detection_id': '0/4/0', 'applicability': 'other_road'}]})
            assert not saved.is_error, saved.content

    run(scenario)
    review = ReviewStore(db_path(tmp_path)).review(ROUTE, RUN_ID, '0/4')
    assert review['reviewer'] == 'ai/Codex' and review['reviewer_kind'] == 'ai'
    assert ReviewStore(db_path(tmp_path)).history(ROUTE, RUN_ID, '0/4')[0]['client'] == 'mcp-stdio'
