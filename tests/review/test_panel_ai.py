"""The AI actions of the panel: "Review a route with AI" and "Measure an AI reviewer", their options, their state, the
commands they run (never a key), who may start them, and their progress.

The helpers are fakes in a temporary checkout that print their arguments: no AI is called, nothing is reviewed.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

import nnslr_tools.panel as panel_module
from nnslr_tools import llm_providers, review_auth
from nnslr_tools.job_progress import progress, with_step_progress
from nnslr_tools.panel import Panel, PanelError
from review_helpers import ROUTE, TOKEN, RunningServer, json_of, make_report, sign_in

REPO = Path(__file__).resolve().parents[2]
KEY = 'sk-panel-test-SECRET-4f2a9c'
FAKE = '#!/usr/bin/env bash\necho "{name} $*"\nsleep "${{FAKE_SLEEP:-0}}"\n'


@pytest.fixture(autouse=True)
def only_this_checkout(monkeypatch):
    """A job really running on this machine must not make these tests busy; the providers file is the temporary
    data root's, never the one of NNSLR_LLM_PROVIDERS."""
    monkeypatch.setattr(panel_module, 'OWN_GROUPS', ('fetch', 'prepare', 'train', 'update', 'ai', 'other'))
    monkeypatch.delenv(llm_providers.PROVIDERS_ENV, raising=False)
    monkeypatch.setattr(review_auth, 'SCRYPT_N', 2 ** 10)


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'helpers').mkdir(parents=True)
    shutil.copy(REPO / 'helpers/route_status.py', repo / 'helpers/route_status.py')
    for name in ('1_prepare.sh', '3_train.sh', 'update.sh', 'ai_review.sh', 'measure_ai_reviewer.sh'):
        script = repo / 'helpers' / name
        script.write_text(FAKE.format(name=name))
        script.chmod(0o755)
    return repo


@pytest.fixture
def root(tmp_path):
    data = tmp_path / 'data'
    make_report(data)
    return data


def passed_test(root: Path, provider_id: str, ok: bool = True) -> None:
    """The last Test of a provider, as Settings records it: only one that passed may review."""
    llm_providers.record_test(root, provider_id, {'ok': ok, 'message': 'Sees images' if ok else 'Cannot see'})


def add_provider(root: Path, name: str = 'fake', *, reviewer: bool = True) -> dict:
    provider = llm_providers.save_provider(root, {'name': name, 'base_url': 'http://127.0.0.1:9/v1',
                                                   'model': 'fake-vl', 'key': KEY})
    if reviewer:
        passed_test(root, provider['id'])
        llm_providers.set_reviewer(root, provider['id'])
    return provider


def finished(panel, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(j for j in panel.jobs() if j['id'] == job_id)
        if job['state'] != 'running':
            return job
        time.sleep(0.05)
    raise AssertionError(f'{job_id} still running')


def next_second() -> None:
    """A job's id has the second it started: two jobs in the same second would share it."""
    time.sleep(1.01 - time.time() % 1)


def log_of(panel, job_id) -> str:
    return (panel.logs / (job_id + '.log')).read_text()


def refused(panel, action, options, status=400, reason='bad_option'):
    with pytest.raises(PanelError) as caught:
        panel.start(action, options)
    assert (caught.value.status, caught.value.reason) == (status, reason), caught.value.message
    return caught.value


def test_review_runs_the_helper_with_the_route_and_provider_never_the_key(root, checkout, monkeypatch):
    add_provider(root)
    calls = []
    real = subprocess.Popen

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(panel_module.subprocess, 'Popen', spy)
    panel = Panel(root, repo=checkout, actions_enabled=True)
    job = panel.start('ai_review', {'route': ROUTE, 'save': 'yes', 'limit': '20'})
    assert job['label'] == 'Review a route with AI' and job['action'] == 'ai_review'
    assert finished(panel, job['id'])['state'] == 'ok'
    log = log_of(panel, job['id'])
    assert log.splitlines()[0] == f'== helpers/ai_review.sh {ROUTE} --provider=fake --save --limit=20'
    assert f'ai_review.sh {ROUTE} --provider=fake --save --limit=20' in log
    # The key is read by the job from the providers file: never an argument, an environment value or a log line.
    (args, kwargs), = calls
    assert KEY not in ' '.join(args[0]) and all(KEY not in value for value in kwargs['env'].values())
    assert KEY not in log

    # Without a provider: the reviewer of Settings. Without save: a trial (no --save), the whole route.
    next_second()
    job = panel.start('ai_review', {'route': ROUTE})
    finished(panel, job['id'])
    assert log_of(panel, job['id']).splitlines()[0] == f'== helpers/ai_review.sh {ROUTE} --provider=fake'
    # Only the frames at a traffic light of the map; "signs" is the default and adds nothing.
    next_second()
    job = panel.start('ai_review', {'route': ROUTE, 'scope': 'lights'})
    finished(panel, job['id'])
    assert log_of(panel, job['id']).splitlines()[0] == f'== helpers/ai_review.sh {ROUTE} --provider=fake --scope=lights'


def test_measure_runs_its_helper(root, checkout):
    add_provider(root, reviewer=False)
    other = add_provider(root, 'Other one', reviewer=False)
    panel = Panel(root, repo=checkout, actions_enabled=True)
    job = panel.start('measure_ai_reviewer', {'provider': other['id'], 'limit': '60'})
    assert job['label'] == 'Measure an AI reviewer'
    finished(panel, job['id'])
    assert log_of(panel, job['id']).splitlines()[0] == '== helpers/measure_ai_reviewer.sh --provider=other-one --limit=60'
    # No provider given and no reviewer chosen.
    refused(panel, 'measure_ai_reviewer', None, 409, 'no_ai_reviewer')


def test_option_checks(root, checkout):
    panel = Panel(root, repo=checkout, actions_enabled=True)
    # No provider at all: the reviewer is missing.
    refused(panel, 'ai_review', {'route': ROUTE}, 409, 'no_ai_reviewer')
    add_provider(root)
    refused(panel, 'ai_review', None)
    refused(panel, 'ai_review', {'route': 'not-a-route'})
    refused(panel, 'ai_review', {'route': '00000009--abcdef1234'})  # well formed, never prepared
    refused(panel, 'ai_review', {'route': '../' + ROUTE})
    refused(panel, 'ai_review', {'route': ROUTE, 'provider': 'missing'})
    refused(panel, 'ai_review', {'route': ROUTE, 'provider': '--save'})
    refused(panel, 'ai_review', {'route': ROUTE, 'provider': 'Fake'})
    refused(panel, 'ai_review', {'route': ROUTE, 'save': 'true'})
    for limit in ('0', '5001', '12345', 'ten', '-1', ' 20', '2.5', ''):
        refused(panel, 'ai_review', {'route': ROUTE, 'limit': limit})
    # Strings only: a number or a boolean is refused, never dropped (a limit dropped would be the whole route).
    assert 'string' in refused(panel, 'ai_review', {'route': ROUTE, 'limit': 20}).message
    refused(panel, 'ai_review', {'route': ROUTE, 'save': True})
    # An option of another action.
    refused(panel, 'measure_ai_reviewer', {'route': ROUTE})
    refused(panel, 'measure_ai_reviewer', {'save': 'yes'})
    refused(panel, 'prepare', {'provider': 'fake'})
    refused(panel, 'train', {'limit': '5'})
    refused(panel, 'ai_review', {'route': ROUTE, 'proposer': 'ours'})
    refused(panel, 'ai_review', {'route': ROUTE, 'scope': 'roads'})
    # Saving needs a provider that passed its last Test: never tested, or failed, it is refused.
    add_provider(root, 'Untested', reviewer=False)
    error = refused(panel, 'ai_review', {'route': ROUTE, 'provider': 'untested', 'save': 'yes'}, 409, 'ai_not_tested')
    assert error.message.startswith('Test Untested in Settings first')
    passed_test(root, 'fake', ok=False)
    error = refused(panel, 'ai_review', {'route': ROUTE, 'save': 'yes'}, 409, 'ai_not_tested')
    assert error.message == 'fake did not pass its last Test. Fix it, then test it again in Settings.'
    assert panel.jobs() == []


def test_read_only_panel_and_missing_helper(root, checkout):
    add_provider(root)
    refused(Panel(root, repo=checkout, actions_enabled=False), 'ai_review', {'route': ROUTE}, 403, 'actions_disabled')
    (checkout / 'helpers/ai_review.sh').unlink()
    panel = Panel(root, repo=checkout, actions_enabled=True)
    refused(panel, 'ai_review', {'route': ROUTE}, 403, 'actions_disabled')
    assert panel.status()['actions']['ai_review']['reason'] == 'helpers/ai_review.sh is missing in this installation'


def test_an_ai_review_never_runs_with_a_preparation_or_an_update(root, checkout, monkeypatch):
    add_provider(root)
    panel = Panel(root, repo=checkout, actions_enabled=True)
    monkeypatch.setenv('FAKE_SLEEP', '5')
    job = panel.start('prepare_again')
    try:
        error = refused(panel, 'ai_review', {'route': ROUTE}, 409, 'busy')
        assert error.message.startswith('A preparation is running')
        assert panel.status()['actions']['ai_review']['reason'] == 'A preparation is running'
    finally:
        panel.stop(job['id'])
        finished(panel, job['id'])
    next_second()
    job = panel.start('ai_review', {'route': ROUTE})
    try:
        assert refused(panel, 'update', None, 409, 'busy').message.startswith('An AI review is running')
        refused(panel, 'measure_ai_reviewer', None, 409, 'busy')
        refused(panel, 'prepare_again', None, 409, 'busy')
        assert panel.status()['actions']['measure_ai_reviewer']['reason'] == 'An AI review is running'
    finally:
        panel.stop(job['id'])
        finished(panel, job['id'])


def test_action_states_and_status(root, checkout):
    panel = Panel(root, repo=checkout, actions_enabled=True)
    status = panel.status()
    assert status['ai_reviewer'] == {'available': False, 'reason': 'Choose an AI reviewer in Settings.',
                                     'provider': None}
    assert status['actions']['ai_review'] == {'label': 'Review a route with AI', 'allowed': False,
                                              'reason': 'no AI reviewer: choose one in Settings'}
    assert status['actions']['measure_ai_reviewer']['reason'] == 'no AI provider: add one in Settings'
    add_provider(root, reviewer=False)
    status = panel.status()
    assert status['actions']['ai_review']['reason'] == 'no AI reviewer: choose one in Settings'
    # A provider, but no frame confirmed by anybody yet: nothing to measure on.
    assert status['actions']['measure_ai_reviewer']['reason'] == 'no confirmed frame'
    passed_test(root, 'fake')
    llm_providers.set_reviewer(root, 'fake')
    # The reviewer chosen fails its Test again: it cannot review until it passes.
    passed_test(root, 'fake', ok=False)
    status = panel.status()
    assert status['actions']['ai_review']['reason'] == 'the AI reviewer has not passed its Test: test it in Settings'
    assert status['ai_reviewer']['available'] is False and status['ai_reviewer']['provider']['id'] == 'fake'
    passed_test(root, 'fake')
    status = panel.status()
    # The AI reviews the frames nobody confirmed: it needs no confirmed frame.
    assert status['actions']['ai_review'] == {'label': 'Review a route with AI', 'allowed': True, 'reason': ''}
    assert status['ai_reviewer']['available'] is True
    assert status['ai_reviewer']['provider']['reviewer_name'] == 'ai/fake/fake-vl'
    assert KEY.encode() not in repr(status).encode()


def test_unreadable_providers_file_keeps_the_panel(root, checkout, monkeypatch, tmp_path):
    # A folder where the file should be: reading it fails, the panel still answers.
    broken = tmp_path / 'broken.json'
    broken.mkdir()
    monkeypatch.setenv(llm_providers.PROVIDERS_ENV, str(broken))
    status = Panel(root, repo=checkout, actions_enabled=True).status()
    assert status['ai_reviewer']['available'] is False and status['ai_reviewer']['reason']
    assert status['actions']['ai_review']['allowed'] is False


def test_ai_reviewers_lists_providers_and_reports_without_keys(root, checkout):
    add_provider(root)
    payload = Panel(root, repo=checkout).ai_reviewers()
    assert payload['reviewer'] == 'fake' and payload['reports'] == []
    assert payload['providers'][0]['has_key'] is True and 'key' not in payload['providers'][0]
    assert KEY not in repr(payload)


def test_progress_of_the_ai_jobs():
    for action, step in (('ai_review', 'review'), ('measure_ai_reviewer', 'measure')):
        result = progress(action, 'running', '== helpers/x.sh\n[1/10] 0/1: no sign · 1.0 s · dry run\n')
        assert [item['key'] for item in result['steps']] == [step]
        assert result['steps'][0]['state'] == 'current' and result['percent'] == 0.0
        assert result['last_line'] == '[1/10] 0/1: no sign · 1.0 s · dry run'
        task = 'ai_review' if action == 'ai_review' else 'ai_measure'
        result = with_step_progress(result, action, {'task': task, 'done': 5, 'total': 20, 'text': '0/5'})
        assert result['step_progress']['percent'] == 25.0 and result['percent'] == 25.0
        assert progress(action, 'ok', '')['percent'] == 100.0


# ---------- over HTTP: the token alone cannot start them ----------

@pytest.fixture
def lan(root, checkout):
    server = RunningServer(root, token=TOKEN, require_token=True)
    server.app.panel = Panel(root, repo=checkout, actions_enabled=True)
    yield server
    server.close()


def test_the_token_alone_cannot_start_an_ai_action(lan, root):
    add_provider(root)
    token = {'X-Review-Token': TOKEN}
    for action in ('ai_review', 'measure_ai_reviewer'):
        status, response, data = lan.request('POST', '/api/panel/run/' + action, body={'route': ROUTE}, headers=token)
        assert status == 403 and json_of(data)['error'] == 'session_required', data
    assert lan.app.panel.jobs() == []
    # The token still starts the other actions, as before.
    next_second()
    assert lan.request('POST', '/api/panel/run/prepare_again', body={}, headers=token)[0] == 200


def test_a_session_starts_an_ai_review_and_options_are_strings(lan, root):
    add_provider(root)
    session = sign_in(lan)
    status, response, data = lan.request('POST', '/api/panel/run/ai_review',
                                         body={'route': ROUTE, 'save': 'no', 'limit': 20}, headers=session)
    assert status == 400 and json_of(data)['error'] == 'bad_option'
    status, response, data = lan.request('POST', '/api/panel/run/ai_review',
                                         body={'route': ROUTE, 'save': 'no', 'limit': '20'}, headers=session)
    assert status == 200, data
    answer = json_of(data)
    assert answer['ok'] is True and answer['action'] == 'ai_review'
    finished(lan.app.panel, answer['id'])
    log = log_of(lan.app.panel, answer['id'])
    assert '--provider=fake --limit=20' in log and '--save' not in log and KEY not in log
    # The log of the job over HTTP holds no key either.
    status, response, data = lan.request('GET', f'/api/panel/job/{answer["id"]}/log', headers=session)
    assert status == 200 and KEY.encode() not in data
    status, response, data = lan.request('GET', '/api/panel/status', headers={'X-Review-Token': TOKEN})
    assert status == 200 and KEY.encode() not in data and json_of(data)['ai_reviewer']['available'] is True


AI_LOG = '''Reviewing the route {route} with the AI, saving its reviews...
Salvato: {route} fotogramma 0/0, revisione 1 (ai/fake/model, ai, web)
[1/3] 0/0: 1 sign (1 added) · 15.0 s · saved
[2/3] 0/1: no sign · 14.0 s · saved
'''


def test_the_ai_review_at_work_shows_on_the_drive_page_live(root, monkeypatch):
    from review_helpers import RUN_ID, page_body
    job = '20261026-111717-ai_review'
    (root / 'logs/panel').mkdir(parents=True)
    (root / 'logs/panel' / (job + '.log')).write_text(AI_LOG.format(route=ROUTE))
    running = [{'id': job, 'action': 'ai_review', 'state': 'running'}]
    monkeypatch.setattr(Panel, 'jobs', lambda self, activities=None: running)
    live = Panel(root).ai_live()
    assert {key: live[key] for key in ('job', 'route', 'saving', 'done', 'total', 'frame_key')} == {
        'job': job, 'route': ROUTE, 'saving': True, 'done': 2, 'total': 3, 'frame_key': '0/1'}
    assert live['seconds_left'] is not None and live['seconds_left'] >= 0
    server = RunningServer(root)
    try:
        status, response, data = server.request('GET', f'/api/route/{ROUTE}/live')
        answer = json_of(data)
        assert status == 200 and answer['run_id'] == RUN_ID and answer['reviews'] == []
        assert answer['ai']['route'] == ROUTE and answer['ai']['done'] == 2
        import json
        report = root / 'derived/preannotations' / ROUTE / RUN_ID / 'preannotations.jsonl'
        rows = [json.loads(line) for line in report.read_text().splitlines()]
        assert server.post_frame(ROUTE, '0/0', page_body(rows[0]))[0] == 200
        answer = json_of(server.request('GET', f'/api/route/{ROUTE}/live')[2])
        [first] = answer['reviews']
        assert first['frame_key'] == '0/0' and first['confirmed']
        # Only the saves after the newest one the page has.
        answer = json_of(server.request('GET', f'/api/route/{ROUTE}/live?since=' + first['updated_at'])[2])
        assert answer['reviews'] == []
        assert server.request('GET', f'/api/route/{ROUTE}/live?since=1%20OR%201')[0] == 400
        # Done: nothing at work any more.
        running.clear()
        assert json_of(server.request('GET', f'/api/route/{ROUTE}/live')[2])['ai'] is None
    finally:
        server.close()
