"""Train › UserNet on the panel (scripts/run_usernet.sh NAME, a stand-in here): when it can start, its stage and its
start in the script's environment, a run of its own, its settings, the options it refuses, and the progress of one
started from a terminal. Nothing here trains."""
from __future__ import annotations

import json
import re
import time

import pytest

from nnslr_tools import train_settings
from nnslr_tools.panel import Panel, PanelError, _run_of, _settings_of
from test_panel import checkout, finished, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)
from test_panel_training import ready_payload

FAKE_USERNET = ('#!/usr/bin/env bash\necho "== scripts/run_usernet.sh $1"\necho "run:          $1"\n'
                'echo "settings:     --stage ${NNSLR_USERNET_STAGE:-unset} --epochs ${NNSLR_USERNET_EPOCHS:-30}"\n'
                'echo "stage=${NNSLR_USERNET_STAGE:-unset} init=${NNSLR_USERNET_INIT_FROM:-fresh}"\n')


def usernet_script(checkout):  # noqa: F811
    script = checkout / 'scripts' / 'run_usernet.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(FAKE_USERNET)
    script.chmod(0o755)
    return script


@pytest.fixture
def bdd(tmp_path, monkeypatch):
    """BDD100K on this server: its samples.json."""
    folder = tmp_path / 'bdd100k'
    folder.mkdir()
    (folder / 'samples.json').write_text('{"samples": []}')
    monkeypatch.setenv('NNSLR_BDD_DIR', str(folder))
    return folder


def usernet_run(root, name: str, *, finished_run: bool = True):  # noqa: F811
    run = root / 'runs' / name
    run.mkdir(parents=True)
    (run / 'usernet-best.pt').write_bytes(b'weights')
    if finished_run:
        (run / 'result.json').write_text(json.dumps({
            'task': 'usernet', 'stage': 'bdd', 'best_epoch': 9, 'epochs': 30, 'history': [{}] * 30, 'uses_bdd': True,
            'validation_source': 'bdd', 'validation': {'frames': 10, 'at_min_score': {'recall': 0.6}}, 'test': None}))
    return run


def test_train_usernet_needs_its_script_and_a_card_but_no_confirmed_sign(root, checkout):  # noqa: F811
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['train_users'] == {
        'label': 'Train UserNet', 'allowed': False, 'reason': 'scripts/run_usernet.sh is missing in this installation'}
    usernet_script(checkout)
    assert panel._action_states(ready_payload(), [])['train_users'] == {'label': 'Train UserNet', 'allowed': True,
                                                                        'reason': ''}
    # BDD100K's pictures need no drive of ours; the drives stage is checked by its script's dry run.
    none = {**ready_payload(), 'routes': [], 'reviewed_routes': 0}
    assert panel._action_states(none, [])['train_users']['allowed'] is True
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_users']['reason'] == (
        'the GPU is busy with another process')
    busy = [{'group': 'train', 'label': 'UserNet training', 'pid': 1}]
    assert panel._action_states(ready_payload(), busy)['train_users']['reason'] == 'the GPU is busy with a UserNet training'


def test_train_usernet_runs_on_a_new_run_with_its_stage_and_settings(root, checkout, bdd, monkeypatch):  # noqa: F811
    usernet_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    train_settings.save(root, {'training': 'train_users', 'values': {'epochs': 12}}, 'cristian')
    job = panel.start('train_users', None)
    done = finished(panel, job['id'])
    log = full_log(panel, job['id'])
    first = log.splitlines()[0].split()
    name = first[2]
    # The run name is its only argument; BDD100K is the stage when none is asked.
    assert first[:2] == ['==', 'scripts/run_usernet.sh'] and len(first) == 3
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-users', name) and 'stage=bdd init=fresh\n' in log
    assert (done['state'], done['run'], done['label']) == ('ok', name, 'Train UserNet')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_users') == name
    assert _settings_of(panel.logs / (job['id'] + '.log'), 'train_users')[:2] == ['BDD100K', '12 epochs']
    time.sleep(1.1)  # a job's log is named by its second
    job = panel.start('train_users', {'stage': 'drives'})
    finished(panel, job['id'])
    assert 'stage=drives init=fresh\n' in full_log(panel, job['id'])
    assert _settings_of(panel.logs / (job['id'] + '.log'), 'train_users')[0] == 'your drives'


@pytest.mark.parametrize('action, options', [
    ('train_users', {'stage': 'mapillary'}), ('train_users', {'stage': 'bdd ; rm -rf /'}), ('train_users', {'stage': 3}),
    ('train_users', {'families': 'all'}), ('train_users', {'start': 'cutter:s1/signnet'}),
    ('train_users', {'start': 'imagenet'}), ('train_cutter', {'stage': 'bdd'})])
def test_bad_options_of_train_usernet_are_refused(root, checkout, bdd, monkeypatch, action, options):  # noqa: F811
    usernet_script(checkout)
    (checkout / 'scripts' / 'run_signnet.sh').write_text('#!/usr/bin/env bash\n')
    (checkout / 'scripts' / 'run_signnet.sh').chmod(0o755)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start(action, options)
    assert (refused.value.status, refused.value.reason) == (400, 'bad_option') and panel.jobs() == []


def test_the_bdd_stage_needs_bdd100k_on_this_server(root, checkout, monkeypatch, tmp_path):  # noqa: F811
    usernet_script(checkout)
    monkeypatch.setenv('NNSLR_BDD_DIR', str(tmp_path / 'nowhere'))
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    for options in (None, {'stage': 'bdd'}):
        with pytest.raises(PanelError) as refused:
            panel.start('train_users', options)
        assert (refused.value.status, refused.value.reason) == (409, 'not_available')
    assert panel.jobs() == []
    job = panel.start('train_users', {'stage': 'drives'})
    assert finished(panel, job['id'])['state'] == 'ok'


def test_train_usernet_goes_on_from_a_finished_usernet_of_models(root, checkout, monkeypatch):  # noqa: F811
    usernet_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    usernet_run(root, 'u1')
    usernet_run(root, 'u2', finished_run=False)
    job = panel.start('train_users', {'stage': 'drives', 'start': 'users:u1/usernet'})
    finished(panel, job['id'])
    init = re.search(r'init=(.*)$', full_log(panel, job['id']), re.M).group(1)
    assert init == str(root / 'runs' / 'u1' / 'usernet-best.pt')
    refused = {'users:u2/usernet': 'That network has not finished its training: choose a finished one.',
               'brain:u1/usernet': 'Unknown network to start from: choose one of Models.',
               'users:nope/usernet': 'Unknown network to start from: choose one of Models.'}
    for start, message in refused.items():
        with pytest.raises(PanelError) as error:
            panel.start('train_users', {'stage': 'drives', 'start': start})
        assert (error.value.status, error.value.message) == (400, message), start
    # A job says the UserNet it went on from.
    log = root / 'job.log'
    log.write_text('== scripts/run_usernet.sh r\nrun:          r\nsettings:     --stage drives --bdd-dir /d --epochs 30 '
                   '--init-from /d/runs/2026-11-12_1200-users/usernet-best.pt\n')
    chips = _settings_of(log, 'train_users')
    assert chips[0] == 'your drives' and chips[-1] == 'from 2026-11-12_1200-users'


def test_the_stage_reaches_the_panel_over_http(root, checkout, bdd):  # noqa: F811
    from review_helpers import RunningServer, json_of
    usernet_script(checkout)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = lambda: []
    try:
        status, _, data = server.request('POST', '/api/panel/run/train_users', body={'stage': 'mapillary'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, _, data = server.request('POST', '/api/panel/run/train_users', body={'stage': 'drives'})
        assert status == 200, data
        job_id = json_of(data)['id']
        finished(server.app.panel, job_id)
        assert 'stage=drives init=fresh\n' in full_log(server.app.panel, job_id)
    finally:
        server.close()


def test_a_usernet_training_started_from_a_terminal_has_the_progress_of_its_log(root, checkout):  # noqa: F811
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / 'u1' / 'usernet.log'
    log.parent.mkdir(parents=True)
    log.write_text('== scripts/run_usernet.sh u1\n== 1/3 Check the training population (CPU-only) ==\n'
                   '== 2/3 Explicit CUDA smoke test ==\n== 3/3 Train UserNet ==\n'
                   'usernet epoch 4/30 train_loss 2.1000 val_recall 0.3000 val_false_per_frame 1.80 60s\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'UserNet training', 'pid': 1}]}
    panel._terminal_training(payload)
    progress = payload['activities'][0]['progress']
    assert [(step['key'], step['state']) for step in progress['steps']] == [
        ('checks', 'done'), ('usernet', 'current'), ('evaluate', 'todo')]
    assert progress['detail'] == 'UserNet · epoch 4 of 30'
    assert progress['curves']['usernet']['points'][0]['val_false_per_frame'] == 1.8
