"""Train › Pipeline: our networks trained one after the other, each a job of its own (Panel.start_pipeline)."""
from __future__ import annotations

import json
import time

import pytest

from nnslr_tools.panel import Panel, PanelError
from test_panel import checkout, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)
from review_helpers import RunningServer, json_of
from test_panel_training import classifier_script, cutter_in_use

# A SignNet that finishes: its weights and its result in its run, as scripts/run_signnet.sh writes them.
FINISHING_SIGNNET = ('#!/usr/bin/env bash\necho "== scripts/run_signnet.sh $1"\necho "run:          $1"\n'
                     'echo "families=${NNSLR_CUTTER_FAMILIES:-}"\nsleep "${FAKE_SIGNNET_SLEEP:-0}"\n'
                     'mkdir -p "$NNSLR_DATA_ROOT/runs/$1"\ntouch "$NNSLR_DATA_ROOT/runs/$1/signnet-best.pt"\n'
                     'echo "{}" > "$NNSLR_DATA_ROOT/runs/$1/result.json"\nexit "${FAKE_SIGNNET_EXIT:-0}"\n')
CLASSIFIER = ('#!/usr/bin/env bash\necho "== Classifier =="\necho "run:          $1"\n'
              'echo "cutter=${NNSLR_CUTTER:-active} mapillary=${NNSLR_CLASSIFIER_MAPILLARY:-0}"\n')
DECIDER = '#!/usr/bin/env bash\necho "== Experiment B =="\necho "run:          $1"\n'


def script(checkout, name, text):
    path = checkout / 'scripts' / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def panel(root, checkout, monkeypatch):
    script(checkout, 'run_signnet.sh', FINISHING_SIGNNET)
    classifier_script(checkout, CLASSIFIER)
    script(checkout, 'run_experiment_b.sh', DECIDER)
    made = Panel(root, repo=checkout)
    monkeypatch.setattr(made, 'activities', lambda: [])
    monkeypatch.setattr(made, 'system', lambda: {'gpu': {'processes': 0}})
    return made


def ended(panel, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = panel.pipeline_view()
        if view and view['state'] != 'running':
            return view
        time.sleep(0.05)
    raise AssertionError(f'the pipeline still runs: {panel.pipeline_view()}')


def test_the_networks_train_one_after_the_other_the_classifier_on_the_new_cutter(panel):
    assert panel.pipeline_view() is None and panel.status()['pipeline'] is None
    view = panel.start_pipeline({'steps': [
        {'action': 'train_cutter', 'options': {'families': 'speed'}},
        {'action': 'train_classifier', 'options': {'mapillary': 'off'}},
        {'action': 'train_decider'}]}, 'admin')
    assert view['state'] == 'running' and view['by'] == 'admin'
    assert [step['state'] for step in view['steps']][1:] == ['todo', 'todo']
    done = ended(panel)
    assert done['state'] == 'done' and done['reason'] == '' and done['ended_at']
    assert [step['state'] for step in done['steps']] == ['ok', 'ok', 'ok']
    cutter, classifier, decider = done['steps']
    assert 'families=speed' in full_log(panel, cutter['job'])
    # The classifier learns the false boxes of the SignNet trained just before, not of the one in use.
    assert f'cutter=cutter:{cutter["run"]}/signnet mapillary=0' in full_log(panel, classifier['job'])
    assert classifier['run'].endswith('-classifier') and decider['run'].endswith('-decider')
    # Each is a job of its own, with its own log and run.
    assert {job['action'] for job in panel.jobs()} == {'train_cutter', 'train_classifier', 'train_decider'}


def test_a_failed_training_stops_the_pipeline(panel, monkeypatch):
    monkeypatch.setenv('FAKE_SIGNNET_EXIT', '4')
    panel.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_classifier'}]}, None)
    done = ended(panel)
    assert done['state'] == 'failed' and done['reason'] == 'Train the cutter failed.'
    assert [step['state'] for step in done['steps']] == ['failed', 'skipped']
    assert [job['action'] for job in panel.jobs()] == ['train_cutter']


def test_stop_after_this_one_lets_the_training_finish_and_starts_no_other(panel, monkeypatch):
    monkeypatch.setenv('FAKE_SIGNNET_SLEEP', '1')
    panel.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_decider'}]}, None)
    with pytest.raises(PanelError) as busy:
        panel.start_pipeline({'steps': [{'action': 'train_decider'}]}, None)
    assert busy.value.status == 409
    stopped = panel.stop_pipeline()
    assert stopped['state'] == 'stopped' and [step['state'] for step in stopped['steps']] == ['running', 'skipped']
    time.sleep(1.5)
    panel.advance_pipeline()
    assert [job['action'] for job in panel.jobs()] == ['train_cutter']
    assert [job['state'] for job in panel.jobs()] == ['ok']
    with pytest.raises(PanelError) as error:
        panel.stop_pipeline()
    assert error.value.status == 409


def test_a_stopped_training_stops_the_pipeline(panel, monkeypatch):
    monkeypatch.setenv('FAKE_SIGNNET_SLEEP', '5')
    view = panel.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_decider'}]}, None)
    time.sleep(0.3)
    panel.stop(view['steps'][0]['job'])
    done = ended(panel)
    assert done['state'] == 'stopped' and done['reason'] == 'Train the cutter was stopped.'


def test_the_pipeline_waits_for_a_busy_gpu_and_goes_on_after_a_restart(panel, root, checkout, monkeypatch):
    panel.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_decider'}]}, None)
    # The next one waits while another program uses the GPU.
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': {'processes': 1}})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and panel.pipeline_view()['steps'][0]['state'] == 'running':
        time.sleep(0.05)
    time.sleep(0.2)
    view = panel.pipeline_view()
    assert view['state'] == 'running' and view['reason'] == 'Waiting: The GPU is busy with another process.'
    assert view['steps'][1]['job'] is None
    # A new server (a restart) goes on from the file: here at once, by hand.
    again = Panel(root, repo=checkout)
    monkeypatch.setattr(again, 'activities', lambda: [])
    monkeypatch.setattr(again, 'system', lambda: {'gpu': {'processes': 0}})
    again.advance_pipeline()
    done = ended(again)
    assert done['state'] == 'done' and [step['state'] for step in done['steps']] == ['ok', 'ok']


def test_the_classifier_alone_uses_the_cutter_in_use(panel, root):
    with pytest.raises(PanelError) as error:
        panel.start_pipeline({'steps': [{'action': 'train_classifier'}]}, None)
    assert error.value.status == 409 and 'cutter' in error.value.message
    cutter_in_use(root, panel)
    view = panel.start_pipeline({'steps': [{'action': 'train_classifier'}]}, None)
    done = ended(panel)
    assert done['state'] == 'done'
    assert 'cutter=active' in full_log(panel, view['steps'][0]['job'])


@pytest.mark.parametrize('body, status', [
    ({}, 400), ({'steps': []}, 400), ({'steps': [{'action': 'train'}]}, 400),
    ({'steps': [{'action': 'update'}]}, 400),
    ({'steps': [{'action': 'train_decider'}, {'action': 'train_cutter'}]}, 400),
    ({'steps': [{'action': 'train_cutter'}, {'action': 'train_cutter'}]}, 400),
    ({'steps': [{'action': 'train_cutter', 'options': {'families': 'other'}}]}, 400),
    ({'steps': [{'action': 'train_cutter', 'options': {'mapillary': 'yes'}}]}, 400),
    # The cutter's Mapillary switch needs its windows cut (helpers/mapillary_windows.sh).
    ({'steps': [{'action': 'train_cutter', 'options': {'mapillary': 'on'}}]}, 409),
    ({'steps': [{'action': 'train_cutter', 'options': {'mapillary': 'off', 'share': '5'}}]}, 400),
    ({'steps': [{'action': 'train_cutter'}, {'action': 'train_classifier', 'options': {'cutter': 'cutter:x/signnet'}}]}, 400),
    ({'steps': [{'action': 'train_cutter'}, {'action': 'train_classifier', 'options': {'mapillary': 'on'}}]}, 409),
    ({'steps': [{'action': 'train_cutter', 'options': {'families': 3}}]}, 400),
    ({'steps': [{'action': 'train_cutter', 'options': {'start': 'cutter:nothing/signnet'}}]}, 400),
    ({'steps': [{'action': 'train_decider', 'options': {'start': 'decider:nothing/experiment-b/crop'}}]}, 400)])
def test_a_bad_pipeline_is_refused_before_anything_starts(panel, body, status):
    with pytest.raises(PanelError) as error:
        panel.start_pipeline(body, None)
    assert error.value.status == status, error.value.message
    assert panel.jobs() == [] and panel.pipeline_view() is None


def test_the_classifier_takes_a_finished_signnet_as_its_cutter(panel, root):
    with pytest.raises(PanelError) as error:
        panel.start('train_classifier', {'cutter': 'cutter:nothing/signnet'})
    assert error.value.status == 409
    for bad in ('/etc/passwd', 'cutter:../x/signnet', 'classifier:c1/classifier'):
        with pytest.raises(PanelError) as error:
            panel.start('train_classifier', {'cutter': bad})
        assert error.value.status == 400
    with pytest.raises(PanelError) as error:
        panel.start('train_cutter', {'cutter': 'cutter:s1/signnet'})
    assert error.value.status == 400
    run = root / 'runs' / 's9'
    run.mkdir(parents=True)
    (run / 'signnet-best.pt').write_bytes(b'')
    (run / 'result.json').write_text(json.dumps({}))
    job = panel.start('train_classifier', {'cutter': 'cutter:s9/signnet'})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and panel._job_state(job['id']) == 'running':
        time.sleep(0.05)
    assert 'cutter=cutter:s9/signnet' in full_log(panel, job['id'])


def test_the_pipeline_is_read_only_on_a_read_only_panel(root, checkout):
    panel = Panel(root, repo=checkout, actions_enabled=False)
    for call in (lambda: panel.start_pipeline({'steps': [{'action': 'train_decider'}]}, None), panel.stop_pipeline):
        with pytest.raises(PanelError) as error:
            call()
        assert error.value.status == 403


def test_the_pipeline_over_http(root, checkout, panel):
    server = RunningServer(root)
    server.app.panel = panel
    try:
        status, _, data = server.request('POST', '/api/panel/pipeline', body={'steps': [{'action': 'train_decider'}]},
                                         headers={'Origin': 'http://evil.example'})
        assert status == 403 and panel.pipeline_view() is None
        status, _, data = server.request('POST', '/api/panel/pipeline', body={'steps': [{'action': 'nothing'}]})
        assert status == 400
        status, _, data = server.request('POST', '/api/panel/pipeline', body={'steps': [{'action': 'train_decider'}]})
        assert status == 200 and json_of(data)['pipeline']['steps'][0]['action'] == 'train_decider'
        done = ended(panel)
        assert done['state'] == 'done'
        assert json_of(server.request('GET', '/api/panel/status')[2])['pipeline']['state'] == 'done'
        status, _, data = server.request('POST', '/api/panel/pipeline/stop', body={})
        assert status == 409
    finally:
        server.close()
