"""Train › Pipeline: our networks trained one after the other, each a job of its own (Panel.start_pipeline)."""
from __future__ import annotations

import json
import time

import pytest

from nnslr_tools.panel import Panel, PanelError
from test_panel import checkout, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)
from review_helpers import RunningServer, json_of
from test_panel_brain import scanned
from test_panel_models import make_signnet
from test_panel_runs import make_classifier_run, write
from test_panel_training import classifier_script, cutter_in_use

# A SignNet that finishes: its weights and its result in its run, as scripts/run_signnet.sh writes them.
FINISHING_SIGNNET = ('#!/usr/bin/env bash\necho "== scripts/run_signnet.sh $1"\necho "run:          $1"\n'
                     'echo "families=${NNSLR_CUTTER_FAMILIES:-}"\nsleep "${FAKE_SIGNNET_SLEEP:-0}"\n'
                     'mkdir -p "$NNSLR_DATA_ROOT/runs/$1"\ntouch "$NNSLR_DATA_ROOT/runs/$1/signnet-best.pt"\n'
                     'echo "{}" > "$NNSLR_DATA_ROOT/runs/$1/result.json"\nexit "${FAKE_SIGNNET_EXIT:-0}"\n')
CLASSIFIER = ('#!/usr/bin/env bash\necho "== Classifier =="\necho "run:          $1"\n'
              'echo "cutter=${NNSLR_CUTTER:-active} mapillary=${NNSLR_CLASSIFIER_MAPILLARY:-0}"\n')
DECIDER = '#!/usr/bin/env bash\necho "== Experiment B =="\necho "run:          $1"\n'

# Trainings that end as their scripts do (#71): each copies into its run a finished one the helpers of Models made, so
# Models takes what it made and the pipeline can put it in use. The brain makes three seeds, NAME-s0..2.
COPY = 'mkdir -p "$NNSLR_DATA_ROOT/runs/$1{seed}"\ncp -r "$NNSLR_DATA_ROOT/runs/{made}/." "$NNSLR_DATA_ROOT/runs/$1{seed}/"\n'
REAL_SIGNNET = FINISHING_SIGNNET.split('mkdir')[0] + COPY.format(made='made-cutter', seed='')
REAL_CLASSIFIER = CLASSIFIER + COPY.format(made='made-classifier', seed='')
REAL_DECIDER = DECIDER + COPY.format(made='made-decider', seed='') + 'exit "${FAKE_DECIDER_EXIT:-0}"\n'
REAL_BRAIN = ('#!/usr/bin/env bash\necho "== Brain =="\necho "run:          $1"\n'
              'echo "signs: ${NNSLR_BRAIN_CUTTER-unset} ${NNSLR_BRAIN_CLASSIFIER-unset} ${NNSLR_BRAIN_DECIDER-unset}"\n'
              'sleep "${FAKE_BRAIN_SLEEP:-0}"\n'
              + ''.join(COPY.format(made=f'made-brain-s{seed}', seed=f'-s{seed}') for seed in range(3)))


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


@pytest.fixture
def real(panel, root, checkout):
    """The panel whose trainings leave finished networks (REAL_*); a drive the brain can learn from. The brain's seeds:
    s1 the smallest mean error on validation, s2 none measured."""
    make_signnet(root, 'made-cutter')
    make_classifier_run(root, 'made-classifier')
    for variant in ('crop', 'crop_geom', 'full'):
        variant_run = root / 'runs' / 'made-decider' / 'experiment-b' / variant
        write(variant_run / 'result.json', {'metrics': {'validation': {}, 'test': {}}, 'best_epoch': 1, 'epochs': 1})
        (variant_run / f'{variant}-best.pt').write_bytes(b'decider')
    for seed, error in enumerate((5.0, 3.0, None)):
        speed = {'speed': {'frames': 10, 'right': 0.5, 'mean_error_kph': error}} if error is not None else {}
        write(root / 'runs' / f'made-brain-s{seed}' / 'brain-last.pt', 'weights')
        write(root / 'runs' / f'made-brain-s{seed}' / 'result.json', {
            'groups': ['signs'], 'seed': seed, 'rate_hz': 4, 'ahead': True, 'history': [{}], 'validation': speed,
            'test': speed})
    script(checkout, 'run_signnet.sh', REAL_SIGNNET)
    classifier_script(checkout, REAL_CLASSIFIER)
    script(checkout, 'run_experiment_b.sh', REAL_DECIDER)
    script(checkout, 'run_brain.sh', REAL_BRAIN)
    scanned(root)
    return panel


def in_use(root):
    """The ids in models/active.json by role, None when the file is not there."""
    path = root / 'models' / 'active.json'
    if not path.is_file():
        return None
    return {role: (entry or {}).get('id') for role, entry in json.loads(path.read_text()).items() if role != 'schema_version'}


def ended(panel, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = panel.pipeline_view()
        if view and view['state'] != 'running':
            return view
        time.sleep(0.05)
    raise AssertionError(f'the pipeline still runs: {panel.pipeline_view()}')


def test_the_networks_train_one_after_the_other_each_on_those_before_and_all_go_in_use(real, root):
    assert real.pipeline_view() is None and real.status()['pipeline'] is None
    view = real.start_pipeline({'steps': [
        {'action': 'train_cutter', 'options': {'families': 'speed'}},
        {'action': 'train_classifier', 'options': {'mapillary': 'off'}},
        {'action': 'train_decider'}, {'action': 'train_brain'}]}, 'admin')
    assert view['state'] == 'running' and view['by'] == 'admin'
    assert [step['state'] for step in view['steps']][1:] == ['todo', 'todo', 'todo']
    done = ended(real)
    assert done['state'] == 'done' and done['ended_at']
    assert [step['state'] for step in done['steps']] == ['ok', 'ok', 'ok', 'ok']
    cutter, classifier, decider, brain = (step['run'] for step in done['steps'])
    assert 'families=speed' in full_log(real, done['steps'][0]['job'])
    # The classifier learns the false boxes of the SignNet trained just before, not of the one in use.
    assert f'cutter=cutter:{cutter}/signnet mapillary=0' in full_log(real, done['steps'][1]['job'])
    assert classifier.endswith('-classifier') and decider.endswith('-decider') and brain.endswith('-brain')
    # The brain reads its signs with the three networks trained before it (#71), the applicability network of the
    # variant in use, full when none is.
    assert (f'signs: cutter:{cutter}/signnet classifier:{classifier}/classifier decider:{decider}/experiment-b/full'
            in full_log(real, done['steps'][3]['job']).splitlines())
    # Each is a job of its own, with its own log and run.
    assert {job['action'] for job in real.jobs()} == {'train_cutter', 'train_classifier', 'train_decider', 'train_brain'}
    # All ended well: every network it trained is in use, of the brain's seeds the one with the smallest error.
    # UserNet is no step of the pipeline: its role stays empty.
    assert in_use(root) == {'cutter': f'cutter:{cutter}/signnet', 'classifier': f'classifier:{classifier}/classifier',
                            'decider': f'decider:{decider}/experiment-b/full', 'brain': f'brain:{brain}-s1/brain',
                            'users': None}
    active = json.loads((root / 'models' / 'active.json').read_text())
    assert {(entry['who'], entry['via']) for role, entry in active.items()
            if role not in ('schema_version', 'users')} == {('admin', 'local')}
    assert done['reason'] == (f'Put in use: cutter {cutter}, classifier {classifier}, applicability {decider} (full), '
                              f'brain {brain}-s1.')


def test_the_brain_of_a_pipeline_takes_no_other_network_for_one_it_trains(real, root):
    make_signnet(root, 'c9')
    for steps in ([{'action': 'train_cutter'}, {'action': 'train_brain', 'options': {'sign_cutter': 'cutter:c9/signnet'}}],
                  [{'action': 'train_decider'}, {'action': 'train_brain', 'options': {'sign_decider': ''}}]):
        with pytest.raises(PanelError) as error:
            real.start_pipeline({'steps': steps}, None)
        assert error.value.status == 400 and 'trained before it' in error.value.message
    assert real.jobs() == [] and real.pipeline_view() is None


def test_the_applicability_network_put_in_use_is_of_the_variant_in_use(real, root):
    real.models_index.set_active('decider', 'decider:made-decider/experiment-b/crop_geom', who=None, via='local')
    view = real.start_pipeline({'steps': [{'action': 'train_decider'}]}, None)
    done = ended(real)
    run = done['steps'][0]['run']
    assert view['steps'][0]['action'] == 'train_decider' and done['state'] == 'done'
    assert in_use(root)['decider'] == f'decider:{run}/experiment-b/crop_geom'
    assert done['reason'] == f'Put in use: applicability {run} (crop_geom).'


def test_the_applicability_network_put_in_use_is_the_one_the_brain_read_with(real, root, monkeypatch):
    for role, model_id in (('cutter', 'cutter:made-cutter/signnet'), ('classifier', 'classifier:made-classifier/classifier'),
                           ('decider', 'decider:made-decider/experiment-b/crop_geom')):
        real.models_index.set_active(role, model_id, who=None, via='local')
    monkeypatch.setenv('FAKE_BRAIN_SLEEP', '1')
    view = real.start_pipeline({'steps': [{'action': 'train_decider'}, {'action': 'train_brain'}]}, None)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and real.pipeline_view()['steps'][1]['state'] != 'running':
        time.sleep(0.05)
    # Another variant chosen by hand while the brain trains: the brain read its signs with crop_geom.
    real.models_index.set_active('decider', 'decider:made-decider/experiment-b/crop', who=None, via='local')
    done = ended(real)
    run = done['steps'][0]['run']
    assert view['state'] == 'running' and done['state'] == 'done'
    assert done['steps'][1]['options']['sign_decider'] == f'decider:{run}/experiment-b/crop_geom'
    assert in_use(root)['decider'] == f'decider:{run}/experiment-b/crop_geom'


def test_a_pipeline_that_did_not_end_well_puts_nothing_in_use(real, root, monkeypatch):
    monkeypatch.setenv('FAKE_DECIDER_EXIT', '3')
    real.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_decider'}]}, None)
    done = ended(real)
    assert done['state'] == 'failed' and [step['state'] for step in done['steps']] == ['ok', 'failed']
    assert in_use(root) is None


def test_a_network_models_refuses_stays_out_and_the_others_go_in_use(real, root, checkout):
    # A cutter that leaves no measures: Models takes it as not finished.
    script(checkout, 'run_signnet.sh', FINISHING_SIGNNET)
    real.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_decider'}]}, None)
    done = ended(real)
    cutter, decider = (step['run'] for step in done['steps'])
    assert done['state'] == 'done' and in_use(root)['decider'] == f'decider:{decider}/experiment-b/full'
    assert in_use(root)['cutter'] is None
    assert done['reason'] == (f'Put in use: applicability {decider} (full). Not put in use: cutter {cutter}, this model '
                              'has not finished its training: its weights may still change.')


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


def test_with_two_gpus_the_pipeline_waits_only_while_both_are_busy(panel, monkeypatch):
    monkeypatch.setenv('FAKE_SIGNNET_SLEEP', '0.5')
    panel.start_pipeline({'steps': [{'action': 'train_cutter'}, {'action': 'train_decider'}]}, None)
    card = {'name': 'Tesla V100-PCIE-32GB', 'memory_used_mb': 9000, 'memory_total_mb': 32768}
    # Programs the server cannot name on both cards (issue #29): the next one waits.
    monkeypatch.setattr(panel, 'system', lambda: {'gpus': [{**card, 'index': 0, 'processes': 1},
                                                           {**card, 'index': 1, 'processes': 1}]})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and panel.pipeline_view()['steps'][0]['state'] == 'running':
        time.sleep(0.05)
    time.sleep(0.2)
    view = panel.pipeline_view()
    assert view['state'] == 'running'
    assert view['reason'] == 'Waiting: Both GPUs are busy with a program the server cannot name.'
    assert view['steps'][1]['job'] is None
    # GPU 1 free again, GPU 0 still in use: the next one starts on GPU 1.
    monkeypatch.setattr(panel, 'system', lambda: {'gpus': [{**card, 'index': 0, 'processes': 1},
                                                           {**card, 'index': 1, 'processes': 0}]})
    panel.advance_pipeline()
    done = ended(panel)
    assert done['state'] == 'done' and [step['state'] for step in done['steps']] == ['ok', 'ok']
    assert json.loads((panel.logs / (done['steps'][1]['job'] + '.json')).read_text())['gpus'] == [1]


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
