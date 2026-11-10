"""The Train tab's data: what the dataset holds, where a training can start, and the start option."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime

import pytest

from nnslr_tools.panel import Panel, PanelError
from test_panel import checkout, finished, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)

ENV_TRAIN = '#!/usr/bin/env bash\necho "train $* init=${NNSLR_TRAIN_INIT:-} from=${NNSLR_TRAIN_FROM_RUN:-}"\n'


def write_dataset(root):
    (root / 'annotations').mkdir(parents=True, exist_ok=True)
    rows = [
        {'route_id': 'r1', 'image_path': 'f/1.png', 'sign_family': 'maximum_speed', 'provenance': {'label_basis': 'human_review'}},
        {'route_id': 'r1', 'image_path': 'f/1.png', 'sign_family': 'not_a_sign', 'provenance': {'label_basis': 'ai_review'}},
        {'route_id': 'r2', 'image_path': 'f/2.png', 'sign_family': 'other_sign', 'provenance': {'label_basis': 'ai_review'}},
    ]
    (root / 'annotations' / 'objects.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'splits').mkdir(exist_ok=True)
    (root / 'splits' / 'abc.json').write_text(json.dumps({'counts': {'train': 2, 'test': 1}, 'warnings': ['w']}))
    (root / 'splits' / 'latest.json').write_text(json.dumps({'path': 'splits/abc.json'}))
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'import.json').write_text(json.dumps({
        'imported_frame_count': 1, 'already_imported_frame_count': 4, 'changed_after_import': ['a', 'b']}))


def test_the_dataset_summary_counts_frames_objects_families_and_reviewers(root, checkout):
    write_dataset(root)
    summary = Panel(root, repo=checkout).dataset_summary()
    assert (summary['frames'], summary['objects'], summary['routes'], summary['human'], summary['ai']) == (2, 3, 2, 1, 2)
    assert [(item['key'], item['count']) for item in summary['families']][0] == ('maximum_speed', 1)
    assert {item['label'] for item in summary['families']} == {'Speed limit', 'Not a sign', 'Other sign'}
    assert summary['per_route'][0] == {'route': 'r1', 'frames': 1, 'objects': 2, 'speed_signs': 1, 'human': 1, 'ai': 1}
    assert summary['split']['counts'] == {'train': 2, 'test': 1} and summary['split']['warnings'] == ['w']
    # An import made before changed frames were replaced: they were left unapplied.
    assert summary['last_import'] == {'at': summary['last_import']['at'], 'new_frames': 1, 'already': 4,
                                      'changed_after_import': 2, 'replaced_frames': 0, 'removed_frames': 0,
                                      'warnings': []}
    (root / 'logs' / 'import.json').write_text(json.dumps({
        'imported_frame_count': 1, 'already_imported_frame_count': 4, 'replaced_frame_count': 2,
        'changed_after_import': ['a', 'b'], 'removed_withdrawn_frame_count': 3}))
    imported = Panel(root, repo=checkout).dataset_summary()['last_import']
    assert (imported['changed_after_import'], imported['replaced_frames'], imported['removed_frames']) == (0, 2, 3)


def test_the_split_says_which_dataset_it_was_made_of_and_the_rejected_proposals_it_leaves_out(root, checkout):
    write_dataset(root)
    sha = hashlib.sha256((root / 'annotations' / 'objects.jsonl').read_bytes()).hexdigest()
    summary = Panel(root, repo=checkout).dataset_summary()
    # A split without them (an older one): nothing to compare, nothing left out.
    assert summary['sha256'] == sha and (summary['split']['dataset_sha256'], summary['split']['rejected']) == (None, 0)
    (root / 'splits' / 'abc.json').write_text(json.dumps({'counts': {'train': 2}, 'warnings': [], 'dataset_sha256': sha,
                                                          'excluded_rejected_ids': ['x']}))
    split = Panel(root, repo=checkout).dataset_summary()['split']
    assert (split['dataset_sha256'], split['rejected']) == (sha, 1)
    # The dataset changes after the split: its sha256 is no longer the one the split recorded.
    with (root / 'annotations' / 'objects.jsonl').open('a') as target:
        target.write(json.dumps({'route_id': 'r3', 'image_path': 'f/3.png', 'sign_family': 'zone'}) + '\n')
    summary = Panel(root, repo=checkout).dataset_summary()
    assert summary['sha256'] != summary['split']['dataset_sha256'] and summary['objects'] == 4


def test_training_starts_list_imagenet_random_stable_and_finished_runs(root, checkout):
    run = root / 'runs' / '2026-10-24_1014'
    for task in ('detector', 'reader'):
        (run / task).mkdir(parents=True)
        (run / task / (task + '-best.pt')).write_bytes(b'x')
    (root / 'runs' / 'unfinished' / 'detector').mkdir(parents=True)
    panel = Panel(root, repo=checkout)
    panel._stable = (time.monotonic(), False)
    starts = {item['key']: item for item in panel.training_starts()}
    assert list(starts) == ['imagenet', 'random', 'stable', 'run:2026-10-24_1014']
    assert starts['imagenet']['default'] and starts['imagenet']['available']
    assert not starts['stable']['available'] and starts['stable']['reason'].startswith('No stable model')


@pytest.mark.parametrize('start, expected', [(None, 'train  init= from='), ('random', 'train  init=random from='),
                                             ('stable', 'train --continue init= from=')])
def test_the_start_of_a_training_becomes_a_flag_or_the_environment(root, checkout, start, expected, monkeypatch):
    (checkout / 'helpers' / '3_train.sh').write_text(ENV_TRAIN)
    panel = Panel(root, repo=checkout)
    # A real training running on this machine would make the panel refuse to start another one.
    monkeypatch.setattr(panel, 'activities', lambda: [])
    panel._stable = (time.monotonic(), True)
    job = panel.start('train', {'start': start} if start else None)
    finished(panel, job['id'])
    assert expected in full_log(panel, job['id'])


def test_a_bad_or_unavailable_start_is_refused(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    panel._stable = (time.monotonic(), False)
    for options, status in (({'start': 'rm -rf'}, 400), ({'start': 'stable'}, 409), ({'start': 'run:nope'}, 400)):
        with pytest.raises(PanelError) as refused:
            panel.start('train', options)
        assert refused.value.status == status, options
    with pytest.raises(PanelError) as other:
        panel.start('prepare', {'start': 'random'})
    assert other.value.status == 400


def test_jobs_keep_their_temporary_files_in_the_data_root(root, checkout, monkeypatch):
    """Not in /tmp: the service's private /tmp is emptied at every restart of the site."""
    (checkout / 'helpers' / '3_train.sh').write_text('#!/usr/bin/env bash\necho "tmp=$TMPDIR"\n')
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train')
    finished(panel, job['id'])
    assert f'tmp={root / "tmp"}' in full_log(panel, job['id']) and (root / 'tmp').is_dir()


def test_a_stopped_training_can_be_resumed_on_its_run(root, checkout):
    from nnslr_tools.panel import PanelError, _run_of
    panel = Panel(root, repo=checkout)
    logs = root / 'logs' / 'panel'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / '20261024-145218-train.log'
    log.write_text('== helpers/3_train.sh\n== a. importo\n== d. training 2026-10-24_1452 (fresh) -> /x\n')
    assert _run_of(log) == '2026-10-24_1452'
    for bad in ({}, {'run': '../etc'}, {'run': 'missing-run'}):
        try:
            panel.start('train_resume', bad)
        except PanelError as error:
            assert error.status == 400
        else:
            raise AssertionError('accepted ' + repr(bad))
    try:
        panel.start('train', {'run': '2026-10-24_1452'})
    except PanelError as error:
        assert error.status == 400


# -- "Train the classifier" (scripts/run_classifier.sh) --------------------------------------------------------------
FAKE_CLASSIFIER = ('#!/usr/bin/env bash\necho "== Classifier (cutter -> classifier -> decider) =="\necho "run:          $1"\n'
                   'echo "args $*"\nsleep "${FAKE_CLASSIFIER_SLEEP:-0}"\n')


def classifier_script(checkout, text=FAKE_CLASSIFIER):
    (checkout / 'scripts').mkdir(exist_ok=True)
    script = checkout / 'scripts' / 'run_classifier.sh'
    script.write_text(text)
    script.chmod(0o755)
    return script


def cutter_in_use(root, panel):
    """A finished SignNet of the car's band and grid (runs/s1: signnet-best.pt, result.json), the cutter in use."""
    from test_panel_models import make_signnet
    from test_panel_runs import make_run
    make_run(root, 'r1')
    run = make_signnet(root, 's1')
    panel.models_index.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    return run


def ready_payload(gpu_processes=0):
    """A status where every rule of the trainings is met: confirmed frames on three reviewed routes, a free GPU."""
    return {'actions_enabled': True, 'routes': [{'state': 'pronta', 'confirmed': 4}] * 3, 'reviewed_routes': 3,
            'min_routes': 3, 'system': {'gpu': {'processes': gpu_processes}}}


def test_the_classifier_training_follows_the_rules_of_a_training_and_needs_a_cutter(root, checkout):
    panel = Panel(root, repo=checkout)
    assert panel.available()
    # Without the script the other actions are there; this one says why it is not.
    assert panel._action_states(ready_payload(), [])['train_classifier'] == {
        'label': 'Train the classifier', 'allowed': False, 'reason': 'scripts/run_classifier.sh is missing in this installation'}
    assert panel._action_states(ready_payload(), [])['train']['allowed']
    classifier_script(checkout)
    assert panel._action_states(ready_payload(), [])['train_classifier']['reason'] == 'no cutter in use: choose one on the Models tab'
    run = cutter_in_use(root, panel)
    assert panel._action_states(ready_payload(), [])['train_classifier'] == {'label': 'Train the classifier', 'allowed': True, 'reason': ''}
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_classifier']['reason'] == 'the GPU is busy with another process'
    few = {**ready_payload(), 'reviewed_routes': 2}
    assert panel._action_states(few, [])['train_classifier']['reason'] == '3 reviewed drives are needed, now 2'
    none = {**ready_payload(), 'routes': [{'state': 'revisionare', 'confirmed': 0}]}
    assert panel._action_states(none, [])['train_classifier']['reason'] == 'no confirmed frame'
    # Two classifiers may train at once, each on a card of its own: one from a terminal on a card not known holds
    # every card, the one GPU here.
    busy = [{'group': 'train', 'label': 'classifier training', 'pid': 1}]
    assert panel._action_states(ready_payload(), busy)['train_classifier']['reason'] == (
        'the GPU is busy with a classifier training')
    (run / 'signnet-best.pt').unlink()
    assert panel._action_states(ready_payload(), [])['train_classifier']['reason'] == (
        'the cutter in use is no longer on this server: choose another one on the Models tab')
    # The page's status has it too, with its reason.
    assert panel.status()['actions']['train_classifier']['allowed'] is False


def test_the_classifier_training_runs_on_a_new_run_named_by_its_time(root, checkout, monkeypatch):
    from nnslr_tools.panel import _run_of
    classifier_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    cutter_in_use(root, panel)
    for options in ({'start': 'random'}, {'run': 'r1'}):
        with pytest.raises(PanelError) as refused:
            panel.start('train_classifier', options)
        assert refused.value.status == 400
    job = panel.start('train_classifier')
    assert (job['action'], job['label']) == ('train_classifier', 'Train the classifier')
    done = finished(panel, job['id'])
    log = full_log(panel, job['id'])
    name = log.splitlines()[0].split()[-1]
    assert log.startswith('== scripts/run_classifier.sh ') and f'args {name}\n' in log
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-classifier', name)
    assert (done['state'], done['run']) == ('ok', name)
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_classifier') == name
    # The run line of the script, when the first line is not the panel's.
    other = panel.logs / '20261024-173000-train_classifier.log'
    other.write_text('something\nrun:          2026-10-24_1730-classifier-2\n')
    assert _run_of(other, 'train_classifier') == '2026-10-24_1730-classifier-2' and _run_of(other) is None


def test_the_name_of_a_classifier_run_never_reuses_a_folder(root, checkout, monkeypatch):
    import nnslr_tools.panel as panel_module

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 24, 17, 30)
    monkeypatch.setattr(panel_module, 'datetime', Fixed)
    panel = Panel(root, repo=checkout)
    assert panel.new_run_name('classifier') == '2026-10-24_1730-classifier'
    (root / 'runs' / '2026-10-24_1730-classifier').mkdir(parents=True)
    assert panel.new_run_name('classifier') == '2026-10-24_1730-classifier-2'
    (root / 'runs' / '2026-10-24_1730-classifier-2').write_text('a file is taken too')
    assert panel.new_run_name('classifier') == '2026-10-24_1730-classifier-3'


def test_the_classifier_training_is_refused_without_its_script_or_a_cutter(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as missing:
        panel.start('train_classifier')
    assert (missing.value.status, missing.value.message) == (403, 'scripts/run_classifier.sh is missing in this installation.')
    classifier_script(checkout)
    with pytest.raises(PanelError) as no_cutter:
        panel.start('train_classifier')
    assert (no_cutter.value.status, no_cutter.value.reason, no_cutter.value.message) == (
        409, 'no_cutter', 'No cutter in use: choose one on the Models tab.')
    assert panel.jobs() == []


def test_a_running_classifier_training_is_seen_and_stopped_after_a_restart(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_CLASSIFIER_SLEEP', '30')
    classifier_script(checkout)
    panel = Panel(root, repo=checkout)
    cutter_in_use(root, panel)
    job = panel.start('train_classifier')
    # The wrapper runs the script with its run: the work of the train group, also as an activity.
    assert any(item['label'] == 'classifier training' for item in panel.activities())
    with pytest.raises(PanelError) as busy:
        panel.start('train')
    assert busy.value.status == 409
    restarted = Panel(root, repo=checkout)
    assert [item['stoppable'] for item in restarted.jobs() if item['id'] == job['id']] == [True]
    assert restarted.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(restarted, job['id'])['state'] in ('interrupted', 'failed')


def test_a_classifier_training_started_from_a_terminal_has_the_progress_of_its_run_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / 'c1' / 'classifier.log'
    log.parent.mkdir(parents=True)
    log.write_text('== 1/7 Check ==\n== 2/7 Smoke ==\n== 3/7 Mine ==\n== 4/7 Train the classifier ==\n'
                   'classifier epoch 2/20 train_loss 0.9000 val_loss 1.0000 train_accuracy 0.5000 val_accuracy 0.6000 30s\n')
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'train-r1.log').write_text('== a. importo\n== 4/10 Train detector ==\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'classifier training', 'pid': 1},
                                          {'group': 'train', 'label': 'dataset and training', 'pid': 2}]}
    panel._terminal_training(payload)
    classifier, training = payload['activities']
    assert classifier['progress']['detail'] == 'classifier · epoch 2 of 20'
    assert [step['key'] for step in classifier['progress']['steps']] == ['checks', 'mine', 'classifier', 'evaluate', 'chain']
    assert training['progress']['steps'][4] == {'key': 'detector', 'label': 'Train the detector', 'state': 'current'}


# -- "Train the cutter" (scripts/run_signnet.sh NAME): SignNet at the car's band ------------------------------------
FAKE_SIGNNET = ('#!/usr/bin/env bash\necho "== scripts/run_signnet.sh $1"\necho "run:          $1"\n'
                'echo "args $* families=${NNSLR_CUTTER_FAMILIES:-}"\nsleep "${FAKE_SIGNNET_SLEEP:-0}"\n')


def signnet_script(checkout, text=FAKE_SIGNNET):
    (checkout / 'scripts').mkdir(exist_ok=True)
    script = checkout / 'scripts' / 'run_signnet.sh'
    script.write_text(text)
    script.chmod(0o755)
    return script


def test_the_cutter_training_follows_the_rules_of_a_training(root, checkout):
    panel = Panel(root, repo=checkout)
    states = panel._action_states(ready_payload(), [])
    assert states['train_cutter'] == {'label': 'Train the cutter', 'allowed': False,
                                      'reason': 'scripts/run_signnet.sh is missing in this installation'}
    assert states['train']['label'] == 'Whole pipeline'
    signnet_script(checkout)
    # No cutter in use is needed: it trains one.
    assert panel._action_states(ready_payload(), [])['train_cutter'] == {'label': 'Train the cutter', 'allowed': True, 'reason': ''}
    assert panel._action_states(ready_payload(gpu_processes=2), [])['train_cutter']['reason'] == 'the GPU is busy with another process'
    assert panel._action_states({**ready_payload(), 'reviewed_routes': 1}, [])['train_cutter']['reason'] == '3 reviewed drives are needed, now 1'
    # The signs and the Mapillary windows are the choices: the band is the car's.
    options = panel.status()['cutter_options']
    assert {key: options[key] for key in ('families', 'family', 'band')} == {'families': ['all', 'speed'], 'family': 'all',
                                                                            'band': '896x288'}
    assert options['mapillary'] == {'available': False, 'windows': 0, 'photos': 0, 'boxes_by_family': {},
                                    'reason': 'the Mapillary windows are not cut yet (helpers/mapillary_windows.sh)',
                                    'licence': 'CC BY-NC-SA 4.0'}
    assert (options['mapillary_regions']['fetched'], options['mapillary_regions']['crops']) == (False, 0)


@pytest.mark.parametrize('options, expected', [(None, 'families=all'), ({'families': 'all'}, 'families=all'),
                                               ({'families': 'speed'}, 'families=speed')])
def test_the_cutter_training_runs_signnet_on_a_new_run(root, checkout, monkeypatch, options, expected):
    from nnslr_tools.panel import _run_of
    signnet_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_cutter', options)
    done = finished(panel, job['id'])
    log = full_log(panel, job['id'])
    first = log.splitlines()[0].split()
    name = first[2]
    # The run name is its only argument.
    assert first[:2] == ['==', 'scripts/run_signnet.sh'] and len(first) == 3
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-cutter', name) and f'args {name} {expected}\n' in log
    assert (done['state'], done['run'], done['label']) == ('ok', name, 'Train the cutter')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_cutter') == name
    # A log of the old detector sizes still names its run.
    other = panel.logs / '20261024-180000-train_cutter.log'
    other.write_text('== scripts/run_detector_sizes.sh 2026-10-24_1800-cutter-2 960x544\n')
    assert _run_of(other, 'train_cutter') == '2026-10-24_1800-cutter-2'


def test_the_decider_is_trained_from_the_train_tab_on_a_new_run(root, checkout, monkeypatch):
    from nnslr_tools.panel import _run_of
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['train_decider'] == {
        'label': 'Train applicability', 'allowed': False, 'reason': 'scripts/run_experiment_b.sh is missing in this installation'}
    script = checkout / 'scripts' / 'run_experiment_b.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Experiment B =="\necho "run:          $1"\necho "== 3/5 Train variant crop =="\n'
                      'echo "experiment_b_crop epoch 2/20 train_loss 1.0 val_loss 2.0 6s"\n')
    script.chmod(0o755)
    assert panel._action_states(ready_payload(), [])['train_decider']['allowed'] is True
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_decider', None)
    done = finished(panel, job['id'])
    first = full_log(panel, job['id']).splitlines()[0].split()
    assert first[:2] == ['==', 'scripts/run_experiment_b.sh'] and re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-decider', first[2])
    assert (done['state'], done['run'], done['label']) == ('ok', first[2], 'Train applicability')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_decider') == first[2]


@pytest.mark.parametrize('action, options, status', [
    ('train_cutter', {'size': '1344x760'}, 400), ('train_cutter', {'size': '896x288'}, 400),
    ('train_cutter', {'size': '960x544 ; rm -rf /'}, 400), ('train_cutter', {'families': 'other'}, 400),
    ('train_cutter', {'families': 'all ; rm -rf /'}, 400), ('train_cutter', {'start': 'random'}, 400),
    ('train_cutter', {'run': 'r1'}, 400), ('train', {'size': '960x544'}, 400), ('train_classifier', {'families': 'all'}, 400)])
def test_bad_options_of_the_cutter_training_are_refused(root, checkout, monkeypatch, action, options, status):
    signnet_script(checkout)
    classifier_script(checkout)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start(action, options)
    assert refused.value.status == status and panel.jobs() == []


def test_the_cutter_training_needs_its_script_and_its_run_is_new(root, checkout, monkeypatch):
    import nnslr_tools.panel as panel_module
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as missing:
        panel.start('train_cutter')
    assert (missing.value.status, missing.value.message) == (403, 'scripts/run_signnet.sh is missing in this installation.')

    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 24, 18, 0)
    monkeypatch.setattr(panel_module, 'datetime', Fixed)
    (root / 'runs' / '2026-10-24_1800-cutter').mkdir(parents=True)
    assert panel.new_run_name('cutter') == '2026-10-24_1800-cutter-2'


def test_a_running_cutter_training_is_seen_and_stopped_after_a_restart(root, checkout, monkeypatch):
    monkeypatch.setenv('FAKE_SIGNNET_SLEEP', '30')
    signnet_script(checkout)
    panel = Panel(root, repo=checkout)
    job = panel.start('train_cutter', {'families': 'speed'})
    assert any(item['label'] == 'cutter training' for item in panel.activities())
    with pytest.raises(PanelError) as busy:
        panel.start('train')
    assert (busy.value.status, busy.value.reason) == (409, 'busy')
    # The wrapper has the run after the command: a new server still knows it as this action's.
    restarted = Panel(root, repo=checkout)
    assert [item['stoppable'] for item in restarted.jobs() if item['id'] == job['id']] == [True]
    assert restarted.stop(job['id']) == {'id': job['id'], 'stopping': True}
    assert finished(restarted, job['id'])['state'] in ('interrupted', 'failed')


def test_the_cutter_options_over_http(root, checkout):
    from review_helpers import RunningServer, json_of
    signnet_script(checkout)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = lambda: []
    try:
        status, response, data = server.request('POST', '/api/panel/run/train_cutter', body={'families': 'x'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, response, data = server.request('POST', '/api/panel/run/train_cutter', body={'size': '1344x760', 'families': 'speed'})
        assert status == 400 and 'band, 896 × 288' in json_of(data)['message']
        # The Mapillary switch reaches the panel: refused here, the windows are not cut.
        status, response, data = server.request('POST', '/api/panel/run/train_cutter', body={'mapillary': 'on'})
        assert status == 409 and json_of(data)['message'] == (
            'The Mapillary windows are not cut yet (helpers/mapillary_windows.sh).')
        status, response, data = server.request('POST', '/api/panel/run/train_cutter', body={'families': 'speed'})
        assert status == 200, data
        job_id = json_of(data)['id']
        finished(server.app.panel, job_id)
        assert ' families=speed\n' in full_log(server.app.panel, job_id)
    finally:
        server.close()


def test_a_cutter_training_started_from_a_terminal_has_the_progress_of_its_signnet_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / 'k1' / 'signnet.log'
    log.parent.mkdir(parents=True)
    log.write_text('== scripts/run_signnet.sh k1\n== 1/3 Check the training population (CPU-only) ==\n'
                   '== 2/3 Explicit CUDA smoke test ==\n== 3/3 Train SignNet ==\n'
                   'signnet epoch 4/40 train_loss 2.1000 val_recall 0.3000 val_false_per_frame 4.80 80s\n')
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'train-r1.log').write_text('== a. importo\n== 5/10 Train reader ==\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'cutter training', 'pid': 1}]}
    panel._terminal_training(payload)
    progress = payload['activities'][0]['progress']
    assert [(step['key'], step['state']) for step in progress['steps']] == [('checks', 'done'), ('signnet', 'current'), ('evaluate', 'todo')]
    assert progress['detail'] == 'SignNet · epoch 4 of 40' and progress['curves']['signnet']['points'][0]['val_false_per_frame'] == 4.8
    # nnslr train-signnet by hand: its train.log beside its weights, read from its epochs alone (the newest log).
    os.utime(log, (time.time() - 60, time.time() - 60))
    hand = root / 'runs' / 'k2'
    hand.mkdir()
    (hand / 'signnet-best.pt').write_bytes(b'weights')
    (hand / 'train.log').write_text('signnet epoch 7/40 train_loss 1.9000 val_recall 0.4000 val_false_per_frame 4.70 81s\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'cutter training', 'pid': 1}]}
    panel._terminal_training(payload)
    assert payload['activities'][0]['progress']['detail'] == 'SignNet · epoch 7 of 40'


def test_an_old_cutter_training_started_from_a_terminal_has_the_progress_of_its_size_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / 'k1' / 'detector-sizes' / '960x544.log'
    log.parent.mkdir(parents=True)
    log.write_text('detector epoch 4/20 train_loss 1.1000 val_loss 1.3000 val_recall 0.5000 80s\n')
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs' / 'train-r1.log').write_text('== a. importo\n== 5/10 Train reader ==\n')
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'cutter training', 'pid': 1}]}
    panel._terminal_training(payload)
    progress = payload['activities'][0]['progress']
    assert [(step['key'], step['state']) for step in progress['steps']] == [('checks', 'done'), ('detector', 'current'), ('evaluate', 'todo')]
    assert progress['detail'] == 'detector · epoch 4 of 20' and progress['curves']['detector']['points'][0]['val_recall'] == 0.5


def test_the_networks_are_packaged_then_published_from_the_models_tab(root, checkout, monkeypatch):
    """Publish (Models tab): Prepare the package only with Our networks whole, Publish only a package prepared here."""
    import json as json_module
    from nnslr_tools import networks_package
    panel = Panel(root, repo=checkout)
    states = panel._action_states(ready_payload(), [])
    assert states['networks_prepare']['reason'] == 'scripts/publish_networks.sh is missing in this installation'
    script = checkout / 'scripts' / 'publish_networks.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Package of the networks"\necho "args: $*"\n'
                      'echo "== 1/4 Export the cutter (c)"\n')
    script.chmod(0o755)
    payload = {**ready_payload(), 'proposers': {'ours': 'Our networks propose only with all three parts in use'},
               'networks': {'packages': []}}
    states = panel._action_states(payload, [])
    assert states['networks_prepare']['reason'] == 'Our networks propose only with all three parts in use'
    assert states['networks_publish']['reason'] == 'no package prepared yet'
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start('networks_prepare', None)
    assert refused.value.reason == 'not_available'
    monkeypatch.setattr('nnslr_tools.preannotate.ours_missing', lambda root: None)
    job = panel.start('networks_prepare', None)
    done = finished(panel, job['id'])
    log = full_log(panel, job['id'])
    assert done['state'] == 'ok' and re.search(r'args: --name=speedlimits-\d{8}-\d{4}$', log, re.M)
    assert panel.job_progress(done, None)['steps'][0]['key'] == 'cutter'
    # Publish: only a package that is here, by its name; never another option.
    with pytest.raises(PanelError) as refused:
        panel.start('networks_publish', {'package': 'nothing-here'})
    assert refused.value.reason == 'bad_option'
    with pytest.raises(PanelError):
        panel.start('networks_prepare', {'package': 'x'})
    folder = root / 'publish' / 'networks' / 'pkg-1'
    folder.mkdir(parents=True)
    (folder / 'networks.json').write_text(json_module.dumps({'created': '2026-10-25T22:00:00Z', 'networks': {}}))
    assert [item['name'] for item in networks_package.packages(root)] == ['pkg-1']
    job = panel.start('networks_publish', {'package': 'pkg-1'})
    finished(panel, job['id'])
    assert re.search(r'args: --upload --name=pkg-1$', full_log(panel, job['id']), re.M)


def test_the_classifier_takes_the_mapillary_signs_only_when_prepared(root, checkout, monkeypatch):
    from nnslr_tools.mapillary import CROPS_FILE
    classifier_script(checkout, FAKE_CLASSIFIER.replace('echo "args $*"', 'echo "args $* mapillary=${NNSLR_CLASSIFIER_MAPILLARY:-0}"'))
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    cutter_in_use(root, panel)
    assert panel.status()['classifier_options']['mapillary']['available'] is False
    with pytest.raises(PanelError) as missing:
        panel.start('train_classifier', {'mapillary': 'on'})
    assert (missing.value.status, missing.value.message) == (409, 'Mapillary is not prepared (helpers/mapillary.sh).')
    for action, options in (('train_classifier', {'mapillary': 'yes'}), ('train_decider', {'mapillary': 'off'})):
        with pytest.raises(PanelError) as refused:
            panel.start(action, options)
        assert refused.value.status == 400
    (root / CROPS_FILE).parent.mkdir(parents=True)
    (root / CROPS_FILE).write_text(json.dumps({'kind': 'mapillary_crops', 'version': 1, 'crops': 2,
                                               'by_label': {'maximum_speed:50': 2}}) + '\n')
    assert panel.status()['classifier_options']['mapillary'] == {'available': True, 'crops': 2, 'labels': 1, 'reason': ''}
    job = panel.start('train_classifier', {'mapillary': 'on'})
    finished(panel, job['id'])
    assert 'mapillary=1\n' in full_log(panel, job['id'])
    time.sleep(1.1)  # a job's log is named by its second
    job = panel.start('train_classifier', {'mapillary': 'off'})
    finished(panel, job['id'])
    assert 'mapillary=0\n' in full_log(panel, job['id'])
    # Two stages: the Mapillary crops alone, then ours from those weights.
    time.sleep(1.1)
    job = panel.start('train_classifier', {'mapillary': 'pretrain', 'share': '50'})
    finished(panel, job['id'])
    assert 'mapillary=pretrain\n' in full_log(panel, job['id'])


def test_the_classifier_takes_a_region_and_a_share_of_mapillary(root, checkout, monkeypatch):
    from nnslr_tools.mapillary import CROPS_FILE, LOCATIONS_FILE
    classifier_script(checkout, FAKE_CLASSIFIER.replace(
        'echo "args $*"', 'echo "args $* region=${NNSLR_CLASSIFIER_MAPILLARY_REGION:-} '
                          'share=${NNSLR_CLASSIFIER_MAPILLARY_SHARE:-}"'))
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    cutter_in_use(root, panel)
    (root / CROPS_FILE).parent.mkdir(parents=True)
    (root / CROPS_FILE).write_text('\n'.join(json.dumps(line) for line in [
        {'kind': 'mapillary_crops', 'version': 1, 'crops': 2, 'by_label': {'maximum_speed:50': 2}},
        {'image_path': 'c/1.jpg', 'label': 'maximum_speed:50', 'mtsd_image': 'zurich', 'mtsd_object': 'a'},
        {'image_path': 'c/2.jpg', 'label': 'maximum_speed:50', 'mtsd_image': 'boston', 'mtsd_object': 'b'}]) + '\n')
    regions = panel.status()['classifier_options']['mapillary_regions']
    assert (regions['fetched'], regions['crops'], regions['continents']) == (False, 2, [])
    # A region needs the positions; a share alone does not.
    with pytest.raises(PanelError) as missing:
        panel.start('train_classifier', {'mapillary': 'on', 'region': 'europe'})
    assert missing.value.status == 409 and 'not fetched' in missing.value.message
    for options in ({'mapillary': 'on', 'region': 'Atlantis'}, {'mapillary': 'on', 'share': '0'},
                    {'mapillary': 'off', 'share': '5'}, {'mapillary': 'on', 'share': 5}):
        with pytest.raises(PanelError) as refused:
            panel.start('train_classifier', options)
        assert refused.value.status == 400, options
    with pytest.raises(PanelError) as other:
        panel.start('train_decider', {'region': 'europe'})
    assert other.value.status == 400
    job = panel.start('train_classifier', {'mapillary': 'on', 'region': 'all', 'share': '5'})
    finished(panel, job['id'])
    assert re.search(r'region= share=5$', full_log(panel, job['id']), re.M)
    (root / LOCATIONS_FILE).write_text(''.join(json.dumps(line) + '\n' for line in [
        {'mtsd_image': 'zurich', 'latitude': 47.4, 'longitude': 8.5, 'country': 'CH', 'continent': 'europe',
         'status': 'located'},
        {'mtsd_image': 'boston', 'latitude': 42.4, 'longitude': -71.1, 'country': 'US', 'continent': 'north_america',
         'status': 'located'}]))
    regions = panel.status()['classifier_options']['mapillary_regions']
    assert [item['key'] for item in regions['continents']] == ['europe', 'north_america']
    assert {item['code'] for item in regions['countries']} == {'CH', 'US'}
    time.sleep(1.1)  # a job's log is named by its second
    job = panel.start('train_classifier', {'mapillary': 'on', 'region': 'ch,europe', 'share': '100'})
    finished(panel, job['id'])
    assert re.search(r'region=europe,CH share=$', full_log(panel, job['id']), re.M)


def test_the_mapillary_positions_need_the_script_and_a_token(root, checkout, monkeypatch):
    monkeypatch.delenv('MAPILLARY_TOKEN', raising=False)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    state = panel.status()['actions']['mapillary_locations']
    assert state['reason'] == 'helpers/mapillary_locations.sh is missing in this installation'
    with pytest.raises(PanelError) as missing:
        panel.start('mapillary_locations')
    assert missing.value.status == 403
    script = checkout / 'helpers' / 'mapillary_locations.sh'
    script.write_text('#!/usr/bin/env bash\necho "== Mapillary: where the photos were taken"\nsleep "${FAKE_SLEEP:-0}"\n')
    script.chmod(0o755)
    assert panel.status()['actions']['mapillary_locations']['reason'] == (
        "no Mapillary token: add MAPILLARY_TOKEN to the server's .env")
    with pytest.raises(PanelError) as no_token:
        panel.start('mapillary_locations')
    assert no_token.value.status == 409
    (checkout / '.env').write_text('MAPILLARY_TOKEN=MLY|1|x\n')
    # No confirmed frame and no GPU needed: the token is enough.
    assert panel.status()['actions']['mapillary_locations'] == {'label': 'Locate the Mapillary photos', 'allowed': True,
                                                                'reason': ''}
    assert panel.status()['classifier_options']['mapillary_regions']['token'] is True
    monkeypatch.setenv('FAKE_SLEEP', '30')
    job = panel.start('mapillary_locations')
    try:
        status = panel.status()
        assert status['actions']['update']['reason'] == 'A search for the Mapillary positions is running'
        assert status['actions']['mapillary_locations']['reason'] == 'A search for the Mapillary positions is running'
        assert status['actions']['prepare']['reason'] != 'A search for the Mapillary positions is running'
    finally:
        panel.stop(job['id'])
        finished(panel, job['id'])


FAKE_CUTTER_MAPILLARY = FAKE_SIGNNET.replace(
    'families=${NNSLR_CUTTER_FAMILIES:-}"', 'mapillary=${NNSLR_CUTTER_MAPILLARY:-0} region=${NNSLR_CUTTER_MAPILLARY_REGION:-} '
    'share=${NNSLR_CUTTER_MAPILLARY_SHARE:-} ratio=${NNSLR_CUTTER_MAPILLARY_RATIO:-} '
    'pretrain_epochs=${NNSLR_CUTTER_PRETRAIN_EPOCHS:-}"')


def cut_windows(root, photos=('zurich', 'boston')):
    """A windows file of nnslr mapillary-windows: one train window of each photo, with a 50 and a give way."""
    from nnslr_tools.mtsd_windows import WINDOWS_FILE, WINDOWS_KIND, WINDOW_VERSION
    path = root / WINDOWS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {'kind': WINDOWS_KIND, 'version': WINDOW_VERSION, 'counts': {'train': {
        'windows': len(photos), 'photos': len(photos), 'boxes_by_family': {'maximum_speed': len(photos), 'other_sign': 3}}}}
    rows = [{'split': 'train', 'image_key': photo, 'objects': [{'status': 'clean', 'family': 'maximum_speed'},
                                                               {'status': 'clean', 'family': 'other_sign'}]}
            for photo in photos]
    path.write_text(''.join(json.dumps(line) + '\n' for line in [header, *rows]))


def cutter_run(panel, options):
    """The fields the fake run_signnet.sh echoes of a cutter training with these options."""
    job = panel.start('train_cutter', options)
    finished(panel, job['id'])
    line = next(line for line in full_log(panel, job['id']).splitlines() if line.startswith('args '))
    time.sleep(1.1)  # a job's log is named by its second
    return dict(field.split('=', 1) for field in line.split()[2:])


def test_the_cutter_takes_the_mapillary_windows_only_when_cut(root, checkout, monkeypatch):
    from nnslr_tools import train_settings
    signnet_script(checkout, FAKE_CUTTER_MAPILLARY)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    for mapillary in ('on', 'pretrain'):
        with pytest.raises(PanelError) as missing:
            panel.start('train_cutter', {'mapillary': mapillary})
        assert (missing.value.status, missing.value.message) == (
            409, 'The Mapillary windows are not cut yet (helpers/mapillary_windows.sh).')
    with pytest.raises(PanelError) as refused:
        panel.start('train_cutter', {'mapillary': 'yes'})
    assert refused.value.status == 400 and panel.jobs() == []
    cut_windows(root)
    # The speed signs only: the other signs of the windows are not what the switch adds.
    assert panel.status()['cutter_options']['mapillary'] == {
        'available': True, 'windows': 2, 'photos': 2, 'boxes_by_family': {'maximum_speed': 2}, 'reason': '',
        'licence': 'CC BY-NC-SA 4.0'}
    assert cutter_run(panel, {'mapillary': 'off'})['mapillary'] == '0'
    assert cutter_run(panel, {'mapillary': 'on'})['mapillary'] == '1'
    assert cutter_run(panel, {'mapillary': 'pretrain'})['mapillary'] == 'pretrain'
    # Train › Cutter › Settings: the cap goes with the windows added to ours only, the first stage's epochs always.
    train_settings.save(root, {'training': 'train_cutter', 'values': {'mapillary_ratio': 2.5, 'pretrain_epochs': 5}},
                        'test')
    on = cutter_run(panel, {'mapillary': 'on'})
    assert (on['ratio'], on['pretrain_epochs']) == ('2.5', '5')
    for mapillary in ('pretrain', 'off'):
        assert cutter_run(panel, {'mapillary': mapillary})['ratio'] == ''


def test_the_cutter_takes_a_region_and_a_share_of_the_mapillary_windows(root, checkout, monkeypatch):
    from nnslr_tools.mapillary import LOCATIONS_FILE
    signnet_script(checkout, FAKE_CUTTER_MAPILLARY)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    cut_windows(root)
    regions = panel.status()['cutter_options']['mapillary_regions']
    assert (regions['fetched'], regions['crops'], regions['photos'], regions['continents']) == (False, 2, 2, [])
    with pytest.raises(PanelError) as missing:
        panel.start('train_cutter', {'mapillary': 'on', 'region': 'europe'})
    assert missing.value.status == 409 and 'not fetched' in missing.value.message
    for options in ({'mapillary': 'off', 'share': '5'}, {'mapillary': 'on', 'share': '0'},
                    {'mapillary': 'on', 'region': 'Atlantis'}):
        with pytest.raises(PanelError) as refused:
            panel.start('train_cutter', options)
        assert refused.value.status == 400, options
    (root / LOCATIONS_FILE).write_text(''.join(json.dumps(line) + '\n' for line in [
        {'mtsd_image': 'zurich', 'latitude': 47.4, 'longitude': 8.5, 'country': 'CH', 'continent': 'europe',
         'status': 'located'},
        {'mtsd_image': 'boston', 'latitude': 42.4, 'longitude': -71.1, 'country': 'US', 'continent': 'north_america',
         'status': 'located'}]))
    # The speed signs of the windows by the place of their photo.
    regions = panel.status()['cutter_options']['mapillary_regions']
    assert (regions['fetched'], regions['located_crops'], regions['located_photos']) == (True, 2, 2)
    assert [(item['key'], item['crops']) for item in regions['continents']] == [('europe', 1), ('north_america', 1)]
    assert {item['code'] for item in regions['countries']} == {'CH', 'US'}
    fields = cutter_run(panel, {'mapillary': 'pretrain', 'region': 'ch,europe', 'share': '50'})
    assert (fields['mapillary'], fields['region'], fields['share']) == ('pretrain', 'europe,CH', '50')
    fields = cutter_run(panel, {'mapillary': 'on', 'region': 'all', 'share': '100'})
    assert (fields['region'], fields['share']) == ('', '')


def test_the_mapillary_windows_are_cut_from_the_panel(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    assert panel.status()['actions']['mapillary_windows'] == {
        'label': 'Cut the Mapillary windows', 'allowed': False,
        'reason': 'helpers/mapillary_windows.sh is missing in this installation'}
    with pytest.raises(PanelError) as missing:
        panel.start('mapillary_windows')
    assert missing.value.status == 403
    script = checkout / 'helpers' / 'mapillary_windows.sh'
    script.write_text('#!/usr/bin/env bash\necho "== Mapillary: the windows of the cutter"\nsleep "${FAKE_SLEEP:-0}"\n')
    script.chmod(0o755)
    # The CPU and the share only: no confirmed frame, no GPU, no token.
    assert panel.status()['actions']['mapillary_windows'] == {'label': 'Cut the Mapillary windows', 'allowed': True,
                                                              'reason': ''}
    with pytest.raises(PanelError) as other:
        panel.start('mapillary_windows', {'mapillary': 'on'})
    assert other.value.status == 400
    monkeypatch.setenv('FAKE_SLEEP', '30')
    job = panel.start('mapillary_windows')
    try:
        status = panel.status()
        for action in ('update', 'mapillary_windows', 'mapillary_locations'):
            assert status['actions'][action]['reason'] == 'The cut of the Mapillary windows is running', action
        assert status['actions']['prepare']['reason'] != 'The cut of the Mapillary windows is running'
    finally:
        panel.stop(job['id'])
        finished(panel, job['id'])


def test_a_classifier_job_shows_its_mapillary_region_and_share(tmp_path):
    from nnslr_tools.panel import _settings_of
    log = tmp_path / 'job.log'
    log.write_text('== scripts/run_classifier.sh r\nrun:          r\nsettings:     --classes v2 --epochs 30 --mapillary '
                   '/d/crops.jsonl --mapillary-region europe,CH --mapillary-share 5\n')
    assert _settings_of(log, 'train_classifier') == ['classes v2', 'Mapillary signs', 'Mapillary Europe, CH',
                                                     'Mapillary 5 %', '30 epochs']


def test_a_classifier_job_in_two_stages_shows_them(tmp_path):
    from nnslr_tools.panel import _settings_of
    log = tmp_path / 'job.log'
    log.write_text('== scripts/run_classifier.sh r\nrun:          r\npretraining:  --classes v2 --epochs 20 --pretrained '
                   '--mapillary /d/crops.jsonl --mapillary-share 5 --mapillary-only\nsettings:     --classes v2 --epochs 30 '
                   '--init-from /d/runs/r/pretrain/reader-best.pt\n')
    assert _settings_of(log, 'train_classifier') == ['classes v2', 'Mapillary first, then ours', 'Mapillary 5 %',
                                                     'Mapillary 20 epochs', '30 epochs', 'pretrained']
    log.write_text('== scripts/run_classifier.sh r\nrun:          r\nsettings:     --classes v2 --epochs 30 '
                   '--init-from /d/runs/2026-10-29_1602-classifier/classifier/reader-best.pt\n')
    assert _settings_of(log, 'train_classifier') == ['classes v2', '30 epochs', 'from 2026-10-29_1602-classifier']


def test_the_classifier_can_start_from_a_finished_classifier_of_models(root, checkout, monkeypatch):
    from nnslr_tools.mapillary import CROPS_FILE
    from test_panel_runs import SPLIT_SHA, make_classifier_run, write
    classifier_script(checkout, FAKE_CLASSIFIER.replace(
        'echo "args $*"', 'echo "args $* init=${NNSLR_CLASSIFIER_INIT_FROM:-imagenet}"'))
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    cutter_in_use(root, panel)
    done = make_classifier_run(root, 'c1')
    make_classifier_run(root, 'c2', upto='classifier')
    write(root / 'runs/r1/reader/reader-best.pt', 'weights')
    (root / CROPS_FILE).parent.mkdir(parents=True)
    (root / CROPS_FILE).write_text(json.dumps({'kind': 'mapillary_crops', 'version': 1, 'crops': 2,
                                               'by_label': {'maximum_speed:50': 2}}) + '\n')
    # The split the next one learns on, for the page's warning; each classifier says the split it learnt on.
    assert panel.status()['classifier_options']['split_sha256'] == SPLIT_SHA
    (root / 'splits/latest.json').unlink()
    assert panel.status()['classifier_options']['split_sha256'] is None
    write(root / 'splits/latest.json', {'path': 'splits/' + 'f' * 64 + '.json', 'sha256': 'f' * 64})
    assert panel.status()['classifier_options']['split_sha256'] == 'f' * 64
    versions = {entry['id']: entry for entry in panel.models_index.artifacts(('classifier',))['classifier']}
    assert versions['classifier:c1/classifier']['split_sha256'] == SPLIT_SHA
    # ImageNet, as before: no start in the environment.
    for options in (None, {'start': 'imagenet'}):
        job = panel.start('train_classifier', options)
        finished(panel, job['id'])
        assert re.search(r'init=imagenet$', full_log(panel, job['id']), re.M)
        time.sleep(1.1)  # a job's log is named by its second
    # A finished classifier with "not a sign", by its id: its reader-best.pt, also with the Mapillary signs in train.
    job = panel.start('train_classifier', {'start': 'classifier:c1/classifier', 'mapillary': 'on'})
    finished(panel, job['id'])
    assert re.search(r'init=' + re.escape(str((done / 'classifier/reader-best.pt').resolve())) + '$',
                     full_log(panel, job['id']), re.M)
    refused = {'classifier:nothing/classifier': 'Unknown classifier to start from: choose one of Models.',
               'classifier:c2/classifier': 'That classifier has not finished its training: choose a finished one.',
               'classifier:r1/reader': 'An old reader cannot be a start: only a classifier with “not a sign”.',
               '/etc/passwd': 'Unknown classifier to start from: choose one of Models.',
               'random': 'Unknown classifier to start from: choose one of Models.'}
    for start, message in refused.items():
        with pytest.raises(PanelError) as error:
            panel.start('train_classifier', {'start': start})
        assert (error.value.status, error.value.message) == (400, message), start
    # Two stages start from their own Mapillary stage: no other start.
    with pytest.raises(PanelError) as error:
        panel.start('train_classifier', {'start': 'classifier:c1/classifier', 'mapillary': 'pretrain'})
    assert error.value.status == 400 and 'its own Mapillary stage' in error.value.message
    # Weights gone since the list was read.
    (done / 'classifier/reader-best.pt').unlink()
    with pytest.raises(PanelError) as error:
        panel.start('train_classifier', {'start': 'classifier:c1/classifier'})
    assert error.value.status == 400
    # The start of the other actions stays theirs.
    with pytest.raises(PanelError) as error:
        panel.start('train_cutter', {'start': 'classifier:c1/classifier'})
    assert (error.value.status, error.value.message) == (400, 'Unknown network to start from: choose one of Models.')
    with pytest.raises(PanelError) as error:
        panel.start('prepare', {'start': 'classifier:c1/classifier'})
    assert (error.value.status, error.value.message) == (400, 'Only a training has a starting point.')


def test_the_cutter_and_applicability_go_on_from_a_finished_version_of_models(root, checkout, monkeypatch):
    from test_panel_models import experiment, make_signnet
    from test_panel_runs import SPLIT_SHA, experiment_result, write
    for name, variable in (('run_signnet.sh', 'NNSLR_CUTTER_INIT_FROM'), ('run_experiment_b.sh', 'NNSLR_DECIDER_INIT_FROM')):
        script = checkout / 'scripts' / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f'#!/usr/bin/env bash\necho "run:          $1"\necho "init=${{{variable}:-fresh}}"\n')
        script.chmod(0o755)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    make_signnet(root, 's1')
    make_signnet(root, 's2', band=[1344, 416], stride=16)
    make_signnet(root, 's3', finished=False)
    experiment(root, 'expB')
    write(root / 'runs/expS/experiment-b/crop/result.json', {**experiment_result('crop', 0.9, 0.6), 'limit': 5})
    write(root / 'runs/expS/experiment-b/crop/crop-best.pt', 'weights')

    def init_of(action, options):
        job = panel.start(action, options)
        finished(panel, job['id'])
        time.sleep(1.1)  # a job's log is named by its second
        return re.search(r'^init=(.*)$', full_log(panel, job['id']), re.M).group(1)

    # The usual start, as before: nothing in the environment.
    assert init_of('train_cutter', None) == init_of('train_decider', {'start': 'fresh'}) == 'fresh'
    # A finished version of Models, by its id: a SignNet's weights, the run's folder of applicability.
    assert init_of('train_cutter', {'families': 'speed', 'start': 'cutter:s1/signnet'}).endswith('/runs/s1/signnet-best.pt')
    assert init_of('train_decider', {'start': 'decider:expB/experiment-b/crop_geom'}).endswith('/runs/expB/experiment-b')
    refused = {('train_cutter', 'cutter:s2/signnet'): 'Only a SignNet of the comma’s band and grid can be a start.',
               ('train_cutter', 'cutter:s3/signnet'): 'That network has not finished its training: choose a finished one.',
               ('train_cutter', '/etc/passwd'): 'Unknown network to start from: choose one of Models.',
               ('train_cutter', 'imagenet'): 'Unknown network to start from: choose one of Models.',
               ('train_decider', 'decider:expS/experiment-b/crop'): 'A smoke run learnt on a few frames: choose a whole training.',
               ('train_decider', 'cutter:s1/signnet'): 'Unknown network to start from: choose one of Models.'}
    for (action, start), message in refused.items():
        with pytest.raises(PanelError) as error:
            panel.start(action, {'start': start})
        assert (error.value.status, error.value.message) == (400, message), start
    # The versions say the split they learnt on and what they went on from: the page warns when it is another split.
    write(root / 'runs/s4/result.json', {**json.loads((root / 'runs/s1/result.json').read_text()), 'split_sha256': SPLIT_SHA,
                                         'start': {'path': '/d/runs/s1/signnet-best.pt', 'run': 's1', 'epoch': 27,
                                                   'split_sha256': 'f' * 64}})
    write(root / 'runs/s4/signnet-best.pt', 'weights')
    found = {entry['id']: entry for role in ('cutter', 'decider')
             for entry in panel.models_index.artifacts((role,))[role]}
    s4 = found['cutter:s4/signnet']
    assert (s4['split_sha256'], s4['initialization'], s4['start']) == (SPLIT_SHA, 'start', {'run': 's1', 'epoch': 27,
                                                                                            'another_split': True})
    assert (found['cutter:s1/signnet']['start'], found['cutter:s1/signnet']['initialization']) == (None, None)
    assert found['decider:expB/experiment-b/crop']['split_sha256'] == SPLIT_SHA


def test_a_job_says_the_run_its_network_went_on_from(tmp_path):
    from nnslr_tools.panel import _settings_of
    log = tmp_path / 'job.log'
    log.write_text('== scripts/run_signnet.sh r\nrun:          r\nband:         896x288\nsettings:     --families all '
                   '--epochs 40 --init-from /d/runs/2026-10-30_1200-cutter/signnet-best.pt\n')
    assert _settings_of(log, 'train_cutter')[-1] == 'from 2026-10-30_1200-cutter'
    log.write_text('== scripts/run_experiment_b.sh r\nrun:          r\nepochs:       20\ninit:         2026-10-30_1300-decider\n')
    assert _settings_of(log, 'train_decider') == ['20 epochs', 'from 2026-10-30_1300-decider']
