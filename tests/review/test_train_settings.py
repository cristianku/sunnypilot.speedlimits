"""The settings of the trainings of Train (nnslr_tools.train_settings): kept in the data root, shown and saved in
Train › <training> › Settings (/api/panel/train-settings), set by Panel.start as the variables of each script, the
same defaults as the scripts; the job's chips say the values a training used."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from nnslr_tools import train_settings
from nnslr_tools.panel import Panel, PanelError, _settings_of
from test_panel import checkout, finished, full_log, root  # noqa: F401  (fixtures and helpers)
from test_panel_training import FAKE_CLASSIFIER, FAKE_SIGNNET, classifier_script, cutter_in_use, signnet_script

REPO = Path(__file__).resolve().parents[2]


def test_the_defaults_until_one_is_saved(tmp_path):
    view = train_settings.view(tmp_path)['trainings']
    assert list(view) == ['train_cutter', 'train_classifier', 'train_decider', 'train_road_type', 'train_brain']
    classifier = {item['key']: item for item in view['train_classifier']['settings']}
    assert {key: item['value'] for key, item in classifier.items()} == {
        'epochs': 30, 'pretrain_epochs': 30, 'patience': 5, 'batch': 64, 'workers': train_settings.DEFAULT_WORKERS,
        'negative_score': 0.1, 'negatives_per_frame': 3, 'negatives_factor': 3.0}
    assert 'env' not in classifier['epochs'] and 'same_as' not in classifier['pretrain_epochs']
    assert view['train_classifier']['by'] is None and view['train_classifier']['fixed'] == []
    # What the car needs is shown, never a setting.
    assert [item['label'] for item in view['train_cutter']['fixed']] == ['Band', 'Grid', 'Channels', 'Fuse']
    assert [item['key'] for item in view['train_cutter']['settings']] == ['epochs', 'pretrain_epochs', 'mapillary_ratio',
                                                                          'batch', 'learning_rate', 'workers']
    for action in train_settings.TRAININGS:
        assert train_settings.environment(tmp_path, action) == {}
    assert train_settings.environment(tmp_path, 'train') == {}


def test_only_the_changed_ones_are_kept_and_null_is_the_default_again(tmp_path):
    record = train_settings.save(tmp_path, {'training': 'train_classifier', 'values': {
        'epochs': 40.0, 'patience': 5, 'negative_score': 0.2}}, 'cristian')
    assert record['values'] == {'epochs': 40, 'negative_score': 0.2} and record['by'] == 'cristian'
    values = train_settings.values(tmp_path, 'train_classifier')
    assert isinstance(values['epochs'], int) and values['epochs'] == 40
    # The first stage follows the epochs while it is not set.
    assert values['pretrain_epochs'] == 40
    assert train_settings.environment(tmp_path, 'train_classifier') == {
        'NNSLR_CLASSIFIER_EPOCHS': '40', 'NNSLR_NEGATIVE_SCORE': '0.2'}
    # Another training is kept apart.
    train_settings.save(tmp_path, {'training': 'train_cutter', 'values': {'learning_rate': 0.00001}}, 'cristian')
    assert train_settings.environment(tmp_path, 'train_cutter') == {'NNSLR_SIGNNET_LEARNING_RATE': '1e-05'}
    assert train_settings.values(tmp_path, 'train_classifier')['epochs'] == 40
    # A key left out keeps its value; null goes back to the default.
    train_settings.save(tmp_path, {'training': 'train_classifier', 'values': {'negative_score': None}}, 'cristian')
    assert train_settings.environment(tmp_path, 'train_classifier') == {'NNSLR_CLASSIFIER_EPOCHS': '40'}
    stored = json.loads((tmp_path / train_settings.SETTINGS_FILE).read_text())
    assert stored['trainings']['train_classifier']['values'] == {'epochs': 40}
    assert stored['trainings']['train_cutter']['values'] == {'learning_rate': 0.00001}


def test_the_first_stage_set_apart_from_the_epochs(tmp_path):
    train_settings.save(tmp_path, {'training': 'train_classifier', 'values': {'pretrain_epochs': 10}}, 'c')
    assert train_settings.environment(tmp_path, 'train_classifier') == {'NNSLR_CLASSIFIER_PRETRAIN_EPOCHS': '10'}
    # Set to the epochs' value: it follows them again.
    train_settings.save(tmp_path, {'training': 'train_classifier', 'values': {'pretrain_epochs': 30}}, 'c')
    train_settings.save(tmp_path, {'training': 'train_classifier', 'values': {'epochs': 50}}, 'c')
    assert train_settings.environment(tmp_path, 'train_classifier') == {'NNSLR_CLASSIFIER_EPOCHS': '50'}
    assert train_settings.values(tmp_path, 'train_classifier')['pretrain_epochs'] == 50
    view = train_settings.view(tmp_path)['trainings']['train_classifier']['settings']
    assert {item['key']: item['default'] for item in view}['pretrain_epochs'] == 50


@pytest.mark.parametrize('body', [
    {'training': 'train_classifier', 'values': {'epochs': 0}}, {'training': 'train_classifier', 'values': {'epochs': 201}},
    {'training': 'train_classifier', 'values': {'epochs': 2.5}}, {'training': 'train_classifier', 'values': {'epochs': '40'}},
    {'training': 'train_classifier', 'values': {'batch': True}}, {'training': 'train_cutter', 'values': {'learning_rate': 0.5}},
    {'training': 'train_cutter', 'values': {'learning_rate': float('nan')}},
    {'training': 'train_cutter', 'values': {'workers': 33}}, {'training': 'train_cutter', 'values': {'seed': 1}},
    {'training': 'train_cutter', 'values': {'band': 960}}, {'training': 'train', 'values': {}},
    {'training': 'train_road_type', 'values': {'patience': 0}}, {'values': {'epochs': 40}}, [],
])
def test_a_wrong_value_is_refused_and_nothing_is_written(tmp_path, body):
    with pytest.raises(train_settings.SettingsError):
        train_settings.save(tmp_path, body, 'cristian')
    assert not (tmp_path / train_settings.SETTINGS_FILE).exists()


def test_a_value_spoiled_in_the_file_is_the_default(tmp_path):
    path = tmp_path / train_settings.SETTINGS_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'trainings': {'train_road_type': {'values': {'epochs': 5000, 'batch': 8}}}}))
    assert train_settings.environment(tmp_path, 'train_road_type') == {'NNSLR_ROAD_TYPE_BATCH': '8'}
    path.write_text('not json')
    assert train_settings.environment(tmp_path, 'train_road_type') == {}


def _script_default(script: str, env: str) -> str:
    match = re.search(r'"\$\{' + env + r':-([^}]*)\}"', (REPO / script).read_text())
    assert match, f'{env} is not read by {script}'
    return match.group(1)


def test_the_same_defaults_as_the_scripts():
    for action, training in train_settings.TRAININGS.items():
        for setting in training['settings']:
            default = _script_default(training['script'], setting['env'])
            if setting['key'] == 'workers':
                assert default == '$WORKERS_DEFAULT'
            elif setting['key'] == 'sign_workers':
                assert default == '$SIGN_WORKERS_DEFAULT'
            elif setting.get('same_as'):
                assert default == '$' + setting['same_as'].upper()
            elif setting['key'] == 'mapillary_ratio':
                # Passed on only when set: nnslr train-signnet's own default (1) without it.
                assert default == '' and setting['default'] == 1.0
            else:
                assert float(default) == setting['default'], (action, setting['key'])


def test_start_sets_the_variables_of_the_changed_settings(root, checkout, monkeypatch):
    train_settings.save(root, {'training': 'train_cutter', 'values': {'epochs': 12, 'learning_rate': 0.001}}, 'c')
    train_settings.save(root, {'training': 'train_classifier', 'values': {'batch': 32, 'negatives_factor': 1.5}}, 'c')
    signnet_script(checkout, FAKE_SIGNNET.replace(
        'families=${NNSLR_CUTTER_FAMILIES:-}', 'epochs=${NNSLR_SIGNNET_EPOCHS:-} rate=${NNSLR_SIGNNET_LEARNING_RATE:-} '
        'batch=${NNSLR_SIGNNET_BATCH:-none} classifier=${NNSLR_CLASSIFIER_BATCH_SIZE:-none}'))
    classifier_script(checkout, FAKE_CLASSIFIER.replace(
        'echo "args $*"', 'echo "args $* batch=${NNSLR_CLASSIFIER_BATCH_SIZE:-} factor=${NNSLR_NEGATIVES_FACTOR:-} '
                          'epochs=${NNSLR_CLASSIFIER_EPOCHS:-none}"'))
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_cutter')
    finished(panel, job['id'])
    assert re.search(r'epochs=12 rate=0\.001 batch=none classifier=none$', full_log(panel, job['id']), re.M)
    cutter_in_use(root, panel)
    time.sleep(1.1)  # a job's log is named by its second
    job = panel.start('train_classifier')
    finished(panel, job['id'])
    assert re.search(r'batch=32 factor=1\.5 epochs=none$', full_log(panel, job['id']), re.M)


def test_a_read_only_panel_shows_them_and_refuses_to_save(root, checkout):
    panel = Panel(root, repo=checkout, actions_enabled=False)
    assert not panel.train_settings()['can_edit']
    with pytest.raises(PanelError) as refused:
        panel.save_train_settings({'training': 'train_cutter', 'values': {'epochs': 12}}, 'c')
    assert (refused.value.status, refused.value.reason) == (403, 'actions_disabled')
    assert not (root / train_settings.SETTINGS_FILE).exists()
    panel = Panel(root, repo=checkout)
    assert panel.train_settings()['can_edit']
    with pytest.raises(PanelError) as refused:
        panel.save_train_settings({'training': 'train_cutter', 'values': {'epochs': 0}}, 'c')
    assert (refused.value.status, refused.value.reason) == (400, 'invalid_settings')


def test_over_http(tmp_path):
    from review_helpers import RunningServer, json_of

    server = RunningServer(tmp_path)
    try:
        status, _, data = server.request('GET', '/api/panel/train-settings')
        view = json_of(data)
        assert status == 200 and set(view['trainings']) == set(train_settings.TRAININGS)
        body = {'training': 'train_road_type', 'values': {'epochs': 80}}
        status, _, data = server.request('POST', '/api/panel/train-settings', body=body)
        view = json_of(data)
        assert status == 200 and view['ok'] and train_settings.values(tmp_path, 'train_road_type')['epochs'] == 80
        assert {item['key']: item['value'] for item in view['trainings']['train_road_type']['settings']}['epochs'] == 80
        status, _, data = server.request('POST', '/api/panel/train-settings',
                                         body={'training': 'train_road_type', 'values': {'epochs': 900}})
        assert status == 400 and json_of(data)['error'] == 'invalid_settings'
        server.app.panel.actions_enabled = False
        status, _, data = server.request('POST', '/api/panel/train-settings',
                                         body={'training': 'train_road_type', 'values': {'epochs': None}})
        assert status == 403 and json_of(data)['error'] == 'actions_disabled'
        assert train_settings.values(tmp_path, 'train_road_type')['epochs'] == 80
        assert not json_of(server.request('GET', '/api/panel/train-settings')[2])['can_edit']
    finally:
        server.close()


def test_the_chips_say_the_values_used(tmp_path):
    log = tmp_path / 'job.log'
    log.write_text('== scripts/run_signnet.sh r\n== scripts/run_signnet.sh r\nrun:          r\nband:         896x288\n'
                   'settings:     --families all --band-width 896 --stride 8 --fuse-kernel 1 --widths 16,24,32,48 '
                   '--epochs 12 --batch 16 --learning-rate 1e-05\nworkers:      8\n')
    assert _settings_of(log, 'train_cutter') == ['896 × 288', 'every sign', 'stride 8', '12 epochs', 'batch 16',
                                                 'learning rate 1e-05', '8 workers']
    head = ('== scripts/run_classifier.sh r\n== Classifier (cutter -> classifier -> decider) ==\nrun:          r\n'
            'negatives:    cutter boxes of score >= {} overlapping no reviewed box, at most {} per frame and {} x the sign '
            'crops per partition\n')
    settings = 'settings:     --classes v2 --epochs 30 --patience 5 --batch-size 64 --num-workers 16 --pretrained\n'
    log.write_text(head.format('0.1', '3', '3') + settings)
    assert _settings_of(log, 'train_classifier') == ['classes v2', '30 epochs', 'patience 5', 'batch 64', '16 workers',
                                                     'pretrained']
    log.write_text(head.format('0.2', '3', '1.5') + settings)
    assert 'false boxes ≥ 0.2, 3 a frame, 1.5 × signs' in _settings_of(log, 'train_classifier')
