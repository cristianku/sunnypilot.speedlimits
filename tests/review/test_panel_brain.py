"""The brain in the panel: "Train the brain" (scripts/run_brain.sh) from Train › Brain, what it learns, its progress read from
its log (nnslr_tools.job_progress), what the labels give it, and which brains can be chosen for the car. Nothing is trained: the script is a stand-in, the log has the lines nnslr_tools.brain_train writes."""
from __future__ import annotations

import re
import sqlite3

import pytest

from nnslr_tools.job_progress import progress
from nnslr_tools.model_roles import selectable
from nnslr_tools.panel import Panel, PanelError, _run_of, _settings_of
from nnslr_tools.speed_review import db_path
from review_helpers import RunningServer, json_of
from test_panel import checkout, finished, full_log, root  # noqa: F401  (fixtures and helpers)

# The lines of scripts/run_brain.sh and of brain_train.py: the dry run's epoch is not a seed's.
LOG = """== Brain ==
run:          2026-11-05_1500-brain
output:       /data/runs/2026-11-05_1500-brain-s{0,1,2}
settings:     --epochs 40 --pool mean
== 1/6 Check the labels, the split and the training (CPU-only) ==
dry_run epoch 1/1 train_loss 2.8000 val_error_kph - val_accuracy - on 0 frames (always 120: 31.2, the sign: 12.4), says [] 5s
== 2/6 Explicit CUDA smoke test ==
== 3/6 Read the signs of the labelled drives as the car does today ==
== 4/6 Train seed 0 ==
brain_s0 epoch 1/40 train_loss 2.5000 val_error_kph 11.32 val_accuracy 0.4120 on 357 frames (always 120: 31.2, the sign: 12.4), says [80, 120] 4s
brain_s0 epoch 2/40 train_loss 2.1000 val_error_kph 9.75 val_accuracy 0.5012 on 357 frames (always 120: 31.2, the sign: 12.4), says [50, 80, 120] 4s
"""


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes}}}


def brain_script(checkout):
    script = checkout / 'scripts' / 'run_brain.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Brain =="\necho "run:          $1"\n'
                      'echo "choices: $NNSLR_BRAIN_WORKS $NNSLR_BRAIN_DRIVER $NNSLR_BRAIN_POOL"\n')
    script.chmod(0o755)
    return script


def speed_marked(root):
    path = db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE IF NOT EXISTS speed_stretches (route TEXT)')
    connection.execute("INSERT INTO speed_stretches VALUES ('r1')")
    connection.commit()
    connection.close()


# -- its progress -------------------------------------------------------------------------------------------------
def test_the_progress_follows_the_steps_and_the_seeds_not_the_dry_run():
    result = progress('train_brain', 'running', LOG)
    assert [(step['key'], step['state']) for step in result['steps']] == [
        ('checks', 'done'), ('signs', 'done'), ('brain_s0', 'current'), ('brain_s1', 'todo'), ('brain_s2', 'todo')]
    assert result['detail'] == 'Seed 0 · epoch 2 of 40'
    # 6 + 22 done, then 2 of the 40 epochs of the 24 of the seed.
    assert result['percent'] == pytest.approx(29.2)
    assert list(result['curves']) == ['brain_s0']
    points = result['curves']['brain_s0']['points']
    assert [(point['epoch'], point['val_error_kph'], point['val_accuracy']) for point in points] == [
        (1, 11.32, 0.412), (2, 9.75, 0.5012)]
    assert points[0]['train_loss'] == 2.5 and points[0]['seconds'] == 4


def test_the_seeds_have_a_curve_each_and_a_seed_without_a_measure_still_counts():
    log = LOG + ('== 5/6 Train seed 1 ==\n'
                 'brain_s1 epoch 1/40 train_loss 2.5000 val_error_kph - val_accuracy - on 0 frames '
                 '(always 120: 31.2, the sign: 12.4), says [] 4s\n')
    result = progress('train_brain', 'running', log)
    assert [step['state'] for step in result['steps']] == ['done', 'done', 'done', 'current', 'todo']
    assert list(result['curves']) == ['brain_s0', 'brain_s1']
    assert 'val_error_kph' not in result['curves']['brain_s1']['points'][0]


def test_a_seed_already_trained_is_a_step_done():
    log = LOG + '== 5/6 Train seed 1 ==\nskip seed 1: already trained\n== 6/6 Train seed 2 ==\n'
    assert [step['state'] for step in progress('train_brain', 'running', log)['steps']] == ['done', 'done', 'done', 'done', 'current']


# -- "Train the brain" --------------------------------------------------------------------------------------------
def test_the_brain_is_trained_from_the_train_tab_once_a_speed_is_marked(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['train_brain'] == {
        'label': 'Train the brain', 'allowed': False, 'reason': 'scripts/run_brain.sh is missing in this installation'}
    brain_script(checkout)
    assert panel._action_states(ready_payload(), [])['train_brain']['reason'] == 'no speeds marked yet: Data › Speed review'
    speed_marked(root)
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_brain']['reason'] == 'the GPU is busy with another process'
    # Its labels are the speeds of Speed review: no reviewed drive of signs needed.
    assert panel._action_states(ready_payload(), [])['train_brain']['allowed'] is True

    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_brain', None)
    done = finished(panel, job['id'])
    lines = full_log(panel, job['id']).splitlines()
    first = lines[0].split()
    assert first[:2] == ['==', 'scripts/run_brain.sh'] and re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-brain', first[2])
    assert (done['state'], done['run'], done['label']) == ('ok', first[2], 'Train the brain')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_brain') == first[2]
    assert 'choices: 1 1 grid' in lines


def test_what_it_learns_reaches_the_script_and_is_shown_on_the_job(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    speed_marked(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_brain', {'works': 'off', 'driver': 'off', 'pool': 'mean'})
    done = finished(panel, job['id'])
    assert 'choices: 0 0 mean' in full_log(panel, job['id']).splitlines()
    assert done['state'] == 'ok'
    # The job's chips, from the settings line of the script.
    log = root / 'brain.log'
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool mean --no-works --no-driver\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs', 'no roadworks', 'no driver’s inputs', 'picture as a mean']
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool grid\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs']


@pytest.mark.parametrize('options', [{'works': 'x'}, {'driver': 'maybe'}, {'pool': 'max'}, {'works': ['on']}, {'pool': {'a': 1}}])
def test_bad_choices_are_refused_before_it_starts(root, checkout, monkeypatch, options):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    speed_marked(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start('train_brain', options)
    assert refused.value.status == 400 and refused.value.reason == 'bad_option'
    assert panel.jobs() == []


@pytest.mark.parametrize('key,value', [('works', 'off'), ('driver', 'off'), ('pool', 'mean')])
def test_its_choices_are_refused_on_another_training(root, checkout, key, value):
    with pytest.raises(PanelError) as refused:
        Panel(root, repo=checkout).start('train_road_type', {key: value})
    assert refused.value.reason == 'bad_option'


def test_it_is_a_step_of_the_pipeline_with_its_choices(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    speed_marked(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {'pool': 'max'}}]}, None)
    assert refused.value.status == 400
    view = panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {'works': 'off', 'pool': 'mean'}}]}, None)
    assert view['steps'][0]['action'] == 'train_brain'
    job = next(item for item in panel.jobs() if item['action'] == 'train_brain')
    finished(panel, job['id'])
    assert 'choices: 0 1 mean' in full_log(panel, job['id']).splitlines()


def test_a_brain_training_started_from_a_terminal_has_the_progress_of_its_run_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / '2026-11-05_1500-brain' / 'brain.log'
    log.parent.mkdir(parents=True)
    log.write_text(LOG)
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'brain training', 'pid': 1}]}
    panel._terminal_training(payload)
    assert payload['activities'][0]['progress']['detail'] == 'Seed 0 · epoch 2 of 40'


# -- what can go to the car ---------------------------------------------------------------------------------------
def test_a_brain_can_be_chosen_for_the_car_with_or_without_the_drivers_inputs():
    usual = {'role': 'brain', 'finished': True, 'groups': ['signs', 'plate', 'hour', 'lead', 'country']}
    assert selectable(usual) == (True, None)
    assert selectable({**usual, 'groups': [*usual['groups'], 'pedals', 'speed']}) == (True, None)
    assert selectable({**usual, 'finished': False}) == (False, 'not_finished')


# -- the signs it reads -------------------------------------------------------------------------------------------
def test_road_type_signs_labelled_reads_the_drives_of_speed_review(root, monkeypatch, capsys):
    from nnslr_tools import brain_targets, cli, road_type_signs
    asked = []
    monkeypatch.setattr(road_type_signs, 'build_signs', lambda data, routes, **kwargs: asked.append(routes) or {})
    monkeypatch.setattr(brain_targets, 'read_labels', lambda data: {})
    assert cli.main(['road-type-signs', '--labelled', '--data-root', str(root)]) == 2
    assert 'no drive with labels' in capsys.readouterr().err and asked == []
    monkeypatch.setattr(brain_targets, 'read_labels', lambda data: {'r2': {}, 'r1': {}})
    assert cli.main(['road-type-signs', '--labelled', '--data-root', str(root)]) == 0
    assert asked == [['r1', 'r2']]


# -- what the labels give it --------------------------------------------------------------------------------------
def test_the_labels_say_the_drives_to_read_again_and_a_problem_is_words(root, checkout, monkeypatch):
    from nnslr_tools import brain_targets
    panel = Panel(root, repo=checkout)
    rows = [{'route': 'r1', 'reader_current': False}, {'route': 'r2', 'reader_current': True}]
    monkeypatch.setattr(brain_targets, 'dataset', lambda *args, **kwargs: [])
    monkeypatch.setattr(brain_targets, 'summary', lambda drives: {'partitions': {}, 'drives': rows})
    assert panel.brain_labels()['to_read'] == 1

    def broken(*args, **kwargs):
        raise OSError('the sunnypilot clone is missing')
    monkeypatch.setattr(brain_targets, 'dataset', broken)
    assert panel.brain_labels() == {'problem': 'the sunnypilot clone is missing'}


def test_the_labels_and_the_choices_go_through_the_site(root, checkout, monkeypatch):
    from nnslr_tools import brain_targets
    brain_script(checkout)
    speed_marked(root)
    monkeypatch.setattr(brain_targets, 'dataset', lambda *args, **kwargs: [])
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    try:
        # Train › Brain's own button: a wrong choice is refused, never dropped (nothing starts).
        status, _, data = server.request('POST', '/api/panel/run/train_brain', body={'pool': 'max'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, _, data = server.request('POST', '/api/panel/run/train_brain', body={'works': 'off', 'pool': 'mean'})
        assert status == 200, data
        job = json_of(data)['id']
        finished(server.app.panel, job)
        assert 'choices: 0 1 mean' in full_log(server.app.panel, job).splitlines()
        status, _, data = server.request('GET', '/api/panel/brain-labels')
        assert status == 200 and json_of(data)['to_read'] == 0
    finally:
        server.close()


def test_models_say_whether_a_brain_reads_the_plates():
    from nnslr_tools.model_roles import brain_digest
    result = {'groups': ['signs', 'plate'], 'test': {}}
    assert brain_digest(result)['plates'] is False  # a brain trained before there were plates
    assert brain_digest({**result, 'plates': False})['plates'] is False
    assert brain_digest({**result, 'plates': True})['plates'] is True
