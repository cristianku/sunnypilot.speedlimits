"""The brain in the panel: "Train the brain" (scripts/run_brain.sh) from Train › Brain, what it learns, its progress read from
its log (nnslr_tools.job_progress), what the labels give it, and which brains can be chosen for the car. Nothing is trained: the script is a stand-in, the log has the lines nnslr_tools.brain_train writes."""
from __future__ import annotations

import json
import re
import sqlite3
import sys

import pytest

from nnslr_tools import drive_review
from nnslr_tools.job_progress import progress
from nnslr_tools.model_roles import selectable
from nnslr_tools.panel import Panel, PanelError, _run_of, _settings_of
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
== 3/6 Read the signs of the drives as the car does today ==
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
                      'echo "choices: $NNSLR_BRAIN_WORKS $NNSLR_BRAIN_ROAD $NNSLR_BRAIN_DRIVER $NNSLR_BRAIN_POOL"\n'
                      'echo "road types: $NNSLR_BRAIN_TYPES"\n')
    script.chmod(0o755)
    return script


NS = 1_000_000_000


def scanned(root, route='r1', frames=True, passive=True):
    """A drive the brain can learn from (brain_targets.training_routes): its logs read to the end and, with frames, its
    frames; with passive False, sunnypilot was on. Nothing to mark: no block, the driver's own speed is the answer."""
    path = drive_review.db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for statement in drive_review.SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute("INSERT INTO scans (route, state, summary) VALUES (?, 'done', ?)",
                       (route, json.dumps({'t0_ns': 0, 'span_ns': [0, 40 * NS],
                                           'meta': {'car': {'brand': 'psa', 'passive': passive}}})))
    connection.commit()
    connection.close()
    if frames:
        listed = root / 'derived' / 'frames' / route / '0' / 'frames.jsonl'
        listed.parent.mkdir(parents=True, exist_ok=True)
        listed.write_text('{}\n')


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
def test_the_brain_is_trained_from_the_train_tab_once_a_drive_has_its_logs_read(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['train_brain'] == {
        'label': 'Train the brain', 'allowed': False, 'reason': 'scripts/run_brain.sh is missing in this installation'}
    brain_script(checkout)
    nothing = 'no drive to learn from yet: prepare one, then Read the logs on Data › Speed review'
    assert panel._action_states(ready_payload(), [])['train_brain']['reason'] == nothing
    # Its logs read but not its frames: still nothing to learn from.
    scanned(root, frames=False)
    assert panel._action_states(ready_payload(), [])['train_brain']['reason'] == nothing
    # Read and prepared, but driven with sunnypilot on: it says so, not to prepare one.
    scanned(root, 'r3', passive=False)
    assert panel._action_states(ready_payload(), [])['train_brain']['reason'] == (
        'no drive to learn from: sunnypilot was on in every drive with its logs read (1); '
        'drive with sunnylink › Toggles › Enable sunnypilot off')
    # A dashcam drive with its logs read and its frames: nothing to mark, it is learnt.
    scanned(root, 'r2')
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_brain']['reason'] == 'the GPU is busy with another process'
    # Its labels are the driver's own speed: no switch, no block and no reviewed drive of signs.
    assert panel._action_states(ready_payload(), [])['train_brain']['allowed'] is True

    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_brain', None)
    done = finished(panel, job['id'])
    lines = full_log(panel, job['id']).splitlines()
    first = lines[0].split()
    assert first[:2] == ['==', 'scripts/run_brain.sh'] and re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-brain', first[2])
    assert (done['state'], done['run'], done['label']) == ('ok', first[2], 'Train the brain')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_brain') == first[2]
    # The usual choices: the roadworks and the kinds of road on, the driver's inputs off (since 8 Nov 2026), the picture
    # as a grid.
    assert 'choices: 1 1 0 grid' in lines and 'road types: 1' in lines


def test_a_step_of_the_pipeline_takes_every_choice_of_the_brain(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': {'processes': 0}})
    view = panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {
        'works': 'on', 'road': 'on', 'types': 'off', 'driver': 'off', 'pool': 'grid'}}]}, 'admin')
    assert view['state'] == 'running' and view['steps'][0]['options']['types'] == 'off'
    done = finished(panel, view['steps'][0]['job'])
    assert done['state'] == 'ok' and 'road types: 0' in full_log(panel, view['steps'][0]['job']).splitlines()


def test_a_start_or_a_pipeline_is_refused_while_there_is_no_dashcam_drive(root, checkout, monkeypatch):
    """Not only the button: a POST of its own and a pipeline's Brain step are refused the same way, before any job."""
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root, passive=False)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': {'processes': 0}})
    refused = ('No drive to learn from: sunnypilot was on in every drive with its logs read (1); '
               'drive with sunnylink › Toggles › Enable sunnypilot off.')
    with pytest.raises(PanelError) as error:
        panel.start('train_brain', None)
    assert (error.value.status, error.value.reason, error.value.message) == (409, 'not_available', refused)
    with pytest.raises(PanelError) as error:
        panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {}}]}, 'admin')
    assert (error.value.status, error.value.message) == (409, refused)
    assert not list(panel.logs.glob('*.log')) and panel.pipeline_view() is None
    scanned(root, 'r2')
    assert finished(panel, panel.start('train_brain', None)['id'])['state'] == 'ok'


def test_what_it_learns_reaches_the_script_and_is_shown_on_the_job(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_brain', {'works': 'off', 'road': 'off', 'types': 'off', 'driver': 'off', 'pool': 'mean'})
    done = finished(panel, job['id'])
    assert {'choices: 0 0 0 mean', 'road types: 0'} <= set(full_log(panel, job['id']).splitlines())
    assert done['state'] == 'ok'
    # The job's chips, from the settings line of the script: the driver's inputs when a run had them (--driver), and
    # the runs of before 8 Nov 2026 that left them out (--no-driver) still say so.
    log = root / 'brain.log'
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool mean --no-works --no-road '
                   '--no-road-types --no-driver\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs', 'no roadworks', 'no kinds of road', 'no road types',
                                                'no driver’s inputs', 'picture as a mean']
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool grid --driver\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs', 'driver’s inputs']
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool grid\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs']


def test_the_drivers_inputs_are_only_there_when_asked(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_brain', {'driver': 'on'})
    finished(panel, job['id'])
    assert 'choices: 1 1 1 grid' in full_log(panel, job['id']).splitlines()


@pytest.mark.parametrize('options', [{'works': 'x'}, {'road': 'no'}, {'driver': 'maybe'}, {'pool': 'max'}, {'works': ['on']},
                                     {'road': ['off']}, {'pool': {'a': 1}}])
def test_bad_choices_are_refused_before_it_starts(root, checkout, monkeypatch, options):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start('train_brain', options)
    assert refused.value.status == 400 and refused.value.reason == 'bad_option'
    assert panel.jobs() == []


@pytest.mark.parametrize('key,value', [('works', 'off'), ('road', 'off'), ('driver', 'off'), ('pool', 'mean')])
def test_its_choices_are_refused_on_another_training(root, checkout, key, value):
    with pytest.raises(PanelError) as refused:
        Panel(root, repo=checkout).start('train_decider', {key: value})
    assert refused.value.reason == 'bad_option'


def test_it_is_a_step_of_the_pipeline_with_its_choices(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {'pool': 'max'}}]}, None)
    assert refused.value.status == 400
    view = panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {'works': 'off', 'pool': 'mean'}}]}, None)
    assert view['steps'][0]['action'] == 'train_brain'
    job = next(item for item in panel.jobs() if item['action'] == 'train_brain')
    finished(panel, job['id'])
    assert 'choices: 0 1 0 mean' in full_log(panel, job['id']).splitlines()


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
def test_road_type_signs_labelled_reads_every_drive_the_brain_learns_from(root, monkeypatch, capsys):
    from nnslr_tools import brain_targets, cli, road_type_signs
    asked = []
    monkeypatch.setattr(road_type_signs, 'build_signs', lambda data, routes, **kwargs: asked.append(routes) or {})
    monkeypatch.setattr(brain_targets, 'training_routes', lambda data: [])
    assert cli.main(['road-type-signs', '--labelled', '--data-root', str(root)]) == 2
    assert 'no drive to learn from' in capsys.readouterr().err and asked == []
    # A drive is the brain's without a mark of Speed review (read_labels is not asked).
    monkeypatch.setattr(brain_targets, 'read_labels', lambda data: pytest.fail('read_labels asked'))
    monkeypatch.setattr(brain_targets, 'training_routes', lambda data: ['r1', 'r2'])
    assert cli.main(['road-type-signs', '--labelled', '--data-root', str(root)]) == 0
    assert asked == [['r1', 'r2']]


# -- the command line ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('flags,driver', [([], False), (['--driver'], True), (['--no-driver'], False)])
def test_train_brain_leaves_the_drivers_inputs_out_unless_asked(tmp_path, monkeypatch, capsys, flags, driver):
    """nnslr train-brain: the driver's inputs out by default since 8 Nov 2026, --driver the comparison, --no-driver (the
    opt-out of before) still the same. Nothing is trained: brain_train is a stand-in."""
    import types

    import nnslr_tools
    from nnslr_tools import cli
    asked = []
    measure = {'speed': 1.0, 'speeds_said': [80], 'one_speed': 2.0, 'follow_the_sign': 3.0}

    def train_brain(*args, **kwargs):
        asked.append(kwargs)
        return {'train': measure, 'validation': measure, 'test': measure,
                'validation_swapped': measure, 'test_swapped': measure}
    fake = types.SimpleNamespace(EPOCHS=40, train_brain=train_brain)
    monkeypatch.setitem(sys.modules, 'nnslr_tools.brain_train', fake)
    monkeypatch.setattr(nnslr_tools, 'brain_train', fake, raising=False)
    cli.main(['train-brain', '--data-root', str(tmp_path / 'data'), '--run-dir', str(tmp_path / 'out'), *flags])
    assert [kwargs['driver'] for kwargs in asked] == [driver]


# -- what the labels give it --------------------------------------------------------------------------------------
def test_the_labels_say_the_drives_to_read_again_and_a_problem_is_words(root, checkout, monkeypatch):
    from nnslr_tools import brain_targets
    panel = Panel(root, repo=checkout)
    rows = [{'route': 'r1', 'reader_current': False}, {'route': 'r2', 'reader_current': True}]
    monkeypatch.setattr(brain_targets, 'dataset', lambda *args, **kwargs: [])
    monkeypatch.setattr(brain_targets, 'summary', lambda drives: {'partitions': {}, 'drives': rows})
    labels = panel.brain_labels()
    assert labels['to_read'] == 1
    assert labels['training'] == {'drives': 0, 'hours': 0.0, 'sunnypilot_on': 0}
    # The names of the kinds of road, the first ones while Speed review has no file yet.
    assert labels['road_labels'] == {'narrow': 'Narrow road', 'country': 'Country road', 'gravel': 'Gravel road',
                                     'potholes': 'Potholes'}

    def broken(*args, **kwargs):
        raise OSError('the sunnypilot clone is missing')
    monkeypatch.setattr(brain_targets, 'dataset', broken)
    assert panel.brain_labels() == {'problem': 'the sunnypilot clone is missing'}


def test_the_labels_and_the_choices_go_through_the_site(root, checkout, monkeypatch):
    from nnslr_tools import brain_targets
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(brain_targets, 'dataset', lambda *args, **kwargs: [])
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    try:
        # Train › Brain's own button: a wrong choice is refused, never dropped (nothing starts).
        for body in ({'pool': 'max'}, {'types': 'maybe'}):
            status, _, data = server.request('POST', '/api/panel/run/train_brain', body=body)
            assert status == 400 and json_of(data)['error'] == 'bad_option', body
        status, _, data = server.request('POST', '/api/panel/run/train_brain',
                                         body={'works': 'off', 'road': 'off', 'pool': 'mean'})
        assert status == 200, data
        job = json_of(data)['id']
        finished(server.app.panel, job)
        assert 'choices: 0 0 0 mean' in full_log(server.app.panel, job).splitlines()
        status, _, data = server.request('GET', '/api/panel/brain-labels')
        assert status == 200 and json_of(data)['to_read'] == 0
        assert json_of(data)['training'] == {'drives': 1, 'hours': 0.01, 'sunnypilot_on': 0}
    finally:
        server.close()


def test_models_say_whether_a_brain_reads_the_plates():
    from nnslr_tools.model_roles import brain_digest
    result = {'groups': ['signs', 'plate'], 'test': {}}
    assert brain_digest(result)['plates'] is False  # a brain trained before there were plates
    assert brain_digest({**result, 'plates': False})['plates'] is False
    assert brain_digest({**result, 'plates': True})['plates'] is True
    assert brain_digest(result)['hours'] is False
    assert brain_digest({**result, 'hours': True})['hours'] is True


def test_models_say_how_the_car_steps_a_brain():
    from nnslr_tools.model_roles import brain_digest
    old = brain_digest({'groups': ['signs'], 'test': {}})  # a brain of before 8 Nov 2026
    assert (old['rate_hz'], old['ahead'], old['hidden'], old['kph']) == (1.0, False, None, {'from': 5, 'to': 200, 'step': 5})
    new = brain_digest({'groups': ['signs'], 'test': {}, 'rate_hz': 4, 'ahead': True, 'hidden': 128,
                        'kph': {'from': 1, 'to': 200, 'step': 1}})
    assert (new['rate_hz'], new['ahead'], new['hidden'], new['kph']) == (4, True, 128, {'from': 1, 'to': 200, 'step': 1})
