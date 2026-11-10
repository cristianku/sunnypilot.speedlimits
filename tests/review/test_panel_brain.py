"""The brain in the panel: "Train the brain" (scripts/run_brain.sh) from Train › Brain, what it learns, its progress read from
its log (nnslr_tools.job_progress), what the labels give it, and which brains can be chosen for the car. Nothing is trained: the script is a stand-in, the log has the lines nnslr_tools.brain_train writes."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time

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


@pytest.fixture(autouse=True)
def no_cards(monkeypatch):
    """No card on this machine: a start takes none and is never refused for the server's own GPUs, busy or not."""
    monkeypatch.setattr(Panel, 'system', lambda self: {'gpu': {'processes': 0}})


# The networks in use that read the brain's signs (models/active.json): without an applicability network the brain is
# refused (#31).
IN_USE = {'cutter': 'cutter:c1/signnet', 'classifier': 'classifier:k1/classifier', 'decider': 'decider:d1/experiment-b/crop_geom'}


@pytest.fixture(autouse=True)
def sign_networks_in_use(monkeypatch):
    from nnslr_tools import preannotate
    in_use = dict(IN_USE)
    monkeypatch.setattr(preannotate, 'networks_in_use', lambda root: dict(in_use))
    return in_use


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes}}}


def brain_script(checkout):
    script = checkout / 'scripts' / 'run_brain.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Brain =="\necho "run:          $1"\n'
                      'echo "choices: $NNSLR_BRAIN_WORKS $NNSLR_BRAIN_ROAD $NNSLR_BRAIN_POOL"\n'
                      'echo "road types: $NNSLR_BRAIN_TYPES"\n'
                      'echo "driver: ${NNSLR_BRAIN_DRIVER-unset}"\n'
                      'echo "init=${NNSLR_BRAIN_INIT_FROM:-fresh}"\n'
                      'echo "signs: ${NNSLR_BRAIN_CUTTER-unset} ${NNSLR_BRAIN_CLASSIFIER-unset} ${NNSLR_BRAIN_DECIDER-unset}"\n')
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


def test_the_reading_of_the_signs_moves_with_the_segments_then_the_drives():
    head = LOG[:LOG.index('== 4/6')]
    result = progress('train_brain', 'running', head)
    # Nothing read yet: the 6 of the checks.
    assert result['steps'][1]['state'] == 'current' and result['percent'] == pytest.approx(6.0)
    log = head + 'road-type signs: 100/400 segments\n'
    result = progress('train_brain', 'running', log)
    assert result['step_progress']['text'] == 'as the car reads · 100 of 400 segments'
    # A quarter of the first half of its 22.
    assert result['percent'] == pytest.approx(6 + 22 * 0.125, abs=0.1)
    log += ('road-type signs: 400/400 segments\nbrain signs: 3 drives to read, 2 at a time, on the cards 0, 1\n'
            'brain signs 1/4 00000001--aaaaaaaaaa: read by these networks already\n'
            'brain signs 2/4 00000002--bbbbbbbbbb: 900 pictures read, 12 boxes of our road, card 1\n')
    result = progress('train_brain', 'running', log)
    assert result['step_progress']['text'] == 'our networks · 2 of 4 drives'
    assert result['percent'] == pytest.approx(6 + 22 * 0.75, abs=0.1)
    # The seeds after it: the step is done whatever its last line said.
    assert progress('train_brain', 'running', log + LOG[LOG.index('== 4/6'):])['percent'] == pytest.approx(29.2)
    # The same NAME again from a terminal, its log written on: its step 3 starts from nothing.
    again = progress('train_brain', 'running', log + '== 3/6 Read the signs of the drives as the car does today ==\n')
    assert again['percent'] == pytest.approx(6.0) and again['step_progress'] is None


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
    # The usual choices: the roadworks and the kinds of road on, the picture as a grid; the driver's inputs are gone
    # (10 Nov 2026).
    assert {'choices: 1 1 grid', 'road types: 1', 'driver: unset'} <= set(lines)


def test_a_step_of_the_pipeline_takes_every_choice_of_the_brain(root, checkout, monkeypatch):
    # driver "off": what a page loaded before 10 Nov 2026 still sends.
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
    assert {'choices: 0 0 mean', 'road types: 0', 'driver: unset'} <= set(full_log(panel, job['id']).splitlines())
    assert done['state'] == 'ok'
    # The job's chips, from the settings line of the script: the runs of before 10 Nov 2026 that had the driver's inputs
    # (--driver), and those of before 8 Nov that left them out (--no-driver), still say so.
    log = root / 'brain.log'
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool mean --no-works --no-road '
                   '--no-road-types --no-driver\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs', 'no roadworks', 'no kinds of road', 'no road types',
                                                'no driver’s inputs', 'picture as a mean']
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool grid --driver\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs', 'driver’s inputs']
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool grid\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs']


def test_the_drivers_inputs_are_refused_and_a_page_of_before_still_starts(root, checkout, monkeypatch):
    """No brain takes the pedals nor the car's speed since 10 Nov 2026: "on" is refused before any job, alone or in a
    pipeline; "off", what a page loaded before sends, starts as usual, and the script is not told about them."""
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start('train_brain', {'driver': 'on'})
    assert refused.value.status == 400 and refused.value.reason == 'bad_option'
    assert 'followed your speed' in refused.value.message
    with pytest.raises(PanelError) as refused:
        panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {'driver': 'on'}}]}, None)
    assert refused.value.status == 400 and panel.jobs() == []
    job = panel.start('train_brain', {'driver': 'off'})
    finished(panel, job['id'])
    assert {'choices: 1 1 grid', 'driver: unset'} <= set(full_log(panel, job['id']).splitlines())


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
    assert 'choices: 0 1 mean' in full_log(panel, job['id']).splitlines()


def brain_run(root, name, *, groups=('signs', 'plate', 'hour'), finished=True, start=None):
    """A brain of Models: its weights, and with finished its result.json."""
    run = root / 'runs' / name
    run.mkdir(parents=True)
    (run / 'brain-last.pt').write_text('weights')
    if finished:
        (run / 'result.json').write_text(json.dumps({'groups': list(groups), 'seed': 1, 'test': {}, 'split_sha256': 'a' * 64,
                                                     'start': start}))


def test_it_goes_on_from_a_finished_brain_of_models(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    brain_run(root, 'b1-s1')
    brain_run(root, 'b0-s1', groups=('signs', 'pedals', 'speed'))
    brain_run(root, 'b2-s0', finished=False)

    def init_of(options, pipeline=False):
        if pipeline:
            panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': options}]}, None)
            job = next(item for item in panel.jobs() if item['action'] == 'train_brain' and item['state'] == 'running')
        else:
            job = panel.start('train_brain', options)
        finished(panel, job['id'])
        time.sleep(1.1)  # a job's log is named by its second
        return re.search(r'^init=(.*)$', full_log(panel, job['id']), re.M).group(1)

    # The usual start, as before: nothing in the environment; a brain of Models by its id: its weights, alone or in a step.
    assert init_of({}) == init_of({'start': 'fresh'}) == 'fresh'
    assert init_of({'start': 'brain:b1-s1/brain', 'pool': 'mean'}).endswith('/runs/b1-s1/brain-last.pt')
    assert init_of({'start': 'brain:b1-s1/brain'}, pipeline=True).endswith('/runs/b1-s1/brain-last.pt')
    # One that took your pedals and the car's speed too: the new one leaves those inputs out.
    assert init_of({'start': 'brain:b0-s1/brain'}).endswith('/runs/b0-s1/brain-last.pt')
    refused = {'brain:b2-s0/brain': 'That network has not finished its training: choose a finished one.',
               'cutter:b1-s1/signnet': 'Unknown network to start from: choose one of Models.',
               '/etc/passwd': 'Unknown network to start from: choose one of Models.'}
    for start, message in refused.items():
        with pytest.raises(PanelError) as error:
            panel.start('train_brain', {'start': start})
        assert (error.value.status, error.value.message) == (400, message), start
    # Models say what a brain went on from; the job's chip too.
    brain_run(root, 'b3-s0', start={'path': '/d/runs/b1-s1/brain-last.pt', 'run': 'b1-s1', 'epoch': 39,
                                    'split_sha256': 'a' * 64})
    found = {entry['id']: entry for entry in panel.models_index.artifacts(('brain',))['brain']}
    assert (found['brain:b3-s0/brain']['initialization'], found['brain:b3-s0/brain']['start']) == \
        ('start', {'run': 'b1-s1', 'epoch': 39, 'another_split': False})
    assert (found['brain:b1-s1/brain']['initialization'], found['brain:b1-s1/brain']['start']) == ('ImageNet', None)
    log = root / 'brain.log'
    log.write_text('== scripts/run_brain.sh r\nrun:          r\nsettings:     --epochs 40 --pool grid --init-from '
                   '/d/runs/b1-s1/brain-last.pt\n')
    assert _settings_of(log, 'train_brain') == ['40 epochs', 'from b1-s1']


def test_a_brain_training_started_from_a_terminal_has_the_progress_of_its_run_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / '2026-11-05_1500-brain' / 'brain.log'
    log.parent.mkdir(parents=True)
    log.write_text(LOG)
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'brain training', 'pid': 1}]}
    panel._terminal_training(payload)
    assert payload['activities'][0]['progress']['detail'] == 'Seed 0 · epoch 2 of 40'


# -- what can go to the car ---------------------------------------------------------------------------------------
def test_a_brain_that_takes_the_drivers_inputs_cannot_be_chosen():
    """With the car's speed in, the brain of 10 Nov 2026 followed his speed: one that takes the pedals or the speed stays
    in its run, to compare, and cannot be put in use; its groups as names or as [name, size]."""
    usual = {'role': 'brain', 'finished': True, 'groups': ['signs', 'plate', 'hour', 'lead', 'country']}
    assert selectable(usual) == (True, None)
    assert selectable({**usual, 'groups': None}) == (True, None)
    for groups in ([*usual['groups'], 'pedals', 'speed'], [*usual['groups'], 'speed'], [*usual['groups'], 'pedals'],
                   [[name, 1] for name in [*usual['groups'], 'speed']]):
        assert selectable({**usual, 'groups': groups}) == (False, 'takes_the_driver'), groups
    assert selectable({**usual, 'groups': [*usual['groups'], 'engaged', 'motion']}) == (True, None)
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
def test_train_brain_has_no_drivers_inputs(tmp_path, monkeypatch, capsys):
    """nnslr train-brain: the driver's inputs are gone since 10 Nov 2026, train_brain is not asked about them and
    --driver, --no-driver are no flags any more. Nothing is trained: brain_train is a stand-in."""
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
    command = ['train-brain', '--data-root', str(tmp_path / 'data'), '--run-dir', str(tmp_path / 'out')]
    cli.main(command)
    assert len(asked) == 1 and 'driver' not in asked[0]
    for flag in ('--driver', '--no-driver'):
        with pytest.raises(SystemExit):
            cli.main([*command, flag])
    assert len(asked) == 1


# -- what the labels give it --------------------------------------------------------------------------------------
def test_the_labels_say_the_drives_to_read_again_and_a_problem_is_words(root, checkout, monkeypatch):
    from nnslr_tools import brain_targets
    panel = Panel(root, repo=checkout)
    rows = [{'route': 'r1', 'reader_current': False}, {'route': 'r2', 'reader_current': True}]
    monkeypatch.setattr(brain_targets, 'dataset', lambda *args, **kwargs: [])
    monkeypatch.setattr(brain_targets, 'summary', lambda drives: {'partitions': {}, 'drives': rows})
    # The networks in use not exported yet for the brain (sign_networks): every drive is read again first.
    assert panel.brain_labels()['to_read'] == 2
    from nnslr_tools import sign_networks
    seen = []
    monkeypatch.setattr(sign_networks, 'exported', lambda root, ids: seen.append(ids) or root / 'set')
    labels = panel.brain_labels()
    assert labels['to_read'] == 1 and seen == [IN_USE]
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
        assert 'choices: 0 0 mean' in full_log(server.app.panel, job).splitlines()
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


def sign_network_entries(panel, monkeypatch):
    """Models with a SignNet of the car and an old Detector, a classifier and an old Reader, a whole applicability run and
    a smoke run."""
    entries = {
        'cutter': [{'id': 'cutter:c2/signnet', 'run': 'c2', 'part': 'signnet', 'finished': True, 'kind': 'signnet',
                    'fits_the_car': True},
                   {'id': 'cutter:c0/detector', 'run': 'c0', 'part': 'detector', 'finished': True, 'kind': 'detector'}],
        'classifier': [{'id': 'classifier:k2/classifier', 'run': 'k2', 'part': 'classifier', 'finished': True},
                       {'id': 'classifier:k0/reader', 'run': 'k0', 'part': 'reader', 'finished': True},
                       {'id': 'classifier:k3/classifier', 'run': 'k3', 'part': 'classifier', 'finished': False}],
        'decider': [{'id': 'decider:d2/experiment-b/crop_geom', 'run': 'd2', 'part': 'experiment-b/crop_geom',
                     'finished': True, 'smoke_limit': None},
                    {'id': 'decider:d3/experiment-b/crop_geom', 'run': 'd3', 'part': 'experiment-b/crop_geom',
                     'finished': True, 'smoke_limit': 20}]}
    monkeypatch.setattr(panel.models_index, 'artifacts', lambda roles=tuple(entries): {role: entries.get(role, []) for role in roles})


def test_the_networks_that_read_its_signs_are_chosen_and_reach_the_script(root, checkout, monkeypatch, sign_networks_in_use):
    """Train › Brain's Signs: the cutter, the classifier and the applicability network that read the signs it learns from
    (#31), those in use when none is given (the script takes them), else finished versions of Models the comma runs."""
    panel = Panel(root, repo=checkout)
    brain_script(checkout)
    scanned(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    sign_network_entries(panel, monkeypatch)

    def signs_of(options, pipeline=False):
        if pipeline:
            panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': options}]}, None)
            job = next(item for item in panel.jobs() if item['action'] == 'train_brain' and item['state'] == 'running')
        else:
            job = panel.start('train_brain', options)
        finished(panel, job['id'])
        time.sleep(1.1)  # a job's log is named by its second
        return re.search(r'^signs: (.*)$', full_log(panel, job['id']), re.M).group(1)

    assert signs_of({}) == 'unset unset unset'
    chosen = {'sign_cutter': 'cutter:c2/signnet', 'sign_classifier': 'classifier:k2/classifier',
              'sign_decider': 'decider:d2/experiment-b/crop_geom'}
    assert signs_of(chosen) == 'cutter:c2/signnet classifier:k2/classifier decider:d2/experiment-b/crop_geom'
    assert signs_of({'sign_decider': 'decider:d2/experiment-b/crop_geom'}, pipeline=True) == \
        'unset unset decider:d2/experiment-b/crop_geom'
    refused = {('sign_cutter', 'cutter:c0/detector'): 'The comma reads its signs with a SignNet of its band and grid.',
               ('sign_classifier', 'classifier:k0/reader'): 'The comma reads its signs with a classifier with “not a sign”.',
               ('sign_classifier', 'classifier:k3/classifier'): 'That network has not finished its training: choose a finished one.',
               ('sign_decider', 'decider:d3/experiment-b/crop_geom'): 'A smoke run learnt on a few frames: choose a whole '
                                                                      'applicability training.',
               ('sign_decider', '/etc/passwd'): 'Unknown network to read the signs with: choose one of Models.',
               ('sign_cutter', 'decider:d2/experiment-b/crop_geom'): 'Unknown network to read the signs with: choose one of Models.'}
    for (key, value), message in refused.items():
        with pytest.raises(PanelError) as error:
            panel.start('train_brain', {key: value})
        assert (error.value.status, error.value.message) == (400, message), (key, value)
    # Without an applicability network in use the brain is refused, unless one is chosen.
    del sign_networks_in_use['decider']
    with pytest.raises(PanelError) as error:
        panel.start('train_brain', {})
    assert (error.value.status, error.value.reason) == (409, 'no_sign_networks')
    assert 'no applicability network' in error.value.message
    with pytest.raises(PanelError):
        panel.start_pipeline({'steps': [{'action': 'train_brain', 'options': {}}]}, None)
    assert signs_of({'sign_decider': 'decider:d2/experiment-b/crop_geom'}) == 'unset unset decider:d2/experiment-b/crop_geom'
    # Only the brain takes them.
    with pytest.raises(PanelError) as error:
        panel.start('train_decider', {'sign_decider': 'decider:d2/experiment-b/crop_geom'})
    assert error.value.status == 400


def test_models_say_which_networks_read_a_brains_signs():
    from nnslr_tools.model_roles import brain_digest
    base = {'groups': ['signs'], 'test': {}}
    assert brain_digest(base)['sign_networks'] is None
    networks = {'cutter': 'cutter:c1/signnet', 'classifier': 'classifier:k1/classifier', 'decider': 'decider:d1/x'}
    assert brain_digest({**base, 'sign_networks': networks})['sign_networks'] == networks
    assert brain_digest({**base, 'sign_networks': {'cutter': 'cutter:c1/signnet'}})['sign_networks'] is None
