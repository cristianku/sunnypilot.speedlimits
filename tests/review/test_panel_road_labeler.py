"""The road labeler in the panel: "Train the road labeler" and "Prepare Speed review" (helpers/road_labeler.sh train and
predict) on Models, their options, their refusals, their progress read from the log, and the runs the card shows
(nnslr_tools.labeler_runs). Nothing is trained: the helper is a stand-in that prints its arguments."""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from nnslr_tools import drive_review, labeler_runs
from nnslr_tools.job_progress import progress
from nnslr_tools.panel import ACTION_OPTIONS, ACTIONS, Panel, PanelError, _run_of
from review_helpers import RunningServer, json_of
from test_panel import checkout, finished, full_log, root  # noqa: F401  (fixtures and helpers)

NS = 1_000_000_000
RUN = '2026-11-08_1200-road-labeler'


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes}}}


def labeler_script(checkout, venv=True):
    """A stand-in for helpers/road_labeler.sh that prints its arguments, and the .venv-gpu it needs."""
    script = checkout / 'helpers' / 'road_labeler.sh'
    script.write_text('#!/usr/bin/env bash\necho "run:          x"\necho "arguments: $*"\n')
    script.chmod(0o755)
    if venv:
        nnslr = checkout / '.venv-gpu' / 'bin' / 'nnslr'
        nnslr.parent.mkdir(parents=True, exist_ok=True)
        nnslr.write_text('#!/bin/sh\n')
        nnslr.chmod(0o755)
    return script


def scanned(root, route='r1'):
    """A drive the labeler can read (road_labeler_data.labeler_routes): its scan finished and its frames."""
    path = drive_review.db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for statement in drive_review.SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute("INSERT INTO scans (route, state, summary) VALUES (?, 'done', ?)",
                       (route, json.dumps({'t0_ns': 0, 'span_ns': [0, 40 * NS]})))
    connection.commit()
    connection.close()
    listed = root / 'derived' / 'frames' / route / '0' / 'frames.jsonl'
    listed.parent.mkdir(parents=True, exist_ok=True)
    listed.write_text('{}\n')


def labeler_run(root, name=RUN, dry_run=False, checkpoint=True, tunnel_gate=None):
    """A run of nnslr road-labeler-train as result.json says it: the tunnel trained with both sides and through its
    gate (unless tunnel_gate says why not), parking with one side alone, the roadworks short of answers."""
    run_dir = root / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=True)
    gate = {'tunnel': {'inside': 400, 'outside': 9000, 'said': 300, 'precision': 0.97, 'why': tunnel_gate},
            'parking': {'inside': 0, 'outside': 9000, 'said': 0, 'precision': None, 'why': '0 steps inside'},
            'works': {'inside': 12, 'outside': 9000, 'said': 0, 'precision': None, 'why': '12 steps inside'},
            'type:motorway': {'inside': 900, 'outside': 100, 'said': 50, 'precision': 0.95, 'why': None},
            'type:city': {'inside': 0, 'outside': 100, 'said': 0, 'precision': None, 'why': 'too few'}}
    result = {'run': name, 'dry_run': dry_run, 'encoder_id': 'e', 'step_hz': 2, 'window_steps': 240,
              'kinds': ['tunnel', 'parking'], 'road_types': ['motorway', 'city'], 'seed': 0,
              'history': [{'epoch': 0}, {'epoch': 1}], 'drives': {'r1': 800, 'r2': 400},
              'train_sides': {'tunnel': {'inside': 900, 'outside': 20000}, 'parking': {'inside': 0, 'outside': 20000},
                              'works': {'inside': 40, 'outside': 20000}},
              'gate': gate,
              'validation': {'types': {'steps': 9000, 'right': 0.91, 'person_steps': 600, 'right_on_person': 0.82,
                                       'map_on_person': 0.7, 'per_type': {}},
                             'kinds': {'tunnel': {'inside': 400, 'outside': 9000, 'precision': 0.95, 'recall': 0.8},
                                       'parking': {'inside': 0, 'outside': 9000, 'precision': None, 'recall': None}},
                             'works': {'inside': 12, 'outside': 9000, 'precision': 0.5, 'recall': 0.25}}}
    (run_dir / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    if checkpoint:
        (run_dir / 'road-labeler.pt').write_bytes(b'weights')
    return run_dir


# -- the panel's key list -----------------------------------------------------------------------------------------
def test_the_two_actions_are_in_the_panel_with_their_options():
    assert ACTIONS['road_labeler_train'] == {'label': 'Train the road labeler', 'group': 'train',
                                             'command': ['helpers/road_labeler.sh', 'train']}
    assert ACTIONS['road_labeler_predict'] == {'label': 'Prepare Speed review', 'group': 'train',
                                               'command': ['helpers/road_labeler.sh', 'predict']}
    assert ACTION_OPTIONS['road_labeler_train'] == ('dry',)
    assert ACTION_OPTIONS['road_labeler_predict'] == ('dry', 'labeler', 'marks')


def test_the_review_server_passes_the_road_labeler_keys_on():
    # review.py's own list of the keys it hands to Panel.start (a key left out of it is dropped without a word: a dry
    # run dropped would start a real training on the GPU).
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / 'src/nnslr_tools/review.py').read_text(encoding='utf-8')
    assert "for key in ('dry', 'labeler', 'marks')" in text


# -- when they can start ------------------------------------------------------------------------------------------
def test_they_say_why_they_cannot_start(root, checkout):
    panel = Panel(root, repo=checkout)
    states = panel._action_states(ready_payload(), [])
    assert states['road_labeler_train'] == {'label': 'Train the road labeler', 'allowed': False,
                                            'reason': 'helpers/road_labeler.sh is missing in this installation'}
    labeler_script(checkout, venv=False)
    assert panel._action_states(ready_payload(), [])['road_labeler_train']['reason'] == (
        'no .venv-gpu: install it with scripts/setup_v100_training_env.sh')
    labeler_script(checkout)
    assert panel._action_states(ready_payload(), [])['road_labeler_train']['reason'] == (
        'no drive with its logs read and its frames')
    scanned(root)
    states = panel._action_states(ready_payload(), [])
    # No confirmed sign needed: its answers are the corrections of Speed review.
    assert states['road_labeler_train']['allowed'] is True
    assert states['road_labeler_predict']['reason'] == 'no road labeler trained yet'
    labeler_run(root, checkpoint=False)
    assert panel._action_states(ready_payload(), [])['road_labeler_predict']['reason'] == 'no road labeler trained yet'
    labeler_run(root)
    assert panel._action_states(ready_payload(), [])['road_labeler_predict']['allowed'] is True
    # A busy GPU leaves them allowed: a dry run on the CPU can start; start() refuses a real run.
    assert panel._action_states(ready_payload(gpu_processes=2), [])['road_labeler_train']['allowed'] is True


# -- starting them --------------------------------------------------------------------------------------------------
@pytest.fixture
def ready(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    labeler_script(checkout)
    scanned(root)
    labeler_run(root)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': {'processes': 0}})
    return panel


def start(panel, action, options=None):
    """panel.start, again a moment later when a job started in the same second (a job's id is its second)."""
    for _ in range(40):
        try:
            return panel.start(action, options)
        except PanelError as busy:
            if busy.reason != 'busy':
                raise
            time.sleep(0.05)
    raise AssertionError('the panel stayed busy')


def arguments(panel, job):
    finished(panel, job['id'])
    return next(line for line in full_log(panel, job['id']).splitlines() if line.startswith('arguments:'))


def test_train_and_its_dry_run_reach_the_helper(ready):
    assert arguments(ready, start(ready, 'road_labeler_train')) == 'arguments: train'
    assert arguments(ready, start(ready, 'road_labeler_train', {'dry': 'on'})) == 'arguments: train --dry-run'


def test_prepare_speed_review_takes_its_run_every_drive_and_the_marks(ready, root):
    run_dir = str(root / 'runs' / RUN)
    job = start(ready, 'road_labeler_predict', {'labeler': RUN})
    assert arguments(ready, job) == f'arguments: predict --run-dir={run_dir} --all'
    # The job names its run from the helper's "run:" line, as a training does.
    assert _run_of(ready.logs / (job['id'] + '.log'), 'road_labeler_predict') == 'x'
    assert next(item for item in ready.jobs() if item['id'] == job['id'])['run'] == 'x'
    job = start(ready, 'road_labeler_predict', {'labeler': RUN, 'marks': 'off', 'dry': 'on'})
    assert arguments(ready, job) == f'arguments: predict --dry-run --run-dir={run_dir} --all --no-marks'


@pytest.mark.parametrize('action,options', [
    ('road_labeler_train', {'dry': 'yes'}), ('road_labeler_train', {'dry': True}), ('road_labeler_train', {'dry': 'off'}),
    ('road_labeler_train', {'labeler': RUN}), ('road_labeler_train', {'marks': 'off'}),
    ('road_labeler_predict', {}), ('road_labeler_predict', {'labeler': 'nope'}), ('road_labeler_predict', {'labeler': '../x'}),
    ('road_labeler_predict', {'labeler': RUN, 'marks': 'maybe'}), ('road_labeler_predict', {'labeler': [RUN]}),
    ('train_decider', {'dry': 'on'}), ('train_brain', {'labeler': RUN}), ('regression', {'marks': 'off'}),
])
def test_a_bad_or_unknown_option_is_refused_and_nothing_starts(ready, action, options):
    with pytest.raises(PanelError) as refused:
        ready.start(action, options)
    assert (refused.value.status, refused.value.reason) == (400, 'bad_option')
    assert ready.jobs() == []


def test_a_run_without_its_checkpoint_cannot_prepare(ready, root):
    labeler_run(root, '2026-11-08_1300-road-labeler', checkpoint=False)
    with pytest.raises(PanelError) as refused:
        ready.start('road_labeler_predict', {'labeler': '2026-11-08_1300-road-labeler'})
    assert refused.value.reason == 'bad_option'


def test_a_real_run_is_refused_on_a_busy_gpu_but_not_a_dry_one(ready, monkeypatch):
    monkeypatch.setattr(ready, 'system', lambda: {'gpu': {'processes': 2}})
    for action, options in (('road_labeler_train', {}), ('road_labeler_predict', {'labeler': RUN})):
        with pytest.raises(PanelError) as refused:
            ready.start(action, options)
        assert (refused.value.status, refused.value.reason) == (409, 'gpu_taken')
    assert ready.jobs() == []
    assert arguments(ready, start(ready, 'road_labeler_train', {'dry': 'on'})) == 'arguments: train --dry-run'


def test_the_review_server_refuses_a_dry_run_that_is_not_a_string(root, checkout):
    labeler_script(checkout)
    scanned(root)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    server.app.panel.system = lambda: {'gpu': {'processes': 0}}
    try:
        # Refused, never dropped into a real training.
        for body in ({'dry': True}, {'dry': 1}, {'dry': 'yes'}, {'labeler': RUN}):
            status, _, data = server.request('POST', '/api/panel/run/road_labeler_train', body=body)
            assert status == 400 and json_of(data)['error'] == 'bad_option', body
        assert server.app.panel.jobs() == []
        status, _, data = server.request('POST', '/api/panel/run/road_labeler_train', body={'dry': 'on'})
        assert status == 200, data
        assert arguments(server.app.panel, json_of(data)) == 'arguments: train --dry-run'
        status, _, data = server.request('GET', '/api/panel/road-labeler')
        assert status == 200 and json_of(data)['runs'] == [] and json_of(data)['drives'] == 1
    finally:
        server.close()


# -- its progress -------------------------------------------------------------------------------------------------
TRAIN_LOG = """== helpers/road_labeler.sh train
run:          2026-11-08_1200-road-labeler
output:       /data/runs/2026-11-08_1200-road-labeler
== 1/2 Check the GPU ==
GPU 0 free: 300 of 32768 MiB used, no other compute process
drives:       4
== 2/2 Read the drives and their pictures, then train ==
r0: no finished scan or no frames, left out
r1: features of 800 steps, 790 with a picture, 300s
"""


def test_the_training_reads_the_drives_then_its_epochs():
    result = progress('road_labeler_train', 'running', TRAIN_LOG)
    assert [(step['key'], step['state']) for step in result['steps']] == [
        ('checks', 'done'), ('features', 'current'), ('road_labeler', 'todo')]
    # 2 + 2 of the 4 drives of the 58 of the reading.
    assert result['percent'] == pytest.approx(31.0)
    assert result['detail'] == '2 of 4 drives done'
    assert result['step_progress']['text'] == 'drive 3 of 4'
    log = TRAIN_LOG + ('r2: features of 400 steps, 400 with a picture, 100s\n'
                       '"tunnel" in train: 900 steps inside, 20000 outside.\n'
                       'road_labeler epoch 3/30 train_loss 0.4100 val_types 0.91 val_on_person 0.82 (map 0.7) 120 steps 40s\n')
    result = progress('road_labeler_train', 'running', log)
    assert [step['state'] for step in result['steps']] == ['done', 'done', 'current']
    assert result['detail'] == 'Road labeler · epoch 3 of 30'
    assert result['percent'] == pytest.approx(64.0)
    point = result['curves']['road_labeler']['points'][0]
    assert (point['epoch'], point['train_loss'], point['val_types'], point['val_on_person'], point['seconds']) == (
        3, 0.41, 0.91, 0.82, 40)
    assert progress('road_labeler_train', 'ok', log)['percent'] == 100.0


def test_prepare_speed_review_goes_drive_by_drive():
    log = ('run:          2026-11-08_1200-road-labeler\n== 1/2 Check the GPU ==\nGPU 0 free\ndrives:       3\n'
           '== 2/2 Prepare Speed review drive by drive ==\n'
           'r1: features of 800 steps, 790 with a picture, 300s\n{"route_id": "r1", "steps": 800}\n')
    result = progress('road_labeler_predict', 'running', log)
    assert [(step['key'], step['state']) for step in result['steps']] == [('checks', 'done'), ('predict', 'current')]
    assert result['detail'] == '1 of 3 drives done'
    assert result['percent'] == pytest.approx(3 + 97 / 3, abs=0.1)


def test_a_busy_gpu_says_why_it_failed():
    log = ('run:          x\n== 1/2 Check the GPU ==\nhelpers/road_labeler.sh: the GPU 0 is busy (31726 of 32768 MiB used, '
           '2 other compute processes): wait for it to finish, or --dry-run on the CPU\n')
    result = progress('road_labeler_train', 'failed', log)
    assert result['steps'][0]['state'] == 'failed'
    assert result['failure'] == 'The GPU was busy with another job: wait for it to finish, or start a dry run on the CPU.'


# -- the runs of the card -----------------------------------------------------------------------------------------
def test_the_card_shows_each_run_with_its_heads_and_whether_they_write(root):
    labeler_run(root)
    labeler_run(root, '2026-11-08_1100-road-labeler-dry', dry_run=True)
    (root / 'runs' / 'brain-run').mkdir(parents=True)
    (root / 'runs' / 'brain-run' / 'result.json').write_text('{"test": {}}')
    view = labeler_runs.summary(root)
    assert view['total'] == 2 and {run['run'] for run in view['runs']} == {RUN, '2026-11-08_1100-road-labeler-dry'}
    run = next(run for run in view['runs'] if run['run'] == RUN)
    assert (run['drives'], run['steps'], run['epochs'], run['checkpoint'], run['dry_run']) == (2, 1200, 2, True, False)
    assert run['types'] == {'steps': 9000, 'right': 0.91, 'person_steps': 600, 'right_on_person': 0.82,
                            'map_on_person': 0.7, 'writing': ['motorway']}
    heads = {head['head']: head for head in run['heads']}
    assert list(heads) == ['tunnel', 'parking', 'works']
    assert heads['tunnel'] == {'head': 'tunnel', 'inside': 400, 'outside': 9000, 'precision': 0.95, 'recall': 0.8,
                               'writes': True, 'why': None}
    assert (heads['parking']['writes'], heads['parking']['why']) == (False, 'not trained with both sides')
    assert (heads['works']['writes'], heads['works']['why']) == (False, '12 steps inside')
    dry = next(run for run in view['runs'] if run['dry_run'])
    assert not any(head['writes'] for head in dry['heads']) and dry['types']['writing'] == []
    # Never prepared yet: nothing it wrote, and no run in Speed review.
    assert run['prepared'] is None and not any(item['current'] for item in view['runs'])
    assert labeler_runs.trained_runs(root) == ['2026-11-08_1100-road-labeler-dry', RUN]
    assert labeler_runs.trained(root, RUN) and not labeler_runs.trained(root, 'brain-run')


def test_a_head_held_back_by_its_gate_says_why(root):
    labeler_run(root, tunnel_gate='sure on 4 steps of validation, 20 needed to tell whether it is right')
    heads = {head['head']: head for head in labeler_runs.summary(root)['runs'][0]['heads']}
    assert heads['tunnel']['writes'] is False and heads['tunnel']['why'].startswith('sure on 4 steps')


def test_the_card_says_what_each_prepare_wrote_and_which_run_speed_review_shows(root):
    older, newer = labeler_run(root, '2026-11-07_1200-road-labeler'), labeler_run(root)
    heads = {'tunnel': {'written': 3, 'removed': 1, 'kept': 1, 'left': 0},
             'works': {'written': 0, 'removed': 2, 'kept': 0, 'left': 0}, 'road_types': 120}
    # The older run prepared last: it is the one in Speed review, whatever the order of training.
    (newer / 'prepared.json').write_text(json.dumps({'run': RUN, 'at': '2026-11-08T13:00:00+01:00', 'drives': 4,
                                                     'failed': 0, 'writes': ['tunnel'], 'heads': heads}))
    (older / 'prepared.json').write_text(json.dumps({'run': older.name, 'at': '2026-11-08T14:00:00+01:00', 'drives': 4,
                                                     'failed': 1, 'writes': [], 'heads': {}}))
    runs = {run['run']: run for run in labeler_runs.summary(root)['runs']}
    assert runs[older.name]['current'] and not runs[RUN]['current']
    assert runs[RUN]['prepared'] == {'at': '2026-11-08T13:00:00+01:00', 'drives': 4, 'failed': 0, 'writes': ['tunnel'],
                                     'heads': {'tunnel': heads['tunnel'], 'works': heads['works']}, 'road_types': 120}
    # A broken prepared.json is as none: the card shows the run without it.
    (older / 'prepared.json').write_text('{"at": "not a time"')
    runs = {run['run']: run for run in labeler_runs.summary(root)['runs']}
    assert runs[older.name]['prepared'] is None and runs[RUN]['current']
