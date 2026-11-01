"""The scene network in the panel: "Train the scene network" (scripts/run_road_type.sh) from the Train tab, its
progress read from its log (nnslr_tools.job_progress), and its two networks, with memory and without, in Models
(nnslr_tools.model_roles). Every run is a synthetic folder whose numbers come from the real metric functions of
nnslr_tools.road_type_metrics: nothing is trained. numpy and road_type_metrics are imported inside the tests: collecting
the tests must leave numpy out (tests/cli/test_import_isolation.py).
"""
from __future__ import annotations

import json
import re

from nnslr_tools.job_progress import progress
from nnslr_tools.model_roles import ModelIndex, road_type_digest
from nnslr_tools.panel import Panel, _run_of
from nnslr_tools.road_types import ROAD_TYPES
from nnslr_tools.runs import RunsIndex
from test_panel import checkout, finished, full_log, root  # noqa: F401  (fixtures and helpers)

TOWN, RURAL, EXPRESSWAY, MOTORWAY, RAMP, PARKING = range(6)

LOG = """== Road type with memory ==
run:          2026-10-29_1200-road-type
== 1/8 Check the road types, the split and the pipeline (CPU-only) ==
dry run road_type_gru epoch 1/1 train_loss 1.7000 val_macro_f1 0.1000 val_accuracy 0.2000 5s
== 2/8 Explicit CUDA smoke test ==
== 3/8 Resize the frames to 256x144 ==
frame cache: 30 drives ready
== 4/8 Read the motorway signs as the car does and the car's motion (CPU) ==
== 5/8 Train with memory, the sign and the car's motion (GRU) ==
road_type_gru epoch 15/60 train_loss 0.8000 val_macro_f1 0.5500 val_accuracy 0.7000 12s
"""


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [{'state': 'pronta', 'confirmed': 4}] * 3, 'reviewed_routes': 3,
            'min_routes': 3, 'system': {'gpu': {'processes': gpu_processes}}}


def road_type_script(checkout):
    script = checkout / 'scripts' / 'run_road_type.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Road type with memory =="\necho "run:          $1"\n'
                      'echo "== 5/8 Train with memory and the sign (GRU) =="\n'
                      'echo "road_type_gru epoch 2/60 train_loss 1.0 val_macro_f1 0.4 val_accuracy 0.6 6s"\n')
    script.chmod(0o755)
    return script


# -- its progress -------------------------------------------------------------------------------------------------
def test_the_progress_counts_the_epochs_of_each_network_and_not_those_of_the_dry_run():
    result = progress('train_road_type', 'running', LOG)
    assert [(step['key'], step['state']) for step in result['steps']] == [
        ('checks', 'done'), ('frames', 'done'), ('signs', 'done'), ('road_type_gru', 'current'),
        ('road_type_gru_nomotion', 'todo'), ('road_type_frame', 'todo'), ('export', 'todo')]
    # 4 + 4 + 8 done, then 15 of the 60 epochs of the 30 of the network with memory and the sign.
    assert result['percent'] == 23.5
    assert result['detail'] == 'Train with memory, the sign and the motion · epoch 15 of 60'
    assert list(result['curves']) == ['road_type_gru']
    assert [point['epoch'] for point in result['curves']['road_type_gru']['points']] == [15]

    # The junction head's own stage after the road type's epochs is no curve of its own.
    later = LOG + ('road_type_gru stops early after epoch 52: no better validation road score for 15 epochs\n'
                   'road_type_gru junction epoch 3/30 train_loss 0.5000 val_junction_f1 0.3000 '
                   'val_false_passed_per_hour 20.0 5s\n'
                   "== 6/8 Train with memory, without the car's motion ==\n"
                   'road_type_gru_nomotion epoch 30/60 train_loss 0.9000 val_macro_f1 0.5000 val_accuracy 0.7000 10s\n')
    result = progress('train_road_type', 'running', later)
    assert [step['state'] for step in result['steps']] == ['done', 'done', 'done', 'done', 'current', 'todo', 'todo']
    assert result['percent'] == 59.5 and result['detail'] == 'Train with memory, no motion · epoch 30 of 60'
    assert list(result['curves']) == ['road_type_gru', 'road_type_gru_nomotion']
    assert [point['epoch'] for point in result['curves']['road_type_gru']['points']] == [15]
    # A run before the car's motion named its sixth network road_type_gru_nosign: its curve is still drawn.
    older = progress('train_road_type', 'running', LOG + 'road_type_gru_nosign epoch 3/60 train_loss 0.9 10s\n')
    assert list(older['curves']) == ['road_type_gru', 'road_type_gru_nosign']

    last = later + ('== 7/8 Train without memory ==\n'
                    'road_type_frame epoch 30/60 train_loss 0.9000 val_macro_f1 0.5000 val_accuracy 0.7000 10s\n')
    result = progress('train_road_type', 'running', last)
    assert [step['state'] for step in result['steps']] == ['done'] * 5 + ['current', 'todo']
    assert result['percent'] == 84.5 and result['detail'] == 'Train without memory · epoch 30 of 60'


# -- "Train the scene network" ------------------------------------------------------------------------------------
def test_the_road_type_is_trained_from_the_train_tab_once_the_road_types_are_found(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['train_road_type'] == {
        'label': 'Train the scene network', 'allowed': False, 'reason': 'scripts/run_road_type.sh is missing in this installation'}
    road_type_script(checkout)
    assert panel._action_states(ready_payload(), [])['train_road_type']['reason'] == \
        'no road types yet: Data › Roads, Find the road types'
    (root / 'derived' / 'road_types').mkdir(parents=True)
    (root / 'derived' / 'road_types' / 'r1.jsonl').write_text('{}\n')
    assert panel._action_states(ready_payload(gpu_processes=1), [])['train_road_type']['reason'] == \
        'the GPU is busy with another process'
    # Its labels are the map's road types: no confirmed sign, no reviewed drive needed.
    payload = {**ready_payload(), 'routes': [], 'reviewed_routes': 0}
    assert panel._action_states(payload, [])['train_road_type']['allowed'] is True

    monkeypatch.setattr(panel, 'activities', lambda: [])
    job = panel.start('train_road_type', None)
    done = finished(panel, job['id'])
    first = full_log(panel, job['id']).splitlines()[0].split()
    assert first[:2] == ['==', 'scripts/run_road_type.sh'] and re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-road-type', first[2])
    assert (done['state'], done['run'], done['label']) == ('ok', first[2], 'Train the scene network')
    assert _run_of(panel.logs / (job['id'] + '.log'), 'train_road_type') == first[2]


def test_a_road_type_training_started_from_a_terminal_has_the_progress_of_its_run_log(root, checkout):
    panel = Panel(root, repo=checkout)
    log = root / 'runs' / '2026-10-29_1200-road-type' / 'road-type.log'
    log.parent.mkdir(parents=True)
    log.write_text(LOG)
    payload = {'jobs': [], 'activities': [{'group': 'train', 'label': 'scene training', 'pid': 1}]}
    panel._terminal_training(payload)
    assert payload['activities'][0]['progress']['detail'] == 'Train with memory, the sign and the motion · epoch 15 of 60'


# -- its two networks in Models -----------------------------------------------------------------------------------
def report(predicted_motorway_after_ramp: bool) -> dict:
    """A partition of one drive: town, a ramp, then the motorway; the network says the motorway after the ramp, or
    rural for its first 20 seconds."""
    import numpy as np

    from nnslr_tools.road_type_metrics import after_merge, sequence_metrics
    labels = np.array([TOWN] * 60 + [RAMP] * 20 + [MOTORWAY] * 100)
    predicted = labels.copy()
    predicted[60:80] = MOTORWAY
    if not predicted_motorway_after_ramp:
        predicted[80:100] = RURAL
    sequences = [{'labels': labels, 'weights': np.ones(len(labels)), 'predicted': predicted}]
    return {**sequence_metrics(sequences, ROAD_TYPES), 'after_merge': after_merge(sequences, ROAD_TYPES)}


def result(memory: str, *, smoothed: bool = False) -> dict:
    gru = memory == 'gru'
    value = {'task': 'road_type', 'memory': memory, 'name': 'road_type_' + ('gru' if gru else 'frame'),
             'architecture': 'mobilenet_v3_small+' + ('gru' if gru else 'mlp'), 'classes': list(ROAD_TYPES),
             'best_epoch': 7, 'config': {'epochs': 60}, 'history': [{'epoch': n} for n in range(1, 23)],
             'parameters': 1101478 if gru else 1018406, 'flops_per_step': 83975936,
             'step_seconds': {'device': 'cuda', 'milliseconds': 3.9}, 'dry_run': False,
             'validation': report(gru), 'test': report(gru)}
    if smoothed:
        value['test_smoothed'] = report(True)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def test_both_networks_are_versions_of_the_road_type_with_their_numbers(tmp_path):
    root = tmp_path / 'data'
    run = root / 'runs' / '2026-10-29_1200-road-type' / 'road-type'
    write(run / 'gru' / 'road-type-best.pt', 'weights')
    write(run / 'gru' / 'result.json', result('gru'))
    write(run / 'frame' / 'road-type-best.pt', 'weights')
    write(run / 'frame' / 'result.json', result('none', smoothed=True))
    # The dry run of the CPU check is no version.
    write(run / 'dry-run' / 'road-type-best.pt', 'weights')
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    models = ModelIndex(runs)

    found = {entry['part']: entry for entry in models.artifacts(('road_type',))['road_type']}
    assert sorted(found) == ['road-type/frame', 'road-type/gru']
    gru, frame = found['road-type/gru'], found['road-type/frame']
    assert (gru['id'], gru['label'], gru['memory'], gru['finished'], gru['onnx']) == (
        'road_type:2026-10-29_1200-road-type/road-type/gru', 'With memory', 'gru', True, False)
    assert (gru['best_epoch'], gru['epochs'], gru['epochs_run'], gru['parameters'], gru['step_ms']) == (7, 60, 22, 1101478, 3.9)
    assert gru['classes'] == list(ROAD_TYPES)
    # With memory the motorway after the ramp is right; without it the first 20 of its 60 seconds are rural.
    assert gru['test']['after_merge'] == {'merges': 1, 'seconds': 60, 'accuracy': 1.0, 'accuracy_debounced': 1.0}
    assert frame['test']['after_merge']['merges'] == 1 and round(frame['test']['after_merge']['accuracy'], 3) == 0.667
    assert gru['test']['transitions']['town_entry'] == {'count': 0, 'found': 0, 'median_delay': None}
    assert gru['test']['transitions']['ramp_to_motorway']['count'] == 1
    assert gru['test']['classes']['expressway'] == {'f1': None, 'support': 0}
    assert frame['test_smoothed'] is not None and gru['test_smoothed'] is None

    write(run / 'export' / 'road-type.onnx', 'onnx')
    write(run / 'export' / 'road-type.onnx.contract.json', {'format': 'road-type-step'})
    found = {entry['part']: entry for entry in models.artifacts(('road_type',))['road_type']}
    assert (found['road-type/gru']['onnx'], found['road-type/frame']['onnx']) == (True, False)

    chosen = models.set_active('road_type', gru['id'], who='admin', via='sign-in')
    assert chosen['id'] == gru['id']
    roles = {role['role']: role for role in models.index()['roles']}
    assert roles['road_type']['label'] == 'Scene' and roles['road_type']['active']['id'] == gru['id']


def test_a_result_of_another_task_is_no_road_type():
    try:
        road_type_digest({'task': 'detector'})
    except ValueError:
        return
    raise AssertionError('a detector result was read as a road type')


def test_the_second_version_shows_its_sign_input_and_when_the_ramp_limit_ends(tmp_path):
    root = tmp_path / 'data'
    run = root / 'runs' / '2026-10-29_1500-road-type' / 'road-type'
    release = {'merges': 7, 'held': 6, 'held_by_ramp': 4, 'held_by_sign': 3, 'on_ramp': 1, 'in_time': 4, 'late': 1,
               'not_held': 1, 'median_release': 11.0, 'releases': [10, 11, -3, None, 12, 9, None]}
    for folder, signs in (('gru', True), ('gru-nosign', False)):
        write(run / folder / 'road-type-best.pt', 'weights')
        write(run / folder / 'result.json', {**result('gru'), 'name': 'road_type_gru' + ('' if signs else '_nosign'),
                                             'signs': signs, 'release': {'validation': release, 'test': release,
                                                                         'merges': [{'route': 'r'}]}})
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    found = {entry['part']: entry for entry in ModelIndex(runs).artifacts(('road_type',))['road_type']}
    assert sorted(found) == ['road-type/gru', 'road-type/gru-nosign']
    sign, nosign = found['road-type/gru'], found['road-type/gru-nosign']
    assert (sign['signs'], sign['label'], sign['detail']) == (True, 'With memory and the sign',
                                                           'GRU over the seconds and the motorway sign')
    assert (nosign['signs'], nosign['label']) == (False, 'With memory, no sign')
    assert nosign['id'] == 'road_type:2026-10-29_1500-road-type/road-type/gru-nosign'
    assert sign['release']['test'] == {'merges': 7, 'held': 6, 'on_ramp': 1, 'in_time': 4, 'late': 1, 'not_held': 1,
                                       'median_release': 11.0}
    assert road_type_digest(result('none'))['release'] is None


def test_the_junction_head_shows_its_f1_and_its_false_passed_per_hour(tmp_path):
    numbers = {'events': 30, 'recall_2s': 0.8, 'recall_4s': 0.9, 'precision_2s': 0.75, 'f1_2s': 0.7742, 'false': 5,
               'false_per_hour': 2.5, 'false_per_10km': 0.6, 'town': {'false': 4, 'per_hour': 4.0, 'per_10km': 1.2},
               'outside': {'false': 1, 'per_hour': 1.0, 'per_10km': 0.2}}
    value = result('gru')
    for partition in ('validation', 'test'):
        value[partition]['junctions'] = {'passed': 28, 'hours': 2.0, 'km': 80.0,
                                         'decodes': {'passed': {'0.5': numbers}, 'exit': {'0.5': {}}}}
    value.update(junctions=True, junction_operating={'decode': 'passed', 'threshold': 0.5, 'f1_2s': 0.7742},
                 junction_needed={'junctions': 20, 'needed': 6, 'caught_2s': 4, 'caught_4s': 5, 'list': [{}]})
    root = tmp_path / 'data'
    run = root / 'runs' / '2026-10-29_1600-road-type' / 'road-type'
    write(run / 'gru' / 'road-type-best.pt', 'weights')
    write(run / 'gru' / 'result.json', value)
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    entry = ModelIndex(runs).artifacts(('road_type',))['road_type'][0]
    assert entry['junctions'] == {
        'decode': 'passed', 'threshold': 0.5,
        'validation': {'passed': 28, 'hours': 2.0, 'km': 80.0, 'f1_2s': 0.7742, 'recall_2s': 0.8, 'recall_4s': 0.9,
                       'precision_2s': 0.75, 'false': 5, 'false_per_hour': 2.5, 'false_per_10km': 0.6,
                       'town_per_hour': 4.0, 'outside_per_hour': 1.0},
        'test': entry['junctions']['validation'],
        'needed': {'junctions': 20, 'needed': 6, 'caught_2s': 4, 'caught_4s': 5},
        'branch': None, 'stage_epoch': None, 'turn_rule': None}
    # The second version had no junction head; a point that is not a decode is not shown.
    assert road_type_digest(result('gru'))['junctions'] is None
    assert road_type_digest({**value, 'junction_operating': {'decode': 'enter', 'threshold': 0.5}})['junctions'] is None


def test_the_fourth_version_shows_the_cars_motion_its_junction_branch_and_the_turn_rule(tmp_path):
    numbers = {'events': 30, 'recall_2s': 0.5, 'recall_4s': 0.6, 'precision_2s': 0.5, 'f1_2s': 0.5, 'false': 5,
               'false_per_hour': 12.5, 'false_per_10km': 1.6, 'town': {'false': 4, 'per_hour': 30.0, 'per_10km': 3.2},
               'outside': {'false': 1, 'per_hour': 4.0, 'per_10km': 0.2}}
    junctions = {'passed': 28, 'hours': 2.0, 'km': 80.0, 'decodes': {'passed': {'0.5': numbers}, 'exit': {'0.5': {}}}}
    value = result('gru')
    for partition in ('validation', 'test'):
        value[partition]['junctions'] = junctions
    turn = {'rule': {'degrees': 60.0, 'radius_m': 40.0, 'min_dps': 2.0}, 'metrics': junctions,
            'passages': {'passed': 28, 'yaw_known': 28, 'turned_60': 9, 'turned_30': 12},
            'network_by_turn': {'decode': 'passed', 'threshold': 0.5, 'turned': {'passed': 9, 'caught_2s': 7,
                                                                                 'caught_4s': 8},
                                'straight': {'passed': 19, 'caught_2s': 5, 'caught_4s': 6}}}
    value.update(junctions=True, motion=True, speed_source='v_ego', junction_branch='separate',
                 junction_operating={'decode': 'passed', 'threshold': 0.5},
                 junction_stage={'branch': 'separate', 'best_epoch': 12, 'history': []},
                 turn_rule={'test': turn})
    root = tmp_path / 'data'
    run = root / 'runs' / '2026-10-29_1700-road-type-motion' / 'road-type'
    for folder in ('gru', 'gru-nomotion'):
        write(run / folder / 'road-type-best.pt', 'weights')
        write(run / folder / 'result.json', value if folder == 'gru' else {**value, 'motion': False,
                                                                          'junction_branch': 'shared'})
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    found = {entry['part']: entry for entry in ModelIndex(runs).artifacts(('road_type',))['road_type']}
    assert sorted(found) == ['road-type/gru', 'road-type/gru-nomotion']
    gru, nomotion = found['road-type/gru'], found['road-type/gru-nomotion']
    assert (gru['label'], gru['motion'], gru['speed_source']) == ('With memory, sign and motion', True, 'v_ego')
    assert (nomotion['label'], nomotion['motion']) == ('With memory, no motion', False)
    assert nomotion['id'] == 'road_type:2026-10-29_1700-road-type-motion/road-type/gru-nomotion'
    assert (gru['junctions']['branch'], gru['junctions']['stage_epoch']) == ('separate', 12)
    rule = gru['junctions']['turn_rule']
    assert (rule['degrees'], rule['radius_m'], rule['passed'], rule['turned_60'], rule['turned_30']) == (60.0, 40.0, 28, 9, 12)
    assert rule['numbers']['town_per_hour'] == 30.0 and rule['numbers']['recall_4s'] == 0.6
    assert rule['caught'] == {'turned': {'passed': 9, 'caught_2s': 7, 'caught_4s': 8},
                              'straight': {'passed': 19, 'caught_2s': 5, 'caught_4s': 6}}
    # A version before the motion: no turn rule, its speed from the GPS.
    older = road_type_digest({**value, 'motion': None, 'speed_source': None, 'turn_rule': None,
                              'junction_branch': None, 'junction_stage': None})
    assert (older['motion'], older['speed_source']) == (False, 'gps')
    assert (older['junctions']['turn_rule'], older['junctions']['branch']) == (None, None)


def test_the_scene_network_reads_the_signs_with_a_chosen_classifier_and_takes_the_board_input(root, checkout, monkeypatch):
    import pytest

    from nnslr_tools.panel import PanelError, _settings_of
    from test_panel import finished, full_log
    from test_panel_runs import make_classifier_run, write
    script = road_type_script(checkout)
    script.write_text(script.read_text() + 'echo "signs=${NNSLR_ROAD_TYPE_CLASSIFIER:-car} boards=${NNSLR_ROAD_TYPE_BOARDS:-auto}"\n')
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', lambda: [])
    done = make_classifier_run(root, 'c1')
    make_classifier_run(root, 'c2', upto='classifier')
    write(root / 'runs/r1/reader/reader-best.pt', 'weights')
    # The car's classifier and the board input when it read the boards, as before: nothing in the environment.
    job = panel.start('train_road_type', {'classifier': 'car', 'boards': 'auto'})
    finished(panel, job['id'])
    assert re.search(r'^signs=car boards=auto$', full_log(panel, job['id']), re.M)
    import time
    time.sleep(1.1)  # a job's log is named by its second
    # A finished classifier with "not a sign" of Models, by its id: its reader-best.pt; the board input always.
    job = panel.start('train_road_type', {'classifier': 'classifier:c1/classifier', 'boards': 'on'})
    finished(panel, job['id'])
    assert re.search(r'^signs=' + re.escape(str((done / 'classifier/reader-best.pt').resolve())) + ' boards=1$',
                     full_log(panel, job['id']), re.M)
    refused = {('classifier:nothing/classifier', None): 'Unknown classifier to read the signs: choose one with “not a sign” of Models.',
               ('classifier:r1/reader', None): 'Unknown classifier to read the signs: choose one with “not a sign” of Models.',
               ('/etc/passwd', None): 'Unknown classifier to read the signs: choose one with “not a sign” of Models.',
               ('classifier:c2/classifier', None): 'That classifier has not finished its training: choose a finished one.',
               (None, 'yes'): 'The board input is auto, on or off.'}
    for (classifier, boards), message in refused.items():
        with pytest.raises(PanelError) as error:
            panel.start('train_road_type', {'classifier': classifier, 'boards': boards})
        assert (error.value.status, error.value.message) == (400, message), (classifier, boards)
    # Only the scene network takes them.
    for key in ('classifier', 'boards'):
        with pytest.raises(PanelError) as error:
            panel.start('train_decider', {key: 'car'})
        assert (error.value.status, error.value.message) == (400, 'Only "Train the scene network" takes a ' + key + '.')
    # The job's chips: the classifier that read the signs, and the board input.
    log = root / 'road-type.log'
    log.write_text('== scripts/run_road_type.sh r\nrun:          r\nsettings:     --epochs 60 --boards\n'
                   'signs read:   /d/runs/2026-11-01_1812-classifier/classifier/reader-best.pt\n')
    assert _settings_of(log, 'train_road_type') == ['60 epochs', 'signs by 2026-11-01_1812-classifier', 'board input']
    log.write_text('== scripts/run_road_type.sh r\nrun:          r\nsettings:     --epochs 60 --no-boards\n'
                   'signs read:   by the classifier of the car\n')
    assert _settings_of(log, 'train_road_type') == ['60 epochs', 'no board input']


def test_the_classifier_and_the_board_input_reach_the_panel_over_http(root, checkout):
    from review_helpers import RunningServer, json_of

    from test_panel import finished, full_log
    from test_panel_runs import make_classifier_run
    script = road_type_script(checkout)
    script.write_text(script.read_text() + 'echo "signs=${NNSLR_ROAD_TYPE_CLASSIFIER:-car} boards=${NNSLR_ROAD_TYPE_BOARDS:-auto}"\n')
    done = make_classifier_run(root, 'c1')
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    try:
        # Train › Scene's own button: a wrong choice is refused, never dropped (nothing starts).
        status, _, data = server.request('POST', '/api/panel/run/train_road_type', body={'boards': 'x'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, _, data = server.request('POST', '/api/panel/run/train_road_type',
                                         body={'classifier': 'classifier:c1/classifier', 'boards': 'on'})
        assert status == 200, data
        job = json_of(data)['id']
        finished(server.app.panel, job)
        assert re.search(r'^signs=' + re.escape(str((done / 'classifier/reader-best.pt').resolve())) + ' boards=1$',
                         full_log(server.app.panel, job), re.M)
    finally:
        server.close()
