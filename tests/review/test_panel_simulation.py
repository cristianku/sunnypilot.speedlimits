"""Models › Drive on the server: the replay of a drive as an action of the panel (its command line, its refusals, its
card), its progress from the log, the three addresses the page reads and writes, and the tab itself. The helper is a
stand-in in the test checkout and the replays are records written by hand: no video is decoded, no network runs, no
card is touched."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from review_helpers import ADMIN, OTHER_ROUTE, ROUTE, TOKEN, RunningServer, json_of, sign_in

from nnslr_tools import drive_review, panel as panel_module, simulation
from nnslr_tools.job_progress import progress, with_step_progress
from nnslr_tools.pages import CONTROL_TABS, OLD_ADDRESSES, SECTIONS, sub_nav
from nnslr_tools.panel import Panel, PanelError
from test_guest_reviewers import fast_scrypt, give_all, outside, proof, proxied  # noqa: F401  (fixtures and helpers)
from test_panel import checkout, finished, full_log, only_this_checkout, root  # noqa: F401  (fixtures and helpers)
from test_panel_gpus import (APPS, FREE, GPU_SCRIPT, cards, fake_nvidia, holders, opened, panel, reasons, record,  # noqa: F401
                             refused, said)
from test_panel_training import signnet_script

NS = 1_000_000_000
# The start of the drive's logs on the comma's clock, and a frame read every 0.2 s from its tenth second.
T0 = 1_000 * NS
STEP = NS // 5
FIRST = T0 + 10 * NS
CUTTER = 'cutter:s1/signnet'
NETWORKS = [{'role': 'finder', 'file': 'signnet.onnx', 'id': CUTTER, 'sha256': 'a' * 64},
            {'role': 'classifier', 'file': 'classifier.onnx', 'id': 'classifier:c1/classifier', 'sha256': 'b' * 64}]
# What prints its arguments, then the card it was given.
SCRIPT = '#!/usr/bin/env bash\necho "arguments: $*"\necho "gpu=${NNSLR_GPU_INDEX:-none}"\n'

HEAD = f"""== helpers/simulate_drive.sh --set=in-use --route={ROUTE}
== 1/2 Get the networks ready ==
the versions in use are exported already: {CUTTER}
== 2/2 Replay the drive ==
"""


@pytest.fixture(autouse=True)
def no_car_clone(tmp_path, monkeypatch):
    """The sunnypilot clone of the tests: an empty folder, never the one next to this checkout."""
    monkeypatch.setenv(simulation.SUNNYPILOT_ENV, str(tmp_path / 'sunnypilot'))
    return tmp_path / 'sunnypilot'


def ready_payload(gpu_processes=0, used_mb=None):
    """A server with no confirmed frame and no reviewed drive: a replay needs neither."""
    memory = {} if used_mb is None else {'memory_used_mb': used_mb, 'memory_total_mb': 32768}
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes, **memory}}}


def simulate_script(checkout, text=SCRIPT):  # noqa: F811
    script = checkout / panel_module.SIMULATE_SCRIPT
    script.write_text(text)
    script.chmod(0o755)
    return script


def known(root, route=ROUTE):  # noqa: F811
    """A drive the portal has: its folder of the comma's files."""
    (root / 'raw' / 'routes' / route).mkdir(parents=True, exist_ok=True)


def scanned(root, route=ROUTE):  # noqa: F811
    """A drive whose logs are read: the summary of its scan says when they start."""
    known(root, route)
    path = drive_review.db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for statement in drive_review.SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.execute("INSERT INTO scans (route, state, summary) VALUES (?, 'done', ?)",
                       (route, json.dumps({'t0_ns': T0, 'span_ns': [T0, T0 + 60 * NS]})))
    connection.commit()
    connection.close()


def replay(set_key='in-use', kph=(50, 50, 50, 80, 80), route=ROUTE, networks=NETWORKS):
    """A replay as simulation.replay writes it (the module's docstring), five frames read: an 80 sign of our road in
    view on the third and the fourth, read on the fourth, where the speed to hold becomes 80."""
    rows = len(kph)
    out = {field: [default] * rows for field, default in simulation.FIELDS.items()}
    out['speedLimit'] = [value / 3.6 for value in kph]
    out['signRead'] = [False, False, False, True, False]
    for name, value in (('rawSignSeen', True), ('rawSignClass', 'maximum_speed:80'), ('rawSignValue', 80.0),
                        ('rawSignSpeed', 80.0), ('rawSignConfidence', 0.97), ('rawSignRoad', 'own_road')):
        out[name] = [value if row in (2, 3) else simulation.FIELDS[name] for row in range(rows)]
    return {'version': simulation.VERSION, 'route': route, 'set': set_key, 'name': 'networks of the test',
            'made_at': '2026-11-10T10:00:00+00:00', 'device': 'cuda', 'frame': [1344, 760], 'networks': networks,
            'code': simulation.code_hashes(), 'videos': [[0, 1000, 1]],
            # The release's settings: none in the test's clone, the car's defaults in the export of the versions in use.
            'settings': simulation.wanted_settings(Path('/nonexistent'), networks) if set_key == 'in-use' else {},
            'frames': {'t_ns': [FIRST + row * STEP for row in range(rows)], 'segment': [0] * rows,
                       'frame': [200 + 4 * row for row in range(rows)]},
            'out': out, 'signs': {'2': [[600.0, 300.0, 640.0, 340.0, 0.9, 'maximum_speed:80', 0.97, 'own_road']]},
            'in': {**{name: [0.0] * rows for name in simulation.INPUTS}, 'brain': {'0': [0.1, 0.2], '3': [0.3, 0.4]}}}


def release(directory, source=None):
    """A release of networks (speed_limits.json, format 2, and a contract each) in a folder, as Publish leaves it in the
    sunnypilot clone and nnslr simulate-set in derived/simulation/sets/in-use (source.json: the versions it is of, and
    the car's settings of the data root in its manifest, as the export writes them)."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'format': simulation.MANIFEST_FORMAT, 'version': '2026-11-08'}
    if source is not None:
        manifest['settings'] = simulation.wanted_settings(directory.parents[3], NETWORKS)
    for network in NETWORKS:
        contract = network['role'] + '.json'
        manifest[network['role']] = {'file': network['file'], 'contract': contract}
        (directory / contract).write_text(json.dumps({'checkpoint': network['id'], 'onnx_sha256': network['sha256']}))
    (directory / simulation.MANIFEST).write_text(json.dumps(manifest))
    if source is not None:
        (directory / simulation.SOURCE_FILE).write_text(json.dumps(source))


def in_use(root, **ids):  # noqa: F811
    """The versions in use on Models (models/active.json), by role."""
    path = root / 'models' / 'active.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': 1, **{role: {'id': model, 'who': 'c', 'via': 'sign-in',
                                                               'at': '2026-11-10T07:21:11+00:00'}
                                                        for role, model in ids.items()}}))


# ---------- the action ----------

def test_the_replay_needs_its_script_and_room_on_a_gpu_and_no_confirmed_frame(root, checkout):  # noqa: F811
    panel = Panel(root, repo=checkout)  # noqa: F811
    assert panel._action_states(ready_payload(), [])['simulate'] == {
        'label': 'Replay a drive', 'allowed': False, 'reason': 'helpers/simulate_drive.sh is missing in this installation'}
    with pytest.raises(PanelError) as missing:
        panel.start('simulate', {'route': ROUTE, 'set': 'car'})
    assert (missing.value.status, missing.value.reason) == (403, 'actions_disabled')
    simulate_script(checkout)
    # It shares the card with what else runs there (strata leaves about 3 GB of each V100)...
    assert panel._action_states(ready_payload(gpu_processes=1, used_mb=29434), [])['simulate'] == {
        'label': 'Replay a drive', 'allowed': True, 'reason': ''}
    # ...and never moves to the CPU: a card without 2 GB free closes the button, and says why.
    assert panel._action_states(ready_payload(gpu_processes=1, used_mb=31500), [])['simulate'] == {
        'label': 'Replay a drive', 'allowed': False,
        'reason': 'A replay needs about 2 GB of a GPU and the freest card has only 1.2 GB free (GPU 0)'}
    assert panel._action_states(ready_payload(), [])['simulate'] == {'label': 'Replay a drive', 'allowed': True, 'reason': ''}
    read_only = panel._action_states({**ready_payload(), 'actions_enabled': False}, [])
    assert read_only['simulate']['reason'] == 'the panel is read-only'


@pytest.mark.parametrize('options, arguments', [
    ({'route': ROUTE, 'set': 'in-use'}, f'--set=in-use --route={ROUTE}'),
    ({'route': ROUTE, 'set': 'car'}, f'--set=car --route={ROUTE}'),
    ({'certified': '1', 'set': 'car'}, '--set=car --certified'),
    ({'certified': '1', 'set': 'in-use'}, '--set=in-use --certified'),
])
def test_a_drive_or_the_certified_drives_are_replayed_with_the_set_chosen(root, checkout, options, arguments):  # noqa: F811
    simulate_script(checkout)
    known(root)
    simulation.write_result(root, replay())
    simulation.certify(root, ROUTE, 'in-use', 'cristian')
    panel = Panel(root, repo=checkout)  # noqa: F811
    job = panel.start('simulate', options)
    assert (job['action'], job['label']) == ('simulate', 'Replay a drive')
    assert finished(panel, job['id'])['state'] == 'ok'
    lines = full_log(panel, job['id']).splitlines()
    assert lines[0] == '== helpers/simulate_drive.sh ' + arguments
    assert 'arguments: ' + arguments in lines


@pytest.mark.parametrize('options, status, message', [
    # The set of networks: always said, one of the two.
    ({}, 400, 'Choose the networks of the replay: car or in-use.'),
    ({'route': ROUTE}, 400, 'Choose the networks of the replay: car or in-use.'),
    ({'route': ROUTE, 'set': 'certified'}, 400, 'Choose the networks of the replay: car or in-use.'),
    ({'route': ROUTE, 'set': 'car;rm'}, 400, 'Choose the networks of the replay: car or in-use.'),
    ({'route': ROUTE, 'set': ['car']}, 400, 'The set is a string.'),
    # One drive, or every certified one: one of the two, never both, never neither.
    ({'set': 'car'}, 400, 'Replay one drive (route) or every certified drive (certified): one of the two.'),
    ({'set': 'car', 'route': ROUTE, 'certified': '1'}, 400,
     'Replay one drive (route) or every certified drive (certified): one of the two.'),
    ({'set': 'car', 'certified': 'yes'}, 400, 'Every certified drive is "1", or none.'),
    ({'set': 'car', 'certified': True}, 400, 'The certified is a string.'),
    # The drive: one the portal has, by its id and nothing else.
    ({'set': 'car', 'route': OTHER_ROUTE}, 400, f'Unknown drive: {OTHER_ROUTE}.'),
    ({'set': 'car', 'route': ROUTE + ' --again'}, 400, f'Unknown drive: {ROUTE} --again.'),
    ({'set': 'car', 'route': '../' + ROUTE}, 400, f'Unknown drive: ../{ROUTE}.'),
    ({'set': 'car', 'route': 7}, 400, 'The route is a string.'),
    # No certified drive yet.
    ({'set': 'in-use', 'certified': '1'}, 409, 'Certify a drive first.'),
    # It has no dry run on the CPU, and none of the other actions' options.
    ({'set': 'car', 'route': ROUTE, 'dry': 'on'}, 400, '"Replay a drive" does not take dry.'),
    ({'set': 'car', 'route': ROUTE, 'provider': 'x'}, 400, '"Replay a drive" does not take a provider.'),
])
def test_a_replay_is_refused_before_anything_starts(root, checkout, options, status, message):  # noqa: F811
    simulate_script(checkout)
    known(root)
    panel = Panel(root, repo=checkout)  # noqa: F811
    with pytest.raises(PanelError) as error:
        panel.start('simulate', options)
    assert (error.value.status, error.value.message) == (status, message)
    assert error.value.reason == ('not_available' if status == 409 else 'bad_option')
    assert panel.jobs() == []


@pytest.mark.parametrize('key, value', [('set', 'car'), ('certified', '1')])
def test_its_choices_are_refused_on_another_action(root, checkout, key, value):  # noqa: F811
    with pytest.raises(PanelError) as error:
        Panel(root, repo=checkout).start('train_decider', {key: value})
    assert (error.value.reason, error.value.message) == ('bad_option', f'"Train applicability" does not take {key}.')


def test_a_replay_takes_the_freest_card_beside_a_training_one_at_a_time(panel, root, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    simulate_script(checkout, GPU_SCRIPT)
    known(root)
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    # The other card, as free and held by nothing: the script and CUDA are given that one only.
    job = panel.start('simulate', {'route': ROUTE, 'set': 'in-use'})
    assert said(panel, job, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    assert said(panel, job, 'progress=') == f'progress={root}/logs/progress-{job["id"]}.json'
    assert record(panel, job) == {'gpus': [1], 'own_progress': True}
    assert holders(panel) == [(0, False, [{'job': cutter['id'], 'label': 'A cutter training'}]),
                              (1, False, [{'job': job['id'], 'label': 'A replay of a drive'}])]
    assert [(item['label'], item['job'], item['gpus']) for item in panel.activities() if item['label'] == 'drive replay'] == [
        ('drive replay', job['id'], [1])]
    # One replay at a time: a second one would write the same results and the same export of the versions in use.
    assert refused(panel, 'simulate', {'route': ROUTE, 'set': 'car'}) == (
        409, 'busy', 'A replay of a drive is running: wait for it to finish.')
    assert reasons(panel)['simulate'] == {'label': 'Replay a drive', 'allowed': False,
                                          'reason': 'A replay of a drive is running'}
    # A training shares a card with a training or a replay (issue #41), and waits, told what holds them, when none
    # has the room.
    assert reasons(panel)['train_cutter']['reason'] == ''
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(30000, 30000), ''))
    assert reasons(panel)['train_cutter']['reason'] == (
        'both GPUs are busy: a cutter training on GPU 0, a replay of a drive on GPU 1')


def test_a_replay_shares_the_card_with_the_most_free_memory(panel, root, checkout, monkeypatch):  # noqa: F811
    # Both cards held by a program the server cannot name (strata), the second with more room.
    held = ('0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 0, 29434, 32768, 31, 25.1\n'
            '1, 00000000:C1:00.0, Tesla V100-PCIE-32GB, 0, 28000, 32768, 30, 24.0\n')
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(held, '00000000:81:00.0, 77, python, 29000\n' + APPS))
    signnet_script(checkout, GPU_SCRIPT)
    simulate_script(checkout, GPU_SCRIPT)
    known(root)
    assert reasons(panel)['simulate'] == {'label': 'Replay a drive', 'allowed': True, 'reason': ''}
    # A training still waits for a card of its own.
    assert reasons(panel)['train_cutter']['reason'] == 'both GPUs are busy with a program the server cannot name'
    job = panel.start('simulate', {'route': ROUTE, 'set': 'in-use'})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    assert record(panel, job)['gpus'] == [1]


def test_without_room_on_a_card_no_replay_starts_and_none_moves_to_the_cpu(panel, root, checkout, monkeypatch):  # noqa: F811
    full = ('0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 90, 31800, 32768, 60, 200.0\n'
            '1, 00000000:C1:00.0, Tesla V100-PCIE-32GB, 87, 31500, 32768, 54, 160.3\n')
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(full, '00000000:81:00.0, 77, python, 29000\n' + APPS))
    simulate_script(checkout, GPU_SCRIPT)
    known(root)
    why = 'A replay needs about 2 GB of a GPU and the freest card has only 1.2 GB free (GPU 1)'
    assert reasons(panel)['simulate'] == {'label': 'Replay a drive', 'allowed': False, 'reason': why}
    assert refused(panel, 'simulate', {'route': ROUTE, 'set': 'in-use'}) == (
        409, 'gpu_taken', why + ': free some memory on a card, then try again.')
    assert panel.jobs() == [] and not list(panel.logs.glob('*-simulate.log'))


def test_a_replay_from_a_terminal_holds_its_card_and_is_one_replay():
    assert ('train', 'drive replay', panel_module.SIMULATE_SCRIPT) in panel_module.ACTIVITIES
    assert panel_module.TERMINAL_KINDS['drive replay'] == 'simulate'
    assert panel_module.ACTIVITY_NAMES['drive replay'] == panel_module.RUNNING_NAMES['simulate'] == 'a replay of a drive'
    assert panel_module._pair_apart('simulate', 'simulate')
    # Beside any training that takes one card; never beside the import or the whole pipeline.
    for other in ('train_cutter', 'train_classifier', 'train_decider', 'train_brain', 'regression',
                  'road_labeler_train', 'road_labeler_predict'):
        assert not panel_module._pair_apart('simulate', other), other
    for other in panel_module.TRAIN_ALONE:
        assert panel_module._pair_apart('simulate', other), other


# ---------- its progress ----------

def test_the_progress_of_one_drive_goes_minute_by_minute():
    result = progress('simulate', 'running', HEAD)
    assert [(step['key'], step['label'], step['state']) for step in result['steps']] == [
        ('networks', 'Get the networks ready', 'done'), ('replay', 'Replay the drive', 'current')]
    assert (result['percent'], result['step_progress']) == (10.0, None)
    # Before the log counts anything, the report of nnslr simulate (nnslr_tools.progress) says how far the drive is.
    reported = with_step_progress(result, 'simulate', {'task': 'simulate', 'done': 1, 'total': 4, 'text': ROUTE})
    assert (reported['percent'], reported['step_progress']) == (
        32.5, {'done': 1, 'total': 4, 'percent': 25.0, 'text': ROUTE})
    log = HEAD + f'drives: 1\n{ROUTE}: in-use, SignNet + Classifier, 4 minutes of video\nminute 1/4: 300 frames read\n'
    result = progress('simulate', 'running', log)
    assert result['percent'] == 32.5
    assert result['step_progress'] == {'done': 0.25, 'total': 1, 'percent': 25.0, 'text': 'minute 2 of 4'}
    # One drive: no count of drives under it.
    assert result['detail'] == ''
    done = progress('simulate', 'ok', log + f'minute 4/4: 300 frames read\ndrive 1/1 {ROUTE}: 1200 frames read, 4 signs read\n')
    assert done['percent'] == 100.0 and {step['state'] for step in done['steps']} == {'done'}


def test_the_progress_of_the_certified_drives_goes_drive_by_drive_and_never_back():
    log = (HEAD + f'drives: 4\n{ROUTE}: in-use, SignNet + Classifier, 4 minutes of video\n'
           + ''.join(f'minute {minute}/4: 300 frames read\n' for minute in range(1, 5))
           + f'drive 1/4 {ROUTE}: 1200 frames read, 4 signs read\n'
           + f'drive 2/4 {OTHER_ROUTE}: replayed already with these networks\n')
    result = progress('simulate', 'running', log)
    assert result['step_progress'] == {'done': 2, 'total': 4, 'percent': 50.0, 'text': 'drive 3 of 4'}
    assert (result['percent'], result['detail']) == (55.0, '2 of 4 drives done')
    # The last report is still the first drive's, all of its minutes: it must not count as the third drive done.
    stale = with_step_progress(result, 'simulate', {'task': 'simulate', 'done': 4, 'total': 4, 'text': ROUTE})
    assert (stale['percent'], stale['step_progress']) == (result['percent'], result['step_progress'])
    result = progress('simulate', 'running', log + 'minute 1/2: 300 frames read\n')
    assert result['step_progress'] == {'done': 2.5, 'total': 4, 'percent': 62.5, 'text': 'drive 3 of 4'}
    assert result['percent'] == pytest.approx(66.2, abs=0.1)


@pytest.mark.parametrize('line, sentence', [
    ('helpers/simulate_drive.sh: GPU 1 has only 900 MB free and the replay needs 2048: free some memory on it (the '
     'replay never runs on the CPU)',
     'No card had the room for the replay: it needs about 2 GB of a V100 and never runs on the CPU. Free some memory on '
     'a card, then try again.'),
    ('helpers/simulate_drive.sh: nvidia-smi sees no GPU to replay on: the replay never runs on the CPU',
     'No GPU for the replay: it always runs on a V100, never on the CPU. Free a card, then try again.'),
    ('helpers/simulate_drive.sh: no nvidia-smi, no GPU to replay on: the replay never runs on the CPU',
     'No GPU for the replay: it always runs on a V100, never on the CPU. Free a card, then try again.'),
])
def test_a_replay_stopped_for_the_card_says_so_in_a_sentence(line, sentence):
    result = progress('simulate', 'failed', HEAD + line + '\n\n== finished with code 3\n')
    assert result['failure'] == sentence
    assert [step['state'] for step in result['steps']] == ['done', 'failed']


# ---------- the page's addresses ----------

@pytest.fixture
def site(root, checkout):  # noqa: F811
    """The site on the test checkout's panel, a drive with its logs read and its replay with the versions in use."""
    simulate_script(checkout)
    scanned(root)
    simulation.write_result(root, replay())
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    yield server
    server.close()


def test_the_list_has_the_sets_and_every_drive_with_its_replays(site, root, no_car_clone):  # noqa: F811
    in_use(root, cutter=CUTTER)
    release(simulation.package_path() / 'models')
    status, _, data = site.request('GET', '/api/simulation/drives')
    assert status == 200, data
    answer = json_of(data)
    assert answer['ok'] is True and list(answer['sets']) == ['car', 'in-use']
    assert answer['sets']['car'] == {'label': 'On the car', 'name': 'speed-limits 2026-11-08', 'networks': NETWORKS,
                                     'settings': {}, 'ready': True, 'reason': '', 'type_head': None}
    # The versions in use on Models, by the car's names of the networks: not exported yet.
    assert answer['sets']['in-use'] == {
        'label': 'In use', 'name': '', 'networks': [], 'settings': {}, 'ready': False, 'wanted': {'finder': CUTTER},
        'reason': 'Not exported yet: Compute exports the versions in use first.', 'type_head': None}
    assert [row['route'] for row in answer['drives']] == [ROUTE]
    row = answer['drives'][0]
    # A row of Speed review's list, with what the drive has here.
    assert {'title', 'when', 'day', 'minutes', 'dashcam'} <= set(row)
    made = row['sets']['in-use']
    assert {key: made[key] for key in ('made_at', 'name', 'frames', 'current')} == {
        'made_at': '2026-11-10T10:00:00+00:00', 'name': 'networks of the test', 'frames': 5, 'current': False}
    # what changed since: the versions in use it waits to export
    assert made['changed'] and all(set(one) == {'what', 'was', 'now'} for one in made['changed'])
    assert (row['certified'], row['regression']) == (None, {})
    # The export of the versions in use is theirs: the set is ready, and the replay made with it is today's.
    release(root / simulation.SETS_RELATIVE / 'in-use', source={'finder': CUTTER})
    answer = json_of(site.request('GET', '/api/simulation/drives')[2])
    assert (answer['sets']['in-use']['ready'], answer['sets']['in-use']['networks']) == (True, NETWORKS)
    assert answer['drives'][0]['sets']['in-use']['current'] is True
    # Another cutter chosen on Models: the export is of the old one until Compute makes it again.
    in_use(root, cutter='cutter:s2/signnet')
    answer = json_of(site.request('GET', '/api/simulation/drives')[2])
    assert answer['sets']['in-use']['ready'] is False
    assert answer['sets']['in-use']['reason'] == 'The versions in use changed: Compute exports them again first.'
    assert answer['drives'][0]['sets']['in-use']['current'] is False


def test_a_drive_is_read_with_the_set_asked_on_the_clock_of_its_logs(site, root):  # noqa: F811
    status, _, data = site.request('GET', f'/api/simulation/drives/{ROUTE}?set=in-use')
    assert status == 200, data
    answer = json_of(data)
    assert set(answer) == {'ok', 'route', 'sets', 'summary', 'view', 'differences'}
    assert (answer['route'], list(answer['sets']), answer['differences']) == (ROUTE, ['car', 'in-use'], None)
    assert answer['summary']['sets']['in-use']['frames'] == 5 and answer['summary']['certified'] is None
    view = answer['view']
    assert (view['route'], view['set'], view['label'], view['frame'], view['frames']) == (
        ROUTE, 'in-use', 'In use', [1344, 760], 5)
    # Seconds from the start of the logs, as the timeline's series: the speed to hold, the sign in view, its reading.
    assert view['series']['vis_limit'] == [[10.0, 50.0], [10.6, 80.0]]
    assert view['series']['vis_raw_seen'] == [[10.0, False], [10.4, True], [10.8, False]]
    assert view['series']['vis_raw_value'] == [[10.0, 0.0], [10.4, 80.0], [10.8, 0.0]]
    assert view['events'] == [[10.6, 'sign', 80.0]]
    assert view['signs'] == [[10.4, [[600.0, 300.0, 640.0, 340.0, 'maximum_speed:80', 0.97, 'own_road']], 0, 208]]
    # The set when not said is the one in use; a set never replayed on the drive is no error.
    assert json_of(site.request('GET', f'/api/simulation/drives/{ROUTE}')[2])['view']['set'] == 'in-use'
    for missing in ('car', 'certified'):
        status, _, data = site.request('GET', f'/api/simulation/drives/{ROUTE}?set={missing}')
        assert status == 200 and json_of(data)['view'] is None and json_of(data)['differences'] is None
    status, _, data = site.request('GET', f'/api/simulation/drives/{ROUTE}?set=other')
    assert (status, json_of(data)['error']) == (400, 'bad_set')
    # A drive whose logs are not read has no clock to draw on.
    known(root, OTHER_ROUTE)
    simulation.write_result(root, replay(route=OTHER_ROUTE))
    status, _, data = site.request('GET', f'/api/simulation/drives/{OTHER_ROUTE}?set=in-use')
    assert (status, json_of(data)['error']) == (409, 'no_logs')
    assert 'Read the logs' in json_of(data)['message']
    for path in (f'/api/simulation/drives/{ROUTE}x', '/api/simulation/sets', f'/api/simulation/drives/{ROUTE}/view'):
        assert site.request('GET', path)[0] == 404
    assert site.request('GET', f'/api/simulation/drives/{ROUTE}/certify')[0] == 405


def test_a_drive_is_certified_and_the_next_replays_are_compared_with_it(site, root):  # noqa: F811
    path = f'/api/simulation/drives/{ROUTE}/certify'
    for body in ({}, {'certified': 'yes', 'set': 'in-use'}, {'certified': True}, {'certified': True, 'set': 'certified'},
                 {'certified': True, 'set': ['in-use']}, [True], {'certified': True, 'set': 'in-use', 'made_at': 5}):
        status, _, data = site.request('POST', path, body=body)
        assert (status, json_of(data)['error']) == (400, 'bad_request'), body
    # Nothing to certify with a set never replayed on the drive.
    status, _, data = site.request('POST', path, body={'certified': True, 'set': 'car'})
    assert (status, json_of(data)['error']) == (409, 'no_simulation')
    assert simulation.certified(root) == {}
    # The page certifies the replay it shows: one replayed since is refused.
    made_at = simulation.read(root, ROUTE, 'in-use')['made_at']
    status, _, data = site.request('POST', path, body={'certified': True, 'set': 'in-use', 'made_at': made_at + 'x'})
    assert (status, json_of(data)['error']) == (409, 'changed') and simulation.certified(root) == {}
    status, _, data = site.request('POST', path, body={'certified': True, 'set': 'in-use', 'made_at': made_at})
    assert status == 200, data
    answer = json_of(data)
    assert set(answer) == {'ok', 'summary'} and answer['summary']['route'] == ROUTE
    entry = answer['summary']['certified']
    assert (entry['who'], entry['set'], entry['frames'], entry['name']) == ('local', 'in-use', 5, 'networks of the test')
    assert answer['summary']['regression']['in-use']['identical'] is True
    assert simulation.certified(root)[ROUTE]['at'] == entry['at']
    # The list and the drive show it; the certified replay is read as a set of its own.
    row = json_of(site.request('GET', '/api/simulation/drives')[2])['drives'][0]
    assert row['certified']['who'] == 'local' and row['regression'] == {'in-use': answer['summary']['regression']['in-use']}
    answer = json_of(site.request('GET', f'/api/simulation/drives/{ROUTE}?set=certified')[2])
    assert answer['view']['frames'] == 5 and answer['differences'] is None
    same = json_of(site.request('GET', f'/api/simulation/drives/{ROUTE}?set=in-use')[2])['differences']
    assert (same['frames'], same['limit'], same['reading'], same['limit_seconds']) == (5, [], [], 0.0)
    # Other networks hold 50 where the certified replay held 80, on the same inputs of the brain: the brain changed.
    simulation.write_result(root, replay('car', kph=(50, 50, 50, 50, 80)))
    answer = json_of(site.request('GET', f'/api/simulation/drives/{ROUTE}?set=car')[2])
    assert answer['view']['series']['vis_limit'] == [[10.0, 50.0], [10.8, 80.0]]
    found = answer['differences']
    assert found['limit'] == [[10.6, 10.6, 80, 50]] and (found['limit_frames'], found['limit_seconds']) == (1, 0.2)
    assert (found['reading'], found['reading_frames'], found['same_input'], found['other_input']) == ([], 0, 1, 0)
    assert answer['summary']['regression']['car']['identical'] is False
    row = json_of(site.request('GET', '/api/simulation/drives')[2])['drives'][0]
    assert {key: value['identical'] for key, value in row['regression'].items()} == {'car': False, 'in-use': True}
    # Removed: the drive is one of the others again, its replays stay.
    status, _, data = site.request('POST', path, body={'certified': False})
    assert status == 200 and json_of(data)['summary']['certified'] is None and json_of(data)['summary']['regression'] == {}
    assert simulation.certified(root) == {} and simulation.reference(root, ROUTE) is None
    assert set(json_of(data)['summary']['sets']) == {'car', 'in-use'}


def test_certifying_has_the_protections_of_every_change_of_the_panel(site, root):  # noqa: F811
    path = f'/api/simulation/drives/{ROUTE}/certify'
    body = {'certified': True, 'set': 'in-use'}
    status, _, data = site.request('POST', path, body=body, headers={'Origin': 'http://evil.example'})
    assert (status, json_of(data)['error']) == (403, 'cross_origin')
    status, _, data = site.request('POST', path, body=b'certified=1',
                                   headers={'Content-Type': 'application/x-www-form-urlencoded'})
    assert status == 415
    site.app.panel.actions_enabled = False
    status, _, data = site.request('POST', path, body=body)
    assert (status, json_of(data)['error']) == (403, 'actions_disabled')
    site.app.panel.actions_enabled = True
    assert site.request('POST', f'/api/simulation/drives/{ROUTE}', body=body)[0] == 404
    assert simulation.certified(root) == {}


def test_only_a_person_signed_in_certifies_an_agent_with_the_token_reads(root, checkout):  # noqa: F811
    scanned(root)
    simulation.write_result(root, replay())
    server = RunningServer(root, token=TOKEN, require_token=True)
    server.app.panel = Panel(root, repo=checkout)
    try:
        path = f'/api/simulation/drives/{ROUTE}/certify'
        assert server.request('GET', '/api/simulation/drives')[0] == 401
        assert server.request('POST', path, body={'certified': True, 'set': 'in-use'})[0] == 401
        token = {'X-Review-Token': TOKEN}
        assert server.request('GET', '/api/simulation/drives', headers=token)[0] == 200
        assert server.request('GET', f'/api/simulation/drives/{ROUTE}?set=in-use', headers=token)[0] == 200
        status, _, data = server.request('POST', path, body={'certified': True, 'set': 'in-use'}, headers=token)
        assert (status, json_of(data)['error']) == (403, 'session_required')
        assert simulation.certified(root) == {}
        status, _, data = server.request('POST', path, body={'certified': True, 'set': 'in-use'}, headers=sign_in(server))
        assert status == 200 and json_of(data)['summary']['certified']['who'] == ADMIN
    finally:
        server.close()


def test_the_replay_starts_through_the_site_and_no_choice_is_dropped(site, root):  # noqa: F811
    run = '/api/panel/run/simulate'
    # A set left out or sent as something else is refused, never dropped into a replay with other networks; a
    # "certified" sent as true is refused, never dropped into the replay of the drive beside it.
    for body in ({'route': ROUTE}, {'route': ROUTE, 'set': 7}, {'route': ROUTE, 'set': 'in-use', 'certified': True},
                 {'set': 'in-use'}, {'route': 7, 'set': 'in-use'}):
        status, _, data = site.request('POST', run, body=body)
        assert (status, json_of(data)['error']) == (400, 'bad_option'), body
    status, _, data = site.request('POST', run, body={'certified': '1', 'set': 'in-use'})
    assert (status, json_of(data)['message']) == (409, 'Certify a drive first.')
    assert site.app.panel.jobs() == []
    status, _, data = site.request('POST', run, body={'route': ROUTE, 'set': 'in-use'})
    assert status == 200, data
    job = json_of(data)
    assert (job['action'], job['label']) == ('simulate', 'Replay a drive')
    assert finished(site.app.panel, job['id'])['state'] == 'ok'
    assert full_log(site.app.panel, job['id']).splitlines()[0] == f'== helpers/simulate_drive.sh --set=in-use --route={ROUTE}'


def test_a_reviewer_from_outside_gets_nothing_of_it(outside):  # noqa: F811
    server, _ = outside
    give_all(server)
    simulation.write_result(server.app.root, replay())
    for method, path, body in (('GET', '/drive', None), ('GET', '/api/simulation/drives', None),
                               ('GET', f'/api/simulation/drives/{ROUTE}?set=in-use', None),
                               ('POST', f'/api/simulation/drives/{ROUTE}/certify', {'certified': True, 'set': 'in-use'}),
                               ('POST', '/api/panel/run/simulate', {'route': ROUTE, 'set': 'in-use'})):
        status, _, data = server.request(method, path, body=body, headers=proof())
        assert status == 403, (method, path, status)
        assert ROUTE.encode() not in data
    assert simulation.certified(server.app.root) == {}


# ---------- the tab ----------

def test_the_tab_is_drive_between_versions_and_results_and_regression_leads_there(site):
    models = next(pages for key, _, _, pages in SECTIONS if key == 'models')
    assert models == (('models', '/models', 'Versions'), ('drive', '/drive', 'Drive'), ('evaluate', '/evaluate', 'Results'),
                      ('publish', '/publish', 'Publish'))
    assert CONTROL_TABS['/drive'] == 'drive' and '/regression' not in CONTROL_TABS
    assert OLD_ADDRESSES['/regression'] == '/drive'
    assert 'href="/drive" aria-current="page">Drive</a>' in sub_nav('drive')
    status, _, data = site.request('GET', '/drive')
    page = data.decode()
    assert status == 200 and '<title>Speed limits · Drive</title>' in page
    # The Models section is marked, and its busy dot counts the tab.
    assert 'data-section="models" data-paths="/models /drive /evaluate /publish" aria-current="page">Models</a>' in page
    # An old link to a drive of Regression opens that drive on the new page.
    for old, new in (('/regression', '/drive'), (f'/regression?drive={ROUTE}', f'/drive?drive={ROUTE}')):
        status, response, _ = site.request('GET', old)
        assert status == 303 and response.getheader('Location') == new, old


def test_signed_out_the_tab_and_its_old_address_show_the_sign_in_page(root, checkout):  # noqa: F811
    from nnslr_tools.pages import asset
    server = RunningServer(root, token=TOKEN, require_token=True)
    server.app.panel = Panel(root, repo=checkout)
    try:
        for path in ('/drive', '/regression'):
            status, _, data = server.request('GET', path)
            assert status == 401 and asset('js/login.js').encode() in data, path
    finally:
        server.close()
