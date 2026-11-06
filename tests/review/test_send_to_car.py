"""Publish to the car (the Publish tab; send_to_car): nnslr_tools.car_release (export, publish in the models clone, on-car.json),
scripts/send_to_car.sh (the sunnypilot submodule and the build) and the panel action send_to_car.

Every repository is a temporary git clone with a bare remote; the export and the build are fakes: nothing is exported
with torch, nothing leaves this machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import date
from pathlib import Path

import pytest
import test_panel
from test_panel import finished, full_log
from test_panel_models import make_signnet, pipeline
from test_panel_runs import write

from nnslr_tools import car_release, car_settings
from nnslr_tools.job_progress import progress
from nnslr_tools.model_roles import NOT_SELECTABLE, ModelChoiceError, ModelIndex, role_of
from nnslr_tools.panel import Panel, PanelError
from nnslr_tools.runs import RunsIndex

# The panel tests' fixtures (pytest finds them by name in this module; only_this_checkout is autouse).
checkout, only_this_checkout, root = test_panel.checkout, test_panel.only_this_checkout, test_panel.root
REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / car_release.CAR_SCRIPT
IDENTITY = {'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@example.com', 'GIT_COMMITTER_NAME': 'test',
            'GIT_COMMITTER_EMAIL': 'test@example.com'}
TODAY = date(2026, 10, 28)


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    for key, value in IDENTITY.items():
        monkeypatch.setenv(key, value)


def git(folder: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(folder), '-c', 'protocol.file.allow=always', *args], check=True,
                          capture_output=True, text=True, env={**os.environ, **IDENTITY}).stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def servers_runs(data: Path) -> None:
    """A finished SignNet s1, an unfinished s3, and the reader of r1 as the classifier in use."""
    pipeline(data, 'r1')
    make_signnet(data, 's1')
    make_signnet(data, 's3', finished=False)
    runs = RunsIndex(data)
    runs.running_runs = dict
    ModelIndex(runs).set_active('classifier', 'classifier:r1/reader', who=None, via='local')


def fake_export(content: bytes):
    """An exporter that writes what the real one writes (both ONNX and their contracts), with these bytes."""
    def exporter(chosen: dict, folder: Path) -> dict:
        for network, name in car_release.FILES.items():
            onnx = content + b' ' + name.encode()
            (folder / name).write_bytes(onnx)
            write(folder / (name + '.contract.json'), {
                'task': network, 'checkpoint': str(chosen[network]['weights']), 'onnx_sha256': sha256(onnx),
                'checkpoint_sha256': sha256(Path(chosen[network]['weights']).read_bytes())})
        return car_release.finish_contracts(chosen, folder)
    return exporter


def models_clone(tmp_path: Path, version: str = '2026-10-27') -> tuple[Path, Path]:
    """The models repository the comma downloads: a bare origin and a clone of it, with older networks."""
    origin = tmp_path / 'models.git'
    git(tmp_path, 'init', '--bare', '-q', '-b', 'main', str(origin))
    clone = tmp_path / 'models'
    git(tmp_path, 'clone', '-q', str(origin), str(clone))
    (clone / 'README.md').write_text('models\n')
    (clone / 'signnet.onnx').write_bytes(b'old')
    write(clone / 'speed_limits.json', car_release.manifest_of(version))
    git(clone, 'add', '-A')
    git(clone, 'commit', '-q', '-m', 'old networks')
    git(clone, 'push', '-q', 'origin', 'HEAD:main')
    return origin, clone


# -- the module ---------------------------------------------------------------------------------------------
def test_the_version_is_the_day_then_numbered_the_same_day():
    assert car_release.next_version('2026-10-27', TODAY) == '2026-10-28'
    assert car_release.next_version(None, TODAY) == '2026-10-28'
    assert car_release.next_version('2026-10-28', TODAY) == '2026-10-28.2'
    assert car_release.next_version('2026-10-28.2', TODAY) == '2026-10-28.3'
    assert car_release.next_version('2026-10-28.x', TODAY) == '2026-10-28.2'


@pytest.mark.parametrize('signnet, classifier, message', [
    ('../s1', None, 'is not a run name'),
    ('nothing', None, 'has no SignNet on the Models tab'),
    ('r1', None, 'has no SignNet on the Models tab'),
    ('s3', None, 'has not finished its training'),
    ('s1', 'cutter:r1/detector', 'is not the id of a classifier'),
    ('s1', 'classifier:r1/signnet', 'is not the id of a classifier'),
    ('s1', 'classifier:r9/reader', 'is not on this server'),
])
def test_only_a_finished_signnet_of_the_models_tab_and_a_classifier_of_it(tmp_path, signnet, classifier, message):
    data = tmp_path / 'data'
    servers_runs(data)
    with pytest.raises(ValueError, match=message):
        car_release.choose(data, signnet, classifier)


def test_the_classifier_is_the_one_in_use_unless_one_is_given(tmp_path):
    data = tmp_path / 'data'
    pipeline(data, 'r1')
    make_signnet(data, 's1')
    with pytest.raises(ValueError, match='no classifier in use'):
        car_release.choose(data, 's1')
    chosen = car_release.choose(data, 's1', 'classifier:r1/reader')
    assert chosen['finder']['id'] == 'cutter:s1/signnet' and chosen['finder']['weights'].name == 'signnet-best.pt'
    assert chosen['classifier']['weights'] == (data / 'runs/r1/reader/reader-best.pt').resolve()


@pytest.mark.parametrize('band, stride, size', [([1344, 416], 8, '1344 x 416 at stride 8'),
                                                ([896, 288], 4, '896 x 288 at stride 4')])
def test_only_a_signnet_of_the_cars_band_and_grid_goes(tmp_path, band, stride, size):
    data = tmp_path / 'data'
    servers_runs(data)
    make_signnet(data, 'wide', band=band, stride=stride)
    with pytest.raises(ValueError, match=f'the SignNet of wide is {size}: the comma runs 896 x 288 at stride 8'):
        car_release.choose(data, 'wide')


def test_without_a_run_the_cutter_in_use_goes(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    with pytest.raises(ValueError, match='no cutter in use: choose one on the Models tab'):
        car_release.choose(data)
    # An old Detector still in use (written before SignNet was the cutter): it never goes on the car.
    active = data / 'models' / 'active.json'
    value = json.loads(active.read_text())
    active.write_text(json.dumps({**value, 'cutter': {'id': 'cutter:r1/detector', 'via': 'local'}}))
    with pytest.raises(ValueError, match='the cutter in use, cutter:r1/detector, is an old Detector: choose a SignNet'):
        car_release.choose(data, None, 'classifier:r1/reader')
    runs = RunsIndex(data)
    runs.running_runs = dict
    ModelIndex(runs).set_active('cutter', 'cutter:s1/signnet', who=None, via='local')
    chosen = car_release.choose(data)
    assert (chosen['finder']['id'], chosen['finder']['run']) == ('cutter:s1/signnet', 's1')
    assert chosen['finder']['weights'] == (data / 'runs/s1/signnet-best.pt').resolve()
    assert chosen['classifier']['id'] == 'classifier:r1/reader'
    # A run given still wins over the one in use.
    make_signnet(data, 's2')
    assert car_release.choose(data, 's2')['finder']['id'] == 'cutter:s2/signnet'


def test_published_once_then_nothing_then_a_new_version_the_same_day(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    origin, clone = models_clone(tmp_path)
    lines = []
    result = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v1'), say=lines.append)
    head = git(clone, 'rev-parse', 'HEAD')
    assert result['changed'] and (result['version'], result['commit']) == ('2026-10-28', head)
    assert git(origin, 'rev-parse', 'main') == head
    assert json.loads((clone / 'speed_limits.json').read_text()) == car_release.manifest_of(
        '2026-10-28', settings=car_settings.DEFAULTS)
    assert (clone / 'speed_limits.json').read_text().startswith('{\n  "format": 2,\n  "version": "2026-10-28",')
    contract = json.loads((clone / 'signnet.onnx.contract.json').read_text())
    assert contract['checkpoint'] == 'cutter:s1/signnet' and contract['onnx_sha256'] == sha256(b'v1 signnet.onnx')
    assert json.loads((clone / 'classifier.onnx.contract.json').read_text())['checkpoint'] == 'classifier:r1/reader'
    # The commit holds the networks and the manifest, nothing else.
    assert sorted(git(clone, 'show', '--name-only', '--format=', 'HEAD').split()) == [
        'classifier.onnx', 'classifier.onnx.contract.json', 'signnet.onnx', 'signnet.onnx.contract.json', 'speed_limits.json']
    record = json.loads((data / 'models/on-car.json').read_text())
    assert (record['cutter']['id'], record['classifier']['id'], record['commit']) == (
        'cutter:s1/signnet', 'classifier:r1/reader', head)
    assert [line for line in lines if line.startswith('== ')] == [
        '== 1/3 Export the SignNet of s1 and the classifier classifier:r1/reader', f'== 2/3 Publish them in {clone}',
        '== 3/3 Record the networks on the car']

    # The same networks again: no commit, no push, on-car.json as it was.
    recorded = (data / 'models/on-car.json').read_text()
    lines.clear()
    again = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v1'), say=lines.append)
    assert (again['changed'], again['commit'], again['version']) == (False, head, '2026-10-28')
    assert git(clone, 'rev-parse', 'HEAD') == head == git(origin, 'rev-parse', 'main')
    assert (data / 'models/on-car.json').read_text() == recorded
    assert any(line.startswith('unchanged: ') for line in lines) and 'models/on-car.json already records them' in lines

    # Other networks the same day: version .2, then .3.
    second = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v2'), say=lines.append)
    assert second['changed'] and second['version'] == '2026-10-28.2' and git(origin, 'rev-parse', 'main') == second['commit']
    third = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v3'), say=lines.append)
    assert third['version'] == '2026-10-28.3'
    assert json.loads((data / 'models/on-car.json').read_text())['version'] == '2026-10-28.3'


def road_type_run(data: Path, name: str, onnx: bytes, *, export: bool = True) -> str:
    """A finished road type with memory (road-type/gru) and, unless not, the ONNX step its training exported."""
    run = data / 'runs' / name / 'road-type'
    weights = b'road type weights ' + name.encode()
    (run / 'gru').mkdir(parents=True)
    (run / 'gru' / 'road-type-best.pt').write_bytes(weights)
    write(run / 'gru' / 'result.json', {'task': 'road_type', 'memory': 'gru', 'motion': True})
    if export:
        (run / 'export').mkdir()
        (run / 'export' / 'road-type.onnx').write_bytes(onnx)
        write(run / 'export' / 'road-type.onnx.contract.json', {
            'task': 'road_type', 'checkpoint': str(run / 'gru' / 'road-type-best.pt'), 'onnx_sha256': sha256(onnx),
            'checkpoint_sha256': sha256(weights)})
    return f'road_type:{name}/road-type/gru'


def use_road_type(data: Path, model_id: str) -> None:
    runs = RunsIndex(data)
    runs.running_runs = dict
    ModelIndex(runs).set_active('road_type', model_id, who=None, via='local')


def fake_export_with_road_type(content: bytes):
    """fake_export, and the road type's step copied as the real export does."""
    signs = fake_export(content)

    def exporter(chosen: dict, folder: Path) -> dict:
        car_release.copy_road_type(chosen, folder)
        return signs(chosen, folder)
    return exporter


def test_the_road_type_in_use_goes_on_the_car_and_goes_away_with_it(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    origin, clone = models_clone(tmp_path)
    use_road_type(data, road_type_run(data, 't1', b'road v1'))
    lines = []
    result = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_road_type(b'v1'),
                                     say=lines.append)
    assert result['changed'] and git(origin, 'rev-parse', 'main') == result['commit']
    manifest = json.loads((clone / 'speed_limits.json').read_text())
    assert manifest == car_release.manifest_of('2026-10-28', road_type=True, settings=car_settings.DEFAULTS)
    assert manifest['road_type'] == {'file': 'road-type.onnx', 'contract': 'road-type.onnx.contract.json'}
    assert (clone / 'road-type.onnx').read_bytes() == b'road v1'
    contract = json.loads((clone / 'road-type.onnx.contract.json').read_text())
    assert contract['checkpoint'] == 'road_type:t1/road-type/gru' and contract['onnx_sha256'] == sha256(b'road v1')
    assert 'road-type.onnx' in git(clone, 'show', '--name-only', '--format=', 'HEAD').split()
    assert 'road type t1' in git(clone, 'log', '-1', '--format=%s')
    record = json.loads((data / 'models/on-car.json').read_text())
    assert record['road_type']['id'] == 'road_type:t1/road-type/gru'
    assert lines[0] == ('== 1/3 Export the SignNet of s1 and the classifier classifier:r1/reader, the road type '
                        'road_type:t1/road-type/gru')
    roles = {role['role']: role for role in ModelIndex(RunsIndex(data)).index()['roles']}
    assert [entry['id'] for entry in roles['road_type']['versions'] if entry['on_car']] == ['road_type:t1/road-type/gru']

    # The same again: nothing; another road type: a new version.
    again = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_road_type(b'v1'),
                                    say=lambda line: None)
    assert not again['changed']
    use_road_type(data, road_type_run(data, 't2', b'road v2'))
    second = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_road_type(b'v1'),
                                     say=lambda line: None)
    assert second['changed'] and (clone / 'road-type.onnx').read_bytes() == b'road v2'

    # No road type in use: the published one is removed, the signs alone.
    active = data / 'models' / 'active.json'
    value = json.loads(active.read_text())
    value.pop('road_type')
    active.write_text(json.dumps(value))
    third = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_road_type(b'v1'),
                                    say=lambda line: None)
    assert third['changed'] and not (clone / 'road-type.onnx').exists()
    assert 'road_type' not in json.loads((clone / 'speed_limits.json').read_text())
    assert git(clone, 'ls-files', 'road-type.onnx', 'road-type.onnx.contract.json') == ''
    assert 'road_type' not in json.loads((data / 'models/on-car.json').read_text())


def brain_run(data: Path, name: str, *, result: dict | None | bool = None) -> str:
    """A brain at the root of its run, finished unless result is False."""
    run = data / 'runs' / name
    run.mkdir(parents=True)
    (run / 'brain-last.pt').write_bytes(b'brain ' + name.encode())
    if result is not False:
        write(run / 'result.json', result or {
            'groups': ['signs', 'lead'], 'seed': 1, 'works': True, 'history': [{}],
            'test': {'speed': {'frames': 10, 'right': 0.9, 'mean_error_kph': 1.0}},
            'validation': {'speed': {'frames': 9, 'right': 0.8, 'mean_error_kph': 2.0}}})
    return f'brain:{name}/brain'


def use_brain(data: Path, model_id: str) -> None:
    runs = RunsIndex(data)
    runs.running_runs = dict
    ModelIndex(runs).set_active('brain', model_id, who=None, via='local')


def fake_export_with_brain(content: bytes, onnx: bytes):
    """fake_export, and a brain's step as the real export writes it."""
    signs = fake_export(content)

    def exporter(chosen: dict, folder: Path) -> dict:
        (folder / car_release.BRAIN_FILE).write_bytes(onnx)
        write(folder / (car_release.BRAIN_FILE + '.contract.json'), {
            'task': 'brain', 'checkpoint': str(chosen['brain']['weights']), 'onnx_sha256': sha256(onnx),
            'checkpoint_sha256': sha256(Path(chosen['brain']['weights']).read_bytes())})
        return signs(chosen, folder)
    return exporter


def test_the_brain_in_use_goes_on_the_car_instead_of_the_road_type_and_goes_away_with_it(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    _, clone = models_clone(tmp_path)
    use_road_type(data, road_type_run(data, 't1', b'road'))
    first = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_road_type(b'v1'))
    assert first['changed'] and (clone / 'road-type.onnx').is_file()
    use_brain(data, brain_run(data, 'b1'))

    second = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_brain(b'v1', b'brain v1'))

    manifest = json.loads((clone / 'speed_limits.json').read_text())
    assert second['changed'] and manifest['brain'] == {'file': 'brain.onnx', 'contract': 'brain.onnx.contract.json'}
    assert 'road_type' not in manifest and not (clone / 'road-type.onnx').exists()  # nothing on the car would read it
    assert manifest['settings']['country_policy'] == car_release.COUNTRY_POLICY
    assert (clone / 'brain.onnx').read_bytes() == b'brain v1'
    assert json.loads((clone / 'brain.onnx.contract.json').read_text())['checkpoint'] == 'brain:b1/brain'
    record = json.loads((data / 'models/on-car.json').read_text())
    assert record['brain']['id'] == 'brain:b1/brain' and 'road_type' not in record
    again = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_brain(b'v1', b'brain v1'))
    assert not again['changed']

    path = data / 'models/active.json'
    value = json.loads(path.read_text())
    value.pop('brain')
    path.write_text(json.dumps(value))
    third = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export_with_road_type(b'v1'))
    manifest = json.loads((clone / 'speed_limits.json').read_text())
    assert third['changed'] and 'brain' not in manifest and not (clone / 'brain.onnx').exists()
    assert 'country_policy' not in manifest['settings'] and manifest['road_type']


def test_a_brain_goes_only_finished_and_its_dry_run_is_not_a_version(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    brain_run(data, 'b1', result={'groups': ['signs'], 'dry_run': True, 'test': {}})
    assert ModelIndex(RunsIndex(data)).artifacts(('brain',))['brain'] == []
    model_id = brain_run(data, 'b2', result=False)
    with pytest.raises(ModelChoiceError):
        use_brain(data, model_id)
    assert car_release.choose(data, 's1')['brain'] is None


def test_the_brain_part_is_the_brains_alone():
    assert role_of('brain:b1/brain') == 'brain'
    assert role_of('cutter:b1/brain') is None
    assert role_of('brain:b1/signnet') is None
    assert role_of('brain:b1/detector') is None


def test_only_a_road_type_with_memory_and_its_exported_step_goes_on_the_car(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    use_road_type(data, road_type_run(data, 't1', b'road', export=False))
    with pytest.raises(ValueError, match='the scene network .* has no exported ONNX step'):
        car_release.choose(data, 's1')
    frame = data / 'runs' / 't3' / 'road-type' / 'frame'
    frame.mkdir(parents=True)
    (frame / 'road-type-best.pt').write_bytes(b'w')
    write(frame / 'result.json', {'task': 'road_type', 'memory': 'none'})
    use_road_type(data, 'road_type:t3/road-type/frame')
    with pytest.raises(ValueError, match='the scene network in use, .*, has no memory'):
        car_release.choose(data, 's1')


def test_a_road_type_with_the_board_input_goes_only_with_a_classifier_that_has_the_boards(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    model_id = road_type_run(data, 't1', b'road')
    contract = data / 'runs/t1/road-type/export/road-type.onnx.contract.json'
    value = json.loads(contract.read_text())
    write(contract, {**value, 'step': {'boards': True, 'board_signs': ['other_sign:motorway_direction']}})
    use_road_type(data, model_id)
    # The classifier in use (classifier:r1/reader) has no such class: refused.
    with pytest.raises(ValueError, match='reads the green boards towards a motorway .* has no such class'):
        car_release.choose(data, 's1')
    classes = data / 'runs/r1/reader/result.json'
    result = json.loads(classes.read_text()) if classes.is_file() else {}
    write(classes, {**result, 'classes': ['maximum_speed:50', 'other_sign:motorway_direction']})
    assert car_release.choose(data, 's1')['road_type']['id'] == model_id
    # A scene network without the board input goes with any classifier.
    write(contract, {**value, 'step': {'boards': False, 'board_signs': []}})
    write(classes, {**result, 'classes': ['maximum_speed:50']})
    assert car_release.choose(data, 's1')['road_type']['id'] == model_id


def test_a_brain_that_reads_the_plates_goes_only_with_a_classifier_that_has_a_class_with_hours(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    use_brain(data, brain_run(data, 'b1', result={'groups': ['signs', 'plate'], 'seed': 1, 'plates': True, 'history': [{}],
                                                  'test': {}, 'validation': {}}))
    classes = data / 'runs/r1/reader/result.json'
    result = json.loads(classes.read_text()) if classes.is_file() else {}
    write(classes, {**result, 'classes': ['maximum_speed:30', 'maximum_speed:50']})
    with pytest.raises(ValueError, match='reads the plates of hours, and the classifier .* has no class with hours'):
        car_release.choose(data, 's1')
    write(classes, {**result, 'classes': ['maximum_speed:30', 'maximum_speed_hours:30']})
    assert car_release.choose(data, 's1')['brain']['id'] == 'brain:b1/brain'


def test_a_class_for_wet_roads_is_no_class_with_hours(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    use_brain(data, brain_run(data, 'b1', result={'groups': ['signs', 'plate'], 'seed': 1, 'plates': True, 'history': [{}],
                                                  'test': {}, 'validation': {}}))
    classes = data / 'runs/r1/reader/result.json'
    result = json.loads(classes.read_text()) if classes.is_file() else {}
    write(classes, {**result, 'classes': ['maximum_speed:30', 'maximum_speed_wet:80']})
    with pytest.raises(ValueError, match='reads the plates of hours, and the classifier .* has no class with hours'):
        car_release.choose(data, 's1')


def test_a_brain_that_reads_the_hours_goes_only_with_a_classifier_that_says_them(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    use_brain(data, brain_run(data, 'b1', result={'groups': ['signs', 'plate'], 'seed': 1, 'plates': True, 'hours': True,
                                                  'history': [{}], 'test': {}, 'validation': {}}))
    classes = data / 'runs/r1/reader/result.json'
    result = json.loads(classes.read_text()) if classes.is_file() else {}
    write(classes, {**result, 'classes': ['maximum_speed:30', 'maximum_speed_hours:30']})
    with pytest.raises(ValueError, match='reads which hours a plate says, and the classifier .* has no class that says them'):
        car_release.choose(data, 's1')
    write(classes, {**result, 'classes': ['maximum_speed_hours:30', 'maximum_speed_hours:30:22-6']})
    assert car_release.choose(data, 's1')['brain']['id'] == 'brain:b1/brain'


def test_a_brain_without_plates_goes_with_any_classifier(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    use_brain(data, brain_run(data, 'b1'))
    write(data / 'runs/r1/reader/result.json', {'classes': ['maximum_speed:50']})
    assert car_release.choose(data, 's1')['brain']['id'] == 'brain:b1/brain'


def test_a_dry_run_exports_and_compares_only(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    origin, clone = models_clone(tmp_path)
    head = git(clone, 'rev-parse', 'HEAD')
    lines = []
    result = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v1'), dry_run=True,
                                     say=lines.append)
    assert (result['changed'], result['commit'], result['version']) == (True, None, '2026-10-28')
    assert git(clone, 'rev-parse', 'HEAD') == head == git(origin, 'rev-parse', 'main')
    assert git(clone, 'status', '--porcelain') == '' and not (data / 'models/on-car.json').exists()
    assert 'dry run: would publish version 2026-10-28 (nothing copied, committed or pushed)' in lines
    assert not list((data / 'tmp').iterdir())


def test_a_failed_push_leaves_the_clone_as_it_was(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    origin, clone = models_clone(tmp_path)
    head = git(clone, 'rev-parse', 'HEAD')
    # A hook of the remote refuses the push after the fetch went well.
    hook = origin / 'hooks' / 'pre-receive'
    hook.write_text('#!/bin/sh\necho refused >&2\nexit 1\n')
    hook.chmod(0o755)
    with pytest.raises(ValueError, match='git push'):
        car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v1'), say=lambda line: None)
    assert git(clone, 'rev-parse', 'HEAD') == head and git(clone, 'status', '--porcelain') == ''
    assert not (data / 'models/on-car.json').exists()


def test_the_clone_is_fast_forwarded_and_refused_with_changes_or_commits_of_its_own(tmp_path):
    data = tmp_path / 'data'
    servers_runs(data)
    origin, clone = models_clone(tmp_path)
    # Somebody else pushed: the clone catches up first.
    other = tmp_path / 'other'
    git(tmp_path, 'clone', '-q', str(origin), str(other))
    (other / 'README.md').write_text('newer\n')
    git(other, 'commit', '-q', '-am', 'readme')
    git(other, 'push', '-q', 'origin', 'HEAD:main')
    result = car_release.send_to_car(data, 's1', models=clone, today=TODAY, exporter=fake_export(b'v1'), say=lambda line: None)
    assert (clone / 'README.md').read_text() == 'newer\n' and git(origin, 'rev-parse', 'main') == result['commit']
    (clone / 'README.md').write_text('changed here\n')
    with pytest.raises(ValueError, match='changes not committed'):
        car_release.send_to_car(data, 's1', models=clone, exporter=fake_export(b'v2'), say=lambda line: None)
    git(clone, 'commit', '-q', '-am', 'local')
    with pytest.raises(ValueError, match=r'1 commit\(s\) not on origin/main'):
        car_release.send_to_car(data, 's1', models=clone, exporter=fake_export(b'v2'), say=lambda line: None)


def test_nnslr_send_to_car_numbers_its_steps_for_the_script(tmp_path, monkeypatch, capsys):
    from nnslr_tools.cli import main
    data = tmp_path / 'data'
    servers_runs(data)
    _, clone = models_clone(tmp_path)
    monkeypatch.setattr(car_release, 'export', fake_export(b'v1'))
    assert main(['send-to-car', '--signnet', 's1', '--models', str(clone), '--data-root', str(data), '--steps-of', '5',
                 '--dry-run']) == 0
    out = capsys.readouterr().out
    assert '== 1/5 Export the SignNet of s1' in out and '== 3/5 Record the networks on the car' in out
    assert main(['send-to-car', '--signnet', '../x', '--models', str(clone), '--data-root', str(data)]) == 2


# -- scripts/send_to_car.sh -----------------------------------------------------------------------------------
FAKE_NNSLR = r'''#!/usr/bin/env bash
# Steps 1-3 of nnslr send-to-car: FAKE_PUBLISH=<bytes> publishes in the models clone (--models), as the real one does.
echo "nnslr $*"
models=""; dry=0
while [ "$#" -gt 0 ]; do case "$1" in --models) models="$2"; shift ;; --dry-run) dry=1 ;; esac; shift; done
echo "== 1/5 Export"
echo "== 2/5 Publish"
if [ -n "${FAKE_PUBLISH:-}" ] && [ "$dry" -eq 1 ]; then echo "dry run: would publish version x"; fi
if [ -n "${FAKE_PUBLISH:-}" ] && [ "$dry" -eq 0 ]; then
  echo "$FAKE_PUBLISH" > "$models/signnet.onnx"
  git -C "$models" commit -q -am publish && git -C "$models" push -q origin HEAD:main
fi
echo "== 3/5 Record"
'''
FAKE_BUILD = '''#!/usr/bin/env bash
echo "build $* WORKSPACE_ROOT=$WORKSPACE_ROOT $GIT_CONFIG_KEY_0=$GIT_CONFIG_VALUE_0 NO_PUSH=${NO_PUSH:-}"
'''
SUBMODULE = 'openpilot/sunnypilot/speed_limits/models'


@pytest.fixture
def car(tmp_path):
    """The clones the script works on: the models (origin and clone), sunnypilot on speed-limits with the models as a
    submodule (origin and clone), the testing branch's repository, a fake export and a fake build."""
    models_origin, clone = models_clone(tmp_path)
    sp_origin = tmp_path / 'sunnypilot.git'
    git(tmp_path, 'init', '--bare', '-q', str(sp_origin))
    sp = tmp_path / 'sunnypilot'
    sp.mkdir()
    git(sp, 'init', '-q', '-b', 'speed-limits')
    (sp / 'README.md').write_text('sunnypilot\n')
    git(sp, 'submodule', 'add', '-q', str(models_origin), SUBMODULE)
    git(sp, 'add', '-A')
    git(sp, 'commit', '-q', '-m', 'speed limits')
    git(sp, 'remote', 'add', 'origin', str(sp_origin))
    git(sp, 'push', '-q', 'origin', 'speed-limits')
    testing = tmp_path / 'openpilot.git'
    git(tmp_path, 'init', '--bare', '-q', str(testing))
    git(sp, 'push', '-q', str(testing), 'speed-limits:refs/heads/psa-torque-sunny-testing')
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin/nnslr').write_text(FAKE_NNSLR)
    (venv / 'bin/nnslr').chmod(0o755)
    scripts = tmp_path / 'openpilot_scripts'
    build = scripts / car_release.BUILD_SCRIPT
    build.parent.mkdir(parents=True)
    build.write_text(FAKE_BUILD)
    build.chmod(0o755)
    env = {**os.environ, **IDENTITY, 'NNSLR_DATA_ROOT': str(tmp_path / 'data'), 'NNSLR_GPU_VENV': str(venv),
           'NNSLR_MODELS_CLONE': str(clone), 'NNSLR_SUNNYPILOT_CLONE': str(sp), 'NNSLR_OPENPILOT_SCRIPTS': str(scripts),
           'NNSLR_CAR_WORKSPACE': str(tmp_path / 'ws'), 'NNSLR_TESTING_REPO': str(testing),
           'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'protocol.file.allow', 'GIT_CONFIG_VALUE_0': 'always'}
    return {'env': env, 'clone': clone, 'sp': sp, 'sp_origin': sp_origin, 'data': tmp_path / 'data', 'testing': testing}


def send(car: dict, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(['bash', str(SCRIPT), *args], capture_output=True, text=True, check=False, timeout=60,
                          env={**car['env'], **env})


def test_the_script_checks_its_arguments(car):
    assert send(car, '--help').returncode == 0 and 'Usage:' in send(car, '--help').stdout
    for args, message in ((('--bogus',), 'unknown argument: --bogus'), ((), 'which SignNet?'),
                          (('--signnet', '../x'), 'invalid run name: ../x'),
                          (('--signnet=s1', '--classifier=cutter:x/y'), 'invalid classifier id'),
                          (('--signnet',), '--signnet needs a run')):
        done = send(car, *args)
        assert done.returncode == 1 and message in done.stderr, (args, done.stderr)
    done = send(car, '--signnet=s1', NNSLR_SUNNYPILOT_CLONE=str(car['sp'].parent / 'nowhere'))
    assert done.returncode == 1 and 'no sunnypilot clone' in done.stderr


def test_nothing_changed_nothing_is_committed_pushed_or_built(car):
    before = git(car['sp_origin'], 'rev-parse', 'speed-limits')
    done = send(car, '--signnet', 's1')
    assert done.returncode == 0, done.stderr
    assert 'nnslr send-to-car --signnet s1 --models' in done.stdout and '--steps-of 5' in done.stdout
    assert 'unchanged: speed-limits already has the models' in done.stdout
    assert 'unchanged: nothing to build' in done.stdout and 'build sunny' not in done.stdout
    assert done.stdout.rstrip().endswith('Nothing to update on the comma.')
    assert git(car['sp_origin'], 'rev-parse', 'speed-limits') == before


def test_a_publication_moves_the_submodule_alone_then_builds(car):
    sp = car['sp']
    # Somebody else's work in the sunnypilot clone: staged and not staged, it stays as it is.
    (sp / 'README.md').write_text('somebody else\n')
    (sp / 'other.txt').write_text('staged by somebody else\n')
    git(sp, 'add', 'other.txt')
    done = send(car, '--signnet', 's1', FAKE_PUBLISH='new networks')
    assert done.returncode == 0, done.stdout + done.stderr
    models_head = git(car['clone'], 'rev-parse', 'HEAD')
    assert git(car['sp_origin'], 'rev-parse', f'speed-limits:{SUBMODULE}') == models_head
    assert git(sp, 'show', '--name-only', '--format=', 'HEAD').split() == [SUBMODULE]
    assert (git(sp, 'diff', '--cached', '--name-only'), git(sp, 'diff', '--name-only')) == ('other.txt', 'README.md')
    assert f'moved: {SUBMODULE}' in done.stdout
    assert ('build sunny WORKSPACE_ROOT=' + str(car['env']['NNSLR_CAR_WORKSPACE'])
            + ' url.git@github.com:cristianku/.pushInsteadOf=https://github.com/cristianku/ NO_PUSH=') in done.stdout
    assert 'psa-torque-sunny-testing is now ' in done.stdout
    assert done.stdout.rstrip().endswith('On the comma: Settings, Software, Check for updates, then install.')
    assert not (car['data'] / car_release.PENDING_FILE).exists()
    # Run again: everything is in place, nothing is built.
    again = send(car, '--signnet', 's1')
    assert again.returncode == 0 and 'build sunny' not in again.stdout and 'unchanged: speed-limits already has' in again.stdout


def test_a_build_without_push_keeps_it_owed(car):
    done = send(car, '--signnet', 's1', '--rebuild', NO_PUSH='true')
    assert done.returncode == 0, done.stderr
    assert 'NO_PUSH=true' in done.stdout and 'not pushed' in done.stdout
    assert 'On the comma' not in done.stdout
    # A build owed (sunnypilot moved, the build did not finish): the next run builds without --rebuild.
    pending = car['data'] / car_release.PENDING_FILE
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text('x\n')
    owed = send(car, '--signnet', 's1')
    assert 'build sunny' in owed.stdout and not pending.exists()


def test_a_dry_run_builds_nothing_and_moves_nothing(car):
    before = git(car['sp_origin'], 'rev-parse', 'speed-limits')
    done = send(car, '--signnet', 's1', '--dry-run', FAKE_PUBLISH='new networks')
    assert done.returncode == 0, done.stderr
    assert 'dry run: would move' in done.stdout and 'dry run: would build' in done.stdout and 'build sunny' not in done.stdout
    assert git(car['sp_origin'], 'rev-parse', 'speed-limits') == before


@pytest.mark.parametrize('change, message', [
    ('branch', 'not speed-limits: switch it back first'),
    ('submodule', 'has changes of its own'),
    ('unpushed', 'commit(s) not pushed'),
])
def test_the_sunnypilot_clone_is_refused_unless_it_is_clean_on_speed_limits(car, change, message):
    sp = car['sp']
    if change == 'branch':
        git(sp, 'checkout', '-q', '-b', 'elsewhere')
    elif change == 'submodule':
        sub = sp / SUBMODULE
        (sub / 'README.md').write_text('changed in the submodule\n')
    else:
        (sp / 'README.md').write_text('local\n')
        git(sp, 'commit', '-q', '-am', 'local')
    before = git(car['sp_origin'], 'rev-parse', 'speed-limits')
    models = git(car['clone'], 'rev-parse', 'HEAD')
    done = send(car, '--signnet', 's1', FAKE_PUBLISH='new networks')
    assert done.returncode == 1 and message in done.stderr, done.stderr
    assert 'build sunny' not in done.stdout and git(car['sp_origin'], 'rev-parse', 'speed-limits') == before
    # Refused before step 1: the networks the comma downloads are not published ahead of the branch.
    assert 'nnslr send-to-car' not in done.stdout and git(car['clone'], 'rev-parse', 'HEAD') == models


def test_a_branch_behind_origin_is_fast_forwarded_first(car, tmp_path):
    other = tmp_path / 'sp-other'
    git(tmp_path, 'clone', '-q', '-b', 'speed-limits', str(car['sp_origin']), str(other))
    (other / 'NEWS').write_text('upstream\n')
    git(other, 'add', 'NEWS')
    git(other, 'commit', '-q', '-m', 'news')
    git(other, 'push', '-q', 'origin', 'speed-limits')
    done = send(car, '--signnet', 's1', FAKE_PUBLISH='new networks')
    assert done.returncode == 0, done.stderr
    assert 'speed-limits fast-forwarded' in done.stdout and (car['sp'] / 'NEWS').is_file()
    assert git(car['sp_origin'], 'rev-parse', f'speed-limits:{SUBMODULE}') == git(car['clone'], 'rev-parse', 'HEAD')


# -- the panel ------------------------------------------------------------------------------------------------
def car_installed(tmp_path: Path, repo: Path, monkeypatch) -> None:
    """What the action needs on the server: the script, the GPU venv, the clones and the build script."""
    script = repo / car_release.CAR_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "== Send to the car: SignNet ${1#--signnet=}"\necho "args: $*"\n'
                      'echo "== 1/5 Export"\necho "== 4/5 Move"\n')
    script.chmod(0o755)
    venv = tmp_path / 'venv'
    write(venv / 'bin/nnslr', '')
    for name in ('models', 'sunnypilot'):
        (tmp_path / name / '.git').mkdir(parents=True)
    write(tmp_path / 'scripts' / car_release.BUILD_SCRIPT, '')
    monkeypatch.setenv('NNSLR_GPU_VENV', str(venv))
    monkeypatch.setenv('NNSLR_MODELS_CLONE', str(tmp_path / 'models'))
    monkeypatch.setenv('NNSLR_SUNNYPILOT_CLONE', str(tmp_path / 'sunnypilot'))
    monkeypatch.setenv('NNSLR_OPENPILOT_SCRIPTS', str(tmp_path / 'scripts'))


def ready_payload():
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3, 'system': {'gpu': None}}


def test_send_to_car_says_why_it_cannot_run(root, checkout, tmp_path, monkeypatch):
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['send_to_car']['reason'] == \
        'scripts/send_to_car.sh is missing in this installation'
    car_installed(tmp_path, checkout, monkeypatch)
    monkeypatch.setenv('NNSLR_SUNNYPILOT_CLONE', str(tmp_path / 'nowhere'))
    assert panel._action_states(ready_payload(), [])['send_to_car']['reason'] == f'no sunnypilot clone in {tmp_path / "nowhere"}'
    monkeypatch.setenv('NNSLR_SUNNYPILOT_CLONE', str(tmp_path / 'sunnypilot'))
    assert panel._action_states(ready_payload(), [])['send_to_car']['reason'] == \
        'no classifier in use: choose one on the Models tab'
    servers_runs(root)
    # No confirmed frame is needed: it sends networks already trained.
    assert panel._action_states(ready_payload(), [])['send_to_car'] == {'label': 'Publish to the car', 'allowed': True, 'reason': ''}
    assert panel.models()['car_build_pending'] is False


def test_send_to_car_takes_only_a_finished_signnet_of_the_models_tab(root, checkout, tmp_path, monkeypatch):
    car_installed(tmp_path, checkout, monkeypatch)
    servers_runs(root)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'activities', list)
    for options in (None, {'signnet': 'nothing'}, {'signnet': 's3'}, {'signnet': 'r1'}, {'signnet': '../s1'},
                    {'signnet': '--rebuild'}):
        with pytest.raises(PanelError) as refused:
            panel.start('send_to_car', options)
        assert (refused.value.status, refused.value.reason) == (400, 'bad_option'), options
    with pytest.raises(PanelError) as refused:
        panel.start('networks_prepare', {'signnet': 's1'})
    assert refused.value.reason == 'bad_option'
    # A finished SignNet of another band or grid: refused before any job starts, in the Models tab's words.
    make_signnet(root, 'wide', band=[1344, 416], stride=4)
    for action in ('send_to_car', 'try_on_comma'):
        with pytest.raises(PanelError) as refused:
            panel.start(action, {'signnet': 'wide'})
        assert (refused.value.status, refused.value.reason) == (409, 'not_for_the_car'), action
        assert str(refused.value) == NOT_SELECTABLE['not_for_the_car'][1]
    assert panel.jobs() == []  # no job started
    job = panel.start('send_to_car', {'signnet': 's1'})
    done = finished(panel, job['id'])
    assert done['state'] == 'ok' and done['run'] == 's1'
    assert 'args: --signnet=s1\n' in full_log(panel, job['id'])
    assert [step['key'] for step in panel.job_progress(done, None)['steps']] == ['export', 'publish', 'record', 'move', 'build']
    # Build again: --rebuild only from 'on', nothing else reaches the command line, and only for this action.
    for value in ('yes', '--dry-run', True):
        with pytest.raises(PanelError) as refused:
            panel.start('send_to_car', {'signnet': 's1', 'rebuild': value})
        assert (refused.value.status, refused.value.reason) == (400, 'bad_option'), value
    with pytest.raises(PanelError) as refused:
        panel.start('try_on_comma', {'signnet': 's1', 'rebuild': 'on'})
    assert refused.value.reason == 'bad_option'
    time.sleep(1.1)  # a job's id is its second
    again = panel.start('send_to_car', {'signnet': 's1', 'rebuild': 'on'})
    assert finished(panel, again['id'])['state'] == 'ok'
    assert 'args: --signnet=s1 --rebuild\n' in full_log(panel, again['id'])


def test_the_signnet_option_over_http(root, checkout, tmp_path, monkeypatch):
    from review_helpers import RunningServer, json_of
    car_installed(tmp_path, checkout, monkeypatch)
    servers_runs(root)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    try:
        status, _, data = server.request('POST', '/api/panel/run/send_to_car', body={'signnet': 'unknown'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, _, data = server.request('POST', '/api/panel/run/send_to_car', body={'signnet': 's1'})
        assert status == 200, data
        finished(server.app.panel, json_of(data)['id'])
    finally:
        server.close()


def test_the_progress_follows_the_five_steps():
    log = '== Send to the car: SignNet s1\n== 1/5 Export the SignNet\n== 2/5 Publish them\n== 3/5 Record\n== 4/5 Move\n'
    running = progress('send_to_car', 'running', log)
    assert [step['state'] for step in running['steps']] == ['done', 'done', 'done', 'current', 'todo']
    assert running['percent'] == 17.0  # export 10, publish 5, record 2 of 100
