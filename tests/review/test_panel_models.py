"""The Models tab's data: every trained cutter, classifier and decider of <data root>/runs by role
(nnslr_tools.model_roles), the model in use per role (<data root>/models/active.json), GET /api/panel/models and
POST /api/panel/models/active.

Every run is a synthetic folder in a temporary data root, the reports built by the real metric functions of
nnslr_tools.evaluation (helpers of test_panel_runs): nothing is trained or evaluated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import nnslr_tools.model_roles as model_roles
from nnslr_tools.evaluation import (detector_metrics, detector_size_metrics, human_reviewed_metrics, recall_by_width,
                                    score_frame, threshold_metrics)
from nnslr_tools.model_roles import ModelChoiceError, ModelIndex
from nnslr_tools.panel import Panel, PanelError
from nnslr_tools.runs import RunsIndex
from review_helpers import TOKEN, RunningServer, json_of, make_report, sign_in
from test_panel_runs import AI, HUMAN, experiment_result, make_classifier_run, make_run, write

THRESHOLDS = (0.05, 0.1, 0.25, 0.3, 0.5)
# Signs 10, 15, 30 and 50 px wide. At score 0.1 the 10 and 30 px ones are found and a 60 px box is false; at 0.25
# only the 30 px one is found.
SIZE_FRAMES = [
    (HUMAN, [[100, 100, 110, 110]], [{'box': [100, 100, 110, 110], 'score': 0.15}]),
    (AI, [[0, 0, 15, 15], [200, 200, 230, 230]],
     [{'box': [200, 200, 230, 230], 'score': 0.8}, {'box': [500, 500, 560, 560], 'score': 0.12}]),
    (AI, [[300, 300, 350, 350]], []),
]


def sizes_report(partition: str = 'test', ms: float = 12.5) -> dict:
    """An evaluation of nnslr_tools.evaluation.evaluate_detector as scripts/run_detector_sizes.sh writes it."""
    scored = {value: [{**score_frame(truth, predictions, score_threshold=value, iou_threshold=0.5), 'label_basis': basis}
                      for basis, truth, predictions in SIZE_FRAMES] for value in THRESHOLDS}
    main = scored[0.25]
    return {'schema_version': 1, 'task': 'detector', 'partition': partition, **detector_metrics(main),
            'recall_by_width': recall_by_width(main), 'human_reviewed': human_reviewed_metrics(main, detector_size_metrics),
            'score_threshold': 0.25, 'iou_threshold': 0.5,
            'by_score_threshold': {f'{value:g}': threshold_metrics(scored[value]) for value in THRESHOLDS},
            'input_size': [640, 360], 'input_size_label': '640x360', 'anchor_min_ratio': 0.1, 'checkpoint_epoch': 6,
            'inference': {'device_name': 'Tesla V100-PCIE-32GB', 'mean_ms_per_frame': ms}, 'images': []}


def make_size(run: Path, size: str, *, finished: bool = True, evaluated: bool = True) -> Path:
    folder = run / 'detector-sizes' / size
    write(folder / 'detector-best.pt', 'weights')
    if finished:
        write(folder / 'result.json', {'task': 'detector', 'architecture': 'ssdlite320_mobilenet_v3_large',
                                       'input_size': [int(side) for side in size.split('x')], 'anchor_min_ratio': 0.1,
                                       'augmentation': {'brightness': [0.4, 1.2]}, 'best_epoch': 6,
                                       'best_validation_loss': 1.1, 'initialization': 'imagenet1k_v1',
                                       'config': {'epochs': 20}, 'history': [{'epoch': n, 'validation_loss': 2.0 - n / 10}
                                                                             for n in range(1, 12)]})
    if evaluated:
        write(folder / 'evaluation/detector-test.json', sizes_report('test'))
        write(folder / 'evaluation/detector-validation.json', sizes_report('validation', ms=13.0))
    return folder


def touch(path: Path, seconds: float) -> None:
    os.utime(path, (seconds, seconds))


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'data'


@pytest.fixture
def models(root):
    runs = RunsIndex(root)
    runs.running_runs = lambda: {}
    return ModelIndex(runs)


def pipeline(root: Path, name: str = 'r1') -> Path:
    """A finished training run with the weights of its detector and reader."""
    run = make_run(root, name)
    write(run / 'detector/detector-best.pt', 'weights')
    write(run / 'reader/reader-best.pt', 'weights')
    return run


def experiment(root: Path, name: str = 'expB', variants=('crop', 'crop_geom')) -> Path:
    run = root / 'runs' / name
    for variant in variants:
        write(run / 'experiment-b' / variant / 'result.json', experiment_result(variant, 0.9, 0.6))
        write(run / 'experiment-b' / variant / f'{variant}-best.pt', 'weights')
    return run


def by_role(index: dict) -> dict:
    return {role['role']: role for role in index['roles']}


# -- the index --------------------------------------------------------------------------------------------
def test_without_runs_every_role_is_empty_and_nothing_is_in_use(models):
    index = models.index()
    assert [role['role'] for role in index['roles']] == ['cutter', 'classifier', 'decider', 'brain']
    assert all(role['versions'] == [] and role['active'] is None for role in index['roles'])
    assert (index['runs_dir_exists'], index['active_problem'], index['cutter_score_threshold']) == (False, None, 0.1)


def test_every_artifact_by_role_with_its_numbers(root, models):
    run = pipeline(root)
    make_size(run, '640x360')
    experiment(root)
    roles = by_role(models.index())
    assert {entry['id'] for entry in roles['cutter']['versions']} == {'cutter:r1/detector', 'cutter:r1/detector-sizes/640x360'}
    cutters = {entry['part']: entry for entry in roles['cutter']['versions']}

    # The detector of the pipeline: its report has no 0.1, so the numbers are at the report's 0.25; the widths come from
    # its frames (detector_report of test_panel_runs: 10 and 30 px found, 15 and 50 px missed, 2 false boxes).
    main = cutters['detector']
    assert (main['label'], main['detail'], main['finished'], main['in_use']) == (
        'Detector 320 × 320', 'old, to compare · training pipeline', True, False)
    # An old Detector: kept to compare, never chosen again.
    assert (main['selectable'], main['not_selectable_reason']) == (False, 'An old Detector: kept to compare.')
    assert (main['input_size'], main['input_size_recorded'], main['anchor_min_ratio'], main['anchor_recorded']) == ('320x320', False, 0.2, False)
    assert main['best_epoch'] == 2 and main['epochs_run'] == 2 and main['trained_at']
    test = main['test']
    assert (test['score_threshold'], test['at_cutter_score'], test['found'], test['truth_boxes'], test['recall']) == (0.25, False, 2, 4, 0.5)
    assert test['boxes_per_frame'] == 1.0 and test['false_per_frame'] == 0.5 and test['ms_per_frame'] is None
    assert [(row['key'], row['found'], row['truth']) for row in test['widths']] == [
        ('lt12', 1, 1), ('12-20', 0, 1), ('20-40', 1, 1), ('ge40', 0, 1)]
    assert test['human'] == {'images': 2, 'truth_boxes': 2, 'found': 1, 'recall': 0.5} and main['validation'] is None

    # A size of scripts/run_detector_sizes.sh: at the cutter's score 0.1, on test and validation.
    size = cutters['detector-sizes/640x360']
    assert (size['label'], size['detail'], size['input_size'], size['anchor_min_ratio'], size['best_epoch']) == (
        'Detector 640 × 360', 'old, to compare · input-size comparison', '640x360', 0.1, 6)
    assert size['augmentation'] is True and size['epochs_run'] == 11
    test = size['test']
    assert (test['score_threshold'], test['at_cutter_score'], test['found'], test['truth_boxes'], test['recall']) == (0.1, True, 2, 4, 0.5)
    assert test['boxes_per_frame'] == 1.0 and test['false_per_frame'] == pytest.approx(1 / 3)
    assert [(row['key'], row['found'], row['truth']) for row in test['widths']] == [
        ('lt12', 1, 1), ('12-20', 0, 1), ('20-40', 1, 1), ('ge40', 0, 1)]
    assert (test['ms_per_frame'], test['device'], test['input_size'], test['checkpoint_epoch']) == (12.5, 'Tesla V100-PCIE-32GB', '640x360', 6)
    assert test['human']['recall'] == 1.0 and size['validation']['partition'] == 'validation'

    [reader] = roles['classifier']['versions']
    assert (reader['id'], reader['label'], reader['finished'], reader['best_epoch'], reader['best_validation_accuracy']) == (
        'classifier:r1/reader', 'Reader', True, 2, 0.8)
    assert reader['test'] == {'samples': 6, 'accuracy': pytest.approx(4 / 6), 'human_samples': 2, 'human_accuracy': 1.0}
    assert reader['classes'] == 3
    assert reader['selectable'] is True and 'not_selectable_reason' not in reader

    deciders = {entry['variant']: entry for entry in roles['decider']['versions']}
    assert set(deciders) == {'crop', 'crop_geom'} and deciders['crop']['id'] == 'decider:expB/experiment-b/crop'
    crop = deciders['crop']
    assert (crop['label'], crop['detail'], crop['inputs'], crop['best_epoch'], crop['smoke_limit']) == (
        'Crop', 'experiment B', ['crop', 'box position'], 7, None)
    assert crop['validation'] == {'boxes': 40, 'reading_accuracy': 0.9, 'ownership_samples': 30, 'macro_f1': 0.6,
                                  'abstain_rate': 0.1, 'other_road_recall': 0.6, 'other_road_precision': 0.5,
                                  'other_road_support': None, 'other_road_found': None}
    assert crop['test']['reading_accuracy'] == 0.9 and crop['selectable'] is True
    assert (roles['cutter']['old_kind_in_use'], roles['cutter']['notice']) == (False, None)
    json.dumps(models.index(), allow_nan=False)


def test_newest_first_by_the_time_of_the_weights(root, models):
    run = pipeline(root)
    old = make_size(run, '320x320')
    new = make_size(run, '960x544')
    touch(run / 'detector/detector-best.pt', 1_000_000)
    touch(old / 'detector-best.pt', 2_000_000)
    touch(new / 'detector-best.pt', 3_000_000)
    roles = by_role(models.index())
    assert [entry['part'] for entry in roles['cutter']['versions']] == [
        'detector-sizes/960x544', 'detector-sizes/320x320', 'detector']


def test_weights_without_a_result_are_not_finished_and_files_unreadable_are_said(root, models):
    run = pipeline(root)
    make_size(run, '640x360', finished=False, evaluated=False)
    (run / 'evaluation/reader-test.json').write_text('{"task": "reader", "samp')
    roles = by_role(models.index())
    size = next(entry for entry in roles['cutter']['versions'] if entry['part'] == 'detector-sizes/640x360')
    assert (size['finished'], size['test'], size['input_size'], size['input_size_recorded']) == (False, None, '640x360', True)
    [reader] = roles['classifier']['versions']
    assert reader['test'] is None and reader['problems'] == ['evaluation/reader-test.json is unreadable']
    # Folders that are not a size, a variant without weights, a run without weights: not models.
    write(run / 'detector-sizes/notasize/detector-best.pt', 'x')
    write(run / 'detector-sizes/640x360.log', 'log')
    write(root / 'runs/expB/experiment-b/full/result.json', experiment_result('full', 0.8, 0.5))
    make_run(root, 'no-weights')
    roles = by_role(models.index())
    assert {entry['id'] for entry in roles['cutter']['versions']} == {'cutter:r1/detector', 'cutter:r1/detector-sizes/640x360'}
    assert roles['decider']['versions'] == [] and [entry['run'] for entry in roles['classifier']['versions']] == ['r1']


def test_a_weights_file_outside_its_run_is_not_listed(root, models, tmp_path):
    run = pipeline(root)
    outside = tmp_path / 'elsewhere'
    write(outside / 'detector-best.pt', 'weights')
    (run / 'detector-sizes/640x360').mkdir(parents=True)
    (run / 'detector-sizes/640x360/detector-best.pt').symlink_to(outside / 'detector-best.pt')
    roles = by_role(models.index())
    assert [entry['id'] for entry in roles['cutter']['versions']] == ['cutter:r1/detector']


def brain(root: Path, name: str, groups: list[str]) -> Path:
    """A finished brain run of ``nnslr train-brain`` that learnt from these groups."""
    run = root / 'runs' / name
    write(run / 'brain-last.pt', 'weights')
    write(run / 'result.json', {'groups': groups, 'seed': 1, 'rate_hz': 4, 'ahead': True, 'history': [{}],
                                'test': {'speed': {'frames': 10, 'right': 0.5, 'mean_error_kph': 3.0}}})
    return run


def test_a_brain_that_takes_the_drivers_inputs_cannot_be_chosen_and_the_one_in_use_says_so(root, models):
    """With the car's speed among its inputs, the brain of 10 Nov 2026 followed his speed on the road: such a brain stays
    in its run to compare, cannot be put in use, and while one chosen before is still in use Models › Brain says so."""
    road = ['signs', 'plate', 'plate_hours', 'hour', 'lead', 'country', 'plate_wet', 'sign_distance', 'motion']
    brain(root, 'b1', [*road[:6], 'pedals', 'speed', *road[6:]])
    brain(root, 'b2', road)
    with pytest.raises(ModelChoiceError) as refused:
        models.set_active('brain', 'brain:b1/brain', who='admin', via='sign-in')
    assert (refused.value.status, refused.value.reason) == (409, 'takes_the_driver')
    entry = models.set_active('brain', 'brain:b2/brain', who='admin', via='sign-in')
    roles = by_role(models.index())
    versions = {version['run']: version for version in roles['brain']['versions']}
    assert versions['b2']['selectable'] and not versions['b1']['selectable']
    assert versions['b1']['not_selectable_reason'] == model_roles.NOT_SELECTABLE['takes_the_driver'][0]
    assert (roles['brain']['old_kind_in_use'], roles['brain']['notice']) == (False, None)
    # One chosen before 10 Nov 2026: it stays in use until another is chosen, and the page says so.
    path = root / 'models' / 'active.json'
    path.write_text(json.dumps({**json.loads(path.read_text()), 'brain': {**entry, 'id': 'brain:b1/brain'}}))
    roles = by_role(models.index())
    assert roles['brain']['active']['found'] and roles['brain']['old_kind_in_use']
    assert roles['brain']['notice'] == model_roles.DRIVER_NOTICE
    assert (roles['cutter']['old_kind_in_use'], roles['cutter']['notice']) == (False, None)


# -- the model in use -------------------------------------------------------------------------------------
def test_choosing_the_model_in_use_writes_only_active_json(root, models):
    pipeline(root)
    make_signnet(root, 's1')
    make_signnet(root, 's2')
    experiment(root)
    entry = models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    assert (entry['id'], entry['who'], entry['via']) == ('cutter:s1/signnet', 'admin', 'sign-in') and entry['at']
    path = root / 'models' / 'active.json'
    saved = json.loads(path.read_text())
    assert saved == {'schema_version': 1, 'cutter': entry, 'classifier': None, 'decider': None, 'brain': None}
    # Atomic: the temporary file is renamed, nothing is left next to it.
    assert sorted(item.name for item in path.parent.iterdir()) == ['active.json']

    models.set_active('decider', 'decider:expB/experiment-b/crop_geom', who=None, via='token')
    roles = by_role(models.index())
    assert roles['cutter']['active']['id'] == 'cutter:s1/signnet' and roles['cutter']['active']['found']
    assert [entry['in_use'] for entry in roles['cutter']['versions'] if entry['run'] == 's1'] == [True]
    assert sum(entry['in_use'] for entry in roles['cutter']['versions']) == 1
    assert roles['decider']['active']['via'] == 'token' and roles['classifier']['active'] is None
    # Another cutter replaces the first; the other roles stay.
    models.set_active('cutter', 'cutter:s2/signnet', who='admin', via='sign-in')
    saved = json.loads(path.read_text())
    assert saved['cutter']['id'] == 'cutter:s2/signnet' and saved['decider']['id'] == 'decider:expB/experiment-b/crop_geom'


@pytest.mark.parametrize('role, model_id, status, reason', [
    ('reader', 'classifier:r1/reader', 400, 'bad_role'),
    (None, 'classifier:r1/reader', 400, 'bad_role'),
    ('cutter', None, 400, 'bad_id'),
    ('cutter', 42, 400, 'bad_id'),
    ('cutter', '../../etc/passwd', 400, 'bad_id'),
    ('cutter', 'cutter:r1/../../etc/detector', 400, 'bad_id'),
    ('cutter', 'cutter:r1/detector/../../x', 400, 'bad_id'),
    ('cutter', 'cutter:/abs/detector', 400, 'bad_id'),
    ('cutter', 'cutter:r1/detector-sizes/../detector', 400, 'bad_id'),
    ('cutter', 'cutter:r1/detector' + 'x' * 400, 400, 'bad_id'),
    ('cutter', 'classifier:r1/reader', 400, 'wrong_role'),
    # The signnet part is the cutter's alone.
    ('classifier', 'classifier:r1/signnet', 400, 'wrong_role'),
    ('cutter', 'cutter:nope/detector', 404, 'unknown_model'),
    ('cutter', 'cutter:nope/signnet', 404, 'unknown_model'),
    ('cutter', 'cutter:r1/detector-sizes/1280x720', 404, 'unknown_model'),
    # The old Detectors, finished or not, are kept to compare only.
    ('cutter', 'cutter:r1/detector', 409, 'compare_only'),
    ('cutter', 'cutter:r1/detector-sizes/640x360', 409, 'compare_only'),
    ('classifier', 'classifier:r2/reader', 409, 'not_finished'),
])
def test_a_choice_is_checked_against_the_index(root, models, role, model_id, status, reason):
    run = pipeline(root)
    make_size(run, '640x360', finished=False)
    write(root / 'runs/r2/reader/reader-best.pt', 'weights')
    with pytest.raises(ModelChoiceError) as error:
        models.set_active(role, model_id, who='admin', via='sign-in')
    assert (error.value.status, error.value.reason) == (status, reason)
    assert not (root / 'models').exists()


def test_an_unreadable_active_json_is_never_overwritten(root, models):
    pipeline(root)
    make_signnet(root, 's1')
    path = write(root / 'models' / 'active.json', '{"cutter": ')
    index = models.index()
    assert index['active_problem'] == 'models/active.json cannot be read'
    assert all(role['active'] is None for role in index['roles'])
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    assert (error.value.status, error.value.reason) == (409, 'active_unreadable') and path.read_text() == '{"cutter": '


def test_a_choice_whose_files_are_gone_and_entries_that_are_not_choices(root, models):
    pipeline(root)
    write(root / 'models' / 'active.json', {
        'cutter': {'id': 'cutter:gone/detector', 'who': 'admin', 'via': 'sign-in', 'at': '2026-10-24T16:00:00+02:00'},
        'classifier': {'id': '../../reader', 'who': 'x'}, 'decider': 'decider:expB/experiment-b/crop'})
    roles = by_role(models.index())
    assert roles['cutter']['active'] == {'id': 'cutter:gone/detector', 'who': 'admin', 'via': 'sign-in',
                                         'at': '2026-10-24T16:00:00+02:00', 'found': False}
    assert not any(entry['in_use'] for entry in roles['cutter']['versions'])
    assert roles['classifier']['active'] is None and roles['decider']['active'] is None


def test_a_failed_write_leaves_the_old_choice_and_no_temporary_file(root, models, monkeypatch):
    pipeline(root)
    make_signnet(root, 's1')
    models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    path = root / 'models' / 'active.json'
    before = path.read_text()

    def broken(source, target):
        raise OSError('disk full')
    monkeypatch.setattr(model_roles.os, 'replace', broken)
    with pytest.raises(OSError):
        models.set_active('classifier', 'classifier:r1/reader', who='admin', via='sign-in')
    assert path.read_text() == before and sorted(item.name for item in path.parent.iterdir()) == ['active.json']


def test_models_folder_leading_out_of_the_data_root_is_refused(root, models, tmp_path):
    pipeline(root)
    outside = tmp_path / 'outside-models'
    outside.mkdir()
    (root / 'models').symlink_to(outside)
    make_signnet(root, 's1')
    assert models.index()['active_problem'] == 'models/ of the data root leads outside it'
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    assert error.value.reason == 'bad_models_folder' and list(outside.iterdir()) == []


def test_a_read_only_panel_refuses_the_choice(root):
    pipeline(root)
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=False)
    assert [entry['id'] for entry in by_role(panel.models())['classifier']['versions']] == ['classifier:r1/reader']
    with pytest.raises(PanelError) as error:
        panel.set_active_model({'role': 'classifier', 'id': 'classifier:r1/reader'}, who='admin', via='sign-in')
    assert error.value.status == 403 and not (root / 'models').exists()
    panel = Panel(root, repo=root / 'no-checkout', actions_enabled=True)
    for body in ([], 'x', None):
        with pytest.raises(PanelError) as error:
            panel.set_active_model(body, who='admin', via='sign-in')
        assert error.value.status == 400


# -- over HTTP --------------------------------------------------------------------------------------------
def test_models_over_http_with_the_sign_in(tmp_path):
    make_report(tmp_path)
    run = pipeline(tmp_path)
    make_size(run, '640x360')
    make_signnet(tmp_path, 's1')
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    server.app.panel.runs_index.running_runs = lambda: {}
    active = tmp_path / 'models' / 'active.json'
    body = {'role': 'cutter', 'id': 'cutter:s1/signnet'}
    try:
        # Signed out: the page is the sign-in page, the API answers 401 and nothing is written.
        status, response, data = server.request('GET', '/models')
        assert status in (200, 401) and b'id="app"' in data and b'aria-current="page">Models' not in data
        assert server.request('GET', '/api/panel/models')[0] == 401
        assert server.request('POST', '/api/panel/models/active', body=body)[0] == 401
        assert not active.exists()

        cookie = sign_in(server)
        status, response, data = server.request('GET', '/models', headers=cookie)
        assert status == 200 and 'aria-current="page">Models</a>' in data.decode()
        status, response, data = server.request('GET', '/api/panel/models', headers=cookie)
        roles = by_role(json_of(data))
        assert status == 200 and [entry['id'] for entry in roles['classifier']['versions']] == ['classifier:r1/reader']
        assert roles['cutter']['active'] is None
        # Counts only: never the per-frame lists of the reports.
        assert len(data) < 20000

        # Same origin and JSON only, as the actions of the panel.
        status, response, data = server.request('POST', '/api/panel/models/active', body=body,
                                                headers={**cookie, 'Origin': 'http://evil.example'})
        assert status == 403 and json_of(data)['error'] == 'cross_origin'
        status, response, data = server.request('POST', '/api/panel/models/active', body=b'role=cutter',
                                                headers={**cookie, 'Content-Type': 'application/x-www-form-urlencoded'})
        assert status == 415
        status, response, data = server.request('POST', '/api/panel/models/active', body={'role': 'cutter', 'id': '/etc/passwd'},
                                                headers=cookie)
        assert status == 400 and json_of(data)['error'] == 'bad_id'
        status, response, data = server.request('POST', '/api/panel/models/active', body={'role': 'cutter', 'id': 'cutter:r2/detector'},
                                                headers=cookie)
        assert status == 404 and json_of(data)['error'] == 'unknown_model'
        status, response, data = server.request('POST', '/api/panel/models/active',
                                                body={'role': 'cutter', 'id': 'cutter:r1/detector-sizes/640x360'},
                                                headers=cookie)
        assert status == 409 and json_of(data)['error'] == 'compare_only'
        assert server.request('GET', '/api/panel/models/active', headers=cookie)[0] == 405
        assert not active.exists()

        status, response, data = server.request('POST', '/api/panel/models/active', body=body,
                                                headers={**cookie, 'Origin': f'http://127.0.0.1:{server.port}'})
        answer = json_of(data)
        assert status == 200 and answer['ok'] and answer['role'] == 'cutter'
        assert (answer['active']['id'], answer['active']['who'], answer['active']['via']) == (body['id'], 'admin', 'sign-in')
        assert json.loads(active.read_text())['cutter']['id'] == body['id']
        roles = by_role(json_of(server.request('GET', '/api/panel/models', headers=cookie)[2]))
        assert roles['cutter']['active']['found'] and [entry['in_use'] for entry in roles['cutter']['versions']].count(True) == 1

        # An agent with the access token in its header: recorded as such, without a name.
        status, response, data = server.request('POST', '/api/panel/models/active',
                                                body={'role': 'classifier', 'id': 'classifier:r1/reader'},
                                                headers={'X-Review-Token': TOKEN})
        assert status == 200 and (json_of(data)['active']['who'], json_of(data)['active']['via']) == (None, 'token')
    finally:
        server.close()


def test_on_127_0_0_1_the_choice_is_recorded_as_local(tmp_path):
    make_report(tmp_path)
    pipeline(tmp_path)
    server = RunningServer(tmp_path)
    server.app.panel.runs_index.running_runs = lambda: {}
    try:
        status, response, data = server.request('POST', '/api/panel/models/active',
                                                body={'role': 'classifier', 'id': 'classifier:r1/reader'})
        assert status == 200 and (json_of(data)['active']['who'], json_of(data)['active']['via']) == (None, 'local')
    finally:
        server.close()


# -- the classifier with "not a sign" (scripts/run_classifier.sh) ----------------------------------------------------
def test_the_classifier_with_not_a_sign_is_a_version_of_its_own(root, models):
    run = make_classifier_run(root)
    write(root / 'runs/r1/reader/reader-best.pt', 'weights')
    touch(root / 'runs/r1/reader/reader-best.pt', 1_000_000)
    roles = by_role(models.index())
    new, old = roles['classifier']['versions']
    assert (new['id'], new['run'], new['part'], new['label'], new['detail']) == (
        f'classifier:{run.name}/classifier', run.name, 'classifier', 'Classifier', 'with “not a sign”')
    assert (new['finished'], new['class_scheme'], new['classes'], new['best_epoch'], new['epochs'], new['epochs_run']) == (
        True, 'v2', 8, 2, 3, 2)
    assert (new['best_validation_balanced_accuracy'], new['best_validation_sign_accuracy'], new['best_validation_rejection']) == (
        0.8, 0.7, 0.9)
    assert (new['architecture'], new['initialization'], new['augmentation'], new['problems']) == (
        'mobilenet_v3_small', 'imagenet1k_v1', True, [])
    test = new['test']
    assert (test['samples'], test['accuracy'], test['human_samples'], test['human_accuracy'], test['family_accuracy']) == (
        5, 0.6, 1, 1.0, 0.8)
    assert (test['rejection_rate'], test['not_a_sign_crops'], test['not_a_sign_rejected']) == (0.75, 4, 3)
    assert (test['signs_rejected'], test['signs_rejected_rate']) == (1, 0.2)
    chain = new['chain']
    assert (chain['score_threshold'], chain['recall'], chain['correct'], chain['truth'], chain['recall_found']) == (0.1, 0.5, 4, 8, 0.75)
    assert (chain['surviving_false_per_frame'], chain['false_per_frame'], chain['human_recall']) == (0.25, 1.0, 1.0)
    assert (new['chain_old_reader']['recall'], new['chain_old_reader']['surviving_false_per_frame']) == (0.375, 1.0)
    # The cutter the chain was measured with (the id it recorded), and a classifier its report does not name.
    assert (chain['cutter_id'], chain['cutter_size'], chain['classifier_id']) == ('cutter:r1/detector', '320x320', None)
    # The reader of the training pipeline stays as it was.
    assert (old['id'], old['label'], old['detail']) == ('classifier:r1/reader', 'Reader', 'training pipeline')
    assert old['test'] == {'samples': 6, 'accuracy': pytest.approx(4 / 6), 'human_samples': 2, 'human_accuracy': 1.0}
    assert 'chain' not in old and 'class_scheme' not in old
    json.dumps(models.index(), allow_nan=False)


@pytest.mark.parametrize('role, model, expected', [
    ('cutter', {'id': None, 'checkpoint': '/srv/sl/data/runs/2026-10-24_1452/detector-sizes/1344x760/detector-best.pt'},
     'cutter:2026-10-24_1452/detector-sizes/1344x760'),
    ('cutter', {'id': None, 'checkpoint': 'data/runs/r1/detector/detector-best.pt'}, 'cutter:r1/detector'),
    ('cutter', {'id': 'cutter:k1/detector', 'checkpoint': '/elsewhere/detector-best.pt'}, 'cutter:k1/detector'),
    ('classifier', {'checkpoint': '/srv/sl/data/runs/2026-10-24_1452/reader/reader-best.pt'}, 'classifier:2026-10-24_1452/reader'),
    ('classifier', {'checkpoint': '/srv/sl/data/runs/c1/classifier/reader-best.pt'}, 'classifier:c1/classifier'),
    ('classifier', {'checkpoint': '/srv/sl/data/runs/c1/classifier/reader-last.pt'}, None),
    ('cutter', {'checkpoint': '/srv/sl/data/runs/r1/reader/reader-best.pt'}, None),
    ('cutter', {'checkpoint': None}, None),
])
def test_the_networks_of_a_chain_by_the_id_of_the_models_tab(role, model, expected):
    from nnslr_tools.runs import _chain_model_id
    assert _chain_model_id(role, model) == expected


def test_a_classifier_still_training_and_one_without_measures(root, models):
    make_classifier_run(root, 'c1', upto='classifier')
    make_classifier_run(root, 'c2', upto='evaluate')
    versions = {entry['run']: entry for entry in by_role(models.index())['classifier']['versions']}
    assert (versions['c1']['finished'], versions['c1']['test'], versions['c1']['chain'], versions['c1']['class_scheme']) == (
        False, None, None, 'v2')
    assert (versions['c2']['finished'], versions['c2']['test'], versions['c2']['chain']) == (True, None, None)
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('classifier', 'classifier:c1/classifier', who='admin', via='sign-in')
    assert (error.value.status, error.value.reason) == (409, 'not_finished')
    (root / 'runs/c2/classifier/result.json').write_text('{"task": "reader", ')
    entry = next(entry for entry in by_role(models.index())['classifier']['versions'] if entry['run'] == 'c2')
    assert entry['finished'] is False and entry['problems'] == ['classifier/result.json is unreadable']


def test_the_classifier_with_not_a_sign_can_be_put_in_use(root, models):
    run = make_classifier_run(root)
    write(root / 'runs/r1/reader/reader-best.pt', 'weights')
    model_id = f'classifier:{run.name}/classifier'
    entry = models.set_active('classifier', model_id, who='admin', via='sign-in')
    assert entry['id'] == model_id
    roles = by_role(models.index())
    assert roles['classifier']['active']['found']
    assert {item['id']: item['in_use'] for item in roles['classifier']['versions']} == {model_id: True, 'classifier:r1/reader': False}
    # The old id of a reader stays valid, and "classifier" is no part of another role.
    models.set_active('classifier', 'classifier:r1/reader', who='admin', via='sign-in')
    for role, wrong, reason in (('cutter', f'cutter:{run.name}/classifier', 404), ('classifier', f'classifier:{run.name}/detector', 404),
                                ('classifier', f'classifier:{run.name}/classifier/x', 400)):
        with pytest.raises(ModelChoiceError) as error:
            models.set_active(role, wrong, who='admin', via='sign-in')
        assert error.value.status == reason, wrong


def test_the_cutter_in_use_with_its_files(root, models):
    assert models.in_use('cutter') == (None, 'no cutter in use: choose one on the Models tab')
    pipeline(root)
    run = make_signnet(root, 's1')
    models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    version, why = models.in_use('cutter')
    assert (version['id'], version['kind'], why) == ('cutter:s1/signnet', 'signnet', '')
    (run / 'signnet-best.pt').unlink()
    assert models.in_use('cutter') == (None, 'the cutter in use is no longer on this server: choose another one on the Models tab')
    (root / 'models/active.json').write_text('{"cutter": ')
    assert models.in_use('cutter') == (None, 'models/active.json cannot be read')


def test_what_a_cutter_finds_and_the_speed_signs_alone(root, models):
    from test_panel_runs import make_cutter_run
    make_cutter_run(root, 'k1', families='all')
    make_cutter_run(root, 'k2', families='speed')
    cutters = {entry['id']: entry for entry in by_role(models.index())['cutter']['versions']}
    every = cutters['cutter:k1/detector-sizes/960x544']
    assert (every['families'], every['families_recorded'], every['input_size']) == ('all', True, '960x544')
    assert every['test']['by_family']['speed'] == {'truth': 3, 'found': 2, 'recall': pytest.approx(2 / 3)}
    assert (every['test']['families'], every['test']['recall']) == ('all', 0.5)
    assert 'other_sign' not in cutters['cutter:k2/detector-sizes/960x544']['test']['by_family']
    # Before it was recorded, a cutter found the speed signs only.
    write(root / 'runs/r1/detector/detector-best.pt', 'weights')
    old = {entry['id']: entry for entry in by_role(models.index())['cutter']['versions']}['cutter:r1/detector']
    assert (old['families'], old['families_recorded'], old['test']['by_family']) == ('speed', False, None)


# -- SignNet, the 5 best boxes, what runs on the car ------------------------------------------------------------
def signnet_metrics(speed: int, other: int, boxes: float) -> dict:
    """The metrics of nnslr_tools.signnet.measure on 4 frames with 6 speed signs and 4 other signs."""
    found = speed + other
    return {'true_positives': found, 'false_negatives': 10 - found, 'false_positives': round(boxes * 4) - found,
            'image_count': 4, 'truth_box_count': 10, 'predictions_per_image': boxes, 'recall_iou': found / 10,
            'recall_by_family': {'speed': {'truth': 6, 'found': speed, 'recall': speed / 6},
                                 'other_sign': {'truth': 4, 'found': other, 'recall': other / 4}},
            'recall_by_width': {'lt12': {'truth': 2, 'found': 1, 'recall': 0.5}}}


def make_signnet(root: Path, name: str = 's1', *, finished: bool = True, share: float | None = None,
                 band: list[int] | None = None, stride: int = 8, car: bool = True, car_centre: bool = False) -> Path:
    """A run folder as ``nnslr train-signnet`` leaves it: by default the car's band (896 x 288) and stride (8), so
    set_active('cutter', 'cutter:<name>/signnet') takes it; car False: a result.json written before measure's 'car';
    car_centre True: one with the lenient car measure too."""
    run = root / 'runs' / name
    write(run / 'signnet-best.pt', 'signnet weights')
    write(run / 'train.log', 'epoch 1\n')
    if share is not None:
        write(run / 'subset.json', {'share': share, 'segments_kept': 4})
    if finished:
        partition = {'by_score_threshold': {'0.05': signnet_metrics(5, 4, 9.0), '0.1': signnet_metrics(4, 2, 3.0),
                                            '0.3': signnet_metrics(3, 1, 1.5)},
                     'by_box_budget': {'3': signnet_metrics(3, 1, 3.0), '5': signnet_metrics(5, 2, 5.0)}}
        if car:
            partition['car'] = signnet_metrics(2, 1, 1.0)
        if car_centre:
            partition['car_centre'] = signnet_metrics(3, 2, 1.5)
        write(run / 'result.json', {'task': 'signnet', 'band': band or [896, 288], 'stride': stride, 'fuse_kernel': 1,
                                    'parameters': 62997, 'best_epoch': 27, 'epochs': 40, 'families': 'all',
                                    'history': [{'epoch': n} for n in range(1, 41)], 'test': partition})
    return run


def test_signnet_with_the_lenient_car_measure(root, models):
    """car_centre: what the car reads matched leniently (signnet.measure), beside car at IoU 0.5."""
    pipeline(root)
    make_signnet(root, 's1', car_centre=True)
    test = {entry['id']: entry for entry in by_role(models.index())['cutter']['versions']}['cutter:s1/signnet']['test']
    lenient = test['car_centre']
    assert (lenient['found'], lenient['truth_boxes'], lenient['recall'], lenient['precision']) == (5, 10, 0.5, pytest.approx(5 / 6))
    assert (lenient['score_threshold'], lenient['max_boxes'], lenient['by_family']['speed']['found']) == (0.3, 5, 3)
    assert (test['car']['found'], test['car']['recall']) == (3, 0.3)


def test_signnet_is_the_cutter_in_use_when_it_fits_the_car(root, models):
    pipeline(root)
    make_signnet(root, 's1')
    make_signnet(root, 's2', share=0.25)
    make_signnet(root, 's3', finished=False)
    make_signnet(root, 'wide', band=[1344, 416])
    make_signnet(root, 'fine', stride=4)
    cutters = {entry['id']: entry for entry in by_role(models.index())['cutter']['versions']}
    signnet = cutters['cutter:s1/signnet']
    assert (signnet['kind'], signnet['label'], signnet['detail']) == ('signnet', 'SignNet 896 × 288', 'the comma’s cutter')
    assert (signnet['finished'], signnet['in_use'], signnet['on_car'], signnet['problems']) == (True, False, False, [])
    assert (signnet['fits_the_car'], signnet['selectable']) == (True, True) and 'not_selectable_reason' not in signnet
    assert (signnet['input_size'], signnet['anchor_min_ratio'], signnet['stride'], signnet['parameters']) == ('896x288', None, 8, 62997)
    assert (signnet['best_epoch'], signnet['epochs_run'], signnet['families']) == (27, 40, 'all')
    test = signnet['test']
    # At the cutter's score, as a detector is judged on the same frames.
    assert (test['at_cutter_score'], test['recall'], test['boxes_per_frame']) == (True, 0.6, 3.0)
    assert test['by_family']['speed'] == {'truth': 6, 'found': 4, 'recall': pytest.approx(4 / 6)}
    # Among its 5 best boxes of each frame.
    assert test['budget'] == {'boxes': 5, 'truth_boxes': 10, 'found': 7, 'recall': 0.7, 'by_family': {
        'speed': {'truth': 6, 'found': 5, 'recall': pytest.approx(5 / 6)}, 'other_sign': {'truth': 4, 'found': 2, 'recall': 0.5}}}
    # What the car reads: its best 5 boxes of score 0.3 or more.
    car = test['car']
    assert (car['found'], car['truth_boxes'], car['recall'], car['false_per_frame'], car['boxes_per_frame']) == (
        3, 10, 0.3, 0.25, 1.0)
    assert (car['score_threshold'], car['max_boxes'], car['by_family']['speed']['found']) == (0.3, 5, 2)
    assert test['car_centre'] is None  # a result.json written before the lenient measure
    assert signnet['validation'] is None
    assert cutters['cutter:s2/signnet']['detail'] == 'the comma’s cutter · 25 % of the training drives'
    assert cutters['cutter:s2/signnet']['train_share'] == 0.25
    assert cutters['cutter:r1/detector']['kind'] == 'detector'
    # Not finished, not the car's band or grid, an old Detector: listed, never chosen.
    reasons = {'s3': ('not_finished', 'Not finished.', 'not finished'),
               'wide': ('not_for_the_car', 'Not the comma’s 896 × 288 band at stride 8: it cannot run on the car.',
                        'not the comma’s band or grid, to compare'),
               'fine': ('not_for_the_car', 'Not the comma’s 896 × 288 band at stride 8: it cannot run on the car.',
                        'not the comma’s band or grid, to compare')}
    for run, (reason, text, detail) in reasons.items():
        entry = cutters[f'cutter:{run}/signnet']
        assert (entry['selectable'], entry['not_selectable_reason'], entry['detail']) == (False, text, detail), run
        with pytest.raises(ModelChoiceError) as error:
            models.set_active('cutter', f'cutter:{run}/signnet', who='admin', via='sign-in')
        assert (error.value.status, error.value.reason) == (409, reason), run
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    assert (error.value.status, error.value.reason) == (409, 'compare_only')
    assert not (root / 'models').exists()

    # The one of the car's band and grid: written, read back, the cutter in use.
    entry = models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    assert json.loads((root / 'models/active.json').read_text())['cutter'] == entry
    active, problem = models.read_active()
    assert (active['cutter']['id'], problem) == ('cutter:s1/signnet', None)
    version, why = models.in_use('cutter')
    assert (version['id'], version['kind'], version['fits_the_car'], why) == ('cutter:s1/signnet', 'signnet', True, '')
    role = by_role(models.index())['cutter']
    assert role['active']['found'] and (role['old_kind_in_use'], role['notice']) == (False, None)
    assert [item['id'] for item in role['versions'] if item['in_use']] == ['cutter:s1/signnet']


def test_an_old_detector_already_in_use_is_still_read_back(root, models):
    pipeline(root)
    make_signnet(root, 's1')
    write(root / 'models/active.json', {'schema_version': 1, 'cutter': {
        'id': 'cutter:r1/detector', 'who': 'admin', 'via': 'sign-in', 'at': '2026-10-30T08:12:00+01:00'}})
    version, why = models.in_use('cutter')
    assert (version['id'], version['kind'], version['selectable'], why) == ('cutter:r1/detector', 'detector', False, '')
    role = by_role(models.index())['cutter']
    assert role['active']['id'] == 'cutter:r1/detector' and role['active']['found']
    assert (role['old_kind_in_use'], role['notice']) == (True, 'The cutter in use is an old Detector: choose a SignNet.')
    # Choosing a SignNet replaces it; a Detector cannot be chosen back.
    models.set_active('cutter', 'cutter:s1/signnet', who='admin', via='sign-in')
    assert by_role(models.index())['cutter']['old_kind_in_use'] is False
    with pytest.raises(ModelChoiceError) as error:
        models.set_active('cutter', 'cutter:r1/detector', who='admin', via='sign-in')
    assert error.value.reason == 'compare_only'


def test_a_result_that_is_not_a_signnet_is_unreadable(root, models):
    run = make_signnet(root, 's1')
    write(run / 'result.json', {'task': 'detector'})
    entry = by_role(models.index())['cutter']['versions'][0]
    assert (entry['id'], entry['finished'], entry['problems']) == ('cutter:s1/signnet', False, ['result.json is unreadable'])


def test_the_speed_signs_in_the_5_best_boxes_of_a_detector_from_the_boxes_of_each_frame():
    report = {'per_frame_boxes': {'max_boxes': 50}, 'images': [
        # Best first: a speed sign found by the 4th box, an other sign only by the 7th.
        {'truth_count': 2, 'truth_families': ['maximum_speed', 'other_sign'],
         'cutter_boxes': ['0 0 9 9 0.900 -1', '0 0 9 9 0.800 -1', 'garbage', '0 0 9 9 0.500 0', '0 0 9 9 0.400 -1',
                          '0 0 9 9 0.300 -1', '0 0 9 9 0.100 1']},
        {'truth_count': 1, 'truth_families': ['zone'], 'cutter_boxes': ['1 1 20 20 0.800 0']},
        {'truth_count': 0, 'truth_families': [], 'cutter_boxes': []},
    ]}
    assert model_roles.cutter_digest(report)['budget'] == {
        'boxes': 5, 'truth_boxes': 3, 'found': 2, 'recall': pytest.approx(2 / 3),
        'by_family': {'other_sign': {'truth': 1, 'found': 0, 'recall': 0.0}, 'speed': {'truth': 2, 'found': 2, 'recall': 1.0}}}
    # A report without the boxes of each frame: not measured.
    assert model_roles.cutter_digest({**report, 'per_frame_boxes': None})['budget'] is None
    assert model_roles.cutter_digest(sizes_report())['budget'] is None


def test_what_runs_on_the_car_is_marked_from_on_car_json(root, models):
    pipeline(root)
    make_signnet(root, 's1')
    write(root / 'models/on-car.json', {'schema_version': 1, 'repository': 'owner/models', 'commit': 'a' * 40,
                                        'version': '2026-10-28', 'recorded_at': '2026-10-28T18:00:00+00:00',
                                        'cutter': {'id': 'cutter:s1/signnet'}, 'classifier': {'id': 'classifier:r1/reader'}})
    index = models.index()
    marked = {entry['id'] for role in index['roles'] for entry in role['versions'] if entry['on_car']}
    assert marked == {'cutter:s1/signnet', 'classifier:r1/reader'}
    assert index['on_car'] == {'repository': 'owner/models', 'commit': 'a' * 40, 'version': '2026-10-28',
                               'recorded_at': '2026-10-28T18:00:00+00:00'}
    assert (index['on_car_problem'], index['box_budget']) == (None, 5)
    # Nothing is in use because of it.
    assert all(role['active'] is None for role in index['roles'])
    assert not (root / 'models/active.json').exists()
    # Ids it could not have written mark nothing.
    write(root / 'models/on-car.json', {'cutter': {'id': 'cutter:s1/../../signnet'}, 'classifier': {'id': 'cutter:r1/detector'},
                                        'commit': 'not a commit'})
    index = models.index()
    assert not any(entry['on_car'] for role in index['roles'] for entry in role['versions'])
    assert index['on_car']['commit'] is None
    write(root / 'models/on-car.json', '{"cutter": ')
    index = models.index()
    assert (index['on_car'], index['on_car_problem']) == (None, 'models/on-car.json cannot be read')


def test_the_overview_has_the_car_at_a_glance(root):
    # Nothing published, nothing in use yet.
    root.mkdir(parents=True)
    panel = Panel(root)
    assert panel.car_summary() == {'in_use': {'cutter': None, 'classifier': None, 'decider': None, 'brain': None},
                                   'on_car': None,
                                   'problem': None, 'build_pending': False}
    # A scene network of before 7 Nov 2026, in use and on the car, is no longer one of them.
    write(root / 'models/active.json', {'cutter': {'id': 'cutter:s2/signnet'}, 'classifier': {'id': 'classifier:k2/classifier'},
                                        'road_type': {'id': 'road_type:t1/road-type/gru'}})
    write(root / 'models/on-car.json', {'cutter': {'id': 'cutter:s1/signnet'}, 'classifier': {'id': 'classifier:k2/classifier'},
                                        'road_type': {'id': 'road_type:t1/road-type/gru'}})
    car = panel.status()['car']
    assert car['in_use'] == {'cutter': 'cutter:s2/signnet', 'classifier': 'classifier:k2/classifier', 'decider': None,
                             'brain': None}
    assert car['on_car'] == {'cutter': 'cutter:s1/signnet', 'classifier': 'classifier:k2/classifier', 'decider': None,
                             'brain': None}
    # The applicability network goes on the car too.
    write(root / 'models/active.json', {'decider': {'id': 'decider:expB/experiment-b/full'}})
    write(root / 'models/on-car.json', {'decider': {'id': 'decider:expB/experiment-b/full'}})
    car = panel.status()['car']
    assert car['in_use']['decider'] == car['on_car']['decider'] == 'decider:expB/experiment-b/full'
    write(root / 'models/on-car.json', '{"cutter": ')
    assert panel.car_summary()['problem'] == 'models/on-car.json cannot be read'


def _git(folder: Path, *args: str) -> None:
    import subprocess
    subprocess.run(['git', '-C', str(folder), '-c', 'user.name=test', '-c', 'user.email=test@example.com', *args],
                   check=True, capture_output=True)


def published_models(tmp_path: Path, root: Path) -> Path:
    """A clone of the published models repository (format 2), pushed, for SignNet s1 and the reader of r1."""
    import hashlib
    sha = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    clone = tmp_path / 'models-clone'
    clone.mkdir()
    (clone / 'signnet.onnx').write_bytes(b'signnet onnx')
    (clone / 'classifier.onnx').write_bytes(b'classifier onnx')
    write(clone / 'signnet.onnx.contract.json', {
        'task': 'signnet', 'checkpoint': 'data/runs/s1/signnet-best.pt', 'onnx_sha256': sha(b'signnet onnx'),
        'checkpoint_sha256': sha((root / 'runs/s1/signnet-best.pt').read_bytes())})
    write(clone / 'classifier.onnx.contract.json', {
        'task': 'reader', 'checkpoint': 'classifier:r1/reader', 'onnx_sha256': sha(b'classifier onnx'),
        'checkpoint_sha256': sha((root / 'runs/r1/reader/reader-best.pt').read_bytes())})
    write(clone / 'speed_limits.json', {'format': 2, 'version': '2026-10-28',
                                        'finder': {'file': 'signnet.onnx', 'contract': 'signnet.onnx.contract.json'},
                                        'classifier': {'file': 'classifier.onnx', 'contract': 'classifier.onnx.contract.json'}})
    origin = tmp_path / 'origin.git'
    _git(tmp_path, 'init', '--bare', '-q', str(origin))
    _git(clone, 'init', '-q', '-b', 'main')
    _git(clone, 'add', '-A')
    _git(clone, 'commit', '-q', '-m', 'models')
    _git(clone, 'remote', 'add', 'origin', str(origin))
    _git(clone, 'push', '-q', 'origin', 'main')
    _git(clone, 'remote', 'set-url', 'origin', 'git@github.com:owner/speed-models.git')
    return clone


def test_on_car_is_recorded_from_a_clone_of_the_published_models(root, models, tmp_path):
    pipeline(root)
    make_signnet(root, 's1')
    clone = published_models(tmp_path, root)
    record = model_roles.record_on_car(root, clone, dry_run=True)
    assert not (root / 'models').exists()
    record = model_roles.record_on_car(root, clone)
    assert (record['cutter']['id'], record['classifier']['id']) == ('cutter:s1/signnet', 'classifier:r1/reader')
    assert (record['repository'], record['version'], len(record['commit'])) == ('owner/speed-models', '2026-10-28', 40)
    assert json.loads((root / 'models/on-car.json').read_text()) == record
    marked = {entry['id'] for role in models.index()['roles'] for entry in role['versions'] if entry['on_car']}
    assert marked == {'cutter:s1/signnet', 'classifier:r1/reader'}
    assert not (root / 'models/active.json').exists()


def test_on_car_records_the_applicability_network_of_the_release(root, models, tmp_path):
    import hashlib
    pipeline(root)
    make_signnet(root, 's1')
    experiment(root, variants=('full',))
    clone = published_models(tmp_path, root)
    write(clone / 'decider.onnx', 'decider onnx')
    write(clone / 'decider.onnx.contract.json', {
        'task': 'decider', 'target': 'car', 'checkpoint': 'decider:expB/experiment-b/full',
        'onnx_sha256': hashlib.sha256((clone / 'decider.onnx').read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256((root / 'runs/expB/experiment-b/full/full-best.pt').read_bytes()).hexdigest()})
    manifest = json.loads((clone / 'speed_limits.json').read_text())
    write(clone / 'speed_limits.json', {**manifest, 'decider': {'file': 'decider.onnx', 'contract': 'decider.onnx.contract.json'}})
    _git(clone, 'add', '-A')
    _git(clone, 'commit', '-q', '-m', 'decider')
    _git(clone, 'remote', 'set-url', 'origin', str(tmp_path / 'origin.git'))
    _git(clone, 'push', '-q', 'origin', 'main')
    record = model_roles.record_on_car(root, clone)
    assert record['decider'] == {'id': 'decider:expB/experiment-b/full', 'file': 'decider.onnx',
                                 'onnx_sha256': hashlib.sha256(b'decider onnx').hexdigest(),
                                 'checkpoint_sha256': hashlib.sha256(b'weights').hexdigest()}
    marked = {entry['id'] for role in models.index()['roles'] for entry in role['versions'] if entry['on_car']}
    assert marked == {'cutter:s1/signnet', 'classifier:r1/reader', 'decider:expB/experiment-b/full'}
    # Other weights on this server than the published ones: not recorded.
    write(root / 'runs/expB/experiment-b/full/full-best.pt', 'other weights')
    with pytest.raises(ValueError, match='the weights of decider:expB/experiment-b/full on this server are not'):
        model_roles.record_on_car(root, clone, dry_run=True)


@pytest.mark.parametrize('change, message', [
    ('onnx', 'signnet.onnx is not the ONNX of its contract'),
    ('weights', 'the weights of cutter:s1/signnet on this server are not the ones of the published finder'),
    ('gone', 'the finder cutter:s1/signnet is not on this server'),
    ('unpushed', 'is on no remote branch'),
    ('format', 'format 2 expected'),
])
def test_on_car_is_refused_unless_the_clone_matches_this_server(root, tmp_path, change, message):
    pipeline(root)
    make_signnet(root, 's1')
    clone = published_models(tmp_path, root)
    if change == 'onnx':
        (clone / 'signnet.onnx').write_bytes(b'another onnx')
    elif change == 'weights':
        write(root / 'runs/s1/signnet-best.pt', 'retrained')
    elif change == 'gone':
        (root / 'runs/s1/signnet-best.pt').unlink()
    elif change == 'unpushed':
        (clone / 'README.md').write_text('more')
        _git(clone, 'add', 'README.md')
        _git(clone, 'commit', '-q', '-m', 'more')
    else:
        write(clone / 'speed_limits.json', {'format': 1})
        _git(clone, 'commit', '-q', '-am', 'format 1')
    with pytest.raises(ValueError, match=message):
        model_roles.record_on_car(root, clone)
    assert not (root / 'models/on-car.json').exists()


def test_on_car_refuses_a_clone_with_changes_not_committed(root, tmp_path):
    pipeline(root)
    make_signnet(root, 's1')
    clone = published_models(tmp_path, root)
    write(clone / 'speed_limits.json', json.loads((clone / 'speed_limits.json').read_text()) | {'version': 'x'})
    with pytest.raises(ValueError, match='changes not committed'):
        model_roles.record_on_car(root, clone)


def test_nnslr_on_car_prints_the_record(root, tmp_path, capsys):
    from nnslr_tools.cli import main
    pipeline(root)
    make_signnet(root, 's1')
    clone = published_models(tmp_path, root)
    assert main(['on-car', '--models', str(clone), '--data-root', str(root), '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['cutter']['id'] == 'cutter:s1/signnet'
    assert not (root / 'models/on-car.json').exists()
    assert main(['on-car', '--models', str(tmp_path / 'nowhere'), '--data-root', str(root)]) == 2


def test_an_old_signnet_result_gives_the_car_its_numbers_at_0_3(root, models):
    """A result.json written before measure's 'car': what the car reads is by_score_threshold 0.3."""
    pipeline(root)
    make_signnet(root, 's1', car=False)
    car = {entry['id']: entry for entry in by_role(models.index())['cutter']['versions']}['cutter:s1/signnet']['test']['car']
    assert (car['found'], car['recall'], car['false_per_frame'], car['score_threshold']) == (4, 0.4, 0.5, 0.3)


def test_the_compare_only_text_does_not_say_the_cutter_in_use_is_a_signnet():
    from nnslr_tools.model_roles import NOT_SELECTABLE
    assert 'in use' not in NOT_SELECTABLE['compare_only'][1] and 'SignNet' in NOT_SELECTABLE['compare_only'][1]


def test_a_classifier_says_its_split_and_the_classifier_it_started_from(root, models):
    from test_panel_runs import SPLIT_SHA
    make_classifier_run(root, 'c1')
    run = make_classifier_run(root, 'c2')
    # Before result.json recorded it, its split is the one of its training plan.
    versions = {entry['run']: entry for entry in by_role(models.index())['classifier']['versions']}
    assert (versions['c1']['split_sha256'], versions['c1']['start']) == (SPLIT_SHA, None)
    result = json.loads((run / 'classifier/result.json').read_text())
    result.update({'split_sha256': 'f' * 64, 'initialization': 'checkpoint',
                   'start': {'path': str(root / 'runs/c1/classifier/reader-best.pt'), 'sha256': 'a' * 64,
                             'split_sha256': SPLIT_SHA, 'mapillary_only': False, 'epoch': 2},
                   'limitations': ['start_of_another_split: the start chose or learnt its weights on another split']})
    write(run / 'classifier/result.json', result)
    entry = next(entry for entry in by_role(models.index())['classifier']['versions'] if entry['run'] == 'c2')
    assert entry['split_sha256'] == 'f' * 64
    assert entry['start'] == {'run': 'c1', 'part': 'classifier', 'mapillary_only': False, 'split_sha256': SPLIT_SHA,
                              'another_split': True}
    json.dumps(models.index(), allow_nan=False)
