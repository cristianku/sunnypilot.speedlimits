"""Models › Regression: the drives he certified, the models run on them only where a result is missing, and the curves of a
drive. Results are written by hand here: nothing runs on a GPU."""
from __future__ import annotations

import json

import pytest
from review_helpers import RunningServer, json_of

from nnslr_tools import regression
from nnslr_tools.job_progress import progress
from nnslr_tools.panel import Panel, PanelError
from test_panel import checkout, finished, full_log, root  # noqa: F401  (fixtures and helpers)

ROUTE = '00000168--2b00afdc52'
OTHER = '00000170--0123456789'
T0 = 1_000_000_000_000
NS = 1_000_000_000

LOG = """== Regression ==
== 1/3 Run the driving models on the certified drives ==
regression item openpilot 1/4 00000168--2b00afdc52
SEGMENT 2/4 2026-11-01--00
"""


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes}}}


def regression_script(checkout):
    script = checkout / 'scripts' / 'run_regression.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "chosen: $NNSLR_REGRESSION_OPENPILOT / $NNSLR_REGRESSION_BRAINS"\n')
    script.chmod(0o755)
    return script


def plan(kph_at_5=100.0):
    return {'route': ROUTE, 'model': 'stock', 'horizons_s': [0.0, 2.0, 5.0, 10.0], 'segments': 1, 'left_out': [],
            't_ns': [T0 + second * NS for second in range(4)], 'kph': [[50.0, 60.0, kph_at_5, 120.0]] * 4}


def merged(speeds=(90.0,) * 12):
    return {'t0': T0, 'series': {'v_ego': [[T0 + second * NS, kph] for second, kph in enumerate(speeds)]}}


def test_a_drive_stays_certified_until_he_removes_it(root):
    regression.certify(root, ROUTE, 'cristian', [ROUTE])
    regression.certify(root, ROUTE, 'someone else', [ROUTE])
    assert regression.certified(root)[ROUTE]['who'] == 'cristian'
    with pytest.raises(regression.RegressionError) as refused:
        regression.certify(root, OTHER, 'cristian', [ROUTE])
    assert refused.value.status == 404
    regression.uncertify(root, ROUTE)
    assert regression.certified(root) == {}


def test_a_run_computes_only_what_is_missing_and_what_can_run(root, tmp_path):
    videos = tmp_path / 'videos'
    (videos / f'{ROUTE}--0').mkdir(parents=True)
    (videos / f'{ROUTE}--0' / 'ecamera.hevc').write_bytes(b'x')
    (videos / f'{OTHER}--0').mkdir()
    regression.certify(root, ROUTE, 'c', [ROUTE, OTHER])
    regression.certify(root, OTHER, 'c', [ROUTE, OTHER])
    regression.write_result(root, ROUTE, 'stock', plan())
    assert regression.wide_index(videos) == {ROUTE: (1, 1), OTHER: (0, 1)}
    # Small is there, Big is not; without the wide camera only what the car logged can run.
    assert regression.missing_driving(root, videos, ['stock', 'teacher', 'logged']) == [
        (ROUTE, ['teacher', 'logged']), (OTHER, ['logged'])]
    # A segment of wide camera that came later: the result does not cover the drive any more.
    (videos / f'{ROUTE}--1').mkdir()
    (videos / f'{ROUTE}--1' / 'ecamera.hevc').write_bytes(b'x')
    assert regression.driving_state(root, ROUTE, 'stock', regression.wide_segments(videos, ROUTE)) == 'more_wide'


def test_a_plan_is_drawn_on_the_moment_it_speaks_of():
    assert regression.plan_points(plan(), T0, 5) == [[5.0, 100.0], [6.0, 100.0], [7.0, 100.0], [8.0, 100.0]]


def test_his_speed_holds_between_two_changes_but_not_across_a_hole_in_the_logs():
    logs = {'t0': T0, 'span': [T0, T0 + 9 * NS], 'series': {'v_ego': [[T0, 50.0], [T0 + 3 * NS, 60.0]]},
            'spans': {0: [T0, T0 + 5 * NS], 2: [T0 + 7 * NS, T0 + 9 * NS]}, 'holes': [[0, 2]]}
    assert regression.own_speed(logs) == {0: 50.0, 1: 50.0, 2: 50.0, 3: 60.0, 4: 60.0, 5: 60.0, 7: 60.0, 8: 60.0}


def test_a_curve_says_how_far_it_stays_from_his_speed():
    own = regression.own_speed(merged((90.0, 90.0, 100.0)))
    assert regression.score([[0, 100.0], [1, 80.0], [2, 100.0], [9, 40.0]], own) == {
        'seconds': 3, 'mean_kph': 0.0, 'abs_kph': 6.7, 'above_s': 1}
    assert regression.score([[9, 40.0]], own) is None


def test_smartspeed_caps_the_model_and_a_failure_is_shown(root):
    models = [{'key': 'stock', 'label': 'Small'}, {'key': 'teacher', 'label': 'Big'}]
    brains = [{'key': 'b1', 'label': 'one'}]
    regression.write_result(root, ROUTE, 'stock', plan(100.0))
    regression.write_result(root, ROUTE, 'teacher', {'error': 'no wide camera'})
    regression.write_result(root, ROUTE, regression.brain_spec('b1'),
                            {'reader': 'r', 't_ns': [T0 + second * NS for second in range(5, 9)], 'kph': [80, 80, 120, 120]})
    view = regression.route_view(root, ROUTE, merged(), models, brains, horizon=5, brain='b1')
    assert {curve['key']: curve['kind'] for curve in view['curves']} == {'stock': 'driving', 'brain:b1': 'brain', 'car:stock': 'car'}
    assert next(curve for curve in view['curves'] if curve['kind'] == 'car')['points'] == [[5.0, 80], [6.0, 80], [7.0, 100.0], [8.0, 100.0]]
    assert view['failures'] == [{'key': 'teacher', 'label': 'Big', 'error': 'no wide camera'}]
    assert 'car:stock' not in {curve['key'] for curve in regression.route_view(root, ROUTE, merged(), models, brains, horizon=5,
                                                                               brain=None)['curves']}
    with pytest.raises(regression.RegressionError):
        regression.route_view(root, ROUTE, merged(), models, brains, horizon=3, brain=None)


def test_a_brain_of_another_reader_of_the_signs_runs_again(root):
    regression.write_result(root, ROUTE, regression.brain_spec('same'), {'reader': 'new', 't_ns': [], 'kph': []})
    regression.write_result(root, ROUTE, regression.brain_spec('older'), {'reader': 'old', 't_ns': [], 'kph': []})
    regression.write_result(root, ROUTE, regression.brain_spec('failed'), {'reader': 'new', 'error': 'x'})
    assert regression.brains_to_run(root, ROUTE, ['same', 'older', 'failed', 'never'], 'new') == ['older', 'failed', 'never']


def test_the_frozen_split_is_read_without_the_sunnypilot_clone(root):
    assert regression.frozen_split(root) is None
    (root / 'splits').mkdir(exist_ok=True)
    (root / 'splits' / 'speed-brain-split.json').write_text(json.dumps({'partitions': {ROUTE: ['train', 'val']}}))
    (root / 'splits' / 'speed-brain-frames.json').write_text(json.dumps({'frames': {ROUTE: ['0/1', '0/2']}}))
    split = regression.frozen_split(root)
    stretches = [{'start_ns': T0, 'end_ns': T0 + 10 * NS, 'kph': 80}]
    assert regression._labelled_in_train(ROUTE, stretches, T0, split) == 1.0
    assert regression._labelled_in_train(OTHER, stretches, T0, split) is None


def test_the_progress_goes_drive_by_drive_and_segment_by_segment():
    result = progress('regression', 'running', LOG)
    assert [(step['key'], step['state']) for step in result['steps']] == [('openpilot', 'current'), ('signs', 'todo'), ('brains', 'todo')]
    # One drive of four done and a quarter of the second: 31.25 % of the 70 of this step.
    assert result['percent'] == pytest.approx(21.9, abs=0.1)
    assert result['step_progress']['text'] == 'drive 2 of 4'


def test_the_run_needs_its_script_a_free_gpu_a_certified_drive_and_a_model(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    assert panel._action_states(ready_payload(), [])['regression']['reason'] == 'scripts/run_regression.sh is missing in this installation'
    regression_script(checkout)
    assert panel._action_states(ready_payload(gpu_processes=1), [])['regression']['reason'] == 'the GPU is busy with another process'
    assert panel._action_states(ready_payload(), [])['regression']['allowed'] is True
    monkeypatch.setattr(panel, 'activities', lambda: [])
    with pytest.raises(PanelError) as refused:
        panel.start('regression', {'openpilot': 'stock'})
    assert refused.value.status == 409
    regression.certify(root, ROUTE, 'c', [ROUTE])
    for options in ({}, {'openpilot': 'nothing'}, {'brains': 'nothing'}, {'openpilot': ['stock']}, {'openpilot': 'stock;rm'}):
        with pytest.raises(PanelError) as refused:
            panel.start('regression', options)
        assert refused.value.status == 400
    assert panel.jobs() == []
    job = panel.start('regression', {'openpilot': 'stock,teacher'})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert 'chosen: stock,teacher / ' in full_log(panel, job['id']).splitlines()


@pytest.mark.parametrize('key', ['openpilot', 'brains'])
def test_its_choices_are_refused_on_another_action(root, checkout, key):
    with pytest.raises(PanelError) as refused:
        Panel(root, repo=checkout).start('train_decider', {key: 'stock'})
    assert refused.value.reason == 'bad_option'


def test_the_page_certifies_lists_and_runs_through_the_site(root, checkout, monkeypatch):
    from nnslr_tools import drive_review
    regression_script(checkout)
    monkeypatch.setattr(drive_review, 'all_routes', lambda _root: [ROUTE])
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    try:
        status, _, data = server.request('POST', f'/api/regression/routes/{ROUTE}/certify', body={'certified': 'yes'})
        assert status == 400, data
        status, _, data = server.request('POST', f'/api/regression/routes/{OTHER}/certify', body={'certified': True})
        assert status == 404 and json_of(data)['error'] == 'unknown_drive'
        status, _, data = server.request('POST', f'/api/regression/routes/{ROUTE}/certify', body={'certified': True})
        assert status == 200, data
        assert ROUTE in regression.certified(root)
        status, _, data = server.request('GET', '/api/regression/routes')
        assert status == 200
        answer = json_of(data)
        assert [model['key'] for model in answer['models'][:3]] == ['stock', 'teacher', 'logged']
        assert answer['horizon'] == 5
        status, _, data = server.request('GET', f'/api/regression/routes/{ROUTE}?horizon=3')
        assert status == 400 and json_of(data)['error'] == 'bad_horizon'
        status, _, data = server.request('GET', f'/api/regression/routes/{ROUTE}x')
        assert status == 404
        # The models go through the site: a wrong one is refused, never dropped (nothing starts).
        status, _, data = server.request('POST', '/api/panel/run/regression', body={'openpilot': 'nothing'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, _, data = server.request('POST', '/api/panel/run/regression', body={'openpilot': 'logged'})
        assert status == 200, data
        job = json_of(data)['id']
        finished(server.app.panel, job)
        assert 'chosen: logged / ' in full_log(server.app.panel, job).splitlines()
        status, _, data = server.request('POST', f'/api/regression/routes/{ROUTE}/certify', body={'certified': False})
        assert status == 200 and regression.certified(root) == {}
    finally:
        server.close()
