"""The signs on every frame in the panel: "Find the signs on every frame" (helpers/sign_tracks.sh) on Models, its
options (a dry run, the drives), its refusals and its progress read from the log. Nothing is decoded: the helper is a
stand-in that prints its arguments."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools.job_progress import progress  # noqa: E402
from nnslr_tools.panel import ACTION_OPTIONS, ACTIONS, Panel, PanelError  # noqa: E402
from review_helpers import RunningServer, json_of  # noqa: E402
from test_panel import checkout, finished, full_log, root  # noqa: E402,F401  (fixtures and helpers)
from test_sign_tracks import drive  # noqa: E402

ROUTE = '000000ca--eb394eea21'
OTHER = '000000cb--247a3022f6'


def ready_payload(gpu_processes=0):
    return {'actions_enabled': True, 'routes': [], 'reviewed_routes': 0, 'min_routes': 3,
            'system': {'gpu': {'processes': gpu_processes}}}


def tracks_script(checkout, venv=True):
    """A stand-in for helpers/sign_tracks.sh that prints its arguments, and the .venv-gpu it needs."""
    script = checkout / 'helpers' / 'sign_tracks.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('#!/usr/bin/env bash\necho "arguments: $*"\n')
    script.chmod(0o755)
    if venv:
        nnslr = checkout / '.venv-gpu' / 'bin' / 'nnslr'
        nnslr.parent.mkdir(parents=True, exist_ok=True)
        nnslr.write_text('#!/bin/sh\n')
        nnslr.chmod(0o755)
    return script


def test_the_action_is_in_the_panel_with_its_options():
    assert ACTIONS['sign_tracks'] == {'label': 'Find the signs on every frame', 'group': 'train',
                                      'command': ['helpers/sign_tracks.sh']}
    assert ACTION_OPTIONS['sign_tracks'] == ('dry', 'routes')


def test_the_review_server_passes_the_drives_on():
    # review.py's own list of the keys it hands to Panel.start (a key left out of it is dropped without a word: the
    # drives chosen dropped would start every drive on the GPU).
    text = (Path(__file__).resolve().parents[2] / 'src/nnslr_tools/review.py').read_text(encoding='utf-8')
    assert "for key in ('routes',)" in text and "for key in ('dry', 'labeler', 'marks')" in text


def test_it_says_why_it_cannot_start(root, checkout):
    panel = Panel(root, repo=checkout)
    reason = lambda: panel._action_states(ready_payload(), [])['sign_tracks']['reason']  # noqa: E731
    assert reason() == 'helpers/sign_tracks.sh is missing in this installation'
    tracks_script(checkout, venv=False)
    assert reason() == 'no .venv-gpu: install it with scripts/setup_v100_training_env.sh'
    tracks_script(checkout)
    assert reason() == 'no drive with its video yet (Drive view makes it)'
    drive(root)
    # No confirmed sign needed, and a busy GPU leaves it allowed: a dry run on the CPU can start.
    assert panel._action_states(ready_payload(), [])['sign_tracks']['allowed'] is True
    assert panel._action_states(ready_payload(gpu_processes=2), [])['sign_tracks']['allowed'] is True


@pytest.fixture
def ready(root, checkout, monkeypatch):
    panel = Panel(root, repo=checkout)
    tracks_script(checkout)
    drive(root)
    drive(root, OTHER, minutes=1)
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


def test_every_drive_the_drives_chosen_and_the_dry_run_reach_the_helper(ready):
    assert arguments(ready, start(ready, 'sign_tracks')) == 'arguments: '
    assert arguments(ready, start(ready, 'sign_tracks', {'routes': f'{ROUTE}, {OTHER},{ROUTE}'})) == (
        f'arguments: --route={ROUTE} --route={OTHER}')
    assert arguments(ready, start(ready, 'sign_tracks', {'dry': 'on', 'routes': OTHER})) == (
        f'arguments: --dry-run --route={OTHER}')


@pytest.mark.parametrize('options', [
    {'dry': 'yes'}, {'dry': True}, {'dry': 'off'}, {'routes': ['x']}, {'routes': 'nope'}, {'routes': '../x'},
    {'routes': '000000cc--0000000000'}, {'labeler': 'x'}, {'marks': 'off'},
])
def test_a_bad_or_unknown_option_is_refused_and_nothing_starts(ready, options):
    with pytest.raises(PanelError) as refused:
        ready.start('sign_tracks', options)
    assert (refused.value.status, refused.value.reason) == (400, 'bad_option')
    assert ready.jobs() == []


def test_no_other_action_takes_the_drives(ready):
    for action in ('train_brain', 'regression', 'fetch', 'train_cutter'):
        with pytest.raises(PanelError) as refused:
            ready.start(action, {'routes': ROUTE})
        assert refused.value.reason == 'bad_option'


def test_a_real_run_is_refused_on_a_busy_gpu_but_not_a_dry_one(ready, monkeypatch):
    monkeypatch.setattr(ready, 'system', lambda: {'gpu': {'processes': 2}})
    with pytest.raises(PanelError) as refused:
        ready.start('sign_tracks', {})
    assert (refused.value.status, refused.value.reason) == (409, 'gpu_taken')
    assert ready.jobs() == []
    assert arguments(ready, start(ready, 'sign_tracks', {'dry': 'on'})) == 'arguments: --dry-run'


def test_the_review_server_refuses_a_dry_run_that_is_not_a_string(root, checkout):
    tracks_script(checkout)
    drive(root)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    server.app.panel.activities = list
    server.app.panel.system = lambda: {'gpu': {'processes': 0}}
    try:
        for body in ({'dry': True}, {'routes': 1}, {'routes': [ROUTE]}):
            status, _, data = server.request('POST', '/api/panel/run/sign_tracks', body=body)
            assert status == 400 and json_of(data)['error'] == 'bad_option', body
        assert server.app.panel.jobs() == []
        status, _, data = server.request('POST', '/api/panel/run/sign_tracks', body={'dry': 'on', 'routes': ROUTE})
        assert status == 200, data
        assert arguments(server.app.panel, json_of(data)) == f'arguments: --dry-run --route={ROUTE}'
    finally:
        server.close()


LOG = """== helpers/sign_tracks.sh
== 1/2 Check the GPU ==
GPU 0 free: 300 of 32768 MiB used, no other compute process
== 2/2 Find the signs on every frame ==
drives:       4
Networks: cutter cutter:c/signnet, classifier classifier:k/classifier on cuda
minute 1/3: segment 0, 1200 frames, 20 boxes, 50 fps
drive 1/4 000000ca--eb394eea21: 3600 frames, 60 boxes in 72 s
drive 2/4 000000cb--247a3022f6: done already
"""


def test_it_goes_drive_by_drive():
    result = progress('sign_tracks', 'running', LOG)
    assert [(step['key'], step['state']) for step in result['steps']] == [('checks', 'done'), ('frames', 'current')]
    # 2 + 2 of the 4 drives of the 98.
    assert result['percent'] == pytest.approx(51.0)
    assert result['detail'] == '2 of 4 drives done'
    assert result['step_progress']['text'] == 'drive 3 of 4'
    # A drive that failed is done too: the run goes on to the next.
    result = progress('sign_tracks', 'running', LOG + 'drive 3/4 000000cc--0000000000: failed: no video\n')
    assert result['detail'] == '3 of 4 drives done'
    assert progress('sign_tracks', 'ok', LOG)['percent'] == 100.0


def test_it_moves_within_a_drive_minute_by_minute():
    # Drive 3 of 4 has 2 of its 3 minutes read: 2 + 98 x (2 + 2/3) / 4.
    result = progress('sign_tracks', 'running', LOG + 'minute 2/3: segment 1, 1200 frames, 9 boxes, 50 fps\n')
    assert result['percent'] == pytest.approx(2 + 98 * (2 + 2 / 3) / 4, abs=0.05)
    assert result['detail'] == '2 of 4 drives done'
    assert result['step_progress']['text'] == 'drive 3 of 4'
    # One drive only: it no longer stays at the start for the whole run.
    one = '== 1/2 Check the GPU ==\n== 2/2 Find the signs on every frame ==\ndrives:       1\n' + ''.join(
        f'minute {j}/30: segment {j - 1}, 1200 frames, 9 boxes, 50 fps\n' for j in range(1, 29))
    assert progress('sign_tracks', 'running', one)['percent'] == pytest.approx(2 + 98 * 28 / 30, abs=0.05)


@pytest.mark.parametrize('line,text', [
    ('helpers/sign_tracks.sh: the GPU 0 is busy (31726 of 32768 MiB used, 2 other compute processes): wait',
     'The GPU was busy with another job: wait for it to finish, or start a dry run on the CPU.'),
    ('helpers/sign_tracks.sh: no nvidia-smi, no GPU to run on (a --dry-run runs on the CPU)',
     'No GPU answered: a dry run runs on the CPU.'),
])
def test_a_refused_gpu_says_so(line, text):
    assert progress('sign_tracks', 'failed', '== 1/2 Check the GPU ==\n' + line + '\n')['failure'] == text
