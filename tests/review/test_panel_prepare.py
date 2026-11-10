"""Prepare in the control panel: who proposes the signs (kept on the server, the best measured before a choice),
Qwen3-VL refused when the GPU is taken, the proposer and the end of each job, a proposer's measure and Propose again
with their own progress.

The helpers are fakes in a temporary checkout: nothing real is prepared or measured.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

import nnslr_tools.panel as panel_module
from nnslr_tools.panel import Panel, PanelError
from review_helpers import RunningServer, json_of, make_report

REPO = Path(__file__).resolve().parents[2]
FAKE = '#!/usr/bin/env bash\necho "$0 $*"\n'
# A GPU of 32 GB with 20 GB in use: Qwen3-VL (about 18 GB) does not fit.
BUSY_GPU = {'name': 'V100', 'utilization': 90, 'memory_used_mb': 20 * 1024, 'memory_total_mb': 32 * 1024,
            'temperature': 60, 'power_w': 200, 'processes': 1, 'process_list': []}


@pytest.fixture(autouse=True)
def only_this_checkout(monkeypatch):
    monkeypatch.setattr(panel_module, 'OWN_GROUPS', ('fetch', 'prepare', 'train', 'update', 'ai', 'other'))


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'helpers').mkdir(parents=True)
    shutil.copy(REPO / 'helpers/route_status.py', repo / 'helpers/route_status.py')
    for name in ('1_prepare.sh', '3_train.sh', 'update.sh', 'fetch_comma_segments.sh', 'measure_proposer.sh'):
        (repo / 'helpers' / name).write_text(FAKE)
        (repo / 'helpers' / name).chmod(0o755)
    return repo


@pytest.fixture
def root(tmp_path):
    data = tmp_path / 'data'
    make_report(data)
    return data


def finished(panel, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(j for j in panel.jobs() if j['id'] == job_id)
        if job['state'] != 'running':
            return job
        time.sleep(0.05)
    raise AssertionError(f'{job_id} still running')


def measured(root, proposer, found, read_right, seconds=1.0):
    folder = root / 'evaluations' / 'proposers'
    folder.mkdir(parents=True, exist_ok=True)
    speed = {'truth': 80, 'found': found, 'read_right': read_right, 'found_share': found / 80,
             'read_right_share': read_right / 80}
    (folder / f'{proposer}-test.json').write_text(json.dumps(
        {'proposer': proposer, 'partition': 'test', 'frames': 604, 'limited': False, 'seconds_per_frame': seconds,
         'measured_at': '2026-10-25T12:09:27+00:00', 'speed_signs': speed}))


def test_the_choice_is_the_best_measured_until_one_is_kept(root, checkout, monkeypatch):
    monkeypatch.setattr('nnslr_tools.preannotate.vlm_missing', lambda root: None)
    # Our networks whole (cutter, classifier and decider in use and applied).
    monkeypatch.setattr('nnslr_tools.preannotate.ours_missing', lambda root: None)
    panel = Panel(root, repo=checkout)
    # Nothing measured: the helpers' own default.
    assert (panel.proposers()['chosen'], panel.proposers()['best']) == ('vietnam', None)
    measured(root, 'vietnam', 27, 2, 0.5)
    measured(root, 'ours', 60, 51, 0.8)
    measured(root, 'qwen', 37, 24, 1.6)
    assert (panel.proposers()['chosen'], panel.proposers()['best']) == ('ours', 'ours')
    # A proposer that cannot run here is never the best.
    measured(root, 'qwen', 70, 60, 1.6)
    assert panel.proposers()['best'] == 'qwen'
    monkeypatch.setattr('nnslr_tools.preannotate.vlm_missing', lambda root: 'Qwen3-VL is not installed yet')
    assert panel.proposers()['best'] == 'ours'
    # Once chosen, kept for every browser (and a new server).
    assert panel.set_proposer({'proposer': 'vietnam'})['chosen'] == 'vietnam'
    assert Panel(root, repo=checkout).proposers()['chosen'] == 'vietnam'
    with pytest.raises(PanelError) as refused:
        panel.set_proposer({'proposer': 'rm -rf'})
    assert refused.value.reason == 'bad_option'
    with pytest.raises(PanelError) as refused:
        Panel(root, repo=checkout, actions_enabled=False).set_proposer({'proposer': 'ours'})
    assert refused.value.status == 403


def test_our_networks_propose_only_whole(root, checkout):
    import json
    from nnslr_tools import preannotate
    panel = Panel(root, repo=checkout)
    # Nothing in use: refused, and said why, on Prepare and on Evaluate's list of proposers.
    why = panel.status()['proposers']['ours']
    assert why and 'cutter, classifier, decider' in why
    assert next(item for item in panel.proposers()['proposers'] if item['key'] == 'ours')['unavailable'] == why
    with pytest.raises(PanelError) as refused:
        panel.start('prepare_again', {'proposer': 'ours'})
    assert refused.value.reason == 'not_available'
    # Cutter and classifier in use: still waiting for the decider.
    (root / 'models').mkdir(parents=True, exist_ok=True)
    (root / 'models' / 'active.json').write_text(json.dumps({'cutter': {'id': 'c'}, 'classifier': {'id': 'k'}, 'decider': None}))
    assert preannotate.ours_missing(root).endswith('decider is not in use yet')
    # All three in use: whole; their weights must still be on this server.
    (root / 'models' / 'active.json').write_text(json.dumps({'cutter': {'id': 'c'}, 'classifier': {'id': 'k'}, 'decider': {'id': 'd'}}))
    assert preannotate.ours_missing(root) is None
    with pytest.raises(preannotate.PreannotationError, match='cannot load a part'):
        preannotate.our_networks(root)
    # Not whole but chosen all the same, it stays chosen (Prepare says why, its buttons off), and is never the best.
    (root / 'models' / 'active.json').write_text(json.dumps({'cutter': {'id': 'c'}, 'classifier': {'id': 'k'}, 'decider': None}))
    assert panel.set_proposer({'proposer': 'ours'})['chosen'] == 'ours'
    assert panel.proposers()['best'] != 'ours'



def test_our_networks_take_signnet_as_the_cutter(root):
    from nnslr_tools import preannotate
    weights = {'s1/signnet-best.pt': 'cutter:s1/signnet', 'k1/classifier/reader-best.pt': 'classifier:k1/classifier',
               'd1/experiment-b/full/full-best.pt': 'decider:d1/experiment-b/full'}
    for path in weights:
        (root / 'runs' / path).parent.mkdir(parents=True, exist_ok=True)
        (root / 'runs' / path).write_bytes(b'weights')
    (root / 'models').mkdir(parents=True, exist_ok=True)
    (root / 'models' / 'active.json').write_text(json.dumps(
        {part: {'id': model_id} for part, model_id in zip(('cutter', 'classifier', 'decider'), weights.values())}))
    networks = preannotate.our_networks(root)
    assert networks['cutter'] == 'cutter:s1/signnet'
    assert networks['cutter_path'] == str((root / 'runs' / 's1' / 'signnet-best.pt').resolve())
    assert (networks['classifier'], networks['decider']) == ('classifier:k1/classifier', 'decider:d1/experiment-b/full')
    (root / 'runs' / 's1' / 'signnet-best.pt').unlink()
    with pytest.raises(preannotate.PreannotationError, match='cutter:s1/signnet has no runs/s1/signnet-best.pt'):
        preannotate.our_networks(root)

def test_qwen_is_refused_and_said_so_when_the_gpu_is_taken(root, checkout, monkeypatch):
    monkeypatch.setattr('nnslr_tools.preannotate.vlm_missing', lambda root: None)
    panel = Panel(root, repo=checkout)
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': dict(BUSY_GPU), 'disk': None, 'cpus': 4, 'load': None,
                                                  'loads': None, 'cpu': None, 'memory': None})
    reason = panel.status()['proposers']['qwen_gpu']
    assert reason.startswith('Qwen3-VL needs about 18 GB of the GPU and only 12.0 GB are free')
    with pytest.raises(PanelError) as refused:
        panel.start('prepare', {'proposer': 'qwen'})
    assert refused.value.reason == 'gpu_taken' and refused.value.message == reason + '.'
    monkeypatch.setattr(panel, 'system', lambda: {'gpu': {**BUSY_GPU, 'memory_used_mb': 1024}, 'disk': None, 'cpus': 4,
                                                  'load': None, 'loads': None, 'cpu': None, 'memory': None})
    assert panel.status()['proposers']['qwen_gpu'] is None


def test_a_job_knows_its_proposer_and_when_it_ended(root, checkout, monkeypatch):
    monkeypatch.setattr('nnslr_tools.preannotate.ours_missing', lambda root: None)
    (root / 'models').mkdir(parents=True, exist_ok=True)
    (root / 'models' / 'active.json').write_text(json.dumps({'cutter': {'id': 'cutter:c/x'}, 'classifier': {'id': 'classifier:k/y'},
                                                             'decider': {'id': 'decider:d/z'}}))
    panel = Panel(root, repo=checkout)
    job = panel.start('prepare_again', {'proposer': 'ours'})
    done = finished(panel, job['id'])
    assert done['proposer'] == 'ours' and done['ended_at'] and done['ended_at'] >= done['started_at']
    assert (panel.logs / (job['id'] + '.json')).is_file()
    # Its networks are the ones in use when it started, even after the Models tab changes them.
    (root / 'models' / 'active.json').write_text(json.dumps({'cutter': {'id': 'cutter:new/x'}}))
    assert next(item for item in panel.jobs() if item['id'] == job['id'])['networks'] == {
        'cutter': 'cutter:c/x', 'classifier': 'classifier:k/y', 'decider': 'decider:d/z'}
    # A measure started before the options were kept: its log names the proposer, and the label says it.
    (panel.logs / '20261025-115313-measure_proposer.log').write_text(
        '== helpers/measure_proposer.sh\nMeasuring the proposer qwen on the test frames...\n'
        'Preannotating 280/604: segment 17, frame 24, 0 proposals\n== finito con codice 0\n')
    measure = next(item for item in panel.jobs() if item['action'] == 'measure_proposer')
    assert (measure['label'], measure['proposer']) == ('Measure Qwen3-VL', 'qwen')
    assert next(item for item in panel.jobs() if item['action'] == 'prepare_again')['label'] == 'Propose again'
    (panel.logs / '20261025-090000-prepare.log').write_text('== helpers/1_prepare.sh --jobs=2\n'
                                                            'r1: proposing the signs (vietnam)...\n')
    assert next(item for item in panel.jobs() if item['action'] == 'prepare')['proposer'] == 'vietnam'


def test_a_preparation_started_without_a_choice_takes_the_one_of_prepare(root, checkout, monkeypatch):
    # The Overview's next step and Try again send no proposer: the choice kept on Prepare, never vietnam behind it.
    (checkout / 'helpers' / '1_prepare.sh').write_text('#!/usr/bin/env bash\necho "proposer=$NNSLR_PROPOSER"\n')
    (root / 'models').mkdir(parents=True, exist_ok=True)
    (root / 'models' / 'active.json').write_text(json.dumps({'cutter': {'id': 'cutter:c/x'}, 'classifier': {'id': 'classifier:k/y'},
                                                             'decider': {'id': 'decider:d/z'}}))
    panel = Panel(root, repo=checkout)
    # Nothing in use is whole yet: chosen all the same, then refused with the reason, not run with another one.
    assert panel.set_proposer({'proposer': 'ours'})['chosen'] == 'ours'
    monkeypatch.setattr('nnslr_tools.preannotate.ours_missing', lambda root: 'the decider is not in use yet')
    with pytest.raises(PanelError) as refused:
        panel.start('prepare')
    assert refused.value.reason == 'not_available'
    monkeypatch.setattr('nnslr_tools.preannotate.ours_missing', lambda root: None)
    job = finished(panel, panel.start('prepare')['id'])
    assert job['proposer'] == 'ours' and 'proposer=ours' in (panel.logs / (job['id'] + '.log')).read_text()
    # Prepare and the Overview say which networks those are.
    networks = {'cutter': 'cutter:c/x', 'classifier': 'classifier:k/y', 'decider': 'decider:d/z'}
    assert panel.proposers()['networks'] == networks
    assert panel.status()['proposers']['networks'] == networks and panel.status()['proposers']['chosen'] == 'ours'


def test_propose_again_running_has_the_progress_of_the_preparation(root, checkout):
    panel = Panel(root, repo=checkout)
    panel.logs.mkdir(parents=True)
    log = panel.logs / '20261025-103736-prepare_again.log'
    log.write_text('== helpers/1_prepare.sh --again\nr1: proposing the signs again (qwen)...\n')
    job = {'id': log.stem, 'action': 'prepare_again', 'state': 'running', 'size': log.stat().st_size,
           'started_at': '2026-10-25T10:37:36+00:00'}
    preparation = {'percent': 40.0, 'routes_done': 1, 'routes_total': 4, 'again': True, 'seconds_left': 600,
                   'proposer': 'Qwen3-VL', 'working': []}
    first = panel.job_progress(job, preparation)
    assert (first['percent'], first['seconds_left'], first['last_line']) == (40.0, 600, '')
    # The log did not grow, the preparation did: never the copy of before.
    assert panel.job_progress(job, {**preparation, 'percent': 55.0})['percent'] == 55.0


def test_the_choice_is_kept_over_http(root, checkout, monkeypatch):
    monkeypatch.setattr('nnslr_tools.preannotate.ours_missing', lambda root: None)
    server = RunningServer(root)
    server.app.panel = Panel(root, repo=checkout)
    try:
        status, response, data = server.request('POST', '/api/panel/proposer', body={'proposer': 'ours'})
        assert status == 200 and json_of(data)['chosen'] == 'ours'
        assert json_of(server.request('GET', '/api/panel/proposers')[2])['chosen'] == 'ours'
        status, response, data = server.request('POST', '/api/panel/proposer', body={'proposer': 'nobody'})
        assert status == 400 and json_of(data)['error'] == 'bad_option'
        status, response, data = server.request('POST', '/api/panel/proposer', body={'proposer': 'qwen'},
                                                headers={'Origin': 'http://evil.example'})
        assert status == 403
        assert json_of(server.request('GET', '/api/panel/proposers')[2])['chosen'] == 'ours'
    finally:
        server.close()
