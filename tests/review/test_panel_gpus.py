"""The two V100s of the panel (issue #29): each card read on its own, by nvidia-smi's index."""
from nnslr_tools import panel as panel_module
from nnslr_tools.panel import Panel

CARDS = ('0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 0, 0, 32768, 31, 25.1\n'
         '1, 00000000:C1:00.0, Tesla V100-PCIE-32GB, 87, 9120, 32768, 54, 160.3\n')
APPS = '00000000:C1:00.0, 429246, /root/.venv-gpu/bin/python, 9100\n00000000:C1:00.0, 429300, ffmpeg, 300\n'


def fake_nvidia(cards=CARDS, apps=APPS):
    def run(arguments):
        return cards if arguments[0].startswith('--query-gpu=') else apps
    return run


def test_each_card_with_its_own_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia())
    system = Panel(tmp_path).system()
    first, second = system['gpus']
    assert (first['index'], first['bus_id'], first['processes'], first['memory_used_mb']) == (0, '81:00.0', 0, 0)
    assert (second['index'], second['bus_id'], second['processes'], second['utilization']) == (1, 'c1:00.0', 2, 87)
    assert [process['name'] for process in second['process_list']] == ['/root/.venv-gpu/bin/python', 'ffmpeg']
    assert second['process_list'][0] == {'pid': 429246, 'name': '/root/.venv-gpu/bin/python', 'memory_mb': 9100}
    # What a page loaded before the update reads: the first card's numbers, the processes of both.
    assert system['gpu']['name'] == 'Tesla V100-PCIE-32GB'
    assert (system['gpu']['utilization'], system['gpu']['processes']) == (0, 2)


def test_no_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(panel_module, '_nvidia', lambda arguments: None)
    system = Panel(tmp_path).system()
    assert (system['gpu'], system['gpus']) == (None, [])


def test_bus_as_nvidia_smi_and_the_driver_write_it():
    assert panel_module._bus('00000000:81:00.0') == '81:00.0'
    assert panel_module._bus('0000:c1:00.0 ') == 'c1:00.0'


# -- Part B: one GPU job per card, so two trainings run at once -------------------------------------------------------
# The jobs are stand-ins that print the card they were given and sleep; nvidia-smi and the driver's files are fakes of
# the two V100s of CT 131 (index 0 on bus 81:00.0, device minor 1; index 1 on bus c1:00.0, minor 0). No CUDA program
# runs and no card is touched.
import builtins  # noqa: E402
import contextlib  # noqa: E402
import fcntl  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import pytest  # noqa: E402

from nnslr_tools import progress as progress_module  # noqa: E402
from nnslr_tools.panel import PanelError  # noqa: E402
from test_panel import checkout, finished, full_log, only_this_checkout, root  # noqa: E402,F401
from test_panel_training import classifier_script, cutter_in_use, ready_payload, signnet_script  # noqa: E402

# The readers as they are, before the fixtures put fakes in their place.
read_lock_holders = panel_module._gpu_lock_holders
tree_cards = panel_module._tree_cards
read_minor_buses = panel_module._gpu_minor_buses
read_minors = panel_module._nvidia_minors

FREE = ('0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 0, 0, 32768, 31, 25.1\n'
        '1, 00000000:C1:00.0, Tesla V100-PCIE-32GB, 0, 0, 32768, 30, 24.0\n')
ONE = '0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 0, 0, 32768, 31, 25.1\n'
# Device minor -> bus, as /proc/driver/nvidia/gpus/*/information writes them on CT 131.
MINORS = {1: '81:00.0', 0: 'c1:00.0'}
# A job that says the card it was given and its progress file, then sleeps ($FAKE_SLEEP).
GPU_SCRIPT = ('#!/usr/bin/env bash\necho "run:          ${1:-x}"\necho "arguments: $*"\n'
              'echo "gpu=${NNSLR_GPU_INDEX:-none} cuda=${CUDA_VISIBLE_DEVICES:-none} order=${CUDA_DEVICE_ORDER:-none}"\n'
              'echo "progress=${NNSLR_PROGRESS_FILE:-none}"\necho "record=${NNSLR_JOB_RECORD:-none}"\n'
              'echo "workers=${NNSLR_NUM_WORKERS:-none}"\nsleep "${FAKE_SLEEP:-0}"\n')


def cards(used0=0, used1=0):
    return (f'0, 00000000:81:00.0, Tesla V100-PCIE-32GB, 0, {used0}, 32768, 31, 25.1\n'
            f'1, 00000000:C1:00.0, Tesla V100-PCIE-32GB, 0, {used1}, 32768, 30, 24.0\n')


@pytest.fixture(autouse=True)
def opened(only_this_checkout, monkeypatch):  # noqa: F811
    """No card env from the machine running the tests, the driver's minors of CT 131, the networks that read the brain's
    signs in use (#31: without them it is refused), and the card files each process has open ({pid: {minor}}, none unless
    a test says so)."""
    from nnslr_tools import preannotate
    monkeypatch.setattr(preannotate, 'networks_in_use', lambda root: {
        'cutter': 'cutter:c1/signnet', 'classifier': 'classifier:k1/classifier', 'decider': 'decider:d1/experiment-b/crop_geom'})
    for name in ('NNSLR_GPU_INDEX', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NNSLR_PROGRESS_FILE', 'NNSLR_JOB_RECORD'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(panel_module, '_CARD_INDEX', {})
    monkeypatch.setattr(panel_module, '_gpu_minor_buses', lambda: dict(MINORS))
    files: dict[int, set[int]] = {}
    monkeypatch.setattr(panel_module, '_nvidia_minors', lambda pid: set(files.get(pid, ())))
    return files


@pytest.fixture
def panel(root, checkout):  # noqa: F811
    """The panel of the test checkout; its jobs still running are stopped at the end."""
    made = Panel(root, repo=checkout)
    yield made
    for job in made.jobs():
        if job['state'] == 'running':
            with contextlib.suppress(PanelError):
                made.stop(job['id'])
    for job in made.jobs():
        if job['state'] == 'running':
            finished(made, job['id'])


@pytest.fixture
def terminal(checkout):  # noqa: F811
    """Work started from a terminal in the checkout, in a session of its own: run(script) -> its process."""
    started = []

    def run(script, *arguments):
        process = subprocess.Popen(['bash', str(script), *arguments], cwd=checkout, start_new_session=True,
                                   env={**os.environ, 'FAKE_SLEEP': '30'}, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(process)
        return process
    yield run
    for process in started:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(5)


def said(panel, job, prefix, timeout=10.0):
    """The first line of a job's log that starts with prefix, once it is there."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in full_log(panel, job['id']).splitlines():
            if line.startswith(prefix):
                return line
        time.sleep(0.05)
    raise AssertionError(f'{job["id"]} never said {prefix!r}')


def record(panel, job):
    return json.loads((panel.logs / (job['id'] + '.json')).read_text())


def holders(panel):
    return [(card['index'], card['free'], card['holders']) for card in panel.system()['gpus']]


def refused(panel, action, options=None):
    with pytest.raises(PanelError) as error:
        panel.start(action, options)
    return error.value.status, error.value.reason, error.value.message


def next_second() -> None:
    """A job's id has the second it started: two jobs in the same second would share it."""
    time.sleep(1.01 - time.time() % 1)


def reasons(panel):
    return panel._action_states({**ready_payload(), 'system': panel.system()}, panel.activities())


def test_two_free_cards_run_two_trainings_each_on_its_own_card(panel, root, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setattr(os, 'cpu_count', lambda: 24)
    monkeypatch.delenv('NNSLR_NUM_WORKERS', raising=False)
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    classifier_script(checkout, GPU_SCRIPT)
    cutter_in_use(root, panel)
    cutter = panel.start('train_cutter')
    classifier = panel.start('train_classifier')
    # Each on its card, by nvidia-smi's index, CUDA counting in the same order and seeing only that one.
    for job, index in ((cutter, 0), (classifier, 1)):
        assert said(panel, job, 'gpu=') == f'gpu={index} cuda={index} order=PCI_BUS_ID'
        assert said(panel, job, 'progress=') == f'progress={root}/logs/progress-{job["id"]}.json'
        assert record(panel, job) == {'gpus': [index], 'own_progress': True}
    running = {job['id']: job['gpus'] for job in panel.running()['jobs']}
    assert running == {cutter['id']: [0], classifier['id']: [1]}
    assert {job['id']: job['gpus'] for job in panel.jobs()} == running
    assert holders(panel) == [(0, False, [{'job': cutter['id'], 'label': 'A cutter training'}]),
                              (1, False, [{'job': classifier['id'], 'label': 'A classifier training'}])]
    # One activity for each label and session, with its job and the job's cards.
    assert sorted((item['label'], item['job'], item['gpus']) for item in panel.activities() if item['group'] == 'train') == [
        ('classifier training', classifier['id'], [1]), ('cutter training', cutter['id'], [0])]
    # Half the cores each for the processes that read their data, as the scripts' default.
    assert [said(panel, job, 'workers=') for job in (cutter, classifier)] == ['workers=12', 'workers=12']
    # A third shares the card with the most room (issue #41): both as free, the one of lowest index; the cores shared
    # among the three.
    assert reasons(panel)['train_cutter'] == {'label': 'Train the cutter', 'allowed': True, 'reason': ''}
    next_second()
    third = panel.start('train_cutter')
    assert said(panel, third, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID' and said(panel, third, 'workers=') == 'workers=8'
    assert record(panel, third) == {'gpus': [0], 'own_progress': True}
    assert holders(panel)[0] == (0, False, [{'job': cutter['id'], 'label': 'A cutter training'},
                                            {'job': third['id'], 'label': 'A cutter training'}])
    # The memory each may take is kept for it while it starts: the next goes beside the classifier, then where most is
    # left.
    next_second()
    fourth = panel.start('train_classifier')
    assert said(panel, fourth, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    # No card with the room: the next waits, and is told what holds them.
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(30000, 29000), ''))
    busy = 'both GPUs are busy: a cutter training on GPU 0, a classifier training on GPU 1'
    assert refused(panel, 'train_cutter') == (409, 'busy', busy[0].upper() + busy[1:] + ': wait for it to finish.')
    assert reasons(panel)['train_cutter'] == {'label': 'Train the cutter', 'allowed': False, 'reason': busy}
    # The legacy key of a page loaded before two cards, as part A left it.
    assert panel.system()['gpu']['processes'] == 0


def test_the_brain_takes_one_card_and_the_regression_never_runs_beside_it(panel, root, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    script = checkout / 'scripts' / 'run_brain.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(GPU_SCRIPT)
    script.chmod(0o755)
    signnet_script(checkout, GPU_SCRIPT)
    monkeypatch.setattr(panel, '_brain_drives_missing', lambda: '')
    brain = panel.start('train_brain')
    assert said(panel, brain, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    assert record(panel, brain)['gpus'] == [0]
    assert holders(panel)[1] == (1, True, [])
    assert reasons(panel)['regression']['reason'] == 'A brain training is running'
    # A cutter beside it, on the other card; a second one shares the cutter's card, never the brain's.
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    assert reasons(panel)['train_cutter']['reason'] == ''
    next_second()
    second = panel.start('train_cutter')
    assert said(panel, second, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'


def test_the_cards_the_brain_borrows_for_its_signs_are_its_own_until_it_gives_them_back(panel, root, checkout,  # noqa: F811
                                                                                         monkeypatch):
    """#36: step 3 of the brain also reads on the other card when it is free (nnslr borrow-cards); the panel names that
    card as the brain's and starts no training there until the step gives it back."""
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setattr(panel_module, 'BORROW_RECHECK_SECONDS', 0)
    monkeypatch.setenv('FAKE_SLEEP', '30')
    script = checkout / 'scripts' / 'run_brain.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(GPU_SCRIPT)
    script.chmod(0o755)
    signnet_script(checkout, GPU_SCRIPT)
    monkeypatch.setattr(panel, '_brain_drives_missing', lambda: '')
    brain = panel.start('train_brain')
    path = panel.logs / (brain['id'] + '.json')
    assert said(panel, brain, 'record=') == f'record={path}'
    # A cutter holds card 1: the brain reads on its own card alone.
    cutter = panel.start('train_cutter')
    said(panel, cutter, 'gpu=')
    assert panel_module.borrow_cards(panel.logs, 0, path) == [0]
    assert 'borrowed' not in record(panel, brain)
    panel.stop(cutter['id'])
    finished(panel, cutter['id'])
    # Card 1 free: borrowed, written to the brain's record, and the brain's for the panel.
    assert panel_module.borrow_cards(panel.logs, 0, path) == [0, 1]
    assert record(panel, brain) == {'gpus': [0], 'own_progress': True, 'borrowed': [1]}
    assert [card for card in holders(panel) if not card[1]][1][2][0]['job'] == brain['id']
    assert reasons(panel)['train_cutter']['reason'] == 'both GPUs are busy with a brain training'
    assert refused(panel, 'train_cutter')[:2] == (409, 'busy')
    assert [job['gpus'] for job in panel.jobs() if job['id'] == brain['id']] == [[0, 1]]
    # Asked again before the second reading: still its own, not taken from itself.
    assert panel_module.borrow_cards(panel.logs, 0, path) == [0, 1]
    # Given back: a cutter starts on card 1.
    panel_module.release_cards(path)
    assert record(panel, brain) == {'gpus': [0], 'own_progress': True}
    # A job's id is its second: the first cutter's is taken.
    time.sleep(1.05 - time.time() % 1)
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'


def test_a_card_taken_while_the_brain_borrows_it_is_left_to_its_job(tmp_path, monkeypatch):
    """The cards are read again a moment after they are written: a job the panel started on one meanwhile keeps it."""
    answers = iter(([1, 2], [2]))
    monkeypatch.setattr(panel_module, '_free_of_others', lambda logs, own, skip: next(answers))
    monkeypatch.setattr(panel_module, 'BORROW_RECHECK_SECONDS', 0)
    path = tmp_path / '20261110-150000-train_brain.json'
    path.write_text(json.dumps({'gpus': [0], 'own_progress': True}))
    assert panel_module.borrow_cards(tmp_path, 0, path) == [0, 2]
    assert json.loads(path.read_text())['borrowed'] == [2]
    # Given by NNSLR_SIGN_GPUS: as given, the others than its own written.
    assert panel_module.borrow_cards(tmp_path, 0, path, [1, 0]) == [1, 0]
    assert json.loads(path.read_text())['borrowed'] == [1]
    # From a terminal: nothing written.
    monkeypatch.setattr(panel_module, '_free_of_others', lambda logs, own, skip: [1])
    assert panel_module.borrow_cards(tmp_path, 0, None) == [0, 1]


def test_an_import_or_the_whole_pipeline_runs_alone_either_way(panel, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    script = checkout / 'helpers' / '3_train.sh'
    script.write_text('#!/usr/bin/env bash\necho "train $*"\nsleep "${FAKE_SLEEP:-0}"\n')
    signnet_script(checkout, GPU_SCRIPT)
    dataset = panel.start('dataset')
    # The import takes no card.
    assert record(panel, dataset) == {'gpus': [], 'own_progress': True}
    assert [card['free'] for card in panel.system()['gpus']] == [True, True]
    assert refused(panel, 'train_cutter') == (409, 'busy', 'An import of the reviews is running: wait for it to finish.')
    assert reasons(panel)['train_cutter']['reason'] == 'An import of the reviews is running'
    panel.stop(dataset['id'])
    finished(panel, dataset['id'])
    cutter = panel.start('train_cutter')
    said(panel, cutter, 'gpu=')
    assert refused(panel, 'dataset') == (409, 'busy', 'A cutter training is running: wait for it to finish.')
    assert reasons(panel)['dataset']['reason'] == 'A cutter training is running'


@pytest.mark.parametrize('first, second, apart', [
    ('train_cutter', 'train_classifier', False), ('train_cutter', 'train_cutter', False),
    ('train_classifier', 'train_decider', False), ('train_brain', 'train_cutter', False),
    ('regression', 'train_classifier', False),
    ('simulate', 'train_cutter', False), ('simulate', 'train_brain', False), ('simulate', 'regression', False),
    ('simulate', 'simulate', True), ('simulate', 'dataset', True),
    ('train_brain', 'train_brain', True), ('regression', 'regression', True),
    ('measure_proposer', 'measure_proposer', True), ('road_labeler_train', 'road_labeler_train', True),
    ('road_labeler_predict', 'road_labeler_predict', True),
    ('train_brain', 'regression', True), ('train_brain', 'road_labeler_predict', True),
    ('regression', 'road_labeler_predict', True), ('road_labeler_train', 'road_labeler_predict', True),
    ('dataset', 'train_cutter', True), ('train_continue', 'train_brain', True),
    ('train_resume', 'train_decider', True), ('setup_vlm', 'train_classifier', True), ('dataset', 'dataset', True),
])
def test_the_pairs_of_the_train_group(first, second, apart):
    assert panel_module._pair_apart(first, second) is apart
    assert panel_module._pair_apart(second, first) is apart


def test_work_from_a_terminal_counts_for_the_pairs_by_its_kind(root):  # noqa: F811
    panel = Panel(root)
    brain = [{'group': 'train', 'label': 'brain training', 'pid': 1, 'job': None, 'gpus': [0]}]
    assert panel._train_clash('regression', brain) == 'A brain training is running'
    assert panel._train_clash('dataset', brain) == 'A brain training is running'
    assert panel._train_clash('train_cutter', brain) == ''
    # An activity of a job of the panel is that job's, checked as a job.
    assert panel._train_clash('regression', [{**brain[0], 'job': '20261110-120000-train_brain'}]) == ''


def test_a_dry_road_labeler_runs_on_the_cpu_beside_trainings_that_hold_every_card(panel, root, checkout,  # noqa: F811
                                                                                   monkeypatch):
    from test_panel_road_labeler import labeler_run, labeler_script, scanned
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(CARDS, APPS))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    labeler_script(checkout).write_text(GPU_SCRIPT.replace('sleep', ': sleep'))
    scanned(root)
    labeler_run(root)
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    busy = 'Both GPUs are busy: a cutter training on GPU 0, a program the server cannot name on GPU 1'
    assert refused(panel, 'road_labeler_train') == (409, 'gpu_taken',
                                                    busy + ': wait for it to finish, or start a dry run on the CPU.')
    dry = panel.start('road_labeler_train', {'dry': 'on'})
    assert finished(panel, dry['id'])['state'] == 'ok'
    assert said(panel, dry, 'arguments:') == 'arguments: train --dry-run'
    assert said(panel, dry, 'gpu=') == 'gpu=none cuda=none order=none'
    assert record(panel, dry) == {'gpus': [], 'own_progress': True}
    assert next(job for job in panel.jobs() if job['id'] == dry['id'])['gpus'] == []


def test_a_preparation_with_qwen_takes_the_card_with_the_most_free_memory(panel, checkout, monkeypatch):  # noqa: F811
    import nnslr_tools.preannotate as preannotate
    monkeypatch.setattr(preannotate, 'vlm_missing', lambda root: '')
    script = checkout / 'helpers' / '1_prepare.sh'
    script.write_text(GPU_SCRIPT)
    # 12.5 GB free on GPU 0, 28.8 on GPU 1: Qwen3-VL (about 18 GB) fits only on GPU 1.
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(20000, 3300), ''))
    job = panel.start('prepare', {'proposer': 'qwen'})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    assert record(panel, job) == {'gpus': [1], 'own_progress': True, 'proposer': 'qwen'}
    # Too little on both: refused, not run out of memory.
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(20000, 20000), ''))
    assert refused(panel, 'prepare_again', {'proposer': 'qwen'}) == (
        409, 'gpu_taken', 'Qwen3-VL needs about 18 GB of the GPU and only 12.5 GB are free: another job or program is '
        'using it.')


def test_a_card_with_a_compute_process_is_not_free(panel, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(CARDS, APPS))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    unnamed = [{'job': None, 'label': 'A program the server cannot name'}]
    assert holders(panel) == [(0, True, []), (1, False, unnamed)]
    assert reasons(panel)['train_cutter']['allowed'] is True
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    # A second shares the cutter's card, never the one of the program the server cannot name.
    assert reasons(panel)['train_cutter']['reason'] == ''
    next_second()
    second = panel.start('train_cutter')
    assert said(panel, second, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'


def test_a_job_started_before_the_cards_is_found_through_the_card_files_it_has_open(panel, root, checkout,  # noqa: F811
                                                                                     monkeypatch, opened):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    job = panel.start('train_cutter')
    said(panel, job, 'gpu=')
    # As a job started before part B: no 'gpus' in its record.
    (panel.logs / (job['id'] + '.json')).write_text(json.dumps({'own_progress': True}) + '\n')
    wrapper = int((panel.logs / (job['id'] + '.pid')).read_text())
    opened[wrapper] = {1}
    restarted = Panel(root, repo=checkout)
    assert next(item for item in restarted.jobs() if item['id'] == job['id'])['gpus'] == [0]
    assert holders(restarted) == [(0, False, [{'job': job['id'], 'label': 'A cutter training'}]), (1, True, [])]
    # Its card files not found: it holds every card.
    opened.clear()
    again = Panel(root, repo=checkout)
    assert next(item for item in again.jobs() if item['id'] == job['id'])['gpus'] is None
    assert [card['free'] for card in again.system()['gpus']] == [False, False]
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(30000, 30000), ''))
    assert refused(again, 'train_cutter') == (409, 'busy', 'Both GPUs are busy with a cutter training: wait for it to '
                                              'finish.')


def test_old_jobs_that_have_ended_without_a_record(panel):
    panel.logs.mkdir(parents=True, exist_ok=True)
    for job_id, first in (('20261101-120000-dataset', 'helpers/3_train.sh --dataset-only'),
                          ('20261101-120100-road_labeler_train', 'helpers/road_labeler.sh train --dry-run'),
                          ('20261101-120200-road_labeler_train', 'helpers/road_labeler.sh train'),
                          ('20261101-120300-train_cutter', 'scripts/run_signnet.sh 2026-11-01_1203-cutter')):
        (panel.logs / (job_id + '.log')).write_text(f'== {first}\n== finished with code 0\n')
    assert {job['id']: job['gpus'] for job in panel.jobs()} == {
        '20261101-120000-dataset': [], '20261101-120100-road_labeler_train': [],
        '20261101-120200-road_labeler_train': None, '20261101-120300-train_cutter': None}


def test_a_training_from_a_terminal_holds_the_card_it_has_open(panel, checkout, monkeypatch, opened, terminal):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    script = signnet_script(checkout, GPU_SCRIPT)
    process = terminal(script, 'x')
    opened[process.pid] = {0}
    activity = [item for item in panel.activities() if item['group'] == 'train']
    assert [(item['label'], item['job'], item['gpus']) for item in activity] == [('cutter training', None, [1])]
    assert holders(panel) == [(0, True, []), (1, False, [{'job': None, 'label': 'A cutter training'}])]
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    assert reasons(panel)['dataset']['reason'] == 'A cutter training is running'


def test_a_training_from_a_terminal_whose_card_is_not_found_holds_every_card(root, checkout, monkeypatch,  # noqa: F811
                                                                             terminal):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    process = terminal(signnet_script(checkout, GPU_SCRIPT), 'x')
    panel = Panel(root, repo=checkout)
    assert [item['gpus'] for item in panel.activities() if item['pid'] == process.pid] == [None]
    holder = [{'job': None, 'label': 'A cutter training'}]
    assert holders(panel) == [(0, False, holder), (1, False, holder)]
    # A cutter may share them (issue #41), with the memory the other may take kept for it on each.
    assert reasons(panel)['train_cutter']['reason'] == ''
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(29000, 29000), ''))
    assert reasons(panel)['train_cutter']['reason'] == 'both GPUs are busy with a cutter training'


def test_the_openpilot_distill_training_and_the_holder_of_its_gpu_lock(root, checkout, monkeypatch, opened,  # noqa: F811
                                                                         terminal):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    training = checkout / 'supervised.train.sh'
    training.write_text('#!/usr/bin/env bash\nsleep "${FAKE_SLEEP:-0}"\n')
    distill = terminal(training)
    opened[distill.pid] = {1}
    # Its lock held by the same work: one holder.
    monkeypatch.setattr(panel_module, '_gpu_lock_holders', lambda path=None: [distill.pid])
    panel = Panel(root, repo=checkout)
    assert [(item['label'], item['job'], item['gpus']) for item in panel.activities()] == [
        ('openpilot.distill training', None, [0])]
    named = [{'job': None, 'label': 'openpilot.distill training'}]
    assert holders(panel) == [(0, False, named), (1, True, [])]
    # A holder of the lock that is no activity (a helper of openpilot.distill): on the card it has open.
    other = checkout / 'helpers' / 'route_speed.sh'
    other.write_text('#!/usr/bin/env bash\nsleep "${FAKE_SLEEP:-0}"\n')
    helper = terminal(other)
    opened[helper.pid] = {0}
    monkeypatch.setattr(panel_module, '_gpu_lock_holders', lambda path=None: [distill.pid, helper.pid])
    panel = Panel(root, repo=checkout)
    assert holders(panel) == [(0, False, named), (1, False, named)]
    signnet_script(checkout)
    assert reasons(panel)['train_cutter']['reason'] == 'both GPUs are busy with openpilot.distill training'


def test_the_holder_of_the_gpu_lock_is_read_from_proc_locks(tmp_path):
    lock = tmp_path / 'v100-gpu.lock'
    lock.touch()
    assert read_lock_holders(str(lock)) == []
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert read_lock_holders(str(lock)) == [os.getpid()]
    assert read_lock_holders(str(lock)) == []
    assert read_lock_holders(str(tmp_path / 'missing.lock')) == []


def test_the_cards_of_a_process_and_its_children(monkeypatch, opened):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    parent = subprocess.Popen(['bash', '-c', 'sleep 30; true'], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not panel_module._children(parent.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        (child,) = panel_module._children(parent.pid)
        assert panel_module._session_of(child) == parent.pid
        assert tree_cards([parent.pid]) is None
        opened[child] = {0}
        assert tree_cards([parent.pid]) == [1]
        opened[parent.pid] = {1}
        assert tree_cards([parent.pid]) == [0, 1]
        # A process with every card open beside one with its own only (wandb's system monitor beside CUDA): the
        # job's card is that one.
        opened[parent.pid] = {0, 1}
        assert tree_cards([parent.pid]) == [1]
        # A card file the driver does not list: not known, so every card.
        opened[child] = {7}
        assert tree_cards([parent.pid]) is None
    finally:
        os.killpg(parent.pid, signal.SIGTERM)
        parent.wait(5)


def test_nvidia_smi_in_a_job_is_not_the_job_on_every_card(tmp_path, monkeypatch, opened):
    """A script checks its card with nvidia-smi -i, which opens the files of every card: not the job's cards."""
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    smi = tmp_path / 'nvidia-smi'
    smi.symlink_to(subprocess.run(['which', 'sleep'], capture_output=True, text=True, check=True).stdout.strip())
    parent = subprocess.Popen(['bash', '-c', '"$0" 30; true', str(smi)], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not panel_module._children(parent.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        (child,) = panel_module._children(parent.pid)
        deadline = time.monotonic() + 5
        while panel_module._command_name(child) != 'nvidia-smi' and time.monotonic() < deadline:
            time.sleep(0.02)
        assert panel_module._command_name(child) == 'nvidia-smi'
        opened[child] = {0, 1}
        assert tree_cards([parent.pid]) is None
        opened[parent.pid] = {1}
        assert tree_cards([parent.pid]) == [0]
    finally:
        os.killpg(parent.pid, signal.SIGTERM)
        parent.wait(5)


def test_one_card_as_before(panel, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(ONE, '00000000:81:00.0, 4242, llama-server, 9000\n'))
    assert reasons(panel)['train_cutter']['reason'] == 'the GPU is busy with another process'
    assert refused(panel, 'train_cutter') == (409, 'busy', 'The GPU is busy with another process: wait for it to finish.')
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(ONE, ''))
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    # A second shares the card while it has the room (issue #41).
    assert reasons(panel)['train_cutter']['reason'] == ''
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(ONE.replace(', 0, 0, 32768', ', 0, 30000, 32768'), ''))
    assert reasons(panel)['train_cutter']['reason'] == 'the GPU is busy with a cutter training'
    assert refused(panel, 'train_cutter') == (409, 'busy', 'The GPU is busy with a cutter training: wait for it to finish.')


def test_without_nvidia_smi_as_before(panel, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    assert panel.system()['gpus'] == [] and reasons(panel)['train_cutter']['allowed'] is True
    cutter = panel.start('train_cutter')
    # No card is named to it: CUDA sees what the server sees.
    assert said(panel, cutter, 'gpu=') == 'gpu=none cuda=none order=none'
    assert record(panel, cutter) == {'gpus': None, 'own_progress': True}
    assert next(job for job in panel.jobs() if job['id'] == cutter['id'])['gpus'] is None
    assert reasons(panel)['train_cutter']['reason'] == 'the GPU is busy with a cutter training'
    assert refused(panel, 'train_cutter')[:2] == (409, 'busy')


def test_the_progress_file_of_a_job(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_DATA_ROOT', str(tmp_path / 'data'))
    assert progress_module.progress_path() == tmp_path / 'data' / 'logs' / 'progress.json'
    own = tmp_path / 'data' / 'logs' / 'progress-20261110-120000-train_cutter.json'
    monkeypatch.setenv('NNSLR_PROGRESS_FILE', str(own))
    assert progress_module.progress_path() == own
    progress_module.report('detector', 4, 4, 'epoch 4')
    assert json.loads(own.read_text())['task'] == 'detector'
    assert not (tmp_path / 'data' / 'logs' / 'progress.json').exists()


def test_each_job_reads_its_own_progress_and_an_old_job_the_shared_file(root, monkeypatch):  # noqa: F811
    panel = Panel(root)
    seen = []
    monkeypatch.setattr(panel_module, 'with_step_progress',
                        lambda result, action, report, setup, download: seen.append(report) or result)
    panel.logs.mkdir(parents=True, exist_ok=True)
    started = datetime(2026, 11, 10, 12, 0).astimezone()
    at = started.timestamp() + 5
    new, old = '20261110-120000-train_cutter', '20261110-120000-train_classifier'
    (panel.logs / (new + '.json')).write_text(json.dumps({'gpus': [0], 'own_progress': True}))
    (panel.logs / (old + '.json')).write_text(json.dumps({'proposer': 'ours'}))
    (root / 'logs' / f'progress-{new}.json').write_text(json.dumps({'task': 'detector', 'done': 2, 'total': 4, 'at': at}))
    (root / 'logs' / 'progress.json').write_text(json.dumps({'task': 'reader', 'done': 1, 'total': 3, 'at': at}))
    for job_id in (new, old):
        panel._with_step_progress({}, {'id': job_id, 'action': job_id.split('-', 2)[2],
                                       'started_at': started.isoformat(timespec='seconds')})
    assert [report['task'] for report in seen] == ['detector', 'reader']
    # A report from before the job started is not its.
    seen.clear()
    later = (started.replace(minute=5)).isoformat(timespec='seconds')
    panel._with_step_progress({}, {'id': new, 'action': 'train_cutter', 'started_at': later})
    assert seen == [None]


def test_the_progress_files_of_jobs_that_left_the_history_are_removed(root):  # noqa: F811
    panel = Panel(root)
    logs = root / 'logs'
    panel.logs.mkdir(parents=True, exist_ok=True)
    kept = '20261110-120000-train_cutter'
    (panel.logs / (kept + '.log')).write_text('== scripts/run_signnet.sh x\n== finished with code 0\n')
    for name in (kept, '20200101-000000-train_cutter'):
        (logs / f'progress-{name}.json').write_text('{}\n')
    (logs / 'progress.json').write_text('{}\n')
    panel._prune_progress()
    assert sorted(path.name for path in logs.glob('progress*.json')) == ['progress-' + kept + '.json', 'progress.json']


def test_a_run_name_handed_out_is_not_handed_out_again(root, monkeypatch):  # noqa: F811
    class Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 10, 12, 0)
    monkeypatch.setattr(panel_module, 'datetime', Fixed)
    panel = Panel(root)
    # Two trainings started in the same minute, before the first made its run folder.
    assert panel.new_run_name('cutter') == '2026-11-10_1200-cutter'
    assert panel.new_run_name('cutter') == '2026-11-10_1200-cutter-2'
    assert panel.new_run_name('classifier') == '2026-11-10_1200-classifier'
    (root / 'runs' / '2026-11-10_1200-cutter-3').mkdir(parents=True)
    assert panel.new_run_name('cutter') == '2026-11-10_1200-cutter-4'


# -- What the reviewers of part B asked to be tested ------------------------------------------------------------------
def test_a_process_gone_before_its_session_is_read_is_no_activity(panel, checkout, monkeypatch, terminal):  # noqa: F811
    """A DataLoader worker of a job, made again each epoch, can go between its command line and its stat: it is left
    out, not taken for work from a terminal that holds every card."""
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    process = terminal(signnet_script(checkout, GPU_SCRIPT), 'x')
    session_of = panel_module._session_of
    monkeypatch.setattr(panel_module, '_session_of', lambda pid: None if pid == process.pid else session_of(pid))
    assert [item for item in panel.activities() if item['pid'] == process.pid] == []
    assert holders(panel) == [(0, True, []), (1, True, [])]


def test_two_trainings_of_the_same_kind_from_a_terminal_are_two_activities(panel, checkout, monkeypatch, opened,  # noqa: F811
                                                                           terminal):
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    script = signnet_script(checkout, GPU_SCRIPT)
    first, second = terminal(script, 'a'), terminal(script, 'b')
    opened[first.pid], opened[second.pid] = {1}, {0}
    activity = sorted((item['pid'], item['job'], item['gpus']) for item in panel.activities() if item['group'] == 'train')
    assert activity == sorted([(first.pid, None, [0]), (second.pid, None, [1])])
    holder = [{'job': None, 'label': 'A cutter training'}]
    assert holders(panel) == [(0, False, holder), (1, False, holder)]
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(30000, 30000), ''))
    assert refused(panel, 'train_cutter') == (409, 'busy', 'Both GPUs are busy with a cutter training: wait for it to '
                                              'finish.')


def test_the_progress_of_work_from_a_terminal_beside_a_training_of_the_panel(root):  # noqa: F811
    panel = Panel(root)
    line = 'signnet epoch {}/40 train_loss 2.1000 val_recall 0.3000 val_false_per_frame 4.80 80s\n'
    run = '2026-11-10_1200-cutter'
    theirs, mine = root / 'runs' / 'k1' / 'signnet.log', root / 'runs' / run / 'signnet.log'
    for path, epoch in ((theirs, 4), (mine, 9)):
        path.parent.mkdir(parents=True)
        path.write_text(line.format(epoch))
    # The log of the panel's run is the newest: still not the terminal's.
    os.utime(theirs, (time.time() - 60, time.time() - 60))
    job = '20261110-120000-train_cutter'
    payload = {'jobs': [{'id': job, 'action': 'train_cutter', 'state': 'running', 'run': run}],
               'activities': [{'group': 'train', 'label': 'cutter training', 'pid': 1, 'job': job, 'gpus': [0]},
                              {'group': 'train', 'label': 'cutter training', 'pid': 2, 'job': None, 'gpus': [1]}]}
    panel._terminal_training(payload)
    own, terminal = payload['activities']
    # The panel's own activity has its job's progress, not one read here.
    assert 'progress' not in own
    assert terminal['progress']['detail'] == 'SignNet · epoch 4 of 40'


def test_a_job_with_its_own_progress_file_never_reads_the_shared_one(root, monkeypatch):  # noqa: F811
    panel = Panel(root)
    seen = []
    monkeypatch.setattr(panel_module, 'with_step_progress',
                        lambda result, action, report, setup, download: seen.append(report) or result)
    panel.logs.mkdir(parents=True, exist_ok=True)
    started = datetime(2026, 11, 10, 12, 0).astimezone()
    job = '20261110-120000-train_cutter'
    (panel.logs / (job + '.json')).write_text(json.dumps({'gpus': [0], 'own_progress': True}))
    # Its own file not written yet; the shared one holds the step of another job, or of a command from a terminal.
    (root / 'logs' / 'progress.json').write_text(
        json.dumps({'task': 'reader', 'done': 1, 'total': 3, 'at': started.timestamp() + 5}))
    panel._with_step_progress({}, {'id': job, 'action': 'train_cutter',
                                   'started_at': started.isoformat(timespec='seconds')})
    assert seen == [None]


def test_a_preparation_needs_no_free_card_and_holds_the_one_it_takes(panel, root, checkout, monkeypatch):  # noqa: F811
    (checkout / 'helpers' / '1_prepare.sh').write_text(GPU_SCRIPT)
    # Both cards in use, GPU 1 with the most free memory: the preparation shares it.
    apps = '00000000:81:00.0, 1111, llama-server, 9000\n00000000:C1:00.0, 2222, ffmpeg, 3000\n'
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(9000, 3000), apps))
    job = panel.start('prepare', {'proposer': 'vietnam'})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    assert record(panel, job)['gpus'] == [1]
    # A preparation at work holds its card: a training takes the other one, a second training finds none.
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(5000, 0), ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    classifier_script(checkout, GPU_SCRIPT)
    cutter_in_use(root, panel)
    preparation = panel.start('prepare_again', {'proposer': 'vietnam'})
    assert said(panel, preparation, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    # A second training shares the cutter's card, never the preparation's; without the room it waits.
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(29000, 0), ''))
    assert refused(panel, 'train_classifier') == (
        409, 'busy', 'Both GPUs are busy: a cutter training on GPU 0, a preparation on GPU 1: wait for it to finish.')
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(5000, 0), ''))
    classifier = panel.start('train_classifier')
    assert said(panel, classifier, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'


def test_a_real_road_labeler_run_takes_the_free_card(panel, root, checkout, monkeypatch):  # noqa: F811
    from test_panel_road_labeler import RUN, labeler_run, labeler_script, scanned
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    # It ends at once, while the cutter holds GPU 0.
    labeler_script(checkout).write_text(GPU_SCRIPT.replace('sleep', ': sleep'))
    scanned(root)
    labeler_run(root)
    cutter = panel.start('train_cutter')
    assert said(panel, cutter, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    job = panel.start('road_labeler_predict', {'labeler': RUN})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    assert record(panel, job) == {'gpus': [1], 'own_progress': True}


def test_the_driver_files_and_the_open_files_of_a_process_as_ct_131_writes_them(tmp_path):
    gpus = tmp_path / 'gpus'
    # As /proc/driver/nvidia/gpus/<bus>/information on CT 131: the minors are crossed (index 0 is minor 1).
    for bus, minor in (('0000:81:00.0', 1), ('0000:c1:00.0', 0)):
        (gpus / bus).mkdir(parents=True)
        (gpus / bus / 'information').write_text(
            'Model: \t\t Tesla V100-PCIE-32GB\nIRQ:   \t\t 193\nGPU UUID: \t GPU-1015f4d2\nVideo BIOS: \t 88.00.48.00.02\n'
            'Bus Type: \t PCIe\nDMA Size: \t 47 bits\nDMA Mask: \t 0x7fffffffffff\n'
            f'Bus Location: \t {bus}\nDevice Minor: \t {minor}\nGPU Excluded:\t No\n')
    assert read_minor_buses(str(gpus)) == {1: '81:00.0', 0: 'c1:00.0'}
    assert read_minor_buses(str(tmp_path / 'no-driver')) == {}
    proc = tmp_path / 'proc'
    fd = proc / '4242' / 'fd'
    fd.mkdir(parents=True)
    # The files every CUDA program has open beside its card's: the control and the memory devices, not cards.
    for number, target in enumerate(('/dev/null', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia1', 'socket:[7]')):
        (fd / str(number)).symlink_to(target)
    assert read_minors(4242, str(proc)) == {1}
    (fd / '5').symlink_to('/dev/nvidia0')
    assert read_minors(4242, str(proc)) == {0, 1}
    assert read_minors(4343, str(proc)) == set()


def test_the_actions_that_take_a_card_of_their_own():
    assert set(panel_module.EXCLUSIVE_GPU) == {
        'train', 'train_continue', 'train_resume', 'train_classifier', 'train_cutter', 'train_decider', 'train_brain',
        'regression', 'measure_proposer', 'setup_vlm', 'road_labeler_train', 'road_labeler_predict'}
    # The replay of a drive (Models › Drive) shares the card with the most free memory: about 1 GB of a V100.
    assert panel_module.SHARED_GPU == {'simulate': 2048}
    # Nor among those with a dry run: it has no run on the CPU.
    assert set(panel_module.DRY_ON_CPU) == {'road_labeler_train', 'road_labeler_predict'}


@pytest.mark.parametrize('action, options', [
    ('train', None), ('train_continue', None), ('train_resume', {'run': '2026-11-10_1200-run'}), ('train_decider', None),
    ('regression', None), ('simulate', None), ('measure_proposer', None), ('setup_vlm', None),
])
def test_each_gpu_action_takes_a_card(panel, root, checkout, monkeypatch, action, options):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    # The regression's drives and models are its own tests' (test_regression*.py), the replay's drive and networks
    # too (test_panel_simulation.py).
    monkeypatch.setattr(panel, '_regression_choices', lambda options: {})
    monkeypatch.setattr(panel, '_simulate_arguments', lambda options: [])
    script = checkout / panel_module.ACTIONS[action]['command'][0]
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(GPU_SCRIPT)
    script.chmod(0o755)
    (root / 'runs' / '2026-11-10_1200-run').mkdir(parents=True)
    job = panel.start(action, options)
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    assert record(panel, job)['gpus'] == [0]


def test_fit_box_runs_on_the_cpu_and_takes_no_card(panel, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    script = checkout / panel_module.BOX_FIT_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(GPU_SCRIPT)
    script.chmod(0o755)
    job = panel.start('box_fit_train', {'refiner': 'off'})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=none cuda=none order=none'
    assert record(panel, job) == {'gpus': [], 'own_progress': True}


def test_qwen3_vl_goes_on_the_card_with_the_most_free_memory(panel, checkout, monkeypatch):  # noqa: F811
    import nnslr_tools.preannotate as preannotate
    monkeypatch.setattr(preannotate, 'vlm_missing', lambda root: None)
    for name in ('scripts/setup_vlm.sh', 'helpers/1_prepare.sh'):
        script = checkout / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(GPU_SCRIPT)
        script.chmod(0o755)
    # llama.cpp of another container on GPU 0 (12.5 GB free there, too little for Qwen3-VL); GPU 1 empty.
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(cards(20000, 0), '00000000:81:00.0, 4242, llama-server, '
                                                          '20000\n'))
    assert reasons(panel)['setup_vlm'] == {'label': 'Install Qwen3-VL', 'allowed': True, 'reason': ''}
    assert panel.status()['proposers']['qwen_gpu'] is None
    install = panel.start('setup_vlm')
    assert finished(panel, install['id'])['state'] == 'ok'
    assert said(panel, install, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'
    job = panel.start('prepare', {'proposer': 'qwen'})
    assert finished(panel, job['id'])['state'] == 'ok'
    assert said(panel, job, 'gpu=') == 'gpu=1 cuda=1 order=PCI_BUS_ID'


def test_the_gpu_lock_is_never_opened(tmp_path, monkeypatch):
    """openpilot.distill's lock is only looked up in /proc/locks: opening it could take or break its flock."""
    lock = tmp_path / 'v100-gpu.lock'
    lock.touch()
    paths = []
    real_open, real_os_open = io.open, os.open

    def tracked(file, *arguments, **keywords):
        paths.append(file if isinstance(file, int) else os.fspath(file))
        return real_open(file, *arguments, **keywords)

    def tracked_os(path, *arguments, **keywords):
        paths.append(os.fspath(path))
        return real_os_open(path, *arguments, **keywords)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with monkeypatch.context() as patch:
            patch.setattr(builtins, 'open', tracked)
            patch.setattr(io, 'open', tracked)
            patch.setattr(os, 'open', tracked_os)
            holders = read_lock_holders(str(lock))
    assert holders == [os.getpid()]
    assert '/proc/locks' in paths and str(lock) not in paths


def test_the_cards_found_through_the_open_files_are_kept_a_while(root, monkeypatch):  # noqa: F811
    panel = Panel(root)
    monkeypatch.setattr(panel_module, '_card_indexes', lambda: {'81:00.0': 0, 'c1:00.0': 1})
    # Process 1 on GPU 0; process 2 with no card open yet; process 3 with every card open (only checking them).
    tree, calls = {1: [0], 2: None, 3: [0, 1]}, []
    monkeypatch.setattr(panel_module, '_tree_cards', lambda pids: calls.append(pids[0]) or tree[pids[0]])
    for pid, seconds in ((1, panel_module.SCAN_FOUND_SECONDS), (2, panel_module.SCAN_UNKNOWN_SECONDS),
                         (3, panel_module.SCAN_UNKNOWN_SECONDS)):
        key = ('session', 'cutter training', pid)
        assert panel._scan_cards(key, [pid]) == tree[pid]
        assert panel._scan_cards(key, [pid]) == tree[pid]
        assert calls.count(pid) == 1
        at, found, kept = panel._scanned[key]
        assert kept == seconds
        # Read again once its time is over.
        panel._scanned[key] = (at - seconds - 1, found, kept)
        assert panel._scan_cards(key, [pid]) == tree[pid]
        assert calls.count(pid) == 2
    assert (panel_module.SCAN_FOUND_SECONDS, panel_module.SCAN_UNKNOWN_SECONDS) == (600, 10)


def test_two_starts_at_the_same_moment_take_two_cards(panel, root, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    classifier_script(checkout, GPU_SCRIPT)
    cutter_in_use(root, panel)
    barrier = threading.Barrier(2)
    started, failed = {}, []

    def start(action):
        barrier.wait()
        try:
            started[action] = panel.start(action)
        except PanelError as error:
            failed.append(error.message)
    threads = [threading.Thread(target=start, args=(action,)) for action in ('train_cutter', 'train_classifier')]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert failed == []
    assert sorted(record(panel, job)['gpus'] for job in started.values()) == [[0], [1]]


def test_a_running_job_keeps_its_progress_file_past_the_history(panel, root, checkout, monkeypatch):  # noqa: F811
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    signnet_script(checkout, GPU_SCRIPT)
    cutter = panel.start('train_cutter')
    said(panel, cutter, 'gpu=')
    # Twelve jobs and one more started after it, while it still runs.
    newer = [panel.logs / f'20991231-1200{second:02d}-dataset.log' for second in range(panel_module.HISTORY + 1)]
    try:
        for path in newer:
            path.write_text('== helpers/3_train.sh --dataset-only\n== finished with code 0\n')
        logs = root / 'logs'
        for name in (cutter['id'], '20200101-000000-train_cutter'):
            (logs / f'progress-{name}.json').write_text('{}\n')
        panel._prune_progress()
        assert sorted(path.name for path in logs.glob('progress-*.json')) == [f'progress-{cutter["id"]}.json']
    finally:
        # Out of the history the fixture would not see it to stop it.
        for path in newer:
            path.unlink(missing_ok=True)


def test_a_dry_run_follows_the_pairs_of_the_train_group(panel, root, checkout, monkeypatch):  # noqa: F811
    from test_panel_road_labeler import RUN, labeler_run, labeler_script, scanned
    monkeypatch.setattr(panel_module, '_nvidia', fake_nvidia(FREE, ''))
    monkeypatch.setenv('FAKE_SLEEP', '30')
    labeler_script(checkout).write_text(GPU_SCRIPT)
    scanned(root)
    labeler_run(root)
    training = panel.start('road_labeler_train')
    assert said(panel, training, 'gpu=') == 'gpu=0 cuda=0 order=PCI_BUS_ID'
    # On the CPU, with GPU 1 free, but it writes the marks the training reads.
    assert refused(panel, 'road_labeler_predict', {'labeler': RUN, 'dry': 'on'}) == (
        409, 'busy', 'A road labeler training is running: wait for it to finish.')


def test_every_label_of_the_train_group_has_its_kind():
    assert {label for group, label, _ in panel_module.ACTIVITIES if group == 'train'} == set(panel_module.TERMINAL_KINDS)


@pytest.mark.parametrize('label, kind', [
    ('cutter training', 'train_cutter'), ('brain training', 'train_brain'), ('classifier training', 'train_classifier'),
    ('regression', 'regression'), ('road labeler', 'road_labeler_train'),
    ('drive replay', 'simulate'), ('dataset and training', 'train'), ('GPU training', 'train'),
])
def test_work_from_a_terminal_follows_the_pairs_of_its_kind(root, label, kind):  # noqa: F811
    panel = Panel(root)
    activity = [{'group': 'train', 'label': label, 'pid': 1, 'job': None, 'gpus': [0]}]
    name = panel_module.ACTIVITY_NAMES[label]
    for action in (key for key, item in panel_module.ACTIONS.items() if item['group'] == 'train'):
        expected = name[0].upper() + name[1:] + ' is running' if panel_module._pair_apart(action, kind) else ''
        assert (action, panel._train_clash(action, activity)) == (action, expected)
