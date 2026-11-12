"""scripts/run_usernet.sh ("Train UserNet") with a stand-in GPU environment: its settings reach the check and the
training, the two stages, a NAME tried again keeps its old log apart, and it stops clearly, never on the CPU, when no
card answers or the card is taken. Nothing here trains: nnslr and nvidia-smi are stand-ins."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'scripts' / 'run_usernet.sh'
# The stand-in nnslr: says whether Python would buffer and what it was given, and prints an epoch line when it trains.
FAKE_NNSLR = '''#!/bin/sh
echo "nnslr $1 PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
echo "args: $*"
case " $* " in
  *" --dry-run "*) ;;
  *" train-usernet "*|"train-usernet "*) echo "usernet epoch 1/30 train_loss 2.0000 val_recall 0.400 val_false_per_frame 1.20 50s" ;;
esac
exit 0
'''
# A card that answers and runs nothing.
FREE_CARD = '#!/bin/sh\nexit 0\n'
# No card at that index: nvidia-smi -i 0 fails, as with no GPU.
NO_CARD = '#!/bin/sh\necho "No devices were found" >&2\nexit 6\n'
# A card with a training on it.
BUSY_CARD = '''#!/bin/sh
case " $* " in *" --query-compute-apps="*) echo "4242, python, 9000" ;; esac
exit 0
'''


def bdd_dir(tmp_path: Path) -> Path:
    folder = tmp_path / 'bdd100k'
    folder.mkdir(exist_ok=True)
    (folder / 'samples.json').write_text('[]')
    return folder


def run_script(tmp_path: Path, name: str = 'u1', env: dict[str, str] | None = None, card: str = FREE_CARD
               ) -> subprocess.CompletedProcess:
    tools, venv = tmp_path / 'bin', tmp_path / 'venv' / 'bin'
    for folder in (tools, venv):
        folder.mkdir(parents=True, exist_ok=True)
    for tool, text in ((tools / 'nvidia-smi', card), (venv / 'python', '#!/bin/sh\nexit 0\n'), (venv / 'nnslr', FAKE_NNSLR)):
        tool.write_text(text)
        tool.chmod(0o755)
    return subprocess.run(['bash', str(SCRIPT), name], capture_output=True, check=False, text=True, timeout=60,
                          env={'PATH': f'{tools}:/usr/bin:/bin', 'NNSLR_DATA_ROOT': str(tmp_path / 'data'),
                               'NNSLR_GPU_VENV': str(venv.parent), 'NNSLR_BDD_DIR': str(bdd_dir(tmp_path)), **(env or {})})


def _calls(log: str, dry: bool) -> list[list[str]]:
    """The arguments of each train-usernet: the checks (dry) or the trainings."""
    return [line.split()[1:] for line in log.splitlines()
            if line.startswith('args: train-usernet ') and ('--dry-run' in line.split()) == dry]


def _after(arguments: list[str], flag: str) -> str | None:
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def test_the_bdd_stage_checks_then_trains_on_the_card_unbuffered_and_logged(tmp_path):
    process = run_script(tmp_path, env={'NNSLR_GPU_INDEX': '1', 'NNSLR_USERNET_EPOCHS': '12'})
    assert process.returncode == 0, process.stderr + process.stdout
    run = tmp_path / 'data' / 'runs' / 'u1'
    log = (run / 'usernet.log').read_text()
    assert log.startswith('== scripts/run_usernet.sh u1\n')  # the line the panel reads the run from
    steps = [line.split()[1] for line in log.splitlines() if line.startswith('== ') and '/3 ' in line]
    assert steps == ['1/3', '2/3', '3/3']
    (check,), (training,) = _calls(log, dry=True), _calls(log, dry=False)
    for arguments in (check, training):
        assert _after(arguments, '--stage') == 'bdd' and _after(arguments, '--bdd-dir') == str(tmp_path / 'bdd100k')
        assert _after(arguments, '--epochs') == '12' and _after(arguments, '--run-dir') == str(run)
        assert '--init-from' not in arguments
    # The training reads the plan its check wrote, in a folder of the job's own, gone at the end.
    assert _after(check, '--plan-out') == _after(training, '--plan')
    assert not Path(_after(training, '--plan')).parent.exists()
    # On the card asked for, as cuda:0, unbuffered: the epochs reach the log as they come.
    assert 'nnslr train-usernet PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1' in log and 'usernet epoch 1/30 ' in log
    assert 'args: env --gpu-smoke --gpu-device 0' in log
    assert 'settings:     --stage bdd ' in log and 'licence:' in log
    assert 'CUDA_VISIBLE_DEVICES=""' not in SCRIPT.read_text() and 'CUDA_VISIBLE_DEVICES= ' not in SCRIPT.read_text()


def test_the_drives_stage_goes_on_from_a_usernet_without_bdd(tmp_path):
    start = tmp_path / 'start' / 'usernet-best.pt'
    start.parent.mkdir()
    start.write_bytes(b'weights')
    process = run_script(tmp_path, env={'NNSLR_USERNET_STAGE': 'drives', 'NNSLR_USERNET_INIT_FROM': str(start),
                                        'NNSLR_BDD_DIR': str(tmp_path / 'nowhere')})
    assert process.returncode == 0, process.stderr + process.stdout
    log = (tmp_path / 'data' / 'runs' / 'u1' / 'usernet.log').read_text()
    (check,), (training,) = _calls(log, dry=True), _calls(log, dry=False)
    assert _after(check, '--stage') == _after(training, '--stage') == 'drives'
    assert _after(check, '--init-from') == _after(training, '--init-from') == str(start)


def test_a_name_tried_again_keeps_its_old_log_apart(tmp_path):
    run = tmp_path / 'data' / 'runs' / 'u1'
    run.mkdir(parents=True)
    (run / 'usernet-best.pt').write_bytes(b'weights of a training stopped half way')
    (run / 'usernet.log').write_text('usernet epoch 9/30 train_loss 1.0 val_recall 0.5 val_false_per_frame 1.0 50s\n')
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (run / 'usernet.log').read_text()
    assert 'epoch 9/30' not in log and 'restart from scratch' in log
    assert (run / 'usernet.log.1').read_text().startswith('usernet epoch 9/30 ')


def test_no_card_stops_before_anything_runs(tmp_path):
    process = run_script(tmp_path, card=NO_CARD)
    assert process.returncode == 2
    assert 'never on the CPU' in process.stderr
    assert not (tmp_path / 'data' / 'runs' / 'u1').exists()


def test_a_card_in_use_is_refused(tmp_path):
    process = run_script(tmp_path, card=BUSY_CARD)
    assert process.returncode == 3
    assert 'one training a card' in process.stderr and '4242' in process.stderr
    assert not (tmp_path / 'data' / 'runs' / 'u1').exists()


def test_wrong_arguments_stop_with_two(tmp_path):
    cases = [{'NNSLR_USERNET_STAGE': 'mapillary'}, {'NNSLR_USERNET_EPOCHS': '0'}, {'NNSLR_USERNET_BATCH': 'many'},
             {'NNSLR_USERNET_LEARNING_RATE': '-1'}, {'NNSLR_USERNET_INIT_FROM': str(tmp_path / 'missing.pt')},
             {'NNSLR_BDD_DIR': str(tmp_path / 'nowhere')}, {'NNSLR_DATASET_KIND': 'silver'}]
    for env in cases:
        process = run_script(tmp_path, env=env)
        assert process.returncode == 2, (env, process.stdout, process.stderr)
        assert process.stderr.startswith('error: '), env
    assert run_script(tmp_path, name='../up').returncode == 2
    usage = subprocess.run(['bash', str(SCRIPT)], capture_output=True, text=True, timeout=60, check=False)
    assert usage.returncode == 2 and 'usage:' in usage.stderr
    assert not (tmp_path / 'data' / 'runs').exists()


def test_another_kind_of_run_or_a_finished_usernet_is_never_written_into(tmp_path):
    runs = tmp_path / 'data' / 'runs'
    (runs / 'cutter').mkdir(parents=True)
    (runs / 'cutter' / 'signnet-best.pt').write_bytes(b'a SignNet')
    (runs / 'done').mkdir()
    (runs / 'done' / 'result.json').write_text('{}')
    for name in ('cutter', 'done'):
        process = run_script(tmp_path, name=name)
        assert process.returncode == 3, process.stderr
        assert not (runs / name / 'usernet.log').exists()
