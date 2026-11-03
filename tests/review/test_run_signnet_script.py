"""scripts/run_signnet.sh ("Train the cutter") with a stand-in GPU environment: the epochs reach signnet.log as they
come (PYTHONUNBUFFERED), a NAME tried again keeps its old log apart (signnet.log.1), so epochs never mix, and the
Mapillary windows are added to train (NNSLR_CUTTER_MAPILLARY=1) or trained first, in a stage of their own
(=pretrain)."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The stand-in nnslr: says whether Python would buffer and what it was given, and prints an epoch line when it
# trains.
FAKE_NNSLR = '''#!/bin/sh
echo "nnslr $1 PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-unset}"
echo "args: $*"
case " $* " in
  *" --dry-run "*) ;;
  *" train-signnet "*|"train-signnet "*) echo "signnet epoch 1/40 train_loss 4.0000 val_recall 0.100 val_false_per_frame 4.90 80s" ;;
esac
exit 0
'''


def run_script(tmp_path: Path, name: str = 'n1', env: dict[str, str] | None = None
               ) -> subprocess.CompletedProcess:
    tools, venv = tmp_path / 'bin', tmp_path / 'venv' / 'bin'
    for folder in (tools, venv):
        folder.mkdir(parents=True, exist_ok=True)
    for tool, text in ((tools / 'nvidia-smi', '#!/bin/sh\nexit 0\n'), (venv / 'python', '#!/bin/sh\nexit 0\n'),
                       (venv / 'nnslr', FAKE_NNSLR)):
        tool.write_text(text)
        tool.chmod(0o755)
    return subprocess.run(['bash', str(REPO / 'scripts' / 'run_signnet.sh'), name], capture_output=True, check=False,
                          text=True, timeout=60, env={'PATH': f'{tools}:/usr/bin:/bin',
                                                      'NNSLR_DATA_ROOT': str(tmp_path / 'data'),
                                                      'NNSLR_GPU_VENV': str(venv.parent), **(env or {})})


def test_the_training_is_unbuffered_and_logged(tmp_path):
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'signnet.log').read_text()
    assert 'nnslr train-signnet PYTHONUNBUFFERED=1' in log and 'signnet epoch 1/40 ' in log
    assert not (tmp_path / 'data' / 'runs' / 'n1' / 'signnet.log.1').exists()


def test_a_name_tried_again_keeps_its_old_log_apart(tmp_path):
    run = tmp_path / 'data' / 'runs' / 'n1'
    run.mkdir(parents=True)
    (run / 'signnet-best.pt').write_bytes(b'weights of a training stopped half way')
    (run / 'signnet.log').write_text('signnet epoch 30/40 train_loss 1.0 val_recall 0.5 val_false_per_frame 4.0 80s\n')
    (run / 'signnet.log.1').write_text('an older try\n')
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (run / 'signnet.log').read_text()
    assert 'epoch 30/40' not in log and 'signnet epoch 1/40 ' in log
    assert 'restart from scratch' in log and f'the log of the earlier try: {run}/signnet.log.1' in log
    assert (run / 'signnet.log.1').read_text().startswith('signnet epoch 30/40 ')  # the older one replaced


# --- The Mapillary windows ------------------------------------------------------------------------------------------

def _windows(tmp_path: Path) -> Path:
    path = tmp_path / 'data' / 'external' / 'mapillary' / 'windows.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"kind": "mapillary_mtsd_windows_v1"}\n')
    return path


def _trainings(log: str) -> list[list[str]]:
    """The arguments of each train-signnet that trained (not the dry runs), in order."""
    return [line.split()[1:] for line in log.splitlines()
            if line.startswith('args: train-signnet ') and '--dry-run' not in line.split()]


def _after(arguments: list[str], flag: str) -> str | None:
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def test_the_two_stages_train_the_windows_first_then_ours_from_them(tmp_path):
    windows = _windows(tmp_path)
    start = tmp_path / 'start' / 'signnet-best.pt'
    start.parent.mkdir()
    start.write_bytes(b'weights')
    process = run_script(tmp_path, env={'NNSLR_CUTTER_MAPILLARY': 'pretrain', 'NNSLR_CUTTER_PRETRAIN_EPOCHS': '7',
                                        'NNSLR_CUTTER_INIT_FROM': str(start)})
    assert process.returncode == 0, process.stderr + process.stdout
    run = tmp_path / 'data' / 'runs' / 'n1'
    log = (run / 'signnet.log').read_text()
    assert 'pretraining:  ' in log and '-- the first stage: the Mapillary windows alone --' in log
    first, second = _trainings(log)
    assert _after(first, '--run-dir') == str(run / 'pretrain') and '--mapillary-only' in first
    assert _after(first, '--mapillary') == str(windows) and _after(first, '--epochs') == '7'
    assert _after(first, '--init-from') == str(start)  # the start of the first stage
    assert _after(second, '--run-dir') == str(run) and _after(second, '--epochs') == '40'
    assert _after(second, '--init-from') == str(run / 'pretrain' / 'signnet-best.pt')
    assert '--mapillary' not in second and '--mapillary-only' not in second and second.count('--init-from') == 1
    # Both plans are checked first; the second one's without the weights the first stage has not made yet.
    checks = [line.split() for line in log.splitlines() if line.startswith('args: train-signnet --dry-run')]
    assert len(checks) == 2 and '--mapillary-only' in checks[0] and '--init-from' not in checks[1]


def test_a_first_stage_trained_is_not_trained_again(tmp_path):
    _windows(tmp_path)
    stage = tmp_path / 'data' / 'runs' / 'n1' / 'pretrain'
    stage.mkdir(parents=True)
    (stage / 'signnet-best.pt').write_bytes(b'weights')
    (stage / 'result.json').write_text('{}')
    process = run_script(tmp_path, env={'NNSLR_CUTTER_MAPILLARY': 'pretrain'})
    assert process.returncode == 0, process.stderr + process.stdout
    log = (stage.parent / 'signnet.log').read_text()
    assert f'skip trained: {stage}/result.json' in log
    assert f'note: {stage}/settings.txt is missing' in log  # a stage of before its settings were kept
    (only,) = _trainings(log)
    assert _after(only, '--run-dir') == str(stage.parent)


def test_a_first_stage_trained_with_other_settings_is_refused(tmp_path):
    _windows(tmp_path)
    env = {'NNSLR_CUTTER_MAPILLARY': 'pretrain', 'NNSLR_CUTTER_PRETRAIN_EPOCHS': '7'}
    process = run_script(tmp_path, env=env)
    assert process.returncode == 0, process.stderr + process.stdout
    stage = tmp_path / 'data' / 'runs' / 'n1' / 'pretrain'
    record = (stage / 'settings.txt').read_text()
    assert record.startswith('--dataset-kind gold ') and '--epochs 7' in record and '--mapillary-only' in record
    (stage / 'result.json').write_text('{}')
    # The same settings: skipped, and the second stage trains from it.
    process = run_script(tmp_path, env=env)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (stage.parent / 'signnet.log').read_text()
    assert f'skip trained: {stage}/result.json' in log and 'note:' not in log and len(_trainings(log)) == 1
    # Other epochs or another share: refused before anything is touched, the log too.
    for changed in ({'NNSLR_CUTTER_PRETRAIN_EPOCHS': '9'}, {'NNSLR_CUTTER_MAPILLARY_SHARE': '50'}):
        process = run_script(tmp_path, env={**env, **changed})
        assert process.returncode == 3 and 'was trained with other settings' in process.stderr
        assert 'remove pretrain/ or use the same settings' in process.stderr
        assert (stage.parent / 'signnet.log').read_text() == log


def test_a_first_stage_stopped_half_way_starts_again(tmp_path):
    _windows(tmp_path)
    stage = tmp_path / 'data' / 'runs' / 'n1' / 'pretrain'
    stage.mkdir(parents=True)
    (stage / 'signnet-best.pt').write_bytes(b'weights of a stage stopped half way')
    process = run_script(tmp_path, env={'NNSLR_CUTTER_MAPILLARY': 'pretrain'})
    assert process.returncode == 0, process.stderr + process.stdout
    log = (stage.parent / 'signnet.log').read_text()
    assert 'restart the first stage from scratch' in log and len(_trainings(log)) == 2


def test_the_windows_added_to_ours_with_their_ratio_region_and_share(tmp_path):
    windows = _windows(tmp_path)
    settings = {'NNSLR_CUTTER_MAPILLARY': '1', 'NNSLR_CUTTER_MAPILLARY_RATIO': '0.5',
                'NNSLR_CUTTER_MAPILLARY_REGION': 'europe', 'NNSLR_CUTTER_MAPILLARY_SHARE': '40'}
    process = run_script(tmp_path, env=settings)
    assert process.returncode == 2 and 'helpers/mapillary_locations.sh' in process.stderr
    (windows.parent / 'locations.jsonl').write_text('{}\n')
    process = run_script(tmp_path, env=settings)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'signnet.log').read_text()
    (only,) = _trainings(log)
    assert _after(only, '--mapillary') == str(windows) and _after(only, '--mapillary-ratio') == '0.5'
    assert _after(only, '--mapillary-region') == 'europe' and _after(only, '--mapillary-share') == '40'
    assert '--mapillary-only' not in only and 'pretraining:' not in log
    process = run_script(tmp_path, name='n2', env={'NNSLR_CUTTER_MAPILLARY': '1'})
    (only,) = _trainings((tmp_path / 'data' / 'runs' / 'n2' / 'signnet.log').read_text())
    assert '--mapillary-ratio' not in only and '--mapillary-region' not in only and '--mapillary-share' not in only


def test_the_mapillary_settings_that_are_refused(tmp_path):
    for settings, words in (
            ({'NNSLR_CUTTER_MAPILLARY_REGION': 'europe'}, 'add NNSLR_CUTTER_MAPILLARY=1'),
            ({'NNSLR_CUTTER_MAPILLARY_SHARE': '50'}, 'add NNSLR_CUTTER_MAPILLARY=1'),
            ({'NNSLR_CUTTER_MAPILLARY_RATIO': '2'}, 'only with NNSLR_CUTTER_MAPILLARY=1'),
            ({'NNSLR_CUTTER_MAPILLARY': 'pretrain', 'NNSLR_CUTTER_MAPILLARY_RATIO': '2'},
             'only with NNSLR_CUTTER_MAPILLARY=1'),
            ({'NNSLR_CUTTER_MAPILLARY': '1', 'NNSLR_CUTTER_MAPILLARY_RATIO': 'many'}, 'must be a number'),
            ({'NNSLR_CUTTER_MAPILLARY': 'yes'}, 'must be 0, 1 or pretrain'),
            ({'NNSLR_CUTTER_MAPILLARY': '1'}, 'run helpers/mapillary_windows.sh first'),
            ({'NNSLR_CUTTER_PRETRAIN_EPOCHS': '0'}, 'PRETRAIN_EPOCHS must be a whole number')):
        process = run_script(tmp_path, env=settings)
        assert process.returncode == 2 and words in process.stderr, (settings, process.stderr)
    assert not (tmp_path / 'data' / 'runs' / 'n1').exists()


def test_a_classifier_stage_or_a_cutter_stage_without_its_mode_is_refused(tmp_path):
    run = tmp_path / 'data' / 'runs' / 'n1'
    (run / 'pretrain').mkdir(parents=True)
    # An empty pretrain/ is nobody's stage.
    assert run_script(tmp_path).returncode == 0
    (run / 'pretrain' / 'signnet-best.pt').write_bytes(b'weights')
    process = run_script(tmp_path)
    assert process.returncode == 3 and 'NNSLR_CUTTER_MAPILLARY=pretrain' in process.stderr
    other = tmp_path / 'data' / 'runs' / 'n2' / 'pretrain'
    other.mkdir(parents=True)
    (other / 'reader-best.pt').write_bytes(b'a classifier')
    process = run_script(tmp_path, name='n2')
    assert process.returncode == 3 and 'another kind of run (it has pretrain/reader-best.pt)' in process.stderr
