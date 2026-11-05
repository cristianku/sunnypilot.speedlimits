"""scripts/run_brain.sh ("Train the brain") with a stand-in GPU environment: its steps in the order the panel reads them,
the three seeds each in a run of its own, the choices of Train › Brain reaching every training, and a NAME run again
finishing what is left."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The stand-in nnslr: says what it was given, and writes the result of a seed.
FAKE_NNSLR = '''#!/bin/sh
echo "args: $*"
if [ "$1" = train-brain ]; then
  case " $* " in *" --dry-run "*) exit 0 ;; esac
  while [ $# -gt 0 ]; do [ "$1" = --run-dir ] && dir=$2; shift; done
  mkdir -p "$dir"; echo '{}' > "$dir/result.json"
fi
exit 0
'''


def run_script(tmp_path: Path, name: str = 'n1', env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    tools, venv = tmp_path / 'bin', tmp_path / 'venv' / 'bin'
    for folder in (tools, venv):
        folder.mkdir(parents=True, exist_ok=True)
    for tool, text in ((tools / 'nvidia-smi', '#!/bin/sh\nexit 0\n'), (venv / 'python', '#!/bin/sh\nexit 0\n'),
                       (venv / 'nnslr', FAKE_NNSLR)):
        tool.write_text(text)
        tool.chmod(0o755)
    # A copy, so that it does not find the .venv of this checkout for the signs.
    script = tmp_path / 'repo' / 'scripts' / 'run_brain.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text((REPO / 'scripts' / 'run_brain.sh').read_text())
    return subprocess.run(['bash', str(script), name], capture_output=True, check=False, text=True,
                          timeout=60, env={'PATH': f'{tools}:/usr/bin:/bin', 'NNSLR_DATA_ROOT': str(tmp_path / 'data'),
                                           'NNSLR_GPU_VENV': str(venv.parent), **(env or {})})


def test_the_steps_read_the_signs_before_the_three_seeds(tmp_path):
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text()
    steps = [line for line in log.splitlines() if line.startswith('== ') and '/6 ' in line]
    assert [line.split()[1] for line in steps] == ['1/6', '2/6', '3/6', '4/6', '5/6', '6/6']
    assert log.index('--dry-run') < log.index('road-type-signs --labelled') < log.index('brain-targets') \
        < log.index('--seed 0') < log.index('--seed 1') < log.index('--seed 2')
    for seed in range(3):
        assert (tmp_path / 'data' / 'runs' / f'n1-s{seed}' / 'result.json').is_file()
    assert log.rstrip().endswith('Brain complete: choose a seed in Models › Brain, then Publish.')


def test_the_choices_reach_the_check_and_every_seed(tmp_path):
    process = run_script(tmp_path, env={'NNSLR_BRAIN_WORKS': '0', 'NNSLR_BRAIN_DRIVER': '1', 'NNSLR_BRAIN_POOL': 'grid',
                                        'NNSLR_BRAIN_EPOCHS': '7'})
    assert process.returncode == 0, process.stderr + process.stdout
    trainings = [line for line in (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text().splitlines()
                 if line.startswith('args: train-brain')]
    assert len(trainings) == 4
    assert all('--epochs 7 --pool grid --no-works --driver' in line for line in trainings)


def test_the_usual_training_keeps_the_roadworks_and_leaves_the_driver_out(tmp_path):
    assert run_script(tmp_path).returncode == 0
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text()
    assert '--epochs 40 --pool mean' in log and '--no-works' not in log and '--driver' not in log


def test_the_same_name_again_trains_only_the_seeds_left(tmp_path):
    finished = tmp_path / 'data' / 'runs' / 'n1-s0'
    finished.mkdir(parents=True)
    (finished / 'result.json').write_text('{}')
    assert run_script(tmp_path).returncode == 0
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text()
    assert 'skip seed 0: already trained' in log and '--seed 0' not in log and '--seed 1' in log and '--seed 2' in log


def test_bad_settings_and_another_kind_of_run_are_refused(tmp_path):
    for env in ({'NNSLR_BRAIN_WORKS': 'x'}, {'NNSLR_BRAIN_DRIVER': 'x'}, {'NNSLR_BRAIN_POOL': 'x'}, {'NNSLR_BRAIN_EPOCHS': 'x'}):
        assert run_script(tmp_path, env=env).returncode == 2, env
    other = tmp_path / 'data' / 'runs' / 'n2' / 'road-type'
    other.mkdir(parents=True)
    process = run_script(tmp_path, name='n2')
    assert process.returncode == 3 and 'another kind of run' in process.stderr
