"""scripts/run_brain.sh ("Train the brain") with a stand-in GPU environment: its steps in the order the panel reads them,
the three seeds each in a run of its own, the choices of Train › Brain reaching every training, and a NAME run again
finishing what is left."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The stand-in nnslr: says what it was given, writes the drives of the dry run (NNSLR_FAKE_ROUTES, brain_train.ROUTES)
# and the result of a seed.
FAKE_NNSLR = '''#!/bin/sh
if [ "$1" = borrow-cards ]; then
  # The cards of step 3 (NNSLR_FAKE_CARDS; "fail": no answer), as the panel's borrow_cards chooses them.
  echo "args: $*" >&2
  case " $* " in *" --release "*) exit 0 ;; esac
  [ "${NNSLR_FAKE_CARDS:-}" = fail ] && exit 1
  while [ $# -gt 0 ]; do [ "$1" = --own ] && own=$2; [ "$1" = --cards ] && given=$2; shift; done
  echo "${given:-${NNSLR_FAKE_CARDS:-$own}}"
  exit 0
fi
echo "args: $*"
if [ "$1" = train-brain ]; then
  dry=0
  case " $* " in *" --dry-run "*) dry=1 ;; esac
  while [ $# -gt 0 ]; do [ "$1" = --run-dir ] && dir=$2; shift; done
  mkdir -p "$dir"; echo "${NNSLR_FAKE_ROUTES:-[\"00000001--aaaaaaaaaa\"]}" > "$dir/routes.json"
  [ "$dry" = 1 ] && exit 0
  echo '{}' > "$dir/result.json"
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
    # No second reading of the signs: it made the answers of the next limit, which no brain is taught (12 Nov 2026).
    assert 'brain-signs' not in log
    for seed in range(3):
        assert (tmp_path / 'data' / 'runs' / f'n1-s{seed}' / 'result.json').is_file()
    assert log.rstrip().endswith('Brain complete: choose a seed in Models › Brain, then Publish.')


def test_the_choices_reach_the_check_and_every_seed(tmp_path):
    process = run_script(tmp_path, env={'NNSLR_BRAIN_WORKS': '0', 'NNSLR_BRAIN_ROAD': '0', 'NNSLR_BRAIN_DRIVER': '0',
                                        'NNSLR_BRAIN_POOL': 'mean', 'NNSLR_BRAIN_EPOCHS': '7', 'NNSLR_BRAIN_TYPES': '0'})
    assert process.returncode == 0, process.stderr + process.stdout
    lines = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text().splitlines()
    trainings = [line for line in lines if line.startswith('args: train-brain')]
    assert len(trainings) == 4
    assert all('--epochs 7 --pool mean --no-works --no-road --no-road-types' in line and '--driver' not in line
               for line in trainings)
    # The labels' line says what the training takes: without the road types, every second with a speed.
    assert [line for line in lines if line.startswith('args: brain-targets')][0].endswith('--no-road-types')


def test_an_earlier_brain_to_go_on_from_reaches_the_check_and_every_seed(tmp_path):
    start = tmp_path / 'data' / 'runs' / 'b1-s1' / 'brain-last.pt'
    start.parent.mkdir(parents=True)
    start.write_text('weights')
    process = run_script(tmp_path, env={'NNSLR_BRAIN_INIT_FROM': str(start)})
    assert process.returncode == 0, process.stderr + process.stdout
    lines = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text().splitlines()
    trainings = [line for line in lines if line.startswith('args: train-brain')]
    assert len(trainings) == 4 and all(f'--init-from {start}' in line for line in trainings)
    # Without it, ImageNet and random weights, as before: no flag.
    assert '--init-from' not in (run_script(tmp_path, name='n2').stdout + (tmp_path / 'data/runs/n2/brain.log').read_text())
    # A start that is not a file stops before anything runs.
    process = run_script(tmp_path, name='n3', env={'NNSLR_BRAIN_INIT_FROM': str(tmp_path / 'gone.pt')})
    assert process.returncode == 2 and 'NNSLR_BRAIN_INIT_FROM is not a file' in process.stderr
    assert not (tmp_path / 'data' / 'runs' / 'n3').exists()


def test_the_usual_training_keeps_the_roadworks_and_the_kinds_of_road_and_leaves_the_driver_out(tmp_path):
    assert run_script(tmp_path).returncode == 0
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text()
    # Without the driver's inputs, the default of train-brain: no flag says it.
    assert '--epochs 40 --pool grid' in log and '--no-works' not in log and '--no-road' not in log and '--no-driver' not in log
    # The road types are on: only the seconds with one go into the training.
    assert '--no-road-types' not in log
    assert '--driver' not in log


def sign_calls(tmp_path: Path, name: str = 'n1') -> tuple[list[str], list[str]]:
    """The lines of the cards of step 3 and the reading's call."""
    lines = (tmp_path / 'data' / 'runs' / name / 'brain.log').read_text().splitlines()
    return ([line for line in lines if line.startswith('sign cards:')],
            [line for line in lines if line.startswith('args: road-type-signs')])


def test_the_signs_are_read_on_the_cards_borrow_cards_chooses(tmp_path):
    process = run_script(tmp_path, env={'NNSLR_FAKE_CARDS': '0,1'})
    assert process.returncode == 0, process.stderr + process.stdout
    cards, calls = sign_calls(tmp_path)
    # Chosen before the reading, then given back before the seeds.
    assert cards == ['sign cards:   0,1'] and len(calls) == 1
    assert '--gpus 0,1 ' in calls[0]
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text().splitlines()
    asked = [line for line in log if line.startswith('args: borrow-cards')]
    assert len(asked) == 2 and '--own 0 ' in asked[0] and '--release' in asked[1]
    # The brain on card 1 asks for its own first.
    process = run_script(tmp_path, name='n2', env={'NNSLR_GPU_INDEX': '1'})
    assert process.returncode == 0, process.stderr + process.stdout
    assert sign_calls(tmp_path, 'n2')[0] == ['sign cards:   1']
    # No answer (no nvidia-smi, a failure): the brain's card alone, and the training goes on.
    process = run_script(tmp_path, name='n3', env={'NNSLR_FAKE_CARDS': 'fail'})
    assert process.returncode == 0, process.stderr + process.stdout
    assert sign_calls(tmp_path, 'n3')[0] == ['sign cards:   0']


def test_the_cards_can_be_given(tmp_path):
    process = run_script(tmp_path, env={'NNSLR_SIGN_GPUS': '2'})
    assert process.returncode == 0, process.stderr + process.stdout
    cards, calls = sign_calls(tmp_path)
    assert cards == ['sign cards:   2'] and all('--gpus 2 ' in line for line in calls)
    assert '--cards 2 ' in (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text()


def test_the_same_name_again_trains_only_the_seeds_left(tmp_path):
    finished = tmp_path / 'data' / 'runs' / 'n1-s0'
    finished.mkdir(parents=True)
    (finished / 'result.json').write_text('{}')
    assert run_script(tmp_path).returncode == 0
    log = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text()
    assert 'skip seed 0: already trained' in log and '--seed 0' not in log and '--seed 1' in log and '--seed 2' in log


def test_bad_settings_and_another_kind_of_run_are_refused(tmp_path):
    for env in ({'NNSLR_BRAIN_WORKS': 'x'}, {'NNSLR_BRAIN_ROAD': 'x'}, {'NNSLR_BRAIN_DRIVER': 'x'}, {'NNSLR_BRAIN_POOL': 'x'},
                {'NNSLR_BRAIN_EPOCHS': 'x'}, {'NNSLR_BRAIN_TYPES': 'x'},
                {'NNSLR_SIGN_GPUS': 'x'}, {'NNSLR_SIGN_GPUS': '0,,1'}, {'NNSLR_SIGN_GPUS': 'cuda:0'}):
        assert run_script(tmp_path, env=env).returncode == 2, env
    # The driver's inputs are gone since 10 Nov 2026: a caller of before that asks for them stops before anything runs.
    process = run_script(tmp_path, env={'NNSLR_BRAIN_DRIVER': '1'})
    assert process.returncode == 2 and 'NNSLR_BRAIN_DRIVER is gone' in process.stderr
    assert not (tmp_path / 'data' / 'runs' / 'n1').exists()
    other = tmp_path / 'data' / 'runs' / 'n2' / 'road-type'
    other.mkdir(parents=True)
    process = run_script(tmp_path, name='n2')
    assert process.returncode == 3 and 'another kind of run' in process.stderr


def test_every_step_and_every_seed_takes_the_drives_of_the_start(tmp_path):
    """The drives the dry run took (the dashcam drives at the start) are kept in routes.json and passed to every step
    after it and every seed: a drive read meanwhile waits for the next run, and the seeds stay alike."""
    first = '["00000001--aaaaaaaaaa", "00000002--bbbbbbbbbb"]'
    assert run_script(tmp_path, env={'NNSLR_FAKE_ROUTES': first}).returncode == 0
    pinned = '--route 00000001--aaaaaaaaaa --route 00000002--bbbbbbbbbb'
    lines = (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text().splitlines()
    calls = [line for line in lines if line.startswith('args: ')]
    assert calls[0].startswith('args: train-brain --dry-run') and '--route' not in calls[0]
    assert all(pinned in line for line in calls[1:]
               if not line.startswith(('args: env', 'args: borrow-cards', 'args: brain-networks'))), calls
    # road-type-signs, brain-targets and the three seeds.
    assert len([line for line in calls if pinned in line]) == 5
    assert 'drives:       2 (routes.json)' in lines
    # The same NAME again, a third drive marked meanwhile: the dry run too takes the drives of the start.
    for seed in range(3):
        (tmp_path / 'data' / 'runs' / f'n1-s{seed}' / 'result.json').unlink()
    (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').unlink()
    process = run_script(tmp_path, env={'NNSLR_FAKE_ROUTES': first[:-1] + ', "00000003--cccccccccc"]'})
    assert process.returncode == 0, process.stderr + process.stdout
    calls = [line for line in (tmp_path / 'data' / 'runs' / 'n1' / 'brain.log').read_text().splitlines()
             if line.startswith('args: ') and not line.startswith(('args: env', 'args: borrow-cards', 'args: brain-networks'))]
    assert len(calls) == 6 and all(pinned in line for line in calls)
    assert not any('00000003--cccccccccc' in line for line in calls)


def test_a_dry_run_that_does_not_say_its_drives_stops_the_run(tmp_path):
    process = run_script(tmp_path, env={'NNSLR_FAKE_ROUTES': '[]'})
    assert process.returncode == 3 and 'error: no drive in' in process.stdout + process.stderr
    # Nothing is kept: the same NAME again takes the drives marked then.
    assert not (tmp_path / 'data' / 'runs' / 'n1' / 'routes.json').exists()


def test_the_networks_that_read_the_signs_are_chosen_once_and_reach_every_reader(tmp_path):
    """Step 3 writes the run's networks of the signs (nnslr brain-networks: those of Train › Brain, else those in use, with
    the applicability network), and the reading as the car, the labels and every seed take that file."""
    chosen = {'NNSLR_BRAIN_CUTTER': 'cutter:c2/signnet', 'NNSLR_BRAIN_DECIDER': 'decider:d2/experiment-b/crop_geom'}
    process = run_script(tmp_path, env=chosen)
    assert process.returncode == 0, process.stderr + process.stdout
    run = tmp_path / 'data' / 'runs' / 'n1'
    calls = [line for line in (run / 'brain.log').read_text().splitlines() if line.startswith('args: ')]
    networks = [line for line in calls if line.startswith('args: brain-networks')]
    assert networks == [f'args: brain-networks --run-dir {run} --data-root {tmp_path / "data"} --cutter cutter:c2/signnet '
                        '--decider decider:d2/experiment-b/crop_geom']
    first = calls.index(networks[0])
    readers = [line for line in calls[first + 1:] if line.startswith(('args: road-type-signs', 'args: brain-targets',
                                                                      'args: train-brain'))]
    assert len(readers) == 5 and all(f'--sign-networks {run}/networks.json' in line for line in readers), readers
    assert '--sign-networks' not in calls[0]
    # Not an id of Models: refused before anything runs.
    process = run_script(tmp_path, name='n3', env={'NNSLR_BRAIN_CLASSIFIER': '../x; rm -rf /'})
    assert process.returncode != 0 and 'must be an id of the Models tab' in process.stderr
    assert not (tmp_path / 'data' / 'runs' / 'n3' / 'brain.log').exists() or \
        'brain-networks' not in (tmp_path / 'data' / 'runs' / 'n3' / 'brain.log').read_text()
