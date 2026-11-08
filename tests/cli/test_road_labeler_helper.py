"""helpers/road_labeler.sh refuses a real run on a busy GPU before it starts: one line, exit code 3, nnslr never called.
The real helper runs in a temporary checkout with a stand-in .venv-gpu (nnslr and python print what they were given) and
a stand-in nvidia-smi on PATH that answers as the GPU it is told to be. Nothing is trained."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / 'helpers' / 'road_labeler.sh'

# The stand-in nvidia-smi: FAKE_GPU_MEMORY "used, total" and FAKE_GPU_APPS pids, one per line; every call noted.
FAKE_SMI = '''#!/usr/bin/env bash
echo "$*" >> "$FAKE_SMI_CALLS"
case "$*" in
    *--query-gpu=*) echo "$FAKE_GPU_MEMORY" ;;
    *--query-compute-apps=*) printf '%s' "$FAKE_GPU_APPS" ;;
esac
'''


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'helpers').mkdir(parents=True)
    shutil.copy(HELPER, repo / 'helpers' / 'road_labeler.sh')
    venv = repo / '.venv-gpu' / 'bin'
    venv.mkdir(parents=True)
    (venv / 'nnslr').write_text('#!/usr/bin/env bash\necho "nnslr $*"\n')
    (venv / 'python').write_text('#!/usr/bin/env bash\necho 2\n')
    for name in ('nnslr', 'python'):
        (venv / name).chmod(0o755)
    tools = tmp_path / 'bin'
    tools.mkdir()
    (tools / 'nvidia-smi').write_text(FAKE_SMI)
    (tools / 'nvidia-smi').chmod(0o755)
    return repo


def run(repo, *arguments, memory='300, 32768', apps='', **env):
    calls = repo.parent / 'smi-calls'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('NNSLR_') and key != 'CUDA_VISIBLE_DEVICES'}
    environment.update(PATH=f'{repo.parent / "bin"}:{environment["PATH"]}', NNSLR_DATA_ROOT=str(repo / 'data'),
                       FAKE_SMI_CALLS=str(calls), FAKE_GPU_MEMORY=memory, FAKE_GPU_APPS=apps, **env)
    done = subprocess.run(['bash', str(repo / 'helpers' / 'road_labeler.sh'), *arguments], env=environment,
                          capture_output=True, text=True, timeout=30)
    return done, calls.read_text() if calls.exists() else ''


BUSY = 'helpers/road_labeler.sh: the GPU 0 is busy ({}): wait for it to finish, or --dry-run on the CPU\n'


@pytest.mark.parametrize('arguments', [('train',), ('predict', '--run-dir', 'data/runs/x', '--all')])
def test_a_real_run_refuses_a_gpu_with_another_process(repo, arguments):
    done, _ = run(repo, *arguments, memory='31726, 32768', apps='4242\n4343\n')
    assert done.returncode == 3
    assert done.stderr == BUSY.format('31726 of 32768 MiB used, 2 other compute processes')
    assert 'nnslr' not in done.stdout


def test_a_real_run_refuses_a_gpu_whose_memory_is_taken(repo):
    done, _ = run(repo, 'train', memory='4000, 32768')
    assert done.returncode == 3
    assert done.stderr == BUSY.format('4000 of 32768 MiB used, 0 other compute processes')
    assert 'nnslr' not in done.stdout


def test_a_real_run_on_a_free_gpu_starts(repo):
    done, calls = run(repo, 'train', '--epochs', '2')
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert 'GPU 0 free: 300 of 32768 MiB used, no other compute process' in lines
    assert 'drives:       2' in lines
    assert lines[-1].startswith(f'nnslr road-labeler-train --data-root {repo / "data"} --run-dir {repo / "data/runs"}/')
    assert lines[-1].endswith('-road-labeler --epochs 2')
    assert '--query-gpu=memory.used,memory.total' in calls and '--query-compute-apps=pid' in calls


def test_a_dry_run_never_looks_at_the_gpu(repo):
    for arguments in (('train', '--dry-run'), ('predict', '--run-dir=data/runs/x', '--all', '--dry-run')):
        done, calls = run(repo, *arguments, memory='31726, 32768', apps='4242\n')
        assert done.returncode == 0, done.stderr
        assert 'A dry run: on the CPU, the GPU is not checked.' in done.stdout
        assert calls == ''
    assert done.stdout.splitlines()[0] == 'run:          x'


def test_a_busy_gpu_shared_on_request(repo):
    done, calls = run(repo, 'train', memory='31726, 32768', apps='4242\n', NNSLR_ALLOW_BUSY_GPU='1')
    assert done.returncode == 0, done.stderr
    assert calls == ''


def test_no_gpu_to_answer_is_refused(repo):
    done, _ = run(repo, 'train', memory='No devices were found')
    assert (done.returncode, done.stderr) == (
        3, 'helpers/road_labeler.sh: the GPU 0 does not answer nvidia-smi (a --dry-run runs on the CPU)\n')
