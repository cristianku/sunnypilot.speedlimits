"""helpers/sign_tracks.sh refuses a real run on a busy GPU before it starts (one line, exit code 3, nnslr never called),
runs a dry run on the CPU without looking at the GPU, and takes every drive when none is named. The real helper runs in a
temporary checkout with a stand-in .venv-gpu (nnslr prints what it was given) and a stand-in nvidia-smi on PATH that
answers as the GPU it is told to be. Nothing is decoded."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / 'helpers' / 'sign_tracks.sh'

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
    shutil.copy(HELPER, repo / 'helpers' / 'sign_tracks.sh')
    venv = repo / '.venv-gpu' / 'bin'
    venv.mkdir(parents=True)
    (venv / 'nnslr').write_text('#!/usr/bin/env bash\necho "nnslr $* cuda=${CUDA_VISIBLE_DEVICES-unset}"\n')
    (venv / 'nnslr').chmod(0o755)
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
    done = subprocess.run(['bash', str(repo / 'helpers' / 'sign_tracks.sh'), *arguments], env=environment,
                          capture_output=True, text=True, timeout=30)
    return done, calls.read_text() if calls.exists() else ''


BUSY = 'helpers/sign_tracks.sh: the GPU 0 is busy ({}): wait for it to finish, or --dry-run on the CPU\n'


def test_the_helper_is_executable():
    assert os.access(HELPER, os.X_OK)


def test_a_real_run_refuses_a_gpu_with_another_process(repo):
    done, _ = run(repo, memory='31726, 32768', apps='4242\n4343\n')
    assert done.returncode == 3
    assert done.stderr == BUSY.format('31726 of 32768 MiB used, 2 other compute processes')
    assert 'nnslr' not in done.stdout


def test_a_real_run_refuses_a_gpu_whose_memory_is_taken(repo):
    done, _ = run(repo, '--route', '000000ca--eb394eea21', memory='4000, 32768')
    assert done.returncode == 3
    assert done.stderr == BUSY.format('4000 of 32768 MiB used, 0 other compute processes')


def test_a_real_run_on_a_free_gpu_takes_every_drive_or_those_named(repo):
    done, calls = run(repo)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[:3] == ['== 1/2 Check the GPU ==', 'GPU 0 free: 300 of 32768 MiB used, no other compute process',
                         '== 2/2 Find the signs on every frame ==']
    assert lines[-1] == f'nnslr sign-tracks --data-root {repo / "data"} --all cuda=unset'
    assert '--query-gpu=memory.used,memory.total' in calls and '--query-compute-apps=pid' in calls
    done, _ = run(repo, '--route=000000ca--eb394eea21', '--force')
    assert done.stdout.splitlines()[-1] == (
        f'nnslr sign-tracks --data-root {repo / "data"} --route=000000ca--eb394eea21 --force cuda=unset')


def test_a_dry_run_never_looks_at_the_gpu(repo):
    done, calls = run(repo, '--dry-run', memory='31726, 32768', apps='4242\n')
    assert done.returncode == 0, done.stderr
    assert 'A dry run: on the CPU, the GPU is not checked.' in done.stdout
    assert done.stdout.splitlines()[-1] == f'nnslr sign-tracks --data-root {repo / "data"} --all --dry-run cuda='
    assert calls == ''


def test_a_busy_gpu_shared_on_request(repo):
    done, calls = run(repo, memory='31726, 32768', apps='4242\n', NNSLR_ALLOW_BUSY_GPU='1')
    assert done.returncode == 0, done.stderr
    assert calls == ''


def test_no_gpu_to_answer_is_refused(repo):
    done, _ = run(repo, memory='No devices were found')
    assert (done.returncode, done.stderr) == (
        3, 'helpers/sign_tracks.sh: the GPU 0 does not answer nvidia-smi (a --dry-run runs on the CPU)\n')
