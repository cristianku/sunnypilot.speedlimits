"""helpers/fetch_comma_segments.sh in automatic mode, against a fake comma (ssh and rsync run on local folders): in
speedlimits every segment of every drive comes in full, the first two, the last two and reverse gear too, whether the
drive is new to data/ or not."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NEW = "000001b0--0a1b2c3d4e"
OLD = "000001af--5f6e7d8c9b"
FILES = ("rlog.zst", "qlog.zst", "fcamera.hevc", "ecamera.hevc")

# ssh [-o X]... host command: the command runs here, the comma's folder is the fake one.
FAKE_SSH = r"""#!/bin/bash
while [ "$1" = "-o" ]; do shift 2; done
shift
bash -c "${*//\/data\/media\/0\/realdata/$FAKE_COMMA}"
"""
# rsync [options] host:source destination
FAKE_RSYNC = r"""#!/bin/bash
args=("$@")
source="${args[-2]#*:}"
cp "${source//\/data\/media\/0\/realdata/$FAKE_COMMA}" "${args[-1]}"
"""
# helpers/comma_fetch_log.py: every drive is long enough unless SHORT is in the environment, a log with "reverse" in it
# is reverse gear.
FAKE_LOG_READER = r"""#!/bin/bash
shift
case "$1" in
  duration) if [ -n "${SHORT:-}" ]; then echo 200; else echo 600; fi ;;
  reverse) if grep -q reverse "$2"; then echo reverse; else echo forward; fi ;;
esac
"""


def script(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def comma(tmp_path):
    repo = tmp_path / "repo"
    (repo / "helpers").mkdir(parents=True)
    (repo / "src/nnslr_tools").mkdir(parents=True)
    shutil.copy(REPO / "helpers/fetch_comma_segments.sh", repo / "helpers")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script(bin_dir / "ssh", FAKE_SSH)
    script(bin_dir / "rsync", FAKE_RSYNC)
    script(tmp_path / "log_reader", FAKE_LOG_READER)
    device = tmp_path / "device"
    for route in (NEW, OLD):
        for index in range(8):
            folder = device / f"{route}--{index}"
            folder.mkdir(parents=True)
            for name in FILES:
                (folder / name).write_text("reverse" if name == "rlog.zst" and index == 4 else name)
    data = tmp_path / "data"
    (data / "raw/routes" / OLD / "3").mkdir(parents=True)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_COMMA": str(device),
           "COMMA_HOST": "comma", "COMMA_VIDEOS_DIR": str(tmp_path / "videos"), "NNSLR_DATA_ROOT": str(data),
           "COMMA_FETCH_PYTHON": str(tmp_path / "log_reader")}
    return repo, data, tmp_path / "videos", env


def fetched(data: Path, route: str) -> list[int]:
    """The segments with their camera linked in data/."""
    folder = data / "raw/routes" / route
    return sorted(int(segment.name) for segment in folder.iterdir() if (segment / "fcamera.hevc").is_symlink())


def fetch(repo: Path, env: dict) -> subprocess.CompletedProcess:
    process = subprocess.run(["bash", str(repo / "helpers/fetch_comma_segments.sh")], env=env, text=True,
                             capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    return process


def test_every_segment_of_every_drive_comes_in_full(comma):
    repo, data, videos, env = comma
    process = fetch(repo, env)
    # The first two, reverse gear (4) and the last two too, of a new drive and of one already in data/.
    assert fetched(data, NEW) == list(range(8))
    assert fetched(data, OLD) == list(range(8))
    for segment in (data / "raw/routes" / NEW).iterdir():
        assert sorted(path.name for path in segment.iterdir()) == ["ecamera.hevc", "fcamera.hevc", "rlog.zst"]
    for segment in videos.iterdir():
        assert sorted(path.name for path in segment.iterdir()) == ["ecamera.hevc", "fcamera.hevc", "rlog.zst"]
    # One line per drive for the panel's count (job_progress.ON_COMMA), with every segment to download.
    assert f"{NEW} on the comma: 0 1 2 3 4 5 6 7. Skip:. Download: 0 1 2 3 4 5 6 7" in process.stdout


def test_a_drive_shorter_than_300_s_is_still_left_out(comma):
    repo, data, _videos, env = comma
    fetch(repo, {**env, "SHORT": "1"})
    assert not (data / "raw/routes" / NEW).exists()
    assert fetched(data, OLD) == []


def test_a_second_run_reads_no_rlog_and_copies_nothing_again(comma):
    repo, data, videos, env = comma
    calls = videos.parent / "calls"
    script(videos.parent / "log_reader", f'#!/bin/bash\necho "$1 $2" >> {calls}\n' + FAKE_LOG_READER.split("\n", 1)[1])
    rsync_log = videos.parent / "rsync_calls"
    script(videos.parent / "bin/rsync", f'#!/bin/bash\necho "$@" >> {rsync_log}\n' + FAKE_RSYNC.split("\n", 1)[1])
    for _ in range(2):
        calls.unlink(missing_ok=True)
        rsync_log.unlink(missing_ok=True)
        process = fetch(repo, env)
    # No rlog is read for reverse gear (only the drive's length, from its qlogs), the files already here are not
    # asked again.
    assert "reverse" not in calls.read_text()
    assert ".hevc" not in rsync_log.read_text() and "rlog" not in rsync_log.read_text()
    assert "already here" in process.stdout
    assert not (videos / f"{NEW}--4/gear").exists()


def test_the_distill_layout_still_leaves_out_the_start_the_end_and_reverse_gear(comma):
    repo, _data, _videos, env = comma
    shutil.rmtree(repo / "src")
    fetch(repo, env)
    linked = sorted(folder.name for folder in (repo / "data").iterdir() if (folder / "fcamera.hevc").is_symlink())
    hex_new, hex_old = NEW.replace("-", ""), OLD.replace("-", "")
    assert linked == sorted(f"{route}{segment:02d}" for route in (hex_new, hex_old) for segment in (2, 3, 5))
