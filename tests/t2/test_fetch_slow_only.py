"""helpers/fetch_comma_segments.sh in automatic mode, against a fake comma (ssh and rsync run on local folders): a drive
new to speedlimits also brings the segments the automatic filters leave out, marked slow-only; a drive already in
data/ does not."""
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
# helpers/comma_fetch_log.py: every drive is long enough, a log with "reverse" in it is reverse gear.
FAKE_LOG_READER = r"""#!/bin/bash
shift
case "$1" in
  duration) echo 600 ;;
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


def fetched(data: Path, route: str) -> dict[int, bool]:
    """{segment: marked slow-only} of the segments with their camera linked in data/."""
    folder = data / "raw/routes" / route
    return {int(segment.name): (segment / "slow-only").is_symlink()
            for segment in folder.iterdir() if (segment / "fcamera.hevc").is_symlink()}


def test_a_new_drive_brings_its_start_end_and_reverse_gear_marked_slow_only(comma):
    repo, data, videos, env = comma
    process = subprocess.run(["bash", str(repo / "helpers/fetch_comma_segments.sh")], env=env, text=True,
                             capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    # The first two, the reverse gear (4) and the last two only for their slow pictures; 2, 3, 5 whole.
    assert fetched(data, NEW) == {0: True, 1: True, 2: False, 3: False, 4: True, 5: False, 6: True, 7: True}
    assert (videos / f"{NEW}--0/slow-only").read_text().startswith("Start or end of the drive")
    # A drive already in data/ gets only what fits, as before.
    assert fetched(data, OLD) == {2: False, 3: False, 5: False}


def test_a_fetch_stopped_halfway_finishes_the_start_and_end_of_a_new_drive(comma):
    repo, data, videos, env = comma
    # The first run stopped after segment 0's mark: the drive is in data/ but still new.
    segment = data / "raw/routes" / NEW / "0"
    segment.mkdir(parents=True)
    (videos / f"{NEW}--0").mkdir(parents=True)
    (videos / f"{NEW}--0/slow-only").write_text("Start or end of the drive")
    (segment / "slow-only").symlink_to(videos / f"{NEW}--0/slow-only")
    process = subprocess.run(["bash", str(repo / "helpers/fetch_comma_segments.sh")], env=env, text=True,
                             capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert sorted(segment for segment, slow in fetched(data, NEW).items() if slow) == [0, 1, 4, 6, 7]


def test_a_second_run_reads_no_rlog_and_copies_nothing_again(comma):
    repo, data, videos, env = comma
    calls = videos.parent / "calls"
    script(videos.parent / "log_reader", f'#!/bin/bash\necho "$2" >> {calls}\n' + FAKE_LOG_READER.split("\n", 1)[1])
    rsync_log = videos.parent / "rsync_calls"
    script(videos.parent / "bin/rsync", f'#!/bin/bash\necho "$@" >> {rsync_log}\n' + FAKE_RSYNC.split("\n", 1)[1])
    for _ in range(2):
        calls.unlink(missing_ok=True)
        rsync_log.unlink(missing_ok=True)
        process = subprocess.run(["bash", str(repo / "helpers/fetch_comma_segments.sh")], env=env, text=True,
                                 capture_output=True, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
    # The gear of every rlog is kept next to it, the files already here are not asked again.
    assert "reverse" not in calls.read_text()
    assert ".hevc" not in rsync_log.read_text()
    assert "already here" in process.stdout
    assert (videos / f"{NEW}--4/gear").read_text().startswith("reverse ")
