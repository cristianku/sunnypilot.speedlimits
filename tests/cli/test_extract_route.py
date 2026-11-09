# [route-extract] - START
"""Route orchestration through the real CLI and tiny synthetic videos."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from nnslr_tools import cli

ROUTE = "000001a3--c20ba54385"


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe needed")
    root = tmp_path / "data root"
    for index in (10, 0, 2):
        segment = root / "raw/routes" / ROUTE / str(index)
        segment.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x48:r=20:d=0.1",
                        "-c:v", "libx265", "-x265-params", "pools=1:frame-threads=1:log-level=error",
                        str(segment / "fcamera.hevc")], check=True, capture_output=True)
        (segment / "ecamera.hevc").write_text("wrong camera")
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(root))
    return root


def run(capsys, *args):
    try:
        code = cli.main(["extract-frames", *args])
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_route_auto_discovers_segments_and_writes_frames_and_manifests(dataset, capsys):
    code, out, err = run(capsys, "--route", ROUTE)
    assert code == 0, err
    report = json.loads(out)
    assert [row["segment_index"] for row in report["segments"]] == [0, 2, 10]
    assert report["frame_count"] == 3
    for index in (0, 2, 10):
        folder = dataset / "derived/frames" / ROUTE / str(index)
        assert (folder / "frame_00000000.png").read_bytes().startswith(b"\x89PNG")
        row = json.loads((folder / "frames.jsonl").read_text())
        assert row["route_id"] == ROUTE
        assert row["segment_index"] == index
        assert row["requested_sample_fps"] == 1.0
        assert row["camera_stream"] == "narrow_road"
    assert report["capture_time_established"] is False


def test_route_uses_default_data_root_when_environment_is_unset(dataset, capsys, monkeypatch):
    monkeypatch.delenv("NNSLR_DATA_ROOT")
    monkeypatch.setattr(cli, "DEFAULT_DATA_ROOT", str(dataset), raising=False)
    code, out, err = run(capsys, "--route", ROUTE)
    assert code == 0, err
    assert json.loads(out)["frame_count"] == 3


def test_route_overrides_and_combined_manifest(dataset, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NNSLR_DATA_ROOT", str(tmp_path / "wrong root"))
    output = tmp_path / "custom frames"
    manifest = tmp_path / "all-frames.jsonl"
    code, out, err = run(capsys, "--route", ROUTE, "--data-root", str(dataset),
                         "--output", str(output), "--manifest", str(manifest), "--fps", "5")
    assert code == 0, err
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [row["segment_index"] for row in rows] == [0, 2, 10]
    assert all(row["requested_sample_fps"] == 5 for row in rows)
    assert (output / "10/frames.jsonl").is_file()
    assert not (dataset / "derived").exists()


def test_missing_camera_is_reported_before_processing_any_segment(dataset, capsys):
    (dataset / "raw/routes" / ROUTE / "2/fcamera.hevc").unlink()
    code, _, err = run(capsys, "--route", ROUTE)
    assert code != 0
    assert "2/fcamera.hevc" in err
    assert not (dataset / "derived").exists()


def test_existing_frames_are_preserved_before_any_new_extraction(dataset, capsys):
    folder = dataset / "derived/frames" / ROUTE / "2"
    folder.mkdir(parents=True)
    existing = folder / "frame_00000000.png"
    existing.write_text("keep")
    code, _, err = run(capsys, "--route", ROUTE)
    assert code != 0
    assert "output_not_empty" in err
    assert existing.read_text() == "keep"
    assert not (folder.parent / "0").exists()


def test_single_video_mode_still_works_and_requires_output(dataset, tmp_path, capsys):
    video = dataset / "raw/routes" / ROUTE / "0/fcamera.hevc"
    code, _, err = run(capsys, "--video", str(video))
    assert code == 2
    assert "--output" in err
    code, out, err = run(capsys, "--video", str(video), "--output", str(tmp_path / "single"))
    assert code == 0, err
    assert json.loads(out)["frame_count"] == 1


@pytest.mark.parametrize("args", [
    ["--route", "../escape"],
    ["--route", ROUTE, "--video", "another.hevc"],
])
def test_invalid_or_ambiguous_input_does_not_write(dataset, capsys, args):
    code, _, _ = run(capsys, *args)
    assert code != 0
    assert not (dataset / "derived").exists()

# [route-extract] - END

# [dataset-review] - START
@pytest.mark.parametrize('alignment_kind',['missing','mismatched','invalid_capture'])
def test_alignment_failure_preserves_images_and_manifest(dataset,capsys,tmp_path,alignment_kind):
    code,out,err=run(capsys,'--route',ROUTE,'--all-frames')
    assert code==0,err
    output=dataset/'derived/frames'/ROUTE
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    alignment=tmp_path/'alignment';alignment.mkdir()
    if alignment_kind!='missing':
        from nnslr_tools.manifest import sha256_of
        for index in (0,2,10):
            video=dataset/'raw/routes'/ROUTE/str(index)/'fcamera.hevc'
            log=video.with_name('rlog.zst');log.write_bytes(b'synthetic')
            header=dict(kind='alignment_report',segment_num=index,stream='narrow_road',log=str(log),
                        source_video_sha256='a'*64 if alignment_kind=='mismatched' else sha256_of(video),source_log_sha256=sha256_of(log))
            frame=dict(kind='frame',decoded_frame_index=1,segment_num=index,alignment_status='exact',capture_mono_ns=-10,capture_reference='sof')
            (alignment/f'{index}.jsonl').write_text(json.dumps(header)+'\n'+json.dumps(frame)+'\n')
    code,out,err=run(capsys,'--route',ROUTE,'--overwrite','--start','0.05','--alignment',str(alignment))
    assert code!=0
    after={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert after==before
# [dataset-review] - END
