from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from nnslr_tools.media import extract_frames, infer_camera_stream, probe_video


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_infer_camera_stream() -> None:
    assert infer_camera_stream(Path("fcamera.hevc")) == "narrow_road"
    assert infer_camera_stream(Path("ecamera.hevc")) == "wide_road"
    assert infer_camera_stream(Path("qcamera.ts")) == "q_narrow_road"
    assert infer_camera_stream(Path("other.mp4")) is None


def test_probe_video_parses_ffprobe_json(tmp_path) -> None:
    video = tmp_path / "fcamera.hevc"
    video.write_bytes(b"x")
    ffprobe = tmp_path / "ffprobe"
    payload = {
        "streams": [{
            "codec_type": "video",
            "codec_name": "hevc",
            "codec_long_name": "HEVC",
            "width": 1928,
            "height": 1208,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "20/1",
            "avg_frame_rate": "20/1",
            "time_base": "1/1200000",
            "nb_read_frames": "1200",
        }],
        "format": {"format_name": "hevc", "duration": "60.0", "start_time": "0.0"},
    }
    _write_executable(
        ffprobe,
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-version\" ]; then echo 'ffprobe fake 1.0'; exit 0; fi\n"
        f"cat <<'EOF'\n{json.dumps(payload)}\nEOF\n",
    )
    result = probe_video(video, ffprobe=str(ffprobe))
    assert result.camera_stream == "narrow_road"
    assert result.codec == "hevc"
    assert result.width == 1928
    assert result.height == 1208
    assert result.average_fps == 20.0
    assert result.duration_s == 60.0
    assert result.frame_count == 1200


def test_extract_frames_uses_deterministic_names(raw_hevc, tmp_path) -> None:
    video = raw_hevc
    output = tmp_path / "frames"
    rows = extract_frames(video, output, fps=1.0, end_s=1)
    assert len(rows) == 1
    assert Path(rows[0].image_path).name == "frame_00000000.png"
    assert rows[0].requested_sample_fps == 1.0

# [t2-validation] - START
from nnslr_tools.media import probe_frame_timestamps
import pytest


def _frame_probe(tmp_path, frames, container="hevc"):
    video = tmp_path / "video.hevc"
    video.write_bytes(b"x")
    ffprobe = tmp_path / "ffprobe"
    payload = {"frames": frames, "format": {"format_name": container}}
    _write_executable(ffprobe, "#!/bin/sh\ncat <<'EOF'\n" + json.dumps(payload) + "\nEOF\n")
    return probe_frame_timestamps(video, ffprobe=str(ffprobe))


def test_raw_hevc_duration_clock_is_explicit_and_not_capture_or_pts(tmp_path):
    frames = _frame_probe(tmp_path, [{"duration_time": "0.05"}, {"duration_time": "0.1"}, {"duration_time": "0.05"}])
    assert [x.media_time_s for x in frames] == pytest.approx([0, .05, .15])
    assert all(x.media_time_source == "raw_hevc_frame_duration" for x in frames)
    assert all(x.pts_time_s is None and x.best_effort_timestamp_s is None for x in frames)


@pytest.mark.parametrize("durations", [[".05", None], [".05", "0"], [".05", "NaN"]])
def test_missing_or_invalid_duration_does_not_fabricate_media_clock(tmp_path, durations):
    frames = _frame_probe(tmp_path, [{"duration_time": d} for d in durations])
    assert all(x.media_time_s is None for x in frames)


def test_container_and_mixed_timestamps_are_not_reconstructed(tmp_path):
    frames = _frame_probe(tmp_path, [{"duration_time": ".05"}, {"duration_time": ".05", "pts_time": "2"}])
    assert frames[0].media_time_s is None
    assert frames[1].media_time_s == 2
    frames = _frame_probe(tmp_path, [{"duration_time": ".05"}], container="mov,mp4")
    assert frames[0].media_time_s is None
# [t2-validation] - END

# [t2-validation] - START
import shutil
import subprocess
from nnslr_tools.media import make_clip


@pytest.fixture
def raw_hevc(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for local media integration")
    video = tmp_path / "fcamera.hevc"
    p = subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x48:r=20:d=1",
        "-f", "lavfi", "-i", "color=blue:s=64x48:r=20:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx265", "-x265-params", "pools=1:frame-threads=1:log-level=error:bframes=0",
        str(video),
    ], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return video


def test_raw_hevc_timing_from_the_packets_is_the_decoded_timing(raw_hevc):
    decoded = probe_frame_timestamps(raw_hevc)
    assert probe_frame_timestamps(raw_hevc, decode=False) == [replace(x, pict_type=None) for x in decoded]
    assert decoded[0].key_frame and decoded[0].media_time_source == "raw_hevc_frame_duration"


def test_raw_hevc_clip_seeks_to_real_frames_and_default_is_usable(raw_hevc, tmp_path):
    clip = make_clip(raw_hevc, tmp_path / "clip.mp4", start_s=1.25, end_s=1.75)
    probe = probe_video(clip)
    assert probe.frame_count == 10
    pixel = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(clip), "-vf", "scale=1:1", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], capture_output=True, check=True).stdout
    assert len(pixel) == 3 and pixel[2] > 150 and pixel[0] < 80


def test_raw_hevc_frame_extraction_can_start_after_zero(raw_hevc, tmp_path):
    frames = extract_frames(raw_hevc, tmp_path / "frames", start_s=1.25, end_s=1.75)
    assert len(frames) == 10
# [t2-validation] - END

# [hevc-seek] - START
from nnslr_tools.media import MediaToolError


def test_empty_clip_is_an_error(raw_hevc, tmp_path):
    with pytest.raises(MediaToolError, match="clip_extraction_failed"):
        make_clip(raw_hevc, tmp_path / "empty.mp4", start_s=4, end_s=5)


def test_empty_extraction_is_an_error(raw_hevc, tmp_path):
    with pytest.raises(MediaToolError, match="frame_extraction_failed"):
        extract_frames(raw_hevc, tmp_path / "empty", start_s=4, end_s=5)
# [hevc-seek] - END

# [frame-provenance] - START
def test_sampled_frames_preserve_source_identity_geometry_and_hash(raw_hevc, tmp_path):
    import hashlib
    frames = extract_frames(raw_hevc, tmp_path / "sampled", fps=2, start_s=.25, end_s=1.75)
    rows = [f.to_dict() for f in frames]
    assert [r.get("decoded_frame_index") for r in rows] == [5, 15, 25]
    assert [r["output_index"] for r in rows] == [0, 1, 2]
    assert [r["media_time_s"] for r in rows] == pytest.approx([.25, .75, 1.25])
    for row in rows:
        assert (row["width"], row["height"]) == (64, 48)
        assert row["source_video_sha256"] == hashlib.sha256(raw_hevc.read_bytes()).hexdigest()
        assert row["image_sha256"] == hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        assert row["capture_mono_ns"] is None
        assert row["capture_time_provenance"] == "unknown"
        assert row["alignment_status"] == "unresolved"
        assert row["camera_stream"] == "narrow_road"


def test_duplicate_segment_directories_rejected_before_inventory(tmp_path):
    from nnslr_tools.route_io import discover_segment_dirs
    from nnslr_tools.manifest import NnslerManifestError
    route = tmp_path / "00000001--abcdef1234"
    (route / "0").mkdir(parents=True)
    (route / "00").mkdir()
    with pytest.raises(NnslerManifestError, match="duplicate_segment"):
        discover_segment_dirs(route)


def test_overwrite_does_not_include_stale_frames(raw_hevc, tmp_path):
    output = tmp_path / "frames"
    extract_frames(raw_hevc, output)
    rows = extract_frames(raw_hevc, output, fps=1, overwrite=True)
    assert len(rows) == 2
    assert len(list(output.glob("frame_*.png"))) == 2


@pytest.mark.parametrize("filename,stream", [("ecamera.hevc", "wide_road"), ("qcamera.ts", "q_narrow_road")])
def test_other_camera_streams_preserve_identity(raw_hevc, tmp_path, filename, stream):
    video = tmp_path / filename
    if filename.endswith(".ts"):
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(raw_hevc), "-c:v", "libx264", "-an", str(video)], check=True)
    else:
        shutil.copyfile(raw_hevc, video)
    frames = extract_frames(video, tmp_path / "other", fps=2)
    assert len(frames) == 4
    assert all(f.camera_stream == stream and f.width == 64 for f in frames)
    assert frames[0].decoded_frame_index == 0


def test_corrupt_video_does_not_replace_existing_images(raw_hevc, tmp_path):
    output = tmp_path / "frames"
    frames = extract_frames(raw_hevc, output, fps=1)
    original = Path(frames[0].image_path).read_bytes()
    raw_hevc.write_bytes(raw_hevc.read_bytes()[:40])
    with pytest.raises(MediaToolError):
        extract_frames(raw_hevc, output, fps=1, overwrite=True)
    assert Path(frames[0].image_path).read_bytes() == original


def test_partial_gop_and_truncated_hevc_do_not_publish_frames(raw_hevc, tmp_path):
    payload = raw_hevc.read_bytes()
    for name, data in [('partial', payload[80:]), ('truncated', payload[:len(payload)//2])]:
        video = tmp_path / (name + '.hevc')
        video.write_bytes(data)
        with pytest.raises(MediaToolError):
            extract_frames(video, tmp_path / name, fps=1)
        assert not list((tmp_path / name).glob('frame_*.png'))


def test_zero_frame_probe_rejected(tmp_path):
    video = tmp_path / "empty.hevc"
    video.write_bytes(b"")
    ffprobe = tmp_path / "ffprobe"
    _write_executable(ffprobe, '#!/bin/sh\nprintf \'{"frames":[],"format":{"format_name":"hevc"}}\'')
    # [t2-ffmpeg-isolation] - START
    # Isolate from a real ffmpeg: the zero-frame rejection fires before any decode,
    # so a stub ffmpeg is resolved (PATH may lack ffmpeg in CI) but never invoked.
    ffmpeg = tmp_path / "ffmpeg"
    _write_executable(ffmpeg, "#!/bin/sh\nexit 1\n")
    with pytest.raises(MediaToolError, match="no frames"):
        extract_frames(video, tmp_path / "empty", ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    # [t2-ffmpeg-isolation] - END


def test_frame_manifest_alignment_requires_matching_source_hashes(raw_hevc, tmp_path):
    from nnslr_tools import media
    assert hasattr(media, "extracted_frame_rows"), "missing provenance-aware frame manifest builder"
    from nnslr_tools.manifest import sha256_of
    frames = extract_frames(raw_hevc, tmp_path / "frames", fps=1)
    log = tmp_path / "rlog.zst"
    log.write_bytes(b"synthetic log identity")
    alignment = tmp_path / "alignment.jsonl"
    header = dict(kind="alignment_report", source_video_sha256=sha256_of(raw_hevc),
                  source_log_sha256=sha256_of(log), log=str(log), segment_num=0, stream="narrow_road")
    rows = [dict(kind="frame", decoded_frame_index=i, capture_mono_ns=1000000000+i*50000000,
                 capture_reference="sof", alignment_status="exact", alignment_reason=None,
                 segment_num=0) for i in (0,20)]
    alignment.write_text('\n'.join(json.dumps(r) for r in [header,*rows]))
    result = media.extracted_frame_rows(frames, route_id="00000001--abcdef1234", segment_index=0, alignment=alignment)
    assert [r["capture_mono_ns"] for r in result] == [1000000000,2000000000]
    assert all(r["capture_time_provenance"]=="encode_index_sof" for r in result)
    assert all(r["source_log_sha256"]==sha256_of(log) for r in result)
    log.write_bytes(b"different log")
    with pytest.raises(MediaToolError, match="alignment_source_mismatch"):
        media.extracted_frame_rows(frames, route_id="00000001--abcdef1234", segment_index=0, alignment=alignment)
# [frame-provenance] - END
