# [nnslr-t2] - START
"""Local video probing and frame extraction for comma camera recordings.

This module never talks to a comma device. It only operates on local files and
uses ffprobe/ffmpeg through subprocess with shell=False.

The contract intentionally keeps video presentation timing separate from camera
capture timing. ffprobe can expose media PTS/DTS; capture_mono_ns is established
later from qlog/rlog alignment and is never fabricated here.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
# [frame-provenance] - START
import hashlib
import struct
import tempfile
import zlib
# [frame-provenance] - END
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence


class MediaToolError(RuntimeError):
    """Stable local-media failure with a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "", *, stderr: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        self.stderr = stderr


def _resolve_executable(name: str, explicit: str | None = None) -> str:
    candidate = explicit or os.environ.get(f"NNSLR_{name.upper()}", "").strip() or name
    resolved = shutil.which(candidate)
    if resolved is None:
        raise MediaToolError("media_tool_not_found", f"{name}: {candidate}")
    return resolved


def _run_json(cmd: Sequence[str], *, reject_stderr: bool = False) -> dict[str, Any]:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    if proc.returncode != 0 or (reject_stderr and proc.stderr.strip()):
        raise MediaToolError("media_tool_failed", " ".join(cmd), stderr=proc.stderr.strip())
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError("invalid_media_tool_json", str(exc), stderr=proc.stderr.strip()) from exc
    if not isinstance(payload, dict):
        raise MediaToolError("invalid_media_tool_json", "top-level payload is not an object")
    return payload


def _ratio_to_float(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        if isinstance(value, str) and "/" in value:
            f = Fraction(value)
            if f.denominator == 0:
                return None
            return float(f)
        out = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return out if math.isfinite(out) else None


def _number_or_none(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_camera_stream(path: Path) -> str | None:
    name = path.name.lower()
    if name == "fcamera.hevc" or name.startswith("fcamera."):
        return "narrow_road"
    if name == "ecamera.hevc" or name.startswith("ecamera."):
        return "wide_road"
    if name == "qcamera.ts" or name.startswith("qcamera."):
        return "q_narrow_road"
    if name == "dcamera.hevc" or name.startswith("dcamera."):
        return "cabin"
    return None


@dataclass(frozen=True)
class VideoProbe:
    path: str
    camera_stream: str | None
    codec: str | None
    codec_long_name: str | None
    container: str | None
    width: int | None
    height: int | None
    pixel_format: str | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    nominal_fps: float | None
    average_fps: float | None
    time_base: str | None
    start_time_s: float | None
    duration_s: float | None
    frame_count: int | None
    ffprobe_version: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "camera_stream": self.camera_stream,
            "codec": self.codec,
            "codec_long_name": self.codec_long_name,
            "container": self.container,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "color_range": self.color_range,
            "color_space": self.color_space,
            "color_transfer": self.color_transfer,
            "color_primaries": self.color_primaries,
            "nominal_fps": self.nominal_fps,
            "average_fps": self.average_fps,
            "time_base": self.time_base,
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "ffprobe_version": self.ffprobe_version,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class VideoFrameTiming:
    decoded_frame_index: int
    pts_time_s: float | None
    dts_time_s: float | None
    best_effort_timestamp_s: float | None
    key_frame: bool
    pict_type: str | None
    # [media-clock] - START
    media_time_s: float | None = None
    media_time_source: str = "unknown"
    # [media-clock] - END

    def to_dict(self) -> dict[str, Any]:
        return {
            "decoded_frame_index": self.decoded_frame_index,
            "pts_time_s": self.pts_time_s,
            "dts_time_s": self.dts_time_s,
            "best_effort_timestamp_s": self.best_effort_timestamp_s,
            "key_frame": self.key_frame,
            "pict_type": self.pict_type,
            "media_time_s": self.media_time_s,
            "media_time_source": self.media_time_source,
        }


@dataclass(frozen=True)
class ExtractedFrame:
    image_path: str
    output_index: int
    source_video: str
    requested_sample_fps: float | None
    start_s: float | None
    end_s: float | None
    # [frame-provenance] - START
    decoded_frame_index: int
    camera_stream: str | None
    width: int
    height: int
    media_time_s: float | None
    media_time_provenance: str
    source_video_sha256: str
    image_sha256: str
    # [frame-provenance] - END

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "output_index": self.output_index,
            "source_video": self.source_video,
            "requested_sample_fps": self.requested_sample_fps,
            "start_s": self.start_s,
            "end_s": self.end_s,
            # [frame-provenance] - START
            "schema_version": 2,
            "decoded_frame_index": self.decoded_frame_index,
            "camera_stream": self.camera_stream,
            "width": self.width,
            "height": self.height,
            "media_time_s": self.media_time_s,
            "media_time_provenance": self.media_time_provenance,
            "capture_mono_ns": None,
            "capture_time_provenance": "unknown",
            "alignment_status": "unresolved",
            "alignment_reason": "capture_alignment_not_supplied",
            "source_video_sha256": self.source_video_sha256,
            "source_log_sha256": None,
            "image_sha256": self.image_sha256,
            # [frame-provenance] - END
        }


def ffprobe_version(*, ffprobe: str | None = None) -> str | None:
    exe = _resolve_executable("ffprobe", ffprobe)
    proc = subprocess.run([exe, "-version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    first = proc.stdout.splitlines()
    return first[0].strip() if first else None


def probe_video(path: Path, *, ffprobe: str | None = None) -> VideoProbe:
    if not path.is_file():
        raise MediaToolError("video_not_found", str(path))
    exe = _resolve_executable("ffprobe", ffprobe)
    payload = _run_json([
        exe, "-v", "warning", "-count_frames", "-show_streams",
        "-show_format", "-print_format", "json", str(path),
    ])

    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        streams = []
    video = next((s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"), None)
    fmt = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    warnings: list[str] = []
    if video is None:
        warnings.append("no_video_stream")
        video = {}

    frame_count = _int_or_none(video.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _int_or_none(video.get("nb_frames"))
        if frame_count is not None:
            warnings.append("frame_count_from_metadata_not_counted")

    duration_s = _number_or_none(video.get("duration"))
    if duration_s is None:
        duration_s = _number_or_none(fmt.get("duration"))
    if duration_s is None:
        warnings.append("duration_unknown")

    nominal = _ratio_to_float(video.get("r_frame_rate"))
    average = _ratio_to_float(video.get("avg_frame_rate"))
    if average is None and nominal is None:
        warnings.append("fps_unknown")

    start_video = _number_or_none(video.get("start_time"))
    return VideoProbe(
        path=str(path),
        camera_stream=infer_camera_stream(path),
        codec=video.get("codec_name") if isinstance(video.get("codec_name"), str) else None,
        codec_long_name=video.get("codec_long_name") if isinstance(video.get("codec_long_name"), str) else None,
        container=fmt.get("format_name") if isinstance(fmt.get("format_name"), str) else None,
        width=_int_or_none(video.get("width")),
        height=_int_or_none(video.get("height")),
        pixel_format=video.get("pix_fmt") if isinstance(video.get("pix_fmt"), str) else None,
        color_range=video.get("color_range") if isinstance(video.get("color_range"), str) else None,
        color_space=video.get("color_space") if isinstance(video.get("color_space"), str) else None,
        color_transfer=video.get("color_transfer") if isinstance(video.get("color_transfer"), str) else None,
        color_primaries=video.get("color_primaries") if isinstance(video.get("color_primaries"), str) else None,
        nominal_fps=nominal,
        average_fps=average,
        time_base=video.get("time_base") if isinstance(video.get("time_base"), str) else None,
        start_time_s=start_video if start_video is not None else _number_or_none(fmt.get("start_time")),
        duration_s=duration_s,
        frame_count=frame_count,
        ffprobe_version=ffprobe_version(ffprobe=exe),
        warnings=tuple(warnings),
    )


def probe_frame_timestamps(path: Path, *, ffprobe: str | None = None) -> list[VideoFrameTiming]:
    """Return media presentation/decode timing; never camera capture timing."""
    if not path.is_file():
        raise MediaToolError("video_not_found", str(path))
    exe = _resolve_executable("ffprobe", ffprobe)
    payload = _run_json([
        exe, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_format",
        "-show_entries",
        "frame=pts_time,pkt_dts_time,best_effort_timestamp_time,key_frame,pict_type,duration_time,pkt_duration_time:format=format_name",
        "-print_format", "json", str(path),
    ], reject_stderr=True)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise MediaToolError("invalid_frame_probe", "frames is not a list")

    out: list[VideoFrameTiming] = []
    for idx, raw in enumerate(frames):
        if not isinstance(raw, dict):
            raise MediaToolError("invalid_frame_probe", f"frame {idx} is not an object")
        out.append(VideoFrameTiming(
            decoded_frame_index=idx,
            pts_time_s=_number_or_none(raw.get("pts_time")),
            dts_time_s=_number_or_none(raw.get("pkt_dts_time")),
            best_effort_timestamp_s=_number_or_none(raw.get("best_effort_timestamp_time")),
            key_frame=bool(raw.get("key_frame") == 1),
            pict_type=raw.get("pict_type") if isinstance(raw.get("pict_type"), str) else None,
        ))
    # [media-clock] - START
    # Raw HEVC carries frame durations but no container timestamps. Reproduce
    # its decoder media clock only when every duration is valid; never use
    # avg_frame_rate (ffprobe may report 25 for a 20 Hz camera).
    durations = [_number_or_none(raw.get("duration_time", raw.get("pkt_duration_time"))) for raw in frames]
    generated = (
        payload.get("format", {}).get("format_name") == "hevc"
        and bool(out)
        and all(f.pts_time_s is None and f.best_effort_timestamp_s is None and f.dts_time_s is None for f in out)
        and all(d is not None and d > 0 for d in durations)
    )
    elapsed = 0.0
    for i, frame in enumerate(out):
        if frame.best_effort_timestamp_s is not None:
            time, source = frame.best_effort_timestamp_s, "best_effort_timestamp"
        elif frame.pts_time_s is not None:
            time, source = frame.pts_time_s, "pts"
        elif generated:
            time, source = elapsed, "raw_hevc_frame_duration"
        else:
            time, source = None, "unknown"
        out[i] = replace(frame, media_time_s=time, media_time_source=source)
        if generated:
            elapsed += durations[i]
    # [media-clock] - END
    return out


def extract_frames(
    video: Path,
    output_dir: Path,
    *,
    fps: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    alignment: Path | None = None,
    route_id: str | None = None,
    segment_index: int | None = None,
    overwrite: bool = False,
    decoded: Sequence[int] | None = None,
    prefix: str = "frame",
    index_base: int | None = None,
    timings: Sequence[VideoFrameTiming] | None = None,
    gop: tuple[int, bytes] | None = None,
) -> list[ExtractedFrame]:
    """Decode local video frames to PNG without resizing. ``decoded``: these frames of the video (decoded frame
    indices) instead of a sampling; ``prefix``: the name of the pictures (``<prefix>_<output index>.png``), only those
    are replaced; ``index_base``: each picture's output index is index_base + its decoded frame index instead of its
    number in the selection. ``timings``: probe_frame_timestamps of the video, when the caller has it already (the
    probe decodes the whole video). ``gop``: (its first decoded frame index, the parameter sets and the bytes of one
    GOP of a raw HEVC video): only that GOP is decoded, for frames of ``decoded`` inside it; its pictures are the same
    as the decode of the whole video."""
    if not video.is_file():
        raise MediaToolError("video_not_found", str(video))
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise MediaToolError("invalid_fps", repr(fps))
    if start_s is not None and (not math.isfinite(start_s) or start_s < 0):
        raise MediaToolError("invalid_start", repr(start_s))
    if end_s is not None and (not math.isfinite(end_s) or end_s < 0):
        raise MediaToolError("invalid_end", repr(end_s))
    if start_s is not None and end_s is not None and end_s <= start_s:
        raise MediaToolError("invalid_range", f"{start_s}..{end_s}")

    # [frame-provenance] - START
    existing = sorted(output_dir.glob(f"{prefix}_*.png"))
    if existing and not overwrite:
        raise MediaToolError("output_not_empty", str(output_dir))
    exe = _resolve_executable("ffmpeg", ffmpeg)
    source_hash = _file_sha256(video)
    if timings is None:
        timings = probe_frame_timestamps(video, ffprobe=ffprobe)
    if gop is not None and decoded is None:
        raise MediaToolError("frame_extraction_failed", "a GOP is decoded only for chosen frames")
    timed_selection = decoded is None and (fps is not None or start_s is not None or end_s is not None)
    if timed_selection and any(f.media_time_s is None for f in timings):
        raise MediaToolError("sampling_time_unknown", str(video))
    origin = timings[0].media_time_s if timings and timings[0].media_time_s is not None else 0.0
    selected = []
    next_sample = start_s or 0.0
    previous = None
    wanted = None if decoded is None else set(decoded)
    for frame in timings:
        t = frame.media_time_s
        if wanted is not None:
            if frame.decoded_frame_index in wanted:
                selected.append(frame)
            continue
        if t is not None:
            if previous is not None and t <= previous:
                raise MediaToolError("nonmonotonic_media_time", str(video))
            previous = t
            relative = t - origin
            if relative + 1e-9 < (start_s or 0.0) or (end_s is not None and relative >= end_s - 1e-9):
                continue
            if fps is not None:
                if relative + 1e-9 < next_sample:
                    continue
                # One source frame per interval; no duplication on sparse/VFR input.
                next_sample = (start_s or 0.0) + (math.floor((relative - (start_s or 0.0)) * fps + 1e-9) + 1) / fps
        selected.append(frame)
    if not selected:
        raise MediaToolError("frame_extraction_failed", "no frames in requested interval")
    first = 0 if gop is None else gop[0]
    if any(f.decoded_frame_index < first for f in selected):
        raise MediaToolError("frame_extraction_failed", f"a frame before the GOP from {first}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage a complete decode before publishing anything or replacing old images.
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output_dir) as temporary:
        staging = Path(temporary)
        expression = "+".join(f"eq(n\\,{f.decoded_frame_index - first})" for f in selected)
        graph = staging / "select.txt"
        graph.write_text("select=" + expression)
        source = ["-i", str(video)] if gop is None else ["-f", "hevc", "-i", "-"]
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-xerror", "-err_detect", "explode",
               "-abort_on", "empty_output", "-n", *source, "-filter_script:v", str(graph),
               "-map", "0:v:0", "-vsync", "0", "-start_number", "0", str(staging / "frame_%08d.png")]
        proc = subprocess.run(cmd, input=None if gop is None else gop[1], capture_output=True, check=False)
        stderr = proc.stderr.decode(errors="replace").strip()
        if proc.returncode != 0 or stderr:
            raise MediaToolError("frame_extraction_failed", str(video), stderr=stderr)
        files = sorted(staging.glob("frame_*.png"))
        if len(files) != len(selected):
            raise MediaToolError("extracted_frame_count_mismatch", f"{len(files)} != {len(selected)}")
        if _file_sha256(video) != source_hash:
            raise MediaToolError("source_video_changed", str(video))
        result = []
        names = []
        for i, (path, frame) in enumerate(zip(files, selected)):
            width, height = png_dimensions(path)
            index = i if index_base is None else index_base + frame.decoded_frame_index
            names.append(f"{prefix}_{index:08d}.png")
            result.append(ExtractedFrame(
                str(output_dir / names[-1]), index, str(video), None if decoded is not None else fps, start_s, end_s,
                frame.decoded_frame_index, infer_camera_stream(video), width, height,
                frame.media_time_s, frame.media_time_source, source_hash, _file_sha256(path),
            ))
        # [dataset-review] - START
        # Complete provenance checks while images are still private staging files.
        if alignment is not None:
            extracted_frame_rows(result, route_id=route_id, segment_index=segment_index, alignment=alignment)
        # [dataset-review] - END
        for path, name in zip(files, names):
            path.replace(output_dir / name)
        current_names = set(names)
        for path in existing:
            if path.name not in current_names:
                path.unlink()
    return result


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extracted_frame_rows(frames: Sequence[ExtractedFrame], *, route_id: str | None = None,
                         segment_index: int | None = None, alignment: Path | None = None) -> list[dict]:
    """Attach capture evidence only from an alignment bound to unchanged sources."""
    rows = [dict(f.to_dict(), route_id=route_id, segment_index=segment_index) for f in frames]
    if alignment is None:
        return rows
    records = [json.loads(line) for line in alignment.read_text().splitlines() if line.strip()]
    if not records or any(not isinstance(r, dict) for r in records):
        raise MediaToolError("invalid_alignment", str(alignment))
    header = records[0]
    log = Path(header.get("log", ""))
    if not log.is_absolute():
        log = alignment.parent / log
    if (header.get("kind") != "alignment_report" or not log.is_file()
            or header.get("source_log_sha256") != _file_sha256(log)
            or any(header.get("source_video_sha256") != f.source_video_sha256 for f in frames)
            or any(header.get("stream") != f.camera_stream for f in frames)
            or (segment_index is not None and header.get("segment_num") != segment_index)):
        raise MediaToolError("alignment_source_mismatch", "regenerate alignment for these video/log sources")
    by_index = {}
    for record in records[1:]:
        index = record.get("decoded_frame_index")
        if record.get("kind") != "frame" or type(index) is not int or index < 0 or index in by_index:
            raise MediaToolError("invalid_alignment", "invalid/duplicate frame index")
        by_index[index] = record
    for row in rows:
        row["source_log_sha256"] = header["source_log_sha256"]
        source = by_index.get(row["decoded_frame_index"])
        if source is None:
            row["alignment_reason"] = "missing_alignment_frame"
        elif source.get("alignment_status") == "exact":
            capture = source.get("capture_mono_ns")
            reference = source.get("capture_reference")
            if (type(capture) is not int or capture <= 0 or reference not in ("sof", "eof")
                    or source.get("segment_num") != header.get("segment_num")):
                raise MediaToolError("invalid_alignment", "invalid exact capture evidence")
            row.update(capture_mono_ns=capture, capture_time_provenance="encode_index_" + reference,
                       alignment_status="exact", alignment_reason=None)
        elif source.get("alignment_status") == "unresolved" and source.get("capture_mono_ns") is None:
            row["alignment_reason"] = source.get("alignment_reason") or "unresolved_alignment"
        else:
            raise MediaToolError("invalid_alignment", "unsupported or inconsistent alignment status")
    return rows


def png_dimensions(path: Path, *, verify: bool = False) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise MediaToolError("invalid_png", str(path))
    width, height = struct.unpack(">II", header[16:24])
    if width < 1 or height < 1:
        raise MediaToolError("invalid_png_dimensions", str(path))
    # [dataset-review] - START
    if verify:
        _verify_png(path, width, height)
    # [dataset-review] - END
    return width, height
    # [frame-provenance] - END


# [dataset-review] - START
def _verify_png(path: Path, width: int, height: int) -> None:
    """Check chunk integrity and complete, correctly sized PNG scanline data."""
    def check(condition):
        if not condition:
            raise MediaToolError('invalid_png', str(path))
    data = path.read_bytes()
    offset, header, ended = 8, None, False
    compressed = bytearray()
    palette = False
    while offset < len(data):
        check(offset + 12 <= len(data))
        length = struct.unpack('>I', data[offset:offset+4])[0]
        kind = data[offset+4:offset+8]
        end = offset + 12 + length
        check(end <= len(data))
        payload = data[offset+8:end-4]
        check(zlib.crc32(kind + payload) == struct.unpack('>I', data[end-4:end])[0])
        if header is None:
            check(kind == b'IHDR' and length == 13)
            header = payload
        elif kind == b'IHDR':
            check(False)
        if kind == b'PLTE':
            check(not compressed and 0 < length <= 768 and length % 3 == 0)
            palette = True
        if kind == b'IDAT':
            compressed.extend(payload)
        if kind == b'IEND':
            check(length == 0 and end == len(data))
            ended = True
        offset = end
    check(ended and header is not None and bool(compressed))
    bit_depth, color, compression, filtering, interlace = header[8:]
    depths = {0: (1,2,4,8,16), 2: (8,16), 3: (1,2,4,8), 4: (8,16), 6: (8,16)}
    check(color in depths and bit_depth in depths[color] and compression == 0 and filtering == 0 and interlace in (0,1))
    check(color != 3 or palette)
    channels = {0:1, 2:3, 3:1, 4:2, 6:4}[color]
    passes = [(0,0,1,1)] if interlace == 0 else [(0,0,8,8),(4,0,8,8),(0,4,4,8),(2,0,4,4),(0,2,2,4),(1,0,2,2),(0,1,1,2)]
    layouts = []
    for x, y, dx, dy in passes:
        w, h = max(0, (width-x+dx-1)//dx), max(0, (height-y+dy-1)//dy)
        if w and h:
            layouts.append(((w * channels * bit_depth + 7)//8 + 1, h))
    expected = sum(size * count for size, count in layouts)
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(compressed, expected + 1)
    except zlib.error as exc:
        raise MediaToolError('invalid_png', str(path)) from exc
    check(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and len(decoded) == expected)
    position = 0
    for size, count in layouts:
        for _ in range(count):
            check(decoded[position] <= 4)
            position += size
# [dataset-review] - END


def make_clip(
    video: Path,
    output: Path,
    *,
    start_s: float,
    end_s: float,
    ffmpeg: str | None = None,
    reencode: bool = False,
) -> Path:
    """Create a local clip for review/candidate labeling."""
    if not video.is_file():
        raise MediaToolError("video_not_found", str(video))
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or start_s < 0 or end_s <= start_s:
        raise MediaToolError("invalid_range", f"{start_s}..{end_s}")
    exe = _resolve_executable("ffmpeg", ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)

    # [hevc-seek] - START
    # Packet-copy seeking cannot cut raw HEVC at a requested media time.
    # Decode from the beginning and encode the selected interval automatically.
    reencode = reencode or video.suffix.lower() in {".hevc", ".h265"}
    cmd = [exe, "-hide_banner", "-loglevel", "warning", "-abort_on", "empty_output", "-y"]
    if not reencode:
        cmd += ["-ss", f"{start_s:.9f}"]
    cmd += ["-i", str(video)]
    if reencode:
        cmd += ["-ss", f"{start_s:.9f}"]
    cmd += ["-t", f"{end_s - start_s:.9f}", "-map", "0:v:0"]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-fps_mode", "passthrough", "-an"]
    # [hevc-seek] - END
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(output))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise MediaToolError("clip_extraction_failed", " ".join(cmd), stderr=proc.stderr.strip())
    return output
# [nnslr-t2] - END
