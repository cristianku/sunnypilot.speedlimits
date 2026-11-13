# [nnslr-t1] - START
"""``nnslr`` command-line interface (T1).

Implemented subcommands (T1):

- ``version``       — print package/core versions.
- ``env``           — machine-readable environment report. Distinguishes
  *available*, *untested* and *failed* for GPU support; never runs a GPU
  workload implicitly (plan §12.4 ``check_environment`` contract).
- ``validate-batch``— validate an ObservationBatch JSON document against the
  §7 contract; deterministic, stable reason codes; nonzero exit on rejection.
- ``selftest``      — run the bundled synthetic fixtures through
  :func:`speed_vision_core.types.validate_batch` and the serialization
  round-trip; the CPU/synthetic quickstart for a clean clone.

Local extraction/alignment (T2) and reviewed annotation import/validation/splits
(T3) are implemented below. Training (T4), evaluation (T5–T6) and export/bundle
(T7–T8) remain explicitly unimplemented.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Importing this module must not drag in torch/CUDA/openpilot (test-guarded).
import speed_vision_core
from speed_vision_core import types
from speed_vision_core.types import (
    MAX_DETECTIONS_PER_BATCH,
    ReasonCode,
    SUPPORTED_SCHEMA_VERSION,
    NnslerContractError,
)

from nnslr_tools import __version__
from nnslr_tools.alignment import align_comma_segment, comma_aligned_frame_from_dict
from nnslr_tools.candidates import find_speed_candidates, project_candidates_to_video
from nnslr_tools.comma_log import (
    CommaLogError,
    encode_events,
    map_speed_events,
    read_log_metadata,
)
from nnslr_tools.media import MediaToolError, extract_frames, make_clip, probe_frame_timestamps, probe_video
# [frame-provenance] - START
from nnslr_tools.media import extracted_frame_rows
from nnslr_tools.manifest import sha256_of
# [frame-provenance] - END
# [route-extract] - START
from nnslr_tools.manifest import NnslerManifestError, RouteManifest, normalize_relpath, parse_route_id
from nnslr_tools.route_io import build_route_manifest, discover_segment_dirs

# Segments of a route extracted at once: Prepare runs two routes, so 20 on the server's 32 cores.
EXTRACT_WORKERS = 10
# [route-extract] - END
# [nnslr-sync] - START
# [route-extract] - START
from nnslr_tools.sync import DEFAULT_DATA_ROOT, DEFAULT_VIDEOS_DIR, link_videos, resolve_videos_dir, sync_route
# [route-extract] - END
# [nnslr-sync] - END


# ---------------------------------------------------------------------------
# Environment report
# ---------------------------------------------------------------------------

def collect_env_report() -> dict[str, Any]:
    """Machine-readable environment facts. No GPU probe without an explicit
    flag; the report marks GPU as *untested* by default."""
    data_root = os.environ.get("NNSLR_DATA_ROOT", "").strip()
    data_root_path = Path(data_root) if data_root else None
    videos_env = os.environ.get("COMMA_VIDEOS_DIR", "").strip()
    videos_dir = resolve_videos_dir()
    if videos_env:
        videos_note = ("COMMA_VIDEOS_DIR is read from the environment. Raw comma files are stored "
                       "there once; <data root>/raw/routes only holds symlinks to them.")
    else:
        videos_note = (f"COMMA_VIDEOS_DIR is not set; raw comma files default to {DEFAULT_VIDEOS_DIR}. "
                       "<data root>/raw/routes only holds symlinks to them.")
    report: dict[str, Any] = {
        "nnslr_tools": __version__,
        "speed_vision_core": speed_vision_core.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nnsler_data_root": {
            "configured": bool(data_root),
            "path": data_root or None,
            "exists": bool(data_root_path and data_root_path.is_dir()),
            "note": (
                "NNSLR_DATA_ROOT is read from the environment, never "
                "hardcoded (decision D4)."
                if data_root
                # [nnslr-sync] - START
                else f"NNSLR_DATA_ROOT is not set; data root defaults to {DEFAULT_DATA_ROOT}."
                # [nnslr-sync] - END
            ),
        },
        "comma_videos_dir": {
            "configured": bool(videos_env),
            "path": str(videos_dir),
            "exists": videos_dir.is_dir(),
            "note": videos_note,
        },
        "cpu": {
            "available": True,
            "status": "available",
        },
        "gpu": {
            "status": "untested",
            "available": None,
            "note": (
                "GPU support is only established by an explicitly authorized "
                "forward/backward/export probe (plan §12.4 check_environment "
                "--gpu-smoke). This report does not probe."
            ),
        },
        "openpilot_dependent": False,
    }
    return report


# ---------------------------------------------------------------------------
# validate-batch
# ---------------------------------------------------------------------------

def _parse_frame_key(key: str) -> types.FrameIdentity:
    """Parse ``session:stream:frame_id`` into a :class:`FrameIdentity`.

    The session id must not contain ``:``; the stream is one of the
    :class:`StreamId` members; the frame id is the trailing integer."""
    session_id, stream, frame_id = key.split(":", 2)
    return types.FrameIdentity(session_id, types.StreamId(stream), int(frame_id))


def validate_batch_from_payload(
    payload: dict[str, Any],
    now_mono_ns: int,
    expected_session_id: str,
    recent_frame_keys: frozenset[str] | None = None,
) -> types.ValidationResult:
    """Deserialize a batch payload and run the §7.4 validation contract.

    ``recent_frame_keys`` are frame keys (``session:stream:frame_id``) of frames
    already accepted in the session; a batch matching one is rejected with
    ``DUPLICATE_FRAME``."""
    batch = types.batch_from_dict(payload)
    recent = (
        frozenset(_parse_frame_key(k) for k in recent_frame_keys)
        if recent_frame_keys is not None
        else None
    )
    return types.validate_batch(
        batch,
        now_mono_ns=now_mono_ns,
        expected_session_id=expected_session_id,
        recent_frame_ids=recent,
    )


def _cmd_validate_batch(args: argparse.Namespace) -> int:
    path = Path(args.batch)
    if not path.is_file():
        print(f"error: batch file not found: {path}", file=sys.stderr)
        return 2
    payload = json.loads(path.read_text(encoding="utf-8"))
    now = int(args.now_mono_ns)
    session = args.expected_session_id
    try:
        result = validate_batch_from_payload(payload, now_mono_ns=now, expected_session_id=session)
    except NnslerContractError as exc:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "batch_identity": None,
                    "reason_codes": [exc.reason_code],
                    "rejected_indices": [],
                    "error": exc.message,
                },
                indent=2,
            )
        )
        return 1
    out = types.validation_result_to_dict(result)
    print(json.dumps(out, indent=2))
    return 0 if result.accepted else 1


# ---------------------------------------------------------------------------
# selftest (synthetic quickstart)
# ---------------------------------------------------------------------------

def _fixture_dir() -> Path:
    """Locate the synthetic fixture directory.

    Preferred order: a ``nnslr_tools/fixtures`` directory shipped with the
    installed package, then the repository checkout's ``tests/fixtures``
    (where the canonical fixtures live).
    """
    import importlib.resources as resources

    try:
        pkg_fixtures = resources.files("nnslr_tools").joinpath("fixtures")
        if pkg_fixtures.is_dir() and any(pkg_fixtures.iterdir()):
            return Path(str(pkg_fixtures))
    except FileNotFoundError:
        pass
    return Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


def _selftest() -> int:
    """Run the bundled synthetic fixtures end-to-end on CPU.

    Returns 0 only if every fixture produced its documented reason code and
    the serialization round-trip preserved absence/unknown.

    Time is injected, never read. Most fixtures use ``now = capture + 10 ms``;
    the fixtures that test the freshness/duplicate rules override ``now`` or
    provide ``recent_frame_keys`` explicitly (see ``_NOW_OVERRIDES`` and
    ``_RECENT_KEYS``).
    """
    # Most fixtures use a small positive offset (fresh capture). A few test the
    # freshness rules and need a specific offset relative to the capture:
    #   stale  -> now = capture + 5 s  (age 5 s > 0.35 s window)
    #   future -> now = capture - 1 s  (capture is in the future of "now")
    _NOW_OFFSETS_NS = {
        "malformed_stale_evidence.json": 5_000_000_000,
        "malformed_invalid_capture_timestamp_future.json": -1_000_000_000,
    }
    _DEFAULT_NOW_OFFSET_NS = 10_000_000  # +10 ms, well inside the 0.35 s window

    # Fixture whose rejection depends on the recent-frame set, not on the
    # payload alone: the frame identity is already present.
    _RECENT_KEYS = {
        "malformed_duplicate_frame.json": "selftest:narrow_road:2",
    }

    failures: list[str] = []
    passed: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        (passed if ok else failures).append(name + (f" — {detail}" if detail and not ok else ""))

    fixtures_dir = _fixture_dir()
    if not fixtures_dir.is_dir():
        print(f"selftest FAILED: fixture directory not found: {fixtures_dir}")
        return 1

    # 1) validate-batch on the valid fixture.
    with (fixtures_dir / "valid_batch.json").open() as f:
        payload = json.load(f)
    now = int(payload["frame"]["capture_mono_ns"]) + _DEFAULT_NOW_OFFSET_NS  # +10 ms
    result = validate_batch_from_payload(payload, now_mono_ns=now, expected_session_id="selftest")
    check("valid_batch accepted", result.accepted, str(result.reason_codes))

    # 2) every malformed fixture must be rejected with its documented code.
    malformed = sorted(p.name for p in fixtures_dir.iterdir() if p.name.startswith("malformed_"))
    for name in malformed:
        with (fixtures_dir / name).open() as f:
            payload = json.load(f)
        expected_code = name[len("malformed_") : -len(".json")]
        capture = int(payload["frame"]["capture_mono_ns"])
        offset = _NOW_OFFSETS_NS.get(name, _DEFAULT_NOW_OFFSET_NS)
        recent_keys = (
            frozenset({_RECENT_KEYS[name]}) if name in _RECENT_KEYS else None
        )
        try:
            result = validate_batch_from_payload(
                payload,
                now_mono_ns=capture + offset,
                expected_session_id="selftest",
                recent_frame_keys=recent_keys,
            )
        except NnslerContractError as exc:
            check(f"{name} rejected", exc.reason_code == expected_code, f"got {exc.reason_code}")
            continue
        codes = set(result.reason_codes)
        check(
            f"{name} rejected with {expected_code}",
            (not result.accepted) and expected_code in codes,
            f"accepted={result.accepted} codes={sorted(codes)}",
        )

    # 3) serialization round-trip preserves absence and never reinterprets zero.
    with (fixtures_dir / "valid_batch.json").open() as f:
        payload = json.load(f)
    batch = types.batch_from_dict(payload)
    again = types.batch_to_dict(batch)
    check(
        "round-trip stable",
        json.dumps(again, sort_keys=True) == json.dumps(payload, sort_keys=True),
    )
    zero_payload = json.loads(json.dumps(payload))
    det = zero_payload["detections"][0]
    det["value_state"] = types.ValueState.UNKNOWN.value
    det["value_kph"] = 0  # zero with has_value=false must be rejected
    try:
        types.batch_from_dict(zero_payload)
        check("zero-not-limit rejected", False, "construction accepted zero as value")
    except NnslerContractError as exc:
        check(
            "zero-not-limit rejected",
            exc.reason_code == ReasonCode.INVALID_VALUE_CONSISTENCY,
            exc.reason_code,
        )

    print(f"nnslr selftest: nnslr_tools {__version__}")
    for line in passed:
        print(f"  PASS  {line}")
    for line in failures:
        print(f"  FAIL  {line}")
    if failures:
        print(f"selftest FAILED: {len(failures)} of {len(passed) + len(failures)}")
        return 1
    print(f"selftest OK: {len(passed)} checks")
    return 0


# ---------------------------------------------------------------------------
# T2 local comma file processing (never device access)
# ---------------------------------------------------------------------------

_STREAM_TO_ENCODE_SERVICE = {
    "narrow_road": "narrowRoadEncodeIdx",
    "wide_road": "wideRoadEncodeIdx",
    "q_narrow_road": "qNarrowRoadEncodeIdx",
}


def _write_jsonl(path: Path | None, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(item)
    return rows


def _resolve_data_root(explicit: str | None) -> Path:
    raw = (explicit or os.environ.get("NNSLR_DATA_ROOT", "")).strip()
    if not raw:
        raw = DEFAULT_DATA_ROOT
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")
    return root


def _cmd_route_manifest(args: argparse.Namespace) -> int:
    root = _resolve_data_root(args.data_root)
    manifest = build_route_manifest(root, Path(args.route))
    text = manifest.to_jsonl()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "declared_segments": list(manifest.declared_segments),
        **manifest.gap_report(),
        "file_count": len(manifest.files),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_manifest(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = RouteManifest.from_jsonl(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "schema_version": manifest.schema_version,
        "declared_segments": list(manifest.declared_segments),
        "file_count": len(manifest.files),
        **manifest.gap_report(),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_video_probe(args: argparse.Namespace) -> int:
    result = probe_video(Path(args.video), ffprobe=args.ffprobe)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_extract_frames(args: argparse.Namespace) -> int:
    # [route-extract] - START
    if args.route:
        return _cmd_extract_route(args)
    if not args.output:
        raise ValueError("--output is required with --video")
    # [route-extract] - END
    frames = extract_frames(
        Path(args.video),
        Path(args.output),
        fps=None if args.all_frames else args.fps,
        start_s=args.start,
        end_s=args.end,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        alignment=Path(args.alignment) if args.alignment else None,
        overwrite=args.overwrite,
    )
    rows = extracted_frame_rows(frames, alignment=Path(args.alignment) if args.alignment else None)
    if args.manifest:
        _write_jsonl(Path(args.manifest), rows)
    print(json.dumps({
        "frame_count": len(rows),
        "output": str(Path(args.output)),
        "manifest": args.manifest,
        "timing": "capture_aligned" if rows and all(r["alignment_status"] == "exact" for r in rows) else "media_only",
        "capture_time_established": bool(rows) and all(r["alignment_status"] == "exact" for r in rows),
    }, indent=2, sort_keys=True))
    return 0


# [route-extract] - START
def _cmd_extract_route(args: argparse.Namespace) -> int:
    """Apply the existing video extraction to every locally available segment."""
    parse_route_id(args.route)
    root = Path(args.data_root or os.environ.get("NNSLR_DATA_ROOT") or DEFAULT_DATA_ROOT).expanduser().resolve()
    route_rel = Path("raw/routes") / args.route
    normalize_relpath(root, route_rel)
    segments = discover_segment_dirs(root / route_rel)
    output = Path(args.output).expanduser() if args.output else root / "derived/frames" / args.route
    if not args.output:
        normalize_relpath(root, output.relative_to(root))

    # Preflight the complete route so a missing camera or existing output in a
    # later segment does not leave earlier segments unnecessarily processed.
    plans = []
    for identity, directory in segments:
        if identity.route_id != args.route:
            raise ValueError(f"unexpected route inside {root / route_rel}: {identity.route_id}")
        video = directory / "fcamera.hevc"
        normalize_relpath(root, video.relative_to(root))
        if not video.is_file():
            raise MediaToolError("video_not_found", str(video))
        segment_output = output / str(identity.segment_index)
        if not args.output:
            normalize_relpath(root, segment_output.relative_to(root))
        if not args.overwrite and next(segment_output.glob("frame_*.png"), None) is not None:
            raise MediaToolError("output_not_empty", str(segment_output))
        # [dataset-review] - START
        alignment_base = Path(args.alignment) if args.alignment else root / "derived/alignment" / args.route
        alignment_path = alignment_base / f"{identity.segment_index}.jsonl"
        if args.alignment and not alignment_path.is_file():
            raise ValueError(f"alignment missing: {alignment_path}")
        plans.append((identity.segment_index, video, segment_output,
                      alignment_path if alignment_path.is_file() else None))
        # [dataset-review] - END

    def extract_segment(plan):
        index, video, segment_output, alignment_path = plan
        print(f"Extracting segment {index}: {video}", file=sys.stderr, flush=True)
        frames = extract_frames(
            video, segment_output, fps=None if args.all_frames else args.fps,
            start_s=args.start, end_s=args.end, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, overwrite=args.overwrite,
            alignment=alignment_path, route_id=args.route, segment_index=index,
        )
        rows = extracted_frame_rows(frames, route_id=args.route, segment_index=index,
                                    alignment=alignment_path)
        manifest = segment_output / "frames.jsonl"
        _write_jsonl(manifest, rows)
        return rows, {"segment_index": index, "frame_count": len(rows),
                      "output": str(segment_output), "manifest": str(manifest)}

    # Several segments at once: ffprobe reads a whole video on one core, so one segment at a time left most of the
    # server idle. The work is in ffmpeg/ffprobe processes, so threads are enough; the results keep the segments' order.
    all_rows = []
    summaries = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for rows, summary in pool.map(extract_segment, plans):
            all_rows.extend(rows)
            summaries.append(summary)
    if args.manifest:
        _write_jsonl(Path(args.manifest), all_rows)
    print(json.dumps({
        "route": args.route, "frame_count": len(all_rows), "output": str(output),
        "segments": summaries, "manifest": args.manifest,
        "timing": "capture_aligned" if all_rows and all(r["alignment_status"] == "exact" for r in all_rows) else "media_only",
        "capture_time_established": bool(all_rows) and all(r["alignment_status"] == "exact" for r in all_rows),
    }, indent=2, sort_keys=True))
    return 0
# [route-extract] - END


def _cmd_log_metadata(args: argparse.Namespace) -> int:
    events = read_log_metadata(
        Path(args.log),
        openpilot_root=Path(args.openpilot_root) if args.openpilot_root else None,
        python_executable=args.openpilot_python,
        opendbc_root=Path(args.opendbc_root) if args.opendbc_root else None,
    )
    rows = [event.to_dict() for event in events]
    _write_jsonl(Path(args.output) if args.output else None, rows)
    return 0


def _cmd_align_route(args: argparse.Namespace) -> int:
    stream = args.stream
    try:
        service = _STREAM_TO_ENCODE_SERVICE[stream]
    except KeyError as exc:
        raise ValueError(f"unsupported stream {stream!r}") from exc

    frames = probe_frame_timestamps(Path(args.video), ffprobe=args.ffprobe)
    events = read_log_metadata(
        Path(args.log),
        openpilot_root=Path(args.openpilot_root) if args.openpilot_root else None,
        python_executable=args.openpilot_python,
        opendbc_root=Path(args.opendbc_root) if args.opendbc_root else None,
    )
    indexes = encode_events(events, service=service, segment_num=args.segment_num)
    aligned, report = align_comma_segment(frames, indexes, segment_num=args.segment_num)

    rows: list[dict[str, Any]] = [{
        "kind": "alignment_report",
        "stream": stream,
        "video": str(Path(args.video)),
        "log": str(Path(args.log)),
        # [frame-provenance] - START
        "source_video_sha256": sha256_of(Path(args.video)),
        "source_log_sha256": sha256_of(Path(args.log)),
        # [frame-provenance] - END
        "segment_num": args.segment_num,
        **report.to_dict(),
    }]
    rows.extend({"kind": "frame", **frame.to_dict()} for frame in aligned)
    _write_jsonl(Path(args.output), rows)

    print(json.dumps({
        "output": str(Path(args.output)),
        "stream": stream,
        "segment_num": args.segment_num,
        **report.to_dict(),
    }, indent=2, sort_keys=True))
    return 0 if report.unresolved == 0 else 1


def _cmd_alignment_report(args: argparse.Namespace) -> int:
    rows = _read_jsonl(Path(args.alignment))
    header = next((row for row in rows if row.get("kind") == "alignment_report"), None)
    if header is None:
        raise ValueError("alignment JSONL has no alignment_report record")
    print(json.dumps(header, indent=2, sort_keys=True))
    return 0


def _cmd_find_candidates(args: argparse.Namespace) -> int:
    events = read_log_metadata(
        Path(args.log),
        openpilot_root=Path(args.openpilot_root) if args.openpilot_root else None,
        python_executable=args.openpilot_python,
        opendbc_root=Path(args.opendbc_root) if args.opendbc_root else None,
    )
    candidates = find_speed_candidates(map_speed_events(events))

    if args.alignment:
        rows = _read_jsonl(Path(args.alignment))
        aligned = [
            comma_aligned_frame_from_dict(row)
            for row in rows
            if row.get("kind") == "frame"
        ]
        candidates = project_candidates_to_video(
            candidates,
            aligned,
            max_error_ns=args.max_projection_error_ns,
        )

    _write_jsonl(
        Path(args.output) if args.output else None,
        [{"kind": "candidate", **candidate.to_dict()} for candidate in candidates],
    )
    return 0


def _cmd_make_clips(args: argparse.Namespace) -> int:
    rows = _read_jsonl(Path(args.candidates))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    made: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        if row.get("kind") != "candidate":
            continue
        center = row.get("video_time_s")
        if not isinstance(center, (int, float)) or isinstance(center, bool):
            skipped += 1
            continue
        start = max(0.0, float(center) - args.before)
        end = float(center) + args.after
        candidate_id = str(row.get("candidate_id", f"candidate-{len(made)}"))
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in candidate_id)
        out = output_dir / f"{safe_id}.mp4"
        make_clip(
            Path(args.video),
            out,
            start_s=start,
            end_s=end,
            ffmpeg=args.ffmpeg,
            reencode=args.reencode,
        )
        made.append({
            "kind": "clip",
            "candidate_id": candidate_id,
            "clip_path": str(out),
            "start_s": start,
            "end_s": end,
            "source_video": str(Path(args.video)),
            "candidate_reason": row.get("candidate_reason"),
            "source_log_mono_time": row.get("log_mono_time"),
        })

    if args.manifest:
        _write_jsonl(Path(args.manifest), made)
    print(json.dumps({
        "clips_created": len(made),
        "candidates_without_video_time": skipped,
        "output": str(output_dir),
        "manifest": args.manifest,
    }, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# [nnslr-sync] - START
def _cmd_sync_routes(args: argparse.Namespace) -> int:
    try:
        report = sync_route(
            route=args.route, host=args.host, segments=args.segments,
            camera=args.camera, data_root=args.data_root, videos_dir=args.videos_dir,
            dry_run=args.dry_run,
        )
    except OSError as exc:
        print(f"error: sync-routes: {exc}", file=sys.stderr)
        return 2
    if report["warning"] is not None:
        print(f"warning: sync-routes: {report['warning']}", file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _cmd_link_videos(args: argparse.Namespace) -> int:
    try:
        report = link_videos(
            route=args.route, videos_dir=args.videos_dir,
            data_root=args.data_root, dry_run=args.dry_run,
        )
    except OSError as exc:
        print(f"error: link-videos: {exc}", file=sys.stderr)
        return 2
    if report["warning"] is not None:
        print(f"warning: link-videos: {report['warning']}", file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1
# [nnslr-sync] - END


# [training-baseline] - START
def _train_config(args: argparse.Namespace):
    """The DetectorTrainConfig or TrainConfig of nnslr train's arguments (torch is not imported)."""
    from nnslr_tools.checkpoints import DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, DEFAULT_DETECTOR_INPUT_SIZE, parse_input_size
    from nnslr_tools.training import DetectorTrainConfig, TrainConfig, _check_detector_config

    init_from = getattr(args, "init_from", None)
    mode = getattr(args, "mode", "fresh")
    if args.task == "detector":
        if init_from and mode != "continue":
            raise ValueError("--init-from starts a detector only with --mode continue")
        if getattr(args, "mapillary_only", False):
            raise ValueError("--mapillary-only is a classifier option: --task reader --classes v2 --mapillary")
        defaults = DetectorTrainConfig()
        config = DetectorTrainConfig(
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size if args.batch_size is not None else 8,
            learning_rate=args.learning_rate if args.learning_rate is not None else 1e-4,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            pretrained_backbone=args.pretrained,
            amp=not args.no_amp,
            input_size=parse_input_size(args.input_size) if args.input_size else DEFAULT_DETECTOR_INPUT_SIZE,
            anchor_min_ratio=(args.anchor_min_ratio if args.anchor_min_ratio is not None
                              else DEFAULT_DETECTOR_ANCHOR_MIN_RATIO),
            augment=not args.no_augment,
            negative_frame_fraction=(args.negative_frame_fraction if args.negative_frame_fraction is not None
                                     else defaults.negative_frame_fraction),
            select_by=args.select_by if args.select_by is not None else defaults.select_by,
            families=args.cutter_families if args.cutter_families is not None else defaults.families,
        )
        _check_detector_config(config)
        return config
    if args.cutter_families is not None:
        raise ValueError("--cutter-families is a detector option")
    classes = getattr(args, "classes", "v1")
    # --no-augment is also the classifier's (v2): its darker crops and box jitter are on by default.
    if (args.anchor_min_ratio is not None or (args.no_augment and classes != "v2")
            or args.negative_frame_fraction is not None or args.select_by is not None):
        raise ValueError("--anchor-min-ratio, --no-augment, --negative-frame-fraction and --select-by are "
                         "detector options (--no-augment also of the reader with --classes v2)")
    if args.input_size is not None and not str(args.input_size).isdigit():
        raise ValueError(f"invalid reader --input-size {args.input_size!r}: the side of the square crop in pixels")
    mapillary_only = bool(getattr(args, "mapillary_only", False))
    if mapillary_only and not getattr(args, "mapillary", None):
        raise ValueError("--mapillary-only trains on the Mapillary crops alone: add --mapillary")
    # --init-from outside continue: the start of a fresh classifier (the second stage of a two-stage run), recorded
    # in its configuration so that a resumed run finds it again.
    start = None
    if init_from and mode != "continue":
        if classes != "v2":
            raise ValueError("--init-from starts a fresh reader only with --classes v2 (or use --mode continue)")
        if args.pretrained and mode == "fresh":
            raise ValueError("--init-from and --pretrained are two different starts: choose one")
        start = str(Path(init_from).resolve())
    return TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size if args.batch_size is not None else 64,
        learning_rate=args.learning_rate if args.learning_rate is not None else 3e-4,
        weight_decay=args.weight_decay,
        input_size=int(args.input_size) if args.input_size is not None else 160,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        pretrained=args.pretrained,
        amp=not args.no_amp,
        class_scheme=classes,
        augment=classes == "v2" and not args.no_augment,
        mapillary=str(Path(args.mapillary).resolve()) if getattr(args, "mapillary", None) else None,
        **_mapillary_choice(args),
        mapillary_only=mapillary_only,
        init_from=start,
    )


def _mapillary_choice(args: argparse.Namespace) -> dict:
    """The region and share of the Mapillary crops of nnslr train (--mapillary-region, --mapillary-share), each in
    its one spelling; only with --mapillary."""
    from nnslr_tools.mapillary import FULL_SHARE, check_share, region_text
    region = region_text(getattr(args, "mapillary_region", None))
    share = check_share(getattr(args, "mapillary_share", None))
    if (region != "all" or share != FULL_SHARE) and not getattr(args, "mapillary", None):
        raise ValueError("--mapillary-region and --mapillary-share choose Mapillary crops: add --mapillary")
    return {"mapillary_region": region, "mapillary_share": share}


def _evaluated_families(args: argparse.Namespace) -> str:
    """The families a detector is measured on: those its checkpoint learnt (a checkpoint that does not say: the speed
    signs); without a checkpoint (--dry-run) --cutter-families, else the default of a new cutter."""
    from nnslr_tools.training import DEFAULT_CUTTER_FAMILIES, cutter_families

    if args.dry_run or not args.checkpoint:
        return args.cutter_families or DEFAULT_CUTTER_FAMILIES
    from nnslr_tools.checkpoints import torch_load
    families = cutter_families(torch_load(Path(args.checkpoint)))
    if args.cutter_families is not None and args.cutter_families != families:
        raise ValueError(f"--cutter-families {args.cutter_families}: the checkpoint cuts the families {families}")
    return families


def _reader_plan_options(args: argparse.Namespace) -> dict:
    """The class scheme and negatives file of a reader plan (nnslr train and evaluate)."""
    if args.task != "reader" and (args.classes != "v1" or args.negatives):
        raise ValueError("--classes and --negatives are reader options")
    if args.task != "detector" and getattr(args, "cutter_families", None) is not None:
        raise ValueError("--cutter-families is a detector option")
    if args.negatives and args.classes != "v2":
        raise ValueError("--negatives are crops of the classifier: add --classes v2")
    options = {"class_scheme": args.classes, "negatives_path": Path(args.negatives) if args.negatives else None}
    if getattr(args, "mapillary", None):
        if args.task != "reader" or args.classes != "v2":
            raise ValueError("--mapillary adds crops to the classifier: --task reader --classes v2")
        options["mapillary_path"] = Path(args.mapillary)
        options.update(_mapillary_choice(args))
        if getattr(args, "mapillary_only", False):
            options["mapillary_only"] = True
    elif getattr(args, "mapillary_region", None) or getattr(args, "mapillary_share", None) is not None:
        _mapillary_choice(args)
    return options


def _cmd_train(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.training import (
        prepare_detector_training,
        prepare_reader_training,
        train_detector,
        train_reader,
    )

    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    kind = args.dataset_kind
    default_dataset = (
        root / "datasets" / "training_candidate_dataset" / "objects.jsonl"
        if kind == "training-candidate"
        else root / "annotations" / "objects.jsonl"
    )
    dataset = Path(args.dataset) if args.dataset else default_dataset
    splits = Path(args.splits) if args.splits else root / "splits" / "latest.json"

    # The configuration is checked before the plan, so that a dry run refuses a wrong setting too.
    config = _train_config(args)
    options = _reader_plan_options(args)
    if args.task == "detector":
        plan = prepare_detector_training(dataset, splits, root, dataset_kind=kind, families=config.families)
    else:
        if getattr(config, "init_from", None):
            from nnslr_tools.training import reader_start
            options["start"] = reader_start(Path(config.init_from))
        plan = prepare_reader_training(dataset, splits, root, dataset_kind=kind, **options)

    if args.dry_run:
        summary = dict(plan)
        detail_key = "frames" if args.task == "detector" else "records"
        summary["record_count"] = len(plan[detail_key])
        summary.pop(detail_key, None)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if plan["trainable"] else 1

    if not args.output:
        raise ValueError("--output is required for a real training run")
    if args.mode == "continue" and not args.hf_repo and not args.init_from:
        raise ValueError("continue requires --hf-repo, NNSLR_HF_REPO or --init-from")

    if args.task == "detector":
        result = train_detector(plan, root, Path(args.output), config,
                                mode=args.mode, hf_repo=args.hf_repo, hf_revision=args.hf_revision,
                                init_from=args.init_from)
    else:
        result = train_reader(plan, root, Path(args.output), config,
                              mode=args.mode, hf_repo=args.hf_repo, hf_revision=args.hf_revision,
                              init_from=args.init_from if args.mode == "continue" else None)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
# [training-baseline] - END


# [evaluation-baseline] - START
def _cmd_evaluate(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.training import prepare_detector_training, prepare_reader_training

    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    kind = args.dataset_kind
    default_dataset = (
        root / "datasets" / "training_candidate_dataset" / "objects.jsonl"
        if kind == "training-candidate"
        else root / "annotations" / "objects.jsonl"
    )
    dataset = Path(args.dataset) if args.dataset else default_dataset
    splits = Path(args.splits) if args.splits else root / "splits" / "latest.json"

    options = _reader_plan_options(args)
    if args.task == "detector":
        plan = prepare_detector_training(dataset, splits, root, dataset_kind=kind,
                                         families=_evaluated_families(args))
        selected = [r for r in plan["frames"] if r["split"] == args.partition]
    else:
        plan = prepare_reader_training(dataset, splits, root, dataset_kind=kind, **options)
        selected = [r for r in plan["records"] if r["split"] == args.partition]

    if args.dry_run:
        from collections import Counter
        from nnslr_tools.evaluation import HUMAN_LABEL_BASIS, label_basis_counts
        report = {
            "task": args.task,
            "partition": args.partition,
            "dataset_kind": kind,
            "dataset_sha256": plan["dataset_sha256"],
            "split_sha256": plan["split_sha256"],
            "sample_count": len(selected),
            "counts_by_label_basis": label_basis_counts(selected),
            "human_reviewed_sample_count": sum(1 for r in selected if r.get("label_basis") == HUMAN_LABEL_BASIS),
            "checkpoint_required_for_execution": True,
        }
        if plan.get("class_scheme") == "v2":
            report.update({"class_scheme": "v2",
                           "counts_by_label": dict(sorted(Counter(r["label"] for r in selected).items())),
                           "not_a_sign_by_source": dict(sorted(Counter(r["source"] for r in selected
                                                                       if r["label"] == "not_a_sign").items()))})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if selected else 1

    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --dry-run")

    from nnslr_tools.evaluation import evaluate_detector, evaluate_reader, write_evaluation

    if args.task == "detector":
        report = evaluate_detector(
            plan,
            Path(args.checkpoint),
            root,
            partition=args.partition,
            device_name=args.device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            num_workers=args.num_workers,
        )
    else:
        report = evaluate_reader(
            plan,
            Path(args.checkpoint),
            root,
            partition=args.partition,
            device_name=args.device,
            batch_size=args.batch_size if args.batch_size is not None else 64,
            num_workers=args.num_workers,
        )
    write_evaluation(report, Path(args.output) if args.output else None)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
# [evaluation-baseline] - END


# [classifier-chain] - START
def _dataset_and_split(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """(data root, dataset, split) of a command with the usual dataset arguments."""
    from nnslr_tools.preannotate import resolve_data_root

    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    default_dataset = (root / "datasets" / "training_candidate_dataset" / "objects.jsonl"
                       if args.dataset_kind == "training-candidate" else root / "annotations" / "objects.jsonl")
    dataset = Path(args.dataset) if args.dataset else default_dataset
    splits = Path(args.splits) if args.splits else root / "splits" / "latest.json"
    return root, dataset, splits


def _cmd_mine_negatives(args: argparse.Namespace) -> int:
    from nnslr_tools.chain import mine_negatives, negatives_dry_run

    root, dataset, splits = _dataset_and_split(args)
    partitions = tuple(args.partition) if args.partition else ("train", "validation", "test")
    if args.dry_run:
        report = negatives_dry_run(dataset, splits, root, dataset_kind=args.dataset_kind, per_frame=args.per_frame,
                                   total_factor=args.total_factor, partitions=partitions)
    else:
        if not args.output:
            raise ValueError("--output is required unless --dry-run")
        report = mine_negatives(dataset, splits, root, args.cutter, Path(args.output), dataset_kind=args.dataset_kind,
                                score_threshold=args.score_threshold, per_frame=args.per_frame,
                                total_factor=args.total_factor, seed=args.seed, sign_score=args.sign_score,
                                partitions=partitions,
                                device_name=args.device, batch_size=args.batch_size, num_workers=args.num_workers)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _cmd_evaluate_proposer(args: argparse.Namespace) -> int:
    from nnslr_tools.proposer_eval import evaluate_proposer

    root, dataset, splits = _dataset_and_split(args)
    report = evaluate_proposer(args.proposer, root, dataset, splits, partition=args.partition,
                               dataset_kind=args.dataset_kind, limit=args.limit)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _cmd_evaluate_chain(args: argparse.Namespace) -> int:
    from nnslr_tools.chain import chain_dry_run, evaluate_chain
    from nnslr_tools.evaluation import write_evaluation

    root, dataset, splits = _dataset_and_split(args)
    if args.dry_run:
        report = chain_dry_run(dataset, splits, root, dataset_kind=args.dataset_kind, partition=args.partition)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["image_count"] else 1
    thresholds = None
    if args.score_thresholds is not None:
        try:
            thresholds = tuple(float(value) for value in args.score_thresholds.split(",") if value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --score-thresholds {args.score_thresholds!r}: numbers separated by commas") from exc
    report = evaluate_chain(dataset, splits, root, args.cutter, args.classifier, dataset_kind=args.dataset_kind,
                            partition=args.partition, score_thresholds=thresholds,
                            main_score_threshold=args.main_score_threshold, iou_threshold=args.iou_threshold,
                            device_name=args.device, batch_size=args.batch_size, num_workers=args.num_workers)
    write_evaluation(report, Path(args.output) if args.output else None)
    # The frames go to the file only: the terminal gets the measures.
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2, sort_keys=True,
                     allow_nan=False))
    return 0
# [classifier-chain] - END


# [experiment-b] - START
def _cmd_experiment_b(args: argparse.Namespace) -> int:
    from nnslr_tools.context_experiment import (
        ExperimentConfig,
        dry_run_summary,
        prepare_experiment_b,
        train_experiment_b,
    )
    from nnslr_tools.preannotate import resolve_data_root

    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    kind = args.dataset_kind
    default_dataset = (
        root / "datasets" / "training_candidate_dataset" / "objects.jsonl"
        if kind == "training-candidate"
        else root / "annotations" / "objects.jsonl"
    )
    dataset = Path(args.dataset) if args.dataset else default_dataset
    splits = Path(args.splits) if args.splits else root / "splits" / "latest.json"
    plan = prepare_experiment_b(dataset, splits, root, dataset_kind=kind, limit=args.limit)

    if args.dry_run:
        print(json.dumps(dry_run_summary(plan, args.variant), indent=2, sort_keys=True))
        return 0 if plan["trainable"] else 1

    if not args.output:
        raise ValueError("--output is required for a real experiment run")
    config = ExperimentConfig(
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        crop_size=args.crop_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        pretrained=args.pretrained,
        amp=not args.no_amp,
        init_from=args.init_from,
    )
    if args.init_from and not Path(args.init_from).is_file():
        raise ValueError(f"--init-from is not a file: {args.init_from}")
    result = train_experiment_b(plan, root, Path(args.output), config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
# [experiment-b] - END


# [hard-example-mining] - START
def _cmd_mine_hard_examples(args: argparse.Namespace) -> int:
    from nnslr_tools.hard_examples import mine_file

    report = mine_file(Path(args.evaluation), Path(args.output))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
# [hard-example-mining] - END


# [onnx-export] - START
def _cmd_export_onnx(args: argparse.Namespace) -> int:
    from nnslr_tools.exporting import export_detector_onnx, export_reader_onnx

    if args.task == "signnet":
        from nnslr_tools.signnet import export_signnet_onnx
        report = export_signnet_onnx(Path(args.checkpoint), Path(args.output), verify=not args.no_verify)
    elif args.task == "usernet":
        from nnslr_tools.usernet import export_usernet_onnx
        report = export_usernet_onnx(Path(args.checkpoint), Path(args.output), verify=not args.no_verify)
    elif args.task == "detector":
        report = export_detector_onnx(
            Path(args.checkpoint),
            Path(args.output),
            verify=not args.no_verify,
            input_width=args.detector_input_width,
            input_height=args.detector_input_height,
        )
    else:
        report = export_reader_onnx(
            Path(args.checkpoint),
            Path(args.output),
            verify=not args.no_verify,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
# [onnx-export] - END


def _cmd_on_car(args: argparse.Namespace) -> int:
    """What runs on the comma, for the Models tab (nnslr_tools.model_roles.record_on_car)."""
    from nnslr_tools.model_roles import record_on_car
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    record = record_on_car(root, Path(args.models), dry_run=args.dry_run)
    print(json.dumps(record, indent=2))
    if args.dry_run:
        print("dry run: nothing written", file=sys.stderr)
    return 0


def _cmd_send_to_car(args: argparse.Namespace) -> int:
    """Steps 1-3 of Publish to the car (nnslr_tools.car_release): export, publish in the models clone, on-car.json."""
    from nnslr_tools.car_release import send_to_car
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.steps_of < 3 or args.steps_of > 9:
        raise ValueError("--steps-of is 3 to 9")
    send_to_car(root, args.signnet, classifier=args.classifier, models=Path(args.models) if args.models else None,
                dry_run=args.dry_run, steps_of=args.steps_of)
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Models › Drive: drives replayed through the car's loop with a set of networks (nnslr_tools.simulation)."""
    from nnslr_tools import simulation
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if bool(args.route) == bool(args.certified):
        raise ValueError("which drives? --route <drive> (may repeat), or --certified")
    routes = sorted(simulation.certified(root)) if args.certified else list(dict.fromkeys(args.route))
    if not routes:
        raise ValueError("no drive is certified yet: certify one on Models › Drive first")
    failed = simulation.run(root, routes, args.set, device=args.device, again=args.again,
                            log=lambda text: print(text, flush=True))
    return 1 if failed else 0


def _cmd_simulate_set(args: argparse.Namespace) -> int:
    """The release of a set of networks made ready for a replay: the versions in use exported (simulation.export_set),
    and the road type head of its brain written beside the replays (simulation.save_type_head)."""
    from nnslr_tools import simulation
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)

    def log(text: str) -> None:
        print(text, flush=True)

    if args.set == 'car':
        directory = simulation.set_directory(root, 'car')
        if simulation.release_networks(directory) is None:
            raise ValueError(f"no release of the networks in {directory}")
        log(f"the networks on the car: {simulation.release_name(directory)}")
    else:
        simulation.export_set(root, log=log)
    simulation.save_type_head(root, args.set, log=log)
    return 0


def _cmd_try_on_comma(args: argparse.Namespace) -> int:
    """Try on the comma now, or back to the build's networks (nnslr_tools.comma_trial)."""
    from nnslr_tools import comma_trial
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.restore:
        if args.signnet or args.classifier or args.dry_run:
            raise ValueError("--restore takes no SignNet, classifier or --dry-run")
        comma_trial.back_to_build(root)
    elif not args.signnet:
        raise ValueError("which SignNet? --signnet <run>, or --restore")
    else:
        comma_trial.try_on_comma(root, args.signnet, classifier=args.classifier, dry_run=args.dry_run)
    return 0


def _cmd_drive_video(args: argparse.Namespace) -> int:
    """A drive as an MP4 for the Drive view of Results (nnslr_tools.drive_video)."""
    from nnslr_tools import drive_video
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    made = None if args.again else drive_video.index(root, args.route)
    if made is None:
        made = drive_video.make(root, args.route,
                                lambda share: print(f'\r{share * 100:5.1f}%', end='', file=sys.stderr, flush=True))
        print(file=sys.stderr)
    print(json.dumps({'video': str(drive_video.paths(root, args.route)[0]), 'minutes': len(made['segments']),
                      'seconds': made['duration']}))
    return 0


# [drive-review] - START
def _cmd_drive_review(args: argparse.Namespace) -> int:
    """Read drives' comma logs on the server (nnslr_tools.drive_review), as Read the logs on Speed review."""
    from nnslr_tools import drive_logs, drive_review
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = drive_review.DriveReviewStore(root)
    routes = list(args.routes)
    if args.all:
        # The drives with logs to read: never scanned, out of date, or with logs found and not imported yet.
        for route in drive_review.all_routes(root):
            state = store.scan_state(route)['state']
            if state in ('not_scanned', 'outdated', 'failed') or (state == 'no_logs' and drive_logs.discover(root, route)):
                routes.append(route)
    if not routes:
        print('Nothing to scan: name the drives, or --all.', file=sys.stderr)
        return 0
    failed = 0
    for route in dict.fromkeys(routes):
        try:
            state = store.scan(route)
        except drive_logs.DriveLogError as exc:
            state = {'route': route, 'state': 'failed', 'error': exc.message}
        failed += state['state'] == 'failed'
        print(json.dumps({key: state.get(key) for key in ('route', 'state', 'segments', 'kinds', 'episodes', 'error')}),
              flush=True)
    return 1 if failed else 0
# [drive-review] - END


# [sign-reads] - START
def _cmd_sign_reads(args: argparse.Namespace) -> int:
    """The exact frame of every sign the comma read, added to Sign review (nnslr_tools.sign_reads)."""
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.sign_reads import extract_reads
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    for route in args.routes:
        parse_route_id(route)
        print(json.dumps(extract_reads(root, route, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, scan=not args.no_scan,
                                       workers=args.workers), sort_keys=True), flush=True)
    return 0
# [sign-reads] - END


# [localizer] - START
def _cmd_localize(args: argparse.Namespace) -> int:
    """openpilot.distill's localizer on the segments of drives (nnslr_tools.localized): the precise position of every
    road-camera frame, which nnslr gps then takes. Exit 1 when a segment failed or a drive named has no segments."""
    from nnslr_tools import localized
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    routes = list(args.routes)
    if args.all:
        routes += localized.all_routes(root)
    if not routes:
        if args.all:
            return 0
        print('nnslr localize: name the drives, or --all', file=sys.stderr)
        return 2
    try:
        localized.distill_python()
    except localized.LocalizerError as exc:
        print(f'nnslr localize: {exc}; the positions stay those of the GPS', file=sys.stderr)
        return 2
    failed = 0

    def report(summary: dict) -> None:
        nonlocal failed
        failed += bool(summary['failed_segments'] or summary.get('error'))
        print(json.dumps(summary, sort_keys=True), flush=True)

    try:
        localized.localize_routes(root, routes, workers=args.workers, again=args.again, on_route=report)
    except ValueError as exc:
        print(f'nnslr localize: {exc}', file=sys.stderr)
        return 2
    return 1 if failed else 0
# [localizer] - END


# [offline-replay] - START
def _cmd_replay(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.replay import load_route_frames, replay_bundle_route, replay_route

    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    frames, manifests = load_route_frames(root, args.route)

    if args.bundle and (args.detector_checkpoint or args.reader_checkpoint):
        raise ValueError("--bundle cannot be combined with checkpoint arguments")

    if args.dry_run:
        report = {
            "valid": True,
            "route_id": args.route,
            "frame_count": len(frames),
            "manifests": manifests,
            "gpu_required": False,
            "checkpoints_required_for_execution": not bool(args.bundle),
            "bundle": args.bundle,
        }
        if args.bundle:
            from nnslr_tools.bundle_runtime import reference_runtime_info
            report["bundle_runtime"] = reference_runtime_info(Path(args.bundle))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if frames else 1

    if not args.output:
        raise ValueError("--output is required unless --dry-run")

    if args.bundle:
        report = replay_bundle_route(
            root,
            args.route,
            Path(args.bundle),
            Path(args.output),
            detector_threshold=args.detector_threshold,
            reader_threshold=args.reader_threshold,
            max_detections=args.max_detections,
        )
    else:
        if not args.detector_checkpoint or not args.reader_checkpoint:
            raise ValueError(
                "--detector-checkpoint and --reader-checkpoint are required "
                "when --bundle is not used"
            )
        report = replay_route(
            root,
            args.route,
            Path(args.detector_checkpoint),
            Path(args.reader_checkpoint),
            Path(args.output),
            device_name=args.device,
            detector_threshold=args.detector_threshold,
            reader_threshold=args.reader_threshold,
            max_detections=args.max_detections,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
# [offline-replay] - END


# [model-bundle] - START
def _cmd_package_model(args: argparse.Namespace) -> int:
    from nnslr_tools.bundle import package_model

    report = package_model(
        Path(args.detector_onnx),
        Path(args.detector_contract),
        Path(args.reader_onnx),
        Path(args.reader_contract),
        Path(args.output),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _cmd_verify_bundle(args: argparse.Namespace) -> int:
    from nnslr_tools.bundle import verify_bundle

    report = verify_bundle(Path(args.bundle))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0
# [model-bundle] - END


# [networks-package] - START
def _cmd_networks_package(args: argparse.Namespace) -> int:
    from nnslr_tools.networks_package import new_name, prepare_package

    report = prepare_package(_resolve_data_root(args.data_root), args.name or new_name(),
                             log=lambda text: print(text, flush=True))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _cmd_verify_networks(args: argparse.Namespace) -> int:
    from nnslr_tools.networks_package import verify_package

    print(json.dumps(verify_package(Path(args.package)), indent=2, sort_keys=True, allow_nan=False))
    return 0


def _cmd_networks_catalog(args: argparse.Namespace) -> int:
    from nnslr_tools.networks_package import add_to_catalog

    print(json.dumps(add_to_catalog(Path(args.catalog), Path(args.package), args.base_url), indent=2,
                     sort_keys=True, allow_nan=False))
    return 0


def _cmd_networks_published(args: argparse.Namespace) -> int:
    from nnslr_tools.networks_package import mark_published

    print(json.dumps(mark_published(Path(args.package), args.repo, args.url), indent=2, sort_keys=True))
    return 0
# [networks-package] - END


# [core-export] - START
def _cmd_export_core(args: argparse.Namespace) -> int:
    from nnslr_tools.core_snapshot import export_core_snapshot, verify_core_snapshot

    repo_root = Path(__file__).resolve().parents[2]
    manifest, manifest_path = export_core_snapshot(repo_root, Path(args.output))
    verify_core_snapshot(Path(args.output), manifest_path)
    print(json.dumps({
        "valid": True,
        "output": str(Path(args.output)),
        "manifest": str(manifest_path),
        **manifest.to_dict(),
    }, indent=2, sort_keys=True))
    return 0
# [core-export] - END


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from nnslr_tools.training import DEFAULT_WORKERS
    parser = argparse.ArgumentParser(
        prog="nnslr",
        description=(
            "NNSLR — NN Vision Speed Limit tooling. Advisory-only project: "
            "never part of the vehicle control path. T1 ships the domain "
            "contracts, a validation command and the synthetic quickstart."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ver = sub.add_parser("version", help="print package versions")
    p_ver.set_defaults(func=lambda a: _cmd_version(a))

    p_env = sub.add_parser(
        "env", help="machine-readable environment report (GPU probe only with --gpu-smoke)"
    )
    p_env.add_argument(
        "--gpu-smoke",
        action="store_true",
        help="explicitly run a CUDA FP16 forward/backward smoke test; allocates GPU memory",
    )
    p_env.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device index for --gpu-smoke (default: 0)",
    )
    p_env.set_defaults(func=lambda a: _cmd_env(a))

    p_check = sub.add_parser(
        "check-environment",
        help="alias of env; GPU probe only with --gpu-smoke",
    )
    p_check.add_argument(
        "--gpu-smoke",
        action="store_true",
        help="explicitly run a CUDA FP16 forward/backward smoke test; allocates GPU memory",
    )
    p_check.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="CUDA device index for --gpu-smoke (default: 0)",
    )
    p_check.set_defaults(func=lambda a: _cmd_env(a))

    p_val = sub.add_parser(
        "validate-batch",
        help="validate an ObservationBatch JSON file against the §7 contract",
    )
    p_val.add_argument("batch", help="path to the batch JSON file")
    p_val.add_argument(
        "--now-mono-ns",
        type=int,
        required=True,
        help="current session-monotonic time in ns (injected, never read live)",
    )
    p_val.add_argument(
        "--expected-session-id",
        required=True,
        help="session the batch must belong to",
    )
    p_val.set_defaults(func=_cmd_validate_batch)

    p_self = sub.add_parser("selftest", help="run the bundled synthetic fixtures (CPU quickstart)")
    p_self.set_defaults(func=lambda a: _selftest())

    # [nnslr-sync] - START
    p_sync = sub.add_parser(
        "sync-routes",
        help="copy one comma route via SSH/rsync (front camera + full log) into the shared videos folder and link it into raw/routes",
    )
    p_sync.add_argument("--route", required=True, help="route ID to copy (always explicit)")
    p_sync.add_argument("--host", help="SSH alias, hostname, IPv4 or user@host (default: NNSLR_COMMA_HOST or comma)")
    p_sync.add_argument("--segments", default="all", help="all (default), 0-3, or 0,2,4-6")
    p_sync.add_argument("--camera", choices=("narrow", "wide", "both"), default="narrow",
                        help="narrow=fcamera (default), wide=ecamera, both=both road cameras")
    p_sync.add_argument("--data-root", help="root of the raw/routes symlinks (default: NNSLR_DATA_ROOT or <repo>/data)")
    p_sync.add_argument("--videos-dir", help="shared folder of the raw comma files (default: COMMA_VIDEOS_DIR or <repo>/../comma-videos); "
                        "readers only accept links into COMMA_VIDEOS_DIR, so also export COMMA_VIDEOS_DIR=<this folder>")
    p_sync.add_argument("--dry-run", action="store_true", help="inventory via SSH and print the plan without copying or writing files")
    p_sync.set_defaults(func=_cmd_sync_routes)

    p_link = sub.add_parser(
        "link-videos",
        help="link segments already in the shared videos folder into raw/routes (no network)",
    )
    p_link.add_argument("--route", help="link only this route ID (default: every route in the shared folder)")
    p_link.add_argument("--videos-dir", help="shared folder of the raw comma files (default: COMMA_VIDEOS_DIR or <repo>/../comma-videos); "
                        "readers only accept links into COMMA_VIDEOS_DIR, so also export COMMA_VIDEOS_DIR=<this folder>")
    p_link.add_argument("--data-root", help="root of the raw/routes symlinks (default: NNSLR_DATA_ROOT or <repo>/data)")
    p_link.add_argument("--dry-run", action="store_true", help="print the plan and the conflicts without creating links")
    p_link.set_defaults(func=_cmd_link_videos)
    # [nnslr-sync] - END

    p_route = sub.add_parser("route-manifest", help="inventory LOCAL route segments and preserve gaps")
    p_route.add_argument("route", help="route directory under the data root, or one local segment directory")
    p_route.add_argument("--data-root")
    p_route.add_argument("--output", required=True)
    p_route.set_defaults(func=_cmd_route_manifest)

    p_inspect = sub.add_parser("inspect-manifest", help="summarize a route manifest JSONL")
    p_inspect.add_argument("manifest")
    p_inspect.set_defaults(func=_cmd_inspect_manifest)

    p_probe = sub.add_parser("video-probe", help="probe a LOCAL camera video with ffprobe")
    p_probe.add_argument("video")
    p_probe.add_argument("--ffprobe")
    p_probe.set_defaults(func=_cmd_video_probe)

    p_extract = sub.add_parser("extract-frames", help="extract LOCAL video frames with ffmpeg")
    # [route-extract] - START
    extract_source = p_extract.add_mutually_exclusive_group(required=True)
    extract_source.add_argument("--video", help="one local video; requires --output")
    extract_source.add_argument("--route", help="automatically extract fcamera.hevc from every local segment of this route")
    p_extract.add_argument("--data-root", help="with --route: NNSLR_DATA_ROOT or <repo>/data by default")
    p_extract.add_argument("--output", help="with --route: defaults to <data-root>/derived/frames/<route>; required with --video")
    # [route-extract] - END
    p_extract.add_argument("--fps", type=float, default=1.0, help="sampling FPS (default: 1 for broad discovery)")
    p_extract.add_argument("--all-frames", action="store_true", help="decode every frame instead of sampling")
    p_extract.add_argument("--start", type=float)
    p_extract.add_argument("--end", type=float)
    p_extract.add_argument("--ffmpeg")
    # [frame-provenance] - START
    p_extract.add_argument("--ffprobe")
    p_extract.add_argument("--alignment", help="verified alignment JSONL; with --route, directory containing <segment>.jsonl")
    # [frame-provenance] - END
    # [route-extract] - START
    p_extract.add_argument("--manifest", help="optional combined frame JSONL; --route also writes frames.jsonl in each segment directory")
    # [route-extract] - END
    p_extract.add_argument("--overwrite", action="store_true")
    p_extract.add_argument("--workers", type=int, default=int(os.environ.get("NNSLR_EXTRACT_WORKERS", EXTRACT_WORKERS)),
                           help=f"with --route: segments extracted at once (NNSLR_EXTRACT_WORKERS, default {EXTRACT_WORKERS})")
    p_extract.set_defaults(func=_cmd_extract_frames)

    p_log = sub.add_parser("log-metadata", help="read LOCAL qlog/rlog camera/map metadata")
    p_log.add_argument("log")
    p_log.add_argument("--openpilot-root")
    p_log.add_argument("--openpilot-python")
    # [schema-reader] - START
    p_log.add_argument("--opendbc-root", help="compatible car.capnp checkout (default: NNSLR_OPENDBC_ROOT or embedded opendbc)")
    # [schema-reader] - END
    p_log.add_argument("--output")
    p_log.set_defaults(func=_cmd_log_metadata)

    p_align = sub.add_parser("align-route", help="align LOCAL video presentation order to EncodeIndex")
    p_align.add_argument("--video", required=True)
    p_align.add_argument("--log", required=True)
    p_align.add_argument("--stream", required=True, choices=sorted(_STREAM_TO_ENCODE_SERVICE))
    p_align.add_argument("--segment-num", required=True, type=int)
    p_align.add_argument("--output", required=True)
    p_align.add_argument("--openpilot-root")
    p_align.add_argument("--openpilot-python")
    # [schema-reader] - START
    p_align.add_argument("--opendbc-root", help="compatible car.capnp checkout (default: NNSLR_OPENDBC_ROOT or embedded opendbc)")
    # [schema-reader] - END
    p_align.add_argument("--ffprobe")
    p_align.set_defaults(func=_cmd_align_route)

    p_report = sub.add_parser("alignment-report", help="print the report header from alignment JSONL")
    p_report.add_argument("alignment")
    p_report.set_defaults(func=_cmd_alignment_report)

    p_candidates = sub.add_parser("find-candidates", help="find map transitions as search hints, never ground truth")
    p_candidates.add_argument("log")
    p_candidates.add_argument("--openpilot-root")
    p_candidates.add_argument("--openpilot-python")
    # [schema-reader] - START
    p_candidates.add_argument("--opendbc-root", help="compatible car.capnp checkout (default: NNSLR_OPENDBC_ROOT or embedded opendbc)")
    # [schema-reader] - END
    p_candidates.add_argument("--alignment")
    p_candidates.add_argument("--output")
    p_candidates.add_argument("--max-projection-error-ns", type=int, default=250_000_000)
    p_candidates.set_defaults(func=_cmd_find_candidates)

    p_clips = sub.add_parser("make-clips", help="make LOCAL review clips around projected candidates")
    p_clips.add_argument("--video", required=True)
    p_clips.add_argument("--candidates", required=True)
    p_clips.add_argument("--output", required=True)
    p_clips.add_argument("--manifest")
    p_clips.add_argument("--before", type=float, default=30.0)
    p_clips.add_argument("--after", type=float, default=30.0)
    p_clips.add_argument("--ffmpeg")
    p_clips.add_argument("--reencode", action="store_true")
    p_clips.set_defaults(func=_cmd_make_clips)

    # [preannotation] - START
    p_pre = sub.add_parser("preannotate", help="propose sign boxes/values on all extracted route frames (CPU)")
    p_pre.add_argument("--route", required=True)
    p_pre.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_pre.add_argument("--confidence", type=float, default=None,
                       help="lowest score of a box (default 0.25 for vietnam, 0.1 for ours)")
    p_pre.add_argument("--proposer", choices=("vietnam", "ours", "qwen"), default=os.environ.get("NNSLR_PROPOSER") or "vietnam",
                       help="vietnam: the Vietnamese traffic-sign YOLO of Hugging Face; ours: the cutter and the "
                            "classifier trained here; qwen: Qwen3-VL-8B (scripts/setup_vlm.sh); ours and qwen use the "
                            "GPU environment. Default: NNSLR_PROPOSER or vietnam")
    p_pre.add_argument("--keep-reviews", action="store_true",
                       help="Propose again on a route in review: the frames confirmed on its current report stay "
                            "confirmed on the new one (copied before it becomes the current one)")
    p_pre.set_defaults(func=_cmd_preannotate)
    p_review = sub.add_parser("review", help="serve the review pages; confirmed frames are saved in <data-root>/annotations/review.sqlite")
    p_review.add_argument("--route", help="open this route directly; default: the index of every route with a report")
    p_review.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_review.add_argument("--host", default="127.0.0.1",
                          help="listen address; default: local only. 0.0.0.0 opens it to the LAN and requires the "
                               "access token (NNSLR_REVIEW_TOKEN or a random one, printed at start)")
    p_review.add_argument("--port", type=int, default=8765)
    p_review.set_defaults(func=_cmd_review)
    p_box = sub.add_parser("review-box", help="the machine of the outside reviewers: drives go there without their places, "
                                              "what people do comes back (nnslr_tools.review_box)")
    box_sub = p_box.add_subparsers(dest="box_command", required=True)
    p_box_link = box_sub.add_parser("link", help="on this server: remember the machine and its host key")
    p_box_link.add_argument("remote", help="account@address of the copy account")
    p_box_link.add_argument("fingerprint", help="SHA256:... of its ed25519 host key, from its administrator")
    p_box_sync = box_sub.add_parser("sync", help="on this server: results here, then the drives and the work it should hold")
    p_box_tick = box_sub.add_parser("tick", help="on the machine: install what arrived, write what people did; a timer runs it")
    p_box_tick.add_argument("--box", required=True, help="the folder of the copy account, with inbox/ and outbox/")
    p_box_status = box_sub.add_parser("status", help="what is on the machine, as Data › Reviewers shows it")
    for item in (p_box_link, p_box_sync, p_box_tick, p_box_status):
        item.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
        item.set_defaults(func=_cmd_review_box)
    p_reclassify = sub.add_parser(
        "review-reclassify-ai",
        help="mark the reviews saved by the named reviewers (AI agents that used the review page, e.g. Codex) "
             "as AI reviews: new revision, reviewer ai/<name>, imported as AI reviews (label_basis ai_review), "
             "never as human ones")
    p_reclassify.add_argument("--reviewer", action="append", required=True,
                              help="reviewer name exactly as stored, e.g. Codex; repeat for more names")
    p_reclassify.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_reclassify.add_argument("--dry-run", action="store_true", help="only count what would change")
    p_reclassify.set_defaults(func=_cmd_review_reclassify)
    p_withdraw_ai = sub.add_parser(
        "review-withdraw-ai",
        help="mark every frame an AI confirmed (reviewer ai/<name>) as not confirmed: the boxes stay, a new revision "
             "goes in the history, the next import takes the frames out of the dataset; a person reviews them again. "
             "A copy of the database is made first")
    p_withdraw_ai.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_withdraw_ai.add_argument("--dry-run", action="store_true", help="only count what would change")
    p_withdraw_ai.set_defaults(func=_cmd_review_withdraw_ai)
    p_mcp =sub.add_parser("mcp", help="MCP server (stdio) with the review tools, for Claude Code, Codex and other "
                                       "MCP clients; needs the optional [mcp] dependencies. The review server "
                                       "also serves MCP over HTTP at /mcp")
    p_mcp.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_mcp.add_argument("--read-only", action="store_true",
                       help="read-only tools: save_frame_review is not offered (also NNSLR_MCP_READ_ONLY=1)")
    p_mcp.set_defaults(func=_cmd_mcp)
    p_gps = sub.add_parser("gps", help="GPS position and country of every frame of a route, from its comma logs "
                                       "(derived/gps/<route>/positions.jsonl); the review proposes that country")
    p_gps.add_argument("--route", action="append", help="route id; repeat for more")
    p_gps.add_argument("--all", action="store_true", help="every route with a pre-annotation report")
    p_gps.add_argument("--missing", action="store_true",
                       help="with --all: only routes without positions yet, or whose positions are older than a "
                            "localizer track of theirs (nnslr localize)")
    p_gps.add_argument("--countries", action="store_true",
                       help="only the country of each frame again, from the positions already read and the borders "
                            "now (after scripts/make_country_borders.py --gisco); the logs are not read")
    p_gps.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_gps.set_defaults(func=_cmd_gps)
    p_road = sub.add_parser("road-types", help="the map road of every frame, its limit and its type (town, rural, "
                                               "expressway, motorway, merge lane, parking, garage...) from its GPS "
                                               "position and the OpenStreetMap roads, derived/road_types/<route>.jsonl: "
                                               "the map's limit Speed review proposes (Read the map)")
    p_road.add_argument("--route", action="append", help="route id; repeat for more")
    p_road.add_argument("--all", action="store_true", help="every route with GPS positions (nnslr gps)")
    p_road.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_road.set_defaults(func=_cmd_road_types)
    p_tunnels = sub.add_parser("tunnels", help="the tunnels of the scanned drives from the map and the comma's GPS, "
                                               "written as marks of the kind tunnel in Speed review where the map is "
                                               "sure (reviewer osm; a person's marks win)")
    p_tunnels.add_argument("--route", action="append", help="route id; repeat for more")
    p_tunnels.add_argument("--all", action="store_true", help="every scanned drive")
    p_tunnels.add_argument("--dry-run", action="store_true", help="print the tunnels found, write nothing")
    p_tunnels.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_tunnels.set_defaults(func=_cmd_tunnels)
    p_brain = sub.add_parser("brain-targets", help="what the brain's training gets today from every dashcam drive "
                                                   "(sunnypilot off) with its logs read and its frames, each whole: "
                                                   "per partition and per drive the frames "
                                                   "with a speed (the driver's own, or a block's of Speed review), "
                                                   "and with an answer on roadworks, "
                                                   "and why the others are out. Reads only: it writes nothing and "
                                                   "trains nothing")
    p_brain.add_argument("--route", action="append", help="route id; repeat for more. Default: every drive the brain "
                                                          "learns from (every dashcam drive with its logs read to the "
                                                          "end and its frames)")
    p_brain.add_argument("--json", action="store_true", help="the same as JSON")
    p_brain.add_argument("--no-road-types", action="store_true",
                         help="as a training with --no-road-types: the seconds without a road type answer too")
    p_brain.add_argument("--sign-networks", default=None,
                         help="a run's networks.json (nnslr brain-networks): its drives read by those networks. "
                              "Default: the car's release of the sunnypilot clone and Our networks in use")
    p_brain.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_brain.set_defaults(func=_cmd_brain_targets)
    p_brain_networks = sub.add_parser("brain-networks", help="scripts/run_brain.sh, step 3: the networks the brain's "
                                                             "signs are read with, for its inputs and its labels: the "
                                                             "cutter (a SignNet the comma runs), the classifier and the "
                                                             "applicability network chosen in Train › Brain, those in use "
                                                             "unless others are given. Exported as Publish exports them "
                                                             "(PyTorch to ONNX, checked; .venv-gpu) into "
                                                             "derived/road_type_signs/sets/<key>/ and written into "
                                                             "--run-dir/networks.json; a run started again keeps its own")
    p_brain_networks.add_argument("--run-dir", required=True, help="the folder of the brain's run")
    p_brain_networks.add_argument("--cutter", default=None, help="a SignNet's id of the Models tab (default: in use)")
    p_brain_networks.add_argument("--classifier", default=None, help="a classifier's id (default: in use)")
    p_brain_networks.add_argument("--decider", default=None,
                                  help="an applicability network's id (default: in use; one is required)")
    p_brain_networks.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_brain_networks.set_defaults(func=_cmd_brain_networks)
    p_borrow = sub.add_parser("borrow-cards", help="scripts/run_brain.sh, step 3 (#36): prints the cards its reading "
                                                   "of the signs runs on (0,1), --own first, then every other card no "
                                                   "compute process is on and no running job of the panel holds. In "
                                                   "a job of the panel (NNSLR_JOB_RECORD) they are written to its "
                                                   "record, so the panel names them as the brain's and starts nothing "
                                                   "else on them until --release")
    p_borrow.add_argument("--own", type=int, help="the brain's own card, by nvidia-smi's number")
    p_borrow.add_argument("--cards", type=_cards, default=None,
                          help="these cards instead of the free ones (NNSLR_SIGN_GPUS), written as they are")
    p_borrow.add_argument("--release", action="store_true", help="give the borrowed cards back")
    p_borrow.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_borrow.set_defaults(func=_cmd_borrow_cards)
    p_regression_plan = sub.add_parser("regression-plan", help="what a run of Models › Regression still has to compute: a "
                                                               "line per certified drive, its route and the driving "
                                                               "models missing there. Reads only")
    p_regression_plan.add_argument("--openpilot", default="", help="the specs chosen, separated by commas")
    p_regression_plan.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_regression_plan.set_defaults(func=_cmd_regression_plan)
    p_regression_brain = sub.add_parser("regression-brain", help="the brains chosen along the certified drives that are "
                                                                 "prepared, as on the car, where they have no result of "
                                                                 "today's reader of the signs (GPU, .venv-gpu)")
    p_regression_brain.add_argument("--brains", required=True, help="the runs of the brains, separated by commas")
    p_regression_brain.add_argument("--device", default="cuda", help="cuda or cpu")
    p_regression_brain.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_regression_brain.set_defaults(func=_cmd_regression_brain)
    p_regression_networks = sub.add_parser("regression-sign-networks", help="scripts/run_regression.sh, step 2: the "
                                                                          "networks each brain's signs were read with "
                                                                          "(#31), one line per choice: '-' for the "
                                                                          "car's release of the sunnypilot clone (a "
                                                                          "brain from before 10 Nov 2026), else a "
                                                                          "networks.json for road-type-signs "
                                                                          "--sign-networks")
    p_regression_networks.add_argument("--brains", required=True, help="the runs of the brains, separated by commas")
    p_regression_networks.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_regression_networks.set_defaults(func=_cmd_regression_sign_networks)
    p_brain_net = sub.add_parser("train-brain", help="train the brain on every dashcam drive with its logs read, "
                                                     "each whole (GPU, "
                                                     ".venv-gpu): the speed to hold, the driver's own corrected by the "
                                                     "blocks of Speed review, with the roadworks and the kinds of road "
                                                     "as tasks of the same network, "
                                                     "measured on "
                                                     "sites it never learned. Writes brain-last.pt and result.json "
                                                     "into --run-dir")
    p_brain_net.add_argument("--run-dir", required=True, help="the folder of this run")
    p_brain_net.add_argument("--route", action="append",
                             help="route id; repeat for more: only these drives, those still dashcam drives with their "
                                  "logs read (scripts/run_brain.sh passes the drives of its start to every seed). Default: every "
                                  "drive the brain learns from")
    p_brain_net.add_argument("--no-works", action="store_true", help="the comparison: the same training without the roadworks")
    p_brain_net.add_argument("--no-road", action="store_true",
                             help="the comparison: the same training without the kinds of road of Speed review")
    p_brain_net.add_argument("--no-road-types", action="store_true",
                             help="the comparison: without the road type of the map and Speed review, and with the "
                                  "seconds that have none")
    p_brain_net.add_argument("--no-users", action="store_true",
                             help="the comparison: without users_head, the road users (nnslr road-users) as a task of "
                                  "the network; they are still measured (users_speed)")
    p_brain_net.add_argument("--pool", choices=("mean", "grid"), default="grid",
                             help="grid keeps where things are in the frame, mean only what is there")
    p_brain_net.add_argument("--epochs", type=int, default=None)
    p_brain_net.add_argument("--seed", type=int, default=0)
    p_brain_net.add_argument("--init-from", default=None,
                             help="an earlier brain's brain-last.pt to go on from, stepped as this one and on the same "
                                  "split (scripts/run_brain.sh: NNSLR_BRAIN_INIT_FROM). Default: ImageNet and random weights")
    p_brain_net.add_argument("--sign-networks", default=None,
                             help="a run's networks.json (nnslr brain-networks; scripts/run_brain.sh): the signs of "
                                  "its inputs and labels read by those networks. Default: the car's release of the "
                                  "sunnypilot clone and Our networks in use")
    p_brain_net.add_argument("--device", help="default: cuda; the CPU for a dry run")
    p_brain_net.add_argument("--dry-run", action="store_true", help="two steps on the CPU on a few frames: the chain works, "
                                                                    "nothing is learned, nothing written outside --run-dir")
    p_brain_net.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_brain_net.set_defaults(func=_cmd_train_brain)
    p_labeler = sub.add_parser("road-labeler-train", help="train the road labeler, an offline network that prepares "
                                                          "the labels of Speed review (road type, kinds of road, "
                                                          "roadworks) from the camera at high definition, the map, the "
                                                          "GPS and the speed; never on the car (GPU, .venv-gpu). Writes "
                                                          "road-labeler.pt and result.json into --run-dir")
    p_labeler.add_argument("--run-dir", required=True, help="the folder of this run")
    p_labeler.add_argument("--route", action="append", help="route id; repeat for more. Default: every scanned drive "
                                                            "with its frames")
    p_labeler.add_argument("--epochs", type=int, default=None)
    p_labeler.add_argument("--seed", type=int, default=0)
    p_labeler.add_argument("--device", help="default: cuda; the CPU for a dry run")
    p_labeler.add_argument("--dry-run", action="store_true", help="a couple of steps on the CPU with a tiny encoder on "
                                                                  "the start of one or two drives: the chain works, "
                                                                  "nothing is learned, nothing written outside --run-dir")
    p_labeler.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_labeler.set_defaults(func=_cmd_road_labeler_train)
    p_labeler_predict = sub.add_parser("road-labeler-predict", help="the road labeler of a run along drives: "
                                                                    "derived/road_labeler/<route>.jsonl (its chances), "
                                                                    "then Prepare Speed review where it is sure: the "
                                                                    "marks of the kinds of road and of the roadworks "
                                                                    "(reviewer net/<run>) and its layer of road types, "
                                                                    "definitive, only of the heads that passed the "
                                                                    "gate of validation; it replaces an older run's, "
                                                                    "and a person's marks, removals and checks win")
    p_labeler_predict.add_argument("--run-dir", required=True, help="the folder of the run with road-labeler.pt")
    p_labeler_predict.add_argument("--route", action="append", help="route id; repeat for more")
    p_labeler_predict.add_argument("--all", action="store_true", help="every scanned drive with its frames")
    p_labeler_predict.add_argument("--no-marks", action="store_true", help="the chances only: nothing written into "
                                                                           "Speed review")
    p_labeler_predict.add_argument("--out-dir", help="default: <data root>/derived/road_labeler")
    p_labeler_predict.add_argument("--device", help="default: cuda when there is one")
    p_labeler_predict.add_argument("--dry-run", action="store_true", help="the start of the drives on the CPU, the "
                                                                          "chances into <run>/chances, nothing "
                                                                          "written into Speed review")
    p_labeler_predict.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_labeler_predict.set_defaults(func=_cmd_road_labeler_predict)
    p_brain_export = sub.add_parser("export-brain", help="the brain of a run as the car runs it: one step a second as ONNX with "
                                                         "its contract, checked against PyTorch and along a real drive (CPU, "
                                                         ".venv-gpu). Writes <run>/export/brain.onnx")
    p_brain_export.add_argument("--run-dir", required=True, help="the folder of the run with brain-last.pt")
    p_brain_export.add_argument("--route", help="the drive of the check, any dashcam drive with the brain's pictures; "
                                                "default: the one with the most frames in test among the dashcam "
                                                "drives")
    p_brain_export.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_brain_export.set_defaults(func=_cmd_export_brain)
    p_junctions = sub.add_parser("junctions", help="the intersections the car drives through, from its GPS track and "
                                                   "the OpenStreetMap roads: derived/junctions/<route>.json (in the "
                                                   "intersection, and the first frame after it); no training reads "
                                                   "them today")
    p_junctions.add_argument("--route", action="append", help="route id; repeat for more")
    p_junctions.add_argument("--all", action="store_true", help="every route with GPS positions (nnslr gps)")
    p_junctions.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_junctions.set_defaults(func=_cmd_junctions)
    from nnslr_tools import signnet as signnet_settings
    car_widths = ",".join(str(value) for value in signnet_settings.CAR_WIDTHS)
    p_signnet = sub.add_parser("train-signnet", help="train SignNet, the cutter: the sign detector the comma runs on "
                                                     "its Adreno, on the cutter's plan (GPU, .venv-gpu); writes "
                                                     "signnet-best.pt and result.json with the cutter's test measures. "
                                                     "The defaults are the car's SignNet")
    p_signnet.add_argument("--run-dir", required=True, help="output folder, e.g. data/runs/<time>-cutter")
    p_signnet.add_argument("--dataset", default=None, help="default: <data root>/annotations/objects.jsonl")
    p_signnet.add_argument("--split", default=None, help="default: <data root>/splits/latest.json")
    p_signnet.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
    p_signnet.add_argument("--families", default="all", choices=("all", "speed"), help="speed or all, as the cutter")
    p_signnet.add_argument("--mapillary", default=None,
                           help="add the train windows of the Mapillary Traffic Sign Dataset (nnslr mapillary-windows: "
                                "<data root>/external/mapillary/windows.jsonl) to our train frames; they are the car's "
                                "band, so --band-width 896 or a width with the same rows")
    p_signnet.add_argument("--mapillary-region", default="all",
                           help="the Mapillary windows of these continents (europe, north_america...) or countries "
                                "(CH, DE...), comma-separated, by the photos' positions (nnslr mapillary-locations); "
                                "default all")
    p_signnet.add_argument("--mapillary-share", type=float, default=100.0,
                           help="the percentage of the Mapillary windows taken, the same each time, after the region "
                                "(more than 0, at most 100; default 100)")
    p_signnet.add_argument("--mapillary-ratio", type=float, default=None,
                           help="the Mapillary speed boxes at most this many times our train speed boxes (every speed "
                                "family, unreadable, zone and road marking too), whole photos in a fixed order "
                                "(default 1: as many as ours); not with --mapillary-only")
    p_signnet.add_argument("--mapillary-only", action="store_true",
                           help="train on the Mapillary windows alone, none of our train frames (the first stage of "
                                "a two-stage cutter); validation and test stay ours")
    p_signnet.add_argument("--stride", type=int, default=signnet_settings.CAR_STRIDE, choices=(4, 8),
                           help=f"the heat map's stride (default {signnet_settings.CAR_STRIDE}, the car's)")
    p_signnet.add_argument("--fuse-kernel", type=int, default=signnet_settings.CAR_FUSE_KERNEL, choices=(1, 3),
                           help="the convolution that joins the deep map with the skip: 3, or 1 (faster on the "
                                f"Adreno; default {signnet_settings.CAR_FUSE_KERNEL}, the car's)")
    p_signnet.add_argument("--widths", default=car_widths,
                           help=f"channels at strides 2,4,8,16 (default {car_widths}, the car's)")
    p_signnet.add_argument("--band-width", type=int, default=signnet_settings.CAR_BAND[0],
                           help="the band's width in pixels: 1344 full resolution, 896 two thirds (default, the "
                                "car's: only 896 with stride 8 runs on the comma), 672 half")
    p_signnet.add_argument("--epochs", type=int, default=40)
    p_signnet.add_argument("--batch", type=int, default=16)
    p_signnet.add_argument("--learning-rate", type=float, default=3e-3)
    p_signnet.add_argument("--workers", type=int, default=8)
    p_signnet.add_argument("--init-from", default=None,
                           help="a finished SignNet's signnet-best.pt to go on from (the same widths, grid, fuse and "
                                "band), instead of random weights")
    p_signnet.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_signnet.add_argument("--dry-run", action="store_true",
                           help="build the plan only (no torch): one JSON line with its frames and boxes by split")
    p_signnet.add_argument("--plan-out", default=None,
                           help="with --dry-run: also write the plan there, for the training's --plan")
    p_signnet.add_argument("--plan", default=None,
                           help="the plan a --dry-run --plan-out wrote, used instead of building it again when the "
                                "dataset, split, Mapillary windows and settings are still the same (else built)")
    p_signnet.set_defaults(func=_cmd_train_signnet)
    p_usernet = sub.add_parser("train-usernet", help="train UserNet, the road users' detector (person, two-wheeler, "
                                                     "vehicle) on the window 672x256 of the frame (rows 72-328 of it "
                                                     "scaled to 672 across): stage bdd on BDD100K's pictures, stage "
                                                     "drives on our reviewed frames. GPU only (.venv-gpu); writes "
                                                     "usernet-best.pt and result.json")
    p_usernet.add_argument("--run-dir", required=True, help="output folder, e.g. data/runs/<time>-users")
    p_usernet.add_argument("--stage", required=True, choices=("bdd", "drives"),
                           help="bdd: BDD100K's pictures; drives: our frames (validation: our ticked frames)")
    p_usernet.add_argument("--bdd-dir", default=None,
                           help="the BDD100K FiftyOne export with samples.json (default: NNSLR_BDD_DIR or "
                                "/data/sunnypilot/SPEEDLIMITS/bdd100k)")
    p_usernet.add_argument("--dataset", default=None, help="default: <data root>/annotations/objects.jsonl")
    p_usernet.add_argument("--split", default=None, help="default: <data root>/splits/latest.json")
    p_usernet.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
    p_usernet.add_argument("--epochs", type=int, default=30)
    p_usernet.add_argument("--batch", type=int, default=32)
    p_usernet.add_argument("--learning-rate", type=float, default=3e-3)
    p_usernet.add_argument("--workers", type=int, default=8)
    p_usernet.add_argument("--init-from", default=None,
                           help="a finished UserNet's usernet-best.pt to go on from (the drives stage from a bdd one)")
    p_usernet.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_usernet.add_argument("--dry-run", action="store_true",
                           help="build the plan only (no torch, no picture read): one JSON line with its frames and "
                                "users by split; exit 1 when it cannot train")
    p_usernet.add_argument("--plan-out", default=None,
                           help="with --dry-run: also write the plan there, for the training's --plan")
    p_usernet.add_argument("--plan", default=None,
                           help="the plan a --dry-run --plan-out wrote, used when the data and settings are still the "
                                "same (else built again)")
    p_usernet.set_defaults(func=_cmd_train_usernet)
    p_remeasure = sub.add_parser("remeasure-signnet", help="add the lenient car measure (a box's centre inside the "
                                                           "sign's, 0.5x to 2x its size) to a finished SignNet run's "
                                                           "result.json, without training; needs torch (.venv-gpu)")
    p_remeasure.add_argument("--run-dir", required=True, help="the run, e.g. data/runs/<time>-cutter")
    p_remeasure.add_argument("--dataset", default=None, help="default: <data root>/annotations/objects.jsonl")
    p_remeasure.add_argument("--split", default=None,
                             help="default: <data root>/splits/<the run's split_sha256>.json, else splits/latest.json")
    p_remeasure.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
    p_remeasure.add_argument("--batch", type=int, default=16)
    p_remeasure.add_argument("--workers", type=int, default=8)
    p_remeasure.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                             help="the card; without one it stops, never the CPU on its own")
    p_remeasure.add_argument("--dry-run", action="store_true", help="measure and print, write nothing")
    p_remeasure.add_argument("--data-root", default=None)
    p_remeasure.set_defaults(func=_cmd_remeasure_signnet)
    p_box_fit = sub.add_parser("train-box-fit", help="train the review page's local box fit (Fit box, key C) on the "
                                                     "boxes people drew: the colour trees and the edge MLPs (GPU, "
                                                     "with the PyTorch of .venv-gpu). "
                                                     "Measures it first on routes left out (--folds), then writes "
                                                     "colour.npz and edges.npz trained on every box")
    p_box_fit.add_argument("--out", default=None,
                           help="folder for colour.npz and edges.npz (default: the package's own, "
                                "src/nnslr_tools/box_fit, which the review site loads)")
    p_box_fit.add_argument("--dataset", default=None, help="default: <data root>/annotations/objects.jsonl")
    p_box_fit.add_argument("--folds", type=int, default=4,
                           help="route groups of the honest measure before the final training (default 4; 0 or 1: "
                                "no measure)")
    p_box_fit.add_argument("--seed", type=int, default=0, help="seed of the trees and the MLPs (default 0)")
    p_box_fit.add_argument("--workers", type=int, default=None,
                           help="processes extracting the candidates (default: half the cores, at most 8)")
    p_box_fit.add_argument("--device", default="cuda",
                           help="the card the trees and the MLPs train on (default cuda: the one CUDA_VISIBLE_DEVICES "
                                "shows); never the CPU")
    p_box_fit.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_box_fit.add_argument("--dry-run", action="store_true",
                           help="only count the boxes, cases and route groups: one JSON line, nothing written")
    p_box_fit.set_defaults(func=_cmd_train_box_fit)
    p_road_signs = sub.add_parser("road-type-signs", help="read the signs of the drives as the comma does (its "
                                                          "vision.py and applicability.py, SignNet, classifier and "
                                                          "applicability network from the sunnypilot clone or "
                                                          "--sign-networks, a sign of another road left out, "
                                                          "4 frames a second of fcamera.hevc): "
                                                          "derived/road_type_signs/<route>.json, the signs the brain is "
                                                          "given (the GPU when there is one)")
    p_road_signs.add_argument("--sunnypilot", help="the sunnypilot clone (default: NNSLR_SUNNYPILOT_CLONE or "
                                                   "/root/sunnypilot); read only")
    p_road_signs.add_argument("--num-workers", "--workers", dest="workers", type=int, default=0,
                              help="processes (default: half the cores)")
    p_road_signs.add_argument("--classifier", default=None,
                              help="a v2 classifier (reader-best.pt) to read with instead of the car's, e.g. one not "
                                   "on the car yet that reads the green boards: exported once to ONNX (checked against "
                                   "PyTorch, .venv-gpu) into derived/road_type_signs/classifiers/")
    p_road_signs.add_argument("--device", choices=("auto", "cpu"), default="auto",
                              help="auto: the video decoded by NVDEC and the networks on the GPU when they work there, "
                                   "else the CPU; cpu: always the CPU")
    p_road_signs.add_argument("--gpus", type=_cards, default=None,
                              help="the cards the processes run on, in turn, by nvidia-smi's numbers: 0,1 (default: "
                                   "every process on the first card this process sees). The NVDEC of one V100 is what "
                                   "holds the reading back")
    p_road_signs.add_argument("--route", action="append",
                              help="route id; repeat for more: only these drives, instead of every one of the list "
                                   "chosen (with --labelled: scripts/run_brain.sh passes the drives of its start; "
                                   "--certified always takes its own)")
    p_road_signs.add_argument("--labelled", action="store_true",
                              help="every drive the brain learns from (every dashcam drive with its logs read to "
                                   "the end and its frames) "
                                   "instead of those with road types, with the brain's pictures: what the brain "
                                   "needs (scripts/run_brain.sh)")
    p_road_signs.add_argument("--certified", action="store_true",
                              help="the drives certified in Models › Regression, those prepared, with the brain's "
                                   "pictures: what its brains need (scripts/run_regression.sh)")
    p_road_signs.add_argument("--sign-networks", default=None,
                              help="a run's networks.json (nnslr brain-networks; scripts/run_brain.sh): read with the "
                                   "release exported there (the cutter, classifier and applicability network chosen "
                                   "for the brain) instead of the clone's, the car's speed from each drive's logs")
    p_road_signs.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_road_signs.set_defaults(func=_cmd_road_type_signs)
    p_road_users = sub.add_parser("road-users", help="the road users of the brain's training (nnslr_tools.road_users): "
                                                     "an off-the-shelf COCO detector (torchvision Faster R-CNN) on "
                                                     "the road video at the brain's 4 readings a second, decoded by "
                                                     "NVDEC, on the V100s only (never the CPU), its boxes kept in "
                                                     "derived/road_users/<route>/<segment>.npz; then the crossing "
                                                     "rule proposes the vehicles that cross our path "
                                                     "(proposals.json, for a person's review); --check: the "
                                                     "detector against the frames ticked road users complete")
    p_road_users.add_argument("--route", action="append",
                              help="route id; repeat for more (default: every drive the brain learns from, "
                                   "brain_targets.training_routes)")
    p_road_users.add_argument("--gpus", type=_cards, default=None,
                              help="the cards, one process each, by nvidia-smi's numbers: 0,1 (default: the first "
                                   "card this process sees, NNSLR_GPU_INDEX or CUDA_VISIBLE_DEVICES)")
    p_road_users.add_argument("--batch", type=int, default=None, help="frames a batch on the card (default 4)")
    p_road_users.add_argument("--proposals-only", action="store_true",
                              help="no detector: only make the proposals again from the readings there (no card)")
    p_road_users.add_argument("--check", action="store_true",
                              help="no detector: the readings against a person's frames ticked road users complete, "
                                   "found and false boxes by kind and height (24-48, 48-96, 96+ px)")
    p_road_users.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_road_users.set_defaults(func=_cmd_road_users)
    p_motion = sub.add_parser("car-motion", help="the car's own motion of every second of the drives with GPS "
                                                 "positions or road types, from their rlogs (deviceMotion/livePose yaw rate, carState "
                                                 "steering angle, blinkers and vEgo): derived/car_motion/<route>.json, "
                                                 "what nnslr road-types reads to tell a garage (CPU; never the map)")
    p_motion.add_argument("--route", action="append", help="a drive (repeatable)")
    p_motion.add_argument("--all", action="store_true", help="every drive with GPS positions or road types")
    p_motion.add_argument("--again", action="store_true", help="read the logs again even when cached")
    p_motion.add_argument("--num-workers", "--workers", dest="workers", type=int, default=0,
                          help="log readers at a time (default: a quarter of the cores)")
    p_motion.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_motion.set_defaults(func=_cmd_car_motion)
    p_mapillary = sub.add_parser("mapillary-crops", help="cut the signs of the Mapillary Traffic Sign Dataset that are "
                                                         "ours into <data root>/external/mapillary/crops and "
                                                         "crops.jsonl: extra train crops of the classifier "
                                                         "(nnslr train --mapillary)")
    p_mapillary.add_argument("--annotations", required=True, help="MTSD's folder of <key>.json annotations, loose or in "
                                                               "its .zip archives (subfolders searched)")
    p_mapillary.add_argument("--images", required=True, help="MTSD's folder of <key>.jpg photos, loose or in its .zip "
                                                          "archives (may be the annotations folder)")
    p_mapillary.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_mapillary.set_defaults(func=_cmd_mapillary_crops)
    p_windows = sub.add_parser("mapillary-windows", help="cut windows of the fully annotated Mapillary Traffic Sign "
                                                         "Dataset that look like our camera's band (1344x432, a speed "
                                                         "sign at our signs' sizes and places) into <data root>/"
                                                         "external/mapillary/windows and windows.jsonl: extra train "
                                                         "frames of the cutter (nnslr train-signnet --mapillary); CPU "
                                                         "only, no network")
    p_windows.add_argument("--mtsd", default=None,
                           help="the folder of MTSD's .zip archives as downloaded, read where they are (default "
                                "/data/mapillary/zips); only the fully annotated ones are opened")
    p_windows.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_windows.add_argument("--workers", type=int, default=None,
                           help="processes that cut the photos (default: the cores, at most 8)")
    p_windows.add_argument("--limit", type=int, default=None,
                           help="only the first N photos with a window of each split (a trial cut)")
    p_windows.add_argument("--with-val", action="store_true",
                           help="also cut MTSD's val split (default train only: the cutter is measured on ours)")
    p_windows.add_argument("--other-windows", type=int, default=0,
                           help="windows per photo around another sign (default 0: the windows add speed signs only)")
    p_windows.add_argument("--dry-run", action="store_true",
                           help="read the annotation archive only and print what a cut would give; write nothing")
    p_windows.set_defaults(func=_cmd_mapillary_windows)
    p_places = sub.add_parser("mapillary-locations", help="fetch from Mapillary where each photo of the Mapillary "
                                                          "Traffic Sign Dataset was taken, with its country and "
                                                          "continent, into <data root>/external/mapillary/"
                                                          "locations.jsonl (the region of nnslr train "
                                                          "--mapillary-region); resumes where it stopped")
    p_places.add_argument("--ids", help="the table of MTSD's old image keys and their ids of today (CSV image_key,"
                                        "image_id); default: <data root>/external/mapillary/image_ids.csv, downloaded "
                                        "from --ids-url when missing")
    p_places.add_argument("--ids-url", help="where the table is downloaded from (default: the table of the Mapillary "
                                            "forum, nnslr_tools.mapillary.IMAGE_IDS_URL)")
    p_places.add_argument("--again", action="store_true", help="ask again for the photos that already have a line")
    p_places.add_argument("--limit", type=int, help="at most this many photos (a test with a handful)")
    p_places.add_argument("--dry-run", action="store_true", help="only count what would be asked; nothing is "
                                                                 "downloaded or written")
    p_places.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_places.set_defaults(func=_cmd_mapillary_locations)
    p_tracks = sub.add_parser("tracks", help="the same sign followed across consecutive confirmed frames "
                                             "(derived/tracks/<route>/tracks.jsonl), with the matching of Apply previous")
    p_tracks.add_argument("--route", action="append", help="route id; repeat for more")
    p_tracks.add_argument("--all", action="store_true", help="every route with a pre-annotation report")
    p_tracks.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_tracks.set_defaults(func=_cmd_tracks)
    # [preannotation] - END

    # [reviewed-dataset] - START
    p_import = sub.add_parser("import-annotations", help="import explicitly reviewed annotations into the canonical dataset")
    p_import.add_argument("inputs", nargs="*", help="review JSONL files; default: <data-root>/annotations/inbox/*.jsonl")
    p_import.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    # [model-review] - START
    p_import.add_argument("--output", help="default: gold <data-root>/annotations/objects.jsonl; training-candidate <data-root>/datasets/training_candidate_dataset/objects.jsonl")
    # [model-review] - END
    p_import.add_argument("--from-db", action="store_true",
                          help="import the confirmed frames saved by 'nnslr review' in <data-root>/annotations/review.sqlite "
                               "(current report of each route; frames already imported are skipped, frames reviewed "
                               "again replace their old rows, withdrawn frames leave the dataset; the rows taken out "
                               "are archived in <data-root>/annotations/replaced/). Human and AI "
                               "confirmations are both imported; AI rows keep reviewer ai/<name> and label_basis ai_review")
    p_import.add_argument("--route", help="with --from-db: only this route")
    p_import.add_argument("--dry-run", action="store_true",
                          help="with --from-db: report what would be new, replaced and removed (validation included) "
                               "without writing anything")
    p_import.set_defaults(func=_cmd_dataset)
    p_validate = sub.add_parser("validate-dataset", help="validate reviewed labels, images, hashes and capture provenance")
    # [model-review] - START
    p_validate.add_argument("dataset", nargs="?", help="default: gold <data-root>/annotations/objects.jsonl; training-candidate <data-root>/datasets/training_candidate_dataset/objects.jsonl")
    # [model-review] - END
    p_validate.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_validate.add_argument("--splits", help="also check a frozen split for leakage and dataset identity")
    p_validate.set_defaults(func=_cmd_dataset)
    p_splits = sub.add_parser("build-splits", help="freeze deterministic route/site-connected train/validation/test partitions")
    # [model-review] - START
    p_splits.add_argument("dataset", nargs="?", help="default: gold <data-root>/annotations/objects.jsonl; training-candidate <data-root>/datasets/training_candidate_dataset/objects.jsonl")
    # [model-review] - END
    p_splits.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_splits.add_argument("--seed", type=int, default=0)
    p_splits.add_argument("--test-route", action="append", default=[], metavar="ROUTE",
                          help="put this route (with its connected group) in test; repeatable. Without it the "
                               "test takes the groups with the most frames confirmed by a person")
    p_splits.add_argument("--output", help="default: <data-root>/splits/<sha256>.json (immutable)")
    p_splits.set_defaults(func=_cmd_dataset)
    # [reviewed-dataset] - END
    # [model-review] - START
    for _parser in (p_import, p_validate, p_splits):
        _parser.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold",
                             help="gold: confirmed reviews, by a person or by an AI reviewer (REST API, MCP); "
                                  "training-candidate: those + model review (default: gold)")
    # [model-review] - END
    # [training-baseline] - START
    p_train = sub.add_parser(
        "train",
        help="train reader or detector baselines; use --dry-run for a CPU-only plan",
    )
    p_train.add_argument("dataset", nargs="?", help="canonical dataset JSONL; defaults from --dataset-kind")
    p_train.add_argument("--task", choices=("reader", "detector"), default="reader",
                         help="baseline task to train (default: reader)")
    p_train.add_argument("--mode", choices=("fresh", "continue", "resume"), default="fresh",
                         help="fresh: new weights; continue: HF stable weights; resume: interrupted local run")
    p_train.add_argument("--hf-repo", default=os.environ.get("NNSLR_HF_REPO", "cristianku/sunnypilot-speedlimits-models"),
                         help="Hugging Face training-weight repository (continue only)")
    p_train.add_argument("--hf-revision", default="main",
                         help="Hugging Face revision for training/stable.json")
    p_train.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_train.add_argument("--splits", help="frozen split JSON; default: <data-root>/splits/latest.json")
    p_train.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
    p_train.add_argument("--dry-run", action="store_true", help="validate and print the exact plan without importing torch or touching a GPU")
    p_train.add_argument("--output", help="new output directory; required unless --dry-run")
    p_train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--patience", type=int, default=5,
                         help="stop when the validation has not improved for this many epochs (0: never); "
                              "the best epoch is the one kept either way")
    p_train.add_argument("--batch-size", type=int, default=None,
                         help="default: reader 64, detector 8")
    p_train.add_argument("--learning-rate", type=float, default=None,
                         help="default: reader 3e-4, detector 1e-4")
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--input-size", default=None,
                         help="reader: side of the square crop in pixels (default 160); detector: WIDTHxHEIGHT "
                              "the frame is resized to inside the model (default 320x320, e.g. 640x360)")
    p_train.add_argument("--anchor-min-ratio", type=float, default=None,
                         help="detector: smallest default box as a fraction of the image side (default 0.2, "
                              "torchvision's; 0.2 of a 1344x760 frame is at least 155x88 pixels)")
    p_train.add_argument("--no-augment", action="store_true",
                         help="detector: no photometric augmentation (brightness incl. darkening, contrast, gamma, "
                              "colour) of the training frames; it is on by default, never flips")
    p_train.add_argument("--negative-frame-fraction", type=float, default=None,
                         help="detector: share of the frames without signs used per epoch, drawn again each epoch "
                              "(validation: one fixed draw); every frame with signs is used (default 0.1; 1.0: all, "
                              "as before; such frames give torchvision's SSD no loss)")
    p_train.add_argument("--select-by", choices=("recall", "loss"), default=None,
                         help="detector: what chooses the best epoch and stops early: validation recall at score "
                              "0.1, ties to fewer false boxes per frame (default), or validation loss")
    p_train.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--init-from", type=Path, default=None,
                         help="continue mode: start from these weights (a <task>-best.pt of an earlier run) "
                              "instead of the stable ones on Hugging Face; reader --classes v2 in a fresh run: start "
                              "from this reader-best.pt instead of ImageNet (the second stage after --mapillary-only), "
                              "its head row by row by class name, the classes its own and ours")
    p_train.add_argument("--pretrained", action="store_true",
                         help="allow torchvision ImageNet weights for the reader/backbone if not cached")
    p_train.add_argument("--no-amp", action="store_true", help="disable CUDA automatic mixed precision")
    # [classifier-chain] - START
    p_train.add_argument("--cutter-families", choices=("all", "speed"), default=None,
                         help="detector: the boxes it learns to cut: all, the speed signs and the other traffic signs "
                              "(default), or speed, the speed signs only (every detector before this option)")
    p_train.add_argument("--classes", choices=("v1", "v2"), default="v1",
                         help="reader: v1 the labels of the first reader (default); v2 the classifier: also "
                              "not_a_sign (removed proposals, boxes marked not a sign, --negatives) and the kinds "
                              "of other signs, with darker crops and box jitter (--no-augment: without)")
    p_train.add_argument("--negatives", help="reader --classes v2: the cutter's false boxes of nnslr "
                                             "mine-negatives (mined on this dataset and split)")
    p_train.add_argument("--mapillary", help="reader --classes v2: the crops file of nnslr mapillary-crops, whose "
                                             "signs join train (never validation or test)")
    p_train.add_argument("--mapillary-only", action="store_true",
                         help="with --mapillary: the first stage of a two-stage classifier, trained on every "
                              "Mapillary crop of a v2 class (no cap per kind; kinds that are no class of ours as "
                              "other_sign) and none of ours; validation our sign crops; the second stage is a fresh "
                              "run on ours with --init-from <this run>/reader-best.pt")
    p_train.add_argument("--mapillary-region", help="with --mapillary: only the photos taken in these continents "
                                                    "(europe, asia, africa, north_america, south_america, oceania) or "
                                                    "countries (CH, DE...), comma-separated; default all (needs nnslr "
                                                    "mapillary-locations, except all)")
    p_train.add_argument("--mapillary-share", type=float, help="with --mapillary: this percentage of the crops of the "
                                                               "region (more than 0, at most 100, default 100), always "
                                                               "the same ones for the same share")
    # [classifier-chain] - END
    p_train.set_defaults(func=_cmd_train)
    # [training-baseline] - END

    # [evaluation-baseline] - START
    p_eval = sub.add_parser(
        "evaluate",
        help="evaluate reader or detector checkpoints on a frozen dataset partition",
    )
    p_eval.add_argument("dataset", nargs="?", help="canonical dataset JSONL; defaults from --dataset-kind")
    p_eval.add_argument("--task", choices=("reader", "detector"), default="reader")
    p_eval.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_eval.add_argument("--splits", help="frozen split JSON; default: <data-root>/splits/latest.json")
    p_eval.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
    p_eval.add_argument(
        "--partition",
        choices=("train", "validation", "test", "route-held-out", "hard-negative"),
        default="test",
    )
    p_eval.add_argument("--checkpoint", help="reader-best.pt or detector-best.pt; required unless --dry-run")
    p_eval.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p_eval.add_argument("--batch-size", type=int, default=None, help="reader evaluation batch size (default: 64)")
    p_eval.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p_eval.add_argument("--score-threshold", type=float, default=.25, help="detector score threshold")
    p_eval.add_argument("--iou-threshold", type=float, default=.5, help="detector IoU match threshold")
    p_eval.add_argument("--dry-run", action="store_true", help="validate the evaluation population without importing torch")
    p_eval.add_argument("--output", help="optional JSON report path")
    # [classifier-chain] - START
    p_eval.add_argument("--cutter-families", choices=("all", "speed"), default=None,
                        help="detector: the families its reviewed boxes are; default those of the checkpoint (one "
                             "that does not say: speed), with --dry-run all")
    p_eval.add_argument("--classes", choices=("v1", "v2"), default="v1",
                        help="reader: the class scheme of the checkpoint (v2: the classifier with not_a_sign)")
    p_eval.add_argument("--negatives", help="reader --classes v2: the negatives file the evaluated partition's "
                                            "mined not_a_sign crops come from")
    # [classifier-chain] - END
    p_eval.set_defaults(func=_cmd_evaluate)
    # [evaluation-baseline] - END

    # [classifier-chain] - START
    def _dataset_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("dataset", nargs="?", help="canonical dataset JSONL; defaults from --dataset-kind")
        parser.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
        parser.add_argument("--splits", help="frozen split JSON; default: <data-root>/splits/latest.json")
        parser.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
        parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        parser.add_argument("--batch-size", type=int, default=8, help="frames per batch of the cutter (default 8)")
        parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)

    p_negatives = sub.add_parser(
        "mine-negatives",
        help="run the cutter on the reviewed frames of train, validation and test and keep its false boxes as "
             "not_a_sign crops for the classifier (GPU); --dry-run for CPU-only counts",
    )
    _dataset_arguments(p_negatives)
    p_negatives.add_argument("--cutter", default="active",
                             help="signnet-best.pt (or an old detector-best.pt), or active: the cutter in use "
                                  "(models/active.json; default). A SignNet gives its best 5 boxes a frame, as the car, "
                                  "but from --score-threshold (the car keeps those of 0.3 or more)")
    p_negatives.add_argument("--output", help="new JSONL file, e.g. <data-root>/runs/<run>/negatives.jsonl; "
                                              "required unless --dry-run")
    p_negatives.add_argument("--score-threshold", type=float, default=0.1,
                             help="keep the cutter's boxes of at least this score (default 0.1)")
    p_negatives.add_argument("--per-frame", type=int, default=3,
                             help="at most this many boxes per frame, the highest scores (default 3)")
    p_negatives.add_argument("--total-factor", type=float, default=3.0,
                             help="at most this many times the sign crops of each partition (default 3)")
    p_negatives.add_argument("--sign-score", type=float, default=0.3,
                             help="in frames whose reviewer did not box the other signs (an AI before the rule), "
                                  "leave out the boxes of at least this score: likely signs nobody boxed "
                                  "(default 0.3)")
    p_negatives.add_argument("--seed", type=int, default=0)
    p_negatives.add_argument("--partition", action="append", choices=("train", "validation", "test"),
                             help="only these partitions (repeatable; default all three)")
    p_negatives.add_argument("--dry-run", action="store_true",
                             help="count the frames, sign crops, caps and the dataset's own negatives without torch")
    p_negatives.set_defaults(func=_cmd_mine_negatives)

    p_proposer = sub.add_parser(
        "evaluate-proposer",
        help="run a proposer of Prepare (vietnam, ours, qwen) on the reviewed frames of a partition and measure its "
             "proposals against the kept boxes: speed signs found and read right, every sign found, false proposals "
             "per frame, seconds per frame (report in <data-root>/evaluations/proposers/)",
    )
    _dataset_arguments(p_proposer)
    p_proposer.add_argument("--proposer", choices=("vietnam", "ours", "qwen"),
                            default=os.environ.get("NNSLR_PROPOSER") or "vietnam")
    p_proposer.add_argument("--partition", choices=("test", "validation"), default="test")
    p_proposer.add_argument("--limit", type=int, default=None, help="only the first N frames (a quick try)")
    p_proposer.set_defaults(func=_cmd_evaluate_proposer)

    p_chain = sub.add_parser(
        "evaluate-chain",
        help="the cutter and the classifier end to end on a partition: signs found and read right, false boxes "
             "per frame that survive, at several cutter thresholds (GPU); --dry-run for CPU-only counts",
    )
    _dataset_arguments(p_chain)
    p_chain.add_argument("--cutter", default="active",
                         help="signnet-best.pt (or an old detector-best.pt), or active: the cutter in use (default)")
    p_chain.add_argument("--classifier", default="active",
                         help="reader-best.pt of a classifier (v2) or of an old reader (v1), or active (default)")
    p_chain.add_argument("--partition", choices=("train", "validation", "test"), default="test")
    from nnslr_tools import chain as chain_settings

    def scores(values) -> str:
        return ",".join(f"{value:g}" for value in values)
    signnet_scores, signnet_main = chain_settings.chain_scores("signnet")
    detector_scores, detector_main = chain_settings.chain_scores("detector")
    p_chain.add_argument("--score-thresholds", default=None,
                         help="cutter score thresholds, separated by commas (default by the cutter's kind: "
                              f"{scores(signnet_scores)} with a SignNet, {scores(detector_scores)} with an old "
                              "Detector)")
    p_chain.add_argument("--main-score-threshold", type=float, default=None,
                         help=f"the threshold whose frames the report lists (default {signnet_main:g} with a SignNet, "
                              f"the car's; {detector_main:g} with an old Detector)")
    p_chain.add_argument("--iou-threshold", type=float, default=.5, help="IoU of a box that finds a sign")
    p_chain.add_argument("--dry-run", action="store_true",
                         help="count the frames and the reviewed boxes without torch")
    p_chain.add_argument("--output", help="JSON report path, e.g. <run>/evaluation/chain-test.json")
    p_chain.set_defaults(func=_cmd_evaluate_chain)
    # [classifier-chain] - END

    # [experiment-b] - START
    p_expb = sub.add_parser(
        "experiment-b",
        help="experiment B on the reviewed boxes: reading and whom a sign applies to, from the crop alone, "
             "with the box geometry and car speed, or with the scene too; --dry-run for CPU-only counts",
    )
    p_expb.add_argument("dataset", nargs="?", help="canonical dataset JSONL; defaults from --dataset-kind")
    p_expb.add_argument("--variant", choices=("crop", "crop_geom", "full"), required=True,
                        help="crop: the box crop only; crop_geom: + box geometry and car speed; "
                             "full: + the whole frame pooled at the box")
    p_expb.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_expb.add_argument("--splits", help="frozen split JSON; default: <data-root>/splits/latest.json")
    p_expb.add_argument("--dataset-kind", choices=("gold", "training-candidate"), default="gold")
    p_expb.add_argument("--dry-run", action="store_true",
                        help="validate and print the counts without importing torch or touching a GPU")
    p_expb.add_argument("--output", help="new output directory; required unless --dry-run")
    p_expb.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p_expb.add_argument("--epochs", type=int, default=20)
    p_expb.add_argument("--batch-size", type=int, default=32,
                        help="frames per batch, each with all its boxes (default: 32)")
    p_expb.add_argument("--learning-rate", type=float, default=3e-4)
    p_expb.add_argument("--weight-decay", type=float, default=1e-4)
    p_expb.add_argument("--crop-size", type=int, default=96, help="side of the square box crop (default: 96)")
    p_expb.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p_expb.add_argument("--seed", type=int, default=0)
    p_expb.add_argument("--limit", type=int, default=None,
                        help="keep only the first N frames of each partition (smoke runs, not a measure)")
    p_expb.add_argument("--pretrained", action="store_true",
                        help="start the MobileNetV3-small backbones from torchvision ImageNet weights")
    p_expb.add_argument("--no-amp", action="store_true", help="disable CUDA automatic mixed precision")
    p_expb.add_argument("--init-from", default=None,
                        help="the <variant>-best.pt of the same variant of an earlier run to go on from: every layer, "
                             "the reading head by class name")
    p_expb.set_defaults(func=_cmd_experiment_b)
    # [experiment-b] - END

    # [hard-example-mining] - START
    p_mine = sub.add_parser(
        "mine-hard-examples",
        help="extract reader/detector errors from an evaluation JSON report",
    )
    p_mine.add_argument("evaluation", help="JSON report produced by nnslr evaluate --output")
    p_mine.add_argument("--output", required=True, help="output JSONL manifest of hard examples")
    p_mine.set_defaults(func=_cmd_mine_hard_examples)
    # [hard-example-mining] - END

    # [onnx-export] - START
    p_export = sub.add_parser(
        "export-onnx",
        help="export a trained reader/detector/signnet/usernet checkpoint to a verified ONNX reference artifact",
    )
    p_export.add_argument("--task", choices=("reader", "detector", "signnet", "usernet"), default="reader")
    p_export.add_argument("--checkpoint", required=True,
                          help="reader-best.pt, detector-best.pt, signnet-best.pt or usernet-best.pt")
    p_export.add_argument("--output", required=True, help="target .onnx path")
    p_export.add_argument("--detector-input-width", type=int, default=1344,
                          help="detector native input width (default: narrow road 1344)")
    p_export.add_argument("--detector-input-height", type=int, default=760,
                          help="detector native input height (default: narrow road 760)")
    p_export.add_argument("--no-verify", action="store_true", help="skip ONNX Runtime numerical parity verification")
    p_export.set_defaults(func=_cmd_export_onnx)
    # [onnx-export] - END

    p_video = sub.add_parser(
        "drive-video",
        help="make the MP4 of a drive (road camera, 960 px, H.264) that the Drive view of Results plays",
    )
    p_video.add_argument("route", help="route id, e.g. 0000012d--6552a45378")
    p_video.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_video.add_argument("--again", action="store_true", help="make it again even when it is up to date")
    p_video.set_defaults(func=_cmd_drive_video)

    # [drive-review] - START
    p_drive_review = sub.add_parser(
        "drive-review",
        help="read drives' comma logs, as Read the logs on Speed review: the series of the drive player, Sign distance "
             "and the brain's training, and the episodes (CPU only)",
    )
    p_drive_review.add_argument("action", choices=("scan",))
    p_drive_review.add_argument("routes", nargs="*", help="the drives, e.g. 00000154--fa7e8f49aa")
    p_drive_review.add_argument("--all", action="store_true",
                                help="every drive with logs never scanned, out of date or not imported yet")
    p_drive_review.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_drive_review.set_defaults(func=_cmd_drive_review)
    # [drive-review] - END

    # [sign-reads] - START
    p_sign_reads = sub.add_parser(
        "sign-reads",
        help="the exact frame of every sign the comma's vision read (signRead in the drive's logs), added to Sign "
             "review next to the pictures at 1 a second: read_<1000 + frame>.png and sign_reads.jsonl per segment "
             "(CPU only; reads the drive's logs first when they are not read; run preannotate after it)",
    )
    p_sign_reads.add_argument("routes", nargs="+", help="the drives, e.g. 00000160--ea56ae0f33")
    p_sign_reads.add_argument("--no-scan", action="store_true", help="do not read the drive's logs, use the last scan")
    p_sign_reads.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_sign_reads.add_argument("--ffmpeg")
    p_sign_reads.add_argument("--ffprobe")
    p_sign_reads.add_argument("--workers", type=int, default=int(os.environ.get("NNSLR_EXTRACT_WORKERS", EXTRACT_WORKERS)),
                              help=f"segments at once (NNSLR_EXTRACT_WORKERS, default {EXTRACT_WORKERS})")
    p_sign_reads.set_defaults(func=_cmd_sign_reads)
    # [sign-reads] - END

    # [localizer] - START
    p_localize = sub.add_parser(
        "localize",
        help="the precise position of every road-camera frame of the drives, from openpilot.distill's localizer "
             "(CPU only, needs its .venv; derived/localizer/<route>/<segment>/), which nnslr gps then takes instead "
             "of the GPS fix",
    )
    p_localize.add_argument("routes", nargs="*", help="the drives, e.g. 00000154--fa7e8f49aa")
    p_localize.add_argument("--all", action="store_true", help="every drive with segments in raw/routes")
    p_localize.add_argument("--workers", type=int, default=6, help="segments localized side by side (default 6)")
    p_localize.add_argument("--again", action="store_true",
                            help="try again the segments the localizer could not use (localizer.failed)")
    p_localize.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_localize.set_defaults(func=_cmd_localize)
    # [localizer] - END

    p_car = sub.add_parser(
        "on-car",
        help="record which networks run on the comma (models/on-car.json, shown on the Models tab) from a clone of "
             "the published models repository",
    )
    p_car.add_argument("--models", required=True,
                       help="clone of the published models repository (speed_limits.json, format 2)")
    p_car.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_car.add_argument("--dry-run", action="store_true", help="check and print the record, write nothing")
    p_car.set_defaults(func=_cmd_on_car)

    p_simulate = sub.add_parser(
        "simulate",
        help="replay drives through the car's own vision loop with a set of networks, on the GPU, and write what the "
             "comma's screen would show along them (derived/simulation/<drive>/<set>.json.gz): Models › Drive; "
             "helpers/simulate_drive.sh",
    )
    p_simulate.add_argument("--set", required=True, choices=("car", "in-use"),
                            help="car: the release in the sunnypilot clone; in-use: the versions in use on Models "
                                 "(exported by nnslr simulate-set)")
    p_simulate.add_argument("--route", action="append", help="a drive (may repeat)")
    p_simulate.add_argument("--certified", action="store_true", help="every certified drive: the regression test")
    p_simulate.add_argument("--again", action="store_true", help="also the drives replayed already with these networks")
    p_simulate.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                            help="cuda (default); cpu is for the tests alone: a replay never moves to the CPU by itself")
    p_simulate.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_simulate.set_defaults(func=_cmd_simulate)

    p_simulate_set = sub.add_parser(
        "simulate-set",
        help="make the networks of a set ready for nnslr simulate: for in-use, export the versions in use on Models "
             "as Publish would (in the GPU environment, PyTorch), into derived/simulation/sets/in-use; nothing when "
             "they are exported already",
    )
    p_simulate_set.add_argument("--set", required=True, choices=("car", "in-use"))
    p_simulate_set.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_simulate_set.set_defaults(func=_cmd_simulate_set)

    p_send = sub.add_parser(
        "send-to-car",
        help="export a SignNet and the classifier in use to ONNX (with the road type in use, when one is chosen), "
             "publish them in the clone of the models the comma "
             "downloads (one commit pushed; nothing when they are the published ones already) and record them in "
             "models/on-car.json: steps 1-3 of scripts/send_to_car.sh",
    )
    p_send.add_argument("--signnet", required=True, help="the run of the SignNet (one listed on the Models tab)")
    p_send.add_argument("--classifier", help="the id of the classifier (default: the one in use, models/active.json)")
    p_send.add_argument("--models", help="clone of cristianku/speed-limits-models "
                                         "(default: NNSLR_MODELS_CLONE or /root/speed-limits-models)")
    p_send.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_send.add_argument("--dry-run", action="store_true",
                        help="export and compare only: nothing copied, committed, pushed or recorded")
    p_send.add_argument("--steps-of", type=int, default=3,
                        help="number the step lines n/N (scripts/send_to_car.sh: 5, its last two steps are its own)")
    p_send.set_defaults(func=_cmd_send_to_car)

    p_try = sub.add_parser(
        "try-on-comma",
        help="copy a SignNet and the classifier in use straight onto the comma over ssh, after checking them with the "
             "comma's own code, and wait for it to compile them (until its next software update; no publication, "
             "no build); --restore puts back the build's networks: scripts/try_on_comma.sh",
    )
    p_try.add_argument("--signnet", help="the run of the SignNet (one listed on the Models tab)")
    p_try.add_argument("--classifier", help="the id of the classifier (default: the one in use, models/active.json)")
    p_try.add_argument("--restore", action="store_true",
                       help="put back the build's networks on the comma (git checkout -- . in its models folder)")
    p_try.add_argument("--dry-run", action="store_true",
                       help="check the comma and the networks with its code (steps 1-3), copy nothing")
    p_try.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_try.set_defaults(func=_cmd_try_on_comma)

    # [offline-replay] - START
    p_replay = sub.add_parser(
        "replay",
        help="run offline detector -> reader perception over extracted route frames",
    )
    p_replay.add_argument("--route", required=True, help="local extracted route ID")
    p_replay.add_argument("--data-root", help="default: NNSLR_DATA_ROOT or <repo>/data")
    p_replay.add_argument("--bundle", help="verified packaged ONNX bundle; uses CPU reference runtime")
    p_replay.add_argument("--detector-checkpoint", help="detector-best.pt")
    p_replay.add_argument("--reader-checkpoint", help="reader-best.pt")
    p_replay.add_argument("--output", help="output JSONL; required unless --dry-run")
    p_replay.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p_replay.add_argument("--detector-threshold", type=float, default=.25)
    p_replay.add_argument("--reader-threshold", type=float, default=0.0)
    p_replay.add_argument("--max-detections", type=int, default=20)
    p_replay.add_argument("--dry-run", action="store_true", help="validate frame manifests only; no torch/GPU")
    p_replay.set_defaults(func=_cmd_replay)
    # [offline-replay] - END

    # [model-bundle] - START
    p_package = sub.add_parser(
        "package-model",
        help="package verified detector+reader ONNX artifacts into an immutable bundle",
    )
    p_package.add_argument("--detector-onnx", required=True)
    p_package.add_argument("--detector-contract", required=True)
    p_package.add_argument("--reader-onnx", required=True)
    p_package.add_argument("--reader-contract", required=True)
    p_package.add_argument("--output", required=True, help="new bundle directory")
    p_package.set_defaults(func=_cmd_package_model)

    p_verify = sub.add_parser(
        "verify-bundle",
        help="verify bundle membership, hashes, provenance and advisory capability",
    )
    p_verify.add_argument("bundle", help="bundle directory")
    p_verify.set_defaults(func=_cmd_verify_bundle)
    # [model-bundle] - END

    # [networks-package] - START
    p_networks = sub.add_parser(
        "networks-package",
        help="export the cutter, classifier and decider in use to ONNX as one package to publish "
             "(<data root>/publish/networks/<name>; GPU environment; uploads nothing)",
    )
    p_networks.add_argument("--name", help="package name (default speedlimits-<UTC date>-<time>)")
    p_networks.add_argument("--data-root")
    p_networks.set_defaults(func=_cmd_networks_package)
    p_verify_networks = sub.add_parser("verify-networks", help="verify a package of networks: files, sha256, contracts")
    p_verify_networks.add_argument("package", help="package folder")
    p_verify_networks.set_defaults(func=_cmd_verify_networks)
    p_networks_catalog = sub.add_parser("networks-catalog",
                                        help="add a package to the catalog speed_networks.json (a missing file "
                                             "starts one)")
    p_networks_catalog.add_argument("catalog", help="speed_networks.json")
    p_networks_catalog.add_argument("--package", required=True, help="package folder")
    p_networks_catalog.add_argument("--base-url", required=True, help="https://huggingface.co/<repo>/resolve/main")
    p_networks_catalog.set_defaults(func=_cmd_networks_catalog)
    p_networks_published = sub.add_parser("networks-published", help="record where a package was published")
    p_networks_published.add_argument("package", help="package folder")
    p_networks_published.add_argument("--repo", required=True)
    p_networks_published.add_argument("--url", required=True)
    p_networks_published.set_defaults(func=_cmd_networks_published)
    # [networks-package] - END

    # [core-export] - START
    p_core = sub.add_parser(
        "export-core",
        help="export and verify the portable stdlib-only speed_vision_core snapshot",
    )
    p_core.add_argument("--output", required=True, help="new destination directory for speed_vision_core")
    p_core.set_defaults(func=_cmd_export_core)
    # [core-export] - END

    # [ai-reviewer] - START
    p_llm = sub.add_parser("llm-provider", help="the LLM providers that can review signs (OpenAI-compatible /v1 "
                                                "servers) and the one that reviews; the keys stay in "
                                                "<data-root>/web/private/llm-providers.json")
    llm_sub = p_llm.add_subparsers(dest="llm_command", required=True)
    p_llm_list = llm_sub.add_parser("list", help="list the providers (never their keys)")
    p_llm_add = llm_sub.add_parser("add", help="add a provider, or update the one with the id of that name")
    p_llm_add.add_argument("--name", required=True, help="a name of 1-40 characters; the id is made from it")
    p_llm_add.add_argument("--base-url", required=True, help="the OpenAI-compatible API, ending in /v1")
    p_llm_add.add_argument("--model", help="the model; default: the first one the server lists")
    p_llm_add.add_argument("--key-file", help="a file holding the API key (read, never printed); without it a "
                                              "new provider has no key and an update keeps its key")
    p_llm_add.add_argument("--reviewer", action="store_true",
                           help="test it, and make it the AI reviewer if it passes")
    p_llm_remove = llm_sub.add_parser("remove", help="remove a provider and its key")
    p_llm_remove.add_argument("id")
    p_llm_use = llm_sub.add_parser("use", help="choose the AI reviewer, one that passed its test ('none' for none)")
    p_llm_use.add_argument("id", help="a provider id, or none")
    p_llm_models = llm_sub.add_parser("models", help="the models the provider's server lists")
    p_llm_models.add_argument("id")
    p_llm_test = llm_sub.add_parser("test", help="ask the provider about a reviewed frame: does it answer in "
                                                 "JSON and see images? (nothing is saved but the result)")
    p_llm_test.add_argument("id")
    for p_item in (p_llm_list, p_llm_add, p_llm_remove, p_llm_use, p_llm_models, p_llm_test):
        p_item.add_argument("--data-root", help="NNSLR_DATA_ROOT or <repo>/data by default")
    p_llm.set_defaults(func=_cmd_llm_provider)

    p_ai = sub.add_parser("ai-review", help="review the frames of a route nobody confirmed with the AI reviewer; "
                                            "a dry run unless --save")
    p_ai.add_argument("route")
    p_ai.add_argument("--provider", help="a provider id; default: the AI reviewer")
    p_ai.add_argument("--save", action="store_true",
                      help="save the complete suggestions as AI reviews: final, the next import puts them into "
                           "training; frames a person confirmed are never touched")
    p_ai.add_argument("--limit", type=int, help="at most this many frames")
    p_ai.add_argument("--scope", choices=("signs", "lights"), default="signs",
                      help="signs: every frame left (default); lights: only the frames where the car comes to a "
                           "traffic light of the map (derived/osm/traffic_signals.jsonl), each still reviewed whole")
    p_ai.add_argument("--data-root", help="NNSLR_DATA_ROOT or <repo>/data by default")
    p_ai.set_defaults(func=_cmd_ai_review)

    p_ai_measure = sub.add_parser("measure-ai-reviewer", help="measure an AI provider on frames a person "
                                                              "reviewed (nothing is saved)")
    p_ai_measure.add_argument("--provider", help="a provider id; default: the AI reviewer")
    p_ai_measure.add_argument("--limit", type=int, default=60, help="frames in the sample (default 60)")
    p_ai_measure.add_argument("--data-root", help="NNSLR_DATA_ROOT or <repo>/data by default")
    p_ai_measure.set_defaults(func=_cmd_measure_ai_reviewer)
    # [ai-reviewer] - END

    return parser


# [reviewed-dataset] - START
def _cmd_dataset(args: argparse.Namespace) -> int:
    from nnslr_tools.annotations import import_reviews, load_jsonl, validate_dataset
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    # [model-review] - START
    kind = args.dataset_kind
    default_dataset = (root / "datasets" / "training_candidate_dataset" / "objects.jsonl"
                       if kind == "training-candidate" else root / "annotations" / "objects.jsonl")
    # [model-review] - END
    try:
        if hasattr(args, "inputs") and args.from_db:
            import sqlite3
            from nnslr_tools.annotations import DatasetError
            from nnslr_tools.review_db import import_from_db
            if args.inputs:
                raise DatasetError("invalid_arguments", "--from-db does not take review files")
            output = Path(args.output) if args.output else default_dataset
            try:
                report = import_from_db(root, output, route=args.route, dataset_kind=kind, dry_run=args.dry_run)
            except sqlite3.Error as exc:
                raise DatasetError("review_db_error", str(exc)) from exc
        elif hasattr(args, "inputs"):
            if args.route or args.dry_run:
                from nnslr_tools.annotations import DatasetError
                raise DatasetError("invalid_arguments", "--route and --dry-run need --from-db")
            inputs = [Path(p) for p in args.inputs] if args.inputs else sorted((root / "annotations/inbox").glob("*.jsonl"))
            # [model-review] - START
            output = Path(args.output) if args.output else default_dataset
            report = import_reviews(inputs, root, output, dataset_kind=kind)
            # [model-review] - END
        else:
            # [model-review] - START
            dataset = Path(args.dataset) if args.dataset else default_dataset
            # [model-review] - END
            rows = load_jsonl(dataset)
            if hasattr(args, "seed"):
                from nnslr_tools.splits import build_splits
                # [model-review] - START
                report = build_splits(rows, root, seed=args.seed, output=Path(args.output) if args.output else None,
                                      dataset_kind=kind, test_routes=args.test_route)
                # [model-review] - END
            else:
                # [model-review] - START
                report = validate_dataset(rows, root, dataset_kind=kind)
                # [model-review] - END
                if args.splits and report["valid"]:
                    from nnslr_tools.splits import validate_splits
                    report["errors"].extend(validate_splits(rows, json.loads(Path(args.splits).read_text()), report["dataset_sha256"], dataset_kind=kind))
                    report["valid"] = not report["errors"]
    except (ValueError, OSError) as exc:
        report = {"valid": False, "errors": [{"reason": getattr(exc, "reason", "dataset_io_error"), "detail": str(exc)}]}
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["valid"] else 1
# [reviewed-dataset] - END


# [preannotation] - START
def _cmd_preannotate(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import run_preannotation
    result = run_preannotation(args.route, data_root=Path(args.data_root) if args.data_root else None,
                               confidence=args.confidence, proposer=args.proposer, keep_reviews=args.keep_reviews)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_review_box(args: argparse.Namespace) -> int:
    from nnslr_tools import review_box

    root = _resolve_data_root(args.data_root)

    def say(line: str) -> None:
        print(line, flush=True)

    try:
        if args.box_command == "link":
            review_box.link(root, args.remote, args.fingerprint)
            say(f"Linked to {args.remote}.")
        elif args.box_command == "tick":
            review_box.tick(root, Path(args.box), log=say)
        elif args.box_command == "sync":
            state = review_box.sync(root, log=say)
            say(f"{len(state['drives'])} drives on the review machine.")
        else:
            say(json.dumps(review_box.view(root), indent=1, ensure_ascii=False))
    except review_box.Busy as exc:
        say(str(exc))
    except review_box.BoxError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from nnslr_tools.review import serve_review
    serve_review(args.route, data_root=Path(args.data_root) if args.data_root else None,
                 host=args.host, port=args.port)
    return 0


def _cmd_review_reclassify(args: argparse.Namespace) -> int:
    from collections import Counter
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.review_db import ReviewStore, db_path
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = ReviewStore(db_path(root))
    report = {"dry_run": args.dry_run, "database": str(db_path(root)), "reviewers": {}}
    for reviewer in args.reviewer:
        changed = store.reclassify_as_ai(reviewer, client="review-reclassify-ai", dry_run=args.dry_run)
        routes = Counter()
        for item in changed:
            routes[item["route_id"] + " " + item["run_id"]] += 1
        report["reviewers"][reviewer] = {"frames": len(changed), "by_route_run": dict(sorted(routes.items()))}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_review_withdraw_ai(args: argparse.Namespace) -> int:
    import sqlite3
    from collections import Counter
    from datetime import datetime, timezone
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.review_db import ReviewStore, db_path
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    path = db_path(root)
    store = ReviewStore(path)
    report = {"dry_run": args.dry_run, "database": str(path)}
    if not args.dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        copy = path.with_name(f"{path.name}.bak-withdraw-ai-{stamp}")
        source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        target = sqlite3.connect(str(copy))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        report["copy"] = str(copy)
    changed = store.withdraw_ai_reviews(client="review-withdraw-ai", dry_run=args.dry_run)
    reviewers = Counter(item["reviewer"] for item in changed)
    routes = Counter(item["route_id"] + " " + item["run_id"] for item in changed)
    report.update({"frames": len(changed), "by_reviewer": dict(reviewers.most_common()),
                   "by_route_run": dict(sorted(routes.items()))})
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_gps(args: argparse.Namespace) -> int:
    from nnslr_tools.gps import GpsError, build_positions, positions_cover, positions_current, recountry_positions
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.review import latest_report, list_reports, report_rows
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    routes = list(args.route or [])
    if args.all:
        routes += [route for route, _ in list_reports(root)
                   if not (args.missing and positions_current(root, route)
                           and positions_cover(root, route, _report_frames(root, route)))]
    if not routes:
        if args.all:
            # --all --missing with every route already done.
            return 0
        print("nnslr gps: name a --route or use --all", file=sys.stderr)
        return 2
    failed = 0
    for route in routes:
        try:
            if args.countries:
                result = recountry_positions(root, route)
            else:
                result = build_positions(root, route, report_rows(latest_report(root, route)))
        except (GpsError, ValueError, OSError) as exc:
            failed += 1
            result = {"route_id": route, "error": str(exc)}
        print(json.dumps(result, sort_keys=True), flush=True)
    return 1 if failed else 0


def _cmd_brain_targets(args: argparse.Namespace) -> int:
    from nnslr_tools import brain_targets
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = DriveReviewStore(root)
    drives = brain_targets.dataset(root, lambda route: store.merged(route, keep=False), args.route,
                                   rate=brain_targets.RATE_HZ, need_pictures=False, inputs=False,
                                   types=not args.no_road_types, networks=_sign_networks(args))
    found = brain_targets.summary(drives)
    print(json.dumps(found, indent=2) if args.json else brain_targets.summary_text(found))
    return 0


def _cards(value: str) -> list[str]:
    """--gpus: cards by nvidia-smi's numbers, "0,1"."""
    cards = [part.strip() for part in value.split(',')]
    if not all(part.isdigit() for part in cards):
        raise argparse.ArgumentTypeError(f"cards by nvidia-smi's numbers, e.g. 0,1: {value!r}")
    return list(dict.fromkeys(cards))


def _cmd_borrow_cards(args: argparse.Namespace) -> int:
    from nnslr_tools import panel
    from nnslr_tools.preannotate import resolve_data_root
    record = Path(os.environ['NNSLR_JOB_RECORD']) if os.environ.get('NNSLR_JOB_RECORD') else None
    if args.release:
        panel.release_cards(record)
        return 0
    if args.own is None:
        raise ValueError("--own is needed: the brain's card, by nvidia-smi's number")
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    cards = panel.borrow_cards(root / 'logs' / 'panel', args.own, record,
                               [int(card) for card in args.cards] if args.cards is not None else None)
    print(','.join(str(card) for card in cards))
    return 0


def _sign_networks(args: argparse.Namespace) -> dict | None:
    """--sign-networks: the run's networks.json (sign_networks.load), None without it."""
    if not getattr(args, 'sign_networks', None):
        return None
    from nnslr_tools import sign_networks
    return sign_networks.load(Path(args.sign_networks))


def _cmd_brain_networks(args: argparse.Namespace) -> int:
    from nnslr_tools import sign_networks
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    chosen = sign_networks.prepare(root, Path(args.run_dir), cutter=args.cutter or None,
                                   classifier=args.classifier or None, decider=args.decider or None,
                                   log=lambda line: print(line, flush=True))
    print(json.dumps({part: chosen[part] for part in sign_networks.PARTS}), flush=True)
    return 0


def _cmd_regression_plan(args: argparse.Namespace) -> int:
    from nnslr_tools import drive_logs, regression
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    specs = [spec for spec in args.openpilot.split(",") if spec]
    for route, wanted in regression.missing_driving(root, drive_logs.videos_dir(), specs):
        print(route, *wanted)
    return 0


def _cmd_regression_brain(args: argparse.Namespace) -> int:
    import torch

    from nnslr_tools import regression
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = DriveReviewStore(root)
    failed = regression.run_brains(root, [run for run in args.brains.split(",") if run],
                                   lambda route: store.merged(route, keep=False), device=torch.device(args.device),
                                   log=lambda line: print(line, flush=True))
    return 1 if failed else 0


def _cmd_regression_sign_networks(args: argparse.Namespace) -> int:
    from nnslr_tools import sign_networks
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    lines = []
    for run in [run for run in args.brains.split(",") if run]:
        try:
            result = json.loads((root / "runs" / run / "result.json").read_text(encoding="utf-8"))
            networks = sign_networks.of_brain(root, result.get("sign_networks"))
        except (OSError, ValueError) as problem:  # regression-brain records it on each drive
            print(f"{run}: {problem}", file=sys.stderr, flush=True)
            continue
        if networks is None:
            line = "-"
        else:
            ids = {part: networks[part] for part in sign_networks.PARTS}
            line = str(sign_networks.write(root / "regression" / "networks" / sign_networks.set_key(ids), ids,
                                           networks["models"]))
        if line not in lines:
            lines.append(line)
    print(*lines, sep="\n")
    return 0


def _cmd_train_brain(args: argparse.Namespace) -> int:
    from nnslr_tools import brain_train
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.init_from and not Path(args.init_from).is_file():
        raise ValueError(f"--init-from is not a file: {args.init_from}")
    store = DriveReviewStore(root)
    result = brain_train.train_brain(root, Path(args.run_dir), lambda route: store.merged(route, keep=False), pool=args.pool,
                                     works=not args.no_works, road=not args.no_road,
                                     types=not args.no_road_types, users=not args.no_users, device=args.device,
                                     dry_run=args.dry_run, routes=args.route,
                                     epochs=args.epochs or brain_train.EPOCHS, seed=args.seed,
                                     init_from=Path(args.init_from) if args.init_from else None,
                                     sign_networks=_sign_networks(args))
    print(json.dumps({**{part: {key: result[part][key] for key in ('speed', 'speeds_said', 'one_speed', 'follow_the_sign')}
                         for part in ('train', 'validation', 'test')},
                      **{part: result[part] for part in ('validation_swapped', 'test_swapped')}}, indent=2))
    return 0


def _cmd_road_labeler_train(args: argparse.Namespace) -> int:
    from nnslr_tools import road_labeler
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = DriveReviewStore(root)
    result = road_labeler.train_labeler(root, Path(args.run_dir), lambda route: store.merged(route, keep=False),
                                        routes=args.route, epochs=args.epochs or road_labeler.EPOCHS,
                                        device=args.device, dry_run=args.dry_run, seed=args.seed)
    print(json.dumps({part: result[part] for part in ('train', 'validation', 'test')}, indent=2))
    return 0


def _cmd_road_labeler_predict(args: argparse.Namespace) -> int:
    from nnslr_tools import road_labeler, road_labeler_data
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = DriveReviewStore(root)
    routes = list(args.route or [])
    if args.all:
        routes += [route for route in road_labeler_data.labeler_routes(root) if route not in routes]
    if not routes:
        print("nnslr road-labeler-predict: name a --route or use --all", file=sys.stderr)
        return 2

    def span_of(route: str):
        summary = store.summary(route)
        if not summary or not summary.get('span_ns') or summary['span_ns'][1] is None:
            return None
        return summary['t0_ns'], summary['span_ns'][1]

    found = road_labeler.predict_labeler(root, Path(args.run_dir), lambda route: store.merged(route, keep=False), span_of,
                                         routes, out_dir=Path(args.out_dir) if args.out_dir else None,
                                         marks=not args.no_marks, dry_run=args.dry_run,
                                         device=args.device)
    print(json.dumps({'drives': len(found)}))
    return 0


def _cmd_export_brain(args: argparse.Namespace) -> int:
    from nnslr_tools import brain_export
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    store = DriveReviewStore(root)
    run = Path(args.run_dir)
    print(json.dumps(brain_export.export_brain_onnx(run / brain_export.CHECKPOINT, run / 'export' / brain_export.BRAIN_FILE,
                                                    data_root=root, merged_of=lambda route: store.merged(route, keep=False),
                                                    route=args.route), indent=2))
    return 0


def _cmd_tunnels(args: argparse.Namespace) -> int:
    from nnslr_tools import drive_review, speed_review, tunnels
    from nnslr_tools.map_context import RoadMap
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    road_map = RoadMap.load(root)
    if road_map is None:
        print("nnslr tunnels: no map of the roads yet: helpers/road_types.sh", file=sys.stderr)
        return 2
    tunnel_roads = tunnels.tunnel_map(road_map)
    if not tunnel_roads.ways:
        print("nnslr tunnels: the map has no tunnels: read it again (helpers/road_types.sh --again)", file=sys.stderr)
        return 2
    drives = drive_review.store_for(root)
    store = speed_review.store_for(root)
    routes = list(args.route or [])
    if args.all:
        routes += [row['route'] for row in drives._rows('SELECT route FROM scans ORDER BY route')]
    if not routes:
        print("nnslr tunnels: name a --route or use --all", file=sys.stderr)
        return 2
    total = {'drives': 0, 'found': 0, 'written': 0, 'kept': 0}
    for route in routes:
        summary = drives.summary(route)
        if not summary or not summary.get('span_ns') or summary['span_ns'][1] is None:
            continue
        merged = drives.merged(route, keep=False)
        if not merged:
            continue
        found = tunnels.drive_tunnels(road_map, merged, tunnel_roads)
        total['drives'] += 1
        total['found'] += len(found)
        if not found:
            continue
        span = (summary['t0_ns'], summary['span_ns'][1])
        clock = lambda ns: f"{int((ns - span[0]) / 1e9) // 60}:{int((ns - span[0]) / 1e9) % 60:02d}"
        result = {'written': 0, 'kept': len(found)} if args.dry_run else tunnels.write_tunnels(store, route, found, span)
        total['written'] += result['written']
        total['kept'] += result['kept']
        print(json.dumps({'route_id': route, 'tunnels': [clock(a) + '-' + clock(b) for a, b in found], **result}))
    print(json.dumps(total))
    return 0


def _report_frames(root: Path, route: str) -> list[dict]:
    """The rows of a route's latest report, [] when it has none."""
    from nnslr_tools.preannotate import PreannotationError
    from nnslr_tools.review import latest_report, report_rows
    try:
        return report_rows(latest_report(root, route))
    except (PreannotationError, ValueError, OSError):
        return []


def _cmd_road_types(args: argparse.Namespace) -> int:
    from nnslr_tools.gps import GPS_RELATIVE_DIR, positions_path
    from nnslr_tools.map_context import RoadMap
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.road_types import build_road_types
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    routes = list(args.route or [])
    if args.all:
        routes += sorted(path.parent.name for path in (root / GPS_RELATIVE_DIR).glob('*/positions.jsonl'))
    if not routes:
        print("nnslr road-types: name a --route or use --all (routes with GPS positions: nnslr gps)", file=sys.stderr)
        return 2
    road_map = RoadMap.load(root)
    if road_map is None:
        print("nnslr road-types: no map of the roads yet: uv run --no-project --with osmium python "
              "scripts/osm_roads.py", file=sys.stderr)
        return 2
    failed = 0
    for route in routes:
        try:
            path = positions_path(root, route)
            if not path.is_file():
                raise ValueError(f"no GPS positions yet: nnslr gps --route {route}")
            positions = [line for line in _read_jsonl(path) if isinstance(line, dict) and "frame_key" in line]
            result = build_road_types(root, route, positions, road_map, frames=_report_frames(root, route))
        except (ValueError, OSError) as exc:
            failed += 1
            result = {"route_id": route, "error": str(exc)}
        print(json.dumps(result, sort_keys=True), flush=True)
    return 1 if failed else 0


def _cmd_junctions(args: argparse.Namespace) -> int:
    from nnslr_tools.gps import GPS_RELATIVE_DIR, positions_path
    from nnslr_tools.junctions import build_junctions, load_map
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    routes = list(args.route or [])
    if args.all:
        routes += sorted(path.parent.name for path in (root / GPS_RELATIVE_DIR).glob('*/positions.jsonl'))
    if not routes:
        print("nnslr junctions: name a --route or use --all (routes with GPS positions: nnslr gps)", file=sys.stderr)
        return 2
    try:
        junctions = load_map(root)
    except ValueError as exc:
        print(f"nnslr junctions: {exc}", file=sys.stderr)
        return 2
    if junctions is None:
        print("nnslr junctions: no map of the roads yet: helpers/road_types.sh", file=sys.stderr)
        return 2
    failed = 0
    for route in routes:
        try:
            path = positions_path(root, route)
            if not path.is_file():
                raise ValueError(f"no GPS positions yet: nnslr gps --route {route}")
            positions = [line for line in _read_jsonl(path) if isinstance(line, dict) and "frame_key" in line]
            result = build_junctions(root, route, positions, junctions)
        except (ValueError, OSError) as exc:
            failed += 1
            result = {"route_id": route, "error": str(exc)}
        print(json.dumps(result, sort_keys=True), flush=True)
    return 1 if failed else 0


def _cmd_mapillary_crops(args: argparse.Namespace) -> int:
    from nnslr_tools.mapillary import make_crops
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    header = make_crops(Path(args.annotations), Path(args.images), root, log=lambda line: print(line, flush=True))
    print(json.dumps(header, sort_keys=True), flush=True)
    return 0 if header["crops"] else 1


def _cmd_mapillary_windows(args: argparse.Namespace) -> int:
    from nnslr_tools.mtsd_windows import DEFAULT_MTSD_DIR, make_windows
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit {args.limit}: at least 1 photo")
    if args.other_windows < 0:
        raise ValueError(f"--other-windows {args.other_windows}: 0 or more")
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    header = make_windows(Path(args.mtsd) if args.mtsd else DEFAULT_MTSD_DIR, root, workers=max(1, workers),
                          other_windows=args.other_windows, limit=args.limit, with_val=args.with_val,
                          dry_run=args.dry_run, log=lambda line: print(line, flush=True))
    print(json.dumps(header, sort_keys=True), flush=True)
    return 0 if (header["counts"].get("train") or {}).get("windows") else 1


def _cmd_mapillary_locations(args: argparse.Namespace) -> int:
    """nnslr mapillary-locations: the token only from the environment (MAPILLARY_TOKEN), never printed."""
    from nnslr_tools.mapillary import (IMAGE_IDS_FILE, IMAGE_IDS_URL, TOKEN_ENV, LocationsError, download_image_ids,
                                       fetch_locations)
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    ids = Path(args.ids) if args.ids else root / IMAGE_IDS_FILE
    if not args.ids and not ids.is_file():
        if args.dry_run:
            print(json.dumps({"ids_table": "missing, would be downloaded", "file": str(ids)}), flush=True)
            return 0
        ids = download_image_ids(root, args.ids_url or IMAGE_IDS_URL)
        print(f"mapillary-locations: key table downloaded into {ids}", flush=True)
    token = os.environ.get(TOKEN_ENV, "").strip() or None
    if not token and not args.dry_run:
        print(f"mapillary-locations: no {TOKEN_ENV}: put the client token of a Mapillary app in the repository's "
              ".env (helpers/mapillary_locations.sh loads it)", file=sys.stderr, flush=True)
        return 2
    try:
        summary = fetch_locations(root, token=token, ids_path=ids, again=args.again, limit=args.limit,
                                  dry_run=args.dry_run, log=lambda line: print(line, flush=True))
    except LocationsError as error:
        print(f"mapillary-locations: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _cmd_train_signnet(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.signnet import band_size, fits_the_car, signnet_not_trainable, train_signnet
    from nnslr_tools.training import prepare_detector_training
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    try:
        widths = tuple(int(value) for value in args.widths.split(","))
    except ValueError as exc:
        raise ValueError(f"invalid --widths {args.widths!r}: four whole numbers separated by commas") from exc
    if len(widths) != 4 or min(widths) < 1:
        raise ValueError(f"invalid --widths {args.widths!r}: four whole numbers separated by commas")
    if args.mapillary_ratio is not None and args.mapillary is None:
        raise ValueError("--mapillary-ratio caps the Mapillary windows: add --mapillary")
    if args.mapillary_ratio is not None and args.mapillary_only:
        raise ValueError("--mapillary-ratio caps the Mapillary windows added to ours: not with --mapillary-only, "
                         "which takes them all")
    if args.mapillary:
        from nnslr_tools.mtsd_windows import WINDOW
        from nnslr_tools.signnet import CAR_BAND
        band = band_size(args.band_width)
        # A window is the frame's top 1344x432 pixels, the car's band exactly: a band of other rows would cut the
        # bottom of every window (its signs there learnt as background) or pad it.
        if band[1] * WINDOW[0] != WINDOW[1] * band[0]:
            raise ValueError(f"--mapillary: the windows are the car's band ({CAR_BAND[0]}x{CAR_BAND[1]}, the frame's top "
                             f"{WINDOW[0]}x{WINDOW[1]}); --band-width {args.band_width} makes a {band[0]}x{band[1]} "
                             "band of other rows")
    if args.plan_out and not args.dry_run:
        raise ValueError("--plan-out writes the plan of a --dry-run: add --dry-run, or --plan to read one")
    dataset = Path(args.dataset or root / "annotations" / "objects.jsonl")
    split = Path(args.split or root / "splits" / "latest.json")
    # What the plan is made of: a plan written by the dry run is the training's only while all of it is the same.
    inputs = {"data_root": str(root.resolve()), "dataset_kind": args.dataset_kind, "families": args.families,
              "mapillary_region": args.mapillary_region, "mapillary_share": args.mapillary_share,
              "mapillary_only": args.mapillary_only, "mapillary_ratio": args.mapillary_ratio,
              **{name: [str(path.resolve()), sha256_of(path) if path.is_file() else None]
                 for name, path in (("dataset", dataset), ("split", split),
                                    ("mapillary", Path(args.mapillary) if args.mapillary else None)) if path}}
    plan = None
    if args.plan:
        try:
            saved = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if saved.get("inputs") == inputs:
            plan = saved["plan"]
            print(f"plan: {args.plan}, written by the check of the population", file=sys.stderr, flush=True)
        else:
            print(f"plan: {args.plan} is missing or of other data or settings, built again", file=sys.stderr,
                  flush=True)
    if plan is None:
        plan = prepare_detector_training(dataset, split, root,
                                         dataset_kind=args.dataset_kind, families=args.families,
                                         mapillary=Path(args.mapillary) if args.mapillary else None,
                                         mapillary_region=args.mapillary_region,
                                         mapillary_share=args.mapillary_share, mapillary_only=args.mapillary_only,
                                         mapillary_ratio=1.0 if args.mapillary_ratio is None else args.mapillary_ratio)
    if args.plan_out:
        out = Path(args.plan_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(out.name + ".tmp").write_text(json.dumps({"inputs": inputs, "plan": plan}), encoding="utf-8")
        out.with_name(out.name + ".tmp").replace(out)
    if args.init_from and not Path(args.init_from).is_file():
        raise ValueError(f"--init-from is not a file: {args.init_from}")
    mapillary = plan.get("mapillary")
    if mapillary:
        trained = plan["trained_on"]
        # Our speed boxes: every speed family, the unreadable signs, zones and road markings too (the ratio's ours).
        print(f"mapillary: train holds {trained['ours']} of our frames ({trained['ours_boxes']} boxes, "
              f"{0 if plan.get('mapillary_only') else mapillary['ours_train_speed_boxes']} of the speed signs, "
              "unreadable, zone and road marking included) and "
              f"{trained['mapillary_windows']} "
              f"Mapillary windows ({trained['mapillary_boxes']} speed boxes"
              + (f", at most {mapillary['ratio']:g} times ours: {mapillary['before_cap']['speed_boxes']} before the cap"
                 if mapillary.get("ratio") is not None else ", all of the region and share") + ")",
              file=sys.stderr, flush=True)
    train_frames = sum(1 for frame in plan["frames"] if frame["split"] == "train")
    # One batch at least (the loader drops a last smaller one): a narrow region or share of the windows can hold
    # fewer; so the check of the population stops it before the GPU.
    short = bool(mapillary) and train_frames < args.batch
    trainable = bool(plan.get("trainable")) and not short
    if args.dry_run:
        print(json.dumps({"dry_run": True, "task": "signnet", "dataset_sha256": plan.get("dataset_sha256"),
                          "split_sha256": plan.get("split_sha256"), "families": plan.get("families"),
                          "counts_by_split": plan.get("counts_by_split"), "boxes_by_split": plan.get("boxes_by_split"),
                          "boxes_by_family": plan.get("boxes_by_family"), "mapillary": mapillary,
                          "mapillary_only": bool(plan.get("mapillary_only")), "trained_on": plan.get("trained_on"),
                          "train_frames": train_frames, "batch": args.batch,
                          "band": list(band_size(args.band_width)), "stride": args.stride,
                          "fits_the_car": fits_the_car(band_size(args.band_width), args.stride),
                          "trainable": trainable},
                         sort_keys=True), flush=True)
        if short:
            print(f"not trainable: {train_frames} train frames, fewer than one batch of {args.batch}",
                  file=sys.stderr, flush=True)
        return 0 if trainable else 1
    if not trainable:
        raise ValueError(signnet_not_trainable(plan)
                         + (f"; {train_frames} train frames, fewer than one batch of {args.batch}" if short else ""))
    result = train_signnet(plan, root, Path(args.run_dir), widths=widths, stride=args.stride, epochs=args.epochs,
                           batch=args.batch, learning_rate=args.learning_rate, workers=args.workers,
                           band_width=args.band_width, fuse_kernel=args.fuse_kernel,
                           init_from=Path(args.init_from) if args.init_from else None,
                           log=lambda line: print(line, flush=True))
    summary = {split: {budget: {key: result[split]["by_box_budget"][budget][key]
                                for key in ("recall_iou", "false_positives_per_image")}
                       for budget in result[split]["by_box_budget"]}
               for split in ("validation", "test")}
    car = {split: {key: result[split]["car"][key] for key in ("recall_iou", "false_positives_per_image")}
           for split in ("validation", "test")}
    print(json.dumps({"run_dir": args.run_dir, "best_epoch": result["best_epoch"], "parameters": result["parameters"],
                      "by_box_budget": summary, "car": car}, sort_keys=True), flush=True)
    return 0


def _cmd_train_usernet(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.usernet import BDD_DIR, prepare_usernet, train_usernet
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.plan_out and not args.dry_run:
        raise ValueError("--plan-out writes the plan of a --dry-run: add --dry-run, or --plan to read one")
    if args.init_from and not Path(args.init_from).is_file():
        raise ValueError(f"--init-from is not a file: {args.init_from}")
    bdd_dir = Path(args.bdd_dir or os.environ.get("NNSLR_BDD_DIR") or BDD_DIR)
    dataset = Path(args.dataset or root / "annotations" / "objects.jsonl")
    split = Path(args.split or root / "splits" / "latest.json")
    samples = bdd_dir / "samples.json"
    # What the plan is made of: a plan written by the dry run is the training's only while all of it is the same.
    inputs = {"data_root": str(root.resolve()), "stage": args.stage, "dataset_kind": args.dataset_kind,
              **{name: [str(path.resolve()), sha256_of(path) if path.is_file() else None]
                 for name, path in (("dataset", dataset), ("split", split), ("bdd", samples))}}
    plan = None
    if args.plan:
        try:
            saved = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if saved.get("inputs") == inputs:
            plan = saved["plan"]
            print(f"plan: {args.plan}, written by the check of the population", file=sys.stderr, flush=True)
        else:
            print(f"plan: {args.plan} is missing or of other data or settings, built again", file=sys.stderr,
                  flush=True)
    if plan is None:
        plan = prepare_usernet(args.stage, root, bdd_dir=bdd_dir, dataset_path=dataset, split_path=split,
                               dataset_kind=args.dataset_kind)
    if args.plan_out:
        out = Path(args.plan_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(out.name + ".tmp").write_text(json.dumps({"inputs": inputs, "plan": plan}), encoding="utf-8")
        out.with_name(out.name + ".tmp").replace(out)
    train_frames = sum(1 for frame in plan["frames"] if frame["split"] == "train")
    short = train_frames < args.batch
    trainable = bool(plan.get("trainable")) and not short
    reason = "; ".join(part for part in (plan.get("not_trainable"), f"{train_frames} train frames, fewer than one "
                                         f"batch of {args.batch}" if short else None) if part)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "task": "usernet", "stage": plan["stage"], "uses_bdd": plan["uses_bdd"],
                          "validation_source": plan["validation_source"], "counts_by_split": plan["counts_by_split"],
                          "dataset_sha256": plan.get("dataset_sha256"), "split_sha256": plan.get("split_sha256"),
                          "bdd": plan.get("bdd"), "drives": plan.get("drives"), "train_frames": train_frames,
                          "batch": args.batch, "trainable": trainable}, sort_keys=True), flush=True)
        if not trainable:
            print(f"not trainable: {reason}", file=sys.stderr, flush=True)
        return 0 if trainable else 1
    if not trainable:
        raise ValueError(f"usernet_not_trainable: {reason}")
    result = train_usernet(plan, root, Path(args.run_dir), epochs=args.epochs, batch=args.batch,
                           learning_rate=args.learning_rate, workers=args.workers,
                           init_from=Path(args.init_from) if args.init_from else None,
                           log=lambda line: print(line, flush=True))
    print(json.dumps({"run_dir": args.run_dir, "best_epoch": result["best_epoch"], "parameters": result["parameters"],
                      "uses_bdd": result["uses_bdd"], "validation_source": result["validation_source"],
                      "validation": result["validation"]["at_min_score"],
                      "test": (result["test"] or {}).get("at_min_score")}, sort_keys=True), flush=True)
    return 0


def _cmd_remeasure_signnet(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.signnet import remeasure_car_centre
    from nnslr_tools.training import prepare_detector_training
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    split = Path(args.split) if args.split else root / "splits" / f"{result.get('split_sha256')}.json"
    if not args.split and not split.is_file():
        split = root / "splits" / "latest.json"
    plan = prepare_detector_training(Path(args.dataset or root / "annotations" / "objects.jsonl"), split, root,
                                     dataset_kind=args.dataset_kind, families=result.get("families") or "all")
    summary = remeasure_car_centre(plan, root, run_dir, device=args.device, batch=args.batch, workers=args.workers,
                                   dry_run=args.dry_run)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _cmd_train_box_fit(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.box_fit import HERE as box_fit_dir
    from nnslr_tools.box_fit.train import DEFAULT_WORKERS, run
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    workers = args.workers if args.workers is not None else DEFAULT_WORKERS
    if workers < 1 or args.folds < 0 or args.folds > 20:
        raise ValueError("--workers must be at least 1 and --folds between 0 and 20")
    summary = run(root, Path(args.out) if args.out else box_fit_dir,
                  objects=Path(args.dataset) if args.dataset else None, seed=args.seed, workers=workers,
                  folds=args.folds, dry_run=args.dry_run, device=args.device,
                  log=lambda line: print(line, file=sys.stderr, flush=True))
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _cmd_road_type_signs(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.brain_targets import training_routes
    from nnslr_tools.road_type_data import road_type_routes
    from nnslr_tools.road_type_signs import board_times, build_signs, package_path, sign_times
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.classifier and not Path(args.classifier).is_file():
        raise ValueError(f"--classifier is not a file: {args.classifier}")
    if args.certified:
        from nnslr_tools import regression
        routes = [route for route in sorted(regression.certified(root)) if regression.frames_ready(root, route)]
        if not routes:
            print("nnslr road-type-signs: no certified drive is prepared", flush=True)
            return 0
    else:
        routes = list(args.route or []) or (training_routes(root) if args.labelled else road_type_routes(root))
    if not routes:
        print("nnslr road-type-signs: " + ("no drive to learn from (sunnypilot off, its logs read to the end and its frames: "
                                           "Read the logs in Data › Speed review)"
                                           if args.labelled else "no drive with road types (nnslr road-types)"),
              file=sys.stderr)
        return 2
    chosen = _sign_networks(args)
    if chosen and args.classifier:
        raise ValueError("--classifier and --sign-networks: the classifier is the one of the networks chosen")
    from nnslr_tools.drive_review import DriveReviewStore
    store = DriveReviewStore(root)
    records = build_signs(root, routes, package=package_path(Path(args.sunnypilot) if args.sunnypilot else None),
                          workers=args.workers or None, device=args.device, gpus=args.gpus,
                          classifier=Path(args.classifier) if args.classifier else None,
                          pictures=bool(args.labelled or args.certified), models=chosen['models'] if chosen else None,
                          speeds_of=lambda route: store.merged(route, keep=False),
                          log=lambda line: print(line, flush=True))
    for route, record in records.items():
        left = record.get('other_road') or []
        print(f"{route}: {sum(record['frames'].values())} frames read, {len(record['readings'])} with a sign, "
              + (f"{len(left)} with a sign of another road left out, " if left else '') +
              f"motorway signs confirmed at {', '.join(f'{t:.0f} s' for t in sign_times(record)) or 'none'}, "
              f"green boards at {', '.join(f'{t:.0f} s' for t in board_times(record)) or 'none'}", flush=True)
    return 0


def _cmd_road_users(args: argparse.Namespace) -> int:
    """The road users of the brain's training: the detector on the drives' road video (the V100s), then the crossing
    rule's proposals; or the check of the readings against a person's ticked frames (nnslr_tools.road_users)."""
    from nnslr_tools import road_users
    from nnslr_tools.brain_targets import _exposures, training_routes
    from nnslr_tools.drive_review import DriveReviewStore
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    if args.check:
        print(road_users.check_text(road_users.check(root)), flush=True)
        return 0
    routes = list(args.route or []) or training_routes(root)
    for route in routes:
        road_users.checked_route(route)
    if not routes:
        print("nnslr road-users: no drive to read (the brain's drives: dashcam drives with their logs read)",
              file=sys.stderr)
        return 2
    if not args.proposals_only:
        cards = args.gpus
        if cards is None and os.environ.get('NNSLR_GPU_INDEX', '').strip():
            cards = _cards(os.environ['NNSLR_GPU_INDEX'].strip())
        done = road_users.read_drives(root, routes, cards=cards, batch=args.batch or road_users.BATCH,
                                      log=lambda line: print(line, flush=True))
        print(f"road users: {done['segments']} segments read, {done['frames']} readings, {done['seconds']:.0f} s "
              "on the cards", flush=True)
    store = DriveReviewStore(root)
    road_users.refresh_proposals(root, routes, lambda route: store.merged(route, keep=False), _exposures,
                                 log=lambda line: print(line, flush=True))
    return 0


def _cmd_car_motion(args: argparse.Namespace) -> int:
    from nnslr_tools.car_motion import MotionError, build_motion, coverage_text
    from nnslr_tools.gps import GPS_RELATIVE_DIR
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.road_type_data import road_type_routes
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    # With GPS but no road types yet: nnslr road-types reads their speed (a garage without GPS is slow).
    with_gps = {path.parent.name for path in (root / GPS_RELATIVE_DIR).glob('*/positions.jsonl')}
    routes = sorted(with_gps | set(road_type_routes(root))) if args.all else list(args.route or [])
    if not routes:
        print("nnslr car-motion: name a --route or use --all (drives with GPS positions: nnslr gps)",
              file=sys.stderr)
        return 2
    try:
        records = build_motion(root, routes, workers=args.workers or None, again=args.again,
                               log=lambda line: print(line, flush=True))
    except MotionError as exc:
        print(f"nnslr car-motion: {exc}", file=sys.stderr)
        return 1
    for route in routes:
        print(coverage_text(records[route]), flush=True)
    return 0


def _cmd_tracks(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.review import latest_report, list_reports, report_rows
    from nnslr_tools.review_db import ReviewStore, db_path
    from nnslr_tools.tracks import build_tracks
    root = resolve_data_root(Path(args.data_root) if args.data_root else None)
    routes = list(args.route or [])
    if args.all:
        routes += [route for route, _ in list_reports(root)]
    if not routes:
        print("nnslr tracks: name a --route or use --all", file=sys.stderr)
        return 2
    store = ReviewStore(db_path(root))
    for route in routes:
        directory = latest_report(root, route)
        result = build_tracks(root, route, report_rows(directory), store.reviews(route, directory.name))
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    from nnslr_tools.preannotate import resolve_data_root
    from nnslr_tools.review_mcp import run_stdio, sdk_available
    if not sdk_available():
        print('nnslr mcp: the MCP SDK is not installed; run  .venv/bin/python -m pip install -e ".[mcp]"',
              file=sys.stderr)
        return 2
    read_only = args.read_only or os.environ.get("NNSLR_MCP_READ_ONLY", "") == "1"
    # stdout carries the MCP messages: everything else goes to stderr.
    run_stdio(resolve_data_root(Path(args.data_root) if args.data_root else None), read_only=read_only)
    return 0
# [preannotation] - END


# [ai-reviewer] - START
def _ai_root(args: argparse.Namespace) -> Path:
    from nnslr_tools.preannotate import resolve_data_root
    return resolve_data_root(Path(args.data_root) if args.data_root else None)


def _ai_service(root: Path):
    from nnslr_tools.review_db import ReviewStore, db_path
    from nnslr_tools.review_service import ReviewService
    return ReviewService(root, ReviewStore(db_path(root)), log=lambda text: print(text, file=sys.stderr, flush=True))


def _ai_provider(root: Path, wanted: str | None) -> dict | None:
    """The private record of --provider, else of the AI reviewer; None (with the error printed) without one."""
    from nnslr_tools import llm_providers
    if wanted:
        return llm_providers.provider(root, wanted)
    record = llm_providers.reviewer_provider(root)
    if record is None:
        print("error: no AI reviewer: nnslr llm-provider use ID", file=sys.stderr)
    return record


def _llm_line(item: dict) -> str:
    line = "  ".join((item["id"], item["name"], item["base_url"], item["model"],
                      "key: set" if item["has_key"] else "key: none"))
    return line + ("  [reviewer]" if item["is_reviewer"] else "")


def _llm_add(root: Path, args: argparse.Namespace) -> int:
    from nnslr_tools import llm_providers
    key = None
    if args.key_file:
        path = Path(args.key_file).expanduser()
        try:
            mode = path.stat().st_mode
            key = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"error: cannot read the key file {path} ({type(exc).__name__})", file=sys.stderr)
            return 2
        if mode & 0o077:
            print(f"warning: other users can read the key file: chmod 600 {path}", file=sys.stderr)
        if not key:
            print(f"error: the key file {path} is empty", file=sys.stderr)
            return 2
    provider_id = llm_providers.id_from_name(args.name)
    existing = None
    for item in llm_providers.public_settings(root)["providers"]:
        if item["id"] == provider_id:
            existing = item
    model = args.model
    if not model:
        body = {"base_url": args.base_url}
        if key is not None:
            body["key"] = key
        elif existing is not None:
            # The stored key goes only to the stored address (list_models checks it).
            body["id"] = provider_id
        models = llm_providers.list_models(root, body)
        if not models:
            print("error: the server lists no model: give --model", file=sys.stderr)
            return 2
        model = models[0]["id"]
        print(f"model: {model} (the first the server lists)", file=sys.stderr)
    body = {"name": args.name, "base_url": args.base_url, "model": model}
    if existing is not None:
        body["id"] = provider_id
    if key is not None:
        body["key"] = key
    saved = llm_providers.save_provider(root, body)
    print(("updated  " if existing is not None else "added  ") + _llm_line(saved))
    if args.reviewer:
        # Only a provider that passed its Test reviews (llm_providers.set_reviewer): test it first.
        from nnslr_tools import ai_review
        result = ai_review.test_provider(_ai_service(root), saved["id"])
        print("test: " + result["message"], file=sys.stderr)
        if not result["ok"]:
            print(f"error: {saved['id']} did not pass its test: it is not the AI reviewer", file=sys.stderr)
            return 1
        llm_providers.set_reviewer(root, saved["id"])
        print(f"AI reviewer: {saved['id']}")
    return 0


def _cmd_llm_provider(args: argparse.Namespace) -> int:
    from nnslr_tools import llm_providers
    root = _ai_root(args)
    command = args.llm_command
    if command == "add":
        return _llm_add(root, args)
    if command == "list":
        providers = llm_providers.public_settings(root)["providers"]
        if not providers:
            print("no AI provider: nnslr llm-provider add --name N --base-url URL [--key-file F]", file=sys.stderr)
        for item in providers:
            print(_llm_line(item))
        return 0
    if command == "remove":
        llm_providers.delete_provider(root, args.id)
        print(f"removed {args.id}")
        return 0
    if command == "use":
        wanted = None if args.id == "none" else args.id
        llm_providers.set_reviewer(root, wanted)
        print(f"AI reviewer: {wanted or 'none'}")
        return 0
    if command == "models":
        for model in llm_providers.list_models(root, {"id": args.id}):
            sees = {True: "yes", False: "no", None: "unknown"}[model["sees_images"]]
            print(f"{model['id']}  sees images: {sees}")
        return 0
    from nnslr_tools import ai_review
    result = ai_review.test_provider(_ai_service(root), args.id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


def _cmd_ai_review(args: argparse.Namespace) -> int:
    from nnslr_tools.ai_review import review_route
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be 1 or more", file=sys.stderr)
        return 2
    root = _ai_root(args)
    record = _ai_provider(root, args.provider)
    if record is None:
        return 2
    from nnslr_tools.review_db import ReviewRejected
    try:
        summary = review_route(_ai_service(root), args.route, record, save=args.save, limit=args.limit,
                               scope=args.scope)
    except ReviewRejected as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["stopped"] else 0


def _cmd_measure_ai_reviewer(args: argparse.Namespace) -> int:
    from nnslr_tools.ai_review import measure_reviewer
    if args.limit < 1:
        print("error: --limit must be 1 or more", file=sys.stderr)
        return 2
    root = _ai_root(args)
    record = _ai_provider(root, args.provider)
    if record is None:
        return 2
    report = measure_reviewer(_ai_service(root), record, limit=args.limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
# [ai-reviewer] - END


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"nnslr_tools {__version__}")
    print("speed_vision_core 0.1.0")
    print(f"python {platform.python_version()}")
    print(f"schema_version {SUPPORTED_SCHEMA_VERSION}")
    print(f"max_detections_per_batch {MAX_DETECTIONS_PER_BATCH}")
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    report = collect_env_report()
    if getattr(args, "gpu_smoke", False):
        from nnslr_tools.training import gpu_smoke
        try:
            report["gpu"] = {"available": True, **gpu_smoke(args.gpu_device)}
        except ValueError as exc:
            report["gpu"] = {
                "status": "failed",
                "available": False,
                "error": str(exc),
            }
    print(json.dumps(report, indent=2))
    return 1 if report["gpu"]["status"] == "failed" else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NnslerContractError as exc:
        print(f"error: {exc.reason_code}: {exc.message}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 2
    except (MediaToolError, CommaLogError, NnslerManifestError) as exc:
        detail = getattr(exc, "detail", "")
        stderr = getattr(exc, "stderr", "")
        print(f"error: {exc}", file=sys.stderr)
        if detail and detail not in str(exc):
            print(detail, file=sys.stderr)
        if stderr:
            print(stderr, file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
# [nnslr-t1] - END
