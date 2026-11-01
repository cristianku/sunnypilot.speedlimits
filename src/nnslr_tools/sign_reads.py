"""The exact frame of every sign the comma's vision read, added to Sign review next to the pictures at 1 a second.

speedvisiond reads the road camera 5 times a second; a sign it reads (signRead, in the drive's logs as Drive review
scans them: drive_extract) is often between two of the pictures at 1 a second, so the reviewer could not see what the
car saw. ``nnslr sign-reads --route R`` (run by helpers/1_prepare.sh after the pictures are taken out) takes the frame
of fcamera.hevc the car was reading at each such moment: the last frame whose exposure started (timestampSof of the
road camera's encode index) at or before the read. One frame per read, the first of each run of the same value
(a value that changes, or READ_GAP_S without a read); none where a picture at 1 a second is that very frame; in a
segment kept only at a car park's speed (the start or the end of a drive), only in the seconds kept.

The pictures are ``read_<output index>.png`` next to the others, listed in ``sign_reads.jsonl`` of the segment (the
rows of frames.jsonl, plus the value read, ``comma_read_kph``, and its time, ``comma_read_mono_ns``). Their output
index is READ_INDEX_BASE + the decoded frame index: it never collides with a picture at 1 a second (0..59) and never
moves, so the reviews saved by frame key stay where they are, and a new extraction of the 1-a-second pictures leaves
them alone. The frames of a segment are in time order (frame_order): never by output index alone.
Only the camera and the car's own logs: no map.
"""
from __future__ import annotations

import bisect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

READ_INDEX_BASE = 1000
CAMERA_FPS = 20.0
READS_MANIFEST = 'sign_reads.jsonl'
READ_PREFIX = 'read'
# A read of the same value this long after the previous one starts a new run (a new sign).
READ_GAP_S = 2.0


def is_read_frame(row: dict) -> bool:
    """A frame added at a comma read (not one of the pictures at 1 a second)."""
    index = row.get('output_index')
    return isinstance(index, int) and index >= READ_INDEX_BASE


def frame_second(row: dict) -> float:
    """The second of its segment a frame was taken at: its output index for the pictures at 1 a second (as every
    1 Hz step reads them), its media time for a frame of a read."""
    index = row['output_index']
    if index < READ_INDEX_BASE:
        return float(index)
    time = row.get('media_time_s')
    return float(time) if isinstance(time, (int, float)) else (index - READ_INDEX_BASE) / CAMERA_FPS


def frame_order(row: dict) -> tuple:
    """The frames of a drive in time order: segment, second, output index."""
    return row['segment_index'], frame_second(row), row['output_index']


def read_frames(merged: dict) -> dict[int, list[dict]]:
    """{segment: [{'decoded', 'kph', 't_ns'}]}: the frame of each run of comma reads (see the module's text), from
    a drive as Drive review joins it (drive_logs.merge)."""
    reads = sorted((event[0], event[2]) for event in merged.get('events') or ()
                   if len(event) >= 3 and event[1] == 'sign' and isinstance(event[0], int))
    starts = []
    previous = None
    for t, kph in reads:
        if previous is None or kph != previous[1] or t - previous[0] >= READ_GAP_S * 1e9:
            starts.append((t, kph))
        previous = (t, kph)
    segments = []
    for segment, frames in (merged.get('frames') or {}).items():
        sof, ids = frames.get('sof') or [], frames.get('ids') or []
        if sof and len(sof) == len(ids):
            pairs = sorted(zip(sof, ids))
            segments.append((pairs[0][0], int(segment), [p[0] for p in pairs], [p[1] for p in pairs]))
    segments.sort()
    found: dict[int, list[dict]] = {}
    for t, kph in starts:
        at = bisect.bisect_right([item[0] for item in segments], t) - 1
        if at < 0:
            continue
        _first, segment, sof, ids = segments[at]
        # The last frame exposed at or before the read; past the end of the segment (a hole) it is not this one.
        index = bisect.bisect_right(sof, t) - 1
        if index < 0 or t - sof[index] > 1e9 / CAMERA_FPS * 2:
            continue
        chosen = found.setdefault(segment, [])
        if all(item['decoded'] != ids[index] for item in chosen):
            chosen.append({'decoded': int(ids[index]), 'kph': kph, 't_ns': t})
    return found


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wanted_reads(reads: list[dict], pictures: list[dict], slow_only: bool) -> list[dict]:
    """The reads of a segment to add: not on a picture at 1 a second, and in a slow-only segment only in its
    seconds kept (pictures: the rows of its frames.jsonl)."""
    taken = {row.get('decoded_frame_index') for row in pictures}
    seconds = {int(frame_second(row)) for row in pictures}
    return [read for read in reads if read['decoded'] not in taken
            and (not slow_only or int(read['decoded'] / CAMERA_FPS) in seconds)]


def _current(directory: Path, manifest: list[dict], wanted: list[dict], video_sha: str | None) -> bool:
    """The pictures of the reads already there, for these reads and this video."""
    have = sorted(row.get('decoded_frame_index') for row in manifest)
    pictures = sorted(directory.glob(f'{READ_PREFIX}_*.png'))
    return (have == sorted(read['decoded'] for read in wanted) and len(pictures) == len(manifest)
            and all(row.get('source_video_sha256') == video_sha for row in manifest)
            and all(Path(row.get('image_path', '')).is_file() for row in manifest))


def extract_reads(root: Path, route: str, *, ffmpeg: str | None = None, ffprobe: str | None = None,
                  scan: bool = True, workers: int = 4) -> dict:
    """Write the pictures of the comma's reads of a drive (see the module's text); returns what was done. scan: read
    the drive's logs first when Drive review has not (or not with these files); workers: segments at once."""
    from nnslr_tools import drive_logs, drive_review
    from nnslr_tools.media import extract_frames, extracted_frame_rows
    from nnslr_tools.route_io import discover_segment_dirs
    from nnslr_tools.sync import SLOW_ONLY_MARK
    root = Path(root)
    frames_dir = root / 'derived/frames' / route
    if not frames_dir.is_dir():
        raise FileNotFoundError(f'no pictures of {route}; run: nnslr extract-frames --route {route}')
    store = drive_review.DriveReviewStore(root)
    state = store.scan_state(route)['state']
    if scan and (state in ('not_scanned', 'outdated', 'failed')
                 or (state == 'no_logs' and drive_logs.discover(root, route))):
        state = store.scan(route)['state']
    merged = store.merged(route)
    reads = read_frames(merged) if merged else {}
    videos = {identity.segment_index: directory for identity, directory in discover_segment_dirs(root / 'raw/routes' / route)}
    summary = {'route': route, 'logs': state, 'reads': sum(len(items) for items in reads.values()),
               'added': 0, 'kept': 0, 'segments': {}}

    def segment_reads(directory: Path) -> tuple[int, str, int]:
        """('added' or 'kept', how many) for a segment's folder of pictures; ('none', 0) without reads to add."""
        segment = int(directory.name)
        pictures = _rows(directory / 'frames.jsonl')
        source = videos.get(segment)
        wanted = wanted_reads(reads.get(segment, []), pictures, bool(source and (source / SLOW_ONLY_MARK).exists()))
        manifest_path = directory / READS_MANIFEST
        manifest = _rows(manifest_path)
        video = source / 'fcamera.hevc' if source else None
        if not wanted or video is None or not video.is_file():
            for path in directory.glob(f'{READ_PREFIX}_*.png'):
                path.unlink()
            manifest_path.unlink(missing_ok=True)
            return segment, 'none', 0
        video_sha = pictures[0].get('source_video_sha256') if pictures else None
        if _current(directory, manifest, wanted, video_sha):
            return segment, 'kept', len(manifest)
        alignment = root / 'derived/alignment' / route / f'{segment}.jsonl'
        alignment = alignment if alignment.is_file() else None
        result = extract_frames(video, directory, ffmpeg=ffmpeg, ffprobe=ffprobe, overwrite=True,
                                decoded=[read['decoded'] for read in wanted], prefix=READ_PREFIX,
                                index_base=READ_INDEX_BASE, alignment=alignment, route_id=route, segment_index=segment)
        by_decoded = {read['decoded']: read for read in wanted}
        rows = []
        for row in extracted_frame_rows(result, route_id=route, segment_index=segment, alignment=alignment):
            if not isinstance(row.get('media_time_s'), (int, float)):
                # Without its time the frame would sit at the end of the segment: left out.
                Path(row['image_path']).unlink(missing_ok=True)
                continue
            read = by_decoded[row['decoded_frame_index']]
            rows.append({**row, 'comma_read_kph': read['kph'], 'comma_read_mono_ns': read['t_ns']})
        temporary = manifest_path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        temporary.replace(manifest_path)
        return segment, 'added', len(rows)

    # Several segments at once, as nnslr extract-frames: ffprobe reads a whole video on one core.
    folders = sorted((path for path in frames_dir.iterdir() if path.name.isdigit()), key=lambda p: int(p.name))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for segment, done, count in pool.map(segment_reads, folders):
            if done != 'none':
                summary[done] += count
            if done == 'added':
                summary['segments'][segment] = count
    return summary
