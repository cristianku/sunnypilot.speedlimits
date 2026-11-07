"""Every frame of the road video in Sign review, not only the pictures at 1 a second.

The timeline keeps one cell per second; the arrows of the page step through every frame of fcamera.hevc (20 a
second). The frames between the pictures are taken out only when the page asks for them, a GOP at a time (a keyframe
and the frames up to the next one, decoded on their own: a third of a second, the same pixels as a decode of the whole
video), as JPEG previews in a cache under derived/video_cache that keeps at most CACHE_BYTES (the least recently shown
go first). Nothing of that is a review or a row of the report.

A frame between the seconds becomes one only when the reviewer confirms it, with boxes or as no sign (materialize):
its picture is
taken out of the video as the others are (``video_<output index>.png`` next to them, listed in ``video_frames.jsonl``
of the segment, the rows of frames.jsonl), with the key scheme of the frames of the comma's reads: output index
READ_INDEX_BASE + decoded frame index. The keys at 1 a second (0..59) never change, so the reviews saved stay where
they are, and a frame that is already a row of the report (a picture at 1 a second, a read) keeps its key. The report
shows the frames saved this way (added_rows) and the next Propose again proposes on them (discover_frames). In a segment
kept only at a car park's speed (the start or the end of a drive), only the seconds kept.
Only the camera: no map, no logs.
"""
from __future__ import annotations

import bisect
import json
import os
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from nnslr_tools.media import (FRAME_SIZE, FULL_SIZE, REDUCE_FILTER, MediaToolError, extract_frames,
                               extracted_frame_rows, full_picture, remove_picture, video_dimensions)
from nnslr_tools.sign_reads import CAMERA_FPS, READ_INDEX_BASE, frame_second

WIDE_VIDEO = 'ecamera.hevc'
VIDEO_MANIFEST = 'video_frames.jsonl'
VIDEO_PREFIX = 'video'
CACHE_DIR = 'derived/video_cache'
CACHE_BYTES = 1 << 30
# ffmpeg's JPEG quality (2 best .. 31): about the quality 92 of the report's previews.
JPEG_QUALITY = '3'
# The parameter sets of fcamera.hevc are at its start, before the first slice: never more than this.
HEADER_BYTES = 1 << 16

_locks: dict[tuple, threading.Lock] = {}
_locks_guard = threading.Lock()
# The frame timings of the videos last saved from (timings_of), oldest first.
TIMINGS_KEPT = 16
_timings: dict[tuple, tuple] = {}
# The GOPs last decoded to NV12 (nv12_frame), oldest first: about 30 MB each for the road camera, 180 MB as recorded at
# full definition.
NV12_GOPS_KEPT = 2
_nv12: dict[tuple, bytes] = {}


class VideoFrameError(ValueError):
    """A frame the video does not have, or a video that cannot be read frame by frame."""


def _lock(*key) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def video_of(root: Path, route: str, segment: int) -> Path:
    """The road video of a segment (data/raw/routes/<route>/<segment>/fcamera.hevc)."""
    video = Path(root) / 'raw/routes' / route / str(int(segment)) / 'fcamera.hevc'
    if not video.is_file():
        raise VideoFrameError(f'no road video for segment {segment}')
    return video


def wide_video_of(root: Path, route: str, segment: int) -> Path:
    """The wide road camera's video of a segment (ecamera.hevc), only when it has as many frames as the road video:
    loggerd starts and cuts both on the same frame id, so its frame n is the road video's frame n (the same frameId at
    each of the 1200 indices of narrowRoadEncodeIdx and wideRoadEncodeIdx, 00000165 minute 5, 3 Nov 2026)."""
    video = Path(root) / 'raw/routes' / route / str(int(segment)) / WIDE_VIDEO
    if not video.is_file():
        raise VideoFrameError(f'no wide road video for segment {segment}')
    if frame_count(video) != frame_count(video_of(root, route, segment)):
        raise VideoFrameError(f'the wide road video of segment {segment} has not the frames of the road video')
    return video


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@lru_cache(maxsize=512)
def _index(path: str, mtime_ns: int, size: int) -> tuple[tuple[int, ...], tuple[int, ...], bytes]:
    """(byte offset of every packet, decoded indices of the keyframes, the parameter sets) of a raw HEVC video, whose
    packets are its frames in decoded order (no B-frames: the comma's encoder never makes them)."""
    del mtime_ns, size
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                            'packet=pos,flags:stream=has_b_frames', '-of', 'json', path],
                           capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise VideoFrameError('the road video cannot be read: ' + probe.stderr.strip()[:200])
    found = json.loads(probe.stdout or '{}')
    if any(stream.get('has_b_frames') for stream in found.get('streams') or ()):
        raise VideoFrameError('the road video has B-frames: it cannot be taken out GOP by GOP')
    packets = found.get('packets') or []
    offsets = tuple(int(packet['pos']) for packet in packets)
    keys = tuple(number for number, packet in enumerate(packets) if 'K' in packet.get('flags', ''))
    if not offsets or not keys or keys[0] != 0 or list(offsets) != sorted(offsets):
        raise VideoFrameError('the road video has no keyframe at its start')
    with open(path, 'rb') as source:
        data = source.read(min(HEADER_BYTES, offsets[1] if len(offsets) > 1 else HEADER_BYTES))
    # The parameter sets (VPS, SPS, PPS: NAL types 32..34) and SEI before the first slice (NAL type below 32).
    start = 0
    while True:
        at = data.find(b'\x00\x00\x01', start)
        if at < 0 or at + 3 >= len(data):
            raise VideoFrameError('the road video has no parameter sets at its start')
        if (data[at + 3] >> 1) & 0x3f < 32:
            return offsets, keys, data[:at - 1 if at and data[at - 1] == 0 else at]
        start = at + 3


def index_of(video: Path) -> tuple[tuple[int, ...], tuple[int, ...], bytes]:
    stat = video.stat()
    return _index(str(video), stat.st_mtime_ns, stat.st_size)


def frame_count(video: Path) -> int:
    return len(index_of(video)[0])


def timings_of(video: Path) -> tuple:
    """probe_frame_timestamps of a video, kept for the last TIMINGS_KEPT videos. Read from the packets: a save waits
    for them, and decoding the whole video takes 35 s for a minute of the road camera."""
    from nnslr_tools.media import probe_frame_timestamps
    stat = video.stat()
    key = (str(video), stat.st_mtime_ns, stat.st_size)
    with _lock('timings', str(video)):
        found = _timings.get(key)
        if found is None:
            found = tuple(probe_frame_timestamps(video, decode=False))
            with _locks_guard:
                _timings[key] = found
                while len(_timings) > TIMINGS_KEPT:
                    del _timings[next(iter(_timings))]
        return found


def read_timings_soon(video: Path) -> None:
    """timings_of in the background while the reviewer looks at the frames, so that a save does not wait for it."""
    stat = video.stat()
    if (str(video), stat.st_mtime_ns, stat.st_size) in _timings or _lock('timings', str(video)).locked():
        return

    def read() -> None:
        try:
            timings_of(video)
        except Exception:  # noqa: BLE001 - the save reads them again and says what went wrong
            pass
    threading.Thread(target=read, name='video-timings', daemon=True).start()


def gop_of(video: Path, decoded: int) -> tuple[int, int]:
    """(first, last) decoded index of the GOP that holds a frame."""
    offsets, keys, _header = index_of(video)
    if not 0 <= decoded < len(offsets):
        raise VideoFrameError(f'frame {decoded} is not in the video ({len(offsets)} frames)')
    at = bisect.bisect_right(keys, decoded) - 1
    last = keys[at + 1] - 1 if at + 1 < len(keys) else len(offsets) - 1
    return keys[at], last


def gop_bytes(video: Path, first: int, last: int) -> bytes:
    """The parameter sets and the frames first..last of a raw HEVC video (a GOP), to decode on their own."""
    offsets, _keys, header = index_of(video)
    with open(video, 'rb') as source:
        source.seek(offsets[first])
        return header + source.read(offsets[last + 1] - offsets[first] if last + 1 < len(offsets) else -1)


@lru_cache(maxsize=512)
def _size(path: str, mtime_ns: int, size: int) -> tuple[int, int]:
    del mtime_ns, size
    try:
        return video_dimensions(Path(path))
    except MediaToolError as error:
        raise VideoFrameError(f'the size of the road video cannot be read: {error.stderr[:200] or error}') from None


def video_size(video: Path) -> tuple[int, int]:
    """(width, height) of a video."""
    stat = video.stat()
    return _size(str(video), stat.st_mtime_ns, stat.st_size)


def nv12_frame(video: Path, decoded: int, *, native: bool = False) -> tuple[bytes, int, int]:
    """(NV12 bytes, width, height) of a frame of the road video, the road camera's pixel format (the one speedvisiond
    reads): its GOP decoded on its own (gop_bytes), the last NV12_GOPS_KEPT GOPs kept. A video at full definition comes
    reduced to the size the cutter reads on the car, or as recorded with ``native`` (what speedvisiond crops from)."""
    first, last = gop_of(video, decoded)
    width, height = video_size(video)
    reduce = (width, height) == FULL_SIZE and not native
    if reduce:
        width, height = FRAME_SIZE
    size = width * height * 3 // 2
    stat = video.stat()
    key = (str(video), stat.st_mtime_ns, stat.st_size, first, reduce)
    with _lock('nv12', str(video), first):
        with _locks_guard:
            frames = _nv12.get(key)
        if frames is None:
            decode = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', '-', '-vsync', '0',
                                     *(['-vf', REDUCE_FILTER] if reduce else []), '-pix_fmt', 'nv12',
                                     '-f', 'rawvideo', '-'], input=gop_bytes(video, first, last),
                                    capture_output=True, check=False)
            frames = decode.stdout
            if decode.returncode != 0 or len(frames) != (last - first + 1) * size:
                raise VideoFrameError(f'GOP {first}..{last} not decoded: '
                                      + decode.stderr.decode(errors='replace').strip()[:200])
            with _locks_guard:
                _nv12[key] = frames
                while len(_nv12) > NV12_GOPS_KEPT:
                    del _nv12[next(iter(_nv12))]
    at = (decoded - first) * size
    return frames[at:at + size], width, height


def kept_seconds(root: Path, route: str, segment: int) -> list[int] | None:
    """The seconds of a segment kept only at a car park's speed (nnslr_tools.sync), None for a whole segment."""
    from nnslr_tools.sync import SLOW_ONLY_MARK
    if not (Path(root) / 'raw/routes' / route / str(int(segment)) / SLOW_ONLY_MARK).exists():
        return None
    pictures = _rows(Path(root) / 'derived/frames' / route / str(int(segment)) / 'frames.jsonl')
    return sorted({int(frame_second(row)) for row in pictures})


def _cache_dir(root: Path, route: str, segment: int, video: Path, wide: bool = False) -> Path:
    """The cache folder of a segment (<segment>-wide for the wide camera), emptied when its video is not the one the
    previews came from."""
    folder = Path(root) / CACHE_DIR / route / (str(int(segment)) + ('-wide' if wide else ''))
    stat = video.stat()
    identity = {'video': str(video.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    stamp = folder / 'video.json'
    try:
        current = json.loads(stamp.read_text()) == identity
    except (OSError, ValueError):
        current = False
    if not current:
        with _lock('cache', route, int(segment), wide):
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
            stamp.write_text(json.dumps(identity))
    return folder


def preview(root: Path, route: str, segment: int, decoded: int, *, wide: bool = False) -> Path:
    """The JPEG of a frame of the road video (wide: of the wide road camera, the same frame), its GOP taken out when
    the cache does not have it."""
    root = Path(root)
    video = wide_video_of(root, route, segment) if wide else video_of(root, route, segment)
    first, last = gop_of(video, decoded)
    folder = _cache_dir(root, route, segment, video, wide)
    wanted = folder / f'{decoded}.jpg'
    with _lock('gop', route, int(segment), first, wide):
        if not wanted.is_file():
            with tempfile.TemporaryDirectory(prefix='.gop-', dir=folder) as temporary:
                staging = Path(temporary)
                decode = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', '-', '-vsync', '0',
                                         '-q:v', JPEG_QUALITY, str(staging / 'f_%04d.jpg')],
                                        input=gop_bytes(video, first, last), capture_output=True, check=False)
                made = sorted(staging.glob('f_*.jpg'))
                if decode.returncode != 0 or len(made) != last - first + 1:
                    raise VideoFrameError(f'GOP {first}..{last} not decoded: '
                                          + decode.stderr.decode(errors='replace').strip()[:200])
                for number, path in enumerate(made):
                    path.replace(folder / f'{first + number}.jpg')
            prune(root)
        os.utime(wanted)
    return wanted


def prune(root: Path, budget: int = CACHE_BYTES) -> None:
    """Drop the least recently shown previews until the cache is within its budget."""
    files = []
    for path in (Path(root) / CACHE_DIR).glob('*/*/*.jpg'):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((stat.st_mtime_ns, stat.st_size, path))
    total = sum(size for _time, size, _path in files)
    for _time, size, path in sorted(files):
        if total <= budget:
            break
        path.unlink(missing_ok=True)
        total -= size


def manifest_stamp(root: Path, route: str) -> tuple:
    """Changes when a frame between the seconds is saved or dropped: the report's cache follows it."""
    stamp = []
    for path in sorted((Path(root) / 'derived/frames' / route).glob('*/' + VIDEO_MANIFEST)):
        try:
            stamp.append((path.parent.name, path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(stamp)


def added_rows(root: Path, route: str) -> list[dict]:
    """The manifest rows of the frames between the seconds saved so far, every segment."""
    rows = []
    for path in sorted((Path(root) / 'derived/frames' / route).glob('*/' + VIDEO_MANIFEST)):
        rows.extend(_rows(path))
    return rows


def materialize(root: Path, route: str, segment: int, decoded: int) -> tuple[dict, bool]:
    """(manifest row, made now) of a frame between the seconds: its PNG taken out of the video as the pictures at 1
    a second are (nnslr_tools.media.extract_frames), added to video_frames.jsonl of its segment."""
    root = Path(root)
    video = video_of(root, route, segment)
    first, last = gop_of(video, decoded)
    seconds = kept_seconds(root, route, segment)
    if seconds is not None and int(decoded / CAMERA_FPS) not in seconds:
        raise VideoFrameError('this second of the segment was not kept (only a car park\'s speed)')
    directory = root / 'derived/frames' / route / str(int(segment))
    manifest = directory / VIDEO_MANIFEST
    with _lock('manifest', route, int(segment)):
        rows = _rows(manifest)
        for row in rows:
            if row.get('decoded_frame_index') == decoded and Path(row.get('image_path', '')).is_file():
                return row, False
        alignment = root / 'derived/alignment' / route / f'{int(segment)}.jsonl'
        alignment = alignment if alignment.is_file() else None
        # Taken out on its own folder: extract_frames replaces every picture with its prefix in the folder it writes.
        with tempfile.TemporaryDirectory(prefix='.video-', dir=directory) as temporary:
            staging = Path(temporary)
            # Only its GOP is decoded (0.3 s, the whole video takes 5), with the timings read when its frames were shown.
            result = extract_frames(video, staging, decoded=[decoded], prefix=VIDEO_PREFIX,
                                    index_base=READ_INDEX_BASE, alignment=alignment, route_id=route,
                                    segment_index=int(segment), timings=timings_of(video),
                                    gop=(first, gop_bytes(video, first, last)))
            made = extracted_frame_rows(result, route_id=route, segment_index=int(segment), alignment=alignment)[0]
            if not isinstance(made.get('media_time_s'), (int, float)):
                raise VideoFrameError('the frame has no time in the video')
            final = directory / Path(made['image_path']).name
            Path(made['image_path']).replace(final)
            if full_picture(made['image_path']).is_file():
                full_picture(made['image_path']).replace(full_picture(final))
        made['image_path'] = str(final)
        rows = [row for row in rows if row.get('decoded_frame_index') != decoded] + [made]
        temporary = manifest.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        temporary.replace(manifest)
    return made, True


def forget(root: Path, route: str, segment: int, decoded: int) -> None:
    """Undo a materialize whose save was refused: no frame between the seconds without a review."""
    directory = Path(root) / 'derived/frames' / route / str(int(segment))
    manifest = directory / VIDEO_MANIFEST
    with _lock('manifest', route, int(segment)):
        rows = _rows(manifest)
        kept = [row for row in rows if row.get('decoded_frame_index') != decoded]
        for row in rows:
            if row.get('decoded_frame_index') == decoded:
                remove_picture(row.get('image_path', ''))
        if not kept:
            manifest.unlink(missing_ok=True)
        elif len(kept) != len(rows):
            temporary = manifest.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in kept))
            temporary.replace(manifest)
