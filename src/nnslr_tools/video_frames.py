"""Every frame of the road video in Sign review, not only the pictures at 1 a second.

The timeline keeps one cell per second; the arrows of the page step through every frame of fcamera.hevc (20 a
second). The frames between the pictures are taken out only when the page asks for them, a GOP at a time (a keyframe
and the frames up to the next one, decoded on their own: a third of a second, the same pixels as a decode of the whole
video), as JPEG previews in a cache under derived/video_cache that keeps at most CACHE_BYTES (the least recently shown
go first). Nothing of that is a review or a row of the report. A frame asked for is given as soon as ffmpeg has written
it, not when its whole GOP is (0.4 s instead of 1.4 s for the first frame of a GOP at 2688 x 1520); the page then asks
for the next GOP to be decoded in the background (read_ahead), and the cache is pruned outside the request (prune_soon).

A frame between the seconds becomes one only when the reviewer confirms it, with boxes or as no sign (materialize):
its picture is
taken out of the video as the others are (``video_<output index>.png`` next to them, listed in ``video_frames.jsonl``
of the segment, the rows of frames.jsonl), with the key scheme of the frames of the comma's reads: output index
READ_INDEX_BASE + decoded frame index. The keys at 1 a second (0..59) never change, so the reviews saved stay where
they are, and a frame that is already a row of the report (a picture at 1 a second, a read) keeps its key. The report
shows the frames saved this way (added_rows) and the next Propose again proposes on them (discover_frames).
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
import time
from functools import lru_cache
from pathlib import Path

from nnslr_tools.media import (FRAME_SIZE, FULL_SIZE, REDUCE_FILTER, MediaToolError, extract_frames,
                               extracted_frame_rows, full_picture, remove_picture, video_dimensions)
from nnslr_tools.sign_reads import READ_INDEX_BASE

WIDE_VIDEO = 'ecamera.hevc'
VIDEO_MANIFEST = 'video_frames.jsonl'
VIDEO_PREFIX = 'video'
CACHE_DIR = 'derived/video_cache'
# About 16 minutes of video at full definition (2688 x 1520: 212 kB a frame, 254 MB a minute), 50 at 1344 x 760.
CACHE_BYTES = 4 << 30
# A prune leaves the cache at this share of CACHE_BYTES, so that the next decodes do not prune again at once.
PRUNE_TO = 0.9
# How often a request looks for its frame while ffmpeg writes the frames of its GOP.
POLL_S = 0.005
# The niceness of a GOP decoded ahead, before any page asks for it (the server's requests run at 0, the trainings at 10).
AHEAD_NICE = 5
# GOPs waiting to be decoded ahead: the newest first, the oldest forgotten.
AHEAD_KEPT = 4
# ffmpeg's JPEG quality (2 best .. 31): about the quality 92 of the report's previews.
JPEG_QUALITY = '3'
# ffmpeg processes decoding at once, and the threads of each: one decode of a GOP at 2688 x 1520 takes about 700 MB with
# ffmpeg's own threads (one per core), 450 MB with 4, as fast. Without a limit a page playing or scrubbing started 55 at
# once and the container ran out of memory (#60).
MAX_DECODES = 4
DECODE_THREADS = '4'
_decode_slots = threading.BoundedSemaphore(MAX_DECODES)
# The parameter sets of fcamera.hevc are at its start, before the first slice: never more than this.
HEADER_BYTES = 1 << 16
# A frame between the seconds taken out ahead (prepare_soon) waits in <segment folder>/.prepared-<decoded frame> until
# its save moves its pictures beside the others (materialize); one never saved goes after this long.
PREPARED_PREFIX = '.prepared-'
PREPARED_KEPT_S = 3600

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


def _cache_dir(root: Path, route: str, segment: int, video: Path, wide: bool = False) -> Path:
    """The cache folder of a segment (<segment>-wide for the wide camera), emptied when its video is not the one the
    previews came from."""
    folder = Path(root) / CACHE_DIR / route / (str(int(segment)) + ('-wide' if wide else ''))
    stat = video.stat()
    identity = {'video': str(video.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    stamp = folder / 'video.json'

    def current() -> bool:
        try:
            return json.loads(stamp.read_text()) == identity
        except (OSError, ValueError):
            return False
    if not current():
        with _lock('cache', route, int(segment), wide):
            # Looked at again: another request may have emptied it a moment ago, and its decode be writing there now.
            if not current():
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True, exist_ok=True)
                stamp.write_text(json.dumps(identity))
    return folder


class _Decode:
    """A GOP being decoded into its cache folder by ffmpeg, its frames appearing there one by one (written to a
    temporary name and renamed: a frame in the folder is whole)."""

    def __init__(self, folder: Path, first: int, last: int):
        self.folder, self.first, self.last = folder, first, last
        self.done = threading.Event()
        self.error = ''

    def wait_for(self, decoded: int) -> Path:
        """The frame once ffmpeg has written it, without waiting for the rest of the GOP."""
        wanted = self.folder / f'{decoded}.jpg'
        while not wanted.is_file():
            if self.done.wait(POLL_S) and not wanted.is_file():
                raise VideoFrameError(self.error or f'frame {decoded} not decoded')
        return wanted


# The GOPs being decoded, by (cache folder, first frame).
_decodes: dict[tuple[str, int], _Decode] = {}
# The bytes of the cache of each data root as last counted (prune) plus the GOPs decoded since; absent: not counted yet.
_cache_total: dict[str, int] = {}
_prune_guard = threading.Lock()


def _gop_cached(folder: Path, first: int, last: int) -> bool:
    return (folder / f'{first}.jpg').is_file() and (folder / f'{last}.jpg').is_file()


def _decode(root: Path, video: Path, folder: Path, first: int, last: int, wanted: Path | None, *,
            ahead: bool = False) -> _Decode | None:
    """The decode of a GOP into its cache folder: the one on its way, or one started now (ahead: niced). None when the
    frame wanted (without one: the GOP) is already there."""
    key = (str(folder), first)
    with _locks_guard:
        job = _decodes.get(key)
        if job is not None:
            return job
        if wanted.is_file() if wanted is not None else _gop_cached(folder, first, last):
            return None
        job = _decodes[key] = _Decode(folder, first, last)
    try:
        data = gop_bytes(video, first, last)
    except (OSError, VideoFrameError) as exc:
        job.error = f'GOP {first}..{last} not decoded: {exc}'
        with _locks_guard:
            _decodes.pop(key, None)
        job.done.set()
        return job
    threading.Thread(target=_finish, args=(root, key, job, data, ahead), name='video-gop', daemon=True).start()
    return job


def _finish(root: Path, key: tuple[str, int], job: _Decode, data: bytes, ahead: bool) -> None:
    """Feeds ffmpeg its GOP once one of the MAX_DECODES slots is free and waits for it: every frame of the GOP
    written, or the decode failed (its frames half written removed, the next request tries again). Then the cache is
    pruned if the frames made it too big."""
    with _decode_slots:
        try:
            # Written in the folder (not a path in the pattern, which may hold a %): <first>.jpg, <first + 1>.jpg...
            process = subprocess.Popen([*(['nice', '-n', str(AHEAD_NICE)] if ahead else []), 'ffmpeg', '-v', 'error',
                                        '-y', '-threads', DECODE_THREADS, '-f', 'hevc', '-i', '-', '-vsync', '0',
                                        '-q:v', JPEG_QUALITY, '-atomic_writing', '1', '-start_number', str(job.first),
                                        '%d.jpg'], cwd=job.folder, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
            _out, errors = process.communicate(data)
            message = errors.decode(errors='replace').strip()
        except OSError as exc:
            process = None
            message = str(exc)
    sizes = []
    for number in range(job.first, job.last + 1):
        try:
            sizes.append((job.folder / f'{number}.jpg').stat().st_size)
        except OSError:
            pass
    if process is None or process.returncode != 0 or len(sizes) != job.last - job.first + 1:
        job.error = f'GOP {job.first}..{job.last} not decoded: ' + message[:200]
        for number in range(job.first, job.last + 1):
            (job.folder / f'{number}.jpg.tmp').unlink(missing_ok=True)
    with _locks_guard:
        total = _cache_total.get(str(root))
        if total is not None:
            total = _cache_total[str(root)] = total + sum(sizes)
    if total is None or total > CACHE_BYTES:
        prune_soon(root)
    with _locks_guard:
        _decodes.pop(key, None)
    job.done.set()


def preview(root: Path, route: str, segment: int, decoded: int, *, wide: bool = False) -> Path:
    """The JPEG of a frame of the road video (wide: of the wide road camera, the same frame), its GOP taken out when
    the cache does not have it: given as soon as ffmpeg has written that frame."""
    root = Path(root)
    video = wide_video_of(root, route, segment) if wide else video_of(root, route, segment)
    first, last = gop_of(video, decoded)
    folder = _cache_dir(root, route, segment, video, wide)
    wanted = folder / f'{decoded}.jpg'
    for _attempt in range(2):
        job = _decode(root, video, folder, first, last, wanted)
        if job is not None:
            job.wait_for(decoded)
        try:
            os.utime(wanted)
            return wanted
        except FileNotFoundError:
            # Pruned between the decode and now: decoded again.
            continue
    raise VideoFrameError(f'frame {decoded} not kept in the cache')


def gop_started(root: Path, route: str, segment: int, decoded: int, *, wide: bool = False) -> tuple[int, int]:
    """(first, last) of the GOP of a frame, its decode started: given once ffmpeg has written the GOP's first frame,
    not the rest (the page asks for its frames next, preview). VideoFrameError for a GOP ffmpeg cannot decode at all:
    ffmpeg gives up on one in about 0.6 s and writes nothing, where it writes the first frame of a good GOP in 0.3 s at
    2688 x 1520."""
    root = Path(root)
    video = wide_video_of(root, route, segment) if wide else video_of(root, route, segment)
    first, last = gop_of(video, decoded)
    job = _decode(root, video, _cache_dir(root, route, segment, video, wide), first, last, None)
    if job is not None:
        job.wait_for(first)
    return first, last


# The GOPs to decode ahead, by (cache folder, first frame), the newest last; one decoded at a time.
_ahead: dict[tuple[str, int], tuple] = {}
_ahead_ready = threading.Condition()
_ahead_state = {'thread': None, 'busy': False}


def read_ahead(root: Path, route: str, segment: int, decoded: int, *, wide: bool = False) -> None:
    """After a page was given a frame: the next GOP of its video decoded in the background (niced), and the one before
    while the frame is in the first third of its GOP, so that a step into either finds its frames ready. Only GOPs the
    cache does not have; nothing is waited for."""
    root = Path(root)
    try:
        video = wide_video_of(root, route, segment) if wide else video_of(root, route, segment)
        first, last = gop_of(video, decoded)
        count = frame_count(video)
        folder = _cache_dir(root, route, segment, video, wide)
    except (VideoFrameError, OSError):
        return
    wanted = []
    if decoded - first < (last - first + 1) / 3 and first > 0:
        wanted.append(gop_of(video, first - 1))
    if last + 1 < count:
        wanted.append(gop_of(video, last + 1))
    with _ahead_ready:
        for gop in wanted:
            if _gop_cached(folder, *gop):
                continue
            key = (str(folder), gop[0])
            _ahead.pop(key, None)
            _ahead[key] = (root, video, folder, *gop)
        while len(_ahead) > AHEAD_KEPT:
            del _ahead[next(iter(_ahead))]
        if _ahead_state['thread'] is None:
            _ahead_state['thread'] = threading.Thread(target=_ahead_loop, name='video-ahead', daemon=True)
            _ahead_state['thread'].start()
        _ahead_ready.notify()


def _ahead_loop() -> None:
    while True:
        with _ahead_ready:
            while not _ahead:
                _ahead_state['busy'] = False
                _ahead_ready.wait()
            _ahead_state['busy'] = True
            _key, (root, video, folder, first, last) = _ahead.popitem()
        try:
            job = _decode(root, video, folder, first, last, None, ahead=True)
            if job is not None:
                job.done.wait()
        except Exception:  # noqa: BLE001 - a GOP not decoded ahead is decoded when a page asks for it
            pass


def settle(timeout: float = 30.0) -> bool:
    """Waits until no GOP is being decoded or waits to be decoded ahead, nor the cache pruned (the tests count the
    files of the cache). False after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _ahead_ready:
            queued = bool(_ahead) or _ahead_state['busy']
        with _locks_guard:
            running = list(_decodes.values())
        if not queued and not running and not _prune_guard.locked():
            return True
        for job in running:
            job.done.wait(0.05)
        time.sleep(0.01)
    return False


def prune_soon(root: Path) -> None:
    """prune in the background, one at a time, down to PRUNE_TO of the budget: never in a request."""
    if not _prune_guard.acquire(blocking=False):
        return

    def run() -> None:
        try:
            prune(root, CACHE_BYTES, low=int(CACHE_BYTES * PRUNE_TO))
        except Exception:  # noqa: BLE001 - the next decode prunes again
            pass
        finally:
            _prune_guard.release()
    threading.Thread(target=run, name='video-cache-prune', daemon=True).start()


def prune(root: Path, budget: int | None = None, low: int | None = None) -> None:
    """Drop the least recently shown previews until the cache is within its budget (CACHE_BYTES unless given; when it is
    over it, down to `low`)."""
    budget = CACHE_BYTES if budget is None else budget
    files = []
    for path in (Path(root) / CACHE_DIR).glob('*/*/*.jpg'):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((stat.st_mtime_ns, stat.st_size, path))
    total = sum(size for _time, size, _path in files)
    target = budget if low is None or total <= budget else min(low, budget)
    for _time, size, path in sorted(files):
        if total <= target:
            break
        path.unlink(missing_ok=True)
        total -= size
    with _locks_guard:
        _cache_total[str(Path(root))] = total


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


def _extract(root: Path, route: str, segment: int, decoded: int, into: Path) -> dict:
    """Its manifest row of a frame between the seconds, its PNG taken out of the video into the new folder ``into`` as
    the pictures at 1 a second are (nnslr_tools.media.extract_frames)."""
    video = video_of(root, route, segment)
    first, last = gop_of(video, decoded)
    alignment = root / 'derived/alignment' / route / f'{int(segment)}.jsonl'
    alignment = alignment if alignment.is_file() else None
    # Only its GOP is decoded (0.3 s, the whole video takes 5), with the timings read when its frames were shown.
    result = extract_frames(video, into, decoded=[decoded], prefix=VIDEO_PREFIX, index_base=READ_INDEX_BASE,
                            alignment=alignment, route_id=route, segment_index=int(segment), timings=timings_of(video),
                            gop=(first, gop_bytes(video, first, last)))
    made = extracted_frame_rows(result, route_id=route, segment_index=int(segment), alignment=alignment)[0]
    if not isinstance(made.get('media_time_s'), (int, float)):
        raise VideoFrameError('the frame has no time in the video')
    return made


def _move_picture(made: dict, directory: Path) -> None:
    """The picture of ``made`` (and its frame at full definition) moved into ``directory``, its image_path with it."""
    picture = Path(made['image_path'])
    final = directory / picture.name
    picture.replace(final)
    if full_picture(picture).is_file():
        full_picture(picture).replace(full_picture(final))
    made['image_path'] = str(final)


def prepare(root: Path, route: str, segment: int, decoded: int) -> None:
    """A frame between the seconds the reviewer works on (Apply previous, Fit box) taken out of the video now, before
    its save asks for it: the save then only moves its pictures (materialize), instead of waiting a second or more
    for ffmpeg. Nothing is a row of the report until then."""
    root = Path(root)
    directory = root / 'derived/frames' / route / str(int(segment))
    folder = directory / f'{PREPARED_PREFIX}{int(decoded)}'
    with _lock('prepare', route, int(segment), int(decoded)):
        if (folder / 'row.json').is_file() or (directory / VIDEO_MANIFEST).is_file() and any(
                row.get('decoded_frame_index') == decoded for row in _rows(directory / VIDEO_MANIFEST)):
            return
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.video-', dir=directory) as temporary:
            staging = Path(temporary) / 'frame'
            made = _extract(root, route, segment, decoded, staging)
            made['image_path'] = Path(made['image_path']).name
            (staging / 'row.json').write_text(json.dumps(made, sort_keys=True))
            shutil.rmtree(folder, ignore_errors=True)
            staging.replace(folder)
    # Those never saved go.
    now = time.time()
    for old in directory.glob(PREPARED_PREFIX + '*'):
        try:
            if now - old.stat().st_mtime > PREPARED_KEPT_S:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass


def prepare_soon(root: Path, route: str, segment: int, decoded: int) -> None:
    """prepare outside the request; a frame it cannot take out is taken out again by its save, which says why."""
    def run() -> None:
        try:
            prepare(root, route, segment, decoded)
        except (VideoFrameError, MediaToolError, OSError, ValueError):
            pass
    threading.Thread(target=run, name='video-prepare', daemon=True).start()


def _take_prepared(directory: Path, route: str, segment: int, decoded: int) -> dict | None:
    """The manifest row of the frame prepare took out, its pictures moved into ``directory``; None without one. Waits
    for a prepare at work on it."""
    folder = directory / f'{PREPARED_PREFIX}{int(decoded)}'
    with _lock('prepare', route, int(segment), int(decoded)):
        try:
            made = json.loads((folder / 'row.json').read_text())
            made['image_path'] = str(folder / made['image_path'])
            if not isinstance(made, dict) or not Path(made['image_path']).is_file():
                return None
            _move_picture(made, directory)
            return made
        except (OSError, ValueError, KeyError, TypeError):
            return None
        finally:
            shutil.rmtree(folder, ignore_errors=True)


def materialize(root: Path, route: str, segment: int, decoded: int) -> tuple[dict, bool]:
    """(manifest row, made now) of a frame between the seconds: its PNG taken out of the video as the pictures at 1
    a second are (nnslr_tools.media.extract_frames), or by prepare before, added to video_frames.jsonl of its
    segment."""
    root = Path(root)
    video_of(root, route, segment)
    directory = root / 'derived/frames' / route / str(int(segment))
    manifest = directory / VIDEO_MANIFEST
    with _lock('manifest', route, int(segment)):
        rows = _rows(manifest)
        for row in rows:
            if row.get('decoded_frame_index') == decoded and Path(row.get('image_path', '')).is_file():
                return row, False
        made = _take_prepared(directory, route, segment, decoded)
        if made is None:
            # Taken out on its own folder: extract_frames replaces every picture with its prefix in the folder it writes.
            with tempfile.TemporaryDirectory(prefix='.video-', dir=directory) as temporary:
                made = _extract(root, route, segment, decoded, Path(temporary))
                _move_picture(made, directory)
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
