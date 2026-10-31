"""A drive as a video the browser can play, for the Drive view of Results (web/src/control/DriveView.vue).

The comma records the road camera of every minute as raw HEVC (data/raw/routes/<route>/<segment>/fcamera.hevc,
1344 x 760, 20 frames a second), which browsers do not play. Here the minutes of a drive, in order, become one H.264
MP4 960 px wide, with an index of where each minute starts in it: derived/video/<route>.mp4 and <route>.json. The
minutes are encoded side by side (WORKERS ffmpeg at low priority with THREADS threads each: it never slows a training)
and then joined without encoding again, about 3.5 times faster than one ffmpeg over the whole drive. A frame of the reports is image N of a minute, N seconds into
it (frames are sampled one a second), so the page finds it in the video at start(minute) + N.

Made once, the first time the Drive view shows the drive (Panel.drive_video), or with `nnslr drive-video ROUTE`; a
made video is kept until its minutes change. Nothing else reads it: it is only how the site shows a drive.
"""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

FPS = 20
WIDTH = 960
# Minutes encoded at once, and the threads of each ffmpeg.
WORKERS = 8
THREADS = 2
ROUTE = re.compile(r'[0-9a-f]{8}--[0-9a-f]{10}')
# One video made at a time, by this process.
_lock = threading.Lock()
_making: dict[str, dict[str, Any]] = {}


def paths(root: Path, route: str) -> tuple[Path, Path]:
    """The video of a drive and its index."""
    folder = Path(root) / 'derived' / 'video'
    return folder / (route + '.mp4'), folder / (route + '.json')


def minutes(root: Path, route: str) -> list[tuple[int, Path]]:
    """The minutes of a drive with a road video, in order: (segment, fcamera.hevc)."""
    base = Path(root) / 'raw' / 'routes' / route
    found = []
    for folder in base.glob('*'):
        video = folder / 'fcamera.hevc'
        if folder.name.isdigit() and video.is_file():
            found.append((int(folder.name), video))
    return sorted(found)


def _frames(video: Path) -> int:
    """How many frames a raw HEVC minute holds (its packets: nothing is decoded)."""
    answer = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries',
                             'stream=nb_read_packets', '-of', 'csv=p=0', str(video)],
                            capture_output=True, text=True, check=True, timeout=120)
    return int(answer.stdout.strip().split(',')[0] or 0)


def _signature(parts: list[tuple[int, Path]]) -> list[list]:
    """What the video is made of: each minute with its size and time, to know when it must be made again."""
    signature = []
    for segment, video in parts:
        info = video.stat()
        signature.append([segment, info.st_size, int(info.st_mtime)])
    return signature


def index(root: Path, route: str) -> dict | None:
    """The index of a made video, still matching its minutes; None when there is none or it is out of date."""
    video, index_path = paths(root, route)
    try:
        value = json.loads(index_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not video.is_file() or value.get('source') != _signature(minutes(root, route)):
        return None
    return value


def _encode(source: Path, target: Path, done: Callable[[int], None]) -> None:
    """One minute of raw HEVC to H.264 (an MP4 of its own); done(frames so far) while ffmpeg works."""
    command = ['nice', '-n', '15', 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostats', '-progress', 'pipe:2',
               '-y', '-f', 'hevc', '-r', str(FPS), '-i', str(source), '-an', '-vf', f'scale={WIDTH}:-2',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '27', '-g', str(2 * FPS), '-pix_fmt', 'yuv420p',
               '-threads', str(THREADS), str(target)]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=False)
    errors = []
    for line in process.stderr:  # type: ignore[union-attr]
        text = line.decode('utf-8', 'replace').strip()
        if text.startswith('frame='):
            done(int(text[6:] or 0))
        elif text and '=' not in text:
            errors.append(text)
    if process.wait() != 0:
        raise RuntimeError(f'ffmpeg failed on minute {source.parent.name}: ' + ' '.join(errors[-3:]))


def make(root: Path, route: str, progress: Callable[[float], None] | None = None) -> dict:
    """Makes the video of a drive and its index, and returns the index. progress(0-1) while ffmpeg works."""
    if not ROUTE.fullmatch(route):
        raise ValueError('invalid route id: ' + route)
    parts = minutes(root, route)
    if not parts:
        raise ValueError('no road video of this drive on the server')
    video, index_path = paths(root, route)
    video.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        counts = list(pool.map(lambda part: _frames(part[1]), parts))
    segments, start = [], 0.0
    for (segment, _path), count in zip(parts, counts):
        segments.append({'segment': segment, 'start': round(start, 3), 'frames': count})
        start += count / FPS
    total = sum(counts)
    # Its own partial file and pieces: the site and `nnslr drive-video` may make the same drive at once; the last one
    # wins whole.
    stem = f'{route}.{os.getpid()}.{threading.get_ident()}'
    partial = video.with_name(stem + '.part.mp4')
    folder = video.with_name(stem + '.parts')
    folder.mkdir(exist_ok=True)
    pieces = [folder / f'{number:04d}.mp4' for number in range(len(parts))]
    done = [0] * len(parts)
    counting = threading.Lock()

    def advance(number: int, frames: int) -> None:
        with counting:
            done[number] = frames
            if progress is not None and total:
                progress(min(1.0, sum(done) / total))

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            jobs = [pool.submit(_encode, path, piece, functools.partial(advance, number))
                    for number, ((_segment, path), piece) in enumerate(zip(parts, pieces))]
            try:
                for job in jobs:
                    job.result()
            except BaseException:
                for job in jobs:
                    job.cancel()
                raise
        # Joined as they are (the concat demuxer reads the names next to its list), the index at the front.
        listing = folder / 'list.txt'
        listing.write_text(''.join(f"file '{piece.name}'\n" for piece in pieces), encoding='utf-8')
        joined = subprocess.run(['nice', '-n', '15', 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                                 '-safe', '0', '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(partial)],
                                capture_output=True, text=True, timeout=600, check=False)
        if joined.returncode != 0:
            raise RuntimeError('ffmpeg could not join the minutes: ' + joined.stderr.strip()[-300:])
        os.replace(partial, video)
    finally:
        partial.unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
    value = {'route': route, 'fps': FPS, 'width': WIDTH, 'segments': segments, 'duration': round(start, 3),
             'source': _signature(parts), 'made_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    index_path.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')
    return value


def state(root: Path, route: str, *, start: bool = False) -> dict:
    """Where the video of a drive is: ready (with its index), being made (how far), failed (why), or missing; start:
    begin making it in the background when it is missing (one at a time: another waits its turn)."""
    if not ROUTE.fullmatch(route):
        return {'state': 'missing', 'reason': 'invalid route id'}
    made = index(root, route)
    if made is not None:
        return {'state': 'ready', 'url': '/route/' + route + '/video.mp4', 'index': made}
    with _lock:
        work = _making.get(route)
        if work is not None and work['state'] in ('waiting', 'making'):
            return {'state': work['state'], 'progress': work['progress']}
        if not minutes(root, route):
            return {'state': 'missing', 'reason': 'no road video of this drive on the server'}
        if not start:
            return {'state': 'failed', 'reason': work['reason']} if work else {'state': 'missing'}
        work = {'state': 'waiting', 'progress': 0.0, 'reason': ''}
        _making[route] = work
    threading.Thread(target=_run, args=(Path(root), route, work), daemon=True).start()
    return {'state': 'waiting', 'progress': 0.0}


_turn = threading.Lock()


def _run(root: Path, route: str, work: dict) -> None:
    with _turn:
        work['state'] = 'making'
        try:
            make(root, route, lambda share: work.__setitem__('progress', round(share, 3)))
            work['state'] = 'ready'
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            work.update(state='failed', reason=str(exc))
