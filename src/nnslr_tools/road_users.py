"""The road users of the brain's training: what an off-the-shelf detector finds on the road camera, four times a second,
and the vehicles a person decided crossed our path. Only answers of the brain's users_head, a head of the training alone
(brain_net.TRAINING_ONLY): nothing here reaches the car, which keeps the same inputs and the same weights.

The detector is torchvision's Faster R-CNN, ResNet-50 FPN v2, on its COCO weights (DETECTOR; the first run downloads them
from download.pytorch.org into torch's hub cache, or scripts/fetch_road_users_detector.sh). It never learns here and it
never runs on the CPU: without a CUDA card it stops (NO_CUDA), and a card is chosen as everywhere else (NNSLR_GPU_INDEX,
--gpus, or CUDA_VISIBLE_DEVICES as run_brain.sh sets it). It reads every segment's road video (fcamera.hevc) at the brain's
own readings: one decoded frame in VIDEO_STEP of the 20 fps, 4 a second, decoded by NVDEC (ffmpeg -hwaccel cuda, the CPU's
decoder only for a video NVDEC cannot decode, as road_type_signs reads them), a video at full definition reduced to
FRAME_SIZE (1344 x 760), in batches under autocast (float16). It keeps the raw boxes of the six COCO classes of the road
users (CLASSES: person; bicycle and motorcycle, the two-wheelers; car, bus and truck, the vehicles), score SCORE_FLOOR or
more, in the frame's pixels: per segment, data/derived/road_users/<route>/<segment>.npz (arrays reading, cls, score, box;
meta: VERSION, DETECTOR, the floor, the video's size and time, the frames read), written whole (.part, then os.replace).
A segment read by the same detector from the same video is never read again; any other is (current).

Persons and two-wheelers are answered by the detector alone: on every reading it read, each of COLUMNS (8) equal columns
of the frame is 1 for a kind when a box of that kind has its centre there, a score of COUNT_SCORE or more and the height
of the review's size rule (usernet.COMPLETE_HEIGHT: 24 px for persons and two-wheelers), with the height of its tallest
such box over the frame's height. A reading the detector did not read (no file, a segment of another detector or another
video, a reading past the frames it read) has weight 0: never "nobody there".

Vehicles are no answer of the detector: a parked car, the car ahead and the oncoming traffic are everywhere, and openpilot
already follows the car ahead. Only a vehicle that crosses our path is one, and only a person says which: the crossing
rule below only PROPOSES events (proposals), a person confirms or rejects each (verdicts), and the brain learns only what
was reviewed (vehicle_answers). A network that learns the applicability of a road user from those verdicts may take over
later, as the applicability network did for the signs; it is not built yet.

The crossing rule (proposals, RULE_VERSION). The vehicle boxes (score COUNT_SCORE or more, TRACK_MIN_HEIGHT px or more)
are tracked from reading to reading: the nearest box of a similar height (MATCH_DISTANCE, MATCH_HEIGHT_RATIO), a track
kept through MAX_GAP_SECONDS without a box. Everything still, oncoming or driving with us moves in the picture along rays
from the point we drive towards (about the centre column): its distance from that column over its own height, u = (x -
centre) / h, stays the same, since both shrink with the distance alike (u is its side offset in its own heights). A
vehicle that crosses our path changes u: over WINDOW_SECONDS (at least JUDGE_SECONDS of its track), u times the height
of its class (VEHICLE_HEIGHT_M) gives its speed across, in m/s. It is proposed when that speed is CROSSING_MPS or more
and goes towards the centre column or across it. Our own turning moves everything sideways: the yaw rate of the logs
(the device's gyroscope, yaw_dev, or carState's yawRate, yaw_car, on a drive without it), integrated between the two
exposures, is taken off the later box's angle (focal length FOCAL_PX, the road camera's at FRAME_SIZE, as the car's
vision.FOCAL_LENGTHS). A reading cannot be judged where the turn is stronger than MAX_YAW_DEG_S over WINDOW_SECONDS, or
the yaw is unknown, nor in the column of a vehicle of the size rule whose track is younger than JUDGE_SECONDS: those
readings and columns ('unjudged') have weight 0, never "no crossing", since no proposal could be made there. An event is
the readings of one track the rule says crossing (gaps of up to EVENT_GAP_READINGS), MIN_EVENT_READINGS or more; it
keeps its boxes and its score, the highest speed across in m/s. Per drive, data/derived/road_users/<route>/proposals.json
(RULE_VERSION, the segments it was made from); made again by nnslr road-users whenever a segment or the rule changed.

The verdicts (a person's, never written by code): annotations/road_user_events/<route>.jsonl, one line a verdict,
appended (add_verdict), {"event", "verdict" (crossing, not_crossing, or clear: taken back), "kind", "class", "score",
"from", "to", "boxes", "reviewer", "at"}; the last line of an event wins; a line of a reviewer NET_PREFIX is never read.
The drive's revision is its number of lines: a verdict sent with an older one is refused (409), so two pages never
overwrite each other. A line keeps the event's own boxes, so a confirmed crossing stays an answer when the rule changes.
The site reads and writes them (drive_view, drives_list, save_verdict: /api/road-users/..., a person signed in); its
review page shows each event on the drive's MP4 (drive_video) with the track's box at each of its readings. The vehicle answer: 1 on the readings and columns of the
boxes of a confirmed crossing; 0 elsewhere only on a drive whose current proposals all have a verdict (no proposal at all
counts as all reviewed), except where unjudged; weight 0 everywhere else (a drive or an event not reviewed).

Design limits: the detector reads 1344 x 760 frames, the brain sees 256 x 144 pictures (a person 80 m away is about 5 px
tall there, about 12 px from about 30 m); the readings are 4 a second, so a track needs JUDGE_SECONDS (3 readings)
before the rule can say anything; the yaw comes from the logs at their own rate (a change of 0.5 deg/s or more), the
camera is taken as pointing where we drive; the heights of the classes are typical, so the speed across is only a scale.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import queue
import subprocess
import threading
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nnslr_tools.drive_logs import ROUTE
from nnslr_tools.media import FRAME_SIZE, FULL_SIZE, REDUCE_FILTER
from nnslr_tools.road_type_signs import SEGMENT_SECONDS, VIDEO_FPS, VIDEO_STEP, _video_size, _videos
from nnslr_tools.usernet import COMPLETE_HEIGHT, DRIVE_KINDS, KINDS

RELATIVE = Path('derived') / 'road_users'
VERDICTS_RELATIVE = Path('annotations') / 'road_user_events'
PROPOSALS = 'proposals.json'
# The readings' format (a segment's .npz); a new one reads every segment again.
VERSION = 1
# The crossing rule's (proposals.json); a new one makes the proposals again from the readings, no GPU.
RULE_VERSION = 1
DETECTOR = 'torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1'
DETECTOR_URL = 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
# The lowest score kept of a box: the answers count from COUNT_SCORE, the rest stays for later rules and the check.
SCORE_FLOOR = 0.3
# The COCO classes kept, by their names in the weights' categories (never by index: COCO's 91 have gaps), and the kind
# of each; every other class is left out. A box's 'cls' is its place in CLASSES.
COCO_KINDS = {'person': 'person', 'bicycle': 'two_wheeler', 'motorcycle': 'two_wheeler', 'car': 'vehicle',
              'bus': 'vehicle', 'truck': 'vehicle'}
CLASSES = tuple(COCO_KINDS)
COLUMNS = 8
COUNT_SCORE = 0.5
# Frames a batch on the card: the cards are often shared, so a small one (about 3 GB with float16).
BATCH = 4
NO_CUDA = ('road users: no CUDA card here (NNSLR_GPU_INDEX, --gpus or CUDA_VISIBLE_DEVICES): the detector runs on a '
           'V100, never on the CPU')
NET_PREFIX = 'net/'
VERDICTS = ('crossing', 'not_crossing')
# A verdict taken back: the event is not reviewed again.
CLEAR = 'clear'

# --- the crossing rule (RULE_VERSION) ---------------------------------------------------------------------------------
# The kinds the rule proposes events of; persons and two-wheelers stay the detector's presence for now.
CROSSING_KINDS = ('vehicle',)
# The road camera's focal length in pixels at FRAME_SIZE: the car's vision.FOCAL_LENGTHS[(1344, 760)].
FOCAL_PX = 1141.5
# A vehicle box must be this tall to start or carry a track: smaller ones are too far to place, and their boxes jump.
TRACK_MIN_HEIGHT = 16.0
# Our own car: the detector reads the bonnet at the bottom of the road frame as a car (a box from about y 670 to the
# frame's bottom, its sides jumping from reading to reading, on the smoke run of 12 Nov 2026). A vehicle box whose top is
# this low (the frame's last 120 px at FRAME_SIZE) is left out of the tracks: a real vehicle that close reaches higher.
HOOD_TOP = 640.0
# A box continues a track when its centre is within MATCH_DISTANCE times the wider of the two boxes from the track's
# last box and the taller is at most MATCH_HEIGHT_RATIO times the other: at 4 a second a car crossing at 10 m/s 20 m
# ahead moves about half its own length, and one driving at us grows by at most about half.
MATCH_DISTANCE = 1.0
MATCH_HEIGHT_RATIO = 1.6
# A track missing for longer ends (2 readings missed).
MAX_GAP_SECONDS = 0.8
# Its speed across is measured over the last WINDOW_SECONDS of the track, and only once it is JUDGE_SECONDS old (3
# readings): one reading's box jumps by a few pixels, which over 0.25 s would look like a car moving across.
WINDOW_SECONDS = 1.0
JUDGE_SECONDS = 0.5
# The typical height of each class in metres: u (its offset in its own heights) times it is metres across.
VEHICLE_HEIGHT_M = {'car': 1.5, 'bus': 3.2, 'truck': 3.0, 'person': 1.7, 'bicycle': 1.7, 'motorcycle': 1.5}
# Proposed from this speed across towards our path, m/s: above the drift of a car in its lane and of most lane changes
# (about 1 m/s), well below a car crossing a junction.
CROSSING_MPS = 1.5
# Stronger turns are not judged: a timing error of the yaw of 50 ms at 10 deg/s moves a box by 10 px.
MAX_YAW_DEG_S = 10.0
# An event: the crossing readings of one track, holes of up to EVENT_GAP_READINGS kept, MIN_EVENT_READINGS or more.
EVENT_GAP_READINGS = 2
MIN_EVENT_READINGS = 2
NS = 1_000_000_000
# The size bins of the check, by a box's height at FRAME_SIZE.
CHECK_BINS = (('24-48', 0.0, 48.0), ('48-96', 48.0, 96.0), ('96+', 96.0, math.inf))


def kind_of(cls: int) -> str:
    return COCO_KINDS[CLASSES[cls]]


def reading_key(segment: int, row: int) -> str:
    """A reading's key, as the brain's 4 Hz frames name theirs: the segment and the seconds into it ("3/12.25")."""
    return f'{segment}/{row * VIDEO_STEP / VIDEO_FPS:g}'


def parse_key(key: str) -> tuple[int, int]:
    """(segment, row) of a reading_key."""
    segment, seconds = str(key).split('/')
    return int(segment), int(round(float(seconds) * VIDEO_FPS / VIDEO_STEP))


def column_of(box) -> int:
    """The column of a box: of its centre's x, COLUMNS equal ones over the frame's width."""
    centre = (float(box[0]) + float(box[2])) / 2
    return min(COLUMNS - 1, max(0, int(centre * COLUMNS // FRAME_SIZE[0])))


def segment_path(root: Path, route: str, segment: int) -> Path:
    return Path(root) / RELATIVE / checked_route(route) / f'{int(segment)}.npz'


def proposals_path(root: Path, route: str) -> Path:
    return Path(root) / RELATIVE / checked_route(route) / PROPOSALS


def checked_route(route: str) -> str:
    """A drive's id as the logs write it (dongle--route); RoadUsersError(400) otherwise: a path is made of it."""
    if not isinstance(route, str) or not ROUTE.fullmatch(route):
        raise RoadUsersError(400, 'invalid_route', f'Not a drive: {route!r}.')
    return route


def verdicts_path(root: Path, route: str) -> Path:
    checked_route(route)
    return Path(root) / VERDICTS_RELATIVE / f'{route}.jsonl'


def video_stamp(path: Path) -> list[int]:
    stat = Path(path).stat()
    return [int(stat.st_size), int(stat.st_mtime)]


def current(meta: dict | None, stamp: list | None) -> bool:
    """Whether a segment's readings are this detector's, of this video as it is now."""
    return bool(meta) and stamp is not None and meta.get('version') == VERSION and meta.get('detector') == DETECTOR \
        and meta.get('score_floor') == SCORE_FLOOR and meta.get('video_step') == VIDEO_STEP \
        and list(meta.get('video') or []) == list(stamp) and isinstance(meta.get('frames'), int)


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.part')
    with temporary.open('wb') as handle:
        write(handle)
    os.replace(temporary, path)


def save_segment(path: Path, meta: dict, reading, cls, score, box) -> None:
    """One segment's readings, written whole."""
    _atomic(Path(path), lambda handle: np.savez_compressed(
        handle, meta=np.array(json.dumps(meta, sort_keys=True)), reading=np.asarray(reading, dtype=np.int32),
        cls=np.asarray(cls, dtype=np.uint8), score=np.asarray(score, dtype=np.float32),
        box=np.asarray(box, dtype=np.float32).reshape(-1, 4)))


def load_segment(path: Path) -> dict | None:
    """{meta, reading, cls, score, box} of a segment's file, None without one or when it cannot be read."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return {'meta': json.loads(str(data['meta'])), **{name: data[name] for name in ('reading', 'cls', 'score', 'box')}}
    except (OSError, ValueError, KeyError):
        return None


def load_readings(root: Path, route: str, videos: dict[int, Path] | None = None) -> dict[int, dict]:
    """The current readings of a drive: {segment: {meta, reading, cls, score, box, frames}} for every segment whose
    file is this detector's reading of its video as it is now (current); the others are as never read."""
    videos = _videos(root, route) if videos is None else videos
    found = {}
    for segment, video in videos.items():
        data = load_segment(segment_path(root, route, segment))
        if data is None:
            continue
        try:
            stamp = video_stamp(video)
        except OSError:
            continue
        if current(data['meta'], stamp):
            found[segment] = {**data, 'frames': int(data['meta']['frames'])}
    return found


def boxes_at(data: dict) -> dict[int, list[tuple[int, float, list[float]]]]:
    """A segment's readings as {row: [(cls, score, box), ...]}."""
    found: dict[int, list] = {}
    for row, cls, score, box in zip(data['reading'].tolist(), data['cls'].tolist(), data['score'].tolist(),
                                    data['box'].tolist()):
        found.setdefault(int(row), []).append((int(cls), float(score), [float(value) for value in box]))
    return found


# --- the detector (GPU) -----------------------------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_start(cards: list[str] | None, turn, batch: int) -> None:
    if cards:
        # This process's card, the next in turn, before torch opens one.
        with turn.get_lock():
            place = turn.value
            turn.value += 1
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=cards[place % len(cards)])
    _WORKER.update(batch=batch)


def _detector():
    """The worker's detector on its card, loaded once; RuntimeError(NO_CUDA) without a card."""
    if 'model' not in _WORKER:
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        if not torch.cuda.is_available():
            raise RuntimeError(NO_CUDA)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=SCORE_FLOOR).eval().to('cuda')
        names = weights.meta['categories']
        keep = {index: CLASSES.index(name) for index, name in enumerate(names) if name in COCO_KINDS}
        if len(keep) != len(CLASSES):
            raise RuntimeError(f'road users: the detector has no class {sorted(set(CLASSES) - set(names))}')
        _WORKER.update(torch=torch, model=model, keep=keep, card=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                       name=torch.cuda.get_device_name(0))
    return _WORKER


def _frames(video: str, reduce: bool, nvdec: bool, out: queue.Queue, stop: threading.Event) -> None:
    """The video's readings as rgb24 arrays into out, then None (a thread: ffmpeg decodes while the card detects);
    stop: ffmpeg is stopped."""
    size = FRAME_SIZE[0] * FRAME_SIZE[1] * 3
    chosen = f'select=not(mod(n\\,{VIDEO_STEP}))' + (',' + REDUCE_FILTER if reduce else '')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *(['-hwaccel', 'cuda'] if nvdec else []), '-i', str(video),
               '-vf', chosen, '-vsync', '0', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
            while not stop.is_set():
                buffer = process.stdout.read(size)
                if len(buffer) < size:
                    break
                out.put(np.frombuffer(buffer, dtype=np.uint8).reshape(FRAME_SIZE[1], FRAME_SIZE[0], 3))
            if stop.is_set():
                process.kill()
    finally:
        out.put(None)


def detect_video(video: str, *, nvdec: bool = True, limit: int | None = None) -> dict:
    """The detector on a video's readings: {frames, reading, cls, score, box, seconds (on the card)}. limit: stop after
    this many readings (a smoke test)."""
    worker = _detector()
    torch, model, keep, batch = worker['torch'], worker['model'], worker['keep'], worker.get('batch') or BATCH
    width, height = _video_size(Path(video))
    reduce = (width, height) != FRAME_SIZE
    if reduce and (width, height) != FULL_SIZE:
        raise ValueError(f'{video}: a road video of {width} x {height}, neither {FRAME_SIZE} nor {FULL_SIZE}')
    frames: queue.Queue = queue.Queue(maxsize=batch * 3)
    stop = threading.Event()
    reader = threading.Thread(target=_frames, args=(video, reduce, nvdec, frames, stop), daemon=True)
    reader.start()
    found: dict[str, list] = {'reading': [], 'cls': [], 'score': [], 'box': []}
    count, spent, ended = 0, 0.0, False

    def run(pictures: list[np.ndarray], first: int) -> None:
        nonlocal spent
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            tensor = torch.from_numpy(np.stack(pictures)).to('cuda', non_blocking=True)
            images = list(tensor.permute(0, 3, 1, 2).float().div_(255))
            said = model(images)
        for place, one in enumerate(said):
            labels, scores, boxes = (one[name].float().cpu().numpy() if name != 'labels' else one[name].cpu().numpy()
                                     for name in ('labels', 'scores', 'boxes'))
            for label, score, box in zip(labels.tolist(), scores.tolist(), boxes.tolist()):
                if label in keep and score >= SCORE_FLOOR:
                    found['reading'].append(first + place)
                    found['cls'].append(keep[label])
                    found['score'].append(round(score, 4))
                    found['box'].append([round(value, 1) for value in box])
        spent += time.monotonic() - started

    pending: list[np.ndarray] = []
    while not ended:
        picture = frames.get()
        if picture is None or (limit is not None and count + len(pending) >= limit):
            ended = True
        else:
            pending.append(picture)
        if pending and (ended or len(pending) >= batch):
            run(pending, count)
            count += len(pending)
            pending = []
    if limit is not None:
        # Stop ffmpeg, and take what it still had written.
        stop.set()
        while reader.is_alive():
            try:
                frames.get(timeout=1)
            except queue.Empty:
                pass
    reader.join(timeout=5)
    return {'frames': count, **found, 'seconds': round(spent, 3)}


def read_segment(task: tuple) -> dict:
    """One segment's readings, saved to its file: task (route, segment, video, path). NVDEC first; a video it cannot
    decode is decoded on the CPU, the detector still on the card."""
    route, segment, video, path = task
    stamp = video_stamp(Path(video))
    nvdec = True
    found = detect_video(video, nvdec=True)
    if not found['frames']:
        nvdec = False
        found = detect_video(video, nvdec=False)
    worker = _detector()
    meta = {'version': VERSION, 'detector': DETECTOR, 'score_floor': SCORE_FLOOR, 'video_step': VIDEO_STEP,
            'frame_size': list(FRAME_SIZE), 'video': stamp, 'frames': found['frames'], 'nvdec': nvdec,
            'card': worker['name'], 'read_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    save_segment(Path(path), meta, found['reading'], found['cls'], found['score'], found['box'])
    return {'route': route, 'segment': segment, 'frames': found['frames'], 'boxes': len(found['reading']),
            'nvdec': nvdec, 'seconds': found['seconds']}


def unread(root: Path, route: str) -> list[tuple[int, Path]]:
    """The segments of a drive whose readings are not current, (segment, video)."""
    found = []
    for segment, video in _videos(root, route).items():
        data = load_segment(segment_path(root, route, segment))
        try:
            stamp = video_stamp(video)
        except OSError:
            continue
        if data is None or not current(data['meta'], stamp):
            found.append((segment, video))
    return found


def read_drives(root: Path, routes: list[str], *, cards: list[str] | None = None, batch: int = BATCH,
                log=print) -> dict:
    """Reads every segment of the drives that is not current, one process a card (cards: nvidia-smi's numbers; None: the
    first card this process sees, as CUDA_VISIBLE_DEVICES says). {'segments', 'frames', 'seconds' (on the cards)}."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not cards:
        raise RuntimeError(NO_CUDA)
    tasks = [(route, segment, str(video), str(segment_path(root, route, segment)))
             for route in routes for segment, video in unread(root, route)]
    log(f'road users: {len(tasks)} segments to read of {len({task[0] for task in tasks})} drives '
        f'({len(routes)} asked), 4 readings a second, {DETECTOR}' + (f", on the cards {', '.join(cards)}" if cards else ''))
    if not tasks:
        return {'segments': 0, 'frames': 0, 'seconds': 0.0}
    context = multiprocessing.get_context('spawn')
    processes = len(cards) if cards else 1
    done, frames, spent, started = 0, 0, 0.0, time.monotonic()
    with ProcessPoolExecutor(max_workers=processes, mp_context=context, initializer=_worker_start,
                             initargs=(cards, context.Value('i', 0), batch)) as pool:
        for result in pool.map(read_segment, tasks):
            done += 1
            frames += result['frames']
            spent += result['seconds']
            if not result['nvdec']:
                log(f"road users: segment {result['segment']} of {result['route']}: NVDEC could not decode it, decoded "
                    'on the CPU (the detector on the card)')
            if done == 1 or done % 10 == 0 or done == len(tasks):
                rate = frames / max(time.monotonic() - started, 1e-6)
                log(f'road users: {done}/{len(tasks)} segments, {frames} readings, {rate:.1f} readings a second')
    return {'segments': done, 'frames': frames, 'seconds': round(spent, 1)}


# --- the yaw ----------------------------------------------------------------------------------------------------------


class Yaw:
    """A yaw-rate step series of the logs ([t_ns, deg/s left positive or None], a change a point) as an integral: the
    angle turned between two times, None where it is unknown for any part (before the first point, or a None)."""

    def __init__(self, points: list | None):
        points = sorted((int(t), value) for t, value in points or [])
        self.times = [t for t, _ in points]
        self.rates = [None if value is None else math.radians(float(value)) for _, value in points]
        self.angle, self.blind = [0.0], [0]
        for index in range(1, len(points)):
            span = self.times[index] - self.times[index - 1]
            rate = self.rates[index - 1]
            self.angle.append(self.angle[-1] + (rate or 0.0) * span / NS)
            self.blind.append(self.blind[-1] + (span if rate is None else 0))

    def _at(self, t: int) -> tuple[float, int] | None:
        index = bisect_right(self.times, t) - 1
        if index < 0:
            return None
        rate, span = self.rates[index], t - self.times[index]
        return self.angle[index] + (rate or 0.0) * span / NS, self.blind[index] + (span if rate is None else 0)

    def turned(self, start: int | None, end: int | None) -> float | None:
        """Radians turned to the left from start to end (ns), None when unknown."""
        if start is None or end is None:
            return None
        a, b = self._at(int(start)), self._at(int(end))
        if a is None or b is None or b[1] != a[1]:
            return None
        return b[0] - a[0]


def drive_yaw(merged: dict | None) -> tuple[str | None, Yaw | None]:
    """The yaw of a drive: the device's gyroscope (yaw_dev), or carState.yawRate (yaw_car) on a drive without it and
    whose car has the sensor (not all 0); (None, None) without either."""
    series = (merged or {}).get('series') or {}
    if any(value is not None for _, value in series.get('yaw_dev') or []):
        return 'yaw_dev', Yaw(series['yaw_dev'])
    if any(value not in (None, 0, 0.0) for _, value in series.get('yaw_car') or []):
        return 'yaw_car', Yaw(series['yaw_car'])
    return None, None


# --- tracks and the crossing rule -------------------------------------------------------------------------------------


def _timeline(readings: dict[int, dict], exposures: dict[int, dict[int, int]]) -> list[dict]:
    """Every reading read, in order: {segment, row, key, t (s from the drive's start), ns (exposure or None), boxes}."""
    found = []
    for segment in sorted(readings):
        at = boxes_at(readings[segment])
        times = exposures.get(segment) or {}
        for row in range(readings[segment]['frames']):
            found.append({'segment': segment, 'row': row, 'key': reading_key(segment, row),
                          't': segment * SEGMENT_SECONDS + row * VIDEO_STEP / VIDEO_FPS,
                          'ns': times.get(row * VIDEO_STEP), 'boxes': at.get(row, [])})
    return found


def track(timeline: list[dict], kinds: tuple[str, ...] = CROSSING_KINDS) -> list[list[dict]]:
    """The tracks of the boxes of these kinds (score COUNT_SCORE or more, TRACK_MIN_HEIGHT px or more, top above
    HOOD_TOP): lists of
    {at (place in the timeline), t, ns, box, cls, score}, in the order they start."""
    tracks: list[list[dict]] = []
    alive: list[int] = []
    for at, reading in enumerate(timeline):
        boxes = [(cls, score, box) for cls, score, box in reading['boxes']
                 if kind_of(cls) in kinds and score >= COUNT_SCORE and box[3] - box[1] >= TRACK_MIN_HEIGHT
                 and box[1] < HOOD_TOP]
        alive = [index for index in alive if reading['t'] - tracks[index][-1]['t'] <= MAX_GAP_SECONDS + 1e-6]
        pairs = []
        for index in alive:
            last = tracks[index][-1]['box']
            for place, (cls, _score, box) in enumerate(boxes):
                if kind_of(cls) != kind_of(tracks[index][-1]['cls']):
                    continue
                heights = sorted((last[3] - last[1], box[3] - box[1]))
                distance = math.hypot((last[0] + last[2] - box[0] - box[2]) / 2, (last[1] + last[3] - box[1] - box[3]) / 2)
                wide = max(last[2] - last[0], box[2] - box[0], 1.0)
                if heights[1] <= MATCH_HEIGHT_RATIO * heights[0] and distance <= MATCH_DISTANCE * wide:
                    # The nearest of a similar size first: a car and a truck beyond it, their boxes on one another, keep
                    # their own tracks.
                    pairs.append((distance / wide + math.log(max(heights[1], 1.0) / max(heights[0], 1.0)), index, place))
        taken_tracks, taken_boxes = set(), set()
        for _cost, index, place in sorted(pairs):
            if index in taken_tracks or place in taken_boxes:
                continue
            taken_tracks.add(index)
            taken_boxes.add(place)
            cls, score, box = boxes[place]
            tracks[index].append({'at': at, 't': reading['t'], 'ns': reading['ns'], 'box': box, 'cls': cls, 'score': score})
        for place, (cls, score, box) in enumerate(boxes):
            if place not in taken_boxes:
                tracks.append([{'at': at, 't': reading['t'], 'ns': reading['ns'], 'box': box, 'cls': cls, 'score': score}])
                alive.append(len(tracks) - 1)
    return tracks


def side_offset(box, height: float, turned: float = 0.0, focal: float = FOCAL_PX) -> float:
    """u: the box's offset from the centre column in its own heights, its angle turned back by turned radians (our turn
    to the left since the reference reading moves the picture to the right)."""
    angle = math.atan(((box[0] + box[2]) / 2 - FRAME_SIZE[0] / 2) / focal) - turned
    return focal * math.tan(angle) / max(height, 1.0)


def across(reference: dict, now: dict, turned: float, focal: float = FOCAL_PX) -> float:
    """The speed across of a tracked box from reference to now, m/s, + to the right, our turn taken off."""
    first = side_offset(reference['box'], reference['box'][3] - reference['box'][1], 0.0, focal)
    last = side_offset(now['box'], now['box'][3] - now['box'][1], turned, focal)
    metres = VEHICLE_HEIGHT_M.get(CLASSES[now['cls']], 1.5)
    return (last - first) * metres / max(now['t'] - reference['t'], 1e-6)


def crossing_of(reference: dict, now: dict, turned: float, focal: float = FOCAL_PX) -> tuple[bool, float]:
    """(crossing, speed across towards our path in m/s): towards the centre column or across it, CROSSING_MPS or more."""
    speed = across(reference, now, turned, focal)
    offset = side_offset(reference['box'], reference['box'][3] - reference['box'][1], 0.0, focal)
    towards = speed * offset < 0
    return towards and abs(speed) >= CROSSING_MPS, abs(speed) if towards else 0.0


def judge(timeline: list[dict], tracks: list[list[dict]], yaw: Yaw | None,
          focal: float = FOCAL_PX) -> tuple[dict[int, dict[int, float]], dict[int, int]]:
    """The rule on every reading: ({track: {at: speed across of a crossing reading}}, {at: unjudged columns as a
    bit mask (bit c: column c)}). A reading's turn: the angle over the WINDOW_SECONDS before it."""
    unjudged: dict[int, int] = {}
    steady: dict[int, bool] = {}
    for at, reading in enumerate(timeline):
        ns = reading['ns']
        turned = yaw.turned(None if ns is None else ns - int(WINDOW_SECONDS * NS), ns) if yaw is not None else None
        steady[at] = turned is not None and abs(math.degrees(turned)) / WINDOW_SECONDS <= MAX_YAW_DEG_S
        if not steady[at]:
            unjudged[at] = (1 << COLUMNS) - 1
    crossing: dict[int, dict[int, float]] = {}
    for index, path in enumerate(tracks):
        for place, now in enumerate(path):
            if not steady[now['at']]:
                continue
            counted = now['box'][3] - now['box'][1] >= COMPLETE_HEIGHT[kind_of(now['cls'])]
            reference = next((entry for entry in path[:place] if now['t'] - entry['t'] <= WINDOW_SECONDS + 1e-6), None)
            turned = None if reference is None else yaw.turned(reference['ns'], now['ns'])
            if reference is None or now['t'] - reference['t'] < JUDGE_SECONDS - 1e-6 or turned is None:
                if counted:
                    unjudged[now['at']] = unjudged.get(now['at'], 0) | (1 << column_of(now['box']))
                continue
            yes, speed = crossing_of(reference, now, turned, focal)
            if yes and counted:
                crossing.setdefault(index, {})[now['at']] = speed
    return crossing, unjudged


def events_of(timeline: list[dict], tracks: list[list[dict]], crossing: dict[int, dict[int, float]]) -> list[dict]:
    """The proposed events: each track's crossing readings joined over holes of up to EVENT_GAP_READINGS, with the
    track's boxes from the first to the last, MIN_EVENT_READINGS or more; in the order of the drive."""
    found = []
    for index, said in crossing.items():
        path = tracks[index]
        places = sorted(said)
        groups, current_group = [], [places[0]]
        for at in places[1:]:
            if at - current_group[-1] <= EVENT_GAP_READINGS + 1:
                current_group.append(at)
            else:
                groups.append(current_group)
                current_group = [at]
        groups.append(current_group)
        for group in groups:
            if len(group) < MIN_EVENT_READINGS:
                continue
            entries = [entry for entry in path if group[0] <= entry['at'] <= group[-1]]
            first = entries[0]
            classes = Counter(CLASSES[entry['cls']] for entry in entries)
            found.append({'id': f"{timeline[first['at']]['key']}@{int(round((first['box'][0] + first['box'][2]) / 2))}",
                          'kind': kind_of(first['cls']), 'class': classes.most_common(1)[0][0],
                          'from': timeline[group[0]]['key'], 'to': timeline[group[-1]]['key'],
                          'score': round(max(said[at] for at in group), 2),
                          'boxes': [[timeline[entry['at']]['key'], *[round(value, 1) for value in entry['box']]]
                                    for entry in entries]})
    return sorted(found, key=lambda event: (parse_key(event['from']), event['id']))


def segments_stamp(readings: dict[int, dict]) -> dict[str, list]:
    """What proposals were made from: {segment: [video size, video time, frames]}."""
    return {str(segment): [*data['meta']['video'], data['frames']] for segment, data in sorted(readings.items())}


def make_proposals(readings: dict[int, dict], exposures: dict[int, dict[int, int]], merged: dict | None,
                   focal: float = FOCAL_PX) -> dict:
    """The proposals of a drive from its readings: {version, detector, segments, yaw, events, unjudged: {key: mask}}."""
    timeline = _timeline(readings, exposures)
    source, yaw = drive_yaw(merged)
    tracks = track(timeline)
    crossing, unjudged = judge(timeline, tracks, yaw, focal)
    return {'version': RULE_VERSION, 'readings_version': VERSION, 'detector': DETECTOR, 'segments': segments_stamp(readings),
            'yaw': source, 'events': events_of(timeline, tracks, crossing),
            'unjudged': {timeline[at]['key']: mask for at, mask in sorted(unjudged.items())},
            'made': datetime.now(timezone.utc).isoformat(timespec='seconds')}


def load_proposals(root: Path, route: str) -> dict | None:
    path = proposals_path(root, route)
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    except (OSError, ValueError):
        return None


def proposals_current(proposals: dict | None, readings: dict[int, dict]) -> bool:
    """Whether a drive's proposals are this rule's, made from its readings as they are now."""
    return bool(proposals) and proposals.get('version') == RULE_VERSION and proposals.get('readings_version') == VERSION \
        and proposals.get('detector') == DETECTOR and proposals.get('segments') == segments_stamp(readings)


def save_proposals(root: Path, route: str, proposals: dict) -> None:
    _atomic(proposals_path(root, route), lambda handle: handle.write(json.dumps(proposals, indent=1).encode('utf-8')))


def refresh_proposals(root: Path, routes: list[str], merged_of, exposures_of, *, log=print) -> dict:
    """Makes the proposals of every drive whose proposals are not current (the CPU, no card): {drives, events}."""
    made, events = 0, 0
    for route in routes:
        readings = load_readings(root, route)
        if not readings or proposals_current(load_proposals(root, route), readings):
            continue
        merged = merged_of(route)
        proposals = make_proposals(readings, exposures_of(merged) if merged else {}, merged)
        save_proposals(root, route, proposals)
        made += 1
        events += len(proposals['events'])
        log(f"road users: {route}: {len(proposals['events'])} crossing events proposed"
            + ('' if proposals['yaw'] else ', no yaw in the logs: every reading unjudged'))
    log(f'road users: proposals made again for {made} drives, {events} events')
    return {'drives': made, 'events': events}


# --- a person's verdicts ----------------------------------------------------------------------------------------------


class RoadUsersError(ValueError):
    """A request the verdicts refuse: status (HTTP), reason (a word for the page) and message."""

    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def _verdict_lines(root: Path, route: str) -> list[dict]:
    """The lines of a drive's verdicts file that can be read, in order (any reviewer)."""
    path = verdicts_path(root, route)
    found = []
    if not path.is_file():
        return found
    for text in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            line = json.loads(text)
        except ValueError:
            continue
        if isinstance(line, dict) and isinstance(line.get('event'), str) and line.get('verdict') in (*VERDICTS, CLEAR) \
                and isinstance(line.get('reviewer'), str) and line['reviewer'] \
                and (line['verdict'] == CLEAR or isinstance(line.get('boxes'), list)):
            found.append(line)
    return found


def verdicts_revision(root: Path, route: str) -> int:
    """The drive's verdicts revision: the number of its verdict lines (a page sends it back with a verdict)."""
    return len(_verdict_lines(root, route))


def load_verdicts(root: Path, route: str) -> dict[str, dict]:
    """A person's verdicts of a drive: {event id: its last line}; an event whose last line is CLEAR has none. A line that
    cannot be read, of no verdict or of a reviewer NET_PREFIX is skipped."""
    found: dict[str, dict] = {}
    for line in _verdict_lines(root, route):
        if line['reviewer'].startswith(NET_PREFIX):
            continue
        if line['verdict'] == CLEAR:
            found.pop(line['event'], None)
        else:
            found[line['event']] = line
    return found


def add_verdict(root: Path, route: str, event: dict, verdict: str, reviewer: str,
                base_revision: int | None = None) -> dict:
    """Appends a person's verdict of a proposed event (a line of proposals' events): crossing, not_crossing, or CLEAR
    (the event back to not reviewed). With base_revision, only if the drive's verdicts are still at it (409 otherwise:
    two pages never overwrite each other). Returns the line, with the new revision."""
    import fcntl
    if verdict not in (*VERDICTS, CLEAR):
        raise RoadUsersError(400, 'invalid_verdict', f'A verdict is one of {(*VERDICTS, CLEAR)}.')
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.strip().startswith(NET_PREFIX):
        raise RoadUsersError(403, 'person_required', 'A verdict is given by a person.')
    line = {'event': event['id'], 'verdict': verdict, 'kind': event['kind'], 'from': event['from'], 'to': event['to'],
            'reviewer': reviewer.strip(), 'at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    if verdict != CLEAR:
        # The event's own boxes: a confirmed crossing stays an answer when the rule or the detector changes.
        line.update({'class': event.get('class'), 'score': event.get('score'), 'boxes': event['boxes']})
    path = verdicts_path(root, route)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        revision = verdicts_revision(root, route)
        if base_revision is not None and int(base_revision) != revision:
            raise RoadUsersError(409, 'stale_revision',
                                 'The verdicts of this drive changed in another page: reload the drive.')
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(line, sort_keys=True) + '\n')
    return {**line, 'revision': revision + 1}


# --- the review's view of a drive (the REST endpoints of the site) ----------------------------------------------------


def _place(key: str, video: dict[int, float] | None) -> dict:
    """Where a reading is in the drive's video: its segment, its frame of the segment's 20 fps video (as the brain's
    frames and Sign review count them), the seconds from the drive's start as the segments' lengths say (segment x 60
    + frame / 20) and, when the drive's MP4 is made (drive_video's index), the seconds where its player seeks it."""
    segment, row = parse_key(key)
    frame = row * VIDEO_STEP
    return {'key': key, 'segment': segment, 'frame': frame,
            'seconds': round(segment * SEGMENT_SECONDS + frame / VIDEO_FPS, 3),
            'video_seconds': None if not video or segment not in video else round(video[segment] + frame / VIDEO_FPS, 3)}


def _video_starts(root: Path, route: str) -> dict[int, float] | None:
    from nnslr_tools import drive_video
    made = drive_video.index(root, route)
    return None if made is None else {int(item['segment']): float(item['start']) for item in made.get('segments') or []}


def drive_view(root: Path, route: str) -> dict:
    """A drive's proposed crossings with their verdicts, for the review: {route, revision, current, frame_size, fps,
    events: [{id, kind, class, score, from, to (_place), boxes: [{..._place, box: [x1, y1, x2, y2]}], verdict}]}; the
    boxes are FRAME_SIZE pixels (1344 x 760, the road frame's), one at each reading of the event's track."""
    checked_route(route)
    proposals = load_proposals(root, route)
    if proposals is None:
        raise RoadUsersError(404, 'no_proposals', 'This drive has no proposed crossings yet: nnslr road-users reads it.')
    verdicts = load_verdicts(root, route)
    starts = _video_starts(root, route)
    events = []
    for event in proposals.get('events') or []:
        said = verdicts.get(event['id'])
        events.append({'id': event['id'], 'kind': event['kind'], 'class': event.get('class'), 'score': event['score'],
                       'from': _place(event['from'], starts), 'to': _place(event['to'], starts),
                       'boxes': [{**_place(key, starts), 'box': box} for key, *box in event['boxes']],
                       'verdict': None if said is None else {name: said[name] for name in ('verdict', 'reviewer', 'at')}})
    readings = load_readings(root, route)
    return {'ok': True, 'route': route, 'revision': verdicts_revision(root, route),
            'current': proposals_current(proposals, readings), 'rule_version': proposals.get('version'),
            'detector': proposals.get('detector'), 'yaw': proposals.get('yaw'), 'made': proposals.get('made'),
            'frame_size': list(FRAME_SIZE), 'fps': VIDEO_FPS, 'readings_per_second': VIDEO_FPS // VIDEO_STEP,
            'video_index': starts is not None, 'unjudged_readings': len(proposals.get('unjudged') or {}),
            'events': events}


def drives_list(root: Path) -> list[dict]:
    """The drives with proposals: {route, proposed, reviewed, crossing, made, yaw}, the newest first."""
    folder = Path(root) / RELATIVE
    found = []
    for path in sorted(folder.glob(f'*/{PROPOSALS}')) if folder.is_dir() else []:
        route = path.parent.name
        proposals = load_proposals(root, route)
        if proposals is None:
            continue
        verdicts = load_verdicts(root, route)
        ids = [event['id'] for event in proposals.get('events') or []]
        found.append({'route': route, 'proposed': len(ids), 'reviewed': sum(1 for id_ in ids if id_ in verdicts),
                      'crossing': sum(1 for id_ in ids if (verdicts.get(id_) or {}).get('verdict') == 'crossing'),
                      'made': proposals.get('made'), 'yaw': proposals.get('yaw')})
    return sorted(found, key=lambda drive: drive['route'], reverse=True)


def save_verdict(root: Path, route: str, body: dict, reviewer: str) -> dict:
    """A verdict from the review ({"event", "verdict" (crossing, not_crossing or clear), "base_revision"}) on an event
    of the drive's proposals: {ok, revision, event, verdict}."""
    checked_route(route)
    if not isinstance(body, dict):
        raise RoadUsersError(400, 'invalid_json', 'A JSON object is required.')
    base = body.get('base_revision')
    if isinstance(base, bool) or not isinstance(base, int):
        raise RoadUsersError(400, 'invalid_base_revision', 'base_revision: the revision the page read.')
    proposals = load_proposals(root, route)
    event = next((item for item in (proposals or {}).get('events') or [] if item['id'] == body.get('event')), None)
    if event is None:
        raise RoadUsersError(404, 'unknown_event', 'This drive has no such proposed crossing.')
    line = add_verdict(root, route, event, body.get('verdict'), reviewer, base)
    return {'ok': True, 'revision': line['revision'], 'event': event['id'],
            'verdict': None if line['verdict'] == CLEAR else {name: line[name] for name in ('verdict', 'reviewer', 'at')}}


# --- the answers ------------------------------------------------------------------------------------------------------


def _detector_answers(boxes: list, presence: np.ndarray, height: np.ndarray, kinds: tuple[str, ...]) -> None:
    for cls, score, box in boxes:
        kind = kind_of(cls)
        tall = box[3] - box[1]
        if kind not in kinds or score < COUNT_SCORE or tall < COMPLETE_HEIGHT[kind]:
            continue
        k, c = KINDS.index(kind), column_of(box)
        presence[k, c] = 1.0
        height[k, c] = max(height[k, c], tall / FRAME_SIZE[1])


def drive_answers(root: Path, route: str, frames: list[dict], *, readings: dict[int, dict] | None = None,
                  proposals: dict | None = None, verdicts: dict[str, dict] | None = None) -> dict:
    """The answers of users_head on frames (with 'segment' and 'decoded', a reading's frame of the video):
    {presence, height, weight}, each (frames, KINDS, COLUMNS) float32, and {read, reviewed, confirmed}: the frames the
    detector read, whether the drive's vehicles are all reviewed, the confirmed events. See the module's text."""
    readings = load_readings(root, route) if readings is None else readings
    proposals = load_proposals(root, route) if proposals is None else proposals
    verdicts = load_verdicts(root, route) if verdicts is None else verdicts
    shape = (len(frames), len(KINDS), COLUMNS)
    presence, height, weight = (np.zeros(shape, dtype=np.float32) for _ in range(3))
    vehicle = KINDS.index('vehicle')
    detector_kinds = tuple(kind for kind in KINDS if kind not in CROSSING_KINDS)
    current_proposals = proposals_current(proposals, readings)
    reviewed = current_proposals and all(event['id'] in verdicts for event in proposals['events'])
    unjudged = (proposals or {}).get('unjudged') or {} if current_proposals else {}
    confirmed: dict[str, list[list[float]]] = {}
    for line in verdicts.values():
        if line['verdict'] == 'crossing' and line.get('kind') == 'vehicle':
            for key, *box in line['boxes']:
                confirmed.setdefault(key, []).append(box)
    by_segment = {segment: boxes_at(data) for segment, data in readings.items()}
    read = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        segment, decoded = frame.get('segment'), frame.get('decoded')
        if segment is None or decoded is None or decoded % VIDEO_STEP:
            continue
        row = int(decoded) // VIDEO_STEP
        key = reading_key(int(segment), row)
        data = readings.get(int(segment))
        if data is not None and row < data['frames']:
            read[index] = True
            for kind in detector_kinds:
                weight[index, KINDS.index(kind)] = 1.0
            _detector_answers(by_segment[int(segment)].get(row, []), presence[index], height[index], detector_kinds)
            if reviewed:
                mask = int(unjudged.get(key, 0))
                weight[index, vehicle] = [0.0 if mask >> column & 1 else 1.0 for column in range(COLUMNS)]
        for box in confirmed.get(key, []):
            column = column_of(box)
            presence[index, vehicle, column] = 1.0
            weight[index, vehicle, column] = 1.0
            height[index, vehicle, column] = max(height[index, vehicle, column], (box[3] - box[1]) / FRAME_SIZE[1])
    return {'presence': presence, 'height': height, 'weight': weight, 'read': read, 'reviewed': bool(reviewed),
            'confirmed': sum(1 for line in verdicts.values() if line['verdict'] == 'crossing'),
            'proposed': len((proposals or {}).get('events') or []) if current_proposals else None}


# --- the check against a person's boxes -------------------------------------------------------------------------------


def size_bin(tall: float) -> str:
    """The CHECK_BINS name of a box tall pixels high in the 1344 x 760 frame (under 24 counts in the first)."""
    return next(name for name, low, high in CHECK_BINS if low <= tall < high)


def reviewed_truth(review_db: Path) -> list[dict]:
    """A person's frames ticked "road users complete" (review.sqlite, opened read only): {route, segment, decoded,
    users: [(kind, box)], ignore: [box]}; their boxes are in FRAME_SIZE pixels, as the detector's."""
    import sqlite3
    if not Path(review_db).is_file():
        return []
    connection = sqlite3.connect(f'file:{Path(review_db)}?mode=ro', uri=True)
    try:
        rows = connection.execute('SELECT route_id, segment_index, payload FROM frame_reviews').fetchall()
    finally:
        connection.close()
    found = []
    for route, segment, payload in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if data.get('users_complete') is not True or not isinstance(data.get('decoded_frame_index'), int):
            continue
        users, ignore = [], []
        for item in data.get('detections') or []:
            box = item.get('bbox_xyxy') if isinstance(item, dict) else None
            if not isinstance(box, list) or len(box) != 4 or not str(item.get('sign_type', '')).startswith('road_user'):
                continue
            kind = DRIVE_KINDS.get(item.get('sign_type'))
            (users.append((kind, [float(v) for v in box])) if kind else ignore.append([float(v) for v in box]))
        found.append({'route': route, 'segment': int(segment), 'decoded': int(data['decoded_frame_index']),
                      'users': users, 'ignore': ignore})
    return found


def check(root: Path, truth: list[dict] | None = None) -> dict:
    """The detector's readings against a person's ticked frames: per kind and size bin (CHECK_BINS) the boxes found at
    IoU 0.5 and the false boxes (score COUNT_SCORE or more, the size rule's height or more, in no ignore box); the frames
    checked and those whose reading the detector has not made (not read, or not on a reading's frame)."""
    from nnslr_tools.evaluation import score_frame
    from nnslr_tools.signnet import _in_ignore
    truth = reviewed_truth(Path(root) / 'annotations' / 'review.sqlite') if truth is None else truth
    rows = {kind: {name: {'truth': 0, 'found': 0, 'false': 0} for name, _, _ in CHECK_BINS} for kind in KINDS}
    checked, missing, cache = 0, 0, {}
    for frame in truth:
        if frame['route'] not in cache:
            cache[frame['route']] = {segment: boxes_at(data) for segment, data in load_readings(root, frame['route']).items()}
        segment = cache[frame['route']].get(frame['segment'])
        frames_read = (load_readings(root, frame['route']).get(frame['segment']) or {}).get('frames', 0) if segment else 0
        if segment is None or frame['decoded'] % VIDEO_STEP or frame['decoded'] // VIDEO_STEP >= frames_read:
            missing += 1
            continue
        checked += 1
        said = segment.get(frame['decoded'] // VIDEO_STEP, [])
        for kind in KINDS:
            wanted = [box for user, box in frame['users'] if user == kind and box[3] - box[1] >= COMPLETE_HEIGHT[kind]]
            dont_care = frame['ignore'] + [box for user, box in frame['users']
                                           if user == kind and box[3] - box[1] < COMPLETE_HEIGHT[kind]]
            found = [{'box': box, 'score': score} for cls, score, box in said
                     if kind_of(cls) == kind and score >= COUNT_SCORE and box[3] - box[1] >= COMPLETE_HEIGHT[kind]
                     and not (dont_care and _in_ignore(box, dont_care))]
            item = score_frame(wanted, found, score_threshold=0.0, iou_threshold=0.5)
            hit = set(item['found_truth_indexes'])
            for place, box in enumerate(wanted):
                row = rows[kind][size_bin(box[3] - box[1])]
                row['truth'] += 1
                row['found'] += int(place in hit)
            for box in item['false_boxes']:
                rows[kind][size_bin(box[3] - box[1])]['false'] += 1
    for kind in KINDS:
        for row in rows[kind].values():
            row['recall'] = row['found'] / row['truth'] if row['truth'] else None
    return {'frames': checked, 'not_read': missing, 'by_kind': rows}


def check_text(found: dict) -> str:
    lines = [f"road users check: {found['frames']} ticked frames checked, {found['not_read']} not read by the detector"]
    if not found['frames']:
        lines.append('nothing to check yet: tick "road users complete" on frames in Sign review, read their drives '
                     '(nnslr road-users), then check again')
    for kind, bins in found['by_kind'].items():
        parts = [f"{name} px: {row['found']}/{row['truth']} found" + (f" ({row['recall']:.0%})" if row['recall'] is not None
                                                                     else '') + f", {row['false']} false"
                 for name, row in bins.items()]
        lines.append(f'  {kind}: ' + '; '.join(parts))
    return '\n'.join(lines)
