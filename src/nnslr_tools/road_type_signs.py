"""The motorway signs the comma confirms on the drives of the road types: an input of the road-type network (how long
ago a motorway or expressway sign was confirmed), derived offline the way the car derives it.

On the car, speedvisiond reads the road camera about 4 times a second with SignNet and the classifier
(vision.read_signs), keeps the signs read sure enough (vision.MIN_CONFIDENCE) and gives their classes to
vision.SpeedLimitMemory. That confirms a sign seen in CONFIRM_FRAMES frames within CONFIRM_SECONDS and notes when a
motorway or expressway sign was confirmed (_road_sign_at). The road-type network gets the seconds since then
(sign_features). In Switzerland the motorway sign (4.01) stands where the entry ramp starts, so it is seen as the ramp
begins, and the network can learn that the carriageway comes after that ramp.

Here the car's own code does the same on the recorded drives: the car's vision.py and its two networks from the
sunnypilot clone (read only: openpilot/sunnypilot/speed_limits and its models), on each segment's fcamera.hevc
decoded to NV12 by ffmpeg (the road camera's pixel format), one decoded frame in VIDEO_STEP (20 fps: 4 a second, as
speedvisiond), the settings of the car's release in place. The reviewed sign labels are not used: most drives were reviewed before the other signs had boxes, and
the car never has them; what the car's networks miss or misread, the road-type network meets the same way.

Per drive, data/derived/road_type_signs/<route>.json: the fingerprint of the code and networks used, the frames read
per segment, every frame with a sign read sure enough (time in seconds from the start of the drive, class, confidence;
'boxes', in the same order, their boxes in the frame's pixels, what vision.SignDistance measures the sign by) and the
road signs SpeedLimitMemory confirms (time, road).

The brain's pictures come from the same pass (pictures=True: every drive the brain learns from and the certified ones): each
frame read through the car's vision.BrainInput, the same class and the same NV12 frame as speedvisiond, saved per segment
as data/derived/brain_frames/<route>/<segment>.npy, uint8 (frames read, BRAIN_HEIGHT, BRAIN_WIDTH, 3), and their count
per segment in the record ('pictures'). A drive asked with pictures and without them is read again; a drive that has them
keeps them when it is read again. Built again when the vision code, a network or the video
changed. On the GPU when there is one (device 'auto'): the video decoded by NVDEC (ffmpeg -hwaccel cuda, bit-exact: HEVC
decoding is), the two networks under onnxruntime-gpu (their outputs differ from the CPU's by about 1e-6); else, or
with device 'cpu', ffmpeg and onnxruntime on the CPU.

The same reading gives the green boards towards a motorway or an expressway the car's memory confirms (its
BOARD_SIGNS, the classifier's other_sign:motorway_direction): the network's input board, the seconds since one (record
'boards'). In Switzerland they stand where the entry ramp starts, and at the interchanges; the car never takes a limit
from them. Only a classifier with that class reads them: the car's of the sunnypilot clone, or another one chosen
(classifier: a v2 reader-best.pt, exported once to ONNX into derived/road_type_signs/classifiers/).

Two sources of the sign times (SIGN_SOURCES):
- 'car': the signs the car's code confirms, as above. On the first 12 drives (October 2026) the car's classifier
  confirmed no motorway sign at all: its motorway_start probability stays below 0.03 on the boxes of the reviewed
  4.01 signs, so today the car gives the network nothing.
- 'map': where the sign stands, from the road types read so far, a person's over the map's (the frames' map_label: an
  input, as on the car, never a label of the training): the first second of every ramp that leads from another road
  onto the motorway or expressway (road_type_metrics.ramp_entries, not from the motorway itself: an interchange has no
  4.01). The reviewed 4.01 signs were seen from 7 s before to 6 s after that second; training moves the time by up to
  SIGN_JITTER seconds (road_type_net). This is the input the network gets once the car reads the sign.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nnslr_tools.media import FRAME_SIZE, FULL_SIZE, REDUCE_FILTER

SIGNS_RELATIVE = Path('derived') / 'road_type_signs'
RAW_RELATIVE = Path('raw') / 'routes'
PACKAGE_RELATIVE = Path('openpilot') / 'sunnypilot' / 'speed_limits'
SUNNYPILOT_ENV, SUNNYPILOT_DEFAULT = 'NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'
NETWORK_FILES = {'finder': 'signnet.onnx', 'classifier': 'classifier.onnx'}
RELEASE_MANIFEST = 'speed_limits.json'
NOT_A_SIGN = 'not_a_sign'  # read sure enough, but neither a speed nor a road sign: left out of the cache
CACHE_VERSION = 3  # 2: the boards; 3: the boxes of the signs
PROBE_ATTEMPTS = 4
PROBE_PAUSE_S = 10
CLASSIFIERS_RELATIVE = SIGNS_RELATIVE / 'classifiers'
VIDEO_FPS = 20
# 20 fps / 5: the 4 frames a second of speedvisiond. Its PERIOD is 0.2 s, but it waits from a frame for the next one on the
# camera's 50 ms grid, and its logs read 4.1 a second.
VIDEO_STEP = 5
SEGMENT_SECONDS = 60
STEPS_PER_SECOND = VIDEO_FPS // VIDEO_STEP
PICTURES_RELATIVE = Path('derived') / 'brain_frames'
# The roads whose confirmed sign the network gets (vision.SpeedLimitMemory sets _road_sign_at for these).
SIGN_ROADS = ('motorway', 'expressway')
# The input: [min(age, SIGN_HORIZON) / SIGN_SCALE, 1 when a sign was confirmed within SIGN_HORIZON seconds]. Never, or
# longer ago: NO_SIGN.
SIGN_HORIZON = 120.0
SIGN_SCALE = 60.0
NO_SIGN = (SIGN_HORIZON / SIGN_SCALE, 0.0)
SIGN_SOURCES = ('map', 'car')
DEFAULT_SOURCE = 'map'


def sign_features(age: float | None) -> tuple[float, float]:
    """The network's sign input for a sign confirmed age seconds ago (None: never)."""
    if age is None or not np.isfinite(age) or age < 0 or age > SIGN_HORIZON:
        return NO_SIGN
    return (float(age) / SIGN_SCALE, 1.0)


def package_path(sunnypilot: Path | None = None) -> Path:
    """The car's speed_limits package in the sunnypilot clone (NNSLR_SUNNYPILOT_CLONE, else /root/sunnypilot)."""
    base = Path(sunnypilot) if sunnypilot else Path(os.environ.get(SUNNYPILOT_ENV) or SUNNYPILOT_DEFAULT)
    return base / PACKAGE_RELATIVE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def network_paths(package: Path, classifier: Path | None = None) -> dict[str, Path]:
    """The ONNX of the two networks: the car's of the sunnypilot clone, the classifier another one when given (an
    ONNX with its .contract.json beside it)."""
    paths = {name: Path(package) / 'models' / file for name, file in NETWORK_FILES.items()}
    if classifier is not None:
        paths['classifier'] = Path(classifier)
    return paths


def release_settings(package: Path) -> dict:
    """The settings of the car's release, as speedvisiond gives them to vision.apply_settings."""
    try:
        settings = json.loads((Path(package) / 'models' / RELEASE_MANIFEST).read_text('utf-8')).get('settings')
    except (OSError, ValueError, AttributeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def fingerprint(package: Path, classifier: Path | None = None) -> dict:
    """What the readings depend on: the car's vision.py and the two networks (sha256), its settings, the rate."""
    package = Path(package)
    return {'version': CACHE_VERSION, 'video_step': VIDEO_STEP, 'settings': release_settings(package),
            'vision': _sha256(package / 'vision.py'),
            **{name: _sha256(path) for name, path in network_paths(package, classifier).items()}}


def classifier_onnx(root: Path, weights: Path) -> Path:
    """A v2 classifier (reader-best.pt) as the car runs it: exported once (checked against PyTorch) to
    derived/road_type_signs/classifiers/<sha256 of the weights>/classifier.onnx, with its contract."""
    from nnslr_tools.exporting import export_reader_onnx
    folder = Path(root) / CLASSIFIERS_RELATIVE / _sha256(Path(weights))[:16]
    onnx = folder / NETWORK_FILES['classifier']
    if not (onnx.is_file() and onnx.with_name(onnx.name + '.contract.json').is_file()):
        folder.mkdir(parents=True, exist_ok=True)
        export_reader_onnx(Path(weights), onnx, verify=True)
    return onnx


def classifier_classes(path: Path) -> list[str]:
    """The classes of a classifier's ONNX, from its contract."""
    path = Path(path)
    return json.loads(path.with_name(path.name + '.contract.json').read_text('utf-8'))['output']['classes']


def load_vision(package: Path):
    """The car's vision.py as speedvisiond starts it, with the settings of its release."""
    from nnslr_tools.comma_trial import load_vision as load
    vision = load(Path(package) / 'vision.py')
    vision.apply_settings(release_settings(package))
    return vision


class _Network:
    """One of the car's networks under onnxruntime, on the GPU when asked and it works there (one inference on zeros
    tries it), else on one CPU thread: numpy in, numpy out."""

    def __init__(self, path: Path, gpu: bool = False):
        import onnxruntime
        from nnslr_tools.onnx_device import CPU, CUDA
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = None
        if gpu and CUDA in onnxruntime.get_available_providers():
            if hasattr(onnxruntime, 'preload_dlls'):
                onnxruntime.preload_dlls()
            try:
                self.session = onnxruntime.InferenceSession(str(path), options, providers=[CUDA, CPU])
                shape = [size if isinstance(size, int) else 1 for size in self.session.get_inputs()[0].shape]
                self.session.run(None, {self.session.get_inputs()[0].name: np.zeros(shape, dtype=np.float32)})
            except Exception:
                self.session = None
        if self.session is None:
            self.session = onnxruntime.InferenceSession(str(path), options, providers=[CPU])
        self.name = self.session.get_inputs()[0].name
        self.gpu = self.session.get_providers()[0] == CUDA

    def __call__(self, x):
        return self.session.run(None, {self.name: np.asarray(x, dtype=np.float32)})[0]


_WORKER: dict = {}


def _worker_start(package: str, gpu: bool = False, classifier: str | None = None, cards: list[str] | None = None,
                  turn=None) -> None:
    if cards:
        # This process's card, the next in turn, before its networks and its ffmpeg open one (#36): the NVDEC of one
        # V100 is what holds the reading back.
        with turn.get_lock():
            place = turn.value
            turn.value += 1
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=cards[place % len(cards)])
    paths = network_paths(Path(package), Path(classifier) if classifier else None)
    _WORKER.update(vision=load_vision(Path(package)), finder=_Network(paths['finder'], gpu),
                   classifier=_Network(paths['classifier'], gpu), classes=classifier_classes(paths['classifier']),
                   nvdec=gpu)


def _video_size(video: Path) -> tuple[int, int]:
    """The size of a video. The videos are on the NAS, on a soft NFS mount: a read that fails there is tried again."""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of',
               'csv=p=0', str(video)]
    for attempt in range(PROBE_ATTEMPTS):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode == 0:
            width, height = (int(value) for value in done.stdout.strip().split(',')[:2])
            return width, height
        if attempt + 1 < PROBE_ATTEMPTS:
            time.sleep(PROBE_PAUSE_S)
    raise RuntimeError(f'ffprobe cannot read {video}: {done.stderr.strip()}')


def read_segment(task: tuple[str, int, str | None]) -> dict:
    """The car's read_signs on one segment's road video, one frame in VIDEO_STEP: {'segment', 'frames' (read), 'signs':
    [[seconds into the segment, [[class, confidence], ...]], ...] (the frames with a sign read sure enough), 'boxes' (their
    boxes, in the same order), 'gpu' (the networks ran there), 'nvdec' (the video was decoded there), 'pictures' (how many
    were saved to the task's path, None without one)}. NVDEC first when the worker may use the GPU: a video it cannot decode
    is read again on the CPU."""
    video, segment, pictures = task
    width, height = _video_size(Path(video))
    # A video at full definition is read at the size of the others: these signs are big, no crop of the frame is needed.
    reduce = (width, height) == FULL_SIZE
    if reduce:
        width, height = FRAME_SIZE
    if _WORKER.get('nvdec'):
        result = _read_video(video, width, height, nvdec=True, reduce=reduce, pictures=pictures)
        if result['frames']:
            return {'segment': segment, **result}
    return {'segment': segment, **_read_video(video, width, height, nvdec=False, reduce=reduce, pictures=pictures)}


def _save_pictures(path: Path, pictures: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npy')
    np.save(temporary, np.stack(pictures) if pictures else np.zeros((0, 0, 0, 3), dtype=np.uint8))
    temporary.replace(path)


def _read_video(video: str, width: int, height: int, *, nvdec: bool, reduce: bool = False,
                pictures: str | None = None) -> dict:
    vision, finder, classifier, classes = (_WORKER[name] for name in ('vision', 'finder', 'classifier', 'classes'))
    size = width * height * 3 // 2
    chosen = f'select=not(mod(n\\,{VIDEO_STEP}))' + (',' + REDUCE_FILTER if reduce else '')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *(['-hwaccel', 'cuda'] if nvdec else []), '-threads', '1',
               '-i', str(video), '-vf', chosen, '-vsync', '0', '-pix_fmt', 'nv12', '-f',
               'rawvideo', '-']
    signnet_input = vision.SignNetInput(width, height)
    brain_input = vision.BrainInput(width, height) if pictures else None
    found, boxes, saved, count = [], [], [], 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        while True:
            buffer = process.stdout.read(size)
            if len(buffer) < size:
                break
            data = np.frombuffer(buffer, dtype=np.uint8)
            y, uv = vision.nv12_planes(data, width, height, width, width * height)
            signs = vision.read_signs(y, uv, signnet_input, finder, classifier, classes)
            kept = [sign for sign in signs if sign['confidence'] >= vision.MIN_CONFIDENCE and sign['class'] != NOT_A_SIGN]
            if kept:
                found.append([round(count * VIDEO_STEP / VIDEO_FPS, 3),
                              [[sign['class'], round(sign['confidence'], 4)] for sign in kept]])
                boxes.append([[round(float(value), 2) for value in sign['box']] for sign in kept])
            if brain_input is not None:
                saved.append(brain_input.planes(y, uv).transpose(1, 2, 0))
            count += 1
    if pictures and count:
        _save_pictures(Path(pictures), saved)
    return {'frames': count, 'signs': found, 'boxes': boxes, 'gpu': finder.gpu, 'nvdec': nvdec,
            'pictures': count if pictures and count else None}


def confirmations(vision, readings: list[tuple[float, list]]) -> list[list]:
    """The motorway and expressway signs SpeedLimitMemory confirms from the readings [(seconds, [[class, confidence],
    ...])] in time order: [[seconds, road], ...], the moments it sets _road_sign_at, as speedvisiond's memory does. A
    frame without a sign changes nothing there, so only the frames with one are given."""
    memory, found = vision.SpeedLimitMemory(), []
    for now, signs in readings:
        classes = [name for name, _ in signs]
        speeds = [vision.class_speed(name) for name in classes]
        before = memory._road_sign_at
        memory.update(float(now), [speed for speed in speeds if speed is not None], classes)
        if memory._road_sign_at != before:
            found.append([float(now), memory.road])
    return found


def board_confirmations(vision, readings: list[tuple[float, list]]) -> list[float]:
    """When SpeedLimitMemory confirms a green board towards a motorway (its BOARD_SIGNS) from the readings, as
    speedvisiond's memory does: [seconds, ...], the moments it sets _board_at. Empty with a vision.py without them."""
    if not hasattr(vision, 'BOARD_SIGNS'):
        return []
    memory, found = vision.SpeedLimitMemory(), []
    for now, signs in readings:
        classes = [name for name, _ in signs]
        speeds = [vision.class_speed(name) for name in classes]
        before = memory._board_at
        memory.update(float(now), [speed for speed in speeds if speed is not None], classes)
        if memory._board_at != before:
            found.append(float(now))
    return found


def number_confirmations(vision, readings: list[tuple[float, list]]) -> list[list]:
    """The speed signs SpeedLimitMemory confirms from the readings, as speedvisiond's memory does: [[seconds, km/h],
    ...], the moments a number (not the end of a limit) became the limit held. The junction head's measure counts the
    intersections a held number would end at (Poland: a sign holds to the next intersection)."""
    memory, found, last = vision.SpeedLimitMemory(), [], None
    for now, signs in readings:
        classes = [name for name, _ in signs]
        speeds = [speed for speed in (vision.class_speed(name) for name in classes) if speed is not None]
        memory.update(float(now), speeds, classes)
        if memory.sign_read and memory.speed_limit > 0 and float(memory.speed_limit) in [float(v) for v in speeds]:
            # One sign read for some frames is one confirmation: a new one after a pause or with another number.
            if last is None or last[1] != memory.speed_limit or float(now) - last[0] > 2.0:
                found.append([float(now), float(memory.speed_limit)])
            last = (float(now), float(memory.speed_limit))
    return found


def cache_path(root: Path, route: str) -> Path:
    return Path(root) / SIGNS_RELATIVE / f'{route}.json'


def picture_path(root: Path, route: str, segment: int) -> Path:
    """The brain's pictures of one segment: uint8 (frames read, BRAIN_HEIGHT, BRAIN_WIDTH, 3), one every VIDEO_STEP."""
    return Path(root) / PICTURES_RELATIVE / route / f'{int(segment)}.npy'


def pictures_ready(root: Path, record: dict | None) -> bool:
    """The drive's record has the brain's pictures of every segment it read, and their files are there."""
    if not record or not record.get('pictures'):
        return False
    return all(record['pictures'].get(segment) == frames and picture_path(root, record['route'], int(segment)).is_file()
               for segment, frames in record['frames'].items() if frames)


def load_pictures(root: Path, route: str, segment: int) -> np.ndarray | None:
    """One segment's pictures, mapped from the disk (None without them)."""
    path = picture_path(root, route, segment)
    return np.load(path, mmap_mode='r') if path.is_file() else None


def _videos(root: Path, route: str) -> dict[int, Path]:
    folder = Path(root) / RAW_RELATIVE / route
    videos = {}
    for path in folder.glob('*/fcamera.hevc') if folder.is_dir() else []:
        if path.parent.name.isdigit():
            videos[int(path.parent.name)] = path
    return dict(sorted(videos.items()))


def _video_stamp(videos: dict[int, Path]) -> list:
    return [[segment, path.stat().st_size, int(path.stat().st_mtime)] for segment, path in videos.items()]


def load_signs(root: Path, route: str) -> dict | None:
    path = cache_path(root, route)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _save_signs(root: Path, route: str, segments: list[dict], stamp, videos: list, classifier: Path | None,
                package: Path) -> dict:
    """The record of a drive read to the end, written to its cache. 'pictures' only when every segment saved them."""
    vision = load_vision(package)
    segments.sort(key=lambda item: item['segment'])
    readings = [(item['segment'] * SEGMENT_SECONDS + seconds, signs)
                for item in segments for seconds, signs in item['signs']]
    record = {'route': route, 'fingerprint': stamp, 'videos': videos,
              'frames': {str(item['segment']): item['frames'] for item in segments},
              'readings': [[round(t, 3), signs] for t, signs in readings],
              'boxes': [boxes for item in segments for boxes in item['boxes']],
              'confirmed': confirmations(vision, readings),
              'boards': board_confirmations(vision, readings),
              'classifier': str(classifier) if classifier is not None else None}
    if all(item.get('pictures') == item['frames'] for item in segments if item['frames']):
        record['pictures'] = {str(item['segment']): item['pictures'] for item in segments if item['frames']}
    path = cache_path(root, route)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)
    return record


def build_signs(root: Path, routes: list[str], *, package: Path | None = None, workers: int | None = None,
                device: str = 'auto', gpus: list[str] | None = None, classifier: Path | None = None,
                pictures: bool = False, log=print) -> dict[str, dict]:
    """The readings of every drive (cached; read again when the code, a network or a video changed): {route: record}.
    The segments of all drives are read in parallel by workers processes (default half the cores), on the GPU when
    there is one (device 'auto'; 'cpu' never), each process on a card of gpus in turn (nvidia-smi's numbers; None: every
    process on the first card this process sees). classifier: a v2 reader-best.pt to read with instead of the car's (classifier_onnx).
    pictures: also the brain's pictures (picture_path), read again where they are missing; a drive that has them gets
    them again whenever it is read."""
    package = package_path() if package is None else Path(package)
    onnx = classifier_onnx(root, classifier) if classifier is not None else None
    stamp = fingerprint(package, onnx)
    records, tasks, stamps = {}, [], {}
    for route in routes:
        videos = _videos(root, route)
        stamps[route] = _video_stamp(videos)
        record = load_signs(root, route)
        if record and record.get('fingerprint') == stamp and record.get('videos') == stamps[route] and \
                (not pictures or pictures_ready(root, record)):
            records[route] = record
            continue
        keep = pictures or bool(record and record.get('pictures'))
        tasks += [(route, segment, str(path), str(picture_path(root, route, segment)) if keep else None)
                  for segment, path in videos.items()]
    if tasks:
        workers = workers or max(1, min(len(tasks), (os.cpu_count() or 4) // 2))
        which = f'the classifier {classifier}' if classifier is not None else "the car's classifier"
        log(f'road-type signs: reading {len(tasks)} segments of {len({t[0] for t in tasks})} drives with the car\'s '
            f'SignNet and {which}, 4 frames a second, {workers} processes'
            + (f', the brain\'s pictures of {sum(1 for t in tasks if t[3])} of them' if any(t[3] for t in tasks) else ''))
        results: dict[str, list] = {}
        left = Counter(task[0] for task in tasks)
        gpu = device != 'cpu'
        cards = [str(card) for card in gpus] if gpu and gpus else None
        context = multiprocessing.get_context()
        if cards and len(cards) > 1:
            log(f"road-type signs: the processes on the cards {', '.join(cards)} in turn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_start,
                                 initargs=(str(package), gpu, str(onnx) if onnx else None, cards,
                                           context.Value('i', 0) if cards else None)) as pool:
            done = 0
            for (route, _, _, _), result in zip(tasks, pool.map(read_segment, [(v, s, p) for _, s, v, p in tasks])):
                if cards and not result['gpu']:
                    # A process whose card does not work stops the reading: on the cards given, never the CPU.
                    raise RuntimeError(f"road-type signs: segment {result['segment']} of {route} was read by the "
                                       f"networks on the CPU, not on a card of {', '.join(cards)}")
                if cards and not result['nvdec']:
                    log(f"road-type signs: segment {result['segment']} of {route}: NVDEC could not decode it, decoded "
                        "on the CPU")
                results.setdefault(route, []).append(result)
                done += 1
                left[route] -= 1
                # A drive is kept as soon as it is read: a failure further on does not cost it again.
                if not left[route]:
                    records[route] = _save_signs(root, route, results.pop(route), stamp, stamps[route], classifier, package)
                if done == 1:
                    log('road-type signs: the networks on the ' + ('GPU' if result['gpu'] else 'CPU') + ', the video decoded'
                        ' on the ' + ('GPU (NVDEC)' if result['nvdec'] else 'CPU'))
                if done % 25 == 0 or done == len(tasks):
                    log(f'road-type signs: {done}/{len(tasks)} segments')
    return records


def sign_times(record: dict | None) -> list[float]:
    """When a motorway or expressway sign was confirmed in a drive (seconds from its start, in order)."""
    if not record:
        return []
    return sorted(float(t) for t, road in record.get('confirmed', []) if road in SIGN_ROADS)


def board_times(record: dict | None) -> list[float]:
    """When a green board towards a motorway was confirmed in a drive (seconds from its start, in order)."""
    return sorted(float(t) for t in (record or {}).get('boards', []))


def frame_seconds(frame: dict) -> float:
    """The time of a 1 Hz frame of the dataset in the drive (its segment's minute and its index)."""
    return float(frame['segment'] * SEGMENT_SECONDS + frame['index'])


def sign_ages(seconds, times) -> np.ndarray:
    """Seconds since the last sign at or before each moment (inf before the first, or without any)."""
    seconds = np.asarray(seconds, dtype=np.float64)
    times = np.sort(np.asarray(times, dtype=np.float64))
    if not len(times):
        return np.full(len(seconds), np.inf)
    last = np.searchsorted(times, seconds, side='right') - 1
    return np.where(last >= 0, seconds - times[np.maximum(last, 0)], np.inf)


def features_at(seconds, times) -> np.ndarray:
    """The sign input (len(seconds), 2) float32 at each moment, sign_features of sign_ages."""
    ages = sign_ages(seconds, times)
    out = np.tile(np.asarray(NO_SIGN, dtype=np.float32), (len(ages), 1))
    seen = (ages >= 0) & (ages <= SIGN_HORIZON)
    out[seen, 0] = ages[seen] / SIGN_SCALE
    out[seen, 1] = 1.0
    return out


def frame_sign_features(frames: list[dict], times: list[float]) -> np.ndarray:
    """The sign input of each frame of a drive (frames, 2): the age of the last sign confirmed at or before it."""
    return features_at([frame_seconds(frame) for frame in frames], times)


def map_sign_times(frames: list[dict], classes: tuple[str, ...]) -> list[float]:
    """Where the map puts the motorway sign in a drive (seconds from its start): the first second of every ramp from
    another road onto the motorway or expressway (frames: the drive's 1 Hz frames with their 'map_label' and
    'map_lead', not their label: a drive a person has not reviewed has its signs too; a ramp a person corrected has
    none)."""
    from nnslr_tools.road_type_metrics import ramp_entries
    labels = np.array([frame['map_label'] for frame in frames], dtype=np.int64)
    lead = np.array([bool(frame.get('map_lead')) for frame in frames], dtype=bool)
    return [frame_seconds(frames[entry['ramp']]) for entry in ramp_entries(labels, classes, lead)
            if entry['before'] != 'motorway']
