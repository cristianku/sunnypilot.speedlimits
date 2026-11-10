"""The motorway signs the comma confirms on the drives of the road types: an input of the road-type network (how long
ago a motorway or expressway sign was confirmed), derived offline the way the car derives it.

On the car, speedvisiond reads the road camera about 4 times a second with SignNet and the classifier
(vision.read_signs), lets the applicability network rule them (applicability.Applicability, with the car's speed)
and leaves out those of another road (applicability.taken), keeps the signs read sure enough (vision.MIN_CONFIDENCE)
and gives their classes to vision.SpeedLimitMemory. That confirms a sign seen in CONFIRM_FRAMES frames within CONFIRM_SECONDS and notes when a
motorway or expressway sign was confirmed (_road_sign_at). The road-type network gets the seconds since then
(sign_features). In Switzerland the motorway sign (4.01) stands where the entry ramp starts, so it is seen as the ramp
begins, and the network can learn that the carriageway comes after that ramp.

Here the car's own code does the same on the recorded drives: the car's vision.py and applicability.py from the
sunnypilot clone (read only: openpilot/sunnypilot/speed_limits) with the networks of a release, its models folder by
default (SignNet, the classifier, and the applicability network when the release has one), or another folder exported
as Publish exports one (models: the brain's, sign_networks), on each segment's fcamera.hevc decoded to NV12 by ffmpeg
(the road camera's pixel format), one decoded frame in VIDEO_STEP (20 fps: 4 a second, as speedvisiond), the settings
of that release in place. The applicability network gets the car's speed of each frame (vEgo of carState at the
frame's exposure, from the drive's joined logs; unknown without them, as the car gives it before its first carState).
The signs it gives to another road are left out as on the car, and kept apart in the record ('other_road'). The reviewed sign labels are not used: most drives were reviewed before the other signs had boxes, and
the car never has them; what the car's networks miss or misread, the road-type network meets the same way.

Per drive, data/derived/road_type_signs/<route>.json: the fingerprint of the code and networks used, the frames read
per segment, every frame with a sign read sure enough (time in seconds from the start of the drive, class, confidence;
'boxes', in the same order, their boxes in the frame's pixels, what vision.SignDistance measures the sign by) and the
road signs SpeedLimitMemory confirms (time, road); 'other_road', the frames with a sign the applicability network left
out (time, [[class, confidence], ...]); 'networks', the ids of the networks read with (their contracts' checkpoint).

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

import bisect
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
RELEASES_RELATIVE = SIGNS_RELATIVE / 'releases'  # the readings by a release other than the clone's (cache_path)
RAW_RELATIVE = Path('raw') / 'routes'
PACKAGE_RELATIVE = Path('openpilot') / 'sunnypilot' / 'speed_limits'
SUNNYPILOT_ENV, SUNNYPILOT_DEFAULT = 'NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'
NETWORK_FILES = {'finder': 'signnet.onnx', 'classifier': 'classifier.onnx'}
DECIDER_FILE = 'decider.onnx'  # the applicability network, when the release has one
RELEASE_MANIFEST = 'speed_limits.json'
NOT_A_SIGN = 'not_a_sign'  # read sure enough, but neither a speed nor a road sign: left out of the cache
CACHE_VERSION = 4  # 2: the boards; 3: the boxes of the signs; 4: the applicability network, a sign of another road left out
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


def models_path(package: Path, models: Path | None = None) -> Path:
    """The folder of the release read with: models, else the car's of the sunnypilot clone (package/models)."""
    return Path(models) if models is not None else Path(package) / 'models'


def network_paths(package: Path, classifier: Path | None = None, models: Path | None = None) -> dict[str, Path]:
    """The ONNX of the networks of a release (models_path): SignNet, the classifier, another one when given (an ONNX
    with its .contract.json beside it), and the applicability network ('decider') when the release has one."""
    folder = models_path(package, models)
    paths = {name: folder / file for name, file in NETWORK_FILES.items()}
    if classifier is not None:
        paths['classifier'] = Path(classifier)
    if (folder / DECIDER_FILE).is_file():
        paths['decider'] = folder / DECIDER_FILE
    return paths


def release_settings(package: Path, models: Path | None = None) -> dict:
    """The settings of the release (models_path), as speedvisiond gives them to vision.apply_settings."""
    try:
        settings = json.loads((models_path(package, models) / RELEASE_MANIFEST).read_text('utf-8')).get('settings')
    except (OSError, ValueError, AttributeError):
        return {}
    return settings if isinstance(settings, dict) else {}


def fingerprint(package: Path, classifier: Path | None = None, models: Path | None = None) -> dict:
    """What the readings depend on: the car's vision.py, its applicability.py with an applicability network, the
    networks (sha256), the release's settings, the rate."""
    package = Path(package)
    paths = network_paths(package, classifier, models)
    return {'version': CACHE_VERSION, 'video_step': VIDEO_STEP, 'settings': release_settings(package, models),
            'vision': _sha256(package / 'vision.py'),
            **({'applicability': _sha256(package / 'applicability.py')} if 'decider' in paths else {}),
            **{name: _sha256(path) for name, path in paths.items()}}


def network_ids(paths: dict[str, Path]) -> dict[str, str | None]:
    """The id of each network read with, as its contract names its checkpoint (None when it does not)."""
    found = {}
    for name, path in paths.items():
        try:
            found[name] = json.loads(Path(path).with_name(Path(path).name + '.contract.json').read_text('utf-8')).get('checkpoint')
        except (OSError, ValueError, AttributeError):
            found[name] = None
    return found


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


def load_vision(package: Path, models: Path | None = None):
    """The car's vision.py as speedvisiond starts it, with the settings of the release (models_path)."""
    from nnslr_tools.comma_trial import load_vision as load
    vision = load(Path(package) / 'vision.py')
    vision.apply_settings(release_settings(package, models))
    return vision


def load_applicability(package: Path):
    """The car's applicability.py as a module of its own: it imports nothing of openpilot, the vision module is given
    to it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('comma_speed_limits_applicability', Path(package) / 'applicability.py')
    if spec is None or spec.loader is None:
        raise RuntimeError(f"the car's applicability.py cannot be read in {package}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Network:
    """One of the car's networks under onnxruntime, on the GPU when asked and it works there (one inference on zeros
    tries it), else on one CPU thread: numpy in, numpy out (the first input, the first output)."""

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
                self.session.run(None, {one.name: np.zeros([size if isinstance(size, int) else 1 for size in one.shape],
                                                           dtype=np.float32) for one in self.session.get_inputs()})
            except Exception:
                self.session = None
        if self.session is None:
            self.session = onnxruntime.InferenceSession(str(path), options, providers=[CPU])
        self.name = self.session.get_inputs()[0].name
        self.gpu = self.session.get_providers()[0] == CUDA

    def __call__(self, x):
        return self.session.run(None, {self.name: np.asarray(x, dtype=np.float32)})[0]


class _StepNetwork(_Network):
    """The applicability network, as the car's runner.StepNetwork: numpy in by name, numpy (float32) out by name."""

    def __init__(self, path: Path, gpu: bool = False):
        super().__init__(path, gpu)
        self.outputs = [one.name for one in self.session.get_outputs()]

    def __call__(self, **inputs):
        feed = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
        return dict(zip(self.outputs, self.session.run(None, feed), strict=True))


_WORKER: dict = {}


def _ruler(width: int, height: int):
    """The worker's applicability network for frames of this size (the car's applicability.Applicability), None
    without one in the release."""
    if _WORKER.get('decider') is None:
        return None
    rulers = _WORKER.setdefault('rulers', {})
    if (width, height) not in rulers:
        rulers[(width, height)] = _WORKER['applicability'].Applicability(
            _WORKER['decider'], _WORKER['decider_contract'], _WORKER['vision'], width, height)
    return rulers[(width, height)]


def _worker_start(package: str, gpu: bool = False, classifier: str | None = None, cards: list[str] | None = None,
                  turn=None, models: str | None = None) -> None:
    if cards:
        # This process's card, the next in turn, before its networks and its ffmpeg open one (#36): the NVDEC of one
        # V100 is what holds the reading back.
        with turn.get_lock():
            place = turn.value
            turn.value += 1
        os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=cards[place % len(cards)])
    models = Path(models) if models else None
    paths = network_paths(Path(package), Path(classifier) if classifier else None, models)
    decider = paths.get('decider')
    _WORKER.update(vision=load_vision(Path(package), models), finder=_Network(paths['finder'], gpu),
                   classifier=_Network(paths['classifier'], gpu), classes=classifier_classes(paths['classifier']),
                   nvdec=gpu, decider=_StepNetwork(decider, gpu) if decider else None,
                   decider_contract=json.loads(decider.with_name(decider.name + '.contract.json').read_text('utf-8'))
                   if decider else None, applicability=load_applicability(Path(package)) if decider else None, rulers={})


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


def read_segment(task: tuple) -> dict:
    """The car's read_signs on one segment's road video, one frame in VIDEO_STEP, ruled by the applicability network
    when the release has one: {'segment', 'frames' (read), 'signs': [[seconds into the segment, [[class, confidence],
    ...]], ...] (the frames with a sign read sure enough and not of another road), 'boxes' (their boxes, in the same
    order), 'other_road' (the frames with a sign read sure enough the applicability network gave to another road, as
    'signs'), 'gpu' (the networks ran there), 'nvdec' (the video was decoded there), 'pictures' (how many were saved to
    the task's path, None without one)}. task: (video, segment, pictures path or None, and the car's speed in m/s of
    each frame read, by its place among them, None where unknown; without it, unknown). NVDEC first when the worker may
    use the GPU: a video it cannot decode is read again on the CPU."""
    video, segment, pictures, *rest = task
    speeds = rest[0] if rest else None
    width, height = _video_size(Path(video))
    # A video at full definition is read at the size of the others: these signs are big, no crop of the frame is needed.
    reduce = (width, height) == FULL_SIZE
    if reduce:
        width, height = FRAME_SIZE
    if _WORKER.get('nvdec'):
        result = _read_video(video, width, height, nvdec=True, reduce=reduce, pictures=pictures, speeds=speeds)
        if result['frames']:
            return {'segment': segment, **result}
    return {'segment': segment, **_read_video(video, width, height, nvdec=False, reduce=reduce, pictures=pictures,
                                              speeds=speeds)}


def _save_pictures(path: Path, pictures: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.tmp.npy')
    np.save(temporary, np.stack(pictures) if pictures else np.zeros((0, 0, 0, 3), dtype=np.uint8))
    temporary.replace(path)


def _read_video(video: str, width: int, height: int, *, nvdec: bool, reduce: bool = False,
                pictures: str | None = None, speeds: list | None = None) -> dict:
    vision, finder, classifier, classes = (_WORKER[name] for name in ('vision', 'finder', 'classifier', 'classes'))
    ruler = _ruler(width, height)
    size = width * height * 3 // 2
    chosen = f'select=not(mod(n\\,{VIDEO_STEP}))' + (',' + REDUCE_FILTER if reduce else '')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *(['-hwaccel', 'cuda'] if nvdec else []), '-threads', '1',
               '-i', str(video), '-vf', chosen, '-vsync', '0', '-pix_fmt', 'nv12', '-f',
               'rawvideo', '-']
    signnet_input = vision.SignNetInput(width, height)
    brain_input = vision.BrainInput(width, height) if pictures else None
    found, boxes, other, saved, count = [], [], [], [], 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
        while True:
            buffer = process.stdout.read(size)
            if len(buffer) < size:
                break
            data = np.frombuffer(buffer, dtype=np.uint8)
            y, uv = vision.nv12_planes(data, width, height, width, width * height)
            signs = vision.read_signs(y, uv, signnet_input, finder, classifier, classes)
            if ruler is not None:
                # As speedvisiond: right after the classifier, with the car's speed; a sign of another road is not taken.
                ruler.rule(y, uv, signs, speeds[count] if speeds is not None and count < len(speeds) else None)
                left = [sign for sign in signs if sign.get('road') == _WORKER['applicability'].OTHER_ROAD
                        and sign['confidence'] >= vision.MIN_CONFIDENCE and sign['class'] != NOT_A_SIGN]
                if left:
                    other.append([round(count * VIDEO_STEP / VIDEO_FPS, 3),
                                  [[sign['class'], round(sign['confidence'], 4)] for sign in left]])
                signs = _WORKER['applicability'].taken(signs)
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
    return {'frames': count, 'signs': found, 'boxes': boxes, 'other_road': other,
            'gpu': finder.gpu and (ruler is None or _WORKER['decider'].gpu), 'nvdec': nvdec,
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


def frame_speeds(merged: dict | None) -> dict[int, list]:
    """The car's speed (m/s) at each frame the reading takes of a drive, as the applicability network gets it on the
    car: {segment: [vEgo of the last carState at or before the frame's exposure, None where unknown, ...]} by the frame's
    place among those read (decoded // VIDEO_STEP). Empty without the drive's joined logs."""
    from nnslr_tools import drive_logs
    if not merged:
        return {}
    points = sorted((merged.get('series') or {}).get('v_ego') or [], key=lambda point: point[0])
    times = [point[0] for point in points]
    found = {}
    for _first, segment, sof, ids in drive_logs.frame_index(merged):
        speeds = found.setdefault(segment, [])
        for decoded, t in sorted(zip(ids, sof)):
            if decoded % VIDEO_STEP:
                continue
            row = decoded // VIDEO_STEP
            at = bisect.bisect_right(times, t) - 1
            speeds.extend([None] * (row + 1 - len(speeds)))
            speeds[row] = None if at < 0 or points[at][1] is None else round(float(points[at][1]) / 3.6, 3)
    return found


def cache_path(root: Path, route: str, models: Path | None = None) -> Path:
    """The readings of a drive: by the car's release of the sunnypilot clone (models None), else by the release in
    models, each its own (RELEASES_RELATIVE/<the sha256 of its folder>/), so that the brain's networks and the clone's
    (Models › Regression) never read a drive again in turn."""
    if models is None:
        return Path(root) / SIGNS_RELATIVE / f'{route}.json'
    key = hashlib.sha256(str(Path(models).resolve()).encode()).hexdigest()[:16]
    return Path(root) / RELEASES_RELATIVE / key / f'{route}.json'


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


def load_signs(root: Path, route: str, models: Path | None = None) -> dict | None:
    path = cache_path(root, route, models)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _save_signs(root: Path, route: str, segments: list[dict], stamp, videos: list, classifier: Path | None,
                package: Path, models: Path | None = None, networks: dict | None = None, speeds: bool = False) -> dict:
    """The record of a drive read to the end, written to its cache. 'pictures' only when every segment saved them."""
    vision = load_vision(package, models)
    segments.sort(key=lambda item: item['segment'])
    readings = [(item['segment'] * SEGMENT_SECONDS + seconds, signs)
                for item in segments for seconds, signs in item['signs']]
    record = {'route': route, 'fingerprint': stamp, 'videos': videos,
              'frames': {str(item['segment']): item['frames'] for item in segments},
              'readings': [[round(t, 3), signs] for t, signs in readings],
              'boxes': [boxes for item in segments for boxes in item['boxes']],
              'confirmed': confirmations(vision, readings),
              'boards': board_confirmations(vision, readings),
              'other_road': [[round(item['segment'] * SEGMENT_SECONDS + seconds, 3), signs]
                             for item in segments for seconds, signs in item.get('other_road') or []],
              'networks': networks, 'speeds': speeds,
              'classifier': str(classifier) if classifier is not None else None}
    if all(item.get('pictures') == item['frames'] for item in segments if item['frames']):
        record['pictures'] = {str(item['segment']): item['pictures'] for item in segments if item['frames']}
    path = cache_path(root, route, models)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)
    return record


def build_signs(root: Path, routes: list[str], *, package: Path | None = None, workers: int | None = None,
                device: str = 'auto', gpus: list[str] | None = None, classifier: Path | None = None,
                pictures: bool = False, models: Path | None = None, speeds_of=None, log=print) -> dict[str, dict]:
    """The readings of every drive (cached; read again when the code, a network or a video changed): {route: record}.
    The segments of all drives are read in parallel by workers processes (default half the cores), on the GPU when
    there is one (device 'auto'; 'cpu' never), each process on a card of gpus in turn (nvidia-smi's numbers; None: every
    process on the first card this process sees). classifier: a v2 reader-best.pt to read with instead of the car's (classifier_onnx).
    pictures: also the brain's pictures (picture_path), read again where they are missing; a drive that has them gets
    them again whenever it is read. models: the release to read with (models_path; the brain's: sign_networks).
    speeds_of: route -> its joined logs (or None), for the car's speed the applicability network gets (frame_speeds);
    without it, or a drive without logs, the speed is unknown as on the car before its first carState."""
    package = package_path() if package is None else Path(package)
    models = Path(models) if models is not None else None
    onnx = classifier_onnx(root, classifier) if classifier is not None else None
    stamp = fingerprint(package, onnx, models)
    paths = network_paths(package, onnx, models)
    ids = network_ids(paths)
    ruled = 'decider' in paths
    records, tasks, stamps, known = {}, [], {}, {}
    for route in routes:
        videos = _videos(root, route)
        stamps[route] = _video_stamp(videos)
        record = load_signs(root, route, models)
        if record and record.get('fingerprint') == stamp and record.get('videos') == stamps[route] and \
                (not pictures or pictures_ready(root, record)):
            records[route] = record
            continue
        keep = pictures or bool(record and record.get('pictures'))
        speeds = frame_speeds(speeds_of(route)) if ruled and speeds_of is not None and videos else {}
        known[route] = bool(speeds)
        tasks += [(route, segment, str(path), str(picture_path(root, route, segment)) if keep else None,
                   speeds.get(segment)) for segment, path in videos.items()]
    if tasks:
        workers = workers or max(1, min(len(tasks), (os.cpu_count() or 4) // 2))
        which = f'the classifier {classifier}' if classifier is not None else f"the classifier {ids.get('classifier')}"
        log(f"road-type signs: reading {len(tasks)} segments of {len({t[0] for t in tasks})} drives with the SignNet "
            f"{ids.get('finder')} and {which}"
            + (f", the applicability network {ids.get('decider')} leaving out the signs of another road (the car's speed "
               f"from the logs of {sum(1 for route in known if known[route])} of them)" if ruled
               else ', no applicability network: every sign taken')
            + f', 4 frames a second, {workers} processes'
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
                                           context.Value('i', 0) if cards else None,
                                           str(models) if models is not None else None)) as pool:
            done = 0
            for (route, *_), result in zip(tasks, pool.map(read_segment, [(v, s, p, f) for _, s, v, p, f in tasks])):
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
                    records[route] = _save_signs(root, route, results.pop(route), stamp, stamps[route], classifier,
                                                 package, models, ids, known[route])
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
