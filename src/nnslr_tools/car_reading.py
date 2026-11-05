"""What the car's networks read on the frame on screen in Sign review (its button "Car networks", key T): SignNet's
boxes (the cutter) and the classifier's reading of each, so the reviewer sees what the car would make of that frame
and takes the boxes worth keeping.

On the car speedvisiond reads the road camera every 0.2 s with vision.read_signs: SignNet finds at most MAX_BOXES boxes
of MIN_BOX_SCORE or more in the top band of the frame, the classifier reads each, and the memory takes the readings of
MIN_CONFIDENCE or more. Here the same code does it: the car's vision.py and the networks of its release (models/
speed_limits.json of the sunnypilot clone, NNSLR_SUNNYPILOT_CLONE, else /root/sunnypilot; its settings put in place as
the car does, vision.apply_settings), on any frame of the drive (a picture at 1 a second, a frame of a read, a frame
between them), decoded from fcamera.hevc to NV12, the camera's pixel format (video_frames.nv12_frame).

Close to the car, not bit for bit: the car reads the camera's frame before the video encoder, with tinygrad on its
GPU; here it is the recorded video, with onnxruntime on one CPU thread. A confidence near MIN_CONFIDENCE can fall on
the other side. One frame is not a limit either: the memory takes a number read in CONFIRM_FRAMES frames within
CONFIRM_SECONDS.

Each box (frame pixels): SignNet's score, the class read, how sure, the speed it gives (read_signs), and what the car
does with it (use): 'speed' (a number for the memory), 'road' (a sign of the kind of road), 'board' (the green board
towards a motorway: only an input of the road-type network), 'nothing' (read sure enough, but nothing for the limit:
another sign, not a sign, unreadable), 'unsure' (below MIN_CONFIDENCE: the car drops it). And as a box of the review
(family, value_kph, sign_type) for "Use". Kept in memory per frame and release. Nothing is saved. Only the camera:
no map.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_signs, video_frames
from nnslr_tools.media import FULL_SIZE
from nnslr_tools.sign_types import class_hours, class_parts

MANIFEST = road_type_signs.RELEASE_MANIFEST
MANIFEST_FORMAT = 2
READINGS_KEPT = 256

_guard = threading.Lock()
_loaded: dict = {}
_readings: dict[tuple, dict] = {}


class CarReadingError(ValueError):
    """The car's networks cannot read here: no release of them in the sunnypilot clone, or one that does not load."""


def _stamp(paths) -> tuple:
    stamps = []
    for path in paths:
        stat = Path(path).stat()
        stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(stamps)


def _commit(path: Path) -> str | None:
    found = subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                           check=False)
    return (found.stdout.strip() or None) if found.returncode == 0 else None


def _release(package: Path) -> dict:
    """The networks of the car's release, as its runner reads them (speed_limits.json, format 2)."""
    models = package / 'models'
    try:
        manifest = json.loads((models / MANIFEST).read_text('utf-8'))
        if manifest.get('format') != MANIFEST_FORMAT:
            raise CarReadingError(f'the release in {models} is not of format {MANIFEST_FORMAT}')
        finder = models / manifest['finder']['file']
        classifier = models / manifest['classifier']['file']
        contracts = {name: json.loads((models / manifest[name]['contract']).read_text('utf-8'))
                     for name in ('finder', 'classifier')}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CarReadingError(f'no release of the car\'s networks in {models}: {exc}') from exc
    return {'version': str(manifest.get('version') or ''), 'finder': finder, 'classifier': classifier,
            'classes': list(contracts['classifier']['output']['classes']),
            'checkpoints': {name: contract.get('checkpoint') for name, contract in contracts.items()},
            'files': [package / 'vision.py', models / MANIFEST, finder, classifier]}


def networks(package: Path | None = None) -> dict:
    """The car's code and networks, loaded once and again when one of their files changes: {'vision', 'finder',
    'classifier', 'classes', 'stamp', 'release'}."""
    global _loaded
    package = Path(package) if package else road_type_signs.package_path()
    release = _release(package)
    try:
        stamp = _stamp(release['files'])
    except OSError as exc:
        raise CarReadingError(f'no release of the car\'s networks in {package}: {exc}') from exc
    with _guard:
        if _loaded.get('stamp') == stamp:
            return _loaded
        try:
            vision = road_type_signs.load_vision(package)
            finder = road_type_signs._Network(release['finder'])
            classifier = road_type_signs._Network(release['classifier'])
        except Exception as exc:  # noqa: BLE001 - whatever the clone's files raise, the car's networks are not here
            raise CarReadingError(f'the car\'s networks do not load: {exc}') from exc
        # A new dict, not the old one filled again: a reading under way keeps the networks it started with.
        _loaded = {'vision': vision, 'finder': finder, 'classifier': classifier, 'classes': release['classes'],
                   'stamp': stamp,
                   'release': {'name': f"speed-limits {release['version']}".strip(),
                               'cutter': release['checkpoints']['finder'],
                               'classifier': release['checkpoints']['classifier'],
                               'models_commit': _commit(package / 'models'), 'sunnypilot_commit': _commit(package),
                               'min_confidence': vision.MIN_CONFIDENCE, 'min_box_score': vision.MIN_BOX_SCORE,
                               'max_boxes': vision.MAX_BOXES}}
        _readings.clear()
        return _loaded


def use_of(vision, sign: dict) -> str:
    """What the car does with a sign of read_signs (see the module's text)."""
    if sign['confidence'] < vision.MIN_CONFIDENCE:
        return 'unsure'
    if sign['speed'] is not None:
        return 'speed'
    if sign['class'] in vision.ROAD_SIGNS:
        return 'road'
    if sign['class'] in vision.BOARD_SIGNS:
        return 'board'
    return 'nothing'


def review_box(name: str) -> dict:
    """A class of the classifier as a box of the review: its family, value, kind and panel ("other_sign:give_way" is an
    other sign of the kind give_way; "road_marking:30" a road marking of 30; "maximum_speed:unknown" a speed limit without
    its number; "maximum_speed_hours:30" a speed limit of 30 with a plate of hours, "maximum_speed_hours:30:22-6" says them)."""
    family, value, panel = class_parts(name)
    if family == 'road_marking':
        family = 'road_marking_candidate'
    box = {'family': family, 'value_kph': int(value) if value.isdigit() else None}
    if family == 'other_sign' and value:
        box['sign_type'] = value
    if panel is not None:
        box['supplementary_panel'] = panel
    if class_hours(name):
        box['plate_hours'] = class_hours(name)
    return box


def read_nv12(loaded: dict, data: bytes, width: int, height: int,
              full: tuple[bytes, int, int] | None = None) -> tuple[list[dict], int | None]:
    """read_signs on one NV12 frame, each sign with its use and its box of the review, and the index of the one the
    car's logs give as the frame's reading (vision.raw_reading, on the confidences before they are rounded: two signs
    of 0.99999 and 0.999999 are both 1.0 here, and the car takes the second). full: (NV12 bytes, width, height) of the
    same frame at full definition, for the classifier's crops, as the car with Read signs at full resolution."""
    vision = loaded['vision']
    pixels = np.frombuffer(data, dtype=np.uint8)
    y, uv = vision.nv12_planes(pixels, width, height, width, width * height)
    planes = None
    if full is not None:
        planes = vision.nv12_planes(np.frombuffer(full[0], dtype=np.uint8), full[1], full[2], full[1], full[1] * full[2])
    signs = vision.read_signs(y, uv, vision.SignNetInput(width, height), loaded['finder'], loaded['classifier'],
                              loaded['classes'], planes)
    surest = vision.raw_reading(signs)
    return ([{'box': [round(float(value), 2) for value in sign['box']], 'score': round(float(sign['score']), 4),
              'class': sign['class'], 'confidence': round(float(sign['confidence']), 4), 'speed': sign['speed'],
              'use': use_of(vision, sign), 'review': review_box(sign['class'])} for sign in signs],
            next((i for i, sign in enumerate(signs) if sign is surest), None))


def read_frame(root: Path, route: str, segment: int, decoded: int, *, width: int | None = None,
               height: int | None = None, package: Path | None = None) -> dict:
    """What the car's networks read on a decoded frame of a segment's road video: {'signs', 'surest' (the index of
    the reading the car's logs would show, vision.raw_reading; None without one), 'width', 'height', 'release', 'ms'}.
    width x height: the frame's size in the review, which the video must have (the boxes are in its pixels).
    Raises video_frames.VideoFrameError (no such frame), CarReadingError (no networks)."""
    loaded = networks(package)
    video = video_frames.video_of(root, route, segment)
    stat = video.stat()
    key = (str(video), stat.st_mtime_ns, stat.st_size, int(decoded), loaded['stamp'])
    with _guard:
        found = _readings.get(key)
    if found is None:
        started = time.monotonic()
        data, video_width, video_height = video_frames.nv12_frame(video, int(decoded))
        full = None
        if video_frames.video_size(video) == FULL_SIZE:
            full = video_frames.nv12_frame(video, int(decoded), native=True)
        signs, surest = read_nv12(loaded, data, video_width, video_height, full)
        found = {'signs': signs, 'surest': surest, 'width': video_width, 'height': video_height,
                 'release': loaded['release'], 'ms': round((time.monotonic() - started) * 1000)}
        with _guard:
            _readings[key] = found
            while len(_readings) > READINGS_KEPT:
                del _readings[next(iter(_readings))]
    if (width, height) != (None, None) and (width, height) != (found['width'], found['height']):
        raise video_frames.VideoFrameError(f"the road video is {found['width']}x{found['height']}, the frame "
                                           f'{width}x{height}: its boxes would not be on the picture')
    return found
