"""UserNet: a second small detector that finds the road users in the comma's road frame (person, two-wheeler, vehicle).

It is SignNet's network (build_signnet: strides 2, 4, 8, 16, one fuse convolution, CenterNet heads) with three heat
channels instead of one, a channel a kind (KINDS), and the same four shape channels shared by every kind. Its input is
a window of the frame, not SignNet's top band: the frame scaled to INPUT_WIDTH (672) pixels across and its rows
WINDOW_TOP to WINDOW_TOP + WINDOW_HEIGHT (72 to 328) of that picture, 672 x 256, where the road users of the road
ahead are (a 1344 x 760 frame: its rows 144 to 656). Grid 8, channels 16, 24, 32, 48, fuse 1 x 1: what runs on the
Adreno today for SignNet. decode gives each user back in the frame's pixels with its kind and score; at most
MAX_USERS a frame, of score MIN_USER_SCORE or more for what the brain will read (``user_boxes``, later); the measures
also go lower.

Two stages (scripts/run_usernet.sh, the panel's Train › UserNet):

- ``bdd``: BDD100K's 10000 pictures of 1280 x 720 (samples.json, a FiftyOne export; NNSLR_BDD_DIR). One window a
  picture, cut from the picture as it is read: no file is written. Its labels become ours (bdd_users): pedestrian is
  a person; a rider on a bicycle or motorcycle is one two-wheeler, the box around both; a bicycle or motorcycle alone
  is a two-wheeler; car, truck and bus are vehicles. A traffic sign or light is background (it is no road user, as on
  our frames); any other label (other person, other vehicle, train, trailer, a rider on nothing) is an ignore box:
  neither a user nor background. The pictures are split by picture name (bdd_split, frozen: a hash of the name),
  the split's sha256 kept with the weights.
- ``drives``: our reviewed frames (the dataset and split of the cutter). A road-user box of the review
  (sign_types.ROAD_USERS) is a person, a two-wheeler (bicycle, motorcycle) or a vehicle (car, truck, bus, tram);
  road_user_other is an ignore box. A frame ticked "road users complete" (users_complete) has every user from the
  size rule boxed: everything else there is background, so it teaches the negatives and is the only kind of frame
  validation and test take. On a frame not ticked only the boxed users are judged (the loss weight is 0 everywhere
  else, a missing box is not learnt as "nobody"); one without any user box is left out. The stage starts from a
  previous UserNet (the BDD stage, as a rule), or from random weights.

In both stages a box below MIN_BOX_HEIGHT pixels of the window is an ignore box, and so is a box whose centre is
outside the window but which reaches into it (half a car at the edge is not "nobody there"). Validation is our
ticked frames; the ``bdd`` stage falls back to BDD's own validation pictures only while there are none
(``validation_source``). The measures are in the frame's pixels and only judge what the review rule promises
(COMPLETE_HEIGHT: persons and two-wheelers from 24 px tall, vehicles from 40 px): a smaller user, found or missed,
counts as neither.

A network that learnt BDD's pictures says so (``uses_bdd``, in usernet-best.pt and result.json, kept along the starts
as SignNet keeps ``uses_mapillary``): their licence goes with the weights. UserNet is not on the car, not in a release
and not one of the cutter's checks (signnet.fits_the_car is untouched). Torch is imported only where a network runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

from nnslr_tools import onnx_device
from nnslr_tools.sign_types import is_road_user
from nnslr_tools.signnet import _torch

KINDS = ('person', 'two_wheeler', 'vehicle')
INPUT_WIDTH = 672
WINDOW_TOP = 72
WINDOW_HEIGHT = 256
STRIDE = 8
WIDTHS = (16, 24, 32, 48)
FUSE_KERNEL = 1
MAX_USERS = 24
MIN_USER_SCORE = 0.5
# Score thresholds of the measures (MIN_USER_SCORE among them: the epoch is chosen there).
SCORE_THRESHOLDS = (0.1, 0.3, 0.5)
# A box lower than this in the window (pixels of the 672-wide picture) is an ignore box: too small to learn.
MIN_BOX_HEIGHT = 12
# The review's size rule of "road users complete" (frame pixels, the box's height): what a ticked frame promises.
COMPLETE_HEIGHT = {'person': 24, 'two_wheeler': 24, 'vehicle': 40}
# A rider is on a bicycle or motorcycle when their boxes overlap by this share of the smaller one.
RIDER_OVERLAP = 0.3
SIGMA_SHARE = 1 / 6
BDD_DIR = Path('/data/sunnypilot/SPEEDLIMITS/bdd100k')
BDD_SIZE = (1280, 720)
BDD_KINDS = {'pedestrian': 'person', 'bicycle': 'two_wheeler', 'motorcycle': 'two_wheeler', 'car': 'vehicle',
             'truck': 'vehicle', 'bus': 'vehicle'}
BDD_BACKGROUND = ('traffic sign', 'traffic light')
BDD_TWO_WHEELERS = ('bicycle', 'motorcycle')
# The BDD split by picture: a hash of the name, so the same picture is always in the same partition.
BDD_SPLIT_SALT = 'usernet-bdd-split-v1'
BDD_SPLIT_SHARES = (('train', 80), ('validation', 10), ('test', 10))
DRIVE_KINDS = {'road_user_person': 'person', 'road_user_bicycle': 'two_wheeler',
               'road_user_motorcycle': 'two_wheeler', 'road_user_car': 'vehicle', 'road_user_truck': 'vehicle',
               'road_user_bus': 'vehicle', 'road_user_tram': 'vehicle'}
STAGES = ('bdd', 'drives')
# What a run of scripts/run_usernet.sh holds: the weights of the best epoch (with result.json beside them) and its log.
WEIGHTS = 'usernet-best.pt'
LOG = 'usernet.log'


def window_size() -> tuple[int, int]:
    """The network's input, width and height."""
    return INPUT_WIDTH, WINDOW_HEIGHT


def grid() -> tuple[int, int]:
    """The heat map's width and height (cells of STRIDE pixels)."""
    return INPUT_WIDTH // STRIDE, WINDOW_HEIGHT // STRIDE


# --- labels -----------------------------------------------------------------------------------------------------


def _overlap(a: list[float], b: list[float]) -> float:
    """The intersection of two boxes over the area of the smaller one."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 0.0


def bdd_users(detections: list[dict[str, Any]], width: float, height: float) -> tuple[list[dict], list[list[float]]]:
    """A BDD picture's labels as ours, in its pixels: the users ({kind, box}) and the ignore boxes.

    bounding_box is [x, y, w, h] in fractions of the picture. Each rider goes with the bicycle or motorcycle it
    overlaps most (RIDER_OVERLAP of the smaller box at least; one rider a two-wheeler, the best pairs first): the
    two-wheeler is then the box around both. A rider on nothing is an ignore box, and so is every label that is
    neither a user nor a traffic sign or light (those are background)."""
    boxes = []
    for detection in detections or []:
        label = detection.get('label')
        box = detection.get('bounding_box')
        if not isinstance(label, str) or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x, y, w, h = (float(value) for value in box)
        boxes.append((label, [x * width, y * height, (x + w) * width, (y + h) * height]))
    riders = [index for index, (label, _) in enumerate(boxes) if label == 'rider']
    wheels = [index for index, (label, _) in enumerate(boxes) if label in BDD_TWO_WHEELERS]
    pairs = sorted(((_overlap(boxes[r][1], boxes[w][1]), r, w) for r in riders for w in wheels), reverse=True)
    rider_of: dict[int, int] = {}
    taken: set[int] = set()
    for overlap, rider, wheel in pairs:
        if overlap < RIDER_OVERLAP:
            break
        if rider in taken or wheel in rider_of:
            continue
        rider_of[wheel] = rider
        taken.add(rider)
    users, ignore = [], []
    for index, (label, box) in enumerate(boxes):
        if label in BDD_TWO_WHEELERS:
            if index in rider_of:
                other = boxes[rider_of[index]][1]
                box = [min(box[0], other[0]), min(box[1], other[1]), max(box[2], other[2]), max(box[3], other[3])]
            users.append({'kind': 'two_wheeler', 'box': box})
        elif label in BDD_KINDS:
            users.append({'kind': BDD_KINDS[label], 'box': box})
        elif label == 'rider':
            if index not in taken:
                ignore.append(box)
        elif label not in BDD_BACKGROUND:
            ignore.append(box)
    return users, ignore


def bdd_split(name: str) -> str:
    """The partition of a BDD picture, by its name (frozen: the same name, the same partition)."""
    value = int(hashlib.sha256(f'{BDD_SPLIT_SALT}:{name}'.encode('utf-8')).hexdigest()[:8], 16) % 100
    edge = 0
    for split, share in BDD_SPLIT_SHARES:
        edge += share
        if value < edge:
            return split
    return BDD_SPLIT_SHARES[-1][0]


def bdd_frames(bdd_dir: Path = BDD_DIR) -> dict[str, Any]:
    """BDD's pictures as frames, from samples.json alone (no picture is read): {frames, samples, samples_sha256,
    split_sha256, unlabelled}. A frame: source 'bdd', its picture's absolute path, size, partition, users and
    ignore boxes; every BDD picture has all its users boxed (complete). A sample without detections is left out."""
    from nnslr_tools.exporting import sha256_file
    bdd_dir = Path(bdd_dir)
    samples_path = bdd_dir / 'samples.json'
    if not samples_path.is_file():
        raise ValueError(f'usernet_bdd_missing: {samples_path} is not there (NNSLR_BDD_DIR: the BDD100K FiftyOne '
                         'export with samples.json and data/)')
    try:
        samples = json.loads(samples_path.read_text(encoding='utf-8')).get('samples')
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f'usernet_bdd_unreadable: {samples_path}') from exc
    if not isinstance(samples, list):
        raise ValueError(f'usernet_bdd_unreadable: {samples_path} has no list "samples"')
    frames, unlabelled, assignments = [], 0, {}
    for sample in samples:
        filepath = sample.get('filepath') if isinstance(sample, dict) else None
        detections = (sample.get('detections') or {}) if isinstance(sample, dict) else {}
        if not isinstance(filepath, str) or not isinstance(detections, dict) or 'detections' not in detections:
            unlabelled += 1
            continue
        metadata = sample.get('metadata') or {}
        width = int(metadata.get('width') or BDD_SIZE[0])
        height = int(metadata.get('height') or BDD_SIZE[1])
        users, ignore = bdd_users(detections.get('detections') or [], width, height)
        name = Path(filepath).stem
        split = bdd_split(name)
        assignments[name] = split
        frames.append({'source': 'bdd', 'id': name, 'image_path': str((bdd_dir / filepath).resolve()),
                       'width': width, 'height': height, 'split': split, 'users': users, 'ignore': ignore,
                       'complete': True})
    split_sha256 = hashlib.sha256(json.dumps(assignments, sort_keys=True).encode('utf-8')).hexdigest()
    return {'frames': frames, 'samples': str(samples_path.resolve()), 'samples_sha256': sha256_file(samples_path),
            'split_sha256': split_sha256, 'unlabelled': unlabelled}


def ticked(frame: dict[str, Any]) -> bool:
    """Whether a reviewed frame (training.reviewed_frames) was ticked "road users complete": on the frame, or on any
    of its kept rows (every row of a ticked frame carries it)."""
    return frame.get('users_complete') is True or any(
        row.get('users_complete') is True for row in frame.get('kept') or [])


def drive_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """One reviewed frame (training.reviewed_frames) as UserNet's: its road-user boxes as users, road_user_other as
    ignore boxes, whether it is ticked (complete). Signs and every other box are background."""
    users, ignore = [], []
    for row in frame.get('kept') or []:
        box = row.get('bbox_xyxy')
        if not is_road_user(row) or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        box = [float(value) for value in box]
        kind = DRIVE_KINDS.get(row.get('sign_type'))
        if kind is None:
            ignore.append(box)
        else:
            users.append({'kind': kind, 'box': box})
    return {'source': 'drive', 'image_path': frame['image_path'], 'image_sha256': frame.get('image_sha256'),
            'width': frame['width'], 'height': frame['height'], 'split': frame['split'],
            'route_id': frame.get('route_id'), 'label_basis': frame.get('label_basis'), 'users': users,
            'ignore': ignore, 'complete': ticked(frame)}


def drive_frames(dataset_path: Path, split_path: Path, data_root: Path, *, dataset_kind: str = 'gold') -> dict[str, Any]:
    """Our reviewed frames as UserNet's (drive_frame): {frames, dataset_sha256, split_sha256}, the dataset and split
    checked as every plan's (training.load_reviewed_dataset)."""
    from nnslr_tools.training import load_reviewed_dataset, reviewed_frames
    loaded = load_reviewed_dataset(Path(dataset_path), Path(split_path), Path(data_root), dataset_kind=dataset_kind)
    frames = [drive_frame(frame) for frame in reviewed_frames(loaded['rows'], loaded['assignments'])]
    return {'frames': frames, 'dataset_sha256': loaded['dataset_sha256'], 'split_sha256': loaded['split_sha256']}


# --- the window -------------------------------------------------------------------------------------------------


def window_boxes(users: list[dict], ignore: list[list[float]], frame_width: float, *,
                 flip: bool = False) -> tuple[list[tuple[int, list[float]]], list[list[float]]]:
    """A frame's users and ignore boxes in the window's pixels: the targets [(kind index, box)] and the ignore boxes.

    The frame is scaled to INPUT_WIDTH across, then WINDOW_TOP is taken off every row. A user whose centre is in the
    window and which is MIN_BOX_HEIGHT tall there or more is a target, cut to the window; one that only reaches into
    it, or is smaller, is an ignore box; one outside is left out. ``flip``: the window mirrored."""
    scale = INPUT_WIDTH / float(frame_width)

    def placed(box):
        x1, y1, x2, y2 = (float(value) * scale for value in box)
        return [x1, y1 - WINDOW_TOP, x2, y2 - WINDOW_TOP]

    def inside(box):
        return box[3] > 0 and box[1] < WINDOW_HEIGHT and box[2] > 0 and box[0] < INPUT_WIDTH

    def cut(box):
        return [max(box[0], 0.0), max(box[1], 0.0), min(box[2], float(INPUT_WIDTH)), min(box[3], float(WINDOW_HEIGHT))]

    targets, ignored = [], []
    for user in users:
        box = placed(user['box'])
        if not inside(box):
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        shown = cut(box)
        if 0 <= cx < INPUT_WIDTH and 0 <= cy < WINDOW_HEIGHT and shown[3] - shown[1] >= MIN_BOX_HEIGHT:
            targets.append((KINDS.index(user['kind']), shown))
        else:
            ignored.append(shown)
    for box in ignore:
        box = placed(box)
        if inside(box):
            ignored.append(cut(box))
    if flip:
        targets = [(kind, [INPUT_WIDTH - x2, y1, INPUT_WIDTH - x1, y2]) for kind, (x1, y1, x2, y2) in targets]
        ignored = [[INPUT_WIDTH - x2, y1, INPUT_WIDTH - x1, y2] for x1, y1, x2, y2 in ignored]
    return targets, ignored


def to_frame(box: list[float], frame_width: float) -> list[float]:
    """A box in the window's pixels back in the frame's: the row offset added back, then the scale undone."""
    scale = INPUT_WIDTH / float(frame_width)
    return [box[0] / scale, (box[1] + WINDOW_TOP) / scale, box[2] / scale, (box[3] + WINDOW_TOP) / scale]


def encode(targets: list[tuple[int, list[float]]], ignore: list[list[float]] = (), *, complete: bool = True):
    """Targets of one window: heat (3, H, W), a channel a kind; shape (4, H, W); mask (H, W) of the cells that hold a
    centre; weight (H, W). ``complete`` (BDD, a ticked frame): weight 1 everywhere but on the ignore boxes; else
    (a frame not ticked) weight 0 everywhere but on the boxed users, so only they are judged. Each box's cells with
    one more around them."""
    torch, _ = _torch()
    width, height = grid()
    heat = torch.zeros(len(KINDS), height, width)
    shape = torch.zeros(4, height, width)
    mask = torch.zeros(height, width)
    weight = torch.ones(height, width) if complete else torch.zeros(height, width)
    ys = torch.arange(height).view(-1, 1).float()
    xs = torch.arange(width).view(1, -1).float()

    def cells(box):
        x1, y1, x2, y2 = box
        return (slice(max(int(y1 / STRIDE) - 1, 0), min(int(y2 / STRIDE) + 1, height - 1) + 1),
                slice(max(int(x1 / STRIDE) - 1, 0), min(int(x2 / STRIDE) + 1, width - 1) + 1))

    for kind, (x1, y1, x2, y2) in targets:
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        ix, iy = min(int(cx), width - 1), min(int(cy), height - 1)
        sigma = max(max(w, h) / STRIDE * SIGMA_SHARE, 0.5)
        gauss = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma * sigma))
        heat[kind] = torch.maximum(heat[kind], gauss)
        shape[:, iy, ix] = torch.tensor([math.log(w), math.log(h), cx - ix, cy - iy])
        mask[iy, ix] = 1.0
        if not complete:
            weight[cells((x1, y1, x2, y2))] = 1.0
    for box in ignore:
        weight[cells(box)] = 0.0
    return heat, shape, mask, weight


def decode(heat, shape, frame_width, *, limit: int = MAX_USERS, min_score: float = MIN_USER_SCORE) -> list[list[dict]]:
    """The users of a batch in the frame's pixels: per kind channel, the local maxima of its heat (3 x 3) of
    ``min_score`` or more, the box from the shared shape channels, the row offset added back and the scale undone
    (``frame_width``: one number, or one a frame of the batch). Each frame's best ``limit`` over every kind, best
    first: {kind, box, score}."""
    _, nn = _torch()
    peaks = heat * (nn.functional.max_pool2d(heat, 3, 1, 1) == heat).float()
    batch, kinds, height, width = heat.shape
    widths = list(frame_width) if isinstance(frame_width, (list, tuple)) else [frame_width] * batch
    found = []
    for b in range(batch):
        frame = []
        for kind in range(kinds):
            scores, indexes = peaks[b, kind].reshape(-1).topk(min(limit, height * width))
            for score, index in zip(scores.tolist(), indexes.tolist()):
                if score < min_score:
                    break
                iy, ix = divmod(index, width)
                log_w, log_h, ox, oy = shape[b, :, iy, ix].tolist()
                w, h = math.exp(min(log_w, 8.0)), math.exp(min(log_h, 8.0))
                cx, cy = (ix + ox) * STRIDE, (iy + oy) * STRIDE
                box = to_frame([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], widths[b])
                frame.append({'kind': KINDS[kind], 'box': box, 'score': score})
        frame.sort(key=lambda found_user: -found_user['score'])
        found.append(frame[:limit])
    return found


def decode_output(out, frame_width, **kw) -> list[list[dict]]:
    """``decode`` on the one tensor the exported ONNX gives: heat in channels 0 to 2, the shape in 3 to 6."""
    kinds = len(KINDS)
    return decode(out[:, :kinds], out[:, kinds:], frame_width, **kw)


def window_of(image):
    """The window of a decoded frame (a PIL image) as a uint8 tensor (3, WINDOW_HEIGHT, INPUT_WIDTH): the frame
    scaled to INPUT_WIDTH across (bilinear), its rows WINDOW_TOP to WINDOW_TOP + WINDOW_HEIGHT (black below a
    short frame). No normalisation: the network takes it / 255."""
    torch, _ = _torch()
    import numpy as np
    from PIL import Image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    width, height = image.size
    if width != INPUT_WIDTH:
        image = image.resize((INPUT_WIDTH, round(height * INPUT_WIDTH / width)), Image.BILINEAR)
    array = np.asarray(image.crop((0, WINDOW_TOP, INPUT_WIDTH, WINDOW_TOP + WINDOW_HEIGHT)), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _window_image(frame: dict[str, Any], data_root: Path):
    from PIL import Image
    with Image.open(Path(data_root) / frame['image_path']) as image:
        return window_of(image)


# --- the plan ---------------------------------------------------------------------------------------------------


def measured_truth(frame: dict[str, Any]) -> tuple[dict[str, list[list[float]]], list[list[float]]]:
    """What a measure judges on a frame, in the frame's pixels: the users of each kind it must find (targets of the
    window of COMPLETE_HEIGHT or more) and the boxes where a found user counts as neither found nor false (the
    ignore boxes of the window and the smaller users)."""
    targets, ignored = window_boxes(frame['users'], frame['ignore'], frame['width'])
    truth: dict[str, list[list[float]]] = {kind: [] for kind in KINDS}
    dont_care = [to_frame(box, frame['width']) for box in ignored]
    for kind, box in targets:
        box = to_frame(box, frame['width'])
        if box[3] - box[1] >= COMPLETE_HEIGHT[KINDS[kind]]:
            truth[KINDS[kind]].append(box)
        else:
            dont_care.append(box)
    return truth, dont_care


def _counts(frames: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = {kind: 0 for kind in KINDS}
    measured = 0
    for frame in frames:
        for user in frame['users']:
            by_kind[user['kind']] += 1
        if frame['complete']:
            measured += sum(len(boxes) for boxes in measured_truth(frame)[0].values())
    return {'frames': len(frames), 'users_by_kind': by_kind, 'ignore_boxes': sum(len(f['ignore']) for f in frames),
            'complete': sum(1 for f in frames if f['complete']), 'measured_users': measured}


def prepare_usernet(stage: str, data_root: Path, *, bdd_dir: Path = BDD_DIR, dataset_path: Path | None = None,
                    split_path: Path | None = None, dataset_kind: str = 'gold') -> dict[str, Any]:
    """The plan of a UserNet training, no torch: its frames (train, validation, test) and what they are.

    ``bdd``: train is BDD's train pictures. Validation and test are our ticked frames of those partitions when there
    are ticked validation frames; else BDD's own validation and test pictures (``validation_source`` says which).
    Our dataset unreadable (none yet) only means BDD's. ``drives``: train is our train frames, ticked or with a user
    box; validation and test are our ticked frames only, and there must be some with users to judge.
    ``trainable`` false says why in ``not_trainable``."""
    if stage not in STAGES:
        raise ValueError(f'usernet_stage: {stage!r}, not one of {", ".join(STAGES)}')
    data_root = Path(data_root)
    dataset_path = Path(dataset_path or data_root / 'annotations' / 'objects.jsonl')
    split_path = Path(split_path or data_root / 'splits' / 'latest.json')
    bdd = drives = None
    drives_reason = None
    if stage == 'bdd':
        bdd = bdd_frames(bdd_dir)
        try:
            drives = drive_frames(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
        except (OSError, ValueError) as exc:
            drives_reason = str(exc) or type(exc).__name__
    else:
        drives = drive_frames(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
    ours = drives['frames'] if drives else []
    ticked_of = {split: [f for f in ours if f['split'] == split and f['complete']] for split in ('validation', 'test')}
    if stage == 'bdd':
        train = [f for f in bdd['frames'] if f['split'] == 'train']
        source = 'drives' if ticked_of['validation'] else 'bdd'
        if source == 'drives':
            validation, test = ticked_of['validation'], ticked_of['test']
        else:
            validation = [f for f in bdd['frames'] if f['split'] == 'validation']
            test = [f for f in bdd['frames'] if f['split'] == 'test']
    else:
        source = 'drives'
        train = [f for f in ours if f['split'] == 'train' and (f['complete'] or f['users'])]
        validation, test = ticked_of['validation'], ticked_of['test']
    frames = ([{**f, 'split': 'train'} for f in train] + [{**f, 'split': 'validation'} for f in validation]
              + [{**f, 'split': 'test'} for f in test])
    counts = {'train': _counts(train), 'validation': _counts(validation), 'test': _counts(test)}
    reasons = []
    if not any(f['users'] for f in train):
        reasons.append('no train frame with a road user box' + ('' if stage == 'bdd' else
                       ' (box the road users in Sign review, and tick "road users complete" where all are boxed)'))
    if not validation:
        reasons.append('no validation frame ticked "road users complete" (Sign review, key J): validation is our '
                       'ticked frames only')
    elif not counts['validation']['measured_users']:
        reasons.append('the validation frames have no road user to find (persons and two-wheelers from 24 px tall, '
                       'vehicles from 40 px, inside the window)')
    plan = {
        'task': 'usernet', 'stage': stage, 'uses_bdd': stage == 'bdd', 'validation_source': source,
        'frames': frames, 'counts_by_split': counts,
        'dataset_sha256': drives['dataset_sha256'] if drives else None,
        'split_sha256': drives['split_sha256'] if drives else None,
        'bdd': ({key: bdd[key] for key in ('samples', 'samples_sha256', 'split_sha256', 'unlabelled')}
                | {'pictures': len(bdd['frames'])}) if bdd else None,
        'drives': ({'frames': len(ours), 'ticked': sum(1 for f in ours if f['complete']),
                    'with_users': sum(1 for f in ours if f['users'])} if drives else {'not_read': drives_reason}),
        'trainable': not reasons, 'not_trainable': '; '.join(reasons) or None,
    }
    return plan


def uses_bdd(plan: dict[str, Any], start: dict[str, Any] | None) -> bool:
    """Whether a network learnt BDD's pictures: its own plan did (the ``bdd`` stage), or its start did, anywhere in
    that start's chain. usernet-best.pt and result.json keep it, so the licence goes with the weights."""
    return bool(plan.get('uses_bdd') or plan.get('stage') == 'bdd' or (start or {}).get('uses_bdd'))


# --- the network ------------------------------------------------------------------------------------------------


def build_usernet():
    """UserNet's network: SignNet's with a heat channel a kind (KINDS), UserNet's widths, grid and fuse."""
    from nnslr_tools.signnet import build_signnet
    return build_signnet(WIDTHS, STRIDE, FUSE_KERNEL, heads=len(KINDS))


def _settings() -> dict[str, Any]:
    return {'widths': list(WIDTHS), 'stride': STRIDE, 'fuse_kernel': FUSE_KERNEL, 'heads': len(KINDS),
            'kinds': list(KINDS), 'input': list(window_size()), 'window_top': WINDOW_TOP}


def checkpoint_of(model, plan: dict[str, Any], *, epoch: int, start: dict[str, Any] | None = None) -> dict[str, Any]:
    """What usernet-best.pt holds: the weights, the network's settings, the stage, the data it learnt from and
    ``uses_bdd`` (its own stage or its start's)."""
    return {'task': 'usernet', 'state_dict': model.state_dict(), **_settings(), 'epoch': epoch,
            'stage': plan.get('stage'), 'dataset_sha256': plan.get('dataset_sha256'),
            'split_sha256': plan.get('split_sha256'), 'bdd_split_sha256': (plan.get('bdd') or {}).get('split_sha256'),
            'validation_source': plan.get('validation_source'), 'start': start, 'uses_bdd': uses_bdd(plan, start)}


def model_of(checkpoint: dict[str, Any]):
    """The network of a loaded usernet-best.pt, in eval mode on the CPU."""
    if not isinstance(checkpoint, dict) or checkpoint.get('task') != 'usernet':
        raise ValueError('checkpoint_task_mismatch: expected usernet')
    if not checkpoint.get('state_dict') or not isinstance(checkpoint['state_dict'], dict):
        raise ValueError('checkpoint_missing_state_dict')
    from nnslr_tools.signnet import build_signnet
    model = build_signnet(tuple(checkpoint['widths']), checkpoint['stride'], checkpoint['fuse_kernel'],
                          heads=int(checkpoint['heads']))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()


def load_usernet(checkpoint_path: Path):
    """The network of a usernet-best.pt in eval mode, and the checkpoint."""
    torch, _ = _torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    return model_of(checkpoint), checkpoint


def usernet_start(path: Path) -> tuple[dict, dict]:
    """The weights of a finished UserNet to go on from (--init-from; the drives stage from the BDD one) and what
    result.json records of it: its file and sha256, run, epoch, stage, data and ``uses_bdd``. Its settings must be
    this network's."""
    torch, _ = _torch()
    from nnslr_tools.exporting import sha256_file
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or saved.get('task') != 'usernet' or not isinstance(saved.get('state_dict'), dict):
        raise ValueError(f'the start is not a UserNet: {path}')
    different = [f'{key} {saved.get(key)} (this run {value})' for key, value in _settings().items()
                 if saved.get(key) != value]
    if different:
        raise ValueError(f'the start is another network: {", ".join(different)}: {path}')
    return saved['state_dict'], {'path': str(path), 'sha256': sha256_file(path), 'run': path.parent.name,
                                 'epoch': saved.get('epoch'), 'stage': saved.get('stage'),
                                 'split_sha256': saved.get('split_sha256'),
                                 'dataset_sha256': saved.get('dataset_sha256'),
                                 'bdd_split_sha256': saved.get('bdd_split_sha256'),
                                 'uses_bdd': bool(saved.get('uses_bdd') or saved.get('stage') == 'bdd')}


# --- training and measures --------------------------------------------------------------------------------------


def _windows(frames: list[dict], data_root: Path, *, train: bool):
    """The windows of ``frames``: training items (window uint8 mirrored half of the times, its targets and its light:
    gain and bias) or measure items (window, index)."""
    torch, _ = _torch()
    from torch.utils.data import Dataset

    class Windows(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            frame = frames[index]
            image = _window_image(frame, data_root)
            if not train:
                return image.contiguous(), index
            # The worker's own generator: torch seeds Python's random in each worker from the loader's.
            flip = random.random() < 0.5
            if flip:
                image = image.flip(-1)
            targets, ignored = window_boxes(frame['users'], frame['ignore'], frame['width'], flip=flip)
            heat, shape, mask, weight = encode(targets, ignored, complete=frame['complete'])
            light = torch.tensor([random.uniform(0.7, 1.3), random.uniform(-0.08, 0.08)], dtype=torch.float32)
            return image.contiguous(), heat, shape, mask, weight, light

    return Windows()


def _scored(frames: list[dict], predictions: dict[int, list[dict]], threshold: float) -> dict[str, Any]:
    from nnslr_tools.evaluation import score_frame
    from nnslr_tools.signnet import _in_ignore
    rows = {kind: {'truth': 0, 'found': 0, 'false': 0} for kind in KINDS}
    for index, frame in enumerate(frames):
        truth, dont_care = measured_truth(frame)
        found = [p for p in predictions.get(index, []) if p['score'] >= threshold]
        found = [p for p in found if p['box'][3] - p['box'][1] >= COMPLETE_HEIGHT[p['kind']]
                 and not (dont_care and _in_ignore(p['box'], dont_care))]
        for kind in KINDS:
            item = score_frame(truth[kind], [p for p in found if p['kind'] == kind], score_threshold=0.0,
                               iou_threshold=0.5)
            rows[kind]['truth'] += len(truth[kind])
            rows[kind]['found'] += item['true_positives']
            rows[kind]['false'] += item['false_positives']

    def rates(row):
        return {**row, 'recall': row['found'] / row['truth'] if row['truth'] else None,
                'precision': row['found'] / (row['found'] + row['false']) if row['found'] + row['false'] else None}

    total = {key: sum(rows[kind][key] for kind in KINDS) for key in ('truth', 'found', 'false')}
    return {**rates(total), 'false_per_frame': total['false'] / len(frames) if frames else None,
            'by_kind': {kind: rates(rows[kind]) for kind in KINDS}}


def measure(model, frames: list[dict], data_root: Path, device, *, batch: int = 16, workers: int = 4) -> dict[str, Any]:
    """UserNet's measures on ``frames`` (complete ones: BDD's, our ticked), per score threshold: per kind and over
    all, the users found (IoU 0.5 with a user of the same kind), missed and the false ones, in the frame's pixels,
    at most MAX_USERS a frame. A user below COMPLETE_HEIGHT, found or not, and a box found on an ignore box count
    as neither (measured_truth). ``at_min_score``: at MIN_USER_SCORE, what the brain will read."""
    torch, _ = _torch()
    from torch.utils.data import DataLoader
    from nnslr_tools.signnet import on_device
    loader = DataLoader(_windows(frames, data_root, train=False), batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=torch.device(device).type == 'cuda')
    predictions: dict[int, list[dict]] = {}
    model.eval()
    with torch.no_grad():
        for image, indexes in loader:
            light = torch.tensor([[1.0, 0.0]]).expand(len(image), 2)
            heat, shape = model(on_device(image, light, device))
            heat, shape = heat.float().cpu(), shape.float().cpu()
            indexes = indexes.tolist()
            found = decode(heat, shape, [frames[i]['width'] for i in indexes], limit=MAX_USERS,
                           min_score=min(SCORE_THRESHOLDS))
            for index, users in zip(indexes, found):
                predictions[index] = users
    by_threshold = {str(t): _scored(frames, predictions, t) for t in SCORE_THRESHOLDS}
    return {'frames': len(frames), 'pixels': 'frame', 'by_score_threshold': by_threshold,
            'at_min_score': by_threshold[str(MIN_USER_SCORE)]}


def train_usernet(plan: dict[str, Any], data_root: Path, run_dir: Path, *, epochs: int = 30, batch: int = 32,
                  learning_rate: float = 3e-3, workers: int = 8, seed: int = 1, init_from: Path | None = None,
                  log=None, device: str = 'cuda') -> dict[str, Any]:
    """Train UserNet on the plan's train frames, keep the epoch with the best validation recall at MIN_USER_SCORE,
    then measure it on validation and test. Writes usernet-best.pt and result.json in run_dir. Every epoch logs one
    line (the panel reads it): ``usernet epoch E/N train_loss L val_recall R val_false_per_frame F Ss``. Then
    "usernet measure the best epoch N on validation and test" and, last, "UserNet complete: <dir>".

    The windows are cut from the pictures as they are read, on ``workers`` processes (mirrored half of the times,
    a light of their own); nothing is written but the two files. ``device``: the card, never the CPU on its own;
    without one it stops before writing anything."""
    if not plan.get('trainable'):
        raise ValueError(f"usernet_not_trainable: {plan.get('not_trainable')}")
    by_split: dict[str, list[dict]] = {}
    for frame in plan['frames']:
        by_split.setdefault(frame['split'], []).append(frame)
    train_frames, validation, test = by_split.get('train', []), by_split.get('validation', []), by_split.get('test', [])
    if len(train_frames) < batch:
        raise ValueError(f'usernet_not_trainable: {len(train_frames)} train frames, fewer than one batch of {batch}')
    if log is None:
        def log(line):
            print(line, flush=True)
    torch, _ = _torch()
    from torch.utils.data import DataLoader
    from nnslr_tools.signnet import loss, on_a_card, on_device
    device = on_a_card(device)
    random.seed(seed)
    torch.manual_seed(seed)
    data_root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_usernet()
    start = None
    if init_from is not None:
        weights, start = usernet_start(init_from)
        model.load_state_dict(weights, strict=True)
        log(f"usernet starts from {start['run']} (its {start['stage']} stage, epoch {start['epoch']})")
    model = model.to(device)
    loader = DataLoader(_windows(train_frames, data_root, train=True), batch_size=batch, shuffle=True,
                        drop_last=True, num_workers=workers, pin_memory=device.type == 'cuda',
                        persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps = len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=epochs * steps)
    print(f"usernet: {len(train_frames)} train windows, {len(validation)} validation frames "
          f"({plan.get('validation_source')}), {len(test)} test frames, {steps} batches an epoch",
          file=sys.stderr, flush=True)
    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, parts = time.monotonic(), 0.0, {'focal': 0.0, 'shape': 0.0}
        for image, heat, shape, mask, weight, light in loader:
            pred_heat, pred_shape = model(on_device(image, light, device))
            value, detail = loss(pred_heat, pred_shape, heat.to(device), shape.to(device), mask.to(device),
                                 weight.to(device))
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            scheduler.step()
            total += value.item()
            for key in parts:
                parts[key] += detail[key]
        checked = measure(model, validation, data_root, device, workers=workers)['at_min_score']
        recall = checked['recall'] or 0.0
        false = checked['false_per_frame'] or 0.0
        history.append({'epoch': epoch, 'loss': total / steps, 'focal': parts['focal'] / steps,
                        'shape': parts['shape'] / steps, 'validation_recall': recall,
                        'validation_false_per_frame': false, 'seconds': round(time.monotonic() - started, 1)})
        log(f"usernet epoch {epoch}/{epochs} train_loss {total / steps:.4f} val_recall {recall:.3f} "
            f"val_false_per_frame {false:.2f} {history[-1]['seconds']:.0f}s")
        if recall > best:
            best, best_epoch = recall, epoch
            torch.save(checkpoint_of(model, plan, epoch=epoch, start=start), run_dir / WEIGHTS)
    log(f"usernet measure the best epoch {best_epoch} on validation and test")
    checkpoint = torch.load(run_dir / WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    result = {
        'task': 'usernet', **_settings(), 'stage': plan.get('stage'),
        'parameters': sum(p.numel() for p in model.parameters()), 'epochs': epochs, 'best_epoch': best_epoch,
        'best_epoch_by': f'validation_recall_at_{MIN_USER_SCORE}', 'max_users': MAX_USERS,
        'min_user_score': MIN_USER_SCORE, 'complete_height': COMPLETE_HEIGHT,
        'dataset_sha256': plan.get('dataset_sha256'), 'split_sha256': plan.get('split_sha256'),
        'bdd': plan.get('bdd'), 'drives': plan.get('drives'), 'validation_source': plan.get('validation_source'),
        'counts_by_split': plan.get('counts_by_split'), 'start': start, 'uses_bdd': uses_bdd(plan, start),
        'history': history,
        'validation': measure(model, validation, data_root, device, workers=workers),
        'test': measure(model, test, data_root, device, workers=workers) if test else None,
    }
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    log(f"UserNet complete: {run_dir}")
    return result


# --- export -----------------------------------------------------------------------------------------------------


def export_usernet_onnx(checkpoint_path: Path, output_path: Path, *, verify: bool = True) -> dict[str, Any]:
    """UserNet as ONNX. Input ``images`` (1, 3, 256, 672), float 0..1, RGB: the window. One output ``out``
    (1, 7, 32, 84): the heat of each kind (sigmoid), log width, log height, offset x, offset y. Checked against
    PyTorch with onnxruntime (onnx_device.check_providers: the card); writes ``<output>.contract.json`` beside it."""
    torch, nn = _torch()
    from nnslr_tools.exporting import _export_onnx, sha256_file
    model, checkpoint = load_usernet(checkpoint_path)
    width, height = checkpoint['input']
    stride = checkpoint['stride']
    kinds = list(checkpoint['kinds'])

    class OneOutput(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            heat, shape = self.net(images)
            return torch.cat([heat, shape], 1)

    wrapped = OneOutput(model).eval()
    example = torch.rand(1, 3, height, width)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _export_onnx(wrapped, example, output_path, input_names=['images'], output_names=['out'])
    try:
        import onnx
    except ImportError as exc:
        raise ValueError('onnx_dependencies_missing: install onnx') from exc
    onnx.checker.check_model(onnx.load(str(output_path)))
    verification: dict[str, Any] = {'enabled': verify, 'passed': None}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError('onnxruntime_missing') from exc
        import numpy as np
        with torch.no_grad():
            expected = wrapped(example).numpy()
        session = ort.InferenceSession(str(output_path), providers=onnx_device.check_providers(ort))
        actual = session.run(['out'], {'images': example.numpy()})[0]
        max_abs_error = float(np.max(np.abs(expected - actual)))
        passed = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5))
        verification = {'enabled': True, 'passed': passed, 'max_abs_error': max_abs_error, 'rtol': 1e-4, 'atol': 1e-5,
                        'provider': session.get_providers()[0]}
        if not passed:
            raise ValueError(f'onnx_parity_failed: max_abs_error={max_abs_error}')
    contract = {
        'task': 'usernet', 'widths': list(checkpoint['widths']), 'stride': stride,
        'fuse_kernel': checkpoint['fuse_kernel'], 'kinds': kinds, 'epoch': checkpoint.get('epoch'),
        'uses_bdd': bool(checkpoint.get('uses_bdd')),
        'input': {'name': 'images', 'shape': [1, 3, height, width], 'dtype': 'float32', 'range': [0.0, 1.0],
                  'layout': f'RGB; the road frame scaled to {width} pixels across, its rows {checkpoint["window_top"]} '
                            f'to {checkpoint["window_top"] + height}'},
        'output': {'name': 'out', 'shape': [1, len(kinds) + 4, height // stride, width // stride],
                   'channels': [f'heat_{kind}' for kind in kinds] + ['log_width', 'log_height', 'offset_x',
                                                                      'offset_y'],
                   'decode': f'a user of a kind at each 3x3 local maximum of its heat channel of {MIN_USER_SCORE} '
                             f'or more, at most {MAX_USERS} a frame; its centre ((x + offset_x) * stride, '
                             '(y + offset_y) * stride) and size (exp(log_width), exp(log_height)) in window pixels; '
                             f'in the frame: add {checkpoint["window_top"]} to the rows, then divide by '
                             f'{width} / frame width'},
        'onnx_sha256': sha256_file(output_path), 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(Path(checkpoint_path)), 'verification': verification,
    }
    contract_path = output_path.with_suffix(output_path.suffix + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'valid': True, 'task': 'usernet', 'onnx': str(output_path), 'onnx_sha256': contract['onnx_sha256'],
            'contract': str(contract_path), 'verification': verification}


def digest(result: dict[str, Any] | None) -> dict[str, Any] | None:
    """The few numbers of a finished UserNet's result.json the panel shows: stage, best epoch, uses_bdd, the
    validation source and, at MIN_USER_SCORE, validation and test recall, precision and false per frame, per kind
    too. None without a result."""
    if not isinstance(result, dict) or result.get('task') != 'usernet':
        return None

    def part(measured):
        found = (measured or {}).get('at_min_score') if isinstance(measured, dict) else None
        if not isinstance(found, dict):
            return None
        return {'frames': measured.get('frames'), 'recall': found.get('recall'), 'precision': found.get('precision'),
                'false_per_frame': found.get('false_per_frame'),
                'recall_by_kind': {kind: (found.get('by_kind') or {}).get(kind, {}).get('recall') for kind in KINDS}}

    start = result.get('start') if isinstance(result.get('start'), dict) else None
    return {'stage': result.get('stage'), 'best_epoch': result.get('best_epoch'), 'epochs': result.get('epochs'),
            'epochs_run': len(result['history']) if isinstance(result.get('history'), list) and result['history'] else None,
            'parameters': result.get('parameters'), 'split_sha256': result.get('split_sha256'),
            'uses_bdd': bool(result.get('uses_bdd')), 'validation_source': result.get('validation_source'),
            # The UserNet it went on from, without its path: its run, epoch, stage, split and whether it learnt BDD100K.
            'start': None if start is None else {key: start.get(key) for key in ('run', 'epoch', 'stage', 'split_sha256',
                                                                                 'uses_bdd')},
            'validation': part(result.get('validation')), 'test': part(result.get('test'))}
