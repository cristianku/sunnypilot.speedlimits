"""SignNet: a sign detector made for the comma's own GPU (the Adreno), to cut the signs out of the road frame in the car.

The old Detector (SSDlite MobileNetV3 at 1344 x 760, the cutter until 30 Oct 2026) takes 540 ms a frame on the Adreno with openpilot's tinygrad
flags, too slow; on the Chestnut it would stall the driving model (docs/NN_VISION_SPEED_LIMIT_REMAINING_WORK.md).
SignNet is the other way round: few layers of plain 3 x 3 convolutions (tinygrad pays for every layer), one output,
and only the top band of the frame (BAND_HEIGHT rows: 97 % of the reviewed signs end above row 420 of 760) at full
resolution, so the small far signs are not lost to a smaller picture. 141 ms on the Adreno untrained (27 Oct 2026).
``band_width`` trains it on a smaller band (672: half the resolution) to see how far the picture can shrink.

The output is a heat map at ``stride`` (a sign centred in this cell, sure this much) and, per cell, the sign's log
width and height in pixels and the offset of its centre in the cell (CenterNet): no default boxes, no non-maximum
suppression but the local maxima of the heat map. The classifier reads each box from the full frame, as with the
old Detector.

``nnslr train-signnet`` trains it on the old Detector's plan (prepare_detector_training: the same frames, boxes,
families and split) and measures it on validation and test with the old Detector's scoring (evaluation.score_frame,
detector_size_metrics), so the two compare one to one. The plan's ignore boxes (switched-off variable displays) are
neither signs nor background: the loss does not judge their cells, and a box found on one is dropped before scoring.
In training, a quarter of the speed signs of a frame are drawn as an end of limit, their box kept (synthetic_signs:
Switzerland has too few real ones to learn them), besides the flip and the light of the band.
Needs the GPU environment (.venv-gpu).

SignNet is the one cutter: the car runs it (sunnypilot's speed_limits/vision.py, mirrored by the CAR_* constants), and
the server's chain (nnslr_tools.chain: Prepare's Our networks, mine-negatives, evaluate-chain) reads it the same way:
the band made by _band_of, the heat map decoded by frame_boxes_of, at most CAR_MAX_BOXES a frame. The car keeps those
of CAR_MIN_SCORE or more; Prepare and mining go down to score 0.1, and only evaluate-chain's main score (0.3) is
exactly what the car reads. Only a
SignNet that fits_the_car can be the cutter in use. Torch is imported only where a network runs: the constants and
helpers below the imports need none.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

from nnslr_tools import synthetic_signs

BAND_WIDTH = 1344
BAND_HEIGHT = 416
DEFAULT_WIDTHS = (16, 32, 48, 64)
DEFAULT_STRIDE = 8
# A centre's heat spreads over the cells around it with this share of the sign's size as its Gaussian sigma.
SIGMA_SHARE = 1 / 6
MAX_DETECTIONS = 50
SCORE_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
# Boxes per frame the classifier after it can afford on the Adreno (about 5 ms a box).
BOX_BUDGETS = (3, 5, 10)
# The SignNet the car runs (sunnypilot/speed_limits/vision.py: SIGNNET_WIDTH x SIGNNET_HEIGHT, SIGNNET_STRIDE,
# MAX_BOXES, MIN_BOX_SCORE): the band, stride and boxes of the cutter in use; "Train the cutter" trains this one
# (the run on the car, 2026-10-28_1028-signnet-twothirds-s8-f1, had these fuse kernel and widths).
CAR_BAND = (896, 288)
CAR_STRIDE = 8
CAR_FUSE_KERNEL = 1
CAR_WIDTHS = (16, 24, 32, 48)
CAR_MAX_BOXES = 5
CAR_MIN_SCORE = 0.3
# What a signnet-best.pt says of its data; one written before 30 Oct 2026 lacks them, its result.json has them.
DATA_KEYS = ('dataset_sha256', 'split_sha256', 'families')


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ValueError("training_dependencies_missing: run it with .venv-gpu") from exc
    return torch, nn


def band_size(band_width: int = BAND_WIDTH) -> tuple[int, int]:
    """The band at ``band_width`` pixels across: the same share of the frame as BAND_HEIGHT, its height rounded up to
    a multiple of 16 (the deepest stride)."""
    return band_width, math.ceil(BAND_HEIGHT * band_width / BAND_WIDTH / 16) * 16


def fits_the_car(band, stride) -> bool:
    """Whether a SignNet of this band [width, height] and stride runs on the car (vision.py takes CAR_BAND and
    CAR_STRIDE only)."""
    try:
        return [int(v) for v in band] == list(CAR_BAND) and int(stride) == CAR_STRIDE
    except (TypeError, ValueError):
        return False


def build_signnet(widths=DEFAULT_WIDTHS, stride: int = DEFAULT_STRIDE, fuse_kernel: int = 3):
    """The network: strides 2, 4, 8, 16; the deepest map brought back to ``stride`` (4 or 8) and joined with the
    map of that stride by a convolution of ``fuse_kernel`` (3, or 1: on the Adreno the 3 x 3 at stride 4 is 40 % of
    the time); out: heat (1, sigmoid) and shape (log width, log height, offset x, offset y)."""
    torch, nn = _torch()
    if stride not in (4, 8):
        raise ValueError("signnet stride must be 4 or 8")
    if fuse_kernel not in (1, 3):
        raise ValueError("signnet fuse kernel must be 1 or 3")

    def block(cin, cout, step, kernel=3):
        return nn.Sequential(nn.Conv2d(cin, cout, kernel, step, kernel // 2, bias=False), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    class SignNet(nn.Module):
        def __init__(self):
            super().__init__()
            a, b, c, d = widths
            self.stride = stride
            self.s2 = block(3, a, 2)
            self.s4 = nn.Sequential(block(a, b, 2), block(b, b, 1))
            self.s8 = nn.Sequential(block(b, c, 2), block(c, c, 1))
            self.s16 = nn.Sequential(block(c, d, 2), block(d, d, 1))
            self.fuse = block(d + (b if stride == 4 else c), c, 1, fuse_kernel)
            self.out = nn.Conv2d(c, 5, 1)
            # Heat starts near 0.01 everywhere (CenterNet's prior): the focal loss starts from few false alarms.
            nn.init.constant_(self.out.bias[:1], -4.6)

        def forward(self, images):
            x4 = self.s4(self.s2(images))
            x8 = self.s8(x4)
            x16 = self.s16(x8)
            skip = x4 if self.stride == 4 else x8
            up = nn.functional.interpolate(x16, size=skip.shape[-2:], mode='nearest')
            y = self.out(self.fuse(torch.cat([up, skip], 1)))
            return torch.sigmoid(y[:, :1]), y[:, 1:]

    return SignNet()


def band_boxes(boxes: list[list[float]], width: int, band_width: int = BAND_WIDTH, *, ignore: bool = False) -> list[list[float]]:
    """Reviewed boxes in the band's pixels (the frame scaled to band_width): those whose centre is in the band;
    with ``ignore`` (ignore boxes) every box that reaches into it, cut to the band."""
    scale = band_width / float(width)
    band_height = band_size(band_width)[1]
    kept = []
    for x1, y1, x2, y2 in boxes:
        box = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        if (box[1] < band_height if ignore else (box[1] + box[3]) / 2 < band_height):
            kept.append([box[0], max(0, box[1]), box[2], min(box[3], float(band_height))])
    return kept


def encode(boxes: list[list[float]], stride: int, band_width: int = BAND_WIDTH, ignore: list[list[float]] = ()):
    """Targets of one band: heat (1, H, W), shape (4, H, W), mask (H, W) of the cells that hold a centre, and
    weight (H, W): 0 on the cells of the ignore boxes (a switched-off display: neither a sign nor background),
    1 elsewhere. The loss does not judge the heat of a cell of weight 0 unless a sign is centred there."""
    torch, _ = _torch()
    band_w, band_h = band_size(band_width)
    height, width = band_h // stride, band_w // stride
    heat = torch.zeros(1, height, width)
    shape = torch.zeros(4, height, width)
    mask = torch.zeros(height, width)
    weight = torch.ones(height, width)
    ys = torch.arange(height).view(-1, 1).float()
    xs = torch.arange(width).view(1, -1).float()
    for x1, y1, x2, y2 in boxes:
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2 / stride, (y1 + y2) / 2 / stride
        ix, iy = min(int(cx), width - 1), min(int(cy), height - 1)
        sigma = max(max(w, h) / stride * SIGMA_SHARE, 0.5)
        gauss = torch.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma * sigma))
        heat[0] = torch.maximum(heat[0], gauss)
        shape[:, iy, ix] = torch.tensor([math.log(w), math.log(h), cx - ix, cy - iy])
        mask[iy, ix] = 1.0
    for x1, y1, x2, y2 in ignore:
        # The cells the box touches, one more around them: a peak next to the panel is still the panel.
        ix1, iy1 = max(int(x1 / stride) - 1, 0), max(int(y1 / stride) - 1, 0)
        ix2, iy2 = min(int(x2 / stride) + 1, width - 1), min(int(y2 / stride) + 1, height - 1)
        weight[iy1:iy2 + 1, ix1:ix2 + 1] = 0.0
    return heat, shape, mask, weight


def loss(pred_heat, pred_shape, heat, shape, mask, weight=None):
    """CenterNet's focal loss on the heat map, plus L1 on the shape at the centres; both per sign. With weight
    (N, H, W), the cells of weight 0 are not judged as background (the ignore boxes of encode)."""
    torch, _ = _torch()
    pred = pred_heat.clamp(1e-4, 1 - 1e-4)
    positive = heat.eq(1).float()
    negative = 1 - positive
    if weight is not None:
        negative = negative * weight.unsqueeze(1)
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * positive
    neg_loss = torch.log(1 - pred) * pred ** 2 * (1 - heat) ** 4 * negative
    count = positive.sum().clamp(min=1)
    focal = -(pos_loss.sum() + neg_loss.sum()) / count
    centre = mask.unsqueeze(1)
    shape_loss = (torch.abs(pred_shape - shape) * centre).sum() / (centre.sum() * 4).clamp(min=1)
    return focal + shape_loss, {'focal': float(focal.detach()), 'shape': float(shape_loss.detach())}


def decode(heat, shape, stride: int, *, limit: int = MAX_DETECTIONS, min_score: float = 0.01) -> list[list[dict]]:
    """Boxes of a batch, in band pixels: the local maxima of the heat map (3 x 3), at most ``limit`` a frame."""
    _, nn = _torch()
    peaks = heat * (nn.functional.max_pool2d(heat, 3, 1, 1) == heat).float()
    batch, _, height, width = heat.shape
    scores, indexes = peaks.view(batch, -1).topk(min(limit, height * width))
    found = []
    for b in range(batch):
        frame = []
        for score, index in zip(scores[b].tolist(), indexes[b].tolist()):
            if score < min_score:
                break
            iy, ix = divmod(index, width)
            log_w, log_h, ox, oy = shape[b, :, iy, ix].tolist()
            w, h = math.exp(min(log_w, 8.0)), math.exp(min(log_h, 8.0))
            cx, cy = (ix + ox) * stride, (iy + oy) * stride
            frame.append({'box': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 'score': score})
        found.append(frame)
    return found


def _band_of(image, band_width: int = BAND_WIDTH):
    """The top band of a decoded frame (a PIL image) as a uint8 tensor (3, band height, band_width): the frame
    scaled to band_width (PIL bilinear, the car does the same arithmetic) and its top rows. No normalisation: the
    network takes it / 255."""
    torch, _ = _torch()
    import numpy as np
    from PIL import Image
    band_w, band_h = band_size(band_width)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    width, height = image.size
    if width != band_w:
        image = image.resize((band_w, round(height * band_w / width)), Image.BILINEAR)
    array = np.asarray(image.crop((0, 0, band_w, band_h)), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def _band_image(path: Path, band_width: int = BAND_WIDTH, paint=None):
    """The frame's top band as a uint8 tensor (3, band height, band_width), and the frame's width. paint: a function
    of the frame (PIL, RGB) that returns it drawn over (training only: synthetic_signs.paint_frame)."""
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB')
        if paint is not None:
            image = paint(image)
        return _band_of(image, band_width), image.size[0]


def frame_boxes_of(heat, shape, stride: int, band_width: int, frame_width: float, *, limit: int = CAR_MAX_BOXES,
                   min_score: float = CAR_MIN_SCORE) -> list[dict]:
    """The boxes of one frame (heat (1, 1, H, W), shape (1, 4, H, W) on the CPU) in frame pixels, best first: the
    local maxima of decode, at most ``limit`` of at least ``min_score``, each coordinate times frame_width /
    band_width, as the car's vision.signnet_boxes (not clamped to the frame)."""
    scale = float(frame_width) / band_width
    return [{'box': [v * scale for v in p['box']], 'score': p['score']}
            for p in decode(heat, shape, stride, limit=limit, min_score=min_score)[0]]


def paints(frame: dict[str, Any], train: bool) -> bool:
    """Whether a frame gets synthetic end-of-limit signs (synthetic_signs.paint_frame): our train frames only, never
    a Mapillary window (source 'mapillary'), whose signs are other countries' real ones."""
    return bool(train) and frame.get('source') != 'mapillary'


def paintable_train_boxes(frames: list[dict[str, Any]]) -> int:
    """How many train boxes synthetic_signs may draw over: those of the frames that paints."""
    return sum(synthetic_signs.paintable(box, family) for frame in frames if paints(frame, True)
               for box, family in zip(frame['boxes'], frame.get('box_families') or []))


def start_run(path: Path) -> tuple[str, bool]:
    """The run of a start's weights and whether they are its Mapillary stage: (<run>, True) for
    runs/<run>/pretrain/signnet-best.pt (the first stage of a two-stage cutter), (<run>, False) for
    runs/<run>/signnet-best.pt."""
    path = Path(path)
    if path.parent.name == 'pretrain' and len(path.parts) >= 3:
        return path.parts[-3], True
    return path.parent.name, False


def learnt_mapillary(plan: dict[str, Any], start: dict[str, Any] | None) -> bool:
    """Whether a network learnt from the Mapillary windows (MTSD, CC BY-NC-SA 4.0): its own plan took them, or its
    start did, anywhere in that start's chain (signnet_start's 'uses_mapillary'). signnet-best.pt and result.json
    keep it, so the licence goes with the weights even without the folders of the starts."""
    start = start if isinstance(start, dict) else {}
    return bool(plan.get('mapillary') or plan.get('mapillary_only') or start.get('uses_mapillary')
                or start.get('mapillary') or start.get('mapillary_only'))


def _dataset(frames: list[dict], data_root: Path, stride: int, *, train: bool, seed: int, band_width: int = BAND_WIDTH):
    _torch()
    from torch.utils.data import Dataset

    class Bands(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            frame = frames[index]
            paint = None
            if paints(frame, train):
                # Some speed signs drawn as an end of limit, the box kept (synthetic_signs): the few real ones are
                # not enough to learn them. Python's random is seeded apart in every DataLoader worker.
                def paint(image):
                    return synthetic_signs.paint_frame(image, frame['boxes'], frame.get('box_families') or [],
                                                       random)[0]
            image, width = _band_image(data_root / frame['image_path'], band_width, paint)
            boxes = band_boxes(frame['boxes'], width, band_width)
            ignore = band_boxes(frame.get('ignore_boxes') or [], width, band_width, ignore=True)
            image = image.float() / 255.0
            if train:
                # Python's random is seeded apart in every DataLoader worker (base seed + worker id).
                rng = random
                if rng.random() < 0.5:
                    image = image.flip(-1)
                    boxes = [[band_width - x2, y1, band_width - x1, y2] for x1, y1, x2, y2 in boxes]
                    ignore = [[band_width - x2, y1, band_width - x1, y2] for x1, y1, x2, y2 in ignore]
                image = (image * rng.uniform(0.7, 1.3) + rng.uniform(-0.08, 0.08)).clamp(0, 1)
            heat, shape, mask, weight = encode(boxes, stride, band_width, ignore)
            return image, heat, shape, mask, weight, index

    return Bands()


def _in_ignore(box: list[float], ignore: list[list[float]]) -> bool:
    """A found box on a switched-off display: its centre inside an ignore box, or half overlapping one."""
    from nnslr_tools.evaluation import box_iou
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 or box_iou(box, [x1, y1, x2, y2]) >= 0.5 for x1, y1, x2, y2 in ignore)


def measure(model, frames: list[dict], data_root: Path, stride: int, device, *, batch: int = 16,
            workers: int = 8, band_width: int = BAND_WIDTH) -> dict[str, Any]:
    """The cutter's measures on ``frames``: per score threshold (detector_size_metrics, reviewed boxes of the whole
    frame, those below the band count as missed), per box budget (the best N boxes of each frame, at any score) and
    ``car``: what the car reads, the best CAR_MAX_BOXES boxes of score CAR_MIN_SCORE or more. All of them match a found
    box to a reviewed one at IoU 0.5; ``car_centre`` is ``car`` with the lenient match (evaluation.centre_match: IoU 0.5,
    or the found box's centre on the reviewed box and its size 0.5 to 2 times it), fairer to the far, small signs. A
    box found on an ignore box of the frame (a switched-off display) is dropped before any of them: neither found nor
    false."""
    torch, _ = _torch()
    from torch.utils.data import DataLoader

    from nnslr_tools.evaluation import detector_size_metrics, score_frame
    loader = DataLoader(_dataset(frames, data_root, stride, train=False, seed=0, band_width=band_width),
                        batch_size=batch, shuffle=False, num_workers=workers)
    predictions: dict[int, list[dict]] = {}
    ignored = 0
    model.eval()
    with torch.no_grad():
        for image, _, _, _, _, indexes in loader:
            heat, shape = model(image.to(device))
            heat, shape = heat.float().cpu(), shape.float().cpu()
            for b, index in enumerate(indexes.tolist()):
                ignore = frames[index].get('ignore_boxes') or []
                kept = []
                for p in frame_boxes_of(heat[b:b + 1], shape[b:b + 1], stride, band_width, frames[index]['width'],
                                        limit=MAX_DETECTIONS, min_score=0.01):
                    if ignore and _in_ignore(p['box'], ignore):
                        ignored += 1
                        continue
                    kept.append(p)
                predictions[index] = kept

    def car_boxes(found: list[dict]) -> list[dict]:
        return [p for p in found if p['score'] >= CAR_MIN_SCORE][:CAR_MAX_BOXES]

    def scored(select, match: str = 'iou') -> dict[str, Any]:
        items = []
        for index, frame in enumerate(frames):
            chosen = select(predictions.get(index, []))
            item = score_frame(frame['boxes'], chosen, score_threshold=0.0, iou_threshold=0.5, match=match)
            item['truth_families'] = list(frame.get('box_families') or [])
            item['label_basis'] = frame.get('label_basis')
            items.append(item)
        return detector_size_metrics(items)

    return {
        'by_score_threshold': {str(t): scored(lambda found, t=t: [p for p in found if p['score'] >= t])
                               for t in SCORE_THRESHOLDS},
        'by_box_budget': {str(n): scored(lambda found, n=n: found[:n]) for n in BOX_BUDGETS},
        'car': scored(car_boxes),
        'car_centre': scored(car_boxes, 'centre'),
        'ignore_boxes': sum(len(f.get('ignore_boxes') or []) for f in frames),
        'found_on_ignore_boxes': ignored,
    }


def car_measure(measured: Any, *, match: str = 'iou') -> dict[str, Any] | None:
    """What the car reads of one partition of a result.json (measure's output): its ``car`` entry, the best
    CAR_MAX_BOXES boxes of score CAR_MIN_SCORE or more; for a result written before it, by_score_threshold at
    CAR_MIN_SCORE (the cap of 5 boxes almost never bites there: about 0.35 boxes a frame). None without either.
    With match 'centre': its ``car_centre`` entry, the same boxes with the lenient match; None in a result.json
    written before it (remeasure_car_centre adds it)."""
    if not isinstance(measured, dict):
        return None
    if match == 'centre':
        found = measured.get('car_centre')
        return found if isinstance(found, dict) else None
    for found in (measured.get('car'), (measured.get('by_score_threshold') or {}).get(str(CAR_MIN_SCORE))):
        if isinstance(found, dict):
            return found
    return None


def checkpoint_of(model, plan: dict[str, Any], *, widths, stride: int, fuse_kernel: int, band_width: int,
                  epoch: int, start: dict[str, Any] | None = None) -> dict[str, Any]:
    """What signnet-best.pt holds: the weights, the network's settings, and the data it learnt from (DATA_KEYS:
    the cutter's checks of the same split and its families read them, as a detector's; and whether it learnt the
    Mapillary windows, the licence of the networks it starts). ``start``: signnet_start's record of the weights it
    went on from, kept with 'uses_mapillary' (learnt_mapillary) so a copy of this file alone still names the licence
    of a Mapillary stage before it."""
    return {'task': 'signnet', 'state_dict': model.state_dict(), 'widths': list(widths), 'stride': stride,
            'fuse_kernel': fuse_kernel, 'band': list(band_size(band_width)), 'epoch': epoch,
            **{key: plan.get(key) for key in DATA_KEYS},
            'mapillary': plan.get('mapillary'), 'mapillary_only': bool(plan.get('mapillary_only')),
            'trained_on': plan.get('trained_on'), 'start': start, 'uses_mapillary': learnt_mapillary(plan, start)}


def checkpoint_data(checkpoint: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """The checkpoint with DATA_KEYS filled from the result.json beside it when it lacks them (one written before
    they were saved in it). Without either, families is 'all': every SignNet so far learnt all the families (the
    legacy 'speed' of a cutter without families is the old Detector's)."""
    if all(checkpoint.get(key) is not None for key in DATA_KEYS):
        return checkpoint
    try:
        result = json.loads((Path(checkpoint_path).parent / 'result.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        result = None
    if isinstance(result, dict) and result.get('task') == 'signnet':
        checkpoint = {**checkpoint, **{key: result.get(key) for key in DATA_KEYS if checkpoint.get(key) is None}}
    if checkpoint.get('task') == 'signnet' and checkpoint.get('families') is None:
        checkpoint = {**checkpoint, 'families': 'all'}
    return checkpoint


def signnet_not_trainable(plan: dict[str, Any]) -> str:
    """Why a plan cannot train a SignNet: it needs frames and boxes in train and validation (the plan's
    ``trainable``, as the old Detector's). Those counts are our frames; with the Mapillary windows, what train
    really holds (plan['trained_on']) comes after them."""
    trained_on = plan.get('trained_on')
    return ("training_plan_not_trainable: SignNet needs frames with boxes in train and validation; this plan has "
            f"frames {plan.get('counts_by_split') or {}} and boxes {plan.get('boxes_by_split') or {}} "
            f"(families {plan.get('families')!r})"
            + (f"; train holds {trained_on}" if plan.get('mapillary') and trained_on else ''))


def signnet_start(path: Path, *, widths, stride: int, fuse_kernel: int, band_width: int) -> tuple[dict, dict]:
    """The weights of a finished SignNet to start from (nnslr train-signnet --init-from) and what result.json records
    of it: its file and sha256, its run (start_run), the epoch, split and families it learnt, and whether it learnt
    the Mapillary windows (its own 'mapillary' and 'mapillary_only', so the lineage survives the start). The heat says
    "a sign is here" whatever the families, so a start of any families fits; its widths, grid, fuse convolution and
    band must be this network's (every layer is copied as it is)."""
    torch, _ = _torch()
    from nnslr_tools.exporting import sha256_file
    path = Path(path)
    saved = checkpoint_data(torch.load(path, map_location='cpu', weights_only=True), path)
    if saved.get('task') != 'signnet' or not isinstance(saved.get('state_dict'), dict):
        raise ValueError(f'the start is not a SignNet: {path}')
    wanted = {'widths': list(widths), 'stride': stride, 'fuse kernel': fuse_kernel, 'band': list(band_size(band_width))}
    found = {'widths': saved.get('widths'), 'stride': saved.get('stride'), 'fuse kernel': saved.get('fuse_kernel', 3),
             'band': saved.get('band')}
    different = [f'{key} {found[key]} (this run {wanted[key]})' for key in wanted if found[key] != wanted[key]]
    if different:
        raise ValueError(f'the start is another network: {", ".join(different)}: {path}')
    run, stage = start_run(path)
    # Whether the start learnt MTSD itself or went on from weights that did (a checkpoint of before uses_mapillary
    # says it by its own mapillary fields only).
    uses = bool(saved.get('uses_mapillary') or saved.get('mapillary') or saved.get('mapillary_only'))
    return saved['state_dict'], {'path': str(path), 'sha256': sha256_file(path), 'run': run, 'mapillary_stage': stage,
                                 'epoch': saved.get('epoch'), 'split_sha256': saved.get('split_sha256'),
                                 'dataset_sha256': saved.get('dataset_sha256'), 'families': saved.get('families'),
                                 'mapillary': saved.get('mapillary'),
                                 'mapillary_only': bool(saved.get('mapillary_only')), 'uses_mapillary': uses}


def train_signnet(plan: dict[str, Any], data_root: Path, run_dir: Path, *, widths=CAR_WIDTHS,
                  stride: int = CAR_STRIDE, epochs: int = 40, batch: int = 16, learning_rate: float = 3e-3,
                  workers: int = 8, seed: int = 1, band_width: int = CAR_BAND[0], fuse_kernel: int = CAR_FUSE_KERNEL,
                  init_from: Path | None = None, log=None) -> dict[str, Any]:
    """Train on the plan's train frames, keep the epoch with the best validation recall at 5 boxes a frame, then
    measure it on validation and test. Writes signnet-best.pt and result.json in run_dir. The defaults are the car's
    SignNet (CAR_*). init_from: a finished SignNet whose weights it goes on from (signnet_start), else random weights.
    Every epoch logs one line (the panel reads it): ``signnet epoch E/N train_loss L val_recall R
    val_false_per_frame F Ss``, with the validation at 5 boxes a frame. ``log`` defaults to a print that flushes:
    the script tees it into a file, and the panel follows the epochs as they come.

    A plan of the Mapillary windows alone (plan['mapillary_only'], the first stage of a two-stage cutter) is named
    signnet_pretrain in those lines and ends with "Mapillary stage complete: <dir>". Its windows have speed signs
    only, so it keeps the epoch with the best validation recall of the speed signs (val_speed_recall in its line):
    its recall over every family, the other signs of our validation included, is low by design."""
    if not plan.get('trainable'):
        raise ValueError(signnet_not_trainable(plan))
    train_count = sum(1 for frame in plan['frames'] if frame['split'] == 'train')
    if train_count < batch:
        # The loader drops a last batch smaller than the others: fewer frames than one batch would train nothing
        # (a narrow region or share of the Mapillary windows alone can hold so few).
        raise ValueError(f"{signnet_not_trainable(plan)}; {train_count} train frames, fewer than one batch of {batch}")
    if log is None:
        def log(line):
            print(line, flush=True)
    torch, _ = _torch()
    from torch.utils.data import DataLoader
    random.seed(seed)
    torch.manual_seed(seed)
    data_root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    by_split: dict[str, list[dict]] = {}
    for frame in plan['frames']:
        by_split.setdefault(frame['split'], []).append(frame)
    train_frames, validation, test = by_split.get('train', []), by_split.get('validation', []), by_split.get('test', [])
    pretrain = bool(plan.get('mapillary_only'))
    name = 'signnet_pretrain' if pretrain else 'signnet'
    if pretrain:
        log(f"{name} keeps the epoch with the best validation recall of the speed signs at 5 boxes a frame: the "
            "Mapillary windows have speed signs only, so the recall over every family is low by design")
    model = build_signnet(widths, stride, fuse_kernel)
    start = None
    if init_from is not None:
        weights, start = signnet_start(init_from, widths=widths, stride=stride, fuse_kernel=fuse_kernel,
                                       band_width=band_width)
        model.load_state_dict(weights, strict=True)
        log(f"{name} starts from {start['run']}" + ("'s Mapillary stage" if start['mapillary_stage'] else '')
            + f": every layer of its epoch {start['epoch']}"
            + ('' if start['split_sha256'] == plan.get('split_sha256') else ', learnt on another split'))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(_dataset(train_frames, data_root, stride, train=True, seed=seed, band_width=band_width),
                        batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                        persistent_workers=workers > 0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=epochs * len(loader))
    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, epochs + 1):
        model.train()
        started, total, parts = time.monotonic(), 0.0, {'focal': 0.0, 'shape': 0.0}
        for image, heat, shape, mask, weight, _ in loader:
            pred_heat, pred_shape = model(image.to(device))
            value, detail = loss(pred_heat, pred_shape, heat.to(device), shape.to(device), mask.to(device),
                                 weight.to(device))
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            scheduler.step()
            total += value.item()
            for key in parts:
                parts[key] += detail[key]
        steps = max(len(loader), 1)
        checked = measure(model, validation, data_root, stride, device, workers=workers,
                          band_width=band_width)['by_box_budget']['5']
        speed = ((checked.get('recall_by_family') or {}).get('speed') or {}).get('recall')
        history.append({'epoch': epoch, 'loss': total / steps, 'focal': parts['focal'] / steps,
                        'shape': parts['shape'] / steps, 'validation_recall_at_5': checked['recall_iou'],
                        'validation_false_positives_per_image_at_5': checked['false_positives_per_image'],
                        'seconds': round(time.monotonic() - started, 1)})
        if pretrain:
            history[-1]['validation_speed_recall_at_5'] = speed
        log(f"{name} epoch {epoch}/{epochs} train_loss {total / steps:.4f} val_recall {checked['recall_iou']:.3f} "
            f"val_false_per_frame {checked['false_positives_per_image']:.2f} "
            + (f"val_speed_recall {speed if speed is not None else checked['recall_iou']:.3f} " if pretrain else '')
            + f"{history[-1]['seconds']:.0f}s")
        chosen = speed if pretrain and speed is not None else checked['recall_iou']
        if chosen > best:
            best, best_epoch = chosen, epoch
            torch.save(checkpoint_of(model, plan, widths=widths, stride=stride, fuse_kernel=fuse_kernel,
                                     band_width=band_width, epoch=epoch, start=start), run_dir / 'signnet-best.pt')
    log(f"{name} measure the best epoch {best_epoch} on validation and test")
    checkpoint = torch.load(run_dir / 'signnet-best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    result = {
        'task': 'signnet', 'widths': list(widths), 'stride': stride, 'fuse_kernel': fuse_kernel,
        'band': list(band_size(band_width)),
        'parameters': sum(p.numel() for p in model.parameters()), 'epochs': epochs, 'best_epoch': best_epoch,
        'dataset_sha256': plan.get('dataset_sha256'), 'split_sha256': plan.get('split_sha256'),
        'families': plan.get('families'), 'start': start, 'counts_by_split': plan.get('counts_by_split'),
        'boxes_by_split': plan.get('boxes_by_split'), 'ignore_boxes_by_split': plan.get('ignore_boxes_by_split'),
        'boxes_by_family': plan.get('boxes_by_family'), 'mapillary': plan.get('mapillary'),
        'mapillary_only': pretrain, 'trained_on': plan.get('trained_on'),
        'uses_mapillary': learnt_mapillary(plan, start),
        'best_epoch_by': 'validation_speed_recall_at_5' if pretrain else 'validation_recall_at_5',
        'synthetic_end_of_limit': {**synthetic_signs.summary(), 'paintable_train_boxes': paintable_train_boxes(
            train_frames)},
        'history': history,
        'validation': measure(model, validation, data_root, stride, device, workers=workers, band_width=band_width),
        'test': measure(model, test, data_root, stride, device, workers=workers, band_width=band_width),
    }
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    log(f"Mapillary stage complete: {run_dir}" if pretrain else f"SignNet complete: {run_dir}")
    return result


def remeasure_car_centre(plan: dict[str, Any], data_root: Path, run_dir: Path, *, device=None, batch: int = 16,
                         workers: int = 8, dry_run: bool = False) -> dict[str, Any]:
    """Add ``car_centre`` (measure's lenient car measure) to validation and test of a finished SignNet run's
    result.json, without training: its signnet-best.pt measured again on the plan's validation and test frames. The
    plan must be the run's (dataset_sha256, split_sha256 and families of its result.json), else the numbers would be
    of other frames. Nothing else of result.json changes; the ``car`` measured again is compared with the one saved
    (``car_unchanged``: False when the device rounds a box across the score 0.3 differently). With dry_run, measures
    and reports but does not write. Returns the run, the device, and per partition the strict and lenient recall
    and precision."""
    run_dir, data_root = Path(run_dir), Path(data_root)
    result_path = run_dir / 'result.json'
    try:
        result = json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'remeasure_not_a_finished_signnet: {result_path} unreadable') from exc
    if not isinstance(result, dict) or result.get('task') != 'signnet':
        raise ValueError(f'remeasure_not_a_finished_signnet: {result_path} is not a SignNet result')
    different = [key for key in DATA_KEYS if plan.get(key) != result.get(key)]
    if different:
        old = data_root / 'annotations' / 'replaced' / f"{result.get('dataset_sha256')}.jsonl"
        hint = f'; its dataset is kept at {old}' if 'dataset_sha256' in different and old.is_file() else ''
        raise ValueError('remeasure_plan_mismatch: the plan differs from the run in '
                         + ', '.join(f'{key} ({plan.get(key)!r} against {result.get(key)!r})' for key in different) + hint)
    torch, _ = _torch()
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, checkpoint = load_signnet(run_dir / 'signnet-best.pt')
    model = model.to(device)
    band_width, stride = int(checkpoint['band'][0]), int(checkpoint['stride'])
    summary: dict[str, Any] = {'run_dir': str(run_dir), 'device': str(device), 'dry_run': dry_run}
    for partition in ('validation', 'test'):
        frames = [frame for frame in plan['frames'] if frame.get('split') == partition]
        if not isinstance(result.get(partition), dict):
            summary[partition] = None
            continue
        measured = measure(model, frames, data_root, stride, device, batch=batch, workers=workers,
                           band_width=band_width)
        saved = result[partition].get('car')
        result[partition]['car_centre'] = measured['car_centre']
        counts = ('true_positives', 'false_positives', 'false_negatives')
        summary[partition] = {
            'frames': len(frames),
            'car_unchanged': (None if not isinstance(saved, dict)
                              else all(saved.get(key) == measured['car'][key] for key in counts)),
            **{f'{name}_{key}': measured[entry][f'{key}_iou']
               for name, entry in (('strict', 'car'), ('centre', 'car_centre')) for key in ('recall', 'precision')},
        }
    if not dry_run:
        temporary = result_path.with_name(result_path.name + '.tmp')
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        temporary.replace(result_path)
    return summary


def decode_output(out, stride: int, **kw) -> list[list[dict]]:
    """``decode`` on the one tensor the exported ONNX gives: heat in channel 0, the shape in channels 1 to 4."""
    return decode(out[:, :1], out[:, 1:], stride, **kw)


def model_of(checkpoint: dict[str, Any]):
    """The network of a loaded signnet-best.pt, in eval mode on the CPU (an old one without fuse_kernel: 3)."""
    if not isinstance(checkpoint, dict) or checkpoint.get('task') != 'signnet':
        raise ValueError('checkpoint_task_mismatch: expected signnet')
    if not checkpoint.get('state_dict') or not isinstance(checkpoint['state_dict'], dict):
        raise ValueError('checkpoint_missing_state_dict')
    model = build_signnet(tuple(checkpoint['widths']), checkpoint['stride'], checkpoint.get('fuse_kernel', 3))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval()


def load_signnet(checkpoint_path: Path):
    """The network of a signnet-best.pt in eval mode, and the checkpoint."""
    torch, _ = _torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
    return model_of(checkpoint), checkpoint


def export_signnet_onnx(checkpoint_path: Path, output_path: Path, *, verify: bool = True) -> dict[str, Any]:
    """SignNet as ONNX for the car. Input ``images`` (1, 3, band height, band width), float 0..1, RGB: the frame
    scaled to the band's width, its top rows. One output ``out`` (1, 5, H, W): heat (sigmoid), log width, log height,
    offset x, offset y (as two outputs the Adreno takes twice the time: 40 ms against 21 at 896 x 288). Checked
    against PyTorch with onnxruntime; writes ``<output>.contract.json`` beside it."""
    torch, nn = _torch()
    from nnslr_tools.exporting import _export_onnx, sha256_file
    model, checkpoint = load_signnet(checkpoint_path)
    band_w, band_h = checkpoint['band']
    stride = checkpoint['stride']

    class OneOutput(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, images):
            heat, shape = self.net(images)
            return torch.cat([heat, shape], 1)

    wrapped = OneOutput(model).eval()
    example = torch.rand(1, 3, band_h, band_w)
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
        session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        actual = session.run(['out'], {'images': example.numpy()})[0]
        max_abs_error = float(np.max(np.abs(expected - actual)))
        passed = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5))
        verification = {'enabled': True, 'passed': passed, 'max_abs_error': max_abs_error, 'rtol': 1e-4, 'atol': 1e-5,
                        'provider': 'CPUExecutionProvider'}
        if not passed:
            raise ValueError(f'onnx_parity_failed: max_abs_error={max_abs_error}')
    contract = {
        'task': 'signnet', 'widths': list(checkpoint['widths']), 'stride': stride,
        'fuse_kernel': checkpoint.get('fuse_kernel', 3), 'epoch': checkpoint.get('epoch'),
        'input': {'name': 'images', 'shape': [1, 3, band_h, band_w], 'dtype': 'float32', 'range': [0.0, 1.0],
                  'layout': 'RGB; the road frame scaled to the band width, its top rows (the band)'},
        'output': {'name': 'out', 'shape': [1, 5, band_h // stride, band_w // stride],
                   'channels': ['heat', 'log_width', 'log_height', 'offset_x', 'offset_y'],
                   'decode': 'a sign at each 3x3 local maximum of heat above the score wanted; its centre '
                             '((x + offset_x) * stride, (y + offset_y) * stride) and size (exp(log_width), '
                             'exp(log_height)), in band pixels; the frame scale is frame width / band width'},
        'onnx_sha256': sha256_file(output_path), 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256_file(Path(checkpoint_path)), 'verification': verification,
    }
    contract_path = output_path.with_suffix(output_path.suffix + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'valid': True, 'task': 'signnet', 'onnx': str(output_path), 'onnx_sha256': contract['onnx_sha256'],
            'contract': str(contract_path), 'verification': verification}
