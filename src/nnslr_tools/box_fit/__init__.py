"""The local box fit of the review page ("Fit box", key C): fits a rough box to the sign under it, on the CPU, in a
few tens of milliseconds, with no AI provider.

Two rankers look at the same decoded frame:

1. :mod:`.colour` makes about 20 candidates from colour masks, grabCut and edge snapping and a small tree model
   predicts each one's IoU with the sign. When it ranks the rough box itself first the answer is None: keep the box.
2. :mod:`.edges` makes candidates from edges, MSER regions and colour contrast and three tiny MLPs rank them.

The answer is the mean of the two boxes, pulled back toward the rough box when it already agrees with it
(``a = clip((IoU(fit, rough) - LO) / (1 - LO))``, ``box = (1 - a) * fit + a * rough``), so a well drawn box is
barely moved. The weights are ``colour.npz`` and ``edges.npz`` next to this file, made by ``nnslr train-box-fit``
from the boxes people drew (:mod:`.train`). numpy and OpenCV only, one OpenCV thread per process.

When ``refiner.onnx`` is next to this file the answer is the refiner's instead (:mod:`.refiner`, a small CNN trained
on the boxes people confirmed by :mod:`.refiner_train`, written only when it beat the two rankers on routes left
out): it also fits the signs the colour rules miss, such as the white ring on the black plate of a gantry display.
"""
from __future__ import annotations

import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLOUR_MODEL = HERE / 'colour.npz'
EDGES_MODEL = HERE / 'edges.npz'

# LO and W_COLOUR were chosen once on the benchmark, whose boxes come from every route: the route-fold measure of
# `nnslr train-box-fit` holds out the weights, not these two, so it is slightly optimistic (about 0.005 IoU).
LO = 0.65          # the pull-back toward the rough box starts when IoU(fit, rough) is above LO
W_COLOUR = 0.5     # the weight of the colour box in the mean
MIN_SIDE = 3.0     # a rough box with a shorter side is kept as it is

_models: tuple | None = None
_lock = threading.Lock()


def models() -> tuple:
    """(colour model, edges model), loaded once per process."""
    global _models
    if _models is None:
        with _lock:
            if _models is None:
                import cv2

                from nnslr_tools.box_fit import edges, gbdt
                cv2.setNumThreads(1)
                _models = (gbdt.load(COLOUR_MODEL), edges.load(EDGES_MODEL))
    return _models


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return 0.0 if u <= 0 else i / u


def clip_box(box, width, height) -> list[float] | None:
    """``box`` ordered and clipped to a ``width`` x ``height`` image; None when a side is shorter than MIN_SIDE."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))))
    y1, y2 = sorted((min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return [x1, y1, x2, y2]


def combine(by_colour, by_edges, rough) -> list[float] | None:
    """The answer from the two rankers' boxes: None (keep the box) when the colour ranker keeps it, else their mean
    pulled back toward ``rough`` as far as it already agrees with it."""
    if by_colour is None:
        return None
    by_edges = by_edges or rough
    out = [W_COLOUR * c + (1 - W_COLOUR) * e for c, e in zip(by_colour, by_edges)]
    a = min(1.0, max(0.0, (_iou(out, rough) - LO) / (1.0 - LO)))
    out = [float((1 - a) * f + a * q) for f, q in zip(out, rough)]
    return out if out[2] - out[0] >= 1 and out[3] - out[1] >= 1 else None


def fit_image(img, box, *, colour_model=None, edges_model=None, use_refiner=True) -> list[float] | None:
    """The fitted box of ``box`` ([x1, y1, x2, y2] in pixels of the decoded BGR image ``img``), or None to keep it.

    None means no better box was found: the sign is not there, or the box is already right. The refiner's answer
    when ``refiner.onnx`` is there (and ``use_refiner``), else the two rankers'."""
    from nnslr_tools.box_fit import colour, edges, refiner
    if use_refiner and colour_model is None and edges_model is None and refiner.available():
        rough = clip_box(box, img.shape[1], img.shape[0])
        if rough is None:
            return None
        fitted = refiner.refine(img, rough)
        # No side moved by half a pixel: the box was already right.
        if fitted is None or max(abs(f - r) for f, r in zip(fitted, rough)) < 0.5:
            return None
        return fitted
    if colour_model is None or edges_model is None:
        loaded = models()
        colour_model = loaded[0] if colour_model is None else colour_model
        edges_model = loaded[1] if edges_model is None else edges_model
    rough = clip_box(box, img.shape[1], img.shape[0])
    if rough is None:
        return None
    by_colour = colour.fit(img, rough, colour_model)
    if by_colour is None:
        return None
    return combine(by_colour, edges.fit(img, rough, edges_model), rough)


def read_image(path):
    """The image at ``path`` decoded as BGR; OSError when it cannot be read."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f'cannot decode {path}')
    return img


def fit_many(path, boxes) -> list[list[float] | None]:
    """:func:`fit_image` of each box on the image at ``path``, decoded once."""
    img = read_image(path)
    models()
    return [fit_image(img, box) for box in boxes]
