"""The box refiner: a small CNN that moves the four sides of a rough box onto the sign under it.

It sees a square crop centred on the rough box, ``CROP`` times its longer side, scaled to ``S`` x ``S`` pixels (BGR
/ 255, the border replicated where the crop leaves the frame), and answers the sign's box in crop units (0..1:
x1, y1, x2, y2). ``STEPS`` passes: each crops again around the previous answer, closer to the sign. It learns from
the boxes people drew (:mod:`.refiner_train`), so it also fits the signs the colour rules miss: a white ring on the
black plate of a gantry display, a grey plate on a grey sky.

Fit box uses it only on a rough box whose shorter side is under ``GATE`` pixels: one pass, then the colour and edge
rankers on its answer (:func:`nnslr_tools.box_fit.fit_image`). Measured on the routes left out (9 Nov 2026, 10698
rough boxes on the full frames): mean IoU 0.792 with the rankers alone, 0.798 so; on the small signs (under 20 px)
0.749 and 0.772; the big ones unchanged. A box that was already tight gets slightly worse (0.857 and 0.835), and
on bigger boxes the refiner loses to the rankers, hence the gate. ``GATE`` was chosen on that same table.

The weights are ``refiner.onnx`` next to this file, run by onnxruntime on one CPU thread; without it the review page
keeps the colour and edge rankers alone.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / 'refiner.onnx'
S = 64            # the crop's side in pixels, what the network sees
CROP = 2.0        # the crop's side as a share of the rough box's longer side
STEPS = 1         # passes, each around the previous answer
GATE = 30         # Fit box uses the refiner only under this shorter side of the rough box, in pixels

_session = None
_lock = threading.Lock()


def available() -> bool:
    """refiner.onnx is there and onnxruntime can run it (the extra ``preannotate`` of .venv)."""
    import importlib.util
    return MODEL.is_file() and importlib.util.find_spec('onnxruntime') is not None


def session():
    """The onnxruntime session of ``refiner.onnx``, made once per process (one thread)."""
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                _session = ort.InferenceSession(str(MODEL), options, providers=['CPUExecutionProvider'])
    return _session


def crop_frame(box) -> tuple[float, float, float]:
    """(x0, y0, side) of the square crop around ``box``, in pixels of the image."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = CROP * max(box[2] - box[0], box[3] - box[1], 4.0)
    return cx - side / 2, cy - side / 2, side


def crop(img, box):
    """The S x S crop (uint8 BGR) around ``box`` and its frame (x0, y0, side)."""
    import cv2
    x0, y0, side = crop_frame(box)
    k = S / side
    m = np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]])
    flags = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (S, S), flags=flags, borderMode=cv2.BORDER_REPLICATE), (x0, y0, side)


def to_image(pred, frame) -> list[float]:
    x0, y0, side = frame
    return [float(x0 + pred[0] * side), float(y0 + pred[1] * side), float(x0 + pred[2] * side),
            float(y0 + pred[3] * side)]


def to_crop(box, frame) -> list[float]:
    x0, y0, side = frame
    return [(box[0] - x0) / side, (box[1] - y0) / side, (box[2] - x0) / side, (box[3] - y0) / side]


def refine(img, box, steps: int = STEPS) -> list[float] | None:
    """The refined box of ``box`` ([x1, y1, x2, y2] in pixels of the BGR image ``img``) after ``steps`` passes, kept
    inside the image; None when the network answers no box (a side collapsed)."""
    height, width = img.shape[:2]
    for _ in range(steps):
        patch, frame = crop(img, box)
        x = patch.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = session().run(None, {'crop': x})[0][0]
        x1, y1, x2, y2 = to_image(pred, frame)
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        box = [x1, y1, x2, y2]
    return box
