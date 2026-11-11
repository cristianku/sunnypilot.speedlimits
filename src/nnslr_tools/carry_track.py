"""Apply previous on the frames of the video: a box confirmed on the frame before, followed onto this frame.

The frame before is 1/20 s away (or one second, on the pictures of the report): the sign looks the same, a little
bigger and a little further out. Its own pixels from the frame before (with a margin around it, so that its edges
count) are searched near where :mod:`nnslr_tools.carry_over` expects it, at a few scales, by normalized correlation
(OpenCV's ``TM_CCOEFF_NORMED``). A match at ``MIN_SCORE`` or better is the sign: the box goes there, no check needed.

Measured on 498 boxes of consecutive frames of the video confirmed by hand (00000154--fa7e8f49aa, 11 Nov 2026), mean
IoU with the box confirmed on the next frame: 0.693 left where it was, 0.871 with Fit box, 0.890 followed (30-60 px
signs 0.858 -> 0.924); about 8 ms a box on the CPU. Nothing is saved here.
"""
from __future__ import annotations

import math

# The scales a sign may take from one frame to the next, as a geometric series.
SCALES = (0.92, 1.15, 13)
# How far from where it is expected the sign is searched: this fraction of its longer side, and never less than REACH_PX.
REACH = 0.5
REACH_PX = 24
# A match at least this good is the sign (in the measure, every box followed scored over 0.6).
MIN_SCORE = 0.7
# The margin of the template around the sign, as a fraction of its shorter side, and at least MARGIN_PX pixels.
MARGIN = 0.15
MARGIN_PX = 2.0
# A sign smaller than this (pixels of the template) is not followed: too few pixels to match.
MIN_TEMPLATE = 6
# A template with less contrast than this (standard deviation of its grey levels) is flat: it matches anything.
MIN_CONTRAST = 4.0


def follow(before, now, old: list, guess: list) -> list[float] | None:
    """Where the sign of box ``old`` on picture ``before`` is on picture ``now``, searched around box ``guess``.

    The pictures are images of the same size (BGR or grey); the boxes ``[x1, y1, x2, y2]`` in their pixels. None when no
    match reaches MIN_SCORE."""
    import cv2
    import numpy as np
    height, width = before.shape[:2]
    if now.shape[:2] != (height, width):
        return None
    x1, y1, x2, y2 = old
    margin = max(MARGIN_PX, MARGIN * min(x2 - x1, y2 - y1))
    tx1, ty1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
    tx2, ty2 = int(min(width, x2 + margin)), int(min(height, y2 + margin))
    template = before[ty1:ty2, tx1:tx2]
    if template.shape[0] < MIN_TEMPLATE or template.shape[1] < MIN_TEMPLATE or float(template.std()) < MIN_CONTRAST:
        return None
    # The search area: the template placed on the guess, widened by the reach and by the largest growth.
    gx, gy = guess[0] - (x1 - tx1), guess[1] - (y1 - ty1)
    reach = max(REACH_PX, REACH * max(x2 - x1, y2 - y1))
    grow = SCALES[1]
    sx1, sy1 = int(max(0, gx - reach)), int(max(0, gy - reach))
    sx2 = int(min(width, gx + template.shape[1] * grow + reach))
    sy2 = int(min(height, gy + template.shape[0] * grow + reach))
    area = now[sy1:sy2, sx1:sx2]
    best_score, best = -2.0, None
    low, high, count = SCALES
    for step in range(count):
        scale = low * (high / low) ** (step / (count - 1))
        w, h = max(4, round(template.shape[1] * scale)), max(4, round(template.shape[0] * scale))
        if w >= area.shape[1] or h >= area.shape[0]:
            continue
        sized = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        result = cv2.matchTemplate(area, sized, cv2.TM_CCOEFF_NORMED)
        _, score, _, at = cv2.minMaxLoc(result)
        if math.isfinite(score) and score > best_score:
            left, top = sx1 + at[0], sy1 + at[1]
            best_score = score
            best = [left + (x1 - tx1) * scale, top + (y1 - ty1) * scale,
                    left + (x2 - tx1) * scale, top + (y2 - ty1) * scale]
    if best is None or best_score < MIN_SCORE:
        return None
    box = [float(np.clip(best[0], 0, width)), float(np.clip(best[1], 0, height)),
           float(np.clip(best[2], 0, width)), float(np.clip(best[3], 0, height))]
    return box if box[2] - box[0] >= 1 and box[3] - box[1] >= 1 else None
