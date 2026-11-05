"""End-of-limit signs drawn over real speed signs, for the training of the cutter and the classifier.

Switzerland has few end-of-limit signs ("fine limite", family cancellation): on 30 Oct 2026 the dataset had 154
train crops of them, from a handful of physical signs, against thousands of speed limits. Both networks learnt
those few by heart (SignNet boxed 135 of the 154, the classifier read all of them), and missed the ones they had not
seen: of 24 in validation and buffer, SignNet boxed 9 for the car (between 20 and 40 pixels 5 of 10, against 63 of 70
speed limits of that size) and the classifier read 5 sure enough for the car, calling 16 "not a sign". The white
face with a thin black ring has nothing of the red ring the networks learn the signs by.

So a training draws, on some of its real speed signs, an end of limit in their place (paint_end_of_limit): the same
place, size, light and blur of a real sign of the road, with the face of the end-of-limit sign. A round speed limit
(family maximum_speed, classifier label maximum_speed:<v>) becomes the Swiss sign 2.53 (a white face, a thin black
ring, the number in black, a band of thin black stripes from bottom left to top right) or, now and then, the end of
all restrictions (a white face, one broad black stripe); a lit variable display (variable_display) becomes the LED
end of limit of the motorway gantries (a dark panel, a light ring, the number and one slash in light). The colours
come from the pixels of the sign replaced: its bright face and its dark digits by day, the glare of the headlights at
night.

The cutter keeps the box (a sign either way); the classifier's crop becomes "cancellation". Each sign is replaced
with probability SHARE, drawn again every epoch: the networks still see every speed sign as it is most of the time.
Pillow and numpy only (the font is Pillow's own), so the loader workers of both trainings can draw.
"""
from __future__ import annotations

import math
import random
from typing import Any

# The family, and classifier label, of an end-of-limit sign.
CANCELLATION = "cancellation"
# The share of the speed signs drawn as an end of limit, each time a training reads them.
SHARE = 0.25
# Below this side (frame pixels) a sign is left as it is: its stripes would not show.
MIN_SIDE = 9
# What each family becomes; the other families are never replaced, and neither is a box around a disc and its plate of
# hours: the classifier's maximum_speed_hours:<number> has no style here, and a frame says its boxes (plate_boxes).
STYLE_OF_FAMILY = {"maximum_speed": "fixed", "variable_display": "led"}
# Of the round signs replaced, this share becomes the end of all restrictions (no number).
ALL_RESTRICTIONS_SHARE = 0.15
# The numbers of the Swiss end-of-limit signs, when the sign replaced has none.
VALUES = (30, 40, 50, 60, 70, 80, 100, 120)
# The box is grown this much on each side before drawing: a reviewed box is tight, and no red of the ring replaced
# may show around the new face (the metal rim of a real sign is there).
GROW = 0.05
# A variable display whose corners (the box outside its ring) are brighter than this (0-255, their median) has no dark
# panel: it is drawn as a fixed sign.
PANEL_LIGHT = 100
# The blurs (Gaussian sigma, frame pixels) a drawn sign may take to look as sharp as the sign it replaces.
BLURS = (0.3, 0.5, 0.8, 1.2, 1.7, 2.4)
# The size the sign is drawn at before it is shrunk to the frame (pixels of its longer side, at least).
DRAW_SIDE = 192


def style_of(family: str | None) -> str | None:
    """"fixed", "led" or None (not replaced) for a box's family, or for a classifier label (maximum_speed:60)."""
    return STYLE_OF_FAMILY.get(str(family or "").partition(":")[0])


def value_of(label: str | None) -> int | None:
    """The number of a classifier label (maximum_speed:60 or maximum_speed_hours:60:22-6 -> 60), None without one."""
    value = str(label or "").partition(":")[2].partition(":")[0]
    return int(value) if value.isdigit() else None


def summary() -> dict[str, Any]:
    """What a training's result.json records of this augmentation."""
    return {"share": SHARE, "styles": dict(STYLE_OF_FAMILY), "all_restrictions_share": ALL_RESTRICTIONS_SHARE,
            "min_side": MIN_SIDE, "label": CANCELLATION}


def _sharpness(pixels) -> float:
    """How sharp an image region (H x W x 3) is: its Laplacian for its gradient. Over an edge blurred by sigma the
    first sums to the contrast over sigma, the second to the contrast: the ratio does not depend on how many edges the
    region has or how strong they are, only on how blurred they are."""
    grey = pixels.mean(axis=2)
    if min(grey.shape) < 3:
        return 0.0
    inner = grey[1:-1, 1:-1]
    laplacian = abs(4 * inner - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:])
    gradient = abs(grey[2:, 1:-1] - grey[:-2, 1:-1]) + abs(grey[1:-1, 2:] - grey[1:-1, :-2])
    return float(laplacian.mean() / (gradient.mean() + 1.0))


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=max(8, size))


def _number_mask(value: int, width: int, height: int, stroke: int):
    """The digits of value as a mask (mode L) of width x height: filling the height, at most the width, as bold as
    stroke (pixels of a 200-pixel font)."""
    from PIL import Image, ImageDraw

    text = str(value)
    font = _font(200)
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    canvas = Image.new("L", (right - left + 4, bottom - top + 4), 0)
    ImageDraw.Draw(canvas).text((2 - left, 2 - top), text, fill=255, font=font, stroke_width=stroke, stroke_fill=255)
    scale = height / canvas.height
    target_width = min(width, max(1, round(canvas.width * scale)))
    return canvas.resize((target_width, max(1, height)), Image.LANCZOS)


def _stripes(draw, size: int, centre: float, offsets: list[float], thickness: float) -> None:
    """Lines from bottom left to top right ("/") across the canvas, each offset from the centre along the normal."""
    reach = size
    for offset in offsets:
        # The normal of "/" in image pixels (y down) is (1, 1) / sqrt(2).
        nx, ny = offset / math.sqrt(2), offset / math.sqrt(2)
        x0, y0 = centre + nx - reach, centre + ny + reach
        x1, y1 = centre + nx + reach, centre + ny - reach
        draw.line((x0, y0, x1, y1), fill=255, width=max(1, round(thickness)))


def _layers(style: str, value: int | None, rng: random.Random, inner: float):
    """The masks of the sign on a square canvas of DRAW_SIDE (mode L, 255 = there): its outline (alpha), its face,
    its ink (the dark parts) and, round signs, the metal rim; inner is the share of the canvas's half side that is
    the reviewed box (the rest is the grown margin)."""
    from PIL import Image, ImageChops, ImageDraw

    size = DRAW_SIDE
    centre = size / 2
    radius = centre * inner
    alpha = Image.new("L", (size, size), 0)
    face = Image.new("L", (size, size), 0)
    ink = Image.new("L", (size, size), 0)
    rim = Image.new("L", (size, size), 0)

    def disc(image, r, fill=255):
        ImageDraw.Draw(image).ellipse((centre - r, centre - r, centre + r, centre + r), fill=fill)

    if style == "led":
        # The panel fills the box (and its margin); the light parts are the "face", drawn in the LED's colour: a ring,
        # the number and one slash across both, as thick as the ring.
        ImageDraw.Draw(alpha).rectangle((0, 0, size, size), fill=255)
        width = rng.uniform(0.09, 0.13)
        ring = Image.new("L", (size, size), 0)
        disc(ring, radius * 0.93)
        disc(ring, radius * (0.93 - width), 0)
        slash = Image.new("L", (size, size), 0)
        _stripes(ImageDraw.Draw(slash), size, centre, [0.0], radius * width)
        bound = Image.new("L", (size, size), 0)
        disc(bound, radius * (0.93 - width / 2))
        lights = ImageChops.lighter(ring, ImageChops.multiply(slash, bound))
        if value is not None:
            digits = _number_mask(value, round(radius * (1.3 if value >= 100 else 1.0)), round(radius * 0.85), 7)
            lights.paste(255, (round(centre - digits.width / 2), round(centre - digits.height / 2)), digits)
        return alpha, lights, None, None

    disc(alpha, centre)
    disc(rim, centre)
    disc(rim, radius, 0)
    disc(face, radius)
    # The thin black ring, a little inside the edge of the white face.
    outer = rng.uniform(0.93, 0.97)
    inner_ring = outer - rng.uniform(0.035, 0.055)
    ring = Image.new("L", (size, size), 0)
    disc(ring, radius * outer)
    disc(ring, radius * inner_ring, 0)
    inside = Image.new("L", (size, size), 0)
    disc(inside, radius * inner_ring)
    marks = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(marks)
    if value is None:
        # The end of all restrictions: one broad stripe.
        _stripes(draw, size, centre, [0.0], radius * rng.uniform(0.25, 0.35))
    else:
        # Thin stripes in a band through the middle, behind the number.
        count = rng.choice((5, 5, 6))
        spacing = radius * rng.uniform(0.085, 0.105)
        _stripes(draw, size, centre, [(k - (count - 1) / 2) * spacing for k in range(count)],
                 spacing * rng.uniform(0.3, 0.42))
    marks = ImageChops.multiply(marks, inside)
    if value is not None:
        digits = _number_mask(value, round(radius * (1.4 if value >= 100 else 1.1)), round(radius * 0.88), 5)
        marks.paste(255, (round(centre - digits.width / 2), round(centre - digits.height / 2)), digits)
    ink = ImageChops.lighter(ring, marks)
    return alpha, face, ink, rim


def paint_end_of_limit(image, box: list[float], style: str, rng: random.Random, value: int | None = None):
    """A copy of image (PIL, RGB) with an end-of-limit sign of style ("fixed" or "led") drawn over the sign in box
    (x1, y1, x2, y2 in image pixels), coloured from that sign's own pixels; image itself when the box is too small.
    value: the number shown (a fixed sign may still become the end of all restrictions); None draws one of VALUES."""
    import numpy as np
    from PIL import Image, ImageFilter

    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = x2 - x1, y2 - y1
    if min(width, height) < MIN_SIDE:
        return image
    left, top = math.floor(x1 - width * GROW), math.floor(y1 - height * GROW)
    right, bottom = math.ceil(x2 + width * GROW), math.ceil(y2 + height * GROW)
    region_w, region_h = right - left, bottom - top
    original = np.asarray(image.crop((left, top, right, bottom)).convert("RGB"), dtype=np.float32)

    # The colours of the sign replaced, from its middle (inside its ring): its bright face and its dark digits by day,
    # the LED and its panel on a display.
    ys, xs = np.mgrid[0:region_h, 0:region_w]
    cx, cy = (x1 + x2) / 2 - left, (y1 + y2) / 2 - top
    middle = ((xs + 0.5 - cx) / (width / 2)) ** 2 + ((ys + 0.5 - cy) / (height / 2)) ** 2 <= 0.6 ** 2
    pixels = original[middle] if middle.any() else original.reshape(-1, 3)
    light = pixels.mean(axis=1)
    bright = pixels[light >= np.percentile(light, 85)].mean(axis=0)
    dark = pixels[light <= np.percentile(light, 10)].mean(axis=0)
    if bright.mean() - dark.mean() < 40:
        # Too blurred or too dark to tell its face from its digits: a face as bright as its brightest part.
        dark = bright * 0.3
    if style == "led":
        # The panel shows in the corners of the box, outside the ring. A variable sign with bright corners (a sign that
        # turns, a white plate) is drawn as a fixed one.
        inbox = (xs + left >= x1) & (xs + left < x2) & (ys + top >= y1) & (ys + top < y2)
        corners = inbox & (((xs + 0.5 - cx) / (width / 2)) ** 2 + ((ys + 0.5 - cy) / (height / 2)) ** 2 > 1.0)
        corner_pixels = original[corners] if corners.any() else original.reshape(-1, 3)
        corner_light = corner_pixels.mean(axis=1)
        panel = corner_pixels[corner_light <= np.percentile(corner_light, 50)].mean(axis=0)
        if float(np.median(corner_light)) > PANEL_LIGHT:
            style = "fixed"
    if style == "led":
        # The LEDs of an end of limit are white, a little of the display's tint left; the panel black, never brighter
        # than a black panel in daylight.
        bright = 0.3 * bright + 0.7 * max(float(bright.mean()), 180.0)
        dark = 0.3 * panel + 0.7 * float(panel.mean())
        dark = dark * min(1.0, 45.0 / max(float(dark.mean()), 1.0))
    else:
        # The face is white or grey and the ink black, never the red of the ring replaced.
        bright = 0.3 * bright + 0.7 * float(bright.mean())
        dark = 0.15 * dark + 0.85 * float(dark.mean())
    # The metal rim of a round sign: a little darker than its face.
    shade_of_rim = rng.uniform(0.15, 0.4)
    rim_colour = (1 - shade_of_rim) * bright + shade_of_rim * dark

    if value is None:
        value = rng.choice(VALUES)
    if style == "fixed" and rng.random() < ALL_RESTRICTIONS_SHARE:
        value = None
    alpha, face, ink, rim = _layers(style, value, rng, 1 / (1 + 2 * GROW))
    angle = rng.uniform(-4.0, 4.0)
    size = (region_w, region_h)

    def shrink(mask):
        return np.asarray(mask.rotate(angle, resample=Image.BICUBIC).resize(size, Image.LANCZOS),
                          dtype=np.float32) / 255.0

    a, f = shrink(alpha), shrink(face)
    if style == "led":
        # A lit LED glows past its edges.
        glow = np.asarray(Image.fromarray((f * 255).astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(max(0.6, min(width, height) * 0.04))), dtype=np.float32) / 255.0
        f = np.maximum(f, glow * rng.uniform(0.3, 0.7))
        drawn = dark[None, None, :] * (1 - f[..., None]) + bright[None, None, :] * f[..., None]
    else:
        k, r = shrink(ink), shrink(rim)
        drawn = (rim_colour[None, None, :] * r[..., None] + bright[None, None, :] * (f - k).clip(0, 1)[..., None]
                 + dark[None, None, :] * k[..., None])
        drawn = drawn / np.maximum(a, 1e-3)[..., None]
    # Light across the face (the sun, the headlights), the noise of the camera.
    gx, gy = rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)
    shade = 1 + gx * (xs / max(region_w - 1, 1) - 0.5) + gy * (ys / max(region_h - 1, 1) - 0.5)
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(1.0, 3.5), drawn.shape)
    drawn = (drawn * shade[..., None] + noise).clip(0, 255)
    out = original * (1 - a[..., None]) + drawn * a[..., None]
    patch = Image.fromarray(out.round().astype(np.uint8), "RGB")
    # As sharp as the sign replaced (the focus, the motion, the compression of the camera): of a few blurs, the one
    # whose edges are as strong, for their contrast, as the original's.
    target = _sharpness(original)
    blurs = [(abs(_sharpness(np.asarray(candidate, dtype=np.float32)) - target), sigma, candidate)
             for sigma in BLURS for candidate in [patch.filter(ImageFilter.GaussianBlur(sigma))]]
    patch = min(blurs, key=lambda item: item[:2])[2]
    painted = image.copy()
    # The blur reaches past the sign: only the sign and a pixel around it are pasted.
    mask = Image.fromarray((a > 0.02).astype(np.uint8) * 255, "L").filter(ImageFilter.MaxFilter(3))
    painted.paste(patch, (left, top), mask)
    return painted


def paintable(box: list[float], family: str | None) -> bool:
    """A box a training may draw as an end of limit: of a family (or classifier label) with a style, and large enough."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return style_of(family) is not None and min(x2 - x1, y2 - y1) >= MIN_SIDE


def frame_families(frame: dict[str, Any]) -> list[str | None]:
    """The family of each box of a cutter's frame as paintable takes it: None for a box around a disc and its plate."""
    plates = frame.get("plate_boxes") or []
    return [None if box in plates else family for box, family in zip(frame["boxes"], frame.get("box_families") or [])]


def paint_frame(image, boxes: list[list[float]], families: list[str | None], rng: random.Random):
    """image (PIL, RGB) with each paintable box drawn as an end of limit with probability SHARE, and how many were."""
    painted = 0
    for box, family in zip(boxes, families):
        if paintable(box, family) and rng.random() < SHARE:
            image = paint_end_of_limit(image, box, style_of(family), rng)
            painted += 1
    return image, painted
