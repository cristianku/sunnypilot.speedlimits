"""The crop cache of a reader/classifier training (nnslr_tools.crop_cache): a crop read from it is the crop the
training used to cut from the frame, jittered or not, and a cache is cut once and found again."""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from nnslr_tools import crop_cache
from nnslr_tools.crop_cache import EXACT, WINDOW, build_crop_cache
from nnslr_tools.training import _sample_seed, crop_rectangle, jitter_box

SIZE = 32


def _frame(path, width=200, height=120, seed=0):
    """A noisy frame, so that a crop one pixel off differs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _records(boxes, image="frames/a.png"):
    return [{"image_path": image, "image_sha256": None, "bbox_xyxy": box} for box in boxes]


def _reference(root, record, box, scheme="v2"):
    """The crop of the training before the cache: the frame cut, then resized to the input size."""
    with Image.open(root / record["image_path"]) as source:
        image = source.convert("RGB")
    if scheme == "v2":
        region = crop_rectangle(box, image.width, image.height)
    else:
        region = tuple(int(value) for value in box)
    return np.asarray(image.crop(region).resize((SIZE, SIZE), Image.BILINEAR), dtype=int)


def _read(cache, index, box):
    return np.asarray(cache.crop(index, box).resize((SIZE, SIZE), Image.BILINEAR), dtype=int)


# A box in the middle, on each border, partly outside the frame, one pixel wide, and a near sign (larger than
# MAX_BOX_SIDE input sizes: its window is kept smaller).
BOXES = [[60.2, 40.7, 80.9, 61.3], [0.0, 0.0, 12.5, 9.0], [185.4, 100.1, 200.0, 120.0], [-5.0, 50.0, 10.0, 70.0],
         [100.0, 30.0, 101.0, 31.0], [20.0, 5.0, 150.0, 115.0]]


@pytest.mark.parametrize("workers", [0, 2])
def test_a_cached_crop_is_the_crop_of_the_frame(tmp_path, workers):
    _frame(tmp_path / "frames" / "a.png")
    records = _records(BOXES) * 2
    kinds = [WINDOW] * len(BOXES) + [EXACT] * len(BOXES)
    cache, summary = build_crop_cache(tmp_path, records, kinds, size=SIZE, scheme="v2", content={"dataset": "x"},
                                      workers=workers)
    assert summary["built"] and summary["crops"] == 12 and summary["windows"] == 6 and summary["frames"] == 1
    # Validation and the crops without jitter: exactly the old crop.
    for index in range(len(BOXES), len(records)):
        assert (np.asarray(cache.crop(index, None), dtype=int) == _reference(tmp_path, records[index],
                                                                             records[index]["bbox_xyxy"])).all()
    # The jittered crops: the same pixels for a window at the frame's resolution, very close for the near sign.
    for epoch in range(25):
        for index, box in enumerate(BOXES):
            moved = jitter_box(box, random.Random(_sample_seed("box", 0, epoch, index)))
            difference = np.abs(_read(cache, index, moved) - _reference(tmp_path, records[index], moved))
            if index == len(BOXES) - 1:
                assert difference.mean() < 40, (epoch, difference.mean())
            else:
                assert difference.max() == 0, (epoch, index, box, moved)
    # The near sign's window is smaller than the frame's pixels; the others keep them.
    assert cache.index[len(BOXES) - 1, 1] < 120 and cache.index[0, 2] == cache.index[0, 5] - cache.index[0, 3]


def test_the_v1_reader_keeps_its_integer_box(tmp_path):
    _frame(tmp_path / "frames" / "a.png")
    records = _records([[10.7, 10.2, 40.9, 30.5], [190.0, 100.0, 215.0, 130.0]])
    cache, _ = build_crop_cache(tmp_path, records, [EXACT, EXACT], size=SIZE, scheme="v1", content={})
    for index, record in enumerate(records):
        assert (np.asarray(cache.crop(index), dtype=int) == _reference(tmp_path, record, record["bbox_xyxy"],
                                                                         scheme="v1")).all()


def test_a_cache_is_cut_once_and_a_change_makes_another(tmp_path):
    _frame(tmp_path / "frames" / "a.png")
    _frame(tmp_path / "frames" / "b.png", seed=1)
    records = _records(BOXES[:2]) + _records(BOXES[:1], image="frames/b.png")
    kinds = [WINDOW, EXACT, WINDOW]
    first, summary = build_crop_cache(tmp_path, records, kinds, size=SIZE, scheme="v2", content={"dataset": "x"})
    assert summary["built"] and summary["frames"] == 2
    again, summary = build_crop_cache(tmp_path, records, kinds, size=SIZE, scheme="v2", content={"dataset": "x"})
    assert not summary["built"] and again.folder == first.folder
    assert (tmp_path / crop_cache.CACHE_FOLDER).is_dir() and len(list((tmp_path / crop_cache.CACHE_FOLDER).iterdir())) == 1
    # Another dataset, box, storage or input size is another cache.
    for change in ({"content": {"dataset": "y"}}, {"kinds": [EXACT, EXACT, WINDOW]}, {"size": SIZE + 1},
                   {"records": _records(BOXES[1:3]) + records[2:]}):
        arguments = {"records": records, "kinds": kinds, "size": SIZE, "content": {"dataset": "x"}, **change}
        other, summary = build_crop_cache(tmp_path, arguments.pop("records"), arguments.pop("kinds"), scheme="v2",
                                          **arguments)
        assert summary["built"] and other.folder != first.folder
    with pytest.raises(ValueError, match="one storage kind per record"):
        build_crop_cache(tmp_path, records, kinds[:2], size=SIZE, scheme="v2", content={})


def test_the_window_holds_the_largest_jitter():
    # Every jittered crop of a box is inside its window, whatever the draw.
    width, height = 1928, 1208
    for box in ([500.0, 300.0, 540.0, 344.0], [3.0, 3.0, 9.0, 8.0], [1900.0, 1180.0, 1928.0, 1208.0]):
        left, top, right, bottom = crop_cache._window(box, width, height)
        for seed in range(2000):
            x1, y1, x2, y2 = crop_rectangle(jitter_box(box, random.Random(seed)), width, height)
            assert left <= x1 and top <= y1 and x2 <= right and y2 <= bottom, (box, seed)


# ---------- the cuts Sign review shares with the training (issue #27) ----------

def _cut_before(task):
    """crop_cache._cut as it was before stored_crop was taken out of it for Sign review: the training's stored crops
    must not change by a byte."""
    from pathlib import Path

    root, image_path, size, scheme, items = task
    rectangle = crop_cache.rectangle_of(scheme)
    image, factor = crop_cache.crop_source(crop_cache._safe_image_path(Path(root), image_path))
    width, height = image.width // factor, image.height // factor
    out = []
    for index, kind, box in items:
        if kind == EXACT:
            region = rectangle(box, width, height)
            pixels = image.crop(tuple(value * factor for value in region)).resize((size, size), Image.BILINEAR)
        else:
            region = crop_cache._window(box, width, height)
            cut = tuple(value * factor for value in region)
            side_x = max(abs(float(box[2]) - float(box[0])), 1.0) * factor
            side_y = max(abs(float(box[3]) - float(box[1])), 1.0) * factor
            keep_x = min(1.0, crop_cache.MAX_BOX_SIDE * size / side_x)
            keep_y = min(1.0, crop_cache.MAX_BOX_SIDE * size / side_y)
            span_x, span_y = cut[2] - cut[0], cut[3] - cut[1]
            if keep_x == 1.0 and keep_y == 1.0:
                pixels = image.crop(cut)
            else:
                pixels = image.resize((max(1, round(span_x * keep_x)), max(1, round(span_y * keep_y))),
                                      Image.BILINEAR, box=cut)
        array = np.ascontiguousarray(np.asarray(pixels, dtype=np.uint8))
        out.append((index, kind, array.tobytes(), array.shape[0], array.shape[1], region, width, height))
    return out


def _crop_before(cache, index, box=None, paint=None):
    """CropCache.crop as it was before window_crop was taken out of it for Sign review."""
    row = cache.index[index]
    image = Image.fromarray(cache.array(index))
    if int(row[9]) == EXACT:
        return image
    left, top, right, bottom = (int(value) for value in row[3:7])
    frame_width, frame_height = int(row[7]), int(row[8])
    if paint is not None:
        to_x, to_y = image.width / (right - left), image.height / (bottom - top)
        image = paint(image, lambda frame_box: [(frame_box[0] - left) * to_x, (frame_box[1] - top) * to_y,
                                                (frame_box[2] - left) * to_x, (frame_box[3] - top) * to_y])
    x1, y1, x2, y2 = crop_rectangle(box, frame_width, frame_height)
    x1, x2 = max(left, min(x1, right - 1)), max(left + 1, min(x2, right))
    y1, y2 = max(top, min(y1, bottom - 1)), max(top + 1, min(y2, bottom))
    if image.width == right - left and image.height == bottom - top:
        return image.crop((x1 - left, y1 - top, x2 - left, y2 - top))
    scale_x, scale_y = image.width / (right - left), image.height / (bottom - top)
    return image.resize((cache.size, cache.size), Image.BILINEAR,
                        box=((x1 - left) * scale_x, (y1 - top) * scale_y, (x2 - left) * scale_x, (y2 - top) * scale_y))


def _full_frame(root):
    """A picture of a drive at full definition (1344 x 760) and its frame (2688 x 1520) beside it, both noisy."""
    from nnslr_tools.media import FRAME_SIZE, FULL_SIZE, full_picture

    _frame(root / "frames" / "f.png", *FRAME_SIZE, seed=2)
    pixels = np.random.default_rng(3).integers(0, 256, (FULL_SIZE[1], FULL_SIZE[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(full_picture(root / "frames" / "f.png"), quality=90)


# On the full-definition frame: a far sign, a near one (its window scaled down), one on the right border.
FULL_BOXES = [[600.3, 300.2, 640.8, 345.1], [100.0, 100.0, 600.0, 700.0], [1330.2, 740.0, 1344.0, 760.0]]


def _outlined(box):
    """A paint of the synthetic signs' kind: the box outlined on the window."""
    from PIL import ImageDraw

    def paint(window, to_window):
        drawn = window.copy()
        ImageDraw.Draw(drawn).rectangle(to_window(box), outline=(255, 0, 0), width=2)
        return drawn
    return paint


def test_the_cuts_taken_out_for_sign_review_leave_the_training_crops_as_they_were(tmp_path):
    _frame(tmp_path / "frames" / "a.png")
    _full_frame(tmp_path)
    for image, boxes, schemes in (("frames/a.png", BOXES, ("v2", "v1")), ("frames/f.png", FULL_BOXES, ("v2",))):
        items = [(index, kind, box) for index, (kind, box) in
                 enumerate((kind, box) for kind in (EXACT, WINDOW) for box in boxes)]
        for scheme in schemes:
            if scheme == "v1":
                items = [(index, kind, box) for index, kind, box in items if kind == EXACT and box[0] >= 0]
            task = (str(tmp_path), image, SIZE, scheme, items)
            assert crop_cache._cut(task) == _cut_before(task), (image, scheme)
        records = _records(boxes, image=image) * 2
        cache, _ = build_crop_cache(tmp_path, records, [EXACT] * len(boxes) + [WINDOW] * len(boxes), size=SIZE,
                                    scheme="v2", content={"dataset": image})
        for index, record in enumerate(records):
            for epoch in range(6):
                moved = jitter_box(record["bbox_xyxy"], random.Random(_sample_seed("box", 0, epoch, index)))
                for paint in (None, _outlined(record["bbox_xyxy"])):
                    if paint is not None and index < len(boxes):
                        continue
                    assert np.array_equal(np.asarray(cache.crop(index, moved, paint)),
                                          np.asarray(_crop_before(cache, index, moved, paint))), (image, index, epoch)


# The first epochs of record 0 of a run of seed 0 whose draw moves the box (epochs 0 and 3 keep it as drawn).
MOVED_EPOCHS = [1, 2, 4, 5]


def test_sign_review_sees_the_crops_the_training_cuts(tmp_path):
    from nnslr_tools.training import crop_source

    _frame(tmp_path / "frames" / "a.png")
    _full_frame(tmp_path)
    for image, boxes in (("frames/a.png", BOXES), ("frames/f.png", FULL_BOXES)):
        source, factor = crop_source(tmp_path / image)
        assert factor == (2 if image == "frames/f.png" else 1)
        records = _records(boxes, image=image) * 2
        cache, _ = build_crop_cache(tmp_path, records, [EXACT] * len(boxes) + [WINDOW] * len(boxes), size=SIZE,
                                    scheme="v2", content={"dataset": image})
        for index, box in enumerate(boxes):
            crops = crop_cache.review_crops(source, factor, box, size=SIZE, samples=4)
            assert len(crops) == 5 and all(crop.size == (SIZE, SIZE) for crop in crops)
            # First the crop of validation, then the box jittered in the first epochs of record 0 of a run of seed 0
            # that move it, cut from the window and resized as the dataset resizes it.
            assert np.array_equal(np.asarray(crops[0]), np.asarray(cache.crop(index)))
            for epoch, crop in zip(MOVED_EPOCHS, crops[1:]):
                moved = jitter_box(box, random.Random(_sample_seed("box", 0, epoch, 0)))
                assert moved != box
                expected = cache.crop(len(boxes) + index, moved).resize((SIZE, SIZE), Image.BILINEAR)
                assert np.array_equal(np.asarray(crop), np.asarray(expected)), (image, box, epoch)
        assert crop_cache.review_crops(source, factor, boxes[0], size=SIZE) == crop_cache.review_crops(
            source, factor, boxes[0], size=SIZE, samples=0)
    # Epochs 0 and 3 keep the box as drawn: the samples skip them, the crop without jitter shows it.
    kept = [jitter_box(BOXES[0], random.Random(_sample_seed("box", 0, epoch, 0))) == BOXES[0] for epoch in range(6)]
    assert kept == [True, False, False, True, False, False]
    assert crop_cache.moved_boxes(BOXES[0], 4) == [jitter_box(BOXES[0], random.Random(_sample_seed("box", 0, epoch, 0)))
                                                   for epoch in MOVED_EPOCHS]
