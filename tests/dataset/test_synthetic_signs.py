"""End-of-limit signs drawn over real speed signs for the trainings (nnslr_tools.synthetic_signs): the sign is
replaced in its own place and nowhere else, with the colours of a Swiss end of limit, the same every time for the
same draw; the crop cache draws on a window before the crop is cut; a classifier training labels those crops
cancellation."""
from __future__ import annotations

import json
import random
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw

import test_classifier
from nnslr_tools import synthetic_signs
from nnslr_tools.crop_cache import WINDOW, build_crop_cache
from nnslr_tools.synthetic_signs import paint_end_of_limit, paint_frame, paintable
from nnslr_tools.training import prepare_reader_training
from test_classifier import _box, _classifier_fixture
from test_detector_sizes import _run, torch_python  # noqa: F401 (fixture)

BOX = [80.0, 30.0, 120.0, 70.0]


def _frame(sign: str = "round", size=(200, 120)) -> Image.Image:
    """A frame of green-grey noise with one sign in BOX: a round speed limit (a white face, a red ring, black
    digits), a lit display (a black panel, a red ring and orange digits), or a variable sign on a white plate."""
    pixels = np.random.default_rng(0).integers(60, 120, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = BOX
    if sign == "display":
        draw.rectangle(BOX, fill=(12, 12, 14))
        draw.ellipse([x1 + 2, y1 + 2, x2 - 2, y2 - 2], outline=(240, 40, 30), width=4)
        draw.text((x1 + 12, y1 + 14), "80", fill=(250, 170, 60))
    else:
        if sign == "plate":
            draw.rectangle(BOX, fill=(235, 235, 235))
        draw.ellipse(BOX, fill=(230, 230, 225), outline=(200, 20, 25), width=6)
        draw.text((x1 + 12, y1 + 14), "60", fill=(10, 10, 10))
    return image


def _red(pixels: np.ndarray) -> int:
    pixels = pixels.astype(int)
    return int(((pixels[..., 0] - np.maximum(pixels[..., 1], pixels[..., 2])) > 60).sum())


def test_a_speed_sign_becomes_an_end_of_limit_in_its_place_only():
    image = _frame()
    before = np.asarray(image).copy()
    painted = paint_end_of_limit(image, BOX, "fixed", random.Random(3), 60)
    after = np.asarray(painted)
    # The frame given is not touched; the same draw gives the same pixels.
    assert (np.asarray(image) == before).all()
    assert (np.asarray(paint_end_of_limit(image, BOX, "fixed", random.Random(3), 60)) == after).all()
    assert not (np.asarray(paint_end_of_limit(image, BOX, "fixed", random.Random(4), 60)) == after).all()
    # Nothing changes outside the box grown by GROW and the pixel of the paste's edge.
    grow = synthetic_signs.GROW * (BOX[2] - BOX[0]) + 2
    outside = np.ones(after.shape[:2], dtype=bool)
    outside[int(BOX[1] - grow):int(BOX[3] + grow) + 1, int(BOX[0] - grow):int(BOX[2] + grow) + 1] = False
    assert (after[outside] == before[outside]).all()
    # The red ring is gone; the face is still light and has dark ink (ring, stripes, number) on it.
    inside = after[int(BOX[1]):int(BOX[3]), int(BOX[0]):int(BOX[2])]
    assert _red(before[int(BOX[1]):int(BOX[3]), int(BOX[0]):int(BOX[2])]) > 100 and _red(inside) == 0
    middle = inside[8:32, 8:32].mean(axis=2)
    assert np.percentile(middle, 80) > 170 and np.percentile(middle, 80) - np.percentile(middle, 10) > 100


def test_a_lit_display_keeps_its_black_panel_and_shows_white_lights():
    image = _frame("display")
    painted = np.asarray(paint_end_of_limit(image, BOX, "led", random.Random(1), 80)).astype(int)
    inside = painted[int(BOX[1]):int(BOX[3]), int(BOX[0]):int(BOX[2])]
    light = inside.mean(axis=2)
    # The corners are the panel: dark. The lights are white, not the red or orange of the speed limit.
    assert inside[:3, :3].mean() < 60 and inside[-3:, -3:].mean() < 60
    lit = inside[light > np.percentile(light, 95)]
    assert lit.mean() > 150 and (lit.max(axis=1) - lit.min(axis=1)).mean() < 50 and _red(inside) == 0


def test_a_variable_sign_on_a_white_plate_is_drawn_as_a_fixed_sign():
    painted = np.asarray(paint_end_of_limit(_frame("plate"), BOX, "led", random.Random(1), 80))
    middle = painted[int(BOX[1]) + 8:int(BOX[3]) - 8, int(BOX[0]) + 8:int(BOX[2]) - 8].mean(axis=2)
    # A white face, not a black panel.
    assert np.median(middle) > 150


def test_only_speed_signs_large_enough_are_drawn():
    assert paintable(BOX, "maximum_speed") and paintable(BOX, "maximum_speed:60") and paintable(BOX, "variable_display:unknown")
    assert not paintable(BOX, "cancellation") and not paintable(BOX, "other_sign:give_way") and not paintable(BOX, None)
    small = [10.0, 10.0, 10.0 + synthetic_signs.MIN_SIDE - 0.5, 30.0]
    assert not paintable(small, "maximum_speed")
    image = _frame()
    assert paint_end_of_limit(image, small, "fixed", random.Random(0)) is image
    assert synthetic_signs.style_of("maximum_speed:60") == "fixed" and synthetic_signs.style_of("variable_display") == "led"
    assert synthetic_signs.value_of("maximum_speed:60") == 60 and synthetic_signs.value_of("maximum_speed:unknown") is None


def test_a_frame_has_about_a_share_of_its_speed_signs_drawn():
    image = _frame()
    rng = random.Random(0)
    counts = [paint_frame(image, [BOX, BOX], ["maximum_speed", "other_sign"], rng)[1] for _ in range(200)]
    assert set(counts) <= {0, 1}
    assert 0.17 < sum(counts) / len(counts) < 0.33


def test_the_crop_cache_draws_on_the_window_before_the_crop_is_cut(tmp_path):
    path = tmp_path / "frames" / "a.png"
    path.parent.mkdir(parents=True)
    _frame().save(path)
    # A sign at the frame's resolution, and a near one whose window is kept smaller (MAX_BOX_SIDE input sizes).
    boxes = [BOX, [20.0, 5.0, 150.0, 115.0]]
    records = [{"image_path": "frames/a.png", "image_sha256": None, "bbox_xyxy": box} for box in boxes]
    cache, _ = build_crop_cache(tmp_path, records, [WINDOW, WINDOW], size=32, scheme="v2", content={})
    for index, box in enumerate(boxes):
        seen = []

        def paint(window, to_window):
            seen.append((window.size, to_window(box)))
            filled = window.copy()
            ImageDraw.Draw(filled).rectangle(to_window(box), fill=(0, 0, 255))
            return filled
        crop = np.asarray(cache.crop(index, box, paint)).astype(int)
        (width, height), mapped = seen[0]
        left, top, right, bottom = (int(value) for value in cache.index[index, 3:7])
        scale_x, scale_y = width / (right - left), height / (bottom - top)
        assert mapped == pytest.approx([(box[0] - left) * scale_x, (box[1] - top) * scale_y,
                                        (box[2] - left) * scale_x, (box[3] - top) * scale_y])
        # The crop of the box is what was drawn there.
        centre = crop[crop.shape[0] // 4:3 * crop.shape[0] // 4, crop.shape[1] // 4:3 * crop.shape[1] // 4]
        assert (np.abs(centre - [0, 0, 255]) <= 2).all()
    # Without a paint, the window is as it was cut.
    assert _red(np.asarray(cache.crop(0, BOX))) > 50


TRAIN_WITH_EVERY_SIGN_DRAWN = r"""
import sys
from nnslr_tools import cli, synthetic_signs
synthetic_signs.SHARE = 1.0
raise SystemExit(cli.main(sys.argv[1:]))
"""


def test_tiny_cpu_classifier_learns_drawn_end_of_limit_signs(tmp_path, capsys, monkeypatch, torch_python):
    """A classifier whose plan has the class cancellation draws it over the speed signs of train (every one here)."""
    frames = list(test_classifier.FRAMES)
    frames.insert(5, ("train", "human", [_box("c", test_classifier.SIGN, "cancellation")], []))
    monkeypatch.setattr(test_classifier, "FRAMES", frames)
    dataset = _classifier_fixture(tmp_path, capsys)
    plan = prepare_reader_training(dataset, tmp_path / "splits" / "latest.json", tmp_path, class_scheme="v2")
    assert "cancellation" in plan["classes"]
    output = tmp_path / "runs" / "tiny" / "classifier"
    process = _run(torch_python, "-c", TRAIN_WITH_EVERY_SIGN_DRAWN, "train", str(dataset), "--task", "reader",
                   "--classes", "v2", "--data-root", str(tmp_path), "--device", "cpu", "--num-workers", "0",
                   "--output", str(output), "--epochs", "2", "--batch-size", "4", "--input-size", "32", "--no-amp")
    assert process.returncode == 0, process.stderr[-3000:]
    # The five speed signs of train (the cancellation and the other signs are never drawn over).
    assert re.search(r"^classifier draws an end of limit over 100% of the 5 speed signs of train each epoch",
                     process.stdout, re.M), process.stdout
    drawn = json.loads((output / "result.json").read_text())["augmentation"]["synthetic_end_of_limit"]
    assert drawn["paintable_train_crops"] == 5 and drawn["label"] == "cancellation" and drawn["share"] == 1.0


def test_a_sign_with_its_plate_of_hours_is_never_drawn_over():
    assert synthetic_signs.style_of("maximum_speed_hours:60") is None
    assert synthetic_signs.value_of("maximum_speed_hours:60") == 60
    assert not paintable(BOX, "maximum_speed_hours:60")
    frame = {"boxes": [BOX, [10.0, 10.0, 40.0, 40.0]], "box_families": ["maximum_speed", "maximum_speed"],
             "plate_boxes": [[10.0, 10.0, 40.0, 40.0]]}
    assert synthetic_signs.frame_families(frame) == ["maximum_speed", None]
    assert synthetic_signs.frame_families({"boxes": [BOX], "box_families": ["maximum_speed"]}) == ["maximum_speed"]
    assert synthetic_signs.frame_families({"boxes": [BOX]}) == []
