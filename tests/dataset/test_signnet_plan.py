"""What SignNet takes from a plan without torch (nnslr_tools.signnet): the band's boxes and ignore boxes, which train
frames get synthetic signs, the run a start comes from, whether a network learnt from MTSD, and the plans
it refuses."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from nnslr_tools.signnet import (BAND_WIDTH, band_boxes, learnt_mapillary, paintable_train_boxes, paints,
                                 signnet_not_trainable, start_run, train_signnet)


def test_an_ignore_box_that_reaches_into_the_band_is_cut_to_it():
    # A tall object whose centre is below the band (a window's plate under its sign): a box of the band it is not,
    # an ignore box it stays, its part inside the band.
    assert band_boxes([[10, 500, 50, 800]], 1600, 896) == []
    (cut,) = band_boxes([[10, 500, 50, 800]], 1600, 896, ignore=True)
    assert cut == pytest.approx([5.6, 280.0, 28.0, 288.0])
    assert band_boxes([[10, 900, 50, 1000]], 1600, 896, ignore=True) == []
    assert band_boxes([[200, 100, 240, 140]], BAND_WIDTH, ignore=True) == band_boxes([[200, 100, 240, 140]], BAND_WIDTH)
    assert "torch" not in sys.modules


def test_only_our_train_frames_get_synthetic_signs():
    ours = {"boxes": [[0, 0, 40, 40]], "box_families": ["maximum_speed"], "source": None}
    window = {**ours, "source": "mapillary"}
    assert paints(ours, True) and not paints(ours, False)
    assert not paints(window, True)
    assert paintable_train_boxes([ours, window, {**ours, "box_families": ["other_sign"]}]) == 1
    # A box around a disc and its plate of hours: an end of limit over it would leave no plate and teach a wrong box.
    assert paintable_train_boxes([{**ours, "plate_boxes": [[0, 0, 40, 40]]}]) == 0


def test_the_run_of_a_start():
    assert start_run(Path("data/models/signnet/runs/r7/signnet-best.pt")) == ("r7", False)
    assert start_run(Path("data/models/signnet/runs/r7/pretrain/signnet-best.pt")) == ("r7", True)
    assert start_run(Path("pretrain/signnet-best.pt")) == ("pretrain", False)


def test_whether_a_network_learnt_from_mapillary():
    assert learnt_mapillary({"mapillary": {"frames": 3}}, None) and learnt_mapillary({"mapillary_only": True}, None)
    assert not learnt_mapillary({"mapillary": None}, None) and not learnt_mapillary({}, {"mapillary": None})
    # From its start: a Mapillary stage (a checkpoint of before uses_mapillary), or weights that went on from one.
    assert learnt_mapillary({}, {"mapillary_only": True}) and learnt_mapillary({}, {"uses_mapillary": True})


def test_a_plan_with_fewer_train_frames_than_one_batch_is_refused_before_torch(tmp_path):
    frames = [{"split": "train"}, {"split": "validation"}]
    plan = {"frames": frames, "trainable": True, "families": "all", "counts_by_split": {"train": 0},
            "boxes_by_split": {"train": 0}, "mapillary": {"frames": 1},
            "trained_on": {"ours": 0, "ours_boxes": 0, "mapillary_windows": 1, "mapillary_boxes": 1}}
    with pytest.raises(ValueError, match="1 train frames, fewer than one batch of 2"):
        train_signnet(plan, tmp_path, tmp_path / "never", batch=2)
    assert not (tmp_path / "never").exists() and "torch" not in sys.modules
    # Its reason names what train holds, not only our frames.
    assert "train holds {'ours': 0" in signnet_not_trainable(plan)
    assert "train holds" not in signnet_not_trainable({**plan, "mapillary": None})
