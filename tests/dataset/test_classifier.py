"""The classifier of the pipeline (scripts/run_classifier.sh): the v2 labels (not_a_sign, the kinds of other signs
folded when rare), the removed proposals as negatives, the frames and their review tiers, the crops and their box
jitter, the selection of mined negatives, the chain's counts and the comparison with the old reader, without
torch; then, when a Python with torch is available (this one or the GPU environment of this checkout), a tiny CPU
run of the whole chain on synthetic frames: a random cutter mines negatives, the classifier trains with them, is
evaluated alone and end to end, with the old reader next to it. The torch work runs in a subprocess, as in
test_detector_sizes.py: torch never enters this interpreter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import zlib

import pytest

from nnslr_tools import cli
from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.chain import (ChainTally, classifier_comparison, frame_candidates, resolve_model,
                               select_negatives, truth_objects)
from nnslr_tools.evaluation import PER_FRAME_MAX_BOXES, PER_FRAME_STEPS, classifier_metrics
from nnslr_tools.training import (MIN_KIND_TRAIN_CROPS, TrainConfig, _classifier_label, _reader_label,
                                  crop_rectangle, family_label, jitter_box, prepare_reader_training,
                                  read_negatives, resume_arguments, review_tier, reviewed_frames)
from test_annotations import invoke, reviewed, write
from test_detector_sizes import _config_of, _run, torch_python  # noqa: F401 (fixture)

REPO = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 64, 48
BEFORE_RULE = "2026-10-23T12:00:00Z"
AFTER_RULE = "2026-10-25T08:00:00Z"


def frame_png(path: Path, seed: int) -> None:
    """A 64 x 48 frame of deterministic noise: every frame its own image (and sha256)."""
    rng = random.Random(seed)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + bytes(rng.randrange(256) for _ in range(WIDTH * 3)) for _ in range(HEIGHT))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _box(detection_id, box, family="maximum_speed", value=None, sign_type=None):
    item = dict(detection_id=detection_id, bbox_xyxy=box, family=family, value_kph=value, value_state="reviewed",
                score=.8, review_state="reviewed", applicability="unresolved", supplementary_panel="unknown")
    if sign_type is not None:
        item["sign_type"] = sign_type
    return item


# frame -> partition, who reviewed it, its kept boxes, its removed proposals (detection id, box).
SIGN = [4, 6, 20, 26]
FRAMES = [
    ("train", "ai", [_box("s", SIGN, value=30), _box("o", [40, 4, 52, 16], "other_sign", sign_type="give_way")],
     [("dup", [5, 6, 20, 27])]),
    ("train", "ai", [_box("s", SIGN, value=50), _box("o", [40, 4, 52, 16], "other_sign", sign_type="give_way")],
     [("free", [30, 30, 44, 44])]),
    ("train", "ai-late", [_box("s", SIGN, value=30), _box("o", [40, 4, 52, 16], "other_sign", sign_type="give_way")], []),
    ("train", "human", [_box("s", SIGN, value=50), _box("n", [44, 30, 58, 44], "not_a_sign"),
                        _box("o", [30, 4, 40, 14], "other_sign", sign_type="town_entry")], []),
    ("train", "human", [_box("s", SIGN, value=30)], []),
    ("train", "human", [], []),
    ("validation", "ai", [_box("s", SIGN, value=30), _box("o", [40, 4, 52, 16], "other_sign", sign_type="give_way")],
     [("free", [30, 30, 44, 44])]),
    ("validation", "human", [_box("s", SIGN, value=50)], []),
    ("validation", "human", [_box("s", SIGN, value=30), _box("o", [40, 4, 52, 16], "other_sign", sign_type="town_entry")], []),
    ("test", "human", [_box("s", SIGN, value=50)], [("free", [30, 30, 44, 44])]),
    ("test", "human", [_box("s", [2, 2, 36, 40], value=30), _box("o", [40, 4, 52, 16], "other_sign")], []),
    ("buffer", "human", [_box("s", SIGN, value=30)], [("free", [30, 30, 44, 44])]),
]


def _classifier_fixture(root: Path, capsys) -> Path:
    """FRAMES as reviewed frames on their own routes, imported, with a frozen split by route."""
    rows, partition = [], {}
    for index, (split, who, kept, removed) in enumerate(FRAMES):
        route = f"{index + 1:08x}--abcdef1234"
        row = reviewed(root, route, index=index)
        frame_png(root / row["image_path"], index)
        row.update(width=WIDTH, height=HEIGHT,
                   frame_sha256=hashlib.sha256((root / row["image_path"]).read_bytes()).hexdigest(),
                   detections=[dict(item) for item in kept],
                   source_proposals=[dict(item) for item in kept] + [_box(name, box) for name, box in removed])
        if who.startswith("ai"):
            row.update(reviewer="ai/Claude", label_basis="ai_review", reviewer_kind="ai", review_channel="rest",
                       review_timestamp=AFTER_RULE if who == "ai-late" else BEFORE_RULE)
        rows.append(row)
        partition[route] = split
    code, out = invoke(capsys, "import-annotations", str(write(root, rows)), "--data-root", str(root))
    assert code == 0, out.err + out.out
    dataset = root / "annotations" / "objects.jsonl"
    canonical = load_jsonl(dataset)
    report = validate_dataset(canonical, root)
    assert report["valid"], report["errors"]
    payload = {"schema_version": 1, "dataset_kind": "gold", "dataset_sha256": report["dataset_sha256"],
               "assignments": {row["annotation_id"]: partition[row["route_id"]] for row in canonical
                               if row["review_state"] in ("accepted", "corrected")}}
    frozen = root / "splits" / "fixture.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (root / "splits" / "latest.json").write_text(json.dumps({"path": "splits/fixture.json", "sha256": digest}) + "\n",
                                                 encoding="utf-8")
    return dataset


# ---------------------------------------------------------------- without torch

def _row(family, value=None, state="accepted", sign_type=None, value_state=None):
    return {"review_state": state, "bbox_xyxy": [0, 0, 10, 10], "sign_family": family, "value_kph": value,
            "value_state": value_state or ("read" if value is not None else "unknown"), "sign_type": sign_type}


def test_v2_labels_add_kinds_and_not_a_sign_and_keep_the_rest():
    assert _classifier_label(_row("maximum_speed", 60)) == _reader_label(_row("maximum_speed", 60)) == "maximum_speed:60"
    assert _classifier_label(_row("maximum_speed")) == "maximum_speed:unknown"
    assert _classifier_label(_row("cancellation")) == "cancellation"
    assert _classifier_label(_row("other_sign", sign_type="give_way")) == "other_sign:give_way"
    assert _classifier_label(_row("other_sign")) == _reader_label(_row("other_sign")) == "other_sign"
    assert _classifier_label(_row("other_sign", sign_type="no_such_kind")) == "other_sign"
    assert _classifier_label(_row("not_a_sign")) == "not_a_sign" and _reader_label(_row("not_a_sign")) is None
    assert _classifier_label(_row("maximum_speed", 60, state="rejected")) is None
    assert _classifier_label({**_row("not_a_sign"), "bbox_xyxy": None}) is None
    assert family_label("other_sign:give_way") == "other_sign" and family_label("maximum_speed:60") == "maximum_speed:60"


def test_a_label_only_in_validation_is_left_out_not_a_stop():
    from collections import Counter
    from nnslr_tools.training import _without_unseen_validation
    records = [{"split": "train", "label": "maximum_speed:40"}, {"split": "validation", "label": "maximum_speed:40"},
               {"split": "validation", "label": "maximum_speed:45"}, {"split": "test", "label": "maximum_speed:45"}]
    kept, skipped = _without_unseen_validation(records, ["maximum_speed:45"], Counter())
    assert kept == [records[0], records[1], records[3]] and skipped == {"validation_label_not_in_train": 1}
    assert _without_unseen_validation(records, [], Counter()) == (records, Counter())


def test_a_training_can_leave_out_the_frames_of_some_reviewers(monkeypatch):
    from collections import Counter
    from nnslr_tools.training import TRAIN_WITHOUT_ENV, _without_reviewers
    rows = [{"image_path": "a.png", "reviewer": "ai/llama-v100/qwen"}, {"image_path": "b.png", "reviewer": "cristianku"}]
    records = [{"split": "train", "image_path": "a.png"}, {"split": "train", "image_path": "b.png"},
               {"split": "validation", "image_path": "a.png"}, {"split": "test", "image_path": "a.png"}]
    monkeypatch.delenv(TRAIN_WITHOUT_ENV, raising=False)
    assert _without_reviewers(records, rows, Counter()) == (records, [])
    monkeypatch.setenv(TRAIN_WITHOUT_ENV, "ai/llama, ai/strata")
    skipped = Counter()
    kept, without = _without_reviewers(records, rows, skipped)
    # Only the train crops of their frames go: validation and test stay whole.
    assert kept == records[1:] and without == ["ai/llama", "ai/strata"]
    assert skipped == {"train_frames_of_reviewers_left_out": 1}


def test_review_tiers_follow_the_other_sign_rule():
    assert review_tier("human_review", BEFORE_RULE) == "human"
    assert review_tier("ai_review", BEFORE_RULE) == "ai_before_rule"
    assert review_tier("ai_review", "2026-10-24T15:33:17Z") == "ai_after_rule"
    assert review_tier("ai_review", "2026-10-24T15:33:16.999Z") == "ai_before_rule"
    assert review_tier("ai_review", None) == "ai_before_rule"
    assert review_tier("mixed", AFTER_RULE) == "other"


def test_crop_rectangle_rounds_outwards_stays_inside_and_is_never_empty():
    assert crop_rectangle([4.2, 6.7, 20.1, 26.0], 64, 48) == (4, 6, 21, 26)
    assert crop_rectangle([-5, -3, 70, 60], 64, 48) == (0, 0, 64, 48)
    left, top, right, bottom = crop_rectangle([30.4, 10, 30.6, 10], 64, 48)
    assert right - left == 2 and bottom - top == 2 and left <= 30.4 <= right
    assert crop_rectangle([63.9, 47.9, 64.5, 48.5], 64, 48) == (62, 46, 64, 48)


def test_box_jitter_is_seeded_and_stays_near_the_box():
    box = [10.0, 10.0, 30.0, 50.0]
    first = [jitter_box(box, random.Random(f"box {index}")) for index in range(200)]
    assert first == [jitter_box(box, random.Random(f"box {index}")) for index in range(200)]
    changed = [item for item in first if item != box]
    assert 120 < len(changed) < 200
    for x1, y1, x2, y2 in changed:
        assert 0.85 * 20 - 1e-9 <= x2 - x1 <= 1.2 * 20 + 1e-9 and 0.85 * 40 - 1e-9 <= y2 - y1 <= 1.2 * 40 + 1e-9
        assert abs((x1 + x2) / 2 - 20) <= 0.1 * 20 + 1e-9 and abs((y1 + y2) / 2 - 30) <= 0.1 * 40 + 1e-9


def test_classifier_plan_from_reviews(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    v1 = prepare_reader_training(dataset, latest, tmp_path)
    assert v1["class_scheme"] == "v1" and "not_a_sign" not in v1["classes"]
    assert "other_sign" in v1["classes"] and not any(":give_way" in label for label in v1["classes"])

    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2")
    assert plan["class_scheme"] == "v2" and plan["trainable"]
    # give_way has 3 training crops (MIN_KIND_TRAIN_CROPS): a class; town_entry has one: folded into other_sign.
    assert MIN_KIND_TRAIN_CROPS == 3
    assert "other_sign:give_way" in plan["classes"] and "not_a_sign" in plan["classes"]
    assert plan["folded_labels"] == {"other_sign:town_entry": "other_sign"}
    assert "other_sign" in plan["classes"] and "other_sign:town_entry" not in plan["classes"]
    # The removed proposals: the duplicate on the sign is no negative, nor the one of the buffer frame.
    assert plan["skipped"]["rejected_overlaps_a_kept_box"] == 1
    assert plan["skipped"]["rejected_outside_train_validation_test"] == 1
    assert plan["negatives"] == {"test": {"rejected": 1, "mined": 0, "reviewed": 0},
                                 "train": {"rejected": 1, "mined": 0, "reviewed": 1},
                                 "validation": {"rejected": 1, "mined": 0, "reviewed": 0}}
    assert plan["negatives_by_label_basis"]["train"] == {"rejected": {"ai_review": 1}, "reviewed": {"human_review": 1}}
    assert plan["negatives_by_tier"]["test"] == {"rejected": {"human": 1}}
    assert plan["negatives_by_tier"]["train"]["rejected"] == {"ai_before_rule": 1}
    assert any("before_the_other_sign_rule" in item for item in plan["limitations"])
    negatives = [r for r in plan["records"] if r["label"] == "not_a_sign"]
    assert {r["source"] for r in negatives} == {"rejected", "reviewed"}
    assert all(r["split"] in ("train", "validation", "test") for r in negatives)
    # The sign crops are the v1 records, with their v2 labels.
    assert len([r for r in plan["records"] if r["label"] != "not_a_sign"]) == len(v1["records"])
    assert plan["positives_by_split"] == {"buffer": 1, "test": 3, "train": 9, "validation": 5}

    with pytest.raises(ValueError, match="--classes v2"):
        prepare_reader_training(dataset, latest, tmp_path, negatives_path=tmp_path / "x.jsonl")


def test_reviewed_frames_keep_their_partition_and_reviewers(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    rows = load_jsonl(dataset)
    assignments = json.loads((tmp_path / "splits" / "fixture.json").read_text())["assignments"]
    frames = reviewed_frames(rows, assignments)
    assert [frame["split"] for frame in frames] == [split for split, *_ in FRAMES]
    assert [frame["tier"] for frame in frames][:4] == ["ai_before_rule", "ai_before_rule", "ai_after_rule", "human"]
    assert [len(frame["rejected"]) for frame in frames][:2] == [1, 1] and len(frames[5]["kept"]) == 1
    # A frame without signs keeps its whole-frame row and has no box.
    assert frames[5]["kept"][0]["bbox_xyxy"] is None


def _negatives_file(path: Path, plan: dict, rows: list[dict], **header) -> Path:
    head = {"kind": "header", "schema_version": 1, "task": "classifier_negatives",
            "dataset_sha256": plan["dataset_sha256"], "split_sha256": plan["split_sha256"], "counts": {}, **header}
    path.write_text("".join(json.dumps(item) + "\n" for item in [head, *rows]), encoding="utf-8")
    return path


def test_mined_negatives_are_checked_against_the_plan(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2")
    frames = reviewed_frames(load_jsonl(dataset), json.loads((tmp_path / "splits" / "fixture.json").read_text())["assignments"])
    test_frame = next(frame for frame in frames if frame["split"] == "test")
    good = {"kind": "negative", "source": "mined", "split": "test", "image_path": test_frame["image_path"],
            "image_sha256": test_frame["image_sha256"], "bbox_xyxy": [50, 30, 60, 40], "score": .4}
    path = _negatives_file(tmp_path / "negatives.jsonl", plan, [good])
    header, rows = read_negatives(path)
    assert header["task"] == "classifier_negatives" and len(rows) == 1
    mined = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", negatives_path=path)
    assert mined["negatives"]["test"]["mined"] == 1 and mined["negatives_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    [record] = [r for r in mined["records"] if r["source"] == "mined"]
    assert record["label"] == "not_a_sign" and record["tier"] == "human" and record["annotation_id"] is None
    # Another partition than the frame's, another dataset, a file without its header: refused.
    _negatives_file(path, plan, [{**good, "split": "train"}])
    with pytest.raises(ValueError, match="negatives_mismatch"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", negatives_path=path)
    _negatives_file(path, {**plan, "dataset_sha256": "0" * 64}, [good])
    with pytest.raises(ValueError, match="negatives_mismatch"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", negatives_path=path)
    path.write_text(json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="negatives_invalid"):
        read_negatives(path)


def test_mining_keeps_free_boxes_and_prefers_reviewed_by_a_person():
    frame = {"kept": [{"bbox_xyxy": [0, 0, 10, 10]}, {"bbox_xyxy": None}], "rejected": [{"bbox_xyxy": [50, 50, 60, 60]}]}
    predictions = [{"box": [1, 1, 10, 10], "score": .9},       # on the kept box
                   {"box": [52, 52, 61, 61], "score": .8},     # on the removed proposal
                   {"box": [9, 0, 30, 10], "score": .7},       # touches the kept box: IoU 1/30 < 0.1, free
                   {"box": [100, 0, 110, 10], "score": .6}, {"box": [200, 0, 210, 10], "score": .5},
                   {"box": [300, 0, 310, 10], "score": .4}]
    free, best = frame_candidates(frame, predictions, per_frame=3)
    assert free == 4 and [item["score"] for item in best] == [.7, .6, .5]

    candidates = ([{"tier": "ai_before_rule", "n": index} for index in range(10)]
                  + [{"tier": "human", "n": index} for index in range(3)]
                  + [{"tier": "ai_after_rule", "n": index} for index in range(2)])
    chosen = select_negatives(candidates, 7, "0 train")
    assert [item["tier"] for item in chosen] == ["human"] * 3 + ["ai_after_rule"] * 2 + ["ai_before_rule"] * 2
    assert chosen == select_negatives(candidates, 7, "0 train")
    assert len(select_negatives(candidates, 100, "0 train")) == 15 and select_negatives(candidates, 0, "x") == []


def test_chain_tally_counts_signs_read_right_and_false_boxes_that_survive():
    truth = [{"box": [0, 0, 10, 10], "label": "maximum_speed:60", "speed_sign": True, "label_basis": "human_review"},
             {"box": [100, 0, 130, 30], "label": "maximum_speed:80", "speed_sign": True, "label_basis": "ai_review"},
             {"box": [200, 0, 250, 50], "label": "other_sign", "speed_sign": False, "label_basis": "ai_review"},
             {"box": [300, 0, 345, 45], "label": "cancellation", "speed_sign": True, "label_basis": "ai_review"}]
    selected = [{"box": [0, 0, 10, 10], "score": .9, "predicted": "maximum_speed:60"},       # found, right
                {"box": [100, 0, 130, 30], "score": .8, "predicted": "maximum_speed:60"},   # found, wrong
                {"box": [200, 0, 250, 50], "score": .7, "predicted": "other_sign"},         # other sign, right
                {"box": [300, 0, 345, 45], "score": .6, "predicted": "not_a_sign"},         # found, rejected
                {"box": [500, 0, 510, 10], "score": .5, "predicted": "not_a_sign"},         # false, rejected
                {"box": [600, 0, 610, 10], "score": .4, "predicted": "maximum_speed:30"}]   # false, survives
    tally = ChainTally()
    tally.add("human_review", truth, selected, 0.5)
    tally.add("ai_review", [], [{"box": [0, 0, 5, 5], "score": .3, "predicted": "variable_display:unknown"}], 0.5)
    measured = tally.metrics()
    speed = measured["speed_signs"]
    assert (speed["truth"], speed["found"], speed["correct"], speed["rejected"]) == (3, 3, 1, 1)
    assert speed["recall"] == 1 / 3 and speed["recall_found"] == 1.0
    assert speed["by_width"]["lt12"]["recall"] == 1.0 and speed["by_width"]["20-40"]["recall"] == 0.0
    assert speed["by_width"]["ge40"]["truth"] == 1 and speed["by_width"]["12-20"]["truth"] == 0
    assert speed["human_reviewed"]["truth"] == 1 and speed["human_reviewed"]["recall"] == 1.0
    assert measured["all_boxes"]["truth"] == 4 and measured["all_boxes"]["correct"] == 2
    assert measured["boxes_per_frame"] == 3.5 and measured["false_boxes"] == 3 and measured["false_per_frame"] == 1.5
    assert measured["surviving_false_boxes"] == 2 and measured["surviving_false_per_frame"] == 1.0
    assert measured["false_rejected_rate"] == 1 / 3
    assert measured["human"] == {"images": 1, "false_per_frame": 2.0, "surviving_false_per_frame": 1.0}
    assert measured["surviving_false_by_label"] == {"maximum_speed:30": 1, "variable_display:unknown": 1}
    assert measured["confusion"]["not_a_sign"] == {"maximum_speed:30": 1, "not_a_sign": 1, "variable_display:unknown": 1}
    assert measured["confusion"]["maximum_speed:80"] == {"maximum_speed:60": 1}


def test_truth_objects_use_the_classifier_labels():
    frame = {"kept": [
        {"bbox_xyxy": [0, 0, 5, 5], "review_state": "accepted", "sign_family": "other_sign", "sign_type": "town_entry",
         "value_kph": None, "value_state": "unknown", "provenance": {"label_basis": "human_review"}},
        {"bbox_xyxy": [9, 0, 15, 5], "review_state": "corrected", "sign_family": "not_a_sign", "value_kph": None,
         "value_state": "not_applicable"},
        {"bbox_xyxy": [20, 0, 25, 5], "review_state": "accepted", "sign_family": "maximum_speed", "value_kph": 60,
         "value_state": "read"},
        {"bbox_xyxy": None, "review_state": "accepted", "sign_family": "not_a_sign"}]}
    v2 = truth_objects(frame, "v2", ["other_sign", "maximum_speed:60", "not_a_sign"], {"other_sign:town_entry": "other_sign"})
    assert [(item["label"], item["speed_sign"]) for item in v2] == [("other_sign", False), ("maximum_speed:60", True)]
    assert v2[0]["label_basis"] == "human_review"
    v1 = truth_objects(frame, "v1", ["other_sign", "maximum_speed:60"])
    assert [item["label"] for item in v1] == ["other_sign", "maximum_speed:60"]


def test_classifier_metrics_split_signs_and_rejections():
    samples = [
        {"truth": "maximum_speed:60", "predicted": "maximum_speed:60", "correct": True, "label_basis": "human_review", "source": "reviewed"},
        {"truth": "other_sign:give_way", "predicted": "other_sign", "correct": False, "label_basis": "ai_review", "source": "reviewed"},
        {"truth": "maximum_speed:80", "predicted": "not_a_sign", "correct": False, "label_basis": "ai_review", "source": "reviewed"},
        {"truth": "not_a_sign", "predicted": "not_a_sign", "correct": True, "label_basis": "ai_review", "source": "rejected"},
        {"truth": "not_a_sign", "predicted": "maximum_speed:30", "correct": False, "label_basis": "human_review", "source": "mined"},
        {"truth": "not_a_sign", "predicted": "not_a_sign", "correct": True, "label_basis": "ai_review", "source": "mined"},
    ]
    measured = classifier_metrics(samples, ["maximum_speed:60", "not_a_sign"])
    signs, negatives = measured["signs"], measured["not_a_sign"]
    assert signs["sample_count"] == 3 and signs["accuracy"] == 1 / 3 and signs["family_accuracy"] == 2 / 3
    assert signs["rejected"] == 1 and signs["human_reviewed"]["sample_count"] == 1
    assert (negatives["crops"], negatives["rejected"]) == (3, 2) and negatives["rejection_rate"] == 2 / 3
    assert negatives["by_source"]["mined"] == {"crops": 2, "rejected": 1, "rejection_rate": .5}
    assert negatives["by_source"]["reviewed"] == {"crops": 0, "rejected": 0, "rejection_rate": None}
    assert negatives["human_reviewed"]["rejection_rate"] == 0.0 and negatives["survivors"] == {"maximum_speed:30": 1}


def test_the_model_in_use_resolves_to_its_weights(tmp_path):
    root = tmp_path / "data"
    weights = root / "runs" / "r1" / "detector-sizes" / "96x64" / "detector-best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    (root / "models").mkdir()
    active = root / "models" / "active.json"
    active.write_text(json.dumps({"cutter": {"id": "cutter:r1/detector-sizes/96x64"}, "classifier": None}))
    assert resolve_model("active", "cutter", root) == {"path": weights.resolve(), "id": "cutter:r1/detector-sizes/96x64"}
    with pytest.raises(ValueError, match="no_classifier_in_use"):
        resolve_model("active", "classifier", root)
    active.write_text(json.dumps({"cutter": {"id": "cutter:../x/detector"}}))
    with pytest.raises(ValueError, match="no_cutter_in_use"):
        resolve_model("active", "cutter", root)
    active.write_text(json.dumps({"cutter": {"id": "cutter:r2/detector"}}))
    with pytest.raises(ValueError, match="cutter_in_use_missing"):
        resolve_model("active", "cutter", root)
    # A path inside runs/ gets its model's id (scripts/run_classifier.sh passes paths); any other path none.
    assert resolve_model(str(weights), "cutter", root)["id"] == "cutter:r1/detector-sizes/96x64"
    elsewhere = tmp_path / "detector-best.pt"
    elsewhere.write_bytes(b"x")
    assert resolve_model(str(elsewhere), "cutter", root)["id"] is None
    assert resolve_model(str(weights), "classifier", root)["id"] is None
    with pytest.raises(ValueError, match="checkpoint_missing"):
        resolve_model(str(root / "nothing.pt"), "classifier", root)
    # SignNet, the cutter: its weights at the root of its run.
    signnet = root / "runs" / "s1" / "signnet-best.pt"
    signnet.parent.mkdir(parents=True)
    signnet.write_bytes(b"x")
    active.write_text(json.dumps({"cutter": {"id": "cutter:s1/signnet"}}))
    assert resolve_model("active", "cutter", root) == {"path": signnet.resolve(), "id": "cutter:s1/signnet"}
    assert resolve_model(str(signnet), "cutter", root)["id"] == "cutter:s1/signnet"
    active.write_text(json.dumps({"cutter": {"id": "cutter:s2/signnet"}}))
    with pytest.raises(ValueError, match=r"cutter_in_use_missing: cutter:s2/signnet has no runs/s2/signnet-best\.pt"):
        resolve_model("active", "cutter", root)
    # A model id of the role, as the Models tab lists it (the Train tab's start of a classifier).
    assert resolve_model("cutter:s1/signnet", "cutter", root) == {"path": signnet.resolve(), "id": "cutter:s1/signnet"}
    classifier = root / "runs" / "c1" / "classifier" / "reader-best.pt"
    classifier.parent.mkdir(parents=True)
    classifier.write_bytes(b"x")
    assert resolve_model("classifier:c1/classifier", "classifier", root) == {
        "path": classifier.resolve(), "id": "classifier:c1/classifier"}
    with pytest.raises(ValueError, match=r"classifier_checkpoint_missing: classifier:c2/classifier has no runs/c2"):
        resolve_model("classifier:c2/classifier", "classifier", root)
    # A SignNet is a cutter only.
    active.write_text(json.dumps({"classifier": {"id": "classifier:s1/signnet"}}))
    with pytest.raises(ValueError, match="no_classifier_in_use"):
        resolve_model("active", "classifier", root)
    assert resolve_model(str(signnet), "classifier", root)["id"] is None
    # The decider's weights are named after its variant.
    decider = root / "runs" / "r3" / "experiment-b" / "crop_geom" / "crop_geom-best.pt"
    decider.parent.mkdir(parents=True)
    decider.write_bytes(b"x")
    active.write_text(json.dumps({"decider": {"id": "decider:r3/experiment-b/crop_geom"}}))
    assert resolve_model("active", "decider", root) == {"path": decider.resolve(), "id": "decider:r3/experiment-b/crop_geom"}
    assert resolve_model(str(decider), "decider", root)["id"] == "decider:r3/experiment-b/crop_geom"
    active.write_text(json.dumps({"decider": {"id": "decider:r3/experiment-b/other"}}))
    with pytest.raises(ValueError, match="no_decider_in_use"):
        resolve_model("active", "decider", root)


def test_reader_config_and_resume_flags_of_the_classifier():
    v2 = TrainConfig(class_scheme="v2", augment=True, pretrained=True, epochs=30).__dict__
    flags = resume_arguments({"task": "reader", "config": v2})
    assert flags[flags.index("--classes") + 1] == "v2" and "--no-augment" not in flags
    assert _config_of(flags, "reader").__dict__ == v2
    plain = TrainConfig(class_scheme="v2", augment=False).__dict__
    flags = resume_arguments({"task": "reader", "config": plain})
    assert "--no-augment" in flags and _config_of(flags, "reader").__dict__ == plain
    # A reader saved before the class schemes resumes as v1, without new flags.
    old = {key: value for key, value in TrainConfig(pretrained=True).__dict__.items() if key not in ("class_scheme", "augment")}
    flags = resume_arguments({"task": "reader", "config": old})
    assert "--classes" not in flags and _config_of(flags, "reader").__dict__ == TrainConfig(pretrained=True).__dict__


def test_cli_options_of_the_classifier(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    base = ["--data-root", str(tmp_path)]
    code, out = invoke(capsys, "train", str(dataset), *base, "--classes", "v2", "--dry-run")
    assert code == 0, out.err
    summary = json.loads(out.out)
    assert summary["class_scheme"] == "v2" and "not_a_sign" in summary["classes"]
    code, out = invoke(capsys, "train", str(dataset), *base, "--classes", "v2", "--no-augment", "--dry-run")
    assert code == 0, out.err
    code, out = invoke(capsys, "train", str(dataset), *base, "--no-augment", "--dry-run")
    assert code == 2 and "detector options" in out.err
    code, out = invoke(capsys, "train", str(dataset), *base, "--negatives", str(tmp_path / "n.jsonl"), "--dry-run")
    assert code == 2 and "--classes v2" in out.err
    code, out = invoke(capsys, "train", str(dataset), *base, "--task", "detector", "--classes", "v2", "--dry-run")
    assert code == 2 and "reader options" in out.err
    code, out = invoke(capsys, "evaluate", str(dataset), *base, "--classes", "v2", "--partition", "test", "--dry-run")
    assert code == 0, out.err
    report = json.loads(out.out)
    assert report["class_scheme"] == "v2" and report["not_a_sign_by_source"] == {"rejected": 1}
    assert report["counts_by_label"] == {"maximum_speed:30": 1, "maximum_speed:50": 1, "not_a_sign": 1, "other_sign": 1}

    code, out = invoke(capsys, "mine-negatives", str(dataset), *base, "--dry-run")
    assert code == 0, out.err
    dry = json.loads(out.out)
    assert dry["counts"]["train"]["frames"] == 6 and dry["counts"]["train"]["positives"] == 9
    assert dry["counts"]["train"]["cap"] == 27 and dry["counts"]["train"]["most_mined"] == 18
    assert dry["counts"]["train"]["frames_by_tier"] == {"human": 3, "ai_after_rule": 1, "ai_before_rule": 2, "other": 0}
    assert dry["counts"]["test"]["dataset_negatives"] == {"rejected": 1, "mined": 0, "reviewed": 0}
    code, out = invoke(capsys, "evaluate-chain", str(dataset), *base, "--partition", "test", "--dry-run")
    assert code == 0, out.err
    chain = json.loads(out.out)
    assert (chain["image_count"], chain["speed_sign_count"], chain["reviewed_box_count"]) == (2, 2, 3)
    code, out = invoke(capsys, "mine-negatives", str(dataset), *base)
    assert code == 2 and "--output" in out.err
    assert "torch" not in sys.modules


def _chain_report(recall, surviving, *, rejects=True):
    level = {"speed_signs": {"truth": 80, "found": 63, "correct": int(recall * 80), "recall": recall, "recall_found": 63 / 80,
                             "by_width": {key: {"truth": 20, "correct": 10, "recall": .5} for key in ("lt12", "12-20", "20-40", "ge40")},
                             "human_reviewed": {"truth": 23, "correct": 12, "recall": 12 / 23}},
             "boxes_per_frame": 42.0, "false_per_frame": 41.0, "surviving_false_per_frame": surviving}
    return {"cutter": {"id": "cutter:r/detector-sizes/960x544"}, "image_count": 604, "speed_sign_count": 80,
            "classifier": {"rejects": rejects}, "by_score_threshold": {"0.1": level, "0.05": level}}


def test_the_comparison_puts_the_new_classifier_next_to_the_old_reader(tmp_path):
    run, old = tmp_path / "runs" / "new", tmp_path / "runs" / "old"
    (run / "evaluation").mkdir(parents=True)
    (run / "classifier").mkdir()
    (old / "evaluation").mkdir(parents=True)
    (run / "classifier" / "result.json").write_text(json.dumps({
        "best_epoch": 7, "history": [{}] * 12, "best_validation_sign_accuracy": .81, "best_validation_rejection": .95,
        "best_validation_balanced_accuracy": .88, "classes": ["a", "b"], "loss_weights": {"not_a_sign": .25}}))
    (run / "evaluation" / "classifier-test.json").write_text(json.dumps({
        "dataset_sha256": "d", "split_sha256": "s",
        "signs": {"sample_count": 134, "accuracy": .70, "family_accuracy": .78, "rejected": 4, "rejected_rate": 4 / 134,
                  "human_reviewed": {"sample_count": 77, "accuracy": .66}},
        "not_a_sign": {"crops": 444, "rejected": 420, "rejection_rate": 420 / 444,
                       "by_source": {"mined": {"crops": 402, "rejected": 390, "rejection_rate": 390 / 402},
                                     "rejected": {"crops": 42, "rejected": 30, "rejection_rate": 30 / 42},
                                     "reviewed": {"crops": 0, "rejected": 0, "rejection_rate": None}}}}))
    (old / "evaluation" / "reader-test.json").write_text(json.dumps({
        "dataset_sha256": "d", "split_sha256": "s", "accuracy": .709, "sample_count": 134,
        "human_reviewed": {"accuracy": .597, "sample_count": 77}}))
    (run / "evaluation" / "chain-test.json").write_text(json.dumps(_chain_report(.55, 1.2)))
    (run / "evaluation" / "chain-test-old-reader.json").write_text(json.dumps(_chain_report(.5, 41.0, rejects=False)))
    text = classifier_comparison(run, old)
    lines = text.splitlines()
    assert lines[0].startswith("== training: best epoch 7 of 12 run, validation: signs read right 0.81")
    assert "(old reader: its own report, same dataset and split)" in text
    row = next(line for line in lines if line.startswith("signs read right (old reader's labels)"))
    assert "0.78 of 134" in row and "0.71 of 134" in row
    assert "0.66 of 77" in next(line for line in lines if line.startswith("  confirmed by a person"))
    assert "0.95 420/444" in next(line for line in lines if line.startswith("not_a_sign rejected"))
    assert "0.97 390/402" in next(line for line in lines if line.startswith("  cutter's false boxes"))
    assert not any(line.startswith("  marked not a sign") for line in lines)
    assert "== the classifier alone, validation: not evaluated ==" in text
    chain = lines[lines.index(next(line for line in lines if line.startswith("== the chain on test"))) + 1:]
    assert chain[0].split() == ["score", "classifier", "e2e", "human", "<12", "12-20", "20-40", ">=40", "cutter_found",
                                "boxes/frame", "false/frame", "surviving/frame"]
    assert chain[1].split()[:4] == ["0.05", "new", "0.55", "44/80"] and chain[1].split()[-1] == "1.20"
    assert chain[2].split()[:5] == ["0.05", "old", "reader", "0.50", "40/80"] and chain[2].split()[-1] == "41.00"
    assert "== the chain on validation: not evaluated ==" in text


def test_the_script_checks_its_arguments_first():
    for args, message in (([], "usage"), (["a", "b"], "usage"), (["../x"], "unsupported characters")):
        process = subprocess.run(["bash", str(REPO / "scripts" / "run_classifier.sh"), *args], capture_output=True,
                                 text=True, timeout=60, env={"PATH": "/usr/bin:/bin", "NNSLR_DATA_ROOT": "/nonexistent"})
        assert process.returncode == 2 and message in process.stderr, process.stderr
    # The two stages: a mode of three, one start, a file, a whole number of epochs.
    for settings, message in (({"NNSLR_CLASSIFIER_MAPILLARY": "2"}, "must be 0, 1 or pretrain"),
                              ({"NNSLR_CLASSIFIER_MAPILLARY": "pretrain", "NNSLR_CLASSIFIER_INIT_FROM": "/etc/hostname"},
                               "two different starts"),
                              ({"NNSLR_CLASSIFIER_INIT_FROM": "/nonexistent/reader-best.pt"}, "is not a file"),
                              ({"NNSLR_CLASSIFIER_PRETRAIN_EPOCHS": "x"}, "PRETRAIN_EPOCHS must be a whole number")):
        process = subprocess.run(["bash", str(REPO / "scripts" / "run_classifier.sh"), "r"], capture_output=True, check=False,
                                 text=True, timeout=60,
                                 env={"PATH": "/usr/bin:/bin", "NNSLR_DATA_ROOT": "/nonexistent", **settings})
        assert process.returncode == 2 and message in process.stderr, process.stderr


def test_the_script_never_writes_into_a_signnet_run(tmp_path):
    # A stand-in GPU environment and an idle GPU: the run folder's guard is what stops it.
    tools, venv = tmp_path / "bin", tmp_path / "venv" / "bin"
    for folder in (tools, venv):
        folder.mkdir(parents=True)
    for tool in (tools / "nvidia-smi", venv / "nnslr", venv / "python"):
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)
    run = tmp_path / "data" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "signnet-best.pt").write_bytes(b"x")
    process = subprocess.run(["bash", str(REPO / "scripts" / "run_classifier.sh"), "r1"], capture_output=True, check=False,
                             text=True, timeout=60, env={"PATH": f"{tools}:/usr/bin:/bin",
                                                         "NNSLR_DATA_ROOT": str(tmp_path / "data"),
                                                         "NNSLR_GPU_VENV": str(venv.parent)})
    assert process.returncode == 3 and "it has signnet-best.pt" in process.stderr, process.stderr


# ---------------------------------------------------------------- with torch, in a subprocess, CPU only

MAKE_MODELS = r"""
import sys
from pathlib import Path
import torch
from torchvision.models import mobilenet_v3_small
from nnslr_tools.checkpoints import build_detector

dataset_sha, split_sha, cutter_path, reader_path = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
torch.manual_seed(0)
# A random cutter: its boxes have scores around 0.5, so there are false boxes to mine.
cutter = build_detector(wide=False, input_size=(96, 64), anchor_min_ratio=0.1)
cutter_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({"format_version": 1, "task": "detector", "architecture": "ssdlite320_mobilenet_v3_large",
            "classes": ["background", "speed_sign"], "input_size": [96, 64], "anchor_min_ratio": 0.1,
            "state_dict": cutter.state_dict(), "dataset_sha256": dataset_sha, "split_sha256": split_sha, "epoch": 1},
           cutter_path)
# An old reader (v1: no class scheme in its checkpoint, no not_a_sign).
reader = mobilenet_v3_small(weights=None)
classes = ["maximum_speed:30", "maximum_speed:50", "other_sign"]
reader.classifier[-1] = torch.nn.Linear(reader.classifier[-1].in_features, len(classes))
reader_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({"format_version": 1, "task": "reader", "architecture": "mobilenet_v3_small", "classes": classes,
            "input_size": 32, "state_dict": reader.state_dict(), "dataset_sha256": dataset_sha,
            "split_sha256": split_sha, "epoch": 1}, reader_path)
print("models ok")
"""

AUGMENT_CHECK = r"""
import torch
from torchvision.transforms import functional as TF
from nnslr_tools.training import CLASSIFIER_AUGMENTATION, DETECTOR_AUGMENTATION, detector_photometric
crop = torch.rand(3, 16, 16)
images = [detector_photometric(torch, TF, crop, torch.Generator().manual_seed(seed), CLASSIFIER_AUGMENTATION)
          for seed in range(60)]
changed = [image for image in images if not torch.equal(image, crop)]
assert 30 <= len(changed) < 60 and all(float(i.min()) >= 0 and float(i.max()) <= 1 for i in images)
assert sum(float(image.mean()) for image in changed) / len(changed) < float(crop.mean())
# The same as the detector's: contrast down to half, and the light blur of rain.
assert CLASSIFIER_AUGMENTATION == DETECTOR_AUGMENTATION and DETECTOR_AUGMENTATION["contrast"][0] == 0.5
assert DETECTOR_AUGMENTATION["blur_probability"] > 0
# The training draws in its loader workers and changes the pixels of the whole batch on the GPU: the same images.
from nnslr_tools.training import photometric_batch, photometric_draws
crops = torch.rand(48, 3, 20, 20, generator=torch.Generator().manual_seed(1))
expected = torch.stack([detector_photometric(torch, TF, crop, torch.Generator().manual_seed(seed),
                                             CLASSIFIER_AUGMENTATION) for seed, crop in enumerate(crops)])
draws = torch.tensor([photometric_draws(torch, torch.Generator().manual_seed(seed), CLASSIFIER_AUGMENTATION)
                      for seed in range(48)], dtype=torch.float32)
assert 0 < int(draws[:, 0].sum()) < 48 and int((draws[:, 0] * draws[:, 8]).sum()) > 0
assert torch.allclose(photometric_batch(torch, TF, crops, draws), expected, atol=1e-5)
print("augment ok")
"""

CHECKPOINT_CHECK = r"""
import sys
from nnslr_tools.checkpoints import safe_torch_load
saved = safe_torch_load(sys.argv[1])
assert saved["class_scheme"] == "v2" and "not_a_sign" in saved["classes"], saved.get("class_scheme")
assert saved["selected_by"] == "balanced_accuracy" and saved["folded_labels"] == {"other_sign:town_entry": "other_sign"}
assert saved["negatives_sha256"] and saved["input_size"] == 32
state = safe_torch_load(sys.argv[2])
assert state["class_scheme"] == "v2" and state["config"]["class_scheme"] == "v2" and state["status"] == "completed"
print("checkpoint ok")
"""


def test_tiny_cpu_classifier_chain(tmp_path, capsys, torch_python):
    dataset = _classifier_fixture(tmp_path, capsys)
    plan = prepare_reader_training(dataset, tmp_path / "splits" / "latest.json", tmp_path, class_scheme="v2")
    run = tmp_path / "runs" / "tiny-classifier"
    cutter = tmp_path / "runs" / "cutter-run" / "detector-sizes" / "96x64" / "detector-best.pt"
    old_reader = tmp_path / "runs" / "old-run" / "reader" / "reader-best.pt"
    process = _run(torch_python, "-c", MAKE_MODELS, plan["dataset_sha256"], plan["split_sha256"], str(cutter),
                   str(old_reader))
    assert process.returncode == 0, process.stderr[-3000:]
    process = _run(torch_python, "-c", AUGMENT_CHECK)
    assert process.returncode == 0 and "augment ok" in process.stdout, process.stderr[-3000:]
    # The cutter in use, as the Models tab writes it.
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "active.json").write_text(json.dumps({"cutter": {"id": "cutter:cutter-run/detector-sizes/96x64"}}))
    base = ["--data-root", str(tmp_path), "--device", "cpu", "--num-workers", "0"]

    negatives = run / "negatives.jsonl"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "mine-negatives", str(dataset), *base, "--cutter", "active",
                   "--output", str(negatives), "--batch-size", "3", "--total-factor", "1")
    assert process.returncode == 0, process.stderr[-3000:]
    header, rows = read_negatives(negatives)
    assert header["cutter"]["id"] == "cutter:cutter-run/detector-sizes/96x64" and header["cutter"]["same_split"]
    assert header["cutter"]["input_size"] == [96, 64] and header["dataset_sha256"] == plan["dataset_sha256"]
    counts = header["counts"]
    assert {split: counts[split]["frames"] for split in counts} == {"train": 6, "validation": 3, "test": 2}
    frames = {frame["image_sha256"]: frame for frame in reviewed_frames(
        load_jsonl(dataset), json.loads((tmp_path / "splits" / "fixture.json").read_text())["assignments"])}
    per_frame = {}
    for row in rows:
        frame = frames[row["image_sha256"]]
        assert row["split"] == frame["split"] and row["tier"] == frame["tier"] and row["score"] >= 0.1
        assert all(_iou(row["bbox_xyxy"], box) < 0.1 for box in
                   [r["bbox_xyxy"] for r in frame["kept"] if r["bbox_xyxy"]] + [r["bbox_xyxy"] for r in frame["rejected"]])
        per_frame[row["image_sha256"]] = per_frame.get(row["image_sha256"], 0) + 1
    assert per_frame and max(per_frame.values()) <= 3
    for split, count in counts.items():
        assert count["selected"] == min(count["cap"], count["after_frame_cap"]) and count["cap"] == count["positives"]
        assert sum(count["selected_by_tier"].values()) == count["selected"] == sum(1 for r in rows if r["split"] == split)
    assert counts["train"]["selected_by_tier"]["human"] == min(counts["train"]["candidates_by_tier"]["human"],
                                                            counts["train"]["cap"])
    process = _run(torch_python, "-m", "nnslr_tools.cli", "mine-negatives", str(dataset), *base, "--cutter", "active",
                   "--output", str(negatives))
    assert process.returncode == 2 and "output_exists" in process.stderr

    output = run / "classifier"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "train", str(dataset), "--task", "reader", "--classes", "v2",
                   "--negatives", str(negatives), *base, "--output", str(output), "--epochs", "2", "--batch-size", "4",
                   "--input-size", "32", "--no-amp")
    assert process.returncode == 0, process.stderr[-3000:]
    process_train = process
    # The epoch line the panel parses: the classifier's measures before the seconds.
    for epoch in (1, 2):
        assert re.search(rf"^classifier epoch {epoch}/2 train_loss \S+ val_loss \S+ train_accuracy \S+ val_accuracy \S+ "
                         rf"val_sign_accuracy [0-9.]+ val_rejection [0-9.]+ \d+s$", process.stdout, re.M), process.stdout
    assert re.search(r"^classifier crops: 9 signs and \d+ not_a_sign in train", process.stdout, re.M)
    result = json.loads((output / "result.json").read_text())
    assert result["class_scheme"] == "v2" and "not_a_sign" in result["classes"] and "other_sign:give_way" in result["classes"]
    assert result["selection"] == {"by": "balanced_accuracy"} and result["config"]["augment"] is True
    assert result["folded_labels"] == {"other_sign:town_entry": "other_sign"}
    assert result["augmentation"]["box_jitter"]["scale"] == [0.85, 1.2]
    assert 0 < result["loss_weights"]["not_a_sign"] <= 1
    assert result["negatives"]["by_split"]["train"]["mined"] == counts["train"]["selected"]
    best = max(result["history"], key=lambda item: item["validation_balanced_accuracy"])
    assert result["best_epoch"] == best["epoch"]
    assert result["best_validation_balanced_accuracy"] == best["validation_balanced_accuracy"]
    assert all(key in item for item in result["history"] for key in ("validation_sign_accuracy", "validation_rejection", "seconds"))
    process = _run(torch_python, "-c", CHECKPOINT_CHECK, str(output / "reader-best.pt"), str(output / "training-state.pt"))
    assert process.returncode == 0 and "checkpoint ok" in process.stdout, process.stderr[-3000:]
    # Every crop was cut from its frame once; a run with loader workers reuses them and trains the same.
    assert re.search(r"^crop cache: cut \d+ crops of \d+ images", process_train.stdout, re.M), process_train.stdout
    assert result["crop_cache"]["crops"] > 9 and 0 < result["crop_cache"]["frames"] <= 9
    process = _run(torch_python, "-m", "nnslr_tools.cli", "train", str(dataset), "--task", "reader", "--classes", "v2",
                   "--negatives", str(negatives), *base[:-1], "2", "--output", str(run / "workers"), "--epochs", "2",
                   "--batch-size", "4", "--input-size", "32", "--no-amp")
    assert process.returncode == 0, process.stderr[-3000:]
    assert re.search(r"^crop cache: reused ", process.stdout, re.M), process.stdout
    again = json.loads((run / "workers" / "result.json").read_text())
    assert again["crop_cache"]["folder"] == result["crop_cache"]["folder"]
    assert [item["train_loss"] for item in again["history"]] == [item["train_loss"] for item in result["history"]]

    checkpoint = output / "reader-best.pt"
    report_path = run / "evaluation" / "classifier-test.json"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate", str(dataset), "--task", "reader", "--classes", "v2",
                   "--negatives", str(negatives), *base, "--partition", "test", "--checkpoint", str(checkpoint),
                   "--output", str(report_path))
    assert process.returncode == 0, process.stderr[-3000:]
    report = json.loads(report_path.read_text())
    assert report["class_scheme"] == "v2" and report["signs"]["sample_count"] == 3
    assert report["not_a_sign"]["crops"] == 1 + counts["test"]["selected"]
    assert report["not_a_sign"]["by_source"]["rejected"]["crops"] == 1
    assert {sample["source"] for sample in report["samples"]} >= {"reviewed", "rejected"}
    # The keys the Evaluate tab reads stay.
    for key in ("accuracy", "per_class", "confusion", "human_reviewed", "counts_by_label_basis", "samples"):
        assert key in report
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate", str(dataset), "--task", "reader", *base,
                   "--partition", "test", "--checkpoint", str(checkpoint))
    assert process.returncode == 2 and "checkpoint_class_scheme_mismatch" in process.stderr

    chain_path = run / "evaluation" / "chain-test.json"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate-chain", str(dataset), *base, "--cutter", "active",
                   "--classifier", str(checkpoint), "--partition", "test", "--output", str(chain_path))
    assert process.returncode == 0, process.stderr[-3000:]
    chain = json.loads(chain_path.read_text())
    chain_classes = set(json.loads((output / "result.json").read_text())["classes"])
    assert chain["task"] == "chain" and chain["classifier"]["class_scheme"] == "v2" and chain["classifier"]["rejects"]
    assert (chain["image_count"], chain["speed_sign_count"], chain["reviewed_box_count"]) == (2, 2, 3)
    assert chain["score_thresholds"] == [0.05, 0.1, 0.25] and set(chain["by_score_threshold"]) == {"0.05", "0.1", "0.25"}
    for metrics in chain["by_score_threshold"].values():
        speed = metrics["speed_signs"]
        assert speed["truth"] == 2 and speed["correct"] <= speed["found"] <= speed["truth"]
        assert metrics["surviving_false_boxes"] <= metrics["false_boxes"]
        assert metrics["boxes_per_frame"] >= metrics["false_per_frame"]
        assert sum(sum(row.values()) for row in metrics["confusion"].values()) == round(metrics["boxes_per_frame"] * 2)
    low, high = chain["by_score_threshold"]["0.05"], chain["by_score_threshold"]["0.25"]
    assert low["boxes_per_frame"] >= high["boxes_per_frame"]
    assert len(chain["images"]) == 2 and all("surviving_false" in image for image in chain["images"])
    # Every box of the cutter from the lowest score, with the classifier's answer, for the Watch panel of the Evaluate
    # tab: at every score the report measured, the signs they find, the boxes and the false boxes let through are its own.
    assert chain["per_frame_boxes"]["min_score"] == 0.05 and chain["per_frame_boxes"]["box"] == "x1 y1 x2 y2 score truth label confidence"
    for image in chain["images"]:
        parts = [line.split(" ") for line in image["cutter_boxes"]]
        assert all(len(part) == 8 and part[6] in chain_classes for part in parts) and len(parts) <= PER_FRAME_MAX_BOXES
        assert all("family" in item for item in image["truth"]) and len(image["size"]) == 2
    for key, metrics in chain["by_score_threshold"].items():
        step, threshold = PER_FRAME_STEPS.index(float(key)), float(key)
        found = sum(len({int(part[5]) for part in (line.split(" ") for line in image["cutter_boxes"])
                         if int(part[5]) >= 0 and float(part[4]) >= threshold}) for image in chain["images"])
        assert found == metrics["all_boxes"]["found"], key
        assert sum(image["cutter_box_counts"][step] for image in chain["images"]) == round(metrics["boxes_per_frame"] * 2), key
        assert sum(image["surviving_counts"][step] for image in chain["images"]) == metrics["surviving_false_boxes"], key

    old_path = run / "evaluation" / "chain-test-old-reader.json"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate-chain", str(dataset), *base, "--cutter", str(cutter),
                   "--classifier", str(old_reader), "--partition", "test", "--output", str(old_path))
    assert process.returncode == 0, process.stderr[-3000:]
    old = json.loads(old_path.read_text())
    assert old["classifier"]["class_scheme"] == "v1" and not old["classifier"]["rejects"]
    for metrics in old["by_score_threshold"].values():
        # The old reader rejects nothing: every false box survives.
        assert metrics["surviving_false_boxes"] == metrics["false_boxes"]
    assert any("without_not_a_sign" in item for item in old["limitations"])
    text = classifier_comparison(run, tmp_path / "runs" / "old-run")
    assert "== the chain on test" in text and "old reader" in text and "new" in text



MAKE_SIGNNET = r"""
import math
import sys
from pathlib import Path
import torch
from nnslr_tools.signnet import build_signnet, checkpoint_of

dataset_sha, split_sha, path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
torch.manual_seed(0)
# A random SignNet at a tiny band: its heat around 0.5 (bias 0), boxes about 12 band pixels a side.
model = build_signnet((4, 4, 4, 4), 8, 1)
with torch.no_grad():
    model.out.bias[0] = 0.0
    model.out.bias[1:3] = math.log(12)
path.parent.mkdir(parents=True, exist_ok=True)
plan = {"dataset_sha256": dataset_sha, "split_sha256": split_sha, "families": "all"}
torch.save(checkpoint_of(model, plan, widths=(4, 4, 4, 4), stride=8, fuse_kernel=1, band_width=96, epoch=1), path)
print("signnet ok")
"""

SIGNNET_BOXES_CHECK = r"""
import json
import sys
from pathlib import Path
import torch
from PIL import Image
from nnslr_tools import chain
from nnslr_tools.signnet import _band_of, frame_boxes_of

data_root, frames = Path(sys.argv[1]), json.loads(sys.argv[2])
resolved = chain.resolve_model(sys.argv[3], "cutter", data_root)
cutter, checkpoint, band = chain._load_cutter(resolved, torch.device("cpu"))
assert chain.cutter_kind(cutter) == "signnet" and band == [96, 32] and checkpoint["families"] == "all"
seen = 0
for frame, pixels, found in chain.cutter_boxes(cutter, torch.device("cpu"), frames, data_root, min_score=0.1,
                                               num_workers=0, batch_size=2):
    with Image.open(data_root / frame["image_path"]) as image:
        image = image.convert("RGB")
        width, height = image.size
        with torch.no_grad():
            heat, shape = cutter.model(_band_of(image, 96)[None].float() / 255.0)
    alone = frame_boxes_of(heat, shape, 8, 96, width, min_score=0.1)
    # The same boxes as the frame alone (a batch of two may round the last bits apart), clamped to the frame.
    assert len(found) == len(alone) and 0 < len(found) <= 5, found
    for mine, theirs in zip(found, alone):
        clamped = [min(max(v, 0.0), limit) for v, limit in zip(theirs["box"], (width, height, width, height))]
        assert abs(mine["score"] - theirs["score"]) < 1e-5, (mine, theirs)
        assert all(abs(a - b) < 1e-3 for a, b in zip(mine["box"], clamped)), (mine, clamped)
    seen += 1
assert seen == len(frames)
print("boxes ok")
"""


def test_tiny_cpu_signnet_chain(tmp_path, capsys, torch_python):
    """The chain with a SignNet as the cutter in use: the car's boxes (its band, at most 5 a frame, in frame pixels)
    for mining the negatives and for evaluate-chain, at the SignNet's scores."""
    dataset = _classifier_fixture(tmp_path, capsys)
    plan = prepare_reader_training(dataset, tmp_path / "splits" / "latest.json", tmp_path, class_scheme="v2")
    signnet = tmp_path / "runs" / "signnet-run" / "signnet-best.pt"
    old_reader = tmp_path / "runs" / "old-run" / "reader" / "reader-best.pt"
    process = _run(torch_python, "-c", MAKE_SIGNNET, plan["dataset_sha256"], plan["split_sha256"], str(signnet))
    assert process.returncode == 0 and "signnet ok" in process.stdout, process.stderr[-3000:]
    process = _run(torch_python, "-c", MAKE_MODELS, plan["dataset_sha256"], plan["split_sha256"],
                   str(tmp_path / "runs" / "unused" / "detector-best.pt"), str(old_reader))
    assert process.returncode == 0, process.stderr[-3000:]
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "active.json").write_text(json.dumps({"cutter": {"id": "cutter:signnet-run/signnet"}}))
    base = ["--data-root", str(tmp_path), "--device", "cpu", "--num-workers", "0"]

    frames = [{"image_path": row["image_path"]} for row in load_jsonl(dataset)][:4]
    process = _run(torch_python, "-c", SIGNNET_BOXES_CHECK, str(tmp_path), json.dumps(frames), "active")
    assert process.returncode == 0 and "boxes ok" in process.stdout, process.stderr[-3000:]

    negatives = tmp_path / "runs" / "tiny" / "negatives.jsonl"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "mine-negatives", str(dataset), *base, "--cutter", "active",
                   "--output", str(negatives), "--batch-size", "3", "--total-factor", "1")
    assert process.returncode == 0, process.stderr[-3000:]
    header, rows = read_negatives(negatives)
    cutter = header["cutter"]
    assert cutter["id"] == "cutter:signnet-run/signnet" and cutter["kind"] == "signnet" and cutter["same_split"]
    assert cutter["band"] == cutter["input_size"] == [96, 32] and cutter["stride"] == 8 and cutter["max_boxes"] == 5
    assert cutter["families"] == "all" and cutter["dataset_sha256"] == plan["dataset_sha256"]
    per_frame = {}
    for row in rows:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        assert 0 <= x1 <= x2 <= WIDTH and 0 <= y1 <= y2 <= HEIGHT and row["score"] >= 0.1
        per_frame[row["image_sha256"]] = per_frame.get(row["image_sha256"], 0) + 1
    assert per_frame and max(per_frame.values()) <= 3

    chain_path = tmp_path / "runs" / "tiny" / "chain-test.json"
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate-chain", str(dataset), *base, "--cutter", "active",
                   "--classifier", str(old_reader), "--partition", "test", "--output", str(chain_path))
    assert process.returncode == 0, process.stderr[-3000:]
    chain = json.loads(chain_path.read_text())
    assert chain["cutter"]["kind"] == "signnet" and chain["cutter"]["id"] == "cutter:signnet-run/signnet"
    assert chain["score_thresholds"] == [0.1, 0.2, 0.3] and chain["main_score_threshold"] == 0.3
    assert set(chain["by_score_threshold"]) == {"0.1", "0.2", "0.3"}
    assert any(item.startswith("signnet_top_band_and_5_boxes_a_frame_as_the_car") for item in chain["limitations"])
    assert chain["per_frame_boxes"]["min_score"] == 0.1 and len(chain["images"]) == 2
    for image in chain["images"]:
        parts = [line.split(" ") for line in image["cutter_boxes"]]
        assert 0 < len(parts) <= 5 and all(float(part[4]) >= 0.1 for part in parts)
        assert all(0 <= float(part[0]) <= float(part[2]) <= WIDTH and 0 <= float(part[1]) <= float(part[3]) <= HEIGHT
                   for part in parts)
    # Scores given on the command line are the caller's, a path to the weights inside runs gets its id.
    process = _run(torch_python, "-m", "nnslr_tools.cli", "evaluate-chain", str(dataset), *base, "--cutter", str(signnet),
                   "--classifier", str(old_reader), "--partition", "test", "--score-thresholds", "0.2,0.4",
                   "--main-score-threshold", "0.4", "--output", str(chain_path.with_name("chain-given.json")))
    assert process.returncode == 0, process.stderr[-3000:]
    given = json.loads(chain_path.with_name("chain-given.json").read_text())
    assert given["score_thresholds"] == [0.2, 0.4] and given["main_score_threshold"] == 0.4
    assert given["cutter"]["id"] == "cutter:signnet-run/signnet"


def _iou(a, b):
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


# ---------------------------------------------------------------- the cutter cuts the other signs too

def test_cutter_families_choose_the_positive_boxes(tmp_path, capsys):
    from nnslr_tools.training import DetectorTrainConfig, cutter_families, prepare_detector_training

    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    every = prepare_detector_training(dataset, latest, tmp_path)
    speed = prepare_detector_training(dataset, latest, tmp_path, families="speed")
    assert every["families"] == "all" and speed["families"] == "speed" and every["classes"] == ["speed_sign"]
    # The other signs are boxes of the cutter; road markings and boxes marked not a sign never are.
    assert every["boxes_by_family"]["train"] == {"maximum_speed": 5, "other_sign": 4}
    assert speed["boxes_by_family"]["train"] == {"maximum_speed": 5}
    assert every["boxes_by_split"]["test"] == 3 and speed["boxes_by_split"]["test"] == 2
    assert all(len(frame["boxes"]) == len(frame["box_families"]) for frame in every["frames"])
    assert "other_signs_unboxed_in_ai_reviews_before_the_other_sign_rule" in every["limitations"]
    with pytest.raises(ValueError, match="unknown cutter families"):
        prepare_detector_training(dataset, latest, tmp_path, families="road")
    # What a checkpoint that does not say was trained on: the speed signs.
    assert cutter_families({}) == "speed" and cutter_families({"families": "all"}) == "all"
    assert DetectorTrainConfig().families == "all"

    base = ["--data-root", str(tmp_path)]
    code, out = invoke(capsys, "train", str(dataset), "--task", "detector", *base, "--cutter-families", "speed", "--dry-run")
    assert code == 0 and json.loads(out.out)["families"] == "speed", out.err
    code, out = invoke(capsys, "train", str(dataset), *base, "--cutter-families", "all", "--dry-run")
    assert code == 2 and "detector option" in out.err
    code, out = invoke(capsys, "evaluate", str(dataset), "--task", "detector", *base, "--partition", "test", "--dry-run")
    assert code == 0 and json.loads(out.out)["sample_count"] == 2, out.err
    code, out = invoke(capsys, "evaluate", str(dataset), "--task", "detector", *base, "--partition", "test",
                       "--cutter-families", "speed", "--dry-run")
    assert code == 0 and json.loads(out.out)["sample_count"] == 2, out.err



def test_train_signnet_dry_run_says_the_data_and_whether_it_fits_the_car(tmp_path, capsys):
    from nnslr_tools.training import prepare_detector_training

    dataset = _classifier_fixture(tmp_path, capsys)
    plan = prepare_detector_training(dataset, tmp_path / "splits" / "latest.json", tmp_path)
    run = tmp_path / "runs" / "dry"
    base = ["train-signnet", "--data-root", str(tmp_path), "--run-dir", str(run), "--dry-run"]
    code, out = invoke(capsys, *base)
    assert code == 0, out.err
    said = json.loads(out.out)
    assert said["dry_run"] and said["task"] == "signnet" and said["families"] == "all"
    assert (said["dataset_sha256"], said["split_sha256"]) == (plan["dataset_sha256"], plan["split_sha256"])
    assert said["band"] == [896, 288] and said["stride"] == 8 and said["fits_the_car"] is True
    assert said["boxes_by_split"] == plan["boxes_by_split"] and not run.exists()
    code, out = invoke(capsys, *base, "--band-width", "448", "--families", "speed")
    said = json.loads(out.out)
    assert code == 0 and said["band"] == [448, 144] and said["fits_the_car"] is False and said["families"] == "speed"
    code, out = invoke(capsys, *base, "--widths", "16,24,32")
    assert code == 2 and "invalid --widths" in out.err
    assert said["trainable"] is True


def test_train_signnet_refuses_a_plan_that_cannot_train_and_flushes_its_lines(tmp_path, capsys, monkeypatch):
    import builtins

    from nnslr_tools import signnet, training

    dataset = _classifier_fixture(tmp_path, capsys)
    real = training.prepare_detector_training
    plan = real(dataset, tmp_path / "splits" / "latest.json", tmp_path)
    monkeypatch.setattr(training, "prepare_detector_training",
                        lambda *a, **kw: {**real(*a, **kw), "trainable": False})
    run = tmp_path / "runs" / "never"
    base = ["train-signnet", "--data-root", str(tmp_path), "--run-dir", str(run)]
    code, out = invoke(capsys, *base, "--dry-run")
    assert code == 1 and json.loads(out.out)["trainable"] is False
    code, out = invoke(capsys, *base)
    assert code == 2 and "training_plan_not_trainable" in out.err and not run.exists()

    # A plan that trains: the lines are printed with flush (signnet.log is a file behind tee), and the car's
    # numbers (best 5 boxes of 0.3 or more) come with the 5-box ones.
    monkeypatch.setattr(training, "prepare_detector_training", lambda *a, **kw: plan)
    given = {}
    measured = {"recall_iou": .4, "false_positives_per_image": .2}

    def fake(plan, root, run_dir, **kw):
        given.update(kw)
        kw["log"]("signnet epoch 1/1 train_loss 1.0000 val_recall 0.400 val_false_per_frame 0.20 3s")
        return {"best_epoch": 1, "parameters": 9, **{split: {"by_box_budget": {"5": measured}, "car": measured}
                                                     for split in ("validation", "test")}}

    monkeypatch.setattr(signnet, "train_signnet", fake)
    flushed = []
    printing = builtins.print
    monkeypatch.setattr(builtins, "print", lambda *a, **kw: (flushed.append(kw.get("flush")), printing(*a, **kw)))
    code, out = invoke(capsys, *base)
    monkeypatch.setattr(builtins, "print", printing)
    assert code == 0, out.err
    assert out.out.startswith("signnet epoch 1/1 ") and flushed[0] is True
    assert json.loads(out.out.splitlines()[-1])["car"]["test"] == measured

def test_cutter_families_resume_as_they_were_trained():
    from nnslr_tools.training import DetectorTrainConfig, _detector_config_differences

    saved = dict(DetectorTrainConfig(families="speed").__dict__, input_size=[320, 320])
    flags = resume_arguments({"task": "detector", "config": saved})
    assert flags[flags.index("--cutter-families") + 1] == "speed"
    assert _detector_config_differences(saved, _config_of(flags, "detector")) == []
    # A detector saved before the setting cut the speed signs only, and resumes so.
    old = {key: value for key, value in saved.items() if key != "families"}
    flags = resume_arguments({"task": "detector", "config": old})
    assert flags[flags.index("--cutter-families") + 1] == "speed"


def test_recall_by_family_keeps_the_speed_signs_comparable():
    from nnslr_tools.evaluation import recall_by_family, score_frame, threshold_metrics

    truth = [[0, 0, 10, 10], [100, 0, 120, 20], [200, 0, 220, 20]]
    scored = {**score_frame(truth, [{"box": [0, 0, 10, 10], "score": .9}, {"box": [200, 0, 220, 20], "score": .8}],
                            score_threshold=.1, iou_threshold=.5),
              "truth_families": ["maximum_speed", "cancellation", "other_sign"], "label_basis": "human_review"}
    assert recall_by_family([scored]) == {"other_sign": {"truth": 1, "found": 1, "recall": 1.0},
                                          "speed": {"truth": 2, "found": 1, "recall": .5}}
    measured = threshold_metrics([scored])
    assert measured["recall_by_family"]["speed"]["recall"] == .5
    assert measured["human_reviewed"]["recall_by_family"]["other_sign"]["found"] == 1
    # Frames scored without their families (older code paths): no such measure.
    assert recall_by_family([{k: v for k, v in scored.items() if k != "truth_families"}]) is None
    assert "recall_by_family" not in threshold_metrics([{k: v for k, v in scored.items() if k != "truth_families"}])


def test_a_box_on_a_vehicle_is_imported_as_not_a_sign_with_its_kind(tmp_path, capsys):
    root = tmp_path / "data"
    row = reviewed(root, "000000aa--abcdef1234", index=0)
    frame_png(root / row["image_path"], 0)
    sticker = _box("v", [4, 6, 20, 26], "not_a_sign", sign_type="on_vehicle")
    # A kind left over from an other sign is not the kind of a "not a sign" box: it is dropped on import.
    stale = _box("n", [30, 4, 40, 14], "not_a_sign", sign_type="give_way")
    row.update(width=WIDTH, height=HEIGHT, frame_sha256=hashlib.sha256((root / row["image_path"]).read_bytes()).hexdigest(),
               detections=[sticker, stale], source_proposals=[dict(sticker), dict(stale)])
    code, out = invoke(capsys, "import-annotations", str(write(root, [row])), "--data-root", str(root))
    assert code == 0, out.err + out.out
    canonical = load_jsonl(root / "annotations" / "objects.jsonl")
    assert validate_dataset(canonical, root)["valid"]
    kinds = sorted(str(r["sign_type"]) for r in canonical if r["sign_family"] == "not_a_sign")
    assert kinds == ["None", "on_vehicle"]
    # The classifier keeps the plain "not a sign" box only: cut to its box, the sticker is a sign.
    from nnslr_tools.training import _classifier_label
    assert sorted(str(_classifier_label(r)) for r in canonical) == ["None", "not_a_sign"]


def test_a_number_with_a_plate_of_hours_is_a_sign_of_its_own():
    hours = {**_row("maximum_speed", 30), "supplementary_panel": "time"}
    assert _classifier_label(hours) == "maximum_speed_hours:30"
    # The v1 reader, the classes it was published with, stays as it was.
    assert _reader_label(hours) == "maximum_speed:30"
    # Without a number there is nothing to read: it stays what it was.
    assert _classifier_label({**_row("maximum_speed"), "supplementary_panel": "time"}) == "maximum_speed:unknown"
    assert _classifier_label({**_row("zone", 30), "supplementary_panel": "time"}) == "zone:30"
    for panel in ("none", "present", "unknown", "unresolved", None):
        assert _classifier_label({**_row("maximum_speed", 30), "supplementary_panel": panel}) == "maximum_speed:30"
    assert family_label("maximum_speed_hours:30") == "maximum_speed:30"


def _frames_with_hours(frames, count):
    """FRAMES with the first boxes of a limit of 30 in train marked Hours, as tall boxes of the disc and its plate."""
    changed, left = [], count
    for split, who, kept, removed in frames:
        boxes = []
        for box in kept:
            if left and split == "train" and box["value_kph"] == 30:
                box, left = {**box, "bbox_xyxy": [4, 6, 20, 36], "supplementary_panel": "time"}, left - 1
            boxes.append(box)
        changed.append((split, who, boxes, removed))
    return changed


def test_the_classes_change_only_with_an_hours_label(tmp_path, capsys, monkeypatch):
    from nnslr_tools.training import prepare_detector_training

    plain = tmp_path / "plain"
    latest = plain / "splits" / "latest.json"
    before = prepare_reader_training(_classifier_fixture(plain, capsys), latest, plain, class_scheme="v2")
    assert not any("hours" in label for label in before["classes"])
    assert not any("holds_plate" in record for record in before["records"])
    cutter = prepare_detector_training(plain / "annotations" / "objects.jsonl", latest, plain)
    assert not any("plate_boxes" in frame for frame in cutter["frames"])

    marked = tmp_path / "marked"
    monkeypatch.setitem(globals(), "FRAMES", _frames_with_hours(FRAMES, 2))
    latest = marked / "splits" / "latest.json"
    plan = prepare_reader_training(_classifier_fixture(marked, capsys), latest, marked, class_scheme="v2")
    assert sorted(set(plan["classes"]) - set(before["classes"])) == ["maximum_speed_hours:30"]
    plated = [record for record in plan["records"] if record["label"] == "maximum_speed_hours:30"]
    assert len(plated) == 2 and all(record["holds_plate"] for record in plated)
    v1 = prepare_reader_training(marked / "annotations" / "objects.jsonl", latest, marked)
    assert not any("hours" in label for label in v1["classes"])
    cutter = prepare_detector_training(marked / "annotations" / "objects.jsonl", latest, marked)
    assert sum(len(frame.get("plate_boxes", [])) for frame in cutter["frames"]) == 2
    assert all(box in frame["boxes"] for frame in cutter["frames"] for box in frame.get("plate_boxes", []))
    # Still a speed limit for the cutter: its family does not change.
    assert cutter["boxes_by_family"]["train"]["maximum_speed"] == 5
