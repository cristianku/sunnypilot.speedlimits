"""Experiment B (nnslr_tools.context_experiment): the CPU-only plan and metrics, and a tiny CPU run of each
variant when torch is importable (the GPU environment); without torch those runs are skipped."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.context_experiment import (
    OWNERSHIP_CLASSES,
    VARIANTS,
    comparison_table,
    decider_label,
    expected_calibration_error,
    geometry_features,
    ownership_class_weights,
    ownership_metrics,
    ownership_target,
    selection_score,
)
from test_annotations import invoke, reviewed, write

TRAIN = ("00000001--abcdef1234", "00000002--abcdef1234")
VALIDATION = "00000003--abcdef1234"
TEST = "00000004--abcdef1234"
BUFFER = "00000005--abcdef1234"
SRC = Path(__file__).resolve().parents[2] / "src"


def _frame(root, route, index, detections, *, speed=None, ai=False):
    """A reviewed frame with boxes side by side: (family, value_kph, applicability) each."""
    row = reviewed(root, route, index=index)
    boxes = []
    for number, (family, value, applicability) in enumerate(detections):
        box = copy.deepcopy(row["detections"][0])
        box.update(detection_id=f"box-{number}", bbox_xyxy=[1 + 16 * number, 2, 15 + 16 * number, 20],
                   family=family, value_kph=value, applicability=applicability)
        boxes.append(box)
    row["detections"] = boxes
    row["source_proposals"] = copy.deepcopy(boxes)
    if speed is not None:
        row["gps"] = {"latitude": 47.0, "longitude": 8.0, "speed_mps": speed, "bearing_deg": 0.0}
        row["country"] = "CH"
    if ai:
        row.update(reviewer="ai/Claude", label_basis="ai_review", reviewer_kind="ai", review_channel="mcp")
    return row


def _experiment_fixture(root, capsys):
    """Two train routes, one validation, one test and one buffer route, split by hand."""
    rows = [
        _frame(root, TRAIN[0], 0, [("maximum_speed", 30, "own_road"), ("maximum_speed", 50, "other_road")], speed=20.0),
        _frame(root, TRAIN[0], 1, [("maximum_speed", 50, "own_road")], ai=True),
        _frame(root, TRAIN[0], 2, [("other_sign", None, "unresolved"), ("cancellation", None, "unresolved")],
               speed=10.0, ai=True),
        _frame(root, TRAIN[1], 0, [("maximum_speed", 30, "unresolved")]),
        _frame(root, TRAIN[1], 1, [("maximum_speed", 30, "own_road"), ("maximum_speed", 50, "own_road")], ai=True),
        _frame(root, VALIDATION, 0, [("maximum_speed", 30, "own_road"), ("maximum_speed", 50, "other_road")],
               speed=25.0),
        _frame(root, VALIDATION, 1, [("other_sign", None, "unresolved")], ai=True),
        _frame(root, TEST, 0, [("maximum_speed", 50, "own_road")]),
        _frame(root, TEST, 1, [("maximum_speed", 30, "other_road"), ("cancellation", None, "own_road")], ai=True),
        _frame(root, BUFFER, 0, [("maximum_speed", 30, "own_road")], ai=True),
    ]
    code, out = invoke(capsys, "import-annotations", str(write(root, rows)), "--data-root", str(root))
    assert code == 0, out.err + out.out
    dataset = root / "annotations" / "objects.jsonl"
    canonical = load_jsonl(dataset)
    report = validate_dataset(canonical, root)
    assert report["valid"], report["errors"]

    partition = {route: "train" for route in TRAIN}
    partition.update({VALIDATION: "validation", TEST: "test", BUFFER: "buffer"})
    payload = {
        "schema_version": 1,
        "dataset_kind": "gold",
        "dataset_sha256": report["dataset_sha256"],
        "assignments": {r["annotation_id"]: partition[r["route_id"]] for r in canonical},
    }
    frozen = root / "splits" / "fixture.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (root / "splits" / "latest.json").write_text(
        json.dumps({"path": "splits/fixture.json", "sha256": digest}) + "\n", encoding="utf-8")
    return dataset, frozen, report["dataset_sha256"]


def test_dry_run_counts_without_torch(tmp_path, capsys):
    _, frozen, dataset_sha = _experiment_fixture(tmp_path, capsys)
    code, out = invoke(capsys, "experiment-b", "--variant", "full", "--data-root", str(tmp_path), "--dry-run")
    assert code == 0, out.err + out.out
    summary = json.loads(out.out)

    assert summary["variant"] == "full" and summary["inputs"] == ["crop", "box_geometry", "car_speed", "scene"]
    assert summary["dataset_sha256"] == dataset_sha
    assert summary["split_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert "records" not in summary and summary["record_count"] == 14
    counts = summary["counts"]
    assert counts["boxes_by_partition"] == {"train": 8, "validation": 3, "test": 3}
    assert counts["frames_by_partition"] == {"train": 5, "validation": 2, "test": 2}
    assert counts["human_review_boxes_by_partition"] == {"train": 3, "validation": 2, "test": 1}
    # The buffer (and any partition other than train/validation/test) is never read.
    assert summary["ignored_partitions"] == {"buffer": 1}
    assert summary["classes"] == ["cancellation", "maximum_speed:30", "maximum_speed:50", "other_sign"]
    assert counts["reading_by_partition"]["test"] == {"cancellation": 1, "maximum_speed:30": 1, "maximum_speed:50": 1}
    # Ownership only on the speed families: other_sign is masked.
    assert counts["ownership_by_partition"]["train"] == {"own_road": 4, "other_road": 1, "unresolved": 2, "masked": 1}
    assert counts["ownership_by_partition"]["validation"] == {"own_road": 1, "other_road": 1, "unresolved": 0, "masked": 1}
    assert counts["ownership_human_review_by_partition"]["train"] == {"own_road": 1, "other_road": 1, "unresolved": 1,
                                                                      "masked": 0}
    assert counts["car_speed_by_partition"]["train"] == {"present": 4, "missing": 4}
    weights = summary["ownership_class_weights"]
    assert weights["other_road"] == pytest.approx(7 / 3) and weights["own_road"] == pytest.approx(7 / 12)
    assert summary["images"] == {"frames": 9, "missing": 0, "missing_examples": []}
    assert summary["unseen_labels"] == {"validation": [], "test": []}
    assert summary["trainable"] is True
    assert "torch" not in sys.modules

    code, out = invoke(capsys, "experiment-b", "--variant", "crop", "--data-root", str(tmp_path),
                       "--dry-run", "--limit", "1")
    assert code == 0, out.err + out.out
    limited = json.loads(out.out)
    assert limited["counts"]["frames_by_partition"] == {"train": 1, "validation": 1, "test": 1}
    assert "limited_smoke_run_not_a_measure" in limited["limitations"]
    assert "torch" not in sys.modules


def test_a_real_run_needs_an_output(tmp_path, capsys):
    _experiment_fixture(tmp_path, capsys)
    code, out = invoke(capsys, "experiment-b", "--variant", "crop", "--data-root", str(tmp_path))
    assert code == 2 and "--output is required" in out.err


def test_ownership_metrics_precision_recall_abstain_and_macro_f1():
    def sample(truth, predicted, confidence=.9):
        return {"ownership_truth": truth, "ownership_predicted": predicted, "ownership_confidence": confidence}

    samples = [sample("own_road", "own_road"), sample("own_road", "own_road"), sample("own_road", "unresolved"),
               sample("other_road", "other_road"), sample("other_road", "own_road"),
               sample("unresolved", "unresolved")]
    metrics = ownership_metrics(samples)
    assert metrics["confusion"]["own_road"] == {"own_road": 2, "other_road": 0, "unresolved": 1}
    assert metrics["other_road"]["precision"] == 1.0 and metrics["other_road"]["recall"] == .5
    assert metrics["per_class"]["own_road"]["precision"] == pytest.approx(2 / 3)
    assert metrics["abstain_rate"] == pytest.approx(2 / 6)
    assert metrics["decided_accuracy"] == pytest.approx(3 / 4)
    f1 = [2 * 2 / (3 + 3), 2 * 1 / (2 + 1), 2 * 1 / (1 + 2)]
    assert metrics["macro_f1"] == pytest.approx(sum(f1) / 3)
    empty = ownership_metrics([])
    assert empty["sample_count"] == 0 and empty["macro_f1"] is None and empty["other_road"]["recall"] is None


def test_the_epoch_is_chosen_on_the_reading_and_the_other_road_f1_not_on_unresolved():
    """#72: one unresolved box on validation must not choose the epoch over the signs of another road found."""
    def sample(truth, predicted):
        return {"ownership_truth": truth, "ownership_predicted": predicted, "ownership_confidence": .9}

    ours = [sample("own_road", "own_road")] * 10
    # A gets the one unresolved box right and finds 1 of 4 signs of another road; B misses it and finds 3 of 4.
    a = ownership_metrics(ours + [sample("other_road", "other_road")] + [sample("other_road", "own_road")] * 3
                          + [sample("unresolved", "unresolved")])
    b = ownership_metrics(ours + [sample("other_road", "other_road")] * 3 + [sample("other_road", "own_road")]
                          + [sample("unresolved", "own_road")])
    assert a["macro_f1"] > b["macro_f1"]
    score_a = selection_score({"reading": {"accuracy": .9}, "ownership": a})
    score_b = selection_score({"reading": {"accuracy": .9}, "ownership": b})
    assert score_a == pytest.approx((.9 + 2 * 1 / (4 + 1)) / 2) and score_b == pytest.approx((.9 + 2 * 3 / (4 + 3)) / 2)
    assert score_b > score_a
    # No sign of another road, true or predicted: its F1 counts 0.
    none = ownership_metrics(ours)
    assert none["other_road"]["f1"] is None and selection_score({"reading": {"accuracy": .9}, "ownership": none}) == .45


def test_calibration_class_weights_and_geometry():
    assert expected_calibration_error([], []) is None
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([.95, .95], [True, False]) == pytest.approx(.45)
    weights = ownership_class_weights({"own_road": 1000, "other_road": 10, "unresolved": 90})
    assert weights["own_road"] == pytest.approx(1100 / 3000) and weights["other_road"] == 10.0
    assert ownership_class_weights({}) == {"own_road": 1.0, "other_road": 1.0, "unresolved": 1.0}
    features = geometry_features([134.4, 76, 268.8, 152], 1344, 760, None)
    assert features == pytest.approx([.1, .1, .2, .2, .1, .1, .15, .15, 0.0, 1.0])
    assert geometry_features([0, 0, 1344, 760], 1344, 760, 20.0)[-2:] == [.5, 0.0]


def test_a_box_on_a_vehicle_teaches_the_decider_not_ours():
    sticker = {"sign_family": "not_a_sign", "sign_type": "on_vehicle", "review_state": "corrected",
               "bbox_xyxy": [10, 10, 30, 30]}
    assert decider_label(sticker) == "not_a_sign" and ownership_target(sticker) == "other_road"
    # A plain "not a sign" stays out of the decider; a speed sign keeps its own road.
    assert decider_label({**sticker, "sign_type": None}) is None
    assert ownership_target({**sticker, "sign_type": None}) is None
    assert ownership_target({**sticker, "review_state": "rejected"}) is None
    speed = {"sign_family": "maximum_speed", "value_state": "read", "value_kph": 80, "applicability": "own_road",
             "review_state": "accepted", "bbox_xyxy": [10, 10, 30, 30]}
    assert decider_label(speed) == "maximum_speed:80" and ownership_target(speed) == "own_road"


def test_comparison_table_reads_the_variants(tmp_path):
    ownership = {"other_road": {"precision": .5, "recall": .25}, "macro_f1": .6, "abstain_rate": .1}
    report = {"box_count": 10, "reading": {"accuracy": .8}, "ownership": ownership}
    (tmp_path / "crop").mkdir()
    (tmp_path / "crop" / "result.json").write_text(json.dumps(
        {"metrics": {"validation": {**report, "human_reviewed": None}, "test": {**report, "human_reviewed": report}}}))
    table = comparison_table(tmp_path)
    lines = table.splitlines()
    assert lines[0].split() == ["variant", "scope", "boxes", "read_acc", "other_P", "other_R", "own_mF1", "abstain"]
    assert lines[1].split() == ["crop", "test", "10", "0.800", "0.500", "0.250", "0.600", "0.100"]
    assert lines[3].split() == ["crop", "val", "(select)", "10", "0.800", "0.500", "0.250", "0.600", "0.100"]
    assert lines[4].split() == ["crop", "val", "human", "0", "-", "-", "-", "-", "-"]
    assert "crop_geom  no result.json" in table and "full" in table


# The tiny runs need torch, torchvision and Pillow (the GPU environment). They run in a subprocess, through
# the CLI: torch never enters this interpreter, whose other tests check that it was not imported.
TRAINING_STACK = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "PIL"))


@pytest.mark.skipif(not TRAINING_STACK, reason="torch/torchvision/Pillow not installed (GPU environment only)")
@pytest.mark.parametrize("variant", VARIANTS)
def test_tiny_cpu_run_of_each_variant(tmp_path, capsys, variant):
    dataset, frozen, dataset_sha = _experiment_fixture(tmp_path, capsys)
    output = tmp_path / "runs" / "tiny" / "experiment-b" / variant
    # Random weights (no download), CPU only, no wandb; few threads, so a busy server does not slow it down.
    environment = dict(os.environ, WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    process = subprocess.run(
        [sys.executable, "-m", "nnslr_tools.cli", "experiment-b", "--variant", variant,
         "--data-root", str(tmp_path), "--output", str(output), "--device", "cpu", "--epochs", "1",
         "--batch-size", "2", "--crop-size", "64", "--num-workers", "0", "--no-amp"],
        capture_output=True, text=True, env=environment, timeout=600,
    )
    assert process.returncode == 0, process.stderr[-3000:]

    assert re.search(rf"^experiment_b_{variant} epoch 1/1 train_loss \S+ val_loss \S+ .* \d+s$", process.stdout, re.M)
    assert (output / f"{variant}-best.pt").is_file()
    saved = json.loads((output / "result.json").read_text())
    assert saved["variant"] == variant and saved["best_epoch"] == 1
    assert saved["dataset_sha256"] == dataset_sha
    assert saved["split_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert saved["counts"]["boxes_by_partition"] == {"train": 8, "validation": 3, "test": 3}
    assert saved["torch_version"] and saved["initialization"] == "random"
    for partition in ("validation", "test"):
        metrics = saved["metrics"][partition]
        assert metrics["box_count"] == 3
        assert {"accuracy", "per_class", "confusion", "expected_calibration_error"} <= set(metrics["reading"])
        assert {"confusion", "per_class", "other_road", "abstain_rate", "macro_f1",
                "expected_calibration_error"} <= set(metrics["ownership"])
        assert {"precision", "recall"} <= set(metrics["ownership"]["other_road"])
        human = metrics["human_reviewed"]
        assert human["label_basis"] == "human_review" and {"reading", "ownership"} <= set(human)
    assert saved["metrics"]["validation"]["human_reviewed"]["box_count"] == 2
    assert saved["metrics"]["test"]["ownership"]["sample_count"] == 3
    assert len((output / "samples.jsonl").read_text().splitlines()) == 6
    # Prepare's "ours" proposer loads these weights as the decider: one answer for each box of a frame.
    script = ("import sys, torch; from PIL import Image; from nnslr_tools.context_experiment import Decider; "
              "decider = Decider(sys.argv[1], torch.device('cpu')); "
              "print(json.dumps(decider.decide(Image.new('RGB', (320, 180)), [[10, 10, 40, 40], [200, 50, 260, 120]], 25.0)))")
    decided = subprocess.run([sys.executable, "-c", "import json; " + script, str(output / f"{variant}-best.pt")],
                             capture_output=True, text=True, env=environment, timeout=300)
    assert decided.returncode == 0, decided.stderr[-3000:]
    answers = json.loads(decided.stdout)
    assert len(answers) == 2 and all(road in OWNERSHIP_CLASSES and 0 < share <= 1 for road, share in answers)
    # And exports to ONNX for the package of the networks, the same answers as PyTorch for one box and for several.
    if importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"):
        script = ("import json, sys; from pathlib import Path; from nnslr_tools.exporting import export_decider_onnx; "
                  "print(json.dumps(export_decider_onnx(Path(sys.argv[1]), Path(sys.argv[2]))['verification']))")
        exported = subprocess.run([sys.executable, "-c", script, str(output / f"{variant}-best.pt"),
                                   str(tmp_path / f"decider-{variant}.onnx")],
                                  capture_output=True, text=True, env=environment, timeout=300)
        assert exported.returncode == 0, exported.stderr[-3000:]
        assert json.loads(exported.stdout.strip().splitlines()[-1])["passed"] is True
        contract = json.loads((tmp_path / f"decider-{variant}.onnx.contract.json").read_text())
        assert contract["task"] == "decider" and contract["inputs"]["crops"]["shape"][0] == "boxes"
        assert ("scene" in contract["inputs"]) == (variant == "full")
    assert "torch" not in sys.modules
