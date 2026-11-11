"""Experiment B: does context teach whom a sign applies to, without hurting the reading?

Step 2 of the plan in docs/NEURAL_NETWORK.md. It uses the boxes already reviewed, so no detector is
involved and detector errors cannot hide the effect of the context. Three variants are trained the
same way on the same frozen split:

- ``crop``: the crop of the box only, through a MobileNetV3-small (the backbone of today's reader);
- ``crop_geom``: the crop plus where the box is in the frame (normalised corners, size and centre) and
  the speed of the car (``gps.speed_mps`` of the row, else 0 with a "missing" flag);
- ``full``: crop_geom plus the scene: the whole frame scaled down through a second MobileNetV3-small,
  its feature map pooled at the box (torchvision RoIAlign) and over the whole map.

Every variant has the same small fusion MLP with two heads on it:

- reading: the labels of the reader baseline (nnslr_tools.training._reader_label);
- ownership: own_road, other_road or unresolved (taught as "abstain"), from the row's applicability,
  only on the rows of the speed families; other rows have no ownership loss. other_road is rare, so
  the ownership loss weighs the classes by inverse frequency (capped).

A box a person marked "not a sign, on a vehicle" (a truck's top-speed sticker) is read as not_a_sign and
taught as other_road: it is no sign for our road, and the scene shows the vehicle it is on.

The best epoch is chosen on validation (SELECTION_METRIC: the reading and the F1 of another road); validation
and test are then measured, and again on the rows a person confirmed (provenance.label_basis human_review).
Train, validation and test are the only partitions read: buffer and the partitions of older split policies
are ignored.

Like nnslr_tools.training, this module imports torch, torchvision and Pillow only when a run starts:
prepare_experiment_b and the metrics are stdlib-only, so ``nnslr experiment-b --dry-run`` needs
neither torch nor a GPU.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from nnslr_tools import tracking
from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.checkpoints import atomic_torch_save, model_weights, torch_load
from nnslr_tools.evaluation import human_reviewed_metrics, label_basis_counts, reader_metrics
from nnslr_tools.progress import report as report_progress
from nnslr_tools.splits import validate_splits
from nnslr_tools.training import (
    DEFAULT_WORKERS,
    ELIGIBLE_REVIEW_STATES,
    READER_NORMALIZE_MEAN,
    READER_NORMALIZE_STD,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    _label_basis,
    _load_split_document,
    _reader_label,
    _safe_image_path,
    _sha256,
)

EXPERIMENT_SCHEMA_VERSION = 1
TASK = "experiment_b"
VARIANTS = ("crop", "crop_geom", "full")
# What each variant sees besides the reading crop.
VARIANT_INPUTS = {
    "crop": ["crop"],
    "crop_geom": ["crop", "box_geometry", "car_speed"],
    "full": ["crop", "box_geometry", "car_speed", "scene"],
}
TEST_SPLIT = "test"
# The only partitions read: the training learns from train, picks its best epoch on validation and is
# measured on validation and test. buffer (split by place) and the old route-held-out / hard-negative
# partitions are ignored.
PARTITIONS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
# Whom a sign applies to (the row's applicability). unresolved is taught as "abstain", not dropped.
OWNERSHIP_CLASSES = ("own_road", "other_road", "unresolved")
ABSTAIN = "unresolved"
# Ownership is learned and measured only on these families; other rows have no ownership loss.
OWNERSHIP_FAMILIES = ("cancellation", "maximum_speed", "variable_display", "zone", "road_marking")
# A "not a sign" box on a vehicle (nnslr_tools.sign_types.NOT_A_SIGN_TYPES): read as not_a_sign, not ours.
ON_VEHICLE = "on_vehicle"
VEHICLE_READING = "not_a_sign"
VEHICLE_OWNERSHIP = "other_road"
# The inverse-frequency weights of the ownership classes are capped here.
OWNERSHIP_WEIGHT_CAP = 10.0
# The crop: the box plus this share of its width and height on each side, clipped to the frame.
CROP_MARGIN = .25
# Train-only jitter of the box. It moves the crop, the geometry and the scene pooling together, like
# an imprecise detector: the size by up to this share, the centre by up to this share of the size.
BOX_SCALE_JITTER = .1
BOX_SHIFT_JITTER = .08
# Train-only colour jitter, one draw per frame for its crops and its scene. There is no horizontal
# flip: signs and road sides are not symmetric.
COLOUR_JITTER = {"brightness": .15, "contrast": .15, "saturation": .1}
# MobileNetV3-small features: 576 channels at stride 32.
BACKBONE_CHANNELS = 576
SCENE_STRIDE = 32
# The scene map is reduced to these channels, then pooled at the box (ROI x ROI) and over the map.
SCENE_CHANNELS = 128
SCENE_ROI_SIZE = 2
GEOMETRY_SIZE = 64
FUSION_SIZE = 256
# The speed of the car is fed as speed_mps / SPEED_SCALE_MPS (40 m/s = 144 km/h).
SPEED_SCALE_MPS = 40.0
GEOMETRY_FEATURES = ("x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y",
                     "speed", "speed_missing")
CALIBRATION_BINS = 15
IGNORE_INDEX = -100
# Each variant keeps the epoch with the best mean of the reading accuracy and the F1 of the signs of another road, on
# validation (since 11 Nov 2026, #72; before, the ownership macro-F1, where unresolved, one box on validation, weighed
# a third). Unresolved and the macro-F1 are still measured, out of the choice.
SELECTION_METRIC = "mean of validation reading accuracy and validation other_road F1"
# Rows of the comparison table printed by scripts/run_experiment_b.sh: test, and its human-reviewed
# boxes; validation too, since it chose the epoch (optimistic) but may hold the only other_road cases.
COMPARISON_SCOPES = (
    ("test", "test", False),
    ("test human", "test", True),
    ("val (select)", "validation", False),
    ("val human", "validation", True),
)


# ---------------------------------------------------------------- labels and geometry (stdlib only)

def on_vehicle(row: dict[str, Any]) -> bool:
    """A reviewed box a person marked not a sign, on a vehicle."""
    box = row.get("bbox_xyxy")
    return (row.get("sign_family") == "not_a_sign" and row.get("sign_type") == ON_VEHICLE
            and row.get("review_state") in ELIGIBLE_REVIEW_STATES and isinstance(box, list) and len(box) == 4)


def decider_label(row: dict[str, Any]) -> str | None:
    """The reading label of a row: the reader's, and not_a_sign for a box on a vehicle."""
    return VEHICLE_READING if on_vehicle(row) else _reader_label(row)


def ownership_target(row: dict[str, Any]) -> str | None:
    """The ownership class of a row of a speed family (other_road for a box on a vehicle), None (no ownership
    loss) for any other row."""
    if on_vehicle(row):
        return VEHICLE_OWNERSHIP
    if row.get("sign_family") not in OWNERSHIP_FAMILIES:
        return None
    value = row.get("applicability")
    return value if value in OWNERSHIP_CLASSES else None


def ownership_class_weights(counts: dict[str, int], cap: float = OWNERSHIP_WEIGHT_CAP) -> dict[str, float]:
    """Inverse-frequency weights of the ownership loss: total / (classes x count), at most cap.

    A class missing from train gets the cap (it has no train loss anyway); without any ownership row
    every class weighs 1.
    """
    total = sum(int(counts.get(name, 0)) for name in OWNERSHIP_CLASSES)
    if total == 0:
        return {name: 1.0 for name in OWNERSHIP_CLASSES}
    weights = {}
    for name in OWNERSHIP_CLASSES:
        count = int(counts.get(name, 0))
        weights[name] = cap if count == 0 else min(cap, total / (len(OWNERSHIP_CLASSES) * count))
    return weights


def speed_of(row: dict[str, Any]) -> float | None:
    """The speed of the car (m/s) from the row's gps, None when the row has none."""
    gps = row.get("gps")
    if not isinstance(gps, dict):
        return None
    speed = gps.get("speed_mps")
    if type(speed) not in (int, float) or not math.isfinite(speed) or speed < 0:
        return None
    return float(speed)


def clip_box(box: list[float], width: float, height: float) -> list[float]:
    """The box inside the frame, at least one pixel wide and high."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, x1 + 1.0), float(width))
    y2 = min(max(y2, y1 + 1.0), float(height))
    return [x1, y1, x2, y2]


def crop_box(box: list[float], width: float, height: float, margin: float = CROP_MARGIN) -> list[float]:
    """The region cut for the crop: the box plus margin of its size on each side, clipped."""
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    return clip_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], width, height)


def jitter_box(box: list[float], width: float, height: float, rng: random.Random | Any) -> list[float]:
    """A train-time variation of the box: a little larger or smaller, a little moved."""
    x1, y1, x2, y2 = (float(v) for v in box)
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = (x1 + x2) / 2 + rng.uniform(-BOX_SHIFT_JITTER, BOX_SHIFT_JITTER) * box_width
    center_y = (y1 + y2) / 2 + rng.uniform(-BOX_SHIFT_JITTER, BOX_SHIFT_JITTER) * box_height
    box_width *= rng.uniform(1 - BOX_SCALE_JITTER, 1 + BOX_SCALE_JITTER)
    box_height *= rng.uniform(1 - BOX_SCALE_JITTER, 1 + BOX_SCALE_JITTER)
    return clip_box([center_x - box_width / 2, center_y - box_height / 2,
                     center_x + box_width / 2, center_y + box_height / 2], width, height)


def geometry_features(box: list[float], width: float, height: float, speed_mps: float | None) -> list[float]:
    """GEOMETRY_FEATURES of a box: corners, size and centre over the frame size, then the car speed."""
    x1, y1, x2, y2 = (float(v) for v in box)
    features = [
        x1 / width, y1 / height, x2 / width, y2 / height,
        (x2 - x1) / width, (y2 - y1) / height,
        (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
    ]
    if speed_mps is None:
        return features + [0.0, 1.0]
    return features + [speed_mps / SPEED_SCALE_MPS, 0.0]


# ---------------------------------------------------------------- the plan (stdlib only)

def _partition_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = Counter()
    frames: dict[str, set[str]] = {name: set() for name in PARTITIONS}
    human = Counter()
    basis: dict[str, Counter] = {name: Counter() for name in PARTITIONS}
    reading: dict[str, Counter] = {name: Counter() for name in PARTITIONS}
    # Ownership classes, and "masked": the boxes outside the speed families (no ownership loss).
    ownership_keys = (*OWNERSHIP_CLASSES, "masked")
    ownership: dict[str, Counter] = {name: Counter(dict.fromkeys(ownership_keys, 0)) for name in PARTITIONS}
    ownership_human: dict[str, Counter] = {name: Counter(dict.fromkeys(ownership_keys, 0)) for name in PARTITIONS}
    speed: dict[str, Counter] = {name: Counter({"present": 0, "missing": 0}) for name in PARTITIONS}
    for record in records:
        name = record["split"]
        is_human = record["label_basis"] == "human_review"
        boxes[name] += 1
        frames[name].add(record["image_path"])
        human[name] += int(is_human)
        basis[name][str(record["label_basis"] or "unknown")] += 1
        reading[name][record["label"]] += 1
        target = record["ownership"] if record["ownership"] is not None else "masked"
        ownership[name][target] += 1
        if is_human:
            ownership_human[name][target] += 1
        speed[name]["present" if record["speed_mps"] is not None else "missing"] += 1
    return {
        "boxes_by_partition": {name: boxes[name] for name in PARTITIONS},
        "frames_by_partition": {name: len(frames[name]) for name in PARTITIONS},
        "human_review_boxes_by_partition": {name: human[name] for name in PARTITIONS},
        "label_basis_by_partition": {name: dict(sorted(basis[name].items())) for name in PARTITIONS},
        "reading_by_partition": {name: dict(sorted(reading[name].items())) for name in PARTITIONS},
        "ownership_by_partition": {name: dict(ownership[name]) for name in PARTITIONS},
        "ownership_human_review_by_partition": {name: dict(ownership_human[name]) for name in PARTITIONS},
        "car_speed_by_partition": {name: dict(speed[name]) for name in PARTITIONS},
    }


def prepare_experiment_b(
    dataset_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    dataset_kind: str = "gold",
    limit: int | None = None,
) -> dict[str, Any]:
    """Validate the dataset and the frozen split and build the experiment-B plan.

    The same checks as nnslr_tools.training.prepare_reader_training (dataset valid, split hash-bound
    to it and without leakage) and the same reading labels. No torch/Pillow import, no GPU. limit
    keeps the first limit frames (in image-path order) of each partition, for smoke runs.
    """
    dataset_path = Path(dataset_path)
    split_path = Path(split_path)
    data_root = Path(data_root)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("invalid_limit: --limit must be a positive number of frames")

    rows = load_jsonl(dataset_path)
    validation = validate_dataset(rows, data_root, dataset_kind=dataset_kind)
    if not validation.get("valid"):
        reasons = sorted({str(e.get("reason", "dataset_invalid")) for e in validation.get("errors", [])})
        raise ValueError("dataset_invalid: " + ",".join(reasons))

    resolved_split_path, split_doc = _load_split_document(split_path, data_root)
    split_errors = validate_splits(rows, split_doc, validation["dataset_sha256"], dataset_kind=dataset_kind)
    if split_errors:
        reasons = sorted({str(e.get("reason", "split_invalid")) for e in split_errors})
        raise ValueError("split_invalid: " + ",".join(reasons))
    assignments = split_doc["assignments"]

    records: list[dict[str, Any]] = []
    skipped = Counter()
    ignored_partitions = Counter()
    missing_images: list[str] = []
    for row in rows:
        label = decider_label(row)
        if label is None:
            skipped["not_reader_eligible"] += 1
            continue
        split_name = assignments.get(row.get("annotation_id"))
        if not isinstance(split_name, str):
            skipped["missing_split"] += 1
            continue
        if split_name not in PARTITIONS:
            ignored_partitions[split_name] += 1
            continue
        image_path = row.get("image_path")
        if not isinstance(image_path, str):
            skipped["missing_image_path"] += 1
            continue
        if not _safe_image_path(data_root, image_path).is_file():
            missing_images.append(image_path)
            continue
        records.append({
            "annotation_id": row["annotation_id"],
            "split": split_name,
            "image_path": image_path,
            "image_sha256": row.get("image_sha256"),
            "width": row.get("width"),
            "height": row.get("height"),
            "bbox_xyxy": row["bbox_xyxy"],
            "label": label,
            "ownership": ownership_target(row),
            "sign_family": row.get("sign_family"),
            "value_state": row.get("value_state"),
            "value_kph": row.get("value_kph"),
            "applicability": row.get("applicability"),
            "speed_mps": speed_of(row),
            "country": row.get("country"),
            "sign_type": row.get("sign_type"),
            "route_id": row.get("route_id"),
            "segment_index": row.get("segment_index"),
            "decoded_frame_index": row.get("decoded_frame_index"),
            "label_basis": _label_basis(row),
        })

    if limit is not None:
        kept = set()
        for name in PARTITIONS:
            paths = sorted({r["image_path"] for r in records if r["split"] == name})
            kept.update((name, path) for path in paths[:limit])
        records = [r for r in records if (r["split"], r["image_path"]) in kept]
    records.sort(key=lambda r: (PARTITIONS.index(r["split"]), r["image_path"], r["annotation_id"]))

    counts = _partition_counts(records)
    classes = sorted(counts["reading_by_partition"][TRAIN_SPLIT])
    unseen = {
        name: sorted(set(counts["reading_by_partition"][name]) - set(classes))
        for name in (VALIDATION_SPLIT, TEST_SPLIT)
    }
    train_ownership = counts["ownership_by_partition"][TRAIN_SPLIT]
    boxes = counts["boxes_by_partition"]

    warnings = []
    for name, labels in unseen.items():
        if labels:
            warnings.append(f"{name}_labels_unseen_in_train: counted as reading errors, no loss: "
                            + ",".join(labels))
    for name in OWNERSHIP_CLASSES:
        if train_ownership[name] == 0:
            warnings.append(f"ownership_class_missing_in_train: {name}")
    if boxes[TEST_SPLIT] == 0:
        warnings.append("test_partition_empty")
    if not any(r["speed_mps"] is not None for r in records):
        warnings.append("no_car_speed: no row has gps.speed_mps, the speed input is always 'missing'")

    trainable = (
        boxes[TRAIN_SPLIT] > 0
        and boxes[VALIDATION_SPLIT] > 0
        and len(classes) >= 2
        and sum(train_ownership[name] for name in OWNERSHIP_CLASSES) > 0
        and not missing_images
    )
    limitations = [
        "reviewed_boxes_no_detector",
        "ownership_only_on_speed_families",
        "single_frame_no_tracking",
        "no_runtime_bundle",
    ]
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")
    if limit is not None:
        limitations.append("limited_smoke_run_not_a_measure")

    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "task": TASK,
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": validation["dataset_sha256"],
        "split_path": str(resolved_split_path),
        "requested_split_path": str(split_path),
        "split_sha256": _sha256(resolved_split_path),
        "split_policy": split_doc.get("policy"),
        "partitions": list(PARTITIONS),
        "classes": classes,
        "ownership_classes": list(OWNERSHIP_CLASSES),
        "ownership_families": list(OWNERSHIP_FAMILIES),
        "ownership_class_weights": ownership_class_weights(train_ownership),
        "records": records,
        "counts": counts,
        "unseen_labels": unseen,
        "ignored_partitions": dict(sorted(ignored_partitions.items())),
        "skipped": dict(sorted(skipped.items())),
        "images": {
            "frames": len({r["image_path"] for r in records}),
            "missing": len(missing_images),
            "missing_examples": sorted(set(missing_images))[:20],
        },
        "limit": limit,
        "trainable": trainable,
        "warnings": warnings,
        "limitations": limitations,
    }


def dry_run_summary(plan: dict[str, Any], variant: str) -> dict[str, Any]:
    """What ``nnslr experiment-b --dry-run`` prints: the plan without its records."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown_variant: {variant}")
    summary = {key: value for key, value in plan.items() if key != "records"}
    summary["variant"] = variant
    summary["inputs"] = list(VARIANT_INPUTS[variant])
    summary["record_count"] = len(plan["records"])
    return summary


# ---------------------------------------------------------------- metrics (stdlib only)

def expected_calibration_error(confidences: list[float], corrects: list[bool],
                               bins: int = CALIBRATION_BINS) -> float | None:
    """Expected calibration error of the top-class confidence: bins of equal width, the gap between
    accuracy and mean confidence of each bin weighted by its share of the samples. None without samples."""
    total = len(confidences)
    if total == 0:
        return None
    count = [0] * bins
    confidence_sum = [0.0] * bins
    correct_sum = [0] * bins
    for confidence, correct in zip(confidences, corrects):
        index = min(max(int(float(confidence) * bins), 0), bins - 1)
        count[index] += 1
        confidence_sum[index] += float(confidence)
        correct_sum[index] += int(bool(correct))
    return sum(
        abs(correct_sum[i] / count[i] - confidence_sum[i] / count[i]) * count[i] / total
        for i in range(bins) if count[i]
    )


def reading_metrics(samples: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """The reader's accuracy, per-label support and confusion (evaluation.reader_metrics) plus calibration."""
    metrics = reader_metrics(samples, classes)
    confidences = [s["confidence"] for s in samples]
    metrics["expected_calibration_error"] = expected_calibration_error(
        confidences, [s["correct"] for s in samples])
    metrics["mean_confidence"] = sum(confidences) / len(confidences) if confidences else None
    return metrics


def ownership_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Confusion, per-class precision/recall/F1, other_road, abstain rate, macro-F1 and calibration.

    samples have ownership_truth, ownership_predicted and ownership_confidence. Precision is None for
    a class never predicted, recall for a class never true. F1 is 2TP / (true + predicted); macro-F1
    averages it over the classes true or predicted at least once (like scikit-learn). The abstain rate
    is the share predicted unresolved; decided_accuracy is the accuracy on the other predictions.
    """
    confusion = {truth: {predicted: 0 for predicted in OWNERSHIP_CLASSES} for truth in OWNERSHIP_CLASSES}
    for sample in samples:
        confusion[sample["ownership_truth"]][sample["ownership_predicted"]] += 1
    total = len(samples)
    per_class = {}
    f1_values = []
    for name in OWNERSHIP_CLASSES:
        true_positives = confusion[name][name]
        support = sum(confusion[name].values())
        predicted = sum(confusion[truth][name] for truth in OWNERSHIP_CLASSES)
        f1 = None
        if support or predicted:
            f1 = 2 * true_positives / (support + predicted)
            f1_values.append(f1)
        per_class[name] = {
            "support": support,
            "predicted": predicted,
            "true_positives": true_positives,
            "precision": true_positives / predicted if predicted else None,
            "recall": true_positives / support if support else None,
            "f1": f1,
        }
    correct = sum(confusion[name][name] for name in OWNERSHIP_CLASSES)
    abstained = sum(confusion[truth][ABSTAIN] for truth in OWNERSHIP_CLASSES)
    decided = total - abstained
    decided_correct = sum(confusion[name][name] for name in OWNERSHIP_CLASSES if name != ABSTAIN)
    confidences = [s["ownership_confidence"] for s in samples]
    return {
        "sample_count": total,
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "abstain_rate": abstained / total if total else None,
        "decided_accuracy": decided_correct / decided if decided else None,
        "other_road": dict(per_class["other_road"]),
        "per_class": per_class,
        "confusion": confusion,
        "expected_calibration_error": expected_calibration_error(
            confidences, [s["ownership_truth"] == s["ownership_predicted"] for s in samples]),
        "mean_confidence": sum(confidences) / total if total else None,
    }


def experiment_metrics(samples: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """Reading on every box, ownership on the boxes of the speed families."""
    owned = [s for s in samples if s.get("ownership_truth") is not None]
    return {
        "box_count": len(samples),
        "reading": reading_metrics(samples, classes),
        "ownership": ownership_metrics(owned),
    }


def partition_report(samples: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """experiment_metrics of a partition, and the same on its human-reviewed boxes (None when none)."""
    return {
        **experiment_metrics(samples, classes),
        "counts_by_label_basis": label_basis_counts(samples),
        "human_reviewed": human_reviewed_metrics(samples, lambda items: experiment_metrics(items, classes)),
    }


def selection_score(metrics: dict[str, Any]) -> float:
    """SELECTION_METRIC of experiment_metrics on validation (a missing other_road F1 counts 0)."""
    return (metrics["reading"]["accuracy"] + (metrics["ownership"]["other_road"]["f1"] or 0.0)) / 2


def _number(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}"


def comparison_table(run_root: Path) -> str:
    """The variants of one run side by side (<run_root>/<variant>/result.json): reading accuracy,
    other_road precision and recall, ownership macro-F1 and abstain rate, per COMPARISON_SCOPES.
    "-" is a measure without samples (e.g. other_road recall when no box is other_road)."""
    header = ("variant", "scope", "boxes", "read_acc", "other_P", "other_R", "own_mF1", "abstain")
    lines = [header]
    for variant in VARIANTS:
        path = Path(run_root) / variant / "result.json"
        if not path.is_file():
            lines.append((variant, "no result.json", "", "", "", "", "", ""))
            continue
        metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
        for scope, partition, human in COMPARISON_SCOPES:
            report = metrics.get(partition)
            if report is not None and human:
                report = report.get("human_reviewed")
            if report is None:
                lines.append((variant, scope, "0", "-", "-", "-", "-", "-"))
                continue
            ownership = report["ownership"]
            lines.append((
                variant, scope, str(report["box_count"]),
                _number(report["reading"]["accuracy"]),
                _number(ownership["other_road"]["precision"]),
                _number(ownership["other_road"]["recall"]),
                _number(ownership["macro_f1"]),
                _number(ownership["abstain_rate"]),
            ))
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip()
                     for line in lines) + "\n"


# ---------------------------------------------------------------- training (torch from here on)

@dataclass(frozen=True)
class ExperimentConfig:
    variant: str = "full"
    epochs: int = 20
    # Frames per batch: all the boxes of a frame go together, so the frame is decoded once.
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    crop_size: int = 96
    scene_width: int = 384
    scene_height: int = 216
    ownership_loss_weight: float = 1.0
    num_workers: int = DEFAULT_WORKERS
    seed: int = 0
    device: str = "cuda"
    pretrained: bool = False
    amp: bool = True
    # The <variant>-best.pt of the same variant of an earlier run to go on from (decider_start), None: ImageNet or
    # random weights (pretrained).
    init_from: str | None = None


def _require_experiment_stack():
    try:
        import torch
        import torchvision
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        from torchvision.ops import roi_align
        from torchvision.transforms import functional as TF
    except ImportError as exc:
        raise ValueError(
            "training_dependencies_missing: experiment B needs PyTorch/torchvision plus Pillow "
            "(the GPU environment .venv-gpu, scripts/setup_v100_training_env.sh)"
        ) from exc
    return torch, torchvision, Image, DataLoader, Dataset, MobileNet_V3_Small_Weights, mobilenet_v3_small, roi_align, TF


def _build_model(torch, mobilenet_v3_small, weights, roi_align, variant: str, num_classes: int):
    """The variant's network: crop backbone, optional geometry and scene branches, fusion MLP, two heads."""
    nn = torch.nn

    class ContextModel(nn.Module):
        def __init__(self):
            super().__init__()
            size = BACKBONE_CHANNELS
            # The detail branch: today's reader backbone on the crop (full-resolution pixels).
            self.crop_backbone = mobilenet_v3_small(weights=weights).features
            self.geometry = None
            self.scene_backbone = None
            self.scene_neck = None
            if variant in ("crop_geom", "full"):
                self.geometry = nn.Sequential(nn.Linear(len(GEOMETRY_FEATURES), GEOMETRY_SIZE), nn.ReLU(inplace=True))
                size += GEOMETRY_SIZE
            if variant == "full":
                # The context branch: the scaled-down frame, pooled at the box and over the whole map.
                self.scene_backbone = mobilenet_v3_small(weights=weights).features
                self.scene_neck = nn.Sequential(
                    nn.Conv2d(BACKBONE_CHANNELS, SCENE_CHANNELS, 1, bias=False),
                    nn.BatchNorm2d(SCENE_CHANNELS),
                    nn.Hardswish(inplace=True),
                )
                size += SCENE_CHANNELS * (SCENE_ROI_SIZE * SCENE_ROI_SIZE + 1)
            self.fusion = nn.Sequential(nn.Linear(size, FUSION_SIZE), nn.Hardswish(inplace=True), nn.Dropout(p=.2))
            self.reading_head = nn.Linear(FUSION_SIZE, num_classes)
            self.ownership_head = nn.Linear(FUSION_SIZE, len(OWNERSHIP_CLASSES))

        def forward(self, crops, geometry, scenes, rois):
            parts = [self.crop_backbone(crops).mean((2, 3)).float()]
            if self.geometry is not None:
                parts.append(self.geometry(geometry).float())
            if self.scene_backbone is not None:
                scene_map = self.scene_neck(self.scene_backbone(scenes)).float()
                # rois: (frame index in the batch, x1, y1, x2, y2) in scene pixels.
                local = roi_align(scene_map, rois.float(), output_size=(SCENE_ROI_SIZE, SCENE_ROI_SIZE),
                                  spatial_scale=1.0 / SCENE_STRIDE, sampling_ratio=2, aligned=True)
                parts.append(local.flatten(1))
                parts.append(scene_map.mean((2, 3))[rois[:, 0].long()])
            fused = self.fusion(torch.cat(parts, 1))
            return self.reading_head(fused), self.ownership_head(fused)

    return ContextModel()


def decider_start(torch, model, path: Path, variant: str, classes: list[str]) -> dict[str, Any]:
    """The weights of the same variant of an earlier run (nnslr experiment-b --init-from) in a fresh network: every
    layer of the same name and shape, the rows of the reading head by class name (a class the start lacks keeps the
    fresh row). Returns what result.json records of it: its file and sha256, run, epoch, split, the layers copied and
    the new classes."""
    path = Path(path)
    saved = torch_load(path)
    if saved.get("task") != TASK or saved.get("variant") != variant:
        raise ValueError(f"the start is not the {variant} variant of experiment B: {path}")
    weights = model_weights(saved)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    head = "reading_head."
    copied = [key for key in state if not key.startswith(head) and key in weights
              and tuple(weights[key].shape) == tuple(state[key].shape)]
    fresh = [key for key in state if not key.startswith(head) and key not in copied]
    if fresh:
        raise ValueError(f"the start's layers are not those of this network ({', '.join(fresh[:3])}): {path}")
    for key in copied:
        state[key] = weights[key]
    rows = {label: row for row, label in enumerate(saved.get("classes") or [])}
    for row, label in enumerate(classes):
        if label in rows:
            state[head + "weight"][row] = weights[head + "weight"][rows[label]]
            state[head + "bias"][row] = weights[head + "bias"][rows[label]]
    model.load_state_dict(state, strict=True)
    return {"path": str(path), "sha256": _sha256(path), "run": path.parents[2].name if len(path.parents) > 2 else None,
            "variant": variant, "epoch": saved.get("epoch"), "split_sha256": saved.get("split_sha256"),
            "dataset_sha256": saved.get("dataset_sha256"),
            "new_classes": [label for label in classes if label not in rows]}


class Decider:
    """A trained variant of experiment B (<variant>-best.pt) used as the decider: whom each box of a frame applies to,
    own_road, other_road or unresolved (it abstains), with the probability of that answer. Prepare's "ours" proposer
    runs it on the boxes the classifier kept (nnslr_tools._chain_proposer). The inputs are made as in the training:
    the same crop and margin, geometry, speed and scene, resized with PIL from the full frame."""

    def __init__(self, path: Path, device):
        (torch, _, Image, _, _, _, mobilenet_v3_small, roi_align, TF) = _require_experiment_stack()
        checkpoint = torch_load(Path(path))
        if checkpoint.get("task") != TASK or checkpoint.get("variant") not in VARIANTS:
            raise ValueError(f"not_a_decider: {path} is not a checkpoint of experiment B")
        self.torch, self.TF, self.bilinear, self.device = torch, TF, Image.BILINEAR, device
        self.variant = checkpoint["variant"]
        self.families = tuple(checkpoint.get("ownership_families") or OWNERSHIP_FAMILIES)
        self.ownership_classes = list(checkpoint.get("ownership_classes") or OWNERSHIP_CLASSES)
        self.crop_size = int(checkpoint.get("crop_size", 96))
        self.crop_margin = float(checkpoint.get("crop_margin", CROP_MARGIN))
        self.scene_width, self.scene_height = (int(v) for v in checkpoint.get("scene_size", (384, 216)))
        self.speed_scale = float(checkpoint.get("speed_scale_mps", SPEED_SCALE_MPS))
        self.mean = torch.tensor(checkpoint.get("normalize_mean", READER_NORMALIZE_MEAN)).view(3, 1, 1)
        self.std = torch.tensor(checkpoint.get("normalize_std", READER_NORMALIZE_STD)).view(3, 1, 1)
        self.model = _build_model(torch, mobilenet_v3_small, None, roi_align, self.variant, len(checkpoint["classes"]))
        self.model.load_state_dict(model_weights(checkpoint), strict=True)
        self.model.to(device).eval()

    def decide(self, image, boxes: list[list[float]], speed_mps: float | None) -> list[tuple[str, float]]:
        """(applicability, probability) of each box (x1, y1, x2, y2 in the pixels of ``image``, a PIL RGB frame):
        all the boxes of the frame in one call."""
        if not boxes:
            return []
        torch, TF = self.torch, self.TF
        width, height = image.size
        crops, geometry, rois = [], [], []
        scale_x, scale_y = self.scene_width / width, self.scene_height / height
        for raw in boxes:
            box = clip_box(raw, width, height)
            region = crop_box(box, width, height, self.crop_margin)
            crop = image.resize((self.crop_size, self.crop_size), self.bilinear, box=tuple(region))
            crops.append(TF.pil_to_tensor(crop))
            features = geometry_features(box, width, height, speed_mps)
            if speed_mps is not None:
                features[-2] = speed_mps / self.speed_scale
            geometry.append(features)
            rois.append([0.0, box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
        crop_tensor = (torch.stack(crops).float().div_(255.0) - self.mean) / self.std
        scenes = roi_tensor = None
        if self.variant == "full":
            scene = TF.pil_to_tensor(image.resize((self.scene_width, self.scene_height), self.bilinear))
            scene = scene.float().div_(255.0)
            scenes = ((scene - self.mean) / self.std).unsqueeze(0).to(self.device)
            roi_tensor = torch.tensor(rois, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, ownership_logits = self.model(crop_tensor.to(self.device),
                                             torch.tensor(geometry, dtype=torch.float32).to(self.device),
                                             scenes, roi_tensor)
            probabilities = torch.softmax(ownership_logits.float(), 1).cpu()
        confidence, index = probabilities.max(1)
        return [(self.ownership_classes[int(i)], float(c)) for i, c in zip(index.tolist(), confidence.tolist())]


def _frames_of(records: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    """The frames of a partition, each with the indices (into records) of its boxes."""
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record["split"] == partition:
            grouped.setdefault(record["image_path"], []).append(index)
    return [{"image_path": path, "records": indices} for path, indices in sorted(grouped.items())]


def train_experiment_b(
    plan: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Train one variant of experiment B, keep its best validation epoch and measure it.

    Writes into the new output_dir: <variant>-best.pt, experiment-plan.json, samples.jsonl (every
    validation and test box with its predictions) and result.json.
    """
    if config.variant not in VARIANTS:
        raise ValueError(f"unknown_variant: {config.variant}")
    if not plan.get("trainable"):
        raise ValueError(
            "experiment_plan_not_trainable: need train and validation boxes, >=2 train reading labels, "
            "ownership rows in train and every image present (see --dry-run)"
        )

    (torch, torchvision, Image, DataLoader, Dataset, Weights, mobilenet_v3_small,
     roi_align, TF) = _require_experiment_stack()

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    variant = config.variant
    task_name = f"{TASK}_{variant}"
    architecture = f"context_{variant}_mobilenet_v3_small"
    classes = list(plan["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    ownership_to_index = {name: index for index, name in enumerate(OWNERSHIP_CLASSES)}
    records = plan["records"]
    root = Path(data_root).resolve()
    use_scene = variant == "full"
    mean = torch.tensor(READER_NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(READER_NORMALIZE_STD).view(3, 1, 1)

    def colour(tensor, params):
        # The same draw for the crops and the scene of a frame (tensors in 0..1, batched or not).
        brightness, contrast, saturation = params
        tensor = TF.adjust_brightness(tensor, brightness)
        tensor = TF.adjust_contrast(tensor, contrast)
        return TF.adjust_saturation(tensor, saturation)

    class FrameDataset(Dataset):
        """One item per frame with all its boxes: the PNG is decoded once for crops and scene."""

        def __init__(self, frames, *, training: bool):
            self.frames = frames
            self.training = training

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, index):
            frame = self.frames[index]
            path = _safe_image_path(root, frame["image_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
            width, height = image.size
            scale_x = config.scene_width / width
            scale_y = config.scene_height / height
            crops, geometry, scene_boxes, reading, ownership = [], [], [], [], []
            for record_index in frame["records"]:
                record = records[record_index]
                box = clip_box(record["bbox_xyxy"], width, height)
                if self.training:
                    box = jitter_box(box, width, height, random)
                region = crop_box(box, width, height)
                crop = image.resize((config.crop_size, config.crop_size), Image.BILINEAR, box=tuple(region))
                crops.append(TF.pil_to_tensor(crop))
                geometry.append(geometry_features(box, width, height, record["speed_mps"]))
                scene_boxes.append([box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y])
                reading.append(class_to_index.get(record["label"], IGNORE_INDEX))
                ownership.append(ownership_to_index[record["ownership"]] if record["ownership"] is not None
                                 else IGNORE_INDEX)
            params = None
            if self.training:
                params = tuple(random.uniform(1 - COLOUR_JITTER[name], 1 + COLOUR_JITTER[name])
                               for name in ("brightness", "contrast", "saturation"))
            crop_tensor = torch.stack(crops).float().div_(255.0)
            if params is not None:
                crop_tensor = colour(crop_tensor, params)
            crop_tensor = (crop_tensor - mean) / std
            if use_scene:
                scene = TF.pil_to_tensor(image.resize((config.scene_width, config.scene_height), Image.BILINEAR))
                scene = scene.float().div_(255.0)
                if params is not None:
                    scene = colour(scene, params)
                scene = (scene - mean) / std
            else:
                scene = torch.zeros(0)
            return (
                crop_tensor,
                torch.tensor(geometry, dtype=torch.float32),
                scene,
                torch.tensor(scene_boxes, dtype=torch.float32),
                torch.tensor(reading, dtype=torch.int64),
                torch.tensor(ownership, dtype=torch.int64),
                torch.tensor(frame["records"], dtype=torch.int64),
            )

    def collate(items):
        crops, geometry, scenes, boxes, reading, ownership, indices = zip(*items)
        rois = None
        scene_batch = None
        if use_scene:
            scene_batch = torch.stack(scenes)
            rois = torch.cat([
                torch.cat([torch.full((len(frame_boxes), 1), float(frame_index)), frame_boxes], 1)
                for frame_index, frame_boxes in enumerate(boxes)
            ])
        return (torch.cat(crops), torch.cat(geometry), scene_batch, rois,
                torch.cat(reading), torch.cat(ownership), torch.cat(indices))

    train_frames = _frames_of(records, TRAIN_SPLIT)
    evaluation_frames = {name: _frames_of(records, name) for name in (VALIDATION_SPLIT, TEST_SPLIT)}
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        FrameDataset(train_frames, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        generator=generator,
    )
    evaluation_loaders = {
        name: DataLoader(
            FrameDataset(frames, training=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        for name, frames in evaluation_frames.items() if frames
    }

    weights = Weights.IMAGENET1K_V1 if config.pretrained else None
    model = _build_model(torch, mobilenet_v3_small, weights, roi_align, variant, len(classes))
    start = None
    if config.init_from:
        start = decider_start(torch, model, Path(config.init_from), variant, classes)
        print(f"{task_name} starts from {start['run']}: every layer, and the reading of "
              f"{len(classes) - len(start['new_classes'])} of its {len(classes)} classes"
              + (f" (new: {', '.join(start['new_classes'])})" if start["new_classes"] else "")
              + ("" if start["split_sha256"] == plan["split_sha256"] else "; learnt on another split"), flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    class_weights = torch.tensor([plan["ownership_class_weights"][name] for name in OWNERSHIP_CLASSES],
                                 dtype=torch.float32, device=device)
    functional = torch.nn.functional

    def losses(reading_logits, ownership_logits, reading, ownership):
        # Reading: mean over the boxes with a train label. Ownership: class-weighted mean over the
        # boxes of the speed families (0 when the batch has none).
        reading_logits = reading_logits.float()
        ownership_logits = ownership_logits.float()
        reading_count = (reading != IGNORE_INDEX).sum().clamp_min(1)
        reading_loss = functional.cross_entropy(
            reading_logits, reading, ignore_index=IGNORE_INDEX, reduction="sum") / reading_count
        owned = ownership != IGNORE_INDEX
        ownership_weight = class_weights[ownership[owned]].sum().clamp_min(1e-8)
        ownership_loss = functional.cross_entropy(
            ownership_logits, ownership, weight=class_weights, ignore_index=IGNORE_INDEX,
            reduction="sum") / ownership_weight
        total = reading_loss + config.ownership_loss_weight * ownership_loss
        return total, reading_loss, ownership_loss

    def to_device(batch):
        crops, geometry, scenes, rois, reading, ownership, indices = batch
        crops = crops.to(device, non_blocking=True)
        geometry = geometry.to(device, non_blocking=True)
        if scenes is not None:
            scenes = scenes.to(device, non_blocking=True)
            rois = rois.to(device, non_blocking=True)
        return (crops, geometry, scenes, rois, reading.to(device, non_blocking=True),
                ownership.to(device, non_blocking=True), indices)

    def predict(partition: str) -> tuple[list[dict[str, Any]], float]:
        """Every box of the partition with its predictions, and the mean loss."""
        model.eval()
        samples: list[dict[str, Any]] = []
        loss_total = 0.0
        seen = 0
        with torch.no_grad():
            for batch in evaluation_loaders[partition]:
                crops, geometry, scenes, rois, reading, ownership, indices = to_device(batch)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    reading_logits, ownership_logits = model(crops, geometry, scenes, rois)
                loss, _, _ = losses(reading_logits, ownership_logits, reading, ownership)
                count = int(reading.numel())
                loss_total += float(loss.detach()) * count
                seen += count
                reading_probabilities = torch.softmax(reading_logits.float(), 1).cpu()
                ownership_probabilities = torch.softmax(ownership_logits.float(), 1).cpu()
                reading_confidence, reading_index = reading_probabilities.max(1)
                ownership_confidence, ownership_index = ownership_probabilities.max(1)
                for position, record_index in enumerate(indices.tolist()):
                    record = records[record_index]
                    predicted = classes[int(reading_index[position])]
                    samples.append({
                        "annotation_id": record["annotation_id"],
                        "partition": partition,
                        "image_path": record["image_path"],
                        "label_basis": record["label_basis"],
                        "sign_family": record["sign_family"],
                        "truth": record["label"],
                        "predicted": predicted,
                        "confidence": float(reading_confidence[position]),
                        "correct": predicted == record["label"],
                        "ownership_truth": record["ownership"],
                        "ownership_predicted": OWNERSHIP_CLASSES[int(ownership_index[position])],
                        "ownership_confidence": float(ownership_confidence[position]),
                        "ownership_probabilities": {
                            name: float(ownership_probabilities[position, i])
                            for i, name in enumerate(OWNERSHIP_CLASSES)
                        },
                    })
        return samples, loss_total / max(seen, 1)

    initialization = "start" if start else "imagenet1k_v1" if config.pretrained else "random"
    initialization_source = (
        start["path"] if start else
        "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1" if config.pretrained else "random"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "experiment-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    best_path = output_dir / f"{variant}-best.pt"

    tracker = tracking.start_training_run(task_name, output_dir, {
        **config.__dict__,
        "task": TASK,
        "architecture": architecture,
        "inputs": VARIANT_INPUTS[variant],
        "dataset_kind": plan["dataset_kind"],
        "dataset_sha256": plan["dataset_sha256"],
        "split_sha256": plan["split_sha256"],
        "classes": classes,
        "ownership_class_weights": plan["ownership_class_weights"],
        "boxes_by_partition": plan["counts"]["boxes_by_partition"],
    })
    history = []
    best_score = -1.0
    best_epoch = 0
    global_step = 0
    try:
        for epoch in range(config.epochs):
            epoch_number = epoch + 1
            epoch_start = time.perf_counter()
            generator.manual_seed(config.seed + epoch)
            model.train()
            train_loss = 0.0
            train_seen = 0
            train_correct = 0
            train_labelled = 0
            # The step time includes the wait for the batch, like distill's step_time.
            step_start = time.perf_counter()
            batch_count = len(train_loader)
            batch_index = 0
            for batch in train_loader:
                # Where the training is, batch by batch, for the panel of the review site.
                batch_index += 1
                report_progress(task_name, epoch * batch_count + batch_index, config.epochs * batch_count,
                                f"epoch {epoch_number} of {config.epochs}, batch {batch_index} of {batch_count}")
                crops, geometry, scenes, rois, reading, ownership, _ = to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    reading_logits, ownership_logits = model(crops, geometry, scenes, rois)
                loss, reading_loss, ownership_loss = losses(reading_logits, ownership_logits, reading, ownership)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_loss = float(loss.detach())
                step_seen = int(reading.numel())
                labelled = reading != IGNORE_INDEX
                step_correct = int((reading_logits.argmax(1)[labelled] == reading[labelled]).sum())
                train_loss += step_loss * step_seen
                train_seen += step_seen
                train_correct += step_correct
                train_labelled += int(labelled.sum())
                global_step += 1
                step_end = time.perf_counter()
                if tracker.enabled:
                    tracker.log({
                        "train/loss": step_loss,
                        "train/reading_loss": float(reading_loss.detach()),
                        "train/ownership_loss": float(ownership_loss.detach()),
                        "train/reading_accuracy": step_correct / max(int(labelled.sum()), 1),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/step_time": step_end - step_start,
                    }, step=global_step)
                step_start = step_end

            validation_samples, validation_loss = predict(VALIDATION_SPLIT)
            validation = experiment_metrics(validation_samples, classes)
            score = selection_score(validation)
            other_road = validation["ownership"]["other_road"]
            epoch_result = {
                "epoch": epoch_number,
                "train_loss": train_loss / max(train_seen, 1),
                "train_accuracy": train_correct / max(train_labelled, 1),
                "validation_loss": validation_loss,
                "validation_accuracy": validation["reading"]["accuracy"],
                "validation_ownership_macro_f1": validation["ownership"]["macro_f1"],
                "validation_other_road_precision": other_road["precision"],
                "validation_other_road_recall": other_road["recall"],
                "validation_other_road_f1": other_road["f1"],
                "validation_selection_score": score,
            }
            history.append(epoch_result)

            if score > best_score:
                best_score = score
                best_epoch = epoch_number
                atomic_torch_save({
                    "format_version": 1,
                    "architecture": architecture,
                    "task": TASK,
                    "variant": variant,
                    "inputs": VARIANT_INPUTS[variant],
                    "classes": classes,
                    "ownership_classes": list(OWNERSHIP_CLASSES),
                    "ownership_families": list(OWNERSHIP_FAMILIES),
                    "geometry_features": list(GEOMETRY_FEATURES),
                    "crop_size": config.crop_size,
                    "crop_margin": CROP_MARGIN,
                    "scene_size": [config.scene_width, config.scene_height],
                    "speed_scale_mps": SPEED_SCALE_MPS,
                    "normalize_mean": list(READER_NORMALIZE_MEAN),
                    "normalize_std": list(READER_NORMALIZE_STD),
                    "state_dict": model.state_dict(),
                    "dataset_sha256": plan["dataset_sha256"],
                    "split_sha256": plan["split_sha256"],
                    "epoch": epoch_number,
                    "validation_selection_score": best_score,
                    "selection_metric": SELECTION_METRIC,
                    "initialization": initialization,
                    "initialization_source": initialization_source,
                    # Plain strings: torch.__version__ is a TorchVersion, which weights_only loading refuses.
                    "torch_version": str(torch.__version__),
                    "torchvision_version": str(torchvision.__version__),
                }, best_path)

            if tracker.enabled:
                tracker.log({
                    "epoch": epoch_number,
                    "train/epoch_loss": epoch_result["train_loss"],
                    "train/epoch_reading_accuracy": epoch_result["train_accuracy"],
                    "validation/loss": validation_loss,
                    "validation/reading_accuracy": epoch_result["validation_accuracy"],
                    "validation/ownership_macro_f1": epoch_result["validation_ownership_macro_f1"],
                    "validation/other_road_precision": other_road["precision"],
                    "validation/other_road_recall": other_road["recall"],
                    "validation/other_road_f1": other_road["f1"],
                    "validation/abstain_rate": validation["ownership"]["abstain_rate"],
                    "validation/selection_score": score,
                    "validation/best_selection_score": best_score,
                    "epoch_time": time.perf_counter() - epoch_start,
                }, step=global_step)
            # One line per epoch, in the format the panel of the review site reads (job_progress).
            print(f"{task_name} epoch {epoch_number}/{config.epochs} train_loss {epoch_result['train_loss']:.4f} "
                  f"val_loss {validation_loss:.4f} "
                  f"train_accuracy {epoch_result['train_accuracy']:.4f} "
                  f"val_accuracy {epoch_result['validation_accuracy']:.4f} "
                  f"val_ownership_macro_f1 {_number(epoch_result['validation_ownership_macro_f1'])} "
                  f"val_other_road_recall {_number(other_road['recall'])} "
                  f"val_other_road_f1 {_number(other_road['f1'])} "
                  f"val_selection_score {score:.4f} "
                  f"{time.perf_counter() - epoch_start:.0f}s", flush=True)

        # The measure is the best validation epoch, reloaded from its checkpoint.
        model.load_state_dict(model_weights(torch_load(best_path)), strict=True)
        metrics: dict[str, Any] = {}
        samples_path = output_dir / "samples.jsonl"
        with samples_path.open("w", encoding="utf-8") as handle:
            for partition in (VALIDATION_SPLIT, TEST_SPLIT):
                if partition not in evaluation_loaders:
                    metrics[partition] = None
                    continue
                samples, loss = predict(partition)
                metrics[partition] = {"loss": loss, **partition_report(samples, classes)}
                for sample in samples:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")

        gpu = None
        if device.type == "cuda":
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            }
        result = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "task": TASK,
            "variant": variant,
            "inputs": VARIANT_INPUTS[variant],
            "architecture": architecture,
            "classes": classes,
            "ownership_classes": list(OWNERSHIP_CLASSES),
            "ownership_families": list(OWNERSHIP_FAMILIES),
            "ownership_class_weights": plan["ownership_class_weights"],
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "selection_metric": SELECTION_METRIC,
            "metrics": metrics,
            "counts": plan["counts"],
            "unseen_labels": plan["unseen_labels"],
            "ignored_partitions": plan["ignored_partitions"],
            "dataset_kind": plan["dataset_kind"],
            "dataset_path": plan["dataset_path"],
            "dataset_sha256": plan["dataset_sha256"],
            "split_path": plan["split_path"],
            "split_sha256": plan["split_sha256"],
            "split_policy": plan["split_policy"],
            "limit": plan["limit"],
            "checkpoint": str(best_path),
            "experiment_plan": str(plan_path),
            "samples": str(samples_path),
            "device": str(device),
            "gpu": gpu,
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
            "initialization": initialization,
            "initialization_source": initialization_source,
            "start": start,
            "config": config.__dict__,
            "history": history,
            "warnings": plan["warnings"],
            "limitations": plan["limitations"],
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        test = metrics.get(TEST_SPLIT)
        tracker.set_summary({
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "checkpoint": str(best_path),
            "test/reading_accuracy": test["reading"]["accuracy"] if test else None,
            "test/ownership_macro_f1": test["ownership"]["macro_f1"] if test else None,
            "test/other_road_recall": test["ownership"]["other_road"]["recall"] if test else None,
        })
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result
