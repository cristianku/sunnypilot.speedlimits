"""Offline evaluation for NNSLR reader and detector baselines.

Evaluation consumes the same hash-bound dataset/split plan used by training.
Torch is imported only by an explicit non-dry-run evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from nnslr_tools.checkpoints import (detector_anchor_min_ratio, detector_input_size, format_input_size,
                                     load_detector, safe_torch_load)
from nnslr_tools.training import (
    DEFAULT_WORKERS,
    DETECTOR_FAMILIES,
    NEGATIVE_SOURCES,
    NOT_A_SIGN,
    _require_detector_stack,
    _require_training_stack,
    _safe_image_path,
    checkpoint_class_scheme,
    crop_rectangle,
    crop_source,
    cutter_families,
    family_label,
)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


# How a found box can match a reviewed box: "iou", the standard rule (IoU at least iou_threshold), or "centre", the
# lenient one for far, small signs, where a box 2 px tighter on each side of a 10 px sign has IoU 0.36 though it sits
# on the sign: IoU at least iou_threshold, or the found box's centre inside the reviewed box and its width and height
# each between CENTRE_SIZE_RATIO[0] and CENTRE_SIZE_RATIO[1] times the reviewed box's (centre_match).
MATCH_RULES = ("iou", "centre")
CENTRE_SIZE_RATIO = (0.5, 2.0)


def centre_match(box: list[float], truth_box: list[float]) -> bool:
    """The lenient match of a found box on a reviewed box: its centre inside the reviewed box (edges included) and
    its width and height each 0.5 to 2 times the reviewed box's. A huge box around a small sign does not match."""
    x1, y1, x2, y2 = (float(value) for value in box)
    tx1, ty1, tx2, ty2 = (float(value) for value in truth_box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if not (tx1 <= cx <= tx2 and ty1 <= cy <= ty2):
        return False
    low, high = CENTRE_SIZE_RATIO
    truth_w, truth_h = tx2 - tx1, ty2 - ty1
    if truth_w <= 0 or truth_h <= 0:
        return False
    return low <= (x2 - x1) / truth_w <= high and low <= (y2 - y1) / truth_h <= high


def _check_match(match: str) -> None:
    if match not in MATCH_RULES:
        raise ValueError(f"invalid_match: {match!r}, one of {', '.join(MATCH_RULES)}")


def greedy_matching(
    predicted: list[dict[str, Any]],
    truth: list[list[float]],
    *,
    iou_threshold: float = 0.5,
    match: str = "iou",
) -> list[tuple[int, int, float]]:
    """The greedy score-ordered one-to-one matching of match_detections: (position in predicted, index of the
    reviewed box it found or -1, best IoU) of every prediction, best score first.

    A prediction is matched against the reviewed boxes left by those of a higher score only, so the matching of the
    predictions at or above a score is this matching restricted to them: the Watch panel of the Evaluate tab draws
    any score from the boxes matched once at the lowest (per_frame_record).

    With match "centre" (MATCH_RULES), a prediction whose best IoU is under iou_threshold still finds the reviewed box
    of highest IoU among those it centre_match-es; the IoU given is then that box's."""
    _check_match(match)
    ordered = sorted(enumerate(predicted), key=lambda item: float(item[1]["score"]), reverse=True)
    unmatched = set(range(len(truth)))
    result = []
    for position, prediction in ordered:
        box = prediction["box"]
        candidates = [
            (box_iou(box, truth[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            result.append((position, best_index, best_iou))
            continue
        if match == "centre":
            centred = [(iou, index) for iou, index in candidates if centre_match(box, truth[index])]
            if centred:
                centre_iou, centre_index = max(centred)
                unmatched.remove(centre_index)
                result.append((position, centre_index, centre_iou))
                continue
        result.append((position, -1, best_iou))
    return result


def match_detections(
    predicted: list[dict[str, Any]],
    truth: list[list[float]],
    *,
    iou_threshold: float = 0.5,
    match: str = "iou",
) -> dict[str, Any]:
    """Greedy score-ordered one-to-one matching for a single detector class (match: MATCH_RULES)."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("invalid_iou_threshold")
    matches = []
    false_positives = []
    false_positive_indexes = []

    for position, truth_index, best_iou in greedy_matching(predicted, truth, iou_threshold=iou_threshold, match=match):
        if truth_index >= 0:
            matches.append({
                "score": float(predicted[position]["score"]),
                "iou": best_iou,
                "truth_index": truth_index,
            })
        else:
            false_positives.append({
                "score": float(predicted[position]["score"]),
                "best_iou": best_iou,
            })
            false_positive_indexes.append(position)

    return {
        "true_positives": len(matches),
        "false_positives": len(false_positives),
        "false_negatives": len(truth) - len(matches),
        "matches": matches,
        "false_positive_details": false_positives,
        # Where the false positives are in ``predicted`` (score order, like false_positive_details).
        "false_positive_indexes": false_positive_indexes,
    }


# The sign sizes of the detector's recall: the width of the reviewed box in pixels of the frame. The same
# buckets as the Evaluate tab of the site (nnslr_tools.runs.SIZE_BUCKETS).
WIDTH_BUCKETS = (("lt12", 0.0, 12.0), ("12-20", 12.0, 20.0), ("20-40", 20.0, 40.0), ("ge40", 40.0, math.inf))
# Score thresholds a detector report always measures, besides its own --score-threshold. The detector is the
# cutter of the pipeline (cutter -> classifier -> decider): what it misses is lost, its false boxes the classifier
# can still reject, so it is judged first on recall at the low thresholds, with the boxes per frame it passes on.
CUTTER_SCORE_THRESHOLDS = (0.05, 0.1)
REPORT_SCORE_THRESHOLDS = (*CUTTER_SCORE_THRESHOLDS, 0.3, 0.5)
# Forward passes on the first frame before the time per frame is measured (CUDA kernels, cuDNN, allocator).
INFERENCE_WARMUP_RUNS = 3

# The cutter's boxes of every frame, kept in a detector or chain report for the Watch panel of the Evaluate tab
# (nnslr_tools.runs.frames_digest), which draws them over the frame at any score of its slider: the boxes of score
# PER_FRAME_MIN_SCORE and more, at most PER_FRAME_MAX_BOXES a frame (every box that found a reviewed box, then the
# best others: a cutter passes on a couple of hundred boxes a frame at 0.05), and how many boxes the cutter passed on
# at each step of the slider, kept or not.
PER_FRAME_MIN_SCORE = 0.05
PER_FRAME_MAX_BOXES = 50
PER_FRAME_STEPS = tuple(step / 100 for step in range(5, 55, 5))


def _score_down(score: float) -> float:
    """A score to 3 decimals, rounded down: compared with a step of the slider it stays on the same side."""
    return math.floor(round(float(score) * 1000, 6)) / 1000


def per_frame_descriptor(*, chain: bool, min_score: float) -> dict[str, Any]:
    """The ``per_frame_boxes`` of a report: how per_frame_record wrote the boxes of each frame."""
    return {
        "min_score": float(min_score),
        "max_boxes": PER_FRAME_MAX_BOXES,
        "steps": list(PER_FRAME_STEPS),
        "box": "x1 y1 x2 y2 score truth" + (" label confidence" if chain else ""),
        "fields": {
            "x1 y1 x2 y2": "the box in pixels of the frame, rounded",
            "score": "the cutter's score, 3 decimals rounded down",
            "truth": ("the reviewed box it found (an index of " + ("truth" if chain else "truth_boxes")
                      + "), matched at min_score as at every score, -1 for none"),
            **({"label confidence": "what the classifier called it, and its probability"} if chain else {}),
        },
        "kept": ("cutter_boxes: every box of min_score or more that found a reviewed box, then the best others"
                 + (" (first those the classifier let through)" if chain else "") + ", at most max_boxes a frame"),
        "counts": ("cutter_box_counts: the boxes of the cutter at or above each step, kept or not"
                   + ("; surviving_counts: of those on no reviewed box, the ones the classifier did not reject" if chain else "")
                   + "; a step under min_score counts the boxes from min_score only"),
    }


def per_frame_record(predictions: list[dict[str, Any]], pairs: dict[int, int], *, chain: bool = False) -> dict[str, Any]:
    """What a report keeps of one frame for the Watch panel (see per_frame_descriptor): predictions are the cutter's
    boxes of the report's min_score or more ({"box", "score"}, in a chain also "predicted" and "confidence"), pairs
    {position in predictions: index of the reviewed box it found} their matching at that score."""
    def survives(position: int) -> bool:
        return chain and position not in pairs and predictions[position]["predicted"] != NOT_A_SIGN

    others = sorted((position for position in range(len(predictions)) if position not in pairs),
                    key=lambda position: (not survives(position), -float(predictions[position]["score"])))
    kept = [position for position in range(len(predictions)) if position in pairs]
    kept += others[:max(0, PER_FRAME_MAX_BOXES - len(kept))]
    kept.sort(key=lambda position: -float(predictions[position]["score"]))
    lines = []
    for position in kept:
        item = predictions[position]
        line = " ".join(str(int(round(float(value)))) for value in item["box"])
        line += f" {_score_down(item['score']):.3f} {pairs.get(position, -1)}"
        if chain:
            line += f" {item['predicted']} {float(item['confidence']):.3f}"
        lines.append(line)
    record: dict[str, Any] = {
        "cutter_boxes": lines,
        "cutter_box_counts": [sum(1 for item in predictions if item["score"] >= step) for step in PER_FRAME_STEPS],
    }
    if chain:
        record["surviving_counts"] = [sum(1 for position, item in enumerate(predictions)
                                          if survives(position) and item["score"] >= step) for step in PER_FRAME_STEPS]
    return record


def width_bucket(width: float) -> str:
    for key, low, high in WIDTH_BUCKETS:
        if low <= width < high:
            return key
    return WIDTH_BUCKETS[-1][0]


def score_frame(truth: list[list[float]], predictions: list[dict[str, Any]], *, score_threshold: float,
                iou_threshold: float, match: str = "iou") -> dict[str, Any]:
    """The predictions of one frame at or above score_threshold, matched to its reviewed boxes: the matching of
    match_detections (match: MATCH_RULES) plus the reviewed boxes found and the false boxes, for recall_by_width."""
    selected = [prediction for prediction in predictions if prediction["score"] >= score_threshold]
    matched = match_detections(selected, truth, iou_threshold=iou_threshold, match=match)
    return {
        **matched,
        "selected": selected,
        "truth_boxes": truth,
        "found_truth_indexes": sorted(match["truth_index"] for match in matched["matches"]),
        "false_boxes": [selected[index]["box"] for index in matched["false_positive_indexes"]],
    }


def recall_by_width(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recall by the width of the reviewed box (WIDTH_BUCKETS), and the false boxes by their own width, of
    frames scored by score_frame. recall is None in a bucket without reviewed boxes."""
    rows = {key: {"truth": 0, "found": 0, "false_positives": 0} for key, _, _ in WIDTH_BUCKETS}
    for item in items:
        found = set(item["found_truth_indexes"])
        for index, box in enumerate(item["truth_boxes"]):
            row = rows[width_bucket(float(box[2]) - float(box[0]))]
            row["truth"] += 1
            row["found"] += int(index in found)
        for box in item["false_boxes"]:
            rows[width_bucket(float(box[2]) - float(box[0]))]["false_positives"] += 1
    return {
        key: {**row, "missed": row["truth"] - row["found"],
              "recall": row["found"] / row["truth"] if row["truth"] else None}
        for key, row in rows.items()
    }


# The labels a person confirmed (provenance.label_basis); most labels are AI reviews.
HUMAN_LABEL_BASIS = "human_review"


def label_basis_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    """Evaluated samples (reader) or images (detector) by who confirmed their labels."""
    counts = Counter(str(item.get("label_basis") or "unknown") for item in items)
    return dict(sorted(counts.items()))


def reader_metrics(samples: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """Accuracy, per-class support and confusion of reader samples (truth, predicted, correct)."""
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    support = Counter()
    correct = Counter()
    for sample in samples:
        support[sample["truth"]] += 1
        confusion[sample["truth"]][sample["predicted"]] += 1
        if sample["correct"]:
            correct[sample["truth"]] += 1
    total = len(samples)
    return {
        "sample_count": total,
        "accuracy": sum(correct.values()) / total if total else 0.0,
        "unseen_truth_labels": sorted(set(support) - set(classes)),
        "per_class": {
            label: {
                "support": support[label],
                "correct": correct[label],
                "accuracy": correct[label] / support[label],
            }
            for label in sorted(support)
        },
        "confusion": {
            truth: dict(sorted(predictions.items()))
            for truth, predictions in sorted(confusion.items())
        },
    }


def detector_metrics(images: list[dict[str, Any]]) -> dict[str, Any]:
    """IoU-based precision, recall, F1 and false positives per image of per-image detector results."""
    tp = sum(image["true_positives"] for image in images)
    fp = sum(image["false_positives"] for image in images)
    fn = sum(image["false_negatives"] for image in images)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "image_count": len(images),
        "truth_box_count": tp + fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_iou": precision,
        "recall_iou": recall,
        "f1_iou": f1,
        "false_positives_per_image": fp / len(images) if images else 0.0,
        # Every box the detector passes on (found or false): the work of the classifier after it.
        "predictions_per_image": (tp + fp) / len(images) if images else 0.0,
    }


def family_group(family: str) -> str:
    """The family groups of a cutter's recall: "speed" (the speed signs, what every cutter finds) or the family."""
    return "speed" if family in DETECTOR_FAMILIES else str(family)


def recall_by_family(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
    """Found reviewed boxes by family group (family_group), of frames scored by score_frame that carry the family of
    each reviewed box (``truth_families``); None when they do not. A cutter of the speed and the other signs is
    compared with the older speed-only ones on "speed"."""
    if not items or not all(isinstance(item.get("truth_families"), list) for item in items):
        return None
    rows: dict[str, dict[str, int]] = {}
    for item in items:
        found = set(item["found_truth_indexes"])
        for index, family in enumerate(item["truth_families"]):
            row = rows.setdefault(family_group(family), {"truth": 0, "found": 0})
            row["truth"] += 1
            row["found"] += int(index in found)
    return {key: {**row, "recall": row["found"] / row["truth"] if row["truth"] else None}
            for key, row in sorted(rows.items())}


def detector_size_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """detector_metrics plus recall_by_width (and recall_by_family when the frames know it), of frames scored by
    score_frame."""
    measured = {**detector_metrics(items), "recall_by_width": recall_by_width(items)}
    by_family = recall_by_family(items)
    if by_family is not None:
        measured["recall_by_family"] = by_family
    return measured


def threshold_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """detector_size_metrics of all the frames and of the frames a person confirmed (None without any)."""
    return {**detector_size_metrics(items), "human_reviewed": human_reviewed_metrics(items, detector_size_metrics)}


def human_reviewed_metrics(items: list[dict[str, Any]], metrics) -> dict[str, Any] | None:
    """The same metrics on the samples or images a person confirmed; None when there are none.

    About nine labels out of ten are AI reviews: this is the measure on human labels only.
    """
    human = [item for item in items if item.get("label_basis") == HUMAN_LABEL_BASIS]
    if not human:
        return None
    return {"label_basis": HUMAN_LABEL_BASIS, **metrics(human)}


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _rejections(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """How many not_a_sign crops the classifier rejected (answered not_a_sign)."""
    rejected = sum(1 for sample in samples if sample["predicted"] == NOT_A_SIGN)
    return {"crops": len(samples), "rejected": rejected, "rejection_rate": _rate(rejected, len(samples))}


def classifier_metrics(samples: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    """What the v2 classifier adds to reader_metrics, from its samples (truth, predicted, correct, source,
    label_basis):

    - ``signs``: reader_metrics of the sign crops alone (truth not not_a_sign), the same on the crops a person
      confirmed, how many it rejected as not_a_sign (a sign lost), and ``family_accuracy``: right answers with
      the kind of an other sign dropped on both sides, comparable with the v1 reader;
    - ``not_a_sign``: the share of the not_a_sign crops it rejected, by where they come from (NEGATIVE_SOURCES),
      on frames a person reviewed, and what it called those it kept (``survivors``).
    """
    signs = [sample for sample in samples if sample["truth"] != NOT_A_SIGN]
    negatives = [sample for sample in samples if sample["truth"] == NOT_A_SIGN]
    rejected_signs = sum(1 for sample in signs if sample["predicted"] == NOT_A_SIGN)
    family_right = sum(1 for sample in signs if family_label(sample["predicted"]) == family_label(sample["truth"]))
    human_negatives = [sample for sample in negatives if sample.get("label_basis") == HUMAN_LABEL_BASIS]
    return {
        "signs": {
            **reader_metrics(signs, classes),
            "human_reviewed": human_reviewed_metrics(signs, lambda items: reader_metrics(items, classes)),
            "rejected": rejected_signs,
            "rejected_rate": _rate(rejected_signs, len(signs)),
            "family_accuracy": _rate(family_right, len(signs)),
        },
        "not_a_sign": {
            **_rejections(negatives),
            "by_source": {source: _rejections([sample for sample in negatives if sample.get("source") == source])
                          for source in NEGATIVE_SOURCES},
            "human_reviewed": _rejections(human_negatives) if human_negatives else None,
            "survivors": dict(sorted(Counter(sample["predicted"] for sample in negatives
                                             if sample["predicted"] != NOT_A_SIGN).items())),
        },
    }


def _load_checkpoint(torch, checkpoint_path: Path, expected_task: str) -> dict[str, Any]:
    payload = safe_torch_load(Path(checkpoint_path))
    if not isinstance(payload, dict) or payload.get("task") != expected_task:
        raise ValueError(f"checkpoint_task_mismatch: expected {expected_task}")
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("checkpoint_missing_state_dict")
    return payload


def evaluate_reader(
    plan: dict[str, Any],
    checkpoint_path: Path,
    data_root: Path,
    *,
    partition: str = "test",
    device_name: str = "cuda",
    batch_size: int = 64,
    num_workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    torch, Image, DataLoader, Dataset, transforms, _, mobilenet_v3_small = _require_training_stack()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(device_name)
    checkpoint = _load_checkpoint(torch, checkpoint_path, "reader")

    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("checkpoint_invalid_classes")
    if checkpoint.get("dataset_sha256") != plan.get("dataset_sha256"):
        raise ValueError("checkpoint_dataset_mismatch")
    if checkpoint.get("split_sha256") != plan.get("split_sha256"):
        raise ValueError("checkpoint_split_mismatch")
    # The labels of the plan and of the checkpoint must be the same scheme (v2: --classes v2 for the plan).
    scheme = checkpoint_class_scheme(checkpoint)
    if scheme != checkpoint_class_scheme(plan):
        raise ValueError(f"checkpoint_class_scheme_mismatch: the checkpoint is {scheme}, the plan "
                         f"{checkpoint_class_scheme(plan)} (--classes)")
    classifier = scheme == "v2"

    records = [r for r in plan["records"] if r["split"] == partition]
    if not records:
        raise ValueError(f"empty_evaluation_partition: {partition}")
    input_size = int(checkpoint.get("input_size", 160))
    root = Path(data_root).resolve()

    class ReaderDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            row = records[index]
            path = _safe_image_path(root, row["image_path"])
            image, factor = crop_source(path)
            if classifier:
                rectangle = crop_rectangle(row["bbox_xyxy"], image.width // factor, image.height // factor)
            else:
                rectangle = [int(v) for v in row["bbox_xyxy"]]
            crop = image.crop(tuple(value * factor for value in rectangle))
            transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
            ])
            return transform(crop), row["label"], index

    loader = DataLoader(
        ReaderDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    samples = []
    with torch.no_grad():
        for inputs, truth_labels, record_indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(1).cpu().tolist()
            confidences = probabilities.max(1).values.cpu().tolist()
            for predicted_index, confidence, truth_label, record_index in zip(
                predictions, confidences, truth_labels, record_indices.tolist()
            ):
                predicted_label = classes[predicted_index]
                row = records[record_index]
                sample = {
                    "annotation_id": row["annotation_id"],
                    "image_path": row["image_path"],
                    "truth": truth_label,
                    "predicted": predicted_label,
                    "confidence": float(confidence),
                    "correct": predicted_label == truth_label,
                    "label_basis": row.get("label_basis"),
                }
                if classifier:
                    # Where the crop comes from: a reviewed box, a removed proposal, a false box of the cutter.
                    sample.update({"source": row.get("source"), "bbox_xyxy": row.get("bbox_xyxy"),
                                   "negative_id": row.get("negative_id")})
                samples.append(sample)

    report = {
        "schema_version": 1,
        "task": "reader",
        "partition": partition,
        "classes": classes,
        **reader_metrics(samples, classes),
        # The same measure on the crops a person confirmed (None when the partition has none).
        "counts_by_label_basis": label_basis_counts(samples),
        "human_reviewed": human_reviewed_metrics(samples, lambda items: reader_metrics(items, classes)),
        "samples": samples,
        "checkpoint": str(checkpoint_path),
        "dataset_sha256": plan["dataset_sha256"],
        "split_sha256": plan["split_sha256"],
        "device": str(device),
    }
    if classifier:
        # accuracy, per_class and confusion above count every crop, the not_a_sign ones too.
        report.update({"class_scheme": scheme, **classifier_metrics(samples, classes),
                       "negatives_sha256": plan.get("negatives_sha256"),
                       "checkpoint_epoch": checkpoint.get("epoch")})
    return report


def evaluate_detector(
    plan: dict[str, Any],
    checkpoint_path: Path,
    data_root: Path,
    *,
    partition: str = "test",
    device_name: str = "cuda",
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    score_thresholds: tuple[float, ...] = REPORT_SCORE_THRESHOLDS,
    num_workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if not 0 <= score_threshold <= 1 or not all(0 <= value <= 1 for value in score_thresholds):
        raise ValueError("invalid_score_threshold")
    if not 0 < iou_threshold <= 1:
        raise ValueError("invalid_iou_threshold")

    torch, Image, _, _, TF, _, _ = _require_detector_stack()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(device_name)
    checkpoint = _load_checkpoint(torch, checkpoint_path, "detector")
    if checkpoint.get("dataset_sha256") != plan.get("dataset_sha256"):
        raise ValueError("checkpoint_dataset_mismatch")
    if checkpoint.get("split_sha256") != plan.get("split_sha256"):
        raise ValueError("checkpoint_split_mismatch")
    # The reviewed boxes it is measured on are the families it learnt to cut (--cutter-families of the plan).
    families = cutter_families(checkpoint)
    if families != cutter_families(plan):
        raise ValueError(f"checkpoint_families_mismatch: the cutter finds the families {families}, the plan has "
                         f"{cutter_families(plan)} (--cutter-families)")

    frames = [r for r in plan["frames"] if r["split"] == partition]
    if not frames:
        raise ValueError(f"empty_evaluation_partition: {partition}")
    root = Path(data_root).resolve()

    # Rebuilt as it was trained: the tail of its weights, its input size and default boxes.
    input_size = detector_input_size(checkpoint)
    model = load_detector(checkpoint)
    model.to(device)
    model.eval()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    thresholds = sorted({float(score_threshold), *(float(value) for value in score_thresholds)})
    scored: dict[float, list[dict[str, Any]]] = {value: [] for value in thresholds}
    seconds = []
    image_results = []
    from torch.utils.data import DataLoader, Dataset

    class Frames(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            with Image.open(_safe_image_path(root, frames[index]["image_path"])) as source:
                return TF.pil_to_tensor(source.convert("RGB"))

    # The frames are read ahead by num_workers processes, so the GPU does not wait for the PNG decoder; the time
    # measured below is still the model's alone.
    loader = DataLoader(Frames(), batch_size=None, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for position, (frame, pixels) in enumerate(zip(frames, loader)):
            tensor = pixels.float().div_(255.0).to(device)

            if position == 0:
                for _ in range(INFERENCE_WARMUP_RUNS):
                    model([tensor])
            # The time of the model alone, one frame at a time as in the car: from the frame on the device to
            # the boxes after non-maximum suppression (resize and normalization included).
            synchronize()
            start = time.perf_counter()
            prediction = model([tensor])[0]
            synchronize()
            seconds.append(time.perf_counter() - start)
            boxes = prediction["boxes"].detach().cpu().tolist()
            scores = prediction["scores"].detach().cpu().tolist()
            labels = prediction["labels"].detach().cpu().tolist()
            # Every speed-sign box the model keeps (its own floor is a score of 0.001), scored at each threshold.
            predictions = [{"box": box, "score": score} for box, score, label in zip(boxes, scores, labels) if label == 1]
            for value in thresholds:
                scored[value].append({**score_frame(frame["boxes"], predictions, score_threshold=value,
                                                    iou_threshold=iou_threshold),
                                      "label_basis": frame.get("label_basis"),
                                      "truth_families": frame.get("box_families")})
            matched = scored[float(score_threshold)][-1]
            selected = matched["selected"]
            # Every box from PER_FRAME_MIN_SCORE, matched there once, for the Watch panel of the Evaluate tab.
            low = [prediction for prediction in predictions if prediction["score"] >= PER_FRAME_MIN_SCORE]
            pairs = {position: index for position, index, _ in
                     greedy_matching(low, frame["boxes"], iou_threshold=iou_threshold) if index >= 0}
            image_results.append({
                "image_path": frame["image_path"],
                "image_sha256": frame["image_sha256"],
                "route_id": frame["route_id"],
                "segment_index": frame["segment_index"],
                "label_basis": frame.get("label_basis"),
                "truth_boxes": frame["boxes"],
                "truth_families": frame.get("box_families"),
                "predictions": selected,
                "truth_count": len(frame["boxes"]),
                "prediction_count": len(selected),
                "true_positives": matched["true_positives"],
                "false_positives": matched["false_positives"],
                "false_negatives": matched["false_negatives"],
                "matches": matched["matches"],
                "false_positive_details": matched["false_positive_details"],
                "size": [int(pixels.shape[2]), int(pixels.shape[1])],
                **per_frame_record(low, pairs),
            })

    main = scored[float(score_threshold)]
    milliseconds = [value * 1000.0 for value in seconds]
    return {
        "schema_version": 1,
        "task": "detector",
        "partition": partition,
        **detector_metrics(image_results),
        # Found and missed reviewed boxes by their width in the frame (WIDTH_BUCKETS), at score_threshold.
        "recall_by_width": recall_by_width(main),
        # The same by family group: "speed" compares with the cutters of the speed signs only.
        "recall_by_family": recall_by_family(main),
        "families": families,
        # The same measure on the frames a person confirmed (None when the partition has none).
        "counts_by_label_basis": label_basis_counts(image_results),
        "human_reviewed": human_reviewed_metrics(main, detector_size_metrics),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        # Every measure again at each score threshold: this report's and REPORT_SCORE_THRESHOLDS.
        "by_score_threshold": {f"{value:g}": threshold_metrics(scored[value]) for value in thresholds},
        "input_size": list(input_size),
        "input_size_label": format_input_size(input_size),
        "anchor_min_ratio": detector_anchor_min_ratio(checkpoint),
        # What the model itself keeps before any threshold here: the boxes per frame cannot exceed max_boxes.
        "model_postprocessing": {"score_floor": float(model.score_thresh), "nms_iou": float(model.nms_thresh),
                                 "max_boxes": int(model.detections_per_img)},
        "checkpoint_epoch": checkpoint.get("epoch"),
        "inference": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
            "frames": len(milliseconds),
            "batch_size": 1,
            "precision": "float32",
            "warmup_runs": INFERENCE_WARMUP_RUNS,
            "mean_ms_per_frame": statistics.fmean(milliseconds),
            "median_ms_per_frame": statistics.median(milliseconds),
            "max_ms_per_frame": max(milliseconds),
            "measures": "model forward of one frame already on the device: internal resize and normalization, "
                        "backbone, head, box decoding and NMS; not the image decoding nor the copy to the device",
        },
        "checkpoint": str(checkpoint_path),
        "dataset_sha256": plan["dataset_sha256"],
        "split_sha256": plan["split_sha256"],
        "device": str(device),
        "images": image_results,
        # How each image keeps the cutter's boxes from PER_FRAME_MIN_SCORE (cutter_boxes, cutter_box_counts).
        "per_frame_boxes": per_frame_descriptor(chain=False, min_score=PER_FRAME_MIN_SCORE),
        "limitations": [
            "single_iou_threshold_not_coco_map",
            "false_positives_per_hour_requires_timed_replay",
        ],
    }


def write_evaluation(report: dict[str, Any], output: Path | None) -> None:
    if output is None:
        return
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# -- the detectors of scripts/run_detector_sizes.sh side by side -----------------------------------------
DETECTOR_SIZES = ("320x320", "640x360", "960x544")
SIZE_PARTITIONS = ("test", "validation")
WIDTH_LABELS = {"lt12": "<12", "12-20": "12-20", "20-40": "20-40", "ge40": ">=40"}


def _read_report(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _fraction(value: Any, digits: int = 3) -> str:
    return "-" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{value:.{digits}f}"


def _found(metrics: dict[str, Any]) -> str:
    """'0.06 5/80': recall with the boxes found and reviewed; '-' without reviewed boxes."""
    truth = metrics.get("truth_box_count", metrics.get("truth"))
    found = metrics.get("true_positives", metrics.get("found"))
    recall = metrics.get("recall_iou", metrics.get("recall"))
    if not truth:
        return "-"
    return f"{_fraction(recall, 2)} {found}/{truth}"


def _table(lines: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip() for line in lines]


def _training_cells(result: dict[str, Any] | None) -> tuple[str, str, str]:
    """(best epoch of the epochs run, mean seconds per epoch, peak GPU memory) of a result.json."""
    if result is None:
        return "-", "-", "-"
    history = [item for item in result.get("history") or [] if isinstance(item, dict)]
    best = result.get("best_epoch")
    if best is None:
        best = next((item.get("epoch") for item in history
                     if item.get("validation_loss") == result.get("best_validation_loss")), None)
    epochs = "-" if best is None else f"{best}/{len(history)}"
    seconds = [item["seconds"] for item in history if isinstance(item.get("seconds"), (int, float))]
    per_epoch = f"{statistics.fmean(seconds):.0f}" if seconds else "-"
    peak = (result.get("gpu") or {}).get("peak_memory_allocated_bytes")
    memory = f"{peak / 2**30:.1f}" if isinstance(peak, (int, float)) else "-"
    return epochs, per_epoch, memory


def _per_frame(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if key == "predictions_per_image" and value is None and metrics.get("image_count"):
        # Reports written before predictions_per_image: found + false boxes per frame.
        value = ((metrics.get("true_positives") or 0) + (metrics.get("false_positives") or 0)) / metrics["image_count"]
    return _fraction(value, 2)


def detector_sizes_table(sizes_root: Path, sizes: list[str] | tuple[str, ...] = DETECTOR_SIZES) -> str:
    """The detectors trained at each input size side by side: <sizes_root>/<size>/result.json and
    <sizes_root>/<size>/evaluation/detector-<partition>.json (scripts/run_detector_sizes.sh).

    The detector is the cutter of the pipeline, judged first on recall: per partition, at every score threshold
    measured (lowest first) and for each size, recall overall and by the width of the reviewed box, the boxes
    per frame it passes on to the classifier and the false ones among them, then precision and F1; on all the
    frames and on the frames a person confirmed. Then what each size costs and how it was trained: milliseconds
    per frame, seconds per training epoch, GPU memory, the epoch the validation chose and by what, the share of
    frames without signs per epoch, augmentation and the smallest default box.
    """
    root = Path(sizes_root)
    out: list[str] = []
    milliseconds: dict[str, str] = {}
    devices: set[str] = set()
    for partition in SIZE_PARTITIONS:
        reports = {size: _read_report(root / size / "evaluation" / f"detector-{partition}.json") for size in sizes}
        first = next((report for report in reports.values() if report is not None), None)
        if out:
            out.append("")
        if first is None:
            out.append(f"== {partition}: not evaluated ==")
            continue
        frames = first.get("image_count")
        human_frames = (first.get("human_reviewed") or {}).get("image_count", 0)
        out.append(f"== {partition}: the cutter, recall first ({frames} frames, {human_frames} confirmed by a person; "
                   f"found = IoU >= {first.get('iou_threshold')} with a reviewed box; width of the reviewed box in "
                   f"frame pixels; speed = the speed signs alone; boxes/frame = boxes passed on to the classifier) ==")
        # "speed": the recall of the speed signs alone, for a cutter of the other signs too (families all); the
        # recall of a speed-only cutter is the same number.
        header = ("score", "rows", "size", "recall", "speed", *WIDTH_LABELS.values(), "boxes/frame", "false/frame",
                  "precision", "F1")
        lines: list[tuple[str, ...]] = [header]
        thresholds = sorted({key for report in reports.values() if report is not None
                             for key in (report.get("by_score_threshold") or {})}, key=float)
        missing = [size for size in sizes if reports[size] is None]
        for score in thresholds:
            if len(lines) > 1:
                lines.append(("",) * len(header))
            for rows in ("all", "human"):
                for size in sizes:
                    report = reports[size]
                    if report is None:
                        continue
                    metrics = (report.get("by_score_threshold") or {}).get(score)
                    if isinstance(metrics, dict) and rows == "human":
                        metrics = metrics.get("human_reviewed")
                    if not isinstance(metrics, dict):
                        lines.append((score, rows, size, *["-"] * (len(header) - 3)))
                        continue
                    widths = metrics.get("recall_by_width") or {}
                    speed = (metrics.get("recall_by_family") or {}).get("speed")
                    if speed is None and report.get("families", "speed") == "speed":
                        speed = metrics
                    lines.append((
                        score, rows, size, _found(metrics), _found(speed or {}),
                        *(_found(widths.get(key) or {}) for key in WIDTH_LABELS),
                        _per_frame(metrics, "predictions_per_image"), _per_frame(metrics, "false_positives_per_image"),
                        _fraction(metrics.get("precision_iou")), _fraction(metrics.get("f1_iou")),
                    ))
        out += _table(lines)
        if missing:
            out.append("not evaluated: " + ", ".join(missing))
        for size, report in reports.items():
            inference = (report or {}).get("inference") or {}
            if partition == "test" or size not in milliseconds:
                if isinstance(inference.get("mean_ms_per_frame"), (int, float)):
                    milliseconds[size] = _fraction(inference["mean_ms_per_frame"], 1)
                    devices.add(str(inference.get("device_name")))

    results = {size: _read_report(root / size / "result.json") for size in sizes}
    cost = [("size", "ms/frame", "s/epoch", "gpu_peak_GB", "best_epoch", "chosen_by", "val_recall",
             "sign_free/epoch", "augmentation", "default_box_min", "families")]
    for size in sizes:
        result = results[size]
        best, per_epoch, memory = _training_cells(result)
        if result is None:
            cost.append((size, milliseconds.get(size, "-"), "no result.json", *[""] * (len(cost[0]) - 3)))
            continue
        config = result.get("config") if isinstance(result.get("config"), dict) else {}
        # Results written before these settings: best epoch by validation loss, every frame without signs.
        chosen_by = (result.get("selection") or {}).get("by") or "loss"
        cost.append((size, milliseconds.get(size, "-"), per_epoch, memory, best, chosen_by,
                     _fraction(result.get("best_validation_recall")),
                     _fraction(config.get("negative_frame_fraction", 1.0), 2),
                     "on" if result.get("augmentation") else "off", _fraction(result.get("anchor_min_ratio"), 2),
                     str(result.get("families") or "speed")))
    out += ["", f"== cost and training: ms/frame of the model alone on {', '.join(sorted(devices)) or '-'} (batch 1, "
                f"float32, on test); s/epoch and GPU memory of the training; best epoch / epochs run, chosen by "
                f"validation recall at score 0.1 (val_recall) or loss; share of the frames without signs per epoch; "
                f"smallest default box as a fraction of the image side; families: what it cuts (speed signs, or all: "
                f"the other signs too) =="]
    out += _table(cost)
    return "\n".join(out) + "\n"
