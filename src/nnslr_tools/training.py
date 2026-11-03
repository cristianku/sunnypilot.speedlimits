"""Reader-baseline training for NNSLR.

This module is isolated from the core/data CLI import surface. PyTorch,
torchvision and Pillow are imported only after an explicit training request,
and so is the optional wandb tracking (nnslr_tools.tracking).
prepare_reader_training is stdlib-only and can validate the exact dataset/split
that would be consumed by a GPU run.

The first baseline trains only the sign reader on reviewed bounding-box crops.
It does not train a detector and it does not produce an on-vehicle bundle.

The reader has two class schemes (``class_scheme``, recorded in the plan, the training state and the
checkpoint; what has none is v1):

- v1, the reader of the first runs: the reviewed boxes of signs, labelled ``family:value``
  (``maximum_speed:60``, ``maximum_speed:unknown``, ``cancellation``, ``other_sign``, ...);
- v2, the classifier of the pipeline (cutter -> classifier -> decider, docs/NEURAL_NETWORK.md): the same
  labels, the kind of an other sign (``other_sign:give_way`` when the review says it, else ``other_sign``),
  and ``not_a_sign``, learnt from boxes that are not signs: the proposals a reviewer removed, the boxes a
  reviewer marked "not a sign" and the false boxes of the cutter mined by nnslr_tools.chain.mine_negatives.
  Its training crops are photometrically augmented (darker, lower contrast) and their boxes jittered, as the
  cutter's boxes are never exact.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any

from nnslr_tools import synthetic_signs, tracking
from nnslr_tools.checkpoints import (DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, DEFAULT_DETECTOR_INPUT_SIZE,
                                     DETECTOR_ANCHOR_MAX_RATIO, MAX_DETECTOR_INPUT_SIDE, atomic_torch_save,
                                     build_detector, detector_is_wide, model_weights, stable_weights, torch_load)
from nnslr_tools.progress import report as report_progress
from nnslr_tools.annotations import load_jsonl, validate_dataset, written_sign_value
from nnslr_tools.sign_types import NOT_A_SIGN_TYPES, SIGN_TYPES

# The kind of a "not a sign" box on a vehicle.
ON_VEHICLE = NOT_A_SIGN_TYPES[0]
from nnslr_tools.splits import validate_splits

TRAINING_SCHEMA_VERSION = 1
ELIGIBLE_REVIEW_STATES = frozenset({"accepted", "corrected"})
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
# The partitions the classifier's negatives come from, each from its own frames only (never the buffer).
NEGATIVE_PARTITIONS = ("train", "validation", "test")
# ImageNet normalization of the reader crops (also undone for the wandb crops).
READER_NORMALIZE_MEAN = (.485, .456, .406)
READER_NORMALIZE_STD = (.229, .224, .225)

# -- the class schemes of the reader ---------------------------------------------------------------------
CLASS_SCHEMES = ("v1", "v2")
NOT_A_SIGN = "not_a_sign"
OTHER_SIGN = "other_sign"
# Where a not_a_sign crop of the v2 scheme comes from: a proposal a reviewer removed (review_state rejected), a
# false box of the cutter (nnslr_tools.chain.mine_negatives), a box a reviewer kept and marked "not a sign".
NEGATIVE_SOURCES = ("rejected", "mined", "reviewed")
# A removed proposal or a mined box is a negative only when it overlaps no box a reviewer kept by this IoU or
# more: a proposal removed as a duplicate of a kept one, or drawn again, still frames a sign.
NEGATIVE_MAX_IOU = 0.1
# The kind of an other sign is a class of its own from this many training crops; rarer kinds are folded into
# "other_sign" in every partition (a class seen once or twice is learnt from nothing and measured on noise).
MIN_KIND_TRAIN_CROPS = 3
# The review rule of 24 October 2026 (commit 6344a8b): from then on the AI reviewer boxes the other traffic signs
# too, with their kind. Before it, a frame an AI reviewed may hold an other sign without a box, so a box there
# that overlaps nothing reviewed may still be a sign: the negatives of such frames are the noisiest.
OTHER_SIGN_RULE_AT = "2026-10-24T15:33:17Z"
# Who reviewed a frame and when, for the negatives: a person; an AI after the rule above; an AI before it.
REVIEW_TIERS = ("human", "ai_after_rule", "ai_before_rule", "other")
# The frames whose reviewer boxed the other traffic signs too: a box left out there is not a sign. In the other
# tiers an AI removed the other-sign proposals and left those signs without a box (1,339 of the 1,434 removed
# proposals of train on 24 October 2026, mostly pedestrian crossings, give way, no parking and direction signs).
BOXES_OTHER_SIGNS = ("human", "ai_after_rule")


# The processes that read the frames for the GPU (DataLoader num_workers): half the cores, 4 to 16. A 1344 × 760 PNG
# takes about 22 ms to decode on one core: 4 workers fed a cutter at most ~180 frames a second, with 28 cores idle.
# How many read the frames changes nothing in what is learnt (a resumed run may use another number).
DEFAULT_WORKERS = min(16, max(4, (os.cpu_count() or 8) // 2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reader_label(row: dict[str, Any]) -> str | None:
    """Return the classification label for a reviewed sign crop."""
    if row.get("review_state") not in ELIGIBLE_REVIEW_STATES:
        return None
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    family = row.get("sign_family")
    if family == "not_a_sign":
        return None

    value_state = row.get("value_state")
    value = written_sign_value(row)
    if value_state == "read" and isinstance(value, int) and not isinstance(value, bool):
        return f"{family}:{value}"
    if family in {"cancellation", "unreadable", "other_sign"}:
        return str(family)
    if value_state in {"unreadable", "unknown", "not_applicable"}:
        return f"{family}:{value_state}"
    return None


def _without_unseen_validation(records: list[dict[str, Any]], unseen: list[str],
                               skipped: Counter) -> tuple[list[dict[str, Any]], Counter]:
    """The records without the validation crops of a label no training crop has: the network cannot learn it, so they
    cannot measure it. Left out (counted in skipped, named in unseen_validation_labels) instead of stopping the
    training: on 26 October 2026 one AI review of a tram's 45 plate, the only 45 of the dataset, stopped a pipeline."""
    if not unseen:
        return records, skipped
    left = set(unseen)
    kept = [r for r in records if not (r["split"] == VALIDATION_SPLIT and r["label"] in left)]
    skipped["validation_label_not_in_train"] += len(records) - len(kept)
    return kept, skipped


# Reviewers whose frames a training leaves out of its train partition (comma-separated starts of their names, e.g.
# "ai/llama,ai/strata"): to measure what their reviews do to a network. Validation and test stay whole.
TRAIN_WITHOUT_ENV = "NNSLR_TRAIN_WITHOUT_REVIEWERS"


def _without_reviewers(records: list[dict[str, Any]], rows: list[dict[str, Any]],
                       skipped: Counter) -> tuple[list[dict[str, Any]], list[str]]:
    """The records without the train crops (signs and not-a-sign) of the frames a reviewer of TRAIN_WITHOUT_ENV
    reviewed, and those reviewers; the records as they are when it is not set."""
    without = [item.strip() for item in os.environ.get(TRAIN_WITHOUT_ENV, "").split(",") if item.strip()]
    if not without:
        return records, []
    frames = {row.get("image_path") for row in rows
              if isinstance(row.get("reviewer"), str) and row["reviewer"].startswith(tuple(without))}
    kept = [r for r in records if not (r["split"] == TRAIN_SPLIT and r["image_path"] in frames)]
    skipped["train_frames_of_reviewers_left_out"] += len(records) - len(kept)
    return kept, without


def _classifier_label(row: dict[str, Any]) -> str | None:
    """The v2 label of a reviewed box: the v1 label, the kind of an other sign (``other_sign:<kind>`` when the
    review gives one, else ``other_sign``), and ``not_a_sign`` for a kept box a reviewer marked so (none for one on
    a vehicle)."""
    if row.get("review_state") not in ELIGIBLE_REVIEW_STATES:
        return None
    bbox = row.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    family = row.get("sign_family")
    if family == NOT_A_SIGN:
        # A sign-like disc on a vehicle (a truck's top-speed sticker) is, cut to its box, the same pixels as a
        # sign: taught as not_a_sign it would teach the classifier to reject signs. The decider learns it, from
        # the scene (nnslr_tools.context_experiment).
        return None if row.get("sign_type") == ON_VEHICLE else NOT_A_SIGN
    if family == OTHER_SIGN:
        kind = row.get("sign_type")
        return f"{OTHER_SIGN}:{kind}" if isinstance(kind, str) and kind in SIGN_TYPES else OTHER_SIGN
    return _reader_label(row)


def scheme_label(row: dict[str, Any], class_scheme: str) -> str | None:
    """The label of a reviewed box in a class scheme (v1: _reader_label, v2: _classifier_label)."""
    if class_scheme not in CLASS_SCHEMES:
        raise ValueError(f"unknown class scheme {class_scheme!r}: one of {', '.join(CLASS_SCHEMES)}")
    return _classifier_label(row) if class_scheme == "v2" else _reader_label(row)


def family_label(label: str) -> str:
    """A label as the v1 reader names it: the kind of an other sign dropped (``other_sign:give_way`` ->
    ``other_sign``), every other label unchanged. Compares the classifier with the old reader."""
    return OTHER_SIGN if str(label).startswith(OTHER_SIGN + ":") else str(label)


def checkpoint_class_scheme(checkpoint: dict[str, Any]) -> str:
    """The class scheme of a reader checkpoint, plan or result; v1 when it does not say (written before v2)."""
    scheme = checkpoint.get("class_scheme", "v1") if isinstance(checkpoint, dict) else "v1"
    if scheme not in CLASS_SCHEMES:
        raise ValueError(f"unknown class scheme {scheme!r}")
    return scheme


def _label_basis(row: dict[str, Any]) -> str | None:
    """Who confirmed a row: human_review, ai_review or model_review (provenance.label_basis)."""
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    return provenance.get("label_basis")


def _timestamp(value: Any) -> datetime | None:
    """An ISO review timestamp ("2026-10-24T02:58:45.917Z") as an aware datetime; None when it is not one."""
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def review_tier(label_basis: str | None, reviewed_at: str | None) -> str:
    """REVIEW_TIERS of a frame: reviewed by a person, by an AI after OTHER_SIGN_RULE_AT, by an AI before it."""
    if label_basis == "human_review":
        return "human"
    if label_basis == "ai_review":
        moment = _timestamp(reviewed_at)
        rule = _timestamp(OTHER_SIGN_RULE_AT)
        return "ai_after_rule" if moment is not None and moment >= rule else "ai_before_rule"
    return "other"


def box_overlap(a: list[float], b: list[float]) -> float:
    """IoU of two boxes x1, y1, x2, y2 (the same measure as nnslr_tools.evaluation.box_iou)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = (max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
             - intersection)
    return 0.0 if union <= 0 else intersection / union


def reviewed_frames(rows: list[dict[str, Any]], assignments: dict[str, Any]) -> list[dict[str, Any]]:
    """The reviewed frames of a dataset with their partition, in a stable order.

    A frame (one image, by its sha256) is in the partition of the rows a reviewer kept (accepted or corrected;
    the split groups identical images, so they all share one). The removed proposals (review_state rejected)
    have no assignment of their own: they follow their frame. Each frame keeps its kept rows (``kept``, a
    whole-frame "no sign" row included), the boxes of the removed proposals (``rejected``), who reviewed it
    (``label_basis``: one basis, or "mixed"), when (``reviewed_at``, the latest review time) and its
    REVIEW_TIERS ``tier``. A frame without any kept row has no partition and is left out.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("image_sha256")
        if not isinstance(key, str):
            continue
        frame = grouped.setdefault(key, {
            "image_path": row.get("image_path"), "image_sha256": key, "width": row.get("width"),
            "height": row.get("height"), "route_id": row.get("route_id"), "segment_index": row.get("segment_index"),
            "output_index": row.get("output_index"), "split": None, "kept": [], "rejected": [],
            "label_bases": set(), "reviewed_at": None})
        if row.get("review_state") in ELIGIBLE_REVIEW_STATES:
            split = assignments.get(row.get("annotation_id"))
            if isinstance(split, str):
                if frame["split"] not in (None, split):
                    raise ValueError("inconsistent_frame_group: one image in two partitions")
                frame["split"] = split
            frame["kept"].append(row)
            frame["label_bases"].add(_label_basis(row))
            stamp = row.get("review_timestamp")
            if _timestamp(stamp) is not None and (frame["reviewed_at"] is None
                                                  or _timestamp(stamp) > _timestamp(frame["reviewed_at"])):
                frame["reviewed_at"] = stamp
        elif row.get("review_state") == "rejected" and isinstance(row.get("bbox_xyxy"), list):
            frame["rejected"].append(row)
    frames = []
    for frame in grouped.values():
        if frame["split"] is None:
            continue
        bases = frame.pop("label_bases")
        frame["label_basis"] = next(iter(bases)) if len(bases) == 1 else "mixed"
        frame["tier"] = review_tier(frame["label_basis"], frame["reviewed_at"])
        frames.append(frame)
    frames.sort(key=lambda f: (str(f["route_id"]), f["segment_index"] if isinstance(f["segment_index"], int) else -1,
                               str(f["image_path"])))
    return frames


def frame_boxes(frame: dict[str, Any]) -> list[list[float]]:
    """Every box a reviewer saw in a frame: the kept ones (any family) and the removed proposals."""
    return ([list(row["bbox_xyxy"]) for row in frame["kept"] if isinstance(row.get("bbox_xyxy"), list)]
            + [list(row["bbox_xyxy"]) for row in frame["rejected"]])


def negative_id(image_sha256: str, box: list[float]) -> str:
    """A stable id of a negative box (mined boxes have no annotation id): its image and its box to 0.1 px."""
    text = json.dumps([image_sha256, [round(float(value), 1) for value in box]])
    return hashlib.sha256(text.encode()).hexdigest()


NEGATIVES_KIND = "classifier_negatives"


def read_negatives(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the rows of a negatives file of nnslr mine-negatives (JSONL: a header line first)."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"negatives_invalid: empty file {path}")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or header.get("kind") != "header" or header.get("task") != NEGATIVES_KIND:
        raise ValueError("negatives_invalid: the first line is not the header of nnslr mine-negatives")
    rows = []
    for number, line in enumerate(lines[1:], 2):
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get("kind") != "negative" or not isinstance(row.get("bbox_xyxy"), list)
                or len(row["bbox_xyxy"]) != 4 or not isinstance(row.get("image_sha256"), str)):
            raise ValueError(f"negatives_invalid: line {number}")
        rows.append(row)
    return header, rows


def _safe_image_path(data_root: Path, image_path: str) -> Path:
    root = data_root.resolve()
    candidate = (root / image_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"unsafe_image_path: {image_path}")
    return candidate


def _load_split_document(split_path: Path, data_root: Path) -> tuple[Path, dict[str, Any]]:
    """Load a frozen split, following the build-splits latest.json pointer.

    The pointer itself is hash-bound to the immutable split file. A changed
    or escaped target is rejected before any training plan is produced.
    """
    requested = Path(split_path)
    document = json.loads(requested.read_text(encoding="utf-8"))
    if isinstance(document, dict) and isinstance(document.get("assignments"), dict):
        return requested, document

    if not isinstance(document, dict) or not isinstance(document.get("path"), str):
        raise ValueError("split_invalid: expected assignments or latest pointer")

    root = Path(data_root).resolve()
    target = Path(document["path"])
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target != root and root not in target.parents:
        raise ValueError("unsafe_split_path")
    if not target.is_file():
        raise ValueError(f"split_target_missing: {target}")

    expected_sha = document.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("split_pointer_invalid_hash")
    actual_sha = _sha256(target)
    if actual_sha != expected_sha:
        raise ValueError("split_pointer_hash_mismatch")

    resolved = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(resolved, dict) or not isinstance(resolved.get("assignments"), dict):
        raise ValueError("split_invalid: target has no assignments")
    return target, resolved


def load_reviewed_dataset(dataset_path: Path, split_path: Path, data_root: Path, *,
                          dataset_kind: str = "gold") -> dict[str, Any]:
    """The dataset rows and the frozen split, both checked (no torch): {rows, dataset_sha256, split_path,
    split_sha256, assignments}. The common start of every plan."""
    dataset_path = Path(dataset_path)
    data_root = Path(data_root)
    rows = load_jsonl(dataset_path)
    validation = validate_dataset(rows, data_root, dataset_kind=dataset_kind)
    if not validation.get("valid"):
        reasons = sorted({str(e.get("reason", "dataset_invalid")) for e in validation.get("errors", [])})
        raise ValueError("dataset_invalid: " + ",".join(reasons))

    resolved_split_path, split_doc = _load_split_document(Path(split_path), data_root)
    split_errors = validate_splits(
        rows,
        split_doc,
        validation["dataset_sha256"],
        dataset_kind=dataset_kind,
    )
    if split_errors:
        reasons = sorted({str(e.get("reason", "split_invalid")) for e in split_errors})
        raise ValueError("split_invalid: " + ",".join(reasons))

    assignments = split_doc.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("split_invalid: assignments")
    return {"rows": rows, "dataset_sha256": validation["dataset_sha256"], "split_path": resolved_split_path,
            "split_sha256": _sha256(resolved_split_path), "assignments": assignments}


def prepare_reader_training(
    dataset_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    dataset_kind: str = "gold",
    class_scheme: str = "v1",
    negatives_path: Path | None = None,
    mapillary_path: Path | None = None,
    mapillary_region: str = "all",
    mapillary_share: float = 100.0,
    mapillary_only: bool = False,
    start: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate inputs and build the immutable reader-training plan.

    No torch/Pillow import and no GPU access happens here.

    class_scheme v1 is the reader as it always was. v2 (the classifier) adds the kinds of the other signs and
    the not_a_sign crops (see _classifier_negatives); negatives_path is the file of nnslr mine-negatives for
    the cutter's false boxes (without it, only the removed proposals and the boxes marked "not a sign");
    mapillary_path is the crops file of nnslr mapillary-crops, extra train crops of v2 (nnslr_tools.mapillary);
    mapillary_region and mapillary_share choose which of them (the photos of some continents or countries, a share).

    The two stages of a classifier pretrained on Mapillary (v2): mapillary_only is the first, every Mapillary crop of
    a v2 class in train and none of ours; start (reader_start of a reader-best.pt, nnslr train --init-from) is the
    second, whose classes are its own and those of the start, so both stages have the same list (_prepare_classifier).
    """
    if class_scheme not in CLASS_SCHEMES:
        raise ValueError(f"unknown class scheme {class_scheme!r}: one of {', '.join(CLASS_SCHEMES)}")
    if negatives_path is not None and class_scheme != "v2":
        raise ValueError("negatives are crops of the v2 classifier: use --classes v2")
    if mapillary_path is not None and class_scheme != "v2":
        raise ValueError("the Mapillary crops are for the v2 classifier: use --classes v2")
    from nnslr_tools.mapillary import FULL_SHARE, check_share, region_text
    if mapillary_path is None and (region_text(mapillary_region) != "all" or check_share(mapillary_share) != FULL_SHARE):
        raise ValueError("--mapillary-region and --mapillary-share choose Mapillary crops: add --mapillary")
    if mapillary_only and mapillary_path is None:
        raise ValueError("--mapillary-only trains on the Mapillary crops alone: add --mapillary")
    if start is not None and class_scheme != "v2":
        raise ValueError("a start of a fresh run (--init-from) is for the v2 classifier: use --classes v2")
    if start is not None and checkpoint_class_scheme(start) != "v2":
        raise ValueError("the start (--init-from) is not a v2 classifier")
    dataset_path = Path(dataset_path)
    split_path = Path(split_path)
    data_root = Path(data_root)

    loaded = load_reviewed_dataset(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
    rows = loaded["rows"]
    validation = {"dataset_sha256": loaded["dataset_sha256"]}
    resolved_split_path = loaded["split_path"]
    assignments = loaded["assignments"]
    if class_scheme == "v2":
        return _prepare_classifier(loaded, dataset_path, split_path, data_root, dataset_kind=dataset_kind,
                                   negatives_path=negatives_path, mapillary_path=mapillary_path,
                                   mapillary_region=mapillary_region, mapillary_share=mapillary_share,
                                   mapillary_only=mapillary_only, start=start)

    records: list[dict[str, Any]] = []
    skipped = Counter()
    for row in rows:
        label = _reader_label(row)
        if label is None:
            skipped["not_reader_eligible"] += 1
            continue
        annotation_id = row.get("annotation_id")
        split_name = assignments.get(annotation_id)
        if not isinstance(split_name, str):
            skipped["missing_split"] += 1
            continue
        image_path = row.get("image_path")
        if not isinstance(image_path, str):
            skipped["missing_image_path"] += 1
            continue
        image = _safe_image_path(data_root, image_path)
        if not image.is_file():
            raise ValueError(f"missing_image: {image_path}")
        records.append({
            "annotation_id": annotation_id,
            "split": split_name,
            "image_path": image_path,
            "image_sha256": row.get("image_sha256"),
            "bbox_xyxy": row["bbox_xyxy"],
            "label": label,
            "sign_family": row.get("sign_family"),
            "value_state": row.get("value_state"),
            "value_kph": row.get("value_kph"),
            **{k: row[k] for k in ('sign_value', 'sign_unit') if k in row},
            "route_id": row.get("route_id"),
            "segment_index": row.get("segment_index"),
            "decoded_frame_index": row.get("decoded_frame_index"),
            "label_basis": _label_basis(row),
        })

    train_labels = sorted({r["label"] for r in records if r["split"] == TRAIN_SPLIT})
    val_labels = sorted({r["label"] for r in records if r["split"] == VALIDATION_SPLIT})
    unseen_validation = sorted(set(val_labels) - set(train_labels))
    records, skipped = _without_unseen_validation(records, unseen_validation, skipped)
    counts_by_split = Counter(r["split"] for r in records)
    counts_by_class = Counter(r["label"] for r in records if r["split"] == TRAIN_SPLIT)

    trainable = (
        counts_by_split[TRAIN_SPLIT] > 0
        and counts_by_split[VALIDATION_SPLIT] > 0
        and len(train_labels) >= 2
    )

    limitations = [
        "reader_only_no_detector",
        "bbox_crops_require_reviewed_localization",
        "no_runtime_bundle",
    ]
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": "reader",
        "class_scheme": "v1",
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": validation["dataset_sha256"],
        "split_path": str(resolved_split_path),
        "requested_split_path": str(split_path),
        "split_sha256": _sha256(resolved_split_path),
        "classes": train_labels,
        "records": records,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "train_counts_by_class": dict(sorted(counts_by_class.items())),
        "skipped": dict(sorted(skipped.items())),
        "unseen_validation_labels": unseen_validation,
        "trainable": trainable,
        "limitations": limitations,
    }


def _record(row: dict[str, Any], split: str, label: str, source: str, **extra: Any) -> dict[str, Any]:
    """A crop of the classifier plan: the fields of a v1 record, where it comes from and who reviewed it."""
    return {
        "annotation_id": row.get("annotation_id"),
        "split": split,
        "image_path": row.get("image_path"),
        "image_sha256": row.get("image_sha256"),
        "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
        "label": label,
        "sign_family": row.get("sign_family"),
        "sign_type": row.get("sign_type"),
        "value_state": row.get("value_state"),
        "value_kph": row.get("value_kph"),
        **{k: row[k] for k in ('sign_value', 'sign_unit') if k in row},
        "route_id": row.get("route_id"),
        "segment_index": row.get("segment_index"),
        "decoded_frame_index": row.get("decoded_frame_index"),
        "label_basis": _label_basis(row),
        "source": source,
        **extra,
    }


def classifier_positives(rows: list[dict[str, Any]], assignments: dict[str, Any]) -> Counter:
    """The sign crops of the v2 scheme (every label but not_a_sign) per partition: what the negatives are
    counted against."""
    counts: Counter = Counter()
    for row in rows:
        label = _classifier_label(row)
        split = assignments.get(row.get("annotation_id"))
        if label is not None and label != NOT_A_SIGN and isinstance(split, str):
            counts[split] += 1
    return counts


def _classifier_negatives(frames: list[dict[str, Any]], skipped: Counter) -> list[dict[str, Any]]:
    """The removed proposals of the frames of train, validation and test that are crops of not_a_sign: a box a
    reviewer took away and that overlaps no kept box (IoU < NEGATIVE_MAX_IOU; a duplicate on a sign, or a box
    drawn again, still frames a sign). Noted with who reviewed the frame and when (tier). An other-sign proposal
    removed in a frame whose reviewer did not box the other signs (not BOXES_OTHER_SIGNS) is left out: it was
    removed by the rule of the time, and is mostly a real sign."""
    found = []
    for frame in frames:
        if frame["split"] not in NEGATIVE_PARTITIONS:
            skipped["rejected_outside_train_validation_test"] += len(frame["rejected"])
            continue
        kept = [row["bbox_xyxy"] for row in frame["kept"] if isinstance(row.get("bbox_xyxy"), list)]
        for row in frame["rejected"]:
            if any(box_overlap(row["bbox_xyxy"], box) >= NEGATIVE_MAX_IOU for box in kept):
                skipped["rejected_overlaps_a_kept_box"] += 1
                continue
            if row.get("sign_family") == "other_sign" and frame["tier"] not in BOXES_OTHER_SIGNS:
                skipped["rejected_other_sign_before_rule"] += 1
                continue
            found.append(_record(row, frame["split"], NOT_A_SIGN, "rejected", tier=frame["tier"],
                                 frame_label_basis=frame["label_basis"], proposal_family=row.get("sign_family"),
                                 negative_id=negative_id(row["image_sha256"], row["bbox_xyxy"])))
    return found


def _mined_negatives(negatives_path: Path, loaded: dict[str, Any], frames: list[dict[str, Any]],
                     data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The not_a_sign crops of a negatives file, checked against this plan: the same dataset and split, every
    box in the partition its frame has here, every image inside the data root."""
    header, rows = read_negatives(negatives_path)
    if header.get("dataset_sha256") != loaded["dataset_sha256"] or header.get("split_sha256") != loaded["split_sha256"]:
        raise ValueError("negatives_mismatch: the negatives were mined on another dataset or split; mine them again")
    partition = {frame["image_sha256"]: frame for frame in frames}
    found = []
    for row in rows:
        frame = partition.get(row["image_sha256"])
        if frame is None or frame["split"] != row.get("split") or row.get("split") not in NEGATIVE_PARTITIONS:
            raise ValueError(f"negatives_mismatch: {row.get('image_path')} is not a frame of {row.get('split')}")
        _safe_image_path(data_root, frame["image_path"])
        found.append({
            "annotation_id": None, "split": frame["split"], "image_path": frame["image_path"],
            "image_sha256": frame["image_sha256"], "bbox_xyxy": [float(value) for value in row["bbox_xyxy"]],
            "label": NOT_A_SIGN, "sign_family": None, "sign_type": None, "value_state": None, "value_kph": None,
            "route_id": frame["route_id"], "segment_index": frame["segment_index"], "decoded_frame_index": None,
            "label_basis": frame["label_basis"], "source": "mined", "tier": frame["tier"],
            "frame_label_basis": frame["label_basis"], "score": row.get("score"),
            "negative_id": row.get("negative_id") or negative_id(frame["image_sha256"], row["bbox_xyxy"]),
        })
    summary = {"file": str(negatives_path), "sha256": _sha256(Path(negatives_path)), "cutter": header.get("cutter"),
               "settings": header.get("settings"), "counts": header.get("counts")}
    return found, summary


def _prepare_classifier(loaded: dict[str, Any], dataset_path: Path, split_path: Path, data_root: Path, *,
                        dataset_kind: str, negatives_path: Path | None,
                        mapillary_path: Path | None = None, mapillary_region: str = "all",
                        mapillary_share: float = 100.0, mapillary_only: bool = False,
                        start: dict[str, Any] | None = None) -> dict[str, Any]:
    """The v2 plan (see prepare_reader_training): the sign crops with their v2 labels, the kinds too rare to
    learn folded into other_sign, the not_a_sign crops of each partition from its own frames only, and the
    Mapillary crops in train when asked (after the folding, which counts our crops only), of the region and share
    asked (nnslr_tools.mapillary.training_records).

    mapillary_only, the first stage of a two-stage run: every Mapillary crop of a v2 class (no cap per label; a kind
    that is no class of ours comes in as other_sign) and none of ours in train. Its classes are those our train
    would have, and Mapillary's: the second stage, trained on ours from it, has the same list. Validation keeps our
    sign crops, to choose the epoch that reads our camera best, without the not_a_sign crops: Mapillary has none, the
    second stage learns them. start, the second stage: the classes are ours and the start's (a class of the start
    that our train lacks keeps what the start learnt, and validation keeps it)."""
    rows = loaded["rows"]
    assignments = loaded["assignments"]
    frames = reviewed_frames(rows, assignments)
    records: list[dict[str, Any]] = []
    skipped = Counter()
    for row in rows:
        label = _classifier_label(row)
        if label is None:
            skipped["not_classifier_eligible"] += 1
            continue
        split_name = assignments.get(row.get("annotation_id"))
        if not isinstance(split_name, str):
            skipped["missing_split"] += 1
            continue
        if not isinstance(row.get("image_path"), str):
            skipped["missing_image_path"] += 1
            continue
        if label == NOT_A_SIGN and split_name not in NEGATIVE_PARTITIONS:
            skipped["reviewed_not_a_sign_outside_train_validation_test"] += 1
            continue
        records.append(_record(row, split_name, label, "reviewed"))
    records += _classifier_negatives(frames, skipped)
    negatives_file = None
    if negatives_path is not None:
        mined, negatives_file = _mined_negatives(Path(negatives_path), loaded, frames, data_root)
        records += mined
    records, without = _without_reviewers(records, rows, skipped)
    for path in sorted({record["image_path"] for record in records}):
        if not _safe_image_path(data_root, path).is_file():
            raise ValueError(f"missing_image: {path}")

    # A kind of other sign with too few training crops is not a class: it is folded into other_sign everywhere.
    kinds = Counter(record["label"] for record in records
                    if record["split"] == TRAIN_SPLIT and record["label"].startswith(OTHER_SIGN + ":"))
    every_kind = {record["label"] for record in records if record["label"].startswith(OTHER_SIGN + ":")}
    folded = {label: OTHER_SIGN for label in sorted(every_kind) if kinds[label] < MIN_KIND_TRAIN_CROPS}
    for record in records:
        if record["label"] in folded:
            record["label"] = OTHER_SIGN
    mapillary = None
    ours = {r["label"] for r in records if r["split"] == TRAIN_SPLIT}
    extra: list[dict[str, Any]] = []
    if mapillary_path is not None:
        from nnslr_tools.mapillary import training_records
        extra, mapillary = training_records(Path(mapillary_path), ours,
                                            {label for label in ours if label.startswith(OTHER_SIGN + ":")},
                                            region=mapillary_region, share=mapillary_share,
                                            capped=not mapillary_only, fold_other_kinds=mapillary_only)
        for path in sorted({record["image_path"] for record in extra}):
            if not _safe_image_path(data_root, path).is_file():
                raise ValueError(f"missing_image: {path} (prepare Mapillary again: helpers/mapillary.sh)")
    if mapillary_only:
        skipped.update("our_train_crops_left_out_of_the_pretraining" if r["split"] == TRAIN_SPLIT else
                       "validation_not_a_sign_left_out_of_the_pretraining" for r in records
                       if r["split"] == TRAIN_SPLIT or (r["split"] == VALIDATION_SPLIT and r["label"] == NOT_A_SIGN))
        records = [r for r in records
                   if r["split"] != TRAIN_SPLIT and not (r["split"] == VALIDATION_SPLIT and r["label"] == NOT_A_SIGN)]
    records += extra

    train_labels = sorted({r["label"] for r in records if r["split"] == TRAIN_SPLIT})
    classes = sorted(set(train_labels) | ours) if mapillary_only else list(train_labels)
    classes_from_start: list[str] = []
    if start is not None:
        classes_from_start = sorted(set(start.get("classes") or []) - set(classes))
        classes = sorted(set(classes) | set(classes_from_start))
    val_labels = sorted({r["label"] for r in records if r["split"] == VALIDATION_SPLIT})
    # A validation label that nothing in this run or its start learnt cannot be read: it leaves validation.
    unseen_validation = sorted(set(val_labels) - set(train_labels) - set(classes_from_start))
    records, skipped = _without_unseen_validation(records, unseen_validation, skipped)
    counts_by_split = Counter(r["split"] for r in records)
    counts_by_class = Counter(r["label"] for r in records if r["split"] == TRAIN_SPLIT)
    positives = Counter(r["split"] for r in records if r["label"] != NOT_A_SIGN)
    negatives: dict[str, dict[str, int]] = {}
    by_basis: dict[str, dict[str, dict[str, int]]] = {}
    by_tier: dict[str, dict[str, dict[str, int]]] = {}
    for record in records:
        if record["label"] != NOT_A_SIGN:
            continue
        split, source = record["split"], record["source"]
        cell = negatives.setdefault(split, {name: 0 for name in NEGATIVE_SOURCES})
        cell[source] += 1
        basis = str(record.get("frame_label_basis") or record.get("label_basis") or "unknown")
        by_basis.setdefault(split, {}).setdefault(source, {}).setdefault(basis, 0)
        by_basis[split][source][basis] += 1
        tier = str(record.get("tier") or review_tier(record.get("label_basis"), None))
        by_tier.setdefault(split, {}).setdefault(source, {}).setdefault(tier, 0)
        by_tier[split][source][tier] += 1

    trainable = (
        counts_by_split[TRAIN_SPLIT] > 0
        and counts_by_split[VALIDATION_SPLIT] > 0
        and len(train_labels) >= 2
        # The pretraining has no not_a_sign crop: the class is there for the second stage.
        and NOT_A_SIGN in (classes if mapillary_only else train_labels)
    )
    limitations = [
        "classifier_on_reviewed_boxes_and_negatives",
        "bbox_crops_require_reviewed_localization",
        "no_runtime_bundle",
    ]
    if negatives_path is None:
        limitations.append("no_mined_negatives: only removed proposals and boxes marked not a sign")
    if mapillary is not None:
        limitations.append("mapillary_crops_in_train: street photos of other cameras, measured on our drives only")
    if mapillary_only:
        limitations.append("mapillary_only_pretraining: no crop of ours and no not_a_sign in train; "
                           "a start for nnslr train --init-from, not a classifier to use")
    if start is not None and start.get("split_sha256") != loaded["split_sha256"]:
        limitations.append("start_of_another_split: the start chose or learnt its weights on another split, "
                           "whose crops may be in this run's validation or test")
    if any(record.get("tier") == "ai_before_rule" for record in records if record["label"] == NOT_A_SIGN):
        limitations.append("negatives_of_ai_reviews_before_the_other_sign_rule_may_be_unboxed_signs")
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": "reader",
        "class_scheme": "v2",
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": loaded["dataset_sha256"],
        "split_path": str(loaded["split_path"]),
        "requested_split_path": str(split_path),
        "split_sha256": loaded["split_sha256"],
        "classes": classes,
        "records": records,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "positives_by_split": dict(sorted(positives.items())),
        "train_counts_by_class": dict(sorted(counts_by_class.items())),
        "negatives": dict(sorted(negatives.items())),
        "negatives_by_label_basis": dict(sorted(by_basis.items())),
        "negatives_by_tier": dict(sorted(by_tier.items())),
        "negatives_file": negatives_file,
        "negatives_sha256": None if negatives_file is None else negatives_file["sha256"],
        "mapillary": mapillary,
        "mapillary_sha256": None if mapillary is None else mapillary["sha256"],
        # Which Mapillary crops were chosen (region, share, cap): a resumed run must choose the same ones.
        "mapillary_selection_sha256": None if mapillary is None else mapillary["selection_sha256"],
        # The first stage of a two-stage run (Mapillary alone), or the start of the second (reader_start).
        "mapillary_only": mapillary_only,
        "start": start,
        "classes_from_start": classes_from_start,
        "folded_labels": folded,
        "min_kind_train_crops": MIN_KIND_TRAIN_CROPS,
        "negative_max_iou": NEGATIVE_MAX_IOU,
        "other_sign_rule_at": OTHER_SIGN_RULE_AT,
        "skipped": dict(sorted(skipped.items())),
        "train_without_reviewers": without,
        "unseen_validation_labels": unseen_validation,
        "trainable": trainable,
        "limitations": limitations,
    }


# The speed families: what every cutter finds. A number painted on the road ("30", "30 ZONE") is a speed limit too.
DETECTOR_FAMILIES = frozenset({
    "maximum_speed",
    "cancellation",
    "zone",
    "variable_display",
    "unreadable",
    "road_marking",
})
# What the cutter (the detector) learns to cut, its "families" (recorded in the plan, the configuration, the checkpoint
# and result.json): "speed", the speed-sign families above (every detector before this setting), or "all", those and
# the other traffic signs, so that the classifier sees them and says what they are. Both with the speed limits painted
# on the road; never the boxes marked "not a sign". The detector keeps one class (its name stays "speed_sign" for the checkpoints).
CUTTER_FAMILIES = {"speed": DETECTOR_FAMILIES, "all": DETECTOR_FAMILIES | {"other_sign"}}
DEFAULT_CUTTER_FAMILIES = "all"
# A detector checkpoint, plan or report that does not say was trained on the speed signs only.
LEGACY_CUTTER_FAMILIES = "speed"


# A variable display that is off: a dark or blank panel showing no number. The review records it as variable_display
# with value_kph null and a note; a lit display too far or too bright to read has a null value too, but is a sign to
# find. Only the note tells the two apart ("cut off by the top edge" is not off).
OFF_DISPLAY_NOTE = re.compile(r"(?<!cut )\b(off|blank|unlit|not active|spent[oaie]|vuot[oaie])\b|nessun valore")


def off_display(row: dict[str, Any]) -> bool:
    """A variable_display row without a value whose note says the panel is off or blank: an ignore box for the
    detector (neither a sign to find nor background), see prepare_detector_training."""
    if row.get("sign_family") != "variable_display" or written_sign_value(row) is not None:
        return False
    return OFF_DISPLAY_NOTE.search(str(row.get("notes") or "").lower()) is not None


def cutter_families(value: Any) -> str:
    """The families setting of a checkpoint, plan, report or configuration; "speed" when it does not say."""
    families = value.get("families", LEGACY_CUTTER_FAMILIES) if isinstance(value, dict) else LEGACY_CUTTER_FAMILIES
    if families not in CUTTER_FAMILIES:
        raise ValueError(f"unknown cutter families {families!r}: one of {', '.join(CUTTER_FAMILIES)}")
    return families


def prepare_detector_training(
    dataset_path: Path,
    split_path: Path,
    data_root: Path,
    *,
    dataset_kind: str = "gold",
    families: str = DEFAULT_CUTTER_FAMILIES,
    mapillary: Path | None = None,
    mapillary_region: str = "all",
    mapillary_share: float = 100.0,
    mapillary_only: bool = False,
    mapillary_ratio: float | None = 1.0,
) -> dict[str, Any]:
    """Build a full-frame detector plan from reviewed annotations.

    The boxes of the families chosen (CUTTER_FAMILIES: the speed signs, or those and the other signs) are the
    positive detector boxes; every frame keeps the family of each box (box_families). A frame is a
    detector negative only when it has an explicit whole-frame not_a_sign
    review. other_sign rows alone are not silently promoted to
    whole-frame negatives. A switched-off variable display (off_display) is an ignore box of its frame
    (ignore_boxes): SignNet neither learns it as a sign nor as background and is not measured on it; the cutter
    reads boxes only, so for it the panel is background.

    mapillary: the windows file of nnslr mapillary-windows (nnslr_tools.mtsd_windows), whose train windows of the
    region and share asked (percent) are added to train (source 'mapillary'), their speed boxes at most
    ``mapillary_ratio`` times our own train speed boxes (whole photos, a fixed order; None: no cap): MTSD's
    positives would else outweigh ours in the loss. mapillary_only, the first stage of a two-stage cutter: every
    window of the region and share and none of our train frames; validation and test stay ours, so both stages choose
    their epoch on our camera. counts_by_split, boxes_by_split, boxes_by_family and the dataset and split digests stay
    ours in every case (remeasure-signnet and the comparisons of Models read them); plan['mapillary'] has the
    windows' summary and plan['trained_on'] what train holds.
    """
    if families not in CUTTER_FAMILIES:
        raise ValueError(f"unknown cutter families {families!r}: one of {', '.join(CUTTER_FAMILIES)}")
    from nnslr_tools.mapillary import FULL_SHARE, check_share, region_text
    if mapillary is None and (region_text(mapillary_region) != "all" or check_share(mapillary_share) != FULL_SHARE):
        raise ValueError("--mapillary-region and --mapillary-share choose Mapillary windows: add --mapillary")
    if mapillary_only and mapillary is None:
        raise ValueError("--mapillary-only trains on the Mapillary windows alone: add --mapillary")
    if mapillary_ratio is not None and not mapillary_ratio > 0:
        raise ValueError(f"--mapillary-ratio {mapillary_ratio!r}: more than 0, the Mapillary speed boxes per one of "
                         "ours")
    positive_families = CUTTER_FAMILIES[families]
    dataset_path = Path(dataset_path)
    split_path = Path(split_path)
    data_root = Path(data_root)

    rows = load_jsonl(dataset_path)
    validation = validate_dataset(rows, data_root, dataset_kind=dataset_kind)
    if not validation.get("valid"):
        reasons = sorted({str(e.get("reason", "dataset_invalid")) for e in validation.get("errors", [])})
        raise ValueError("dataset_invalid: " + ",".join(reasons))

    resolved_split_path, split_doc = _load_split_document(split_path, data_root)
    split_errors = validate_splits(
        rows,
        split_doc,
        validation["dataset_sha256"],
        dataset_kind=dataset_kind,
    )
    if split_errors:
        reasons = sorted({str(e.get("reason", "split_invalid")) for e in split_errors})
        raise ValueError("split_invalid: " + ",".join(reasons))
    assignments = split_doc["assignments"]

    grouped: dict[str, dict[str, Any]] = {}
    ignored = Counter()
    for row in rows:
        if row.get("review_state") not in ELIGIBLE_REVIEW_STATES:
            ignored["rejected_or_pending"] += 1
            continue
        annotation_id = row["annotation_id"]
        split_name = assignments.get(annotation_id)
        if not isinstance(split_name, str):
            ignored["missing_split"] += 1
            continue
        key = row["image_sha256"]
        frame = grouped.setdefault(key, {
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "width": row["width"],
            "height": row["height"],
            "split": split_name,
            "route_id": row["route_id"],
            "segment_index": row["segment_index"],
            "decoded_frame_index": row["decoded_frame_index"],
            "boxes": [],
            "box_families": [],
            "ignore_boxes": [],
            "annotation_ids": [],
            "explicit_negative": False,
            "label_basis": _label_basis(row),
        })
        if (
            frame["image_path"] != row["image_path"]
            or frame["split"] != split_name
            or frame["width"] != row["width"]
            or frame["height"] != row["height"]
        ):
            raise ValueError("inconsistent_frame_group")
        if frame["label_basis"] != _label_basis(row):
            # The same image confirmed by different kinds of reviewer: it counts as neither.
            frame["label_basis"] = "mixed"

        family = row["sign_family"]
        if row["bbox_xyxy"] is None and family == "not_a_sign":
            frame["explicit_negative"] = True
            frame["annotation_ids"].append(annotation_id)
        elif row["bbox_xyxy"] is not None and family in positive_families and off_display(row):
            frame["ignore_boxes"].append(list(row["bbox_xyxy"]))
            frame["annotation_ids"].append(annotation_id)
        elif row["bbox_xyxy"] is not None and family in positive_families:
            frame["boxes"].append(list(row["bbox_xyxy"]))
            frame["box_families"].append(family)
            frame["annotation_ids"].append(annotation_id)
        else:
            ignored["non_detector_label"] += 1

    frames = []
    for frame in grouped.values():
        if not frame["boxes"] and not frame["ignore_boxes"] and not frame["explicit_negative"]:
            continue
        image = _safe_image_path(data_root, frame["image_path"])
        if not image.is_file():
            raise ValueError(f"missing_image: {frame['image_path']}")
        frames.append(frame)
    frames.sort(key=lambda r: (r["route_id"], r["segment_index"], r["image_path"]))

    counts_by_split = Counter(r["split"] for r in frames)
    boxes_by_split = Counter()
    ignore_by_split = Counter()
    negatives_by_split = Counter()
    boxes_by_family: dict[str, Counter] = {}
    for frame in frames:
        boxes_by_split[frame["split"]] += len(frame["boxes"])
        ignore_by_split[frame["split"]] += len(frame["ignore_boxes"])
        if frame["explicit_negative"] and not frame["boxes"]:
            negatives_by_split[frame["split"]] += 1
        for family in frame["box_families"]:
            boxes_by_family.setdefault(frame["split"], Counter())[family] += 1

    trainable = (
        counts_by_split[TRAIN_SPLIT] > 0
        and counts_by_split[VALIDATION_SPLIT] > 0
        and boxes_by_split[TRAIN_SPLIT] > 0
        and boxes_by_split[VALIDATION_SPLIT] > 0
    )

    limitations = [
        "single_class_vertical_speed_sign_detector" if families == "speed"
        else "single_class_sign_detector_speed_and_other_signs",
        "other_sign_not_assumed_whole_frame_negative",
        "no_runtime_bundle",
    ]
    if families == "all":
        # Before the other-sign rule an AI reviewer left the other traffic signs without a box: there they are
        # background for the cutter.
        limitations.append("other_signs_unboxed_in_ai_reviews_before_the_other_sign_rule")
    if dataset_kind == "training-candidate":
        limitations.append("model_review_candidates_are_not_human_gold")

    plan = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "task": "detector",
        "families": families,
        "dataset_kind": dataset_kind,
        "dataset_path": str(dataset_path),
        "dataset_sha256": validation["dataset_sha256"],
        "split_path": str(resolved_split_path),
        "requested_split_path": str(split_path),
        "split_sha256": _sha256(resolved_split_path),
        "classes": ["speed_sign"],
        "frames": frames,
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "boxes_by_split": dict(sorted(boxes_by_split.items())),
        "boxes_by_family": {split: dict(sorted(counts.items())) for split, counts in sorted(boxes_by_family.items())},
        "ignore_boxes_by_split": dict(sorted(ignore_by_split.items())),
        "negative_frames_by_split": dict(sorted(negatives_by_split.items())),
        "ignored": dict(sorted(ignored.items())),
        "trainable": trainable,
        "limitations": limitations,
    }
    ours_train = [frame for frame in frames if frame["split"] == TRAIN_SPLIT]
    ours_speed = sum(1 for frame in ours_train for family in frame["box_families"] if family in DETECTOR_FAMILIES)
    plan["mapillary"] = None
    plan["mapillary_only"] = bool(mapillary_only)
    plan["trained_on"] = {"ours": len(ours_train), "ours_boxes": sum(len(frame["boxes"]) for frame in ours_train),
                          "mapillary_windows": 0, "mapillary_boxes": 0}
    if mapillary is not None:
        from nnslr_tools.mtsd_windows import plan_frames
        cap = None if mapillary_only or mapillary_ratio is None else math.floor(mapillary_ratio * ours_speed)
        extra, summary = plan_frames(Path(mapillary), data_root, families=positive_families,
                                     region=mapillary_region, share=mapillary_share, own_frames=frames,
                                     max_speed_boxes=cap)
        summary.update({"ratio": None if mapillary_only else mapillary_ratio, "ours_train_speed_boxes": ours_speed})
        if mapillary_only:
            frames = [frame for frame in frames if frame["split"] != TRAIN_SPLIT]
        frames = frames + extra
        mapillary_boxes = sum(len(frame["boxes"]) for frame in extra)
        plan["frames"] = frames
        plan["mapillary"] = summary
        plan["trained_on"] = {"ours": 0 if mapillary_only else len(ours_train),
                              "ours_boxes": 0 if mapillary_only else plan["trained_on"]["ours_boxes"],
                              "mapillary_windows": len(extra), "mapillary_boxes": mapillary_boxes}
        train_boxes = plan["trained_on"]["ours_boxes"] + mapillary_boxes
        plan["trainable"] = (train_boxes > 0 and counts_by_split[VALIDATION_SPLIT] > 0
                             and boxes_by_split[VALIDATION_SPLIT] > 0)
    return plan


def _tracking_config(plan: dict[str, Any], config: Any, architecture: str) -> dict[str, Any]:
    """Hyperparameters plus dataset identity for the wandb run config."""
    values = dict(config.__dict__)
    values["task"] = plan["task"]
    values["architecture"] = architecture
    values["dataset_kind"] = plan["dataset_kind"]
    values["dataset_sha256"] = plan["dataset_sha256"]
    values["split_sha256"] = plan["split_sha256"]
    values["classes"] = list(plan["classes"])
    values["counts_by_split"] = dict(plan["counts_by_split"])
    if "boxes_by_split" in plan:
        values["boxes_by_split"] = dict(plan["boxes_by_split"])
    return values


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    # Stop when the validation has not improved for this many epochs (0: never stop early).
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    input_size: int = 160
    num_workers: int = DEFAULT_WORKERS
    seed: int = 0
    device: str = "cuda"
    pretrained: bool = False
    amp: bool = True
    # The labels (CLASS_SCHEMES): v1 the reader, v2 the classifier with not_a_sign and the kinds of other signs.
    class_scheme: str = "v1"
    # v2 only: the darker, lower-contrast photometric augmentation (CLASSIFIER_AUGMENTATION) and the jitter of the
    # box (CLASSIFIER_BOX_JITTER) of the training crops, in place of the light colour jitter of v1.
    augment: bool = False
    # v2 only: the crops file of nnslr mapillary-crops whose signs join train (nnslr_tools.mapillary), or None.
    mapillary: str | None = None
    # v2 with mapillary only: the photos of these continents and countries (nnslr_tools.mapillary.region_text), and
    # this share of them in percent (a fixed choice: the same share takes the same crops).
    mapillary_region: str = "all"
    mapillary_share: float = 100.0
    # v2 with mapillary only: the Mapillary crops alone in train, none of ours (the first stage of a two-stage run,
    # a start for init_from; prepare_reader_training).
    mapillary_only: bool = False
    # v2, a fresh run only: the reader-best.pt it starts from (nnslr train --init-from, the second stage of a
    # two-stage run) in place of ImageNet; its weights are copied, its head row by row by class name.
    init_from: str | None = None


# The settings a reader run saved before they existed was trained with: resuming such a run compares against these.
READER_CONFIG_BEFORE_SETTINGS = {"class_scheme": "v1", "augment": False, "mapillary": None,
                                 "mapillary_region": "all", "mapillary_share": 100.0, "mapillary_only": False,
                                 "init_from": None}

# The box of a training crop moved and resized before the crop, as the cutter's boxes are never exact: each side
# scaled by a factor in "scale", the centre moved by up to "shift" of the box's size, with this probability.
CLASSIFIER_BOX_JITTER = {"probability": 0.8, "scale": (0.85, 1.2), "shift": 0.1}
# The smallest crop, in pixels of the frame, on each side: a box thinner than that is widened around its centre.
MIN_CROP_SIDE = 2


def crop_rectangle(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The pixels of a box (x1, y1, x2, y2 in frame pixels) to cut from a frame of width x height: rounded
    outwards, inside the frame, at least MIN_CROP_SIDE pixels on each side. The classifier's crops (v2 training
    and evaluation, the chain) all come from here; the v1 reader keeps its own crop."""
    x1, y1, x2, y2 = (float(value) for value in box)
    left, top, right, bottom = math.floor(min(x1, x2)), math.floor(min(y1, y2)), math.ceil(max(x1, x2)), math.ceil(max(y1, y2))

    def widen(low: int, high: int, limit: int) -> tuple[int, int]:
        side = min(MIN_CROP_SIDE, limit)
        if high - low < side:
            middle = (low + high) / 2
            low = int(math.floor(middle - side / 2))
            high = low + side
        low = max(0, min(low, limit - side))
        high = min(limit, max(high, low + side))
        return low, high

    left, right = widen(max(0, left), min(int(width), right), int(width))
    top, bottom = widen(max(0, top), min(int(height), bottom), int(height))
    return left, top, right, bottom


def jitter_box(box: list[float], rng: random.Random, jitter: dict[str, Any] = CLASSIFIER_BOX_JITTER) -> list[float]:
    """A training box moved and resized at random (CLASSIFIER_BOX_JITTER), from a seeded random.Random; every
    value is drawn either way, so each crop uses the same amount of rng."""
    apply = rng.random() < jitter["probability"]
    low, high = jitter["scale"]
    scale_x, scale_y = rng.uniform(low, high), rng.uniform(low, high)
    shift_x, shift_y = rng.uniform(-jitter["shift"], jitter["shift"]), rng.uniform(-jitter["shift"], jitter["shift"])
    if not apply:
        return list(box)
    x1, y1, x2, y2 = (float(value) for value in box)
    width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    centre_x, centre_y = (x1 + x2) / 2 + shift_x * width, (y1 + y2) / 2 + shift_y * height
    half_w, half_h = width * scale_x / 2, height * scale_y / 2
    return [centre_x - half_w, centre_y - half_h, centre_x + half_w, centre_y + half_h]


def _check_reader_config(config: "TrainConfig") -> None:
    if config.class_scheme not in CLASS_SCHEMES:
        raise ValueError(f"invalid reader class_scheme {config.class_scheme!r}: one of {', '.join(CLASS_SCHEMES)}")
    if config.augment and config.class_scheme != "v2":
        raise ValueError("the photometric augmentation and box jitter are settings of the v2 classifier")


def _load_start(torch, model, saved: dict[str, Any], classes: list[str], input_size: int, name: str,
                origin: str) -> None:
    """The weights of a start (nnslr train --init-from) in a fresh classifier: every layer copied, the rows of the
    head (classifier.3) by class name; a class the start lacks keeps the fresh row of the new head."""
    if saved.get("task") != "reader" or saved.get("architecture") != "mobilenet_v3_small":
        raise ValueError("the start is not a reader/classifier checkpoint")
    if checkpoint_class_scheme(saved) != "v2":
        raise ValueError("the start is not a v2 classifier")
    if saved.get("input_size") != input_size:
        raise ValueError(f"the start has input size {saved.get('input_size')}, this run {input_size}")
    weights = model_weights(saved)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    head = f"classifier.{len(model.classifier) - 1}."
    body = {key for key in state if not key.startswith(head)}
    if body != {key for key in weights if not key.startswith(head)}:
        raise ValueError("the start's layers are not those of this network")
    for key in body:
        state[key] = weights[key]
    rows = {label: row for row, label in enumerate(saved.get("classes") or [])}
    carried = [label for label in classes if label in rows]
    for row, label in enumerate(classes):
        if label in rows:
            state[head + "weight"][row] = weights[head + "weight"][rows[label]]
            state[head + "bias"][row] = weights[head + "bias"][rows[label]]
    model.load_state_dict(state, strict=True)
    new = [label for label in classes if label not in rows]
    print(f"{name} starts from {origin}: every layer, and the head of {len(carried)} of its {len(classes)} classes"
          + (f" (new: {', '.join(new)})" if new else ""), flush=True)


def _sample_seed(*parts: Any) -> int:
    """A 63-bit seed from text parts (the same on every machine and in every loader worker)."""
    return int.from_bytes(hashlib.sha256(" ".join(str(part) for part in parts).encode()).digest()[:8], "little") >> 1


def _require_training_stack():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise ValueError(
            "training_dependencies_missing: install a V100-compatible PyTorch/torchvision "
            "environment plus Pillow; training dependencies are intentionally not part of "
            "the core CPU profile"
        ) from exc
    return torch, Image, DataLoader, Dataset, transforms, MobileNet_V3_Small_Weights, mobilenet_v3_small


def gpu_smoke(device_index: int = 0) -> dict[str, Any]:
    """Run an explicit CUDA forward/backward smoke test.

    This function is never called implicitly. It verifies that the installed
    PyTorch binary can actually execute kernels for the selected GPU.
    """
    torch, *_ = _require_training_stack()
    if not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(f"cuda_device_out_of_range: {device_index}")

    device = torch.device(f"cuda:{device_index}")
    capability = tuple(torch.cuda.get_device_capability(device))
    compiled_arches = list(torch.cuda.get_arch_list())
    expected_arch = f"sm_{capability[0]}{capability[1]}"
    if expected_arch not in compiled_arches:
        raise ValueError(
            f"cuda_arch_not_compiled: device={expected_arch} compiled={compiled_arches}"
        )

    # Small enough to be a smoke test, large enough to execute real FP16 GEMM
    # and autograd kernels. This intentionally allocates GPU memory.
    x = torch.randn((1024, 1024), device=device, dtype=torch.float16, requires_grad=True)
    y = (x @ x.T).mean()
    y.backward()
    torch.cuda.synchronize(device)

    props = torch.cuda.get_device_properties(device)
    return {
        "status": "available",
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "device_index": device_index,
        "device_name": props.name,
        "capability": list(capability),
        "compiled_arches": compiled_arches,
        "total_memory_bytes": int(props.total_memory),
        "fp16_forward_backward": True,
        "finite_result": bool(torch.isfinite(y.detach()).item()),
    }


@dataclass(frozen=True)
class DetectorTrainConfig:
    epochs: int = 20
    # Stop when the validation loss has not improved for this many epochs (0: never stop early).
    patience: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = DEFAULT_WORKERS
    seed: int = 0
    device: str = "cuda"
    pretrained_backbone: bool = False
    amp: bool = True
    # The image the detector sees, (width, height): the frame is resized to it inside the model.
    input_size: tuple[int, int] = DEFAULT_DETECTOR_INPUT_SIZE
    # The smallest default box, as a fraction of the image side (nnslr_tools.checkpoints.build_detector).
    anchor_min_ratio: float = DEFAULT_DETECTOR_ANCHOR_MIN_RATIO
    # Photometric augmentation of the training frames (DETECTOR_AUGMENTATION).
    augment: bool = True
    # The share of the frames without signs that an epoch uses: torchvision's SSD takes its negative boxes only
    # inside frames with signs, so a frame without any gives no loss and only costs time. Each epoch draws its
    # own share; validation keeps one fixed draw, so its epochs compare. 1.0: every frame, as before.
    negative_frame_fraction: float = 0.1
    # What chooses the best epoch, and what early stopping waits for: "recall" (validation recall at score
    # SELECT_SCORE_THRESHOLD, ties to fewer false boxes per frame: the detector is the cutter, what it misses
    # is lost) or "loss" (validation loss, as before).
    select_by: str = "recall"
    # What the cutter learns to cut (CUTTER_FAMILIES): "all", the speed signs and the other traffic signs (the
    # default), or "speed", the speed signs only (every detector before this setting).
    families: str = DEFAULT_CUTTER_FAMILIES


# The settings a detector run saved before they existed was trained with: resuming such a run compares
# against these.
DETECTOR_CONFIG_BEFORE_SETTINGS = {"input_size": DEFAULT_DETECTOR_INPUT_SIZE,
                                   "anchor_min_ratio": DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, "augment": False,
                                   "negative_frame_fraction": 1.0, "select_by": "loss",
                                   "families": LEGACY_CUTTER_FAMILIES}
DETECTOR_SELECT_BY = ("recall", "loss")
# The validation measure of select_by "recall": a reviewed box is found by a box of score >= this with IoU >= 0.5.
SELECT_SCORE_THRESHOLD = 0.1
SELECT_IOU_THRESHOLD = 0.5


def epoch_frame_indices(frames: list[dict[str, Any]], fraction: float, seed: Any) -> list[int]:
    """The frames an epoch (or the validation) uses, in their order: every frame with signs, and
    round(fraction x the frames without) of those without, drawn with seed."""
    with_signs = [index for index, frame in enumerate(frames) if frame["boxes"]]
    sign_free = [index for index, frame in enumerate(frames) if not frame["boxes"]]
    count = min(len(sign_free), int(fraction * len(sign_free) + 0.5))
    chosen = sign_free if count == len(sign_free) else random.Random(seed).sample(sign_free, count)
    return sorted(with_signs + chosen)

# Photometric augmentation of the detector's training frames, never of validation, evaluation or export. The
# test route is mostly at dusk (35% of its frames are dark, 7% in train), so the brightness range leans to
# darkening. No flip and no geometric change: the side of the road a sign stands on matters, and the boxes stay
# where they are. Each frame is changed with this probability, every factor drawn uniformly in its range.
DETECTOR_AUGMENTATION = {
    "probability": 0.8,
    "brightness": (0.35, 1.25),
    # Down to half: mist, rain and spray wash a frame out.
    "contrast": (0.5, 1.3),
    "gamma": (0.8, 1.6),
    "saturation": (0.7, 1.3),
    # A light colour cast: each of R, G and B multiplied by its own factor.
    "channel_gain": (0.92, 1.08),
    # Drops and spray on the windscreen: a light Gaussian blur on this share of the frames changed, of a sigma (in
    # pixels of the image changed) drawn in its range.
    "blur_probability": 0.3,
    "blur_sigma": (0.3, 1.2),
}
# The classifier's training crops: the same. Until 26 October 2026 only the crops went down to half the contrast,
# and nothing was blurred.
CLASSIFIER_AUGMENTATION = dict(DETECTOR_AUGMENTATION)


# The values of photometric_draws, in this order: whether it applies (1.0 or 0.0), the factors, whether it blurs.
PHOTOMETRIC_DRAWS = ("apply", "gamma", "brightness", "contrast", "saturation", "gain_r", "gain_g", "gain_b", "blur",
                     "sigma")


def photometric_draws(torch, generator, ranges=None) -> list[float]:
    """The random values of detector_photometric (PHOTOMETRIC_DRAWS), from the CPU generator: every factor is drawn
    either way, so each image uses the same amount of the generator."""
    ranges = DETECTOR_AUGMENTATION if ranges is None else ranges

    def draw(name, count=1):
        low, high = ranges[name]
        return (torch.rand(count, generator=generator, dtype=torch.float64) * (high - low) + low).tolist()

    apply = float(torch.rand(1, generator=generator)) < ranges["probability"]
    gamma, brightness, contrast, saturation = (draw(name)[0] for name in ("gamma", "brightness", "contrast",
                                                                           "saturation"))
    gains = draw("channel_gain", 3)
    blur = float(torch.rand(1, generator=generator)) < ranges.get("blur_probability", 0.0)
    sigma = draw("blur_sigma")[0] if "blur_sigma" in ranges else 0.0
    return [float(apply), gamma, brightness, contrast, saturation, *gains, float(blur), sigma]


def photometric_batch(torch, TF, images, draws):
    """detector_photometric on a batch (B x 3 x H x W, values 0..1, on the GPU): each image with its own row of
    photometric_draws (B x len(PHOTOMETRIC_DRAWS)), the same operations in the same order as torchvision's
    adjust_gamma, adjust_brightness, adjust_contrast and adjust_saturation; an image whose draw does not apply is
    left as it is. The classifier's loader workers only draw: the GPU changes the pixels."""
    draws = draws.to(device=images.device, dtype=images.dtype)

    def column(name):
        return draws[:, PHOTOMETRIC_DRAWS.index(name)].view(-1, 1, 1, 1)

    def grey(batch):
        red, green, blue = batch.unbind(dim=-3)
        return (0.2989 * red + 0.587 * green + 0.114 * blue).unsqueeze(dim=-3)

    def blend(first, second, ratio):
        return (ratio * first + (1.0 - ratio) * second).clamp(0.0, 1.0)

    changed = images.pow(column("gamma")).clamp(0.0, 1.0)
    changed = blend(changed, torch.zeros_like(changed), column("brightness"))
    changed = blend(changed, grey(changed).mean(dim=(-3, -2, -1), keepdim=True), column("contrast"))
    changed = blend(changed, grey(changed), column("saturation"))
    gains = draws[:, PHOTOMETRIC_DRAWS.index("gain_r"):PHOTOMETRIC_DRAWS.index("gain_b") + 1].view(-1, 3, 1, 1)
    changed = (changed * gains).clamp(0.0, 1.0)
    sigmas = draws[:, PHOTOMETRIC_DRAWS.index("sigma")].tolist()
    blurred = (draws[:, PHOTOMETRIC_DRAWS.index("blur")] > 0.5).tolist()
    for index, (blur, sigma) in enumerate(zip(blurred, sigmas)):
        if blur and sigma > 0:
            side = 2 * math.ceil(3 * sigma) + 1
            changed[index] = TF.gaussian_blur(changed[index], [side, side], [sigma, sigma])
    return torch.where(column("apply") > 0.5, changed, images)


def detector_photometric(torch, TF, image, generator, ranges=None):
    """A training frame (3 x H x W, values 0..1, on any device) with random gamma, brightness, contrast,
    saturation, colour cast and sometimes a light blur (DETECTOR_AUGMENTATION, or the ranges given:
    CLASSIFIER_AUGMENTATION for the classifier's crops; ranges without the blur's keys never blur); the factors come
    from the CPU generator (photometric_draws)."""
    apply, gamma, brightness, contrast, saturation, *rest = photometric_draws(torch, generator, ranges)
    gains, blur, sigma = rest[:3], rest[3] > 0.5, rest[4]
    if not apply:
        return image
    image = TF.adjust_gamma(image, gamma)
    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)
    image = TF.adjust_saturation(image, saturation)
    gains = torch.tensor(gains, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = (image * gains).clamp_(0.0, 1.0)
    if blur and sigma > 0:
        side = 2 * math.ceil(3 * sigma) + 1
        image = TF.gaussian_blur(image, [side, side], [sigma, sigma])
    return image


def _check_detector_config(config: "DetectorTrainConfig") -> None:
    size = config.input_size
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or not all(type(side) is int and 32 <= side <= MAX_DETECTOR_INPUT_SIDE for side in size)
    ):
        raise ValueError(f"invalid detector input_size {size!r}: (width, height), 32 to {MAX_DETECTOR_INPUT_SIDE}")
    ratio = config.anchor_min_ratio
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio < DETECTOR_ANCHOR_MAX_RATIO:
        raise ValueError(f"invalid detector anchor_min_ratio {ratio!r}: between 0 and {DETECTOR_ANCHOR_MAX_RATIO}")
    fraction = config.negative_frame_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError(f"invalid detector negative_frame_fraction {fraction!r}: from 0 to 1")
    if config.select_by not in DETECTOR_SELECT_BY:
        raise ValueError(f"invalid detector select_by {config.select_by!r}: one of {', '.join(DETECTOR_SELECT_BY)}")
    if config.families not in CUTTER_FAMILIES:
        raise ValueError(f"invalid detector families {config.families!r}: one of {', '.join(CUTTER_FAMILIES)}")


def _detector_config_differences(saved: Any, config: "DetectorTrainConfig") -> list[str]:
    """The settings of a saved detector run that differ from config (a run saved before a setting existed
    was trained with DETECTOR_CONFIG_BEFORE_SETTINGS)."""
    before = {**DETECTOR_CONFIG_BEFORE_SETTINGS, **(saved if isinstance(saved, dict) else {})}
    if isinstance(before.get("input_size"), list):
        before["input_size"] = tuple(before["input_size"])
    current = config.__dict__
    # The number of frame readers changes nothing in what is learnt.
    return [f"{key} {before.get(key)!r} -> {current.get(key)!r}" for key in sorted(set(before) | set(current))
            if before.get(key) != current.get(key) and key != "num_workers"]


# The flags of nnslr train for each setting of a saved configuration (bool settings are switches).
_RESUME_FLAGS = (("epochs", "--epochs"), ("patience", "--patience"), ("batch_size", "--batch-size"),
                 ("learning_rate", "--learning-rate"), ("weight_decay", "--weight-decay"),
                 ("num_workers", "--num-workers"), ("seed", "--seed"), ("device", "--device"),
                 ("anchor_min_ratio", "--anchor-min-ratio"), ("negative_frame_fraction", "--negative-frame-fraction"),
                 ("select_by", "--select-by"), ("families", "--cutter-families"))


def resume_arguments(state: dict[str, Any]) -> list[str]:
    """The nnslr train flags that give a saved run (its training-state.pt) back its configuration: a resumed
    run must train with exactly the settings it started with, ImageNet start (--pretrained) included."""
    task = state.get("task") if isinstance(state, dict) else None
    config = state.get("config") if isinstance(state, dict) else None
    if task not in ("detector", "reader") or not isinstance(config, dict):
        raise ValueError("invalid training state: no task or configuration")
    if task == "detector":
        config = {**DETECTOR_CONFIG_BEFORE_SETTINGS, **config}
    else:
        config = {**READER_CONFIG_BEFORE_SETTINGS, **config}
    flags: list[str] = []
    for key, flag in _RESUME_FLAGS:
        if key in config:
            flags += [flag, str(config[key])]
    size = config.get("input_size")
    if task == "detector" and size is not None:
        flags += ["--input-size", f"{int(size[0])}x{int(size[1])}"]
    elif size is not None:
        flags += ["--input-size", str(int(size))]
    if config.get("pretrained_backbone" if task == "detector" else "pretrained"):
        flags.append("--pretrained")
    if config.get("amp") is False:
        flags.append("--no-amp")
    if task == "detector" and config.get("augment") is False:
        flags.append("--no-augment")
    if task == "reader" and config.get("class_scheme") == "v2":
        # The classifier: its scheme, and its augmentation only when it trained without (on by default for v2).
        flags += ["--classes", "v2"]
        if config.get("augment") is False:
            flags.append("--no-augment")
        if config.get("mapillary"):
            flags += ["--mapillary", str(config["mapillary"])]
            if config.get("mapillary_region", "all") != "all":
                flags += ["--mapillary-region", str(config["mapillary_region"])]
            if float(config.get("mapillary_share", 100.0)) != 100.0:
                flags += ["--mapillary-share", f"{float(config['mapillary_share']):g}"]
            if config.get("mapillary_only"):
                flags.append("--mapillary-only")
        if config.get("init_from"):
            flags += ["--init-from", str(config["init_from"])]
    return flags


def resume_arguments_file(path: Path) -> list[str]:
    """resume_arguments of a training-state.pt (scripts/run_v100_baseline.sh, scripts/run_detector_sizes.sh)."""
    return resume_arguments(torch_load(Path(path)))


def _require_detector_stack():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision.transforms import functional as TF
        from torchvision.models import MobileNet_V3_Large_Weights
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    except ImportError as exc:
        raise ValueError(
            "training_dependencies_missing: detector training needs a V100-compatible "
            "PyTorch/torchvision environment plus Pillow"
        ) from exc
    return (
        torch,
        Image,
        DataLoader,
        Dataset,
        TF,
        MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )


def train_detector(
    plan: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    config: DetectorTrainConfig,
    *,
    mode: str = "fresh",
    hf_repo: str | None = None,
    hf_revision: str = "main",
    init_from: Path | None = None,
) -> dict[str, Any]:
    """Train the first single-class full-frame speed-sign detector baseline (with the other signs too: families)."""
    _check_detector_config(config)
    if cutter_families(plan) != config.families:
        raise ValueError(f"the plan's boxes are the families {cutter_families(plan)}, the training configuration "
                         f"{config.families} (--cutter-families)")
    if not plan.get("trainable"):
        raise ValueError(
            "training_plan_not_trainable: detector needs positive train and validation frames"
        )
    # The measure of the validation (evaluation imports this module: imported here, not at the top).
    from nnslr_tools.evaluation import detector_metrics, score_frame

    (
        torch,
        Image,
        DataLoader,
        Dataset,
        TF,
        BackboneWeights,
        ssdlite320_mobilenet_v3_large,
    ) = _require_detector_stack()
    import torchvision

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    root = Path(data_root).resolve()

    class DetectionDataset(Dataset):
        def __init__(self, frames):
            self.frames = frames

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, index):
            row = self.frames[index]
            path = _safe_image_path(root, row["image_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
                tensor = TF.pil_to_tensor(image).float().div_(255.0)

            boxes = torch.tensor(row["boxes"], dtype=torch.float32)
            if boxes.numel() == 0:
                boxes = boxes.reshape(0, 4)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([index], dtype=torch.int64),
            }
            return tensor, target

    def collate(batch):
        return tuple(zip(*batch))

    train_frames = [r for r in plan["frames"] if r["split"] == TRAIN_SPLIT]
    all_val_frames = [r for r in plan["frames"] if r["split"] == VALIDATION_SPLIT]
    # Validation: every frame with signs and one fixed draw of those without, the same frames every epoch.
    val_frames = [all_val_frames[index] for index in
                  epoch_frame_indices(all_val_frames, config.negative_frame_fraction, f"validation {config.seed}")]
    val_dataset = DetectionDataset(val_frames)
    generator = torch.Generator().manual_seed(config.seed)

    class EpochSampler(torch.utils.data.Sampler):
        """The frames of the current epoch (set_epoch) in random order. With every frame (fraction 1.0) the
        order is the one of shuffle=True with the same generator."""

        def __init__(self):
            self.indices = list(range(len(train_frames)))

        def set_epoch(self, epoch: int) -> None:
            self.indices = epoch_frame_indices(train_frames, config.negative_frame_fraction,
                                               f"train {config.seed} {epoch}")

        def __len__(self):
            # A last batch of one frame fails in BatchNorm (the last feature map is 1 x 1 at 320 x 320): that
            # frame, the last of the shuffle, waits for another epoch.
            count = len(self.indices)
            return count - 1 if count > 1 and count % config.batch_size == 1 else count

        def __iter__(self):
            order = torch.randperm(len(self.indices), generator=generator).tolist()
            return iter([self.indices[position] for position in order][:len(self)])

    train_sampler = EpochSampler()
    train_sampler.set_epoch(0)
    train_loader = DataLoader(
        DetectionDataset(train_frames),
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        generator=generator,
    )
    sign_free_train = sum(1 for frame in train_frames if not frame["boxes"])
    frame_counts = {
        "train_with_signs": len(train_frames) - sign_free_train,
        "train_sign_free": sign_free_train,
        "train_sign_free_per_epoch": len(train_sampler.indices) - (len(train_frames) - sign_free_train),
        "train_frames_per_epoch": len(train_sampler),
        "validation_with_signs": sum(1 for frame in all_val_frames if frame["boxes"]),
        "validation_sign_free": sum(1 for frame in all_val_frames if not frame["boxes"]),
        "validation_sign_free_used": sum(1 for frame in val_frames if not frame["boxes"]),
    }
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    # ImageNet backbone weights for a fresh run; continue and resume load their own weights below.
    geometry = {"input_size": config.input_size, "anchor_min_ratio": config.anchor_min_ratio}
    model = build_detector(wide=config.pretrained_backbone and mode == "fresh", **geometry)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "training-state.pt"
    if mode not in ("fresh", "continue", "resume"):
        raise ValueError(f"invalid training mode: {mode}")
    if mode == "resume" and not checkpoint_path.is_file():
        raise ValueError(f"no resumable training checkpoint: {checkpoint_path}")
    plan_path = output_dir / "training-plan.json"

    def move_targets(targets):
        return [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]

    def validate() -> dict[str, Any]:
        """The validation loss, and the boxes the cutter passes on scored as nnslr evaluate scores them:
        recall at SELECT_SCORE_THRESHOLD, false boxes and boxes per frame. One pass over the frames."""
        total = 0.0
        seen = 0
        scored = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device, non_blocking=True) for img in images]
                # torchvision detection models expose training losses only in train mode.
                # Keep BatchNorm frozen so validation does not mutate running statistics.
                model.train()
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    losses = model(images, move_targets(targets))
                    loss = sum(losses.values())
                total += float(loss.detach()) * len(images)
                seen += len(images)
                # The boxes, in float32 like the evaluation.
                model.eval()
                for detection, target in zip(model(images), targets):
                    keep = detection["labels"] == 1
                    predictions = [{"box": box, "score": score} for box, score in
                                   zip(detection["boxes"][keep].float().cpu().tolist(),
                                       detection["scores"][keep].float().cpu().tolist())]
                    scored.append(score_frame(target["boxes"].tolist(), predictions,
                                              score_threshold=SELECT_SCORE_THRESHOLD,
                                              iou_threshold=SELECT_IOU_THRESHOLD))
        model.train()
        measured = detector_metrics(scored)
        return {
            "loss": total / max(seen, 1),
            "recall": measured["recall_iou"],
            "true_positives": measured["true_positives"],
            "false_positives": measured["false_positives"],
            "false_positives_per_image": measured["false_positives_per_image"],
            "predictions_per_image": measured["predictions_per_image"],
        }

    def selection_key(measured: dict[str, Any]) -> tuple | None:
        """Higher is better; None never counts as better (a validation loss that is not a number)."""
        if config.select_by == "recall":
            # The same validation frames every epoch: counts compare as recall and false boxes per frame do.
            return (measured["true_positives"], -measured["false_positives"])
        return (-measured["loss"],) if math.isfinite(measured["loss"]) else None

    def detector_samples() -> list[Any]:
        # The first validation frames with reviewed boxes (fixed order, so the
        # same frames every time), with predicted and reviewed boxes.
        chosen = []
        for index in range(len(val_frames)):
            if len(val_frames[index]["boxes"]) > 0:
                chosen.append(index)
            if len(chosen) >= tracking.DETECTOR_IMAGE_SAMPLES:
                break
        samples = []
        model.eval()
        with torch.no_grad():
            for index in chosen:
                frame = val_frames[index]
                tensor, _ = val_dataset[index]
                prediction = model([tensor.to(device)])[0]
                height = int(tensor.shape[1])
                width = int(tensor.shape[2])
                predicted = tracking.detector_box_data(
                    prediction["boxes"].detach().float().cpu().tolist(),
                    width,
                    height,
                    scores=prediction["scores"].detach().float().cpu().tolist(),
                )
                reviewed = tracking.detector_box_data(frame["boxes"], width, height)
                boxes = {
                    "predictions": {"box_data": predicted, "class_labels": tracking.DETECTOR_CLASS_LABELS},
                    "ground_truth": {"box_data": reviewed, "class_labels": tracking.DETECTOR_CLASS_LABELS},
                }
                image = TF.to_pil_image(tensor)
                image.thumbnail((tracking.IMAGE_MAX_SIDE, tracking.IMAGE_MAX_SIDE))
                caption = (
                    f"{frame['image_path']}: {len(reviewed)} reviewed, {len(predicted)} predicted "
                    f"(score >= {tracking.DETECTOR_IMAGE_MIN_SCORE})"
                )
                samples.append(tracker.image(image, caption, boxes=boxes))
        model.train()
        return samples

    history = []
    best_validation_loss = float("inf")
    # The epoch whose weights are detector-best.pt: its selection key and its validation measures.
    best_selection: dict[str, Any] | None = None
    stale = 0
    best_path = output_dir / "detector-best.pt"
    global_step = 0

    def match_detector_tail(model, optimizer, weights):
        # Weights trained from ImageNet have the full tail: a run continued or resumed from them needs that shape.
        if detector_is_wide(weights) == detector_is_wide(model.state_dict()):
            return model, optimizer
        model = build_detector(wide=detector_is_wide(weights), **geometry).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        return model, optimizer

    start_epoch = 0
    if mode == "continue":
        if init_from is not None:
            # The best weights of an earlier run of this server, checked as the stable ones are.
            saved, origin = torch_load(Path(init_from)), str(init_from)
        elif not hf_repo:
            raise ValueError("continue requires --hf-repo or --init-from")
        else:
            saved, origin = stable_weights(hf_repo, "detector", revision=hf_revision)
        if saved.get("task") != "detector" or saved.get("architecture") != "ssdlite320_mobilenet_v3_large":
            raise ValueError("stable weight task/architecture mismatch")
        if saved.get("classes") != ["background", "speed_sign"]:
            raise ValueError("stable detector classes mismatch")
        model, optimizer = match_detector_tail(model, optimizer, model_weights(saved))
        model.load_state_dict(model_weights(saved), strict=True)
        print(f"continue detector from weights: {origin}")
        # No weight depends on the input size or on the default boxes: the new run uses its own.
        saved_size = tuple(saved.get("input_size") or DEFAULT_DETECTOR_INPUT_SIZE)
        if saved_size != config.input_size:
            print(f"the weights were trained at {saved_size[0]}x{saved_size[1]}; this run trains at "
                  f"{config.input_size[0]}x{config.input_size[1]}")
    elif mode == "resume":
        saved = torch_load(checkpoint_path)
        if saved.get("format_version") != 1 or saved.get("task") != "detector":
            raise ValueError("invalid detector training-state checkpoint")
        if saved.get("status") == "completed":
            raise ValueError("training already completed; use --mode continue for a new run")
        if saved.get("status") != "in_progress":
            raise ValueError("invalid checkpoint status")
        differences = _detector_config_differences(saved.get("config"), config)
        if differences:
            raise ValueError("resume requires exactly the original training configuration: " + "; ".join(differences))
        if saved.get("dataset_sha256") != plan["dataset_sha256"] or saved.get("split_sha256") != plan["split_sha256"]:
            raise ValueError("resume dataset/split mismatch")
        start_epoch = saved.get("epoch")
        if not isinstance(start_epoch, int) or not 0 <= start_epoch < config.epochs:
            raise ValueError("invalid or already completed resume epoch")
        model, optimizer = match_detector_tail(model, optimizer, model_weights(saved))
        model.load_state_dict(model_weights(saved), strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        global_step = saved["global_step"]
        history = saved["history"]
        best_validation_loss = saved["best_validation_loss"]
        best_selection = saved.get("best_selection")
        if best_selection is None and math.isfinite(best_validation_loss):
            # A run saved before best_selection existed chose its best epoch by validation loss.
            best_selection = {"key": [-best_validation_loss], "validation_loss": best_validation_loss}
        stale = int(saved.get("stale", 0))
        print(f"resume detector at epoch {start_epoch}/{config.epochs}")

    def save_training_state(epoch: int, status: str) -> None:
        atomic_torch_save({
            "format_version": 1,
            "task": "detector",
            "architecture": "ssdlite320_mobilenet_v3_large",
            "status": status,
            "epoch": epoch,
            "global_step": global_step,
            "config": config.__dict__,
            "dataset_sha256": plan["dataset_sha256"],
            "split_sha256": plan["split_sha256"],
            "classes": ["background", "speed_sign"],
            "families": config.families,
            "input_size": list(config.input_size),
            "anchor_min_ratio": config.anchor_min_ratio,
            "best_validation_loss": best_validation_loss,
            "best_selection": best_selection,
            "stale": stale,
            "history": history,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, checkpoint_path)

    if mode != "resume":
        output_dir.mkdir(parents=True, exist_ok=False)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_training_state(0, "in_progress")

    tracker = tracking.start_training_run(
        "detector",
        output_dir,
        _tracking_config(plan, config, "ssdlite320_mobilenet_v3_large"),
    )
    print(f"detector input {config.input_size[0]}x{config.input_size[1]}, smallest default box "
          f"{config.anchor_min_ratio:g} of the image, photometric augmentation {'on' if config.augment else 'off'}",
          flush=True)
    print("detector boxes: " + ("the speed signs" if config.families == "speed" else
                                "the speed signs and the other traffic signs") + f" (families {config.families})",
          flush=True)
    # The factors of the photometric augmentation, drawn here (not in the loader workers): the same for any
    # number of workers, and reproduced after a resume.
    augment_generator = torch.Generator().manual_seed(config.seed)
    print(f"detector frames per epoch: {frame_counts['train_with_signs']} with signs and "
          f"{frame_counts['train_sign_free_per_epoch']} of {frame_counts['train_sign_free']} without; validation "
          f"{frame_counts['validation_with_signs']} with signs and {frame_counts['validation_sign_free_used']} of "
          f"{frame_counts['validation_sign_free']} without; best epoch by "
          + (f"validation recall at score {SELECT_SCORE_THRESHOLD:g}" if config.select_by == "recall"
             else "validation loss"), flush=True)
    try:
        stopped_early = None
        for epoch in range(start_epoch, config.epochs):
            epoch_number = epoch + 1
            epoch_start = time.perf_counter()
            # Reproduce each epoch's frames and DataLoader shuffle after an epoch-boundary resume.
            train_sampler.set_epoch(epoch)
            generator.manual_seed(config.seed + epoch)
            augment_generator.manual_seed(config.seed + 1_000_003 * (epoch + 1))
            model.train()
            train_total = 0.0
            train_seen = 0
            # The step time includes the wait for the batch, like distill's step_time.
            step_start = time.perf_counter()
            batch_count = len(train_loader)
            batch_index = 0
            for images, targets in train_loader:
                # Where the training is, batch by batch, for the panel of the review site.
                batch_index += 1
                report_progress("detector", epoch * batch_count + batch_index, config.epochs * batch_count,
                                f"epoch {epoch_number} of {config.epochs}, batch {batch_index} of {batch_count}")
                images = [img.to(device, non_blocking=True) for img in images]
                if config.augment:
                    images = [detector_photometric(torch, TF, image, augment_generator) for image in images]
                targets = move_targets(targets)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_loss = float(loss.detach())
                train_total += step_loss * len(images)
                train_seen += len(images)
                global_step += 1
                step_end = time.perf_counter()

                if tracker.enabled:
                    step_metrics = {"train/loss": step_loss}
                    for name in losses:
                        step_metrics["train/loss_" + name] = float(losses[name].detach())
                    step_metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                    step_metrics["train/step_time"] = step_end - step_start
                    tracker.log(step_metrics, step=global_step)
                step_start = step_end

            measured = validate()
            val_loss = measured["loss"]
            epoch_result = {
                "epoch": epoch_number,
                "train_loss": train_total / max(train_seen, 1),
                "validation_loss": val_loss,
                # The cutter on the validation frames, at score SELECT_SCORE_THRESHOLD.
                "validation_recall": measured["recall"],
                "validation_false_positives_per_image": measured["false_positives_per_image"],
                "validation_predictions_per_image": measured["predictions_per_image"],
                # Training and validation of the epoch, for the cost of an input size.
                "seconds": time.perf_counter() - epoch_start,
            }
            history.append(epoch_result)

            key = selection_key(measured)
            improved = key is not None and (best_selection is None or list(key) > best_selection["key"])
            if improved:
                best_validation_loss = val_loss
                best_selection = {"key": list(key), "epoch": epoch_number, "validation_loss": val_loss,
                                  "validation_recall": measured["recall"],
                                  "validation_false_positives_per_image": measured["false_positives_per_image"]}
                torch.save({
                    "format_version": 1,
                    "architecture": "ssdlite320_mobilenet_v3_large",
                    "task": "detector",
                    "classes": ["background", "speed_sign"],
                    # What its one class means: the speed signs, or the other signs too (CUTTER_FAMILIES).
                    "families": config.families,
                    # What load_detector rebuilds the detector with (nnslr_tools.checkpoints).
                    "input_size": list(config.input_size),
                    "anchor_min_ratio": config.anchor_min_ratio,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": plan["dataset_sha256"],
                    "split_sha256": plan["split_sha256"],
                    "epoch": epoch_number,
                    "validation_loss": best_validation_loss,
                    "validation_recall": measured["recall"],
                    "validation_false_positives_per_image": measured["false_positives_per_image"],
                    "selected_by": config.select_by,
                    "initialization": "imagenet1k_v1" if config.pretrained_backbone else "random",
                    "initialization_source": (
                        "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1"
                        if config.pretrained_backbone else "random"
                    ),
                    "torch_version": str(torch.__version__),
                    "torchvision_version": torchvision.__version__,
                }, best_path)

            if tracker.enabled:
                epoch_metrics = {
                    "epoch": epoch_number,
                    "train/epoch_loss": epoch_result["train_loss"],
                    "validation/loss": val_loss,
                    "validation/best_loss": best_validation_loss,
                    "validation/recall": measured["recall"],
                    "validation/false_per_image": measured["false_positives_per_image"],
                }
                if tracker.should_log_images(epoch_number, config.epochs):
                    epoch_metrics["validation/detections"] = detector_samples()
                epoch_metrics["epoch_time"] = time.perf_counter() - epoch_start
                tracker.log(epoch_metrics, step=global_step)
            # Early stop: the best weights are already saved; more epochs would only memorise the training images.
            stale = 0 if improved else stale + 1
            save_training_state(epoch_number, "in_progress")
            # One line per epoch: the panel of the review site reads the progress from the log (the seconds last).
            print(f"detector epoch {epoch_number}/{config.epochs} train_loss {epoch_result['train_loss']:.4f} "
                  f"val_loss {val_loss:.4f} val_recall {measured['recall']:.4f} "
                  f"{time.perf_counter() - epoch_start:.0f}s", flush=True)
            if config.patience and stale >= config.patience and epoch_number < config.epochs:
                stopped_early = epoch_number
                measure = (f"validation recall at score {SELECT_SCORE_THRESHOLD:g}" if config.select_by == "recall"
                           else "validation loss")
                print(f"detector stops early after epoch {epoch_number}: {measure} not better for {stale} epochs "
                      f"(best epoch {(best_selection or {}).get('epoch', '-')}, "
                      f"val_loss {best_validation_loss:.4f}, "
                      f"val_recall {(best_selection or {}).get('validation_recall', float('nan')):.4f})", flush=True)
                break

        gpu = None
        if device.type == "cuda":
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
                # The most memory the tensors of this process held (since the start or the resume).
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }

        result = {
            "task": "detector",
            "architecture": "ssdlite320_mobilenet_v3_large",
            "classes": ["background", "speed_sign"],
            "families": config.families,
            "boxes_by_family": plan.get("boxes_by_family"),
            "input_size": list(config.input_size),
            "anchor_min_ratio": config.anchor_min_ratio,
            "augmentation": ({key: list(value) if isinstance(value, tuple) else value
                              for key, value in DETECTOR_AUGMENTATION.items()} if config.augment else None),
            # The validation measures of the epoch whose weights are detector-best.pt (chosen by selection.by).
            "best_validation_loss": best_validation_loss,
            "best_epoch": (best_selection or {}).get("epoch") or next(
                (item["epoch"] for item in history if item["validation_loss"] == best_validation_loss), None),
            "best_validation_recall": (best_selection or {}).get("validation_recall"),
            "best_validation_false_positives_per_image": (best_selection or {}).get(
                "validation_false_positives_per_image"),
            "selection": {"by": config.select_by, "score_threshold": SELECT_SCORE_THRESHOLD,
                          "iou_threshold": SELECT_IOU_THRESHOLD,
                          "ties": "fewer false boxes per frame" if config.select_by == "recall" else None},
            "frames": frame_counts,
            "stopped_early_after_epoch": stopped_early,
            "checkpoint": str(best_path),
            "training_plan": str(plan_path),
            "device": str(device),
            "gpu": gpu,
            "torch_version": str(torch.__version__),
            "torchvision_version": torchvision.__version__,
            "initialization": "imagenet1k_v1" if config.pretrained_backbone else "random",
            "initialization_source": (
                "torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1"
                if config.pretrained_backbone else "random"
            ),
            "config": config.__dict__,
            "history": history,
            "limitations": plan["limitations"],
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tracker.set_summary({
            "best_validation_loss": best_validation_loss,
            "best_validation_recall": (best_selection or {}).get("validation_recall"),
            "checkpoint": str(best_path),
        })
        save_training_state(config.epochs, "completed")
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result


def reader_start(path: Path) -> dict[str, Any]:
    """What the plan of a fresh classifier run records of the reader-best.pt it starts from (nnslr train
    --init-from): its file and sha256, classes, class scheme, input size, split, how it started itself, and whether
    it was the pretraining on Mapillary alone. Reads the checkpoint (torch)."""
    path = Path(path)
    saved = torch_load(path)
    if saved.get("task") != "reader" or saved.get("architecture") != "mobilenet_v3_small":
        raise ValueError(f"the start is not a reader/classifier checkpoint: {path}")
    if not isinstance(saved.get("classes"), list) or not saved["classes"]:
        raise ValueError(f"the start has no classes: {path}")
    return {"path": str(path), "sha256": _sha256(path), "classes": list(saved["classes"]),
            "class_scheme": checkpoint_class_scheme(saved), "input_size": saved.get("input_size"),
            "split_sha256": saved.get("split_sha256"), "dataset_sha256": saved.get("dataset_sha256"),
            "epoch": saved.get("epoch"), "initialization": saved.get("initialization"),
            "mapillary_only": bool(saved.get("mapillary_only", False))}


def train_reader(
    plan: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    config: TrainConfig,
    *,
    mode: str = "fresh",
    hf_repo: str | None = None,
    hf_revision: str = "main",
    init_from: Path | None = None,
) -> dict[str, Any]:
    """Execute the first reader baseline after explicit compute opt-in.

    The plan's class scheme decides what is trained. v1: the reader, its best epoch the highest validation
    accuracy. v2: the classifier with not_a_sign (prepare_reader_training), its log lines "classifier epoch ...",
    its training crops darker and lower in contrast and their boxes jittered (config.augment), with an end-of-limit
    sign drawn over a quarter of its speed signs, labelled cancellation (synthetic_signs, with the augmentation and the
    class cancellation in the plan), the not_a_sign
    crops weighing in the loss as much as all the sign crops together when they are more, and its best epoch
    the highest validation balanced accuracy: the mean of the right answers on the sign crops and the share of
    the not_a_sign crops it rejects, so that neither the signs nor the junk boxes of the cutter decide alone.

    The two stages of a classifier pretrained on Mapillary: a plan of Mapillary alone (mapillary_only) trains the
    "pretrain" network (its log lines "pretrain epoch ..."); a fresh run with config.init_from starts from that
    reader-best.pt instead of ImageNet: every weight copied, the head's rows by class name (a class the start
    lacks starts from zero), and its checkpoint and result say initialization "mapillary_pretraining" (or
    "checkpoint", from any other classifier) and the start.
    """
    _check_reader_config(config)
    scheme = checkpoint_class_scheme(plan)
    if config.class_scheme != scheme:
        raise ValueError(f"the plan has class scheme {scheme}, the training configuration {config.class_scheme}")
    if not plan.get("trainable"):
        raise ValueError(
            "training_plan_not_trainable: need >=2 train classes, non-empty train/validation, "
            "and no validation-only classes" + (" and not_a_sign crops in train" if scheme == "v2" else "")
        )

    torch, _, DataLoader, Dataset, transforms, Weights, mobilenet_v3_small = _require_training_stack()
    import torchvision
    from torchvision.transforms import functional as TF

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    classifier = scheme == "v2"
    # The name of the network in the log lines, the progress of the panel and wandb.
    pretraining = classifier and bool(plan.get("mapillary_only"))
    name = "pretrain" if pretraining else "classifier" if classifier else "reader"
    start = plan.get("start") if classifier else None
    if bool(config.init_from) != bool(start) or (start and mode == "continue"):
        raise ValueError("a start (--init-from) is in both the plan and the configuration of a fresh run, or in neither")
    if start and _sha256(Path(config.init_from)) != start.get("sha256"):
        raise ValueError(f"the start changed since the plan was made: {config.init_from}")
    augment = classifier and config.augment
    classes = list(plan["classes"])
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    # End-of-limit signs drawn over speed signs (synthetic_signs): with the augmentation, when the plan has the class.
    synthetic = augment and synthetic_signs.CANCELLATION in class_to_index
    negative_index = class_to_index.get(NOT_A_SIGN) if classifier else None
    root = Path(data_root).resolve()

    train_records = [r for r in plan["records"] if r["split"] == TRAIN_SPLIT]
    val_records = [r for r in plan["records"] if r["split"] == VALIDATION_SPLIT]
    # Every crop cut from its frame once (nnslr_tools.crop_cache): an epoch reads small arrays, not full frames.
    import numpy as np
    from torchvision.transforms import InterpolationMode

    from nnslr_tools.crop_cache import EXACT, WINDOW, build_crop_cache
    cache, cache_summary = build_crop_cache(
        root, train_records + val_records,
        [WINDOW if augment else EXACT] * len(train_records) + [EXACT] * len(val_records),
        size=config.input_size, scheme=scheme, workers=config.num_workers,
        content={"dataset_sha256": plan["dataset_sha256"], "mapillary_sha256": plan.get("mapillary_sha256")})
    print(f"crop cache: {'cut' if cache_summary['built'] else 'reused'} {cache_summary['crops']} crops of "
          f"{cache_summary['frames']} images, {cache_summary['megabytes']} MB, {cache_summary['seconds']}s: "
          f"{cache_summary['folder']}", flush=True)

    class CropDataset(Dataset):
        """The crops of train (first 0) or validation (first len(train)) from the cache. A key is
        epoch * len(records) + index (EpochSampler): the workers stay up from one epoch to the next, so the epoch
        of the jitter and photometric draws comes with the index."""

        def __init__(self, records, *, first: int, training: bool):
            self.records = records
            self.first = first
            self.training = training
            pipeline = [transforms.Resize((config.input_size, config.input_size))]
            if training and not augment:
                pipeline.extend([
                    transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
                    transforms.RandomAffine(degrees=4, translate=(.03, .03), scale=(.95, 1.05)),
                ])
            self.geometry = transforms.Compose(pipeline)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, key):
            epoch, index = divmod(int(key), len(self.records))
            row = self.records[index]
            box = row["bbox_xyxy"]
            label = row["label"]
            paint = None
            if self.training and synthetic and synthetic_signs.paintable(box, label):
                # A speed sign drawn as an end of limit now and then (synthetic_signs): the crop is a cancellation.
                draw = random.Random(_sample_seed("cancellation", config.seed, epoch, index))
                if draw.random() < synthetic_signs.SHARE:
                    style, value = synthetic_signs.style_of(label), synthetic_signs.value_of(label)

                    def paint(window, to_window):
                        return synthetic_signs.paint_end_of_limit(window, to_window(row["bbox_xyxy"]), style, draw,
                                                                  value)
                    label = synthetic_signs.CANCELLATION
            if self.training and augment:
                box = jitter_box(box, random.Random(_sample_seed("box", config.seed, epoch, index)))
            crop = self.geometry(cache.crop(self.first + index, box, paint))
            if self.training and augment:
                # The colour comes from the photometric augmentation; the box jitter moves and resizes; a rotation
                # of up to 4 degrees as RandomAffine(degrees=4) draws it, from the crop's own seed.
                angle = random.Random(_sample_seed("rotation", config.seed, epoch, index)).uniform(-4.0, 4.0)
                crop = TF.affine(crop, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0],
                                 interpolation=InterpolationMode.NEAREST, fill=0)
            # The pixels as they are (uint8, 3 x H x W) and the photometric draws: the GPU applies them to the batch
            # (photometric_batch) and normalises; a crop without augmentation has the draw that does not apply.
            pixels = torch.from_numpy(np.asarray(crop, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()
            draws = [0.0] * len(PHOTOMETRIC_DRAWS)
            if self.training and augment:
                generator = torch.Generator().manual_seed(_sample_seed("photometric", config.seed, epoch, index))
                draws = photometric_draws(torch, generator, CLASSIFIER_AUGMENTATION)
            return pixels, torch.tensor(draws, dtype=torch.float32), class_to_index[label]

    class EpochSampler(torch.utils.data.Sampler):
        """The train crops of an epoch in the order of seed + epoch, as keys of CropDataset."""

        def __init__(self, count: int):
            self.count = count
            self.epoch = 0

        def __len__(self):
            return self.count

        def __iter__(self):
            order = torch.randperm(self.count, generator=torch.Generator().manual_seed(config.seed + self.epoch))
            return iter([self.epoch * self.count + index for index in order.tolist()])

    normalize_mean = torch.tensor(READER_NORMALIZE_MEAN, device=device).view(1, 3, 1, 1)
    normalize_std = torch.tensor(READER_NORMALIZE_STD, device=device).view(1, 3, 1, 1)

    def batch_inputs(pixels, draws):
        """A batch of uint8 crops on the device: values 0..1 (as ToTensor), the photometric augmentation of each
        crop (photometric_batch) when draws are given, the ImageNet normalisation."""
        images = pixels.to(device, non_blocking=True).float().div_(255.0)
        if draws is not None:
            images = photometric_batch(torch, TF, images, draws)
        return (images - normalize_mean) / normalize_std

    loader_options = {"num_workers": config.num_workers, "pin_memory": device.type == "cuda"}
    if config.num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=4)
    # Each loader seeds its workers from its own generator, not from torch's global one: the dropout draws the same
    # numbers with or without loader workers (persistent workers seed once, a loader without them every epoch).
    loader_seeds = {"train": torch.Generator().manual_seed(config.seed),
                    "validation": torch.Generator().manual_seed(config.seed + 1)}
    train_dataset = CropDataset(train_records, first=0, training=True)
    train_sampler = EpochSampler(len(train_records))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        generator=loader_seeds["train"],
        # A last batch of one crop fails in BatchNorm when the last feature map is 1 x 1 (small --input-size): that
        # crop, the last of the shuffle, waits for another epoch.
        drop_last=len(train_records) > 1 and len(train_records) % config.batch_size == 1,
        **loader_options,
    )
    val_loader = DataLoader(
        CropDataset(val_records, first=len(train_records), training=False),
        batch_size=config.batch_size,
        shuffle=False,
        generator=loader_seeds["validation"],
        **loader_options,
    )

    weights = Weights.IMAGENET1K_V1 if config.pretrained and mode == "fresh" and not start else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    if start and mode == "fresh":
        _load_start(torch, model, torch_load(Path(config.init_from)), classes, config.input_size, name,
                    config.init_from)
    model.to(device)
    if start:
        initialization = "mapillary_pretraining" if start.get("mapillary_only") else "checkpoint"
        initialization_source = str(start.get("path"))
    elif config.pretrained:
        initialization = "imagenet1k_v1"
        initialization_source = "torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    else:
        initialization = initialization_source = "random"
    start_summary = None if not start else {key: value for key, value in start.items() if key != "classes"}

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_weights = None
    if classifier:
        # The not_a_sign crops (removed proposals and the cutter's false boxes) outnumber the signs: together they
        # weigh at most as much as all the sign crops, so the classifier is not pushed to call a sign junk.
        train_negatives = sum(1 for record in train_records if record["label"] == NOT_A_SIGN)
        train_signs = len(train_records) - train_negatives
        negative_weight = min(1.0, train_signs / train_negatives) if train_negatives else 1.0
        loss_weights = {NOT_A_SIGN: negative_weight}
        class_weights = torch.ones(len(classes))
        class_weights[negative_index] = negative_weight
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = torch.nn.CrossEntropyLoss()

    use_amp = bool(config.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "training-state.pt"
    if mode not in ("fresh", "continue", "resume"):
        raise ValueError(f"invalid training mode: {mode}")
    if mode == "resume" and not checkpoint_path.is_file():
        raise ValueError(f"no resumable training checkpoint: {checkpoint_path}")
    plan_path = output_dir / "training-plan.json"

    def reader_samples(inputs, targets, logits) -> list[Any]:
        # The first crops of the first validation batch (no shuffling, so the
        # same crops every time) with the reviewed and the predicted label.
        to_image = transforms.ToPILImage()
        mean = torch.tensor(READER_NORMALIZE_MEAN).view(3, 1, 1)
        std = torch.tensor(READER_NORMALIZE_STD).view(3, 1, 1)
        expected = targets.detach().cpu().tolist()
        predicted = logits.detach().argmax(1).cpu().tolist()
        count = min(tracking.READER_IMAGE_SAMPLES, len(expected))
        samples = []
        for index in range(count):
            pixels = inputs[index].detach().float().cpu() * std + mean
            image = to_image(pixels.clamp(0.0, 1.0))
            caption = f"true {classes[expected[index]]} / pred {classes[predicted[index]]}"
            samples.append(tracker.image(image, caption))
        return samples

    history = []
    # The best value of what chooses the epoch: validation accuracy (v1), validation balanced accuracy (v2).
    best_accuracy = -1.0
    best_selection: dict[str, Any] | None = None
    stale = 0
    best_path = output_dir / "reader-best.pt"
    global_step = 0
    selected_by = "balanced_accuracy" if classifier else "accuracy"

    start_epoch = 0
    if mode == "continue":
        if init_from is not None:
            # The best weights of an earlier run of this server, checked as the stable ones are.
            saved, origin = torch_load(Path(init_from)), str(init_from)
        elif not hf_repo:
            raise ValueError("continue requires --hf-repo or --init-from")
        else:
            saved, origin = stable_weights(hf_repo, "reader", revision=hf_revision)
        if saved.get("task") != "reader" or saved.get("architecture") != "mobilenet_v3_small":
            raise ValueError("stable weight task/architecture mismatch")
        if saved.get("classes") != classes or saved.get("input_size") != config.input_size:
            raise ValueError("stable reader classes/input size mismatch")
        model.load_state_dict(model_weights(saved), strict=True)
        print(f"continue {name} from weights: {origin}")
    elif mode == "resume":
        saved = torch_load(checkpoint_path)
        if saved.get("format_version") != 1 or saved.get("task") != "reader":
            raise ValueError("invalid reader training-state checkpoint")
        if saved.get("status") == "completed":
            raise ValueError("training already completed; use --mode continue for a new run")
        if saved.get("status") != "in_progress":
            raise ValueError("invalid checkpoint status")
        saved_config = {**READER_CONFIG_BEFORE_SETTINGS, **(saved.get("config") or {})}
        if saved_config != config.__dict__:
            raise ValueError("resume requires exactly the original training configuration")
        if saved.get("dataset_sha256") != plan["dataset_sha256"] or saved.get("split_sha256") != plan["split_sha256"]:
            raise ValueError("resume dataset/split mismatch")
        if saved.get("classes") != classes:
            raise ValueError("resume reader classes mismatch")
        if classifier and saved.get("negatives_sha256") != plan.get("negatives_sha256"):
            raise ValueError("resume negatives mismatch: the run started with another negatives file")
        if classifier and saved.get("start_sha256") != (start or {}).get("sha256"):
            raise ValueError("resume start mismatch: the run started from other weights (--init-from)")
        if classifier and saved.get("mapillary_sha256") != plan.get("mapillary_sha256"):
            raise ValueError("resume Mapillary mismatch: the run started with other Mapillary crops")
        if (classifier and "mapillary_selection_sha256" in saved
                and saved["mapillary_selection_sha256"] != plan.get("mapillary_selection_sha256")):
            raise ValueError("resume Mapillary mismatch: the region or the positions now choose other crops")
        start_epoch = saved.get("epoch")
        if not isinstance(start_epoch, int) or not 0 <= start_epoch < config.epochs:
            raise ValueError("invalid or already completed resume epoch")
        model.load_state_dict(model_weights(saved), strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        global_step = saved["global_step"]
        history = saved["history"]
        best_accuracy = saved["best_accuracy"]
        best_selection = saved.get("best_selection")
        stale = int(saved.get("stale", 0))
        print(f"resume {name} at epoch {start_epoch}/{config.epochs}")

    def save_training_state(epoch: int, status: str) -> None:
        atomic_torch_save({
            "format_version": 1,
            "task": "reader",
            "architecture": "mobilenet_v3_small",
            "status": status,
            "epoch": epoch,
            "global_step": global_step,
            "config": config.__dict__,
            "dataset_sha256": plan["dataset_sha256"],
            "split_sha256": plan["split_sha256"],
            "class_scheme": scheme,
            "negatives_sha256": plan.get("negatives_sha256"),
            "mapillary_sha256": plan.get("mapillary_sha256"),
            "mapillary_selection_sha256": plan.get("mapillary_selection_sha256"),
            "start_sha256": (start or {}).get("sha256"),
            "classes": classes,
            "input_size": config.input_size,
            "best_accuracy": best_accuracy,
            "best_selection": best_selection,
            "stale": stale,
            "history": history,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }, checkpoint_path)

    if mode != "resume":
        output_dir.mkdir(parents=True, exist_ok=False)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_training_state(0, "in_progress")

    tracker = tracking.start_training_run(
        name,
        output_dir,
        _tracking_config(plan, config, "mobilenet_v3_small"),
    )
    augmentation = None
    if augment:
        augmentation = {"photometric": {key: list(value) if isinstance(value, tuple) else value
                                        for key, value in CLASSIFIER_AUGMENTATION.items()},
                        "box_jitter": {key: list(value) if isinstance(value, tuple) else value
                                       for key, value in CLASSIFIER_BOX_JITTER.items()},
                        "rotation_degrees": 4,
                        "synthetic_end_of_limit": {**synthetic_signs.summary(), "paintable_train_crops": sum(
                            synthetic_signs.paintable(r["bbox_xyxy"], r["label"]) for r in train_records)}
                        if synthetic else None}
    if classifier:
        counts = Counter(record["label"] == NOT_A_SIGN for record in train_records)
        print(f"classifier crops: {counts[False]} signs and {counts[True]} not_a_sign in train, "
              f"{sum(1 for r in val_records if r['label'] != NOT_A_SIGN)} and "
              f"{sum(1 for r in val_records if r['label'] == NOT_A_SIGN)} in validation; {len(classes)} classes; "
              f"not_a_sign loss weight {loss_weights[NOT_A_SIGN]:.3f}; augmentation {'on' if augment else 'off'}; "
              f"best epoch by validation balanced accuracy (signs read right, not_a_sign rejected)", flush=True)
    if synthetic:
        print(f"classifier draws an end of limit over {synthetic_signs.SHARE:.0%} of the "
              f"{augmentation['synthetic_end_of_limit']['paintable_train_crops']} speed signs of train each epoch "
              f"(cancellation)", flush=True)
    try:
        stopped_early = None
        for epoch in range(start_epoch, config.epochs):
            epoch_number = epoch + 1
            epoch_start = time.perf_counter()
            # The epoch's shuffle and the crops' augmentation, the same after an epoch-boundary resume.
            train_sampler.epoch = epoch
            model.train()
            train_loss = 0.0
            train_seen = 0
            train_correct = 0
            # The step time includes the wait for the batch, like distill's step_time.
            step_start = time.perf_counter()
            batch_count = len(train_loader)
            batch_index = 0
            for pixels, draws, targets in train_loader:
                # Where the training is, batch by batch, for the panel of the review site.
                batch_index += 1
                report_progress(name, epoch * batch_count + batch_index, config.epochs * batch_count,
                                f"epoch {epoch_number} of {config.epochs}, batch {batch_index} of {batch_count}")
                inputs = batch_inputs(pixels, draws if augment else None)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step_loss = float(loss.detach())
                step_seen = targets.numel()
                step_correct = int((logits.argmax(1) == targets).sum())
                train_loss += step_loss * step_seen
                train_seen += step_seen
                train_correct += step_correct
                global_step += 1
                step_end = time.perf_counter()

                if tracker.enabled:
                    tracker.log({
                        "train/loss": step_loss,
                        "train/accuracy": step_correct / max(step_seen, 1),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/step_time": step_end - step_start,
                    }, step=global_step)
                step_start = step_end

            want_images = tracker.should_log_images(epoch_number, config.epochs)
            sample_crops = []
            model.eval()
            val_loss = 0.0
            val_seen = 0
            val_correct = 0
            # v2: the sign crops read right and the not_a_sign crops rejected.
            sign_seen = sign_correct = negative_seen = negative_rejected = 0
            with torch.no_grad():
                for pixels, _, targets in val_loader:
                    inputs = batch_inputs(pixels, None)
                    targets = targets.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(inputs)
                        loss = criterion(logits, targets)
                    predicted = logits.argmax(1)
                    val_loss += float(loss.detach()) * targets.numel()
                    val_seen += targets.numel()
                    val_correct += int((predicted == targets).sum())
                    if classifier:
                        negative = targets == negative_index
                        sign_seen += int((~negative).sum())
                        sign_correct += int(((predicted == targets) & ~negative).sum())
                        negative_seen += int(negative.sum())
                        negative_rejected += int(((predicted == negative_index) & negative).sum())
                    if want_images and len(sample_crops) == 0:
                        sample_crops = reader_samples(inputs, targets, logits)

            epoch_result = {
                "epoch": epoch_number,
                "train_loss": train_loss / max(train_seen, 1),
                "train_accuracy": train_correct / max(train_seen, 1),
                "validation_loss": val_loss / max(val_seen, 1),
                "validation_accuracy": val_correct / max(val_seen, 1),
            }
            score = epoch_result["validation_accuracy"]
            if classifier:
                sign_accuracy = sign_correct / sign_seen if sign_seen else None
                rejection = negative_rejected / negative_seen if negative_seen else None
                parts = [value for value in (sign_accuracy, rejection) if value is not None]
                balanced = sum(parts) / len(parts) if parts else epoch_result["validation_accuracy"]
                epoch_result.update({"validation_sign_accuracy": sign_accuracy, "validation_rejection": rejection,
                                     "validation_balanced_accuracy": balanced})
                score = balanced
            epoch_result["seconds"] = time.perf_counter() - epoch_start
            history.append(epoch_result)

            improved = score > best_accuracy
            if improved:
                best_accuracy = score
                best_selection = {key: epoch_result.get(key) for key in (
                    "epoch", "validation_accuracy", "validation_sign_accuracy", "validation_rejection",
                    "validation_balanced_accuracy") if key in epoch_result}
                saved_best = {
                    "format_version": 1,
                    "architecture": "mobilenet_v3_small",
                    "task": "reader",
                    "class_scheme": scheme,
                    "classes": classes,
                    "input_size": config.input_size,
                    "state_dict": model.state_dict(),
                    "dataset_sha256": plan["dataset_sha256"],
                    "split_sha256": plan["split_sha256"],
                    "epoch": epoch_number,
                    "validation_accuracy": epoch_result["validation_accuracy"],
                    "initialization": initialization,
                    "initialization_source": initialization_source,
                    "torch_version": str(torch.__version__),
                    "torchvision_version": torchvision.__version__,
                }
                if classifier:
                    saved_best.update({
                        "selected_by": selected_by,
                        "validation_sign_accuracy": epoch_result["validation_sign_accuracy"],
                        "validation_rejection": epoch_result["validation_rejection"],
                        "validation_balanced_accuracy": epoch_result["validation_balanced_accuracy"],
                        "folded_labels": dict(plan.get("folded_labels") or {}),
                        "negatives_sha256": plan.get("negatives_sha256"),
                        "mapillary_only": pretraining,
                        "start": start_summary,
                    })
                torch.save(saved_best, best_path)

            if tracker.enabled:
                epoch_metrics = {
                    "epoch": epoch_number,
                    "train/epoch_loss": epoch_result["train_loss"],
                    "train/epoch_accuracy": epoch_result["train_accuracy"],
                    "validation/loss": epoch_result["validation_loss"],
                    "validation/accuracy": epoch_result["validation_accuracy"],
                    "validation/best_accuracy": best_accuracy,
                    "epoch_time": time.perf_counter() - epoch_start,
                }
                if classifier:
                    for key in ("sign_accuracy", "rejection", "balanced_accuracy"):
                        if epoch_result["validation_" + key] is not None:
                            epoch_metrics["validation/" + key] = epoch_result["validation_" + key]
                if len(sample_crops) > 0:
                    epoch_metrics["validation/crops"] = sample_crops
                tracker.log(epoch_metrics, step=global_step)
            stale = 0 if improved else stale + 1
            save_training_state(epoch_number, "in_progress")
            # One line per epoch: the panel of the review site reads the progress from the log (the seconds last).
            extra = ""
            if classifier:
                extra = "".join(f" {label} {epoch_result[key]:.4f}" for label, key in (
                    ("val_sign_accuracy", "validation_sign_accuracy"), ("val_rejection", "validation_rejection"))
                    if epoch_result[key] is not None)
            print(f"{name} epoch {epoch_number}/{config.epochs} train_loss {epoch_result['train_loss']:.4f} "
                  f"val_loss {epoch_result['validation_loss']:.4f} "
                  f"train_accuracy {epoch_result['train_accuracy']:.4f} "
                  f"val_accuracy {epoch_result['validation_accuracy']:.4f}{extra} "
                  f"{time.perf_counter() - epoch_start:.0f}s", flush=True)
            if config.patience and stale >= config.patience and epoch_number < config.epochs:
                stopped_early = epoch_number
                measure = "validation balanced accuracy" if classifier else "validation accuracy"
                print(f"{name} stops early after epoch {epoch_number}: {measure} not better for "
                      f"{stale} epochs (best {best_accuracy:.4f})", flush=True)
                break

        gpu = None
        if device.type == "cuda":
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            }

        best = best_selection or {}
        result = {
            "task": "reader",
            "architecture": "mobilenet_v3_small",
            "class_scheme": scheme,
            "classes": classes,
            # v1: the best validation accuracy, which chose the epoch; v2: the accuracy on every validation crop at
            # the epoch the balanced accuracy chose (best_validation_balanced_accuracy).
            "best_validation_accuracy": best.get("validation_accuracy", best_accuracy),
            "best_epoch": best.get("epoch") or next(
                (item["epoch"] for item in history if item["validation_accuracy"] == best_accuracy), None),
            "selection": {"by": selected_by},
            "stopped_early_after_epoch": stopped_early,
            "checkpoint": str(best_path),
            "training_plan": str(plan_path),
            "device": str(device),
            "gpu": gpu,
            "torch_version": str(torch.__version__),
            "torchvision_version": torchvision.__version__,
            "initialization": initialization,
            "initialization_source": initialization_source,
            "config": config.__dict__,
            # The dataset and split it learnt and chose its epoch on: a later classifier that starts from it
            # (--init-from) is measured fairly only on the same split (the Train tab warns otherwise).
            "dataset_sha256": plan["dataset_sha256"],
            "split_sha256": plan["split_sha256"],
            "crop_cache": {key: cache_summary.get(key) for key in ("folder", "key", "crops", "frames", "megabytes")},
            "history": history,
            "limitations": plan["limitations"],
        }
        if classifier:
            result.update({
                "best_validation_sign_accuracy": best.get("validation_sign_accuracy"),
                "best_validation_rejection": best.get("validation_rejection"),
                "best_validation_balanced_accuracy": best.get("validation_balanced_accuracy"),
                "loss_weights": loss_weights,
                "augmentation": augmentation,
                "mapillary_only": pretraining,
                "start": start_summary,
                "folded_labels": dict(plan.get("folded_labels") or {}),
                "negatives": {"file": (plan.get("negatives_file") or {}).get("file"),
                              "sha256": plan.get("negatives_sha256"), "by_split": plan.get("negatives")},
                "crops": {"train": len(train_records), "validation": len(val_records),
                          "train_not_a_sign": sum(1 for r in train_records if r["label"] == NOT_A_SIGN),
                          "validation_not_a_sign": sum(1 for r in val_records if r["label"] == NOT_A_SIGN)},
            })
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tracker.set_summary({
            "best_validation_accuracy": result["best_validation_accuracy"],
            "checkpoint": str(best_path),
        })
        save_training_state(config.epochs, "completed")
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result
