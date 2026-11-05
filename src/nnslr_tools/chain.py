"""The cutter -> classifier chain (docs/NEURAL_NETWORK.md): the false boxes of the cutter mined as "not a sign"
crops for the classifier (nnslr mine-negatives), the two networks measured end to end (nnslr evaluate-chain), and
the comparison of a classifier with the old reader (scripts/run_classifier.sh).

- mine_negatives runs the cutter on every reviewed frame of train, validation and test. A box with a score of at
  least score_threshold that overlaps no box a reviewer saw in that frame (IoU < NEGATIVE_MAX_IOU with every kept
  box and every removed proposal) is a candidate; each frame gives its per_frame best, and each partition at most
  total_factor x its sign crops, drawn with a seed from the frames a person reviewed first, then those an AI
  reviewed after the other-sign rule (training.OTHER_SIGN_RULE_AT), then the rest: before that rule an AI left the
  other traffic signs without a box, so a "false" box there may be a sign. In those frames a box of at least
  sign_score is left out too: the cutter's surest "false" boxes there are mostly signs nobody boxed. Every partition takes its negatives
  from its own frames only. The file starts with a header (dataset, split, cutter, settings, counts), so a
  training reads it only with the dataset and split it was mined on.
- evaluate_chain runs the cutter on every reviewed frame of a partition, cuts each box at full resolution from
  the frame, classifies it, and counts at each cutter score threshold: the speed signs found AND read right (the
  end-to-end recall, overall, by width and on the signs a person confirmed), the false boxes per frame that
  survive (answered anything but not_a_sign), and the confusion of every box passed on. Each frame keeps its boxes
  with the classifier's answers (evaluation.per_frame_record), for the Watch panel of the Evaluate tab.
- A cutter or classifier is a checkpoint path or "active": the model in use of its role, from
  <data root>/models/active.json (written by the Models tab of the site).
- The cutter is SignNet, the network the car runs (nnslr_tools.signnet; "cutter:<run>/signnet", its weights
  <run>/signnet-best.pt), or an old Detector (SSDlite, ".../detector-best.pt") kept to compare: its kind comes from the
  checkpoint's task, never from the id. A SignNet is read as the car reads it: the frame's top band
  (signnet._band_of), the heat map's best CAR_MAX_BOXES of at least the score asked, scaled to frame pixels (frame
  width / band width), with no MAX_SIGN_SIDE filter (the car has none). The car keeps those of CAR_MIN_SCORE (0.3)
  or more: mining and Prepare's Our networks go down to 0.1, so they also take boxes the car drops; only
  evaluate-chain at its main score 0.3 is exactly the car. A Detector keeps its own resize and boxes, as before.

Torch is imported only by mine_negatives and evaluate_chain; the dry runs, the reports and the comparison are
stdlib-only.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Iterator

from nnslr_tools.media import full_picture
from nnslr_tools.progress import report as report_progress
from nnslr_tools.signnet import CAR_MAX_BOXES, CAR_MIN_SCORE
from nnslr_tools.training import (BOXES_OTHER_SIGNS, DEFAULT_WORKERS, DETECTOR_FAMILIES, NEGATIVE_MAX_IOU, NEGATIVE_PARTITIONS, NEGATIVES_KIND,
                                  NOT_A_SIGN, OTHER_SIGN_RULE_AT, READER_NORMALIZE_MEAN, READER_NORMALIZE_STD,
                                  REVIEW_TIERS, _label_basis, _safe_image_path, box_overlap, checkpoint_class_scheme,
                                  classifier_positives, crop_rectangle, crop_source, cutter_families, family_label, frame_boxes,
                                  load_reviewed_dataset, negative_id, _prepare_classifier, reviewed_frames,
                                  scheme_label)

# The thresholds of an old Detector's score at which the chain is measured; the first measure of the site is at 0.1.
CHAIN_SCORE_THRESHOLDS = (0.05, 0.1, 0.25)
CHAIN_MAIN_SCORE = 0.1
# A SignNet's (heat peaks, its best 5 boxes a frame, as the car): its main score is the car's floor, so at the main
# score the chain reads exactly the car's boxes.
SIGNNET_CHAIN_SCORE_THRESHOLDS = (0.1, 0.2, CAR_MIN_SCORE)
SIGNNET_CHAIN_MAIN_SCORE = CAR_MIN_SCORE
# A box of the cutter finds a reviewed box when their IoU is at least this (as nnslr evaluate).
CHAIN_IOU_THRESHOLD = 0.5
# Mining: a box of at least this score, at most PER_FRAME of them per frame, at most TOTAL_FACTOR x the sign crops
# of the partition.
MINE_SCORE_THRESHOLD = 0.1
MINE_PER_FRAME = 3
MINE_TOTAL_FACTOR = 3.0
# In a frame whose reviewer did not box the other signs, a box of at least this score is not mined: on 24 October
# 2026 about 15 of the 40 best-scored such boxes of train were real signs (no entry, no overtaking, roadworks).
MINE_SIGN_SCORE = 0.3
# The order in which the frames give their negatives when a partition has more than it takes.
TIER_PREFERENCE = ("human", "ai_after_rule", "ai_before_rule", "other")
# A model in use (models/active.json): "<role>:<run>/<part>"; the weights file of each role inside that part.
MODEL_REFERENCE = re.compile(r'(cutter|classifier|decider):([A-Za-z0-9][A-Za-z0-9._-]{0,99})/'
                             r'(detector|detector-sizes/[1-9][0-9]{0,4}x[1-9][0-9]{0,4}|reader|classifier'
                             r'|experiment-b/(crop|crop_geom|full))')
# The decider's weights are named after its variant: experiment-b/full/full-best.pt.
WEIGHTS_FILE = {"cutter": "detector-best.pt", "classifier": "reader-best.pt", "decider": "{variant}-best.pt"}
# SignNet, the cutter: "cutter:<run>/signnet", its weights at the root of its run (model_roles.SIGNNET_ID).
SIGNNET_REFERENCE = re.compile(r'cutter:([A-Za-z0-9][A-Za-z0-9._-]{0,99})/signnet')
SIGNNET_FILE = "signnet-best.pt"
# No sign is boxed bigger than this share of the frame height on its longest side: in the reviews the biggest sign
# boxes are about 240 px and a road marking 312 px wide, on 760 px; the cutter's boxes over half the sky or the road
# are junk the classifier is not trained to reject.
MAX_SIGN_SIDE = .5
# Frames per batch of the cutter, crops per batch of the classifier.
CUTTER_BATCH = 8
CLASSIFIER_BATCH = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weights_of(model_id: str, role: str) -> str | None:
    """The weights of a model id, relative to <data root>/runs; None when the id is not one of the role."""
    if role == "cutter" and (signnet := SIGNNET_REFERENCE.fullmatch(model_id)):
        return f"{signnet.group(1)}/{SIGNNET_FILE}"
    match = MODEL_REFERENCE.fullmatch(model_id)
    if match is None or match.group(1) != role:
        return None
    return f"{match.group(2)}/{match.group(3)}/{WEIGHTS_FILE[role].format(variant=match.group(4))}"


def _id_of(path: Path, role: str, runs: Path) -> str | None:
    """The model id of a checkpoint path inside <data root>/runs, None for any other file: a script that passes the
    weights by path still gets its headers and reports named after the model."""
    try:
        parts = path.resolve().relative_to(runs).parts
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    candidate = (f"{role}:{parts[0]}/signnet" if role == "cutter" and parts[1:] == (SIGNNET_FILE,)
                 else f"{role}:{parts[0]}/{'/'.join(parts[1:-1])}")
    weights = _weights_of(candidate, role)
    return candidate if weights is not None and tuple(weights.split("/")) == parts else None


def resolve_model(spec: str, role: str, data_root: Path) -> dict[str, Any]:
    """{"path", "id"} of a cutter, classifier or decider given as a checkpoint path, as a model id of the role
    ("classifier:<run>/classifier": the version of the Models tab) or as "active" (the model in use of the role in
    <data root>/models/active.json). The weights of an id must be inside <data root>/runs. A path inside
    <data root>/runs gets the id of its model; any other path None."""
    if role not in WEIGHTS_FILE:
        raise ValueError(f"unknown role {role!r}")
    runs = (Path(data_root) / "runs").resolve()
    if spec != "active" and _weights_of(spec, role) is not None:
        path = (runs / _weights_of(spec, role)).resolve()
        if not path.is_relative_to(runs) or not path.is_file():
            raise ValueError(f"{role}_checkpoint_missing: {spec} has no runs/{_weights_of(spec, role)} on this server")
        return {"path": path, "id": spec}
    if spec != "active":
        path = Path(spec)
        if not path.is_file():
            raise ValueError(f"{role}_checkpoint_missing: {spec}")
        return {"path": path, "id": _id_of(path, role, runs)}
    root = Path(data_root)
    try:
        chosen = json.loads((root / "models" / "active.json").read_text(encoding="utf-8")).get(role)
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(f"no_{role}_in_use: models/active.json cannot be read ({exc})") from exc
    model_id = chosen.get("id") if isinstance(chosen, dict) else None
    weights = _weights_of(model_id, role) if isinstance(model_id, str) else None
    if weights is None:
        raise ValueError(f"no_{role}_in_use: choose one on the Models tab of the site (models/active.json)")
    path = (runs / weights).resolve()
    if not path.is_relative_to(runs) or not path.is_file():
        raise ValueError(f"{role}_in_use_missing: {model_id} has no runs/{weights} on this server")
    return {"path": path, "id": model_id}


def _model_summary(checkpoint: dict[str, Any], resolved: dict[str, Any], split_sha256: str) -> dict[str, Any]:
    return {"checkpoint": str(resolved["path"]), "id": resolved["id"], "sha256": _sha256(resolved["path"]),
            "dataset_sha256": checkpoint.get("dataset_sha256"), "split_sha256": checkpoint.get("split_sha256"),
            "same_split": checkpoint.get("split_sha256") == split_sha256, "epoch": checkpoint.get("epoch")}


# -- the cutter -------------------------------------------------------------------------------------------
class SignNetCutter:
    """A SignNet as the cutter (cutter_boxes tells it from a Detector by its kind): the network, its stride and its
    band [width, height]."""
    kind = "signnet"

    def __init__(self, model, stride: int, band: list[int]):
        self.model, self.stride, self.band = model, int(stride), [int(band[0]), int(band[1])]

    def __call__(self, bands):
        return self.model(bands)


def cutter_kind(cutter) -> str:
    """"signnet" or "detector" for a cutter of _load_cutter."""
    return getattr(cutter, "kind", "detector")


def _load_cutter(resolved: dict[str, Any], device):
    """(the cutter, its checkpoint, its input size [width, height]) by the checkpoint's task: a SignNet
    (SignNetCutter; its input is its band, its checkpoint's data from result.json when it lacks them) or a
    Detector (the torchvision model, as before)."""
    from nnslr_tools import checkpoints, signnet

    _require()
    checkpoint = checkpoints.safe_torch_load(Path(resolved["path"]))
    task = checkpoint.get("task") if isinstance(checkpoint, dict) else None
    if task == "signnet":
        model = signnet.model_of(checkpoint)
        band = [int(value) for value in checkpoint.get("band") or ()]
        if len(band) != 2 or list(signnet.band_size(band[0])) != band:
            raise ValueError(f"signnet_band_invalid: {checkpoint.get('band')!r}")
        model.to(device)
        checkpoint = signnet.checkpoint_data(checkpoint, Path(resolved["path"]))
        return SignNetCutter(model, checkpoint["stride"], band), checkpoint, band
    if task != "detector":
        raise ValueError("checkpoint_task_mismatch: expected signnet or detector")
    if not checkpoint.get("state_dict") or not isinstance(checkpoint["state_dict"], dict):
        raise ValueError("checkpoint_missing_state_dict")
    model = checkpoints.load_detector(checkpoint)
    model.to(device)
    model.eval()
    return model, checkpoint, list(checkpoints.detector_input_size(checkpoint))


def cutter_summary(cutter, checkpoint: dict[str, Any], resolved: dict[str, Any], split_sha256: str,
                   input_size: list[int]) -> dict[str, Any]:
    """The cutter block of the negatives header and the chain report: the model, its data and split, its kind, its
    input (a SignNet's: its band, and its stride), its families."""
    info = {**_model_summary(checkpoint, resolved, split_sha256), "kind": cutter_kind(cutter),
            "input_size": list(input_size), "families": cutter_families(checkpoint)}
    if info["kind"] == "signnet":
        info.update(band=list(cutter.band), stride=cutter.stride, max_boxes=CAR_MAX_BOXES)
    return info


def _require():
    from nnslr_tools.training import _require_detector_stack
    return _require_detector_stack()


def _device(torch, name: str):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(name)
    # On the GPU the networks need no CPU threads, and cutting the boxes out of a frame is hundreds of tiny tensor
    # operations on the CPU: with PyTorch's threads (one per core) they spend their time waking each other, 19 s for
    # the 200 boxes of a frame on the server's 32 cores against 0.17 s on one thread.
    if device.type == "cuda":
        torch.set_num_threads(1)
    return device


def _signnet_boxes(cutter: SignNetCutter, heat, shape, width: int, height: int, min_score: float,
                   limit: int) -> list[dict[str, Any]]:
    """The boxes of one frame from SignNet's output as the car decodes them (signnet.frame_boxes_of, at most
    ``limit`` of at least ``min_score``: the car's own at CAR_MIN_SCORE), clamped to the frame."""
    from nnslr_tools.signnet import frame_boxes_of

    found = frame_boxes_of(heat, shape, cutter.stride, cutter.band[0], width, limit=limit, min_score=min_score)
    for item in found:
        x1, y1, x2, y2 = item["box"]
        item["box"] = [min(max(x1, 0.0), width), min(max(y1, 0.0), height),
                       min(max(x2, 0.0), width), min(max(y2, 0.0), height)]
    return found


def crop_cutter(data_root: Path, frame: dict[str, Any], pixels):
    """What turns a rectangle of a frame (pixels, its tensor) into its PIL crop: cut from the frame at full definition
    when its drive has one (training.crop_source), the only case in which a file is read here."""
    import torchvision.transforms.functional as TF

    path = _safe_image_path(data_root, frame["image_path"])
    if not full_picture(path).is_file():
        return lambda rectangle: TF.to_pil_image(pixels[:, rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]])
    image, factor = crop_source(path)
    return lambda rectangle: image.crop(tuple(value * factor for value in rectangle))


def cutter_boxes(model, device, frames: list[dict[str, Any]], data_root: Path, *, min_score: float,
                 batch_size: int = CUTTER_BATCH, num_workers: int = DEFAULT_WORKERS, task: str = "chain",
                 limit: int = CAR_MAX_BOXES) -> Iterator[tuple[dict[str, Any], Any, list[dict[str, Any]]]]:
    """(frame, its pixels as a 3 x H x W uint8 tensor, the cutter's boxes of score >= min_score in frame pixels,
    best first) for every frame, in order. The frames are decoded by loader workers. A SignNet (see the module)
    sees the band made from the same decoded frame, values / 255, and gives its best ``limit`` boxes (the car's
    CAR_MAX_BOXES; the car's own boxes only when min_score is CAR_MIN_SCORE) clamped to the frame. A Detector sees the frame as nnslr evaluate does (values 0..1, its own
    resize inside the model); its boxes bigger than MAX_SIGN_SIDE are dropped."""
    torch, Image, DataLoader, Dataset, TF, _, _ = _require()
    root = Path(data_root).resolve()
    from nnslr_tools.signnet import _band_of

    signnet = model if cutter_kind(model) == "signnet" else None

    class Frames(Dataset):
        def __len__(self):
            return len(frames)

        def __getitem__(self, index):
            with Image.open(_safe_image_path(root, frames[index]["image_path"])) as source:
                image = source.convert("RGB")
                band = _band_of(image, signnet.band[0]) if signnet is not None else None
                return index, TF.pil_to_tensor(image), band

    loader = DataLoader(Frames(), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=lambda batch: batch)
    done = 0
    with torch.no_grad():
        for batch in loader:
            if signnet is not None:
                heat, shape = signnet(torch.stack([band for _, _, band in batch]).to(device).float().div_(255.0))
                heat, shape = heat.float().cpu(), shape.float().cpu()
                found = [_signnet_boxes(signnet, heat[b:b + 1], shape[b:b + 1], int(pixels.shape[2]),
                                        int(pixels.shape[1]), min_score, limit)
                         for b, (_, pixels, _) in enumerate(batch)]
            else:
                outputs = model([pixels.to(device).float().div_(255.0) for _, pixels, _ in batch])
                found = []
                for (_, pixels, _), output in zip(batch, outputs):
                    keep = (output["labels"] == 1) & (output["scores"] >= min_score)
                    boxes = output["boxes"][keep].float().cpu().tolist()
                    scores = output["scores"][keep].float().cpu().tolist()
                    largest = MAX_SIGN_SIDE * int(pixels.shape[1])
                    found.append(sorted(({"box": box, "score": score} for box, score in zip(boxes, scores)
                                         if max(box[2] - box[0], box[3] - box[1]) <= largest),
                                        key=lambda item: -item["score"]))
            for (index, pixels, _), predictions in zip(batch, found):
                done += 1
                report_progress(task, done, len(frames), f"{done} of {len(frames)} frames")
                yield frames[index], pixels, predictions


# -- mining the negatives -----------------------------------------------------------------------------------
def frame_candidates(frame: dict[str, Any], predictions: list[dict[str, Any]], *, per_frame: int,
                     sign_score: float | None = None) -> tuple[int, list]:
    """(how many of the predictions overlap no box a reviewer saw, the best per_frame of them). With sign_score,
    a frame whose reviewer did not box the other signs (not BOXES_OTHER_SIGNS) loses its boxes of at least that
    score first: likely signs, not junk."""
    seen = frame_boxes(frame)
    free = [item for item in predictions if all(box_overlap(item["box"], box) < NEGATIVE_MAX_IOU for box in seen)]
    if sign_score is not None and frame.get("tier") not in BOXES_OTHER_SIGNS:
        free = [item for item in free if item["score"] < sign_score]
    return len(free), free[:per_frame]


def select_negatives(candidates: list[dict[str, Any]], cap: int, seed: Any) -> list[dict[str, Any]]:
    """At most cap candidates, the frames of TIER_PREFERENCE first; inside a tier a seeded random draw."""
    chosen: list[dict[str, Any]] = []
    for tier in TIER_PREFERENCE:
        pool = [item for item in candidates if item["tier"] == tier]
        random.Random(f"{seed} {tier}").shuffle(pool)
        chosen += pool[:max(0, cap - len(chosen))]
    return chosen


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".tmp", delete=False)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _tier_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item["tier"] for item in items)
    return {tier: counts[tier] for tier in REVIEW_TIERS}


def negatives_dry_run(dataset_path: Path, split_path: Path, data_root: Path, *, dataset_kind: str = "gold",
                      per_frame: int = MINE_PER_FRAME, total_factor: float = MINE_TOTAL_FACTOR,
                      partitions: tuple[str, ...] = NEGATIVE_PARTITIONS) -> dict[str, Any]:
    """What mining would work on, without torch: per partition the frames by who reviewed them, the sign crops
    and the cap of mined boxes, and the not_a_sign crops the dataset gives by itself (removed proposals, boxes
    marked not a sign) with who reviewed them."""
    loaded = load_reviewed_dataset(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
    plan = _prepare_classifier(loaded, Path(dataset_path), Path(split_path), Path(data_root),
                               dataset_kind=dataset_kind, negatives_path=None)
    frames = [frame for frame in reviewed_frames(loaded["rows"], loaded["assignments"]) if frame["split"] in partitions]
    positives = classifier_positives(loaded["rows"], loaded["assignments"])
    counts = {}
    for partition in partitions:
        mine = [frame for frame in frames if frame["split"] == partition]
        cap = int(round(total_factor * positives[partition]))
        counts[partition] = {"frames": len(mine), "frames_by_tier": _tier_counts(mine),
                             "positives": positives[partition], "cap": cap,
                             "most_mined": min(cap, per_frame * len(mine)),
                             "dataset_negatives": (plan["negatives"] or {}).get(partition),
                             "dataset_negatives_by_label_basis": (plan["negatives_by_label_basis"] or {}).get(partition),
                             "dataset_negatives_by_tier": (plan["negatives_by_tier"] or {}).get(partition)}
    return {"task": NEGATIVES_KIND, "dry_run": True, "dataset_sha256": loaded["dataset_sha256"],
            "split_sha256": loaded["split_sha256"], "classes": plan["classes"],
            "train_counts_by_class": plan["train_counts_by_class"], "positives_by_split": plan["positives_by_split"],
            "folded_labels": plan["folded_labels"], "skipped": plan["skipped"],
            "settings": {"per_frame": per_frame, "total_factor": total_factor, "max_iou": NEGATIVE_MAX_IOU,
                         "rule_change_at": OTHER_SIGN_RULE_AT},
            "counts": counts, "gpu_required_for_mining": True}


def mine_negatives(dataset_path: Path, split_path: Path, data_root: Path, cutter: str, output: Path, *,
                   dataset_kind: str = "gold", score_threshold: float = MINE_SCORE_THRESHOLD,
                   per_frame: int = MINE_PER_FRAME, total_factor: float = MINE_TOTAL_FACTOR, seed: int = 0,
                   sign_score: float | None = MINE_SIGN_SCORE,
                   partitions: tuple[str, ...] = NEGATIVE_PARTITIONS, device_name: str = "cuda",
                   batch_size: int = CUTTER_BATCH, num_workers: int = DEFAULT_WORKERS) -> dict[str, Any]:
    """The cutter's false boxes of every partition (see the module), written to output (JSONL, header first).
    Returns the header."""
    if not 0 <= score_threshold <= 1:
        raise ValueError("invalid_score_threshold")
    if per_frame < 1 or total_factor <= 0:
        raise ValueError("invalid_caps: per_frame >= 1 and total_factor > 0")
    if sign_score is not None and not score_threshold < sign_score <= 1:
        raise ValueError("invalid_sign_score: above score_threshold and at most 1")
    if any(partition not in NEGATIVE_PARTITIONS for partition in partitions):
        raise ValueError(f"negatives come from {', '.join(NEGATIVE_PARTITIONS)} only")
    output = Path(output)
    if output.exists():
        raise ValueError(f"output_exists: {output} (mined negatives are not overwritten)")
    loaded = load_reviewed_dataset(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
    frames = [frame for frame in reviewed_frames(loaded["rows"], loaded["assignments"]) if frame["split"] in partitions]
    positives = classifier_positives(loaded["rows"], loaded["assignments"])
    resolved = resolve_model(cutter, "cutter", data_root)

    torch, *_ = _require()
    device = _device(torch, device_name)
    model, checkpoint, input_size = _load_cutter(resolved, device)
    started = time.perf_counter()
    candidates: dict[str, list[dict[str, Any]]] = {partition: [] for partition in partitions}
    counts = {partition: {"frames": 0, "frames_with_candidates": 0, "candidates": 0, "after_frame_cap": 0,
                          "likely_unboxed_signs": 0} for partition in partitions}
    for frame, _, predictions in cutter_boxes(model, device, frames, data_root, min_score=score_threshold,
                                              batch_size=batch_size, num_workers=num_workers, task="mine"):
        count = counts[frame["split"]]
        free, best = frame_candidates(frame, predictions, per_frame=per_frame, sign_score=sign_score)
        count["likely_unboxed_signs"] += frame_candidates(frame, predictions, per_frame=per_frame)[0] - free
        count["frames"] += 1
        count["candidates"] += free
        count["after_frame_cap"] += len(best)
        count["frames_with_candidates"] += bool(free)
        for item in best:
            candidates[frame["split"]].append({
                "kind": "negative", "source": "mined", "split": frame["split"], "image_path": frame["image_path"],
                "image_sha256": frame["image_sha256"], "route_id": frame["route_id"],
                "segment_index": frame["segment_index"], "output_index": frame["output_index"],
                "bbox_xyxy": [round(float(value), 2) for value in item["box"]], "score": round(float(item["score"]), 5),
                "label_basis": frame["label_basis"], "reviewed_at": frame["reviewed_at"], "tier": frame["tier"],
                "negative_id": negative_id(frame["image_sha256"], item["box"])})
    rows = []
    for partition in partitions:
        cap = int(round(total_factor * positives[partition]))
        chosen = select_negatives(candidates[partition], cap, f"{seed} {partition}")
        counts[partition].update({"positives": positives[partition], "cap": cap, "selected": len(chosen),
                                  "candidates_by_tier": _tier_counts(candidates[partition]),
                                  "selected_by_tier": _tier_counts(chosen)})
        rows += sorted(chosen, key=lambda item: (item["image_path"], -item["score"]))
    header = {
        "kind": "header", "schema_version": 1, "task": NEGATIVES_KIND, "dataset_kind": dataset_kind,
        "dataset_sha256": loaded["dataset_sha256"], "split_sha256": loaded["split_sha256"],
        "split_path": str(loaded["split_path"]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutter": cutter_summary(model, checkpoint, resolved, loaded["split_sha256"], input_size),
        "settings": {"score_threshold": score_threshold, "max_iou": NEGATIVE_MAX_IOU, "per_frame": per_frame,
                     "total_factor": total_factor, "seed": seed, "rule_change_at": OTHER_SIGN_RULE_AT,
                     "sign_score": sign_score, "boxes_other_signs": list(BOXES_OTHER_SIGNS),
                     "tier_preference": list(TIER_PREFERENCE), "partitions": list(partitions)},
        "counts": counts, "negative_count": len(rows), "device": str(device),
        "seconds": round(time.perf_counter() - started, 1),
    }
    _write_jsonl_atomic(output, [header, *rows])
    return header


# -- the chain ---------------------------------------------------------------------------------------------
def _load_classifier(resolved: dict[str, Any], device):
    from nnslr_tools.evaluation import _load_checkpoint
    from nnslr_tools.training import _require_training_stack

    torch, _, _, _, transforms, _, mobilenet_v3_small = _require_training_stack()
    checkpoint = _load_checkpoint(torch, resolved["path"], "reader")
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("checkpoint_invalid_classes")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    size = int(checkpoint.get("input_size", 160))
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                    transforms.Normalize(mean=READER_NORMALIZE_MEAN, std=READER_NORMALIZE_STD)])
    return model, checkpoint, classes, transform


def truth_objects(frame: dict[str, Any], class_scheme: str, classes: list[str],
                  folded: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The reviewed boxes of a frame the chain must find and read: every kept box with a label of the
    classifier's scheme, except those marked not a sign. A kind the classifier does not know is its family
    (folded_labels of the checkpoint, else other_sign)."""
    found = []
    for row in frame["kept"]:
        label = scheme_label(row, class_scheme)
        if label is None or label == NOT_A_SIGN:
            continue
        if label not in classes:
            label = (folded or {}).get(label) or (family_label(label) if family_label(label) in classes else label)
        found.append({"box": [float(value) for value in row["bbox_xyxy"]], "label": label,
                      "family": row.get("sign_family"), "speed_sign": row.get("sign_family") in DETECTOR_FAMILIES,
                      "label_basis": _label_basis(row)})
    return found


def match_boxes(selected: list[dict[str, Any]], truth: list[dict[str, Any]], iou_threshold: float) -> dict[int, int]:
    """{selected index: truth index}: the greedy score-ordered one-to-one matching of nnslr evaluate."""
    unmatched = set(range(len(truth)))
    pairs: dict[int, int] = {}
    for position in sorted(range(len(selected)), key=lambda index: -selected[index]["score"]):
        best_iou, best = max(((box_overlap(selected[position]["box"], truth[index]["box"]), index)
                              for index in unmatched), default=(0.0, -1))
        if best >= 0 and best_iou >= iou_threshold:
            unmatched.discard(best)
            pairs[position] = best
    return pairs


def _found_counts() -> dict[str, int]:
    return {"truth": 0, "found": 0, "correct": 0, "rejected": 0}


def _finish(counts: dict[str, int]) -> dict[str, Any]:
    truth, found, correct = counts["truth"], counts["found"], counts["correct"]
    return {**counts, "recall": correct / truth if truth else None, "recall_found": found / truth if truth else None,
            "read_right_when_found": correct / found if found else None}


class ChainTally:
    """The counts of the chain at one score threshold, frame after frame (see chain_metrics)."""

    def __init__(self):
        from nnslr_tools.evaluation import WIDTH_BUCKETS
        self.buckets = [key for key, _, _ in WIDTH_BUCKETS]
        self.speed = _found_counts()
        self.speed_human = _found_counts()
        self.widths = {key: _found_counts() for key in self.buckets}
        self.every = _found_counts()
        self.every_human = _found_counts()
        self.images = self.human_images = 0
        self.boxes = self.false_boxes = self.surviving = self.false_rejected = 0
        self.human_false = self.human_surviving = 0
        self.survivor_labels: Counter = Counter()
        self.confusion: dict[str, Counter] = {}

    def add(self, frame_basis: str | None, truth: list[dict[str, Any]], selected: list[dict[str, Any]],
            iou_threshold: float) -> dict[int, int]:
        """One frame: its truth objects and the boxes passed on, each with its "predicted" label."""
        from nnslr_tools.evaluation import HUMAN_LABEL_BASIS, width_bucket

        pairs = match_boxes(selected, truth, iou_threshold)
        by_truth = {index: position for position, index in pairs.items()}
        human_frame = frame_basis == HUMAN_LABEL_BASIS
        self.images += 1
        self.human_images += human_frame
        self.boxes += len(selected)
        for index, item in enumerate(truth):
            position = by_truth.get(index)
            found = position is not None
            predicted = selected[position]["predicted"] if found else None
            groups = [self.every]
            if item["speed_sign"]:
                groups += [self.speed, self.widths[width_bucket(item["box"][2] - item["box"][0])]]
            if item["label_basis"] == HUMAN_LABEL_BASIS:
                groups.append(self.every_human)
                if item["speed_sign"]:
                    groups.append(self.speed_human)
            for group in groups:
                group["truth"] += 1
                group["found"] += found
                group["correct"] += found and predicted == item["label"]
                group["rejected"] += found and predicted == NOT_A_SIGN
        for position, item in enumerate(selected):
            truth_label = truth[pairs[position]]["label"] if position in pairs else NOT_A_SIGN
            self.confusion.setdefault(truth_label, Counter())[item["predicted"]] += 1
            if position in pairs:
                continue
            survives = item["predicted"] != NOT_A_SIGN
            self.false_boxes += 1
            self.surviving += survives
            self.false_rejected += not survives
            if survives:
                self.survivor_labels[item["predicted"]] += 1
            if human_frame:
                self.human_false += 1
                self.human_surviving += survives
        return pairs

    def metrics(self) -> dict[str, Any]:
        images = self.images
        speed = _finish(self.speed)
        speed["by_width"] = {key: {name: value for name, value in _finish(self.widths[key]).items()
                                   if name != "read_right_when_found"} for key in self.buckets}
        speed["human_reviewed"] = _finish(self.speed_human) if self.speed_human["truth"] else None
        every = _finish(self.every)
        every["human_reviewed"] = _finish(self.every_human) if self.every_human["truth"] else None
        return {
            "speed_signs": speed,
            "all_boxes": every,
            "boxes_per_frame": self.boxes / images if images else None,
            "false_boxes": self.false_boxes,
            "false_per_frame": self.false_boxes / images if images else None,
            "surviving_false_boxes": self.surviving,
            "surviving_false_per_frame": self.surviving / images if images else None,
            "false_rejected_rate": self.false_rejected / self.false_boxes if self.false_boxes else None,
            "human": None if not self.human_images else {
                "images": self.human_images, "false_per_frame": self.human_false / self.human_images,
                "surviving_false_per_frame": self.human_surviving / self.human_images},
            "surviving_false_by_label": dict(self.survivor_labels.most_common()),
            "confusion": {truth: dict(sorted(answers.items())) for truth, answers in sorted(self.confusion.items())},
        }


def chain_dry_run(dataset_path: Path, split_path: Path, data_root: Path, *, dataset_kind: str = "gold",
                  partition: str = "test") -> dict[str, Any]:
    """The frames of a partition and the reviewed boxes the chain must find, without torch."""
    loaded = load_reviewed_dataset(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
    frames = [frame for frame in reviewed_frames(loaded["rows"], loaded["assignments"]) if frame["split"] == partition]
    truth = [item for frame in frames for item in truth_objects(frame, "v2", [])]
    return {"task": "chain", "dry_run": True, "partition": partition, "dataset_sha256": loaded["dataset_sha256"],
            "split_sha256": loaded["split_sha256"], "image_count": len(frames),
            "human_image_count": sum(1 for frame in frames if frame["label_basis"] == "human_review"),
            "frames_by_tier": _tier_counts(frames),
            "speed_sign_count": sum(1 for item in truth if item["speed_sign"]),
            "human_speed_sign_count": sum(1 for item in truth if item["speed_sign"] and item["label_basis"] == "human_review"),
            "reviewed_box_count": len(truth), "gpu_required": True}


def chain_scores(kind: str) -> tuple[tuple[float, ...], float]:
    """(the score thresholds, the main one) the chain is measured at by default for a cutter of this kind."""
    if kind == "signnet":
        return SIGNNET_CHAIN_SCORE_THRESHOLDS, SIGNNET_CHAIN_MAIN_SCORE
    return CHAIN_SCORE_THRESHOLDS, CHAIN_MAIN_SCORE


def evaluate_chain(dataset_path: Path, split_path: Path, data_root: Path, cutter: str, classifier: str, *,
                   dataset_kind: str = "gold", partition: str = "test",
                   score_thresholds: tuple[float, ...] | None = None,
                   main_score_threshold: float | None = None, iou_threshold: float = CHAIN_IOU_THRESHOLD,
                   device_name: str = "cuda", batch_size: int = CUTTER_BATCH, num_workers: int = DEFAULT_WORKERS) -> dict[str, Any]:
    """The cutter and the classifier on every reviewed frame of a partition (see the module). The score thresholds
    and the main one default by the cutter's kind (chain_scores): a SignNet's main score is the car's floor."""
    from nnslr_tools.evaluation import per_frame_descriptor, per_frame_record

    given = [*(score_thresholds or ()), *(() if main_score_threshold is None else (main_score_threshold,))]
    if not all(0 <= float(value) <= 1 for value in given):
        raise ValueError("invalid_score_threshold")
    if not 0 < iou_threshold <= 1:
        raise ValueError("invalid_iou_threshold")
    loaded = load_reviewed_dataset(dataset_path, split_path, data_root, dataset_kind=dataset_kind)
    frames = [frame for frame in reviewed_frames(loaded["rows"], loaded["assignments"]) if frame["split"] == partition]
    if not frames:
        raise ValueError(f"empty_evaluation_partition: {partition}")
    cutter_ref = resolve_model(cutter, "cutter", data_root)
    classifier_ref = resolve_model(classifier, "classifier", data_root)

    torch, _, _, _, TF, _, _ = _require()
    device = _device(torch, device_name)
    started = time.perf_counter()
    cutter_model, cutter_checkpoint, input_size = _load_cutter(cutter_ref, device)
    default_thresholds, default_main = chain_scores(cutter_kind(cutter_model))
    main_score_threshold = float(default_main if main_score_threshold is None else main_score_threshold)
    thresholds = sorted({float(value) for value in (*(score_thresholds or default_thresholds), main_score_threshold)})
    model, checkpoint, classes, transform = _load_classifier(classifier_ref, device)
    scheme = checkpoint_class_scheme(checkpoint)
    folded = checkpoint.get("folded_labels") if isinstance(checkpoint.get("folded_labels"), dict) else {}

    tallies = {value: ChainTally() for value in thresholds}
    images = []
    for frame, pixels, predictions in cutter_boxes(cutter_model, device, frames, data_root, min_score=thresholds[0],
                                                   batch_size=batch_size, num_workers=num_workers, task="chain"):
        height, width = int(pixels.shape[1]), int(pixels.shape[2])
        cut = crop_cutter(data_root, frame, pixels)
        # Every box of the lowest threshold, cut from the frame at full resolution and classified once.
        for start in range(0, len(predictions), CLASSIFIER_BATCH):
            chunk = predictions[start:start + CLASSIFIER_BATCH]
            crops = []
            for item in chunk:
                crops.append(transform(cut(crop_rectangle(item["box"], width, height))))
            with torch.no_grad():
                probabilities = torch.softmax(model(torch.stack(crops).to(device)).float(), dim=1)
            for item, index, confidence in zip(chunk, probabilities.argmax(1).tolist(),
                                               probabilities.max(1).values.tolist()):
                item["predicted"], item["confidence"] = classes[index], float(confidence)
        truth = truth_objects(frame, scheme, classes, folded)
        # Every box of the lowest threshold, matched there once, for the Watch panel of the Evaluate tab.
        floor_pairs = match_boxes(predictions, truth, iou_threshold)
        for value in thresholds:
            selected = [item for item in predictions if item["score"] >= value]
            pairs = tallies[value].add(frame["label_basis"], truth, selected, iou_threshold)
            if value == main_score_threshold:
                by_truth = {index: selected[position] for position, index in pairs.items()}
                images.append({
                    "image_path": frame["image_path"], "image_sha256": frame["image_sha256"],
                    "route_id": frame["route_id"], "segment_index": frame["segment_index"],
                    "label_basis": frame["label_basis"], "boxes": len(selected), "size": [width, height],
                    "truth": [{"box": item["box"], "label": item["label"], "speed_sign": item["speed_sign"],
                               "family": item["family"], "label_basis": item["label_basis"], "found": index in by_truth,
                               "predicted": by_truth[index]["predicted"] if index in by_truth else None,
                               "confidence": by_truth[index]["confidence"] if index in by_truth else None,
                               "correct": index in by_truth and by_truth[index]["predicted"] == item["label"]}
                              for index, item in enumerate(truth)],
                    "surviving_false": [{"box": [round(v, 1) for v in item["box"]], "score": round(item["score"], 4),
                                         "predicted": item["predicted"], "confidence": round(item["confidence"], 4)}
                                        for position, item in enumerate(selected)
                                        if position not in pairs and item["predicted"] != NOT_A_SIGN],
                    **per_frame_record(predictions, floor_pairs, chain=True),
                })

    split_sha = loaded["split_sha256"]
    cutter_info = cutter_summary(cutter_model, cutter_checkpoint, cutter_ref, split_sha, input_size)
    classifier_info = {**_model_summary(checkpoint, classifier_ref, split_sha), "class_scheme": scheme,
                       "classes": len(classes), "rejects": NOT_A_SIGN in classes}
    limitations = ["speed_signs_are_the_speed_sign_families_other_signs_are_in_all_boxes",
                   "found_is_iou_0.5_greedy_by_score_one_box_per_sign",
                   "crops_cut_from_the_cutter_box_at_full_resolution"]
    if cutter_info["kind"] == "signnet":
        limitations.append(f"signnet_top_band_and_{CAR_MAX_BOXES}_boxes_a_frame_as_the_car: a sign below the band "
                           "counts as missed")
    if not cutter_info["same_split"]:
        limitations.append("cutter_trained_on_another_split: the partition may hold frames it trained on")
    if not classifier_info["same_split"]:
        limitations.append("classifier_trained_on_another_split: the partition may hold frames it trained on")
    if not classifier_info["rejects"]:
        limitations.append("classifier_without_not_a_sign: every false box survives")
    truth_all = [item for frame in frames for item in truth_objects(frame, scheme, classes, folded)]
    return {
        "schema_version": 1, "task": "chain", "partition": partition,
        "cutter": cutter_info, "classifier": classifier_info,
        "dataset_sha256": loaded["dataset_sha256"], "split_sha256": split_sha, "iou_threshold": iou_threshold,
        "image_count": len(frames),
        "human_image_count": sum(1 for frame in frames if frame["label_basis"] == "human_review"),
        "speed_sign_count": sum(1 for item in truth_all if item["speed_sign"]),
        "reviewed_box_count": len(truth_all),
        "score_thresholds": thresholds, "main_score_threshold": main_score_threshold,
        "by_score_threshold": {f"{value:g}": tallies[value].metrics() for value in thresholds},
        "images": images,
        # How each image keeps every box of the cutter from the lowest threshold, with the classifier's answer.
        "per_frame_boxes": per_frame_descriptor(chain=True, min_score=thresholds[0]),
        "device": str(device), "seconds": round(time.perf_counter() - started, 1),
        "limitations": limitations,
    }


# -- the comparison with the old reader (scripts/run_classifier.sh) -------------------------------------------
def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _share(value: Any, part: Any = None, whole: Any = None) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    text = f"{value:.2f}"
    return f"{text} {part}/{whole}" if part is not None and whole else text


def _number(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{value:.2f}"


def _table(lines: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip() for line in lines]


def classifier_comparison(run_root: Path, old_run_root: Path | None = None) -> str:
    """The new classifier of run_root (scripts/run_classifier.sh) next to the old reader of old_run_root, where
    they compare: alone on the test sign crops (with the old reader's labels: the kind of an other sign dropped),
    and in the chain with the same cutter on the same test frames (evaluation/chain-test.json and
    chain-test-old-reader.json). What the old reader cannot do (reject a box) is shown for the new one only."""
    run_root = Path(run_root)
    out: list[str] = []
    result = _read(run_root / "classifier" / "result.json") or {}
    if result:
        out.append(f"== training: best epoch {result.get('best_epoch', '-')} of {len(result.get('history') or [])} run, "
                   f"validation: signs read right {_number(result.get('best_validation_sign_accuracy'))}, not_a_sign "
                   f"rejected {_number(result.get('best_validation_rejection'))}, balanced "
                   f"{_number(result.get('best_validation_balanced_accuracy'))}; {len(result.get('classes') or [])} "
                   f"classes; not_a_sign loss weight {_number((result.get('loss_weights') or {}).get(NOT_A_SIGN))} ==")
    old_alone = _read(old_run_root / "evaluation" / "reader-test.json") if old_run_root else None
    for partition in ("test", "validation"):
        new = _read(run_root / "evaluation" / f"classifier-{partition}.json")
        if new is None:
            out += ["", f"== the classifier alone, {partition}: not evaluated =="]
            continue
        signs = new.get("signs") or {}
        negatives = new.get("not_a_sign") or {}
        human = signs.get("human_reviewed") or {}
        old = old_alone if partition == "test" else None
        same = None
        if old is not None:
            same = (old.get("dataset_sha256") == new.get("dataset_sha256")
                    and old.get("split_sha256") == new.get("split_sha256"))
        old_human = (old or {}).get("human_reviewed") or {}
        out += ["", f"== the classifier alone, {partition}: {signs.get('sample_count', 0)} sign crops "
                    f"({human.get('sample_count', 0)} confirmed by a person) and {negatives.get('crops', 0)} not_a_sign "
                    f"crops ==" + ("" if old is None else
                                   f" (old reader: its own report, {'same' if same else 'ANOTHER'} dataset and split)")]
        rows = [("", "new classifier", "old reader" if old is not None else "")]

        def right(report, key="accuracy"):
            value, count = (report or {}).get(key), (report or {}).get("sample_count")
            return "-" if value is None else f"{value:.2f}" + (f" of {count}" if count else "")
        rows.append(("signs read right (old reader's labels)", right(signs, "family_accuracy"), right(old)))
        rows.append(("  confirmed by a person", right(human), right(old_human)))
        rows.append(("signs read right (own labels, kinds too)", right(signs), ""))
        rows.append(("signs called not_a_sign", _share(signs.get("rejected_rate"), signs.get("rejected"),
                                                       signs.get("sample_count")), "no such class" if old else ""))
        rows.append(("not_a_sign rejected", _share(negatives.get("rejection_rate"), negatives.get("rejected"),
                                                   negatives.get("crops")), "no such class" if old else ""))
        for source, label in (("rejected", "  removed proposals"), ("mined", "  cutter's false boxes"),
                              ("reviewed", "  marked not a sign")):
            cell = (negatives.get("by_source") or {}).get(source) or {}
            if cell.get("crops"):
                rows.append((label, _share(cell.get("rejection_rate"), cell.get("rejected"), cell.get("crops")), ""))
        out += _table(rows)

    for partition in ("test", "validation"):
        new = _read(run_root / "evaluation" / f"chain-{partition}.json")
        old = _read(run_root / "evaluation" / f"chain-{partition}-old-reader.json")
        if new is None:
            out += ["", f"== the chain on {partition}: not evaluated =="]
            continue
        cutter = new.get("cutter") or {}
        out += ["", f"== the chain on {partition}: cutter {cutter.get('id') or cutter.get('checkpoint')} "
                    f"({new.get('image_count')} frames, {new.get('speed_sign_count')} speed signs; e2e = found by the "
                    f"cutter AND read right; widths of the reviewed box in frame pixels; surviving = false boxes not "
                    f"rejected) =="]
        header = ("score", "classifier", "e2e", "human", "<12", "12-20", "20-40", ">=40", "cutter_found",
                  "boxes/frame", "false/frame", "surviving/frame")
        lines: list[tuple[str, ...]] = [header]
        thresholds = sorted((new.get("by_score_threshold") or {}), key=float)
        for score in thresholds:
            for name, report in (("new", new), ("old reader", old)):
                if report is None:
                    continue
                metrics = (report.get("by_score_threshold") or {}).get(score)
                if not isinstance(metrics, dict):
                    continue
                speed = metrics.get("speed_signs") or {}
                widths = speed.get("by_width") or {}
                human = speed.get("human_reviewed") or {}
                lines.append((score, name, _share(speed.get("recall"), speed.get("correct"), speed.get("truth")),
                              _share(human.get("recall"), human.get("correct"), human.get("truth")),
                              *(_share((widths.get(key) or {}).get("recall"), (widths.get(key) or {}).get("correct"),
                                       (widths.get(key) or {}).get("truth")) for key in ("lt12", "12-20", "20-40", "ge40")),
                              _share(speed.get("recall_found"), speed.get("found"), speed.get("truth")),
                              _number(metrics.get("boxes_per_frame")), _number(metrics.get("false_per_frame")),
                              _number(metrics.get("surviving_false_per_frame"))))
        out += _table(lines)
    return "\n".join(out).lstrip("\n") + "\n"
