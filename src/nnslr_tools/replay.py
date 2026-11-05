"""Offline detector -> reader replay over extracted route frames.

This is perception replay only. It does not perform temporal consensus, road
ownership, passage, current-limit inference or vehicle control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nnslr_tools.checkpoints import load_detector, safe_torch_load
from nnslr_tools.sign_types import class_parts
from nnslr_tools.training import _require_detector_stack, _require_training_stack


def decode_reader_label(label: str) -> dict[str, Any]:
    if not isinstance(label, str) or not label:
        raise ValueError("invalid_reader_label")
    family, suffix, panel = class_parts(label)
    if panel is not None:
        if not suffix.isdigit() or int(suffix) <= 0:
            raise ValueError(f"invalid_reader_label: {label}")
        return {
            "sign_family": family,
            "value_state": "read",
            "value_kph": int(suffix),
            "supplementary_panel": panel,
            "supported_domain": True,
        }
    if ":" in label:
        try:
            value = int(suffix)
        except ValueError:
            if suffix not in {"unknown", "unreadable", "not_applicable"}:
                raise ValueError(f"invalid_reader_label: {label}")
            return {
                "sign_family": family,
                "value_state": suffix,
                "value_kph": None,
                "supported_domain": family != "other_sign",
            }
        if value <= 0:
            raise ValueError(f"invalid_reader_label: {label}")
        return {
            "sign_family": family,
            "value_state": "read",
            "value_kph": value,
            "supported_domain": family != "other_sign",
        }

    if label == "unreadable":
        return {
            "sign_family": "unreadable",
            "value_state": "unreadable",
            "value_kph": None,
            "supported_domain": True,
        }
    if label == "other_sign":
        return {
            "sign_family": "other_sign",
            "value_state": "not_applicable",
            "value_kph": None,
            "supported_domain": False,
        }
    if label == "cancellation":
        return {
            "sign_family": "cancellation",
            "value_state": "unknown",
            "value_kph": None,
            "supported_domain": True,
        }
    raise ValueError(f"invalid_reader_label: {label}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_frame_path(root: Path, raw: str) -> Path:
    root = root.resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe_image_path: {raw}")
    if not path.is_file():
        raise ValueError(f"missing_image: {raw}")
    return path


def load_route_frames(data_root: Path, route_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    base = Path(data_root) / "derived" / "frames" / route_id
    if not base.is_dir():
        raise ValueError(f"route_frames_not_found: {route_id}")
    manifests = sorted(
        base.glob("*/frames.jsonl"),
        key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name,
    )
    if not manifests:
        raise ValueError(f"frame_manifests_not_found: {route_id}")

    frames = []
    for manifest in manifests:
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("image_path"), str):
                raise ValueError(f"invalid_frame_manifest: {manifest}:{line_no}")
            if row.get("route_id") not in (None, route_id):
                raise ValueError(f"route_mismatch: {manifest}:{line_no}")
            _safe_frame_path(Path(data_root), row["image_path"])
            frames.append(row)
    frames.sort(
        key=lambda r: (
            -1 if r.get("segment_index") is None else r["segment_index"],
            r.get("decoded_frame_index", r.get("output_index", 0)),
        )
    )
    return frames, [str(path) for path in manifests]


def _load_models(detector_checkpoint: Path, reader_checkpoint: Path, device_name: str):
    (
        torch,
        Image,
        _,
        _,
        TF,
        _,
        ssdlite320_mobilenet_v3_large,
    ) = _require_detector_stack()
    _, _, _, _, transforms, _, mobilenet_v3_small = _require_training_stack()

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda_unavailable")
    device = torch.device(device_name)

    detector_payload = safe_torch_load(Path(detector_checkpoint))
    reader_payload = safe_torch_load(Path(reader_checkpoint))
    if detector_payload.get("task") != "detector":
        raise ValueError("checkpoint_task_mismatch: expected detector")
    if reader_payload.get("task") != "reader":
        raise ValueError("checkpoint_task_mismatch: expected reader")
    if (
        detector_payload.get("dataset_sha256") != reader_payload.get("dataset_sha256")
        or detector_payload.get("split_sha256") != reader_payload.get("split_sha256")
    ):
        raise ValueError("checkpoint_provenance_mismatch")

    # Rebuilt as it was trained: the tail of its weights, its input size and default boxes.
    detector = load_detector(detector_payload)
    detector.to(device).eval()

    classes = reader_payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("checkpoint_invalid_classes")
    input_size = int(reader_payload.get("input_size", 160))
    reader = mobilenet_v3_small(weights=None)
    reader.classifier[-1] = torch.nn.Linear(reader.classifier[-1].in_features, len(classes))
    reader.load_state_dict(reader_payload["state_dict"], strict=True)
    reader.to(device).eval()
    reader_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225)),
    ])
    return torch, Image, TF, device, detector, reader, reader_transform, classes, detector_payload, reader_payload


def replay_route(
    data_root: Path,
    route_id: str,
    detector_checkpoint: Path,
    reader_checkpoint: Path,
    output_path: Path,
    *,
    device_name: str = "cuda",
    detector_threshold: float = .25,
    reader_threshold: float = 0.0,
    max_detections: int = 20,
) -> dict[str, Any]:
    if not 0 <= detector_threshold <= 1 or not 0 <= reader_threshold <= 1:
        raise ValueError("invalid_confidence_threshold")
    if type(max_detections) is not int or max_detections < 1:
        raise ValueError("invalid_max_detections")

    frames, manifests = load_route_frames(Path(data_root), route_id)
    (
        torch,
        Image,
        TF,
        device,
        detector,
        reader,
        reader_transform,
        classes,
        detector_payload,
        reader_payload,
    ) = _load_models(detector_checkpoint, reader_checkpoint, device_name)

    rows = []
    detection_count = 0
    supported_count = 0
    with torch.no_grad():
        for frame in frames:
            path = _safe_frame_path(Path(data_root), frame["image_path"])
            with Image.open(path) as source:
                image = source.convert("RGB")
                full_tensor = TF.pil_to_tensor(image).float().div_(255.0).to(device)
                det = detector([full_tensor])[0]
                boxes = det["boxes"].detach().cpu().tolist()
                scores = det["scores"].detach().cpu().tolist()
                labels = det["labels"].detach().cpu().tolist()

                candidates = [
                    (box, float(score))
                    for box, score, label in zip(boxes, scores, labels)
                    if label == 1 and score >= detector_threshold
                ][:max_detections]

                detections = []
                for box, detector_score in candidates:
                    x1, y1, x2, y2 = box
                    x1 = max(0, min(image.width - 1, int(x1)))
                    y1 = max(0, min(image.height - 1, int(y1)))
                    x2 = max(x1 + 1, min(image.width, int(x2 + .999)))
                    y2 = max(y1 + 1, min(image.height, int(y2 + .999)))
                    crop = image.crop((x1, y1, x2, y2))
                    logits = reader(reader_transform(crop).unsqueeze(0).to(device))
                    probabilities = torch.softmax(logits, dim=1)[0]
                    reader_score, class_index = probabilities.max(0)
                    score = float(reader_score.detach().cpu())
                    label = classes[int(class_index.detach().cpu())]
                    decoded = decode_reader_label(label)
                    if score < reader_threshold:
                        decoded = {
                            "sign_family": decoded["sign_family"],
                            "value_state": "unknown",
                            "value_kph": None,
                            "supported_domain": decoded["supported_domain"],
                        }
                    detections.append({
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "detection_score": detector_score,
                        "classification_score": score,
                        "reader_label": label,
                        **decoded,
                    })
                    detection_count += 1
                    supported_count += int(decoded["supported_domain"])

            rows.append({
                "schema_version": 1,
                "kind": "offline_perception_replay",
                "route_id": route_id,
                "segment_index": frame.get("segment_index"),
                "output_index": frame.get("output_index"),
                "decoded_frame_index": frame.get("decoded_frame_index"),
                "camera_stream": frame.get("camera_stream"),
                "image_path": frame["image_path"],
                "image_sha256": frame.get("image_sha256"),
                "media_time_s": frame.get("media_time_s"),
                "media_time_provenance": frame.get("media_time_provenance"),
                "capture_mono_ns": frame.get("capture_mono_ns"),
                "capture_time_provenance": frame.get("capture_time_provenance"),
                "alignment_status": frame.get("alignment_status"),
                "detections": detections,
            })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "valid": True,
        "route_id": route_id,
        "frame_count": len(rows),
        "detection_count": detection_count,
        "supported_detection_count": supported_count,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "manifests": manifests,
        "detector_checkpoint": str(detector_checkpoint),
        "reader_checkpoint": str(reader_checkpoint),
        "dataset_sha256": detector_payload.get("dataset_sha256"),
        "split_sha256": detector_payload.get("split_sha256"),
        "device": str(device),
        "limitations": [
            "perception_only_no_temporal_consensus",
            "no_road_ownership",
            "no_passage_or_current_limit",
            "no_vehicle_control",
        ],
    }



def replay_bundle_route(
    data_root: Path,
    route_id: str,
    bundle_dir: Path,
    output_path: Path,
    *,
    detector_threshold: float | None = None,
    reader_threshold: float | None = None,
    max_detections: int = 20,
) -> dict[str, Any]:
    """Replay a verified packaged ONNX bundle through the CPU reference runtime."""
    from nnslr_tools.bundle_runtime import ReferenceBundleRunner

    frames, manifests = load_route_frames(Path(data_root), route_id)
    runner = ReferenceBundleRunner(Path(bundle_dir))

    rows = []
    detection_count = 0
    supported_count = 0
    for frame in frames:
        path = _safe_frame_path(Path(data_root), frame["image_path"])
        detections = runner.run_image(
            path,
            detector_threshold=detector_threshold,
            reader_threshold=reader_threshold,
            max_detections=max_detections,
        )
        detection_count += len(detections)
        supported_count += sum(int(d["supported_domain"]) for d in detections)
        rows.append({
            "schema_version": 1,
            "kind": "offline_bundle_perception_replay",
            "route_id": route_id,
            "segment_index": frame.get("segment_index"),
            "output_index": frame.get("output_index"),
            "decoded_frame_index": frame.get("decoded_frame_index"),
            "camera_stream": frame.get("camera_stream"),
            "image_path": frame["image_path"],
            "image_sha256": frame.get("image_sha256"),
            "media_time_s": frame.get("media_time_s"),
            "media_time_provenance": frame.get("media_time_provenance"),
            "capture_mono_ns": frame.get("capture_mono_ns"),
            "capture_time_provenance": frame.get("capture_time_provenance"),
            "alignment_status": frame.get("alignment_status"),
            "detections": detections,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return {
        "valid": True,
        "route_id": route_id,
        "frame_count": len(rows),
        "detection_count": detection_count,
        "supported_detection_count": supported_count,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "manifests": manifests,
        "bundle": str(Path(bundle_dir)),
        "bundle_digest": runner.bundle_report["bundle_digest"],
        "dataset_sha256": runner.bundle_report["dataset_sha256"],
        "split_sha256": runner.bundle_report["split_sha256"],
        "runtime": "onnxruntime_cpu_reference",
        "target_compatible": False,
        "limitations": [
            "perception_only_no_temporal_consensus",
            "no_road_ownership",
            "no_passage_or_current_limit",
            "no_vehicle_control",
            "reference_runtime_not_target_backend",
        ],
    }
