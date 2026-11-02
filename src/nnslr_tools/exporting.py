"""Reference ONNX export for trained NNSLR baselines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nnslr_tools.checkpoints import (DETECTOR_ANCHOR_MAX_RATIO, detector_anchor_min_ratio, detector_input_size,
                                     load_detector, safe_torch_load)
from nnslr_tools.training import _require_detector_stack, _require_training_stack


def _export_onnx(module, example, output_path: Path, *, input_names: list[str], output_names: list[str]) -> None:
    """Write ``module`` as ONNX: with torch.export (dynamo) first, else with the TorchScript exporter.

    torch 2.14 cannot torch.export the SSDlite detector with its post-processing (non-maximum suppression has
    data-dependent shapes); the TorchScript exporter (opset 17) handles it, and the parity check that follows
    compares the result with PyTorch either way.
    """
    import torch

    try:
        program = torch.onnx.export(module, (example,), input_names=input_names, output_names=output_names,
                                    dynamo=True, external_data=False)
        program.save(str(output_path))
        return
    except ImportError as exc:
        raise ValueError(
            "onnx_dependencies_missing: install onnx and onnxscript in the training environment"
        ) from exc
    except Exception as exc:  # torch.export refuses the graph: the TorchScript exporter below
        print(f"torch.export could not export {type(module).__name__} ({type(exc).__name__}); "
              f"using the TorchScript ONNX exporter", flush=True)
    torch.onnx.export(module, (example,), str(output_path), input_names=input_names, output_names=output_names,
                      dynamo=False, opset_version=17)


def _matched_detections(expected, actual, np, *, box_tolerance: float = 0.05,
                        score_tolerance: float = 1e-4) -> tuple[bool, bool, bool]:
    """(boxes, scores, labels) agree between two detector outputs, matching boxes one to one."""
    expected_boxes, expected_scores, expected_labels = expected
    actual_boxes, actual_scores, actual_labels = actual
    if sorted(expected_labels.tolist()) != sorted(actual_labels.tolist()):
        return False, False, False
    free = list(range(len(actual_boxes)))
    boxes_ok = scores_ok = True
    for index in range(len(expected_boxes)):
        candidates = [j for j in free if actual_labels[j] == expected_labels[index]]
        if not candidates:
            return False, False, False
        distance = [float(np.abs(actual_boxes[j] - expected_boxes[index]).max()) for j in candidates]
        best = candidates[int(np.argmin(distance))]
        free.remove(best)
        boxes_ok = boxes_ok and min(distance) <= box_tolerance
        scores_ok = scores_ok and abs(float(actual_scores[best]) - float(expected_scores[index])) <= score_tolerance
    return boxes_ok, scores_ok, True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reader_io_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    classes = checkpoint.get("classes")
    input_size = checkpoint.get("input_size")
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("checkpoint_invalid_classes")
    if type(input_size) is not int or input_size <= 0:
        raise ValueError("checkpoint_invalid_input_size")
    return {
        "schema_version": 1,
        "task": "reader",
        "architecture": "mobilenet_v3_small",
        "input": {
            "name": "images",
            "shape": [1, 3, input_size, input_size],
            "dtype": "float32",
            "layout": "NCHW",
            "color": "RGB",
            "source": "reviewed_bbox_crop",
            "resize": [input_size, input_size],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "output": {
            "name": "logits",
            "shape": [1, len(classes)],
            "dtype": "float32",
            "classes": classes,
            "decode": "argmax",
            "number_semantics": "written_sign_value",
            "unit_source": "runtime_country",
        },
        "dataset_sha256": checkpoint.get("dataset_sha256"),
        "split_sha256": checkpoint.get("split_sha256"),
        "training_initialization": checkpoint.get("initialization", "unknown"),
    }


def detector_io_contract(
    checkpoint: dict[str, Any],
    *,
    input_width: int = 1344,
    input_height: int = 760,
) -> dict[str, Any]:
    if checkpoint.get("task") != "detector":
        raise ValueError("checkpoint_task_mismatch: expected detector")
    if type(input_width) is not int or input_width <= 0:
        raise ValueError("invalid_detector_input_width")
    if type(input_height) is not int or input_height <= 0:
        raise ValueError("invalid_detector_input_height")
    classes = checkpoint.get("classes")
    if classes not in (["background", "speed_sign"], ["speed_sign"]):
        raise ValueError("checkpoint_invalid_classes")
    # The size the model resizes the frame to, as it was trained (320 x 320 for checkpoints that do not say).
    internal_width, internal_height = detector_input_size(checkpoint)
    return {
        "schema_version": 1,
        "task": "detector",
        "architecture": "ssdlite320_mobilenet_v3_large",
        "input": {
            "name": "images",
            "shape": [1, 3, input_height, input_width],
            "dtype": "float32",
            "layout": "NCHW",
            "color": "RGB",
            "value_range": [0.0, 1.0],
            "source": "native_narrow_road_frame",
            "external_resize": None,
            "model_internal_resize": [internal_width, internal_height],
            "model_internal_resize_order": "width_height",
            "model_internal_resize_mode": "bilinear",
            "model_internal_normalization": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
        },
        "outputs": {
            "boxes": {"shape": ["detections", 4], "dtype": "float32", "format": "xyxy"},
            "scores": {"shape": ["detections"], "dtype": "float32"},
            "labels": {"shape": ["detections"], "dtype": "int64", "mapping": {"1": "speed_sign"}},
        },
        "postprocessing": "model_includes_score_filtering_and_nms",
        "default_boxes": {
            "min_ratio": detector_anchor_min_ratio(checkpoint),
            "max_ratio": DETECTOR_ANCHOR_MAX_RATIO,
            "aspect_ratios": [2, 3],
        },
        "dataset_sha256": checkpoint.get("dataset_sha256"),
        "split_sha256": checkpoint.get("split_sha256"),
        "training_initialization": checkpoint.get("initialization", "unknown"),
    }


def export_detector_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    verify: bool = True,
    input_width: int = 1344,
    input_height: int = 760,
    example_image: Path | None = None,
) -> dict[str, Any]:
    """The cutter (detector) as ONNX with its post-processing, checked against PyTorch with onnxruntime. With
    example_image (a frame of that size), the check runs on it: on random noise the detector returns hundreds of
    overlapping boxes that cannot be matched one to one reliably."""
    (
        torch,
        _,
        _,
        _,
        _,
        _,
        ssdlite320_mobilenet_v3_large,
    ) = _require_detector_stack()

    checkpoint = safe_torch_load(Path(checkpoint_path))
    contract = detector_io_contract(
        checkpoint,
        input_width=input_width,
        input_height=input_height,
    )

    # Rebuilt as it was trained: the tail of its weights, its input size and default boxes.
    model = load_detector(checkpoint)
    model.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, images):
            result = self.detector([images[0]])[0]
            return result["boxes"], result["scores"], result["labels"]

    wrapper = ExportWrapper(model).eval()
    example = torch.rand(1, 3, input_height, input_width, dtype=torch.float32)
    if example_image is not None:
        from PIL import Image
        from torchvision.transforms import functional as TF
        with Image.open(example_image) as image:
            frame = TF.pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        if tuple(frame.shape) != (3, input_height, input_width):
            raise ValueError(f"example_image_size: {example_image} is not {input_width} x {input_height}")
        example = frame.unsqueeze(0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _export_onnx(wrapper, example, output_path, input_names=["images"], output_names=["boxes", "scores", "labels"])

    try:
        import onnx
    except ImportError as exc:
        raise ValueError("onnx_dependencies_missing: install onnx") from exc
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    verification = {"enabled": verify, "passed": None}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("onnxruntime_missing") from exc
        with torch.no_grad():
            expected = wrapper(example)
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        actual = session.run(["boxes", "scores", "labels"], {"images": example.numpy()})
        import numpy as np
        expected_np = [tensor.detach().cpu().numpy() for tensor in expected]
        shape_match = all(a.shape == b.shape for a, b in zip(expected_np, actual))
        # Boxes with nearly equal scores may come out in another order: each PyTorch box is matched to the
        # closest ONNX box of the same label instead of being compared position by position.
        boxes_match = scores_match = labels_match = False
        if shape_match:
            boxes_match, scores_match, labels_match = _matched_detections(expected_np, actual, np)
        passed = bool(shape_match and boxes_match and scores_match and labels_match)
        verification = {
            "enabled": True,
            "passed": passed,
            "shape_match": bool(shape_match),
            "boxes_match": bool(boxes_match),
            "scores_match": bool(scores_match),
            "labels_match": bool(labels_match),
            "provider": "CPUExecutionProvider",
            "input": "a frame of the dataset" if example_image is not None else "random values",
        }
        if not passed:
            raise ValueError("onnx_parity_failed: detector outputs differ")

    contract["onnx_sha256"] = sha256_file(output_path)
    contract["checkpoint"] = str(checkpoint_path)
    contract["checkpoint_sha256"] = sha256_file(Path(checkpoint_path))
    contract["verification"] = verification
    contract_path = output_path.with_suffix(output_path.suffix + ".contract.json")
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "valid": True,
        "task": "detector",
        "onnx": str(output_path),
        "onnx_sha256": contract["onnx_sha256"],
        "contract": str(contract_path),
        "verification": verification,
    }


def export_reader_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    torch, _, _, _, _, _, mobilenet_v3_small = _require_training_stack()
    checkpoint = safe_torch_load(Path(checkpoint_path))
    if not isinstance(checkpoint, dict) or checkpoint.get("task") != "reader":
        raise ValueError("checkpoint_task_mismatch: expected reader")
    contract = reader_io_contract(checkpoint)
    classes = contract["output"]["classes"]
    input_size = contract["input"]["resize"][0]

    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    _export_onnx(model, example, output_path, input_names=["images"], output_names=["logits"])

    try:
        import onnx
    except ImportError as exc:
        raise ValueError("onnx_dependencies_missing: install onnx") from exc
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    verification = {"enabled": verify, "passed": None}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("onnxruntime_missing") from exc
        with torch.no_grad():
            expected = model(example).detach().cpu().numpy()
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        actual = session.run(["logits"], {"images": example.numpy()})[0]
        import numpy as np
        max_abs_error = float(np.max(np.abs(expected - actual)))
        passed = bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5))
        verification = {
            "enabled": True,
            "passed": passed,
            "max_abs_error": max_abs_error,
            "rtol": 1e-4,
            "atol": 1e-5,
            "provider": "CPUExecutionProvider",
        }
        if not passed:
            raise ValueError(f"onnx_parity_failed: max_abs_error={max_abs_error}")

    contract["onnx_sha256"] = sha256_file(output_path)
    contract["checkpoint"] = str(checkpoint_path)
    contract["checkpoint_sha256"] = sha256_file(Path(checkpoint_path))
    contract["verification"] = verification
    contract_path = output_path.with_suffix(output_path.suffix + ".contract.json")
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "valid": True,
        "task": "reader",
        "onnx": str(output_path),
        "onnx_sha256": contract["onnx_sha256"],
        "contract": str(contract_path),
        "verification": verification,
    }


def decider_io_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """What the decider's ONNX takes and gives: per variant (crop, crop_geom, full) its inputs, all the boxes of
    one frame at once (``boxes`` is dynamic), and the logits of its two heads."""
    from nnslr_tools.context_experiment import GEOMETRY_FEATURES, OWNERSHIP_CLASSES, OWNERSHIP_FAMILIES, VARIANTS

    variant = checkpoint.get("variant")
    if checkpoint.get("task") != "experiment_b" or variant not in VARIANTS:
        raise ValueError("checkpoint_task_mismatch: expected a decider (experiment B)")
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(v, str) for v in classes):
        raise ValueError("checkpoint_invalid_classes")
    crop = int(checkpoint.get("crop_size", 96))
    scene_width, scene_height = (int(v) for v in checkpoint.get("scene_size", (384, 216)))
    normalization = {"mean": list(checkpoint.get("normalize_mean", (0.485, 0.456, 0.406))),
                     "std": list(checkpoint.get("normalize_std", (0.229, 0.224, 0.225)))}
    inputs = {"crops": {"shape": ["boxes", 3, crop, crop], "dtype": "float32", "layout": "NCHW", "color": "RGB",
                        "source": "the box plus crop_margin of its width and height on each side, clipped to the "
                                  "frame, resized bilinear (PIL) to crop x crop",
                        "crop_margin": float(checkpoint.get("crop_margin", 0.25)), "normalization": normalization}}
    if variant in ("crop_geom", "full"):
        inputs["geometry"] = {"shape": ["boxes", len(GEOMETRY_FEATURES)], "dtype": "float32",
                              "features": list(checkpoint.get("geometry_features", GEOMETRY_FEATURES)),
                              "source": "corners, size and centre over the frame size; speed_mps / speed_scale_mps "
                                        "and 0, or 0 and 1 when the speed is missing",
                              "speed_scale_mps": float(checkpoint.get("speed_scale_mps", 40.0))}
    if variant == "full":
        inputs["scene"] = {"shape": [1, 3, scene_height, scene_width], "dtype": "float32", "layout": "NCHW",
                           "color": "RGB", "source": "the whole frame resized bilinear (PIL)",
                           "normalization": normalization}
        inputs["rois"] = {"shape": ["boxes", 5], "dtype": "float32",
                          "format": "0 (the frame), x1, y1, x2, y2 of the box in scene pixels"}
    return {
        "schema_version": 1,
        "task": "decider",
        "architecture": checkpoint.get("architecture"),
        "variant": variant,
        "inputs": inputs,
        "outputs": {
            "reading_logits": {"shape": ["boxes", len(classes)], "dtype": "float32", "classes": classes,
                               "decode": "softmax, argmax"},
            "ownership_logits": {"shape": ["boxes", len(OWNERSHIP_CLASSES)], "dtype": "float32",
                                 "classes": list(checkpoint.get("ownership_classes", OWNERSHIP_CLASSES)),
                                 "decode": "softmax, argmax; unresolved means it abstains"},
        },
        "ownership_families": list(checkpoint.get("ownership_families", OWNERSHIP_FAMILIES)),
        "dataset_sha256": checkpoint.get("dataset_sha256"),
        "split_sha256": checkpoint.get("split_sha256"),
        "training_initialization": checkpoint.get("initialization", "unknown"),
    }


def export_decider_onnx(checkpoint_path: Path, output_path: Path, *, verify: bool = True) -> dict[str, Any]:
    """The decider (a variant of experiment B) as ONNX, all the boxes of a frame in one call, checked against
    PyTorch with onnxruntime on one and on several boxes."""
    from nnslr_tools.checkpoints import model_weights
    from nnslr_tools.context_experiment import _build_model, _require_experiment_stack

    torch, _, _, _, _, _, mobilenet_v3_small, roi_align, _ = _require_experiment_stack()
    checkpoint = safe_torch_load(Path(checkpoint_path))
    contract = decider_io_contract(checkpoint)
    variant = contract["variant"]
    names = list(contract["inputs"])
    classes = contract["outputs"]["reading_logits"]["classes"]
    model = _build_model(torch, mobilenet_v3_small, None, roi_align, variant, len(classes))
    model.load_state_dict(model_weights(checkpoint), strict=True)
    model.eval()

    class ExportWrapper(torch.nn.Module):
        # Only the inputs of the variant: the ONNX has no unused input.
        def __init__(self, decider):
            super().__init__()
            self.decider = decider

        def forward(self, *inputs):
            given = dict(zip(names, inputs))
            return self.decider(given["crops"], given.get("geometry"), given.get("scene"), given.get("rois"))

    wrapper = ExportWrapper(model).eval()

    def example(boxes: int) -> tuple:
        values = {"crops": torch.randn(boxes, *contract["inputs"]["crops"]["shape"][1:])}
        if "geometry" in names:
            values["geometry"] = torch.rand(boxes, contract["inputs"]["geometry"]["shape"][1])
        if "scene" in names:
            _, _, height, width = contract["inputs"]["scene"]["shape"]
            values["scene"] = torch.randn(1, 3, height, width)
            corners = torch.rand(boxes, 2) * torch.tensor([width * .8, height * .8])
            sizes = 4 + torch.rand(boxes, 2) * 40
            values["rois"] = torch.cat([torch.zeros(boxes, 1), corners, corners + sizes], 1)
        return tuple(values[name] for name in names)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic = {name: {0: "boxes"} for name in names if name != "scene"}
    dynamic.update({"reading_logits": {0: "boxes"}, "ownership_logits": {0: "boxes"}})
    # The TorchScript exporter: RoIAlign (aligned, half_pixel) needs opset 16 or newer, and the box axis stays
    # dynamic.
    torch.onnx.export(wrapper, example(3), str(output_path), input_names=names,
                      output_names=["reading_logits", "ownership_logits"], dynamic_axes=dynamic, dynamo=False,
                      opset_version=17)
    try:
        import onnx
    except ImportError as exc:
        raise ValueError("onnx_dependencies_missing: install onnx") from exc
    onnx.checker.check_model(onnx.load(str(output_path)))

    verification = {"enabled": verify, "passed": None}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError("onnxruntime_missing") from exc
        import numpy as np
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        worst = 0.0
        passed = True
        for boxes in (1, 5):
            inputs = example(boxes)
            with torch.no_grad():
                expected = [tensor.numpy() for tensor in wrapper(*inputs)]
            actual = session.run(["reading_logits", "ownership_logits"],
                                 {name: value.numpy() for name, value in zip(names, inputs)})
            for want, got in zip(expected, actual):
                worst = max(worst, float(np.max(np.abs(want - got))))
                passed = passed and want.shape == got.shape and bool(np.allclose(want, got, rtol=1e-3, atol=1e-4))
        verification = {"enabled": True, "passed": passed, "max_abs_error": worst, "rtol": 1e-3, "atol": 1e-4,
                        "boxes_tried": [1, 5], "provider": "CPUExecutionProvider"}
        if not passed:
            raise ValueError(f"onnx_parity_failed: decider max_abs_error={worst}")

    contract["onnx_sha256"] = sha256_file(output_path)
    contract["checkpoint"] = str(checkpoint_path)
    contract["checkpoint_sha256"] = sha256_file(Path(checkpoint_path))
    contract["verification"] = verification
    contract_path = output_path.with_suffix(output_path.suffix + ".contract.json")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"valid": True, "task": "decider", "onnx": str(output_path), "onnx_sha256": contract["onnx_sha256"],
            "contract": str(contract_path), "verification": verification}
