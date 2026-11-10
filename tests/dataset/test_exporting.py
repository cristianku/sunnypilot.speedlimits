from __future__ import annotations

import pytest

from nnslr_tools.exporting import detector_io_contract, reader_io_contract


def test_reader_io_contract_freezes_preprocessing_and_classes() -> None:
    contract = reader_io_contract(
        {
            "task": "reader",
            "classes": ["maximum_speed:30", "maximum_speed:50"],
            "input_size": 160,
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        }
    )

    assert contract["input"]["shape"] == [1, 3, 160, 160]
    assert contract["input"]["layout"] == "NCHW"
    assert contract["input"]["color"] == "RGB"
    assert contract["input"]["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert contract["output"]["shape"] == [1, 2]
    assert contract["output"]["classes"] == ["maximum_speed:30", "maximum_speed:50"]


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"classes": [], "input_size": 160},
        {"classes": ["maximum_speed:30"], "input_size": 0},
        {"classes": [30], "input_size": 160},
    ],
)
def test_reader_io_contract_rejects_incomplete_checkpoint_metadata(checkpoint) -> None:
    with pytest.raises(ValueError):
        reader_io_contract(checkpoint)



def test_detector_io_contract_freezes_full_frame_interface() -> None:
    contract = detector_io_contract(
        {
            "task": "detector",
            "classes": ["background", "speed_sign"],
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        }
    )

    assert contract["input"]["shape"] == [1, 3, 760, 1344]
    assert contract["input"]["value_range"] == [0.0, 1.0]
    assert contract["input"]["external_resize"] is None
    assert contract["input"]["model_internal_resize"] == [320, 320]
    assert contract["input"]["model_internal_normalization"] == {
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    assert contract["outputs"]["boxes"]["format"] == "xyxy"
    assert contract["outputs"]["labels"]["mapping"] == {"1": "speed_sign"}
    assert contract["postprocessing"] == "model_includes_score_filtering_and_nms"


def test_detector_io_contract_rejects_wrong_checkpoint() -> None:
    with pytest.raises(ValueError):
        detector_io_contract({"task": "reader", "classes": ["speed_sign"]})



def test_detector_io_contract_supports_explicit_native_dimensions() -> None:
    contract = detector_io_contract(
        {
            "task": "detector",
            "classes": ["background", "speed_sign"],
            "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64,
        },
        input_width=1920,
        input_height=1080,
    )
    assert contract["input"]["shape"] == [1, 3, 1080, 1920]


@pytest.mark.parametrize("width,height", [(0, 760), (1344, 0), (-1, 760)])
def test_detector_io_contract_rejects_invalid_native_dimensions(width, height) -> None:
    with pytest.raises(ValueError):
        detector_io_contract(
            {
                "task": "detector",
                "classes": ["background", "speed_sign"],
            },
            input_width=width,
            input_height=height,
        )


def _decider_checkpoint(variant: str) -> dict:
    return {"task": "experiment_b", "variant": variant, "architecture": "context_" + variant + "_mobilenet_v3_small",
            "classes": ["maximum_speed:50", "maximum_speed:80", "not_a_sign"], "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64, "initialization": "random"}


def test_the_cars_decider_contract_has_fixed_boxes_and_roi_weights() -> None:
    from nnslr_tools.exporting import car_decider_contract

    contract = car_decider_contract(_decider_checkpoint("full"), 5, (7, 12))
    assert (contract["task"], contract["target"], contract["boxes"]) == ("decider", "car", 5)
    inputs = contract["inputs"]
    assert list(inputs) == ["crops", "geometry", "scene", "roi_weights"]  # no rois: tinygrad has no RoiAlign
    assert inputs["crops"]["shape"] == [5, 3, 96, 96] and inputs["geometry"]["shape"] == [5, 10]
    assert inputs["scene"]["shape"] == [1, 3, 216, 384]
    assert inputs["roi_weights"]["shape"] == [5, 4, 84] and inputs["roi_weights"]["map"] == [7, 12]
    assert (inputs["roi_weights"]["stride"], inputs["roi_weights"]["size"], inputs["roi_weights"]["sampling_ratio"]) == (32, 2, 2)
    assert contract["outputs"] == {"ownership": {**contract["outputs"]["ownership"], "shape": [5, 3],
                                                 "classes": ["own_road", "other_road", "unresolved"]}}
    assert contract["plate_families"] == {"maximum_speed_hours": "maximum_speed", "maximum_speed_wet": "maximum_speed"}
    assert list(car_decider_contract(_decider_checkpoint("crop"), 5, (0, 0))["inputs"]) == ["crops"]


@pytest.mark.parametrize("variant", ["crop_geom", "full"])
def test_the_cars_decider_is_the_decider_of_pytorch(tmp_path, variant) -> None:
    # Random weights, no training: the export with RoIAlign as weights against torchvision's roi_align, the weights
    # made by the comma's own applicability.py.
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import json

    from nnslr_tools.context_experiment import _build_model, _require_experiment_stack
    from nnslr_tools.exporting import CAR_DECIDER_FILE, export_decider_car_onnx
    from nnslr_tools.road_type_signs import package_path

    if not (package_path() / CAR_DECIDER_FILE).is_file():
        pytest.skip("no sunnypilot clone with the comma's applicability.py")
    _, _, _, _, _, _, mobilenet_v3_small, roi_align, _ = _require_experiment_stack()
    torch.manual_seed(0)
    checkpoint = _decider_checkpoint(variant)
    model = _build_model(torch, mobilenet_v3_small, None, roi_align, variant, len(checkpoint["classes"]))
    torch.save({**checkpoint, "state_dict": model.state_dict()}, tmp_path / "decider.pt")
    result = export_decider_car_onnx(tmp_path / "decider.pt", tmp_path / "decider.onnx")
    assert result["verification"]["passed"] is True and result["verification"]["boxes_tried"] == [1, 3, 5]
    contract = json.loads((tmp_path / "decider.onnx.contract.json").read_text())
    assert contract["target"] == "car" and contract["onnx_sha256"] == result["onnx_sha256"]
    assert ("roi_weights" in contract["inputs"]) == (variant == "full")
    # The same bytes each time: Publish finds nothing new when nothing changed.
    again = export_decider_car_onnx(tmp_path / "decider.pt", tmp_path / "again.onnx", verify=False)
    assert again["onnx_sha256"] == result["onnx_sha256"]
