from __future__ import annotations

import hashlib
import json
import sys

import pytest

from nnslr_tools.annotations import load_jsonl, validate_dataset
from nnslr_tools.training import prepare_detector_training, prepare_reader_training
from test_annotations import invoke, reviewed, write


def _training_fixture(root, capsys, *, ai=False):
    source_rows = []
    for index in range(8):
        row = reviewed(root, f"{index + 1:08x}--abcdef1234", index=index, negative=index == 0)
        if not index == 0:
            value = 30 if index % 2 else 50
            row["detections"][0]["value_kph"] = value
            row["source_proposals"][0]["value_kph"] = value
        if ai and index != 1:
            # Confirmed by an AI reviewer through the REST API or MCP; frame 1 stays a human review.
            row.update(reviewer="ai/Claude", label_basis="ai_review", reviewer_kind="ai",
                       review_channel="rest" if index % 2 else "mcp")
        source_rows.append(row)

    source = write(root, source_rows)
    code, out = invoke(capsys, "import-annotations", str(source), "--data-root", str(root))
    assert code == 0, out.err + out.out

    dataset = root / "annotations" / "objects.jsonl"
    canonical = load_jsonl(dataset)
    report = validate_dataset(canonical, root)
    assert report["valid"]

    negative = next(r for r in canonical if r["sign_family"] == "not_a_sign")
    thirties = [r for r in canonical if r["value_kph"] == 30]
    fifties = [r for r in canonical if r["value_kph"] == 50]
    assert len(thirties) >= 2 and len(fifties) >= 2

    assignments = {r["annotation_id"]: "test" for r in canonical}
    assignments[negative["annotation_id"]] = "hard-negative"
    assignments[thirties[0]["annotation_id"]] = "train"
    assignments[fifties[0]["annotation_id"]] = "train"
    assignments[thirties[1]["annotation_id"]] = "validation"
    assignments[fifties[1]["annotation_id"]] = "validation"
    remaining = [r for r in canonical if assignments[r["annotation_id"]] == "test"]
    if remaining:
        assignments[remaining[0]["annotation_id"]] = "route-held-out"

    payload = {
        "schema_version": 1,
        "dataset_kind": "gold",
        "dataset_sha256": report["dataset_sha256"],
        "assignments": dict(sorted(assignments.items())),
    }
    frozen = root / "splits" / "fixture.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    latest = root / "splits" / "latest.json"
    latest.write_text(
        json.dumps({"path": "splits/fixture.json", "sha256": digest}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset, frozen, latest


def test_prepare_reader_training_follows_hash_bound_latest_pointer(tmp_path, capsys):
    dataset, frozen, latest = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    plan = prepare_reader_training(dataset, latest, tmp_path)

    assert plan["trainable"] is True
    assert plan["classes"] == ["maximum_speed:30", "maximum_speed:50"]
    assert plan["split_path"] == str(frozen)
    assert plan["requested_split_path"] == str(latest)
    assert plan["split_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert plan["counts_by_split"]["train"] == 2
    assert plan["counts_by_split"]["validation"] == 2
    assert "torch" not in sys.modules


def test_training_plans_load_ai_reviewed_labels(tmp_path, capsys):
    dataset, frozen, latest = _training_fixture(tmp_path, capsys, ai=True)
    rows = load_jsonl(dataset)
    assert {r["provenance"]["label_basis"] for r in rows} == {"ai_review", "human_review"}

    reader = prepare_reader_training(dataset, latest, tmp_path)
    assert reader["trainable"] is True
    assert reader["classes"] == ["maximum_speed:30", "maximum_speed:50"]
    assert reader["counts_by_split"]["train"] == 2 and reader["counts_by_split"]["validation"] == 2
    ai_ids = {r["annotation_id"] for r in rows if r["provenance"]["label_basis"] == "ai_review"}
    assert len(ai_ids & {r["annotation_id"] for r in reader["records"]}) >= 3
    detector = prepare_detector_training(dataset, latest, tmp_path)
    assert detector["trainable"] is True
    detector_ids = set()
    for frame in detector["frames"]:
        detector_ids.update(frame["annotation_ids"])
    assert ai_ids & detector_ids
    assert "torch" not in sys.modules
    code, out = invoke(capsys, "build-splits", "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out


def test_train_dry_run_uses_latest_pointer_without_gpu_stack(tmp_path, capsys):
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    code, out = invoke(
        capsys,
        "train",
        str(dataset),
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )

    assert code == 0, out.err + out.out
    summary = json.loads(out.out)
    assert summary["trainable"] is True
    assert summary["record_count"] >= 4
    assert "records" not in summary
    assert "torch" not in sys.modules


def test_latest_pointer_hash_mismatch_is_rejected(tmp_path, capsys):
    dataset, _, latest = _training_fixture(tmp_path, capsys)
    pointer = json.loads(latest.read_text())
    pointer["sha256"] = "0" * 64
    latest.write_text(json.dumps(pointer) + "\n")

    try:
        prepare_reader_training(dataset, latest, tmp_path)
    except ValueError as exc:
        assert "split_pointer_hash_mismatch" in str(exc)
    else:
        raise AssertionError("tampered split pointer was accepted")



def test_off_displays_are_ignore_boxes_of_the_detector_plan(tmp_path, capsys):
    from nnslr_tools.training import off_display
    off = {"sign_family": "variable_display", "value_kph": None, "notes": "LED panel on the gantry is off/blank"}
    assert off_display(off) and off_display({**off, "notes": "pannello variabile spento: nessun valore"})
    assert not off_display({**off, "notes": "lit led display cut off by the top edge, number not readable"})
    assert not off_display({**off, "notes": "far led display, number not readable"})
    assert not off_display({**off, "value_kph": 80, "notes": "off"})
    assert not off_display({**off, "sign_family": "maximum_speed"})
    rows = []
    for index in range(6):
        row = reviewed(tmp_path, f"{index + 1:08x}--abcdef1234", index=index)
        if index < 2:
            # Two variable displays without a value: one off (an ignore box), one lit but too far (a sign to find).
            row["detections"][0].update(family="variable_display", value_kph=None, value_state="unknown",
                                        notes="led panel on the gantry, off/blank" if index == 0 else "lit led panel, too far")
        rows.append(row)
    source = write(tmp_path, rows)
    code, out = invoke(capsys, "import-annotations", str(source), "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out
    code, out = invoke(capsys, "build-splits", "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out
    plan = prepare_detector_training(tmp_path / "annotations" / "objects.jsonl", tmp_path / "splits" / "latest.json", tmp_path)
    by_route = {frame["route_id"][:8]: frame for frame in plan["frames"]}
    # The frame with the off display alone stays in the plan: background everywhere but on the panel.
    assert by_route["00000001"]["boxes"] == [] and by_route["00000001"]["ignore_boxes"] == [[1, 2, 15, 20]]
    assert by_route["00000002"]["boxes"] == [[1, 2, 15, 20]] and by_route["00000002"]["box_families"] == ["variable_display"]
    assert sum(plan["ignore_boxes_by_split"].values()) == 1 and sum(plan["boxes_by_split"].values()) == 5
    assert all(frame["ignore_boxes"] == [] for route, frame in by_route.items() if route != "00000001")


def test_a_road_user_is_an_ignore_box_of_the_detector_plan_and_a_class_of_the_classifier(tmp_path, capsys):
    from nnslr_tools.sign_types import ROAD_USERS, is_road_user
    from nnslr_tools.training import ELIGIBLE_REVIEW_STATES, _classifier_label
    assert 'road_user_person' in ROAD_USERS and 'road_user_car' in ROAD_USERS and 'give_way' not in ROAD_USERS
    assert is_road_user({"sign_family": "other_sign", "sign_type": "road_user_car"})
    assert not is_road_user({"sign_family": "other_sign", "sign_type": "give_way"})
    assert not is_road_user({"sign_family": "not_a_sign", "sign_type": "road_user_car"})
    rows = []
    for index in range(6):
        row = reviewed(tmp_path, f"{index + 1:08x}--abcdef1234", index=index)
        if index == 0:  # a car, boxed by the reviewer
            row["detections"][0].update(family="other_sign", sign_type="road_user_car", value_kph=None, value_state="unknown")
        rows.append(row)
    source = write(tmp_path, rows)
    code, out = invoke(capsys, "import-annotations", str(source), "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out
    code, out = invoke(capsys, "build-splits", "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out
    plan = prepare_detector_training(tmp_path / "annotations" / "objects.jsonl", tmp_path / "splits" / "latest.json", tmp_path)
    by_route = {frame["route_id"][:8]: frame for frame in plan["frames"]}
    # The cutter does not learn road users yet: neither a sign to find nor background.
    assert by_route["00000001"]["boxes"] == [] and by_route["00000001"]["ignore_boxes"] == [[1, 2, 15, 20]]
    assert sum(plan["ignore_boxes_by_split"].values()) == 1 and sum(plan["boxes_by_split"].values()) == 5
    # The classifier names it as any other kind.
    car = {"sign_family": "other_sign", "sign_type": "road_user_car", "bbox_xyxy": [1, 2, 15, 20],
           "review_state": sorted(ELIGIBLE_REVIEW_STATES)[0]}
    assert _classifier_label(car) == "other_sign:road_user_car"


def test_a_frame_ticked_road_users_complete_says_so_in_the_training_frames(tmp_path, capsys):
    """"Road users complete" (the tick of Sign review): every frame export says it, false on every other frame; the
    cutter's plan is otherwise the same (the road user an ignore box)."""
    from nnslr_tools.training import reviewed_frames
    rows = []
    for index in range(6):
        row = reviewed(tmp_path, f"{index + 1:08x}--abcdef1234", index=index, negative=index == 1)
        if index == 0:  # a car boxed, every road user of the frame boxed
            row["detections"][0].update(family="other_sign", sign_type="road_user_car", value_kph=None, value_state="unknown")
            row["users_complete"] = True
        if index == 1:  # no box at all, ticked: no road user on it
            row["users_complete"] = True
        rows.append(row)
    source = write(tmp_path, rows)
    code, out = invoke(capsys, "import-annotations", str(source), "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out
    code, out = invoke(capsys, "build-splits", "--data-root", str(tmp_path))
    assert code == 0, out.err + out.out
    dataset = load_jsonl(tmp_path / "annotations" / "objects.jsonl")
    assert {row["route_id"][:8] for row in dataset if row.get("users_complete") is True} == {"00000001", "00000002"}
    assert all("users_complete" not in row for row in dataset if row["route_id"][:8] not in ("00000001", "00000002"))
    plan = prepare_detector_training(tmp_path / "annotations" / "objects.jsonl", tmp_path / "splits" / "latest.json", tmp_path)
    by_route = {frame["route_id"][:8]: frame for frame in plan["frames"]}
    assert by_route["00000001"]["users_complete"] is True and by_route["00000001"]["ignore_boxes"] == [[1, 2, 15, 20]]
    assert by_route["00000001"]["boxes"] == []
    assert by_route["00000002"]["users_complete"] is True and by_route["00000002"]["explicit_negative"] is True
    assert all(frame["users_complete"] is False for route, frame in by_route.items() if route not in ("00000001", "00000002"))
    frames = reviewed_frames(dataset, {row["annotation_id"]: "train" for row in dataset})
    assert {frame["route_id"][:8] for frame in frames if frame["users_complete"]} == {"00000001", "00000002"}
    assert sum(1 for frame in frames if frame["users_complete"] is False) == 4


def test_prepare_detector_training_groups_full_frames_without_torch(tmp_path, capsys):
    dataset, frozen, latest = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    plan = prepare_detector_training(dataset, latest, tmp_path)

    assert plan["trainable"] is True
    assert plan["classes"] == ["speed_sign"]
    assert plan["split_path"] == str(frozen)
    assert plan["boxes_by_split"]["train"] >= 2
    assert plan["boxes_by_split"]["validation"] >= 2
    assert plan["negative_frames_by_split"]["hard-negative"] == 1
    assert all(frame["image_sha256"] for frame in plan["frames"])
    assert "torch" not in sys.modules


def test_detector_train_dry_run_is_cpu_only(tmp_path, capsys):
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    assert "torch" not in sys.modules

    code, out = invoke(
        capsys,
        "train",
        str(dataset),
        "--task",
        "detector",
        "--data-root",
        str(tmp_path),
        "--dry-run",
    )

    assert code == 0, out.err + out.out
    summary = json.loads(out.out)
    assert summary["task"] == "detector"
    assert summary["trainable"] is True
    assert summary["record_count"] >= 5
    assert "frames" not in summary
    assert "torch" not in sys.modules


def test_training_plans_carry_who_confirmed_each_label(tmp_path, capsys):
    dataset, _, latest = _training_fixture(tmp_path, capsys, ai=True)
    rows = {r["annotation_id"]: r["provenance"]["label_basis"] for r in load_jsonl(dataset)}
    reader = prepare_reader_training(dataset, latest, tmp_path)
    for record in reader["records"]:
        assert record["label_basis"] == rows[record["annotation_id"]]
    detector = prepare_detector_training(dataset, latest, tmp_path)
    for frame in detector["frames"]:
        assert {rows[i] for i in frame["annotation_ids"]} == {frame["label_basis"]}
    assert "human_review" in {r["label_basis"] for r in reader["records"]}


def test_the_mapillary_windows_join_train_and_the_plan_stays_ours(tmp_path, capsys):
    from test_mtsd_windows import _windows_file
    dataset, _, latest = _training_fixture(tmp_path, capsys)
    base = prepare_detector_training(dataset, latest, tmp_path)
    windows = _windows_file(tmp_path, {f"k{number:02d}": [1] for number in range(10)})
    plan = prepare_detector_training(dataset, latest, tmp_path, mapillary=windows, mapillary_ratio=None)

    added = [frame for frame in plan["frames"] if frame.get("source") == "mapillary"]
    assert len(added) == 10 and {frame["split"] for frame in added} == {"train"}
    assert [frame for frame in plan["frames"] if frame.get("source") != "mapillary"] == base["frames"]
    # The digests and the counts stay ours: remeasure-signnet and the comparisons of Models read them.
    for key in ("dataset_sha256", "split_sha256", "counts_by_split", "boxes_by_split", "boxes_by_family"):
        assert plan[key] == base[key], key
    ours = [frame for frame in base["frames"] if frame["split"] == "train"]
    assert base["mapillary"] is None and base["mapillary_only"] is False
    assert base["trained_on"] == {"ours": len(ours), "ours_boxes": sum(len(f["boxes"]) for f in ours),
                                  "mapillary_windows": 0, "mapillary_boxes": 0}
    assert plan["trained_on"] == {**base["trained_on"], "mapillary_windows": 10, "mapillary_boxes": 10}
    assert plan["mapillary"]["boxes_by_family"] == {"maximum_speed": 10} and plan["mapillary"]["ratio"] is None
    assert plan["mapillary"]["ours_train_speed_boxes"] == len(ours) == 2


def test_the_ratio_caps_the_mapillary_speed_boxes_at_a_part_of_ours(tmp_path, capsys):
    from test_mtsd_windows import _windows_file
    dataset, _, latest = _training_fixture(tmp_path, capsys)
    windows = _windows_file(tmp_path, {f"k{number:02d}": [1] for number in range(10)})

    def taken(**kwargs):
        plan = prepare_detector_training(dataset, latest, tmp_path, mapillary=windows, **kwargs)
        return {frame["window_id"] for frame in plan["frames"] if frame.get("source") == "mapillary"}, plan

    # Our train holds 2 speed boxes: as many Mapillary ones by default, a part of them at 0.5, three times at 3.
    one, plan = taken()
    half, _ = taken(mapillary_ratio=0.5)
    three, _ = taken(mapillary_ratio=3)
    assert len(half) == 1 and len(one) == 2 and len(three) == 6 and half < one < three
    assert plan["mapillary"]["ratio"] == 1.0 and plan["mapillary"]["max_speed_boxes"] == 2
    assert plan["mapillary"]["before_cap"] == {"windows": 10, "speed_boxes": 10}
    with pytest.raises(ValueError, match="mapillary-ratio"):
        taken(mapillary_ratio=0)


def test_the_mapillary_stage_trains_on_the_windows_alone(tmp_path, capsys):
    from test_mtsd_windows import _windows_file
    dataset, _, latest = _training_fixture(tmp_path, capsys)
    base = prepare_detector_training(dataset, latest, tmp_path)
    windows = _windows_file(tmp_path, {f"k{number:02d}": [1] for number in range(10)})
    plan = prepare_detector_training(dataset, latest, tmp_path, mapillary=windows, mapillary_only=True)

    train = [frame for frame in plan["frames"] if frame["split"] == "train"]
    assert len(train) == 10 and all(frame["source"] == "mapillary" for frame in train)
    assert ([frame for frame in plan["frames"] if frame["split"] != "train"]
            == [frame for frame in base["frames"] if frame["split"] != "train"])
    assert plan["trainable"] is True and plan["mapillary_only"] is True and plan["mapillary"]["ratio"] is None
    assert plan["trained_on"] == {"ours": 0, "ours_boxes": 0, "mapillary_windows": 10, "mapillary_boxes": 10}
    assert plan["counts_by_split"] == base["counts_by_split"] and plan["split_sha256"] == base["split_sha256"]


def test_the_mapillary_choices_need_the_windows(tmp_path, capsys):
    dataset, _, latest = _training_fixture(tmp_path, capsys)
    with pytest.raises(ValueError, match="add --mapillary"):
        prepare_detector_training(dataset, latest, tmp_path, mapillary_region="europe")
    with pytest.raises(ValueError, match="add --mapillary"):
        prepare_detector_training(dataset, latest, tmp_path, mapillary_share=50)
    with pytest.raises(ValueError, match="add --mapillary"):
        prepare_detector_training(dataset, latest, tmp_path, mapillary_only=True)
    with pytest.raises(ValueError, match="mapillary_share"):
        prepare_detector_training(dataset, latest, tmp_path, mapillary_share=0)


def test_the_signnet_dry_run_says_what_train_holds(tmp_path, capsys):
    from test_mtsd_windows import _windows_file
    dataset, _, _ = _training_fixture(tmp_path, capsys)
    windows = _windows_file(tmp_path, {f"k{number:02d}": [1] for number in range(10)})

    def signnet(*options, batch="2"):
        return invoke(capsys, "train-signnet", "--run-dir", str(tmp_path / "run"), "--data-root", str(tmp_path),
                      "--dry-run", "--batch", batch, *options)

    code, out = signnet("--mapillary", str(windows), "--mapillary-ratio", "2")
    assert code == 0, out.err + out.out
    summary = json.loads(out.out)
    assert summary["trained_on"]["mapillary_windows"] == 4 and summary["mapillary"]["frames"] == 4
    assert "and 4 Mapillary windows (4 speed boxes, at most 2 times ours: 10 before the cap)" in out.err
    code, out = signnet("--mapillary", str(windows), "--mapillary-only", "--mapillary-ratio", "2")
    assert code == 2 and "not with --mapillary-only" in out.err
    code, out = signnet("--mapillary-ratio", "2")
    assert code == 2 and "add --mapillary" in out.err
    code, out = signnet("--mapillary", str(windows), "--mapillary-only")
    assert code == 0 and ("train holds 0 of our frames (0 boxes, 0 of the speed signs, unreadable, zone and road "
                          "marking included) and 10 Mapillary") in out.err
    assert json.loads(out.out)["train_frames"] == 10
    # Fewer windows than one batch would train nothing (the loader drops a last smaller batch): stopped here.
    code, out = signnet("--mapillary", str(windows), "--mapillary-only", batch="16")
    assert code == 1 and json.loads(out.out)["trainable"] is False and "fewer than one batch of 16" in out.err
    # The windows are the car's band: a band of other rows would cut them.
    code, out = signnet("--mapillary", str(windows), "--band-width", "1344")
    assert code == 2 and "the windows are the car's band" in out.err
    assert "torch" not in sys.modules


def test_the_signnet_training_reads_the_plan_of_its_check(tmp_path, capsys, monkeypatch):
    """#70: the check of the population writes its plan (--dry-run --plan-out) and the training reads it (--plan)
    instead of building it again; with another setting, or no file, the training builds its own."""
    from nnslr_tools import signnet, training
    _training_fixture(tmp_path, capsys)
    saved = tmp_path / "plans" / "main.json"

    def signnet_cli(*options):
        return invoke(capsys, "train-signnet", "--run-dir", str(tmp_path / "run"), "--data-root", str(tmp_path),
                      "--batch", "2", *options)

    code, out = signnet_cli("--dry-run", "--plan-out", str(saved))
    assert code == 0 and saved.is_file() and not (tmp_path / "run").exists()
    code, out = signnet_cli("--plan-out", str(saved))
    assert code == 2 and "add --dry-run" in out.err
    built, trained = [], []
    real = training.prepare_detector_training
    monkeypatch.setattr(training, "prepare_detector_training", lambda *a, **kw: built.append(1) or real(*a, **kw))
    measured = {"recall_iou": 0.5, "false_positives_per_image": 1.0}
    result = {"best_epoch": 1, "parameters": 1,
              **{split: {"by_box_budget": {"5": measured}, "car": measured} for split in ("validation", "test")}}
    monkeypatch.setattr(signnet, "train_signnet", lambda plan, *a, **kw: trained.append(plan) or result)
    code, out = signnet_cli("--plan", str(saved))
    assert code == 0 and built == [] and "written by the check of the population" in out.err
    assert trained[0] == json.loads(saved.read_text(encoding="utf-8"))["plan"]
    assert trained[0] == real(tmp_path / "annotations" / "objects.jsonl", tmp_path / "splits" / "latest.json",
                              tmp_path)
    code, out = signnet_cli("--plan", str(saved), "--families", "speed")
    assert code == 0 and built == [1] and "built again" in out.err and trained[-1]["families"] == "speed"
    saved.unlink()
    code, out = signnet_cli("--plan", str(saved))
    assert code == 0 and built == [1, 1] and "built again" in out.err and trained[-1] == trained[0]
