"""The signs of the Mapillary Traffic Sign Dataset in the classifier's train: which MTSD labels are ours, the crops
cut from a tiny MTSD-like folder, the cap per label, and the plan of a v2 classifier with them."""
from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

import pytest
from PIL import Image

from nnslr_tools import mapillary
from nnslr_tools.mapillary import (CROPS_FILE, crops_state, make_crops, our_label, read_crops,
                                   training_records)
from nnslr_tools.training import TrainConfig, prepare_reader_training, resume_arguments
from test_classifier import _classifier_fixture
from test_detector_sizes import torch_python  # noqa: F401 (fixture)


def test_the_mtsd_labels_that_are_ours():
    assert our_label("regulatory--maximum-speed-limit-50--g1") == "maximum_speed:50"
    assert our_label("regulatory--maximum-speed-limit-120--g1") == "maximum_speed:120"
    # Any design and any written value is a limit; a plate under a sign is not.
    assert our_label("regulatory--maximum-speed-limit-45--g3") == "maximum_speed:45"
    assert our_label("regulatory--maximum-speed-limit-5--g1") == "maximum_speed:5"
    assert our_label("complementary--maximum-speed-limit-25--g1") is None
    assert our_label("regulatory--yield--g1") == "other_sign:give_way"
    assert our_label("information--parking--g1") is None


def _sign(key, label, box, **properties):
    return {"key": key, "label": label, "properties": properties,
            "bbox": dict(zip(("xmin", "ymin", "xmax", "ymax"), box))}


def _mtsd(folder: Path) -> tuple[Path, Path]:
    """Three photos: two speed signs and a give way (one occluded, one too small, one ambiguous), one photo
    without a sign of ours, one annotation whose photo is not downloaded."""
    annotations, images = folder / "annotations", folder / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    documents = {
        "p1": [_sign("s50", "regulatory--maximum-speed-limit-50--g1", [100, 40, 300, 240]),
               _sign("tiny", "regulatory--maximum-speed-limit-80--g1", [10, 10, 20, 20]),
               _sign("yield", "regulatory--yield--g1", [320, 20, 360, 60], occluded=True)],
        "p2": [_sign("s80", "regulatory--maximum-speed-limit-80--g1", [0, 0, 50, 50]),
               _sign("unsure", "regulatory--maximum-speed-limit-30--g1", [60, 0, 110, 50], ambiguous=True),
               _sign("parking", "information--parking--g1", [200, 0, 250, 50])],
        "p3": [_sign("park", "information--parking--g1", [0, 0, 40, 40])],
        "p4": [_sign("away", "regulatory--maximum-speed-limit-60--g1", [0, 0, 40, 40])],
    }
    for key, objects in documents.items():
        (annotations / f"{key}.json").write_text(json.dumps({"width": 400, "height": 300, "ispano": False,
                                                             "objects": objects}))
        if key != "p4":
            Image.new("RGB", (400, 300), (90, 120, 160)).save(images / f"{key}.jpg")
    return annotations, images


def test_the_crops_of_a_tiny_mtsd(tmp_path):
    annotations, images = _mtsd(tmp_path / "mtsd")
    root = tmp_path / "data"
    assert crops_state(root) == {"available": False, "crops": 0, "labels": 0,
                                 "reason": "Mapillary is not prepared (helpers/mapillary.sh)"}
    header = make_crops(annotations, images, root, log=lambda *_: None)
    assert header["by_label"] == {"maximum_speed:50": 1, "maximum_speed:80": 1, "other_sign:give_way": 1}
    assert header["skipped"] == {"ambiguous": 1, "not_ours": 2, "photo_not_downloaded": 1, "too_small": 1}
    _, rows = read_crops(root / CROPS_FILE)
    by_object = {row["mtsd_object"]: row for row in rows}
    # The 200 px sign is cut with half its side around it and scaled down to MAX_SIGN_SIDE.
    big = by_object["s50"]
    with Image.open(root / big["image_path"]) as crop:
        assert crop.size == (192, 144)  # 300 px of the photo, its whole height
    assert big["bbox_xyxy"] == [48.0, 19.2, 144.0, 115.2] and not big["occluded"]
    # A sign on the photo's border keeps its pixels: the region stops at the border.
    assert by_object["s80"]["bbox_xyxy"] == [0.0, 0.0, 50.0, 50.0] and by_object["yield"]["occluded"]
    assert crops_state(root) == {"available": True, "crops": 3, "labels": 3, "reason": ""}
    with pytest.raises(ValueError, match="no photos folder"):
        make_crops(annotations, tmp_path / "nowhere", root)


def test_the_crops_straight_from_the_zips_and_a_resume(tmp_path):
    """MTSD as downloaded: the annotations in one zip, the photos in another, a third still being copied."""
    annotations, images = _mtsd(tmp_path / "mtsd")
    share = tmp_path / "share" / "zips"
    share.mkdir(parents=True)
    with zipfile.ZipFile(share / "mtsd_v2_fully_annotated_annotation.zip", "w") as archive:
        for path in annotations.iterdir():
            archive.write(path, f"mtsd_v2_fully_annotated/annotations/{path.name}")
        archive.writestr("mtsd_v2_fully_annotated/splits/train.txt", "p1\n")
    with zipfile.ZipFile(share / "mtsd_fully_annotated_images.train.0.zip", "w") as archive:
        for path in images.iterdir():
            archive.write(path, f"images/{path.name}")
            archive.writestr(f"__MACOSX/images/._{path.name}", b"junk")
    (share / "mtsd_fully_annotated_images.train.1.zip").write_bytes(b"PK\x03\x04 half copied")
    root = tmp_path / "data"
    lines = []
    header = make_crops(share.parent, share.parent, root, log=lines.append)
    assert header["by_label"] == {"maximum_speed:50": 1, "maximum_speed:80": 1, "other_sign:give_way": 1}
    assert header["skipped"] == {"ambiguous": 1, "not_ours": 2, "photo_not_downloaded": 1, "too_small": 1}
    assert header["zips_left_out"] == [str(share / "mtsd_fully_annotated_images.train.1.zip")]
    first = (root / CROPS_FILE).read_text()
    # Run again with photos that do not decode any more: none is read, the crops already cut are kept.
    with zipfile.ZipFile(share / "mtsd_fully_annotated_images.train.0.zip", "w") as archive:
        for path in images.iterdir():
            archive.writestr(f"images/{path.name}", b"not a photo any more")
    lines.clear()
    assert make_crops(share.parent, share.parent, root, log=lines.append)["crops"] == 3
    assert (root / CROPS_FILE).read_text() == first and "mapillary: 3 crops already cut, kept" in lines


def _crops_file(root: Path, rows: list[dict]) -> Path:
    path = root / CROPS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"kind": "mapillary_crops", "version": 1, "crops": len(rows), "by_label": {}}
    path.write_text("\n".join(json.dumps(item) for item in [header] + rows) + "\n")
    return path


def test_only_the_european_design_says_kilometres(tmp_path):
    rows = [{"image_path": "c/a.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:45", "mtsd_object": "a",
             "design": "g3"},
            {"image_path": "c/b.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:50", "mtsd_object": "b",
             "design": "g1"}]
    records, _ = training_records(_crops_file(tmp_path, rows), set(), set())
    by_label = {r["label"]: r for r in records}
    assert (by_label["maximum_speed:45"]["sign_value"], by_label["maximum_speed:45"]["value_kph"]) == (45, None)
    assert (by_label["maximum_speed:50"]["sign_value"], by_label["maximum_speed:50"]["value_kph"]) == (50, 50)


def test_the_train_records_keep_our_kinds_and_a_cap_per_label(tmp_path, monkeypatch):
    monkeypatch.setattr(mapillary, "PER_LABEL", 2)
    rows = ([{"image_path": f"c/{n}.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:120", "mtsd_object": f"a{n}"}
             for n in range(3)]
            + [{"image_path": "c/g.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "other_sign:give_way", "mtsd_object": "g"},
               {"image_path": "c/s.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "other_sign:stop", "mtsd_object": "s"}])
    path = _crops_file(tmp_path, rows)
    records, summary = training_records(path, {"maximum_speed:50", "other_sign:give_way"}, {"other_sign:give_way"})
    assert summary["by_label"] == {"maximum_speed:120": 2, "other_sign:give_way": 1}
    assert summary["left_out"] == {"kind_not_a_class_of_ours": 1, "over_the_per_label_limit": 1}
    assert summary["new_classes"] == ["maximum_speed:120"] and summary["per_label"] == 2
    speed = next(r for r in records if r["label"] == "maximum_speed:120")
    assert (speed["split"], speed["sign_family"], speed["value_kph"], speed["label_basis"]) == (
        "train", "maximum_speed", 120, "mapillary")
    give_way = next(r for r in records if r["label"] == "other_sign:give_way")
    assert (give_way["sign_type"], give_way["value_kph"]) == ("give_way", None)
    # The same file gives the same records: the cap keeps a fixed choice.
    assert training_records(path, set(), {"other_sign:give_way"})[0] == records


def test_the_classifier_plan_with_mapillary(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    plain = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2")
    Image.new("RGB", (40, 40)).save(tmp_path / "s.jpg")
    path = _crops_file(tmp_path, [
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "maximum_speed:120", "mtsd_object": "m"},
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "other_sign:roundabout", "mtsd_object": "r"}])
    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path)
    assert plan["classes"] == sorted(plain["classes"] + ["maximum_speed:120"])
    assert plan["mapillary"]["by_label"] == {"maximum_speed:120": 1}
    assert plan["mapillary"]["left_out"] == {"kind_not_a_class_of_ours": 1, "over_the_per_label_limit": 0}
    assert plan["mapillary_sha256"] == plan["mapillary"]["sha256"] and plain["mapillary_sha256"] is None
    assert any(item.startswith("mapillary_crops_in_train") for item in plan["limitations"])
    # Validation and test are our drives only.
    assert {r["split"] for r in plan["records"] if r["label_basis"] == "mapillary"} == {"train"}
    assert plan["counts_by_split"]["validation"] == plain["counts_by_split"]["validation"]
    with pytest.raises(ValueError, match="--classes v2"):
        prepare_reader_training(dataset, latest, tmp_path, mapillary_path=path)
    (tmp_path / "s.jpg").unlink()
    with pytest.raises(ValueError, match="missing_image: s.jpg"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path)


def test_a_classifier_with_mapillary_resumes_with_it():
    config = TrainConfig(class_scheme="v2", augment=True, mapillary="/data/external/mapillary/crops.jsonl").__dict__
    flags = resume_arguments({"task": "reader", "config": config})
    assert flags[flags.index("--mapillary") + 1] == "/data/external/mapillary/crops.jsonl"
    assert "--mapillary" not in resume_arguments({"task": "reader", "config": TrainConfig(class_scheme="v2").__dict__})


# --- The region and the share of the crops, and where the photos were taken ------------------------------------------

def _locations(root: Path, lines: list[dict]) -> Path:
    path = root / mapillary.LOCATIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _placed(key, country, continent, status="located"):
    there = status == "located"
    return {"mtsd_image": key, "image_id": "1", "latitude": 47.0 if there else None,
            "longitude": 8.0 if there else None, "country": country, "continent": continent, "status": status}


def test_a_region_and_a_share_have_one_spelling():
    assert mapillary.parse_region(None) is None and mapillary.parse_region(" all ") is None
    assert mapillary.region_text("ch, Europe ,de") == "europe,CH,DE"
    assert mapillary.region_text("") == "all"
    for wrong in ("Atlantis", "CHE", "eu"):
        with pytest.raises(ValueError, match="mapillary_region"):
            mapillary.parse_region(wrong)
    assert mapillary.check_share(None) == 100.0 and mapillary.check_share("5") == 5.0
    for wrong in ("0", "-1", "101", "five"):
        with pytest.raises(ValueError, match="mapillary_share"):
            mapillary.check_share(wrong)


def test_the_crops_are_chosen_by_region_then_share_then_cap(tmp_path, monkeypatch):
    rows = [{"image_path": f"c/{n}.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:50",
             "mtsd_image": ("zurich", "paris", "boston", "gone")[n % 4], "mtsd_object": f"o{n}"} for n in range(400)]
    path = _crops_file(tmp_path, rows)
    # A region needs the positions: no file, no guess.
    with pytest.raises(ValueError, match="mapillary_locations_missing"):
        training_records(path, set(), set(), region="europe")
    _locations(tmp_path, [_placed("zurich", "CH", "europe"), _placed("paris", "FR", "europe"),
                          _placed("boston", "US", "north_america"), _placed("gone", None, None, "no_v4_id")])
    records, summary = training_records(path, set(), set(), region="europe")
    assert {r["image_path"] for r in records} <= {f"c/{n}.jpg" for n in range(400) if n % 4 in (0, 1)}
    assert summary["left_out"]["outside_the_region"] == 100 and summary["left_out"]["photo_without_position"] == 100
    assert (summary["region"], summary["share"], summary["in_region"]) == ("europe", 100.0, 200)
    assert summary["locations"]["located_photos"] == 3
    assert training_records(path, set(), set(), region="CH,US")[1]["in_region"] == 200
    # "all" takes every crop, also the ones without a position, and reads no positions.
    everything = training_records(path, set(), set())[1]
    assert everything["in_region"] == 400 and everything["locations"] is None
    # The share: a fixed part, the same crops every time, a larger share takes them all and more.
    ten = {r["mtsd_object"] for r in training_records(path, set(), set(), share=10)[0]}
    twenty = {r["mtsd_object"] for r in training_records(path, set(), set(), share=20)[0]}
    assert 20 <= len(ten) <= 60 and ten < twenty
    assert ten == {r["mtsd_object"] for r in training_records(path, set(), set(), share=10)[0]}
    # The cap comes last: it keeps PER_LABEL of what the region and the share chose.
    monkeypatch.setattr(mapillary, "PER_LABEL", 5)
    capped, summary = training_records(path, set(), set(), region="europe", share=50)
    chosen = {r["mtsd_object"] for r in training_records(path, set(), set(), region="europe", share=50)[0]}
    assert len(capped) == 5 and {r["mtsd_object"] for r in capped} <= chosen
    assert summary["left_out"]["over_the_per_label_limit"] > 0 and summary["left_out"]["outside_the_share"] > 0
    assert summary["selection_sha256"] != training_records(path, set(), set(), region="CH", share=50)[1]["selection_sha256"]


def test_the_classifier_plan_records_the_region_and_the_share(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    Image.new("RGB", (40, 40)).save(tmp_path / "s.jpg")
    path = _crops_file(tmp_path, [
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "maximum_speed:120", "mtsd_image": "zurich",
         "mtsd_object": "m"},
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "maximum_speed:130", "mtsd_image": "boston",
         "mtsd_object": "n"}])
    _locations(tmp_path, [_placed("zurich", "CH", "europe"), _placed("boston", "US", "north_america")])
    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path,
                                   mapillary_region="CH")
    assert plan["mapillary"]["by_label"] == {"maximum_speed:120": 1} and plan["mapillary"]["region"] == "CH"
    assert plan["mapillary_selection_sha256"] == plan["mapillary"]["selection_sha256"]
    every = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path)
    assert every["mapillary_selection_sha256"] != plan["mapillary_selection_sha256"]
    with pytest.raises(ValueError, match="add --mapillary"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_region="europe")


def test_a_classifier_resumes_with_its_region_and_share():
    config = TrainConfig(class_scheme="v2", augment=True, mapillary="/m/crops.jsonl", mapillary_region="europe,CH",
                         mapillary_share=5.0).__dict__
    flags = resume_arguments({"task": "reader", "config": config})
    assert flags[flags.index("--mapillary-region") + 1] == "europe,CH"
    assert flags[flags.index("--mapillary-share") + 1] == "5"
    plain = resume_arguments({"task": "reader", "config": TrainConfig(class_scheme="v2", mapillary="/m").__dict__})
    assert "--mapillary-region" not in plain and "--mapillary-share" not in plain
    # A run saved before the region existed resumes as "all", 100 %.
    from nnslr_tools.training import READER_CONFIG_BEFORE_SETTINGS
    assert (READER_CONFIG_BEFORE_SETTINGS["mapillary_region"], READER_CONFIG_BEFORE_SETTINGS["mapillary_share"]) == (
        TrainConfig().mapillary_region, TrainConfig().mapillary_share)


def test_the_train_command_takes_a_region_only_with_mapillary(tmp_path, capsys):
    from nnslr_tools.cli import main
    assert main(["train", "--task", "reader", "--classes", "v2", "--mapillary-region", "europe", "--dry-run",
                 "--data-root", str(tmp_path)]) != 0
    assert "add --mapillary" in capsys.readouterr().err


class _FakeGraph:
    """Mapillary's Graph API: 111 placed in Zurich, 444 without a position, 222 deleted (a batch with it is
    refused as a whole), and what each request carried."""

    def __init__(self, answers=None):
        self.urls, self.headers = [], []
        self.answers = list(answers or [])

    def __call__(self, url, headers):
        self.urls.append(url)
        self.headers.append(headers)
        if self.answers:
            return self.answers.pop(0)
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(url)
        images = {"111": {"id": "111", "computed_geometry": {"type": "Point", "coordinates": [8.5417, 47.3769]}},
                  "444": {"id": "444"},
                  "555": {"id": "555", "geometry": {"type": "Point", "coordinates": [-74.006, 40.7128]}}}
        if parsed.path.strip("/"):
            item = images.get(parsed.path.strip("/"))
            return (200, json.dumps(item).encode()) if item else (404, b'{"error": {"message": "does not exist"}}')
        ids = parse_qs(parsed.query)["ids"][0].split(",")
        if any(image_id not in images for image_id in ids):
            return 400, b'{"error": {"message": "Unsupported get request"}}'
        return 200, json.dumps({image_id: images[image_id] for image_id in ids}).encode()


def _ids_table(root: Path, rows: dict[str, str]) -> Path:
    path = root / mapillary.IMAGE_IDS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("image_key,image_id\n" + "".join(f"{key},{value}\n" for key, value in rows.items()))
    return path


def test_the_positions_are_fetched_in_batches_and_resume(tmp_path):
    ids = _ids_table(tmp_path, {"kZurich": "111", "kGone": "222", "kOld": "kOld", "kBlind": "444"})
    assert mapillary.read_image_ids(ids) == {"kZurich": "111", "kGone": "222", "kOld": None, "kBlind": "444"}
    graph, waits = _FakeGraph(), []
    summary = mapillary.fetch_locations(tmp_path, token="secret-token", fetch=graph, sleep=waits.append,
                                        clock=lambda: 0.0, log=lambda line: None)
    lines = mapillary.read_locations(tmp_path / mapillary.LOCATIONS_FILE)
    assert {key: line["status"] for key, line in lines.items()} == {
        "kZurich": "located", "kGone": "not_found", "kOld": "no_v4_id", "kBlind": "no_position"}
    zurich = lines["kZurich"]
    assert (zurich["latitude"], zurich["longitude"], zurich["country"], zurich["continent"]) == (
        47.3769, 8.5417, "CH", "europe")
    assert (summary["located"], summary["not_found"], summary["no_v4_id"], summary["no_position"]) == (1, 1, 1, 1)
    # One batch refused as a whole, then one request each; the token only in the header, never in a URL.
    assert len(graph.urls) == 4 and "ids=444%2C222%2C111" in graph.urls[0]
    assert all(h["Authorization"] == "OAuth secret-token" for h in graph.headers)
    assert not any("secret" in url for url in graph.urls)
    # At most REQUESTS_PER_SECOND: the clock stands still, so each request after the first waits.
    assert waits == [1.0 / mapillary.REQUESTS_PER_SECOND] * 3
    # Again: every photo has its line, nothing is asked.
    again = _FakeGraph()
    assert mapillary.fetch_locations(tmp_path, token="t", fetch=again, sleep=lambda s: None,
                                     log=lambda line: None)["todo"] == 0 and again.urls == []
    # The table now resolves the old key: only that photo is asked.
    _ids_table(tmp_path, {"kZurich": "111", "kGone": "222", "kOld": "555", "kBlind": "444"})
    newer = _FakeGraph()
    mapillary.fetch_locations(tmp_path, token="t", fetch=newer, sleep=lambda s: None, log=lambda line: None)
    assert len(newer.urls) == 1
    assert mapillary.read_locations(tmp_path / mapillary.LOCATIONS_FILE)["kOld"]["country"] == "US"
    # --limit asks a handful, --again asks the photos already placed too, the new line winning.
    few = _FakeGraph()
    assert mapillary.fetch_locations(tmp_path, token="t", again=True, limit=1, fetch=few, sleep=lambda s: None,
                                     log=lambda line: None)["todo"] == 1
    assert len(mapillary.read_locations(tmp_path / mapillary.LOCATIONS_FILE)) == 4


def test_a_refused_token_stops_and_a_busy_mapillary_is_waited_for(tmp_path):
    _ids_table(tmp_path, {"kZurich": "111"})
    with pytest.raises(mapillary.LocationsError, match="mapillary_token_refused") as refused:
        mapillary.fetch_locations(tmp_path, token="secret-token", fetch=_FakeGraph([(401, b"{}")]),
                                  sleep=lambda s: None, log=lambda line: None)
    assert "secret" not in str(refused.value)
    assert not (tmp_path / mapillary.LOCATIONS_FILE).exists()
    waits = []
    busy = _FakeGraph([(429, b""), (503, b"")])
    mapillary.fetch_locations(tmp_path, token="t", fetch=busy, sleep=waits.append, clock=lambda: 100.0,
                              log=lambda line: None)
    assert [wait for wait in waits if wait >= 1] == [2.0, 4.0]
    assert mapillary.read_locations(tmp_path / mapillary.LOCATIONS_FILE)["kZurich"]["status"] == "located"
    # Mapillary out of reach: the run stops, nothing is written for the photos it could not ask.
    _ids_table(tmp_path, {"kZurich": "111", "kBlind": "444"})
    down = _FakeGraph([(503, b"")] * (mapillary.RETRIES + 1))
    with pytest.raises(mapillary.LocationsError, match="mapillary_unreachable"):
        mapillary.fetch_locations(tmp_path, token="t", fetch=down, sleep=lambda s: None, log=lambda line: None)
    assert "kBlind" not in mapillary.read_locations(tmp_path / mapillary.LOCATIONS_FILE)


def test_a_write_cut_short_costs_one_line(tmp_path):
    path = _locations(tmp_path, [_placed("a", "CH", "europe")])
    with path.open("a") as handle:
        handle.write('{"mtsd_image": "b", "lat')
    mapillary._append(path, [_placed("c", "FR", "europe")])
    assert set(mapillary.read_locations(path)) == {"a", "c"}


def test_the_regions_offered_count_the_crops_of_each_place(tmp_path, monkeypatch):
    monkeypatch.delenv(mapillary.TOKEN_ENV, raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    state = mapillary.regions_state(tmp_path / "data", repo)
    assert (state["fetched"], state["crops"], state["token"], state["continents"]) == (False, 0, False, [])
    root = tmp_path / "data"
    _crops_file(root, [{"image_path": "c/1.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:50",
                        "mtsd_image": image, "mtsd_object": f"o{n}"}
                       for n, image in enumerate(["zurich", "zurich", "paris", "boston", "gone"])])
    _locations(root, [_placed("zurich", "CH", "europe"), _placed("paris", "FR", "europe"),
                      _placed("boston", "US", "north_america"), _placed("gone", None, None, "no_v4_id")])
    (repo / ".env").write_text("NNSLR_REVIEW_TOKEN=x\nMAPILLARY_TOKEN=\n")
    state = mapillary.regions_state(root, repo)
    assert (state["fetched"], state["crops"], state["located_crops"], state["photos"], state["located_photos"]) == (
        True, 5, 4, 4, 3)
    assert state["continents"] == [{"key": "europe", "name": "Europe", "crops": 3},
                                   {"key": "north_america", "name": "North America", "crops": 1}]
    assert state["countries"][0] == {"code": "CH", "name": "Switzerland", "continent": "europe", "crops": 2}
    assert state["token"] is False
    (repo / ".env").write_text("MAPILLARY_TOKEN='abc'\n")
    assert mapillary.regions_state(root, repo)["token"] is True


def test_a_place_anywhere_has_its_country_and_continent():
    from nnslr_tools.gps import world_place
    assert world_place(47.3769, 8.5417) == ("CH", "europe")
    assert world_place(40.7128, -74.006) == ("US", "north_america")
    assert world_place(35.68, 139.69) == ("JP", "asia")
    assert world_place(-33.87, 151.21) == ("AU", "oceania")
    assert world_place(-23.55, -46.63) == ("BR", "south_america")
    assert world_place(0.0, -30.0) == (None, None)
    assert world_place(None, 8.0) == (None, None)


def test_the_locations_command_counts_without_a_token(tmp_path, monkeypatch, capsys):
    from nnslr_tools.cli import main
    monkeypatch.delenv(mapillary.TOKEN_ENV, raising=False)
    ids = _ids_table(tmp_path, {"kZurich": "111", "kOld": "kOld"})
    assert main(["mapillary-locations", "--dry-run", "--data-root", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["todo"] == 2
    assert main(["mapillary-locations", "--data-root", str(tmp_path), "--ids", str(ids)]) == 2
    assert "MAPILLARY_TOKEN" in capsys.readouterr().err


# --- Two stages: Mapillary alone, then ours from it -------------------------------------------------------------------

def test_the_pretraining_takes_every_crop_and_folds_the_kinds_that_are_not_ours(tmp_path, monkeypatch):
    monkeypatch.setattr(mapillary, "PER_LABEL", 2)
    rows = ([{"image_path": f"c/{n}.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "maximum_speed:120", "mtsd_object": f"a{n}"}
             for n in range(3)]
            + [{"image_path": "c/g.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "other_sign:give_way", "mtsd_object": "g"},
               {"image_path": "c/s.jpg", "bbox_xyxy": [1, 2, 3, 4], "label": "other_sign:stop", "mtsd_object": "s"}])
    path = _crops_file(tmp_path, rows)
    records, summary = training_records(path, {"other_sign:give_way"}, {"other_sign:give_way"}, capped=False,
                                        fold_other_kinds=True)
    assert summary["by_label"] == {"maximum_speed:120": 3, "other_sign": 1, "other_sign:give_way": 1}
    assert summary["per_label"] is None and summary["order"].endswith("no cap")
    assert summary["left_out"] == {"over_the_per_label_limit": 0}
    assert summary["folded_into_other_sign"] == {"other_sign:stop": 1}
    stop = next(r for r in records if r["mtsd_object"] == "s")
    assert (stop["label"], stop["sign_family"], stop["sign_type"]) == ("other_sign", "other_sign", "stop")
    # The capped choice of the mixed run is unchanged.
    assert "folded_into_other_sign" not in training_records(path, set(), {"other_sign:give_way"})[1]


def _two_stage_crops(root: Path) -> Path:
    Image.new("RGB", (40, 40)).save(root / "s.jpg")
    return _crops_file(root, [
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "maximum_speed:120", "mtsd_object": "m"},
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "maximum_speed:50", "mtsd_object": "f"},
        {"image_path": "s.jpg", "bbox_xyxy": [10, 10, 30, 30], "label": "other_sign:roundabout", "mtsd_object": "r"}])


def test_the_plan_of_the_pretraining_on_mapillary_alone(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    plain = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2")
    path = _two_stage_crops(tmp_path)
    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path,
                                   mapillary_only=True)
    # Train is Mapillary alone; the classes are ours and Mapillary's (a kind not ours as other_sign), so that the
    # second stage, on ours, has the same list.
    assert {r["label_basis"] for r in plan["records"] if r["split"] == "train"} == {"mapillary"}
    assert plan["classes"] == sorted(set(plain["classes"]) | {"maximum_speed:120", "other_sign"})
    assert "not_a_sign" in plan["classes"] and plan["trainable"] and plan["mapillary_only"]
    assert plan["mapillary"]["by_label"] == {"maximum_speed:120": 1, "maximum_speed:50": 1, "other_sign": 1}
    # Validation: our sign crops of a class Mapillary has, without the not_a_sign crops.
    validation = [r for r in plan["records"] if r["split"] == "validation"]
    assert validation and {r["label"] for r in validation} <= set(plan["mapillary"]["by_label"])
    assert plan["skipped"]["our_train_crops_left_out_of_the_pretraining"] == plain["counts_by_split"]["train"]
    assert plan["skipped"]["validation_not_a_sign_left_out_of_the_pretraining"] > 0
    assert any(item.startswith("mapillary_only_pretraining") for item in plan["limitations"])
    assert plain["mapillary_only"] is False and plain["start"] is None
    with pytest.raises(ValueError, match="add --mapillary"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_only=True)


def test_the_plan_of_the_second_stage_has_the_classes_of_its_start(tmp_path, capsys):
    dataset = _classifier_fixture(tmp_path, capsys)
    latest = tmp_path / "splits" / "latest.json"
    path = _two_stage_crops(tmp_path)
    first = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", mapillary_path=path,
                                    mapillary_only=True)
    start = {"path": "/runs/r/pretrain/reader-best.pt", "sha256": "ab", "classes": first["classes"],
             "class_scheme": "v2", "split_sha256": first["split_sha256"], "mapillary_only": True}
    plan = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", start=start)
    assert plan["classes"] == first["classes"] and plan["start"] == start
    ours = {r["label"] for r in plan["records"] if r["split"] == "train"}
    assert plan["classes_from_start"] == sorted(set(first["classes"]) - ours) and "maximum_speed:120" not in ours
    assert "maximum_speed:120" in plan["classes_from_start"]
    assert {r["label_basis"] for r in plan["records"] if r["split"] == "train"} != {"mapillary"}
    assert plan["trainable"] and not any(item.startswith("start_of_another_split") for item in plan["limitations"])
    other = prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", start=dict(start, split_sha256="x"))
    assert any(item.startswith("start_of_another_split") for item in other["limitations"])
    with pytest.raises(ValueError, match="--classes v2"):
        prepare_reader_training(dataset, latest, tmp_path, start=start)
    with pytest.raises(ValueError, match="not a v2 classifier"):
        prepare_reader_training(dataset, latest, tmp_path, class_scheme="v2", start=dict(start, class_scheme="v1"))


def test_the_two_stages_resume_with_their_start(tmp_path):
    from test_detector_sizes import _config_of
    first = TrainConfig(class_scheme="v2", augment=True, pretrained=True, mapillary="/m/crops.jsonl",
                        mapillary_only=True).__dict__
    flags = resume_arguments({"task": "reader", "config": first})
    assert "--mapillary-only" in flags and "--init-from" not in flags and _config_of(flags, "reader").__dict__ == first
    start = str(tmp_path / "pretrain" / "reader-best.pt")
    second = TrainConfig(class_scheme="v2", augment=True, init_from=start).__dict__
    flags = resume_arguments({"task": "reader", "config": second})
    assert flags[flags.index("--init-from") + 1] == start and "--pretrained" not in flags
    assert _config_of(flags, "reader").__dict__ == second
    # A run saved before the two stages resumes without them.
    from nnslr_tools.training import READER_CONFIG_BEFORE_SETTINGS
    assert (READER_CONFIG_BEFORE_SETTINGS["mapillary_only"], READER_CONFIG_BEFORE_SETTINGS["init_from"]) == (False, None)


def test_the_train_command_checks_the_two_stages(tmp_path, capsys):
    from nnslr_tools.cli import main
    base = ["train", "--dry-run", "--data-root", str(tmp_path)]
    cases = [
        (["--task", "reader", "--classes", "v2", "--mapillary-only"], "add --mapillary"),
        (["--task", "reader", "--classes", "v2", "--init-from", "x.pt", "--pretrained"], "two different starts"),
        (["--task", "reader", "--init-from", "x.pt"], "--classes v2"),
        (["--task", "detector", "--init-from", "x.pt"], "only with --mode continue"),
        (["--task", "detector", "--mapillary-only"], "classifier option"),
    ]
    for arguments, message in cases:
        assert main(base + arguments) != 0, arguments
        assert message in capsys.readouterr().err, arguments


START_CHECK = r"""
import sys
import torch
from torchvision.models import mobilenet_v3_small
from nnslr_tools.checkpoints import safe_torch_load
from nnslr_tools.training import _load_start
saved = safe_torch_load(sys.argv[1])
assert saved["mapillary_only"] is True and saved["start"] is None and saved["initialization"] == "random"
# Another order of the classes and one the start lacks: every layer copied, the head row by row by class name.
classes = list(reversed(saved["classes"])) + ["maximum_speed:999"]
model = mobilenet_v3_small()
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
fresh = model.state_dict()["classifier.3.weight"][-1].clone()
_load_start(torch, model, saved, classes, saved["input_size"], "classifier", sys.argv[1])
state, weights = model.state_dict(), saved["state_dict"]
assert all(torch.equal(state[key], weights[key]) for key in weights if not key.startswith("classifier.3."))
for row, label in enumerate(classes[:-1]):
    assert torch.equal(state["classifier.3.weight"][row], weights["classifier.3.weight"][saved["classes"].index(label)])
assert torch.equal(state["classifier.3.weight"][-1], fresh)
second = safe_torch_load(sys.argv[2])
assert second["initialization"] == "mapillary_pretraining" and second["classes"] == saved["classes"]
assert second["start"]["path"] == sys.argv[1] and second["start"]["mapillary_only"] is True
print("start ok")
"""


def test_tiny_cpu_two_stages(tmp_path, capsys, torch_python):
    """Stage 1 on the Mapillary crops alone, stage 2 on ours from it: the same classes, the start recorded."""
    from test_detector_sizes import _run
    dataset = _classifier_fixture(tmp_path, capsys)
    crops = _two_stage_crops(tmp_path)
    base = ["-m", "nnslr_tools.cli", "train", str(dataset), "--task", "reader", "--classes", "v2", "--data-root",
            str(tmp_path), "--device", "cpu", "--num-workers", "0", "--epochs", "1", "--batch-size", "4",
            "--input-size", "32", "--no-amp"]
    first, second = tmp_path / "run" / "pretrain", tmp_path / "run" / "classifier"
    process = _run(torch_python, *base, "--mapillary", str(crops), "--mapillary-only", "--output", str(first))
    assert process.returncode == 0, process.stderr[-3000:]
    assert re.search(r"^pretrain epoch 1/1 train_loss \S+ val_loss \S+ train_accuracy \S+ val_accuracy \S+ "
                     r"val_sign_accuracy [0-9.]+ \d+s$", process.stdout, re.MULTILINE), process.stdout
    pretrained = json.loads((first / "result.json").read_text())
    assert pretrained["mapillary_only"] is True and pretrained["start"] is None
    process = _run(torch_python, *base, "--init-from", str(first / "reader-best.pt"), "--output", str(second))
    assert process.returncode == 0, process.stderr[-3000:]
    assert f"classifier starts from {first / 'reader-best.pt'}: every layer, and the head of " in process.stdout
    assert re.search(r"^classifier epoch 1/1 ", process.stdout, re.MULTILINE), process.stdout
    result = json.loads((second / "result.json").read_text())
    assert result["classes"] == pretrained["classes"] and result["initialization"] == "mapillary_pretraining"
    assert result["start"]["sha256"] and result["config"]["init_from"] == str(first / "reader-best.pt")
    plan = json.loads((second / "training-plan.json").read_text())
    assert plan["start"]["mapillary_only"] is True and "maximum_speed:120" in plan["classes_from_start"]
    process = _run(torch_python, "-c", START_CHECK, str(first / "reader-best.pt"), str(second / "reader-best.pt"))
    assert process.returncode == 0 and "start ok" in process.stdout, process.stderr[-3000:]
