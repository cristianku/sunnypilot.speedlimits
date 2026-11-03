"""The MTSD windows of the cutter (nnslr_tools.mtsd_windows): the fully annotated archives only, the window's geometry,
its pixels, the reuse of a cut, and the train frames a plan takes from them (region, share, cap)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image, ImageDraw

from nnslr_tools import mapillary, mtsd_windows
from nnslr_tools.mtsd_windows import (ANNOTATION_ZIP, ANNOTATIONS_PREFIX, IMAGE_ZIPS, SPLITS_PREFIX, WINDOW,
                                      WINDOWS_DIR, WINDOWS_FILE, WINDOWS_KIND, WIDTH_QUANTILES, family_of,
                                      make_windows, mapillary_lineage, plan_frames, plan_photo, read_windows,
                                      window_id, windows_state)
from nnslr_tools.training import CUTTER_FAMILIES
from test_annotations import invoke

PARTIAL_ANNOTATIONS = "mtsd_v2_partially_annotated_annotation.zip"
PARTIAL_IMAGES = "mtsd_v2_partially_annotated_images.train.0.zip"
RED = (220, 20, 20)


def test_the_families_of_the_mtsd_labels():
    assert family_of("regulatory--maximum-speed-limit-50--g1") == ("maximum_speed", "g1", 50, "speed")
    assert family_of("regulatory--maximum-speed-limit-55--g2") == ("maximum_speed", "g2", 55, "speed")
    assert family_of("regulatory--maximum-speed-limit-led-60--g1") == ("variable_display", "g1", 60, "speed")
    assert family_of("regulatory--end-of-maximum-speed-limit-70--g1") == ("cancellation", "g1", 70, "speed")
    assert family_of("regulatory--end-of-speed-limit-zone--g1") == ("cancellation", "g1", None, "speed")
    assert family_of("regulatory--end-of-prohibition--g1") == ("cancellation", "g1", None, "speed")
    # A speed that is no limit of the road, a plate, anything else.
    assert family_of("complementary--maximum-speed-limit-30--g1") == (None, "g1", 30, "speed_like")
    assert family_of("regulatory--minimum-speed-40--g1") == (None, "g1", 40, "speed_like")
    assert family_of("complementary--tow-away-zone--g1") == (None, "g1", None, "plate")
    assert family_of("warning--curve-left--g1") == ("other_sign", "g1", None, "other")
    assert family_of("other-sign") == ("other_sign", None, None, "other")


# --- A fake MTSD: the fully annotated archives, and the partially annotated ones beside them ------------------------

def _obj(key, label, box, **properties):
    return {"key": key, "label": label, "bbox": dict(zip(("xmin", "ymin", "xmax", "ymax"), box)),
            "properties": properties}


def _jpeg(width, height, boxes):
    image = Image.new("RGB", (width, height), (128, 128, 128))
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill=RED)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


PHOTOS = {
    # key: (split, width, height, objects, in the fully annotated images)
    "k_one": ("train", 1600, 900, [_obj("a", "regulatory--maximum-speed-limit-50--g1", [700, 300, 740, 340])], True),
    "k_two": ("train", 1600, 900, [
        _obj("b", "regulatory--maximum-speed-limit-80--g1", [300, 250, 340, 290]),
        _obj("c", "regulatory--maximum-speed-limit-led-60--g1", [1150, 350, 1186, 386]),
        _obj("d", "warning--curve-left--g1", [800, 300, 840, 340]),
        _obj("e", "complementary--tow-away-zone--g1", [300, 292, 340, 306]),
        _obj("f", "complementary--maximum-speed-limit-30--g1", [1000, 300, 1030, 330]),
        _obj("g", "regulatory--maximum-speed-limit-30--g1", [500, 300, 530, 330], ambiguous=True),
    ], True),
    "k_pano": ("train", 1600, 900, [_obj("h", "regulatory--maximum-speed-limit-50--g1", [700, 300, 740, 340])], True),
    "k_tiny": ("train", 1000, 300, [_obj("i", "regulatory--maximum-speed-limit-50--g1", [400, 100, 440, 140])], True),
    "k_other": ("train", 1600, 900, [_obj("j", "warning--curve-left--g1", [700, 300, 740, 340])], True),
    # Its photo is only in the partially annotated images: it counts as missing.
    "k_missing": ("train", 1600, 900, [_obj("k", "regulatory--maximum-speed-limit-50--g1", [700, 300, 740, 340])],
                  False),
    "k_val": ("val", 1600, 900, [_obj("l", "regulatory--maximum-speed-limit-60--g1", [700, 300, 740, 340])], True),
}


def _fake_mtsd(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(folder / ANNOTATION_ZIP, "w") as archive:
        for split in ("train", "val"):
            keys = [key for key, photo in PHOTOS.items() if photo[0] == split]
            archive.writestr(SPLITS_PREFIX + split + ".txt", "\n".join(keys) + "\n")
        for key, (split, width, height, objects, _) in PHOTOS.items():
            document = {"width": width, "height": height, "ispano": key == "k_pano", "objects": objects}
            archive.writestr(ANNOTATIONS_PREFIX + key + ".json", json.dumps(document))
    for split, name in (("train", IMAGE_ZIPS["train"][0]), ("val", IMAGE_ZIPS["val"][0])):
        with zipfile.ZipFile(folder / name, "w") as archive:
            for key, (photo_split, width, height, objects, full) in PHOTOS.items():
                if photo_split == split and full:
                    boxes = [list(obj["bbox"].values()) for obj in objects]
                    archive.writestr(f"images/{key}.jpg", _jpeg(width, height, boxes))
    # The partially annotated MTSD: a photo the fully annotated splits name, and one of its own.
    with zipfile.ZipFile(folder / PARTIAL_ANNOTATIONS, "w") as archive:
        archive.writestr("mtsd_v2_partially_annotated/annotations/k_partial.json",
                         json.dumps({"width": 1600, "height": 900, "objects": []}))
    with zipfile.ZipFile(folder / PARTIAL_IMAGES, "w") as archive:
        archive.writestr("images/k_missing.jpg", _jpeg(1600, 900, []))
        archive.writestr("images/k_partial.jpg", _jpeg(1600, 900, []))
    return folder


@pytest.fixture
def opened(monkeypatch):
    """Every archive opened while the test runs; a partially annotated one fails the test at once."""
    names = []
    original = zipfile.ZipFile

    class Recording(original):
        def __init__(self, file, mode="r", *args, **kwargs):
            if mode == "r":
                name = Path(file).name
                assert "partially" not in name, f"{name} opened"
                names.append(name)
            super().__init__(file, mode, *args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", Recording)
    return names


def _rows(root):
    return read_windows(root / WINDOWS_FILE)[1]


def test_a_cut_reads_the_fully_annotated_archives_only_and_the_train_split(tmp_path, opened):
    mtsd = _fake_mtsd(tmp_path / "zips")
    lines = []
    header = make_windows(mtsd, tmp_path / "data", workers=1, log=lines.append)

    assert set(opened) == {ANNOTATION_ZIP, IMAGE_ZIPS["train"][0]}
    rows = _rows(tmp_path / "data")
    assert {row["image_key"] for row in rows} == {"k_one", "k_two"} and {row["split"] for row in rows} == {"train"}
    assert not (tmp_path / "data" / WINDOWS_DIR / "val").exists()
    skipped = header["counts"]["train"]["skipped"]
    assert skipped == {"missing": 1, "panorama": 1, "photo_too_small": 1}
    assert header["archives_missing"] == list(IMAGE_ZIPS["train"][1:])
    assert header["archives"]["images"] == {"train": [IMAGE_ZIPS["train"][0]]}
    assert header["kind"] == WINDOWS_KIND and header["window"] == [1344, 432] and header["other_windows"] == 0
    assert header["quantiles"]["width"] == [list(pair) for pair in WIDTH_QUANTILES]
    assert header["counts"]["train"]["windows"] == len(rows) == 3
    assert any("2/2 photos cut" in line for line in lines)
    with pytest.raises(ValueError, match="mapillary_partial"):
        mtsd_windows._open(mtsd, PARTIAL_ANNOTATIONS)


def test_a_window_is_the_band_and_its_boxes_are_where_the_signs_are(tmp_path, opened):
    mtsd = _fake_mtsd(tmp_path / "zips")
    make_windows(mtsd, tmp_path / "data", workers=1, log=lambda line: None)
    for row in _rows(tmp_path / "data"):
        path = tmp_path / "data" / row["image_path"]
        assert row["image_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["window_id"] == window_id(row["image_key"], row["scale"], *row["origin"])
        with Image.open(path) as image:
            assert image.size == WINDOW
            pixels = image.convert("RGB")
            for obj in row["objects"]:
                if obj["status"] != "clean":
                    continue
                x1, y1, x2, y2 = obj["box"]
                red = pixels.getpixel((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                assert red[0] > 150 and red[1] < 90 and red[2] < 90, (row["window_id"], obj["label"], red)
        anchor = next(obj for obj in row["objects"] if obj["label"] == row["anchor"]["label"])
        assert anchor["status"] == "clean" and anchor["family"] == row["anchor"]["family"]
    two = [row for row in _rows(tmp_path / "data") if row["image_key"] == "k_two"]
    assert {row["anchor"]["label"] for row in two} == {"regulatory--maximum-speed-limit-80--g1",
                                                       "regulatory--maximum-speed-limit-led-60--g1"}
    statuses = {obj["label"]: (obj["status"], obj["family"], obj["kind"]) for obj in two[0]["objects"]}
    assert statuses["regulatory--maximum-speed-limit-30--g1"][0] == "ambiguous"
    assert statuses["warning--curve-left--g1"] == ("clean", "other_sign", "other")
    assert statuses["complementary--maximum-speed-limit-30--g1"][1:] == (None, "speed_like")


def test_the_geometry_draws_our_sizes_and_stays_in_the_photo():
    document = {"width": 13440, "height": 4320, "objects": [
        _obj("a", "regulatory--maximum-speed-limit-50--g1", [6000, 1500, 6100, 1600])]}
    widths = []
    for number in range(60):
        key = f"key{number}"
        windows, skipped = plan_photo(key, document)
        assert not skipped and len(windows) == 1
        window = windows[0]
        scale, (ox, oy) = window["scale"], window["origin"]
        assert 0 <= ox <= 13440 * scale - WINDOW[0] + 1e-6 and 0 <= oy <= 4320 * scale - WINDOW[1] + 1e-6
        (anchor,) = [obj for obj in window["objects"] if obj["status"] == "clean"]
        widths.append(anchor["box"][2] - anchor["box"][0])
        assert plan_photo(key, document)[0] == windows  # the same windows every time
    # The smallest scale (0.1) makes the sign 10 pixels wide; above it the widths are our quantiles'.
    assert min(widths) >= 10 - 0.01 and max(widths) <= WIDTH_QUANTILES[-1][1] + 0.01
    assert len({round(width) for width in widths}) > 10


def test_the_statuses_of_the_objects_in_a_window():
    def placed(box, **properties):
        return mtsd_windows._in_window(mtsd_windows._objects({"objects": [_obj("o", "warning--x--g1", box,
                                                                                  **properties)]})[0], 1.0, 0, 0)
    edge = placed([-10, 10, 30, 40])
    assert edge["status"] == "edge" and edge["box"] == [0.0, 10.0, 30.0, 40.0]
    assert placed([100, 100, 104, 120])["status"] == "small"
    assert placed([100, 100, 130, 130], occluded=True)["status"] == "clean"
    assert placed([100, 100, 130, 130], ambiguous=True)["status"] == "ambiguous"
    assert placed([2000, 100, 2030, 130]) is None


def test_the_photos_that_give_no_window():
    sign = [_obj("a", "regulatory--maximum-speed-limit-50--g1", [400, 100, 440, 140])]
    assert plan_photo("k", {"width": 1000, "height": 300, "objects": sign})[1] == {"photo_too_small": 1}
    windows, _ = plan_photo("k", {"width": 1100, "height": 400, "objects": sign})
    assert windows and windows[0]["scale"] >= 1344 / 1100  # scaled up by at most MAX_UPSCALE
    assert plan_photo("k", {"width": 1600, "height": 900, "ispano": True, "objects": sign})[1] == {"panorama": 1}
    assert plan_photo("k", {"objects": sign})[1] == {"no_size": 1}
    other = [_obj("a", "warning--curve-left--g1", [700, 300, 740, 340])]
    assert plan_photo("k", {"width": 1600, "height": 900, "objects": other}) == ([], {})
    windows, _ = plan_photo("k", {"width": 1600, "height": 900, "objects": other}, other_windows=1)
    assert [window["anchor"]["family"] for window in windows] == ["other_sign"]


def test_a_second_cut_reuses_the_files_and_another_geometry_writes_new_ones(tmp_path, opened, monkeypatch):
    mtsd = _fake_mtsd(tmp_path / "zips")
    data = tmp_path / "data"
    make_windows(mtsd, data, workers=1, log=lambda line: None)
    first = {row["window_id"]: row["image_sha256"] for row in _rows(data)}

    def never(*args, **kwargs):
        raise AssertionError("a window already cut was cut again")
    with monkeypatch.context() as patch:
        patch.setattr(mtsd_windows, "_cut", never)
        make_windows(mtsd, data, workers=1, log=lambda line: None)
    assert {row["window_id"]: row["image_sha256"] for row in _rows(data)} == first

    with monkeypatch.context() as patch:
        patch.setattr(mtsd_windows, "WINDOW_VERSION", 2)
        header = make_windows(mtsd, data, workers=1, log=lambda line: None)
        second = {row["window_id"] for row in _rows(data)}
    assert header["version"] == 2 and not second & set(first)
    assert all((data / WINDOWS_DIR / "train" / f"{name}.jpg").is_file() for name in second | set(first))
    with pytest.raises(ValueError, match="mapillary_invalid"):
        read_windows(data / WINDOWS_FILE)


def test_a_pool_of_processes_cuts_the_same_pixels(tmp_path, opened):
    mtsd = _fake_mtsd(tmp_path / "zips")
    make_windows(mtsd, tmp_path / "one", workers=1, log=lambda line: None)
    make_windows(mtsd, tmp_path / "two", workers=2, log=lambda line: None)
    assert ({row["window_id"]: row["image_sha256"] for row in _rows(tmp_path / "one")}
            == {row["window_id"]: row["image_sha256"] for row in _rows(tmp_path / "two")})


def test_val_only_on_request_a_limit_and_a_dry_run_that_writes_nothing(tmp_path, opened):
    mtsd = _fake_mtsd(tmp_path / "zips")
    header = make_windows(mtsd, tmp_path / "dry", dry_run=True, with_val=True, log=lambda line: None)
    assert header["dry_run"] and not (tmp_path / "dry").exists()
    assert header["counts"]["val"]["windows"] == 1 and header["counts"]["train"]["windows"] == 4
    assert set(opened) == {ANNOTATION_ZIP}
    # The first photos by key, before their pixels are looked for: k_missing, then k_one.
    make_windows(mtsd, tmp_path / "data", with_val=True, limit=2, workers=1, log=lambda line: None)
    rows = _rows(tmp_path / "data")
    assert sorted((row["split"], row["image_key"]) for row in rows) == [("train", "k_one"), ("val", "k_val")]
    with pytest.raises(ValueError, match="mapillary_missing"):
        make_windows(tmp_path / "nowhere", tmp_path / "data", log=lambda line: None)


def test_the_command_cuts_into_the_data_root_given(tmp_path, capsys, opened):
    mtsd = _fake_mtsd(tmp_path / "zips")
    code, out = invoke(capsys, "mapillary-windows", "--mtsd", str(mtsd), "--data-root", str(tmp_path / "data"),
                       "--workers", "1")
    assert code == 0, out.err + out.out
    assert json.loads(out.out.strip().splitlines()[-1])["counts"]["train"]["windows"] == 3
    code, out = invoke(capsys, "mapillary-windows", "--mtsd", str(mtsd), "--limit", "0")
    assert code == 2 and "--limit" in out.err


# --- The train frames of a plan ---------------------------------------------------------------------------------------

def _clean(box, family="maximum_speed", design="g1"):
    return {"box": box, "family": family, "design": design, "status": "clean", "label": "x", "kind": "speed",
            "value": 50}


def _windows_file(root: Path, photos: dict[str, list[int]], *, val: bool = True) -> Path:
    """A windows file of make_windows' shape: each photo's windows with this many clean speed boxes each, one other
    sign and one small speed sign; and a val window, which a plan never takes."""
    rows = []
    for key, windows in photos.items():
        for number, speed in enumerate(windows):
            name = f"{key}_{number:010d}"
            objects = [_clean([10 + 30 * i, 10, 30 + 30 * i, 30], design="g1" if i % 2 == 0 else "g3")
                       for i in range(speed)]
            objects += [_clean([500, 10, 520, 30], family="other_sign", design=None),
                        {**_clean([600, 10, 603, 13]), "status": "small"}]
            rows.append((name, key, "train", objects))
    if val:
        rows.append(("v_val_0000000000", "v_val", "val", [_clean([10, 10, 30, 30])]))
    lines = [json.dumps({"kind": WINDOWS_KIND, "version": mtsd_windows.WINDOW_VERSION,
                         "counts": {"train": {"windows": sum(len(w) for w in photos.values()), "photos": len(photos),
                                              "boxes_by_family": {"maximum_speed": sum(map(sum, photos.values()))}}}})]
    for name, key, split, objects in rows:
        path = root / WINDOWS_DIR / split / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        lines.append(json.dumps({"window_id": name, "image_key": key, "split": split,
                                 "image_path": path.relative_to(root).as_posix(),
                                 "image_sha256": hashlib.sha256(name.encode()).hexdigest(), "width": 1344,
                                 "height": 432, "objects": objects}))
    target = root / WINDOWS_FILE
    target.write_text("\n".join(lines) + "\n")
    return target


def _plan(path, root, **kwargs):
    return plan_frames(path, root, families=CUTTER_FAMILIES["all"], **kwargs)


def test_a_plan_takes_the_clean_speed_signs_and_ignores_every_other_object(tmp_path):
    path = _windows_file(tmp_path, {"k_one": [1], "k_two": [2, 1]})
    frames, summary = _plan(path, tmp_path)
    assert [frame["window_id"] for frame in frames] == ["k_one_0000000000", "k_two_0000000000", "k_two_0000000001"]
    first = frames[1]
    assert first["boxes"] == [[10, 10, 30, 30], [40, 10, 60, 30]] and first["box_families"] == ["maximum_speed"] * 2
    # The other sign and the small speed sign: neither a sign to find nor background.
    assert first["ignore_boxes"] == [[500, 10, 520, 30], [600, 10, 603, 13]]
    assert all(frame["split"] == "train" and frame["source"] == "mapillary" and frame["route_id"] == "mapillary"
               and frame["label_basis"] == "mapillary_mtsd" and "designs" not in frame for frame in frames)
    assert summary["frames"] == 3 and summary["photos"] == 2 and summary["boxes_by_family"] == {"maximum_speed": 4}
    assert summary["ignore_boxes"] == 6 and summary["by_design"] == {"g1": 3, "g3": 1}
    assert summary["region"] == "all" and summary["share"] == 100.0 and summary["licence"] == "CC BY-NC-SA 4.0"
    assert summary["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["before_cap"] == {"windows": 3, "speed_boxes": 4} and summary["max_speed_boxes"] is None


def test_the_counts_of_a_split_box_the_speed_signs_and_count_the_other_signs_apart():
    rows = [{"image_key": "k", "objects": [_clean([0, 0, 20, 20]), _clean([30, 0, 50, 20], family="other_sign"),
                                           {**_clean([60, 0, 63, 3]), "status": "small"}]},
            {"image_key": "k", "objects": [_clean([0, 0, 20, 20], family="cancellation")]}]
    assert mtsd_windows._counts(rows) == {"windows": 2, "photos": 1, "objects_by_status": {"clean": 3, "small": 1},
                                          "boxes_by_family": {"cancellation": 1, "maximum_speed": 1},
                                          "other_signs_ignored": 1}


def test_a_windows_file_of_another_data_root_is_refused(tmp_path):
    # Its image paths are of the root it was cut into: in another root they would be other pixels, or none.
    path = _windows_file(tmp_path / "sample", {"k_one": [1]})
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="mapillary_outside_root"):
        _plan(path, tmp_path / "data")
    (tmp_path / "link").symlink_to(tmp_path / "sample")
    assert len(_plan(path, tmp_path / "link")[0]) == len(_plan(path, tmp_path / "sample")[0]) == 1


def test_a_window_with_our_pixels_or_without_its_file_is_refused(tmp_path):
    path = _windows_file(tmp_path, {"k_one": [1], "k_two": [1]})
    ours = [{"image_sha256": hashlib.sha256(b"k_two_0000000000").hexdigest()}]
    with pytest.raises(ValueError, match="duplicate"):
        _plan(path, tmp_path, own_frames=ours)
    (tmp_path / WINDOWS_DIR / "train" / "k_one_0000000000.jpg").unlink()
    with pytest.raises(ValueError, match="mapillary_window_missing"):
        _plan(path, tmp_path)


def _placed(key, country, continent, located=True):
    return {"mtsd_image": key, "image_id": "1", "latitude": 47.0 if located else None,
            "longitude": 8.0 if located else None, "country": country, "continent": continent,
            "status": "located" if located else "not_found"}


def test_the_region_then_the_share_of_the_photos(tmp_path):
    photos = {f"k{number:02d}": [1, 1] for number in range(40)}
    path = _windows_file(tmp_path, photos)
    with pytest.raises(ValueError, match="mapillary_locations_missing"):
        _plan(path, tmp_path, region="europe")
    lines = [_placed(key, "CH", "europe") if number % 2 == 0 else _placed(key, "US", "north_america")
             for number, key in enumerate(photos)]
    lines[0] = _placed("k00", None, None, located=False)
    locations = tmp_path / mapillary.LOCATIONS_FILE
    locations.write_text("".join(json.dumps(line) + "\n" for line in lines))

    europe, summary = _plan(path, tmp_path, region="Europe")
    assert {frame["image_key"] for frame in europe} == {f"k{n:02d}" for n in range(2, 40, 2)}
    assert summary["region"] == "europe" and summary["locations"]["located_photos"] == 39
    half, _ = _plan(path, tmp_path, region="europe", share=50)
    more, _ = _plan(path, tmp_path, region="europe", share=60)
    assert {f["window_id"] for f in half} <= {f["window_id"] for f in more} <= {f["window_id"] for f in europe}
    everything, _ = _plan(path, tmp_path, share=50)
    keys = {frame["image_key"] for frame in everything}
    assert 0 < len(keys) < 40
    # The windows of a photo come and go together.
    assert sorted(frame["window_id"] for frame in everything) == sorted(f"{key}_{n:010d}" for key in keys
                                                                        for n in range(2))
    with pytest.raises(ValueError, match="mapillary_share"):
        _plan(path, tmp_path, share="0")


def test_the_cap_keeps_whole_photos_and_a_smaller_cap_is_a_part_of_a_larger_one(tmp_path):
    path = _windows_file(tmp_path, {f"k{number:02d}": [2, 1] for number in range(20)})
    small, summary = _plan(path, tmp_path, max_speed_boxes=10)
    large, _ = _plan(path, tmp_path, max_speed_boxes=21)
    assert sum(len(frame["boxes"]) for frame in small) == 9 and sum(len(frame["boxes"]) for frame in large) == 21
    assert {frame["window_id"] for frame in small} < {frame["window_id"] for frame in large}
    for frames in (small, large):
        keys = {frame["image_key"] for frame in frames}
        assert len(frames) == 2 * len(keys)
    assert summary["max_speed_boxes"] == 10 and summary["before_cap"] == {"windows": 40, "speed_boxes": 60}
    assert summary["frames"] == len(small)
    assert summary["selection_sha256"] != _plan(path, tmp_path)[1]["selection_sha256"]


def test_the_state_of_the_windows_for_the_panel(tmp_path):
    state = windows_state(tmp_path)
    assert state["available"] is False and "helpers/mapillary_windows.sh" in state["reason"]
    _windows_file(tmp_path, {"k_one": [1], "k_two": [2, 1]})
    state = windows_state(tmp_path)
    assert state == {"available": True, "windows": 3, "photos": 2, "boxes_by_family": {"maximum_speed": 4},
                     "reason": ""}
    _windows_file(tmp_path, {"k_one": [1]})
    assert windows_state(tmp_path)["windows"] == 1
    # The header of the first cut of 3 Nov 2026 counted the other signs among the boxes: the state leaves them out.
    path = tmp_path / WINDOWS_FILE
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["counts"]["train"]["boxes_by_family"]["other_sign"] = 7
    path.write_text("\n".join([json.dumps(header)] + lines[1:] + [""]))
    assert windows_state(tmp_path)["boxes_by_family"] == {"maximum_speed": 1}


# --- The licence of a network: whether MTSD is anywhere in its chain of starts -------------------------------------

def _run(root: Path, name: str, **result) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps(result))
    return folder / "signnet-best.pt"


def test_the_lineage_follows_the_starts(tmp_path):
    summary = {"frames": 10}
    stage = _run(tmp_path, "A/pretrain", mapillary=summary, mapillary_only=True)
    a = _run(tmp_path, "A", mapillary=None, mapillary_only=False,
             start={"path": str(stage), "mapillary": summary, "mapillary_only": True})
    b = _run(tmp_path, "B", start={"path": str(a)})
    assert mapillary_lineage(a) == {"uses_mapillary": True, "runs": [str(stage.parent)], "licence": "CC BY-NC-SA 4.0"}
    assert mapillary_lineage(b)["runs"] == [str(stage.parent)]
    # A start whose folder is gone still counts by what the run recorded of it.
    gone = _run(tmp_path, "C", start={"path": str(tmp_path / "gone" / "signnet-best.pt"), "mapillary_only": True})
    assert mapillary_lineage(gone)["runs"] == [str(tmp_path / "gone")]
    # A start that went on from a Mapillary stage, its folder gone: signnet_start's 'uses_mapillary' counts.
    later = _run(tmp_path, "H", start={"path": str(tmp_path / "away" / "signnet-best.pt"), "mapillary": None,
                                       "mapillary_only": False, "uses_mapillary": True})
    assert mapillary_lineage(later) == {"uses_mapillary": True, "runs": [str(tmp_path / "away")],
                                        "licence": "CC BY-NC-SA 4.0"}
    # With that start's folder there, the chain goes on to the stage that learnt them: the start itself, ours only,
    # is not named.
    second = _run(tmp_path, "I", mapillary=None, start={"path": str(stage), "mapillary": summary, "mapillary_only": True})
    after = _run(tmp_path, "J", start={"path": str(second), "mapillary": None, "uses_mapillary": True})
    assert mapillary_lineage(after)["runs"] == [str(stage.parent)]
    # A classifier trained on Mapillary alone: its training-plan.json says so.
    classifier = tmp_path / "D"
    classifier.mkdir()
    (classifier / "training-plan.json").write_text(json.dumps({"mapillary_only": True}))
    assert mapillary_lineage(classifier / "reader-best.pt")["uses_mapillary"] is True
    assert mapillary_lineage(_run(tmp_path, "E", mapillary=None)) == {"uses_mapillary": False, "runs": [],
                                                                     "licence": "CC BY-NC-SA 4.0"}
    f = tmp_path / "F" / "signnet-best.pt"
    _run(tmp_path, "G", start={"path": str(f)})
    _run(tmp_path, "F", start={"path": str(tmp_path / "G" / "signnet-best.pt")})
    assert mapillary_lineage(f)["uses_mapillary"] is False
