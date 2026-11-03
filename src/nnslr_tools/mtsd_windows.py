"""The Mapillary Traffic Sign Dataset (MTSD) as windows for the cutter (SignNet): pieces of its photos, scaled and
cut so that they look like the top band of our road camera's frames, with the box of every sign in them.

What it is for: the cutter learns the speed signs of other countries (their designs, their sizes) from MTSD, as
the classifier already does from its crops (nnslr_tools.mapillary). ``nnslr mapillary-windows``
(helpers/mapillary_windows.sh) cuts them once, on the CPU (make_windows); ``nnslr train-signnet --mapillary``
takes them into the cutter's plan (training.prepare_detector_training, through plan_frames).

- Only the fully annotated MTSD is read, its archives by name (ANNOTATION_ZIP, IMAGE_ZIPS): the partially
  annotated one leaves signs without a box, and the cutter would learn them as background. Its test split has no
  public annotations; its val split is cut only on request (--with-val): the cutter is measured on our frames.
- A window is WINDOW (1344x432) pixels: our frame's width and its top band, exactly the car's 896x288 band once
  resized (signnet.CAR_BAND). For each clean speed sign of a photo (at most SPEED_WINDOWS_PER_PHOTO), the photo
  is scaled so that the sign has a width drawn from our own train boxes (WIDTH_QUANTILES), then cut so that the
  sign sits where our boxes do (CENTRE_X_QUANTILES, CENTRE_Y_QUANTILES). Only as near as MTSD allows: its photos
  show most signs close up, and a photo is never scaled below the window, so the speed signs of the windows come
  out larger than ours (in the cut of 3 Nov 2026 a median width of 25.5 px against our 18, and few under our q25),
  half of the windows at that smallest scale, where the sign keeps its place across the photo.
- Every MTSD object in a window is listed with its status: 'clean' (a box to learn), a property MTSD marks
  (mapillary.SKIPPED_PROPERTIES: ambiguous, dummy...), 'edge' (cut by the window's border) or 'small' (a side
  under MIN_SIDE). In a plan only the clean objects of the speed families are boxes; every other object, the
  other signs included, is an ignore box: neither a sign to find nor background.
- The geometry (WINDOW_VERSION, the scale and the origin) is in each window's id, so a cut with another geometry
  writes new files and never pairs old pixels with new boxes.

Licence: MTSD is CC BY-NC-SA 4.0 (non-commercial, attribution, share alike). mapillary_lineage says whether a
network learnt from it, through its chain of starts. Standard library only; PIL is imported when it cuts.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any
import zipfile

from nnslr_tools import mapillary
from nnslr_tools.annotations import atomic_write

WINDOWS_KIND = "mapillary_mtsd_windows_v1"
WINDOW_VERSION = 1
WINDOWS_DIR = Path("external") / "mapillary" / "windows"
WINDOWS_FILE = Path("external") / "mapillary" / "windows.jsonl"
DEFAULT_MTSD_DIR = Path("/data/mapillary/zips")
# The fully annotated MTSD, its archives as downloaded. The partially annotated ones (PARTIAL in the name) are
# never opened: their photos have signs without a box.
ANNOTATION_ZIP = "mtsd_v2_fully_annotated_annotation.zip"
IMAGE_ZIPS = {
    "train": ("mtsd_v2_fully_annotated_images.train.0.zip", "mtsd_v2_fully_annotated_images.train.1.zip",
              "mtsd_v2_fully_annotated_images.train.2.zip"),
    "val": ("mtsd_v2_fully_annotated_images.val.zip",),
}
PARTIAL = "partially_annotated"
ANNOTATIONS_PREFIX = "mtsd_v2_fully_annotated/annotations/"
SPLITS_PREFIX = "mtsd_v2_fully_annotated/splits/"
IMAGES_PREFIX = "images/"
TRAIN = "train"

WINDOW = (1344, 432)
JPEG_QUALITY = 95
# A photo smaller than the window is scaled up by at most this much, else it gives no window.
MAX_UPSCALE = 1.25
# An object with a side under this many window pixels is 'small': an ignore box, too small to learn.
MIN_SIDE = 6
SPEED_WINDOWS_PER_PHOTO = 2
OTHER_WINDOWS_PER_PHOTO = 0
SPEED_FAMILIES = ("maximum_speed", "variable_display", "cancellation")
OTHER_SIGN = "other_sign"
CLEAN = "clean"
SOURCE = "mapillary"
ROUTE = "mapillary"
LABEL_BASIS = "mapillary_mtsd"
LICENCE = "CC BY-NC-SA 4.0"
# Our train boxes in the band (data of 2 Nov 2026), in frame pixels (1344 wide), as (quantile, value): the width of
# those of the speed families (1781 boxes, unreadable and zone included); the centre (x, and y in the band's 432
# rows) of every box, the other signs' 1102 included (2883). Our speed signs alone sit lower and more to the left
# (3 Nov 2026: centre y 270/330/361/380/392, x 465/597/729/859/987): tables of those would place the windows'
# signs 10 to 25 rows lower. A change of a table changes the cut: WINDOW_VERSION up, and the windows cut again.
# A window draws its sign's width and place from them (_draw).
WIDTH_QUANTILES = ((0.05, 8.7), (0.10, 10.1), (0.25, 12.1), (0.50, 18.0), (0.75, 30.6), (0.90, 51.4), (0.95, 67.1))
CENTRE_X_QUANTILES = ((0.10, 489.0), (0.25, 629.0), (0.50, 776.0), (0.75, 913.0), (0.90, 1088.0))
CENTRE_Y_QUANTILES = ((0.10, 247.0), (0.25, 312.0), (0.50, 350.0), (0.75, 375.0), (0.90, 388.0))

_MAXIMUM = re.compile(r"^regulatory--maximum-speed-limit-(\d+)--(g\d+)$")
_LED = re.compile(r"^regulatory--maximum-speed-limit-led-(\d+)--(g\d+)$")
_END = re.compile(r"^regulatory--end-of-(?:maximum-speed-limit-(\d+)|speed-limit-zone|prohibition)--(g\d+)$")
_SPEED_LIKE = re.compile(r"^(?:complementary--maximum-speed-limit|information--minimum-speed|regulatory--minimum-speed"
                         r"|regulatory--truck-speed-limit)-(\d+)--(g\d+)$")
_DESIGN = re.compile(r"--(g\d+)$")


def family_of(label: str) -> tuple[str | None, str | None, int | None, str]:
    """(family, design, value, kind) of an MTSD label. A maximum speed of any design (g1 round, g2/g3/g6 the
    American and Canadian rectangles) is maximum_speed, its LED version variable_display, the end of a limit, of a
    speed zone or of every prohibition cancellation: kind 'speed', the cutter's boxes. A speed that is no limit of
    the road (a plate under a sign, a minimum, a truck's) has no family, kind 'speed_like'; any other plate
    (complementary--*) kind 'plate'; everything else is other_sign, kind 'other'. Its own patterns: never
    mapillary.SPEED_LABEL."""
    label = str(label)
    for pattern, family in ((_MAXIMUM, "maximum_speed"), (_LED, "variable_display"), (_END, "cancellation")):
        found = pattern.match(label)
        if found:
            value = found.group(1)
            return family, found.group(2), int(value) if value else None, "speed"
    found = _SPEED_LIKE.match(label)
    if found:
        return None, found.group(2), int(found.group(1)), "speed_like"
    design = _DESIGN.search(label)
    design = design.group(1) if design else None
    if label.startswith("complementary--"):
        return None, design, None, "plate"
    return OTHER_SIGN, design, None, "other"


def photo_share_point(image_key: str) -> float:
    """Where a photo falls in [0, 1) for the share: its windows are in a share of p percent when this is below
    p / 100, so the windows of a photo come and go together. A copy of mapillary._share_point's formula on the
    photo's key (that one hashes a crop's object key): to be shared once mapillary.py can be edited."""
    digest = hashlib.sha1((mapillary.SHARE_SALT + "photo:" + str(image_key)).encode()).hexdigest()
    return int(digest[:13], 16) / float(16 ** 13)


def _draw(rng: random.Random, table) -> float:
    """A value drawn from a quantile table: the piecewise-linear inverse of its distribution, between its first and
    its last quantile."""
    point = rng.uniform(table[0][0], table[-1][0])
    for (q0, v0), (q1, v1) in zip(table, table[1:]):
        if point <= q1:
            return v0 + (v1 - v0) * (point - q0) / (q1 - q0)
    return float(table[-1][1])


def _objects(document: dict[str, Any]) -> list[dict[str, Any]]:
    """The objects of an MTSD annotation with a box, in photo pixels: label, family, design, value, kind, the
    property that MTSD marks (or None) and the box."""
    found = []
    for item in document.get("objects") or []:
        bbox = item.get("bbox") or {}
        try:
            box = [float(bbox["xmin"]), float(bbox["ymin"]), float(bbox["xmax"]), float(bbox["ymax"])]
        except (KeyError, TypeError, ValueError):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        properties = item.get("properties") or {}
        label = str(item.get("label", ""))
        family, design, value, kind = family_of(label)
        found.append({"key": str(item.get("key") or ""), "label": label, "family": family, "design": design,
                      "value": value, "kind": kind, "box": box,
                      "marked": next((name for name in mapillary.SKIPPED_PROPERTIES if properties.get(name)), None)})
    return found


def _in_window(obj: dict[str, Any], scale: float, ox: int, oy: int) -> dict[str, Any] | None:
    """The object in window pixels with its status (CLEAN, its MTSD property, 'edge' or 'small'); None when it is
    not in the window at all. An edge object's box is the part inside the window."""
    x1, y1, x2, y2 = (obj["box"][0] * scale - ox, obj["box"][1] * scale - oy,
                      obj["box"][2] * scale - ox, obj["box"][3] * scale - oy)
    width, height = WINDOW
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    inside = x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height
    box = [round(max(0.0, x1), 2), round(max(0.0, y1), 2), round(min(float(width), x2), 2),
           round(min(float(height), y2), 2)]
    if obj["marked"]:
        status = obj["marked"]
    elif not inside:
        status = "edge"
    elif min(box[2] - box[0], box[3] - box[1]) < MIN_SIDE:
        status = "small"
    else:
        status = CLEAN
    return {"box": box, "label": obj["label"], "family": obj["family"], "kind": obj["kind"],
            "design": obj["design"], "value": obj["value"], "status": status}


def window_id(key: str, scale: float, ox: int, oy: int) -> str:
    """The window's name: the photo's key and a digest of its geometry (WINDOW_VERSION, scale, origin, size)."""
    digest = hashlib.sha1(f"{WINDOW_VERSION}|{key}|{scale:.6f}|{ox}|{oy}|{WINDOW[0]}x{WINDOW[1]}".encode())
    return f"{key}_{digest.hexdigest()[:10]}"


def plan_photo(key: str, document: dict[str, Any], *, other_windows: int = OTHER_WINDOWS_PER_PHOTO
               ) -> tuple[list[dict[str, Any]], Counter]:
    """The windows of one photo from its annotation alone (no pixels): [{window_id, scale, origin, anchor,
    objects}], and a Counter of the anchors that gave none and why. Anchors: up to SPEED_WINDOWS_PER_PHOTO clean
    objects of the speed families and up to ``other_windows`` clean other signs, chosen with a generator seeded by
    the key and WINDOW_VERSION, so a cut gives the same windows every time."""
    skipped: Counter = Counter()
    if document.get("ispano"):
        skipped["panorama"] += 1
        return [], skipped
    width, height = document.get("width"), document.get("height")
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        skipped["no_size"] += 1
        return [], skipped
    objects = _objects(document)
    rng = random.Random(int(hashlib.sha256(f"{key}{WINDOW_VERSION}".encode()).hexdigest()[:16], 16))
    anchors = []
    for families, count in ((SPEED_FAMILIES, SPEED_WINDOWS_PER_PHOTO), ((OTHER_SIGN,), int(other_windows or 0))):
        candidates = sorted((obj for obj in objects if obj["family"] in families and not obj["marked"]),
                            key=lambda obj: (obj["key"], obj["box"]))
        rng.shuffle(candidates)
        anchors += candidates[:max(0, count)]
    if not anchors:
        return [], skipped
    smallest = max(WINDOW[0] / width, WINDOW[1] / height)
    if smallest > MAX_UPSCALE:
        skipped["photo_too_small"] += len(anchors)
        return [], skipped
    windows, seen = [], set()
    for anchor in anchors:
        x1, y1, x2, y2 = anchor["box"]
        scale = min(max(_draw(rng, WIDTH_QUANTILES) / (x2 - x1), smallest), max(smallest, 1.0))
        # Six decimals, the id's and the file's, never below the smallest that covers the window.
        scale = max(round(scale, 6), math.ceil(smallest * 1e6) / 1e6)
        aw, ah = (x2 - x1) * scale, (y2 - y1) * scale
        cx = min(max(_draw(rng, CENTRE_X_QUANTILES), aw / 2), WINDOW[0] - aw / 2)
        cy = min(max(_draw(rng, CENTRE_Y_QUANTILES), ah / 2), WINDOW[1] - ah / 2)
        ox = round((x1 + x2) / 2 * scale - cx)
        oy = round((y1 + y2) / 2 * scale - cy)
        ox = min(max(ox, 0), max(0, math.floor(width * scale - WINDOW[0] + 1e-6)))
        oy = min(max(oy, 0), max(0, math.floor(height * scale - WINDOW[1] + 1e-6)))
        placed = _in_window(anchor, scale, ox, oy)
        if placed is None or placed["status"] != CLEAN:
            skipped["anchor_" + ("edge" if placed is None else placed["status"])] += 1
            continue
        name = window_id(key, scale, ox, oy)
        if name in seen:
            skipped["same_window"] += 1
            continue
        seen.add(name)
        inside = [found for found in (_in_window(obj, scale, ox, oy) for obj in objects) if found is not None]
        windows.append({"window_id": name, "scale": scale, "origin": [ox, oy],
                        "anchor": {"family": anchor["family"], "design": anchor["design"], "label": anchor["label"]},
                        "objects": inside})
    return windows, skipped


def _cut(data: bytes, size: tuple[int, int], cuts: list[tuple[float, int, int, str]], quality: int) -> list[str]:
    """Cut the windows of one photo (a process of the pool): each (scale, ox, oy, file) resized from the photo
    (decoded smaller first with PIL's draft when every scale is 0.5 or less), written to a hidden file and then
    renamed. Returns the sha256 of each file."""
    from PIL import Image
    width, height = size
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (width, height):
            raise ValueError(f"mapillary_size: the photo is {image.size[0]}x{image.size[1]}, its annotation "
                             f"{width}x{height}")
        smallest = min(scale for scale, *_ in cuts)
        if smallest <= 0.5:
            image.draft("RGB", (max(1, math.ceil(width * smallest)), max(1, math.ceil(height * smallest))))
        photo = image.convert("RGB")
    fx, fy = photo.width / width, photo.height / height
    hashes = []
    for scale, ox, oy, path in cuts:
        box = (ox / scale * fx, oy / scale * fy, min(photo.width, (ox + WINDOW[0]) / scale * fx),
               min(photo.height, (oy + WINDOW[1]) / scale * fy))
        buffer = io.BytesIO()
        photo.resize(WINDOW, Image.BILINEAR, box=box).save(buffer, "JPEG", quality=quality)
        payload = buffer.getvalue()
        target = Path(path)
        pending = target.with_name("." + target.name + ".pending")
        pending.write_bytes(payload)
        os.replace(pending, target)
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _open(folder: Path, name: str) -> zipfile.ZipFile:
    """An archive of the fully annotated MTSD, opened for reading by its name; a partially annotated one never."""
    if PARTIAL in name:
        raise ValueError(f"mapillary_partial: {name} is partially annotated, never read")
    return zipfile.ZipFile(Path(folder) / name)


def _split_keys(archive: zipfile.ZipFile, split: str) -> list[str]:
    try:
        text = archive.read(SPLITS_PREFIX + split + ".txt").decode("utf-8")
    except KeyError:
        raise ValueError(f"mapillary_invalid: {ANNOTATION_ZIP} has no {SPLITS_PREFIX}{split}.txt") from None
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The header's counts of a split: its boxes are the clean speed signs (a plan boxes nothing else); its clean
    other signs, ignore boxes in a plan, are counted apart."""
    status, families, others = Counter(), Counter(), 0
    for row in rows:
        for obj in row["objects"]:
            status[obj["status"]] += 1
            if obj["status"] == CLEAN and obj["family"] in SPEED_FAMILIES:
                families[obj["family"]] += 1
            elif obj["status"] == CLEAN and obj["family"] == OTHER_SIGN:
                others += 1
    return {"windows": len(rows), "photos": len({row["image_key"] for row in rows}),
            "objects_by_status": dict(sorted(status.items())), "boxes_by_family": dict(sorted(families.items())),
            "other_signs_ignored": others}


def make_windows(mtsd_dir: Path, data_root: Path, *, workers: int = 4, other_windows: int = OTHER_WINDOWS_PER_PHOTO,
                 limit: int | None = None, with_val: bool = False, dry_run: bool = False, log=print
                 ) -> dict[str, Any]:
    """Cut the windows of the fully annotated MTSD under ``mtsd_dir`` (its archives as downloaded, read where they
    are) into <data root>/WINDOWS_DIR/<split>/ and write WINDOWS_FILE, atomically at the end. The train split only,
    its val too with ``with_val``. ``limit``: the first N photos with a window of each split (a trial cut).
    ``dry_run``: the annotation archive alone is read, nothing written; the header says what a cut would give.

    The photos are read in the order they lie in each archive (header_offset: a network share reads forward) and
    cut in a pool of ``workers`` processes, a few photos in flight at a time. Run again, it reuses each window file
    already there (its id holds the geometry). A key of the split without its photo in the fully annotated
    archives counts as 'missing'. Returns the header."""
    mtsd_dir, data_root = Path(mtsd_dir), Path(data_root)
    if not (mtsd_dir / ANNOTATION_ZIP).is_file():
        raise ValueError(f"mapillary_missing: no {mtsd_dir / ANNOTATION_ZIP}")
    splits = (TRAIN, "val") if with_val else (TRAIN,)
    planned: dict[str, dict[str, Any]] = {}
    skipped: dict[str, Counter] = {split: Counter() for split in splits}
    with _open(mtsd_dir, ANNOTATION_ZIP) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        for split in splits:
            keys = _split_keys(archive, split)
            members = []
            for key in keys:
                info = infos.get(ANNOTATIONS_PREFIX + key + ".json")
                if info is None:
                    skipped[split]["no_annotation"] += 1
                else:
                    members.append((info, key))
            members.sort(key=lambda pair: pair[0].header_offset)
            found: dict[str, dict[str, Any]] = {}
            for number, (info, key) in enumerate(members, 1):
                if number % 5000 == 0:
                    log(f"mapillary-windows: {split} {number}/{len(members)} annotations read")
                try:
                    document = json.loads(archive.read(info))
                except ValueError:
                    skipped[split]["unreadable_annotation"] += 1
                    continue
                windows, why = plan_photo(key, document, other_windows=other_windows)
                skipped[split].update(why)
                if windows:
                    found[key] = {"split": split, "size": (document["width"], document["height"]),
                                  "windows": windows}
            chosen = sorted(found)[:limit] if limit else sorted(found)
            planned.update((key, found[key]) for key in chosen)
            log(f"mapillary-windows: {split} {len(chosen)} photos with a window "
                f"({sum(len(found[key]['windows']) for key in chosen)} windows) of {len(keys)}")
    rows = []
    archives = {"annotations": ANNOTATION_ZIP, "images": {}}
    missing_archives = []
    if not dry_run:
        for split in splits:
            (data_root / WINDOWS_DIR / split).mkdir(parents=True, exist_ok=True)
        rows, missing_archives = _cut_all(mtsd_dir, data_root, planned, splits, workers, archives, skipped, log)
    else:
        for key, photo in sorted(planned.items()):
            rows += [_row(key, photo, window, None) for window in photo["windows"]]
    rows.sort(key=lambda row: (row["split"], row["window_id"]))
    header = {
        "kind": WINDOWS_KIND, "version": WINDOW_VERSION, "mtsd_dir": str(mtsd_dir), "archives": archives,
        "window": list(WINDOW), "jpeg_quality": JPEG_QUALITY, "max_upscale": MAX_UPSCALE, "min_side": MIN_SIDE,
        "speed_windows_per_photo": SPEED_WINDOWS_PER_PHOTO, "other_windows": int(other_windows or 0),
        "quantiles": {"width": [list(pair) for pair in WIDTH_QUANTILES],
                      "centre_x": [list(pair) for pair in CENTRE_X_QUANTILES],
                      "centre_y": [list(pair) for pair in CENTRE_Y_QUANTILES]},
        "counts": {split: {**_counts([row for row in rows if row["split"] == split]),
                           "skipped": dict(sorted(skipped[split].items()))} for split in splits},
        "limit": limit, "licence": f"{LICENCE}, Mapillary Traffic Sign Dataset",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if missing_archives:
        header["archives_missing"] = missing_archives
    if dry_run:
        header["dry_run"] = True
        return header
    lines = [json.dumps(header, sort_keys=True)] + [json.dumps(row, sort_keys=True) for row in rows]
    atomic_write(data_root / WINDOWS_FILE, ("\n".join(lines) + "\n").encode("utf-8"))
    return header


def _row(key: str, photo: dict[str, Any], window: dict[str, Any], sha256: str | None) -> dict[str, Any]:
    return {"window_id": window["window_id"], "image_key": key, "split": photo["split"],
            "image_path": (WINDOWS_DIR / photo["split"] / (window["window_id"] + ".jpg")).as_posix(),
            "image_sha256": sha256, "width": WINDOW[0], "height": WINDOW[1], "source_size": list(photo["size"]),
            "scale": window["scale"], "origin": window["origin"], "anchor": window["anchor"],
            "objects": window["objects"]}


def _cut_all(mtsd_dir: Path, data_root: Path, planned: dict[str, dict[str, Any]], splits, workers: int,
             archives: dict[str, Any], skipped: dict[str, Counter], log) -> tuple[list[dict[str, Any]], list[str]]:
    """make_windows' pixels: every planned photo read from its split's image archives and cut (_cut), in a pool
    when ``workers`` is more than 1. Returns the rows and the image archives that are not there."""
    from concurrent.futures import ProcessPoolExecutor
    rows: list[dict[str, Any]] = []
    missing_archives: list[str] = []
    opened: list[tuple[zipfile.ZipFile, list[tuple[zipfile.ZipInfo, str]]]] = []
    try:
        # First every archive's list of members (its central directory: cheap), so the photos that are in none of
        # them are known before the first is read.
        for split in splits:
            wanted = {key for key, photo in planned.items() if photo["split"] == split}
            archives["images"][split] = []
            seen: set[str] = set()
            for name in IMAGE_ZIPS[split]:
                if not (mtsd_dir / name).is_file():
                    log(f"mapillary-windows: {mtsd_dir / name} is not there: its photos count as missing")
                    missing_archives.append(name)
                    continue
                archive = _open(mtsd_dir, name)
                archives["images"][split].append(name)
                members = []
                for info in archive.infolist():
                    if info.filename.startswith(IMAGES_PREFIX) and info.filename.endswith(".jpg"):
                        key = info.filename[len(IMAGES_PREFIX):-len(".jpg")]
                        if key in wanted and key not in seen:
                            members.append((info, key))
                            seen.add(key)
                members.sort(key=lambda pair: pair[0].header_offset)
                opened.append((archive, members))
            if wanted - seen:
                skipped[split]["missing"] += len(wanted - seen)
                log(f"mapillary-windows: {split} {len(wanted - seen)} photos are not in the fully annotated archives")
        total = sum(len(members) for _, members in opened)
        log(f"mapillary-windows: {total} photos to cut, {max(1, workers)} processes")
        done = 0

        def finish(key: str, hashes: list[str]) -> None:
            nonlocal done
            photo = planned[key]
            rows.extend(_row(key, photo, window, sha) for window, sha in zip(photo["windows"], hashes))
            done += 1
            if done % 200 == 0 or done == total:
                log(f"mapillary-windows: {done}/{total} photos cut")

        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        pending: deque = deque()
        try:
            for archive, members in opened:
                for info, key in members:
                    photo = planned[key]
                    files = [data_root / WINDOWS_DIR / photo["split"] / (window["window_id"] + ".jpg")
                             for window in photo["windows"]]
                    if all(path.is_file() for path in files):
                        finish(key, [_sha256_file(path) for path in files])
                        continue
                    cuts = [(window["scale"], window["origin"][0], window["origin"][1], str(path))
                            for window, path in zip(photo["windows"], files)]
                    data = archive.read(info)
                    if pool is None:
                        finish(key, _cut(data, photo["size"], cuts, JPEG_QUALITY))
                        continue
                    # A few photos in flight only: the bytes of a whole archive would not fit in memory.
                    pending.append((pool.submit(_cut, data, photo["size"], cuts, JPEG_QUALITY), key))
                    while len(pending) > 2 * workers:
                        future, done_key = pending.popleft()
                        finish(done_key, future.result())
            while pending:
                future, done_key = pending.popleft()
                finish(done_key, future.result())
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    finally:
        for archive, _ in opened:
            archive.close()
    return rows, missing_archives


def read_windows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the rows of a windows file of make_windows."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"mapillary_invalid: empty file {path}")
    header = json.loads(lines[0])
    if header.get("kind") != WINDOWS_KIND or header.get("version") != WINDOW_VERSION:
        raise ValueError(f"mapillary_invalid: {path} is not a windows file of nnslr mapillary-windows "
                         f"(version {WINDOW_VERSION})")
    return header, [json.loads(line) for line in lines[1:]]


def _located(location: dict[str, Any] | None, region: frozenset[str]) -> bool:
    """A photo with a position, in one of the region's continents or countries (mapillary._in_region's rule)."""
    if not location or location.get("latitude") is None:
        return False
    return location.get("continent") in region or location.get("country") in region


def _under(path: Path, root: Path) -> bool:
    """Whether ``path`` is inside ``root``, as written or with the links followed."""
    return (Path(os.path.abspath(path)).is_relative_to(os.path.abspath(root))
            or path.resolve().is_relative_to(root.resolve()))


def plan_frames(path: Path, data_root: Path, *, families, region: str | None = mapillary.ALL_REGIONS,
                share: float | None = mapillary.FULL_SHARE, own_frames=(), max_speed_boxes: int | None = None
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The train frames of the cutter's plan from the windows file ``path``, and its summary for the plan.

    Only train windows. The region first (mapillary.parse_region; the positions of locations.jsonl beside the file:
    a photo with none is out when a region is chosen), then the share in percent (photo_share_point: a photo's
    windows together), then ``max_speed_boxes`` when given: whole photos in a fixed order (the smallest sha1 of their
    window ids) while their speed boxes stay within it, so a smaller cap takes a part of a larger one.

    A frame's boxes are its clean objects of ``families`` (other_sign never: MTSD's other signs are not boxed as
    ours are); every other object is an ignore box. A missing window file is an error, so is a window with the same
    pixels (sha256) as one of ``own_frames``, and so is a windows file outside ``data_root``: its image paths are of
    the root it was cut into."""
    path, data_root = Path(path), Path(data_root)
    if not _under(path, data_root):
        raise ValueError(f"mapillary_outside_root: {path} is not under the data root {data_root}; its windows are "
                         "in the data root it was cut into: give that one (--data-root)")
    header, rows = read_windows(path)
    wanted = set(families) - {OTHER_SIGN}
    parts = mapillary.parse_region(region)
    percent = mapillary.check_share(share)
    rows = [row for row in rows if row.get("split") == TRAIN]
    locations_summary = None
    if parts is not None:
        locations_path = mapillary.locations_path_of(path)
        if not locations_path.is_file():
            raise ValueError(f"mapillary_locations_missing: no {locations_path}: run helpers/mapillary_locations.sh "
                             "first")
        locations = mapillary.read_locations(locations_path)
        photos = {row["image_key"] for row in rows}
        locations_summary = {"file": str(locations_path), "sha256": _sha256_file(locations_path),
                             "located_photos": sum(1 for key in photos
                                                   if (locations.get(key) or {}).get("latitude") is not None)}
        rows = [row for row in rows if _located(locations.get(row["image_key"]), parts)]
    if percent < mapillary.FULL_SHARE:
        rows = [row for row in rows if photo_share_point(row["image_key"]) < percent / 100.0]

    frames = []
    for row in sorted(rows, key=lambda row: row["window_id"]):
        boxes, box_families, ignore = [], [], []
        for obj in row["objects"]:
            if obj["status"] == CLEAN and obj["family"] in wanted:
                boxes.append(list(obj["box"]))
                box_families.append(obj["family"])
            else:
                ignore.append(list(obj["box"]))
        frames.append({
            "image_path": row["image_path"], "image_sha256": row["image_sha256"], "width": row["width"],
            "height": row["height"], "split": TRAIN, "route_id": ROUTE, "segment_index": 0,
            "decoded_frame_index": 0, "boxes": boxes, "box_families": box_families, "ignore_boxes": ignore,
            "annotation_ids": [], "explicit_negative": False, "label_basis": LABEL_BASIS, "source": SOURCE,
            "window_id": row["window_id"], "image_key": row["image_key"],
            "designs": [obj["design"] for obj in row["objects"] if obj["status"] == CLEAN and obj["family"] in wanted],
        })
    before = {"windows": len(frames), "speed_boxes": sum(len(frame["boxes"]) for frame in frames)}
    if max_speed_boxes is not None:
        by_photo: dict[str, list[dict[str, Any]]] = {}
        for frame in frames:
            by_photo.setdefault(frame["image_key"], []).append(frame)
        order = sorted(by_photo, key=lambda key: min(hashlib.sha1(frame["window_id"].encode()).hexdigest()
                                                     for frame in by_photo[key]))
        kept, total = [], 0
        for key in order:
            boxes = sum(len(frame["boxes"]) for frame in by_photo[key])
            if total + boxes > max_speed_boxes:
                break
            kept += by_photo[key]
            total += boxes
        frames = sorted(kept, key=lambda frame: frame["window_id"])

    own = {frame.get("image_sha256") for frame in own_frames}
    for frame in frames:
        if not frame["image_sha256"] or not (data_root / frame["image_path"]).is_file():
            raise ValueError(f"mapillary_window_missing: {frame['image_path']} (cut the windows again: "
                             "helpers/mapillary_windows.sh)")
        if frame["image_sha256"] in own:
            raise ValueError(f"duplicate: the Mapillary window {frame['window_id']} has the pixels of one of our "
                             "frames")
    families_count, designs = Counter(), Counter()
    for frame in frames:
        families_count.update(frame["box_families"])
        designs.update(design or "none" for design in frame.pop("designs"))
    summary = {
        "file": str(path), "file_sha256": _sha256_file(path), "header_version": header.get("version"),
        "region": mapillary.region_text(region), "share": percent,
        "selection_sha256": hashlib.sha256("\n".join(sorted(frame["window_id"] for frame in frames)).encode()
                                           ).hexdigest(),
        "frames": len(frames), "photos": len({frame["image_key"] for frame in frames}),
        "boxes_by_family": dict(sorted(families_count.items())),
        "ignore_boxes": sum(len(frame["ignore_boxes"]) for frame in frames),
        "by_design": dict(sorted(designs.items())),
        "max_speed_boxes": max_speed_boxes, "before_cap": before,
        "licence": LICENCE,
    }
    if locations_summary is not None:
        summary["locations"] = locations_summary
    return frames, summary


_STATE_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def windows_state(data_root: Path) -> dict[str, Any]:
    """Whether the windows are cut, for the panel: {'available', 'windows', 'photos', 'boxes_by_family', 'reason'}
    of the train split (what a plan takes; its boxes the speed signs only, as a plan's, also from a header of the
    first cut of 3 Nov 2026, which counted the other signs among them). Read again only when the file changes."""
    path = Path(data_root) / WINDOWS_FILE
    try:
        stat = path.stat()
    except OSError:
        return {"available": False, "windows": 0, "photos": 0, "boxes_by_family": {},
                "reason": "the Mapillary windows are not cut yet (helpers/mapillary_windows.sh)"}
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _STATE_CACHE.get(str(path))
    if cached and cached[0] == stamp:
        return dict(cached[1])
    try:
        header, _ = read_windows(path)
    except (OSError, ValueError):
        state = {"available": False, "windows": 0, "photos": 0, "boxes_by_family": {},
                 "reason": "the Mapillary windows file is not readable"}
    else:
        train = (header.get("counts") or {}).get(TRAIN) or {}
        windows = int(train.get("windows") or 0)
        state = {"available": windows > 0, "windows": windows, "photos": int(train.get("photos") or 0),
                 "boxes_by_family": {family: count for family, count in (train.get("boxes_by_family") or {}).items()
                                     if family in SPEED_FAMILIES},
                 "reason": "" if windows else "the Mapillary windows file has no train window"}
    _STATE_CACHE[str(path)] = (stamp, state)
    return dict(state)


_REGIONS_CACHE: dict[str, Any] = {}


def windows_regions_state(data_root: Path, repo: Path | None = None) -> dict[str, Any]:
    """What the region choice of Train › Cutter offers, in the shape of mapillary.regions_state: whether the positions
    were fetched, and the speed signs of the train windows (its 'crops') of each continent and country, most first, by
    the position of their photo (locations.jsonl, as plan_frames reads it). Read again only when a file changed."""
    from nnslr_tools.gps import CONTINENTS, world_names
    data_root = Path(data_root)
    path, where = data_root / WINDOWS_FILE, data_root / mapillary.LOCATIONS_FILE
    stamp = (str(data_root), mapillary._stamp(path), mapillary._stamp(where))
    if _REGIONS_CACHE.get("stamp") != stamp:
        try:
            rows = [row for row in read_windows(path)[1] if row.get("split") == TRAIN] if path.is_file() else []
        except (OSError, ValueError):
            rows = []
        locations = mapillary.read_locations(where)
        state: dict[str, Any] = {"fetched": where.is_file(), "crops": 0, "located_crops": 0}
        by_continent: Counter = Counter()
        by_country: Counter = Counter()
        continent_of_country: dict[str, str | None] = {}
        photos: set[str] = set()
        placed: set[str] = set()
        for row in rows:
            signs = sum(1 for obj in row.get("objects") or ()
                        if obj.get("status") == CLEAN and obj.get("family") in SPEED_FAMILIES)
            photos.add(row["image_key"])
            state["crops"] += signs
            location = locations.get(row["image_key"])
            if not location or location.get("latitude") is None:
                continue
            placed.add(row["image_key"])
            state["located_crops"] += signs
            if location.get("continent"):
                by_continent[location["continent"]] += signs
            if location.get("country"):
                by_country[location["country"]] += signs
                continent_of_country.setdefault(location["country"], location.get("continent"))
        names = world_names() if by_country else {}
        state.update({
            "photos": len(photos), "located_photos": len(placed),
            "fetched_photos": sum(1 for key in photos if key in locations),
            "continents": [{"key": key, "name": CONTINENTS.get(key, key), "crops": count}
                           for key, count in sorted(by_continent.items(), key=lambda item: (-item[1], item[0]))],
            "countries": [{"code": code, "name": names.get(code, code), "continent": continent_of_country.get(code),
                           "crops": count}
                          for code, count in sorted(by_country.items(), key=lambda item: (-item[1], item[0]))],
        })
        _REGIONS_CACHE.clear()
        _REGIONS_CACHE.update({"stamp": stamp, "state": state})
    state = dict(_REGIONS_CACHE["state"])
    state["token"] = mapillary.token_ready(repo)
    state["ids_table"] = (data_root / mapillary.IMAGE_IDS_FILE).is_file()
    return state


def _uses_mapillary(document: Any) -> bool:
    return isinstance(document, dict) and bool(document.get("mapillary") or document.get("mapillary_only"))


def _has_documents(folder: Path) -> bool:
    return any((folder / name).is_file() for name in ("result.json", "training-plan.json"))


def mapillary_lineage(weights: Path) -> dict[str, Any]:
    """Whether a network learnt from MTSD anywhere in its chain of starts, for a package or a release that must name
    the licence: {'uses_mapillary', 'runs' (the folders that did, nearest first), 'licence'}. Reads the result.json and
    training-plan.json beside each weights file and follows their start.path (at most 10 steps, each file once).
    The weights themselves keep it too (signnet-best.pt's 'uses_mapillary'), for a copy without the folders. No
    torch."""
    runs: list[str] = []
    seen: set[str] = set()
    current: Path | None = Path(weights)
    for _ in range(10):
        if current is None or str(current) in seen:
            break
        seen.add(str(current))
        documents = []
        for name in ("result.json", "training-plan.json"):
            try:
                documents.append(json.loads((current.parent / name).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        if any(_uses_mapillary(document) for document in documents):
            runs.append(str(current.parent))
        current = None
        for document in documents:
            start = document.get("start") if isinstance(document, dict) else None
            if isinstance(start, dict) and start.get("path"):
                current = Path(start["path"])
                if _uses_mapillary(start) or (start.get("uses_mapillary") and not _has_documents(current.parent)):
                    # What the run recorded of its start counts even when that run's folder is gone: the start's own
                    # windows, or a Mapillary stage further down its chain (signnet_start's 'uses_mapillary'; with
                    # the folder there, the chain goes on to that stage itself).
                    runs.append(str(current.parent))
                break
    runs = list(dict.fromkeys(runs))
    return {"uses_mapillary": bool(runs), "runs": runs, "licence": LICENCE}
