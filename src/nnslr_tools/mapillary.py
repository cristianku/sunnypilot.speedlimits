"""The signs of the Mapillary Traffic Sign Dataset (MTSD) as extra training crops of the classifier.

MTSD (https://www.mapillary.com/dataset/trafficsign, CC BY-NC-SA 4.0) is about 52,000 street photos from all over the
world with every traffic sign boxed and named (``regulatory--maximum-speed-limit-50--g1``). Its signs that are ours
are cut once (``nnslr mapillary-crops``, helpers/mapillary.sh) into small crops with a margin around the sign, read
straight from the zips of the download (on the server they sit on the TrueNAS share /data/mapillary):

    <data root>/external/mapillary/crops/<object key>.jpg
    <data root>/external/mapillary/crops.jsonl    a header line, then one line per crop: path, box, our label

The classifier's training takes them when asked (``nnslr train --task reader --classes v2 --mapillary
<crops.jsonl>``, scripts/run_classifier.sh with NNSLR_CLASSIFIER_MAPILLARY=1, the Mapillary switch of Train ›
Classifier): in train only, never in validation or test, so every measure stays on our own drives. Only the labels
of MAPPED_LABELS and SPEED_LABEL come in: a speed sign of any design, its end, and the kinds of
other signs our classifier already has as classes (a kind we fold into other_sign gets no Mapillary crops: they
would make a class of photos that are not ours). At most PER_LABEL crops of each label, so that Mapillary helps the
rare classes without burying the common ones.

Or in two stages (``--mapillary-only``, then ``--init-from``; NNSLR_CLASSIFIER_MAPILLARY=pretrain, the "Mapillary,
then ours" choice of Train › Classifier): first a classifier is trained on the Mapillary crops alone, every one of
them (no cap), an other sign whose kind is not a class of ours as other_sign (what our plan folds it into), and our
validation signs of those classes choosing its epoch; then the classifier is trained on our crops alone, "not a
sign" included, starting from those weights and keeping their classes.

MTSD's photos are not the comma's road camera: other lenses, mostly daylight, often closer. The crops are scaled so
that the sign is at most MAX_SIGN_SIDE pixels, the size of a near sign in a comma frame.

Which part of MTSD comes in is chosen in Train › Classifier (``nnslr train --mapillary-region``,
``--mapillary-share``): the photos of some continents or countries only, and a share of them. MTSD's annotations say
nothing of where a photo was taken (width, height, ispano and the signs only), so its position comes from Mapillary
(``nnslr mapillary-locations``, helpers/mapillary_locations.sh), once, into LOCATIONS_FILE. MTSD names its photos by
the image keys of Mapillary's old API (v3, 22 characters); the Graph API of today (v4) takes numeric ids only and
has no lookup by an old key, so the keys go through a table of old keys and new ids (IMAGE_IDS_URL, made by a
Mapillary user from the redirect of the old app links; a key it has no id for has no position). In training_records
the crops are chosen in this order: the labels that are ours, the region (a crop whose photo has no position is
outside every region but "all"), the share (a fixed pseudo-random part of what is left, by a hash of the sign's key:
the same share always takes the same crops, a larger share takes them all and more), and at last at most PER_LABEL
of each label, so that the cap keeps the classes balanced whatever was chosen before.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

from nnslr_tools.annotations import atomic_write

CROPS_KIND = "mapillary_crops"
CROPS_VERSION = 1
# Where helpers/mapillary.sh writes the crops, under the data root.
CROPS_DIR = Path("external") / "mapillary" / "crops"
CROPS_FILE = Path("external") / "mapillary" / "crops.jsonl"
# The label basis of a Mapillary crop in a training plan (a label of MTSD's annotators, not of our reviews).
LABEL_BASIS = "mapillary"
# At most this many crops of one label in a training plan (the first by a hash of the object key: always the same).
PER_LABEL = 300
# The label of an other sign whose kind is not a class (nnslr_tools.training.OTHER_SIGN).
OTHER_SIGN = "other_sign"
# A sign smaller than this (its longer side, in MTSD pixels) is left out: too few pixels to read.
MIN_SIGN_SIDE = 16
# A crop is scaled down until its sign is at most this many pixels on its longer side.
MAX_SIGN_SIDE = 96
# The margin cut around the sign, as a share of its side: room for the box jitter of the training (at most 1.2 x
# the side and 0.1 of it off centre, nnslr_tools.training.CLASSIFIER_BOX_JITTER).
MARGIN = 0.5
# Where each photo was taken (nnslr mapillary-locations): one line per MTSD image key, the last one of a key wins.
LOCATIONS_FILE = Path("external") / "mapillary" / "locations.jsonl"
# The old image keys of MTSD and their ids of today: a CSV "image_key,image_id" (a key it could not resolve has
# itself as its id). Made by a Mapillary user in March 2026 from the redirect of the old app links
# (https://forum.mapillary.com/t/mapillary-traffic-sign-dataset-keys/10282): 32,524 of the 41,909 keys of MTSD's
# fully annotated train and val have a numeric id. Downloaded once into IMAGE_IDS_FILE.
IMAGE_IDS_URL = ("https://raw.githubusercontent.com/viajerus/mapillary_keys_ids/refs/heads/main/"
                 "mapillary_image_key_image_id.csv")
IMAGE_IDS_FILE = Path("external") / "mapillary" / "image_ids.csv"
# The Graph API (v4) of Mapillary: several images in one request (?ids=), the client token of the app in the
# Authorization header, never in a URL or a log.
GRAPH_URL = "https://graph.mapillary.com/"
TOKEN_ENV = "MAPILLARY_TOKEN"
IDS_PER_REQUEST = 50
# At most this many requests a second, well under the 60,000 a minute Mapillary allows an app.
REQUESTS_PER_SECOND = 4
# Every crop, or the photos of some continents or countries (parse_region).
ALL_REGIONS = "all"
# A random share of the crops chosen by region, in percent; 100: all of them.
FULL_SHARE = 100.0
# What makes the share's hash differ from the cap's (_order), so that the two choices are independent.
SHARE_SALT = "mapillary-share-1:"
# MTSD's flags of a sign that make it a poor example: a sign MTSD itself could not name (ambiguous), a fake one
# (dummy: a sign on an advert), cut by the photo's edge, on the back of something (exterior), inside another sign.
SKIPPED_PROPERTIES = ("ambiguous", "dummy", "out-of-frame", "exterior", "included")

# A speed sign of MTSD, of any design: g1 is the European round one, the others American and Canadian rectangles, in
# miles per hour some of them. The label keeps the written digits; only g1 says they are km/h.
SPEED_LABEL = re.compile(r"^regulatory--maximum-speed-limit-(\d+)--g\d+$")
SPEED_VALUES = frozenset(range(5, 251))  # written digits; design never determines mph
# MTSD's labels that are signs of ours, each checked on its crops: the European design only.
MAPPED_LABELS = {
    "regulatory--end-of-maximum-speed-limit-30--g2": "cancellation",
    "regulatory--yield--g1": "other_sign:give_way",
    "regulatory--stop--g1": "other_sign:stop",
    "regulatory--no-entry--g1": "other_sign:no_entry",
    "regulatory--priority-road--g4": "other_sign:main_road",
    "regulatory--end-of-priority-road--g1": "other_sign:main_road_end",
    "regulatory--roundabout--g1": "other_sign:roundabout",
    "regulatory--height-limit--g1": "other_sign:height_limit",
    "regulatory--weight-limit--g1": "other_sign:weight_limit",
    "regulatory--go-straight--g1": "other_sign:ahead_only",
    "information--pedestrians-crossing--g1": "other_sign:pedestrian_crossing",
    "information--motorway--g1": "other_sign:motorway_start",
    "information--end-of-motorway--g1": "other_sign:motorway_end",
    "information--limited-access-road--g1": "other_sign:expressway_start",
    "information--end-of-limited-access-road--g1": "other_sign:expressway_end",
    "information--end-of-built-up-area--g1": "other_sign:town_exit",
    "warning--other-danger--g1": "other_sign:danger_other",
    "warning--roadworks--g1": "other_sign:roadworks",
}


def our_label(mtsd_label: str) -> str | None:
    """Our classifier label of an MTSD label, or None when it is not one of ours (or not its European design)."""
    match = SPEED_LABEL.match(mtsd_label)
    if match:
        value = int(match.group(1))
        return f"maximum_speed:{value}" if value in SPEED_VALUES else None
    return MAPPED_LABELS.get(mtsd_label)


def _region(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The pixels cut around a sign: its box and MARGIN of its side on every side, inside the photo."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    pad = MARGIN * side
    return (max(0, int(x1 - pad)), max(0, int(y1 - pad)), min(width, int(x2 + pad + 1)), min(height, int(y2 + pad + 1)))


def _usable(item: dict[str, Any], ispano: bool) -> tuple[list[float] | None, str | None]:
    """The box of an MTSD sign worth a crop, or None and why not."""
    properties = item.get("properties") or {}
    for name in SKIPPED_PROPERTIES:
        if properties.get(name):
            return None, name
    bbox = item.get("bbox") or {}
    if "cross_boundary" in bbox or ispano:
        return None, "panorama"
    try:
        box = [float(bbox["xmin"]), float(bbox["ymin"]), float(bbox["xmax"]), float(bbox["ymax"])]
    except (KeyError, TypeError, ValueError):
        return None, "no_box"
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "no_box"
    if max(box[2] - box[0], box[3] - box[1]) < MIN_SIGN_SIDE:
        return None, "too_small"
    return box, None


class _Files:
    """The ``<key>.json`` and ``<key>.jpg`` files under some folders, loose or inside the .zip archives of MTSD's
    download (read from the zip without unzipping it: on a network share a hundred thousand small files cost far
    more than reading each photo once). A zip that does not open (still being copied) is left out and named; a
    hidden file (the copy in flight, .part) and one that failed its md5 check (UNVERIFIED-*) are not looked at."""

    SUFFIXES = (".json", ".jpg")

    def __init__(self, folders: list[Path], log) -> None:
        self.where: dict[str, dict[str, tuple[Path | None, str]]] = {suffix: {} for suffix in self.SUFFIXES}
        self.broken: list[str] = []
        self._zips: dict[Path, zipfile.ZipFile] = {}
        for folder in dict.fromkeys(folders):
            for path in sorted(folder.rglob("*")):
                suffix = path.suffix.lower()
                if path.name.startswith((".", "UNVERIFIED-")) or not path.is_file():
                    continue
                if suffix in self.where:
                    self.where[suffix].setdefault(path.stem, (None, str(path)))
                elif suffix == ".zip" and path not in self._zips:
                    try:
                        archive = zipfile.ZipFile(path)
                    except (OSError, zipfile.BadZipFile) as error:
                        log(f"mapillary: {path} left out, it does not open ({error})")
                        self.broken.append(str(path))
                        continue
                    self._zips[path] = archive
                    for name in archive.namelist():
                        member = PurePosixPath(name)
                        if member.suffix.lower() in self.where and not member.name.startswith("._"):
                            self.where[member.suffix.lower()].setdefault(member.stem, (path, name))

    def keys(self, suffix: str) -> list[str]:
        return sorted(self.where[suffix])

    def has(self, key: str, suffix: str) -> bool:
        return key in self.where[suffix]

    def read(self, key: str, suffix: str) -> bytes:
        archive, name = self.where[suffix][key]
        return Path(name).read_bytes() if archive is None else self._zips[archive].read(name)

    def close(self) -> None:
        for archive in self._zips.values():
            archive.close()


def make_crops(annotations: Path, images: Path, data_root: Path, *, log=print) -> dict[str, Any]:
    """Cut the MTSD signs that are ours (our_label) from the photos (``<key>.jpg``) under ``images`` named by the
    annotations (``<key>.json``) under ``annotations`` into CROPS_DIR, and write CROPS_FILE. Both are folders
    searched with their subfolders, loose files or MTSD's .zip archives as downloaded (they may be the same
    folder). A photo whose annotation has no sign of ours is never opened; one without its photo is counted, not
    an error (the annotations come in one download, the photos in several). Run again, it resumes: a photo whose
    crops are all cut already is not read again. Returns the header written."""
    from PIL import Image

    annotations, images, data_root = Path(annotations), Path(images), Path(data_root)
    if not annotations.is_dir():
        raise ValueError(f"mapillary_missing: no annotations folder {annotations}")
    if not images.is_dir():
        raise ValueError(f"mapillary_missing: no photos folder {images}")
    out_dir = data_root / CROPS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    reused = 0
    files = _Files([annotations, images], log)
    try:
        keys = files.keys(".json")
        for number, stem in enumerate(keys, 1):
            if number % 2000 == 0:
                log(f"mapillary: {number}/{len(keys)} annotations, {len(rows)} crops")
            try:
                document = json.loads(files.read(stem, ".json"))
            except (OSError, ValueError, zipfile.BadZipFile, zlib.error):
                skipped["unreadable_annotation"] += 1
                continue
            if not isinstance(document, dict) or "objects" not in document:
                continue  # not an annotation (a README, a split list)
            wanted = []
            for item in document.get("objects") or []:
                label = our_label(str(item.get("label", "")))
                if label is None:
                    skipped["not_ours"] += 1
                    continue
                box, why = _usable(item, bool(document.get("ispano")))
                if box is None:
                    skipped[why] += 1
                    continue
                key = str(item.get("key") or hashlib.sha1(f"{stem}{box}".encode()).hexdigest()[:22])
                wanted.append((item, label, box, key))
            if not wanted:
                continue
            if not files.has(stem, ".jpg"):
                skipped["photo_not_downloaded"] += len(wanted)
                continue
            width, height = document.get("width"), document.get("height")
            image = None
            if not (isinstance(width, int) and isinstance(height, int)
                    and all((out_dir / (key + ".jpg")).is_file() for *_, key in wanted)):
                try:
                    with Image.open(io.BytesIO(files.read(stem, ".jpg"))) as source:
                        image = source.convert("RGB")
                except (OSError, zipfile.BadZipFile, zlib.error):
                    skipped["photo_unreadable"] += len(wanted)
                    continue
                width, height = image.width, image.height
            else:
                reused += len(wanted)
            for item, label, box, key in wanted:
                left, top, right, bottom = _region(box, width, height)
                scale = min(1.0, MAX_SIGN_SIDE / max(box[2] - box[0], box[3] - box[1]))
                name = CROPS_DIR / (key + ".jpg")
                if image is not None:
                    crop = image.crop((left, top, right, bottom))
                    if scale < 1.0:
                        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                                           Image.LANCZOS)
                    crop.save(data_root / name, quality=92)
                rows.append({
                    "image_path": name.as_posix(),
                    "bbox_xyxy": [round((box[0] - left) * scale, 2), round((box[1] - top) * scale, 2),
                                  round((box[2] - left) * scale, 2), round((box[3] - top) * scale, 2)],
                    "label": label, "mtsd_label": item.get("label"), "mtsd_image": stem, "mtsd_object": key,
                    "occluded": bool((item.get("properties") or {}).get("occluded")),
                    "original_annotation": item, "sign_unit": "unknown",
                    "sign_value": int(label.partition(':')[2]) if label.partition(':')[2].isdigit() else None,
                    "design": str(item.get('label','')).rsplit('--',1)[-1],
                })
        broken = list(files.broken)
    finally:
        files.close()
    if reused:
        log(f"mapillary: {reused} crops already cut, kept")
    header = {"kind": CROPS_KIND, "version": CROPS_VERSION, "annotations": str(annotations), "images": str(images),
              "crops": len(rows), "by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
              "skipped": dict(sorted(skipped.items())), "license": "CC BY-NC-SA 4.0, Mapillary Traffic Sign Dataset"}
    if broken:
        header["zips_left_out"] = broken
    lines = [json.dumps(header, sort_keys=True)] + [json.dumps(row, sort_keys=True) for row in rows]
    atomic_write(data_root / CROPS_FILE, ("\n".join(lines) + "\n").encode("utf-8"))
    return header


def read_crops(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the rows of a crops file of make_crops."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"mapillary_invalid: empty file {path}")
    header = json.loads(lines[0])
    if header.get("kind") != CROPS_KIND or header.get("version") != CROPS_VERSION:
        raise ValueError(f"mapillary_invalid: {path} is not a crops file of nnslr mapillary-crops")
    return header, [json.loads(line) for line in lines[1:]]


def crops_state(data_root: Path) -> dict[str, Any]:
    """Whether the crops are ready, for the panel: {'available', 'crops', 'labels', 'reason'}."""
    path = Path(data_root) / CROPS_FILE
    if not path.is_file():
        return {"available": False, "crops": 0, "labels": 0, "reason": "Mapillary is not prepared (helpers/mapillary.sh)"}
    try:
        header, _ = read_crops(path)
    except (OSError, ValueError):
        return {"available": False, "crops": 0, "labels": 0, "reason": "the Mapillary crops file is not readable"}
    crops = int(header.get("crops") or 0)
    return {"available": crops > 0, "crops": crops, "labels": len(header.get("by_label") or {}),
            "reason": "" if crops else "no Mapillary sign of ours was found"}


def _order(row: dict[str, Any]) -> str:
    return hashlib.sha1(str(row.get("mtsd_object")).encode()).hexdigest()


def _share_point(row: dict[str, Any]) -> float:
    """Where a crop falls in [0, 1) for the share: a crop is in a share of p percent when this is below p / 100."""
    digest = hashlib.sha1((SHARE_SALT + str(row.get("mtsd_object"))).encode()).hexdigest()
    return int(digest[:13], 16) / float(16 ** 13)


def parse_region(text: str | None) -> frozenset[str] | None:
    """The region of a classifier plan: None for every crop ("all", empty), else the continents (keys of
    nnslr_tools.gps.CONTINENTS: europe, north_america...) and countries (ISO codes: CH, DE...) whose photos come in,
    from a comma-separated text. ValueError on anything else."""
    from nnslr_tools.gps import CONTINENTS, world_names
    if text is None or not str(text).strip() or str(text).strip().lower() == ALL_REGIONS:
        return None
    parts = set()
    for part in str(text).split(","):
        part = part.strip()
        if part.lower() in CONTINENTS:
            parts.add(part.lower())
        elif re.fullmatch(r"[A-Za-z]{2}", part) and part.upper() in world_names():
            parts.add(part.upper())
        elif part:
            raise ValueError(f"mapillary_region: {part!r} is neither a continent ({', '.join(CONTINENTS)}) "
                             f"nor a country code (CH, DE...)")
    if not parts:
        return None
    return frozenset(parts)


def region_text(text: str | None) -> str:
    """The region in one fixed spelling, for the plan and the resume flags: "all", or its continents then its
    countries, sorted ("europe,US")."""
    parts = parse_region(text)
    if parts is None:
        return ALL_REGIONS
    return ",".join(sorted(part for part in parts if part.islower()) + sorted(part for part in parts if part.isupper()))


def check_share(share: float | int | str | None) -> float:
    """The share of the crops, in percent: more than 0, at most 100 (FULL_SHARE when None)."""
    if share is None:
        return FULL_SHARE
    try:
        value = float(share)
    except (TypeError, ValueError):
        raise ValueError(f"mapillary_share: {share!r} is not a percentage") from None
    if not 0 < value <= 100:
        raise ValueError(f"mapillary_share: {share!r} must be more than 0 and at most 100")
    return round(value, 2)


def locations_path_of(crops_path: Path) -> Path:
    """The locations file next to a crops file (LOCATIONS_FILE next to CROPS_FILE)."""
    return Path(crops_path).parent / LOCATIONS_FILE.name


def read_locations(path: Path) -> dict[str, dict[str, Any]]:
    """{MTSD image key: its line} of a locations file, the last line of a key winning; a broken line (a write cut
    short) is skipped. {} without the file."""
    found: dict[str, dict[str, Any]] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return found
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("mtsd_image"), str):
            found[row["mtsd_image"]] = row
    return found


def _in_region(location: dict[str, Any] | None, region: frozenset[str]) -> bool:
    if not location or location.get("latitude") is None:
        return False
    return location.get("continent") in region or location.get("country") in region


def training_records(path: Path, classes: set[str], other_kinds: set[str], *, region: str | None = ALL_REGIONS,
                     share: float | None = FULL_SHARE, capped: bool = True,
                     fold_other_kinds: bool = False) -> tuple[list[dict[str, Any]], dict]:
    """The train records of the crops file ``path`` for a classifier plan: the speed signs and their end always
    (a value we have never seen becomes a class: the comma reads any value), an other sign only when its kind is
    one of ``other_kinds`` (our kinds that are classes); then only the photos of ``region`` (parse_region; the
    positions of the locations file next to the crops file), then ``share`` percent of what is left (a fixed choice
    by a hash of the sign), then at most PER_LABEL of each label (not ``capped``: every one). With
    ``fold_other_kinds`` (the pretraining on Mapillary alone, nnslr train --mapillary-only) an other sign whose kind is not a class
    of ours comes in as other_sign, as our plan folds a kind too rare to learn, instead of staying out. Returns the
    records and a summary for the plan (file, sha256, region, share, the locations file, crops by label, what was
    left out or folded, and the sha256 of the signs chosen, which a resumed run must find again)."""
    header, rows = read_crops(path)
    wanted = parse_region(region)
    share = check_share(share)
    locations_file = None
    locations: dict[str, dict[str, Any]] = {}
    if wanted is not None:
        where = locations_path_of(path)
        if not where.is_file():
            raise ValueError(f"mapillary_locations_missing: {where} (fetch the positions first: "
                             "helpers/mapillary_locations.sh)")
        locations = read_locations(where)
        from nnslr_tools.training import _sha256
        locations_file = {"file": str(where), "sha256": _sha256(where),
                          "located_photos": sum(1 for row in locations.values() if row.get("latitude") is not None)}
    by_label: dict[str, list[dict[str, Any]]] = {}
    left_out: Counter = Counter()
    folded: Counter = Counter()
    eligible = in_region = 0
    for row in rows:
        label = row.get("label")
        if not isinstance(label, str):
            left_out["no_label"] += 1
            continue
        if label.startswith("other_sign:") and label not in other_kinds:
            if not fold_other_kinds:
                left_out["kind_not_a_class_of_ours"] += 1
                continue
            folded[label] += 1
            row = {**row, "kind": label.partition(":")[2], "label": OTHER_SIGN}
            label = OTHER_SIGN
        eligible += 1
        if wanted is not None:
            location = locations.get(str(row.get("mtsd_image")))
            if not location or location.get("latitude") is None:
                left_out["photo_without_position"] += 1
                continue
            if not _in_region(location, wanted):
                left_out["outside_the_region"] += 1
                continue
        in_region += 1
        if share < FULL_SHARE and _share_point(row) >= share / 100.0:
            left_out["outside_the_share"] += 1
            continue
        by_label.setdefault(label, []).append(row)
    per_label = PER_LABEL if capped else None
    records = []
    for label, items in sorted(by_label.items()):
        items.sort(key=_order)
        kept = items if per_label is None else items[:per_label]
        left_out["over_the_per_label_limit"] += len(items) - len(kept)
        for row in kept:
            family, _, value = label.partition(":")
            records.append({
                "annotation_id": None, "split": "train", "image_path": row["image_path"], "image_sha256": None,
                "bbox_xyxy": [float(v) for v in row["bbox_xyxy"]], "label": label,
                "sign_family": family, "sign_type": (row.get("kind") or value or None) if family == "other_sign" else None,
                "value_state": "read" if value.isdigit() else None,
                "value_kph": int(value) if value.isdigit() and row.get("design", "g1") == "g1" else None,
                "sign_value": int(value) if value.isdigit() else None, "sign_unit": row.get('sign_unit','unknown'),
                "mtsd_label": row.get('mtsd_label'), "design": row.get('design'),
                "route_id": None, "segment_index": None, "decoded_frame_index": None,
                "label_basis": LABEL_BASIS, "source": LABEL_BASIS, "mtsd_object": row.get("mtsd_object"),
            })
    from nnslr_tools.training import _sha256
    chosen = hashlib.sha256("\n".join(sorted(str(r["mtsd_object"]) for r in records)).encode()).hexdigest()
    summary = {"file": str(path), "sha256": _sha256(Path(path)), "crops": len(records),
               "new_classes": sorted({r["label"] for r in records} - classes),
               "by_label": dict(sorted(Counter(r["label"] for r in records).items())),
               "left_out": dict(sorted(left_out.items())), "per_label": per_label,
               "region": region_text(region), "share": share, "locations": locations_file,
               "eligible": eligible, "in_region": in_region, "selection_sha256": chosen,
               "order": ("labels of ours, region, share" + (", per-label cap" if per_label is not None else ", no cap"))}
    if fold_other_kinds:
        summary["folded_into_other_sign"] = dict(sorted(folded.items()))
    return records, summary


# --- Where the photos were taken (nnslr mapillary-locations) -------------------------------------------------------

# The statuses of a line of the locations file; each is final (a network error writes nothing, so it is tried again).
LOCATED = "located"          # a position, and its country and continent when on land
NO_V4_ID = "no_v4_id"        # the key table has no id of today for this key (a photo deleted since, most likely)
NOT_FOUND = "not_found"      # Mapillary does not know the id (deleted)
NO_POSITION = "no_position"  # Mapillary knows the photo but gives no position
# Retries of a request refused for the moment (429, 5xx, no network), waiting twice as long each time; then the run
# stops and the next one resumes.
RETRIES = 5
FIRST_WAIT = 2.0


class LocationsError(RuntimeError):
    """The fetch cannot go on (a token Mapillary refuses, Mapillary out of reach): what was fetched stays."""


def read_image_ids(path: Path) -> dict[str, str | None]:
    """{old image key: its numeric id of today, or None when the table could not resolve it} of a key table
    (IMAGE_IDS_URL: a CSV with the columns image_key and image_id)."""
    found: dict[str, str | None] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_key", "image_id"} <= set(reader.fieldnames):
            raise ValueError(f"mapillary_ids_invalid: {path} has no columns image_key, image_id")
        for row in reader:
            key, image_id = (row.get("image_key") or "").strip(), (row.get("image_id") or "").strip()
            if key:
                found[key] = image_id if image_id.isdigit() else None
    return found


def download_image_ids(data_root: Path, url: str = IMAGE_IDS_URL, *, timeout: float = 120) -> Path:
    """Download the key table into IMAGE_IDS_FILE (once: a table already there is kept) and return its path."""
    path = Path(data_root) / IMAGE_IDS_FILE
    if path.is_file():
        return path
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
    text = payload.decode("utf-8")
    if not text.startswith("image_key,image_id"):
        raise ValueError(f"mapillary_ids_invalid: {url} is not a table image_key,image_id")
    atomic_write(path, payload)
    return path


def wanted_keys(data_root: Path, ids: dict[str, str | None]) -> list[str]:
    """The photos whose position is wanted: every key of the table (MTSD's train and val photos, so the positions can
    be fetched before the crops are cut) and every photo of the crops file."""
    keys = set(ids)
    crops = Path(data_root) / CROPS_FILE
    if crops.is_file():
        try:
            keys.update(str(row["mtsd_image"]) for row in read_crops(crops)[1] if row.get("mtsd_image"))
        except (OSError, ValueError):
            pass
    return sorted(keys)


def token_ready(repo: Path | None = None) -> bool:
    """Whether a Mapillary token is set: in the environment or as a MAPILLARY_TOKEN= line of the repository's .env.
    Only whether; its value is never read out of here."""
    if os.environ.get(TOKEN_ENV, "").strip():
        return True
    if repo is None:
        return False
    try:
        lines = (Path(repo) / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        name, _, value = line.strip().removeprefix("export ").partition("=")
        if name.strip() == TOKEN_ENV and value.strip().strip("'\""):
            return True
    return False


Fetch = Callable[[str, dict[str, str]], tuple[int, bytes]]


def _http_get(url: str, headers: dict[str, str]) -> tuple[int, bytes]:
    """GET ``url``: (status, body), an HTTP error too; OSError when Mapillary is out of reach."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _position(item: dict[str, Any]) -> tuple[float, float] | None:
    """(latitude, longitude) of a Graph API image: computed_geometry (where Mapillary's reconstruction put it) or
    else geometry (the camera's GPS); GeoJSON points are [longitude, latitude]."""
    for name in ("computed_geometry", "geometry"):
        geometry = item.get(name)
        if isinstance(geometry, dict) and geometry.get("type") == "Point":
            coordinates = geometry.get("coordinates") or []
            if len(coordinates) >= 2 and all(isinstance(value, (int, float)) for value in coordinates[:2]):
                return round(float(coordinates[1]), 6), round(float(coordinates[0]), 6)
    return None


class _Graph:
    """The Graph API requests of a fetch: at most REQUESTS_PER_SECOND, a refused request tried again later."""

    def __init__(self, token: str, fetch: Fetch, sleep: Callable[[float], None], clock: Callable[[], float], log):
        self._headers = {"Authorization": f"OAuth {token}", "Accept": "application/json"}
        self._fetch, self._sleep, self._clock, self._log = fetch, sleep, clock, log
        self._last = None
        self.requests = 0

    def get(self, path: str, query: dict[str, str]) -> tuple[int, Any]:
        url = GRAPH_URL + path + "?" + urllib.parse.urlencode(query)
        wait = FIRST_WAIT
        for attempt in range(RETRIES + 1):
            if self._last is not None:
                gap = 1.0 / REQUESTS_PER_SECOND - (self._clock() - self._last)
                if gap > 0:
                    self._sleep(gap)
            self._last = self._clock()
            self.requests += 1
            try:
                status, body = self._fetch(url, self._headers)
            except OSError as error:
                status, body = None, str(error).encode()
            if status in (401, 403):
                raise LocationsError(f"mapillary_token_refused: Mapillary answered {status}; check {TOKEN_ENV} in .env")
            if status is not None and status != 429 and status < 500:
                try:
                    return status, json.loads(body or b"null")
                except ValueError:
                    return status, None
            if attempt < RETRIES:
                self._log(f"mapillary-locations: Mapillary answered {status or 'nothing'}, again in {wait:.0f} s")
                self._sleep(wait)
                wait *= 2
        raise LocationsError(f"mapillary_unreachable: Mapillary answered {status or 'nothing'} {RETRIES + 1} times; "
                             "run again later, it resumes")


FIELDS = "id,geometry,computed_geometry"
# What the Graph API answers for an id it does not have (a photo deleted since).
GONE = (400, 404, 410)


def _locate_batch(graph: _Graph, ids: list[str]) -> dict[str, dict[str, Any] | None]:
    """{id: its Graph API image, or None when Mapillary does not know it} of a batch: one request for all; when the
    batch is refused as a whole (an id of it deleted), one request each."""
    status, payload = graph.get("", {"ids": ",".join(ids), "fields": FIELDS})
    if status == 200 and isinstance(payload, dict) and "error" not in payload:
        return {image_id: payload.get(image_id) if isinstance(payload.get(image_id), dict) else None for image_id in ids}
    found: dict[str, dict[str, Any] | None] = {}
    for image_id in ids:
        status, payload = graph.get(image_id, {"fields": FIELDS})
        if status == 200 and isinstance(payload, dict) and "error" not in payload:
            found[image_id] = payload
        elif status in GONE:
            found[image_id] = None
        else:
            raise LocationsError(f"mapillary_unexpected: Mapillary answered {status} for one photo; nothing of this "
                                 "batch was written")
    return found


def _append(path: Path, lines: list[dict[str, Any]]) -> None:
    """Add lines to the locations file, on a line of their own even after a write cut short."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        if handle.tell():
            with path.open("rb") as reader:
                reader.seek(-1, os.SEEK_END)
                if reader.read(1) != b"\n":
                    handle.write(b"\n")
        handle.write("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def fetch_locations(data_root: Path, *, token: str | None, ids_path: Path | None = None, again: bool = False,
                    limit: int | None = None, dry_run: bool = False, fetch: Fetch | None = None,
                    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                    log=print) -> dict[str, Any]:
    """Fetch where MTSD's photos were taken into LOCATIONS_FILE: for each wanted photo (wanted_keys) its id of today
    from the key table, its position from Mapillary's Graph API (IDS_PER_REQUEST a request, REQUESTS_PER_SECOND at
    most), its country and continent from our borders (nnslr_tools.gps.world_place). A line is written after each
    batch, so a run stopped resumes where it was: a photo with a line is not asked again (but one the table could not
    resolve is, once the table has its id), unless ``again`` (asked again, the new line winning over the old).
    ``limit``: at most that many photos (a first try with a handful). ``dry_run``: only count. Returns the counts."""
    from nnslr_tools.gps import world_place
    data_root = Path(data_root)
    ids_path = Path(ids_path) if ids_path else data_root / IMAGE_IDS_FILE
    if not ids_path.is_file():
        raise ValueError(f"mapillary_ids_missing: no key table {ids_path} (download it: nnslr mapillary-locations "
                         "without --ids does)")
    ids = read_image_ids(ids_path)
    path = data_root / LOCATIONS_FILE
    known = {} if again else read_locations(path)
    todo = [key for key in wanted_keys(data_root, ids)
            if key not in known or (known[key].get("status") == NO_V4_ID and ids.get(key))]
    if limit is not None:
        todo = todo[:max(0, limit)]
    counts: Counter = Counter()
    summary = {"file": str(path), "wanted": len(wanted_keys(data_root, ids)), "already": len(known), "todo": len(todo),
               "requests": 0}
    if dry_run:
        summary["requests"] = -(-sum(1 for key in todo if ids.get(key)) // IDS_PER_REQUEST)
        return summary
    if not token:
        raise LocationsError(f"mapillary_token_missing: no {TOKEN_ENV} (a client token of a Mapillary app, in .env)")
    graph = _Graph(token, fetch or _http_get, sleep, clock, log)
    without = [key for key in todo if not ids.get(key)]
    if without:
        _append(path, [{"mtsd_image": key, "image_id": None, "latitude": None, "longitude": None, "country": None,
                        "continent": None, "status": NO_V4_ID} for key in without])
        counts[NO_V4_ID] += len(without)
    with_id = [key for key in todo if ids.get(key)]
    try:
        for start in range(0, len(with_id), IDS_PER_REQUEST):
            batch = with_id[start:start + IDS_PER_REQUEST]
            found = _locate_batch(graph, [str(ids[key]) for key in batch])
            lines = []
            for key in batch:
                image_id = str(ids[key])
                item = found.get(image_id)
                position = _position(item) if item else None
                country, continent = world_place(*position, root=data_root) if position else (None, None)
                status = LOCATED if position else (NOT_FOUND if item is None else NO_POSITION)
                counts[status] += 1
                lines.append({"mtsd_image": key, "image_id": image_id,
                              "latitude": position[0] if position else None,
                              "longitude": position[1] if position else None,
                              "country": country, "continent": continent, "status": status})
            _append(path, lines)
            done = start + len(batch)
            if done % (IDS_PER_REQUEST * 20) == 0 or done == len(with_id):
                log(f"mapillary-locations: {done}/{len(with_id)} photos, {counts[LOCATED]} placed")
    finally:
        summary["requests"] = graph.requests
        summary.update({name: counts[name] for name in (LOCATED, NO_V4_ID, NOT_FOUND, NO_POSITION)})
    return summary


_REGIONS_CACHE: dict[str, Any] = {}


def _stamp(path: Path) -> tuple | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def regions_state(data_root: Path, repo: Path | None = None) -> dict[str, Any]:
    """What the region choice of Train › Classifier offers: whether the positions were fetched, how many crops have
    one, and the continents and countries with their crops (most first). Read again only when a file changed."""
    from nnslr_tools.gps import CONTINENTS, world_names
    data_root = Path(data_root)
    crops, where = data_root / CROPS_FILE, data_root / LOCATIONS_FILE
    stamp = (str(data_root), _stamp(crops), _stamp(where))
    if _REGIONS_CACHE.get("stamp") != stamp:
        state: dict[str, Any] = {"fetched": where.is_file(), "crops": 0, "located_crops": 0, "photos": 0,
                                 "located_photos": 0, "continents": [], "countries": []}
        try:
            rows = read_crops(crops)[1] if crops.is_file() else []
        except (OSError, ValueError):
            rows = []
        locations = read_locations(where)
        by_continent: Counter = Counter()
        by_country: Counter = Counter()
        continent_of_country: dict[str, str | None] = {}
        photos: set[str] = set()
        placed: set[str] = set()
        for row in rows:
            image = str(row.get("mtsd_image"))
            photos.add(image)
            location = locations.get(image)
            if not location or location.get("latitude") is None:
                continue
            placed.add(image)
            state["located_crops"] += 1
            if location.get("continent"):
                by_continent[location["continent"]] += 1
            if location.get("country"):
                by_country[location["country"]] += 1
                continent_of_country.setdefault(location["country"], location.get("continent"))
        names = world_names() if by_country else {}
        state.update({
            "crops": len(rows), "photos": len(photos), "located_photos": len(placed),
            "fetched_photos": sum(1 for row in locations.values() if row.get("status") != NO_V4_ID),
            "continents": [{"key": key, "name": CONTINENTS.get(key, key), "crops": count}
                           for key, count in sorted(by_continent.items(), key=lambda item: (-item[1], item[0]))],
            "countries": [{"code": code, "name": names.get(code, code), "continent": continent_of_country.get(code),
                           "crops": count}
                          for code, count in sorted(by_country.items(), key=lambda item: (-item[1], item[0]))],
        })
        _REGIONS_CACHE.clear()
        _REGIONS_CACHE.update({"stamp": stamp, "state": state})
    state = dict(_REGIONS_CACHE["state"])
    state["token"] = token_ready(repo)
    state["ids_table"] = (data_root / IMAGE_IDS_FILE).is_file()
    return state
