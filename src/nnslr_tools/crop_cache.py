"""The crops of a reader/classifier training, cut from their frames once (nnslr train --task reader).

A training epoch used to open and decode every full frame (a 1928 x 1208 JPEG) to cut one small crop out of it:
the loader workers were the bottleneck, and the GPU waited. Now every record of the run is cut once, before the
first epoch, into a cache of small uint8 arrays; an epoch only reads those. What a crop looks like does not change:

- a crop taken as it is (validation, and the train crops of a run without box jitter: the v1 reader, or v2 with
  --no-augment) is stored already cut as the training cuts it (crop_rectangle for v2, the integer box for v1) and
  resized to the input size, the very pixels the old path gave (Resize to the same size is a copy);
- a train crop whose box is jittered (v2 with augmentation) is stored as a window around its box, wide enough for
  the largest jitter (CLASSIFIER_BOX_JITTER: 1.2 times the box, moved by 0.1 of it), at the resolution of the frame;
  the jittered box is then cut from that window, rounded and kept inside the frame exactly as crop_rectangle does,
  so the crop is again the pixels of the frame. Only a box larger than MAX_BOX_SIDE input sizes is kept smaller (its
  window scaled down, the crop resampled from it): a near sign, downscaled to the input size anyway.

A frame of a drive at full definition is cut from the frame beside its picture (training.crop_source): rows and boxes
stay in the picture's pixels, so a crop has twice the pixels and only the stored window is finer than its rectangle.

The evaluation (evaluate-classifier, evaluate-chain) and the comma never use this cache: they cut from the frame.
Sign review shows the crops of a box as these cuts make them (review_crops, with no cache: the square by the zoom).

The cache is a folder of data/derived/crop-cache named by a hash of what it holds (the version of this format, the
input size, the class scheme, every image path, its sha256 when the dataset gives it, whether it has a frame at full
definition, the box and how it is stored, and the sha256 of the dataset and of the Mapillary crops): two runs of the same plan share it (the control and the
second stage of a two-stage run, for example), and a change of any of these makes a new one. Old folders can be
deleted at any time (rm -r data/derived/crop-cache): the next run cuts its crops again.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from nnslr_tools.media import full_picture
from nnslr_tools.training import CLASSIFIER_BOX_JITTER, _safe_image_path, crop_rectangle, crop_source

FORMAT_VERSION = 1
CACHE_FOLDER = Path("derived") / "crop-cache"
# How a record is stored: cut and resized (EXACT), or a window around its box for the jitter (WINDOW).
EXACT, WINDOW = 0, 1
# Half a window: this much of the box's side on each side of its centre, and WINDOW_PAD pixels more (the rounding of
# crop_rectangle and its smallest side). The largest jitter reaches 0.6 + 0.1 = 0.7 of the side.
WINDOW_HALF = 0.75
WINDOW_PAD = 2
# A window keeps the frame's resolution while its box is at most this many input sizes wide and high.
MAX_BOX_SIDE = 2
# The columns of index.npy, one row per record.
COLUMNS = ("offset", "height", "width", "left", "top", "right", "bottom", "frame_width", "frame_height", "kind")


def _check_jitter() -> None:
    high = CLASSIFIER_BOX_JITTER["scale"][1]
    if high / 2 + CLASSIFIER_BOX_JITTER["shift"] > WINDOW_HALF - 0.04:
        raise ValueError("the crop cache's window is narrower than the largest box jitter: widen WINDOW_HALF")


def _v1_rectangle(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The crop of the v1 reader: the box's integer part, as it always was (outside the frame is black)."""
    x1, y1, x2, y2 = (int(value) for value in box)
    return x1, y1, x2, y2


def rectangle_of(scheme: str):
    return crop_rectangle if scheme == "v2" else _v1_rectangle


def _window(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """The pixels of the frame kept around a box whose crops are jittered: every jittered crop_rectangle is inside."""
    x1, y1, x2, y2 = (float(value) for value in box)
    side_x, side_y = max(abs(x2 - x1), 1.0), max(abs(y2 - y1), 1.0)
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = math.floor(centre_x - WINDOW_HALF * side_x - WINDOW_PAD)
    top = math.floor(centre_y - WINDOW_HALF * side_y - WINDOW_PAD)
    right = math.ceil(centre_x + WINDOW_HALF * side_x + WINDOW_PAD)
    bottom = math.ceil(centre_y + WINDOW_HALF * side_y + WINDOW_PAD)
    # The crop of the box itself (crop_rectangle moves a box outside the frame inside it) is always in the window.
    exact = crop_rectangle(box, width, height)
    left, top = max(0, min(left, exact[0])), max(0, min(top, exact[1]))
    right, bottom = min(width, max(right, exact[2])), min(height, max(bottom, exact[3]))
    return left, top, right, bottom


def stored_crop(image, factor: int, kind: int, box: list[float], size: int, scheme: str = "v2"):
    """What the cache keeps of one record, cut from its crop source (training.crop_source: the image and how many of
    its pixels make one of the picture): (the PIL pixels, their rectangle in the picture's pixels). EXACT: the box cut
    as the training cuts it (crop_rectangle for v2, the integer box for v1) and resized to the input size. WINDOW: the
    pixels around the box that every jitter of it falls in (_window), at the source's resolution while the box is at
    most MAX_BOX_SIDE input sizes wide and high, else scaled down."""
    from PIL import Image

    width, height = image.width // factor, image.height // factor
    if kind == EXACT:
        region = rectangle_of(scheme)(box, width, height)
        return image.crop(tuple(value * factor for value in region)).resize((size, size), Image.BILINEAR), region
    region = _window(box, width, height)
    cut = tuple(value * factor for value in region)
    side_x = max(abs(float(box[2]) - float(box[0])), 1.0) * factor
    side_y = max(abs(float(box[3]) - float(box[1])), 1.0) * factor
    keep_x = min(1.0, MAX_BOX_SIDE * size / side_x)
    keep_y = min(1.0, MAX_BOX_SIDE * size / side_y)
    span_x, span_y = cut[2] - cut[0], cut[3] - cut[1]
    if keep_x == 1.0 and keep_y == 1.0:
        return image.crop(cut), region
    return image.resize((max(1, round(span_x * keep_x)), max(1, round(span_y * keep_y))), Image.BILINEAR,
                        box=cut), region


def _cut(task: tuple[str, str, int, str, list[tuple[int, int, list[float]]]]) -> list[tuple]:
    """The stored pixels of the records of one image (a loader process: Pillow only)."""
    import numpy as np

    root, image_path, size, scheme, items = task
    image, factor = crop_source(_safe_image_path(Path(root), image_path))
    width, height = image.width // factor, image.height // factor
    out = []
    for index, kind, box in items:
        pixels, region = stored_crop(image, factor, kind, box, size, scheme)
        array = np.ascontiguousarray(np.asarray(pixels, dtype=np.uint8))
        out.append((index, kind, array.tobytes(), array.shape[0], array.shape[1], region, width, height))
    return out


def window_crop(window, region: tuple[int, int, int, int], frame_size: tuple[int, int], box: list[float], size: int,
                paint=None):
    """The crop of ``box`` (a record's box, jittered) from the window stored around it (stored_crop, WINDOW: its PIL
    pixels and their rectangle in a frame of frame_size), cut as crop_rectangle cuts it from the frame. paint: see
    CropCache.crop."""
    from PIL import Image

    image = window
    left, top, right, bottom = region
    frame_width, frame_height = frame_size
    if paint is not None:
        to_x, to_y = image.width / (right - left), image.height / (bottom - top)
        image = paint(image, lambda frame_box: [(frame_box[0] - left) * to_x, (frame_box[1] - top) * to_y,
                                                (frame_box[2] - left) * to_x, (frame_box[3] - top) * to_y])
    x1, y1, x2, y2 = crop_rectangle(box, frame_width, frame_height)
    # Inside the window (always, by construction: the clamp only guards against rounding).
    x1, x2 = max(left, min(x1, right - 1)), max(left + 1, min(x2, right))
    y1, y2 = max(top, min(y1, bottom - 1)), max(top + 1, min(y2, bottom))
    if image.width == right - left and image.height == bottom - top:
        return image.crop((x1 - left, y1 - top, x2 - left, y2 - top))
    scale_x, scale_y = image.width / (right - left), image.height / (bottom - top)
    return image.resize((size, size), Image.BILINEAR,
                        box=((x1 - left) * scale_x, (y1 - top) * scale_y,
                             (x2 - left) * scale_x, (y2 - top) * scale_y))


def review_crops(image, factor: int, box: list[float], *, size: int, samples: int = 0) -> list:
    """The crops the v2 classifier trains on for a box of a picture, made by this module's own cuts from the picture's
    crop source (training.crop_source), for Sign review's square beside the zoom: first the crop without jitter (EXACT,
    the crop validation gives the classifier), then ``samples`` crops of the box jittered as the augmented
    training jitters it: the first epochs whose draw moves it (jitter_box with the draws of the record of index 0 in a
    run of seed 0; 1 epoch in 5 keeps the box as drawn, the crop above), each cut from the window as CropCache.crop
    cuts it, then the dataset's Resize to the input size, which on a PIL image is Pillow's bilinear resize). Not the
    rotation of up to 4 degrees nor the colours of the augmentation: those need torch. Each crop is a PIL image of
    size x size."""
    from PIL import Image

    crops = [stored_crop(image, factor, EXACT, box, size)[0]]
    if samples > 0:
        window, region = stored_crop(image, factor, WINDOW, box, size)
        frame_size = (image.width // factor, image.height // factor)
        for moved in moved_boxes(box, samples):
            crops.append(window_crop(window, region, frame_size, moved, size).resize((size, size), Image.BILINEAR))
    return crops


def moved_boxes(box: list[float], count: int) -> list[list[float]]:
    """The box as the augmented training jitters it in the first ``count`` epochs whose draw moves it (jitter_box, the
    draws of the record of index 0 in a run of seed 0): the same epochs for every box."""
    import random

    from nnslr_tools.training import _sample_seed, jitter_box

    drawn = [float(value) for value in box]
    out = []
    for epoch in range(50 * count):
        moved = jitter_box(box, random.Random(_sample_seed("box", 0, epoch, 0)))
        if [float(value) for value in moved] != drawn:
            out.append(moved)
            if len(out) == count:
                break
    return out


def cache_key(records: list[dict[str, Any]], kinds: list[int], full_frames: set[str], *, size: int, scheme: str,
              content: dict[str, Any]) -> str:
    """The name of the cache of these records stored this way (see the module docstring)."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"version": FORMAT_VERSION, "size": size, "scheme": scheme, "half": WINDOW_HALF,
                              "pad": WINDOW_PAD, "max_box_side": MAX_BOX_SIDE, "content": content},
                             sort_keys=True).encode())
    for record, kind in zip(records, kinds):
        full = ["full"] if record["image_path"] in full_frames else []
        digest.update(json.dumps([record["image_path"], record.get("image_sha256"),
                                  [float(value) for value in record["bbox_xyxy"]], kind, *full]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class CropCache:
    """The stored crops of one run's records (build_crop_cache): crop(index, box) gives the PIL crop of a record,
    its box jittered or not, as cutting it from the frame would."""

    def __init__(self, folder: Path, size: int, scheme: str):
        import numpy as np

        self.folder = Path(folder)
        self.size = size
        self.scheme = scheme
        self.index = np.load(self.folder / "index.npy")
        self._pixels = None

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getstate__(self):
        # A loader worker opens the pixels itself (a memory map, shared with every process through the page cache).
        state = dict(self.__dict__)
        state["_pixels"] = None
        return state

    def array(self, index: int):
        import numpy as np

        if self._pixels is None:
            self._pixels = np.memmap(self.folder / "pixels.bin", dtype=np.uint8, mode="r")
        offset, height, width = (int(value) for value in self.index[index, :3])
        return np.asarray(self._pixels[offset:offset + height * width * 3]).reshape(height, width, 3)

    def crop(self, index: int, box: list[float] | None = None, paint=None):
        """The crop of record ``index``: the stored one, or for a window the crop of ``box`` (the record's box,
        jittered) cut as crop_rectangle cuts it from the frame. paint (a window only): a function of the window (PIL)
        and of a function that turns a box of the frame into one of the window's pixels, that returns the window
        drawn over (synthetic_signs) before the crop is cut."""
        from PIL import Image

        row = self.index[index]
        image = Image.fromarray(self.array(index))
        if int(row[9]) == EXACT:
            return image
        region = tuple(int(value) for value in row[3:7])
        return window_crop(image, region, (int(row[7]), int(row[8])), box, self.size, paint)


def build_crop_cache(data_root: Path, records: list[dict[str, Any]], kinds: list[int], *, size: int,
                     scheme: str, content: dict[str, Any], workers: int = 0) -> tuple[CropCache, dict[str, Any]]:
    """The cache of these records (kinds: EXACT or WINDOW for each), built with ``workers`` processes when it is
    not there yet. Returns it and a summary for the log (folder, built or reused, crops, frames, megabytes, seconds)."""
    import numpy as np

    _check_jitter()
    if len(records) != len(kinds):
        raise ValueError("one storage kind per record")
    root = Path(data_root).resolve()
    full_frames = {path for path in {record["image_path"] for record in records}
                   if full_picture(_safe_image_path(root, path)).is_file()}
    key = cache_key(records, kinds, full_frames, size=size, scheme=scheme, content=content)
    folder = root / CACHE_FOLDER / key[:32]
    started = time.perf_counter()
    if (folder / "meta.json").is_file():
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return CropCache(folder, size, scheme), {**meta, "folder": str(folder), "built": False, "seconds": 0.0}

    by_image: dict[str, list[tuple[int, int, list[float]]]] = {}
    for index, (record, kind) in enumerate(zip(records, kinds)):
        by_image.setdefault(record["image_path"], []).append((index, kind, [float(v) for v in record["bbox_xyxy"]]))
    tasks = [(str(root), path, size, scheme, items) for path, items in by_image.items()]
    staging = folder.with_name(f"{folder.name}.building-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    index = np.zeros((len(records), len(COLUMNS)), dtype=np.float64)
    offset = 0
    try:
        with open(staging / "pixels.bin", "wb") as out:
            def store(results):
                nonlocal offset
                for item, kind, data, height, width, region, frame_width, frame_height in results:
                    out.write(data)
                    index[item] = (offset, height, width, *region, frame_width, frame_height, kind)
                    offset += len(data)

            if workers > 0 and len(tasks) > 1:
                context = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                    for results in pool.map(_cut, tasks, chunksize=16):
                        store(results)
            else:
                for task in tasks:
                    store(_cut(task))
        np.save(staging / "index.npy", index)
        meta = {"format_version": FORMAT_VERSION, "key": key, "size": size, "scheme": scheme,
                "crops": len(records), "windows": int(sum(1 for kind in kinds if kind == WINDOW)),
                "frames": len(tasks), "full_frames": len(full_frames), "megabytes": round(offset / 1e6, 1), "content": content,
                "columns": list(COLUMNS)}
        (staging / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            staging.rename(folder)
        except OSError:
            # Another run built the same cache meanwhile: use that one.
            if not (folder / "meta.json").is_file():
                raise
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return CropCache(folder, size, scheme), {**meta, "folder": str(folder), "built": True,
                                             "seconds": round(time.perf_counter() - started, 1)}
