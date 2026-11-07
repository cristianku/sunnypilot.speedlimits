"""From Speed review to the brain's training: what every frame of a drive teaches, and what it may look at.

What it is for. A label in annotations/speed_review.sqlite changes nothing by itself. It reaches the network only as an
answer on a frame: the training compares the answer with what the network said, and that error moves the weights
(brain_net.brain_loss). This module makes those answers. It reads the labels and writes nothing: no training, no GPU.

A drive here is its frames in drive order, each on the comma's clock by the encode index (drive_logs.frame_time): labels
and frames meet on logMonoTime, never on a count of seconds. The list of frames needs no road types: the road type is only
data now, and no field of the map comes in here. Two rates (dataset's rate):

- 1 Hz, the brains before 7 Nov 2026: the frames of derived/frames, the road-type network's pictures (PIL);
- RATE_HZ (4 a second): every reading of speedvisiond, as the brain steps on the car. The frames of a second are its 1 Hz
  frame and the three readings after it (VIDEO_STEP decoded frames apart), each with its partition and area, so that the
  frozen split stays the one it was frozen on. Their pictures are vision.BrainInput's of the same NV12 frame as the car's,
  made while the signs are read (road_type_signs, pictures=True): a frame without its picture has no answer.

The speed (speed_answers), class kph / 5 - 1 and a weight of 1 or 0. Out of the loss, each with its reason:

- not_examined: nobody looked at it, a proposal left alone never counts as an answer;
- undetermined: a person looked and could not say;
- not_known_yet: between start_ns and known_ns, where the past is not enough yet: it would teach to guess the future;
- boundary: EDGE_SECONDS around a change of the speed, since a tap is only so exact (two stretches with the same speed
  are no change);
- no_frame: no picture, or a frame the encode index lacks.

The roadworks (works_answers), a target of 1 or 0 and a weight of 1 or 0, apart from the speed:

- inside a mark with an end: 1;
- from the start of the drive to speed_works_checked, outside the marks: 0, a person went through it;
- anywhere else: out of the loss. A stretch with no mark is not verified, it is not "no works";
- from the start of a mark with no end, and EDGE_SECONDS around every tap: out of the loss.

The two answers never touch: the end of works says nothing of the speed, and no code turns one into the other.

The limit ahead (ahead_answers), what the brain says beside the speed so that sunnypilot can slow down for it as for the
map's: the next limit (a class of the speed, or NONE_AHEAD) and the metres to it. A change is where his labels go from one
speed to another with no gap between the two; the metres are the car's own, vEgo integrated from the frame to it. Within
AHEAD_METRES of a change: its speed and its metres. Farther than AHEAD_MARGIN_METRES from any end of the speed in force:
none, and no metres. Between the two, and before an end that is no change (a gap, an undetermined stretch, the end of the
labels): out of the loss. Only on frames with an answer of the speed: boundary and not_known_yet are out here too.

The split (frame_partitions) is the frozen one, data/splits/speed-brain-split.json, never computed again: a drive of the
freeze joins it frame by frame through speed-brain-frames.json and fails loudly when its frames changed; a later drive
goes by the area of each frame, and an area the freeze never saw is train, or buffer next to a held-out one or without any
position. It is a mask on the loss: a drive stays whole, so that its memory runs through it as on the car.

Works sites stay whole (keep_sites_whole): a site with a frame in test or validation, SITE_MARGIN_S around it, has no
frame in train, and neither has any other site on those areas, on another drive too. So the measure is on sites the
network never learned, approach and exit included.

The inputs (drive_inputs), one group each so that a comparison can leave one out, all made only from what comes at or
before the frame:

- signs, plate, plate_hours, plate_wet, hour, lead, country, pedals, speed, engaged, sign_distance: what the car has,
  described in sunnypilot/speed_limits/brain_inputs.py. A drive not read yet has its signs unknown throughout. The hour and the country come from the last fix at or before the frame
  (merged['gps']), the country through the borders (gps.py): a frame never takes the country of a later fix. The lead comes
  from the series of the logs, which leave out a change under 2 m or 2 km/h (drive_extract): a little coarser than the car's.
  The pedals are in the last second, from the same carState (_pressed, as brain_inputs.PedalInputs does on the car).
  engaged goes through the car's own brain_inputs.EngagedInputs with the changes of sd_enabled, gas and brake in their
  order (engaged_inputs). sign_distance is the car's vision.SignDistance given every reading with its boxes and vEgo on a
  grid of SPEED_GRID_SECONDS, asked at each frame as speedvisiond asks it at each step (sign_distance_inputs); unknown on a
  drive read before the boxes were kept.

They are made by the car's own file, sunnypilot/speed_limits/brain_inputs.py, loaded here by its path (BRAIN_INPUTS): what
the network learns on is what it gets on the car, and the checkpoint keeps the file's hash.

pedals and speed are the driver, and they are apart so that a comparison can leave them out (engaged is made but never
learned, LEFT_OUT): on the drives the gas is often
pressed just where the label is above the limit the car held (he overrode it), and the car's speed is what the network itself
decides on the car. The measure says it: the frames with no gas, and the ones with the car's speed far from the label.

The sequences (drive_chunks, learning_pieces): the whole drive in order, state carried from piece to piece, so that
approach, crossing and exit are one sequence and the network sees only the past, as on the car. The pieces only bound the
gradient, and they never mix train with the rest: nnslr_tools.brain_train. The memory starts again after MEMORY_GAP_SECONDS
without a frame (memory_cuts), as on the car.

The hour in training (epoch_drives). Every labelled drive is by day, so the real hour says which drive a frame is from and
nothing of the speed, unless a plate gives it a meaning. So every epoch a drive takes a clock moved by a random number of
hours, one for all its frames, since on the car the clock runs on. A drive with a plate of hours on frames of train keeps its
real clock: there the hour is part of the answer. Validation and test always have the real one.

The night copies (night_copy). A plate of hours read outside them, with his label another speed than the number, says only
that the number does not hold now. Nothing shows when it does, and drives after 22:00 are few. So such a drive goes through
the epoch once more with its clock moved into the plate's hours, and on those frames the answer is the number: the same
pictures, made only here, nothing on the car, and no answer on roadworks. The plate's hours are the ones the reader read
(plate_hours), the night (NIGHT_FROM to NIGHT_TO) when it read none. Only frames of train with his speed above the number, a
plate of hours, a number that is a class of the speed and was not ended, and a known hour outside those hours. A label under
the number has another reason, roadworks for one, and a copy would raise it. A label equal to the number, on one such frame
of the passage (the frames in a row with that number and plate), says it holds then: such a plate has other hours, and the
passage has no copy. The limits: a plate of other hours with none read, driven outside them (7-17 at 19:00), gets a night
answer that is wrong, a number stays until the next sign also where the road was left, and the picture is of the day. The
plate comes from the class the reader gives (vision.class_plate and class_hours: maximum_speed_hours:<number>[:<from>-<to>], a
sign with its plate in one box), so with a classifier without that class there is no copy at all: plate is unknown on every
frame, and the export says so (plates_in_train, hours_in_train).

A plate for wet roads has no copy. Whether it holds the network sees in the picture, and the picture of a copy is of a dry
road: the number as its answer would teach it to slow down on a dry one. So it learns only from the plate passed on a wet road
and on a dry one, with his speed of each. The plate comes from the class the reader gives (vision.class_plate:
maximum_speed_wet:<number>, a sign with its plate in one box), so with a classifier without that class it is unknown on every
frame, and the export says so (wet_in_train).

The numbers swapped (swap_passages, swapped). On a few drives the picture tells the road and a road seen in training tells its
speed, so the network has no reason to read the sign: the second run said 50 past a 60, and took no notice of the signs. So
every epoch half of the passages (the frames in a row with one number) whose answers all say the number go with that number
and those answers changed to the other of its family (SWAPS): the same place with two limits, and only the sign tells which.
A family is numbers the same kind of road can have, so that no scene turns absurd. Only answers of train count. A passage with
an answer off the number stays as it is: the sign was wrong there, and that too is for the brain to learn. A frame with no
number in force keeps its answer, since the road alone says it there, and an end read keeps the answer after it. Besides the
number, the single readings equal to it change, on the frames of the passage. Nothing of this reaches a measure, but
brain_train's test_swapped and validation_swapped, which ask exactly this of their frames.

What the training refuses (check_drives), so that nothing goes wrong without a word: a drive not read or read by another
reader than the car's today, since an unread drive looks like a drive with no sign and then "no reading" is learned as a
speed; a drive with no frame; at RATE_HZ, a drive without the brain's pictures.
"""
from __future__ import annotations

import bisect
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from nnslr_tools import drive_logs
from nnslr_tools.gps import frame_country, load_positions
from nnslr_tools.media import FRAME_SIZE
from nnslr_tools.road_type_data import FRAMES_RELATIVE, frame_cells, frame_path
from nnslr_tools.road_type_signs import (SEGMENT_SECONDS, STEPS_PER_SECOND, VIDEO_FPS, VIDEO_STEP, fingerprint, load_signs,
                                         load_vision, package_path, picture_path, pictures_ready)
from nnslr_tools.speed_review import GRID_KPH, MAX_KPH, NS, db_path

SPLIT_RELATIVE = Path('splits') / 'speed-brain-split.json'
SPLIT_FRAMES_RELATIVE = Path('splits') / 'speed-brain-frames.json'
SPEED_CLASSES = MAX_KPH // GRID_KPH
EDGE_SECONDS = 2
SITE_MARGIN_S = 60
READ_COMMAND = 'nnslr road-type-signs --labelled'
NOT_EXAMINED, UNDETERMINED, NOT_KNOWN_YET, BOUNDARY, NO_FRAME = range(1, 6)
OUT_REASONS = {NOT_EXAMINED: 'not_examined', UNDETERMINED: 'undetermined', NOT_KNOWN_YET: 'not_known_yet', BOUNDARY: 'boundary',
               NO_FRAME: 'no_frame'}
HELD_OUT = ('test', 'validation')
NIGHT_FROM, NIGHT_TO = 22, 6
SWAPS = {30: 40, 40: 30, 50: 60, 60: 50, 70: 80, 80: 70, 100: 120, 120: 100}
RATE_HZ = float(STEPS_PER_SECOND)  # the brain's steps a second since 7 Nov 2026: every reading of speedvisiond
AHEAD_METRES = 100.0
AHEAD_MARGIN_METRES = 130.0
METRES_SCALE = 100.0  # the metres output is the metres over this (contract output.metres_scale)
NONE_AHEAD = SPEED_CLASSES  # the last class of next: no limit ahead
SPEED_GRID_SECONDS = 0.05  # vEgo to vision.SignDistance on this grid; the car gives it every carState, 100 a second


def _load_brain_inputs(path: Path):
    """The car's brain_inputs.py (numpy only) as a module of its own."""
    spec = importlib.util.spec_from_file_location('comma_brain_inputs', path)
    if spec is None or spec.loader is None or not path.is_file():
        raise ValueError(f"no {path}: the car's file of the brain's inputs, in the clone of the branch speed-limits")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BRAIN_INPUTS_PATH = package_path() / 'brain_inputs.py'
BRAIN_INPUTS = _load_brain_inputs(BRAIN_INPUTS_PATH)
VISION = load_vision(package_path())
COUNTRIES = BRAIN_INPUTS.COUNTRIES
INPUT_SIZES = BRAIN_INPUTS.INPUT_SIZES
DRIVER_INPUTS = ('pedals', 'speed')
# Groups the car still makes that no brain learns from: engaged (openpilot on or off, 7 Nov 2026) only confused the
# network, his call the same night. A brain built before keeps it: its groups are in its checkpoint and contract.
LEFT_OUT = ('engaged',)
pack_inputs = BRAIN_INPUTS.pack_inputs


def speed_class(kph: int) -> int:
    return kph // GRID_KPH - 1


def class_kph(index):
    """The km/h of a class, or of an array of them."""
    return (index + 1) * GRID_KPH


def read_labels(root: Path) -> dict[str, dict]:
    """{route: {'stretches', 'works', 'checked_ns'}} of every drive with a label or a mark. Read only: this database holds
    what only a person can write."""
    path = db_path(root)
    if not path.is_file():
        return {}
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        labels: dict[str, dict] = {}

        def drive(route: str) -> dict:
            return labels.setdefault(route, {'stretches': [], 'works': [], 'checked_ns': None})

        for row in connection.execute('SELECT * FROM speed_stretches ORDER BY route, start_ns'):
            drive(row['route'])['stretches'].append(dict(row))
        for row in connection.execute('SELECT * FROM speed_works ORDER BY route, start_ns'):
            drive(row['route'])['works'].append(dict(row))
        for row in connection.execute('SELECT * FROM speed_works_checked'):
            drive(row['route'])['checked_ns'] = row['end_ns']
        return labels
    finally:
        connection.close()


def speed_answers(ns: np.ndarray, stretches: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(class per frame, -1 without an answer; why a frame is out of the loss, 0 when it is in) at the times ns."""
    classes = np.full(len(ns), -1, dtype=np.int64)
    out = np.full(len(ns), NOT_EXAMINED, dtype=np.int64)
    ordered = sorted(stretches, key=lambda row: row['start_ns'])
    for row in ordered:
        inside = (ns >= row['start_ns']) & (ns < row['end_ns'])
        if row['state'] == 'undetermined' or row['kph'] is None:
            out[inside] = UNDETERMINED
            continue
        classes[inside] = speed_class(row['kph'])
        out[inside] = 0
        if row['known_ns'] is not None:
            out[inside & (ns < row['known_ns'])] = NOT_KNOWN_YET
    for before, after in zip([None, *ordered], [*ordered, None]):
        same = before and after and before['end_ns'] == after['start_ns'] and before['kph'] == after['kph']
        for edge in () if same else {row[key] for row, key in ((before, 'end_ns'), (after, 'start_ns')) if row}:
            near = (np.abs(ns - edge) < EDGE_SECONDS * NS) & (out == 0)
            out[near] = BOUNDARY
    classes[out != 0] = -1
    return classes, out


def works_answers(ns: np.ndarray, works: list[dict], checked_ns: int | None) -> tuple[np.ndarray, np.ndarray]:
    """(target, weight) float32 per frame at the times ns: see the module's text."""
    target = np.zeros(len(ns), dtype=np.float32)
    weight = np.zeros(len(ns), dtype=np.float32)
    if checked_ns is not None:
        weight[ns < checked_ns] = 1
    edges = []
    for mark in works:
        edges.append(mark['start_ns'])
        if mark['end_ns'] is None:
            weight[ns >= mark['start_ns']] = 0
            continue
        edges.append(mark['end_ns'])
        inside = (ns >= mark['start_ns']) & (ns < mark['end_ns'])
        target[inside] = 1
        weight[inside] = 1
    for edge in edges:
        weight[np.abs(ns - edge) < EDGE_SECONDS * NS] = 0
    return target, weight


def drive_frames(root: Path, route: str, merged: dict) -> list[dict]:
    """The 1 Hz frames of a drive in drive order: key, segment, index, seconds (from the start of the drive), decoded (the
    frame of the segment's video), ns (the exposure start; None when the encode index lacks it), latitude, longitude,
    image, present."""
    index = drive_logs.frame_index(merged)
    positions = load_positions(root, route)
    frames = []
    for path in sorted((Path(root) / FRAMES_RELATIVE / route).glob('*/frames.jsonl')):
        for text in path.read_text(encoding='utf-8').splitlines():
            if not text.strip():
                continue
            row = json.loads(text)
            segment, second = row['segment_index'], row['output_index']
            key = f'{segment}/{second}'
            image = frame_path(root, route, segment, second)
            gps = positions.get(key, {})
            frames.append({'route': route, 'key': key, 'segment': segment, 'index': second,
                           'seconds': float(segment * SEGMENT_SECONDS + second), 'decoded': row['decoded_frame_index'],
                           'ns': drive_logs.frame_time(index, segment, row['decoded_frame_index']),
                           'latitude': gps.get('latitude'), 'longitude': gps.get('longitude'), 'image': str(image),
                           'present': image.is_file()})
    frames.sort(key=lambda frame: (frame['segment'], frame['index']))
    return frames


def _exposures(merged: dict) -> dict[int, dict[int, int]]:
    """{segment: {decoded frame: exposure start}} of the encode index, as drive_logs.frame_time reads it."""
    found = {}
    for _first, segment, sof, ids in drive_logs.frame_index(merged):
        times = found.setdefault(segment, {})
        for decoded, t in zip(ids, sof):
            times.setdefault(decoded, t)
    return found


def brain_frames(root: Path, frames: list[dict], partitions: list[str], cells: list, merged: dict,
                 record: dict | None) -> tuple[list[dict], list[str], list]:
    """The frames at RATE_HZ from the 1 Hz ones: each 1 Hz frame and the STEPS_PER_SECOND - 1 readings after it, with its
    partition and area; step (0 for the 1 Hz frame), picture (the row of the segment's brain pictures) and present (that
    picture is there)."""
    exposures, counts = _exposures(merged), (record or {}).get('pictures') or {}
    files: dict[int, bool] = {}
    found, parts, places = [], [], []
    for frame, partition, cell in zip(frames, partitions, cells):
        segment = frame['segment']
        if frame['decoded'] % VIDEO_STEP:
            raise ValueError(f"{frame['route']} {frame['key']}: a 1 Hz frame off the readings ({frame['decoded']})")
        if segment not in files:
            files[segment] = picture_path(root, frame['route'], segment).is_file()
        for step in range(STEPS_PER_SECOND):
            decoded = frame['decoded'] + step * VIDEO_STEP
            row = decoded // VIDEO_STEP
            found.append({**frame, 'key': f'{segment}/{decoded / VIDEO_FPS:g}', 'decoded': decoded, 'step': step,
                          'seconds': segment * SEGMENT_SECONDS + decoded / VIDEO_FPS,
                          'ns': exposures.get(segment, {}).get(decoded), 'picture': row,
                          'present': files[segment] and row < int(counts.get(str(segment)) or 0)})
            parts.append(partition)
            places.append(cell)
    return found, parts, places


def read_split(root: Path) -> dict:
    """The frozen split with the frame keys of its drives."""
    split = json.loads((Path(root) / SPLIT_RELATIVE).read_text(encoding='utf-8'))
    frames = json.loads((Path(root) / SPLIT_FRAMES_RELATIVE).read_text(encoding='utf-8'))
    if frames['split_sha256'] != split['sha256']:
        raise ValueError('the frames of the split are not the ones it was frozen on')
    return {**split, 'frame_keys': frames['frames']}


def _cell_partition(cell: tuple[int, int] | None, cells: dict[str, str]) -> str:
    """The partition of an area the way a drive after the freeze takes it."""
    if cell is None:
        return 'buffer'
    found = cells.get(f'{cell[0]},{cell[1]}')
    if found is not None:
        return found
    around = (cells.get(f'{cell[0] + a},{cell[1] + b}') for a in (-1, 0, 1) for b in (-1, 0, 1))
    return 'buffer' if any(name in HELD_OUT for name in around) else 'train'


def frame_partitions(route: str, frames: list[dict], split: dict) -> list[str]:
    """The partition of each frame: see the module's text."""
    frozen = split['partitions'].get(route)
    if frozen is None:
        return [_cell_partition(cell, split['cells']) for cell in frame_cells(frames)]
    by_key = dict(zip(split['frame_keys'][route], frozen))
    missing = [frame['key'] for frame in frames if frame['key'] not in by_key]
    if missing or len(by_key) != len(frames):
        raise ValueError(f'{route}: the frames changed since the split was frozen ({len(frames)} now, {len(by_key)} then)')
    return [by_key[frame['key']] for frame in frames]


def _last_at(points: list, ns: np.ndarray) -> list:
    """The value of a step series at each time: its last point at or before it, None before the first."""
    times = [point[0] for point in points]
    found = []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        found.append(points[at][1] if at >= 0 else None)
    return found


def _pressed(points: list, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pressed at some moment of the last second, known) of a step series of booleans at each time."""
    times = [point[0] for point in points]
    pressed = np.zeros(len(ns), dtype=np.float32)
    known = np.zeros(len(ns), dtype=np.float32)
    for row, t in enumerate(ns):
        last = bisect.bisect_right(times, int(t)) - 1
        first = max(bisect.bisect_right(times, int(t) - NS) - 1, 0)
        values = [points[at][1] for at in range(first, last + 1)]
        known[row] = float(any(value is not None for value in values))
        pressed[row] = float(any(value is True for value in values))
    return pressed, known


def engaged_inputs(series: dict, ns: np.ndarray) -> np.ndarray:
    """The engaged group (frames, size): the car's EngagedInputs given the changes of gas, brake and sd_enabled in their
    order, the pedals before openpilot at the same moment as on the car, and asked at each frame. Unknown at a frame with no
    time."""
    events = sorted([(t, 0, name, value) for name in ('gas', 'brake') for t, value in series.get(name) or []] +
                    [(t, 1, 'enabled', value) for t, value in series.get('sd_enabled') or []], key=lambda event: event[:2])
    engaged, pedals = BRAIN_INPUTS.EngagedInputs(), {'gas': False, 'brake': False}
    rows = np.zeros((len(ns), INPUT_SIZES['engaged']), dtype=np.float32)
    at = 0
    for row in sorted(np.flatnonzero(ns >= 0), key=lambda row: ns[row]):
        while at < len(events) and events[at][0] <= ns[row]:
            t, _, name, value = events[at]
            if name == 'enabled':
                engaged.add_enabled(t / NS, value is True)
            else:
                pedals[name] = value is True
                engaged.add_pedals(t / NS, pedals['gas'], pedals['brake'])
            at += 1
        rows[row] = engaged.row(ns[row] / NS)
    return rows


class _SpeedFeed:
    """vEgo (a step series [t_ns, km/h]) to a vision.SignDistance on a grid of SPEED_GRID_SECONDS from its first point."""

    def __init__(self, points: list):
        self.points, self.at = points, 0
        self.t = int(points[0][0]) if points else None

    def to(self, distance, until_ns: int) -> None:
        step = int(SPEED_GRID_SECONDS * NS)
        while self.t is not None and self.t <= until_ns:
            while self.at + 1 < len(self.points) and self.points[self.at + 1][0] <= self.t:
                self.at += 1
            kph = self.points[self.at][1]
            if kph is not None:
                distance.add_speed(self.t / NS, kph / 3.6)
            self.t += step


def sign_distance_inputs(ns: np.ndarray, merged: dict, readings: list | None, boxes: list | None) -> np.ndarray:
    """The sign_distance group (frames, size): see the module's text. Readings [[seconds, [[class, confidence], ...]], ...]
    and their boxes in the same order; at a frame with no time, or with no boxes, unknown."""
    rows = np.zeros((len(ns), INPUT_SIZES['sign_distance']), dtype=np.float32)
    if not readings or boxes is None or len(boxes) != len(readings):
        return rows
    exposures, seen = _exposures(merged), {}
    for (seconds, read), places in zip(readings, boxes):
        segment = int(seconds // SEGMENT_SECONDS)
        at = exposures.get(segment, {}).get(int(round((seconds - segment * SEGMENT_SECONDS) * VIDEO_FPS)))
        if at is not None:
            seen[int(at)] = [{'class': name, 'confidence': confidence, 'box': tuple(box)}
                             for (name, confidence), box in zip(read, places)]
    asked: dict[int, list[int]] = {}
    for row in np.flatnonzero(ns >= 0):
        asked.setdefault(int(ns[row]), []).append(int(row))
    distance = VISION.SignDistance(VISION.FOCAL_LENGTHS.get(FRAME_SIZE, 0.0))
    speeds = _SpeedFeed(sorted((merged.get('series') or {}).get('v_ego') or [], key=lambda point: point[0]))
    for t in sorted(set(seen) | set(asked)):
        speeds.to(distance, t)
        metres = distance.update(t / NS, seen.get(t, []), None)
        for row in asked.get(t, ()):
            rows[row] = BRAIN_INPUTS.sign_distance_row(metres)
    return rows


def metres_driven(points: list):
    """ns -> metres driven since the first point of vEgo (a step series [t_ns, km/h]), the speed held between points."""
    points = [point for point in sorted(points, key=lambda point: point[0]) if point[1] is not None]
    if not points:
        return lambda ns: np.zeros(np.shape(ns))
    times = np.array([point[0] for point in points], dtype=np.int64)
    speed = np.array([max(point[1], 0.0) / 3.6 for point in points], dtype=np.float64)
    before = np.concatenate([[0.0], np.cumsum(speed[:-1] * np.diff(times) / NS)])

    def at(ns):
        ns = np.asarray(ns, dtype=np.int64)
        last = np.maximum(np.searchsorted(times, ns, side='right') - 1, 0)
        return before[last] + speed[last] * np.maximum(ns - times[last], 0) / NS

    return at


def ahead_answers(ns: np.ndarray, stretches: list[dict], metres_at, asked: np.ndarray) -> tuple[np.ndarray, ...]:
    """(next class per frame, its weight, the metres to it over METRES_SCALE, their weight) at the times ns, on the frames
    asked (those with an answer of the speed): see the module's text."""
    following = np.full(len(ns), NONE_AHEAD, dtype=np.int64)
    next_weight, metres, metres_weight = (np.zeros(len(ns), dtype=np.float32) for _ in range(3))
    runs: list[list] = []  # [start_ns, end_ns, kph or None]: the same speed in stretches that touch, as one
    for row in sorted(stretches, key=lambda row: row['start_ns']):
        kph = None if row['state'] == 'undetermined' else row['kph']
        if runs and kph is not None and runs[-1][2] == kph and runs[-1][1] == row['start_ns']:
            runs[-1][1] = row['end_ns']
            continue
        runs.append([row['start_ns'], row['end_ns'], kph])
    for this, after in zip(runs, [*runs[1:], None]):
        rows = np.flatnonzero((ns >= this[0]) & (ns < this[1]) & asked) if this[2] is not None else []
        if not len(rows):
            continue
        left = metres_at(this[1]) - metres_at(ns[rows])
        if after is not None and after[2] is not None and after[0] == this[1]:
            near = rows[left <= AHEAD_METRES]
            following[near], next_weight[near] = speed_class(after[2]), 1
            metres[near], metres_weight[near] = left[left <= AHEAD_METRES] / METRES_SCALE, 1
        next_weight[rows[left > AHEAD_MARGIN_METRES]] = 1
    return following, next_weight, metres, metres_weight


def sign_inputs(seconds: np.ndarray, readings: list | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The signs, plate, plate_hours and plate_wet groups (frames, size) from the sure readings [[seconds, [[class, confidence], ...]], ...] of a
    drive, by the car's own file. Only readings at or before the frame count; None: the drive was not read, all unknown."""
    signs = np.zeros((len(seconds), INPUT_SIZES['signs']), dtype=np.float32)
    plates = np.zeros((len(seconds), INPUT_SIZES['plate']), dtype=np.float32)
    hours = np.zeros((len(seconds), INPUT_SIZES['plate_hours']), dtype=np.float32)
    wet = np.zeros((len(seconds), INPUT_SIZES['plate_wet']), dtype=np.float32)
    ordered = sorted(readings or [], key=lambda item: item[0])
    memory, pointer = BRAIN_INPUTS.SignInputs(VISION), 0
    for row, second in enumerate(seconds):
        while pointer < len(ordered) and ordered[pointer][0] <= second:
            now, read = ordered[pointer]
            memory.add(now, [(name, confidence, VISION.class_plate(name), VISION.class_hours(name))
                             for name, confidence in read])
            pointer += 1
        signs[row], plates[row], hours[row] = memory.features(second), memory.plate_row(), memory.plate_hours_row()
        wet[row] = memory.plate_wet_row()
    return signs, plates, hours, wet


def countries_so_far(gps: list, ns: np.ndarray, root: Path | None) -> list[str | None]:
    """The country at each time as the car has it: that of the last fix at or before it [t_ns, latitude, longitude,
    unix_ms], none before the first."""
    times, of_fix, found = [row[0] for row in gps], {}, []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        if at >= 0 and at not in of_fix:
            of_fix[at] = frame_country(gps[at][1], gps[at][2], root)
        found.append(of_fix.get(at))
    return found


def unix_at(gps: list, ns: np.ndarray) -> list[float | None]:
    """The GPS clock at each time, seconds: its last fix at or before it [t_ns, latitude, longitude, unix_ms] and the time
    since. None before the first."""
    fixes = [row for row in gps if row[3]]
    times = [row[0] for row in fixes]
    found = []
    for t in ns:
        at = bisect.bisect_right(times, int(t)) - 1
        found.append(None if at < 0 or t < 0 else fixes[at][3] / 1000 + (int(t) - times[at]) / NS)
    return found


def drive_inputs(frames: list[dict], ns: np.ndarray, merged: dict, readings: list | None,
                 root: Path | None, boxes: list | None = None) -> dict[str, np.ndarray]:
    """{group: (frames, size) float32}: see the module's text. Nothing after a frame reaches it."""
    series, gps = merged.get('series') or {}, merged.get('gps') or []
    seconds = np.array([frame.get('seconds', frame['segment'] * SEGMENT_SECONDS + frame['index']) for frame in frames],
                       dtype=np.float64)
    distance, lead_speed = _last_at(series.get('lead_d') or [], ns), _last_at(series.get('lead_v') or [], ns)
    lead = np.array([BRAIN_INPUTS.lead_row(metres, kph) for metres, kph in zip(distance, lead_speed)], dtype=np.float32)
    gas, gas_known = _pressed(series.get('gas') or [], ns)
    brake, brake_known = _pressed(series.get('brake') or [], ns)
    speed = [BRAIN_INPUTS.speed_row(None if kph is None else kph / 3.6) for kph in _last_at(series.get('v_ego') or [], ns)]
    countries = countries_so_far(gps, ns, root)
    country = np.array([BRAIN_INPUTS.country_row(name) for name in countries], dtype=np.float32)
    hour = np.array([BRAIN_INPUTS.hour_row(BRAIN_INPUTS.local_hour(unix, name))
                     for unix, name in zip(unix_at(gps, ns), countries)], dtype=np.float32)
    signs, plate, plate_hours, plate_wet = sign_inputs(seconds, readings)
    return {'signs': signs, 'plate': plate, 'plate_hours': plate_hours, 'plate_wet': plate_wet,
            'hour': hour.reshape(len(ns), INPUT_SIZES['hour']),
            'lead': lead.reshape(len(ns), INPUT_SIZES['lead']), 'country': country.reshape(len(ns), INPUT_SIZES['country']),
            'pedals': np.stack([gas, brake, gas_known * brake_known], axis=1),
            'speed': np.array(speed, dtype=np.float32).reshape(len(ns), INPUT_SIZES['speed']),
            'engaged': engaged_inputs(series, ns), 'sign_distance': sign_distance_inputs(ns, merged, readings, boxes)}


def inputs_sha256() -> str:
    """The hash of the car's file of the inputs, for the checkpoint: the car refuses a network made with another."""
    return hashlib.sha256(BRAIN_INPUTS_PATH.read_bytes()).hexdigest()


def memory_cuts(ns: np.ndarray) -> set[int]:
    """The frames the memory starts again at: after MEMORY_GAP_SECONDS without a frame."""
    known = np.flatnonzero(ns >= 0)
    return {int(at) for at in known[1:][np.diff(ns[known]) > BRAIN_INPUTS.MEMORY_GAP_SECONDS * NS]}


def drive_targets(root: Path, route: str, labels: dict, merged: dict, split: dict, *, rate: float = 1.0,
                  need_pictures: bool = True) -> dict:
    """A drive for the training at rate frames a second (1 or RATE_HZ): frames, ns, partitions, cells, cuts (memory_cuts),
    speed (class per frame), speed_out (why a frame is out of the loss, 0 when in), speed_weight, works, works_weight, next,
    next_weight, metres, metres_weight (ahead_answers), measured (the frames a measure counts: one a second, so that it
    reads the same at both rates), inputs, own_kph (the car's speed, NaN unknown: for the measure, no input), signs_read,
    reader (what read them), pictures (at RATE_HZ, the brain's pictures are there), no_picture_s (labelled seconds without a
    frame at all: prepare keeps only the slow ones at the start and the end of a drive). need_pictures=False (a dry run): a
    frame without its picture keeps its answers."""
    frames = drive_frames(root, route, merged)
    partitions = frame_partitions(route, frames, split)
    cells = frame_cells(frames)
    record = load_signs(root, route)
    if rate == RATE_HZ:
        frames, partitions, cells = brain_frames(root, frames, partitions, cells, merged, record)
    elif rate != 1.0:
        raise ValueError(f'the brain runs at 1 or {RATE_HZ:g} frames a second, not {rate}')
    ns = np.array([-1 if frame['ns'] is None else frame['ns'] for frame in frames], dtype=np.int64)
    usable = np.array([frame['ns'] is not None and (frame['present'] or not need_pictures) for frame in frames], dtype=bool)
    speed, speed_out = speed_answers(ns, labels['stretches'])
    speed_out[~usable] = NO_FRAME
    speed[~usable] = -1
    works, works_weight = works_answers(ns, labels['works'], labels['checked_ns'])
    works_weight[~usable] = 0
    readings = record['readings'] if record else None
    v_ego = (merged.get('series') or {}).get('v_ego') or []
    own = _last_at(v_ego, ns)
    following, next_weight, metres, metres_weight = ahead_answers(ns, labels['stretches'], metres_driven(v_ego),
                                                                  speed_out == 0)
    labelled = sum(((ns >= row['start_ns']) & (ns < row['end_ns'])).sum() for row in labels['stretches'])
    return {'route': route, 'rate': rate, 'frames': frames, 'ns': ns, 'partitions': partitions,
            'own_kph': np.array([np.nan if kph is None else kph for kph in own], dtype=np.float32),
            'cells': cells, 'cuts': memory_cuts(ns), 'speed': speed, 'speed_out': speed_out,
            'speed_weight': (speed_out == 0).astype(np.float32), 'works': works, 'works_weight': works_weight,
            'next': following, 'next_weight': next_weight, 'metres': metres, 'metres_weight': metres_weight,
            'measured': np.array([not frame.get('step') for frame in frames], dtype=bool),
            'marks': [mark for mark in labels['works'] if mark['end_ns'] is not None],
            'inputs': drive_inputs(frames, ns, merged, readings, root, record.get('boxes') if record else None),
            'signs_read': readings is not None,
            'reader': record.get('fingerprint') if record else None,
            'pictures': rate == 1.0 or pictures_ready(root, record),
            'no_picture_s': max(round(sum(row['end_ns'] - row['start_ns'] for row in labels['stretches']) / NS -
                                      labelled / rate), 0)}


def keep_sites_whole(drives: list[dict]) -> None:
    """Train frames of a held-out works site, and of any site on its areas, become buffer: see the module's text."""
    sites = []
    for drive in drives:
        for mark in drive['marks']:
            near = np.flatnonzero((drive['ns'] >= mark['start_ns'] - SITE_MARGIN_S * NS) &
                                  (drive['ns'] < mark['end_ns'] + SITE_MARGIN_S * NS))
            sites.append((drive, near, {drive['cells'][i] for i in near} - {None}))
    held = set()
    for drive, near, cells in sites:
        if any(drive['partitions'][i] in HELD_OUT for i in near):
            held |= cells
    for drive, near, cells in sites:
        if cells & held:
            for i in near:
                if drive['partitions'][i] == 'train':
                    drive['partitions'][i] = 'buffer'


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def dataset(root: Path, merged_of, routes: list[str] | None = None, log=_say, *, rate: float = 1.0,
            need_pictures: bool = True) -> list[dict]:
    """Every drive with a label or a mark (or only routes), as drive_targets at rate, each with reader_current: read by the
    car's reader of today. merged_of(route): the drive's joined logs, None before a scan. Sites stay whole only across the
    drives asked for."""
    split = read_split(root)
    reader = fingerprint(package_path())
    drives = []
    for route, labels in sorted(read_labels(root).items()):
        if routes is not None and route not in routes:
            continue
        merged = merged_of(route)
        if merged is None:
            log(f'{route}: labels and no scan, left out')
            continue
        drive = drive_targets(root, route, labels, merged, split, rate=rate, need_pictures=need_pictures)
        drives.append({**drive, 'reader_current': drive['reader'] == reader})
    keep_sites_whole(drives)
    return drives


def check_drives(drives: list[dict]) -> None:
    """Raises on what would train wrong without a word: see the module's text."""
    empty = [drive['route'] for drive in drives if not len(drive['frames'])]
    if empty:
        raise ValueError(f'drives with labels and no frame: {", ".join(empty)}')
    unread = [drive['route'] for drive in drives if not drive['signs_read']]
    if unread:
        raise ValueError(f'drives with labels and no reading of the signs: {", ".join(unread)}. Run {READ_COMMAND}')
    other = [drive['route'] for drive in drives if not drive['reader_current']]
    if other:
        raise ValueError(f"drives read by another reader than the car's of today: {', '.join(other)}. Run {READ_COMMAND}")
    bare = [drive['route'] for drive in drives if not drive.get('pictures', True)]
    if bare:
        raise ValueError(f"drives without the brain's pictures: {', '.join(bare)}. Run {READ_COMMAND}")


def loss_weights(drive: dict, partition: str) -> tuple[np.ndarray, np.ndarray]:
    """(speed weight, works weight) of a drive in one partition: the rest of the drive still runs through the memory."""
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['speed_weight'] * inside, drive['works_weight'] * inside


def ahead_weights(drive: dict, partition: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(next weight, metres weight) of a drive in one partition; None for a drive without them."""
    if 'next_weight' not in drive:
        return None
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['next_weight'] * inside, drive['metres_weight'] * inside


def other_hour(hour: np.ndarray, hours: float) -> np.ndarray:
    """The hour group with the clock moved by hours on every frame. An unknown hour stays unknown."""
    sin, cos, angle = hour[:, 0], hour[:, 1], 2 * np.pi * hours / 24
    return np.stack([sin * np.cos(angle) + cos * np.sin(angle), cos * np.cos(angle) - sin * np.sin(angle), hour[:, 2]],
                    axis=1).astype(np.float32)


def _clock(hour: np.ndarray) -> np.ndarray:
    return np.arctan2(hour[:, 0], hour[:, 1]) / (2 * np.pi) * 24 % 24


def _plate_window(drive: dict) -> tuple[np.ndarray, np.ndarray]:
    """(from, how many hours) of the plate on each frame: the hours read, the night when none were."""
    hours = drive['inputs']['plate_hours']
    start = np.where(hours[:, 4] > 0, np.rint(_clock(hours[:, :2])), NIGHT_FROM)
    end = np.where(hours[:, 4] > 0, np.rint(_clock(hours[:, 2:4])), NIGHT_TO)
    return start % 24, (end - start) % 24


def _inside(clock: np.ndarray, window: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (clock - window[0]) % 24 < window[1]


def _hours_plate(drive: dict) -> np.ndarray:
    return drive['inputs']['plate'][:, BRAIN_INPUTS.PLATES.index('time')] > 0


def _wet_plate(drive: dict) -> np.ndarray:
    return drive['inputs']['plate_wet'][:, 0] > 0


def plates_in_train(drives: list[dict]) -> bool:
    """Whether a brain trained on these drives learnt the plate: a plate of hours on a frame of train with a speed to answer.
    The car gives a brain its plates only then (contract step.plates), since weights never trained are no reading."""
    return any(bool((_hours_plate(drive) & (loss_weights(drive, 'train')[0] > 0)).any()) for drive in drives)


def hours_in_train(drives: list[dict]) -> bool:
    """Whether a brain trained on these drives learnt the hours of a plate: as plates_in_train, with hours read."""
    return any(bool(((drive['inputs']['plate_hours'][:, 4] > 0) & (loss_weights(drive, 'train')[0] > 0)).any())
               for drive in drives)


def wet_in_train(drives: list[dict]) -> bool:
    """Whether a brain trained on these drives learnt the plate for wet roads: as plates_in_train."""
    return any(bool((_wet_plate(drive) & (loss_weights(drive, 'train')[0] > 0)).any()) for drive in drives)


def _with_hour(drive: dict, groups: tuple[str, ...], hours: float) -> dict:
    inputs = {**drive['inputs'], 'hour': other_hour(drive['inputs']['hour'], hours)}
    return {**drive, 'inputs': inputs, 'extra': pack_inputs(inputs, groups)}


def sign_kph(signs: np.ndarray) -> np.ndarray:
    """The km/h of the number confirmed in the rows of the signs input."""
    return np.rint(signs[:, 0] * BRAIN_INPUTS.SIGN_KPH_SCALE).astype(np.int64)


def night_frames(drive: dict) -> tuple[np.ndarray, np.ndarray]:
    """(the frames a night copy can answer, the class of the number on each): see the module's text."""
    signs, hour = drive['inputs']['signs'], drive['inputs']['hour']
    kph = sign_kph(signs)
    a_class = (kph >= GRID_KPH) & (kph <= MAX_KPH) & (kph % GRID_KPH == 0) & (signs[:, 6] == 0)
    plated, label = _hours_plate(drive) & a_class, class_kph(drive['speed'])
    answers = (hour[:, 2] > 0) & ~_inside(_clock(hour), _plate_window(drive)) & (loss_weights(drive, 'train')[0] > 0)
    frames = plated & answers & (label > kph)
    for passage in np.split(np.arange(len(kph)), np.flatnonzero((np.diff(kph) != 0) | (np.diff(plated) != 0)) + 1):
        if (answers[passage] & (label[passage] == kph[passage])).any():
            frames[passage] = False
    return frames, speed_class(kph)


def night_copy(drive: dict, groups: tuple[str, ...], rng) -> dict | None:
    """The drive with its first frame to answer at a random hour of its plate's, the number the answer of the frames that
    fall in their plate's hours: see the module's text. None with nothing to answer."""
    frames, number = night_frames(drive)
    if not frames.any():
        return None
    clock = _clock(drive['inputs']['hour'])
    window, first = _plate_window(drive), np.flatnonzero(frames)[0]
    hours = window[0][first] + rng.uniform(0, window[1][first]) - clock[first]
    answered = frames & _inside((clock + hours) % 24, window)
    copy = {**_with_hour(drive, groups, hours), 'speed': np.where(answered, number, drive['speed']),
            'speed_weight': answered.astype(np.float32), 'works_weight': np.zeros_like(drive['works_weight'])}
    if 'next_weight' in drive:
        copy['next_weight'], copy['metres_weight'] = np.zeros_like(drive['next_weight']), np.zeros_like(drive['metres_weight'])
    return copy


def swap_passages(drive: dict, partition: str = 'train') -> list[np.ndarray]:
    """The passages of a number of SWAPS with an answer of this partition, in force, and all such answers the number: see
    the module's text."""
    signs = drive['inputs']['signs']
    kph, label = sign_kph(signs), class_kph(drive['speed'])
    answers = (loss_weights(drive, partition)[0] > 0) & (signs[:, 6] == 0)
    found = []
    for passage in np.split(np.arange(len(kph)), np.flatnonzero(np.diff(kph) != 0) + 1):
        number, asked = int(kph[passage[0]]), passage[answers[passage]]
        said = asked[label[asked] == number]
        # Before the sign the speed may still be the one before it: the number is read some way ahead of it.
        if number in SWAPS and len(said) and (label[asked[asked >= said[0]]] == number).all():
            found.append(passage)
    return found


def swapped(drive: dict, passages: list[np.ndarray], groups: tuple[str, ...]) -> dict:
    """The drive with the number of these passages, and what says it, as the other of its family: see the module's text."""
    signs, speed, label = drive['inputs']['signs'].copy(), drive['speed'].copy(), class_kph(drive['speed'])
    following = drive['next'].copy() if 'next' in drive else None
    scale = BRAIN_INPUTS.SIGN_KPH_SCALE
    for passage in passages:
        number = int(sign_kph(signs[passage[:1]])[0])
        said = passage[(label[passage] == number) & (signs[passage, 6] == 0)]
        single = passage[np.rint(signs[passage, 8] * scale) == number]
        signs[passage, 0] = signs[single, 8] = SWAPS[number] / scale
        speed[said] = speed_class(SWAPS[number])
        if following is not None:
            ahead = passage[following[passage] == speed_class(number)]
            following[ahead] = speed_class(SWAPS[number])
    inputs = {**drive['inputs'], 'signs': signs}
    found = {**drive, 'inputs': inputs, 'speed': speed, 'extra': pack_inputs(inputs, groups)}
    if following is not None:
        found['next'] = following
    return found


def _half_swapped(drive: dict, groups: tuple[str, ...], rng) -> dict:
    passages = [passage for passage in swap_passages(drive) if rng.random() < .5]
    return swapped(drive, passages, groups) if passages else drive


def epoch_drives(drives: list[dict], groups: tuple[str, ...], rng) -> list[dict]:
    """What an epoch of training goes through, new every epoch: the drives with a moved clock and half of their numbers
    swapped, and the night copies. See the module's text. Never for a measure: drives stays as it is."""
    found = []
    for drive in drives:
        if (_hours_plate(drive) & (np.array(drive['partitions']) == 'train')).any():
            copy = night_copy(drive, groups, rng)
            seen = _half_swapped(drive, groups, rng)
            found += [seen] if copy is None else [seen, copy]
        else:
            found.append(_half_swapped(_with_hour(drive, groups, rng.uniform(0, 24)), groups, rng))
    return found


def drive_chunks(length: int, chunk: int, offset: int = 0) -> list[tuple[int, int]]:
    """(from, to) of the pieces of a drive in order, covering every frame once: the first one offset frames long (a
    different offset every epoch moves the cuts), then chunk frames each. The state goes from one to the next."""
    edges = [0, *range(offset or chunk, length, chunk), length]
    return [(first, last) for first, last in zip(edges, edges[1:]) if first < last]


def learning_pieces(partitions: list[str], chunk: int, offset: int = 0, cuts: set[int] = frozenset()) -> list[tuple[int, int, bool, bool]]:
    """(from, to, learn, fresh) of a drive in order, fresh: the memory starts again here. The pieces never mix train with the
    rest, and frames of test and validation are in none: their pictures run through nothing that moves a weight, not even
    through the memory, which starts again after them. cuts: frames it starts again at anyway."""
    def kind(name: str) -> str:
        return 'held' if name in HELD_OUT else name == 'train'

    pieces, first, fresh = [], 0, True
    for last in range(1, len(partitions) + 1):
        if last == len(partitions) or kind(partitions[last]) != kind(partitions[first]) or last in cuts:
            learn = partitions[first] == 'train'
            if kind(partitions[first]) == 'held':
                fresh = True
            else:
                for a, b in drive_chunks(last - first, chunk, offset if learn else 0):
                    pieces.append((first + a, first + b, learn, fresh))
                    fresh = False
            first = last
            fresh = fresh or last in cuts
    return pieces


def _per_second(drive: dict) -> dict:
    """The answers of a drive on its measured frames only, one a second."""
    if 'measured' not in drive:
        return {}
    kept = drive['measured']
    return {'frames': [frame for frame, measured in zip(drive['frames'], kept) if measured],
            'speed_weight': drive['speed_weight'] * kept, 'works_weight': drive['works_weight'] * kept,
            'speed_out': np.where(kept, drive['speed_out'], -1)}


def summary(drives: list[dict]) -> dict:
    """What the labels give the training today: per partition the frames with a speed and the works frames, per drive the
    same with why the rest is out. Frames one a second (the measured ones) at either rate: seconds of the drive."""
    parts = {name: {'speed': 0, 'works_in': 0, 'works_out': 0, 'sites': 0} for name in ('train', 'validation', 'test', 'buffer')}
    rows = []
    for drive in drives:
        names = np.array(drive['partitions'])
        drive = {**drive, **_per_second(drive)}
        for name, part in parts.items():
            here = names == name
            part['speed'] += int(drive['speed_weight'][here].sum())
            part['works_in'] += int((drive['works_weight'][here] * drive['works'][here]).sum())
            part['works_out'] += int((drive['works_weight'][here] * (1 - drive['works'][here])).sum())
        for mark in drive['marks']:
            inside = np.flatnonzero((drive['ns'] >= mark['start_ns']) & (drive['ns'] < mark['end_ns']))
            if len(inside):
                parts[names[inside[len(inside) // 2]]]['sites'] += 1
        rows.append({'route': drive['route'], 'frames': len(drive['frames']), 'speed': int(drive['speed_weight'].sum()),
                     'out': {name: int((drive['speed_out'] == reason).sum()) for reason, name in OUT_REASONS.items()},
                     'works_in': int((drive['works_weight'] * drive['works']).sum()),
                     'works_out': int((drive['works_weight'] * (1 - drive['works'])).sum()), 'marks': len(drive['marks']),
                     'signs_read': drive['signs_read'], 'reader_current': drive.get('reader_current', False),
                     'pictures': drive.get('pictures', True), 'no_picture_s': drive['no_picture_s']})
    return {'partitions': parts, 'drives': rows}


def summary_text(found: dict) -> str:
    lines = ['Per partition (frames, one a second):',
             f"  {'':<11}{'speed':>7}{'works in':>10}{'works out':>11}{'sites':>7}"]
    for name, part in found['partitions'].items():
        lines.append(f"  {name:<11}{part['speed']:>7}{part['works_in']:>10}{part['works_out']:>11}{part['sites']:>7}")
    lines.append('Per drive:')
    for row in found['drives']:
        out = ', '.join(f'{name} {count}' for name, count in row['out'].items() if count)
        signs = 'NOT read' if not row['signs_read'] else 'read' if row['reader_current'] else 'read by ANOTHER reader'
        signs += '' if row.get('pictures', True) else ", the brain's pictures NOT made"
        lines.append(f"  {row['route']}: {row['speed']} of {row['frames']} with a speed ({out}); works {row['works_in']} in, "
                     f"{row['works_out']} out, {row['marks']} marks; signs {signs}; "
                     f"{row['no_picture_s']} labelled seconds without a picture")
    return '\n'.join(lines)
