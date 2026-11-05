"""From Speed review to the brain's training: what every frame of a drive teaches, and what it may look at.

What it is for. A label in annotations/speed_review.sqlite changes nothing by itself. It reaches the network only as an
answer on a frame: the training compares the answer with what the network said, and that error moves the weights
(brain_net.brain_loss). This module makes those answers. It reads the labels and writes nothing: no training, no GPU.

A drive here is its 1 Hz frames in drive order, the same pictures as the road-type network (derived/frames), each on the
comma's clock by the encode index (drive_logs.frame_time): labels and frames meet on logMonoTime, never on a count of
seconds. The list of frames needs no road types: the road type is only data now, and no field of the map comes in here.

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

The split (frame_partitions) is the frozen one, data/splits/speed-brain-split.json, never computed again: a drive of the
freeze joins it frame by frame through speed-brain-frames.json and fails loudly when its frames changed; a later drive
goes by the area of each frame, and an area the freeze never saw is train, or buffer next to a held-out one or without any
position. It is a mask on the loss: a drive stays whole, so that its memory runs through it as on the car.

Works sites stay whole (keep_sites_whole): a site with a frame in test or validation, SITE_MARGIN_S around it, has no
frame in train, and neither has any other site on those areas, on another drive too. So the measure is on sites the
network never learned, approach and exit included.

The inputs (drive_inputs), one group each so that a comparison can leave one out, all made only from what comes at or
before the frame:

- signs: the last number read, whether a display showed it, its age, whether an end was read since, a number read in this
  second. From the car's own code on the server (road_type_signs.readings). A drive not read yet is unknown throughout;
- lead: distance and speed of the car ahead;
- country: where the frame is, from the GPS and the borders (gps.py): a country, no map;
- pedals: gas and brake in the last second;
- speed: the car's own speed.

All but the two of the driver are made by the car's own file, sunnypilot/speed_limits/brain_inputs.py, loaded here by its path
(BRAIN_INPUTS): what the network learns on is what it gets on the car, and the checkpoint keeps the file's hash.

pedals and speed are the driver, and they are apart for a reason: on the drives the gas is often pressed just where the
label is above the limit the car held (he overrode it), and the car's speed is what the network itself decides on the
car. A network that has them can learn to follow the driver, so they stay out until a comparison says otherwise.

The sequences (drive_chunks, learning_pieces): the whole drive in order, state carried from piece to piece, so that
approach, crossing and exit are one sequence and the network sees only the past, as on the car. The pieces only bound the
gradient, and they never mix train with the rest: nnslr_tools.brain_train. The memory starts again after MEMORY_GAP_SECONDS
without a frame (memory_cuts), as on the car.

What the training refuses (check_drives), so that nothing goes wrong without a word: a drive not read or read by another
reader than the car's today, since an unread drive looks like a drive with no sign and then "no reading" is learned as a
speed; a drive with no frame; a validation with no answer, which would keep the first epoch.
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
from nnslr_tools.gps import load_positions
from nnslr_tools.road_type_data import FRAMES_RELATIVE, frame_cells, frame_path, speed_features
from nnslr_tools.road_type_signs import fingerprint, load_signs, package_path
from nnslr_tools.speed_review import GRID_KPH, MAX_KPH, NS, db_path

SPLIT_RELATIVE = Path('splits') / 'speed-brain-split.json'
SPLIT_FRAMES_RELATIVE = Path('splits') / 'speed-brain-frames.json'
SPEED_CLASSES = MAX_KPH // GRID_KPH
EDGE_SECONDS = 2
SITE_MARGIN_S = 60
READ_COMMAND = 'nnslr road-type-signs --device cpu --num-workers 8'
NOT_EXAMINED, UNDETERMINED, NOT_KNOWN_YET, BOUNDARY, NO_FRAME = range(1, 6)
OUT_REASONS = {NOT_EXAMINED: 'not_examined', UNDETERMINED: 'undetermined', NOT_KNOWN_YET: 'not_known_yet', BOUNDARY: 'boundary',
               NO_FRAME: 'no_frame'}
HELD_OUT = ('test', 'validation')


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
COUNTRIES = BRAIN_INPUTS.COUNTRIES
INPUT_SIZES = {**BRAIN_INPUTS.INPUT_SIZES, 'pedals': 3, 'speed': 2}
DRIVER_INPUTS = ('pedals', 'speed')
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
    """The 1 Hz frames of a drive in drive order: key, segment, index, ns (the exposure start; None when the encode index
    lacks it), latitude, longitude, country, image, present."""
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
                           'ns': drive_logs.frame_time(index, segment, row['decoded_frame_index']),
                           'latitude': gps.get('latitude'), 'longitude': gps.get('longitude'), 'country': gps.get('country'),
                           'image': str(image), 'present': image.is_file()})
    frames.sort(key=lambda frame: (frame['segment'], frame['index']))
    return frames


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


def sign_inputs(seconds: np.ndarray, readings: list | None) -> np.ndarray:
    """The signs group (frames, size) from the sure readings [[seconds, [[class, confidence], ...]], ...] of a drive, by the
    car's own file. Only readings at or before the frame count; None: the drive was not read, all unknown."""
    out = np.zeros((len(seconds), INPUT_SIZES['signs']), dtype=np.float32)
    if readings is None:
        return out
    ordered = sorted(readings, key=lambda item: item[0])
    signs, pointer = BRAIN_INPUTS.SignInputs(), 0
    for row, second in enumerate(seconds):
        while pointer < len(ordered) and ordered[pointer][0] <= second:
            signs.add(*ordered[pointer])
            pointer += 1
        out[row] = signs.features(second)
    return out


def drive_inputs(frames: list[dict], ns: np.ndarray, merged: dict, readings: list | None) -> dict[str, np.ndarray]:
    """{group: (frames, size) float32}: see the module's text. Nothing after a frame reaches it."""
    series = merged.get('series') or {}
    seconds = np.array([frame['segment'] * 60 + frame['index'] for frame in frames], dtype=np.float64)
    distance, lead_speed = _last_at(series.get('lead_d') or [], ns), _last_at(series.get('lead_v') or [], ns)
    lead = np.array([BRAIN_INPUTS.lead_row(metres, kph) for metres, kph in zip(distance, lead_speed)], dtype=np.float32)
    gas, gas_known = _pressed(series.get('gas') or [], ns)
    brake, brake_known = _pressed(series.get('brake') or [], ns)
    speed = [speed_features(None if kph is None else kph / 3.6) for kph in _last_at(series.get('v_ego') or [], ns)]
    country = np.array([BRAIN_INPUTS.country_row(frame['country']) for frame in frames], dtype=np.float32)
    return {'signs': sign_inputs(seconds, readings), 'lead': lead.reshape(len(ns), INPUT_SIZES['lead']),
            'country': country.reshape(len(ns), INPUT_SIZES['country']),
            'pedals': np.stack([gas, brake, gas_known * brake_known], axis=1),
            'speed': np.array(speed, dtype=np.float32).reshape(len(ns), INPUT_SIZES['speed'])}


def inputs_sha256() -> str:
    """The hash of the car's file of the inputs, for the checkpoint: the car refuses a network made with another."""
    return hashlib.sha256(BRAIN_INPUTS_PATH.read_bytes()).hexdigest()


def memory_cuts(ns: np.ndarray) -> set[int]:
    """The frames the memory starts again at: after MEMORY_GAP_SECONDS without a frame."""
    known = np.flatnonzero(ns >= 0)
    return {int(at) for at in known[1:][np.diff(ns[known]) > BRAIN_INPUTS.MEMORY_GAP_SECONDS * NS]}


def drive_targets(root: Path, route: str, labels: dict, merged: dict, split: dict) -> dict:
    """A drive for the training: frames, ns, partitions, cells, cuts (memory_cuts), speed (class per frame), speed_out (why a
    frame is out of the loss, 0 when in), speed_weight, works, works_weight, inputs, own_kph (the car's speed, NaN unknown:
    for the measure, no input), signs_read, reader (what read them), no_picture_s (labelled seconds without a frame at all:
    prepare keeps only the slow ones at the start and the end of a drive)."""
    frames = drive_frames(root, route, merged)
    ns = np.array([-1 if frame['ns'] is None else frame['ns'] for frame in frames], dtype=np.int64)
    usable = np.array([frame['ns'] is not None and frame['present'] for frame in frames], dtype=bool)
    speed, speed_out = speed_answers(ns, labels['stretches'])
    speed_out[~usable] = NO_FRAME
    speed[~usable] = -1
    works, works_weight = works_answers(ns, labels['works'], labels['checked_ns'])
    works_weight[~usable] = 0
    record = load_signs(root, route)
    readings = record['readings'] if record else None
    own = _last_at((merged.get('series') or {}).get('v_ego') or [], ns)
    return {'route': route, 'frames': frames, 'ns': ns, 'partitions': frame_partitions(route, frames, split),
            'own_kph': np.array([np.nan if kph is None else kph for kph in own], dtype=np.float32),
            'cells': frame_cells(frames), 'cuts': memory_cuts(ns), 'speed': speed, 'speed_out': speed_out,
            'speed_weight': (speed_out == 0).astype(np.float32), 'works': works, 'works_weight': works_weight,
            'marks': [mark for mark in labels['works'] if mark['end_ns'] is not None],
            'inputs': drive_inputs(frames, ns, merged, readings), 'signs_read': readings is not None,
            'reader': record.get('fingerprint') if record else None,
            'no_picture_s': max(round(sum(row['end_ns'] - row['start_ns'] for row in labels['stretches']) / NS) -
                                int(sum(((ns >= row['start_ns']) & (ns < row['end_ns'])).sum() for row in labels['stretches'])), 0)}


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


def dataset(root: Path, merged_of, routes: list[str] | None = None, log=_say) -> list[dict]:
    """Every drive with a label or a mark (or only routes), as drive_targets, each with reader_current: read by the car's
    reader of today. merged_of(route): the drive's joined logs, None before a scan. Sites stay whole only across the drives
    asked for."""
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
        drive = drive_targets(root, route, labels, merged, split)
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
    if not any(loss_weights(drive, 'validation')[0].any() for drive in drives):
        raise ValueError('no speed labels in validation: no way to choose the epoch')


def loss_weights(drive: dict, partition: str) -> tuple[np.ndarray, np.ndarray]:
    """(speed weight, works weight) of a drive in one partition: the rest of the drive still runs through the memory."""
    inside = np.array([name == partition for name in drive['partitions']], dtype=np.float32)
    return drive['speed_weight'] * inside, drive['works_weight'] * inside


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


def summary(drives: list[dict]) -> dict:
    """What the labels give the training today: per partition the frames with a speed and the works frames, per drive the
    same with why the rest is out."""
    parts = {name: {'speed': 0, 'works_in': 0, 'works_out': 0, 'sites': 0} for name in ('train', 'validation', 'test', 'buffer')}
    rows = []
    for drive in drives:
        names = np.array(drive['partitions'])
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
                     'no_picture_s': drive['no_picture_s']})
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
        lines.append(f"  {row['route']}: {row['speed']} of {row['frames']} with a speed ({out}); works {row['works_in']} in, "
                     f"{row['works_out']} out, {row['marks']} marks; signs {signs}; "
                     f"{row['no_picture_s']} labelled seconds without a picture")
    return '\n'.join(lines)
