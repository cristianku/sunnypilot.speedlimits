"""From a drive to what the road labeler sees and what it learns: its steps, its inputs and its answers. No torch here.

What it is for. The road labeler (nnslr_tools.road_labeler) is an offline network that labels the drives of Speed review
so that a person has less to do: the road type of every moment, the kinds of road (a tunnel, a narrow road... the kinds
of speed_road_kinds) and where roadworks start and end. It never runs on the car and is never an input of the car. What
it writes is definitive (his rule of 9 Nov 2026): its road types replace the map's as the base of Speed review, its marks
are answers of the brain, and a person only corrects it where it is wrong; those corrections are what it learns from.
Because it is offline, the OpenStreetMap map IS one of its inputs here (AGENTS.md's rule "vision never reads the map" is
about the car).

The steps (drive_steps): STEP_HZ a second, the 1 Hz frames of the drive (brain_targets.drive_frames, the same frames the
frozen split knows) and the frame of the road video HALF_STEP decoded frames after each, each on the comma's clock by the
encode index (its exposure start). A step's picture is that frame of fcamera.hevc (road_labeler_net decodes it); the
half-second step has the map line, the partition and the road type of its second.

The inputs of a step, besides its picture (step_inputs, INPUT_NAMES in this order):

- the map scene of that second (derived/road_types/<route>.jsonl as nnslr road-types wrote it, map_road_types, never
  with the corrections over it: a person's correction is an answer and never an input): map_type and highway one-hot,
  the maxspeed in km/h (an mph tag converted) with whether it has one, the family of its basis (no gps, no road, a car
  park or a garage of the map, slow without gps, filled, smoothed, the acceleration lane), whether the map is sure, and
  whether the map's way is a tunnel (derived/osm/roads.jsonl's tunnel tag, tunnels.is_tunnel);
- the GPS: whether a fix is within GPS_FIX_S of the step, and the seconds since the last fix and to the next one (over
  GPS_GAP_SCALE_S, at most 1): a garage or a tunnel is where the GPS is lost, and the network looks both ways;
- the car's speed as the dashboard shows it (drive_logs.dashboard_speed), over SPEED_SCALE_KPH, with whether it is known.

The answers (drive_answers), each with a weight a step, 0 where nothing is known:

- road type: one of road_types.ROAD_TYPES. A person's (a correction or a confirmation in Speed review) weighs
  STRONG_WEIGHT; the map's alone is a weak answer, MAP_SURE_WEIGHT where the map is sure and MAP_WEIGHT where not; "can't
  tell" and a second without a type are masked. map_type keeps the map's own type of every step for the measure: on the
  steps a person set, the map's type is what the network has to beat;
- the kinds of road, one column a kind (brain_targets.road_answers): 1 inside a mark with an end, 0 before that kind's
  "checked to here" outside the marks, masked elsewhere. The marks of nnslr tunnels (reviewer 'osm') are answers like a
  person's;
- roadworks: the same with speed_works and speed_works_checked (brain_targets.works_answers);
- a person's removal (removed_marks): a person who removes a mark says that stretch is not of that kind (or not
  roadworks), so its steps answer 0 at STRONG_WEIGHT where nothing else answers, also past "checked to here". Left out:
  the removals of the labeler and of nnslr tunnels (their own undo), a mark removed before it had an end (a start
  tapped by mistake, logged to the drive's end), and the steps within EDGE_SECONDS of a mark of that kind there now (a
  mark moved is removed and set again);
- the tunnel's weak "no" (tunnel_free): where the GPS has a fix and our road of the map (map_context.our_road, the way
  nnslr road-types found that second) is no tunnel, the tunnel column answers 0 at TUNNEL_FREE_WEIGHT where nothing
  else answers, at least TUNNEL_MARGIN_S from any tunnel mark (any reviewer, one with no end too) and from any second
  whose road of the map is a tunnel. The map makes this label offline; it never reaches anything of the car.

What the labeler wrote itself (reviewer NET_PREFIX + its run) is never an answer of its own training, though it is the
brain's: its marks are left out, and their steps (with EDGE_SECONDS around them) masked (own_steps), so that neither
their inside nor a "checked to here" past them teaches it its own word; its road types are a layer the map's lines and a
person's corrections never include here (map_road_types, load_corrections without its lines).

Everything here reads: Speed review through brain_targets.read_labels (read only), speed_history of the removals through
a read-only connection (removed_marks), the map and the drive's logs. No write, no training, no GPU.
"""
from __future__ import annotations

import bisect
import json
import sqlite3
from pathlib import Path

import numpy as np

from nnslr_tools import drive_logs
from nnslr_tools.annotations import load_jsonl
from nnslr_tools.brain_targets import (EDGE_SECONDS, FramesChanged, NO_LABELS, _cell_partition, drive_frames,
                                       frame_partitions, read_labels, read_split, road_answers, road_kinds,
                                       road_labels, works_answers)
from nnslr_tools.map_context import ROADS_RELATIVE
from nnslr_tools.road_type_data import frame_cells
from nnslr_tools.road_types import (NET_PREFIX, ROAD_TYPES, by_labeler, corrected, load_corrections, map_road_types,
                                    maxspeed_limit, person_removals)
from nnslr_tools.speed_review import NS, db_path as speed_review_path

STEP_HZ = 2
VIDEO_FPS = 20
HALF_STEP = VIDEO_FPS // STEP_HZ  # decoded frames from a step to the next
STRONG_WEIGHT = 1.0
MAP_SURE_WEIGHT = 0.3
MAP_WEIGHT = 0.1
# The tunnel's weak "no" (tunnel_free): a GPS fix on a road of the map that is no tunnel is rarely in one, but the map can
# be old or wrong, so it weighs like a sure map; and nothing within TUNNEL_MARGIN_S of a tunnel, whose portals are where
# the map and the marks are least exact.
TUNNEL_FREE_WEIGHT = MAP_SURE_WEIGHT
TUNNEL_MARGIN_S = 5.0
TUNNEL_KIND = 'tunnel'
GPS_FIX_S = 1.0
GPS_GAP_SCALE_S = 60.0
SPEED_SCALE_KPH = 100.0
MAXSPEED_SCALE_KPH = 100.0
MPH = 1.609344
HIGHWAYS = ('motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link',
            'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street', 'service', 'track')
# The families of a map line's basis, by what it starts with or holds (road_types.frame_types).
BASES = ('no gps', 'no road', 'slow without gps', 'parking aisle', 'car park on the map', 'garage on the map', 'filled',
         'smoothed', 'acceleration lane')
PARTITIONS = ('train', 'validation', 'test')


def _names() -> tuple[str, ...]:
    return (*(f'map_type:{name}' for name in ROAD_TYPES), 'map_type:none',
            *(f'highway:{name}' for name in HIGHWAYS), 'highway:other', 'highway:none',
            'maxspeed', 'maxspeed_known', *(f'basis:{name}' for name in BASES), 'basis:other',
            'map_sure', 'map_tunnel', 'map_line',
            'gps_fix', 'gps_since', 'gps_until',
            'speed', 'speed_known')


INPUT_NAMES = _names()
INPUTS = len(INPUT_NAMES)
_AT = {name: index for index, name in enumerate(INPUT_NAMES)}


def labeler_routes(root: Path) -> list[str]:
    """Every drive with a finished scan (drive_review.sqlite, read only) and its 1 Hz frames, in order: the labeler
    labels any drive, dashcam or not (the brain picks its own)."""
    from nnslr_tools import drive_review
    from nnslr_tools.road_type_data import FRAMES_RELATIVE
    path = drive_review.db_path(root)
    if not path.is_file():
        return []
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        rows = connection.execute('SELECT route, summary FROM scans').fetchall()
    finally:
        connection.close()
    found = []
    for route, text in rows:
        try:
            summary = json.loads(text) if text else None
        except ValueError:
            summary = None
        span = summary.get('span_ns') if isinstance(summary, dict) else None
        if summary is None or summary.get('t0_ns') is None or not span or span[-1] is None:
            continue
        if any((Path(root) / FRAMES_RELATIVE / route).glob('*/frames.jsonl')):
            found.append(route)
    return sorted(found)


def tunnel_ways(root: Path) -> set:
    """The ids of the ways of derived/osm/roads.jsonl that are a tunnel (tunnels.is_tunnel); empty without the file."""
    from nnslr_tools.tunnels import is_tunnel
    path = Path(root) / ROADS_RELATIVE
    if not path.is_file():
        return set()
    return {way.get('id') for way in load_jsonl(path) if isinstance(way, dict) and is_tunnel(way)}


def _exposures(merged: dict) -> dict[int, dict[int, int]]:
    """{segment: {decoded frame: exposure start}} of the drive's encode index."""
    found: dict[int, dict[int, int]] = {}
    for _first, segment, sof, ids in drive_logs.frame_index(merged):
        times = found.setdefault(segment, {})
        for decoded, t in zip(ids, sof):
            times.setdefault(decoded, t)
    return found


def drive_steps(root: Path, route: str, merged: dict) -> tuple[list[dict], list[dict]]:
    """(steps, frames): the 1 Hz frames of the drive (brain_targets.drive_frames) and its steps at STEP_HZ, each
    {segment, decoded, step (0 the 1 Hz frame, 1 the half second after), frame (its 1 Hz frame's position in frames),
    key (its 1 Hz frame's key, the map line's), ns (None where the encode index lacks it), image (the 1 Hz PNG)}."""
    frames = drive_frames(root, route, merged)
    exposures = _exposures(merged)
    steps = []
    for at, frame in enumerate(frames):
        for step in range(STEP_HZ):
            decoded = frame['decoded'] + step * HALF_STEP
            ns = frame['ns'] if step == 0 else exposures.get(frame['segment'], {}).get(decoded)
            steps.append({'segment': frame['segment'], 'decoded': decoded, 'step': step, 'frame': at, 'key': frame['key'],
                          'ns': ns, 'image': frame['image']})
    return steps, frames


def step_partitions(root: Path, route: str, frames: list[dict], steps: list[dict], log=None) -> list[str]:
    """The partition of each step: its 1 Hz frame's in the brain's frozen split (brain_targets.read_split), so that the
    places held out of the brain are held out here too; a drive whose frames changed takes its areas' partitions, and
    without a split every step is train (said in log)."""
    try:
        split = read_split(root)
    except (OSError, ValueError, KeyError):
        if log:
            log(f'{route}: no frozen split (splits/speed-brain-split.json): every step is train')
        return ['train'] * len(steps)
    try:
        partitions = frame_partitions(route, frames, split)
    except FramesChanged:
        partitions = [_cell_partition(cell, split['cells']) for cell in frame_cells(frames)]
    return [partitions[step['frame']] for step in steps]


def _maxspeed_kph(value) -> float | None:
    limit = maxspeed_limit(value)
    if limit is None:
        return None
    return float(limit[0]) * (MPH if limit[1] == 'mph' else 1.0)


def _basis(text) -> str:
    text = text or ''
    for name in BASES:
        if name in text:
            return name
    return 'other'


def map_vector(line: dict | None, tunnels: set) -> np.ndarray:
    """The map scene of a second (a line of map_road_types, None without one) as the first inputs of INPUT_NAMES."""
    found = np.zeros(INPUTS, dtype=np.float32)
    if line is None:
        found[_AT['map_type:none']] = 1
        found[_AT['highway:none']] = 1
        return found
    found[_AT['map_line']] = 1
    kind = line.get('road_type')
    found[_AT[f'map_type:{kind}' if kind in ROAD_TYPES else 'map_type:none']] = 1
    highway = line.get('highway')
    found[_AT[f'highway:{highway}' if highway in HIGHWAYS else 'highway:other' if highway else 'highway:none']] = 1
    kph = _maxspeed_kph(line.get('maxspeed'))
    if kph is not None:
        found[_AT['maxspeed']] = min(kph / MAXSPEED_SCALE_KPH, 2.0)
        found[_AT['maxspeed_known']] = 1
    found[_AT[f'basis:{_basis(line.get("basis"))}']] = 1
    found[_AT['map_sure']] = float(line.get('sure') is True)
    found[_AT['map_tunnel']] = float(line.get('way_id') is not None and line.get('way_id') in tunnels)
    return found


def gps_inputs(ns: np.ndarray, gps: list) -> np.ndarray:
    """(steps, 3): a fix within GPS_FIX_S, seconds since the last fix and to the next one over GPS_GAP_SCALE_S (1 at most,
    and 1 before the first fix or after the last). ns -1: no time, all 0 but the gaps at 1."""
    times = sorted(int(point[0]) for point in gps or () if point[1] is not None and point[2] is not None)
    found = np.zeros((len(ns), 3), dtype=np.float32)
    for row, t in enumerate(np.asarray(ns, dtype=np.int64)):
        if t < 0 or not times:
            found[row] = (0.0, 1.0, 1.0)
            continue
        after = bisect.bisect_left(times, int(t))
        since = (t - times[after - 1]) / NS if after > 0 else None
        until = (times[after] - t) / NS if after < len(times) else None
        near = min(value for value in (since, until, float('inf')) if value is not None)
        found[row, 0] = float(near <= GPS_FIX_S)
        found[row, 1] = 1.0 if since is None else min(since / GPS_GAP_SCALE_S, 1.0)
        found[row, 2] = 1.0 if until is None else min(until / GPS_GAP_SCALE_S, 1.0)
    return found


def speed_inputs(ns: np.ndarray, speed: list) -> np.ndarray:
    """(steps, 2): the dashboard's km/h at each step (the last value at or before it) over SPEED_SCALE_KPH, and known."""
    times = [int(point[0]) for point in speed or ()]
    found = np.zeros((len(ns), 2), dtype=np.float32)
    for row, t in enumerate(np.asarray(ns, dtype=np.int64)):
        at = bisect.bisect_right(times, int(t)) - 1 if t >= 0 else -1
        value = speed[at][1] if at >= 0 else None
        if value is not None:
            found[row] = (min(float(value) / SPEED_SCALE_KPH, 2.5), 1.0)
    return found


def step_inputs(steps: list[dict], map_lines: dict[str, dict], tunnels: set, merged: dict) -> np.ndarray:
    """(steps, INPUTS) float32 in the order of INPUT_NAMES."""
    ns = np.array([-1 if step['ns'] is None else step['ns'] for step in steps], dtype=np.int64)
    found = np.stack([map_vector(map_lines.get(step['key']), tunnels) for step in steps]) if steps else \
        np.zeros((0, INPUTS), dtype=np.float32)
    found[:, _AT['gps_fix']:_AT['gps_until'] + 1] = gps_inputs(ns, merged.get('gps') or [])
    found[:, _AT['speed']:_AT['speed_known'] + 1] = speed_inputs(ns, drive_logs.dashboard_speed(merged))
    return found


def own(reviewer) -> bool:
    """A write of the labeler itself (reviewer net/<run>, road_types.by_labeler)."""
    return by_labeler(reviewer)


def type_answers(map_lines: list[dict], corrections: list[dict], steps: list[dict]) -> dict[str, np.ndarray]:
    """{type (class of ROAD_TYPES a step, -1 none), type_weight, type_person (a person set it), map_type (the map's own
    class, -1 none)}: see the module's text. The labeler's own corrections are left out."""
    theirs = [line for line in corrections if not own(line.get('reviewer'))]
    lines = corrected(map_lines, theirs) if theirs else map_lines
    by_key = {line['frame_key']: line for line in lines}
    raw = {line['frame_key']: line for line in map_lines}
    found = {'type': np.full(len(steps), -1, dtype=np.int64), 'type_weight': np.zeros(len(steps), dtype=np.float32),
             'type_person': np.zeros(len(steps), dtype=bool), 'map_type': np.full(len(steps), -1, dtype=np.int64)}
    for row, step in enumerate(steps):
        base = raw.get(step['key'])
        if base is not None and base.get('road_type') in ROAD_TYPES:
            found['map_type'][row] = ROAD_TYPES.index(base['road_type'])
        line = by_key.get(step['key'])
        if line is None or line.get('road_type') not in ROAD_TYPES:
            continue
        found['type'][row] = ROAD_TYPES.index(line['road_type'])
        person = line.get('reviewed') is True
        found['type_person'][row] = person
        found['type_weight'][row] = STRONG_WEIGHT if person else MAP_SURE_WEIGHT if line.get('sure') else MAP_WEIGHT
    return found


def _near(ns: np.ndarray, start: int, end: int | None, margin_ns: int) -> np.ndarray:
    """The steps within margin_ns of [start, end) (to the drive's end without one)."""
    last = np.iinfo(np.int64).max if end is None else end + margin_ns
    return (ns >= start - margin_ns) & (ns < last)


def own_steps(ns: np.ndarray, marks: list[dict]) -> np.ndarray:
    """The steps inside a mark of the labeler's own, or within EDGE_SECONDS of it (to the drive's end without an end):
    out of its own loss, whatever else answers there."""
    ns = np.asarray(ns, dtype=np.int64)
    where = np.zeros(len(ns), dtype=bool)
    for mark in marks:
        if own(mark.get('reviewer')):
            where |= _near(ns, mark['start_ns'], mark['end_ns'], EDGE_SECONDS * NS)
    return where


def removal_answers(ns: np.ndarray, removed: list[dict], marks: list[dict]) -> np.ndarray:
    """The steps a person's removals (removed_marks' of one kind, or of the roadworks) say "no" at: inside a removed
    stretch, at least EDGE_SECONDS from the marks of that kind there now (see the module's text)."""
    ns = np.asarray(ns, dtype=np.int64)
    where = np.zeros(len(ns), dtype=bool)
    for row in removed:
        where |= (ns >= row['start_ns']) & (ns < row['end_ns'])
    for mark in marks:
        where &= ~_near(ns, mark['start_ns'], mark['end_ns'], EDGE_SECONDS * NS)
    return where & (ns >= 0)


def drive_answers(ns: np.ndarray, labels: dict, kinds: tuple[str, ...],
                  removed: list[dict] = ()) -> dict[str, np.ndarray]:
    """{kinds, kinds_weight (steps, kinds), works, works_weight (steps)} from read_labels' of the drive and the person's
    removals of removed_marks: see the module's text. A step without a time (ns -1) answers nothing."""
    ns = np.asarray(ns, dtype=np.int64)
    checked = labels.get('road_checked') or {}
    road = labels.get('road') or []
    target, weight = road_answers(ns, [mark for mark in road if not own(mark.get('reviewer'))], checked, kinds)
    works = labels.get('works') or []
    works_target, works_weight = works_answers(ns, [mark for mark in works if not own(mark.get('reviewer'))],
                                               labels.get('checked_ns'))
    for column, kind in enumerate(kinds):
        weight[own_steps(ns, [mark for mark in road if mark['kind'] == kind]), column] = 0
    works_weight[own_steps(ns, works)] = 0
    for column, kind in enumerate(kinds):
        fill = removal_answers(ns, [row for row in removed if row['kind'] == kind],
                               [mark for mark in road if mark['kind'] == kind]) & (weight[:, column] == 0)
        target[fill, column], weight[fill, column] = 0, STRONG_WEIGHT
    fill = removal_answers(ns, [row for row in removed if row['kind'] is None], works) & (works_weight == 0)
    works_target[fill], works_weight[fill] = 0, STRONG_WEIGHT
    known = ns >= 0
    weight[~known] = 0
    works_weight[~known] = 0
    return {'kinds': target, 'kinds_weight': weight, 'works': works_target, 'works_weight': works_weight}


def tunnel_kind(root: Path) -> str | None:
    """The kind of road of the tunnels in Speed review (its kind or its label "tunnel", as tunnels.write_tunnels finds
    it), None without one."""
    return next((kind for kind, label in road_labels(root).items()
                 if kind == TUNNEL_KIND or str(label).lower() == TUNNEL_KIND), None)


def tunnel_free(ns: np.ndarray, steps: list[dict], map_lines: dict[str, dict], tunnels: set, gps_fix: np.ndarray,
                marks: list[dict]) -> np.ndarray:
    """The steps where the tunnel column answers its weak "no": see the module's text. marks: the tunnel marks of the
    drive, of any reviewer."""
    ns = np.asarray(ns, dtype=np.int64)
    margin = int(TUNNEL_MARGIN_S * NS)
    ways = [(map_lines.get(step['key']) or {}).get('way_id') for step in steps]
    free = np.array([way is not None and way not in tunnels for way in ways], dtype=bool)
    free &= (np.asarray(gps_fix) > 0) & (ns >= 0)
    for mark in marks:
        free &= ~_near(ns, mark['start_ns'], mark['end_ns'], margin)
    inside = np.sort(ns[np.array([way is not None and way in tunnels for way in ways], dtype=bool) & (ns >= 0)])
    if len(inside) and len(ns):
        at = np.searchsorted(inside, ns)
        before = np.abs(ns - inside[np.clip(at - 1, 0, len(inside) - 1)])
        after = np.abs(inside[np.clip(at, 0, len(inside) - 1)] - ns)
        free &= np.minimum(before, after) >= margin
    return free


def drive_data(root: Path, route: str, merged: dict, *, labels: dict | None = None, kinds: tuple[str, ...] = (),
               tunnels: set | None = None, removed: list[dict] = (), tunnel: str | None = None, log=None) -> dict:
    """Everything of a drive the training and the prediction need, but the pictures: route, steps, frames, ns (-1 without
    a time), partitions, inputs (steps, INPUTS), the answers of type_answers and drive_answers (with removed, the
    person's removals of removed_marks, and the tunnel's weak "no" of tunnel_free in the column of the kind tunnel),
    kind_names and map_lines (the map's, in order: what the labeler's road types are written over)."""
    steps, frames = drive_steps(root, route, merged)
    ns = np.array([-1 if step['ns'] is None else step['ns'] for step in steps], dtype=np.int64)
    map_lines = map_road_types(root, route)
    corrections = load_corrections(root, route) if map_lines else []
    by_key = {line['frame_key']: line for line in map_lines}
    tunnels = tunnel_ways(root) if tunnels is None else tunnels
    labels = NO_LABELS if labels is None else labels
    data = {'route': route, 'steps': steps, 'frames': frames, 'ns': ns,
            'partitions': step_partitions(root, route, frames, steps, log),
            'inputs': step_inputs(steps, by_key, tunnels, merged),
            'kind_names': list(kinds), 'map_lines': map_lines}
    data.update(type_answers(map_lines, corrections, steps))
    data.update(drive_answers(ns, labels, kinds, removed))
    if tunnel in kinds:
        column = kinds.index(tunnel)
        marks = [mark for mark in labels.get('road') or [] if mark['kind'] == tunnel]
        free = tunnel_free(ns, steps, by_key, tunnels, data['inputs'][:, _AT['gps_fix']], marks)
        free &= data['kinds_weight'][:, column] == 0
        data['kinds'][free, column] = 0
        data['kinds_weight'][free, column] = TUNNEL_FREE_WEIGHT
    return data


def dataset(root: Path, merged_of, routes: list[str] | None = None, log=None) -> list[dict]:
    """drive_data of every drive of labeler_routes (or of routes), with the kinds of road of Speed review of today in the
    same columns (brain_targets.road_kinds, read only). merged_of(route): the drive's joined logs, None before a scan."""
    kinds = road_kinds(root)
    labels = read_labels(root)
    tunnels = tunnel_ways(root)
    tunnel = tunnel_kind(root)
    known = labeler_routes(root)
    drives = []
    for route in routes if routes is not None else known:
        if route not in known:
            if log:
                log(f'{route}: no finished scan or no frames, left out')
            continue
        merged = merged_of(route)
        if not merged:
            if log:
                log(f'{route}: no scan, left out')
            continue
        drives.append(drive_data(root, route, merged, labels=labels.get(route, NO_LABELS), kinds=kinds, tunnels=tunnels,
                                 removed=removed_marks(root, route), tunnel=tunnel, log=log))
    return drives


def both_sides(drives: list[dict], partition: str = 'train') -> dict:
    """What each kind and the roadworks have to learn from in a partition: {name: {'inside', 'outside'}} in steps of
    weight. A side alone teaches a constant (the tunnels of the map with no "checked to here" are all inside)."""
    found = {}
    for drive in drives:
        here = np.array(drive['partitions']) == partition
        for column, kind in enumerate(drive['kind_names']):
            weight = drive['kinds_weight'][:, column] * here
            item = found.setdefault(kind, {'inside': 0, 'outside': 0})
            item['inside'] += int((weight * drive['kinds'][:, column]).sum())
            item['outside'] += int((weight * (1 - drive['kinds'][:, column])).sum())
        weight = drive['works_weight'] * here
        item = found.setdefault('works', {'inside': 0, 'outside': 0})
        item['inside'] += int((weight * drive['works']).sum())
        item['outside'] += int((weight * (1 - drive['works'])).sum())
    return found


def keep_both_sides(drives: list[dict], log=None) -> dict:
    """both_sides of train; a kind (or the roadworks) with one side alone in train leaves the loss of train, said in log.
    Returns both_sides."""
    sides = both_sides(drives)
    for name, item in sides.items():
        if item['inside'] and item['outside']:
            if log:
                log(f'"{name}" in train: {item["inside"]} steps inside, {item["outside"]} outside.')
            continue
        if log and (item['inside'] or item['outside']):
            log(f'"{name}" in train: {item["inside"]} steps inside, {item["outside"]} outside: one side alone teaches '
                'nothing, left out. It needs "checked to here" in Speed review.')
        for drive in drives:
            train = np.array(drive['partitions']) == 'train'
            if name == 'works':
                drive['works_weight'] = drive['works_weight'] * ~train
            elif name in drive['kind_names']:
                drive['kinds_weight'][:, drive['kind_names'].index(name)] *= ~train
    return sides


def removed_marks(root: Path, route: str) -> list[dict]:
    """The marks of a drive a person removed (person_removals of its speed_history, read only): answers "no" of their
    kind (drive_answers). [] without the database."""
    path = speed_review_path(root)
    if not path.is_file():
        return []
    connection = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return person_removals([dict(row) for row in connection.execute(
            "SELECT action, state, start_ns, end_ns, reviewer FROM speed_history WHERE route = ? AND action IN "
            "('road_start', 'road_end', 'road_removed', 'works_start', 'works_end', 'works_removed') ORDER BY revision",
            (route,))])
    except sqlite3.OperationalError:
        return []
    finally:
        connection.close()
