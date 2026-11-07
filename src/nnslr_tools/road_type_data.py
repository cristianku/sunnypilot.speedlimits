"""The data of the road-type network with memory: every frame of every drive with its kind of road, in drive order.

The car needs the kind of road it is on when no sign says the limit: on a Swiss on-ramp a 60 sign is never ended by a
sign, the motorway's 120 starts where the ramp merges (vision.SpeedLimitMemory). The network sees the road from the
camera, one frame a second, and keeps a memory (a GRU) of what it saw, so that the moment of the merge, the entry
into a town and the exit from it are found from the change of the picture, not from a sign.

Labels: the kind of road a person gave a frame in Data › Roads (until 7 Nov 2026), confirming the map's proposal or correcting it
(road_types.confirmed, annotations/road_types): nothing else is a label. The classes are road_types.ROAD_TYPES, in the order
of the network's output and with the same names the car's SpeedLimitMemory takes: town, rural, expressway, motorway,
ramp (an OSM link of a motorway or an expressway, highway motorway_link or trunk_link out of town), parking (a car
park in the open), zone30 (a 30 zone), garage (a car park underground or indoors) and merge (the acceleration lane of
an entry, where the ramp's limit ends: road_types.merge_lanes); zone30, garage and merge were added last on 30 Oct
2026: the car learns them with its next road-type network.

A frame's label and loss weight (frame_label), the same for the training, the validation and the test:

* a person confirmed or corrected it: its type, weight 1;
* everything else: no label, weight 0, out of the loss and of every measure, still fed to the memory. So the map's
  proposal, however sure (nnslr road-types, data/derived/road_types: "sure", an OSM link of a ramp, an acceleration
  lane), a frame without a road type and a person's "can't tell". Until 4 Nov 2026 the proposals were labels too, at
  weight 1 or 0.3: a frame beside a ramp was taught as the ramp (18/57 of 00000168--2b00afdc52), and the measures took
  the map for the truth. The numbers of a network trained then do not compare with a later one's.

A frame also keeps the road type read so far, a person's over the map's (map_label, map_lead): it says only where the
sign input of source 'map' stands (road_type_signs.map_sign_times), an input like the speed. It is never a label.

Enough to train: a training stops, with what is missing, while the partitions hold fewer confirmed frames than
ENOUGH_LABELS asks (missing_labels). It never goes back to the proposals.

Split (SPLIT_POLICY, like splits.py's place-v4 but over frames): the drives pass the same roads again and again, so a
split by drive or by random frames would test the network on roads it learned. The frames are grouped in areas of
about 300 m (splits.PLACE_CELL, the GPS of the frame, else of the nearest frame of the drive with one). The test is
every area the test drives of the sign split went through (data/splits/latest.json: the same held-out drive), so the
whole test drive is measured as it was driven, with its ramps and towns. The areas next to the test are the buffer.
Validation grows in VALIDATION_CLUSTERS compact clusters of areas (seeded where it adds the classes it lacks, never
without a class of MAIN_CLASS_SHARE of the frames) to about EVALUATION_SHARE of the labelled frames, through the areas
in between too: the frames a person confirmed lie scattered, and a cluster of labelled areas alone stops at the first
gap; the areas next to it are the buffer too. Everything else is train, so the split moves as a person reviews. A
sequence is a run of frames of one drive, without a missing frame, in one partition: the network is trained and
measured on whole stretches, never on lone frames.

Junction labels (the junction head, nnslr junctions, data/derived/junctions, with a person's corrections from the
review page over them, annotations/junctions): per frame junction_in (inside an intersection of public roads),
junction_passed (the first frame after leaving it: where the limit of a Polish sign ends) and junction_known (the
drive's junctions were found and the frame has a GPS fix; else out of the junction loss).

The pictures: the full road frame (1344 x 760) resized to FRAME_WIDTH x FRAME_HEIGHT with PIL's bilinear resize, the
same arithmetic as the car's vision.resize, kept as uint8 in data/derived/road_type_frames (build_frame_cache).
numpy and PIL only: the GPU part is road_type_net.
"""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nnslr_tools.junctions import labelled_junctions
from nnslr_tools.road_types import ROAD_TYPES, ROAD_TYPES_RELATIVE, confirmed, load_road_types
from nnslr_tools.splits import PLACE_CELL

CLASSES = ROAD_TYPES
CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}
# What a run records of its labels: a person's only (frame_label). Before it the map's proposals were labels too.
LABEL_POLICY = 'confirmed-only'
# A frame and its car speed: GPS speed in m/s, clipped and scaled (on the car: carState.vEgo), and a flag that says
# whether the speed is known.
SPEED_MAX = 60.0
SPEED_SCALE = 30.0
FRAME_WIDTH, FRAME_HEIGHT = 256, 144
SOURCE_SIZE = (1344, 760)
FRAMES_PER_SEGMENT = 60
FRAMES_RELATIVE = Path('derived') / 'frames'
GPS_RELATIVE = Path('derived') / 'gps'
CACHE_RELATIVE = Path('derived') / 'road_type_frames'
CACHE_VERSION = 1
SPLITS_RELATIVE = Path('splits') / 'latest.json'
SPLIT_POLICY = 'road-type-place-v3'
PARTITIONS = ('train', 'validation', 'test')
EVALUATION_SHARE = .15
VALIDATION_CLUSTERS = 2
# A class with this share of the candidates' frames is never left out of the validation: the epoch is chosen there.
MAIN_CLASS_SHARE = .1
# Shorter runs of one partition (frames) are dropped: too short to train or measure a memory.
MIN_SEQUENCE = 30
# The fewest confirmed frames in the sequences of each partition (the only frames a training sees) and of each kind of
# road the car's rules turn on: the ramp, the motorway after it, the entry into a town and the exit from it. The other
# classes may be missing: a class without frames is not learned and stays out of the macro F1. Fewer stops a training
# (missing_labels); a validation without a ramp would choose the epoch blind to it (31 Oct: the same without the
# motorway gave a motorway recall of 0.11 on test).
ENOUGH_LABELS = {'train': 2000, 'validation': 300, 'test': 300}
NEEDED_CLASSES = ('town', 'rural', 'motorway', 'ramp')
ENOUGH_OF_A_CLASS = 30
UNSEEN_EXIT_S = 4.0  # seconds after a passage whose exit was not seen in which a "passed" is not judged


def frame_label(line: dict) -> tuple[int, float]:
    """(class index, loss weight) of a road-types line: (class, 1.0) where a person confirmed or corrected its type,
    else (-1, 0.0), however sure the map is (see the module's text)."""
    return (CLASS_INDEX[line['road_type']], 1.0) if confirmed(line) else (-1, 0.0)


def speed_features(speed_mps) -> tuple[float, float]:
    """The speed input of a frame: (speed clipped to 0..SPEED_MAX m/s / SPEED_SCALE, 1) or (0, 0) when unknown."""
    if type(speed_mps) not in (int, float) or not math.isfinite(speed_mps):
        return 0.0, 0.0
    return min(max(float(speed_mps), 0.0), SPEED_MAX) / SPEED_SCALE, 1.0


def _seconds(line: dict) -> float:
    return line['segment_index'] * FRAMES_PER_SEGMENT + line['output_index']


def frame_path(root: Path, route: str, segment: int, index: int) -> Path:
    return Path(root) / FRAMES_RELATIVE / route / str(segment) / f'frame_{index:08d}.png'


def road_type_routes(root: Path) -> list[str]:
    """The drives with a road-types file, in name order."""
    folder = Path(root) / ROAD_TYPES_RELATIVE
    return sorted(path.stem for path in folder.glob('*.jsonl')) if folder.is_dir() else []


def _positions(root: Path, route: str) -> dict[str, dict]:
    path = Path(root) / GPS_RELATIVE / route / 'positions.jsonl'
    if not path.is_file():
        return {}
    found = {}
    for text in path.read_text(encoding='utf-8').splitlines():
        if text.strip():
            line = json.loads(text)
            found[line['frame_key']] = line
    return found


def route_frames(root: Path, route: str) -> list[dict]:
    """The frames of a drive in drive order: key, segment, index, label, weight, reviewed, sure, basis, lead (an
    acceleration lane frame that was the ramp's: road_types), map_label and map_lead (the same two of the road type read
    so far, for the sign input alone: see the module's text), latitude, longitude, speed_mps, image (path), present (the picture is
    on disk) and the junction labels (junction_in, junction_passed,
    junction_known, junction: the intersection's id; junction_ignore: near the end of a passage whose exit was not
    seen, where a "passed" is neither right nor wrong; distance_m: metres driven since the start of the drive)."""
    positions = _positions(root, route)
    record = labelled_junctions(root, route)
    junctions = {line['frame_key']: line for line in record['frames']} if record else {}
    # A passage whose exit was not seen (the GPS stops in it): no "passed" is labelled, so none is wrong there.
    unseen = [(p['t_in'], p['t_out']) for p in record['passages'] if not p['passed_key']] if record else []
    frames = []
    for line in load_road_types(root, route):
        label, weight = frame_label(line)
        map_label = CLASS_INDEX.get(line.get('road_type'), -1)
        gps = positions.get(line['frame_key'], {})
        junction = junctions.get(line['frame_key'])
        image = frame_path(root, route, line['segment_index'], line['output_index'])
        frames.append({
            'route': route, 'key': line['frame_key'], 'segment': line['segment_index'], 'index': line['output_index'],
            'label': label, 'weight': weight, 'reviewed': line.get('reviewed') is True,
            'sure': bool(line.get('sure')), 'basis': line.get('basis'),
            'lead': label == CLASS_INDEX['merge'] and bool(line.get('lead')),
            'map_label': map_label, 'map_lead': map_label == CLASS_INDEX['merge'] and bool(line.get('lead')),
            'latitude': gps.get('latitude'), 'longitude': gps.get('longitude'), 'speed_mps': gps.get('speed_mps'),
            'image': str(image), 'present': image.is_file(),
            'junction_known': bool(junction and junction['known']),
            'junction_in': bool(junction and junction['in_junction']),
            'junction_passed': bool(junction and junction['passed']),
            'junction': junction['junction'] if junction else None,
            'distance_m': junction['distance_m'] if junction else None,
            'junction_ignore': any(t_in <= _seconds(line) <= t_out + UNSEEN_EXIT_S for t_in, t_out in unseen)})
    frames.sort(key=lambda frame: (frame['segment'], frame['index']))
    return frames


def _cell(frame: dict) -> tuple[int, int] | None:
    latitude, longitude = frame['latitude'], frame['longitude']
    if type(latitude) not in (int, float) or type(longitude) not in (int, float):
        return None
    return int(latitude // PLACE_CELL[0]), int(longitude // PLACE_CELL[1])


def frame_cells(frames: list[dict]) -> list[tuple[int, int] | None]:
    """The area of each frame of a drive: its own GPS, else the nearest frame's (in drive order) with a position;
    None for a drive without any position."""
    own = [_cell(frame) for frame in frames]
    known = [i for i, cell in enumerate(own) if cell is not None]
    if not known:
        return own
    cells, pointer = [], 0
    for i, cell in enumerate(own):
        if cell is not None:
            cells.append(cell)
            continue
        while pointer + 1 < len(known) and abs(known[pointer + 1] - i) <= abs(known[pointer] - i):
            pointer += 1
        cells.append(own[known[pointer]])
    return cells


def _around(cells) -> set[tuple[int, int]]:
    return {(a + da, b + db) for a, b in cells for da in (-1, 0, 1) for db in (-1, 0, 1)}


def test_routes(root: Path) -> list[str]:
    """The held-out test drives of the sign split in use (data/splits/latest.json), [] without one."""
    latest = Path(root) / SPLITS_RELATIVE
    if not latest.is_file():
        return []
    pointer = json.loads(latest.read_text(encoding='utf-8'))
    split = Path(root) / pointer['path']
    if not split.is_file():
        return []
    return list(json.loads(split.read_text(encoding='utf-8')).get('test_routes') or [])


def place_split(frames_by_route: dict[str, list[dict]], tests: list[str] | tuple[str, ...], *,
                share: float = EVALUATION_SHARE, clusters: int = VALIDATION_CLUSTERS) -> dict:
    """{'policy', 'test_routes', 'cells': {"lat,lon": partition}, 'partitions': {route: [partition per frame]},
    'counts': {partition: frames}} (see the module's text). A drive without any GPS position goes to train
    whole, unless it is a test drive (then test)."""
    tests = [route for route in tests if route in frames_by_route]
    if not tests:
        raise ValueError('no test drive with road types: run nnslr road-types on the held-out drive first')
    cells_of = {route: frame_cells(frames) for route, frames in frames_by_route.items()}
    weight_of: dict[tuple[int, int], int] = {}
    classes_of: dict[tuple[int, int], dict[int, int]] = {}
    for route, frames in frames_by_route.items():
        for frame, cell in zip(frames, cells_of[route]):
            if cell is None:
                continue
            weight_of.setdefault(cell, 0)
            counts_here = classes_of.setdefault(cell, {})
            if frame['label'] >= 0:
                weight_of[cell] += 1
                counts_here[frame['label']] = counts_here.get(frame['label'], 0) + 1
    test = {cell for route in tests for cell in cells_of[route] if cell is not None}
    shield = _around(test) - test
    candidates = {cell for cell in weight_of if cell not in test and cell not in shield}
    validation = _validation_cells(candidates, weight_of, classes_of, share, max(clusters, 1))
    shield |= _around(validation) - validation - test
    partition_of = {}
    for cell in set(weight_of) | test | shield:
        partition_of[cell] = ('test' if cell in test else 'validation' if cell in validation
                              else 'buffer' if cell in shield else 'train')
    partitions, counts = {}, {name: 0 for name in (*PARTITIONS, 'buffer')}
    for route, frames in frames_by_route.items():
        whole = 'test' if route in tests else 'train'
        assigned = [whole if route in tests else (partition_of.get(cell, 'train') if cell is not None else whole)
                    for cell in cells_of[route]]
        partitions[route] = assigned
        for partition, frame in zip(assigned, frames):
            if frame['label'] >= 0:
                counts[partition] += 1
    return {'policy': SPLIT_POLICY, 'test_routes': sorted(tests), 'cell': list(PLACE_CELL),
            'share': share, 'clusters': clusters,
            'cells': {f'{a},{b}': partition_of[(a, b)] for a, b in sorted(partition_of)},
            'partitions': partitions, 'counts': counts}


def _cluster(seed, candidates: set, taken: set, weight_of: dict, cap: float) -> set:
    """Areas grown breadth first from seed over neighbouring candidates (not taken) until they hold cap frames."""
    cluster, size, frontier = set(), 0, [seed]
    while frontier and size < cap:
        cell = frontier.pop(0)
        if cell in cluster or cell in taken or cell not in candidates:
            continue
        cluster.add(cell)
        size += weight_of[cell]
        frontier.extend(sorted(n for n in _around([cell]) if n in candidates and n not in cluster and n not in taken))
    return cluster


def _validation_cells(candidates: set, weight_of: dict, classes_of: dict, share: float, clusters: int) -> set:
    """Validation areas: clusters compact clusters, each holding about share / clusters of the candidates' labelled
    frames, the one at each step whose seed brings every class closest to its share of the candidates' frames (the
    ramps sit at interchanges: a cluster grown blindly on one takes most of them from the training) with the fewest
    training frames lost to its buffer. A cluster that leaves out a class of MAIN_CLASS_SHARE of the frames, or one of
    NEEDED_CLASSES, loses to any that holds it: with eight classes the rare ones weigh as much as the motorway, and a
    validation without the motorway chose the epoch blind to it (31 Oct: motorway recall 0.11 on test)."""
    pool: dict[int, int] = {}
    for cell in candidates:
        for label, count in classes_of[cell].items():
            pool[label] = pool.get(label, 0) + count
    total = sum(weight_of[cell] for cell in candidates) or 1
    main = [label for label, count in pool.items()
            if count >= MAIN_CLASS_SHARE * total or CLASSES[label] in NEEDED_CLASSES]
    chosen: set = set()
    for step in range(1, clusters + 1):
        target = share * step / clusters
        best, best_score = None, None
        for seed in sorted(cell for cell in candidates - _around(chosen) if weight_of[cell]):
            cluster = _cluster(seed, candidates, _around(chosen), weight_of, share * total / clusters)
            together = chosen | cluster
            have: dict[int, int] = {}
            for cell in together:
                for label, count in classes_of[cell].items():
                    have[label] = have.get(label, 0) + count
            lost = sum(weight_of[cell] for cell in (_around(together) - together) & candidates)
            score = (sum((have.get(label, 0) / count - target) ** 2 for label, count in pool.items()) + lost / total
                     + sum(1 for label in main if not have.get(label)))
            if best_score is None or score < best_score:
                best, best_score = cluster, score
        if not best:
            break
        chosen |= best
    return chosen


def _consecutive(previous: dict, frame: dict, per_segment: dict[int, int]) -> bool:
    if frame['segment'] == previous['segment']:
        return frame['index'] == previous['index'] + 1
    return (frame['segment'] == previous['segment'] + 1 and frame['index'] == 0
            and previous['index'] == per_segment.get(previous['segment'], FRAMES_PER_SEGMENT) - 1)


def sequences(frames: list[dict], partitions: list[str], *, min_length: int = MIN_SEQUENCE) -> list[dict]:
    """The runs of a drive's frames in one partition (not buffer), without a missing frame or picture, at least
    min_length frames: [{'route', 'partition', 'start' (frame position in the drive), 'positions': [...]}]."""
    per_segment: dict[int, int] = {}
    for frame in frames:
        per_segment[frame['segment']] = max(per_segment.get(frame['segment'], 0), frame['index'] + 1)
    runs, current = [], []
    for position, (frame, partition) in enumerate(zip(frames, partitions)):
        joins = (current and partitions[current[-1]] == partition and frame['present']
                 and _consecutive(frames[current[-1]], frame, per_segment))
        if not joins:
            runs.append(current)
            current = []
        if frame['present'] and partition != 'buffer':
            current.append(position)
    runs.append(current)
    return [{'route': frames[run[0]]['route'], 'partition': partitions[run[0]], 'start': run[0], 'positions': run}
            for run in runs if len(run) >= min_length]


def split_sha256(split: dict) -> str:
    return hashlib.sha256(json.dumps(split, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dataset_summary(frames_by_route: dict[str, list[dict]], split: dict, runs: list[dict]) -> dict:
    """The labelled frames per partition and class in the sequences, with the frames there still to review, and the same
    over every frame with those a person could not tell."""
    summary = {partition: {'sequences': 0, 'frames': 0, 'labelled': {name: 0 for name in CLASSES}, 'to_review': 0}
               for partition in PARTITIONS}
    for run in runs:
        part = summary[run['partition']]
        part['sequences'] += 1
        part['frames'] += len(run['positions'])
        frames = frames_by_route[run['route']]
        for position in run['positions']:
            frame = frames[position]
            if frame['label'] >= 0:
                part['labelled'][CLASSES[frame['label']]] += 1
            elif not frame['reviewed']:
                part['to_review'] += 1
    every = {name: 0 for name in CLASSES}
    cant_tell = to_review = 0
    for frames in frames_by_route.values():
        for frame in frames:
            if frame['label'] >= 0:
                every[CLASSES[frame['label']]] += 1
            elif frame['reviewed']:
                cant_tell += 1
            else:
                to_review += 1
    return {'policy': split['policy'], 'label_policy': LABEL_POLICY, 'test_routes': split['test_routes'],
            'partitions': summary, 'all_frames': {'labelled': every, 'cant_tell': cant_tell, 'to_review': to_review}}


def missing_labels(summary: dict) -> list[str]:
    """What the sequences of dataset_summary lack to train (ENOUGH_LABELS, NEEDED_CLASSES), one text each; [] when they
    have enough."""
    missing = []
    for partition, enough in ENOUGH_LABELS.items():
        labelled = summary['partitions'][partition]['labelled']
        if sum(labelled.values()) < enough:
            missing.append(f'{partition} has {sum(labelled.values())} of {enough} confirmed frames')
        missing += [f'{partition} has {labelled[name]} of {ENOUGH_OF_A_CLASS} {name} frames'
                    for name in NEEDED_CLASSES if labelled[name] < ENOUGH_OF_A_CLASS]
    return missing


def load_dataset(root: Path, routes: list[str] | None = None, tests: list[str] | None = None,
                 min_length: int = MIN_SEQUENCE) -> dict:
    """{'frames': {route: frames}, 'split': place_split, 'split_sha256', 'sequences': [...], 'summary'} for every
    drive with road types (or routes), the test drives of the sign split unless tests is given."""
    root = Path(root)
    routes = routes or road_type_routes(root)
    frames_by_route = {route: route_frames(root, route) for route in routes}
    frames_by_route = {route: frames for route, frames in frames_by_route.items() if frames}
    split = place_split(frames_by_route, tests if tests is not None else test_routes(root))
    runs = [run for route, frames in frames_by_route.items()
            for run in sequences(frames, split['partitions'][route], min_length=min_length)]
    return {'frames': frames_by_route, 'split': split, 'split_sha256': split_sha256(split), 'sequences': runs,
            'summary': dataset_summary(frames_by_route, split, runs)}


def resized_frame(path: str, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    """A road frame as the network takes it: RGB uint8 (height, width, 3), PIL's bilinear resize of the whole frame
    (the arithmetic of the car's vision.resize)."""
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((width, height), Image.BILINEAR), dtype=np.uint8)


def _fingerprint(frames: list[dict]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        path = Path(frame['image'])
        stat = path.stat() if frame['present'] else None
        digest.update(f"{frame['key']}:{stat.st_size if stat else -1}:{int(stat.st_mtime) if stat else -1}\n".encode())
    return digest.hexdigest()


def cache_paths(root: Path, route: str, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> tuple[Path, Path]:
    folder = Path(root) / CACHE_RELATIVE / route
    return folder / f'frames_{width}x{height}.npy', folder / f'frames_{width}x{height}.json'


def _resize_many(args: tuple[list[str], int, int]) -> np.ndarray:
    paths, width, height = args
    return np.stack([resized_frame(path, width, height) for path in paths])


def build_frame_cache(root: Path, frames_by_route: dict[str, list[dict]], *, width: int = FRAME_WIDTH,
                      height: int = FRAME_HEIGHT, workers: int = 8, log=print) -> dict[str, Path]:
    """The resized pictures of every drive, one uint8 array (frames, height, width, 3) per drive in drive order
    (a missing picture: zeros), built again only when the drive's frames changed. {route: npy path}."""
    found = {}
    for route, frames in frames_by_route.items():
        array_path, index_path = cache_paths(root, route, width, height)
        fingerprint = _fingerprint(frames)
        if array_path.is_file() and index_path.is_file():
            index = json.loads(index_path.read_text(encoding='utf-8'))
            if index.get('fingerprint') == fingerprint and index.get('version') == CACHE_VERSION:
                found[route] = array_path
                continue
        present = [i for i, frame in enumerate(frames) if frame['present']]
        chunks = [present[i:i + 64] for i in range(0, len(present), 64)]
        array = np.zeros((len(frames), height, width, 3), dtype=np.uint8)
        jobs = [([frames[i]['image'] for i in chunk], width, height) for chunk in chunks]
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_resize_many, jobs))
        else:
            results = [_resize_many(job) for job in jobs]
        for chunk, result in zip(chunks, results):
            array[chunk] = result
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix('.tmp.npy')
        np.save(temporary, array)
        temporary.replace(array_path)
        index_path.write_text(json.dumps({'version': CACHE_VERSION, 'route': route, 'width': width, 'height': height,
                                          'keys': [frame['key'] for frame in frames], 'fingerprint': fingerprint},
                                         indent=1) + '\n', encoding='utf-8')
        log(f'road-type frames: {route} {len(present)} pictures at {width}x{height}')
        found[route] = array_path
    return found
