"""The intersections the car drives through, from the map: the labels of the junction head of the road-type network.

Why: in Poland, and in Switzerland by OSStr art. 16 cpv. 2, a speed sign holds at most to the end of the next
intersection. Poland has no end sign: the car must know by itself that it crossed one, so that the limit goes back to
the general one. The comma reads nothing new from the map: the road-type network learns to see the intersections from
the camera (its junction head), and these labels teach it.

``nnslr junctions --all`` reads the GPS track of every drive (``derived/gps/<route>/positions.jsonl``: position, speed
and heading of every frame, nnslr gps) and the OpenStreetMap roads near the drives (``derived/osm/roads.jsonl``,
scripts/osm_roads.py, with their junction and access tags), offline. The rules:

- an intersection is a node of the map where at least three arms of public roads meet: highway trunk, primary,
  secondary, tertiary, unclassified, residential, living_street and their ``_link``; motorway and motorway_link only
  where they meet another road at grade (a node where only motorways, expressways and their ramps meet is the diverge
  or the merge of an interchange, never an intersection). Service roads, tracks, paths, footways, cycleways,
  pedestrian ways, steps and bridleways are not roads here (the law leaves out driveways and the exits of internal and
  unpaved roads), nor is a road whose access or motor_vehicle tag is in NOT_PUBLIC (private, no...);
- two arms of one road (same kind, no other name or ref) within SPLIT_ANGLE of each other, one of them one-way,
  count as one: the start of a dual carriageway, a lane that splits off and joins again, the bypass of a roundabout
  are not intersections by themselves;
- a roundabout (``junction=roundabout`` or ``circular``) is one intersection, from where the car enters it to where it
  leaves it;
- the nodes of one intersection close together are one intersection: two nodes joined by CLUSTER_M or less of road
  (the two carriageways of a dual carriageway crossed, a staggered crossing) or by a turning lane (a non-motorway
  ``_link`` of LINK_M or less), at most MAX_SPAN_M across;
- a crossing at another level shares no node with our road: a bridge or a tunnel is never an intersection by itself;
- the car drives through an intersection when its track passes within PASS_M of a node of it and the road it is on
  (map_context.our_road), within NEAR_SECONDS of that moment, is one of the roads of that intersection.

The area of an intersection along the track: from RADIUS before its first node to RADIUS after its last one, RADIUS
being half the width of its widest road (lanes x LANE_M, or USUAL_LANES of its kind) plus SETBACK_M (the stop line,
the corner), kept between MIN_RADIUS_M and MAX_RADIUS_M: 10 m for a crossing of two-lane roads, 10.5 m with a road
of four lanes, 13.75 m with six. The distance along the track is the integral of the GPS speed (a car waiting at a
red light does not move, its GPS position does), the place of a node on it the closest point of the track.

The labels of a frame: in_junction (its time is inside the area), passed (the first frame after the car left the
area: where the limit of a sign ends), junction (the intersection's id) and known (false without a GPS fix: no
label there). Writes ``derived/junctions/<route>.json``: {'route_id', 'version', 'rules', 'passages': [{'id', 'kind'
(crossing, roundabout), 'arms', 'highways', 'radius_m', 't_in', 't_out' (seconds of the drive), 'enter_key',
'passed_key', 'complete', ...}], 'rejected' (passed near but on another road: a bridge), 'frames': [{'frame_key',
'segment_index', 'output_index', 'known', 'in_junction', 'passed', 'junction', 'distance_m'}]}.

The map labels are a proposal: OpenStreetMap can be wrong or old, and a node of the map is not the stop line.
OpenStreetMap data (c) OpenStreetMap contributors (ODbL).
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime
from pathlib import Path

from nnslr_tools.annotations import atomic_write, jsonl_bytes, strict_json_loads
from nnslr_tools.manifest import parse_route_id
from nnslr_tools.map_context import CELL_DEGREES, OUR_ROAD_M, RoadMap, _angle, _bearing, _xy, our_road
from nnslr_tools.sign_reads import is_read_frame

JUNCTIONS_RELATIVE = Path('derived') / 'junctions'
VERSION = 2
PUBLIC = frozenset({'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential',
                    'living_street', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
                    'tertiary_link'})
GRADE_SEPARATED = frozenset({'motorway', 'motorway_link', 'trunk', 'trunk_link'})
NOT_PUBLIC = frozenset({'private', 'no', 'agricultural', 'forestry', 'customers', 'delivery', 'permit', 'emergency'})
ROUNDABOUT = frozenset({'roundabout', 'circular'})
LANE_M = 3.25
# The lanes of a road of each kind without a lanes tag (both directions; a one-way road or a ramp: 1).
USUAL_LANES = {'motorway': 4, 'trunk': 2, 'primary': 2, 'secondary': 2, 'tertiary': 2, 'unclassified': 2,
               'residential': 2, 'living_street': 2}
SETBACK_M = 4.0
MIN_RADIUS_M = 10.0
MAX_RADIUS_M = 20.0
PASS_M = 12.0
CLUSTER_M = 25.0
LINK_M = 80.0
MAX_SPAN_M = 60.0
# Two one-way arms of one road this close in direction (degrees), one in and one out: the carriageways splitting.
SPLIT_ANGLE = 50.0
NEAR_SECONDS = 3
# The same intersection met again along the track this much later (metres): a second passage (a loop, a way back).
AGAIN_M = 150.0
# A frame this long after the end of an area (seconds) is still "the first frame after it"; later, a gap of the GPS.
PASSED_WITHIN_S = 1.5


def rules() -> dict:
    """The thresholds of the labels, written with them."""
    return {'public': sorted(PUBLIC), 'grade_separated': sorted(GRADE_SEPARATED), 'not_public': sorted(NOT_PUBLIC),
            'lane_m': LANE_M, 'setback_m': SETBACK_M, 'min_radius_m': MIN_RADIUS_M, 'max_radius_m': MAX_RADIUS_M,
            'pass_m': PASS_M, 'cluster_m': CLUSTER_M, 'link_m': LINK_M, 'max_span_m': MAX_SPAN_M,
            'split_angle': SPLIT_ANGLE, 'near_seconds': NEAR_SECONDS}


def is_public(way: dict) -> bool:
    return (way.get('highway') in PUBLIC and way.get('access') not in NOT_PUBLIC
            and way.get('motor_vehicle') not in NOT_PUBLIC)


def road_width(way: dict) -> float:
    """Metres: the lanes tag (or USUAL_LANES of its kind, 1 for a ramp or a one-way road) x LANE_M."""
    lanes = None
    text = way.get('lanes')
    if isinstance(text, str):
        try:
            lanes = float(text.split(';')[0])
        except ValueError:
            lanes = None
    if not lanes or lanes <= 0:
        highway = way.get('highway') or ''
        one_way = way.get('oneway') in ('yes', '1', '-1') or way.get('junction') in ROUNDABOUT
        lanes = 1 if highway.endswith('_link') or one_way else USUAL_LANES.get(highway, 2)
    return lanes * LANE_M


def radius(widest: float) -> float:
    return min(MAX_RADIUS_M, max(MIN_RADIUS_M, widest / 2 + SETBACK_M))


def same_road(a: dict, b: dict) -> bool:
    """Two ways of one road: the same kind, both of a roundabout or neither, and neither their names nor their refs
    differ (a missing one agrees)."""
    if a.get('highway') != b.get('highway') or (a.get('junction') in ROUNDABOUT) != (b.get('junction') in ROUNDABOUT):
        return False
    return all(a.get(tag) is None or b.get(tag) is None or a.get(tag) == b.get(tag) for tag in ('name', 'ref'))


def _distance(a, b) -> float:
    x, y = _xy(b[0], b[1], a[0], a[1])
    return math.hypot(x, y)


def _direction(way: dict) -> int:
    """1: one-way along the nodes, -1: against them, 0: both ways."""
    oneway = way.get('oneway')
    if oneway == '-1':
        return -1
    if oneway in ('yes', '1', 'true') or way.get('junction') in ROUNDABOUT:
        return 1
    return 0


class Junctions:
    """The intersections of a road map (RoadMap): the nodes where three arms of public roads meet, grouped."""

    def __init__(self, road_map: RoadMap):
        self.road_map = road_map
        ways = road_map.ways
        self.public = [is_public(way) for way in ways]
        # Every node of every way: [(way, position)].
        self.node_ways: dict[tuple[float, float], list[tuple[int, int]]] = {}
        for w, way in enumerate(ways):
            nodes = way['nodes']
            closed = len(nodes) > 2 and nodes[0] == nodes[-1]
            for k, node in enumerate(nodes[:-1] if closed else nodes):
                self.node_ways.setdefault((node[0], node[1]), []).append((w, k))
        junction_nodes = {node for node, members in self.node_ways.items() if self._is_junction(node, members)}
        self.junction_nodes = junction_nodes
        self.clusters = self._clusters(junction_nodes)
        self.cells: dict[tuple[int, int], list[tuple[int, tuple[float, float]]]] = {}
        for c, cluster in enumerate(self.clusters):
            for node in cluster['area']:
                self.cells.setdefault((int(node[0] // CELL_DEGREES), int(node[1] // CELL_DEGREES)), []).append(
                    (c, node))

    def _arms(self, node, members) -> list[tuple[int, float, int]]:
        """The arms of public roads at a node: (way, bearing from the node, flow: 1 out of the node, -1 into it,
        0 both)."""
        arms = []
        for w, k in members:
            if not self.public[w]:
                continue
            way = self.road_map.ways[w]
            nodes = way['nodes']
            closed = len(nodes) > 2 and nodes[0] == nodes[-1]
            direction = _direction(way)
            for step in (1, -1):
                j = k + step
                if closed:
                    j %= len(nodes) - 1
                elif not 0 <= j < len(nodes):
                    continue
                other = nodes[j]
                x, y = _xy(other[0], other[1], node[0], node[1])
                arms.append((w, _bearing(x, y), direction * step))
        return arms

    def _is_junction(self, node, members) -> bool:
        if sum(1 for w, _ in members if self.public[w]) < 2:
            return False
        arms = self._arms(node, members)
        if len(arms) < 3:
            return False
        highways = {self.road_map.ways[w]['highway'] for w, _, _ in arms}
        if highways <= GRADE_SEPARATED:
            return False
        # One road splitting or merging: two arms of it close together in direction, one of them one-way (its two
        # carriageways, a lane of it that splits off or joins it again, a bypass of a roundabout), count as one.
        merged = set()
        for i, (wa, ba, fa) in enumerate(arms):
            for j in range(i + 1, len(arms)):
                wb, bb, fb = arms[j]
                if i in merged or j in merged or wa == wb or (fa == 0 and fb == 0):
                    continue
                if same_road(self.road_map.ways[wa], self.road_map.ways[wb]) and _angle(ba, bb) <= SPLIT_ANGLE:
                    merged.add(j)
        return len(arms) - len(merged) >= 3

    def _clusters(self, junction_nodes: set) -> list[dict]:
        ways = self.road_map.ways
        parent = {node: node for node in junction_nodes}
        members = {node: [node] for node in junction_nodes}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a, b, capped: bool) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if capped and any(_distance(p, q) > MAX_SPAN_M for p in members[ra] for q in members[rb]):
                return
            parent[rb] = ra
            members[ra] += members.pop(rb)

        # Roundabouts: the ring ways joined by their shared nodes; every node of a ring is one intersection's area.
        ring_of: dict[int, int] = {}
        rings: list[set[int]] = []
        for w, way in enumerate(ways):
            if way.get('junction') not in ROUNDABOUT or not self.public[w]:
                continue
            touching = {ring_of[o] for node in way['nodes'] for o, _ in self.node_ways.get((node[0], node[1]), ())
                        if o in ring_of}
            if touching:
                keep = min(touching)
                for other in sorted(touching - {keep}):
                    for o in rings[other]:
                        ring_of[o] = keep
                    rings[keep] |= rings[other]
                    rings[other] = set()
                rings[keep].add(w)
                ring_of[w] = keep
            else:
                ring_of[w] = len(rings)
                rings.append({w})
        ring_area: dict[tuple, set] = {}
        for ring in rings:
            if not ring:
                continue
            area = {(n[0], n[1]) for w in ring for n in ways[w]['nodes']}
            inside = sorted(area & junction_nodes)
            for node in inside[1:]:
                union(inside[0], node, capped=False)
            if inside:
                ring_area[inside[0]] = area
        # Nodes joined by a short stretch of road, or by a turning lane: shortest first.
        pairs = []
        for w, way in enumerate(ways):
            if not self.public[w]:
                continue
            nodes = [(n[0], n[1]) for n in way['nodes']]
            length, last, last_at = 0.0, None, 0.0
            for k, node in enumerate(nodes):
                if k:
                    length += _distance(nodes[k - 1], node)
                if node in junction_nodes:
                    if last is not None and last != node and length - last_at <= CLUSTER_M:
                        pairs.append((length - last_at, last, node))
                    last, last_at = node, length
            highway = way['highway']
            if highway.endswith('_link') and highway not in GRADE_SEPARATED and length <= LINK_M:
                ends = [node for node in nodes if node in junction_nodes]
                if len(ends) >= 2 and ends[0] != ends[-1]:
                    pairs.append((length, ends[0], ends[-1]))
        for _, a, b in sorted(pairs):
            union(a, b, capped=True)
        clusters = []
        for root, nodes in members.items():
            area = set(nodes)
            roundabout = False
            for node in nodes:
                if node in ring_area:
                    area |= ring_area[node]
                    roundabout = True
            touching = sorted({w for node in area for w, _ in self.node_ways.get(node, ())})
            public = [w for w in touching if self.public[w]]
            widest = max((road_width(ways[w]) for w in public), default=2 * LANE_M)
            arms = max(len(self._arms(node, self.node_ways[node])) for node in nodes)
            ident = hashlib.sha1(json.dumps(sorted(nodes)).encode()).hexdigest()[:10]
            lat = sum(n[0] for n in nodes) / len(nodes)
            lon = sum(n[1] for n in nodes) / len(nodes)
            clusters.append({'id': 'J' + ident, 'kind': 'roundabout' if roundabout else 'crossing',
                             'nodes': sorted(nodes), 'area': sorted(area), 'ways': set(touching),
                             'highways': sorted({ways[w]['highway'] for w in public}), 'arms': arms,
                             'radius_m': round(radius(widest), 2), 'widest_m': round(widest, 2),
                             'latitude': round(lat, 7), 'longitude': round(lon, 7)})
        clusters.sort(key=lambda c: c['id'])
        return clusters

    def near(self, lat: float, lon: float):
        """(cluster index, node) of the area nodes in the 3 x 3 grid cells around a position."""
        a, b = int(lat // CELL_DEGREES), int(lon // CELL_DEGREES)
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                yield from self.cells.get((a + da, b + db), ())


def _frame_seconds(line: dict) -> float:
    return float(line['segment_index']) * 60.0 + float(line['output_index'])


def _pieces(lines: list[dict]) -> list[list[int]]:
    """The runs of frames with a GPS fix, without a gap of more than 2 seconds."""
    pieces, current = [], []
    for i, line in enumerate(lines):
        fix = line.get('latitude') is not None and line.get('longitude') is not None
        if current and (not fix or _frame_seconds(line) - _frame_seconds(lines[current[-1]]) > 2.0):
            pieces.append(current)
            current = []
        if fix:
            current.append(i)
    pieces.append(current)
    return [piece for piece in pieces if len(piece) >= 2]


def _speed(line: dict) -> float | None:
    value = line.get('speed_mps')
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _along(lines: list[dict], piece: list[int]) -> list[float]:
    """The distance along the track at each frame of a piece (metres), the integral of the GPS speed (the distance
    between the fixes where a speed is missing)."""
    s = [0.0]
    for a, b in zip(piece, piece[1:]):
        va, vb = _speed(lines[a]), _speed(lines[b])
        dt = _frame_seconds(lines[b]) - _frame_seconds(lines[a])
        if va is not None and vb is not None:
            step = (va + vb) / 2 * dt
        else:
            step = _distance((lines[a]['latitude'], lines[a]['longitude']), (lines[b]['latitude'], lines[b]['longitude']))
        s.append(s[-1] + step)
    return s


def _time_at(seconds: list[float], along: list[float], value: float) -> float:
    """The first moment the distance along the track reaches value (between two frames: interpolated)."""
    if value <= along[0]:
        return seconds[0]
    for i in range(1, len(along)):
        if along[i] >= value:
            span = along[i] - along[i - 1]
            u = 0.0 if span <= 0 else (value - along[i - 1]) / span
            return seconds[i - 1] + u * (seconds[i] - seconds[i - 1])
    return seconds[-1]


def route_junctions(junctions: Junctions, lines: list[dict]) -> dict:
    """The passages through intersections of one drive and the labels of its frames (see the module's text); lines:
    the drive's positions.jsonl lines."""
    lines = sorted(lines, key=lambda line: (line['segment_index'], line['output_index']))
    road_map = junctions.road_map
    matched: dict[int, int | None] = {}

    def road_at(i: int) -> int | None:
        if i not in matched:
            line = lines[i]
            heading = line.get('bearing_deg')
            if line.get('latitude') is None or heading is None:
                matched[i] = None
            else:
                near = road_map.nearest(line['latitude'], line['longitude'], OUR_ROAD_M)
                matched[i] = our_road(near, heading)
        return matched[i]

    passages, rejected = [], []
    for piece in _pieces(lines):
        along = _along(lines, piece)
        seconds = [_frame_seconds(lines[i]) for i in piece]
        touches: dict[int, list[tuple[float, float, int]]] = {}
        for p in range(len(piece) - 1):
            a, b = lines[piece[p]], lines[piece[p + 1]]
            bx, by = _xy(b['latitude'], b['longitude'], a['latitude'], a['longitude'])
            length2 = bx * bx + by * by
            seen = set()
            for lat, lon in ((a['latitude'], a['longitude']), (b['latitude'], b['longitude'])):
                for c, node in junctions.near(lat, lon):
                    if (c, node) in seen:
                        continue
                    seen.add((c, node))
                    nx, ny = _xy(node[0], node[1], a['latitude'], a['longitude'])
                    u = 0.0 if length2 == 0 else max(0.0, min(1.0, (nx * bx + ny * by) / length2))
                    d = math.hypot(nx - u * bx, ny - u * by)
                    if d <= PASS_M:
                        touches.setdefault(c, []).append((along[p] + u * (along[p + 1] - along[p]), d, p, node))
        for c, found in touches.items():
            found.sort()
            groups, current = [], [found[0]]
            for item in found[1:]:
                if item[0] - current[-1][0] > AGAIN_M:
                    groups.append(current)
                    current = []
                current.append(item)
            groups.append(current)
            cluster = junctions.clusters[c]
            for group in groups:
                first = min(item[2] for item in group)
                last = max(item[2] for item in group) + 1
                window = range(max(0, first - NEAR_SECONDS), min(len(piece), last + NEAR_SECONDS + 1))
                roads = {road_at(piece[p]) for p in window} - {None}
                closest = min(item[1] for item in group)
                record = {'id': cluster['id'], 'kind': cluster['kind'], 'arms': cluster['arms'],
                          'highways': cluster['highways'], 'radius_m': cluster['radius_m'],
                          'latitude': cluster['latitude'], 'longitude': cluster['longitude'],
                          'closest_m': round(closest, 1)}
                if not roads & cluster['ways']:
                    # The frame where the track passes closest to it.
                    nearest = min(group, key=lambda item: item[1])[0]
                    at = min(range(first, last + 1), key=lambda k: abs(along[k] - nearest))
                    rejected.append({**record, 'key': lines[piece[at]]['frame_key'],
                                     'why': 'near, but the car is on another road' if roads else 'no road matched'})
                    continue
                # The place of each node along the track: the closest point of the track to it (the segments before
                # and after that point reach the node too, from their ends, farther away: not its place).
                place: dict[tuple, tuple[float, float]] = {}
                for s, d, _, node in group:
                    if node not in place or d < place[node][0]:
                        place[node] = (d, s)
                s_first = min(s for _, s in place.values()) - cluster['radius_m']
                s_last = max(s for _, s in place.values()) + cluster['radius_m']
                complete = s_last <= along[-1] and s_first >= along[0]
                t_in = _time_at(seconds, along, s_first)
                t_out = _time_at(seconds, along, s_last)
                passages.append({**record, 't_in': round(t_in, 2), 't_out': round(t_out, 2),
                                 's_in': round(s_first, 1), 's_out': round(s_last, 1), 'complete': complete,
                                 'piece_end': seconds[-1]})
    passages.sort(key=lambda item: item['t_in'])
    distance = 0.0
    frames = []
    previous = None
    for line in lines:
        t = _frame_seconds(line)
        speed = _speed(line)
        if previous is not None and speed is not None and previous[1] is not None:
            distance += (speed + previous[1]) / 2 * (t - previous[0])
        previous = (t, speed)
        frames.append({'frame_key': line['frame_key'], 'segment_index': line['segment_index'],
                       'output_index': line['output_index'],
                       'known': line.get('latitude') is not None, 'in_junction': False, 'passed': False,
                       'junction': None, 'distance_m': round(distance, 1)})
    times = [_frame_seconds(line) for line in lines]
    for passage in passages:
        inside = [i for i, t in enumerate(times) if passage['t_in'] <= t <= passage['t_out']]
        for i in inside:
            frames[i]['in_junction'] = True
            frames[i]['junction'] = passage['id']
        passage['enter_key'] = frames[inside[0]]['frame_key'] if inside else None
        passage['frames_inside'] = len(inside)
        after = next((i for i, t in enumerate(times) if t > passage['t_out']), None)
        passage['passed_key'] = None
        if passage['complete'] and after is not None and times[after] - passage['t_out'] <= PASSED_WITHIN_S \
                and frames[after]['known']:
            frames[after]['passed'] = True
            frames[after]['junction'] = frames[after]['junction'] or passage['id']
            passage['passed_key'] = frames[after]['frame_key']
            passage['passed_at'] = times[after]
        del passage['piece_end']
    return {'passages': passages, 'rejected': rejected, 'frames': frames}


def junctions_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / JUNCTIONS_RELATIVE / f'{route}.json'


def build_junctions(root: Path, route: str, positions: list[dict], junctions: Junctions) -> dict:
    """Write derived/junctions/<route>.json; returns its summary."""
    # At 1 a second: the frames of the comma's reads (sign_reads) are only for the sign review.
    found = route_junctions(junctions, [line for line in positions if not is_read_frame(line)])
    record = {'route_id': route, 'version': VERSION, 'rules': rules(), **found}
    atomic_write(junctions_path(root, route), (json.dumps(record, separators=(',', ':')) + '\n').encode())
    return {**summary(route, record), 'output': str(junctions_path(root, route))}


def summary(route: str, record: dict) -> dict:
    passages = record['passages']
    frames = record['frames']
    known = [f for f in frames if f['known']]
    km = (frames[-1]['distance_m'] / 1000.0) if frames else 0.0
    return {'route_id': route, 'junctions': len(passages),
            'roundabouts': sum(1 for p in passages if p['kind'] == 'roundabout'),
            'passed': sum(1 for p in passages if p['passed_key']),
            'incomplete': sum(1 for p in passages if not p['passed_key']),
            'rejected': len(record['rejected']), 'frames': len(frames), 'known': len(known),
            'in_junction_frames': sum(1 for f in frames if f['in_junction']),
            'without_frame_inside': sum(1 for p in passages if p['frames_inside'] == 0),
            'km': round(km, 1), 'per_km': round(len(passages) / km, 2) if km else None}


def load_junctions(root: Path, route: str) -> dict | None:
    """The record of nnslr junctions for a drive, None when it has not run (or ran with an older version)."""
    path = junctions_path(root, route)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    return record if record.get('version') == VERSION else None


def load_map(root: Path) -> Junctions | None:
    """The intersections of the map near the drives; None without the map. Raises ValueError for a map read before
    the junction tags were kept (scripts/osm_roads.py again: helpers/road_types.sh --again)."""
    road_map = RoadMap.load(root)
    if road_map is None:
        return None
    if road_map.ways and not any('junction' in way for way in road_map.ways):
        raise ValueError('the map of the roads was read before the junction and access tags were kept: read it again '
                         '(helpers/road_types.sh --again)')
    return Junctions(road_map)


# -- a person's corrections (Data › Roads, a drive opened, until 7 Nov 2026) ----------------------------------------------
# The map misses junctions (a new road, a crossing drawn a few metres off) and has some a driver would not call one (a
# car park's exit tagged as a road, a bridge right over a crossing). On the review page of a drive a person marked the
# picture shown "Junction here" or "Not a junction"; each tap is a line appended to annotations/junctions/<route>.jsonl
# ({'frame_key', 'junction': 'here' | 'none', 'reviewer', 'at'}), and nnslr junctions never
# touches the file. The labels of a training are the map's with these over them (labelled_junctions), the last line
# about a junction winning:
# - "none" inside a junction of the map (or on the frame just after it) drops it: its frames are not a junction, its
#   "passed" frame is not one either;
# - "here" inside a junction of the map confirms it (no longer to check); elsewhere it adds one, the frames within
#   MIN_RADIUS_M of the picture along the track (at most PERSON_SECONDS on each side: a car waiting there), "passed"
#   the frame after them;
# - "none" inside a junction a person added takes it away again.
CORRECTIONS_RELATIVE = Path('annotations') / 'junctions'
CHOICES = ('here', 'none')
# To check (Next to check of the page): a junction of the map the track passes farther than FAR_M from (a GPS offset, a
# road beside it), one the car stays in longer than LONG_S (waiting at a light, or stopped beside it), one whose exit
# was not seen (the GPS stops in it), and a junction of the map next to the track on another road (rejected: a bridge,
# an underpass, a road alongside: not a label, but the map may have matched the wrong road).
FAR_M = 8.0
LONG_S = 30
NEAR_FRAMES = 2
PERSON_SECONDS = 10
_CORRECTING = threading.Lock()


class CorrectionError(ValueError):
    """A junction correction that cannot be kept: a frame that is not of the drive, a choice that is not one."""


def corrections_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / CORRECTIONS_RELATIVE / f'{route}.jsonl'


def load_corrections(root: Path, route: str) -> list[dict]:
    """A person's junction corrections of a route in the order they were made, [] before the first; a line that cannot
    be read is skipped (the others still count)."""
    path = corrections_path(root, route)
    if not path.is_file():
        return []
    found = []
    for text in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            line = strict_json_loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(line, dict) and isinstance(line.get('frame_key'), str) and line.get('junction') in CHOICES:
            found.append(line)
    return found


def correct(root: Path, route: str, frame_key, junction, reviewer: str) -> dict:
    """Keeps a person's "Junction here" (junction 'here') or "Not a junction" ('none') for a frame of the drive,
    appended to its corrections. Returns the line kept."""
    record = load_junctions(root, route)
    if record is None:
        raise CorrectionError('The junctions of this drive have not been found yet: Find the road types reads them.')
    if not isinstance(frame_key, str) or frame_key not in {frame['frame_key'] for frame in record['frames']}:
        raise CorrectionError('frame_key is a frame of this drive.')
    if junction not in CHOICES:
        raise CorrectionError('junction is "here" or "none".')
    line = {'frame_key': frame_key, 'junction': junction, 'reviewer': reviewer,
            'at': datetime.now().astimezone().isoformat(timespec='seconds')}
    path = corrections_path(root, route)
    with _CORRECTING:
        before = path.read_bytes() if path.is_file() else b''
        if before and not before.endswith(b'\n'):
            before += b'\n'
        atomic_write(path, before + jsonl_bytes([line]))
    return line


def _doubt(passage: dict) -> str | None:
    """Why a junction of the map is worth a look, None when nothing is odd about it."""
    if not passage.get('passed_key'):
        return 'The GPS stops inside this junction: where the car left it is not known.'
    if passage.get('closest_m', 0) > FAR_M:
        return f'The track passes {passage["closest_m"]:.0f} m from the junction: a GPS offset, or a road beside it?'
    if passage.get('frames_inside', 0) > LONG_S:
        return (f'The car stays {passage["frames_inside"]} s inside: waiting at a light, or stopped beside the '
                'junction?')
    return None


def _person_span(frames: list[dict], times: list[float], index: int) -> tuple[int, int, int | None]:
    """The frames of a junction a person marked at a picture: those within MIN_RADIUS_M of it along the track (at most
    PERSON_SECONDS away, none across a gap), and its "passed" frame (the next one, None at the end of the drive)."""
    first = last = index
    if frames[index]['known']:
        here = frames[index]['distance_m']
        while first > 0 and frames[first - 1]['known'] and times[first] - times[first - 1] <= 2.0 \
                and times[index] - times[first - 1] <= PERSON_SECONDS \
                and abs(here - frames[first - 1]['distance_m']) <= MIN_RADIUS_M:
            first -= 1
        while last < len(frames) - 1 and frames[last + 1]['known'] and times[last + 1] - times[last] <= 2.0 \
                and times[last + 1] - times[index] <= PERSON_SECONDS \
                and abs(frames[last + 1]['distance_m'] - here) <= MIN_RADIUS_M:
            last += 1
    after = last + 1 if last + 1 < len(frames) and times[last + 1] - times[last] <= 2.0 else None
    return first, last, after


def corrected(record: dict, corrections: list[dict]) -> dict:
    """The map's junctions of a drive (load_junctions) with a person's corrections over them: {'passages', 'frames',
    'rejected'} as the map's record (a dropped junction is out of passages, one a person added is in, by 'person'), and
    'marks': every junction of the map, every one a person added and every junction of the map rejected next to the
    track, as {'id', 'kind', 'first', 'last', 'passed' (frame indices; passed None when not labelled), 'label'
    ('junction': a label; 'dropped': a person said it is not one; 'maybe': the map's, on another road: not a label),
    'by' ('map', 'person'), 'reviewed' (a person settled it), 'reviewer', 'doubt' (why to check it, None once reviewed or
    when nothing is odd), 'passage' (the map's record, None for a person's)}."""
    frames = [dict(frame) for frame in record['frames']]
    index = {frame['frame_key']: i for i, frame in enumerate(frames)}
    times = [_frame_seconds(frame) for frame in frames]
    marks = []
    for passage in record['passages']:
        inside = [i for i, t in enumerate(times) if passage['t_in'] <= t <= passage['t_out']]
        if not inside:
            middle = (passage['t_in'] + passage['t_out']) / 2
            inside = [min(range(len(times)), key=lambda i: abs(times[i] - middle))] if times else []
        if not inside:
            continue
        marks.append({'id': passage['id'], 'kind': passage['kind'], 'first': inside[0], 'last': inside[-1],
                      'passed': index.get(passage.get('passed_key')), 'label': 'junction', 'by': 'map',
                      'reviewed': False, 'reviewer': None, 'doubt': _doubt(passage), 'passage': passage})
    seen = set()
    for rejected in record.get('rejected', []):
        at = index.get(rejected.get('key'))
        if at is None or (rejected['id'], at) in seen:
            continue
        seen.add((rejected['id'], at))
        marks.append({'id': rejected['id'], 'kind': rejected['kind'], 'first': at, 'last': at, 'passed': None,
                      'label': 'maybe', 'by': 'map', 'reviewed': False, 'reviewer': None,
                      'doubt': f'The map has a junction {rejected["closest_m"]:.0f} m from the track, on another road: '
                               'a bridge, an underpass, or a road beside this one?', 'passage': rejected})
    for correction in corrections:
        at = index.get(correction['frame_key'])
        if at is None:
            continue

        def covers(mark: dict) -> bool:
            return mark['first'] <= at <= mark['last'] or mark['passed'] == at

        target = next((mark for mark in marks if mark['label'] != 'maybe' and covers(mark)), None) \
            or next((mark for mark in marks if mark['label'] == 'maybe' and abs(mark['first'] - at) <= NEAR_FRAMES),
                    None)
        who = correction.get('reviewer')
        if correction['junction'] == 'none':
            if target is None:
                continue
            if target['by'] == 'person':
                marks.remove(target)
            else:
                target.update(label='maybe' if target['label'] == 'maybe' else 'dropped', reviewed=True,
                              reviewer=who, doubt=None)
        elif target is not None and target['label'] != 'maybe':
            target.update(label='junction', reviewed=True, reviewer=who, doubt=None)
        else:
            first, last, after = _person_span(frames, times, at)
            if target is not None:
                marks.remove(target)
            marks.append({'id': 'P' + correction['frame_key'].replace('/', '-'), 'kind': 'crossing', 'first': first,
                          'last': last, 'passed': after, 'label': 'junction', 'by': 'person', 'reviewed': True,
                          'reviewer': who, 'doubt': None, 'passage': None})
    # In the order of the drive, the map's junctions as nnslr junctions labelled them (two areas can overlap).
    marks.sort(key=lambda mark: (mark['passage']['t_in'] if mark['passage'] and 't_in' in mark['passage']
                                 else times[mark['first']], mark['last']))
    for frame in frames:
        frame.update(in_junction=False, passed=False, junction=None)
    passages = []
    for mark in marks:
        if mark['label'] != 'junction':
            continue
        if mark['by'] == 'person':
            for i in range(mark['first'], (mark['passed'] if mark['passed'] is not None else mark['last']) + 1):
                frames[i]['known'] = True
            passages.append({'id': mark['id'], 'kind': mark['kind'], 'by': 'person', 'reviewer': mark['reviewer'],
                             't_in': times[mark['first']], 't_out': times[mark['last']],
                             'enter_key': frames[mark['first']]['frame_key'],
                             'passed_key': frames[mark['passed']]['frame_key'] if mark['passed'] is not None else None,
                             'complete': mark['passed'] is not None, 'frames_inside': mark['last'] - mark['first'] + 1})
        else:
            passages.append({**mark['passage'], 'reviewed': mark['reviewed']})
        for i in range(mark['first'], mark['last'] + 1):
            frames[i]['in_junction'] = True
            frames[i]['junction'] = mark['id']
        if mark['passed'] is not None:
            frames[mark['passed']]['passed'] = True
            frames[mark['passed']]['junction'] = frames[mark['passed']]['junction'] or mark['id']
    passages.sort(key=lambda passage: passage['t_in'])
    return {**record, 'passages': passages, 'frames': frames, 'marks': marks, 'corrections': len(corrections)}


def labelled_junctions(root: Path, route: str) -> dict | None:
    """The junction labels of a drive for a training and the review page: the map's (load_junctions) with a person's
    corrections over them (corrected), None before nnslr junctions ran for it."""
    record = load_junctions(root, route)
    if record is None:
        return None
    return corrected(record, load_corrections(root, route))
