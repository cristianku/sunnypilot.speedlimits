#!/usr/bin/env python3
"""The OpenStreetMap roads and car parks near the drives, for the "another road" candidates (nnslr_tools.map_context)
and the road types (nnslr_tools.road_types).

  uv run --no-project --with osmium python scripts/osm_roads.py [--data-root DIR] [--pbf FILE ...] [--margin-m 300]

Reads every ``derived/gps/<route>/positions.jsonl`` of the data root and keeps the OSM ways with a road
``highway`` tag that have a node within about --margin-m of a position. Writes ``derived/osm/roads.jsonl``:
one line per way, ``{"id", "highway", "service" (parking_aisle, driveway... of a service road), "oneway", "name",
"ref", "lanes", "maxspeed", "maxspeed_type", "motorroad", "junction" (roundabout, circular...), "access",
"motor_vehicle", "tunnel" (yes, building_passage, avalanche_protector... where the road runs under
something: nnslr_tools.tunnels), "nodes": [[lat, lon], ...]}``. Used by nnslr map-candidates, nnslr road-types and nnslr junctions (a
node shared by two ways is the same [lat, lon] in both: an intersection of the map).

The car parks near the drives, by the same rule, go to ``derived/osm/parking.jsonl``: one line per ``amenity=parking``
outline (a closed way or a multipolygon relation, assembled by osmium), ``{"id", "type" (way or relation), "name",
"parking" (surface, underground, multi-storey... when the map says it), "rings": [[[lat, lon], ...], ...]}``, outer
and inner rings together: a position is inside when it is inside an odd number of them. The parking along a street
(STREET_PARKING) is left out. Used by nnslr road-types (the type "parking").

The map: every ``raw/osm/*.osm.pbf`` (Geofabrik extracts of OpenStreetMap, © OpenStreetMap contributors, ODbL):
Switzerland, and Baden-Württemberg for the drives across the Rhine. A road or a car park crossing the border is in
both extracts: it is kept once, by its id. It is read here, offline: no position of the drives leaves the server.
osmium (pyosmium) comes from uv's cache for this run only; the environments of the site and of the training stay as
they are. Both extracts (1.2 GB) take about 2 minutes: osmium reads each twice (the multipolygon relations first).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

# The roads a car drives on (and their ramps); paths, tracks and cycleways are left out.
ROAD_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street',
    'service', 'road', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
}
CELL_DEGREES = 0.003  # about 330 m of latitude
# Parking along a street (a parking lane drawn over the carriageway, bays beside it): part of the street, not a car
# park; the car passing it at a red light is not parking.
STREET_PARKING = {'street_side', 'lane', 'on_kerb', 'half_on_kerb', 'shoulder'}


def cells_near(root: Path, margin_m: float) -> set[tuple[int, int]]:
    """The grid cells within about margin_m of every GPS position of the data root."""
    reach = max(1, math.ceil(margin_m / 330))
    cells = set()
    for path in sorted((root / 'derived' / 'gps').glob('*/positions.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('latitude') is None:
                continue
            a = int(item['latitude'] // CELL_DEGREES)
            b = int(item['longitude'] // CELL_DEGREES)
            for da in range(-reach, reach + 1):
                for db in range(-reach, reach + 1):
                    cells.add((a + da, b + db))
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default=os.environ.get('NNSLR_DATA_ROOT') or str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--pbf', nargs='+', help='default: every <data root>/raw/osm/*.osm.pbf')
    parser.add_argument('--margin-m', type=float, default=300.0)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    pbfs = [Path(name) for name in args.pbf] if args.pbf else sorted((root / 'raw' / 'osm').glob('*.osm.pbf'))
    missing = [str(pbf) for pbf in pbfs if not pbf.is_file()]
    if not pbfs or missing:
        print(f'osm_roads: missing {", ".join(missing) or root / "raw" / "osm" / "*.osm.pbf"}', file=sys.stderr)
        return 2
    import osmium

    cells = cells_near(root, args.margin_m)
    if not cells:
        print('osm_roads: no GPS positions yet (nnslr gps --all)', file=sys.stderr)
        return 2

    def points(nodes) -> tuple[list[list[float]], bool]:
        """The [lat, lon] of the nodes with a location, and whether one is near a drive."""
        found = []
        near = False
        for node in nodes:
            if not node.location.valid():
                continue
            lat, lon = node.location.lat, node.location.lon
            found.append([round(lat, 7), round(lon, 7)])
            if not near and (int(lat // CELL_DEGREES), int(lon // CELL_DEGREES)) in cells:
                near = True
        return found, near

    class Roads(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []
            self.seen = set()
            self.parks = []
            self.parks_seen = set()

        def way(self, way):
            kind = way.tags.get('highway')
            if kind not in ROAD_TYPES or way.id in self.seen:
                return
            nodes, near = points(way.nodes)
            if near and len(nodes) >= 2:
                self.seen.add(way.id)
                tags = way.tags
                # maxspeed_type: the kind of limit when the map says it (CH:urban, CH:rural, CH:zone30...), from
                # maxspeed:type, else source:maxspeed (nnslr_tools.road_types).
                self.ways.append({'id': way.id, 'highway': kind, 'service': tags.get('service'),
                                  'oneway': tags.get('oneway'), 'name': tags.get('name'), 'ref': tags.get('ref'),
                                  'lanes': tags.get('lanes'), 'maxspeed': tags.get('maxspeed'),
                                  'maxspeed_type': tags.get('maxspeed:type') or tags.get('source:maxspeed'),
                                  'motorroad': tags.get('motorroad'), 'junction': tags.get('junction'),
                                  'access': tags.get('access'), 'motor_vehicle': tags.get('motor_vehicle'),
                                  'tunnel': tags.get('tunnel'),
                                  'nodes': nodes})

        def area(self, area):
            # A car park drawn as a closed way or a multipolygon relation; area.id tells the two apart.
            if area.tags.get('amenity') != 'parking' or area.tags.get('parking') in STREET_PARKING \
                    or area.id in self.parks_seen:
                return
            rings = []
            near = False
            for outer in area.outer_rings():
                for ring in (outer, *area.inner_rings(outer)):
                    nodes, close = points(ring)
                    near = near or close
                    if len(nodes) >= 4:
                        rings.append(nodes)
            if near and rings:
                self.parks_seen.add(area.id)
                self.parks.append({'id': area.orig_id(), 'type': 'way' if area.from_way() else 'relation',
                                   'name': area.tags.get('name'), 'parking': area.tags.get('parking'),
                                   'rings': rings})

    handler = Roads()
    # Node locations kept in a file index: a whole country does not fit comfortably in memory otherwise. With an
    # area callback osmium reads each file twice (the multipolygon relations first); only the objects with a highway
    # or an amenity tag reach Python, the others stay in osmium.
    with tempfile.TemporaryDirectory(dir=root / 'tmp' if (root / 'tmp').is_dir() else None) as scratch:
        for number, pbf in enumerate(pbfs):
            handler.apply_file(str(pbf), locations=True, idx=f'sparse_file_array,{scratch}/nodes-{number}.idx',
                               filters=[osmium.filter.KeyFilter('highway', 'amenity')])
    target = root / 'derived' / 'osm' / 'roads.jsonl'
    parking = root / 'derived' / 'osm' / 'parking.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, items in ((target, handler.ways), (parking, handler.parks)):
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(item, separators=(',', ':')) + '\n' for item in items))
        temporary.replace(path)
    kinds = {}
    for way in handler.ways:
        kinds[way['highway']] = kinds.get(way['highway'], 0) + 1
    print(json.dumps({'ways': len(handler.ways), 'car_parks': len(handler.parks), 'cells': len(cells),
                      'by_type': dict(sorted(kinds.items())), 'output': str(target), 'parking': str(parking)},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
