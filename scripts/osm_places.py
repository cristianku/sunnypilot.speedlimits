#!/usr/bin/env python3
"""The names of the towns and villages, to call each drive by where it went (nnslr_tools.drives).

  uv run --no-project --with osmium python scripts/osm_places.py [--data-root DIR] [--pbf FILE ...]

Keeps the OSM nodes tagged ``place`` = city, town, village, suburb or hamlet that have a name (a hamlet names a drive
only where no other place is near: nnslr_tools.drives.FALLBACK). Writes
``derived/osm/places.jsonl``: one line per place, ``{"name", "place", "lat", "lon", "population"}``.

The map: every ``raw/osm/*.osm.pbf`` (Geofabrik extracts of OpenStreetMap, © OpenStreetMap contributors, ODbL), as
scripts/osm_roads.py reads them: Switzerland, Baden-Württemberg, the US states of the drives there (scripts/osm_maps.py
downloads the extracts the drives need). A place near a border is in both extracts: it is kept once, by its id. Read
here, offline: no position of the drives leaves the server. osmium (pyosmium) comes from uv's cache for this run only;
the environments of the site and of the training stay as they are. helpers/road_types.sh runs it again whenever an
extract is newer than places.jsonl; about a minute for five extracts (2 GB), as every block is unpacked.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PLACES = {'city', 'town', 'village', 'suburb', 'hamlet'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default=os.environ.get('NNSLR_DATA_ROOT') or str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--pbf', action='extend', nargs='+', help='default: every <data-root>/raw/osm/*.osm.pbf')
    args = parser.parse_args()
    root = Path(args.data_root)
    pbfs = [Path(name) for name in args.pbf] if args.pbf else sorted((root / 'raw' / 'osm').glob('*.osm.pbf'))
    missing = [str(pbf) for pbf in pbfs if not pbf.is_file()]
    if not pbfs or missing:
        print(f'osm_places: missing {", ".join(missing) or root / "raw" / "osm" / "*.osm.pbf"}', file=sys.stderr)
        return 2
    import osmium

    found = []
    seen = set()

    class Places(osmium.SimpleHandler):
        def node(self, node):
            kind = node.tags.get('place')
            name = node.tags.get('name')
            if kind in PLACES and name and node.location.valid() and node.id not in seen:
                seen.add(node.id)
                population = node.tags.get('population', '')
                found.append({'name': name, 'place': kind, 'lat': round(node.location.lat, 5),
                              'lon': round(node.location.lon, 5),
                              'population': int(population) if population.isdigit() else None})

    # Only the objects with a place tag reach Python, the others stay in osmium.
    handler = Places()
    for pbf in pbfs:
        handler.apply_file(str(pbf), filters=[osmium.filter.KeyFilter('place')])
    target = root / 'derived' / 'osm' / 'places.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=target.parent, delete=False, suffix='.tmp') as handle:
        for item in sorted(found, key=lambda entry: (entry['name'], entry['lat'])):
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(handle.name, target)
    print(f'{len(found)} places from {len(pbfs)} extracts written to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
