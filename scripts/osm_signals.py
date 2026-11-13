"""The traffic lights of the map, to pick the frames where the car comes to one (Review with AI › Traffic lights).

  uv run --no-project --with osmium python scripts/osm_signals.py [--data-root DIR] [--pbf FILE ...]

Keeps the OSM nodes of a traffic light for the road: ``highway`` = traffic_signals (at a junction, or alone), and a
pedestrian crossing with lights (``highway`` = crossing with ``crossing`` = traffic_signals). Writes
``derived/osm/traffic_signals.jsonl``: one line per node, ``{"id", "lat", "lon", "kind"}`` (kind: junction or crossing).

Used only offline, to choose which frames an AI reviewer looks at (nnslr_tools.ai_review.traffic_light_frames): the
map never reaches the car's vision. The map: every ``raw/osm/*.osm.pbf`` (Geofabrik extracts of OpenStreetMap,
© OpenStreetMap contributors, ODbL), read here, offline. A node near a border is in two extracts: it is kept once, by
its id. osmium (pyosmium) comes from uv's cache for this run only; the environments of the site and of the training
stay as they are. helpers/road_types.sh runs it again whenever an extract is newer than traffic_signals.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default=os.environ.get('NNSLR_DATA_ROOT') or str(Path(__file__).resolve().parents[1] / 'data'))
    parser.add_argument('--pbf', action='extend', nargs='+', help='default: every <data-root>/raw/osm/*.osm.pbf')
    args = parser.parse_args()
    root = Path(args.data_root)
    pbfs = [Path(name) for name in args.pbf] if args.pbf else sorted((root / 'raw' / 'osm').glob('*.osm.pbf'))
    missing = [str(pbf) for pbf in pbfs if not pbf.is_file()]
    if not pbfs or missing:
        print(f'osm_signals: missing {", ".join(missing) or root / "raw" / "osm" / "*.osm.pbf"}', file=sys.stderr)
        return 2
    import osmium

    found = []
    seen = set()

    class Signals(osmium.SimpleHandler):
        def node(self, node):
            highway = node.tags.get('highway')
            if highway == 'traffic_signals':
                kind = 'junction'
            elif highway == 'crossing' and node.tags.get('crossing') == 'traffic_signals':
                kind = 'crossing'
            else:
                return
            if node.location.valid() and node.id not in seen:
                seen.add(node.id)
                found.append({'id': node.id, 'lat': round(node.location.lat, 6), 'lon': round(node.location.lon, 6),
                              'kind': kind})

    handler = Signals()
    for pbf in pbfs:
        handler.apply_file(str(pbf), filters=[osmium.filter.KeyFilter('highway')])
    target = root / 'derived' / 'osm' / 'traffic_signals.jsonl'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=target.parent, delete=False, suffix='.tmp') as handle:
        for item in sorted(found, key=lambda entry: entry['id']):
            handle.write(json.dumps(item) + '\n')
    os.replace(handle.name, target)
    print(f'{len(found)} traffic lights from {len(pbfs)} extracts written to {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
