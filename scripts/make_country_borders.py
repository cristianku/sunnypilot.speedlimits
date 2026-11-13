#!/usr/bin/env python3
"""Make src/nnslr_tools/data/countries.json.gz, the country borders the review uses to propose the country of a
frame from its GPS position (nnslr_tools.gps.country_of).

Source: Natural Earth 1:10m admin-0 countries (public domain, https://www.naturalearthdata.com), the GeoJSON of
github.com/nvkelso/natural-earth-vector. Only the countries in the AREA box (Europe and around it) are kept,
with coordinates rounded to 4 decimals (about 10 m), so the file stays small; elsewhere country_of answers
None. Run again with a larger AREA to cover more countries.

  python scripts/make_country_borders.py                 download the source and write the file
  python scripts/make_country_borders.py <file.geojson>  use a source already downloaded

Natural Earth is coarse where a border follows a river (along the Rhine it puts Bad Säckingen in Switzerland).
--gisco writes finer borders into the data root instead: GISCO 1:1 million (Eurostat, administrative boundaries
© EuroGeographics, non-commercial use only, so the file stays out of git), <data root>/derived/gps/borders.json.gz,
used by nnslr_tools.gps when there; then `nnslr gps --all --countries` gives the frames their country again.

  python scripts/make_country_borders.py --gisco [--data-root DIR] [<file.geojson>]

--world writes src/nnslr_tools/data/world.json.gz: every country of the world, coarser (Natural Earth 1:50m, public
domain, 3 decimals: about 100 m), each with its continent. It places the photos of the Mapillary Traffic Sign Dataset
outside Europe (nnslr_tools.gps.world_place, nnslr mapillary-locations); Europe keeps the finer borders above.

  python scripts/make_country_borders.py --world [<file.geojson>]

--car [<clone>] writes the comma's own borders, openpilot/sunnypilot/speed_limits/country-borders.json.gz in the sunnypilot
clone (default $NNSLR_SUNNYPILOT_CLONE or /root/sunnypilot), from the two files above, as gps.frame_country takes them:
the European countries first, then (tier 1, asked only where Europe has no country) every country outside Europe and the
parts of the European ones that lie wholly outside AREA.

  python scripts/make_country_borders.py --car [<clone>]
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

SOURCE = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
          'ne_10m_admin_0_countries.geojson')
# lon_min, lat_min, lon_max, lat_max: Europe, from Portugal and Iceland to the Urals, Turkey and North Africa.
AREA = (-32.0, 27.0, 50.0, 72.0)
OUTPUT = Path(__file__).resolve().parents[1] / 'src/nnslr_tools/data/countries.json.gz'


def rounded(ring: list, digits: int = 4) -> list:
    points = []
    for lon, lat, *_ in ring:
        point = [round(lon, digits), round(lat, digits)]
        if not points or points[-1] != point:
            points.append(point)
    return points


GISCO_SOURCE = ('https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/'
                'CNTR_RG_01M_2020_4326.geojson')
# GISCO codes the countries as Eurostat does; the review keeps ISO 3166-1.
GISCO_CODES = {'EL': 'GR', 'UK': 'GB'}


def europe(features: list, code_of, name_of) -> list:
    """The countries of ``features`` (GeoJSON) inside AREA, with their polygons, bbox, ISO code and name."""
    countries = []
    for feature in features:
        properties = feature['properties']
        code = code_of(properties)
        if not code or code == '-99':
            continue
        geometry = feature['geometry']
        if geometry is None:
            continue
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        kept = []
        for polygon in polygons:
            outer = polygon[0]
            lons = [point[0] for point in outer]
            lats = [point[1] for point in outer]
            if max(lons) < AREA[0] or min(lons) > AREA[2] or max(lats) < AREA[1] or min(lats) > AREA[3]:
                continue
            kept.append([rounded(ring) for ring in polygon])
        if not kept:
            continue
        lons = [point[0] for polygon in kept for point in polygon[0]]
        lats = [point[1] for polygon in kept for point in polygon[0]]
        countries.append({'code': code.upper(), 'name': name_of(properties),
                          'bbox': [min(lons), min(lats), max(lons), max(lats)], 'polygons': kept})
    countries.sort(key=lambda country: country['code'])
    return countries


def gisco(arguments: list[str]) -> int:
    root = None
    if '--data-root' in arguments:
        index = arguments.index('--data-root')
        root = Path(arguments[index + 1])
        del arguments[index:index + 2]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from nnslr_tools.gps import DETAILED_BORDERS
    from nnslr_tools.preannotate import resolve_data_root
    root = resolve_data_root(root)
    if arguments:
        data = json.loads(Path(arguments[0]).read_text())
    else:
        with urllib.request.urlopen(GISCO_SOURCE, timeout=300) as response:
            data = json.loads(response.read())
    countries = europe(data['features'], lambda properties: GISCO_CODES.get(properties.get('CNTR_ID'),
                                                                             properties.get('CNTR_ID')),
                       lambda properties: properties.get('NAME_ENGL') or properties.get('CNTR_NAME'))
    payload = {'source': GISCO_SOURCE, 'license': 'GISCO (Eurostat), administrative boundaries © EuroGeographics: '
               'non-commercial use only', 'area': list(AREA), 'countries': countries}
    output = root / DETAILED_BORDERS
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    temporary.replace(output)
    print(f'{len(countries)} countries -> {output} ({output.stat().st_size // 1024} KiB)')
    return 0


WORLD_SOURCE = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
                'ne_50m_admin_0_countries.geojson')
WORLD_OUTPUT = OUTPUT.with_name('world.json.gz')
# Natural Earth's continents, as the keys of nnslr_tools.gps.CONTINENTS.
CONTINENT_KEYS = {'Europe': 'europe', 'Asia': 'asia', 'Africa': 'africa', 'North America': 'north_america',
                  'South America': 'south_america', 'Oceania': 'oceania', 'Antarctica': 'antarctica'}


def world_continent(properties: dict) -> str | None:
    """The continent of a Natural Earth country; the small islands it puts in "Seven seas (open ocean)" by their UN
    region (Mauritius in Africa, the Maldives in Asia)."""
    continent = CONTINENT_KEYS.get(properties.get('CONTINENT'))
    if continent:
        return continent
    region, subregion = properties.get('REGION_UN'), properties.get('SUBREGION')
    if region == 'Americas':
        return 'south_america' if subregion == 'South America' else 'north_america'
    return CONTINENT_KEYS.get(region)


def world(arguments: list[str]) -> int:
    if arguments:
        data = json.loads(Path(arguments[0]).read_text())
    else:
        with urllib.request.urlopen(WORLD_SOURCE, timeout=300) as response:
            data = json.loads(response.read())
    countries = []
    for feature in data['features']:
        properties = feature['properties']
        code = properties.get('ISO_A2_EH') or properties.get('ISO_A2')
        geometry = feature['geometry']
        if not code or code == '-99' or geometry is None:
            continue
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        kept = [[rounded(ring, 3) for ring in polygon] for polygon in polygons]
        lons = [point[0] for polygon in kept for point in polygon[0]]
        lats = [point[1] for polygon in kept for point in polygon[0]]
        countries.append({'code': code.upper(), 'name': properties.get('NAME_EN') or properties.get('NAME'),
                          'continent': world_continent(properties),
                          'bbox': [min(lons), min(lats), max(lons), max(lats)], 'polygons': kept})
    countries.sort(key=lambda country: country['code'])
    payload = {'source': WORLD_SOURCE, 'license': 'Natural Earth, public domain', 'countries': countries}
    WORLD_OUTPUT.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    print(f'{len(countries)} countries -> {WORLD_OUTPUT} ({WORLD_OUTPUT.stat().st_size // 1024} KiB)')
    return 0


def touches_area(ring: list) -> bool:
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    return not (max(lons) < AREA[0] or min(lons) > AREA[2] or max(lats) < AREA[1] or min(lats) > AREA[3])


def car(arguments: list[str]) -> int:
    clone = Path(arguments[0] if arguments else os.environ.get('NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'))
    output = clone / 'openpilot/sunnypilot/speed_limits/country-borders.json.gz'
    europe_file = json.loads(gzip.decompress(OUTPUT.read_bytes()))
    world_file = json.loads(gzip.decompress(WORLD_OUTPUT.read_bytes()))
    outside = []
    for country in world_file['countries']:
        polygons = country['polygons']
        if country['continent'] == 'europe':
            # Europe's file has each polygon that touches AREA: the world gives the rest (Svalbard, Sakhalin, Reunion).
            polygons = [polygon for polygon in polygons if not touches_area(polygon[0])]
        if polygons:
            lons = [point[0] for polygon in polygons for point in polygon[0]]
            lats = [point[1] for polygon in polygons for point in polygon[0]]
            outside.append({'code': country['code'], 'name': country['name'], 'tier': 1,
                            'bbox': [min(lons), min(lats), max(lons), max(lats)], 'polygons': polygons})
    payload = {'source': [europe_file['source'], world_file['source']], 'license': 'Natural Earth, public domain',
               'tiers': ['Europe, 1:10m', 'outside Europe, 1:50m: only where Europe has no country'],
               'countries': europe_file['countries'] + outside}
    output.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    print(f'{len(europe_file["countries"])} + {len(outside)} countries -> {output} ({output.stat().st_size // 1024} KiB)')
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == '--car':
        return car(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == '--gisco':
        return gisco(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == '--world':
        return world(sys.argv[2:])
    if len(sys.argv) > 1:
        data = json.loads(Path(sys.argv[1]).read_text())
    else:
        with urllib.request.urlopen(SOURCE, timeout=120) as response:
            data = json.loads(response.read())
    countries = []
    for feature in data['features']:
        properties = feature['properties']
        # ISO_A2_EH also codes the countries ISO_A2 leaves as -99 (France, Norway).
        code = properties.get('ISO_A2_EH') or properties.get('ISO_A2')
        if not code or code == '-99':
            continue
        geometry = feature['geometry']
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        kept = []
        for polygon in polygons:
            outer = polygon[0]
            lons = [point[0] for point in outer]
            lats = [point[1] for point in outer]
            if max(lons) < AREA[0] or min(lons) > AREA[2] or max(lats) < AREA[1] or min(lats) > AREA[3]:
                continue
            kept.append([rounded(ring) for ring in polygon])
        if not kept:
            continue
        lons = [point[0] for polygon in kept for point in polygon[0]]
        lats = [point[1] for polygon in kept for point in polygon[0]]
        countries.append({'code': code.upper(), 'name': properties.get('NAME_EN') or properties.get('NAME'),
                          'bbox': [min(lons), min(lats), max(lons), max(lats)], 'polygons': kept})
    countries.sort(key=lambda country: country['code'])
    payload = {'source': SOURCE, 'license': 'Natural Earth, public domain', 'area': list(AREA),
               'countries': countries}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
    print(f'{len(countries)} countries -> {OUTPUT} ({OUTPUT.stat().st_size // 1024} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
