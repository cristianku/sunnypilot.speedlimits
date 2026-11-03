"""scripts/osm_maps.py, the maps the drives need, with a stand-in Geofabrik: the smallest extract that holds a
position, the maps present covering theirs, a stray square that pulls no state into the region around it, the names
of the files and of the extracts, a download kept only whole and with the md5 of the file of the day, a part of another
download left alone and one of a stopped run removed, the limit of a run, and only Geofabrik asked, redirects too."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Looked for, not imported: the collection of the tests keeps numpy out of the process (tests/cli/test_import_isolation).
pytestmark = pytest.mark.skipif(importlib.util.find_spec('shapely') is None, reason='shapely is not installed')
REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('osm_maps', REPO / 'scripts' / 'osm_maps.py')
osm_maps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(osm_maps)


def box(lon1, lat1, lon2, lat2):
    return {'type': 'Polygon', 'coordinates': [[[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]]]}


def feature(id_, path, geometry, name=None):
    return {'type': 'Feature', 'geometry': geometry,
            'properties': {'id': id_, 'name': name or id_, 'urls': {'pbf': osm_maps.HOST + path}}}


INDEX = {'type': 'FeatureCollection', 'features': [
    feature('europe', 'europe-latest.osm.pbf', box(0, 40, 20, 60), 'Europe'),
    feature('switzerland', 'europe/switzerland-latest.osm.pbf', box(6, 45.8, 10, 47.8), 'Switzerland'),
    feature('baden-wuerttemberg', 'europe/germany/baden-wuerttemberg-latest.osm.pbf', box(7.5, 47.5, 10.5, 49.8),
            'Baden-Württemberg'),
    feature('alsace', 'europe/france/alsace-latest.osm.pbf', box(6.8, 47.4, 8.3, 49.1), 'Alsace'),
    feature('nord-norge', 'europe/norway/nord-norge-latest.osm.pbf', box(12, 64, 20, 60.5),
            'Nord-Norge<br />(Northern Norway)'),
    # Two extracts that Geofabrik names the same file in two folders.
    feature('georgia', 'asia/georgia-latest.osm.pbf', box(40, 41, 46, 43.6), 'Georgia'),
    feature('us/georgia', 'north-america/us/georgia-latest.osm.pbf', box(-85.6, 30.3, -80.8, 35), 'us/georgia'),
    feature('us/district-of-columbia', 'north-america/us/district-of-columbia-latest.osm.pbf',
            box(-77.2, 38.8, -76.9, 39.0), 'us/district-of-columbia'),
    # The regions of regions around the states: Lake Michigan is only in us-midwest, the Atlantic only in the US.
    feature('us', 'north-america/us-latest.osm.pbf', box(-125, 24, -66, 50), 'United States of America'),
    feature('us-midwest', 'north-america/us-midwest-latest.osm.pbf', box(-104, 36, -80, 49.4), 'US Midwest'),
    feature('us/illinois', 'north-america/us/illinois-latest.osm.pbf', box(-91.5, 37, -87.5, 42.5), 'us/illinois'),
    feature('us/indiana', 'north-america/us/indiana-latest.osm.pbf', box(-88.1, 37.8, -84.8, 41.8), 'us/indiana'),
]}
BASEL, STUTTGART, SEA = (47.55, 7.6), (48.78, 9.18), (0.0, 0.0)
CHICAGO = [(41.805 + 0.01 * step, -87.70) for step in range(8)]
LAKE_MICHIGAN, ATLANTIC = (42.127, -87.382), (33.599, -75.246)


class Response:
    def __init__(self, data: bytes, url: str = ''):
        self.data, self.at, self.url = data, 0, url
        self.headers = {'Content-Length': str(len(data))}

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        end = len(self.data) if size < 0 else self.at + size
        chunk, self.at = self.data[self.at:end], min(end, len(self.data))
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(monkeypatch, files: dict[str, bytes], wrong_md5: bool = False, index: bool = True,
          redirects: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """A stand-in Geofabrik with these files (path under HOST), the md5 of each and the index, and the redirects of
    -latest to the file of the day ({path: path}); the calls made."""
    calls = []
    served = {'index-v1.json': json.dumps(INDEX).encode(), **files} if index else dict(files)
    for path, data in files.items():
        digest = hashlib.md5(data + (b'x' if wrong_md5 else b'')).hexdigest()
        served[path + '.md5'] = f'{digest}  {path.rsplit("/", 1)[1]}\n'.encode()

    def fake_open(url, method='GET'):
        calls.append((method, url))
        if not url.startswith(osm_maps.HOST):
            raise osm_maps.MapError('not a Geofabrik address: ' + url)
        path = (redirects or {}).get(url[len(osm_maps.HOST):], url[len(osm_maps.HOST):])
        if path not in served:
            raise urllib.error.URLError('not found: ' + path)
        return Response(served[path], osm_maps.HOST + path)

    monkeypatch.setattr(osm_maps, '_open', fake_open)
    return calls


def drive(root: Path, route: str, *positions):
    path = root / 'derived' / 'gps' / route / 'positions.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'latitude': lat, 'longitude': lon}) + '\n' for lat, lon in positions)
                    + json.dumps({'latitude': None, 'longitude': None}) + '\n')


def test_the_extracts_by_their_file_and_their_name():
    found = {extract['id']: extract for extract in osm_maps.extracts(INDEX)}
    assert found['baden-wuerttemberg']['file'] == 'baden-wuerttemberg-latest.osm.pbf'
    # One name for two files: each by its id.
    assert (found['georgia']['file'], found['us/georgia']['file']) == ('georgia-latest.osm.pbf',
                                                                       'us-georgia-latest.osm.pbf')
    assert found['us/georgia']['name'] == 'Georgia'
    assert found['us/district-of-columbia']['name'] == 'District of Columbia'
    assert found['nord-norge']['name'] == 'Nord-Norge'
    assert osm_maps.stem_name('my-region-latest.osm.pbf') == 'My Region'


def test_the_smallest_extract_and_the_maps_present(tmp_path):
    all_extracts = osm_maps.extracts(INDEX)
    drive(tmp_path, 'r1', BASEL)
    cells = osm_maps.grid_positions(tmp_path)
    needed, nowhere, outside = osm_maps.plan(cells, all_extracts, set())
    # Basel is in Switzerland, Baden-Württemberg, Alsace and Europe: Alsace is the smallest.
    assert list(needed) == ['alsace'] and needed['alsace']['routes'] == ['r1'] and not nowhere
    assert set(outside) == {'r1'}
    # Switzerland present: nothing needed.
    assert osm_maps.plan(cells, all_extracts, {'switzerland-latest.osm.pbf'})[0] == {}
    # With Stuttgart too: Baden-Württemberg holds both, one download instead of two.
    drive(tmp_path, 'r2', STUTTGART, SEA)
    needed, nowhere, outside = osm_maps.plan(osm_maps.grid_positions(tmp_path), all_extracts, set())
    assert list(needed) == ['baden-wuerttemberg'] and needed['baden-wuerttemberg']['routes'] == ['r1', 'r2']
    # The sea is on no map of Geofabrik.
    assert len(nowhere) == 1 and outside['r2'] and len(outside['r1']) == 1


def test_a_stray_square_pulls_no_state_into_the_region_around_it(tmp_path):
    all_extracts = osm_maps.extracts(INDEX)
    drive(tmp_path, 'r1', *CHICAGO, LAKE_MICHIGAN)
    needed, nowhere, outside = osm_maps.plan(osm_maps.grid_positions(tmp_path), all_extracts, set())
    # Not us-midwest for all of them: Illinois, and the square in the lake said like one on no map.
    assert list(needed) == ['us/illinois'] and len(needed['us/illinois']['cells']) == len(CHICAGO)
    assert len(nowhere) == 1 and len(outside['r1']) == len(CHICAGO) + 1
    # Illinois there: nothing to download for a square in the lake, nor for one in the Atlantic (only in the US).
    drive(tmp_path, 'r2', ATLANTIC)
    cells = osm_maps.grid_positions(tmp_path)
    needed, nowhere, _ = osm_maps.plan(cells, all_extracts, {'illinois-latest.osm.pbf'})
    assert needed == {} and len(nowhere) == 2
    # More squares than a stray between the states (a ferry): the region, with the square in the lake, and Illinois
    # still on its own; the Atlantic stays on no map.
    drive(tmp_path, 'r3', *[(43.0, -86.505 + 0.01 * step) for step in range(osm_maps.GAP_SQUARES + 1)])
    needed, nowhere, _ = osm_maps.plan(osm_maps.grid_positions(tmp_path), all_extracts, set())
    assert sorted(needed) == ['us-midwest', 'us/illinois'] and len(nowhere) == 1
    assert len(needed['us-midwest']['cells']) == osm_maps.GAP_SQUARES + 2
    # Indiana beside Illinois (2×): its one square in both goes with Illinois, one download fewer.
    drive(tmp_path, 'r4', (41.0, -87.8))
    needed = osm_maps.plan(osm_maps.grid_positions(tmp_path), all_extracts, set())[0]
    assert 'us/indiana' not in needed and needed['us/illinois']['routes'] == ['r1', 'r4']


def test_a_download_is_kept_only_whole_and_checked(tmp_path, monkeypatch):
    data = os.urandom(3000)
    path = 'europe/france/alsace-latest.osm.pbf'
    extract = next(item for item in osm_maps.extracts(INDEX) if item['id'] == 'alsace')
    serve(monkeypatch, {path: data})
    assert osm_maps.download(extract, tmp_path, 10_000) == 3000
    assert (tmp_path / 'alsace-latest.osm.pbf').read_bytes() == data
    assert not (tmp_path / '.alsace-latest.osm.pbf.part').exists()
    # Already there: never downloaded again over it.
    with pytest.raises(osm_maps.MapError, match='already there'):
        osm_maps.download(extract, tmp_path, 10_000)


def test_the_md5_of_the_file_of_the_day(tmp_path, monkeypatch):
    # -latest leads to the file of the day; the md5 of -latest is already of the next day's.
    extract = next(item for item in osm_maps.extracts(INDEX) if item['id'] == 'alsace')
    data = os.urandom(3000)
    calls = serve(monkeypatch, {'europe/france/alsace-261102.osm.pbf': data,
                                'europe/france/alsace-latest.osm.pbf.md5': b'0' * 32 + b'  alsace-latest.osm.pbf\n'},
                  redirects={'europe/france/alsace-latest.osm.pbf': 'europe/france/alsace-261102.osm.pbf'})
    assert osm_maps.download(extract, tmp_path, 10_000) == 3000
    assert (tmp_path / 'alsace-latest.osm.pbf').read_bytes() == data
    assert ('GET', osm_maps.HOST + 'europe/france/alsace-261102.osm.pbf.md5') in calls
    assert not any(url.endswith('-latest.osm.pbf.md5') for _, url in calls)


def test_a_stopped_download_leaves_no_part(tmp_path, monkeypatch):
    # Stopped from the portal: SIGTERM is an exit (the script's handler), and the part goes with it.
    class Stopped(Response):
        def read(self, size=-1):
            if self.at:
                raise SystemExit(143)
            return super().read(size)

    extract = next(item for item in osm_maps.extracts(INDEX) if item['id'] == 'alsace')
    data = b'a' * (2 * osm_maps.CHUNK)
    serve(monkeypatch, {'europe/france/alsace-latest.osm.pbf': data})
    answers = osm_maps._open
    monkeypatch.setattr(osm_maps, '_open', lambda url, method='GET': Stopped(data, url) if url.endswith('.pbf')
                        else answers(url, method))
    with pytest.raises(SystemExit):
        osm_maps.download(extract, tmp_path, 10 * len(data))
    assert list(tmp_path.iterdir()) == []


def test_a_wrong_md5_or_a_file_too_big_keeps_nothing(tmp_path, monkeypatch):
    extract = next(item for item in osm_maps.extracts(INDEX) if item['id'] == 'alsace')
    serve(monkeypatch, {'europe/france/alsace-latest.osm.pbf': b'a' * 3000}, wrong_md5=True)
    with pytest.raises(osm_maps.MapError, match='not kept'):
        osm_maps.download(extract, tmp_path, 10_000)
    assert list(tmp_path.iterdir()) == []
    serve(monkeypatch, {'europe/france/alsace-latest.osm.pbf': b'a' * 3000})
    with pytest.raises(osm_maps.MapError, match='max-gb'):
        osm_maps.download(extract, tmp_path, 1000)
    assert list(tmp_path.iterdir()) == []


def test_the_part_of_another_download_is_left_alone(tmp_path, monkeypatch):
    extract = next(item for item in osm_maps.extracts(INDEX) if item['id'] == 'alsace')
    serve(monkeypatch, {'europe/france/alsace-latest.osm.pbf': b'a' * 3000})
    part = tmp_path / '.alsace-latest.osm.pbf.part'
    part.write_bytes(b'half')
    with pytest.raises(osm_maps.MapError, match='left alone'):
        osm_maps.download(extract, tmp_path, 10_000)
    assert part.read_bytes() == b'half'
    # One of a run that stopped long ago: started again.
    os.utime(part, (1_000_000_000, 1_000_000_000))
    assert osm_maps.download(extract, tmp_path, 10_000) == 3000 and not part.exists()


def test_only_geofabrik_is_asked():
    with pytest.raises(osm_maps.MapError, match='not a Geofabrik address'):
        osm_maps._open('https://example.com/index-v1.json')
    # Nor by a redirect: one on Geofabrik is followed, one elsewhere refused before it is asked.
    handler, request = osm_maps._GeofabrikOnly(), urllib.request.Request(osm_maps.HOST + 'europe-latest.osm.pbf')
    followed = handler.redirect_request(request, io.BytesIO(), 307, 'Temporary Redirect', {},
                                        osm_maps.HOST + 'europe-261102.osm.pbf')
    assert followed.full_url == osm_maps.HOST + 'europe-261102.osm.pbf'
    # Elsewhere also when the address starts like Geofabrik's: a longer host, or Geofabrik's name as a user before '@'.
    for elsewhere in ('https://example.com/x.pbf', 'https://download.geofabrik.de.example.com/x.pbf',
                      'https://download.geofabrik.de@example.com/x.pbf', 'http://download.geofabrik.de/x.pbf'):
        with pytest.raises(osm_maps.MapError, match='elsewhere'):
            handler.redirect_request(request, io.BytesIO(), 307, 'Temporary Redirect', {}, elsewhere)


def test_a_run_downloads_the_maps_the_drives_need(tmp_path, monkeypatch, capsys):
    calls = serve(monkeypatch, {'europe/germany/baden-wuerttemberg-latest.osm.pbf': b'b' * 4000})
    drive(tmp_path, 'r1', BASEL)
    drive(tmp_path, 'r2', STUTTGART)
    # A dry run says what and how big, and downloads nothing.
    assert osm_maps.main(['--data-root', str(tmp_path), '--dry-run']) == 0
    status = json.loads((tmp_path / 'derived' / 'osm' / 'maps.json').read_text())
    assert [(row['name'], row['state'], row['bytes']) for row in status['needed']] == [
        ('Baden-Württemberg', 'to download', 4000)]
    assert not (tmp_path / 'raw' / 'osm' / 'baden-wuerttemberg-latest.osm.pbf').exists()
    assert (tmp_path / 'raw' / 'osm' / osm_maps.INDEX_NAME).is_file()
    # The run.
    assert osm_maps.main(['--data-root', str(tmp_path)]) == 0
    assert (tmp_path / 'raw' / 'osm' / 'baden-wuerttemberg-latest.osm.pbf').read_bytes() == b'b' * 4000
    status = json.loads((tmp_path / 'derived' / 'osm' / 'maps.json').read_text())
    assert status['needed'][0]['state'] == 'downloaded' and status['outside'] == []
    assert [item['name'] for item in status['maps']] == ['Baden-Württemberg']
    assert status['downloads'][0]['bytes'] == 4000
    # Then nothing is needed; the index kept is used, not fetched again.
    before = len(calls)
    assert osm_maps.main(['--data-root', str(tmp_path)]) == 0
    assert json.loads((tmp_path / 'derived' / 'osm' / 'maps.json').read_text())['needed'] == []
    assert len(calls) == before
    # Only Geofabrik, and never a position of the drives in what was asked.
    assert all(url.startswith(osm_maps.HOST) for _, url in calls)
    assert not any('47.55' in url or '7.6' in url for _, url in calls)


def test_a_map_beyond_the_limit_is_not_downloaded(tmp_path, monkeypatch):
    serve(monkeypatch, {'europe/germany/baden-wuerttemberg-latest.osm.pbf': b'b' * 4000})
    drive(tmp_path, 'r2', STUTTGART)
    assert osm_maps.main(['--data-root', str(tmp_path), '--max-gb', '0.000001']) == 1
    status = json.loads((tmp_path / 'derived' / 'osm' / 'maps.json').read_text())
    assert status['needed'][0]['state'] == 'too big' and 'max-gb' in status['needed'][0]['reason']
    assert status['outside'] == [{'route': 'r2', 'positions': 1}]
    assert not (tmp_path / 'raw' / 'osm' / 'baden-wuerttemberg-latest.osm.pbf').exists()


def test_a_failed_download_is_said_and_the_kept_index_used_offline(tmp_path, monkeypatch, capsys):
    # The index kept, older than INDEX_DAYS; Geofabrik out of reach.
    folder = tmp_path / 'raw' / 'osm'
    folder.mkdir(parents=True)
    (folder / osm_maps.INDEX_NAME).write_text(json.dumps(INDEX))
    os.utime(folder / osm_maps.INDEX_NAME, (1_000_000_000, 1_000_000_000))
    serve(monkeypatch, {}, index=False)
    drive(tmp_path, 'r2', STUTTGART)
    assert osm_maps.main(['--data-root', str(tmp_path)]) == 1
    status = json.loads((tmp_path / 'derived' / 'osm' / 'maps.json').read_text())
    assert status['index']['offline'] is True
    assert status['needed'][0]['state'] == 'failed' and status['needed'][0]['bytes'] is None
    assert [path.name for path in folder.iterdir() if not path.name.startswith('.')] == [osm_maps.INDEX_NAME]
    # A dry run without the sizes says so, not "1 MB".
    capsys.readouterr()
    assert osm_maps.main(['--data-root', str(tmp_path), '--dry-run']) == 0
    assert 'Dry run: 1 to download, size unknown; nothing downloaded.' in capsys.readouterr().out


def test_without_an_index_nothing_is_chosen(tmp_path, monkeypatch):
    serve(monkeypatch, {}, index=False)
    drive(tmp_path, 'r2', STUTTGART)
    assert osm_maps.main(['--data-root', str(tmp_path)]) == 1
    assert json.loads((tmp_path / 'derived' / 'osm' / 'maps.json').read_text())['index'] is None
    # No drive with positions: nothing to choose, nothing missing.
    assert osm_maps.main(['--data-root', str(tmp_path / 'empty')]) == 0
