"""scripts/osm_places.py, the names of the places from every extract of raw/osm: a place in two extracts (near a
border) kept once, a hamlet kept for drives.FALLBACK, a missing map said before osmium is needed. osmium is never in
the .venv: without it the script runs as helpers/road_types.sh runs it, through uv."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('osm_places', REPO / 'scripts' / 'osm_places.py')
osm_places = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(osm_places)

NODE = '<node id="{id}" version="1" lat="{lat}" lon="{lon}">{tags}</node>'


def extract(path: Path, *nodes: tuple[int, float, float, dict]) -> Path:
    body = ''.join(NODE.format(id=id_, lat=lat, lon=lon,
                               tags=''.join(f'<tag k="{key}" v="{value}"/>' for key, value in tags.items()))
                   for id_, lat, lon, tags in nodes)
    path.write_text(f'<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">{body}</osm>')
    return path


def test_without_a_map_it_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['osm_places.py', '--data-root', str(tmp_path)])
    assert osm_places.main() == 2
    assert 'missing' in capsys.readouterr().err
    monkeypatch.setattr(sys, 'argv', ['osm_places.py', '--data-root', str(tmp_path), '--pbf', str(tmp_path / 'no.pbf')])
    assert osm_places.main() == 2


def run(arguments: list[str], monkeypatch) -> int:
    """The script with these arguments: here when osmium is importable, else through uv as helpers/road_types.sh runs
    it (skipped only without uv)."""
    if importlib.util.find_spec('osmium') is not None:
        monkeypatch.setattr(sys, 'argv', ['osm_places.py', *arguments])
        return osm_places.main()
    uv = shutil.which('uv')
    if uv is None:
        pytest.skip('neither osmium nor uv')
    process = subprocess.run([uv, 'run', '--quiet', '--no-project', '--with', 'osmium', 'python',
                              str(REPO / 'scripts' / 'osm_places.py'), *arguments],
                             capture_output=True, text=True, timeout=300, check=False)
    assert process.returncode == 0, process.stderr + process.stdout
    return process.returncode


def test_every_extract_read_and_a_place_kept_once(tmp_path, monkeypatch):
    basel = (1, 47.5596, 7.5886, {'place': 'city', 'name': 'Basel', 'population': '173000'})
    first = extract(tmp_path / 'switzerland.osm', basel, (2, 47.0, 8.0, {'place': 'hamlet', 'name': 'Hof'}),
                    (3, 47.1, 8.1, {'place': 'locality', 'name': 'Feld'}), (4, 47.2, 8.2, {'place': 'village'}))
    second = extract(tmp_path / 'alsace.osm', basel, (5, 47.75, 7.34, {'place': 'city', 'name': 'Mulhouse'}))
    assert run(['--data-root', str(tmp_path), '--pbf', str(first), str(second)], monkeypatch) == 0
    lines = (tmp_path / 'derived' / 'osm' / 'places.jsonl').read_text().splitlines()
    places = [json.loads(line) for line in lines]
    assert [(item['name'], item['place']) for item in places] == [('Basel', 'city'), ('Hof', 'hamlet'),
                                                                  ('Mulhouse', 'city')]
    assert places[0]['population'] == 173000 and places[1]['population'] is None
