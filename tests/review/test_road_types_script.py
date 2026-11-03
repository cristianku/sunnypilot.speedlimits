"""helpers/road_types.sh ("Find the road types") with stand-in tools: a first run without any map gets the maps the
drives need first (scripts/osm_maps.py) and goes on; a download that fails leaves the maps present and the job goes
on, and stops only when there is no map at all; the names of the places are read again only with a newer map; its
steps are the ones of nnslr_tools.job_progress."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from nnslr_tools.job_progress import progress

REPO = Path(__file__).resolve().parents[2]
# The stand-in scripts write what the real ones would, and say they ran.
FAKE_MAPS = '''import os, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--data-root') + 1])
print('osm_maps ran')
if os.environ.get('FAKE_MAPS') == 'fail':
    print('Alsace: failed: no network')
    raise SystemExit(1)
target = root / 'raw' / 'osm' / 'alsace-latest.osm.pbf'
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'pbf')
    print('Alsace: downloaded')
'''
FAKE_ROADS = '''import sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--data-root') + 1])
print('osm_roads ran')
(root / 'derived' / 'osm').mkdir(parents=True, exist_ok=True)
(root / 'derived' / 'osm' / 'roads.jsonl').write_text('{"maxspeed_type": null, "junction": null}\\n')
(root / 'derived' / 'osm' / 'parking.jsonl').write_text('')
'''
FAKE_PLACES = '''import sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--data-root') + 1])
print('osm_places ran')
(root / 'derived' / 'osm').mkdir(parents=True, exist_ok=True)
(root / 'derived' / 'osm' / 'places.jsonl').write_text('')
'''


def run_script(tmp_path: Path, maps: str = 'download', touch: bool = True) -> subprocess.CompletedProcess:
    root, tools = tmp_path / 'repo', tmp_path / 'bin'
    if not root.exists():
        (root / 'helpers').mkdir(parents=True)
        (root / 'scripts').mkdir()
        (root / '.venv' / 'bin').mkdir(parents=True)
        tools.mkdir()
        shutil.copy(REPO / 'helpers' / 'road_types.sh', root / 'helpers' / 'road_types.sh')
        for name, text in (('osm_maps.py', FAKE_MAPS), ('osm_roads.py', FAKE_ROADS), ('osm_places.py', FAKE_PLACES)):
            (root / 'scripts' / name).write_text(text)
        executables = {
            root / '.venv' / 'bin' / 'nnslr': '#!/bin/sh\necho "nnslr $1"\n',
            # "import shapely" answered yes, the rest is Python.
            root / '.venv' / 'bin' / 'python': f'#!/bin/sh\n[ "$1" = -c ] && exit 0\nexec {sys.executable} "$@"\n',
            # uv run --no-project --with osmium python SCRIPT ...: the script, with this Python.
            tools / 'uv': f'#!/bin/sh\nwhile [ "$1" != python ]; do shift; done\nshift\nexec {sys.executable} "$@"\n',
        }
        for tool, text in executables.items():
            tool.write_text(text)
            tool.chmod(0o755)
    positions = tmp_path / 'data' / 'derived' / 'gps' / 'r1' / 'positions.jsonl'
    if touch:
        positions.parent.mkdir(parents=True, exist_ok=True)
        positions.touch()
    return subprocess.run(['bash', str(root / 'helpers' / 'road_types.sh')], capture_output=True, check=False,
                          text=True, timeout=60,
                          env={'PATH': f'{tools}:/usr/bin:/bin', 'HOME': str(tmp_path), 'FAKE_MAPS': maps,
                               'NNSLR_DATA_ROOT': str(tmp_path / 'data')})


def test_a_first_run_without_a_map_downloads_the_maps_and_goes_on(tmp_path):
    process = run_script(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    lines = process.stdout.splitlines()
    steps = [line for line in lines if line.startswith('== ')]
    assert steps == ['== a. GPS of the drives', '== b. Maps the drives need', '== c. Speed of the drives',
                     '== d. Map roads and car parks near the drives', '== e. Road type of every frame',
                     '== f. Junctions of the drives']
    assert lines.index('Alsace: downloaded') < lines.index('osm_roads ran') < lines.index('osm_places ran')
    assert (tmp_path / 'data' / 'derived' / 'osm' / 'places.jsonl').exists()
    # Every step of the page's bar is reached by the log.
    running = progress('road_types', 'running', process.stdout.split('Done:')[0])
    assert {step['key']: step['state'] for step in running['steps']} == {
        'gps': 'done', 'maps': 'done', 'speed': 'done', 'map_roads': 'done', 'road_types': 'done',
        'junctions': 'current'}


def test_a_download_that_fails_leaves_the_maps_present(tmp_path):
    maps = tmp_path / 'data' / 'raw' / 'osm'
    maps.mkdir(parents=True)
    (maps / 'switzerland-latest.osm.pbf').write_bytes(b'pbf')
    process = run_script(tmp_path, maps='fail')
    assert process.returncode == 0, process.stderr + process.stdout
    assert 'the maps present are used, the job goes on' in process.stdout
    assert 'nnslr road-types' in process.stdout and 'nnslr junctions' in process.stdout


def test_without_any_map_it_stops_after_the_download(tmp_path):
    process = run_script(tmp_path, maps='fail')
    assert process.returncode == 2
    assert 'osm_maps ran' in process.stdout and 'The map is missing' in process.stderr
    assert '== c. Speed of the drives' not in process.stdout


def test_the_map_and_the_places_are_read_again_only_when_newer(tmp_path):
    assert run_script(tmp_path).returncode == 0
    data = tmp_path / 'data'
    pbf = data / 'raw' / 'osm' / 'alsace-latest.osm.pbf'
    for path in (pbf, data / 'derived' / 'gps' / 'r1' / 'positions.jsonl'):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    for name in ('roads.jsonl', 'places.jsonl', 'parking.jsonl'):
        os.utime(data / 'derived' / 'osm' / name, (1_500_000_000, 1_500_000_000))
    same = run_script(tmp_path, touch=False)
    assert same.returncode == 0, same.stderr + same.stdout
    assert 'osm_roads ran' not in same.stdout and 'osm_places ran' not in same.stdout
    assert 'The map roads and car parks are up to date.' in same.stdout
    # A map downloaded again: its roads and its places read again.
    os.utime(pbf, (1_800_000_000, 1_800_000_000))
    newer = run_script(tmp_path, touch=False)
    assert 'osm_roads ran' in newer.stdout and 'osm_places ran' in newer.stdout
