"""Exercise an unapplied runtime patch, never modify the live runtime checkout."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = '8d37cf946'
RUNTIME = Path(os.environ.get('NNSLR_UNITS_RUNTIME', '/root/sunnypilot'))
FILES = ('openpilot/sunnypilot/speed_limits/vision.py',
         'openpilot/sunnypilot/speed_limits/speedvisiond.py',
         'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py',
         'openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_assist.py',
         'openpilot/cereal/custom.capnp',
         'openpilot/cereal/include/c++.capnp',
         'openpilot/sunnypilot/speed_limits/tests/test_vision.py',
         'openpilot/sunnypilot/speed_limits/tests/test_speed_vision_sp.py')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def candidate(tmp_path_factory):
    target = tmp_path_factory.mktemp('units-runtime')
    for name in FILES:
        content = subprocess.check_output(['git', '-C', str(RUNTIME), 'show', f'{BASE}:{name}'])
        p = target / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    patch = ROOT / 'patches/speed-limits-unit-normalization/runtime.patch'
    if patch.exists() and not os.environ.get('NNSLR_UNITS_UNPATCHED'):
        subprocess.run(['patch', '--batch', '-p1', '-d', str(target), '-i', str(patch)],
                       check=True, capture_output=True)
    return target


@pytest.fixture(scope='session')
def vision(candidate):
    return load(candidate / FILES[0], 'units_vision')


@pytest.fixture(scope='session')
def country(candidate):
    path = candidate / 'openpilot/sunnypilot/speed_limits/country.py'
    assert path.exists(), 'offline country resolver is not prepared'
    return load(path, 'units_country')
