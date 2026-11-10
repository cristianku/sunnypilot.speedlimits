"""``nnslr try-on-comma``: "Try on the comma now" (Models › a SignNet version) and "Back to the build's networks".

A SignNet of this server and the classifier in use copied straight onto the comma over ssh, into the models folder of
the build it runs (the submodule openpilot/sunnypilot/speed_limits/models): no publication, no build. The comma's
speed_limits_warmup (offroad, while Show Vision Speed Limit is on or Speed Limit Source is Smartspeed) compiles every network whose compiled file is missing, and speedvisiond loads them when a drive starts: copying is
enough. The trial lasts until the comma's next software update, which puts back the build's networks; --restore does it
now.

1. Check the comma: it answers over ssh, it is parked (IsOnroad), its build runs the speed-limits networks
   (runner.py with MANIFEST_FORMAT = 2, warmup.py, vision.py) and prepares them (the toggle or the source), and where
   it keeps the compiled networks (Paths.model_root in hw.py). Its vision.py is copied here, read-only.
2. Export the SignNet and the classifier as Publish does (nnslr_tools.car_release).
3. Check them with the comma's code, before anything is copied: the contracts against the comma's vision.py (SignNet's
   input band and grid step, its one output of 5 channels, heat first; the classifier's input size, normalisation and
   classes as the comma reads them into speeds), then the comma's own read_signs, with both ONNX under onnxruntime on
   reviewed frames of the test with a confirmed speed sign: it must read the confirmed speed on most of them.
4. Copy them: each file to <name>.tmp, checked by its sha256 on the comma, then moved into place, speed_limits.json
   last with the version <day>-try. Recorded in models/on-comma-trial.json of the data root.
5. Wait for the comma to compile them (both <model_root>/speed_limits/<stem>_<sha256[:16]>_tinygrad.pkl, read every
   15 s for up to 6 minutes). When they do not appear, the build's networks are put back (the comma is never left
   without a reading) and the job fails with the warmup's lines of the comma's log.

Nothing else on the comma is touched: no reboot, no process restarted, no git command but the one of --restore
(git checkout -- . in the models folder). Every path is fixed here or by the environment of the site; the SignNet is a
run name of the Models tab, the classifier an id of the Models tab.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nnslr_tools.car_release import (
    BRAIN_FILE,
    DECIDER_FILE,
    FILES,
    GPU_VENV_ENV,
    MANIFEST,
    choose,
    export,
    manifest_of,
)
from nnslr_tools import car_settings
from nnslr_tools.model_roles import _sha256
from nnslr_tools.sign_types import HOURS_CLASS, WET_CLASS

TRY_SCRIPT = 'scripts/try_on_comma.sh'
TRIAL_FILE = 'models/on-comma-trial.json'
# The comma, as root's ssh config names it, and its folders (the environment overrides them: the tests' fake comma).
HOST_ENV, HOST = 'NNSLR_COMMA_SSH', 'comma'
OPENPILOT_ENV, OPENPILOT = 'NNSLR_COMMA_OPENPILOT', '/data/openpilot'
PARAMS_ENV, PARAMS = 'NNSLR_COMMA_PARAMS', '/data/params/d'
LOG_ENV, LOG = 'NNSLR_COMMA_LOG', '/data/log'
PACKAGE = 'openpilot/sunnypilot/speed_limits'
MODELS_DIR = PACKAGE + '/models'
HARDWARE = 'openpilot/common/hardware/hw.py'
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10')
SSH_TIMEOUT = 120
# The comma's params: the Speed Limit Sources that use the networks, by their SpeedLimitPolicy (speed_limit/common.py
# Policy.vision_only, Smartspeed; a 5, the old Map Data + Vision, still on a comma not updated yet).
VISION_POLICIES = {'5': 'Map Data + Vision', '6': 'Smartspeed'}
POLL_SECONDS = 15
WAIT_SECONDS = 360
FRAMES = 5
FRAMES_READ = 3
# The classes that give a speed, as the dataset names them (annotations.py: the families with a value, and a number with
# a plate): the family, then the km/h; the end of a limit is 0.
SPEED_FAMILIES = ('maximum_speed', HOURS_CLASS, WET_CLASS, 'variable_display', 'zone', 'road_marking')
CANCELLATION = 'cancellation'
VISION_NAMES = ('SIGNNET_WIDTH', 'SIGNNET_HEIGHT', 'SIGNNET_STRIDE', 'CLASSIFIER_SIZE', 'CLASSIFIER_MEAN',
                'CLASSIFIER_STD', 'SignNetInput', 'read_signs', 'class_speed')
BUILD_NEEDED = 'a build is needed: use Publish'
NOT_PREPARED = ("turn on Show Vision Speed Limit, or choose Speed Limit Source Smartspeed, on "
                "the comma: it prepares the networks only then")


class CommaError(ValueError):
    """The comma cannot take the networks now (or the copy failed): why, in plain words."""


def _say(line: str) -> None:
    print(line, flush=True)


# -- the comma over ssh -------------------------------------------------------------------------------------
class Comma:
    """The comma over ssh (BatchMode: root's key, never a password) and its folders."""

    def __init__(self, host: str | None = None):
        self.host = host or os.environ.get(HOST_ENV) or HOST
        self.openpilot = os.environ.get(OPENPILOT_ENV) or OPENPILOT
        self.params = os.environ.get(PARAMS_ENV) or PARAMS
        self.log = os.environ.get(LOG_ENV) or LOG
        self.models = f'{self.openpilot}/{MODELS_DIR}'

    def run(self, command: str, *, timeout: int = SSH_TIMEOUT) -> str:
        try:
            done = subprocess.run(['ssh', *SSH_OPTIONS, self.host, command], capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise CommaError(f'the comma did not answer within {timeout} s') from error
        if done.returncode == 255:
            raise CommaError(f'the comma does not answer over ssh ({self.host}): it answers while the car is on or '
                             f'just parked; {(done.stderr.strip().splitlines() or ["no answer"])[-1]}')
        if done.returncode != 0:
            raise CommaError(f'on the comma: {(done.stderr.strip() or done.stdout.strip() or "failed")[-500:]}')
        return done.stdout

    def _scp(self, source: str, target: str) -> None:
        try:
            done = subprocess.run(['scp', '-q', *SSH_OPTIONS, source, target], capture_output=True, text=True,
                                  timeout=SSH_TIMEOUT, check=False)
        except subprocess.TimeoutExpired as error:
            raise CommaError(f'the copy did not finish within {SSH_TIMEOUT} s') from error
        if done.returncode != 0:
            raise CommaError(f'the copy failed: {(done.stderr.strip() or "scp failed")[-500:]}')

    def get(self, remote: str, local: Path) -> None:
        self._scp(f'{self.host}:{remote}', str(local))

    def put(self, local: Path, remote: str) -> None:
        self._scp(str(local), f'{self.host}:{remote}')


def _q(value: str) -> str:
    return shlex.quote(value)


def _fields(text: str) -> dict[str, str]:
    return {key: value.strip() for key, _, value in (line.partition('=') for line in text.splitlines()) if key}


def inspect(comma: Comma) -> dict:
    """What the comma is and does now, in one ssh call, read-only: driving, the toggle and the source, the build's
    speed-limits files, the version of its networks, where it keeps the compiled ones."""
    package = f'{comma.openpilot}/{PACKAGE}'
    params = comma.params
    script = '\n'.join((
        f'printf "params=%s\\n" "$(test -d {_q(params)} && echo yes)"',
        f'printf "onroad=%s\\n" "$(cat {_q(params + "/IsOnroad")} 2>/dev/null)"',
        f'printf "display=%s\\n" "$(cat {_q(params + "/SpeedLimitVisionDisplay")} 2>/dev/null)"',
        f'printf "policy=%s\\n" "$(cat {_q(params + "/SpeedLimitPolicy")} 2>/dev/null)"',
        f'printf "commit=%s\\n" "$(cut -c1-9 {_q(params + "/GitCommit")} 2>/dev/null)"',
        f'printf "runner=%s\\n" "$(grep -c "^MANIFEST_FORMAT = 2$" {_q(package + "/runner.py")} 2>/dev/null)"',
        f'printf "warmup=%s\\n" "$(test -f {_q(package + "/warmup.py")} && echo yes)"',
        f'printf "vision=%s\\n" "$(test -f {_q(package + "/vision.py")} && echo yes)"',
        f'printf "models=%s\\n" "$(test -e {_q(comma.models + "/.git")} && echo yes)"',
        f'printf "manifest=%s\\n" "$(tr -d "\\n" < {_q(comma.models + "/" + MANIFEST)} 2>/dev/null)"',
        'echo ---',
        f'grep -A 8 "def model_root" {_q(comma.openpilot + "/" + HARDWARE)} 2>/dev/null; true',
    ))
    fields, _, hardware = comma.run(script).partition('\n---\n')
    found = _fields(fields)
    root = None
    # Paths.model_root(): the path it returns on the device (the one not under PC), up to the next function.
    for line in hardware.splitlines()[1:]:
        if re.match(r'\s*def ', line):
            break
        root = root or re.search(r'return "(/[^"]+)"', line)
    try:
        manifest = json.loads(found.get('manifest') or 'null')
    except ValueError:
        manifest = None
    return {'params': found.get('params') == 'yes', 'onroad': found.get('onroad') == '1',
            'display': found.get('display') == '1', 'policy': found.get('policy', ''),
            'commit': found.get('commit', ''), 'runner': found.get('runner', '0') not in ('', '0'),
            'warmup': found.get('warmup') == 'yes', 'vision': found.get('vision') == 'yes',
            'models': found.get('models') == 'yes', 'model_root': root.group(1) if root else None,
            'version': manifest.get('version') if isinstance(manifest, dict) else None}


def check_comma(comma: Comma, *, restore: bool = False) -> dict:
    """The comma reachable, parked, and (unless restore) running a build that prepares and runs these networks;
    raises CommaError with why not."""
    state = inspect(comma)
    build = f' (build {state["commit"]})' if state['commit'] else ''
    if not state['params']:
        raise CommaError(f'no openpilot params on the comma ({comma.params}): is openpilot installed?')
    if state['onroad']:
        raise CommaError('the comma is driving: try again when the car is parked')
    if not restore and not (state['runner'] and state['warmup'] and state['vision']):
        raise CommaError(f'the comma{build} cannot run these networks (no speedvisiond: {PACKAGE}/runner.py with '
                         'MANIFEST_FORMAT = 2, warmup.py and vision.py): install the testing build first '
                         '(Publish), then try again')
    if not state['models']:
        raise CommaError(f'the comma{build} has no speed-limits models folder ({comma.models}): '
                         'install the testing build first (Publish)')
    if restore:
        return state
    if not (state['display'] or state['policy'] in VISION_POLICIES):
        raise CommaError(NOT_PREPARED)
    if state['model_root'] is None:
        raise CommaError(f'where the comma keeps compiled networks is not readable (Paths.model_root in {HARDWARE})')
    return state


# -- the checks with the comma's code -------------------------------------------------------------------------
class Onnx:
    """A network under onnxruntime on the CPU: its inputs and outputs (name, shape), numpy in, numpy out."""

    def __init__(self, path: Path):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.inputs = [(item.name, list(item.shape)) for item in self.session.get_inputs()]
        self.outputs = [(item.name, list(item.shape)) for item in self.session.get_outputs()]

    def __call__(self, x):
        import numpy as np
        return self.session.run(None, {self.inputs[0][0]: np.asarray(x, dtype=np.float32)})[0]


def load_vision(path: Path):
    """The comma's vision.py (numpy only) as a module of its own."""
    spec = importlib.util.spec_from_file_location('comma_speed_limits_vision', path)
    if spec is None or spec.loader is None:
        raise CommaError("the comma's vision.py cannot be read")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # whatever the comma's file raises, it is a reason to refuse
        raise CommaError(f"the comma's vision.py does not load here: {error}; {BUILD_NEEDED}") from error
    return module


def speed_of(name: str) -> float | None:
    """The speed a class gives as the dataset names it: the km/h of a family with a value, 0 for the end of a limit,
    None for the others."""
    if name == CANCELLATION:
        return 0.0
    family, _, value = name.partition(':')
    value = value.partition(':')[0]  # a number with hours: maximum_speed_hours:30:22-6
    return float(value) if family in SPEED_FAMILIES and value.isdigit() else None


def _ints(shape: Any) -> list | None:
    return list(shape) if isinstance(shape, list) else None


def _fits(actual: list, wanted: list) -> bool:
    """An ONNX shape against the one wanted: a dimension that is not a number (dynamic) fits any."""
    return len(actual) == len(wanted) and all(not isinstance(a, int) or a == w for a, w in zip(actual, wanted))


def contract_problems(vision, folder: Path, networks: dict) -> list[str]:
    """Where the export does not fit the comma's code (empty: it fits). networks: the ONNX of each, as Onnx."""
    import numpy as np
    missing = [name for name in VISION_NAMES if not hasattr(vision, name)]
    if missing:
        return [f"the comma's vision.py has no {', '.join(missing)}"]
    finder = json.loads((folder / (FILES['finder'] + '.contract.json')).read_text(encoding='utf-8'))
    reader = json.loads((folder / (FILES['classifier'] + '.contract.json')).read_text(encoding='utf-8'))
    width, height, stride, size = (int(vision.SIGNNET_WIDTH), int(vision.SIGNNET_HEIGHT), int(vision.SIGNNET_STRIDE),
                                   int(vision.CLASSIFIER_SIZE))
    problems = []
    band = [1, 3, height, width]
    if _ints((finder.get('input') or {}).get('shape')) != band or not _fits(networks['finder'].inputs[0][1], band):
        problems.append(f'SignNet takes {(finder.get("input") or {}).get("shape")}, the comma gives it {band}')
    if finder.get('stride') != stride:
        problems.append(f'SignNet has a grid step of {finder.get("stride")} px, the comma decodes {stride} px')
    output = finder.get('output') or {}
    grid = [1, 5, height // stride, width // stride]
    if len(networks['finder'].outputs) != 1:
        problems.append(f'SignNet has {len(networks["finder"].outputs)} outputs, the comma reads one')
    elif _ints(output.get('shape')) != grid or not _fits(networks['finder'].outputs[0][1], grid):
        problems.append(f'SignNet gives {output.get("shape")}, the comma reads {grid} (heat, then 4 shape channels)')
    channels = output.get('channels')
    if channels is not None and (not isinstance(channels, list) or len(channels) != 5 or channels[0] != 'heat'):
        problems.append(f'SignNet gives the channels {channels}, the comma reads heat, then 4 shape channels')
    crop = [1, 3, size, size]
    if _ints((reader.get('input') or {}).get('shape')) != crop or not _fits(networks['classifier'].inputs[0][1], crop):
        problems.append(f'the classifier takes {(reader.get("input") or {}).get("shape")}, the comma gives it {crop}')
    normalization = (reader.get('input') or {}).get('normalization') or {}
    for key, wanted in (('mean', vision.CLASSIFIER_MEAN), ('std', vision.CLASSIFIER_STD)):
        value = normalization.get(key)
        wanted = np.asarray(wanted, dtype=np.float64).reshape(-1)
        if not isinstance(value, list) or len(value) != len(wanted) or not np.allclose(value, wanted, atol=1e-6):
            problems.append(f'the classifier was trained with the {key} {value}, the comma gives it {wanted.tolist()}')
    classes = (reader.get('output') or {}).get('classes')
    logits = _ints((reader.get('output') or {}).get('shape'))
    if not isinstance(classes, list) or not classes or not all(isinstance(name, str) for name in classes):
        problems.append('the classifier names no classes')
        return problems
    if len(networks['classifier'].outputs) != 1 or logits is None or logits[-1] != len(classes) \
            or not _fits(networks['classifier'].outputs[0][1][-1:], [len(classes)]):
        problems.append(f'the classifier gives {logits}, not one score for each of its {len(classes)} classes')
    wrong = []
    for name in classes:
        try:
            read = vision.class_speed(name)
        except Exception:  # noqa: BLE001 - a class the comma's code cannot take is a mismatch
            read = 'an error'
        if read != speed_of(name):
            wrong.append(f'{name} as {read}')
    if wrong:
        problems.append('the comma reads the classes ' + ', '.join(wrong[:5]) + (' ...' if len(wrong) > 5 else ''))
    if not any(speed_of(name) for name in classes):
        problems.append('no class of the classifier gives a speed')
    return problems


def speed_frames(root: Path, count: int = FRAMES) -> list[dict]:
    """Reviewed frames with one confirmed speed, on signs of a usual size, for the reading check: frames of the test
    partition first (the networks never trained on them), reviewed by a person first, every speed sign whole in the
    frame and in the band SignNet reads, the widest 36 to 72 px (a sign at a usual distance: neither far nor right
    beside the car), the nearest to 50 px first, a drive each while there are others.
    [{'image', 'name', 'speeds', 'route'}]."""
    from nnslr_tools.training import ELIGIBLE_REVIEW_STATES
    root = Path(root)
    dataset = root / 'annotations' / 'objects.jsonl'
    try:
        latest = json.loads((root / 'splits' / 'latest.json').read_text(encoding='utf-8'))
        assignments = json.loads((root / latest['path']).read_text(encoding='utf-8'))['assignments']
    except (OSError, ValueError, KeyError, TypeError):
        assignments = {}
    frames: dict[str, dict] = {}
    for line in dataset.read_text(encoding='utf-8').splitlines() if dataset.is_file() else ():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        box, value = row.get('bbox_xyxy'), row.get('value_kph')
        if row.get('review_state') not in ELIGIBLE_REVIEW_STATES or row.get('sign_family') != 'maximum_speed' \
                or not isinstance(box, list) or len(box) != 4:
            continue
        relative = Path(str(row.get('image_path') or ''))
        if relative.is_absolute() or not relative.parts or '..' in relative.parts:
            continue
        width = row.get('width') or 0
        x1, _, x2, y2 = (float(v) for v in box)
        frame = frames.setdefault(str(relative), {
            'image': root / relative, 'name': str(relative), 'speeds': set(), 'route': row.get('route_id'),
            'width': 0.0, 'fits': True, 'test': assignments.get(row.get('annotation_id')) == 'test',
            'person': (row.get('provenance') or {}).get('label_basis') == 'human_review'})
        # Every speed sign of the frame: a number, whole in the frame, in the band (288 rows of it scaled to 896).
        if not isinstance(value, int) or isinstance(value, bool) or not width \
                or y2 > 288 * width / 896 or x1 <= 2 or x2 >= width - 2:
            frame['fits'] = False
            continue
        frame['speeds'].add(float(value))
        frame['width'] = max(frame['width'], x2 - x1)
    usable = [frame for frame in frames.values() if frame['fits'] and len(frame['speeds']) == 1
              and 36 <= frame['width'] <= 72 and frame['image'].is_file()]
    ranked = sorted(usable, key=lambda f: (not f['test'], not f['person'], abs(f['width'] - 50), f['name']))
    chosen, routes = [], set()
    for again in (False, True):
        for frame in ranked:
            if len(chosen) == count:
                break
            if frame in chosen or (not again and frame['route'] in routes):
                continue
            chosen.append(frame)
            routes.add(frame['route'])
    return [{'image': f['image'], 'name': f['name'], 'speeds': sorted(f['speeds']), 'route': f['route']} for f in chosen]


def nv12_of(rgb):
    """The NV12 planes of an RGB frame (H, W, 3 uint8), as the road camera gives them: BT.601 limited range, the
    chroma of each 2 x 2 block; the inverse of how ffmpeg decoded the frames of the dataset."""
    import numpy as np
    pixels = np.asarray(rgb, dtype=np.float64)
    height, width = pixels.shape[0] // 2 * 2, pixels.shape[1] // 2 * 2
    r, g, b = (pixels[:height, :width, channel] for channel in range(3))
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255

    def quarter(plane):
        return plane.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))

    uv = np.empty((height // 2, width), dtype=np.float64)
    uv[:, 0::2], uv[:, 1::2] = quarter(u), quarter(v)
    return (np.clip(np.rint(y), 0, 255).astype(np.uint8), np.clip(np.rint(uv), 0, 255).astype(np.uint8))


def read_frames(vision, folder: Path, frames: list[dict], networks: dict) -> list[dict]:
    """The comma's read_signs on each frame, with both networks: what it reads, and whether the confirmed speed is
    among it."""
    from PIL import Image
    classes = json.loads((folder / (FILES['classifier'] + '.contract.json')).read_text(encoding='utf-8'))['output']['classes']
    readings = []
    for frame in frames:
        with Image.open(frame['image']) as image:
            y, uv = nv12_of(image.convert('RGB'))
        band = vision.SignNetInput(y.shape[1], y.shape[0])
        signs = vision.read_signs(y, uv, band, networks['finder'], networks['classifier'], classes)
        read = sorted({float(sign['speed']) for sign in signs if sign.get('speed') is not None})
        readings.append({'name': frame.get('name') or Path(frame['image']).name, 'speeds': frame['speeds'], 'read': read,
                         'right': any(speed in read for speed in frame['speeds'])})
    return readings


def check_networks(root: Path, vision_path: Path, folder: Path, *, say: Callable[[str], None] = _say,
                   frames: list[dict] | None = None) -> None:
    """Step 3: the contracts against the comma's vision.py, then its read_signs on reviewed frames; raises CommaError
    (nothing is copied then)."""
    vision = load_vision(vision_path)
    networks = {network: Onnx(folder / name) for network, name in FILES.items()}
    problems = contract_problems(vision, folder, networks)
    if problems:
        raise CommaError('these networks do not fit the build on the comma: ' + '; '.join(problems) + f'; {BUILD_NEEDED}')
    say(f'contracts: SignNet {vision.SIGNNET_HEIGHT} x {vision.SIGNNET_WIDTH}, grid {vision.SIGNNET_STRIDE} px; '
        f'classifier {vision.CLASSIFIER_SIZE} x {vision.CLASSIFIER_SIZE}, its classes read as the comma reads them')
    frames = speed_frames(root) if frames is None else frames
    if len(frames) < FRAMES:
        raise CommaError(f'fewer than {FRAMES} reviewed frames with a confirmed speed sign to check them on')
    try:
        readings = read_frames(vision, folder, frames, networks)
    except Exception as error:  # the comma's code failing on them is the reason to refuse
        raise CommaError(f"the comma's read_signs fails with these networks: {type(error).__name__}: {error}; "
                         f'{BUILD_NEEDED}') from error
    for reading in readings:
        say(f'  {reading["name"]}: confirmed {_kph(reading["speeds"])}, read {_kph(reading["read"]) or "nothing"}'
            f'{"" if reading["right"] else " (wrong)"}')
    right = sum(reading['right'] for reading in readings)
    if right < FRAMES_READ:
        raise CommaError(f"the comma's code reads the confirmed speed on {right} of {len(readings)} frames only "
                         f'(at least {FRAMES_READ} wanted): nothing copied')
    say(f"reading: the confirmed speed on {right} of {len(readings)} frames, with the comma's code")


def _kph(values: list[float]) -> str:
    return ', '.join(f'{value:g}' for value in values) + (' km/h' if values else '')


# -- the copy, the wait, the restore --------------------------------------------------------------------------
def compiled_name(stem: str, digest: str) -> str:
    """The comma's compiled network for an ONNX (runner.compiled_path): by the first 16 hex of its sha256."""
    return f'{stem}_{digest[:16]}_tinygrad.pkl'


def copy_files(comma: Comma, folder: Path, version: str, settings: dict | None = None) -> list[str]:
    """Each file to <name>.tmp in the comma's models folder, its sha256 checked there, then all moved into place,
    speed_limits.json last (with the car's settings); the build's brain and applicability network, when it has them,
    stay in use. Returns the files copied."""
    names = [name for network_file in FILES.values() for name in (network_file, network_file + '.contract.json')]
    brain, decider = ('yes' in comma.run('cd ' + _q(comma.models) + f' && test -f {file} && '
                                         f'test -f {file}.contract.json && echo yes || true')
                      for file in (BRAIN_FILE, DECIDER_FILE))
    (folder / MANIFEST).write_text(json.dumps(manifest_of(version, settings, brain, decider), indent=2) + '\n',
                                   encoding='utf-8')
    names.append(MANIFEST)
    try:
        for name in names:
            comma.put(folder / name, f'{comma.models}/{name}.tmp')
        sums = _fields(comma.run('cd ' + _q(comma.models) + ' && sha256sum -- '
                                 + ' '.join(_q(name + '.tmp') for name in names) + " | awk '{print $2\"=\"$1}'"))
        wrong = [name for name in names if sums.get(name + '.tmp') != _sha256(folder / name)]
        if wrong:
            raise CommaError('the copy arrived wrong on the comma: ' + ', '.join(wrong))
        comma.run('cd ' + _q(comma.models) + ' && ' + ' && '.join(f'mv -f -- {_q(name + ".tmp")} {_q(name)}'
                                                                   for name in names))
    except CommaError:
        _remove_partial(comma)
        raise
    return names


def _remove_partial(comma: Comma) -> None:
    names = [name for network_file in FILES.values() for name in (network_file, network_file + '.contract.json')]
    try:
        comma.run('cd ' + _q(comma.models) + ' && rm -f -- ' + ' '.join(_q(name + '.tmp') for name in [*names, MANIFEST]))
    except CommaError:
        pass


def restore(comma: Comma) -> str:
    """The build's networks back on the comma: git checkout -- . in its models folder, the .tmp of a copy removed.
    Returns the version of the networks there now."""
    names = [name for network_file in FILES.values() for name in (network_file, network_file + '.contract.json')]
    comma.run('cd ' + _q(comma.models) + ' && git checkout -- . && rm -f -- '
              + ' '.join(_q(name + '.tmp') for name in [*names, MANIFEST]))
    return inspect(comma)['version'] or 'unknown'


def warmup_lines(comma: Comma, limit: int = 12) -> list[str]:
    """The last lines of the comma's log about the warmup (what it compiled, or why not)."""
    try:
        text = comma.run('cd ' + _q(comma.log) + ' 2>/dev/null && ls -t | grep "^swaglog" | head -n 20 '
                         '| xargs -r grep -h -a "speed limits warmup" 2>/dev/null | tail -n ' + str(limit) + '; true')
    except CommaError as error:
        return [str(error)]
    return [_log_line(line) for line in text.splitlines() if line.strip()]


def _log_line(line: str) -> str:
    """A line of the comma's swaglog (JSON: ctx, created, then msg$s for a text, msg for an event) as its UTC time
    and its message."""
    try:
        entry = json.loads(line)
    except ValueError:
        entry = None
    key = next((key for key in ('msg$s', 'msg') if isinstance(entry, dict) and key in entry), None)
    if key is None:
        return line[:600]
    message = entry[key] if isinstance(entry[key], str) else json.dumps(entry[key])
    created = entry.get('created')
    when = (datetime.fromtimestamp(created, UTC).strftime('%Y-%m-%d %H:%M:%S UTC ')
            if isinstance(created, (int, float)) else '')
    return (when + message)[:600]


def wait_compiled(comma: Comma, model_root: str, digests: dict[str, str], *, poll: float = POLL_SECONDS,
                  wait: float = WAIT_SECONDS, say: Callable[[str], None] = _say,
                  sleep: Callable[[float], None] = time.sleep) -> str:
    """Reads, every poll seconds for up to wait seconds, whether the comma has compiled both networks: 'ready',
    'onroad' (it started a drive: speedvisiond compiles them then) or 'timeout'."""
    files = {stem: f'{model_root}/speed_limits/{compiled_name(stem, digest)}' for stem, digest in digests.items()}
    checks = ' '.join(f'$(test -f {_q(path)} && echo {stem})' for stem, path in files.items())
    started = time.monotonic()
    seen: set[str] = set()
    while True:
        found = _fields(comma.run(f'printf "ready=%s\\n" "{checks}"; printf "onroad=%s\\n" '
                                  f'"$(cat {_q(comma.params + "/IsOnroad")} 2>/dev/null)"'))
        ready = set(found.get('ready', '').split())
        for stem in sorted(ready - seen):
            say(f'compiled on the comma: {stem} ({Path(files[stem]).name}, after {time.monotonic() - started:.0f} s)')
        seen |= ready
        if ready >= set(files):
            return 'ready'
        if found.get('onroad') == '1':
            return 'onroad'
        elapsed = time.monotonic() - started
        if elapsed >= wait:
            return 'timeout'
        if int(elapsed // 60) > int((elapsed - poll) // 60) and elapsed >= 60:
            say(f'waiting: {elapsed:.0f} s, {len(seen)} of {len(files)} compiled')
        sleep(poll)


# -- the record -------------------------------------------------------------------------------------------------
def read_trial(root: Path) -> dict | None:
    """The networks being tried on the comma (models/on-comma-trial.json), None when none."""
    path = Path(root) / TRIAL_FILE
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get('signnet'), dict):
        return None
    keep = ('version', 'state', 'copied_at', 'ready_at', 'comma_build')
    return {'signnet': {key: value['signnet'].get(key) for key in ('id', 'run', 'onnx_sha256')},
            'classifier': {key: (value.get('classifier') or {}).get(key) for key in ('id', 'run', 'onnx_sha256')},
            **{key: value.get(key) for key in keep}}


def write_trial(root: Path, record: dict) -> None:
    path = Path(root) / TRIAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    partial.replace(path)


def clear_trial(root: Path) -> bool:
    path = Path(root) / TRIAL_FILE
    if path.exists() or path.is_symlink():
        path.unlink()
        return True
    return False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')


# -- the job ----------------------------------------------------------------------------------------------------
def try_on_comma(root: Path, signnet: Any, *, classifier: Any = None, comma: Comma | None = None,
                 today: date | None = None, exporter: Callable[[dict, Path], dict] | None = None,
                 frames: list[dict] | None = None, poll: float = POLL_SECONDS, wait: float = WAIT_SECONDS,
                 sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = _say,
                 dry_run: bool = False) -> dict:
    """Steps 1 to 5 (see the module). Raises CommaError (or ValueError for a wrong choice) with why it stopped.
    dry_run: steps 1 to 3 only, the comma only read: nothing copied, nothing recorded."""
    root = Path(root)
    comma = comma or Comma()
    # The trial copies the two sign networks alone: the comma keeps its build's brain, so the brain in use is neither
    # refused (one that takes the driver's inputs, Publish's rule) nor exported (its check along a drive needs a GPU,
    # which this job leaves to the trainings).
    chosen = {**choose(root, signnet, classifier, brain_on_car=False), 'brain': None}
    say('== 1/5 Check the comma')
    state = check_comma(comma)
    why = 'Show Vision Speed Limit is on' if state['display'] else f'Speed Limit Source is {VISION_POLICIES[state["policy"]]}'
    say(f'the comma{" (build " + state["commit"] + ")" if state["commit"] else ""} is parked, runs speed-limits '
        f'{state["version"] or "networks"} and prepares new ones ({why})')
    work = root / 'tmp'
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='try-on-comma-', dir=work) as name:
        folder = Path(name)
        vision_path = folder / 'comma-vision.py'
        comma.get(f'{comma.openpilot}/{PACKAGE}/vision.py', vision_path)
        say(f'== 2/5 Export the SignNet of {chosen["finder"]["run"]} and the classifier {chosen["classifier"]["id"]}')
        exported = (exporter or export)(chosen, folder)
        for network in FILES:
            say(f'{network}: {exported[network]["id"]} onnx sha256 {exported[network]["onnx_sha256"]}')
        say("== 3/5 Check them with the comma's code")
        check_networks(root, vision_path, folder, say=say, frames=frames)
        if dry_run:
            say('dry run: they fit the comma; nothing copied, nothing recorded')
            return {'outcome': 'dry_run', 'record': None}
        say('== 4/5 Copy them onto the comma')
        version = f'{(today or date.today()).isoformat()}-try'
        copy_files(comma, folder, version, car_settings.values(root))
    record = {'signnet': {**{key: chosen['finder'][key] for key in ('id', 'run')},
                          'onnx_sha256': exported['finder']['onnx_sha256']},
              'classifier': {**{key: chosen['classifier'][key] for key in ('id', 'run')},
                             'onnx_sha256': exported['classifier']['onnx_sha256']},
              'version': version, 'state': 'preparing', 'copied_at': _now(), 'ready_at': None,
              'comma_build': state['commit'] or None}
    write_trial(root, record)
    say(f'copied: speed-limits {version} in {comma.models} (it was {state["version"] or "unknown"}); '
        f'recorded in {TRIAL_FILE}')
    say('== 5/5 Wait for the comma to prepare them')
    digests = {Path(FILES[network]).stem: exported[network]['onnx_sha256'] for network in FILES}
    outcome = wait_compiled(comma, state['model_root'], digests, poll=poll, wait=wait, say=say, sleep=sleep)
    if outcome == 'ready':
        write_trial(root, {**record, 'state': 'ready', 'ready_at': _now()})
        say('ready: used from the next drive (until the comma\'s next software update)')
        return {'outcome': outcome, 'record': record}
    if outcome == 'onroad':
        raise CommaError('the comma started a drive before preparing them: speedvisiond prepares them at its start '
                         "(no reading for a couple of minutes); \"Back to the build's networks\" puts the build's "
                         'back once parked')
    lines = warmup_lines(comma)
    back = restore(comma)
    clear_trial(root)
    for line in lines:
        say('comma log: ' + line)
    raise CommaError(f'the comma did not prepare them within {wait / 60:g} minutes'
                     f'{"" if lines else " (its log has no line of the warmup)"}: the build\'s networks are back '
                     f'(speed-limits {back})')


def back_to_build(root: Path, *, comma: Comma | None = None, say: Callable[[str], None] = _say) -> str:
    """--restore: the build's networks back on the comma, the trial record cleared."""
    comma = comma or Comma()
    say('== 1/2 Check the comma')
    state = check_comma(comma, restore=True)
    say(f'the comma is parked; its networks now: speed-limits {state["version"] or "unknown"}')
    say("== 2/2 Put back the build's networks")
    version = restore(comma)
    cleared = clear_trial(root)
    say(f"back: the comma has the build's networks again (speed-limits {version})"
        + (f'; {TRIAL_FILE} cleared' if cleared else ''))
    return version


# -- for the panel ----------------------------------------------------------------------------------------------
def trial_missing(repo: Path) -> str:
    """Why "Try on the comma now" cannot run on this server ('' when it can)."""
    repo = Path(repo)
    if not (repo / TRY_SCRIPT).is_file():
        return TRY_SCRIPT + ' is missing in this installation'
    if not (Path(os.environ.get(GPU_VENV_ENV) or repo / '.venv-gpu') / 'bin' / 'nnslr').is_file():
        return 'the GPU environment (.venv-gpu) is missing: it exports the networks'
    if shutil.which('ssh') is None or shutil.which('scp') is None:
        return 'ssh is missing on this server'
    return ''


class CommaProbe:
    """Whether the comma answers on its ssh port, known without making a page wait: a TCP connect, in a thread, at
    most once a minute (ssh -G gives the address of the host, without connecting). None until known."""

    def __init__(self, host: str | None = None, ttl: float = 60.0, timeout: float = 2.0):
        self.host = host
        self.ttl, self.timeout = ttl, timeout
        self.lock = threading.Lock()
        self.answer: bool | None = None
        self.checked = -math.inf
        self.thread: threading.Thread | None = None

    def _target(self) -> tuple[str, int] | None:
        host = self.host or os.environ.get(HOST_ENV) or HOST
        try:
            done = subprocess.run(['ssh', '-G', host], capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        found = dict(line.split(' ', 1) for line in done.stdout.splitlines() if ' ' in line)
        try:
            return found['hostname'], int(found.get('port', '22'))
        except (KeyError, ValueError):
            return None

    def _check(self) -> None:
        answer = False
        target = self._target()
        if target is not None:
            try:
                with socket.create_connection(target, timeout=self.timeout):
                    answer = True
            except OSError:
                answer = False
        with self.lock:
            self.answer, self.checked, self.thread = answer, time.monotonic(), None

    def state(self) -> bool | None:
        with self.lock:
            if time.monotonic() - self.checked > self.ttl and self.thread is None:
                self.thread = threading.Thread(target=self._check, name='comma-probe', daemon=True)
                self.thread.start()
            return self.answer

