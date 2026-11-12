"""Models › Drive: a recorded drive replayed through the car's own vision loop, to see on its video what the comma's
screen would show with a set of networks, and to keep the drives where that was right as the reference of the next sets.

What it is for. Before a set of networks goes on the car he opens a drive, presses play and sees the two squares of
the comma's screen along it: the speed to hold (speedVisionSP.speedLimit) and the last sign read (rawSign*, the
distance to it, the pulse of signRead). A drive where all of it is right he certifies; a new set is then replayed on
the certified drives and compared with what was certified, frame by frame: the regression test.

The same code as the car. The body of speedvisiond's loop is pipeline.SpeedVision of the car's package (the sunnypilot
clone, NNSLR_SUNNYPILOT_CLONE, else /root/sunnypilot: openpilot/sunnypilot/speed_limits). replay() gives it what
speedvisiond receives, in the same order:

- the frames of every minute's fcamera.hevc, decoded by the GPU (NVDEC) to NV12, the camera's pixel format; a drive at
  full definition reduced to the low frame the car's networks read (media.REDUCE_FILTER);
- each frame at the time speedvisiond got it: the logMonoTime of its narrowRoadCameraState (the frame of the video
  through narrowRoadEncodeIdx), about 42 ms after its exposure started;
- before each frame, the car's own messages logged up to that time: every carState, selfdriveState and deviceMotion,
  the last radarState; and before a frame it reads (pipeline.PERIOD after the last one read) the last GPS message;
- SignNet, the classifier, the applicability network and the brain of the set, their ONNX under onnxruntime on the GPU.

Not the car bit for bit: the video comes after the comma's encoder, the car reads the camera's own frame with tinygrad
on its GPU. A confidence right at a threshold can fall on the other side, and the frames read are not the very ones the
car read that day (it reads the first frame 0.2 s after the last one, whenever that came). ALWAYS ON THE GPU: without a
card the replay stops and says so, it never moves to the CPU (device 'cpu' is for the tests alone). THE MAP IS NEVER
READ: the vision limit comes from the camera and the car's own signals.

Two sets (SETS): 'car', the release in the sunnypilot clone (models/speed_limits.json: what the comma downloads, as
models/on-car.json records it), and 'in-use', the versions in use on Models exported as Publish would (export_set, in
the GPU environment for PyTorch: derived/simulation/sets/in-use/, made again when a version in use changes).

A result, derived/simulation/<route>/<set>.json.gz, is the whole reading of the drive, a row per frame read (5 a second):

  {version, route, set, name, made_at, device, frame [width, height], networks [{role, file, id, sha256}],
   code {file: sha256}, settings {...}, videos [[segment, size, mtime]], brain_memory_gap (s, None without a brain),
   brain_groups [[name, size]] (the groups of its input row, in order; None without a brain),
   sources {video [[segment, sha256 of its fcamera.hevc]], logs [[segment, kind, sha256 of the log read]]},
   frames {t_ns, segment, frame},                    which frame of which minute's video, and its time on the log's clock
   out {speedLimit, speedLimitAhead, ..., rawSignSeen},   speedVisionSP as the car would send it, a column a field
   signs {"<row>": [[x1, y1, x2, y2, score, class, confidence, road], ...]},   SignNet's boxes of the rows with any
   in {v_ego, steering_deg, left, right, yaw_rate, gas, brake, enabled, yaw_device, lead_m, lead_kph, country, unix_s,
       brain {"<row>": [the brain's input row at that step]}},     what went in: the car's signals at that frame
   brain_type {"<row>": [road type, its probability]},   the road type the brain's type_head names at each of its steps,
   brain_type_head}                                      on the car's own memory; with the checkpoint_sha256 of its head

Certifying a drive (certify) keeps its result as it is in simulation/certified/<route>.json.gz of the data root, with
who and when in simulation/certified.json: a test case complete in itself, what went in and what came out. compare()
puts another result of the same drive next to it, by frame: where the speed to hold differs, where the sign read
differs, and whether what went into the brain was the same, on the groups both brains take, picture included (then
it is the brain that changed) or not (the reader or the applicability network before it, or another video or log). The pictures stay in the drive's video, named by minute and frame.

The page reads view(): the columns as the step series of the drive player (seconds from the start of the logs, the
names of drive_extract's speedVisionSP series), so the squares of a replay and those the comma logged that day are
drawn by the same code.
"""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

VERSION = 2
SETS = {'car': 'On the car', 'in-use': 'In use'}
RESULTS_RELATIVE = Path('derived') / 'simulation'
SETS_RELATIVE = RESULTS_RELATIVE / 'sets'
CERTIFIED_RELATIVE = Path('simulation') / 'certified'
# The road type head of a set's brain (W, b and the names of its types), by the brain's checkpoint_sha256: type_head is
# for the training only, so no release has it, and the replay applies it on the car brain's memory (save_type_head).
TYPE_HEADS_RELATIVE = RESULTS_RELATIVE / 'type-heads'
CERTIFIED_INDEX = Path('simulation') / 'certified.json'
SUNNYPILOT_ENV, SUNNYPILOT_DEFAULT = 'NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'
PACKAGE_RELATIVE = Path('openpilot') / 'sunnypilot' / 'speed_limits'
MANIFEST = 'speed_limits.json'
MANIFEST_FORMAT = 2
SOURCE_FILE = 'source.json'  # of an exported set: the versions it was exported from
# The car's code a result depends on: another one of these and the result is of another loop.
CODE_FILES = ('pipeline.py', 'vision.py', 'applicability.py', 'brain.py', 'brain_inputs.py', 'country.py')
ROLES = {'finder': 'SignNet', 'classifier': 'Classifier', 'decider': 'Applicability', 'brain': 'Brain'}
# speedVisionSP's fields (cereal/custom.capnp) with the value of a field the car leaves unset.
FIELDS = {'speedLimit': 0.0, 'speedLimitAhead': 0.0, 'speedLimitAheadDistance': 0.0, 'signRead': False, 'roadType': '',
          'speedLimitState': '', 'speedLimitReason': '', 'rawSignClass': '', 'rawSignSpeed': 0.0,
          'rawSignConfidence': 0.0, 'rawSignTaken': False, 'rawSignValue': 0.0, 'rawSignUnit': '', 'rawSignCountry': '',
          'rawSignRoad': '', 'signDistance': 0.0, 'signSize': 0.0, 'rawSignSeen': False}
# The car's signals kept with every frame read: what went in.
INPUTS = ('v_ego', 'steering_deg', 'left', 'right', 'yaw_rate', 'gas', 'brake', 'enabled', 'yaw_device', 'lead_m',
          'lead_kph', 'country', 'unix_s')
# The series of the page (view), named as drive_extract names speedVisionSP's: (series, field, scale, decimals).
SERIES = (('vis_limit', 'speedLimit', 3.6, 2), ('vis_ahead', 'speedLimitAhead', 3.6, 2),
          ('vis_ahead_dist', 'speedLimitAheadDistance', 1, 1), ('vis_road', 'roadType', None, None),
          ('vis_state', 'speedLimitState', None, None), ('vis_reason', 'speedLimitReason', None, None),
          ('vis_raw_class', 'rawSignClass', None, None), ('vis_raw_speed', 'rawSignSpeed', 1, 2),
          ('vis_raw_value', 'rawSignValue', 1, 2), ('vis_raw_unit', 'rawSignUnit', None, None),
          ('vis_raw_conf', 'rawSignConfidence', 1, 2), ('vis_raw_taken', 'rawSignTaken', None, None),
          ('vis_raw_seen', 'rawSignSeen', None, None), ('vis_raw_road', 'rawSignRoad', None, None),
          ('vis_sign_dist', 'signDistance', 1, 4), ('vis_sign_size', 'signSize', 1, 1))
META_SUFFIX = '.meta.json'  # beside a result: what the lists show without reading its rows
REGRESSION_SUFFIX = '.regression.json'  # beside a result: how it stands next to the certified one (compare's counts)
# What a result's meta keeps of it (the rows aside); COMPARE_VERSION: of compare()'s counts kept beside a result, made
# again when compare() looks at more.
META_FIELDS = ('version', 'route', 'set', 'name', 'made_at', 'device', 'networks', 'code', 'settings', 'brain_type_head')
COMPARE_VERSION = 3
# The messages of the log a replay gives the car's loop, and those that say when each frame of the video came.
GPS_SERVICES = ('gpsLocationExternal', 'gpsLocation')
LOG_WORKERS = 8
# narrowRoadCameraState comes this long after the frame's end of exposure (measured: 30.8 to 32 ms): the time of a frame
# whose camera state the log does not have.
CAMERA_STATE_LAG_NS = 31_000_000
PIPE_BYTES = 1 << 20
HOLE_SECONDS = 3.0  # longer between two frames read: the series end there (a minute without its video or its log)
# Two speeds to hold are another speed from half a km/h; two inputs of the brain another row from this much.
KPH_TOLERANCE = 0.5
BRAIN_TOLERANCE = 1e-3


class SimulationError(Exception):
    """A request the replay cannot serve, as the page's API says it."""
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status, self.reason, self.message = status, reason, message


# -- where things are ----------------------------------------------------------------------------------------------------
def package_path(sunnypilot: Path | None = None) -> Path:
    """The car's package in the sunnypilot clone."""
    return Path(sunnypilot or os.environ.get(SUNNYPILOT_ENV) or SUNNYPILOT_DEFAULT) / PACKAGE_RELATIVE


def set_directory(root: Path, set_key: str, package: Path | None = None) -> Path:
    """The release of a set: the clone's models for 'car', the export of the versions in use for 'in-use'."""
    if set_key not in SETS:
        raise SimulationError(400, 'bad_set', f'No such set of networks: {set_key}.')
    return (package or package_path()) / 'models' if set_key == 'car' else Path(root) / SETS_RELATIVE / set_key


def result_path(root: Path, route: str, set_key: str) -> Path:
    return Path(root) / RESULTS_RELATIVE / route / f'{set_key}.json.gz'


def certified_path(root: Path, route: str) -> Path:
    return Path(root) / CERTIFIED_RELATIVE / f'{route}.json.gz'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_gz(path: Path) -> dict | None:
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            found = json.load(handle)
    except (OSError, ValueError, EOFError, zlib.error):
        return None
    return found if isinstance(found, dict) and found.get('version') == VERSION else None


def _write_atomic(path: Path, write) -> None:
    """A file written whole: a temporary one beside it, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dump_gz(path: Path, record: dict) -> None:
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as handle:
        json.dump(record, handle, separators=(',', ':'), allow_nan=False)


def write_result(root: Path, record: dict) -> Path:
    path = result_path(root, record['route'], record['set'])
    _write_atomic(path, lambda temporary: _dump_gz(temporary, record))
    meta = {name: record.get(name) for name in META_FIELDS}
    meta['frames'] = len(record['frames']['t_ns'])
    _write_atomic(path.with_name(path.name + META_SUFFIX), lambda temporary: temporary.write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8'))
    return path


def result_meta(root: Path, route: str, set_key: str) -> dict | None:
    """What a result is, without its rows: {name, made_at, device, networks, code, settings, frames}; None without one."""
    path = result_path(root, route, set_key)
    if not path.is_file():
        return None
    meta_path = path.with_name(path.name + META_SUFFIX)
    try:
        found = json.loads(meta_path.read_text(encoding='utf-8'))
        if isinstance(found, dict) and found.get('version') == VERSION and all(name in found for name in META_FIELDS):
            return found
    except (OSError, ValueError):
        pass
    record = _read_gz(path)  # a result without its meta (or an older one): read once, its meta written again
    if record is None:
        return None
    meta = {**{name: record.get(name) for name in META_FIELDS}, 'frames': len(record['frames']['t_ns'])}
    try:
        _write_atomic(meta_path, lambda temporary: temporary.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8'))
    except OSError:
        pass
    return meta


def read(root: Path, route: str, set_key: str) -> dict | None:
    """The result of a drive with a set, None when it was never replayed (or by an older format)."""
    return _read_gz(result_path(root, route, set_key)) if set_key in SETS else None


# -- the sets of networks ------------------------------------------------------------------------------------------------
def code_hashes(package: Path | None = None) -> dict[str, str]:
    """The car's code a replay runs, by file: its SHA-256, '' for a file the clone does not have."""
    package = package or package_path()
    return {name: _sha256(package / name) if (package / name).is_file() else '' for name in CODE_FILES}


def release_networks(directory: Path) -> list[dict] | None:
    """The networks of a release's directory as its manifest names them: [{role, file, id, sha256}], the id being the
    version of the Models tab its contract names; None without a release of today's format."""
    try:
        manifest = json.loads((directory / MANIFEST).read_text(encoding='utf-8'))
        if manifest.get('format') != MANIFEST_FORMAT:
            return None
        found = []
        for role in ROLES:
            if role not in manifest:
                continue
            contract = json.loads((directory / manifest[role]['contract']).read_text(encoding='utf-8'))
            found.append({'role': role, 'file': manifest[role]['file'], 'id': str(contract.get('checkpoint') or ''),
                          'sha256': contract.get('onnx_sha256') or _sha256(directory / manifest[role]['file'])})
        return found
    except (OSError, ValueError, KeyError, TypeError):
        return None


def release_settings(directory: Path) -> dict:
    """The car's settings of a release, as pipeline.Networks reads them from speed_limits.json: {} without any."""
    try:
        found = json.loads((directory / MANIFEST).read_text(encoding='utf-8')).get('settings')
    except (OSError, ValueError, AttributeError):
        return {}
    return found if isinstance(found, dict) else {}


def wanted_settings(root: Path, networks: list[dict]) -> dict:
    """The settings an export of the versions in use carries today: the car's settings of Models (car_settings), as
    Publish writes them in speed_limits.json for these networks."""
    from nnslr_tools import car_release, car_settings

    roles = {one['role'] for one in networks}
    return car_release.manifest_of('in-use', car_settings.values(root), 'brain' in roles, 'decider' in roles).get(
        'settings', {})


def release_name(directory: Path) -> str:
    try:
        return f"speed-limits {json.loads((directory / MANIFEST).read_text(encoding='utf-8'))['version']}"
    except (OSError, ValueError, KeyError, TypeError):
        return ''


def brain_contract(directory: Path) -> dict | None:
    """The contract of a release's brain; None without one."""
    try:
        manifest = json.loads((directory / MANIFEST).read_text(encoding='utf-8'))
        found = json.loads((directory / manifest['brain']['contract']).read_text(encoding='utf-8'))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return found if isinstance(found, dict) else None


def type_head_path(root: Path, contract: dict | None) -> Path | None:
    """Where the road type head of a brain is kept, by its checkpoint_sha256; None for a contract without one."""
    sha = (contract or {}).get('checkpoint_sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any(digit not in '0123456789abcdef' for digit in sha):
        return None
    return Path(root) / TYPE_HEADS_RELATIVE / f'{sha}.npz'


def load_type_head(root: Path, contract: dict | None) -> dict | None:
    """The road type head of a set's brain, {weight (types x memory), bias, names, sha256}; None while save_type_head
    has not written it or for a brain that learnt no road type."""
    import numpy as np

    path = type_head_path(root, contract)
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path) as found:
            weight, bias, names = found['weight'].astype(np.float64), found['bias'].astype(np.float64), found['names']
    except (OSError, ValueError, KeyError):
        return None
    names = [str(name) for name in names.tolist()]
    if not names or weight.ndim != 2 or weight.shape[0] != len(names) or bias.shape != (len(names),):
        return None
    return {'weight': weight, 'bias': bias, 'names': names, 'sha256': contract['checkpoint_sha256']}


def save_type_head(root: Path, set_key: str, package: Path | None = None, log=print) -> Path | None:
    """The road type head of a set's brain written beside the replays, from its run's weights (brain-last.pt, the one
    the contract's checkpoint_sha256 names): type_head is for the training only (brain_net.TRAINING_ONLY), so the ONNX
    of a brain exported before 12 Nov 2026 has it not, and the replay applies it on the memory of the car's brain at each
    of its steps. In the GPU environment, whose PyTorch reads the weights; nothing for a set without a brain or a brain that learnt no road type."""
    import numpy as np

    from nnslr_tools.model_roles import BRAIN_WEIGHTS
    from nnslr_tools.runs import RUN_NAME

    contract = brain_contract(set_directory(root, set_key, package))
    path = type_head_path(root, contract)
    if contract is None or path is None:
        log('no brain in this set: no road type of the brain')
        return None
    if path.is_file():
        log('the road type of the brain: its head is ready')
        return path
    version = str(contract.get('checkpoint') or '')
    run = version.split(':', 1)[-1].split('/', 1)[0]
    checkpoint = Path(root) / 'runs' / run / BRAIN_WEIGHTS
    if not RUN_NAME.fullmatch(run) or not checkpoint.is_file():
        log(f'the brain {version} has no weights on this server: no road type of the brain')
        return None
    if _sha256(checkpoint) != contract['checkpoint_sha256']:
        log(f'the weights of the brain {version} changed since its export: no road type of the brain')
        return None
    try:
        import torch
    except ImportError:
        log('no PyTorch in this environment (.venv-gpu reads the weights): no road type of the brain')
        return None
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights, names = saved.get('weights') or {}, [str(name) for name in saved.get('road_types') or []]
    if 'type_head.weight' not in weights or not names:
        log(f'the brain {version} learnt no road type (Road type off): no road type of the brain')
        return None

    def write(temporary: Path) -> None:
        with temporary.open('wb') as handle:
            np.savez(handle, weight=weights['type_head.weight'].float().numpy(),
                     bias=weights['type_head.bias'].float().numpy(), names=np.array(names))

    _write_atomic(path, write)
    log(f'the road type of the brain: its head written ({len(names)} types of {version})')
    return path


def set_states(root: Path, in_use: dict[str, str | None] | None = None, package: Path | None = None) -> dict[str, dict]:
    """Each set as the page shows it: {label, name, networks, settings, ready, reason}. in_use: the versions in use on
    Models by role (cutter, classifier, decider, brain), to say whether the export of 'in-use' is still theirs, and of
    the car's settings of today; None: not asked."""
    states = {}
    for key, label in SETS.items():
        directory = set_directory(root, key, package)
        networks = release_networks(directory)
        head = type_head_path(root, brain_contract(directory))
        state = {'label': label, 'name': release_name(directory), 'networks': networks or [],
                 'settings': release_settings(directory), 'ready': networks is not None,
                 'reason': '' if networks is not None else 'No release of the networks in the sunnypilot clone.',
                 # the checkpoint_sha256 of its brain's road type head, once save_type_head wrote it
                 'type_head': head.stem if head is not None and head.is_file() else None}
        if key == 'in-use':
            wanted = {role: version for role, version in (in_use or {}).items() if version}
            exported = exported_source(root)
            state['wanted'] = wanted
            if networks is None:
                state['reason'] = 'Not exported yet: Compute exports the versions in use first.'
            elif in_use is not None and exported != wanted:
                state['ready'] = False
                state['reason'] = 'The versions in use changed: Compute exports them again first.'
            elif in_use is not None and state['settings'] != wanted_settings(root, networks):
                state['ready'] = False
                state['reason'] = "The car's settings changed: Compute exports them again first."
        states[key] = state
    return states


def exported_source(root: Path) -> dict[str, str]:
    """The versions the export of 'in-use' was made from, by role; {} without one."""
    try:
        found = json.loads((Path(root) / SETS_RELATIVE / 'in-use' / SOURCE_FILE).read_text(encoding='utf-8'))
        return {str(role): str(version) for role, version in found.items()} if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def is_current(record: dict, networks: list[dict] | None, code: dict[str, str], settings: dict | None = None,
               type_head: str | None = None) -> bool:
    """Whether a result is the one a replay of today would give: the same networks, the same code of the car and, when
    given, the same settings of the release (a set's 'settings') and the road type of its brain (a set's 'type_head')."""
    mine = {(one['role'], one['sha256']) for one in record.get('networks') or []}
    return networks is not None and mine == {(one['role'], one['sha256']) for one in networks} and \
        record.get('code') == code and (settings is None or (record.get('settings') or {}) == settings) and \
        (type_head is None or record.get('brain_type_head') == type_head)



def changes(record: dict, state: dict, code: dict[str, str]) -> list[dict]:
    """What a result was made with that is not today's, for the page to say why it is out of date: [{what, was, now}],
    what being a role of ROLES (was/now its versions: 'version unknown' for a version not kept, '' for a network not
    there), 'code', 'settings' or 'type_head'. Today's networks are those of the set when it is ready, else the
    versions in use it waits to export (state 'wanted'); with neither, the networks are not compared."""
    mine = {one['role']: one for one in record.get('networks') or []}
    today = {one['role']: one for one in state.get('networks') or []} if state.get('ready') else {}
    wanted = {} if today else state.get('wanted') or {}
    found = []
    for role in ROLES:
        was, now = mine.get(role), today.get(role)
        if today:
            if (was or {}).get('sha256') != (now or {}).get('sha256'):
                found.append({'what': role, 'was': (was or {}).get('id', ''), 'now': (now or {}).get('id', '')})
        elif wanted and (was or {}).get('id', '') != wanted.get(role, ''):
            found.append({'what': role, 'was': (was or {}).get('id', ''), 'now': wanted.get(role, '')})
    if record.get('code') != code:
        found.append({'what': 'code', 'was': '', 'now': ''})
    if state.get('ready') and (record.get('settings') or {}) != (state.get('settings') or {}):
        found.append({'what': 'settings', 'was': '', 'now': ''})
    if state.get('ready') and state.get('type_head') is not None and record.get('brain_type_head') != state['type_head']:
        found.append({'what': 'type_head', 'was': '', 'now': ''})
    return found


# -- the page ------------------------------------------------------------------------------------------------------------
def _seconds(t_ns: int, t0_ns: int) -> float:
    return round((t_ns - t0_ns) / 1e9, 3)


def _steps(times: list[float], values: list, gap: float = HOLE_SECONDS) -> list[list]:
    """A column as the step series of the drive player: [seconds, value] where the value changes, None where the frames
    stop for longer than gap."""
    series: list[list] = []
    last_time = None
    for at, value in zip(times, values, strict=True):
        if last_time is not None and at - last_time > gap:
            series.append([round(last_time + gap, 3), None])
        if not series or series[-1][1] != value:
            series.append([at, value])
        last_time = at
    return series


def view(record: dict, t0_ns: int) -> dict:
    """A result as the page draws it: the series of the squares (seconds from t0_ns, the start of the drive's logs, as
    the timeline's), the readings that pulse (events, as the timeline's 'sign') and SignNet's boxes by frame read:
    [time, boxes, minute, frame k of the minute]."""
    times = [_seconds(t_ns, t0_ns) for t_ns in record['frames']['t_ns']]
    out = record['out']
    series = {}
    for name, field, scale, decimals in SERIES:
        column = out[field]
        series[name] = _steps(times, [round(value * scale, decimals) for value in column] if scale else column)
    kph = series_values(out['speedLimit'])
    events = [[times[row], 'sign', kph[row]] for row, read in enumerate(out['signRead']) if read]
    # Each frame read with its minute and its frame k of that minute's fcamera.hevc: the page draws its boxes on that very
    # frame of the video, not on the frame of a time (a sign close by moves 25-30 px a frame).
    segments = record['frames'].get('segment') or [None] * len(times)
    indexes = record['frames'].get('frame') or [None] * len(times)
    signs = [[times[int(row)], [[*box[:4], box[5], box[6], box[7]] for box in boxes], segments[int(row)], indexes[int(row)]]
             for row, boxes in sorted(record.get('signs', {}).items(), key=lambda item: int(item[0]))]
    if record.get('brain_type'):
        # The road type the brain names (its type_head, for the training only: the car does not show it), held from
        # one of its steps to the next, as its speed to hold.
        kinds, chances, held = [], [], (None, None)
        for row in range(len(times)):
            found = record['brain_type'].get(str(row))
            if found:
                held = (found[0], round(found[1], 2))
            kinds.append(held[0])
            chances.append(held[1])
        series['brain_type'], series['brain_type_p'] = _steps(times, kinds), _steps(times, chances)
    return {'route': record['route'], 'set': record['set'], 'label': SETS.get(record['set'], record['set']),
            'name': record.get('name', ''), 'made_at': record.get('made_at'), 'networks': record.get('networks', []),
            'frame': record.get('frame'), 'frames': len(times), 'series': series, 'events': events, 'signs': signs}


def series_values(column: list[float]) -> list[float]:
    """A column of m/s in km/h, as the screen rounds it."""
    return [round(value * 3.6, 2) for value in column]


# -- certified drives and the regression test ----------------------------------------------------------------------------
def certified(root: Path) -> dict[str, dict]:
    """The certified drives: {route: {at, who, set, name, networks, frames}}."""
    try:
        found = json.loads((Path(root) / CERTIFIED_INDEX).read_text(encoding='utf-8')).get('routes')
    except (OSError, ValueError, AttributeError):
        return {}
    return found if isinstance(found, dict) else {}


def _write_certified(root: Path, routes: dict[str, dict]) -> None:
    _write_atomic(Path(root) / CERTIFIED_INDEX, lambda temporary: temporary.write_text(
        json.dumps({'routes': routes}, indent=2, sort_keys=True) + '\n', encoding='utf-8'))


# certified.json is read, changed and written back: two at once (two people, or certify and remove) one after the other.
_CERTIFY_LOCK = threading.Lock()


def certify(root: Path, route: str, set_key: str, who: str, made_at: str | None = None) -> dict:
    """The drive's result with that set becomes its reference: kept whole, what went in and what came out. made_at: of
    the result the person looked at; another one on disk (replayed since) is refused, 409 changed."""
    with _CERTIFY_LOCK:
        record = read(root, route, set_key)
        if record is None:
            raise SimulationError(409, 'no_simulation', 'Compute this drive with that set first: there is nothing to certify.')
        if made_at is not None and record.get('made_at') != made_at:
            raise SimulationError(409, 'changed', 'This drive was replayed again since you looked at it: look at the new '
                                  'replay, then certify it.')
        target = certified_path(root, route)
        _write_atomic(target, lambda temporary: _dump_gz(temporary, record))
        entry = {'at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'who': who, 'set': set_key,
                 'name': record.get('name', ''), 'networks': record.get('networks', []), 'made_at': record.get('made_at'),
                 'frames': len(record['frames']['t_ns'])}
        _write_certified(root, {**certified(root), route: entry})
        return entry


def uncertify(root: Path, route: str) -> None:
    with _CERTIFY_LOCK:
        routes = certified(root)
        if routes.pop(route, None) is not None:
            _write_certified(root, routes)
        certified_path(root, route).unlink(missing_ok=True)


def reference(root: Path, route: str) -> dict | None:
    """The certified result of a drive, None when it is not certified."""
    return _read_gz(certified_path(root, route)) if route in certified(root) else None


def reading_text(out: dict, row: int) -> str:
    """What the reading square says at a row while a sign is in view, in a word: the sign as written, with 'other road'
    when the applicability network gave it to another road, and 'read' on the frame it pulses (signRead); '' with no
    sign in view. The square then still shows the last sign, grey: that tail follows from the frames in view, so it is not
    compared again. Its confidence and the distance are not in it: they move by a point or a metre with any other
    network."""
    if not out['rawSignSeen'][row]:
        return 'read' if out['signRead'][row] else ''
    name, value = out['rawSignClass'][row], out['rawSignValue'][row]
    text = 'END' if name == 'cancellation' else name.split(':', 1)[1] if name.startswith('other_sign:') else \
        str(round(value)) if value > 0 else '?'
    return text + (' other road' if out['rawSignRoad'][row] == 'other_road' else '') + (' read' if out['signRead'][row]
                                                                                         else '')


def _runs(rows: list[int], times: list[int], texts: list[tuple]) -> list[list]:
    """Rows that differ, joined while they follow each other with the same two values and no hole between them (more
    than HOLE_SECONDS without a frame, as the series of the page): [from_ns, to_ns, was, now]."""
    runs: list[list] = []
    last = None
    for row, text in zip(rows, texts, strict=True):
        if runs and last is not None and row == last + 1 and times[row] - times[last] <= HOLE_SECONDS * 1e9 and \
                runs[-1][2:] == list(text):
            runs[-1][1] = times[row]
        else:
            runs.append([times[row], times[row], *text])
        last = row
    return runs


def _same_rows(given: list[float] | None, got: list[float] | None) -> bool:
    if given is None or got is None:
        return given is got
    return len(given) == len(got) and all(abs(a - b) <= BRAIN_TOLERANCE for a, b in zip(given, got, strict=True))


def _brain_unchanged(was: dict, now: dict) -> bool:
    """The same brain in both results: its ONNX and the car's code that steps it."""
    def brain(record: dict) -> tuple:
        sha = next((one['sha256'] for one in record.get('networks') or [] if one['role'] == 'brain'), None)
        return sha, (record.get('code') or {}).get('brain.py')

    return brain(was) == brain(now) and brain(was)[0] is not None


def _changed_minutes(was: dict, now: dict, kind: str) -> list[int]:
    """The minutes whose 'video' or 'logs' the two results did not read alike, by SHA-256 (and the kind of log): a
    minute one of them read and the other not counts. A result without them says nothing (since VERSION 2 every result
    keeps them)."""
    mine, theirs = (was.get('sources') or {}).get(kind), (now.get('sources') or {}).get(kind)
    if mine is None or theirs is None:
        return []
    before, after = {row[0]: row[1:] for row in mine}, {row[0]: row[1:] for row in theirs}
    return [segment for segment in sorted(set(before) | set(after)) if before.get(segment) != after.get(segment)]


def _source_changes(was: dict, now: dict) -> list[str]:
    """What the two results did not read alike, in words: 'video of minute 3', 'log of minute 5'."""
    return [f'{word} of minute {segment}' for kind, word in (('video', 'video'), ('logs', 'log'))
            for segment in _changed_minutes(was, now, kind)]


def _restarts(brain: dict, times: list[int], gap: float | None) -> set[int]:
    """The rows where a brain's memory started from zeros, as brain.py does: its first step, and each step more than
    gap seconds after its own one before (steps on frames the other result did not read included)."""
    steps = sorted(int(row) for row in brain)
    return {row for position, row in enumerate(steps)
            if position == 0 or (gap is not None and (times[row] - times[steps[position - 1]]) * 1e-9 > gap)}


def _shared_cuts(was: dict, now: dict) -> tuple[list | None, list | None]:
    """Where each brain's input row has the groups both brains take, in one order: [(from, to)] of each, or None for the
    whole row when they take the same groups (brain_groups [[name, size]])."""
    mine, theirs = was.get('brain_groups'), now.get('brain_groups')
    if not mine or not theirs or mine == theirs:
        return None, None

    def places(groups: list) -> dict[str, tuple[int, int]]:
        found, at = {}, 0
        for name, size in groups:
            found[name], at = (at, at + size), at + size
        return found

    was_places, now_places = places(mine), places(theirs)
    shared = [name for name, _size in mine if name in now_places]
    return [was_places[name] for name in shared], [now_places[name] for name in shared]


def _cut(row: list[float] | None, cuts: list | None) -> list[float] | None:
    return row if row is None or cuts is None else [value for start, end in cuts for value in row[start:end]]


def compare(was: dict, now: dict) -> dict:
    """A result of a drive next to its certified one, by frame (minute and frame of the video): where the speed to hold
    differs (limit), where the reading square says otherwise or pulses at another frame (reading), each as [from_ns,
    to_ns, certified, now]; and of the frames with another speed, how many came from the brain alone (same_input: the
    brain was given the same rows, on the groups both brains take, and the same picture up to there, since its memory
    last started again in both, so the brain changed) or not (other_input: what is read before it changed). With the
    same brain in both, every one is other_input. only_was,
    only_now: frames read in one of the two. inputs_frames: frames where the car's signals kept differ; sources: the
    minutes whose video or log is not the one certified (then the two did not read the same drive)."""
    place = {key: row for row, key in enumerate(zip(now['frames']['segment'], now['frames']['frame'], strict=True))}
    pairs = [(row, place[key]) for row, key in enumerate(zip(was['frames']['segment'], was['frames']['frame'], strict=True))
             if key in place]
    times = was['frames']['t_ns']
    limit_rows, limit_texts, reading_rows, reading_texts = [], [], [], []
    same_input = other_input = inputs_frames = 0
    was_in, now_in = was.get('in', {}), now.get('in', {})
    was_brain, now_brain = was_in.get('brain', {}), now_in.get('brain', {})
    brain_unchanged = _brain_unchanged(was, now)
    was_gap, now_gap = was.get('brain_memory_gap'), now.get('brain_memory_gap')
    was_restarts = _restarts(was_brain, times, was_gap if was_gap is not None else now_gap)
    now_restarts = _restarts(now_brain, now['frames']['t_ns'], now_gap if now_gap is not None else was_gap)
    was_cut, now_cut = _shared_cuts(was, now)
    # The picture the brain steps on: another video of the minute, or other code that makes it, is another input.
    changed_video = set(_changed_minutes(was, now, 'video'))
    other_picture = (was.get('code') or {}).get('vision.py') != (now.get('code') or {}).get('vision.py')
    diverged, previous = False, (-1, -1)
    for mine, theirs in pairs:
        # The brain's memory: once given another row or picture (or a step in one result alone, on frames the other
        # did not read) it carries that until both start again, each after a pause longer than its memory_gap_seconds.
        if any(str(row) in was_brain for row in range(previous[0] + 1, mine)) or \
                any(str(row) in now_brain for row in range(previous[1] + 1, theirs)):
            diverged = True
        given, got = was_brain.get(str(mine)), now_brain.get(str(theirs))
        if given is not None or got is not None:
            if mine in was_restarts and theirs in now_restarts:
                diverged = False
            diverged = diverged or not _same_rows(_cut(given, was_cut), _cut(got, now_cut)) or other_picture or \
                was['frames']['segment'][mine] in changed_video
        previous = (mine, theirs)
        before, after = was['out']['speedLimit'][mine] * 3.6, now['out']['speedLimit'][theirs] * 3.6
        if abs(before - after) >= KPH_TOLERANCE:
            limit_rows.append(mine)
            limit_texts.append((round(before), round(after)))
            given, got = _brain_row(was_brain, mine), _brain_row(now_brain, theirs)
            if brain_unchanged or diverged or not _same_rows(_cut(given, was_cut), _cut(got, now_cut)):
                other_input += 1
            elif given is not None:
                same_input += 1
        read_before, read_after = reading_text(was['out'], mine), reading_text(now['out'], theirs)
        if read_before != read_after:
            reading_rows.append(mine)
            reading_texts.append((read_before, read_after))
        if any(name in was_in and name in now_in and was_in[name][mine] != now_in[name][theirs] for name in INPUTS):
            inputs_frames += 1
    return {'frames': len(pairs), 'only_was': len(times) - len(pairs), 'only_now': len(now['frames']['t_ns']) - len(pairs),
            'limit': _runs(limit_rows, times, limit_texts), 'limit_frames': len(limit_rows),
            'reading': _runs(reading_rows, times, reading_texts), 'reading_frames': len(reading_rows),
            'same_input': same_input, 'other_input': other_input, 'inputs_frames': inputs_frames,
            'sources': _source_changes(was, now)}


def _brain_row(rows: dict, row: int) -> list[float] | None:
    """The brain's input row at a frame: the one of its last step at or before it."""
    while row >= 0:
        found = rows.get(str(row))
        if found is not None:
            return found
        row -= 1
    return None


def differences(found: dict, t0_ns: int) -> dict:
    """compare()'s answer for the page: its runs in seconds from t0_ns and the seconds they last (a frame is PERIOD)."""
    def seconds(runs: list[list]) -> list[list]:
        return [[_seconds(start, t0_ns), _seconds(end, t0_ns), was, now] for start, end, was, now in runs]

    return {**found, 'limit': seconds(found['limit']), 'reading': seconds(found['reading']),
            'limit_seconds': round(found['limit_frames'] * 0.2, 1), 'reading_seconds': round(found['reading_frames'] * 0.2, 1)}


def regression_counts(root: Path, route: str, set_key: str, entry: dict, meta: dict) -> dict | None:
    """How a drive's result stands next to its certified one: compare()'s counts with 'identical', kept beside the
    result until one of the two changes."""
    path = result_path(root, route, set_key)
    cache = path.with_name(path.name + REGRESSION_SUFFIX)
    stamp = [entry.get('at'), meta.get('made_at'), COMPARE_VERSION]
    try:
        kept = json.loads(cache.read_text(encoding='utf-8'))
        if kept.get('stamp') == stamp:
            return kept['counts']
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    was, now = reference(root, route), read(root, route, set_key)
    if was is None or now is None:
        return None
    found = compare(was, now)
    counts = {name: found[name] for name in ('frames', 'only_was', 'only_now', 'limit_frames', 'reading_frames',
                                             'same_input', 'other_input', 'inputs_frames', 'sources')}
    counts['identical'] = not (found['limit_frames'] or found['reading_frames'] or found['only_was'] or found['only_now']
                               or found['sources'])
    try:
        _write_atomic(cache, lambda temporary: temporary.write_text(json.dumps({'stamp': stamp, 'counts': counts}) + '\n',
                                                                    encoding='utf-8'))
    except OSError:
        pass
    return counts


def summary(root: Path, route: str, states: dict[str, dict] | None = None, code: dict[str, str] | None = None) -> dict:
    """What a drive has, for the list and the drive's page: its results by set ({made_at, name, frames, current, changed:
    what of it is not today's}), its
    certification and, for each result, how it stands next to the certified one (regression_counts)."""
    states = states if states is not None else set_states(root)
    code = code if code is not None else code_hashes()
    entry = certified(root).get(route)
    sets, regression = {}, {}
    for key in SETS:
        meta = result_meta(root, route, key)
        if meta is None:
            continue
        sets[key] = {'made_at': meta.get('made_at'), 'name': meta.get('name') or '', 'frames': meta.get('frames', 0),
                     'current': is_current(meta, states[key]['networks'] if states[key]['ready'] else None, code,
                                           states[key].get('settings'), states[key].get('type_head')),
                     'changed': changes(meta, states[key], code)}
        counts = regression_counts(root, route, key, entry, meta) if entry else None
        if counts is not None:
            regression[key] = counts
    return {'route': route, 'sets': sets, 'certified': entry, 'regression': regression}


# -- the replay: the car's loop over a recorded drive --------------------------------------------------------------------
def car_modules(package: Path | None = None) -> SimpleNamespace:
    """The car's package from the sunnypilot clone, as the car imports it (openpilot.sunnypilot.speed_limits): its
    pipeline and the modules the pipeline is built with. For the process of a replay only: the settings of a release
    are put in place in these modules (vision.apply_settings)."""
    package = package or package_path()
    clone = str(package.parents[2])
    if clone not in sys.path:
        sys.path.insert(0, clone)
    name = '.'.join(PACKAGE_RELATIVE.parts)
    try:
        return SimpleNamespace(**{module: importlib.import_module(f'{name}.{module}')
                                  for module in ('pipeline', 'vision', 'applicability', 'brain', 'country')})
    except ImportError as exc:
        raise SimulationError(503, 'no_car_code', f'The car\'s code in {package} does not load ({exc}): it needs '
                              'pipeline.py of the speed-limits branch.') from exc


class _Session:
    """A network's ONNX under onnxruntime, on the GPU: no card, no session (device 'cpu' is for the tests alone)."""
    def __init__(self, path: Path, device: str):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.log_severity_level = 3
        providers: list = ['CPUExecutionProvider']
        if device != 'cpu':
            if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
                raise SimulationError(503, 'no_gpu', 'This environment has no onnxruntime for the GPU: the replay runs on a '
                                      'V100, never on the CPU.')
            if hasattr(onnxruntime, 'preload_dlls'):
                onnxruntime.preload_dlls()
            # The networks are small: the memory they ask for and no more, beside whatever else holds the card.
            providers = [('CUDAExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kSameAsRequested',
                                                    'cudnn_conv_algo_search': 'HEURISTIC'})]
        self.session = onnxruntime.InferenceSession(str(path), options, providers=providers)
        if device != 'cpu' and self.session.get_providers()[0] != 'CUDAExecutionProvider':
            raise SimulationError(503, 'no_gpu', f'{Path(path).name} did not load on the GPU: free a card, the replay never '
                                  'moves to the CPU.')
        self.inputs = {one.name: tuple(size if isinstance(size, int) else 1 for size in one.shape)
                       for one in self.session.get_inputs()}
        self.outputs = [one.name for one in self.session.get_outputs()]


class OneNetwork(_Session):
    """The car's runner.Network: one input, one output, numpy in, numpy (float32) out."""
    def __init__(self, path: Path, device: str):
        super().__init__(path, device)
        (self.name,) = self.inputs

    def __call__(self, x):
        import numpy as np
        return self.session.run(None, {self.name: np.asarray(x, dtype=np.float32)})[0]


class StepNetwork(_Session):
    """The car's runner.StepNetwork: numpy in by name (each broadcast to its input's shape, as the car copies it into a
    buffer of that shape), numpy (float32) out by name. given: what the last call was given."""
    def __init__(self, path: Path, device: str):
        super().__init__(path, device)
        self.given: dict | None = None

    def input_shape(self, name: str) -> tuple[int, ...]:
        return self.inputs[name]

    def __call__(self, **inputs):
        import numpy as np
        feed = {name: np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=np.float32), self.inputs[name]))
                for name, value in inputs.items()}
        self.given = feed
        return dict(zip(self.outputs, self.session.run(None, feed), strict=True))


def _read_log(task: tuple[str, str]) -> dict:
    """What a replay takes from one log file: the rows of each message, its logMonoTime (ns) first, and the file's
    SHA-256."""
    from nnslr_tools.drive_extract import _messages

    path, schema = task
    found: dict = {name: [] for name in ('car', 'enabled', 'motion', 'radar', 'gps', 'encode', 'camera')}
    found['sha256'] = _sha256(Path(path))
    for event in _messages(Path(path), Path(schema), Path(schema)):
        try:
            which = event.which()
        except Exception:  # a member this schema does not know
            continue
        at = event.logMonoTime
        if which == 'carState':
            state = event.carState
            found['car'].append((at, state.vEgo, state.steeringAngleDeg, state.leftBlinker, state.rightBlinker,
                                 state.yawRate, state.gasPressed, state.brakePressed))
        elif which == 'selfdriveState':
            found['enabled'].append((at, event.selfdriveState.enabled))
        elif which == 'deviceMotion':
            yaw = event.deviceMotion.angularVelocityDevice
            found['motion'].append((at, yaw.z, yaw.valid))
        elif which == 'radarState':
            one = event.radarState.leadOne
            found['radar'].append((at, one.present, one.dRel, one.vLead))
        elif which in GPS_SERVICES:
            fix = getattr(event, which)
            found['gps'].append((at, fix.latitude, fix.longitude, fix.horizontalAccuracy, bool(event.valid and fix.hasFix),
                                 fix.unixTimestampMillis))
        elif which == 'narrowRoadEncodeIdx':
            index = event.narrowRoadEncodeIdx
            found['encode'].append((index.segmentNum, index.segmentId, index.frameId, index.timestampEof))
        elif which == 'narrowRoadCameraState':
            found['camera'].append((event.narrowRoadCameraState.frameId, at))
    return found


def read_logs(root: Path, route: str, *, workers: int | None = None) -> dict:
    """The car's messages of a drive for its replay, each kind in the order of its time (the files overlap and are not
    in time order), and when each frame of each minute's video came: frames {segment: [ns or None by frame]}, the
    logMonoTime of the frame's narrowRoadCameraState (the frame named by narrowRoadEncodeIdx), which is the time
    speedvisiond had it; files [[segment, kind, sha256]], the log read of each minute."""
    from nnslr_tools import drive_logs

    files: dict[int, dict] = {}
    for found in drive_logs.discover(root, route):  # by minute, the rlog first
        files.setdefault(found['segment'], found)
    if not files:
        raise SimulationError(409, 'no_logs', f'{route} has no log on this server: the replay needs the car\'s own signals.')
    schema = str(drive_logs.schema_dir())
    tasks = [(files[segment]['path'], schema) for segment in sorted(files)]
    count = min(len(tasks), workers or LOG_WORKERS)
    if count > 1:
        with ProcessPoolExecutor(count) as pool:
            parts = list(pool.map(_read_log, tasks))
    else:
        parts = [_read_log(task) for task in tasks]
    merged = {name: sorted({row for part in parts for row in part[name]}) for name in ('car', 'enabled', 'motion', 'radar', 'gps')}
    camera = {frame_id: at for part in parts for frame_id, at in part['camera']}
    frames: dict[int, list] = {}
    for part in parts:
        for segment, position, frame_id, exposed in part['encode']:
            times = frames.setdefault(segment, [])
            times.extend([None] * (position + 1 - len(times)))
            times[position] = camera.get(frame_id) or (exposed + CAMERA_STATE_LAG_NS if exposed else None)
    read = [[segment, files[segment]['kind'], part['sha256']] for segment, part in zip(sorted(files), parts, strict=True)]
    return {**merged, 'frames': frames, 'files': read}


def _decode(video: Path, width: int, height: int, *, reduce: bool, device: str):
    """Every frame of a minute's video, in order, as NV12 bytes of width x height (the same buffer each time: use it
    before the next one), decoded by the GPU; reduce: a video at full definition, to the low frame."""
    from nnslr_tools.media import REDUCE_FILTER

    size = width * height * 3 // 2
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *(['-hwaccel', 'cuda'] if device != 'cpu' else []),
               '-threads', '1', '-i', str(video), *(['-vf', REDUCE_FILTER] if reduce else []), '-vsync', '0',
               '-pix_fmt', 'nv12', '-f', 'rawvideo', '-']
    buffer = bytearray(size)
    window = memoryview(buffer)
    with tempfile.TemporaryFile() as errors, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
        try:
            # A frame is 1.5 MB: through the default 64 KB pipe it is two dozen reads, most of the replay's time.
            fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETPIPE_SZ, PIPE_BYTES)
        except OSError:
            pass
        try:
            while True:
                filled = 0
                while filled < size:
                    got = process.stdout.readinto(window[filled:])
                    if not got:
                        break
                    filled += got
                if filled < size:
                    break
                yield buffer
        finally:
            process.kill()
            process.wait()
            errors.seek(0)
            said = errors.read().decode('utf-8', 'replace')
        if device != 'cpu' and 'hwaccel' in said.lower():
            raise SimulationError(503, 'no_gpu', f'The GPU did not decode {video}: {said.strip()[-200:]}')


def _number(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def replay(root: Path, route: str, set_key: str, *, device: str = 'cuda', package: Path | None = None,
           workers: int | None = None, settings_country: str | None = None, log=print) -> dict:
    """The drive through the car's loop with the networks of a set: its result (the module's docstring), not written.
    settings_country: the country until GPS gives one (the comma's Settings › OSM), the pipeline's default without it."""
    import numpy as np

    from nnslr_tools import progress
    from nnslr_tools.media import FRAME_SIZE, FULL_SIZE
    from nnslr_tools.road_type_signs import _video_size, _video_stamp, _videos

    package = package or package_path()
    car = car_modules(package)
    directory = set_directory(root, set_key, package)
    files = car.pipeline.release(directory)
    networks = release_networks(directory)
    if files is None or networks is None:
        raise SimulationError(409, 'no_release', f'No networks of the set "{SETS[set_key]}" in {directory}.')
    videos = _videos(Path(root), route)
    if not videos:
        raise SimulationError(409, 'no_video', f'{route} has no video on this server.')
    logs = read_logs(root, route, workers=workers)
    segments = [segment for segment in sorted(videos) if segment in logs['frames']]
    if not segments:
        raise SimulationError(409, 'no_logs', f'The logs of {route} do not say when the frames of its video came.')
    size = _video_size(videos[segments[0]])
    reduce = size == FULL_SIZE
    width, height = FRAME_SIZE if reduce else size

    taken = car.vision.apply_settings(files.settings)
    finder, classifier = OneNetwork(files.finder, device), OneNetwork(files.classifier, device)
    brain_network = StepNetwork(files.brain, device) if files.brain is not None else None
    reader = car.brain.BrainReader(brain_network, files.brain_contract, car.vision, width, height) if brain_network else None
    head = load_type_head(root, files.brain_contract) if reader is not None else None
    if head is not None and head['weight'].shape[1] != reader.hidden.size:
        head = None
    road_types = None
    if files.road_type is not None and reader is None:
        road_types = car.vision.RoadTypeReader(StepNetwork(files.road_type, device), files.road_types, width, height)
    ruler = None
    if files.decider is not None:
        ruler = car.applicability.Applicability(StepNetwork(files.decider, device), files.decider_contract, car.vision,
                                                width, height)
    speed_vision = car.pipeline.SpeedVision(
        files, width, height, car.vision.FOCAL_LENGTHS.get((width, height), 0.0), finder, classifier, reader=reader,
        road_types=road_types, ruler=ruler, settings_country=settings_country or car.pipeline.DEFAULT_COUNTRY)
    log(f'{route}: {files.name}, {" + ".join(ROLES[one["role"]] for one in networks)}, {len(segments)} minutes of '
        f'{width} x {height}' + (f', settings {taken}' if taken else ''))

    messages = {name: logs[name] for name in ('car', 'enabled', 'motion', 'radar', 'gps')}
    at = dict.fromkeys(messages, 0)  # the next message of each kind
    frames: dict[str, list] = {'t_ns': [], 'segment': [], 'frame': []}
    out: dict[str, list] = {field: [] for field in FIELDS}
    signs: dict[str, list] = {}
    inputs: dict[str, list] = {name: [] for name in INPUTS}
    brain_rows: dict[str, list] = {}
    brain_types: dict[str, list] = {}
    car_now = (None,) * 7
    enabled_now = yaw_now = None
    video_sha: list[list] = []
    for position, segment in enumerate(segments):
        times = logs['frames'][segment]
        for index, data in enumerate(_decode(videos[segment], width, height, reduce=reduce, device=device)):
            now_ns = times[index] if index < len(times) else None
            if now_ns is None:
                continue
            now = now_ns * 1e-9
            # As speedvisiond drains its sockets at every frame of the camera: all of each kind, in this order.
            rows = messages['car']
            while at['car'] < len(rows) and rows[at['car']][0] <= now_ns:
                row = rows[at['car']]
                speed_vision.car_state(now, row[0] * 1e-9, *row[1:])
                car_now = row[1:]
                at['car'] += 1
            rows = messages['enabled']
            while at['enabled'] < len(rows) and rows[at['enabled']][0] <= now_ns:
                enabled_now = rows[at['enabled']][1]
                speed_vision.enabled(now, enabled_now)
                at['enabled'] += 1
            rows = messages['motion']
            while at['motion'] < len(rows) and rows[at['motion']][0] <= now_ns:
                _, yaw_now, valid = rows[at['motion']]
                speed_vision.device_motion(now, yaw_now, valid)
                at['motion'] += 1
            rows, last = messages['radar'], None
            while at['radar'] < len(rows) and rows[at['radar']][0] <= now_ns:
                last = rows[at['radar']]
                at['radar'] += 1
            if last is not None:  # its socket keeps the last one only
                speed_vision.radar(*last[1:])
            if not speed_vision.due(now):
                continue
            rows, last = messages['gps'], None
            while at['gps'] < len(rows) and rows[at['gps']][0] <= now_ns:
                last = rows[at['gps']]
                at['gps'] += 1
            if last is not None:
                speed_vision.gps(car.country.GpsFix(last[0] * 1e-9, *last[1:5]), last[5])

            y, uv = car.vision.nv12_planes(np.frombuffer(data, dtype=np.uint8), width, height, width, width * height)
            if brain_network is not None:
                brain_network.given = None
            sent = speed_vision.frame(now, y, uv)
            row = len(frames['t_ns'])
            frames['t_ns'].append(int(now_ns))
            frames['segment'].append(segment)
            frames['frame'].append(index)
            for field, unset in FIELDS.items():
                value = sent.get(field, unset)
                out[field].append(_number(value) if isinstance(unset, float) else bool(value) if isinstance(unset, bool)
                                  else str(value))
            if speed_vision.signs:
                signs[str(row)] = [[*(round(float(corner), 1) for corner in sign['box']), _number(sign.get('score', 0.0)),
                                    sign['class'], _number(sign['confidence']), sign.get('road', '')]
                                   for sign in speed_vision.signs]
            lead, clock = speed_vision.lead, speed_vision.gps_clock
            for name, value in zip(INPUTS, (*car_now, enabled_now, yaw_now, lead[0] if lead else None,
                                            lead[1] if lead else None, speed_vision.gps_country or '',
                                            clock[0] + now - clock[1] if clock else None), strict=True):
                inputs[name].append(value if isinstance(value, (bool, str)) or value is None else _number(value))
            if brain_network is not None and brain_network.given is not None:
                brain_rows[str(row)] = [_number(value) for value in brain_network.given['extra'].reshape(-1)]
                if head is not None:  # type_head on the memory the car's brain just made, as its speed_head reads it
                    logits = head['weight'] @ np.asarray(reader.hidden, dtype=np.float64).reshape(-1) + head['bias']
                    chances = np.exp(logits - logits.max())
                    best = int(chances.argmax())
                    brain_types[str(row)] = [head['names'][best], round(float(chances[best] / chances.sum()), 3)]
        video_sha.append([segment, _sha256(videos[segment])])  # read once more right after the decoder: in the cache
        log(f'minute {position + 1}/{len(segments)}: {len(frames["t_ns"])} frames read')
        progress.report('simulate', position + 1, len(segments), route)
    if not frames['t_ns']:
        raise SimulationError(409, 'no_frames', f'No frame of {route} could be read with its time.')
    return {'version': VERSION, 'route': route, 'set': set_key, 'name': files.name,
            'made_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'device': device,
            'frame': [width, height], 'networks': networks, 'code': code_hashes(package), 'settings': files.settings,
            'videos': _video_stamp({segment: videos[segment] for segment in segments}),
            'brain_memory_gap': (files.brain_contract.get('step') or {}).get('memory_gap_seconds') if reader else None,
            'brain_groups': [[name, car.brain.brain_inputs.INPUT_SIZES[name]] for name in reader.groups] if reader else None,
            'sources': {'video': video_sha, 'logs': [row for row in logs['files'] if row[0] in segments]},
            'frames': frames, 'out': out, 'signs': signs, 'in': {**inputs, 'brain': brain_rows},
            **({'brain_type': brain_types, 'brain_type_head': head['sha256']} if head is not None else {})}


def run(root: Path, routes: list[str], set_key: str, *, device: str = 'cuda', again: bool = False,
        package: Path | None = None, log=print) -> int:
    """The drives replayed with a set and their results written, one after the other; a drive whose result is the one
    of today's networks and code is passed by (again: replayed anyway). The drives that failed."""
    states = set_states(root, package=package)
    code = code_hashes(package)
    failed = 0
    log(f'drives: {len(routes)}')
    for position, route in enumerate(routes):
        meta = result_meta(root, route, set_key)
        if not again and meta is not None and is_current(meta, states[set_key]['networks'] or None, code,
                                                          states[set_key]['settings'], states[set_key]['type_head']):
            log(f'drive {position + 1}/{len(routes)} {route}: replayed already with these networks')
            continue
        try:
            record = replay(root, route, set_key, device=device, package=package, log=log)
        except SimulationError as exc:
            failed += 1
            log(f'drive {position + 1}/{len(routes)} {route}: FAILED, {exc.message}')
            continue
        except Exception as exc:  # a broken video or log, a card out of memory: said on the drive's line, the next go on
            failed += 1
            log(f'drive {position + 1}/{len(routes)} {route}: FAILED, {type(exc).__name__}: {exc}')
            continue
        try:
            write_result(root, record)
        except OSError as exc:
            failed += 1
            log(f'drive {position + 1}/{len(routes)} {route}: FAILED, not written: {exc}')
            continue
        log(f'drive {position + 1}/{len(routes)} {route}: {len(record["frames"]["t_ns"])} frames read, '
            f'{sum(record["out"]["signRead"])} signs read')
    return failed


def export_set(root: Path, *, log=print) -> Path:
    """The versions in use on Models exported as Publish exports them (ONNX, contracts, speed_limits.json with the car's
    settings of today), into the release of the set 'in-use'; nothing when it is theirs already, only speed_limits.json
    again when the car's settings alone changed. In the GPU environment: the export is PyTorch's. Each ONNX is checked
    against PyTorch as Publish checks it, but for the brain's check along a drive. A brain in use that takes the
    driver's inputs is exported too, which Publish refuses: a replay never reaches the car."""
    from nnslr_tools import car_release, car_settings

    chosen = car_release.choose(root, None, brain_on_car=False)
    source = {role: chosen[role]['id'] for role in ROLES if chosen.get(role)}
    target = Path(root) / SETS_RELATIVE / 'in-use'
    networks = release_networks(target)
    if exported_source(root) == source and networks is not None:
        manifest = car_release.manifest_of('in-use', car_settings.values(root), bool(chosen.get('brain')),
                                           bool(chosen.get('decider')))
        if release_settings(target) != manifest.get('settings', {}):
            _write_atomic(target / MANIFEST, lambda temporary: temporary.write_text(
                json.dumps(manifest, indent=2) + '\n', encoding='utf-8'))
            log("the car's settings changed: written again in the export of " + ', '.join(source.values()))
            return target
        log('the versions in use are exported already: ' + ', '.join(source.values()))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='in-use-', dir=target.parent))
    try:
        car_release.export(chosen, folder, drive_check=False)
        manifest = car_release.manifest_of('in-use', car_settings.values(root), bool(chosen.get('brain')),
                                           bool(chosen.get('decider')))
        (folder / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        (folder / SOURCE_FILE).write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
        old = target.with_name(target.name + '.old')
        shutil.rmtree(old, ignore_errors=True)
        if target.exists():
            target.rename(old)
        folder.rename(target)
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    log('exported: ' + ', '.join(source.values()))
    return target
