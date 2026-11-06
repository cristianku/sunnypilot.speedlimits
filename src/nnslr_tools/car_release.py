"""``nnslr send-to-car``: the first three steps of Publish to the car (the Publish tab), the networks the
comma downloads replaced by a SignNet of this server, the classifier in use and the road type or the brain in use.

1. Export: the SignNet of a run (or the cutter in use) and the classifier in use (models/active.json, or one given)
   to ONNX on the CPU, each
   checked against PyTorch with onnxruntime; the road type in use (with memory: road-type/gru), when one is chosen,
   is the ONNX step its training exported and checked (<run>/road-type/export); the brain in use, when one is chosen, is
   exported here too, and with it no road type goes. Each contract names its network by its id of the Models tab.
2. Publish: the ONNX, their contracts and speed_limits.json (format 2: finder, classifier, and road_type or brain when
   there is one in use) in the clone of the models repository the comma downloads (cristianku/speed-limits-models), one commit
   pushed to its main. The export is the same byte for byte each time: when every ONNX is already the one there,
   nothing changes and nothing is pushed. Without a road type in use the published one is removed: speedvisiond
   then reads the signs alone. The same for a brain: speedvisiond then runs its rules again.
3. Record: models/on-car.json (record_on_car, what ``nnslr on-car`` writes), the "On the car" of the Models tab.

scripts/send_to_car.sh runs it, then moves the models submodule of the sunnypilot speed-limits branch (step 4) and
builds cristianku/openpilot psa-torque-sunny-testing (step 5). Nothing from a request reaches a path unchecked: the
SignNet is a run name of the Models tab, the classifier an id of the Models tab. Only a SignNet of the car's band and
grid goes (nnslr_tools.signnet.fits_the_car: the 896 x 288 band at stride 8 the comma's vision.py runs).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from nnslr_tools import car_settings
from nnslr_tools.model_roles import (
    BRAIN_WEIGHTS,
    ON_CAR_FILE,
    ROAD_TYPE_PART,
    SIGNNET_PART,
    WEIGHTS,
    ModelIndex,
    _sha256,
    record_on_car,
    role_of,
)
from nnslr_tools.runs import RUN_NAME, RunsIndex
from nnslr_tools.sign_types import class_hours, class_parts
from nnslr_tools.signnet import CAR_BAND, CAR_STRIDE

CAR_SCRIPT = 'scripts/send_to_car.sh'
# The clones and the build, as scripts/send_to_car.sh finds them (the same variables, the same defaults).
MODELS_CLONE_ENV, MODELS_CLONE = 'NNSLR_MODELS_CLONE', '/root/speed-limits-models'
SUNNYPILOT_CLONE_ENV, SUNNYPILOT_CLONE = 'NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'
OPENDBC_CLONE_ENV, OPENDBC_CLONE = 'NNSLR_OPENDBC_CLONE', '/root/work/opendbc'
OPENPILOT_SCRIPTS_ENV, OPENPILOT_SCRIPTS = 'NNSLR_OPENPILOT_SCRIPTS', '/root/openpilot_scripts'
GPU_VENV_ENV = 'NNSLR_GPU_VENV'
BUILD_SCRIPT = 'OP-setup-psa-torque-testing/scripts/setup_psa_torque.sh'
# Written by the script when sunnypilot's speed-limits moved and the build has not finished yet (step 5 runs then).
PENDING_FILE = 'models/send-to-car.pending'
# The files of the models repository, as speed_limits.json names them; the road type's only when one is in use.
FILES = {'finder': 'signnet.onnx', 'classifier': 'classifier.onnx'}
ROAD_TYPE_FILE = 'road-type.onnx'
BRAIN_FILE = 'brain.onnx'
# The brain's country (speed_limits/country.py): a first try, not validated on the road. Only a release with a brain has it.
COUNTRY_POLICY = {'gps_max_age_s': 5, 'gps_max_accuracy_m': 50, 'border_margin_m': 200, 'confirm_s': 10, 'max_sample_gap_s': 5}
# The road type that goes on the car: the one with memory, whose training exported its ONNX step (one second each).
ROAD_TYPE_ON_CAR = ROAD_TYPE_PART + '/gru'
ROAD_TYPE_EXPORT = ROAD_TYPE_PART + '/export/'
MANIFEST = 'speed_limits.json'
MAX_VERSION = 40
GIT_TIMEOUT = 180


def car_paths(repo: Path) -> dict[str, Path]:
    """Where the job finds its clones and tools (the environment of the site, else the defaults)."""
    return {'models': Path(os.environ.get(MODELS_CLONE_ENV) or MODELS_CLONE),
            'sunnypilot': Path(os.environ.get(SUNNYPILOT_CLONE_ENV) or SUNNYPILOT_CLONE),
            'build': Path(os.environ.get(OPENPILOT_SCRIPTS_ENV) or OPENPILOT_SCRIPTS) / BUILD_SCRIPT,
            'nnslr': Path(os.environ.get(GPU_VENV_ENV) or Path(repo) / '.venv-gpu') / 'bin' / 'nnslr'}


def car_missing(repo: Path) -> str:
    """Why Publish to the car cannot run on this server ('' when it can)."""
    repo = Path(repo)
    if not (repo / CAR_SCRIPT).is_file():
        return CAR_SCRIPT + ' is missing in this installation'
    paths = car_paths(repo)
    if not paths['nnslr'].is_file():
        return 'the GPU environment (.venv-gpu) is missing: it exports the networks'
    if not (paths['models'] / '.git').exists():
        return f'no clone of the published models in {paths["models"]}'
    if not (paths['sunnypilot'] / '.git').exists():
        return f'no sunnypilot clone in {paths["sunnypilot"]}'
    if not paths['build'].is_file():
        return f'the build script of openpilot_scripts is missing ({paths["build"]})'
    return ''


def car_files(chosen: dict) -> dict[str, str]:
    """The networks of a publication and their file: the SignNet and the classifier, the road type or the brain when chosen."""
    return {**FILES, **({'road_type': ROAD_TYPE_FILE} if chosen.get('road_type') else {}),
            **({'brain': BRAIN_FILE} if chosen.get('brain') else {})}


def build_pending(root: Path) -> bool:
    """Whether a build of psa-torque-sunny-testing is still owed (sunnypilot moved, the build did not finish)."""
    return (Path(root) / PENDING_FILE).is_file()


# -- 1. which networks --------------------------------------------------------------------------------------
def choose(root: Path, signnet: Any = None, classifier: Any = None) -> dict:
    """The SignNet of a run listed on the Models tab (None: the cutter in use, which must be a SignNet), finished and
    of the car's band and grid, and the classifier: the one given (an id of the Models tab) or the one in use; the
    road type in use, or None when none is chosen; the brain in use, or None: with a brain no road type goes, nothing on
    the car would read it. Each with its id and its weights on this server (the road type with its exported ONNX step
    and contract)."""
    if signnet is not None and (not isinstance(signnet, str) or not RUN_NAME.fullmatch(signnet)):
        raise ValueError(f'{signnet!r} is not a run name')
    runs = RunsIndex(Path(root))
    index = ModelIndex(runs)
    artifacts = index.artifacts(('cutter', 'classifier'))
    if signnet is None:
        finder, why = index.in_use('cutter')
        if finder is None:
            raise ValueError(why)
        if finder.get('kind') != 'signnet':
            raise ValueError(f'the cutter in use, {finder["id"]}, is an old Detector: choose a SignNet on the Models tab')
        signnet = finder['run']
    else:
        finder_id = f'cutter:{signnet}/{SIGNNET_PART}'
        finder = next((entry for entry in artifacts['cutter'] if entry['id'] == finder_id), None)
        if finder is None:
            raise ValueError(f'{signnet} has no SignNet on the Models tab')
    if not finder['finished']:
        raise ValueError(f'the SignNet of {signnet} has not finished its training')
    if not finder.get('fits_the_car'):
        size = (finder.get('input_size') or '?').replace('x', ' x ')
        raise ValueError(f'the SignNet of {signnet} is {size} at stride {finder.get("stride")}: the comma runs '
                         f'{CAR_BAND[0]} x {CAR_BAND[1]} at stride {CAR_STRIDE} (its vision.py), so it cannot go on '
                         'the car')
    if classifier is None:
        chosen, why = index.in_use('classifier')
        if chosen is None:
            raise ValueError(why)
    else:
        if role_of(classifier) != 'classifier':
            raise ValueError(f'{classifier!r} is not the id of a classifier')
        chosen = next((entry for entry in artifacts['classifier'] if entry['id'] == classifier), None)
        if chosen is None:
            raise ValueError(f'the classifier {classifier} is not on this server')
    if not chosen['finished']:
        raise ValueError(f'the classifier {chosen["id"]} has not finished its training')
    found = {}
    for network, entry, relative in (('finder', finder, SIGNNET_PART + '-best.pt'),
                                     ('classifier', chosen, f'{chosen["part"]}/{WEIGHTS["classifier"]}')):
        run = runs._run_dir(entry['run'])
        weights = None if run is None else runs._file(run, relative)
        if weights is None:
            raise ValueError(f'{entry["id"]} has no weights on this server')
        found[network] = {'id': entry['id'], 'run': entry['run'], 'weights': weights}
    found['brain'] = _brain(Path(root), runs, index)
    found['road_type'] = None if found['brain'] else _road_type(runs, index)
    _check_boards(runs, found, chosen)
    _check_plates(runs, found, chosen)
    _check_wipers(found)
    return found


def _classes(runs: RunsIndex, classifier: dict) -> Any:
    """The classes of a classifier (its result.json), None when they cannot be read."""
    run = runs._run_dir(classifier['run'])
    result = None if run is None else runs._file(run, f'{classifier["part"]}/result.json')
    try:
        return json.loads(Path(result).read_text(encoding='utf-8')).get('classes') if result else None
    except (OSError, ValueError):
        return None


def _check_boards(runs: RunsIndex, found: dict, classifier: dict) -> None:
    """A scene network with the board input only with a classifier that has the green boards towards a motorway among
    its classes: with another one the car never confirms a board, and the input stays "no board" for ever. A check of
    the class's name only (its result.json), not of how well it reads the boards: that is its validation report."""
    road_type = found.get('road_type')
    if not road_type:
        return
    try:
        step = json.loads(Path(road_type['contract']).read_text(encoding='utf-8')).get('step') or {}
    except (OSError, ValueError):
        raise ValueError(f'the contract of the scene network {road_type["id"]} cannot be read') from None
    if not step.get('boards'):
        return
    classes = _classes(runs, classifier)
    wanted = step.get('board_signs') or []
    if not isinstance(classes, list) or not set(wanted) <= set(classes):
        raise ValueError(f'the scene network {road_type["id"]} reads the green boards towards a motorway '
                         f'({", ".join(wanted)}), and the classifier {classifier["id"]} has no such class: choose a '
                         'classifier that has it, or a scene network without the board input')


def _check_plates(runs: RunsIndex, found: dict, classifier: dict) -> None:
    """A brain that learnt the plates of hours, or those for wet roads, only with a classifier that has such a number among
    its classes: with another one the car never reads the plate, and the brain takes every such limit for a plain number. A
    check of the class's name only, as _check_boards."""
    brain = found.get('brain')
    if not brain or not (brain['plates'] or brain['wet']):
        return
    classes = _classes(runs, classifier)
    if brain['wet'] and (not isinstance(classes, list) or not any(class_parts(name)[2] == 'wet' for name in classes)):
        raise ValueError(f'the brain {brain["id"]} reads the plates for wet roads, and the classifier {classifier["id"]} has no '
                         'such class: choose a classifier that has one, or a brain trained without them')
    if not brain['plates']:
        return
    if not isinstance(classes, list) or not any(class_parts(name)[2] == 'time' for name in classes):
        raise ValueError(f'the brain {brain["id"]} reads the plates of hours, and the classifier {classifier["id"]} has no '
                         'class with hours: choose a classifier that has one, or a brain trained without them')
    if brain['hours'] and not any(class_hours(name) for name in classes):
        raise ValueError(f'the brain {brain["id"]} reads which hours a plate says, and the classifier {classifier["id"]} has no '
                         'class that says them: choose a classifier that has one, or a brain trained without them')


def _check_wipers(found: dict) -> None:
    """A brain with the wipers among its inputs only on a car that gives them: every frame of its training said them, and
    "unknown" on all of them it has never seen. A check of the code that reads them in the opendbc clone."""
    brain = found.get('brain')
    if not brain or 'wipers' not in (brain['groups'] or []):
        return
    carstate = Path(os.environ.get(OPENDBC_CLONE_ENV) or OPENDBC_CLONE) / 'opendbc' / 'car' / 'psa' / 'carstate.py'
    if not carstate.is_file() or 'frontWipers' not in carstate.read_text(encoding='utf-8'):
        raise ValueError(f'the brain {brain["id"]} reads the wipers, and opendbc does not give them yet ({carstate}): '
                         'choose a brain trained without them')


def _road_type(runs: RunsIndex, index: ModelIndex) -> dict | None:
    active, problem = index.read_active()
    if problem is not None or active.get('road_type') is None:
        return None
    entry, why = index.in_use('road_type')
    if entry is None:
        raise ValueError(why.replace('road_type', 'scene network'))
    if entry['part'] != ROAD_TYPE_ON_CAR:
        raise ValueError(f'the scene network in use, {entry["id"]}, has no memory: only the one with memory goes on '
                         'the car')
    if not entry['finished']:
        raise ValueError(f'the scene network {entry["id"]} has not finished its training')
    run = runs._run_dir(entry['run'])
    files = {key: None if run is None else runs._file(run, relative) for key, relative in (
        ('weights', f'{ROAD_TYPE_ON_CAR}/road-type-best.pt'), ('onnx', ROAD_TYPE_EXPORT + ROAD_TYPE_FILE),
        ('contract', ROAD_TYPE_EXPORT + ROAD_TYPE_FILE + '.contract.json'))}
    if None in files.values():
        raise ValueError(f'the scene network {entry["id"]} has no exported ONNX step on this server')
    return {'id': entry['id'], 'run': entry['run'], **files}


def _brain(root: Path, runs: RunsIndex, index: ModelIndex) -> dict | None:
    active, problem = index.read_active()
    if problem is not None or active.get('brain') is None:
        return None
    entry, why = index.in_use('brain')
    if entry is None:
        raise ValueError(why)
    if not entry['finished']:
        raise ValueError(f'the brain {entry["id"]} has not finished its training')
    run = runs._run_dir(entry['run'])
    weights = None if run is None else runs._file(run, BRAIN_WEIGHTS)
    if weights is None:
        raise ValueError(f'{entry["id"]} has no weights on this server')
    return {'id': entry['id'], 'run': entry['run'], 'weights': weights, 'root': root, 'plates': entry['plates'], 'hours': entry['hours'],
            'wet': entry['wet'], 'groups': entry['groups']}


def export_brain(chosen: dict, folder: Path) -> None:
    """The brain's step to ONNX in folder, checked against PyTorch and along a real drive (it raises when it differs)."""
    if chosen.get('brain'):
        from nnslr_tools.brain_export import export_brain_onnx
        from nnslr_tools.drive_review import DriveReviewStore

        root = chosen['brain']['root']
        store = DriveReviewStore(root)
        export_brain_onnx(chosen['brain']['weights'], folder / BRAIN_FILE, data_root=root,
                          merged_of=lambda route: store.merged(route, keep=False))


def copy_road_type(chosen: dict, folder: Path) -> None:
    """The road type's ONNX step and its contract in folder, as its training exported and checked them."""
    if chosen.get('road_type'):
        shutil.copyfile(chosen['road_type']['onnx'], folder / ROAD_TYPE_FILE)
        shutil.copyfile(chosen['road_type']['contract'], folder / (ROAD_TYPE_FILE + '.contract.json'))


# -- 2. the export ------------------------------------------------------------------------------------------
def export(chosen: dict, folder: Path) -> dict:
    """Both sign networks to ONNX in folder (signnet.onnx, classifier.onnx and their contracts), each checked against
    PyTorch with onnxruntime (they raise when it differs), and the road type's or the brain's step when one is in use;
    each contract names its checkpoint by its id."""
    from nnslr_tools.exporting import export_reader_onnx
    from nnslr_tools.signnet import export_signnet_onnx

    exporters = {'finder': export_signnet_onnx, 'classifier': export_reader_onnx}
    for network, name in FILES.items():
        exporters[network](chosen[network]['weights'], folder / name, verify=True)
    copy_road_type(chosen, folder)
    export_brain(chosen, folder)
    return finish_contracts(chosen, folder)


def finish_contracts(chosen: dict, folder: Path) -> dict:
    """The contracts of an export with the ids of the Models tab (not a path of this server) as their checkpoint;
    the sha256 of each ONNX, checked against its contract."""
    exported = {}
    for network, name in car_files(chosen).items():
        onnx = folder / name
        contract_path = folder / (name + '.contract.json')
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        digest = _sha256(onnx)
        if contract.get('onnx_sha256') != digest:
            raise ValueError(f'{name} is not the ONNX of its contract')
        contract['checkpoint'] = chosen[network]['id']
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        exported[network] = {'id': chosen[network]['id'], 'file': name, 'onnx_sha256': digest}
    return exported


# -- 3. the models repository -------------------------------------------------------------------------------
def _git(folder: Path, *args: str) -> str:
    done = subprocess.run(['git', '-C', str(folder), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT,
                          check=False)
    if done.returncode != 0:
        raise ValueError(f'git {" ".join(args)} in {folder}: {done.stderr.strip() or "failed"}')
    return done.stdout.strip()


def manifest_of(version: str, road_type: bool = False, settings: dict | None = None, brain: bool = False) -> dict:
    """speed_limits.json; settings: the car's (car_settings.values), each a constant of vision.py on the comma, and with a
    brain the policy of its country."""
    files = {**FILES, **({'road_type': ROAD_TYPE_FILE} if road_type else {}), **({'brain': BRAIN_FILE} if brain else {})}
    settings = {**(settings or {}), **({'country_policy': COUNTRY_POLICY} if brain else {})}
    return {'format': 2, 'version': version,
            **{network: {'file': name, 'contract': name + '.contract.json'} for network, name in files.items()},
            **({'settings': settings} if settings else {})}


def published_settings(models: Path | None = None) -> dict | None:
    """The car's settings in the published speed_limits.json ({} when it has none: the comma's defaults), None
    without a manifest."""
    manifest = _read_manifest(Path(models or os.environ.get(MODELS_CLONE_ENV) or MODELS_CLONE))
    if manifest is None:
        return None
    settings = manifest.get('settings')
    return settings if isinstance(settings, dict) else {}


def next_version(current: Any, today: date) -> str:
    """The version of a new publication: the day; the day with .2, .3... when the current one is already of that day."""
    day = today.isoformat()
    if not isinstance(current, str) or (current != day and not current.startswith(day + '.')):
        return day
    suffix = current[len(day) + 1:]
    return f'{day}.{int(suffix) + 1 if suffix.isdigit() and len(suffix) < 6 else 2}'[:MAX_VERSION]


def _read_manifest(models: Path) -> dict | None:
    path = models / MANIFEST
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sync(models: Path, *, dry_run: bool) -> str:
    """The clone on main, without changes of its own, equal to origin's main (fast-forwarded when only behind): the
    commit it is at."""
    if _git(models, 'symbolic-ref', '--quiet', '--short', 'HEAD') != 'main':
        raise ValueError(f'{models} is not on its main branch')
    if _git(models, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError(f'{models} has changes not committed: commit or drop them first')
    _git(models, 'fetch', '--quiet', 'origin', '+refs/heads/main:refs/remotes/origin/main')
    ahead, behind = (int(value) for value in _git(models, 'rev-list', '--left-right', '--count',
                                                    'HEAD...refs/remotes/origin/main').split())
    if ahead:
        raise ValueError(f'{models} has {ahead} commit(s) not on origin/main: push or drop them first')
    if behind and not dry_run:
        _git(models, 'merge', '--quiet', '--ff-only', 'refs/remotes/origin/main')
    return _git(models, 'rev-parse', 'HEAD')


def publish(models: Path, chosen: dict, folder: Path, exported: dict, *, today: date | None = None,
            dry_run: bool = False, settings: dict | None = None) -> dict:
    """The export in the clone of the models repository and pushed (step 2): {'changed', 'commit', 'version'}. A
    change of the settings alone is a new version too."""
    models = Path(models)
    if not (models / '.git').exists():
        raise ValueError(f'{models} is not a git clone')
    start = _sync(models, dry_run=dry_run)
    manifest = _read_manifest(models)
    current_version = (manifest or {}).get('version')
    files = car_files(chosen)
    road_type, brain = 'road_type' in files, 'brain' in files
    same_manifest = manifest is not None and {key: value for key, value in manifest.items() if key != 'version'} == {
        key: value for key, value in manifest_of('', road_type, settings, brain).items() if key != 'version'}
    same = same_manifest and all(
        (models / name).is_file() and not (models / name).is_symlink()
        and _sha256(models / name) == exported[network]['onnx_sha256'] for network, name in files.items())
    if same:
        return {'changed': False, 'commit': start, 'version': current_version}
    version = next_version(current_version, today or date.today())
    if dry_run:
        return {'changed': True, 'commit': None, 'version': version}
    paths = []
    # No road type in use: the published one goes, speedvisiond reads the signs alone. The same for the brain.
    unused = [name for name, used in ((ROAD_TYPE_FILE, road_type), (BRAIN_FILE, brain)) if not used]
    gone = [file for name in unused for file in (name, name + '.contract.json') if _git(models, 'ls-files', '--', file)]
    for name in files.values():
        for file in (name, name + '.contract.json'):
            target = models / file
            if target.is_symlink():
                raise ValueError(f'{target} is a link: the clone must hold the files themselves')
            shutil.copyfile(folder / file, target)
            paths.append(file)
    (models / MANIFEST).write_text(json.dumps(manifest_of(version, road_type, settings, brain), indent=2) + '\n',
                                   encoding='utf-8')
    paths.append(MANIFEST)
    message = (f'speed limits {version}: SignNet {chosen["finder"]["run"]}, classifier {chosen["classifier"]["run"]}'
               + (f', road type {chosen["road_type"]["run"]}' if road_type else '')
               + (f', brain {chosen["brain"]["run"]}' if brain else '') + '\n\n'
               + ''.join(f'{network} {exported[network]["id"]} (onnx sha256 {exported[network]["onnx_sha256"][:12]})\n'
                         for network in files)
               + 'Sent from the review site (Models, Send to the car: nnslr send-to-car).\n')
    try:
        _git(models, 'add', '--', *paths)
        if gone:
            _git(models, 'rm', '--quiet', '--', *gone)
        _git(models, 'commit', '--quiet', '-m', message, '--', *paths, *gone)
        commit = _git(models, 'rev-parse', 'HEAD')
        _git(models, 'push', '--quiet', 'origin', 'HEAD:refs/heads/main')
    except (ValueError, subprocess.TimeoutExpired):
        # Back where it was (it was clean): the next run starts from origin's main again, nothing half done here.
        _git(models, 'reset', '--quiet', '--hard', start)
        raise
    _git(models, 'fetch', '--quiet', 'origin', '+refs/heads/main:refs/remotes/origin/main')
    return {'changed': True, 'commit': commit, 'version': version}


# -- 4. the record --------------------------------------------------------------------------------------------
def _same_record(root: Path, record: dict) -> bool:
    try:
        old = json.loads((Path(root) / ON_CAR_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if not isinstance(old, dict):
        return False
    return {key: item for key, item in old.items() if key != 'recorded_at'} == {
        key: item for key, item in record.items() if key != 'recorded_at'}


def send_to_car(root: Path, signnet: Any = None, *, classifier: Any = None, models: Path | None = None,
                dry_run: bool = False, today: date | None = None, steps_of: int = 3,
                exporter: Callable[[dict, Path], dict] | None = None,
                say: Callable[[str], None] = lambda line: print(line, flush=True)) -> dict:
    """Steps 1 to 3 of Publish to the car (see the module): the export, its publication when it differs from the
    published networks, and on-car.json; signnet None: the cutter in use. Returns {'changed', 'commit', 'version', 'record'}."""
    root = Path(root)
    models = Path(models or os.environ.get(MODELS_CLONE_ENV) or MODELS_CLONE)
    chosen = choose(root, signnet, classifier)
    road_type = f', the road type {chosen["road_type"]["id"]}' if chosen.get('road_type') else ''
    brain = f', the brain {chosen["brain"]["id"]}' if chosen.get('brain') else ''
    say(f'== 1/{steps_of} Export the SignNet of {chosen["finder"]["run"]} and the classifier {chosen["classifier"]["id"]}'
        + road_type + brain)
    work = root / 'tmp'
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='send-to-car-', dir=work) as name:
        folder = Path(name)
        exported = (exporter or export)(chosen, folder)
        for network in car_files(chosen):
            say(f'{network}: {exported[network]["id"]} onnx sha256 {exported[network]["onnx_sha256"]}')
        say(f'== 2/{steps_of} Publish them in {models}')
        published = publish(models, chosen, folder, exported, today=today, dry_run=dry_run,
                            settings=car_settings.values(root))
    if not published['changed']:
        say(f'unchanged: the published networks are these ones already (version {published["version"]}, '
            f'commit {published["commit"][:7]}); no commit, no push')
    elif dry_run:
        say(f'dry run: would publish version {published["version"]} (nothing copied, committed or pushed)')
    else:
        say(f'published: version {published["version"]}, commit {published["commit"]} pushed to origin main')
    say(f'== 3/{steps_of} Record the networks on the car')
    record = None
    if dry_run and published['changed']:
        say('dry run: would record the new commit in ' + ON_CAR_FILE)
    else:
        record = record_on_car(root, models, dry_run=True)
        if dry_run:
            say('dry run: ' + ON_CAR_FILE + ' not written')
        elif _same_record(root, record):
            say(ON_CAR_FILE + ' already records them')
        else:
            record = record_on_car(root, models)
            say('recorded in ' + ON_CAR_FILE)
        road = ''.join(f', {record[role]["id"]}' for role in ('road_type', 'brain') if record.get(role))
        say(f'on the car: {record["cutter"]["id"]} and {record["classifier"]["id"]}{road}, commit {record["commit"][:7]}')
    return {**published, 'record': record}
