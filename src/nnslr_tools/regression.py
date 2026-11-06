"""Models › Regression: drives he certified as driven well, kept to try every new model on them.

What it is for. He picks drives where the car went as it should. They stay certified, so after a training, or with
another openpilot model, he opens one and sees along its video his own speed, the speed each driving model asks for and
the speed Smartspeed (the brain) says. Nothing here trains, changes a label or goes near the car.

Open loop. The models run on the recorded video: each sees the speed he held, not its own. So a curve says what the model
would ask from that moment, never where a car driven by it would have been.

Who computes what:

- the driving models, in openpilot.distill (helpers/route_speed.sh, evaluation/route_speed.py): stock is Small (CD210, the
  comma's), teacher Big (MD v3), logged what really ran in the car (no camera needed), run:<folder> a distilled run;
- the brain, here (run_brains, nnslr regression-brain): along the whole drive as on the car, with the signs read by the
  car's reader of today (road_type_signs). A brain made with other inputs than today's is refused, with the reason on the
  page. A reader that changed makes its results old.

Each result is a file of its own, <root>/regression/results/<route>/<model>.json.gz, so after a training only the new
model runs (missing). A failed model writes {"error"} and is tried again by the next Run. A model that ran on part of a
drive runs again when more of its wide camera arrived.

The curves (route_view) are on seconds from the drive's start. A driving model's plan at horizon h, made at t, is drawn at
t + h: every number on the chart is then about the same moment, what it asked h seconds before for the time shown. The
brain says the speed to hold now, so it is not moved. "The car with Smartspeed" is the lower of a model and the brain: the
brain is a cap, the model drives under it, which is what the car would ask for.

The score of a curve is against his own speed at the same moment: the mean difference, the mean distance, and the seconds
it asked TOO_FAST_KPH or more above him. He drove these drives well, so a model far above him is the thing to look at.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from pathlib import Path

ROUTES_RELATIVE = Path('regression') / 'routes.json'
RESULTS_RELATIVE = Path('regression') / 'results'
# brain_targets' files, by name: importing it needs the sunnypilot clone.
SPLIT_RELATIVE = Path('splits') / 'speed-brain-split.json'
SPLIT_FRAMES_RELATIVE = Path('splits') / 'speed-brain-frames.json'
# The specs of evaluation/specs.py this page offers, with the name he knows them by.
DRIVING_MODELS = (('stock', 'Small', 'CD210, the comma’s'), ('teacher', 'Big', 'MD v3'),
                  ('logged', 'In the car', 'what really ran on that drive'))
LOGGED = 'logged'
RUN_SPEC = re.compile(r'run:[0-9A-Za-z][0-9A-Za-z._-]*')
BRAIN_RUN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
RUN_TIME = re.compile(r'(\d{4})-(\d\d)-(\d\d)_(\d\d)(\d\d)')
MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
HORIZONS_S = (2, 5, 10)
DEFAULT_HORIZON_S = 5
TOO_FAST_KPH = 10
# Rows further apart than this are a hole in a curve, not a line across it.
HOLE_S = 3
NS = 1_000_000_000

_write_lock = threading.Lock()


class RegressionError(Exception):
    def __init__(self, status: int, reason: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


# -- the certified drives -----------------------------------------------------------------------------------------------
def certified(root: Path) -> dict[str, dict]:
    """{route: {'at', 'who'}}."""
    path = Path(root) / ROUTES_RELATIVE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))['routes']


def _write_routes(root: Path, routes: dict[str, dict]) -> None:
    path = Path(root) / ROUTES_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'routes': routes}, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def certify(root: Path, route: str, who: str, known: list[str]) -> None:
    if route not in known:
        raise RegressionError(404, 'unknown_drive', 'This drive is not on the server.')
    with _write_lock:
        routes = certified(root)
        routes.setdefault(route, {'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'who': who})
        _write_routes(root, routes)


def uncertify(root: Path, route: str) -> None:
    """The results stay: certified again, it needs no new run."""
    with _write_lock:
        routes = certified(root)
        routes.pop(route, None)
        _write_routes(root, routes)


# -- the models ---------------------------------------------------------------------------------------------------------
def _run_text(name: str) -> str:
    found = RUN_TIME.match(name)
    if not found:
        return name
    _, month, day, hour, minute = found.groups()
    return f'{int(day)} {MONTHS[int(month) - 1]} {hour}:{minute}'


def driving_models(distill: Path) -> list[dict]:
    """Small, Big, In the car, then the finished runs of openpilot.distill, the newest first: a .training.pt alone is a
    training that did not end."""
    models = [{'key': key, 'label': label, 'detail': detail, 'needs_wide': key != LOGGED}
              for key, label, detail in DRIVING_MODELS]
    runs = sorted((Path(distill) / 'outputs' / 'supervised').glob('*/supercombo.pt'), reverse=True)
    return models + [{'key': 'run:' + path.parent.name, 'label': 'Distilled ' + _run_text(path.parent.name),
                      'detail': path.parent.name, 'needs_wide': True} for path in runs]


def brain_models(versions: list[dict]) -> list[dict]:
    """The versions of Models › Brain (model_roles), the newest first, the one in use marked."""
    return [{'key': version['run'], 'label': _run_text(version['run']) + (f', seed {version["seed"]}'
                                                                          if version.get('seed') is not None else ''),
             'detail': version['run'], 'in_use': bool(version.get('in_use'))}
            for version in versions if version.get('finished')]


def checked_choice(text, known: set[str], what: str) -> list[str]:
    """A comma list of the page, each of them known."""
    if not isinstance(text, str):
        raise RegressionError(400, 'bad_option', f'Choose the {what} as a list of names.')
    chosen = [name for name in text.split(',') if name]
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise RegressionError(400, 'bad_option', f'Unknown {what}: {unknown[0]}.')
    return chosen


# -- the results --------------------------------------------------------------------------------------------------------
def result_path(root: Path, route: str, spec: str) -> Path:
    return Path(root) / RESULTS_RELATIVE / route / (spec.replace(':', '_') + '.json.gz')


def brain_spec(run: str) -> str:
    return 'brain:' + run


def read_result(root: Path, route: str, spec: str) -> dict | None:
    path = result_path(root, route, spec)
    if not path.is_file():
        return None
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        return json.load(source)


def write_result(root: Path, route: str, spec: str, body: dict) -> Path:
    path = result_path(root, route, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8') as output:
        json.dump({'route': route, 'model': spec, **body}, output)
    os.replace(temporary, path)
    return path


def wide_segments(videos: Path, route: str) -> tuple[int, int]:
    """(segments with the wide camera, segments) of a drive in the comma's folders."""
    found = [path for path in Path(videos).glob(route + '--*') if path.is_dir() and path.name.rsplit('--', 1)[1].isdigit()]
    return sum((path / 'ecamera.hevc').is_file() for path in found), len(found)


def wide_index(videos: Path) -> dict[str, tuple[int, int]]:
    """wide_segments of every drive, reading the folder once."""
    found: dict[str, list[int]] = {}
    for path in Path(videos).glob('*--*--*'):
        route, _sep, segment = path.name.rpartition('--')
        if segment.isdigit() and path.is_dir():
            counts = found.setdefault(route, [0, 0])
            counts[0] += (path / 'ecamera.hevc').is_file()
            counts[1] += 1
    return {route: (wide, total) for route, (wide, total) in found.items()}


def driving_state(root: Path, route: str, spec: str, wide: tuple[int, int]) -> str:
    """done, missing (a Run computes it), failed (tried, tried again), no_wide (a Run cannot) or more_wide."""
    needs_wide = spec != LOGGED
    if needs_wide and not wide[0]:
        return 'no_wide'
    result = read_result(root, route, spec)
    if result is None:
        return 'missing'
    if result.get('error'):
        return 'failed'
    if needs_wide and result.get('segments', 0) < wide[0]:
        return 'more_wide'
    return 'done'


def missing_driving(root: Path, videos: Path, specs: list[str]) -> list[tuple[str, list[str]]]:
    """[(route, the specs a Run has to compute there)] of the certified drives."""
    plan = []
    for route in sorted(certified(root)):
        wide = wide_segments(videos, route)
        wanted = [spec for spec in specs if driving_state(root, route, spec, wide) in ('missing', 'failed', 'more_wide')]
        if wanted:
            plan.append((route, wanted))
    return plan


def frames_ready(root: Path, route: str) -> bool:
    """The drive was prepared: the brain sees the frames of the preparation, one a second."""
    from nnslr_tools.road_type_data import FRAMES_RELATIVE
    return any((Path(root) / FRAMES_RELATIVE / route).glob('*/frames.jsonl'))


# -- the page -----------------------------------------------------------------------------------------------------------
def _labelled_in_train(route: str, stretches: list[dict], start_ns: int, split: dict) -> float | None:
    """The seconds of a drive the brain learned from: labelled and in train. None for a drive of no frozen split, where
    only the labels are known. A frame key is <segment>/<second>, a second of the drive."""
    if route not in split['partitions']:
        return None
    spans = [((row['start_ns'] - start_ns) / NS, (row['end_ns'] - start_ns) / NS) for row in stretches if row.get('kph')]
    seconds = 0
    for key, partition in zip(split['frame_keys'][route], split['partitions'][route]):
        segment, second = key.split('/')
        at = int(segment) * 60 + int(second)
        seconds += partition == 'train' and any(first <= at < last for first, last in spans)
    return float(seconds)


def distill_segments(distill: Path) -> dict[str, int]:
    """{route with no dashes: segments in the data of openpilot.distill}: what its runs could learn from, its held-out
    ones among them (only its own dataset code knows those)."""
    counts: dict[str, int] = {}
    for path in (Path(distill) / 'data').glob('*/fcamera.hevc'):
        name = path.parent.name
        if len(name) == 20:
            counts[name[:18]] = counts.get(name[:18], 0) + 1
    return counts


def frozen_split(root: Path) -> dict | None:
    """The partitions of the brain's frozen split with their frame keys, None before there is one."""
    try:
        split = json.loads((Path(root) / SPLIT_RELATIVE).read_text(encoding='utf-8'))
        frames = json.loads((Path(root) / SPLIT_FRAMES_RELATIVE).read_text(encoding='utf-8'))
    except OSError:
        return None
    return {'partitions': split['partitions'], 'frame_keys': frames['frames']}


def route_rows(root: Path, drives: list[dict], videos: Path, distill: Path, specs: list[str], brains: list[str],
               split: dict | None, stretches_of, start_of) -> list[dict]:
    """Every drive for the list (speed_review.drives_list); a certified one also with what has run on it and how much of
    it the models learned from. stretches_of(route), start_of(route): its labels and its first ns."""
    chosen = certified(root)
    distilled = distill_segments(distill)
    wides = wide_index(videos)
    rows = []
    for drive in drives:
        route = drive['route']
        wide = wides.get(route, (0, 0))
        row = {**drive, 'certified': chosen.get(route), 'wide': list(wide)}
        if route in chosen:
            ready = frames_ready(root, route)
            learned = None
            if split is not None and start_of(route) is not None:
                learned = _labelled_in_train(route, stretches_of(route), start_of(route), split)
            row.update({'frames': ready, 'brain_learned_s': learned,
                        'distill_segments': distilled.get(route.replace('--', ''), 0),
                        'results': {**{spec: driving_state(root, route, spec, wide) for spec in specs},
                                    **{brain_spec(run): _brain_state(root, route, run, ready) for run in brains}}})
        rows.append(row)
    return rows


def _brain_state(root: Path, route: str, run: str, ready: bool) -> str:
    if not ready:
        return 'no_frames'
    result = read_result(root, route, brain_spec(run))
    return 'missing' if result is None else 'failed' if result.get('error') else 'done'


def _curve(key: str, label: str, kind: str, points: list[list[float]], own: dict[int, float]) -> dict:
    return {'key': key, 'label': label, 'kind': kind, 'points': points, 'score': score(points, own)}


def score(points: list[list[float]], own: dict[int, float]) -> dict | None:
    """The curve against his speed at the same second, None when they never meet."""
    gaps = [kph - own[round(at)] for at, kph in points if round(at) in own]
    if not gaps:
        return None
    return {'seconds': len(gaps), 'mean_kph': round(sum(gaps) / len(gaps), 1),
            'abs_kph': round(sum(abs(gap) for gap in gaps) / len(gaps), 1),
            'above_s': sum(gap >= TOO_FAST_KPH for gap in gaps)}


def plan_points(result: dict, t0: int, horizon: int) -> list[list[float]]:
    """The plan at a horizon, each value on the moment it is about (t + horizon), in seconds from the drive's start."""
    column = result['horizons_s'].index(float(horizon))
    points = []
    for t_ns, row in zip(result['t_ns'], result['kph']):
        if row[column] is not None:
            points.append([round((t_ns - t0) / NS + horizon, 1), row[column]])
    return points


def brain_points(result: dict, t0: int) -> list[list[float]]:
    return [[round((t_ns - t0) / NS, 1), kph] for t_ns, kph in zip(result['t_ns'], result['kph'])]


def lower_of(model: list[list[float]], brain: list[list[float]]) -> list[list[float]]:
    """The lower of two curves at the seconds both have: the brain caps, the model drives under it."""
    caps = {round(at): kph for at, kph in brain}
    return [[at, min(kph, caps[round(at)])] for at, kph in model if round(at) in caps]


def own_speed(merged: dict) -> dict[int, float]:
    """{second from the start: his speed in km/h}. The logs keep a speed only when it changes: each one holds until the
    next, and until the end of the drive, but not across the segments missing from the logs."""
    t0 = merged['t0']
    rows = [(t_ns, kph) for t_ns, kph in (merged.get('series') or {}).get('v_ego') or [] if isinstance(kph, (int, float))]
    spans = merged.get('spans') or {}
    holes = [(spans[after][1], spans[before][0]) for after, before in merged.get('holes') or []
             if after in spans and before in spans]
    end = (merged.get('span') or [t0, rows[-1][0] if rows else t0])[1]
    speeds: dict[int, float] = {}
    for (t_ns, kph), next_ns in zip(rows, [row[0] for row in rows[1:]] + [end]):
        for second in range(round((t_ns - t0) / NS), max(round((next_ns - t0) / NS), round((t_ns - t0) / NS) + 1)):
            at = t0 + second * NS
            if not any(first < at < last for first, last in holes):
                speeds.setdefault(second, round(kph, 1))
    return speeds


def route_view(root: Path, route: str, merged: dict | None, models: list[dict], brains: list[dict], *, horizon: int,
               brain: str | None) -> dict:
    """The curves of a drive at a horizon: his speed, each driving model with a result, each brain and, with the brain
    chosen, the car with Smartspeed for each driving model."""
    if horizon not in HORIZONS_S:
        raise RegressionError(400, 'bad_horizon', 'The horizon is 2, 5 or 10 seconds.')
    if merged is None or merged.get('t0') is None:
        raise RegressionError(404, 'not_scanned', 'This drive has no scanned logs.')
    t0 = merged['t0']
    own = own_speed(merged)
    curves, failures, driving = [], [], []
    for model in models:
        result = read_result(root, route, model['key'])
        if result is None:
            continue
        if result.get('error'):
            failures.append({'key': model['key'], 'label': model['label'], 'error': result['error']})
            continue
        points = plan_points(result, t0, horizon)
        driving.append((model, points))
        curves.append(_curve(model['key'], model['label'], 'driving', points, own))
    said = {}
    for item in brains:
        result = read_result(root, route, brain_spec(item['key']))
        if result is None:
            continue
        if result.get('error'):
            failures.append({'key': brain_spec(item['key']), 'label': 'Smartspeed ' + item['label'], 'error': result['error']})
            continue
        said[item['key']] = brain_points(result, t0)
        curves.append(_curve(brain_spec(item['key']), 'Smartspeed ' + item['label'], 'brain', said[item['key']], own))
    if brain in said:
        for model, points in driving:
            curves.append(_curve('car:' + model['key'], model['label'] + ' with Smartspeed', 'car', lower_of(points, said[brain]), own))
    return {'ok': True, 'route': route, 'horizon': horizon, 'brain': brain if brain in said else None,
            'own': [[at, kph] for at, kph in sorted(own.items())], 'curves': curves, 'failures': failures, 'hole_s': HOLE_S,
            'too_fast_kph': TOO_FAST_KPH}


# -- the brain along a drive (.venv-gpu) ----------------------------------------------------------------------------------
def brain_drive(root: Path, route: str, merged: dict) -> dict:
    """A drive for brain_train.predict, labels or not: brain_targets.dataset() only knows the labelled ones."""
    import numpy as np

    from nnslr_tools.brain_targets import drive_frames, drive_inputs, memory_cuts
    from nnslr_tools.drive_wipers import load_wipers
    from nnslr_tools.road_type_signs import load_signs
    frames = drive_frames(root, route, merged)
    ns = np.array([-1 if frame['ns'] is None else frame['ns'] for frame in frames], dtype=np.int64)
    record = load_signs(root, route)
    return {'route': route, 'frames': frames, 'ns': ns, 'cuts': memory_cuts(ns), 'partitions': ['train'] * len(frames),
            'inputs': drive_inputs(frames, ns, merged, record['readings'] if record else None, root, load_wipers(root, route)),
            'reader': record.get('fingerprint') if record else None, 'signs_read': record is not None}


def brains_to_run(root: Path, route: str, runs: list[str], reader) -> list[str]:
    """The brains with no result on the drive, a failed one, or one from another reader of the signs."""
    wanted = []
    for run in runs:
        result = read_result(root, route, brain_spec(run))
        if result is None or result.get('error') or result.get('reader') != reader:
            wanted.append(run)
    return wanted


def run_brains(root: Path, runs: list[str], merged_of, *, device, log=print) -> int:
    """Each brain along each certified drive still missing it; a drive or a brain that cannot run stops only itself.
    The drives that could not run, for the exit code."""
    from nnslr_tools.brain_export import load_brain
    from nnslr_tools.brain_targets import BRAIN_INPUTS, inputs_sha256, pack_inputs
    from nnslr_tools.brain_train import CHUNK, _pictures, predict
    from nnslr_tools.model_roles import BRAIN_WEIGHTS
    from nnslr_tools.road_type_net import FrameStore
    routes = [route for route in sorted(certified(root)) if frames_ready(root, route)]
    accepted = (inputs_sha256(), *BRAIN_INPUTS.SAME_GROUPS)
    failed = 0
    for position, route in enumerate(routes, 1):
        log(f'regression item brains {position - 1}/{len(routes)} {route}')
        try:
            merged = merged_of(route)
            if merged is None:
                raise ValueError('no scanned logs')
            drive = brain_drive(root, route, merged)
            wanted = brains_to_run(root, route, runs, drive['reader'])
            if not wanted:
                continue
            if not drive['signs_read']:
                raise ValueError('its signs were not read')
            store = FrameStore(_pictures(root, [drive], dry_run=False, log=log), device)
        except Exception as problem:
            log(f'{route}: left out: {problem}')
            failed += 1
            continue
        for run in wanted:
            try:
                model, saved = load_brain(Path(root) / 'runs' / run / BRAIN_WEIGHTS)
                if saved['inputs_sha256'] not in accepted:
                    raise ValueError('made with other inputs than the car has today: train it again')
                drive['extra'] = pack_inputs(drive['inputs'], tuple(saved['groups']))
                said = predict(model.to(device), drive, store, chunk=CHUNK, device=device)
                usable = [at for at, frame in enumerate(drive['frames']) if frame['ns'] is not None and frame['present']]
                write_result(root, route, brain_spec(run), {
                    'reader': drive['reader'], 't_ns': [int(drive['ns'][at]) for at in usable],
                    'kph': [int(said['kph'][at]) for at in usable]})
                log(f'{route}: brain {run} done')
            except Exception as problem:
                log(f'{route}: brain {run} failed: {problem}')
                write_result(root, route, brain_spec(run), {'reader': drive['reader'], 'error': str(problem)})
                failed += 1
    log(f'regression item brains {len(routes)}/{len(routes)} done')
    return failed
