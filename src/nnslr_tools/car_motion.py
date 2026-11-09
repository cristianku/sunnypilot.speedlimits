"""The car's own motion at every second of a drive, from its logs: how much it turned, its steering wheel, its
blinkers and its speed. The inputs of the road-type network beside the camera, the speed and the sign, so its memory
knows when the car turned at an intersection (road_type_net, MOTION_INPUTS).

Only the car's own signals, never the map: the comma's vision limit (speedvisiond) reads the camera and the car, and
the map is weighed against it only by sunnypilot's resolver. OpenStreetMap makes training labels only (junctions.py).

Per segment the log reader (the log Python of gps.log_python, comma_fetch_log) keeps, from the rlog (the qlog when
there is none):

- deviceMotion (livePose in older openpilot; 20 Hz): angularVelocityDevice.z and its valid flag. The device frame's z
  points down, so the yaw rate of the car (positive to the left, as carState) is -z;
- carState (100 Hz, every MOTION_DECIMATE-th kept, the blinkers of the skipped ones kept too): vEgo, yawRate (the
  car's own sensor, 0 on cars without one), steeringAngleDeg (positive to the left), leftBlinker, rightBlinker.

logMonoTime runs on across the segments of a drive, so the samples of all its segments are joined and every second is
summarised from the whole drive: frame k of segment s (its 1 Hz frame, as nnslr gps: the first road frame of the
segment plus k s) takes the step (t - 1 s, t]:

- yaw: the mean yaw rate over the step in deg/s, which is the heading change of the step in degrees, and the largest
  |yaw rate| in it; from deviceMotion (at least MIN_SHARE of its samples valid), else from carState.yawRate when the
  car gives one; else unknown;
- steer: the mean steeringAngleDeg over the step (unknown without carState);
- blinkers: left and right, each "on at some sample of the last BLINKER_SECONDS" (a blinker lamp flashes, the lever
  is held for seconds);
- v_ego: carState.vEgo of the last sample at or before t (within VEGO_MAX_AGE).

Cached per drive in derived/car_motion/<route>.json, read again when a log or VERSION changed.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nnslr_tools.gps import HELPERS, log_python

MOTION_RELATIVE = Path('derived') / 'car_motion'
RAW_RELATIVE = Path('raw') / 'routes'
VERSION = 1
SECONDS_PER_SEGMENT = 60
MOTION_DECIMATE = 5  # carState at 100 Hz kept at 20 Hz (the blinkers of the skipped messages are kept)
STEP_SECONDS = 1.0
BLINKER_SECONDS = 2.0
MIN_SHARE = 0.5  # of the samples expected in a step (20 a second) for its yaw or steer to be known
SAMPLE_HZ = 20.0
VEGO_MAX_AGE = 0.5
CAR_YAW_MIN = 1e-4  # rad/s: a car whose yawRate never leaves 0 has no sensor behind it

# The road-type network's motion input, (MOTION_INPUTS,) float32 per second (motion_features).
MOTION_FIELDS = ('heading_change', 'yaw_max', 'yaw_known', 'steer', 'steer_known', 'left_blinker', 'right_blinker',
                 'blinker_known')
MOTION_INPUTS = len(MOTION_FIELDS)
YAW_SCALE = 30.0  # deg/s (or deg of heading over the 1 s step)
YAW_CLIP = 3.0
STEER_SCALE = 180.0  # deg of the steering wheel
STEER_CLIP = 4.0
NO_MOTION = (0.0,) * MOTION_INPUTS

# Run by the log Python: per segment log, the first encoded road frame, deviceMotion and carState, as one JSON line.
_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import comma_fetch_log
from pathlib import Path
ENCODE = ('roadEncodeIdx', 'narrowRoadEncodeIdx')
POSE = ('deviceMotion', 'livePose')
DECIMATE = int(sys.argv[2])
for name in sys.argv[3:]:
    first, pose, car = None, [], []
    count, left, right = 0, False, False
    try:
        for message in comma_fetch_log._messages(Path(name)):
            which = message.which()
            if which in ENCODE:
                if first is None and int(getattr(message, which).segmentId) == 0:
                    first = int(message.logMonoTime)
            elif which in POSE:
                w = getattr(message, which).angularVelocityDevice
                pose.append([int(message.logMonoTime), round(float(w.z), 6), int(bool(w.valid))])
            elif which == 'carState':
                c = message.carState
                left, right = left or bool(c.leftBlinker), right or bool(c.rightBlinker)
                count += 1
                if count % DECIMATE == 0:
                    car.append([int(message.logMonoTime), round(float(c.vEgo), 4), round(float(c.yawRate), 6),
                                round(float(c.steeringAngleDeg), 3), int(left), int(right)])
                    left = right = False
        error = None
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:200]
    print(json.dumps({'log': name, 'first_frame_mono': first, 'pose': pose, 'car': car, 'error': error}), flush=True)
"""


class MotionError(RuntimeError):
    pass


def motion_path(root: Path, route: str) -> Path:
    return Path(root) / MOTION_RELATIVE / f'{route}.json'


def segment_logs(root: Path, route: str) -> dict[int, Path]:
    """{segment: its rlog (the qlog when there is none)} of a drive."""
    folder = Path(root) / RAW_RELATIVE / route
    found = {}
    for directory in folder.iterdir() if folder.is_dir() else []:
        if not directory.name.isdigit():
            continue
        for name in ('rlog.zst', 'rlog.bz2', 'rlog', 'qlog.zst', 'qlog.bz2', 'qlog'):
            if (directory / name).is_file():
                found[int(directory.name)] = directory / name
                break
    return dict(sorted(found.items()))


def _stamp(logs: dict[int, Path]) -> dict[str, list]:
    return {str(segment): [path.name, path.stat().st_size, int(path.stat().st_mtime)] for segment, path in logs.items()}


def read_logs(paths: list[Path], *, python: str | None = None) -> dict[str, dict]:
    """{log path: {'first_frame_mono', 'pose': [[mono, z, valid]], 'car': [[mono, vEgo, yawRate, steer, left, right]],
    'error'}}."""
    if not paths:
        return {}
    proc = subprocess.run([python or log_python(), '-c', _CHILD, str(HELPERS), str(MOTION_DECIMATE)]
                          + [str(p) for p in paths], capture_output=True, text=True, check=False, timeout=1800)
    if proc.returncode != 0:
        raise MotionError('log reader failed: ' + proc.stderr.strip()[-300:])
    result = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            item = json.loads(line)
            result[item['log']] = item
    return result


def _window(times: np.ndarray, start: float, end: float) -> slice:
    """The samples with start < time <= end (times sorted)."""
    return slice(int(np.searchsorted(times, start, side='right')), int(np.searchsorted(times, end, side='right')))


def summarise(frame_times: np.ndarray, pose: np.ndarray, car: np.ndarray) -> list[dict]:
    """The summary of every second (see the module's text). frame_times: seconds (any clock, the samples' too);
    pose (n, 3): time, angularVelocityDevice.z rad/s, valid; car (m, 6): time, vEgo, yawRate rad/s, steeringAngleDeg,
    leftBlinker, rightBlinker. Returns per frame {'yaw_dps', 'yaw_max_dps', 'yaw_source' ('device', 'car' or None),
    'steer_deg', 'left', 'right', 'car_known', 'v_ego'} (None where unknown)."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1, 3)
    car = np.asarray(car, dtype=np.float64).reshape(-1, 6)
    pose = pose[np.argsort(pose[:, 0], kind='stable')]
    car = car[np.argsort(car[:, 0], kind='stable')]
    expected = SAMPLE_HZ * STEP_SECONDS * MIN_SHARE
    car_has_yaw = bool(len(car)) and bool(np.any(np.abs(car[:, 2]) > CAR_YAW_MIN))
    out = []
    for t in np.asarray(frame_times, dtype=np.float64):
        item = {'yaw_dps': None, 'yaw_max_dps': None, 'yaw_source': None, 'steer_deg': None, 'left': None,
                'right': None, 'car_known': False, 'v_ego': None}
        step = pose[_window(pose[:, 0], t - STEP_SECONDS, t)]
        valid = step[step[:, 2] > 0, 1] if len(step) else step
        cars = car[_window(car[:, 0], t - STEP_SECONDS, t)]
        if len(valid) >= expected:
            yaw = -np.degrees(valid)
            item.update(yaw_dps=float(yaw.mean()), yaw_max_dps=float(np.abs(yaw).max()), yaw_source='device')
        elif car_has_yaw and len(cars) >= expected:
            yaw = np.degrees(cars[:, 2])
            item.update(yaw_dps=float(yaw.mean()), yaw_max_dps=float(np.abs(yaw).max()), yaw_source='car')
        if len(cars) >= expected:
            item.update(steer_deg=float(cars[:, 3].mean()), car_known=True)
            recent = car[_window(car[:, 0], t - BLINKER_SECONDS, t)]
            item.update(left=bool(recent[:, 4].any()), right=bool(recent[:, 5].any()))
        last = int(np.searchsorted(car[:, 0], t, side='right')) - 1 if len(car) else -1
        if last >= 0 and t - car[last, 0] <= VEGO_MAX_AGE:
            item['v_ego'] = float(car[last, 1])
        out.append(item)
    return out


def _rounded(item: dict) -> list:
    """A second's summary as a compact row: yaw_dps, yaw_max_dps, yaw_source, steer_deg, left, right, car_known,
    v_ego (ROW)."""
    def r(value, digits):
        return None if value is None else round(value, digits)
    return [r(item['yaw_dps'], 3), r(item['yaw_max_dps'], 3), item['yaw_source'], r(item['steer_deg'], 2),
            None if item['left'] is None else int(item['left']), None if item['right'] is None else int(item['right']),
            int(item['car_known']), r(item['v_ego'], 3)]


ROW = ('yaw_dps', 'yaw_max_dps', 'yaw_source', 'steer_deg', 'left', 'right', 'car_known', 'v_ego')


def build_route(root: Path, route: str, logs: dict[int, Path], read: dict[str, dict]) -> dict:
    """The record of a drive from its segments' readings (read_logs): {'route', 'version', 'logs', 'row',
    'segments': {segment: {'error', 'seconds': [ROW per second 0..59] or []}}, 'coverage'}."""
    pose, car, firsts, errors = [], [], {}, {}
    for segment, path in logs.items():
        item = read.get(str(path)) or {'error': 'not read'}
        if item.get('error'):
            errors[segment] = item['error']
        if item.get('first_frame_mono') is not None:
            firsts[segment] = item['first_frame_mono']
        pose += item.get('pose') or []
        car += item.get('car') or []
    pose_array = np.array(pose, dtype=np.float64).reshape(-1, 3)
    car_array = np.array(car, dtype=np.float64).reshape(-1, 6)
    pose_array[:, 0] /= 1e9
    car_array[:, 0] /= 1e9
    segments, counts = {}, {'seconds': 0, 'yaw_device': 0, 'yaw_car': 0, 'car': 0, 'v_ego': 0}
    for segment in logs:
        if segment not in firsts:
            segments[str(segment)] = {'error': errors.get(segment) or 'no road frame in the log', 'seconds': []}
            continue
        times = firsts[segment] / 1e9 + np.arange(SECONDS_PER_SEGMENT, dtype=np.float64)
        rows = [_rounded(item) for item in summarise(times, pose_array, car_array)]
        segments[str(segment)] = {'error': errors.get(segment), 'seconds': rows}
        counts['seconds'] += len(rows)
        counts['yaw_device'] += sum(1 for row in rows if row[2] == 'device')
        counts['yaw_car'] += sum(1 for row in rows if row[2] == 'car')
        counts['car'] += sum(1 for row in rows if row[6])
        counts['v_ego'] += sum(1 for row in rows if row[7] is not None)
    return {'route': route, 'version': VERSION, 'logs': _stamp(logs), 'row': list(ROW), 'segments': segments,
            'coverage': counts}


def load_motion(root: Path, route: str) -> dict | None:
    path = motion_path(root, route)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    return record if record.get('version') == VERSION else None


def build_motion(root: Path, routes: list[str], *, workers: int | None = None, python: str | None = None,
                 again: bool = False, log=print) -> dict[str, dict]:
    """The motion of every drive (cached; read again when a log changed): {route: record}. The segments of all drives
    are read in parallel, workers log readers at a time."""
    root = Path(root)
    records, todo = {}, {}
    for route in routes:
        logs = segment_logs(root, route)
        record = None if again else load_motion(root, route)
        if record and record.get('logs') == _stamp(logs):
            records[route] = record
        else:
            todo[route] = logs
    paths = [path for logs in todo.values() for path in logs.values()]
    if paths:
        workers = workers or max(1, min(len(paths), (os.cpu_count() or 4) // 4))
        log(f'car motion: reading {len(paths)} segment logs of {len(todo)} drives, {workers} at a time')
        groups = [[path] for path in paths]
        read, done = {}, 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(lambda group: read_logs(group, python=python), groups):
                read.update(part)
                done += 1
                if done % 25 == 0 or done == len(groups):
                    log(f'car motion: {done}/{len(groups)} segments')
        for route, logs in todo.items():
            record = build_route(root, route, logs, read)
            path = motion_path(root, route)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, separators=(',', ':')), encoding='utf-8')
            temporary.replace(path)
            records[route] = record
    return records


def frame_motion(record: dict | None, segment: int, index: int) -> dict | None:
    """A second's summary as a dict (ROW's names), None when the drive or the second was not read."""
    if not record:
        return None
    rows = (record.get('segments', {}).get(str(segment)) or {}).get('seconds') or []
    if not 0 <= index < len(rows):
        return None
    return dict(zip(record.get('row', ROW), rows[index]))


def motion_features(item: dict | None) -> tuple[float, ...]:
    """The motion input of a second (MOTION_FIELDS), NO_MOTION (all unknown) without its summary: heading change of
    the step / YAW_SCALE and the largest |yaw rate| / YAW_SCALE (clipped to YAW_CLIP), yaw known; the mean steering
    angle / STEER_SCALE (clipped to STEER_CLIP), steer known; the left and the right blinker on in the last
    BLINKER_SECONDS, blinkers known. Left is positive, as in carState."""
    if not item:
        return NO_MOTION
    out = list(NO_MOTION)
    if item.get('yaw_dps') is not None and math.isfinite(item['yaw_dps']):
        out[0] = float(np.clip(item['yaw_dps'] / YAW_SCALE, -YAW_CLIP, YAW_CLIP))
        out[1] = float(np.clip((item.get('yaw_max_dps') or 0.0) / YAW_SCALE, 0.0, YAW_CLIP))
        out[2] = 1.0
    if item.get('car_known'):
        if item.get('steer_deg') is not None and math.isfinite(item['steer_deg']):
            out[3] = float(np.clip(item['steer_deg'] / STEER_SCALE, -STEER_CLIP, STEER_CLIP))
            out[4] = 1.0
        out[5], out[6], out[7] = float(bool(item.get('left'))), float(bool(item.get('right'))), 1.0
    return tuple(out)


def coverage_text(record: dict) -> str:
    counts = record.get('coverage') or {}
    seconds = max(1, counts.get('seconds', 0))
    errors = [f"{segment}: {item['error']}" for segment, item in record.get('segments', {}).items() if item.get('error')]
    return (f"{record['route']}: {counts.get('seconds', 0)} s; yaw from deviceMotion {counts.get('yaw_device', 0) / seconds:.0%}"
            f", from carState.yawRate {counts.get('yaw_car', 0) / seconds:.0%}; carState {counts.get('car', 0) / seconds:.0%}"
            f", vEgo {counts.get('v_ego', 0) / seconds:.0%}" + (f"; {len(errors)} segments with errors ({errors[0]})"
                                                               if errors else ''))
