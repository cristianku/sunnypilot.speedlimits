"""The front wipers of a drive, from the raw CAN of its rlogs: what tells the brain that the road is wet.

What it is for. A number valid only on a wet road (the class maximum_speed_wet) means nothing without knowing that it rains,
and the picture alone shows it too rarely to learn. The car says it: on a Peugeot the message 0x452 (HS2_DAT_MDD_CMD_452,
FRONT_WIPER_STATUS, the two top bits of its first byte) comes 20 times a second, and the rlog keeps it. No drive's log has it
as a signal of its own, so it is read here, offline, once per drive, and the car reads the same bits into
carStateSP.frontWipers (brain_inputs.WIPERS: 0 off, 1 on, 2 low, 3 high).

Per drive, data/derived/drive_wipers/<route>.json: version, the segments read with the hash of their rlog, the cars, and
series, [[t_ns, state], ...], a point at each change on the comma's clock, as the series of a drive's logs. Each segment ends
with None, so that nothing carries over a gap, and a drive of a car that is not a Peugeot or Citroën has no series at all:
another car may use the same address for something else, so unknown it stays.

A separate cache and not one more series of drive_extract: that would read every drive again and touch the scans of Drive
review, for one signal of a few drives. Reading needs capnp, so it runs in .venv (nnslr drive-wipers in scripts/run_brain.sh);
.venv-gpu only reads the file (load_wipers).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from nnslr_tools import drive_extract, drive_logs

VERSION = 1
WIPERS_RELATIVE = Path('derived') / 'drive_wipers'
ADDRESS = 0x452
CAR_PREFIX = 'PSA_'
SENT_BY_PANDA = 128


def wipers_path(root: Path, route: str) -> Path:
    return root / WIPERS_RELATIVE / f'{drive_logs.check_route(route)}.json'


def load_wipers(root: Path, route: str) -> list:
    """The drive's series, [] when it was not read: every frame unknown."""
    path = wipers_path(root, route)
    if not path.is_file():
        return []
    record = json.loads(path.read_text(encoding='utf-8'))
    return record['series'] if record.get('version') == VERSION else []


def read_segment(path: Path, schema: Path) -> dict:
    """{'car': the fingerprint in this rlog or None, 'series': [[t_ns, state], ...] at each change, then [last, None]}."""
    series: list = []
    car = None
    last = None
    for message in drive_extract._messages(path, schema, schema):
        try:
            which = message.which()
        except Exception:  # a service this schema does not know
            continue
        if which == 'carParams':
            car = car or message.carParams.carFingerprint
        elif which == 'can':
            for frame in message.can:
                if frame.address == ADDRESS and frame.src < SENT_BY_PANDA and len(frame.dat):
                    last = int(message.logMonoTime)
                    state = frame.dat[0] >> 6 & 3
                    if not series or series[-1][1] != state:
                        series.append([last, state])
    return {'car': car, 'series': [*series, [last, None]] if series else []}


def joined(parts: list[dict]) -> list:
    """The segments' series as one, in order. The logger closes a minute a little after the next has begun: a point of that
    overlap goes at the last time already there, as in drive_logs.merge, so that the end of a minute never hides the next."""
    series: list = []
    for part in parts:
        for t, state in part['series']:
            series.append([max(t, series[-1][0]) if series else t, state])
    return series


def _read(job: tuple[str, str]) -> dict:
    return read_segment(Path(job[0]), Path(job[1]))


def build_wipers(root: Path, routes: list[str], *, workers: int | None = None, log=lambda line: None) -> dict[str, dict]:
    """Every route's record; one whose rlogs are the ones already read is not read again."""
    schema = drive_logs.schema_dir()
    records = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for route in routes:
            logs = {row['segment']: row for row in reversed(drive_logs.discover(root, route)) if row['kind'] == 'rlog'}
            logs = [logs[segment] for segment in sorted(logs)]
            read = {str(row['segment']): drive_logs.sha256_file(Path(row['path'])) for row in logs}
            path = wipers_path(root, route)
            before = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
            if before is not None and before.get('version') == VERSION and before.get('segments') == read:
                records[route] = before
                continue
            parts = list(pool.map(_read, [(row['path'], str(schema)) for row in logs]))
            cars = sorted({part['car'] for part in parts if part['car']})
            ours = any(car.startswith(CAR_PREFIX) for car in cars)
            records[route] = {'version': VERSION, 'segments': read, 'cars': cars, 'series': joined(parts) if ours else []}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(records[route]), encoding='utf-8')
            log(f'{route}: {len(logs)} segments read')
    return records
