"""The signs our networks in use read on every drive the brain learns from: the answers of its next limit and the metres
to it where nobody confirmed the frames in Sign review (brain_targets.reviewed_signs), his rule of 8 Nov 2026: "every
dashcam drive, read with our cutter, classifier and applicability; where I confirmed something different, take mine".

The networks are those in use (Models tab, models/active.json), as Our networks propose in Prepare: the cutter finds the
boxes, the classifier reads each, the applicability network says whom each speed sign applies to (nnslr_tools.
_chain_proposer, on the GPU environment). They read the drive's pictures at 1 a second (derived/frames, the frames of
Sign review: a frame he confirmed is the same frame, and his answer replaces theirs). Nothing in Sign review changes: its
proposals stay as they were, and a drive he has started reviewing is read here all the same.

Per drive, derived/brain_signs/<route>.jsonl: a first line with what read it (CACHE_VERSION, the three networks with the
size and time of their weights, the pictures' fingerprint), then one line per picture (frame_key, segment_index,
decoded_frame_index, output_index, detections as in a report). Read again when a network in use or the pictures change,
never otherwise. The signs the car is given (brain_targets.sign_inputs) never come from here: the car has no
applicability network, and the brain is given what the car reads."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from nnslr_tools.manifest import parse_route_id
from nnslr_tools.sign_reads import READ_INDEX_BASE

CACHE_RELATIVE = Path('derived') / 'brain_signs'
CACHE_VERSION = 1
PARTS = ('cutter', 'classifier', 'decider')
# What a picture keeps: enough to put it on the drive's clock and its boxes.
FRAME_FIELDS = ('frame_key', 'segment_index', 'decoded_frame_index', 'output_index')


def cache_path(root: Path, route: str) -> Path:
    parse_route_id(route)
    return Path(root) / CACHE_RELATIVE / f'{route}.jsonl'


def networks_stamp(model: dict) -> dict:
    """{part: {id, size, mtime_ns}} of the three networks of preannotate.our_networks: a part trained again in place
    reads the drives again."""
    found = {}
    for part in PARTS:
        stat = Path(model[f'{part}_path']).stat()
        found[part] = {'id': model[part], 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return found


def pictures_stamp(frames: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[frame['frame_key'], frame.get('frame_sha256')] for frame in frames]).encode()
                          ).hexdigest()[:16]


def pictures(root: Path, route: str) -> list[dict]:
    """The drive's pictures at 1 a second (the frames of the comma's reads and those saved between the seconds left out)."""
    from nnslr_tools.preannotate import discover_frames
    return [frame for frame in discover_frames(root, route) if frame['output_index'] < READ_INDEX_BASE]


def load(root: Path, route: str) -> dict | None:
    """{'read_by': the first line, 'rows': [...]} of a drive, or None when it was never read."""
    path = cache_path(root, route)
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        return None
    if not lines:
        return None
    head = json.loads(lines[0])
    if head.get('version') != CACHE_VERSION:
        return None
    return {'read_by': head, 'rows': [json.loads(line) for line in lines[1:] if line.strip()]}


def _current(root: Path, route: str, stamp: dict, frames: list[dict]) -> bool:
    try:
        with cache_path(root, route).open(encoding='utf-8') as handle:
            head = json.loads(handle.readline() or '{}')
    except (FileNotFoundError, ValueError):
        return False
    return (head.get('version') == CACHE_VERSION and head.get('networks') == stamp
            and head.get('pictures') == pictures_stamp(frames))


def stamp_in_use(root: Path) -> dict | None:
    """networks_stamp of Our networks in use, None when they are not whole: a reading by other networks is not used
    (brain_targets.reviewed_signs)."""
    from nnslr_tools.preannotate import PreannotationError, our_networks
    try:
        return networks_stamp(our_networks(root))
    except (PreannotationError, OSError):
        return None


def unread(root: Path, routes: list[str]) -> int | None:
    """How many of routes the networks in use have not read (never, or by other networks), for Train › Brain before the
    click; None when Our networks are not whole. Without the pictures' fingerprint: the training reads a drive whose
    pictures changed again all the same."""
    stamp = stamp_in_use(root)
    if stamp is None:
        return None
    count = 0
    for route in routes:
        try:
            with cache_path(root, route).open(encoding='utf-8') as handle:
                head = json.loads(handle.readline() or '{}')
        except (FileNotFoundError, ValueError):
            head = {}
        count += head.get('version') != CACHE_VERSION or head.get('networks') != stamp
    return count


def reader_setup(root: Path) -> dict:
    """The proposer_setup of Our networks in use, without preview pictures. Raises preannotate.PreannotationError when
    they cannot run (a part not in use, no GPU environment). The GPU environment is NNSLR_GPU_VENV when set, as for the
    brain's training (scripts/run_brain.sh)."""
    from nnslr_tools.preannotate import proposer_setup
    venv = os.environ.get('NNSLR_GPU_VENV')
    setup = proposer_setup('ours', Path(root), python_executable=str(Path(venv) / 'bin' / 'python') if venv else None)
    setup['request'] = {**setup['request'], 'previews': False}
    return setup


def read_drives(root: Path, routes: list[str], *, setup: dict | None = None, log=print) -> dict[str, str]:
    """Reads with the networks of setup (reader_setup) every drive of routes not read by them yet: {route: 'read' |
    'current' | why not}. Raises preannotate.PreannotationError when a drive's reading fails: its work folder stays
    beside the readings, with the worker's request."""
    from nnslr_tools.preannotate import PreannotationError, run_proposer_worker
    root = Path(root)
    setup = setup or reader_setup(root)
    stamp = networks_stamp(setup['model'])
    log('Our networks in use: ' + ', '.join(f"{part} {stamp[part]['id']}" for part in PARTS))
    found = {}
    for count, route in enumerate(routes, 1):
        try:
            frames = pictures(root, route)
        except PreannotationError as exc:
            found[route] = f'no pictures: {exc}'
            log(f'brain signs {count}/{len(routes)} {route}: {found[route]}')
            continue
        if _current(root, route, stamp, frames):
            found[route] = 'current'
            log(f'brain signs {count}/{len(routes)} {route}: read by these networks already')
            continue
        target = cache_path(root, route)
        target.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f'.{route}-', dir=target.parent))
        rows = run_proposer_worker(setup, root, frames, work, f'brain-signs-{uuid.uuid4().hex[:8]}', route)
        head = {'version': CACHE_VERSION, 'networks': stamp, 'pictures': pictures_stamp(frames),
                'confidence': setup['confidence']}
        partial = work / 'signs.jsonl'
        with partial.open('w', encoding='utf-8') as handle:
            handle.write(json.dumps(head) + '\n')
            for row in rows:
                handle.write(json.dumps({**{field: row[field] for field in FRAME_FIELDS},
                                         'detections': row['detections']}, allow_nan=False) + '\n')
        os.replace(partial, target)
        shutil.rmtree(work, ignore_errors=True)
        found[route] = 'read'
        signs = sum(1 for row in rows for box in row['detections'] if box.get('applicability') == 'own_road')
        log(f'brain signs {count}/{len(routes)} {route}: {len(rows)} pictures read, {signs} boxes of our road')
    return found
