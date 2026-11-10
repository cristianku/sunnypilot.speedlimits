"""The signs our networks in use read on every drive the brain learns from: the answers of its next limit and the metres
to it where nobody confirmed the frames in Sign review (brain_targets.reviewed_signs), his rule of 8 Nov 2026: "every
dashcam drive, read with our cutter, classifier and applicability; where I confirmed something different, take mine".

The networks are those in use (Models tab, models/active.json; preannotate.our_networks), as Our networks propose in Prepare: the cutter finds the
boxes, the classifier reads each, the applicability network says whom each speed sign applies to (nnslr_tools.
_chain_proposer, on the GPU environment). They read the drive's pictures at 1 a second (derived/frames, the frames of
Sign review: a frame he confirmed is the same frame, and his answer replaces theirs). Nothing in Sign review changes: its
proposals stay as they were, and a drive he has started reviewing is read here all the same.

Per drive, derived/brain_signs/<route>.jsonl: a first line with what read it (CACHE_VERSION, the three networks with the
size and time of their weights, the pictures' fingerprint), then one line per picture (frame_key, segment_index,
decoded_frame_index, output_index, detections as in a report). Read again when one of these networks or the pictures change,
never otherwise. The networks are those in use, or the three chosen in Train › Brain (sign_networks: reader_setup's
chosen), the same that read the brain's inputs. Those inputs (brain_targets.sign_inputs) never come from here: the
brain is given what the car reads, its vision.py and applicability.py on the road video (road_type_signs).

Several drives are read at once, each by a worker of its own on a card of those given, in turn (#36, 10 Nov 2026): one
worker keeps about 3 cores and 500 MB of a card busy, so one drive at a time left the GPU nearly idle."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nnslr_tools.manifest import parse_route_id
from nnslr_tools.sign_reads import READ_INDEX_BASE

CACHE_RELATIVE = Path('derived') / 'brain_signs'
CACHE_VERSION = 1
PARTS = ('cutter', 'classifier', 'decider')
# What a picture keeps: enough to put it on the drive's clock and its boxes.
FRAME_FIELDS = ('frame_key', 'segment_index', 'decoded_frame_index', 'output_index')
# A drive's worker keeps about 3 cores busy (its own and the 2 that load its pictures): one drive at once per DRIVE_CORES
# cores, the rest for the server.
DRIVE_CORES = 4
# The last lines of a failed worker's log that its error shows.
FAILED_LINES = 20


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


def stamp_in_use(root: Path, chosen: dict | None = None) -> dict | None:
    """networks_stamp of Our networks (preannotate.our_networks; chosen: those three instead, sign_networks), None when
    they are not whole: a reading by other networks is not used (brain_targets.reviewed_signs)."""
    from nnslr_tools.preannotate import PreannotationError, our_networks
    try:
        return networks_stamp(our_networks(root, chosen))
    except (PreannotationError, OSError):
        return None


def unread(root: Path, routes: list[str], chosen: dict | None = None) -> int | None:
    """How many of routes Our networks (chosen: those three) have not read (never, or by other networks), for Train ›
    Brain before the click; None when they are not whole. Without the pictures' fingerprint: the training reads a drive
    whose pictures changed again all the same."""
    stamp = stamp_in_use(root, chosen)
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


def reader_setup(root: Path, chosen: dict | None = None) -> dict:
    """The proposer_setup of Our networks (chosen: these three instead, {part: id}, sign_networks), without preview
    pictures. Raises preannotate.PreannotationError when they cannot run (a part not in use, no GPU environment). The GPU
    environment is NNSLR_GPU_VENV when set, as for the brain's training (scripts/run_brain.sh)."""
    from nnslr_tools.preannotate import proposer_setup
    venv = os.environ.get('NNSLR_GPU_VENV')
    setup = proposer_setup('ours', Path(root), python_executable=str(Path(venv) / 'bin' / 'python') if venv else None,
                           networks=chosen)
    # Always on a card: a worker without one stops instead of reading on the CPU.
    setup['request'] = {**setup['request'], 'previews': False, 'require_gpu': True}
    return setup


def default_jobs(drives: int) -> int:
    """How many drives are read at once by default: one per DRIVE_CORES cores, at least one, at most drives."""
    return max(1, min(drives, (os.cpu_count() or DRIVE_CORES) // DRIVE_CORES))


def read_drives(root: Path, routes: list[str], *, setup: dict | None = None, log=print, jobs: int | None = None,
                gpus: list[str] | None = None) -> dict[str, str]:
    """Reads with the networks of setup (reader_setup) every drive of routes not read by them yet: {route: 'read' |
    'current' | why not}. jobs drives at once (default_jobs), each by a worker of its own on the next card of gpus in
    turn (nvidia-smi's numbers; None: every worker on the first card this process sees), its lines in its own log. Raises
    preannotate.PreannotationError when a drive's reading fails, with the end of its worker's log: no drive starts after
    it, those under way are finished and kept, and its work folder stays beside the readings with the worker's request
    and log."""
    from nnslr_tools.preannotate import PreannotationError
    root = Path(root)
    # A drive named twice is read once: two workers would write the same reading.
    routes = list(dict.fromkeys(routes))
    setup = setup or reader_setup(root)
    stamp = networks_stamp(setup['model'])
    log('Our networks: ' + ', '.join(f"{part} {stamp[part]['id']}" for part in PARTS))
    found, todo, count = {}, [], 0
    for route in routes:
        try:
            frames = pictures(root, route)
        except PreannotationError as exc:
            count += 1
            found[route] = f'no pictures: {exc}'
            log(f'brain signs {count}/{len(routes)} {route}: {found[route]}')
            continue
        if _current(root, route, stamp, frames):
            count += 1
            found[route] = 'current'
            log(f'brain signs {count}/{len(routes)} {route}: read by these networks already')
            continue
        todo.append((route, frames))
    if not todo:
        return found
    jobs = max(1, min(jobs or default_jobs(len(todo)), len(todo)))
    cards = [str(card) for card in gpus] if gpus else [None]
    # A place per drive at once, each with its card: the drives under way are spread evenly over the cards.
    places: queue.SimpleQueue = queue.SimpleQueue()
    for place in range(jobs):
        places.put(cards[place % len(cards)])
    on = '' if cards == [None] else f", on the card{'s' if len(cards) > 1 else ''} {', '.join(cards)}"
    log(f"brain signs: {len(todo)} drive{'s' if len(todo) > 1 else ''} to read, {jobs} at a time{on}")

    # Set by the drive that fails, before its error reaches this thread: no drive starts after it.
    stop = threading.Event()

    def read(route: str, frames: list[dict]) -> tuple[list[dict] | None, str | None]:
        if stop.is_set():
            return None, None
        card = places.get()
        try:
            return _read_drive(root, setup, stamp, route, frames, card), card
        except BaseException:
            stop.set()
            raise
        finally:
            places.put(card)

    failure: BaseException | None = None
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(read, route, frames): route for route, frames in todo}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            route = futures[future]
            try:
                rows, card = future.result()
            except Exception as exc:  # noqa: BLE001 - the first failure is raised once the drives under way are kept
                if failure is None:
                    failure = exc
                    for other in futures:
                        other.cancel()
                else:
                    log(f'brain signs {route}: failed as well: {exc}')
                continue
            if rows is None:
                continue
            count += 1
            found[route] = 'read'
            signs = sum(1 for row in rows for box in row['detections'] if box.get('applicability') == 'own_road')
            log(f'brain signs {count}/{len(routes)} {route}: {len(rows)} pictures read, {signs} boxes of our road'
                + ('' if card is None else f', card {card}'))
    if failure is not None:
        raise failure
    return found


def _read_drive(root: Path, setup: dict, stamp: dict, route: str, frames: list[dict], card: str | None) -> list[dict]:
    """One drive read by a worker on card (None: the first card this process sees), its reading written: its rows."""
    from nnslr_tools.preannotate import PreannotationError, run_proposer_worker
    target = cache_path(root, route)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{route}-', dir=target.parent))
    worker_log = work / 'worker.log'
    try:
        rows = run_proposer_worker(setup, root, frames, work, f'brain-signs-{uuid.uuid4().hex[:8]}', route, gpu=card,
                                   log_path=worker_log)
    except PreannotationError as exc:
        raise PreannotationError(f'{route}: {exc}{_last_lines(worker_log)}') from exc
    if rows and (rows[0].get('inference') or {}).get('provider') == 'CPUExecutionProvider':
        raise PreannotationError(f'{route}: the worker read it on the CPU, never on a card; its log: {worker_log}')
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
    return rows


def _last_lines(path: Path) -> str:
    """The end of a worker's log, for its error ('' without one)."""
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()[-FAILED_LINES:]
    except OSError:
        return ''
    return '; the end of its log:' + ''.join('\n  ' + line for line in lines) if lines else ''
