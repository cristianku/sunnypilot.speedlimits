"""The signs on every frame of a drive's video, for Speed review's picture: the boxes follow a sign as the car moves (20
frames a second), not only on the frame a second Sign review extracted (drive_review.sign_frames, derived/frames).

The same chain as Our networks propose with in Prepare (nnslr_tools._chain_proposer), without the decider: the cutter in
use (a SignNet) finds the boxes in the frame's top band (signnet._band_of, its best CAR_MAX_BOXES of score MIN_SCORE or
more, chain._signnet_boxes), the classifier in use reads each box (training.crop_rectangle, cut from the frame at full
definition when the drive has one, x 2), "not a sign" is dropped and the overlaps suppressed (preannotate.
suppress_overlaps). The decider moves no box and has no speed a frame here: it is not run. Its frames are decoded straight
from the minutes' raw HEVC on the CPU (decode_minute: rgb24, bit for bit the pictures of Sign review), one ffmpeg a minute
(two for a drive at full definition: the 1344 frame for the cutter, the 2688 one for the classifier's crops), read by a
thread ahead of the networks. Offline, on the GPU of .venv-gpu (nnslr sign-tracks, helpers/sign_tracks.sh, Models › Find the
signs on every frame); never on the car.

Not derived/tracks/ (nnslr tracks): that is one sign followed across the confirmed frames a second. Here every frame of the
video has its boxes, proposals of the networks only.

The key of a box is (segment, k): k the decoded frame 0..frames-1 of raw/routes/<route>/<segment>/fcamera.hevc, the
same as frame round(start(segment) x 20) + k of derived/video/<route>.mp4 (drive_video). The time of the logs never enters it.

derived/sign_tracks/<route>.json, written whole (a temporary file and os.replace), is also the body of
GET /api/speed-review/drives/<route>/sign-track (read: with "current"):
  {format, route, fps, width, height, networks {cutter, classifier: {id, size, mtime_ns}}, min_score, made_at,
   labels [{model_label, family, value_kph, supplementary_panel, plate_hours?, sign_type?}],
   segments [{segment, frames, source [size, mtime], boxes [[k, label index, x0, y0, x1, y1, score %], ...]}]}
Corners in whole pixels of the 1344 x 760 frame; the rows by k, then score from the best; a frame with no box has no row.
frames is the minute's decoded count, the video index's; source its raw minute as drive_video's index says it.
<route>.meta.json beside it holds what the card shows without reading the boxes (summary); <route>.partial.jsonl the
minutes done of a drive being read (a first line with the networks, then a minute a line, its boxes by model_label), so a
stopped run goes on from there. A drive is read again when a network in use or a minute changed, never otherwise.

Torch and numpy are imported only by the reading (Finder, decode_minute): the site imports this module from .venv for
read and summary.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from nnslr_tools import drive_video
from nnslr_tools.media import FRAME_SIZE, FULL_SIZE, REDUCE_FILTER

FORMAT = 1
FPS = drive_video.FPS
WIDTH, HEIGHT = FRAME_SIZE
# The cutter's lowest score, as Prepare's Our networks (the car keeps 0.3 or more: signnet.CAR_MIN_SCORE).
MIN_SCORE = 0.1
# The networks that make the boxes (brain_signs.networks_stamp without the decider).
PARTS = ('cutter', 'classifier')
RELATIVE = Path('derived') / 'sign_tracks'
# A dry run: the first DRY_FRAMES frames of the first minute, on the CPU, into DRY_FOLDER (never served).
DRY_FRAMES = 41
DRY_FOLDER = 'dry'
# Frames decoded ahead of the networks, frames a batch of the cutter, the threads of each ffmpeg.
QUEUE_FRAMES = 64
CUTTER_FRAMES = 32
DECODE_THREADS = 4
# The answer of the API for a drive not read yet: not the generic "Not found.", so the page tells it from an old server.
NO_TRACK = 'This drive has no signs on every frame yet.'


class SignTrackError(RuntimeError):
    """Why a drive cannot be read: the drive fails, the others go on."""


def folder(root: Path) -> Path:
    return Path(root) / RELATIVE


def track_path(root: Path, route: str) -> Path:
    if not drive_video.ROUTE.fullmatch(route):
        raise ValueError('invalid route id: ' + route)
    return folder(root) / f'{route}.json'


def meta_path(path: Path) -> Path:
    return path.with_name(path.name[:-len('.json')] + '.meta.json')


def partial_path(path: Path) -> Path:
    return path.with_name(path.name[:-len('.json')] + '.partial.jsonl')


def networks_of(model: dict) -> dict:
    """{cutter, classifier: {id, size, mtime_ns}} of preannotate.our_networks: a part trained again in place reads the
    drives again."""
    from nnslr_tools.brain_signs import networks_stamp
    return {part: value for part, value in networks_stamp(model).items() if part in PARTS}


def networks_in_use(root: Path) -> dict | None:
    """networks_of Our networks in use, None when they are not whole (the Models tab)."""
    from nnslr_tools.preannotate import PreannotationError, our_networks
    try:
        return networks_of(our_networks(Path(root)))
    except (PreannotationError, OSError):
        return None


def sources(root: Path, route: str) -> dict[int, list[int]]:
    """{segment: [size, mtime]} of the drive's raw minutes now, as drive_video's index records them."""
    return {segment: [size, mtime] for segment, size, mtime in drive_video._signature(drive_video.minutes(root, route))}


def video_routes(root: Path) -> list[str]:
    """The drives with a video (derived/video/<route>.json), in order: what --all reads."""
    found = (path.stem for path in (Path(root) / 'derived' / 'video').glob('*.json'))
    return sorted(route for route in found if drive_video.ROUTE.fullmatch(route))


def label_entry(model_label: str) -> dict:
    """The fields of a classifier class as a proposal of Prepare has them (_chain_proposer): family, value_kph,
    supplementary_panel, plate_hours for one with hours, sign_type for an other sign."""
    from nnslr_tools._chain_proposer import proposal_of, proposal_panel
    from nnslr_tools.sign_types import class_hours, class_parts
    family, value = proposal_of(model_label)
    entry = {'model_label': model_label, 'family': family, 'value_kph': value,
             'supplementary_panel': proposal_panel(model_label)}
    if class_hours(model_label):
        entry['plate_hours'] = class_hours(model_label)
    if family == 'other_sign' and class_parts(model_label)[1]:
        entry['sign_type'] = class_parts(model_label)[1]
    return entry


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _partial_minutes(path: Path, networks: dict) -> dict[int, dict]:
    """The minutes done of a partial file read by these networks: {segment: its line}; none when other networks wrote it."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return {}
    try:
        head = json.loads(lines[0]) if lines else {}
    except ValueError:
        return {}
    if not isinstance(head, dict) or head.get('networks') != networks:
        return {}
    found = {}
    for line in lines[1:]:
        try:
            item = json.loads(line)
        except ValueError:
            continue  # the last line cut by a stop
        if isinstance(item, dict) and isinstance(item.get('segment'), int):
            found[item['segment']] = item
    return found


def track_of(route: str, networks: dict, minutes: list[dict], made_at: str) -> dict:
    """The file of a drive from its minutes as the partial file holds them (boxes [k, model_label, x0, y0, x1, y1, score]):
    the labels numbered as they first appear."""
    labels: dict[str, int] = {}
    segments = []
    for minute in sorted(minutes, key=lambda item: item['segment']):
        boxes = []
        for k, model_label, *rest in minute['boxes']:
            index = labels.setdefault(model_label, len(labels))
            boxes.append([k, index, *rest])
        segments.append({'segment': minute['segment'], 'frames': minute['frames'], 'source': minute['source'],
                         'boxes': boxes})
    return {'format': FORMAT, 'route': route, 'fps': FPS, 'width': WIDTH, 'height': HEIGHT, 'networks': networks,
            'min_score': MIN_SCORE, 'made_at': made_at, 'labels': [label_entry(label) for label in labels],
            'segments': segments}


def minute_rows(found: list[tuple[int, list[dict]]]) -> list[list]:
    """The rows of a minute from the boxes of its frames ((k, detections of Finder.find)): whole pixels clamped to the
    frame, the score in percent, by k then score from the best."""
    rows = []
    for k, detections in found:
        for item in detections:
            x0, y0, x1, y1 = (int(round(value)) for value in item['bbox_xyxy'])
            rows.append([k, item['model_label'], min(max(x0, 0), WIDTH), min(max(y0, 0), HEIGHT),
                         min(max(x1, 0), WIDTH), min(max(y1, 0), HEIGHT), int(round(item['score'] * 100))])
    rows.sort(key=lambda row: (row[0], -row[6]))
    return rows


# -- reading the frames ------------------------------------------------------------------------------------------------
def _frames_of(command: list[str], size: int):
    """A started ffmpeg writing raw frames of size bytes, and a function that reads its next frame (None at the end)."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def next_frame():
        buffer = process.stdout.read(size)  # type: ignore[union-attr]
        return buffer if len(buffer) == size else None
    return process, next_frame


def decode_minute(video: Path, limit: int | None = None) -> Iterator[tuple[Any, Any]]:
    """(the 1344 x 760 frame, the frame at full definition or None) of every frame of a raw minute, in order, as uint8
    arrays (H, W, 3): decoded on the CPU to rgb24 as Sign review's pictures were (media.extract_frames), by a thread that
    keeps QUEUE_FRAMES ahead. A drive at full definition (media.FULL_SIZE): two ffmpeg side by side, the first reduced by
    media.REDUCE_FILTER, the second as recorded; both must end on the same count. limit: only the first frames."""
    import numpy as np
    from nnslr_tools.media import video_dimensions

    full = video_dimensions(video) == FULL_SIZE
    base = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', str(DECODE_THREADS), '-i', str(video), '-vsync', '0']
    tail = [*(['-frames:v', str(limit)] if limit else []), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']
    readers = [_frames_of([*base, *(['-vf', REDUCE_FILTER] if full else []), *tail], WIDTH * HEIGHT * 3)]
    if full:
        readers.append(_frames_of([*base, *tail], FULL_SIZE[0] * FULL_SIZE[1] * 3))
    frames: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
    stop = threading.Event()

    def read() -> None:
        try:
            while not stop.is_set():
                buffers = [next_frame() for _, next_frame in readers]
                if any(buffer is None for buffer in buffers):
                    if not all(buffer is None for buffer in buffers):
                        frames.put(RuntimeError(f'{video}: the two decodes at full definition end on different counts'))
                        return
                    break
                frame = np.frombuffer(buffers[0], np.uint8).reshape(HEIGHT, WIDTH, 3)
                whole = (np.frombuffer(buffers[1], np.uint8).reshape(FULL_SIZE[1], FULL_SIZE[0], 3) if full else None)
                while not stop.is_set():
                    try:
                        frames.put((frame, whole), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            for process, _ in readers:
                if process.wait() != 0:
                    message = process.stderr.read().decode('utf-8', 'replace').strip()  # type: ignore[union-attr]
                    frames.put(RuntimeError(f'{video}: ffmpeg failed: {message[-300:]}'))
                    return
            frames.put(None)
        except Exception as exc:  # noqa: BLE001 - said in the drive's line
            frames.put(exc)

    thread = threading.Thread(target=read, name='sign-tracks-decode', daemon=True)
    thread.start()
    try:
        while True:
            item = frames.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise SignTrackError(str(item))
            yield item
    finally:
        stop.set()
        for process, _ in readers:
            if process.poll() is None:
                process.kill()
        while thread.is_alive():
            try:
                frames.get_nowait()
            except queue.Empty:
                thread.join(0.1)
        for process, _ in readers:
            process.wait()
            process.stdout.close()  # type: ignore[union-attr]
            process.stderr.close()  # type: ignore[union-attr]


# -- the networks ------------------------------------------------------------------------------------------------------
class Finder:
    """The cutter and the classifier in use on a device, as _chain_proposer runs them: find(batch) gives the boxes of each
    frame of a batch, the crops of every frame read by the classifier in batches of up to CLASSIFIER_BATCH."""

    def __init__(self, model: dict, device_name: str):
        import torch
        from nnslr_tools import chain
        chain.report_progress = lambda *args, **kwargs: None
        self.torch = torch
        self.device = chain._device(torch, device_name)
        self.cutter, _, _ = chain._load_cutter({'path': Path(model['cutter_path']), 'id': model['cutter']}, self.device)
        if chain.cutter_kind(self.cutter) != 'signnet':
            raise SignTrackError(f"the cutter in use {model['cutter']} is not a SignNet")
        self.classifier, _, self.classes, self.transform = chain._load_classifier(
            {'path': Path(model['classifier_path']), 'id': model['classifier']}, self.device)
        self.networks = networks_of(model)

    def find(self, batch: list[tuple[int, Any, Any]]) -> list[tuple[int, list[dict]]]:
        """(k, its detections {model_label, class_id, bbox_xyxy, score}) of every (k, 1344 frame, full frame or None)."""
        from PIL import Image
        from nnslr_tools.chain import CLASSIFIER_BATCH, _signnet_boxes
        from nnslr_tools.preannotate import suppress_overlaps
        from nnslr_tools.signnet import CAR_MAX_BOXES, _band_of
        from nnslr_tools.training import crop_rectangle

        torch, cutter = self.torch, self.cutter
        bands = torch.stack([_band_of(Image.fromarray(frame), cutter.band[0]) for _, frame, _ in batch])
        with torch.no_grad():
            heat, shape = cutter(bands.to(self.device).float().div_(255.0))
        heat, shape = heat.float().cpu(), shape.float().cpu()
        crops, owners = [], []
        for b, (_, frame, whole) in enumerate(batch):
            picture = Image.fromarray(frame if whole is None else whole)
            factor = 1 if whole is None else whole.shape[1] // WIDTH
            for item in _signnet_boxes(cutter, heat[b:b + 1], shape[b:b + 1], WIDTH, HEIGHT, MIN_SCORE, CAR_MAX_BOXES):
                rectangle = crop_rectangle(item['box'], WIDTH, HEIGHT)
                crops.append(self.transform(picture.crop(tuple(value * factor for value in rectangle))))
                owners.append((b, item))
        detections: list[list[dict]] = [[] for _ in batch]
        for start in range(0, len(crops), CLASSIFIER_BATCH):
            with torch.no_grad():
                probabilities = torch.softmax(self.classifier(torch.stack(crops[start:start + CLASSIFIER_BATCH])
                                                              .to(self.device)).float(), dim=1)
            for (b, item), index, confidence in zip(owners[start:start + CLASSIFIER_BATCH],
                                                    probabilities.argmax(1).tolist(), probabilities.max(1).values.tolist()):
                label = self.classes[index]
                box = [round(float(value), 1) for value in item['box']]
                # Not a sign is dropped as in Prepare; a box with no area after clamping too (suppress_overlaps refuses it).
                if label == 'not_a_sign' or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detections[b].append({'model_label': label, 'class_id': index, 'bbox_xyxy': box,
                                      'score': round(float(confidence), 4)})
        return [(k, suppress_overlaps(found)) for (k, _, _), found in zip(batch, detections)]


def load_finder(root: Path, device_name: str) -> Finder:
    """The Finder of Our networks in use. Raises SignTrackError when they are not whole."""
    from nnslr_tools.preannotate import PreannotationError, our_networks
    try:
        model = our_networks(Path(root))
    except PreannotationError as exc:
        raise SignTrackError(str(exc)) from exc
    return Finder(model, device_name)


# -- a drive -----------------------------------------------------------------------------------------------------------
def run_drive(root: Path, route: str, finder, *, limit_frames: int | None = None, out_dir: Path | None = None,
              force: bool = False, decoder: Callable[..., Iterator] = decode_minute,
              log: Callable[[str], None] = print) -> dict:
    """Reads a drive with finder (Finder, or anything with .networks and .find) and writes its file: {'state': 'done' |
    'current', 'frames', 'boxes', 'seconds', 'path'}. limit_frames (a dry run): the first frames of the first minute
    only, no resume, no skip. Raises SignTrackError when it cannot: no video of the current minutes, a minute decoded to
    another count than the video's."""
    root = Path(root)
    target = (Path(out_dir) if out_dir else folder(root)) / f'{route}.json'
    track_path(root, route)
    index = drive_video.index(root, route)
    if index is None:
        raise SignTrackError('no video of its current minutes: make its video first (Drive view)')
    now = sources(root, route)
    videos = dict(drive_video.minutes(root, route))
    segments = [item for item in index['segments'] if item['segment'] in now]
    if limit_frames:
        segments = segments[:1]
    networks = finder.networks
    meta, partial = meta_path(target), partial_path(target)
    plan = [[item['segment'], now[item['segment']]] for item in segments]
    if not force and not limit_frames and target.is_file():
        written = _read_json(meta)
        if written and written.get('networks') == networks and written.get('segments') == plan:
            return {'state': 'current', 'path': target, 'frames': written.get('frames', 0),
                    'boxes': written.get('boxes', 0), 'seconds': 0.0}
    done = {} if force or limit_frames else _partial_minutes(partial, networks)
    done = {segment: line for segment, line in done.items()
            if any(item['segment'] == segment and item['frames'] == line.get('frames') and now[segment] == line.get('source')
                   for item in segments)}
    started = time.monotonic()
    if not limit_frames:
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open('w', encoding='utf-8') as handle:
            handle.write(json.dumps({'networks': networks}) + '\n')
            for segment in sorted(done):
                handle.write(json.dumps(done[segment]) + '\n')
    minutes = list(done.values())
    for position, item in enumerate(segments, 1):
        segment, expected = item['segment'], item['frames']
        if segment in done:
            continue
        begun, found, pending, count = time.monotonic(), [], [], 0
        for frame, whole in decoder(videos[segment], limit_frames):
            pending.append((count, frame, whole))
            count += 1
            if len(pending) == CUTTER_FRAMES:
                found += finder.find(pending)
                pending = []
        if pending:
            found += finder.find(pending)
        if not limit_frames and count != expected:
            raise SignTrackError(f'minute {segment}: decoded {count} frames, the video has {expected}')
        rows = minute_rows(found)
        line = {'segment': segment, 'frames': count, 'source': now[segment], 'boxes': rows}
        minutes.append(line)
        if not limit_frames:
            with partial.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(line) + '\n')
        seconds = max(time.monotonic() - begun, 1e-6)
        log(f'minute {position}/{len(segments)}: segment {segment}, {count} frames, {len(rows)} boxes, '
            f'{count / seconds:.1f} fps')
    made_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    track = track_of(route, networks, minutes, made_at)
    frames = sum(minute['frames'] for minute in minutes)
    boxes = sum(len(minute['boxes']) for minute in minutes)
    _write_atomic(target, json.dumps(track, separators=(',', ':')) + '\n')
    _write_atomic(meta, json.dumps({'networks': networks, 'made_at': made_at, 'segments': plan, 'frames': frames,
                                    'boxes': boxes}) + '\n')
    partial.unlink(missing_ok=True)
    return {'state': 'done', 'path': target, 'frames': frames, 'boxes': boxes, 'seconds': time.monotonic() - started}


# -- the check of a dry run against Sign review ------------------------------------------------------------------------
def report_rows(root: Path, route: str) -> list[dict]:
    """The rows of the drive's report in Sign review (derived/preannotations/<route>/latest.json), [] without one."""
    base = Path(root) / 'derived' / 'preannotations' / route
    run = (_read_json(base / 'latest.json') or {}).get('run_id')
    if not isinstance(run, str) or '/' in run or not run:
        return []
    try:
        lines = (base / run / 'preannotations.jsonl').read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def parity(root: Path, track: dict, rows: list[dict]) -> dict:
    """How many of Sign review's frames inside the track (its rows made by the same cutter and classifier, k = 20 N)
    have the same boxes: the same labels, each corner within 0.5 px, the score within 1 percent point. {'frames',
    'matching', 'boxes' (compared)}."""
    networks = track['networks']
    labels = [entry['model_label'] for entry in track['labels']]
    by_frame: dict[tuple[int, int], list] = {}
    frames = {}
    for segment in track['segments']:
        frames[segment['segment']] = segment['frames']
        for k, label, *box in segment['boxes']:
            by_frame.setdefault((segment['segment'], k), []).append([labels[label], *box])
    counted = {'frames': 0, 'matching': 0, 'boxes': 0}
    for row in rows:
        model = row.get('model') or {}
        segment, k = row.get('segment_index'), row.get('decoded_frame_index')
        if model.get('cutter') != networks['cutter']['id'] or model.get('classifier') != networks['classifier']['id']:
            continue
        if segment not in frames or not isinstance(k, int) or not 0 <= k < frames[segment] or k % FPS:
            continue
        ours = by_frame.get((segment, k), [])
        theirs = row.get('detections') or []
        counted['frames'] += 1
        counted['boxes'] += len(theirs)
        left = list(ours)
        for item in theirs:
            same = next((box for box in left if box[0] == item.get('model_label')
                         and all(abs(a - b) <= 0.5 + 1e-6 for a, b in zip(box[1:5], item['bbox_xyxy']))
                         and abs(box[5] - item['score'] * 100) <= 1 + 1e-6), None)
            if same is None:
                break
            left.remove(same)
        else:
            counted['matching'] += not left
    return counted


# -- the site ----------------------------------------------------------------------------------------------------------
def read(root: Path, route: str) -> dict | None:
    """The file of a drive with "current": read by the networks in use and on the drive's minutes as they are now; None
    when it has none."""
    track = _read_json(track_path(root, route))
    if track is None or track.get('format') != FORMAT:
        return None
    now = sources(root, route)
    track['current'] = (track.get('networks') == networks_in_use(root)
                        and {item.get('segment') for item in track.get('segments', ())} == set(now)
                        and all(now.get(item.get('segment')) == item.get('source') for item in track.get('segments', ())))
    return track


def summary(root: Path) -> dict:
    """The card of Models: the networks in use, and every drive with a video: its minutes, those read, its state ('done'
    by the networks in use on its minutes now, 'partial' being read by them, 'old' by other networks or other minutes,
    'none'), when it was made, its frames and boxes; video false when its video must be made again first."""
    root = Path(root)
    in_use = networks_in_use(root)
    drives = []
    for route in video_routes(root):
        now = sources(root, route)
        index = drive_video.index(root, route)
        path = track_path(root, route)
        meta = _read_json(meta_path(path)) if path.is_file() else None
        item = {'route': route, 'minutes': len(now), 'done_minutes': 0, 'state': 'none', 'made_at': None, 'frames': 0,
                'boxes': 0, 'video': index is not None}
        if meta is not None:
            current = (meta.get('networks') == in_use and [[segment, source] for segment, source in sorted(now.items())]
                       == meta.get('segments'))
            item.update(state='done' if current else 'old', made_at=meta.get('made_at'), frames=meta.get('frames', 0),
                        boxes=meta.get('boxes', 0), done_minutes=len(meta.get('segments') or ()))
        if item['state'] != 'done' and in_use is not None and partial_path(path).is_file():
            done = {segment for segment, line in _partial_minutes(partial_path(path), in_use).items()
                    if now.get(segment) == line.get('source')}
            if done:
                item.update(state='partial', done_minutes=len(done))
        drives.append(item)
    return {'networks_in_use': in_use, 'drives': drives}


# -- nnslr sign-tracks -------------------------------------------------------------------------------------------------
def run(root: Path, routes: list[str], *, device: str | None = None, dry_run: bool = False,
        limit_frames: int | None = None, force: bool = False, out_dir: Path | None = None,
        log: Callable[[str], None] = lambda text: print(text, flush=True)) -> int:
    """nnslr sign-tracks: every drive of routes in turn, one line each ("drive i/N <route>: ..."); a drive that fails
    says why and the others go on. A dry run: on the CPU, limit_frames (DRY_FRAMES) of the first minute, into
    derived/sign_tracks/dry (or out_dir), with the check against Sign review's frames. Returns how many failed."""
    root = Path(root)
    if dry_run:
        device = 'cpu'
        limit_frames = limit_frames or DRY_FRAMES
        out_dir = out_dir or folder(root) / DRY_FOLDER
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    finder = load_finder(root, device)
    log('Networks: ' + ', '.join(f"{part} {finder.networks[part]['id']}" for part in PARTS) + f' on {device}')
    failed = 0
    for count, route in enumerate(routes, 1):
        try:
            result = run_drive(root, route, finder, limit_frames=limit_frames, out_dir=out_dir, force=force, log=log)
        except (SignTrackError, ValueError, OSError) as exc:
            failed += 1
            log(f'drive {count}/{len(routes)} {route}: failed: {exc}')
            continue
        if result['state'] == 'current':
            log(f'drive {count}/{len(routes)} {route}: done already')
            continue
        log(f"drive {count}/{len(routes)} {route}: {result['frames']} frames, {result['boxes']} boxes in "
            f"{result['seconds']:.0f} s")
        if dry_run:
            found = parity(root, json.loads(result['path'].read_text(encoding='utf-8')), report_rows(root, route))
            log(f"parity: {found['matching']}/{found['frames']} frames of Sign review match "
                f"({found['boxes']} boxes compared)")
            log(f"written: {result['path']}")
    return failed
