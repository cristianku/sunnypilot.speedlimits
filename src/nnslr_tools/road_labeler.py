"""The road labeler: an offline network that prepares the labels of Speed review, so that a person has less to fix. It
says, at every moment of a drive, the road type, the kinds of road (a tunnel and the other kinds of speed_road_kinds)
and whether we are inside roadworks, from the road camera at high definition, the map scene of that moment, the GPS and
the dashboard speed (nnslr_tools.road_labeler_data, nnslr_tools.road_labeler_net). It never runs on the car and is never
an input of the car: it writes proposals, Speed review shows them, and the brain learns from Speed review. Started only on
request: nnslr road-labeler-train, nnslr road-labeler-predict, helpers/road_labeler.sh.

The training (train_labeler). The features of every drive are made once (the frozen encoder on the GPU, kept in
derived/road_labeler_features/<encoder_id>/), then the memory learns on windows of WINDOW steps, each inside one stretch
of train (a picture of validation or test never runs in a training step, nor in the memory of one, as for the brain: the
brain's frozen split, so the places held out of the brain are held out here too). A quarter of the windows (MAP_DROP_SHARE)
lose the map's inputs, so the pictures have to say the road too where the map is wrong or missing. A kind of road, or the
roadworks, with one side alone in train (all inside, or all outside) leaves the loss of train (road_labeler_data
.keep_both_sides): the tunnels of the map without a "checked to here" are all inside, and alone they would teach a
constant yes. The last epoch is kept: road-labeler.pt and result.json in the run's folder.

The measure (result.json), per partition: types, the share of steps whose road type is right, on all those with one and
on those a person set, beside the map's own type on the same steps (map_on_person): the steps a person set are where the
map was looked at, and the map's share there is what the network has to beat; most other answers are the map itself,
which is one of the inputs. kinds and works: precision and recall at 0.5 of each kind and of the roadworks, on the steps
with an answer, None where the partition has no step to tell them by.

The prediction (predict_labeler) goes through a whole drive in windows of WINDOW steps that overlap by MARGIN on each side,
keeping the middle of each: every step is said with at least MARGIN steps around it both ways, as in training. It writes
derived/road_labeler/<route>.jsonl: a first line {run, step_hz, road_types, kinds, made}, then one line a step {ns,
segment, decoded, frame_key, partition, road_type, road_type_p, map_type, kinds: {kind: chance}, works}. Where the network
is sure it also writes into Speed review, the way nnslr tunnels does (write_marks, as tunnels.write_tunnels, reviewer
NET_PREFIX + the run's name): never over a person's mark, a mark of the kind already there, where a person removed one, or
before that kind's "checked to here". Only the heads trained with both sides write (writable): a kind or the roadworks
left out of the loss of train still has its chances in the proposals, and never a mark.

- the kinds of road: a stretch of at least MIN_MARK_S where its chance stays at SURE or more (stretches closer than JOIN_S
  are one), by default;
- the roadworks (the same, start and end taps) and the road types (only with --write): a road type is written where the
  map's line has no person's type and says another one, for at least TYPE_MIN_FRAMES seconds at TYPE_SURE or more
  (speed_review's set_road_type, which keeps it as a correction of annotations/road_types).

A road type the labeler writes is a correction like a person's in annotations/road_types: road_types.corrected marks it
reviewed, so the brain's type_head and the road-type network take it as a person's answer (their readers do not tell the
reviewer). That is why it waits for --write. The labeler's own writes are never answers of its own training
(road_labeler_data).

A dry run (train --dry-run) is a few steps on the CPU with ENCODER_TINY on the first DRY_STEPS steps of one or two drives:
the chain works, nothing is learned, nothing written outside --run-dir. Its checkpoint never writes into Speed review, and
its proposals go to <run>/proposals.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from nnslr_tools import road_labeler_data as data
from nnslr_tools.road_labeler_data import NET_PREFIX, NS, PARTITIONS, STEP_HZ
from nnslr_tools.road_types import ROAD_TYPES

FEATURES_RELATIVE = Path('derived') / 'road_labeler_features'
PROPOSALS_RELATIVE = Path('derived') / 'road_labeler'
CHECKPOINT = 'road-labeler.pt'
RESULT = 'result.json'
WINDOW = 8 * 60 * STEP_HZ  # 8 minutes of memory, both ways
MARGIN = 2 * 60 * STEP_HZ  # at prediction, at least 2 minutes around every step said
BATCH = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
CLIP = 5.0
MAP_DROP_SHARE = 0.25
SURE = 0.9
TYPE_SURE = 0.9
MIN_MARK_S = 2.0
JOIN_S = 3.0
TYPE_MIN_FRAMES = 5
DRY_STEPS = 48
DRY_DRIVES = 2
DRY_TRAIN_STEPS = 2
# The map's inputs of road_labeler_data, set to "no line" when the map is dropped.
_MAP_INPUTS = [i for i, name in enumerate(data.INPUT_NAMES) if name.split(':')[0] in
               ('map_type', 'highway', 'maxspeed', 'maxspeed_known', 'basis', 'map_sure', 'map_tunnel', 'map_line')]
_NO_LINE = data.map_vector(None, set())


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def train_pieces(partitions: list[str], window: int, offset: int = 0) -> list[tuple[int, int]]:
    """The windows of a drive the training learns on: [first, last) inside one stretch of train, cut every window steps
    from offset (so the cuts move every epoch), none shorter than a tenth of a window unless its stretch is."""
    found = []
    at = 0
    while at < len(partitions):
        if partitions[at] != 'train':
            at += 1
            continue
        end = at
        while end < len(partitions) and partitions[end] == 'train':
            end += 1
        cuts = [at] + [c for c in range(at + (offset % window or window), end, window)] + [end]
        pieces = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
        merged: list[list[int]] = []
        for a, b in pieces:
            if merged and (b - a < window // 10 or merged[-1][1] - merged[-1][0] < window // 10):
                merged[-1][1] = b
            else:
                merged.append([a, b])
        found += [(a, b) for a, b in merged]
        at = end
    return found


def windows(length: int, window: int = WINDOW, margin: int = MARGIN) -> list[tuple[int, int, int, int]]:
    """The windows of the prediction over a drive of length steps: [(first, last, keep_first, keep_last)], each kept part
    with margin steps around it (fewer only at the drive's ends), the kept parts covering the drive once."""
    if length <= window:
        return [(0, length, 0, length)]
    stride = window - 2 * margin
    return [(max(0, start - margin), min(length, start + stride + margin), start, min(length, start + stride))
            for start in range(0, length, stride)]


def sure_stretches(ns: np.ndarray, chance: np.ndarray, threshold: float = SURE, min_s: float = MIN_MARK_S,
                   join_s: float = JOIN_S) -> list[tuple[int, int]]:
    """[(start_ns, end_ns)] where chance stays at threshold or more, step after step with a time: from its first step to
    the end of its last (a step is 1 / STEP_HZ long), stretches at most join_s apart joined, those shorter than min_s
    left out. A hole in time between two steps (a minute without its encode index or its frames, a slow-only minute)
    ends a stretch: nothing is said sure across what the network did not see."""
    found: list[list[int]] = []
    length = NS // STEP_HZ
    last = None
    for t, sure in zip(np.asarray(ns, dtype=np.int64), np.asarray(chance) >= threshold):
        if t < 0:
            continue
        if found and last is not None and t - last > 2 * length:
            found[-1][2] = False
        last = int(t)
        if sure:
            if found and found[-1][2]:
                found[-1][1] = int(t) + length
            elif found and t - found[-1][1] <= join_s * NS:
                found[-1][1], found[-1][2] = int(t) + length, True
            else:
                found.append([int(t), int(t) + length, True])
        elif found:
            found[-1][2] = False
    return [(a, b) for a, b, _open in found if b - a >= min_s * NS]


def _free(start: int, end: int, marks: list[dict], removed: list[tuple[int, int]], checked: int | None) -> bool:
    """Nothing of a person's in [start, end): no mark there already, no removal, not before "checked to here"."""
    clash = any(row['start_ns'] < end and (row['end_ns'] is None or start < row['end_ns']) for row in marks)
    return not clash and not any(a < end and start < b for a, b in removed) and (checked is None or start >= checked)


def write_marks(store, route: str, kind: str | None, found: list[tuple[int, int]], span: tuple[int, int],
                reviewer: str) -> dict:
    """The stretches found as marks in Speed review (speed_review.SpeedReviewStore): of a kind of road (tap_road), or the
    roadworks (kind None: tap_works), with the rules of tunnels.write_tunnels. Returns {written, kept} (kept: left out for
    a person's mark, removal or check, or refused by the store; a start whose end is refused is taken back, and the
    labeler's own removals do not count as a person's)."""
    from nnslr_tools.speed_review import SpeedReviewError
    action = 'works_removed' if kind is None else 'road_removed'
    removed = [(row['start_ns'], row['end_ns']) for row in store.history(route) if row['action'] == action
               and (kind is None or row['state'] == kind) and not data.own(row.get('reviewer'))]
    checked = store.works_checked(route) if kind is None else store.road_checked(route).get(kind)
    written = kept = 0
    for start, end in found:
        start, end = max(start, span[0]), min(end, span[1])
        if end - start < MIN_MARK_S * NS:
            continue
        marks = store.works(route) if kind is None else [row for row in store.road(route) if row['kind'] == kind]
        if not _free(start, end, marks, removed, checked):
            kept += 1
            continue
        def tap(name: str, at: int) -> None:
            if kind is None:
                store.tap_works(route, name, at, span[1], store.revision(route), reviewer)
            else:
                store.tap_road(route, kind, name, at, span[1], store.revision(route), reviewer)

        try:
            tap('start', start)
        except SpeedReviewError:
            kept += 1
            continue
        try:
            tap('end', end)
            written += 1
        except SpeedReviewError:
            tap('remove', start)  # no mark left open: the store refused its end
            kept += 1
    return {'written': written, 'kept': kept}


def type_stretches(lines: list[dict], chance_of: dict[str, np.ndarray], threshold: float = TYPE_SURE,
                   min_frames: int = TYPE_MIN_FRAMES) -> list[tuple[str, str, str]]:
    """[(first frame key, last frame key, road type)]: runs of the map's lines (road_types.load_road_types' order, every
    correction over them) with no person's type, where the network says one type at threshold or more and the line says
    another, at least min_frames long. chance_of: {frame key: chances over ROAD_TYPES}."""
    found: list[list] = []
    previous = None
    for at, line in enumerate(lines):
        chance = chance_of.get(line['frame_key'])
        said = None
        if chance is not None and line.get('reviewed') is not True:
            best = int(np.argmax(chance))
            if chance[best] >= threshold and line.get('road_type') != ROAD_TYPES[best]:
                said = ROAD_TYPES[best]
        if said is not None and found and found[-1][2] == said and previous == at - 1:
            found[-1][1] = line['frame_key']
            found[-1][3] += 1
        elif said is not None:
            found.append([line['frame_key'], line['frame_key'], said, 1])
        previous = at if said is not None else previous
    return [(a, b, kind) for a, b, kind, count in found if count >= min_frames]


def write_types(store, route: str, found: list[tuple[str, str, str]], ns_of: dict[str, int], span: tuple[int, int],
                reviewer: str) -> dict:
    """The road types of type_stretches into Speed review (set_road_type, as a person's correction of a stretch).
    Returns {written, kept}."""
    from nnslr_tools.speed_review import SpeedReviewError
    written = kept = 0
    for first, last, kind in found:
        start, end = ns_of.get(first), ns_of.get(last)
        if start is None or end is None:
            kept += 1
            continue
        end = min(end + NS, span[1])
        try:
            store.set_road_type(route, first, last, start, max(end, start + 1), kind, store.revision(route), reviewer)
            written += 1
        except SpeedReviewError:
            kept += 1
    return {'written': written, 'kept': kept}


def _cut(drive: dict, length: int) -> dict:
    """The first length steps from the first step of train with an answer on: for a dry run."""
    weights = drive['type_weight'] + drive['works_weight'] + drive['kinds_weight'].sum(axis=1)
    first = next((i for i, name in enumerate(drive['partitions']) if name == 'train' and weights[i] > 0), 0)
    part = slice(first, first + length)
    found = dict(drive)
    for name in ('ns', 'inputs', 'type', 'type_weight', 'type_person', 'map_type', 'kinds', 'kinds_weight', 'works',
                 'works_weight'):
        found[name] = drive[name][part]
    found['steps'], found['partitions'] = drive['steps'][part], drive['partitions'][part]
    return found


def _net_inputs(drive: dict, present: np.ndarray) -> np.ndarray:
    return np.concatenate([drive['inputs'], present[:, None].astype(np.float32)], axis=1)


def _encoder_for(name: str, device, state: dict | None = None, pretrained: bool = True):
    from nnslr_tools.road_labeler_net import build_encoder
    encoder = build_encoder(name, pretrained=pretrained and state is None)
    if state is not None:
        encoder.load_state_dict(state)
    return encoder.to(device)


def _features(root: Path, drives: list[dict], name: str, encoder, device, cache_dir: Path | None, log) -> None:
    from nnslr_tools.road_labeler_net import drive_features
    for drive in drives:
        started = time.monotonic()
        drive['features'], drive['present'] = drive_features(root, drive['route'], drive['steps'], encoder, name, device,
                                                             cache_dir, log)
        log(f"{drive['route']}: features of {len(drive['steps'])} steps, {int(drive['present'].sum())} with a picture, "
            f'{time.monotonic() - started:.0f}s')


def _batches(drives: list[dict], rng: random.Random, window: int) -> list[list[tuple[int, int, int]]]:
    """The windows of an epoch, [(drive, first, last)], shuffled, in batches of windows of one length (no padding,
    which a GRU going both ways would read)."""
    pieces = []
    for number, drive in enumerate(drives):
        weights = drive['type_weight'] + drive['works_weight'] + drive['kinds_weight'].sum(axis=1)
        for first, last in train_pieces(drive['partitions'], window, rng.randrange(window)):
            if weights[first:last].sum() > 0:
                pieces.append((number, first, last))
    rng.shuffle(pieces)
    by_length: dict[int, list] = {}
    for piece in pieces:
        by_length.setdefault(piece[2] - piece[1], []).append(piece)
    batches = [group[i:i + BATCH] for group in by_length.values() for i in range(0, len(group), BATCH)]
    rng.shuffle(batches)
    return batches


def _stack(drives: list[dict], batch: list[tuple[int, int, int]], device, rng: random.Random | None):
    """The tensors of a batch of windows: features, inputs (the map dropped on MAP_DROP_SHARE of them when rng is
    given) and the answers, the weights of train only."""
    import torch
    feats, extra, answers = [], [], {key: [] for key in ('type', 'type_weight', 'kinds', 'kinds_weight', 'works',
                                                        'works_weight')}
    for number, first, last in batch:
        drive = drives[number]
        feats.append(drive['features'][first:last].astype(np.float32))
        inputs = _net_inputs(drive, drive['present'])[first:last].copy()
        if rng is not None and rng.random() < MAP_DROP_SHARE:
            inputs[:, _MAP_INPUTS] = _NO_LINE[_MAP_INPUTS]
        extra.append(inputs)
        train = (np.array(drive['partitions'][first:last]) == 'train').astype(np.float32)
        for key in answers:
            value = drive[key][first:last]
            if key.endswith('weight'):
                value = value * (train[:, None] if value.ndim == 2 else train)
            answers[key].append(value)
    tensor = lambda values, kind=torch.float32: torch.from_numpy(np.stack(values)).to(device=device, dtype=kind)
    return tensor(feats), tensor(extra), {key: tensor(values, torch.long if key == 'type' else torch.float32)
                                          for key, values in answers.items()}


def predict_drive(model, drive: dict, device, window: int = WINDOW, margin: int = MARGIN) -> dict:
    """The chances of a whole drive (road_labeler_net.chances), window by window: see the module's text."""
    import torch
    from nnslr_tools.road_labeler_net import chances
    model.eval()
    length = len(drive['steps'])
    kinds = len(drive['kind_names'])
    found = {'types': np.zeros((length, len(ROAD_TYPES)), dtype=np.float32),
             'kinds': np.zeros((length, kinds), dtype=np.float32), 'works': np.zeros(length, dtype=np.float32)}
    extra = _net_inputs(drive, drive['present'])
    with torch.no_grad():
        for first, last, keep_first, keep_last in windows(length, window, margin):
            if last <= first:
                continue
            said = chances(model(torch.from_numpy(drive['features'][first:last].astype(np.float32))[None].to(device),
                                 torch.from_numpy(extra[first:last])[None].to(device)))
            inside = slice(keep_first - first, keep_last - first)
            found['types'][keep_first:keep_last] = said['types'][inside]
            found['works'][keep_first:keep_last] = said['works'][inside]
            if kinds:
                found['kinds'][keep_first:keep_last] = said['kinds'][inside]
    return found


def _share(hits: int, count: int) -> float | None:
    return round(hits / count, 4) if count else None


def _binary(chance: np.ndarray, answer: np.ndarray, weight: np.ndarray) -> dict:
    known = weight > 0
    said, truth = chance[known] >= 0.5, answer[known] >= 0.5
    inside, outside = int(truth.sum()), int((~truth).sum())
    hits = int((said & truth).sum())
    return {'inside': inside, 'outside': outside,
            'precision': _share(hits, int(said.sum())) if inside and outside else None,
            'recall': _share(hits, inside) if inside and outside else None}


def measure(drives: list[dict], said: list[dict], partition: str) -> dict:
    """See the module's text."""
    typed = person = right = right_person = map_right = 0
    per_type = {name: {'steps': 0, 'right': 0} for name in ROAD_TYPES}
    kinds: dict[str, list] = {}
    works = []
    for drive, chance in zip(drives, said):
        here = np.array(drive['partitions']) == partition
        has = here & (drive['type_weight'] > 0)
        guess = chance['types'].argmax(axis=1)
        typed += int(has.sum())
        right += int((guess[has] == drive['type'][has]).sum())
        mine = has & drive['type_person']
        person += int(mine.sum())
        right_person += int((guess[mine] == drive['type'][mine]).sum())
        map_right += int((drive['map_type'][mine] == drive['type'][mine]).sum())
        for index, name in enumerate(ROAD_TYPES):
            of = has & (drive['type'] == index)
            per_type[name]['steps'] += int(of.sum())
            per_type[name]['right'] += int((guess[of] == index).sum())
        for column, kind in enumerate(drive['kind_names']):
            kinds.setdefault(kind, []).append((chance['kinds'][here, column], drive['kinds'][here, column],
                                               drive['kinds_weight'][here, column]))
        works.append((chance['works'][here], drive['works'][here], drive['works_weight'][here]))

    def joined(parts):
        return _binary(*(np.concatenate([part[i] for part in parts]) for i in range(3))) if parts else None

    return {'types': {'steps': typed, 'right': _share(right, typed), 'person_steps': person,
                      'right_on_person': _share(right_person, person), 'map_on_person': _share(map_right, person),
                      'per_type': {name: {**item, 'right': _share(item['right'], item['steps'])}
                                   for name, item in per_type.items() if item['steps']}},
            'kinds': {kind: joined(parts) for kind, parts in kinds.items()}, 'works': joined(works)}


def train_labeler(data_root: Path, run_dir: Path, merged_of, *, routes: list[str] | None = None, epochs: int = EPOCHS,
                  device: str | None = None, dry_run: bool = False, seed: int = 0, log=_say) -> dict:
    """Train on train, keep the last epoch, measure once: road-labeler.pt and result.json in run_dir (see the module's
    text). merged_of(route): the drive's joined logs."""
    import torch
    from nnslr_tools.road_labeler_net import ENCODER, ENCODER_TINY, build_labeler_net, encoder_id, feature_size, labeler_loss
    root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    drives = data.dataset(root, merged_of, routes, log)
    if dry_run:
        drives = [_cut(drive, DRY_STEPS) for drive in drives[:DRY_DRIVES]]
    if not drives:
        raise ValueError('no drive to learn from: a scanned drive with its frames is needed')
    sides = data.keep_both_sides(drives, log)
    name = ENCODER_TINY if dry_run else ENCODER
    encoder = _encoder_for(name, device, pretrained=not dry_run)
    cache = run_dir / 'features' if dry_run else root / FEATURES_RELATIVE / encoder_id(name)
    _features(root, drives, name, encoder, device, cache, log)
    kinds = drives[0]['kind_names']
    model = build_labeler_net(feature_size(name), len(kinds)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    window = DRY_STEPS // 2 if dry_run else WINDOW
    epochs = 1 if dry_run else epochs
    history = []
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        errors = []
        for batch in _batches(drives, rng, window):
            if dry_run and len(errors) >= DRY_TRAIN_STEPS:
                break
            feats, extra, answers = _stack(drives, batch, device, rng)
            loss = labeler_loss(model(feats, extra), answers)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            errors.append(float(loss.detach()))
        said = [predict_drive(model, drive, device, window, window // 4) for drive in drives]
        validation = measure(drives, said, 'validation')
        error = float(np.mean(errors)) if errors else None
        history.append({'epoch': epoch, 'error': error, 'validation': validation['types']})
        right = validation['types']['right_on_person']
        log(f'road_labeler epoch {epoch + 1}/{epochs} train_loss {"-" if error is None else f"{error:.4f}"} '
            f'val_types {validation["types"]["right"]} val_on_person {"-" if right is None else right} '
            f'(map {validation["types"]["map_on_person"]}) {len(errors)} steps {time.monotonic() - started:.0f}s')
    run = run_dir.name
    checkpoint = {'weights': model.state_dict(), 'epoch': epochs - 1, 'encoder': name, 'encoder_id': encoder_id(name),
                  'encoder_weights': encoder.state_dict() if name == ENCODER_TINY else None,
                  'feature_dim': feature_size(name), 'kinds': kinds, 'road_types': list(ROAD_TYPES),
                  'inputs': list(data.INPUT_NAMES), 'step_hz': STEP_HZ, 'window': window, 'dry_run': dry_run,
                  'run': run, 'seed': seed, 'trained': trained_heads(sides)}
    torch.save(checkpoint, run_dir / CHECKPOINT)
    result = {'run': run, 'dry_run': dry_run, 'encoder_id': encoder_id(name), 'step_hz': STEP_HZ, 'window_steps': window,
              'kinds': kinds, 'road_types': list(ROAD_TYPES), 'seed': seed, 'history': history,
              'drives': {drive['route']: len(drive['steps']) for drive in drives}, 'train_sides': sides,
              **{partition: measure(drives, said, partition) for partition in PARTITIONS}}
    (run_dir / RESULT).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def trained_heads(sides: dict) -> list[str]:
    """The kinds (and 'works') that had both sides in train (road_labeler_data.both_sides): the only heads that learned."""
    return [name for name, item in sides.items() if item['inside'] and item['outside']]


def writable(checkpoint: dict) -> tuple[list[str], bool]:
    """(the kinds whose sure stretches may be written, whether the roadworks may): only the heads trained with both
    sides. A head masked in training says chances from its first weights, and a mark of it would teach the brain noise;
    its chances still go into the proposals."""
    trained = set(checkpoint.get('trained') or ())
    return [kind for kind in checkpoint['kinds'] if kind in trained], 'works' in trained


def load_labeler(run_dir: Path, device):
    """(model, encoder, checkpoint) of a run."""
    import torch
    from nnslr_tools.road_labeler_net import build_labeler_net
    checkpoint = torch.load(Path(run_dir) / CHECKPOINT, map_location=device, weights_only=False)
    model = build_labeler_net(checkpoint['feature_dim'], len(checkpoint['kinds'])).to(device)
    model.load_state_dict(checkpoint['weights'])
    encoder = _encoder_for(checkpoint['encoder'], device, checkpoint.get('encoder_weights'))
    return model.eval(), encoder, checkpoint


def proposal_lines(drive: dict, said: dict, checkpoint: dict) -> list[dict]:
    """The lines of derived/road_labeler/<route>.jsonl: see the module's text."""
    lines = [{'run': checkpoint['run'], 'step_hz': STEP_HZ, 'road_types': list(ROAD_TYPES), 'kinds': drive['kind_names'],
              'made': datetime.now().astimezone().isoformat(timespec='seconds')}]
    for at, step in enumerate(drive['steps']):
        best = int(said['types'][at].argmax())
        lines.append({'ns': None if step['ns'] is None else int(step['ns']), 'segment': step['segment'],
                      'decoded': step['decoded'], 'frame_key': step['key'], 'partition': drive['partitions'][at],
                      'road_type': ROAD_TYPES[best], 'road_type_p': round(float(said['types'][at, best]), 3),
                      'map_type': None if drive['map_type'][at] < 0 else ROAD_TYPES[drive['map_type'][at]],
                      'kinds': {kind: round(float(said['kinds'][at, column]), 3)
                                for column, kind in enumerate(drive['kind_names'])},
                      'works': round(float(said['works'][at]), 3)})
    return lines


def frame_chances(drive: dict, said: dict) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """({frame key: the mean chances over ROAD_TYPES of its steps}, {frame key: the ns of its 1 Hz step})."""
    sums: dict[str, list] = {}
    ns_of: dict[str, int] = {}
    for at, step in enumerate(drive['steps']):
        sums.setdefault(step['key'], []).append(said['types'][at])
        if step['step'] == 0 and step['ns'] is not None:
            ns_of[step['key']] = int(step['ns'])
    return {key: np.mean(values, axis=0) for key, values in sums.items()}, ns_of


def predict_labeler(data_root: Path, run_dir: Path, merged_of, span_of, routes: list[str], *, out_dir: Path | None = None,
                    marks: bool = True, write: bool = False, dry_run: bool = False, device: str | None = None,
                    log=_say) -> list[dict]:
    """The proposals of each drive of routes into out_dir (derived/road_labeler by default) and, where sure, the writes
    of the module's text. span_of(route): the drive's (first, last) ns, None before a scan. marks: the kinds of road;
    write: also the roadworks and the road types. dry_run, or a checkpoint of a dry run: the first DRY_STEPS steps only
    for dry_run, the proposals in <run>/proposals, nothing written into Speed review. Returns a summary a drive."""
    import torch
    root, run_dir = Path(data_root), Path(run_dir)
    made_dry = torch.load(run_dir / CHECKPOINT, map_location='cpu', weights_only=False)['dry_run']
    trial = dry_run or made_dry
    device = torch.device(device or ('cuda' if torch.cuda.is_available() and not trial else 'cpu'))
    model, encoder, checkpoint = load_labeler(run_dir, device)
    target = run_dir / 'proposals' if trial else Path(out_dir) if out_dir else root / PROPOSALS_RELATIVE
    cache = run_dir / 'features' if trial else root / FEATURES_RELATIVE / checkpoint['encoder_id']
    reviewer = NET_PREFIX + checkpoint['run']
    kinds = tuple(checkpoint['kinds'])
    marked, works_trained = writable(checkpoint)
    untrained = [name for name in (*kinds, 'works') if name not in marked and not (name == 'works' and works_trained)]
    if untrained:
        log(f'Not trained with both sides, never written: {", ".join(untrained)} (their chances are in the proposals).')
    store = None
    if not trial and (marks or write):
        from nnslr_tools.speed_review import store_for
        store = store_for(root)
    if trial and (marks or write):
        log('A dry run: nothing is written into Speed review.')
    tunnels = data.tunnel_ways(root)
    found = []
    for route in routes:
        merged = merged_of(route)
        span = span_of(route)
        if not merged or span is None:
            log(f'{route}: no scan, left out')
            continue
        drive = data.drive_data(root, route, merged, kinds=kinds, tunnels=tunnels, log=log)
        if dry_run:
            drive = _cut(drive, DRY_STEPS)
        _features(root, [drive], checkpoint['encoder'], encoder, device, cache, log)
        said = predict_drive(model, drive, device, checkpoint['window'], checkpoint['window'] // 4)
        from nnslr_tools.annotations import atomic_write, jsonl_bytes
        path = target / f'{route}.jsonl'
        atomic_write(path, jsonl_bytes(proposal_lines(drive, said, checkpoint)))
        summary = {'route_id': route, 'steps': len(drive['steps']), 'proposals': str(path)}
        stretches = {kind: sure_stretches(drive['ns'], said['kinds'][:, column]) for column, kind in enumerate(kinds)}
        works = sure_stretches(drive['ns'], said['works'])
        chance_of, ns_of = frame_chances(drive, said)
        types = type_stretches(drive['lines'], chance_of)
        summary['sure'] = {**{kind: len(items) for kind, items in stretches.items()}, 'works': len(works),
                           'road_types': len(types)}
        summary['untrained'] = untrained
        if store is not None:
            known = {row['kind'] for row in store.road_kinds()}
            if marks:
                summary['marks'] = {kind: write_marks(store, route, kind, items, span, reviewer)
                                    for kind, items in stretches.items() if kind in known and kind in marked}
            if write:
                if works_trained:
                    summary['works'] = write_marks(store, route, None, works, span, reviewer)
                summary['road_types'] = write_types(store, route, types, ns_of, span, reviewer)
        log(json.dumps(summary))
        found.append(summary)
    return found
