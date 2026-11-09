"""The road labeler: an offline network that labels Speed review, the base layer a person corrects only where it is
wrong. It says, at every moment of a drive, the road type, the kinds of road (a tunnel and the other kinds of
speed_road_kinds) and whether we are inside roadworks, from the road camera at high definition, the map scene of that
moment, the GPS and the dashboard speed (nnslr_tools.road_labeler_data, nnslr_tools.road_labeler_net). It never runs on
the car and is never an input of the car: it writes into Speed review, and the brain learns from Speed review. Started
only on request: nnslr road-labeler-train, nnslr road-labeler-predict, helpers/road_labeler.sh, the portal's Road
labeler card (Models).

The training (train_labeler). The features of every drive are made once (the frozen encoder on the GPU, kept in
derived/road_labeler_features/<encoder_id>/), then the memory learns on windows of WINDOW steps, each inside one stretch
of train (a picture of validation or test never runs in a training step, nor in the memory of one, as for the brain: the
brain's frozen split, so the places held out of the brain are held out here too). A quarter of the windows (MAP_DROP_SHARE)
lose the map's inputs, so the pictures have to say the road too where the map is wrong or missing. A kind of road, or the
roadworks, with one side alone in train (all inside, or all outside) leaves the loss of train (road_labeler_data
.keep_both_sides): the marks of a kind with no "checked to here" are all inside, and alone they would teach a constant
yes (the tunnels have the weak "no" of road_labeler_data.tunnel_free besides). The last epoch is kept, with the gate of
each head on validation (earned_heads): road-labeler.pt and result.json (its gate) in the run's folder.

The measure (result.json), per partition: types, the share of steps whose road type is right, on all those with one and
on those a person set, beside the map's own type on the same steps (map_on_person): the steps a person set are where the
map was looked at, and the map's share there is what the network has to beat; most other answers are the map itself,
which is one of the inputs. kinds and works: precision and recall at 0.5 of each kind and of the roadworks, on the steps
with an answer, None where the partition has no step to tell them by.

The prediction (predict_labeler) goes through a whole drive in windows of WINDOW steps that overlap by MARGIN on each side,
keeping the middle of each: every step is said with at least MARGIN steps around it both ways, as in training. It writes
derived/road_labeler/<route>.jsonl, the chances: a first line {run, step_hz, road_types, kinds, made}, then one line a
step {ns, segment, decoded, frame_key, partition, road_type, road_type_p, map_type, kinds: {kind: chance}, works}.

Prepare Speed review (the same command, the portal's button) then writes, where the network is sure, what the labeler
says, and that is definitive, his rule of 9 Nov 2026: nobody confirms it, Speed review shows it, the brain learns it, and
a person corrects it only where it is wrong (prepare_drive):

- the road types: a layer of their own per drive, derived/road_labeler_types/<route>.jsonl (road_types
  .write_labeler_types), over the map and under a person's corrections (road_types.load_road_types). A frame is in the
  layer where the network says a type whose head passed the gate at TYPE_SURE or more, in a run of at least
  TYPE_MIN_FRAMES frames of that type (layer_types); elsewhere the map shows through;
- the kinds of road and the roadworks: a mark (reviewer NET_PREFIX + the run's name) on a stretch of at least MIN_MARK_S
  where its chance stays at SURE or more (stretches closer than JOIN_S are one), as nnslr tunnels marks: never over a
  person's mark or a mark of nnslr tunnels, where a person removed one, or before that kind's "checked to here".

Only the heads that earned it write (writable): trained with both sides (a kind or the roadworks left out of the loss of
train still has its chances in the chances file, and never a mark), and passed the gate of validation (earned_heads: at
least GATE_MIN_ANSWERS steps of full weight on each side, sure on at least GATE_MIN_SAID of them, and right on at least
GATE_PRECISION of those it was sure of). A head held back writes nothing and says why in the log and in the summary of
each drive (held_back).

A newer run replaces the older one, drive by drive, in one write of Speed review (all or nothing, every tap a row of
speed_history): its layer replaces the older layer whole, and every mark of the labeler that no person kept (a mark
that starts before its kind's "checked to here", kept_by_person) comes out, of any run and any head, also of a head the
new run holds back: what an older run said is never left behind as if this run had said it. Then the new run's marks go
in. A person's marks, removals and checks, and the marks of nnslr tunnels, are never touched. What it wrote, head by
head, goes to <run>/prepared.json for the card.

What the labeler wrote is never an answer of its own training (road_labeler_data masks it) nor of the road-type network
(road_type_data): a network never learns to say again what it said.

A dry run (train --dry-run) is a few steps on the CPU with ENCODER_TINY on the first DRY_STEPS steps of one or two drives:
the chain works, nothing is learned, nothing written outside --run-dir. Its checkpoint never writes into Speed review, and
its chances go to <run>/chances.
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from nnslr_tools import road_labeler_data as data
from nnslr_tools.road_labeler_data import NET_PREFIX, NS, PARTITIONS, STEP_HZ
from nnslr_tools.road_types import ROAD_TYPES, by_labeler, person_removals

FEATURES_RELATIVE = Path('derived') / 'road_labeler_features'
CHANCES_RELATIVE = Path('derived') / 'road_labeler'
CHECKPOINT = 'road-labeler.pt'
RESULT = 'result.json'
# What the last Prepare Speed review of a run wrote (predict_labeler), for the portal.
PREPARED = 'prepared.json'
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
# The gate of a head's writes (earned_heads), on the steps of validation with an answer of full weight (a person's, or a
# mark of nnslr tunnels; never the map's weak ones): at least GATE_MIN_ANSWERS of them inside and as many outside (a
# minute of steps each), sure (SURE, or TYPE_SURE against the map for a road type) on at least GATE_MIN_SAID, and right
# on at least GATE_PRECISION of those: a head that cannot show it is right where it would write writes nothing.
GATE_MIN_ANSWERS = 60 * STEP_HZ
GATE_MIN_SAID = 10 * STEP_HZ
GATE_PRECISION = 0.9
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
    left out. A hole in time between two steps (a minute without its encode index or its frames) ends a stretch:
    nothing is said sure across what the network did not see."""
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


def kept_by_person(mark: dict, checked: int | None) -> bool:
    """A mark of the labeler a person kept: it starts before its kind's "checked to here" (a person went past it and left
    it). A newer run never takes it out."""
    return checked is not None and mark['start_ns'] < checked


def layer_types(lines: list[dict], chance_of: dict[str, np.ndarray], earned: list[str], threshold: float = TYPE_SURE,
                min_frames: int = TYPE_MIN_FRAMES) -> dict[str, tuple[str, float]]:
    """{frame key: (road type, chance)} of the labeler's layer of road types (road_types.write_labeler_types): runs of the
    map's lines, in order, where the network says one type of earned at threshold or more, at least min_frames frames
    long. Elsewhere it says nothing and the map shows through. chance_of: {frame key: chances over ROAD_TYPES}."""
    found: dict[str, tuple[str, float]] = {}
    run: list[tuple[str, float]] = []
    said_before = None

    def close() -> None:
        if len(run) >= min_frames:
            found.update({key: (said_before, p) for key, p in run})

    for line in lines:
        chance = chance_of.get(line['frame_key'])
        said = p = None
        if chance is not None:
            best = int(np.argmax(chance))
            if chance[best] >= threshold and ROAD_TYPES[best] in earned:
                said, p = ROAD_TYPES[best], float(chance[best])
        if said != said_before:
            close()
            run = []
        if said is not None:
            run.append((line['frame_key'], p))
        said_before = said
    close()
    return found


def prepare_drive(store, root: Path, route: str, run: str, span: tuple[int, int], stretches: dict, earned: set,
                  types: dict[str, tuple[str, float]], types_earned: list[str]) -> dict:
    """Prepare Speed review on one drive with the run given, in one write of the store (SpeedReviewStore.batch, all or
    nothing): the marks of the labeler that no person kept (kept_by_person) come out, of any run and any kind, also of a
    head this run holds back, then this run's sure stretches go in (stretches: {kind, None for the roadworks: [(start_ns,
    end_ns)]}) for the heads of earned only, never over a person's mark, a mark of nnslr tunnels, a mark kept, a
    person's removal, or before that kind's "checked to here" (_free). A mark of this run already where it would go
    stays as it is. Then the layer of road types (types, of the heads types_earned) replaces the older one, after the
    write, with a row types_labeled in it; the same layer again writes nothing. Returns {head: {written, removed, kept,
    left}} (kept: the labeler's marks a person kept; left: a stretch left out for a person's word or refused by the
    store) and road_types: the frames of the layer."""
    from nnslr_tools import road_types
    from nnslr_tools.speed_review import SpeedReviewError
    reviewer = NET_PREFIX + run
    before_run, before_types = road_types.load_labeler_types(root, route)
    layer = {key: {'road_type': kind, 'p': round(float(p), 3)} for key, (kind, p) in types.items()}
    same_layer = before_run == run and {key: {'road_type': line['road_type'], 'p': line.get('p')}
                                        for key, line in before_types.items()} == layer
    for attempt in range(3):
        report: dict = {}
        try:
            with store.batch(route, store.revision(route)) as batch:
                known = batch.kinds()
                removals = person_removals(batch.history())
                for kind in (*known, None):
                    head = 'works' if kind is None else kind
                    checked = batch.checked(kind)
                    marks = batch.marks(kind)
                    old = [mark for mark in marks if by_labeler(mark['reviewer']) and not kept_by_person(mark, checked)]
                    others = [mark for mark in marks if mark not in old]
                    removed = [(row['start_ns'], row['end_ns']) for row in removals if row['kind'] == kind]
                    wanted = []
                    for start, end in stretches.get(kind, []) if kind in earned else []:
                        start, end = max(start, span[0]), min(end, span[1])
                        if end - start >= MIN_MARK_S * NS and _free(start, end, others, removed, checked):
                            wanted.append((start, end))
                    left = len(stretches.get(kind, [])) - len(wanted) if kind in earned else 0
                    stay = {(mark['start_ns'], mark['end_ns']) for mark in old if mark['reviewer'] == reviewer} & set(wanted)
                    out = [mark for mark in old if (mark['start_ns'], mark['end_ns']) not in stay or mark['reviewer'] != reviewer]
                    for mark in out:
                        batch.remove(kind, mark['start_ns'], span[1], reviewer)
                    written = 0
                    for start, end in wanted:
                        if (start, end) in stay:
                            continue
                        if batch.mark(kind, start, end, span[1], reviewer):
                            written += 1
                        else:
                            left += 1
                    item = {'written': written, 'removed': len(out),
                            'kept': sum(1 for mark in marks if by_labeler(mark['reviewer']) and mark not in old),
                            'left': left}
                    if any(item.values()) or kind in earned:
                        report[head] = item
                if not same_layer:
                    batch.log('types_labeled', span[0], span[1], reviewer, state=run,
                              note=f'{len(layer)} frames of {", ".join(types_earned) or "no road type"}')
                    # The last thing of the write: a layer that cannot be written takes the marks back with it.
                    road_types.write_labeler_types(root, route, run, types, types_earned)
            report['road_types'] = len(layer)
            return report
        except SpeedReviewError as exc:
            if exc.reason != 'stale_revision' or attempt == 2:
                raise
    raise AssertionError('unreachable')


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


def _gate(truth: np.ndarray, sure: np.ndarray) -> dict:
    """One head's gate from the answers of full weight (truth: inside) and where it was sure: see GATE_*."""
    inside, outside = int(truth.sum()), int((~truth).sum())
    said, hits = int(sure.sum()), int((sure & truth).sum())
    precision = _share(hits, said)
    if inside < GATE_MIN_ANSWERS or outside < GATE_MIN_ANSWERS:
        why = (f'{inside} steps inside and {outside} outside of full weight on validation, {GATE_MIN_ANSWERS} of each '
               'needed')
    elif said < GATE_MIN_SAID:
        why = f'sure on {said} steps of validation, {GATE_MIN_SAID} needed to tell whether it is right'
    elif precision < GATE_PRECISION:
        why = f'right on {precision:.0%} of the {said} steps of validation it was sure of, {GATE_PRECISION:.0%} needed'
    else:
        why = None
    return {'inside': inside, 'outside': outside, 'said': said, 'precision': precision, 'why': why}


def earned_heads(drives: list[dict], said: list[dict], partition: str = 'validation') -> dict:
    """{head: {inside, outside, said, precision, why}} of every kind, of 'works' and of 'type:<road type>' on partition,
    why None where the head earned its writes (GATE_*). A kind or the roadworks: its steps with an answer of full
    weight, sure at SURE. A road type: the steps a person set, inside where it is that type, sure where the network
    says it at TYPE_SURE and the map says another (where a write would go).
    """
    parts: dict[str, list] = {}
    for drive, chance in zip(drives, said):
        here = np.array(drive['partitions']) == partition
        for column, kind in enumerate(drive['kind_names']):
            full = here & (drive['kinds_weight'][:, column] >= data.STRONG_WEIGHT)
            parts.setdefault(kind, []).append((drive['kinds'][full, column] >= 0.5, chance['kinds'][full, column] >= SURE))
        full = here & (drive['works_weight'] >= data.STRONG_WEIGHT)
        parts.setdefault('works', []).append((drive['works'][full] >= 0.5, chance['works'][full] >= SURE))
        person = here & drive['type_person'] & (drive['type_weight'] > 0)
        best, top = chance['types'][person].argmax(axis=1), chance['types'][person].max(axis=1)
        for index, name in enumerate(ROAD_TYPES):
            sure = (best == index) & (top >= TYPE_SURE) & (drive['map_type'][person] != index)
            parts.setdefault(f'type:{name}', []).append((drive['type'][person] == index, sure))
    return {head: _gate(np.concatenate([truth for truth, _ in items]).astype(bool),
                        np.concatenate([sure for _, sure in items]).astype(bool)) for head, items in parts.items()}


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
    earned = earned_heads(drives, said)
    log('road_labeler gate on validation: ' + '; '.join(
        f'{head} {"writes" if item["why"] is None else "held back: " + item["why"]}' for head, item in earned.items()))
    checkpoint = {'weights': model.state_dict(), 'epoch': epochs - 1, 'encoder': name, 'encoder_id': encoder_id(name),
                  'encoder_weights': encoder.state_dict() if name == ENCODER_TINY else None,
                  'feature_dim': feature_size(name), 'kinds': kinds, 'road_types': list(ROAD_TYPES),
                  'inputs': list(data.INPUT_NAMES), 'step_hz': STEP_HZ, 'window': window, 'dry_run': dry_run,
                  'run': run, 'seed': seed, 'trained': trained_heads(sides), 'earned': earned}
    torch.save(checkpoint, run_dir / CHECKPOINT)
    result = {'run': run, 'dry_run': dry_run, 'encoder_id': encoder_id(name), 'step_hz': STEP_HZ, 'window_steps': window,
              'kinds': kinds, 'road_types': list(ROAD_TYPES), 'seed': seed, 'history': history,
              'drives': {drive['route']: len(drive['steps']) for drive in drives}, 'train_sides': sides, 'gate': earned,
              **{partition: measure(drives, said, partition) for partition in PARTITIONS}}
    (run_dir / RESULT).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def trained_heads(sides: dict) -> list[str]:
    """The kinds (and 'works') that had both sides in train (road_labeler_data.both_sides): the only heads that learned."""
    return [name for name, item in sides.items() if item['inside'] and item['outside']]


def held_back(checkpoint: dict) -> dict[str, str]:
    """{head: why it writes nothing into Speed review} of a run's heads (the kinds, 'works', 'type:<road type>'): not
    trained with both sides (a head masked in training says chances from its first weights, and a mark of it would teach
    the brain noise), or not through the gate of validation (earned_heads; a checkpoint without one, made before it,
    writes nothing). Their chances still go into the chances file."""
    trained = set(checkpoint.get('trained') or ())
    earned = checkpoint.get('earned') if isinstance(checkpoint.get('earned'), dict) else {}
    found = {}
    for head in (*checkpoint['kinds'], 'works', *(f'type:{name}' for name in checkpoint.get('road_types', ROAD_TYPES))):
        if not head.startswith('type:') and head not in trained:
            found[head] = 'not trained with both sides'
        elif head not in earned:
            found[head] = 'no gate of validation in the checkpoint'
        elif earned[head].get('why'):
            found[head] = earned[head]['why']
    return found


def writable(checkpoint: dict) -> tuple[list[str], bool, list[str]]:
    """(the kinds whose sure stretches may be written, whether the roadworks may, the road types that may): the heads
    not held_back."""
    held = held_back(checkpoint)
    return ([kind for kind in checkpoint['kinds'] if kind not in held], 'works' not in held,
            [name for name in checkpoint.get('road_types', ROAD_TYPES) if f'type:{name}' not in held])


def load_labeler(run_dir: Path, device):
    """(model, encoder, checkpoint) of a run."""
    import torch
    from nnslr_tools.road_labeler_net import build_labeler_net
    checkpoint = torch.load(Path(run_dir) / CHECKPOINT, map_location=device, weights_only=False)
    model = build_labeler_net(checkpoint['feature_dim'], len(checkpoint['kinds'])).to(device)
    model.load_state_dict(checkpoint['weights'])
    encoder = _encoder_for(checkpoint['encoder'], device, checkpoint.get('encoder_weights'))
    return model.eval(), encoder, checkpoint


def chance_lines(drive: dict, said: dict, checkpoint: dict) -> list[dict]:
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
                    marks: bool = True, dry_run: bool = False, device: str | None = None, log=_say) -> list[dict]:
    """The chances of each drive of routes into out_dir (derived/road_labeler by default) and, with marks, Prepare Speed
    review on each (prepare_drive: the marks of the heads that earned it, the layer of road types). span_of(route): the
    drive's (first, last) ns, None before a scan. marks False: the chances only, nothing written into Speed review.
    dry_run, or a checkpoint of a dry run: the first DRY_STEPS steps only for dry_run, the chances in <run>/chances,
    nothing written into Speed review. What a run wrote, head by head, goes to <run>/prepared.json (PREPARED) for the
    portal's card, again after each drive prepared (a job stopped part-way, by a drive or by the panel's SIGTERM, which
    skips any finally, still says what it wrote), and only once a drive is prepared: a Prepare that prepared none leaves
    the run that Speed review shows as it was. Returns a summary a drive."""
    import torch
    root, run_dir = Path(data_root), Path(run_dir)
    made_dry = torch.load(run_dir / CHECKPOINT, map_location='cpu', weights_only=False)['dry_run']
    trial = dry_run or made_dry
    device = torch.device(device or ('cuda' if torch.cuda.is_available() and not trial else 'cpu'))
    model, encoder, checkpoint = load_labeler(run_dir, device)
    target = run_dir / 'chances' if trial else Path(out_dir) if out_dir else root / CHANCES_RELATIVE
    cache = run_dir / 'features' if trial else root / FEATURES_RELATIVE / checkpoint['encoder_id']
    run = checkpoint['run']
    kinds = tuple(checkpoint['kinds'])
    marked, works_earned, types_earned = writable(checkpoint)
    earned = set(marked) | ({None} if works_earned else set())
    held = held_back(checkpoint)
    if held:
        log('Never written (their chances are in the chances file): ' +
            '; '.join(f'{head}: {why}' for head, why in held.items()))
    store = None
    from nnslr_tools.speed_review import SpeedReviewError
    if not trial and marks:
        from nnslr_tools.speed_review import store_for
        store = store_for(root)
    if trial and marks:
        log('A dry run: nothing is written into Speed review.')
    tunnels = data.tunnel_ways(root)
    found = []
    totals: dict = {}

    def save_prepared() -> None:
        drives = sum(1 for item in found if 'prepared' in item)
        if store is None or not drives:
            return
        prepared = {'run': run, 'at': datetime.now().astimezone().isoformat(timespec='seconds'), 'drives': drives,
                    'failed': sum(1 for item in found if 'error' in item),
                    'writes': sorted(head if head is not None else 'works' for head in earned)
                    + [f'type:{name}' for name in types_earned], 'heads': totals}
        from nnslr_tools.annotations import atomic_write
        atomic_write(run_dir / PREPARED, json.dumps(prepared, indent=2).encode('utf-8'))

    try:
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
            atomic_write(path, jsonl_bytes(chance_lines(drive, said, checkpoint)))
            summary = {'route_id': route, 'steps': len(drive['steps']), 'chances': str(path)}
            stretches = {kind: sure_stretches(drive['ns'], said['kinds'][:, column]) for column, kind in enumerate(kinds)}
            stretches[None] = sure_stretches(drive['ns'], said['works'])
            chance_of, _ns_of = frame_chances(drive, said)
            types = layer_types(drive['map_lines'], chance_of, types_earned)
            summary['sure'] = {**{kind: len(items) for kind, items in stretches.items() if kind is not None},
                               'works': len(stretches[None]), 'road_types': len(types)}
            summary['held_back'] = held
            if store is not None:
                try:
                    summary['prepared'] = prepare_drive(store, root, route, run, span, stretches, earned, types, types_earned)
                except (SpeedReviewError, sqlite3.Error, OSError) as exc:
                    summary['error'] = f'Speed review not prepared: {exc}'
                for head, item in (summary.get('prepared') or {}).items():
                    if isinstance(item, dict):
                        total = totals.setdefault(head, {'written': 0, 'removed': 0, 'kept': 0, 'left': 0})
                        for key in total:
                            total[key] += item.get(key, 0)
                    else:
                        totals[head] = totals.get(head, 0) + item
            log(json.dumps(summary))
            found.append(summary)
            if 'prepared' in summary:
                save_prepared()
    finally:
        # Also when a drive stops the job part-way (an exception): the failures after the last drive prepared.
        save_prepared()
    return found
