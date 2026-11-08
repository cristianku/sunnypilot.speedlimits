"""The brain's training: a person's labels into the weights, and what it is worth on sites it never learned. Started only on
request (nnslr train-brain); a dry run is two steps on the CPU, nothing kept but its own folder.

An epoch goes drive after drive in a random order, each drive in its order with the state carried from piece to piece, so
that approach, crossing and exit are one sequence and the network sees only the past, as on the car. The pieces move their
cuts every epoch, and the gradient stops at each of them. Every epoch the drives get another clock and half of their numbers
swapped, and those with a plate of hours go through once more at night (brain_targets.epoch_drives): the measure has only
the real hour and the real numbers.

The last epoch is what is kept. Validation is a few stretches of two drives today: choosing on it rewards the epoch that
remembers those roads, the opposite of what the swap asks. It is shown every epoch and it chooses nothing.

A picture of test or validation runs in no training step and in no memory of one: the training skips those frames, and the
memory starts again after them (brain_targets.learning_pieces). A frame of the buffer runs under no_grad, only to keep the
memory going. What does cross from a held-out stretch into train is the number read there, as an input, the way the car
carries it: never its picture, never its answer.

The measure (measure), at the end on train, validation and test, frames with an answer only; train against test is the gap
between roads seen and not seen:

- speed: right (within RIGHT_KPH of the answer: at a km/h a class nobody says the dashboard's speed to the km/h, and on the
  5 km/h table of the brains before it is the same as the class itself), and the mean error in km/h, on all of them and on
  those without gas in the last second: the frames where the driver gives nothing away, so a network that follows the pedal
  shows here;
- speeds_said: every speed the network said there: one alone is a network that learned only how often each speed comes;
- one_speed: the error of the speed most frames have, said everywhere: what the network has to beat;
- follow_the_sign: on the frames with a number in force, the error of that number and the network's own there: the other
  thing it has to beat, or it adds nothing to the car's rule;
- far_from_own_speed: the same on the frames where the car's speed is OWN_SPEED_GAP_KPH or more from his label, and how
  many of them the network puts nearer the car's speed than the label. Leaving pedals and speed out of the inputs is not
  enough: the video shows how fast the car goes, and at the end of works the label and the car's speed rise together, so
  a network that follows the car looks right there. Here following the car is the wrong answer;
- in_works: the same inside the marks;
- in_blocks: the same inside the blocks of Speed review with a speed, the only frames whose answer is a person's correction
  and not the speed he drove: everywhere else his speed is the answer, and a network that only knows how fast the car goes
  is right there. copy_own_speed: of those frames, where the car's speed is more than RIGHT_KPH from the block's, the share
  where the network said the car's speed (within RIGHT_KPH) and not the block's;
- works: precision and recall of works_head, on the frames with an answer of its own, None where the partition has no
  frame to tell them by: with no "no" a constant yes would read 1.0;
- road: the same for each kind of road (road_head), and in_road the speed inside each kind's marks;
- types: the share of frames on which type_head names the road type right, on all and per type, None without the head;
  in_type the speed on each type, the head or not;
- sites: per mark of the partition, how long after his end tap the speed reaches (within RIGHT_KPH) the speed he drove
  after it, and how much of the mark was above his label: said too early, the end of works that is not there. The speed
  after is the 90th percentile of his answers in END_WINDOW_S after the tap, the one he reaches, not his first answer there,
  which is still on the way up; there is an end only where it is more than RIGHT_KPH above his median inside the mark.

predict goes through a whole drive as on the car, so a held-out stretch starts from the memory of the train frames before it,
and a network that only holds its last speed scores well there. So the result has each partition twice: as on the car, and
cold, with the memory starting again where the stretch out of train begins.

test_swapped and validation_swapped ask the network what the swap asks in training, on roads it never saw: the frames whose
passage says its number (brain_targets.swap_passages: within SWAP_TOLERANCE_KPH), with that number changed to the other of its
family and his answers moved as much. A network deaf to the sign is still right on the frames that go to the speed it says, so
moved counts the answers that changed by more than RIGHT_KPH: the network reads the sign when they follow it.

The brain steps at RATE_HZ, every reading of the car (brain_targets): its pictures are those vision.BrainInput makes on the
car, kept while the signs are read. Each measure counts one frame a second, the frames of the old 1 Hz brains, so that the
numbers of a brain of today and of one before read the same. ahead, the limit ahead on frames with its answer: right (none
included), changes (of the frames before a change, those that say its speed), false_ahead (of the frames with none, those
that say a limit), metres_error (mean, where the limit is right). The next limit is a sign's number: right is its class,
exact.

With types (the usual since 8 Nov 2026, his rule) only the seconds with a road type teach anything (brain_targets'
types): the measure is on those too, so a brain with types and one without measure different frames.

The comparison he asked for is the same training twice, with works and with works=False, on the same split. The driver's
inputs (brain_targets.DRIVER_INPUTS, his pedals and the car's speed) are out unless driver=True, since 8 Nov 2026: the
answer is his own speed, and with the car's among the inputs the network learns to say it back, on the car its own.
"""
from __future__ import annotations

import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools.brain_net import brain_loss, build_brain_net, outputs
from nnslr_tools.brain_targets import (DRIVER_INPUTS, INPUT_SIZES, LEFT_OUT, METRES_SCALE, NONE_AHEAD, NO_TYPE, NS, RATE_HZ,
                                       ahead_weights,
                                       check_drives, class_kph, dataset, epoch_drives, hours_in_train, inputs_sha256,
                                       learning_pieces, loss_weights, night_frames, pack_inputs, plates_in_train, read_split,
                                       road_weights, sign_kph, swap_passages, swapped, type_weights, wet_in_train)
from nnslr_tools.road_types import ROAD_TYPES
from nnslr_tools.road_type_net import FEATURES, FrameStore, HIDDEN, LEARNING_RATE, WEIGHT_DECAY, _augment, _print, _torch
from nnslr_tools.road_type_signs import load_pictures

EPOCHS = 40
CHUNK = int(64 * RATE_HZ)  # a piece of 64 s, as the 64 frames of the 1 Hz brains
CLIP = 5.0
CHECKPOINT = 'brain-last.pt'
RESULT = 'result.json'
END_WINDOW_S = 60
OWN_SPEED_GAP_KPH = 20
RIGHT_KPH = 2  # a speed said this near the answer is right (the site says "within 2 km/h")
AFTER_PERCENTILE = 90  # the speed he reaches after the end of works (_site)
DRY_FRAMES = 12
DRY_STEPS = 2
LEARNED_INPUTS = tuple(name for name in INPUT_SIZES if name not in LEFT_OUT)
ROAD_INPUTS = tuple(name for name in LEARNED_INPUTS if name not in DRIVER_INPUTS)


def _piece(drive: dict, first: int, last: int, store: FrameStore, device):
    torch = _torch()
    images = store.batch([(drive['route'], list(range(first, last)))])
    return images, torch.from_numpy(drive['extra'][first:last])[None].to(device)


def _same_augment(torch, images, seed: int, device):
    """_augment with the numbers of seed: one look for a whole drive, since the car never sees the light jump between two
    pieces. Only the centre of the contrast is the piece's own, its mean light: a step far smaller than the look."""
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        return _augment(torch, images)


def train_epoch(model, optimizer, drives: list[dict], store: FrameStore, *, chunk: int, rng: random.Random, device,
                augment: bool = True, steps: int | None = None) -> float:
    """One pass over the train frames: the mean error of its steps."""
    torch = _torch()
    model.train()
    errors = []
    for drive in rng.sample(drives, len(drives)):
        speed_weight, works_weight = loss_weights(drive, 'train')
        if model.works_head is None:
            works_weight = np.zeros_like(works_weight)
        ahead = ahead_weights(drive, 'train') if getattr(model, 'ahead', False) else None
        road = road_weights(drive, 'train') if getattr(model, 'road_head', None) is not None else None
        typed = type_weights(drive, 'train') if getattr(model, 'type_head', None) is not None else None
        look, state = rng.getrandbits(32), None
        for first, last, learn, fresh in learning_pieces(drive['partitions'], chunk, rng.randrange(chunk), drive['cuts']):
            state = None if fresh else state
            images, extra = _piece(drive, first, last, store, device)
            images = _same_augment(torch, images, look, device) if augment else images
            answers = speed_weight[first:last].sum() + works_weight[first:last].sum()
            if ahead is not None:
                answers += ahead[0][first:last].sum() + ahead[1][first:last].sum()
            if road is not None:
                answers += road[first:last].sum()
            if typed is not None:
                answers += typed[first:last].sum()
            if not learn or not answers or (steps is not None and len(errors) >= steps):
                with torch.no_grad():
                    _speed, _works, state = model(images, extra, state)
                continue

            def piece(values):
                return torch.from_numpy(values[first:last])[None].to(device)

            asked = {'ahead': ahead is not None, 'road': road is not None, 'types': typed is not None}
            said = outputs(model(images, extra, state, **asked), **asked)
            speed_logits, works_logit, state = said['speed'], said['works'], said['state']
            found = None if ahead is None else (said['next'], said['metres'], *(piece(values) for values in
                                                                                 (drive['next'], ahead[0],
                                                                                  drive['metres'].astype(np.float32),
                                                                                  ahead[1])))
            kinds = None if road is None else (said['road'], piece(drive['road']), piece(road))
            types = None if typed is None else (said['types'], piece(drive['road_type']), piece(typed))
            loss = brain_loss(speed_logits, works_logit,
                              *(piece(values) for values in (drive['speed'], speed_weight, drive['works'], works_weight)),
                              frames=chunk, ahead=found, kph=model.class_kph, road=kinds, types=types)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            state = state.detach()
            errors.append(float(loss.detach()))
    return float(np.mean(errors)) if errors else 0.0


def _stretches_out_of_train(partitions: list[str]) -> set[int]:
    return {at for at in range(1, len(partitions)) if partitions[at] != 'train' and partitions[at - 1] == 'train'}


def predict(model, drive: dict, store: FrameStore, *, chunk: int, device, cold: bool = False) -> dict:
    """What the network says along a whole drive, as on the car: {'kph': per frame, the km/h of its class in the network's
    own table (class_kph), 'works': chance per frame, or None, 'road': chance per frame and kind of road, or None, 'types':
    chance per frame and road type (road_types.ROAD_TYPES), or None}; a network with the limit ahead also 'next' (a class of
    that table per frame, 'none' for none: the last one, len(class_kph)) and 'metres'. cold: the memory also starts again
    where a stretch out of train begins."""
    torch = _torch()
    model.eval()
    asked = {'ahead': getattr(model, 'ahead', False), 'road': getattr(model, 'road_head', None) is not None,
             'types': getattr(model, 'type_head', None) is not None}
    cuts = drive['cuts'] | (_stretches_out_of_train(drive['partitions']) if cold else set())
    state, speed, works, road, types, following, metres = None, [], [], [], [], [], []
    with torch.no_grad():
        for first, last, _learn, fresh in learning_pieces(['train'] * len(drive['frames']), chunk, cuts=cuts):
            state = None if fresh else state
            images, extra = _piece(drive, first, last, store, device)
            said = outputs(model(images, extra, state, **asked), **asked)
            state = said['state']
            if asked['ahead']:
                following.append(said['next'][0].argmax(dim=-1).cpu().numpy())
                metres.append(said['metres'][0].clamp_min(0).cpu().numpy() * METRES_SCALE)
            if asked['road']:
                road.append(torch.sigmoid(said['road'][0]).cpu().numpy())
            if asked['types']:
                types.append(torch.softmax(said['types'][0], dim=-1).cpu().numpy())
            speed.append(said['speed'][0].argmax(dim=-1).cpu().numpy())
            if said['works'] is not None:
                works.append(torch.sigmoid(said['works'][0]).cpu().numpy())
    table = np.rint(model.class_kph.detach().cpu().numpy()).astype(np.int64)
    found = {'kph': table[np.concatenate(speed)], 'works': np.concatenate(works) if works else None,
             'road': np.concatenate(road) if road else None, 'types': np.concatenate(types) if types else None}
    ahead = asked['ahead']
    if ahead:
        found['next'], found['metres'], found['none'] = np.concatenate(following), np.concatenate(metres), len(table)
    return found


def _speed_measure(said: np.ndarray, truth: np.ndarray) -> dict:
    if not len(truth):
        return {'frames': 0, 'right': None, 'mean_error_kph': None}
    error = np.abs(said - truth)
    return {'frames': int(len(truth)), 'right': float((error <= RIGHT_KPH).mean()), 'mean_error_kph': float(error.mean())}


def _one_speed(truth: np.ndarray) -> dict:
    """The speed most frames have, said everywhere."""
    if not len(truth):
        return {'kph': None, 'mean_error_kph': None}
    values, counts = np.unique(truth, return_counts=True)
    kph = values[counts.argmax()]
    return {'kph': int(kph), 'mean_error_kph': float(np.abs(truth - kph).mean())}


def _follow_the_sign(said: np.ndarray, truth: np.ndarray, number: np.ndarray) -> dict:
    """The frames with a number in force (-1: none): that number said there, and the network on the same frames."""
    held = number >= 0
    return {'sign': _speed_measure(number[held], truth[held]), 'network': _speed_measure(said[held], truth[held])}


def _far_from_own_speed(said: np.ndarray, truth: np.ndarray, own: np.ndarray) -> dict:
    """The frames where following the car is the wrong answer. An unknown speed (NaN) is not among them."""
    far = np.abs(own - truth) >= OWN_SPEED_GAP_KPH
    said, truth, own = said[far], truth[far], own[far]
    nearer = float((np.abs(said - own) < np.abs(said - truth)).mean()) if far.any() else None
    return {**_speed_measure(said, truth), 'nearer_own_speed': nearer}


def _in_blocks(said: np.ndarray, truth: np.ndarray, own: np.ndarray) -> dict:
    """The frames inside a block with a speed: the network against the block's, and how often it said the car's instead,
    where the two are apart. An unknown speed of the car (NaN) is not among those."""
    apart = np.abs(own - truth) > RIGHT_KPH
    copied = (np.abs(said - own) <= RIGHT_KPH) & (np.abs(said - truth) > RIGHT_KPH)
    return {**_speed_measure(said, truth), 'copy_own_speed': float(copied[apart].mean()) if apart.any() else None}


def _site(drive: dict, said: dict, mark: dict, truth: np.ndarray, asked: np.ndarray) -> dict:
    """A mark: the frames inside it above his label, and how long after its end the speed reaches the one he drove after it
    (see the module's text)."""
    ns = drive['ns']
    inside = (ns >= mark['start_ns']) & (ns < mark['end_ns']) & asked
    window = (ns >= mark['end_ns']) & (ns < mark['end_ns'] + END_WINDOW_S * NS)
    after = window & asked
    site = {'route': drive['route'], 'frames': int(inside.sum()),
            'above_label': float((said['kph'][inside] > truth[inside]).mean()) if inside.any() else None, 'end_delay_s': None}
    if after.any() and inside.any():
        reached_kph = float(np.percentile(truth[after], AFTER_PERCENTILE))
        if reached_kph > float(np.median(truth[inside])) + RIGHT_KPH:
            reached = np.flatnonzero(window & (said['kph'] >= reached_kph - RIGHT_KPH))
            if len(reached):
                site['end_delay_s'] = float((ns[reached[0]] - mark['end_ns']) / NS)
    return site


def _in_force(signs: np.ndarray) -> np.ndarray:
    return (signs[:, 1] > 0) & (signs[:, 6] == 0)


def _measured(drive: dict) -> np.ndarray:
    """The frames a measure counts: one a second."""
    return drive.get('measured', np.ones(len(drive['frames']), dtype=bool))


def _ahead_measure(drives: list[dict], answers: list[dict], partition: str) -> dict | None:
    """The limit ahead: see the module's text. None for a network without it."""
    said, truth, metres, said_metres, other = [], [], [], [], []
    for drive, answer in zip(drives, answers):
        weights = ahead_weights(drive, partition)
        if 'next' not in answer or weights is None:
            continue
        asked = (weights[0] > 0) & _measured(drive)
        said.append(answer['next'][asked])
        truth.append(drive['next'][asked])
        # Its none is the last class of the network's table (predict); the speed said is no limit to compare it with: it is
        # his speed, the next limit a sign's number.
        other.append(answer['next'][asked] != answer.get('none', NONE_AHEAD))
        timed = (weights[1] > 0) & _measured(drive) & (answer['next'] == drive['next'])
        metres.append(drive['metres'][timed] * METRES_SCALE)
        said_metres.append(answer['metres'][timed])
    if not said:
        return None
    said, truth, metres, said_metres, other = (np.concatenate(part) for part in (said, truth, metres, said_metres, other))
    change, none = truth != NONE_AHEAD, truth == NONE_AHEAD
    return {'frames': int(len(truth)), 'right': float((said == truth).mean()) if len(truth) else None,
            'changes': int(change.sum()), 'changes_right': float((said[change] == truth[change]).mean()) if change.any() else None,
            'false_ahead': float(other[none].mean()) if none.any() else None,
            'metres_error': float(np.abs(said_metres - metres).mean()) if len(metres) else None}


def measure(drives: list[dict], answers: list[dict], partition: str) -> dict:
    """What the network is worth on one partition: see the module's text."""
    said_all, truth_all, own_all, number_all, free, works_frames, blocked, sites = [], [], [], [], [], [], [], []
    hits = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    told = {'yes': 0, 'no': 0}
    kinds = next((drive['road_kinds'] for drive in drives if drive.get('road_kinds')), [])
    road_hits = {kind: ({'true_positive': 0, 'false_positive': 0, 'false_negative': 0}, {'yes': 0, 'no': 0}) for kind in kinds}
    road_frames: dict[str, list] = {kind: [] for kind in kinds}
    type_all, type_said, type_truth = [], [], []
    for drive, said in zip(drives, answers):
        speed_weight, works_weight = loss_weights(drive, partition)
        asked = (speed_weight > 0) & _measured(drive)
        type_all.append(drive['road_type'][asked] if 'road_type' in drive else np.full(int(asked.sum()), -1))
        typed = type_weights(drive, partition)
        if typed is not None and said.get('types') is not None:
            known = (typed > 0) & _measured(drive)
            type_said.append(said['types'][known].argmax(axis=-1))
            type_truth.append(drive['road_type'][known])
        road_weight = road_weights(drive, partition)
        for column, kind in enumerate(kinds):
            if road_weight is None:
                road_frames[kind].append(np.zeros(int(asked.sum()), dtype=bool))
                continue
            road_frames[kind].append(drive['road'][asked, column] > 0)
            if said.get('road') is not None:
                _count(said['road'][:, column], drive['road'][:, column], (road_weight[:, column] > 0) & _measured(drive),
                       *road_hits[kind])
        truth = class_kph(drive['speed'])
        said_all.append(said['kph'][asked])
        truth_all.append(truth[asked])
        own_all.append(drive['own_kph'][asked])
        signs = drive['inputs']['signs'][asked]
        number_all.append(np.where(_in_force(signs), sign_kph(signs), -1))
        free.append(drive['inputs']['pedals'][asked, 0] == 0)
        works_frames.append(drive['works'][asked] > 0)
        blocked.append(drive['blocked'][asked] if 'blocked' in drive else np.zeros(int(asked.sum()), dtype=bool))
        if said['works'] is not None:
            _count(said['works'], drive['works'], (works_weight > 0) & _measured(drive), hits, told)
        held = np.array([name == partition for name in drive['partitions']])
        for mark in drive['marks']:
            if held[(drive['ns'] >= mark['start_ns']) & (drive['ns'] < mark['end_ns'])].any():
                sites.append(_site(drive, said, mark, truth, (drive['speed_weight'] > 0) & _measured(drive)))
    said_all, truth_all, own_all, number_all = (np.concatenate(part or [[]]) for part in
                                                (said_all, truth_all, own_all, number_all))
    free, works_frames, blocked = (np.concatenate(part or [[]]).astype(bool) for part in (free, works_frames, blocked))
    road_frames = {kind: np.concatenate(part or [[]]).astype(bool) for kind, part in road_frames.items()}
    type_all = np.concatenate(type_all or [[]]).astype(np.int64)
    ahead = _ahead_measure(drives, answers, partition)
    return {**({'ahead': ahead} if ahead else {}),
            'speed': _speed_measure(said_all, truth_all), 'speeds_said': sorted({int(kph) for kph in said_all}),
            'one_speed': _one_speed(truth_all), 'follow_the_sign': _follow_the_sign(said_all, truth_all, number_all),
            'no_gas': _speed_measure(said_all[free], truth_all[free]),
            'far_from_own_speed': _far_from_own_speed(said_all, truth_all, own_all),
            'in_works': _speed_measure(said_all[works_frames], truth_all[works_frames]),
            'in_blocks': _in_blocks(said_all[blocked], truth_all[blocked], own_all[blocked]),
            'works': _told(hits, told), 'sites': sites,
            'road': {kind: _told(*road_hits[kind]) for kind in kinds},
            'in_road': {kind: _speed_measure(said_all[road_frames[kind]], truth_all[road_frames[kind]]) for kind in kinds},
            'types': _types_measure(type_said, type_truth),
            'in_type': {kind: _speed_measure(said_all[type_all == column], truth_all[type_all == column])
                        for column, kind in enumerate(ROAD_TYPES) if (type_all == column).any()}}


def _types_measure(said: list[np.ndarray], truth: list[np.ndarray]) -> dict | None:
    """type_head on the frames with a road type: the share it names right, on all and per type; None for a network
    without it."""
    if not said:
        return None
    said, truth = np.concatenate(said), np.concatenate(truth)
    per_type = {kind: {'frames': int((truth == column).sum()),
                       'right': float((said[truth == column] == column).mean()) if (truth == column).any() else None}
                for column, kind in enumerate(ROAD_TYPES) if (truth == column).any()}
    return {'frames': int(len(truth)), 'right': float((said == truth).mean()) if len(truth) else None, 'per_type': per_type}


def _count(chance: np.ndarray, answer: np.ndarray, known: np.ndarray, hits: dict, told: dict) -> None:
    """A head of yes or no (works_head, a kind of road_head) against the answers on the frames known."""
    yes, real = chance[known] >= .5, answer[known] > 0
    hits['true_positive'] += int((yes & real).sum())
    hits['false_positive'] += int((yes & ~real).sum())
    hits['false_negative'] += int((~yes & real).sum())
    told['yes'] += int(real.sum())
    told['no'] += int((~real).sum())


def _told(hits: dict, told: dict) -> dict:
    """Precision and recall of _count, None where the partition has no frame to tell them by."""
    found = hits['true_positive']
    return {**hits, 'precision': found / max(found + hits['false_positive'], 1) if told['no'] else None,
            'recall': found / max(found + hits['false_negative'], 1) if told['yes'] else None}


def _short(drive: dict, length: int) -> dict:
    """The first length frames from the first answer of train on: for a dry run."""
    first = next((i for i, name in enumerate(drive['partitions']) if name == 'train' and drive['speed_weight'][i]), 0)
    part = slice(first, first + length)
    cut = {name: drive[name][part] for name in ('frames', 'ns', 'partitions', 'cells', 'speed', 'speed_out', 'speed_weight',
                                                'works', 'works_weight', 'road', 'road_weight', 'road_type',
                                                'type_weight', 'typed_by_person', 'own_kph', 'measured',
                                                'next', 'next_weight',
                                                'metres', 'metres_weight', 'blocked') if name in drive}
    return {**drive, **cut, 'inputs': {name: value[part] for name, value in drive['inputs'].items()},
            'cuts': {at - first for at in drive['cuts'] if first < at < first + length}}


class _BrainPictures:
    """The brain's pictures of a drive at RATE_HZ, read only when FrameStore copies them: one drive in memory at a time."""

    def __init__(self, root: Path, drive: dict, dry_run: bool):
        self.root, self.drive, self.dry_run = root, drive, dry_run

    def __len__(self) -> int:
        return len(self.drive['frames'])

    def __array__(self, *_args, **_options) -> np.ndarray:
        """uint8 (frames, H, W, 3): the rows of the segments' pictures (road_type_signs.picture_path), zeros where one is
        missing; in a dry run, a frame without its picture has the one of its second (the 1 Hz PNG)."""
        frames = self.drive['frames']
        found = np.zeros((len(frames), data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
        segments: dict[int, list[int]] = {}
        for at, frame in enumerate(frames):
            segments.setdefault(frame['segment'], []).append(at)
        for segment, rows in segments.items():
            pictures = load_pictures(self.root, self.drive['route'], segment)
            kept = [at for at in rows if frames[at]['present']] if pictures is not None else []
            if kept:
                found[kept] = pictures[[frames[at]['picture'] for at in kept]]
            if self.dry_run:
                for at in set(rows) - set(kept):
                    if Path(frames[at]['image']).is_file():
                        found[at] = data.resized_frame(frames[at]['image'])
        return found


def _pictures(root: Path, drives: list[dict], *, dry_run: bool, log) -> dict:
    """{route: uint8 (frames, H, W, 3)} for FrameStore: at RATE_HZ the brain's pictures (_BrainPictures); at 1 Hz the frame
    cache (made when missing), or from the PNGs for a dry run, nothing written."""
    brain = {drive['route']: _BrainPictures(root, drive, dry_run) for drive in drives if drive.get('rate', 1.0) == RATE_HZ}
    drives = [drive for drive in drives if drive['route'] not in brain]
    if not drives:
        return brain
    if dry_run:
        return {**brain, **{drive['route']: np.stack([data.resized_frame(frame['image']) if frame['present'] else
                                                       np.zeros((data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
                                                       for frame in drive['frames']]) for drive in drives}}
    paths = data.build_frame_cache(root, {drive['route']: drive['frames'] for drive in drives}, log=log)
    return {**brain, **{route: np.load(path) for route, path in paths.items()}}


def _works_needs_both(drives: list[dict], log) -> None:
    """With only marks in train, or nothing, the works task leaves the loss of train: its marks alone would teach a constant
    yes to the memory the speed shares."""
    weights = [loss_weights(drive, 'train')[1] for drive in drives]
    yes = sum(float((weight * drive['works']).sum()) for weight, drive in zip(weights, drives))
    no = sum(float((weight * (1 - drive['works'])).sum()) for weight, drive in zip(weights, drives))
    if yes and no:
        return
    log('No answer on roadworks in train: the works task teaches nothing yet.' if not yes and not no else
        f'Roadworks in train: {int(yes)} frames inside and {int(no)} outside, one side alone teaches nothing: left out. '
        'A mark needs "checked to here" in Speed review.')
    for drive in drives:
        drive['works_weight'] = drive['works_weight'] * (np.array(drive['partitions']) != 'train')


def _road_needs_both(drives: list[dict], log) -> None:
    """_works_needs_both for each kind of road: a kind with only marks in train, or only frames without, leaves the loss of
    train; a kind with nothing says nothing."""
    kinds = next((drive['road_kinds'] for drive in drives if drive.get('road_kinds')), [])
    road = [(road_weights(drive, 'train'), drive) for drive in drives if 'road_weight' in drive]
    for column, kind in enumerate(kinds):
        yes = sum(float((weight[:, column] * drive['road'][:, column]).sum()) for weight, drive in road)
        no = sum(float((weight[:, column] * (1 - drive['road'][:, column])).sum()) for weight, drive in road)
        if yes and no:
            log(f'Road "{kind}" in train: {int(yes)} frames inside and {int(no)} outside.')
            continue
        if yes or no:
            log(f'Road "{kind}" in train: {int(yes)} frames inside and {int(no)} outside, one side alone teaches nothing: '
                'left out. A mark needs "checked to here" in Speed review.')
        for _weight, drive in road:
            drive['road_weight'][:, column] *= np.array(drive['partitions']) != 'train'


def _types_in_train(drives: list[dict], log) -> None:
    """What the road type leaves to learn from, in the log: the seconds of train with a type and those cut without one.
    Only the drives built with the road types (drive_targets' types) count."""
    drives = [drive for drive in drives if drive.get('types')]
    if not drives:
        return
    measured = [(drive, _measured(drive) & (np.array(drive['partitions']) == 'train')) for drive in drives]
    typed = sum(int((drive['type_weight'] * here).sum()) for drive, here in measured)
    cut = sum(int((drive['speed_out'][here] == NO_TYPE).sum()) for drive, here in measured)
    bare = [drive['route'] for drive in drives if not drive['type_weight'].any()]
    log(f'Road types in train: {typed} seconds with one; {cut} seconds with a speed and no road type, left out.')
    if bare:
        log(f"No road type on {len(bare)} drive{'s' if len(bare) > 1 else ''} ({', '.join(bare)}): nothing of "
            f"{'them' if len(bare) > 1 else 'it'} is learned. Read the map in Data › Prepare.")


def _swapped_measure(model, drives: list[dict], answers: list[dict], partition: str, groups: tuple[str, ...], store: FrameStore, *,
                     chunk: int, device) -> dict:
    """follow_the_sign on the frames of a partition whose number was swapped, and how many answers moved: see the module's
    text."""
    held, said, moved = [], [], 0
    for drive, before in zip(drives, answers):
        passages = swap_passages(drive, partition)
        if passages:
            inside = np.zeros_like(drive['speed_weight'])
            inside[np.concatenate(passages)] = 1
            held.append({**swapped(drive, passages, groups), 'speed_weight': drive['speed_weight'] * inside})
            said.append(predict(model, held[-1], store, chunk=chunk, device=device))
            asked = (loss_weights(held[-1], partition)[0] > 0) & _in_force(held[-1]['inputs']['signs']) & _measured(drive)
            moved += int((np.abs(said[-1]['kph'][asked] - before['kph'][asked]) > RIGHT_KPH).sum())
    return {**measure(held, said, partition)['follow_the_sign'], 'moved': moved}


def _spoken(speeds: list[int]) -> str:
    """The speeds said, for the log: how many and from where to where (the whole list is in result.json)."""
    if not speeds:
        return 'nothing'
    return f'{len(speeds)} speed{"s" if len(speeds) > 1 else ""}, {speeds[0]}-{speeds[-1]} km/h'


def train_brain(data_root: Path, run_dir: Path, merged_of, *, pool: str = 'grid', works: bool = True, driver: bool = False,
                road: bool = True, types: bool = True, epochs: int = EPOCHS, chunk: int = CHUNK, device: str | None = None,
                dry_run: bool = False, seed: int = 0, log=_print) -> dict:
    """Train on train, keep the last epoch, measure once. Writes brain-last.pt and result.json into run_dir; the checkpoint
    says what it came from, its table of the speeds too (the class_kph of its weights), and its kinds of road. Refuses what
    would train wrong: brain_targets.check_drives. driver: with his pedals and the car's speed among the inputs, the
    comparison. road: with the kinds of road of Speed review as a task of their own (road_head); False, the comparison.
    types: with the road type as a task of its own (type_head), and only the seconds that have one (brain_targets'
    types); False, the comparison."""
    torch = _torch()
    root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    groups = LEARNED_INPUTS if driver else ROAD_INPUTS
    drives = dataset(root, merged_of, log=log, rate=RATE_HZ, need_pictures=not dry_run, types=types)
    if dry_run:
        drives = [_short(drive, DRY_FRAMES) for drive in drives[:1]]
        epochs, chunk = 1, DRY_FRAMES // 2
    else:
        check_drives(drives)
    for drive in drives:
        drive['extra'] = pack_inputs(drive['inputs'], groups)
    if not any(loss_weights(drive, 'train')[0].any() for drive in drives):
        raise ValueError('no speed labels in train')
    if works:
        _works_needs_both(drives, log)
    kinds = list(drives[0].get('road_kinds', ())) if road and drives else []
    if kinds:
        _road_needs_both(drives, log)
    if types:
        _types_in_train(drives, log)
    store = FrameStore(_pictures(root, drives, dry_run=dry_run, log=log), device)
    model = build_brain_net(groups, pool=pool, works=works, ahead=True, pretrained=not dry_run, road_kinds=len(kinds),
                            road_types=len(ROAD_TYPES) if types else 0).to(device)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    plates = 'plate' in groups and plates_in_train(drives)
    hours = 'plate_hours' in groups and hours_in_train(drives)
    wet = 'plate_wet' in groups and wet_in_train(drives)
    # rate_hz and picture: how the car steps it (brain_export); a checkpoint without them is a 1 Hz brain of PIL pictures.
    source = {'groups': groups, 'pool': pool, 'works': works, 'road_kinds': kinds,
              'road_types': list(ROAD_TYPES) if types else [], 'features': FEATURES, 'hidden': HIDDEN, 'plates': plates,
              'hours': hours, 'wet': wet, 'ahead': getattr(model, 'ahead', False),
              'rate_hz': drives[0].get('rate', 1.0), 'picture': 'brain' if drives[0].get('rate', 1.0) == RATE_HZ else 'road_type',
              'split': read_split(root)['sha256'], 'reader': drives[0]['reader'], 'inputs_sha256': inputs_sha256()}
    log(f'Frames with a night answer: {sum(int(night_frames(drive)[0].sum()) for drive in drives)}')
    history = []
    name = 'dry_run' if dry_run else f'brain_s{seed}'
    for epoch in range(epochs):
        started = time.monotonic()
        error = train_epoch(model, optimizer, epoch_drives(drives, groups, rng), store, chunk=chunk, rng=rng, device=device,
                            steps=DRY_STEPS if dry_run else None)
        answers = [predict(model, drive, store, chunk=chunk, device=device) for drive in drives]
        validation = measure(drives, answers, 'validation')
        score = validation['speed']['mean_error_kph']
        history.append({'epoch': epoch, 'error': error, 'validation': validation['speed']})
        # Read by nnslr_tools.job_progress: the seed's curve in Train › Brain.
        right, spoken = validation['speed']['right'], validation['speeds_said']
        log(f'{name} epoch {epoch + 1}/{epochs} train_loss {error:.4f} val_error_kph {"-" if score is None else f"{score:.2f}"} '
            f'val_accuracy {"-" if right is None else f"{right:.4f}"} on {validation["speed"]["frames"]} frames '
            f'(always {validation["one_speed"]["kph"]}: {validation["one_speed"]["mean_error_kph"]}, '
            f'the sign: {validation["follow_the_sign"]["sign"]["mean_error_kph"]}), says {_spoken(spoken)} '
            f'{time.monotonic() - started:.0f}s')
    torch.save({'weights': model.state_dict(), 'epoch': epochs - 1, **source}, run_dir / CHECKPOINT)
    cold = [predict(model, drive, store, chunk=chunk, device=device, cold=True) for drive in drives]
    swapped_measure = partial(_swapped_measure, model, drives, answers, groups=groups, store=store, chunk=chunk, device=device)
    result = {'groups': list(groups), 'pool': pool, 'works': works, 'road_kinds': kinds,
              'road_types': list(ROAD_TYPES) if types else [], 'plates': plates, 'hours': hours, 'wet': wet, 'dry_run': dry_run,
              'seed': seed, 'right_kph': RIGHT_KPH, 'history': history,
              'answers': {drive['route']: int((drive['speed_weight'] > 0).sum()) for drive in drives},
              'train': measure(drives, answers, 'train'), 'validation': measure(drives, answers, 'validation'),
              'test': measure(drives, answers, 'test'), 'validation_cold': measure(drives, cold, 'validation'),
              'test_cold': measure(drives, cold, 'test'),
              'validation_swapped': swapped_measure('validation'), 'test_swapped': swapped_measure('test')}
    (run_dir / RESULT).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
