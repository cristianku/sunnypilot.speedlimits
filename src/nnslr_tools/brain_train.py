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

- speed: right, and the mean error in km/h, on all of them and on those without gas in the last second: the frames where
  the driver gives nothing away, so a network that follows the pedal shows here;
- speeds_said: every speed the network said there: one alone is a network that learned only how often each speed comes;
- one_speed: the error of the speed most frames have, said everywhere: what the network has to beat;
- follow_the_sign: on the frames with a number in force, the error of that number and the network's own there: the other
  thing it has to beat, or it adds nothing to the car's rule;
- far_from_own_speed: the same on the frames where the car's speed is OWN_SPEED_GAP_KPH or more from his label, and how
  many of them the network puts nearer the car's speed than the label. Leaving pedals and speed out of the inputs is not
  enough: the video shows how fast the car goes, and at the end of works the label and the car's speed rise together, so
  a network that follows the car looks right there. Here following the car is the wrong answer;
- in_works: the same inside the marks;
- works: precision and recall of works_head, on the frames with an answer of its own, None where the partition has no
  frame to tell them by: with no "no" a constant yes would read 1.0;
- sites: per mark of the partition, how long after his end tap the speed reaches what he labelled after it, and how much
  of the mark was above his label: said too early, the end of works that is not there.

predict goes through a whole drive as on the car, so a held-out stretch starts from the memory of the train frames before it,
and a network that only holds its last speed scores well there. So the result has each partition twice: as on the car, and
cold, with the memory starting again where the stretch out of train begins.

test_swapped and validation_swapped ask the network what the swap asks in training, on roads it never saw: the frames whose
passage says its number, with that number changed to the other of its family. A network deaf to the sign is still right on the
frames that go to the speed it says, so moved counts the answers that changed: the network reads the sign when they follow it.

The comparison he asked for is the same training twice, with works and with works=False, on the same split. The driver's
inputs are in unless driver=False: see brain_targets.
"""
from __future__ import annotations

import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools.brain_net import brain_loss, build_brain_net
from nnslr_tools.brain_targets import (DRIVER_INPUTS, INPUT_SIZES, NS, check_drives, class_kph, dataset, epoch_drives,
                                       hours_in_train, inputs_sha256, learning_pieces, loss_weights, night_frames, pack_inputs,
                                       plates_in_train, read_split, sign_kph, swap_passages, swapped, wet_in_train)
from nnslr_tools.road_type_net import FEATURES, FrameStore, HIDDEN, LEARNING_RATE, WEIGHT_DECAY, _augment, _print, _torch

EPOCHS = 40
CHUNK = 64
CLIP = 5.0
CHECKPOINT = 'brain-last.pt'
RESULT = 'result.json'
END_WINDOW_S = 60
OWN_SPEED_GAP_KPH = 20
DRY_FRAMES = 12
DRY_STEPS = 2
ROAD_INPUTS = tuple(name for name in INPUT_SIZES if name not in DRIVER_INPUTS)


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
        look, state = rng.getrandbits(32), None
        for first, last, learn, fresh in learning_pieces(drive['partitions'], chunk, rng.randrange(chunk), drive['cuts']):
            state = None if fresh else state
            images, extra = _piece(drive, first, last, store, device)
            images = _same_augment(torch, images, look, device) if augment else images
            answers = speed_weight[first:last].sum() + works_weight[first:last].sum()
            if not learn or not answers or (steps is not None and len(errors) >= steps):
                with torch.no_grad():
                    _speed, _works, state = model(images, extra, state)
                continue
            speed_logits, works_logit, state = model(images, extra, state)
            loss = brain_loss(speed_logits, works_logit,
                              *(torch.from_numpy(values[first:last])[None].to(device) for values in
                                (drive['speed'], speed_weight, drive['works'], works_weight)), frames=chunk)
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
    """What the network says along a whole drive, as on the car: {'kph': per frame, 'works': chance per frame, or None}.
    cold: the memory also starts again where a stretch out of train begins."""
    torch = _torch()
    model.eval()
    cuts = drive['cuts'] | (_stretches_out_of_train(drive['partitions']) if cold else set())
    state, speed, works = None, [], []
    with torch.no_grad():
        for first, last, _learn, fresh in learning_pieces(['train'] * len(drive['frames']), chunk, cuts=cuts):
            state = None if fresh else state
            images, extra = _piece(drive, first, last, store, device)
            logits, works_logit, state = model(images, extra, state)
            speed.append(logits[0].argmax(dim=-1).cpu().numpy())
            if works_logit is not None:
                works.append(torch.sigmoid(works_logit[0]).cpu().numpy())
    return {'kph': class_kph(np.concatenate(speed)), 'works': np.concatenate(works) if works else None}


def _speed_measure(said: np.ndarray, truth: np.ndarray) -> dict:
    if not len(truth):
        return {'frames': 0, 'right': None, 'mean_error_kph': None}
    return {'frames': int(len(truth)), 'right': float((said == truth).mean()), 'mean_error_kph': float(np.abs(said - truth).mean())}


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


def _site(drive: dict, said: dict, mark: dict, truth: np.ndarray, asked: np.ndarray) -> dict:
    """A mark: the frames inside it above his label, and how long after its end the speed reaches what he said after it."""
    ns = drive['ns']
    inside = (ns >= mark['start_ns']) & (ns < mark['end_ns']) & asked
    after = np.flatnonzero((ns >= mark['end_ns']) & (ns < mark['end_ns'] + END_WINDOW_S * NS) & asked)
    site = {'route': drive['route'], 'frames': int(inside.sum()),
            'above_label': float((said['kph'][inside] > truth[inside]).mean()) if inside.any() else None, 'end_delay_s': None}
    if len(after) and inside.any() and truth[after[0]] > truth[inside][-1]:
        late = ns < mark['end_ns'] + END_WINDOW_S * NS
        reached = np.flatnonzero((ns >= mark['end_ns']) & late & (said['kph'] >= truth[after[0]]))
        if len(reached):
            site['end_delay_s'] = float((ns[reached[0]] - mark['end_ns']) / NS)
    return site


def _in_force(signs: np.ndarray) -> np.ndarray:
    return (signs[:, 1] > 0) & (signs[:, 6] == 0)


def measure(drives: list[dict], answers: list[dict], partition: str) -> dict:
    """What the network is worth on one partition: see the module's text."""
    said_all, truth_all, own_all, number_all, free, works_frames, sites = [], [], [], [], [], [], []
    hits = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    told = {'yes': 0, 'no': 0}
    for drive, said in zip(drives, answers):
        speed_weight, works_weight = loss_weights(drive, partition)
        asked = speed_weight > 0
        truth = class_kph(drive['speed'])
        said_all.append(said['kph'][asked])
        truth_all.append(truth[asked])
        own_all.append(drive['own_kph'][asked])
        signs = drive['inputs']['signs'][asked]
        number_all.append(np.where(_in_force(signs), sign_kph(signs), -1))
        free.append(drive['inputs']['pedals'][asked, 0] == 0)
        works_frames.append(drive['works'][asked] > 0)
        if said['works'] is not None:
            known = works_weight > 0
            yes, real = said['works'][known] >= .5, drive['works'][known] > 0
            hits['true_positive'] += int((yes & real).sum())
            hits['false_positive'] += int((yes & ~real).sum())
            hits['false_negative'] += int((~yes & real).sum())
            told['yes'] += int(real.sum())
            told['no'] += int((~real).sum())
        held = np.array([name == partition for name in drive['partitions']])
        for mark in drive['marks']:
            if held[(drive['ns'] >= mark['start_ns']) & (drive['ns'] < mark['end_ns'])].any():
                sites.append(_site(drive, said, mark, truth, drive['speed_weight'] > 0))
    said_all, truth_all, own_all, number_all = (np.concatenate(part or [[]]) for part in
                                                (said_all, truth_all, own_all, number_all))
    free, works_frames = np.concatenate(free or [[]]).astype(bool), np.concatenate(works_frames or [[]]).astype(bool)
    found = hits['true_positive']
    return {'speed': _speed_measure(said_all, truth_all), 'speeds_said': sorted({int(kph) for kph in said_all}),
            'one_speed': _one_speed(truth_all), 'follow_the_sign': _follow_the_sign(said_all, truth_all, number_all),
            'no_gas': _speed_measure(said_all[free], truth_all[free]),
            'far_from_own_speed': _far_from_own_speed(said_all, truth_all, own_all),
            'in_works': _speed_measure(said_all[works_frames], truth_all[works_frames]),
            'works': {**hits, 'precision': found / max(found + hits['false_positive'], 1) if told['no'] else None,
                      'recall': found / max(found + hits['false_negative'], 1) if told['yes'] else None}, 'sites': sites}


def _short(drive: dict, length: int) -> dict:
    """The first length frames from the first answer of train on: for a dry run."""
    first = next((i for i, name in enumerate(drive['partitions']) if name == 'train' and drive['speed_weight'][i]), 0)
    part = slice(first, first + length)
    cut = {name: drive[name][part] for name in ('frames', 'ns', 'partitions', 'cells', 'speed', 'speed_out', 'speed_weight',
                                                'works', 'works_weight', 'own_kph')}
    return {**drive, **cut, 'inputs': {name: value[part] for name, value in drive['inputs'].items()},
            'cuts': {at - first for at in drive['cuts'] if first < at < first + length}}


def _pictures(root: Path, drives: list[dict], *, dry_run: bool, log) -> dict:
    """{route: uint8 (frames, H, W, 3)}: the frame cache (made when missing), or from the PNGs for a dry run, nothing written."""
    if dry_run:
        return {drive['route']: np.stack([data.resized_frame(frame['image']) if frame['present'] else
                                          np.zeros((data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
                                          for frame in drive['frames']]) for drive in drives}
    paths = data.build_frame_cache(root, {drive['route']: drive['frames'] for drive in drives}, log=log)
    return {route: np.load(path) for route, path in paths.items()}


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
            asked = (loss_weights(held[-1], partition)[0] > 0) & _in_force(held[-1]['inputs']['signs'])
            moved += int((said[-1]['kph'][asked] != before['kph'][asked]).sum())
    return {**measure(held, said, partition)['follow_the_sign'], 'moved': moved}


def train_brain(data_root: Path, run_dir: Path, merged_of, *, pool: str = 'grid', works: bool = True, driver: bool = True,
                epochs: int = EPOCHS, chunk: int = CHUNK, device: str | None = None, dry_run: bool = False, seed: int = 0,
                log=_print) -> dict:
    """Train on train, keep the last epoch, measure once. Writes brain-last.pt and result.json into run_dir; the checkpoint
    says what it came from. Refuses what would train wrong: brain_targets.check_drives."""
    torch = _torch()
    root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    groups = tuple(INPUT_SIZES) if driver else ROAD_INPUTS
    drives = dataset(root, merged_of, log=log)
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
    store = FrameStore(_pictures(root, drives, dry_run=dry_run, log=log), device)
    model = build_brain_net(groups, pool=pool, works=works, pretrained=not dry_run).to(device)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    plates = 'plate' in groups and plates_in_train(drives)
    hours = 'plate_hours' in groups and hours_in_train(drives)
    wet = 'plate_wet' in groups and wet_in_train(drives)
    source = {'groups': groups, 'pool': pool, 'works': works, 'features': FEATURES, 'hidden': HIDDEN, 'plates': plates,
              'hours': hours, 'wet': wet,
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
        right = validation['speed']['right']
        log(f'{name} epoch {epoch + 1}/{epochs} train_loss {error:.4f} val_error_kph {"-" if score is None else f"{score:.2f}"} '
            f'val_accuracy {"-" if right is None else f"{right:.4f}"} on {validation["speed"]["frames"]} frames '
            f'(always {validation["one_speed"]["kph"]}: {validation["one_speed"]["mean_error_kph"]}, '
            f'the sign: {validation["follow_the_sign"]["sign"]["mean_error_kph"]}), says {validation["speeds_said"]} '
            f'{time.monotonic() - started:.0f}s')
    torch.save({'weights': model.state_dict(), 'epoch': epochs - 1, **source}, run_dir / CHECKPOINT)
    cold = [predict(model, drive, store, chunk=chunk, device=device, cold=True) for drive in drives]
    swapped_measure = partial(_swapped_measure, model, drives, answers, groups=groups, store=store, chunk=chunk, device=device)
    result = {'groups': list(groups), 'pool': pool, 'works': works, 'plates': plates, 'hours': hours, 'wet': wet, 'dry_run': dry_run,
              'seed': seed, 'history': history,
              'answers': {drive['route']: int((drive['speed_weight'] > 0).sum()) for drive in drives},
              'train': measure(drives, answers, 'train'), 'validation': measure(drives, answers, 'validation'),
              'test': measure(drives, answers, 'test'), 'validation_cold': measure(drives, cold, 'validation'),
              'test_cold': measure(drives, cold, 'test'),
              'validation_swapped': swapped_measure('validation'), 'test_swapped': swapped_measure('test')}
    (run_dir / RESULT).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
