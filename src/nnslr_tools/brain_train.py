"""The brain's training: a person's labels into the weights, and what it is worth on sites it never learned. Started only on
request (nnslr train-brain); a dry run is two steps on the CPU, nothing kept but its own folder.

An epoch goes drive after drive in a random order, each drive in its order with the state carried from piece to piece, so
that approach, crossing and exit are one sequence and the network sees only the past, as on the car. The pieces move their
cuts every epoch, and the gradient stops at each of them.

A frame out of train never moves a weight: the pieces stop at it (brain_targets.learning_pieces) and it runs under no_grad,
only to keep the memory going. A train frame right after a frame of test would otherwise send its error back through that
picture.

The measure (measure), on validation to choose the epoch and on test once at the end, frames with an answer only:

- speed: right, and the mean error in km/h, on all of them and on those without gas in the last second: the frames where
  the driver gives nothing away, so a network that follows the pedal shows here;
- in_works: the same inside the marks;
- works: precision and recall of works_head, on the frames with an answer of its own;
- sites: per mark of the partition, how long after his end tap the speed reaches what he labelled after it, and how much
  of the mark was above his label: said too early, the end of works that is not there.

The comparison he asked for is the same training twice, with works and with works=False, on the same split. The driver's
inputs stay out unless driver=True: see brain_targets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools.brain_net import brain_loss, build_brain_net
from nnslr_tools.brain_targets import (DRIVER_INPUTS, INPUT_SIZES, NS, class_kph, dataset, drive_chunks, learning_pieces,
                                       loss_weights, pack_inputs)
from nnslr_tools.road_type_net import (BACKBONE_LEARNING_RATE, FrameStore, LEARNING_RATE, WEIGHT_DECAY, _augment, _print,
                                       _torch)

EPOCHS = 40
PATIENCE = 10
CHUNK = 64
CLIP = 5.0
CHECKPOINT = 'brain-best.pt'
RESULT = 'result.json'
END_WINDOW_S = 60
DRY_FRAMES = 12
DRY_STEPS = 2
ROAD_INPUTS = tuple(name for name in INPUT_SIZES if name not in DRIVER_INPUTS)


def _piece(drive: dict, first: int, last: int, store: FrameStore, device):
    torch = _torch()
    images = store.batch([(drive['route'], list(range(first, last)))])
    return images, torch.from_numpy(drive['extra'][first:last])[None].to(device)


def train_epoch(model, optimizer, drives: list[dict], store: FrameStore, *, chunk: int, rng: random.Random, device,
                augment: bool = True, steps: int | None = None) -> float:
    """One pass over the train frames: the mean error of its steps."""
    torch = _torch()
    model.train()
    errors = []
    for drive in rng.sample(drives, len(drives)):
        speed_weight, works_weight = loss_weights(drive, 'train')
        state = None
        for first, last, learn in learning_pieces(drive['partitions'], chunk, rng.randrange(chunk)):
            images, extra = _piece(drive, first, last, store, device)
            answers = speed_weight[first:last].sum() + works_weight[first:last].sum()
            if not learn or not answers or (steps is not None and len(errors) >= steps):
                with torch.no_grad():
                    _speed, _works, state = model(images, extra, state)
                continue
            speed_logits, works_logit, state = model(_augment(torch, images) if augment else images, extra, state)
            loss = brain_loss(speed_logits, works_logit,
                              *(torch.from_numpy(values[first:last])[None].to(device) for values in
                                (drive['speed'], speed_weight, drive['works'], works_weight)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            state = state.detach()
            errors.append(float(loss.detach()))
    return float(np.mean(errors)) if errors else 0.0


def predict(model, drive: dict, store: FrameStore, *, chunk: int, device) -> dict:
    """What the network says along a whole drive, as on the car: {'kph': per frame, 'works': chance per frame, or None}."""
    torch = _torch()
    model.eval()
    state, speed, works = None, [], []
    with torch.no_grad():
        for first, last in drive_chunks(len(drive['frames']), chunk):
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


def _site(drive: dict, said: dict, mark: dict, truth: np.ndarray, asked: np.ndarray) -> dict:
    """A mark: the frames inside it above his label, and how long after its end the speed reaches what he said after it."""
    ns = drive['ns']
    inside = (ns >= mark['start_ns']) & (ns < mark['end_ns']) & asked
    after = np.flatnonzero((ns >= mark['end_ns']) & asked)
    site = {'route': drive['route'], 'frames': int(inside.sum()),
            'above_label': float((said['kph'][inside] > truth[inside]).mean()) if inside.any() else None, 'end_delay_s': None}
    if len(after) and inside.any() and truth[after[0]] > truth[inside][-1]:
        late = ns < mark['end_ns'] + END_WINDOW_S * NS
        reached = np.flatnonzero((ns >= mark['end_ns']) & late & (said['kph'] >= truth[after[0]]))
        if len(reached):
            site['end_delay_s'] = float((ns[reached[0]] - mark['end_ns']) / NS)
    return site


def measure(drives: list[dict], answers: list[dict], partition: str) -> dict:
    """What the network is worth on one partition: see the module's text."""
    said_all, truth_all, free, works_frames, sites = [], [], [], [], []
    hits = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    for drive, said in zip(drives, answers):
        speed_weight, works_weight = loss_weights(drive, partition)
        asked = speed_weight > 0
        truth = class_kph(drive['speed'])
        said_all.append(said['kph'][asked])
        truth_all.append(truth[asked])
        free.append(drive['inputs']['pedals'][asked, 0] == 0)
        works_frames.append(drive['works'][asked] > 0)
        if said['works'] is not None:
            known = works_weight > 0
            yes, real = said['works'][known] >= .5, drive['works'][known] > 0
            hits['true_positive'] += int((yes & real).sum())
            hits['false_positive'] += int((yes & ~real).sum())
            hits['false_negative'] += int((~yes & real).sum())
        held = np.array([name == partition for name in drive['partitions']])
        for mark in drive['marks']:
            middle = np.flatnonzero((drive['ns'] >= mark['start_ns']) & (drive['ns'] < mark['end_ns']))
            if len(middle) and held[middle[len(middle) // 2]]:
                sites.append(_site(drive, said, mark, truth, drive['speed_weight'] > 0))
    said_all, truth_all = np.concatenate(said_all or [[]]), np.concatenate(truth_all or [[]])
    free, works_frames = np.concatenate(free or [[]]).astype(bool), np.concatenate(works_frames or [[]]).astype(bool)
    found = hits['true_positive']
    return {'speed': _speed_measure(said_all, truth_all), 'no_gas': _speed_measure(said_all[free], truth_all[free]),
            'in_works': _speed_measure(said_all[works_frames], truth_all[works_frames]),
            'works': {**hits, 'precision': found / max(found + hits['false_positive'], 1),
                      'recall': found / max(found + hits['false_negative'], 1)}, 'sites': sites}


def _short(drive: dict, length: int) -> dict:
    """The first length frames from the first answer of train on: for a dry run."""
    first = next((i for i, name in enumerate(drive['partitions']) if name == 'train' and drive['speed_weight'][i]), 0)
    part = slice(first, first + length)
    cut = {name: drive[name][part] for name in ('frames', 'ns', 'partitions', 'cells', 'speed', 'speed_out', 'speed_weight',
                                                'works', 'works_weight')}
    return {**drive, **cut, 'inputs': {name: value[part] for name, value in drive['inputs'].items()}}


def _pictures(root: Path, drives: list[dict], *, dry_run: bool, log) -> dict:
    """{route: uint8 (frames, H, W, 3)}: the frame cache (made when missing), or from the PNGs for a dry run, nothing written."""
    if dry_run:
        return {drive['route']: np.stack([data.resized_frame(frame['image']) if frame['present'] else
                                          np.zeros((data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
                                          for frame in drive['frames']]) for drive in drives}
    paths = data.build_frame_cache(root, {drive['route']: drive['frames'] for drive in drives}, log=log)
    return {route: np.load(path) for route, path in paths.items()}


def train_brain(data_root: Path, run_dir: Path, merged_of, *, pool: str = 'mean', works: bool = True, driver: bool = False,
                epochs: int = EPOCHS, patience: int = PATIENCE, chunk: int = CHUNK, device: str | None = None,
                dry_run: bool = False, seed: int = 0, log=_print) -> dict:
    """Train on train, choose the epoch on validation, measure test once. Writes brain-best.pt and result.json into run_dir."""
    torch = _torch()
    root, run_dir = Path(data_root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    groups = tuple(INPUT_SIZES) if driver else ROAD_INPUTS
    drives = dataset(root, merged_of)
    if dry_run:
        drives = [_short(drive, DRY_FRAMES) for drive in drives[:1]]
        epochs, chunk = 1, DRY_FRAMES // 2
    for drive in drives:
        drive['extra'] = pack_inputs(drive['inputs'], groups)
    if not any(loss_weights(drive, 'train')[0].any() for drive in drives):
        raise ValueError('no speed labels in train')
    if works and not any(loss_weights(drive, 'train')[1].any() for drive in drives):
        log('No answer on roadworks in train: the works task teaches nothing yet.')
    store = FrameStore(_pictures(root, drives, dry_run=dry_run, log=log), device)
    model = build_brain_net(groups, pool=pool, works=works, pretrained=not dry_run).to(device)
    backbone = [value for name, value in model.named_parameters() if name.startswith('backbone')]
    others = [value for name, value in model.named_parameters() if not name.startswith('backbone')]
    optimizer = torch.optim.AdamW([{'params': others, 'lr': LEARNING_RATE}, {'params': backbone, 'lr': BACKBONE_LEARNING_RATE}],
                                  weight_decay=WEIGHT_DECAY)
    best, best_epoch, history = None, None, []
    for epoch in range(epochs):
        error = train_epoch(model, optimizer, drives, store, chunk=chunk, rng=rng, device=device,
                            steps=DRY_STEPS if dry_run else None)
        answers = [predict(model, drive, store, chunk=chunk, device=device) for drive in drives]
        validation = measure(drives, answers, 'validation')
        score = validation['speed']['mean_error_kph']
        history.append({'epoch': epoch, 'error': error, 'validation': validation['speed']})
        log(f'Epoch {epoch}: error {error:.3f}, validation {score} km/h off on {validation["speed"]["frames"]} frames')
        if best is None or (score is not None and score < best):
            best, best_epoch = score if score is not None else float('inf'), epoch
            torch.save({'weights': model.state_dict(), 'groups': groups, 'pool': pool, 'works': works}, run_dir / CHECKPOINT)
        elif epoch - best_epoch >= patience:
            break
    saved = torch.load(run_dir / CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(saved['weights'])
    answers = [predict(model, drive, store, chunk=chunk, device=device) for drive in drives]
    result = {'groups': list(groups), 'pool': pool, 'works': works, 'dry_run': dry_run, 'best_epoch': best_epoch, 'history': history,
              'validation': measure(drives, answers, 'validation'), 'test': measure(drives, answers, 'test')}
    (run_dir / RESULT).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
