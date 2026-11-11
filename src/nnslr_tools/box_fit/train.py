"""Training of the local box fit (``nnslr train-box-fit``): both rankers learn from the boxes people drew.

The boxes: every sign box of ``annotations/objects.jsonl`` a person confirmed (label basis ``human_review`` or a
reviewer that is not ``ai/...``), kept (accepted or corrected) and drawn by them (proposal source ``human``) or
moved (the box differs from the model's proposal); not ``not_a_sign``, at least 4 px a side, once per image and
box. A box a person merely accepted is left out: it is the model's drawing, not a person's.

Each box gets three seeded rough boxes, as a reviewer draws them: loose (each side pushed out by 10-40 % of the
box), shifted (moved by up to 25 %, each side then +-15 %) and tight (each side +-8 %). The candidates of every
rough box are extracted once (in ``workers`` processes) and labelled with their IoU with the person's box; then

- the colour model (:mod:`.gbdt`, 300 trees of depth 6) learns to predict that IoU from the candidate's features
  and their difference from the rough box's;
- the edges model (three MLPs, seeds ``seed``, ``seed + 1``, ``seed + 2``) learns listwise to rank them.

``folds`` > 1 first measures the whole fit on routes left out: the routes are split into that many groups, and the
boxes of each group are fitted by models trained on the other groups only (the weights are held out; the two blend
constants of :mod:`nnslr_tools.box_fit`, chosen once on the benchmark, are not). The models written to ``colour.npz`` and
``edges.npz`` are then trained on every box. The candidates are extracted on the CPU (OpenCV, ``workers`` processes);
the trees and the MLPs train on the GPU (:mod:`.gpu`, PyTorch: the environment of .venv-gpu), never on the CPU. The
runtime reads the files with numpy alone.
"""
from __future__ import annotations

import collections
import json
import os
import random
import statistics
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from nnslr_tools import box_fit
from nnslr_tools.box_fit import colour, edges, gbdt, gpu

CASE_SEED = 20261030      # the rough boxes of each person's box (the benchmark's)
KINDS = ('loose', 'shifted', 'tight')
COLOUR_TREES, COLOUR_DEPTH = 300, 6
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))
MIN_BOXES = 20            # fewer boxes drawn by people: nothing to train


def _iou(a, b) -> float:
    return box_fit._iou(a, b)


def human_boxes(objects_path: Path, data_root: Path) -> list[dict]:
    """The boxes people drew or moved (see the module's description), sorted by annotation id."""
    out, seen = [], set()
    with open(objects_path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            b = r.get('bbox_xyxy')
            provenance = r.get('provenance') or {}
            reviewer = r.get('reviewer') or ''
            if reviewer.startswith('ai/'):
                continue
            if not (provenance.get('label_basis') == 'human_review' or reviewer):
                continue
            if not isinstance(b, list) or len(b) != 4 or r.get('review_state') not in ('accepted', 'corrected'):
                continue
            original = (provenance.get('original_proposal') or {}).get('bbox_xyxy')
            drawn = r.get('proposal_source') == 'human' or (original is not None and original != b)
            family = r.get('sign_family')
            if not drawn or family in (None, 'not_a_sign') or b[2] - b[0] < 4 or b[3] - b[1] < 4:
                continue
            key = (r.get('image_sha256'), tuple(round(v) for v in b))
            if key in seen:
                continue
            path = data_root / r['image_path']
            if not path.exists():
                continue
            seen.add(key)
            side = min(b[2] - b[0], b[3] - b[1])
            out.append({'annotation_id': r['annotation_id'], 'image_path': str(path),
                        'route': r.get('route_id') or Path(r['image_path']).parts[-3], 'family': family,
                        'size': 'small' if side < 20 else ('mid' if side < 50 else 'big'),
                        'width': int(r.get('width') or 1344), 'height': int(r.get('height') or 760),
                        'truth': [float(v) for v in b]})
    out.sort(key=lambda item: item['annotation_id'])
    return out


def rough_boxes(truth, rng: random.Random, width: int, height: int) -> dict:
    """The loose, shifted and tight rough boxes of ``truth``, clipped to the image."""
    x1, y1, x2, y2 = truth
    w, h = x2 - x1, y2 - y1
    out = {'loose': [x1 - rng.uniform(.10, .40) * w, y1 - rng.uniform(.10, .40) * h,
                     x2 + rng.uniform(.10, .40) * w, y2 + rng.uniform(.10, .40) * h]}
    dx, dy = rng.uniform(-.25, .25) * w, rng.uniform(-.25, .25) * h
    out['shifted'] = [x1 + dx + rng.uniform(-.15, .15) * w, y1 + dy + rng.uniform(-.15, .15) * h,
                      x2 + dx + rng.uniform(-.15, .15) * w, y2 + dy + rng.uniform(-.15, .15) * h]
    out['tight'] = [x1 + rng.uniform(-.08, .08) * w, y1 + rng.uniform(-.08, .08) * h,
                    x2 + rng.uniform(-.08, .08) * w, y2 + rng.uniform(-.08, .08) * h]
    clipped = {}
    for kind, (a, b, c, d) in out.items():
        a, c = sorted((min(max(a, 0.0), width), min(max(c, 0.0), width)))
        b, d = sorted((min(max(b, 0.0), height), min(max(d, 0.0), height)))
        clipped[kind] = [round(a, 1), round(b, 1), round(c, 1), round(d, 1)]
    return clipped


def make_cases(items: list[dict]) -> list[dict]:
    """Three rough boxes per person's box, each seeded by its annotation id: the same boxes give the same cases."""
    cases = []
    for item in items:
        rng = random.Random(f"{CASE_SEED}:{item['annotation_id']}")
        for kind, rough in rough_boxes(item['truth'], rng, item['width'], item['height']).items():
            cases.append({**item, 'kind': kind, 'rough': rough})
    return cases


def route_folds(items: list[dict], folds: int) -> dict:
    """{route: fold}: whole routes, the largest first into the fold with the fewest boxes so far."""
    counts = collections.Counter(item['route'] for item in items)
    sizes = [0] * folds
    out = {}
    for route, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        fold = min(range(folds), key=lambda f: (sizes[f], f))
        out[route] = fold
        sizes[fold] += n
    return out


def _labels(boxes, truth) -> np.ndarray:
    return np.array([_iou(b, truth) for b in boxes], np.float32)


def _init_worker():
    import cv2
    cv2.setNumThreads(1)


def extract_case(case: dict) -> dict | None:
    """The candidates of both rankers for one case, labelled with their IoU with the person's box."""
    img = box_fit.read_image(case['image_path'])
    height, width = img.shape[:2]
    rough = box_fit.clip_box(case['rough'], width, height)
    if rough is None:
        return None
    t = time.perf_counter()
    colour_boxes, colour_rows = colour.extract(img, rough)
    edges_boxes, edges_features = edges.extract(img, rough)
    ms = (time.perf_counter() - t) * 1000.0
    return {'annotation_id': case['annotation_id'], 'route': case['route'], 'kind': case['kind'],
            'family': case['family'], 'size': case['size'], 'truth': case['truth'], 'rough': rough,
            'width': width, 'height': height, 'ms': ms,
            'colour_boxes': colour_boxes, 'colour_rows': colour_rows, 'colour_y': _labels(colour_boxes, case['truth']),
            'edges_boxes': edges_boxes, 'edges_features': edges_features.astype(np.float32),
            'edges_y': _labels(edges_boxes, case['truth'])}


def extract_all(cases: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        _init_worker()
        rows = [extract_case(case) for case in cases]
    else:
        with get_context('fork').Pool(workers, initializer=_init_worker) as pool:
            rows = pool.map(extract_case, cases, chunksize=8)
    return [row for row in rows if row is not None]


def train_colour(rows: list[dict], seed: int, dev) -> dict:
    X = np.concatenate([row['colour_rows'] for row in rows])
    y = np.concatenate([row['colour_y'] for row in rows])
    return gbdt.pack(gpu.train_trees(X, y, n_trees=COLOUR_TREES, depth=COLOUR_DEPTH, seed=seed, dev=dev))


def train_edges(rows: list[dict], seed: int, dev) -> list:
    cases = [(row['edges_features'], row['edges_y']) for row in rows]
    return gpu.train_rankers(cases, [seed + s for s in range(edges.ENSEMBLE)], dev=dev)


def fit_row(row: dict, colour_model, edges_model) -> list[float] | None:
    """The fit of an extracted case, as :func:`nnslr_tools.box_fit.fit_image` gives it."""
    by_colour = colour.pick(row['colour_boxes'], row['colour_rows'], colour_model, row['width'], row['height'])
    if by_colour is None:
        return None
    by_edges = edges.pick(row['edges_boxes'], row['edges_features'], edges_model, row['rough'], row['width'],
                          row['height'])
    return box_fit.combine(by_colour, by_edges, row['rough'])


def measures(rows: list[dict], boxes: list) -> dict:
    """Mean IoU with the person's box (a None keeps the rough box), and how often the fit made a box worse."""
    fitted = [_iou(row['rough'] if box is None else box, row['truth']) for row, box in zip(rows, boxes)]
    rough = [_iou(row['rough'], row['truth']) for row in rows]
    n = max(1, len(rows))
    return {'cases': len(rows), 'mean_iou': round(float(np.mean(fitted)), 4) if rows else None,
            'rough_mean_iou': round(float(np.mean(rough)), 4) if rows else None,
            'iou_ge_0.7': round(sum(v >= 0.7 for v in fitted) / n, 4),
            'iou_ge_0.85': round(sum(v >= 0.85 for v in fitted) / n, 4),
            'worse_by_0.05': round(sum(f < r - 0.05 for f, r in zip(fitted, rough)) / n, 4),
            'kept': round(sum(box is None for box in boxes) / n, 4)}


def _by(rows, boxes, key) -> dict:
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[key]].append(i)
    return {name: measures([rows[i] for i in idx], [boxes[i] for i in idx]) for name, idx in sorted(groups.items())}


def cross_validate(rows: list[dict], folds: dict, seed: int, dev, log=print) -> dict:
    """Each fold's boxes fitted by models trained on the other folds only; the measures over all folds."""
    boxes = [None] * len(rows)
    per_fold = {}
    for fold in sorted(set(folds.values())):
        train = [row for row in rows if folds[row['route']] != fold]
        test = [i for i, row in enumerate(rows) if folds[row['route']] == fold]
        t = time.time()
        colour_model, edges_model = train_colour(train, seed, dev), train_edges(train, seed, dev)
        for i in test:
            boxes[i] = fit_row(rows[i], colour_model, edges_model)
        per_fold[str(fold)] = {'routes': sorted(r for r, f in folds.items() if f == fold),
                               **measures([rows[i] for i in test], [boxes[i] for i in test])}
        log(f'fold {fold}: {len(test)} cases, mean IoU {per_fold[str(fold)]["mean_iou"]} '
            f'(rough {per_fold[str(fold)]["rough_mean_iou"]}), trained in {time.time() - t:.0f} s')
    return {'folds': per_fold, 'overall': measures(rows, boxes), 'by_kind': _by(rows, boxes, 'kind'),
            'by_size': _by(rows, boxes, 'size'), 'by_family': _by(rows, boxes, 'family')}


def _card_name(dev) -> str:
    import torch
    return torch.cuda.get_device_name(dev)


def save(out_dir: Path, colour_model: dict, edges_model: list) -> dict:
    """Write colour.npz and edges.npz into ``out_dir`` (each replaced in one step); their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {'colour.npz': {k: np.asarray(v) for k, v in colour_model.items()}, 'edges.npz': {}}
    for s, (mu, sd, ps) in enumerate(edges_model):
        arrays['edges.npz'][f'mu{s}'] = mu
        arrays['edges.npz'][f'sd{s}'] = sd
        for i, p in enumerate(ps):
            arrays['edges.npz'][f'p{s}_{i}'] = p
    paths = {}
    for name, data in arrays.items():
        target = out_dir / name
        tmp = out_dir / f'.{name}.tmp.npz'
        np.savez(tmp, **data)
        os.replace(tmp, target)
        paths[name] = str(target)
    return paths


def run(data_root: Path, out_dir: Path, *, objects: Path | None = None, seed: int = 0,
        workers: int = DEFAULT_WORKERS, folds: int = 4, dry_run: bool = False, device: str = 'cuda',
        log=print) -> dict:
    """Select the boxes, measure with ``folds`` route groups (when > 1), train on every box and save the models."""
    items = human_boxes(objects or data_root / 'annotations' / 'objects.jsonl', data_root)
    groups = route_folds(items, folds) if folds > 1 else {}
    summary = {'boxes': len(items), 'cases': len(items) * len(KINDS), 'routes': len({i['route'] for i in items}),
               'by_family': dict(sorted(collections.Counter(i['family'] for i in items).items())),
               'by_size': dict(sorted(collections.Counter(i['size'] for i in items).items())),
               'min_boxes': MIN_BOXES, 'folds': folds if folds > 1 else 0, 'seed': seed, 'out': str(out_dir)}
    if groups:
        summary['boxes_by_fold'] = {str(f): sum(groups[i['route']] == f for i in items) for f in range(folds)}
    if dry_run:
        return {'dry_run': True, **summary}
    if len(items) < MIN_BOXES:
        raise ValueError(f'only {len(items)} boxes drawn by people: too few to train the box fit')
    # The card first: no card, no extraction of half an hour for nothing.
    dev = gpu.device(device)
    log(f'training on {dev} ({_card_name(dev)})')
    t = time.time()
    rows = extract_all(make_cases(items), workers)
    ms = [row['ms'] for row in rows]
    summary['extracted_cases'] = len(rows)
    summary['extract_ms_median'] = round(statistics.median(ms), 1)
    log(f'{len(rows)} cases extracted in {time.time() - t:.0f} s')
    if groups:
        summary['cross_validation'] = cross_validate(rows, groups, seed, dev, log=log)
    t = time.time()
    colour_model, edges_model = train_colour(rows, seed, dev), train_edges(rows, seed, dev)
    log(f'trained on every box in {time.time() - t:.0f} s')
    summary['in_sample'] = measures(rows, [fit_row(row, colour_model, edges_model) for row in rows])
    summary['written'] = save(out_dir, colour_model, edges_model)
    return summary
