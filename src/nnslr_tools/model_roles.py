# [model-roles] - START
"""The networks of the pipeline by role, for the Models tab of the site (docs/NEURAL_NETWORK.md): the cutter finds
every box that could be a sign (SignNet, the network the comma runs), the classifier says what each box is, the
decider says whom a sign applies to (today only the prototypes of experiment B).

Every trained artifact under <data root>/runs is listed by role, newest first, with the numbers that judge it, read
from what the training and the evaluation wrote:

- cutter: ``<run>/signnet-best.pt`` with its ``<run>/result.json`` (``nnslr train-signnet``, scripts/run_signnet.sh:
  SignNet, THE cutter, the same network the comma runs): recall overall and by sign width at the cutter's score
  (CUTTER_SCORE), boxes per frame, and the signs found among its BOX_BUDGET best boxes of each frame, what the car
  reads. Only a finished SignNet of the car's band and grid (nnslr_tools.signnet.fits_the_car: 896 x 288, stride 8,
  what the comma's vision.py runs) can be the cutter in use. The old Detectors, ``<run>/detector/detector-best.pt``
  (``<run>/evaluation/detector-test.json``) and ``<run>/detector-sizes/<size>/detector-best.pt``
  (scripts/run_detector_sizes.sh, ``<size>/evaluation/detector-{test,validation}.json``), stay listed to compare,
  measured on the same frames with the same scoring, never chosen again: an old Detector id still in active.json is
  read back and still resolves until a SignNet is chosen;
- classifier: ``<run>/reader/reader-best.pt`` (``<run>/evaluation/reader-test.json``): right answers on the test,
  overall and on the crops a person confirmed; ``<run>/classifier/reader-best.pt`` of scripts/run_classifier.sh, the
  classifier with "not a sign" (``<run>/evaluation/classifier-test.json``, ``<run>/evaluation/chain-test.json``):
  right answers on the sign crops, the "not a sign" crops it rejects, and the chain end to end at the cutter's score
  (the speed signs found and read right, the false boxes per frame that survive it);
- decider: ``<run>/experiment-b/<variant>/<variant>-best.pt`` with its ``result.json`` (nnslr_tools.context_experiment):
  whom the sign applies to on validation and test, and the reading;
- brain: ``<run>/brain-last.pt`` with its ``result.json`` (``nnslr train-brain``): the speed it says against the
  labels of Speed review, on validation and test.

Each artifact has a stable id ``<role>:<run>/<part>``: ``cutter:2026-10-24_1452/detector``,
``cutter:<run>/detector-sizes/640x360``, ``cutter:<run>/signnet``, ``classifier:<run>/reader``,
``classifier:<run>/classifier``, ``decider:<run>/experiment-b/crop_geom``, ``brain:<run>/brain``.

What runs on the comma is <data root>/models/on-car.json, written by ``nnslr on-car --models <clone>``
(record_on_car) from a clone of the published models repository (speed_limits.json and each network's contract):
the id of each network there, checked against the sha256 of its weights on this server, and the commit. The Models
tab marks those versions "On the car"; nothing else reads the file.

The model in use for each role is <data root>/models/active.json, ``{"cutter": {"id", "who", "via", "at"} or null,
"classifier": ..., "decider": ...}``: null until someone picks one, never a default here. Every version says whether
it can be chosen (``selectable``) and, when not, why in a few words (``not_selectable_reason``). Prepare's "ours" proposer
reads the three in use, only when all three are (nnslr_tools.preannotate.our_networks); scripts/run_classifier.sh the
cutter in use.
Choosing copies, exports or deploys nothing. Only ModelIndex.set_active writes, only that file, atomically; the id
must be one of the index, and nothing else of a request reaches the file system.

The files are read through the RunsIndex of the Evaluate tab (its strict names, its checks that a file is inside its
run, its digests kept while a file does not change).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from nnslr_tools.checkpoints import DEFAULT_DETECTOR_ANCHOR_MIN_RATIO, DEFAULT_DETECTOR_INPUT_SIZE
from nnslr_tools.runs import (CHAIN_WEIGHTS, EXPERIMENT_DIR, INPUT_NAMES, RUN_NAME, SIZE_BUCKETS, UNREADABLE,
                              VARIANT_LABELS, VARIANTS, RunsIndex, _dig, _int, _iso, _mtime, _num, _ratio, _size_table,
                              _words, chain_digest, chain_headline, classifier_headline, classifier_result_digest,
                              plan_digest, reader_digest, result_digest)
from nnslr_tools.signnet import CAR_MAX_BOXES, CAR_MIN_SCORE, car_measure, fits_the_car
from nnslr_tools.training import DETECTOR_FAMILIES

ROLES = ('cutter', 'classifier', 'decider', 'brain')
ROLE_LABELS = {'cutter': 'Cutter', 'classifier': 'Classifier', 'decider': 'Applicability', 'brain': 'Brain'}
# The cutter must miss nothing: it is judged on recall at a low score (nnslr_tools.evaluation.CUTTER_SCORE_THRESHOLDS).
CUTTER_SCORE = 0.1
# What a cutter learnt to find (result.json and its reports, "families"): every sign (the speed signs and the other
# signs) or the speed signs only (every cutter trained before it was recorded).
FAMILIES = ('all', 'speed')
# A detector folder of scripts/run_detector_sizes.sh: WIDTHxHEIGHT.
SIZE_NAME = re.compile(r'[1-9][0-9]{0,4}x[1-9][0-9]{0,4}')
_RUN = RUN_NAME.pattern
# SignNet, the cutter: its weights at the root of its run (<run>/signnet-best.pt), its id cutter:<run>/signnet.
SIGNNET_PART = 'signnet'
SIGNNET_ID = re.compile(rf'cutter:({_RUN})/{SIGNNET_PART}')
# The brain (nnslr train-brain): its weights at the root of its run too, its id brain:<run>/brain.
BRAIN_PART = 'brain'
# Any id of the index; the signnet part only for the cutter (role_of).
MODEL_ID = re.compile(rf'(cutter|classifier|decider|brain):({_RUN})/({SIGNNET_PART}|{BRAIN_PART}|detector|'
                      rf'detector-sizes/{SIZE_NAME.pattern}|reader|classifier|{EXPERIMENT_DIR}/{_RUN})')
# The boxes a frame the classifier reads after the cutter in the car (SignNet keeps its epoch with the best recall
# at this many, nnslr_tools.signnet.BOX_BUDGETS): every cutter is also judged on the speed signs among its best 5.
BOX_BUDGET = 5
MAX_ID = 320
ACTIVE_FILE = 'models/active.json'
ON_CAR_FILE = 'models/on-car.json'
# The networks of the published models repository (speed_limits.json, format 2) by the role they play here.
CAR_NETWORKS = {'finder': 'cutter', 'classifier': 'classifier', 'brain': 'brain'}
# A release may have no brain: speedvisiond then reads the signs alone. With a brain it says the limit itself.
OPTIONAL_CAR_NETWORKS = ('brain',)
SIGNNET_WEIGHTS = re.compile(rf'/runs/({_RUN})/{SIGNNET_PART}-best\.pt$')
# The weights file of a version, inside its part (SignNet's is at the root of its run).
WEIGHTS = {'cutter': 'detector-best.pt', 'classifier': 'reader-best.pt'}
BRAIN_WEIGHTS = 'brain-last.pt'
SHA256 = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
# How the person who chose was recognised: signed in on the site, the access token of an agent, 127.0.0.1 without
# sign-in.
VIA = ('sign-in', 'token', 'local')
DECIDER_PARTITIONS = ('validation', 'test')
# Why a version cannot be chosen (not_selectable_reason, short for the site; set_active says it longer).
NOT_SELECTABLE = {
    'not_finished': ('Not finished.', 'This model has not finished its training: its weights may still change.'),
    'compare_only': ('An old Detector: kept to compare.',
                     'An old Detector, kept to compare: the cutter is a SignNet, the network the car runs.'),
    'not_for_the_car': ('Not the comma’s 896 × 288 band at stride 8: it cannot run on the car.',
                        ('This SignNet is not of the comma’s 896 × 288 band at stride 8: the car’s vision.py cannot '
                         'run it, so it cannot be the cutter in use.')),
}
OLD_KIND_NOTICE = 'The cutter in use is an old Detector: choose a SignNet.'


class ModelChoiceError(Exception):
    """A choice of the model in use that is refused: HTTP status, reason, message (as PanelError)."""

    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def _size_label(value: Any) -> str | None:
    """[640, 360] or '640x360' -> '640x360'; None when it is neither."""
    if isinstance(value, str) and SIZE_NAME.fullmatch(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        sides = [_int(side) for side in value]
        if all(side is not None and side > 0 for side in sides):
            return f'{sides[0]}x{sides[1]}'
    return None


def _pretty_size(size: str) -> str:
    return size.replace('x', ' × ')


# -- digests of the files ---------------------------------------------------------------------------------
def _widths(table: Any) -> list[dict] | None:
    """recall_by_width of nnslr_tools.evaluation ({key: {truth, found, recall}}) or the rows of runs._size_table,
    as rows in the order of SIZE_BUCKETS."""
    if isinstance(table, list):
        table = {row.get('key'): row for row in table if isinstance(row, dict)}
    if not isinstance(table, dict):
        return None
    rows = []
    for key, label, _, _ in SIZE_BUCKETS:
        row = table.get(key) if isinstance(table.get(key), dict) else {}
        truth, found = _int(row.get('truth')) or 0, _int(row.get('found')) or 0
        recall = _num(row.get('recall'))
        rows.append({'key': key, 'label': label, 'truth': truth, 'found': found,
                     'recall': recall if recall is not None else _ratio(found, truth)})
    return rows


def _box_counts(metrics: dict) -> dict:
    """Found, reviewed, recall, precision, boxes and false boxes per frame of detector_metrics (or a report)."""
    found, missed, false_boxes = (_int(metrics.get(key)) for key in ('true_positives', 'false_negatives', 'false_positives'))
    truth = _int(metrics.get('truth_box_count'))
    if truth is None and found is not None and missed is not None:
        truth = found + missed
    images = _int(metrics.get('image_count'))
    recall = _num(metrics.get('recall_iou'))
    precision = _num(metrics.get('precision_iou'))
    boxes = _num(metrics.get('predictions_per_image'))
    false_per_frame = _num(metrics.get('false_positives_per_image'))
    if recall is None and found is not None and truth:
        recall = found / truth
    if precision is None and found is not None and false_boxes is not None:
        precision = _ratio(found, found + false_boxes)
    if boxes is None and images and found is not None and false_boxes is not None:
        # Reports written before predictions_per_image: found + false boxes per frame.
        boxes = (found + false_boxes) / images
    if false_per_frame is None and images and false_boxes is not None:
        false_per_frame = false_boxes / images
    return {'images': images, 'truth_boxes': truth, 'found': found, 'false_boxes': false_boxes, 'recall': recall,
            'precision': precision, 'boxes_per_frame': boxes, 'false_per_frame': false_per_frame}


def _families(value: Any) -> str | None:
    return value if value in FAMILIES else None


def _by_family(table: Any) -> dict | None:
    """recall_by_family of a report ({"speed": {truth, found, recall}, "other_sign": ...}): the speed signs keep the
    headline of a cutter that also finds the other signs comparable with one that finds the speed signs only."""
    if not isinstance(table, dict):
        return None
    rows = {}
    for key, row in table.items():
        if isinstance(row, dict):
            truth, found = _int(row.get('truth')) or 0, _int(row.get('found')) or 0
            recall = _num(row.get('recall'))
            rows[str(key)] = {'truth': truth, 'found': found, 'recall': recall if recall is not None else _ratio(found, truth)}
    return rows or None


def _family_group(family: str) -> str:
    """As nnslr_tools.evaluation.family_group: "speed" for the speed families, else the family."""
    return 'speed' if family in DETECTOR_FAMILIES else family


def _budget(found: int, truth: int, by_family: dict | None) -> dict:
    return {'boxes': BOX_BUDGET, 'truth_boxes': truth, 'found': found, 'recall': _ratio(found, truth),
            'by_family': by_family}


def _frames_budget(report: dict) -> dict | None:
    """The signs found among the BOX_BUDGET best boxes of each frame, from the boxes a newer report keeps frame by
    frame (per_frame_boxes of nnslr_tools.evaluation: every box that found a reviewed box, then the best others, best
    first, each with the reviewed box it found in the greedy best-first matching, so the first N boxes found what N
    boxes alone find). None for a report without them."""
    images = report.get('images')
    if not isinstance(report.get('per_frame_boxes'), dict) or not isinstance(images, list) or not images:
        return None
    truth = found = 0
    families: dict[str, list[int]] = {}
    every_family = True
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get('cutter_boxes'), list):
            return None
        hits = set()
        for line in image['cutter_boxes'][:BOX_BUDGET]:
            fields = line.split() if isinstance(line, str) else []
            if len(fields) >= 6 and fields[5].isdigit():
                hits.add(int(fields[5]))
        count = _int(image.get('truth_count'))
        if count is None:
            count = len(image['truth_boxes']) if isinstance(image.get('truth_boxes'), list) else 0
        truth += count
        found += len([index for index in hits if index < count])
        kinds = image.get('truth_families')
        if isinstance(kinds, list) and len(kinds) == count:
            for index, family in enumerate(kinds):
                row = families.setdefault(_family_group(str(family)), [0, 0])
                row[0] += 1
                row[1] += int(index in hits)
        else:
            every_family = False
    by_family = {key: {'truth': whole, 'found': part, 'recall': _ratio(part, whole)}
                 for key, (whole, part) in sorted(families.items())} if every_family and families else None
    return _budget(found, truth, by_family)


def cutter_digest(report: Any) -> dict:
    """A detector evaluation (nnslr_tools.evaluation.evaluate_detector) as the cutter is judged: at CUTTER_SCORE when
    the report measured it (by_score_threshold), else at the report's own threshold; recall by the width of the
    reviewed box, and by family (the speed signs apart) for a cutter that also finds the other signs; the boxes per
    frame passed on; the time per frame; the input size and default boxes it ran with, and what it finds; the signs
    among the BOX_BUDGET best boxes of each frame, when the report kept its boxes frame by frame."""
    if not isinstance(report, dict):
        raise ValueError('not a report')
    chosen = None
    by_score = report.get('by_score_threshold')
    if isinstance(by_score, dict):
        for key, metrics in by_score.items():
            try:
                score = float(key)
            except (TypeError, ValueError):
                continue
            if abs(score - CUTTER_SCORE) < 1e-9 and isinstance(metrics, dict):
                chosen = metrics
    metrics = chosen if chosen is not None else report
    widths = _widths(metrics.get('recall_by_width'))
    if widths is None and chosen is None and isinstance(report.get('images'), list):
        # Reports written before recall_by_width: from the frames, as the Evaluate tab does.
        widths = _widths(_size_table([item for item in report['images'] if isinstance(item, dict)],
                                     _num(report.get('iou_threshold')) or 0.5))
    human = metrics.get('human_reviewed') if isinstance(metrics.get('human_reviewed'), dict) else None
    inference = report.get('inference') if isinstance(report.get('inference'), dict) else {}
    device = inference.get('device_name') or inference.get('device')
    return {
        'partition': str(report.get('partition') or 'test'),
        'score_threshold': CUTTER_SCORE if chosen is not None else _num(report.get('score_threshold')),
        'at_cutter_score': chosen is not None,
        **_box_counts(metrics),
        'widths': widths,
        'human': None if human is None else {key: value for key, value in _box_counts(human).items()
                                             if key in ('images', 'truth_boxes', 'found', 'recall')},
        'ms_per_frame': _num(inference.get('mean_ms_per_frame')),
        'device': str(device) if isinstance(device, str) and device else None,
        'input_size': _size_label(report.get('input_size_label')) or _size_label(report.get('input_size')),
        'anchor_min_ratio': _num(report.get('anchor_min_ratio')),
        'checkpoint_epoch': _int(report.get('checkpoint_epoch')),
        'families': _families(report.get('families')),
        'by_family': _by_family(metrics.get('recall_by_family')),
        'budget': _frames_budget(report),
    }


def network_start_digest(start: Any, split_sha256: Any) -> dict | None:
    """The start of a SignNet or an applicability variant (their result.json "start", --init-from) without its
    path: the run it went on from, that run's epoch, and whether it learnt on another split than this one (its test
    may then hold frames it saw); an old result may say instead whether it was tested on other drives."""
    if not isinstance(start, dict):
        return None
    run = start.get('run')
    if 'same_test_drives' in start:
        another = start['same_test_drives'] is False
    else:
        theirs = start.get('split_sha256')
        another = isinstance(theirs, str) and isinstance(split_sha256, str) and theirs != split_sha256
    return {'run': run if isinstance(run, str) and RUN_NAME.fullmatch(run) else None,
            'epoch': _int(start.get('epoch')), 'another_split': another}


def signnet_digest(result: Any) -> dict:
    """result.json of ``nnslr train-signnet``: how it was trained (no history), and test and validation as a cutter
    is judged (cutter_digest of their by_score_threshold, the cutter's scoring on the cutter's frames), with the signs
    among its BOX_BUDGET best boxes a frame at any score (by_box_budget) and ``car``: what the car reads, those best
    boxes of score 0.3 or more (signnet.car_measure: found, reviewed, recall, boxes and false boxes per frame, by
    family, at score_threshold with max_boxes), and ``car_centre``: the same with the lenient match (IoU 0.5, or the
    found box's centre on the sign and its size 0.5 to 2 times it; None for a result.json written before it)."""
    if not isinstance(result, dict) or result.get('task') != 'signnet':
        raise ValueError('not a SignNet result')
    band = _size_label(result.get('band'))
    families = _families(result.get('families'))
    measured: dict[str, dict | None] = {}
    for partition in DECIDER_PARTITIONS:
        report = result.get(partition)
        if not isinstance(report, dict):
            measured[partition] = None
            continue
        digest = cutter_digest({key: value for key, value in report.items() if key not in ('images', 'per_frame_boxes')}
                               | {'partition': partition, 'input_size': band, 'families': families})
        budget = _dig(report, ('by_box_budget', str(BOX_BUDGET)))
        if isinstance(budget, dict):
            counts = _box_counts(budget)
            digest['budget'] = _budget(counts['found'] or 0, counts['truth_boxes'] or 0,
                                       _by_family(budget.get('recall_by_family')))
        for key, match in (('car', 'iou'), ('car_centre', 'centre')):
            car = car_measure(report, match=match)
            digest[key] = None if car is None else {
                **_box_counts(car), 'by_family': _by_family(car.get('recall_by_family')),
                'score_threshold': CAR_MIN_SCORE, 'max_boxes': CAR_MAX_BOXES}
        measured[partition] = digest
    history = result.get('history')
    return {
        'band': band, 'families': families,
        # What the comma's vision.py runs (nnslr_tools.signnet.fits_the_car): only such a SignNet can be in use.
        'fits_the_car': band is not None and fits_the_car(result.get('band'), result.get('stride')),
        'stride': _int(result.get('stride')), 'fuse_kernel': _int(result.get('fuse_kernel')),
        'parameters': _int(result.get('parameters')),
        'best_epoch': _int(result.get('best_epoch')), 'epochs': _int(result.get('epochs')),
        'epochs_run': len(history) if isinstance(history, list) and history else None,
        'split_sha256': result.get('split_sha256') if isinstance(result.get('split_sha256'), str) else None,
        'start': network_start_digest(result.get('start'), result.get('split_sha256')),
        **measured,
    }


def subset_digest(subset: Any) -> dict:
    """subset.json of a SignNet trained on a share of the training drives (the learning curve)."""
    if not isinstance(subset, dict):
        raise ValueError('not a subset')
    share = _num(subset.get('share'))
    return {'share': share if share is not None and 0 < share <= 1 else None}


def model_result_digest(result: Any) -> dict:
    """result.json of a detector or a reader (nnslr_tools.training): how it was trained, without its history."""
    base = result_digest(result)
    augmentation = result.get('augmentation')
    return {
        'architecture': base['architecture'] or None,
        'initialization': base['initialization'],
        # The epoch whose weights are *-best.pt: recorded by newer trainings, else found in the history.
        'best_epoch': _int(result.get('best_epoch')) or base['best_epoch'],
        'epochs': base['epochs'],
        'epochs_run': len(base['history']) or None,
        'classes': base['classes'],
        'best_validation_loss': base['best_validation_loss'],
        'best_validation_accuracy': base['best_validation_accuracy'],
        'input_size': _size_label(result.get('input_size')),
        'anchor_min_ratio': _num(result.get('anchor_min_ratio')),
        'augmentation': None if 'augmentation' not in result else bool(augmentation),
        'gpu': base['gpu'],
        'families': _families(result.get('families')),
    }


def brain_digest(result: Any) -> dict:
    """result.json of ``nnslr train-brain``: what it was given and how it did on validation and test, the speed it says
    against the labels of Speed review (nnslr_tools.brain_train)."""
    if not isinstance(result, dict) or not isinstance(result.get('groups'), list) or 'test' not in result:
        raise ValueError('not a brain result')

    def speed(partition: str) -> dict | None:
        found = _dig(result, (partition, 'speed'))
        return {key: found.get(key) for key in ('frames', 'right', 'mean_error_kph')} if isinstance(found, dict) else None

    # A result without rate_hz, ahead and kph is a brain of before 8 Nov 2026: 1 Hz, the speed alone, a class every 5 km/h.
    kph = result.get('kph') if isinstance(result.get('kph'), dict) else {'from': 5, 'to': 200, 'step': 5}
    return {'groups': result['groups'], 'seed': result.get('seed'), 'works': result.get('works'), 'pool': result.get('pool'),
            'rate_hz': _num(result.get('rate_hz')) or 1.0, 'ahead': bool(result.get('ahead')), 'hidden': _int(result.get('hidden')),
            'kph': {key: _int(kph.get(key)) for key in ('from', 'to', 'step')},
            'road_kinds': list(result.get('road_kinds') or []), 'road_types': list(result.get('road_types') or []),
            'plates': bool(result.get('plates')), 'hours': bool(result.get('hours')), 'wet': bool(result.get('wet')), 'dry_run': bool(result.get('dry_run')), 'epochs': len(result.get('history') or []),
            'validation': speed('validation'), 'test': speed('test')}


def decider_digest(result: Any) -> dict:
    """result.json of an experiment B variant: whom the sign applies to (the "Another" road found and right,
    macro-F1, abstentions) and the reading, on validation and test."""
    if not isinstance(result, dict):
        raise ValueError('not a result')
    metrics = result.get('metrics') if isinstance(result.get('metrics'), dict) else {}
    partitions: dict[str, dict | None] = {}
    for name in DECIDER_PARTITIONS:
        report = metrics.get(name)
        if not isinstance(report, dict):
            partitions[name] = None
            continue
        other = _dig(report, ('ownership', 'other_road'))
        other = other if isinstance(other, dict) else {}
        partitions[name] = {
            'boxes': _int(report.get('box_count')),
            'reading_accuracy': _num(_dig(report, ('reading', 'accuracy'))),
            'ownership_samples': _int(_dig(report, ('ownership', 'sample_count'))),
            'macro_f1': _num(_dig(report, ('ownership', 'macro_f1'))),
            'abstain_rate': _num(_dig(report, ('ownership', 'abstain_rate'))),
            'other_road_recall': _num(other.get('recall')),
            'other_road_precision': _num(other.get('precision')),
            'other_road_support': _int(other.get('support')),
            'other_road_found': _int(other.get('true_positives')),
        }
    inputs = result.get('inputs')
    text = lambda key: result.get(key) if isinstance(result.get(key), str) else None  # noqa: E731
    return {
        'partitions': partitions,
        'inputs': [INPUT_NAMES.get(str(item), _words(str(item))) for item in inputs] if isinstance(inputs, list) else None,
        'architecture': text('architecture'),
        'initialization': text('initialization'),
        'best_epoch': _int(result.get('best_epoch')),
        'epochs': _int(_dig(result, ('config', 'epochs'))),
        'epochs_run': len(result['history']) if isinstance(result.get('history'), list) and result['history'] else None,
        'limit': _int(result.get('limit')),
        'split_sha256': text('split_sha256'),
        'start': network_start_digest(result.get('start'), result.get('split_sha256')),
    }


def role_of(model_id: Any) -> str | None:
    """The role of an id of the index (None when it is not one): the signnet part is the cutter's alone, the brain part
    the brain's, and the brain has no other."""
    if not isinstance(model_id, str) or len(model_id) > MAX_ID:
        return None
    match = MODEL_ID.fullmatch(model_id)
    if match is None or (match.group(3) == SIGNNET_PART and match.group(1) != 'cutter'):
        return None
    if (match.group(3) == BRAIN_PART) != (match.group(1) == 'brain'):
        return None
    return match.group(1)


def selectable(entry: dict) -> tuple[bool, str | None]:
    """Whether a version of the index can be the one in use, else the key of NOT_SELECTABLE: the cutter only a
    SignNet that fits the car (the old Detectors are kept to compare), every role only a finished version."""
    if entry.get('role') == 'cutter' and entry.get('kind') != 'signnet':
        return False, 'compare_only'
    if not entry.get('finished'):
        return False, 'not_finished'
    if entry.get('role') == 'cutter' and not entry.get('fits_the_car'):
        return False, 'not_for_the_car'
    return True, None


def _active_entry(role: str, value: Any) -> dict | None:
    """One role of active.json, checked: None when null or not a choice this module could have written (an old
    Detector id is still one: it is read back until a SignNet is chosen)."""
    if not isinstance(value, dict) or not isinstance(value.get('id'), str):
        return None
    if role_of(value['id']) != role:
        return None
    who = value.get('who')
    via = value.get('via')
    at = value.get('at')
    return {'id': value['id'], 'who': who if isinstance(who, str) and len(who) <= 100 else None,
            'via': via if via in VIA else None, 'at': at if isinstance(at, str) and len(at) <= 40 else None}


def _write_atomic(path: Path, value: dict) -> None:
    """The JSON written next to path, flushed to the disk, then renamed over it: a reader sees the old file or the
    new one, never half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False)
    try:
        with handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise
    with contextlib.suppress(OSError):
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _car_id_ok(role: str, model_id: Any) -> bool:
    """An id of a network on the car: a SignNet or a detector for the cutter, a reader or classifier."""
    return role_of(model_id) == role


def _on_car_record(data: Any) -> dict:
    """on-car.json checked: the id of each network on the car (None when not one this module could have written),
    the repository, commit and version of the published models, and when it was recorded."""
    if not isinstance(data, dict):
        raise ValueError('not a JSON object')
    text = lambda key, limit: data.get(key) if isinstance(data.get(key), str) and len(data[key]) <= limit else None  # noqa: E731
    ids = {}
    for role in CAR_NETWORKS.values():
        value = data.get(role)
        model_id = value.get('id') if isinstance(value, dict) else None
        ids[role] = model_id if _car_id_ok(role, model_id) else None
    commit = text('commit', 40)
    return {'ids': ids, 'repository': text('repository', 100), 'commit': commit if commit and COMMIT.fullmatch(commit) else None,
            'version': text('version', 40), 'recorded_at': text('recorded_at', 40)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _models_json(folder: Path, name: Any) -> tuple[Path, Any]:
    """A file of the published models clone, by a name of its speed_limits.json: inside the clone, never a link
    out of it; and its JSON when it is one."""
    if not isinstance(name, str) or not name or '/' in name or name.startswith('.'):
        raise ValueError(f'speed_limits.json names {name!r}, not a file of the clone')
    path = RunsIndex._inside(folder / name, folder)
    if path is None:
        raise ValueError(f'{name} is not a file of {folder}')
    if not name.endswith('.json'):
        return path, None
    try:
        return path, json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'{name} cannot be read: {exc}') from exc


def _car_model_id(role: str, checkpoint: Any) -> str:
    """The id of the Models tab for the checkpoint a contract names: an id as is, else a path of the weights of a
    run (<run>/signnet-best.pt, <run>/<part>/<weights>)."""
    if _car_id_ok(role, checkpoint):
        return checkpoint
    if isinstance(checkpoint, str) and len(checkpoint) <= 1000:
        found = SIGNNET_WEIGHTS.search('/' + checkpoint) if role == 'cutter' else None
        if found is not None:
            return f'cutter:{found.group(1)}/{SIGNNET_PART}'
        found = CHAIN_WEIGHTS[role].search('/' + checkpoint) if role in CHAIN_WEIGHTS else None
        if found is not None:
            return f'{role}:{found.group(1)}/{found.group(2)}'
    raise ValueError(f'the {role} contract names {checkpoint!r}, not weights of a run')


def _git(folder: Path, *args: str) -> str:
    done = subprocess.run(['git', '-C', str(folder), *args], capture_output=True, text=True, timeout=30, check=False)
    if done.returncode != 0:
        raise ValueError(f'git {" ".join(args)} in {folder}: {done.stderr.strip() or "failed"}')
    return done.stdout.strip()


def record_on_car(root: Path, models: Path, *, dry_run: bool = False) -> dict:
    """``nnslr on-car``: <root>/models/on-car.json from a clone of the published models repository, the networks
    the comma downloads. Each network of its speed_limits.json (format 2: finder, classifier and brain when
    there is one) must match its contract (the ONNX sha256), name weights of a run on this server, and those weights must be
    the ones the contract was made from (checkpoint sha256); the clone must be clean and its commit on a remote
    branch. Reads only the clone (no network) and the runs; writes the one file, unless dry_run."""
    root, models = Path(root), Path(models)
    if not models.is_dir():
        raise ValueError(f'{models} is not a folder')
    _, manifest = _models_json(models, 'speed_limits.json')
    if not isinstance(manifest, dict) or manifest.get('format') != 2:
        raise ValueError('speed_limits.json: format 2 expected (a finder and a classifier)')
    runs = RunsIndex(root)
    artifacts = ModelIndex(runs).artifacts(tuple(CAR_NETWORKS.values()))
    record: dict[str, Any] = {'schema_version': 1}
    for network, role in CAR_NETWORKS.items():
        item = manifest.get(network)
        if item is None and network in OPTIONAL_CAR_NETWORKS:
            continue
        if not isinstance(item, dict):
            raise ValueError(f'speed_limits.json has no {network}')
        onnx, _ = _models_json(models, item.get('file'))
        _, contract = _models_json(models, item.get('contract'))
        if not isinstance(contract, dict):
            raise ValueError(f'{item.get("contract")} is not a JSON object')
        onnx_sha256 = _sha256(onnx)
        if contract.get('onnx_sha256') != onnx_sha256:
            raise ValueError(f'{item["file"]} is not the ONNX of its contract (sha256)')
        model_id = _car_model_id(role, contract.get('checkpoint'))
        if not any(entry['id'] == model_id for entry in artifacts[role]):
            raise ValueError(f'the {network} {model_id} is not on this server')
        run_name, part = model_id.split(':', 1)[1].split('/', 1)
        run = runs._run_dir(run_name)
        relative = {SIGNNET_PART: SIGNNET_PART + '-best.pt', BRAIN_PART: BRAIN_WEIGHTS}.get(part) or f'{part}/{WEIGHTS[role]}'
        weights = None if run is None else runs._file(run, relative)
        wanted = contract.get('checkpoint_sha256')
        if weights is None or not isinstance(wanted, str) or not SHA256.fullmatch(wanted) or _sha256(weights) != wanted:
            raise ValueError(f'the weights of {model_id} on this server are not the ones of the published {network}')
        record[role] = {'id': model_id, 'file': item['file'], 'onnx_sha256': onnx_sha256, 'checkpoint_sha256': wanted}
    if _git(models, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError(f'{models} has changes not committed: the record names a commit')
    commit = _git(models, 'rev-parse', 'HEAD')
    if not COMMIT.fullmatch(commit):
        raise ValueError(f'{models}: no commit')
    if not _git(models, 'branch', '--remotes', '--contains', commit):
        raise ValueError(f'{models}: commit {commit[:7]} is on no remote branch (not pushed, or fetch first)')
    url = _git(models, 'remote', 'get-url', 'origin')
    found = re.search(r'github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?/?$', url)
    version = manifest.get('version')
    record.update({'repository': found.group(1) if found else None, 'commit': commit,
                   'version': version if isinstance(version, str) and len(version) <= 40 else None,
                   'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds')})
    if not dry_run:
        path = root / ON_CAR_FILE
        if not path.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError('models/ of the data root leads outside it')
        _write_atomic(path, record)
    return record


# -- the index --------------------------------------------------------------------------------------------
class ModelIndex:
    """Every trained artifact of <root>/runs by role, and the one in use for each role (models/active.json)."""

    def __init__(self, runs: RunsIndex):
        self.runs = runs
        self.root = runs.root
        self._write_lock = threading.Lock()

    # -- the artifacts ---------------------------------------------------------------------------------
    def _base(self, role: str, run: str, part: str, checkpoint: Path, result: Any, problems: list[str]) -> dict:
        return {'id': f'{role}:{run}/{part}', 'role': role, 'run': run, 'part': part,
                'trained_at': _iso(_mtime(checkpoint)), '_mtime': _mtime(checkpoint) or 0.0,
                'finished': isinstance(result, dict), 'in_use': False, 'on_car': False, 'problems': problems}

    def _read(self, run: Path, relative: str, digest, problems: list[str], name: str | None = None) -> Any:
        """The digest of a file of the run (None when missing); an unreadable file is a problem of the version."""
        value = self.runs._load(self.runs._file(run, relative), digest)
        if value is UNREADABLE:
            problems.append((name or relative) + ' is unreadable')
            return None
        return value

    def _cutter(self, name: str, run: Path, folder: str, evaluation: str, part: str, size: str | None) -> dict | None:
        checkpoint = self.runs._file(run, folder + '/detector-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, folder + '/result.json', model_result_digest, problems)
        test = self._read(run, evaluation + '/detector-test.json', cutter_digest, problems)
        validation = self._read(run, evaluation + '/detector-validation.json', cutter_digest, problems)
        trained = result or {}
        measured = test or validation or {}
        # The input size and default boxes it ran with; before they were recorded every detector had the defaults.
        input_size = measured.get('input_size') or trained.get('input_size') or size
        anchor = measured.get('anchor_min_ratio')
        if anchor is None:
            anchor = trained.get('anchor_min_ratio')
        default_size = '{}x{}'.format(*DEFAULT_DETECTOR_INPUT_SIZE)
        families = trained.get('families') or measured.get('families')
        entry = self._base('cutter', name, part, checkpoint, result, problems)
        entry.update({
            'kind': 'detector',
            'label': 'Detector ' + _pretty_size(input_size or default_size),
            # The cutter before SignNet: kept to compare, never chosen again.
            'detail': 'old, to compare · ' + ('input-size comparison' if size else 'training pipeline'),
            'architecture': trained.get('architecture'), 'initialization': trained.get('initialization'),
            'best_epoch': trained.get('best_epoch') or measured.get('checkpoint_epoch'),
            'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs_run'),
            'input_size': input_size or default_size, 'input_size_recorded': input_size is not None,
            'anchor_min_ratio': anchor if anchor is not None else DEFAULT_DETECTOR_ANCHOR_MIN_RATIO,
            'anchor_recorded': anchor is not None,
            'augmentation': trained.get('augmentation'),
            # What it finds: every sign or the speed signs only (before it was recorded, the speed signs only).
            'families': families or 'speed', 'families_recorded': families is not None,
            'test': test, 'validation': validation,
        })
        return entry

    def _signnet(self, name: str, run: Path) -> dict | None:
        """A SignNet of ``nnslr train-signnet``: THE cutter, the small finder the comma runs, judged on the cutter's
        test frames with the cutter's scoring. Finished and of the car's band and grid (fits_the_car, read from
        result.json: the review server has no torch to open the weights), it can be the cutter in use; another band
        or stride is research, listed to compare."""
        checkpoint = self.runs._file(run, SIGNNET_PART + '-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, 'result.json', signnet_digest, problems)
        subset = self._read(run, 'subset.json', subset_digest, problems)
        trained = result or {}
        share = (subset or {}).get('share')
        entry = self._base('cutter', name, SIGNNET_PART, checkpoint, result, problems)
        band = trained.get('band')
        fits = bool(trained.get('fits_the_car'))
        entry.update({
            'kind': 'signnet',
            'label': 'SignNet' + (' ' + _pretty_size(band) if band else ''),
            'detail': ('the comma’s cutter' if fits else 'not finished' if result is None
                       else 'not the comma’s band or grid, to compare')
                      + (f' · {round(share * 100)} % of the training drives' if share else ''),
            'fits_the_car': fits,
            'architecture': None, 'initialization': 'start' if trained.get('start') else None,
            # The split it learnt on, for the Train tab's "Start from", and the SignNet it went on from, if one.
            'split_sha256': trained.get('split_sha256'), 'start': trained.get('start'),
            'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'),
            'epochs_run': trained.get('epochs_run'),
            'input_size': band, 'input_size_recorded': band is not None,
            'anchor_min_ratio': None, 'anchor_recorded': False, 'augmentation': None,
            'stride': trained.get('stride'), 'parameters': trained.get('parameters'), 'train_share': share,
            'families': trained.get('families') or 'all', 'families_recorded': trained.get('families') is not None,
            'test': trained.get('test'), 'validation': trained.get('validation'),
        })
        return entry

    def _classifier(self, name: str, run: Path) -> dict | None:
        checkpoint = self.runs._file(run, 'reader/reader-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, 'reader/result.json', model_result_digest, problems)
        report = self._read(run, 'evaluation/reader-test.json', reader_digest, problems)
        trained = result or {}
        entry = self._base('classifier', name, 'reader', checkpoint, result, problems)
        entry.update({
            'label': 'Reader', 'detail': 'training pipeline',
            'architecture': trained.get('architecture'), 'initialization': trained.get('initialization'),
            'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs_run'),
            'classes': trained.get('classes') or (report or {}).get('classes'),
            'best_validation_accuracy': trained.get('best_validation_accuracy'),
            'test': None if report is None else {
                'samples': report['samples'], 'accuracy': report['accuracy'],
                'human_samples': (report['human'] or {}).get('samples'),
                'human_accuracy': (report['human'] or {}).get('accuracy')},
        })
        return entry

    def _plan_split(self, run: Path) -> str | None:
        """The split of classifier/training-plan.json (None when missing or unreadable)."""
        plan = self.runs._load(self.runs._file(run, 'classifier/training-plan.json'), plan_digest)
        return plan.get('split_sha256') if isinstance(plan, dict) else None

    def _classifier_v2(self, name: str, run: Path) -> dict | None:
        """The classifier with "not a sign" of scripts/run_classifier.sh (its weights still say task "reader")."""
        checkpoint = self.runs._file(run, 'classifier/reader-best.pt')
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, 'classifier/result.json', classifier_result_digest, problems)
        report = self._read(run, 'evaluation/classifier-test.json', reader_digest, problems)
        chain = self._read(run, 'evaluation/chain-test.json', chain_digest, problems)
        old_reader = self._read(run, 'evaluation/chain-test-old-reader.json', chain_digest, problems)
        trained = result or {}
        entry = self._base('classifier', name, 'classifier', checkpoint, result, problems)
        entry.update({
            'label': 'Classifier', 'detail': 'with “not a sign”',
            'class_scheme': trained.get('class_scheme') or (report or {}).get('class_scheme') or 'v2',
            'architecture': trained.get('architecture') or None, 'initialization': trained.get('initialization'),
            'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs_run'),
            'classes': trained.get('classes') or (report or {}).get('classes'),
            'best_validation_accuracy': trained.get('best_validation_accuracy'),
            # The mean of the two below chose its best epoch.
            'best_validation_balanced_accuracy': trained.get('best_validation_balanced_accuracy'),
            'best_validation_sign_accuracy': trained.get('best_validation_sign_accuracy'),
            'best_validation_rejection': trained.get('best_validation_rejection'),
            'augmentation': trained.get('augmentation'),
            # The split it learnt on, for the Train tab's "Start from" (before result.json recorded it, the one of its
            # training plan, read once and cached as for the Runs tab); and the classifier it started from, if one.
            'split_sha256': trained.get('split_sha256') or self._plan_split(run),
            'start': trained.get('start'),
            'test': None if report is None else classifier_headline(report),
            # The chain end to end at the cutter's score, and the old reader in the same chain.
            'chain': None if chain is None else chain_headline(chain),
            'chain_old_reader': None if old_reader is None else chain_headline(old_reader),
        })
        return entry

    def _deciders(self, name: str, run: Path) -> list[dict]:
        base = run / EXPERIMENT_DIR
        names = list(VARIANTS)
        try:
            if base.is_dir() and not base.is_symlink():
                names += sorted(entry.name for entry in base.iterdir()
                                if RUN_NAME.fullmatch(entry.name) and entry.name not in VARIANTS
                                and entry.is_dir() and not entry.is_symlink())
        except OSError:
            pass
        found = []
        for variant in names:
            checkpoint = self.runs._file(run, f'{EXPERIMENT_DIR}/{variant}/{variant}-best.pt')
            if checkpoint is None:
                continue
            problems: list[str] = []
            result = self._read(run, f'{EXPERIMENT_DIR}/{variant}/result.json', decider_digest, problems)
            trained = result or {}
            entry = self._base('decider', name, f'{EXPERIMENT_DIR}/{variant}', checkpoint, result, problems)
            entry.update({
                'label': VARIANT_LABELS.get(variant, _words(variant)), 'detail': 'experiment B prototype',
                'variant': variant, 'inputs': trained.get('inputs'),
                'architecture': trained.get('architecture'), 'initialization': trained.get('initialization'),
                'best_epoch': trained.get('best_epoch'), 'epochs': trained.get('epochs'),
                'epochs_run': trained.get('epochs_run'),
                # A smoke run of experiment B (--limit): a few frames per partition, not a measure.
                'smoke_limit': trained.get('limit'),
                'split_sha256': trained.get('split_sha256'), 'start': trained.get('start'),
                'validation': (trained.get('partitions') or {}).get('validation'),
                'test': (trained.get('partitions') or {}).get('test'),
            })
            found.append(entry)
        return found

    def _brain(self, name: str, run: Path) -> dict | None:
        """The brain of a run of ``nnslr train-brain`` (its dry run is not a version)."""
        checkpoint = self.runs._file(run, BRAIN_WEIGHTS)
        if checkpoint is None:
            return None
        problems: list[str] = []
        result = self._read(run, 'result.json', brain_digest, problems)
        trained = result or {}
        if trained.get('dry_run'):
            return None
        entry = self._base('brain', name, BRAIN_PART, checkpoint, result, problems)
        seed = trained.get('seed')
        entry.update({'label': 'Brain' if seed is None else f'Brain, seed {seed}', 'detail': 'says the speed to hold',
                      'groups': trained.get('groups'),
                      'architecture': 'mobilenet_v3_small+gru', 'initialization': 'ImageNet', 'best_epoch': None,
                      'seed': seed, 'works': trained.get('works'), 'plates': bool(trained.get('plates')),
                      'hours': bool(trained.get('hours')), 'wet': bool(trained.get('wet')),
                      'epochs': trained.get('epochs'), 'epochs_run': trained.get('epochs'),
                      'validation': trained.get('validation'), 'test': trained.get('test')})
        return entry

    def artifacts(self, roles: tuple[str, ...] = ROLES) -> dict[str, list[dict]]:
        """Every trained artifact by role (of the roles asked), newest first (by the time of its weights)."""
        found: dict[str, list[dict]] = {role: [] for role in ROLES}
        for name in self.runs.names():
            run = self.runs._run_dir(name)
            if run is None:
                continue
            if 'classifier' in roles:
                found['classifier'] += [entry for entry in (self._classifier(name, run), self._classifier_v2(name, run))
                                        if entry is not None]
            if 'decider' in roles:
                found['decider'] += self._deciders(name, run)
            if 'brain' in roles:
                found['brain'] += [entry for entry in (self._brain(name, run),) if entry is not None]
            if 'cutter' not in roles:
                continue
            cutters = [self._cutter(name, run, 'detector', 'evaluation', 'detector', None)]
            sizes = run / 'detector-sizes'
            try:
                folders = sorted(entry.name for entry in sizes.iterdir()
                                 if SIZE_NAME.fullmatch(entry.name) and entry.is_dir() and not entry.is_symlink()) \
                    if sizes.is_dir() and not sizes.is_symlink() else []
            except OSError:
                folders = []
            for size in folders:
                folder = 'detector-sizes/' + size
                cutters.append(self._cutter(name, run, folder, folder + '/evaluation', folder, size))
            cutters.append(self._signnet(name, run))
            found['cutter'] += [entry for entry in cutters if entry is not None]
        for versions in found.values():
            versions.sort(key=lambda entry: (entry['_mtime'], entry['id']), reverse=True)
            for entry in versions:
                del entry['_mtime']
                ok, why = selectable(entry)
                entry['selectable'] = ok
                if not ok:
                    entry['not_selectable_reason'] = NOT_SELECTABLE[why][0]
        return found

    # -- the model in use ------------------------------------------------------------------------------
    def _active_path(self, name: str = 'active.json') -> Path | None:
        """<root>/models/active.json (or another file of models/), None when models/ leads out of the data root."""
        models = self.root / 'models'
        try:
            if not models.resolve().is_relative_to(self.root.resolve()):
                return None
        except (OSError, RuntimeError):
            return None
        return models / name

    def read_on_car(self) -> tuple[dict | None, str | None]:
        """What runs on the comma (models/on-car.json of record_on_car): None when nothing was recorded; and why the
        file cannot be used, if it cannot."""
        path = self._active_path(Path(ON_CAR_FILE).name)
        if path is None:
            return None, 'models/ of the data root leads outside it'
        if not path.exists() and not path.is_symlink():
            return None, None
        real = self.runs._inside(path, path.parent)
        if real is None:
            return None, ON_CAR_FILE + ' is not a file inside models/'
        try:
            return _on_car_record(json.loads(real.read_text(encoding='utf-8'))), None
        except (OSError, ValueError):
            return None, ON_CAR_FILE + ' cannot be read'

    def read_active(self) -> tuple[dict[str, dict | None], str | None]:
        """The model in use per role (None: nobody picked one), and why the file cannot be used, if it cannot."""
        empty: dict[str, dict | None] = {role: None for role in ROLES}
        path = self._active_path()
        if path is None:
            return empty, 'models/ of the data root leads outside it'
        if not path.exists() and not path.is_symlink():
            return empty, None
        real = self.runs._inside(path, path.parent)
        if real is None:
            return empty, ACTIVE_FILE + ' is not a file inside models/'
        try:
            data = json.loads(real.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return empty, ACTIVE_FILE + ' cannot be read'
        if not isinstance(data, dict):
            return empty, ACTIVE_FILE + ' is not a JSON object'
        return {role: _active_entry(role, data.get(role)) for role in ROLES}, None

    def in_use(self, role: str) -> tuple[dict | None, str]:
        """The version in use for a role, its files still on this server; else None and why (for the panel: the
        classifier's training mines the false boxes of the cutter in use)."""
        active, problem = self.read_active()
        if problem is not None:
            return None, problem
        chosen = active.get(role)
        if chosen is None:
            return None, 'no ' + role + ' in use: choose one on the Models tab'
        version = next((entry for entry in self.artifacts((role,))[role] if entry['id'] == chosen['id']), None)
        if version is None:
            return None, 'the ' + role + ' in use is no longer on this server: choose another one on the Models tab'
        return version, ''

    def index(self) -> dict:
        """GET /api/panel/models: the roles (the three of the sign pipeline and the brain), each with its versions
        newest first and the one in use."""
        artifacts = self.artifacts()
        active, problem = self.read_active()
        on_car, car_problem = self.read_on_car()
        car_ids = (on_car or {}).get('ids') or {}
        roles = []
        for role in ROLES:
            chosen = active[role]
            versions = artifacts[role]
            for entry in versions:
                entry['in_use'] = chosen is not None and entry['id'] == chosen['id']
                entry['on_car'] = car_ids.get(role) is not None and entry['id'] == car_ids[role]
            # An old Detector id still in active.json: it keeps working until a SignNet is chosen; the site says so.
            old_kind = role == 'cutter' and chosen is not None and not SIGNNET_ID.fullmatch(chosen['id'])
            roles.append({'role': role, 'label': ROLE_LABELS[role], 'versions': versions,
                          'active': None if chosen is None else {
                              **chosen, 'found': any(entry['in_use'] for entry in versions)},
                          'old_kind_in_use': old_kind, 'notice': OLD_KIND_NOTICE if old_kind else None})
        self.runs._prune()
        return {'roles': roles, 'active_problem': problem, 'active_file': ACTIVE_FILE,
                'runs_dir_exists': self.runs.runs_dir.is_dir(), 'cutter_score_threshold': CUTTER_SCORE,
                'box_budget': BOX_BUDGET, 'on_car_file': ON_CAR_FILE, 'on_car_problem': car_problem,
                'on_car': None if on_car is None else {
                    key: on_car[key] for key in ('repository', 'commit', 'version', 'recorded_at')}}

    def set_active(self, role: Any, model_id: Any, *, who: str | None, via: str) -> dict:
        """POST /api/panel/models/active: the model in use for a role. The id must be an artifact of the index that
        can be chosen (selectable: finished; for the cutter a SignNet that fits the car, never an old Detector); the
        file keeps the other roles as they are."""
        if not isinstance(role, str) or role not in ROLES:
            raise ModelChoiceError(400, 'bad_role', 'Unknown role: cutter, classifier, decider or brain.')
        if not isinstance(model_id, str) or len(model_id) > MAX_ID or MODEL_ID.fullmatch(model_id) is None:
            raise ModelChoiceError(400, 'bad_id', 'Not a model id.')
        if role_of(model_id) != role:
            raise ModelChoiceError(400, 'wrong_role', 'That model is not a ' + role + '.')
        if via not in VIA:
            raise ValueError('via')
        version = next((entry for entry in self.artifacts()[role] if entry['id'] == model_id), None)
        if version is None:
            raise ModelChoiceError(404, 'unknown_model', 'No such model on this server.')
        ok, why = selectable(version)
        if not ok:
            raise ModelChoiceError(409, why, NOT_SELECTABLE[why][1])
        path = self._active_path()
        if path is None:
            raise ModelChoiceError(409, 'bad_models_folder', 'models/ of the data root leads outside it.')
        with self._write_lock:
            current, problem = self.read_active()
            if problem is not None:
                # Never overwrite a file that cannot be read: it may hold choices someone wants to keep.
                raise ModelChoiceError(409, 'active_unreadable', problem + ': fix or remove it first.')
            entry = {'id': version['id'], 'who': who[:100] if isinstance(who, str) else None, 'via': via,
                     'at': datetime.now().astimezone().isoformat(timespec='seconds')}
            _write_atomic(path, {'schema_version': 1, **current, role: entry})
        return entry
# [model-roles] - END
