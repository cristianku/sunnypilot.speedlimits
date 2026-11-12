# [control-panel] - START
"""The training runs of this server, for the Evaluate tab of the site: what each run under
<data root>/runs/<run>/ is (how it started, how far it got) and how its networks did on the test partition.

Everything is read from what the training writes (scripts/run_v100_baseline.sh, nnslr_tools.training,
nnslr_tools.evaluation) and from the log of helpers/3_train.sh (<data root>/logs/train-<run>.log):

- ``detector/training-plan.json``, ``reader/training-plan.json``: the dataset and split of the run (the split
  file itself is the one the plan names, under <data root>/splits/);
- ``detector/result.json``, ``reader/result.json``: a network trained to the end (history, initialization);
- ``evaluation/detector-test.json``, ``evaluation/reader-test.json``: the measures on the test partition,
  with every test image and crop;
- ``export/detector.onnx``, ``export/reader.onnx``, ``bundle/sha256sums.json``: the last steps;
- ``experiment-b/<variant>/result.json``: experiment B (nnslr_tools.context_experiment), read defensively.

A run of scripts/run_classifier.sh (the classifier with "not a sign", trained on the false boxes of the cutter in
use; kind ``classifier``) has its own files, read the same way:

- ``classifier.log``: the whole output of the script (its steps and the curves of the classifier);
- ``negatives.jsonl``: the false boxes mined with the cutter; only its first line, a header with the counts, is read;
- ``classifier/training-plan.json``, ``classifier/result.json``, ``classifier/reader-best.pt``: its training;
- ``evaluation/classifier-{test,validation}.json``: the classifier alone on the crops (the reader's report, plus
  the sign crops apart and the "not a sign" crops it rejects);
- ``evaluation/chain-{test,validation}.json``, ``evaluation/chain-test-old-reader.json``: the chain end to end
  (cutter -> classifier), with the new classifier and with the old reader, at a few scores of the cutter;
- ``comparison.txt``: the new classifier against the old reader, in plain text.

A run of scripts/run_signnet.sh (the panel's "Train the cutter": SignNet, the cutter the comma runs; kind ``cutter``)
has at its root ``signnet-best.pt``, ``result.json`` (task signnet: how it was trained, its history, its measures on
validation and test) and ``signnet.log`` (the whole output of the script, its epochs); a SignNet trained by hand with
nnslr train-signnet has ``train.log`` instead of signnet.log.

A run of scripts/run_usernet.sh (the panel's "Train UserNet": the road users' detector, nnslr_tools.usernet; kind
``usernet``) has at its root ``usernet-best.pt``, ``result.json`` (task usernet: its stage, its history, its measures
on validation and test at its score) and ``usernet.log`` (the whole output of the script, its epochs).

A run of scripts/run_detector_sizes.sh alone (the old cutter, a detector of one input size, kept to compare; kind
``cutter`` too) has ``detector-sizes/<size>/`` (training-plan.json, detector-best.pt, result.json,
evaluation/detector-{test,validation}.json), ``detector-sizes/<size>.log`` (the epochs) and
``detector-sizes/comparison.txt``; the same folders in a run of the pipeline are shown with it.

The Watch panel of the Evaluate tab plays the frames of one route of an evaluation (the detector of the pipeline, a
cutter of the sizes, the chain and the chain with the old reader) with the reviewed boxes and the cutter's boxes over
each frame (RunsIndex.frames, frames_digest): from the boxes the report keeps frame by frame (``per_frame_boxes`` of
nnslr_tools.evaluation and nnslr_tools.chain), else from what an older report has (the detector's boxes at its own
threshold, the chain's verdicts at its main score), and the previews of the review pages.

Every file is reduced once to a small digest, kept while its size and mtime stay the same: the evaluation
reports and the training plans (megabytes, a row per sample) are never parsed again at every request.
Run and variant names follow a strict pattern, and a file is read only when its real path (after the
symlinks) is inside the run's own folder. Nothing here writes anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from nnslr_tools.job_progress import progress
from nnslr_tools.sign_types import HOURS_CLASS, WET_CLASS, class_hours, class_parts
from nnslr_tools.usernet import LOG as USERNET_LOG
from nnslr_tools.usernet import MIN_USER_SCORE
from nnslr_tools.usernet import WEIGHTS as USERNET_WEIGHTS
from nnslr_tools.usernet import digest as usernet_result

# The name of a run (helpers/3_train.sh: letters, digits, . _ -), never "." or "..".
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}')
# helpers/3_train.sh names a run by the time it starts: 2026-10-24_1452; the panel's classifier training adds what it
# is after the time: 2026-10-24_1730-classifier (-2, -3... when that folder exists).
TIME_NAME = re.compile(r'(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(?:-[A-Za-z0-9._-]+)?')
LOG_TAIL_BYTES = 4 * 1024 * 1024
HUMAN = 'human_review'

# The steps of a run in order (scripts/run_v100_baseline.sh) and the words for each, while it runs.
STAGE_LABELS = {'checks': 'checking the dataset and the GPU', 'detector': 'training the detector',
                'reader': 'training the reader', 'evaluate': 'evaluating on the test', 'export': 'exporting to ONNX',
                'bundle': 'packaging the bundle', 'done': 'finished',
                # A folder of scripts/run_experiment_b.sh alone.
                'experiment': 'training the variants of experiment B',
                # A run of scripts/run_classifier.sh (its checks are the dataset, the split and the cutter).
                'mine': 'mining the cutter’s false boxes', 'classifier': 'training the classifier',
                'chain': 'evaluating the chain end to end',
                # A SignNet run (scripts/run_signnet.sh).
                'signnet': 'training SignNet',
                # A UserNet run (scripts/run_usernet.sh).
                'usernet': 'training UserNet'}
# The same steps in a classifier run and in a cutter run, where they mean something else.
CLASSIFIER_STAGE_LABELS = {'checks': 'checking the dataset, the split and the cutter',
                           'evaluate': 'evaluating the classifier alone'}
CUTTER_STAGE_LABELS = {'detector': 'training the cutter', 'evaluate': 'evaluating on test and validation',
                       'signnet': 'training SignNet'}
USERNET_STAGE_LABELS = {'checks': 'checking the pictures and the GPU', 'evaluate': 'measuring on validation and test'}
CLASSIFIER_STAGES = ('checks', 'mine', 'classifier', 'evaluate', 'chain', 'done')

# The scripts that write into a run folder, named by the run in their first argument.
SCRIPTS = ((b'run_v100_baseline.sh', 'pipeline'), (b'run_experiment_b.sh', 'experiment'),
           (b'run_classifier.sh', 'classifier'), (b'run_signnet.sh', 'cutter'), (b'run_detector_sizes.sh', 'cutter'),
           (b'run_usernet.sh', 'usernet'))
# Folders and files of the training pipeline (a folder with none of them holds only experiment B).
PIPELINE_PARTS = ('detector', 'reader', 'evaluation', 'export', 'bundle')
# What scripts/run_classifier.sh writes first; a folder with one of them and none of the pipeline's own networks
# (evaluation/ is in both) is a classifier run.
CLASSIFIER_PARTS = ('classifier', 'negatives.jsonl', 'classifier.log')
PIPELINE_ONLY = ('detector', 'reader', 'export', 'bundle')
# scripts/run_detector_sizes.sh: a folder per input size (WIDTHxHEIGHT) and its log next to it.
SIZES_DIR = 'detector-sizes'
SIZE_FOLDER = re.compile(r'[1-9][0-9]{0,4}x[1-9][0-9]{0,4}')
# scripts/run_signnet.sh: SignNet's weights, result and log at the root of its run; train.log for one trained by hand.
SIGNNET_FILES = {'weights': 'signnet-best.pt', 'result': 'result.json', 'log': 'signnet.log', 'hand_log': 'train.log'}
# scripts/run_usernet.sh: UserNet's weights, result and log at the root of its run (nnslr_tools.usernet).
USERNET_FILES = {'weights': USERNET_WEIGHTS, 'result': 'result.json', 'log': USERNET_LOG}
# SignNet's best 5 boxes of a frame at any score (signnet.CAR_MAX_BOXES; what its training keeps the best epoch by).
# What the car reads is narrower: those 5 of score 0.3 or more (signnet.car_measure, each partition's 'car').
SIGNNET_BUDGET = '5'
# The score of the cutter the chain is judged at when its report names none (the old Detector's,
# nnslr_tools.model_roles.CUTTER_SCORE); a report made with a SignNet names its own (the car's 0.3).
CHAIN_SCORE = 0.1
# The weights a chain report names, <data root>/runs/<run>/<part>/<file> (a SignNet: <run>/signnet-best.pt): the id of
# the Models tab, <role>:<run>/<part>.
CHAIN_WEIGHTS = {
    'cutter': re.compile(rf'/runs/({RUN_NAME.pattern})/((?:detector|{SIZES_DIR}/{SIZE_FOLDER.pattern})(?=/detector-best\.pt$)'
                         r'|signnet(?=-best\.pt$))(?:/detector)?-best\.pt$'),
    'classifier': re.compile(rf'/runs/({RUN_NAME.pattern})/(reader|classifier)/reader-best\.pt$'),
}
NOT_A_SIGN = 'not_a_sign'
# Where a "not a sign" crop comes from (nnslr_tools.training.NEGATIVE_SOURCES): a proposal a reviewer removed, a false
# box of the cutter mined for the training, a box a reviewer kept and marked "not a sign".
SOURCES = ('rejected', 'mined', 'reviewed')
SOURCE_LABELS = {'rejected': 'Removed proposals', 'mined': 'Cutter’s false boxes', 'reviewed': 'Reviewed “not a sign”'}
# Rows of a label table of the chain, and labels of its lists, kept in a digest.
MAX_LABEL_ROWS = 40
MAX_LABELS = 15
# The plain text comparison of a classifier run, as much as the page shows.
MAX_COMPARISON = 12000
# The header of negatives.jsonl is one line; never more than this is read.
MAX_HEADER_BYTES = 4 * 1024 * 1024
EXPERIMENT_EPOCH = re.compile(r'^experiment_b_(\w+) epoch (\d+)/(\d+)\b', re.M)

# Sign sizes for the detector's recall: the width of the reviewed box in pixels of the frame.
SIZE_BUCKETS = (('lt12', '< 12 px', 0.0, 12.0), ('12-20', '12–20 px', 12.0, 20.0), ('20-40', '20–40 px', 20.0, 40.0),
                ('ge40', '≥ 40 px', 40.0, math.inf))

# Experiment B: the variants in the order of the plan (docs/NEURAL_NETWORK.md, step 2).
EXPERIMENT_DIR = 'experiment-b'
VARIANTS = ('crop', 'crop_geom', 'full')
VARIANT_LABELS = {'crop': 'Crop', 'crop_geom': 'Crop + geometry', 'full': 'Crop + geometry + scene'}
# Parts of an experiment report that are not measures (lists of samples, settings).
SKIPPED_KEYS = {'samples', 'images', 'predictions', 'records', 'history', 'config', 'frames', 'rows', 'per_sample',
                'errors', 'warnings', 'limitations', 'classes', 'labels', 'schema_version', 'seed', 'batch_size',
                'learning_rate', 'weight_decay', 'num_workers', 'input_size'}
HUMAN_KEYS = {'human_reviewed', 'human'}
MAX_METRICS = 60
MAX_MATRIX = 8


class WatchError(Exception):
    """A request of the Watch panel that is refused: HTTP status, reason, message (as panel.PanelError)."""

    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


class _Unreadable:
    """A file that exists but could not be digested (a broken or half-written JSON)."""


UNREADABLE = _Unreadable()


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(value)


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec='seconds')


def _mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def label_name(label: str) -> str:
    """'maximum_speed:60' -> 'Speed limit 60', 'maximum_speed_hours:30' -> 'Speed limit 30, with hours',
    'maximum_speed_hours:30:22-6' -> 'Speed limit 30, hours 22-6', 'maximum_speed_wet:80' -> 'Speed limit 80, when wet',
    'variable_display:unknown' -> 'Variable speed limit, value unknown', 'other_sign:give_way' -> 'Other sign: give way',
    'not_a_sign' -> 'Not a sign'."""
    from nnslr_tools.panel import FAMILY_LABELS
    family, value, panel = class_parts(label)
    name = FAMILY_LABELS.get(family, family.replace('_', ' ').capitalize())
    if panel == 'wet':
        return f'{name} {value}, when wet'
    if panel is not None:
        return f'{name} {value}, hours {class_hours(label)}' if class_hours(label) else f'{name} {value}, with hours'
    if value == 'unknown':
        return name + ', value unknown'
    if family == 'other_sign' and value:
        # The kind of another sign (nnslr_tools.sign_types.SIGN_TYPES), learnt by the classifier with "not a sign".
        return name + ': ' + value.replace('_', ' ')
    return name + (' ' + value if value else '')


# -- the reader's report ---------------------------------------------------------------------------------
def _tables_from_samples(samples: list) -> tuple[dict, dict]:
    per_class: dict[str, dict] = {}
    confusion: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get('truth'), str):
            continue
        truth = sample['truth']
        predicted = str(sample.get('predicted'))
        row = per_class.setdefault(truth, {'support': 0, 'correct': 0})
        row['support'] += 1
        correct = sample.get('correct')
        if correct is None:
            correct = predicted == truth
        row['correct'] += bool(correct)
        confusion[truth][predicted] += 1
    return per_class, {truth: dict(counts) for truth, counts in confusion.items()}


def _label_table(per_class: Any, confusion: Any) -> list[dict]:
    """Per label, most frequent first: count, correct, accuracy and the wrong answer it gets most."""
    if not isinstance(per_class, dict):
        return []
    confusion = confusion if isinstance(confusion, dict) else {}
    rows = []
    for label, values in per_class.items():
        if not isinstance(values, dict):
            continue
        count = _int(values.get('support')) or 0
        correct = _int(values.get('correct')) or 0
        wrong = None
        answers = confusion.get(label)
        if isinstance(answers, dict):
            mistakes = [(n, str(answer)) for answer, n in answers.items() if answer != label and _int(n)]
            if mistakes:
                top = sorted(mistakes, key=lambda pair: (-pair[0], pair[1]))[0]
                wrong = {'label': top[1], 'name': label_name(top[1]), 'count': int(top[0])}
        rows.append({'label': str(label), 'name': label_name(label), 'count': count, 'correct': correct,
                     'accuracy': _ratio(correct, count), 'wrong': wrong})
    return sorted(rows, key=lambda row: (-row['count'], row['label']))


def _human_share(value: Any) -> dict | None:
    """human_reviewed of a reader report: {'samples', 'accuracy', 'correct'}, None without it."""
    if not isinstance(value, dict):
        return None
    count = _int(value.get('sample_count')) or 0
    accuracy = _num(value.get('accuracy'))
    return {'samples': count, 'accuracy': accuracy, 'correct': round(accuracy * count) if accuracy is not None else None}


# The labels of speed signs (the family before ":"): the classifier's number on them alone is the one that matters.
SPEED_LABEL_FAMILIES = ('maximum_speed', HOURS_CLASS, WET_CLASS, 'variable_display', 'zone', 'cancellation', 'road_marking')


def _speed_share(labels: list[dict]) -> dict:
    """Of the label table: the sign crops of a speed sign and how many it reads right."""
    rows = [row for row in labels if row['label'].split(':', 1)[0] in SPEED_LABEL_FAMILIES]
    count, correct = sum(row['count'] for row in rows), sum(row['correct'] for row in rows)
    return {'samples': count, 'correct': correct, 'accuracy': _ratio(correct, count)}


def _signs_digest(signs: Any, samples: list | None) -> dict | None:
    """The sign crops of a classifier evaluation (``signs``: every crop that is not "not a sign"): how many it reads
    right, also with the kinds of other signs folded (comparable with the old reader), how many it throws away as
    "not a sign", and label by label, as the reader's table."""
    if not isinstance(signs, dict):
        return None
    if samples is not None:
        own = [item for item in samples if isinstance(item, dict) and item.get('truth') != NOT_A_SIGN]
        per_class, confusion = _tables_from_samples(own)
        human_samples = [item for item in own if item.get('label_basis') == HUMAN]
        human_tables = _tables_from_samples(human_samples) if human_samples else None
    else:
        per_class, confusion = signs.get('per_class'), signs.get('confusion')
        human_report = signs.get('human_reviewed')
        human_tables = ((human_report.get('per_class'), human_report.get('confusion'))
                        if isinstance(human_report, dict) else None)
    labels = _label_table(per_class, confusion)
    count = _int(signs.get('sample_count'))
    if count is None:
        count = sum(row['count'] for row in labels)
    correct = sum(row['correct'] for row in labels)
    accuracy = _num(signs.get('accuracy'))
    rejected = _int(signs.get('rejected'))
    rejected_rate = _num(signs.get('rejected_rate'))
    return {
        'samples': count, 'correct': correct, 'accuracy': accuracy if accuracy is not None else _ratio(correct, count),
        'family_accuracy': _num(signs.get('family_accuracy')),
        'speed': _speed_share(labels),
        'rejected': rejected,
        'rejected_rate': rejected_rate if rejected_rate is not None or rejected is None else _ratio(rejected, count),
        'human': _human_share(signs.get('human_reviewed')),
        'labels': labels,
        'labels_human': _label_table(*human_tables) if human_tables else None,
        'unseen_labels': [str(item) for item in signs.get('unseen_truth_labels') or [] if isinstance(item, str)],
    }


def _rejection(value: Any) -> dict | None:
    """{'crops', 'rejected', 'rejection_rate'} of "not a sign" crops, None when it is not such a count."""
    if not isinstance(value, dict):
        return None
    crops, rejected = _int(value.get('crops')) or 0, _int(value.get('rejected')) or 0
    rate = _num(value.get('rejection_rate'))
    return {'crops': crops, 'rejected': rejected, 'rejection_rate': rate if rate is not None else _ratio(rejected, crops)}


def _label_counts(value: Any, limit: int = MAX_LABELS) -> list[dict]:
    """{label: count} as rows, the most frequent first: [{'label', 'name', 'count'}]."""
    if not isinstance(value, dict):
        return []
    rows = [{'label': str(label), 'name': label_name(str(label)), 'count': _int(count) or 0}
            for label, count in value.items() if _int(count)]
    return sorted(rows, key=lambda row: (-row['count'], row['label']))[:limit]


def _not_a_sign_digest(value: Any) -> dict | None:
    """The "not a sign" crops of a classifier evaluation: how many it rejects, by where they come from (SOURCES), on
    the frames a person reviewed, and what it calls the ones it lets through."""
    total = _rejection(value)
    if total is None:
        return None
    by_source = value.get('by_source') if isinstance(value.get('by_source'), dict) else {}
    keys = [key for key in SOURCES if key in by_source] + sorted(str(key) for key in by_source if key not in SOURCES)
    sources = []
    for key in keys:
        counts = _rejection(by_source.get(key))
        if counts is not None:
            sources.append({'source': key, 'label': SOURCE_LABELS.get(key, _words(key)), **counts})
    return {**total, 'by_source': sources, 'human': _rejection(value.get('human_reviewed')),
            'survivors': _label_counts(value.get('survivors'))}


def reader_digest(report: Any) -> dict:
    """The reader's evaluation (nnslr_tools.evaluation.evaluate_reader) without its list of samples. A report of the
    classifier with "not a sign" (class_scheme v2: every crop counted, "not a sign" included) also has its sign
    crops apart (``signs``) and its "not a sign" crops (``not_a_sign``); an older report has None for both."""
    if not isinstance(report, dict):
        raise ValueError('not a report')
    samples = report.get('samples') if isinstance(report.get('samples'), list) else None
    if samples is not None:
        per_class, confusion = _tables_from_samples(samples)
        human_samples = [item for item in samples if isinstance(item, dict) and item.get('label_basis') == HUMAN]
        human_tables = _tables_from_samples(human_samples) if human_samples else None
    else:
        per_class, confusion = report.get('per_class'), report.get('confusion')
        human_report = report.get('human_reviewed')
        human_tables = ((human_report.get('per_class'), human_report.get('confusion'))
                        if isinstance(human_report, dict) else None)
    labels = _label_table(per_class, confusion)
    count = _int(report.get('sample_count'))
    if count is None:
        count = sum(row['count'] for row in labels)
    correct = sum(row['correct'] for row in labels)
    accuracy = _num(report.get('accuracy'))
    if accuracy is None:
        accuracy = _ratio(correct, count)
    human = None
    human_report = report.get('human_reviewed')
    if isinstance(human_report, dict):
        human_count = _int(human_report.get('sample_count')) or 0
        human_accuracy = _num(human_report.get('accuracy'))
        human = {'samples': human_count, 'accuracy': human_accuracy,
                 'correct': round(human_accuracy * human_count) if human_accuracy is not None else None}
    classes = report.get('classes')
    return {
        'partition': str(report.get('partition') or 'test'),
        'samples': count, 'correct': correct, 'accuracy': accuracy, 'human': human,
        'counts_by_label_basis': {str(key): _int(value) or 0 for key, value in (report.get('counts_by_label_basis') or {}).items()}
        if isinstance(report.get('counts_by_label_basis'), dict) else {},
        'labels': labels,
        'labels_human': _label_table(*human_tables) if human_tables else None,
        'unseen_labels': [str(item) for item in report.get('unseen_truth_labels') or [] if isinstance(item, str)],
        'classes': len(classes) if isinstance(classes, list) else None,
        'dataset_sha256': report.get('dataset_sha256') if isinstance(report.get('dataset_sha256'), str) else None,
        'split_sha256': report.get('split_sha256') if isinstance(report.get('split_sha256'), str) else None,
        'class_scheme': report.get('class_scheme') if isinstance(report.get('class_scheme'), str) else None,
        'signs': _signs_digest(report.get('signs'), samples),
        'not_a_sign': _not_a_sign_digest(report.get('not_a_sign')),
    }


def classifier_headline(digest: dict | None) -> dict:
    """The numbers that judge a classifier, from reader_digest: right answers on the sign crops (on every crop for
    an old reader, which has no "not a sign"), on the speed signs alone, on those a person confirmed, with the kinds
    of other signs folded;
    the "not a sign" crops it rejects and the sign crops it throws away. Every value None without a report."""
    if digest is None:
        digest = {'samples': None, 'accuracy': None, 'human': None, 'signs': None, 'not_a_sign': None, 'class_scheme': None}
    signs = digest.get('signs')
    rejection = digest.get('not_a_sign') or {}
    measured = signs if signs is not None else digest
    human = measured.get('human') or {}
    return {'samples': measured['samples'], 'accuracy': measured['accuracy'],
            'human_samples': human.get('samples'), 'human_accuracy': human.get('accuracy'),
            'family_accuracy': (signs or {}).get('family_accuracy'),
            'speed_samples': ((signs or {}).get('speed') or {}).get('samples'),
            'speed_correct': ((signs or {}).get('speed') or {}).get('correct'),
            'speed_accuracy': ((signs or {}).get('speed') or {}).get('accuracy'),
            'not_a_sign_crops': rejection.get('crops'), 'not_a_sign_rejected': rejection.get('rejected'),
            'rejection_rate': rejection.get('rejection_rate'),
            'signs_rejected': (signs or {}).get('rejected'), 'signs_rejected_rate': (signs or {}).get('rejected_rate'),
            'class_scheme': digest.get('class_scheme')}


# -- the chain: the cutter's boxes read by the classifier ---------------------------------------------------
def _chain_counts(value: Any, widths: bool) -> dict | None:
    """The reviewed boxes at one score of the cutter: found by the cutter, read right by the classifier (correct),
    thrown away as "not a sign" (rejected); recall end to end (found and read right), of the cutter alone
    (recall_found), and read right once found; by width and on the frames a person reviewed."""
    if not isinstance(value, dict):
        return None
    truth, found, correct = (_int(value.get(key)) or 0 for key in ('truth', 'found', 'correct'))
    pick = lambda key, part, whole: (_num(value.get(key)) if _num(value.get(key)) is not None  # noqa: E731
                                     else _ratio(part, whole))
    counts: dict[str, Any] = {'truth': truth, 'found': found, 'correct': correct, 'rejected': _int(value.get('rejected')),
                              'recall': pick('recall', correct, truth), 'recall_found': pick('recall_found', found, truth),
                              'read_right_when_found': pick('read_right_when_found', correct, found)}
    human = value.get('human_reviewed')
    counts['human'] = None
    if isinstance(human, dict):
        part = _chain_counts({key: item for key, item in human.items() if key != 'human_reviewed'}, False) or {}
        counts['human'] = {key: part.get(key) for key in ('truth', 'found', 'correct', 'recall', 'recall_found')}
    if widths:
        table = value.get('by_width')
        rows = None
        if isinstance(table, dict):
            rows = []
            for key, label, _, _ in SIZE_BUCKETS:
                row = _chain_counts(table.get(key) if isinstance(table.get(key), dict) else {}, False) or {}
                rows.append({'key': key, 'label': label,
                             **{name: row.get(name) for name in ('truth', 'found', 'correct', 'recall', 'recall_found')}})
        counts['widths'] = rows
    return counts


def _chain_labels(confusion: Any) -> list[dict]:
    """What the classifier answered on the boxes the cutter passed on, by the review of the box ("not a sign" for a
    false box): as the reader's label table, its correct answer being the review (rejecting a false box)."""
    if not isinstance(confusion, dict):
        return []
    per_class = {}
    for truth, answers in confusion.items():
        if isinstance(answers, dict):
            per_class[str(truth)] = {'support': sum(_int(count) or 0 for count in answers.values()),
                                     'correct': _int(answers.get(truth)) or 0}
    return _label_table(per_class, confusion)[:MAX_LABEL_ROWS]


def chain_digest(report: Any) -> dict:
    """An evaluation of the chain (cutter -> classifier, evaluation/chain-<partition>.json) without its frames: per
    score of the cutter, the speed signs and every reviewed box found and read right, the boxes and false boxes per
    frame the cutter passes on and the false ones the classifier lets through; who the two networks were."""
    if not isinstance(report, dict) or report.get('task', 'chain') != 'chain':
        raise ValueError('not a chain report')
    text = lambda value, key: value.get(key) if isinstance(value, dict) and isinstance(value.get(key), str) else None  # noqa: E731
    flag = lambda value, key: value.get(key) if isinstance(value, dict) and isinstance(value.get(key), bool) else None  # noqa: E731
    main = _num(report.get('main_score_threshold'))
    by_score = report.get('by_score_threshold') if isinstance(report.get('by_score_threshold'), dict) else {}
    thresholds = []
    for key, metrics in by_score.items():
        try:
            score = float(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict) or not math.isfinite(score):
            continue
        human = metrics.get('human')
        thresholds.append({
            'score': score,
            'speed_signs': _chain_counts(metrics.get('speed_signs'), True),
            'all_boxes': _chain_counts(metrics.get('all_boxes'), False),
            **{name: _num(metrics.get(name)) for name in ('boxes_per_frame', 'false_per_frame', 'surviving_false_per_frame',
                                                          'false_rejected_rate')},
            'false_boxes': _int(metrics.get('false_boxes')), 'surviving_false_boxes': _int(metrics.get('surviving_false_boxes')),
            'human': None if not isinstance(human, dict) else {
                'images': _int(human.get('images')), 'false_per_frame': _num(human.get('false_per_frame')),
                'surviving_false_per_frame': _num(human.get('surviving_false_per_frame'))},
            'surviving_false_by_label': _label_counts(metrics.get('surviving_false_by_label')),
            # The answers label by label only at the main score: the table is the longest part of the report.
            'labels': None,
            '_confusion': metrics.get('confusion'),
        })
    thresholds.sort(key=lambda item: item['score'])
    headline = _chain_entry(thresholds, main)
    for item in thresholds:
        confusion = item.pop('_confusion')
        if item is headline:
            item['labels'] = _chain_labels(confusion)
    cutter = report.get('cutter') if isinstance(report.get('cutter'), dict) else {}
    classifier = report.get('classifier') if isinstance(report.get('classifier'), dict) else {}
    limitations = report.get('limitations') if isinstance(report.get('limitations'), list) else []
    size = cutter.get('input_size')
    return {
        'partition': str(report.get('partition') or 'test'),
        'cutter': {'checkpoint': text(cutter, 'checkpoint'), 'id': text(cutter, 'id'), 'same_split': flag(cutter, 'same_split'),
                   'input_size': f'{size[0]}x{size[1]}' if isinstance(size, list) and len(size) == 2
                   and all(_int(side) for side in size) else None},
        'classifier': {'checkpoint': text(classifier, 'checkpoint'), 'class_scheme': text(classifier, 'class_scheme'),
                       'classes': _int(classifier.get('classes')), 'rejects': flag(classifier, 'rejects'),
                       'same_split': flag(classifier, 'same_split')},
        'iou_threshold': _num(report.get('iou_threshold')),
        **{key: _int(report.get(key)) for key in ('image_count', 'human_image_count', 'speed_sign_count', 'reviewed_box_count')},
        'main_score_threshold': main,
        'thresholds': thresholds,
        'device': str(report.get('device')) if isinstance(report.get('device'), str) else None,
        'seconds': _num(report.get('seconds')),
        'limitations': [str(item)[:300] for item in limitations if isinstance(item, str)][:10],
        'dataset_sha256': text(report, 'dataset_sha256'), 'split_sha256': text(report, 'split_sha256'),
    }


def _chain_entry(thresholds: list[dict], main: float | None) -> dict | None:
    """The score the chain is judged at: the report's main score (a SignNet's 0.3, the car's floor; a Detector's 0.1)
    when measured, else CHAIN_SCORE, else the first."""
    for wanted in (main, CHAIN_SCORE):
        found = next((item for item in thresholds if wanted is not None and abs(item['score'] - wanted) < 1e-9), None)
        if found is not None:
            return found
    return thresholds[0] if thresholds else None


def chain_headline(digest: dict) -> dict | None:
    """The numbers that judge the chain, at its report's main score (0.3 with a SignNet, 0.1 with a Detector; CHAIN_SCORE
    when it names none): the speed signs found and read right end to end, found by the cutter alone, on the frames a
    person reviewed; the boxes and false boxes per frame of the cutter, and the false ones that survive the
    classifier. None when the report has no score."""
    main = digest.get('main_score_threshold')
    entry = _chain_entry(digest['thresholds'], main)
    if entry is None:
        return None
    signs = entry['speed_signs'] or {}
    human = signs.get('human') or {}
    wanted = main if main is not None else CHAIN_SCORE
    return {'score_threshold': entry['score'], 'at_chain_score': abs(entry['score'] - wanted) < 1e-9,
            'truth': signs.get('truth'), 'found': signs.get('found'), 'correct': signs.get('correct'),
            'recall': signs.get('recall'), 'recall_found': signs.get('recall_found'),
            'read_right_when_found': signs.get('read_right_when_found'),
            'human_truth': human.get('truth'), 'human_recall': human.get('recall'),
            'boxes_per_frame': entry['boxes_per_frame'], 'false_per_frame': entry['false_per_frame'],
            'surviving_false_per_frame': entry['surviving_false_per_frame'],
            'rejects': digest['classifier']['rejects'], 'images': digest['image_count'],
            # The two networks measured: chains measured with different cutters are not comparable.
            'cutter_id': _chain_model_id('cutter', digest['cutter']), 'cutter_size': digest['cutter']['input_size'],
            'classifier_id': _chain_model_id('classifier', digest['classifier'])}


def _chain_model_id(role: str, model: dict) -> str | None:
    """The id of the cutter or classifier of a chain report (cutter:<run>/<part>): the one it recorded, else read from
    the path of its weights; None when neither says."""
    if model.get('id'):
        return model['id']
    match = CHAIN_WEIGHTS[role].search(model.get('checkpoint') or '')
    return f'{role}:{match.group(1)}/{match.group(2)}' if match else None


# -- the detector's report -------------------------------------------------------------------------------
def _box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_num(item) for item in value]
    if any(item is None for item in numbers):
        return None
    return numbers  # type: ignore[return-value]


def _iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return 0.0 if union <= 0 else inter / union


def _match(predictions: list[tuple[int, list[float], float]], truth: list[list[float]], threshold: float) -> tuple[set, set]:
    """The greedy score-ordered matching of nnslr_tools.evaluation.match_detections, keeping which prediction
    matched which reviewed box: (matched truth indexes, matched prediction indexes)."""
    unmatched = set(range(len(truth)))
    matched_truth, matched_predictions = set(), set()
    for index, box, _ in sorted(predictions, key=lambda item: item[2], reverse=True):
        best_iou, best = max(((_iou(box, truth[t]), t) for t in unmatched), default=(0.0, -1))
        if best >= 0 and best_iou >= threshold:
            unmatched.discard(best)
            matched_truth.add(best)
            matched_predictions.add(index)
    return matched_truth, matched_predictions


def _bucket(width: float) -> str:
    for key, _, low, high in SIZE_BUCKETS:
        if low <= width < high:
            return key
    return SIZE_BUCKETS[-1][0]


def _size_table(images: list[dict], threshold: float) -> list[dict] | None:
    """Recall by the width of the reviewed box, and the false boxes by their own width; None without boxes."""
    rows = {key: {'key': key, 'label': label, 'truth': 0, 'found': 0, 'false_boxes': 0} for key, label, _, _ in SIZE_BUCKETS}
    seen = False
    for image in images:
        truth_raw = image.get('truth_boxes')
        if not isinstance(truth_raw, list):
            continue
        seen = True
        truth = [(index, box) for index, box in ((index, _box(item)) for index, item in enumerate(truth_raw)) if box is not None]
        predictions = []
        for index, item in enumerate(image.get('predictions') or []):
            box = _box(item.get('box')) if isinstance(item, dict) else None
            score = _num(item.get('score')) if isinstance(item, dict) else None
            if box is not None and score is not None:
                predictions.append((index, box, score))
        local_truth, local_predictions = _match(predictions, [box for _, box in truth], threshold)
        recorded = image.get('matches')
        if isinstance(recorded, list) and all(isinstance(item, dict) for item in recorded):
            # The evaluation's own matching, by the index of the reviewed box.
            found = {item.get('truth_index') for item in recorded if isinstance(item.get('truth_index'), int)}
        else:
            found = {truth[position][0] for position in local_truth}
        for index, box in truth:
            row = rows[_bucket(box[2] - box[0])]
            row['truth'] += 1
            row['found'] += index in found
        for index, box, _ in predictions:
            if index not in local_predictions:
                rows[_bucket(box[2] - box[0])]['false_boxes'] += 1
    if not seen:
        return None
    return [{**row, 'missed': row['truth'] - row['found'], 'recall': _ratio(row['found'], row['truth'])} for row in rows.values()]


def _counts(found: int, missed: int, false_boxes: int, images: int | None) -> dict:
    precision = _ratio(found, found + false_boxes)
    recall = _ratio(found, found + missed)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'images': images, 'truth_boxes': found + missed, 'found': found, 'missed': missed, 'false_boxes': false_boxes,
            'recall': recall, 'precision': precision, 'f1': f1}


def detector_digest(report: Any) -> dict:
    """The detector's evaluation (nnslr_tools.evaluation.evaluate_detector) without its list of images."""
    if not isinstance(report, dict):
        raise ValueError('not a report')
    images = [item for item in report.get('images') or [] if isinstance(item, dict)] if isinstance(report.get('images'), list) else []
    threshold = _num(report.get('iou_threshold')) or 0.5
    found, missed, false_boxes = (_int(report.get(key)) for key in ('true_positives', 'false_negatives', 'false_positives'))
    if found is None or missed is None or false_boxes is None:
        found = sum(_int(item.get('true_positives')) or 0 for item in images)
        missed = sum(_int(item.get('false_negatives')) or 0 for item in images)
        false_boxes = sum(_int(item.get('false_positives')) or 0 for item in images)
    image_count = _int(report.get('image_count'))
    overall = _counts(found, missed, false_boxes, len(images) if image_count is None else image_count)
    human = None
    human_report = report.get('human_reviewed')
    if isinstance(human_report, dict):
        human = _counts(_int(human_report.get('true_positives')) or 0, _int(human_report.get('false_negatives')) or 0,
                        _int(human_report.get('false_positives')) or 0, _int(human_report.get('image_count')))
    routes: dict[str, dict] = {}
    for image in images:
        row = routes.setdefault(str(image.get('route_id') or '?'), {'images': 0, 'found': 0, 'missed': 0, 'false_boxes': 0})
        row['images'] += 1
        row['found'] += _int(image.get('true_positives')) or 0
        row['missed'] += _int(image.get('false_negatives')) or 0
        row['false_boxes'] += _int(image.get('false_positives')) or 0
    human_images = [image for image in images if image.get('label_basis') == HUMAN]
    return {
        'partition': str(report.get('partition') or 'test'),
        **overall,
        'false_per_image': _num(report.get('false_positives_per_image')),
        'score_threshold': _num(report.get('score_threshold')),
        'iou_threshold': threshold,
        'human': human,
        'counts_by_label_basis': {str(key): _int(value) or 0 for key, value in (report.get('counts_by_label_basis') or {}).items()}
        if isinstance(report.get('counts_by_label_basis'), dict) else {},
        'sizes': _size_table(images, threshold),
        'sizes_human': _size_table(human_images, threshold) if human_images else None,
        'routes': [{'route': name, **_counts(row['found'], row['missed'], row['false_boxes'], row['images'])}
                   for name, row in sorted(routes.items())],
        'dataset_sha256': report.get('dataset_sha256') if isinstance(report.get('dataset_sha256'), str) else None,
        'split_sha256': report.get('split_sha256') if isinstance(report.get('split_sha256'), str) else None,
    }


# -- the frames of an evaluation, for the Watch panel ------------------------------------------------------
# The reports the Watch panel of the Evaluate tab plays, frame after frame, with the boxes over the frame
# (GET /api/panel/runs/<run>/frames?model=&partition=&route=): the detector of the pipeline, a cutter of
# scripts/run_detector_sizes.sh, the chain (cutter -> classifier) and the same chain with the old reader.
WATCH_MODEL = re.compile(r'detector|detector-sizes/[1-9][0-9]{0,4}x[1-9][0-9]{0,4}|chain|chain-old-reader')
WATCH_PARTITIONS = ('test', 'validation')
WATCH_ROUTE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
# A frame of the dataset: derived/frames/<route>/<segment>/frame_<image>.png.
FRAME_FILE = re.compile(r'(?:^|/)([0-9]{1,9})/frame_([0-9]{1,9})\.(?:png|jpe?g)$')
# The preview of a frame in the current pre-annotation report of its route (served by /route/<route>/images/...).
PREVIEW_PATH = re.compile(r'images/[0-9]{8}\.jpg')
# What a report keeps of the cutter's boxes, frame by frame (nnslr_tools.evaluation.per_frame_boxes):
# 'all' every box from its floor (0.05), at most a few dozen per frame, with the counts at each step of the slider;
# 'above' (a detector report written before) its boxes at its own threshold only; 'verdicts' (a chain report written
# before) the reviewed boxes found and what the classifier called them, and the false boxes it let through, at its
# main score only; 'none' the reviewed boxes alone.
WATCH_LEVELS = ('all', 'above', 'verdicts', 'none')


def watch_file(model: str, partition: str) -> str | None:
    """The report of a model of the Watch panel on a partition, relative to its run; None when there is none."""
    if not isinstance(model, str) or not WATCH_MODEL.fullmatch(model) or partition not in WATCH_PARTITIONS:
        return None
    if model == 'detector':
        return f'evaluation/detector-{partition}.json'
    if model.startswith(SIZES_DIR + '/'):
        return f'{model}/evaluation/detector-{partition}.json'
    if model == 'chain':
        return f'evaluation/chain-{partition}.json'
    # The old reader in the same chain is measured on the test only.
    return 'evaluation/chain-test-old-reader.json' if partition == 'test' else None


def watch_label(model: str) -> str:
    if model == 'detector':
        return 'Detector of the pipeline'
    if model.startswith(SIZES_DIR + '/'):
        return 'Cutter ' + model[len(SIZES_DIR) + 1:].replace('x', ' × ')
    return 'Chain' if model == 'chain' else 'Chain, old reader'


def _pairs(boxes: list[list[float]], scores: list[float], truth: list[list[float]], threshold: float) -> dict[int, int]:
    """{box index: truth index}: the greedy score-ordered matching of nnslr_tools.evaluation.match_detections."""
    unmatched = set(range(len(truth)))
    pairs: dict[int, int] = {}
    for index in sorted(range(len(boxes)), key=lambda item: scores[item], reverse=True):
        best_iou, best = max(((_iou(boxes[index], truth[t]), t) for t in unmatched), default=(0.0, -1))
        if best >= 0 and best_iou >= threshold:
            unmatched.discard(best)
            pairs[index] = best
    return pairs


def _px(box: list[float]) -> list[int]:
    return [int(round(value)) for value in box]


def _score(value: float) -> float:
    """A score down to 3 decimals, never up: compared with a step of the slider it stays on the same side (as
    nnslr_tools.evaluation.per_frame_record writes it)."""
    return math.floor(round(value * 1000, 6)) / 1000


def _counts_list(value: Any, length: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    numbers = [_int(item) for item in value]
    return None if any(item is None or item < 0 for item in numbers) else numbers  # type: ignore[misc]


def _confidence(item: dict) -> float | None:
    value = _num(item.get('confidence'))
    return None if value is None else round(value, 3)


def _stored_box(text: Any, chain: bool) -> dict | None:
    """One box of per_frame_boxes: "x1 y1 x2 y2 score truth" (chain: "... label confidence"); truth -1 when the box
    found no reviewed box."""
    if not isinstance(text, str):
        return None
    parts = text.split(' ')
    if len(parts) != (8 if chain else 6):
        return None
    try:
        box = [float(value) for value in parts[:4]]
        score, match = float(parts[4]), int(parts[5])
        confidence = float(parts[7]) if chain else None
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in [*box, score, *([] if confidence is None else [confidence])]):
        return None
    # Written with 3 decimals already (rounded down there).
    item = {'box': _px(box), 'score': round(score, 3), 'match': match}
    if chain:
        item.update(label=parts[6], confidence=round(confidence, 3))
    return item


def _watch_frame(image: dict, level: str, chain: bool, iou: float) -> dict | None:
    """A frame of a report as the Watch panel draws it: its key and image, the reviewed boxes (family, label, and
    for an old chain report whether the cutter found it and what the classifier called it), the boxes of the cutter
    (score, the reviewed box each found or -1, and the classifier's answer in a chain), and how many boxes the cutter
    passed on at each step of the slider when the report counted them."""
    route = image.get('route_id')
    path = image.get('image_path')
    if not isinstance(route, str) or not WATCH_ROUTE.fullmatch(route) or not isinstance(path, str):
        return None
    named = FRAME_FILE.search(path)
    segment = _int(image.get('segment_index'))
    if segment is None and named:
        segment = int(named.group(1))
    index = int(named.group(2)) if named and (segment is None or int(named.group(1)) == segment) else None
    size = image.get('size')
    width = height = None
    if isinstance(size, list) and len(size) == 2 and all(_int(side) for side in size):
        width, height = int(size[0]), int(size[1])
    truth = []
    if chain:
        for item in image.get('truth') or []:
            box = _box(item.get('box')) if isinstance(item, dict) else None
            if box is None:
                continue
            label = item.get('label') if isinstance(item.get('label'), str) else None
            family = item.get('family') if isinstance(item.get('family'), str) else class_parts(label or '')[0] or None
            row = {'box': _px(box), 'family': family, 'label': label}
            if level == 'verdicts':
                row.update(found=bool(item.get('found')),
                           verdict=item.get('predicted') if isinstance(item.get('predicted'), str) else None,
                           confidence=_confidence(item))
            truth.append(row)
    else:
        families = image.get('truth_families') if isinstance(image.get('truth_families'), list) else []
        for position, item in enumerate(image.get('truth_boxes') or []):
            box = _box(item)
            if box is None:
                continue
            family = families[position] if position < len(families) and isinstance(families[position], str) else None
            truth.append({'box': _px(box), 'family': family, 'label': None})
    boxes: list[dict] = []
    counts = surviving = None
    if level == 'all':
        boxes = [item for item in (_stored_box(text, chain) for text in image.get('cutter_boxes') or []) if item is not None]
        counts = _counts_list(image.get('cutter_box_counts'), len(PER_FRAME_STEPS))
        surviving = _counts_list(image.get('surviving_counts'), len(PER_FRAME_STEPS)) if chain else None
    elif level == 'above':
        found = []
        for item in image.get('predictions') or []:
            box = _box(item.get('box')) if isinstance(item, dict) else None
            score = _num(item.get('score')) if isinstance(item, dict) else None
            if box is not None and score is not None:
                found.append((box, score))
        truth_boxes = [_box(item) for item in image.get('truth_boxes') or []]
        pairs = _pairs([box for box, _ in found], [score for _, score in found],
                       [box for box in truth_boxes if box is not None], iou)
        boxes = [{'box': _px(box), 'score': _score(score), 'match': pairs.get(position, -1)}
                 for position, (box, score) in enumerate(found)]
    elif level == 'verdicts':
        for item in image.get('surviving_false') or []:
            box = _box(item.get('box')) if isinstance(item, dict) else None
            score = _num(item.get('score')) if isinstance(item, dict) else None
            if box is None or score is None:
                continue
            boxes.append({'box': _px(box), 'score': _score(score), 'match': -1,
                          'label': item.get('predicted') if isinstance(item.get('predicted'), str) else None,
                          'confidence': _confidence(item)})
    # Matches of a box to a reviewed box that is not there (a report edited by hand) are dropped.
    for item in boxes:
        if not 0 <= item['match'] < len(truth):
            item['match'] = -1
    boxes.sort(key=lambda item: -item['score'])
    basis = image.get('label_basis')
    return {'route': route, 'key': None if segment is None or index is None else f'{segment}/{index}',
            'segment': segment, 'index': index, 'image_path': path,
            'image_sha256': image.get('image_sha256') if isinstance(image.get('image_sha256'), str) else None,
            'width': width, 'height': height, 'basis': basis if isinstance(basis, str) else None,
            'truth': truth, 'boxes': boxes, 'counts': counts, 'surviving': surviving}


# The steps of the score slider of the Watch panel, at which a report counts the boxes of each frame
# (nnslr_tools.evaluation.PER_FRAME_STEPS: the same numbers).
PER_FRAME_STEPS = tuple(step / 100 for step in range(5, 55, 5))


def frames_digest(report: Any) -> dict:
    """A detector or chain report as the Watch panel plays it: what it kept of the cutter's boxes (WATCH_LEVELS),
    from which score, its main score and IoU, and its frames by route in the order of the drive (segment, image)."""
    if not isinstance(report, dict) or report.get('task') not in ('detector', 'chain'):
        raise ValueError('not a detector or chain report')
    chain = report['task'] == 'chain'
    images = [item for item in report.get('images') or [] if isinstance(item, dict)] if isinstance(report.get('images'), list) else []
    stored = report.get('per_frame_boxes') if isinstance(report.get('per_frame_boxes'), dict) else None
    main = _num(report.get('main_score_threshold' if chain else 'score_threshold'))
    steps = stored.get('steps') if stored else None
    if stored and isinstance(steps, list) and [_num(step) for step in steps] == list(PER_FRAME_STEPS):
        level, floor = 'all', _num(stored.get('min_score'))
    elif not chain and images and all(isinstance(item.get('predictions'), list) for item in images):
        level, floor = 'above', main
    elif chain and images and all(isinstance(item.get('truth'), list) and isinstance(item.get('surviving_false'), list)
                                  for item in images):
        level, floor = 'verdicts', main
    else:
        level, floor = 'none', None
    iou = _num(report.get('iou_threshold')) or 0.5
    routes: dict[str, list[dict]] = {}
    for image in images:
        frame = _watch_frame(image, level, chain, iou)
        if frame is not None:
            routes.setdefault(frame.pop('route'), []).append(frame)
    for frames in routes.values():
        frames.sort(key=lambda item: (item['segment'] if item['segment'] is not None else -1,
                                      item['index'] if item['index'] is not None else -1, item['image_path']))
    classifier = report.get('classifier') if isinstance(report.get('classifier'), dict) else {}
    return {'kind': 'chain' if chain else 'cutter', 'level': level, 'min_score': floor, 'main_score': main,
            'max_boxes': _int((stored or {}).get('max_boxes')), 'iou_threshold': iou,
            'rejects': classifier.get('rejects') if chain and isinstance(classifier.get('rejects'), bool) else None,
            'partition': str(report.get('partition') or 'test'), 'image_count': len(images), 'routes': routes}


def default_score(digest: dict) -> float | None:
    """The score the Watch panel starts at: the chain's main score; for a cutter, its own score (0.1, as the Cutter
    section and the Models tab judge it) when the report kept boxes that low, else the report's own threshold."""
    if digest['level'] == 'none':
        return None
    if digest['kind'] == 'chain' or digest['level'] != 'all':
        return digest['main_score'] if digest['main_score'] is not None else digest['min_score']
    floor = digest['min_score'] or 0.0
    return CHAIN_SCORE if floor <= CHAIN_SCORE + 1e-9 else floor


def _frame_outcome(frame: dict, score: float | None, level: str) -> tuple[int, int | None]:
    """(reviewed boxes, those missed) of a frame at a score of the cutter; missed None when the report does not say."""
    truth = len(frame['truth'])
    if level == 'verdicts':
        return truth, sum(1 for item in frame['truth'] if not item.get('found'))
    if level == 'none' or score is None:
        return truth, None
    found = {item['match'] for item in frame['boxes'] if item['match'] >= 0 and item['score'] >= score}
    return truth, truth - len(found)


# -- training results, plans, the split, the log ----------------------------------------------------------
def result_digest(result: Any) -> dict:
    """result.json of a network (nnslr_tools.training): best value and epoch, epochs, initialization, history."""
    if not isinstance(result, dict):
        raise ValueError('not a result')
    history = []
    for item in result.get('history') or []:
        if not isinstance(item, dict) or _int(item.get('epoch')) is None:
            continue
        point: dict[str, float | int] = {'epoch': int(item['epoch'])}
        for source, target in (('train_loss', 'train_loss'), ('validation_loss', 'val_loss'),
                               ('train_accuracy', 'train_accuracy'), ('validation_accuracy', 'val_accuracy')):
            value = _num(item.get(source))
            if value is not None:
                point[target] = value
        history.append(point)
    config = result.get('config') if isinstance(result.get('config'), dict) else {}
    best_loss = _num(result.get('best_validation_loss'))
    best_accuracy = _num(result.get('best_validation_accuracy'))
    best_epoch = None
    if best_accuracy is not None:
        best_epoch = next((p['epoch'] for p in history if p.get('val_accuracy') == best_accuracy), None)
    elif best_loss is not None:
        best_epoch = next((p['epoch'] for p in history if p.get('val_loss') == best_loss), None)
    gpu = result.get('gpu') if isinstance(result.get('gpu'), dict) else None
    classes = result.get('classes')
    return {
        'architecture': str(result.get('architecture') or ''),
        'initialization': str(result.get('initialization') or '') or None,
        'epochs': _int(config.get('epochs')) or (history[-1]['epoch'] if history else None),
        'best_validation_loss': best_loss, 'best_validation_accuracy': best_accuracy, 'best_epoch': best_epoch,
        'classes': len(classes) if isinstance(classes, list) else None,
        'device': str(result.get('device') or '') or None,
        'gpu': str(gpu.get('name')) if gpu and gpu.get('name') else None,
        'history': history,
    }


# The values per epoch of the classifier with "not a sign" (result.json history -> the curves' names).
CLASSIFIER_HISTORY = (('validation_sign_accuracy', 'val_sign_accuracy'), ('validation_rejection', 'val_rejection'),
                      ('validation_balanced_accuracy', 'val_balanced_accuracy'), ('seconds', 'seconds'))


def classifier_result_digest(result: Any) -> dict:
    """result.json of the classifier with "not a sign" (nnslr_tools.training, task still "reader"): result_digest plus
    the best epoch it chose and its numbers on validation (the sign crops read right, the "not a sign" crops
    rejected, their mean that chose the epoch), the weight of "not a sign" in its loss, the augmentation and the
    negatives it learnt from. An older reader's result.json has None for all of them."""
    base = result_digest(result)
    by_epoch = {point['epoch']: point for point in base['history']}
    for item in result.get('history') or []:
        if not isinstance(item, dict) or _int(item.get('epoch')) not in by_epoch:
            continue
        for source, target in CLASSIFIER_HISTORY:
            value = _num(item.get(source))
            if value is not None:
                by_epoch[int(item['epoch'])][target] = int(value) if target == 'seconds' else value
    selection = result.get('selection') if isinstance(result.get('selection'), dict) else {}
    weights = result.get('loss_weights') if isinstance(result.get('loss_weights'), dict) else {}
    negatives = result.get('negatives') if isinstance(result.get('negatives'), dict) else None
    limitations = [item for item in result.get('limitations') or [] if isinstance(item, str)]
    return {
        **base,
        # The split it learnt on (None before it was recorded: its test report says it then), and the classifier it
        # started from in place of ImageNet (nnslr train --init-from): its run, whether it was the Mapillary stage of
        # a two-stage run, and whether it learnt on another split (its test may then hold crops it saw).
        'split_sha256': result.get('split_sha256') if isinstance(result.get('split_sha256'), str) else None,
        'start': _start_digest(result.get('start'),
                               any(item.startswith('start_of_another_split') for item in limitations)),
        'best_epoch': _int(result.get('best_epoch')) or base['best_epoch'],
        'epochs_run': len(base['history']) or None,
        'stopped_early': _int(result.get('stopped_early_after_epoch')),
        'class_scheme': result.get('class_scheme') if isinstance(result.get('class_scheme'), str) else None,
        'best_validation_sign_accuracy': _num(result.get('best_validation_sign_accuracy')),
        'best_validation_rejection': _num(result.get('best_validation_rejection')),
        'best_validation_balanced_accuracy': _num(result.get('best_validation_balanced_accuracy')),
        'selection': str(selection['by']) if isinstance(selection.get('by'), str) else None,
        'not_a_sign_weight': _num(weights.get('not_a_sign')),
        'augmentation': None if 'augmentation' not in result else bool(result['augmentation']),
        'negatives': None if negatives is None else {
            'file': Path(negatives['file']).name if isinstance(negatives.get('file'), str) else None,
            'sha256': negatives.get('sha256') if isinstance(negatives.get('sha256'), str) else None,
            'by_split': _nested_counts(negatives.get('by_split'))},
    }


def _start_digest(start: Any, another_split: bool) -> dict | None:
    """The start of a classifier (result.json "start", nnslr_tools.training.reader_start) without its path: the run
    and part of its .../runs/<run>/<part>/reader-best.pt (None for a file elsewhere), whether it was the pretraining
    on Mapillary alone, its split, and whether that split is not the one this classifier learnt on."""
    if not isinstance(start, dict):
        return None
    parts = Path(start['path']).parts if isinstance(start.get('path'), str) else ()
    inside = len(parts) >= 4 and parts[-4] == 'runs' and parts[-1] == 'reader-best.pt'
    return {'run': parts[-3] if inside else None, 'part': parts[-2] if inside else None,
            'mapillary_only': bool(start.get('mapillary_only')),
            'split_sha256': start.get('split_sha256') if isinstance(start.get('split_sha256'), str) else None,
            'another_split': another_split}


def _counts_map(value: Any) -> dict[str, int]:
    return {str(key): _int(count) or 0 for key, count in value.items()} if isinstance(value, dict) else {}


def _nested_counts(value: Any) -> dict[str, dict[str, int] | int]:
    """{split: {source: count}} (or {split: count}) of the negatives, numbers only."""
    if not isinstance(value, dict):
        return {}
    return {str(key): _counts_map(item) if isinstance(item, dict) else _int(item) or 0 for key, item in value.items()
            if isinstance(item, dict) or _int(item) is not None}


def _folded(value: Any) -> list[dict] | None:
    """folded_labels of a classifier plan: the labels learnt as a broader one (too few crops of their own), as rows
    {'label', 'name', 'into', 'into_name'}; a count when the plan says only how many."""
    if not isinstance(value, dict):
        return None
    rows = []
    for label, into in list(value.items())[:MAX_LABEL_ROWS]:
        if isinstance(into, str):
            rows.append({'label': str(label), 'name': label_name(str(label)), 'into': into, 'into_name': label_name(into)})
        elif _int(into) is not None:
            rows.append({'label': str(label), 'name': label_name(str(label)), 'into': None, 'into_name': None,
                         'count': _int(into)})
    return rows


def plan_digest(plan: Any) -> dict:
    """training-plan.json of a network: dataset and split identity and the counts, without its rows."""
    if not isinstance(plan, dict):
        raise ValueError('not a plan')
    text = lambda key: plan.get(key) if isinstance(plan.get(key), str) else None  # noqa: E731
    classes = plan.get('classes')
    return {'task': text('task'), 'dataset_kind': text('dataset_kind'), 'dataset_sha256': text('dataset_sha256'),
            'split_sha256': text('split_sha256'), 'split_path': text('split_path'),
            'counts_by_split': _counts_map(plan.get('counts_by_split')),
            'boxes_by_split': _counts_map(plan.get('boxes_by_split')),
            'negative_frames_by_split': _counts_map(plan.get('negative_frames_by_split')),
            'classes': len(classes) if isinstance(classes, list) else None,
            # The classifier with "not a sign": its "not a sign" crops per split by where they come from.
            'class_scheme': text('class_scheme'), 'negatives': _nested_counts(plan.get('negatives')),
            'folded_labels': _folded(plan.get('folded_labels'))}


NEGATIVE_COUNTS = ('frames', 'frames_with_candidates', 'candidates', 'after_frame_cap', 'positives', 'cap', 'selected')
NEGATIVE_SETTINGS = ('score_threshold', 'max_iou', 'per_frame', 'total_factor', 'seed')


def negatives_digest(header: Any) -> dict:
    """The first line of negatives.jsonl (the false boxes of the cutter mined for the classifier): the cutter that
    drew them, the settings of the mining and, per split, the frames, the candidate boxes and the ones kept, by
    tier of the review of their frame."""
    if not isinstance(header, dict) or header.get('kind') != 'header':
        raise ValueError('no header')
    text = lambda value, key: value.get(key) if isinstance(value, dict) and isinstance(value.get(key), str) else None  # noqa: E731
    cutter = header.get('cutter') if isinstance(header.get('cutter'), dict) else {}
    settings = header.get('settings') if isinstance(header.get('settings'), dict) else {}
    counts = header.get('counts') if isinstance(header.get('counts'), dict) else {}
    size = cutter.get('input_size')
    splits = []
    for key in sorted(counts, key=lambda name: (('train', 'validation', 'test').index(name)
                                                 if name in ('train', 'validation', 'test') else 3, str(name))):
        value = counts[key]
        if isinstance(value, dict):
            splits.append({'split': str(key), **{name: _int(value.get(name)) for name in NEGATIVE_COUNTS},
                           'by_tier': _counts_map(value.get('selected_by_tier'))})
    return {
        'created_at': text(header, 'created_at'), 'dataset_kind': text(header, 'dataset_kind'),
        'dataset_sha256': text(header, 'dataset_sha256'), 'split_sha256': text(header, 'split_sha256'),
        'cutter': {'checkpoint': text(cutter, 'checkpoint'), 'id': text(cutter, 'id'),
                   'input_size': f'{size[0]}x{size[1]}' if isinstance(size, list) and len(size) == 2
                   and all(_int(side) for side in size) else None,
                   'same_split': cutter.get('same_split') if isinstance(cutter.get('same_split'), bool) else None},
        'settings': {**{name: _num(settings.get(name)) for name in NEGATIVE_SETTINGS},
                     'rule_change_at': text(settings, 'rule_change_at')},
        'splits': splits,
    }


def split_digest(split: Any) -> dict:
    """A split file (data/splits/<sha>.json) without its assignments."""
    if not isinstance(split, dict):
        raise ValueError('not a split')
    routes = split.get('routes') if isinstance(split.get('routes'), dict) else {}
    return {
        'policy': split.get('policy') if isinstance(split.get('policy'), str) else None,
        'seed': _int(split.get('seed')),
        'grouping': [str(item) for item in split.get('grouping') or [] if isinstance(item, str)],
        'test_routes': [str(item) for item in split.get('test_routes') or [] if isinstance(item, str)],
        'routes': {str(key): [str(item) for item in value if isinstance(item, str)]
                   for key, value in routes.items() if isinstance(value, list)},
        'counts': _counts_map(split.get('counts')),
        'human_reviewed_counts': _counts_map(split.get('human_reviewed_counts')),
        'warnings': [str(item) for item in split.get('warnings') or [] if isinstance(item, str)],
    }


MODE_LINE = re.compile(r'^training mode: (\w+)', re.M)
WEIGHTS_LINE = re.compile(r'^continue (?:detector|reader) from weights: (.+?)\s*$', re.M)
ERROR_LINE = re.compile(r'^(?:ERROR: |error: |[A-Za-z_][\w.]*(?:Error|Exception): )')
FROM_RUN = re.compile(r'/runs/([A-Za-z0-9][A-Za-z0-9._-]{0,99})/(?:detector|reader)/')


def log_digest(text: str) -> dict:
    """The log of helpers/3_train.sh for a run: the training mode(s), where the weights came from, the curves,
    the step it reached, the last error line."""
    modes = MODE_LINE.findall(text)
    weights = WEIGHTS_LINE.findall(text)
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    running = progress('train', 'running', text)
    return {'modes': modes, 'weights': weights[0] if weights else None, 'error': error,
            'curves': running['curves'], 'detail': running['detail'], 'percent': running['percent'],
            'complete': 'Pipeline complete:' in text}


def classifier_log_digest(text: str) -> dict:
    """<run>/classifier.log (scripts/run_classifier.sh): the step it reached, the curves of the classifier, the last
    error line."""
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    running = progress('train_classifier', 'running', text)
    step = next((row['key'] for row in running['steps'] if row['state'] == 'current'), None)
    return {'step': step, 'error': error, 'curves': running['curves'], 'detail': running['detail'],
            'percent': running['percent']}


def cutter_log_digest(text: str) -> dict:
    """<run>/signnet.log (scripts/run_signnet.sh; train.log of a SignNet trained by hand) or detector-sizes/<size>.log
    (the training of one size of the old scripts/run_detector_sizes.sh): the curves of SignNet or of the detector,
    its last epoch in words, the step it reached, the last error line."""
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    running = progress('train_cutter', 'running', text)
    curves = running['curves']
    net = 'signnet' if 'signnet' in curves else 'detector'
    name = 'SignNet' if net == 'signnet' else 'detector'
    curve = curves.get(net)
    detail = ''
    if curve and curve['points']:
        last = max(point['epoch'] for point in curve['points'])
        detail = (name + ' · stopped early after epoch ' + str(curve['stopped_early']) if curve.get('stopped_early')
                  else name + ' · epoch ' + str(last) + ' of ' + str(curve['epochs']))
    step = next((row['key'] for row in running['steps'] if row['state'] == 'current'), None)
    return {'curves': curves, 'detail': detail, 'error': error, 'step': step,
            'complete': all(row['state'] == 'done' for row in running['steps'])}


def usernet_log_digest(text: str) -> dict:
    """<run>/usernet.log (scripts/run_usernet.sh): UserNet's curve, its last epoch in words ("UserNet · epoch 3 of 30"),
    the step it reached (checks, usernet, evaluate), whether it wrote "UserNet complete", the last error line."""
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    running = progress('train_users', 'running', text)
    step = next((row['key'] for row in running['steps'] if row['state'] == 'current'), None)
    return {'curves': running['curves'], 'detail': running['detail'], 'error': error, 'step': step,
            'complete': all(row['state'] == 'done' for row in running['steps'])}


def usernet_run_digest(result: Any) -> dict | None:
    """<run>/result.json of UserNet (nnslr train-usernet): the numbers Models shows (usernet.digest: its stage, uses_bdd,
    where it was validated, the road users found at its score on validation and test) and its epochs, as the points of
    its curve. None for a result.json of another network."""
    digest = usernet_result(result)
    if digest is None:
        return None
    history = result.get('history') if isinstance(result.get('history'), list) else []
    points = [{'epoch': item['epoch'], 'train_loss': _num(item.get('loss')),
               'val_recall': _num(item.get('validation_recall')),
               'val_false_per_frame': _num(item.get('validation_false_per_frame')),
               'seconds': _num(item.get('seconds'))}
              for item in history if isinstance(item, dict) and isinstance(item.get('epoch'), int)]
    return {**digest, 'dataset_sha256': result.get('dataset_sha256') if isinstance(result.get('dataset_sha256'), str)
            else None, 'history': points}


def signnet_run_digest(result: Any) -> dict:
    """<run>/result.json of SignNet (nnslr train-signnet): its shape (band, stride, fuse kernel, widths), its training,
    whether the car runs it as it is (signnet.fits_the_car), and on validation and test the signs found and the false
    boxes per frame among its 5 best boxes of a frame at any score (SIGNNET_BUDGET), with ``car``: the same as the car
    reads it, the best 5 of score 0.3 or more (signnet.car_measure, with the found and reviewed boxes, the speed
    signs' recall and the boxes passed on per frame for the run's cutter line); its epochs as a curve."""
    from nnslr_tools import signnet
    if not isinstance(result, dict) or result.get('task') != 'signnet':
        raise ValueError('not a SignNet result')
    band = result.get('band')
    band = [int(side) for side in band] if (isinstance(band, list) and len(band) == 2
                                            and all(_int(side) and side > 0 for side in band)) else None
    widths = result.get('widths')
    widths = [int(width) for width in widths] if isinstance(widths, list) and widths and all(_int(width) for width in widths) else None
    stride = _int(result.get('stride'))

    def car(partition: str) -> dict | None:
        found = signnet.car_measure(result.get(partition))
        if found is None:
            return None
        speed = _dig(found, ('recall_by_family', 'speed'))
        return {'recall': _num(found.get('recall_iou')), 'false_per_frame': _num(found.get('false_positives_per_image')),
                'found': _int(found.get('true_positives')), 'truth_boxes': _int(found.get('truth_box_count')),
                'speed_recall': _num(speed.get('recall')) if isinstance(speed, dict) else None,
                'boxes_per_frame': _num(found.get('predictions_per_image')),
                'score_threshold': signnet.CAR_MIN_SCORE, 'max_boxes': signnet.CAR_MAX_BOXES}

    def budget(partition: str) -> dict | None:
        found = _dig(result, (partition, 'by_box_budget', SIGNNET_BUDGET))
        at_car = car(partition)
        if not isinstance(found, dict) and at_car is None:
            return None
        found = found if isinstance(found, dict) else {}
        return {'recall': _num(found.get('recall_iou')), 'false_per_frame': _num(found.get('false_positives_per_image')),
                'car': at_car}
    points = []
    for item in result.get('history') if isinstance(result.get('history'), list) else []:
        if not isinstance(item, dict) or _int(item.get('epoch')) is None:
            continue
        point: dict = {'epoch': _int(item['epoch'])}
        for key, name in (('loss', 'train_loss'), ('validation_recall_at_5', 'val_recall'),
                          ('validation_false_positives_per_image_at_5', 'val_false_per_frame'), ('seconds', 'seconds')):
            if _num(item.get(key)) is not None:
                point[name] = _num(item[key])
        points.append(point)
    text = lambda key: result.get(key) if isinstance(result.get(key), str) else None  # noqa: E731
    return {
        'band': band, 'stride': stride, 'fuse_kernel': _int(result.get('fuse_kernel')), 'widths': widths,
        'parameters': _int(result.get('parameters')), 'epochs': _int(result.get('epochs')),
        'best_epoch': _int(result.get('best_epoch')), 'families': text('families'), 'split_sha256': text('split_sha256'),
        'fits_the_car': band is not None and stride is not None and signnet.fits_the_car(band, stride),
        'validation': budget('validation'), 'test': budget('test'),
        # Not part of the detail's signnet_run: the summary's dataset, the run it went on from, and the curve when the
        # log has none.
        'dataset_sha256': text('dataset_sha256'), 'start': start_digest(result.get('start')), 'history': points,
    }


def start_digest(start: Any) -> dict | None:
    """The earlier run a SignNet or an applicability variant went on from (its result.json "start", --init-from): that
    run's name and the file of its weights."""
    if not isinstance(start, dict) or not isinstance(start.get('run'), str) or not RUN_NAME.fullmatch(start['run']):
        return None
    return {'run': start['run'], 'source': start['path'] if isinstance(start.get('path'), str) else None}


# What the detail of a run says of its SignNet (signnet_run_digest without its history).
SIGNNET_RUN_KEYS = ('band', 'stride', 'fuse_kernel', 'widths', 'parameters', 'epochs', 'best_epoch', 'families',
                    'split_sha256', 'fits_the_car', 'validation', 'test')


def comparison_digest(text: str) -> str:
    """comparison.txt of a classifier run, as much as the page shows."""
    return text[:MAX_COMPARISON]


def experiment_log_digest(text: str) -> dict:
    """<run>/experiment-b/<variant>.log: the last epoch line and the last error line."""
    epochs = EXPERIMENT_EPOCH.findall(text)
    error = None
    for line in reversed(text.splitlines()[-400:]):
        line = line.strip()
        if ERROR_LINE.match(line):
            error = line[:300]
            break
    last = epochs[-1] if epochs else None
    return {'variant': last[0] if last else None, 'epoch': int(last[1]) if last else None,
            'epochs': int(last[2]) if last else None, 'error': error}


# -- experiment B -----------------------------------------------------------------------------------------
# result.json of nnslr_tools.context_experiment: metrics.<validation|test> = {loss, box_count, reading: {accuracy,
# expected_calibration_error, ...}, ownership: {accuracy, macro_f1, abstain_rate, decided_accuracy, other_road:
# {precision, recall, f1}, confusion, expected_calibration_error, ...}, human_reviewed: the same on the boxes a
# person confirmed, or None}. The rows of the comparison: (key, label, path in a partition, unit, better).
EXPERIMENT_ROWS = (
    ('reading_accuracy', 'Reading accuracy', ('reading', 'accuracy'), 'percent', 'higher'),
    ('other_road_precision', 'Another road: precision', ('ownership', 'other_road', 'precision'), 'percent', 'higher'),
    ('other_road_recall', 'Another road: recall', ('ownership', 'other_road', 'recall'), 'percent', 'higher'),
    ('other_road_f1', 'Another road: F1', ('ownership', 'other_road', 'f1'), 'percent', 'higher'),
    ('ownership_macro_f1', 'Whom it applies to: macro-F1', ('ownership', 'macro_f1'), 'percent', 'higher'),
    ('ownership_accuracy', 'Whom it applies to: accuracy', ('ownership', 'accuracy'), 'percent', 'higher'),
    ('decided_accuracy', 'Accuracy when it decides', ('ownership', 'decided_accuracy'), 'percent', 'higher'),
    ('abstain_rate', 'Abstains (unclear)', ('ownership', 'abstain_rate'), 'percent', None),
    ('reading_calibration', 'Reading calibration error', ('reading', 'expected_calibration_error'), 'percent', 'lower'),
    ('ownership_calibration', 'Ownership calibration error', ('ownership', 'expected_calibration_error'), 'percent', 'lower'),
    ('loss', 'Loss', ('loss',), 'number', 'lower'),
    ('box_count', 'Boxes', ('box_count',), 'count', None),
    ('ownership_boxes', 'Boxes with whom it applies to', ('ownership', 'sample_count'), 'count', None),
)
EXPERIMENT_PARTITIONS = ('test', 'validation')
OWNERSHIP_ORDER = ('own_road', 'other_road', 'unresolved')
OWNERSHIP_NAMES = {'own_road': 'Ours', 'other_road': 'Another', 'unresolved': 'Unclear'}
INPUT_NAMES = {'crop': 'crop', 'box_geometry': 'box position', 'car_speed': 'car speed', 'scene': 'scene'}


def _dig(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _matrix(value: Any) -> dict | None:
    """A confusion matrix as {'labels', 'matrix'} (rows: truth, columns: answer), from a dict of dicts, a dict
    with labels and matrix, or a list of rows."""
    labels: list[str]
    rows: list
    if isinstance(value, dict) and isinstance(value.get('matrix'), list):
        rows = value['matrix']
        labels = [str(item) for item in value.get('labels') or range(len(rows))]
    elif isinstance(value, dict) and value and all(isinstance(item, dict) for item in value.values()):
        names = {str(key) for key in value} | {str(key) for item in value.values() for key in item}
        labels = [name for name in OWNERSHIP_ORDER if name in names] + sorted(names - set(OWNERSHIP_ORDER))
        rows = [[(value.get(truth) or {}).get(answer, 0) for answer in labels] for truth in labels]
    elif isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        rows = value
        labels = [str(index) for index in range(len(rows))]
    else:
        return None
    if not labels or len(labels) > MAX_MATRIX or len(rows) != len(labels):
        return None
    cleaned = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(labels):
            return None
        cleaned.append([_num(item) or 0 for item in row])
    return {'labels': labels, 'names': [OWNERSHIP_NAMES.get(label, _words(label)) for label in labels], 'matrix': cleaned}


def _flatten(value: Any, path: tuple[str, ...], human: bool, metrics: dict, depth: int = 0) -> None:
    """Every number of an unknown report by its path; a human_reviewed object (anywhere) gives the 'human'
    value of the same path."""
    if depth > 5:
        return
    number = _num(value)
    if number is not None:
        if path:
            metrics.setdefault(path, {})['human' if human else 'all'] = number
        return
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        if not isinstance(key, str) or key in SKIPPED_KEYS or 'confusion' in key.lower():
            continue
        if key in HUMAN_KEYS:
            _flatten(item, path, True, metrics, depth + 1)
        else:
            _flatten(item, path + (key,), human, metrics, depth + 1)


def _words(segment: str) -> str:
    words = segment.replace('_', ' ').replace('-', ' ').split()
    fixed = {'f1': 'F1', 'ece': 'ECE', 'iou': 'IoU', 'auc': 'AUC'}
    return ' '.join(fixed.get(word.lower(), word) for word in words)


def _metric_label(path: tuple[str, ...]) -> str:
    text = ' › '.join(_words(segment) for segment in path)
    return text[:1].upper() + text[1:]


LOWER_IS_BETTER = ('error', 'loss', 'calibration', 'brier', 'false', 'missed', 'wrong')
HIGHER_IS_BETTER = ('accuracy', 'precision', 'recall', 'f1', 'auc', 'correct')


def _metric_kind(path: tuple[str, ...], values: list[float]) -> tuple[str | None, str]:
    """(better: higher | lower | None, unit: percent | count | number) of a measure found by its path."""
    text = '.'.join(path).lower()
    better = None
    if any(word in text for word in LOWER_IS_BETTER):
        better = 'lower'
    elif any(word in text for word in HIGHER_IS_BETTER):
        better = 'higher'
    if 'count' in text or (values and all(float(value).is_integer() and value > 1 for value in values)):
        return better, 'count'
    if better is not None and values and all(0 <= value <= 1 for value in values):
        return better, 'percent'
    return better, 'number'


def experiment_digest(result: Any) -> dict:
    """One variant's result.json: the rows of EXPERIMENT_ROWS by partition (all boxes and human-reviewed), the
    ownership confusion, the variant's inputs and chosen epoch. A report of another shape: every number of it
    under the partition 'result' (generic rows)."""
    if not isinstance(result, dict):
        raise ValueError('not a result')
    partitions: dict[str, dict] = {}
    matrices: dict[str, dict] = {}
    generic: dict[str, dict] = {}
    metrics = result.get('metrics')
    if isinstance(metrics, dict):
        for name, report in metrics.items():
            if not isinstance(name, str):
                continue
            values: dict[str, dict] = {}
            human = report.get('human_reviewed') if isinstance(report, dict) and isinstance(report.get('human_reviewed'), dict) else None
            for key, _, path, _, _ in EXPERIMENT_ROWS:
                every = _num(_dig(report, path))
                if every is None and isinstance(report, dict):
                    every = _num(report.get(key))
                mine = _num(_dig(human, path))
                if mine is None:
                    # A human_reviewed object next to the measure, if a report puts it there.
                    mine = _num(_dig(report, path[:-1] + ('human_reviewed', path[-1])))
                if every is not None or mine is not None:
                    values[key] = {'all': every, 'human': mine}
            partitions[name] = values
            matrices[name] = {'all': _matrix(_dig(report, ('ownership', 'confusion'))),
                              'human': _matrix(_dig(human, ('ownership', 'confusion')))}
    else:
        flat: dict[tuple[str, ...], dict] = {}
        _flatten(result, (), False, flat)
        values = {}
        for path, cell in list(flat.items())[:MAX_METRICS]:
            key = '.'.join(path)
            better, unit = _metric_kind(path, [value for value in cell.values() if value is not None])
            generic[key] = {'label': _metric_label(path), 'unit': unit, 'better': better}
            values[key] = {'all': cell.get('all'), 'human': cell.get('human')}
        partitions['result'] = values
    inputs = result.get('inputs')
    text = lambda key: result.get(key) if isinstance(result.get(key), str) else None  # noqa: E731
    return {
        'partitions': partitions, 'matrices': matrices, 'generic': generic,
        'inputs': [INPUT_NAMES.get(str(item), _words(str(item))) for item in inputs] if isinstance(inputs, list) else None,
        'best_epoch': _int(result.get('best_epoch')),
        'epochs': _int(_dig(result, ('config', 'epochs'))),
        'selection_score': _num(result.get('best_validation_selection_score')),
        'selection_metric': text('selection_metric'),
        'initialization': text('initialization'), 'start': start_digest(result.get('start')),
        'dataset_sha256': text('dataset_sha256'), 'split_sha256': text('split_sha256'),
        'limit': _int(result.get('limit')),
    }


def experiment_table(digests: dict[str, dict], variants: list[str]) -> dict:
    """The variants side by side: one row per measure, a cell per variant and partition."""
    present = {name for digest in digests.values() for name in digest['partitions']}
    partitions = [name for name in EXPERIMENT_PARTITIONS if name in present] + sorted(present - set(EXPERIMENT_PARTITIONS))
    definitions = [(key, label, unit, better) for key, label, _, unit, better in EXPERIMENT_ROWS]
    seen = {key for key, _, _, _ in definitions}
    for digest in digests.values():
        for key, meta in digest['generic'].items():
            if key not in seen:
                seen.add(key)
                definitions.append((key, meta['label'], meta['unit'], meta['better']))
    empty = {'all': None, 'human': None}
    rows = []
    for key, label, unit, better in definitions:
        values = {variant: {partition: (digests[variant]['partitions'].get(partition, {}).get(key, empty)
                                        if variant in digests else empty) for partition in partitions}
                  for variant in variants}
        if any(cell['all'] is not None or cell['human'] is not None for row in values.values() for cell in row.values()):
            rows.append({'key': key, 'label': label, 'unit': unit, 'better': better, 'values': values})
    confusions = []
    if any(digest['matrices'] for digest in digests.values()):
        confusions.append({'key': 'ownership', 'label': 'Whom it applies to',
                           'variants': {variant: {partition: (digests[variant]['matrices'].get(partition, empty)
                                                              if variant in digests else empty)
                                                  for partition in partitions} for variant in variants}})
    return {'partitions': partitions, 'metrics': rows, 'confusions': confusions}


# -- the runs ---------------------------------------------------------------------------------------------
class RunsIndex:
    """The runs of <root>/runs, summarised from their files; digests kept by (mtime, size) of each file."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        # By (file, digest): one file can be reduced by two digests (the Evaluate and the Models tab).
        self._cache: dict[tuple[str, Callable[[Any], Any]], tuple[tuple[int, int], Any]] = {}

    # -- files -----------------------------------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return self.root / 'runs'

    def _run_dir(self, name: str) -> Path | None:
        """The folder of a run: a strict name, a real directory directly inside <root>/runs."""
        if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
            return None
        path = self.runs_dir / name
        try:
            if path.is_symlink() or not path.is_dir():
                return None
            if path.resolve().parent != self.runs_dir.resolve() or not path.resolve().is_relative_to(self.root.resolve()):
                return None
        except OSError:
            return None
        return path

    @staticmethod
    def _inside(path: Path, base: Path) -> Path | None:
        """path when it is a regular file whose real path is inside base, else None."""
        try:
            real = path.resolve(strict=True)
            if not real.is_relative_to(base.resolve()) or not real.is_file():
                return None
        except (OSError, RuntimeError):
            return None
        return real

    def _load(self, path: Path | None, digest: Callable[[Any], Any], *, text: bool = False, head: bool = False) -> Any:
        """The digest of a file (None when missing, UNREADABLE when it cannot be parsed), parsed again only
        when its mtime or size changed. text: the end of a log; head: the JSON of the first line only (the header
        of a JSON Lines file, never the rest)."""
        if path is None:
            return None
        try:
            info = path.stat()
        except OSError:
            return None
        key = (info.st_mtime_ns, info.st_size)
        slot = (str(path), digest)
        with self._lock:
            hit = self._cache.get(slot)
            if hit is not None and hit[0] == key:
                return hit[1]
            try:
                if text:
                    with path.open('rb') as source:
                        source.seek(max(0, info.st_size - LOG_TAIL_BYTES))
                        value = digest(source.read().decode('utf-8', 'replace'))
                elif head:
                    with path.open('rb') as source:
                        line = source.readline(MAX_HEADER_BYTES)
                    if not line.endswith(b'\n') and len(line) >= MAX_HEADER_BYTES:
                        raise ValueError('header too long')
                    value = digest(json.loads(line.decode('utf-8')))
                else:
                    value = digest(_read_json(path))
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
                value = UNREADABLE
            self._cache[slot] = (key, value)
            return value

    def _file(self, run: Path, relative: str) -> Path | None:
        return self._inside(run / relative, run)

    def _log(self, name: str) -> Path | None:
        return self._inside(self.root / 'logs' / ('train-' + name + '.log'), self.root / 'logs')

    def _split_file(self, plan: dict | None) -> Path | None:
        """The split a plan names, only inside <root>/splits; else the content-addressed file of its sha."""
        splits = self.root / 'splits'
        if plan is None:
            return None
        if plan.get('split_path'):
            found = self._inside(Path(plan['split_path']), splits)
            if found is not None:
                return found
        sha = plan.get('split_sha256')
        if isinstance(sha, str) and re.fullmatch(r'[0-9a-f]{64}', sha):
            return self._inside(splits / (sha + '.json'), splits)
        return None

    # -- what is running -------------------------------------------------------------------------------
    def running_runs(self) -> dict[str, set[str]]:
        """Runs with work running on this data root, and which: 'pipeline' (scripts/run_v100_baseline.sh <run>),
        'experiment' (scripts/run_experiment_b.sh <run>) and/or 'classifier' (scripts/run_classifier.sh <run>)."""
        found: dict[str, set[str]] = {}
        proc = Path('/proc')
        if not proc.is_dir():
            return found
        root = self.root.resolve()
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                arguments = (entry / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            for index, argument in enumerate(arguments[:-1]):
                kind = next((kind for script, kind in SCRIPTS if argument.endswith(script)), None)
                if kind is None:
                    continue
                name = arguments[index + 1].decode('utf-8', 'replace')
                if RUN_NAME.fullmatch(name) and self._same_root(entry, argument.decode('utf-8', 'replace'), root):
                    found.setdefault(name, set()).add(kind)
        return found

    @staticmethod
    def _same_root(process: Path, script: str, root: Path) -> bool:
        """The pipeline writes to NNSLR_DATA_ROOT, else to data/ of its checkout."""
        try:
            environment = (process / 'environ').read_bytes().split(b'\0')
        except OSError:
            return True
        for item in environment:
            if item.startswith(b'NNSLR_DATA_ROOT='):
                return Path(item[len(b'NNSLR_DATA_ROOT='):].decode('utf-8', 'replace')).resolve() == root
        try:
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = Path(os.readlink(process / 'cwd')) / script_path
            return (script_path.resolve().parents[1] / 'data').resolve() == root
        except (OSError, IndexError):
            return True

    # -- one run ---------------------------------------------------------------------------------------
    def _gather(self, name: str, run: Path, running: dict[str, set[str]]) -> dict:
        files = {
            'detector_plan': self._file(run, 'detector/training-plan.json'),
            'reader_plan': self._file(run, 'reader/training-plan.json'),
            'detector_result': self._file(run, 'detector/result.json'),
            'reader_result': self._file(run, 'reader/result.json'),
            'detector_eval': self._file(run, 'evaluation/detector-test.json'),
            'reader_eval': self._file(run, 'evaluation/reader-test.json'),
            'detector_onnx': self._file(run, 'export/detector.onnx'),
            'reader_onnx': self._file(run, 'export/reader.onnx'),
            'bundle': self._file(run, 'bundle/sha256sums.json'),
        }
        digests = {
            'detector_plan': self._load(files['detector_plan'], plan_digest),
            'reader_plan': self._load(files['reader_plan'], plan_digest),
            'detector_result': self._load(files['detector_result'], result_digest),
            'reader_result': self._load(files['reader_result'], result_digest),
            'detector_eval': self._load(files['detector_eval'], detector_digest),
            'reader_eval': self._load(files['reader_eval'], reader_digest),
        }
        log_path = self._log(name)
        log = self._load(log_path, log_digest, text=True)
        log = None if log is UNREADABLE else log
        problems = [key for key, value in digests.items() if value is UNREADABLE]
        digests = {key: (None if value is UNREADABLE else value) for key, value in digests.items()}
        # What scripts/run_classifier.sh writes (in a folder of its own, or into a run of the pipeline).
        classifier_files = {key: self._file(run, relative) for key, relative in CLASSIFIER_FILES.items()}
        classifier = {
            'negatives': self._load(classifier_files['negatives'], negatives_digest, head=True),
            'classifier_plan': self._load(classifier_files['classifier_plan'], plan_digest),
            'classifier_result': self._load(classifier_files['classifier_result'], classifier_result_digest),
            'classifier_eval': self._load(classifier_files['classifier_eval'], reader_digest),
            'classifier_eval_validation': self._load(classifier_files['classifier_eval_validation'], reader_digest),
            'chain': self._load(classifier_files['chain'], chain_digest),
            'chain_validation': self._load(classifier_files['chain_validation'], chain_digest),
            'chain_old_reader': self._load(classifier_files['chain_old_reader'], chain_digest),
        }
        problems += [key for key, value in classifier.items() if value is UNREADABLE]
        classifier = {key: (None if value is UNREADABLE else value) for key, value in classifier.items()}
        classifier_log = self._load(classifier_files['classifier_log'], classifier_log_digest, text=True)
        classifier_log = None if classifier_log is UNREADABLE else classifier_log

        cutters = self._cutters(run)
        problems += cutters['problems']
        signnet = self._signnet(run)
        problems += signnet['problems']
        usernet = self._usernet(run)
        problems += usernet['problems']

        kinds = running.get(name, set())
        variants = self._variants(run)
        # A folder of scripts/run_experiment_b.sh alone has experiment-b/ and nothing of the pipeline; a folder of
        # scripts/run_classifier.sh has its files and none of the pipeline's networks; a folder of
        # scripts/run_signnet.sh has SignNet's files at its root, one of scripts/run_usernet.sh UserNet's, one of
        # scripts/run_detector_sizes.sh alone has detector-sizes/, and none of them.
        pipeline = log_path is not None or any((run / part).exists() for part in PIPELINE_PARTS)
        alone = log_path is None and not any((run / part).exists() for part in PIPELINE_ONLY)
        if alone and ('classifier' in kinds or any((run / part).exists() for part in CLASSIFIER_PARTS)):
            kind = 'classifier'
        elif alone and ('usernet' in kinds or usernet['found']):
            kind = 'usernet'
        elif alone and ('cutter' in kinds or signnet['found'] or (run / SIZES_DIR).is_dir()):
            kind = 'cutter'
        else:
            kind = 'training' if pipeline or not (run / EXPERIMENT_DIR).is_dir() else 'experiment'
        experiment_log = None
        if kind == 'experiment' or 'experiment' in kinds:
            experiment_log = self._experiment_log(run)
        if kind == 'classifier':
            stage = self._classifier_stage(classifier_files, classifier_log, 'classifier' in kinds)
        elif kind == 'usernet':
            stage = usernet['stage']
        elif kind == 'cutter' and (signnet['found'] or not cutters['sizes']):
            stage = signnet['stage']
        elif kind == 'cutter':
            sizes = cutters['sizes']
            stage = ('checks' if not sizes else 'detector' if not all(item['files']['result'] for item in sizes)
                     else 'evaluate' if not all(item['files']['test'] for item in sizes) else 'done')
        elif kind == 'experiment':
            done = sum(1 for variant, path in variants if variant in VARIANTS and path is not None)
            stage = 'done' if done == len(VARIANTS) else 'experiment'
        elif not files['detector_plan'] and not files['detector_result']:
            stage = 'checks'
        elif not files['detector_result']:
            stage = 'detector'
        elif not files['reader_result']:
            stage = 'reader'
        elif not (files['detector_eval'] and files['reader_eval']):
            stage = 'evaluate'
        elif not (files['detector_onnx'] and files['reader_onnx']):
            stage = 'export'
        elif not files['bundle']:
            stage = 'bundle'
        else:
            stage = 'done'
        is_running = {'training': 'pipeline', 'experiment': 'experiment', 'classifier': 'classifier', 'cutter': 'cutter',
                      'usernet': 'usernet'}[kind] in kinds
        state = 'running' if is_running else 'finished' if stage == 'done' else 'incomplete'

        experiment_files = [path for _, path in variants]
        experiment_files += [self._inside(run / EXPERIMENT_DIR / (variant + '.log'), run) for variant in VARIANTS]
        times = [time for time in (_mtime(path) for path in [*files.values(), log_path, *experiment_files,
                                                             *classifier_files.values(), *cutters['paths'],
                                                             *signnet['paths'], *usernet['paths']])
                 if time is not None]
        started = None
        match = TIME_NAME.fullmatch(name)
        if match:
            try:
                started = datetime(*(int(part) for part in match.groups())).astimezone().isoformat(timespec='seconds')
            except ValueError:
                started = None
        if started is None and kind == 'training':
            started = _iso(_mtime(files['detector_plan']) or _mtime(run))
        if started is None and kind == 'experiment':
            started = _iso(_mtime(run / EXPERIMENT_DIR / 'dry-run.json') or _mtime(run))
        if started is None and kind == 'classifier':
            started = _iso(min((time for time in (_mtime(classifier_files['classifier_log']), _mtime(classifier_files['negatives']))
                                if time is not None), default=None) or _mtime(run))
        if started is None and kind == 'usernet':
            started = _iso(_mtime(usernet['files']['log']) or _mtime(run))
        if started is None and kind == 'cutter':
            logs = [item['files']['log'] for item in cutters['sizes']] + [signnet['files']['log']]
            started = _iso(min((time for time in (_mtime(path) for path in logs) if time is not None),
                               default=None) or _mtime(run))
        finished = None
        if stage == 'done' and kind == 'training':
            finished = _iso(_mtime(files['bundle']))
        elif stage == 'done' and kind == 'experiment':
            finished = max((_iso(_mtime(path)) or '' for _, path in variants), default=None) or None
        elif stage == 'done' and kind == 'usernet':
            # UserNet writes its result.json last, once measured on validation and test.
            finished = _iso(_mtime(usernet['files']['result']))
        elif stage == 'done' and kind == 'cutter' and signnet['found']:
            # SignNet writes its result.json last, once measured on validation and test.
            finished = _iso(_mtime(signnet['files']['result']))
        elif stage == 'done' and kind == 'cutter':
            # The comparison is the last file of the sizes, else their last evaluation.
            finished = _iso(max((time for time in [_mtime(cutters['comparison'])] + [
                _mtime(item['files'][key]) for item in cutters['sizes'] for key in ('test', 'validation')] if time is not None),
                default=None))
        elif stage == 'done':
            # The chain and the comparison with the old reader are the last files of a classifier run.
            finished = _iso(max((time for time in (_mtime(classifier_files[key]) for key in
                                                   ('chain', 'chain_validation', 'chain_old_reader', 'comparison'))
                                 if time is not None), default=None))
        return {'name': name, 'run': run, 'kind': kind, 'files': files, 'digests': digests, 'log': log, 'problems': problems,
                'stage': stage, 'state': state, 'started_at': started, 'variants': variants,
                'experiment_running': 'experiment' in kinds, 'experiment_log': experiment_log,
                'classifier_files': classifier_files, 'classifier': classifier, 'classifier_log': classifier_log,
                'cutters': cutters, 'signnet': signnet, 'usernet': usernet,
                'updated_at': _iso(max(times)) if times else None,
                'finished_at': finished}

    def _signnet(self, run: Path) -> dict:
        """SignNet in this run (scripts/run_signnet.sh, or nnslr train-signnet by hand): its files at the run's root,
        its result.json (signnet_run_digest), its log (cutter_log_digest: signnet.log, else the train.log beside a
        signnet-best.pt), the step it reached (checks, signnet, evaluate, done once result.json is written).
        {'found', 'files', 'result', 'log', 'stage', 'paths', 'problems'}."""
        weights = self._file(run, SIGNNET_FILES['weights'])
        hand_log = self._file(run, SIGNNET_FILES['hand_log']) if weights is not None else None
        files = {'weights': weights, 'result': self._file(run, SIGNNET_FILES['result']),
                 'log': self._file(run, SIGNNET_FILES['log']) or hand_log}
        result = self._load(files['result'], signnet_run_digest)
        log = self._load(files['log'], cutter_log_digest, text=True)
        log = None if log is UNREADABLE else log
        # A result.json of another kind (a detector's, in an old run) is not SignNet's: neither found nor unreadable.
        found = files['weights'] is not None or files['log'] is not None or (result is not None and result is not UNREADABLE)
        problems = ['signnet_result'] if found and result is UNREADABLE and files['weights'] is not None else []
        result = None if result is UNREADABLE else result
        step = (log or {}).get('step')
        if result is not None:
            stage = 'done'
        elif step == 'evaluate' or (log or {}).get('complete'):
            # Measuring the best epoch, or done by its log with no result.json to read.
            stage = 'evaluate'
        elif files['weights'] is not None or (log or {}).get('curves', {}).get('signnet') or step == 'signnet':
            stage = 'signnet'
        else:
            stage = 'checks'
        return {'found': found, 'files': files, 'result': result, 'log': log, 'stage': stage,
                'paths': [path for path in files.values() if path is not None], 'problems': problems}

    def _usernet(self, run: Path) -> dict:
        """UserNet in this run (scripts/run_usernet.sh): its files at the run's root, its result.json
        (usernet_run_digest), its log (usernet_log_digest), the step it reached (checks, usernet, evaluate, done once its
        result.json is written). {'found', 'files', 'result', 'log', 'stage', 'paths', 'problems'}."""
        files = {key: self._file(run, relative) for key, relative in USERNET_FILES.items()}
        result = self._load(files['result'], usernet_run_digest)
        log = self._load(files['log'], usernet_log_digest, text=True)
        log = None if log is UNREADABLE else log
        # A result.json of another network is not UserNet's (None); one that cannot be read is, beside its weights.
        found = files['weights'] is not None or files['log'] is not None or (result is not None and result is not UNREADABLE)
        problems = ['usernet_result'] if result is UNREADABLE and files['weights'] is not None else []
        result = None if result is UNREADABLE else result
        step = (log or {}).get('step')
        if result is not None:
            stage = 'done'
        elif step == 'evaluate' or (log or {}).get('complete'):
            stage = 'evaluate'
        elif files['weights'] is not None or step == 'usernet' or ((log or {}).get('curves') or {}).get('usernet'):
            stage = 'usernet'
        else:
            stage = 'checks'
        return {'found': found, 'files': files, 'result': result, 'log': log, 'stage': stage,
                'paths': [path for path in files.values() if path is not None], 'problems': problems}

    def _cutters(self, run: Path) -> dict:
        """The detectors of scripts/run_detector_sizes.sh in this run (detector-sizes/<size>/ or its log), with their
        digests: as the Models tab judges a cutter (model_roles.cutter_digest at score 0.1), their training, their
        curves and plan. {'sizes': [...], 'paths', 'problems', 'comparison', 'log': the newest size log}."""
        from nnslr_tools.model_roles import cutter_digest, model_result_digest
        base = run / SIZES_DIR
        names: set[str] = set()
        try:
            if base.is_dir() and not base.is_symlink():
                for entry in base.iterdir():
                    stem = entry.name[:-len('.log')] if entry.name.endswith('.log') else entry.name
                    if SIZE_FOLDER.fullmatch(stem) and (entry.name.endswith('.log') or (entry.is_dir() and not entry.is_symlink())):
                        names.add(stem)
        except OSError:
            pass
        sizes, paths, problems = [], [], []
        newest: tuple[float, str, dict] | None = None
        for size in sorted(names, key=lambda item: [int(side) for side in item.split('x')]):
            folder = f'{SIZES_DIR}/{size}'
            files = {'weights': self._file(run, folder + '/detector-best.pt'), 'result': self._file(run, folder + '/result.json'),
                     'plan': self._file(run, folder + '/training-plan.json'),
                     'test': self._file(run, folder + '/evaluation/detector-test.json'),
                     'validation': self._file(run, folder + '/evaluation/detector-validation.json'),
                     'log': self._file(run, folder + '.log')}
            paths += [path for path in files.values() if path is not None]
            loaded = {'training': self._load(files['result'], model_result_digest),
                      'history': self._load(files['result'], result_digest),
                      'plan': self._load(files['plan'], plan_digest),
                      'test': self._load(files['test'], cutter_digest),
                      'validation': self._load(files['validation'], cutter_digest)}
            relative = {'training': '/result.json', 'plan': '/training-plan.json', 'test': '/evaluation/detector-test.json',
                        'validation': '/evaluation/detector-validation.json'}
            problems += sorted({folder + relative[key] for key, value in loaded.items() if value is UNREADABLE and key in relative})
            loaded = {key: (None if value is UNREADABLE else value) for key, value in loaded.items()}
            log = self._load(files['log'], cutter_log_digest, text=True)
            log = None if log is UNREADABLE else log
            if log is not None and (newest is None or (_mtime(files['log']) or 0) > newest[0]):
                newest = (_mtime(files['log']) or 0, size, log)
            sizes.append({'size': size, 'files': files, 'log': log, **loaded})
        comparison = self._file(run, SIZES_DIR + '/comparison.txt')
        if comparison is not None:
            paths.append(comparison)
        return {'sizes': sizes, 'paths': paths, 'problems': problems, 'comparison': comparison,
                'log': None if newest is None else {**newest[2], 'size': newest[1]}}

    @staticmethod
    def _classifier_stage(files: dict[str, Path | None], log: dict | None, running: bool) -> str:
        """The step a classifier run reached: the first of its files still missing; before the negatives, the
        mining once the log says it started. While it runs, a step its log is still in (a file written as it goes)."""
        if not files['negatives']:
            stage = 'checks'
        elif not files['classifier_result']:
            stage = 'classifier'
        elif not files['classifier_eval']:
            stage = 'evaluate'
        elif not files['chain']:
            stage = 'chain'
        else:
            stage = 'done'
        step = (log or {}).get('step')
        if step not in CLASSIFIER_STAGES:
            return stage
        if stage == 'checks' and step == 'mine':
            return 'mine'
        if running and CLASSIFIER_STAGES.index(step) < CLASSIFIER_STAGES.index(stage):
            return step
        return stage

    def _start(self, gathered: dict) -> dict:
        """How the networks started: ImageNet, random weights, the stable model or an earlier run."""
        log = gathered['log'] or {}
        modes = log.get('modes') or []
        mode = modes[0] if modes else None
        resumed = 'resume' in modes[1:] or mode == 'resume'
        weights = log.get('weights')
        results = [gathered['digests']['detector_result'], gathered['digests']['reader_result']]
        if gathered['kind'] == 'classifier':
            results = [gathered['classifier']['classifier_result']]
        if gathered['kind'] == 'cutter':
            results = [item['training'] for item in gathered['cutters']['sizes']]
            signnet = gathered['signnet']['result']
            if signnet is not None:
                # SignNet starts from random weights (nnslr_tools.signnet.build_signnet), or goes on from an earlier one.
                results.append({'initialization': 'start', 'start': signnet['start']} if signnet['start'] else {'initialization': 'random'})
        if gathered['kind'] == 'experiment':
            results = [self._load(path, experiment_digest) for _, path in gathered['variants'] if path is not None]
        if gathered['kind'] == 'usernet':
            # UserNet starts from random weights (nnslr_tools.usernet.build_usernet), or goes on from an earlier one.
            usernet = gathered['usernet']['result']
            went = (usernet or {}).get('start') or {}
            results = [] if usernet is None else [
                {'initialization': 'start', 'start': {'run': went['run'], 'source': None}} if went.get('run')
                else {'initialization': 'random'}]
        initialization = next((digest['initialization'] for digest in results
                               if isinstance(digest, dict) and digest.get('initialization')), None)
        # A SignNet or an applicability variant that went on from an earlier run (--init-from); a classifier's start has
        # no source (_start_digest) and is told by its initialization below.
        went_on = next((digest['start'] for digest in results if isinstance(digest, dict) and digest.get('start')),
                       None) if gathered['kind'] in ('cutter', 'experiment', 'usernet') else None
        if weights:
            from_run = FROM_RUN.search(weights)
            if from_run:
                start = {'kind': 'run', 'label': 'Continued from run ' + from_run.group(1), 'run': from_run.group(1)}
            elif '@' in weights:
                start = {'kind': 'stable', 'label': 'Continued from the stable model', 'run': None}
            else:
                start = {'kind': 'weights', 'label': 'Continued from ' + Path(weights).name, 'run': None}
            start['source'] = weights
        elif mode == 'continue':
            start = {'kind': 'stable', 'label': 'Continued from the stable model', 'run': None, 'source': None}
        elif went_on:
            start = {'kind': 'run', 'label': 'Continued from run ' + went_on['run'], 'run': went_on['run'],
                     'source': went_on['source']}
        elif initialization == 'imagenet1k_v1':
            start = {'kind': 'imagenet', 'label': 'Pre-trained on ImageNet', 'run': None, 'source': None}
        elif initialization == 'random':
            start = {'kind': 'random', 'label': 'Random weights', 'run': None, 'source': None}
        elif initialization and gathered['kind'] in ('classifier', 'cutter'):
            # The classifier may start from other weights (an earlier reader): its result.json says which.
            start = {'kind': 'weights', 'label': 'Started from ' + Path(initialization).name, 'run': None,
                     'source': initialization}
        else:
            start = {'kind': None, 'label': 'Not recorded yet' if gathered['state'] == 'running' else 'Not recorded',
                     'run': None, 'source': None}
        return {'mode': mode, 'resumed': resumed, 'start': start}

    def _summary(self, gathered: dict) -> dict:
        digests = gathered['digests']
        log = gathered['log'] or {}
        reader = digests['reader_eval']
        detector = digests['detector_eval']
        plan = digests['detector_plan'] or digests['reader_plan'] or {}
        split_sha = ((reader or {}).get('split_sha256') or (detector or {}).get('split_sha256') or plan.get('split_sha256'))
        dataset_sha = ((reader or {}).get('dataset_sha256') or (detector or {}).get('dataset_sha256') or plan.get('dataset_sha256'))
        experiment = gathered['variants']
        stage = gathered['stage']
        state = gathered['state']
        experiment_log = gathered['experiment_log'] or {}
        experiment_progress = None
        if gathered['experiment_running']:
            experiment_progress = ' · '.join(
                ['experiment B']
                + ([experiment_log['variant']] if experiment_log.get('variant') else [])
                + (['epoch ' + str(experiment_log['epoch']) + ' of ' + str(experiment_log['epochs'])]
                   if experiment_log.get('epoch') is not None else []))
        classifier = gathered['classifier']
        classifier_eval, chain = classifier['classifier_eval'], classifier['chain']
        if gathered['kind'] == 'experiment':
            first = next((self._load(path, experiment_digest) for _, path in experiment if path is not None), None)
            first = first if isinstance(first, dict) else {}
            split_sha = split_sha or first.get('split_sha256')
            dataset_sha = dataset_sha or first.get('dataset_sha256')
            progress_text, error = experiment_progress, experiment_log.get('error')
        elif gathered['kind'] == 'classifier':
            sources = [classifier_eval, chain, classifier['classifier_plan'], classifier['negatives']]
            split_sha = next((item['split_sha256'] for item in sources if item and item.get('split_sha256')), None)
            dataset_sha = next((item['dataset_sha256'] for item in sources if item and item.get('dataset_sha256')), None)
            classifier_log = gathered['classifier_log'] or {}
            progress_text, error = classifier_log.get('detail') or None, classifier_log.get('error')
        elif gathered['kind'] == 'usernet':
            usernet = gathered['usernet']
            split_sha = (usernet['result'] or {}).get('split_sha256')
            dataset_sha = (usernet['result'] or {}).get('dataset_sha256')
            progress_text, error = (usernet['log'] or {}).get('detail') or None, (usernet['log'] or {}).get('error')
        elif gathered['kind'] == 'cutter' and gathered['signnet']['found']:
            signnet = gathered['signnet']
            split_sha = (signnet['result'] or {}).get('split_sha256')
            dataset_sha = (signnet['result'] or {}).get('dataset_sha256')
            progress_text, error = (signnet['log'] or {}).get('detail') or None, (signnet['log'] or {}).get('error')
        elif gathered['kind'] == 'cutter':
            plans = [item['plan'] for item in gathered['cutters']['sizes'] if item['plan']]
            split_sha = next((plan['split_sha256'] for plan in plans if plan.get('split_sha256')), None)
            dataset_sha = next((plan['dataset_sha256'] for plan in plans if plan.get('dataset_sha256')), None)
            size_log = gathered['cutters']['log'] or {}
            progress_text = size_log['size'] + ' · ' + size_log['detail'] if size_log.get('detail') else None
            error = size_log.get('error')
        else:
            progress_text, error = log.get('detail') or None, log.get('error')
        trained = classifier['classifier_result'] or {}
        stage_label = STAGE_LABELS[stage]
        if gathered['kind'] == 'classifier':
            stage_label = CLASSIFIER_STAGE_LABELS.get(stage, stage_label)
        if gathered['kind'] == 'cutter':
            stage_label = CUTTER_STAGE_LABELS.get(stage, stage_label)
        if gathered['kind'] == 'usernet':
            stage_label = USERNET_STAGE_LABELS.get(stage, stage_label)
        usernet_line = self._usernet_headline(gathered['usernet'])
        cutters = [self._cutter_headline(item) for item in gathered['cutters']['sizes']]
        signnet_line = self._signnet_headline(gathered['signnet'])
        if signnet_line is not None:
            cutters.append(signnet_line)
        summary = {
            'name': gathered['name'],
            'kind': gathered['kind'],
            'state': state,
            'stage': stage,
            'stage_label': stage_label,
            'progress': progress_text if state == 'running' else None,
            'error': error if state == 'incomplete' else None,
            'experiment_running': gathered['experiment_running'],
            'experiment_progress': experiment_progress,
            'started_at': gathered['started_at'],
            'updated_at': gathered['updated_at'],
            'finished_at': gathered['finished_at'],
            **self._start(gathered),
            'dataset_sha256': dataset_sha,
            'split_sha256': split_sha,
            'evaluated': (reader is not None or detector is not None or classifier_eval is not None or chain is not None
                          or any(item['evaluated'] for item in cutters)
                          or bool(gathered['signnet']['result'] and gathered['signnet']['result']['test'])
                          or bool(usernet_line and usernet_line['partition'])),
            'reader': None if reader is None else {
                'samples': reader['samples'], 'accuracy': reader['accuracy'],
                'human_samples': (reader['human'] or {}).get('samples'), 'human_accuracy': (reader['human'] or {}).get('accuracy')},
            'detector': None if detector is None else {
                'truth_boxes': detector['truth_boxes'], 'found': detector['found'], 'missed': detector['missed'],
                'false_boxes': detector['false_boxes'], 'recall': detector['recall'], 'precision': detector['precision'],
                'f1': detector['f1'],
                'human_recall': (detector['human'] or {}).get('recall'), 'human_precision': (detector['human'] or {}).get('precision')},
            'best': {'detector_val_loss': (digests['detector_result'] or {}).get('best_validation_loss'),
                     'reader_val_accuracy': (digests['reader_result'] or {}).get('best_validation_accuracy')},
            # The classifier with "not a sign" (scripts/run_classifier.sh): alone on the test crops, its best epoch on
            # validation, and the chain end to end at the cutter's score 0.1, with the old reader beside it.
            'classifier': None if classifier_eval is None and not trained else {
                **classifier_headline(classifier_eval),
                'best_validation_balanced_accuracy': trained.get('best_validation_balanced_accuracy'),
                'best_validation_accuracy': trained.get('best_validation_accuracy'),
                'best_epoch': trained.get('best_epoch')},
            'chain': None if chain is None else chain_headline(chain),
            'chain_old_reader': None if classifier['chain_old_reader'] is None else chain_headline(classifier['chain_old_reader']),
            # The detectors of scripts/run_detector_sizes.sh (a cutter run, or the sizes compared in a pipeline run),
            # kind 'detector' at their score 0.1; then the run's SignNet, kind 'signnet': what the car reads of it
            # (its best max_boxes boxes of score_threshold 0.3 or more).
            'cutters': cutters or None,
            # A UserNet run (scripts/run_usernet.sh): its stage and, on the test (else validation), the road users found
            # at its score (usernet.MIN_USER_SCORE) and its false boxes a frame.
            'usernet': usernet_line,
            'experiment_b': [variant for variant, path in experiment if path is not None],
            'problems': [PROBLEM_FILES.get(key, key) + ' is unreadable' for key in gathered['problems']],
        }
        summary['version'] = hashlib.sha1(json.dumps([summary, self._stat_key(gathered)], sort_keys=True, default=str)
                                          .encode()).hexdigest()[:12]
        return summary

    @staticmethod
    def _stat_key(gathered: dict) -> list:
        key = []
        for path in [*gathered['files'].values(), *gathered['classifier_files'].values(), *gathered['cutters']['paths'],
                     *gathered['signnet']['paths'], *gathered['usernet']['paths']]:
            if path is not None:
                try:
                    info = path.stat()
                    key.append((str(path), info.st_mtime_ns, info.st_size))
                except OSError:
                    pass
        return key

    @staticmethod
    def _signnet_headline(signnet: dict) -> dict | None:
        """The SignNet of this run in the shape of a cutter line (kind 'signnet'): what the car reads of it, its best 5
        boxes of score 0.3 or more, on test, else validation. None without its result.json."""
        result = signnet['result']
        if result is None:
            return None
        partition = next((name for name in ('test', 'validation') if (result[name] or {}).get('car')), None)
        car = (result[partition] or {})['car'] if partition else {}
        band = result['band']
        from nnslr_tools.signnet import CAR_MAX_BOXES, CAR_MIN_SCORE
        return {'kind': 'signnet', 'size': f'{band[0]}x{band[1]}' if band else 'signnet',
                'families': result['families'] or 'all', 'finished': True, 'evaluated': partition is not None,
                'partition': partition, 'score_threshold': car.get('score_threshold', CAR_MIN_SCORE),
                'max_boxes': car.get('max_boxes', CAR_MAX_BOXES), 'recall': car.get('recall'), 'found': car.get('found'),
                'truth_boxes': car.get('truth_boxes'), 'speed_recall': car.get('speed_recall'),
                'boxes_per_frame': car.get('boxes_per_frame'), 'false_per_frame': car.get('false_per_frame'),
                'ms_per_frame': None, 'fits_the_car': result['fits_the_car']}

    @staticmethod
    def _usernet_headline(usernet: dict) -> dict | None:
        """The UserNet of this run in a line: its stage, whether it carries BDD100K's licence, where it was validated,
        and on the test (else validation) the road users found at its score and its false boxes a frame. None without
        its result.json."""
        result = usernet['result']
        if result is None:
            return None
        partition = next((name for name in ('test', 'validation') if (result[name] or {}).get('frames')), None)
        measured = (result[partition] or {}) if partition else {}
        return {'stage': result['stage'], 'uses_bdd': result['uses_bdd'], 'validation_source': result['validation_source'],
                'best_epoch': result['best_epoch'], 'partition': partition, 'frames': measured.get('frames'),
                'score_threshold': MIN_USER_SCORE, 'recall': measured.get('recall'),
                'precision': measured.get('precision'), 'false_per_frame': measured.get('false_per_frame')}

    @staticmethod
    def _cutter_headline(item: dict) -> dict:
        """One size of the cutter in a line: its signs found at score 0.1 (on test, else validation), the speed signs
        among them for a cutter that also finds the other signs, the boxes per frame it passes on."""
        measured = item['test'] or item['validation'] or {}
        trained = item['training'] or {}
        speed = (measured.get('by_family') or {}).get('speed') or {}
        return {'kind': 'detector', 'size': item['size'],
                'families': measured.get('families') or trained.get('families') or 'speed',
                'finished': item['training'] is not None, 'evaluated': bool(measured),
                'partition': measured.get('partition'), 'score_threshold': measured.get('score_threshold'),
                'recall': measured.get('recall'), 'found': measured.get('found'), 'truth_boxes': measured.get('truth_boxes'),
                'speed_recall': speed.get('recall'), 'boxes_per_frame': measured.get('boxes_per_frame'),
                'false_per_frame': measured.get('false_per_frame'), 'ms_per_frame': measured.get('ms_per_frame')}

    def _cutter_detail(self, gathered: dict) -> dict | None:
        """What scripts/run_detector_sizes.sh wrote in this run: each size with its training and its numbers on test
        and validation (as the Models tab judges a cutter), and the comparison in plain text. None without a size."""
        cutters = gathered['cutters']
        if not cutters['sizes']:
            return None
        comparison = self._load(cutters['comparison'], comparison_digest, text=True)
        return {
            'sizes': [{'size': item['size'], 'label': item['size'].replace('x', ' × '),
                       'families': self._cutter_headline(item)['families'], 'finished': item['training'] is not None,
                       'training': item['training'], 'test': item['test'], 'validation': item['validation'],
                       'files': {key: path is not None for key, path in item['files'].items()}}
                      for item in cutters['sizes']],
            'comparison': None if comparison is UNREADABLE else comparison,
        }

    def _variants(self, run: Path) -> list[tuple[str, Path | None]]:
        """Experiment B: the known variants first, then any other folder with a result.json."""
        base = run / EXPERIMENT_DIR
        names = list(VARIANTS)
        try:
            if base.is_dir() and not base.is_symlink():
                names += sorted(entry.name for entry in base.iterdir()
                                if RUN_NAME.fullmatch(entry.name) and entry.name not in VARIANTS
                                and entry.is_dir() and not entry.is_symlink())
        except OSError:
            pass
        found = [(variant, self._inside(base / variant / 'result.json', run)) for variant in names]
        return found if any(path is not None for _, path in found) else []

    def _experiment_log(self, run: Path) -> dict | None:
        """The newest <run>/experiment-b/<variant>.log: the variant training now, its epoch, the last error."""
        base = run / EXPERIMENT_DIR
        newest = None
        for variant in VARIANTS:
            path = self._inside(base / (variant + '.log'), run)
            if path is not None and (newest is None or (_mtime(path) or 0) > (_mtime(newest) or 0)):
                newest = path
        digest = self._load(newest, experiment_log_digest, text=True)
        if not isinstance(digest, dict) or newest is None:
            return None
        # Before its first epoch line, the variant is the name of its log.
        return {**digest, 'variant': digest['variant'] or newest.stem}

    def _experiment(self, run: Path) -> dict | None:
        """Experiment B of this run: the variants side by side, None when no variant has a result yet."""
        variants = self._variants(run)
        if not variants:
            return None
        digests = {}
        listed = []
        for variant, path in variants:
            digest = self._load(path, experiment_digest)
            entry = {'key': variant, 'label': VARIANT_LABELS.get(variant, _words(variant)), 'found': path is not None,
                     'unreadable': digest is UNREADABLE, 'at': _iso(_mtime(path)), 'inputs': None, 'best_epoch': None,
                     'epochs': None, 'selection_score': None, 'initialization': None, 'split_sha256': None, 'limit': None}
            if digest is not None and digest is not UNREADABLE:
                digests[variant] = digest
                entry.update({key: digest[key] for key in ('inputs', 'best_epoch', 'epochs', 'selection_score',
                                                           'initialization', 'split_sha256', 'limit')})
            listed.append(entry)
        selection = next((digest['selection_metric'] for digest in digests.values() if digest['selection_metric']), None)
        return {'variants': listed, 'selection_metric': selection,
                **experiment_table(digests, [entry['key'] for entry in listed])}

    def _dataset(self, gathered: dict) -> dict | None:
        digests = gathered['digests']
        detector_plan, reader_plan = digests['detector_plan'], digests['reader_plan']
        classifier_plan = gathered['classifier']['classifier_plan']
        if detector_plan is None and gathered['kind'] == 'cutter':
            # A cutter run: the plan of its (newest) size is its detector's.
            detector_plan = next((item['plan'] for item in reversed(gathered['cutters']['sizes']) if item['plan']), None)
        plan = detector_plan or reader_plan or classifier_plan
        if plan is None:
            return None
        split_path = self._split_file(plan)
        split = self._load(split_path, split_digest)
        latest = None
        try:
            pointer = _read_json(self.root / 'splits' / 'latest.json')
            latest = pointer.get('sha256') if isinstance(pointer, dict) else None
        except (OSError, ValueError):
            pass
        relative = None
        if split_path is not None:
            try:
                relative = str(split_path.relative_to(self.root.resolve()))
            except ValueError:
                relative = None
        return {
            'dataset_sha256': plan.get('dataset_sha256'), 'split_sha256': plan.get('split_sha256'),
            'dataset_kind': plan.get('dataset_kind'), 'split_file': relative,
            'split': None if split is None or split is UNREADABLE else split,
            'current_split': latest is not None and latest == plan.get('split_sha256'),
            'detector': None if detector_plan is None else {
                'frames': detector_plan['counts_by_split'], 'boxes': detector_plan['boxes_by_split'],
                'negative_frames': detector_plan['negative_frames_by_split']},
            'reader': None if reader_plan is None else {'crops': reader_plan['counts_by_split'], 'classes': reader_plan['classes']},
            # The classifier with "not a sign": its crops, its "not a sign" crops by where they come from, the labels
            # it learnt as a broader one.
            'classifier': None if classifier_plan is None else {
                'crops': classifier_plan['counts_by_split'], 'classes': classifier_plan['classes'],
                'class_scheme': classifier_plan['class_scheme'], 'negatives': classifier_plan['negatives'],
                'folded_labels': classifier_plan['folded_labels']},
        }

    @staticmethod
    def _curves(gathered: dict) -> dict:
        """The curves of the networks: from the log (with the time of each epoch), else from result.json; the
        classifier with "not a sign" from its own classifier.log, else from its result.json, and its pretraining on
        the Mapillary signs alone (two stages, "pretrain") from that log."""
        curves = dict((gathered['log'] or {}).get('curves') or {})
        for net in ('detector', 'reader'):
            result = gathered['digests'][net + '_result']
            if net not in curves and result and result['history']:
                curves[net] = {'epochs': result['epochs'] or result['history'][-1]['epoch'], 'points': result['history']}
        if gathered['kind'] == 'cutter':
            # The detector of the size trained last: from its log, else from its result.json.
            size_log = gathered['cutters']['log']
            if size_log and (size_log.get('curves') or {}).get('detector'):
                curves['detector'] = size_log['curves']['detector']
            else:
                history = next((item['history'] for item in reversed(gathered['cutters']['sizes'])
                                if item['history'] and item['history']['history']), None)
                if history:
                    curves['detector'] = {'epochs': history['epochs'] or history['history'][-1]['epoch'],
                                          'points': history['history']}
        signnet = gathered['signnet']
        if signnet['found']:
            # SignNet: from its log when its epochs carry the numbers (scripts/run_signnet.sh), else from its
            # result.json (a SignNet trained by hand logged only the loss and the recall in words).
            logged = ((signnet['log'] or {}).get('curves') or {}).get('signnet')
            history = (signnet['result'] or {}).get('history')
            if logged and any('train_loss' in point for point in logged['points']):
                curves['signnet'] = logged
            elif history:
                curves['signnet'] = {'epochs': signnet['result']['epochs'] or history[-1]['epoch'], 'points': history}
            elif logged:
                curves['signnet'] = logged
        usernet = gathered['usernet']
        if usernet['found']:
            # UserNet: from its log (with the time of each epoch), else from its result.json.
            logged = ((usernet['log'] or {}).get('curves') or {}).get('usernet')
            history = (usernet['result'] or {}).get('history')
            if logged and logged['points']:
                curves['usernet'] = logged
            elif history:
                curves['usernet'] = {'epochs': usernet['result']['epochs'] or history[-1]['epoch'], 'points': history}
        logged = ((gathered['classifier_log'] or {}).get('curves') or {}).get('classifier')
        pretraining = ((gathered['classifier_log'] or {}).get('curves') or {}).get('pretrain')
        if pretraining:
            curves['pretrain'] = pretraining
        result = gathered['classifier']['classifier_result']
        if logged:
            curves['classifier'] = logged
        elif result and result['history']:
            curves['classifier'] = {'epochs': result['epochs'] or result['history'][-1]['epoch'], 'points': result['history']}
            if result['stopped_early']:
                curves['classifier']['stopped_early'] = result['stopped_early']
        return curves

    def _classifier_detail(self, gathered: dict) -> dict | None:
        """What scripts/run_classifier.sh wrote in this run: the classifier alone on test and validation, the chain
        (test, validation, the old reader in the same chain), the negatives it learnt from, its training, the
        comparison in plain text. None when it wrote nothing."""
        files = gathered['classifier_files']
        if not any(path is not None for key, path in files.items() if key != 'classifier_log'):
            return None
        digests = gathered['classifier']
        result = digests['classifier_result']
        comparison = self._load(files['comparison'], comparison_digest, text=True)
        return {
            'files': {key: path is not None for key, path in files.items()},
            'training': None if result is None else {key: value for key, value in result.items() if key != 'history'},
            'negatives': digests['negatives'],
            'evaluations': {'test': digests['classifier_eval'], 'validation': digests['classifier_eval_validation']},
            'chains': {'test': digests['chain'], 'validation': digests['chain_validation'],
                       'old_reader': digests['chain_old_reader']},
            'comparison': None if comparison is UNREADABLE else comparison,
        }

    # -- the endpoints ---------------------------------------------------------------------------------
    def names(self) -> list[str]:
        runs = self.runs_dir
        try:
            if not runs.is_dir():
                return []
            return [entry.name for entry in runs.iterdir() if self._run_dir(entry.name) is not None]
        except OSError:
            return []

    def summaries(self) -> dict:
        """GET /api/panel/runs: every run, newest first."""
        running = self.running_runs()
        rows = []
        for name in self.names():
            run = self._run_dir(name)
            if run is not None:
                rows.append(self._summary(self._gather(name, run, running)))
        rows.sort(key=lambda row: (row['started_at'] or '', row['name']), reverse=True)
        self._prune()
        return {'runs': rows, 'runs_dir_exists': self.runs_dir.is_dir()}

    def detail(self, name: str) -> dict | None:
        """GET /api/panel/runs/<run>: everything about one run; None when there is no such run."""
        run = self._run_dir(name)
        if run is None:
            return None
        gathered = self._gather(name, run, self.running_runs())
        digests = gathered['digests']
        training = {}
        for net in ('detector', 'reader'):
            result = digests[net + '_result']
            training[net] = None if result is None else {key: value for key, value in result.items() if key != 'history'}
        return {
            **self._summary(gathered),
            'files': {key: path is not None for key, path in gathered['files'].items()},
            'training': training,
            'reader_eval': digests['reader_eval'],
            'detector_eval': digests['detector_eval'],
            'dataset': self._dataset(gathered),
            'curves': self._curves(gathered),
            'experiment': self._experiment(run),
            'classifier_run': self._classifier_detail(gathered),
            'cutter_run': self._cutter_detail(gathered),
            # A SignNet run: its shape, training and numbers at its best 5 boxes a frame at any score, and in each
            # partition's 'car' as the car reads it (those 5 of score 0.3 or more).
            'signnet_run': (None if gathered['signnet']['result'] is None else
                            {key: gathered['signnet']['result'][key] for key in SIGNNET_RUN_KEYS}),
            # A UserNet run: its numbers as Models has them (usernet.digest), without its epochs (in curves).
            'usernet_run': (None if gathered['usernet']['result'] is None else
                            {key: value for key, value in gathered['usernet']['result'].items() if key != 'history'}),
            'watch': self._watch_models(run, gathered),
        }

    def _watch_models(self, run: Path, gathered: dict) -> list[dict]:
        """The reports of this run the Watch panel can play, by model, with the partitions that have one (only
        which files exist: a report is read when its frames are asked for)."""
        models = ['detector'] + [f"{SIZES_DIR}/{item['size']}" for item in gathered['cutters']['sizes']] + ['chain', 'chain-old-reader']
        found = []
        for model in models:
            partitions = [partition for partition in WATCH_PARTITIONS
                          if (relative := watch_file(model, partition)) is not None and self._file(run, relative) is not None]
            if partitions:
                found.append({'model': model, 'label': watch_label(model),
                              'kind': 'chain' if model.startswith('chain') else 'cutter', 'partitions': partitions})
        return found

    def frames(self, name: str, model: str, partition: str, route: str | None = None,
               previews: Callable[[str], dict] | None = None) -> dict:
        """GET /api/panel/runs/<run>/frames: the frames of one route of a report of the Watch panel, in the order of the
        drive, with the boxes to draw over each (the route asked for, else the one with the most signs); and every
        route of the report with its frames, signs and misses at the score the panel starts at. previews(route) gives
        the rows of the current pre-annotation report of a route by frame key ({} without one): a frame shows its
        preview (/route/<route>/images/NNNNNNNN.jpg) only when the row is the same image."""
        run = self._run_dir(name)
        if run is None:
            raise WatchError(404, 'unknown_run', 'No such training run.')
        if not isinstance(model, str) or not WATCH_MODEL.fullmatch(model):
            raise WatchError(400, 'invalid_model', 'model must be detector, detector-sizes/<W>x<H>, chain or chain-old-reader.')
        if partition not in WATCH_PARTITIONS:
            raise WatchError(400, 'invalid_partition', 'partition must be test or validation.')
        if route is not None and (not isinstance(route, str) or not WATCH_ROUTE.fullmatch(route)):
            raise WatchError(400, 'invalid_route', 'Invalid route id.')
        relative = watch_file(model, partition)
        path = self._file(run, relative) if relative is not None else None
        if path is None:
            raise WatchError(404, 'no_report', f'{watch_label(model)} has no evaluation on {partition} in this run.')
        digest = self._load(path, frames_digest)
        if digest is None or digest is UNREADABLE:
            raise WatchError(409, 'unreadable_report', f'{relative} cannot be read.')
        level, start = digest['level'], default_score(digest)
        listed = []
        for key, frames in sorted(digest['routes'].items()):
            outcomes = [_frame_outcome(frame, start, level) for frame in frames]
            missed = [count for _, count in outcomes]
            listed.append({'route': key, 'frames': len(frames), 'signs': sum(signs for signs, _ in outcomes),
                           'frames_with_signs': sum(1 for signs, _ in outcomes if signs),
                           'missed': None if any(count is None for count in missed) else sum(missed),  # type: ignore[arg-type]
                           'frames_with_misses': None if any(count is None for count in missed)
                           else sum(1 for count in missed if count)})
        asked = route
        if route not in digest['routes']:
            # None, or a route this report has no frame of (the panel keeps its route when another report is chosen):
            # the route with the most reviewed signs, the first by name on a tie.
            route = max(listed, key=lambda item: (item['signs'], item['frames']))['route'] if listed else None
        rows = {}
        if route is not None and previews is not None:
            try:
                rows = previews(route) or {}
            except (OSError, ValueError):
                rows = {}
        frames = []
        for frame in digest['routes'].get(route, []) if route is not None else []:
            row = rows.get(frame['key']) if frame['key'] is not None else None
            image = None
            width, height = frame['width'], frame['height']
            if isinstance(row, dict) and row.get('image_path') == frame['image_path'] and (
                    not isinstance(row.get('frame_sha256'), str) or frame['image_sha256'] is None
                    or row['frame_sha256'] == frame['image_sha256']):
                preview = row.get('preview_path')
                if isinstance(preview, str) and PREVIEW_PATH.fullmatch(preview):
                    image = f'/route/{route}/{preview}'
                if width is None or height is None:
                    width, height = _int(row.get('width')), _int(row.get('height'))
            frames.append({'key': frame['key'], 'segment': frame['segment'], 'index': frame['index'], 'image': image,
                           'width': width, 'height': height, 'basis': frame['basis'], 'truth': frame['truth'],
                           # [x1, y1, x2, y2, score, truth] (a chain: + label, confidence): a route has thousands.
                           'boxes': [[*item['box'], item['score'], item['match'],
                                      *((item['label'], item['confidence']) if 'label' in item else ())]
                                     for item in frame['boxes']],
                           'counts': frame['counts'], 'surviving': frame['surviving']})
        return {'run': name, 'model': model, 'label': watch_label(model), 'partition': partition, 'kind': digest['kind'],
                'level': level, 'min_score': digest['min_score'], 'main_score': digest['main_score'],
                'default_score': start, 'max_boxes': digest['max_boxes'],
                'steps': list(PER_FRAME_STEPS) if level == 'all' else None, 'iou_threshold': digest['iou_threshold'],
                'rejects': digest['rejects'], 'image_count': digest['image_count'], 'routes': listed, 'route': route,
                # The route asked for when this report has no frame of it.
                'route_not_here': asked if asked is not None and asked != route else None,
                'with_image': sum(1 for frame in frames if frame['image'] is not None), 'frames': frames}

    def _prune(self) -> None:
        """Forget the digests of files that are gone (a run deleted)."""
        with self._lock:
            for slot in [slot for slot in self._cache if not Path(slot[0]).exists()]:
                del self._cache[slot]


# The files of a classifier run (scripts/run_classifier.sh), by the key of RunsIndex._gather.
CLASSIFIER_FILES = {
    'negatives': 'negatives.jsonl', 'classifier_plan': 'classifier/training-plan.json',
    'classifier_result': 'classifier/result.json', 'classifier_weights': 'classifier/reader-best.pt',
    'classifier_eval': 'evaluation/classifier-test.json', 'classifier_eval_validation': 'evaluation/classifier-validation.json',
    'chain': 'evaluation/chain-test.json', 'chain_validation': 'evaluation/chain-validation.json',
    'chain_old_reader': 'evaluation/chain-test-old-reader.json', 'comparison': 'comparison.txt',
    'classifier_log': 'classifier.log',
}
PROBLEM_FILES = {'detector_plan': 'detector/training-plan.json', 'reader_plan': 'reader/training-plan.json',
                 'detector_result': 'detector/result.json', 'reader_result': 'reader/result.json',
                 'detector_eval': 'evaluation/detector-test.json', 'reader_eval': 'evaluation/reader-test.json',
                 'signnet_result': 'result.json', 'usernet_result': 'result.json', **CLASSIFIER_FILES}
# [control-panel] - END
