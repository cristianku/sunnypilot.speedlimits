"""The settings of the trainings of Train (Cutter, Classifier, Applicability, Scene): the epochs, the patience, the batch,
the learning rate, the processes reading the frames and the classifier's false boxes. Each one is an environment
variable its script already reads (scripts/run_signnet.sh, run_classifier.sh, run_experiment_b.sh, run_road_type.sh)
with the same default: the panel sets the ones changed here when it starts the next training of that kind
(Panel.start, environment), the one running keeps its own. The job's chips say the values a training used (the head
of its log, panel._settings_of).

Kept in ``models/train-settings.json`` of the data root: {"trainings": {action: {"values": {key: number}, "by",
"at"}}}, only the values changed from the default. What the car needs (the cutter's band, grid, channels, fuse) is
shown as fixed, never set here; nothing that only makes runs harder to compare, such as a seed, is a setting.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

from nnslr_tools.annotations import atomic_write

SETTINGS_FILE = Path('models') / 'train-settings.json'
# The processes reading the frames when none is set: half the cores, 4 to 16, as the scripts compute it
# (nnslr_tools.training.DEFAULT_WORKERS, without importing torch).
DEFAULT_WORKERS = min(16, max(4, (os.cpu_count() or 8) // 2))


def _workers(help_text: str) -> dict:
    return {'key': 'workers', 'env': 'NNSLR_NUM_WORKERS', 'label': 'Processes reading the frames',
            'help': help_text + ' Only the speed changes. Default: half the cores, 4 to 16.',
            'default': DEFAULT_WORKERS, 'min': 1, 'max': 32, 'step': 1, 'unit': 'workers'}


# By action of the panel: its label, its settings (key, the environment variable of its script, label, what it does,
# default, min, max, step, unit; same_as: the default is the value of that setting, as in the script) and what is
# fixed (label, value, why).
TRAININGS = {
    'train_cutter': {'label': 'Cutter', 'script': 'scripts/run_signnet.sh', 'settings': (
        {'key': 'epochs', 'env': 'NNSLR_SIGNNET_EPOCHS', 'label': 'Epochs',
         'help': 'Passes over the training frames; the best one on validation is kept (recall in the 5 best boxes of '
                 'a frame).', 'default': 40, 'min': 1, 'max': 200, 'step': 1, 'unit': 'epochs'},
        {'key': 'batch', 'env': 'NNSLR_SIGNNET_BATCH', 'label': 'Batch',
         'help': 'Frames in each step of the training: more takes more of the GPU.', 'default': 16, 'min': 1,
         'max': 128, 'step': 1, 'unit': 'frames'},
        {'key': 'learning_rate', 'env': 'NNSLR_SIGNNET_LEARNING_RATE', 'label': 'Learning rate',
         'help': 'How far each step moves the weights.', 'default': 0.003, 'min': 0.00001, 'max': 0.1,
         'step': 0.00001, 'unit': ''},
        _workers('The processes reading the frames for the GPU.'),
    ), 'fixed': (
        {'label': 'Band', 'value': '896 × 288', 'help': 'The top of the frame the car looks at.'},
        {'label': 'Grid', 'value': 'stride 8', 'help': 'A box can start every 8 pixels.'},
        {'label': 'Channels', 'value': '16, 24, 32, 48', 'help': 'The size of the network.'},
        {'label': 'Fuse', 'value': '1 × 1', 'help': 'The convolution that joins the scales.'},
    )},
    'train_classifier': {'label': 'Classifier', 'script': 'scripts/run_classifier.sh', 'settings': (
        {'key': 'epochs', 'env': 'NNSLR_CLASSIFIER_EPOCHS', 'label': 'Epochs',
         'help': 'Passes over the training crops; the best one on validation is kept (balanced accuracy).',
         'default': 30, 'min': 1, 'max': 200, 'step': 1, 'unit': 'epochs'},
        {'key': 'pretrain_epochs', 'env': 'NNSLR_CLASSIFIER_PRETRAIN_EPOCHS', 'label': 'Epochs on Mapillary alone',
         'help': 'Only with “Mapillary, then ours”: the first stage. Default: the same as the epochs.',
         'default': 30, 'same_as': 'epochs', 'min': 1, 'max': 200, 'step': 1, 'unit': 'epochs'},
        {'key': 'patience', 'env': 'NNSLR_CLASSIFIER_PATIENCE', 'label': 'Patience',
         'help': 'Stops after this many epochs without a better validation; 0: never stops early.', 'default': 5,
         'min': 0, 'max': 200, 'step': 1, 'unit': 'epochs'},
        {'key': 'batch', 'env': 'NNSLR_CLASSIFIER_BATCH_SIZE', 'label': 'Batch',
         'help': 'Crops in each step of the training: more takes more of the GPU.', 'default': 64, 'min': 1,
         'max': 256, 'step': 1, 'unit': 'crops'},
        _workers('The processes reading the crops and the frames, also for mining and measuring.'),
        {'key': 'negative_score', 'env': 'NNSLR_NEGATIVE_SCORE', 'label': 'False boxes from score',
         'help': 'The cutter’s boxes on no sign learnt as “not a sign”: those of this score or more. Another value '
                 'changes the “not a sign” crops of the test too: compare classifiers by the chain.',
         'default': 0.1, 'min': 0.01, 'max': 0.9, 'step': 0.01, 'unit': 'score'},
        {'key': 'negatives_per_frame', 'env': 'NNSLR_NEGATIVES_PER_FRAME', 'label': 'False boxes a frame',
         'help': 'At most this many of a frame.', 'default': 3, 'min': 1, 'max': 20, 'step': 1, 'unit': 'boxes'},
        {'key': 'negatives_factor', 'env': 'NNSLR_NEGATIVES_FACTOR', 'label': 'False boxes in all',
         'help': 'At most this many times the sign crops of each partition.', 'default': 3.0, 'min': 0.5,
         'max': 20.0, 'step': 0.5, 'unit': '× signs'},
    ), 'fixed': ()},
    'train_decider': {'label': 'Applicability', 'script': 'scripts/run_experiment_b.sh', 'settings': (
        {'key': 'epochs', 'env': 'NNSLR_EXPERIMENT_EPOCHS', 'label': 'Epochs',
         'help': 'Passes over the training signs, for each of the three versions.', 'default': 20, 'min': 1,
         'max': 200, 'step': 1, 'unit': 'epochs'},
    ), 'fixed': ()},
    'train_road_type': {'label': 'Scene', 'script': 'scripts/run_road_type.sh', 'settings': (
        {'key': 'epochs', 'env': 'NNSLR_ROAD_TYPE_EPOCHS', 'label': 'Epochs',
         'help': 'Passes over the drives, for each of the three networks; the best one on validation is kept.',
         'default': 60, 'min': 1, 'max': 200, 'step': 1, 'unit': 'epochs'},
        {'key': 'patience', 'env': 'NNSLR_ROAD_TYPE_PATIENCE', 'label': 'Patience',
         'help': 'Stops after this many epochs without a better validation.', 'default': 15, 'min': 1, 'max': 200,
         'step': 1, 'unit': 'epochs'},
        {'key': 'batch', 'env': 'NNSLR_ROAD_TYPE_BATCH', 'label': 'Batch',
         'help': 'Stretches of drive in each step of the training: more takes more of the GPU.', 'default': 12,
         'min': 1, 'max': 128, 'step': 1, 'unit': 'stretches'},
        _workers('The processes reading the frames.'),
        {'key': 'sign_workers', 'env': 'NNSLR_SIGN_WORKERS', 'label': 'Processes reading the signs',
         'help': 'The processes reading the signs and the car’s motion of the drives not read yet, each with its '
                 'ffmpeg, on the CPU. Only the speed changes. Default: half the cores.',
         'default': max(1, (os.cpu_count() or 8) // 2), 'min': 1, 'max': 64, 'step': 1, 'unit': 'workers'},
    ), 'fixed': ()},
}


class SettingsError(ValueError):
    """A training that has no settings, a value that is not a setting, not a number or out of its bounds."""


def _training(action) -> dict:
    training = TRAININGS.get(action) if isinstance(action, str) else None
    if training is None:
        raise SettingsError('Unknown training: ' + ', '.join(TRAININGS) + '.')
    return training


def _setting(training: dict, key: str) -> dict | None:
    return next((setting for setting in training['settings'] if setting['key'] == key), None)


def _number(setting: dict, value) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SettingsError(f'{setting["key"]} must be a number.')
    if not setting['min'] <= value <= setting['max']:
        raise SettingsError(f'{setting["key"]} goes from {setting["min"]:g} to {setting["max"]:g}.')
    if isinstance(setting['default'], int):
        if value != int(value):
            raise SettingsError(f'{setting["key"]} is a whole number.')
        return int(value)
    return float(value)


def _stored(root: Path) -> dict:
    try:
        value = json.loads((Path(root) / SETTINGS_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    trainings = value.get('trainings') if isinstance(value, dict) else None
    return trainings if isinstance(trainings, dict) else {}


def _record(root: Path, action: str) -> dict:
    record = _stored(root).get(action)
    return record if isinstance(record, dict) else {}


def _defaults(training: dict, values: dict) -> dict:
    """The default of every setting, a same_as one following the value of its setting."""
    return {setting['key']: values[setting['same_as']] if setting.get('same_as') else setting['default']
            for setting in training['settings']}


def _resolve(training: dict, stored: dict) -> tuple[dict, dict]:
    """(values, defaults): the stored value of each setting when still valid, else its default."""
    values = {}
    for setting in training['settings']:
        try:
            values[setting['key']] = _number(setting, stored[setting['key']])
        except (KeyError, SettingsError):
            values[setting['key']] = None
    defaults = {}
    for setting in training['settings']:
        if not setting.get('same_as'):
            defaults[setting['key']] = setting['default']
            if values[setting['key']] is None:
                values[setting['key']] = setting['default']
    for setting in training['settings']:
        if setting.get('same_as'):
            defaults[setting['key']] = values[setting['same_as']]
            if values[setting['key']] is None:
                values[setting['key']] = defaults[setting['key']]
    return values, defaults


def values(root: Path, action: str) -> dict:
    """Every setting of a training with its value: the one set (when still valid) or the default."""
    training = _training(action)
    stored = _record(root, action).get('values')
    return _resolve(training, stored if isinstance(stored, dict) else {})[0]


def environment(root: Path, action: str) -> dict:
    """The environment of the next training of this kind: the variables of the settings changed from the default
    (the others stay the script's own); {} for an action without settings."""
    if action not in TRAININGS:
        return {}
    training = TRAININGS[action]
    stored = _record(root, action).get('values')
    current, defaults = _resolve(training, stored if isinstance(stored, dict) else {})
    return {setting['env']: _text(current[setting['key']]) for setting in training['settings']
            if current[setting['key']] != defaults[setting['key']]}


def _text(value: float | int) -> str:
    """A value as its script reads it: 30, 0.003, 1e-05, 3.5."""
    return str(value) if isinstance(value, int) else f'{value:g}'


def view(root: Path) -> dict:
    """For Train: every training with its settings (value, default and bounds), what is fixed, who set them last."""
    trainings = {}
    for action, training in TRAININGS.items():
        record = _record(root, action)
        stored = record.get('values')
        current, defaults = _resolve(training, stored if isinstance(stored, dict) else {})
        trainings[action] = {
            'label': training['label'],
            'settings': [{**{key: value for key, value in setting.items() if key not in ('env', 'same_as')},
                          'default': defaults[setting['key']], 'value': current[setting['key']]}
                         for setting in training['settings']],
            'fixed': [dict(item) for item in training['fixed']],
            'by': record.get('by'), 'at': record.get('at')}
    return {'trainings': trainings}


def save(root: Path, body: dict, who: str) -> dict:
    """{"training": action, "values": {key: number}}: the settings of one training changed (a key left out keeps its
    value; null: back to its default). Only the values that differ from the default are kept."""
    if not isinstance(body, dict) or set(body) != {'training', 'values'} or not isinstance(body['values'], dict):
        raise SettingsError('Send {"training": action, "values": {setting: number}}.')
    action = body['training']
    training = _training(action)
    stored = _record(root, action).get('values')
    current = dict(_resolve(training, stored if isinstance(stored, dict) else {})[0])
    # A same_as setting left at its default keeps following its setting.
    following = {setting['key'] for setting in training['settings'] if setting.get('same_as')
                 and current[setting['key']] == current[setting['same_as']]}
    for key, value in body['values'].items():
        setting = _setting(training, key)
        if setting is None:
            words = 'scene network' if action == 'train_road_type' else training['label'].lower()
            raise SettingsError(f'{key} is not a setting of the {words}.')
        if value is None:
            if setting.get('same_as'):
                following.add(key)
            else:
                current[key] = setting['default']
            continue
        current[key] = _number(setting, value)
        following.discard(key)
    for setting in training['settings']:
        if setting['key'] in following:
            current[setting['key']] = current[setting['same_as']]
    defaults = _defaults(training, current)
    changed = {key: value for key, value in current.items() if value != defaults[key]}
    trainings = _stored(root)
    record = {'values': dict(sorted(changed.items())), 'by': who,
              'at': datetime.now().astimezone().isoformat(timespec='seconds')}
    trainings = {**{name: item for name, item in trainings.items() if name in TRAININGS}, action: record}
    atomic_write(Path(root) / SETTINGS_FILE,
                 (json.dumps({'trainings': trainings}, indent=2) + '\n').encode('utf-8'))
    return record
