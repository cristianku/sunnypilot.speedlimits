"""The settings of the comma's vision limit (sunnypilot speed_limits/vision.py): how sure a sign or a road type must be
and how long it must be seen, set in Models › On the car and sent with the networks (the "settings" of
speed_limits.json, car_release.manifest_of). The comma takes each one it knows within its bounds and keeps its own
default for the others (vision.apply_settings): a setting of this list is a constant of vision.py, same default.

Kept in ``models/car-settings.json`` of the data root: {"values": {key: number}, "by", "at"}, only the values changed
from the default. They reach the car with the next Publish (or Try on the comma), never on their own.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from nnslr_tools.annotations import atomic_write

SETTINGS_FILE = Path('models') / 'car-settings.json'
# key (the name in the manifest), the constant of vision.py, label, what it does, default, min, max, step, unit, group.
SETTINGS = (
    {'key': 'min_confidence', 'constant': 'MIN_CONFIDENCE', 'label': 'A sign read when the classifier is',
     'help': 'How sure the classifier must be of a sign for its number to count.', 'default': 0.7, 'min': 0.3,
     'max': 0.99, 'step': 0.01, 'unit': 'sure', 'group': 'signs'},
    {'key': 'confirm_frames', 'constant': 'CONFIRM_FRAMES', 'label': 'Seen in',
     'help': 'A number becomes the limit when it is read in this many pictures (5 a second)…', 'default': 2, 'min': 1,
     'max': 10, 'step': 1, 'unit': 'pictures', 'group': 'signs'},
    {'key': 'confirm_seconds', 'constant': 'CONFIRM_SECONDS', 'label': 'Within',
     'help': '…within this many seconds.', 'default': 1.5, 'min': 0.2, 'max': 5.0, 'step': 0.1, 'unit': 's',
     'group': 'signs'},
    {'key': 'road_type_debounce', 'constant': 'ROAD_TYPE_DEBOUNCE', 'label': 'A road type counts after',
     'help': 'The scene network answers once a second; its answer counts once it is the same this many times '
             'in a row.', 'default': 3, 'min': 1, 'max': 10, 'step': 1, 'unit': 'answers', 'group': 'road'},
    {'key': 'road_type_seconds', 'constant': 'ROAD_TYPE_SECONDS', 'label': 'A new road type gives its limit after',
     'help': 'Where no sign holds the limit, a new road type (town, rural, a 30 zone…) gives its general limit only '
             'once seen this long.', 'default': 10.0,
     'min': 2.0, 'max': 60.0, 'step': 1.0, 'unit': 's', 'group': 'road'},
    {'key': 'zone_confidence', 'constant': 'ZONE_CONFIDENCE', 'label': 'A 30 zone when the network is',
     'help': 'The scene network says "30 zone" only this sure, else town: with Map Data + Vision or Vision Only '
             'the car takes vision first, and a wrong 30 on a 50 street brakes for nothing.', 'default': 0.9, 'min': 0.5,
     'max': 0.999, 'step': 0.01, 'unit': 'sure', 'group': 'road'},
    {'key': 'parking_seconds', 'constant': 'PARKING_SECONDS', 'label': 'A car park after',
     'help': 'A car park gives its limit once the network has seen it this long, the car slow.', 'default': 5.0,
     'min': 1.0, 'max': 30.0, 'step': 1.0, 'unit': 's', 'group': 'road'},
    {'key': 'parking_exit_seconds', 'constant': 'PARKING_EXIT_SECONDS', 'label': 'Out of a car park after',
     'help': 'The car park\'s limit ends once the network has seen a road this long (after its answers agree, about '
             '3 s): the car speeds up to the street\'s limit.', 'default': 3.0, 'min': 0.0, 'max': 30.0, 'step': 1.0,
     'unit': 's', 'group': 'road'},
    {'key': 'parking_limit', 'constant': 'PARKING_LIMIT', 'label': 'Limit in a car park in the open',
     'help': 'The limit a car park in the open gives (a sign there counts only when lower). A scene network trained '
             'before the class garage cannot tell the two apart: its car parks take the garage\'s limit.',
     'default': 20.0, 'min': 5.0, 'max': 30.0, 'step': 1.0, 'unit': 'km/h', 'group': 'road'},
    {'key': 'garage_limit', 'constant': 'GARAGE_LIMIT', 'label': 'Limit in a garage',
     'help': 'The limit an underground or indoor car park gives (a sign there counts only when lower).',
     'default': 10.0, 'min': 5.0, 'max': 30.0, 'step': 1.0, 'unit': 'km/h', 'group': 'road'},
    {'key': 'ramp_seconds', 'constant': 'RAMP_SECONDS', 'label': 'A ramp\'s number within',
     'help': 'A number read this soon after a motorway or expressway sign is the ramp\'s: it ends where the ramp '
             'merges.', 'default': 45.0, 'min': 5.0, 'max': 120.0, 'step': 1.0, 'unit': 's', 'group': 'road'},
    {'key': 'merge_seconds', 'constant': 'MERGE_SECONDS', 'label': 'Merged after',
     'help': 'The ramp\'s limit ends once the network sees the main carriageway this long.', 'default': 8.0,
     'min': 2.0, 'max': 30.0, 'step': 1.0, 'unit': 's', 'group': 'road'},
    {'key': 'carriageway_lag', 'constant': 'CARRIAGEWAY_LAG', 'label': 'A number off the motorway ends if read',
     'help': 'A number read off the motorway (its end sign or the ramp missed) ends once the network has seen the '
             'motorway or expressway as long as a new road type needs, if it was read at least this long before the '
             'network saw it: read later, it may be the motorway\'s own (the network lags a few seconds).',
     'default': 5.0, 'min': 0.0, 'max': 30.0, 'step': 1.0, 'unit': 's before it', 'group': 'road'},
    {'key': 'distance_lead_seconds', 'constant': 'DISTANCE_LEAD_SECONDS', 'label': 'The distance shown shorter by',
     'help': 'The frame is old when read, the networks take their time and each value stays 0.2 s on the screen: the '
             'distance shown is shorter by this many seconds of the speed. Measured in Data › Sign distance (28 signs, '
             '3 Nov 2026).', 'default': 0.34, 'min': 0.0, 'max': 1.5, 'step': 0.01, 'unit': 's of the speed',
     'group': 'distance'},
    {'key': 'distance_lead_metres', 'constant': 'DISTANCE_LEAD_METRES', 'label': '…and by',
     'help': 'The zero is where the sign leaves the picture, a few metres before the car passes it (the same '
             'measurement).', 'default': 1.8, 'min': 0.0, 'max': 15.0, 'step': 0.1, 'unit': 'm', 'group': 'distance'},
    {'key': 'variable_display_diameter', 'constant': 'VARIABLE_DISPLAY_DIAMETER', 'label': 'A gantry\'s LED limit is',
     'help': 'Its distance comes from its size on the camera: the median of 10 approaches on our drives, from how its '
             'box grew with the metres driven.', 'default': 1.4, 'min': 0.6, 'max': 3.0, 'step': 0.05,
     'unit': 'm across', 'group': 'distance'},
)
BY_KEY = {setting['key']: setting for setting in SETTINGS}
DEFAULTS = {setting['key']: setting['default'] for setting in SETTINGS}


class SettingsError(ValueError):
    """A value that is not a setting, not a number or out of its bounds."""


def _number(setting: dict, value) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SettingsError(f'{setting["key"]} must be a number.')
    if not setting['min'] <= value <= setting['max']:
        raise SettingsError(f'{setting["key"]} goes from {setting["min"]:g} to {setting["max"]:g}.')
    return int(value) if isinstance(setting['default'], int) else float(value)


def _stored(root: Path) -> dict:
    path = Path(root) / SETTINGS_FILE
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def values(root: Path) -> dict:
    """Every setting with its value: the one set (when still valid) or the default. What the manifest carries."""
    stored = _stored(root).get('values')
    stored = stored if isinstance(stored, dict) else {}
    result = {}
    for setting in SETTINGS:
        try:
            result[setting['key']] = _number(setting, stored[setting['key']])
        except (KeyError, SettingsError):
            result[setting['key']] = setting['default']
    return result


def view(root: Path, published: dict | None = None) -> dict:
    """For Models › On the car: every setting with its value, default and bounds, who set them last, and the values
    on the car (published: the "settings" of the manifest in the models repository; None when unknown)."""
    stored = _stored(root)
    current = values(root)
    on_car = published if isinstance(published, dict) else None
    return {'settings': [{**{key: value for key, value in setting.items() if key != 'constant'},
                          'value': current[setting['key']],
                          'on_car': (on_car or {}).get(setting['key'], setting['default']) if on_car is not None
                          else None} for setting in SETTINGS],
            'by': stored.get('by'), 'at': stored.get('at'),
            'sent': on_car is not None and all(on_car.get(key, BY_KEY[key]['default']) == value
                                               for key, value in current.items())}


def save(root: Path, body: dict, who: str) -> dict:
    """{"values": {key: number}}: the settings changed (a key left out keeps its value; null: back to its default)."""
    if not isinstance(body, dict) or set(body) != {'values'} or not isinstance(body['values'], dict):
        raise SettingsError('Send {"values": {setting: number}}.')
    current = {key: value for key, value in values(root).items() if value != BY_KEY[key]['default']}
    for key, value in body['values'].items():
        setting = BY_KEY.get(key)
        if setting is None:
            raise SettingsError(f'{key} is not a setting of the car.')
        if value is None:
            current.pop(key, None)
            continue
        number = _number(setting, value)
        if number == setting['default']:
            current.pop(key, None)
        else:
            current[key] = number
    record = {'values': dict(sorted(current.items())), 'by': who,
              'at': datetime.now().astimezone().isoformat(timespec='seconds')}
    atomic_write(Path(root) / SETTINGS_FILE, (json.dumps(record, indent=2) + '\n').encode('utf-8'))
    return record
