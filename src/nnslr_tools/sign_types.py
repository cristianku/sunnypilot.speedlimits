"""What an "Other sign" box is (``sign_type``): the kinds the review offers, the same ids as the page's
"Kind" select (web/src/review/catalog.ts SIGN_KIND_GROUPS).

Some give a limit without a number (town entry, motorway, expressway), some carry numbers that are no
speed (weight, height, width, minimum speed): the model has to know them apart from the speed signs.
Direction signs say where they point; a green one (a board with places or an arrow: in Switzerland the way to a
motorway or an expressway) is ``motorway_direction`` whatever its arrow, a hint that a motorway is near for the
road-type network. The motorway sign itself is ``motorway_start``: the green square with the motorway pictogram
alone, no places and no arrow. ``motorway_exit`` is the sign at the gore of a motorway or expressway exit (the blue arrow up to the right, "Ausfahrt"): ours when the car takes that exit, another road when it stays on. ``no_vehicles`` is a round ban with a red ring and a white disc, empty or
with pictograms (motor vehicles, motorcycles, lorries, bicycles...): the sign that looks most like a speed limit.
A traffic light is a kind too, one for what is lit in the picture: ``traffic_light_red`` (red, also red with amber),
``traffic_light_amber``, ``traffic_light_green``, ``traffic_light_off`` (no lamp lit). One box around the whole light of
our lane. A flashing amber is amber on one frame and off on the next: each frame says what it shows, and the brain
learns the flashing from the sequence.

The review is no longer of signs alone (12 Nov 2026): a road user gets a box too, with a kind of its own (ROAD_USERS):
``road_user_person`` (on foot, also with a pushchair or a wheelchair), ``road_user_bicycle`` and
``road_user_motorcycle`` (one box around the vehicle with its rider), ``road_user_car`` (a van too),
``road_user_truck``, ``road_user_bus``, ``road_user_tram``, ``road_user_other`` (an animal, a tractor, a scooter). One
box around the whole of it, as much as is seen. The classifier learns them as it learns the other kinds
(``other_sign:road_user_car``) and the brain gets them with every other kind it reads. The cutter does not learn them
yet: until it does, their boxes are ignore boxes of its training (training.prepare_detector_training), neither a sign
to find nor background, so the frames reviewed before, where no road user has a box, do not teach it the opposite.
Only a box of family ``other_sign`` has a kind; ``None`` means not said (older reviews, or not chosen).

A box of family ``not_a_sign`` may say what it is instead (NOT_A_SIGN_TYPES): ``on_vehicle``, a sign-like
disc or panel on a vehicle, such as the top-speed sticker on the back of a truck. It stays "not a sign" for
the training; the kind keeps these cases countable apart.

An "Other sign" may also be painted on the road (``painted``: true), such as the children's triangle with
"SCHULE" in front of a school. Its kind stays what the sign means; the flag keeps these cases countable apart.
"""
from __future__ import annotations

import re
from typing import Any

SIGN_TYPES = (
    # Danger
    'pedestrian_crossing', 'children', 'cyclists', 'traffic_queue', 'roadworks', 'danger_other',
    # Priority
    'give_way', 'stop', 'priority_to_right', 'main_road', 'main_road_end',
    # Bans and other numbers
    'no_overtaking', 'no_overtaking_end', 'no_entry', 'no_vehicles', 'minimum_speed', 'weight_limit',
    'height_limit', 'width_limit',
    # Mandatory (round, blue)
    'keep_right', 'keep_left', 'ahead_only',
    # Roads and places
    'town_entry', 'town_exit', 'motorway_start', 'motorway_end', 'expressway_start', 'expressway_end',
    'roundabout', 'one_way',
    # Directions
    'direction_left', 'direction_right', 'direction_straight', 'direction_several', 'motorway_direction',
    'motorway_exit',
    'other',
    # Traffic lights, by what is lit (added 12 Nov 2026, after 'other': the kinds before keep their place)
    'traffic_light_red', 'traffic_light_amber', 'traffic_light_green', 'traffic_light_off',
    # Road users (added 12 Nov 2026): ROAD_USERS
    'road_user_person', 'road_user_bicycle', 'road_user_motorcycle', 'road_user_car', 'road_user_truck',
    'road_user_bus', 'road_user_tram', 'road_user_other',
)
ROAD_USER_PREFIX = 'road_user_'
# The kinds that are no sign: who is on the road.
ROAD_USERS = tuple(kind for kind in SIGN_TYPES if kind.startswith(ROAD_USER_PREFIX))
# What a "not a sign" box is, when said.
NOT_A_SIGN_TYPES = ('on_vehicle',)
# The kind of a classifier's class for a speed limit with a plate of hours under it (maximum_speed_hours:30): one box
# around disc and plate, a sign of its own for the classifier. A review row stays family maximum_speed with
# supplementary_panel "time"; this name lives only in the classes (the car's vision.MAXIMUM_SPEED_HOURS).
HOURS_CLASS = 'maximum_speed_hours'
# The same for a limit with a plate that says it holds only on a wet road (maximum_speed_wet:80, the German "bei Nässe" or
# its cloud with rain): review rows have supplementary_panel "wet"; the car's vision.MAXIMUM_SPEED_WET.
WET_CLASS = 'maximum_speed_wet'
# The class of a number with a plate, and the panel it means.
PLATE_CLASSES = {HOURS_CLASS: 'time', WET_CLASS: 'wet'}
# The hours of a plate when the reviewer read them (plate_hours, "22-6": from and to, whole hours 0-23, no leading zero):
# one spelling, because they end the class (maximum_speed_hours:30:22-6). A plate with days, minutes or two ranges, or
# one too small to read, says none: its class is the number alone (maximum_speed_hours:30).
PLATE_HOURS = re.compile(r'([0-9]|1[0-9]|2[0-3])-([0-9]|1[0-9]|2[0-3])')


def is_road_user(row: dict) -> bool:
    """Whether a review row is a road user: family other_sign with a kind of ROAD_USERS."""
    return row.get('sign_family') == 'other_sign' and row.get('sign_type') in ROAD_USERS


def class_parts(label: Any) -> tuple[str, str, str | None]:
    """(family, its value, panel) of a classifier class: "maximum_speed_hours:30" and "maximum_speed_hours:30:22-6" are
    ("maximum_speed", "30", "time"), "maximum_speed_wet:80" is ("maximum_speed", "80", "wet"), any other class has no panel
    ("maximum_speed:30" is ("maximum_speed", "30", None)). The hours: class_hours."""
    family, _, value = str(label).partition(':')
    panel = PLATE_CLASSES.get(family)
    return ('maximum_speed', value.partition(':')[0], panel) if panel else (family, value, None)


def class_sign_type(label: Any) -> str | None:
    """The kind of an Other sign a classifier class names ("roundabout" for "other_sign:roundabout"), the sign_type of
    its review box; None for any other class or a kind the review does not know."""
    family, value, _ = class_parts(label)
    return value if family == 'other_sign' and value in SIGN_TYPES else None


def with_sign_type(detection: dict) -> dict:
    """A proposal of Other sign with the kind its class names (model_label) when it has none: Prepare's reports before
    14 Nov 2026 kept only "Other sign" of "other_sign:roundabout", "other_sign:main_road"... The same detection."""
    if detection.get('family') == 'other_sign' and not detection.get('sign_type'):
        kind = class_sign_type(detection.get('model_label'))
        if kind:
            detection['sign_type'] = kind
    return detection


def class_hours(label: Any) -> str | None:
    """The hours a classifier class says: "22-6" for "maximum_speed_hours:30:22-6", None for any other class."""
    family, _, value = str(label).partition(':')
    return (value.partition(':')[2] or None) if family == HOURS_CLASS else None


def without_hours(label: str) -> str:
    """A class without the hours it says: "maximum_speed_hours:30:22-6" is "maximum_speed_hours:30"."""
    return label.rpartition(':')[0] if class_hours(label) else label


def holds_plate(row: dict) -> bool:
    """Whether a box takes in a plate as well as its sign (the review's Panel below: Hours or Wet)."""
    return row.get('supplementary_panel') in PLATE_CLASSES.values()


def holds_hours(row: dict) -> bool:
    """Whether the plate of a box is one of hours (the review's Panel below: Hours)."""
    return row.get('supplementary_panel') == 'time'


def plate_hours_problem(detection: dict) -> str | None:
    """Why the plate_hours of a box are not allowed, or None when they are."""
    value = detection.get('plate_hours')
    if value is None:
        return None
    if detection.get('family') != 'maximum_speed' or not holds_hours(detection):
        return 'plate_hours is only for boxes of family maximum_speed with supplementary_panel time'
    read = PLATE_HOURS.fullmatch(value) if isinstance(value, str) else None
    if read is None or read[1] == read[2]:
        return 'plate_hours must be from-to in whole hours from 0 to 23, such as 22-6, or null'
    return None


def sign_type_problem(detection: dict) -> str | None:
    """Why the sign_type of a box is not allowed, or None when it is."""
    value = detection.get('sign_type')
    if value is None:
        return None
    if value in NOT_A_SIGN_TYPES:
        return None if detection.get('family') == 'not_a_sign' else f'sign_type {value} is only for family not_a_sign'
    if value not in SIGN_TYPES:
        return 'sign_type must be one of: ' + ', '.join(SIGN_TYPES + NOT_A_SIGN_TYPES) + ', or null'
    if detection.get('family') != 'other_sign':
        return 'sign_type is only for boxes of family other_sign (or on_vehicle for not_a_sign)'
    return None


def painted_problem(detection: dict) -> str | None:
    """Why the painted flag of a box is not allowed, or None when it is."""
    value = detection.get('painted')
    if value is None or value is False:
        return None
    if value is not True:
        return 'painted must be true, false or null'
    if detection.get('family') != 'other_sign':
        return 'painted is only for boxes of family other_sign'
    return None
