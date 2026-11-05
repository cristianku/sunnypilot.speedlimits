"""What an "Other sign" box is (``sign_type``): the kinds the review offers, the same ids as the page's
"Kind" select (web/src/review/catalog.ts SIGN_KIND_GROUPS).

Some give a limit without a number (town entry, motorway, expressway), some carry numbers that are no
speed (weight, height, width, minimum speed): the model has to know them apart from the speed signs.
Direction signs say where they point; a green one (a board with places or an arrow: in Switzerland the way to a
motorway or an expressway) is ``motorway_direction`` whatever its arrow, a hint that a motorway is near for the
road-type network. The motorway sign itself is ``motorway_start``: the green square with the motorway pictogram
alone, no places and no arrow. ``no_vehicles`` is a round ban with a red ring and a white disc, empty or
with pictograms (motor vehicles, motorcycles, lorries, bicycles...): the sign that looks most like a speed limit.
Only a box of family ``other_sign`` has a kind; ``None`` means not said (older reviews, or not chosen).

A box of family ``not_a_sign`` may say what it is instead (NOT_A_SIGN_TYPES): ``on_vehicle``, a sign-like
disc or panel on a vehicle, such as the top-speed sticker on the back of a truck. It stays "not a sign" for
the training; the kind keeps these cases countable apart.

An "Other sign" may also be painted on the road (``painted``: true), such as the children's triangle with
"SCHULE" in front of a school. Its kind stays what the sign means; the flag keeps these cases countable apart.
"""
from __future__ import annotations

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
    'other',
)
# What a "not a sign" box is, when said.
NOT_A_SIGN_TYPES = ('on_vehicle',)
# The kind of a classifier's class for a speed limit with a plate of hours under it (maximum_speed_hours:30): one box
# around disc and plate, a sign of its own for the classifier. A review row stays family maximum_speed with
# supplementary_panel "time"; this name lives only in the classes (the car's vision.MAXIMUM_SPEED_HOURS).
HOURS_CLASS = 'maximum_speed_hours'


def class_parts(label: Any) -> tuple[str, str, str | None]:
    """(family, what follows the ":", panel) of a classifier class: "maximum_speed_hours:30" is ("maximum_speed", "30",
    "time"), any other class has no panel ("maximum_speed:30" is ("maximum_speed", "30", None))."""
    family, _, value = str(label).partition(':')
    return ('maximum_speed', value, 'time') if family == HOURS_CLASS else (family, value, None)


def holds_plate(row: dict) -> bool:
    """Whether a box takes in a plate of hours as well as its sign (the review's Panel below: Hours)."""
    return row.get('supplementary_panel') == 'time'


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
