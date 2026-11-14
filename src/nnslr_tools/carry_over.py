"""Apply previous: the boxes confirmed on the previous frame, carried onto the next one.

Frames are one second apart, so on a road the same signs are usually still there, bigger and further
from the centre of the view. For every box confirmed on the previous frame:

- when a box of this frame is the same sign (it moved and grew as the car got closer), that box keeps
  its position and takes the labels of the previous one (family, value, road, panel...);
- when no box of this frame is that sign, a new box is made where the sign should be now: moved like
  the nearest sign that was found, or at its old position when none was. Its position is a guess, so
  it carries ``check_position: true`` and no save accepts it until the reviewer has looked at it
  (moved it onto the sign, or removed it) and the flag is gone.

A box of this frame of another shape (MAX_SHAPE_CHANGE) is never the same sign. A box of the disc alone is the sign of a
box around a disc and its plate of hours (Panel below: Hours), but the plate would be lost (PLATE_SHAPE): it is taken down
to that shape, and where the plate ends is a guess, so it carries ``check_position`` too. Boxes of this frame that match nothing stay as they are. Boxes the reviewer already changed keep their labels: changed from ``proposals``, which
holds the model's proposals and, on a frame confirmed before, its saved boxes (a drive reviewed again takes the
labels of the frame before, as one never reviewed does). Signs only grow and move outwards from the point the road runs to (``FOCUS``), which is
what the matching expects; it was measured on 359 pairs of confirmed boxes one frame apart.

The page, the REST API and the MCP server all call :func:`carry_over` through
``ReviewService.apply_previous``; nothing is saved here.
"""
from __future__ import annotations

import copy
import math

from nnslr_tools.sign_types import holds_hours, holds_plate

# Where the road runs to, as a fraction of width and height: signs move away from it as they come
# closer (median of the confirmed pairs: x 0.52, y 0.53).
FOCUS = (0.52, 0.53)
# How much a sign may grow from one frame to the next (the pairs: 1.0 to 2.0, a few up to 3).
MIN_SCALE = 0.75
MAX_SCALE = 3.5
# A sign keeps its shape as it comes closer: a box of this frame whose width/height ratio is more than this many times
# the old one's (or less than its inverse) is another sign. On 815 pairs of confirmed boxes one frame apart (26 October
# 2026, boxes cut by the edge left out) the ratio changed by 1.04 times at the median, 1.44 at the 99th percentile,
# once by 2.35; a wide direction sign "matched" to the tall pedestrian sign beside it changed by 3.3. A box touching the
# left or right edge (EDGE pixels) is not checked: the edge cuts it.
MAX_SHAPE_CHANGE = 2.0
EDGE = 2
# A box around a disc and its plate is about 1.6 times taller than the disc alone, under MAX_SHAPE_CHANGE: a box of this
# frame flatter than the old one by more than this holds the disc only, and the plate would be lost with Hours kept.
# Such a box has no shape of its own to match on: it is measured as if it had the old one's (_with_plate).
PLATE_SHAPE = 1.3
# How far the whole view may slide in a turn, as a fraction of the width, and what a slide costs: a slide
# of TURN_COST * width costs as much as one pair at the edge of its tolerance.
MAX_TURN = 0.35
TURN_COST = 0.3
# A turn slides the view sideways; up or down it moves little in a second (a bump, a slope): at most this fraction of
# the height. On 1,110 pairs of confirmed boxes one frame apart (26 October 2026) the vertical shift was 5 px at the
# median, 64 at the 99th percentile, over 76 px (0.1 of 760) for 5; a danger sign "matched" to a proposal 470 px away
# needed 110 px up.
MAX_TURN_UP = 0.1
# The labels a box takes from the previous frame. Visibility and occlusion belong to this picture.
CARRIED_KEYS = ('family', 'value_kph', 'applicability', 'supplementary_panel', 'temporary', 'variable_display',
                'sign_type', 'painted', 'notes', 'sign_value', 'sign_unit', 'plate_hours')
# The labels that tell whether the reviewer already changed a box (the review page's fields).
LABEL_KEYS = ('family', 'value_kph', 'applicability', 'supplementary_panel', 'visibility', 'occlusion',
              'temporary', 'variable_display', 'sign_type', 'painted', 'sign_value', 'sign_unit', 'plate_hours')
VALUE_FAMILIES = ('maximum_speed', 'cancellation', 'zone', 'variable_display', 'road_marking_candidate')
CHECK_FIELD = 'check_position'
FROM_FIELD = 'carried_from'
# What last set a box's position (review_db.BOX_SOURCES, #80): a box placed here, or taken down over a plate, is the
# carry's; a box of this frame that takes the labels keeps its position, so it keeps its source too.
SOURCE_FIELD = 'box_source'
CARRIED = 'carried'
# A carried box mostly out of the picture is dropped: the car has passed the sign.
MIN_INSIDE = 0.3
MIN_SIDE = 4
# A box placed where a sign should be takes a box of this frame covering this much of it (overlap of the
# smaller box); a leftover model box this much over a carried one is a duplicate of it.
SNAP = 0.5
DUPLICATE = 0.6


def _box(item) -> list[float] | None:
    box = item.get('bbox_xyxy') if isinstance(item, dict) else None
    if not isinstance(box, list) or len(box) != 4:
        return None
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return [float(value) for value in box]


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _size(box: list[float]) -> float:
    return math.sqrt((box[2] - box[0]) * (box[3] - box[1]))


def _pair(old: list[float], new: list[float], focus: tuple[float, float]):
    """(scale, shift, moved) of ``new`` seen as the sign ``old`` one frame later, or None when it cannot
    be: a sign does not shrink nor grow beyond MAX_SCALE in a second. ``shift`` is how far ``new`` is from
    where the growth alone puts it: the turn of the car (the whole view slides) plus the error."""
    scale = _size(new) / _size(old)
    if scale < MIN_SCALE or scale > MAX_SCALE:
        return None
    old_center = _center(old)
    new_center = _center(new)
    expected = (focus[0] + scale * (old_center[0] - focus[0]), focus[1] + scale * (old_center[1] - focus[1]))
    shift = (new_center[0] - expected[0], new_center[1] - expected[1])
    return scale, shift, math.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1])


def _same_shape(old: list[float], new: list[float], width: float) -> bool:
    """Whether ``new`` has about the shape of ``old`` (MAX_SHAPE_CHANGE), or is cut by the left or right edge."""
    if width and (new[0] <= EDGE or new[2] >= width - EDGE):
        return True
    change = ((old[2] - old[0]) / (old[3] - old[1])) / ((new[2] - new[0]) / (new[3] - new[1]))
    return max(change, 1 / change) <= MAX_SHAPE_CHANGE


def _with_plate(old: dict, new: list[float]) -> list[float] | None:
    """``new``, or when it holds only the disc of ``old``, a box with a plate (PLATE_SHAPE): taken down to the
    shape of ``old``, its top and its width staying. None when it is flatter than a disc too: that is no part of ``old``."""
    box = _box(old)
    tall = (box[3] - box[1]) / (box[2] - box[0])
    shape = (new[3] - new[1]) / (new[2] - new[0])
    if not holds_plate(old) or tall <= PLATE_SHAPE * shape:
        return new
    return None if PLATE_SHAPE * shape < 1 else [new[0], new[1], new[2], new[1] + (new[2] - new[0]) * tall]


def _cost(pair, new: list[float], turn: tuple[float, float], unit: float) -> float | None:
    """How far a pair is from the move expected with the view slid by ``turn``: <= 1, or None."""
    scale, shift, moved = pair
    # The focus is a guess (slopes, lane changes): the further a sign moved, the larger the error allowed.
    tolerance = 0.8 * _size(new) + 0.25 * moved + 10 * unit
    cost = math.hypot(shift[0] - turn[0], shift[1] - turn[1]) / tolerance
    if cost > 1:
        return None
    return cost


def _assign(olds: list, news: list, candidates: dict, turn: tuple[float, float], unit: float):
    """Greedy pairs, cheapest first: ({old index: (new index, scale, shift)}, total cost)."""
    pairs = []
    for (i, j), pair in candidates.items():
        cost = _cost(pair, _box(news[j]), turn, unit)
        if cost is not None:
            pairs.append((cost * _same_sign_bonus(olds[i], news[j]), i, j))
    pairs.sort()
    taken_old = {}
    taken_new = set()
    total = 0.0
    for cost, i, j in pairs:
        if i in taken_old or j in taken_new:
            continue
        scale, shift, moved = candidates[(i, j)]
        taken_old[i] = (j, scale, shift)
        taken_new.add(j)
        total = total + cost
    return taken_old, total


def _same_sign_bonus(old: dict, new: dict) -> float:
    """Lower cost when the box of this frame already reads the same sign."""
    factor = 1.0
    if new.get('family') == old.get('family'):
        factor = factor * 0.8
    value = old.get('value_kph')
    if value is not None and new.get('value_kph') == value:
        factor = factor * 0.6
    return factor


def _edited(item: dict, proposals: dict) -> bool:
    """True when the reviewer already changed this box (or drew it): its labels are theirs."""
    original = proposals.get(item.get('detection_id'))
    if original is None:
        return True
    for key in LABEL_KEYS:
        if item.get(key) != original.get(key):
            return True
    return False


def _new_id(taken: set, source_key: str) -> str:
    base = 'previous-' + source_key.replace('/', '-')
    number = 1
    while base + '-' + str(number) in taken:
        number = number + 1
    taken.add(base + '-' + str(number))
    return base + '-' + str(number)


def match_boxes(olds: list, news: list, *, width: float, height: float) -> dict:
    """Which box of ``news`` (this frame) is the same sign as each box of ``olds`` (the frame before):
    {old index: (new index, scale, shift)}. Boxes without a valid bbox_xyxy match nothing.

    Signs grow and move away from FOCUS as the car gets closer; in a turn the whole view slides as well.
    """
    unit = width / 1344 if width else 1.0
    focus = (FOCUS[0] * width, FOCUS[1] * height)
    candidates = {}
    for i, old in enumerate(olds):
        old_box = _box(old)
        if old_box is None:
            continue
        for j, new in enumerate(news):
            new_box = _box(new)
            if new_box is None:
                continue
            seen = _with_plate(old, new_box)
            pair = None if seen is None else _pair(old_box, seen, focus)
            if pair is not None and _same_shape(old_box, seen, width):
                candidates[(i, j)] = pair
    # In a turn the whole view slides: try no slide and the slide each candidate pair implies, keep the one
    # that explains most signs (then the smallest slide). One sign alone can always be explained by its slide:
    # a large one only when nothing else is there, and it still has to have grown like a sign coming closer.
    turns = [(0.0, 0.0)] + [pair[1] for pair in candidates.values()]
    best = None
    for turn in turns:
        slide = math.hypot(turn[0], turn[1])
        if slide > MAX_TURN * width or (height and abs(turn[1]) > MAX_TURN_UP * height):
            continue
        taken, total = _assign(olds, news, candidates, turn, unit)
        score = (len(taken), -(total + slide / (TURN_COST * width)))
        if best is None or score > best[0]:
            best = (score, taken)
    return best[1] if best is not None else {}


def carry_over(previous: list, current: list, *, width: float, height: float, source_key: str,
               proposals: list | None = None, follow=None) -> dict:
    """The boxes of this frame after "Apply previous", and what happened to each previous box.

    ``previous``: the confirmed boxes of the previous frame. ``current``: the boxes of this frame as
    they are now (the model proposals, or the reviewer's working copy). ``proposals``: the model
    proposals of this frame, to tell boxes the reviewer changed (kept as they are) from the others. ``follow``:
    ``follow(old box, expected box)`` gives where a sign found on no box of this frame is in its picture, or None
    (:func:`nnslr_tools.carry_track.follow` on the two pictures); a box placed there needs no check.
    Returns {'detections', 'matched', 'estimated', 'followed', 'dropped', 'kept'}; a carried box has
    ``carried_from`` (the previous frame key) and, when its position is a guess, ``check_position``.
    """
    unit = width / 1344 if width else 1.0
    focus = (FOCUS[0] * width, FOCUS[1] * height)
    by_id = {}
    for item in proposals or []:
        if isinstance(item, dict) and isinstance(item.get('detection_id'), str):
            by_id[item['detection_id']] = item
    result = [copy.deepcopy(item) for item in current if isinstance(item, dict)]
    olds = [item for item in previous if isinstance(item, dict) and _box(item) is not None]
    taken_old = match_boxes(olds, result, width=width, height=height)

    matched = 0
    estimated = 0
    followed = 0
    kept = 0
    for i, (j, scale, shift) in taken_old.items():
        target = result[j]
        if _edited(target, by_id):
            kept = kept + 1
            continue
        if _carry(target, olds[i], source_key, height):
            matched = matched + 1
        else:
            estimated = estimated + 1

    ids = {item.get('detection_id') for item in result if isinstance(item.get('detection_id'), str)}
    ids.update(by_id)
    taken_new = {j for j, scale, shift in taken_old.values()}
    dropped = 0
    for i, old in enumerate(olds):
        # A box the reviewer marked "not a sign" is only carried onto a box of this frame; one on a vehicle (a
        # truck's sticker) is placed like a sign: the truck ahead is still there.
        if i in taken_old or (old.get('family') == 'not_a_sign' and old.get('sign_type') != 'on_vehicle'):
            continue
        box = box_before = _box(old)
        if taken_old:
            # Moved like the nearest sign that was found (same gantry, same distance: same move).
            center = _center(box)
            nearest = min(taken_old, key=lambda k: math.dist(center, _center(_box(olds[k]))))
            j, scale, shift = taken_old[nearest]
            moved = (focus[0] + scale * (center[0] - focus[0]) + shift[0],
                     focus[1] + scale * (center[1] - focus[1]) + shift[1])
            half_w = (box[2] - box[0]) * scale / 2
            half_h = (box[3] - box[1]) * scale / 2
            box = [moved[0] - half_w, moved[1] - half_h, moved[0] + half_w, moved[1] + half_h]
        area = (box[2] - box[0]) * (box[3] - box[1])
        clipped = [max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3])]
        inside = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if clipped[2] - clipped[0] < MIN_SIDE or clipped[3] - clipped[1] < MIN_SIDE or inside < MIN_INSIDE * area:
            dropped = dropped + 1
            continue
        # A box of this frame where the sign should be (the growth was off, a proposal smaller than the
        # reviewer's box): that is the sign, at a position the model found.
        snap = _covering(clipped, result, taken_new, old)
        if snap is not None:
            taken_new.add(snap)
            if _edited(result[snap], by_id):
                kept = kept + 1
            elif _carry(result[snap], old, source_key, height):
                matched = matched + 1
            else:
                estimated = estimated + 1
            continue
        # Its own pixels from the frame before, found in this picture: the sign is there.
        found = follow(box_before, clipped) if follow is not None else None
        if found is not None:
            clipped = found
        item = {'bbox_xyxy': [round(value, 1) for value in clipped], 'score': None, 'model_label': None,
                'class_id': None, 'review_state': 'pending', 'visibility': 'unknown', 'occlusion': 'unknown'}
        for key in CARRIED_KEYS:
            if key in old:
                item[key] = copy.deepcopy(old[key])
        item['value_state'] = 'unknown' if item.get('value_kph') is None else 'reviewed'
        item['detection_id'] = _new_id(ids, source_key)
        item[FROM_FIELD] = source_key
        item[SOURCE_FIELD] = CARRIED
        result.append(item)
        if found is not None:
            followed = followed + 1
        else:
            item[CHECK_FIELD] = True
            estimated = estimated + 1

    # A model box left over on a sign that already has its carried box is the same sign twice: the
    # reviewer would delete it (a proposal left out is recorded as rejected).
    duplicates = []
    for j, item in enumerate(result):
        if j in taken_new or item.get(FROM_FIELD) or _edited(item, by_id) or _box(item) is None:
            continue
        for other in result:
            if other.get(FROM_FIELD) and _box(other) is not None and _overlap(_box(item), _box(other)) >= DUPLICATE:
                duplicates.append(j)
                break
    for j in reversed(duplicates):
        del result[j]
    return {'detections': result, 'matched': matched, 'estimated': estimated, 'followed': followed, 'dropped': dropped,
            'kept': kept, 'duplicates': len(duplicates)}


def _carry(target: dict, old: dict, source_key: str, height: float) -> bool:
    """The labels of ``old`` on a box of this frame: True when the box stays where it is, False when it held the disc
    alone of ``old`` and went down over the plate (_with_plate), a guess to check."""
    _take_labels(target, old, source_key)
    box = _box(target)
    tall = _with_plate(old, box)
    if tall == box:
        return True
    target['bbox_xyxy'] = box[:3] + [round(min(float(height), tall[3]), 1)]
    target[CHECK_FIELD] = True
    target[SOURCE_FIELD] = CARRIED
    return False


def _take_labels(target: dict, old: dict, source_key: str) -> None:
    """The labels of the previous box on a box of this frame, whose position stays."""
    # Closer, a number unreadable before may be readable now: a value the model reads here is kept.
    read_here = target.get('value_kph')
    keep_reading = (old.get('value_kph') is None and isinstance(read_here, int) and not isinstance(read_here, bool)
                    and old.get('family') in VALUE_FAMILIES + ('unreadable',)
                    and target.get('family') in VALUE_FAMILIES)
    reading = (target.get('family'), read_here)
    # So may a plate: far away the sign read as a plain number. A plate read here is kept, with the box around disc and plate.
    plate_here = target.get('supplementary_panel') if holds_plate(target) else None
    original_reading = {k: target[k] for k in ('sign_value','sign_unit') if k in target}
    for key in CARRIED_KEYS:
        if key in old:
            target[key] = copy.deepcopy(old[key])
        elif key in ('sign_value','sign_unit'):
            target.pop(key, None)
    if keep_reading:
        if old.get('family') == 'unreadable':
            target['family'] = reading[0]
        target['value_kph'] = reading[1]
        target.update(original_reading)
    if plate_here and target.get('family') == 'maximum_speed':
        target['supplementary_panel'] = plate_here
    # The kind read here on a proposal (an Other sign's, since 14 Nov 2026) goes when the previous box says another
    # family and names no kind: a speed limit never keeps "give way". An Other sign carried without a kind keeps the
    # kind read here, as a number read here is kept.
    if target.get('family') != 'other_sign' and 'sign_type' not in old:
        target.pop('sign_type', None)
    if target.get('family') != 'maximum_speed' or not holds_hours(target):
        target.pop('plate_hours', None)
    target['value_state'] = 'unknown' if target.get('value_kph') is None else 'reviewed'
    target.pop(CHECK_FIELD, None)
    target[FROM_FIELD] = source_key


def _overlap(a: list[float], b: list[float]) -> float:
    """Shared area over the area of the smaller box (the page's overlap): 1 when one holds the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


def _covering(box: list[float], boxes: list, taken: set, old: dict) -> int | None:
    """The free box of this frame that is where ``box`` was placed for ``old``, of a similar size, or None."""
    best = None
    for j, item in enumerate(boxes):
        other = _box(item)
        if j in taken or other is None or _with_plate(old, other) is None:
            continue
        ratio = _size(other) / _size(box)
        share = _overlap(box, other)
        if share >= SNAP and 0.5 <= ratio <= 2 and (best is None or share > best[0]):
            best = (share, j)
    return None if best is None else best[1]


def unchecked_boxes(detections) -> list[int]:
    """1-based numbers of the boxes still waiting for their position to be checked."""
    numbers = []
    if not isinstance(detections, list):
        return numbers
    for index, item in enumerate(detections):
        if isinstance(item, dict) and item.get(CHECK_FIELD):
            numbers.append(index + 1)
    return numbers
