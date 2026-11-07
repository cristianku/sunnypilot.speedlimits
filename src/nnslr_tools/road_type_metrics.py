"""How well the road-type network finds the kind of road and its changes, from its answers on whole sequences.

The truth of every measure is what a person confirmed or corrected in Data › Roads until 7 Nov 2026 (road_type_data.frame_label): the
map's proposals are no truth, their frames are unknown here (label -1, weight 0), so a network is measured only on the
confirmed stretches and a change between two of them only when they are at most GAP_SECONDS apart.

* per class: precision, recall and F1 over the frames of weight 1 (the confirmed ones, every label), and macro F1
  over the classes that have frames;
* transitions: a change of road type with at least STABLE_SECONDS of the old type before it and of the new
  one after it (a gap of frames without a type up to GAP_SECONDS between them). Its delay is the time from the change
  to the first second from which the network answers the new type CONFIRM_SECONDS in a row, looked for from
  STABLE_SECONDS before the change (an early answer is a negative delay) to WINDOW_SECONDS after it; later it is
  missed. Kinds: ramp_to_motorway (the merge), motorway_to_ramp (the exit), town_entry, town_exit, other;
* false switches per hour: changes of the network's answer inside a stretch of one road type, MARGIN_SECONDS away from
  its ends, raw (every argmax change) and debounced (an answer taken only after CONFIRM_SECONDS in a row, like the
  car's memory does);
* after the merge: Cristian's case for the memory. After a ramp the motorway can look like a rural road frame by
  frame; the memory should keep "motorway" because it just saw the ramp. For every merge (a ramp of at least
  MERGE_RAMP_SECONDS, then motorway or expressway after a gap of at most GAP_SECONDS) the accuracy over the first
  AFTER_MERGE_SECONDS of the carriageway (until another type starts), raw and debounced, and what was said instead.

The acceleration lane (merge, road_types.merge_lanes) is the carriageway's first metres: the transitions, the false
switches and the seconds after the merge fold it into the carriageway after it, in the map's labels and in the answers
alike (fold_merge), so that they keep their meaning and a network trained before the class is measured the same way;
its lead, the ramp's last metres, folds back into the ramp, where "ramp" and "merge" are both right. The debounced
measures debounce the answers first and fold them after, as the car holds merge and motorway apart. The per-class F1
keeps the lane apart, and the comma's rule ends the ramp's limit on it (comma_release).

One frame a second: a frame's position in its sequence is its time in seconds. numpy only.
"""
from __future__ import annotations

import numpy as np

STABLE_SECONDS = 10
GAP_SECONDS = 10
WINDOW_SECONDS = 60
CONFIRM_SECONDS = 3
MARGIN_SECONDS = 10
TRANSITION_KINDS = ('ramp_to_motorway', 'motorway_to_ramp', 'town_entry', 'town_exit', 'other')
MERGE_RAMP_SECONDS = 3
AFTER_MERGE_SECONDS = 60


def per_class_f1(truth: np.ndarray, predicted: np.ndarray, mask: np.ndarray, classes: tuple[str, ...]) -> dict:
    """{'classes': {name: {precision, recall, f1, support}}, 'macro_f1', 'accuracy', 'frames'} over the frames of
    mask; a class without support has None values and stays out of the macro F1."""
    truth, predicted, mask = np.asarray(truth), np.asarray(predicted), np.asarray(mask, dtype=bool)
    truth, predicted = truth[mask], predicted[mask]
    found, f1s = {}, []
    for index, name in enumerate(classes):
        support = int((truth == index).sum())
        hit = int(((truth == index) & (predicted == index)).sum())
        said = int((predicted == index).sum())
        if support == 0:
            found[name] = {'precision': None, 'recall': None, 'f1': None, 'support': 0, 'predicted': said}
            continue
        precision = hit / said if said else 0.0
        recall = hit / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        found[name] = {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
                       'support': support, 'predicted': said}
        f1s.append(f1)
    return {'classes': found, 'macro_f1': round(float(np.mean(f1s)), 4) if f1s else None,
            'accuracy': round(float((truth == predicted).mean()), 4) if len(truth) else None, 'frames': int(len(truth))}


def confusion(truth: np.ndarray, predicted: np.ndarray, mask: np.ndarray, classes: int) -> list[list[int]]:
    """Frames of mask by [true class][answer]."""
    truth, predicted, mask = np.asarray(truth), np.asarray(predicted), np.asarray(mask, dtype=bool)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    keep = mask & (truth >= 0)
    np.add.at(matrix, (truth[keep].astype(int), predicted[keep].astype(int)), 1)
    return matrix.tolist()


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(label, start, end exclusive) of every run of one value."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((int(labels[start]), start, i))
            start = i
    return runs


# A 30 zone is in a town: from town into it is no town entry.
TOWNS = ('town', 'zone30')
CAR_PARKS = ('parking', 'garage')
CARRIAGEWAYS = ('motorway', 'expressway')


def fold_labels(labels: np.ndarray, classes: tuple[str, ...], lead: np.ndarray | None = None) -> np.ndarray:
    """The labels with every stretch of acceleration lane (merge) as the carriageway after it (motorway when none
    follows) and its lead (the frames that were the ramp's, road_types' "lead") as the ramp; the same labels when the
    classes have no merge."""
    labels = np.asarray(labels)
    if 'merge' not in classes:
        return labels
    lane, motorway = classes.index('merge'), classes.index('motorway')
    roads = {classes.index(name) for name in CARRIAGEWAYS}
    folded = labels.copy()
    runs = _runs(labels)
    for at, (value, start, end) in enumerate(runs):
        if value == lane:
            after = runs[at + 1][0] if at + 1 < len(runs) else -1
            folded[start:end] = after if after in roads else motorway
    if lead is not None:
        folded[np.asarray(lead, dtype=bool) & (labels == lane)] = classes.index('ramp')
    return folded


def fold_merge(labels: np.ndarray, predicted: np.ndarray, classes: tuple[str, ...],
               lead: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(labels, answers) with the acceleration lane folded into its carriageway and its lead into the ramp
    (fold_labels): an answer "merge" is the label there on a carriageway or the lead, motorway elsewhere."""
    labels, predicted = fold_labels(labels, classes, lead), np.asarray(predicted)
    if 'merge' not in classes:
        return labels, predicted
    keep = np.isin(labels, [classes.index(name) for name in CARRIAGEWAYS])
    if lead is not None:
        keep |= np.asarray(lead, dtype=bool)
    said = predicted == classes.index('merge')
    folded = predicted.copy()
    folded[said] = np.where(keep[said], labels[said], classes.index('motorway'))
    return labels, folded


def transition_kind(old: int, new: int, classes: tuple[str, ...]) -> str:
    a, b = classes[old], classes[new]
    if a == 'ramp' and b in CARRIAGEWAYS:
        return 'ramp_to_motorway'
    if a in CARRIAGEWAYS and b == 'ramp':
        return 'motorway_to_ramp'
    if b in TOWNS and a not in (*TOWNS, *CAR_PARKS):
        return 'town_entry'
    if a in TOWNS and b not in (*TOWNS, *CAR_PARKS):
        return 'town_exit'
    return 'other'


def change_points(labels: np.ndarray, classes: tuple[str, ...]) -> list[dict]:
    """The changes of road type with STABLE_SECONDS of each side (labels: class per second, -1 unknown):
    [{'time', 'old', 'new', 'kind'}]; the time is the first second of the new type."""
    runs = [run for run in _runs(np.asarray(labels)) if run[0] >= 0]
    found = []
    for (old, start_a, end_a), (new, start_b, end_b) in zip(runs, runs[1:]):
        if old == new or end_a - start_a < STABLE_SECONDS or end_b - start_b < STABLE_SECONDS:
            continue
        if start_b - end_a > GAP_SECONDS:
            continue
        found.append({'time': start_b, 'old': old, 'new': new, 'kind': transition_kind(old, new, classes)})
    return found


def detection_delay(predicted: np.ndarray, change: dict, raw: np.ndarray | None = None) -> int | None:
    """Seconds from a change to the first second of CONFIRM_SECONDS answers of its new type in a row (see the module's
    text), None when missed. raw: the answers before fold_merge, of which the run needs one class (merge, merge,
    motorway is no run for the car)."""
    predicted = np.asarray(predicted)
    begin = max(change['time'] - STABLE_SECONDS, 0)
    end = min(change['time'] + WINDOW_SECONDS, len(predicted))
    run = 0
    for i in range(begin, end):
        if predicted[i] != change['new']:
            run = 0
        else:
            run = run + 1 if run and (raw is None or raw[i] == raw[i - 1]) else 1
        if run >= CONFIRM_SECONDS:
            return i - CONFIRM_SECONDS + 1 - change['time']
    return None


def debounced(predicted: np.ndarray, confirm: int = CONFIRM_SECONDS) -> np.ndarray:
    """The answer the car would hold: a new class only after confirm seconds of it in a row (causal)."""
    predicted = np.asarray(predicted)
    if not len(predicted):
        return predicted.copy()
    held, candidate, run, out = int(predicted[0]), int(predicted[0]), 0, np.empty_like(predicted)
    for i, value in enumerate(predicted):
        value = int(value)
        if value == held:
            run = 0
        else:
            run = run + 1 if value == candidate else 1
            candidate = value
            if run >= confirm:
                held, run = value, 0
        out[i] = held
    return out


def majority(predicted: np.ndarray, window: int = 5, classes: int = 6) -> np.ndarray:
    """A causal majority vote over the last window answers (a frame network's cheap memory; ties to the newest)."""
    predicted = np.asarray(predicted)
    out = np.empty_like(predicted)
    for i in range(len(predicted)):
        recent = predicted[max(0, i - window + 1):i + 1]
        counts = np.bincount(recent, minlength=classes)
        best = counts.max()
        out[i] = next(int(v) for v in recent[::-1] if counts[v] == best)
    return out


def switches(predicted: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    """(answer changes, seconds) inside the stretches of one known road type, MARGIN_SECONDS from their ends."""
    predicted, count, seconds = np.asarray(predicted), 0, 0
    for label, start, end in _runs(np.asarray(labels)):
        if label < 0:
            continue
        inner_start, inner_end = start + MARGIN_SECONDS, end - MARGIN_SECONDS
        if inner_end - inner_start < 2:
            continue
        seconds += inner_end - inner_start
        part = predicted[inner_start:inner_end]
        count += int((part[1:] != part[:-1]).sum())
    return count, seconds


def merges(labels: np.ndarray, classes: tuple[str, ...]) -> list[tuple[int, int]]:
    """(first second, end exclusive) of the carriageway after every merge: a ramp of MERGE_RAMP_SECONDS or more, then
    motorway or expressway (its acceleration lane first) within GAP_SECONDS, up to AFTER_MERGE_SECONDS or the next
    other type."""
    ramp = classes.index('ramp')
    roads = {classes.index(name) for name in CARRIAGEWAYS if name in classes}
    runs = [run for run in _runs(fold_labels(labels, classes)) if run[0] >= 0]
    found = []
    for (old, start_a, end_a), (new, start_b, end_b) in zip(runs, runs[1:]):
        if old == ramp and new in roads and end_a - start_a >= MERGE_RAMP_SECONDS and start_b - end_a <= GAP_SECONDS:
            found.append((start_b, min(end_b, start_b + AFTER_MERGE_SECONDS)))
    return found


def after_merge(sequences: list[dict], classes: tuple[str, ...]) -> dict:
    """{'merges', 'frames', 'accuracy', 'accuracy_debounced', 'said': {class: frames}, 'per_merge': [{'frames',
    'correct'}]} over the first seconds of the carriageway after each merge (see the module's text)."""
    total = correct = steady = 0
    said = {name: 0 for name in classes}
    per_merge = []
    for sequence in sequences:
        lead = sequence.get('lead')
        truth, answer = fold_merge(sequence['labels'], sequence['predicted'], classes, lead)
        held = fold_merge(sequence['labels'], debounced(sequence['predicted']), classes, lead)[1]
        for start, end in merges(truth, classes):
            hits = int((answer[start:end] == truth[start:end]).sum())
            total += end - start
            correct += hits
            steady += int((held[start:end] == truth[start:end]).sum())
            for value in answer[start:end]:
                said[classes[int(value)]] += 1
            per_merge.append({'frames': end - start, 'correct': hits})
    return {'merges': len(per_merge), 'frames': total,
            'accuracy': round(correct / total, 4) if total else None,
            'accuracy_debounced': round(steady / total, 4) if total else None,
            'said': said, 'per_merge': per_merge}


def sequence_metrics(sequences: list[dict], classes: tuple[str, ...]) -> dict:
    """The measure over sequences [{'labels', 'weights', 'predicted'}] (arrays per second): per_class_f1 of weight-1
    frames and of every labelled frame, the transitions by kind (count, found, missed, delays, median and mean delay)
    and the false switches per hour, raw and debounced (these two with the acceleration lane folded: fold_merge, after
    the debounce for the debounced ones). A sequence's 'lead' (optional): its acceleration lane frames that were the
    ramp's."""
    labels = np.concatenate([np.asarray(s['labels']) for s in sequences]) if sequences else np.zeros(0, int)
    weights = np.concatenate([np.asarray(s['weights']) for s in sequences]) if sequences else np.zeros(0)
    predicted = np.concatenate([np.asarray(s['predicted']) for s in sequences]) if sequences else np.zeros(0, int)
    kinds = {kind: {'count': 0, 'found': 0, 'missed': 0, 'delays': []} for kind in TRANSITION_KINDS}
    raw = steady = seconds = 0
    for sequence in sequences:
        lead = sequence.get('lead')
        truth, answer = fold_merge(sequence['labels'], sequence['predicted'], classes, lead)
        for change in change_points(truth, classes):
            delay = detection_delay(answer, change, raw=np.asarray(sequence['predicted']))
            entry = kinds[change['kind']]
            entry['count'] += 1
            if delay is None:
                entry['missed'] += 1
            else:
                entry['found'] += 1
                entry['delays'].append(delay)
        count, length = switches(answer, truth)
        raw += count
        seconds += length
        steady += switches(fold_merge(sequence['labels'], debounced(sequence['predicted']), classes, lead)[1], truth)[0]
    for entry in kinds.values():
        delays = entry['delays']
        entry['median_delay'] = float(np.median(delays)) if delays else None
        entry['mean_delay'] = round(float(np.mean(delays)), 2) if delays else None
    hours = seconds / 3600
    return {'weight_1': per_class_f1(labels, predicted, weights >= 1, classes),
            'labelled': per_class_f1(labels, predicted, labels >= 0, classes),
            'confusion': confusion(labels, predicted, weights >= 1, len(classes)),
            'transitions': kinds,
            'false_switches_per_hour': round(raw / hours, 2) if hours else None,
            'false_switches_per_hour_debounced': round(steady / hours, 2) if hours else None,
            'stable_hours': round(hours, 3),
            'after_merge': after_merge(sequences, classes)}


# -- the comma's rule: when the limit of a ramp ends on the carriageway --------------------------------------------
# vision.SpeedLimitMemory (sunnypilot speed_limits): a number read on a ramp (the road type says "ramp") or within
# RAMP_SECONDS of a motorway or expressway sign (and not on the acceleration lane, nor past a ramp's limit, since that
# sign) is the ramp's limit; it ends, and the motorway's general limit comes back, as soon as the road type says merge
# (the acceleration lane) when the sign was that near and still names the carriageway (the car has not been on one for
# ROAD_TYPE_SECONDS since it), else once it says merge, motorway or expressway MERGE_SECONDS in a row and the step is
# motorway or expressway (the first such step starts the count). The road type given to it is the
# network's answer after the CONFIRM_SECONDS debounce, one step a second. The merge is the first second of the
# acceleration lane, which the map's labels start MERGE_LEAD_M before the node of the ramp's end. Any number read off
# the carriageways (the road type not motorway, expressway or merge: a ramp the network missed, said town or rural)
# also ends once the road type has said one carriageway ROAD_TYPE_SECONDS, if that began at least CARRIAGEWAY_LAG after
# the number: whichever rule comes first.
RAMP_SECONDS = 45.0
MERGE_SECONDS = 8
ROAD_TYPE_SECONDS = 10  # vision.ROAD_TYPE_SECONDS: a road type held this long is the road
CARRIAGEWAY_LAG = 5  # vision.CARRIAGEWAY_LAG: a number read later than this before the carriageway may be its own
LATE_SECONDS = 60
NUMBER_SECOND = 2  # the ramp's number is read at its third second (the middle of a shorter ramp)
RELEASE_OUTCOMES = ('on_ramp', 'in_time', 'late', 'not_held')


def ramp_entries(labels: np.ndarray, classes: tuple[str, ...], lead: np.ndarray | None = None) -> list[dict]:
    """Every ramp followed by the carriageway (motorway or expressway within GAP_SECONDS), as merges() finds them, with
    the ramp's first second: [{'ramp': first second of the ramp, 'merge': first second of the carriageway (of its
    acceleration lane when labelled, its lead included), 'before': the road type before the ramp (None: none
    known)}]. lead: the acceleration lane frames that were the ramp's (fold_labels), counted in the ramp's length."""
    ramp = classes.index('ramp')
    roads = {classes.index(name) for name in CARRIAGEWAYS if name in classes}
    lead = None if lead is None else np.asarray(lead, dtype=bool)
    runs = [run for run in _runs(fold_labels(labels, classes, lead)) if run[0] >= 0]
    found = []
    for i, ((old, start_a, end_a), (new, start_b, _)) in enumerate(zip(runs, runs[1:])):
        if old == ramp and new in roads and end_a - start_a >= MERGE_RAMP_SECONDS and start_b - end_a <= GAP_SECONDS:
            before = classes[runs[i - 1][0]] if i > 0 and start_a - runs[i - 1][2] <= GAP_SECONDS else None
            merge = start_b
            if lead is not None and start_b == end_a:
                while merge > start_a and lead[merge - 1]:
                    merge -= 1
            found.append({'ramp': start_a, 'merge': merge, 'before': before})
    return found


def _road_type_since(held: np.ndarray, t: int, lane: int | None) -> int:
    """The second the road type held at second t began, as vision's _road_type_since: a carriageway right after the
    acceleration lane counts from the lane's start."""
    start = t
    while start > 0 and held[start - 1] == held[t]:
        start -= 1
    while lane is not None and held[t] != lane and start > 0 and held[start - 1] == lane:
        start -= 1
    return start


def comma_release(held: np.ndarray, sign_age: np.ndarray, entry: dict, classes: tuple[str, ...],
                  joined_at: int | None = None) -> dict:
    """The comma's rule on one merge (held: the debounced answer per second of the drive; sign_age: seconds since the
    last motorway or expressway sign the car confirmed, inf when none; joined_at: the second the drive's last ramp's
    limit ended before this ramp, None: none): a number read at the ramp's NUMBER_SECOND; held as the ramp's when the
    road type says ramp then, or when the sign is at most RAMP_SECONDS old and neither a merge nor joined_at came since
    it; released on the first merge (the acceleration lane) when held by the sign and the car has not been on a
    carriageway for ROAD_TYPE_SECONDS since it (the sign still names the carriageway), else on motorway or expressway
    after MERGE_SECONDS of merge, motorway or expressway in a row; read off the carriageways, also released by the
    off-carriageway rule (see above), which alone holds it when neither the ramp nor the sign does. {'number_at',
    'held_by' (ramp, sign, ramp+sign, off or None), 'release' (seconds after the merge, negative: on the ramp; None:
    not by LATE_SECONDS after it), 'outcome'}."""
    held, sign_age = np.asarray(held), np.asarray(sign_age, dtype=float)
    ramp, roads = classes.index('ramp'), {classes.index(n) for n in CARRIAGEWAYS if n in classes}
    lane = classes.index('merge') if 'merge' in classes else None
    start, merge = entry['ramp'], entry['merge']
    number_at = start + min(NUMBER_SECOND, max(0, (merge - start) // 2))
    by_ramp = int(held[number_at]) == ramp
    by_sign = bool(sign_age[number_at] <= RAMP_SECONDS)
    known = False
    if by_sign:
        sign_at = max(0, number_at - int(np.floor(sign_age[number_at])))
        # On the acceleration lane, or past a ramp's limit, since the sign: the number is the carriageway's (vision.py).
        by_sign = not (lane is not None and bool(np.any(held[sign_at:number_at + 1] == lane))
                       or joined_at is not None and joined_at >= sign_at)
        # The sign names the carriageway the lane joins until the car has been on one for ROAD_TYPE_SECONDS since it.
        known = by_sign and lane is not None and not any(
            int(held[t]) in roads and t - max(_road_type_since(held, t, lane), sign_at) >= ROAD_TYPE_SECONDS
            for t in range(sign_at, number_at + 1))
    off = int(held[number_at]) not in roads and int(held[number_at]) != lane
    held_by = '+'.join(name for name, flag in (('ramp', by_ramp), ('sign', by_sign)) if flag) or ('off' if off else None)
    result = {'number_at': number_at - merge, 'held_by': held_by, 'release': None}
    if held_by is None:
        return {**result, 'outcome': 'not_held'}
    since = None
    for t in range(number_at + 1, min(len(held), merge + LATE_SECONDS + 1)):
        answer = int(held[t])
        released = held_by != 'off' and answer == lane and known
        if answer not in roads and answer != lane:
            since = None
        elif since is None:
            since = t
        released = released or held_by != 'off' and answer in roads and t - since >= MERGE_SECONDS
        if off and answer in roads:
            began = _road_type_since(held, t, lane)
            released = released or began - number_at >= CARRIAGEWAY_LAG and t - began >= ROAD_TYPE_SECONDS
        if released:
            return {**result, 'release': t - merge, 'outcome': 'on_ramp' if t < merge else 'in_time'}
    return {**result, 'outcome': 'late'}


def release_summary(found: list[dict]) -> dict:
    """{'merges', 'held', 'held_by_ramp', 'held_by_sign', 'held_by_off', 'on_ramp', 'in_time', 'late', 'not_held', 'median_release'
    (of the releases on the carriageway within LATE_SECONDS), 'releases' (every release, None for none)} of
    comma_release results."""
    counts = {outcome: sum(1 for item in found if item['outcome'] == outcome) for outcome in RELEASE_OUTCOMES}
    timely = [item['release'] for item in found if item['outcome'] == 'in_time']
    return {'merges': len(found), 'held': sum(1 for item in found if item['held_by']),
            'held_by_ramp': sum(1 for item in found if item['held_by'] and 'ramp' in item['held_by']),
            'held_by_sign': sum(1 for item in found if item['held_by'] and 'sign' in item['held_by']),
            'held_by_off': sum(1 for item in found if item['held_by'] == 'off'),
            **counts, 'median_release': float(np.median(timely)) if timely else None,
            'releases': [item['release'] for item in found]}


# -- the junction head: when the car has passed an intersection ------------------------------------------------------
# A false "passed" lifts the limit of a sign (Poland: a sign holds to the next intersection), the dangerous error: it is
# counted per hour and per 10 km, in town and outside. An event is the moment the decode says "passed"; it matches a
# labelled "passed" (the first frame after the intersection) within a tolerance, one to one, the closest first.
JUNCTION_TOLERANCES = (2.0, 4.0)
JUNCTION_THRESHOLDS = (0.3, 0.5, 0.7)
JUNCTION_GAP_SECONDS = 3.0  # a new event only this long after the last one (the same intersection said twice)
# 'passed': the passed probability rises to the threshold; 'exit': the in-junction probability falls below it after at
# least one second at or above it (the event is the first second below).
JUNCTION_DECODES = ('passed', 'exit')


def junction_events(p_passed: np.ndarray, p_in: np.ndarray, threshold: float, decode: str = 'passed',
                    gap: float = JUNCTION_GAP_SECONDS, seconds: np.ndarray | None = None) -> list[int]:
    """The positions of the "passed" events of a sequence (see JUNCTION_DECODES)."""
    seconds = np.arange(len(p_in), dtype=np.float64) if seconds is None else np.asarray(seconds, dtype=np.float64)
    if decode == 'passed':
        above = np.asarray(p_passed) >= threshold
        starts = [t for t in range(len(above)) if above[t] and (t == 0 or not above[t - 1])]
    elif decode == 'exit':
        inside = np.asarray(p_in) >= threshold
        starts = [t for t in range(1, len(inside)) if inside[t - 1] and not inside[t]]
    else:
        raise ValueError(f'unknown junction decode {decode!r}')
    events, last = [], -np.inf
    for t in starts:
        if seconds[t] - last >= gap:
            events.append(t)
            last = seconds[t]
    return events


def match_events(events: list[float], truth: list[float], tolerance: float) -> list[tuple[int, int]]:
    """One-to-one pairs (event, truth) within tolerance seconds, the closest pairs first."""
    pairs = sorted((abs(e - t), i, j) for i, e in enumerate(events) for j, t in enumerate(truth)
                   if abs(e - t) <= tolerance)
    used_e, used_t, found = set(), set(), []
    for _, i, j in pairs:
        if i not in used_e and j not in used_t:
            used_e.add(i)
            used_t.add(j)
            found.append((i, j))
    return found


def _ratio(a: float, b: float, digits: int = 4) -> float | None:
    return round(a / b, digits) if b else None


def junction_area(label: int, predicted: int, town: set[int]) -> str:
    """town or outside, by the road type of the frame (the network's answer where the map has none); town: the
    indices of TOWNS."""
    return 'town' if (label if label >= 0 else predicted) in town else 'outside'


def junction_metrics(sequences: list[dict], classes: tuple[str, ...], *,
                     thresholds: tuple[float, ...] = JUNCTION_THRESHOLDS) -> dict:
    """The junction head over whole sequences (each run from zeros, as the car): sequences [{'seconds', 'p_in',
    'p_passed', 'in', 'passed', 'known', 'ignore', 'labels', 'predicted', 'speed' (m/s, nan unknown)}].

    Per decode and threshold: events, recall of the labelled "passed" within 2 and 4 s, precision and F1 at 2 s, false
    events (matched to no "passed" within 4 s; an event on a frame without junction labels or just after a passage
    whose exit was not seen is not judged) per hour and per 10 km, in town and outside. Plus the in-junction frames
    (precision, recall, F1 at 0.5)."""
    town = {classes.index(name) for name in TOWNS if name in classes}
    areas = {'town': {'seconds': 0.0, 'metres': 0.0}, 'outside': {'seconds': 0.0, 'metres': 0.0}}
    truth_count, tp, fp, fn = 0, 0, 0, 0
    prepared = []
    for item in sequences:
        known = np.asarray(item['known'], dtype=bool)
        labels, predicted = np.asarray(item['labels']), np.asarray(item['predicted'])
        speed = np.nan_to_num(np.asarray(item['speed'], dtype=np.float64), nan=0.0)
        area = [junction_area(int(labels[t]), int(predicted[t]), town) for t in range(len(known))]
        for t in np.flatnonzero(known):
            areas[area[t]]['seconds'] += 1.0
            areas[area[t]]['metres'] += float(speed[t])
        seconds = np.asarray(item['seconds'], dtype=np.float64)
        truth = [float(seconds[t]) for t in np.flatnonzero(np.asarray(item['passed'], dtype=bool) & known)]
        truth_count += len(truth)
        inside, said = np.asarray(item['in'], dtype=bool), np.asarray(item['p_in']) >= 0.5
        tp += int((inside & said & known).sum())
        fp += int((~inside & said & known).sum())
        fn += int((inside & ~said & known).sum())
        judged = known & ~np.asarray(item['ignore'], dtype=bool)
        prepared.append((item, seconds, truth, judged, area))
    hours = sum(a['seconds'] for a in areas.values()) / 3600
    km = sum(a['metres'] for a in areas.values()) / 1000
    decodes = {}
    for decode in JUNCTION_DECODES:
        decodes[decode] = {}
        for threshold in thresholds:
            matched = {tolerance: 0 for tolerance in JUNCTION_TOLERANCES}
            events = false = 0
            false_area = {'town': 0, 'outside': 0}
            for item, seconds, truth, judged, area in prepared:
                found = [t for t in junction_events(item['p_passed'], item['p_in'], threshold, decode,
                                                    seconds=seconds) if judged[t]]
                times = [float(seconds[t]) for t in found]
                events += len(found)
                for tolerance in JUNCTION_TOLERANCES:
                    matched[tolerance] += len(match_events(times, truth, tolerance))
                hit = {i for i, _ in match_events(times, truth, max(JUNCTION_TOLERANCES))}
                for i, t in enumerate(found):
                    if i not in hit:
                        false += 1
                        false_area[area[t]] += 1
            precision = _ratio(matched[2.0], events)
            recall = _ratio(matched[2.0], truth_count)
            f1 = (round(2 * precision * recall / (precision + recall), 4)
                  if precision is not None and recall is not None and precision + recall > 0 else
                  0.0 if truth_count else None)
            decodes[decode][f'{threshold:g}'] = {
                'events': events, 'recall_2s': recall, 'recall_4s': _ratio(matched[4.0], truth_count),
                'precision_2s': precision, 'f1_2s': f1, 'false': false,
                'false_per_hour': _ratio(false, hours, 2), 'false_per_10km': _ratio(false, km / 10, 2),
                **{name: {'false': false_area[name],
                          'per_hour': _ratio(false_area[name], areas[name]['seconds'] / 3600, 2),
                          'per_10km': _ratio(false_area[name], areas[name]['metres'] / 10000, 2)}
                   for name in areas}}
    in_precision, in_recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    in_f1 = (round(2 * in_precision * in_recall / (in_precision + in_recall), 4)
             if in_precision and in_recall else 0.0 if tp + fn else None)
    return {'passed': truth_count, 'hours': round(hours, 3), 'km': round(km, 2),
            'areas': {name: {'hours': round(a['seconds'] / 3600, 3), 'km': round(a['metres'] / 1000, 2)}
                      for name, a in areas.items()},
            'in_junction': {'precision': in_precision, 'recall': in_recall, 'f1': in_f1},
            'decodes': decodes}


def operating_point(metrics: dict) -> dict:
    """The decode and threshold of the car, chosen on validation: the best F1 at 2 s; on a tie the fewer false events.
    {'decode', 'threshold', ...its numbers}."""
    best = None
    for decode, by_threshold in metrics['decodes'].items():
        for threshold, numbers in by_threshold.items():
            key = (numbers['f1_2s'] or 0.0, -(numbers['false_per_hour'] or 0.0), float(threshold))
            if best is None or key > best[0]:
                best = (key, {'decode': decode, 'threshold': float(threshold), **numbers})
    return best[1] if best else {}


# -- a turn as a plain rule, without the network: what the car's own motion alone says ------------------------------
# A turn at an intersection is a large change of heading on a small radius. Geometry only, no speed threshold: the
# radius is the path driven during the turn (the speed of each second) over the heading change in radians.
TURN_DEGREES = 60.0
TURN_RADIUS_M = 40.0
TURN_MIN_DPS = 2.0  # deg/s: a second turning less than this (or the other way, or unknown) ends a turn


def turns(yaw_dps: np.ndarray, speed_mps: np.ndarray, min_dps: float = TURN_MIN_DPS) -> list[dict]:
    """The turns of a sequence: runs of consecutive seconds turning the same way at min_dps or more (yaw_dps: the
    heading change of each 1 s step in degrees, positive left, nan unknown). Each {'start', 'end' (exclusive),
    'degrees' (signed), 'metres' (nan when a speed is unknown), 'radius_m'}."""
    yaw = np.asarray(yaw_dps, dtype=np.float64)
    speed = np.asarray(speed_mps, dtype=np.float64)
    found, start = [], None

    def close(end: int) -> None:
        degrees = float(yaw[start:end].sum())
        metres = float(speed[start:end].sum())
        radians = abs(degrees) * np.pi / 180
        found.append({'start': start, 'end': end, 'degrees': degrees, 'metres': metres,
                      'radius_m': metres / radians if radians > 0 and np.isfinite(metres) else float('nan')})

    for t in range(len(yaw)):
        turning = np.isfinite(yaw[t]) and abs(yaw[t]) >= min_dps
        if start is not None and (not turning or np.sign(yaw[t]) != np.sign(yaw[start])):
            close(t)
            start = None
        if turning and start is None:
            start = t
    if start is not None:
        close(len(yaw))
    return found


def turn_events(yaw_dps: np.ndarray, speed_mps: np.ndarray, *, degrees: float = TURN_DEGREES,
                radius_m: float = TURN_RADIUS_M, min_dps: float = TURN_MIN_DPS) -> list[int]:
    """The "turned at an intersection" events of the plain rule: the first second after every turn (turns) of at least
    degrees on a radius of at most radius_m; none for a turn still going at the end of the sequence."""
    return [turn['end'] for turn in turns(yaw_dps, speed_mps, min_dps)
            if abs(turn['degrees']) >= degrees and turn['radius_m'] <= radius_m and turn['end'] < len(yaw_dps)]
