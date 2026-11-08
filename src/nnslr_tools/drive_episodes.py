"""The episodes of a recorded drive worth a look (the Episodes lane of Regression): found in the comma's logs alone (CPU, no
image, no neural network), on a drive joined by drive_logs.merge.

Two channels, independent of each other:

- interventions of the driver while openpilot drove: the accelerator while openpilot held the speed, the brake, the
  steering wheel really forced (held, not touched), a disengagement (the driver's own, or openpilot's), a set speed
  changed by hand. Each is judged on the state just before it (100 ms): a pedal pressed while the driver already drove
  by hand is not an intervention, and neither is a set speed that Speed Limit Assist or ICBM changed. A set speed
  changed with ICBM idle (state inactive, no button sent) is the driver's even where the car logs no cruise buttons
  (the PSA): only the driver's stalk or ICBM's buttons move the car's set speed;
- the comma's vision limit and the map's limit far apart (5 km/h or more for 3 s or more) with no intervention at all.
  The vision reads a sign tens of metres before the car reaches it, the map changes only where its limit begins (at
  the sign or past it, at its next update): when the vision changed first and the map came to the same limit within
  10 s, the vision was ahead, not in disagreement, and nothing is found (counted as vision_ahead).

Clues close in time (8 s), also across the minutes of the drive, make one episode; an intervention during a
disagreement belongs to it. Episodes with several clues that agree come first (priority, with its reasons). The
speed the driver chose is only a hint of which limit was right (it is never a limit). A value the log does not have
is missing, never a disagreement: the vision is "not sure" (0) or absent, the map has no valid limit, and nothing is
found there.

A few spot checks (30 s far from any episode, one every 15 minutes, the same ones at every scan of the same drive)
let a person confirm that the comma was right where nothing looked wrong.

The vision limit here is the one the comma showed (speedVisionSP.speedLimit, the memory's limit), not a sign read in a
frame: signRead marks the moments it confirmed a sign. Which limit the car used is the resolver's (map and vision
weighed by sunnypilot, longitudinalPlanSP.speedLimit.resolver).
"""
from __future__ import annotations

import bisect
import hashlib
import random
import statistics

SCANNER_VERSION = 4
NS = 1_000_000_000
# The state an intervention is judged on: this long before it.
BEFORE_NS = NS // 10
# Clues closer than this make one episode.
GROUP_GAP_NS = 8 * NS
# A steering override is the wheel held at least this long while openpilot steered; shorter is a touch.
STEER_HOLD_NS = NS
# Vision and map disagree when their limits are this far apart for this long; runs closer than MERGE_NS are one.
DISAGREE_KPH = 5
DISAGREE_NS = 3 * NS
MERGE_NS = 5 * NS
# The vision ahead of the map: a run this short that the map ends by coming to the vision's limit.
LEAD_NS = 10 * NS
# A disengagement's cause is looked for in this window around it.
CAUSE_BEFORE_NS = 2 * NS
CAUSE_AFTER_NS = 3 * NS // 10
# A set speed changed within this long of a cruise button, or of a button ICBM sent.
BUTTON_NS = NS
SET_BUTTONS = {'accelCruise', 'decelCruise', 'setCruise', 'resumeCruise'}
CANCEL_BUTTONS = {'cancel', 'mainCruise'}
# How long after a pedal the driver's speed is read, for the hint.
HINT_NS = 10 * NS
SPOT_NS = 30 * NS
SPOT_EVERY_MINUTES = 15
SPOT_CLEAR_NS = 60 * NS
# carState.vCruise when no speed is set.
NO_SET_SPEED = 250

INTERVENTIONS = ('gas', 'brake', 'steer', 'set_speed_manual', 'disengage_user', 'disengage_auto', 'disengage_unknown')
# Clues that only add to an episode: a touch of the wheel, a set speed changed by the car or by an unknown hand.
EVIDENCE = ('steer_contact', 'set_speed_auto', 'set_speed_unknown')
WEIGHT = {'gas': 2, 'brake': 2, 'set_speed_manual': 2, 'disengage_user': 2, 'steer': 1, 'disengage_auto': 1,
          'disengage_unknown': 1, 'disagreement': 2}
KIND_TEXT = {
    'gas': 'accelerator while openpilot held the speed',
    'brake': 'brake while openpilot was engaged',
    'steer': 'steering wheel held against openpilot',
    'set_speed_manual': 'set speed changed by hand',
    'disengage_user': 'openpilot disengaged by the driver',
    'disengage_auto': 'openpilot disengaged by itself',
    'disengage_unknown': 'openpilot disengaged (cause not in the log)',
    'disagreement': 'vision and map limits disagree',
    'steer_contact': 'steering wheel touched',
    'set_speed_auto': 'set speed changed by the car (ICBM)',
    'set_speed_unknown': 'set speed changed (by hand or by the car: the log does not say)',
    'spot_check': 'spot check: nothing looked wrong here',
}


class Step:
    """A series of drive_extract as a step function: its value at any time (None before its first point)."""

    def __init__(self, points: list | None) -> None:
        points = points or []
        self.t = [p[0] for p in points]
        self.v = [p[1] for p in points]

    def __bool__(self) -> bool:
        return bool(self.t)

    def at(self, t: int):
        index = bisect.bisect_right(self.t, t) - 1
        return self.v[index] if index >= 0 else None

    def between(self, start: int, end: int) -> list:
        """[[t, value]] from start (its value then) to end."""
        first = bisect.bisect_right(self.t, start)
        out = [[start, self.at(start)]]
        for index in range(first, len(self.t)):
            if self.t[index] > end:
                break
            out.append([self.t[index], self.v[index]])
        return out

    def present(self, start: int, end: int) -> bool:
        return any(value is not None for _t, value in self.between(start, end))

    def changes(self):
        return zip(self.t, self.v)


def _limit(value) -> int | None:
    """A limit of the logs in whole km/h; None when it is missing or 0 (not sure)."""
    if value is None:
        return None
    number = round(float(value))
    return number if number > 0 else None


def _set_speed(value) -> int | None:
    if value is None or value <= 0 or value >= NO_SET_SPEED:
        return None
    return round(float(value))


class Drive:
    """The series of a merged drive and the questions the scanner asks them."""

    def __init__(self, merged: dict) -> None:
        self.merged = merged
        self.s = {name: Step(points) for name, points in merged.get('series', {}).items()}
        self.events = merged.get('events', [])
        self.event_t = [item[0] for item in self.events]
        car = (merged.get('meta') or {}).get('car') or {}
        self.op_long = car.get('openpilot_longitudinal')

    def get(self, name: str) -> Step:
        return self.s.get(name) or Step([])

    def events_in(self, start: int, end: int, kind: str | None = None) -> list:
        first = bisect.bisect_left(self.event_t, start)
        out = []
        for index in range(first, len(self.events)):
            if self.events[index][0] > end:
                break
            if kind is None or self.events[index][1] == kind:
                out.append(self.events[index])
        return out

    def engaged(self, t: int) -> bool | None:
        """openpilot engaged at t (selfdriveState.enabled); None when the log does not say."""
        return self.get('sd_enabled').at(t)

    def long_engaged(self, t: int) -> bool:
        """openpilot held the speed at t: engaged with its longitudinal control active, or with the car's cruise
        control when openpilot does not drive the pedals on this car."""
        if self.engaged(t) is not True:
            return False
        if self.get('long_active').at(t) is True:
            return True
        return self.op_long is False and self.get('cruise_enabled').at(t) is True

    def lat_engaged(self, t: int) -> bool:
        lat = self.get('lat_active').at(t)
        if lat is not None:
            return lat is True
        return self.get('sd_active').at(t) is True or self.get('mads_active').at(t) is True

    def vision(self, t: int) -> int | None:
        return _limit(self.get('vis_limit').at(t))

    def map(self, t: int) -> int | None:
        return _limit(self.get('map_limit').at(t)) if self.get('map_valid').at(t) is True else None

    def state(self, t: int) -> dict:
        """What the comma recorded at t (the state an episode is judged on)."""
        get = lambda name: self.get(name).at(t)  # noqa: E731
        return {'vision_kph': _limit(get('vis_limit')), 'vision_logged': get('vis_limit') is not None,
                'map_kph': self.map(t), 'map_logged': get('map_valid') is not None,
                'resolver_kph': _limit(get('res_limit')), 'resolver_source': get('res_source'),
                'sla_state': get('sla_state'), 'v_ego_kph': get('v_ego'), 'set_speed_kph': _set_speed(get('v_cruise')),
                'openpilot': get('sd_state'), 'engaged': get('sd_enabled'), 'long_active': get('long_active'),
                'lat_active': get('lat_active'), 'road_type': get('vis_road') or None,
                'road_name': get('road_name') or None, 'vision_model': get('vis_model') or None}


# ---------- the interventions ----------

def _rising(step: Step):
    """The times a boolean series turns from False to True, with the time it stops being True (None: it never does).
    After a missing value (None) the first True is no edge: when it began is not known."""
    before = None
    pending = None
    for t, value in step.changes():
        if value is True and before is False:
            pending = t
        elif value is not True and pending is not None:
            yield pending, t
            pending = None
        before = value
    if pending is not None:
        yield pending, None


def interventions(drive: Drive) -> list[dict]:
    """Every intervention and piece of evidence of the drive: {kind, t_ns, end_ns, detail}, in time order."""
    out = []
    for t, _end in _rising(drive.get('gas')):
        if drive.long_engaged(t - BEFORE_NS):
            out.append({'kind': 'gas', 't_ns': t, 'end_ns': t, 'detail': {}})
    for t, _end in _rising(drive.get('brake')):
        if drive.engaged(t - BEFORE_NS) is True:
            out.append({'kind': 'brake', 't_ns': t, 'end_ns': t, 'detail': {}})
    for t, end in _rising(drive.get('steer_pressed')):
        if not drive.lat_engaged(t - BEFORE_NS):
            continue
        held = (end - t) if end is not None else None
        kind = 'steer' if held is None or held >= STEER_HOLD_NS else 'steer_contact'
        out.append({'kind': kind, 't_ns': t, 'end_ns': end if end is not None else t,
                    'detail': {'held_s': round(held / NS, 2) if held is not None else None}})
    out.extend(_disengagements(drive))
    out.extend(_set_speed_changes(drive))
    out.sort(key=lambda item: item['t_ns'])
    return out


def _disengagements(drive: Drive) -> list[dict]:
    out = []
    before = None
    for t, value in drive.get('sd_enabled').changes():
        # Through a hole of the log (None) the moment is not known: no disengagement is placed there.
        if before is True and value is False:
            out.append(_disengagement(drive, t))
        before = value
    return out


def _disengagement(drive: Drive, t: int) -> dict:
    start, end = t - CAUSE_BEFORE_NS, t + CAUSE_AFTER_NS
    reasons, kind = [], 'disengage_unknown'
    for event in drive.events_in(start, end, 'onroad'):
        for name, flags in event[3]:
            if 'softDisable' in flags or 'immediateDisable' in flags:
                reasons.append(name)
                kind = 'disengage_auto'
            elif 'userDisable' in flags and kind != 'disengage_auto':
                reasons.append(name)
                kind = 'disengage_user'
    if kind == 'disengage_unknown':
        for event in drive.events_in(start, end, 'button'):
            if event[2] in CANCEL_BUTTONS:
                reasons.append(event[2])
                kind = 'disengage_user'
        for name in ('brake', 'gas'):
            for edge, _end in _rising(Step(drive.get(name).between(start, end))):
                if edge > start:
                    reasons.append(name + 'Pressed')
                    kind = 'disengage_user'
    return {'kind': kind, 't_ns': t, 'end_ns': t, 'detail': {'causes': sorted(set(reasons))}}


def _set_speed_changes(drive: Drive) -> list[dict]:
    out = []
    previous = None
    icbm = drive.get('icbm_send')
    icbm_state = drive.get('icbm_state')
    for t, value in drive.get('v_cruise').changes():
        now = _set_speed(value)
        was, previous = previous, now
        if was is None or now is None or abs(now - was) < 1:
            continue
        if drive.engaged(t - BEFORE_NS) is not True or drive.engaged(t) is not True:
            continue
        sent = [value for _t, value in icbm.between(t - BUTTON_NS, t + BUTTON_NS) if value not in (None, 'none')]
        buttons = [event[2] for event in drive.events_in(t - BUTTON_NS, t + BUTTON_NS, 'button')
                   if event[2] in SET_BUTTONS]
        # ICBM idle the whole second around the change: no button of the car's was pressed by openpilot.
        idle = all(state == 'inactive' for _t, state in icbm_state.between(t - BUTTON_NS, t + BUTTON_NS))
        kind = ('set_speed_auto' if sent else 'set_speed_manual' if buttons or idle else 'set_speed_unknown')
        out.append({'kind': kind, 't_ns': t, 'end_ns': t,
                    'detail': {'from_kph': was, 'to_kph': now, 'buttons': sorted(set(buttons)), 'icbm': sorted(set(sent)),
                               'icbm_idle': idle}})
    return out


# ---------- vision against map ----------

def _vision_ahead(drive: Drive, start: int, end: int, parts: list) -> bool:
    """A run that began with a new vision limit (the map unchanged) and ended within LEAD_NS with the map on that
    limit: the vision read the sign before the place where the map's limit begins."""
    if end - start > LEAD_NS or len({part[:2] for part in parts}) != 1:
        return False
    vision, map_limit = parts[0][:2]
    after = drive.map(end)
    return (drive.map(start - 1) == map_limit and drive.vision(end) == vision and after is not None
            and abs(after - vision) < DISAGREE_KPH)


def disagreements(drive: Drive) -> list[dict]:
    """Runs where the vision limit and the map's are both known and 5 km/h or more apart for 3 s or more:
    {kind: 'disagreement', t_ns, end_ns, detail: {pairs: [[vision, map, seconds]], seconds}}; a run where the vision
    was only ahead of the map has the kind 'vision_ahead' instead."""
    times = sorted({t for name in ('vis_limit', 'map_limit', 'map_valid') for t in drive.get(name).t})
    runs: list[list] = []
    for index, t in enumerate(times):
        end = times[index + 1] if index + 1 < len(times) else None
        vision, map_limit = drive.vision(t), drive.map(t)
        if vision is None or map_limit is None or abs(vision - map_limit) < DISAGREE_KPH or end is None:
            continue
        if runs and t - runs[-1][1] < MERGE_NS:
            runs[-1][1] = end
            runs[-1][2].append((vision, map_limit, end - t))
        else:
            runs.append([t, end, [(vision, map_limit, end - t)]])
    out = []
    for start, end, parts in runs:
        apart = sum(part[2] for part in parts)
        if apart < DISAGREE_NS:
            continue
        pairs: dict[tuple, int] = {}
        for vision, map_limit, length in parts:
            pairs[(vision, map_limit)] = pairs.get((vision, map_limit), 0) + length
        out.append({'kind': 'vision_ahead' if _vision_ahead(drive, start, end, parts) else 'disagreement',
                    't_ns': start, 'end_ns': end,
                    'detail': {'seconds': round(apart / NS, 1),
                               'pairs': [[v, m, round(n / NS, 1)] for (v, m), n in
                                         sorted(pairs.items(), key=lambda item: -item[1])]}})
    return out


# ---------- episodes ----------

def _group(clues: list[dict], runs: list[dict]) -> list[list[dict]]:
    """Clues into episodes: a disagreement with the interventions during it (and 8 s around), then the other
    interventions, 8 s apart at most. Evidence only joins an episode that has an intervention or a disagreement."""
    groups: list[dict] = [{'start': run['t_ns'], 'end': run['end_ns'], 'clues': [run]} for run in runs]
    loose: list[dict] = []
    for clue in clues:
        home = next((group for group in groups if group['start'] - GROUP_GAP_NS <= clue['t_ns']
                     <= group['end'] + GROUP_GAP_NS), None)
        if home is not None:
            home['clues'].append(clue)
            home['start'] = min(home['start'], clue['t_ns'])
            home['end'] = max(home['end'], clue['end_ns'])
        else:
            loose.append(clue)
    alone: list[dict] = []
    for clue in loose:
        if alone and clue['t_ns'] <= alone[-1]['end'] + GROUP_GAP_NS:
            alone[-1]['clues'].append(clue)
            alone[-1]['end'] = max(alone[-1]['end'], clue['end_ns'])
        else:
            alone.append({'start': clue['t_ns'], 'end': clue['end_ns'], 'clues': [clue]})
    out = []
    for group in groups + alone:
        if any(clue['kind'] in INTERVENTIONS or clue['kind'] == 'disagreement' for clue in group['clues']):
            out.append(sorted(group['clues'], key=lambda clue: clue['t_ns']))
    return sorted(out, key=lambda clues: clues[0]['t_ns'])


def _hint(drive: Drive, clues: list[dict], anchor: int, vision: int | None, map_limit: int | None) -> dict | None:
    """The speed the driver chose and which of the two limits it is closer to: a hint, never a limit."""
    if vision is None or map_limit is None or vision == map_limit:
        return None
    speed = source = None
    for clue in clues:
        if clue['kind'] == 'set_speed_manual':
            speed, source = clue['detail']['to_kph'], 'set speed chosen by hand'
            break
        if clue['kind'] in ('gas', 'brake'):
            values = [v for _t, v in drive.get('v_ego').between(clue['t_ns'], clue['t_ns'] + HINT_NS) if v is not None]
            if values:
                speed = max(values) if clue['kind'] == 'gas' else min(values)
                source = 'speed reached after the accelerator' if clue['kind'] == 'gas' else 'speed after the brake'
                break
    if speed is None:
        run = next((clue for clue in clues if clue['kind'] == 'disagreement'), None)
        if run is None:
            return None
        values = [v for _t, v in drive.get('v_ego').between(run['t_ns'], run['end_ns']) if v is not None]
        if not values:
            return None
        speed, source = statistics.median(values), 'driver\'s speed during the disagreement'
    to_vision, to_map = abs(speed - vision), abs(speed - map_limit)
    closer = None if abs(to_vision - to_map) < DISAGREE_KPH else 'vision' if to_vision < to_map else 'map'
    return {'speed_kph': round(speed), 'source': source, 'closer_to': closer}


def _availability(drive: Drive, start: int, end: int, kinds: dict[int, str], spans: dict) -> dict:
    logs = sorted({kinds[segment] for segment, span in spans.items()
                   if segment in kinds and span[0] <= end and span[1] >= start})
    return {'vision': drive.get('vis_limit').present(start, end), 'map': drive.get('map_valid').present(start, end),
            'resolver': drive.get('res_source').present(start, end), 'state': drive.get('sd_enabled').present(start, end),
            'road_type': any(v for _t, v in drive.get('vis_road').between(start, end)),
            'log': logs[0] if len(logs) == 1 else 'mixed' if logs else None}


def previous_change(drive: Drive, before: int) -> dict | None:
    """The last time before `before` the vision limit took a known value other than its previous known one (a 0, "not
    sure", between two equal values is no change): {t_ns, vision_kph, from_kph}."""
    step = drive.get('vis_limit')
    known = [(t, _limit(v)) for t, v in step.changes() if t < before and _limit(v) is not None]
    for index in range(len(known) - 1, -1, -1):
        if index == 0 or known[index][1] != known[index - 1][1]:
            return {'t_ns': known[index][0], 'vision_kph': known[index][1],
                    'from_kph': known[index - 1][1] if index else None}
    return None


def _episode(drive: Drive, clues: list[dict], kinds: dict[int, str], spans: dict) -> dict:
    start = min(clue['t_ns'] for clue in clues)
    end = max(clue['end_ns'] for clue in clues)
    acts = [clue for clue in clues if clue['kind'] in INTERVENTIONS]
    run = next((clue for clue in clues if clue['kind'] == 'disagreement'), None)
    anchor = acts[0]['t_ns'] if acts else start
    before = drive.state(anchor - BEFORE_NS)
    names = sorted({clue['kind'] for clue in clues})
    channel = 'both' if acts and run else 'intervention' if acts else 'disagreement'
    priority, reasons = 0, []
    for name in names:
        if name in WEIGHT:
            priority += WEIGHT[name]
            reasons.append(KIND_TEXT[name])
    if acts and run:
        priority += 2
        reasons.append('an intervention during a disagreement of the limits')
    if run is not None:
        sources = {v for _t, v in drive.get('res_source').between(run['t_ns'], run['end_ns'])}
        if 'vision' in sources:
            priority += 1
            reasons.append("the car's limit came from the vision")
        sla = {v for _t, v in drive.get('sla_state').between(start, end)}
        if sla & {'active', 'adapting', 'pending'}:
            priority += 1
            reasons.append('Speed Limit Assist was acting on the limit')
        vision, map_limit = run['detail']['pairs'][0][0], run['detail']['pairs'][0][1]
    else:
        vision, map_limit = before['vision_kph'], before['map_kph']
    hint = _hint(drive, clues, anchor, vision, map_limit)
    if hint and hint['closer_to']:
        priority += 1
        reasons.append(f"the driver's speed ({hint['speed_kph']} km/h) is closer to the {hint['closer_to']} "
                       '(a hint only)')
    segments = [segment for segment, span in spans.items() if span[0] <= end and span[1] >= start]
    return {'start_ns': start, 'end_ns': end, 'anchor_ns': anchor, 'channel': channel, 'kinds': names,
            'clues': clues, 'priority': priority, 'reasons': reasons, 'hint': hint, 'before': before,
            'data': _availability(drive, start, end, kinds, spans),
            'segments': [min(segments), max(segments)] if segments else None,
            'previous_change': previous_change(drive, start)}


def _spot_checks(route: str, drive: Drive, episodes: list[dict], kinds: dict[int, str], spans: dict) -> list[dict]:
    """Windows of 30 s far from any episode, with the vision logged, picked the same way at every scan."""
    if not spans:
        return []
    begin = min(span[0] for span in spans.values())
    finish = max(span[1] for span in spans.values())
    count = max(1, int((finish - begin) / NS / 60 // SPOT_EVERY_MINUTES))
    busy = [(item['start_ns'] - SPOT_CLEAR_NS, item['end_ns'] + SPOT_CLEAR_NS) for item in episodes]
    candidates = []
    t = begin + SPOT_NS
    while t + SPOT_NS < finish:
        end = t + SPOT_NS
        if (not any(a < end and b > t for a, b in busy) and drive.get('vis_limit').present(t, end)
                and any(span[0] <= t and span[1] >= end for span in spans.values())):
            candidates.append(t)
        t += SPOT_NS
    chooser = random.Random(int(hashlib.sha256(route.encode()).hexdigest()[:12], 16))
    picked = sorted(chooser.sample(candidates, min(count, len(candidates)))) if candidates else []
    out = []
    for start in picked:
        end = start + SPOT_NS
        segments = [segment for segment, span in spans.items() if span[0] <= end and span[1] >= start]
        out.append({'start_ns': start, 'end_ns': end, 'anchor_ns': start, 'channel': 'spot_check',
                    'kinds': ['spot_check'], 'clues': [], 'priority': 0, 'reasons': [KIND_TEXT['spot_check']],
                    'hint': None, 'before': drive.state(start), 'data': _availability(drive, start, end, kinds, spans),
                    'segments': [min(segments), max(segments)], 'previous_change': previous_change(drive, start)})
    return out


def _seconds(drive: Drive, test, spans: dict) -> float:
    """How long (s) within the logged minutes a test of the state holds, sampled every 0.5 s."""
    total = 0
    for span in spans.values():
        t = span[0]
        while t < span[1]:
            if test(t):
                total += 1
            t += NS // 2
    return total / 2


def scan(route: str, merged: dict, log_kinds: dict[int, str]) -> dict:
    """The episodes of a merged drive (drive_logs.merge) and what the scan could see: {scanner_version, episodes,
    counts, coverage}. log_kinds: the kind of log read for each segment (rlog, qlog)."""
    drive = Drive(merged)
    spans = {int(segment): span for segment, span in (merged.get('spans') or {}).items()}
    found = interventions(drive)
    runs = disagreements(drive)
    ahead = [run for run in runs if run['kind'] == 'vision_ahead']
    runs = [run for run in runs if run['kind'] == 'disagreement']
    episodes = [_episode(drive, clues, log_kinds, spans) for clues in _group(found, runs)]
    episodes += _spot_checks(route, drive, episodes, log_kinds, spans)
    episodes.sort(key=lambda item: item['start_ns'])
    counts: dict[str, int] = {}
    for clue in found + runs + ahead:
        counts[clue['kind']] = counts.get(clue['kind'], 0) + 1
    coverage = {
        'seconds': round(sum((span[1] - span[0]) / NS for span in spans.values()), 1),
        'vision': round(_seconds(drive, lambda t: drive.get('vis_limit').at(t) is not None, spans), 1),
        'vision_limit': round(_seconds(drive, lambda t: drive.vision(t) is not None, spans), 1),
        'map': round(_seconds(drive, lambda t: drive.map(t) is not None, spans), 1),
        'resolver': round(_seconds(drive, lambda t: drive.get('res_source').at(t) is not None, spans), 1),
        'engaged': round(_seconds(drive, lambda t: drive.engaged(t) is True, spans), 1),
        'state': round(_seconds(drive, lambda t: drive.engaged(t) is not None, spans), 1),
        'disagreement': round(sum(run['detail']['seconds'] for run in runs), 1),
        'buttons': any(event[1] == 'button' for event in drive.events),
        'logs': sorted(set(log_kinds.values())),
    }
    car = (merged.get('meta') or {}).get('car') or {}
    return {'scanner_version': SCANNER_VERSION, 'episodes': episodes, 'counts': counts, 'coverage': coverage,
            'dashcam': car.get('dashcam_only')}
