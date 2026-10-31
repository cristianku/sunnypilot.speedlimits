"""One comma log (rlog or qlog) of one segment as the compact time series of the drive review (nnslr_tools.drive_logs).

    python -m nnslr_tools.drive_extract <cereal dir> <car.capnp dir> <log file>

prints one JSON object. It runs in a child process, so a log pycapnp cannot read never stops the site. It only reads
the log: no network, no GPU, no neural network. Every time is the message's logMonoTime (ns, the comma's monotonic
clock, which runs on across the segments of a drive); the civil time of the GPS is kept only to show the date.

- series: {name: [[t, value], ...]}, each a step function written only when the value changes (a number by at least
  its tolerance). A service that stops sending (its process stopped, a gap in the log) ends its series with a value
  of None after GAP_SECONDS[service]: a missing value is never carried on as if it were still read.
- events: [[t, kind, ...]]: the buttons of carState, the signs speedvisiond confirmed (signRead), every change of the
  onroad events (their name and their type: userDisable, softDisable, immediateDisable, ...).
- frames: the road camera's encode index (roadEncodeIdx, or narrowRoadEncodeIdx on this car): the video frame
  (segmentId, its index in fcamera.hevc in presentation order) and the start of its exposure (timestampSof, the same
  monotonic clock). This joins the logs to the video, never the file times or the names.
- gps: [[t, lat, lon, unix ms]] about once a second, with a fix only.
- meta: the build (initData), the car (carParams), the counts of messages per service and the span of the log.

The schema is the one of the runtime branch (drive_logs.schema_dir): speedVisionSP, longitudinalPlanSP.speedLimit
and liveMapDataSP are sunnypilot's (cereal/custom.capnp). A field the schema or an older log lacks reads as None.
"""
from __future__ import annotations

import bz2
import json
import sys
from pathlib import Path

EXTRACT_VERSION = 1

# (series, service, field path, kind, tolerance, scale): numbers are km/h where the log has m/s (scale 3.6).
SERIES = (
    ('v_ego', 'carState', 'vEgo', 'num', 0.5, 3.6),
    ('gas', 'carState', 'gasPressed', 'bool', 0, 1),
    ('brake', 'carState', 'brakePressed', 'bool', 0, 1),
    ('steer_pressed', 'carState', 'steeringPressed', 'bool', 0, 1),
    ('cruise_enabled', 'carState', 'cruiseState.enabled', 'bool', 0, 1),
    ('cruise_available', 'carState', 'cruiseState.available', 'bool', 0, 1),
    # carState.vCruise is already km/h; 255 means no set speed.
    ('v_cruise', 'carState', 'vCruise', 'num', 0.5, 1),
    ('cruise_speed', 'carState', 'cruiseState.speed', 'num', 0.5, 3.6),
    ('sd_state', 'selfdriveState', 'state', 'str', 0, 1),
    ('sd_enabled', 'selfdriveState', 'enabled', 'bool', 0, 1),
    ('sd_active', 'selfdriveState', 'active', 'bool', 0, 1),
    ('alert', 'selfdriveState', 'alertType', 'str', 0, 1),
    ('mads_state', 'selfdriveStateSP', 'mads.state', 'str', 0, 1),
    ('mads_active', 'selfdriveStateSP', 'mads.active', 'bool', 0, 1),
    ('icbm_state', 'selfdriveStateSP', 'intelligentCruiseButtonManagement.state', 'str', 0, 1),
    ('icbm_send', 'selfdriveStateSP', 'intelligentCruiseButtonManagement.sendButton', 'str', 0, 1),
    ('lat_active', 'carControl', 'latActive', 'bool', 0, 1),
    ('long_active', 'carControl', 'longActive', 'bool', 0, 1),
    ('res_limit', 'longitudinalPlanSP', 'speedLimit.resolver.speedLimit', 'num', 0.5, 3.6),
    ('res_source', 'longitudinalPlanSP', 'speedLimit.resolver.source', 'str', 0, 1),
    ('res_final', 'longitudinalPlanSP', 'speedLimit.resolver.speedLimitFinal', 'num', 0.5, 3.6),
    ('sla_state', 'longitudinalPlanSP', 'speedLimit.assist.state', 'str', 0, 1),
    ('sla_active', 'longitudinalPlanSP', 'speedLimit.assist.active', 'bool', 0, 1),
    ('plan_source', 'longitudinalPlanSP', 'longitudinalPlanSource', 'str', 0, 1),
    ('v_target', 'longitudinalPlanSP', 'vTarget', 'num', 0.5, 3.6),
    ('a_target', 'longitudinalPlan', 'aTarget', 'num', 0.25, 1),
    ('vis_limit', 'speedVisionSP', 'speedLimit', 'num', 0.5, 3.6),
    ('vis_road', 'speedVisionSP', 'roadType', 'str', 0, 1),
    ('vis_model', 'speedVisionSP', 'modelName', 'str', 0, 1),
    ('map_valid', 'liveMapDataSP', 'speedLimitValid', 'bool', 0, 1),
    ('map_limit', 'liveMapDataSP', 'speedLimit', 'num', 0.5, 3.6),
    ('map_ahead_valid', 'liveMapDataSP', 'speedLimitAheadValid', 'bool', 0, 1),
    ('map_ahead', 'liveMapDataSP', 'speedLimitAhead', 'num', 0.5, 3.6),
    ('map_ahead_dist', 'liveMapDataSP', 'speedLimitAheadDistance', 'num', 25, 1),
    ('road_name', 'liveMapDataSP', 'roadName', 'str', 0, 1),
)
# After this long without a message of the service (s), its series turn to None. Long enough for a qlog (carState
# 10 Hz, longitudinalPlanSP 2 Hz, speedVisionSP 1 Hz, liveMapDataSP 1 Hz).
GAP_SECONDS = {'carState': 0.5, 'selfdriveState': 0.5, 'selfdriveStateSP': 0.5, 'carControl': 0.5,
               'longitudinalPlanSP': 1.5, 'longitudinalPlan': 1.5, 'speedVisionSP': 3.0, 'liveMapDataSP': 3.0}
ENCODE = ('roadEncodeIdx', 'narrowRoadEncodeIdx')
GPS = ('gpsLocationExternal', 'gpsLocation')
EVENT_FLAGS = ('enable', 'noEntry', 'warning', 'userDisable', 'softDisable', 'immediateDisable', 'preEnable',
               'permanent', 'overrideLateral', 'overrideLongitudinal')
COUNTED = tuple(sorted({item[1] for item in SERIES} | set(ENCODE) | set(GPS) | {'onroadEvents', 'onroadEventsSP'}))


def _get(obj, path: str):
    try:
        for part in path.split('.'):
            obj = getattr(obj, part)
    except Exception:  # a field this schema or this log does not have
        return None
    return obj


def _value(raw, kind: str, scale: float):
    if raw is None:
        return None
    if kind == 'bool':
        return bool(raw)
    if kind == 'str':
        return str(raw)
    try:
        return round(float(raw) * scale, 2)
    except (TypeError, ValueError):
        return None


def _messages(path: Path, cereal: Path, car: Path):
    import capnp
    capnp.remove_import_hook()
    schema = capnp.load(str(cereal / 'log.capnp'), imports=[str(cereal), str(car)])
    if path.name.endswith('.zst'):
        import zstandard
        with path.open('rb') as compressed:
            # stream_reader handles the concatenated zstd frames of a comma log.
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                data = reader.read()
    elif path.name.endswith('.bz2'):
        data = bz2.decompress(path.read_bytes())
    else:
        data = path.read_bytes()
    if not data:
        raise ValueError('empty log')
    yield from schema.Event.read_multiple_bytes(data)


def _events_of(which: str, payload) -> list[list]:
    items = _get(payload, 'events') if which == 'onroadEventsSP' else payload
    out = []
    for item in items or ():
        flags = [flag for flag in EVENT_FLAGS if _get(item, flag)]
        out.append([str(_get(item, 'name')), flags])
    return sorted(out)


def extract(path: Path, cereal: Path, car: Path) -> dict:
    by_service: dict[str, list[tuple]] = {}
    for spec in SERIES:
        by_service.setdefault(spec[1], []).append(spec)
    series: dict[str, list] = {spec[0]: [] for spec in SERIES}
    last_value: dict[str, object] = {}
    last_seen: dict[str, int] = {}
    events: list[list] = []
    onroad_last: dict[str, list] = {}
    frames = {'service': None, 'ids': [], 'sof': []}
    gps: list[list] = []
    gps_service = None
    counts: dict[str, int] = {}
    meta: dict = {}
    first = last = None
    total = 0

    def put(name: str, t: int, value, tolerance: float) -> None:
        before = last_value.get(name, '<none>')
        if before != '<none>' and (before == value or (
                value is not None and before is not None and tolerance and abs(value - before) < tolerance)):
            return
        series[name].append([t, value])
        last_value[name] = value

    for message in _messages(path, cereal, car):
        try:
            which = message.which()
        except Exception:  # a service this schema does not know
            continue
        t = int(message.logMonoTime)
        if which == 'initData':
            init = message.initData
            meta['build'] = {'version': _get(init, 'version'), 'branch': _get(init, 'gitBranch'),
                             'commit': _get(init, 'gitCommit'), 'device': str(_get(init, 'deviceType'))}
            continue
        total += 1
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
        if which in COUNTED:
            counts[which] = counts.get(which, 0) + 1
        if which == 'carParams':
            params = message.carParams
            meta['car'] = {'brand': _get(params, 'brand'), 'fingerprint': _get(params, 'carFingerprint'),
                           'dashcam_only': _get(params, 'dashcamOnly'),
                           'openpilot_longitudinal': _get(params, 'openpilotLongitudinalControl')}
            continue
        specs = by_service.get(which)
        if specs:
            payload = getattr(message, which)
            gap = GAP_SECONDS[which]
            seen = last_seen.get(which)
            if seen is not None and t - seen > gap * 1e9:
                for spec in specs:
                    put(spec[0], seen + int(gap * 1e9), None, 0)
            last_seen[which] = t
            for name, _service, field, kind, tolerance, scale in specs:
                put(name, t, _value(_get(payload, field), kind, scale), tolerance)
            if which == 'carState':
                for button in _get(payload, 'buttonEvents') or ():
                    events.append([t, 'button', str(_get(button, 'type')), bool(_get(button, 'pressed'))])
            elif which == 'speedVisionSP' and _get(payload, 'signRead'):
                events.append([t, 'sign', last_value.get('vis_limit')])
            continue
        if which in ('onroadEvents', 'onroadEventsSP'):
            now = _events_of(which, getattr(message, which))
            if onroad_last.get(which) != now:
                events.append([t, 'onroad', which, now])
                onroad_last[which] = now
        elif which in ENCODE:
            if frames['service'] is None:
                frames['service'] = which
            if frames['service'] == which:
                index = message.roadEncodeIdx if which == 'roadEncodeIdx' else message.narrowRoadEncodeIdx
                sof = _get(index, 'timestampSof') or _get(index, 'timestampEof') or t
                frames['ids'].append(int(_get(index, 'segmentId') or 0))
                frames['sof'].append(int(sof))
        elif which in GPS:
            if gps_service is None or (which == 'gpsLocationExternal' and gps_service != which):
                if which == 'gpsLocationExternal' and gps_service == 'gpsLocation':
                    gps.clear()
                gps_service = which
            if which != gps_service:
                continue
            fix = getattr(message, which)
            latitude, longitude = _get(fix, 'latitude'), _get(fix, 'longitude')
            has_fix = _get(fix, 'hasFix')
            if (has_fix is False) or not (latitude or longitude):
                continue
            if gps and t - gps[-1][0] < 9e8:
                continue
            gps.append([t, round(float(latitude), 6), round(float(longitude), 6),
                        int(_get(fix, 'unixTimestampMillis') or 0) or None])
    if total == 0:
        raise ValueError('no messages in the log')
    # The services that stopped before the end of the log: None from their last message on.
    for service, specs in by_service.items():
        seen = last_seen.get(service)
        if seen is not None and last - seen > GAP_SECONDS[service] * 1e9:
            for spec in specs:
                put(spec[0], seen + int(GAP_SECONDS[service] * 1e9), None, 0)
    events.sort(key=lambda item: item[0])
    return {'extract_version': EXTRACT_VERSION, 'span': [first, last], 'messages': total, 'counts': counts,
            'meta': meta, 'series': {name: points for name, points in series.items() if points},
            'events': events, 'frames': frames, 'gps': gps, 'gps_service': gps_service}


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print('usage: python -m nnslr_tools.drive_extract <cereal dir> <car dir> <log>', file=sys.stderr)
        return 2
    try:
        result = extract(Path(argv[3]), Path(argv[1]), Path(argv[2]))
    except Exception as exc:  # reported to the parent as the reason the log is unreadable
        print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)[:300]}))
        return 1
    sys.stdout.write(json.dumps(result, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
