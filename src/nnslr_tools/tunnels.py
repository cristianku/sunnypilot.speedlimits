"""The tunnels of a drive, from the map and the comma's GPS: marks of the kind "tunnel" in Speed review where the map is
sure, written without a person (asked by Cristian on 8 Nov 2026: "mettili già definitivi dove sei sicuro").

``nnslr tunnels --all`` reads the comma's GPS fixes of each scanned drive (drive_logs: [t_ns, lat, lon] on the comma's
clock), the OpenStreetMap roads near the drives with their ``tunnel`` tag (``derived/osm/roads.jsonl``,
scripts/osm_roads.py) and the dashboard speed (drive_logs.dashboard_speed). A tunnel is found two ways:

- the GPS is lost (no fix for GAP_S or more) and the straight line from the last fix to the next one runs along a road
  of the map in a tunnel for at least SURE_SHARE of its length (within NEAR_M, in its direction). The tunnel starts and
  ends where the line meets the tunnel's road, placed in time by the distance the car drove (the dashboard speed
  integrated; the line is shorter than a bent road, so its metres are stretched to the distance driven). A bent tunnel
  leaves the line in its middle (the Bözbergtunnel, 3.7 km): the hole is one too when both its ends are within PORTAL_M
  of the same tunnel of the map (one way, or ways of one name);
- the GPS still has fixes (a short tunnel, a gallery) and our road (map_context.our_road, the closest along the heading)
  is a tunnel of the map for at least MIN_S.

Stretches closer than JOIN_S are one tunnel. The marks go in Speed review as a person's taps would (speed_review
tap_road, start and end, kind "tunnel", reviewer REVIEWER), in speed_history like every write. A person's word wins: no
mark is written over a mark of the kind already there, nor where a person removed one (road_removed), nor before the
kind's "checked to here". Running it again adds only what is new. The map is never an input of the car: the marks are
labels, for the brain (its road head) and for the network of the road types.

OpenStreetMap data © OpenStreetMap contributors (ODbL).
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path

from nnslr_tools import drive_logs
from nnslr_tools.map_context import OUR_ROAD_M, RoadMap, _angle, _xy, our_road

KIND = 'tunnel'
LABEL = 'Tunnel'
REVIEWER = 'osm'
NS = 1_000_000_000
# A GPS lost this long is a hole a tunnel may explain.
GAP_S = 3.0
# A line from fix to fix this short is the car standing still: no tunnel to find.
MIN_LINE_M = 50.0
# The line is sampled every STEP_M; a sample is in the tunnel when a road of the map in a tunnel lies within NEAR_M of
# it along the line's direction (ANGLE degrees, either way: a tube of the other carriageway is a tunnel too).
STEP_M = 10.0
NEAR_M = 60.0
ANGLE = 35.0
# The share of the line in a tunnel for the hole to be one.
SURE_SHARE = 0.7
# A fix this close to a tunnel of the map is at its portal.
PORTAL_M = 150.0
# With fixes: our road a tunnel this long at least.
MIN_S = 2.0
JOIN_S = 3.0
# The tags of a road under something: OSM's tunnel=yes, building_passage, avalanche_protector...; not these.
NOT_TUNNEL = ('no', 'culvert', 'flooded')


def is_tunnel(way: dict) -> bool:
    value = way.get('tunnel')
    return bool(value) and value not in NOT_TUNNEL


def tunnel_map(road_map: RoadMap) -> RoadMap:
    """The roads of the map in a tunnel, as a map of their own."""
    return RoadMap([way for way in road_map.ways if is_tunnel(way)])


def _heading(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    x, y = _xy(b[0], b[1], a[0], a[1])
    if math.hypot(x, y) < 1.0:
        return None
    return math.degrees(math.atan2(x, y)) % 360.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(*_xy(b[0], b[1], a[0], a[1]))


class Driven:
    """The distance the car drove from a time, by the dashboard speed: [t_ns, km/h] steps."""

    def __init__(self, speed: list) -> None:
        self.times = [int(point[0]) for point in speed]
        self.metres = [0.0]
        for (t, kph), (t_next, _) in zip(speed, speed[1:]):
            self.metres.append(self.metres[-1] + (kph or 0.0) / 3.6 * (t_next - t) / NS)
        self.kph = [point[1] or 0.0 for point in speed]

    def at(self, t: int) -> float:
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            return 0.0
        return self.metres[i] + self.kph[i] / 3.6 * (t - self.times[i]) / NS

    def time_of(self, metres: float, t0: int, t1: int) -> int:
        """The time between t0 and t1 at which the car had driven these metres since t0 (linear in time when the
        speed says nothing)."""
        start, total = self.at(t0), self.at(t1) - self.at(t0)
        if total <= 1.0:
            return int(t0 + (t1 - t0) * max(0.0, min(1.0, metres / max(1.0, metres))))
        lo, hi = t0, t1
        for _ in range(40):
            mid = (lo + hi) // 2
            if self.at(mid) - start < metres:
                lo = mid
            else:
                hi = mid
        return hi


def _along_tunnel(tunnels: RoadMap, lat: float, lon: float, heading: float) -> bool:
    for item in tunnels.nearest(lat, lon, NEAR_M).values():
        if min(_angle(item['bearing'], heading), _angle(item['bearing'] + 180.0, heading)) <= ANGLE:
            return True
    return False


def same_tunnel(tunnels: RoadMap, a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Both fixes are at a portal of the same tunnel of the map: one way, or ways of one name."""
    def near(point):
        ways = tunnels.nearest(point[0], point[1], PORTAL_M)
        return set(ways), {tunnels.ways[w].get('name') for w in ways} - {None}
    (ways_a, names_a), (ways_b, names_b) = near(a), near(b)
    return bool(ways_a & ways_b or names_a & names_b)


def hole_tunnel(tunnels: RoadMap, driven: Driven, a: list, b: list) -> tuple[int, int] | None:
    """The tunnel that explains the GPS lost between fix a and fix b ([t_ns, lat, lon]), as (start_ns, end_ns), or None
    where the map has no tunnel along the line for SURE_SHARE of it."""
    pa, pb = (a[1], a[2]), (b[1], b[2])
    line = _distance(pa, pb)
    heading = _heading(pa, pb)
    if line < MIN_LINE_M or heading is None:
        return None
    steps = max(2, int(line // STEP_M) + 1)
    inside = []
    for i in range(steps + 1):
        f = i / steps
        inside.append(_along_tunnel(tunnels, pa[0] + f * (pb[0] - pa[0]), pa[1] + f * (pb[1] - pa[1]), heading))
    if sum(inside) / len(inside) < SURE_SHARE and not same_tunnel(tunnels, pa, pb):
        return None
    if not any(inside):
        return None
    first = inside.index(True) / steps
    last = (len(inside) - 1 - inside[::-1].index(True)) / steps
    total = driven.at(int(b[0])) - driven.at(int(a[0]))
    if total <= line * 0.5:
        # The speed does not say how far the car went: the time of the hole, shared as the line.
        span = int(b[0]) - int(a[0])
        return int(a[0]) + int(first * span), int(a[0]) + int(last * span)
    return driven.time_of(first * total, int(a[0]), int(b[0])), driven.time_of(last * total, int(a[0]), int(b[0]))


def fix_tunnels(road_map: RoadMap, gps: list) -> list[tuple[int, int]]:
    """The stretches where the GPS has fixes and our road is a tunnel of the map, at most one fix a second."""
    found: list[list[int]] = []
    last_t = None
    for i in range(1, len(gps) - 1):
        t = int(gps[i][0])
        if last_t is not None and t - last_t < NS:
            continue
        last_t = t
        before, after = gps[i - 1], gps[i + 1]
        if (int(after[0]) - int(before[0])) > GAP_S * NS:
            continue
        heading = _heading((before[1], before[2]), (after[1], after[2]))
        if heading is None:
            continue
        ours = our_road(road_map.nearest(gps[i][1], gps[i][2], OUR_ROAD_M), heading)
        if ours is not None and is_tunnel(road_map.ways[ours]):
            if found and t - found[-1][1] <= 1.5 * NS:
                found[-1][1] = t
            else:
                found.append([t, t])
    return [(start, end + NS // 2) for start, end in found if end - start >= MIN_S * NS]


def drive_tunnels(road_map: RoadMap, merged: dict, tunnels: RoadMap | None = None) -> list[tuple[int, int]]:
    """The tunnels of a scanned drive (drive_logs' merged logs) on the comma's clock: [(start_ns, end_ns)], in order."""
    tunnels = tunnel_map(road_map) if tunnels is None else tunnels
    gps = sorted((list(point[:3]) for point in merged.get('gps') or [] if point[1] is not None and point[2] is not None),
                 key=lambda point: point[0])
    if len(gps) < 2 or not tunnels.ways:
        return []
    driven = Driven(drive_logs.dashboard_speed(merged))
    found = fix_tunnels(road_map, gps)
    for a, b in zip(gps, gps[1:]):
        if int(b[0]) - int(a[0]) >= GAP_S * NS:
            hole = hole_tunnel(tunnels, driven, a, b)
            if hole is not None and hole[1] > hole[0]:
                found.append(hole)
    found.sort()
    joined: list[list[int]] = []
    for start, end in found:
        if joined and start - joined[-1][1] <= JOIN_S * NS:
            joined[-1][1] = max(joined[-1][1], end)
        else:
            joined.append([start, end])
    return [(start, end) for start, end in joined if end - start >= MIN_S * NS]


def write_tunnels(store, route: str, found: list[tuple[int, int]], span: tuple[int, int]) -> dict:
    """Writes the tunnels found as marks of the kind KIND in Speed review (speed_review.SpeedReviewStore), where a
    person said nothing there: returns {written, kept} (kept: left out for a person's mark, removal or check)."""
    from nnslr_tools.speed_review import SpeedReviewError
    kind = next((row['kind'] for row in store.road_kinds() if row['kind'] == KIND or row['label'].lower() == KIND), None)
    if kind is None:
        kind = store.add_road_kind(LABEL, REVIEWER)
    removed = [(row['start_ns'], row['end_ns']) for row in store.history(route)
               if row['action'] == 'road_removed' and row['state'] == kind]
    checked = store.road_checked(route).get(kind)
    written = kept = 0
    for start, end in found:
        start, end = max(start, span[0]), min(end, span[1])
        if end - start < MIN_S * NS:
            continue
        marks = [row for row in store.road(route) if row['kind'] == kind]
        clash = any(row['start_ns'] < end and (row['end_ns'] is None or start < row['end_ns']) for row in marks)
        if (clash or any(a < end and start < b for a, b in removed) or (checked is not None and start < checked)):
            kept += 1
            continue
        try:
            store.tap_road(route, kind, 'start', start, span[1], store.revision(route), REVIEWER)
            store.tap_road(route, kind, 'end', end, span[1], store.revision(route), REVIEWER)
            written += 1
        except SpeedReviewError:
            kept += 1
    return {'written': written, 'kept': kept}
