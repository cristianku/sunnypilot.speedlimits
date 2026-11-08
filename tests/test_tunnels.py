"""The tunnels of the map: a lost GPS along a tunnel is one, the same hole beside it is not."""
from nnslr_tools import tunnels
from nnslr_tools.map_context import RoadMap

NS = 1_000_000_000


def _road(lon0, lon1, tunnel):
    return {'id': 1, 'highway': 'motorway', 'tunnel': tunnel, 'nodes': [[47.0, lon0], [47.0, lon1]]}


def test_a_hole_along_a_tunnel_is_one():
    road_map = RoadMap([_road(8.0, 8.02, None), _road(8.0046, 8.0151, 'yes')])
    lon = lambda t: 8.0 + t * 25.0 / 75900.0
    gps = [[t * NS, 47.0, lon(t), 0] for t in list(range(0, 11)) + list(range(50, 61))]
    found = tunnels.drive_tunnels(road_map, {'gps': gps, 'series': {'v_ego': [[t * NS, 25.0] for t in range(61)]}})
    assert len(found) == 1
    start, end = found[0]
    assert 10 * NS <= start <= 16 * NS and 40 * NS <= end <= 50 * NS


def test_a_hole_off_any_tunnel_is_none():
    road_map = RoadMap([_road(8.0, 8.02, None), _road(8.0046, 8.0151, 'yes')])
    gps = [[t * NS, 47.01, 8.0 + t * 25.0 / 75900.0, 0] for t in list(range(0, 11)) + list(range(50, 61))]
    assert tunnels.drive_tunnels(road_map, {'gps': gps, 'series': {'v_ego': [[t * NS, 25.0] for t in range(61)]}}) == []
