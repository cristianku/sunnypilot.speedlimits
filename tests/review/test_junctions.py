"""The intersections the car drives through, from the map (nnslr_tools.junctions): the rules, the labels of the
frames and a person's corrections (made on Data › Roads until 7 Nov 2026)."""
import json

import pytest

from nnslr_tools.cli import main
from nnslr_tools.junctions import (CorrectionError, Junctions, build_junctions, correct, corrected, corrections_path,
                                   is_public, labelled_junctions, load_junctions, radius, road_width)
from nnslr_tools.map_context import RoadMap

ROUTE = '0000002a--0123456789'
# Metres per degree at lat 47 (map_context._xy): east and north.
EAST = 111320.0 * 0.6819983600624985
NORTH = 110540.0


def _write_map(tmp_path, ways):
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    return RoadMap.load(tmp_path)


# A street going east along lat 47.0 with: a side street at lon 8.0 (a T junction); a service road at 8.001 and a
# private street at 8.0025 (not junctions); a road on a bridge over it at 7.998 sharing no node with it, with a street
# joining the bridge 4.4 m north of our street (a junction near the track, on another road).
STREET = [{'id': 1, 'highway': 'residential', 'name': 'Main', 'junction': None,
           'nodes': [[47.0, 7.996], [47.0, 8.0], [47.0, 8.001], [47.0, 8.0025], [47.0, 8.005]]},
          {'id': 2, 'highway': 'residential', 'name': 'Side', 'nodes': [[47.0, 8.0], [47.001, 8.0]]},
          {'id': 3, 'highway': 'service', 'nodes': [[47.0, 8.001], [46.999, 8.001]]},
          {'id': 4, 'highway': 'residential', 'access': 'private', 'nodes': [[47.0, 8.0025], [47.001, 8.0025]]},
          {'id': 5, 'highway': 'primary', 'bridge': 'yes', 'nodes': [[46.999, 7.998], [47.00004, 7.998], [47.001, 7.998]]},
          {'id': 6, 'highway': 'tertiary', 'nodes': [[47.00004, 7.998], [47.0005, 7.9975]]}]


def _positions(count=40, speed=10.0, start=7.997):
    """A drive east along the street at `speed` m/s, one frame a second."""
    return [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.0,
             'longitude': start + index * speed / EAST, 'bearing_deg': 90.0, 'speed_mps': speed}
            for index in range(count)]


def _street_drive(tmp_path, count=40):
    road_map = _write_map(tmp_path, STREET)
    return build_junctions(tmp_path, ROUTE, _positions(count), Junctions(road_map))


def test_public_roads_widths_and_the_radius():
    assert is_public({'highway': 'residential'}) and is_public({'highway': 'primary_link'})
    for way in ({'highway': 'service'}, {'highway': 'track'}, {'highway': 'footway'}, {'highway': 'cycleway'},
                {'highway': 'residential', 'access': 'private'}, {'highway': 'tertiary', 'motor_vehicle': 'no'}):
        assert not is_public(way), way
    assert road_width({'highway': 'residential'}) == 6.5
    assert road_width({'highway': 'primary', 'lanes': '4'}) == 13.0
    assert road_width({'highway': 'primary_link'}) == 3.25
    # Half the widest road and the setback, between 10 and 20 m.
    assert radius(6.5) == 10.0 and radius(13.0) == 10.5 and radius(19.5) == 13.75 and radius(60.0) == 20.0


def test_a_t_junction_is_one_service_roads_private_streets_and_bridges_are_none(tmp_path):
    result = _street_drive(tmp_path)
    assert (result['junctions'], result['passed'], result['rejected'], result['without_frame_inside']) == (1, 1, 1, 0)
    record = load_junctions(tmp_path, ROUTE)
    passage = record['passages'][0]
    assert (passage['kind'], passage['arms'], passage['radius_m'], passage['highways']) == (
        'crossing', 3, 10.0, ['residential'])
    # The node is 300 m * 0.759... = 227.8 m along: 10 m before it to 10 m after it, at 10 m/s.
    assert passage['t_in'] == pytest.approx(21.78, abs=0.02) and passage['t_out'] == pytest.approx(23.78, abs=0.02)
    frames = record['frames']
    assert [i for i, frame in enumerate(frames) if frame['in_junction']] == [22, 23]
    assert [i for i, frame in enumerate(frames) if frame['passed']] == [24]
    assert passage['enter_key'] == '0/22' and passage['passed_key'] == '0/24'
    assert frames[24]['junction'] == passage['id'] and frames[10]['junction'] is None
    assert frames[30]['distance_m'] == 300.0
    # The junction on the bridge: near the track, on another road.
    assert record['rejected'][0]['why'] == 'near, but the car is on another road'
    assert record['rejected'][0]['closest_m'] == pytest.approx(4.4, abs=0.1)


def test_a_roundabout_is_one_intersection_from_entry_to_exit(tmp_path):
    lat, lon = 47.01, 8.0
    dlat, dlon = 15 / NORTH, 15 / EAST
    north, west, south, east = [lat + dlat, lon], [lat, lon - dlon], [lat - dlat, lon], [lat, lon + dlon]
    ways = [{'id': 1, 'highway': 'residential', 'junction': 'roundabout', 'nodes': [north, west, south, east, north]},
            {'id': 2, 'highway': 'residential', 'nodes': [[47.009, 8.0], south]},
            {'id': 3, 'highway': 'residential', 'nodes': [north, [47.011, 8.0]]}]
    junctions = Junctions(_write_map(tmp_path, ways))
    assert len(junctions.clusters) == 1 and junctions.clusters[0]['kind'] == 'roundabout'
    assert len(junctions.clusters[0]['nodes']) == 2 and len(junctions.clusters[0]['area']) == 4
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index,
                  'latitude': 47.009 + index * 8.0 / NORTH, 'longitude': lon, 'bearing_deg': 0.0, 'speed_mps': 8.0}
                 for index in range(26)]
    build_junctions(tmp_path, ROUTE, positions, junctions)
    record = load_junctions(tmp_path, ROUTE)
    assert len(record['passages']) == 1 and record['passages'][0]['kind'] == 'roundabout'
    # From 10 m before the south node (95.5 m along) to 10 m after the north one (125.5 m): at 8 m/s.
    assert [i for i, frame in enumerate(record['frames']) if frame['in_junction']] == list(range(11, 17))
    assert [i for i, frame in enumerate(record['frames']) if frame['passed']] == [17]


def test_splits_interchanges_and_staggered_crossings(tmp_path):
    ways = [
        # A road splitting into its two one-way carriageways: not an intersection.
        {'id': 1, 'highway': 'primary', 'name': 'Long', 'nodes': [[47.02, 8.0], [47.021, 8.0]]},
        {'id': 2, 'highway': 'primary', 'name': 'Long', 'oneway': 'yes', 'nodes': [[47.021, 8.0], [47.022, 8.0001]]},
        {'id': 3, 'highway': 'primary', 'name': 'Long', 'oneway': 'yes', 'nodes': [[47.022, 7.9999], [47.021, 8.0]]},
        # A motorway's exit ramp leaving it: grade separated, not an intersection.
        {'id': 4, 'highway': 'motorway', 'nodes': [[47.03, 8.0], [47.031, 8.0], [47.032, 8.0]]},
        {'id': 5, 'highway': 'motorway_link', 'nodes': [[47.031, 8.0], [47.0315, 8.0005]]},
        # Two side streets 15 m apart on either side of a street: one staggered crossing.
        {'id': 6, 'highway': 'residential', 'nodes': [[47.04, 7.999], [47.04, 8.0], [47.04, 8.0002], [47.04, 8.001]]},
        {'id': 7, 'highway': 'residential', 'nodes': [[47.04, 8.0], [47.041, 8.0]]},
        {'id': 8, 'highway': 'residential', 'nodes': [[47.04, 8.0002], [47.039, 8.0002]]},
    ]
    junctions = Junctions(_write_map(tmp_path, ways))
    assert (47.021, 8.0) not in junctions.junction_nodes and (47.031, 8.0) not in junctions.junction_nodes
    assert junctions.junction_nodes == {(47.04, 8.0), (47.04, 8.0002)}
    assert len(junctions.clusters) == 1 and junctions.clusters[0]['nodes'] == [(47.04, 8.0), (47.04, 8.0002)]


def test_the_gps_stopping_inside_labels_no_passed_and_asks_for_a_look(tmp_path):
    _street_drive(tmp_path, count=24)
    record = labelled_junctions(tmp_path, ROUTE)
    passage = record['passages'][0]
    assert not passage['complete'] and passage['passed_key'] is None
    assert not any(frame['passed'] for frame in record['frames'])
    mark = next(mark for mark in record['marks'] if mark['label'] == 'junction')
    assert mark['passed'] is None and mark['doubt'].startswith('The GPS stops inside')


def test_cli_junctions_all(tmp_path, capsys):
    _write_map(tmp_path, [{**way, 'junction': None} for way in STREET])
    gps = tmp_path / 'derived' / 'gps' / ROUTE / 'positions.jsonl'
    gps.parent.mkdir(parents=True)
    gps.write_text(''.join(json.dumps(line) + '\n' for line in _positions()))
    assert main(['junctions', '--all', '--data-root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['junctions'] == 1
    # A map read before the junction tags were kept: read it again.
    _write_map(tmp_path, [{key: value for key, value in way.items() if key != 'junction'} for way in STREET])
    assert main(['junctions', '--all', '--data-root', str(tmp_path)]) == 2
    assert 'again' in capsys.readouterr().err


# -- a person's corrections -----------------------------------------------------------------------------------------
def _labels(record):
    frames = record['frames']
    return ([i for i, f in enumerate(frames) if f['in_junction']], [i for i, f in enumerate(frames) if f['passed']])


def test_corrections_drop_confirm_add_and_the_last_wins(tmp_path):
    _street_drive(tmp_path)
    map_record = load_junctions(tmp_path, ROUTE)
    assert corrected(map_record, [])['frames'] == map_record['frames']
    marks = labelled_junctions(tmp_path, ROUTE)['marks']
    assert [(mark['label'], mark['by'], mark['first'], mark['passed']) for mark in marks] == [
        ('maybe', 'map', 8, None), ('junction', 'map', 22, 24)]
    assert marks[0]['doubt'].startswith('The map has a junction 4 m from the track') and marks[1]['doubt'] is None

    # Not a junction, on its "passed" frame: dropped, no label left.
    correct(tmp_path, ROUTE, '0/24', 'none', 'admin')
    record = labelled_junctions(tmp_path, ROUTE)
    assert record['passages'] == [] and _labels(record) == ([], [])
    assert [(mark['label'], mark['reviewed'], mark['reviewer']) for mark in record['marks']][1] == (
        'dropped', True, 'admin')
    # Junction here, after it: the last word wins, confirmed.
    correct(tmp_path, ROUTE, '0/23', 'here', 'admin')
    record = labelled_junctions(tmp_path, ROUTE)
    assert _labels(record) == ([22, 23], [24]) and record['passages'][0]['reviewed']
    # One the map missed: the frames within 10 m of the picture along the track, "passed" the next one.
    correct(tmp_path, ROUTE, '0/30', 'here', 'admin')
    record = labelled_junctions(tmp_path, ROUTE)
    assert _labels(record) == ([22, 23, 29, 30, 31], [24, 32])
    person = record['passages'][1]
    assert (person['by'], person['id'], person['passed_key'], person['frames_inside']) == ('person', 'P0-30', '0/32', 3)
    # Not a junction on a person's: taken away again.
    correct(tmp_path, ROUTE, '0/31', 'none', 'admin')
    assert _labels(labelled_junctions(tmp_path, ROUTE)) == ([22, 23], [24])
    # The bridge's junction: "none" settles it (never a label); "here" makes it a person's junction.
    correct(tmp_path, ROUTE, '0/8', 'none', 'admin')
    maybe = labelled_junctions(tmp_path, ROUTE)['marks'][0]
    assert (maybe['label'], maybe['reviewed'], maybe['doubt']) == ('maybe', True, None)
    correct(tmp_path, ROUTE, '0/7', 'here', 'admin')
    record = labelled_junctions(tmp_path, ROUTE)
    assert [mark['label'] for mark in record['marks']] == ['junction', 'junction']
    assert _labels(record) == ([6, 7, 8, 22, 23], [9, 24])

    # The map found again: the corrections are kept apart and still apply.
    lines = corrections_path(tmp_path, ROUTE).read_text().splitlines()
    assert len(lines) == 6 and all(json.loads(line)['reviewer'] == 'admin' for line in lines)
    _street_drive(tmp_path)
    assert corrections_path(tmp_path, ROUTE).read_text().splitlines() == lines
    assert load_junctions(tmp_path, ROUTE)['frames'] == map_record['frames']
    assert _labels(labelled_junctions(tmp_path, ROUTE)) == ([6, 7, 8, 22, 23], [9, 24])


def test_a_correction_names_a_frame_of_the_drive_and_a_choice(tmp_path):
    with pytest.raises(CorrectionError, match='not been found'):
        correct(tmp_path, ROUTE, '0/1', 'here', 'admin')
    _street_drive(tmp_path)
    for key, choice in (('9/9', 'here'), (3, 'here'), (None, 'none'), ('../0', 'none'), ('0/1', 'yes'),
                        ('0/1', None), ('0/1', 'dropped')):
        with pytest.raises(CorrectionError):
            correct(tmp_path, ROUTE, key, choice, 'admin')
    with pytest.raises(ValueError):
        correct(tmp_path, '../etc', '0/1', 'here', 'admin')
    assert not corrections_path(tmp_path, ROUTE).exists()
    # A line that cannot be read is skipped, the others still count.
    path = corrections_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    good = {'frame_key': '0/23', 'junction': 'none', 'reviewer': 'admin', 'at': '2026-10-29T12:00:00+01:00'}
    path.write_text('\n'.join(['{"frame_key": "0/1", "junction":', '[1]', json.dumps({**good, 'junction': 'maybe'}),
                               json.dumps({**good, 'frame_key': 7}), json.dumps(good)]) + '\n')
    assert labelled_junctions(tmp_path, ROUTE)['passages'] == []
