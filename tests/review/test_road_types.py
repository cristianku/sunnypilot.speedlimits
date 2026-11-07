"""The road type of every frame from the map (nnslr_tools.road_types): the type of a way, the drive smoothed, the file."""
import json

from nnslr_tools.cli import main
from nnslr_tools.map_context import RoadMap
from nnslr_tools.road_types import (build_road_types, confirmed, correct, load_road_types, merge_lanes,
                                    review_counts, smooth, stretches, summary, way_type)

ROUTE = '0000002a--0123456789'


def test_way_type_from_kind_limit_and_highway():
    assert way_type({'highway': 'motorway', 'maxspeed': '100'})[0] == 'motorway'
    assert way_type({'highway': 'motorway_link'})[0] == 'ramp'
    assert way_type({'highway': 'trunk_link'})[0] == 'ramp'
    assert way_type({'highway': 'trunk_link', 'maxspeed': '50'})[0] == 'town'
    assert way_type({'highway': 'trunk', 'maxspeed': '100'})[0] == 'expressway'
    # A trunk road at 50 runs through a town.
    assert way_type({'highway': 'trunk', 'maxspeed': '50'})[0] == 'town'
    # The kind of limit wins over the number: a rural road signed 50 stays rural.
    assert way_type({'highway': 'secondary', 'maxspeed': '50', 'maxspeed_type': 'CH:rural'})[0] == 'rural'
    assert way_type({'highway': 'tertiary', 'maxspeed_type': 'CH:urban'})[0] == 'town'
    # A 30 zone is a type of its own: sure from its kind, a guess from a bare 30; another zone is town.
    assert way_type({'highway': 'residential', 'maxspeed_type': 'CH:zone30'}) == ('zone30', 'limit CH:zone30')
    assert way_type({'highway': 'residential', 'maxspeed': '30'}) == ('zone30', 'maxspeed 30')
    assert way_type({'highway': 'residential', 'maxspeed': '20', 'maxspeed_type': 'CH:zone20'})[0] == 'town'
    assert way_type({'highway': 'residential', 'maxspeed': '30', 'maxspeed_type': 'CH:urban'})[0] == 'town'
    assert way_type({'highway': 'primary', 'maxspeed': 'CH:rural'})[0] == 'rural'
    assert way_type({'highway': 'primary', 'maxspeed': '50'}) == ('town', 'maxspeed 50')
    assert way_type({'highway': 'primary', 'maxspeed': '60'})[0] == 'rural'
    assert way_type({'highway': 'secondary', 'maxspeed': '80'})[0] == 'rural'
    assert way_type({'highway': 'residential'})[0] == 'town'
    # An ordinary road the map gives no limit: unknown, never guessed.
    assert way_type({'highway': 'unclassified'})[0] is None
    assert way_type({'highway': 'primary', 'maxspeed': 'signals'})[0] is None


def test_a_limit_in_mph_by_the_us_cut():
    # Up to 45 mph a town, 50 mph and more rural, by the US cut (US_TOWN_MPH); the reason keeps the unit.
    assert way_type({'highway': 'primary', 'maxspeed': '55 mph'}) == ('rural', 'maxspeed 55 mph')
    assert way_type({'highway': 'primary', 'maxspeed': '50 mph'}) == ('rural', 'maxspeed 50 mph')
    assert way_type({'highway': 'residential', 'maxspeed': '35mph'}) == ('town', 'maxspeed 35 mph')
    assert way_type({'highway': 'secondary', 'maxspeed': '40 mph'}) == ('town', 'maxspeed 40 mph')
    assert way_type({'highway': 'primary', 'maxspeed': '45 mph'}) == ('town', 'maxspeed 45 mph')
    assert way_type({'highway': 'residential', 'maxspeed': '25 mph'})[0] == 'town'
    # 30 mph is an ordinary US street, never a 30 zone.
    assert way_type({'highway': 'residential', 'maxspeed': '30 mph'}) == ('town', 'maxspeed 30 mph')
    # A trunk road at 45 mph is a town's arterial with signals (US 45 in Orland Park), not an expressway, and not
    # sure; at 55 mph it is an expressway.
    assert way_type({'highway': 'trunk', 'maxspeed': '35 mph'})[0] == 'town'
    assert way_type({'highway': 'trunk', 'maxspeed': '45 mph'}) == ('town', 'highway trunk in town')
    assert way_type({'highway': 'trunk', 'maxspeed': '55 mph'})[0] == 'expressway'
    # The kind of limit wins, as in Switzerland.
    assert way_type({'highway': 'primary', 'maxspeed': '55 mph', 'maxspeed_type': 'US-IL:urban'})[0] == 'town'
    # Two values or a unit of its own: no limit read.
    assert way_type({'highway': 'primary', 'maxspeed': '55 mph;45 mph'})[0] is None
    assert way_type({'highway': 'primary', 'maxspeed': '50 knots'})[0] is None


def test_smooth_fills_gaps_and_drops_slips():
    town, rural = 'town', 'rural'
    # A tunnel (unknown) between two rural stretches; a GPS slip of 2 frames to a town street.
    types = [rural] * 5 + [None] * 3 + [rural] * 5 + [town] * 2 + [rural] * 5
    result = smooth(types)
    assert [value for value, _ in result] == [rural] * 20
    assert [how for _, how in result][5:8] == ['filled'] * 3
    assert [how for _, how in result][13:15] == ['smoothed'] * 2
    # A real change of road stays: unknown between two different types is not filled.
    types = [rural] * 5 + [None] * 2 + [town] * 5
    assert [value for value, _ in smooth(types)] == types
    # Unknown at the start and the end has only one side: it stays unknown.
    assert smooth([None, town, town])[0] == (None, None)


def _map(tmp_path):
    # Two roads crossing: a motorway going north at lon 8.0, a town street going east at lat 47.001.
    ways = [{'id': 1, 'highway': 'motorway', 'maxspeed': '120', 'nodes': [[47.0, 8.0], [47.002, 8.0]]},
            {'id': 2, 'highway': 'secondary', 'maxspeed': '50', 'nodes': [[47.001, 7.999], [47.001, 8.001]]}]
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    return RoadMap.load(tmp_path)


def test_build_follows_heading_at_a_crossing(tmp_path):
    road_map = _map(tmp_path)
    positions = [
        # Driving north on the motorway, right over the town street: the heading says which road is ours.
        {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': 47.001, 'longitude': 8.0, 'bearing_deg': 2.0},
        {'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.0005, 'longitude': 8.0, 'bearing_deg': 0.0},
        # Driving east on the street.
        {'frame_key': '0/2', 'segment_index': 0, 'output_index': 2, 'latitude': 47.001, 'longitude': 8.0007, 'bearing_deg': 90.0},
        # No fix, then far from every road.
        {'frame_key': '0/3', 'segment_index': 0, 'output_index': 3, 'latitude': None, 'longitude': None, 'bearing_deg': None},
        {'frame_key': '0/4', 'segment_index': 0, 'output_index': 4, 'latitude': 47.5, 'longitude': 8.5, 'bearing_deg': 0.0},
    ]
    result = build_road_types(tmp_path, ROUTE, positions, road_map)
    assert result['types'] == {'motorway': 2, 'town': 1, 'unknown': 2}
    lines = load_road_types(tmp_path, ROUTE)
    # Written in the order of the drive.
    assert [line['frame_key'] for line in lines] == ['0/0', '0/1', '0/2', '0/3', '0/4']
    assert [line['road_type'] for line in lines] == ['motorway', 'motorway', 'town', None, None]
    assert lines[0]['way_id'] == 1 and lines[2]['way_id'] == 2 and lines[2]['basis'] == 'maxspeed 50'
    assert (lines[3]['basis'], lines[4]['basis']) == ('no gps', 'no road')
    assert [line['sure'] for line in lines] == [True, True, True, False, False]
    assert stretches(lines) == [{'type': 'motorway', 'from': '0/0', 'to': '0/1', 'frames': 2},
                                {'type': 'town', 'from': '0/2', 'to': '0/2', 'frames': 1},
                                {'type': None, 'from': '0/3', 'to': '0/4', 'frames': 2}]


def test_cli_road_types_all(tmp_path, capsys):
    _map(tmp_path)
    gps = tmp_path / 'derived' / 'gps' / ROUTE / 'positions.jsonl'
    gps.parent.mkdir(parents=True)
    gps.write_text(json.dumps({'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.0005,
                               'longitude': 8.0, 'bearing_deg': 180.0}) + '\n')
    assert main(['road-types', '--all', '--data-root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['types'] == {'motorway': 1}
    # Without the map of the roads it says how to make it.
    (tmp_path / 'derived' / 'osm' / 'roads.jsonl').unlink()
    assert main(['road-types', '--all', '--data-root', str(tmp_path)]) == 2
    assert 'osm_roads.py' in capsys.readouterr().err


def test_summary_names_each_drive_with_its_types_and_the_drives_left(tmp_path):
    from review_helpers import make_report

    from nnslr_tools.road_types import summary
    assert summary(tmp_path) == {'routes': [], 'made_at': None, 'drives_without': 0,
                                 'review': {'usable': 0, 'cant_tell': 0, 'to_review': 0},
                                 'maps': {'names': [], 'needed': [], 'outside': 0, 'nowhere': 0, 'max_gb': None,
                                          'checked_at': None}}
    # A prepared drive without road types yet.
    make_report(tmp_path, ROUTE)
    assert summary(tmp_path)['drives_without'] == 1
    road_map = _map(tmp_path)
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.0005,
                  'longitude': 8.0, 'bearing_deg': 0.0} for index in range(3)]
    positions.append({'frame_key': '0/3', 'segment_index': 0, 'output_index': 3, 'latitude': None, 'longitude': None,
                      'bearing_deg': None})
    build_road_types(tmp_path, ROUTE, positions, road_map)
    result = summary(tmp_path)
    assert result['drives_without'] == 0 and result['made_at'] is not None
    drive = result['routes'][0]
    assert (drive['route'], drive['frames']) == (ROUTE, 4)
    assert drive['types'] == {'town': 0, 'rural': 0, 'expressway': 0, 'motorway': 3, 'ramp': 0, 'parking': 0, 'zone30': 0,
                             'garage': 0, 'merge': 0, 'unknown': 1}
    assert drive['stretches'] == [{'type': 'motorway', 'from': '0/0', 'to': '0/2', 'frames': 3},
                                  {'type': None, 'from': '0/3', 'to': '0/3', 'frames': 1}]
    # Sure on the map or not, every frame is to review until a person says.
    assert (drive['reviewed'], drive['usable'], drive['cant_tell'], drive['to_review']) == (0, 0, 0, 4)
    assert result['review'] == {'usable': 0, 'cant_tell': 0, 'to_review': 4}
    # Read again when the file changes.
    build_road_types(tmp_path, ROUTE, positions[:2], road_map)
    assert summary(tmp_path)['routes'][0]['frames'] == 2


def test_weak_reasons_are_not_sure(tmp_path):
    # A road at 60 (in a town one time in three) and a ramp: labelled, but not sure.
    ways = [{'id': 1, 'highway': 'primary', 'maxspeed': '60', 'nodes': [[47.0, 8.0], [47.002, 8.0]]},
            {'id': 2, 'highway': 'motorway_link', 'nodes': [[47.1, 8.0], [47.102, 8.0]]}]
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    positions = [{'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': 47.001, 'longitude': 8.0, 'bearing_deg': 0.0},
                 {'frame_key': '0/1', 'segment_index': 0, 'output_index': 1, 'latitude': 47.101, 'longitude': 8.0, 'bearing_deg': 0.0}]
    build_road_types(tmp_path, ROUTE, positions, RoadMap.load(tmp_path))
    lines = load_road_types(tmp_path, ROUTE)
    assert [(line['road_type'], line['sure']) for line in lines] == [('rural', False), ('ramp', False)]


def test_the_limits_around_the_us_cut_are_not_sure(tmp_path):
    ways = [{'id': 1, 'highway': 'primary', 'maxspeed': '35 mph', 'nodes': [[40.0, -88.0], [40.002, -88.0]]},
            {'id': 2, 'highway': 'primary', 'maxspeed': '45 mph', 'nodes': [[40.1, -88.0], [40.102, -88.0]]},
            {'id': 3, 'highway': 'primary', 'maxspeed': '55 mph', 'nodes': [[40.2, -88.0], [40.202, -88.0]]},
            # A trunk at 45 mph, a town's arterial: never a sure expressway.
            {'id': 4, 'highway': 'trunk', 'maxspeed': '45 mph', 'nodes': [[40.3, -88.0], [40.302, -88.0]]}]
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': latitude,
                  'longitude': -88.0, 'bearing_deg': 0.0}
                 for index, latitude in enumerate((40.201, 40.001, 40.101, 40.301))]
    build_road_types(tmp_path, ROUTE, positions, RoadMap.load(tmp_path))
    lines = load_road_types(tmp_path, ROUTE)
    assert [(line['road_type'], line['basis'], line['sure']) for line in lines] == [
        ('rural', 'maxspeed 55 mph', True), ('town', 'maxspeed 35 mph', False), ('town', 'maxspeed 45 mph', False),
        ('town', 'highway trunk in town', False)]


def test_the_maps_of_the_server_and_those_the_drives_need(tmp_path):
    import os

    from nnslr_tools.road_types import maps_status
    folder = tmp_path / 'raw' / 'osm'
    folder.mkdir(parents=True)
    for stamp, name in ((1_700_000_000, 'switzerland-latest.osm.pbf'), (1_800_000_000, 'illinois-latest.osm.pbf'),
                        (1_750_000_000, 'baden-wuerttemberg-latest.osm.pbf')):
        (folder / name).write_bytes(b'pbf')
        os.utime(folder / name, (stamp, stamp))
    status = tmp_path / 'derived' / 'osm' / 'maps.json'
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({
        'checked_at': '2026-11-03T10:00:00+01:00', 'max_gb': 5.0,
        'maps': [{'file': 'switzerland-latest.osm.pbf', 'name': 'Switzerland'},
                 {'file': 'baden-wuerttemberg-latest.osm.pbf', 'name': 'Baden-Württemberg'}],
        'needed': [{'name': 'Alsace', 'file': 'alsace-latest.osm.pbf', 'bytes': 132_000_000, 'state': 'failed',
                    'reason': 'no network', 'drives': ['r1', 'r2']},
                   {'name': 'Michigan', 'file': 'michigan-latest.osm.pbf', 'bytes': 316_000_000, 'state': 'too big',
                    'reason': 'beyond', 'drives': ['r2', 'r3']},
                   # Downloaded by that run, or since: no longer needed.
                   {'name': 'Indiana', 'file': 'indiana-latest.osm.pbf', 'bytes': 1, 'state': 'downloaded',
                    'drives': ['r4']},
                   {'name': 'Illinois', 'file': 'illinois-latest.osm.pbf', 'bytes': 1, 'state': 'failed',
                    'drives': ['r5']}],
        'nowhere': ['r6']}))
    maps = maps_status(tmp_path)
    # Oldest first; a map the status has not seen yet by its file.
    assert maps['names'] == ['Switzerland', 'Baden-Württemberg', 'Illinois']
    assert [(item['name'], item['state'], item['drives']) for item in maps['needed']] == [
        ('Alsace', 'failed', 2), ('Michigan', 'too big', 2)]
    assert (maps['outside'], maps['nowhere'], maps['max_gb']) == (3, 1, 5.0)
    # A broken status: the maps of the folder only.
    status.write_text('{')
    assert maps_status(tmp_path)['names'] == ['Switzerland', 'Baden Wuerttemberg', 'Illinois']



def _seconds(count, speed, start=0):
    return [{'segment_index': (start + index) // 60, 'output_index': (start + index) % 60, 'speed_mps': speed}
            for index in range(count)]


def test_the_acceleration_lane_of_an_entry():
    # 10 s of ramp at 20 m/s, then 20 s of motorway: the last 50 m of the ramp (2 s) and the first 200 m of the
    # carriageway (11 s: 0 to 200 m) are the lane.
    types = ['town'] * 3 + ['ramp'] * 10 + ['motorway'] * 20
    lane = merge_lanes(types, _seconds(len(types), 20.0))
    assert [i for i, merge in enumerate(lane) if merge] == list(range(11, 24))
    # Onto an expressway too; never onto anything else, never at an exit.
    assert any(merge_lanes(['ramp'] * 5 + ['expressway'] * 5, _seconds(10, 20.0)))
    assert not any(merge_lanes(['ramp'] * 5 + ['rural'] * 5, _seconds(10, 20.0)))
    assert not any(merge_lanes(['motorway'] * 5 + ['ramp'] * 5, _seconds(10, 20.0)))
    # A missing second between the ramp and the carriageway: not a merge seen.
    positions = _seconds(5, 20.0) + _seconds(5, 20.0, start=30)
    assert not any(merge_lanes(['ramp'] * 5 + ['motorway'] * 5, positions))
    # Without the GPS speed the speed of the second before counts; slow traffic makes a lane of more seconds.
    slow = _seconds(30, 5.0)
    slow[14]['speed_mps'] = None
    assert sum(merge_lanes(['ramp'] * 10 + ['motorway'] * 20, slow)) == 10 + 20
    # A missing second inside the lane ends it there, on the carriageway and on the ramp before.
    gap = _seconds(16, 20.0) + _seconds(17, 20.0, start=40)
    assert [i for i, merge in enumerate(merge_lanes(types, gap)) if merge] == list(range(11, 16))
    gap = _seconds(12, 5.0) + _seconds(21, 5.0, start=40)
    assert [i for i, merge in enumerate(merge_lanes(types, gap)) if merge] == list(range(12, 33))


def test_a_short_ramp_between_two_stretches_of_motorway_is_a_slip_of_the_gps():
    # Passing a junction, the GPS on the link beside the carriageway for 6 s: no way off and back on, no lane.
    slip = smooth(['motorway'] * 20 + ['ramp'] * 6 + ['motorway'] * 20)
    assert [value for value, _ in slip] == ['motorway'] * 46
    assert not any(merge_lanes([value for value, _ in slip], _seconds(46, 25.0)))
    # A link of 12 s from one carriageway to another stays, with its lane.
    link = [value for value, _ in smooth(['motorway'] * 20 + ['ramp'] * 12 + ['motorway'] * 20)]
    assert link[20:32] == ['ramp'] * 12 and any(merge_lanes(link, _seconds(52, 25.0)))
    # A slip back to the carriageway inside a ramp goes first: the exit and the 13 s link stay whole.
    exit_ramp = [value for value, _ in smooth(['motorway'] * 30 + ['ramp'] * 5 + ['motorway'] * 2 + ['ramp'] * 20
                                              + ['town'] * 10)]
    assert exit_ramp == ['motorway'] * 30 + ['ramp'] * 27 + ['town'] * 10
    link = [value for value, _ in smooth(['motorway'] * 30 + ['ramp'] * 6 + ['motorway'] * 2 + ['ramp'] * 5
                                         + ['motorway'] * 30)]
    assert link == ['motorway'] * 30 + ['ramp'] * 13 + ['motorway'] * 30
    # A short ramp between two other roads keeps the usual MIN_RUN.
    assert [value for value, _ in smooth(['rural'] * 10 + ['ramp'] * 5 + ['rural'] * 10)][10:15] == ['ramp'] * 5


def test_build_labels_the_lane_after_an_entry_ramp(tmp_path):
    from nnslr_tools import road_type_data as data
    # A ramp north to (47.002, 8), then the motorway on: 20 m/s, a frame a second (0.00018 degrees of latitude).
    ways = [{'id': 1, 'highway': 'motorway_link', 'oneway': 'yes', 'nodes': [[47.0, 8.0], [47.002, 8.0]]},
            {'id': 2, 'highway': 'motorway', 'oneway': 'yes', 'nodes': [[47.002, 8.0], [47.02, 8.0]]}]
    path = tmp_path / 'derived' / 'osm' / 'roads.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(way) + '\n' for way in ways))
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index,
                  'latitude': 47.0001 + index * 0.00018, 'longitude': 8.0, 'bearing_deg': 0.0, 'speed_mps': 20.0}
                 for index in range(30)]
    build_road_types(tmp_path, ROUTE, positions, RoadMap.load(tmp_path))
    lines = load_road_types(tmp_path, ROUTE)
    kinds = [line['road_type'] for line in lines]
    assert kinds[0] == 'ramp' and kinds[-1] == 'motorway' and 'merge' in kinds
    first, last = kinds.index('merge'), len(kinds) - 1 - kinds[::-1].index('merge')
    assert set(kinds[first:last + 1]) == {'merge'} and 12 <= last - first + 1 <= 14
    # The ramp's metres weigh as the ramp, the carriageway's as the motorway.
    on_ramp, on_road = lines[first], lines[last]
    assert on_ramp['basis'] == 'acceleration lane (highway motorway_link)' and not on_ramp['sure']
    # The ramp's frames of the lane are its lead, for the measures to fold back into the ramp.
    assert on_ramp['lead'] is True and on_road['lead'] is False
    frames = data.route_frames(tmp_path, ROUTE)
    assert frames[first]['map_lead'] is True and frames[first]['lead'] is False
    assert on_road['basis'] == 'acceleration lane (highway motorway)' and on_road['sure']
    # Proposals: no label, and a person's confirmation makes them one, the lead kept.
    assert data.frame_label(on_ramp) == data.frame_label(on_road) == (-1, 0.0)
    correct(tmp_path, ROUTE, lines[first]['frame_key'], lines[last]['frame_key'], 'merge', 'someone')
    frames = data.route_frames(tmp_path, ROUTE)
    assert frames[first]['label'] == frames[last]['label'] == data.CLASS_INDEX['merge']
    assert frames[first]['weight'] == frames[last]['weight'] == 1.0
    assert frames[first]['lead'] is True and frames[last]['lead'] is False



def test_a_person_s_lane_takes_the_ramp_s_frames_as_its_lead():
    from nnslr_tools.road_types import corrected
    # The map: ramp, its lane (two frames of lead), the motorway. A person starts the lane two frames earlier.
    kinds = ['ramp'] * 4 + ['merge'] * 4 + ['motorway'] * 2
    lines = [{'frame_key': f'0/{i}', 'road_type': kind, 'lead': kind == 'merge' and i < 6} for i, kind in enumerate(kinds)]
    lane = corrected(lines, [{'from': '0/2', 'to': '0/7', 'road_type': 'merge', 'reviewer': 'p'}])
    assert [line['lead'] for line in lane[2:8]] == [True] * 4 + [False] * 2
    # A frame of the lane said motorway by the person is no lead any more.
    assert corrected(lines, [{'from': '0/4', 'to': '0/4', 'road_type': 'motorway', 'reviewer': 'p'}])[4]['lead'] is False

# -- a person's corrections (Data › Roads, until 7 Nov 2026) ---------------------------------------------------------
def _drive_of(tmp_path, count=6):
    """A drive of `count` frames on the motorway of _map, the last two without GPS."""
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.0005,
                  'longitude': 8.0, 'bearing_deg': 0.0} for index in range(count - 2)]
    positions += [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': None,
                   'longitude': None, 'bearing_deg': None} for index in range(count - 2, count)]
    road_map = _map(tmp_path)
    build_road_types(tmp_path, ROUTE, positions, road_map)
    return positions, road_map


def test_corrections_over_the_map_the_last_wins_and_a_new_map_keeps_them(tmp_path):
    from nnslr_tools.road_types import corrections_path, correct, map_road_types, summary
    positions, road_map = _drive_of(tmp_path)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 4 + [None] * 2
    # From the end to the start works too; the last correction covering a frame wins.
    correct(tmp_path, ROUTE, '0/5', '0/3', 'town', 'admin')
    correct(tmp_path, ROUTE, '0/4', '0/4', None, 'admin')
    lines = load_road_types(tmp_path, ROUTE)
    assert [line['road_type'] for line in lines] == ['motorway'] * 3 + ['town', None, 'town']
    assert [line.get('reviewed', False) for line in lines] == [False] * 3 + [True] * 3
    assert (lines[3]['basis'], lines[3]['sure'], lines[3]['reviewer']) == ('reviewed', True, 'admin')
    # The map stays as it was, the corrections apart, one line each.
    assert [line['road_type'] for line in map_road_types(tmp_path, ROUTE)] == ['motorway'] * 4 + [None] * 2
    kept = [json.loads(line) for line in corrections_path(tmp_path, ROUTE).read_text().splitlines()]
    assert [(line['from'], line['to'], line['road_type']) for line in kept] == [('0/5', '0/3', 'town'), ('0/4', '0/4', None)]
    assert all(line['reviewer'] == 'admin' and line['at'] for line in kept)
    drive = summary(tmp_path)['routes'][0]
    assert drive['reviewed'] == 3 and drive['types'] == {'town': 2, 'rural': 0, 'expressway': 0, 'motorway': 3,
                                                         'ramp': 0, 'parking': 0, 'zone30': 0, 'garage': 0,
                                                         'merge': 0, 'unknown': 1}

    # Find again: the map is written anew, the corrections are not touched and still win.
    build_road_types(tmp_path, ROUTE, positions, road_map)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 3 + ['town', None, 'town']
    assert len(corrections_path(tmp_path, ROUTE).read_text().splitlines()) == 2

    # A correction whose frames left the drive (a shorter map) is skipped, not applied elsewhere.
    correct(tmp_path, ROUTE, '0/0', '0/5', 'rural', 'admin')
    build_road_types(tmp_path, ROUTE, positions[:4], road_map)
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['motorway'] * 4


def test_what_a_person_did_of_a_drive_is_counted_for_the_page(tmp_path):
    positions, road_map = _drive_of(tmp_path)
    assert review_counts(load_road_types(tmp_path, ROUTE)) == {'usable': 0, 'cant_tell': 0, 'to_review': 6}
    assert confirmed(load_road_types(tmp_path, ROUTE)[0]) is False  # motorway, sure on the map
    correct(tmp_path, ROUTE, '0/0', '0/1', 'motorway', 'someone')  # the proposal confirmed
    correct(tmp_path, ROUTE, '0/2', '0/2', 'rural', 'someone')  # corrected
    correct(tmp_path, ROUTE, '0/3', '0/3', None, 'someone')  # can't tell
    lines = load_road_types(tmp_path, ROUTE)
    assert [confirmed(line) for line in lines] == [True, True, True, False, False, False]
    assert review_counts(lines) == {'usable': 3, 'cant_tell': 1, 'to_review': 2}
    # The same for the list, and after the map is read again.
    build_road_types(tmp_path, ROUTE, positions, road_map)
    drive = summary(tmp_path)
    assert drive['review'] == {'usable': 3, 'cant_tell': 1, 'to_review': 2}
    assert (drive['routes'][0]['usable'], drive['routes'][0]['to_review']) == (3, 2)


def test_a_correction_names_frames_of_the_drive_and_a_type(tmp_path):
    import pytest

    from nnslr_tools.road_types import CorrectionError, corrections_path, correct
    _drive_of(tmp_path)
    for start, end, kind in (('0/9', '0/1', 'town'), ('0/0', '1/0', 'town'), (0, '0/1', 'town'), ('0/0', None, 'town'),
                             ('../0', '0/1', 'town'), ('0/0', '0/1', 'highway'), ('0/0', '0/1', 'unknown'),
                             ('0/0', '0/1', 3)):
        with pytest.raises(CorrectionError):
            correct(tmp_path, ROUTE, start, end, kind, 'admin')
    with pytest.raises(ValueError):
        correct(tmp_path, '../etc', '0/0', '0/1', 'town', 'admin')
    # A drive without road types has no frame to correct.
    with pytest.raises(CorrectionError):
        correct(tmp_path, '0000002b--0123456789', '0/0', '0/0', 'town', 'admin')
    assert not corrections_path(tmp_path, ROUTE).exists()


def test_the_drives_page_is_gone_and_the_list_stays(tmp_path):
    """Data › Roads went on 7 Nov 2026: its drive and its pictures answer no more; the list is the card of Find the
    road types on Speed review."""
    from review_helpers import TOKEN, RunningServer, json_of, make_report, real_previews, sign_in

    from nnslr_tools.road_types import corrections_path
    make_report(tmp_path, ROUTE, count=6)
    real_previews(tmp_path, ROUTE)
    _drive_of(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    address = f'/api/panel/road-types/{ROUTE}'
    try:
        cookie = sign_in(server)
        origin = {**cookie, 'Origin': f'http://127.0.0.1:{server.port}'}
        assert server.request('GET', address, headers=cookie)[0] == 404
        for path in (address, address + '/undo', address + '/junction', address + '/junction/undo'):
            assert server.request('POST', path, body={'from': '0/1', 'to': '0/2', 'road_type': 'rural'}, headers=origin)[0] == 404
        assert server.request('GET', f'/route/{ROUTE}/small/images/00000000.jpg', headers=cookie)[0] == 404
        assert not corrections_path(tmp_path, ROUTE).exists()
        status, response, data = server.request('GET', '/api/panel/road-types', headers=cookie)
        assert status == 200 and json_of(data)['routes'][0]['route'] == ROUTE
        # The old address goes to Speed review.
        status, response, data = server.request('GET', '/roads', headers=cookie)
        assert status == 303 and response.getheader('Location') == '/speed-review'
    finally:
        server.close()


# -- car parks: parking in the open, garage underground or indoors ---------------------------------------------------------------------------------------
def _parking_map(tmp_path, ways, parks):
    """The roads and car parks of the map in the data root, as scripts/osm_roads.py writes them."""
    folder = tmp_path / 'derived' / 'osm'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'roads.jsonl').write_text(''.join(json.dumps(way) + '\n' for way in ways))
    (folder / 'parking.jsonl').write_text(''.join(json.dumps(park) + '\n' for park in parks))
    return RoadMap.load(tmp_path)


def _at(lat, lon, heading, speed):
    return {'frame_key': '0/0', 'segment_index': 0, 'output_index': 0, 'latitude': lat, 'longitude': lon,
            'bearing_deg': heading, 'speed_mps': speed}


def _type(road_map, parking, lat, lon, heading=90.0, speed=4.0):
    from nnslr_tools.road_types import frame_types
    line = frame_types(road_map, [_at(lat, lon, heading, speed)], parking)[0]
    return line['road_type'], line['basis'], line['sure']


# A car park from lat 47.010 to 47.012, lon 8.010 to 8.013, with a hole (a building) in its south-west corner; a
# primary road at 50 crosses it from west to east at lat 47.011, a service road runs north in its east part.
CAR_PARK = {'id': 7, 'type': 'way', 'name': 'P', 'parking': 'surface',
            'rings': [[[47.010, 8.010], [47.010, 8.013], [47.012, 8.013], [47.012, 8.010], [47.010, 8.010]],
                      [[47.0102, 8.0102], [47.0106, 8.0102], [47.0106, 8.0108], [47.0102, 8.0108], [47.0102, 8.0102]]]}
CROSSING = [{'id': 1, 'highway': 'primary', 'maxspeed': '50', 'nodes': [[47.011, 8.009], [47.011, 8.014]]},
            {'id': 2, 'highway': 'service', 'nodes': [[47.0102, 8.0125], [47.0118, 8.0125]]}]


def test_parking_from_an_aisle(tmp_path):
    road_map = _parking_map(tmp_path, [{'id': 3, 'highway': 'service', 'service': 'parking_aisle',
                                        'nodes': [[47.02, 8.0], [47.02, 8.002]]}], [])
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, 'latitude': 47.02,
                  'longitude': 8.0002 + index * 0.0003, 'bearing_deg': 90.0, 'speed_mps': 4.0} for index in range(5)]
    # No outline around it: the tag of the aisle is enough, and sure.
    assert build_road_types(tmp_path, ROUTE, positions, road_map)['types'] == {'parking': 5}
    lines = load_road_types(tmp_path, ROUTE)
    assert {(line['road_type'], line['basis'], line['sure'], line['way_id']) for line in lines} == {
        ('parking', 'parking aisle', True, 3)}
    # At 50 km/h on an aisle: the GPS slipping onto a car park beside a ramp (Basel St. Jakob), not a car park.
    from nnslr_tools.road_types import ParkingMap
    assert _type(road_map, ParkingMap([]), 47.02, 8.001, speed=13.0) == (None, 'highway service without a limit', False)
    # A car park beside the road for less than MIN_RUN frames is a slip, as any other.
    assert [value for value, _ in smooth(['town'] * 5 + ['parking'] * 2 + ['town'] * 5)] == ['town'] * 12


def test_parking_from_an_outline_with_no_road(tmp_path):
    from nnslr_tools.road_types import ParkingMap
    road_map = _parking_map(tmp_path, CROSSING, [CAR_PARK])
    parking = ParkingMap.load(tmp_path)
    # Inside the outline, no road within reach: a car park, but one signal only.
    assert _type(road_map, parking, 47.0118, 8.0105) == ('parking', 'car park on the map', False)
    # On the service road inside it: the outline and the road agree.
    assert _type(road_map, parking, 47.0104, 8.0125, heading=0.0) == ('parking', 'car park on the map', True)
    # In the hole of the outline, or outside it: no road, no type.
    assert _type(road_map, parking, 47.0104, 8.0105) == (None, 'no road', False)
    assert _type(road_map, parking, 47.0135, 8.0105) == (None, 'no road', False)
    # Without parking.jsonl (a map read before the car parks were kept) the outline is not known.
    (tmp_path / 'derived' / 'osm' / 'parking.jsonl').unlink()
    assert _type(road_map, ParkingMap.load(tmp_path), 47.0118, 8.0105) == (None, 'no road', False)


def test_not_parking_on_a_road_through_a_car_park_at_speed(tmp_path):
    from nnslr_tools.road_types import ParkingMap
    road_map = _parking_map(tmp_path, CROSSING, [CAR_PARK])
    parking = ParkingMap.load(tmp_path)
    # On the primary road across the outline at 50 km/h: the road, not the car park.
    assert _type(road_map, parking, 47.011, 8.0110, speed=14.0) == ('town', 'maxspeed 50', True)
    # Without a speed the road stays too.
    assert _type(road_map, parking, 47.011, 8.0110, speed=None) == ('town', 'maxspeed 50', True)
    # The service road inside the outline at 45 km/h: the GPS slipping, not the car park.
    assert _type(road_map, parking, 47.0104, 8.0125, heading=0.0, speed=12.5) == (
        None, 'highway service without a limit', False)
    # Slower than a walk there: the GPS finds the road beside the car park, the car is in it (not sure).
    assert _type(road_map, parking, 47.011, 8.0110, speed=1.0) == ('parking', 'car park on the map', False)
    # An underground car park lies under the street: slow over it is not parking.
    underground = ParkingMap([{**CAR_PARK, 'parking': 'underground'}])
    assert _type(road_map, underground, 47.011, 8.0110, speed=1.0) == ('town', 'maxspeed 50', True)


def test_a_garage_underground_or_indoors(tmp_path):
    from nnslr_tools.road_types import ParkingMap
    road_map = _parking_map(tmp_path, CROSSING, [{**CAR_PARK, 'parking': 'underground'}])
    garage = ParkingMap.load(tmp_path)
    # Its service road and outline agree: a garage, sure; no road found inside it: a garage of one signal.
    assert _type(road_map, garage, 47.0104, 8.0125, heading=0.0) == ('garage', 'garage on the map', True)
    assert _type(road_map, garage, 47.0118, 8.0105) == ('garage', 'garage on the map', False)
    # A multi-storey one is a building: slow beside the street is inside it.
    storeys = ParkingMap([{**CAR_PARK, 'parking': 'multi-storey'}])
    assert _type(road_map, storeys, 47.011, 8.0110, speed=1.0) == ('garage', 'garage on the map', False)
    # Its roof, and a car park the map does not describe: in the open.
    for kind in ('rooftop', None):
        assert _type(road_map, ParkingMap([{**CAR_PARK, 'parking': kind}]), 47.0118, 8.0105)[0] == 'parking'


def _drive_without_gps(road_map, gps, speeds):
    """A drive along the primary road of CROSSING: gps[i] False is a frame without a fix; the car's own speeds."""
    from nnslr_tools.road_types import ParkingMap, frame_types
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index,
                  **({'latitude': 47.011, 'longitude': 8.0095 + index * 0.0001, 'bearing_deg': 90.0, 'speed_mps': 4.0}
                     if fix else {'latitude': None, 'longitude': None, 'bearing_deg': None, 'speed_mps': None})}
                 for index, fix in enumerate(gps)]
    lines = frame_types(road_map, positions, ParkingMap([]), speeds)
    return [(line['road_type'], line['sure']) for line in lines]


def test_a_garage_without_gps_at_the_start_or_the_end_of_a_drive_when_slow(tmp_path):
    road_map = _parking_map(tmp_path, CROSSING, [])
    town = ('town', True)
    slow = [2.0, 3.0, None, 4.0, 1.0]
    # The drive starts in a garage: no fix, the car slow, then the street.
    assert _drive_without_gps(road_map, [False] * 5 + [True] * 6, slow + [8.0] * 6) == [('garage', False)] * 5 + [town] * 6
    # And ends in one.
    assert _drive_without_gps(road_map, [True] * 6 + [False] * 5, [8.0] * 6 + slow) == [town] * 6 + [('garage', False)] * 5
    # Too fast once (a cold GPS on the road), the speed not known, or a tunnel in the middle of the drive: not a garage.
    assert _drive_without_gps(road_map, [False] * 5 + [True] * 6, [2.0, 3.0, 11.0, 4.0, 1.0] + [8.0] * 6)[0] == (None, False)
    assert _drive_without_gps(road_map, [False] * 5 + [True] * 6, [None] * 11)[0] == (None, False)
    middle = _drive_without_gps(road_map, [True] * 6 + [False] * 5 + [True] * 6, [8.0] * 6 + slow + [8.0] * 6)
    assert middle[6] == ('town', False)  # the gap filled from the town on both sides


def test_a_garage_without_gps_beside_a_frame_of_a_garage(tmp_path):
    from nnslr_tools.road_types import ParkingMap, frame_types
    road_map = _parking_map(tmp_path, CROSSING, [{**CAR_PARK, 'parking': 'underground'}])
    # The last fix inside the outline (no road there), then the GPS lost down the ramp, then out on the street.
    inside = {'latitude': 47.0118, 'longitude': 8.0105, 'bearing_deg': 90.0, 'speed_mps': 2.0}
    street = {'latitude': 47.011, 'longitude': 8.0095, 'bearing_deg': 90.0, 'speed_mps': 8.0}
    lost = {'latitude': None, 'longitude': None, 'bearing_deg': None, 'speed_mps': None}
    kinds = [street] * 6 + [inside] * 2 + [lost] * 6 + [street] * 6
    positions = [{'frame_key': f'0/{index}', 'segment_index': 0, 'output_index': index, **kind}
                 for index, kind in enumerate(kinds)]
    lines = frame_types(road_map, positions, ParkingMap.load(tmp_path), [8.0] * 6 + [2.0] * 8 + [8.0] * 6)
    assert [line['road_type'] for line in lines[6:14]] == ['garage'] * 8
    assert [line['basis'] for line in lines[8:14]] == ['slow without gps'] * 6


def test_a_correction_to_parking(tmp_path):
    from nnslr_tools.road_types import correct, summary
    _drive_of(tmp_path)
    correct(tmp_path, ROUTE, '0/0', '0/1', 'parking', 'admin')
    lines = load_road_types(tmp_path, ROUTE)
    assert [line['road_type'] for line in lines] == ['parking'] * 2 + ['motorway'] * 2 + [None] * 2
    assert (lines[0]['basis'], lines[0]['sure'], lines[0]['reviewed']) == ('reviewed', True, True)
    assert summary(tmp_path)['routes'][0]['types'] == {'town': 0, 'rural': 0, 'expressway': 0, 'motorway': 2,
                                                       'ramp': 0, 'parking': 2, 'zone30': 0, 'garage': 0,
                                                       'merge': 0, 'unknown': 2}


def test_a_broken_corrections_line_is_skipped(tmp_path):
    from nnslr_tools.road_types import corrections_path, correct, load_corrections, summary
    _drive_of(tmp_path)
    path = corrections_path(tmp_path, ROUTE)
    path.parent.mkdir(parents=True)
    good = {'from': '0/0', 'to': '0/0', 'road_type': 'town', 'reviewer': 'admin', 'at': '2026-10-27T12:00:00+01:00'}
    path.write_text('\n'.join([
        json.dumps(good),
        '{"from": "0/1", "to":',                                        # cut short
        '[1, 2]',                                                       # not an object
        json.dumps({**good, 'from': '0/1', 'to': '0/1', 'road_type': 'highway'}),   # not a type
        json.dumps({**good, 'from': 1, 'to': '0/1'}),                   # not a frame key
        json.dumps({'from': '0/1', 'to': '0/1', 'reviewer': 'admin'}),  # no road_type
        '{"from": "0/1", "to": "0/1", "road_type": NaN}',
        '\xff\xfe not text',
        json.dumps({**good, 'from': '0/2', 'to': '0/2', 'road_type': None}),
    ]) + '\n', encoding='latin-1')
    assert [(line['from'], line['road_type']) for line in load_corrections(tmp_path, ROUTE)] == [('0/0', 'town'),
                                                                                                ('0/2', None)]
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)] == ['town', 'motorway', None, 'motorway',
                                                                                None, None]
    # The list still answers.
    assert summary(tmp_path)['routes'][0]['reviewed'] == 2
    # A new correction is appended after them and counts.
    correct(tmp_path, ROUTE, '0/1', '0/1', 'rural', 'admin')
    assert [line['road_type'] for line in load_road_types(tmp_path, ROUTE)][:3] == ['town', 'rural', None]


def test_the_overview_counts_the_prepared_drives_without_road_types(tmp_path):
    from nnslr_tools.road_types import drives_without
    from review_helpers import make_report
    make_report(tmp_path, route='00000001--aaaaaaaaaa')
    make_report(tmp_path, route='00000002--bbbbbbbbbb')
    assert drives_without(tmp_path) == 2
    (tmp_path / 'derived' / 'road_types').mkdir(parents=True)
    (tmp_path / 'derived' / 'road_types' / '00000001--aaaaaaaaaa.jsonl').write_text('{}\n')
    assert drives_without(tmp_path) == 1
