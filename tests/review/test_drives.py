"""A route as the drive it was (nnslr_tools.drives): when from the last segment video, where from the GPS ends."""
import json
import os

from nnslr_tools.drives import drive_of

ROUTE = '0000002a--0123456789'


def test_drive_named_by_time_and_places(tmp_path):
    # Segments 0-2; the first video has the comma's clock before GPS time (a date months back): only the last counts.
    for segment, stamp in ((0, 1_700_000_000), (1, 1_792_734_000), (2, 1_792_734_060)):
        video = tmp_path / 'raw' / 'routes' / ROUTE / str(segment) / 'fcamera.hevc'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'')
        os.utime(video, (stamp, stamp))
    places = tmp_path / 'derived' / 'osm' / 'places.jsonl'
    places.parent.mkdir(parents=True)
    places.write_text('\n'.join(json.dumps(item) for item in (
        {'name': 'Aarau', 'place': 'town', 'lat': 47.392, 'lon': 8.044},
        {'name': 'Altstadt', 'place': 'suburb', 'lat': 47.39, 'lon': 8.045},
        {'name': 'Herznach', 'place': 'village', 'lat': 47.472, 'lon': 8.048})) + '\n')
    gps = tmp_path / 'derived' / 'gps' / ROUTE / 'positions.jsonl'
    gps.parent.mkdir(parents=True)
    gps.write_text('\n'.join(json.dumps(item) for item in (
        {'latitude': None, 'longitude': None}, {'latitude': 47.47, 'longitude': 8.05},
        {'latitude': 47.391, 'longitude': 8.045})) + '\n')
    drive = drive_of(tmp_path, ROUTE)
    # The last video ends at 1_792_734_060 (23 Oct 2026, 05:41 UTC); 3 segments: started 3 minutes before.
    assert drive['start'] == '2026-10-23T05:38+00:00' and drive['minutes'] == 3
    assert drive['when'] == 'Fri 23 Oct 2026 · 07:38'
    # A town beats a district next to it.
    assert (drive['from'], drive['to'], drive['title']) == ('Herznach', 'Aarau', 'Herznach → Aarau')


def test_a_hamlet_names_a_position_only_where_no_other_place_is_near(tmp_path):
    from nnslr_tools.drives import place_name
    places = tmp_path / 'derived' / 'osm' / 'places.jsonl'
    places.parent.mkdir(parents=True)
    places.write_text('\n'.join(json.dumps(item) for item in (
        {'name': 'Arcola', 'place': 'town', 'lat': 39.685, 'lon': -88.306},
        {'name': 'Bourbon', 'place': 'hamlet', 'lat': 39.69, 'lon': -88.30},
        {'name': 'Hayes', 'place': 'hamlet', 'lat': 39.80, 'lon': -88.55},
        # Across a square's edge (BOX_LAT, BOX_LON): still found.
        {'name': 'Lone Grove', 'place': 'hamlet', 'lat': 40.001, 'lon': -88.0})) + '\n')
    # On the hamlet, the town under a km away wins.
    assert place_name(tmp_path, 39.69, -88.30) == 'Arcola'
    # Nothing else within reach: the hamlet.
    assert place_name(tmp_path, 39.801, -88.55) == 'Hayes'
    assert place_name(tmp_path, 39.999, -88.0) == 'Lone Grove'
    # Far from all of them: no name.
    assert place_name(tmp_path, 41.0, -90.0) is None
