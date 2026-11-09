"""The signs on every frame of a drive (nnslr_tools.sign_tracks): the file and its atomic write, a drive left as it is when
its networks and minutes are the same and read again when either changed, a stopped run going on from its partial file,
a minute decoded to another count than the video's failing its drive, and what the site reads (read, summary). The frames
and the networks are stand-ins: nothing is decoded, no network runs."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nnslr_tools import drive_video, sign_tracks
from nnslr_tools.sign_tracks import SignTrackError

ROUTE = '000000ca--eb394eea21'
NETWORKS = {'cutter': {'id': 'cutter:c/signnet', 'size': 1, 'mtime_ns': 1},
            'classifier': {'id': 'classifier:k/classifier', 'size': 2, 'mtime_ns': 2}}


def drive(root: Path, route: str = ROUTE, minutes: int = 2, frames: int = 5) -> dict:
    """A drive with its raw minutes (those already there kept) and its video made from them (drive_video.index)."""
    for segment in range(minutes):
        video = root / 'raw' / 'routes' / route / str(segment) / 'fcamera.hevc'
        video.parent.mkdir(parents=True, exist_ok=True)
        if not video.exists():
            video.write_bytes(b'hevc' * (segment + 1))
    mp4, index_path = drive_video.paths(root, route)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    mp4.write_bytes(b'mp4')
    index = {'route': route, 'fps': 20, 'width': 960,
             'segments': [{'segment': segment, 'start': segment * frames / 20, 'frames': frames}
                          for segment in range(minutes)],
             'source': drive_video._signature(drive_video.minutes(root, route))}
    index_path.write_text(json.dumps(index))
    return index


class Finder:
    """Two boxes on every frame k divisible by 2 (one of them not a sign is the networks' business: Finder.find drops
    it), the frames it was given noted."""

    def __init__(self, networks=NETWORKS):
        self.networks = networks
        self.seen = []

    def find(self, batch):
        found = []
        for k, frame, whole in batch:
            self.seen.append(frame)
            found.append((k, [{'model_label': 'maximum_speed:60', 'class_id': 1, 'bbox_xyxy': [10.4, 20.6, 30.0, 40.2],
                               'score': 0.9712},
                              {'model_label': 'other_sign:pedestrian_crossing', 'class_id': 2,
                               'bbox_xyxy': [-3.0, 700.0, 50.0, 790.0], 'score': 0.5}] if k % 2 == 0 else []))
        return found


def decoder(counts: dict | None = None, calls: list | None = None):
    """Frames named after their minute and k; counts: a minute decoded to another count."""
    def decode(video, limit=None):
        segment = int(Path(video).parent.name)
        if calls is not None:
            calls.append(segment)
        total = (counts or {}).get(segment, 5)
        for k in range(min(total, limit) if limit else total):
            yield f'{segment}/{k}', None
    return decode


def test_the_file_holds_every_frame_s_boxes_and_is_written_whole(tmp_path):
    drive(tmp_path)
    result = sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(), log=lambda text: None)
    assert (result['state'], result['frames'], result['boxes']) == ('done', 10, 12)
    track = json.loads(sign_tracks.track_path(tmp_path, ROUTE).read_text())
    assert {key: track[key] for key in ('format', 'route', 'fps', 'width', 'height', 'networks', 'min_score')} == {
        'format': 1, 'route': ROUTE, 'fps': 20, 'width': 1344, 'height': 760, 'networks': NETWORKS, 'min_score': 0.1}
    assert track['labels'] == [
        {'model_label': 'maximum_speed:60', 'family': 'maximum_speed', 'value_kph': 60, 'supplementary_panel': 'unknown'},
        {'model_label': 'other_sign:pedestrian_crossing', 'family': 'other_sign', 'value_kph': None,
         'supplementary_panel': 'unknown', 'sign_type': 'pedestrian_crossing'}]
    first = track['segments'][0]
    sizes = sign_tracks.sources(tmp_path, ROUTE)
    assert (first['segment'], first['frames'], first['source']) == (0, 5, sizes[0])
    # Whole pixels clamped to the frame, the score in percent, by k then the best score first.
    assert first['boxes'][:2] == [[0, 0, 10, 21, 30, 40, 97], [0, 1, 0, 700, 50, 760, 50]]
    assert [box[0] for box in first['boxes']] == [0, 0, 2, 2, 4, 4]
    meta = json.loads(sign_tracks.meta_path(sign_tracks.track_path(tmp_path, ROUTE)).read_text())
    assert meta['segments'] == [[0, sizes[0]], [1, sizes[1]]] and (meta['frames'], meta['boxes']) == (10, 12)
    # Nothing left beside it: no partial file, no temporary one.
    assert sorted(path.name for path in sign_tracks.folder(tmp_path).iterdir()) == [
        f'{ROUTE}.json', f'{ROUTE}.meta.json']


def test_an_hours_class_keeps_its_hours_and_not_a_sign_never_has_a_label():
    assert sign_tracks.label_entry('maximum_speed_hours:30:22-6') == {
        'model_label': 'maximum_speed_hours:30:22-6', 'family': 'maximum_speed', 'value_kph': 30,
        'supplementary_panel': 'time', 'plate_hours': '22-6'}
    track = sign_tracks.track_of(ROUTE, NETWORKS, [{'segment': 1, 'frames': 3, 'source': [1, 2], 'boxes': [
        [0, 'b', 1, 1, 2, 2, 50]]}, {'segment': 0, 'frames': 3, 'source': [1, 2], 'boxes': [[2, 'a', 1, 1, 2, 2, 50]]}],
        'now')
    # Numbered as they first appear, minute by minute.
    assert [segment['segment'] for segment in track['segments']] == [0, 1]
    assert [segment['boxes'][0][1] for segment in track['segments']] == [0, 1]
    assert [entry['model_label'] for entry in track['labels']] == ['a', 'b']


def test_a_drive_read_by_the_same_networks_on_the_same_minutes_is_left_as_it_is(tmp_path):
    drive(tmp_path)
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(), log=lambda text: None)
    calls = []
    result = sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(calls=calls), log=lambda text: None)
    assert result['state'] == 'current' and calls == []
    # --force reads it again.
    result = sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(calls=calls), force=True,
                                   log=lambda text: None)
    assert result['state'] == 'done' and calls == [0, 1]


def test_a_network_trained_again_or_a_minute_changed_reads_it_again(tmp_path):
    drive(tmp_path)
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(), log=lambda text: None)
    calls = []
    other = {**NETWORKS, 'classifier': {**NETWORKS['classifier'], 'mtime_ns': 3}}
    assert sign_tracks.run_drive(tmp_path, ROUTE, Finder(other), decoder=decoder(calls=calls),
                                 log=lambda text: None)['state'] == 'done'
    assert calls == [0, 1]
    # A minute that changed makes its video again first: without it the drive is refused, then read once it is made.
    (tmp_path / 'raw' / 'routes' / ROUTE / '1' / 'fcamera.hevc').write_bytes(b'longer minute')
    with pytest.raises(SignTrackError, match='make its video first'):
        sign_tracks.run_drive(tmp_path, ROUTE, Finder(other), decoder=decoder(), log=lambda text: None)
    drive(tmp_path)
    calls.clear()
    assert sign_tracks.run_drive(tmp_path, ROUTE, Finder(other), decoder=decoder(calls=calls),
                                 log=lambda text: None)['state'] == 'done'
    assert calls == [0, 1]


def test_a_stopped_run_goes_on_from_its_partial_file(tmp_path):
    drive(tmp_path, minutes=3)
    target = sign_tracks.track_path(tmp_path, ROUTE)
    partial = sign_tracks.partial_path(target)
    partial.parent.mkdir(parents=True)
    sizes = sign_tracks.sources(tmp_path, ROUTE)
    lines = [{'networks': NETWORKS},
             # Minute 1 done, its labels by name: the numbers are given at the end, in the order of the minutes.
             {'segment': 1, 'frames': 5, 'source': sizes[1], 'boxes': [[3, 'maximum_speed:80', 1, 2, 3, 4, 88]]},
             # Minute 2 done on another recording of it: read again.
             {'segment': 2, 'frames': 5, 'source': [1, 1], 'boxes': []}]
    partial.write_text(''.join(json.dumps(line) + '\n' for line in lines) + '{"segment": 0, "fra')
    calls = []
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(calls=calls), log=lambda text: None)
    assert calls == [0, 2]
    track = json.loads(target.read_text())
    labels = [entry['model_label'] for entry in track['labels']]
    assert labels == ['maximum_speed:60', 'other_sign:pedestrian_crossing', 'maximum_speed:80']
    assert track['segments'][1]['boxes'] == [[3, 2, 1, 2, 3, 4, 88]]
    assert not partial.exists()
    # A partial file of other networks is started again.
    partial.write_text(json.dumps({'networks': {}}) + '\n' + json.dumps(lines[1]) + '\n')
    calls.clear()
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(calls=calls), force=False, log=lambda text: None)
    assert calls == []  # current: the file is whole, the stale partial file is not even read
    target.unlink()
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(calls=calls), log=lambda text: None)
    assert calls == [0, 1, 2]


def test_a_minute_decoded_to_another_count_fails_its_drive_and_keeps_the_minutes_done(tmp_path):
    drive(tmp_path, minutes=2)
    logged = []
    with pytest.raises(SignTrackError, match='minute 1: decoded 4 frames, the video has 5'):
        sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder({1: 4}), log=logged.append)
    assert logged[0].startswith('minute 1/2: segment 0, 5 frames, 6 boxes, ')
    target = sign_tracks.track_path(tmp_path, ROUTE)
    assert not target.exists()
    assert [json.loads(line).get('segment') for line in sign_tracks.partial_path(target).read_text().splitlines()] == [
        None, 0]


def test_run_says_each_drive_and_goes_on_after_one_fails(tmp_path, monkeypatch):
    drive(tmp_path)
    monkeypatch.setattr(sign_tracks, 'load_finder', lambda root, device: Finder())
    monkeypatch.setattr(sign_tracks, 'decode_minute', decoder())
    monkeypatch.setattr(sign_tracks.run_drive, '__kwdefaults__',
                        {**sign_tracks.run_drive.__kwdefaults__, 'decoder': decoder()})
    logged = []
    failed = sign_tracks.run(tmp_path, ['00000001--0000000000', ROUTE, ROUTE], device='cpu', log=logged.append)
    assert failed == 1
    drives = [line for line in logged if line.startswith('drive ')]
    assert drives[0].startswith('drive 1/3 00000001--0000000000: failed: no video of its current minutes')
    assert drives[1].startswith(f'drive 2/3 {ROUTE}: 10 frames, 12 boxes in ')
    assert drives[2] == f'drive 3/3 {ROUTE}: done already'


def test_a_dry_run_reads_the_first_frames_of_the_first_minute_apart_and_checks_them(tmp_path, monkeypatch):
    drive(tmp_path)
    monkeypatch.setattr(sign_tracks, 'load_finder', lambda root, device: Finder())
    monkeypatch.setattr(sign_tracks.run_drive, '__kwdefaults__',
                        {**sign_tracks.run_drive.__kwdefaults__, 'decoder': decoder()})
    report = tmp_path / 'derived' / 'preannotations' / ROUTE
    (report / 'run1').mkdir(parents=True)
    (report / 'latest.json').write_text('{"run_id": "run1"}')
    model = {'cutter': NETWORKS['cutter']['id'], 'classifier': NETWORKS['classifier']['id']}
    rows = [{'segment_index': 0, 'decoded_frame_index': 0, 'model': model, 'detections': [
                {'model_label': 'maximum_speed:60', 'bbox_xyxy': [10.4, 20.6, 30.0, 40.2], 'score': 0.9712},
                {'model_label': 'other_sign:pedestrian_crossing', 'bbox_xyxy': [0.0, 700.0, 50.0, 760.0], 'score': 0.5}]},
            # Beyond the frames read, of other networks, of another minute: not compared.
            {'segment_index': 0, 'decoded_frame_index': 20, 'model': model, 'detections': []},
            {'segment_index': 0, 'decoded_frame_index': 0, 'model': {}, 'detections': []},
            {'segment_index': 1, 'decoded_frame_index': 0, 'model': model, 'detections': []}]
    (report / 'run1' / 'preannotations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    logged = []
    assert sign_tracks.run(tmp_path, [ROUTE], dry_run=True, limit_frames=3, log=logged.append) == 0
    assert f'drive 1/1 {ROUTE}: 3 frames, 4 boxes in 0 s' in logged
    assert 'parity: 1/1 frames of Sign review match (2 boxes compared)' in logged
    dry = sign_tracks.folder(tmp_path) / 'dry' / f'{ROUTE}.json'
    assert json.loads(dry.read_text())['segments'][0]['frames'] == 3
    # The real file is never written by a dry run.
    assert not sign_tracks.track_path(tmp_path, ROUTE).exists()


def test_parity_tells_a_different_box_label_or_score():
    track = sign_tracks.track_of(ROUTE, NETWORKS, [{'segment': 0, 'frames': 40, 'source': [1, 1], 'boxes': [
        [20, 'maximum_speed:60', 10, 20, 30, 40, 90]]}], 'now')
    model = {'cutter': NETWORKS['cutter']['id'], 'classifier': NETWORKS['classifier']['id']}

    def row(**box):
        detection = {'model_label': 'maximum_speed:60', 'bbox_xyxy': [10.4, 20.0, 30.0, 40.0], 'score': 0.905, **box}
        return {'segment_index': 0, 'decoded_frame_index': 20, 'model': model, 'detections': [detection]}
    assert sign_tracks.parity(None, track, [row()]) == {'frames': 1, 'matching': 1, 'boxes': 1}
    for changed in ({'bbox_xyxy': [10.6, 20.0, 30.0, 40.0]}, {'model_label': 'maximum_speed:80'}, {'score': 0.88}):
        assert sign_tracks.parity(None, track, [row(**changed)])['matching'] == 0, changed
    # A box of ours Sign review does not have is a difference too.
    assert sign_tracks.parity(None, track, [{**row(), 'detections': []}])['matching'] == 0


def test_read_says_whether_it_is_current_and_summary_lists_every_drive_with_a_video(tmp_path, monkeypatch):
    drive(tmp_path)
    drive(tmp_path, '000000cb--247a3022f6', minutes=1)
    drive(tmp_path, '000000cc--0000000000', minutes=3)
    monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: NETWORKS)
    assert sign_tracks.read(tmp_path, ROUTE) is None
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(), log=lambda text: None)
    with pytest.raises(SignTrackError):
        sign_tracks.run_drive(tmp_path, '000000cc--0000000000', Finder(), decoder=decoder({1: 2}), log=lambda text: None)
    track = sign_tracks.read(tmp_path, ROUTE)
    assert track['current'] is True and track['route'] == ROUTE
    view = sign_tracks.summary(tmp_path)
    assert view['networks_in_use'] == NETWORKS
    drives = {item['route']: item for item in view['drives']}
    assert list(drives) == ['000000ca--eb394eea21', '000000cb--247a3022f6', '000000cc--0000000000']
    assert {key: drives[ROUTE][key] for key in ('minutes', 'done_minutes', 'state', 'frames', 'boxes', 'video')} == {
        'minutes': 2, 'done_minutes': 2, 'state': 'done', 'frames': 10, 'boxes': 12, 'video': True}
    assert drives['000000cb--247a3022f6']['state'] == 'none'
    assert (drives['000000cc--0000000000']['state'], drives['000000cc--0000000000']['done_minutes']) == ('partial', 1)
    # Other networks in use: old, and not current.
    monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: {})
    assert sign_tracks.read(tmp_path, ROUTE)['current'] is False
    assert {item['route']: item['state'] for item in sign_tracks.summary(tmp_path)['drives']}[ROUTE] == 'old'
    # A minute recorded again: old, and its video made again first.
    monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: NETWORKS)
    video = tmp_path / 'raw' / 'routes' / ROUTE / '0' / 'fcamera.hevc'
    video.write_bytes(b'again')
    os.utime(video, (1, 1))
    assert sign_tracks.read(tmp_path, ROUTE)['current'] is False
    item = {item['route']: item for item in sign_tracks.summary(tmp_path)['drives']}[ROUTE]
    assert (item['state'], item['video']) == ('old', False)


def test_a_minute_added_to_the_drive_is_not_current(tmp_path, monkeypatch):
    drive(tmp_path)
    monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: NETWORKS)
    sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(), log=lambda text: None)
    assert sign_tracks.read(tmp_path, ROUTE)['current'] is True
    # A third minute fetched and the video made again: the file has no boxes of it.
    drive(tmp_path, minutes=3)
    assert sign_tracks.read(tmp_path, ROUTE)['current'] is False
    assert {item['route']: item['state'] for item in sign_tracks.summary(tmp_path)['drives']}[ROUTE] == 'old'


def _cli_routes(tmp_path, monkeypatch, *arguments):
    """The drives nnslr sign-tracks hands to run, and what it printed."""
    from nnslr_tools import cli
    given = []
    monkeypatch.setattr(sign_tracks, 'run', lambda root, routes, **options: given.append((routes, options)) or 0)
    code = cli.main(['sign-tracks', '--data-root', str(tmp_path), *arguments])
    return code, given


def test_a_dry_run_takes_only_the_first_drive_named_or_found(tmp_path, monkeypatch):
    drive(tmp_path)
    drive(tmp_path, '000000cb--247a3022f6', minutes=1)
    code, given = _cli_routes(tmp_path, monkeypatch, '--dry-run', '--route', '000000cb--247a3022f6', '--route', ROUTE)
    assert code == 0 and given[0][0] == ['000000cb--247a3022f6'] and given[0][1]['dry_run'] is True
    code, given = _cli_routes(tmp_path, monkeypatch, '--dry-run', '--all')
    assert code == 0 and given[0][0] == [ROUTE]
    code, given = _cli_routes(tmp_path, monkeypatch, '--route', '000000cb--247a3022f6', '--route', ROUTE)
    assert given[0][0] == ['000000cb--247a3022f6', ROUTE]


def test_every_drive_leaves_out_one_whose_video_must_be_made_again(tmp_path, monkeypatch, capsys):
    drive(tmp_path)
    drive(tmp_path, '000000cb--247a3022f6', minutes=1)
    # A minute fetched after the video was made: no video of its minutes now.
    video = tmp_path / 'raw' / 'routes' / '000000cb--247a3022f6' / '1' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'new')
    assert drive_video.index(tmp_path, '000000cb--247a3022f6') is None
    code, given = _cli_routes(tmp_path, monkeypatch, '--all')
    assert code == 0 and given[0][0] == [ROUTE]
    assert 'left out, their video must be made again first: 000000cb--247a3022f6' in capsys.readouterr().out
