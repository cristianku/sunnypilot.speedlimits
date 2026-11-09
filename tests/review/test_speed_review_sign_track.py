"""Speed review reads the signs on every frame of a drive (GET /api/speed-review/drives/<route>/sign-track,
nnslr_tools.sign_tracks.read): the file as nnslr sign-tracks wrote it and whether the networks in use made it on these
minutes; a 404 the page takes as "none yet"; never written through the site, a guest's only with signs work (it holds no
place), never sent to the review machine. The drive and its boxes are stand-ins: nothing is decoded, no network runs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools import review_box, sign_tracks  # noqa: E402
from nnslr_tools.review_access import guest_rule  # noqa: E402
from review_helpers import RunningServer, json_of  # noqa: E402
from test_sign_tracks import NETWORKS, ROUTE, Finder, decoder, drive  # noqa: E402

ADDRESS = f'/api/speed-review/drives/{ROUTE}/sign-track'


def test_the_drive_s_boxes_on_every_frame_and_whether_they_are_current(tmp_path, monkeypatch):
    drive(tmp_path)
    monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: NETWORKS)
    server = RunningServer(tmp_path)
    try:
        status, _response, data = server.request('GET', ADDRESS)
        assert (status, json_of(data)) == (404, {'ok': False, 'error': 'no_sign_track',
                                                 'message': 'This drive has no signs on every frame yet.'})
        sign_tracks.run_drive(tmp_path, ROUTE, Finder(), decoder=decoder(), log=lambda text: None)
        status, _response, data = server.request('GET', ADDRESS)
        assert status == 200
        answer = json_of(data)
        assert answer['ok'] is True and answer['current'] is True
        assert (answer['format'], answer['route'], answer['fps'], answer['width'], answer['height']) == (
            1, ROUTE, 20, 1344, 760)
        assert [label['model_label'] for label in answer['labels']] == [
            'maximum_speed:60', 'other_sign:pedestrian_crossing']
        assert answer['segments'][0]['boxes'][0] == [0, 0, 10, 21, 30, 40, 97]
        # Other networks in use since: still shown, not current.
        monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: {})
        assert json_of(server.request('GET', ADDRESS)[2])['current'] is False
        # Only nnslr sign-tracks writes it.
        assert server.request('POST', ADDRESS, body={})[0] == 405
        # Not a drive's address: nothing.
        assert server.request('GET', '/api/speed-review/drives/000000ca--eb394eea2/sign-track')[0] == 404
    finally:
        server.close()


def test_a_guest_reads_it_only_with_signs_work_and_the_review_machine_never_gets_it():
    # Open for the drives a guest reviews the signs of (the Signs side reads it), not for those of speed work only.
    assert guest_rule('GET', ADDRESS) == (ROUTE, ('signs',))
    assert 'speed' not in guest_rule('GET', ADDRESS)[1]
    assert guest_rule('POST', ADDRESS) is None
    assert not any('sign_tracks' in item for item in review_box.AS_THEY_ARE)


def test_the_models_card_reads_the_drives(tmp_path, monkeypatch):
    drive(tmp_path)
    monkeypatch.setattr(sign_tracks, 'networks_in_use', lambda root: NETWORKS)
    server = RunningServer(tmp_path)
    try:
        status, _response, data = server.request('GET', '/api/panel/sign-tracks')
        assert status == 200
        answer = json_of(data)
        assert answer['networks_in_use'] == NETWORKS
        assert [(item['route'], item['state'], item['minutes']) for item in answer['drives']] == [(ROUTE, 'none', 2)]
    finally:
        server.close()
