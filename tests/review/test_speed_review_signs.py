"""Speed review shows the signs of Sign review (GET /api/speed-review/drives/<route>/signs, drive_review.sign_frames):
the boxes confirmed on a frame, else the model's proposals, every frame placed at its second of the drive.

Every drive and label here is SYNTHETIC: made up by these tests. Nothing here is a real drive or a person's label.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnslr_tools import drive_review  # noqa: E402
from review_helpers import ROUTE, RunningServer, json_of, make_report, page_body  # noqa: E402

CONTROL_JS = Path(__file__).resolve().parents[2] / 'src/nnslr_tools/web/js/control.js'


def test_the_signs_say_whether_the_networks_in_use_proposed_them(tmp_path):
    import json
    make_report(tmp_path)
    in_use = {'cutter': 'cutter:c2/signnet', 'classifier': 'classifier:k2/classifier', 'decider': 'decider:d2/full'}
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models' / 'active.json').write_text(json.dumps({part: {'id': key} for part, key in in_use.items()}))
    run = next((tmp_path / 'derived/preannotations' / ROUTE).glob('*/run.json'))
    older = dict(in_use, cutter='cutter:c1/signnet')
    server = RunningServer(tmp_path)
    try:
        for model, expected in ((older, False), (in_use, True)):
            run.write_text(json.dumps({'run_id': run.parent.name, 'proposer': 'ours', 'model': {'proposer': 'ours', **model}}))
            answer = json_of(server.request('GET', f'/api/speed-review/drives/{ROUTE}/signs')[2])
            assert answer['proposed_by'] == {'proposer': 'ours', 'networks': model, 'in_use': expected}
    finally:
        server.close()


def test_a_frame_confirmed_shows_its_own_boxes_and_never_the_proposals_again(tmp_path):
    rows = {row['frame_key']: row for row in make_report(tmp_path, count=4)}
    server = RunningServer(tmp_path)
    try:
        # 0/0: confirmed at 50 km/h; 0/2: confirmed with no sign at all (the proposal deleted); 0/1, 0/3 not reviewed.
        assert server.post_frame(ROUTE, '0/0', page_body(rows['0/0'], value_kph=50))[0] == 200
        assert server.post_frame(ROUTE, '0/2', page_body(rows['0/2'], drop_proposals=True))[0] == 200
        status, _response, data = server.request('GET', f'/api/speed-review/drives/{ROUTE}/signs')
        assert status == 200
        answer = json_of(data)
        frames = {item['frame_key']: item for item in answer['frames']}
        assert list(frames) == ['0/0', '0/1', '0/2', '0/3']
        assert frames['0/0']['confirmed'] is True and frames['0/0']['reviewer'] == 'Cristian'
        assert [box['value_kph'] for box in frames['0/0']['boxes']] == [50]
        assert frames['0/0']['boxes'][0]['bbox_xyxy'] == [1, 2, 15, 20] and 'detection_id' not in frames['0/0']['boxes'][0]
        # A confirmed frame without a sign stays without one: the proposal the reviewer deleted is not drawn.
        assert frames['0/2']['confirmed'] is True and frames['0/2']['boxes'] == []
        # Not reviewed: the model's proposals (frame 1 has none in the report).
        assert frames['0/1']['confirmed'] is False and frames['0/1']['boxes'] == []
        assert frames['0/3']['confirmed'] is False and frames['0/3']['boxes'] == []
        # One picture a second of the segment: its second is its place.
        assert [item['second'] for item in answer['frames']] == [0, 1, 2, 3]
        assert frames['0/0']['width'] and frames['0/0']['height']
        # Who proposed the boxes (issue #42): a report of before the choice is the Vietnamese YOLO's, not the networks
        # in use.
        assert answer['proposed_by'] == {'proposer': 'vietnam', 'networks': {}, 'in_use': False}
        # A drive without frames in Sign review: a 404 the page takes as "no signs", nothing more.
        status, _response, data = server.request('GET', '/api/speed-review/drives/00000002--abcdef1234/signs')
        assert status == 404
        # Only reading: the boxes are changed in Sign review.
        assert server.request('POST', f'/api/speed-review/drives/{ROUTE}/signs', body={})[0] in (403, 404, 405)
    finally:
        server.close()


def test_a_frame_is_placed_by_its_media_time_its_video_frame_or_its_place_among_the_pictures():
    base = {'segment_index': 3, 'width': 1344, 'height': 760, 'requested_sample_fps': 1.0, 'detections': []}
    rows = [
        {**base, 'frame_key': '3/5', 'output_index': 5, 'media_time_s': 5.0},
        # A frame at a sign the comma read: 1000 + its frame in the minute, never 1005 seconds.
        {**base, 'frame_key': '3/1107', 'output_index': 1107, 'media_time_s': None, 'decoded_frame_index': 107},
        {**base, 'frame_key': '3/1200', 'output_index': 1200, 'media_time_s': None, 'decoded_frame_index': None},
        {**base, 'frame_key': '3/7', 'output_index': 7, 'media_time_s': None},
    ]
    proposal = {'bbox_xyxy': [10, 20, 30, 40], 'family': 'maximum_speed', 'value_kph': 80, 'score': 0.9}
    rows[0]['detections'] = [proposal, {'family': 'maximum_speed', 'value_kph': 60}]
    answer = drive_review.sign_frames('run', rows, [{'frame_key': '3/7', 'confirmed': False, 'payload': {}}])
    placed = {item['frame_key']: item['second'] for item in answer['frames']}
    assert placed == {'3/5': 5.0, '3/1107': 5.35, '3/7': 7.0}
    first = answer['frames'][0]
    # A box without its place on the picture is left out; a review not confirmed shows the proposals.
    assert first['boxes'] == [proposal] and answer['frames'][-1]['confirmed'] is False


def test_a_frame_of_the_site_ai_reviewer_is_confirmed_with_its_own_boxes():
    row = {'frame_key': '0/4', 'segment_index': 0, 'output_index': 4, 'media_time_s': 4.0, 'width': 1344,
           'height': 760, 'detections': [{'bbox_xyxy': [1, 2, 3, 4], 'family': 'maximum_speed', 'value_kph': 120}]}
    saved = {'bbox_xyxy': [5, 6, 7, 8], 'family': 'maximum_speed', 'value_kph': 80}
    review = {'frame_key': '0/4', 'confirmed': True, 'reviewer': 'ai/site/claude', 'reviewer_kind': 'ai',
              'payload': {'detections': [saved]}}
    frame = drive_review.sign_frames('run', [row], [review])['frames'][0]
    # Final since 13 Nov 2026 (#79): confirmed, with the AI's boxes, not the model's.
    assert frame['confirmed'] is True and frame['needs_person'] is False
    assert [box['value_kph'] for box in frame['boxes']] == [80]
    # An agent's review through the API (ai/<name>) is confirmed.
    frame = drive_review.sign_frames('run', [row], [dict(review, reviewer='ai/claude')])['frames'][0]
    assert frame['confirmed'] is True and frame['needs_person'] is False


def test_the_drive_player_draws_the_signs_and_falls_back_without_them():
    script = CONTROL_JS.read_text(encoding='utf-8')
    assert '/signs' in script and 'sr-sign-box' in script
