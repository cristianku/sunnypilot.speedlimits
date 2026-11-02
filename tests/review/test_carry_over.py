"""Apply previous: the boxes of the frame before carried onto the next one (nnslr_tools.carry_over), on the
page API, the REST API and MCP, and the save that refuses a box nobody checked."""
from __future__ import annotations

import pytest

from nnslr_tools.carry_over import FOCUS, carry_over, unchecked_boxes
from nnslr_tools.review_mcp import ReviewTools
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

W, H = 1344, 760
FX, FY = FOCUS[0] * W, FOCUS[1] * H


def box_at(cx, cy, size):
    return [cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2]


def closer(box, scale, slide=(0.0, 0.0)):
    """The same sign one second later: grown by ``scale`` away from the focus, plus the view sliding."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    size = box[2] - box[0]
    return box_at(FX + scale * (cx - FX) + slide[0], FY + scale * (cy - FY) + slide[1], size * scale)


def confirmed(box, family='variable_display', value=80, **extra):
    item = {'bbox_xyxy': box, 'family': family, 'value_kph': value, 'applicability': 'own_road',
            'supplementary_panel': 'none', 'detection_id': 'old-' + str(int(box[0]))}
    item.update(extra)
    return item


def proposal(box, ident, family='other_sign', value=None):
    return {'bbox_xyxy': box, 'family': family, 'value_kph': value, 'score': 0.6, 'model_label': 'x',
            'applicability': 'unresolved', 'supplementary_panel': 'unknown', 'detection_id': ident}


def test_original_mph_digits_and_unit_are_carried_together():
    box=box_at(560,330,30)
    old=confirmed(box,family='maximum_speed',value=25,sign_value=25,sign_unit='mph')
    now=proposal(closer(box,1.3),'p')
    result=carry_over([old],[now],width=W,height=H,source_key='1/2',proposals=[now])
    carried=result['detections'][0]
    assert (carried['value_kph'],carried['sign_value'],carried['sign_unit'])==(25,25,'mph')


def test_legacy_previous_number_drops_stale_original_metadata():
    box=box_at(560,330,30)
    old=confirmed(box,family='maximum_speed',value=80)
    now=dict(proposal(closer(box,1.3),'p'),sign_value=25,sign_unit='mph')
    result=carry_over([old],[now],width=W,height=H,source_key='1/2',proposals=[now])
    carried=result['detections'][0]
    assert carried['value_kph']==80
    assert 'sign_value' not in carried and 'sign_unit' not in carried


def test_the_same_sign_grown_takes_the_labels_and_other_boxes_stay():
    left = box_at(560, 330, 30)
    right = box_at(840, 330, 30)
    previous = [confirmed(left), confirmed(right), confirmed(box_at(700, 330, 30), family='other_sign', value=None)]
    current = [proposal(closer(left, 1.3), 'p1'), proposal(closer(right, 1.3), 'p2'),
               proposal(closer(box_at(700, 330, 30), 1.3), 'p3'), proposal(box_at(100, 600, 40), 'far')]
    result = carry_over(previous, current, width=W, height=H, source_key='2/52', proposals=current)
    boxes = {item['detection_id']: item for item in result['detections']}
    assert result['matched'] == 3 and result['estimated'] == 0
    assert boxes['p1']['family'] == 'variable_display' and boxes['p1']['value_kph'] == 80
    assert boxes['p1']['applicability'] == 'own_road' and boxes['p1']['carried_from'] == '2/52'
    assert boxes['p1']['bbox_xyxy'] == current[0]['bbox_xyxy'] and 'check_position' not in boxes['p1']
    assert boxes['p3']['family'] == 'other_sign'
    # A box of this frame that is no sign of the previous one stays a proposal.
    assert boxes['far']['family'] == 'other_sign' and 'carried_from' not in boxes['far']


def test_a_turn_slides_the_whole_view():
    signs = [box_at(500, 340, 26), box_at(820, 320, 26)]
    current = [proposal(closer(box, 1.25, slide=(-260, 10)), 'p' + str(i)) for i, box in enumerate(signs)]
    result = carry_over([confirmed(box) for box in signs], current, width=W, height=H, source_key='4/9',
                        proposals=current)
    assert result['matched'] == 2 and result['estimated'] == 0


def test_a_sign_of_another_shape_is_not_the_same_sign():
    # 34/45 -> 34/46 of a drive: a give way and a wide direction sign confirmed; the next frame has only a proposal
    # on the tall pedestrian sign beside the give way. A slide of the view "explained" the direction sign there.
    give_way = confirmed([1026, 264, 1047, 288], family='other_sign', value=None, sign_type='give_way')
    direction = confirmed([690, 270, 744, 292], family='other_sign', value=None, sign_type='direction_right')
    current = [proposal([1052, 249, 1102, 316], 'pedestrian')]
    result = carry_over([give_way, direction], current, width=W, height=H, source_key='34/45', proposals=current)
    boxes = {item['detection_id']: item for item in result['detections']}
    assert result['matched'] == 0 and result['estimated'] == 2 and result['dropped'] == 0
    assert 'sign_type' not in boxes['pedestrian'] and 'carried_from' not in boxes['pedestrian']
    placed = [item for item in result['detections'] if item.get('check_position')]
    assert sorted(item['sign_type'] for item in placed) == ['direction_right', 'give_way']
    # The same shape, grown, is still found; a box cut by the edge is not held to its shape.
    wide = box_at(400, 330, 40)
    wide[1], wide[3] = 320, 340
    grown = closer(wide, 1.4)
    assert carry_over([confirmed(wide)], [proposal(grown, 'g')], width=W, height=H, source_key='1/1',
                      proposals=[proposal(grown, 'g')])['matched'] == 1
    edge = confirmed([1300, 300, 1340, 316])
    cut = proposal([1320, 290, W, 330], 'cut')
    assert carry_over([edge], [cut], width=W, height=H, source_key='1/1', proposals=[cut])['matched'] == 1


def test_a_slide_of_the_view_far_up_or_down_is_no_turn():
    # 5/46 -> 5/47 of a drive: an "other danger" sign and, on the next frame, only a proposal far to the right and up;
    # a slide of 417 px sideways and 110 px up "explained" it. Up or down the view moves little in a second.
    danger = confirmed([736, 366, 756, 391], family='other_sign', value=None, sign_type='danger_other')
    far = proposal([1204, 206, 1262, 259], 'far', family='cancellation')
    result = carry_over([danger], [far], width=W, height=H, source_key='5/46', proposals=[far])
    assert result['matched'] == 0 and result['estimated'] == 1
    boxes = {item['detection_id']: item for item in result['detections']}
    assert boxes['far']['family'] == 'cancellation' and 'carried_from' not in boxes['far']
    # A turn sideways is still one.
    assert test_a_turn_slides_the_whole_view() is None


def test_a_sign_not_found_is_placed_like_its_neighbour_and_must_be_checked():
    left = box_at(560, 330, 30)
    right = box_at(840, 330, 30)
    current = [proposal(closer(left, 1.4), 'p1')]
    result = carry_over([confirmed(left), confirmed(right, value=100)], current, width=W, height=H,
                        source_key='2/52', proposals=current)
    assert result['matched'] == 1 and result['estimated'] == 1
    placed = result['detections'][1]
    expected = closer(right, 1.4)
    assert placed['check_position'] is True and placed['carried_from'] == '2/52'
    assert placed['value_kph'] == 100 and placed['detection_id'] == 'previous-2-52-1'
    assert all(abs(a - b) < 1 for a, b in zip(placed['bbox_xyxy'], expected))
    assert unchecked_boxes(result['detections']) == [2]


def test_without_any_sign_found_the_old_position_is_kept_and_gone_signs_are_dropped():
    stays = box_at(900, 300, 30)
    gone = [W - 6, 300, W + 60, 360]
    result = carry_over([confirmed(stays), confirmed(gone), confirmed(box_at(300, 300, 30), family='not_a_sign',
                                                                        value=None)],
                        [], width=W, height=H, source_key='0/1')
    assert result['estimated'] == 1 and result['dropped'] == 1
    # A "not a sign" box is carried only onto a box of this frame.
    assert len(result['detections']) == 1
    assert result['detections'][0]['bbox_xyxy'] == [round(v, 1) for v in stays]
    # Unless it is on a vehicle (a truck's sticker): the truck ahead is still there.
    sticker = confirmed(box_at(300, 300, 30), family='not_a_sign', value=None, sign_type='on_vehicle',
                        applicability='other_road')
    result = carry_over([confirmed(stays), sticker], [], width=W, height=H, source_key='0/1')
    carried = result['detections'][1]
    assert carried['family'] == 'not_a_sign' and carried['sign_type'] == 'on_vehicle'
    assert carried['applicability'] == 'other_road' and carried['check_position'] is True


def test_boxes_the_reviewer_changed_keep_their_labels():
    sign = box_at(600, 330, 30)
    current = [proposal(closer(sign, 1.2), 'p1')]
    edited = [dict(current[0], family='maximum_speed', value_kph=60)]
    result = carry_over([confirmed(sign)], edited, width=W, height=H, source_key='1/1', proposals=current)
    assert result['kept'] == 1 and result['matched'] == 0 and result['estimated'] == 0
    assert result['detections'][0]['value_kph'] == 60


def test_a_number_readable_only_now_is_kept():
    sign = box_at(600, 330, 20)
    current = [proposal(closer(sign, 1.5), 'p1', family='maximum_speed', value=60)]
    result = carry_over([confirmed(sign, family='maximum_speed', value=None)], current, width=W, height=H,
                        source_key='1/1', proposals=current)
    item = result['detections'][0]
    assert item['family'] == 'maximum_speed' and item['value_kph'] == 60 and item['value_state'] == 'reviewed'
    assert item['applicability'] == 'own_road'


# ---------- the server: page API, REST, MCP, the save check ----------

KEYS = [(0, 0), (0, 1), (0, 2), (2, 0)]
AUTH = {'Authorization': 'Bearer ' + TOKEN}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path, keys=KEYS)
    running = RunningServer(tmp_path)
    yield running, {row['frame_key']: row for row in rows}
    running.close()


def apply_on_page(running, key, body):
    status, response, data = running.request('POST', f'/api/route/{ROUTE}/frame/{key}/apply-previous', body=body)
    return status, json_of(data)


def test_the_page_carries_the_previous_frame_and_the_save_waits_for_the_check(server):
    running, rows = server
    status, answer = apply_on_page(running, '0/1', {'run_id': RUN_ID, 'detections': []})
    assert status == 409 and answer['error'] == 'previous_not_confirmed'
    body = page_body(rows['0/0'], value_kph=50)
    body['detections'].append({'bbox_xyxy': [18, 2, 30, 14], 'family': 'other_sign', 'value_kph': None,
                               'applicability': 'own_road', 'detection_id': 'manual-1'})
    assert running.post_frame(ROUTE, '0/0', body)[0] == 200
    status, answer = apply_on_page(running, '0/1', {'run_id': RUN_ID, 'detections': []})
    assert status == 200 and answer['source_frame_key'] == '0/0' and answer['estimated'] == 2
    assert answer['to_check'] == [1, 2] and 'check' in answer['message']
    carried = answer['detections']
    assert carried[0]['value_kph'] == 50 and carried[0]['check_position'] is True
    save = page_body(rows['0/1'], drop_proposals=True)
    save['detections'] = carried
    status, response, data = running.post_frame(ROUTE, '0/1', save)
    assert status == 400 and json_of(data)['error'] == 'position_not_checked'
    for item in carried:
        del item['check_position']
    status, response, data = running.post_frame(ROUTE, '0/1', save)
    assert status == 200
    saved = json_of(data)['review']['detections']
    assert [item['value_kph'] for item in saved] == [50, None]
    assert all('carried_from' not in item and 'check_position' not in item for item in saved)
    status, answer = apply_on_page(running, '0/1', {'run_id': 'old-run'})
    assert status == 409 and answer['error'] == 'stale_report'


def test_a_frame_confirmed_before_takes_the_labels_of_the_frame_before(server):
    # A drive reviewed again: 0/1 was confirmed with a box drawn by hand, "unreadable"; 0/0 is now an end of limit.
    running, rows = server
    old = page_body(rows['0/1'], drop_proposals=True)
    old['detections'] = [{'bbox_xyxy': [18, 2, 30, 14], 'family': 'unreadable', 'value_kph': None,
                          'applicability': 'own_road', 'detection_id': 'manual-7'}]
    status, response, data = running.post_frame(ROUTE, '0/1', old)
    assert status == 200
    saved = json_of(data)['review']['detections']
    first = page_body(rows['0/0'], drop_proposals=True)
    first['detections'] = [{'bbox_xyxy': [19, 3, 29, 13], 'family': 'cancellation', 'value_kph': None,
                            'applicability': 'own_road', 'detection_id': 'manual-6'}]
    assert running.post_frame(ROUTE, '0/0', first)[0] == 200
    status, answer = apply_on_page(running, '0/1', {'run_id': RUN_ID, 'detections': saved})
    assert status == 200 and answer['matched'] == 1 and answer['kept'] == 0
    assert [item['family'] for item in answer['detections']] == ['cancellation']
    # The same box changed on screen first keeps what the reviewer set.
    changed = [dict(saved[0], family='zone', value_kph=30)]
    status, answer = apply_on_page(running, '0/1', {'run_id': RUN_ID, 'detections': changed})
    assert status == 200 and answer['kept'] == 1 and answer['detections'][0]['family'] == 'zone'


def test_rest_and_mcp_offer_it_too(server):
    running, rows = server
    path = f'/api/v1/routes/{ROUTE}/frames/0/1/apply-previous'
    assert running.request('POST', path, body={})[0] == 401
    assert running.request('GET', path, headers=AUTH)[0] == 405
    assert running.post_frame(ROUTE, '0/0', page_body(rows['0/0'], value_kph=60))[0] == 200
    status, response, data = running.request('POST', path, body={}, headers=AUTH)
    answer = json_of(data)
    assert status == 200 and answer['estimated'] == 1 and answer['save'] == {'run_id': RUN_ID, 'base_revision': 0}
    tools = ReviewTools(running.app, client='test')
    result = tools.apply_previous_frame(ROUTE, '0/1')
    assert result['detections'][0]['value_kph'] == 60 and result['to_check'] == [1]
    body = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
            'detections': result['detections']}
    status, response, data = running.request('POST', f'/api/v1/routes/{ROUTE}/frames/0/1/review', body=body,
                                             headers=AUTH)
    assert status == 400 and json_of(data)['error'] == 'position_not_checked'


def test_only_the_frame_just_before(server):
    running, rows = server
    assert running.post_frame(ROUTE, '0/2', page_body(rows['0/2']))[0] == 200
    # 2/0 follows 0/2 in the timeline, but segment 1 is missing: not the frame just before.
    status, answer = apply_on_page(running, '2/0', {'run_id': RUN_ID})
    assert status == 409 and answer['error'] == 'no_previous_frame'
    status, answer = apply_on_page(running, '0/0', {'run_id': RUN_ID})
    assert status == 409 and answer['error'] == 'no_previous_frame'


def test_tracks_follow_the_same_sign_across_consecutive_frames():
    from nnslr_tools.tracks import link_tracks
    sign = box_at(600, 330, 24)
    other = box_at(900, 320, 24)
    frames = []
    for position in range(4):
        boxes = [confirmed(closer(sign, 1.25 ** position))]
        if position < 2:
            boxes.append(confirmed(closer(other, 1.25 ** position), family='other_sign', value=None))
        frames.append({'frame_key': f'0/{position}', 'segment_index': 0, 'output_index': position,
                       'position': position, 'detections': boxes})
    # A gap: the frame at position 5 does not follow position 3, its sign starts a new track.
    frames.append({'frame_key': '0/5', 'segment_index': 0, 'output_index': 5, 'position': 5,
                   'detections': [confirmed(sign)]})
    lines = link_tracks(frames, width=W, height=H)
    speed = [line['track_id'] for line in lines if line['family'] == 'variable_display']
    assert len(set(speed[:4])) == 1 and speed[4] != speed[0]
    others = {line['track_id'] for line in lines if line['family'] == 'other_sign'}
    assert len(others) == 1 and others.isdisjoint(speed)
