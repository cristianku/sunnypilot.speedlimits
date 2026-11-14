"""box_source (#80): what last set a box's position, saved with the review and read by Fit box's training."""
from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from nnslr_tools.box_fit.refiner_train import box_source, confirmed_boxes
from nnslr_tools.carry_over import carry_over
from nnslr_tools.review_db import build_reviewed_frame
from review_helpers import ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, page_body

W, H = 1344, 760


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSLR_REVIEW_TOKEN', TOKEN)
    rows = make_report(tmp_path)
    running = RunningServer(tmp_path)
    yield running, {row['frame_key']: row for row in rows}, tmp_path
    running.close()


def test_a_save_keeps_the_source_of_each_box_and_counts_as_accepted(server):
    running, rows, _ = server
    body = page_body(rows['0/0'], manual_box=[18, 2, 30, 14])
    body['detections'][0]['box_source'] = 'fit'
    body['detections'][1]['box_source'] = 'hand'
    status, _, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 200
    saved = json_of(data)['review']['detections']
    assert [item.get('box_source') for item in saved] == ['fit', 'hand']
    # A box saved again without one (a review from before #80) has none.
    again = page_body(rows['0/0'], base_revision=json_of(data)['review']['revision'])
    status, _, data = running.post_frame(ROUTE, '0/0', again)
    assert status == 200 and 'box_source' not in json_of(data)['review']['detections'][0]


@pytest.mark.parametrize('value', ['by hand', 7, 'model'])
def test_a_source_that_is_not_one_of_the_list_is_refused(server, value):
    running, rows, _ = server
    body = page_body(rows['0/0'])
    body['detections'][0]['box_source'] = value
    status, _, data = running.post_frame(ROUTE, '0/0', body)
    assert status == 400 and json_of(data)['error'] == 'invalid_box_source'


def test_an_ai_reviewer_s_own_boxes_are_the_ai_s():
    proposal = {'bbox_xyxy': [10, 10, 30, 30], 'family': 'maximum_speed', 'value_kph': 50, 'detection_id': 'p1'}
    row = {'frame_key': '0/0', 'detections': [proposal], 'route_id': ROUTE}
    kept, moved = copy.deepcopy(proposal), dict(copy.deepcopy(proposal), detection_id='p2', bbox_xyxy=[11, 10, 30, 30])
    frame = build_reviewed_frame(row, {'detections': [kept, moved], 'reviewer': 'claude'}, 'now', reviewer_kind='ai',
                                 channel='rest')
    assert [d.get('box_source') for d in frame['detections']] == [None, 'ai']
    person = build_reviewed_frame(row, {'detections': [moved], 'reviewer': 'Cristian'}, 'now')
    assert 'box_source' not in person['detections'][0]
    # A detection_id that is no string: built without a crash, for the checks of the save to refuse it.
    odd = dict(copy.deepcopy(proposal), detection_id=['x'])
    build_reviewed_frame(row, {'detections': [odd], 'reviewer': 'claude'}, 'now', reviewer_kind='ai', channel='rest')


def test_a_box_that_says_its_family_does_not_keep_the_kind_of_its_proposal():
    """14 Nov 2026: a proposal of an Other sign has its kind (sign_types.with_sign_type). A box the reviewer made a
    speed limit, or left an Other sign without a kind, must not keep it; a box that only names its proposal does."""
    proposal = {'bbox_xyxy': [10, 10, 30, 30], 'family': 'other_sign', 'value_kph': None, 'sign_type': 'give_way',
                'model_label': 'other_sign:give_way', 'detection_id': 'p1'}
    row = {'frame_key': '0/0', 'detections': [proposal], 'route_id': ROUTE}

    def saved(item):
        return build_reviewed_frame(row, {'detections': [item], 'reviewer': 'Cristian'}, 'now')['detections'][0]

    limit = saved({'detection_id': 'p1', 'bbox_xyxy': [10, 10, 30, 30], 'family': 'maximum_speed', 'value_kph': 50})
    assert limit['family'] == 'maximum_speed' and 'sign_type' not in limit
    assert 'sign_type' not in saved({'detection_id': 'p1', 'bbox_xyxy': [10, 10, 30, 30], 'family': 'other_sign'})
    assert saved({'detection_id': 'p1', 'family': 'other_sign', 'sign_type': 'stop'})['sign_type'] == 'stop'
    assert saved({'detection_id': 'p1'})['sign_type'] == 'give_way'


def test_apply_previous_marks_the_boxes_it_places_and_leaves_the_ones_it_labels():
    old = [{'bbox_xyxy': [545, 315, 575, 345], 'family': 'maximum_speed', 'value_kph': 80, 'detection_id': 'o1',
            'applicability': 'own_road', 'supplementary_panel': 'none'},
           {'bbox_xyxy': [825, 315, 855, 345], 'family': 'maximum_speed', 'value_kph': 100, 'detection_id': 'o2',
            'applicability': 'own_road', 'supplementary_panel': 'none'}]
    # The left sign grown one second later, found by the model and once moved by hand; the right one not found.
    now = {'bbox_xyxy': [543.0, 309.0, 585.0, 351.0], 'family': 'other_sign', 'value_kph': None, 'score': 0.6,
           'detection_id': 'p1', 'applicability': 'unresolved', 'supplementary_panel': 'unknown'}
    here = dict(now, box_source='hand')
    result = carry_over(old, [here], width=W, height=H, source_key='0/0', proposals=[now])
    matched, placed = result['detections']
    assert result['matched'] == 1 and matched['box_source'] == 'hand' and matched['value_kph'] == 80
    assert placed['box_source'] == 'carried' and placed['check_position'] is True
    # The source is no label: the box the model drew still counts as not edited.
    result = carry_over(old, [now], width=W, height=H, source_key='0/0', proposals=[now])
    assert 'box_source' not in result['detections'][0] and result['matched'] == 1


def test_the_training_reads_each_box_s_source(tmp_path):
    assert box_source({'box_source': 'hand', 'bbox_xyxy': [1, 1, 9, 9]}, {}) == 'hand'
    assert box_source({'detection_id': 'p', 'bbox_xyxy': [1, 1, 9, 9]}, {'p': [1, 1, 9, 9]}) == 'model'
    assert box_source({'detection_id': 'p', 'bbox_xyxy': [1, 1, 9, 10]}, {'p': [1, 1, 9, 9]}) == 'unknown'
    (tmp_path / 'f.jpg').write_bytes(b'x')
    db = tmp_path / 'review.sqlite'
    frame = {'image_path': 'f.jpg', 'width': W, 'height': H,
             'source_proposals': [{'detection_id': 'p', 'bbox_xyxy': [10, 10, 30, 30]}],
             'detections': [{'detection_id': 'p', 'bbox_xyxy': [10, 10, 30, 30], 'family': 'maximum_speed'},
                            {'detection_id': 'm', 'bbox_xyxy': [50, 10, 70, 30], 'family': 'other_sign',
                             'box_source': 'hand'},
                            {'detection_id': 'f', 'bbox_xyxy': [90, 10, 110, 30], 'family': 'other_sign',
                             'box_source': 'fit'}]}
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE frame_reviews (route_id, frame_key, payload, reviewer, confirmed)')
        conn.execute('INSERT INTO frame_reviews VALUES (?, ?, ?, ?, 1)', (ROUTE, '0/0', json.dumps(frame), 'Cristian'))
    assert [b['source'] for b in confirmed_boxes(db, tmp_path)] == ['model', 'hand', 'fit']
