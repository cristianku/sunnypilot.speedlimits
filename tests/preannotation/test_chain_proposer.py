"""The labels of our classifier as proposals of the review page (nnslr preannotate --proposer ours)."""
import json

from nnslr_tools._chain_proposer import car_speed, proposal_of, proposal_panel
from nnslr_tools.review import report_rows
from nnslr_tools.sign_types import class_sign_type, with_sign_type


def test_classifier_labels_become_proposals():
    assert proposal_of('maximum_speed:60') == ('maximum_speed', 60)
    assert proposal_of('maximum_speed:unknown') == ('maximum_speed', None)
    assert proposal_of('variable_display:100') == ('variable_display', 100)
    assert proposal_of('road_marking:30') == ('road_marking_candidate', 30)
    assert proposal_of('other_sign') == ('other_sign', None)
    assert proposal_of('cancellation') == ('cancellation', None)
    assert proposal_of('not_a_sign') is None


def test_an_other_sign_class_names_the_kind_of_its_proposal():
    """14 Nov 2026: Prepare kept only "Other sign" of what the classifier read as a roundabout or a main road."""
    assert class_sign_type('other_sign:roundabout') == 'roundabout'
    assert class_sign_type('other_sign:main_road') == 'main_road'
    # No kind: a class without one, one the review does not know, any other family.
    assert class_sign_type('other_sign') is None and class_sign_type('other_sign:no_such_kind') is None
    assert class_sign_type('maximum_speed:50') is None and class_sign_type('not_a_sign') is None
    assert class_sign_type('maximum_speed_hours:30:22-6') is None


def test_a_report_from_before_shows_the_kind_its_class_names(tmp_path):
    """The reports Prepare wrote without sign_type: the kind comes from model_label when the report is read. A kind
    already there (a later report) and the other families stay as they are."""
    rows = [{'frame_key': '35/19', 'detections': [
        {'family': 'other_sign', 'model_label': 'other_sign:roundabout', 'value_kph': None},
        {'family': 'other_sign', 'model_label': 'other_sign:roundabout', 'sign_type': 'main_road'},
        {'family': 'other_sign', 'model_label': 'other_sign'},
        {'family': 'other_sign', 'model_label': None},
        {'family': 'maximum_speed', 'model_label': 'maximum_speed:50', 'value_kph': 50}]}]
    (tmp_path / 'preannotations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    found = report_rows(tmp_path)[0]['detections']
    assert [item.get('sign_type') for item in found] == ['roundabout', 'main_road', None, None, None]
    assert with_sign_type({'family': 'other_sign', 'model_label': 'other_sign:give_way'})['sign_type'] == 'give_way'


def test_a_number_with_hours_is_a_proposal_of_a_limit_with_its_panel():
    assert proposal_of('maximum_speed_hours:30') == ('maximum_speed', 30)
    assert proposal_panel('maximum_speed_hours:30') == 'time'
    assert proposal_panel('maximum_speed:30') == 'unknown' and proposal_panel('other_sign:give_way') == 'unknown'
    assert proposal_of('maximum_speed_hours:30:22-6') == ('maximum_speed', 30)
    assert proposal_panel('maximum_speed_hours:30:22-6') == 'time'


def test_the_decider_gets_the_car_speed_of_each_frame_from_its_route(tmp_path):
    folder = tmp_path / 'derived' / 'gps' / '0000012a--19b6717805'
    folder.mkdir(parents=True)
    (folder / 'positions.jsonl').write_text('{"frame_key": "3/36", "speed_mps": 22.5}\n'
                                           '{"frame_key": "3/40", "speed_mps": null}\n')
    positions = {}
    frame = {'frame_key': '3/36', 'route_id': '0000012a--19b6717805'}
    assert car_speed(tmp_path, positions, frame, 'several') == 22.5
    assert car_speed(tmp_path, positions, {**frame, 'frame_key': '3/40'}, 'several') is None
    # A route without positions, or a measure's "several" with frames that do not say their route: no speed.
    assert car_speed(tmp_path, positions, {'frame_key': '3/36', 'route_id': '0000012b--0123456789'}, 'x') is None
    assert car_speed(tmp_path, positions, {'frame_key': '3/36'}, 'several') is None
