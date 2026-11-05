"""The labels of our classifier as proposals of the review page (nnslr preannotate --proposer ours)."""
from nnslr_tools._chain_proposer import car_speed, proposal_of, proposal_panel


def test_classifier_labels_become_proposals():
    assert proposal_of('maximum_speed:60') == ('maximum_speed', 60)
    assert proposal_of('maximum_speed:unknown') == ('maximum_speed', None)
    assert proposal_of('variable_display:100') == ('variable_display', 100)
    assert proposal_of('road_marking:30') == ('road_marking_candidate', 30)
    assert proposal_of('other_sign') == ('other_sign', None)
    assert proposal_of('cancellation') == ('cancellation', None)
    assert proposal_of('not_a_sign') is None


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
