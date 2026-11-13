"""The AI reviewer: what the model is asked, how its answer becomes a suggestion in frame pixels, route runs
(dry run and save), the measure and the Test of Settings. The model is fake_llm's server; nothing is saved
outside the temporary data root."""
from __future__ import annotations

import json
import re

import pytest

from fake_llm import NO_SIGN_ANSWER, FakeLlm, answer_json
from nnslr_tools import ai_review, llm_providers
from nnslr_tools.ai_review import parse_answer, to_suggestion
from nnslr_tools.annotations import reviewed_frame_annotations, validate_dataset
from nnslr_tools.llm_providers import LlmError
from nnslr_tools.review_db import ReviewRejected, ReviewStore, db_path, import_from_db, save_frame_review
from nnslr_tools.review_service import ReviewService
from review_helpers import HEIGHT, ROUTE, RUN_ID, WIDTH, make_report, page_body

# Frames 0/0, 0/2 and 0/4 have one speed proposal [1, 2, 15, 20] (30 km/h); 0/1, 0/3 and 0/5 none.
KEYS = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
SENT = [{'detection_id': 'a', 'bbox_xyxy': [100, 100, 200, 200], 'family': 'maximum_speed', 'value_kph': 80,
         'applicability': 'own_road', 'occlusion': 'partial', 'supplementary_panel': 'none', 'temporary': True},
        {'detection_id': 'b', 'bbox_xyxy': [300, 300, 400, 400], 'family': 'other_sign', 'value_kph': None,
         'sign_type': 'give_way', 'applicability': 'unresolved'}]


@pytest.fixture
def site(tmp_path):
    rows = make_report(tmp_path, keys=KEYS)
    service = ReviewService(tmp_path, ReviewStore(db_path(tmp_path)))
    return service, {row['frame_key']: row for row in rows}


def fake_provider(root, fake, model='fake-vl', reviewer=True):
    """A provider of the fake server; the reviewer has passed its Test (only such a one reviews or saves)."""
    saved = llm_providers.save_provider(root, {'name': 'fake', 'base_url': fake.base_url, 'model': model,
                                               'key': 'sk-fake-key-123456'})
    if reviewer:
        llm_providers.record_test(root, saved['id'], {'ok': True, 'message': 'Sees images · 1.0 s'})
        llm_providers.set_reviewer(root, saved['id'])
    return llm_providers.provider(root, saved['id'])


def human(service, row, **options):
    body = page_body(row, **options)
    return save_frame_review(service.store, service.root, route_id=ROUTE, run_id=RUN_ID, report_row=row, body=body,
                             client='test')


def frame_text(body):
    return body['messages'][1]['content'][0]['text']


# ---------- parse_answer ----------

def test_parse_strips_think_blocks_and_fences():
    text = '<think>the sign {says} 80</think>\n```json\n{"boxes": [{"n": 1, "verdict": "KEEP"}], "no_sign": false}\n```'
    answer = parse_answer(text)
    assert answer['boxes'] == [{'n': 1, 'verdict': 'keep'}]
    assert answer['repaired'] is False
    assert parse_answer('reasoning only</think>{"added": []}')['added'] == []
    assert parse_answer('{"no_sign": true} and then {more}')['no_sign'] is True


def test_parse_closes_a_cut_off_answer():
    answer = parse_answer('{"boxes": [{"n": 1, "verdict": "keep"}], "added": [{"box": [1, 2, 3, 4], "notes": "cut')
    assert answer['repaired'] is True
    assert answer['added'][0]['notes'] == 'cut'
    # Cut back to the last comma: box 2 lost its verdict, so it stays unjudged.
    answer = parse_answer('{"boxes": [{"n": 1, "verdict": "keep"}, {"n": 2, "verdict":')
    assert answer['boxes'] == [{'n': 1, 'verdict': 'keep'}, {'n': 2}]
    with pytest.raises(LlmError) as error:
        parse_answer('I cannot see any image.')
    assert error.value.status == 502 and error.value.reason == 'ai_bad_answer'
    with pytest.raises(LlmError):
        parse_answer('{"boxes": [1, 2,, 3]}')


def test_parse_accepts_aliases():
    answer = parse_answer(json.dumps({'proposals': [{'n': 1, 'verdict': 'Correct', 'bbox_2d': [1, 2, 3, 4],
                                                     'family': 'Speed_Limit'}],
                                      'added': [{'bbox': [5, 6, 7, 8], 'family': 'end_of_limit'},
                                                {'box': [1, 1, 9, 9], 'family': 'road_marking'}]}))
    assert answer['boxes'][0]['box'] == [1, 2, 3, 4]
    assert answer['boxes'][0]['family'] == 'maximum_speed' and answer['boxes'][0]['verdict'] == 'correct'
    assert [item['family'] for item in answer['added']] == ['cancellation', 'road_marking_candidate']
    assert answer['added'][0]['box'] == [5, 6, 7, 8]


# ---------- to_suggestion ----------

def suggest(answer, boxes=SENT, width=1000, height=1000):
    return to_suggestion(parse_answer(json.dumps(answer)), json.loads(json.dumps(boxes)), width, height)


def test_keep_correct_delete_and_add():
    result = suggest({'boxes': [{'n': 1, 'verdict': 'keep', 'applicability': 'own_road'},
                                {'n': 2, 'verdict': 'delete', 'notes': 'advert'}],
                      'added': [{'box': [500, 500, 600, 700], 'family': 'maximum_speed', 'value_kph': 60,
                                 'applicability': 'other_road', 'visibility': 'clear'}]},
                     width=2000, height=1000)
    kept, added = result['detections']
    assert kept['detection_id'] == 'a' and kept['ai_action'] == 'keep'
    assert kept['bbox_xyxy'] == [100, 100, 200, 200] and kept['value_kph'] == 80
    assert kept['occlusion'] == 'partial' and kept['supplementary_panel'] == 'none' and kept['temporary'] is True
    assert added == {'detection_id': 'ai-1', 'bbox_xyxy': [1000.0, 500.0, 1200.0, 700.0], 'family': 'maximum_speed',
                     'value_kph': 60, 'applicability': 'other_road', 'visibility': 'clear', 'occlusion': 'unknown',
                     'supplementary_panel': 'unknown', 'temporary': None, 'variable_display': None, 'notes': '',
                     'ai_action': 'add'}
    assert result['deleted'] == [{'number': 2, 'detection_id': 'b', 'family': 'other_sign', 'value_kph': None,
                                  'bbox_xyxy': [300, 300, 400, 400], 'notes': 'advert'}]
    assert result['complete'] is True and result['no_sign'] is False and result['warnings'] == []


def test_correct_takes_the_model_box_and_labels_but_never_echoes_the_old_value():
    result = suggest({'boxes': [{'n': 1, 'verdict': 'correct', 'box': [110, 90, 190, 210], 'family': 'zone'},
                                {'n': 2, 'verdict': 'correct', 'family': 'other_sign', 'sign_type': 'stop'}]},
                     width=500, height=1000)
    first, second = result['detections']
    assert first['detection_id'] == 'a' and first['ai_action'] == 'correct'
    assert first['bbox_xyxy'] == [55.0, 90.0, 95.0, 210.0]
    assert first['family'] == 'zone' and first['value_kph'] is None and first['applicability'] == 'own_road'
    assert first['occlusion'] == 'partial'
    assert second['bbox_xyxy'] == [300, 300, 400, 400] and second['sign_type'] == 'stop'


def test_a_box_with_its_plate_of_hours_keeps_its_place():
    held = [{**SENT[0], 'supplementary_panel': 'time'}]
    result = suggest({'boxes': [{'n': 1, 'verdict': 'correct', 'box': [110, 90, 190, 210], 'family': 'maximum_speed',
                                 'value_kph': 30}]}, held)
    assert result['detections'][0]['bbox_xyxy'] == held[0]['bbox_xyxy']
    assert result['detections'][0]['supplementary_panel'] == 'time'
    assert any('its plate' in warning for warning in result['warnings'])


def test_scale_of_the_coordinates_is_detected():
    unit = suggest({'added': [{'box': [0.1, 0.2, 0.3, 0.4], 'family': 'other_sign'}]}, [], 1164, 874)
    assert unit['detections'][0]['bbox_xyxy'] == [116.4, 174.8, 349.2, 349.6]
    pixels = suggest({'added': [{'box': [1010, 20, 1100, 120], 'family': 'other_sign'}]}, [], 1164, 874)
    assert pixels['detections'][0]['bbox_xyxy'] == [1010.0, 20.0, 1100.0, 120.0]
    relative = suggest({'added': [{'box': [500, 500, 600, 600], 'family': 'other_sign'}]}, [], 1164, 874)
    assert relative['detections'][0]['bbox_xyxy'] == [582.0, 437.0, 698.4, 524.4]


def test_boxes_are_clamped_sorted_and_tiny_ones_dropped():
    result = suggest({'added': [{'box': [1200, 900, -50, 990], 'family': 'maximum_speed', 'value_kph': 50},
                                {'box': [10, 10, 11, 500], 'family': 'other_sign'}]}, [], 100, 100)
    assert [item['bbox_xyxy'] for item in result['detections']] == [[0.0, 90.0, 100.0, 99.0]]
    assert any('smaller than 2 px' in warning for warning in result['warnings'])


def test_labels_follow_the_review_rules():
    result = suggest({'added': [
        {'box': [0, 0, 100, 100], 'family': 'cancellation', 'value_kph': 50, 'applicability': 'ours'},
        {'box': [100, 0, 200, 100], 'family': 'maximum_speed', 'value_kph': 300, 'visibility': 'great'},
        {'box': [200, 0, 300, 100], 'family': 'maximum_speed', 'value_kph': '80', 'sign_type': 'stop'},
        {'box': [300, 0, 400, 100], 'family': 'other_sign', 'sign_type': 'dragon', 'value_kph': 3.5},
        {'box': [400, 0, 500, 100], 'family': 'a dragon'},
        {'box': [500, 0, 600, 100], 'family': 'other_sign', 'sign_type': 'town_entry', 'notes': None},
    ]}, [])
    cancel, high, speed, other, unknown, town = result['detections']
    assert cancel['value_kph'] is None and cancel['notes'] == 'crossed-out 50'
    assert cancel['applicability'] == 'unresolved'
    assert high['value_kph'] is None and high['visibility'] == 'unknown'
    assert speed['value_kph'] == 80 and 'sign_type' not in speed
    assert other['sign_type'] is None and other['value_kph'] is None
    assert unknown['family'] == 'unreadable' and unknown['notes'] == 'number not readable'
    assert town['sign_type'] == 'town_entry' and town['notes'] == ''
    assert any('unknown family' in warning for warning in result['warnings'])


def test_unjudged_boxes_make_it_incomplete():
    result = suggest({'boxes': [{'n': 2, 'verdict': 'keep'}, {'n': 7, 'verdict': 'keep'},
                                {'n': 2, 'verdict': 'delete'}]})
    first, second = result['detections']
    assert first['ai_action'] == 'unjudged' and first['value_kph'] == 80 and first['detection_id'] == 'a'
    assert second['ai_action'] == 'keep'
    assert result['complete'] is False
    assert 'box 1 not judged' in result['warnings']
    assert 'an answer box without a valid number was ignored' in result['warnings']
    assert 'box 2 judged twice: the first verdict counts' in result['warnings']


def test_duplicates_and_contradictions():
    result = suggest({'boxes': [{'n': 1, 'verdict': 'keep'}, {'n': 2, 'verdict': 'delete'}],
                      'added': [{'box': [100, 100, 200, 200], 'family': 'maximum_speed', 'value_kph': 80},
                                {'box': [700, 700, 800, 800], 'family': 'zone', 'value_kph': 30},
                                {'box': [700, 700, 800, 800], 'family': 'zone', 'value_kph': 20}],
                      'no_sign': True})
    assert [item['detection_id'] for item in result['detections']] == ['a', 'ai-2']
    assert result['no_sign'] is False
    assert 'the answer says no sign but adds boxes: the boxes stay' in result['warnings']
    assert sum('same box' in warning for warning in result['warnings']) == 2


def test_added_ids_never_collide_with_the_sent_ones():
    boxes = [dict(SENT[0], detection_id='ai-1')]
    result = suggest({'boxes': [{'n': 1, 'verdict': 'keep'}],
                      'added': [{'box': [0, 0, 50, 50], 'family': 'other_sign', 'sign_type': 'stop'}]}, boxes)
    assert [item['detection_id'] for item in result['detections']] == ['ai-1', 'ai-2']


def test_no_sign_answer():
    result = suggest({'boxes': [{'n': 1, 'verdict': 'delete'}, {'n': 2, 'verdict': 'delete'}], 'no_sign': True})
    assert result['detections'] == [] and result['no_sign'] is True and result['complete'] is True
    assert len(result['deleted']) == 2


def test_a_repaired_answer_is_never_complete():
    result = to_suggestion(parse_answer('{"boxes": [], "added": [{"box": [0, 0, 500, 500], "family": "zone"'),
                           [], 100, 100)
    assert result['complete'] is False and len(result['detections']) == 1


# ---------- suggest_frame ----------

def test_suggest_frame_asks_with_the_image_and_returns_pixels(site, tmp_path):
    service, rows = site
    answer = answer_json(boxes=[{'n': 1, 'verdict': 'correct', 'box': [0, 0, 500, 500], 'family': 'maximum_speed',
                                 'value_kph': 50, 'applicability': 'own_road'}],
                         added=[{'box': [500, 500, 1000, 1000], 'family': 'other_sign', 'sign_type': 'give_way'}])
    with FakeLlm(answer) as fake:
        provider = fake_provider(tmp_path, fake)
        result = ai_review.suggest_frame(service, ROUTE, '0/0', provider, run_id=RUN_ID)
    body = fake.requests[0]
    assert body['messages'][0] == {'role': 'system', 'content': ai_review.SYSTEM_PROMPT}
    text = frame_text(body)
    assert text.startswith(f'Frame {ROUTE} 0/0, {WIDTH}x{HEIGHT} pixels.')
    # The proposal [1, 2, 15, 20] of a 32x24 frame, in 0-1000.
    assert '1. [31,83,469,833] maximum_speed 30, unresolved' in text
    assert 'Answer with this JSON only: ' in text
    image = body['messages'][1]['content'][1]
    assert image['type'] == 'image_url' and image['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert result['ok'] is True and result['route_id'] == ROUTE and result['frame_key'] == '0/0'
    assert result['run_id'] == RUN_ID and result['reviewer'] == 'ai/fake/fake-vl'
    assert result['provider'] == {'id': 'fake', 'name': 'fake', 'model': 'fake-vl'}
    corrected, added = result['detections']
    assert corrected['detection_id'] == '0/0/0' and corrected['bbox_xyxy'] == [0.0, 0.0, 16.0, 12.0]
    assert added['bbox_xyxy'] == [16.0, 12.0, 32.0, 24.0] and added['detection_id'] == 'ai-1'
    for detection in result['detections']:
        x1, y1, x2, y2 = detection['bbox_xyxy']
        assert 0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT
    assert result['message'].startswith('fake-vl: 2 signs (1 corrected, 1 added) · ')
    assert 'sk-fake-key-123456' not in json.dumps(result)


def test_suggest_frame_sends_the_page_boxes_and_hints(site, tmp_path):
    service, rows = site
    hints = tmp_path / 'derived' / 'hints' / ROUTE / 'second.jsonl'
    hints.parent.mkdir(parents=True)
    hints.write_text(json.dumps({'route_id': ROUTE, 'frame_key': '0/1', 'segment_index': 0, 'output_index': 1,
                                 'source': 'test', 'signs': [{'type': 'maximum_speed', 'value_kph': 80,
                                                              'confidence': 'high', 'box': [16, 12, 32, 24]}]})
                     + '\n')
    with FakeLlm() as fake:
        provider = fake_provider(tmp_path, fake)
        edits = [{'bbox_xyxy': [-5, 0, 16, 40], 'family': 'other_sign', 'sign_type': 'stop', 'notes': 'x'},
                 {'bbox_xyxy': [1, 2], 'family': 'zone'}, 'junk']
        result = ai_review.suggest_frame(service, ROUTE, '0/1', provider, detections=edits)
        text = frame_text(fake.requests[0])
        assert '1. [0,0,500,1000] other_sign stop, unresolved' in text
        assert 'Second reading (hints only, may be wrong): 80 at [500,500,1000,1000]' in text
        # The no-sign answer leaves the box sent unjudged.
        assert result['detections'][0]['detection_id'] == 'box-1'
        assert result['detections'][0]['ai_action'] == 'unjudged' and result['complete'] is False
        ai_review.suggest_frame(service, ROUTE, '0/1', provider, detections=[])
        assert 'Boxes already on this frame, numbered:\nnone\n' in frame_text(fake.requests[1])
        with pytest.raises(ReviewRejected) as error:
            ai_review.suggest_frame(service, ROUTE, '0/1', provider, run_id='20261021T100000Z-ffffffff')
        assert error.value.status == 409 and error.value.reason == 'stale_report'
        with pytest.raises(ReviewRejected) as error:
            ai_review.suggest_frame(service, ROUTE, '0/1', provider, detections=[{'bbox_xyxy': [0, 0, 1, 1]}] * 101)
        assert error.value.status == 400 and error.value.reason == 'invalid_detections'
    assert len(fake.requests) == 2


def test_the_original_image_is_sent_when_the_preview_is_missing(site, tmp_path):
    service, rows = site
    (tmp_path / 'derived' / 'preannotations' / ROUTE / RUN_ID / 'images' / '00000000.jpg').unlink()
    with FakeLlm() as fake:
        ai_review.suggest_frame(service, ROUTE, '0/0', fake_provider(tmp_path, fake))
    assert fake.requests[0]['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


# ---------- review_route ----------

def keep_everything(body):
    count = len(re.findall(r'^\d+\. \[', frame_text(body), re.M))
    if not count:
        return NO_SIGN_ANSWER
    return answer_json(boxes=[{'n': n, 'verdict': 'keep', 'applicability': 'own_road'} for n in range(1, count + 1)])


def test_a_dry_run_writes_its_suggestions_and_saves_nothing(site, tmp_path):
    service, rows = site
    human(service, rows['0/2'], value_kph=60)
    lines = []
    with FakeLlm(keep_everything) as fake:
        provider = fake_provider(tmp_path, fake)
        summary = ai_review.review_route(service, ROUTE, provider, save=False, log=lines.append)
    assert summary['frames'] == 5 and summary['suggested'] == 5 and summary['saved'] == 0
    assert summary['no_sign'] == 3 and summary['failed'] == 0 and summary['stopped'] is None
    assert summary['reviewer'] == 'ai/fake/fake-vl' and summary['save'] is False
    assert summary['path'].startswith(f'ai_reviews/{ROUTE}/') and summary['path'].endswith('-fake.json')
    written = json.loads((tmp_path / summary['path']).read_text())
    assert [frame['frame_key'] for frame in written['frames']] == ['0/0', '0/1', '0/3', '0/4', '0/5']
    assert {frame['outcome'] for frame in written['frames']} == {'dry_run'}
    assert written['frames'][0]['detections'][0]['ai_action'] == 'keep'
    assert written['run_id'] == RUN_ID and written['finished_at'] and written['save'] is False
    assert [review['frame_key'] for review in service.store.reviews(ROUTE, RUN_ID)] == ['0/2']
    assert lines[0].startswith('[1/5] 0/0: 1 sign (1 kept) · ') and lines[0].endswith(' s · dry run')
    assert lines[1].startswith('[2/5] 0/1: no sign · ')
    assert lines[-1] == f'AI review done: 5 suggested, 0 skipped, 0 failed. Suggestions: {summary["path"]}'
    assert not list(tmp_path.rglob('*.tmp'))
    assert 'sk-fake-key-123456' not in (tmp_path / summary['path']).read_text()


def test_a_save_run_records_final_ai_reviews_that_the_import_takes(site, tmp_path):
    service, rows = site
    human(service, rows['0/0'], value_kph=60)

    def answer(body):
        text = frame_text(body)
        if ' 0/2,' in text:
            # A person confirms 0/3 while the AI looks at 0/2.
            human(service, rows['0/3'], drop_proposals=True)
        if ' 0/4,' in text:
            return answer_json(added=[{'box': [0, 0, 300, 300], 'family': 'zone'}])
        return keep_everything(body)
    lines = []
    with FakeLlm(answer) as fake:
        provider = fake_provider(tmp_path, fake)
        summary = ai_review.review_route(service, ROUTE, provider, save=True, limit=10, log=lines.append)
    assert (summary['saved'], summary['skipped'], summary['failed']) == (3, 1, 1)
    outcomes = {frame['frame_key']: (frame['outcome'], frame['reason'])
                for frame in json.loads((tmp_path / summary['path']).read_text())['frames']}
    assert outcomes == {'0/1': ('saved', None), '0/2': ('saved', None), '0/3': ('skipped', 'already_confirmed'),
                        '0/4': ('failed', 'incomplete'), '0/5': ('saved', None)}
    assert lines[-1].startswith('AI review done: 3 saved, 1 skipped, 1 failed. Suggestions: ai_reviews/')
    reviews = {review['frame_key']: review for review in service.store.reviews(ROUTE, RUN_ID)}
    for key in ('0/1', '0/2', '0/5'):
        review = reviews[key]
        assert review['reviewer'] == 'ai/fake/fake-vl' and review['reviewer_kind'] == 'ai'
        assert review['channel'] == 'web' and review['confirmed'] is True
        payload = review['payload']
        assert payload['label_basis'] == 'ai_review' and payload['review_channel'] == 'web'
        assert all('ai_action' not in detection for detection in payload['detections'])
        annotations = reviewed_frame_annotations(payload, '0' * 64, dataset_kind='gold', name='test')
        assert validate_dataset(annotations, tmp_path, dataset_kind='gold')['valid'] is True
    assert reviews['0/2']['payload']['detections'][0]['detection_id'] == '0/2/0'
    assert reviews['0/2']['payload']['detections'][0]['applicability'] == 'own_road'
    assert reviews['0/0']['reviewer_kind'] == 'human' and reviews['0/3']['reviewer_kind'] == 'human'
    assert '0/4' not in reviews
    # Saved by the site's AI reviewer: final since 13 Nov 2026 (#79), confirmed and imported as AI reviews.
    from nnslr_tools.review_db import public_review
    assert not any(public_review(reviews[key])['needs_person'] for key in ('0/0', '0/1', '0/2', '0/5'))
    assert service.store.waiting_count(ROUTE, RUN_ID) == 0 and service.store.confirmed_count(ROUTE, RUN_ID) == 5
    report = import_from_db(tmp_path, tmp_path / 'dataset.jsonl', store=service.store)
    assert report['waiting_for_a_person_frame_count'] == 0
    imported = [json.loads(line) for line in (tmp_path / 'dataset.jsonl').read_text().splitlines()]
    assert {row['output_index'] for row in imported if row['provenance']['label_basis'] == 'ai_review'} >= {1, 2, 5}
    assert {row['output_index'] for row in imported} >= {0, 1, 2, 3, 5}
    # A person corrects one: at the next import it is theirs.
    human(service, rows['0/2'])
    import_from_db(tmp_path, tmp_path / 'dataset.jsonl', store=service.store)
    imported = [json.loads(line) for line in (tmp_path / 'dataset.jsonl').read_text().splitlines()]
    assert {row['provenance']['label_basis'] for row in imported if row['output_index'] == 2} == {'human_review'}


def test_a_human_confirmation_during_the_answer_is_never_overwritten(site, tmp_path):
    service, rows = site

    def answer(body):
        if ' 0/0,' in frame_text(body):
            human(service, rows['0/0'], value_kph=60)
        return keep_everything(body)
    with FakeLlm(answer) as fake:
        summary = ai_review.review_route(service, ROUTE, fake_provider(tmp_path, fake), save=True, limit=1,
                                         log=lambda text: None)
    frame = json.loads((tmp_path / summary['path']).read_text())['frames'][0]
    assert frame['outcome'] == 'skipped' and frame['reason'] in ('revision_conflict', 'human_review_exists')
    review = service.store.review(ROUTE, RUN_ID, '0/0')
    assert review['reviewer_kind'] == 'human' and review['payload']['detections'][0]['value_kph'] == 60


def test_failures_do_not_stop_the_run(site, tmp_path):
    service, rows = site
    lines = []
    with FakeLlm('no json here') as fake:
        summary = ai_review.review_route(service, ROUTE, fake_provider(tmp_path, fake), save=True, limit=2,
                                         log=lines.append)
    assert summary['failed'] == 2 and summary['saved'] == 0 and summary['frames'] == 2
    assert lines[0] == '[1/2] 0/0: failed: The AI answer has no JSON object.'
    assert service.store.reviews(ROUTE, RUN_ID) == []


def test_a_rebuilt_report_stops_the_run(site, tmp_path):
    service, rows = site
    base = tmp_path / 'derived' / 'preannotations' / ROUTE

    def answer(body):
        if ' 0/0,' in frame_text(body):
            new_run = '20261022T100000Z-0123abcd'
            (base / RUN_ID).rename(base / new_run)
            (base / 'latest.json').write_text(json.dumps({'run_id': new_run}))
        return NO_SIGN_ANSWER
    with FakeLlm(answer) as fake:
        summary = ai_review.review_route(service, ROUTE, fake_provider(tmp_path, fake), save=True,
                                         log=lambda text: None)
    assert summary['stopped'] == 'stale_report' and summary['saved'] == 0


def test_saving_needs_a_provider_that_passed_its_test(site, tmp_path):
    service, rows = site
    with FakeLlm(keep_everything) as fake:
        provider = fake_provider(tmp_path, fake, reviewer=False)
        with pytest.raises(ReviewRejected) as error:
            ai_review.review_route(service, ROUTE, provider, save=True, log=lambda text: None)
        assert (error.value.status, error.value.reason) == (409, 'ai_not_tested')
        assert fake.requests == [] and not (tmp_path / 'ai_reviews').exists()
        llm_providers.record_test(tmp_path, 'fake', {'ok': False, 'message': 'It cannot see the frames.'})
        with pytest.raises(ReviewRejected):
            ai_review.review_route(service, ROUTE, llm_providers.provider(tmp_path, 'fake'), save=True,
                                   log=lambda text: None)
        # A trial saves nothing: any provider may try.
        summary = ai_review.review_route(service, ROUTE, provider, save=False, limit=1, log=lambda text: None)
    assert summary['suggested'] == 1 and service.store.reviews(ROUTE, RUN_ID) == []


def test_the_review_page_reads_the_last_trial(site, tmp_path):
    service, rows = site
    assert ai_review.latest_trial(service, ROUTE) is None
    with FakeLlm(keep_everything) as fake:
        provider = fake_provider(tmp_path, fake)
        first = ai_review.review_route(service, ROUTE, provider, save=False, limit=2, log=lambda text: None)
        # A later trial wins (the file names start with the time).
        older = tmp_path / first['path']
        older.rename(older.with_name('20200101T000000Z-fake.json'))
        last = ai_review.review_route(service, ROUTE, provider, save=False, limit=3, log=lambda text: None)
    trial = ai_review.latest_trial(service, ROUTE)
    assert trial['run_id'] == RUN_ID and trial['reviewer'] == 'ai/fake/fake-vl' and trial['finished_at']
    assert trial['provider'] == {'id': 'fake', 'name': 'fake', 'model': 'fake-vl'}
    assert [frame['frame_key'] for frame in trial['frames']] == ['0/0', '0/1', '0/2']
    first_frame = trial['frames'][0]
    assert first_frame['message'] == 'fake-vl: 1 sign (1 kept)' and first_frame['complete'] is True
    assert first_frame['detections'][0]['ai_action'] == 'keep' and first_frame['deleted'] == []
    assert trial['frames'][1]['no_sign'] is True and trial['frames'][1]['message'] == 'fake-vl: no sign'
    assert 'sk-fake-key-123456' not in json.dumps(trial)
    # A frame a person confirmed since drops out; a save run is not a trial.
    human(service, rows['0/1'], drop_proposals=True)
    assert [frame['frame_key'] for frame in ai_review.latest_trial(service, ROUTE)['frames']] == ['0/0', '0/2']
    trial_start = json.loads((tmp_path / last['path']).read_text())['started_at']
    saved_run = json.loads((tmp_path / last['path']).read_text())
    saved_run.update(save=True, started_at='2999-01-01T00:00:00+00:00')
    (tmp_path / 'ai_reviews' / ROUTE / '29990101T000000Z-fake.json').write_text(json.dumps(saved_run))
    (tmp_path / 'ai_reviews' / ROUTE / '29990102T000000Z-fake.json').write_text('not json')
    assert ai_review.latest_trial(service, ROUTE)['started_at'] == trial_start
    # A trial of an older report of the route is not shown.
    base = tmp_path / 'derived' / 'preannotations' / ROUTE
    new_run = '20261022T100000Z-0123abcd'
    (base / RUN_ID).rename(base / new_run)
    (base / 'latest.json').write_text(json.dumps({'run_id': new_run}))
    assert ai_review.latest_trial(service, ROUTE) is None


# ---------- measure ----------

def reviewed_root(service, rows):
    """0/0 and 0/4: a 30 km/h sign (human); 0/2: a give-way sign; 0/1 and 0/3: no sign; 0/5 an AI review."""
    human(service, rows['0/0'], value_kph=30)
    human(service, rows['0/4'], value_kph=30)
    body = page_body(rows['0/2'], drop_proposals=True)
    body['detections'] = [{'bbox_xyxy': [4, 4, 20, 20], 'family': 'other_sign', 'sign_type': 'give_way',
                           'value_kph': None, 'applicability': 'own_road', 'detection_id': 'manual-1'}]
    save_frame_review(service.store, service.root, route_id=ROUTE, run_id=RUN_ID, report_row=rows['0/2'], body=body,
                      client='test')
    human(service, rows['0/1'], drop_proposals=True)
    human(service, rows['0/3'], drop_proposals=True)
    ai = {'run_id': RUN_ID, 'base_revision': 0, 'reviewer': 'Claude', 'reviewer_kind': 'ai', 'confirmed': True,
          'detections': []}
    save_frame_review(service.store, service.root, route_id=ROUTE, run_id=RUN_ID, report_row=rows['0/5'], body=ai,
                      client='test', channel='mcp', require_versioning=True)


def test_the_sample_is_deterministic_and_balanced(site):
    service, rows = site
    reviewed_root(service, rows)
    assert ai_review.measure_sample(service, 60) == [(ROUTE, '0/0'), (ROUTE, '0/1'), (ROUTE, '0/2'), (ROUTE, '0/3'),
                                                     (ROUTE, '0/4')]
    # 3 frames: 2 with speed signs (ceil(3/2)), then 1 of the other signs, none without.
    assert ai_review.measure_sample(service, 3) == [(ROUTE, '0/0'), (ROUTE, '0/2'), (ROUTE, '0/4')]
    assert ai_review.measure_sample(service, 1) == [(ROUTE, '0/0')]
    assert ai_review.measure_sample(service, 4) == ai_review.measure_sample(service, 4)


def test_measure_report(site, tmp_path):
    service, rows = site
    reviewed_root(service, rows)

    def answer(body):
        text = frame_text(body)
        if ' 0/0,' in text:
            return answer_json(boxes=[{'n': 1, 'verdict': 'keep', 'value_kph': 30, 'applicability': 'unresolved'}])
        if ' 0/4,' in text:
            return answer_json(boxes=[{'n': 1, 'verdict': 'correct', 'value_kph': 50}])
        if ' 0/2,' in text:
            return answer_json(boxes=[{'n': 1, 'verdict': 'delete'}],
                               added=[{'box': [125, 167, 625, 833], 'family': 'other_sign', 'sign_type': 'stop'}])
        if ' 0/1,' in text:
            return answer_json(added=[{'box': [0, 0, 300, 300], 'family': 'maximum_speed', 'value_kph': 80}])
        return 'broken'
    lines = []
    with FakeLlm(answer) as fake:
        provider = fake_provider(tmp_path, fake)
        report = ai_review.measure_reviewer(service, provider, limit=60, log=lines.append)
        again = ai_review.measure_reviewer(service, provider, limit=60, log=lambda text: None)
    assert report['sample_sha256'] == again['sample_sha256']
    assert report['task'] == 'ai_reviewer' and report['provider_id'] == 'fake' and report['model'] == 'fake-vl'
    assert report['reviewer'] == 'ai/fake/fake-vl' and report['limit'] == 60 and report['frames'] == 5
    assert (report['frames_measured'], report['frames_failed'], report['frames_incomplete']) == (4, 1, 0)
    assert report['failures'] == [{'route_id': ROUTE, 'frame_key': '0/3', 'reason': 'ai_bad_answer',
                                   'message': 'The AI answer has no JSON object.'}]
    assert report['no_sign'] == {'truth': 1, 'agreed': 0, 'agreed_share': 0.0, 'said_no_sign_wrongly': 0}
    speed = report['speed_signs']
    assert (speed['truth'], speed['found'], speed['read_right'], speed['kind_right'], speed['road_right']) == \
        (2, 2, 1, 2, 2)
    assert speed['read_right_share'] == 0.5 and speed['found_share'] == 1.0
    other = report['other_signs']
    assert (other['truth'], other['found'], other['kind_right']) == (1, 1, 0)
    assert report['every_sign'] == {'truth': 3, 'found': 3, 'found_share': 1.0}
    assert report['ai_boxes_per_frame'] == 1.0 and report['false_per_frame'] == 0.25
    assert report['iou_threshold'] == 0.5 and report['seconds_per_frame'] is not None
    expected = {'schema_version', 'task', 'provider_id', 'provider_name', 'model', 'reviewer', 'limit', 'frames',
                'frames_measured', 'frames_failed', 'frames_incomplete', 'failures', 'sample_sha256', 'iou_threshold',
                'seconds', 'seconds_per_frame', 'measured_at', 'no_sign', 'speed_signs', 'other_signs', 'every_sign',
                'ai_boxes_per_frame', 'false_per_frame'}
    assert set(report) == expected
    folder = ai_review.reports_dir(tmp_path)
    assert (folder / 'fake--fake-vl.json').is_file()
    assert len([path for path in folder.glob('*.json') if path.name[0].isdigit()]) >= 1
    assert ai_review.latest_reports(tmp_path) == [again]
    assert len(service.store.reviews(ROUTE, RUN_ID, confirmed_only=True)) == 6
    assert lines[-1].startswith('AI reviewer measured: 4 of 5 frames, 1 failed.')


def test_measure_without_human_reviews(site, tmp_path):
    service, rows = site
    with FakeLlm() as fake:
        with pytest.raises(ValueError, match='no frame reviewed by a person'):
            ai_review.measure_reviewer(service, fake_provider(tmp_path, fake), log=lambda text: None)


# ---------- Test in Settings ----------

def test_a_provider_that_reads_the_sign_sees_images(site, tmp_path):
    service, rows = site
    human(service, rows['0/2'], value_kph=80)
    with FakeLlm(answer_json(added=[{'box': [0, 0, 500, 500], 'family': 'maximum_speed', 'value_kph': 80}])) as fake:
        provider = fake_provider(tmp_path, fake)
        result = ai_review.test_provider(service, provider['id'])
    assert result['ok'] is True and result['answered_json'] is True and result['sees_images'] is True
    assert result['frame'] == {'route_id': ROUTE, 'frame_key': '0/2'}
    assert re.fullmatch(r'Sees images · \d+\.\d s', result['message'])
    # No proposal is sent: the value cannot be echoed back.
    assert 'numbered:\nnone\n' in frame_text(fake.requests[0])
    stored = llm_providers.public_settings(tmp_path)['providers'][0]['last_test']
    assert stored == result


def test_a_provider_that_misses_the_sign(site, tmp_path):
    service, rows = site
    human(service, rows['0/2'], value_kph=80)
    with FakeLlm(NO_SIGN_ANSWER) as fake:
        result = ai_review.test_provider(service, fake_provider(tmp_path, fake)['id'])
    assert result['ok'] is False and result['answered_json'] is True and result['sees_images'] is False
    assert result['message'].startswith('Answered in JSON, but missed the 80 sign of the test frame · ')


def test_a_text_only_model_is_never_asked(site, tmp_path):
    service, rows = site
    human(service, rows['0/2'], value_kph=80)
    with FakeLlm(models=('fake-vl',), text_only=True) as fake:
        result = ai_review.test_provider(service, fake_provider(tmp_path, fake)['id'])
    assert result['ok'] is False and result['sees_images'] is False
    assert result['message'] == 'The server says fake-vl takes text only: it cannot see the frames.'
    assert fake.requests == []


def test_without_a_reviewed_sign_the_answer_proves_only_json(site, tmp_path):
    service, rows = site
    with FakeLlm() as fake:
        result = ai_review.test_provider(service, fake_provider(tmp_path, fake)['id'])
    assert result['ok'] is True and result['sees_images'] is None
    assert result['frame'] == {'route_id': ROUTE, 'frame_key': '0/0'}
    assert result['message'].endswith('(no reviewed speed sign to prove it sees images)')


def test_an_unreachable_provider_fails_the_test(site, tmp_path):
    service, rows = site
    with FakeLlm(status=500, error_message='down') as fake:
        result = ai_review.test_provider(service, fake_provider(tmp_path, fake)['id'])
    assert result['ok'] is False and result['answered_json'] is False and result['message'] == 'HTTP 500: down'
    with FakeLlm() as fake:
        provider = fake_provider(tmp_path, fake)
    result = ai_review.test_provider(service, provider['id'])
    assert result['ok'] is False and result['message'].startswith('Cannot reach the AI server')


# ---------- scope lights: the frames where the car comes to a traffic light of the map ----------

def write_lights(root, positions, signals=((47.0010, 8.0000),)):
    gps = root / 'derived' / 'gps' / ROUTE
    gps.mkdir(parents=True, exist_ok=True)
    gps.joinpath('positions.jsonl').write_text(''.join(
        json.dumps({'frame_key': key, 'latitude': lat, 'longitude': lon, 'bearing_deg': bearing}) + '\n'
        for key, lat, lon, bearing in positions))
    osm = root / 'derived' / 'osm'
    osm.mkdir(parents=True, exist_ok=True)
    osm.joinpath('traffic_signals.jsonl').write_text(''.join(
        json.dumps({'id': n, 'lat': lat, 'lon': lon, 'kind': 'junction'}) + '\n' for n, (lat, lon) in enumerate(signals)))


def test_traffic_light_frames_keeps_lights_ahead_between_15_and_80_m(tmp_path):
    # The light is 0.0010° north, about 111 m: 0/0 is too far, 0/1 sees it 56 m ahead, 0/2 drives away from it,
    # 0/3 is 7 m from it (already under it), 0/4 heads east past it, 0/5 has no bearing.
    write_lights(tmp_path, [('0/0', 47.0000, 8.0, 0.0), ('0/1', 47.0005, 8.0, 10.0), ('0/2', 47.0005, 8.0, 180.0),
                            ('0/3', 47.00094, 8.0, 0.0), ('0/4', 47.0005, 8.0, 90.0), ('0/5', 47.0005, 8.0, None)])
    assert ai_review.traffic_light_frames(tmp_path, ROUTE) == {'0/1': 56}


def test_traffic_light_frames_needs_gps_and_the_map(tmp_path):
    with pytest.raises(ReviewRejected) as caught:
        ai_review.traffic_light_frames(tmp_path, ROUTE)
    assert caught.value.reason == 'no_gps'
    write_lights(tmp_path, [('0/1', 47.0005, 8.0, 0.0)])
    ai_review.signals_path(tmp_path).unlink()
    with pytest.raises(ReviewRejected) as caught:
        ai_review.traffic_light_frames(tmp_path, ROUTE)
    assert caught.value.reason == 'no_traffic_lights'


def test_a_lights_run_reviews_only_the_light_frames_whole_with_the_hint(site, tmp_path):
    service, rows = site
    write_lights(tmp_path, [(key, 47.0005, 8.0, 0.0 if key in ('0/1', '0/4') else 180.0) for key in rows])
    texts = []

    def answer(body):
        texts.append(frame_text(body))
        return keep_everything(body)
    with FakeLlm(answer) as fake:
        provider = fake_provider(tmp_path, fake)
        summary = ai_review.review_route(service, ROUTE, provider, save=False, scope='lights', log=lambda line: None)
        assert summary['scope'] == 'lights' and summary['frames'] == 2
        written = json.loads((tmp_path / summary['path']).read_text())
        assert [frame['frame_key'] for frame in written['frames']] == ['0/1', '0/4']
        assert all('traffic light 56 m ahead' in text for text in texts) and len(texts) == 2
        assert ai_review.review_route(service, ROUTE, provider, save=False, log=lambda line: None)['scope'] == 'signs'
        with pytest.raises(ReviewRejected):
            ai_review.review_route(service, ROUTE, provider, save=False, scope='roads')


def test_frames_without_gps_are_never_reviewed(site, tmp_path):
    # An underground garage: 0/0 and 0/1 have no position; the limit counts only the frames the AI reviews.
    service, rows = site
    write_lights(tmp_path, [(key, None if key in ('0/0', '0/1') else 47.0005, 8.0, 0.0) for key in rows])
    assert ai_review.frames_without_gps(tmp_path, ROUTE) == {'0/0', '0/1'}
    lines = []
    with FakeLlm(keep_everything) as fake:
        provider = fake_provider(tmp_path, fake)
        summary = ai_review.review_route(service, ROUTE, provider, save=False, limit=2, log=lines.append)
    written = json.loads((tmp_path / summary['path']).read_text())
    assert [frame['frame_key'] for frame in written['frames']] == ['0/2', '0/3']
    assert lines[0] == '2 frames without GPS left out (a garage or a tunnel)'


def test_a_server_that_drops_is_waited_for_then_the_run_stops(site, tmp_path):
    service, rows = site
    lines, waits = [], []
    with FakeLlm(keep_everything, status=502) as fake:
        provider = fake_provider(tmp_path, fake)

        def back_after_two(seconds):
            waits.append(seconds)
            if len(waits) == 2:
                fake.status = 200
        summary = ai_review.review_route(service, ROUTE, provider, save=False, log=lines.append, sleep=back_after_two)
        # Down twice on the first frame, then back: every frame suggested, none failed.
        assert waits == [ai_review.DOWN_WAIT] * 2 and summary['failed'] == 0 and summary['stopped'] is None
        assert summary['suggested'] == summary['frames'] == 6
        assert 'the AI server is down' in lines[0]
        fake.status = 502
        waits.clear()
        summary = ai_review.review_route(service, ROUTE, provider, save=False, log=lines.append,
                                         sleep=waits.append)
    # Never back: one frame waited for DOWN_RETRIES times, then the run stops instead of failing the rest.
    assert len(waits) == ai_review.DOWN_RETRIES and summary['stopped'] == 'ai_down' and summary['frames'] == 1
    assert lines[-1].startswith('AI review stopped (the AI server is down)')


def test_an_answer_cut_at_max_tokens_is_asked_again_with_more_room_not_taken_for_a_server_down():
    from nnslr_tools.llm_providers import LlmError
    cut = LlmError(502, 'ai_http_error', 'HTTP 502: structured output was incomplete (finish_reason=length); increase '
                                         'the output budget', http_code=502)
    assert llm_providers.incomplete(cut) and not ai_review._server_down(cut)
    broken = LlmError(502, 'ai_http_error', 'HTTP 502: model did not return valid JSON: Expecting value: line 1 column 8575 '
                                            '(char 8574)', http_code=502)
    assert llm_providers.incomplete(broken) and not ai_review._server_down(broken)
    down = LlmError(502, 'ai_http_error', 'HTTP 502: the AI server strata is down or restarting, try later', http_code=502)
    assert ai_review._server_down(down) and not llm_providers.incomplete(down)
