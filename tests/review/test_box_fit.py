"""Fit box (key C) of the review page: the local box fit (nnslr_tools.box_fit), its route ``…/frame/<key>/fit`` and its
training command (``nnslr train-box-fit``).

The fit runs on the CPU with the weights shipped in the package; the frames are synthetic, except one test that uses
the real frames and boxes of this clone when they are there (skipped elsewhere).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

cv2 = pytest.importorskip('cv2')
np = pytest.importorskip('numpy')

from nnslr_tools import box_fit, cli  # noqa: E402
from nnslr_tools.box_fit import train  # noqa: E402
from review_helpers import (ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report, sign_in)  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REAL_DATA = REPO / 'data'
# A red ring on white on a grey 200 x 150 picture: the sign, and the box a person would draw around it.
RING_TRUTH = [85, 60, 116, 91]
FIT = f'/api/route/{ROUTE}/frame/0/0/fit'


def ring_image():
    img = np.full((150, 200, 3), 128, np.uint8)
    cv2.circle(img, (100, 75), 15, (255, 255, 255), -1)
    cv2.circle(img, (100, 75), 13, (0, 0, 220), 4)
    return img


# ---------- the fit ----------

@pytest.mark.parametrize('loose', [[75, 50, 125, 100], [70, 52, 128, 96], [80, 55, 122, 97]])
def test_a_loose_box_around_a_red_ring_gets_tighter(loose):
    # A flat grey picture is far from a road, so only the colour ranker is held to a close fit; the answer (its mean
    # with the edge ranker, pulled back toward the rough box) must still be closer to the ring than the rough box.
    from nnslr_tools.box_fit import colour
    img = ring_image()
    by_colour = colour.fit(img, [float(v) for v in loose], box_fit.models()[0])
    assert by_colour is not None and box_fit._iou(by_colour, RING_TRUTH) > 0.7
    fitted = box_fit.fit_image(img, loose)
    assert fitted is not None
    assert box_fit._iou(fitted, RING_TRUTH) > box_fit._iou(loose, RING_TRUTH) + 0.05


def test_the_same_box_always_gets_the_same_fit_and_one_decode_fits_many(tmp_path):
    img = ring_image()
    path = tmp_path / 'ring.png'
    cv2.imwrite(str(path), img)
    boxes = [[75, 50, 125, 100], RING_TRUTH, [1, 1, 2, 2]]
    once = [box_fit.fit_image(img, box) for box in boxes]
    assert box_fit.fit_many(path, boxes) == once
    assert box_fit.fit_many(path, boxes) == once
    # A box drawn right is kept (None) or barely moved; a box too small to fit is kept as it is.
    assert once[1] is None or box_fit._iou(once[1], RING_TRUTH) > 0.9
    assert once[2] is None


def test_boxes_outside_or_reversed_are_clipped_or_kept():
    img = ring_image()
    assert box_fit.fit_image(img, [300, 300, 400, 400]) is None
    reversed_box = box_fit.fit_image(img, [125, 100, 75, 50])
    assert reversed_box == box_fit.fit_image(img, [75, 50, 125, 100])
    assert box_fit.clip_box([-5, -5, 250, 10], 200, 150) == [0.0, 0.0, 200.0, 10.0]
    assert box_fit.clip_box([10, 10, 12, 40], 200, 150) is None


def test_fits_from_several_threads_at_once_match_the_fits_one_at_a_time():
    # The review server runs up to MAX_LOCAL_FITS fits at once: no OpenCV object may be shared between threads (a
    # shared MSER crashed the process). A frame-sized picture with six rings, fitted by four threads together.
    import threading
    img = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 255, (874, 1164, 3), np.uint8), (0, 0), 3)
    for i in range(6):
        cv2.circle(img, (150 + 150 * i, 400), 30, (255, 255, 255), -1)
        cv2.circle(img, (150 + 150 * i, 400), 26, (0, 0, 220), 7)
    boxes = [[110 + 150 * i, 360, 195 + 150 * i, 445] for i in range(6)]
    alone = [box_fit.fit_image(img, box) for box in boxes]
    results, errors = [], []

    def work():
        try:
            for _ in range(5):
                results.append([box_fit.fit_image(img, box) for box in boxes])
        except Exception as error:  # noqa: BLE001 - reported by the assert below
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(results) == 20
    assert all(result == alone for result in results)


def test_unreadable_image_is_an_os_error(tmp_path):
    bad = tmp_path / 'bad.png'
    bad.write_bytes(b'not a picture')
    with pytest.raises(OSError):
        box_fit.fit_many(bad, [[0, 0, 10, 10]])


def test_the_refiner_moves_only_small_boxes_and_the_rankers_finish_its_box(monkeypatch):
    from nnslr_tools.box_fit import refiner
    calls = []

    def refine(img, box):
        calls.append(box)
        return [box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2]

    monkeypatch.setattr(refiner, 'available', lambda: True)
    monkeypatch.setattr(refiner, 'refine', refine)
    monkeypatch.setattr(refiner, 'judge', lambda img, box: None)
    img = ring_image()
    big = [10, 10, 10 + refiner.GATE + 10, 10 + refiner.GATE + 10]
    assert box_fit.fit_image(img, big) == box_fit.fit_image(img, big, use_refiner=False) and not calls
    small = [90, 65, 112, 87]
    fitted = box_fit.fit_image(img, small)
    assert calls == [small]
    assert fitted == (box_fit.fit_image(img, [92, 67, 110, 85], use_refiner=False) or [92, 67, 110, 85])


def test_the_shipped_refiner_runs_and_keeps_a_small_box_inside_the_picture():
    from nnslr_tools.box_fit import refiner
    if not refiner.available():
        pytest.skip('no refiner.onnx in this checkout')
    img = ring_image()
    box = refiner.refine(img, [88, 63, 113, 88])
    assert box is not None and 0 <= box[0] < box[2] <= 200 and 0 <= box[1] < box[3] <= 150


def test_on_the_boxes_people_drew_the_fit_is_closer_than_the_loose_box():
    objects = REAL_DATA / 'annotations' / 'objects.jsonl'
    if not objects.exists() or not (REAL_DATA / 'derived' / 'frames').is_dir():
        pytest.skip('no reviewed frames in this clone')
    items = train.human_boxes(objects, REAL_DATA)
    if len(items) < 30:
        pytest.skip('too few boxes drawn by people in this clone')
    picked = random.Random(0).sample(items, 30)
    cases = [case for case in train.make_cases(picked) if case['kind'] == 'loose']
    rough = [box_fit._iou(case['rough'], case['truth']) for case in cases]
    fitted = []
    for case in cases:
        box = box_fit.fit_many(case['image_path'], [case['rough']])[0]
        fitted.append(box_fit._iou(box or case['rough'], case['truth']))
    assert np.mean(fitted) > np.mean(rough) + 0.1


# ---------- the training ----------

def test_rough_boxes_are_seeded_by_the_annotation_and_inside_the_image():
    items = [{'annotation_id': 'a', 'image_path': 'x.png', 'route': 'r1', 'family': 'maximum_speed', 'size': 'mid',
              'width': 100, 'height': 80, 'truth': [0.0, 10.0, 30.0, 40.0]}]
    first, again = train.make_cases(items), train.make_cases(items)
    assert first == again and [case['kind'] for case in first] == list(train.KINDS)
    for case in first:
        x1, y1, x2, y2 = case['rough']
        assert 0 <= x1 < x2 <= 100 and 0 <= y1 < y2 <= 80


def test_folds_keep_whole_routes_and_balance_the_boxes():
    items = [{'route': route} for route, n in (('a', 50), ('b', 30), ('c', 20), ('d', 20), ('e', 10))
             for _ in range(n)]
    folds = train.route_folds(items, 2)
    assert set(folds) == {'a', 'b', 'c', 'd', 'e'}
    sizes = [sum(folds[item['route']] == f for item in items) for f in range(2)]
    assert sorted(sizes) == [60, 70]


def drawn_row(n: int, image: str, box, **changes) -> dict:
    row = {'annotation_id': f'{n:064x}', 'bbox_xyxy': box, 'image_path': image, 'image_sha256': f'{n:064x}',
           'route_id': image.split('/')[2], 'width': 200, 'height': 150, 'sign_family': 'maximum_speed',
           'review_state': 'accepted', 'proposal_source': 'human', 'reviewer': 'cristianku',
           'provenance': {'label_basis': 'human_review', 'original_proposal': None}}
    row.update(changes)
    return row


def test_train_box_fit_dry_run_counts_only_the_boxes_people_drew(tmp_path, capsys):
    root = tmp_path / 'data'
    for route in ('0000000a--aaaaaaaaaa', '0000000b--bbbbbbbbbb'):
        image = root / 'derived/frames' / route / '0/frame_00000000.png'
        image.parent.mkdir(parents=True)
        cv2.imwrite(str(image), ring_image())
    a = 'derived/frames/0000000a--aaaaaaaaaa/0/frame_00000000.png'
    b = 'derived/frames/0000000b--bbbbbbbbbb/0/frame_00000000.png'
    rows = [
        drawn_row(1, a, RING_TRUTH),
        drawn_row(2, b, [10, 10, 40, 40], sign_family='other_sign'),
        # A model box only accepted, an AI's box, not a sign, a tiny box, a duplicate, a missing image: left out.
        drawn_row(3, a, [50, 50, 80, 80], proposal_source='preannotation:x',
                  provenance={'label_basis': 'human_review', 'original_proposal': {'bbox_xyxy': [50, 50, 80, 80]}}),
        drawn_row(4, a, [50, 50, 80, 80], reviewer='ai/fake/model', provenance={'label_basis': 'ai_review'}),
        drawn_row(5, a, [50, 50, 80, 80], sign_family='not_a_sign'),
        drawn_row(6, a, [50, 50, 52, 80]),
        drawn_row(7, a, RING_TRUTH, image_sha256=f'{1:064x}'),
        drawn_row(8, 'derived/frames/0000000a--aaaaaaaaaa/0/frame_00000009.png', [10, 10, 40, 40]),
        # A model box a person moved: drawn by them.
        drawn_row(9, b, [60, 60, 90, 95], proposal_source='preannotation:x',
                  provenance={'label_basis': 'human_review', 'original_proposal': {'bbox_xyxy': [58, 61, 92, 99]}}),
    ]
    (root / 'annotations').mkdir(parents=True)
    (root / 'annotations' / 'objects.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = tmp_path / 'models'
    code = cli.main(['train-box-fit', '--dry-run', '--data-root', str(root), '--out', str(out), '--folds', '2'])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary['dry_run'] is True
    assert summary['boxes'] == 3 and summary['cases'] == 9 and summary['routes'] == 2
    assert summary['by_family'] == {'maximum_speed': 2, 'other_sign': 1}
    assert summary['folds'] == 2 and sorted(summary['boxes_by_fold'].values()) == [1, 2]
    assert summary['min_boxes'] == train.MIN_BOXES
    assert summary['out'] == str(out) and not out.exists()


def test_a_training_with_too_few_boxes_is_refused(tmp_path):
    root = tmp_path / 'data'
    (root / 'annotations').mkdir(parents=True)
    (root / 'annotations' / 'objects.jsonl').write_text('')
    with pytest.raises(ValueError, match='too few'):
        train.run(root, tmp_path / 'models', folds=0, workers=1)
    assert not (tmp_path / 'models').exists()


def test_saved_models_load_and_fit_as_the_shipped_ones(tmp_path):
    colour_model, edges_model = box_fit.models()
    from nnslr_tools.box_fit import edges, gbdt
    edges_arrays = [(mu, sd, ps) for mu, sd, ps in edges_model]
    paths = train.save(tmp_path, colour_model, edges_arrays)
    assert sorted(paths) == ['colour.npz', 'edges.npz']
    again = (gbdt.load(tmp_path / 'colour.npz'), edges.load(tmp_path / 'edges.npz'))
    loose = [75, 50, 125, 100]
    assert box_fit.fit_image(ring_image(), loose, colour_model=again[0], edges_model=again[1]) == \
        box_fit.fit_image(ring_image(), loose)


# ---------- the route ----------

@pytest.fixture
def local(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    yield server, tmp_path
    server.close()


@pytest.fixture
def lan(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    yield server, tmp_path
    server.close()


def test_fit_answers_one_box_or_null_for_each_box_sent_and_saves_nothing(local):
    server, root = local
    status, response, data = server.request('POST', FIT, body={'run_id': RUN_ID, 'boxes': [[1, 2, 15, 20],
                                                                                           [0, 0, 32, 24]]})
    assert status == 200, data
    answer = json_of(data)
    assert answer['ok'] is True and isinstance(answer['ms'], int) and len(answer['boxes']) == 2
    for box in answer['boxes']:
        assert box is None or (len(box) == 4 and 0 <= box[0] < box[2] <= 32 and 0 <= box[1] < box[3] <= 24)
    assert server.app.store.confirmed_count(ROUTE, RUN_ID) == 0


def test_fit_of_a_rebuilt_report_is_refused(local):
    server, root = local
    status, response, data = server.request('POST', FIT, body={'run_id': 'another-run', 'boxes': [[1, 2, 15, 20]]})
    assert status == 409 and json_of(data)['error'] == 'stale_report'


@pytest.mark.parametrize('body', [
    [1, 2],
    {'run_id': 5, 'boxes': [[1, 2, 15, 20]]},
    {'run_id': RUN_ID},
    {'run_id': RUN_ID, 'boxes': []},
    {'run_id': RUN_ID, 'boxes': [[1, 2, 15, 20]] * 65},
    {'run_id': RUN_ID, 'boxes': [[1, 2, 15]]},
    {'run_id': RUN_ID, 'boxes': [[1, 2, 15, '20']]},
    {'run_id': RUN_ID, 'boxes': [[1, 2, 15, True]]},
    {'run_id': RUN_ID, 'boxes': [[1, 2, 15, float('inf')]]},
    {'run_id': RUN_ID, 'boxes': [[15, 2, 1, 20]]},
    {'run_id': RUN_ID, 'boxes': [[1, 2, 40, 20]]},
    {'run_id': RUN_ID, 'boxes': [[-3, 2, 15, 20]]},
])
def test_fit_refuses_a_bad_body(local, body):
    server, root = local
    status, response, data = server.request('POST', FIT, body=json.dumps(body).encode())
    assert status == 400, data
    assert json_of(data)['error'] in ('invalid_json', 'invalid_boxes')


def test_fit_of_an_unknown_frame_or_without_json_is_refused(local):
    server, root = local
    status, response, data = server.request('POST', f'/api/route/{ROUTE}/frame/0/99/fit',
                                            body={'run_id': RUN_ID, 'boxes': [[1, 2, 15, 20]]})
    assert status == 404
    status, response, data = server.request('POST', FIT, body=b'{}', headers={'Content-Type': 'text/plain'})
    assert status == 415
    status, response, data = server.request('POST', FIT, body={'run_id': RUN_ID, 'boxes': [[1, 2, 15, 20]]},
                                            headers={'Origin': 'http://evil.example'})
    assert status == 403 and json_of(data)['error'] == 'cross_origin'


def test_fit_falls_back_to_the_preview_and_says_when_no_image_can_be_read(local):
    server, root = local
    # The original frame is gone; the preview of make_report is not a picture either.
    (root / 'derived/frames' / ROUTE / '0' / 'frame_00000000.png').unlink()
    status, response, data = server.request('POST', FIT, body={'run_id': RUN_ID, 'boxes': [[1, 2, 15, 20]]})
    assert status == 409 and json_of(data)['error'] == 'invalid_image'


def test_on_the_lan_fit_needs_a_session_not_the_token(lan):
    server, root = lan
    body = {'run_id': RUN_ID, 'boxes': [[1, 2, 15, 20]]}
    status, response, data = server.request('POST', FIT, body=body, headers={'X-Review-Token': TOKEN})
    assert status == 403 and json_of(data)['error'] == 'session_required'
    status, response, data = server.request('POST', FIT, body=body, headers=sign_in(server))
    assert status == 200 and len(json_of(data)['boxes']) == 1


def test_a_small_box_the_refiner_judges_right_is_left_to_the_rankers(monkeypatch):
    from nnslr_tools.box_fit import refiner
    calls = []
    monkeypatch.setattr(refiner, 'available', lambda: True)
    monkeypatch.setattr(refiner, 'refine', lambda img, box: calls.append(box))
    monkeypatch.setattr(refiner, 'judge', lambda img, box: refiner.KEEP + 0.05)
    img = ring_image()
    small = [90, 65, 112, 87]
    assert box_fit.fit_image(img, small) == box_fit.fit_image(img, small, use_refiner=False) and not calls
