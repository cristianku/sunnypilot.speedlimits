"""Sign review's square beside the zoom (GET /api/route/<drive>/frame/<key>/classifier-crop): the crops the classifier's
training cuts of the box being edited, from the frame at full definition when the drive has one, pixel for pixel as the
training's crop cache gives them. Made-up drives and people on 127.0.0.1."""
from __future__ import annotations

import hashlib
import io
import json
import random
import shutil

import numpy as np
import pytest
from PIL import Image

from nnslr_tools.crop_cache import EXACT, WINDOW, build_crop_cache
from nnslr_tools.media import FRAME_SIZE, FULL_SIZE, full_picture
from nnslr_tools.review import write_review
from nnslr_tools.review_access import PROXY_SECRET_HEADER
from nnslr_tools.training import TrainConfig, _sample_seed, crop_rectangle, jitter_box
from review_helpers import OTHER_ROUTE, ROUTE, RUN_ID, TOKEN, RunningServer, json_of, make_report

SIZE = TrainConfig().input_size
# A sign far away, a near one (larger than MAX_BOX_SIDE input sizes: its window is scaled down for the jitter), one on
# the frame's right border and one with a fraction of a pixel outside it.
BOXES = [[1190.5, 260.5, 1226.9, 296.6], [400.2, 150.7, 900.4, 700.1], [1320.3, 380.0, 1344.0, 420.6],
         [-0.4, 700.2, 30.8, 760.3]]
# The first epochs of record 0 of a run of seed 0 whose draw moves the box: epochs 0 and 3 keep it as drawn.
MOVED_EPOCHS = [1, 2, 4, 5]


@pytest.fixture(scope='module')
def pictures(tmp_path_factory):
    """A noisy picture (1344 x 760 PNG) and a noisy frame at full definition (2688 x 1520 JPEG), so that a crop one
    pixel off differs."""
    folder = tmp_path_factory.mktemp('pictures')
    for name, size, seed in (('picture.png', FRAME_SIZE, 1), ('full.jpg', FULL_SIZE, 2)):
        pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / name, quality=90)
    return folder


def a_drive(root, pictures, full):
    """The report's frame 0/0 made a real picture of the drive, with its frame at full definition beside it or not."""
    rows = make_report(root, count=2)
    picture = root / rows[0]['image_path']
    shutil.copyfile(pictures / 'picture.png', picture)
    if full:
        shutil.copyfile(pictures / 'full.jpg', full_picture(picture))
    rows[0].update(width=FRAME_SIZE[0], height=FRAME_SIZE[1],
                   frame_sha256=hashlib.sha256(picture.read_bytes()).hexdigest())
    directory = root / 'derived/preannotations' / ROUTE / RUN_ID
    (directory / 'preannotations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write_review(directory, rows, RUN_ID)
    return rows[0]


@pytest.fixture(params=['full', 'old'])
def drive(request, tmp_path, pictures):
    """(full definition or not, server, data root, the frame's row) of a drive on the site of this computer."""
    row = a_drive(tmp_path, pictures, request.param == 'full')
    server = RunningServer(tmp_path)
    yield request.param == 'full', server, tmp_path, row
    server.close()


def address(box, samples=None, key='0/0', route=ROUTE):
    query = 'box=' + ','.join(str(value) for value in box)
    if samples is not None:
        query += f'&samples={samples}'
    return f'/api/route/{route}/frame/{key}/classifier-crop?{query}'


def tiles(data):
    with Image.open(io.BytesIO(data)) as strip:
        assert strip.mode == 'RGB'
        pixels = np.asarray(strip)
    assert pixels.shape[0] == SIZE and pixels.shape[1] % SIZE == 0
    return [pixels[:, place * SIZE:(place + 1) * SIZE] for place in range(pixels.shape[1] // SIZE)]


def training_crops(root, row, box, samples):
    """The crops the training gives the classifier for this box, from its crop cache: the crop of validation (EXACT),
    then the box jittered in the first epochs of a run of seed 0 that move record 0 (1 epoch in 5 keeps it as drawn),
    cut from its window and resized to the input size (the dataset's transforms.Resize, which on a PIL image is
    Pillow's bilinear resize)."""
    records = [{'image_path': row['image_path'], 'image_sha256': None, 'bbox_xyxy': box}] * 2
    cache, _ = build_crop_cache(root, records, [EXACT, WINDOW], size=SIZE, scheme='v2', content={'test': 'sign review'})
    crops = [np.asarray(cache.crop(0))]
    for epoch in MOVED_EPOCHS[:samples]:
        moved = jitter_box(box, random.Random(_sample_seed('box', 0, epoch, 0)))
        crops.append(np.asarray(cache.crop(1, moved).resize((SIZE, SIZE), Image.BILINEAR)))
    return crops


def test_the_square_is_the_crop_the_training_cuts(drive):
    full, server, root, row = drive
    factor = 2 if full else 1
    picture = root / row['image_path']
    with Image.open(full_picture(picture) if full else picture) as source:
        image = source.convert('RGB')
    for box in BOXES:
        status, response, data = server.request('GET', address(box, 4))
        assert status == 200 and response.getheader('Content-Type') == 'image/png'
        got = tiles(data)
        assert len(got) == 5
        for place, (crop, expected) in enumerate(zip(got, training_crops(root, row, box, 4))):
            assert np.array_equal(crop, expected), (full, box, place)
        # The crop without jitter by hand: the box's pixels of the frame at full definition when there is one.
        left, top, right, bottom = crop_rectangle(box, *FRAME_SIZE)
        hand = image.crop((left * factor, top * factor, right * factor, bottom * factor))
        hand = hand.resize((SIZE, SIZE), Image.BILINEAR)
        assert np.array_equal(got[0], np.asarray(hand))
        assert response.getheader('X-Crop-Source') == ('2688x1520' if full else '1344x760')
        assert response.getheader('X-Crop-Pixels') == f'{(right - left) * factor}x{(bottom - top) * factor}'
        assert response.getheader('X-Crop-Size') == str(SIZE) and response.getheader('X-Crop-Samples') == '4'
        assert response.getheader('Cache-Control') == 'private, max-age=3600'
    # Without samples, only the crop without jitter; the same address, the same pixels.
    status, response, data = server.request('GET', address(BOXES[0]))
    assert status == 200 and len(tiles(data)) == 1 and response.getheader('X-Crop-Samples') == '0'
    assert np.array_equal(tiles(data)[0], tiles(server.request('GET', address(BOXES[0], 4))[2])[0])
    assert server.request('GET', address(BOXES[0]) + '&run=whatever')[2] == data


@pytest.mark.parametrize('query', [
    '', 'box=', 'box=1,2,3', 'box=1,2,3,4,5', 'box=a,b,c,d', 'box=nan,1,20,30', 'box=1,2,inf,40', 'box=1,2,1e999,40',
    'box=10,10,5,20', 'box=10,10,20,10', 'box=-5,10,20,30', 'box=10,10,1400,30', 'box=10,10,20,800',
    'box=1,2,30,40&box=1,2,30,40',
])
def test_a_bad_box_is_400(drive, query):
    _, server, _, _ = drive
    status, _, data = server.request('GET', f'/api/route/{ROUTE}/frame/0/0/classifier-crop?{query}')
    assert status == 400 and json_of(data)['error'] == 'invalid_box', query


@pytest.mark.parametrize('samples', ['9', '-1', 'x', '', '100', '2&samples=3'])
def test_bad_samples_are_400(drive, samples):
    _, server, _, _ = drive
    status, _, data = server.request('GET', address(BOXES[0], samples))
    assert status == 400 and json_of(data)['error'] == 'invalid_samples', samples


def test_an_unknown_frame_or_drive_is_404_and_a_changed_picture_409(tmp_path, pictures):
    row = a_drive(tmp_path, pictures, True)
    server = RunningServer(tmp_path)
    try:
        for path, error in ((address(BOXES[0], key='0/9'), 'unknown_frame'),
                            (address(BOXES[0], key='7/0'), 'unknown_frame'),
                            (address(BOXES[0], route=OTHER_ROUTE), 'unknown_route'),
                            (address(BOXES[0], route='not-a-drive'), 'unknown_route')):
            status, _, data = server.request('GET', path)
            assert status == 404 and json_of(data)['error'] == error, path
        # A frame of the video not saved: not in the report, the page keeps its own square.
        assert server.request('GET', address(BOXES[0], key='0/12345'))[0] == 404
        # Only GET.
        assert server.request('POST', address(BOXES[0]), body={})[0] in (403, 404, 405)
        # The picture changed after the report: the crop would not be the frame reviewed.
        (tmp_path / row['image_path']).write_bytes(b'not the picture')
        status, _, data = server.request('GET', address(BOXES[0]))
        assert status == 409 and json_of(data)['error'] == 'image_changed'
    finally:
        server.close()


def test_a_guest_sees_it_only_on_a_drive_whose_signs_they_have(tmp_path, pictures, monkeypatch):
    monkeypatch.setenv('NNSLR_PROXY_IPS', '127.0.0.1')
    monkeypatch.setenv('NNSLR_PROXY_SECRET', 'proxy-secret-0123456789')
    monkeypatch.setenv('NNSLR_PUBLIC_HOSTS', 'reviews.example')
    a_drive(tmp_path, pictures, False)
    make_report(tmp_path, route=OTHER_ROUTE)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    guest = {PROXY_SECRET_HEADER: 'proxy-secret-0123456789', 'X-authentik-username': 'ana',
             'X-authentik-groups': 'reviewers'}
    try:
        server.app.assignments.add_person('ana')
        server.app.assignments.set_work('ana', ROUTE, 'speed', True)
        assert server.request('GET', address(BOXES[0]), headers=guest)[0] == 403
        server.app.assignments.set_work('ana', ROUTE, 'signs', True)
        status, response, data = server.request('GET', address(BOXES[0], 2), headers=guest)
        assert status == 200 and len(tiles(data)) == 3
        assert server.request('GET', address([1, 1, 20, 20], route=OTHER_ROUTE), headers=guest)[0] == 403
        # Nobody else: no token, no session.
        assert server.request('GET', address(BOXES[0]))[0] == 401
    finally:
        server.close()
