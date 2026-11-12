"""UserNet's network with random weights (nnslr_tools.usernet): its shapes, SignNet unchanged by ``heads``, three
kinds through encode and decode with the window's row offset, the loss weight of a frame not ticked, the checkpoint
and its start, the ONNX export with its parity check, and no training without a card. Nothing here trains."""
from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

from nnslr_tools.signnet import build_signnet, loss  # noqa: E402
from nnslr_tools.usernet import (INPUT_WIDTH, KINDS, MAX_USERS, STRIDE, WINDOW_HEIGHT, build_usernet,  # noqa: E402
                                 checkpoint_of, decode, decode_output, encode, grid, load_usernet, train_usernet,
                                 usernet_start, window_boxes)

FRAME_WIDTH = 1344
USERS = [{'kind': 'person', 'box': [400.0, 300.0, 440.0, 380.0]},
         {'kind': 'two_wheeler', 'box': [800.0, 350.0, 860.0, 450.0]},
         {'kind': 'vehicle', 'box': [100.0, 400.0, 300.0, 560.0]}]


def test_usernet_has_a_heat_channel_a_kind_and_the_shared_shape():
    model = build_usernet().eval()
    with torch.no_grad():
        heat, shape = model(torch.rand(2, 3, WINDOW_HEIGHT, INPUT_WIDTH))
    width, height = grid()
    assert (width, height) == (84, 32)
    assert heat.shape == (2, len(KINDS), height, width) and shape.shape == (2, 4, height, width)
    assert float(heat.min()) > 0 and float(heat.max()) < 1
    assert model.out.out_channels == 7 and model.fuse[0].kernel_size == (1, 1)


def test_signnet_is_the_same_network_with_one_head():
    assert build_signnet().out.out_channels == 5
    torch.manual_seed(7)
    before = build_signnet((8, 8, 8, 8), 8, 1).state_dict()
    torch.manual_seed(7)
    now = build_signnet((8, 8, 8, 8), 8, 1, heads=1).state_dict()
    assert list(before) == list(now) and all(torch.equal(before[key], now[key]) for key in before)
    for heads in (0, 1.5):
        with pytest.raises(ValueError):
            build_signnet(heads=heads)


def test_three_kinds_come_back_from_the_heat_map_in_the_frames_pixels():
    targets, ignored = window_boxes(USERS, [], FRAME_WIDTH)
    assert len(targets) == 3 and ignored == []
    heat, shape, mask, weight = encode(targets, ignored)
    width, height = grid()
    assert heat.shape == (3, height, width) and shape.shape == (4, height, width) and mask.sum() == 3
    assert [float(heat[kind].max()) for kind in range(3)] == [1.0, 1.0, 1.0]
    assert float(weight.min()) == 1.0
    found = decode(heat[None], shape[None], FRAME_WIDTH)[0]
    assert sorted(user['kind'] for user in found) == sorted(KINDS)
    for user in found:
        expected = next(item['box'] for item in USERS if item['kind'] == user['kind'])
        assert user['score'] == pytest.approx(1.0)
        assert user['box'] == pytest.approx(expected, abs=1e-3)  # the row offset added back, the scale undone
    # The one output of the export gives the same; a batch can hold frames of other widths.
    assert decode_output(torch.cat([heat, shape])[None], FRAME_WIDTH)[0] == found
    twice = decode(torch.stack([heat, heat]), torch.stack([shape, shape]), [FRAME_WIDTH, FRAME_WIDTH * 2])
    assert twice[1][0]['box'] == pytest.approx([value * 2 for value in twice[0][0]['box']], abs=1e-3)


def test_at_most_max_users_of_the_score_or_more():
    width, height = grid()
    heat = torch.zeros(1, 3, height, width)
    for index in range(40):
        heat[0, index % 3, 2 + 3 * (index // 14), 2 + 5 * (index % 14)] = 0.6 + index / 1000
    heat[0, 0, 30, 80] = 0.4  # below MIN_USER_SCORE
    found = decode(heat, torch.zeros(1, 4, height, width), FRAME_WIDTH)[0]
    assert len(found) == MAX_USERS and all(user['score'] >= 0.5 for user in found)
    assert [user['score'] for user in found] == sorted((user['score'] for user in found), reverse=True)
    assert len(decode(heat, torch.zeros(1, 4, height, width), FRAME_WIDTH, limit=100, min_score=0.3)[0]) == 41


def test_a_frame_not_ticked_is_judged_on_its_boxed_users_only():
    targets, _ = window_boxes(USERS[:1], [], FRAME_WIDTH)
    heat, shape, mask, weight = encode(targets, complete=False)
    # Weight 1 on the person's cells and one around them, 0 everywhere else.
    x1, y1, x2, y2 = targets[0][1]
    assert float(weight[int((y1 + y2) / 2 / STRIDE), int((x1 + x2) / 2 / STRIDE)]) == 1.0
    assert float(weight[0, 0]) == 0.0 and float(weight[-1, -1]) == 0.0
    assert float(weight.sum()) == 5 * 8  # x 200..220, y 78..118 of the window: cells 24..28 and 8..15
    # A confident vehicle far away costs nothing there, and does on a ticked frame.
    pred = heat.clamp(1e-4, 1 - 1e-4)[None].clone()
    pred[0, 2, 30, 80] = 0.9
    zero = torch.zeros(1, 4, *weight.shape)
    untouched, _ = loss(heat.clamp(1e-4, 1 - 1e-4)[None], zero, heat[None], shape[None], mask[None], weight[None])
    not_ticked, _ = loss(pred, zero, heat[None], shape[None], mask[None], weight[None])
    _, _, _, full = encode(targets, complete=True)
    ticked, _ = loss(pred, zero, heat[None], shape[None], mask[None], full[None])
    assert float(not_ticked) == pytest.approx(float(untouched), abs=1e-4) and float(ticked) > float(untouched) + 1
    # An ignore box on a ticked frame: weight 0 on its cells.
    _, _, _, ignoring = encode([], [[0.0, 0.0, 40.0, 40.0]], complete=True)
    assert float(ignoring[0, 0]) == 0.0 and float(ignoring[-1, -1]) == 1.0


def plan(stage: str) -> dict:
    return {'stage': stage, 'uses_bdd': stage == 'bdd', 'dataset_sha256': 'd' * 64, 'split_sha256': 's' * 64,
            'validation_source': 'drives', 'bdd': {'split_sha256': 'b' * 64} if stage == 'bdd' else None}


def test_the_checkpoint_loads_back_and_carries_the_licence_along_its_start(tmp_path):
    model = build_usernet()
    torch.save(checkpoint_of(model, plan('bdd'), epoch=4), tmp_path / 'usernet-best.pt')
    loaded, checkpoint = load_usernet(tmp_path / 'usernet-best.pt')
    assert (checkpoint['heads'], checkpoint['kinds'], checkpoint['input'], checkpoint['window_top']) == (
        3, list(KINDS), [INPUT_WIDTH, WINDOW_HEIGHT], 72)
    assert checkpoint['uses_bdd'] and checkpoint['bdd_split_sha256'] == 'b' * 64
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    weights, start = usernet_start(tmp_path / 'usernet-best.pt')
    assert (start['run'], start['epoch'], start['stage'], start['uses_bdd']) == (tmp_path.name, 4, 'bdd', True)
    # The drives stage from it keeps BDD's licence.
    assert checkpoint_of(model, plan('drives'), epoch=1, start=start)['uses_bdd']
    assert not checkpoint_of(model, plan('drives'), epoch=1)['uses_bdd']
    # A SignNet, or a UserNet of other settings, is no start.
    torch.save({'task': 'signnet', 'state_dict': build_signnet().state_dict()}, tmp_path / 'signnet-best.pt')
    with pytest.raises(ValueError, match='not a UserNet'):
        usernet_start(tmp_path / 'signnet-best.pt')
    torch.save({**checkpoint_of(model, plan('bdd'), epoch=4), 'stride': 4}, tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='another network'):
        usernet_start(tmp_path / 'other.pt')


def test_the_onnx_export_matches_pytorch(tmp_path, monkeypatch):
    pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    import json

    import numpy as np

    from nnslr_tools.usernet import export_usernet_onnx
    monkeypatch.setenv('NNSLR_CHECK_DEVICE', 'cpu')
    model = build_usernet().eval()
    torch.save(checkpoint_of(model, plan('bdd'), epoch=2), tmp_path / 'usernet-best.pt')
    report = export_usernet_onnx(tmp_path / 'usernet-best.pt', tmp_path / 'usernet.onnx')
    assert report['verification']['passed'] and report['task'] == 'usernet'
    contract = json.loads((tmp_path / 'usernet.onnx.contract.json').read_text())
    assert contract['output']['shape'] == [1, 7, 32, 84] and contract['uses_bdd']
    assert contract['output']['channels'][:3] == ['heat_person', 'heat_two_wheeler', 'heat_vehicle']
    session = ort.InferenceSession(str(tmp_path / 'usernet.onnx'), providers=['CPUExecutionProvider'])
    images = np.random.rand(1, 3, WINDOW_HEIGHT, INPUT_WIDTH).astype(np.float32)
    out = session.run(['out'], {'images': images})[0]
    with torch.no_grad():
        heat, shape = model(torch.from_numpy(images))
    assert np.allclose(out, torch.cat([heat, shape], 1).numpy(), atol=1e-4)


def test_without_a_card_it_stops_before_writing_anything(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    frames = [{'split': 'train', 'users': [], 'ignore': [], 'complete': True, 'width': FRAME_WIDTH}] * 4
    with pytest.raises(ValueError, match='no_cuda_card'):
        train_usernet({'trainable': True, 'frames': frames}, tmp_path, tmp_path / 'run', batch=2, log=lambda line: None)
    assert not (tmp_path / 'run').exists()
    with pytest.raises(ValueError, match='usernet_not_trainable'):
        train_usernet({'trainable': False, 'not_trainable': 'no frames'}, tmp_path, tmp_path / 'run')
    with pytest.raises(ValueError, match='fewer than one batch'):
        train_usernet({'trainable': True, 'frames': frames}, tmp_path, tmp_path / 'run', batch=32)
