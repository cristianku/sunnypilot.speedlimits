"""SignNet's targets and decoding (nnslr_tools.signnet): a box encoded into the heat map comes back out as itself."""
import math

import pytest

torch = pytest.importorskip('torch')

from nnslr_tools.signnet import (
    BAND_HEIGHT,
    BAND_WIDTH,
    CAR_BAND,
    CAR_MAX_BOXES,
    CAR_MIN_SCORE,
    CAR_STRIDE,
    DATA_KEYS,
    band_boxes,
    band_size,
    build_signnet,
    checkpoint_data,
    checkpoint_of,
    decode,
    encode,
    fits_the_car,
    frame_boxes_of,
    loss,
    model_of,
    train_signnet,
)


def test_band_boxes_keep_the_signs_of_the_band():
    # A frame twice as wide: scaled to the band; a sign centred below the band is left out.
    boxes = band_boxes([[200, 100, 240, 140], [400, 900, 440, 960]], width=BAND_WIDTH * 2)
    assert boxes == [[100.0, 50.0, 120.0, 70.0]]


@pytest.mark.parametrize('stride', (4, 8))
def test_encode_then_decode_gives_the_box_back(stride):
    box = [601.0, 133.0, 627.0, 161.0]
    heat, shape, mask, weight = encode([box], stride)
    assert heat.shape == (1, BAND_HEIGHT // stride, BAND_WIDTH // stride) and mask.sum() == 1
    assert weight.shape == mask.shape and float(weight.min()) == 1.0
    assert float(heat.max()) == 1.0
    found = decode(heat[None], shape[None], stride)[0]
    assert found[0]['score'] == pytest.approx(1.0)
    assert found[0]['box'] == pytest.approx(box, abs=1e-3)
    assert math.isclose(len([f for f in found if f['score'] > 0.5]), 1)


def test_network_shapes_and_loss_at_the_target():
    model = build_signnet(stride=8).eval()
    heat, shape = model(torch.rand(2, 3, BAND_HEIGHT, BAND_WIDTH))
    assert heat.shape == (2, 1, BAND_HEIGHT // 8, BAND_WIDTH // 8) and shape.shape == (2, 4, BAND_HEIGHT // 8, BAND_WIDTH // 8)
    target_heat, target_shape, mask, _ = encode([[100.0, 100.0, 130.0, 130.0]], 8)
    perfect, _ = loss(target_heat[None].clamp(1e-4, 1 - 1e-4), target_shape[None], target_heat[None], target_shape[None],
                      mask[None])
    random_guess, _ = loss(heat[:1], shape[:1], target_heat[None], target_shape[None], mask[None])
    assert float(perfect) < float(random_guess)


def test_half_band():
    # Half the resolution: half the band both ways; boxes scaled to it, and the targets the size of its maps.
    assert band_size(BAND_WIDTH // 2) == (BAND_WIDTH // 2, BAND_HEIGHT // 2)
    assert band_size(896)[1] % 16 == 0
    assert band_boxes([[200, 100, 240, 140]], width=BAND_WIDTH, band_width=BAND_WIDTH // 2) == [[100.0, 50.0, 120.0, 70.0]]
    heat, *_ = encode([[100.0, 50.0, 120.0, 70.0]], 4, BAND_WIDTH // 2)
    assert heat.shape == (1, BAND_HEIGHT // 8, BAND_WIDTH // 8)
    model = build_signnet(stride=4).eval()
    out, _ = model(torch.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2))
    assert out.shape == heat[None].shape


def test_ignore_boxes_are_neither_signs_nor_background():
    from nnslr_tools.signnet import _in_ignore
    # A switched-off display at 100..130: no centre, weight 0 on its cells and one cell around (stride 8: 11..17).
    heat, shape, mask, weight = encode([], 8, ignore=[[100.0, 100.0, 130.0, 130.0]])
    assert float(heat.max()) == 0.0 and mask.sum() == 0
    assert weight[14, 14] == 0 and weight[11, 11] == 0 and weight[17, 17] == 0
    assert weight[18, 18] == 1 and weight[10, 10] == 1 and weight[0, 0] == 1
    # A confident peak on the panel costs nothing with the weight, and does without it.
    pred = torch.full((1, 1) + tuple(weight.shape), 1e-4)
    pred[0, 0, 14, 14] = 0.9
    zero_shape = torch.zeros(1, 4, *weight.shape)
    judged, _ = loss(pred, zero_shape, heat[None], shape[None], mask[None], weight[None])
    unjudged, _ = loss(pred, zero_shape, heat[None], shape[None], mask[None])
    assert float(judged) == pytest.approx(0.0, abs=1e-3) and float(unjudged) > 1.0
    # A sign centred on an ignored cell still counts as a sign.
    heat, shape, mask, weight = encode([[110.0, 110.0, 120.0, 120.0]], 8, ignore=[[100.0, 100.0, 130.0, 130.0]])
    assert mask.sum() == 1 and float(heat.max()) == 1.0
    missed, _ = loss(torch.full((1, 1) + tuple(weight.shape), 1e-4), zero_shape, heat[None], shape[None], mask[None],
                     weight[None])
    assert float(missed) > 1.0
    # When measuring, a box found on the panel is dropped: its centre inside, or half of it over the panel.
    panel = [[100.0, 100.0, 130.0, 130.0]]
    assert _in_ignore([105.0, 105.0, 125.0, 125.0], panel) and _in_ignore([95.0, 95.0, 128.0, 128.0], panel)
    assert not _in_ignore([140.0, 100.0, 170.0, 130.0], panel) and not _in_ignore([120.0, 120.0, 180.0, 180.0], panel)


def test_fuse_kernel_one_gives_the_same_maps():
    model = build_signnet(stride=4, fuse_kernel=1).eval()
    heat, shape = model(torch.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2))
    assert heat.shape == (1, 1, BAND_HEIGHT // 8, BAND_WIDTH // 8) and shape.shape[1] == 4
    assert model.fuse[0].kernel_size == (1, 1)
    with pytest.raises(ValueError):
        build_signnet(fuse_kernel=2)


def test_export_gives_one_output_equal_to_the_two(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    import numpy as np

    from nnslr_tools.signnet import decode_output, export_signnet_onnx
    model = build_signnet((8, 8, 8, 8), 8, 1)
    torch.save({'task': 'signnet', 'state_dict': model.state_dict(), 'widths': [8, 8, 8, 8], 'stride': 8,
                'fuse_kernel': 1, 'band': [BAND_WIDTH // 2, BAND_HEIGHT // 2], 'epoch': 1}, tmp_path / 'signnet-best.pt')
    report = export_signnet_onnx(tmp_path / 'signnet-best.pt', tmp_path / 'signnet.onnx')
    assert report['verification']['passed'] and (tmp_path / 'signnet.onnx.contract.json').exists()
    session = ort.InferenceSession(str(tmp_path / 'signnet.onnx'), providers=['CPUExecutionProvider'])
    assert [o.name for o in session.get_outputs()] == ['out']
    images = np.random.rand(1, 3, BAND_HEIGHT // 2, BAND_WIDTH // 2).astype(np.float32)
    out = session.run(['out'], {'images': images})[0]
    assert out.shape == (1, 5, BAND_HEIGHT // 16, BAND_WIDTH // 16)
    with torch.no_grad():
        heat, shape = model.eval()(torch.from_numpy(images))
    assert np.allclose(out, torch.cat([heat, shape], 1).numpy(), atol=1e-4)
    first = decode_output(torch.from_numpy(out), 8, min_score=0.0)[0][0]
    same = decode(heat, shape, 8, min_score=0.0)[0][0]
    assert first['score'] == pytest.approx(same['score'], abs=1e-4) and first['box'] == pytest.approx(same['box'], abs=0.05)


def _car_vision():
    """The car's vision.py of the sunnypilot clone, loaded by its path (it imports only math and numpy), or None."""
    import importlib.util
    import os
    from pathlib import Path
    clone = Path(os.environ.get('NNSLR_SUNNYPILOT_CLONE', '/root/sunnypilot'))
    path = clone / 'openpilot' / 'sunnypilot' / 'speed_limits' / 'vision.py'
    if not path.is_file():
        path = clone / 'sunnypilot' / 'speed_limits' / 'vision.py'
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location('car_vision_for_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_car_constants_are_the_cars():
    assert band_size(CAR_BAND[0]) == CAR_BAND and fits_the_car(CAR_BAND, CAR_STRIDE)
    assert not fits_the_car((96, 32), CAR_STRIDE) and not fits_the_car(CAR_BAND, 16)
    vision = _car_vision()
    if vision is None:
        pytest.skip('no sunnypilot clone with speed_limits/vision.py on this machine')
    assert (vision.SIGNNET_WIDTH, vision.SIGNNET_HEIGHT) == CAR_BAND and vision.SIGNNET_STRIDE == CAR_STRIDE
    assert vision.MAX_BOXES == CAR_MAX_BOXES and vision.MIN_BOX_SCORE == CAR_MIN_SCORE


def test_the_server_decode_gives_the_boxes_of_the_car():
    vision = _car_vision()
    if vision is None:
        pytest.skip('no sunnypilot clone with speed_limits/vision.py on this machine')
    generator = torch.Generator().manual_seed(7)
    height, width = CAR_BAND[1] // CAR_STRIDE, CAR_BAND[0] // CAR_STRIDE
    out = torch.cat([torch.rand(1, 1, height, width, generator=generator),
                     torch.randn(1, 2, height, width, generator=generator) + math.log(20),
                     torch.rand(1, 2, height, width, generator=generator)], 1)
    for frame_width in (1344, 1928, CAR_BAND[0]):
        car = vision.signnet_boxes(out.numpy(), frame_width / CAR_BAND[0])
        ours = frame_boxes_of(out[:, :1], out[:, 1:], CAR_STRIDE, CAR_BAND[0], frame_width)
        assert len(ours) == len(car) == 5
        for mine, theirs in zip(ours, car):
            assert mine['score'] == theirs['score'] and mine['box'] == pytest.approx(theirs['box'], rel=1e-12)
        few = frame_boxes_of(out[:, :1], out[:, 1:], CAR_STRIDE, CAR_BAND[0], frame_width, limit=50, min_score=0.9)
        assert [p['score'] for p in few] == [p['score'] for p in vision.signnet_boxes(out.numpy(),
                                             frame_width / CAR_BAND[0], 50, 0.9)]


def test_the_checkpoint_carries_the_data_it_learnt_from(tmp_path):
    import json
    plan = {'dataset_sha256': 'd' * 64, 'split_sha256': 's' * 64, 'families': 'all', 'frames': []}
    saved = checkpoint_of(build_signnet((4, 4, 4, 4), 8, 1), plan, widths=(4, 4, 4, 4), stride=8, fuse_kernel=1,
                          band_width=96, epoch=3)
    assert {key: saved[key] for key in DATA_KEYS} == {key: plan[key] for key in DATA_KEYS}
    assert saved['task'] == 'signnet' and saved['band'] == list(band_size(96)) and saved['epoch'] == 3
    assert checkpoint_data(saved, tmp_path / 'signnet-best.pt') is saved
    old = {key: value for key, value in saved.items() if key not in DATA_KEYS}
    # No result.json beside it: every SignNet so far learnt all the families, never the legacy 'speed'.
    alone = checkpoint_data(old, tmp_path / 'signnet-best.pt')
    assert alone['families'] == 'all' and alone.get('dataset_sha256') is None and alone.get('split_sha256') is None
    assert 'families' not in old and alone['state_dict'] is old['state_dict']
    (tmp_path / 'result.json').write_text(json.dumps({'task': 'detector', **plan, 'families': 'speed'}),
                                          encoding='utf-8')
    other = checkpoint_data(old, tmp_path / 'signnet-best.pt')  # another kind of run: not read
    assert other['families'] == 'all' and other.get('dataset_sha256') is None
    (tmp_path / 'result.json').write_text(json.dumps({'task': 'signnet', **plan}), encoding='utf-8')
    filled = checkpoint_data(old, tmp_path / 'signnet-best.pt')
    assert {key: filled[key] for key in DATA_KEYS} == {key: plan[key] for key in DATA_KEYS}
    (tmp_path / 'result.json').write_text(json.dumps({'task': 'signnet', **plan, 'families': 'speed'}),
                                          encoding='utf-8')
    assert checkpoint_data(old, tmp_path / 'signnet-best.pt')['families'] == 'speed'  # its result.json says so
    detector = {'task': 'detector', 'state_dict': {}}
    (tmp_path / 'result.json').unlink()
    assert checkpoint_data(detector, tmp_path / 'detector-best.pt') is detector  # a Detector keeps its legacy


def test_a_tiny_training_logs_its_lines_and_saves_the_data(tmp_path):
    from PIL import Image
    frames = []
    for index, split in enumerate(['train'] * 4 + ['validation', 'test']):
        image = Image.new('RGB', (128, 96), (90, 90, 90))
        image.paste((240, 20, 20), (40 + index, 6, 56 + index, 22))
        image.save(tmp_path / f'f{index}.jpg')
        frames.append({'image_path': f'f{index}.jpg', 'split': split, 'width': 128, 'height': 96,
                       'boxes': [[40.0 + index, 6.0, 56.0 + index, 22.0]], 'box_families': ['speed_limit'],
                       'label_basis': 'human_review'})
    plan = {'frames': frames, 'dataset_sha256': 'd' * 64, 'split_sha256': 's' * 64, 'families': 'all',
            'trainable': True}
    lines = []
    result = train_signnet(plan, tmp_path, tmp_path / 'run', widths=(4, 4, 4, 4), epochs=2, batch=2, workers=0,
                           band_width=96, log=lines.append)
    assert [line.split(' train_loss ')[0] for line in lines[:2]] == ['signnet epoch 1/2', 'signnet epoch 2/2']
    assert all(' val_recall ' in line and ' val_false_per_frame ' in line and line.endswith('s') for line in lines[:2])
    assert lines[2] == f"signnet measure the best epoch {result['best_epoch']} on validation and test"
    assert lines[3] == f"SignNet complete: {tmp_path / 'run'}" and len(lines) == 4
    saved = torch.load(tmp_path / 'run' / 'signnet-best.pt', map_location='cpu', weights_only=True)
    assert {key: saved[key] for key in DATA_KEYS} == {key: plan[key] for key in DATA_KEYS}
    assert saved['band'] == list(band_size(96)) and saved['fuse_kernel'] == 1 and saved['stride'] == 8
    assert model_of(saved).training is False
    assert {key: result[key] for key in DATA_KEYS} == {key: plan[key] for key in DATA_KEYS}
    for split in ('validation', 'test'):
        assert set(result[split]['car']) == set(result[split]['by_box_budget']['5'])


def test_the_band_cache_gives_the_bands_of_the_frames(tmp_path, monkeypatch):
    import random

    from PIL import Image

    from nnslr_tools import signnet, synthetic_signs
    frames = []
    for index in range(3):
        image = Image.new('RGB', (128, 96), (90 + 20 * index, 90, 90))
        image.paste((240, 20, 20), (40, 30, 64, 54))
        image.save(tmp_path / f'f{index}.png')
        frames.append({'image_path': f'f{index}.png', 'split': 'train', 'width': 128, 'height': 96,
                       'boxes': [[40.0, 30.0, 64.0, 54.0]], 'box_families': ['maximum_speed']})
    frames.append({'image_path': 'gone.png', 'split': 'train', 'width': 128, 'height': 96, 'boxes': [],
                   'box_families': []})
    cache = signnet.band_cache(frames, tmp_path, 96, workers=0)
    assert (cache['frames'], cache['cut'], cache['kept']) == (4, 3, 0) and 'gone.png' not in cache['files']
    assert signnet.band_cache(frames, tmp_path, 96, workers=0)['kept'] == 3  # cut once
    plain = signnet._dataset(frames[:3], tmp_path, 8, train=False, seed=0, band_width=96)
    ready = signnet._dataset(frames[:3], tmp_path, 8, train=False, seed=0, band_width=96, cached=cache['files'])
    for index in range(3):
        assert torch.equal(plain[index][0], ready[index][0])  # the very pixels of the frame's band
        assert all(torch.equal(a, b) for a, b in zip(plain[index][1:5], ready[index][1:5]))
    # In training, a frame with a speed sign is decoded only when a sign is drawn on it; otherwise its band is read.
    decoded = []
    band_image = signnet._band_image
    monkeypatch.setattr(signnet, '_band_image', lambda *args: decoded.append(args) or band_image(*args))
    train = signnet._dataset(frames[:3], tmp_path, 8, train=True, seed=0, band_width=96, cached=cache['files'])
    monkeypatch.setattr(synthetic_signs, 'SHARE', 0.0)
    train[0]
    assert decoded == []
    monkeypatch.setattr(synthetic_signs, 'SHARE', 1.0)
    random.seed(0)
    image = train[1][0]
    assert len(decoded) == 1 and decoded[0][2] is not None and image.shape == plain[1][0].shape


def test_the_mapillary_stage_names_its_lines_and_keeps_the_epoch_of_the_speed_signs(tmp_path):
    from PIL import Image
    frames = []
    for index, split in enumerate(['train'] * 4 + ['validation', 'test']):
        image = Image.new('RGB', (128, 96), (90, 90, 90))
        image.paste((240, 20, 20), (40 + index, 6, 56 + index, 22))
        image.save(tmp_path / f'f{index}.jpg')
        frames.append({'image_path': f'f{index}.jpg', 'split': split, 'width': 128, 'height': 96,
                       'boxes': [[40.0 + index, 6.0, 56.0 + index, 22.0]], 'box_families': ['maximum_speed'],
                       'source': 'mapillary' if split == 'train' else None})
    summary = {'frames': 4, 'boxes_by_family': {'maximum_speed': 4}}
    plan = {'frames': frames, 'dataset_sha256': 'd' * 64, 'split_sha256': 's' * 64, 'families': 'all',
            'trainable': True, 'mapillary': summary, 'mapillary_only': True,
            'trained_on': {'ours': 0, 'ours_boxes': 0, 'mapillary_windows': 4, 'mapillary_boxes': 4}}
    lines = []
    result = train_signnet(plan, tmp_path, tmp_path / 'run' / 'pretrain', widths=(4, 4, 4, 4), epochs=2, batch=2,
                           workers=0, band_width=96, log=lines.append)
    assert lines[0].startswith('signnet_pretrain keeps the epoch with the best validation recall of the speed signs')
    assert [line.split(' train_loss ')[0] for line in lines[1:3]] == ['signnet_pretrain epoch 1/2',
                                                                      'signnet_pretrain epoch 2/2']
    assert all(' val_speed_recall ' in line for line in lines[1:3])
    assert lines[-1] == f"Mapillary stage complete: {tmp_path / 'run' / 'pretrain'}"
    assert result['best_epoch_by'] == 'validation_speed_recall_at_5' and result['mapillary_only'] is True
    assert result['mapillary'] == summary and result['trained_on'] == plan['trained_on']
    assert result['synthetic_end_of_limit']['paintable_train_boxes'] == 0  # no synthetic sign on a window
    assert all('validation_speed_recall_at_5' in epoch for epoch in result['history'])
    stage = tmp_path / 'run' / 'pretrain' / 'signnet-best.pt'
    saved = torch.load(stage, map_location='cpu', weights_only=True)
    assert saved['mapillary_only'] is True and saved['mapillary'] == summary and saved['uses_mapillary'] is True
    # The second stage, from it: its lines say so, and its result keeps what the start learnt from.
    ours = [{**frame, 'source': None} for frame in frames]
    lines = []
    second = train_signnet({**plan, 'frames': ours, 'mapillary': None, 'mapillary_only': False}, tmp_path,
                           tmp_path / 'run', widths=(4, 4, 4, 4), epochs=1, batch=2, workers=0, band_width=96,
                           init_from=stage, log=lines.append)
    assert lines[0].startswith("signnet starts from run's Mapillary stage: every layer of its epoch ")
    assert second['start']['mapillary_only'] is True and second['start']['mapillary'] == summary
    assert second['start']['run'] == 'run' and second['start']['mapillary_stage'] is True
    assert second['best_epoch_by'] == 'validation_recall_at_5' and lines[-1].startswith('SignNet complete: ')
    # The licence goes with the weights: the second stage's file alone says it learnt from a Mapillary stage.
    kept = torch.load(tmp_path / 'run' / 'signnet-best.pt', map_location='cpu', weights_only=True)
    assert kept['mapillary'] is None and kept['uses_mapillary'] is True and second['uses_mapillary'] is True
    assert kept['start']['run'] == 'run' and kept['start']['mapillary_stage'] is True


def test_a_plan_that_cannot_train_is_refused_before_training(tmp_path):
    plan = {'frames': [], 'trainable': False, 'families': 'all', 'counts_by_split': {'train': 0},
            'boxes_by_split': {'train': 0}}
    with pytest.raises(ValueError, match='training_plan_not_trainable'):
        train_signnet(plan, tmp_path, tmp_path / 'run', log=lambda line: None)
    assert not (tmp_path / 'run').exists()


def test_the_default_log_flushes_every_line(tmp_path, monkeypatch):
    """The script tees the lines into signnet.log: each must reach it as it comes, not when the process ends."""
    import builtins

    from PIL import Image

    from nnslr_tools import signnet
    printed = []
    monkeypatch.setattr(builtins, 'print', lambda *args, **kw: printed.append(kw.get('flush')))
    measured = {'recall_iou': 0.5, 'false_positives_per_image': 1.0}
    monkeypatch.setattr(signnet, 'measure', lambda *a, **kw: {'by_box_budget': {'5': measured}, 'car': measured})
    frames = []
    for index, split in enumerate(['train', 'train', 'validation', 'test']):
        Image.new('RGB', (128, 96)).save(tmp_path / f'f{index}.jpg')
        frames.append({'image_path': f'f{index}.jpg', 'split': split, 'width': 128, 'height': 96,
                       'boxes': [[40.0, 6.0, 56.0, 22.0]], 'box_families': ['speed_limit']})
    plan = {'frames': frames, 'trainable': True, 'families': 'all'}
    train_signnet(plan, tmp_path, tmp_path / 'run', widths=(4, 4, 4, 4), epochs=1, batch=2, workers=0, band_width=96)
    assert printed == [True, True, True, True]  # the band cache, the epoch, the measure and the complete lines


def test_the_car_measure_keeps_the_best_5_boxes_of_0_3(tmp_path, monkeypatch):
    """measure()['car'] is what the car reads: the best CAR_MAX_BOXES of score CAR_MIN_SCORE or more. A sign found
    only by the 7th box (0.35) counts at 0.3 but not for the car; at 5 boxes at any score neither."""
    from PIL import Image

    from nnslr_tools import signnet
    Image.new('RGB', (128, 96)).save(tmp_path / 'f.jpg')
    truth = [100.0, 60.0, 120.0, 80.0]
    found = [{'box': [10.0 * i, 0.0, 10.0 * i + 8, 8.0], 'score': score}
             for i, score in enumerate((0.9, 0.8, 0.7, 0.6, 0.5, 0.4))]
    found += [{'box': list(truth), 'score': 0.35}, {'box': [0.0, 50.0, 8.0, 58.0], 'score': 0.2}]
    monkeypatch.setattr(signnet, 'frame_boxes_of', lambda *a, **kw: [dict(p) for p in found])
    frames = [{'image_path': 'f.jpg', 'split': 'test', 'width': 128, 'height': 96, 'boxes': [truth],
               'box_families': ['speed_limit']}]
    result = signnet.measure(build_signnet((4, 4, 4, 4), 8, 1), frames, tmp_path, 8, 'cpu', workers=0, band_width=96)
    assert result['car']['true_positives'] == 0 and result['car']['false_positives'] == CAR_MAX_BOXES
    assert result['by_score_threshold'][str(CAR_MIN_SCORE)]['true_positives'] == 1
    assert result['by_score_threshold'][str(CAR_MIN_SCORE)]['false_positives'] == 6
    assert result['by_box_budget']['5']['true_positives'] == 0


def test_the_car_measure_also_with_the_lenient_match(tmp_path, monkeypatch):
    """measure()['car_centre']: the car's boxes matched leniently. A box 2 px tighter on each side of a far, 10 px sign
    (IoU 0.36) is a miss and a false box at IoU 0.5, found with the lenient match; a huge box on the other sign is
    false in both."""
    from PIL import Image

    from nnslr_tools import signnet
    Image.new('RGB', (128, 96)).save(tmp_path / 'f.jpg')
    truth = [[100.0, 50.0, 110.0, 60.0], [20.0, 20.0, 30.0, 30.0]]
    found = [{'box': [102.0, 52.0, 108.0, 58.0], 'score': 0.9}, {'box': [0.0, 0.0, 50.0, 50.0], 'score': 0.8}]
    monkeypatch.setattr(signnet, 'frame_boxes_of', lambda *a, **kw: [dict(p) for p in found])
    frames = [{'image_path': 'f.jpg', 'split': 'test', 'width': 128, 'height': 96, 'boxes': truth,
               'box_families': ['maximum_speed', 'other_sign']}]
    result = signnet.measure(build_signnet((4, 4, 4, 4), 8, 1), frames, tmp_path, 8, 'cpu', workers=0, band_width=96)
    car, lenient = result['car'], result['car_centre']
    assert (car['true_positives'], car['false_positives'], car['recall_iou']) == (0, 2, 0.0)
    assert (lenient['true_positives'], lenient['false_positives'], lenient['recall_iou']) == (1, 1, 0.5)
    assert lenient['recall_by_family']['speed']['found'] == 1 and set(lenient) == set(car)
    assert signnet.car_measure(result, match='centre') is lenient and signnet.car_measure(result) is car
    assert signnet.car_measure({'car': car}, match='centre') is None  # a result.json written before it


def _tiny_run(tmp_path):
    """A tiny SignNet run on the CPU (untrained weights): its plan, and a result.json as one written before
    car_centre (measure's output without it)."""
    import json

    from PIL import Image

    from nnslr_tools import signnet
    frames = []
    for index, split in enumerate(['train', 'validation', 'validation', 'test']):
        image = Image.new('RGB', (128, 96), (90, 90, 90))
        image.paste((240, 20, 20), (40 + index, 6, 56 + index, 22))
        image.save(tmp_path / f'f{index}.jpg')
        frames.append({'image_path': f'f{index}.jpg', 'split': split, 'width': 128, 'height': 96,
                       'boxes': [[40.0 + index, 6.0, 56.0 + index, 22.0]], 'box_families': ['speed_limit']})
    plan = {'frames': frames, 'dataset_sha256': 'd' * 64, 'split_sha256': 's' * 64, 'families': 'all'}
    torch.manual_seed(3)
    model = build_signnet((4, 4, 4, 4), 8, 1).eval()
    run = tmp_path / 'run'
    run.mkdir()
    torch.save(checkpoint_of(model, plan, widths=(4, 4, 4, 4), stride=8, fuse_kernel=1, band_width=96, epoch=1),
               run / 'signnet-best.pt')
    result = {'task': 'signnet', 'band': list(band_size(96)), 'stride': 8, 'best_epoch': 1,
              **{key: plan[key] for key in DATA_KEYS}}
    for split in ('validation', 'test'):
        measured = signnet.measure(model, [f for f in frames if f['split'] == split], tmp_path, 8, 'cpu', workers=0,
                                   band_width=96)
        measured.pop('car_centre')
        result[split] = measured
    (run / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return plan, run, result


def test_remeasure_adds_the_lenient_car_measure_without_training(tmp_path):
    import json

    from nnslr_tools import signnet
    plan, run, before = _tiny_run(tmp_path)
    dry = signnet.remeasure_car_centre(plan, tmp_path, run, device='cpu', workers=0, dry_run=True)
    assert dry['dry_run'] and 'car_centre' not in json.loads((run / 'result.json').read_text())['test']
    summary = signnet.remeasure_car_centre(plan, tmp_path, run, device='cpu', workers=0)
    after = json.loads((run / 'result.json').read_text())
    for split, frames in (('validation', 2), ('test', 1)):
        assert summary[split]['frames'] == frames and summary[split]['car_unchanged'] is True
        assert set(after[split]['car_centre']) == set(before[split]['car'])
        assert after[split]['car_centre']['recall_iou'] == summary[split]['centre_recall']
        # Nothing else changes, and the lenient match never finds fewer.
        assert {key: value for key, value in after[split].items() if key != 'car_centre'} == before[split]
        assert after[split]['car_centre']['true_positives'] >= after[split]['car']['true_positives']
    assert {key: value for key, value in after.items() if key not in ('validation', 'test')} == {
        key: value for key, value in before.items() if key not in ('validation', 'test')}
    assert not (run / 'result.json.tmp').exists()


def test_remeasure_refuses_another_plan(tmp_path):
    from nnslr_tools import signnet
    plan, run, _ = _tiny_run(tmp_path)
    with pytest.raises(ValueError, match='remeasure_plan_mismatch: .*split_sha256'):
        signnet.remeasure_car_centre({**plan, 'split_sha256': 'x' * 64}, tmp_path, run, device='cpu', workers=0)
    with pytest.raises(ValueError, match='remeasure_not_a_finished_signnet'):
        signnet.remeasure_car_centre(plan, tmp_path, tmp_path / 'nothing', device='cpu', workers=0)
