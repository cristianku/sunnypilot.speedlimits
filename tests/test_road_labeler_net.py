"""The road labeler's network on the CPU with synthetic drives: shapes, the masked loss, the windows of the prediction and
a whole dry training with its checkpoint (torch: .venv-gpu)."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from nnslr_tools import road_labeler, road_labeler_data as data, road_labeler_net as net  # noqa: E402
from nnslr_tools.road_types import ROAD_TYPES  # noqa: E402

NS = 1_000_000_000
KINDS = ['tunnel', 'narrow']


def synthetic_drive(route: str, length: int, feature_dim: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    kinds = (np.arange(length)[:, None] % 40 < 10).astype(np.float32).repeat(len(KINDS), axis=1)
    return {'route': route, 'steps': [{'key': f'0/{n // 2}', 'segment': 0, 'decoded': 10 * n, 'step': n % 2,
                                       'frame': n // 2, 'ns': n * NS // 2, 'image': None} for n in range(length)],
            'ns': np.arange(length, dtype=np.int64) * NS // 2,
            'partitions': ['train'] * 30 + ['validation'] * 20 + ['train'] * (length - 50),
            'inputs': rng.random((length, data.INPUTS)).astype(np.float32), 'kind_names': list(KINDS),
            'type': (np.arange(length) // 30 % 2).astype(np.int64), 'type_weight': np.ones(length, dtype=np.float32),
            'type_person': np.arange(length) % 3 == 0, 'map_type': np.zeros(length, dtype=np.int64),
            'kinds': kinds, 'kinds_weight': np.ones_like(kinds), 'works': kinds[:, 0].copy(),
            'works_weight': np.ones(length, dtype=np.float32), 'lines': [],
            'features': rng.standard_normal((length, feature_dim)).astype(np.float16),
            'present': np.ones(length, dtype=bool)}


def test_the_tiny_encoder_gives_a_grid_of_features():
    encoder = net.build_encoder(net.ENCODER_TINY)
    width, height = net.SIZES[net.ENCODER_TINY]
    out = encoder(torch.rand(2, 3, height, width))
    assert out.shape == (2, net.feature_size(net.ENCODER_TINY))
    assert not any(p.requires_grad for p in encoder.parameters())
    # The same seed, the same random encoder: a dry run's checkpoint keeps its weights anyway.
    again = net.build_encoder(net.ENCODER_TINY)
    assert all(torch.equal(a, b) for a, b in zip(encoder.state_dict().values(), again.state_dict().values()))


def test_heads_and_a_loss_that_reads_only_the_weights():
    model = net.build_labeler_net(32, len(KINDS))
    said = model(torch.randn(2, 7, 32), torch.randn(2, 7, net.NET_INPUTS))
    assert said['types'].shape == (2, 7, len(ROAD_TYPES)) and said['kinds'].shape == (2, 7, 2)
    assert said['works'].shape == (2, 7)
    answers = {'type': torch.full((2, 7), -1), 'type_weight': torch.zeros(2, 7), 'kinds': torch.ones(2, 7, 2),
               'kinds_weight': torch.zeros(2, 7, 2), 'works': torch.ones(2, 7), 'works_weight': torch.zeros(2, 7)}
    loss = net.labeler_loss(said, answers)
    assert float(loss.detach()) == 0.0
    loss.backward()   # nothing known: no error and no gradient from nowhere
    answers['works_weight'][0, 3] = 1.0
    assert float(net.labeler_loss(model(torch.randn(2, 7, 32), torch.randn(2, 7, net.NET_INPUTS)), answers).detach()) > 0
    assert net.build_labeler_net(32, 0)(torch.randn(1, 3, 32), torch.randn(1, 3, net.NET_INPUTS))['kinds'] is None


def test_the_prediction_in_windows_covers_the_drive():
    torch.manual_seed(0)
    model = net.build_labeler_net(16, len(KINDS))
    drive = synthetic_drive('00000001--0123456789', 230, 16)
    said = road_labeler.predict_drive(model, drive, 'cpu', window=100, margin=25)
    assert said['types'].shape == (230, len(ROAD_TYPES)) and said['kinds'].shape == (230, 2)
    assert np.allclose(said['types'].sum(axis=1), 1, atol=1e-5) and (said['works'] > 0).all()
    whole = road_labeler.predict_drive(model, drive, 'cpu', window=1000, margin=25)
    # A bidirectional memory sees a different context in a window; the middle is close, never the same call.
    assert said['types'].shape == whole['types'].shape


def test_a_dropped_map_looks_like_no_line_and_only_train_weighs():
    drive = synthetic_drive('00000001--0123456789', 60, 8)
    feats, extra, answers = road_labeler._stack([drive], [(0, 30, 50)], 'cpu', None)
    assert feats.shape == (1, 20, 8) and extra.shape == (1, 20, net.NET_INPUTS)
    assert float(answers['type_weight'].sum()) == 0      # 30-50 is validation
    road_labeler.MAP_DROP_SHARE, share = 1.0, road_labeler.MAP_DROP_SHARE
    try:
        _, extra, _ = road_labeler._stack([drive], [(0, 0, 20)], 'cpu', __import__('random').Random(0))
    finally:
        road_labeler.MAP_DROP_SHARE = share
    no_line = torch.from_numpy(data.map_vector(None, set())[road_labeler._MAP_INPUTS])
    assert torch.equal(extra[0, :, road_labeler._MAP_INPUTS], no_line.expand(20, -1))
    # The GPS, the speed and the picture flag stay.
    assert torch.equal(extra[0, :, -1], torch.ones(20))


def test_a_dry_training_writes_its_checkpoint_and_measure(tmp_path, monkeypatch):
    dim = net.feature_size(net.ENCODER_TINY)
    drives = [synthetic_drive('00000001--0123456789', 120, dim, 1), synthetic_drive('00000002--0123456789', 120, dim, 2)]
    monkeypatch.setattr(data, 'dataset', lambda root, merged_of, routes=None, log=None: drives)

    def features(root, cut, *args):
        for drive in cut:   # as drive_features after the dry run's cut
            drive['features'], drive['present'] = drive['features'][:len(drive['steps'])], drive['present'][:len(drive['steps'])]

    monkeypatch.setattr(road_labeler, '_features', features)
    run = tmp_path / '2026-11-08_2200-road-labeler'
    result = road_labeler.train_labeler(tmp_path, run, lambda route: {}, dry_run=True, log=lambda text: None)
    assert result['dry_run'] and result['kinds'] == KINDS and len(result['history']) == 1
    assert result['validation']['types']['steps'] > 0 and result['validation']['types']['map_on_person'] is not None
    assert set(result['validation']['kinds']) == set(KINDS)
    assert json.loads((run / road_labeler.RESULT).read_text())['run'] == run.name
    model, encoder, checkpoint = road_labeler.load_labeler(run, 'cpu')
    assert checkpoint['dry_run'] and checkpoint['encoder'] == net.ENCODER_TINY and checkpoint['kinds'] == KINDS
    assert sorted(checkpoint['trained']) == sorted(KINDS + ['works'])
    drive = synthetic_drive('00000001--0123456789', 120, dim, 1)
    lines = road_labeler.proposal_lines(drive, road_labeler.predict_drive(model, drive, 'cpu'), checkpoint)
    assert lines[0]['run'] == run.name and len(lines) == 121
    assert set(lines[1]['kinds']) == set(KINDS) and lines[1]['road_type'] in ROAD_TYPES


def test_a_broken_road_video_falls_back_to_the_1_hz_pictures(tmp_path, monkeypatch):
    from PIL import Image
    route = '00000001--0123456789'
    video = tmp_path / 'raw/routes' / route / '0' / 'fcamera.hevc'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'not a video')
    png = tmp_path / '0-0.png'
    Image.new('RGB', (64, 36), (200, 10, 10)).save(png)
    steps = [(0, {'segment': 0, 'decoded': 0, 'image': str(png)}), (1, {'segment': 0, 'decoded': 10, 'image': str(png)})]

    def broken(*args):
        raise ValueError('not decoded')

    monkeypatch.setattr(net, 'decode_segment', broken)
    pictures = net.segment_pictures(tmp_path, route, 0, steps, net.SIZES[net.ENCODER_TINY])
    width, height = net.SIZES[net.ENCODER_TINY]
    assert sorted(pictures) == [0, 1] and pictures[0].shape == (height, width, 3)
