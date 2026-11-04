"""The training loop on a tiny drive, and the measure by hand. The measure needs only numpy: the rest is skipped without
torch (see test_brain_net.py for how to run it)."""
import random

import pytest

NS = 1_000_000_000
T = 8
GROUPS = ('signs', 'lead', 'country')


def fake_drive(partitions, *, speed=19, works=None, marks=(), route='00000001--abcdef0123'):
    import numpy
    from nnslr_tools.brain_targets import INPUT_SIZES, pack_inputs
    count = len(partitions)
    inputs = {name: numpy.zeros((count, size), dtype=numpy.float32) for name, size in INPUT_SIZES.items()}
    drive = {'route': route, 'frames': [{}] * count, 'ns': numpy.arange(count, dtype=numpy.int64) * NS, 'partitions': list(partitions),
             'speed': numpy.full(count, speed, dtype=numpy.int64), 'speed_weight': numpy.ones(count, dtype=numpy.float32),
             'works': numpy.zeros(count, dtype=numpy.float32) if works is None else numpy.array(works, dtype=numpy.float32),
             'works_weight': numpy.ones(count, dtype=numpy.float32), 'inputs': inputs, 'marks': list(marks),
             'own_kph': numpy.full(count, numpy.nan, dtype=numpy.float32)}
    drive['extra'] = pack_inputs(inputs, GROUPS)
    return drive


def said(kph, works=None):
    import numpy
    return {'kph': numpy.array(kph), 'works': None if works is None else numpy.array(works, dtype=numpy.float32)}


def test_the_measure_counts_only_the_frames_of_its_partition():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test', 'test', 'train', 'test'])
    found = measure([drive], [said([100, 110, 30, 100])], 'test')
    assert found['speed']['frames'] == 3
    assert found['speed']['right'] == pytest.approx(2 / 3)
    assert found['speed']['mean_error_kph'] == pytest.approx(10 / 3)


def test_the_frames_without_gas_are_measured_apart():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 3)
    drive['inputs']['pedals'][1, 0] = 1
    found = measure([drive], [said([100, 130, 100])], 'test')
    assert found['speed']['mean_error_kph'] == pytest.approx(10)
    assert found['no_gas'] == {'frames': 2, 'right': 1.0, 'mean_error_kph': 0.0}


def test_a_network_that_follows_the_car_shows_where_the_car_is_far_from_the_label():
    import numpy
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 4, speed=23)
    drive['own_kph'][:] = 90, 90, 115, numpy.nan
    found = measure([drive], [said([90, 120, 115, 120])], 'test')
    assert found['speed']['frames'] == 4
    assert found['far_from_own_speed'] == {'frames': 2, 'right': 0.5, 'mean_error_kph': 15.0, 'nearer_own_speed': 0.5}


def test_with_the_car_near_the_label_the_measure_is_empty():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 2, speed=23)
    drive['own_kph'][:] = 118
    assert measure([drive], [said([120, 120])], 'test')['far_from_own_speed']['nearer_own_speed'] is None


def test_the_works_output_is_measured_on_its_own_answers():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 4, works=[1, 1, 0, 0])
    drive['works_weight'][3] = 0
    found = measure([drive], [said([100] * 4, [.9, .1, .9, .9])], 'test')['works']
    assert (found['true_positive'], found['false_positive'], found['false_negative']) == (1, 1, 1)
    assert found['precision'] == pytest.approx(.5) and found['recall'] == pytest.approx(.5)


def test_a_site_says_how_late_the_speed_came_back_and_how_much_was_said_too_early():
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 10, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS, 'route': 'r'}])
    drive['speed'][:6] = speed_class(100)
    drive['speed'][6:] = speed_class(120)
    found = measure([drive], [said([100, 100, 100, 120, 100, 100, 100, 100, 120, 120])], 'test')['sites']
    assert len(found) == 1
    assert found[0]['above_label'] == pytest.approx(.25)
    assert found[0]['end_delay_s'] == pytest.approx(2)


def test_a_site_where_the_restriction_goes_on_has_no_delay():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 10, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS}])
    assert measure([drive], [said([100] * 10)], 'test')['sites'][0]['end_delay_s'] is None


def test_a_site_of_another_partition_is_not_here():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['train'] * 10, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS}])
    assert measure([drive], [said([100] * 10)], 'test')['sites'] == []


def test_no_frames_is_no_measure():
    from nnslr_tools.brain_train import measure
    assert measure([], [], 'test')['speed'] == {'frames': 0, 'right': None, 'mean_error_kph': None}


def torch_or_skip():
    pytest.importorskip('torchvision')
    return pytest.importorskip('torch')


def tiny(torch, works=True):
    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(GROUPS, pretrained=False, works=works)
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0.0
    return model


def pictures(drive, seed=0):
    import numpy
    return numpy.random.default_rng(seed).integers(0, 255, (len(drive['partitions']), 144, 256, 3), dtype=numpy.uint8)


def one_epoch(torch, model, drive, array):
    from nnslr_tools.brain_train import train_epoch
    from nnslr_tools.road_type_net import FrameStore
    store = FrameStore({drive['route']: array}, torch.device('cpu'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return train_epoch(model, optimizer, [drive], store, chunk=4, rng=random.Random(0), device=torch.device('cpu'), augment=False)


def test_a_drive_of_labels_moves_the_shared_weights():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T, works=[0, 0, 0, 1, 1, 1, 0, 0])
    model = tiny(torch)
    before = {name: value.clone() for name, value in model.named_parameters()}
    assert one_epoch(torch, model, drive, pictures(drive)) > 0
    assert all(not torch.equal(before[name], value) for name, value in model.named_parameters()
               if name.startswith(('gru', 'project', 'speed_head', 'works_head')))


def test_a_held_out_picture_is_never_in_a_step_of_the_gradient():
    torch = torch_or_skip()
    drive = fake_drive(['test'] * 3 + ['train'] * 5)
    model = tiny(torch)
    seen = []
    model.backbone.register_forward_pre_hook(lambda _layer, args: seen.append((args[0].shape[0], torch.is_grad_enabled())))
    one_epoch(torch, model, drive, pictures(drive))
    assert seen[0] == (3, False)
    assert all(learning for _frames, learning in seen[1:])
    assert sum(frames for frames, _learning in seen[1:]) == 5


def test_a_piece_with_no_answer_is_no_step():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T)
    drive['speed_weight'][:] = 0
    drive['works_weight'][:] = 0
    model = tiny(torch)
    before = [value.clone() for value in model.parameters()]
    assert one_epoch(torch, model, drive, pictures(drive)) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_the_network_goes_through_a_whole_drive_as_on_the_car():
    torch = torch_or_skip()
    from nnslr_tools.brain_train import predict
    from nnslr_tools.road_type_net import FrameStore
    drive = fake_drive(['test'] * T)
    store = FrameStore({drive['route']: pictures(drive)}, torch.device('cpu'))
    found = predict(tiny(torch), drive, store, chunk=3, device=torch.device('cpu'))
    assert found['kph'].shape == (T,) and found['works'].shape == (T,)
    assert found['kph'].min() >= 5 and found['kph'].max() <= 200
    assert predict(tiny(torch, works=False), drive, store, chunk=3, device=torch.device('cpu'))['works'] is None
