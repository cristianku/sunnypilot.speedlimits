"""The training loop on a tiny drive, and the measure by hand. The measure needs only numpy: the rest is skipped without
torch (see test_brain_net.py for how to run it)."""
import random

import pytest

NS = 1_000_000_000
T = 8
GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')


def fake_drive(partitions, *, speed=19, works=None, marks=(), route='00000001--abcdef0123'):
    import numpy
    from nnslr_tools.brain_targets import INPUT_SIZES, pack_inputs
    count = len(partitions)
    inputs = {name: numpy.zeros((count, size), dtype=numpy.float32) for name, size in INPUT_SIZES.items()}
    drive = {'route': route, 'frames': [{}] * count, 'ns': numpy.arange(count, dtype=numpy.int64) * NS, 'partitions': list(partitions),
             'cuts': set(),
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


def test_a_site_counts_where_any_of_it_is_held_out():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 3 + ['buffer'] * 7, marks=[{'start_ns': 2 * NS, 'end_ns': 8 * NS}])
    assert len(measure([drive], [said([100] * 10)], 'test')['sites']) == 1


def test_a_label_far_after_the_end_of_works_is_no_end_of_works():
    import numpy
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import END_WINDOW_S, measure
    count = END_WINDOW_S + 20
    drive = fake_drive(['test'] * count, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS}])
    drive['speed'][:] = speed_class(100)
    drive['speed'][END_WINDOW_S + 10:] = speed_class(120)
    drive['speed_weight'][6:END_WINDOW_S + 10] = 0
    found = measure([drive], [{'kph': numpy.full(count, 120), 'works': None}], 'test')
    assert found['sites'][0]['end_delay_s'] is None


def test_the_measure_says_which_speeds_the_network_said():
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test', 'test', 'test', 'train'], speed=speed_class(100))
    assert measure([drive], [said([100, 120, 100, 80])], 'test')['speeds_said'] == [100, 120]


def test_the_network_has_to_beat_the_sign_on_the_frames_with_a_number_in_force():
    from nnslr_tools.brain_targets import BRAIN_INPUTS, speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 4 + ['train'], speed=speed_class(100))
    signs = drive['inputs']['signs']
    signs[:3, 0], signs[:3, 1] = 80 / BRAIN_INPUTS.SIGN_KPH_SCALE, 1
    signs[0, 0] = 100 / BRAIN_INPUTS.SIGN_KPH_SCALE
    signs[2, 6] = 1
    signs[4, 0], signs[4, 1] = 100 / BRAIN_INPUTS.SIGN_KPH_SCALE, 1
    found = measure([drive], [said([100, 120, 90, 90, 90])], 'test')['follow_the_sign']
    assert found['sign'] == {'frames': 2, 'right': 0.5, 'mean_error_kph': 10.0}
    assert found['network'] == {'frames': 2, 'right': 0.5, 'mean_error_kph': 10.0}


def test_the_network_has_to_beat_one_speed_said_everywhere():
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 4, speed=speed_class(120))
    drive['speed'][0] = speed_class(100)
    found = measure([drive], [said([120] * 4)], 'test')
    assert found['one_speed'] == {'kph': 120, 'mean_error_kph': pytest.approx(5)}
    assert measure([], [], 'test')['one_speed'] == {'kph': None, 'mean_error_kph': None}


def test_with_only_marks_a_constant_yes_has_no_precision_to_show():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 3, works=[1, 1, 1])
    found = measure([drive], [said([100] * 3, [.9, .9, .9])], 'test')['works']
    assert found['precision'] is None and found['recall'] == 1.0
    drive = fake_drive(['test'] * 3)
    found = measure([drive], [said([100] * 3, [.1, .1, .1])], 'test')['works']
    assert found['recall'] is None and found['precision'] == 0


def test_marks_alone_leave_the_loss_of_train_and_stay_in_the_measure():
    from nnslr_tools.brain_train import _works_needs_both
    said_to = []
    drive = fake_drive(['train', 'train', 'test'], works=[1, 1, 1])
    _works_needs_both([drive], said_to.append)
    assert list(drive['works_weight']) == [0, 0, 1] and 'checked to here' in said_to[0]


def test_marks_with_a_stretch_checked_stay_in_the_loss():
    from nnslr_tools.brain_train import _works_needs_both
    said_to = []
    drive = fake_drive(['train', 'train', 'test'], works=[1, 0, 1])
    _works_needs_both([drive], said_to.append)
    assert list(drive['works_weight']) == [1, 1, 1] and not said_to


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
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.running_var.mul_(0.2)  # or an encoder that learned nothing makes every picture the same: 1e-8 apart
    return model


def pictures(drive, seed=0):
    import numpy
    return numpy.random.default_rng(seed).integers(0, 255, (len(drive['partitions']), 144, 256, 3), dtype=numpy.uint8)


def one_epoch(torch, model, drive, array, augment=False):
    from nnslr_tools.brain_train import train_epoch
    from nnslr_tools.road_type_net import FrameStore
    store = FrameStore({drive['route']: array}, torch.device('cpu'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return train_epoch(model, optimizer, [drive], store, chunk=4, rng=random.Random(0), device=torch.device('cpu'), augment=augment)


def test_a_drive_of_labels_moves_the_shared_weights():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T, works=[0, 0, 0, 1, 1, 1, 0, 0])
    model = tiny(torch)
    before = {name: value.clone() for name, value in model.named_parameters()}
    assert one_epoch(torch, model, drive, pictures(drive)) > 0
    assert all(not torch.equal(before[name], value) for name, value in model.named_parameters()
               if name.startswith(('gru', 'project', 'speed_head', 'works_head')))


def test_a_held_out_picture_is_never_run_in_training():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * 3 + ['test'] * 3 + ['train'] * 5)
    model = tiny(torch)
    seen = []
    model.backbone.register_forward_pre_hook(lambda _layer, args: seen.append(args[0].shape[0]))
    one_epoch(torch, model, drive, pictures(drive))
    assert sum(seen) == 8


@pytest.mark.parametrize('partitions', [['train'] * 6 + ['test'] * 4 + ['train'] * 6,
                                        ['train'] * 6 + ['test'] * 3 + ['buffer'] + ['train'] * 6])
def test_a_held_out_picture_moves_no_weight_also_through_the_memory(partitions):
    torch = torch_or_skip()
    drive = fake_drive(partitions)
    array, other = pictures(drive), pictures(drive)
    held = [name == 'test' for name in partitions]
    other[held] = 255 - other[held]
    trained = []
    for pictures_of_run in (array, other):
        model = tiny(torch)
        one_epoch(torch, model, drive, pictures_of_run)
        trained.append(model.state_dict())
    assert all(torch.equal(value, trained[1][name]) for name, value in trained[0].items())


def test_the_memory_goes_through_a_train_stretch_from_piece_to_piece():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T)
    drive['speed_weight'][:T - 1] = 0  # the piece of the last frame starts after the fourth, whatever the offset
    drive['works_weight'][:] = 0
    array, other = pictures(drive), pictures(drive, 1)
    other[4:] = array[4:]
    trained = []
    for pictures_of_run in (array, other):
        model = tiny(torch)
        one_epoch(torch, model, drive, pictures_of_run)
        trained.append(model.state_dict())
    assert not all(torch.equal(value, trained[1][name]) for name, value in trained[0].items())


def test_a_cut_stops_the_memory_in_training():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T)
    drive['speed_weight'][:4] = 0
    drive['works_weight'][:] = 0
    drive['cuts'] = {4}
    array, other = pictures(drive), pictures(drive, 1)
    other[4:] = array[4:]
    trained = []
    for pictures_of_run in (array, other):
        model = tiny(torch)
        one_epoch(torch, model, drive, pictures_of_run)
        trained.append(model.state_dict())
    assert all(torch.equal(value, trained[1][name]) for name, value in trained[0].items())


def test_without_the_works_task_a_works_answer_is_no_step():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T, works=[1] * T)
    drive['speed_weight'][:] = 0
    model = tiny(torch, works=False)
    before = [value.clone() for value in model.parameters()]
    assert one_epoch(torch, model, drive, pictures(drive)) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_one_drive_has_one_look_in_all_its_pieces():
    torch = torch_or_skip()
    from nnslr_tools.brain_train import _same_augment
    images = torch.rand(1, 2, 3, 144, 256)
    device = torch.device('cpu')
    assert torch.equal(_same_augment(torch, images, 7, device), _same_augment(torch, images, 7, device))
    assert not torch.equal(_same_augment(torch, images, 7, device), _same_augment(torch, images, 8, device))


def test_a_look_of_a_drive_leaves_the_random_numbers_of_the_training_alone():
    torch = torch_or_skip()
    from nnslr_tools.brain_train import _same_augment
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    _same_augment(torch, torch.rand(1, 1, 3, 144, 256) * 0, 7, torch.device('cpu'))
    torch.manual_seed(0)
    state = torch.get_rng_state()
    _same_augment(torch, torch.zeros(1, 1, 3, 144, 256), 7, torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(torch.rand(1), expected)


def test_an_augmented_epoch_runs():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T, works=[0, 0, 0, 1, 1, 1, 0, 0])
    assert one_epoch(torch, tiny(torch), drive, pictures(drive), augment=True) > 0


def test_a_piece_with_no_answer_is_no_step():
    torch = torch_or_skip()
    drive = fake_drive(['train'] * T)
    drive['speed_weight'][:] = 0
    drive['works_weight'][:] = 0
    model = tiny(torch)
    before = [value.clone() for value in model.parameters()]
    assert one_epoch(torch, model, drive, pictures(drive)) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def predicted(torch, drive, array, **options):
    from nnslr_tools.brain_train import predict
    from nnslr_tools.road_type_net import FrameStore
    store = FrameStore({drive['route']: array}, torch.device('cpu'))
    return predict(tiny(torch), drive, store, chunk=3, device=torch.device('cpu'), **options)['works']


def after_something_else_at_the_start(torch, drive, **options):
    """What is said from the fourth frame on, with and without other inputs in the first three."""
    array = pictures(drive)
    plain = predicted(torch, drive, array, **options)[3:]
    drive['extra'] = drive['extra'].copy()
    drive['extra'][:3] = 1
    return plain, predicted(torch, drive, array, **options)[3:]


def test_on_a_drive_the_memory_goes_from_piece_to_piece():
    torch = torch_or_skip()
    plain, other = after_something_else_at_the_start(torch, fake_drive(['test'] * T))
    assert not (plain == other).all()


def test_a_cut_stops_the_memory_on_a_drive_too():
    torch = torch_or_skip()
    drive = fake_drive(['test'] * T)
    drive['cuts'] = {3}
    plain, other = after_something_else_at_the_start(torch, drive)
    assert (plain == other).all()


def test_cold_starts_the_memory_again_where_a_stretch_out_of_train_begins():
    torch = torch_or_skip()
    partitions = ['train'] * 3 + ['buffer'] * 2 + ['test'] * 3
    plain, other = after_something_else_at_the_start(torch, fake_drive(partitions))
    assert not (plain == other).all()
    plain, other = after_something_else_at_the_start(torch, fake_drive(partitions), cold=True)
    assert (plain == other).all()


def run_training(torch, tmp_path, monkeypatch, scores, change=lambda _drive: None):
    from nnslr_tools import brain_train
    drive = fake_drive(['train'] * 4 + ['validation'] * 2 + ['test'] * 2)
    change(drive)
    drive['reader'], drive['reader_current'], drive['signs_read'] = {'vision': 'abc'}, True, True
    monkeypatch.setattr(brain_train, 'dataset', lambda *_args, **_options: [drive])
    monkeypatch.setattr(brain_train, 'read_split', lambda _root: {'sha256': 'split'})
    monkeypatch.setattr(brain_train, '_pictures', lambda *_args, **_options: {drive['route']: pictures(drive)})
    monkeypatch.setattr(brain_train, 'build_brain_net', lambda *_args, **_options: tiny(torch))
    real, left = brain_train.measure, list(scores)

    def scripted(drives, answers, partition):
        found = real(drives, answers, partition)
        if partition == 'validation' and left:
            found['speed']['mean_error_kph'] = left.pop(0)
        return found

    monkeypatch.setattr(brain_train, 'measure', scripted)
    result = brain_train.train_brain(tmp_path, tmp_path / 'run', None, epochs=len(scores), patience=2, chunk=4, device='cpu',
                                     log=lambda _text: None)
    return result, torch.load(tmp_path / 'run' / 'brain-best.pt', weights_only=False)


def test_the_epoch_kept_is_the_best_on_validation_and_the_checkpoint_says_what_it_came_from(tmp_path, monkeypatch):
    torch = torch_or_skip()
    result, saved = run_training(torch, tmp_path, monkeypatch, [9.0, 3.0, 5.0])
    assert result['best_epoch'] == 1 and saved['epoch'] == 1
    assert saved['split'] == 'split' and saved['reader'] == {'vision': 'abc'}
    assert saved['features'] == 128 and saved['hidden'] == 128
    assert {'validation_cold', 'test_cold'} <= set(result)


def at_two_in_the_afternoon(drive):
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    drive['inputs']['hour'][:] = BRAIN_INPUTS.hour_row(14.0)


def plate_of_hours(drive):
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    at_two_in_the_afternoon(drive)
    drive['inputs']['signs'][:, 0], drive['inputs']['signs'][:, 1] = 30 / BRAIN_INPUTS.SIGN_KPH_SCALE, 1
    drive['inputs']['plate'][:, BRAIN_INPUTS.PLATES.index('time')] = 1


def watched(monkeypatch):
    """What each epoch trains on and what the network is measured on: the hours of them, as rows."""
    from nnslr_tools import brain_train
    trained, measured = [], []
    train, predict = brain_train.train_epoch, brain_train.predict

    def train_watched(model, optimizer, drives, *args, **options):
        trained.append([drive['inputs']['hour'].copy() for drive in drives])
        return train(model, optimizer, drives, *args, **options)

    def predict_watched(model, drive, *args, **options):
        measured.append(drive['extra'].copy())
        return predict(model, drive, *args, **options)

    monkeypatch.setattr(brain_train, 'train_epoch', train_watched)
    monkeypatch.setattr(brain_train, 'predict', predict_watched)
    return trained, measured


def test_the_training_moves_the_clock_every_epoch_and_the_measure_has_the_real_one(tmp_path, monkeypatch):
    import numpy
    from nnslr_tools.brain_targets import BRAIN_INPUTS, pack_inputs
    torch = torch_or_skip()
    trained, measured = watched(monkeypatch)
    run_training(torch, tmp_path, monkeypatch, [5.0, 4.0], at_two_in_the_afternoon)
    real = BRAIN_INPUTS.hour_row(14.0)
    assert [len(epoch) for epoch in trained] == [1, 1]
    assert not numpy.allclose(trained[0][0][0], real) and not numpy.allclose(trained[0][0][0], trained[1][0][0])
    drive = fake_drive(['train'] * T)
    at_two_in_the_afternoon(drive)
    assert len(measured) >= 4 and all(numpy.array_equal(extra, pack_inputs(drive['inputs'], GROUPS)) for extra in measured)


def test_a_night_copy_goes_through_the_training_and_not_through_the_measure(tmp_path, monkeypatch):
    import numpy
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    torch = torch_or_skip()
    trained, measured = watched(monkeypatch)
    run_training(torch, tmp_path, monkeypatch, [5.0, 4.0], plate_of_hours)
    real = BRAIN_INPUTS.hour_row(14.0)
    assert [len(epoch) for epoch in trained] == [2, 2]
    for day, night in trained:
        assert numpy.allclose(day[0], real) and not numpy.allclose(night[0], real)
    assert len(measured) == 4


def test_the_training_stops_when_validation_gets_no_better(tmp_path, monkeypatch):
    torch = torch_or_skip()
    result, _saved = run_training(torch, tmp_path, monkeypatch, [1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert result['best_epoch'] == 0 and len(result['history']) == 3


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
