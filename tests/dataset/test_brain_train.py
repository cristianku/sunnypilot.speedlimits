"""The training loop on a tiny drive, and the measure by hand. The measure needs only numpy: the rest is skipped without
torch (see test_brain_net.py for how to run it)."""
import json
import random
import re

import pytest

NS = 1_000_000_000
T = 8
# The groups of the tiny brains of these tests: those of a brain of 8 Nov 2026 with the driver's inputs (pedals, speed), which
# no brain learns since 10 Nov (brain_targets.LEFT_OUT); a brain of before keeps packing what it took.
GROUPS = ('signs', 'plate', 'plate_hours', 'hour', 'lead', 'country', 'pedals', 'speed', 'plate_wet', 'sign_distance')


def fake_drive(partitions, *, speed=None, works=None, marks=(), route='00000001--abcdef0123'):
    """A drive of brain_targets.drive_targets: his speed 100 km/h (speed: a class) everywhere, unless speed says another."""
    import numpy
    from nnslr_tools.brain_targets import INPUT_SIZES, pack_inputs, speed_class
    speed = speed_class(100) if speed is None else speed
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


def test_a_speed_within_2_km_h_of_his_is_right():
    from nnslr_tools.brain_train import RIGHT_KPH, measure
    drive = fake_drive(['test'] * 5)
    found = measure([drive], [said([100, 102, 98, 103, 97])], 'test')['speed']
    assert RIGHT_KPH == 2
    assert found == {'frames': 5, 'right': pytest.approx(3 / 5), 'mean_error_kph': pytest.approx(10 / 5)}


def test_inside_a_block_the_network_is_measured_against_the_block_and_against_the_car():
    import numpy
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 6)
    drive['speed'][:4] = speed_class(80)
    drive['own_kph'][:] = 100, 100, 81, numpy.nan, 100, 100
    drive['blocked'] = numpy.array([True, True, True, True, False, False])
    found = measure([drive], [said([101, 80, 81, 80, 100, 100])], 'test')['in_blocks']
    assert found == {'frames': 4, 'right': pytest.approx(3 / 4), 'mean_error_kph': pytest.approx(22 / 4),
                     'copy_own_speed': pytest.approx(1 / 2)}


def test_with_no_block_apart_from_the_car_there_is_no_copy_to_count():
    import numpy
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 2)
    drive['own_kph'][:] = 101
    drive['blocked'] = numpy.ones(2, dtype=bool)
    assert measure([drive], [said([100, 100])], 'test')['in_blocks']['copy_own_speed'] is None
    assert measure([fake_drive(['test'] * 2)], [said([100, 100])], 'test')['in_blocks'] == {
        'frames': 0, 'right': None, 'mean_error_kph': None, 'copy_own_speed': None}


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
    from nnslr_tools.brain_targets import speed_class
    drive = fake_drive(['test'] * 4, speed=speed_class(120))
    drive['own_kph'][:] = 90, 90, 115, numpy.nan
    found = measure([drive], [said([90, 120, 115, 120])], 'test')
    assert found['speed']['frames'] == 4
    assert found['far_from_own_speed'] == {'frames': 2, 'right': 0.5, 'mean_error_kph': 15.0, 'nearer_own_speed': 0.5}


def test_with_the_car_near_the_label_the_measure_is_empty():
    from nnslr_tools.brain_train import measure
    from nnslr_tools.brain_targets import speed_class
    drive = fake_drive(['test'] * 2, speed=speed_class(120))
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


def test_the_speed_after_works_is_the_one_he_reaches_not_the_first_on_the_way_up():
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 20, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS}])
    drive['speed'][:6] = speed_class(60)
    drive['speed'][6:] = [speed_class(kph) for kph in (63, 70, 80, 90, 100, 110, 118, 119, 120, 120, 120, 120, 120, 120)]
    kph = [60] * 6 + [64, 70, 80, 90, 100, 110, 118, 119, 120, 120, 120, 120, 120, 120]
    found = measure([drive], [said(kph)], 'test')['sites'][0]
    assert found['end_delay_s'] == pytest.approx(6)
    assert measure([drive], [said([60] * 6 + [64] * 14)], 'test')['sites'][0]['end_delay_s'] is None


def test_a_site_whose_speed_after_is_his_inside_within_2_km_h_has_no_end():
    from nnslr_tools.brain_targets import speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 10, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS}])
    drive['speed'][:6] = speed_class(80)
    drive['speed'][6:] = speed_class(82)
    assert measure([drive], [said([80] * 6 + [82] * 4)], 'test')['sites'][0]['end_delay_s'] is None


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


def test_the_log_says_how_many_speeds_and_from_where_to_where():
    from nnslr_tools.brain_train import _spoken
    assert _spoken([52, 60, 61, 118]) == '4 speeds, 52-118 km/h'
    assert _spoken([100]) == '1 speed, 100-100 km/h'
    assert _spoken([]) == 'nothing'


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


def roads(drive, road, weight=None):
    """The drive with two kinds of road, narrow and gravel: road their answers per frame, weight 1 everywhere unless said."""
    import numpy
    drive['road_kinds'] = ['narrow', 'gravel']
    drive['road'] = numpy.array(road, dtype=numpy.float32)
    drive['road_weight'] = numpy.ones_like(drive['road']) if weight is None else numpy.array(weight, dtype=numpy.float32)
    return drive


def test_a_kind_of_road_with_one_side_alone_leaves_the_loss_of_train_and_the_other_stays():
    from nnslr_tools.brain_train import _road_needs_both
    said_to = []
    drive = roads(fake_drive(['train', 'train', 'test']), [[1, 1], [0, 1], [1, 1]])
    _road_needs_both([drive, fake_drive(['train'])], said_to.append)
    assert drive['road_weight'].tolist() == [[1, 0], [1, 0], [1, 1]]
    assert 'Road "narrow" in train: 1 frames inside and 1 outside.' in said_to
    assert any('"gravel"' in line and 'checked to here' in line for line in said_to)


def test_each_kind_of_road_is_measured_on_its_own_answers_and_the_speed_inside_its_marks():
    import numpy
    from nnslr_tools.brain_train import measure
    drive = roads(fake_drive(['test'] * 4), [[1, 0], [1, 0], [0, 0], [0, 1]], [[1, 1], [1, 1], [1, 1], [0, 1]])
    answer = said([100, 90, 100, 100])
    answer['road'] = numpy.array([[.9, .9], [.1, .1], [.9, .1], [.9, .9]], dtype=numpy.float32)
    found = measure([drive], [answer], 'test')
    narrow, gravel = found['road']['narrow'], found['road']['gravel']
    assert (narrow['true_positive'], narrow['false_positive'], narrow['false_negative']) == (1, 1, 1)
    assert (gravel['true_positive'], gravel['false_positive'], gravel['false_negative']) == (1, 1, 0)
    assert found['in_road']['narrow']['frames'] == 2 and found['in_road']['gravel']['frames'] == 1
    # A network without the road head: the speed inside each kind is still measured, no precision.
    found = measure([drive], [said([100] * 4)], 'test')
    assert found['road']['narrow']['precision'] is None and found['in_road']['narrow']['frames'] == 2


def typed(drive, types, by_person=None):
    """The drive with a road type per frame (a ROAD_TYPES name, None: none), as drive_targets(types=True) leaves it."""
    import numpy
    from nnslr_tools.brain_targets import NO_TYPE
    from nnslr_tools.road_types import ROAD_TYPES
    drive['road_type'] = numpy.array([-1 if kind is None else ROAD_TYPES.index(kind) for kind in types], dtype=numpy.int64)
    drive['type_weight'] = (drive['road_type'] >= 0).astype(numpy.float32)
    drive['typed_by_person'] = numpy.zeros(len(types), dtype=bool) if by_person is None else numpy.array(by_person)
    drive['speed_out'] = numpy.where(drive['road_type'] >= 0, 0, NO_TYPE)
    drive['speed_weight'] = (drive['speed_out'] == 0).astype(numpy.float32)
    drive['types'] = True
    return drive


def test_the_road_type_is_measured_on_the_frames_with_one_and_the_speed_on_each_type():
    import numpy
    from nnslr_tools.brain_train import measure
    from nnslr_tools.road_types import ROAD_TYPES
    drive = typed(fake_drive(['test'] * 4 + ['train']), ['town', 'town', 'motorway', None, 'town'])
    answer = said([100, 90, 100, 100, 100])
    answer['types'] = numpy.eye(len(ROAD_TYPES), dtype=numpy.float32)[[ROAD_TYPES.index(kind) for kind in
                                                                         ('town', 'rural', 'motorway', 'town', 'town')]]
    found = measure([drive], [answer], 'test')
    assert found['types']['frames'] == 3 and found['types']['right'] == pytest.approx(2 / 3)
    assert found['types']['per_type']['town'] == {'frames': 2, 'right': .5}
    assert found['types']['per_type']['motorway'] == {'frames': 1, 'right': 1.0}
    assert set(found['in_type']) == {'town', 'motorway'} and found['in_type']['town']['frames'] == 2
    # A network without the type head: no measure of the type, the speed on each type still.
    found = measure([drive], [said([100] * 5)], 'test')
    assert found['types'] is None and found['in_type']['motorway']['frames'] == 1


def test_the_log_says_the_seconds_with_a_road_type_and_the_drives_with_none():
    from nnslr_tools.brain_train import _types_in_train
    lines = []
    one = typed(fake_drive(['train'] * 4 + ['test']), ['town', None, 'rural', 'rural', 'town'])
    bare = typed(fake_drive(['train'] * 2, route='00000002--abcdef0123'), [None, None])
    _types_in_train([one, bare, fake_drive(['train'])], lines.append)
    assert lines[0] == 'Road types in train: 3 seconds with one; 3 seconds with a speed and no road type, left out.'
    assert '00000002--abcdef0123' in lines[1] and 'Read the map in Data › Prepare' in lines[1]
    lines.clear()
    _types_in_train([fake_drive(['train'])], lines.append)
    assert lines == []


def test_a_site_of_another_partition_is_not_here():
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['train'] * 10, marks=[{'start_ns': 2 * NS, 'end_ns': 6 * NS}])
    assert measure([drive], [said([100] * 10)], 'test')['sites'] == []


def test_the_limit_ahead_is_right_on_its_class_and_false_where_none_is_ahead_and_it_says_one():
    import numpy
    from nnslr_tools.brain_targets import NONE_AHEAD, speed_class
    from nnslr_tools.brain_train import measure
    drive = fake_drive(['test'] * 6)
    drive['next'] = numpy.array([speed_class(80)] * 3 + [NONE_AHEAD] * 3)
    drive['next_weight'] = numpy.ones(6, dtype=numpy.float32)
    drive['metres'] = numpy.array([.5, .4, .3, 0, 0, 0], dtype=numpy.float32)
    drive['metres_weight'] = numpy.array([1, 1, 1, 0, 0, 0], dtype=numpy.float32)
    answer = {**said([100] * 6), 'next': numpy.array([speed_class(80), speed_class(80), speed_class(50), NONE_AHEAD,
                                                      NONE_AHEAD, speed_class(100)]),
              'metres': numpy.array([50.0, 30.0, 0, 0, 0, 0]), 'none': NONE_AHEAD}
    found = measure([drive], [answer], 'test')['ahead']
    assert found['frames'] == 6 and found['right'] == pytest.approx(4 / 6)
    assert found['changes'] == 3 and found['changes_right'] == pytest.approx(2 / 3)
    assert found['false_ahead'] == pytest.approx(1 / 3)


def test_the_changes_read_are_those_whose_number_the_signs_input_already_holds():
    import numpy
    from nnslr_tools.brain_targets import INPUT_SIZES, NONE_AHEAD, speed_class
    from nnslr_tools.brain_train import SIGN_KPH_SCALE, measure
    drive = fake_drive(['test'] * 4)
    drive['next'] = numpy.array([speed_class(80)] * 3 + [NONE_AHEAD])
    drive['next_weight'] = numpy.ones(4, dtype=numpy.float32)
    drive['metres'] = drive['metres_weight'] = numpy.zeros(4, dtype=numpy.float32)
    signs = numpy.zeros((4, INPUT_SIZES['signs']), dtype=numpy.float32)
    signs[0, :2] = 80 / SIGN_KPH_SCALE, 1  # confirmed
    signs[1, 8:10] = 80 / SIGN_KPH_SCALE, 1  # read once
    signs[2, :2] = 50 / SIGN_KPH_SCALE, 1  # still the one in force
    drive['inputs']['signs'] = signs
    answer = {**said([100] * 4), 'next': numpy.array([speed_class(80), NONE_AHEAD, NONE_AHEAD, NONE_AHEAD]),
              'metres': numpy.zeros(4), 'none': NONE_AHEAD}
    found = measure([drive], [answer], 'test')['ahead']
    assert found['changes'] == 3 and found['changes_read'] == 2 and found['changes_read_right'] == pytest.approx(1 / 2)


def test_a_number_ahead_weighs_as_much_all_together_as_none():
    import numpy
    from nnslr_tools.brain_targets import NONE_AHEAD, speed_class
    from nnslr_tools.brain_train import _balanced, next_balance
    drive = fake_drive(['train'] * 10)
    drive['next'] = numpy.array([speed_class(80)] * 2 + [NONE_AHEAD] * 8)
    drive['next_weight'] = numpy.ones(10, dtype=numpy.float32)
    drive['metres_weight'] = numpy.ones(10, dtype=numpy.float32)
    balance = next_balance([drive])
    assert balance == pytest.approx(4)
    weights = _balanced((drive['next_weight'], drive['metres_weight']), drive, balance)[0]
    assert weights[:2].sum() == pytest.approx(weights[2:].sum()) and weights.sum() == pytest.approx(8)
    assert next_balance([fake_drive(['train'] * 3)]) == 1.0


def test_no_frames_is_no_measure():
    from nnslr_tools.brain_train import measure
    assert measure([], [], 'test')['speed'] == {'frames': 0, 'right': None, 'mean_error_kph': None}


def torch_or_skip():
    pytest.importorskip('torchvision')
    return pytest.importorskip('torch')


def tiny(torch, works=True, groups=GROUPS, **options):
    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(groups, pretrained=False, works=works, **options)
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


def test_a_kind_of_road_alone_is_a_step_of_its_head_and_the_prediction_says_each_kind():
    torch = torch_or_skip()
    from nnslr_tools.brain_train import predict
    from nnslr_tools.road_type_net import FrameStore
    drive = roads(fake_drive(['train'] * T), [[1, 0]] * 4 + [[0, 0]] * 4)
    drive['speed_weight'][:], drive['works_weight'][:] = 0, 0
    model = tiny(torch, road_kinds=2)
    before = {name: value.clone() for name, value in model.named_parameters()}
    array = pictures(drive)
    assert one_epoch(torch, model, drive, array) > 0
    assert not torch.equal(before['road_head.weight'], model.road_head.weight)
    said = predict(model, drive, FrameStore({drive['route']: array}, torch.device('cpu')), chunk=3, device=torch.device('cpu'))
    assert said['road'].shape == (T, 2) and ((said['road'] >= 0) & (said['road'] <= 1)).all()
    # A network without the head: no road said.
    assert predict(tiny(torch), drive, FrameStore({drive['route']: array}, torch.device('cpu')), chunk=3,
                   device=torch.device('cpu'))['road'] is None


def test_a_road_type_alone_is_a_step_of_its_head_and_the_prediction_says_the_chance_of_each():
    torch = torch_or_skip()
    from nnslr_tools.brain_train import predict
    from nnslr_tools.road_type_net import FrameStore
    from nnslr_tools.road_types import ROAD_TYPES
    drive = typed(fake_drive(['train'] * T), ['town'] * 4 + ['motorway'] * 3 + [None])
    drive['speed_weight'][:], drive['works_weight'][:] = 0, 0
    model = tiny(torch, road_types=len(ROAD_TYPES))
    before = {name: value.clone() for name, value in model.named_parameters()}
    array = pictures(drive)
    assert one_epoch(torch, model, drive, array) > 0
    assert not torch.equal(before['type_head.weight'], model.type_head.weight)
    said = predict(model, drive, FrameStore({drive['route']: array}, torch.device('cpu')), chunk=3, device=torch.device('cpu'))
    assert said['types'].shape == (T, len(ROAD_TYPES)) and numpy_sum_is_one(said['types'])
    assert predict(tiny(torch), drive, FrameStore({drive['route']: array}, torch.device('cpu')), chunk=3,
                   device=torch.device('cpu'))['types'] is None


def numpy_sum_is_one(chances):
    return bool((abs(chances.sum(axis=-1) - 1) < 1e-4).all())


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
    """Every epoch's validation scores what scores says: nothing may follow from it."""
    from nnslr_tools import brain_train
    drive = fake_drive(['train'] * 4 + ['validation'] * 2 + ['test'] * 2)
    change(drive)
    drive['reader'], drive['reader_current'], drive['signs_read'] = {'vision': 'abc'}, True, True
    monkeypatch.setattr(brain_train, 'dataset', lambda *_args, **_options: [drive])
    monkeypatch.setattr(brain_train, 'read_split', lambda _root: {'sha256': 'split'})
    monkeypatch.setattr(brain_train, '_pictures', lambda *_args, **_options: {drive['route']: pictures(drive)})
    model = tiny(torch, groups=brain_train.LEARNED_INPUTS)
    monkeypatch.setattr(brain_train, 'build_brain_net', lambda *_args, **_options: model)
    real, left = brain_train.measure, list(scores)

    def scripted(drives, answers, partition):
        found = real(drives, answers, partition)
        if partition == 'validation' and left:
            found['speed']['mean_error_kph'] = left.pop(0)
        return found

    monkeypatch.setattr(brain_train, 'measure', scripted)
    result = brain_train.train_brain(tmp_path, tmp_path / 'run', None, epochs=len(scores), chunk=4, device='cpu',
                                     log=lambda _text: None)
    return result, torch.load(tmp_path / 'run' / 'brain-last.pt', weights_only=False), model


def test_the_epoch_kept_is_the_last_whatever_validation_says_and_the_checkpoint_says_what_it_came_from(tmp_path, monkeypatch):
    torch = torch_or_skip()
    result, saved, model = run_training(torch, tmp_path, monkeypatch, [1.0, 9.0, 9.0, 9.0, 9.0])
    assert saved['epoch'] == 4 and len(result['history']) == 5
    assert all(torch.equal(value, saved['weights'][name]) for name, value in model.state_dict().items())
    assert saved['split'] == 'split' and saved['reader'] == {'vision': 'abc'}
    assert saved['features'] == 128 and saved['hidden'] == 128
    assert {'validation_cold', 'test_cold'} <= set(result)
    assert saved['weights']['class_kph'].tolist() == [float(kph) for kph in range(1, 201)]
    assert result['right_kph'] == 2 and 'in_blocks' in result['test']


def test_a_brain_never_learns_the_driver(tmp_path, monkeypatch):
    torch = torch_or_skip()
    from nnslr_tools.brain_train import LEARNED_INPUTS
    _result, saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0])
    assert tuple(saved['groups']) == LEARNED_INPUTS and not {'pedals', 'speed'} & set(saved['groups'])


def test_the_epoch_line_keeps_what_the_site_reads_and_says_the_speeds_in_short(tmp_path, monkeypatch):
    torch = torch_or_skip()
    from nnslr_tools import brain_train
    lines = []
    real = brain_train.train_brain

    def logged(*args, **options):
        return real(*args, **{**options, 'log': lines.append})

    monkeypatch.setattr(brain_train, 'train_brain', logged)
    run_training(torch, tmp_path, monkeypatch, [1.0])
    epoch = next(line for line in lines if ' epoch 1/1 ' in line)
    assert 'val_error_kph 1.00 ' in epoch and ' val_accuracy ' in epoch
    assert re.search(r' says (nothing|\d+ speeds?, \d+-\d+ km/h) \d+s$', epoch), epoch


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
    assert len(measured) >= 3 and all(numpy.array_equal(extra, pack_inputs(drive['inputs'], GROUPS)) for extra in measured)


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
    assert len(measured) == 3


def test_a_brain_says_whether_it_learnt_the_plates(tmp_path, monkeypatch):
    torch = torch_or_skip()
    result, saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0])
    assert result['plates'] is False and saved['plates'] is False
    result, saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0], plate_of_hours)
    assert result['plates'] is True and saved['plates'] is True


def test_a_brain_says_whether_it_learnt_the_hours(tmp_path, monkeypatch):
    torch = torch_or_skip()
    from nnslr_tools.brain_targets import BRAIN_INPUTS

    def plate_with_hours(drive):
        plate_of_hours(drive)
        drive['inputs']['plate_hours'][:] = [*BRAIN_INPUTS.hour_row(22)[:2], *BRAIN_INPUTS.hour_row(6)[:2], 1]

    result, saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0], plate_of_hours)
    assert result['hours'] is False and saved['hours'] is False
    result, saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0], plate_with_hours)
    assert result['hours'] is True and saved['hours'] is True


def test_the_result_has_the_roads_seen_beside_those_not_seen(tmp_path, monkeypatch):
    torch = torch_or_skip()
    result, _saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0])
    assert result['train']['speed']['frames'] == 4 and result['test']['speed']['frames'] == 2
    assert result['test_swapped']['network']['frames'] == 0 and result['validation_swapped']['moved'] == 0
    assert result['answers'] == {'00000001--abcdef0123': 8}


def test_the_result_counts_the_labelled_frames_of_each_drive(tmp_path, monkeypatch):
    torch = torch_or_skip()

    def one_frame_less(drive):
        drive['speed_weight'][0] = 0

    result, _saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0], one_frame_less)
    assert result['answers'] == {'00000001--abcdef0123': 7}


def signed_drive(partitions=('test',) * 7):
    """A number that its answers say, no number, and a number its answers do not say."""
    import numpy
    from nnslr_tools.brain_targets import BRAIN_INPUTS, pack_inputs, speed_class
    drive = fake_drive(list(partitions))
    kph = numpy.array([100, 100, 100, 0, 0, 60, 60])
    drive['inputs']['signs'][:, 0], drive['inputs']['signs'][:, 1] = kph / BRAIN_INPUTS.SIGN_KPH_SCALE, kph > 0
    drive['speed'] = numpy.array([speed_class(value) for value in [100, 100, 100, 90, 90, 70, 70]])
    drive['extra'] = pack_inputs(drive['inputs'], GROUPS)
    return drive


def swapped_measure(monkeypatch, answer, *drives, partition='test'):
    from nnslr_tools import brain_train
    monkeypatch.setattr(brain_train, 'predict', lambda _model, seen, _store, **_options: said(answer(seen)))
    before = [said(answer(drive)) for drive in drives]
    return brain_train._swapped_measure(None, list(drives), before, partition, GROUPS, None, chunk=4, device='cpu')


def test_test_swapped_asks_only_the_frames_whose_number_changed_and_the_network_must_change_with_it(monkeypatch):
    import numpy
    from nnslr_tools.brain_targets import sign_kph
    follows = swapped_measure(monkeypatch, lambda seen: sign_kph(seen['inputs']['signs']), signed_drive())
    assert follows['network'] == {'frames': 3, 'right': 1.0, 'mean_error_kph': 0.0} and follows['sign']['right'] == 1.0
    assert follows['moved'] == 3
    stays = swapped_measure(monkeypatch, lambda seen: numpy.full(7, 100), signed_drive())
    assert stays['network'] == {'frames': 3, 'right': 0.0, 'mean_error_kph': 20.0} and stays['moved'] == 0


def test_a_network_deaf_to_the_sign_can_be_right_on_a_swap_and_moves_nothing(monkeypatch):
    import numpy
    found = swapped_measure(monkeypatch, lambda seen: numpy.full(7, 120), signed_drive())
    assert found['network']['right'] == 1.0 and found['moved'] == 0


def test_a_passage_his_speed_keeps_within_5_km_h_of_the_number_is_swapped_with_his_answers(monkeypatch):
    import numpy
    from nnslr_tools.brain_targets import sign_kph, speed_class
    drive = signed_drive()
    drive['speed'][:3] = [speed_class(kph) for kph in (98, 102, 103)]
    follows = swapped_measure(monkeypatch, lambda seen: sign_kph(seen['inputs']['signs']), drive)
    assert follows['network'] == {'frames': 3, 'right': pytest.approx(2 / 3), 'mean_error_kph': pytest.approx(7 / 3)}
    assert follows['moved'] == 3
    drive['speed'][2] = speed_class(106)
    assert swapped_measure(monkeypatch, lambda seen: numpy.full(7, 100), drive)['network']['frames'] == 0


def test_an_answer_that_moves_2_km_h_has_not_moved(monkeypatch):
    import numpy
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    found = swapped_measure(monkeypatch, lambda seen: numpy.where(
        seen['inputs']['signs'][:, 0] * BRAIN_INPUTS.SIGN_KPH_SCALE > 110, 102, 100), signed_drive())
    assert found['moved'] == 0
    found = swapped_measure(monkeypatch, lambda seen: numpy.where(
        seen['inputs']['signs'][:, 0] * BRAIN_INPUTS.SIGN_KPH_SCALE > 110, 103, 100), signed_drive())
    assert found['moved'] == 3


@pytest.mark.parametrize('partitions, frames', [(['validation'] * 7, 3), (['validation'] * 2 + ['test'] * 5, 2)])
def test_the_swap_asks_the_frames_of_its_partition_alone(monkeypatch, partitions, frames):
    from nnslr_tools.brain_targets import sign_kph
    drive = signed_drive(partitions)
    found = swapped_measure(monkeypatch, lambda seen: sign_kph(seen['inputs']['signs']), drive, partition='validation')
    assert found['network']['frames'] == frames and found['moved'] == frames


def test_each_drive_is_compared_with_its_own_answers(monkeypatch):
    from nnslr_tools.brain_targets import BRAIN_INPUTS, sign_kph
    from nnslr_tools.brain_targets import speed_class
    plain = signed_drive()
    plain['speed'][:] = speed_class(85)
    plain['inputs']['signs'][:3, 0] = 120 / BRAIN_INPUTS.SIGN_KPH_SCALE
    found = swapped_measure(monkeypatch, lambda seen: sign_kph(seen['inputs']['signs']), plain, signed_drive())
    assert found['network']['frames'] == 3 and found['moved'] == 3


def test_a_frame_after_an_end_read_is_not_asked_to_move(monkeypatch):
    from nnslr_tools.brain_targets import pack_inputs, sign_kph
    drive = signed_drive()
    drive['inputs']['signs'][2, 6] = 1
    drive['extra'] = pack_inputs(drive['inputs'], GROUPS)
    found = swapped_measure(monkeypatch, lambda seen: sign_kph(seen['inputs']['signs']), drive)
    assert found['network']['frames'] == 2 and found['moved'] == 2


def test_the_swap_is_measured_on_validation_and_on_test_each_on_its_own(tmp_path, monkeypatch):
    torch = torch_or_skip()
    from nnslr_tools import brain_train
    real = brain_train.predict
    monkeypatch.setattr(brain_train, 'predict', lambda *args, cold=False, **options: {**real(*args, cold=cold, **options), 'cold': cold})
    monkeypatch.setattr(brain_train, '_swapped_measure', lambda *found, **_options: [found[3], found[2][0]['cold']])
    result, _saved, _model = run_training(torch, tmp_path, monkeypatch, [1.0])
    assert result['validation_swapped'] == ['validation', False] and result['test_swapped'] == ['test', False]


def test_test_swapped_with_no_passage_to_change_has_no_frames(monkeypatch):
    import numpy
    from nnslr_tools.brain_targets import speed_class
    drive = signed_drive()
    drive['speed'][:] = speed_class(85)
    assert swapped_measure(monkeypatch, lambda seen: numpy.zeros(7), drive)['network']['frames'] == 0


def test_the_network_goes_through_a_whole_drive_as_on_the_car():
    torch = torch_or_skip()
    from nnslr_tools.brain_train import predict
    from nnslr_tools.road_type_net import FrameStore
    drive = fake_drive(['test'] * T)
    store = FrameStore({drive['route']: pictures(drive)}, torch.device('cpu'))
    found = predict(tiny(torch), drive, store, chunk=3, device=torch.device('cpu'))
    assert found['kph'].shape == (T,) and found['works'].shape == (T,)
    assert found['kph'].min() >= 1 and found['kph'].max() <= 200
    assert predict(tiny(torch, works=False), drive, store, chunk=3, device=torch.device('cpu'))['works'] is None


def test_a_brain_of_before_says_its_speeds_in_its_own_table():
    import numpy
    torch = torch_or_skip()
    from nnslr_tools.brain_train import predict
    from nnslr_tools.road_type_net import FrameStore
    drive = fake_drive(['test'] * T)
    store = FrameStore({drive['route']: pictures(drive)}, torch.device('cpu'))
    model = tiny(torch, ahead=True, kph=list(range(5, 205, 5)))
    with torch.no_grad():
        model.speed_head.bias[9] = 100
        model.next_head.bias[40] = 100
    found = predict(model, drive, store, chunk=3, device=torch.device('cpu'))
    assert found['kph'].tolist() == [50] * T and (found['kph'] % 5 == 0).all()
    assert found['next'].tolist() == [40] * T and found['none'] == 40
    assert numpy.issubdtype(found['kph'].dtype, numpy.integer)


def test_no_brain_learns_engaged_the_pedals_or_the_speed():
    import inspect
    from nnslr_tools import brain_train
    assert set(brain_train.INPUT_SIZES) - set(brain_train.LEARNED_INPUTS) == {'engaged', 'pedals', 'speed'}
    assert 'driver' not in inspect.signature(brain_train.train_brain).parameters


def test_the_log_says_the_dashcam_drives_and_a_train_without_them_says_how_to_get_one(tmp_path, monkeypatch):
    """Every dashcam drive is learnt, each whole: the first line says how many, how long and how many had sunnypilot on,
    before the dry run keeps a piece of the first. Nothing is trained: it stops at the labels, before any network is
    built."""
    import types

    from nnslr_tools import brain_train
    monkeypatch.setattr(brain_train, '_torch', lambda: types.SimpleNamespace(manual_seed=lambda _seed: None,
                                                                             device=lambda name: name))
    counts = {'drives': 0, 'hours': 0.0, 'sunnypilot_on': 1}
    monkeypatch.setattr(brain_train, 'training_counts', lambda root: counts)
    monkeypatch.setattr(brain_train, 'dataset', lambda *_args, **_options: [])
    lines = []
    with pytest.raises(ValueError, match='no dashcam drive to learn from: drive with sunnylink'):
        brain_train.train_brain(tmp_path, tmp_path / 'run', None, dry_run=True, log=lines.append)
    # Dashcam drives, none of them usable (no frame, no pictures...): the lines of the dataset say why.
    counts.update(drives=2, hours=1.25)
    lines.clear()
    with pytest.raises(ValueError, match='none of the 2 dashcam drives gave the brain a drive to learn'):
        brain_train.train_brain(tmp_path, tmp_path / 'run', None, dry_run=True, log=lines.append)
    assert lines == ['Dashcam drives: 2, 1.2 h; 1 drive with sunnypilot on, left out whole.']
    # Dashcam drives, but none of them in train: the split put them in val and test, and the message says to drive more.
    monkeypatch.setattr(brain_train, 'dataset', lambda *_args, **_options: [{'route': 'r1', 'inputs': None}])
    monkeypatch.setattr(brain_train, '_short', lambda drive, _frames: drive)
    monkeypatch.setattr(brain_train, 'pack_inputs', lambda _inputs, _groups: None)
    monkeypatch.setattr(brain_train, 'loss_weights', lambda _drive, _part: (brain_train.np.zeros(4, dtype=bool),))
    with pytest.raises(ValueError, match='fall in val and test of the split; drive more'):
        brain_train.train_brain(tmp_path, tmp_path / 'run', None, dry_run=True, log=lines.append)


def test_the_dry_run_takes_the_first_drive_with_an_answer_in_train(tmp_path, monkeypatch):
    """A drive the frozen split put all in val or test comes first by route: the dry run passes it by and takes the next
    one with an answer in train, instead of saying the dashcam drives fall in val and test."""
    import types

    import numpy

    from nnslr_tools import brain_train
    monkeypatch.setattr(brain_train, '_torch', lambda: types.SimpleNamespace(manual_seed=lambda _seed: None,
                                                                             device=lambda name: name))
    monkeypatch.setattr(brain_train, 'training_counts',
                        lambda root: {'drives': 2, 'hours': 1.0, 'sunnypilot_on': 0})

    def drive(route, partition):
        return {'route': route, 'partitions': [partition] * 4, 'speed_weight': numpy.ones(4, dtype=numpy.float32),
                'works_weight': numpy.zeros(4, dtype=numpy.float32), 'inputs': None}

    asked = []

    def dataset(_root, _merged_of, routes, **_options):
        asked.append(routes)
        return [drive('a', 'validation'), drive('b', 'train')]

    monkeypatch.setattr(brain_train, 'dataset', dataset)
    taken = []

    class Taken(Exception):
        pass

    def short(drive, _frames):
        taken.append(drive['route'])
        raise Taken

    monkeypatch.setattr(brain_train, '_short', short)
    lines = []
    with pytest.raises(Taken):
        brain_train.train_brain(tmp_path, tmp_path / 'run', None, dry_run=True, routes=['a', 'b'], log=lines.append)
    assert taken == ['b'] and asked == [['a', 'b']]
    assert lines[1] == 'Only the 2 drives of the start of this run (--route): one added since waits for the next run.'
    # Every drive it takes, before the dry run's cut: what run_brain.sh keeps for its steps and seeds.
    assert json.loads((tmp_path / 'run' / brain_train.ROUTES).read_text()) == ['a', 'b']


def test_the_export_checks_the_brain_along_a_drive_it_learnt_from_with_its_pictures(tmp_path, monkeypatch):
    """Publish's check of the car's step along a real drive (brain_export.rollout_drive): with no route, a dashcam drive
    with its pictures (the ones run_brain.sh made); a route named may be any dashcam drive. A drive without the brain's
    pictures is refused: the check would run on black pictures."""
    from nnslr_tools import brain_export
    asked = []
    found = [{'route': 'r1', 'partitions': ['test'], 'inputs': None, 'pictures': True}]

    def dataset(_root, _merged_of, routes, *, rate, networks=None):
        asked.append(routes)
        return [dict(drive) for drive in found]

    monkeypatch.setattr(brain_export, 'dataset', dataset)
    monkeypatch.setattr(brain_export, 'pack_inputs', lambda _inputs, _groups: None)
    monkeypatch.setattr(brain_export, '_pictures', lambda *_args, **_options: None)
    monkeypatch.setattr(brain_export, 'FrameStore', lambda pictures, device: (pictures, device))
    assert brain_export.rollout_drive(tmp_path, None, ())[0]['route'] == 'r1'
    brain_export.rollout_drive(tmp_path, None, (), 'r1')
    assert asked == [None, ['r1']]
    # r1 driven after the training (more seconds in test, no pictures yet) is passed by: Publish still checks along r2.
    found[:] = [{'route': 'r1', 'partitions': ['test'] * 3, 'inputs': None, 'pictures': False},
                {'route': 'r2', 'partitions': ['test'], 'inputs': None, 'pictures': True}]
    assert brain_export.rollout_drive(tmp_path, None, ())[0]['route'] == 'r2'
    found[1]['pictures'] = False
    with pytest.raises(ValueError, match="pictures are not made. Run nnslr road-type-signs --labelled --route r1 --route r2"):
        brain_export.rollout_drive(tmp_path, None, ())
    found[:] = found[:1]
    with pytest.raises(ValueError, match="r1: the brain's pictures are not made"):
        brain_export.rollout_drive(tmp_path, None, (), 'r1')


# -- going on from an earlier brain (brain_start) -----------------------------------------------------------------
def start_file(torch, tmp_path, model, groups=GROUPS, **saved):
    """The brain-last.pt of an earlier run, as train_brain writes it."""
    from nnslr_tools.road_type_net import FEATURES, HIDDEN
    path = tmp_path / 'runs' / 'b1-s1' / 'brain-last.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'weights': model.state_dict(), 'epoch': 39, 'groups': list(groups), 'pool': 'grid', 'works': True,
                'road_kinds': ['tunnel'], 'road_types': [], 'features': FEATURES, 'hidden': HIDDEN, 'rate_hz': 4.0,
                'picture': 'brain', 'split': 'a' * 64, 'inputs_sha256': 'b' * 64, **saved}, path)
    return path


def source_of(groups, **changed):
    from nnslr_tools.road_type_net import FEATURES, HIDDEN
    return {'groups': tuple(groups), 'pool': 'grid', 'features': FEATURES, 'hidden': HIDDEN, 'rate_hz': 4.0,
            'picture': 'brain', 'road_kinds': ['tunnel'], 'road_types': [], 'split': 'a' * 64, 'inputs_sha256': 'b' * 64,
            **changed}


def said_by(torch, model, groups, drive, array):
    from nnslr_tools.brain_targets import pack_inputs
    model.eval()
    images = torch.from_numpy(array).permute(0, 3, 1, 2).float()[None] / 255
    with torch.no_grad():
        speed, _works, _state, following, _metres, road = model(
            images, torch.from_numpy(pack_inputs(drive['inputs'], groups))[None], ahead=True, road=True)
    return speed, following, road


def test_a_brain_going_on_from_another_says_what_it_said_without_the_inputs_it_drops_and_with_those_it_adds(tmp_path):
    from nnslr_tools.brain_train import brain_start
    torch = torch_or_skip()
    old = tiny(torch, ahead=True, road_kinds=1)
    for value in old.parameters():  # not the fresh numbers both start from
        value.data.add_(torch.randn_like(value) * 0.05)
    path = start_file(torch, tmp_path, old)
    # Without the driver's pedals and speed, with an input of its own the start never took.
    groups = tuple(name for name in GROUPS if name not in ('pedals', 'speed')) + ('engaged',)
    new = tiny(torch, groups=groups, ahead=True, road_kinds=1)
    start = brain_start(torch, new, path, source_of(groups))
    assert (start['run'], start['epoch'], start['split_sha256'], start['fresh']) == ('b1-s1', 39, 'a' * 64, [])
    assert (start['inputs_added'], start['inputs_left_out'], start['same_inputs_file']) == (['engaged'], ['pedals', 'speed'], True)
    drive = fake_drive(['train'] * T)
    for name in ('signs', 'lead', 'engaged'):
        drive['inputs'][name][:] = numpy_random(drive['inputs'][name].shape)
    array = pictures(drive)
    # The start with its dropped inputs at zero; the new one whatever its new input says.
    for got, wanted in zip(said_by(torch, new, groups, drive, array), said_by(torch, old, GROUPS, drive, array)):
        assert torch.allclose(got, wanted, atol=1e-5)


def numpy_random(shape):
    import numpy
    return numpy.random.default_rng(1).random(shape).astype(numpy.float32)


def test_the_speeds_go_on_by_km_h_and_a_speed_the_start_lacks_starts_fresh(tmp_path):
    from nnslr_tools.brain_train import brain_start
    torch = torch_or_skip()
    old = tiny(torch, ahead=True, kph=[10, 20, 30], road_kinds=1)
    path = start_file(torch, tmp_path, old)
    new = tiny(torch, ahead=True, kph=[20, 30, 40], road_kinds=1)
    start = brain_start(torch, new, path, source_of(GROUPS))
    assert sorted(start['fresh']) == ['next_head.bias (40)', 'next_head.weight (40)', 'speed_head.bias (40)',
                                      'speed_head.weight (40)']
    weights = old.state_dict()
    assert torch.equal(new.speed_head.weight[:2], weights['speed_head.weight'][1:])
    # The next head's last row is "none ahead", the start's own.
    assert torch.equal(new.next_head.weight[-1], weights['next_head.weight'][-1])
    assert new.class_kph.tolist() == [20, 30, 40]


@pytest.mark.parametrize('changed,words', [({'pool': 'mean'}, 'stepped otherwise'), ({'rate_hz': 1.0}, 'stepped otherwise'),
                                           ({'split': 'c' * 64}, 'another split')])
def test_a_start_stepped_otherwise_or_on_another_split_is_refused(tmp_path, changed, words):
    from nnslr_tools.brain_train import brain_start
    torch = torch_or_skip()
    path = start_file(torch, tmp_path, tiny(torch, ahead=True, road_kinds=1), **changed)
    with pytest.raises(ValueError, match=words):
        brain_start(torch, tiny(torch, ahead=True, road_kinds=1), path, source_of(GROUPS))


def test_a_file_that_is_not_a_brain_is_refused(tmp_path):
    from nnslr_tools.brain_train import brain_start
    torch = torch_or_skip()
    path = tmp_path / 'brain-last.pt'
    torch.save({'state_dict': {}}, path)
    with pytest.raises(ValueError, match='not a brain'):
        brain_start(torch, tiny(torch, ahead=True), path, source_of(GROUPS))
