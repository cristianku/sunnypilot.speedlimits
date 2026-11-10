"""The step of the brain as the car runs it, on an untrained network. Skipped in .venv: run it as test_brain_net.py says."""
import json

import pytest

pytest.importorskip('torch')
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from nnslr_tools import brain_export  # noqa: E402
from nnslr_tools.brain_targets import inputs_sha256  # noqa: E402

GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')


def checkpoint(path, monkeypatch, ahead=False, **changes):
    """load_brain giving an untrained network, and what a checkpoint says of itself."""
    from nnslr_tools.brain_net import build_brain_net
    model = build_brain_net(GROUPS, works=False, ahead=ahead, pretrained=False).eval()
    saved = {'groups': GROUPS, 'hidden': model.gru.hidden_size, 'inputs_sha256': inputs_sha256(), 'epoch': 0, 'split': 'x',
             'reader': None, **changes}
    path.write_bytes(b'weights')
    monkeypatch.setattr(brain_export, 'load_brain', lambda _: (model, saved))
    return path


def test_refuses_a_network_trained_with_another_file_of_the_inputs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, inputs_sha256='0' * 64)
    with pytest.raises(ValueError, match='train it again'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')


def test_refuses_a_network_with_an_input_the_car_does_not_give(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, groups=(*GROUPS, 'wheel'))
    with pytest.raises(ValueError, match='does not give'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')


def test_the_contract_says_what_the_car_reads(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch)
    done = brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert done['verification']['passed']
    assert contract['inputs_sha256'] == inputs_sha256()
    assert contract['step']['groups'] == list(GROUPS) and contract['step']['rate_hz'] == 1.0
    assert contract['output']['kph'] == list(range(1, 201)) and contract['output']['shape'] == [1, 200]


def test_the_limit_ahead_has_none_after_the_last_speed(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, ahead=True)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    shapes = {output['name']: output['shape'] for output in contract['outputs']}
    assert shapes['speed'] == [1, 200] and shapes['next'] == [1, 201] and shapes['metres'] == [1, 1]


def saved_brain(path, kph=None, ahead=True, road_kinds=()):
    """A checkpoint as brain_train writes it, works_head (and road_head with road_kinds) and its table of the speeds among
    the weights."""
    import torch

    from nnslr_tools.brain_net import build_brain_net
    torch.manual_seed(0)
    model = build_brain_net(GROUPS, pool='grid', works=True, ahead=ahead, pretrained=False, kph=kph,
                            road_kinds=len(road_kinds))
    torch.save({'weights': model.state_dict(), 'epoch': 0, 'groups': GROUPS, 'pool': 'grid', 'works': True,
                'road_kinds': list(road_kinds), 'features': 128, 'hidden': 128, 'ahead': ahead, 'rate_hz': 4.0,
                'picture': 'brain', 'split': 'x', 'reader': None, 'inputs_sha256': inputs_sha256()}, path)
    return model


def test_a_brain_that_learnt_the_kinds_of_road_goes_to_the_car_as_one_that_did_not(tmp_path):
    # road_head is only in training: the car's network, its weights and its contract are those of a brain without it.
    import torch

    from nnslr_tools.brain_net import car_weights
    with_road, without = tmp_path / 'road' / 'brain-last.pt', tmp_path / 'plain' / 'brain-last.pt'
    with_road.parent.mkdir()
    without.parent.mkdir()
    trained = saved_brain(with_road, road_kinds=('narrow', 'country', 'gravel', 'potholes'))
    assert trained.road_head is not None and 'road_head.weight' in torch.load(with_road, weights_only=False)['weights']
    saved_brain(without)
    model, _saved = brain_export.load_brain(with_road)
    plain, _ = brain_export.load_brain(without)
    assert model.road_head is None and model.works_head is None
    weights, kept = car_weights(model), car_weights(trained)
    assert set(weights) == set(car_weights(plain)) == set(kept)
    assert all(torch.equal(weights[name], kept[name]) for name in weights)
    contracts = []
    for path in (with_road, without):
        brain_export.export_brain_onnx(path, path.parent / 'brain.onnx')
        contract = json.loads((path.parent / 'brain.onnx.contract.json').read_text())
        contracts.append((contract['inputs'], contract['outputs'], contract['output']))
    assert contracts[0] == contracts[1]


def test_a_brain_of_5_km_h_classes_loads_with_its_own_table_and_says_it_to_the_car(tmp_path):
    import torch
    path = tmp_path / 'brain-last.pt'
    trained = saved_brain(path, kph=list(range(5, 205, 5)))
    model, saved = brain_export.load_brain(path)
    assert model.class_kph.tolist() == [float(kph) for kph in range(5, 205, 5)]
    assert (model.speed_head.out_features, model.next_head.out_features) == (40, 41)
    assert torch.equal(model.speed_head.weight, trained.speed_head.weight) and model.works_head is None
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert contract['output']['kph'] == list(range(5, 205, 5)) and contract['output']['shape'] == [1, 40]
    assert {output['name']: output['shape'] for output in contract['outputs']}['next'] == [1, 41]


def test_a_brain_of_today_loads_with_a_class_a_km_h(tmp_path):
    path = tmp_path / 'brain-last.pt'
    saved_brain(path)
    model, _saved = brain_export.load_brain(path)
    assert brain_export.table_of(model) == list(range(1, 201)) and model.next_head.out_features == 201


def test_along_a_drive_the_step_of_a_brain_of_before_says_its_own_km_h(tmp_path):
    import numpy

    from nnslr_tools.brain_targets import INPUT_SIZES
    from nnslr_tools.road_type_net import FrameStore
    path = tmp_path / 'brain-last.pt'
    saved_brain(path, kph=list(range(5, 205, 5)))
    model, _saved = brain_export.load_brain(path)
    step = brain_export.step_module(model)
    frames = 5
    pictures = numpy.random.default_rng(0).integers(0, 255, (frames, 144, 256, 3), dtype=numpy.uint8)
    width = sum(INPUT_SIZES[name] for name in GROUPS)
    drive = {'route': 'r', 'frames': [{}] * frames, 'partitions': ['test'] * frames, 'cuts': {3},
             'extra': numpy.random.default_rng(1).random((frames, width), dtype=numpy.float32)}

    def run(image, extra, memory):
        import torch
        with torch.no_grad():
            return [value.numpy() for value in step(*(torch.from_numpy(value) for value in (image, extra, memory)))]

    import torch
    threads, heard = torch.get_num_threads(), []
    found = brain_export.drive_rollout(model, run, drive, FrameStore({'r': pictures}, 'cpu'), log=heard.append)
    assert found['other_speed'] == 0 and found['other_ahead'] == 0 and found['frames'] == frames
    assert found['said'] and all(kph % 5 == 0 for kph in found['said'])
    # The log hears the drive and its progress; PyTorch has its threads back.
    assert heard[0] == 'brain check along r: 5 frames, one at a time as on the car'
    assert heard[1:] == [f'brain check: {n} of 5 frames' for n in range(1, 5)]
    assert torch.get_num_threads() == threads


def test_the_checks_onnxruntime_runs_on_one_thread_not_spinning(tmp_path):
    import onnxruntime
    path = tmp_path / 'brain-last.pt'
    saved_brain(path)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    session = brain_export.check_session(onnxruntime, tmp_path / 'brain.onnx')
    options = session.get_session_options()
    assert options.intra_op_num_threads == 1 and options.inter_op_num_threads == 1
    assert options.get_session_config_entry('session.intra_op.allow_spinning') == '0'


def test_a_network_of_an_earlier_file_with_the_same_groups_exports_with_its_own_hash(tmp_path, monkeypatch):
    from nnslr_tools.brain_targets import BRAIN_INPUTS
    earlier = BRAIN_INPUTS.SAME_GROUPS[0]
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, inputs_sha256=earlier)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    assert json.loads((tmp_path / 'brain.onnx.contract.json').read_text())['inputs_sha256'] == earlier


@pytest.mark.parametrize('saved, hours', [({}, False), ({'hours': True}, True)])
def test_the_contract_says_whether_the_car_gives_the_hours(tmp_path, monkeypatch, saved, hours):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, **saved)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    assert json.loads((tmp_path / 'brain.onnx.contract.json').read_text())['step']['hours'] is hours


@pytest.mark.parametrize('saved, plates', [({}, False), ({'plates': False}, False), ({'plates': True}, True)])
def test_the_contract_says_whether_the_car_gives_the_plates(tmp_path, monkeypatch, saved, plates):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, **saved)
    brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert contract['step']['plates'] is plates
