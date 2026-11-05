"""The step of the brain as the car runs it, on an untrained network. Skipped in .venv: run it as test_brain_net.py says."""
import json

import pytest

pytest.importorskip('torch')
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from nnslr_tools import brain_export  # noqa: E402
from nnslr_tools.brain_targets import inputs_sha256  # noqa: E402

GROUPS = ('signs', 'plate', 'hour', 'lead', 'country')


def checkpoint(path, monkeypatch, **changes):
    """load_brain giving an untrained network, and what a checkpoint says of itself."""
    from nnslr_tools.brain_net import build_brain_net
    model = build_brain_net(GROUPS, works=False, pretrained=False).eval()
    saved = {'groups': GROUPS, 'hidden': model.gru.hidden_size, 'inputs_sha256': inputs_sha256(), 'epoch': 0, 'split': 'x',
             'reader': None, **changes}
    path.write_bytes(b'weights')
    monkeypatch.setattr(brain_export, 'load_brain', lambda _: (model, saved))
    return path


def test_refuses_a_network_trained_with_another_file_of_the_inputs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, inputs_sha256='0' * 64)
    with pytest.raises(ValueError, match='train it again'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')


def test_refuses_a_network_with_the_driver_among_its_inputs(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch, groups=(*GROUPS, 'pedals'))
    with pytest.raises(ValueError, match='does not give'):
        brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')


def test_the_contract_says_what_the_car_reads(tmp_path, monkeypatch):
    path = checkpoint(tmp_path / 'brain-last.pt', monkeypatch)
    done = brain_export.export_brain_onnx(path, tmp_path / 'brain.onnx')
    contract = json.loads((tmp_path / 'brain.onnx.contract.json').read_text())
    assert done['verification']['passed']
    assert contract['inputs_sha256'] == inputs_sha256()
    assert contract['step']['groups'] == list(GROUPS) and contract['step']['rate_hz'] == 1.0
    assert contract['output']['kph'][:2] == [5, 10]
