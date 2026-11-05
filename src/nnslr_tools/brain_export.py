"""The brain as the car runs it: one step a second as ONNX with its contract (<output>.contract.json).

The step: image (1, 3, 144, 256) 0..1, extra (1, inputs) the groups of the checkpoint side by side (brain_inputs.pack_inputs)
and hidden (1, hidden), zeros at the start -> speed (1, SPEED_CLASSES) logits and hidden_out, given back as hidden at the next
step. The speed said is contract['output']['kph'][argmax(speed)]; no softmax, nothing is read from how sure it is. The GRU is
written out with matrix products as the road-type network's (gru_step): no GRU node in the graph the comma compiles.

works_head stays out (brain_net.car_weights). What comes out of the export is checked twice: against PyTorch with onnxruntime
on random inputs, and along a whole real drive against what training measured (brain_train.predict), the memory fed back as
on the car. The contract keeps the hash of the car's file of the inputs: the car refuses a network made with another.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools.brain_net import build_brain_net, car_weights
from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES, SPEED_CLASSES, class_kph, dataset, inputs_sha256, pack_inputs
from nnslr_tools.brain_train import CHECKPOINT, CHUNK, _pictures, predict
from nnslr_tools.road_type_net import FrameStore, _torch, gru_step

BRAIN_FILE = 'brain.onnx'
INPUT_NAMES = ['image', 'extra', 'hidden']
OUTPUT_NAMES = ['speed', 'hidden_out']
RTOL, ATOL = 1e-4, 1e-4


def load_brain(checkpoint: Path):
    """(the network on the CPU in eval mode, without works_head; what the checkpoint says of itself)."""
    torch = _torch()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_brain_net(tuple(saved['groups']), pool=saved['pool'], works=False, pretrained=False,
                            features=saved['features'], hidden=saved['hidden'])
    weights = {name: value for name, value in saved['weights'].items() if name in car_weights(model)}
    model.load_state_dict(weights, strict=True)
    return model.eval(), saved


def step_module(model):
    """The second of the car: (image, extra, hidden) -> (speed logits, hidden_out), the numbers of BrainNet.forward."""
    torch = _torch()
    from torch import nn

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, image, extra, hidden):
            x = torch.cat([self.model.encode(image), extra], dim=-1)
            hidden_out = gru_step(self.model.gru, x, hidden)
            speed = self.model.speed_head(hidden_out) + self.model.trust(hidden_out) * self.model.number_in_force(extra)
            return speed, hidden_out

    return Step().eval()


def _random_inputs(torch, width: int, hidden: int, generator) -> tuple:
    return (torch.rand(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH, generator=generator),
            torch.rand(1, width, generator=generator), torch.rand(1, hidden, generator=generator) * 2 - 1)


def drive_rollout(model, step_run, drive: dict, store: FrameStore, device: str = 'cpu') -> dict:
    """A whole drive by step_run(image, extra, hidden) -> (logits, hidden_out), numpy, against brain_train.predict: the frames
    and how many of them say another speed. The memory starts again where the car's does (drive['cuts'])."""
    expected = predict(model.to(device), drive, store, chunk=CHUNK, device=device)['kph']
    hidden = np.zeros((1, model.gru.hidden_size), dtype=np.float32)
    said = np.zeros(len(expected), dtype=np.int64)
    for frame in range(len(expected)):
        if frame in drive['cuts']:
            hidden = np.zeros_like(hidden)
        image = store.batch([(drive['route'], [frame])])[0].cpu().numpy()
        logits, hidden = step_run(image, drive['extra'][frame:frame + 1], hidden)
        said[frame] = int(class_kph(int(np.argmax(logits))))
    other = int((said != expected).sum())
    return {'route': drive['route'], 'frames': len(expected), 'other_speed': other, 'said': sorted({int(v) for v in said})}


def rollout_drive(root: Path, merged_of, groups: tuple[str, ...], route: str | None = None,
                  device: str = 'cpu') -> tuple[dict, FrameStore]:
    """A drive of the labels, with its pictures: route, or the one with the most frames in test."""
    drives = dataset(root, merged_of, [route] if route else None)
    drive = max(drives, key=lambda item: sum(partition == 'test' for partition in item['partitions']))
    drive['extra'] = pack_inputs(drive['inputs'], groups)
    return drive, FrameStore(_pictures(root, [drive], dry_run=False, log=lambda text: None), device)


def _export(step, example: tuple, output: Path) -> None:
    torch = _torch()
    try:
        program = torch.onnx.export(step, example, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamo=True,
                                    external_data=False)
        program.save(str(output))
    except ImportError as exc:
        raise ValueError('onnx_dependencies_missing: install onnx and onnxscript in the training environment') from exc
    except Exception as exc:  # torch.export refuses the graph: the TorchScript exporter
        print(f'torch.export could not export the brain step ({type(exc).__name__}); using TorchScript', flush=True)
        torch.onnx.export(step, example, str(output), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamo=False,
                          opset_version=17)


def export_brain_onnx(checkpoint: Path, output: Path, *, data_root: Path | None = None, merged_of=None,
                      route: str | None = None) -> dict:
    """The step and its contract, checked: raises when it differs from PyTorch or when the car's file of the inputs is not the
    one the network was trained with. data_root and merged_of add the check along a real drive."""
    torch = _torch()
    import onnx
    import onnxruntime

    from nnslr_tools.exporting import sha256_file

    checkpoint, output = Path(checkpoint), Path(output)
    model, saved = load_brain(checkpoint)
    if saved['inputs_sha256'] not in (inputs_sha256(), *BRAIN_INPUTS.SAME_GROUPS):
        raise ValueError('the car\'s brain_inputs.py is not the one this network was trained with: train it again')
    groups = tuple(saved['groups'])
    if not set(groups) <= set(BRAIN_INPUTS.INPUT_SIZES):
        raise ValueError(f'this network has inputs the car does not give: {groups}')
    width, hidden = sum(INPUT_SIZES[name] for name in groups), saved['hidden']
    step = step_module(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    _export(step, _random_inputs(torch, width, hidden, torch.Generator().manual_seed(1)), output)
    onnx.checker.check_model(onnx.load(str(output)))
    session = onnxruntime.InferenceSession(str(output), providers=['CPUExecutionProvider'])

    def run(image, extra, memory):
        return session.run(None, {'image': image, 'extra': extra, 'hidden': memory})

    generator, worst, passed = torch.Generator().manual_seed(0), 0.0, True
    for _ in range(4):
        inputs = _random_inputs(torch, width, hidden, generator)
        with torch.no_grad():
            expected = [value.numpy() for value in step(*inputs)]
        for wanted, got in zip(expected, run(*(value.numpy() for value in inputs))):
            worst = max(worst, float(np.abs(wanted - got).max()))
            passed = passed and bool(np.allclose(got, wanted, rtol=RTOL, atol=ATOL))
    verification = {'passed': passed, 'max_abs_error': worst, 'rtol': RTOL, 'atol': ATOL, 'provider': 'CPUExecutionProvider'}
    if not passed:
        raise ValueError(f'brain ONNX differs from PyTorch: max abs error {worst}')
    if data_root is not None:
        drive, store = rollout_drive(Path(data_root), merged_of, groups, route)
        verification['rollout'] = drive_rollout(model, run, drive, store)
        if verification['rollout']['other_speed']:
            raise ValueError(f'brain ONNX says another speed than PyTorch along a drive: {verification["rollout"]}')
    shapes = {'image': [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH], 'extra': [1, width], 'hidden': [1, hidden]}
    contract = {
        'schema_version': 1, 'task': 'brain', 'architecture': 'mobilenet_v3_small+gru', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint), 'onnx_sha256': sha256_file(output), 'split_sha256': saved['split'],
        'epoch': saved['epoch'], 'inputs_sha256': saved['inputs_sha256'], 'reader': saved['reader'],
        'input': {'name': 'image', 'shape': shapes['image'], 'dtype': 'float32', 'layout': 'NCHW', 'color': 'RGB', 'range': '0..1',
                  'normalization': 'none: ImageNet mean and std are applied inside the network',
                  'resize': {'method': 'PIL BILINEAR (antialiased), vision.resize', 'from': list(data.SOURCE_SIZE),
                             'to': [data.FRAME_WIDTH, data.FRAME_HEIGHT], 'crop': 'none, the whole frame'},
                  'source': 'narrow road camera NV12 -> vision.yuv_to_rgb -> vision.resize'},
        'inputs': [{'name': name, 'shape': shapes[name], 'dtype': 'float32'} for name in INPUT_NAMES],
        'output': {'name': 'speed', 'shape': [1, SPEED_CLASSES], 'kph': [int(class_kph(index)) for index in range(SPEED_CLASSES)],
                   'decode': 'kph[argmax(speed)]'},
        'outputs': [{'name': 'speed', 'shape': [1, SPEED_CLASSES], 'dtype': 'float32'},
                    {'name': 'hidden_out', 'shape': [1, hidden], 'dtype': 'float32', 'meaning': 'give it back as hidden'}],
        'step': {'rate_hz': 1.0, 'groups': list(groups), 'memory_gap_seconds': BRAIN_INPUTS.MEMORY_GAP_SECONDS,
                 'inputs': INPUT_NAMES, 'outputs': OUTPUT_NAMES, 'plates': bool(saved.get('plates', False)),
                 'hours': bool(saved.get('hours', False))},
        'verification': verification,
    }
    contract_path = output.with_name(output.name + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'onnx': str(output), 'contract': str(contract_path), 'onnx_sha256': contract['onnx_sha256'],
            'verification': verification}
