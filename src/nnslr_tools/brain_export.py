"""The brain as the car runs it: one step at each of its frames (the checkpoint's rate_hz: 1 for the first brains, every
reading of the car since 7 Nov 2026) as ONNX with its contract (<output>.contract.json).

The step: image (1, 3, 144, 256) 0..1, extra (1, inputs) the groups of the checkpoint side by side (brain_inputs.pack_inputs)
and hidden (1, hidden), zeros at the start -> speed (1, classes) logits and hidden_out, given back as hidden at the next
step. The speed said is contract['output']['kph'][argmax(speed)]; no softmax, nothing is read from how sure it is. The table
is the network's own (its class_kph, kept in the checkpoint): 200 classes of a km/h for the brains since 8 Nov 2026, 40 of 5
km/h for those before, which load and export as they were. The GRU is written out with matrix products as the road-type
network's (gru_step): no GRU node in the graph the comma compiles. A brain with the limit ahead (trained from 7 to 11 Nov
2026; none since, brain_net) also gives next (1, classes + 1), kph[argmax] or none for the last class, and metres (1,
1), the metres to it over output.metres_scale: the car no longer reads the two, and they stay in the export under the
same names and places. Such a brain that also learnt the road type (2026-11-11_2241-brain-s0) now exports with road_type
and output.road_types as well (below), so its file and contract are no longer the ones published on 11 Nov. The contract's
step says how the car steps it: rate_hz, clock ("schedule": on the grid of the rate, as the training's frames) and picture
("brain": vision.BrainInput; "road_type": the PIL picture).

works_head and road_head stay out (brain_net.car_weights). A brain that learnt the road type also gives road_type (1,
types), the logits of output.road_types: the car only shows it on the screen, the speed is the same with or without it. What comes out of the export is checked twice: against PyTorch
with onnxruntime on random inputs, and along a whole real drive against what training measured (brain_train.predict), the
memory fed back as on the car. The contract keeps the hash of the car's file of the inputs: the car refuses a network made with another.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nnslr_tools import road_type_data as data
from nnslr_tools import onnx_device
from nnslr_tools.brain_net import build_brain_net, car_weights
from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES, METRES_SCALE, dataset, inputs_sha256, pack_inputs
from nnslr_tools.brain_train import CHECKPOINT, CHUNK, _pictures, predict
from nnslr_tools.road_type_net import FrameStore, _torch, gru_step

BRAIN_FILE = 'brain.onnx'
INPUT_NAMES = ['image', 'extra', 'hidden']
OUTPUT_NAMES = ['speed', 'hidden_out']
AHEAD_OUTPUT_NAMES = ['speed', 'next', 'metres', 'hidden_out']  # of a checkpoint with the limit ahead, as it was published
RTOL, ATOL = 1e-4, 1e-4
# Along a drive PyTorch runs the memory in chunks on a V100 and ONNX one frame at a time on the CPU: two classes whose
# scores stand closer than this are a tie, and either may win by the last bits of the float.
TIE = 1e-3


def load_brain(checkpoint: Path):
    """(the network on the CPU in eval mode, without works_head and road_head, with type_head when it learnt the road
    type; what the checkpoint says of itself). The network has the table of its checkpoint (class_kph): a brain of 5 km/h
    classes stays one. A checkpoint with the limit ahead (its 'ahead': the brains of 7 to 11 Nov 2026) keeps those heads."""
    torch = _torch()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    table = saved['weights'].get('class_kph')
    model = build_brain_net(tuple(saved['groups']), pool=saved['pool'], works=False, ahead=saved.get('ahead', False),
                            pretrained=False, features=saved['features'], hidden=saved['hidden'],
                            kph=None if table is None else table.tolist(), road_types=len(road_types_of(saved)))
    weights = {name: value for name, value in saved['weights'].items()
               if name in car_weights(model) or name.startswith('type_head.')}
    model.load_state_dict(weights, strict=True)
    return model.eval(), saved


def road_types_of(saved: dict) -> list[str]:
    """The road types a checkpoint's type_head says, in its order: none for a brain that did not learn them."""
    return list(saved.get('road_types') or []) if 'type_head.weight' in saved.get('weights', {}) else []


def output_names(model) -> list[str]:
    names = AHEAD_OUTPUT_NAMES if getattr(model, 'ahead', False) else OUTPUT_NAMES
    if getattr(model, 'type_head', None) is not None:
        names = [*names[:-1], 'road_type', names[-1]]
    return names


def step_module(model):
    """A step of the car: (image, extra, hidden) -> (speed logits, hidden_out), of a brain with the limit ahead (speed logits, next
    logits, metres, hidden_out), with the road type its logits just before hidden_out: the numbers of BrainNet.forward."""
    torch = _torch()
    from torch import nn

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, image, extra, hidden):
            x = torch.cat([self.model.encode(image), extra], dim=-1)
            hidden_out = gru_step(self.model.gru, x, hidden)
            number = self.model.number_in_force(extra)
            speed = self.model.speed_head(hidden_out) + self.model.trust(hidden_out) * number
            found = (speed,)
            if self.model.ahead:
                following = (self.model.next_head(hidden_out) +
                             self.model.trust_next(hidden_out) * torch.cat([number, number[..., :1] * 0], dim=-1))
                metres = (self.model.metres_head(hidden_out) +
                          self.model.metres_gate(hidden_out) * self.model.sign_metres(extra)[..., None])
                found += (following, metres)
            if getattr(self.model, 'type_head', None) is not None:
                found += (self.model.type_head(hidden_out),)
            return (*found, hidden_out)

    return Step().eval()


def _random_inputs(torch, width: int, hidden: int, generator) -> tuple:
    return (torch.rand(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH, generator=generator),
            torch.rand(1, width, generator=generator), torch.rand(1, hidden, generator=generator) * 2 - 1)


def table_of(model) -> list[int]:
    """The km/h of each class of the network, as the contract says them to the car."""
    return [int(round(value)) for value in model.class_kph.detach().cpu().tolist()]


def drive_rollout(model, step_run, drive: dict, store: FrameStore, device: str = 'cpu', log=None) -> dict:
    """A whole drive by step_run(image, extra, hidden) -> the step's outputs (speed logits first, hidden_out last), numpy,
    against brain_train.predict: the frames and how many of them say another speed, or, for a brain that has it, another
    limit ahead (a frame where ONNX scores both classes within TIE is a tie, counted apart). The memory
    starts again where the car's does (drive['cuts']). log, when given, hears the drive and every tenth of it.
    The frames go one at a time, so PyTorch runs on one thread meanwhile: its pool waited on onnxruntime's at every
    frame, 67 ms a frame instead of 2 (a drive of 11040 frames at 4 Hz: 12 minutes instead of 22 seconds)."""
    torch = _torch()
    try:
        expected = predict(model.to(device), drive, store, chunk=CHUNK, device=device)
    finally:
        model.to('cpu')  # the step exported and its other checks are the CPU's
    table = table_of(model)
    hidden = np.zeros((1, model.gru.hidden_size), dtype=np.float32)
    said = np.zeros(len(expected['kph']), dtype=np.int64)
    ahead = np.zeros(len(expected['kph']), dtype=np.int64)
    index = {kph: number for number, kph in reversed(list(enumerate(table)))}
    gap = np.zeros(len(said), dtype=np.float32)  # ONNX's best score less its score of PyTorch's speed
    gap_ahead = np.zeros(len(said), dtype=np.float32)
    tenth = max(1, len(said) // 10)
    if log:
        log(f"brain check along {drive['route']}: {len(said)} frames, one at a time as on the car")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for frame in range(len(said)):
            if frame in drive['cuts']:
                hidden = np.zeros_like(hidden)
            image = store.batch([(drive['route'], [frame])])[0].cpu().numpy()
            outputs = step_run(image, drive['extra'][frame:frame + 1], hidden)
            hidden = outputs[-1]
            scores = outputs[0].reshape(-1)
            said[frame] = table[int(np.argmax(scores))]
            gap[frame] = scores.max() - scores[index[int(expected['kph'][frame])]]
            if 'next' in expected:
                scores = outputs[1].reshape(-1)
                ahead[frame] = int(np.argmax(scores))
                gap_ahead[frame] = scores.max() - scores[int(expected['next'][frame])]
            if log and (frame + 1) % tenth == 0 and frame + 1 < len(said):
                log(f'brain check: {frame + 1} of {len(said)} frames')
    finally:
        torch.set_num_threads(threads)
    other = said != expected['kph']
    found = {'route': drive['route'], 'frames': len(said), 'other_speed': int((other & (gap > TIE)).sum()),
             'tied_speed': int((other & (gap <= TIE)).sum()), 'widest_gap': float(gap.max()),
             'said': sorted({int(v) for v in said})}
    if 'next' in expected:
        other = ahead != expected['next']
        found['other_ahead'] = int((other & (gap_ahead > TIE)).sum())
        found['tied_ahead'] = int((other & (gap_ahead <= TIE)).sum())
        found['widest_gap_ahead'] = float(gap_ahead.max())
    return found


def rollout_drive(root: Path, merged_of, groups: tuple[str, ...], route: str | None = None, device: str = 'cpu',
                  rate: float = 1.0, networks: dict | None = None) -> tuple[dict, FrameStore]:
    """A drive of the labels at the brain's rate, with its pictures: route, or the one with the most frames in test. With
    no route, among the drives the brain learns from (training_routes), the ones with their pictures (a drive added after
    the training has none until the next one: it is passed by). A drive without the brain's pictures is refused: the
    check would run on black pictures and say nothing. networks: the ones the brain's signs were read with
    (sign_networks.of_brain), None for the car's release of the sunnypilot clone."""
    drives = dataset(root, merged_of, [route] if route else None, rate=rate, networks=networks)
    if not drives:
        raise ValueError(f'{route or "no drive"}: not a drive the brain can run along (brain_targets.training_routes)')
    pictured = [drive for drive in drives if drive.get('pictures', True)]
    if not pictured:
        names = ' '.join(f"--route {drive['route']}" for drive in drives)
        raise ValueError(f"{route or 'every dashcam drive'}: the brain's pictures are not made. Run "
                         f"nnslr road-type-signs --labelled {names}")
    drive = max(pictured, key=lambda item: sum(partition == 'test' for partition in item['partitions']))
    drive['extra'] = pack_inputs(drive['inputs'], groups)
    return drive, FrameStore(_pictures(root, [drive], dry_run=False, log=lambda text: None), device)


def _export(step, example: tuple, output: Path, names: list[str] = OUTPUT_NAMES) -> None:
    torch = _torch()
    try:
        program = torch.onnx.export(step, example, input_names=INPUT_NAMES, output_names=names, dynamo=True,
                                    external_data=False)
        program.save(str(output))
    except ImportError as exc:
        raise ValueError('onnx_dependencies_missing: install onnx and onnxscript in the training environment') from exc
    except Exception as exc:  # torch.export refuses the graph: the TorchScript exporter
        print(f'torch.export could not export the brain step ({type(exc).__name__}); using TorchScript', flush=True)
        torch.onnx.export(step, example, str(output), input_names=INPUT_NAMES, output_names=names, dynamo=False,
                          opset_version=17)


def check_session(onnxruntime, path: Path):
    """onnxruntime for the checks, on the GPU (onnx_device.check_providers): one thread, not spinning. The step is small and goes one frame at a time between
    PyTorch's operations; a pool of a thread per core spinning after each run only takes the cores from PyTorch."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    return onnxruntime.InferenceSession(str(path), options, providers=onnx_device.check_providers(onnxruntime))


def export_brain_onnx(checkpoint: Path, output: Path, *, data_root: Path | None = None, merged_of=None,
                      route: str | None = None, log=None, device: str = 'cpu', networks: dict | None = None) -> dict:
    """The step and its contract, checked: raises when it differs from PyTorch or when the car's file of the inputs is not the
    one the network was trained with. data_root and merged_of add the check along a real drive; log hears its progress;
    device: where PyTorch runs that drive (Publish: the GPU, car_release.check_device); networks: the ones its signs
    were read with (sign_networks.of_brain)."""
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
    step, names = step_module(model), output_names(model)
    rate = float(saved.get('rate_hz', 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    _export(step, _random_inputs(torch, width, hidden, torch.Generator().manual_seed(1)), output, names)
    onnx.checker.check_model(onnx.load(str(output)))
    session = check_session(onnxruntime, output)

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
    verification = {'passed': passed, 'max_abs_error': worst, 'rtol': RTOL, 'atol': ATOL, 'provider': session.get_providers()[0]}
    if not passed:
        raise ValueError(f'brain ONNX differs from PyTorch: max abs error {worst}')
    if data_root is not None:
        drive, store = rollout_drive(Path(data_root), merged_of, groups, route, device=device, rate=rate,
                                     networks=networks)
        verification['rollout'] = drive_rollout(model, run, drive, store, device=device, log=log)
        if verification['rollout']['other_speed'] or verification['rollout'].get('other_ahead'):
            raise ValueError(f'brain ONNX says another speed than PyTorch along a drive: {verification["rollout"]}')
    shapes = {'image': [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH], 'extra': [1, width], 'hidden': [1, hidden]}
    brain_picture = saved.get('picture', 'road_type') == 'brain'
    resize = ({'method': 'vision.BrainInput: the NV12 planes scaled first with PIL bilinear weights, then RGB',
               'from': list(data.SOURCE_SIZE), 'to': [data.FRAME_WIDTH, data.FRAME_HEIGHT], 'crop': 'none, the whole frame'}
              if brain_picture else
              {'method': 'PIL BILINEAR (antialiased), vision.resize', 'from': list(data.SOURCE_SIZE),
               'to': [data.FRAME_WIDTH, data.FRAME_HEIGHT], 'crop': 'none, the whole frame'})
    table = table_of(model)
    outputs = [{'name': 'speed', 'shape': [1, len(table)], 'dtype': 'float32'}]
    if model.ahead:
        outputs += [{'name': 'next', 'shape': [1, len(table) + 1], 'dtype': 'float32',
                     'meaning': 'the next limit: kph[argmax], none for the last class'},
                    {'name': 'metres', 'shape': [1, 1], 'dtype': 'float32', 'meaning': 'the metres to it over metres_scale'}]
    road_types = road_types_of(saved)
    if road_types:
        outputs.append({'name': 'road_type', 'shape': [1, len(road_types)], 'dtype': 'float32',
                        'meaning': 'the road type: road_types[argmax], shown on the screen only'})
    outputs.append({'name': 'hidden_out', 'shape': [1, hidden], 'dtype': 'float32', 'meaning': 'give it back as hidden'})
    contract = {
        'schema_version': 1, 'task': 'brain', 'architecture': 'mobilenet_v3_small+gru', 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256_file(checkpoint), 'onnx_sha256': sha256_file(output), 'split_sha256': saved['split'],
        'epoch': saved['epoch'], 'inputs_sha256': saved['inputs_sha256'], 'reader': saved['reader'],
        'input': {'name': 'image', 'shape': shapes['image'], 'dtype': 'float32', 'layout': 'NCHW', 'color': 'RGB', 'range': '0..1',
                  'normalization': 'none: ImageNet mean and std are applied inside the network',
                  'resize': resize,
                  'source': ('narrow road camera NV12 -> vision.BrainInput' if brain_picture else
                             'narrow road camera NV12 -> vision.yuv_to_rgb -> vision.resize')},
        'inputs': [{'name': name, 'shape': shapes[name], 'dtype': 'float32'} for name in INPUT_NAMES],
        'output': {'name': 'speed', 'shape': [1, len(table)], 'kph': table,
                   'decode': 'kph[argmax(speed)]', **({'metres_scale': METRES_SCALE} if model.ahead else {}),
                   **({'road_types': road_types} if road_types else {})},
        'outputs': outputs,
        'step': {'rate_hz': rate, 'clock': 'last' if rate == 1.0 else 'schedule',
                 'picture': 'brain' if brain_picture else 'road_type', 'groups': list(groups),
                 'memory_gap_seconds': BRAIN_INPUTS.MEMORY_GAP_SECONDS,
                 'inputs': INPUT_NAMES, 'outputs': names, 'plates': bool(saved.get('plates', False)),
                 'hours': bool(saved.get('hours', False)), 'wet': bool(saved.get('wet', False))},
        'verification': verification,
    }
    contract_path = output.with_name(output.name + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'onnx': str(output), 'contract': str(contract_path), 'onnx_sha256': contract['onnx_sha256'],
            'verification': verification}
