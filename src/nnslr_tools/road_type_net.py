"""The road-type network with memory: the kind of road from the camera, one frame a second, and its changes.

Each second the car gives it the road frame (resized to 256 x 144, road_type_data), its speed and, with signs=True,
the seconds since its memory confirmed a motorway sign; with motion=True also its own motion of the last second
(car_motion: the heading change and the largest yaw rate, the steering angle, the blinkers, each with a known flag;
the car's own signals, never the map); with boards=True also the seconds since its memory confirmed a green board
towards a motorway or an expressway (road_type_signs: in Switzerland where the entry ramp starts, and at the
interchanges; the car never takes a limit from one, it only tells this network that a carriageway is near). A small pretrained encoder (torchvision's MobileNetV3-small, ImageNet weights)
makes 128 numbers of the frame, the other inputs are added, and a GRU of 128 keeps what the network saw before: the merge of a ramp, the first houses of a town and the end of them are
found from how the picture changes over seconds, not from one frame. The output is the probability of each class
(road_type_data.CLASSES). ``memory='none'`` is the same encoder and speed with a small head per frame: the network
without memory it is compared with (and a causal majority vote of 5 seconds over its answers, the cheap memory).

Training (train_road_type): windows of WINDOW seconds cut from the train sequences, FOCUS_SHARE of
them around a change of road type; half the batches start from the memory of BURN_IN seconds before the window (as on
the car, where the memory is never empty after the first seconds), the others from zeros; per sequence the same
random colour and framing; a second dropped now and then; the speed scaled by +-5 % and, in one sequence in ten,
unknown; the sign and the board missed or misread now and then; the car's motion unknown in MOTION_DROP_SHARE of the windows and its
yaw or its carState part in MOTION_PART_SHARE, so that the network still works without them. The speed input is
carState.vEgo when the motion is read (the GPS speed where it is missing). Loss: cross-entropy over the frames a person
confirmed or corrected in Data › Roads until 7 Nov 2026 (road_type_data.frame_label; every other frame only feeds the memory, and the
windows drawn anywhere are drawn around a confirmed frame), weighted by 1 / sqrt(class frequency). The training stops
at once, a dry run too, when the partitions hold too few of them (road_type_data.missing_labels); the best epoch by the road type's own measure on validation
(road_score: macro F1, the first minute after a merge, false switches; the whole sequences run with their memory, as
on the car).

The junction head (junctions=True, the default when nnslr junctions has run): two more outputs from the same memory,
the probability that the car is inside an intersection of public roads and that it has just left one ("passed": the
first second after it). In Poland, and in Switzerland by OSStr art. 16 cpv. 2, a speed sign holds at most to the end
of the next intersection; Poland has no end sign, so the car must see by itself that it crossed one. Labels from the
map and a person's corrections (junctions.py), binary cross-entropy beside the road type's loss, JUNCTION_SHARE of
the windows around a "passed". Once the road type's epoch is chosen, the junction head trains on its own
(train_junction_stage: the rest frozen; with memory, on a copy of the GRU, junction_branch 'separate', its own hidden
state), kept only when it beats the shared head on validation. Measured as events
(road_type_metrics.junction_metrics): recall within 2 and 4 s and, the dangerous error, false "passed" per hour and
per 10 km in town and outside; beside it the car's turn alone as a plain rule (turn_rule: a heading change of at
least TURN_DEGREES on a radius of at most TURN_RADIUS_M, the radius from the path and the heading).

The ONNX export (export_road_type_onnx) is one STEP (step_input_names, step_output_names): image (1, 3, 144, 256)
float RGB 0..1, speed (1, 2), sign (1, 2) with signs, motion (1, MOTION_INPUTS) with motion, board (1, 2) with
boards, hidden (1, 128) and
junction_hidden (1, 128) with a junction GRU of its own in; probabilities (1, classes) and hidden_out (1, 128) out, then
junction (1, 2) with the junction head and junction_hidden_out. The car keeps the hidden states and gives them back
the next second (zeros at the start). The GRU is written out with matrix products, sigmoid and tanh (no ONNX GRU node), the
ImageNet normalisation is inside. The contract beside it says how to make the inputs from the NV12 road frame, the
sign memory and the cereal fields of the car's motion (car_input).
Needs the GPU environment (.venv-gpu); the dry run works on the CPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from nnslr_tools import car_motion as motion_data
from nnslr_tools import road_type_data as data
from nnslr_tools import road_type_signs as signs_data
from nnslr_tools.car_motion import MOTION_INPUTS, NO_MOTION
from nnslr_tools.road_type_metrics import (AFTER_MERGE_SECONDS, JUNCTION_GAP_SECONDS, JUNCTION_TOLERANCES,
                                           LATE_SECONDS, TURN_DEGREES, TURN_MIN_DPS, TURN_RADIUS_M, change_points,
                                           comma_release, debounced, junction_events, junction_metrics, majority,
                                           match_events, operating_point, ramp_entries, release_summary,
                                           sequence_metrics, turn_events)
from nnslr_tools.road_type_signs import NO_SIGN

SIGN_INPUTS = 2
BOARD_INPUTS = 2
BOARD_CLASS = 'other_sign:motorway_direction'  # the car's vision.BOARD_SIGNS
JUNCTION_OUTPUTS = ('in_junction', 'passed')
JUNCTION_SHARE = 0.2  # of the train windows around a "passed" (taken from the windows drawn anywhere)
JUNCTION_LOSS = 1.0  # the junction head's loss beside the road type's
JUNCTION_DONT_CARE = 1  # seconds around a "passed" where the passed output is not taught (the map's timing is +-1 s)
NEEDED_METRES = 300.0  # an intersection "needed" by the car: a number sign confirmed within this before it

FEATURES = 128
HIDDEN = 128
# Every train window has the same length: cuDNN plans the half-precision backward of every new batch shape anew, about
# 6 s each on the V100 (29 Oct 2026), longer than the step itself.
WINDOW = 48
BURN_IN = 32
FOCUS_SHARE = 0.35
DROP_FRAME = 0.04
NO_SPEED_SHARE = 0.1
# The merge, where the ramp's limit ends (road_type_metrics.comma_release): MERGE_SHARE of the windows hold one (the
# ramp in the window or in its burn-in, the carriageway after it), and the loss weighs the ramp and the first
# AFTER_MERGE_SECONDS of the carriageway MERGE_WEIGHT times.
MERGE_SHARE = 0.25
MERGE_WEIGHT = 3.0
# The sign input (road_type_signs): in SIGN_DROP_SHARE of the windows the car missed the sign (none seen), in
# SIGN_FAKE_SHARE of those without one a sign is confirmed where there is none (a misreading).
SIGN_DROP_SHARE = 0.25
SIGN_FAKE_SHARE = 0.05
# The sign of a train window comes up to SIGN_JITTER seconds earlier or later: the reviewed 4.01 signs were seen from
# 7 s before to 6 s after the first second of the map's ramp, and the car confirms a sign while it sees it.
SIGN_JITTER = 4.0
# The board input (road_type_signs: the green boards the car's code confirmed) is missed, misread and moved as the sign
# input is (SIGN_DROP_SHARE, SIGN_FAKE_SHARE, SIGN_JITTER), each on its own. A start without it (--init-from) gets its
# columns of the memory's input as zeros: the network starts as the start and learns to read it.
MIN_BOARD_DRIVES = 3  # boards=None: the board input when the car's code confirmed boards on at least so many drives
# The car's own motion (car_motion: the turn of each second, the steering wheel, the blinkers): in MOTION_DROP_SHARE of
# the windows all of it is unknown (a car or a log without it: the network must work without), in MOTION_PART_SHARE
# one of its parts (the yaw, or the carState signals).
MOTION_DROP_SHARE = 0.2
MOTION_PART_SHARE = 0.1
# The road type's epoch is chosen on its own measure (road_score on validation): the macro F1, the first minute after
# a merge (debounced, ROAD_SCORE_MERGE for all of it right) and the false switches (debounced, ROAD_SCORE_SWITCHES
# per switch an hour); the junction head never chooses it, nor stops the training early.
ROAD_SCORE_MERGE = 0.1
ROAD_SCORE_SWITCHES = 0.0005
# Then the junction head on its own (train_junction_stage): the encoder and the road type frozen at that epoch, a
# junction GRU of its own (a copy of the road GRU to start from) and its head trained JUNCTION_EPOCHS at most, with
# JUNCTION_STAGE_SHARE of the windows around a "passed", the best epoch by the junction F1 at 2 s on validation. The
# shared head of the joint training at the road type's epoch is kept instead when it is as good on validation.
JUNCTION_EPOCHS = 30
JUNCTION_PATIENCE = 10
JUNCTION_LEARNING_RATE = 1e-3
JUNCTION_STAGE_SHARE = 0.5
JUNCTION_BRANCHES = ('shared', 'separate')
MIN_DRIVE = 30
# One letter a second for the answers around a merge in the result (release_evaluation).
ANSWER_LETTERS = {'town': 'T', 'rural': 'R', 'expressway': 'E', 'motorway': 'M', 'ramp': '/', 'parking': 'P',
                  'zone30': 'Z', 'garage': 'G', 'merge': 'A'}
BATCH = 12
EPOCHS = 60
PATIENCE = 15
LEARNING_RATE = 1e-3
BACKBONE_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SMOOTH_SECONDS = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT = 'road-type-best.pt'
MEMORY_NAMES = {'gru': 'road_type_gru', 'none': 'road_type_frame'}


def network_name(memory: str, signs: bool = True, motion: bool | None = None) -> str:
    """The name of a training in its log lines (job_progress reads them): road_type_gru, road_type_gru_nosign,
    road_type_gru_nomotion (the car's motion left out on purpose), ..."""
    return MEMORY_NAMES[memory] + ('' if signs else '_nosign') + ('_nomotion' if motion is False else '')


def _torch():
    import torch
    return torch


def _print(text: str) -> None:
    print(text, flush=True)


def build_road_type_net(memory: str = 'gru', pretrained: bool = True, features: int = FEATURES, hidden: int = HIDDEN,
                        signs: bool = False, junctions: bool = False, motion: bool = False,
                        junction_branch: str = 'shared', classes: int | None = None, boards: bool = False):
    """The network (torch.nn.Module): forward(images (B, T, 3, H, W) float 0..1, speed (B, T, 2), hidden (B, hidden)
    or None, sign (B, T, 2) or None, motion (B, T, MOTION_INPUTS) or None, junction_hidden (B, hidden) or None,
    board (B, T, 2) or None) ->
    (logits (B, T, classes), hidden (B, hidden), junction logits (B, T, 2) or None, junction hidden (B, hidden) or
    None). With signs, the input of the memory has the sign input too (road_type_signs.sign_features; None: no sign
    seen); with motion, the car's own motion (car_motion.motion_features; None: unknown); with boards, the board
    input after it (road_type_signs.sign_features of the last green board; None: none seen); with junctions, a junction
    head (in_junction, passed: JUNCTION_OUTPUTS) on the same memory (junction_branch 'shared') or on a junction GRU of
    its own over the same input ('separate', with memory only: its own hidden state). classes: the road types it
    answers, len(CLASSES) when None (a checkpoint trained before zone30 has six)."""
    torch = _torch()
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if memory not in MEMORY_NAMES:
        raise ValueError(f'memory must be one of {sorted(MEMORY_NAMES)}')
    if junction_branch not in JUNCTION_BRANCHES or (junction_branch == 'separate' and (memory != 'gru' or not junctions)):
        raise ValueError(f'junction_branch {junction_branch!r}: shared, or separate with memory and the junction head')
    extra = 2 + (SIGN_INPUTS if signs else 0) + (MOTION_INPUTS if motion else 0) + (BOARD_INPUTS if boards else 0)
    classes = len(data.CLASSES) if classes is None else classes

    class RoadTypeNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.memory, self.hidden_size, self.signs, self.junctions = memory, hidden, signs, junctions
            self.motion, self.junction_branch, self.boards = motion, junction_branch, boards
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.backbone = backbone.features
            self.project = nn.Sequential(nn.Linear(576, features), nn.Hardswish(), nn.Dropout(0.2))
            if memory == 'gru':
                self.gru = nn.GRU(features + extra, hidden, batch_first=True)
                self.head = nn.Linear(hidden, classes)
                if junctions:
                    self.junction_head = nn.Linear(hidden, len(JUNCTION_OUTPUTS))
                if junction_branch == 'separate':
                    self.junction_gru = nn.GRU(features + extra, hidden, batch_first=True)
            else:
                self.head = nn.Sequential(nn.Linear(features + extra, hidden), nn.ReLU(), nn.Linear(hidden, classes))
                if junctions:
                    self.junction_head = nn.Sequential(nn.Linear(features + extra, hidden), nn.ReLU(),
                                                       nn.Linear(hidden, len(JUNCTION_OUTPUTS)))

        def separate_junctions(self):
            """The junction branch of its own (train_junction_stage): a junction GRU copied from the road GRU, the
            head as it is; with memory only."""
            import copy
            if self.memory != 'gru' or not self.junctions:
                raise ValueError('a junction branch of its own needs memory and the junction head')
            if self.junction_branch != 'separate':
                self.junction_gru = copy.deepcopy(self.gru)
                self.junction_branch = 'separate'
            return self

        def encode(self, images):
            """(N, 3, H, W) 0..1 -> (N, features)."""
            x = self.backbone((images - self.mean) / self.std)
            return self.project(x.mean(dim=(2, 3)))

        def inputs(self, encoded, speed, sign=None, motion=None, board=None):
            """The input of the memory (..., features + 2 [+ 2] [+ MOTION_INPUTS] [+ 2]): the frame's numbers, the
            speed, the sign input, the car's motion, the board input."""
            parts = [encoded, speed.to(encoded.dtype)]
            if self.signs:
                if sign is None:
                    sign = encoded.new_tensor(NO_SIGN).expand(*encoded.shape[:-1], SIGN_INPUTS)
                parts.append(sign.to(encoded.dtype))
            if self.motion:
                if motion is None:
                    motion = encoded.new_tensor(NO_MOTION).expand(*encoded.shape[:-1], MOTION_INPUTS)
                parts.append(motion.to(encoded.dtype))
            if self.boards:
                if board is None:
                    board = encoded.new_tensor(NO_SIGN).expand(*encoded.shape[:-1], BOARD_INPUTS)
                parts.append(board.to(encoded.dtype))
            return torch.cat(parts, dim=-1)

        def forward(self, images, speed, hidden=None, sign=None, motion=None, junction_hidden=None, board=None):
            b, t = images.shape[:2]
            return self.run(self.inputs(self.encode(images.flatten(0, 1)).view(b, t, -1), speed, sign, motion, board),
                            hidden, junction_hidden)

        def run(self, x, hidden=None, junction_hidden=None):
            """The memory's input (B, T, features + extra) -> (logits, hidden, junction logits or None, junction
            hidden or None)."""
            if self.memory == 'gru':
                out, last = self.gru(x, None if hidden is None else hidden[None].contiguous())
                if self.junction_branch == 'separate':
                    out_j, last_j = self.junction_gru(x, None if junction_hidden is None else junction_hidden[None].contiguous())
                    return self.head(out), last[0], self.junction_head(out_j), last_j[0]
                return self.head(out), last[0], self.junction_head(out) if self.junctions else None, None
            b = x.shape[0]
            return (self.head(x), hidden if hidden is not None else x.new_zeros(b, self.hidden_size),
                    self.junction_head(x) if self.junctions else None, None)

    return RoadTypeNet()


def gru_step(gru, x, hidden):
    """One step of a torch.nn.GRU (one layer) with matrix products, sigmoid and tanh: the same numbers, no GRU node in
    the ONNX graph (the comma compiles the graph with tinygrad)."""
    torch = _torch()
    import torch.nn.functional as F
    gates_x = F.linear(x, gru.weight_ih_l0, gru.bias_ih_l0)
    gates_h = F.linear(hidden, gru.weight_hh_l0, gru.bias_hh_l0)
    xr, xz, xn = gates_x.chunk(3, dim=1)
    hr, hz, hn = gates_h.chunk(3, dim=1)
    reset = torch.sigmoid(xr + hr)
    update = torch.sigmoid(xz + hz)
    candidate = torch.tanh(xn + reset * hn)
    return (1 - update) * candidate + update * hidden


def step_input_names(model) -> list[str]:
    """The inputs of the car's step, in order: image, speed, [sign,] [motion,] [board,] hidden[, junction_hidden]."""
    return ['image', 'speed', *(['sign'] if getattr(model, 'signs', False) else []),
            *(['motion'] if getattr(model, 'motion', False) else []),
            *(['board'] if getattr(model, 'boards', False) else []), 'hidden',
            *(['junction_hidden'] if getattr(model, 'junction_branch', 'shared') == 'separate' else [])]


def step_output_names(model) -> list[str]:
    """The outputs of the car's step, in order: probabilities, hidden_out[, junction][, junction_hidden_out]."""
    return ['probabilities', 'hidden_out', *(['junction'] if getattr(model, 'junctions', False) else []),
            *(['junction_hidden_out'] if getattr(model, 'junction_branch', 'shared') == 'separate' else [])]


def step_module(model):
    """The one-second step of the car: the inputs of step_input_names ((1, 3, H, W) 0..1, (1, 2), [(1, 2),]
    [(1, MOTION_INPUTS),] [(1, 2),] (1, hidden)[, (1, hidden)]) -> the outputs of step_output_names (probabilities (1, classes),
    hidden_out (1, hidden)[, junction (1, 2): sigmoid probabilities of JUNCTION_OUTPUTS][, junction_hidden_out])."""
    torch = _torch()
    from torch import nn

    def advance(image, speed, hidden, sign=None, motion=None, junction_hidden=None, board=None):
        x = model.inputs(model.encode(image), speed, sign, motion, board)
        junction_out = None
        if model.memory == 'gru':
            hidden_out = gru_step(model.gru, x, hidden)
            logits = model.head(hidden_out)
            if model.junction_branch == 'separate':
                junction_out = gru_step(model.junction_gru, x, junction_hidden)
                junction = model.junction_head(junction_out)
            else:
                junction = model.junction_head(hidden_out) if model.junctions else None
        else:
            hidden_out = hidden * 1.0
            logits = model.head(x)
            junction = model.junction_head(x) if model.junctions else None
        out = [torch.softmax(logits, dim=1), hidden_out]
        if junction is not None:
            out.append(torch.sigmoid(junction))
        if junction_out is not None:
            out.append(junction_out)
        return tuple(out)

    names = step_input_names(model)
    # A forward with the inputs by name, in the car's order (torch.export reads its signature).
    source = (f"def forward(self, {', '.join(names)}):\n"
              f"    return self.advance({', '.join(f'{name}={name}' for name in names)})\n")
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - the names are ours (step_input_names), not data

    class RoadTypeStep(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def advance(self, **inputs):
            return advance(**inputs)

    RoadTypeStep.forward = namespace['forward']
    return RoadTypeStep().eval()


def step_example(model, torch=None):
    """Zero inputs of one step, in the order of step_input_names (no sign or board seen, the motion unknown)."""
    torch = torch or _torch()
    values = {'image': torch.zeros(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH), 'speed': torch.zeros(1, 2),
              'sign': torch.tensor([NO_SIGN]), 'motion': torch.tensor([NO_MOTION]), 'board': torch.tensor([NO_SIGN]),
              'hidden': torch.zeros(1, model.hidden_size), 'junction_hidden': torch.zeros(1, model.hidden_size)}
    return tuple(values[name] for name in step_input_names(model))


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def step_flops(model, width: int = data.FRAME_WIDTH, height: int = data.FRAME_HEIGHT) -> int:
    """Floating point operations of one step (torch's FlopCounterMode: 2 per multiply-add of convolutions and matrix
    products)."""
    torch = _torch()
    import copy
    from torch.utils.flop_counter import FlopCounterMode
    step = step_module(copy.deepcopy(model).cpu())
    example = step_example(model, torch)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        step(*example)
    return int(counter.get_total_flops())


class FrameStore:
    """The resized pictures of the drives in one uint8 tensor (frames, H, W, 3) on the device, and each drive's
    first row in it. A drive's pictures can be anything with a length that np.asarray reads: they are read one drive at a
    time, straight into their place, so that the memory holds one drive besides the store."""

    def __init__(self, arrays: dict[str, np.ndarray], device):
        torch = _torch()
        self.offset, start = {}, 0
        for route, array in arrays.items():
            self.offset[route] = start
            start += len(array)
        self.pixels = None
        for route, array in arrays.items():
            values = torch.from_numpy(np.ascontiguousarray(np.asarray(array)))
            if self.pixels is None:
                self.pixels = torch.empty((start, *values.shape[1:]), dtype=values.dtype, device=device)
            self.pixels[self.offset[route]:self.offset[route] + len(values)] = values.to(device)
        if self.pixels is None:
            raise ValueError('no pictures')

    def batch(self, route_positions: list[tuple[str, list[int]]]):
        """(B, T, 3, H, W) float 0..1 of equal-length position lists."""
        torch = _torch()
        index = torch.tensor([[self.offset[route] + p for p in positions] for route, positions in route_positions],
                             device=self.pixels.device)
        return self.pixels[index].permute(0, 1, 4, 2, 3).float().div_(255)


def attach_signs(dataset: dict, root: Path, source: str = signs_data.DEFAULT_SOURCE) -> dict:
    """The sign input of every frame (dataset['signs'][route], (frames, 2)) from the motorway signs of source
    (road_type_signs.SIGN_SOURCES: 'car', those the car's code confirmed, read before by nnslr road-type-signs, a drive
    without them: none seen; 'map', the first second of every entry ramp of the map), their times
    (dataset['sign_times']), the time of every frame (dataset['seconds']) and the loss emphasis of the merges
    (dataset['emphasis'][route]). Returns the coverage."""
    if source not in signs_data.SIGN_SOURCES:
        raise ValueError(f'unknown sign source {source!r}: one of {", ".join(signs_data.SIGN_SOURCES)}')
    dataset['signs'], dataset['sign_times'], dataset['emphasis'], dataset['seconds'] = {}, {}, {}, {}
    read = []
    for route, frames in dataset['frames'].items():
        if source == 'car':
            record = signs_data.load_signs(root, route)
            if record:
                read.append(route)
            times = signs_data.sign_times(record)
        else:
            read.append(route)
            times = signs_data.map_sign_times(frames, data.CLASSES)
        dataset['sign_times'][route] = times
        dataset['seconds'][route] = np.array([signs_data.frame_seconds(f) for f in frames], dtype=np.float64)
        dataset['signs'][route] = signs_data.frame_sign_features(frames, times)
        labels = np.array([f['label'] for f in frames], dtype=np.int64)
        lead = np.array([bool(f.get('lead')) for f in frames], dtype=bool)
        emphasis = np.ones(len(frames), dtype=np.float32)
        for entry in ramp_entries(labels, data.CLASSES, lead):
            emphasis[entry['ramp']:entry['merge'] + AFTER_MERGE_SECONDS] = MERGE_WEIGHT
        dataset['emphasis'][route] = emphasis
    coverage = {'source': source, 'drives': len(dataset['frames']), 'read': len(read),
                'signs': sum(len(t) for t in dataset['sign_times'].values()),
                'drives_with_signs': sum(1 for t in dataset['sign_times'].values() if t)}
    dataset['sign_coverage'] = coverage
    return coverage


def attach_boards(dataset: dict, root: Path) -> dict:
    """The board input of every frame (dataset['boards'][route], (frames, 2)) from the green boards towards a motorway
    the car's code confirmed (read before by nnslr road-type-signs; a drive read before the boards, or not read:
    none seen) and their times (dataset['board_times']). Returns the coverage: the drives read with the boards, the
    boards and the drives with one, and the classifier they were read with (None: the car's)."""
    dataset['boards'], dataset['board_times'] = {}, {}
    read, classifiers = [], set()
    for route, frames in dataset['frames'].items():
        record = signs_data.load_signs(root, route)
        if record and 'boards' in record:
            read.append(route)
            classifiers.add(record.get('classifier'))
        times = signs_data.board_times(record)
        dataset['board_times'][route] = times
        dataset['boards'][route] = signs_data.frame_sign_features(frames, times)
    coverage = {'drives': len(dataset['frames']), 'read': len(read),
                'boards': sum(len(t) for t in dataset['board_times'].values()),
                'drives_with_boards': sum(1 for t in dataset['board_times'].values() if t),
                'classifiers': sorted(str(c) for c in classifiers)}
    dataset['board_coverage'] = coverage
    return coverage


def attach_motion(dataset: dict, root: Path, *, v_ego: bool = True) -> dict:
    """The car's own motion at every frame (dataset['motion'][route], (frames, MOTION_INPUTS); NO_MOTION where the
    drive's logs were not read: nnslr car-motion) and, with v_ego, the speed input from carState.vEgo
    (dataset['speed_input'][route], (frames, 2); the GPS speed where vEgo is unknown). Returns the coverage."""
    dataset['motion'], counts = {}, {'drives': len(dataset['frames']), 'read': 0, 'frames': 0, 'yaw': 0, 'car': 0,
                                     'v_ego': 0, 'gps_speed': 0}
    if v_ego:
        dataset['speed_input'] = {}
    for route, frames in dataset['frames'].items():
        record = motion_data.load_motion(root, route)
        counts['read'] += record is not None
        items = [motion_data.frame_motion(record, f['segment'], f['index']) for f in frames]
        features = np.array([motion_data.motion_features(item) for item in items], dtype=np.float32
                            ).reshape(len(frames), MOTION_INPUTS)
        dataset['motion'][route] = features
        counts['frames'] += len(frames)
        counts['yaw'] += int((features[:, 2] > 0).sum())
        counts['car'] += int((features[:, 7] > 0).sum())
        counts['gps_speed'] += sum(1 for f in frames if data.speed_features(f['speed_mps'])[1])
        known = [item.get('v_ego') if item else None for item in items]
        counts['v_ego'] += sum(1 for v in known if v is not None)
        if v_ego:
            dataset['speed_input'][route] = np.array(
                [data.speed_features(v if v is not None else f['speed_mps']) for v, f in zip(known, frames)],
                dtype=np.float32).reshape(len(frames), 2)
    coverage = {**counts, 'speed_source': 'v_ego' if v_ego else 'gps'}
    dataset['motion_coverage'] = coverage
    return coverage


def _sequence_arrays(dataset: dict, run: dict) -> dict:
    route, positions = run['route'], run['positions']
    frames = dataset['frames'][route]
    chosen = [frames[p] for p in positions]
    signs = dataset.get('signs', {}).get(route)
    boards = dataset.get('boards', {}).get(route)
    emphasis = dataset.get('emphasis', {}).get(route)
    seconds = dataset.get('seconds', {}).get(route)
    motion = dataset.get('motion', {}).get(route)
    speed = dataset.get('speed_input', {}).get(route)
    return {'labels': np.array([f['label'] for f in chosen], dtype=np.int64),
            'lead': np.array([bool(f.get('lead')) for f in chosen], dtype=bool),
            'weights': np.array([f['weight'] for f in chosen], dtype=np.float32),
            'speed': (speed[positions] if speed is not None else
                      np.array([data.speed_features(f['speed_mps']) for f in chosen], dtype=np.float32)
                      ).astype(np.float32).reshape(len(chosen), 2),
            'sign': (signs[positions] if signs is not None else np.tile(np.float32(NO_SIGN), (len(chosen), 1))
                     ).astype(np.float32),
            'motion': (motion[positions] if motion is not None else
                       np.tile(np.float32(NO_MOTION), (len(chosen), 1))).astype(np.float32),
            'board': (boards[positions] if boards is not None else np.tile(np.float32(NO_SIGN), (len(chosen), 1))
                      ).astype(np.float32),
            'emphasis': (emphasis[positions] if emphasis is not None else np.ones(len(chosen))).astype(np.float32),
            'seconds': (seconds[positions] if seconds is not None else
                        np.array([signs_data.frame_seconds(f) for f in chosen], dtype=np.float64)),
            'keys': [f['key'] for f in chosen], **junction_arrays(chosen)}


def junction_arrays(frames: list[dict]) -> dict:
    """The junction head's targets of a run of frames: 'junction' (T, 2) in_junction and passed, 'junction_weight'
    (T, 2) (0 where the junctions are unknown; the passed output not taught JUNCTION_DONT_CARE seconds around a
    "passed"), 'junction_known', 'junction_ignore' and 'speed_mps' (nan unknown) for the measure."""
    known = np.array([bool(f.get('junction_known')) for f in frames])
    inside = np.array([bool(f.get('junction_in')) for f in frames])
    passed = np.array([bool(f.get('junction_passed')) for f in frames])
    weight = np.repeat(known[:, None].astype(np.float32), len(JUNCTION_OUTPUTS), axis=1)
    for t in np.flatnonzero(passed):
        for near in range(max(0, t - JUNCTION_DONT_CARE), min(len(frames), t + JUNCTION_DONT_CARE + 1)):
            if not passed[near]:
                weight[near, 1] = 0.0
    return {'junction': np.stack([inside, passed], axis=1).astype(np.float32), 'junction_weight': weight,
            'junction_known': known, 'junction_ignore': np.array([bool(f.get('junction_ignore')) for f in frames]),
            'speed_mps': np.array([np.nan if f.get('speed_mps') is None else float(f['speed_mps']) for f in frames])}


def class_weights(dataset: dict, runs: list[dict]) -> np.ndarray:
    """1 / sqrt(frequency) of each class over the train frames (by their weights), mean 1 over the frames; 0 for a
    class without frames."""
    totals = np.zeros(len(data.CLASSES))
    for run in runs:
        arrays = _sequence_arrays(dataset, run)
        for label, weight in zip(arrays['labels'], arrays['weights']):
            if label >= 0:
                totals[label] += weight
    weights = np.where(totals > 0, 1 / np.sqrt(np.maximum(totals, 1e-9)), 0.0)
    scale = (totals * weights).sum() / max(totals.sum(), 1e-9)
    return (weights / scale).astype(np.float32) if scale > 0 else weights.astype(np.float32)


def junction_pos_weights(dataset: dict, runs: list[dict]) -> np.ndarray:
    """The positive weight of each junction output over the train frames with labels: sqrt(negatives / positives),
    1 without positives."""
    positive = np.zeros(len(JUNCTION_OUTPUTS))
    total = np.zeros(len(JUNCTION_OUTPUTS))
    for run in runs:
        arrays = _sequence_arrays(dataset, run)
        positive += (arrays['junction'] * arrays['junction_weight']).sum(axis=0)
        total += arrays['junction_weight'].sum(axis=0)
    return np.where(positive > 0, np.sqrt(np.maximum(total - positive, 1) / np.maximum(positive, 1)), 1.0
                    ).astype(np.float32)


def junction_rules() -> dict:
    from nnslr_tools.junctions import rules
    return rules()


def junction_text(metrics: dict, numbers: dict) -> str:
    """One line of a junction measure: 'N passed: recall R2 (2 s) / R4 (4 s), F1 F; false passed X/h, Y/10 km (town
    ..., outside ...)'."""
    town, outside = numbers['town'], numbers['outside']
    return (f"{metrics['passed']} passed: recall {_numbers(numbers['recall_2s'])} (2 s) / "
            f"{_numbers(numbers['recall_4s'])} (4 s), F1 {_numbers(numbers['f1_2s'])}; false passed "
            f"{_numbers(numbers['false_per_hour'])}/h, {_numbers(numbers['false_per_10km'])}/10 km (town "
            f"{_numbers(town['per_hour'])}/h {_numbers(town['per_10km'])}/10 km, outside "
            f"{_numbers(outside['per_hour'])}/h {_numbers(outside['per_10km'])}/10 km)")


def _augment(torch, images, generator_seed: int | None = None):
    """The same colour and framing on every frame of a sequence, different per sequence: brightness, contrast and
    saturation +-25 %, a zoom of up to 12 % and a shift and turn of a few pixels and degrees. No mirror: traffic keeps
    right."""
    import torch.nn.functional as F
    b, t, c, h, w = images.shape
    device = images.device
    brightness = torch.empty(b, 1, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    contrast = torch.empty(b, 1, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    saturation = torch.empty(b, 1, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    mean = images.mean(dim=(1, 2, 3, 4), keepdim=True)
    images = (images - mean) * contrast + mean
    images = images * brightness
    gray = images.mean(dim=2, keepdim=True)
    images = (images - gray) * saturation + gray
    scale = torch.empty(b, device=device).uniform_(1 / 1.12, 1.0)
    angle = torch.empty(b, device=device).uniform_(-2, 2) * math.pi / 180
    shift_x = torch.empty(b, device=device).uniform_(-1, 1) * (1 - scale)
    shift_y = torch.empty(b, device=device).uniform_(-1, 1) * (1 - scale)
    theta = torch.zeros(b, 2, 3, device=device)
    theta[:, 0, 0] = scale * torch.cos(angle)
    theta[:, 0, 1] = -scale * torch.sin(angle) * h / w
    theta[:, 1, 0] = scale * torch.sin(angle) * w / h
    theta[:, 1, 1] = scale * torch.cos(angle)
    theta[:, 0, 2], theta[:, 1, 2] = shift_x, shift_y
    theta = theta.repeat_interleave(t, dim=0)
    flat = images.reshape(b * t, c, h, w)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return flat.view(b, t, c, h, w).clamp_(0, 1)


class WindowSampler:
    """Train windows: (route, positions, labels, weights, speed, sign, emphasis, junction, junction weight, motion,
    board) of
    length burn + window, a second dropped now and then, merge_share of them around a merge, focus_share around a
    change of road type, junction_share (with the junction head) around a "passed" and the others anywhere around a
    confirmed frame (the frames without a label teach the road type nothing; around_labels False: anywhere, for the
    junction head alone); the sign input missed or
    misread now and then (SIGN_DROP_SHARE, SIGN_FAKE_SHARE) and confirmed up to SIGN_JITTER seconds earlier or later,
    the board input the same on its own;
    the car's motion unknown now and then (MOTION_DROP_SHARE, MOTION_PART_SHARE)."""

    def __init__(self, dataset: dict, runs: list[dict], rng: random.Random, junctions: bool = False, *,
                 merge_share: float = MERGE_SHARE, focus_share: float = FOCUS_SHARE,
                 junction_share: float = JUNCTION_SHARE, around_labels: bool = True):
        self.rng = rng
        self.shares = (merge_share, focus_share, junction_share)
        self.times = dataset.get('sign_times', {})
        self.board_times = dataset.get('board_times', {})
        self.runs = [(run, _sequence_arrays(dataset, run)) for run in runs]
        # With the junction head, JUNCTION_SHARE of the windows around a "passed" (none without: the draws stay those
        # of the road type alone).
        self.junctions = [(i, int(t)) for i, (run, arrays) in enumerate(self.runs)
                          for t in np.flatnonzero(arrays['junction'][:, 1] > 0)] if junctions else []
        self.changes = [(i, change['time']) for i, (run, arrays) in enumerate(self.runs)
                        for change in change_points(arrays['labels'], data.CLASSES)]
        self.merges = [(i, entry['merge']) for i, (run, arrays) in enumerate(self.runs)
                       for entry in ramp_entries(arrays['labels'], data.CLASSES, arrays['lead'])]
        self.lengths = [len(run['positions']) for run, _ in self.runs]
        self.anchors = [(np.flatnonzero(arrays['labels'] >= 0) if around_labels else np.arange(len(arrays['labels']))
                         ).tolist() for _, arrays in self.runs]

    def window(self, length: int):
        rng = self.rng
        fits = [i for i, n in enumerate(self.lengths) if n >= length]
        if not fits:
            raise ValueError(f'no train sequence of {length} frames')
        fitting = set(fits)
        focus = [(i, t) for i, t in self.changes if i in fitting]
        merges = [(i, t) for i, t in self.merges if i in fitting]
        crossings = [(i, t) for i, t in self.junctions if i in fitting]
        draw = rng.random()
        merge_share, focus_share, junction_share = self.shares
        if merges and draw < merge_share:
            index, moment = rng.choice(merges)
            start = min(max(moment - rng.randrange(min(10, length - 1), max(length - 10, 11)), 0),
                        self.lengths[index] - length)
        elif focus and draw < merge_share + focus_share:
            index, moment = rng.choice(focus)
            start = min(max(moment - rng.randrange(length), 0), self.lengths[index] - length)
        elif crossings and draw < merge_share + focus_share + junction_share:
            index, moment = rng.choice(crossings)
            start = min(max(moment - rng.randrange(min(5, length - 1), length), 0), self.lengths[index] - length)
        else:
            anchors = [len(self.anchors[i]) for i in fits]
            index = rng.choices(fits, weights=anchors)[0] if any(anchors) else rng.choice(fits)
            moment = rng.choice(self.anchors[index]) if self.anchors[index] else rng.randrange(self.lengths[index])
            start = min(max(moment - rng.randrange(length), 0), self.lengths[index] - length)
        run, arrays = self.runs[index]
        picks, cursor = [], start
        while len(picks) < length and cursor < self.lengths[index]:
            picks.append(cursor)
            cursor += 2 if rng.random() < DROP_FRAME else 1
        while len(picks) < length:
            if picks[0] > 0:
                picks.insert(0, picks[0] - 1)
            else:
                picks.append(picks[-1])
        return (run['route'], [run['positions'][p] for p in picks], arrays['labels'][picks],
                arrays['weights'][picks], arrays['speed'][picks].copy(),
                self._sign(arrays['sign'][picks].copy(), arrays['seconds'][picks], self.times.get(run['route'], [])),
                arrays['emphasis'][picks], arrays['junction'][picks], arrays['junction_weight'][picks],
                self._motion(arrays['motion'][picks].copy()),
                self._sign(arrays['board'][picks].copy(), arrays['seconds'][picks],
                           self.board_times.get(run['route'], [])))

    def _motion(self, motion: np.ndarray) -> np.ndarray:
        """The motion input of a window, all unknown (MOTION_DROP_SHARE) or its yaw or its carState part unknown
        (MOTION_PART_SHARE) now and then."""
        rng = self.rng
        draw = rng.random()
        if draw < MOTION_DROP_SHARE:
            motion[:] = NO_MOTION
        elif draw < MOTION_DROP_SHARE + MOTION_PART_SHARE:
            if rng.random() < 0.5:
                motion[:, 0:3] = 0.0  # heading change, largest yaw rate, yaw known
            else:
                motion[:, 3:8] = 0.0  # steering, steer known, the blinkers, blinkers known
        return motion

    def _sign(self, sign: np.ndarray, seconds: np.ndarray, times: list[float]) -> np.ndarray:
        """The sign input of a window, missed (SIGN_DROP_SHARE) or with a misreading (SIGN_FAKE_SHARE) now and then,
        its signs moved by up to SIGN_JITTER seconds."""
        rng = self.rng
        if rng.random() < SIGN_DROP_SHARE:
            sign[:] = NO_SIGN
            return sign
        if len(times) and SIGN_JITTER:
            shift = rng.uniform(-SIGN_JITTER, SIGN_JITTER)
            sign = signs_data.features_at(seconds, [t + shift for t in times])
        if rng.random() < SIGN_FAKE_SHARE:
            at = rng.randrange(len(sign))
            for k in range(at, len(sign)):
                fake = signs_data.sign_features(float(k - at))
                if sign[k, 1] == 0 or sign[k, 0] > fake[0]:
                    sign[k] = fake
        return sign


def _speed_noise(rng: random.Random, speed: np.ndarray) -> np.ndarray:
    if rng.random() < NO_SPEED_SHARE:
        return np.zeros_like(speed)
    speed[:, 0] *= rng.uniform(0.95, 1.05)
    return speed


def _predict_sequence(torch, model, store: FrameStore, route: str, positions: list[int], speed: np.ndarray,
                      device, chunk: int = 256, sign: np.ndarray | None = None, junction: bool = False,
                      motion: np.ndarray | None = None, board: np.ndarray | None = None):
    """Probabilities (T, classes) of a whole sequence run from zeros with its memory, as the car would; with junction,
    (probabilities, junction probabilities (T, 2) or None)."""
    model.eval()
    with torch.no_grad():
        features = []
        for start in range(0, len(positions), chunk):
            images = store.batch([(route, positions[start:start + chunk])])[0]
            features.append(model.encode(images))
        x = model.inputs(torch.cat(features), torch.from_numpy(speed).to(device),
                         None if sign is None else torch.from_numpy(sign).to(device),
                         None if motion is None else torch.from_numpy(motion).to(device),
                         None if board is None else torch.from_numpy(board).to(device))[None]
        logits, _, crossing, _ = model.run(x)
        probabilities = torch.softmax(logits[0].float(), dim=1).cpu().numpy()
        if not junction:
            return probabilities
        return probabilities, None if crossing is None else torch.sigmoid(crossing[0].float()).cpu().numpy()


def _junction_sequence(arrays: dict, predicted: np.ndarray, crossing: np.ndarray) -> dict:
    """A sequence for road_type_metrics.junction_metrics."""
    return {'seconds': arrays['seconds'], 'p_in': crossing[:, 0], 'p_passed': crossing[:, 1],
            'in': arrays['junction'][:, 0] > 0, 'passed': arrays['junction'][:, 1] > 0,
            'known': arrays['junction_known'], 'ignore': arrays['junction_ignore'], 'labels': arrays['labels'],
            'predicted': predicted, 'speed': arrays['speed_mps']}


def evaluate(torch, model, store: FrameStore, dataset: dict, runs: list[dict], device, *, smooth: bool = False) -> dict:
    """sequence_metrics of a partition's sequences (and of the 5-second majority vote of the answers with smooth),
    plus the answers per sequence; with the junction head its measure too ('junctions': junction_metrics)."""
    found, smoothed, answers, crossings = [], [], [], []
    with_junctions = getattr(model, 'junctions', False)
    for run in runs:
        arrays = _sequence_arrays(dataset, run)
        probabilities, crossing = _predict_sequence(torch, model, store, run['route'], run['positions'],
                                                    arrays['speed'], device, sign=arrays['sign'], junction=True,
                                                    motion=arrays['motion'], board=arrays['board'])
        predicted = probabilities.argmax(axis=1)
        found.append({'labels': arrays['labels'], 'lead': arrays['lead'], 'weights': arrays['weights'],
                      'predicted': predicted})
        if smooth:
            smoothed.append({**found[-1], 'predicted': majority(predicted, SMOOTH_SECONDS, len(data.CLASSES))})
        answer = {'route': run['route'], 'keys': arrays['keys'], 'labels': arrays['labels'].tolist(),
                  'predicted': predicted.tolist(), 'probabilities': np.round(probabilities, 4).tolist()}
        if with_junctions and crossing is not None:
            crossings.append(_junction_sequence(arrays, predicted, crossing))
            answer.update(junction=np.round(crossing, 4).tolist(),
                          junction_labels=arrays['junction'].astype(int).tolist(),
                          seconds=arrays['seconds'].tolist())
        answers.append(answer)
    result = {'metrics': sequence_metrics(found, data.CLASSES), 'answers': answers}
    if with_junctions:
        result['metrics']['junctions'] = junction_metrics(crossings, data.CLASSES)
    if smooth:
        result['smoothed'] = sequence_metrics(smoothed, data.CLASSES)
    return result


def needed_junctions(dataset: dict, answers: list[dict], root: Path, operating: dict) -> dict | None:
    """The intersections of the test where the car would have needed the junction head: a number sign confirmed by
    the car's own memory (vision.SpeedLimitMemory over the readings of nnslr road-type-signs) within NEEDED_METRES of
    track before the intersection (in Poland that limit ends there), and how many of them the head says "passed" for
    within 2 and 4 s at the operating decode and threshold. answers: evaluate's answers of the test. None without the
    car's readings or the sunnypilot clone."""
    try:
        vision = signs_data.load_vision(signs_data.package_path(None))
    except (OSError, ValueError, ImportError, AttributeError):
        return None
    confirmed: dict[str, list] = {}
    found = []
    for answer in answers:
        route = answer['route']
        if 'junction' not in answer:
            return None
        if route not in confirmed:
            record = signs_data.load_signs(root, route)
            if record is None:
                return None
            confirmed[route] = signs_data.number_confirmations(vision, record['readings'])
        frames = dataset['frames'][route]
        by_key = {f['key']: f for f in frames}
        drive_t = np.array([signs_data.frame_seconds(f) for f in frames if f.get('distance_m') is not None])
        drive_d = np.array([f['distance_m'] for f in frames if f.get('distance_m') is not None])
        if not len(drive_t):
            continue
        seconds = np.asarray(answer['seconds'])
        labels = np.asarray(answer['junction_labels'])
        crossing = np.asarray(answer['junction'])
        events = [float(seconds[t]) for t in junction_events(crossing[:, 1], crossing[:, 0], operating['threshold'],
                                                              operating['decode'], seconds=seconds)]
        for t in np.flatnonzero(labels[:, 1] > 0):
            start = t
            while start > 0 and labels[start - 1, 0] > 0:
                start -= 1
            enter = float(seconds[start if start < t else t])
            at = float(np.interp(enter, drive_t, drive_d))
            signs = [(when, value) for when, value in confirmed[route]
                     if when <= enter and at - float(np.interp(when, drive_t, drive_d)) <= NEEDED_METRES]
            frame = by_key[answer['keys'][t]]
            found.append({'route': route, 'passed_key': frame['key'], 'junction': frame.get('junction'),
                          'needed': bool(signs), 'sign': signs[-1][1] if signs else None,
                          **{f'caught_{tolerance:g}s': bool(match_events(events, [float(seconds[t])], tolerance))
                             for tolerance in JUNCTION_TOLERANCES}})
    needed = [item for item in found if item['needed']]
    return {'junctions': len(found), 'needed': len(needed), 'metres': NEEDED_METRES,
            'decode': operating['decode'], 'threshold': operating['threshold'],
            **{f'caught_{tolerance:g}s': sum(1 for item in needed if item[f'caught_{tolerance:g}s'])
               for tolerance in JUNCTION_TOLERANCES}, 'list': found}


def drive_runs(dataset: dict, min_length: int = MIN_DRIVE) -> list[dict]:
    """Every drive as the car drives it: the runs of its frames without a missing second, whatever their partition."""
    return [run for frames in dataset['frames'].values()
            for run in data.sequences(frames, ['drive'] * len(frames), min_length=min_length)]


def merge_partition(partitions: list[str]) -> str | None:
    """The partition a merge is measured in: none when a second of its ramp or of the LATE_SECONDS after it was
    trained on, else test or validation (the buffer between them is never trained on)."""
    if 'train' in partitions:
        return None
    return 'test' if 'test' in partitions else 'validation' if 'validation' in partitions else None


def release_evaluation(torch, model, store: FrameStore, dataset: dict, device, *, answers: dict | None = None) -> dict:
    """The comma's rule (road_type_metrics.comma_release) on every merge of validation and test: each drive run whole
    from zeros, as on the car, the answer debounced; the age of the last motorway sign at each second from the signs
    attached (attach_signs: dataset['sign_times'], the network's sign input too). {'sign_source', 'sign_coverage',
    'validation': release_summary, 'test': ..., 'merges': [...]}; answers: the answers per drive run already computed
    ({(route, start): predicted}), else the model runs."""
    found = []
    for run in drive_runs(dataset):
        route, positions = run['route'], run['positions']
        arrays = _sequence_arrays(dataset, run)
        entries = ramp_entries(arrays['labels'], data.CLASSES, arrays['lead'])
        partitions = dataset['split']['partitions'][route]
        measured = [(entry, merge_partition([partitions[positions[t]] for t in
                                             range(entry['ramp'], min(len(positions), entry['merge'] + LATE_SECONDS))]))
                    for entry in entries]
        if not any(partition for _, partition in measured):
            continue
        key = (route, positions[0])
        if answers is not None and key in answers:
            predicted = np.asarray(answers[key])
        else:
            predicted = _predict_sequence(torch, model, store, route, positions, arrays['speed'], device,
                                          sign=arrays['sign'], motion=arrays['motion'],
                                          board=arrays['board']).argmax(axis=1)
        held = debounced(predicted)
        age = signs_data.sign_ages(arrays['seconds'], dataset.get('sign_times', {}).get(route, []))
        joined = None  # the car's _joined_at: the last ramp's limit ended, measured or not
        for entry, partition in measured:
            result = comma_release(held, age, entry, data.CLASSES, joined)
            if result['release'] is not None:
                joined = entry['merge'] + result['release']
            if not partition:
                continue
            said = [data.CLASSES[int(v)] for v in held[entry['ramp']:min(len(held), entry['merge'] + LATE_SECONDS)]]
            found.append({'route': route, 'partition': partition, 'ramp_key': arrays['keys'][entry['ramp']],
                          'merge_key': arrays['keys'][entry['merge']], 'before': entry['before'],
                          'ramp_seconds': entry['merge'] - entry['ramp'], **result,
                          'held_answers': ''.join(ANSWER_LETTERS[name] for name in said)})
    out = {partition: release_summary([m for m in found if m['partition'] == partition])
           for partition in ('validation', 'test')}
    coverage = dataset.get('sign_coverage') or {}
    out.update(merges=found, sign_source=coverage.get('source'), sign_coverage=coverage)
    return out


def car_release(torch, model, store: FrameStore, dataset: dict, root: Path, device) -> dict | None:
    """release_evaluation with the signs the car's code confirmed (nnslr road-type-signs), the network's sign input
    too: the comma's rule as the car runs today. None when those signs were not read on every drive."""
    view = dict(dataset)
    coverage = attach_signs(view, root, 'car')
    if coverage['read'] < coverage['drives']:
        return None
    return release_evaluation(torch, model, store, view, device)


def _dataset_sha256(dataset: dict) -> str:
    digest = hashlib.sha256()
    for route in sorted(dataset['frames']):
        for frame in dataset['frames'][route]:
            digest.update(f"{route}/{frame['key']}:{frame['label']}:{frame['weight']}:{frame['speed_mps']}\n".encode())
    return digest.hexdigest()


def _junction_sha256(dataset: dict) -> str:
    digest = hashlib.sha256()
    for route in sorted(dataset['frames']):
        for frame in dataset['frames'][route]:
            digest.update(f"{route}/{frame['key']}:{int(frame.get('junction_known', False))}"
                          f"{int(frame.get('junction_in', False))}{int(frame.get('junction_passed', False))}\n".encode())
    return digest.hexdigest()


def _load_pictures(root: Path, dataset: dict, runs: list[dict], *, dry_run: bool, workers: int, log) -> dict:
    """{route: uint8 (frames, H, W, 3)}: the frame cache (built when missing), or for a dry run only the frames of its
    sequences read from the PNGs (nothing written)."""
    routes = sorted({run['route'] for run in runs})
    if not dry_run:
        paths = data.build_frame_cache(root, {r: dataset['frames'][r] for r in routes}, workers=workers, log=log)
        return {route: np.load(paths[route]) for route in routes}
    arrays = {}
    for route in routes:
        frames = dataset['frames'][route]
        array = np.zeros((len(frames), data.FRAME_HEIGHT, data.FRAME_WIDTH, 3), dtype=np.uint8)
        for run in runs:
            if run['route'] == route:
                for p in run['positions']:
                    array[p] = data.resized_frame(frames[p]['image'])
        arrays[route] = array
    return arrays


def _dry_runs(dataset: dict) -> list[dict]:
    """A dry run's few short sequences: two of train, one of validation and of test, those with the most labelled
    frames, 40 frames each from the first of them on."""
    def labelled(run: dict) -> list[int]:
        frames = dataset['frames'][run['route']]
        return [at for at, position in enumerate(run['positions']) if frames[position]['label'] >= 0]
    picked = []
    for partition, count in (('train', 2), ('validation', 1), ('test', 1)):
        runs = [run for run in dataset['sequences'] if run['partition'] == partition]
        for run in sorted(runs, key=lambda item: len(labelled(item)), reverse=True)[:count]:
            first = min(next(iter(labelled(run)), 0), max(len(run['positions']) - 40, 0))
            positions = run['positions'][first:first + 40]
            picked.append({**run, 'start': positions[0], 'positions': positions})
    return picked


def road_score(metrics: dict) -> float:
    """The road type's own measure of an epoch (sequence_metrics on validation): macro F1, plus ROAD_SCORE_MERGE times
    the share of the first minute after a merge right (debounced), less ROAD_SCORE_SWITCHES per false switch an hour
    (debounced). The junction head plays no part in it."""
    f1 = metrics['weight_1']['macro_f1'] or 0.0
    merge = (metrics.get('after_merge') or {}).get('accuracy_debounced')
    switches = metrics.get('false_switches_per_hour_debounced') or 0.0
    return round(f1 + ROAD_SCORE_MERGE * (merge or 0.0) - ROAD_SCORE_SWITCHES * switches, 4)


def _junction_key(operating: dict) -> tuple:
    return (operating.get('f1_2s') or 0.0, -(operating.get('false_per_hour') or 0.0))


def _tensors(torch, windows: list, device, rng: random.Random) -> dict:
    """A batch of WindowSampler windows as tensors on the device (the speed noised: _speed_noise)."""
    def stack(index):
        return torch.from_numpy(np.stack([w[index] for w in windows])).to(device)
    return {'speed': torch.from_numpy(np.stack([_speed_noise(rng, w[4]) for w in windows])).to(device),
            'sign': stack(5), 'labels': stack(2),
            'frame_weights': torch.from_numpy(np.stack([w[3] * w[6] for w in windows])).to(device),
            'targets': stack(7), 'target_weights': stack(8), 'motion': stack(9), 'board': stack(10)}


def _junction_loss(torch, crossing, targets, target_weights, balance):
    """Each junction output its own mean over the frames with junction labels, its positives weighted up."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(crossing.float(), targets, pos_weight=balance,
                                                               reduction='none')
    return ((bce * target_weights).sum(dim=(0, 1)) / target_weights.sum(dim=(0, 1)).clamp(min=1e-6)).sum()


def train_junction_stage(torch, model, store: FrameStore, dataset: dict, by_partition: dict, rng: random.Random,
                         device, *, epochs: int, patience: int, batch: int, window: int, burn: int, steps: int,
                         balance, name: str, log=_print) -> dict:
    """The junction head on its own, after the road type's epoch was chosen: everything else frozen (the backbone's
    batch norm as on the car), a junction GRU of its own copied from the road GRU and its head trained (the head alone
    without memory), JUNCTION_STAGE_SHARE of the windows around a "passed", the best epoch by the junction F1 at 2 s on
    validation (operating_point; on a tie the fewer false "passed"). Leaves the model with the best branch; returns
    {'history', 'best_epoch', 'validation' (its operating point)}."""
    import copy
    if model.memory == 'gru':
        model.separate_junctions()
        branch = [model.junction_gru, model.junction_head]
    else:
        branch = [model.junction_head]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trained = [parameter for module in branch for parameter in module.parameters()]
    for parameter in trained:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trained, lr=JUNCTION_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total = max(1, epochs * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    sampler = WindowSampler(dataset, by_partition['train'], rng, junctions=True, merge_share=0.0, focus_share=0.0,
                            junction_share=JUNCTION_STAGE_SHARE, around_labels=False)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history, best, best_epoch, best_state, since = [], None, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.eval()
        for module in branch:  # cuDNN runs a GRU's backward only in training mode; the rest stays as on the car
            module.train()
        started, losses = time.time(), []
        for _ in range(steps):
            warm = model.memory == 'gru' and rng.random() < 0.5
            windows = [sampler.window(window + (burn if warm else 0)) for _ in range(batch)]
            images = _augment(torch, store.batch([(route, positions) for route, positions, *_ in windows]))
            batch_of = _tensors(torch, windows, device, rng)
            with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                b, t = images.shape[:2]
                x = model.inputs(model.encode(images.flatten(0, 1)).view(b, t, -1), batch_of['speed'],
                                 batch_of['sign'], batch_of['motion'], batch_of['board'])
            junction_hidden, targets, target_weights = None, batch_of['targets'], batch_of['target_weights']
            if warm:
                with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                    junction_hidden = model.run(x[:, :burn])[3].float()
                x, targets, target_weights = x[:, burn:], targets[:, burn:], target_weights[:, burn:]
            with torch.autocast(device.type, enabled=use_amp):
                crossing = model.run(x.detach(), None, junction_hidden)[2]
            loss = _junction_loss(torch, crossing, targets, target_weights, balance)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trained, 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        chosen = operating_point(evaluate(torch, model, store, dataset, by_partition['validation'], device)
                                 ['metrics']['junctions'])
        point = {'epoch': epoch, 'train_loss': round(float(np.mean(losses)), 4),
                 'val_junction_f1': chosen.get('f1_2s') or 0.0, 'val_false_passed_per_hour': chosen.get('false_per_hour'),
                 'decode': chosen.get('decode'), 'threshold': chosen.get('threshold'),
                 'seconds': round(time.time() - started, 1)}
        history.append(point)
        log(f"{name} junction epoch {epoch}/{epochs} train_loss {point['train_loss']:.4f} val_junction_f1 "
            f"{point['val_junction_f1']:.4f} val_false_passed_per_hour {point['val_false_passed_per_hour']} "
            f"{int(point['seconds'])}s")
        if best is None or _junction_key(chosen) > best:
            best, best_epoch, since = _junction_key(chosen), epoch, 0
            best_state = [copy.deepcopy(module.state_dict()) for module in branch]
        else:
            since += 1
            if since >= patience:
                log(f'{name} junction stage stops after epoch {epoch}: no better validation junction F1 for '
                    f'{patience} epochs')
                break
    for module, state in zip(branch, best_state or []):
        module.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    validation = operating_point(evaluate(torch, model, store, dataset, by_partition['validation'], device)
                                 ['metrics']['junctions'])
    return {'history': history, 'best_epoch': best_epoch, 'validation': validation}


def _passage_turns(arrays: dict) -> list[dict]:
    """The labelled "passed" of a sequence with the car's heading change through the intersection: from 2 s before
    its first in-junction second to the second after the "passed" (degrees, positive left; None when the yaw is
    unknown there)."""
    labels = arrays['junction']
    yaw = np.where(arrays['motion'][:, 2] > 0, arrays['motion'][:, 0] * motion_data.YAW_SCALE, np.nan)
    found = []
    for t in np.flatnonzero((labels[:, 1] > 0) & arrays['junction_known']):
        start = t
        while start > 0 and labels[start - 1, 0] > 0:
            start -= 1
        part = yaw[max(0, start - 2):min(len(yaw), t + 2)]
        found.append({'t': int(t), 'degrees': float(np.nansum(part)) if np.isfinite(part).any() else None})
    return found


TURN_GRID = ((45.0, 40.0), (60.0, 25.0), (60.0, 40.0), (60.0, 60.0), (90.0, 40.0))


def turn_rule(dataset: dict, runs: list[dict], answers: list[dict] | None = None, *, degrees: float = TURN_DEGREES,
              radius_m: float = TURN_RADIUS_M, operating: dict | None = None) -> dict:
    """The plain rule without the network (road_type_metrics.turn_events: a heading change of at least degrees on a
    radius of at most radius_m, the event the first second after it) scored as a "passed" detector with
    junction_metrics on the same sequences, its 'grid' of other angles and radii, and the labelled intersections by
    how much the car turned in them ('passages'). answers: evaluate's answers of the same runs (the network's road
    type where the map has none, for town and outside; and with operating, which intersections the network caught,
    turned or straight)."""
    by_run = []
    for index, run in enumerate(runs):
        arrays = _sequence_arrays(dataset, run)
        yaw = np.where(arrays['motion'][:, 2] > 0, arrays['motion'][:, 0] * motion_data.YAW_SCALE, np.nan)
        speed = np.where(arrays['speed'][:, 1] > 0, arrays['speed'][:, 0] * data.SPEED_SCALE, np.nan)
        answer = answers[index] if answers else None
        predicted = (np.asarray(answer['predicted']) if answer else
                     np.where(arrays['labels'] >= 0, arrays['labels'], data.CLASSES.index('rural')))
        by_run.append((arrays, yaw, speed, predicted, answer))

    def score(angle: float, radius: float) -> dict:
        found = []
        for arrays, yaw, speed, predicted, _ in by_run:
            pulse = np.zeros(len(yaw))
            pulse[turn_events(yaw, speed, degrees=angle, radius_m=radius)] = 1.0
            found.append({'seconds': arrays['seconds'], 'p_in': np.zeros(len(yaw)), 'p_passed': pulse,
                          'in': arrays['junction'][:, 0] > 0, 'passed': arrays['junction'][:, 1] > 0,
                          'known': arrays['junction_known'], 'ignore': arrays['junction_ignore'],
                          'labels': arrays['labels'], 'predicted': predicted, 'speed': arrays['speed_mps']})
        return junction_metrics(found, data.CLASSES, thresholds=(0.5,))

    main = score(degrees, radius_m)
    passages = {'passed': 0, 'yaw_known': 0, 'turned_30': 0, 'turned_60': 0}
    caught = {kind: {'passed': 0, 'caught_2s': 0, 'caught_4s': 0} for kind in ('turned', 'straight')}
    for arrays, yaw, speed, predicted, answer in by_run:
        turned = _passage_turns(arrays)
        passages['passed'] += len(turned)
        known = [p for p in turned if p['degrees'] is not None]
        passages['yaw_known'] += len(known)
        passages['turned_30'] += sum(1 for p in known if abs(p['degrees']) >= 30)
        passages['turned_60'] += sum(1 for p in known if abs(p['degrees']) >= 60)
        if not (answer and operating and 'junction' in answer):
            continue
        crossing = np.asarray(answer['junction'])
        seconds = arrays['seconds']
        events = [float(seconds[t]) for t in junction_events(crossing[:, 1], crossing[:, 0], operating['threshold'],
                                                              operating['decode'], seconds=seconds)]
        for passage in known:
            kind = 'turned' if abs(passage['degrees']) >= degrees else 'straight'
            caught[kind]['passed'] += 1
            for tolerance in JUNCTION_TOLERANCES:
                caught[kind][f'caught_{tolerance:g}s'] += bool(match_events(events, [float(seconds[passage['t']])],
                                                                            tolerance))
    out = {'rule': {'degrees': degrees, 'radius_m': radius_m, 'min_dps': TURN_MIN_DPS},
           'metrics': main, 'numbers': main['decodes']['passed']['0.5'], 'passages': passages,
           'grid': [{'degrees': angle, 'radius_m': radius, **score(angle, radius)['decodes']['passed']['0.5']}
                    for angle, radius in TURN_GRID]}
    if operating and any(c['passed'] for c in caught.values()):
        out['network_by_turn'] = {'decode': operating['decode'], 'threshold': operating['threshold'], **caught}
    return out


def motion_unknown(torch, model, store: FrameStore, dataset: dict, runs: list[dict], device) -> dict:
    """The network on runs with the car's motion all unknown (a car or a log without it), as evaluate's metrics: how
    much the road type and the junction head lean on the motion."""
    known = dataset.get('motion', {})
    dataset['motion'] = {route: np.tile(np.float32(NO_MOTION), (len(rows), 1)) for route, rows in known.items()}
    try:
        return evaluate(torch, model, store, dataset, runs, device)['metrics']
    finally:
        dataset['motion'] = known


def motion_unknown_text(result: dict) -> str:
    """The test with the motion known and unknown, side by side: macro F1, after a merge, false switches (debounced)
    and the junction head at the car's operating point."""
    known, blind = result.get('test') or {}, result.get('test_motion_unknown') or {}
    if not blind:
        return ''
    parts = [f"macro F1 {_numbers(known['weight_1']['macro_f1'])} -> {_numbers(blind['weight_1']['macro_f1'])}",
             f"after a merge {_numbers(known['after_merge']['accuracy_debounced'])} -> "
             f"{_numbers(blind['after_merge']['accuracy_debounced'])}",
             f"false switches/h {_numbers(known['false_switches_per_hour_debounced'])} -> "
             f"{_numbers(blind['false_switches_per_hour_debounced'])}"]
    operating = result.get('junction_operating')
    if operating and 'junctions' in known and 'junctions' in blind:
        at = [part['junctions']['decodes'][operating['decode']][f"{operating['threshold']:g}"]
              for part in (known, blind)]
        parts.append(f"junctions ({operating['decode']} >= {operating['threshold']:g}) recall 2 s "
                     f"{_numbers(at[0]['recall_2s'])} -> {_numbers(at[1]['recall_2s'])}, false "
                     f"{_numbers(at[0]['false_per_hour'])}/h -> {_numbers(at[1]['false_per_hour'])}/h (town "
                     f"{_numbers(at[0]['town']['per_hour'])} -> {_numbers(at[1]['town']['per_hour'])})")
    return '; '.join(parts)


def turn_text(turn: dict) -> str:
    """One line of the plain turn rule: 'turn >= 60 deg, radius <= 40 m: ...junction_text'."""
    rule = turn['rule']
    return (f"turn >= {rule['degrees']:g} deg on a radius <= {rule['radius_m']:g} m: "
            + junction_text(turn['metrics'], turn['numbers']))


def road_type_start(torch, model, path: Path, memory: str, split: dict) -> dict:
    """The weights of the same network of an earlier run (nnslr train-road-type --init-from) in a fresh one: every
    layer of the same name and shape, the rows of the road-type head by class name (a class the start lacks keeps the
    fresh row); a layer the start has in another shape (a new input of the memory, the junction head it lacks) stays
    fresh, but for the board input added to a start without it: the start's columns of the memory's input as they
    are, the board's two zeros (widened), so the network starts as the start. The image backbone must be the start's
    whole. Returns what result.json records of it: its file and sha256, run, epoch, split, the layers left fresh and
    widened, whether it was tested on the same drives as this one (its split.json beside it; None when missing) and its
    label policy (None: trained before road_type_data.LABEL_POLICY, the map's proposals among its labels)."""
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or saved.get('task') != 'road_type' or saved.get('memory') != memory:
        raise ValueError(f'the start is not a road-type network {"with" if memory == "gru" else "without"} memory: {path}')
    weights = saved['state_dict']
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    head = 'head.' if memory == 'gru' else 'head.2.'
    rows = {label: row for row, label in enumerate(saved.get('classes') or [])}
    # The memory's input layers (gru, junction_gru; the first layer of the heads without memory) take the board's
    # columns last: a start with the same sign and motion inputs and no board fills every other column.
    widen = (getattr(model, 'boards', False) and not saved.get('boards')
             and bool(saved.get('signs')) == bool(model.signs) and bool(saved.get('motion')) == bool(model.motion))
    inputs = ('gru.weight_ih_l0', 'junction_gru.weight_ih_l0', 'head.0.weight', 'junction_head.0.weight')
    fresh, widened = [], []
    for key, current in state.items():
        if key in (head + 'weight', head + 'bias') and key in weights:
            for row, label in enumerate(data.CLASSES):
                if label in rows:
                    current[row] = weights[key][rows[label]]
            if any(label not in rows for label in data.CLASSES):
                fresh.append(key + ' (' + ', '.join(label for label in data.CLASSES if label not in rows) + ')')
        elif key in weights and tuple(weights[key].shape) == tuple(current.shape):
            state[key] = weights[key]
        elif (widen and key in inputs and key in weights and weights[key].shape[0] == current.shape[0]
              and current.shape[1] - weights[key].shape[1] == BOARD_INPUTS):
            current.zero_()
            current[:, :weights[key].shape[1]] = weights[key]
            widened.append(key)
        else:
            fresh.append(key)
    if any(key.startswith('backbone.') for key in fresh):
        raise ValueError(f'the start has another image backbone: {path}')
    model.load_state_dict(state, strict=True)
    try:
        tested = json.loads((path.parent / 'split.json').read_text(encoding='utf-8')).get('test_routes')
    except (OSError, ValueError, AttributeError):
        tested = None
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'run': path.parents[2].name if len(path.parents) > 2 else None, 'epoch': saved.get('epoch'),
            'split_sha256': saved.get('split_sha256'), 'fresh': fresh, 'widened': widened,
            'label_policy': (saved.get('config') or {}).get('label_policy'),
            'same_test_drives': None if not isinstance(tested, list)
            else sorted(tested) == sorted(split.get('test_routes') or [])}


def train_road_type(data_root: Path, run_dir: Path, *, memory: str = 'gru', epochs: int = EPOCHS,
                    patience: int = PATIENCE, batch: int = BATCH, device: str | None = None, dry_run: bool = False,
                    pretrained: bool | None = None, workers: int = 8, seed: int = 0, signs: bool = True,
                    sign_source: str = signs_data.DEFAULT_SOURCE, junctions: bool | None = None,
                    motion: bool | None = None, junction_epochs: int = JUNCTION_EPOCHS,
                    init_from: Path | None = None, boards: bool | None = None, log=_print) -> dict:
    """Train the road-type network on the drives with road types and measure it on validation and test (see the
    module's text). Writes into run_dir: split.json, dataset.json, road-type-best.pt, result.json and the answers on
    test and validation (answers-test.json, answers-validation.json). Too few confirmed frames stop it before anything
    is written (road_type_data.missing_labels), a dry run too. A dry run: the CPU, a few short sequences, one
    epoch of two steps, no ImageNet download, nothing written outside run_dir. junctions: with the junction head (None:
    when nnslr junctions has labelled the drives); motion: with the car's own motion as input (None: when nnslr
    car-motion has read every drive; the speed input is then carState.vEgo, else the GPS speed). The road type's epoch
    is chosen by road_score on validation, and only it stops the training early; the junction head then gets its own
    stage (train_junction_stage, junction_epochs; 0: the shared head of the joint training as it is). init_from: the
    road-type-best.pt of the same network of an earlier run to go on from (road_type_start), else ImageNet. boards:
    with the board input (the green boards towards a motorway the car's code confirmed, nnslr road-type-signs; None:
    when every drive was read with them and they were confirmed on MIN_BOARD_DRIVES drives at least)."""
    data_root, run_dir = Path(data_root), Path(run_dir)
    dataset = data.load_dataset(data_root)
    missing = data.missing_labels(dataset['summary'])
    if missing:
        raise ValueError('too few road types confirmed by a person to train, and a proposal of the map is never a label: '
                         + '; '.join(missing) + '. A person confirmed them in Data › Roads, until 7 Nov 2026: that page '
                         'is gone, no more can be added.')
    torch = _torch()
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ('cpu' if dry_run else 'cuda'))
    pretrained = (not dry_run) if pretrained is None else pretrained
    coverage = attach_signs(dataset, data_root, sign_source)
    if signs and coverage['read'] < coverage['drives'] and not dry_run:
        raise ValueError(f"the motorway signs were read on {coverage['read']} of {coverage['drives']} drives: run "
                         'nnslr road-type-signs first')
    motion_read = sum(1 for route in dataset['frames'] if motion_data.load_motion(data_root, route) is not None)
    every_drive = bool(dataset['frames']) and motion_read == len(dataset['frames'])
    requested_motion = motion
    if motion is None:
        motion = every_drive
    if motion and not every_drive and not dry_run:
        raise ValueError(f"the car's motion was read on {motion_read} of {len(dataset['frames'])} drives: run nnslr "
                         'car-motion --all first (or train with --no-motion)')
    motion_coverage = attach_motion(dataset, data_root, v_ego=every_drive)
    board_coverage = attach_boards(dataset, data_root)
    boards_read = board_coverage['read'] == board_coverage['drives']
    if boards is None:
        boards = boards_read and board_coverage['drives_with_boards'] >= MIN_BOARD_DRIVES
    if boards and not dry_run:
        if not boards_read:
            raise ValueError(f"the green boards were read on {board_coverage['read']} of {board_coverage['drives']} "
                             'drives: run nnslr road-type-signs again (or train with --no-boards)')
        if board_coverage['drives_with_boards'] < MIN_BOARD_DRIVES:
            raise ValueError(f"the car's code confirmed green boards on {board_coverage['drives_with_boards']} drives "
                             f'(at least {MIN_BOARD_DRIVES}): the classifier does not read them yet; train it with more '
                             'boxed boards (or train with --no-boards)')
    runs = _dry_runs(dataset) if dry_run else dataset['sequences']
    by_partition = {name: [run for run in runs if run['partition'] == name] for name in data.PARTITIONS}
    labelled = sum(1 for frames in dataset['frames'].values() for f in frames if f.get('junction_known'))
    if junctions is None:
        junctions = labelled > 0
    if junctions and not labelled and not dry_run:
        raise ValueError('no junction labels: run nnslr junctions --all first (or train with --no-junctions)')
    if not by_partition['train'] or not by_partition['validation']:
        raise ValueError('the road types give no train or validation sequence: run nnslr road-types on more drives')
    (run_dir / 'split.json').write_text(json.dumps({**dataset['split'], 'sha256': dataset['split_sha256']},
                                                   sort_keys=True) + '\n', encoding='utf-8')
    (run_dir / 'dataset.json').write_text(json.dumps(dataset['summary'], indent=2, sort_keys=True) + '\n',
                                          encoding='utf-8')
    # A dry run's lines start with "dry run": the panel draws the curves of the real epochs only (job_progress).
    name = ('dry run ' if dry_run else '') + network_name(memory, signs, requested_motion)
    log(f'{name}: {len(by_partition["train"])} train, {len(by_partition["validation"])} validation, '
        f'{len(by_partition["test"])} test sequences; split {dataset["split_sha256"][:12]}; '
        + (f"sign input ({sign_source}): {coverage['signs']} motorway signs on {coverage['drives_with_signs']} of "
           f"{coverage['drives']} drives" if signs else 'no sign input')
        + (f"; the car's motion of {motion_read} drives (yaw known {motion_coverage['yaw'] / max(1, motion_coverage['frames']):.1%} "
           f"of the frames)" if motion else "; without the car's motion")
        + (f"; board input: {board_coverage['boards']} green boards on {board_coverage['drives_with_boards']} drives"
           if boards else '; no board input')
        + f"; speed from {motion_coverage['speed_source']}")
    pictures = runs if dry_run else runs + drive_runs(dataset)
    store = FrameStore(_load_pictures(data_root, dataset, pictures, dry_run=dry_run, workers=workers, log=log), device)
    model = build_road_type_net(memory, pretrained=pretrained, signs=signs, junctions=junctions, motion=motion,
                                boards=boards)
    start = None
    if init_from is not None:
        start = road_type_start(torch, model, Path(init_from), memory, dataset['split'])
        log(f"{name} starts from {start['run']} (epoch {start['epoch']}): every layer"
            + (f" but {', '.join(start['fresh'])}" if start['fresh'] else '')
            + (f"; the board's columns added to {', '.join(start['widened'])}" if start['widened'] else '')
            + ('; tested on other drives' if start['same_test_drives'] is False else '')
            + ("; its weights learned from the map's proposals too" if start['label_policy'] != data.LABEL_POLICY else ''))
    model = model.to(device)
    weights_of_class = torch.tensor(class_weights(dataset, by_partition['train']), device=device)
    junction_balance = torch.tensor(junction_pos_weights(dataset, by_partition['train']), device=device)
    backbone = list(model.backbone.parameters())
    others = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': BACKBONE_LEARNING_RATE},
                                   {'params': others, 'lr': LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    sampler = WindowSampler(dataset, by_partition['train'], rng, junctions=junctions)
    train_frames = sum(len(run['positions']) for run in by_partition['train'])
    steps = 2 if dry_run else max(10, math.ceil(train_frames / (batch * WINDOW)))
    window = 8 if dry_run else WINDOW
    burn = 4 if dry_run else BURN_IN
    epochs = 1 if dry_run else epochs
    junction_epochs = min(1, junction_epochs) if dry_run else junction_epochs
    batch = 2 if dry_run else batch
    total_steps, warmup = epochs * steps, steps * 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history, best, best_epoch, since = [], None, 0, 0
    config = {'memory': memory, 'epochs': epochs, 'patience': patience, 'batch': batch, 'steps_per_epoch': steps,
              'window': window, 'burn_in': burn, 'focus_share': FOCUS_SHARE, 'drop_frame': DROP_FRAME,
              'no_speed_share': NO_SPEED_SHARE, 'merge_share': MERGE_SHARE, 'merge_weight': MERGE_WEIGHT,
              'signs': signs, 'sign_source': sign_source, 'sign_drop_share': SIGN_DROP_SHARE,
              'sign_fake_share': SIGN_FAKE_SHARE, 'sign_jitter': SIGN_JITTER, 'sign_coverage': coverage,
              'learning_rate': LEARNING_RATE,
              'backbone_learning_rate': BACKBONE_LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'seed': seed,
              'pretrained': pretrained, 'width': data.FRAME_WIDTH, 'height': data.FRAME_HEIGHT,
              'features': FEATURES, 'hidden': HIDDEN, 'label_policy': data.LABEL_POLICY,
              'class_weights': [round(float(w), 4) for w in weights_of_class],
              'junctions': junctions, 'motion': motion, 'motion_fields': list(motion_data.MOTION_FIELDS) if motion else [],
              'motion_drop_share': MOTION_DROP_SHARE, 'motion_part_share': MOTION_PART_SHARE,
              'motion_coverage': motion_coverage, 'speed_source': motion_coverage['speed_source'],
              'boards': boards, 'board_coverage': board_coverage,
              'selection': {'road': 'road_score: macro F1 + %g x after-merge (debounced) - %g x false switches/h '
                                    '(debounced), on validation' % (ROAD_SCORE_MERGE, ROAD_SCORE_SWITCHES),
                            'junction': 'its own stage: the junction F1 at 2 s on validation'}}
    if junctions:
        config.update(junction_share=JUNCTION_SHARE, junction_loss=JUNCTION_LOSS,
                      junction_dont_care=JUNCTION_DONT_CARE,
                      junction_pos_weights=[round(float(w), 3) for w in junction_balance],
                      junction_rules=junction_rules(), junction_epochs=junction_epochs,
                      junction_patience=JUNCTION_PATIENCE, junction_learning_rate=JUNCTION_LEARNING_RATE,
                      junction_stage_share=JUNCTION_STAGE_SHARE)

    def save(epoch: int) -> None:
        torch.save({'task': 'road_type', 'architecture': f'mobilenet_v3_small+{memory}', 'memory': memory,
                    'signs': signs, 'sign_source': sign_source, 'classes': list(data.CLASSES),
                    'junctions': junctions, 'junction_outputs': list(JUNCTION_OUTPUTS) if junctions else [],
                    'junction_branch': model.junction_branch, 'motion': motion,
                    'motion_fields': list(motion_data.MOTION_FIELDS) if motion else [],
                    'speed_source': motion_coverage['speed_source'], 'boards': boards,
                    'board_signs': [BOARD_CLASS] if boards else [],
                    'state_dict': model.state_dict(), 'epoch': epoch,
                    'config': config, 'split_sha256': dataset['split_sha256'],
                    'dataset_sha256': _dataset_sha256(dataset),
                    'junction_sha256': _junction_sha256(dataset) if junctions else None}, run_dir / CHECKPOINT)

    for epoch in range(1, epochs + 1):
        model.train()
        started, losses = time.time(), []
        for _ in range(steps):
            length = window
            warm = memory == 'gru' and rng.random() < 0.5
            windows = [sampler.window(length + (burn if warm else 0)) for _ in range(batch)]
            images = store.batch([(route, positions) for route, positions, *_ in windows])
            images = _augment(torch, images)
            batch_of = _tensors(torch, windows, device, rng)
            speed, sign, labels, frame_weights = (batch_of['speed'], batch_of['sign'], batch_of['labels'],
                                                  batch_of['frame_weights'])
            targets, target_weights, motion_in = batch_of['targets'], batch_of['target_weights'], batch_of['motion']
            board = batch_of['board']
            hidden = None
            if warm:
                with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                    hidden = model(images[:, :burn], speed[:, :burn], None, sign[:, :burn], motion_in[:, :burn],
                                   board=board[:, :burn])[1]
                hidden = hidden.float()
                images, speed, sign, labels, frame_weights, targets, target_weights, motion_in, board = (
                    images[:, burn:], speed[:, burn:], sign[:, burn:], labels[:, burn:], frame_weights[:, burn:],
                    targets[:, burn:], target_weights[:, burn:], motion_in[:, burn:], board[:, burn:])
            with torch.autocast(device.type, enabled=use_amp):
                logits, _, crossing, _ = model(images, speed, hidden, sign, motion_in, board=board)
            known = labels >= 0
            per_frame = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.clamp(min=0).flatten(),
                                                          reduction='none').view_as(frame_weights)
            weight = frame_weights * weights_of_class[labels.clamp(min=0)] * known
            loss = (per_frame * weight).sum() / weight.sum().clamp(min=1e-6)
            if crossing is not None:
                loss = loss + JUNCTION_LOSS * _junction_loss(torch, crossing, targets, target_weights, junction_balance)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        validation = evaluate(torch, model, store, dataset, by_partition['validation'], device)['metrics']
        road_f1 = validation['weight_1']['macro_f1'] or 0.0
        score = road_score(validation)
        point = {'epoch': epoch, 'train_loss': round(float(np.mean(losses)), 4), 'val_macro_f1': road_f1,
                 'val_accuracy': validation['weight_1']['accuracy'],
                 'val_switches_per_hour': validation['false_switches_per_hour'],
                 'val_switches_per_hour_debounced': validation['false_switches_per_hour_debounced'],
                 'val_after_merge_debounced': (validation.get('after_merge') or {}).get('accuracy_debounced'),
                 'val_road_score': score, 'seconds': round(time.time() - started, 1)}
        if junctions:
            chosen = operating_point(validation['junctions'])
            point.update(val_junction_f1=chosen.get('f1_2s') or 0.0,
                         val_false_passed_per_hour=chosen.get('false_per_hour'))
        history.append(point)
        log(f"{name} epoch {epoch}/{epochs} train_loss {point['train_loss']:.4f} val_macro_f1 {road_f1:.4f} "
            f"val_accuracy {point['val_accuracy'] or 0:.4f} val_road_score {score:.4f} "
            + (f"val_junction_f1 {point['val_junction_f1']:.4f} " if junctions else '')
            + f"{int(point['seconds'])}s")
        if best is None or score > best:
            best, best_epoch, since = score, epoch, 0
            save(epoch)
        else:
            since += 1
            if since >= patience:
                log(f'{name} stops early after epoch {epoch}: no better validation road score for {patience} epochs')
                break
    saved = torch.load(run_dir / CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(saved['state_dict'])
    stage = None
    if junctions:
        import copy
        shared = operating_point(evaluate(torch, model, store, dataset, by_partition['validation'], device)
                                 ['metrics']['junctions'])
        stage = {'shared_validation': shared, 'branch': 'shared', 'best_epoch': None, 'history': []}
        if junction_epochs > 0:
            kept = copy.deepcopy(model.junction_head.state_dict())
            found = train_junction_stage(torch, model, store, dataset, by_partition, rng, device,
                                         epochs=junction_epochs, patience=JUNCTION_PATIENCE, batch=batch,
                                         window=window, burn=burn, steps=steps, balance=junction_balance, name=name,
                                         log=log)
            stage.update(history=found['history'], best_epoch=found['best_epoch'],
                         stage_validation=found['validation'])
            if _junction_key(found['validation']) > _junction_key(shared):
                stage['branch'] = model.junction_branch if memory == 'gru' else 'shared (its own stage)'
            else:
                # The shared head of the joint training is as good on validation: one memory on the car.
                model.junction_head.load_state_dict(kept)
                if memory == 'gru':
                    del model.junction_gru
                    model.junction_branch = 'shared'
        log(f"{name}: junction head {stage['branch']} (validation F1 at 2 s: shared head "
            f"{_numbers(shared.get('f1_2s'))}, own stage {_numbers((stage.get('stage_validation') or {}).get('f1_2s'))}"
            f", best stage epoch {stage['best_epoch']})")
        save(best_epoch)
    result = {'task': 'road_type', 'memory': memory, 'signs': signs, 'sign_source': sign_source, 'name': name,
              'dry_run': dry_run, 'junctions': junctions, 'motion': motion, 'boards': boards,
              'speed_source': motion_coverage['speed_source'], 'junction_branch': model.junction_branch,
              'architecture': saved['architecture'], 'classes': list(data.CLASSES), 'best_epoch': best_epoch,
              'history': history, 'config': config, 'parameters': count_parameters(model),
              'flops_per_step': step_flops(model), 'split_sha256': dataset['split_sha256'],
              'dataset_sha256': saved['dataset_sha256'], 'dataset': dataset['summary'], 'start': start,
              'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    if stage is not None:
        result['junction_stage'] = stage
    answers_of = {}
    for partition in ('validation', 'test'):
        measured = evaluate(torch, model, store, dataset, by_partition[partition], device, smooth=memory == 'none')
        result[partition] = measured['metrics']
        answers_of[partition] = measured['answers']
        if 'smoothed' in measured:
            result[f'{partition}_smoothed'] = measured['smoothed']
        (run_dir / f'answers-{partition}.json').write_text(json.dumps(measured['answers']) + '\n', encoding='utf-8')
        if junctions and partition == 'validation':
            result['junction_operating'] = operating_point(measured['metrics']['junctions'])
        if junctions and partition == 'test' and not dry_run:
            result['junction_needed'] = needed_junctions(dataset, measured['answers'], data_root,
                                                         result['junction_operating'])
    if junctions:
        operating = result['junction_operating']
        on_test = result['test']['junctions']['decodes'][operating['decode']][f"{operating['threshold']:g}"]
        log(f"{name}: junctions on test ({operating['decode']} >= {operating['threshold']:g}, chosen on validation): "
            + junction_text(result['test']['junctions'], on_test))
        if every_drive:
            result['turn_rule'] = {partition: turn_rule(dataset, by_partition[partition], answers_of[partition],
                                                        operating=operating)
                                   for partition in ('validation', 'test')}
            log(f"{name}: the plain turn rule on test, without the network: " + turn_text(result['turn_rule']['test']))
    if motion and not dry_run:
        result['test_motion_unknown'] = motion_unknown(torch, model, store, dataset, by_partition['test'], device)
        log(f"{name}: on test with the car's motion unknown: " + motion_unknown_text(result))
    if not dry_run:
        result['release'] = release_evaluation(torch, model, store, dataset, device)
        log(f"{name}: the comma's rule on test ({sign_source} signs): " + release_text(result['release']['test']))
        if sign_source != 'car':
            today = car_release(torch, model, store, dataset, data_root, device)
            if today:
                result['release_car'] = today
                log(f"{name}: the comma's rule on test (the car's signs): " + release_text(today['test']))
    result['step_seconds'] = _step_seconds(torch, model, device)
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    log(f"{name}: best epoch {best_epoch}, test macro F1 {result['test']['weight_1']['macro_f1']}, "
        f"{result['parameters']:,} parameters, {result['flops_per_step'] / 1e6:.0f} MFLOPs a step")
    return result


def _step_seconds(torch, model, device) -> dict:
    """The time of one step at batch 1 on this machine's device (warm), a hint only: the car's GPU is another."""
    step = step_module(model).to(device)
    example = [t.to(device) for t in step_example(model, torch)]
    example[0] = torch.rand_like(example[0])
    with torch.no_grad():
        for _ in range(3):
            step(*example)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            step(*example)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    return {'device': str(device), 'milliseconds': round((time.perf_counter() - started) / 20 * 1000, 2)}


def load_road_type(checkpoint: Path, device: str = 'cpu'):
    """(model in eval mode, the saved checkpoint dict)."""
    torch = _torch()
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if saved.get('task') != 'road_type':
        raise ValueError(f'{checkpoint} is not a road-type checkpoint')
    model = build_road_type_net(saved['memory'], pretrained=False, signs=bool(saved.get('signs', False)),
                                junctions=bool(saved.get('junctions', False)), motion=bool(saved.get('motion', False)),
                                junction_branch=saved.get('junction_branch') or 'shared',
                                classes=len(saved.get('classes') or data.CLASSES), boards=bool(saved.get('boards', False)))
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved


def every_drive_moved(root: Path, dataset: dict) -> bool:
    """Whether nnslr car-motion has read every drive of the dataset."""
    return bool(dataset['frames']) and all(motion_data.load_motion(root, route) is not None
                                           for route in dataset['frames'])


def release_of_checkpoint(data_root: Path, checkpoint: Path, output: Path | None = None, device: str | None = None,
                          sign_source: str = signs_data.DEFAULT_SOURCE) -> dict:
    """The comma's rule (release_evaluation) and the measure per class (with the confusion) on validation and test for
    any road-type checkpoint, the first version too: {'checkpoint', 'memory', 'signs', 'release' (with the signs of
    sign_source), 'release_car' (with the car's, when read), 'metrics': {partition: sequence_metrics}}, written to
    output when given. With a junction head also its operating point (from the result.json beside the checkpoint, else
    chosen on validation), the intersections the car needed and, when the car's motion was read, the passages turned
    and straight it caught (turn_rule). The speed input is the one the checkpoint was trained with (GPS or vEgo)."""
    torch = _torch()
    data_root = Path(data_root)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, saved = load_road_type(checkpoint, device=str(device))
    dataset = data.load_dataset(data_root)
    attach_signs(dataset, data_root, sign_source)
    attach_boards(dataset, data_root)
    moved = every_drive_moved(data_root, dataset)
    attach_motion(dataset, data_root, v_ego=saved.get('speed_source') == 'v_ego')
    store = FrameStore(_load_pictures(data_root, dataset, dataset['sequences'] + drive_runs(dataset), dry_run=False,
                                      workers=8, log=_print), device)
    result = {'checkpoint': str(checkpoint), 'memory': saved['memory'], 'signs': bool(saved.get('signs', False)),
              'junctions': bool(model.junctions), 'motion': bool(model.motion), 'boards': bool(model.boards),
              'speed_source': saved.get('speed_source', 'gps'), 'junction_branch': model.junction_branch,
              'epoch': saved.get('epoch'), 'split_sha256': dataset['split_sha256'], 'metrics': {},
              'release': release_evaluation(torch, model, store, dataset, device)}
    if sign_source != 'car':
        today = car_release(torch, model, store, dataset, data_root, device)
        if today:
            result['release_car'] = today
    answers = {}
    for partition in ('validation', 'test'):
        runs = [run for run in dataset['sequences'] if run['partition'] == partition]
        measured = evaluate(torch, model, store, dataset, runs, device)
        result['metrics'][partition], answers[partition] = measured['metrics'], measured['answers']
    if model.junctions:
        beside = Path(checkpoint).parent / 'result.json'
        operating = (json.loads(beside.read_text(encoding='utf-8')).get('junction_operating')
                     if beside.is_file() else None)
        operating = operating or operating_point(result['metrics']['validation']['junctions'])
        result['junction_operating'] = operating
        result['junction_needed'] = needed_junctions(dataset, answers['test'], data_root, operating)
        if moved:
            runs = [run for run in dataset['sequences'] if run['partition'] == 'test']
            result['turn_rule'] = {'test': turn_rule(dataset, runs, answers['test'], operating=operating)}
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return result


def release_text(summary: dict) -> str:
    """One line of release_summary: 'N merges: held H (ramp R, sign S, off the carriageway F); on the ramp O, in time I (median M s), late
    L, not held X'."""
    median = summary['median_release']
    return (f"{summary['merges']} merges: held {summary['held']} (ramp {summary['held_by_ramp']}, sign "
            f"{summary['held_by_sign']}, off the carriageway {summary.get('held_by_off', 0)}); released on the ramp {summary['on_ramp']}, in time {summary['in_time']}"
            + (f' (median {median:g} s after the merge)' if median is not None else '')
            + f", late {summary['late']}, never held {summary['not_held']}")


def _numbers(value: float) -> str:
    return '-' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)


COMPARED = (('gru', 'gru'), ('gru-nomotion', 'gru-nomotion'), ('gru-nosign', 'gru-nosign'), ('frame', 'frame'))
# The comma's rule with the signs of the map (where 4.01 stands: what the sign input gives once the car reads it) and
# with the signs the car's code confirms today.
_SCENARIOS = (('release', 'With the sign at the start of every entry ramp of the map (the car reading every 4.01):'),
              ('release_car', "With the signs the car's code confirms today:"))
# <run>/road-type/baseline/release.json (and baseline-2/, baseline-3/...): earlier versions measured by nnslr
# release-road-type.
BASELINE = 'baseline'


def _variant_label(result: dict) -> str:
    if result.get('label'):
        return result['label']
    motion = ' and motion' if result.get('motion') else ''
    if result.get('memory') == 'none':
        return 'without memory' + (f' (sign{motion})' if result.get('signs') else f' ({motion[5:]})' if motion else '')
    if not result.get('signs'):
        return 'with memory, no sign' + (', motion' if motion else '')
    return f'with memory, sign{motion}' if motion else 'with memory and the sign, no motion'


def baselines(out_dir: Path) -> list[dict]:
    """The earlier versions measured beside a run (road-type/baseline*/release.json), each with its 'label'
    ('baseline <its run>')."""
    found = []
    for folder in sorted(Path(out_dir).glob(BASELINE + '*'), key=lambda f: (len(f.name), f.name)):
        path = folder / 'release.json'
        if path.is_file():
            measured = json.loads(path.read_text(encoding='utf-8'))
            measured['label'] = f"baseline {Path(measured['checkpoint']).parent.parent.parent.name}"
            measured['folder'] = folder.name
            found.append(measured)
    return found


def _as_result(baseline: dict) -> dict:
    """A baseline's release.json in the shape of a result.json (for the junction and turn lines)."""
    return {'label': baseline['label'], 'memory': baseline['memory'], 'signs': baseline['signs'],
            'junctions': baseline.get('junctions') and 'junctions' in baseline['metrics']['test'],
            'motion': baseline.get('motion'), 'test': baseline['metrics']['test'],
            'validation': baseline['metrics']['validation'], 'junction_operating': baseline.get('junction_operating'),
            'junction_needed': baseline.get('junction_needed'), 'turn_rule': baseline.get('turn_rule')}


def compare_road_type(run_root: Path) -> str:
    """The comparison on test (and validation for the comma's rule), as text, also written to
    run_root/road-type/comparison.txt: the comma's rule (when the ramp's limit ends) of every variant and of the
    baseline (an earlier version, road-type/baseline/release.json), the F1 per class and the confusion of the ramp, the
    changes of road type, the false switches and the first minute after a merge."""
    run_root = Path(run_root)
    out_dir = run_root / 'road-type'
    results = {}
    for key, folder in COMPARED:
        path = out_dir / folder / 'result.json'
        if path.is_file():
            results[key] = json.loads(path.read_text(encoding='utf-8'))
    earlier = baselines(out_dir)
    rows = []  # (label, test metrics, validation metrics, {scenario: release})
    for baseline in earlier:
        rows.append((baseline['label'], baseline['metrics']['test'], baseline['metrics']['validation'],
                     {'release': baseline.get('release'), 'release_car': baseline.get('release_car')}))
    for key, result in results.items():
        rows.append((_variant_label(result), result['test'], result['validation'],
                     {'release': result.get('release'), 'release_car': result.get('release_car')}))
    for key, result in results.items():
        if 'test_smoothed' in result:
            rows.append((_variant_label(result) + f', {SMOOTH_SECONDS} s majority', result['test_smoothed'],
                         result.get('validation_smoothed'), {}))
    width = max([32] + [len(row[0]) + 1 for row in rows])
    lines = ["The comma's rule: a number read on the ramp (its 3rd second) is the ramp's limit when the road type says "
             'ramp then or a motorway sign was confirmed within 45 s; it ends after 8 s of motorway in a row '
             '(debounced answer). Per merge: released on the ramp (a 120 too early), in time (within 60 s of the '
             'merge, median seconds after it), late (later or never), never held.']
    for scenario, title in _SCENARIOS:
        if not any((releases.get(scenario) or {}).get('test') for *_, releases in rows):
            continue
        lines += ['', title]
        for partition in ('test', 'validation'):
            for label, _, _, releases in rows:
                release = releases.get(scenario)
                if release and release.get(partition):
                    lines.append(f'{partition:<10} {label:{width}} {release_text(release[partition])}')
    lines += ['', 'Per merge (test and validation): release seconds after the merge, how it was held, the debounced '
              'answer from the ramp on (/ ramp, A acceleration lane, M motorway, E expressway, R rural, T town, '
              'P parking, Z 30 zone, G garage)']
    for scenario, title in _SCENARIOS:
        for label, _, _, releases in rows:
            release = releases.get(scenario)
            if not release:
                continue
            lines.append(f"{label} - {title.split(':')[0].lower()}")
            for merge in release.get('merges', []):
                when = '-' if merge['release'] is None else f"{merge['release']:+d} s"
                lines.append(f"  {merge['partition']:<10} {merge['route']} {merge['ramp_key']:>6}->"
                             f"{merge['merge_key']:<6} {merge['ramp_seconds']:>3} s ramp from {merge['before'] or '?':<10} "
                             f"held by {merge['held_by'] or '-':<9} {merge['outcome']:<8} {when:>6}  "
                             f"{merge.get('held_answers', '')}")
    lines += ['', 'Road type on test (areas of the held-out drive, weight-1 frames): F1']
    lines.append(f"{'':{width}} {'macro F1':>8} " + ' '.join(f'{name[:9]:>9}' for name in data.CLASSES))
    for label, metrics, _, _ in rows:
        f1 = metrics['weight_1']
        lines.append(f'{label:{width}} {_numbers(f1["macro_f1"]):>8} '
                     + ' '.join(f'{_numbers(f1["classes"][c]["f1"]):>9}' for c in data.CLASSES))
    ramp = data.CLASSES.index('ramp')
    lines += ['', 'The ramp on test and validation (weight-1 frames): seconds of ramp answered as each class']
    for label, test, validation, _ in rows:
        for partition, metrics in (('test', test), ('validation', validation)):
            matrix = (metrics or {}).get('confusion')
            if matrix:
                said = ', '.join(f'{name} {count}' for name, count in zip(data.CLASSES, matrix[ramp]) if count)
                taken = sum(row[ramp] for i, row in enumerate(matrix) if i != ramp)
                lines.append(f'{partition:<10} {label:{width}} {sum(matrix[ramp])} s: {said or "-"}; '
                             f'other roads called ramp {taken} s')
    lines += ['', 'Transitions on test: found / count, median delay in seconds (negative: early)']
    for label, metrics, _, _ in rows:
        parts = [f"{kind} {entry['found']}/{entry['count']} {_numbers(entry['median_delay'])} s"
                 for kind, entry in metrics['transitions'].items() if entry['count']]
        lines.append(f'{label:{width}} ' + '; '.join(parts))
    lines += ['', 'False switches per hour inside one road type on test (raw / after 3 s debounce)']
    for label, metrics, _, _ in rows:
        lines.append(f"{label:{width}} {_numbers(metrics['false_switches_per_hour'])} / "
                     f"{_numbers(metrics['false_switches_per_hour_debounced'])}")
    lines += ['', f'After a merge: the first {AFTER_MERGE_SECONDS} s on the motorway after a ramp, frames right (raw / '
              'after 3 s debounce) and what was said instead']
    for label, test, validation, _ in rows:
        for partition, metrics in (('test', test), ('validation', validation)):
            merge = (metrics or {}).get('after_merge')
            if not merge:
                continue
            said = ', '.join(f'{name} {count}' for name, count in merge['said'].items() if count)
            lines.append(f"{partition:<10} {label:{width}} {_numbers(merge['accuracy'])} / "
                         f"{_numbers(merge['accuracy_debounced'])} over {merge['merges']} merges, {merge['frames']} s"
                         + (f' (said: {said})' if said else ''))
    with_baselines = {**{f"baseline:{b['folder']}": _as_result(b) for b in earlier}, **results}
    lines += junction_comparison(with_baselines)
    lines += turn_comparison(with_baselines)
    blind = [(_variant_label(r), motion_unknown_text(r)) for r in results.values() if r.get('test_motion_unknown')]
    if blind:
        lines += ['', "With the car's motion unknown on test (a car or a log without it): known -> unknown"]
        lines += [f'{label}: {text}' for label, text in blind]
    lines += selection_lines(results)
    lines.append('')
    for key, result in results.items():
        coverage = result.get('config', {}).get('sign_coverage') or {}
        moved = result.get('config', {}).get('motion_coverage') or {}
        lines.append(f"{result.get('name', key)}: {result['parameters']:,} parameters, "
                     f"{result['flops_per_step'] / 1e6:.0f} MFLOPs a step, best epoch {result['best_epoch']}"
                     + (f"; sign input from the {coverage.get('source', 'car')}: {coverage.get('signs')} motorway "
                        f"signs on {coverage.get('drives_with_signs', coverage.get('read'))} of "
                        f"{coverage.get('drives')} drives" if result.get('signs') else '')
                     + (f"; the car's motion of {moved.get('read')} of {moved.get('drives')} drives"
                        if result.get('motion') else '')
                     + f"; speed from {result.get('speed_source', 'gps')}")
    for baseline in earlier:
        lines.append(f"{baseline['label']}: epoch {baseline.get('epoch')}, speed from "
                     f"{baseline.get('speed_source', 'gps')}, {baseline['checkpoint']}")
    for releases in ([row[3] for row in rows]):
        car = (releases.get('release_car') or {}).get('sign_coverage')
        if car:
            lines.append(f"The car's own signs (nnslr road-type-signs): {car['signs']} motorway signs confirmed on "
                         f"{car['drives_with_signs']} of {car['drives']} drives")
            break
    text = '\n'.join(lines) + '\n'
    out = out_dir / 'comparison.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    return text


def junction_comparison(results: dict) -> list[str]:
    """The junction head of every variant that has one: per decode and threshold on test, the operating point chosen
    on validation, and the intersections the car needed."""
    with_head = {key: result for key, result in results.items()
                 if result.get('junctions') and 'junctions' in (result.get('test') or {})}
    if not with_head:
        return []
    width = max(32, *(len(_variant_label(r)) + 1 for r in with_head.values()))
    first = next(iter(with_head.values()))['test']['junctions']
    lines = ['', f"Junctions on test: {first['passed']} intersections passed in {first['hours']:.2f} h, "
                 f"{first['km']:.1f} km (town {first['areas']['town']['hours']:.2f} h {first['areas']['town']['km']:.1f} km, "
                 f"outside {first['areas']['outside']['hours']:.2f} h {first['areas']['outside']['km']:.1f} km). "
                 'An event says "passed" (passed: its probability rises to the threshold; exit: the in-junction '
                 'probability falls below it); a false one lifts the limit of a sign: per hour and per 10 km']
    for result in with_head.values():
        label = _variant_label(result)
        metrics = result['test']['junctions']
        for decode, by_threshold in metrics['decodes'].items():
            for threshold, numbers in by_threshold.items():
                lines.append(f'{label:{width}} {decode:<6} >= {threshold:<4} {junction_text(metrics, numbers)}')
        inside = metrics['in_junction']
        lines.append(f"{label:{width}} in the intersection (frames, 0.5): precision {_numbers(inside['precision'])}, "
                     f"recall {_numbers(inside['recall'])}, F1 {_numbers(inside['f1'])}")
    lines += ['', 'Operating point (the best F1 at 2 s on validation) and the intersections the car needed (a number '
              f'sign confirmed by its own memory within {NEEDED_METRES:g} m before them)']
    for result in with_head.values():
        label, operating = _variant_label(result), result.get('junction_operating') or {}
        if not operating:
            continue
        on_test = result['test']['junctions']['decodes'][operating['decode']][f"{operating['threshold']:g}"]
        lines.append(f"{label:{width}} {operating['decode']} >= {operating['threshold']:g}: test "
                     + junction_text(result['test']['junctions'], on_test))
        needed = result.get('junction_needed')
        if needed:
            lines.append(f"{'':{width}} needed {needed['needed']} of {needed['junctions']}: caught "
                         f"{needed['caught_2s']} within 2 s, {needed['caught_4s']} within 4 s")
    return lines


def turn_comparison(results: dict) -> list[str]:
    """The plain turn rule (turn_rule: no network) on test, its grid of angles and radii, how many intersections the
    car turned in, and which of them every variant with a junction head caught, turned and straight."""
    turned = [r for r in results.values() if (r.get('turn_rule') or {}).get('test')]
    if not turned:
        return []
    first = turned[-1]['turn_rule']['test']
    passages = first['passages']
    lines = ['', 'The car\'s turn alone, a plain rule without the network (geometry, no speed threshold): a run of '
                 f"seconds turning the same way at {first['rule']['min_dps']:g} deg/s or more, its heading change and "
                 'its radius (the path driven over the heading change in radians); the event the second after it',
             turn_text(first)]
    for cell in first['grid']:
        lines.append(f"  turn >= {cell['degrees']:g} deg, radius <= {cell['radius_m']:g} m: recall "
                     f"{_numbers(cell['recall_2s'])} / {_numbers(cell['recall_4s'])} (2 s / 4 s), {cell['events']} "
                     f"events, false {_numbers(cell['false_per_hour'])}/h (town "
                     f"{_numbers(cell['town']['per_hour'])}, outside {_numbers(cell['outside']['per_hour'])})")
    lines.append(f"Intersections passed on test: {passages['passed']}, the yaw known through {passages['yaw_known']}; "
                 f"the car turned >= 60 deg in {passages['turned_60']}, >= 30 deg in {passages['turned_30']} (from 2 s "
                 'before the intersection to 1 s after the "passed")')
    width = max([32] + [len(_variant_label(r)) + 1 for r in turned])
    for result in turned:
        split = result['turn_rule']['test'].get('network_by_turn')
        if not split:
            continue
        parts = [f"{kind} {split[kind]['caught_2s']}/{split[kind]['passed']} within 2 s, {split[kind]['caught_4s']} "
                 'within 4 s' for kind in ('turned', 'straight')]
        lines.append(f"{_variant_label(result):{width}} caught: " + '; '.join(parts))
    return lines


def selection_lines(results: dict) -> list[str]:
    """How every variant's epochs were chosen: the road type's epoch by road_score, the junction head's stage and
    the branch kept."""
    lines = []
    for result in results.values():
        stage = result.get('junction_stage')
        best = next((p for p in result.get('history', []) if p['epoch'] == result['best_epoch']), {})
        text = (f"{_variant_label(result)}: road epoch {result['best_epoch']} of {len(result.get('history', []))}"
                + (f" (validation road score {_numbers(best.get('val_road_score'))}, macro F1 "
                   f"{_numbers(best.get('val_macro_f1'))})" if best.get('val_road_score') is not None else ''))
        if stage:
            text += (f"; junction head {stage['branch']}: validation F1 at 2 s shared "
                     f"{_numbers((stage.get('shared_validation') or {}).get('f1_2s'))}, own stage "
                     f"{_numbers((stage.get('stage_validation') or {}).get('f1_2s'))} (best stage epoch "
                     f"{stage.get('best_epoch')} of {len(stage.get('history', []))})")
        lines.append(text)
    return ['', 'Selection (validation): the road type by road_score = macro F1 + %g x after-merge (debounced) - %g x '
                'false switches/h (debounced); the junction head on its own stage' % (ROAD_SCORE_MERGE,
                                                                                   ROAD_SCORE_SWITCHES)] + lines if lines else []


def motion_steps() -> list[str]:
    """How the car makes the motion input of a step (car_motion.summarise and motion_features, in the car's code)."""
    return [
        'Subscribe to livePose (named deviceMotion in this sunnypilot, 20 Hz) and carState (100 Hz); at each step t '
        'take the samples of the last second (t - 1 s, t]',
        f'yaw: the samples with angularVelocityDevice.valid; at least {int(motion_data.SAMPLE_HZ * motion_data.MIN_SHARE)}'
        ' of them: yaw_dps = mean(-degrees(angularVelocityDevice.z)) (the device z points down; left positive, the '
        'heading change of the step in degrees), yaw_max = max(|degrees(z)|); fewer: from carState.yawRate (rad/s, '
        'left positive) the same way when the car fills it (not always 0); else unknown',
        f'steer = mean(carState.steeringAngleDeg) over the step (left positive); left / right = carState.leftBlinker /'
        f' rightBlinker true at any sample of the last {motion_data.BLINKER_SECONDS:g} s; with fewer than '
        f'{int(motion_data.SAMPLE_HZ * motion_data.MIN_SHARE)} carState samples in the step all of these are unknown',
        f'motion = [[clip(yaw_dps / {motion_data.YAW_SCALE:g}, -{motion_data.YAW_CLIP:g}, {motion_data.YAW_CLIP:g}), '
        f'clip(yaw_max / {motion_data.YAW_SCALE:g}, 0, {motion_data.YAW_CLIP:g}), 1, clip(steer / '
        f'{motion_data.STEER_SCALE:g}, -{motion_data.STEER_CLIP:g}, {motion_data.STEER_CLIP:g}), 1, left, right, 1]]  '
        '# car_motion.motion_features; an unknown part is all zeros, its known flag 0 (yaw: 0..2, steer: 3..4, '
        'blinkers: 5..7). Never a map field: the car\'s own signals only',
    ]


def car_input_spec(width: int = data.FRAME_WIDTH, height: int = data.FRAME_HEIGHT, signs: bool = False,
                   junctions: dict | None = None, motion: bool = False, separate: bool = False,
                   boards: bool = False) -> dict:
    """How the car makes the inputs of a step from the narrow road camera (sunnypilot's speed_limits/vision.py) and,
    with signs, from speedvisiond's memory of the signs, with motion from livePose and carState; junctions: the
    operating point of the junction head ({'decode', 'threshold'}) when it has one; separate: the junction head has a
    memory of its own (junction_hidden); boards: the board input, from the same memory."""
    sign_steps = [
        'age = now - memory._road_sign_at  # speedvisiond\'s SpeedLimitMemory: when it last confirmed '
        'other_sign:motorway_start or expressway_start (read 5 times a second, MIN_CONFIDENCE, CONFIRM_FRAMES within '
        'CONFIRM_SECONDS); -inf before the first: age inf',
        f'sign = [[min(age, {signs_data.SIGN_HORIZON:g}) / {signs_data.SIGN_SCALE:g}, 1.0]] if age <= '
        f'{signs_data.SIGN_HORIZON:g} else [{list(NO_SIGN)}]  # road_type_signs.sign_features',
    ] if signs else []
    board_steps = [
        f'age = memory.board_age(now)  # now - _board_at: when it last confirmed {BOARD_CLASS} (as the signs, never '
        'a limit); inf before the first',
        f'board = [[min(age, {signs_data.SIGN_HORIZON:g}) / {signs_data.SIGN_SCALE:g}, 1.0]] if age <= '
        f'{signs_data.SIGN_HORIZON:g} else [{list(NO_SIGN)}]  # vision.age_input',
    ] if boards else []
    names_in = ['image', 'speed', *(['sign'] if signs else []), *(['motion'] if motion else []),
                *(['board'] if boards else []), 'hidden',
                *(['junction_hidden'] if separate else [])]
    names_out = ['probabilities', 'hidden', *(['junction'] if junctions else []),
                 *(['junction_hidden'] if separate else [])]
    return {
        'camera': 'VISION_STREAM_ROAD (the narrow road camera), the whole frame: 1344 x 760 on the comma 4, as the '
                  'frames of the dataset (frames.jsonl width 1344, height 760, camera_stream narrow_road)',
        'steps': [
            'y, uv = vision.nv12_planes(buf.data, buf.width, buf.height, buf.stride, buf.uv_offset)',
            'planes = vision.yuv_to_rgb(y, uv, 0, buf.height, 0, buf.width)  # uint8 (3, 760, 1344), ffmpeg BT.601 '
            'limited range, as the dataset PNGs',
            f'planes = vision.resize(planes, {width}, {height})  # PIL BILINEAR (antialiased), columns then rows: '
            'bit-exact with the training cache (road_type_data.resized_frame)',
            'image = planes[None].astype(np.float32) / 255.0  # (1, 3, 144, 256) RGB 0..1, NCHW: no mean/std, the '
            'network normalises inside',
            f'speed = [[min(max(vEgo, 0), {data.SPEED_MAX}) / {data.SPEED_SCALE}, 1.0]]  # carState.vEgo m/s; '
            '[[0, 0]] when unknown',
            *sign_steps,
            *(motion_steps() if motion else []),
            *board_steps,
            'hidden = zeros((1, 128), float32) at the start of the drive (and after more than 5 s without a step); '
            'then the hidden_out of the previous step' + ('; junction_hidden the same with junction_hidden_out'
                                                           if separate else ''),
            f"{', '.join(names_out)} = session.run(None, {{"
            + ', '.join(f'"{name}": {name}' for name in names_in) + '})',
        ],
        'rate': 'one step a second (the training frames are 1 s apart; 2 Hz would make its memory twice as fast: '
                'keep 1 Hz)',
        'decode': 'road_type = classes[argmax(probabilities)], classes exactly the contract\'s classes (road_types.ROAD_TYPES '
                  'when trained: town, rural, expressway, motorway, ramp, parking, zone30, garage, merge); feed '
                  'SpeedLimitMemory.update(road_type=...) only after the same class 3 steps in a row (the measure\'s '
                  'debounce). "ramp" is an on- or off-ramp; "merge" the acceleration lane after an on-ramp, where the '
                  'ramp\'s limit ends without a sign (ramp then motorway or expressway too, for a network without '
                  'it)',
        **({'junction_decode': junction_decode(junctions)} if junctions else {}),
    }


def junction_decode(operating: dict) -> str:
    """The car's rule for "the intersection is passed" from the junction output."""
    if operating['decode'] == 'exit':
        rule = (f"in_junction = junction[0, 0]; passed when in_junction < {operating['threshold']:g} after a step "
                f"with in_junction >= {operating['threshold']:g}")
    else:
        rule = (f"passed = junction[0, 1]; an intersection is passed when passed >= {operating['threshold']:g} and "
                f"the previous step was below")
    return (rule + f'; at most one event in {JUNCTION_GAP_SECONDS:g} s. A false "passed" lifts the limit of a sign '
            '(Poland: a sign holds to the next intersection): use it only where the law says so, never in place of a '
            'sign that ends a limit')


def _random_inputs(torch, model, generator) -> tuple:
    """Random inputs of one step in the order of step_input_names (the motion like a real second: its flags 0 or 1)."""
    hidden = model.hidden_size
    values = {'image': lambda: torch.rand(1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH, generator=generator),
              'speed': lambda: torch.rand(1, 2, generator=generator),
              'sign': lambda: torch.rand(1, 2, generator=generator) * 2,
              'board': lambda: torch.rand(1, 2, generator=generator) * 2,
              'motion': lambda: torch.cat([torch.rand(1, 2, generator=generator) * 2 - 1,
                                           torch.randint(0, 2, (1, 1), generator=generator).float(),
                                           torch.rand(1, 1, generator=generator) * 2 - 1,
                                           torch.randint(0, 2, (1, 4), generator=generator).float()], dim=1),
              'hidden': lambda: torch.rand(1, hidden, generator=generator) * 2 - 1,
              'junction_hidden': lambda: torch.rand(1, hidden, generator=generator) * 2 - 1}
    return tuple(values[name]() for name in step_input_names(model))


def export_road_type_onnx(checkpoint: Path, output: Path, *, verify: bool = True, data_root: Path | None = None) -> dict:
    """The one-second step as ONNX and its contract (<output>.contract.json): see the module's text. The parity with
    PyTorch is checked with onnxruntime on random inputs (rtol 1e-4, atol 1e-5) and over a rollout of 120 real
    seconds of the test when data_root is given (the hidden states fed back, the largest difference reported)."""
    torch = _torch()
    from nnslr_tools.exporting import sha256_file
    checkpoint, output = Path(checkpoint), Path(output)
    model, saved = load_road_type(checkpoint)
    step = step_module(model)
    hidden_size, signs, motion, boards = model.hidden_size, model.signs, model.motion, model.boards
    separate = model.junction_branch == 'separate'
    example = _random_inputs(torch, model, torch.Generator().manual_seed(1))
    output.parent.mkdir(parents=True, exist_ok=True)
    names_in, names_out = step_input_names(model), step_output_names(model)
    operating = None
    if model.junctions:
        result_path = checkpoint.parent / 'result.json'
        if result_path.is_file():
            operating = json.loads(result_path.read_text(encoding='utf-8')).get('junction_operating')
        operating = {'decode': (operating or {}).get('decode', 'passed'),
                     'threshold': (operating or {}).get('threshold', 0.5)}
    try:
        program = torch.onnx.export(step, example, input_names=names_in, output_names=names_out, dynamo=True,
                                    external_data=False)
        program.save(str(output))
    except ImportError as exc:
        raise ValueError('onnx_dependencies_missing: install onnx and onnxscript in the training environment') from exc
    except Exception as exc:  # torch.export refuses the graph: the TorchScript exporter
        print(f'torch.export could not export the road-type step ({type(exc).__name__}); using TorchScript', flush=True)
        torch.onnx.export(step, example, str(output), input_names=names_in, output_names=names_out, dynamo=False,
                          opset_version=17)
    verification = {'enabled': verify, 'passed': None, 'max_abs_error': None, 'rtol': 1e-4, 'atol': 1e-5,
                    'provider': 'CPUExecutionProvider'}
    if verify:
        import onnx
        import onnxruntime
        onnx.checker.check_model(onnx.load(str(output)))
        graph = onnx.load(str(output)).graph
        exported = [o.name for o in graph.output]
        if exported != names_out:
            raise ValueError(f'road-type ONNX outputs {exported}, expected {names_out}')
        taken = [i.name for i in graph.input]
        if taken != names_in:
            raise ValueError(f'road-type ONNX inputs {taken}, expected {names_in}')
        session = onnxruntime.InferenceSession(str(output), providers=['CPUExecutionProvider'])
        generator = torch.Generator().manual_seed(0)
        worst, passed = 0.0, True
        for _ in range(4):
            inputs = _random_inputs(torch, model, generator)
            with torch.no_grad():
                expected = [t.numpy() for t in step(*inputs)]
            actual = session.run(None, dict(zip(names_in, (t.numpy() for t in inputs))))
            for e, a in zip(expected, actual):
                worst = max(worst, float(np.abs(e - a).max()))
                passed = passed and bool(np.allclose(a, e, rtol=1e-4, atol=1e-5))
        verification.update(passed=passed, max_abs_error=worst)
        if data_root is not None:
            verification['rollout'] = _rollout_check(torch, model, step, session, Path(data_root),
                                                     speed_source=saved.get('speed_source', 'gps'))
        if not passed:
            raise ValueError(f'road-type ONNX differs from PyTorch: max abs error {worst}')
    shapes = {'image': [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH], 'speed': [1, 2], 'sign': [1, SIGN_INPUTS],
              'motion': [1, MOTION_INPUTS], 'board': [1, BOARD_INPUTS], 'hidden': [1, hidden_size],
              'junction_hidden': [1, hidden_size]}
    meanings = {
        'speed': [f'min(max(vEgo m/s, 0), {data.SPEED_MAX}) / {data.SPEED_SCALE}', '1 when the speed is known, else 0 '
                  '(and the first value 0)'],
        'sign': [f'min(age, {signs_data.SIGN_HORIZON:g}) / {signs_data.SIGN_SCALE:g}: age = seconds since '
                 'SpeedLimitMemory last confirmed a motorway or expressway sign (now - _road_sign_at)',
                 f'1 when that is at most {signs_data.SIGN_HORIZON:g} s ago, else 0; never or longer ago: '
                 f'{list(NO_SIGN)}'],
        'board': [f'min(age, {signs_data.SIGN_HORIZON:g}) / {signs_data.SIGN_SCALE:g}: age = seconds since '
                  'SpeedLimitMemory last confirmed a green board towards a motorway or an expressway (now - _board_at, '
                  'its board_age; never a limit)',
                  f'1 when that is at most {signs_data.SIGN_HORIZON:g} s ago, else 0; never or longer ago: '
                  f'{list(NO_SIGN)}'],
        'motion': [f'heading change of the last second: mean yaw rate deg/s (left positive) / '
                   f'{motion_data.YAW_SCALE:g}, clipped to +-{motion_data.YAW_CLIP:g}',
                   f'largest |yaw rate| in it deg/s / {motion_data.YAW_SCALE:g}, clipped to 0..{motion_data.YAW_CLIP:g}',
                   '1 when the yaw is known (livePose valid, else carState.yawRate), else 0 (and both values 0)',
                   f'mean carState.steeringAngleDeg of the last second / {motion_data.STEER_SCALE:g}, clipped to '
                   f'+-{motion_data.STEER_CLIP:g} (left positive)',
                   '1 when carState is known, else 0 (and the steering 0)',
                   f'1 when carState.leftBlinker was on in the last {motion_data.BLINKER_SECONDS:g} s',
                   f'1 when carState.rightBlinker was on in the last {motion_data.BLINKER_SECONDS:g} s',
                   '1 when carState is known (the blinkers), else 0'],
        'hidden': 'the memory: zeros at the start, then the previous step\'s hidden_out',
        'junction_hidden': 'the junction head\'s own memory: zeros at the start, then the previous step\'s '
                           'junction_hidden_out'}
    classes = list(saved.get('classes') or data.CLASSES)
    out_shapes = {'probabilities': [1, len(classes)], 'hidden_out': [1, hidden_size],
                  'junction': [1, len(JUNCTION_OUTPUTS)], 'junction_hidden_out': [1, hidden_size]}
    out_meanings = {
        'hidden_out': 'give it back as hidden at the next step',
        'junction_hidden_out': 'give it back as junction_hidden at the next step'}
    outputs = []
    for name in names_out:
        entry = {'name': name, 'shape': out_shapes[name], 'dtype': 'float32'}
        if name in out_meanings:
            entry['meaning'] = out_meanings[name]
        if name == 'junction':
            entry.update(meaning=['the probability that the car is inside an intersection of public roads (sigmoid)',
                                  'the probability that it has just left one: the first second after it (sigmoid)'],
                         names=list(JUNCTION_OUTPUTS), decode=junction_decode(operating), operating=operating)
        outputs.append(entry)
    contract = {
        'schema_version': 1, 'task': 'road_type', 'architecture': saved['architecture'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
        'dataset_sha256': saved.get('dataset_sha256'), 'split_sha256': saved.get('split_sha256'),
        'onnx_sha256': sha256_file(output), 'epoch': saved.get('epoch'),
        'input': {'name': 'image', 'shape': [1, 3, data.FRAME_HEIGHT, data.FRAME_WIDTH], 'dtype': 'float32',
                  'layout': 'NCHW', 'color': 'RGB', 'range': '0..1',
                  'normalization': 'none: ImageNet mean and std are applied inside the network',
                  'resize': {'method': 'PIL BILINEAR (antialiased), vision.resize', 'from': list(data.SOURCE_SIZE),
                             'to': [data.FRAME_WIDTH, data.FRAME_HEIGHT], 'crop': 'none, the whole frame'},
                  'source': 'narrow road camera NV12 -> vision.yuv_to_rgb -> vision.resize'},
        'inputs': [{'name': name, 'shape': shapes[name], 'dtype': 'float32',
                    **({'meaning': meanings[name]} if name in meanings else {}),
                    **({'names': list(motion_data.MOTION_FIELDS)} if name == 'motion' else {})} for name in names_in],
        'output': {'name': 'probabilities', 'shape': [1, len(classes)], 'classes': classes,
                   'decode': 'softmax probabilities; the class is the argmax'},
        'outputs': outputs,
        'step': {'rate_hz': 1.0, 'memory': saved['memory'], 'signs': signs, 'motion': motion, 'boards': boards,
                 'board_signs': list(saved.get('board_signs') or ([BOARD_CLASS] if boards else [])),
                 'junctions': bool(model.junctions), 'junction_branch': model.junction_branch,
                 'speed_source': saved.get('speed_source', 'gps'), 'inputs': names_in, 'outputs': names_out},
        'car_input': car_input_spec(signs=signs, junctions=operating, motion=motion, separate=separate,
                                    boards=boards),
        'verification': verification,
    }
    if motion:
        contract['car_input']['motion'] = {
            'fields': list(motion_data.MOTION_FIELDS),
            'cereal': {'yaw': 'livePose.angularVelocityDevice.z (rad/s, device frame, z down: yaw = -z) and .valid; '
                              'this sunnypilot names the service deviceMotion; fallback carState.yawRate (rad/s, left '
                              'positive)',
                       'steer': 'carState.steeringAngleDeg', 'blinkers': 'carState.leftBlinker, carState.rightBlinker',
                       'speed': 'carState.vEgo'},
            'summary': {'step_seconds': motion_data.STEP_SECONDS, 'blinker_seconds': motion_data.BLINKER_SECONDS,
                        'min_samples': int(motion_data.SAMPLE_HZ * motion_data.MIN_SHARE),
                        'yaw_scale': motion_data.YAW_SCALE, 'yaw_clip': motion_data.YAW_CLIP,
                        'steer_scale': motion_data.STEER_SCALE, 'steer_clip': motion_data.STEER_CLIP},
            'never': 'a map field: the vision side reads the camera and the car only'}
    contract_path = output.with_name(output.name + '.contract.json')
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'onnx': str(output), 'contract': str(contract_path), 'onnx_sha256': contract['onnx_sha256'],
            'verification': verification}


def _rollout_check(torch, model, step, session, data_root: Path, seconds: int = 120,
                   speed_source: str = 'gps') -> dict:
    """The step run for seconds of a test sequence by PyTorch and by onnxruntime, each with its own hidden states fed
    back: the largest difference of the outputs and whether the answers agree. With the sign input: from 30 s before
    the first motorway sign the car confirmed on the test (the first test sequence when there is none); the motion as
    read from the drive's logs."""
    dataset = data.load_dataset(data_root)
    attach_signs(dataset, data_root)
    attach_motion(dataset, data_root, v_ego=speed_source == 'v_ego')
    attach_boards(dataset, data_root)
    tests = [run for run in dataset['sequences'] if run['partition'] == 'test']
    if not tests:
        return {'checked': False}
    run, begin = tests[0], 0
    if model.signs:
        for candidate in tests:
            seen = np.flatnonzero(_sequence_arrays(dataset, candidate)['sign'][:, 1] > 0)
            if len(seen):
                run, begin = candidate, max(0, int(seen[0]) - 30)
                break
    arrays = _sequence_arrays(dataset, run)
    frames = dataset['frames'][run['route']]
    names_in, names_out = step_input_names(model), step_output_names(model)
    state_t = {name: torch.zeros(1, model.hidden_size) for name in ('hidden', 'junction_hidden')}
    state_o = {name: np.zeros((1, model.hidden_size), dtype=np.float32) for name in ('hidden', 'junction_hidden')}
    worst, agree, count = 0.0, 0, 0
    for offset, position in enumerate(run['positions'][begin:begin + seconds]):
        frame = frames[position]
        at = begin + offset
        given = {'image': data.resized_frame(frame['image']).transpose(2, 0, 1)[None].astype(np.float32) / 255,
                 'speed': arrays['speed'][at][None].astype(np.float32), 'sign': arrays['sign'][at][None],
                 'motion': arrays['motion'][at][None], 'board': arrays['board'][at][None]}
        with torch.no_grad():
            out_t = step(*(state_t[n] if n in state_t else torch.from_numpy(given[n]) for n in names_in))
        out_o = session.run(None, {n: state_o[n] if n in state_o else given[n] for n in names_in})
        named_t, named_o = dict(zip(names_out, out_t)), dict(zip(names_out, out_o))
        for name in names_out:
            if name not in ('hidden_out', 'junction_hidden_out'):
                worst = max(worst, float(np.abs(named_t[name].numpy() - named_o[name]).max()))
        state_t['hidden'], state_o['hidden'] = named_t['hidden_out'], named_o['hidden_out']
        if 'junction_hidden_out' in named_t:
            state_t['junction_hidden'], state_o['junction_hidden'] = (named_t['junction_hidden_out'],
                                                                      named_o['junction_hidden_out'])
        agree += int(named_t['probabilities'].numpy().argmax() == named_o['probabilities'].argmax())
        count += 1
    return {'checked': True, 'route': run['route'], 'first_key': arrays['keys'][begin], 'seconds': count,
            'max_abs_error': worst, 'same_answer': agree,
            'sign_seconds': int((arrays['sign'][begin:begin + seconds, 1] > 0).sum()),
            'motion_seconds': int((arrays['motion'][begin:begin + seconds, 2] > 0).sum())}
