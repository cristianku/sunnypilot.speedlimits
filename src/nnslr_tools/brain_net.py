"""The brain: one network with memory that says the speed to hold, from the road camera and what was read, and nothing of a
map. Here are the network and its loss, the end of the road from a person's labels to the weights
(nnslr_tools.brain_targets makes the answers, nnslr_tools.brain_train trains it, only on request).

The network, per frame: encoder (MobileNetV3-small, as the road-type network) -> features, with the inputs of the frame
(brain_targets.INPUT_SIZES) -> GRU -> h, then two heads on h:

    speed  = speed_head(h) + trust(h) * number   the speed to hold, a class of the table: the only output on the car
    works  = works_head(h)                       are we inside roadworks: only in training

The table is the network's own (build_brain_net's kph, the class_kph buffer, kept in every checkpoint): a class a km/h from
1 to 200 since 8 Nov 2026 (brain_targets.SPEED_CLASSES), a class every 5 km/h before. A brain of before keeps its own: its
checkpoint loads with it (brain_export.load_brain), and its contract says it to the car.

loss = cross entropy(speed, his speed) + WORKS_LOSS * binary cross entropy(works, his marks), each only on the frames with
an answer of its own. The works error goes back through h into the GRU and the layer on the picture's features, which the
speed uses too: that is how a mark changes the network, without ever being an input. works_head is one layer, left out of the car's network
(car_weights), and no code goes from works to speed. Whether it helps is measured: the same training without it
(works=False), on sites of test and validation.

His speed is no single class but a bell over the table around it (speed_targets, SPEED_SIGMA_KPH wide): at a km/h a
class, a hard answer would cost 52 for 53 as much as 52 for 120, and a speed off the dashboard is only so exact. With the
bell a near speed is a small error, so the network learns the order of the speeds; the car still decodes by argmax. A frame
of weight 0 adds nothing, whatever its answer says.

The number goes straight to the answer too: number is 1 on the class of the number in force, so trust(h) is added to that
class alone. As one value among the GRU's inputs it hardly reached the answer: with the numbers swapped in training (5 Nov
2026) the answer moved on almost no frame when the number changed, and on few even with one constant picture in place of the
road. The network still decides: trust comes from the memory and can be 0 or below on a sign it has reason to doubt, and
speed_head can say any speed. It is all comparisons, which export as they are. A comparison with no signs among the groups
has no number and no such path.

The limit ahead (ahead=True, the brains since 7 Nov 2026): two more heads on the same h, so that sunnypilot slows down for
the next limit as for the map's (brain_targets.reviewed_signs and ahead_from_signs):

    next   = next_head(h) + trust_next(h) * number   the next limit: a class of the speed's table, or the last one, none
    metres = metres_head(h) + metres_gate(h) * sign  the metres to it, over METRES_SCALE

The number is the one read, and a sign is read some way before it: on the way to it the number is the next limit, so it
goes straight to next as to the speed, and the memory decides which of the two it is (sign_distance among its inputs).
sign is vision.SignDistance's metres to the sign ahead, the same scale: metres can be that, or anything the memory says.
loss += NEXT_LOSS * cross entropy(next) + METRES_LOSS * smooth L1(metres), each on its own frames. The next limit is a sign's
number, exact: its answer is one class, no bell.

The picture's features. pool 'mean' is the road-type network's: the mean of the 8 x 5 map keeps what is in a frame and loses
most of where, and a truck in the lane beside ours is a matter of where. pool 'grid' keeps the map: a 1 x 1 convolution to
GRID_CHANNELS, then all its cells into one layer. Which one sees works is measured too.

Batch norm stays frozen in training: its statistics of a batch would let the frames after a moment reach it. On the car the
network only ever has the past, so here too. With its statistics frozen and one sequence a step nothing brings a moving
encoder back: in the first run (5 Nov 2026) its output grew 250 times, the GRU saturated, the signs reached nothing and the
network said one speed on every frame. So the encoder does not learn, for now: a few drives would only teach it their roads.
The picture's features go through a LayerNorm, so that they keep the scale of the inputs beside them in the GRU.
"""
from __future__ import annotations

from nnslr_tools.brain_targets import BRAIN_INPUTS, INPUT_SIZES, SPEED_CLASSES, class_kph
from nnslr_tools.road_type_net import FEATURES, HIDDEN, IMAGENET_MEAN, IMAGENET_STD, _torch

WORKS_LOSS = 0.3
NEXT_LOSS = 0.5
METRES_LOSS = 1.0
METRES_BETA = 0.1  # smooth L1 within 10 m (over METRES_SCALE), then linear
SPEED_SIGMA_KPH = 2.0  # the width of the bell around his speed (speed_targets)
POOLS = ('mean', 'grid')
GRID_CHANNELS = 32
GRID_CELLS = 8 * 5  # the encoder's map of a 256 x 144 frame
BACKBONE_CHANNELS = 576
# Out of the car's network.
TRAINING_ONLY = 'works_head.'


def training_table() -> list[float]:
    """The km/h of each class of the brains trained today (brain_targets): 1 to 200."""
    return [float(class_kph(index)) for index in range(SPEED_CLASSES)]


def _table(kph) -> list[float]:
    """kph as a list of km/h in order, one a class: training_table() for None."""
    if kph is None:
        return training_table()
    values = [float(value) for value in (kph.flatten().tolist() if hasattr(kph, 'flatten') else kph)]
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f'a table of the speeds is km/h going up, one a class: {values[:5]}...')
    return values


def build_brain_net(groups: tuple[str, ...], *, pool: str = 'grid', works: bool = True, ahead: bool = False,
                    pretrained: bool = True, features: int = FEATURES, hidden: int = HIDDEN, kph=None):
    """The network (torch.nn.Module): forward(images (B, T, 3, H, W) float 0..1, extra (B, T, inputs of the groups), state
    (B, hidden) or None) -> (speed logits (B, T, classes), works logit (B, T) or None, state); with ahead=True asked of
    forward (a network built with ahead) also next logits (B, T, classes + 1) and metres (B, T).
    kph: the km/h of each class, in order (a list, or the class_kph buffer of a checkpoint): the heads have as many classes,
    and a number in force falls on the class within half the table's step of it. None: the table of today's training."""
    torch = _torch()
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if pool not in POOLS:
        raise ValueError(f'pool must be one of {POOLS}')
    table = _table(kph)
    classes = len(table)
    # A plain number, no buffer: a checkpoint of before has none to load, and the export writes it as a constant.
    near_kph = (min(b - a for a, b in zip(table, table[1:])) if classes > 1 else table[0]) / 2
    extra = sum(INPUT_SIZES[name] for name in groups)
    signs_at = sum(INPUT_SIZES[name] for name in groups[:groups.index('signs')]) if 'signs' in groups else None
    distance_at = (sum(INPUT_SIZES[name] for name in groups[:groups.index('sign_distance')])
                   if 'sign_distance' in groups else None)

    class BrainNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.groups, self.pool = groups, pool
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.backbone = backbone.features.requires_grad_(False)
            if pool == 'grid':
                self.squeeze = nn.Conv2d(BACKBONE_CHANNELS, GRID_CHANNELS, 1)
            width = BACKBONE_CHANNELS if pool == 'mean' else GRID_CHANNELS * GRID_CELLS
            self.project = nn.Sequential(nn.Linear(width, features), nn.LayerNorm(features), nn.Hardswish(), nn.Dropout(0.2))
            self.gru = nn.GRU(features + extra, hidden, batch_first=True)
            self.speed_head = nn.Linear(hidden, classes)
            self.trust = nn.Linear(hidden, 1)
            self.works_head = nn.Linear(hidden, 1) if works else None
            self.ahead = ahead
            if ahead:
                self.next_head = nn.Linear(hidden, classes + 1)
                self.trust_next = nn.Linear(hidden, 1)
                self.metres_head = nn.Linear(hidden, 1)
                self.metres_gate = nn.Linear(hidden, 1)
            self.register_buffer('class_kph', torch.tensor(table, dtype=torch.float32))
            self.near_kph = near_kph

        def train(self, mode: bool = True):
            super().train(mode)
            for layer in self.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
            return self

        def encode(self, images):
            """(N, 3, H, W) 0..1 -> (N, features)."""
            x = self.backbone((images - self.mean) / self.std)
            x = x.mean(dim=(2, 3)) if self.pool == 'mean' else self.squeeze(x).flatten(1)
            return self.project(x)

        def number_in_force(self, extra_input):
            """(B, T, inputs) -> (B, T, classes): 1 on the class of the number in force, all 0 with none."""
            if signs_at is None:
                return extra_input.new_zeros(*extra_input.shape[:-1], classes)
            signs = extra_input[..., signs_at:signs_at + INPUT_SIZES['signs']]
            held = (signs[..., 1:2] > 0) & (signs[..., 6:7] == 0)
            near = (signs[..., :1] * BRAIN_INPUTS.SIGN_KPH_SCALE - self.class_kph).abs() < self.near_kph
            return (near & held).to(extra_input.dtype)

        def sign_metres(self, extra_input):
            """(B, T, inputs) -> (B, T): vision.SignDistance's metres over METRES_SCALE, 0 without a sign or the input."""
            if distance_at is None:
                return extra_input.new_zeros(extra_input.shape[:-1])
            return extra_input[..., distance_at]

        def forward(self, images, extra_input, state=None, ahead: bool = False):
            b, t = images.shape[:2]
            extra_input = extra_input.to(images.dtype)
            x = torch.cat([self.encode(images.flatten(0, 1)).view(b, t, -1), extra_input], dim=-1)
            out, last = self.gru(x, None if state is None else state[None].contiguous())
            number = self.number_in_force(extra_input)
            speed = self.speed_head(out) + self.trust(out) * number
            works = None if self.works_head is None else self.works_head(out)[..., 0]
            if not ahead:
                return speed, works, last[0]
            following = self.next_head(out) + self.trust_next(out) * torch.cat([number, number[..., :1] * 0], dim=-1)
            metres = self.metres_head(out)[..., 0] + self.metres_gate(out)[..., 0] * self.sign_metres(extra_input)
            return speed, works, last[0], following, metres

    return BrainNet()


def speed_targets(speed, kph=None, sigma_kph: float = SPEED_SIGMA_KPH):
    """(..., classes) the answer of each frame as chances over the classes: a bell of sigma_kph around the km/h of its class,
    cut at the ends of the table and summing to 1. speed: classes (long), -1 (no class) as the first one, a frame that
    has no weight anyway. kph: the km/h of each class (the network's class_kph), None the table of today's training."""
    torch = _torch()
    table = (torch.tensor(training_table(), device=speed.device) if kph is None else
             torch.as_tensor(kph, device=speed.device).flatten()).to(torch.float32)
    answer = table[speed.clamp(0, len(table) - 1)]
    bell = torch.exp(-0.5 * ((table - answer[..., None]) / sigma_kph) ** 2)
    return bell / bell.sum(dim=-1, keepdim=True)


def brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, *, works_share: float = WORKS_LOSS,
               frames: int | None = None, ahead: tuple | None = None, kph=None):
    """The error of a batch, (B, T) answers and weights of brain_targets: each task a mean over its own frames, 0 with none.
    A frame of weight 0 gives no gradient: not examined is not an answer, and an unmarked stretch is not "no works".
    The speed's answer is speed_targets' bell over the table kph (the network's class_kph; None: today's training's).
    frames: divide by this many instead, so that a frame weighs the same in a piece with one answer as in a full one.
    ahead: (next logits, metres, next, next_weight, metres answers, metres_weight) of a network with the limit ahead."""
    torch = _torch()
    from torch.nn import functional

    def mean(errors, weight):
        return (errors * weight).sum() / (frames or weight.sum().clamp_min(1))

    weight = speed_weight.flatten()
    errors = functional.cross_entropy(speed_logits.flatten(0, 1), speed_targets(speed.flatten(), kph), reduction='none')
    total = mean(torch.where(weight > 0, errors, torch.zeros_like(errors)), weight)
    if works_logit is not None:
        total = total + works_share * mean(functional.binary_cross_entropy_with_logits(works_logit, works, reduction='none'),
                                           works_weight)
    if ahead is not None:
        next_logits, metres, following, next_weight, target, metres_weight = ahead
        errors = functional.cross_entropy(next_logits.flatten(0, 1), following.flatten(), reduction='none')
        total = total + NEXT_LOSS * mean(errors, next_weight.flatten())
        total = total + METRES_LOSS * mean(functional.smooth_l1_loss(metres, target, reduction='none', beta=METRES_BETA),
                                           metres_weight)
    return total


def car_weights(model) -> dict:
    """The weights that go to the car: all but works_head (the heads of the limit ahead go too)."""
    return {name: value for name, value in model.state_dict().items() if not name.startswith(TRAINING_ONLY)}
