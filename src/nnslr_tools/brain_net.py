"""The brain: one network with memory that says the speed to hold, from the road camera and what was read, and nothing of a
map. Here are the network and its loss, the end of the road from a person's labels to the weights
(nnslr_tools.brain_targets makes the answers). There is no training yet: that is started only on request.

The network, per frame: encoder (MobileNetV3-small, as the road-type network) -> features, with the inputs of the frame
(brain_targets.INPUT_SIZES) -> GRU -> h, then two heads on h:

    speed  = speed_head(h)   the speed to hold, a class of 5 km/h: the only output on the car
    works  = works_head(h)   are we inside roadworks: only in training

loss = cross entropy(speed, his speed) + WORKS_LOSS * binary cross entropy(works, his marks), each only on the frames with
an answer of its own. The works error goes back through h into the GRU and the encoder, which the speed uses too: that is
how a mark changes the network, without ever being an input. works_head is one layer, left out of the car's network
(car_weights), and no code goes from works to speed. Whether it helps is measured: the same training without it
(works=False), on sites of test and validation.

The picture's features. pool 'mean' is the road-type network's: the mean of the 8 x 5 map keeps what is in a frame and loses
most of where, and a truck in the lane beside ours is a matter of where. pool 'grid' keeps the map: a 1 x 1 convolution to
GRID_CHANNELS, then all its cells into one layer. Which one sees works is measured too.

Batch norm stays frozen in training: its statistics of a batch would let the frames after a moment reach it. On the car the
network only ever has the past, so here too.
"""
from __future__ import annotations

from nnslr_tools.brain_targets import INPUT_SIZES, SPEED_CLASSES
from nnslr_tools.road_type_net import FEATURES, HIDDEN, IMAGENET_MEAN, IMAGENET_STD, _torch

WORKS_LOSS = 0.3
POOLS = ('mean', 'grid')
GRID_CHANNELS = 32
GRID_CELLS = 8 * 5  # the encoder's map of a 256 x 144 frame
BACKBONE_CHANNELS = 576
# Out of the car's network.
TRAINING_ONLY = 'works_head.'


def build_brain_net(groups: tuple[str, ...], *, pool: str = 'mean', works: bool = True, pretrained: bool = True,
                    features: int = FEATURES, hidden: int = HIDDEN):
    """The network (torch.nn.Module): forward(images (B, T, 3, H, W) float 0..1, extra (B, T, inputs of the groups), state
    (B, hidden) or None) -> (speed logits (B, T, SPEED_CLASSES), works logit (B, T) or None, state)."""
    torch = _torch()
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if pool not in POOLS:
        raise ValueError(f'pool must be one of {POOLS}')
    extra = sum(INPUT_SIZES[name] for name in groups)

    class BrainNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.groups, self.pool = groups, pool
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            self.backbone = backbone.features
            if pool == 'grid':
                self.squeeze = nn.Conv2d(BACKBONE_CHANNELS, GRID_CHANNELS, 1)
            width = BACKBONE_CHANNELS if pool == 'mean' else GRID_CHANNELS * GRID_CELLS
            self.project = nn.Sequential(nn.Linear(width, features), nn.Hardswish(), nn.Dropout(0.2))
            self.gru = nn.GRU(features + extra, hidden, batch_first=True)
            self.speed_head = nn.Linear(hidden, SPEED_CLASSES)
            self.works_head = nn.Linear(hidden, 1) if works else None

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

        def forward(self, images, extra_input, state=None):
            b, t = images.shape[:2]
            x = torch.cat([self.encode(images.flatten(0, 1)).view(b, t, -1), extra_input.to(images.dtype)], dim=-1)
            out, last = self.gru(x, None if state is None else state[None].contiguous())
            return self.speed_head(out), None if self.works_head is None else self.works_head(out)[..., 0], last[0]

    return BrainNet()


def brain_loss(speed_logits, works_logit, speed, speed_weight, works, works_weight, *, works_share: float = WORKS_LOSS,
               frames: int | None = None):
    """The error of a batch, (B, T) answers and weights of brain_targets: each task a mean over its own frames, 0 with none.
    A frame of weight 0 gives no gradient: not examined is not an answer, and an unmarked stretch is not "no works".
    frames: divide by this many instead, so that a frame weighs the same in a piece with one answer as in a full one."""
    torch = _torch()
    from torch.nn import functional

    errors = functional.cross_entropy(speed_logits.flatten(0, 1), speed.clamp_min(0).flatten(), reduction='none')
    total = (errors * speed_weight.flatten()).sum() / (frames or speed_weight.sum().clamp_min(1))
    if works_logit is None:
        return total
    errors = functional.binary_cross_entropy_with_logits(works_logit, works, reduction='none')
    return total + works_share * (errors * works_weight).sum() / (frames or works_weight.sum().clamp_min(1))


def car_weights(model) -> dict:
    """The weights that go to the car: all but works_head."""
    return {name: value for name, value in model.state_dict().items() if not name.startswith(TRAINING_ONLY)}
