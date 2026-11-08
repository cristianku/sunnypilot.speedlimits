"""The road labeler's network: the pictures of a drive into features once, then a memory over the whole drive that looks
both ways and says, at every step, the road type, the kinds of road and whether we are inside roadworks. Offline only:
see nnslr_tools.road_labeler_data for what it is for, its inputs and its answers.

The picture. Every step's frame of fcamera.hevc (STEP_HZ a second: the decoded frames 0, 10, 20... of each minute) is
decoded by ffmpeg straight to the encoder's size (decode_segment), at its full definition when the drive was recorded so:
ENCODER, MobileNetV3-large of ImageNet at 768 x 432, five times the pixels the brain's 256 x 144 has (a tunnel's portal, a
garage's barrier, a roadworks board from far). The encoder does not learn: its map of 24 x 14 cells is pooled to GRID
(2 rows, 4 columns: the sky and the road, left to right) and kept as float16 in a cache a drive (drive_features), so the
training of the memory reads features, not pictures, and an epoch over every drive takes seconds. Which encoder, size
and grid make the features is in the cache's name (encoder_id): a change makes a new cache, never a mix.

The memory, per step: LayerNorm and a layer from the picture's features to FEATURES, beside the inputs of
road_labeler_data (the map scene, the GPS, the dashboard speed) and whether the step has its picture, into a GRU of
LAYERS layers, both ways (bidirectional: offline it can look ahead, so a tunnel ends where its walls stop being seen while
the memory knows it was in one). Three heads on the two directions:

    types = type_head(h)    a logit a road type of road_types.ROAD_TYPES (cross entropy, weighted: a person's 1, the map's less)
    kinds = kind_head(h)    a logit a kind of road of Speed review (binary cross entropy a kind)
    works = works_head(h)   inside roadworks (binary cross entropy); its starts and ends follow from it

loss = TYPE_LOSS * types + KIND_LOSS * each kind + WORKS_LOSS * works, each a mean over its own weights: a step of weight
0 teaches nothing, a kind with no answer adds nothing.

ENCODER_TINY is a small random network at 128 x 72 for a dry run on the CPU: the chain works, nothing is learned. Its
weights are kept in the checkpoint, since nobody else has them.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nnslr_tools.road_labeler_data import HALF_STEP, INPUTS
from nnslr_tools.road_types import ROAD_TYPES

ENCODER = 'mobilenet_v3_large'
ENCODER_TINY = 'tiny'
SIZES = {ENCODER: (768, 432), ENCODER_TINY: (128, 72)}
CHANNELS = {ENCODER: 960, ENCODER_TINY: 16}
GRID = (2, 4)
FEATURE_VERSION = 1
FEATURES = 256
HIDDEN = 128
LAYERS = 2
TYPE_LOSS = 1.0
KIND_LOSS = 0.5
WORKS_LOSS = 0.5
NET_INPUTS = INPUTS + 1  # and whether the step has its picture
ENCODE_BATCH = 32
DECODE_WORKERS = 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _torch():
    import torch
    return torch


def encoder_id(name: str) -> str:
    """The name of the features an encoder makes: what made them, at which size and grid."""
    width, height = SIZES[name]
    return f'{name}-{width}x{height}-g{GRID[0]}x{GRID[1]}-v{FEATURE_VERSION}'


def feature_size(name: str) -> int:
    return CHANNELS[name] * GRID[0] * GRID[1]


def build_encoder(name: str, *, pretrained: bool = True, seed: int = 0):
    """(N, 3, H, W) float 0..1 -> (N, feature_size(name)): the frozen encoder, in eval mode."""
    torch = _torch()
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            if name == ENCODER:
                from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
                self.body = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained
                                               else None).features
            elif name == ENCODER_TINY:
                generator = torch.random.fork_rng()
                with generator:
                    torch.manual_seed(seed)
                    self.body = nn.Sequential(nn.Conv2d(3, 8, 5, stride=4, padding=2), nn.ReLU(),
                                              nn.Conv2d(8, CHANNELS[ENCODER_TINY], 3, stride=2, padding=1), nn.ReLU())
            else:
                raise ValueError(f'no encoder {name}: {ENCODER} or {ENCODER_TINY}')
            self.requires_grad_(False)

        def forward(self, images):
            x = self.body((images - self.mean) / self.std)
            return nn.functional.adaptive_avg_pool2d(x, GRID).flatten(1)

    return Encoder().eval()


def decode_segment(video: Path, size: tuple[int, int], last_decoded: int) -> np.ndarray:
    """uint8 (frames, H, W, 3): the decoded frames 0, HALF_STEP, 2 HALF_STEP... of a segment's road video, up to
    last_decoded, scaled by ffmpeg to size (width, height). Fewer when the video is shorter."""
    width, height = size
    wanted = last_decoded // HALF_STEP + 1
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'hevc', '-i', str(video), '-vf',
                             f"select='not(mod(n\\,{HALF_STEP}))',scale={width}:{height}:flags=area", '-vsync', '0',
                             '-frames:v', str(wanted), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'],
                            capture_output=True, check=False)
    frame = width * height * 3
    if decode.returncode != 0 and not decode.stdout:
        raise ValueError(f'{video}: not decoded: ' + decode.stderr.decode(errors='replace').strip()[:200])
    count = len(decode.stdout) // frame
    return np.frombuffer(decode.stdout[:count * frame], dtype=np.uint8).reshape(count, height, width, 3)


def _png(path: str, size: tuple[int, int]) -> np.ndarray | None:
    from PIL import Image
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB').resize(size, Image.BILINEAR), dtype=np.uint8)
    except OSError:
        return None


def segment_pictures(root: Path, route: str, segment: int, steps: list[tuple[int, dict]], size: tuple[int, int]) -> dict:
    """{position of the step: uint8 (H, W, 3)} of a segment's steps ([(position, step)]): from its road video, or, without
    one (or when ffmpeg decodes nothing of it), the 1 Hz PNG of the step's second for both its steps. A step the video
    does not reach has none."""
    from nnslr_tools.video_frames import VideoFrameError, video_of
    found = {}
    try:
        video = video_of(root, route, segment)
    except VideoFrameError:
        video = None
    if video is not None:
        try:
            frames = decode_segment(video, size, max(step['decoded'] for _at, step in steps))
        except ValueError:
            frames = []  # a broken video: the 1 Hz pictures below
    if video is not None and len(frames):
        for at, step in steps:
            if step['decoded'] % HALF_STEP == 0 and step['decoded'] // HALF_STEP < len(frames):
                found[at] = frames[step['decoded'] // HALF_STEP]
        return found
    for at, step in steps:
        picture = _png(step['image'], size)
        if picture is not None:
            found[at] = picture
    return found


def drive_features(root: Path, route: str, steps: list[dict], encoder, name: str, device, cache_dir: Path | None,
                   log=None) -> tuple[np.ndarray, np.ndarray]:
    """(float16 (steps, feature_size), bool present) of a drive: read from cache_dir/<route>.npz when it was made for these
    steps, else every segment decoded (DECODE_WORKERS side by side) and encoded, then kept there. cache_dir None: kept
    nowhere."""
    torch = _torch()
    keys = np.array([f"{step['segment']}/{step['decoded']}" for step in steps])
    path = None if cache_dir is None else Path(cache_dir) / f'{route}.npz'
    if path is not None and path.is_file():
        with np.load(path) as kept:
            if np.array_equal(kept['keys'], keys):
                return kept['features'], kept['present']
    size = SIZES[name]
    features = np.zeros((len(steps), feature_size(name)), dtype=np.float16)
    present = np.zeros(len(steps), dtype=bool)
    segments: dict[int, list[tuple[int, dict]]] = {}
    for at, step in enumerate(steps):
        segments.setdefault(step['segment'], []).append((at, step))
    kind = getattr(device, 'type', str(device))
    use_half = kind == 'cuda'
    def decoded_segments(pool):
        # At most 2 * DECODE_WORKERS segments decoded ahead of the encoder (about 120 MB each at 768 x 432): a slow
        # encoder (the CPU) never holds a whole drive's pictures.
        items = sorted(segments.items())
        for first in range(0, len(items), 2 * DECODE_WORKERS):
            yield from pool.map(lambda item: (item[0], segment_pictures(root, route, item[0], item[1], size)),
                                items[first:first + 2 * DECODE_WORKERS])

    with ThreadPoolExecutor(DECODE_WORKERS) as pool:
        for segment, pictures in decoded_segments(pool):
            if not pictures:
                if log:
                    log(f'{route} minute {segment}: no picture')
                continue
            rows = sorted(pictures)
            for first in range(0, len(rows), ENCODE_BATCH):
                part = rows[first:first + ENCODE_BATCH]
                batch = torch.from_numpy(np.stack([pictures[at] for at in part])).to(device)
                batch = batch.permute(0, 3, 1, 2).float().div_(255)
                with torch.no_grad(), torch.autocast(device_type=kind, enabled=use_half):
                    features[part] = encoder(batch).float().cpu().numpy().astype(np.float16)
                present[part] = True
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez(temporary, keys=keys, features=features, present=present)
        temporary.replace(path)
    return features, present


def build_labeler_net(feature_dim: int, kinds: int, *, features: int = FEATURES, hidden: int = HIDDEN,
                      layers: int = LAYERS, types: int = len(ROAD_TYPES)):
    """forward(features (B, T, feature_dim), inputs (B, T, NET_INPUTS)) -> {'types' (B, T, types), 'kinds' (B, T, kinds)
    or None without kinds, 'works' (B, T)}, logits."""
    torch = _torch()
    from torch import nn

    class RoadLabelerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.picture = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, features), nn.GELU(),
                                         nn.Dropout(0.2))
            self.gru = nn.GRU(features + NET_INPUTS, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=0.1 if layers > 1 else 0.0)
            self.type_head = nn.Linear(2 * hidden, types)
            self.kind_head = nn.Linear(2 * hidden, kinds) if kinds else None
            self.works_head = nn.Linear(2 * hidden, 1)

        def forward(self, feature_input, extra_input):
            x = torch.cat([self.picture(feature_input.float()), extra_input.float()], dim=-1)
            out, _state = self.gru(x)
            return {'types': self.type_head(out), 'kinds': None if self.kind_head is None else self.kind_head(out),
                    'works': self.works_head(out)[..., 0]}

    return RoadLabelerNet()


def labeler_loss(said: dict, answers: dict):
    """The error of a batch: answers {type (B, T) long -1 none, type_weight, kinds (B, T, K), kinds_weight, works,
    works_weight}, each head a mean over its own weights (0 without any)."""
    torch = _torch()
    from torch.nn import functional

    def mean(errors, weight):
        return (errors * weight).sum() / weight.sum().clamp_min(1e-6) if float(weight.sum()) > 0 else errors.sum() * 0

    errors = functional.cross_entropy(said['types'].flatten(0, 1), answers['type'].flatten().clamp_min(0), reduction='none')
    total = TYPE_LOSS * mean(errors, answers['type_weight'].flatten())
    if said['kinds'] is not None:
        errors = functional.binary_cross_entropy_with_logits(said['kinds'], answers['kinds'], reduction='none')
        for kind in range(said['kinds'].shape[-1]):
            total = total + KIND_LOSS * mean(errors[..., kind], answers['kinds_weight'][..., kind])
    errors = functional.binary_cross_entropy_with_logits(said['works'], answers['works'], reduction='none')
    return total + WORKS_LOSS * mean(errors, answers['works_weight'])


def chances(said: dict) -> dict:
    """The logits of forward as chances, numpy, for one sequence: types (T, types), kinds (T, K), works (T)."""
    torch = _torch()
    return {'types': torch.softmax(said['types'][0].float(), dim=-1).cpu().numpy(),
            'kinds': (np.zeros((said['types'].shape[1], 0), dtype=np.float32) if said['kinds'] is None
                      else torch.sigmoid(said['kinds'][0].float()).cpu().numpy()),
            'works': torch.sigmoid(said['works'][0].float()).cpu().numpy()}
