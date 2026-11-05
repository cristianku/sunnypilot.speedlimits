"""Whom the boxes of a frame apply to, for "Car networks" of Sign review (nnslr_tools.car_reading): the applicability
network in use, kept loaded between the frames. The site's environment has no torch, so this runs in the GPU
environment (.venv-gpu), on the CPU: the GPU stays free for the trainings.

    python -m nnslr_tools._decider_worker <checkpoint>

One line of JSON in, one out: {'image', 'width', 'height', 'boxes': [{'box', 'family'}], 'speed'} gives {'answers': [...]},
per box [applicability, probability], or null for a box read as not a sign. A first line {'ready': true} says it is
loaded. It leaves after IDLE_SECONDS without a frame; the site starts it again at the next one."""
from __future__ import annotations

import json
import select
import sys
from pathlib import Path

IDLE_SECONDS = 600
THREADS = 2


def answers(decider, request: dict) -> list:
    from PIL import Image
    ruled = [i for i, item in enumerate(request['boxes']) if item['family'] != 'not_a_sign']
    found: list = [None] * len(request['boxes'])
    if ruled:
        with Image.open(request['image']) as opened:
            image = opened.convert('RGB').resize((request['width'], request['height']))
        said = decider.decide(image, [request['boxes'][i]['box'] for i in ruled], request['speed'])
        for i, answer in zip(ruled, said):
            found[i] = list(answer)
    return found


def main(checkpoint: str) -> None:
    import torch
    from nnslr_tools.context_experiment import Decider
    torch.set_num_threads(THREADS)
    decider = Decider(Path(checkpoint), torch.device('cpu'))
    print(json.dumps({'ready': True}), flush=True)
    while select.select([sys.stdin], [], [], IDLE_SECONDS)[0]:
        line = sys.stdin.readline()
        if not line:
            return
        print(json.dumps({'answers': answers(decider, json.loads(line))}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
