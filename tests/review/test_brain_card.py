"""Train › Brain says what the brain gets on the car and what only its training has (web/src/control/brain-card.ts): its
groups must be the car's INPUT_SIZES, each with its size and with what a training of today does with it."""
from __future__ import annotations

import re
from pathlib import Path

from nnslr_tools.brain_targets import DRIVER_INPUTS, INPUT_SIZES, LEFT_OUT, OUT_REASONS
from nnslr_tools.brain_train import ROAD_INPUTS

CARD = Path(__file__).resolve().parents[2] / 'web/src/control/brain-card.ts'


def _card() -> str:
    return CARD.read_text(encoding='utf-8')


def test_the_card_lists_every_input_group_with_its_size_and_its_use():
    groups = {name: (int(size), use) for name, size, use in
              re.findall(r"\{name: '(\w+)', size: (\d+), use: '(\w+)'", _card())}
    assert {name: size for name, (size, _) in groups.items()} == INPUT_SIZES
    expected = {name: 'never' if name in LEFT_OUT else 'driver' if name in DRIVER_INPUTS else 'car' for name in INPUT_SIZES}
    assert {name: use for name, (_, use) in groups.items()} == expected
    assert {name for name, use in expected.items() if use == 'car'} == set(ROAD_INPUTS)


def test_the_card_lists_every_reason_a_second_answers_nothing():
    out = _card().split('export const BRAIN_OUT:')[1]
    assert sorted(re.findall(r"\{name: '(\w+)'", out)) == sorted(OUT_REASONS.values())
