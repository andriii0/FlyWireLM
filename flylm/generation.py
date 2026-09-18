from __future__ import annotations

import math

import torch

from flylm.data import TINY_PATTERN
from flylm.model import FrozenFlyWireLM


@torch.no_grad()
def generate(
    model: FrozenFlyWireLM,
    seed: bytes,
    length: int,
    temperature: float = 0.0,
) -> bytes:
    if not seed:
        raise ValueError("generation seed cannot be empty")
    device = model.embedding.weight.device
    state = model.initial_state(1)
    logits = None
    output = bytearray(seed)

    for value in seed:
        token = torch.tensor([[value]], dtype=torch.long, device=device)
        logits, state = model(token, state)

    for _ in range(length):
        assert logits is not None
        next_logits = logits[-1, 0]
        if temperature > 0.0:
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            next_byte = int(torch.multinomial(probabilities, 1))
        else:
            next_byte = int(next_logits.argmax())
        output.append(next_byte)
        token = torch.tensor([[next_byte]], dtype=torch.long, device=device)
        logits, state = model(token, state)

    return bytes(output)


def generation_match(generation: bytes) -> float:
    repeats = math.ceil(len(generation) / len(TINY_PATTERN))
    expected = (TINY_PATTERN * repeats)[: len(generation)]
    matches = sum(
        actual == wanted
        for actual, wanted in zip(generation, expected)
    )
    return matches / len(generation)
