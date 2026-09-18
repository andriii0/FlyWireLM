from __future__ import annotations

import math
from pathlib import Path

import torch


TINY_PATTERN = b"the fly brain says hello. the fly learns.\n"


def batchify(
    data: bytes,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.tensor(list(data), dtype=torch.long)
    usable = (len(values) // batch_size) * batch_size
    if usable <= batch_size:
        raise ValueError("corpus is too small for the selected batch size")
    return (
        values[:usable]
        .view(batch_size, -1)
        .transpose(0, 1)
        .contiguous()
        .to(device)
    )


def load_corpus_range(path: Path, offset: int, byte_count: int) -> bytes:
    corpus = path.read_bytes()[offset:]
    return corpus[:byte_count] if byte_count else corpus


def make_tiny_stream(
    batch_size: int,
    bptt: int,
    device: torch.device,
) -> torch.Tensor:
    minimum_bytes = batch_size * (bptt * 8 + 1)
    repeats = math.ceil(minimum_bytes / len(TINY_PATTERN))
    return batchify(TINY_PATTERN * repeats, batch_size, device)
