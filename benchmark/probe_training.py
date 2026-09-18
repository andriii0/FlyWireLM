from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flylm.data import make_tiny_stream
from flylm.model import load_flywire_model
from flylm.training import assert_frozen_connectome, train_step


def probe_configuration(
    graph: Path,
    embedding_dim: int,
    batch_size: int,
    bptt: int,
    recurrent_steps: int,
    alpha: float,
    recurrent_scale: float,
    learning_rate: float,
) -> dict[str, float | int]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    model = load_flywire_model(
        graph,
        device,
        embedding_dim,
        recurrent_steps,
        alpha,
        recurrent_scale,
    )
    assert_frozen_connectome(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stream = make_tiny_stream(batch_size, bptt, device)
    inputs = stream[:bptt]
    targets = stream[1 : bptt + 1]
    state = model.initial_state(batch_size)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss, accuracy, _ = train_step(
        model, optimizer, inputs, targets, state
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "batch_size": batch_size,
        "bptt": bptt,
        "recurrent_steps": recurrent_steps,
        "seconds": elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "initial_loss": loss,
        "initial_accuracy": accuracy,
    }
    del optimizer, model, stream, inputs, targets, state
    torch.cuda.empty_cache()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Measure one full-connectome training step"
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=project_dir / "datasets/processed/flywire_v783_graph.npz",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--recurrent-steps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--recurrent-scale", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    result = probe_configuration(
        **vars(arguments),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
