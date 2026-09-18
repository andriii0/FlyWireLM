from __future__ import annotations

import argparse
from pathlib import Path

from flylm.config import TrainingConfig
from flylm.training import FlyLmTrainer


def parse_config(argv: list[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train the minimal frozen FlyWire language model"
    )
    project_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--graph",
        type=Path,
        default=project_dir / "datasets/processed/flywire_v783_graph.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "artifacts/tiny",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-offset", type=int, default=0)
    parser.add_argument("--corpus-bytes", type=int, default=0)
    parser.add_argument("--validation-corpus", type=Path)
    parser.add_argument("--validation-bytes", type=int, default=0)
    parser.add_argument("--generation-seed", default=" = ")
    parser.add_argument("--generation-length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--recurrent-steps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--recurrent-scale", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--target-accuracy", type=float, default=0.995)
    parser.add_argument(
        "--target-generation-accuracy",
        type=float,
        default=0.95,
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return TrainingConfig(**vars(parser.parse_args(argv)))


def main() -> None:
    FlyLmTrainer(parse_config()).run()


if __name__ == "__main__":
    main()
