from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.runner import BenchmarkConfig, GraphBenchmark


def parse_config(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark full FlyWire propagation"
    )
    project_dir = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--graph",
        type=Path,
        default=project_dir / "datasets/processed/flywire_v783_graph.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "benchmark/results/results.json",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["csr", "edge_index_add"],
        default=["csr", "edge_index_add"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    return BenchmarkConfig(**vars(parser.parse_args(argv)))


def main() -> None:
    GraphBenchmark(parse_config()).run()


if __name__ == "__main__":
    main()
