from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmark.operations import (
    correctness_test,
    csr_propagate,
    edge_propagate,
)


ALPHA = 0.5
RECURRENT_SCALE = 0.9
Propagation = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class BenchmarkConfig:
    graph: Path
    output: Path
    batch_sizes: list[int]
    methods: list[str]
    warmup: int = 10
    repeats: int = 30

    def validate(self) -> None:
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if self.warmup < 0:
            raise ValueError("warmup cannot be negative")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")


@dataclass(frozen=True)
class GraphData:
    node_count: int
    edge_count: int
    adjacency: torch.Tensor
    pre: torch.Tensor
    post: torch.Tensor
    values: torch.Tensor


class GraphBenchmark:
    def __init__(self, config: BenchmarkConfig) -> None:
        config.validate()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the full graph benchmark")
        self.config = config
        self.device = torch.device("cuda")

    def run(self) -> None:
        print(f"PyTorch: {torch.__version__}", flush=True)
        print(f"CUDA runtime: {torch.version.cuda}", flush=True)
        print(
            f"GPU: {torch.cuda.get_device_name(self.device)}",
            flush=True,
        )
        correctness_test(self.device)
        print("Synthetic correctness test: passed", flush=True)

        graph = self._load_graph()
        print(
            f"Full graph: {graph.node_count:,} nodes, "
            f"{graph.edge_count:,} directed connections",
            flush=True,
        )
        methods = self._operations(graph)
        results: list[dict[str, float | int | str]] = []

        for method_name in self.config.methods:
            method_failed = False
            for batch_size in self.config.batch_sizes:
                if method_failed:
                    break
                for recurrent_update in (False, True):
                    for backward in (False, True):
                        try:
                            result = self._timed_run(
                                methods[method_name],
                                graph,
                                batch_size,
                                recurrent_update,
                                backward,
                            )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(
                                f"{method_name} batch={batch_size}: CUDA OOM",
                                flush=True,
                            )
                            method_failed = True
                            break
                        except RuntimeError as error:
                            torch.cuda.empty_cache()
                            print(
                                f"{method_name} batch={batch_size} "
                                f"unsupported/error: {error}",
                                flush=True,
                            )
                            method_failed = True
                            break

                        result["method"] = method_name
                        elapsed_seconds = float(result["median_ms"]) / 1000.0
                        result["connections_per_second"] = (
                            graph.edge_count * batch_size / elapsed_seconds
                        )
                        result["neurons_per_second"] = (
                            graph.node_count * batch_size / elapsed_seconds
                        )
                        results.append(result)
                        self._print_result(method_name, result)
                    if method_failed:
                        break

        self._save_results(graph, results)

    def _load_graph(self) -> GraphData:
        graph = np.load(self.config.graph)
        node_count = len(graph["root_ids"])
        crow = torch.from_numpy(graph["crow_indices"]).to(self.device)
        col = torch.from_numpy(graph["col_indices"]).to(self.device)
        values = torch.from_numpy(graph["values"]).to(self.device)
        pre = torch.from_numpy(graph["pre_indices"]).to(self.device)
        post = torch.from_numpy(graph["post_indices"]).to(self.device)
        adjacency = torch.sparse_csr_tensor(
            crow,
            col,
            values,
            size=(node_count, node_count),
            device=self.device,
            check_invariants=False,
        )
        return GraphData(
            node_count,
            len(values),
            adjacency,
            pre,
            post,
            values,
        )

    @staticmethod
    def _operations(graph: GraphData) -> dict[str, Propagation]:
        return {
            "csr": lambda state: csr_propagate(graph.adjacency, state),
            "edge_index_add": lambda state: edge_propagate(
                graph.pre,
                graph.post,
                graph.values,
                state,
            ),
        }

    def _timed_run(
        self,
        operation: Propagation,
        graph: GraphData,
        batch_size: int,
        recurrent_update: bool,
        backward: bool,
    ) -> dict[str, float | int | str]:
        state = torch.randn(
            graph.node_count,
            batch_size,
            device=self.device,
            requires_grad=backward,
        )
        external_input = None
        if recurrent_update:
            external_input = torch.randn(
                graph.node_count,
                batch_size,
                device=self.device,
                requires_grad=backward,
            )

        def step() -> torch.Tensor:
            propagated = operation(state)
            if recurrent_update:
                assert external_input is not None
                output = (1.0 - ALPHA) * state + ALPHA * torch.tanh(
                    RECURRENT_SCALE * propagated + external_input
                )
            else:
                output = propagated
            if backward:
                output.square().mean().backward()
                state.grad = None
                if external_input is not None:
                    external_input.grad = None
            return output

        for _ in range(self.config.warmup):
            step()
        self._synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)

        samples: list[float] = []
        for _ in range(self.config.repeats):
            self._synchronize()
            started = time.perf_counter()
            step()
            self._synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)

        return {
            "batch_size": batch_size,
            "mode": "forward_backward" if backward else "forward",
            "operation": (
                "recurrent_update" if recurrent_update else "matmul"
            ),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "p95_ms": float(np.percentile(samples, 95)),
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            ),
        }

    @staticmethod
    def _synchronize() -> None:
        torch.cuda.synchronize()

    @staticmethod
    def _print_result(
        method_name: str,
        result: dict[str, float | int | str],
    ) -> None:
        print(
            f"{method_name:16s} batch={result['batch_size']:4d} "
            f"{result['operation']:16s} {result['mode']:16s} "
            f"{result['median_ms']:9.3f} ms "
            f"peak={result['peak_vram_mib']:8.1f} MiB",
            flush=True,
        )

    def _save_results(
        self,
        graph: GraphData,
        results: list[dict[str, float | int | str]],
    ) -> None:
        payload = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device),
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "alpha_for_timing": ALPHA,
            "recurrent_scale_for_timing": RECURRENT_SCALE,
            "warmup": self.config.warmup,
            "repeats": self.config.repeats,
            "results": results,
        }
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        self.config.output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"Saved {self.config.output}", flush=True)
