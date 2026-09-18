from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPORT_HOPS = (1, 2, 3, 4, 5, 6, 8)


def analyze(graph_path: Path) -> dict[str, object]:
    graph = np.load(graph_path)
    node_count = len(graph["root_ids"])
    pre = graph["pre_indices"]
    post = graph["post_indices"]
    afferent = graph["afferent_indices"]
    efferent = graph["efferent_indices"]

    distance = np.full(node_count, -1, dtype=np.int16)
    distance[afferent] = 0
    frontier = np.zeros(node_count, dtype=bool)
    frontier[afferent] = True
    hop = 0
    node_counts_by_hop: dict[int, int] = {}

    while frontier.any():
        hop += 1
        next_nodes = np.unique(post[frontier[pre]])
        next_nodes = next_nodes[distance[next_nodes] < 0]
        if not len(next_nodes):
            break
        distance[next_nodes] = hop
        frontier.fill(False)
        frontier[next_nodes] = True
        node_counts_by_hop[hop] = int(len(next_nodes))

    efferent_distance = distance[efferent]
    reachable = efferent_distance >= 0
    reached_distances = efferent_distance[reachable]
    requested_hops = list(REPORT_HOPS)
    if len(reached_distances):
        max_distance = int(reached_distances.max())
        requested_hops.extend(
            candidate
            for candidate in range(REPORT_HOPS[-1] + 1, max_distance + 1)
            if candidate not in requested_hops
        )

    cumulative = {
        str(candidate): {
            "count": int(
                ((efferent_distance >= 0) & (efferent_distance <= candidate)).sum()
            ),
            "percent": float(
                100
                * ((efferent_distance >= 0) & (efferent_distance <= candidate)).mean()
            ),
        }
        for candidate in requested_hops
    }
    exact_values, exact_counts = np.unique(reached_distances, return_counts=True)

    return {
        "node_count": int(node_count),
        "edge_count": int(len(pre)),
        "afferent_count": int(len(afferent)),
        "efferent_count": int(len(efferent)),
        "reachable_efferent_count": int(reachable.sum()),
        "reachable_efferent_percent": float(100 * reachable.mean()),
        "unreachable_efferent_count": int((~reachable).sum()),
        "cumulative_efferent_by_hop": cumulative,
        "exact_efferent_distance_counts": {
            str(int(value)): int(count)
            for value, count in zip(exact_values, exact_counts, strict=True)
        },
        "distance_statistics": {
            "minimum": int(reached_distances.min()),
            "median": float(np.median(reached_distances)),
            "mean": float(reached_distances.mean()),
            "p90": float(np.percentile(reached_distances, 90)),
            "p95": float(np.percentile(reached_distances, 95)),
            "maximum": int(reached_distances.max()),
        },
        "newly_reached_nodes_by_hop": {
            str(key): value for key, value in node_counts_by_hop.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Directed FlyWire hop analysis")
    dataset_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--graph", type=Path, default=dataset_dir / "processed/flywire_v783_graph.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=dataset_dir / "processed/reachability.json"
    )
    args = parser.parse_args()

    result = analyze(args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
