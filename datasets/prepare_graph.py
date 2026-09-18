from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


DATA_URLS = {
    "proofread_root_ids_783.npy": (
        "https://zenodo.org/records/10676866/files/"
        "proofread_root_ids_783.npy?download=1"
    ),
    "proofread_connections_783.feather": (
        "https://zenodo.org/records/10676866/files/"
        "proofread_connections_783.feather?download=1"
    ),
    "Supplemental_file1_neuron_annotations.tsv": (
        "https://raw.githubusercontent.com/flyconnectome/"
        "flywire_annotations/v2.1.0/supplemental_files/"
        "Supplemental_file1_neuron_annotations.tsv"
    ),
}

EXPECTED_MD5 = {
    "proofread_root_ids_783.npy": "e0e6c19732fd8c7a4e39a2d170105421",
    "proofread_connections_783.feather": "f48f972d262323a102aed49af1396b8a",
}


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Using existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            copied = 0
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                copied += len(chunk)
                if total:
                    print(
                        f"  {copied / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                        end="\r",
                    )
        partial.replace(destination)
        print()
    finally:
        if partial.exists():
            partial.unlink()


def ensure_source_data(data_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename, url in DATA_URLS.items():
        path = data_dir / filename
        download(url, path)
        digest = file_md5(path)
        expected = EXPECTED_MD5.get(filename)
        if expected is not None and digest != expected:
            raise RuntimeError(
                f"MD5 mismatch for {filename}: expected {expected}, got {digest}"
            )
        hashes[filename] = digest
        print(f"Verified {filename}: md5={digest}")
    return hashes


def map_root_ids(root_ids: np.ndarray, values: np.ndarray, label: str) -> np.ndarray:
    index = pd.Index(root_ids)
    mapped = index.get_indexer(values)
    missing = int((mapped < 0).sum())
    if missing:
        raise RuntimeError(f"{missing} {label} root IDs are absent from proofread roots")
    return mapped.astype(np.int64, copy=False)


def build_graph(data_dir: Path, output_dir: Path) -> None:
    hashes = ensure_source_data(data_dir)
    roots_path = data_dir / "proofread_root_ids_783.npy"
    connections_path = data_dir / "proofread_connections_783.feather"
    annotations_path = data_dir / "Supplemental_file1_neuron_annotations.tsv"

    root_ids = np.load(roots_path).astype(np.int64, copy=False)
    if root_ids.ndim != 1 or len(np.unique(root_ids)) != len(root_ids):
        raise RuntimeError("proofread root IDs must be a unique one-dimensional array")
    print(f"Proofread neurons: {len(root_ids):,}")

    print("Reading connection columns...")
    connections = pd.read_feather(
        connections_path,
        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"],
    )
    source_rows = len(connections)
    print(f"Connection-neuropil rows: {source_rows:,}")
    print("Aggregating neuropils by directed neuron pair...")
    grouped = (
        connections.groupby(
            ["pre_pt_root_id", "post_pt_root_id"],
            sort=False,
            observed=True,
            as_index=False,
        )["syn_count"]
        .sum()
    )
    del connections
    pair_count_before_threshold = len(grouped)
    strong = grouped.loc[grouped["syn_count"] >= 5].copy()
    del grouped
    edge_count = len(strong)
    print(f"Neuron pairs before threshold: {pair_count_before_threshold:,}")
    print(f"Directed connections with syn_count >= 5: {edge_count:,}")

    pre_index = map_root_ids(
        root_ids,
        strong["pre_pt_root_id"].to_numpy(dtype=np.int64, copy=False),
        "presynaptic",
    )
    post_index = map_root_ids(
        root_ids,
        strong["post_pt_root_id"].to_numpy(dtype=np.int64, copy=False),
        "postsynaptic",
    )
    syn_count = strong["syn_count"].to_numpy(dtype=np.float64, copy=False)
    del strong

    order = np.lexsort((pre_index, post_index))
    pre_index = pre_index[order]
    post_index = post_index[order]
    syn_count = syn_count[order]

    incoming_sum = np.bincount(
        post_index, weights=syn_count, minlength=len(root_ids)
    )
    if np.any(incoming_sum[post_index] <= 0):
        raise RuntimeError("non-positive incoming synapse sum encountered")
    values = (syn_count / incoming_sum[post_index]).astype(np.float32)
    syn_count = syn_count.astype(np.int64)

    row_counts = np.bincount(post_index, minlength=len(root_ids))
    crow_indices = np.empty(len(root_ids) + 1, dtype=np.int64)
    crow_indices[0] = 0
    np.cumsum(row_counts, out=crow_indices[1:])

    annotations = pd.read_csv(annotations_path, sep="\t", low_memory=False)
    required = {"root_id", "flow"}
    missing_columns = required.difference(annotations.columns)
    if missing_columns:
        raise RuntimeError(f"annotation columns missing: {sorted(missing_columns)}")
    annotations = annotations.loc[:, ["root_id", "flow"]].dropna(subset=["root_id"])
    annotations["root_id"] = annotations["root_id"].astype(np.int64)
    annotations = annotations.drop_duplicates(subset="root_id", keep="first")
    annotation_index = pd.Index(annotations["root_id"])
    locations = annotation_index.get_indexer(root_ids)
    flow = np.full(len(root_ids), "", dtype="U16")
    present = locations >= 0
    flow[present] = (
        annotations["flow"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .to_numpy()[locations[present]]
    )
    afferent_indices = np.flatnonzero(flow == "afferent").astype(np.int64)
    efferent_indices = np.flatnonzero(flow == "efferent").astype(np.int64)
    if not len(afferent_indices) or not len(efferent_indices):
        raise RuntimeError(
            f"invalid flow annotations: {len(afferent_indices)} afferent, "
            f"{len(efferent_indices)} efferent"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "flywire_v783_graph.npz"
    np.savez(
        graph_path,
        root_ids=root_ids,
        crow_indices=crow_indices,
        col_indices=pre_index,
        values=values,
        pre_indices=pre_index,
        post_indices=post_index,
        syn_count=syn_count,
        afferent_indices=afferent_indices,
        efferent_indices=efferent_indices,
    )

    connected = int(np.count_nonzero(np.bincount(pre_index, minlength=len(root_ids)) + row_counts))
    manifest = {
        "dataset": "FlyWire FAFB v783",
        "annotation_release": "v2.1.0",
        "node_count": int(len(root_ids)),
        "source_connection_neuropil_rows": int(source_rows),
        "pair_count_before_threshold": int(pair_count_before_threshold),
        "edge_count": int(edge_count),
        "connected_node_count": connected,
        "threshold": 5,
        "orientation": "A[post, pre]",
        "normalization": "syn_count / sum incoming syn_count per post neuron",
        "afferent_count": int(len(afferent_indices)),
        "efferent_count": int(len(efferent_indices)),
        "source_md5": hashes,
    }
    manifest_path = output_dir / "flywire_v783_graph.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Afferent neurons: {len(afferent_indices):,}")
    print(f"Efferent neurons: {len(efferent_indices):,}")
    print(f"Connected neurons after threshold: {connected:,}")
    print(f"Saved {graph_path}")
    print(f"Saved {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the full FlyWire v783 graph")
    dataset_dir = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=dataset_dir / "raw")
    parser.add_argument("--output-dir", type=Path, default=dataset_dir / "processed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_graph(args.data_dir, args.output_dir)
