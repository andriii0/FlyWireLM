from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as parquet


DATASET = "roneneldan/TinyStories"
CONFIG = "default"
API_URL = "https://datasets-server.huggingface.co/parquet"
MIB = 1024 * 1024


def sha256(path: Path, chunk_size: int = 8 * MIB) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def export_text(
    parquet_paths: list[Path],
    destination: Path,
    byte_limit: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with destination.open("wb") as output:
        for parquet_path in parquet_paths:
            source = parquet.ParquetFile(parquet_path)
            for batch in source.iter_batches(columns=["text"], batch_size=1024):
                for value in batch.column(0).to_pylist():
                    story = value.rstrip("\r\n").encode("utf-8") + b"\n\n"
                    if bytes_written and bytes_written + len(story) > byte_limit:
                        return bytes_written
                    output.write(story)
                    bytes_written += len(story)
    return bytes_written


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download a TinyStories subset")
    parser.add_argument("--train-mib", type=int, default=25)
    parser.add_argument("--validation-mib", type=int, default=1)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=dataset_dir / "raw/tinystories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_dir / "corpora/tinystories-25m",
    )
    args = parser.parse_args()
    if args.train_mib <= 0 or args.validation_mib <= 0:
        raise ValueError("subset sizes must be positive")

    query = urllib.parse.urlencode({"dataset": DATASET})
    with urllib.request.urlopen(f"{API_URL}?{query}") as response:
        payload = json.load(response)
    files = [
        item
        for item in payload["parquet_files"]
        if item["config"] == CONFIG
    ]

    manifest: dict[str, object] = {
        "dataset": DATASET,
        "config": CONFIG,
        "splits": {},
    }
    for split, byte_limit in (
        ("train", args.train_mib * MIB),
        ("validation", args.validation_mib * MIB),
    ):
        split_files = sorted(
            (item for item in files if item["split"] == split),
            key=lambda item: item["filename"],
        )
        if not split_files:
            raise RuntimeError(f"TinyStories {split} split is unavailable")

        downloaded: list[Path] = []
        available_bytes = 0
        for item in split_files:
            parquet_path = args.raw_dir / split / item["filename"]
            print(f"Downloading {split}/{item['filename']}...")
            download(item["url"], parquet_path)
            downloaded.append(parquet_path)
            available_bytes += int(item["size"])
            if available_bytes >= byte_limit:
                break

        text_path = args.output_dir / f"{split}.txt"
        text_bytes = export_text(downloaded, text_path, byte_limit)
        manifest["splits"][split] = {
            "requested_bytes": byte_limit,
            "text_bytes": text_bytes,
            "text_sha256": sha256(text_path),
            "source_files": [path.name for path in downloaded],
        }
        print(f"Prepared {text_path}: {text_bytes:,} UTF-8 bytes")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
