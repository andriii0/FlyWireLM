from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as parquet


DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"
API_URL = "https://datasets-server.huggingface.co/parquet"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
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


def export_text(parquet_path: Path, destination: Path) -> int:
    rows = parquet.read_table(parquet_path, columns=["text"])["text"].to_pylist()
    text = "\n".join(row.rstrip("\r\n") for row in rows) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return len(rows)


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download WikiText-2 Raw")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=dataset_dir / "raw/wikitext-2-raw-v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_dir / "corpora/wikitext-2-raw-v1",
    )
    args = parser.parse_args()

    query = urllib.parse.urlencode({"dataset": DATASET})
    with urllib.request.urlopen(f"{API_URL}?{query}") as response:
        payload = json.load(response)
    files = [
        item
        for item in payload["parquet_files"]
        if item["config"] == CONFIG
    ]
    if {item["split"] for item in files} != {"train", "validation", "test"}:
        raise RuntimeError("WikiText-2 Raw splits are incomplete")

    manifest: dict[str, object] = {
        "dataset": DATASET,
        "config": CONFIG,
        "splits": {},
    }
    for item in sorted(files, key=lambda value: value["split"]):
        split = item["split"]
        parquet_path = args.raw_dir / f"{split}.parquet"
        text_path = args.output_dir / f"{split}.txt"
        print(f"Preparing {split}...")
        download(item["url"], parquet_path)
        rows = export_text(parquet_path, text_path)
        manifest["splits"][split] = {
            "rows": rows,
            "parquet_bytes": parquet_path.stat().st_size,
            "text_bytes": text_path.stat().st_size,
            "parquet_sha256": sha256(parquet_path),
            "text_sha256": sha256(text_path),
            "source_url": item["url"],
        }
        print(f"  {rows:,} rows, {text_path.stat().st_size:,} UTF-8 bytes")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
