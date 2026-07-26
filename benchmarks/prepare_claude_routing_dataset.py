#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from build_claude_routing_grid import is_default_route_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-limit", type=int, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source_records = [
        json.loads(line)
        for line in (args.trace_dir / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ][: args.record_limit]
    valid_records = []
    excluded = []
    for source_index, record in enumerate(source_records):
        source = args.trace_dir / record["file"]
        routes = np.load(source, mmap_mode="r")
        if is_default_route_trace(routes):
            excluded.append(record["file"])
            continue
        valid_records.append((source_index, record, source))
    if not 0 < args.train_count < len(valid_records):
        raise ValueError("Training count must leave valid held-out requests")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
    manifest = []
    for valid_index, (source_index, record, source) in enumerate(valid_records):
        destination = args.output_dir / record["file"]
        shutil.copy2(source, destination)
        manifest.append(
            {
                "file": destination.name,
                "request_hash": hashlib.sha256(
                    record["request_id"].encode()
                ).hexdigest(),
                "route_sha256": file_sha256(destination),
                "source_record_index": source_index,
                "routed_positions": record["routed_positions"],
                "output_tokens": record["output_tokens"],
                "split": "train" if valid_index < args.train_count else "heldout",
            }
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "source_record_count": len(source_records),
                "valid_request_count": len(valid_records),
                "training_request_count": args.train_count,
                "heldout_request_count": len(valid_records) - args.train_count,
                "excluded_default_route_files": excluded,
                "routed_positions": sum(
                    record["routed_positions"] for _, record, _ in valid_records
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
