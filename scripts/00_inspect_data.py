#!/usr/bin/env python3
"""Inspect the external Parquet dataset without modifying it.

The dataset is expected in ../parquet_discharge relative to the repository
root. Files are opened explicitly in binary read-only mode ("rb").
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT.parent / "parquet_discharge"


def inspect_parquet(path: Path, sample_rows: int = 3) -> None:
    """Print Parquet metadata, schema, and a small sample in read-only mode."""
    print(f"\n=== {path.name} ===")

    # Explicit read-only file descriptor. No code in this script opens a
    # Parquet file for writing, appending, truncating, or in-place editing.
    with path.open("rb") as stream:
        parquet_file = pq.ParquetFile(stream)

        metadata = parquet_file.metadata
        print(f"rows: {metadata.num_rows}")
        print(f"row groups: {metadata.num_row_groups}")
        print(f"columns: {metadata.num_columns}")
        print("schema:")
        print(parquet_file.schema_arrow)

        if sample_rows > 0 and metadata.num_row_groups > 0:
            sample = parquet_file.read_row_group(0).slice(0, sample_rows)
            print(f"first {sample.num_rows} row(s):")
            for row in sample.to_pylist():
                print(row)


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {DATA_DIR}\n"
            "Expected ../parquet_discharge next to the repository."
        )

    paths = sorted(DATA_DIR.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No Parquet files found in {DATA_DIR}")

    print(f"dataset: {DATA_DIR}")
    print(f"files: {len(paths)}")

    for path in paths:
        inspect_parquet(path)


if __name__ == "__main__":
    main()
