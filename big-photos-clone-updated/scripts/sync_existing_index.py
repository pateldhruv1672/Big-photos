#!/usr/bin/env python3
"""Copy an already-built local hnswlib index/id map into HDFS.

Use this when existing metadata and index are already aligned. If not certain,
prefer scripts/build_hnsw_from_metadata.py because rebuilding is safer.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.common import hdfs

VECTOR_INDEX_ROOT = os.getenv("VECTOR_INDEX_ROOT", "/photos/vector_index/current")
EXISTING_INDEX_DIR = os.getenv("EXISTING_INDEX_DIR", "/app/data/existing_index")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=EXISTING_INDEX_DIR)
    parser.add_argument("--hdfs-dir", default=VECTOR_INDEX_ROOT)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    args = parser.parse_args()
    base = Path(args.index_dir)
    index_file = base / "hnsw_index.bin"
    map_file = base / "id_map.json"
    if not index_file.exists() or not map_file.exists():
        raise SystemExit(f"Expected {index_file} and {map_file}; run build_hnsw_from_metadata.py instead.")
    id_map = json.loads(map_file.read_text())
    if not isinstance(id_map, dict) or not id_map:
        raise SystemExit("id_map.json must be a non-empty JSON object mapping integer labels to image_id")
    # Validate integer-like keys.
    for key in id_map.keys():
        int(key)
    hdfs.mkdirs(args.hdfs_dir)
    hdfs.upload_local_file(str(index_file), f"{args.hdfs_dir}/hnsw_index.bin", overwrite=True)
    hdfs.upload_local_file(str(map_file), f"{args.hdfs_dir}/id_map.json", overwrite=True)
    manifest = {
        "embedding_dim": args.embedding_dim,
        "space": "cosine",
        "total_vectors": len(id_map),
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "source": str(base),
    }
    hdfs.write_json(f"{args.hdfs_dir}/manifest.json", manifest, overwrite=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
