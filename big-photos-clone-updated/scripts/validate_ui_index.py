#!/usr/bin/env python3
from __future__ import annotations

import json
import os

from src.common import hdfs
from src.search.hnsw_index import HNSWManager, load_active_metadata

VECTOR_INDEX_ROOT = os.getenv("VECTOR_INDEX_ROOT", "/photos/vector_index/current")


def main() -> None:
    df = load_active_metadata()
    mgr = HNSWManager()
    mgr.load()
    missing_meta = []
    meta_ids = set(df["image_id"].astype(str).tolist()) if not df.empty and "image_id" in df.columns else set()
    for image_id in mgr.label_to_image.values():
        if image_id not in meta_ids:
            missing_meta.append(image_id)
    report = {
        "metadata_rows": len(df),
        "hnsw_vectors": len(mgr.label_to_image),
        "missing_metadata_for_index": missing_meta[:20],
        "missing_metadata_count": len(missing_meta),
        "has_hnsw_index_bin": hdfs.exists(f"{VECTOR_INDEX_ROOT}/hnsw_index.bin"),
        "has_id_map": hdfs.exists(f"{VECTOR_INDEX_ROOT}/id_map.json"),
        "has_manifest": hdfs.exists(f"{VECTOR_INDEX_ROOT}/manifest.json"),
    }
    print(json.dumps(report, indent=2))
    if report["metadata_rows"] == 0 or report["hnsw_vectors"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
