#!/usr/bin/env python3
from __future__ import annotations

import json

from src.search.hnsw_index import build_from_hdfs

if __name__ == "__main__":
    manager = build_from_hdfs()
    print(json.dumps({"vectors": len(manager.label_to_image), "hdfs_dir": manager.hdfs_dir}, indent=2))
