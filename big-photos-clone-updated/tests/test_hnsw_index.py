from __future__ import annotations

import sys

from src.search.hnsw_index import HNSWManager


def main() -> int:
    mgr = HNSWManager(dim=128, local_dir="/tmp/bigphotos-hnsw-test")
    records = [
        {
            "image_id": "img_a",
            "caption": "beach sunset",
            "labels": ["beach", "sunset"],
            "category": "travel",
            "embedding": [0.01] * 128,
            "deleted": False,
        },
        {
            "image_id": "img_b",
            "caption": "city skyline",
            "labels": ["city", "night"],
            "category": "city",
            "embedding": [0.02] * 128,
            "deleted": False,
        },
    ]
    mgr.rebuild_from_records(records, save=False)
    assert len(mgr.label_to_image) == 2
    hits = mgr.search_text("beach", top_k=1)
    assert len(hits) == 1
    assert hits[0][0] in {"img_a", "img_b"}
    print("hnsw manager smoke test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
