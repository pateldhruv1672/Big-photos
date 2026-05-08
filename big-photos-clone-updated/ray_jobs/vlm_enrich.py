from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

try:
    import ray
except Exception:  # pragma: no cover
    ray = None

from src.common import hdfs
from src.common.embeddings import record_embedding
from src.common.image_utils import infer_fallback_category, infer_fallback_labels, make_caption

BASIC_MIR_ROOT = os.getenv("MIRFLICKR25K_BASIC_ROOT", "/photos/metadata/mirflickr25k/basic")
UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
ENRICHED_METADATA_ROOT = os.getenv("ENRICHED_METADATA_ROOT", "/photos/metadata/enriched")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))
MAX_VLM_IMAGES = int(os.getenv("MAX_VLM_IMAGES", "0"))
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "auto")


def deterministic_label(file_name: str, tags: List[str]) -> Dict[str, Any]:
    """Ray-only deterministic labeling from existing metadata/tags."""
    if any(str(t).strip().lower() == "other" for t in tags):
        category = "other"
        labels = ["other"]
    else:
        category = infer_fallback_category(" ".join(tags) or file_name)
        labels = tags if tags else infer_fallback_labels(file_name)
    return {
        "caption": make_caption(category, labels, file_name),
        "vlm_labels": labels,
        "objects": labels[:3],
        "category": category,
        "quality_score": 0.75,
        "vlm_source": "ray_deterministic",
    }


def enrich_row(row: Dict[str, Any]) -> Dict[str, Any]:
    tags_raw = row.get("tags")
    if tags_raw is None or (isinstance(tags_raw, float) and pd.isna(tags_raw)):
        tags_raw = row.get("labels")
    if tags_raw is None or (isinstance(tags_raw, float) and pd.isna(tags_raw)):
        tags = []
    elif isinstance(tags_raw, list):
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
    elif hasattr(tags_raw, "tolist"):
        tags = [str(t).strip() for t in tags_raw.tolist() if str(t).strip()]
    else:
        s = str(tags_raw).strip()
        if s.startswith("[") and s.endswith("]"):
            cleaned = s.strip("[]").replace("'", "").replace('"', "")
            tags = [x.strip() for x in cleaned.split(",") if x.strip()]
        else:
            tags = [s] if s else []
    result = deterministic_label(str(row.get("file_name") or row.get("image_id")), tags)
    embedding = record_embedding(
        caption=result["caption"], labels=result["vlm_labels"], category=result["category"], tags=tags, dim=EMBEDDING_DIM
    )
    return {
        "image_id": str(row.get("image_id")),
        "user_id": str(row.get("user_id") or "team_gallery"),
        "image_uri": str(row.get("image_uri")),
        "thumbnail_uri": row.get("thumbnail_uri"),
        "caption": result["caption"],
        "vlm_labels": result["vlm_labels"],
        "labels": result["vlm_labels"],
        "objects": result["objects"],
        "category": result["category"],
        "embedding": embedding,
        "embedding_dim": len(embedding),
        "quality_score": result["quality_score"],
        "processed_at": datetime.now(timezone.utc),
        "vlm_source": result["vlm_source"],
        "width": int(row.get("width") or 0),
        "height": int(row.get("height") or 0),
        "file_size": int(row.get("file_size") or 0),
        "taken_at": row.get("taken_at"),
        "location": row.get("location") or "Unknown",
        "deleted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=UI_ACTIVE_METADATA_ROOT)
    parser.add_argument("--output", default=ENRICHED_METADATA_ROOT)
    parser.add_argument("--limit", type=int, default=MAX_VLM_IMAGES)
    args = parser.parse_args()

    df = hdfs.read_parquet_dataset(args.input, limit=args.limit if args.limit > 0 else None)
    if df.empty:
        raise SystemExit(f"No basic metadata found at {args.input}")
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    records = df.to_dict("records")

    if ray is not None:
        try:
            ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
        except Exception:
            ray.init(ignore_reinit_error=True)

        @ray.remote(num_cpus=1)
        def enrich_remote(r):
            return enrich_row(r)

        # Stream results to avoid large in-memory object refs in constrained containers.
        enriched = []
        for r in records:
            enriched.append(ray.get(enrich_remote.remote(r)))
    else:
        enriched = [enrich_row(r) for r in records]

    out = pd.DataFrame(enriched)
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    path = hdfs.write_dataframe_parquet(out, f"{args.output.rstrip('/')}/{date_part}")
    print(json.dumps({"rows": len(out), "output": path}, indent=2))


if __name__ == "__main__":
    main()
