#!/usr/bin/env python3
"""Generate missing thumbnails for active UI metadata."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common import hdfs
from src.common.image_utils import image_info_and_thumbnail
from src.search.hnsw_index import load_active_metadata

THUMBNAIL_UI_ROOT = os.getenv("THUMBNAIL_UI_ROOT", "/photos/thumbnails/team_gallery")
UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
MAX_THUMBNAILS = int(os.getenv("MAX_THUMBNAILS", "0"))  # 0 = all


def thumbnail_path_for(image_id: str) -> str:
    shard = str(abs(hash(image_id)) % 10)
    return f"{THUMBNAIL_UI_ROOT.rstrip('/')}/{shard}/{image_id}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=MAX_THUMBNAILS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = load_active_metadata()
    if df.empty:
        raise SystemExit("No active metadata found. Run scripts/import_existing_metadata.py first.")
    updated = []
    created = 0
    failed = 0
    for row in df.to_dict("records"):
        image_id = str(row.get("image_id"))
        thumb_uri = row.get("thumbnail_uri")
        if not thumb_uri or str(thumb_uri) == "nan":
            thumb_uri = hdfs.to_hdfs_uri(thumbnail_path_for(image_id))
        if (not args.force) and hdfs.exists(str(thumb_uri)):
            row["thumbnail_uri"] = thumb_uri
            updated.append(row)
            continue
        if args.limit and created >= args.limit:
            row["thumbnail_uri"] = thumb_uri
            updated.append(row)
            continue
        try:
            image_data = hdfs.read_bytes(str(row.get("image_uri")))
            info, thumb = image_info_and_thumbnail(image_data, max_size=256)
            hdfs.write_bytes(thumb_uri, thumb, overwrite=True)
            row["thumbnail_uri"] = thumb_uri
            if not row.get("width"):
                row["width"] = info.get("width", 0)
            if not row.get("height"):
                row["height"] = info.get("height", 0)
            created += 1
            if created % 1000 == 0:
                print(f"Created {created} thumbnails")
        except Exception as exc:
            failed += 1
            print(f"Thumbnail failed for {image_id}: {exc}")
        updated.append(row)

    out = pd.DataFrame(updated).drop_duplicates(subset=["image_id"], keep="last")
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    path = hdfs.write_dataframe_parquet(out, f"{UI_ACTIVE_METADATA_ROOT.rstrip('/')}/{date_part}", filename="ui_active_with_thumbnails.parquet")
    print(json.dumps({"rows": len(out), "created": created, "failed": failed, "output": path}, indent=2))


if __name__ == "__main__":
    main()
