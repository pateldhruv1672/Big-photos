#!/usr/bin/env python3
"""Build UI metadata from local shard folders (exif/tags/thumb) and write to HDFS.

Expected local structure:
  data/exif_11db/{0..9}/{id}.txt
  data/tags_11gb/{0..9}/{id}.txt
  data/thumb_11gb/{1..10}/im{id}.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.common import hdfs
from src.common.image_utils import infer_fallback_location, infer_fallback_taken_at
from src.common.metadata_normalizer import normalize_dataframe

PROJECT_DATA_ROOT = os.getenv("PROJECT_DATA_ROOT", "/app/data")
EXIF_ROOT = os.getenv("EXIF_SHARD_ROOT", f"{PROJECT_DATA_ROOT}/exif_11db")
TAGS_ROOT = os.getenv("TAGS_SHARD_ROOT", f"{PROJECT_DATA_ROOT}/tags_11gb")
THUMB_ROOT = os.getenv("THUMB_SHARD_ROOT", f"{PROJECT_DATA_ROOT}/thumb_11gb")
RAW_UI_IMAGE_ROOT = os.getenv("RAW_UI_IMAGE_ROOT", "/photos/raw/team_gallery/images")
IMPORTED_METADATA_ROOT = os.getenv("IMPORTED_METADATA_ROOT", "/photos/metadata/imported")
UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "team_gallery")
MAX_ROWS = int(os.getenv("SHARD_IMPORT_MAX_ROWS", "0"))


def _parse_exif(path: Path) -> Tuple[int, int, Optional[datetime]]:
    width = 0
    height = 0
    taken_at: Optional[datetime] = None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return width, height, taken_at
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "-Image Width" and i + 1 < len(lines):
            try:
                width = int(re.findall(r"\d+", lines[i + 1])[0])
            except Exception:
                pass
        elif s == "-Image Length" and i + 1 < len(lines):
            try:
                height = int(re.findall(r"\d+", lines[i + 1])[0])
            except Exception:
                pass
        elif s in {"-Date and Time (Original)", "-Date and Time"} and i + 1 < len(lines):
            t = lines[i + 1].strip()
            try:
                taken_at = datetime.strptime(t, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return width, height, taken_at


def _parse_tags(path: Path) -> List[str]:
    try:
        tags = [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        return tags[:32]
    except Exception:
        return []


def _thumb_hdfs_uri(image_id: str) -> str:
    # Local thumb shards are 1..10; map image id deterministically.
    shard = (int(image_id) % 10) + 1
    return hdfs.to_hdfs_uri(f"/photos/thumbnails/team_gallery/{shard}/im{int(image_id)}.jpg")


def _image_hdfs_uri(image_id: str) -> str:
    shard = str(int(image_id) % 10)
    return hdfs.to_hdfs_uri(f"{RAW_UI_IMAGE_ROOT.rstrip('/')}/{shard}/{int(image_id)}.jpg")


def build_rows(exif_root: Path, tags_root: Path) -> pd.DataFrame:
    rows = []
    count = 0
    for shard_dir in sorted([p for p in exif_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        shard = shard_dir.name
        for exif_file in sorted(shard_dir.glob("*.txt")):
            image_id = exif_file.stem
            tag_file = tags_root / shard / f"{image_id}.txt"
            width, height, taken_at = _parse_exif(exif_file)
            tags = _parse_tags(tag_file) if tag_file.exists() else []
            image_uri = _image_hdfs_uri(image_id)
            rows.append(
                {
                    "image_id": image_id,
                    "user_id": DEFAULT_USER_ID,
                    "image_uri": image_uri,
                    "thumbnail_uri": _thumb_hdfs_uri(image_id),
                    "file_name": f"{int(image_id)}.jpg",
                    "dataset": "team_gallery",
                    "tags": tags,
                    "labels": tags,
                    # Avoid per-file WebHDFS lookups; size is optional for UI/search.
                    "file_size": 0,
                    "width": width,
                    "height": height,
                    "taken_at": taken_at or infer_fallback_taken_at(f"/{shard}/{image_id}.jpg"),
                    "location": infer_fallback_location(f"/{shard}/{image_id}.jpg"),
                }
            )
            count += 1
            if MAX_ROWS and count >= MAX_ROWS:
                return pd.DataFrame(rows)
            if count % 10000 == 0:
                print(f"Prepared {count} shard-metadata rows")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exif-root", default=EXIF_ROOT)
    parser.add_argument("--tags-root", default=TAGS_ROOT)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    exif_root = Path(args.exif_root)
    tags_root = Path(args.tags_root)
    if not exif_root.exists():
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "exif_root_missing", "path": str(exif_root)}, indent=2))
            return
        raise SystemExit(f"Missing exif root: {exif_root}")

    df = build_rows(exif_root, tags_root)
    if df.empty:
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "no_rows_built"}, indent=2))
            return
        raise SystemExit("No shard metadata rows built")

    normalized = normalize_dataframe(df, dataset="team_gallery", raw_root=RAW_UI_IMAGE_ROOT)
    normalized = normalized[normalized["deleted"] != True].drop_duplicates(subset=["image_id"], keep="last")
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    imported_path = hdfs.write_dataframe_parquet(
        normalized,
        f"{IMPORTED_METADATA_ROOT.rstrip('/')}/{date_part}",
        filename="imported_shard_metadata.parquet",
    )
    active_path = hdfs.write_dataframe_parquet(
        normalized,
        f"{UI_ACTIVE_METADATA_ROOT.rstrip('/')}/{date_part}",
        filename="ui_active_shard_metadata.parquet",
    )
    print(json.dumps({"rows": len(normalized), "imported_path": imported_path, "ui_active_path": active_path}, indent=2))


if __name__ == "__main__":
    main()
