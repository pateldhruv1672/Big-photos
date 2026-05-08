#!/usr/bin/env python3
"""Import existing metadata files from local filesystem into HDFS Parquet.

Expected local path inside container:
  /app/data/existing_metadata

Supported files:
  *.csv, *.json, *.jsonl, *.parquet

The script normalizes all rows into the UI schema and writes both:
  /photos/metadata/imported/date=YYYY-MM-DD/imported_metadata.parquet
  /photos/metadata/ui_active/date=YYYY-MM-DD/ui_active.parquet

If no external metadata is present, it can build a metadata snapshot by scanning
HDFS raw images. That lets the UI still work with your existing 100k-image HDFS
collection while enriched fields are generated deterministically.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from src.common import hdfs
from src.common.image_utils import image_id_from_path, infer_fallback_location, infer_fallback_taken_at
from src.common.metadata_normalizer import normalize_dataframe

METADATA_INPUT_DIR = os.getenv("METADATA_INPUT_DIR", "/app/data/existing_metadata")
RAW_UI_IMAGE_ROOT = os.getenv("RAW_UI_IMAGE_ROOT", "/photos/raw/team_gallery/images")
IMPORTED_METADATA_ROOT = os.getenv("IMPORTED_METADATA_ROOT", "/photos/metadata/imported")
UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "team_gallery")
SCAN_HDFS_IF_MISSING = os.getenv("SCAN_HDFS_IF_METADATA_MISSING", "true").lower() == "true"
MAX_SCAN_FILES = int(os.getenv("MAX_IMPORT_SCAN_FILES", "0"))  # 0 = all


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                data = data["items"]
            else:
                data = [data]
        return pd.DataFrame(data)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def read_local_metadata(input_dir: str) -> pd.DataFrame:
    base = Path(input_dir)
    if not base.exists():
        return pd.DataFrame()
    files: List[Path] = []
    for suffix in ["*.csv", "*.json", "*.jsonl", "*.parquet"]:
        files.extend(base.rglob(suffix))
    frames = []
    for file in sorted(files):
        try:
            df = _read_one(file)
            if not df.empty:
                df["_source_file"] = str(file)
                frames.append(df)
                print(f"Loaded {len(df)} rows from {file}")
        except Exception as exc:
            print(f"Skipping {file}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def scan_hdfs_images(raw_root: str) -> pd.DataFrame:
    print(f"Scanning HDFS images under {raw_root}")
    rows = []
    count = 0
    for path in hdfs.walk(raw_root):
        lower = path.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        image_id = image_id_from_path(path)
        file_name = path.rstrip("/").split("/")[-1]
        status = hdfs.file_status(path) or {}
        rows.append(
            {
                "image_id": image_id,
                "user_id": DEFAULT_USER_ID,
                "image_uri": hdfs.to_hdfs_uri(path),
                "file_name": file_name,
                "dataset": "team_gallery",
                "file_size": int(status.get("length", 0) or 0),
                "taken_at": infer_fallback_taken_at(path),
                "location": infer_fallback_location(path),
            }
        )
        count += 1
        if MAX_SCAN_FILES and count >= MAX_SCAN_FILES:
            break
        if count % 10000 == 0:
            print(f"Scanned {count} images")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=METADATA_INPUT_DIR)
    parser.add_argument("--raw-root", default=RAW_UI_IMAGE_ROOT)
    parser.add_argument("--dataset", default="team_gallery")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--scan-hdfs-if-missing", default=str(SCAN_HDFS_IF_MISSING).lower())
    args = parser.parse_args()

    scan_hdfs = str(args.scan_hdfs_if_missing).strip().lower() in {"1", "true", "yes", "y"}
    df = read_local_metadata(args.input_dir)
    if df.empty and scan_hdfs:
        df = scan_hdfs_images(args.raw_root)
    if df.empty:
        if args.allow_empty:
            print(
                json.dumps(
                    {
                        "skipped": True,
                        "reason": "no_metadata_found",
                        "input_dir": args.input_dir,
                        "raw_root": args.raw_root,
                        "scan_hdfs_if_missing": scan_hdfs,
                    },
                    indent=2,
                )
            )
            return
        raise SystemExit(f"No metadata found in {args.input_dir} and no HDFS images found under {args.raw_root}")

    normalized = normalize_dataframe(df, dataset=args.dataset, raw_root=args.raw_root)
    normalized = normalized[normalized["deleted"] != True].drop_duplicates(subset=["image_id"], keep="last")
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    imported_path = hdfs.write_dataframe_parquet(normalized, f"{IMPORTED_METADATA_ROOT.rstrip('/')}/{date_part}", filename="imported_metadata.parquet")
    active_path = hdfs.write_dataframe_parquet(normalized, f"{UI_ACTIVE_METADATA_ROOT.rstrip('/')}/{date_part}", filename="ui_active.parquet")
    print(json.dumps({"rows": len(normalized), "imported_path": imported_path, "ui_active_path": active_path}, indent=2))


if __name__ == "__main__":
    main()
