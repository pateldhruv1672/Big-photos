#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common import hdfs

CANONICAL_METADATA_ROOT = os.getenv("CANONICAL_METADATA_ROOT", "/photos/metadata/canonical_images")
CANONICAL_CURRENT_PATH = os.getenv("CANONICAL_CURRENT_PATH", "/photos/metadata/canonical_images/current_manifest.json")
METADATA_QUALITY_ROOT = os.getenv("METADATA_QUALITY_ROOT", "/photos/metadata/quality")
LOCAL_CANONICAL_OUTPUT_DIR = os.getenv("LOCAL_CANONICAL_OUTPUT_DIR", "/app/outputs/metadata")


def _parse_label_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return 0
        if v.startswith("[") and v.endswith("]"):
            return max(0, len([x for x in v.strip("[]").split(",") if x.strip()]))
        return 1
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", default="")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    root = str(args.canonical_root).strip()
    local_manifest_path = Path(LOCAL_CANONICAL_OUTPUT_DIR) / "canonical_current_manifest.json"
    if not root:
        manifest = hdfs.read_json(CANONICAL_CURRENT_PATH, default={}) or {}
        root = str(manifest.get("canonical_root") or "")
        canonical_file = str(manifest.get("canonical_file") or "")
    else:
        canonical_file = ""
    if (not root) and local_manifest_path.exists():
        local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
        root = str(local_manifest.get("canonical_root") or "")
        canonical_file = str(local_manifest.get("canonical_file") or "")
    if not root:
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "canonical_root_missing"}, indent=2))
            return
        raise SystemExit("No canonical metadata root provided/found")

    df = pd.DataFrame()
    if canonical_file and canonical_file.startswith("/"):
        p = Path(canonical_file)
        if p.exists():
            df = pd.read_parquet(p)
    if df.empty:
        df = hdfs.read_parquet_dataset(root)
    if df.empty:
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "canonical_table_empty", "root": root}, indent=2))
            return
        raise SystemExit(f"No rows found under canonical root: {root}")

    total_rows = int(len(df))
    missing_image_uri = int(((df["image_uri"].isna()) | (df["image_uri"].astype(str).str.strip() == "")).sum()) if "image_uri" in df.columns else total_rows
    missing_thumb_uri = int(((df["thumbnail_uri"].isna()) | (df["thumbnail_uri"].astype(str).str.strip() == "")).sum()) if "thumbnail_uri" in df.columns else total_rows
    duplicate_image_ids = int(total_rows - df["image_id"].astype(str).nunique()) if "image_id" in df.columns else total_rows

    label_counts = df["labels"].apply(_parse_label_count) if "labels" in df.columns else pd.Series([0] * total_rows)
    no_labels = int((label_counts == 0).sum())
    multi_labels = int((label_counts > 1).sum())
    other_count = 0
    if "category" in df.columns:
        other_count = int((df["category"].astype(str).str.lower() == "other").sum())

    by_dataset: List[Dict[str, Any]] = []
    if "dataset" in df.columns:
        tmp = df.groupby("dataset").size().reset_index(name="rows").sort_values("rows", ascending=False)
        by_dataset = tmp.to_dict("records")

    summary = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "canonical_root": root,
        "total_rows": total_rows,
        "missing_image_uri": missing_image_uri,
        "missing_thumbnail_uri": missing_thumb_uri,
        "duplicate_image_ids": duplicate_image_ids,
        "rows_with_no_labels": no_labels,
        "rows_with_multi_labels": multi_labels,
        "rows_with_other_category": other_count,
        "dataset_breakdown": by_dataset,
    }

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = f"{METADATA_QUALITY_ROOT.rstrip('/')}/run={run_tag}"
    try:
        hdfs.write_json(f"{out_root}/summary.json", summary, overwrite=True)
        hdfs.write_dataframe_parquet(pd.DataFrame([summary]), out_root, filename="summary.parquet")
    except Exception as exc:
        local_dir = Path(LOCAL_CANONICAL_OUTPUT_DIR).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / f"quality_summary_{run_tag}.json").write_text(json.dumps({**summary, "warning": f"hdfs_write_failed: {exc}"}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
