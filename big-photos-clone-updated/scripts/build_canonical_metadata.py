#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.common import hdfs

UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
UPLOAD_METADATA_ROOT = os.getenv("UPLOAD_METADATA_ROOT", "/photos/metadata/uploads")
ENRICHED_METADATA_ROOT = os.getenv("ENRICHED_METADATA_ROOT", "/photos/metadata/enriched")
CANONICAL_METADATA_ROOT = os.getenv("CANONICAL_METADATA_ROOT", "/photos/metadata/canonical_images")
CANONICAL_CURRENT_PATH = os.getenv("CANONICAL_CURRENT_PATH", "/photos/metadata/canonical_images/current_manifest.json")
LOCAL_CANONICAL_OUTPUT_DIR = os.getenv("LOCAL_CANONICAL_OUTPUT_DIR", "/app/outputs/metadata")


def _norm_col(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        df[col] = default
    return df


def _pick_latest(rows: List[Dict]) -> List[Dict]:
    by_id: Dict[str, Dict] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            continue
        prev = by_id.get(image_id)
        if prev is None:
            by_id[image_id] = row
            continue
        prev_ts = str(prev.get("updated_at") or prev.get("created_at") or "")
        row_ts = str(row.get("updated_at") or row.get("created_at") or "")
        if row_ts >= prev_ts:
            by_id[image_id] = row
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--include-enriched", action="store_true")
    args = parser.parse_args()

    frames = []
    roots = [UI_ACTIVE_METADATA_ROOT, UPLOAD_METADATA_ROOT]
    if args.include_enriched:
        roots.append(ENRICHED_METADATA_ROOT)
    for root in roots:
        try:
            df = hdfs.read_parquet_dataset(root)
            if not df.empty:
                df["_source_root"] = root
                frames.append(df)
        except Exception:
            continue
    if not frames:
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "no_input_metadata"}, indent=2))
            return
        raise SystemExit("No metadata found to build canonical table")

    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["image_id", "image_uri", "thumbnail_uri", "labels", "category", "caption", "dataset", "user_id", "location", "taken_at"]:
        _norm_col(df, c, None)
    _norm_col(df, "updated_at", datetime.now(timezone.utc))
    _norm_col(df, "created_at", datetime.now(timezone.utc))
    _norm_col(df, "deleted", False)
    _norm_col(df, "pred_label", None)
    _norm_col(df, "pred_score", None)
    _norm_col(df, "model_version", None)

    df = df[df["image_id"].notna()]
    df["image_id"] = df["image_id"].astype(str)
    df = df[df["image_id"] != ""]
    df = df[(df["deleted"] != True) | (df["deleted"].isna())]

    records = _pick_latest(df.to_dict("records"))
    out = pd.DataFrame(records)
    out["metadata_version"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out["quality_missing_image_uri"] = out["image_uri"].isna() | (out["image_uri"].astype(str).str.strip() == "")
    out["quality_missing_thumbnail_uri"] = out["thumbnail_uri"].isna() | (out["thumbnail_uri"].astype(str).str.strip() == "")
    out["quality_missing_labels"] = out["labels"].isna()

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = f"{CANONICAL_METADATA_ROOT.rstrip('/')}/run={run_tag}"
    manifest = {
        "run_tag": run_tag,
        "canonical_root": target_dir,
        "canonical_file": None,
        "rows": int(len(out)),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "storage": "hdfs",
    }
    try:
        target_path = hdfs.write_dataframe_parquet(out, target_dir, filename="canonical_images.parquet")
        manifest["canonical_file"] = target_path
        hdfs.write_json(CANONICAL_CURRENT_PATH, manifest, overwrite=True)
    except Exception as exc:
        local_dir = Path(LOCAL_CANONICAL_OUTPUT_DIR).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / f"canonical_images_{run_tag}.parquet"
        local_manifest = local_dir / "canonical_current_manifest.json"
        out.to_parquet(local_file, index=False)
        manifest["storage"] = "local_fallback"
        manifest["canonical_root"] = str(local_dir)
        manifest["canonical_file"] = str(local_file)
        manifest["warning"] = f"hdfs_write_failed: {exc}"
        local_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
