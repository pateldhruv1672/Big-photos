#!/usr/bin/env python3
"""Import MIRFLICKR-style annotation text files into HDFS Parquet metadata.

Annotation files are expected to contain one numeric image id per line.
File names become labels, for example:
  clouds_r1.txt -> label "clouds"
  people.txt    -> label "people"
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from src.common import hdfs
from src.common.image_utils import infer_fallback_location, infer_fallback_taken_at
from src.common.metadata_normalizer import normalize_dataframe

ANNOTATION_DIR = os.getenv("ANNOTATION_DIR", "/app/data/annotations")
ANNOTATION_ALT_DIR = os.getenv("ANNOTATION_ALT_DIR", "/app/data/mirflickr25k_annotations_v080")
RAW_UI_IMAGE_ROOT = os.getenv("RAW_UI_IMAGE_ROOT", "/photos/raw/team_gallery/images")
MIRFLICKR25K_RAW_ROOT = os.getenv("MIRFLICKR25K_RAW_ROOT", "/photos/raw/mirflickr25k/images")
IMPORTED_METADATA_ROOT = os.getenv("IMPORTED_METADATA_ROOT", "/photos/metadata/imported")
UI_ACTIVE_METADATA_ROOT = os.getenv("UI_ACTIVE_METADATA_ROOT", "/photos/metadata/ui_active")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "team_gallery")
DEFAULT_DATASET = os.getenv("ANNOTATION_DATASET", "mirflickr25k")
MAX_ROWS = int(os.getenv("ANNOTATION_MAX_ROWS", "0"))
SKIP_HDFS_EXISTS_CHECK = os.getenv("ANNOTATION_SKIP_HDFS_EXISTS_CHECK", "true").lower() in {"1", "true", "yes"}


def _sanitize_label(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"_r\d+$", "", stem)
    stem = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    return stem or "unknown"


def _iter_ids(path: Path) -> Iterable[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not re.fullmatch(r"\d+", s):
            continue
        yield str(int(s))


def _discover_annotation_files(annotation_dir: Path) -> List[Path]:
    files = sorted(p for p in annotation_dir.glob("*.txt") if p.is_file())
    if not files:
        return files
    r1 = [p for p in files if re.search(r"_r1\.txt$", p.name, flags=re.IGNORECASE)]
    return r1 or files


def _discover_all_annotation_files(annotation_dir: Path) -> List[Path]:
    return sorted(p for p in annotation_dir.glob("*.txt") if p.is_file())


def _candidate_paths(image_id: str, roots: List[str]) -> List[str]:
    numeric = int(image_id)
    shard = str(numeric % 10)
    shard_plus_one = str((numeric % 10) + 1)
    # Existing team_gallery layout uses numeric filenames.
    names = [f"{numeric}.jpg", f"im{numeric}.jpg", f"{numeric}.jpeg", f"im{numeric}.jpeg", f"{numeric}.png"]
    out = []
    for root in roots:
        base = root.rstrip("/")
        for name in names:
            out.append(f"{base}/{shard}/{name}")
            out.append(f"{base}/{shard_plus_one}/{name}")
            out.append(f"{base}/{name}")
    return out


def _resolve_hdfs_image_path(image_id: str, roots: List[str]) -> Optional[Tuple[str, str, int]]:
    for candidate in _candidate_paths(image_id, roots):
        try:
            status = hdfs.file_status(candidate)
            if status and status.get("type") == "FILE":
                return hdfs.to_hdfs_uri(candidate), Path(candidate).name, int(status.get("length", 0) or 0)
        except Exception:
            continue
    return None


def _build_image_index(roots: List[str]) -> Dict[str, Tuple[str, str, int]]:
    out: Dict[str, Tuple[str, str, int]] = {}
    for root in roots:
        base = root.rstrip("/")
        try:
            for path in hdfs.walk(base):
                name = Path(path).name.lower()
                m = re.fullmatch(r"(?:im)?(\d+)\.(?:jpg|jpeg|png)", name)
                if not m:
                    continue
                image_id = str(int(m.group(1)))
                if image_id in out:
                    continue
                status = hdfs.file_status(path) or {}
                out[image_id] = (hdfs.to_hdfs_uri(path), Path(path).name, int(status.get("length", 0) or 0))
        except Exception:
            continue
    return out


def build_rows(
    annotation_dir: Path,
    roots: List[str],
    dataset: str,
    include_all_labels: bool = False,
    allowed_labels: Optional[Set[str]] = None,
    add_other_class: bool = False,
    other_label: str = "other",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    files = _discover_all_annotation_files(annotation_dir) if include_all_labels else _discover_annotation_files(annotation_dir)
    if not files:
        raise SystemExit(f"No annotation txt files found in {annotation_dir}")

    labels_by_id: Dict[str, Set[str]] = defaultdict(set)
    for file_path in files:
        label = _sanitize_label(file_path.name)
        if allowed_labels is not None and label not in allowed_labels:
            continue
        for image_id in _iter_ids(file_path):
            labels_by_id[image_id].add(label)

    image_index: Dict[str, Tuple[str, str, int]] = {}
    if SKIP_HDFS_EXISTS_CHECK:
        image_index = _build_image_index(roots)

    rows = []
    unresolved = 0
    other_rows = 0
    for idx, (image_id, labels) in enumerate(labels_by_id.items(), start=1):
        if MAX_ROWS and idx > MAX_ROWS:
            break
        selected_labels = sorted(labels)
        if allowed_labels is not None:
            selected_labels = sorted([x for x in labels if x in allowed_labels])
            if not selected_labels:
                if add_other_class:
                    selected_labels = [other_label]
                    other_rows += 1
                else:
                    continue

        if image_index:
            resolved = image_index.get(image_id)
        else:
            resolved = _resolve_hdfs_image_path(image_id, roots)
        if not resolved:
            unresolved += 1
            continue
        image_uri, file_name, file_size = resolved
        raw_hdfs_path = image_uri.split("hdfs://", 1)[-1]
        raw_hdfs_path = "/" + raw_hdfs_path.split("/", 1)[-1] if "/" in raw_hdfs_path else f"/{file_name}"
        rows.append(
            {
                "image_id": str(image_id),
                "user_id": DEFAULT_USER_ID,
                "image_uri": image_uri,
                "file_name": file_name,
                "dataset": dataset,
                "labels": selected_labels,
                "tags": selected_labels,
                "file_size": file_size,
                "taken_at": infer_fallback_taken_at(raw_hdfs_path),
                "location": infer_fallback_location(raw_hdfs_path),
            }
        )
        if idx % 5000 == 0:
            print(f"Resolved {idx} annotation ids...")

    df = pd.DataFrame(rows)
    stats = {
        "annotation_files": len(files),
        "distinct_annotated_ids": len(labels_by_id),
        "resolved_rows": len(df),
        "unresolved_ids": unresolved,
        "other_rows": other_rows,
    }
    return df, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", default=ANNOTATION_DIR)
    parser.add_argument("--fallback-annotation-dir", default=ANNOTATION_ALT_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--raw-roots", default=f"{RAW_UI_IMAGE_ROOT},{MIRFLICKR25K_RAW_ROOT}")
    parser.add_argument("--imported-root", default=IMPORTED_METADATA_ROOT)
    parser.add_argument("--active-root", default=UI_ACTIVE_METADATA_ROOT)
    parser.add_argument("--include-all-label-files", action="store_true")
    parser.add_argument("--allowed-labels", default="")
    parser.add_argument("--add-other-class", action="store_true")
    parser.add_argument("--other-label", default="other")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    if not ann_dir.exists() or not list(ann_dir.glob("*.txt")):
        alt = Path(args.fallback_annotation_dir)
        if alt.exists() and list(alt.glob("*.txt")):
            ann_dir = alt
    if not ann_dir.exists():
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "annotation_dir_not_found", "path": str(args.annotation_dir)}, indent=2))
            return
        raise SystemExit(f"Annotation directory does not exist: {args.annotation_dir}")

    roots = [x.strip() for x in str(args.raw_roots).split(",") if x.strip()]
    allowed = None
    if str(args.allowed_labels).strip():
        allowed = {x.strip().lower() for x in str(args.allowed_labels).split(",") if x.strip()}
    df, stats = build_rows(
        ann_dir,
        roots,
        args.dataset,
        include_all_labels=bool(args.include_all_label_files),
        allowed_labels=allowed,
        add_other_class=bool(args.add_other_class),
        other_label=str(args.other_label).strip() or "other",
    )
    if df.empty:
        if args.allow_empty:
            print(json.dumps({"skipped": True, "reason": "no_rows_resolved", **stats}, indent=2))
            return
        raise SystemExit(f"No rows resolved from annotations under {ann_dir}")

    normalized = normalize_dataframe(df, dataset=args.dataset, raw_root=roots[0] if roots else RAW_UI_IMAGE_ROOT)
    normalized = normalized[normalized["deleted"] != True].drop_duplicates(subset=["image_id"], keep="last")
    date_part = datetime.now(timezone.utc).strftime("date=%Y-%m-%d")
    imported_path = hdfs.write_dataframe_parquet(
        normalized,
        f"{args.imported_root.rstrip('/')}/{date_part}",
        filename="annotations_imported.parquet",
    )
    active_path = hdfs.write_dataframe_parquet(
        normalized,
        f"{args.active_root.rstrip('/')}/{date_part}",
        filename="ui_active_annotations.parquet",
    )
    print(
        json.dumps(
            {
                "annotation_dir": str(ann_dir),
                **stats,
                "rows_written": len(normalized),
                "imported_path": imported_path,
                "ui_active_path": active_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
