#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.common import hdfs

SAMPLE_DATA_DIR = Path(os.getenv("SAMPLE_DATA_DIR", "/app/data/sample_images"))
MIR_ROOT = os.getenv("MIRFLICKR25K_RAW_ROOT", "/photos/raw/mirflickr25k/images")
EXISTING_METADATA_DIR = Path(os.getenv("METADATA_INPUT_DIR", "/app/data/existing_metadata"))
RAW_UI_IMAGE_ROOT = os.getenv("RAW_UI_IMAGE_ROOT", "/photos/raw/team_gallery/images")


def main() -> None:
    if not SAMPLE_DATA_DIR.exists():
        raise SystemExit(f"Missing {SAMPLE_DATA_DIR}; run generate_sample_dataset.py first")
    rows = []
    for img in sorted(SAMPLE_DATA_DIR.glob("*.jpg")):
        shard = str(abs(hash(img.name)) % 10)
        # Put sample images into both MIRFLICKR demo root and team_gallery root so
        # make all works even without the full external dataset mounted.
        for root in [MIR_ROOT, RAW_UI_IMAGE_ROOT]:
            hdfs.upload_local_file(str(img), f"{root.rstrip('/')}/{shard}/{img.name}", overwrite=True)
        rows.append({"file_name": img.name, "image_uri": hdfs.to_hdfs_uri(f"{RAW_UI_IMAGE_ROOT.rstrip('/')}/{shard}/{img.name}")})
    EXISTING_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EXISTING_METADATA_DIR / "sample_existing_metadata.csv", index=False)
    print(f"Uploaded {len(rows)} sample images to HDFS and wrote local metadata")


if __name__ == "__main__":
    main()
