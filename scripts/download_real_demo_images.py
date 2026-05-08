#!/usr/bin/env python3
"""
Replace synthetic JPEG tiles with deterministic real-photo thumbnails from Picsum.

Reads image IDs from data/mirflickr/tags.json (keys only). Does not modify tags.json.
Downloads https://picsum.photos/seed/{image_id}/256/256 → data/mirflickr/images/{image_id}.jpg

Run from project root (or /app in Docker): same paths as generate_sample_dataset.py.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TAGS_JSON = Path("data/mirflickr/tags.json")
IMG_DIR = Path("data/mirflickr/images")
USER_AGENT = "DistributedPhotoIntel-demo/1.0 (picsum thumbnails; +https://picsum.photos)"


def _download_one(url: str, dest: Path, timeout: int = 90) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise RuntimeError("empty response")
    dest.write_bytes(data)


def main() -> int:
    if not TAGS_JSON.is_file():
        print(f"ERROR: {TAGS_JSON} not found. Run `make part1` first.", file=sys.stderr)
        return 1

    payload = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        print("ERROR: tags.json has no entries.", file=sys.stderr)
        return 1

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    keys = sorted(payload.keys())
    ok = 0
    failed: list[tuple[str, str]] = []

    for i, image_id in enumerate(keys):
        # Keys come from our generator (e.g. im481741); still avoid odd paths.
        if not isinstance(image_id, str) or "/" in image_id or image_id.startswith("."):
            failed.append((str(image_id), "invalid id"))
            continue

        safe_seed = urllib.parse.quote(image_id, safe="")
        url = f"https://picsum.photos/seed/{safe_seed}/256/256"
        out_path = IMG_DIR / f"{image_id}.jpg"
        try:
            _download_one(url, out_path)
            ok += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(keys):
                print(f"  {i + 1}/{len(keys)} downloaded …")
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
            failed.append((image_id, str(e)))
            print(f"WARNING: {image_id}: {e}", file=sys.stderr)

        # Light pacing for remote API politeness
        time.sleep(0.05)

    print(f"Done. Replaced {ok} image(s) under {IMG_DIR.resolve()}")
    if failed:
        print(f"WARNING: {len(failed)} failure(s). First few:", file=sys.stderr)
        for fid, msg in failed[:5]:
            print(f"  {fid}: {msg}", file=sys.stderr)
        return 1 if ok == 0 else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
